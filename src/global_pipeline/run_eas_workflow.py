"""EAS real-data run, sharing the Taiwan NN/nested-CV/support/RL executor.

Wait only for this explicitly supplied local preprocessing job, fail closed
on missing data, verify the saved diagnostic geometry, then run ONE freshly
timed NN -> FFS/GP nested CV -> OOF support/RL workflow. No NN retraining,
800-point cap, temporal selection, hidden repairs or simulated-truth claims.
"""
from __future__ import annotations
import sys
from pathlib import Path
SOURCE_ROOT=Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path: sys.path.insert(0,str(SOURCE_ROOT))

import argparse
import json
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from global_pipeline import diagnose_eas_buffer_folds as eas
from taiwan_simulation import run_timed_workflow as workflow
from taiwan_simulation.diagnose_buffer_folds import geometry_diagnostics
from taiwan_simulation.cheng_spatial_cv import join_predictors
from research_workflows import COEFFICIENTS, YEARS, TEMPORAL_POLICY
from train_cheng_nn import atomic_json, atomic_csv, file_sha256


def wait_predictors(folder, status_path, *, wait=False, timeout=86400):
    started=time.perf_counter()
    while True:
        report_path=folder/'run_report.json'
        if report_path.is_file():
            report=json.loads(report_path.read_text(encoding='utf-8'))
            if report.get('status')!='completed' or report.get('complete_grids')!=7822:
                raise ValueError('EAS preprocessing incomplete/has unresolved data issues; no GRID dropped.')
            return report
        status_file=folder/'run_status.json'
        status=json.loads(status_file.read_text(encoding='utf-8')) if status_file.is_file() else {}
        if status.get('status')=='failed':
            raise RuntimeError(f"Preprocessing failed: {status.get('error')}")
        if not wait or time.perf_counter()-started>timeout:
            raise TimeoutError('EAS predictors not ready; NN/full-workflow timer has not started.')
        atomic_json(status_path,dict(status='waiting_for_predictors',predictor_status=status,
            elapsed_wait_seconds=time.perf_counter()-started,nn_timer_started=False))
        time.sleep(30)


def make_config(groups, n_grids):
    return dict(temporal_policy=TEMPORAL_POLICY,candidate_groups=groups,
        buffers_km={name:500. for name in COEFFICIENTS},outer_folds=5,inner_folds=3,
        min_train=100,max_train=n_grids,seed=20260929,n_jobs=1,n_restarts=0,
        min_relative_improvement=.01,maximum_allowed_vif=5.,max_steps=None,
        kernel_candidates=[['RBF',None],['Matern',.5],['Matern',1.5],['Matern',2.5]],
        buffer_distance_mode='wgs84',
        gp_options=dict(length_scale_bounds=[1.,6000.],initial_length_scale=300.))


def load_inputs(diagnostics, predictors_path, config_path):
    folder=predictors_path.parent
    report=json.loads((folder/'run_report.json').read_text(encoding='utf-8'))
    protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    diagnostic=json.loads((diagnostics/'run_report.json').read_text(encoding='utf-8'))
    if (report.get('status')!='completed' or report.get('n_grids')!=7822
            or report.get('complete_grids')!=7822 or report.get('annual_missing_rows')!=0
            or diagnostic.get('region')!='EAS' or diagnostic.get('years')!=list(YEARS)
            or protocol.get('years')!=list(YEARS)
            or protocol.get('registry_sha256')!=file_sha256(diagnostics/'grid_index.csv')):
        raise ValueError('Incomplete or incompatible EAS predictor/diagnostic contract.')
    hashes={}
    for name in ('grid_index.csv','grid_predictors.csv','annual_weather.csv','static_predictors.csv'):
        path=folder/name
        digest=file_sha256(path)
        if digest!=report['output_sha256'].get(name):
            raise ValueError(f'Changed EAS predictor output: {name}')
        hashes[str(path)]=digest
    # Re-verify original annual cache, AR6 registry and actual source identity;
    # diagnostic NN estimates are NOT used as this run's NN outputs.
    grids,values,_,source_hashes=eas.prepare(eas.NN_ROOT,eas.DATA_ROOT)
    for name,expected in (('annual_maxima.npy',values),('years.npy',np.asarray(YEARS))):
        if not np.array_equal(np.load(diagnostics/name,allow_pickle=False),expected):
            raise ValueError('Saved diagnostic data changed or no longer match verified source.')
    saved=pd.read_csv(diagnostics/'grid_index.csv',dtype={'grid_id':str})
    paired=pd.read_csv(folder/'grid_index.csv',dtype={'grid_id':str})
    for frame in (saved,paired):
        if (frame.grid_id.tolist()!=grids.grid_id.tolist()
                or not np.allclose(frame[['longitude_180','latitude','x_km','y_km','lsm']],
                                   grids[['longitude_180','latitude','x_km','y_km','lsm']],rtol=0,atol=1e-6)):
            raise ValueError('EAS GRID identities/coordinates changed.')
    predictors=pd.read_csv(predictors_path,dtype={'grid_id':str})
    groups=json.loads((folder/'candidate_groups.json').read_text(encoding='utf-8'))
    config=json.loads(config_path.read_text(encoding='utf-8'))
    if config!=make_config(groups,len(grids)):
        raise ValueError('Only the explicitly diagnosed EAS full-training configuration is allowed.')
    if (set(predictors.grid_id)!=set(grids.grid_id) or predictors.grid_id.duplicated().any()
            or not predictors.candidate_data_complete.eq(True).all()):
        raise ValueError('Incomplete/duplicate EAS predictors; no silent removal.')
    dummy=pd.concat([grids,pd.DataFrame(np.zeros((len(grids),5)),columns=COEFFICIENTS)],axis=1)
    join_predictors(dummy,predictors,groups,coordinate_crs=eas.EAS_CRS)
    config.pop('temporal_policy')
    hashes.update(source_hashes)
    for path in (config_path,diagnostics/'run_report.json',folder/'run_report.json',
                 folder/'candidate_groups.json',folder/'protocol.json'):
        hashes[str(path)]=file_sha256(path)
    return values,grids,predictors,config,hashes


def verify_geometry(grids,diagnostics,output):
    distance=eas.geodesic_matrix(grids)
    result=geometry_diagnostics(grids,outer_counts=[5],inner_counts=[3],buffers=[500.],
        min_train=100,seed=20260929,distance_matrix=distance)
    if not result['summary'].all_folds_meet_min_train.all():
        raise ValueError('EAS split no longer meets minimum training size.')
    old=pd.read_csv(diagnostics/'splits.csv').query('outer_folds == 5 and inner_folds == 3 and buffer_km == 500')
    keys=['level','outer_fold','inner_fold']
    old=old.set_index(keys).sort_index(); new=result['splits'].set_index(keys).sort_index()
    if not old[['train_ids','test_ids']].equals(new[['train_ids','test_ids']]):
        raise ValueError('Actual membership differs from diagnosed membership; inspect before fitting.')
    for name,table in result.items(): atomic_csv(output/f'actual_{name}.csv',table)
    return result['summary'].to_dict(orient='records')[0]


def launch(diagnostics,predictors_dir,output,*,wait=False):
    diagnostics,predictors_dir,output=map(lambda p:Path(p).resolve(),(diagnostics,predictors_dir,output))
    if output.exists(): raise FileExistsError('Use a new launch/output folder.')
    output.mkdir(parents=True)
    status_path=output/'launch_status.json'
    started=time.perf_counter()
    def status(stage,**extra):
        atomic_json(status_path,dict(status=stage,elapsed_launch_seconds=time.perf_counter()-started,**extra))
        print(stage,extra,flush=True)
    try:
        comparison=pd.read_csv(diagnostics/'summary.csv')
        choose=((comparison.outer_folds.eq(5)&comparison.inner_folds.eq(3)&comparison.buffer_km.isin([300,500,750,1000]))
            |(comparison.outer_folds.eq(5)&comparison.inner_folds.eq(5)&comparison.buffer_km.eq(500))
            |(comparison.outer_folds.eq(6)&comparison.inner_folds.eq(3)&comparison.buffer_km.eq(500)))
        atomic_csv(output/'diagnostic_comparison.csv',comparison.loc[choose])
        atomic_json(output/'research_settings.json',dict(region='EAS',outer_folds=5,inner_folds=3,
            buffer_km=500,min_train=100,seed=20260929,all_retained_training_grids=True,
            interpretation='Development setting, not proof of spatial independence; 300/750 km future sensitivity.',
            user_confirmed_all_retained_grids=True,formal_nn_replaced=False))
        status('waiting_for_predictors',nn_timer_started=False)
        prep=wait_predictors(predictors_dir,status_path,wait=wait)
        status('verify_inputs_and_geometry',nn_timer_started=False)
        groups=json.loads((predictors_dir/'candidate_groups.json').read_text(encoding='utf-8'))
        config_path=output/'config.json'
        atomic_json(config_path,make_config(groups,7822))
        values,grids,_,_,_=load_inputs(diagnostics,predictors_dir/'grid_predictors.csv',config_path)
        del values
        with threadpool_limits(limits=4):
            geometry=verify_geometry(grids,diagnostics,output)
            sources=[Path(__file__),Path(workflow.__file__),SOURCE_ROOT/'taiwan_simulation/cheng_spatial_cv.py',
                     SOURCE_ROOT/'spatial_predictor_selection.py',SOURCE_ROOT/'elevation_gp_analysis.py',
                     SOURCE_ROOT/'land_cover_gp_analysis.py']
            hashes={str(p):file_sha256(p) for p in sources}
            atomic_json(output/'launch_manifest.json',dict(region='EAS',max_train_cap_applied=False,
                source_sha256=hashes,diagnosed_setting=geometry,
                preprocessing_elapsed_seconds=prep['elapsed_seconds'],
                validation_role='development spatial OOF against frozen NN, not known truth',
                covariance_distances='Euclidean LCC km; WGS84 distances are for buffers only',
                workflow_scope='fresh bound NN -> grouped FFS/kernel selection -> nested spatial OOF -> support/RL; no temporal selection, bootstrap or full-data final GP'))
            status('workflow_started',nn_timer_started=True,progress_file=str(output/'run/run_status.json'))
            report=workflow.run(diagnostics,predictors_dir/'grid_predictors.csv',config_path,output/'run',
                model_kind='bound',device='cpu',threads=4,input_loader=load_inputs,
                coordinate_crs=eas.EAS_CRS,region='EAS')
            changed=[str(p) for p in sources if file_sha256(p)!=hashes[str(p)]]
            if changed: raise RuntimeError('Workflow sources changed during run: '+', '.join(changed))
        status(report['status'],workflow_completed=True,
            nn_to_outputs_seconds=report['timings']['total_from_nn_start_until_outputs_seconds'])
        return report
    except BaseException as error:
        status('failed',workflow_completed=False,error_type=type(error).__name__,error=str(error))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics',type=Path,required=True)
    parser.add_argument('--predictors-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    launch(args.diagnostics,args.predictors_dir,args.output,wait=args.wait)
