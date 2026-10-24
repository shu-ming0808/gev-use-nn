"""One timed real-data Taiwan run: frozen NN -> nested CV -> support and RL.

Requires complete aligned predictors and explicit research settings. No NN
training, temporal selection, bootstrap, final full-data GP fit or global run.
The clock covers a fresh NN call through final output writing in ONE process;
old pilot coefficients/timings are never reused as the current run's results.
"""
from __future__ import annotations

import sys
from pathlib import Path
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from taiwan_simulation import prepare_taiwan_cheng as tw
from taiwan_simulation import cheng_spatial_cv as cv
from research_workflows import (CONTRACT, COEFFICIENTS, YEARS, REFERENCE_YEAR,
                                TIME_SCALE_YEARS, TEMPORAL_POLICY, validate_annual_sequences)
from cheng_nn_evaluation import check_gev_predictions, conditional_return_levels
from train_cheng_nn import atomic_csv, atomic_json, file_sha256


def load_inputs(prepared, predictors_path, config_path):
    """Fail closed on changed source/cache, dates, incomplete grids or settings."""
    report_path = prepared/'run_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    expected = dict(status='completed', completed_stage='taiwan_preparation_and_nn_only',
        contract=CONTRACT, data_role='real_data_pilot_not_simulated_truth',
        coordinate_crs='EPSG:3826', coordinate_units='km')
    if any(report.get(k) != v for k,v in expected.items()) or report.get('years') != list(YEARS):
        raise ValueError('Expected a completed, compatible 1976-2025 Taiwan pilot.')
    # Older pilots did not record this field. Reuse only their raw sequences,
    # never old coefficient estimates; the NEW config must declare the policy.
    if 'temporal_policy' in report and report['temporal_policy'] != TEMPORAL_POLICY:
        raise ValueError('Pilot explicitly records an incompatible temporal policy.')
    hashes = {str(report_path): file_sha256(report_path)}
    for name in ('annual_maxima.npy', 'years.npy', 'grid_index.csv'):
        path = prepared/name
        digest = file_sha256(path)
        if digest != report.get('output_sha256', {}).get(name):
            raise ValueError(f'Prepared output hash mismatch: {name}')
        hashes[str(path)] = digest
    raw = Path(report['input_path']).resolve()
    digest = file_sha256(raw)
    if digest != report.get('source_sha256', {}).get(str(raw)):
        raise ValueError('Original annual source hash mismatch.')
    hashes[str(raw)] = digest
    years = np.load(prepared/'years.npy', allow_pickle=False)
    values = validate_annual_sequences(years, np.load(prepared/'annual_maxima.npy', allow_pickle=False))
    grids = pd.read_csv(prepared/'grid_index.csv', dtype={'grid_id':str})
    if len(grids) != len(values) or report.get('n_grids') != len(grids):
        raise ValueError('GRID/sequence/report counts differ.')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.pop('temporal_policy', None) != TEMPORAL_POLICY:
        raise ValueError('Require explicit full_no_selection; no temporal selection.')
    buffers = config.get('buffers_km', {})
    if (set(buffers) != set(COEFFICIENTS)
            or any(v is None or not np.isfinite(v) or v < 0 for v in buffers.values())):
        raise ValueError('Supply calibrated finite buffers for all five coefficients; null is not allowed.')
    for key in ('candidate_groups', 'outer_folds', 'inner_folds', 'min_train', 'max_train'):
        if key not in config:
            raise ValueError(f'Missing explicit spatial setting: {key}')
    predictors = pd.read_csv(predictors_path, dtype={'grid_id':str})
    if 'candidate_data_complete' in predictors:
        flags = predictors.loc[predictors.grid_id.isin(grids.grid_id), 'candidate_data_complete']
        if not flags.eq(True).all():
            raise ValueError('Predictor candidate_data_complete flag is false or missing.')
    # Validate exact grid pairing/coordinates/all selected candidates before NN.
    dummy = pd.concat([grids, pd.DataFrame(np.zeros((len(grids),5)),columns=COEFFICIENTS)],axis=1)
    cv.join_predictors(dummy, predictors, config['candidate_groups'])
    for path in (predictors_path, config_path):
        hashes[str(path)] = file_sha256(path)
    return values, grids, predictors, config, hashes


def ordered_oof(oof, grids, nn):
    """Require exactly one same-fold prediction per GRID and coefficient."""
    required = {'grid_id','coefficient','outer_fold','reference_nn','prediction'}
    if (not required.issubset(oof) or len(oof) != len(grids)*5
            or oof.duplicated(['grid_id','coefficient']).any()
            or set(oof.grid_id) != set(grids.grid_id)
            or set(oof.coefficient) != set(COEFFICIENTS)
            or not oof.groupby('grid_id').outer_fold.nunique().eq(1).all()
            or oof.outer_fold.isna().any()):
        raise ValueError('Incomplete, duplicated or inconsistent OOF coverage.')
    predictions = oof.pivot(index='grid_id',columns='coefficient',values='prediction').reindex(
        index=grids.grid_id,columns=COEFFICIENTS).to_numpy(float)
    reference = oof.pivot(index='grid_id',columns='coefficient',values='reference_nn').reindex(
        index=grids.grid_id,columns=COEFFICIENTS).to_numpy(float)
    if not np.isfinite(predictions).all() or not np.allclose(reference,nn,rtol=1e-12,atol=1e-12):
        raise ValueError('OOF predictions/reference do not match this NN run.')
    return predictions


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(torch.device(device))


def run(prepared, predictors_path, config_path, output, *, model_kind, device='auto',
        batch_size=2048, threads=4, input_loader=None, coordinate_crs='EPSG:3826',
        region='Taiwan'):
    prepared,predictors_path,config_path,output = [Path(p).resolve() for p in
                                                (prepared,predictors_path,config_path,output)]
    if output.exists():
        raise FileExistsError('Use a new output directory; previous runs are read-only.')
    if model_kind not in tw.MODELS or threads < 1 or batch_size < 1:
        raise ValueError('Explicit original/bound model and positive threads/batch size required.')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    if device not in ('cpu','cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested inference device is unavailable.')
    torch.set_num_threads(threads)
    loader = load_inputs if input_loader is None else input_loader
    values,grids,predictors,config,input_hashes = loader(prepared,predictors_path,config_path)
    output.mkdir(parents=True)
    status_path=output/'run_status.json'
    timings={}; started=None
    def status(stage, **extra):
        atomic_json(status_path,dict(status='running',stage=stage,region=region,workflow_completed=False,
            elapsed_from_nn_start_seconds=time.perf_counter()-started if started is not None else None, **extra))
    def progress(stage, detail, tables):
        if tables is not None:
            folder=output/'checkpoints'/f"outer_{detail['outer_fold']}_{detail['coefficient']}"
            folder.mkdir(parents=True,exist_ok=True)
            for name,table in tables.items(): atomic_csv(folder/f'{name}.csv',table)
        status(stage, **detail)
        print(stage,detail,flush=True)
    try:
        status('nn')
        synchronize(device)
        started=time.perf_counter()
        result=tw.infer_sequences(values,model_kind=model_kind,device=device,batch_size=batch_size)
        synchronize(device)
        timings['nn_including_load_transform_decoder_diagnostics_seconds']=time.perf_counter()-started
        nn=np.asarray(result['prediction'],dtype=float)
        if nn.shape != (len(grids),5) or not np.isfinite(nn).all():
            raise ValueError('NN must return five finite coefficients per GRID.')
        responses=pd.concat([grids.reset_index(drop=True),pd.DataFrame(nn,columns=COEFFICIENTS)],axis=1)
        atomic_csv(output/'nn_coefficients.csv',responses)
        status('join_and_nested_cv')
        checkpoint=time.perf_counter()
        joined=cv.join_predictors(responses,predictors,config['candidate_groups'],
            **({'coordinate_crs':coordinate_crs} if coordinate_crs != 'EPSG:3826' else {}))
        cv_results=cv.nested_cv(joined,**config,progress=progress)
        timings['join_preflight_and_nested_cv_seconds']=time.perf_counter()-checkpoint
        status('support_and_return_levels')
        checkpoint=time.perf_counter()
        gp=ordered_oof(cv_results['oof'],grids,nn)
        checks={}
        for name,coeff in (('nn',nn),('oof',gp)):
            checks[name]=check_gev_predictions(coeff,values,np.asarray(YEARS),
                reference_year=REFERENCE_YEAR,time_scale_years=TIME_SCALE_YEARS)
            checks[name].insert(0,'grid_id',grids.grid_id.to_numpy())
        # Test observations diagnose support ONLY; never alter/project GP outputs.
        periods=(50.,100.)
        nn_rl=conditional_return_levels(nn,np.asarray(YEARS),periods,reference_year=REFERENCE_YEAR,
                                       time_scale_years=TIME_SCALE_YEARS)
        gp_rl=conditional_return_levels(gp,np.asarray(YEARS),periods,reference_year=REFERENCE_YEAR,
                                       time_scale_years=TIME_SCALE_YEARS)
        levels=pd.MultiIndex.from_product([grids.grid_id,YEARS,periods],
            names=['grid_id','year','return_period']).to_frame(index=False)
        levels['nn_reference']=nn_rl.ravel(); levels['gp_oof']=gp_rl.ravel()
        levels['finite_pair']=np.isfinite(levels.nn_reference)&np.isfinite(levels.gp_oof)
        metrics=[]
        for period,part in levels.groupby('return_period'):
            error=(part.gp_oof-part.nn_reference)[part.finite_pair].to_numpy()
            metrics.append(dict(return_period=period,n_total=len(part),n_used=len(error),
                RMSE_vs_NN_reference=float(np.sqrt(np.mean(error**2))) if len(error) else np.nan))
        timings['support_and_return_levels_seconds']=time.perf_counter()-checkpoint
        status('saving_outputs')
        checkpoint=time.perf_counter()
        atomic_csv(output/'nn_coefficients.csv',responses)
        atomic_csv(output/'oof_coefficients.csv',pd.concat([grids,pd.DataFrame(gp,columns=COEFFICIENTS)],axis=1))
        for name,table in cv_results.items(): atomic_csv(output/f'spatial_{name}.csv',table)
        for name,table in checks.items(): atomic_csv(output/f'{name}_support.csv',table)
        atomic_csv(output/'return_levels.csv',levels)
        atomic_csv(output/'return_level_metrics.csv',pd.DataFrame(metrics))
        timings['save_outputs_seconds']=time.perf_counter()-checkpoint
        atomic_csv(output/'timing.csv',pd.DataFrame([dict(stage=k,seconds=v) for k,v in timings.items()]))
        output_hashes={p.name:file_sha256(p) for p in output.glob('*.csv')}
        synchronize(device)
        timings['total_from_nn_start_until_outputs_seconds']=time.perf_counter()-started
        invalid={name:int((~table.gev_valid_all_years).sum()) for name,table in checks.items()}
        report=dict(status='completed_with_support_or_rl_warnings' if any(invalid.values()) or not levels.finite_pair.all() else 'completed',
            workflow_completed=True,full_research_pipeline_completed=False,
            completed_stage=('taiwan' if region == 'Taiwan' else region.lower())+'_nn_nested_cv_support_and_conditional_rl',
            region=region, coordinate_crs=coordinate_crs,
            contract=CONTRACT,temporal_policy=TEMPORAL_POLICY,data_role='real_data_pilot_not_simulated_truth',
            completed_utc=tw.shared.utc_now(),n_grids=len(grids),years=list(YEARS),return_periods=list(periods),
            model=result['model'],device=device,threads=threads,config=config,timings=timings,
            nn_component_timings=result.get('timings',{}),invalid_support_grids=invalid,
            timing_scope='One process, fresh NN including model load/features/decoder through CV, RL, output writing and output hashes; input validation and final report serialization excluded.',
            reference='Frozen NN estimates, NOT known climate truth; pooled year errors are not independent samples.',
            support_policy='Observed sequences diagnose OOF support only; never repair GP predictions using held-out observations.',
            not_run=['full_data_final_gp_fit','spatial_residual_diagnostics','known_truth_spatial_simulation','bootstrap',
                     'global_workflow' if region == 'Taiwan' else 'other_global_regions'],
            input_sha256=input_hashes,output_sha256=output_hashes)
        atomic_json(output/'run_report.json',report)
        atomic_json(status_path,dict(status=report['status'],workflow_completed=True,full_research_pipeline_completed=False))
        return report
    except BaseException as error:
        atomic_json(status_path,dict(status='failed',workflow_completed=False,error_type=type(error).__name__,
            error=str(error),elapsed_from_nn_start_seconds=time.perf_counter()-started if started is not None else None))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','predictors','config','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--model',choices=list(tw.MODELS),required=True)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--batch-size',type=int,default=2048)
    a=parser.parse_args(argv)
    report=run(a.prepared,a.predictors,a.config,a.output,model_kind=a.model,device=a.device,
               threads=a.threads,batch_size=a.batch_size)
    print(json.dumps(dict(status=report['status'],seconds=report['timings']['total_from_nn_start_until_outputs_seconds'],
                         not_run=report['not_run']),indent=2))


if __name__=='__main__':
    main()
