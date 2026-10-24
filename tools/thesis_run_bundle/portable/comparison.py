"""Paired NEW simulations: NN -> nested GP versus MLE -> the same nested GP.

No forced winner, no reused real-data error as truth, no post-hoc GRID filtering.
Each arm selects its own predictors/kernel inside identical buffered folds.
"""
from __future__ import annotations

import hashlib
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import scipy
import sklearn
import torch

from . import inference, workflow, simulation
from .common import SEED, identity, read, write, sha, latest_completed, attempt_folder
from research_workflows import COEFFICIENTS, YEARS
from train_cheng_nn import atomic_csv


def array_hash(array):
    a=np.ascontiguousarray(array,dtype=np.float64)
    return identity(dict(shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest()))


def environment(threads, gp_jobs, mle_jobs):
    return dict(host=platform.node(),platform=platform.platform(),processor=platform.processor(),
        logical_cpus=psutil.cpu_count(),ram_total_bytes=int(psutil.virtual_memory().total),
        python=sys.version,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
        torch=torch.__version__,threads=threads,gp_jobs=gp_jobs,mle_jobs=mle_jobs)


def code_identity():
    from taiwan_simulation import mle_estimator, cheng_spatial_cv, gev_crps
    import cheng_nn_mle
    import spatial_predictor_selection
    import land_cover_gp_analysis
    modules=(sys.modules[__name__],workflow,simulation,inference,mle_estimator,
             cheng_nn_mle,cheng_spatial_cv,spatial_predictor_selection,land_cover_gp_analysis,gev_crps)
    return {module.__name__:sha(Path(module.__file__)) for module in modules}


def validate_real_source(folder, inputs):
    """Allow verified old EAS results without requiring the NEW runner code hash.

    This is only a calibration source, never reused as paired simulation timing.
    The completed-output checksums, config, frozen model and GRID order must match.
    """
    from .common import completed
    report=completed(folder)
    if (not report or report.get('region')!='EAS' or report.get('data_role')!='real_data'
        or report.get('model',{}).get('sha256')!=inference.DIGEST or report.get('config')!=inputs[3]
        or report.get('years')!=list(YEARS) or report.get('n_grids')!=len(inputs[1])):
        raise ValueError('Need a completed compatible real EAS run, with the same spatial settings and BOUND model.')
    saved=pd.read_csv(folder/'nn_coefficients.csv',dtype={'grid_id':str})
    if saved.grid_id.tolist()!=inputs[1].grid_id.tolist():
        raise ValueError('Real calibration GRID order differs')
    return report


def paired_accuracy(nn, mle):
    keys=['stage','coefficient','n_grids']
    a=nn.copy();b=mle.copy()
    a['stage']=a.stage.replace({'nn':'estimation'})
    b['stage']=b.stage.replace({'mle':'estimation'})
    if a.duplicated(keys).any() or b.duplicated(keys).any(): raise ValueError('Duplicate metrics')
    joined=a.merge(b,on=keys,suffixes=('_NN','_MLE'),how='outer',indicator=True,validate='one_to_one')
    if len(joined)!=10 or not joined._merge.eq('both').all():
        raise ValueError('Both methods must cover all five coefficients and both stages on identical GRID counts')
    joined=joined.drop(columns='_merge')
    for c in ('RMSE_NN','RMSE_MLE','MAE_NN','MAE_MLE','Bias_NN','Bias_MLE'):
        if not np.isfinite(joined[c]).all(): raise ValueError('Invalid paired metrics')
    joined['NN_improvement_percent']=100*(1-np.divide(joined.RMSE_NN,joined.RMSE_MLE,
        out=np.full(len(joined),np.nan),where=joined.RMSE_MLE.to_numpy()>0))
    return joined


def summarize_pair(index, branches, values, truth, grids):
    reports={name:read(folder/'run_report.json') for name,folder in branches.items()}
    for name,folder in branches.items():
        if not np.array_equal(np.load(folder/'annual_maxima.npy',allow_pickle=False),values):
            raise ValueError('Paired observations differ')
        if not np.array_equal(np.load(folder/'truth.npy',allow_pickle=False),truth):
            raise ValueError('Paired truth differs')
        frame=pd.read_csv(folder/f'{name}_coefficients.csv',dtype={'grid_id':str})
        if frame.grid_id.tolist()!=grids.grid_id.tolist(): raise ValueError('Paired GRID order differs')
    keys=['grid_id','coefficient','outer_fold']
    folds=[pd.read_csv(branches[m]/'spatial_oof.csv',dtype={'grid_id':str})[keys].sort_values(keys).reset_index(drop=True)
           for m in ('nn','mle')]
    if not folds[0].equals(folds[1]): raise ValueError('Outer fold memberships differ')
    accuracy=paired_accuracy(*(pd.read_csv(branches[m]/'truth_parameter_metrics.csv') for m in ('nn','mle')))
    accuracy.insert(0,'replicate',index+1)
    timing=dict(replicate=index+1,timing_comparable=reports['nn']['run_environment']==reports['mle']['run_environment'],
        same_estimator_worker_budget=reports['mle']['model']['jobs']==1,
        nn_output=str(branches['nn']),mle_output=str(branches['mle']))
    for stage,key in (('estimator','estimator_seconds'),('spatial_cv','join_preflight_nested_cv_seconds'),
                      ('workflow','total_from_estimator_start_until_outputs_seconds')):
        for method in ('nn','mle'):
            timing[f'{stage}_{method}_seconds']=reports[method]['timings'][key]
        timing[f'{stage}_speedup_MLE_over_NN']=(timing[f'{stage}_mle_seconds']/timing[f'{stage}_nn_seconds']
            if timing['timing_comparable'] and timing[f'{stage}_nn_seconds']>0 else np.nan)
    quality=[]
    for method in ('nn','mle'):
        report=reports[method]
        selection=pd.read_csv(branches[method]/'spatial_selections.csv')
        row=dict(replicate=index+1,method=method,n_grids=len(grids),failed_estimation_grids=0,
            estimator_support_failures=report['support_failure_grids'][method],
            gp_oof_support_failures=report['support_failure_grids']['gp_oof'],
            outer_gp_optimizer_warnings=int((~selection.optimizer_success.astype(bool)).sum()),
            mle_boundary_grids=np.nan,mle_retry_grids=np.nan,mle_lower_nonconverged_nll_grids=np.nan)
        if method=='mle':
            fits=pd.read_csv(branches[method]/'mle_domain_checks.csv')
            row.update(mle_boundary_grids=int(fits.on_boundary.sum()),mle_retry_grids=int(fits.retried.sum()),
                mle_lower_nonconverged_nll_grids=int((fits.lower_nonconverged_nll_gap>1e-4).sum()))
        quality.append(row)
    return accuracy,timing,pd.DataFrame(quality)


def paired_crps(index, branches, evaluation_values, *, yearly=False):
    """Paired full-coverage means, never a subset chosen by score availability."""
    tables={}
    filename='crps_yearly_metrics.csv' if yearly else 'crps_metrics.csv'
    for method,folder in branches.items():
        saved=np.load(folder/'crps_evaluation_maxima.npy',allow_pickle=False)
        if not np.array_equal(saved,evaluation_values): raise ValueError('Paired CRPS evaluation observations differ')
        table=pd.read_csv(folder/filename,dtype={'year':str})
        table['stage']=table.stage.replace({method:'estimation'})
        if table.duplicated(['stage','year']).any(): raise ValueError('Duplicate CRPS metrics')
        tables[method]=table
    keys=['stage','year','n_total']
    a,b=(tables[m] for m in ('nn','mle'))
    oracle=a[a.stage.eq('truth')].sort_values('year').reset_index(drop=True)
    other=b[b.stage.eq('truth')].sort_values('year').reset_index(drop=True)
    if not oracle.equals(other): raise ValueError('Paired CRPS oracle differs')
    result=a[~a.stage.eq('truth')].merge(b[~b.stage.eq('truth')],on=keys,
        suffixes=('_NN','_MLE'),how='outer',indicator=True,validate='one_to_one')
    if len(result)!=(100 if yearly else 2) or not result._merge.eq('both').all():
        raise ValueError('Both CRPS methods must cover every GRID/year and stage')
    expected=len(evaluation_values) if yearly else evaluation_values.size
    if not result.n_total.eq(expected).all(): raise ValueError('CRPS scoring silently changed coverage')
    result=result.drop(columns='_merge').merge(oracle[['year','mean_CRPS_C','n_invalid']],on='year',validate='many_to_one')
    result=result.rename(columns={'mean_CRPS_C':'oracle_CRPS_C','n_invalid':'oracle_invalid_cells'})
    result['NN_improvement_percent']=100*(1-result.mean_CRPS_C_NN/result.mean_CRPS_C_MLE.replace(0,np.nan))
    result.insert(0,'replicate',index+1)
    return result


def run_replicates(inputs, truth, output, count, *, calibration_id, threads=1, mle_options=None):
    if not 1<=count<=100: raise ValueError('Replicates must be 1..100')
    _,grids,predictors,config,crs=inputs
    truth=np.asarray(truth,float)
    if truth.shape!=(len(grids),5) or not np.isfinite(truth).all(): raise ValueError('Invalid calibrated truth')
    options=dict(domain='wide_domain',jobs=1,maxiter=600,retry_maxiter=1800)
    options.update(mle_options or {})
    # Validate MLE settings before any expensive GP work.
    if (options['domain'] not in ('wide_domain','training_domain') or options['jobs']==0
        or options['maxiter']<1 or options['retry_maxiter']<options['maxiter']):
        raise ValueError('Invalid MLE settings')
    protocol=dict(version=2,calibration_id=calibration_id,truth_sha256=array_hash(truth),
        grids_sha256=identity(grids.to_json()),predictors_sha256=identity(predictors.to_json()),
        config=config,coordinate_crs=crs,years=list(YEARS),seed=SEED,threads=threads,
        bound_model_sha256=inference.DIGEST,mle_options=options,source_sha256=code_identity(),
        crps=dict(target='Annual-maximum temperature plug-in GEV; degrees C, lower is better',
            fit_stream='SeedSequence(seed, spawn_key=(replicate,))',
            evaluation_stream='SeedSequence(seed, spawn_key=(replicate, 0x43525053))',
            evaluation='Independent sample at same 50 years and GRID, shared by both arms. Never used to fit or select.',
            domain='xi < 1; outside-support observations scored. Any invalid cell invalidates the aggregate mean.',
            uncertainty='No integration over parameter or GP uncertainty; oracle true-GEV reference need not be zero.'),
        model_selection='Same candidates and folds; each arm selects predictors/kernel independently inside inner CV.',
        ordering='Odd replicates NN first, even replicates MLE first; original completed timings retained on resume.',
        failure_policy='No GRID deletion or NN fallback; any failed MLE blocks that replicate before MLE GP.',
        timing_scope='Wall seconds: fresh estimator separately and estimator-through-full-output. No calibration/generation/preprocessing.',
        calibration_limitation='Truth is NN/GP-calibrated; conclusions conditional on this DGP, not proof of universal superiority.')
    output.mkdir(parents=True,exist_ok=True)
    pp=output/'comparison_protocol.json'
    if pp.exists() and read(pp)!=protocol: raise ValueError('Comparison protocol changed; use a new comparison output directory')
    saved_metrics=output/'parameter_comparison.csv'
    if saved_metrics.exists() and pd.read_csv(saved_metrics).replicate.max()>count:
        raise ValueError('Requested count would truncate completed comparison summaries; retain the larger count')
    write(pp,protocol);pid=identity(protocol)
    env=environment(threads,config['n_jobs'],options['jobs'])
    metrics=[];times=[];quality=[];crps=[];crps_yearly=[]
    try:
        write(output/'run_status.json',dict(status='running',completed_pairs=0,total=count,stage='validate_or_run_pairs'))
        for index in range(count):
            started=time.perf_counter();values=simulation.draw_annual(truth,index)
            evaluation_values=simulation.draw_annual(truth,index,purpose='crps_evaluation')
            generation_seconds=time.perf_counter()-started
            # The identical generated array is passed to both arms, never resampled.
            sample_id=identity(dict(protocol=pid,replicate=index+1,values=array_hash(values),
                                   evaluation_values=array_hash(evaluation_values)))
            branches={}; order=('nn','mle') if index%2==0 else ('mle','nn')
            for method in order:
                parent=output/'replicates'/f'replicate_{index+1:03d}'/method
                method_id=identity(dict(sample=sample_id,estimator=method))
                prior=latest_completed(parent,method_id)
                if prior and (prior[1].get('sample_id')!=sample_id or 'run_environment' not in prior[1]):
                    # Interrupted between workflow completion and paired metadata:
                    # keep the old attempt, but do not invent its hardware/timing.
                    prior=None
                if prior:
                    folder,_=prior
                else:
                    folder=attempt_folder(parent)
                    write(output/'run_status.json',dict(status='running',replicate=index+1,total=count,
                        estimator=method,completed_pairs=len(metrics),progress_file=str(folder/'run_status.json')))
                    report=workflow.execute(values,grids,predictors,config,crs,folder,truth=truth,
                        protocol_id=method_id,threads=threads,estimator=method,mle_options=options,
                        evaluation_values=evaluation_values)
                    report.update(replicate=index+1,run_environment=env,sample_id=sample_id,
                        method_order=list(order),generation_seconds=generation_seconds)
                    write(folder/'run_report.json',report)
                branches[method]=folder
            a,t,q=summarize_pair(index,branches,values,truth,grids)
            crps.append(paired_crps(index,branches,evaluation_values))
            crps_yearly.append(paired_crps(index,branches,evaluation_values,yearly=True))
            t['generation_seconds']=generation_seconds
            for method,folder in branches.items():
                t[f'crps_{method}_seconds']=read(folder/'run_report.json')['timings']['crps_scoring_and_writing_seconds']
            metrics.append(a);times.append(t);quality.append(q)
            all_metrics=pd.concat(metrics,ignore_index=True)
            atomic_csv(output/'parameter_comparison.csv',all_metrics)
            atomic_csv(output/'timing_comparison.csv',pd.DataFrame(times))
            atomic_csv(output/'quality_comparison.csv',pd.concat(quality,ignore_index=True))
            all_crps=pd.concat(crps,ignore_index=True)
            atomic_csv(output/'crps_comparison.csv',all_crps)
            atomic_csv(output/'crps_yearly_comparison.csv',pd.concat(crps_yearly,ignore_index=True))
            cs=all_crps.groupby('stage',sort=False).agg(n_replicates=('replicate','count'),
                NN_mean_CRPS_C=('mean_CRPS_C_NN','mean'),MLE_mean_CRPS_C=('mean_CRPS_C_MLE','mean'),
                NN_sd_CRPS_C=('mean_CRPS_C_NN','std'),MLE_sd_CRPS_C=('mean_CRPS_C_MLE','std'),
                NN_invalid_cells=('n_invalid_NN','sum'),MLE_invalid_cells=('n_invalid_MLE','sum'),
                oracle_invalid_cells=('oracle_invalid_cells','sum'),
                oracle_mean_CRPS_C=('oracle_CRPS_C','mean'),oracle_valid_replicates=('oracle_CRPS_C','count')).reset_index()
            for method in ('NN','MLE'):
                cs.loc[cs[f'{method}_invalid_cells']>0,[f'{method}_mean_CRPS_C',f'{method}_sd_CRPS_C']]=np.nan
            cs.loc[cs.oracle_valid_replicates!=cs.n_replicates,'oracle_mean_CRPS_C']=np.nan
            cs['NN_improvement_of_mean_CRPS_percent']=100*(1-cs.NN_mean_CRPS_C/cs.MLE_mean_CRPS_C.replace(0,np.nan))
            atomic_csv(output/'crps_summary.csv',cs)
            summary=all_metrics.groupby(['stage','coefficient'],sort=False).agg(
                n_replicates=('replicate','count'),NN_mean_RMSE=('RMSE_NN','mean'),MLE_mean_RMSE=('RMSE_MLE','mean'),
                NN_sd_RMSE=('RMSE_NN','std'),MLE_sd_RMSE=('RMSE_MLE','std')).reset_index()
            summary['NN_improvement_of_mean_RMSE_percent']=100*(1-summary.NN_mean_RMSE/summary.MLE_mean_RMSE.replace(0,np.nan))
            atomic_csv(output/'parameter_summary.csv',summary)
            write(output/'run_status.json',dict(status='running',completed_pairs=len(metrics),total=count))
        invalid_crps=int(all_crps[['n_invalid_NN','n_invalid_MLE']].to_numpy().sum())
        oracle_invalid=int(all_crps.drop_duplicates('replicate').oracle_invalid_cells.sum())
        report=dict(status='completed_with_warnings' if invalid_crps or oracle_invalid else 'completed',completed_pairs=len(metrics),total=count,protocol_id=pid,
            crps_invalid_cells=invalid_crps,crps_evaluated=True,
            crps_oracle_invalid_cells=oracle_invalid,
            timing_noncomparable_pairs=sum(not t['timing_comparable'] for t in times),
            all_grids_retained=True,formal_significance_tests=False,
            output_sha256={p.name:sha(p) for p in output.glob('*.csv')})
        write(output/'run_report.json',report);write(output/'run_status.json',report)
        return report
    except BaseException as exc:
        write(output/'run_status.json',dict(status='failed',completed_pairs=len(metrics),total=count,
            replicate=index+1 if 'index' in locals() else None,error_type=type(exc).__name__,error=str(exc)))
        raise


def compare_simulations(real_folder, inputs, experiment, count, *, threads=1, mle_options=None):
    validate_real_source(real_folder,inputs)
    _,grids,predictors,config,crs=inputs
    truth,calibration=simulation.calibrate(real_folder,grids,predictors,config,crs,
                                          experiment/'calibration',threads=threads)
    return run_replicates(inputs,truth,experiment/'mle_comparison_v2_crps',count,
        calibration_id=calibration['protocol_id'],threads=threads,mle_options=mle_options)
