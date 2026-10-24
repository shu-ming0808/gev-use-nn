"""CPU NN or MLE -> shared grouped FFS/nested GP -> OOF/support/RL."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import torch
import psutil
from joblib import effective_n_jobs
from threadpoolctl import threadpool_limits
from .common import ROOT, SEED, EAS_CRS, read, write, sha
from .inference import infer
from research_workflows import COEFFICIENTS, YEARS, REFERENCE_YEAR, TIME_SCALE_YEARS, validate_annual_sequences
from taiwan_simulation import cheng_spatial_cv as cv
from taiwan_simulation.run_timed_workflow import ordered_oof
from cheng_nn_evaluation import check_gev_predictions, conditional_return_levels
from train_cheng_nn import atomic_csv, atomic_array
from taiwan_simulation.gev_crps import score_temperature_distribution

def settings(groups, n, *, outer=5, inner=3, buffer=500., min_train=100, seed=SEED):
    return dict(candidate_groups=groups, buffers_km={c:float(buffer) for c in COEFFICIENTS},
        outer_folds=outer, inner_folds=inner, min_train=min_train, max_train=n, seed=seed,
        n_jobs=1, n_restarts=0, min_relative_improvement=.01, maximum_allowed_vif=5., max_steps=None,
        kernel_candidates=[['RBF',None],['Matern',.5],['Matern',1.5],['Matern',2.5]],
        buffer_distance_mode='wgs84', gp_options=dict(length_scale_bounds=[1.,6000.],initial_length_scale=300.))

def eas_inputs():
    folder = ROOT / 'inputs/eas'
    grids = pd.read_csv(folder/'grid_index.csv', dtype={'grid_id':str})
    values = validate_annual_sequences(np.load(folder/'years.npy'), np.load(folder/'annual_maxima.npy'))
    predictors = pd.read_csv(folder/'grid_predictors.csv', dtype={'grid_id':str})
    groups = read(folder/'candidate_groups.json')
    if len(grids) != 7822 or len(values) != 7822 or not grids.lsm.gt(.5).all():
        raise ValueError('EAS expected 7,822 land GRID')
    if read(folder/'predictor_protocol.json')['registry_sha256']!=sha(folder/'grid_index.csv'):
        raise ValueError('Predictor GRID registry provenance changed')
    source=pd.read_csv(ROOT/'inputs/global_annual/grid_index.csv',dtype={'grid_id':str}).set_index('grid_id')
    indices=source.index.get_indexer(grids.grid_id)
    if (indices<0).any() or source.index.duplicated().any():raise ValueError('Annual source GRID mismatch')
    actual=np.load(ROOT/'inputs/global_annual/annual_maxima_land.npy',mmap_mode='r',allow_pickle=False)[indices]
    if not np.array_equal(values,actual):raise ValueError('EAS sequences differ from verified global annual source')
    if not np.allclose(source.iloc[indices][['longitude_180','latitude','lsm']],
                       grids[['longitude_180','latitude','lsm']],rtol=0,atol=1e-6):
        raise ValueError('EAS annual/grid coordinates differ')
    validate_inputs(values, grids, predictors, settings(groups,len(grids)), EAS_CRS)
    return values, grids, predictors, settings(groups,len(grids)), EAS_CRS

def validate_inputs(values, grids, predictors, config, crs):
    validate_annual_sequences(np.asarray(YEARS), values)
    if len(values) != len(grids) or set(grids.grid_id) != set(predictors.grid_id):
        raise ValueError('GRID identity/sequence coverage differs')
    if not predictors.candidate_data_complete.eq(True).all():
        raise ValueError('Incomplete predictors; no silent deletion/imputation')
    dummy = pd.concat([grids.reset_index(drop=True),pd.DataFrame(np.zeros((len(grids),5)),columns=COEFFICIENTS)],axis=1)
    cv.join_predictors(dummy,predictors,config['candidate_groups'],coordinate_crs=crs)

def memory_plan(n, n_jobs=1):
    """Conservative full-GRID envelope per worker, not a measured peak.

    Candidates may have fewer buffered training points or fewer simultaneous
    tasks. Do not silently reduce the requested jobs or training GRID count.
    """
    if n < 1 or n_jobs == 0: raise ValueError('Positive GRID count and nonzero jobs required')
    workers = effective_n_jobs(n_jobs)
    per_worker = int(8 * n * n * 10 + 2_000_000_000)
    required = per_worker * workers
    available = int(psutil.virtual_memory().available)
    return dict(requested_jobs=int(n_jobs),effective_workers=int(workers),
        per_worker_estimate_bytes=per_worker,required_bytes=required,available_bytes=available,
        minimum_available_GB=required/.85/1e9,passes=bool(required<=.85*available),
        safety_fraction=.85,estimate='Conservative full-GRID per-worker estimate times workers; not measured peak RAM')

def memory_guard(n, n_jobs=1, *, enforce=False):
    """Report the estimate without blocking by default, at the user's request.

    This does not catch real allocation failures or change jobs/GRID counts.
    Explicit enforce=True retains the original screening for callers/tests.
    """
    plan = memory_plan(n, n_jobs)
    if not plan['passes']:
        message = (f"Exact GP screening: jobs={n_jobs}, effective workers={plan['effective_workers']}; "
            f"conservative estimate {plan['required_bytes']/1e9:.1f} GB; "
            f"available {plan['available_bytes']/1e9:.1f} GB; "
            f"need at least {plan['minimum_available_GB']:.1f} GB available including 15% reserve. "
            'Not a measured peak. No hidden worker reduction or training GRID cap. '
            'Choose fewer -Jobs explicitly or use more RAM; -Jobs 1 -Threads 20 uses one GP with matrix threads.')
        if enforce:
            raise MemoryError(message)
        print('[MEMORY ADVISORY ONLY] ' + message +
              ' Continuing as requested; real allocation errors or worker termination will still fail the run.', flush=True)
    return plan['required_bytes']

def errors(prediction, truth, stage):
    prediction=np.asarray(prediction); truth=np.asarray(truth)
    if prediction.shape != truth.shape or not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError('Cannot score nonfinite or mismatched coefficients')
    d=prediction-truth
    return pd.DataFrame([dict(stage=stage,coefficient=c,n_grids=len(d),
        RMSE=float(np.sqrt(np.mean(d[:,i]**2))),MAE=float(np.mean(np.abs(d[:,i]))),
        Bias=float(np.mean(d[:,i]))) for i,c in enumerate(COEFFICIENTS)])

def execute(values, grids, predictors, config, crs, output, *, region='EAS', truth=None,
            protocol_id='', threads=4, check_memory=True, estimator='nn', mle_options=None,
            evaluation_values=None):
    if estimator not in ('nn','mle'): raise ValueError('Estimator must be nn or mle')
    if output.exists(): raise FileExistsError(output)
    validate_inputs(values,grids,predictors,config,crs)
    if evaluation_values is not None:
        evaluation_values=validate_annual_sequences(np.asarray(YEARS),np.asarray(evaluation_values,float))
        if truth is None or evaluation_values.shape!=values.shape or np.array_equal(evaluation_values,values):
            raise ValueError('CRPS requires separate simulated evaluation sequences matching the fitted GRID/years')
    if check_memory: memory_guard(len(grids),config['n_jobs'])
    output.mkdir(parents=True)
    torch.set_num_threads(threads)
    start=time.perf_counter(); timings={}
    def status(stage, **detail):
        write(output/'run_status.json',dict(status='running',stage=stage,region=region,
              estimator=estimator,elapsed_from_estimator_start_seconds=time.perf_counter()-start,
              **({'elapsed_from_nn_start_seconds':time.perf_counter()-start} if estimator=='nn' else {}),**detail))
        print(stage,detail,flush=True)
    def progress(stage, detail, tables):
        if tables is not None:
            sub=output/'checkpoints'/f"outer_{detail['outer_fold']}_{detail['coefficient']}"
            sub.mkdir(parents=True,exist_ok=True)
            for name,table in tables.items(): atomic_csv(sub/f'{name}.csv',table)
        status(stage,**detail)
    try:
        with threadpool_limits(limits=threads):
            status(estimator)
            if estimator=='nn':
                result=infer(values)
            else:
                from taiwan_simulation.mle_estimator import infer_mle
                result=infer_mle(values,threads=threads,**(mle_options or {}))
            nn=np.asarray(result['prediction'],float)  # shared downstream response, not NN for the MLE arm
            timings['estimator_seconds']=time.perf_counter()-start
            if estimator=='nn':
                timings['nn_load_features_decoder_diagnostics_seconds']=timings['estimator_seconds']
            responses=pd.concat([grids.reset_index(drop=True),pd.DataFrame(nn,columns=COEFFICIENTS)],axis=1)
            atomic_csv(output/f'{estimator}_coefficients.csv',responses)
            atomic_csv(output/f'{estimator}_domain_checks.csv',pd.concat([grids[['grid_id']].reset_index(drop=True),result['checks']],axis=1))
            if estimator=='mle':
                from taiwan_simulation.mle_estimator import MLEFitFailure
                failed=~result['checks'].success.to_numpy(bool) | ~np.isfinite(nn).all(axis=1)
                write(output/'mle_fit_report.json',dict(model=result['model'],timings=result['timings'],
                    n_requested=len(nn),n_failed=int(failed.sum()),
                    failure_policy='Stop before GP; retain every GRID and report failures, never use NN fallback.'))
                if failed.any():
                    raise MLEFitFailure(f'{int(failed.sum())}/{len(nn)} MLE GRID failed; diagnostics saved, spatial CV not started.')
            t=time.perf_counter(); status('nested_spatial_cv')
            joined=cv.join_predictors(responses,predictors,config['candidate_groups'],coordinate_crs=crs)
            tables=cv.nested_cv(joined,**config,progress=progress)
            gp=ordered_oof(tables['oof'],grids,nn)
            if estimator=='mle':
                for key,table in tables.items():
                    tables[key]=table.rename(columns={c:c.replace('NN_reference','MLE_reference').replace('reference_nn','reference_mle')
                                                      for c in table.columns})
            timings['join_preflight_nested_cv_seconds']=time.perf_counter()-t
            atomic_csv(output/'oof_coefficients.csv',pd.concat([grids,pd.DataFrame(gp,columns=COEFFICIENTS)],axis=1))
            for name,table in tables.items(): atomic_csv(output/f'spatial_{name}.csv',table)
            t=time.perf_counter(); status('support_rl_and_truth_scoring')
            failures={}; arrays={estimator:nn,'gp_oof':gp}
            if truth is not None:
                truth=np.asarray(truth,float)
                atomic_array(output/'truth.npy',truth); atomic_array(output/'annual_maxima.npy',values)
                metrics=pd.concat([errors(nn,truth,estimator),errors(gp,truth,'gp_oof')],ignore_index=True)
                atomic_csv(output/'truth_parameter_metrics.csv',metrics)
                arrays['truth']=truth
            for name,coeff in arrays.items():
                checks=check_gev_predictions(coeff,values,np.asarray(YEARS),reference_year=REFERENCE_YEAR,
                                            time_scale_years=TIME_SCALE_YEARS)
                checks.insert(0,'grid_id',grids.grid_id.to_numpy())
                failures[name]=int((~checks.gev_valid_all_years).sum())
                atomic_csv(output/f'{name}_support.csv',checks)
            levels={k:conditional_return_levels(v,np.asarray(YEARS),(50.,100.),
                reference_year=REFERENCE_YEAR,time_scale_years=TIME_SCALE_YEARS) for k,v in arrays.items()}
            rl=pd.MultiIndex.from_product([grids.grid_id,YEARS,(50.,100.)],
                names=['grid_id','year','return_period']).to_frame(index=False)
            for name,a in levels.items(): rl[name]=a.ravel()
            atomic_csv(output/'return_levels.csv',rl)
            rows=[]; reference='truth' if truth is not None else estimator
            for name in (estimator,'gp_oof'):
                if name==reference: continue
                for j,period in enumerate((50,100)):
                    d=levels[name][:,:,j]-levels[reference][:,:,j]; finite=np.isfinite(d)
                    rows.append(dict(stage=name,reference=reference,return_period=period,
                        n_total=d.size,n_used=int(finite.sum()),n_invalid=int((~finite).sum()),
                        RMSE=float(np.sqrt(np.mean(d[finite]**2))) if finite.any() else None))
            atomic_csv(output/'return_level_metrics.csv',pd.DataFrame(rows))
            timings['support_rl_truth_scoring_and_writing_seconds']=time.perf_counter()-t
            crps_invalid={}
            if evaluation_values is not None:
                t=time.perf_counter();status('independent_temperature_crps')
                atomic_array(output/'crps_evaluation_maxima.npy',evaluation_values)
                overall=[];yearly=[]
                for name,coeff in arrays.items():
                    scores,summary,by_year=score_temperature_distribution(coeff,evaluation_values,stage=name)
                    atomic_array(output/f'{name}_crps.npy',scores)
                    overall.append(summary);yearly.append(by_year)
                    crps_invalid[name]=int(summary.iloc[0].n_invalid)
                atomic_csv(output/'crps_metrics.csv',pd.concat(overall,ignore_index=True))
                atomic_csv(output/'crps_yearly_metrics.csv',pd.concat(yearly,ignore_index=True))
                timings['crps_scoring_and_writing_seconds']=time.perf_counter()-t
            output_hashes={p.name:sha(p) for p in output.iterdir() if p.suffix in ('.csv','.npy')}
            timings['total_from_estimator_start_until_outputs_seconds']=time.perf_counter()-start
            if estimator=='nn':
                timings['total_from_nn_start_until_outputs_seconds']=timings['total_from_estimator_start_until_outputs_seconds']
            report=dict(status='completed_with_warnings' if any(failures.values()) or any(r['n_invalid'] for r in rows) or any(crps_invalid.values()) else 'completed',
                workflow_completed=True, protocol_id=protocol_id, region=region,config=config,coordinate_crs=crs,
                n_grids=len(grids),years=list(YEARS),model=result['model'],device='cpu',threads=threads,
                memory_screen_policy='advisory_only_user_requested',
                estimator=estimator,timings=timings,estimator_component_timings=result['timings'],
                **({'nn_component_timings':result['timings']} if estimator=='nn' else {}),support_failure_grids=failures,
                data_role='calibrated_simulation' if truth is not None else 'real_data',
                crps_evaluated=evaluation_values is not None,crps_invalid_cells=crps_invalid,
                crps_protocol=('Independent 50-year sample at the same GRID/years and truth; plug-in GEV, xi < 1. '
                    'Outside-support observations are scored; any invalid cell makes the aggregate mean unavailable. '
                    'Evaluation data never enter inference, decoder, GP fitting or model selection.'
                    if evaluation_values is not None else None),
                metric_reference='known_simulated_truth' if truth is not None else estimator.upper()+'_reference_not_climate_truth',
                observed_support_policy=('NN decoder uses each input sequence.' if estimator=='nn' else
                    'MLE likelihood enforces support on each fitted input sequence.')+
                    ' OOF GP predictions never repaired using held-out observations.',
                not_run=['temporal_model_selection','change_point','formal_model_replacement','spatial_residual_diagnostics'],
                timing_scope='Fresh estimator (NN load/features/decoder or MLE fitting) through nested CV, RL/truth and optional CRPS scoring, output and hashes. Excludes input validation, calibration, generation and report JSON.',
                output_sha256=output_hashes)
            write(output/'run_report.json',report)
            write(output/'run_status.json',dict(status=report['status'],workflow_completed=True))
            return report
    except BaseException as exc:
        write(output/'run_status.json',dict(status='failed',workflow_completed=False,error=str(exc),
            estimator=estimator,elapsed_from_estimator_start_seconds=time.perf_counter()-start,
            **({'elapsed_from_nn_start_seconds':time.perf_counter()-start} if estimator=='nn' else {})))
        raise
