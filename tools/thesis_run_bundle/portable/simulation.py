"""Conditional parametric simulation on a fixed, real-data-calibrated surface.

This version draws new observations, not new GP surfaces. Given the five
coefficient surfaces, annual/grid draws are independent. It does not establish
spatial joint-event uncertainty or real-world confidence-interval coverage.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from .common import SEED, read, write, sha, identity, latest_completed, attempt_folder
from .workflow import errors, execute, memory_guard
from research_workflows import COEFFICIENTS, YEARS
from spatial_predictor_selection import spatial_forward_selection
from land_cover_gp_analysis import fit_gp_with_covariates, predict_gp_with_covariates
from taiwan_simulation.cheng_spatial_cv import join_predictors, TARGET_COLUMNS
from train_cheng_nn import atomic_csv, atomic_array
from gp_execution import refit_context, describe

def draw_annual(truth, replicate, master_seed=SEED, *, purpose='fit'):
    """Distinct reproducible stream per replicate; EVT xi (not scipy's -xi)."""
    theta=np.asarray(truth,dtype=float)
    if theta.ndim!=2 or theta.shape[1]!=5 or not np.isfinite(theta).all():
        raise ValueError('Five finite truth coefficients required')
    if replicate < 0: raise ValueError('Replicate must be nonnegative')
    if purpose not in ('fit','crps_evaluation'): raise ValueError('Unknown simulation stream purpose')
    # Preserve every existing fitting draw. Evaluation uses a separate tagged
    # stream at the SAME 50 years, never extra future years or shortened inputs.
    key=(int(replicate),) if purpose=='fit' else (int(replicate),0x43525053)
    rng=np.random.default_rng(np.random.SeedSequence(master_seed,spawn_key=key))
    t=(np.asarray(YEARS)-2000.5)/10
    mu=theta[:,0,None]+theta[:,1,None]*t
    with np.errstate(over='raise',invalid='raise'):
        sigma=np.exp(theta[:,2,None]+theta[:,3,None]*t)
        u=rng.random(mu.shape)
        u=np.clip(u,np.nextafter(0.,1.),np.nextafter(1.,0.))
        q=-np.log(-np.log(u)); xi=theta[:,4,None]
        # expm1 avoids cancellation; use the Gumbel limit only at exactly xi=0.
        ratio=np.divide(np.expm1(xi*q),xi,out=q.copy(),where=(xi!=0))
        y=mu+sigma*ratio
    support=1+xi*(y-mu)/sigma
    if not np.isfinite(y).all() or not np.isfinite(sigma).all() or np.any(sigma<=0) or np.any(support<=0):
        raise ValueError('Generation overflow/support failure; no clipping of temperatures or truth')
    return y

def calibrate(real_folder, grids, predictors, config, crs, output, *, threads=4):
    """One full-domain FFS refit per coefficient; posterior means become fixed truth.

    Full real data may calibrate a DGP, but its fitted errors are never called
    OOF validation. Each simulation independently repeats nested FFS evaluation.
    """
    real=read(real_folder/'run_report.json')
    if not real.get('workflow_completed') or real.get('region')!='EAS' or real.get('data_role')!='real_data':
        raise ValueError('A completed EAS real run is required for calibration')
    src=real_folder/'nn_coefficients.csv'
    if sha(src)!=real['output_sha256']['nn_coefficients.csv']:
        raise ValueError('Real NN coefficients changed')
    protocol=dict(version=1,real_nn_sha256=sha(src),config=config,coordinate_crs=crs,execution_policy=describe(),
        truth_method='full-domain FFS/kernel selection then GP posterior mean, fixed across replicates',
        generation='independent annual/grid GEV conditional on truth; no new latent GP draw',
        coefficient_names=list(COEFFICIENTS),years=list(YEARS),master_seed=SEED)
    pid=identity(protocol)
    output.mkdir(parents=True,exist_ok=True)
    pp=output/'protocol.json'
    if pp.exists() and read(pp)!=protocol: raise ValueError('Calibration inputs changed; use a new experiment name')
    write(pp,protocol)
    final=output/'run_report.json'
    if final.exists():
        report=read(final)
        if report.get('protocol_id')!=pid: raise ValueError('Calibration protocol mismatch')
        for name,digest in report['output_sha256'].items():
            if sha(output/name)!=digest: raise ValueError('Calibration output changed')
        return np.load(output/'truth.npy'),report
    memory_guard(len(grids),config['n_jobs'])
    response=pd.read_csv(src,dtype={'grid_id':str}).set_index('grid_id').loc[grids.grid_id].reset_index()
    frame=join_predictors(response,predictors,config['candidate_groups'],coordinate_crs=crs)
    start=time.perf_counter(); columns=[]; summaries=[]
    with threadpool_limits(limits=threads):
        frame['spatial_fold']=KMeans(n_clusters=config['inner_folds'],n_init=100,
            random_state=SEED+9000).fit_predict(frame[['x_km','y_km']])
        for i,target in enumerate(COEFFICIENTS):
            part=output/target; part.mkdir(exist_ok=True)
            saved=part/'completed.json'
            if saved.exists():
                prior=read(saved)
                if prior['protocol_id']!=pid: raise ValueError('Changed coefficient calibration')
                for name,digest in prior['hashes'].items():
                    if sha(part/name)!=digest: raise ValueError('Changed calibration checkpoint')
                columns.append(np.load(part/'truth.npy')); summaries.append(prior['summary']); continue
            write(output/'run_status.json',dict(status='running',stage='full_data_calibration',coefficient=target,
                completed_coefficients=i,total_coefficients=5))
            print('calibrate',target,flush=True)
            trials,path,winner=spatial_forward_selection(frame,target,target_columns=TARGET_COLUMNS,
                candidate_groups=config['candidate_groups'],buffer_km=config['buffers_km'][target],
                n_folds=config['inner_folds'],min_train=config['min_train'],max_train=len(frame),
                n_jobs=config['n_jobs'],n_restarts=config['n_restarts'],random_state=SEED+9000,
                min_relative_improvement=config['min_relative_improvement'],
                maximum_allowed_vif=config['maximum_allowed_vif'],max_steps=config['max_steps'],
                kernel_candidates=tuple(tuple(k) for k in config['kernel_candidates']),
                gp_options=config['gp_options'],buffer_distance_mode=config['buffer_distance_mode'])
            cols=winner['predictor_names']; xy=frame[['x_km','y_km']].to_numpy()
            with refit_context(f'calibration_{target}_refit'):
                fit=fit_gp_with_covariates(xy,frame[cols].to_numpy(),frame[target].to_numpy(),cols,
                    kernel_name=winner['kernel'],nu=winner['nu'],n_restarts=config['n_restarts'],
                    seed=SEED+9000,**config['gp_options'])
                # Predict in chunks to bound the N x N cross-covariance allocation.
                truth=np.concatenate([predict_gp_with_covariates(fit,xy[j:j+512],
                    frame[cols].iloc[j:j+512].to_numpy()) for j in range(0,len(frame),512)])
            if not np.isfinite(truth).all(): raise ValueError('Nonfinite calibrated surface')
            summary=dict(coefficient=target,predictors=cols,kernel=winner['kernel'],
                nu=None if pd.isna(winner['nu']) else float(winner['nu']),
                optimizer_success=bool(fit['optimizer_success']),optimizer_message=str(fit['optimizer_message']),
                fitted_kernel=str(fit['kernel']),n_grids=len(frame),min=float(truth.min()),max=float(truth.max()))
            atomic_array(part/'truth.npy',truth); atomic_csv(part/'trials.csv',trials); atomic_csv(part/'path.csv',path)
            write(saved,dict(protocol_id=pid,summary=summary,
                hashes={n:sha(part/n) for n in ('truth.npy','trials.csv','path.csv')}))
            columns.append(truth); summaries.append(summary)
    truth=np.column_stack(columns)
    draw_annual(truth,0)  # numerical validation, never a hidden repair
    atomic_array(output/'truth.npy',truth)
    atomic_csv(output/'truth_coefficients.csv',pd.concat([grids,pd.DataFrame(truth,columns=COEFFICIENTS)],axis=1))
    write(output/'selected_models.json',summaries)
    report=dict(status='completed',protocol_id=pid,protocol=protocol,
        elapsed_this_calibration_attempt_seconds=time.perf_counter()-start,
        optimizer_warning_coefficients=[s['coefficient'] for s in summaries if not s['optimizer_success']],
        output_sha256={n:sha(output/n) for n in ('truth.npy','truth_coefficients.csv','selected_models.json')},
        limitation='Conditional simulation, not known climate truth. No cross-grid event or serial dependence, no uncertainty in calibration surfaces.')
    write(final,report); write(output/'run_status.json',dict(status='completed'))
    return truth,report

def simulate(real_folder, inputs, experiment, count, *, threads=4):
    if count not in (1,100): raise ValueError('Use 1 or 100 replicates')
    _,grids,predictors,config,crs=inputs
    truth,calibration=calibrate(real_folder,grids,predictors,config,crs,experiment/'calibration',threads=threads)
    protocol=dict(version=1,calibration_id=calibration['protocol_id'],truth_sha256=sha(experiment/'calibration/truth.npy'),
        data_seed=SEED,fold_seed=config['seed'],random_stream='SeedSequence(master_seed, spawn_key=(replicate,))',
        config=config,coordinate_crs=crs,bound_model='03890b4877a40cf93822877d09231c003c920f34c68ed911b1da481ce396318f')
    pid=identity(protocol); target=experiment/'simulation_protocol.json'
    if target.exists() and read(target)!=protocol: raise ValueError('Simulation protocol changed')
    write(target,protocol)
    rows=[]; times=[]
    for index in range(count):
        parent=experiment/'replicates'/f'replicate_{index+1:03d}'
        prior=latest_completed(parent,pid)
        if prior:
            folder,report=prior
        else:
            before=time.perf_counter(); values=draw_annual(truth,index)
            gen=time.perf_counter()-before; folder=attempt_folder(parent)
            write(experiment/'run_status.json',dict(status='running',stage='simulate_and_evaluate',
                replicate=index+1,total=count,progress_file=str(folder/'run_status.json')))
            report=execute(values,grids,predictors,config,crs,folder,truth=truth,
                protocol_id=pid,threads=threads,region='EAS')
            report['generation_seconds']=gen
            report['replicate']=index+1
            write(folder/'run_report.json',report)
        rows.append(pd.read_csv(folder/'truth_parameter_metrics.csv').assign(replicate=index+1))
        times.append(dict(replicate=index+1,nn_to_end_seconds=report['timings']['total_from_nn_start_until_outputs_seconds'],
            generation_seconds=report['generation_seconds'],nn_support_failure_grids=report['support_failure_grids']['nn'],
            gp_support_failure_grids=report['support_failure_grids']['gp_oof']))
        all_metrics=pd.concat(rows,ignore_index=True)
        atomic_csv(experiment/'parameter_metrics_all_replicates.csv',all_metrics)
        atomic_csv(experiment/'timing_and_support.csv',pd.DataFrame(times))
        summary=all_metrics.groupby(['stage','coefficient']).RMSE.agg(['count','mean','std','min','median','max']).reset_index()
        atomic_csv(experiment/'parameter_rmse_summary.csv',summary)
        write(experiment/'run_status.json',dict(status='running',completed_replicates=index+1,total=count))
    write(experiment/'run_status.json',dict(status='completed',completed_replicates=count,total=count,
        interpretation='Repeated generated observations on fixed calibrated surfaces; not real-data reruns.'))
