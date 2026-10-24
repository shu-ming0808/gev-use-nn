"""Post-hoc location diagnostics. Does not repair OOF predictions or select models."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from portable.common import ROOT, read, write, sha
from elevation_gp_analysis import _buffered_training_indices, sample_indices
from land_cover_gp_analysis import fit_gp_with_covariates, predict_gp_with_covariates, _standardized_design
from taiwan_simulation.cheng_spatial_cv import join_predictors
from spatial_coordinates import project_lonlat_to_km


def diagnose(bundle, result, output, refit_folds=(1,3), threads=4):
    if output.exists(): raise FileExistsError(output)
    report=read(result/'run_report.json'); config=report['config']
    if not report['workflow_completed']: raise ValueError('Completed source required')
    for name,h in report['output_sha256'].items():
        if sha(result/name)!=h: raise ValueError('Source output changed: '+name)
    nn=pd.read_csv(result/'nn_coefficients.csv'); oof=pd.read_csv(result/'spatial_oof.csv')
    predictors=pd.read_csv(bundle/'inputs/eas/grid_predictors.csv')
    frame=join_predictors(nn,predictors,config['candidate_groups'],coordinate_crs=report['coordinate_crs'])
    mu=oof[oof.coefficient.eq('mu0')].set_index('grid_id').loc[frame.grid_id].reset_index()
    if not np.allclose(mu.reference_nn,frame.mu0,atol=1e-10,rtol=0): raise ValueError('Reference mismatch')
    frame['fold']=mu.outer_fold.to_numpy(); frame['pred']=mu.prediction.to_numpy()
    frame['error']=frame.pred-frame.mu0
    support=pd.read_csv(result/'gp_oof_support.csv').set_index('grid_id').loc[frame.grid_id]
    frame['support_failure']=~support.gev_valid_all_years.to_numpy(bool)
    selection=pd.read_csv(result/'spatial_selections.csv'); selection=selection[selection.coefficient.eq('mu0')]
    output.mkdir(parents=True); rows=[]; ranges=[]; decomposed=[]
    for row in selection.itertuples():
        fold=int(row.outer_fold); test_idx=frame.index[frame.fold.eq(fold)].to_numpy()
        candidates=frame.index[frame.fold.ne(fold)].to_numpy()
        train_idx=_buffered_training_indices(frame,candidates,test_idx,config['buffers_km']['mu0'],distance_mode=config['buffer_distance_mode'])
        train_idx=sample_indices(train_idx,config['max_train'],config['seed']+fold)
        if len(train_idx)!=row.n_retained_train: raise ValueError('Training membership/count mismatch')
        train=frame.loc[train_idx]; test=frame.loc[test_idx]
        cols=[] if row.predictors=='intercept' else row.predictors.split('+')
        dist,_=cKDTree(train[['x_km','y_km']]).query(test[['x_km','y_km']],k=1)
        item=dict(fold=fold,n_train=len(train),n_test=len(test),bias_C=float(test.error.mean()),
            rmse_C=float(np.sqrt(np.mean(test.error**2))),support_failure_grids=int(test.support_failure.sum()),
            reference_mu_mean=float(test.mu0.mean()),prediction_mu_mean=float(test.pred.mean()),
            train_mu_mean=float(train.mu0.mean()),longitude_mean=float(test.longitude_180.mean()),latitude_mean=float(test.latitude.mean()),
            nearest_train_projected_km_median=float(np.median(dist)),kernel=row.kernel,length_scale_km=float(row.fitted_length_scale_km),
            selected_predictors=cols,includes_elevation='elevation_m' in cols,includes_latitude=False)
        for c in cols:
            outside=(test[c]<train[c].min())|(test[c]>train[c].max())
            ranges.append(dict(fold=fold,predictor=c,train_min=float(train[c].min()),train_max=float(train[c].max()),
                test_min=float(test[c].min()),test_max=float(test[c].max()),fraction_outside_training_range=float(outside.mean())))
        if fold in refit_folds:
            print('Diagnostic selected-model refit, fold',fold,'n=',len(train),flush=True)
            start=time.perf_counter()
            with threadpool_limits(limits=threads):
                fit=fit_gp_with_covariates(train[['x_km','y_km']].to_numpy(),train[cols].to_numpy(),train.mu0.to_numpy(),cols,
                    kernel_name=row.kernel,nu=row.nu,n_restarts=config['n_restarts'],seed=config['seed']+fold,**config['gp_options'])
                pred=predict_gp_with_covariates(fit,test[['x_km','y_km']].to_numpy(),test[cols].to_numpy())
                design,_,_=_standardized_design(test[cols].to_numpy(),fit['covariate_center'],fit['covariate_scale'])
                mean=design@fit['beta']; correction=pred-mean
                # Independent unit/centering path: direct physical-coordinate kernel.
                cross=fit['kernel'](train[['x_km','y_km']].to_numpy(),test[['x_km','y_km']].to_numpy())
                direct=mean+cross.T@fit['alpha']
            item.update(refit_seconds=time.perf_counter()-start,optimizer_success=bool(fit['optimizer_success']),
                refit_kernel=str(fit['kernel']),prediction_replay_rmse_C=float(np.sqrt(np.mean((pred-test.pred.to_numpy())**2))),
                centering_invariance_max_abs_C=float(np.max(np.abs(direct-pred))),
                mean_component_bias_C=float(np.mean(mean-test.mu0.to_numpy())),spatial_correction_mean_C=float(correction.mean()),
                spatial_correction_abs_median_C=float(np.median(np.abs(correction))),
                regression_beta=fit['beta'].tolist())
            decomposed.extend(dict(grid_id=g,fold=fold,mean_component=float(m),spatial_correction=float(c),
                replay_prediction=float(p),saved_prediction=float(s),reference_nn=float(t))
                for g,m,c,p,s,t in zip(test.grid_id,mean,correction,pred,test.pred,test.mu0))
        rows.append(item)
        write(output/'progress.json',dict(completed_folds=len(rows),total=5,rows=rows))
    write(output/'report.json',dict(source=str(result),source_report_sha256=sha(result/'run_report.json'),
        coordinate_contract_passed=True,reference_alignment_passed=True,folds=rows,predictor_ranges=ranges,
        prediction_decomposition=decomposed,
        limitations=['Errors are relative to NN estimates, not climate truth.',
        'Post-hoc diagnosis, no held-out values used to repair predictions or select new settings.',
        'Only selected final fits replayed; no repeated FFS. No NN/model replacement.']))
    print('Saved',output/'report.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=ROOT)
    p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--refit-folds',type=int,nargs='*',default=[1,3]);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();diagnose(a.bundle,a.result,a.output,a.refit_folds,a.threads)
