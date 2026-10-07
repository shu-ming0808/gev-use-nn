"""Paired 17/21/50-input experiment. Never loads test or publishes a model.

Use the notebook's unchanged optimizer, loss, target transform and stopping
rules. Only the input representation (and first-layer width) changes.
Completed 17D baseline checkpoints are reused after provenance checks.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cheng_nn_simulation import COEFFICIENT_NAMES, QUANTILE_LEVELS, build_input_features
from cheng_nn_evaluation import check_gev_predictions, gev_parameter_curves
from train_cheng_nn import (PROJECT_ROOT, atomic_array, atomic_csv, atomic_json,
                            atomic_torch_save, export_predictions, file_sha256,
                            load_notebook_components)
from compare_cheng_nn_lr_l2 import SEEDS, BASELINE, validate_run, experiment_lock

DEFAULT_BASELINE = PROJECT_ROOT / 'results/cheng_nn_17d/lr_l2_small_100k_20261003'
DEFAULT_DATA = PROJECT_ROOT / 'data/simulated/cheng_nn_17d'
DEFAULT_OUT = PROJECT_ROOT / 'results/cheng_nn_17d/input_comparison_20261004'
FIXED = dict(hidden_sizes=[128, 128, 64], optimizer='Adam', learning_rate=1e-3,
             weight_decay=1e-4, dropout_p=0., batch_size=128, max_epochs=300,
             patience=20, threads=4, test_used=False, return_periods=[])


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def make_features(values, location_scale, dimension):
    """Standardize the entire sequence ONCE, using the original saved scaling."""
    values = np.asarray(values, dtype=float)
    scaling = np.asarray(location_scale, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 50 or scaling.shape != (len(values), 2)
            or not np.isfinite(values).all() or not np.isfinite(scaling).all()
            or np.any(scaling[:, 1] <= 0)):
        raise ValueError('Need finite 50-year sequences and positive saved IQRs.')
    if dimension not in (17, 21, 50):
        raise ValueError('Input dimension must be 17, 21 or 50.')
    z = (values - scaling[:, :1]) / scaling[:, 1:2]
    if dimension == 50:
        return z.astype(np.float32)
    features = [np.quantile(z, QUANTILE_LEVELS, axis=1).T]
    segments = 3 if dimension == 17 else 5
    # Match the original 17D convention: 16,17,17 years; 21D: five x 10.
    boundaries = np.linspace(0, 50, segments + 1).astype(int)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        q = np.quantile(z[:, start:stop], [.25, .5, .75], axis=1)
        features.extend([q[1, :, None], (q[2] - q[0])[:, None]])
    return np.concatenate(features, axis=1).astype(np.float32)


def load_data(data_dir):
    ns = load_notebook_components()
    ns['DATA_DIR'] = Path(data_dir)
    data = {}
    for split, n in [('train', 80000), ('validation', 10000)]:
        ns['N_TRAIN' if split == 'train' else 'N_VALIDATION'] = n
        data[split] = ns['load_split'](split)
        for name in ('inputs', 'targets', 'coefficients', 'location_scale', 'annual_maxima'):
            if not np.isfinite(data[split][name]).all():
                raise ValueError(f'Nonfinite source array: {split}/{name}')
    return ns, data


def prepare(output, data_dir, baseline):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ns, data = load_data(data_dir)
    original = read_json(baseline / 'experiment.json')
    if original['status'] != 'completed':
        raise ValueError('Original comparison is incomplete.')
    for split in data:
        for name, digest in original['dataset_files_sha256'][split].items():
            if file_sha256(data_dir / split / name) != digest:
                raise ValueError(f'Baseline data hash mismatch: {split}/{name}')
    runs = []
    for entry in original['runs']:
        if entry['case'] == BASELINE:
            path, meta, *_ = validate_run(baseline, entry, original)
            runs.append(dict(dimension=17, seed=entry['seed'], path=str(path),
                             status='completed', reused=True,
                             checkpoint_sha256=file_sha256(path/'best.pt')))
    if {r['seed'] for r in runs} != set(SEEDS):
        raise ValueError('Three paired baseline seeds are required.')
    feature_paths = {}
    for dim in (17, 21, 50):
        feature_paths[str(dim)] = {}
        for split, d in data.items():
            x = make_features(d['annual_maxima'], d['location_scale'], dim)
            if dim == 17:
                np.testing.assert_allclose(x, d['inputs'], rtol=1e-6, atol=1e-6)
                # Exact existing features are the authoritative baseline.
                x = np.asarray(d['inputs'])
            path = output / f'features_{dim}_{split}.npy'
            atomic_array(path, x)
            feature_paths[str(dim)][split] = dict(path=str(path), sha256=file_sha256(path))
    for i, seed in enumerate(SEEDS):
        for dim in ((21, 50) if i % 2 == 0 else (50, 21)):
            runs.append(dict(dimension=dim, seed=seed, path=str(output/f'input{dim}_seed{seed}'),
                             status='pending', reused=False))
    manifest = dict(status='prepared', created_utc=now(), fixed=FIXED,
                    data_directory=str(data_dir), baseline_directory=str(baseline),
                    dataset_files_sha256=original['dataset_files_sha256'],
                    dataset_metadata_sha256=original['dataset_metadata_sha256'],
                    features=feature_paths, runs=runs,
                    source_hashes={str(p): file_sha256(p) for p in [Path(__file__),
                        PROJECT_ROOT/'notebooks/cheng_NN.ipynb', PROJECT_ROOT/'src/train_cheng_nn.py',
                        PROJECT_ROOT/'src/cheng_nn_simulation.py', PROJECT_ROOT/'src/cheng_nn_evaluation.py']},
                    selection='mean over three seeds of validation train-target-z RMSE',
                    segment_lengths={'17': [16, 17, 17], '21': [10]*5},
                    note='Same cases and target scaling. No test access. No model publication.')
    atomic_json(output/'experiment.json', manifest)
    return manifest


def train_one(output, dimension, seed, run_dir):
    manifest = read_json(output/'experiment.json')
    ns, data = load_data(manifest['data_directory'])
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the matched training protocol.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    for split in data:
        feature = manifest['features'][str(dimension)][split]
        if file_sha256(feature['path']) != feature['sha256']:
            raise ValueError('Feature hash mismatch.')
        data[split]['inputs'] = np.load(feature['path'], mmap_mode='r')
    # Definitions from the notebook share this isolated namespace. The model
    # already derives its first layer from len(INPUT_FEATURE_NAMES).
    ns['INPUT_FEATURE_NAMES'] = tuple(f'input_{i}' for i in range(dimension))
    ns['LEARNING_RATE'] = .001
    scaler = ns['TargetScaler'].fit(data['train']['targets'])
    generator = torch.Generator().manual_seed(seed)
    loaders = {s: DataLoader(ns['ChengSimulationDataset'](d['inputs'], d['targets'], scaler),
                            batch_size=128, shuffle=s == 'train', num_workers=0, pin_memory=True,
                            generator=generator if s == 'train' else None) for s, d in data.items()}
    train_eval = DataLoader(loaders['train'].dataset, batch_size=2048, shuffle=False,
                           num_workers=0, pin_memory=True,
                           generator=torch.Generator().manual_seed(seed))
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    architecture = [dimension, 128, 128, 64, 5]
    meta = dict(status='running', created_utc=now(), seed=seed, input_dimension=dimension,
                architecture=architecture, device='cuda', torch_version=str(torch.__version__),
                trainable_parameters=sum((a+1)*b for a,b in zip(architecture[:-1], architecture[1:])),
                coefficient_only=True, split_sizes={'train':80000,'validation':10000},
                dataset_metadata_sha256=manifest['dataset_metadata_sha256'],
                source_hashes=manifest['source_hashes'], best_epoch=None, **FIXED)
    def checkpoint(model, optimizer, scheduler, history, improved, stale):
        if improved:
            meta['best_epoch'] = history[-1]['epoch']
            meta['best_validation_loss'] = history[-1]['validation_loss']
        meta.update(last_completed_epoch=history[-1]['epoch'], elapsed_seconds=time.perf_counter()-started)
        payload = dict(model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                       target_scaler=scaler.state_dict(), coefficient_names=list(COEFFICIENT_NAMES),
                       input_shape=[dimension], hidden_sizes=[128,128,64], dropout_p=0.,
                       architecture=architecture, simulation_metadata=data['train']['metadata'],
                       input_representation=f'{dimension}D', source_hashes=manifest['source_hashes'],
                       epoch=history[-1]['epoch'], best_epoch=meta['best_epoch'],
                       optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict())
        if improved:
            atomic_torch_save(run_dir/'best.pt', payload)
        atomic_torch_save(run_dir/'latest.pt', payload)
        atomic_csv(run_dir/'training_history.csv', pd.DataFrame(history))
        atomic_json(run_dir/'run_metadata.json', meta)
    atomic_json(run_dir/'run_metadata.json', meta)
    try:
        model, history = ns['train_cheng_model'](loaders['train'],loaders['validation'],300,20,'cuda',
            epoch_callback=checkpoint,weight_decay=1e-4,hidden_sizes=(128,128,64),dropout_p=0.,
            optimizer_name='Adam',l1_lambda=0.,train_eval_loader=train_eval)
        meta.update(status='evaluating', epochs_completed=len(history),training_seconds=time.perf_counter()-started)
        for split in data:
            meta.setdefault('evaluation',{})[split] = export_predictions(ns,model,scaler,data[split],split,run_dir,meta,'cuda')
        meta.update(status='completed',completed_utc=now(),total_seconds=time.perf_counter()-started)
    except BaseException as exc:
        meta.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed', error=repr(exc))
        raise
    finally:
        atomic_json(run_dir/'run_metadata.json',meta)


def group_definitions(coefficients):
    mu, bm, eta, bs, xi = np.asarray(coefficients).T
    sigma = np.exp(eta)
    # Fixed, interpretable bins set before inspecting errors; no test-driven cuts.
    return {
        'sigma0': (sigma, [.2, .5, 1, 2, 4, 8]),
        'beta_mu_over_sigma0': (bm/sigma, [-.5, -.2, -.05, .05, .2, .5]),
        'beta_sigma': (bs, [-.2, -.1, -.03, .03, .1, .2]),
        'xi0': (xi, [-.4, -.2, 0, .2, .4]),
    }


def binned_metrics(truth, prediction, z_errors, dimension, seed):
    errors = np.asarray(prediction)-np.asarray(truth)
    if not np.isfinite(errors).all() or not np.isfinite(z_errors).all():
        raise ValueError('Nonfinite errors must not be silently omitted.')
    rows = []
    for axis, (values, edges) in group_definitions(truth).items():
        if np.any(values < edges[0]-1e-10) or np.any(values > edges[-1]+1e-10):
            raise ValueError(f'Cases outside prespecified bins: {axis}')
        group = np.searchsorted(edges[1:-1], values, side='right')
        for g in range(len(edges)-1):
            use = group == g
            for j, coefficient in enumerate(COEFFICIENT_NAMES):
                e = errors[use,j]
                ze = z_errors[use,j]
                rows.append(dict(dimension=dimension,seed=seed,axis=axis,group=g,
                    lower=edges[g],upper=edges[g+1],upper_inclusive=g==len(edges)-2,
                    coefficient=coefficient,n_cases=int(use.sum()),
                    RMSE=float(np.sqrt(np.mean(e**2))) if len(e) else np.nan,
                    z_RMSE=float(np.sqrt(np.mean(ze**2))) if len(e) else np.nan,
                    bias=float(np.mean(e)) if len(e) else np.nan))
    return rows


def summarize(output):
    manifest = read_json(output/'experiment.json')
    ns, data = load_data(manifest['data_directory'])
    scaler = ns['TargetScaler'].fit(data['train']['targets'])
    truth = np.asarray(data['validation']['coefficients'])
    targets = np.asarray(data['validation']['targets'])
    rows, groups, valid_rows, failures, distributions = [],[],[],[],[]
    for entry in manifest['runs']:
        if entry['status'] != 'completed':
            continue
        dim, seed, path = entry['dimension'],entry['seed'],Path(entry['path'])
        meta = read_json(path/'run_metadata.json')
        if meta['status'] != 'completed' or meta['seed'] != seed:
            raise ValueError('Incomplete or mismatched run.')
        if (path/'test').exists() or 'published_model' in meta:
            raise ValueError('Unexpected test evaluation or publication.')
        for split in ('train','validation'):
            marker=read_json(path/split/'prediction_metadata.json')
            if marker['checkpoint_sha256'] != file_sha256(path/'best.pt'):
                raise ValueError('Prediction/checkpoint mismatch.')
        predicted = np.load(path/'validation/predictions_original.npy')
        standardized = np.load(path/'validation/predictions_standardized.npy')
        if predicted.shape != truth.shape or not np.isfinite(predicted).all():
            raise ValueError('Invalid prediction size/values; no case exclusion allowed.')
        ze = scaler.transform(standardized).astype(float)-scaler.transform(targets).astype(float)
        train_metrics=pd.read_csv(path/'train/coefficient_metrics.csv')
        val_metrics=pd.read_csv(path/'validation/coefficient_metrics.csv')
        train_z = train_metrics.loc[train_metrics.scale.eq('train_target_z'),'RMSE'].to_numpy()
        validation_z = val_metrics.loc[val_metrics.scale.eq('train_target_z'),'RMSE'].to_numpy()
        tr=float(np.sqrt(np.mean(train_z**2))); va=float(np.sqrt(np.mean(validation_z**2)))
        np.testing.assert_allclose(va,np.sqrt(np.mean(ze**2)),rtol=1e-6)
        base=dict(dimension=dim,seed=seed,train_scaled_RMSE=tr,validation_scaled_RMSE=va,
                  gap_percent=100*(va/tr-1),best_epoch=meta['best_epoch'],
                  trainable_parameters=meta['trainable_parameters'],training_seconds=meta['training_seconds'])
        error=predicted-truth
        for j,name in enumerate(COEFFICIENT_NAMES):
            rmse=float(np.sqrt(np.mean(error[:,j]**2)))
            recorded=val_metrics.loc[(val_metrics.scale=='original') & (val_metrics.coefficient==name),'RMSE'].item()
            np.testing.assert_allclose(rmse,recorded,rtol=1e-10)
            base[name]=rmse
            q=np.quantile(error[:,j],[.05,.25,.5,.75,.95])
            distributions.append(dict(dimension=dim,seed=seed,coefficient=name,n_cases=len(error),
                 RMSE=rmse,bias=float(error[:,j].mean()),MAE=float(np.abs(error[:,j]).mean()),
                 **dict(zip(['p05','q25','median','q75','p95'],q))))
        rows.append(base)
        groups.extend(binned_metrics(truth,predicted,ze,dim,seed))
        cfg=ns['SIMULATION_CONFIG']
        validity=check_gev_predictions(predicted,data['validation']['annual_maxima'],cfg.years,
                       reference_year=cfg.reference_year,time_scale_years=cfg.time_scale_years)
        saved=pd.read_csv(path/'validation/gev_validity.csv')
        np.testing.assert_array_equal(validity.gev_valid_all_years,saved.gev_valid_all_years)
        valid_rows.append(dict(dimension=dim,seed=seed,n_cases=len(truth),
            invalid_parameters=int((~validity.parameters_valid_all_years).sum()),
            support_failed_cases=int((validity.n_support_violations>0).sum()),
            support_failed_years=int(validity.n_support_violations.sum()),
            numerical_failed_cases=int((validity.n_support_numerical_failures>0).sum()),
            compatible_percent=100*float(validity.gev_valid_all_years.mean()),
            xi_outside_training_range=int(validity.xi_outside_training_range.sum())))
        mu,sigma,xi,_=gev_parameter_curves(predicted,cfg.years,reference_year=cfg.reference_year)
        margin=1+xi*(np.asarray(data['validation']['annual_maxima'])-mu)/sigma
        for index in np.where(~validity.gev_valid_all_years)[0]:
            for year_idx in np.where((margin[index]<=0)|(~np.isfinite(margin[index])))[0]:
                f=dict(dimension=dim,seed=seed,sample_index=int(index),year=int(cfg.years[year_idx]),
                       observed=float(data['validation']['annual_maxima'][index,year_idx]),
                       support_margin=float(margin[index,year_idx]))
                f.update({f'true_{n}':float(truth[index,j]) for j,n in enumerate(COEFFICIENT_NAMES)})
                f.update({f'pred_{n}':float(predicted[index,j]) for j,n in enumerate(COEFFICIENT_NAMES)})
                failures.append(f)
    frame=pd.DataFrame(rows)
    atomic_csv(output/'rmse_per_seed.csv',frame)
    summary=frame.groupby('dimension').agg({c:['mean','std'] for c in
        ['validation_scaled_RMSE','train_scaled_RMSE','gap_percent',*COEFFICIENT_NAMES]}).reset_index()
    summary.columns=['_'.join(x).rstrip('_') for x in summary.columns]
    atomic_csv(output/'comparison_summary.csv',summary)
    atomic_csv(output/'grouped_rmse_per_seed.csv',pd.DataFrame(groups))
    atomic_csv(output/'residual_distribution.csv',pd.DataFrame(distributions))
    atomic_csv(output/'gev_validity_summary.csv',pd.DataFrame(valid_rows))
    atomic_csv(output/'support_failure_cases.csv',pd.DataFrame(failures))
    return frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--baseline',type=Path,default=DEFAULT_BASELINE)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--train-one',type=int,choices=(21,50))
    parser.add_argument('--seed',type=int)
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    output=args.output.resolve()
    if args.train_one:
        train_one(output,args.train_one,args.seed,args.run_dir)
        return
    if args.report_only:
        summarize(output)
        return
    output.mkdir(parents=True,exist_ok=True)
    with experiment_lock(output):
        if (output/'experiment.json').exists():
            if not args.resume:
                raise FileExistsError('Use --resume; existing results are never overwritten.')
            manifest=read_json(output/'experiment.json')
            for path,digest in manifest['source_hashes'].items():
                if file_sha256(path)!=digest:
                    raise ValueError(f'Source changed since study start: {path}')
        else:
            manifest=prepare(output,args.data.resolve(),args.baseline.resolve())
        manifest['status']='running'
        atomic_json(output/'experiment.json',manifest)
        summarize(output)
        for entry in manifest['runs']:
            if entry['status']=='completed':
                continue
            path=Path(entry['path'])
            if path.exists():
                # Recover completed work after a coordinator interruption.
                marker=path/'run_metadata.json'
                if marker.exists() and read_json(marker).get('status')=='completed':
                    entry.update(status='completed',checkpoint_sha256=file_sha256(path/'best.pt'))
                    atomic_json(output/'experiment.json',manifest)
                    continue
                attempt=2
                while path.with_name(path.name+f'_attempt{attempt}').exists():
                    attempt+=1
                path=path.with_name(path.name+f'_attempt{attempt}')
                entry['path']=str(path)
            entry.update(status='running',started_utc=now())
            atomic_json(output/'experiment.json',manifest)
            print(f"START {entry['dimension']}D seed={entry['seed']}",flush=True)
            with path.with_suffix('.log').open('x',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,'-u',str(Path(__file__)), '--output',str(output),
                      '--train-one',str(entry['dimension']),'--seed',str(entry['seed']),
                      '--run-dir',str(path)],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                entry['status']='failed'; manifest['status']='failed'
                atomic_json(output/'experiment.json',manifest)
                raise RuntimeError(f"Run failed; inspect {path.with_suffix('.log')}")
            entry.update(status='completed',completed_utc=now(),checkpoint_sha256=file_sha256(path/'best.pt'))
            atomic_json(output/'experiment.json',manifest)
            summarize(output)
            print(f"DONE {entry['dimension']}D seed={entry['seed']}",flush=True)
        manifest.update(status='completed',completed_utc=now())
        atomic_json(output/'experiment.json',manifest)
        summarize(output)
        print('All nine paired models complete (three reused, six new).',flush=True)


if __name__=='__main__':
    main()
