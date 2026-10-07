"""Paired 17D/21D Cheng NN experiment with four balanced trend scenarios.

Generates a NEW 100k dataset (80k/10k/10k); trains three paired seeds using
the existing notebook implementation. Test is generated and sealed, never
loaded by training/reporting. No original dataset, checkpoint, or latest-run
pointer is replaced. Resume keeps complete runs and retries partial runs in
new attempt directories.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cheng_nn_simulation import (
    COEFFICIENT_NAMES, DATA_SCHEMA_VERSION, INPUT_FEATURE_NAMES, QUANTILE_LEVELS,
    ParameterRanges, SimulationConfig, _open_split_arrays, build_input_features,
    draw_coefficients, gev_inverse_cdf,
)
from compare_cheng_nn_inputs import FIXED, make_features, now, read_json
from compare_cheng_nn_lr_l2 import SEEDS, experiment_lock
from train_cheng_nn import (
    PROJECT_ROOT, atomic_array, atomic_csv, atomic_json, atomic_torch_save,
    export_predictions, file_sha256, load_notebook_components,
)

SCHEME = 'balanced_four_time_structures_v1'
SCENARIOS = ('M0', 'M_mu', 'M_sigma', 'M_mu_sigma')
DEFAULT_DATA = PROJECT_ROOT / 'data/simulated/cheng_nn_mixed_100k_20261006'
DEFAULT_OUTPUT = PROJECT_ROOT / 'results/cheng_nn_17d/mixed_sampling_20261006'
ARRAY_FILES = {
    'inputs': 'inputs.npy', 'targets': 'targets_standardized.npy',
    'coefficients': 'coefficients_original.npy',
    'location_scale': 'sample_location_scale.npy', 'annual_maxima': 'annual_maxima.npy',
}


def balanced_scenarios(n_samples, rng):
    if n_samples < 4 or n_samples % 4:
        raise ValueError('Each split must have a positive multiple of four cases.')
    labels = np.repeat(np.arange(4, dtype=np.int8), n_samples // 4)
    rng.shuffle(labels)
    return labels


def mixed_chunk(rng, labels, config, ranges):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, np.arange(4)).all():
        raise ValueError('Scenario IDs must be 0, 1, 2, or 3.')
    coefficients = draw_coefficients(rng, len(labels), ranges)
    coefficients[~np.isin(labels, [1, 3]), 1] = 0.0
    coefficients[~np.isin(labels, [2, 3]), 3] = 0.0
    mu0, beta_mu, eta0, beta_sigma, xi0 = coefficients.T
    t = config.centered_time[None, :]
    mu = mu0[:, None] + beta_mu[:, None] * t
    sigma = np.exp(eta0[:, None] + beta_sigma[:, None] * t)
    values = gev_inverse_cdf(rng.random((len(labels), config.n_years)),
                             mu, sigma, xi0[:, None])
    inputs, scaling = build_input_features(values)
    median, iqr = scaling.T
    targets = np.column_stack(((mu0 - median) / iqr, beta_mu / iqr,
                              eta0 - np.log(iqr), beta_sigma, xi0))
    return inputs.astype(np.float32), targets.astype(np.float32), coefficients, scaling, values


def check_hashes(directory, hashes):
    for name, expected in hashes.items():
        if file_sha256(Path(directory) / name) != expected:
            raise ValueError(f'Changed or damaged file: {Path(directory) / name}')


def generate_split(root, split, n_samples, seed, config, ranges):
    """Atomic publication; exact balance within EACH split, including test."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / split
    expected = dict(sampling_scheme=SCHEME, split=split, n_samples=n_samples,
                    seed=int(seed), years=config.years.tolist(),
                    parameter_ranges=json.loads(json.dumps(asdict(ranges))),
                    time_scale_years=config.time_scale_years,
                    generator_sha256=file_sha256(Path(__file__)))
    if final.exists():
        metadata = read_json(final / 'metadata.json')
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Existing split differs from requested design: {final}')
        check_hashes(final, metadata['files_sha256'])
        return metadata
    partial = root / f'.{split}.partial'
    attempt = 1
    while partial.exists():
        attempt += 1
        partial = root / f'.{split}.partial_attempt{attempt}'
    rng = np.random.default_rng(seed)
    labels = balanced_scenarios(n_samples, rng)
    partial.mkdir()
    arrays = _open_split_arrays(partial, n_samples, config.n_years)
    for start in range(0, n_samples, config.chunk_size):
        stop = min(start + config.chunk_size, n_samples)
        chunk = mixed_chunk(rng, labels[start:stop], config, ranges)
        for key, values in zip(ARRAY_FILES, chunk):
            arrays[key][start:stop] = values
    for array in arrays.values():
        array.flush()
    del array, arrays  # Close memmaps before renaming on Windows.
    atomic_array(partial / 'scenario_id.npy', labels)
    files = {p.name: file_sha256(p) for p in sorted(partial.glob('*.npy'))}
    metadata = dict(expected, data_schema_version=DATA_SCHEMA_VERSION,
                    created_utc=now(), reference_year=config.reference_year,
                    centered_time=config.centered_time.tolist(),
                    input_shape_per_sample=[17], input_columns=list(INPUT_FEATURE_NAMES),
                    input_quantile_levels=list(QUANTILE_LEVELS),
                    input_standardization='one whole-sequence median/IQR; no within-period scaling',
                    input_period_lengths=[16, 17, 17],
                    target_columns=list(COEFFICIENT_NAMES),
                    target_scale='sample median/IQR standardized GEV coefficients',
                    annual_maxima_shape_per_sample=[config.n_years],
                    annual_maxima_scale='original temperature scale, ordered by year',
                    scenario_names=list(SCENARIOS),
                    scenario_counts={name: n_samples // 4 for name in SCENARIOS},
                    scenario_labels_used_as_inputs=False,
                    temporal_sampling='independent annual maxima conditional on coefficient vector',
                    coefficient_relationships={'sigma0': 'exp(eta0)',
                        'beta_mu': 'active_mu * uniform(-0.5,0.5) * sigma0',
                        'beta_sigma': 'active_sigma * uniform(-0.2,0.2)'},
                    zero_mass_note='Balanced design, not inferred global scenario prevalence.',
                    files_sha256=files)
    atomic_json(partial / 'metadata.json', metadata)
    partial.replace(final)
    return metadata


def load_data(directory):
    """Use the unchanged notebook loader with an explicitly named new design."""
    ns = load_notebook_components()
    ns['DATA_DIR'] = Path(directory)
    ns['SAMPLING_SCHEME'] = SCHEME
    data = {}
    for split, size_key in [('train', 'N_TRAIN'), ('validation', 'N_VALIDATION')]:
        metadata = read_json(Path(directory) / split / 'metadata.json')
        if metadata['sampling_scheme'] != SCHEME or metadata['scenario_names'] != list(SCENARIOS):
            raise ValueError('Not the requested four-scenario mixture.')
        ns[size_key] = metadata['n_samples']
        data[split] = ns['load_split'](split)
        data[split]['scenario'] = np.load(Path(directory) / split / 'scenario_id.npy')
        if not np.array_equal(np.bincount(data[split]['scenario'], minlength=4),
                              np.repeat(metadata['n_samples'] // 4, 4)):
            raise ValueError('Scenario balance mismatch.')
        for name in ARRAY_FILES:
            if not np.isfinite(data[split][name]).all():
                raise ValueError(f'Nonfinite data: {split}/{name}')
    return ns, data


def prepare(output, data_dir, sizes=(80000, 10000, 10000), seed=20261006):
    output, data_dir = Path(output).resolve(), Path(data_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config, ranges = SimulationConfig(seed=seed), ParameterRanges()
    config.validate(); ranges.validate()
    children = np.random.SeedSequence(seed).spawn(3)
    metadata = {}
    for split, n, child in zip(('train', 'validation', 'test'), sizes, children):
        split_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        metadata[split] = generate_split(data_dir, split, n, split_seed, config, ranges)
        print(f'GENERATED {split}: {n:,}; each scenario={n//4:,}', flush=True)
    ns, data = load_data(data_dir)
    features = {}
    for dimension in (17, 21):
        features[str(dimension)] = {}
        for split, d in data.items():
            x = make_features(d['annual_maxima'], d['location_scale'], dimension)
            if dimension == 17:
                np.testing.assert_allclose(x, d['inputs'], rtol=1e-6, atol=1e-6)
            path = output / f'features_{dimension}_{split}.npy'
            atomic_array(path, x)
            features[str(dimension)][split] = dict(path=str(path), sha256=file_sha256(path))
    sources = [Path(__file__), PROJECT_ROOT / 'notebooks/cheng_NN.ipynb'] + [
        PROJECT_ROOT / 'src' / name for name in ('cheng_nn_simulation.py', 'cheng_nn_evaluation.py',
            'train_cheng_nn.py', 'compare_cheng_nn_inputs.py', 'compare_cheng_nn_lr_l2.py')]
    runs = [dict(dimension=dim, seed=s, status='pending',
                 path=str(output / f'input{dim}_seed{s}'))
            for i, s in enumerate(SEEDS) for dim in ((17, 21) if i % 2 == 0 else (21, 17))]
    manifest = dict(status='prepared', created_utc=now(), fixed=FIXED, runs=runs,
        data_directory=str(data_dir), scenario_names=list(SCENARIOS),
        split_sizes=dict(zip(('train','validation','test'), sizes)), simulation_seed=seed,
        dataset_files_sha256={s: metadata[s]['files_sha256'] for s in data},
        dataset_metadata_sha256={s: file_sha256(data_dir/s/'metadata.json') for s in data},
        features=features, source_hashes={str(p): file_sha256(p) for p in sources},
        checkpoint_selection='validation train-target-z coefficient MSE; no test access',
        test_role='generated and sealed; no evaluation or model selection',
        note='Scenario ID is NEVER a network input. No RL or formal-model replacement.')
    atomic_json(output / 'experiment.json', manifest)
    return manifest


def validate_manifest(manifest):
    for path, digest in manifest['source_hashes'].items():
        if file_sha256(path) != digest:
            raise ValueError(f'Source changed since experiment start: {path}')
    root = Path(manifest['data_directory'])
    for split, hashes in manifest['dataset_files_sha256'].items():
        check_hashes(root / split, hashes)
        if file_sha256(root/split/'metadata.json') != manifest['dataset_metadata_sha256'][split]:
            raise ValueError(f'Dataset metadata changed: {split}')
    for features in manifest['features'].values():
        for item in features.values():
            if file_sha256(item['path']) != item['sha256']:
                raise ValueError('Input features changed.')


def train_one(output, dimension, seed, run_dir, *, device='cuda'):
    manifest = read_json(output / 'experiment.json')
    validate_manifest(manifest)
    settings = manifest['fixed']
    ns, data = load_data(manifest['data_directory'])
    torch.set_num_threads(settings['threads'])
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; do not silently change benchmark device.')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    for split in data:
        data[split]['inputs'] = np.load(manifest['features'][str(dimension)][split]['path'], mmap_mode='r')
    ns['INPUT_FEATURE_NAMES'] = tuple(f'input_{i}' for i in range(dimension))
    ns['LEARNING_RATE'] = settings['learning_rate']
    scaler = ns['TargetScaler'].fit(data['train']['targets'])
    generator = torch.Generator().manual_seed(seed)
    loaders = {s: DataLoader(ns['ChengSimulationDataset'](d['inputs'], d['targets'], scaler),
        batch_size=settings['batch_size'], shuffle=s=='train', num_workers=0,
        pin_memory=device=='cuda', generator=generator if s=='train' else None) for s,d in data.items()}
    train_eval = DataLoader(loaders['train'].dataset, batch_size=2048, shuffle=False,
        num_workers=0, pin_memory=device=='cuda', generator=torch.Generator().manual_seed(seed))
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    architecture = [dimension, *settings['hidden_sizes'], 5]
    meta = dict(status='running', created_utc=now(), seed=seed, input_dimension=dimension,
        architecture=architecture, device=device, torch_version=str(torch.__version__),
        trainable_parameters=sum((a+1)*b for a,b in zip(architecture[:-1],architecture[1:])),
        coefficient_only=True, split_sizes={s:len(d['inputs']) for s,d in data.items()},
        dataset_metadata_sha256=manifest['dataset_metadata_sha256'],
        source_hashes=manifest['source_hashes'], best_epoch=None, **settings)

    def checkpoint(model, optimizer, scheduler, history, improved, stale):
        if improved:
            meta.update(best_epoch=history[-1]['epoch'], best_validation_loss=history[-1]['validation_loss'])
        meta.update(last_completed_epoch=history[-1]['epoch'], elapsed_seconds=time.perf_counter()-started)
        payload = dict(model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            target_scaler=scaler.state_dict(), coefficient_names=list(COEFFICIENT_NAMES),
            input_shape=[dimension], hidden_sizes=settings['hidden_sizes'], dropout_p=0.,
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
        model, history = ns['train_cheng_model'](loaders['train'], loaders['validation'],
            settings['max_epochs'], settings['patience'], device, epoch_callback=checkpoint,
            weight_decay=settings['weight_decay'], hidden_sizes=tuple(settings['hidden_sizes']),
            dropout_p=settings['dropout_p'], optimizer_name=settings['optimizer'],
            l1_lambda=0., train_eval_loader=train_eval)
        meta.update(status='evaluating', epochs_completed=len(history), training_seconds=time.perf_counter()-started)
        for split, d in data.items():
            meta.setdefault('evaluation', {})[split] = export_predictions(ns, model, scaler, d, split, run_dir, meta, device)
        meta.update(status='completed', completed_utc=now(), total_seconds=time.perf_counter()-started)
    except BaseException as exc:
        meta.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed', error=repr(exc))
        raise
    finally:
        atomic_json(run_dir/'run_metadata.json', meta)


def scenario_metrics(truth, prediction, labels, dimension, seed):
    errors = np.asarray(prediction) - np.asarray(truth)
    if errors.shape != truth.shape or not np.isfinite(errors).all():
        raise ValueError('Nonfinite/mismatched predictions; do not drop failed cases.')
    rows = []
    for scenario, name in enumerate(SCENARIOS):
        use = labels == scenario
        for j, coefficient in enumerate(COEFFICIENT_NAMES):
            e = errors[use, j]
            rows.append(dict(dimension=dimension, seed=seed, scenario=name,
                coefficient=coefficient, n_cases=int(use.sum()),
                RMSE=float(np.sqrt(np.mean(e**2))), bias=float(np.mean(e)),
                MAE=float(np.mean(np.abs(e)))))
    return rows


def validate_run(entry, manifest):
    path = Path(entry['path'])
    meta = read_json(path/'run_metadata.json')
    if (meta['status'] != 'completed' or meta['seed'] != entry['seed']
            or meta['input_dimension'] != entry['dimension']
            or meta['source_hashes'] != manifest['source_hashes']
            or meta['dataset_metadata_sha256'] != manifest['dataset_metadata_sha256']
            or any(meta.get(k) != v for k,v in manifest['fixed'].items())):
        raise ValueError('Completed run provenance/settings mismatch.')
    digest = file_sha256(path/'best.pt')
    if entry.get('checkpoint_sha256', digest) != digest:
        raise ValueError('Completed checkpoint changed.')
    if (path/'test').exists():
        raise ValueError('Unexpected test evaluation.')
    for split in ('train', 'validation'):
        marker = read_json(path/split/'prediction_metadata.json')
        if (marker['checkpoint_sha256'] != digest or
                marker['dataset_metadata_sha256'] != manifest['dataset_metadata_sha256'][split]):
            raise ValueError('Prediction provenance mismatch.')
    return meta


def summarize(output):
    manifest = read_json(output/'experiment.json')
    _, data = load_data(manifest['data_directory'])
    rows, grouped, validity = [], [], []
    for entry in manifest['runs']:
        if entry['status'] != 'completed':
            continue
        meta = validate_run(entry, manifest)
        path, dim, seed = Path(entry['path']), entry['dimension'], entry['seed']
        row = dict(dimension=dim, seed=seed, best_epoch=meta['best_epoch'],
            training_seconds=meta['training_seconds'], total_seconds=meta['total_seconds'])
        for split, d in data.items():
            metrics = pd.read_csv(path/split/'coefficient_metrics.csv')
            if not (metrics.n_used == metrics.n_total).all():
                raise ValueError('Metrics excluded cases.')
            z_rmse = metrics.loc[metrics.scale.eq('train_target_z'), 'RMSE'].to_numpy()
            row[f'{split}_scaled_RMSE'] = float(np.sqrt(np.mean(z_rmse**2)))
        row['gap_percent'] = 100*(row['validation_scaled_RMSE']/row['train_scaled_RMSE']-1)
        truth = np.asarray(data['validation']['coefficients'])
        pred = np.load(path/'validation/predictions_original.npy')
        grouped.extend(scenario_metrics(truth, pred, data['validation']['scenario'], dim, seed))
        for j, name in enumerate(COEFFICIENT_NAMES):
            row[name] = float(np.sqrt(np.mean((pred[:,j]-truth[:,j])**2)))
        rows.append(row)
        valid = pd.read_csv(path/'validation/gev_validity.csv')
        validity.append(dict(dimension=dim, seed=seed, n_cases=len(valid),
            invalid_parameters=int((~valid.parameters_valid_all_years).sum()),
            support_failed_cases=int((valid.n_support_violations>0).sum()),
            numerical_failed_cases=int((valid.n_support_numerical_failures>0).sum())))
    if not rows:
        return
    frame = pd.DataFrame(rows)
    atomic_csv(output/'rmse_per_seed.csv', frame)
    columns = ['validation_scaled_RMSE','train_scaled_RMSE','gap_percent',
               'training_seconds',*COEFFICIENT_NAMES]
    summary = frame.groupby('dimension').agg({c:['mean','std'] for c in columns}).reset_index()
    summary.columns = ['_'.join(c).rstrip('_') for c in summary.columns]
    summary.insert(1, 'completed_seeds', frame.groupby('dimension').size().to_numpy())
    atomic_csv(output/'comparison_summary.csv', summary)
    groups = pd.DataFrame(grouped)
    atomic_csv(output/'scenario_rmse_per_seed.csv', groups)
    group_summary = groups.groupby(['dimension','scenario','coefficient']).agg(
        n_cases=('n_cases','first'), RMSE_mean=('RMSE','mean'), RMSE_seed_SD=('RMSE','std'),
        bias_mean=('bias','mean'), MAE_mean=('MAE','mean')).reset_index()
    atomic_csv(output/'scenario_summary.csv', group_summary)
    atomic_csv(output/'gev_validity_summary.csv', pd.DataFrame(validity))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--train-one', type=int, choices=(17,21))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.train_one:
        train_one(output, args.train_one, args.seed, args.run_dir)
        return
    if args.report_only:
        summarize(output)
        return
    output.mkdir(parents=True, exist_ok=True)
    with experiment_lock(output):
        marker = output/'experiment.json'
        if marker.exists():
            if not args.resume:
                raise FileExistsError('Use --resume; existing experiment is never replaced.')
            manifest = read_json(marker)
            if Path(manifest['data_directory']) != args.data.resolve():
                raise ValueError('Resume data directory differs.')
            validate_manifest(manifest)
        else:
            manifest = prepare(output, args.data.resolve())
        if args.prepare_only:
            return
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; data preserved, no training started.')
        manifest.update(status='running', resumed_utc=now())
        atomic_json(marker, manifest)
        try:
            for entry in manifest['runs']:
                path = Path(entry['path'])
                if entry['status'] == 'completed':
                    validate_run(entry, manifest)
                    continue
                if path.exists():
                    old_marker = path/'run_metadata.json'
                    if old_marker.exists() and read_json(old_marker).get('status') == 'completed':
                        validate_run(entry, manifest)
                        entry.update(status='completed', checkpoint_sha256=file_sha256(path/'best.pt'))
                        atomic_json(marker, manifest)
                        continue
                    attempt = 2
                    while path.with_name(path.name+f'_attempt{attempt}').exists():
                        attempt += 1
                    path = path.with_name(path.name+f'_attempt{attempt}')
                    entry['path'] = str(path)
                entry.update(status='running', started_utc=now())
                atomic_json(marker, manifest)
                print(f"START {entry['dimension']}D seed={entry['seed']}", flush=True)
                with path.with_suffix('.log').open('x', encoding='utf-8') as log:
                    result = subprocess.run([sys.executable, '-u', str(Path(__file__)),
                        '--output', str(output), '--train-one', str(entry['dimension']),
                        '--seed', str(entry['seed']), '--run-dir', str(path)],
                        stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    entry['status'] = 'failed'
                    raise RuntimeError(f'Run failed; inspect {path.with_suffix(".log")}')
                validate_run(entry, manifest)
                entry.update(status='completed', completed_utc=now(), checkpoint_sha256=file_sha256(path/'best.pt'))
                atomic_json(marker, manifest)
                summarize(output)
                print(f"DONE {entry['dimension']}D seed={entry['seed']}", flush=True)
            manifest.update(status='completed', completed_utc=now())
        except BaseException as exc:
            manifest.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed', error=repr(exc))
            raise
        finally:
            atomic_json(marker, manifest)
        summarize(output)
        print('Six new paired models complete. Test remains sealed.', flush=True)


if __name__ == '__main__':
    main()
