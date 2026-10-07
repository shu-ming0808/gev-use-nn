"""Frozen 21D mixed-100k checkpoint versus five-coefficient MLE on all test cases.

No training, model selection, RL evaluation or scenario-oracle fitting. Outputs
are separate from historical validation-only experiments. Completed MLE cases
are checkpointed atomically and reused on resume after provenance validation.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from cheng_nn_diagnostics import load_diagnostic_split
from cheng_nn_evaluation import check_gev_predictions
from cheng_nn_mle import fit_time_varying_gev
from cheng_nn_simulation import COEFFICIENT_NAMES, ParameterRanges
from compare_cheng_nn_inputs import make_features
from compare_cheng_nn_mixed_sampling import validate_manifest, validate_run
from train_cheng_nn import (PROJECT_ROOT, atomic_array, atomic_csv, atomic_json,
                            file_sha256, load_notebook_components)

EXPERIMENT = PROJECT_ROOT / 'results/cheng_nn_17d/mixed_sampling_20261006'
DEFAULT_OUTPUT = PROJECT_ROOT / 'results/cheng_nn_21d/test_mle_100k_20261007'
SEED = 20260929


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def fit_case(index, observations, centered_time, ranges):
    """Data-only fitting; the worker has no truth, scenario ID or NN estimate."""
    first = fit_time_varying_gev(observations, centered_time, ranges=ranges,
                                domain='training_domain', maxiter=600)
    seconds = first['seconds']
    retry = not first['success'] or first['lower_nonconverged_nll_gap'] > 1e-4
    result = first
    if retry:
        second = fit_time_varying_gev(observations, centered_time, ranges=ranges,
                                     domain='training_domain', maxiter=1800)
        seconds += second['seconds']
        if second['success'] and (not first['success'] or second['nll'] < first['nll']):
            result = second
    coefficients = result.pop('coefficients')
    return dict(sample_index=int(index), **{**result, 'seconds': seconds},
                retried=bool(retry), **dict(zip(COEFFICIENT_NAMES, coefficients)))


def validate_fit_rows(frame, n_cases):
    if frame.empty:
        return
    index = frame.sample_index.to_numpy()
    if (len(np.unique(index)) != len(index) or (index < 0).any()
            or (index >= n_cases).any() or not np.equal(index, index.astype(int)).all()
            or not frame.domain.eq('training_domain').all()):
        raise ValueError('Invalid, duplicated or out-of-range MLE case identifiers.')
    if not frame.success.isin([True, False]).all():
        raise ValueError('MLE success flags must be Boolean.')
    success = frame.success.to_numpy(bool)
    if not np.isfinite(frame.loc[success, list(COEFFICIENT_NAMES)].to_numpy(float)).all():
        raise ValueError('Successful fits must contain finite coefficients.')


def paired_metrics(truth, prediction, fitted, success):
    """Same rows for both methods. Non-support NN cases stay in coefficient RMSE."""
    truth, prediction, fitted = map(lambda x: np.asarray(x, float),
                                    (truth, prediction, fitted))
    if (truth.shape != prediction.shape or fitted.shape != truth.shape
            or truth.ndim != 2 or truth.shape[1] != 5
            or not np.isfinite(truth).all() or not np.isfinite(prediction).all()):
        raise ValueError('Need aligned finite truth/NN arrays, each with five coefficients.')
    use = np.asarray(success, bool) & np.isfinite(fitted).all(axis=1)
    if use.shape != (len(truth),) or not use.any():
        raise ValueError('No common successful cases.')
    nn_rmse = np.sqrt(np.mean((prediction[use] - truth[use])**2, axis=0))
    mle_rmse = np.sqrt(np.mean((fitted[use] - truth[use])**2, axis=0))
    improvement = np.divide(nn_rmse, mle_rmse, out=np.full(5, np.nan), where=mle_rmse > 0)
    return pd.DataFrame(dict(coefficient=COEFFICIENT_NAMES,
        n_requested=len(truth), n_paired=int(use.sum()), n_excluded=int((~use).sum()),
        MLE_RMSE=mle_rmse, NN_RMSE=nn_rmse,
        NN_improvement_percent=100 * (1 - improvement)))


def check_sequence_separation(data_directory, test_values):
    """Exact duplicate audit only. Generation seeds and hashes remain recorded."""
    held_out = {hashlib.sha256(np.asarray(row, np.float64).tobytes()).digest()
                for row in test_values}
    if len(held_out) != len(test_values):
        raise ValueError('Duplicate test sequences.')
    for split in ('train', 'validation'):
        values = np.load(data_directory / split / 'annual_maxima.npy', mmap_mode='r')
        if any(hashlib.sha256(np.asarray(row, np.float64).tobytes()).digest() in held_out
               for row in values):
            raise ValueError(f'Test sequence overlap with {split}.')


def prepare(output, n_jobs):
    manifest = read_json(EXPERIMENT / 'experiment.json')
    validate_manifest(manifest)
    entry, = [r for r in manifest['runs'] if r['dimension'] == 21 and r['seed'] == SEED]
    meta = validate_run(entry, manifest)
    data_dir = Path(manifest['data_directory'])
    test_meta = read_json(data_dir / 'test/metadata.json')
    if (manifest['split_sizes'] != dict(train=80000, validation=10000, test=10000)
            or test_meta['n_samples'] != 10000
            or test_meta['sampling_scheme'] != 'balanced_four_time_structures_v1'
            or list(test_meta['scenario_counts'].values()) != [2500]*4):
        raise ValueError('Expected mixed 100k dataset with balanced 10k test split.')
    for filename, digest in test_meta['files_sha256'].items():
        if file_sha256(data_dir / 'test' / filename) != digest:
            raise ValueError(f'Test file changed: {filename}')
    fixed = dict(
        experiment_directory=str(EXPERIMENT), data_directory=str(data_dir),
        run_directory=entry['path'], checkpoint_sha256=entry['checkpoint_sha256'],
        seed=SEED, dimension=21, architecture=[21, 128, 128, 64, 5],
        training_settings={k: v for k, v in manifest['fixed'].items() if k != 'test_used'},
        split_sizes=manifest['split_sizes'], best_epoch=meta['best_epoch'],
        checkpoint_selection='Existing best validation checkpoint, frozen before test access',
        dataset_test_metadata_sha256=file_sha256(data_dir / 'test/metadata.json'),
        dataset_test_files_sha256=test_meta['files_sha256'], n_requested=10000,
        scenario_counts=test_meta['scenario_counts'], test_used=True, nn_retrained=False,
        coefficient_names=list(COEFFICIENT_NAMES), return_periods=[],
        mle_domain='training_domain', parameter_ranges=test_meta['parameter_ranges'],
        mle_model='mu and log-sigma linear in centered time; xi constant; five free coefficients',
        mle_initialization='Four deterministic data-only starts; no truth, NN or scenario labels',
        mle_maxiter=600, mle_retry_maxiter=1800,
        mle_retry_rule='No successful fit or a lower nonconverged feasible NLL candidate',
        mle_global_optimum_guaranteed=False,
        failed_fit_policy='Report failures; paired metrics use the same successful cases',
        nn_support_policy='Report support failures; do not remove them from coefficient RMSE',
        statistical_tests=[], training_source_hashes=manifest['source_hashes'],
        benchmark_source_hashes={str(p): file_sha256(p) for p in
            (Path(__file__), PROJECT_ROOT/'src/cheng_nn_mle.py',
             PROJECT_ROOT/'src/cheng_nn_diagnostics.py')})
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        if protocol['fixed'] != fixed:
            raise ValueError('Resume provenance mismatch; do not mix benchmark versions.')
    else:
        if any(output.iterdir()):
            raise ValueError('Output is nonempty without its protocol.')
        protocol = dict(status='prepared', created_utc=now(), fixed=fixed, n_jobs=n_jobs,
                        fit_wall_seconds=0.0)
        atomic_json(protocol_path, protocol)  # Freeze design before computing test errors.
    data = load_diagnostic_split(data_dir, 'test', test_meta['files_sha256'])
    scenario = np.load(data_dir / 'test/scenario_id.npy', allow_pickle=False)
    if not np.array_equal(np.bincount(scenario, minlength=4), [2500]*4):
        raise ValueError('Unexpected test scenario allocation.')
    for name in ('annual_maxima', 'coefficients', 'location_scale'):
        if not np.isfinite(data[name]).all():
            raise ValueError(f'Nonfinite test array: {name}')
    check_sequence_separation(data_dir, data['annual_maxima'])
    return protocol, data, scenario


def predict(protocol, data, output):
    import torch
    torch.set_num_threads(4)
    fixed = protocol['fixed']
    marker = output / 'nn_prediction_metadata.json'
    if marker.exists():
        record = read_json(marker)
        if (record['checkpoint_sha256'] != fixed['checkpoint_sha256'] or
                record['test_metadata_sha256'] != fixed['dataset_test_metadata_sha256'] or
                file_sha256(output/'nn_predictions_original.npy') != record['prediction_sha256']):
            raise ValueError('Saved NN prediction provenance mismatch.')
        return np.load(output/'nn_predictions_original.npy', allow_pickle=False)
    with redirect_stdout(io.StringIO()):
        ns = load_notebook_components()
    payload = torch.load(Path(fixed['run_directory'])/'best.pt', map_location='cpu', weights_only=False)
    if (payload['architecture'] != fixed['architecture'] or
            payload['coefficient_names'] != list(COEFFICIENT_NAMES)):
        raise ValueError('Checkpoint architecture or output order mismatch.')
    ns['INPUT_FEATURE_NAMES'] = tuple(f'input_{i}' for i in range(21))
    model = ns['ChengGEVNet'](hidden_sizes=(128, 128, 64), dropout_p=0.)
    model.load_state_dict(payload['model_state'])
    model.eval()
    scaler = ns['TargetScaler'](**payload['target_scaler'])
    started = time.perf_counter()
    features = make_features(data['annual_maxima'], data['location_scale'], 21)
    standardized = ns['predict_standardized_coefficients'](model, features, scaler, 'cpu', batch_size=2048)
    prediction = ns['inverse_sample_standardization'](standardized, data['location_scale'])
    seconds = time.perf_counter() - started
    if prediction.shape != (10000, 5) or not np.isfinite(prediction).all():
        raise ValueError('Invalid NN prediction shape or nonfinite coefficients.')
    atomic_array(output/'nn_predictions_original.npy', prediction)
    atomic_json(marker, dict(checkpoint_sha256=fixed['checkpoint_sha256'],
        test_metadata_sha256=fixed['dataset_test_metadata_sha256'],
        prediction_sha256=file_sha256(output/'nn_predictions_original.npy'),
        device='cpu', threads=4, n_cases=10000, seconds_including_features_inverse=seconds))
    return prediction


def summarize(output, protocol, data, scenario, prediction, fits):
    fits = fits.sort_values('sample_index').reset_index(drop=True)
    if not np.array_equal(fits.sample_index.to_numpy(), np.arange(10000)):
        raise ValueError('All 10,000 requested MLE attempts must be present.')
    truth = np.asarray(data['coefficients'])
    estimated = fits[list(COEFFICIENT_NAMES)].to_numpy(float)
    success = fits.success.to_numpy(bool)
    rows = paired_metrics(truth, prediction, estimated, success)
    atomic_csv(output/'comparison.csv', rows)
    groups = []
    for group, name in enumerate(data['metadata']['scenario_names']):
        use = scenario == group
        part = paired_metrics(truth[use], prediction[use], estimated[use], success[use])
        part.insert(0, 'scenario', name)
        groups.append(part)
    atomic_csv(output/'comparison_by_scenario.csv', pd.concat(groups, ignore_index=True))
    quality = []
    meta = data['metadata']
    for name, coef in [('NN', prediction), ('MLE', estimated)]:
        valid = check_gev_predictions(coef, data['annual_maxima'], np.asarray(meta['years']),
            reference_year=meta['reference_year'], time_scale_years=meta['time_scale_years'])
        atomic_csv(output/f'{name.lower()}_support.csv', valid)
        quality.append(dict(model=name, n_cases=len(coef),
            successful_cases=int(success.sum()) if name == 'MLE' else len(coef),
            invalid_parameter_cases=int((~valid.parameters_valid_all_years).sum()),
            support_failed_cases=int((~valid.gev_valid_all_years).sum()),
            on_boundary=int(fits.on_boundary.sum()) if name == 'MLE' else None,
            lower_nonconverged_candidates=int((fits.lower_nonconverged_nll_gap > 1e-4).sum()) if name == 'MLE' else None))
    atomic_csv(output/'fit_quality.csv', pd.DataFrame(quality))
    nn_full = pd.DataFrame(dict(coefficient=COEFFICIENT_NAMES, n_cases=len(truth),
        RMSE=np.sqrt(np.mean((prediction-truth)**2, axis=0))))
    atomic_csv(output/'nn_all_test_rmse.csv', nn_full)
    protocol.update(status='completed', completed_utc=now(), n_mle_success=int(success.sum()),
                    n_mle_failed=int((~success).sum()), n_retried=int(fits.retried.sum()),
                    comparison_sha256=file_sha256(output/'comparison.csv'))
    atomic_json(output/'protocol.json', protocol)
    print(rows.to_string(index=False), flush=True)
    print(pd.DataFrame(quality).to_string(index=False), flush=True)


def run(args):
    output = Path(args.output_directory).resolve()
    protocol, data, scenario = prepare(output, args.n_jobs)
    prediction = predict(protocol, data, output)
    fit_path = output/'mle_fits.csv'
    fits = pd.read_csv(fit_path) if fit_path.exists() else pd.DataFrame()
    validate_fit_rows(fits, 10000)
    done = set(fits.sample_index.astype(int)) if len(fits) else set()
    rows = fits.to_dict('records')
    ranges = ParameterRanges(**{k: tuple(v) for k,v in data['metadata']['parameter_ranges'].items()})
    pending = [i for i in range(10000) if i not in done]
    if pending:
        protocol.update(status='fitting', resumed_utc=now(), n_jobs=args.n_jobs)
        atomic_json(output/'protocol.json', protocol)
        started = time.perf_counter()
        try:
            with parallel_config(backend='loky', inner_max_num_threads=1):
                stream = Parallel(n_jobs=args.n_jobs, max_nbytes=None,
                    return_as='generator_unordered', idle_worker_timeout=1800)(
                    delayed(fit_case)(i, np.asarray(data['annual_maxima'][i]),
                        np.asarray(data['metadata']['centered_time']), ranges) for i in pending)
                for result in stream:
                    rows.append(result)
                    if len(rows) % 100 == 0 or len(rows) == 10000:
                        fits = pd.DataFrame(rows).sort_values('sample_index')
                        atomic_csv(fit_path, fits)
                        print(f'MLE {len(rows)}/10000; failed={(~fits.success).sum()}; '
                              f'elapsed={time.perf_counter()-started:.0f}s', flush=True)
        except BaseException as exc:
            protocol.update(status='interrupted', error=repr(exc))
            raise
        finally:
            if rows:
                fits = pd.DataFrame(rows).sort_values('sample_index')
                atomic_csv(fit_path, fits)
            protocol['fit_wall_seconds'] += time.perf_counter()-started
            atomic_json(output/'protocol.json', protocol)
    summarize(output, protocol, data, scenario, prediction, fits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--n-jobs', type=int, default=6)
    run(parser.parse_args())
