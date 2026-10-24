"""Warm CPU throughput: raw 50-year sequences -> five original-scale coefficients.

Read-only access to frozen data/checkpoints. No training, selection or production
replacement. Disk I/O, model/pool setup, warm-up and post-fit diagnostics are
outside both timers. NN preprocessing and support decoding are inside its timer.
MLE uses four data-only starts and the existing retry rule, with no NN/truth input.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits, threadpool_info

from benchmark_cheng_21d_test_mle import fit_case, paired_metrics, read_json, now
from cheng_nn_evaluation import check_gev_predictions
from cheng_nn_simulation import COEFFICIENT_NAMES, ParameterRanges
from compare_cheng_nn_inputs import make_features
from compare_cheng_nn_reparameterization import DECODER_ID, predict_sample_coefficients
from train_cheng_nn import (PROJECT_ROOT, atomic_json, atomic_csv, file_sha256,
                            load_notebook_components)

STUDY = PROJECT_ROOT / 'results/cheng_nn_21d/reparameterization_100k_20261012'
MLE_STUDY = PROJECT_ROOT / 'results/cheng_nn_21d/test_mle_100k_20261007'
DEFAULT_OUTPUT = PROJECT_ROOT / 'results/cheng_nn_21d/timing_bound_vs_mle_20261013_cpu4'


def sample_scaling(values):
    """Compute, rather than reuse, the actual inference-time median and IQR."""
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 50 or not np.isfinite(y).all():
        raise ValueError('Need finite (N,50) raw annual maxima.')
    median = np.median(y, axis=1)
    q1, q3 = np.quantile(y, [.25, .75], axis=1)
    iqr = q3-q1
    if np.any(iqr <= 1e-12):
        raise ValueError('Positive IQR required.')
    return np.column_stack((median, iqr))


def nn_estimate(values, centered_time, model, scaler, inverse, margin, batch_size=2048):
    """Complete warm NN estimator, including the observation-dependent decoder."""
    scaling = sample_scaling(values)
    features = make_features(values, scaling, 21)
    z = (values-scaling[:, :1])/scaling[:, 1:2]
    coefficients = predict_sample_coefficients(
        model, features, scaler, z, centered_time, 'cpu', True,
        margin=margin, batch_size=batch_size)
    return inverse(coefficients, scaling)


def measure(function):
    started = time.perf_counter()
    value = function()
    return time.perf_counter()-started, value


def summarize_times(rows, n_cases):
    result = {}
    for method in ('MLE', 'constrained_NN'):
        seconds = np.array([r['seconds'] for r in rows if r['method'] == method])
        if len(seconds) == 0 or not np.isfinite(seconds).all() or np.any(seconds <= 0):
            raise ValueError('Need positive timings for both methods.')
        median = float(np.median(seconds))
        result[method] = dict(median_seconds=median, min_seconds=float(seconds.min()),
            max_seconds=float(seconds.max()), repeats=len(seconds),
            amortized_ms_per_sequence=1000*median/n_cases)
    a, b = result['MLE']['median_seconds'], result['constrained_NN']['median_seconds']
    return dict(methods=result, speedup_ratio=a/b, time_reduction_pct=100*(1-b/a))


def run(output=DEFAULT_OUTPUT, repeats=5, workers=4):
    output = Path(output)
    if repeats < 3 or workers < 1:
        raise ValueError('Use at least 3 repeats and at least 1 CPU worker/thread.')
    if output.exists():
        raise FileExistsError('Use a new output folder; never overwrite a benchmark.')
    experiment = read_json(STUDY/'protocol.json')
    old_mle = read_json(MLE_STUDY/'protocol.json')
    checkpoint = STUDY/'reparameterized/best.pt'
    training = read_json(STUDY/'reparameterized/run_metadata.json')
    if file_sha256(checkpoint) != training['checkpoint_sha256']:
        raise ValueError('Constrained checkpoint hash mismatch.')
    directory = Path(experiment['data_directory'])/'test'
    for name, digest in experiment['data_sha256']['test'].items():
        if file_sha256(directory/name) != digest:
            raise ValueError('Test data hash mismatch: '+name)
        if old_mle['fixed']['dataset_test_files_sha256'][name] != digest:
            raise ValueError('MLE and constrained NN test sets differ.')
    values = np.array(np.load(directory/'annual_maxima.npy', allow_pickle=False),
                      dtype=np.float64, copy=True)
    truth = np.load(directory/'coefficients_original.npy', allow_pickle=False)
    saved_scaling = np.load(directory/'sample_location_scale.npy', allow_pickle=False)
    metadata = read_json(directory/'metadata.json')
    centered_time = np.asarray(metadata['centered_time'], dtype=np.float64)
    if values.shape != (10000,50):
        raise ValueError('This benchmark requires all 10,000 paired test cases.')
    np.testing.assert_array_equal(sample_scaling(values), saved_scaling)
    ranges = ParameterRanges(**{k:tuple(v) for k,v in metadata['parameter_ranges'].items()})
    torch.set_num_threads(workers)
    torch.set_num_interop_threads(1)
    with redirect_stdout(io.StringIO()):
        ns = load_notebook_components()
    ns['INPUT_FEATURE_NAMES'] = tuple(f'input_{i}' for i in range(21))
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('decoder') != DECODER_ID or payload['architecture'] != [21,128,128,64,5]:
        raise ValueError('Unexpected constrained estimator.')
    model = ns['ChengGEVNet'](hidden_sizes=(128,128,64), dropout_p=0.)
    model.load_state_dict(payload['model_state'])
    model.eval()
    scaler = ns['TargetScaler'](**payload['target_scaler'])
    inverse = ns['inverse_sample_standardization']
    margin = payload['support_margin']
    expected_nn = np.load(STUDY/'reparameterized/test/predictions_original.npy', allow_pickle=False)
    expected_mle = pd.read_csv(MLE_STUDY/'mle_fits.csv').sort_values('sample_index')
    sources = [Path(__file__), PROJECT_ROOT/'src/cheng_nn_mle.py',
               PROJECT_ROOT/'src/benchmark_cheng_21d_test_mle.py',
               PROJECT_ROOT/'src/compare_cheng_nn_reparameterization.py',
               PROJECT_ROOT/'src/compare_cheng_nn_inputs.py',
               PROJECT_ROOT/'notebooks/cheng_NN.ipynb']
    report = dict(status='running', created_utc=now(), n_cases=10000, n_years=50,
        repeats=repeats, device='cpu', nn_threads=workers, mle_processes=workers,
        mle_threads_per_process=1, nn_batch_size=2048, training_seed=training['seed'],
        checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
        test_data_sha256=experiment['data_sha256']['test'],
        source_sha256={str(p):file_sha256(p) for p in sources},
        hardware=dict(processor=platform.processor(), platform=platform.platform(),
                      logical_cpu_count=os.cpu_count()),
        versions=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__),
        timer='time.perf_counter wall time',
        scope='RAM raw annual maxima to five original-scale coefficients, warm estimator',
        nn_included=['median/IQR recomputation','standardization','21D features','MLP',
                     'target inverse scaling','support decoder on all 50 observations','sample inverse scaling'],
        mle_included=['per-sequence preprocessing','four starts','optimization and existing retry rule',
                      'inverse scaling','dispatch/IPC and collecting results'],
        excluded=['imports','disk read/write','model loading','pool initialization','warm-up',
                  'NN training','post-fit diagnostics and RMSE'],
        warmup='One full NN batch of 10,000 sequences; MLE pool prewarmed with 4*workers fits',
        repeats_note='Repeated timing of the same frozen estimators/data, not new seeds or model selection',
        test_note='Previously inspected test set; not an independent new confirmatory accuracy test',
        training_seconds_separate=training['training_seconds'], training_device=training['device'],
        production_model_replaced=False, observations_used_by_decoder=True,
        nn_reproduction_tolerance=dict(rtol=1e-6, atol=1e-6,
            reason='Float32 CPU inference versus saved GPU evaluation; not bitwise equality'))
    output.mkdir(parents=True)
    atomic_json(output/'report.json', report)
    rows, predictions = [], {}
    try:
        with threadpool_limits(limits=1, user_api='blas'):
            report['parent_threadpools'] = threadpool_info()
            call_nn = lambda: nn_estimate(values, centered_time, model, scaler, inverse, margin)
            warm = call_nn()
            np.testing.assert_allclose(warm, expected_nn, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(np.sqrt(np.mean((warm-truth)**2,axis=0)),
                np.sqrt(np.mean((expected_nn-truth)**2,axis=0)),rtol=1e-6,atol=1e-7)
            report['nn_max_absolute_difference_from_saved_gpu'] = float(np.max(np.abs(warm-expected_nn)))
            with parallel_config(backend='loky', inner_max_num_threads=1):
                with Parallel(n_jobs=workers, max_nbytes=None) as pool:
                    pool(delayed(fit_case)(i, values[i], centered_time, ranges)
                         for i in range(4*workers))
                    def call_mle():
                        return pool(delayed(fit_case)(i, values[i], centered_time, ranges)
                                    for i in range(len(values)))
                    for repeat in range(1,repeats+1):
                        order = ('constrained_NN','MLE') if repeat % 2 else ('MLE','constrained_NN')
                        for method in order:
                            seconds, value = measure(call_nn if method == 'constrained_NN' else call_mle)
                            if method == 'MLE':
                                fits = pd.DataFrame(value).sort_values('sample_index')
                                if not fits.success.all() or len(fits) != len(values):
                                    raise RuntimeError('MLE failures must be reported, not silently excluded.')
                                prediction = fits[list(COEFFICIENT_NAMES)].to_numpy(float)
                                np.testing.assert_allclose(prediction, expected_mle[list(COEFFICIENT_NAMES)],
                                                           rtol=1e-6, atol=1e-7)
                            else:
                                prediction = value
                                np.testing.assert_allclose(prediction, expected_nn, rtol=1e-6, atol=1e-6)
                            predictions[method] = prediction
                            rows.append(dict(repeat=repeat, method=method, seconds=seconds))
                            atomic_csv(output/'timing_repeats.csv', pd.DataFrame(rows))
                            print(f'Repeat {repeat}/{repeats} {method}: {seconds:.6f} seconds', flush=True)
            summary = summarize_times(rows,len(values))
            comparison = paired_metrics(truth,predictions['constrained_NN'],predictions['MLE'],
                                        np.ones(len(values), dtype=bool))
            atomic_csv(output/'comparison.csv',comparison)
            checks = check_gev_predictions(predictions['constrained_NN'], values,
                       np.asarray(metadata['years']), reference_year=metadata['reference_year'],
                       time_scale_years=metadata['time_scale_years'])
            report.update(status='completed',completed_utc=now(), **summary,
                all_predictions_match_previous_estimates=True, mle_success=10000,
                constrained_support_failed=int((~checks.gev_valid_all_years).sum()))
            print(summary, flush=True)
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        atomic_json(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    run(args.output_directory,args.repeats,args.workers)
