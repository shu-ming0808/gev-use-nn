"""Experimental global support-decoder inference; never replaces production.

Re-read and verify the identical annual NetCDF and 21D transformation. Reuse
the hash-verified original predictions, infer with the frozen experimental
checkpoint, and audit both models independently in Celsius. Real-data
coefficient differences are NOT RMSE against truth. No RL or model selection.
"""
from __future__ import annotations

# Keep direct-script execution independent of the working directory.
import sys as _sys
from pathlib import Path as _SourcePath
_SOURCE_ROOT = _SourcePath(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import time
PROCESS_START = time.perf_counter()

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import genextreme

from global_pipeline import infer_global_cheng_nn as base
from compare_cheng_nn_reparameterization import (
    DECODER_ID, MARGIN, predict_sample_coefficients,
)
from train_cheng_nn import atomic_array, atomic_csv, atomic_json, file_sha256

DEFAULT_BASELINE = Path('D:/論文資料/ERA5/cheng_nn_21d_1976_2025/run_20261011_01')
CANDIDATE = base.PROJECT_ROOT / 'results/cheng_nn_21d/reparameterization_100k_20261012/reparameterized/best.pt'
CANDIDATE_HASH = '03890b4877a40cf93822877d09231c003c920f34c68ed911b1da481ce396318f'


def independent_audit(coefficients, observations, chunk_size=4096):
    """Independent endpoint inequalities and SciPy logpdf, both shape signs."""
    parts = []
    t = (np.asarray(base.YEARS)-base.REFERENCE_YEAR)/base.TIME_SCALE_YEARS
    for start in range(0, len(observations), chunk_size):
        c = np.asarray(coefficients[start:start+chunk_size], dtype=np.float64)
        y = np.asarray(observations[start:start+chunk_size], dtype=np.float64)
        mu = c[:, :1]+c[:, 1:2]*t
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            sigma = np.exp(c[:, 2:3]+c[:, 3:4]*t)
            xi = c[:, 4:5]
            endpoint = mu-sigma/xi
            excess = np.where(xi < 0, y-endpoint,
                              np.where(xi > 0, endpoint-y, -np.inf))
            margin = 1+xi*(y-mu)/sigma
            logpdf = genextreme.logpdf(y, c=-xi, loc=mu, scale=sigma)
        parts.append(pd.DataFrame({
            'endpoint_violation_years': (excess >= 0).sum(axis=1),
            'max_endpoint_excess_C': np.maximum(excess.max(axis=1), 0),
            'minimum_support_margin_independent': margin.min(axis=1),
            'nonfinite_logpdf_years': (~np.isfinite(logpdf)).sum(axis=1),
            'numerically_valid_parameters': (np.isfinite(c).all(axis=1)
                & np.isfinite(mu).all(axis=1) & np.isfinite(sigma).all(axis=1)
                & (sigma > 0).all(axis=1)),
        }))
    return pd.concat(parts, ignore_index=True)


def compare_tables(grids, original, candidate, old_checks, new_checks, old_audit, new_audit):
    """Preserve grid order, identify resolved, persistent, and new failures."""
    n = len(grids)
    if grids.grid_id.duplicated().any() or any(len(x) != n for x in
            (original, candidate, old_checks, new_checks, old_audit, new_audit)):
        raise ValueError('Grid identity/row count mismatch.')
    table = grids.copy().reset_index(drop=True)
    for arm, pred, checks, audit in [('original', original, old_checks, old_audit),
                                     ('bound', candidate, new_checks, new_audit)]:
        np.testing.assert_array_equal(checks.n_support_violations, audit.endpoint_violation_years)
        np.testing.assert_allclose(checks.min_support_margin,
                                   audit.minimum_support_margin_independent, atol=1e-12, rtol=1e-12)
        for j, name in enumerate(base.COEFFICIENTS):
            table[f'{arm}_{name}'] = pred[:, j]
        for name in ('n_support_violations', 'min_support_margin', 'gev_valid_all_years',
                     'n_support_numerical_failures', 'parameters_valid_all_years',
                     'any_coefficient_outside_simulation_design', 'needs_review'):
            table[f'{arm}_{name}'] = checks[name].to_numpy()
        table[f'{arm}_max_endpoint_excess_C'] = audit.max_endpoint_excess_C.to_numpy()
    old_bad = table.original_n_support_violations.gt(0)
    new_bad = table.bound_n_support_violations.gt(0)
    table['support_transition'] = np.select(
        [old_bad & table.bound_gev_valid_all_years, old_bad & ~table.bound_gev_valid_all_years,
         ~old_bad & new_bad, ~old_bad & ~table.bound_gev_valid_all_years],
        ['resolved', 'unresolved', 'new_violation', 'new_other_invalid'], default='valid_both')
    stats = []
    for subset, use in [('all', np.ones(n, dtype=bool)), ('original_support_failures', old_bad),
                        ('original_excess_gt_1C', table.original_max_endpoint_excess_C.gt(1))]:
        for j, name in enumerate(base.COEFFICIENTS):
            delta = candidate[use, j]-original[use, j]
            stats.append(dict(subset=subset, coefficient=name, n=len(delta),
                mean_change=float(np.mean(delta)) if len(delta) else None,
                median_change=float(np.median(delta)) if len(delta) else None,
                median_absolute_change=float(np.median(np.abs(delta))) if len(delta) else None,
                p95_absolute_change=float(np.quantile(np.abs(delta), .95)) if len(delta) else None,
                maximum_absolute_change=float(np.max(np.abs(delta))) if len(delta) else None))
    return table, pd.DataFrame(stats)


def run(baseline, output, candidate=CANDIDATE, candidate_hash=CANDIDATE_HASH,
        device='cuda', batch_size=2048, process_start=None):
    start = time.perf_counter() if process_start is None else process_start
    baseline, output, candidate = [Path(p).resolve() for p in (baseline, output, candidate)]
    if output.exists():
        raise FileExistsError('Use a new output directory; no existing artifacts are overwritten.')
    if batch_size < 1 or device not in {'cpu', 'cuda'}:
        raise ValueError('Invalid batch size or device.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent substitution.')
    torch.set_num_threads(4)
    output.mkdir(parents=True)
    state = dict(status='running', started_utc=base.utc_now())
    timings = {}

    def stage(name):
        state.update(stage=name, updated_utc=base.utc_now())
        atomic_json(output/'run_status.json', state)
        print(f'[{time.perf_counter()-start:.2f}s] {name}', flush=True)

    try:
        stage('verify_baseline_and_model_identities')
        with base.timed(timings, 'identity_verification_seconds'):
            old_report = json.loads((baseline/'run_report.json').read_text(encoding='utf-8'))
            if old_report['status'] != 'completed' or old_report['model']['sha256'] != base.CHECKPOINT_HASH:
                raise ValueError('Baseline is not the frozen completed global run.')
            if old_report['years'] != list(base.YEARS) or old_report['input']['land_threshold'] != 'LSM > 0.5':
                raise ValueError('Baseline time or land contract mismatch.')
            protected = {baseline/'run_report.json': file_sha256(baseline/'run_report.json'),
                         base.CHECKPOINT: base.CHECKPOINT_HASH, candidate: candidate_hash}
            protected.update({baseline/name: h for name, h in old_report['output_sha256'].items()})
            for p, h in protected.items():
                if file_sha256(p) != h:
                    raise ValueError(f'Protected artifact hash mismatch: {p.name}')
            ns, model, scaler, payload = base.load_frozen_model(candidate, candidate_hash, device)
            if (payload.get('decoder') != DECODER_ID or payload.get('support_margin') != MARGIN
                    or payload.get('requires_observed_sequence_at_inference') is not True
                    or payload.get('latent_coefficient_names') != ['mu0','beta_mu','delta','beta_sigma','xi0']):
                raise ValueError('Wrong experimental decoder contract.')
            for p, h in payload['source_hashes'].items():
                if file_sha256(p) != h:
                    raise ValueError(f'Frozen candidate source changed: {p}')
                protected[Path(p)] = h
            for name in ('mean', 'std'):
                np.testing.assert_array_equal(getattr(scaler, name), old_report['model']['target_scaler'][name])
        stage('reread_identical_annual_land_data')
        with base.timed(timings, 'read_verify_annual_data_seconds'):
            input_path = Path(old_report['input']['path'])
            if file_sha256(input_path) != old_report['input']['sha256']:
                raise ValueError('Annual input changed since the original run.')
            before = (input_path.stat().st_size, input_path.stat().st_mtime_ns)
            y, grids = base.read_annual_land(input_path)
            saved_grids = pd.read_csv(baseline/'grid_index.csv', float_precision='round_trip')
            pd.testing.assert_frame_equal(grids, saved_grids)
            np.testing.assert_array_equal(y, np.load(baseline/'annual_maxima_land.npy', allow_pickle=False))
            original = np.load(baseline/'coefficients_original.npy', allow_pickle=False)
        stage('transform_21D_and_standardize_observations')
        with base.timed(timings, 'feature_transform_seconds'):
            x, scaling = base.transform_sequences(y)
            np.testing.assert_array_equal(x, np.load(baseline/'inputs_21d.npy', allow_pickle=False))
            np.testing.assert_array_equal(scaling, np.load(baseline/'sample_location_scale.npy', allow_pickle=False))
            z = (y-scaling[:, :1])/scaling[:, 1:2]
        stage('bound_NN_inference')
        with base.timed(timings, 'bound_NN_and_decoder_seconds', device):
            sample_pred = predict_sample_coefficients(model, x, scaler, z,
                (np.asarray(base.YEARS)-base.REFERENCE_YEAR)/base.TIME_SCALE_YEARS,
                device, True, margin=MARGIN, batch_size=batch_size)
        with base.timed(timings, 'inverse_scaling_seconds'):
            prediction = ns['inverse_sample_standardization'](sample_pred, scaling)
        stage('dual_support_and_endpoint_audit')
        with base.timed(timings, 'both_models_audit_seconds'):
            low = np.asarray(old_report['feature_bounds_training_min'])
            high = np.asarray(old_report['feature_bounds_training_max'])
            ranges = old_report['parameter_ranges']
            diagnostics = [base.diagnostic_table(p, y, x, low, high, ranges) for p in (original, prediction)]
            audits = [independent_audit(p, y) for p in (original, prediction)]
            old, new = diagnostics
            if int(old.n_support_violations.gt(0).sum()) != old_report['summary']['observed_support_failure_grids']:
                raise RuntimeError('Original support-failure count not reproduced.')
            table, changes = compare_tables(grids, original, prediction, old, new, *audits)
            summary = []
            for arm, checks, audit in zip(('original', 'bound'), diagnostics, audits):
                bad = checks.n_support_violations.gt(0)
                summary.append(dict(model=arm, n_grids=len(y), failed_grids=int(bad.sum()),
                    failure_rate_percent=100*float(bad.mean()),
                    violating_grid_years=int(checks.n_support_violations.sum()),
                    invalid_parameter_grids=int((~checks.parameters_valid_all_years).sum()),
                    numerical_support_failure_grids=int(checks.n_support_numerical_failures.gt(0).sum()),
                    minimum_support_margin=float(checks.min_support_margin.min()),
                    nonfinite_logpdf_grids=int(audit.nonfinite_logpdf_years.gt(0).sum()),
                    coefficient_range_warning_grids=int(checks.any_coefficient_outside_simulation_design.sum()),
                    feature_range_warning_grids=int(checks.features_outside_train_minmax.sum()),
                    review_required_grids=int(checks.needs_review.sum())))
            passed = (bool(new.gev_valid_all_years.all())
                      and bool((new.min_support_margin >= MARGIN-1e-10).all())
                      and bool((audits[1].nonfinite_logpdf_years == 0).all()))
        stage('save_separate_experimental_outputs')
        with base.timed(timings, 'save_outputs_seconds'):
            atomic_array(output/'coefficients_original.npy', prediction)
            atomic_array(output/'coefficients_standardized.npy', sample_pred)
            atomic_csv(output/'grid_comparison.csv', table)
            atomic_csv(output/'original_302_review.csv', table.loc[table.original_n_support_violations.gt(0)])
            atomic_csv(output/'original_severe_review.csv', table.loc[table.original_max_endpoint_excess_C.gt(1)])
            atomic_csv(output/'new_failures.csv', table.loc[table.support_transition.isin(['new_violation','new_other_invalid'])])
            atomic_csv(output/'coefficient_changes.csv', changes)
            atomic_csv(output/'support_comparison.csv', pd.DataFrame(summary))
            atomic_csv(output/'bound_coefficients_and_checks.csv', pd.concat(
                [grids, pd.DataFrame(prediction, columns=base.COEFFICIENTS), new], axis=1))
            for arm, audit in zip(('original','bound'), audits):
                atomic_csv(output/f'{arm}_independent_audit.csv', pd.concat([grids[['grid_id']], audit], axis=1))
        stage('final_integrity_check')
        for p, h in protected.items():
            if file_sha256(p) != h:
                raise RuntimeError(f'Protected artifact changed: {p}')
        if before != (input_path.stat().st_size, input_path.stat().st_mtime_ns):
            raise RuntimeError('Annual input changed during run.')
        timings['bound_pipeline_seconds'] = sum(timings[k] for k in
            ('feature_transform_seconds','bound_NN_and_decoder_seconds','inverse_scaling_seconds'))
        timings['total_comparison_seconds_until_report'] = time.perf_counter()-start
        report = dict(status='completed', completed_utc=base.utc_now(), summary=summary,
            support_transitions={str(k): int(v) for k,v in table.support_transition.value_counts().items()},
            decoder_audit_passed=passed, n_original_severe_grids=int(table.original_max_endpoint_excess_C.gt(1).sum()),
            n_severe_resolved=int((table.original_max_endpoint_excess_C.gt(1) & table.bound_gev_valid_all_years).sum()),
            baseline_directory=str(baseline), baseline_checkpoint_sha256=base.CHECKPOINT_HASH,
            candidate_checkpoint=str(candidate), candidate_checkpoint_sha256=candidate_hash,
            candidate_best_epoch=payload['best_epoch'], input=old_report['input'],
            years=list(base.YEARS), n_years=50, decoder=DECODER_ID, epsilon=MARGIN,
            timing=timings, runtime=dict(device=device, batch_size=batch_size, cpu_threads=4),
            original_predictions_reused=True, exact_input_and_feature_match=True,
            protected_artifacts_unchanged=True, official_model_replaced=False,
            limitations=['No real-data coefficient truth: changes are not estimation RMSE.',
                '21D MLP plus the same observed 50-year sequence in the decoder.',
                'Observed support is constrained by construction, not an independent predictive validation.',
                'No guarantee for future observations or RL accuracy; no RL or temporal selection performed.',
                'Timing includes paired audits and output writing; not a controlled speed comparison.'],
            source_sha256=file_sha256(__file__),
            output_sha256={p.name:file_sha256(p) for p in output.iterdir() if p.suffix in {'.csv','.npy'}})
        atomic_json(output/'run_report.json', report)
        state.update(status='completed', stage='completed', summary=summary, decoder_audit_passed=passed)
        atomic_json(output/'run_status.json', state)
        print(json.dumps({k:report[k] for k in ('summary','support_transitions','decoder_audit_passed','timing')}, ensure_ascii=False), flush=True)
        return report
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), updated_utc=base.utc_now())
        atomic_json(output/'run_status.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-directory', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    args = parser.parse_args()
    run(args.baseline_directory, args.output_directory, device=args.device, process_start=PROCESS_START)
