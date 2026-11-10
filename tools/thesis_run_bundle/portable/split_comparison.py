"""Run NN and MLE on separate computers, then merge four paired simulations.

Each computer needs a byte-identical copy of the bundle, EAS inputs, and the
completed calibration folder. No arm reads or writes the other arm's results.
The merger runs only after both four-replicate arms have been copied together.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import comparison, inference, simulation, workflow
from .common import SEED, completed, identity, latest_completed, attempt_folder, read, sha, write
from research_workflows import COEFFICIENTS, YEARS
from train_cheng_nn import atomic_csv, atomic_array


def _array_identity(array):
    data = np.ascontiguousarray(array, dtype=np.float64)
    return dict(shape=list(data.shape), sha256=hashlib.sha256(data.tobytes()).hexdigest())


def _calibration(calibration_dir, real_folder, inputs, terrain_provenance):
    _, grids, predictors, config, crs = inputs
    real = comparison.validate_real_source(real_folder, inputs, terrain_provenance=terrain_provenance)
    report = read(calibration_dir / 'run_report.json')
    if report.get('status') != 'completed' or report.get('protocol_id') != identity(report.get('protocol')):
        raise ValueError('Need a completed, internally consistent calibration report')
    for name, digest in report['output_sha256'].items():
        if Path(name).name != name or sha(calibration_dir / name) != digest:
            raise ValueError(f'Calibration output changed: {name}')
    protocol = report['protocol']
    if (protocol.get('real_nn_sha256') != real['output_sha256']['nn_coefficients.csv']
            or not comparison.calibration_configs_match(protocol.get('config'), config)
            or protocol.get('coordinate_crs') != crs
            or protocol.get('coefficient_names') != list(COEFFICIENTS)
            or protocol.get('years') != list(YEARS)
            or protocol.get('master_seed') != SEED):
        raise ValueError('Calibration does not match this real EAS source or scientific configuration')
    terrain = comparison.terrain_input_metadata(inputs, terrain_provenance)
    if any(protocol.get(key) != value for key, value in terrain.items()):
        raise ValueError('Calibration terrain/latitude provenance differs')
    truth = np.load(calibration_dir / 'truth.npy', allow_pickle=False)
    if truth.shape != (len(grids), 5) or not np.isfinite(truth).all():
        raise ValueError('Calibration truth has wrong GRID order/shape or nonfinite values')
    saved = pd.read_csv(calibration_dir / 'truth_coefficients.csv', dtype={'grid_id': str})
    if saved.grid_id.tolist() != grids.grid_id.tolist() or not np.allclose(
            saved[list(COEFFICIENTS)].to_numpy(float), truth, atol=1e-12, rtol=0):
        raise ValueError('Calibration GRID order or saved truth differs')
    return truth, report


def _science_protocol(inputs, truth, calibration_report, mle_options):
    _, grids, predictors, config, crs = inputs
    scientific_config = {key: value for key, value in config.items() if key != 'n_jobs'}
    return dict(
        version=1, design='separate-host-paired-NN-MLE-four-replicates',
        calibration_id=calibration_report['protocol_id'],
        truth=_array_identity(truth),
        grid_id_sha256=identity(grids.grid_id.tolist()),
        grids_sha256=identity(grids.to_json()),
        predictors_sha256=identity(predictors.to_json()),
        scientific_config=scientific_config, coordinate_crs=crs,
        years=list(YEARS), seed=SEED, bound_model_sha256=inference.DIGEST,
        mle_options={key: value for key, value in mle_options.items() if key != 'jobs'},
        source_sha256=comparison.code_identity(), split_code_sha256=sha(Path(__file__)),
        fit_stream='SeedSequence(seed, spawn_key=(replicate_zero_based,))',
        evaluation_stream='SeedSequence(seed, spawn_key=(replicate_zero_based, 0x43525053))',
        rl_periods=[50, 100],
        comparison='All 7,822 EAS grids and all 50 years; known simulated RL truth. '
                   'Each method selects predictors/kernel independently inside identical buffered folds.',
    )


def _write_or_check_protocol(root, protocol):
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'split_protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Split science protocol differs; use the identical bundle/calibration on both computers')
    write(path, protocol)
    return identity(protocol)


def _method_options(mle_options):
    options = dict(domain='wide_domain', jobs=1, maxiter=600, retry_maxiter=1800)
    options.update(mle_options or {})
    if (options['domain'] not in ('wide_domain', 'training_domain') or options['jobs'] == 0
            or options['maxiter'] < 1 or options['retry_maxiter'] < options['maxiter']):
        raise ValueError('Invalid MLE settings')
    return options


def prepare_shared(real_folder, calibration_dir, inputs, root, *, mle_options=None,
                   terrain_provenance=None):
    """Freeze four fit/evaluation arrays ONCE, before duplicating the disk."""
    truth, calibration_report = _calibration(calibration_dir, real_folder, inputs, terrain_provenance)
    protocol = _science_protocol(inputs, truth, calibration_report, _method_options(mle_options))
    pid = _write_or_check_protocol(root, protocol)
    shared = root / 'shared'
    manifest_path = shared / 'sample_manifest.json'
    if manifest_path.exists():
        _load_shared(root, pid, truth)
        return manifest_path
    if shared.exists() and any(shared.iterdir()):
        raise ValueError('Incomplete shared samples exist; preserve them and inspect before preparing again')
    shared.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(4):
        folder = shared / f'replicate_{index + 1:03d}'
        folder.mkdir()
        values = simulation.draw_annual(truth, index)
        evaluation = simulation.draw_annual(truth, index, purpose='crps_evaluation')
        atomic_array(folder / 'annual_maxima.npy', values)
        atomic_array(folder / 'crps_evaluation_maxima.npy', evaluation)
        records.append(dict(replicate=index + 1,
            annual_sha256=sha(folder / 'annual_maxima.npy'),
            evaluation_sha256=sha(folder / 'crps_evaluation_maxima.npy')))
    write(manifest_path, dict(split_science_protocol_id=pid, n_grids=len(inputs[1]),
                              n_years=len(YEARS), replicates=records))
    return manifest_path


def _load_shared(root, protocol_id, truth):
    shared = root / 'shared'
    manifest = read(shared / 'sample_manifest.json')
    if (manifest.get('split_science_protocol_id') != protocol_id
            or manifest.get('n_grids') != len(truth)
            or manifest.get('n_years') != len(YEARS)
            or len(manifest.get('replicates', [])) != 4):
        raise ValueError('Shared sample manifest differs from the split experiment')
    samples = []
    for index, record in enumerate(manifest['replicates'], 1):
        folder = shared / f'replicate_{index:03d}'
        fit = folder / 'annual_maxima.npy'
        evaluation = folder / 'crps_evaluation_maxima.npy'
        if (record['replicate'] != index or sha(fit) != record['annual_sha256']
                or sha(evaluation) != record['evaluation_sha256']):
            raise ValueError(f'Shared replicate {index} checksum differs')
        values = np.load(fit, allow_pickle=False)
        heldout = np.load(evaluation, allow_pickle=False)
        if (values.shape != (len(truth), len(YEARS)) or heldout.shape != values.shape
                or not np.isfinite(values).all() or not np.isfinite(heldout).all()
                or np.array_equal(values, heldout)):
            raise ValueError(f'Shared replicate {index} shape/finite/independence check failed')
        samples.append((values, heldout, record))
    return samples


def run_arm(real_folder, calibration_dir, inputs, root, method, count, *, threads,
            mle_options=None, terrain_provenance=None):
    if method not in ('nn', 'mle') or count != 4:
        raise ValueError('Choose one arm, nn or mle, with exactly four replicates')
    truth, calibration_report = _calibration(calibration_dir, real_folder, inputs, terrain_provenance)
    options = _method_options(mle_options)
    protocol = _science_protocol(inputs, truth, calibration_report, options)
    pid = _write_or_check_protocol(root, protocol)
    samples = _load_shared(root, pid, truth)
    _, grids, predictors, config, crs = inputs
    arm_root = root / 'arms' / method
    arm_root.mkdir(parents=True, exist_ok=True)
    status_path = arm_root / 'arm_status.json'
    for index in range(count):
        values, evaluation, sample_record = samples[index]
        generation_seconds = 0.0  # prepared once, before running on separate hosts
        sample = identity(dict(protocol_id=pid, replicate=index + 1,
            fit_sha256=sample_record['annual_sha256'],
            evaluation_sha256=sample_record['evaluation_sha256']))
        method_id = identity(dict(sample_id=sample, estimator=method))
        parent = arm_root / f'replicate_{index + 1:03d}'
        prior = latest_completed(parent, method_id)
        if prior:
            folder, report = prior
            if 'sample_id' not in report:
                if (report.get('estimator') != method
                        or not np.array_equal(np.load(folder / 'annual_maxima.npy', allow_pickle=False), values)
                        or not np.array_equal(np.load(folder / 'crps_evaluation_maxima.npy', allow_pickle=False), evaluation)
                        or not np.array_equal(np.load(folder / 'truth.npy', allow_pickle=False), truth)):
                    raise ValueError('Interrupted completed arm does not match the regenerated sample')
                report.update(replicate=index + 1, sample_id=sample,
                    split_science_protocol_id=pid, generation_seconds=generation_seconds,
                    run_environment=dict(host='unknown_after_metadata_recovery', timing_comparable=False))
                write(folder / 'run_report.json', report)
            if (report.get('sample_id') != sample or report.get('replicate') != index + 1
                    or report.get('estimator') != method):
                raise ValueError('Completed replicate metadata differs')
            print(f'{method} replicate {index + 1}/4: verified existing {folder}', flush=True)
            continue
        folder = attempt_folder(parent)
        write(status_path, dict(status='running', method=method, replicate=index + 1,
                                total=4, output=str(folder)))
        print(f'{method} replicate {index + 1}/4: starting {folder}', flush=True)
        report = workflow.execute(values, grids, predictors, config, crs, folder,
            truth=truth, protocol_id=method_id, threads=threads, estimator=method,
            mle_options=options, evaluation_values=evaluation)
        report.update(replicate=index + 1, sample_id=sample,
                      split_science_protocol_id=pid, generation_seconds=generation_seconds,
                      run_environment=comparison.environment(threads, config['n_jobs'], options['jobs']))
        write(folder / 'run_report.json', report)
        write(status_path, dict(status='running', method=method, completed_replicates=index + 1,
                                total=4, last_output=str(folder)))
    write(status_path, dict(status='completed', method=method, completed_replicates=4, total=4,
                            split_science_protocol_id=pid))
    return status_path


def _branch(root, method, index, protocol_id, truth):
    parent = root / 'arms' / method / f'replicate_{index:03d}'
    branches = []
    for folder in sorted(parent.glob('attempt_*'), reverse=True):
        report = completed(folder)
        if report:
            if (report.get('replicate') != index or report.get('estimator') != method
                    or report.get('split_science_protocol_id') != protocol_id
                    or not np.array_equal(np.load(folder / 'truth.npy', allow_pickle=False), truth)):
                raise ValueError(f'Completed arm has incompatible protocol or truth: {folder}')
            branches.append(folder)
    if len(branches) != 1:
        raise ValueError(f'Expected exactly one verified completed {method} replicate {index}; found {len(branches)}')
    return branches[0]


def _return_levels(index, branches):
    frames = {}
    keys = ['grid_id', 'year', 'return_period']
    for method, folder in branches.items():
        frame = pd.read_csv(folder / 'return_levels.csv', dtype={'grid_id': str})
        if frame.duplicated(keys).any() or len(frame) != 7822 * 50 * 2:
            raise ValueError(f'RL grid/year/period coverage differs: {folder}')
        frames[method] = frame
    nn, mle = frames['nn'], frames['mle']
    if (not nn[keys].equals(mle[keys]) or not np.allclose(nn.truth.to_numpy(float), mle.truth.to_numpy(float),
            atol=1e-10, rtol=0)
            or sorted(nn.return_period.unique().tolist()) != [50, 100]):
        raise ValueError('Paired RL rows, return periods, or known truth differ')
    rows = []
    for period in (50, 100):
        mask = nn.return_period.eq(period).to_numpy()
        truth = nn.loc[mask, 'truth'].to_numpy(float)
        for stage in ('estimation', 'gp_oof'):
            vectors = {}
            for method, frame in frames.items():
                column = method if stage == 'estimation' else 'gp_oof'
                predicted = frame.loc[mask, column].to_numpy(float)
                if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
                    raise ValueError('Invalid RL cells: refuse post-hoc filtering or unequal coverage')
                vectors[method] = predicted - truth
            rows.append(dict(replicate=index, stage=stage, return_period=period,
                n_grid_years=int(mask.sum()),
                NN_RMSE=float(np.sqrt(np.mean(vectors['nn'] ** 2))),
                MLE_RMSE=float(np.sqrt(np.mean(vectors['mle'] ** 2))),
                NN_MAE=float(np.mean(np.abs(vectors['nn']))),
                MLE_MAE=float(np.mean(np.abs(vectors['mle']))),
                NN_Bias=float(np.mean(vectors['nn'])),
                MLE_Bias=float(np.mean(vectors['mle']))))
    table = pd.DataFrame(rows)
    table['NN_RMSE_improvement_percent'] = 100 * (1 - table.NN_RMSE / table.MLE_RMSE)
    table['NN_MAE_improvement_percent'] = 100 * (1 - table.NN_MAE / table.MLE_MAE)
    return table


def _paired_crps_cross_host(index, branches, evaluation_values, *, yearly=False):
    """Match identical evaluation cells, allowing only floating-point score roundoff."""
    tables = {}
    filename = 'crps_yearly_metrics.csv' if yearly else 'crps_metrics.csv'
    for method, folder in branches.items():
        if not np.array_equal(np.load(folder / 'crps_evaluation_maxima.npy', allow_pickle=False), evaluation_values):
            raise ValueError('Cross-host CRPS evaluation observations differ')
        table = pd.read_csv(folder / filename, dtype={'year': str})
        table['stage'] = table.stage.replace({method: 'estimation'})
        if table.duplicated(['stage', 'year']).any():
            raise ValueError('Duplicate CRPS metrics')
        tables[method] = table
    a, b = tables['nn'], tables['mle']
    oracle_a = a[a.stage.eq('truth')].sort_values('year').reset_index(drop=True)
    oracle_b = b[b.stage.eq('truth')].sort_values('year').reset_index(drop=True)
    if (not oracle_a[['stage', 'year', 'n_total', 'n_invalid']].equals(
            oracle_b[['stage', 'year', 'n_total', 'n_invalid']])
            or not np.allclose(oracle_a.mean_CRPS_C, oracle_b.mean_CRPS_C, atol=1e-10, rtol=0)):
        raise ValueError('Cross-host oracle CRPS differs beyond numeric roundoff')
    keys = ['stage', 'year', 'n_total']
    result = a[~a.stage.eq('truth')].merge(b[~b.stage.eq('truth')], on=keys,
        suffixes=('_NN', '_MLE'), how='outer', indicator=True, validate='one_to_one')
    if len(result) != (100 if yearly else 2) or not result._merge.eq('both').all():
        raise ValueError('CRPS methods do not cover identical years and stages')
    expected = len(evaluation_values) if yearly else evaluation_values.size
    if not result.n_total.eq(expected).all():
        raise ValueError('CRPS scoring silently changed coverage')
    result = result.drop(columns='_merge').merge(
        oracle_a[['year', 'mean_CRPS_C', 'n_invalid']], on='year', validate='many_to_one')
    result = result.rename(columns={'mean_CRPS_C': 'oracle_CRPS_C',
                                     'n_invalid': 'oracle_invalid_cells'})
    result['NN_improvement_percent'] = 100 * (
        1 - result.mean_CRPS_C_NN / result.mean_CRPS_C_MLE.replace(0, np.nan))
    result.insert(0, 'replicate', index)
    return result


def merge_arms(real_folder, calibration_dir, inputs, root, *, mle_options=None,
               terrain_provenance=None):
    truth, calibration_report = _calibration(calibration_dir, real_folder, inputs, terrain_provenance)
    protocol = _science_protocol(inputs, truth, calibration_report, _method_options(mle_options))
    pid = _write_or_check_protocol(root, protocol)
    samples = _load_shared(root, pid, truth)
    grids = inputs[1]
    accuracy, timing, quality, crps, crps_yearly, rl = [], [], [], [], [], []
    for index in range(1, 5):
        branches = {method: _branch(root, method, index, pid, truth) for method in ('nn', 'mle')}
        values, evaluation, _ = samples[index - 1]
        a, t, q = comparison.summarize_pair(index - 1, branches, values, truth, grids)
        accuracy.append(a)
        timing.append(t)
        quality.append(q)
        crps.append(_paired_crps_cross_host(index, branches, evaluation))
        crps_yearly.append(_paired_crps_cross_host(index, branches, evaluation, yearly=True))
        rl.append(_return_levels(index, branches))
        print(f'verified pair {index}/4', flush=True)
    out = root / 'merged'
    out.mkdir(parents=True, exist_ok=True)
    tables = dict(parameter_comparison=pd.concat(accuracy, ignore_index=True),
                  timing_comparison=pd.DataFrame(timing),
                  quality_comparison=pd.concat(quality, ignore_index=True),
                  crps_comparison=pd.concat(crps, ignore_index=True),
                  crps_yearly_comparison=pd.concat(crps_yearly, ignore_index=True),
                  return_level_comparison=pd.concat(rl, ignore_index=True))
    for name, table in tables.items():
        atomic_csv(out / f'{name}.csv', table)
    summary = tables['return_level_comparison'].groupby(['stage', 'return_period'], sort=False).agg(
        n_replicates=('replicate', 'nunique'), NN_mean_RMSE=('NN_RMSE', 'mean'),
        MLE_mean_RMSE=('MLE_RMSE', 'mean'), NN_mean_MAE=('NN_MAE', 'mean'),
        MLE_mean_MAE=('MLE_MAE', 'mean'), NN_mean_Bias=('NN_Bias', 'mean'),
        MLE_mean_Bias=('MLE_Bias', 'mean')).reset_index()
    summary['NN_RMSE_improvement_percent'] = 100 * (1 - summary.NN_mean_RMSE / summary.MLE_mean_RMSE)
    summary['NN_MAE_improvement_percent'] = 100 * (1 - summary.NN_mean_MAE / summary.MLE_mean_MAE)
    atomic_csv(out / 'return_level_summary.csv', summary)
    crps_summary = tables['crps_comparison'].groupby('stage', sort=False).agg(
        n_replicates=('replicate', 'nunique'), NN_mean_CRPS_C=('mean_CRPS_C_NN', 'mean'),
        MLE_mean_CRPS_C=('mean_CRPS_C_MLE', 'mean')).reset_index()
    crps_summary['NN_improvement_percent'] = 100 * (1 - crps_summary.NN_mean_CRPS_C / crps_summary.MLE_mean_CRPS_C)
    atomic_csv(out / 'crps_summary.csv', crps_summary)
    report = dict(status='completed', pairs=4, grids=len(grids), years=len(YEARS),
                  split_science_protocol_id=pid, timing_comparable=False,
                  timing_note='Different hosts: report wall times separately, not a causal NN/MLE speedup ratio.',
                  interpretation='Conditional calibrated simulation; four draws on one fixed truth surface. '
                                 'RL50/RL100 scored against known simulated truth, not real-world observed truth.',
                  output_sha256={p.name: sha(p) for p in out.glob('*.csv')})
    write(out / 'run_report.json', report)
    print(summary.to_string(index=False), flush=True)
    return out
