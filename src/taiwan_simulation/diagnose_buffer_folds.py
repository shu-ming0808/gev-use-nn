"""Development diagnostics for 50 Taiwan ERA5 GRID and five frozen NN coefficients.

No GP selection, final spatial CV, return levels, bootstrap or NN training.
Geometry exactly follows cheng_spatial_cv; variograms are descriptive, not a
certificate of independence or a rule to minimise validation RMSE.
"""
from __future__ import annotations

import sys
from pathlib import Path
SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from research_workflows import CONTRACT, COEFFICIENTS, YEARS, validate_annual_sequences
from spatial_coordinates import project_lonlat_to_twd97_km
from elevation_gp_analysis import _buffered_training_indices
from train_cheng_nn import atomic_csv, atomic_json, file_sha256


def validate_geometry(grids):
    required = {'grid_id', 'x_km', 'y_km'}
    if not required.issubset(grids):
        raise ValueError('GRID identifiers and metric x_km/y_km are required.')
    xy = grids[['x_km', 'y_km']].to_numpy(float)
    if (grids.grid_id.isna().any() or grids.grid_id.duplicated().any()
            or not np.isfinite(xy).all() or len(np.unique(xy, axis=0)) != len(grids)):
        raise ValueError('Duplicate or invalid GRID identifiers/coordinates.')
    return xy


def _counts(values, n, label):
    values = list(values)
    if (not values or len(set(values)) != len(values)
            or any(not isinstance(v, (int, np.integer)) or not 2 <= v < n for v in values)):
        raise ValueError(f'Invalid {label} fold counts.')
    return values


def geometry_diagnostics(grids, *, outer_counts, inner_counts, buffers,
                         min_train=20, seed=20260929, distance_matrix=None,
                         progress=None):
    """Use coordinate-only splits; no response/predictor affects these tables.

    Inner folds are recomputed AFTER each outer buffer, with the exact seeds
    used by nested_cv. Unconstructable splits are failures, not omitted rows.
    No training cap is applied (the real runner must use max_train >= n_grid).
    """
    frame = grids.reset_index(drop=True).copy()
    xy = validate_geometry(frame)
    if distance_matrix is not None:
        distance_matrix = np.asarray(distance_matrix)
        if (distance_matrix.shape != (len(frame), len(frame))
                or not np.isfinite(distance_matrix).all() or (distance_matrix < 0).any()
                or not np.allclose(distance_matrix, distance_matrix.T)
                or not np.allclose(np.diag(distance_matrix), 0)):
            raise ValueError('Distance matrix must be finite, symmetric, nonnegative and aligned.')

    def retain(candidates, test, buffer):
        if distance_matrix is None:
            return _buffered_training_indices(frame, candidates, test, float(buffer))
        return candidates[distance_matrix[np.ix_(candidates, test)].min(axis=1) > buffer]
    outer_counts = _counts(outer_counts, len(frame), 'outer')
    inner_counts = _counts(inner_counts, len(frame), 'inner')
    buffers = list(buffers)
    if (not buffers or len(set(buffers)) != len(buffers)
            or any(not np.isfinite(b) or b < 0 for b in buffers)
            or not isinstance(min_train, (int, np.integer)) or min_train < 2):
        raise ValueError('Buffers must be unique finite nonnegative km; min_train >= 2.')
    all_splits, summaries, assignments, quality = [], [], [], []

    def split_row(train, test, candidates, **context):
        if len(train) and len(test):
            matrix = (cdist(xy[train], xy[test]) if distance_matrix is None
                      else distance_matrix[np.ix_(train, test)])
            distance = matrix.min(axis=0)
        else:
            distance = np.array([])
        return dict(**context, n_train=len(train), n_test=len(test),
            n_removed=len(candidates)-len(train),
            train_fraction=len(train)/len(candidates) if len(candidates) else 0.,
            nearest_distance_km=float(distance.min()) if len(distance) else np.nan,
            median_test_nearest_distance_km=float(np.median(distance)) if len(distance) else np.nan,
            max_test_nearest_distance_km=float(distance.max()) if len(distance) else np.nan,
            train_ids='|'.join(frame.iloc[train].grid_id.astype(str)),
            test_ids='|'.join(frame.iloc[test].grid_id.astype(str)),
            min_train_pass=len(train) >= min_train)

    # Keep native threading identical to the formal runner: even tied KMeans
    # solutions can change with floating-point summation/thread counts.
    for outer_k in outer_counts:
        km = KMeans(n_clusters=outer_k, n_init=100, random_state=seed).fit(xy)
        labels = km.labels_
        sizes = np.bincount(labels, minlength=outer_k)
        quality.append(dict(outer_folds=outer_k, inertia=float(km.inertia_),
            silhouette=float(silhouette_score(xy, labels, sample_size=min(2000, len(xy)),
                                              random_state=seed)), min_fold_size=int(sizes.min()),
            max_fold_size=int(sizes.max()), size_ratio=float(sizes.max()/sizes.min())))
        assignments.extend(dict(grid_id=g, outer_folds=outer_k, outer_fold=int(f))
                           for g, f in zip(frame.grid_id, labels))
        for buffer in buffers:
            # Same outer geometry is reused for every inner count.
            outer_parts = []
            for fold in range(outer_k):
                test = np.flatnonzero(labels == fold)
                candidates = np.flatnonzero(labels != fold)
                train = retain(candidates, test, buffer)
                outer_parts.append((fold, train, test, candidates))
            for inner_k in inner_counts:
                context = dict(outer_folds=outer_k, inner_folds=inner_k, buffer_km=float(buffer))
                if progress is not None:
                    progress(context)
                rows = []
                for fold, train, test, candidates in outer_parts:
                    rows.append(split_row(train, test, candidates, **context,
                                          level='outer', outer_fold=fold, inner_fold=-1))
                    if len(train) >= inner_k:
                        ilabel = KMeans(n_clusters=inner_k, n_init=100,
                            random_state=seed+1000+fold).fit_predict(xy[train])
                        for inner in range(inner_k):
                            itest = train[ilabel == inner]
                            icandidates = train[ilabel != inner]
                            itrain = retain(icandidates, itest, buffer)
                            rows.append(split_row(itrain, itest, icandidates, **context,
                                level='inner', outer_fold=fold, inner_fold=inner))
                    else:
                        for inner in range(inner_k):
                            rows.append(split_row(np.array([], dtype=int), np.array([], dtype=int),
                                train, **context, level='inner', outer_fold=fold, inner_fold=inner))
                part = pd.DataFrame(rows)
                summary = dict(**context, min_train_required=min_train)
                for level in ('outer', 'inner'):
                    p = part.loc[part.level.eq(level)]
                    for name in ('train', 'test'):
                        summary[f'{level}_{name}_min'] = int(p[f'n_{name}'].min())
                        summary[f'{level}_{name}_max'] = int(p[f'n_{name}'].max())
                        summary[f'{level}_{name}_median'] = float(p[f'n_{name}'].median())
                    summary[f'{level}_failed_splits'] = int((~p.min_train_pass).sum())
                    summary[f'{level}_total_splits'] = len(p)
                    summary[f'{level}_removed_mean'] = float(p.n_removed.mean())
                    summary[f'{level}_retained_fraction_min'] = float(p.train_fraction.min())
                    summary[f'actual_{level}_min_distance_km'] = float(p.nearest_distance_km.min())
                    observed = p.median_test_nearest_distance_km.dropna()
                    summary[f'{level}_test_nearest_distance_median_km'] = float(observed.median()) if len(observed) else np.nan
                summary['all_folds_meet_min_train'] = bool(part.min_train_pass.all())
                summary['all_test_folds_at_least_two'] = bool(part.n_test.ge(2).all())
                summary['outer_test_at_least_five'] = summary['outer_test_min'] >= 5
                summary['inner_test_at_least_three'] = summary['inner_test_min'] >= 3
                summaries.append(summary)
                all_splits.extend(rows)
    return dict(summary=pd.DataFrame(summaries), splits=pd.DataFrame(all_splits),
                outer_assignments=pd.DataFrame(assignments), fold_quality=pd.DataFrame(quality))


def elevation_residuals(coefficients, elevation):
    """Predeclared intercept + elevation OLS, fitted only to supplied rows.

    This is a variogram mean-removal diagnostic, NOT the final GP mean model.
    """
    y = coefficients.to_numpy(float)
    h = np.asarray(elevation, dtype=float)
    if h.shape != (len(y),) or not np.isfinite(h).all() or not np.isfinite(y).all():
        raise ValueError('Finite, paired elevation and coefficients are required.')
    scale = float(np.std(h))
    if scale <= 1e-12:
        raise ValueError('Elevation has no variation in this subset.')
    x = np.column_stack([np.ones(len(h)), (h-h.mean())/scale])
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != 2:
        raise ValueError('Elevation diagnostic baseline is rank deficient.')
    return pd.DataFrame(y-x@beta, columns=coefficients.columns, index=coefficients.index)


def load_diagnostic_inputs(prepared, predictors_path):
    """Validate real-data/raw provenance without loading old NN predictions."""
    report_path = prepared/'run_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    expected = dict(status='completed', completed_stage='taiwan_preparation_and_nn_only',
        contract=CONTRACT, data_role='real_data_pilot_not_simulated_truth',
        coordinate_crs='EPSG:3826', coordinate_units='km', n_grids=50)
    if any(report.get(k) != v for k, v in expected.items()) or report.get('years') != list(YEARS):
        raise ValueError('Require the complete Taiwan 50 GRID, 1976-2025 pilot contract.')
    hashes = {str(report_path): file_sha256(report_path)}
    for name in ('annual_maxima.npy', 'years.npy', 'grid_index.csv'):
        path = prepared/name
        digest = file_sha256(path)
        if digest != report.get('output_sha256', {}).get(name):
            raise ValueError(f'Pilot input hash mismatch: {name}')
        hashes[str(path)] = digest
    raw = Path(report['input_path']).resolve()
    digest = file_sha256(raw)
    if digest != report.get('source_sha256', {}).get(str(raw)):
        raise ValueError('Original annual source hash mismatch.')
    hashes[str(raw)] = digest
    values = validate_annual_sequences(np.load(prepared/'years.npy', allow_pickle=False),
                                      np.load(prepared/'annual_maxima.npy', allow_pickle=False))
    grids = pd.read_csv(prepared/'grid_index.csv', dtype={'grid_id': str})
    xy = validate_geometry(grids)
    if len(grids) != 50 or len(values) != 50:
        raise ValueError('Exactly 50 aligned Taiwan GRID are required.')
    if not np.allclose(xy, project_lonlat_to_twd97_km(grids.longitude_180, grids.latitude), atol=1e-6, rtol=0):
        raise ValueError('Expected Taiwan EPSG:3826 coordinates in km.')
    predictors = pd.read_csv(predictors_path, dtype={'grid_id': str})
    if (predictors.grid_id.isna().any() or predictors.grid_id.duplicated().any()
            or set(predictors.grid_id) != set(grids.grid_id)):
        raise ValueError('Predictor GRID keys differ or are duplicated.')
    predictors = predictors.set_index('grid_id').loc[grids.grid_id].reset_index()
    if not predictors.candidate_data_complete.eq(True).all() or not np.isfinite(predictors.elevation_m).all():
        raise ValueError('Complete predictors/elevation are required.')
    hashes[str(predictors_path)] = file_sha256(predictors_path)
    return values, grids, predictors, hashes


def variogram_diagnostics(grids, coefficients, elevation, outer_assignments, *, seed=20260929):
    from taiwan_simulation.buffer_variograms import diagnose_variograms
    xy = validate_geometry(grids)
    outputs = {name: [] for name in ('bins', 'fits', 'summary')}

    def append(part, kind, outer_k=-1, fold=-1, indices=None):
        indices = np.arange(len(grids)) if indices is None else indices
        result = diagnose_variograms(xy[indices], part, n_bins=8, max_distance_fraction=.6, min_pairs=10)
        for name, table in result.items():
            table = table.copy()
            table.insert(0, 'kind', kind)
            table.insert(1, 'outer_folds', outer_k)
            table.insert(2, 'excluded_outer_fold', fold)
            table.insert(3, 'n_grids', len(indices))
            outputs[name].append(table)

    append(coefficients, 'full_raw_exploratory')
    append(elevation_residuals(coefficients, elevation), 'full_elevation_residual_exploratory')
    for k, labels in outer_assignments.groupby('outer_folds', sort=True):
        by_grid = labels.set_index('grid_id').loc[grids.grid_id, 'outer_fold'].to_numpy()
        for fold in range(int(k)):
            index = np.flatnonzero(by_grid != fold)
            residual = elevation_residuals(coefficients.iloc[index], np.asarray(elevation)[index])
            append(residual, 'outer_training_elevation_residual', int(k), fold, index)
    return {name: pd.concat(parts, ignore_index=True) for name, parts in outputs.items()}


def diagnostic_plots(grids, geometry, variograms, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), constrained_layout=True)
    for ax, k in zip(axes, (3, 5, 8, 10)):
        labels = geometry['outer_assignments'].query('outer_folds == @k')
        if labels.empty:
            ax.axis('off'); continue
        colors = labels.set_index('grid_id').loc[grids.grid_id, 'outer_fold']
        ax.scatter(grids.x_km, grids.y_km, c=colors, cmap='tab10', s=24)
        ax.set(title=f'Outer K={k}', xlabel='TWD97 x (km)', ylabel='TWD97 y (km)', aspect='equal')
    fig.suptitle('Coordinate-only folds; buffers not shown')
    fig.savefig(output/'outer_fold_maps.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    for row, kind in enumerate(('full_raw_exploratory', 'full_elevation_residual_exploratory')):
        for ax, coefficient in zip(axes[row], COEFFICIENTS):
            bins = variograms['bins'].query('kind == @kind and coefficient == @coefficient')
            ax.plot(bins.lag_km, bins.semivariance, 'o-', markersize=3)
            ax.set(title=coefficient, xlabel='Distance (km)', ylabel='Semivariance')
            ax.grid(alpha=.2)
    axes[0, 0].set_ylabel('Raw estimate semivariance')
    axes[1, 0].set_ylabel('Elevation-residual semivariance')
    fig.suptitle('Exploratory empirical variograms (not independent-test evidence)')
    fig.savefig(output/'five_coefficient_variograms.png', dpi=150); plt.close(fig)


def run(prepared, predictors_path, output, *, model_kind='bound', seed=20260929,
        outer_counts=tuple(range(3, 11)), inner_counts=(2, 3, 4, 5),
        buffers=(0., 25., 30., 40., 50., 60., 75., 100., 125., 150., 200.),
        min_train=20):
    from taiwan_simulation import prepare_taiwan_cheng as tw
    import torch
    prepared, predictors_path, output = map(lambda p: Path(p).resolve(), (prepared, predictors_path, output))
    if output.exists():
        raise FileExistsError('Use a new diagnostic output; prior results are read-only.')
    values, grids, predictors, hashes = load_diagnostic_inputs(prepared, predictors_path)
    output.mkdir(parents=True)
    start = time.perf_counter()
    def status(stage):
        atomic_json(output/'run_status.json', dict(status='running', stage=stage, full_pipeline_executed=False))
        print(stage, flush=True)
    try:
        status('fresh_frozen_nn_for_diagnostics')
        torch.set_num_threads(4)
        nn_start = time.perf_counter()
        result = tw.infer_sequences(values, model_kind=model_kind, device='cpu')
        nn_seconds = time.perf_counter()-nn_start
        coefficients = pd.DataFrame(result['prediction'], columns=COEFFICIENTS)
        if not np.isfinite(coefficients.to_numpy()).all():
            raise ValueError('Nonfinite frozen NN estimates.')
        atomic_csv(output/'nn_coefficients.csv', pd.concat([grids, coefficients], axis=1))
        atomic_csv(output/'nn_checks.csv', pd.concat([grids[['grid_id']], result['checks']], axis=1))
        status('coordinate_only_nested_fold_buffer_sweep')
        geometry = geometry_diagnostics(grids, outer_counts=outer_counts, inner_counts=inner_counts,
                                        buffers=buffers, min_train=min_train, seed=seed)
        for name, table in geometry.items(): atomic_csv(output/f'{name}.csv', table)
        status('five_coefficient_exploratory_and_training_only_variograms')
        variograms = variogram_diagnostics(grids, coefficients, predictors.elevation_m,
                                         geometry['outer_assignments'], seed=seed)
        for name, table in variograms.items(): atomic_csv(output/f'variogram_{name}.csv', table)
        status('plotting_and_provenance_checks')
        diagnostic_plots(grids, geometry, variograms, output)
        for path, digest in hashes.items():
            if file_sha256(path) != digest:
                raise RuntimeError(f'Protected input changed during diagnostics: {path}')
        xy = validate_geometry(grids)
        distances = cdist(xy, xy); np.fill_diagonal(distances, np.inf)
        source_files = [Path(__file__), SOURCE_ROOT/'taiwan_simulation/buffer_variograms.py',
                        SOURCE_ROOT/'taiwan_simulation/cheng_spatial_cv.py',
                        SOURCE_ROOT/'elevation_gp_analysis.py']
        report = dict(status='completed_diagnostics_only', contract=CONTRACT, n_grids=50,
            years=list(YEARS), coefficient_names=list(COEFFICIENTS), model=result['model'],
            completed_utc=tw.shared.utc_now(), seed=seed, min_train=min_train,
            outer_counts=list(outer_counts), inner_counts=list(inner_counts), buffers_km=list(buffers),
            n_geometry_combinations=len(geometry['summary']),
            n_feasible_combinations=int(geometry['summary'].all_folds_meet_min_train.sum()),
            nearest_grid_distance_km=dict(min=float(distances.min(axis=1).min()),
                                         median=float(np.median(distances.min(axis=1))),
                                         max=float(distances.min(axis=1).max())),
            nn_wall_seconds=nn_seconds, diagnostic_wall_seconds=time.perf_counter()-start,
            full_pipeline_executed=False, production_config_changed=False, nn_retrained=False,
            formal_model_replaced=False, selected_production_configuration=None,
            not_run=['GP/FFS model comparison', 'formal OOF scoring', 'return levels', 'bootstrap', 'global CV'],
            range_scope='Full-data fits exploratory. Training-only fits exclude each outer test fold but use its unbuffered training subset; these are not final selected-GP residuals.',
            restrictions=['No buffer chosen by minimum CV RMSE.',
                'No range from all 50 responses may be presented as training-only calibration.',
                'Effective ranges are descriptive; short unresolved ranges, boundary fits and missing sill are flagged.',
                'Geometry mirrors current nested_cv only when max_train >= 50; inner folds are rebuilt after outer buffering.',
                'A geometrically feasible buffer does not certify residual independence or global transfer.',
                'Any response-informed design changes make these data development data, not an untouched final test.'],
            input_sha256=hashes, source_sha256={str(p):file_sha256(p) for p in source_files},
            output_sha256={p.name:file_sha256(p) for p in output.iterdir() if p.suffix in ('.csv', '.png')})
        atomic_json(output/'run_report.json', report)
        atomic_json(output/'run_status.json', dict(status=report['status'], full_pipeline_executed=False))
        return report
    except BaseException as error:
        atomic_json(output/'run_status.json', dict(status='failed', error_type=type(error).__name__,
            error=str(error), full_pipeline_executed=False))
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'predictors', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--model', choices=['original', 'bound'], default='bound')
    p.add_argument('--seed', type=int, default=20260929)
    p.add_argument('--outer-folds', type=int, nargs='+', default=list(range(3, 11)))
    p.add_argument('--inner-folds', type=int, nargs='+', default=[2, 3, 4, 5])
    p.add_argument('--buffers', type=float, nargs='+', default=[0,25,30,40,50,60,75,100,125,150,200])
    p.add_argument('--min-train', type=int, default=20)
    a = p.parse_args(argv)
    report = run(a.prepared, a.predictors, a.output, model_kind=a.model, seed=a.seed,
        outer_counts=a.outer_folds, inner_counts=a.inner_folds, buffers=a.buffers, min_train=a.min_train)
    print(json.dumps({k:report[k] for k in ('status', 'n_geometry_combinations',
        'n_feasible_combinations', 'diagnostic_wall_seconds')}, indent=2))


if __name__ == '__main__':
    main()
