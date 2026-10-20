"""ERA5 Taiwan pilot for the frozen Cheng NN; no training or trend selection.

Keep the original global GRID identifiers and use a documented country polygon
plus LSM > .5. The main-island default matches the historical study scope, not
its old 0.05-degree connectivity rule. Real-data estimates are NOT known truth.
"""
from __future__ import annotations

# Keep direct-script execution independent of the working directory.
import sys as _sys
from pathlib import Path as _SourcePath
_SOURCE_ROOT = _SourcePath(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import argparse
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import torch

from global_pipeline import infer_global_cheng_nn as shared
from compare_cheng_nn_reparameterization import (
    DECODER_ID, MARGIN, DEFAULT_OUTPUT as BOUND_DIRECTORY, predict_sample_coefficients,
)
from research_workflows import CONTRACT, COEFFICIENTS, YEARS, REFERENCE_YEAR, TIME_SCALE_YEARS, TEMPORAL_POLICY
from spatial_coordinates import add_projected_km_columns
from train_cheng_nn import atomic_array, atomic_csv, atomic_json, file_sha256


DEFAULT_BOUNDARY = shared.PROJECT_ROOT / 'data/shapefile/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp'
DEFAULT_OUTPUT = shared.PROJECT_ROOT / 'results/simulation/taiwan_validation_era5_1976_2025'
MODELS = {
    'original': (shared.CHECKPOINT, shared.CHECKPOINT_HASH),
    'bound': (BOUND_DIRECTORY / 'reparameterized/best.pt',
              '03890b4877a40cf93822877d09231c003c920f34c68ed911b1da481ce396318f'),
}


def taiwan_geometry(path, *, scope='main-island'):
    """Select TWN, then optionally its largest polygon in the metric CRS.

    Natural Earth's cartographic boundary is not a legal boundary. Grid-centre
    inclusion may exclude coastal cells; record this exact domain convention.
    """
    if scope not in {'main-island', 'all-islands'}:
        raise ValueError('Unknown Taiwan domain scope.')
    frame = gpd.read_file(path)
    if frame.crs is None or 'ADM0_A3' not in frame:
        raise ValueError('Boundary needs a CRS and the ADM0_A3 country identifier.')
    country = frame.loc[frame.ADM0_A3.eq('TWN')].to_crs('EPSG:4326')
    if len(country) != 1:
        raise ValueError('Expected exactly one TWN country feature.')
    geometry = country.geometry.iloc[0]
    if geometry.is_empty or not geometry.is_valid or geometry.geom_type not in {'Polygon', 'MultiPolygon'}:
        raise ValueError('Taiwan geometry must be a valid nonempty polygon.')
    if scope == 'main-island' and geometry.geom_type == 'MultiPolygon':
        parts = gpd.GeoSeries(list(geometry.geoms), crs='EPSG:4326')
        geometry = parts.iloc[int(np.argmax(parts.to_crs('EPSG:3826').area.to_numpy()))]
    return geometry


def geographic_mask(grids, geometry):
    points = shapely.points(grids.longitude_180.to_numpy(), grids.latitude.to_numpy())
    return np.asarray(shapely.covers(geometry, points), dtype=bool)


def boundary_hashes(path):
    path = Path(path).resolve()
    files = [path]
    if path.suffix.lower() == '.shp':
        files = [path.with_suffix(ext) for ext in ('.shp', '.shx', '.dbf', '.prj')]
    return {str(p): file_sha256(p) for p in files}


def infer_sequences(values, *, model_kind='original', device='cpu', batch_size=2048,
                    train_directory=shared.TRAIN_DIRECTORY):
    """Shared pilot/replicate inference; never receives simulated truth labels.

    Recomputes sample median/IQR, 21D features and the support decoder for EACH
    sequence. The decoder never reuses the real-data A or location/scale.
    """
    if batch_size <= 0 or model_kind not in MODELS:
        raise ValueError('Invalid inference model/batch size.')
    checkpoint, expected_hash = MODELS[model_kind]
    timing = {}
    with shared.timed(timing, 'model_load_seconds', device):
        ns, model, scaler, payload = shared.load_frozen_model(checkpoint, expected_hash, device)
        if model_kind == 'bound':
            if (payload.get('decoder') != DECODER_ID or payload.get('support_margin') != MARGIN
                    or payload.get('requires_observed_sequence_at_inference') is not True
                    or payload.get('latent_coefficient_names') != ['mu0','beta_mu','delta','beta_sigma','xi0']):
                raise ValueError('Experimental support-decoder contract mismatch.')
            for source, digest in payload['source_hashes'].items():
                if file_sha256(source) != digest:
                    raise ValueError(f'Frozen decoder source changed: {Path(source).name}')
    with shared.timed(timing, 'feature_transform_seconds'):
        x, scaling = shared.transform_sequences(values)
        z = (values-scaling[:, :1])/scaling[:, 1:2]
    with shared.timed(timing, 'nn_and_decoder_seconds', device):
        standardized = predict_sample_coefficients(
            model, x, scaler, z, (np.asarray(YEARS)-REFERENCE_YEAR)/TIME_SCALE_YEARS,
            device, model_kind == 'bound', margin=MARGIN, batch_size=batch_size)
    with shared.timed(timing, 'inverse_scaling_seconds'):
        prediction = ns['inverse_sample_standardization'](standardized, scaling)
    with shared.timed(timing, 'diagnostic_seconds'):
        low, high = shared.training_feature_envelope(train_directory, payload['simulation_metadata'])
        checks = shared.diagnostic_table(prediction, values, x, low, high,
                                         payload['simulation_metadata']['parameter_ranges'])
    timing['nn_pipeline_seconds'] = sum(timing[k] for k in
        ('feature_transform_seconds', 'nn_and_decoder_seconds', 'inverse_scaling_seconds'))
    return dict(prediction=prediction, standardized=standardized, features=x, scaling=scaling,
                checks=checks, timings=timing, model=dict(kind=model_kind, path=str(checkpoint),
                sha256=expected_hash, best_epoch=int(payload['best_epoch']),
                is_experimental=model_kind == 'bound', formal_model_replaced=False))


def coefficient_figures(frame, output):
    """Descriptive real-data maps/histograms, never truth-error plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    units = ['C', 'C / decade', 'log scale', '/ decade', 'dimensionless']
    for kind in ('maps', 'histograms'):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        for ax, name, unit in zip(axes.flat, COEFFICIENTS, units):
            if kind == 'maps':
                plot = ax.scatter(frame.x_km, frame.y_km, c=frame[name], s=20, cmap='viridis')
                fig.colorbar(plot, ax=ax, label=unit)
                ax.set(xlabel='TWD97 x (km)', ylabel='TWD97 y (km)', aspect='equal')
            else:
                ax.hist(frame[name], bins='auto', edgecolor='white')
                ax.set(xlabel=unit, ylabel='GRID count')
            ax.set_title(r'$\eta_0$' if name == 'eta0' else name)
        axes.flat[-1].axis('off')
        axes.flat[-1].text(0, .85, f'Taiwan ERA5, 1976-2025\n{len(frame)} land GRID\nFrozen Cheng NN estimates\nNo trend significance test\nNot known parameter truth', va='top')
        fig.savefig(Path(output)/f'coefficient_{kind}.png', dpi=160)
        plt.close(fig)


def prepare(input_path, output, *, boundary=DEFAULT_BOUNDARY, scope='main-island',
            model_kind='original', device='auto', batch_size=2048, threads=4, plots=True):
    start = time.perf_counter()
    input_path, output, boundary = map(Path, (input_path, output, boundary))
    if output.exists():
        raise FileExistsError('Use a new run directory; previous Taiwan/global results are read-only.')
    if threads < 1:
        raise ValueError('threads must be positive.')
    torch.set_num_threads(threads)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    output.mkdir(parents=True)
    status_path = output/'run_status.json'
    atomic_json(status_path, dict(status='running', stage='taiwan_preparation', started_utc=shared.utc_now()))
    try:
        protected = boundary_hashes(boundary)
        protected[str(input_path.resolve())] = file_sha256(input_path)
        geometry = taiwan_geometry(boundary, scope=scope)
        timings = {}
        with shared.timed(timings, 'read_validate_taiwan_seconds'):
            values, grids = shared.read_annual_land(input_path,
                grid_selector=lambda g: geographic_mask(g, geometry))
            grids = add_projected_km_columns(grids, target_crs='EPSG:3826',
                                             lon_col='longitude_180', lat_col='latitude')
        result = infer_sequences(values, model_kind=model_kind, device=device, batch_size=batch_size)
        timings.update(result['timings'])
        protected[result['model']['path']] = result['model']['sha256']
        frame = pd.concat([grids, pd.DataFrame(result['prediction'], columns=COEFFICIENTS), result['checks']], axis=1)
        frame['sample_median_C'], frame['sample_iqr_C'] = result['scaling'].T
        frame['temporal_structure_status'] = 'full_five_coefficients_not_selected'
        with shared.timed(timings, 'save_seconds'):
            for name, array in [('annual_maxima.npy', values), ('inputs_21d.npy', result['features']),
                                ('sample_location_scale.npy', result['scaling']),
                                ('coefficients_original.npy', result['prediction']),
                                ('coefficients_standardized.npy', result['standardized']),
                                ('years.npy', np.asarray(YEARS))]:
                atomic_array(output/name, array)
            atomic_csv(output/'grid_index.csv', grids)
            atomic_csv(output/'coefficients_and_checks.csv', frame)
            atomic_csv(output/'review_required.csv', frame.loc[frame.needs_review])
            # Spatial responses have explicit five-coefficient names, not old mu/sigma/xi aliases.
            atomic_csv(output/'spatial_responses.csv', frame[['grid_id','longitude_180','latitude','x_km','y_km',*COEFFICIENTS]])
            atomic_csv(output/'coefficient_summary.csv', frame[list(COEFFICIENTS)].describe().T.reset_index(names='coefficient'))
            if plots:
                coefficient_figures(frame, output)
        for path, digest in protected.items():
            if file_sha256(path) != digest:
                raise RuntimeError(f'Protected source changed during preparation: {Path(path).name}')
        timings['total_wall_seconds_until_report'] = time.perf_counter()-start
        report = dict(status='completed', completed_stage='taiwan_preparation_and_nn_only',
            contract=CONTRACT, completed_utc=shared.utc_now(), data_role='real_data_pilot_not_simulated_truth',
            years=list(YEARS), scope=scope, boundary_source='Natural Earth admin_0, ADM0_A3=TWN',
            spatial_rule='grid centre covered by selected polygon AND LSM > 0.5; no interpolation',
            coordinate_crs='EPSG:3826', coordinate_units='km', input_path=str(input_path.resolve()),
            source_sha256=protected, model=result['model'], device=str(device),
            n_grids=len(values), coefficient_names=list(COEFFICIENTS),
            summary=dict(support_failure_grids=int(frame.n_support_violations.gt(0).sum()),
                         invalid_parameter_grids=int((~frame.parameters_valid_all_years).sum()),
                         review_required_grids=int(frame.needs_review.sum())),
            temporal_policy=TEMPORAL_POLICY,
            timings=timings, pending=['aligned_complete_predictors',
                'calibrated_buffer_and_folds', 'nested_spatial_cv_execution', 'spatial_simulation_and_bootstrap'],
            output_sha256={p.name:file_sha256(p) for p in output.iterdir() if p.suffix in {'.npy','.csv','.png'}})
        atomic_json(output/'run_report.json', report)
        atomic_json(status_path, dict(status='completed', completed_stage=report['completed_stage'], n_grids=len(values)))
        return report
    except BaseException as error:
        atomic_json(status_path, dict(status='failed', error_type=type(error).__name__,
                                     full_pipeline_completed=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=shared.DEFAULT_INPUT)
    parser.add_argument('--boundary', type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument('--scope', choices=['main-island','all-islands'], default='main-island')
    parser.add_argument('--model', choices=list(MODELS), default='original')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    report = prepare(args.input, args.output, boundary=args.boundary, scope=args.scope,
                     model_kind=args.model, device=args.device, threads=args.threads, plots=not args.no_plots)
    print(f"Taiwan NN preparation completed: {report['n_grids']} GRID, {report['timings']['nn_pipeline_seconds']:.3f} s NN pipeline.")
    print('Temporal selection, full spatial CV and bootstrap have NOT been executed.')


if __name__ == '__main__':
    main()
