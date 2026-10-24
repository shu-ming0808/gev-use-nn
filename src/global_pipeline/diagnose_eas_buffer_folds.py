"""EAS-only development diagnostics, never silently run an incomplete GP workflow.

Frozen 21D BOUND NN, exact AR6/LSM registry, audited regional LCC coordinates
for KMeans, WGS84 geodesic buffer distances. Raw and elevation-residual
variograms use a disclosed coordinate-only subsample; not final GP residuals.
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
import xarray as xr
from pyproj import Geod, Proj
from threadpoolctl import threadpool_limits
import torch

from global_pipeline.global_climate_regions import DEFAULT_OUTPUT, DEFAULT_BOUNDARIES, load_regions
from global_pipeline.export_global_beamer_maps import verified_output
from spatial_coordinates import project_lonlat_to_km
from taiwan_simulation.diagnose_buffer_folds import geometry_diagnostics, elevation_residuals
from taiwan_simulation.buffer_variograms import diagnose_variograms
from taiwan_simulation.prepare_taiwan_cheng import infer_sequences
from taiwan_simulation import match_annual_events as io
from research_workflows import COEFFICIENTS, YEARS, CONTRACT, validate_annual_sequences
from train_cheng_nn import atomic_csv, atomic_json, atomic_array, file_sha256

ROOT = SOURCE_ROOT.parent
NN_ROOT = Path('D:/論文資料/ERA5/cheng_nn_21d_1976_2025/run_20261011_01')
DATA_ROOT = Path('D:/論文資料/ERA5/global_predictors_1976_2025')
# Explicit regional research projection, NOT an official national CRS.
EAS_CRS = '+proj=lcc +lat_1=25 +lat_2=40 +lat_0=32.5 +lon_0=120 +datum=WGS84 +units=m +no_defs'


def geodesic_matrix(grids, progress=None):
    """All-pair WGS84 ellipsoid distances in km, float64; ~489 MB for EAS."""
    lon, lat = grids.longitude_180.to_numpy(), grids.latitude.to_numpy()
    geod = Geod(ellps='WGS84')
    matrix = np.zeros((len(grids), len(grids)), dtype=np.float64)
    for i in range(len(grids)-1):
        n = len(grids)-i-1
        distance = geod.inv(np.full(n, lon[i]), np.full(n, lat[i]), lon[i+1:], lat[i+1:])[2]/1000
        matrix[i, i+1:] = distance
        matrix[i+1:, i] = distance
        if progress is not None and i % 1000 == 0:
            progress(dict(distance_rows=i, distance_rows_total=len(grids)))
    return matrix


def projection_audit(grids, xy, distances, seed):
    rng = np.random.default_rng(seed)
    first, second = rng.integers(0, len(grids), size=(2, 100_000))
    valid = first != second
    first, second = first[valid], second[valid]
    error = np.linalg.norm(xy[first]-xy[second], axis=1)/distances[first, second]-1
    factors = Proj(EAS_CRS).get_factors(grids.longitude_180.to_numpy(), grids.latitude.to_numpy())
    local = np.concatenate([np.asarray(factors.meridional_scale)-1,
                            np.asarray(factors.parallel_scale)-1])
    report = dict(crs=EAS_CRS, units='km', n_random_pairs=len(error),
        pair_distance_relative_error_quantiles=np.quantile(error, [0,.01,.5,.99,1]).tolist(),
        max_abs_sampled_pair_relative_error=float(np.max(np.abs(error))),
        max_abs_all_grid_local_scale_error=float(np.max(np.abs(local))),
        distance_policy='KMeans and variograms: LCC; buffers: WGS84 geodesic. Pair audit is a sample, not an all-pair bound.')
    if max(report['max_abs_sampled_pair_relative_error'], report['max_abs_all_grid_local_scale_error']) > .05:
        raise ValueError('EAS projection exceeds predeclared 5% diagnostic distortion tolerance.')
    return report


def prepare(nn_root, data_root):
    grid_path, grid_hash = verified_output(DEFAULT_OUTPUT/'region_manifest.json', 'era5_ar6_grid.nc')
    registry_path, registry_hash = verified_output(nn_root/'run_report.json', 'grid_index.csv')
    array_path, array_hash = verified_output(nn_root/'run_report.json', 'annual_maxima_land.npy')
    report = json.loads((nn_root/'run_report.json').read_text(encoding='utf-8'))
    if report['years'] != list(YEARS):
        raise ValueError('Require 1976-2025 in the verified original annual cache.')
    eas = next(r for r in load_regions(DEFAULT_BOUNDARIES) if r.acronym == 'EAS')
    with xr.open_dataset(grid_path, engine='scipy') as grid:
        mask = (grid.ar6_region_id.values == eas.id) & (grid.spatial_cv_eligible.values == 1)
        rr, cc = np.where(mask)
        ids = [f'G{r:03d}_{c:04d}' for r,c in zip(rr,cc)]
    registry = pd.read_csv(registry_path, dtype={'grid_id':str})
    if registry.grid_id.duplicated().any():
        raise ValueError('Duplicate source registry.')
    indices = registry.set_index('grid_id').index.get_indexer(ids)
    if np.any(indices < 0):
        raise ValueError('EAS GRID absent from annual cache.')
    grids = registry.iloc[indices].reset_index(drop=True)
    if not grids.lsm.gt(.5).all():
        raise ValueError('EAS registry must be land only.')
    values = validate_annual_sequences(np.asarray(YEARS),
        np.load(array_path, mmap_mode='r', allow_pickle=False)[indices])
    xy = project_lonlat_to_km(grids.longitude_180, grids.latitude, target_crs=EAS_CRS)
    grids[['x_km','y_km']] = xy
    height = data_root/'static/era5_surface_geopotential_global_025.nc'
    with io.open_nc(height) as ds:
        if getattr(ds['z'], 'units', '') not in ('m**2 s**-2', 'm2 s-2', 'm^2 s^-2'):
            raise ValueError('Unexpected geopotential units.')
        elevation = io.gather(ds['z'], io.spatial_indices(ds, grids), len(grids))/9.80665
    if not np.isfinite(elevation).all():
        raise ValueError('Missing EAS elevation.')
    hashes = {str(grid_path):grid_hash, str(registry_path):registry_hash,
              str(array_path):array_hash, str(height):file_sha256(height)}
    return grids, values, elevation, hashes


def variograms(grids, coefficients, elevation, assignments, *, seed, maximum=1500):
    """Whole-region exploratory and each outer-training-only mean removal."""
    tables = {name:[] for name in ('bins','fits','summary')}
    xy = grids[['x_km','y_km']].to_numpy()
    rng = np.random.default_rng(seed)

    def one(indices, kind, k=-1, fold=-1, residual=True):
        # Fit mean only to the supplied population, then sample pairs by points.
        values = coefficients.iloc[indices]
        if residual:
            values = elevation_residuals(values, elevation[indices])
        pick = np.sort(rng.choice(len(indices), min(maximum,len(indices)), replace=False))
        result = diagnose_variograms(xy[indices[pick]], values.iloc[pick].reset_index(drop=True),
                                    n_bins=20, max_distance_fraction=.6, min_pairs=30)
        for name, table in result.items():
            table = table.assign(kind=kind, outer_folds=k, excluded_outer_fold=fold,
                population_grids=len(indices), sampled_grids=len(pick))
            tables[name].append(table)

    one(np.arange(len(grids)), 'full_raw_exploratory', residual=False)
    one(np.arange(len(grids)), 'full_elevation_residual_exploratory')
    for k, part in assignments.groupby('outer_folds'):
        labels = part.set_index('grid_id').loc[grids.grid_id, 'outer_fold'].to_numpy()
        for fold in range(k):
            one(np.flatnonzero(labels != fold), 'outer_training_elevation_residual', k, fold)
    return {name:pd.concat(parts,ignore_index=True) for name,parts in tables.items()}


def run(output, *, nn_root=NN_ROOT, data_root=DATA_ROOT, seed=20260929,
        outer_counts=(4,5,6), inner_counts=(3,5), buffers=(0,50,100,200,300,500,750,1000), min_train=100):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new diagnostic output directory.')
    output.mkdir(parents=True)
    started = time.perf_counter()
    def status(stage, **extra):
        atomic_json(output/'run_status.json', dict(status='running', stage=stage,
            elapsed_seconds=time.perf_counter()-started, full_pipeline_executed=False, **extra))
        print(stage, extra, flush=True)
    try:
        status('validate_and_extract_eas_raw_sequences')
        grids, values, elevation, hashes = prepare(Path(nn_root), Path(data_root))
        atomic_csv(output/'grid_index.csv', grids)
        atomic_array(output/'annual_maxima.npy', values)
        atomic_array(output/'years.npy', np.asarray(YEARS))
        atomic_csv(output/'diagnostic_elevation.csv', pd.DataFrame({'grid_id':grids.grid_id,'elevation_m':elevation}))
        status('fresh_frozen_bound_nn')
        torch.set_num_threads(4)
        nn_start = time.perf_counter()
        inferred = infer_sequences(values, model_kind='bound', device='cpu')
        nn_seconds = time.perf_counter()-nn_start
        coefficients = pd.DataFrame(inferred['prediction'], columns=COEFFICIENTS)
        atomic_csv(output/'nn_coefficients.csv', pd.concat([grids,coefficients],axis=1))
        atomic_csv(output/'nn_checks.csv', pd.concat([grids[['grid_id']],inferred['checks']],axis=1))
        status('geodesic_distances')
        distances = geodesic_matrix(grids, lambda item: status('geodesic_distances', **item))
        audit = projection_audit(grids, grids[['x_km','y_km']].to_numpy(), distances, seed)
        atomic_json(output/'projection_audit.json', audit)
        status('nested_geometry_sweep')
        with threadpool_limits(limits=4):
            geometry = geometry_diagnostics(grids, outer_counts=outer_counts, inner_counts=inner_counts,
                buffers=buffers, min_train=min_train, seed=seed, distance_matrix=distances,
                progress=lambda item: status('nested_geometry_sweep', **item))
        for name, table in geometry.items():
            atomic_csv(output/f'{name}.csv', table)
        del distances
        status('five_coefficient_variograms')
        tables = variograms(grids,coefficients,elevation,geometry['outer_assignments'],seed=seed)
        for name, table in tables.items():
            atomic_csv(output/f'variogram_{name}.csv',table)
        report = dict(status='completed_diagnostics_only', region='EAS', n_grids=len(grids),
            contract=CONTRACT, years=list(YEARS), seed=seed, min_train=min_train,
            kmeans_n_init=100, native_threads=4, outer_counts=list(outer_counts),
            inner_counts=list(inner_counts), buffers_km=list(buffers), projection=audit,
            n_combinations=len(geometry['summary']),
            n_feasible=int(geometry['summary'].all_folds_meet_min_train.sum()),
            nn_seconds=nn_seconds, diagnostic_seconds=time.perf_counter()-started,
            full_pipeline_executed=False, nn_retrained=False, formal_model_replaced=False,
            full_pipeline_blockers=['EAS 7,822-GRID complete weather/static predictor table not yet prepared',
                'Regional GP/FFS needs the same projection and geodesic-buffer contract; Taiwan-only runner cannot be reused unchanged'],
            variogram_scope='Exploratory and outer-training-only elevation OLS residuals, up to 1,500 coordinate-random points per subset; projected LCC pair distances; not final GP residuals.',
            input_sha256=hashes, model=inferred['model'],
            source_sha256={str(Path(__file__)):file_sha256(__file__),
                str(SOURCE_ROOT/'taiwan_simulation/diagnose_buffer_folds.py'):file_sha256(SOURCE_ROOT/'taiwan_simulation/diagnose_buffer_folds.py')})
        atomic_json(output/'run_report.json',report)
        atomic_json(output/'run_status.json',dict(status=report['status'],full_pipeline_executed=False))
        return report
    except BaseException as error:
        atomic_json(output/'run_status.json',dict(status='failed',error_type=type(error).__name__,
            error=str(error),full_pipeline_executed=False,elapsed_seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.output)
