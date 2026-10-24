"""Audited 50-year ERA5 land sequences -> frozen 21D NN, with wall timers.

No training, trend significance tests, spatial fitting, RL or data imputation.
Each run requires a new output directory. All input files remain read-only.
"""
from __future__ import annotations

# Keep direct-script execution independent of the working directory.
import sys as _sys
from pathlib import Path as _SourcePath
_SOURCE_ROOT = _SourcePath(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import time
PROCESS_START = time.perf_counter()  # Includes scientific imports in CLI timing.

import argparse
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import torch

from cheng_nn_evaluation import check_gev_predictions
from cheng_nn_simulation import QUANTILE_LEVELS
from compare_cheng_nn_inputs import make_features
from global_pipeline.prepare_era5_annual_maxima import open_nc
from research_workflows import COEFFICIENTS, REFERENCE_YEAR, TIME_SCALE_YEARS, YEARS, validate_annual_sequences
from train_cheng_nn import PROJECT_ROOT, atomic_array, atomic_csv, atomic_json, file_sha256, load_notebook_components

CHECKPOINT = PROJECT_ROOT / 'results/cheng_nn_17d/mixed_sampling_20261006/input21_seed20260929/best.pt'
CHECKPOINT_HASH = '534e3877a15826f5b443f71d74804edc8628821a3f0fec4b32b192c2162134b2'
TRAIN_DIRECTORY = PROJECT_ROOT / 'data/simulated/cheng_nn_mixed_100k_20261006/train'
DEFAULT_INPUT = Path('D:/論文資料/ERA5/1976-2004_global_annual_maxima/era5_t2m_annual_maxima_1976_2025_global.nc')
PERIODS = [(1976 + 10*i, 1985 + 10*i) for i in range(5)]
FEATURE_NAMES = [f'q_{q:g}' for q in QUANTILE_LEVELS] + [
    f'{a}_{b}_{stat}' for a, b in PERIODS for stat in ('median', 'iqr')]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


@contextmanager
def timed(timings, name, device='cpu'):
    synchronize(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        synchronize(device)
        timings[name] = time.perf_counter() - start


def read_annual_land(path, *, grid_selector=None):
    """Preserve existing GRID IDs and exact float32 NetCDF values, not CSV rounding."""
    import calendar
    with open_nc(path) as ds:
        years = np.asarray(ds['year'][:])
        if not np.array_equal(years, YEARS):
            raise ValueError('Require ordered 1976--2025, exactly 50 years.')
        variable = ds['annual_maximum_t2m']
        if variable.dimensions != ('year', 'latitude', 'longitude'):
            raise ValueError('Unexpected annual-maxima dimensions.')
        if str(getattr(variable, 'units', '')).lower() not in {'degrees_celsius', 'degree_celsius', 'degc', 'celsius'}:
            raise ValueError('Input must explicitly be Celsius; do not subtract 273.15 twice.')
        lat = np.asarray(ds['latitude'][:], dtype=float)
        lon = np.asarray(ds['longitude'][:], dtype=float)
        lsm = np.asarray(np.ma.filled(ds['lsm'][:], np.nan), dtype=float)
        if (lat.ndim != 1 or lon.ndim != 1 or lsm.shape != (len(lat), len(lon))
                or not np.isfinite(lat).all() or not np.isfinite(lon).all()
                or not np.isfinite(lsm).all() or np.any((lsm < 0) | (lsm > 1))
                or len(np.unique(lat)) != len(lat) or len(np.unique(lon)) != len(lon)
                or np.any(np.abs(lat) > 90) or np.any((lon < -180) | (lon >= 360))):
            raise ValueError('Invalid geographic coordinates or land-sea mask.')
        selected = np.flatnonzero(lsm.ravel() > .5)
        if not len(selected):
            raise ValueError('No land cells with LSM > 0.5.')
        rows, cols = np.divmod(selected, len(lon))
        grids = pd.DataFrame(dict(
            grid_id=[f'G{r:03d}_{c:04d}' for r,c in zip(rows, cols)],
            latitude=lat[rows], longitude=lon[cols], longitude_180=(lon[cols]+180)%360-180,
            lsm=lsm.ravel()[selected], is_pole=np.isclose(np.abs(lat[rows]), 90, atol=1e-10, rtol=0)))
        # Region selection happens before reading observations, using geography only.
        # The default remains the identical global land domain and GRID ordering.
        if grid_selector is not None:
            keep = np.asarray(grid_selector(grids.copy()))
            if keep.dtype != np.bool_ or keep.shape != (len(grids),):
                raise ValueError('grid_selector must return one boolean per land GRID.')
            selected = selected[keep]
            grids = grids.loc[keep].reset_index(drop=True)
            if not len(selected):
                raise ValueError('Geographic selection contains no land GRID.')
        values = np.empty((len(selected), 50), dtype=np.float32)
        for i, year in enumerate(years):
            values[:, i] = np.asarray(np.ma.filled(variable[i], np.nan)).ravel()[selected]
            counts = np.asarray(np.ma.filled(ds['valid_hour_count'][i], 0)).ravel()[selected]
            expected = (366 if calendar.isleap(int(year)) else 365) * 24
            if np.any(counts != expected):
                raise ValueError(f'Incomplete land-cell hourly coverage in {year}; no imputation.')
        validate_annual_sequences(years, values)
    return values, grids


def transform_sequences(values, chunk_size=4096):
    features = np.empty((len(values), 21), dtype=np.float32)
    scaling = np.empty((len(values), 2), dtype=np.float64)
    for start in range(0, len(values), chunk_size):
        block = validate_annual_sequences(YEARS, values[start:start+chunk_size])
        q25, q75 = np.quantile(block, [.25, .75], axis=1)
        location_scale = np.column_stack((np.median(block, axis=1), q75-q25))
        scaling[start:start+len(block)] = location_scale
        features[start:start+len(block)] = make_features(block, location_scale, 21)
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite 21D features; no silent replacement.')
    return features, scaling


def load_frozen_model(checkpoint, expected_hash, device):
    if file_sha256(checkpoint) != expected_hash:
        raise ValueError('Checkpoint SHA256 does not match the frozen selected model.')
    # Only deserialize the locally trained, hash-verified checkpoint.
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (payload['architecture'] != [21, 128, 128, 64, 5]
            or payload['input_shape'] != [21] or payload['dropout_p'] != 0
            or payload['coefficient_names'] != list(COEFFICIENTS)):
        raise ValueError('Wrong model architecture or coefficient order.')
    meta = payload['simulation_metadata']
    if (meta['years'] != list(YEARS) or meta['reference_year'] != REFERENCE_YEAR
            or meta['time_scale_years'] != TIME_SCALE_YEARS
            or meta['input_quantile_levels'] != list(QUANTILE_LEVELS)):
        raise ValueError('Training time/quantile contract mismatch.')
    for name in ('cheng_NN.ipynb', 'train_cheng_nn.py', 'compare_cheng_nn_inputs.py', 'cheng_nn_simulation.py'):
        # Training metadata can use Windows paths even when inference is Linux.
        matches = [(p,h) for p,h in payload['source_hashes'].items() if PureWindowsPath(p).name == name]
        actual = PROJECT_ROOT / ('notebooks' if name.endswith('.ipynb') else 'src') / name
        if len(matches) != 1 or file_sha256(actual) != matches[0][1]:
            raise ValueError(f'Frozen model implementation changed: {name}')
    with redirect_stdout(io.StringIO()):
        ns = load_notebook_components()
    ns['INPUT_FEATURE_NAMES'] = tuple(FEATURE_NAMES)
    model = ns['ChengGEVNet'](hidden_sizes=(128,128,64), dropout_p=0.)
    model.load_state_dict(payload['model_state'], strict=True)
    model.to(device).eval()
    scaler = ns['TargetScaler'](**payload['target_scaler'])
    if not np.isfinite(scaler.mean).all() or not np.isfinite(scaler.std).all():
        raise ValueError('Nonfinite saved target scaler.')
    synchronize(device)
    return ns, model, scaler, payload


def training_feature_envelope(directory, metadata):
    """Training only, not validation/test. Marginal bounds are warnings, not OOD proof."""
    directory = Path(directory)
    for name in ('annual_maxima.npy', 'sample_location_scale.npy'):
        if file_sha256(directory/name) != metadata['files_sha256'][name]:
            raise ValueError(f'Training reference hash mismatch: {name}')
    values = np.load(directory/'annual_maxima.npy', mmap_mode='r', allow_pickle=False)
    scaling = np.load(directory/'sample_location_scale.npy', mmap_mode='r', allow_pickle=False)
    low, high = np.full(21, np.inf), np.full(21, -np.inf)
    for start in range(0,len(values),4096):
        x = make_features(values[start:start+4096], scaling[start:start+4096], 21)
        low, high = np.minimum(low,x.min(axis=0)), np.maximum(high,x.max(axis=0))
    return low, high


def diagnostic_table(prediction, values, features, low, high, ranges, chunk_size=4096):
    parts = []
    for start in range(0,len(values),chunk_size):
        result = check_gev_predictions(prediction[start:start+chunk_size], values[start:start+chunk_size],
                                       np.asarray(YEARS), reference_year=REFERENCE_YEAR,
                                       xi_training_range=tuple(ranges['xi0']))
        parts.append(result.drop(columns=['sample_index', 'xi']))
    frame = pd.concat(parts, ignore_index=True)
    # A small tolerance avoids treating the constructionally zero pooled median
    # as OOD due only to floating-point roundoff.
    tolerance = 1e-6 * np.maximum(1., np.maximum(np.abs(low), np.abs(high)))
    frame['n_features_outside_train_minmax'] = ((features < low-tolerance) | (features > high+tolerance)).sum(axis=1)
    frame['features_outside_train_minmax'] = frame['n_features_outside_train_minmax'] > 0
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        mu_ratio = prediction[:,1] / np.exp(prediction[:,2])
    candidates = dict(mu0=prediction[:,0], beta_mu_over_sigma0=mu_ratio,
                      eta0=prediction[:,2], beta_sigma=prediction[:,3], xi0=prediction[:,4])
    flags = []
    for name, x in candidates.items():
        lower, upper = ranges[name]
        flag = ~np.isfinite(x) | (x < lower) | (x > upper)
        frame[f'{name}_outside_simulation_design'] = flag
        flags.append(flag)
    frame['any_coefficient_outside_simulation_design'] = np.logical_or.reduce(flags)
    frame['needs_review'] = (~frame['gev_valid_all_years'] | frame['features_outside_train_minmax']
                             | frame['any_coefficient_outside_simulation_design'])
    return frame


def run(input_path, output, checkpoint=CHECKPOINT, train_directory=TRAIN_DIRECTORY,
        device='auto', batch_size=2048, threads=4, process_start=None):
    start = time.perf_counter() if process_start is None else process_start
    run_start = time.perf_counter()
    input_path, output, checkpoint = map(lambda p: Path(p).resolve(), (input_path, output, checkpoint))
    if batch_size < 1 or threads < 1:
        raise ValueError('batch_size and threads must be positive.')
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output.exists():
        raise FileExistsError('Use a new run directory; existing results are never overwritten.')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device not in {'cpu','cuda'} or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested device is unavailable.')
    torch.set_num_threads(threads)
    output.mkdir(parents=True)
    state = dict(status='running', started_utc=utc_now(), input=str(input_path), output=str(output))
    timings = {'startup_imports_and_cli_seconds':run_start-start}
    def stage(name):
        state.update(stage=name, updated_utc=utc_now())
        atomic_json(output/'run_status.json', state)
        print(f'[{time.perf_counter()-start:.2f}s] {name}', flush=True)
    try:
        stage('load_frozen_model')
        with timed(timings, 'model_load_seconds', device):
            ns, model, scaler, payload = load_frozen_model(checkpoint, CHECKPOINT_HASH, device)
        stage('read_and_validate_50_years')
        before = (input_path.stat().st_size, input_path.stat().st_mtime_ns)
        with timed(timings, 'input_read_hash_validation_seconds'):
            input_hash = file_sha256(input_path)
            values, grids = read_annual_land(input_path)
        state['n_grids'] = len(values)
        stage('transform_21D')
        with timed(timings, 'feature_transform_seconds'):
            features, scaling = transform_sequences(values)
        stage('NN_inference')
        with timed(timings, 'nn_inference_seconds', device):
            standardized = ns['predict_standardized_coefficients'](model, features, scaler, device, batch_size=batch_size)
        stage('restore_Celsius_coefficients')
        with timed(timings, 'inverse_sample_scaling_seconds'):
            prediction = ns['inverse_sample_standardization'](standardized, scaling)
        stage('training_reference_bounds')
        with timed(timings, 'training_reference_seconds'):
            low, high = training_feature_envelope(train_directory, payload['simulation_metadata'])
        stage('GEV_support_and_domain_checks')
        with timed(timings, 'diagnostics_seconds'):
            diagnostics = diagnostic_table(prediction, values, features, low, high,
                                           payload['simulation_metadata']['parameter_ranges'])
            results = pd.concat([grids, pd.DataFrame(prediction,columns=COEFFICIENTS), diagnostics],axis=1)
            results['sample_median_C'] = scaling[:,0]
            results['sample_iqr_C'] = scaling[:,1]
        stage('save_prepared_data_and_predictions')
        with timed(timings, 'save_outputs_seconds'):
            for name, array in [('annual_maxima_land.npy',values),('inputs_21d.npy',features),
                                ('sample_location_scale.npy',scaling),('coefficients_original.npy',prediction)]:
                atomic_array(output/name,array)
            atomic_csv(output/'grid_index.csv', grids)
            atomic_csv(output/'coefficients_and_checks.csv', results)
            atomic_csv(output/'review_required.csv', results.loc[results['needs_review']])
        stage('verify_saved_artifacts')
        with timed(timings, 'output_hash_seconds'):
            hashes = {p.name:file_sha256(p) for p in output.iterdir() if p.suffix in {'.npy','.csv'}}
            if before != (input_path.stat().st_size,input_path.stat().st_mtime_ns):
                raise ValueError('Input file changed during the run.')
        timings['nn_pipeline_seconds'] = sum(timings[x] for x in (
            'feature_transform_seconds','nn_inference_seconds','inverse_sample_scaling_seconds'))
        timings['total_wall_seconds_until_report'] = time.perf_counter()-start
        summary = dict(
            n_grids=len(values), n_years=50, n_features=21,
            invalid_parameter_grids=int((~diagnostics.parameters_valid_all_years).sum()),
            observed_support_failure_grids=int((diagnostics.n_support_violations > 0).sum()),
            numerical_support_failure_grids=int((diagnostics.n_support_numerical_failures > 0).sum()),
            feature_range_warning_grids=int(diagnostics.features_outside_train_minmax.sum()),
            coefficient_range_warning_grids=int(diagnostics.any_coefficient_outside_simulation_design.sum()),
            review_required_grids=int(diagnostics.needs_review.sum()),
            pole_grids_preserved=int(grids.is_pole.sum()))
        report = dict(status='completed', completed_utc=utc_now(), summary=summary, timings=timings,
            input=dict(path=str(input_path),sha256=input_hash,units='degrees_Celsius',land_threshold='LSM > 0.5'),
            model=dict(path=str(checkpoint),sha256=CHECKPOINT_HASH,architecture=payload['architecture'],
                       best_epoch=payload['best_epoch'], retrained=False,
                       target_scaler={k:np.asarray(v).tolist() for k,v in payload['target_scaler'].items()}),
            years=list(YEARS),reference_year=REFERENCE_YEAR,time_scale_years=TIME_SCALE_YEARS,
            feature_names=FEATURE_NAMES,periods=PERIODS,coefficient_names=list(COEFFICIENTS),
            feature_bounds_training_min=low.tolist(),feature_bounds_training_max=high.tolist(),
            parameter_ranges=payload['simulation_metadata']['parameter_ranges'],
            runtime=dict(device=device,device_name=torch.cuda.get_device_name(0) if device=='cuda' else 'CPU',
                         torch_version=torch.__version__,batch_size=batch_size,cpu_threads=threads,warmup=False),
            timing_notes={
                'total':'Wall time from process timer before scientific imports through output writing/hashing; excludes final report/status serialization and process shutdown.',
                'nn_inference':'Batched forward + CPU/device transfers + saved target-scaler inverse; CUDA synchronized before/after. No warm-up run.',
                'nn_pipeline':'Feature transform + NN inference + sample inverse scaling; excludes file I/O, model load and diagnostics.',
                'benchmark':'One observed run, not an isolated repeated speed benchmark; no MLE comparison.'},
            limitations=['No ground truth or real-data RMSE.', 'Marginal training feature bounds are warnings, not a calibrated OOD test.',
                         'All flagged predictions retained, not clipped or automatically refitted.',
                         'No temporal structure selection, spatial CV or RL performed.',
                         'Pole grid aliases retained for provenance; spatial modelling must handle duplicates.'],
            output_sha256=hashes)
        atomic_json(output/'run_report.json',report)
        atomic_csv(output/'timing.csv',pd.DataFrame([dict(stage=k,seconds=v) for k,v in timings.items()]))
        state.update(status='completed',stage='completed',completed_utc=utc_now(),summary=summary,
                     total_seconds=timings['total_wall_seconds_until_report'])
        atomic_json(output/'run_status.json',state)
        print(json.dumps({'summary':summary,'timings':timings,'output':str(output)},ensure_ascii=False),flush=True)
        return report
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                     error_type=type(error).__name__,error=str(error),updated_utc=utc_now())
        atomic_json(output/'run_status.json',state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--output-directory',type=Path,default=None)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--batch-size',type=int,default=2048)
    parser.add_argument('--threads',type=int,default=4)
    args = parser.parse_args(argv)
    output = args.output_directory or PROJECT_ROOT/'results/global/nn_inference'/datetime.now().strftime('run_%Y%m%d_%H%M%S')
    run(args.input,output,device=args.device,batch_size=args.batch_size,threads=args.threads,process_start=PROCESS_START)


if __name__ == '__main__':
    main()
