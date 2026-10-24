"""Five 3-by-5 spatial figures from one verified paired full-50-year simulation.

Columns are evaluation years, NOT different observation lengths or refits.
Each figure is one quantity; rows are known simulated truth, MLE, and BOUND NN.
No model fitting, coefficient clipping, spatial interpolation or GRID deletion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

from research_workflows import COEFFICIENTS, YEARS, REFERENCE_YEAR, TIME_SCALE_YEARS

YEAR_NUMBERS = (10, 20, 30, 40, 50)
QUANTITIES = ('mu_t', 'beta_mu', 'eta_t', 'beta_sigma', 'xi0')
LABELS = (r'$\mu(t)$ (°C)', r'$\beta_\mu$ (°C / decade)',
          r'$\eta(t)=\log\sigma(t)$', r'$\beta_\sigma$ (per decade)', r'$\xi_0$')
METHODS = ('truth', 'mle', 'nn')
FIXED_ROWS = (1, 3, 4)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parameters_at_years(coefficients, year_numbers=YEAR_NUMBERS):
    """Return GRID x year x quantity; eta0 becomes eta(t), not sigma(t)."""
    theta = np.asarray(coefficients, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != 5 or len(theta) == 0 or not np.isfinite(theta).all():
        raise ValueError('Expected finite nonempty GRID-by-five original-scale coefficients')
    numbers = np.asarray(year_numbers)
    if (numbers.ndim != 1 or len(numbers) != 5 or not np.issubdtype(numbers.dtype, np.integer)
            or np.any(numbers < 1) or np.any(numbers > len(YEARS)) or len(set(numbers)) != 5):
        raise ValueError('Need five distinct one-based year numbers within the full 50-year sequence')
    years = np.asarray(YEARS)[numbers - 1]
    t = (years - REFERENCE_YEAR) / TIME_SCALE_YEARS
    result = np.repeat(theta[:, None, :], len(years), axis=1)
    result[:, :, 0] = theta[:, 0, None] + theta[:, 1, None] * t
    result[:, :, 2] = theta[:, 2, None] + theta[:, 3, None] * t
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite reconstructed parameter; no clipping or GRID filtering')
    return result, years


def _attempt(root, raw_path, replicate, method):
    """Relocate a recorded D:/F: attempt only within this comparison folder."""
    recorded = str(raw_path).replace('\\', '/').rstrip('/')
    name = recorded.rsplit('/', 1)[-1]
    if not re.fullmatch(r'attempt_\d{3,}', name):
        raise ValueError('Invalid attempt name in timing comparison')
    relative = Path('replicates') / f'replicate_{replicate:03d}' / method / name
    if not recorded.endswith('/' + relative.as_posix()) and recorded != relative.as_posix():
        raise ValueError('Recorded path does not match requested replicate and estimator')
    folder = (root / relative).resolve()
    if not folder.is_relative_to(root):
        raise ValueError('Attempt escapes comparison directory')
    return folder


def load_pair(root, replicate=1, stage='estimation'):
    """Only a completed paired calibrated simulation is eligible for the figures."""
    if stage not in ('estimation', 'gp_oof'):
        raise ValueError('stage must be estimation or gp_oof')
    if not isinstance(replicate, int) or replicate < 1:
        raise ValueError('replicate must be a positive integer')
    root = Path(root).resolve()
    protocol = _read(root / 'comparison_protocol.json')
    if protocol['years'] != list(YEARS):
        raise ValueError('These figures require the complete 1976--2025 contract')
    timings = pd.read_csv(root / 'timing_comparison.csv')
    selected = timings[timings.replicate.eq(replicate)]
    if len(selected) != 1:
        raise ValueError('Requested replicate has no unique completed paired comparison')
    row = selected.iloc[0]
    arrays = {}; reports = {}; identities = {}; hashes = {}; grids = None
    for method in ('nn', 'mle'):
        folder = _attempt(root, row[f'{method}_output'], replicate, method)
        report = _read(folder / 'run_report.json')
        if (not report.get('workflow_completed') or report.get('data_role') != 'calibrated_simulation'
                or report.get('estimator') != method or report.get('replicate') != replicate
                or report.get('metric_reference') != 'known_simulated_truth'
                or report.get('years') != list(YEARS)):
            raise ValueError('Need completed paired simulation outputs, not real data or old test results')
        filename = f'{method}_coefficients.csv' if stage == 'estimation' else 'oof_coefficients.csv'
        used = (filename, 'truth.npy', 'annual_maxima.npy')
        for name in used:
            digest = report['output_sha256'].get(name)
            if not digest or _sha(folder / name) != digest:
                raise ValueError(f'Missing or changed verified source: {method}/{name}')
        frame = pd.read_csv(folder / filename, dtype={'grid_id': str})
        required = ['grid_id', 'x_km', 'y_km', *COEFFICIENTS]
        if not set(required).issubset(frame.columns) or frame.grid_id.duplicated().any():
            raise ValueError('Missing/duplicate GRID or projected coordinates')
        if not np.isfinite(frame[['x_km', 'y_km', *COEFFICIENTS]].to_numpy()).all():
            raise ValueError('Nonfinite GRID or coefficient; no silent deletion')
        current_grids = frame[['grid_id', 'x_km', 'y_km']].reset_index(drop=True)
        if grids is not None and not grids.equals(current_grids):
            raise ValueError('NN and MLE GRID order/coordinates differ')
        grids = current_grids
        truth = np.load(folder / 'truth.npy', allow_pickle=False)
        if truth.shape != (len(grids), 5):
            raise ValueError('Truth and GRID count differ')
        if 'truth' in arrays and not np.array_equal(arrays['truth'], truth):
            raise ValueError('NN and MLE simulated truth differs')
        arrays['truth'] = truth
        arrays[method] = frame[list(COEFFICIENTS)].to_numpy(float)
        reports[method] = report
        identities[method] = report.get('sample_id')
        hashes[method] = {name: report['output_sha256'][name] for name in used}
    if not identities['nn'] or identities['nn'] != identities['mle']:
        raise ValueError('NN and MLE are not the same simulation sample')
    if (hashes['nn']['annual_maxima.npy'] != hashes['mle']['annual_maxima.npy']
            or reports['nn']['coordinate_crs'] != reports['mle']['coordinate_crs']):
        raise ValueError('Different observations or coordinate systems')
    if reports['nn']['model'].get('sha256') != protocol['bound_model_sha256']:
        raise ValueError('Unexpected NN checkpoint')
    return dict(coefficients=arrays, grids=grids, metadata=dict(replicate=replicate, stage=stage,
        source_hashes=hashes, sample_id=identities['nn'], coordinate_crs=reports['nn']['coordinate_crs'],
        comparison_root=str(root), data_role='calibrated_simulation', fitted_years=list(YEARS)))


def make_figures(pair, year_numbers=YEAR_NUMBERS):
    """Five figures, each 3 methods x 5 years; identical color limits per quantity."""
    values = {}; years = None
    for method in METHODS:
        values[method], years = parameters_at_years(pair['coefficients'][method], year_numbers)
    grids = pair['grids']
    if any(len(v) != len(grids) for v in values.values()):
        raise ValueError('GRID counts differ')
    xy = grids[['x_km', 'y_km']].to_numpy(float)
    if not np.isfinite(xy).all() or len(xy) == 0:
        raise ValueError('Finite projected GRID coordinates required')
    extent = []
    for dim in range(2):
        lo, hi = xy[:, dim].min(), xy[:, dim].max()
        pad = max((hi-lo)*.035, 1.)
        extent.append((lo-pad, hi+pad))
    stage = pair['metadata']['stage']
    row_names = ['Simulated truth', 'MLE' if stage=='estimation' else 'MLE + GP OOF',
                 'BOUND NN' if stage=='estimation' else 'BOUND NN + GP OOF']
    figures = {}; metrics = []; color_limits = {}
    for q, name in enumerate(QUANTITIES):
        lo = min(v[:, :, q].min() for v in values.values())
        hi = max(v[:, :, q].max() for v in values.values())
        if lo == hi:
            pad = max(abs(lo)*.01, .001); lo -= pad; hi += pad
        color_limits[name] = [float(lo), float(hi)]
        norm = Normalize(vmin=lo, vmax=hi)
        fig, axes = plt.subplots(3, 5, figsize=(16, 10), sharex=True, sharey=True, layout='constrained')
        figures[name] = fig
        constant = ' | Time-constant coefficient: repeated across years' if q in FIXED_ROWS else ''
        fig.suptitle(f'{LABELS[q]} | Replicate {pair["metadata"]["replicate"]}{constant}\n'
                     'All estimates use the full 50-year sequence (1976–2025)', fontsize=13, color='black')
        for r, method in enumerate(METHODS):
            for col, (number, year) in enumerate(zip(year_numbers, years)):
                ax = axes[r, col]
                artist = ax.scatter(xy[:, 0], xy[:, 1], c=values[method][:, col, q], norm=norm,
                    cmap='viridis', s=max(1., min(16., 10000/len(xy))), marker='s', linewidths=0,
                    rasterized=True)
                ax.set_xlim(*extent[0]); ax.set_ylim(*extent[1]); ax.set_aspect('equal', adjustable='box')
                ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(3))
                ax.tick_params(labelsize=8, colors='black')
                if r == 0: ax.set_title(f'Year {number} ({year})', fontsize=11, color='black')
                if col == 0: ax.set_ylabel(f'{row_names[r]}\nNorthing (km)', fontsize=11, color='black')
                if r == 2: ax.set_xlabel('Easting (km)', fontsize=10, color='black')
                if method != 'truth':
                    d = values[method][:, col, q] - values['truth'][:, col, q]
                    rmse = float(np.sqrt(np.mean(d*d)))
                    metrics.append(dict(replicate=pair['metadata']['replicate'], stage=stage,
                        method=method, quantity=name, year_number=int(number), year=int(year),
                        n_grids=len(d), RMSE=rmse, MAE=float(np.mean(np.abs(d))), Bias=float(d.mean())))
                    ax.text(.02,.02,f'RMSE = {rmse:.4g}',transform=ax.transAxes,fontsize=8,color='black',
                        va='bottom',bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=1.5))
        bar = fig.colorbar(artist, ax=axes, location='right', fraction=.025, pad=.012, shrink=.9)
        bar.set_label(LABELS[q], fontsize=11, color='black'); bar.ax.tick_params(labelsize=9, colors='black')
    metadata = dict(pair['metadata'], year_numbers=list(year_numbers), calendar_years=years.tolist(),
        quantities=list(QUANTITIES), rows=list(METHODS), layout='3 rows x 5 columns per quantity',
        shared_color_limits=color_limits, full_50_year_fit=True,
        time_constant_quantities=[QUANTITIES[i] for i in FIXED_ROWS],
        interpretation='Known calibrated truth, not unknown real climate truth. GRID centers, no interpolation.',
        evaluation='Selected full-sequence evaluation years, not 10/20/30/40/50-year refits.',
        plotting_source_sha256=_sha(Path(__file__)))
    return figures, pd.DataFrame(metrics), metadata


def save_figures(figures, metrics, metadata, output, *, dpi=300):
    """Write only derived figures/metrics; do not touch models or experiment outputs."""
    if set(figures) != set(QUANTITIES) or dpi < 50:
        raise ValueError('Expected five parameter figures and dpi >= 50')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    files = []
    for name in QUANTITIES:
        for extension in ('png', 'svg'):
            path = output / f'{name}_truth_mle_nn_3x5.{extension}'
            figures[name].savefig(path, dpi=dpi, facecolor='white')
            files.append(path)
    metric_path = output / 'year_slice_metrics.csv'
    metrics.to_csv(metric_path, index=False); files.append(metric_path)
    report = dict(metadata, output_sha256={p.name:_sha(p) for p in files})
    (output / 'figure_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2,
        allow_nan=False), encoding='utf-8')
    return report
