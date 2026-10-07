"""Paired five-coefficient comparison of saved MLE and current small/large NNs.

No training, MLE refitting, RL evaluation, or checkpoint selection occurs here.
Inference is conditional on two fixed checkpoints, not all possible NN seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import torch

from cheng_nn_diagnostics import load_diagnostic_split
from cheng_nn_evaluation import check_gev_predictions
from cheng_nn_simulation import COEFFICIENT_NAMES
from train_cheng_nn import PROJECT_ROOT, atomic_array, atomic_csv, atomic_json, file_sha256, load_notebook_components

PARAMETERS = list(COEFFICIENT_NAMES)
SYMBOLS = [r'$\mu_0$', r'$\beta_\mu$', r'$\eta_0=\log\sigma_0$', r'$\beta_\sigma$', r'$\xi_0$']
UNITS = ['C', 'C / decade', 'log scale', 'per decade', 'unitless']
DOMAIN_NAMES = ('training_domain', 'wide_domain')


def holm_adjust(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Need finite p-values in [0,1].')
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1., np.maximum.accumulate((len(p)-np.arange(len(p))) * p[order]))
    return adjusted


def compare_errors(errors, repeats=5000):
    shapes = {np.asarray(e).shape for e in errors.values()}
    if len(shapes) != 1 or next(iter(shapes))[1:] != (5,):
        raise ValueError('Aligned N x 5 coefficient errors are required.')
    if next(iter(shapes))[0] < 2 or not all(np.isfinite(e).all() for e in errors.values()):
        raise ValueError('Do not silently exclude nonfinite predictions.')
    se = {k: np.asarray(e, dtype=float)**2 for k, e in errors.items()}
    rmse = {k: np.sqrt(v.mean(axis=0)) for k, v in se.items()}
    if np.any(rmse['MLE'] <= 0):
        raise ValueError('Cannot express improvements relative to zero MLE RMSE.')
    rows = []
    for j, name in enumerate(PARAMETERS):
        delta = se['Small'][:, j] - se['Large'][:, j]
        # H0: E[SE_small-SE_large] <= 0, H1: > 0. RMSE is monotone in MSE.
        p = (0. if delta.mean() > 0 else 1.) if np.std(delta) == 0 else float(
            ttest_rel(se['Small'][:, j], se['Large'][:, j], alternative='greater').pvalue)
        rng = np.random.default_rng(20261003+j)
        intervals = []
        for start in range(0, repeats, 100):
            ids = rng.integers(0, len(delta), size=(min(100, repeats-start), len(delta)))
            intervals.extend(np.sqrt(se['Small'][ids, j].mean(axis=1)) - np.sqrt(se['Large'][ids, j].mean(axis=1)))
        low, high = np.quantile(intervals, [.025, .975])
        rows.append(dict(coefficient=name, n_cases=len(delta), MLE_RMSE=rmse['MLE'][j],
                         Small_RMSE=rmse['Small'][j], Large_RMSE=rmse['Large'][j],
                         Small_improvement_percent=100*(1-rmse['Small'][j]/rmse['MLE'][j]),
                         Large_improvement_percent=100*(1-rmse['Large'][j]/rmse['MLE'][j]),
                         Small_minus_Large_MSE=delta.mean(), p_one_sided=p,
                         Small_minus_Large_RMSE_CI_low=low, Small_minus_Large_RMSE_CI_high=high))
    table = pd.DataFrame(rows)
    table['p_Holm'] = holm_adjust(table.p_one_sided)
    table['Large_significantly_better'] = (table.p_Holm < .05) & (table.Small_minus_Large_MSE > 0)
    return table


def sequence_hashes(values):
    return {hashlib.sha256(np.asarray(row, dtype=np.float64).tobytes()).hexdigest() for row in values}


def make_figure(errors, output):
    fig, axes = plt.subplots(5, 3, figsize=(10.8, 13), sharey='row')
    colors = ['#9aa0a6', '#3875ba', '#db792e']
    labels = ['MLE\n(constrained)', 'Small NN', 'Large NN']
    for i, model in enumerate(('MLE', 'Small', 'Large')):
        for j in range(5):
            ax = axes[j, i]
            bp = ax.boxplot([errors[model][:, j]], widths=.42,
                            patch_artist=True, whis=1.5,
                            medianprops={'color': 'black', 'linewidth': 1.6},
                            flierprops={'marker': '.', 'markersize': 2.5, 'alpha': .3})
            bp['boxes'][0].set_facecolor(colors[i])
            bp['boxes'][0].set_alpha(.7)
            ax.axhline(0., color='#555555', ls='--', lw=1)
            ax.set_xticks([])
            if j == 0:
                ax.set_title(labels[i], fontsize=14, pad=10)
            if i == 0:
                ax.set_ylabel(f'{SYMBOLS[j]}\n({UNITS[j]})', fontsize=13)
            ax.grid(axis='y', alpha=.2)
    fig.suptitle('Coefficient estimation errors: MLE vs small and large NN', fontsize=16, y=.98)
    fig.subplots_adjust(left=.135, right=.985, top=.90, bottom=.13, hspace=.18, wspace=.12)
    fig.text(.135, .09,
             'Error = estimate - truth. Same vertical scale within each row.', fontsize=10)
    fig.text(.135, .068,
             'Box: Q1-Q3. Black line: median. Whiskers: within 1.5 IQR.', fontsize=10)
    fig.text(.135, .046,
             f'{len(errors["MLE"]):,} paired cases, 50 years each. All outliers shown and included in RMSE.', fontsize=10)
    fig.text(.135, .024,
             'MLE: training-domain constraints. NNs: 100k, L2, seed 20260929.', fontsize=10)
    path = output / 'coefficient_error_boxplots_5x3.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def refresh_plots(output):
    """Redraw saved case-level errors without fitting or recomputing tests."""
    output = output.resolve()
    source = output / 'paired_coefficient_errors.csv'
    saved = pd.read_csv(source)
    errors = {}
    reference_index = None
    for model, stored_name in (('MLE', 'MLE_training_domain'), ('Small', 'Small'), ('Large', 'Large')):
        pivot = saved.loc[saved.model == stored_name].pivot(
            index='sample_index', columns='coefficient', values='error').sort_index()
        matrix = pivot.reindex(columns=PARAMETERS).to_numpy(dtype=float)
        if not len(pivot) or not np.isfinite(matrix).all():
            raise ValueError(f'Missing or nonfinite coefficient errors for {model}.')
        if reference_index is None:
            reference_index = pivot.index
        elif not pivot.index.equals(reference_index):
            raise ValueError('All three methods must have the same paired cases.')
        errors[model] = matrix
    figure = make_figure(errors, output)
    write_report(output, figure)
    atomic_json(output / 'plot_metadata.json', dict(
        layout='5 coefficient rows x 3 model columns (MLE, Small, Large)',
        shared_y='within each coefficient row', n_cases=len(reference_index),
        source_errors_sha256=file_sha256(source), source_sha256=file_sha256(__file__),
        figure=figure.name, numerical_comparison_unchanged=True))
    print('FIGURE:', figure)


def beamer_source(table):
    rows = []
    for symbol, row in zip(SYMBOLS, table.itertuples(index=False)):
        p = '<0.001' if row.p_Holm < .001 else f'={row.p_Holm:.3f}'
        decision = 'Yes' if row.Large_significantly_better else 'No'
        rows.append(f'{symbol} & {row.Small_improvement_percent:+.2f}\\% & '
                    f'{row.Large_improvement_percent:+.2f}\\% & {decision} ($p_{{\\rm Holm}}{p}$) \\\\')
    return '\n'.join([
        '% Requires \\usepackage{booktabs}',
        '% Fixed: 100k simulations (80k/10k/10k), 17 inputs, 5 outputs, 50 years.',
        '% Adam, lr=0.001, L2=1e-4, dropout=0, batch=128, seed=20260929.',
        '% Small: 128-128-64. Large: 512-512-512-128-128.',
        '% Frozen best checkpoints selected using validation coefficient MSE only.',
        '% Evaluation: 2,000 existing MLE validation cases from the older simulation pool.',
        '% Primary MLE: training-domain constrained likelihood, four data-only starts.',
        '% One-sided paired t-test on per-sequence squared errors; Holm across 5 coefficients.',
        '% H0: E[SE_small-SE_large] <= 0; H1: > 0. alpha=0.05.',
        '% Conditional on these fixed checkpoints; no architecture-wide or RL claim.',
        r'\begin{frame}{Parameter RMSE Improvement Relative to MLE}',
        r'\centering\small',
        r'\[\mathrm{Improvement}=100\left(1-\frac{\mathrm{RMSE}_{\mathrm{NN}}}{\mathrm{RMSE}_{\mathrm{MLE}}}\right)\%.\]',
        r'\begin{tabular}{lrrl}', r'\toprule',
        r'Coefficient & Small NN & Large NN & Large better? \\', r'\midrule',
        *rows, r'\bottomrule', r'\end{tabular}', r'\medskip',
        r'\begin{minipage}{0.96\linewidth}\footnotesize',
        r'Positive: lower RMSE than constrained MLE. Negative: higher RMSE.\\',
        r'2,000 paired cases; one-sided paired $t$-test on squared errors,\\',
        r'Holm adjustment ($\alpha=0.05$). Fixed-model, exploratory comparison.',
        r'\end{minipage}', r'\end{frame}', ''])


def run(args):
    old = json.loads((args.mle_directory / 'protocol.json').read_text(encoding='utf-8'))
    if old['status'] != 'completed':
        raise ValueError('MLE benchmark is incomplete.')
    old_meta = json.loads((Path(old['run_directory']) / 'run_metadata.json').read_text(encoding='utf-8'))
    data = load_diagnostic_split(old_meta['data_directory'], 'validation', old['validation_files_sha256'])
    if data['metadata_sha256'] != old['dataset_metadata_sha256']:
        raise ValueError('MLE source data metadata changed.')
    indices = pd.read_csv(args.mle_directory / 'sample_indices.csv').sample_index.to_numpy(dtype=int)
    if len(indices) != old['n_samples'] or len(np.unique(indices)) != len(indices):
        raise ValueError('Invalid MLE sample alignment.')
    fits = pd.read_csv(args.mle_directory / 'fits.csv')
    truth = np.asarray(data['coefficients'][indices], dtype=float)
    observations = np.asarray(data['annual_maxima'][indices])
    ns = load_notebook_components()
    torch.set_num_threads(4)
    estimates, metadata, quality = {}, {}, []
    for name, path in (('Small', args.small_run), ('Large', args.large_run)):
        meta = json.loads((path / 'run_metadata.json').read_text(encoding='utf-8'))
        if meta['status'] != 'completed' or meta['test_used_for_training_or_selection']:
            raise ValueError('Need completed, validation-selected NN runs.')
        for key in ('years', 'reference_year', 'time_scale_years', 'parameter_ranges', 'input_columns', 'target_columns'):
            current_data = json.loads((Path(meta['data_directory']) / 'train/metadata.json').read_text(encoding='utf-8'))
            if current_data[key] != data['metadata'][key]:
                raise ValueError(f'Simulation design mismatch: {key}')
        model, scaler, checkpoint = ns['load_cheng_checkpoint'](path / 'best.pt', args.device)
        if checkpoint['epoch'] != meta['best_epoch']:
            raise ValueError('Not the original best validation checkpoint.')
        standardized = ns['predict_standardized_coefficients'](model, data['inputs'][indices], scaler, args.device)
        estimates[name] = ns['inverse_sample_standardization'](standardized, data['location_scale'][indices])
        metadata[name] = {**meta, 'checkpoint_sha256': file_sha256(path / 'best.pt')}
        del model
    comparable = ('data_directory', 'dataset_files_sha256', 'seed', 'optimizer', 'initial_learning_rate',
                  'weight_decay', 'dropout_p', 'batch_size', 'max_epochs', 'patience')
    if any(metadata['Small'][k] != metadata['Large'][k] for k in comparable):
        raise ValueError('Small/large fitting conditions differ.')
    overlaps = {}
    case_hashes = sequence_hashes(observations)
    for split in ('train', 'validation', 'test'):
        array = np.load(Path(metadata['Small']['data_directory']) / split / 'annual_maxima.npy', mmap_mode='r')
        overlaps[split] = len(case_hashes & sequence_hashes(array))
    if any(overlaps.values()):
        raise ValueError('MLE cases overlap current model data; do not label them held out.')
    options = dict(reference_year=data['metadata']['reference_year'], time_scale_years=data['metadata']['time_scale_years'])
    years = np.array(data['metadata']['years'])
    for domain in DOMAIN_NAMES:
        part = fits.loc[fits.domain.eq(domain)]
        if part.sample_index.duplicated().any():
            raise ValueError('Duplicated MLE sample ids.')
        part = part.set_index('sample_index').loc[indices]
        if not part.success.eq(True).all():
            raise ValueError('MLE failure: report explicit common-case handling before comparison.')
        estimates['MLE_'+domain] = part[PARAMETERS].to_numpy(float)
        quality.append(dict(model='MLE_'+domain, n_cases=len(indices), n_failed=0,
                            on_boundary=int(part.on_boundary.sum())))
    for name, prediction in estimates.items():
        valid = check_gev_predictions(prediction, observations, years, **options)
        quality.append(dict(model=name, n_cases=len(indices), observed_support_valid=int(valid.gev_valid_all_years.sum())))
    errors = {name: prediction - truth for name, prediction in estimates.items()}
    tables = {}
    for domain in DOMAIN_NAMES:
        selected = {'MLE': errors['MLE_'+domain], 'Small': errors['Small'], 'Large': errors['Large']}
        tables[domain] = compare_errors(selected)
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'protocol.json', dict(
        status='completed', source_mle=str(args.mle_directory.resolve()),
        source_protocol_sha256=file_sha256(args.mle_directory / 'protocol.json'),
        source_fits_sha256=file_sha256(args.mle_directory / 'fits.csv'),
        source_data_sha256=old['validation_files_sha256'],
        model_runs={name: dict(path=str(path.resolve()), best_epoch=metadata[name]['best_epoch'],
                              checkpoint_sha256=metadata[name]['checkpoint_sha256'])
                    for name, path in (('Small', args.small_run), ('Large', args.large_run))},
        n_cases=len(indices), exact_sequence_overlaps=overlaps, seed=metadata['Small']['seed'],
        test='One-sided paired t-test of SE_small-SE_large; Holm family=5',
        bootstrap='5000 paired sequences; pointwise 95% RMSE-difference interval',
        comparison='Exploratory reused MLE benchmark, conditional on two fixed checkpoints',
        primary_mle_domain='training_domain', no_RL=True, source_sha256=file_sha256(__file__)))
    atomic_csv(output / 'fit_quality.csv', pd.DataFrame(quality))
    atomic_csv(output / 'sample_indices.csv', pd.DataFrame({'sample_index': indices}))
    long = []
    for model, error in errors.items():
        for j, name in enumerate(PARAMETERS):
            long.append(pd.DataFrame(dict(sample_index=indices, model=model, coefficient=name,
                                          truth=truth[:, j], estimate=estimates[model][:, j], error=error[:, j])))
    atomic_csv(output / 'paired_coefficient_errors.csv', pd.concat(long, ignore_index=True))
    for domain, table in tables.items():
        atomic_csv(output / f'comparison_{domain}.csv', table)
    primary = {'MLE': errors['MLE_training_domain'], 'Small': errors['Small'], 'Large': errors['Large']}
    figure = make_figure(primary, output)
    quartiles = []
    for model, error in primary.items():
        for j, parameter in enumerate(PARAMETERS):
            q1, median, q3 = np.quantile(error[:, j], [.25, .5, .75])
            quartiles.append(dict(model=model, coefficient=parameter, Q1=q1, median=median, Q3=q3, IQR=q3-q1))
    atomic_csv(output / 'error_quartiles.csv', pd.DataFrame(quartiles))
    (output / 'parameter_comparison_beamer.tex').write_text(beamer_source(tables['training_domain']), encoding='utf-8')
    write_report(output, figure)
    print(tables['training_domain'].to_string(index=False))
    print('OUTPUT:', output)


def write_report(output, figure):
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.v4.new_notebook()
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.cells = [nbformat.v4.new_markdown_cell(
        '# MLE 與大小模型：五係數比較\n\n'
        '使用既有 MLE 的 2,000 組舊 validation 序列，每組 50 年。重新評估目前 100k 實驗的兩個固定 NN，'
        '不使用舊 1M NN 的誤差，不重訓、不重做 MLE。未發現與目前 train/validation/test 完全相同的年序列。\n\n'
        '主表為 training-domain 受限 MLE（參數範圍與模擬一致）；另附 wide-domain 敏感度表，'
        '不把受限 MLE 說成無限制 MLE或理論誤差下限。\n\n'
        '改善率 = 100 × (1 − NN RMSE / MLE RMSE)，正值較好。五係數均用原尺度，不能跨單位直接比較 RMSE。'
        '箱型圖為 5 列 × 3 欄：列依序為 mu0、beta_mu、eta0、beta_sigma、xi0；欄依序為 MLE、小模型、大模型。同列共用縱軸刻度。'
        '箱體為誤差 Q1–Q3，不是真實參數的分布；所有離群值保留在 RMSE 與檢定中。\n\n'
        '檢定以每組序列為配對單位，d = SE_small − SE_large，H0: E(d)≤0，H1: E(d)>0。'
        '用單尾配對 t 檢定，五個 p 值做 Holm 校正，α=0.05。以 MSE 檢定與 RMSE 排序一致。'
        '依賴不同模擬案例獨立與樣本平均近似常態；不是無假設的 exact test。'
        '另附 5,000 次 sequence bootstrap 的點別 95% RMSE 差區間，未做同時區間校正。\n\n'
        '這是既有 benchmark 上的探索性、固定模型比較；p 值不包含重訓 seed 的不確定性，'
        '不代表所有大模型都優於小模型。不顯著也不代表等效。本次不比較 RL。\n\n'
        '參考：[SciPy paired t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)、'
        '[Holm correction](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html)。'),
        nbformat.v4.new_code_cell('from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\n'
            f'OUT=Path({str(output)!r})\n'
            "display(pd.read_csv(OUT/'comparison_training_domain.csv').round(5))\n"
            "display(pd.read_csv(OUT/'fit_quality.csv'))\n"
            f'display(Image(filename={str(figure)!r}))'),
        nbformat.v4.new_markdown_cell('## MLE 範圍敏感度與誤差四分位數'),
        nbformat.v4.new_code_cell("display(pd.read_csv(OUT/'comparison_wide_domain.csv').round(5))\n"
                                 "display(pd.read_csv(OUT/'error_quartiles.csv').round(5))")]
    NotebookClient(nb, timeout=90, kernel_name='python3', resources={'metadata': {'path': str(PROJECT_ROOT)}}).execute()
    nbformat.write(nb, output / 'cheng_NN_MLE_small_large.ipynb')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    base = PROJECT_ROOT / 'results/cheng_nn_17d'
    p.add_argument('--mle-directory', type=Path, default=base/'run_20260930_gpu_1m_l2/mle_comparison_2000')
    p.add_argument('--small-run', type=Path, default=base/'three_split_curves_20261003/small_seed_20260929')
    p.add_argument('--large-run', type=Path, default=base/'three_split_curves_20261003/large_seed_20260929')
    p.add_argument('--output-directory', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--plot-only', action='store_true', help='Refresh figure/report from saved paired errors only.')
    args = p.parse_args()
    if args.plot_only:
        refresh_plots(args.output_directory)
    else:
        run(args)
