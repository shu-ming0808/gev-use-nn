"""Paired validation-only NN / full-sequence MLE diagnostic, with no retraining.

Defaults to 2,000 uniformly selected validation sequences, never test data.
Includes bounded-likelihood sensitivity, explicit failure rates, and paired
sequence-bootstrap intervals. No claim of MLE efficiency at n=50 is made.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from cheng_nn_diagnostics import file_sha256, load_diagnostic_split, load_saved_predictions
from cheng_nn_evaluation import check_gev_predictions, conditional_return_levels
from cheng_nn_mle import fit_time_varying_gev, nll_and_gradient, standardized_coefficients
from cheng_nn_simulation import COEFFICIENT_NAMES, ParameterRanges

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'results/cheng_nn_17d/run_20260930_gpu_l2_1e4'
DOMAINS = ('training_domain', 'wide_domain')


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def atomic_csv(path, frame):
    path = Path(path)
    temp = path.with_name(path.name + '.partial')
    frame.to_csv(temp, index=False)
    temp.replace(path)


def fit_task(index, values, time_values, ranges, domain):
    result = fit_time_varying_gev(values, time_values, ranges=ranges, domain=domain)
    coefficients = result.pop('coefficients')
    return {'sample_index': int(index), **result,
            **dict(zip(COEFFICIENT_NAMES, coefficients))}


def paired_rmse_interval(nn_squared_error, mle_squared_error, *, seed=9030, repeats=2000):
    """Resample independent sequences, keeping all 50 years paired inside each."""
    nn_se, mle_se = np.asarray(nn_squared_error), np.asarray(mle_squared_error)
    if nn_se.ndim != 1 or nn_se.shape != mle_se.shape or not len(nn_se):
        raise ValueError('Need aligned, nonempty per-sequence squared errors.')
    if not np.isfinite(nn_se).all() or not np.isfinite(mle_se).all():
        raise ValueError('Bootstrap errors must be finite; exclusions must be explicit.')
    rng, results = np.random.default_rng(seed), []
    for start in range(0, repeats, 100):
        indices = rng.integers(0, len(nn_se), size=(min(100, repeats-start), len(nn_se)))
        results.extend(np.sqrt(nn_se[indices].mean(axis=1)) - np.sqrt(mle_se[indices].mean(axis=1)))
    low, high = np.quantile(results, [.025, .975])
    return float(low), float(high)


def summarize(output, data, predictions, indices, fits, target_std):
    meta = data['metadata']
    time_values = np.asarray(meta['centered_time'])
    years = np.asarray(meta['years'])
    options = {'reference_year': meta['reference_year'], 'time_scale_years': meta['time_scale_years']}
    truth = np.asarray(data['coefficients'][indices])
    observations = np.asarray(data['annual_maxima'][indices])
    transforms = np.asarray(data['location_scale'][indices])
    truth_z = standardized_coefficients(truth, transforms)
    estimates = {'NN': np.asarray(predictions['original'][indices])}
    success = {'NN': np.isfinite(estimates['NN']).all(axis=1)}
    for domain in DOMAINS:
        part = fits.loc[fits.domain.eq(domain)].set_index('sample_index').loc[indices]
        estimates['MLE_' + domain] = part[list(COEFFICIENT_NAMES)].to_numpy(float)
        success['MLE_' + domain] = part.success.to_numpy(bool)
    true_rl = conditional_return_levels(truth, years, **options)
    if not np.isfinite(true_rl).all():
        raise ValueError('Simulation truth has invalid return levels.')
    levels, validity, squares, absolute = {}, {}, {}, {}
    quality, boundaries, per_case = [], [], []
    for name, coefficient in estimates.items():
        levels[name] = conditional_return_levels(coefficient, years, **options)
        validity[name] = check_gev_predictions(coefficient, observations, years, **options)
        z = standardized_coefficients(coefficient, transforms)
        squares[name] = {}
        absolute[name] = {}
        for j, parameter in enumerate(COEFFICIENT_NAMES):
            difference = coefficient[:, j] - truth[:, j]
            squares[name][parameter] = difference**2
            absolute[name][parameter] = np.abs(difference)
        squares[name]['scaled_coefficients'] = np.mean(((z-truth_z)/target_std)**2, axis=1)
        absolute[name]['scaled_coefficients'] = np.mean(np.abs((z-truth_z)/target_std), axis=1)
        for k, period in enumerate((50, 100)):
            error = levels[name][:, :, k] - true_rl[:, :, k]
            squares[name][f'RL{period}'] = np.mean(error**2, axis=1)
            absolute[name][f'RL{period}'] = np.mean(np.abs(error), axis=1)
        row = {'model': name, 'n_requested': len(indices), 'n_success': int(success[name].sum()),
               'n_failed': int((~success[name]).sum()),
               'n_valid_observed_support': int(validity[name].gev_valid_all_years.sum()),
               'n_finite_RL': int(np.isfinite(levels[name]).all(axis=(1, 2)).sum())}
        if name != 'NN':
            fit = fits.loc[fits.domain.eq(name.removeprefix('MLE_'))]
            row.update({'n_on_boundary': int(fit.on_boundary.sum()),
                        'n_at_least_two_agreeing_starts': int((fit.agreeing_starts >= 2).sum()),
                        'n_lower_unconverged_candidate': int((fit.lower_nonconverged_nll_gap > 1e-4).sum()),
                        'median_fit_seconds': fit.seconds.median(), 'sum_worker_seconds': fit.seconds.sum()})
            for coordinate in ('mu0', 'beta_mu_over_sigma0', 'eta0', 'beta_sigma', 'xi0'):
                boundaries.append({'model': name, 'coordinate': coordinate,
                    'n_boundary': int(fit.boundary_coordinates.fillna('').str.split(',').map(lambda names: coordinate in names).sum())})
        quality.append(row)
        for i, index in enumerate(indices):
            per_case.append({'sample_index': int(index), 'model': name,
                'success': bool(success[name][i]),
                'support_valid': bool(validity[name].gev_valid_all_years.iloc[i]),
                'true_xi': truth[i, 4], 'true_sigma0': np.exp(truth[i, 2]),
                **{f'SE_{metric}': value[i] for metric, value in squares[name].items()}})
    quality = pd.DataFrame(quality)
    # Audit only AFTER fitting: true labels never initialize/select the optimizer.
    truth_losses = []
    for z_coef, y, transform in zip(truth_z, observations, transforms):
        p = z_coef.copy()
        p[1] /= np.exp(p[2])
        nll = nll_and_gradient(p, (y-transform[0])/transform[1], time_values)[0]
        truth_losses.append(nll + len(years)*np.log(transform[1]))
    audit = pd.DataFrame({'sample_index': indices, 'true_nll': truth_losses})
    for domain in DOMAINS:
        fit = fits.loc[fits.domain.eq(domain)].set_index('sample_index').loc[indices]
        audit[f'{domain}_nll_minus_truth'] = fit.nll.to_numpy() - np.array(truth_losses)
        quality.loc[quality.model.eq('MLE_'+domain), 'n_nll_worse_than_true_parameters'] = int(
            (audit[f'{domain}_nll_minus_truth'] > 1e-4).sum())
    atomic_csv(output / 'likelihood_audit.csv', audit)
    atomic_csv(output / 'fit_quality.csv', quality)
    atomic_csv(output / 'boundary_counts.csv', pd.DataFrame(boundaries))
    atomic_csv(output / 'per_sequence_errors.csv', pd.DataFrame(per_case))
    # Each comparison has its own explicit paired subset: no unequal denominators.
    rows, paired, by_year = [], [], []
    metrics = [*COEFFICIENT_NAMES, 'scaled_coefficients', 'RL50', 'RL100']
    for domain in DOMAINS:
        mle = 'MLE_' + domain
        common = (success['NN'] & success[mle]
                  & np.isfinite(levels['NN']).all(axis=(1, 2))
                  & np.isfinite(levels[mle]).all(axis=(1, 2)))
        if not common.any():
            raise ValueError(f'No common usable sequences for {domain}.')
        for subset, use in [('paired_finite_RL', common),
                            ('paired_valid_support', common & validity['NN'].gev_valid_all_years.to_numpy()
                             & validity[mle].gev_valid_all_years.to_numpy())]:
            if not use.any():
                continue
            for name in ('NN', mle):
                for metric in metrics:
                    rows.append({'comparison': domain, 'subset': subset, 'model': name, 'metric': metric,
                        'n_sequences': int(use.sum()), 'n_excluded': int(len(use)-use.sum()),
                        'RMSE': np.sqrt(squares[name][metric][use].mean()),
                        'MAE': absolute[name][metric][use].mean()})
            if subset == 'paired_finite_RL':
                for metric in metrics:
                    nn_se, mle_se = squares['NN'][metric][use], squares[mle][metric][use]
                    low, high = paired_rmse_interval(nn_se, mle_se)
                    nn_rmse, mle_rmse = np.sqrt(nn_se.mean()), np.sqrt(mle_se.mean())
                    paired.append({'comparison': domain, 'metric': metric, 'n_sequences': int(use.sum()),
                                   'NN_RMSE': nn_rmse, 'MLE_RMSE': mle_rmse,
                                   'NN_minus_MLE_RMSE': nn_rmse-mle_rmse,
                                   'delta_CI_low': low, 'delta_CI_high': high,
                                   'MLE_improvement_percent': 100*(nn_rmse-mle_rmse)/nn_rmse})
                for k, period in enumerate((50, 100)):
                    for name in ('NN', mle):
                        rmse = np.sqrt(np.mean((levels[name][use, :, k]-true_rl[use, :, k])**2, axis=0))
                        by_year.extend({'comparison': domain, 'model': name, 'period': period,
                                        'year': int(year), 'RMSE': float(value), 'n_sequences': int(use.sum())}
                                       for year, value in zip(years, rmse))
    atomic_csv(output / 'metrics.csv', pd.DataFrame(rows))
    atomic_csv(output / 'paired_comparison.csv', pd.DataFrame(paired))
    atomic_csv(output / 'rl_by_year.csv', pd.DataFrame(by_year))
    return pd.DataFrame(paired)


def write_notebook(output):
    import nbformat
    from nbclient import NotebookClient
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    notebook = nbformat.v4.new_notebook(cells=[
        md('# Cheng NN 與完整時間序列 MLE 比較\n\n'
           f"固定抽樣 {protocol['n_samples']:,} 組 validation 的診斷；每組 50 年。"
           '兩者估計相同五個時間變化 GEV 係數。NN 使用 17 維摘要；MLE 使用完整年序列。'
           '**沒有重新訓練 NN，也沒有使用 test。**\n\n'
           '**MLE 並非理論誤差下限**：50 筆資料下，MLE 可有較大的抽樣變異；'
           '以模擬資料和 MSE 訓練的 NN 則可能因學到抽樣分布而有收縮效果。'
           '因此 MLE 勝出表示存在可利用的實證差距，NN 勝出也不能證明 NN 已達最佳。'),
        code("from pathlib import Path\nimport json\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
             "from IPython.display import display\n"
             f"OUT = Path({str(output)!r})\n"
             "protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))\n"
             "display(pd.Series({k: protocol[k] for k in ['n_samples', 'sample_seed', 'nn_checkpoint_sha256', 'test_used', 'fit_wall_seconds']}))"),
        md('## 1. 比較範圍與數值品質\n\n'
           '- `training_domain`：MLE 的五個係數限制與模擬抽樣範圍一致。這是受限 MLE，不是 MAP。\n'
           '- `wide_domain`：標準化位置 [-20,20]、位置斜率/尺度 [-3,3]、log 尺度 [-7,4]、'
           'log 尺度斜率 [-1,1]、shape [-0.49,0.8]；是較寬的數值保護範圍，不是無限制 MLE。\n'
           '- 四個起點均由觀測資料建立；不以 NN、真值作起點。多起點並不保證全域最優。\n'
           '- 所有失敗與邊界解在此列出；後續表格比較相同成功案例，不把失敗誤差當零。\n'
           '- fit 時間是 CPU 多起點最佳化時間；不能直接和 GPU NN 的批次推論時間當成相同硬體比較。'),
        code("quality = pd.read_csv(OUT / 'fit_quality.csv')\ndisplay(quality.round(4))\n"
             "display(pd.read_csv(OUT / 'boundary_counts.csv'))"),
        md('## 2. 同一批案例的誤差\n\n'
           '正的 `NN_minus_MLE_RMSE` 代表 MLE 誤差較小，負值代表 NN 較小。'
           '區間為配對 sequence bootstrap 2,000 次所得的 95% 百分位區間，'
           '每次一起重抽該序列的全部 50 年；不是把 100,000 個年值當成獨立案例。'
           '這些是探索性區間，未校正多重比較，也未包含重新訓練 NN 的 seed 變異。\n\n'
           '`scaled_coefficients` 是用現有 NN checkpoint 的 training-only 標準差計算的整體係數 RMSE。'
           'RL 指固定年份的條件年分布分位數，不是未來等待時間。'),
        code("paired = pd.read_csv(OUT / 'paired_comparison.csv')\ndisplay(paired.round(5))\n"
             "metrics = pd.read_csv(OUT / 'metrics.csv')\n"
             "display(metrics.loc[metrics.subset.eq('paired_valid_support')].round(5))"),
        code("fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)\n"
             "for ax, domain in zip(axes, ['training_domain', 'wide_domain']):\n"
             "    part = paired.loc[paired.comparison.eq(domain) & paired.metric.isin(['RL50', 'RL100'])]\n"
             "    part.set_index('metric')[['NN_RMSE', 'MLE_RMSE']].plot.bar(ax=ax, rot=0)\n"
             "    ax.set(title=domain, ylabel='Conditional annual RL RMSE (C)')\n"
             "fig.savefig(OUT / 'nn_vs_mle_rl_rmse.png', dpi=160)\nplt.show()"),
        code("yearly = pd.read_csv(OUT / 'rl_by_year.csv')\n"
             "fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)\n"
             "for ax, period in zip(axes, [50, 100]):\n"
             "    part = yearly.loc[yearly.comparison.eq('training_domain') & yearly.period.eq(period)]\n"
             "    for model, group in part.groupby('model'):\n"
             "        ax.plot(group.year, group.RMSE, label=model)\n"
             "    ax.set(title=f'RL{period}(t)', xlabel='Year', ylabel='RMSE (C)')\n"
             "    ax.legend()\n"
             "fig.savefig(OUT / 'nn_vs_mle_rl_by_year.png', dpi=160)\nplt.show()"),
        md('## 3. 可以／不可以推論的事情\n\n'
           '- 若 MLE 較好：完整序列存在可利用的改善機會，但不能直接歸因於節點數；'
           '摘要資訊、loss、抽樣範圍與最佳化都不同。\n'
           '- 若 NN 較好：目前不支持「換成 MLE 就能改善」；不能因此宣稱沒有剩餘改善空間。\n'
           '- Dropout 測試應固定資料、架構、optimizer、L2，先比較 p=0、0.1、0.2；'
           'p=0.5 可作較強的對照，不是刪除或辨認節點數。\n'
           '- 若「sensitive function」是指 activation function，可另做 ReLU／GELU／SiLU 比較。'
           '不要與 Dropout 同時更動，否則無法分辨效果來源。\n'
           '- 五個輸出為可正可負的連續係數，沒有總和為 1 的限制，保留 linear output；'
           '不能換成 Softmax。尺度由 exp(log sigma) 保證為正，這和 Softmax 不同。\n\n'
           '參考：[SciPy genextreme](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html)、'
           '[PyTorch Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html)、'
           '[PyTorch Softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html)。'),
    ], metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    path = output / 'cheng_NN_MLE_comparison.ipynb'
    nbformat.write(notebook, path)
    NotebookClient(notebook, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    nbformat.write(notebook, path)
    return path


def run(args):
    import torch
    run_dir = Path(args.run_directory).resolve()
    output = Path(args.output_directory).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory; existing experiments are not overwritten.')
    meta = json.loads((run_dir / 'run_metadata.json').read_text(encoding='utf-8'))
    if meta['status'] != 'completed':
        raise ValueError('NN run must be completed.')
    data = load_diagnostic_split(meta['data_directory'], 'validation', meta['dataset_files_sha256']['validation'])
    predictions = load_saved_predictions(run_dir, 'validation', data)
    if predictions is None:
        raise ValueError('Validated saved NN predictions are required.')
    if not 1 <= args.n_samples <= len(data['inputs']):
        raise ValueError('n_samples must fit the validation split.')
    checkpoint = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=False)
    target_std = np.asarray(checkpoint['target_scaler']['std'], dtype=float)
    indices = np.sort(np.random.default_rng(args.seed).choice(len(data['inputs']), args.n_samples, replace=False))
    ranges = ParameterRanges(**{k: tuple(v) for k, v in data['metadata']['parameter_ranges'].items()})
    output.mkdir(parents=True)
    protocol = {'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'run_directory': str(run_dir), 'n_samples': args.n_samples, 'sample_seed': args.seed,
        'dataset_metadata_sha256': data['metadata_sha256'],
        'validation_files_sha256': meta['dataset_files_sha256']['validation'],
        'nn_checkpoint_sha256': file_sha256(run_dir / 'best.pt'),
        'mle_source_sha256': file_sha256(Path(__file__).with_name('cheng_nn_mle.py')),
        'comparison_source_sha256': file_sha256(__file__),
        'test_used': False, 'nn_retrained': False, 'n_jobs': args.n_jobs,
        'parameter_ranges': data['metadata']['parameter_ranges'],
        'wide_domain_bounds': [[-20,20], [-3,3], [-7,4], [-1,1], [-0.49,0.8]],
        'starts_per_fit': 4, 'bootstrap_repeats': 2000,
        'interval_unit': 'independent simulated sequence, all 50 years kept together',
        'common_target_std': target_std.tolist(),
        'model_is_lower_error_bound': False}
    atomic_json(output / 'protocol.json', protocol)
    atomic_csv(output / 'sample_indices.csv', pd.DataFrame({'sample_index': indices}))
    all_rows = []
    started = time.perf_counter()
    tasks = (delayed(fit_task)(index, np.asarray(data['annual_maxima'][index]),
                              np.asarray(data['metadata']['centered_time']), ranges, domain)
             for domain in DOMAINS for index in indices)
    # Small per-sequence tasks: disable memmaps; do not use user temp folders.
    with parallel_config(backend='loky', inner_max_num_threads=1):
        stream = Parallel(n_jobs=args.n_jobs, return_as='generator_unordered', max_nbytes=None,
                          idle_worker_timeout=1800)(tasks)
        for result in stream:
            all_rows.append(result)
            if len(all_rows) % 100 == 0 or len(all_rows) == 2*len(indices):
                frame = pd.DataFrame(all_rows).sort_values(['domain', 'sample_index'])
                atomic_csv(output / 'fits.csv', frame)
                print(f'MLE fits {len(all_rows)}/{2*len(indices)}, '
                      f'elapsed={time.perf_counter()-started:.1f}s, '
                      f'failed={int((~frame.success).sum())}', flush=True)
    protocol['fit_wall_seconds'] = time.perf_counter()-started
    fits = pd.DataFrame(all_rows).sort_values(['domain', 'sample_index'])
    paired = summarize(output, data, predictions, indices, fits, target_std)
    protocol['status'] = 'completed'
    protocol['completed_utc'] = datetime.now(timezone.utc).isoformat()
    atomic_json(output / 'protocol.json', protocol)
    print(paired.to_string(index=False), flush=True)
    print('Report:', write_notebook(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output-directory', type=Path, default=DEFAULT_RUN / 'mle_comparison_2000')
    parser.add_argument('--n-samples', type=int, default=2000)
    parser.add_argument('--n-jobs', type=int, default=6)
    parser.add_argument('--seed', type=int, default=20260930)
    run(parser.parse_args())
