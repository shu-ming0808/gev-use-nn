"""Paired regularization pilot: 100k fixed sequences, seven RMSE outcomes.

Training remains coefficient MSE (not an RL-aware loss); all reported errors
are RMSE. Every method retains the same validation-based early stopping.
No test predictions, simulation regeneration, or production-model replacement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from cheng_nn_evaluation import conditional_return_levels
from cheng_nn_simulation import COEFFICIENT_NAMES
from train_cheng_nn import PROJECT_ROOT, atomic_csv, atomic_json, file_sha256

SEEDS = (20260929, 20260930, 20261001)
PERIODS = (20., 100.)
OUTCOMES = (*COEFFICIENT_NAMES, 'RL20', 'RL100')
# These are declared pilot strengths, not independently tuned optima. A single
# coefficient is not comparable in strength between L1, L2 and AdamW.
CASES = {
    'baseline': dict(optimizer='Adam', weight_decay=0., l1_lambda=0., dropout_p=0.),
    'L1_1e-5': dict(optimizer='Adam', weight_decay=0., l1_lambda=1e-5, dropout_p=0.),
    'L2_1e-4': dict(optimizer='Adam', weight_decay=1e-4, l1_lambda=0., dropout_p=0.),
    'AdamW_1e-2': dict(optimizer='AdamW', weight_decay=1e-2, l1_lambda=0., dropout_p=0.),
    'Dropout_0.1': dict(optimizer='Adam', weight_decay=0., l1_lambda=0., dropout_p=.1),
    'Dropout_0.2': dict(optimizer='Adam', weight_decay=0., l1_lambda=0., dropout_p=.2),
}
FIXED_FIELDS = (
    'data_directory', 'dataset_metadata_sha256', 'dataset_files_sha256',
    'split_sizes', 'device', 'torch_version', 'cpu_threads', 'batch_size',
    'initial_learning_rate', 'architecture', 'max_epochs', 'patience',
    'monitor_train_eval', 'return_periods', 'loss', 'notebook_sha256', 'runner_sha256',
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def improvement_percent(baseline, candidate):
    """Positive = better; never round inputs before calculating percentages."""
    baseline, candidate = np.asarray(baseline), np.asarray(candidate)
    if not np.all(np.isfinite(baseline) & (baseline > 0)) or not np.all(np.isfinite(candidate)):
        raise ValueError('Improvement needs finite RMSE and a positive baseline.')
    return 100. * (baseline - candidate) / baseline


def metrics_for_split(path, split):
    coefficients = pd.read_csv(path / split / 'coefficient_metrics.csv')
    part = coefficients.loc[coefficients.scale.eq('original')].set_index('coefficient')
    if set(part.index) != set(COEFFICIENT_NAMES) or not part.index.is_unique:
        raise ValueError('Need exactly five original-scale coefficient RMSEs.')
    if not (part.n_used == part.n_total).all():
        raise ValueError('Cannot rank models after silently excluding coefficient failures.')
    values = {name: float(part.loc[name, 'RMSE']) for name in COEFFICIENT_NAMES}
    rl = pd.read_csv(path / split / 'return_level_overall.csv')
    for period in PERIODS:
        row = rl.loc[rl.subset.eq('finite_RL') & rl.return_period.eq(period)]
        if len(row) != 1 or int(row.iloc[0].n_used) != int(row.iloc[0].n_total):
            raise ValueError('Missing RL20/RL100 or nonfinite predictions; do not silently exclude failures.')
        values[f'RL{int(period)}'] = float(row.iloc[0].RMSE)
    if not np.isfinite(list(values.values())).all():
        raise ValueError('Nonfinite RMSE.')
    scaled = coefficients.loc[coefficients.scale.eq('train_target_z')]
    if len(scaled) != 5 or not (scaled.n_used == scaled.n_total).all():
        raise ValueError('Invalid scaled coefficients.')
    return values, float(np.sqrt(np.mean(scaled.RMSE.to_numpy() ** 2)))


def validate_run(directory, entry, reference=None):
    path = (directory / entry['directory']).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError('Run path escapes the experiment.')
    meta = read_json(path / 'run_metadata.json')
    if meta['status'] != 'completed' or meta['seed'] != entry['seed']:
        raise ValueError('Run is incomplete or has a different seed.')
    if any(meta[k] != v for k, v in CASES[entry['case']].items()):
        raise ValueError('Regularization settings differ from protocol.')
    if meta['split_sizes'] != {'train': 80000, 'validation': 10000}:
        raise ValueError('This comparison requires the fixed 80k/10k splits.')
    if meta['test_evaluation_requested'] or meta['test_used_for_training_or_selection'] or (path / 'test').exists():
        raise ValueError('Test data must not enter regularization selection.')
    if set(meta['dataset_files_sha256']) != {'train', 'validation'}:
        raise ValueError('Unexpected dataset split.')
    signature = {key: meta[key] for key in FIXED_FIELDS}
    if reference is not None and signature != reference:
        raise ValueError('Data or fixed training conditions changed between runs.')
    if meta['return_periods'] != list(PERIODS) or not meta['monitor_train_eval']:
        raise ValueError('Missing requested RL periods or eval-mode training diagnostics.')
    digest = file_sha256(path / 'best.pt')
    for split in ('train', 'validation'):
        marker = read_json(path / split / 'prediction_metadata.json')
        if (marker['checkpoint_sha256'] != digest
                or marker['best_epoch'] != meta['best_epoch']
                or marker['n_samples'] != meta['split_sizes'][split]
                or marker['dataset_metadata_sha256'] != meta['dataset_metadata_sha256'][split]
                or marker['return_periods'] != list(PERIODS)):
            raise ValueError('Prediction provenance does not match checkpoint/data.')
    return path, meta, signature


def aggregate_metrics(rows, seeds):
    per_run = pd.DataFrame(rows)
    expected = {(c, s, o) for c in CASES for s in seeds for o in OUTCOMES}
    actual = set(per_run[['case', 'seed', 'outcome']].itertuples(index=False, name=None))
    if actual != expected or len(per_run) != len(expected):
        raise ValueError('Need all paired cases, seeds and seven outcomes, without duplicates.')
    base = per_run.loc[per_run.case.eq('baseline'), ['seed', 'outcome', 'validation_RMSE']]
    base = base.rename(columns={'validation_RMSE': 'baseline_RMSE'})
    paired = per_run.merge(base, on=['seed', 'outcome'], validate='many_to_one')
    paired['improvement_percent'] = improvement_percent(paired.baseline_RMSE, paired.validation_RMSE)
    paired['validation_minus_train_RMSE'] = paired.validation_RMSE - paired.train_RMSE
    paired['gap_percent'] = 100 * paired.validation_minus_train_RMSE / paired.train_RMSE
    summary = paired.groupby(['case', 'outcome'], sort=False).agg(
        n_seeds=('seed', 'size'),
        train_RMSE_mean=('train_RMSE', 'mean'), train_RMSE_SD=('train_RMSE', 'std'),
        validation_RMSE_mean=('validation_RMSE', 'mean'), validation_RMSE_SD=('validation_RMSE', 'std'),
        improvement_percent_mean=('improvement_percent', 'mean'),
        improvement_percent_SD=('improvement_percent', 'std'),
        gap_mean=('validation_minus_train_RMSE', 'mean'), gap_percent_mean=('gap_percent', 'mean'),
    ).reset_index()
    return paired, summary


def summarize(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / 'experiment.json')
    rows, diagnostics, histories, reference = [], [], [], None
    for entry in manifest['runs']:
        path, meta, reference = validate_run(directory, entry, reference)
        train, train_scaled = metrics_for_split(path, 'train')
        validation, validation_scaled = metrics_for_split(path, 'validation')
        if not np.isclose(validation_scaled ** 2, meta['best_validation_loss'], rtol=2e-6, atol=1e-7):
            raise ValueError('Reported RMSE does not match selected checkpoint.')
        for outcome in OUTCOMES:
            rows.append(dict(case=entry['case'], seed=entry['seed'], outcome=outcome,
                             train_RMSE=train[outcome], validation_RMSE=validation[outcome]))
        history = pd.read_csv(path / 'training_history.csv')
        best = history.loc[history.epoch.eq(meta['best_epoch'])].iloc[0]
        last = history.iloc[-1]
        if not np.isclose(best.train_RMSE_eval, train_scaled, rtol=2e-6, atol=1e-7):
            raise ValueError('Train eval-mode RMSE differs from selected checkpoint export.')
        diagnostics.append(dict(
            case=entry['case'], seed=entry['seed'], best_epoch=meta['best_epoch'],
            epochs_completed=meta['epochs_completed'], training_seconds=meta['training_seconds'],
            train_scaled_RMSE=train_scaled, validation_scaled_RMSE=validation_scaled,
            scaled_gap=validation_scaled-train_scaled,
            final_validation_RMSE=float(last.validation_RMSE_eval),
            final_vs_best_validation_change_percent=float(100*(last.validation_RMSE_eval/validation_scaled-1)),
            final_vs_best_train_change_percent=float(100*(last.train_RMSE_eval/best.train_RMSE_eval-1)),
            GEV_support_valid_fraction=meta['evaluation']['validation']['valid_gev_fraction'],
        ))
        histories.append(history.assign(case=entry['case'], seed=entry['seed']))
    paired, summary = aggregate_metrics(rows, manifest['seeds'])
    atomic_csv(directory / 'rmse_per_seed.csv', paired)
    atomic_csv(directory / 'comparison_summary.csv', summary)
    atomic_csv(directory / 'overfit_diagnostics.csv', pd.DataFrame(diagnostics))
    atomic_csv(directory / 'learning_curves.csv', pd.concat(histories, ignore_index=True))
    display = summary.copy()
    display['RMSE (improvement %)'] = display.apply(
        lambda r: f'{r.validation_RMSE_mean:.4f} ({r.improvement_percent_mean:+.2f}%)', axis=1)
    table = display.pivot(index='case', columns='outcome', values='RMSE (improvement %)').reindex(index=CASES, columns=OUTCOMES)
    atomic_csv(directory / 'comparison_table.csv', table.reset_index())
    return table, summary, pd.DataFrame(diagnostics)


def previous_dropout_results(directory):
    """Rescore saved 1M-run predictions for RL20; never retrain/rewrite old runs."""
    directory = Path(directory).resolve()
    manifest = read_json(directory / 'experiment.json')
    if manifest['status'] != 'completed':
        raise ValueError('Dropout experiment is not complete.')
    data = Path(manifest['baseline_conditions']['data_directory']) / 'validation'
    truth = np.load(data / 'coefficients_original.npy', mmap_mode='r')
    info = read_json(data / 'metadata.json')
    options = dict(reference_year=info['reference_year'], time_scale_years=info['time_scale_years'])
    actual = conditional_return_levels(truth, np.array(info['years']), PERIODS, **options)
    rows = []
    for entry in manifest['runs']:
        path = Path(entry['directory'])
        marker = read_json(path / 'validation/prediction_metadata.json')
        if marker['checkpoint_sha256'] != file_sha256(path / 'best.pt'):
            raise ValueError('Checkpoint changed.')
        if (marker['dataset_metadata_sha256'] != file_sha256(data / 'metadata.json')
                or manifest['baseline_conditions']['dataset_files_sha256']['validation']['coefficients_original.npy'] != file_sha256(data / 'coefficients_original.npy')):
            raise ValueError('Dropout reference data changed.')
        predicted = np.load(path / 'validation/predictions_original.npy', mmap_mode='r')
        row = {'dropout_p': entry['dropout_p']}
        for j, name in enumerate(COEFFICIENT_NAMES):
            row[name] = float(np.sqrt(np.mean((predicted[:, j]-truth[:, j])**2)))
        levels = conditional_return_levels(predicted, np.array(info['years']), PERIODS, **options)
        if not np.isfinite(levels).all() or not np.isfinite(actual).all():
            raise ValueError('Nonfinite RL; cannot silently exclude samples.')
        for j, period in enumerate(PERIODS):
            row[f'RL{int(period)}'] = float(np.sqrt(np.mean((levels[:, :, j]-actual[:, :, j])**2)))
        rows.append(row)
    table = pd.DataFrame(rows).sort_values('dropout_p')
    atomic_csv(directory / 'rmse_RL20_RL100.csv', table)
    return table


def render_report(directory):
    import nbformat
    from nbclient import NotebookClient
    directory = Path(directory).resolve()
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    relative = directory.relative_to(PROJECT_ROOT).as_posix()
    nb = nbformat.v4.new_notebook(cells=[
        md('# 10 萬組：正則化比較（七項 RMSE）\n\n'
           '固定 80,000 train／10,000 validation／10,000 test；每組 50 年、17 維輸入、5 係數輸出。'
           '本實驗不讀取 test 陣列、不改模擬、不發布或替換模型。所有方法保留 early stopping，'
           '所以 baseline 是「無額外 L1/L2/Dropout」，不是完全沒有正則化。\n\n'
           '固定 512–512–512–128–128、ReLU、linear output、lr=0.001、batch=128、'
           '最多 300 epochs、patience=20、相同 ReduceLROnPlateau；每組三個配對種子。'
           'L1=1e-5（所有權重及 bias 的絕對值總和）、Adam L2=1e-4、AdamW decay=1e-2、'
           'Dropout=0.1/0.2（無 L2）。每次只比較列明的設定，不表示各方法已調到最佳強度。\n\n'
           '訓練仍用五係數標準化 MSE；選 checkpoint 的 MSE 與其平方根 RMSE 排序相同。'
           '表格全部改用 RMSE。RL20/RL100 是固定年份條件式年分布分位數，'
           '不是未來非定常氣候下的等待時間。\n\n'
           '改善率 = 100 × (baseline RMSE − method RMSE) / baseline RMSE，先在相同種子內配對，'
           '再取平均；正值改善、負值變差。SD 是訓練種子間變動，不是信賴區間或顯著性檢定。'
           'RL 合併 10,000 序列 × 50 年，不視為 500,000 個獨立樣本。'),
        code('from pathlib import Path\nimport sys\nimport pandas as pd\nimport matplotlib.pyplot as plt\n'
             'from IPython.display import display\n'
             'ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src/train_cheng_nn.py").exists())\n'
             'sys.path.insert(0, str(ROOT / "src"))\n'
             'from compare_cheng_nn_regularization import summarize, CASES\n'
             f'EXPERIMENT = ROOT / {relative!r}\n'
             'table, detail, overfit = summarize(EXPERIMENT)\ndisplay(table)\n'),
        md('## RMSE 與種子穩定度\n\n五係數原尺度的單位不同，不直接比較不同欄誰的 RMSE 最大。'
           '先看同一 outcome 的模型差異；RL 的單位為 °C。'),
        code('display(detail.round(5))\n'),
        md('## 過擬合與 GEV 支持範圍\n\n同一最佳 checkpoint、eval mode 比較 train/validation；'
           '兩者均關閉 Dropout、不含懲罰項。正的 gap 不單獨證明過擬合，'
           '小 gap 也可能兩者都學不好。合併檢視 validation RMSE、學習曲線、種子變動與 GEV support。'
           '最後 epoch 的 validation 惡化配合 train 下降，才是過擬合警訊之一；採用的是較早的最佳 checkpoint。'
           'GEV support 比例檢查所有觀測是否落在預測分布的定義域；其失敗案例不從主要 RMSE 中偷偷刪除。'),
        code('display(overfit.round(5))\n'
             'display(detail[["case", "outcome", "train_RMSE_mean", "validation_RMSE_mean", "gap_mean", "gap_percent_mean"]].round(5))\n'),
        code('curves = pd.read_csv(EXPERIMENT / "learning_curves.csv")\n'
             'fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True, constrained_layout=True)\n'
             'for ax, case in zip(axes.flat, CASES):\n'
             '    for seed, h in curves.loc[curves.case.eq(case)].groupby("seed"):\n'
             '        line, = ax.plot(h.epoch, h.validation_RMSE_eval, label=f"{seed}: val")\n'
             '        ax.plot(h.epoch, h.train_RMSE_eval, "--", color=line.get_color(), alpha=.6)\n'
             '    ax.set_title(case); ax.set_xlabel("Epoch"); ax.grid(alpha=.2)\n'
             'axes[0,0].legend(fontsize=7)\n'
             'axes[0,0].set_ylabel("Scaled coefficient RMSE; dashed = train eval")\n'
             'fig.savefig(EXPERIMENT / "regularization_learning_curves.png", dpi=150)\nplt.show()'),
        md('## 判斷原則\n\n優先檢查 RL20/RL100 validation RMSE 是否一致改善，'
           '再看五係數有無明顯代價、跨種子穩定度、support 和過擬合。'
           '不只追求更小 gap，也不依 test 反覆挑模型。這是固定強度的探索比較，'
           '不是方法優劣的普遍定論。既有 100k test 曾在歷史模型評估中使用；'
           '本輪不重看它，最終確認宜另外保留未參與調整的獨立模擬資料。\n\n'
           '方法文獻：[Dropout (2014)](https://jmlr.org/papers/v15/srivastava14a.html)、'
           '[AdamW (2019)](https://arxiv.org/abs/1711.05101)。'),
    ])
    nb.metadata['kernelspec'] = dict(display_name='Python 3', language='python', name='python3')
    NotebookClient(nb, timeout=240, kernel_name='python3', resources={'metadata': {'path': str(PROJECT_ROOT)}}).execute()
    report = directory / 'cheng_NN_regularization_comparison.ipynb'
    nbformat.write(nb, report)
    return report


def run_experiment(directory, device='cuda'):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    data = PROJECT_ROOT / 'data/simulated/cheng_nn_17d'
    for split, n in [('train', 80000), ('validation', 10000), ('test', 10000)]:
        if read_json(data / split / 'metadata.json')['n_samples'] != n:
            raise ValueError('Expected unchanged 100k simulation split sizes.')
    snapshot = directory / 'source_snapshot'
    snapshot.mkdir()
    sources = [Path(__file__), PROJECT_ROOT / 'src/train_cheng_nn.py', PROJECT_ROOT / 'notebooks/cheng_NN.ipynb']
    hashes = {str(p): file_sha256(p) for p in sources}
    for p in sources:
        shutil.copy2(p, snapshot / p.name)
    manifest = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
                    cases=CASES, seeds=list(SEEDS), return_periods=list(PERIODS),
                    test_used=False, completed_runs=0, source_hashes=hashes, runs=[])
    names = list(CASES)
    # Baseline first for the first seed; rotate remaining orders by seed.
    for i, seed in enumerate(SEEDS):
        for name in names[i:] + names[:i]:
            manifest['runs'].append(dict(case=name, seed=seed, directory=f'{name}_seed{seed}'))
    atomic_json(directory / 'experiment.json', manifest)
    started = time.perf_counter()
    reference = None
    try:
        for number, entry in enumerate(manifest['runs'], 1):
            if any(file_sha256(p) != digest for p, digest in hashes.items()):
                raise RuntimeError('Sources changed during experiment; stopped before mixing implementations.')
            print(f"RUN {number}/{len(manifest['runs'])}: {entry['case']}, seed={entry['seed']}", flush=True)
            settings = CASES[entry['case']]
            command = [sys.executable, '-u', str(PROJECT_ROOT / 'src/train_cheng_nn.py'),
                       '--data-directory', str(data), '--run-directory', str(directory / entry['directory']),
                       '--device', device, '--seed', str(entry['seed']), '--no-update-latest',
                       '--monitor-train-eval', '--return-periods', '20', '100']
            for key, value in settings.items():
                command += ['--' + key.replace('_', '-'), str(value)]
            # Full epoch output is already durably mirrored into each run .log.
            # Avoid flooding the app while serial jobs run on one GPU.
            subprocess.run(command, cwd=PROJECT_ROOT, check=True, stdout=subprocess.DEVNULL)
            _, _, reference = validate_run(directory, entry, reference)
            manifest['completed_runs'] = number
            atomic_json(directory / 'experiment.json', manifest)
            print(f"COMPLETED {number}/{len(manifest['runs'])}", flush=True)
        summarize(directory)
        manifest['status'] = 'rendering_report'
        atomic_json(directory / 'experiment.json', manifest)
        report = render_report(directory)
        manifest.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
                        wall_seconds=time.perf_counter()-started, report=str(report))
        atomic_json(directory / 'experiment.json', manifest)
        print(f'REPORT={report}', flush=True)
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        error=f'{type(exc).__name__}: {exc}', wall_seconds=time.perf_counter()-started)
        atomic_json(directory / 'experiment.json', manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--previous-dropout-results', type=Path)
    args = parser.parse_args()
    if args.previous_dropout_results:
        print(previous_dropout_results(args.previous_dropout_results).to_string(index=False))
    elif args.output_directory:
        if args.report_only:
            print(render_report(args.output_directory))
        else:
            run_experiment(args.output_directory, args.device)
    else:
        parser.error('Provide --output-directory or --previous-dropout-results.')
