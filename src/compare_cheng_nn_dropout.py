"""Controlled p=0/0.1/0.2/0.5 dropout pilot on the 100k-sequence design.

One paired seed, 80k train / 10k validation; no test access. The verified,
completed p=0 baseline is reused read-only. Other settings and checkpoint
selection rules are fixed. No model is published and latest_run is unchanged.
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

from train_cheng_nn import PROJECT_ROOT, atomic_json, atomic_csv, file_sha256

PROBABILITIES = (0.0, 0.1, 0.2, 0.5)
BASELINE = PROJECT_ROOT / 'results/cheng_nn_17d/run_20260930_gpu_l2_1e4'
FIXED_FIELDS = (
    'data_directory', 'dataset_metadata_sha256', 'dataset_files_sha256',
    'split_sizes', 'seed', 'architecture', 'trainable_parameters', 'device',
    'torch_version', 'cpu_threads', 'batch_size', 'optimizer',
    'initial_learning_rate', 'weight_decay', 'regularization',
    'reported_loss_includes_regularization', 'max_epochs', 'patience', 'loss',
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_one(entry, reference, *, require_complete=True):
    path = Path(entry['directory']).resolve()
    meta = read_json(path / 'run_metadata.json')
    if require_complete and meta['status'] != 'completed':
        raise ValueError(f'Incomplete run: {path.name}')
    if meta.get('dropout_p', 0.0) != entry['dropout_p']:
        raise ValueError('Recorded dropout probability differs from protocol.')
    if meta['test_evaluation_requested'] or meta['test_used_for_training_or_selection'] or (path / 'test').exists():
        raise ValueError('Dropout tuning must not evaluate the test split.')
    if set(meta['dataset_files_sha256']) != {'train', 'validation'}:
        raise ValueError('Only train/validation datasets may be used.')
    if any(meta[field] != reference[field] for field in FIXED_FIELDS):
        raise ValueError('Run data or fixed training conditions differ from baseline.')
    checkpoint_hash = file_sha256(path / 'best.pt')
    if entry.get('checkpoint_sha256') and checkpoint_hash != entry['checkpoint_sha256']:
        raise ValueError('Baseline or completed checkpoint changed after registration.')
    if require_complete:
        for split in ('train', 'validation'):
            marker = read_json(path / split / 'prediction_metadata.json')
            if (marker['checkpoint_sha256'] != checkpoint_hash
                    or marker['best_epoch'] != meta['best_epoch']
                    or marker['dataset_metadata_sha256'] != meta['dataset_metadata_sha256'][split]
                    or marker['n_samples'] != meta['split_sizes'][split]):
                raise ValueError('Prediction provenance differs from the checkpoint/data.')
    return path, meta


def summarize(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / 'experiment.json')
    probabilities = [entry['dropout_p'] for entry in manifest['runs']]
    if sorted(probabilities) != list(PROBABILITIES):
        raise ValueError('Need each of the four probabilities exactly once.')
    rows, coefficients, yearly, history = [], [], [], []
    for entry in manifest['runs']:
        path, meta = validate_one(entry, manifest['baseline_conditions'])
        row = {'dropout_p': entry['dropout_p'], 'seed': meta['seed'],
               'baseline_reused': entry['reused'],
               'best_epoch': meta['best_epoch'], 'epochs_completed': meta['epochs_completed'],
               'training_seconds': meta['training_seconds'], 'total_seconds': meta['total_seconds'],
               'trainable_parameters': meta['trainable_parameters'],
               'GEV_valid_fraction': meta['evaluation']['validation']['valid_gev_fraction']}
        for split in ('train', 'validation'):
            table = pd.read_csv(path / split / 'coefficient_metrics.csv')
            scaled = table.loc[table.scale.eq('train_target_z')]
            if len(scaled) != 5 or not (scaled.n_used == scaled.n_total).all():
                raise ValueError('Missing/nonfinite coefficient evaluation.')
            row[f'{split}_MSE_eval_mode'] = float(np.mean(scaled.RMSE.to_numpy()**2))
            original = table.loc[table.scale.eq('original')].copy()
            original['dropout_p'] = entry['dropout_p']
            coefficients.append(original)
        if not np.isclose(row['validation_MSE_eval_mode'], meta['best_validation_loss'], rtol=2e-6, atol=1e-7):
            raise ValueError('Exported predictions do not match the best checkpoint.')
        pooled = pd.read_csv(path / 'validation/return_level_overall.csv')
        for period in (50, 100):
            part = pooled.loc[pooled.subset.eq('finite_RL') & pooled.return_period.eq(period)]
            if len(part) != 1 or part.iloc[0].n_total != meta['split_sizes']['validation']*50:
                raise ValueError('Missing or wrong return-level denominator.')
            row[f'RL{period}_RMSE'] = float(part.iloc[0].RMSE)
            row[f'RL{period}_n_used'] = int(part.iloc[0].n_used)
            row[f'RL{period}_n_excluded'] = int(part.iloc[0].n_total-part.iloc[0].n_used)
        rows.append(row)
        year_table = pd.read_csv(path / 'validation/return_level_by_year.csv')
        year_table['dropout_p'] = entry['dropout_p']
        yearly.append(year_table)
        h = pd.read_csv(path / 'training_history.csv')
        h['dropout_p'] = entry['dropout_p']
        history.append(h)
    summary = pd.DataFrame(rows).sort_values('dropout_p').reset_index(drop=True)
    baseline = summary.loc[summary.dropout_p.eq(0)].iloc[0]
    for metric in ('validation_MSE_eval_mode', 'RL50_RMSE', 'RL100_RMSE'):
        summary[metric + '_change_percent'] = 100*(summary[metric]/baseline[metric]-1)
    atomic_csv(directory / 'comparison_summary.csv', summary)
    atomic_csv(directory / 'coefficient_metrics.csv', pd.concat(coefficients, ignore_index=True))
    atomic_csv(directory / 'return_level_by_year.csv', pd.concat(yearly, ignore_index=True))
    atomic_csv(directory / 'training_histories.csv', pd.concat(history, ignore_index=True))
    return summary


def render_report(directory, *, replace=False):
    import nbformat
    from nbclient import NotebookClient
    directory = Path(directory).resolve()
    report = directory / 'cheng_NN_dropout_comparison.ipynb'
    if report.exists() and not replace:
        raise FileExistsError(report)
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    nb = nbformat.v4.new_notebook(cells=[
        md('# Dropout 比較：只改 p，不改節點數\n\n'
           'p = 0、0.1、0.2、0.5；皆為 17→512→512→512→128→128→5。'
           'Dropout 放在五個隱藏 ReLU 層之後；輸入與 linear 輸出不加。'
           '預測與 validation 使用 eval mode，Dropout 關閉。\n\n'
           '固定 8 萬組 train／1 萬組 validation、seed=20260929、Adam lr=0.001、'
           'L2=1e-4、batch=128、最多 300 epochs、early stopping patience=20。'
           '相同 LR scheduler 規則（實際降 LR 的 epoch 可以不同）。'
           '每個 p 以 validation 係數 MSE 選 checkpoint，不以 RL 選 epoch。\n\n'
           '**不使用 test、不覆蓋正式模型。p=0 沿用設定相同的既有完成結果。**'
           '這是單一 seed 的初步比較，不是顯著性檢定；若改善很小，需要多 seed 確認。'),
        code('from pathlib import Path\nimport sys\nimport pandas as pd\nimport matplotlib.pyplot as plt\n'
             'from IPython.display import display\n'
             f'ROOT = Path({str(PROJECT_ROOT)!r})\n'
             'sys.path.insert(0, str(ROOT / "src"))\n'
             'from compare_cheng_nn_dropout import summarize\n'
             f'OUT = Path({str(directory)!r})\n'
             'summary = summarize(OUT)\n'
             'display(summary.round(6))'),
        md('## 1. Validation 的係數與 RL 誤差\n\n'
           '兩種 RMSE 都與已知模擬真值比較。RL 是固定年份的條件年 return level。'
           'RL 數量為 1 萬組 × 50 年，不是 50 萬組獨立序列。'
           'finite_RL 與 GEV support 有效比例分開呈現；表格列出未使用數量。'),
        code('fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)\n'
             'colors = ["#4477AA", "#66CCEE", "#228833", "#CCBB44"]\n'
             'for ax, metric, title in zip(axes, ["validation_MSE_eval_mode", "RL50_RMSE", "RL100_RMSE"], ["Validation coefficient MSE", "RL50(t) RMSE (C)", "RL100(t) RMSE (C)"]):\n'
             '    bars = ax.bar(summary.dropout_p.astype(str), summary[metric], color=colors)\n'
             '    ax.bar_label(bars, fmt="%.4f", padding=3)\n'
             '    ax.set(xlabel="Dropout p", title=title, ylim=(0, summary[metric].max()*1.15))\n'
             'fig.savefig(OUT / "dropout_validation_metrics.png", dpi=160)\nplt.show()'),
        md('## 2. 學習曲線與正確的 train／validation 比較\n\n'
           '曲線中的 train 是訓練中啟用 Dropout 的 online loss；validation 則關閉 Dropout，'
           '**不可把這兩條線的差直接當作泛化落差**。'
           '下方表格另列最佳 checkpoint、關閉 Dropout 後的 train／validation MSE，才是同模式比較。'),
        code('histories = pd.read_csv(OUT / "training_histories.csv")\n'
             'fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True, sharey=True)\n'
             'for ax, p in zip(axes.flat, summary.dropout_p):\n'
             '    h = histories.loc[histories.dropout_p.eq(p)]\n'
             '    best = summary.loc[summary.dropout_p.eq(p)].iloc[0]\n'
             '    ax.plot(h.epoch, h.train_loss, "--", label="Train online (dropout on)")\n'
             '    ax.plot(h.epoch, h.validation_loss, label="Validation (dropout off)")\n'
             '    ax.axvline(best.best_epoch, color="black", linestyle=":", label="Best epoch")\n'
             '    ax.set(title=f"p = {p}", xlabel="Epoch", ylabel="Coefficient MSE")\n'
             '    ax.legend(fontsize=8); ax.grid(alpha=.2)\n'
             'fig.savefig(OUT / "dropout_learning_curves.png", dpi=160)\nplt.show()\n'
             'display(summary[["dropout_p", "train_MSE_eval_mode", "validation_MSE_eval_mode", "GEV_valid_fraction", "best_epoch", "training_seconds"]].round(6))'),
        md('## 3. 五個係數及每年 RL\n\n不同係數的單位不同，請在同一係數內比較 p。'),
        code('coefficients = pd.read_csv(OUT / "coefficient_metrics.csv")\n'
             'val = coefficients.loc[coefficients.split.eq("validation")]\n'
             'display(val.pivot(index="coefficient", columns="dropout_p", values="RMSE").round(5))\n'
             'yearly = pd.read_csv(OUT / "return_level_by_year.csv")\n'
             'fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)\n'
             'for ax, period in zip(axes, [50, 100]):\n'
             '    for p in summary.dropout_p:\n'
             '        part = yearly.loc[yearly.dropout_p.eq(p) & yearly.return_period.eq(period) & yearly.subset.eq("finite_RL")]\n'
             '        ax.plot(part.year, part.RMSE, label=f"p={p}")\n'
             '    ax.set(title=f"RL{period}(t)", xlabel="Year", ylabel="RMSE (C)"); ax.legend()\n'
             'fig.savefig(OUT / "dropout_rl_by_year.png", dpi=160)\nplt.show()'),
    ], metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    # Preserve readable source even if notebook execution is interrupted.
    nbformat.write(nb, report)
    NotebookClient(nb, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(PROJECT_ROOT)}}).execute()
    nbformat.write(nb, report)
    return report


def run_experiment(directory, baseline=BASELINE):
    directory, baseline = Path(directory).resolve(), Path(baseline).resolve()
    reference = read_json(baseline / 'run_metadata.json')
    if (reference.get('dropout_p', 0.0) != 0.0 or reference['seed'] != 20260929
            or reference['split_sizes'] != {'train': 80000, 'validation': 10000}
            or reference['architecture'] != [17, 512, 512, 512, 128, 128, 5]):
        raise ValueError('Use the intended 80k-training, p=0 baseline.')
    baseline_entry = {'dropout_p': 0.0, 'directory': str(baseline), 'reused': True,
                      'checkpoint_sha256': file_sha256(baseline / 'best.pt')}
    validate_one(baseline_entry, reference)
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / 'source_snapshot'
    source.mkdir()
    for file in (PROJECT_ROOT / 'notebooks/cheng_NN.ipynb', Path(__file__),
                 PROJECT_ROOT / 'src/train_cheng_nn.py'):
        shutil.copy2(file, source / file.name)
    manifest = {'status': 'running', 'probabilities': list(PROBABILITIES),
                'seed': reference['seed'], 'n_seeds': 1, 'test_used': False,
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'baseline_conditions': {key: reference[key] for key in FIXED_FIELDS},
                'runs': [baseline_entry], 'completed_runs': 1,
                'dropout_placement': 'after each hidden ReLU, not input/output',
                'notebook_sha256': file_sha256(source / 'cheng_NN.ipynb'),
                'runner_sha256': file_sha256(source / 'train_cheng_nn.py')}
    for p in PROBABILITIES[1:]:
        manifest['runs'].append({'dropout_p': p, 'directory': str(directory / f'p{p:g}_seed{reference["seed"]}'), 'reused': False})
    atomic_json(directory / 'experiment.json', manifest)
    started = time.perf_counter()
    try:
        for entry in manifest['runs'][1:]:
            print(f"TRAINING dropout_p={entry['dropout_p']}; seed={reference['seed']}", flush=True)
            if (file_sha256(PROJECT_ROOT / 'notebooks/cheng_NN.ipynb') != manifest['notebook_sha256']
                    or file_sha256(PROJECT_ROOT / 'src/train_cheng_nn.py') != manifest['runner_sha256']):
                raise ValueError('Training source changed mid-comparison; refusing mixed-code runs.')
            command = [sys.executable, '-u', str(PROJECT_ROOT / 'src/train_cheng_nn.py'),
                '--data-directory', reference['data_directory'], '--run-directory', entry['directory'],
                '--device', reference['device'], '--seed', str(reference['seed']),
                '--threads', str(reference['cpu_threads']), '--epochs', str(reference['max_epochs']),
                '--patience', str(reference['patience']), '--batch-size', str(reference['batch_size']),
                '--learning-rate', str(reference['initial_learning_rate']),
                '--weight-decay', str(reference['weight_decay']), '--dropout-p', str(entry['dropout_p']),
                '--hidden-sizes', *map(str, reference['architecture'][1:-1]), '--no-update-latest']
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            path, _ = validate_one(entry, reference)
            entry['checkpoint_sha256'] = file_sha256(path / 'best.pt')
            manifest['completed_runs'] += 1
            atomic_json(directory / 'experiment.json', manifest)
        print(summarize(directory).to_string(index=False), flush=True)
        manifest.update(status='completed', wall_seconds=time.perf_counter()-started,
                        completed_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(directory / 'experiment.json', manifest)
        print(f'REPORT={render_report(directory)}', flush=True)
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        error=f'{type(exc).__name__}: {exc}', wall_seconds=time.perf_counter()-started)
        atomic_json(directory / 'experiment.json', manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path, default=BASELINE)
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--replace-report', action='store_true')
    args = parser.parse_args()
    if args.report_only:
        print(render_report(args.output_directory, replace=args.replace_report))
    else:
        run_experiment(args.output_directory, args.baseline_run)
