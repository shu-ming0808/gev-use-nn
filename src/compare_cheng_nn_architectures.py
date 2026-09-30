"""Controlled large/small Cheng NN comparison, three paired seeds, no test use.

Each run trains from scratch on the existing train split and selects its own
checkpoint by validation coefficient MSE. Does not publish/replace a model or
change the single-run diagnostics pointer. Runs serially on one GPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from train_cheng_nn import PROJECT_ROOT, atomic_csv, atomic_json, file_sha256

ARCHITECTURES = {'large': [512, 512, 512, 128, 128], 'small': [128, 128, 64]}
SEEDS = (20260929, 20260930, 20261001)
COMPARABLE_FIELDS = (
    'data_directory', 'dataset_files_sha256', 'dataset_metadata_sha256',
    'device', 'torch_version', 'cpu_threads', 'batch_size', 'optimizer',
    'initial_learning_rate', 'weight_decay', 'regularization', 'max_epochs',
    'patience', 'loss', 'notebook_sha256', 'runner_sha256',
)
METRICS = ['train_MSE', 'validation_MSE', 'validation_RL50_RMSE',
           'validation_RL100_RMSE', 'valid_GEV_fraction', 'training_seconds',
           'total_seconds', 'best_epoch', 'epochs_completed']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_runs(directory):
    """Reject partial, mismatched, duplicate, or test-contaminated comparisons."""
    directory = Path(directory)
    manifest = read_json(directory / 'experiment.json')
    expected = {(name, seed) for name in ARCHITECTURES for seed in manifest['seeds']}
    seen, reference, runs = set(), None, []
    for entry in manifest['runs']:
        key = (entry['model'], entry['seed'])
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicate model/seed.')
        seen.add(key)
        path = (directory / entry['directory']).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError('Run path must stay inside experiment directory.')
        meta = read_json(path / 'run_metadata.json')
        if meta['status'] != 'completed':
            raise ValueError(f'Incomplete run: {path.name}')
        if meta['test_evaluation_requested'] or meta['test_used_for_training_or_selection'] or (path / 'test').exists():
            raise ValueError('Architecture comparison must not evaluate the test split.')
        if meta['seed'] != entry['seed'] or meta['architecture'] != [17, *ARCHITECTURES[entry['model']], 5]:
            raise ValueError('Run architecture/seed does not match the experiment.')
        if set(meta['dataset_files_sha256']) != {'train', 'validation'}:
            raise ValueError('Only train and validation may be loaded.')
        signature = {k: meta[k] for k in COMPARABLE_FIELDS}
        if reference is not None and signature != reference:
            raise ValueError('Training conditions or data differ across runs.')
        reference = signature
        checkpoint_hash = file_sha256(path / 'best.pt')
        for split in ('train', 'validation'):
            marker = read_json(path / split / 'prediction_metadata.json')
            if (marker['checkpoint_sha256'] != checkpoint_hash
                    or marker['best_epoch'] != meta['best_epoch']
                    or marker['dataset_metadata_sha256'] != meta['dataset_metadata_sha256'][split]):
                raise ValueError('Prediction provenance does not match the checkpoint/data.')
        runs.append((entry, path, meta))
    if seen != expected:
        raise ValueError('Both architectures need every planned seed before reporting.')
    return manifest, runs


def summarize(directory):
    directory = Path(directory)
    manifest, runs = validate_runs(directory)
    rows, coefficient_rows = [], []
    for entry, path, meta in runs:
        row = {'model': entry['model'], 'seed': entry['seed'],
               'trainable_parameters': meta['trainable_parameters']}
        for key in ('training_seconds', 'total_seconds', 'best_epoch', 'epochs_completed'):
            row[key] = meta[key]
        for split in ('train', 'validation'):
            table = pd.read_csv(path / split / 'coefficient_metrics.csv')
            scaled = table.loc[table['scale'].eq('train_target_z')]
            if len(scaled) != 5 or not (scaled['n_total'] == scaled['n_used']).all():
                raise ValueError('Missing or nonfinite coefficient evaluations.')
            row[f'{split}_MSE'] = float(np.mean(scaled['RMSE'].to_numpy() ** 2))
            original = table.loc[table['scale'].eq('original')].copy()
            original['model'], original['seed'] = entry['model'], entry['seed']
            coefficient_rows.append(original)
        if not np.isclose(row['validation_MSE'], meta['best_validation_loss'], rtol=2e-6, atol=1e-7):
            raise ValueError('Exported predictions do not match selected validation MSE.')
        rl = pd.read_csv(path / 'validation/return_level_overall.csv')
        for period in (50, 100):
            selected = rl.loc[rl['subset'].eq('finite_RL') & rl['return_period'].eq(period)]
            if len(selected) != 1:
                raise ValueError('Missing pooled return-level metric.')
            row[f'validation_RL{period}_RMSE'] = selected.iloc[0]['RMSE']
            row[f'RL{period}_n_used'] = int(selected.iloc[0]['n_used'])
            row[f'RL{period}_n_total'] = int(selected.iloc[0]['n_total'])
        row['valid_GEV_fraction'] = meta['evaluation']['validation']['valid_gev_fraction']
        rows.append(row)
    per_run = pd.DataFrame(rows).sort_values(['model', 'seed']).reset_index(drop=True)
    coefficients = pd.concat(coefficient_rows, ignore_index=True)
    aggregate = per_run.groupby('model')[METRICS].agg(['mean', 'std'])
    aggregate.columns = ['_'.join(c) for c in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate.insert(1, 'n_seeds', aggregate['model'].map(per_run.groupby('model').size()))
    paired = per_run.set_index(['seed', 'model'])[METRICS].unstack('model')
    differences = pd.DataFrame({'seed': paired.index})
    for metric in METRICS:
        differences[f'{metric}_small_minus_large'] = (paired[(metric, 'small')] - paired[(metric, 'large')]).to_numpy()
    atomic_csv(directory / 'comparison_per_run.csv', per_run)
    atomic_csv(directory / 'comparison_summary.csv', aggregate)
    atomic_csv(directory / 'paired_seed_differences.csv', differences)
    atomic_csv(directory / 'coefficient_metrics_all_runs.csv', coefficients)
    return manifest, runs, per_run, aggregate, coefficients


def render_report(directory, *, replace=False):
    """Create a reproducible executed notebook; reading it never trains models."""
    import nbformat
    from nbclient import NotebookClient

    directory = Path(directory).resolve()
    relative = directory.relative_to(PROJECT_ROOT).as_posix()
    nb = nbformat.v4.new_notebook()
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            '# 大、小網路比較：固定 17 維輸入\n\n'
            'Large：17→512→512→512→128→128→5；Small：17→128→128→64→5。\n\n'
            '相同 8 萬組 train、1 萬組 validation；每組 50 年。相同 Adam、L2=1e-4、'
            'batch=128、early stopping=20、最多 300 epochs。兩種網路各跑 3 個相同種子；'
            '**不讀取 test、不產生新模擬資料、不替換正式模型。**\n\n'
            '每次由 validation 係數 MSE 選取 checkpoint。MSE 是訓練集標準化後五係數的平均誤差，'
            '不含 L2 懲罰；RL RMSE 單位為 °C，合併 1 萬序列 × 50 年的條件式 RL 誤差。'
            'finite_RL 與 GEV support 有效比例分開呈現，不悄悄排除 support 違反的案例。'
            '平均 ± SD 表示三個訓練種子的變動，不是資料抽樣信賴區間，也不作顯著性宣稱。'),
        nbformat.v4.new_code_cell(
            'from pathlib import Path\nimport sys\nimport numpy as np\nimport pandas as pd\n'
            'import matplotlib.pyplot as plt\nfrom IPython.display import display\n'
            'ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src/train_cheng_nn.py").is_file())\n'
            'sys.path.insert(0, str(ROOT / "src"))\n'
            'from compare_cheng_nn_architectures import summarize\n'
            f'EXPERIMENT = ROOT / {relative!r}\n'
            'manifest, runs, per_run, summary, coefficients = summarize(EXPERIMENT)\n'
            'display(per_run.round(5))\n'),
        nbformat.v4.new_markdown_cell('## 1. 平均與種子間 SD\n\n數值越低越好；GEV 有效比例則越高越好。下方配對圖的縱軸為局部放大，線段斜率看似明顯，不代表實際改善幅度很大；請以表格數值為準。'),
        nbformat.v4.new_code_cell(
            'shown = pd.DataFrame(index=summary.model)\n'
            'for metric in ["validation_MSE", "validation_RL50_RMSE", "validation_RL100_RMSE", "training_seconds", "valid_GEV_fraction"]:\n'
            '    shown[metric] = [f"{row[metric + \'_mean\']:.5f} ± {row[metric + \'_std\']:.5f}" for _, row in summary.iterrows()]\n'
            'display(shown)\n'
            'display(per_run[["model", "seed", "RL50_n_used", "RL50_n_total", "RL100_n_used", "RL100_n_total"]])\n'
            'fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)\n'
            'for ax, metric, title in zip(axes, ["validation_MSE", "validation_RL50_RMSE", "validation_RL100_RMSE"], ["Validation coefficient MSE", "Validation RL50 RMSE (°C)", "Validation RL100 RMSE (°C)"]):\n'
            '    for _, part in per_run.groupby("seed"):\n'
            '        part = part.set_index("model").reindex(["large", "small"])\n'
            '        ax.plot([0, 1], part[metric], "o-", alpha=0.65, label=str(int(part.seed.iloc[0])))\n'
            '    ax.set_xticks([0, 1], ["Large", "Small"]); ax.set_title(title); ax.grid(alpha=.2)\n'
            'axes[0].legend(title="Paired seed", fontsize=8)\n'
            'fig.savefig(EXPERIMENT / "comparison_metrics.png", dpi=160)\nplt.show()'),
        nbformat.v4.new_markdown_cell('## 2. 學習曲線\n\nTrain 為每個 epoch 更新過程的平均 loss；validation 用當時模型評估。實線 validation、虛線 train；圓點標記最佳 validation epoch。'),
        nbformat.v4.new_code_cell(
            'fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True, constrained_layout=True)\n'
            'for ax, model in zip(axes, ["large", "small"]):\n'
            '    for entry, path, meta in runs:\n'
            '        if entry["model"] != model: continue\n'
            '        h = pd.read_csv(path / "training_history.csv")\n'
            '        line, = ax.plot(h.epoch, h.validation_loss, label=str(entry["seed"]))\n'
            '        ax.plot(h.epoch, h.train_loss, "--", color=line.get_color(), alpha=.5)\n'
            '        best = h.loc[h.epoch.eq(meta["best_epoch"])].iloc[0]\n'
            '        ax.scatter([best.epoch], [best.validation_loss], color=line.get_color())\n'
            '    ax.set_title(model.capitalize()); ax.set_xlabel("Epoch"); ax.set_ylim(.30, .65); ax.grid(alpha=.2); ax.legend(title="Seed")\n'
            'axes[0].set_ylabel("Coefficient MSE (zoom: 0.30–0.65)")\n'
            'fig.savefig(EXPERIMENT / "comparison_learning_curves.png", dpi=160)\nplt.show()'),
        nbformat.v4.new_markdown_cell('## 3. 五係數 validation RMSE\n\n各係數單位不同，請比較同一子圖的大／小網路，不直接比較不同係數的 RMSE 大小。誤差棒為三個種子的 SD，不是信賴區間；縱軸是局部放大。'),
        nbformat.v4.new_code_cell(
            'fig, axes = plt.subplots(1, 5, figsize=(15, 3.3), constrained_layout=True)\n'
            'val = coefficients.loc[coefficients.split.eq("validation")]\n'
            'for ax, name in zip(axes, val.coefficient.drop_duplicates()):\n'
            '    part = val.loc[val.coefficient.eq(name)].groupby("model").RMSE.agg(["mean", "std"]).reindex(["large", "small"])\n'
            '    ax.errorbar([0, 1], part["mean"], yerr=part["std"], fmt="o", capsize=5)\n'
            '    ax.ticklabel_format(axis="y", style="plain", useOffset=False)\n'
            '    ax.set_xticks([0, 1], ["Large", "Small"]); ax.set_title(name); ax.grid(alpha=.2)\n'
            'axes[0].set_ylabel("Validation RMSE (original coefficient scale)")\n'
            'fig.savefig(EXPERIMENT / "comparison_coefficient_rmse.png", dpi=160)\nplt.show()'),
    ]
    report = directory / 'cheng_NN_architecture_comparison.ipynb'
    if report.exists() and not replace:
        raise FileExistsError(report)
    NotebookClient(nb, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(PROJECT_ROOT)}}).execute()
    nbformat.write(nb, report)
    return report


def run_experiment(directory, device='cuda'):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'seeds': list(SEEDS), 'architectures': ARCHITECTURES,
                'created_utc': datetime.now(timezone.utc).isoformat(), 'test_used': False, 'runs': []}
    # Alternate order to avoid always running one architecture earlier.
    for i, seed in enumerate(SEEDS):
        for name in (('large', 'small') if i % 2 == 0 else ('small', 'large')):
            manifest['runs'].append({'model': name, 'seed': seed, 'directory': f'{name}_seed_{seed}'})
    atomic_json(directory / 'experiment.json', manifest)
    started = time.perf_counter()
    try:
        for number, entry in enumerate(manifest['runs'], 1):
            print(f"RUN {number}/6: {entry['model']} seed={entry['seed']}", flush=True)
            command = [sys.executable, '-u', str(PROJECT_ROOT / 'src/train_cheng_nn.py'),
                       '--data-directory', str(PROJECT_ROOT / 'data/simulated/cheng_nn_17d'),
                       '--run-directory', str(directory / entry['directory']), '--device', device,
                       '--seed', str(entry['seed']), '--weight-decay', '0.0001', '--no-update-latest',
                       '--hidden-sizes', *map(str, ARCHITECTURES[entry['model']])]
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            manifest['completed_runs'] = number
            atomic_json(directory / 'experiment.json', manifest)
        summarize(directory)
        manifest.update(status='completed', wall_seconds=time.perf_counter()-started,
                        completed_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(directory / 'experiment.json', manifest)
        report = render_report(directory)
        print(f'REPORT={report}', flush=True)
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        error=f'{type(exc).__name__}: {exc}', wall_seconds=time.perf_counter()-started)
        atomic_json(directory / 'experiment.json', manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--report-only', action='store_true', help='Validate completed runs and create a notebook without training.')
    parser.add_argument('--replace-report', action='store_true', help='With --report-only, regenerate the existing report/figures, never checkpoints.')
    args = parser.parse_args()
    if args.report_only:
        print(render_report(args.output_directory, replace=args.replace_report))
    else:
        run_experiment(args.output_directory, args.device)
