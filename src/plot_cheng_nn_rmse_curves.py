"""Plot two fixed L2 runs; each epoch has real train/validation/test metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_cheng_nn import PROJECT_ROOT

SPLITS = ('train', 'validation', 'test')
COLORS = {'train': '#2563eb', 'validation': '#ed8b23', 'test': '#17864a'}
LABELS = {
    'coefficients_train_z': 'Combined coefficient RMSE (train-z scale)',
    'mu0': r'$\mu_0$ RMSE ($^\circ$C)',
    'beta_mu': r'$\beta_\mu$ RMSE ($^\circ$C / decade)',
    'eta0': r'$\eta_0$ RMSE (log scale)',
    'beta_sigma': r'$\beta_\sigma$ RMSE (per decade)',
    'xi0': r'$\xi_0$ RMSE',
    'RL20': r'$RL_{20}(t)$ RMSE ($^\circ$C)',
    'RL100': r'$RL_{100}(t)$ RMSE ($^\circ$C)',
}


def read_run(path):
    meta = json.loads((path / 'run_metadata.json').read_text(encoding='utf-8'))
    if meta['status'] != 'completed' or not meta.get('monitor_split_rmse') or meta['test_used_for_training_or_selection']:
        raise ValueError('Need completed diagnostic-only split monitoring.')
    frame = pd.read_csv(path / 'epoch_rmse.csv')
    if frame.duplicated(['epoch', 'split', 'outcome']).any() or not np.isfinite(frame.RMSE).all():
        raise ValueError('Invalid curve metrics.')
    expected = pd.MultiIndex.from_product([range(1, meta['epochs_completed'] + 1), SPLITS, LABELS],
                                         names=['epoch', 'split', 'outcome'])
    actual = pd.MultiIndex.from_frame(frame[['epoch', 'split', 'outcome']])
    if len(expected.difference(actual)) or len(actual.difference(expected)):
        raise ValueError('Every epoch needs all split/outcome metrics; do not interpolate test.')
    for split in SPLITS:
        if not frame.loc[frame.split.eq(split), 'n_samples'].eq(meta['split_sizes'][split]).all():
            raise ValueError('Split size changed within curves.')
    return frame, meta


def create_figures(runs, output):
    output.mkdir(parents=True, exist_ok=True)
    last_epoch = max(meta['epochs_completed'] for _, _, meta in runs)
    groups = [('rmse_learning_curves', ['coefficients_train_z']),
              ('rmse_parameter_curves', ['mu0', 'beta_mu', 'eta0', 'beta_sigma', 'xi0']),
              ('rmse_return_level_curves', ['RL20', 'RL100'])]
    files = []
    for filename, outcomes in groups:
        fig, axes = plt.subplots(len(outcomes), 2, figsize=(12, 3.25*len(outcomes)+.6),
                                 squeeze=False, sharey='row', layout='constrained')
        for col, (name, frame, meta) in enumerate(runs):
            for row, outcome in enumerate(outcomes):
                ax = axes[row, col]
                for split in SPLITS:
                    part = frame.loc[frame.outcome.eq(outcome) & frame.split.eq(split)].sort_values('epoch')
                    ax.plot(part.epoch, part.RMSE, color=COLORS[split], label=split.capitalize(), linewidth=1.65)
                ax.axvline(meta['best_epoch'], color='#666666', ls=':', lw=1.2,
                           label=f"Best validation epoch: {meta['best_epoch']}")
                ax.set(title=f'{name} + L2', xlabel='Epoch', ylabel=LABELS[outcome])
                ax.set_xlim(1, max(2, last_epoch))
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
        fig.suptitle('Same end-of-epoch weights; test is diagnostic only', fontsize=12)
        path = output / f'{filename}.png'
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(path)
    return files


def report(large, small, output):
    runs = [('5 hidden layers', *read_run(large)), ('3 hidden layers', *read_run(small))]
    if (runs[0][2]['architecture'] != [17, 512, 512, 512, 128, 128, 5]
            or runs[1][2]['architecture'] != [17, 128, 128, 64, 5]):
        raise ValueError('Run paths must match the labeled large/small architectures.')
    fields = ('seed', 'dataset_files_sha256', 'dataset_metadata_sha256', 'split_sizes',
              'optimizer', 'initial_learning_rate', 'weight_decay', 'dropout_p',
              'l1_lambda', 'batch_size', 'patience', 'max_epochs', 'return_periods',
              'loss', 'device', 'cpu_threads', 'torch_version', 'notebook_sha256', 'runner_sha256')
    if any(runs[0][2][k] != runs[1][2][k] for k in fields):
        raise ValueError('Training settings/data differ; cannot label a controlled comparison.')
    images = create_figures(runs, output)
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.v4.new_notebook()
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.cells = [nbformat.v4.new_markdown_cell(
        '# 三層／五層 L2：Train、Validation、Test RMSE 曲線\n\n'
        '各重新訓練一次，固定 seed=20260929，原 10 萬組資料（80k/10k/10k）、Adam、lr=0.001、'
        'L2=1e-4、batch=128、Dropout=0、patience=20、最多 300 epochs。\n\n'
        '每個 epoch 使用同一組結束權重、eval 模式計算三個 split，並非 batch 更新中的 train loss。'
        '合併係數 RMSE 使用只由 train 配適的 z-scale；另列五係數原尺度與 RL20/RL100。'
        'RL 誤差合併每筆序列與 50 個年份計算，並未排除 support 違反案例。\n\n'
        '灰色直線標記 validation 係數 MSE 最佳 epoch。Test 不參與梯度、scheduler、early stopping 或選模；'
        '但看完 test 曲線後不可再把同一批資料當作全新的最終驗證。本次是單一 seed，不是前次三個 seed 的平均。')]
    for title, path in zip(('合併係數 RMSE', '五個係數', 'RL20 與 RL100'), images):
        nb.cells.append(nbformat.v4.new_markdown_cell(f'## {title}'))
        nb.cells.append(nbformat.v4.new_code_cell(
            'from IPython.display import Image, display\n' + f'display(Image(filename={str(path)!r}))'))
    NotebookClient(nb, timeout=60, kernel_name='python3', resources={'metadata': {'path': str(PROJECT_ROOT)}}).execute()
    destination = output / 'cheng_NN_three_split_rmse.ipynb'
    nbformat.write(nb, destination)
    print(destination, flush=True)
    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--large-run', type=Path, required=True)
    parser.add_argument('--small-run', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    report(args.large_run.resolve(), args.small_run.resolve(), args.output_directory.resolve())
