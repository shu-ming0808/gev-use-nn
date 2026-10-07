"""Six paired learning-rate/L2 cases on the fixed 100k Cheng NN dataset.

Only train and validation are loaded. No RL, test evaluation, simulation
regeneration, or publication of a replacement model occurs in this study.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from cheng_nn_simulation import COEFFICIENT_NAMES
from train_cheng_nn import PROJECT_ROOT, atomic_csv, atomic_json, file_sha256

SEEDS = (20260929, 20260930, 20261001)
CASES = {
    'lr1e-3_l2_1e-4': (1e-3, 1e-4),
    'lr1e-3_l2_1e-5': (1e-3, 1e-5),
    'lr1e-3_l2_0': (1e-3, 0.),
    'lr3e-4_l2_1e-4': (3e-4, 1e-4),
    'lr3e-4_l2_1e-5': (3e-4, 1e-5),
    'lr3e-4_l2_0': (3e-4, 0.),
}
BASELINE = 'lr1e-3_l2_1e-4'
FIXED = dict(architecture=[17, 128, 128, 64, 5], optimizer='Adam',
             l1_lambda=0., dropout_p=0., batch_size=128, max_epochs=300,
             patience=20, cpu_threads=4, coefficient_only=True,
             monitor_train_eval=True, monitor_split_rmse=False,
             test_evaluation_requested=False, test_used_for_training_or_selection=False,
             return_periods=[], split_sizes={'train': 80000, 'validation': 10000})


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def make_plan():
    names = list(CASES)
    return [dict(case=name, seed=seed, directory=f'{name}_seed{seed}', status='pending')
            for i, seed in enumerate(SEEDS) for name in names[i:] + names[:i]]


def training_command(data, output, entry, device):
    lr, l2 = CASES[entry['case']]
    return [sys.executable, '-u', str(PROJECT_ROOT / 'src/train_cheng_nn.py'),
            '--data-directory', str(data), '--run-directory', str(output / entry['directory']),
            '--device', device, '--seed', str(entry['seed']), '--threads', '4',
            '--epochs', '300', '--patience', '20', '--batch-size', '128',
            '--learning-rate', str(lr), '--weight-decay', str(l2),
            '--optimizer', 'Adam', '--l1-lambda', '0', '--dropout-p', '0',
            '--hidden-sizes', '128', '128', '64', '--monitor-train-eval',
            '--coefficient-only', '--no-update-latest']


@contextmanager
def experiment_lock(output):
    """The OS releases this lock on interruption, leaving completed runs intact."""
    import msvcrt
    with (output / 'experiment.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def validate_run(output, entry, manifest):
    path = (output / entry['directory']).resolve()
    if not path.is_relative_to(output.resolve()):
        raise ValueError('Run directory escapes experiment.')
    meta = read_json(path / 'run_metadata.json')
    if meta['status'] != 'completed' or meta['seed'] != entry['seed']:
        raise ValueError('Run is incomplete or seed differs.')
    expected = dict(FIXED, device=manifest['device'], initial_learning_rate=CASES[entry['case']][0],
                    weight_decay=CASES[entry['case']][1])
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError('Training settings do not match the locked protocol.')
    if (path / 'test').exists() or 'published_model' in meta:
        raise ValueError('Unexpected test evaluation or model publication.')
    if meta['dataset_files_sha256'] != manifest['dataset_files_sha256']:
        raise ValueError('Dataset differs between runs.')
    if meta['dataset_metadata_sha256'] != manifest['dataset_metadata_sha256']:
        raise ValueError('Dataset metadata differs between runs.')
    for key, source in (('runner_sha256', PROJECT_ROOT / 'src/train_cheng_nn.py'),
                        ('notebook_sha256', PROJECT_ROOT / 'notebooks/cheng_NN.ipynb')):
        if meta[key] != manifest['source_hashes'][str(source)]:
            raise ValueError('Training source differs between runs.')
    checkpoint_hash = file_sha256(path / 'best.pt')
    metrics = {}
    for split in ('train', 'validation'):
        marker = read_json(path / split / 'prediction_metadata.json')
        if (marker['checkpoint_sha256'] != checkpoint_hash or marker['best_epoch'] != meta['best_epoch']
                or marker['dataset_metadata_sha256'] != meta['dataset_metadata_sha256'][split]
                or marker['n_samples'] != FIXED['split_sizes'][split] or marker['return_periods'] != []):
            raise ValueError('Prediction provenance does not match the selected checkpoint.')
        if (path / split / 'return_level_overall.csv').exists():
            raise ValueError('Unexpected RL export in coefficient-only study.')
        frame = pd.read_csv(path / split / 'coefficient_metrics.csv')
        metrics[split] = {}
        for scale in ('original', 'train_target_z'):
            part = frame.loc[frame.scale.eq(scale)].set_index('coefficient')
            if (not part.index.is_unique or set(part.index) != set(COEFFICIENT_NAMES)
                    or not (part.n_used == FIXED['split_sizes'][split]).all()
                    or not (part.n_total == FIXED['split_sizes'][split]).all()
                    or not np.isfinite(part.RMSE).all()):
                raise ValueError('Missing coefficient metrics or silently excluded cases.')
            metrics[split][scale] = part.RMSE.reindex(COEFFICIENT_NAMES).to_numpy()
    normalized = {s: float(np.sqrt(np.mean(v['train_target_z']**2))) for s, v in metrics.items()}
    if not np.isclose(normalized['validation']**2, meta['best_validation_loss'], rtol=2e-6, atol=1e-7):
        raise ValueError('Validation metrics differ from the selected checkpoint.')
    history = pd.read_csv(path / 'training_history.csv')
    best = history.loc[history.epoch.eq(meta['best_epoch'])].iloc[0]
    if not np.isclose(best.train_RMSE_eval, normalized['train'], rtol=2e-6, atol=1e-7):
        raise ValueError('Train metric is not from the same eval-mode checkpoint.')
    return path, meta, metrics, normalized, history


def aggregate(rows, require_complete=True):
    frame = pd.DataFrame(rows)
    keys = ['case', 'seed', 'coefficient']
    expected = {(c, s, p) for c in CASES for s in SEEDS for p in COEFFICIENT_NAMES}
    actual = set(frame[keys].itertuples(index=False, name=None))
    if frame.duplicated(keys).any() or not actual <= expected or (require_complete and actual != expected):
        raise ValueError('Need unique, paired cases/seeds/coefficients.')
    baseline = frame.loc[frame.case.eq(BASELINE), ['seed', 'coefficient', 'validation_RMSE']]
    baseline = baseline.rename(columns={'validation_RMSE': 'baseline_RMSE'})
    frame = frame.merge(baseline, on=['seed', 'coefficient'], how='left', validate='many_to_one')
    frame['improvement_percent'] = 100*(1-frame.validation_RMSE/frame.baseline_RMSE)
    frame['gap_percent'] = 100*(frame.validation_RMSE/frame.train_RMSE-1)
    summary = frame.groupby(['case', 'coefficient'], sort=False).agg(
        n_seeds=('seed', 'size'), train_RMSE_mean=('train_RMSE', 'mean'),
        validation_RMSE_mean=('validation_RMSE', 'mean'), validation_RMSE_SD=('validation_RMSE', 'std'),
        improvement_percent_mean=('improvement_percent', 'mean'),
        paired_baseline_count=('improvement_percent', 'count'), gap_percent_mean=('gap_percent', 'mean'),
    ).reset_index()
    return frame, summary


def summarize(output, complete=True):
    manifest = read_json(output / 'experiment.json')
    rows, diagnostics, histories = [], [], []
    for entry in manifest['runs']:
        if entry['status'] != 'completed':
            continue
        _, meta, metrics, scaled, history = validate_run(output, entry, manifest)
        for j, name in enumerate(COEFFICIENT_NAMES):
            rows.append(dict(case=entry['case'], seed=entry['seed'], coefficient=name,
                             train_RMSE=metrics['train']['original'][j],
                             validation_RMSE=metrics['validation']['original'][j]))
        last = history.iloc[-1]
        diagnostics.append(dict(case=entry['case'], seed=entry['seed'], best_epoch=meta['best_epoch'],
            epochs_completed=meta['epochs_completed'], total_seconds=meta['total_seconds'],
            train_scaled_RMSE=scaled['train'], validation_scaled_RMSE=scaled['validation'],
            scaled_gap_percent=100*(scaled['validation']/scaled['train']-1),
            final_vs_best_validation_percent=100*(last.validation_RMSE_eval/scaled['validation']-1),
            valid_GEV_fraction=meta['evaluation']['validation']['valid_gev_fraction']))
        histories.append(history.assign(case=entry['case'], seed=entry['seed']))
    if not rows:
        return
    paired, summary = aggregate(rows, require_complete=complete)
    diagnostic = pd.DataFrame(diagnostics)
    ranking = diagnostic.groupby('case', sort=False).agg(
        n_seeds=('seed', 'size'), validation_scaled_RMSE_mean=('validation_scaled_RMSE', 'mean'),
        validation_scaled_RMSE_SD=('validation_scaled_RMSE', 'std'),
        train_scaled_RMSE_mean=('train_scaled_RMSE', 'mean'),
        gap_percent_mean=('scaled_gap_percent', 'mean'), valid_GEV_fraction_mean=('valid_GEV_fraction', 'mean'),
        total_seconds=('total_seconds', 'sum')).reset_index()
    ranking['learning_rate'] = ranking.case.map(lambda c: CASES[c][0])
    ranking['L2'] = ranking.case.map(lambda c: CASES[c][1])
    ranking = ranking.sort_values('validation_scaled_RMSE_mean')
    for filename, data in [('rmse_per_seed.csv', paired), ('comparison_summary.csv', summary),
                           ('overfit_diagnostics.csv', diagnostic), ('selection_summary.csv', ranking),
                           ('learning_curves.csv', pd.concat(histories, ignore_index=True))]:
        atomic_csv(output / filename, data)
    wide = summary.pivot(index='case', columns='coefficient', values='validation_RMSE_mean')
    atomic_csv(output / 'five_coefficient_rmse.csv', wide.reindex(columns=COEFFICIENT_NAMES).reset_index())


def render_report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import nbformat
    from nbclient import NotebookClient
    curves = pd.read_csv(output / 'learning_curves.csv')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True, layout='constrained')
    for ax, case in zip(axes.flat, CASES):
        for seed, history in curves.loc[curves.case.eq(case)].groupby('seed'):
            line, = ax.plot(history.epoch, history.validation_RMSE_eval, label=str(seed))
            ax.plot(history.epoch, history.train_RMSE_eval, '--', color=line.get_color(), alpha=.7)
        lr, l2 = CASES[case]
        ax.set_title(f'LR={lr:g}, L2={l2:g}')
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.2)
    axes[0, 0].legend(title='Training seed', fontsize=8)
    axes[0, 0].set_ylabel('Scaled coefficient RMSE')
    axes[1, 0].set_ylabel('Scaled coefficient RMSE')
    fig.suptitle('Small NN: solid = validation, dashed = training (eval mode)')
    figure = output / 'lr_l2_learning_curves.png'
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    nb = nbformat.v4.new_notebook(cells=[
        md('# 小模型：learning rate × L2 比較\n\n'
           '固定 10 萬筆資料中的 80,000 train / 10,000 validation；本輪完全不讀 test。'
           '17 → 128 → 128 → 64 → 5，Adam、Dropout=0、batch=128，最多 300 epochs，early stopping patience=20。'
           '沿用 ReduceLROnPlateau（factor=0.5、patience=6、min_lr=1e-6）。'
           '六組設定各三個相同 seeds，不改模擬範圍、訓練 loss 或正式模型。\n\n'
           '每次依 validation 標準化係數 MSE 選 checkpoint；設定排序使用三個 seeds 的平均 validation 標準化係數 RMSE。'
           '五個原尺度 RMSE 分開呈現，不把不同單位的誤差相加。這是 validation 調參，非獨立 test 結論。'
           '本輪不計算 RL，不進行調參後的顯著性宣稱。'),
        code('from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\n'
             f'OUT=Path({str(output)!r})\n'
             "display(pd.read_csv(OUT/'five_coefficient_rmse.csv').round(6))\n"
             "display(pd.read_csv(OUT/'selection_summary.csv').round(6))"),
        md('## 改善率與跨 seed 變動\n\n'
           'Baseline 為目前 lr=0.001、L2=0.0001。每個 seed 先算 100×(1−候選 RMSE/baseline RMSE)，再平均。'
           '正值改善、負值退步。SD 是三個訓練 seed 間變動，不是信賴區間。'),
        code("display(pd.read_csv(OUT/'comparison_summary.csv').round(6))"),
        md('## 泛化落差與 GEV 支持範圍\n\n'
           '使用同一 checkpoint 的 eval-mode train/validation。小 gap 不等於較好，還要看 validation RMSE。'
           '支持範圍不合的有限係數估計仍納入 RMSE，不偷偷刪除困難案例。'),
        code("display(pd.read_csv(OUT/'overfit_diagnostics.csv').round(6))\n"
             f'display(Image(filename={str(figure)!r}))'),
        md('## 使用限制\n\n'
           '三個 seed 共用同一批 validation，不能視為三批獨立驗證資料。'
           '完成選模後需另外生成未用於調參的最終確認資料，不能以這張表宣稱全球真實資料表現。'),
    ])
    nb.metadata['kernelspec'] = dict(display_name='Python 3', language='python', name='python3')
    NotebookClient(nb, timeout=120, kernel_name='python3', resources={'metadata': {'path': str(PROJECT_ROOT)}}).execute()
    path = output / 'cheng_NN_lr_l2_comparison.ipynb'
    nbformat.write(nb, path)
    return path


def run_experiment(output, device='cuda', resume=False):
    output = Path(output).resolve()
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not (output / 'experiment.json').is_file():
        raise FileNotFoundError('Resume requires an existing experiment.json.')
    with experiment_lock(output):
        data = PROJECT_ROOT / 'data/simulated/cheng_nn_17d'
        source_paths = [Path(__file__), PROJECT_ROOT / 'src/train_cheng_nn.py',
                        PROJECT_ROOT / 'notebooks/cheng_NN.ipynb',
                        PROJECT_ROOT / 'src/cheng_nn_simulation.py', PROJECT_ROOT / 'src/cheng_nn_evaluation.py']
        hashes = {str(p): file_sha256(p) for p in source_paths}
        data_hashes = {s: {p.name: file_sha256(p) for p in sorted((data / s).glob('*.npy'))}
                       for s in ('train', 'validation')}
        metadata_hashes = {s: file_sha256(data / s / 'metadata.json') for s in data_hashes}
        for split, n in FIXED['split_sizes'].items():
            if read_json(data / split / 'metadata.json')['n_samples'] != n:
                raise ValueError('This study requires the unchanged 80k/10k dataset.')
        if resume:
            manifest = read_json(output / 'experiment.json')
            if (manifest['source_hashes'] != hashes or manifest['dataset_files_sha256'] != data_hashes
                    or manifest['dataset_metadata_sha256'] != metadata_hashes or manifest['device'] != device):
                raise ValueError('Sources, data, or device changed; do not mix experiments.')
        else:
            snapshot = output / 'source_snapshot'
            snapshot.mkdir()
            for path in source_paths:
                shutil.copy2(path, snapshot / path.name)
            manifest = dict(status='running', created_utc=utc_now(), device=device,
                cases=CASES, seeds=list(SEEDS), fixed=FIXED, source_hashes=hashes,
                dataset_files_sha256=data_hashes, dataset_metadata_sha256=metadata_hashes,
                test_used=False, no_RL=True, completed_runs=0, runs=make_plan(), session_seconds=[])
            atomic_json(output / 'experiment.json', manifest)
        started = time.perf_counter()
        try:
            for number, entry in enumerate(manifest['runs'], 1):
                if entry['status'] == 'completed':
                    validate_run(output, entry, manifest)
                    continue
                if any(file_sha256(p) != digest for p, digest in hashes.items()):
                    raise RuntimeError('Sources changed during training; stopped before mixing implementations.')
                # Never delete interrupted checkpoints. A resumed incomplete run restarts in a new attempt.
                base = f"{entry['case']}_seed{entry['seed']}"
                attempt = 1
                while (output / entry['directory']).exists() or (output / (entry['directory']+'.log')).exists():
                    attempt += 1
                    entry['directory'] = f'{base}_attempt{attempt}'
                entry.update(status='running', started_utc=utc_now())
                manifest.update(status='running', current_run=number)
                atomic_json(output / 'experiment.json', manifest)
                print(f"RUN {number}/18: {entry['case']}, seed={entry['seed']}", flush=True)
                subprocess.run(training_command(data, output, entry, device), cwd=PROJECT_ROOT,
                               check=True, stdout=subprocess.DEVNULL)
                _, meta, _, _, _ = validate_run(output, entry, manifest)
                entry.update(status='completed', total_seconds=meta['total_seconds'], completed_utc=utc_now())
                manifest['completed_runs'] = sum(e['status'] == 'completed' for e in manifest['runs'])
                atomic_json(output / 'experiment.json', manifest)
                summarize(output, complete=False)
                print(f"COMPLETED {manifest['completed_runs']}/18 ({meta['total_seconds']:.1f} s)", flush=True)
            summarize(output)
            manifest['status'] = 'rendering_report'
            atomic_json(output / 'experiment.json', manifest)
            report = render_report(output)
            manifest['session_seconds'].append(time.perf_counter()-started)
            manifest.update(status='completed', completed_utc=utc_now(), report=str(report),
                            wall_seconds=sum(manifest['session_seconds']))
            atomic_json(output / 'experiment.json', manifest)
            print(f'REPORT={report}', flush=True)
        except BaseException as exc:
            manifest['session_seconds'].append(time.perf_counter()-started)
            manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                            error=f'{type(exc).__name__}: {exc}', wall_seconds=sum(manifest['session_seconds']))
            atomic_json(output / 'experiment.json', manifest)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.report_only:
        summarize(args.output_directory.resolve())
        print(render_report(args.output_directory.resolve()))
    else:
        run_experiment(args.output_directory, args.device, args.resume)
