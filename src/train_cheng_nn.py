"""Run the notebook's Cheng NN with durable per-epoch checkpoints.

The architecture, loss, scaler and fitting loop are read from cheng_NN.ipynb,
so this command and the interactive notebook use the same implementation.
No simulation is regenerated. Test predictions are opt-in; per-epoch test
monitoring is diagnostic only and never controls optimization or selection.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Tee:
    """Mirror progress to a persistent UTF-8 run log and the terminal."""
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_array(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def atomic_csv(path, frame):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_notebook_components():
    """Load definitions; never execute training, evaluation or generation cells."""
    path = PROJECT_ROOT / 'notebooks' / 'cheng_NN.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    sources = {c['id']: ''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'}
    namespace = {'__name__': 'cheng_training_notebook'}
    previous = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        exec(compile(sources['cheng-setup'], str(path) + ':setup', 'exec'), namespace)
        assert not any(namespace[key] for key in ('RUN_SIMULATION', 'RUN_TRAINING', 'RUN_EVALUATION'))
        exec(compile(sources['cheng-generate'], str(path) + ':configuration', 'exec'), namespace)
        for cell_id in ('cheng-load', 'cheng-model', 'cheng-dataset', 'cheng-training-functions', 'cheng-evaluation'):
            tree = ast.parse(sources[cell_id])
            definitions = ast.Module(body=[n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))], type_ignores=[])
            exec(compile(definitions, str(path) + ':' + cell_id, 'exec'), namespace)
    finally:
        os.chdir(previous)
    return namespace


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def export_predictions(ns, model, scaler, data, split, run_dir, metadata, device):
    """Evaluate the same selected checkpoint on each split without updates."""
    directory = run_dir / split
    directory.mkdir(exist_ok=False)
    standardized = ns['predict_standardized_coefficients'](
        model, data['inputs'], scaler, device, batch_size=2048,
    )
    original = ns['inverse_sample_standardization'](standardized, data['location_scale'])
    atomic_array(directory / 'predictions_standardized.npy', standardized)
    atomic_array(directory / 'predictions_original.npy', original)
    rows = []
    for scale, predicted, truth in (
        ('original', original, data['coefficients']),
        ('sample_standardized', standardized, data['targets']),
        ('train_target_z', scaler.transform(standardized), scaler.transform(data['targets'])),
    ):
        errors = np.asarray(predicted, dtype=float) - np.asarray(truth, dtype=float)
        for j, name in enumerate(ns['COEFFICIENT_NAMES']):
            use = np.isfinite(errors[:, j])
            e = errors[use, j]
            rows.append({'split': split, 'scale': scale, 'coefficient': name,
                         'n_total': len(errors), 'n_used': int(use.sum()),
                         'RMSE': float(np.sqrt(np.mean(e**2))) if len(e) else np.nan,
                         'MAE': float(np.mean(np.abs(e))) if len(e) else np.nan,
                         'bias': float(np.mean(e)) if len(e) else np.nan})
    atomic_csv(directory / 'coefficient_metrics.csv', pd.DataFrame(rows))
    config = ns['SIMULATION_CONFIG']
    options = dict(reference_year=config.reference_year, time_scale_years=config.time_scale_years)
    periods = tuple(metadata.get('return_periods', (50., 100.)))
    validity = ns['check_gev_predictions'](original, data['annual_maxima'], config.years, **options)
    atomic_csv(directory / 'gev_validity.csv', validity)
    if not metadata.get('coefficient_only', False):
        true_rl = ns['conditional_return_levels'](data['coefficients'], config.years, periods, **options)
        predicted_rl = ns['conditional_return_levels'](original, config.years, periods, **options)
        yearly, pooled = ns['return_level_recovery_metrics'](
            predicted_rl, true_rl, config.years, periods, validity['gev_valid_all_years'].to_numpy(),
        )
        atomic_csv(directory / 'return_level_by_year.csv', yearly)
        atomic_csv(directory / 'return_level_overall.csv', pooled)
    atomic_json(directory / 'prediction_metadata.json', {
        'checkpoint': 'best.pt', 'checkpoint_sha256': file_sha256(run_dir / 'best.pt'),
        'best_epoch': metadata['best_epoch'], 'n_samples': len(standardized),
        'dataset_metadata_sha256': metadata['dataset_metadata_sha256'][split],
        'return_periods': list(periods),
        'coefficient_only': metadata.get('coefficient_only', False),
        'completed_utc': datetime.now(timezone.utc).isoformat(),
    })
    return {'valid_gev_fraction': float(validity['gev_valid_all_years'].mean()),
            'n_samples': len(standardized)}


class SplitRMSEMonitor:
    """End-of-epoch metrics; no gradients, data loaders, updates or selection."""
    def __init__(self, ns, scaler, data, periods, device):
        self.ns, self.scaler, self.data = ns, scaler, data
        self.periods, self.device = periods, device
        cfg = ns['SIMULATION_CONFIG']
        self.years = cfg.years
        self.options = dict(reference_year=cfg.reference_year, time_scale_years=cfg.time_scale_years)
        self.truth_rl = {s: ns['conditional_return_levels'](d['coefficients'], self.years, periods, **self.options)
                         for s, d in data.items()}

    def __call__(self, model, epoch):
        was_training = model.training
        rows = []
        try:
            for split, data in self.data.items():
                standardized = self.ns['predict_standardized_coefficients'](
                    model, data['inputs'], self.scaler, self.device, batch_size=2048)
                original = self.ns['inverse_sample_standardization'](standardized, data['location_scale'])
                z_error = self.scaler.transform(standardized) - self.scaler.transform(data['targets'])
                error = original - np.asarray(data['coefficients'], dtype=float)
                predicted_rl = self.ns['conditional_return_levels'](original, self.years, self.periods, **self.options)
                rl_error = predicted_rl - self.truth_rl[split]
                if not all(np.isfinite(a).all() for a in (z_error, error, rl_error)):
                    raise FloatingPointError('Nonfinite diagnostic prediction; no cases may be silently omitted.')
                values = {'coefficients_train_z': float(np.sqrt(np.mean(z_error.astype(float)**2)))}
                values.update(zip(self.ns['COEFFICIENT_NAMES'], np.sqrt(np.mean(error**2, axis=0))))
                for j, period in enumerate(self.periods):
                    values[f'RL{period:g}'] = float(np.sqrt(np.mean(rl_error[..., j]**2)))
                rows.extend(dict(epoch=epoch, split=split, outcome=k, RMSE=float(v),
                                 n_samples=len(original)) for k, v in values.items())
        finally:
            model.train(was_training)
        return rows


def run_training(args):
    ns = load_notebook_components()
    optimizer_name = getattr(args, 'optimizer', 'Adam')
    l1_lambda = float(getattr(args, 'l1_lambda', 0.0))
    periods = tuple(float(p) for p in getattr(args, 'return_periods', (50., 100.)))
    monitor_train_eval = bool(getattr(args, 'monitor_train_eval', False))
    monitor_split_rmse = bool(getattr(args, 'monitor_split_rmse', False))
    coefficient_only = bool(getattr(args, 'coefficient_only', False))
    if coefficient_only and monitor_split_rmse:
        raise ValueError('--coefficient-only supports --monitor-train-eval, not the RL split monitor.')
    if monitor_split_rmse and not args.evaluate_test:
        raise ValueError('--monitor-split-rmse requires explicit --evaluate-test.')
    if optimizer_name not in ('Adam', 'AdamW'):
        raise ValueError('Optimizer must be Adam or AdamW.')
    if not np.isfinite(l1_lambda) or l1_lambda < 0:
        raise ValueError('L1 lambda must be finite and nonnegative.')
    if not periods or len(set(periods)) != len(periods) or not all(np.isfinite(p) and p > 1 for p in periods):
        raise ValueError('Return periods must be distinct, finite, and greater than one.')
    if coefficient_only:
        periods = ()
    dropout_p = float(getattr(args, 'dropout_p', 0.0))
    if not np.isfinite(dropout_p) or not 0.0 <= dropout_p < 1.0:
        raise ValueError('Dropout probability must be finite and in [0, 1).')
    hidden_sizes = tuple(getattr(args, 'hidden_sizes', (512, 512, 512, 128, 128)))
    if not hidden_sizes or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in hidden_sizes):
        raise ValueError('Hidden sizes must be positive integers.')
    architecture = [17, *hidden_sizes, 5]
    if args.threads < 1 or args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError('Threads, epochs, patience and batch size must be positive.')
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError('Weight decay must be finite and nonnegative.')
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device == 'auto':
        device = 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('This PyTorch installation cannot use CUDA.')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    # No cuDNN/TF32/AMP shortcuts: retain the requested float32 baseline.
    ns['DATA_DIR'] = args.data_directory.resolve()
    ns['LEARNING_RATE'] = args.learning_rate
    ns['WEIGHT_DECAY'] = args.weight_decay
    evaluated_splits = ('train', 'validation', 'test') if args.evaluate_test else ('train', 'validation')
    # Explicit old/new data directories remain usable when notebook defaults change.
    # The notebook loader still validates the design and actual array dimensions.
    size_keys = {'train': 'N_TRAIN', 'validation': 'N_VALIDATION', 'test': 'N_TEST'}
    for split in evaluated_splits:
        split_metadata = json.loads((args.data_directory / split / 'metadata.json').read_text(encoding='utf-8'))
        count = split_metadata.get('n_samples')
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f'{split}: invalid sample count in metadata.')
        ns[size_keys[split]] = count
    data = {split: ns['load_split'](split) for split in evaluated_splits}
    if args.publish_model and args.publish_model.exists():
        raise FileExistsError(f'Refusing to overwrite existing model: {args.publish_model}')
    run_dir = args.run_directory.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    metadata = {
        'status': 'running', 'run_directory': str(run_dir),
        'data_directory': str(args.data_directory.resolve()),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'pid': os.getpid(), 'device': device, 'torch_version': str(torch.__version__),
        'cpu_threads': args.threads, 'seed': args.seed, 'batch_size': args.batch_size,
        'optimizer': optimizer_name, 'initial_learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'l1_lambda': l1_lambda, 'l1_definition': 'lambda * sum(abs(all weights and biases))',
        'return_periods': list(periods), 'monitor_train_eval': monitor_train_eval,
        'coefficient_only': coefficient_only,
        'monitor_split_rmse': monitor_split_rmse,
        'test_monitoring_role': 'diagnostic_only' if monitor_split_rmse else 'final_only',
        'checkpoint_selection': 'validation coefficient MSE only',
        'dropout_p': dropout_p, 'dropout_placement': 'after_each_hidden_relu',
        'regularization': ('Adam coupled L2 on all trainable parameters' if optimizer_name == 'Adam'
                           else 'AdamW decoupled decay on all trainable parameters'),
        'reported_loss_includes_regularization': False,
        'test_evaluation_requested': args.evaluate_test,
        'split_sizes': {split: len(data[split]['inputs']) for split in data},
        'max_epochs': args.epochs, 'patience': args.patience,
        'architecture': architecture,
        'trainable_parameters': sum((a + 1) * b for a, b in zip(architecture[:-1], architecture[1:])),
        'loss': 'mean squared error of train-only z-scaled coefficient targets',
        'dataset_metadata_sha256': {k: file_sha256(args.data_directory / k / 'metadata.json') for k in data},
        'dataset_files_sha256': {k: {p.name: file_sha256(p) for p in sorted((args.data_directory / k).glob('*.npy'))} for k in data},
        'notebook_sha256': file_sha256(PROJECT_ROOT / 'notebooks' / 'cheng_NN.ipynb'),
        'runner_sha256': file_sha256(Path(__file__)),
        'best_epoch': None, 'best_validation_loss': None,
        'test_used_for_training_or_selection': False,
    }
    atomic_json(run_dir / 'run_metadata.json', metadata)
    pointer = PROJECT_ROOT / 'results' / 'cheng_nn_17d' / 'latest_run.json'
    pointer.parent.mkdir(parents=True, exist_ok=True)
    if not getattr(args, 'no_update_latest', False):
        atomic_json(pointer, {'run_dir': str(run_dir)})
    scaler = ns['TargetScaler'].fit(data['train']['targets'])
    generator = torch.Generator().manual_seed(args.seed)
    # Data are only ~50 MiB, and single-process loaders avoid Windows worker startup.
    loaders = {
        split: DataLoader(ns['ChengSimulationDataset'](data[split]['inputs'], data[split]['targets'], scaler),
                          batch_size=args.batch_size, shuffle=(split == 'train'), num_workers=0,
                          pin_memory=(device == 'cuda'), generator=generator if split == 'train' else None)
        for split in ('train', 'validation')
    }
    # Separate, non-shuffled eval loader: diagnostics never consume the train
    # shuffling stream or global torch RNG (important when Dropout is enabled).
    train_eval_loader = None
    if monitor_train_eval:
        train_eval_loader = DataLoader(
            loaders['train'].dataset, batch_size=2048, shuffle=False, num_workers=0,
            pin_memory=(device == 'cuda'), generator=torch.Generator().manual_seed(args.seed),
        )

    metric_monitor = SplitRMSEMonitor(ns, scaler, data, periods, device) if monitor_split_rmse else None
    epoch_metrics = []

    def checkpoint_epoch(model, optimizer, scheduler, history, improved, stale_epochs):
        if metric_monitor is not None:
            diagnostic_started = time.perf_counter()
            epoch_metrics.extend(metric_monitor(model, history[-1]['epoch']))
            diagnostic_seconds = time.perf_counter() - diagnostic_started
            history[-1]['diagnostic_seconds'] = diagnostic_seconds
            history[-1]['seconds'] += diagnostic_seconds
            atomic_csv(run_dir / 'epoch_rmse.csv', pd.DataFrame(epoch_metrics))
        if improved:
            metadata['best_epoch'] = history[-1]['epoch']
            metadata['best_validation_loss'] = history[-1]['validation_loss']
        metadata['last_completed_epoch'] = history[-1]['epoch']
        metadata['elapsed_seconds'] = time.perf_counter() - started
        frame = pd.DataFrame(history)
        frame['elapsed_seconds'] = frame['seconds'].cumsum()
        payload = {
            'model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'target_scaler': scaler.state_dict(), 'coefficient_names': list(ns['COEFFICIENT_NAMES']),
            'input_shape': [17], 'simulation_metadata': data['train']['metadata'],
            'hidden_sizes': list(hidden_sizes), 'architecture': architecture,
            'dropout_p': dropout_p, 'dropout_placement': 'after_each_hidden_relu',
            'optimizer': optimizer_name, 'initial_learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'l1_lambda': l1_lambda, 'return_periods': list(periods),
            'regularization': metadata['regularization'],
            'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(),
            'epoch': history[-1]['epoch'], 'best_epoch': metadata['best_epoch'],
            'best_validation_loss': metadata['best_validation_loss'],
            'epochs_without_improvement': stale_epochs, 'history': frame.to_dict('records'),
            'torch_rng_state': torch.get_rng_state(), 'shuffle_rng_state': generator.get_state(),
            'numpy_rng_state': np.random.get_state(), 'python_rng_state': random.getstate(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if device == 'cuda' else None,
        }
        if improved:
            atomic_torch_save(run_dir / 'best.pt', payload)
        atomic_torch_save(run_dir / 'latest.pt', payload)
        atomic_csv(run_dir / 'training_history.csv', frame)
        atomic_json(run_dir / 'run_metadata.json', metadata)

    try:
        print(f'RUN_DIR={run_dir}\nDevice={device}; threads={args.threads}; seed={args.seed}; weight_decay={args.weight_decay:g}; dropout_p={dropout_p:g}', flush=True)
        model, history = ns['train_cheng_model'](
            loaders['train'], loaders['validation'], args.epochs, args.patience, device,
            epoch_callback=checkpoint_epoch,
            weight_decay=args.weight_decay,
            hidden_sizes=hidden_sizes,
            dropout_p=dropout_p,
            optimizer_name=optimizer_name, l1_lambda=l1_lambda,
            train_eval_loader=train_eval_loader,
        )
        metadata.update(status='evaluating', training_seconds=time.perf_counter() - started,
                        epochs_completed=len(history))
        atomic_json(run_dir / 'run_metadata.json', metadata)
        # train_cheng_model restores its best validation checkpoint before returning.
        for split in evaluated_splits:
            print(f'Evaluating best checkpoint on {split}...', flush=True)
            metadata.setdefault('evaluation', {})[split] = export_predictions(ns, model, scaler, data[split], split, run_dir, metadata, device)
            atomic_json(run_dir / 'run_metadata.json', metadata)
        if args.publish_model:
            import shutil
            args.publish_model.parent.mkdir(parents=True, exist_ok=True)
            if args.publish_model.exists():
                raise FileExistsError('Publish destination appeared during training; retained run outputs without overwriting it.')
            temporary = args.publish_model.with_name(args.publish_model.name + '.partial')
            shutil.copy2(run_dir / 'best.pt', temporary)
            temporary.replace(args.publish_model)
            metadata['published_model'] = str(args.publish_model.resolve())
        metadata.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
                        total_seconds=time.perf_counter() - started)
        atomic_json(run_dir / 'run_metadata.json', metadata)
        print(json.dumps(metadata, indent=2), flush=True)
    except BaseException as exc:
        metadata.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.perf_counter()-started)
        atomic_json(run_dir / 'run_metadata.json', metadata)
        raise
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-directory', type=Path, default=PROJECT_ROOT / 'data/simulated/cheng_nn_17d')
    parser.add_argument('--run-directory', type=Path, default=PROJECT_ROOT / 'results/cheng_nn_17d' / datetime.now().strftime('run_%Y%m%d_%H%M%S'))
    parser.add_argument('--publish-model', type=Path)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260929)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Adam coupled L2; use 0 for the unregularized baseline.')
    parser.add_argument('--optimizer', choices=('Adam', 'AdamW'), default='Adam')
    parser.add_argument('--l1-lambda', type=float, default=0.0, help='Coefficient of sum(abs(all trainable parameters)); only applied during training.')
    parser.add_argument('--return-periods', type=float, nargs='+', default=[50., 100.])
    parser.add_argument('--monitor-train-eval', action='store_true', help='Evaluate train RMSE with Dropout off at each epoch for comparable overfit curves.')
    parser.add_argument('--coefficient-only', action='store_true', help='Export five coefficient metrics and GEV support without calculating return levels.')
    parser.add_argument('--monitor-split-rmse', action='store_true', help='Diagnostic train/validation/test RMSE each epoch; requires --evaluate-test; test never selects models.')
    parser.add_argument('--dropout-p', type=float, default=0.0, help='Training-only dropout after each hidden ReLU; output remains linear.')
    parser.add_argument('--evaluate-test', action='store_true', help='Opt in to final test evaluation; omit while tuning.')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[512, 512, 512, 128, 128], help='Hidden ReLU widths; output remains five linear coefficients.')
    parser.add_argument('--no-update-latest', action='store_true', help='Keep the existing single-run diagnostics pointer during comparisons.')
    args = parser.parse_args()
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('Learning rate must be positive and finite.')
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('Weight decay must be nonnegative and finite.')
    if not np.isfinite(args.dropout_p) or not 0.0 <= args.dropout_p < 1.0:
        parser.error('Dropout probability must be finite and in [0, 1).')
    args.run_directory.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.run_directory.with_name(args.run_directory.name + '.log')
    with log_path.open('x', encoding='utf-8') as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            run_training(args)


if __name__ == '__main__':
    main()
