"""Small durable-training integration check; never runs the 100k study."""
from argparse import Namespace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC)) if str(SRC) not in sys.path else None
import train_cheng_nn as runner
from cheng_nn_simulation import generate_all_splits


@pytest.mark.parametrize('evaluate_test', [False, True])
@pytest.mark.parametrize('hidden_sizes', [(512, 512, 512, 128, 128), (128, 128, 64)])
@pytest.mark.parametrize('dropout_p', [0.0, 0.2])
def test_checkpoint_and_requested_split_exports(tmp_path, monkeypatch, evaluate_test, hidden_sizes, dropout_p):
    root = tmp_path / 'project'
    data_dir = root / 'data'
    generate_all_splits(data_dir, n_train=8, n_validation=4, n_test=4)
    ns = runner.load_notebook_components()
    ns.update(N_TRAIN=8, N_VALIDATION=4, N_TEST=4, DATA_DIR=data_dir)
    original_loader = ns['load_split']
    loaded_splits = []
    def tracked_loader(split):
        loaded_splits.append(split)
        return original_loader(split)
    ns['load_split'] = tracked_loader
    monkeypatch.setattr(runner, 'load_notebook_components', lambda: ns)
    # Only the run pointer uses PROJECT_ROOT; provide the source snapshot required by provenance.
    (root / 'notebooks').mkdir(parents=True)
    (root / 'notebooks/cheng_NN.ipynb').write_bytes((SRC.parent / 'notebooks/cheng_NN.ipynb').read_bytes())
    monkeypatch.setattr(runner, 'PROJECT_ROOT', root)
    args = Namespace(threads=1, epochs=1, patience=1, batch_size=4, seed=7,
                     device='cpu', data_directory=data_dir,
                     run_directory=root / 'results/run_test', publish_model=root / 'model.pt',
                     learning_rate=0.001, weight_decay=1e-4, evaluate_test=evaluate_test,
                     hidden_sizes=hidden_sizes, no_update_latest=True, dropout_p=dropout_p)
    run_dir = runner.run_training(args)
    metadata = json.loads((run_dir / 'run_metadata.json').read_text(encoding='utf-8'))
    assert metadata['status'] == 'completed'
    assert metadata['best_epoch'] == 1
    assert not metadata['test_used_for_training_or_selection']
    assert metadata['weight_decay'] == 1e-4
    assert metadata['dropout_p'] == dropout_p
    assert metadata['architecture'] == [17, *hidden_sizes, 5]
    assert not (root / 'results/cheng_nn_17d/latest_run.json').exists()
    assert metadata['test_evaluation_requested'] == evaluate_test
    assert not metadata['reported_loss_includes_regularization']
    assert (run_dir / 'latest.pt').is_file() and args.publish_model.is_file()
    assert args.publish_model.read_bytes() == (run_dir / 'best.pt').read_bytes()
    checkpoint = torch.load(run_dir / 'latest.pt', weights_only=False)
    assert checkpoint['epoch'] == 1
    assert checkpoint['weight_decay'] == 1e-4
    assert checkpoint['hidden_sizes'] == list(hidden_sizes)
    restored, _, _ = ns['load_cheng_checkpoint'](args.publish_model, 'cpu')
    assert restored.hidden_sizes == hidden_sizes
    assert restored.dropout_p == dropout_p
    assert sum(p.numel() for p in restored.parameters()) == metadata['trainable_parameters']
    assert checkpoint['optimizer_state']['param_groups'][0]['weight_decay'] == 1e-4
    assert 'optimizer_state' in checkpoint and 'shuffle_rng_state' in checkpoint
    history = pd.read_csv(run_dir / 'training_history.csv')
    assert len(history) == 1 and np.isfinite(history['validation_loss']).all()
    expected_splits = [('train',8),('validation',4)] + ([('test',4)] if evaluate_test else [])
    assert loaded_splits == [name for name, _ in expected_splits]
    assert (run_dir / 'test').exists() == evaluate_test
    for split, size in expected_splits:
        predictions = np.load(run_dir / split / 'predictions_original.npy')
        assert predictions.shape == (size,5)
        marker = json.loads((run_dir / split / 'prediction_metadata.json').read_text(encoding='utf-8'))
        assert marker['best_epoch'] == metadata['best_epoch']
        assert marker['dataset_metadata_sha256'] == metadata['dataset_metadata_sha256'][split]
        assert marker['checkpoint_sha256'] == runner.file_sha256(run_dir / 'best.pt')
        assert (run_dir / split / 'return_level_by_year.csv').is_file()


def test_epoch_monitor_does_not_change_training_or_checkpoint_selection(tmp_path, monkeypatch):
    data_dir = tmp_path / 'data'
    generate_all_splits(data_dir, n_train=8, n_validation=4, n_test=4)
    results = []
    for monitor in (False, True):
        args = Namespace(threads=1, epochs=3, patience=3, batch_size=4, seed=9,
                         device='cpu', data_directory=data_dir,
                         run_directory=tmp_path / f'run_{monitor}', publish_model=None,
                         learning_rate=0.001, weight_decay=1e-4, evaluate_test=True,
                         hidden_sizes=(8,), no_update_latest=True, dropout_p=0.2,
                         return_periods=(20., 100.), monitor_split_rmse=monitor)
        path = runner.run_training(args)
        results.append((path, torch.load(path / 'best.pt', weights_only=False)))
    baseline, measured = (item[1] for item in results)
    assert baseline['epoch'] == measured['epoch']
    assert baseline['best_validation_loss'] == measured['best_validation_loss']
    for key, value in baseline['model_state'].items():
        torch.testing.assert_close(value, measured['model_state'][key], rtol=0, atol=0)
    assert torch.equal(baseline['torch_rng_state'], measured['torch_rng_state'])
    assert torch.equal(baseline['shuffle_rng_state'], measured['shuffle_rng_state'])
    baseline_history = pd.read_csv(results[0][0] / 'training_history.csv')
    monitored_history = pd.read_csv(results[1][0] / 'training_history.csv')
    for column in ('train_loss', 'validation_loss', 'learning_rate'):
        np.testing.assert_array_equal(baseline_history[column], monitored_history[column])
    from plot_cheng_nn_rmse_curves import read_run
    curves, meta = read_run(results[1][0])
    assert len(curves) == 3 * 3 * 8
    best = curves.loc[curves.epoch.eq(meta['best_epoch'])]
    for split in ('train', 'validation', 'test'):
        exported = pd.read_csv(results[1][0] / split / 'coefficient_metrics.csv')
        original = exported.loc[exported.scale.eq('original')].set_index('coefficient')
        values = best.loc[best.split.eq(split)].set_index('outcome').RMSE
        np.testing.assert_allclose(values.loc[original.index], original.RMSE)
        rl = pd.read_csv(results[1][0] / split / 'return_level_overall.csv')
        for period in (20, 100):
            expected = rl.loc[rl.subset.eq('finite_RL') & rl.return_period.eq(period), 'RMSE'].item()
            assert values[f'RL{period}'] == pytest.approx(expected)
    # A missing test point must not be interpolated or silently plotted.
    curves.iloc[1:].to_csv(results[1][0] / 'epoch_rmse.csv', index=False)
    with pytest.raises(ValueError, match='Every epoch'):
        read_run(results[1][0])


def test_coefficient_only_keeps_training_and_never_reads_test_or_computes_rl(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    generate_all_splits(data, n_train=8, n_validation=4, n_test=4)
    checkpoints = []
    for coefficient_only in (False, True):
        ns = runner.load_notebook_components()
        if coefficient_only:
            def forbidden_rl(*args, **kwargs):
                raise AssertionError('Coefficient-only run must not calculate RL.')
            ns['conditional_return_levels'] = forbidden_rl
            original_loader = ns['load_split']
            def no_test(split):
                assert split != 'test'
                return original_loader(split)
            ns['load_split'] = no_test
        monkeypatch.setattr(runner, 'load_notebook_components', lambda: ns)
        args = Namespace(threads=1, epochs=2, patience=2, batch_size=4, seed=9,
                         device='cpu', data_directory=data,
                         run_directory=tmp_path / f'coeff_{coefficient_only}', publish_model=None,
                         learning_rate=0.0003, weight_decay=1e-5, evaluate_test=False,
                         hidden_sizes=(8,), no_update_latest=True, dropout_p=0.,
                         coefficient_only=coefficient_only, monitor_train_eval=True)
        path = runner.run_training(args)
        checkpoints.append(torch.load(path / 'best.pt', weights_only=False))
        assert not (path / 'test').exists()
        assert (path / 'validation/return_level_overall.csv').exists() != coefficient_only
        if coefficient_only:
            meta = json.loads((path / 'run_metadata.json').read_text(encoding='utf-8'))
            assert meta['return_periods'] == [] and meta['coefficient_only']
            assert meta['initial_learning_rate'] == 0.0003
            assert (path / 'validation/gev_validity.csv').exists()
        # Undo just this replacement before loading fresh notebook definitions.
        monkeypatch.undo()
    assert checkpoints[0]['best_validation_loss'] == checkpoints[1]['best_validation_loss']
    for key, value in checkpoints[0]['model_state'].items():
        torch.testing.assert_close(value, checkpoints[1]['model_state'][key], rtol=0, atol=0)
