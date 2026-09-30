"""Small deterministic checks; no full training or simulation study."""
from argparse import Namespace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC)) if str(SRC) not in sys.path else None
import compare_cheng_nn_regularization as comparison
import train_cheng_nn as runner
from cheng_nn_simulation import generate_all_splits


def test_improvement_sign_and_paired_seeds():
    assert comparison.improvement_percent(2., 1.5) == pytest.approx(25.)
    assert comparison.improvement_percent(2., 2.2) == pytest.approx(-10.)
    rows = []
    for case in comparison.CASES:
        for seed, value in [(1, 2.), (2, 4.)]:
            for outcome in comparison.OUTCOMES:
                rmse = value if case == 'baseline' else value*.9
                rows.append(dict(case=case, seed=seed, outcome=outcome,
                                 train_RMSE=rmse*.8, validation_RMSE=rmse))
    paired, summary = comparison.aggregate_metrics(rows, [1, 2])
    assert np.allclose(summary.loc[~summary.case.eq('baseline'), 'improvement_percent_mean'], 10.)
    assert np.allclose(paired.gap_percent, 25.)
    with pytest.raises(ValueError, match='paired'):
        comparison.aggregate_metrics(rows[:-1], [1, 2])
    with pytest.raises(ValueError, match='positive'):
        comparison.improvement_percent(0., 1.)


def test_l1_changes_weights_but_not_reported_data_loss():
    ns = runner.load_notebook_components()
    model = torch.nn.Linear(17, 5)
    with torch.no_grad():
        model.weight.fill_(.1)
        model.bias.fill_(.2)
    x = torch.zeros(3, 17)
    y = model(x).detach().clone()
    before = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    loss = ns['mean_batch_loss'](model, [(x, y)], 'cpu', optimizer, l1_lambda=.01)
    assert loss == 0.
    assert torch.all(model.weight < before)
    ns['mean_batch_loss'](model, [(x, y)], 'cpu', l1_lambda=1.)
    assert not model.training


@pytest.mark.parametrize('optimizer,l1', [('Adam', 1e-5), ('AdamW', 0.)])
def test_extended_runner_exports_rl20_and_eval_curves(tmp_path, monkeypatch, optimizer, l1):
    data = tmp_path / 'data'
    generate_all_splits(data, n_train=8, n_validation=4, n_test=4)
    ns = runner.load_notebook_components()
    ns.update(DATA_DIR=data, N_TRAIN=8, N_VALIDATION=4, N_TEST=4)
    loaded = []
    load = ns['load_split']
    def tracked(split):
        loaded.append(split)
        return load(split)
    ns['load_split'] = tracked
    monkeypatch.setattr(runner, 'load_notebook_components', lambda: ns)
    args = Namespace(threads=1, epochs=2, patience=2, batch_size=4, seed=9,
                     device='cpu', data_directory=data, run_directory=tmp_path / optimizer,
                     publish_model=None, learning_rate=.001, weight_decay=1e-4,
                     l1_lambda=l1, optimizer=optimizer, monitor_train_eval=True,
                     return_periods=[20., 100.], evaluate_test=False,
                     hidden_sizes=[8, 8], dropout_p=.1, no_update_latest=True)
    path = runner.run_training(args)
    meta = json.loads((path / 'run_metadata.json').read_text(encoding='utf-8'))
    assert meta['optimizer'] == optimizer and meta['l1_lambda'] == l1
    assert loaded == ['train', 'validation']
    for split in loaded:
        values, scaled = comparison.metrics_for_split(path, split)
        assert set(values) == set(comparison.OUTCOMES)
        assert np.isfinite(scaled)
    h = comparison.pd.read_csv(path / 'training_history.csv')
    assert np.isfinite(h[['train_RMSE_eval', 'validation_RMSE_eval']]).all().all()
    checkpoint = torch.load(path / 'best.pt', weights_only=False)
    assert checkpoint['optimizer'] == optimizer
    assert checkpoint['return_periods'] == [20., 100.]


def test_eval_loader_does_not_advance_training_rng():
    from torch.utils.data import DataLoader, TensorDataset
    ns = runner.load_notebook_components()
    model = ns['ChengGEVNet'](hidden_sizes=(8,), dropout_p=.5)
    dataset = TensorDataset(torch.zeros(6, 17), torch.zeros(6, 5))
    loader = DataLoader(dataset, batch_size=3, generator=torch.Generator().manual_seed(10))
    state = torch.get_rng_state().clone()
    ns['mean_batch_loss'](model, loader, 'cpu')
    assert torch.equal(state, torch.get_rng_state())
    assert not model.training
