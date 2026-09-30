"""Small notebook integration checks; never run the full simulation or training."""

import ast
import copy
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cheng_nn_evaluation import check_gev_predictions, conditional_return_levels, return_level_recovery_metrics
from cheng_nn_simulation import COEFFICIENT_NAMES, INPUT_FEATURE_NAMES, ParameterRanges, SimulationConfig, build_input_features, simulate_chunk


@pytest.fixture
def notebook_namespace(tmp_path):
    notebook_path = SRC.parent / "notebooks" / "cheng_NN.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}
    for identifier, source in sources.items():
        compile(source, identifier, "exec")
    namespace = {
        "np": np, "pd": pd, "torch": torch, "nn": torch.nn, "copy": copy,
        "time": time, "plt": plt, "display": lambda *args: None, "Dataset": Dataset,
        "COEFFICIENT_NAMES": COEFFICIENT_NAMES, "INPUT_FEATURE_NAMES": INPUT_FEATURE_NAMES,
        "SIMULATION_CONFIG": SimulationConfig(), "PARAMETER_RANGES": ParameterRanges(),
        "LEARNING_RATE": 0.001, "WEIGHT_DECAY": 1e-4, "DEVICE": "cpu", "RETURN_PERIODS": (50.0, 100.0),
        "MODEL_PATH": tmp_path / "model.pt", "TABLE_DIR": tmp_path / "tables",
        "FIGURE_DIR": tmp_path / "figures", "build_input_features": build_input_features,
        "check_gev_predictions": check_gev_predictions,
        "conditional_return_levels": conditional_return_levels,
        "return_level_recovery_metrics": return_level_recovery_metrics,
    }
    for identifier in ("cheng-model", "cheng-dataset", "cheng-training-functions", "cheng-evaluation", "cheng-real-inference"):
        tree = ast.parse(sources[identifier])
        definitions = ast.Module(body=[node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))], type_ignores=[])
        exec(compile(definitions, identifier, "exec"), namespace)
    yield namespace, sources
    plt.close("all")


@pytest.mark.parametrize('weight_decay', [0.0, 1e-4])
def test_one_epoch_uses_adam_and_finite_parameter_mse(notebook_namespace, monkeypatch, weight_decay):
    ns, _ = notebook_namespace
    x, y, *_ = simulate_chunk(np.random.default_rng(41), 8, SimulationConfig(), ParameterRanges())
    scaler = ns["TargetScaler"].fit(y[:6])
    train = DataLoader(ns["ChengSimulationDataset"](x[:6], y[:6], scaler), batch_size=3)
    validation = DataLoader(ns["ChengSimulationDataset"](x[6:], y[6:], scaler), batch_size=2)
    original_adam = torch.optim.Adam
    optimizers = []

    def record_adam(*args, **kwargs):
        optimizer = original_adam(*args, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr(torch.optim, "Adam", record_adam)
    _, history = ns["train_cheng_model"](train, validation, 1, 1, "cpu", weight_decay=weight_decay)
    assert len(optimizers) == 1
    assert optimizers[0].defaults["lr"] == 0.001
    assert optimizers[0].defaults['weight_decay'] == weight_decay
    assert len(history) == 1
    assert np.isfinite(history[["train_loss", "validation_loss"]].to_numpy()).all()


def test_reported_loss_does_not_include_l2_penalty(notebook_namespace):
    ns, _ = notebook_namespace
    model = torch.nn.Linear(17, 5)
    with torch.no_grad():
        model.weight.fill_(0.25)
        model.bias.fill_(0.5)
    inputs, targets = torch.ones(3, 17), torch.zeros(3, 5)
    batches = [(inputs, targets)]
    expected = float(torch.mean((model(inputs) - targets)**2).detach())
    # lr=0 holds predictions fixed while the train path and nonzero decay are exercised.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, weight_decay=0.1)
    assert ns['mean_batch_loss'](model, batches, 'cpu', optimizer) == pytest.approx(expected)
    assert ns['mean_batch_loss'](model, batches, 'cpu') == pytest.approx(expected)


def test_default_architecture_preserves_legacy_weights(notebook_namespace):
    ns, _ = notebook_namespace
    torch.manual_seed(17)
    current = ns['ChengGEVNet']()
    torch.manual_seed(17)
    legacy = torch.nn.Sequential(
        torch.nn.Linear(17, 512), torch.nn.ReLU(),
        torch.nn.Linear(512, 512), torch.nn.ReLU(),
        torch.nn.Linear(512, 512), torch.nn.ReLU(),
        torch.nn.Linear(512, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 5),
    )
    assert sum(p.numel() for p in current.parameters()) == 617349
    for key, tensor in legacy.state_dict().items():
        assert torch.equal(current.state_dict()['network.' + key], tensor)
    small = ns['ChengGEVNet']((128, 128, 64))
    assert small(torch.zeros(2, 17)).shape == (2, 5)
    assert sum(p.numel() for p in small.parameters()) == 27397


@pytest.mark.parametrize('probability', [0.0, 0.1, 0.2, 0.5])
def test_dropout_keeps_initial_weights_and_is_disabled_at_evaluation(notebook_namespace, probability):
    ns, _ = notebook_namespace
    torch.manual_seed(123)
    baseline = ns['ChengGEVNet']()
    torch.manual_seed(123)
    model = ns['ChengGEVNet'](dropout_p=probability)
    for expected, actual in zip(baseline.parameters(), model.parameters()):
        assert torch.equal(expected, actual)
    dropouts = [layer for layer in model.modules() if isinstance(layer, torch.nn.Dropout)]
    assert len(dropouts) == (5 if probability else 0)
    assert all(layer.p == probability for layer in dropouts)
    assert isinstance(model.network[-1], torch.nn.Linear)
    x = torch.randn(16, 17)
    baseline.eval()
    model.eval()
    torch.testing.assert_close(model(x), baseline(x), rtol=0, atol=0)
    model.train()
    if probability:
        assert not torch.equal(model(x), model(x))


@pytest.mark.parametrize('probability', [-0.1, 1.0, float('nan'), float('inf')])
def test_dropout_rejects_invalid_probability(notebook_namespace, probability):
    ns, _ = notebook_namespace
    with pytest.raises(ValueError, match='dropout_p'):
        ns['ChengGEVNet'](dropout_p=probability)


@pytest.mark.parametrize('bad', [(), (0,), (-1,), (2.5,), (True,)])
def test_architecture_rejects_invalid_widths(notebook_namespace, bad):
    ns, _ = notebook_namespace
    with pytest.raises(ValueError, match='hidden_sizes'):
        ns['ChengGEVNet'](bad)


@pytest.mark.parametrize('bad', [-0.1, np.nan, np.inf])
def test_rejects_invalid_regularization_before_training(notebook_namespace, bad):
    ns, _ = notebook_namespace
    with pytest.raises(ValueError, match='weight_decay'):
        ns['train_cheng_model'](None, None, 1, 1, 'cpu', weight_decay=bad)


def test_notebook_writes_validity_and_yearly_rl_tables(notebook_namespace):
    ns, sources = notebook_namespace
    _, _, truth, _, observations = simulate_chunk(np.random.default_rng(9), 4, SimulationConfig(), ParameterRanges())
    ns.update(RUN_EVALUATION=True, true_original=truth, predicted_original=truth.copy(), test_data={"annual_maxima": observations})
    exec(compile(sources["cheng-gev-rl-evaluation"], "cheng-gev-rl-evaluation", "exec"), ns)
    metrics = pd.read_csv(ns["TABLE_DIR"] / "return_level_recovery_by_year.csv")
    assert len(metrics) == 50 * 2 * 2
    assert (metrics["n_used"] == 4).all()
    assert (metrics["RMSE"] == 0).all()
    assert ns["gev_validity"]["gev_valid_all_years"].all()
    assert (ns["FIGURE_DIR"] / "conditional_return_level_rmse.png").exists()


def test_real_inference_checks_support_and_can_return_rl_curves(notebook_namespace):
    ns, _ = notebook_namespace
    _, targets, truth, _, observations = simulate_chunk(np.random.default_rng(7), 1, SimulationConfig(), ParameterRanges())

    class ConstantModel(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

        def forward(self, values):
            return self.values.expand(len(values), -1)

    scaler = ns["TargetScaler"](np.zeros(5), np.ones(5))
    ns["load_cheng_checkpoint"] = lambda *args: (ConstantModel(targets[0]), scaler, {})
    result = ns["estimate_real_sequence"](SimulationConfig().years, observations[0], return_details=True)
    np.testing.assert_allclose(list(result["coefficients"].values()), truth[0], rtol=1e-5, atol=1e-5)
    assert result["return_levels"].shape == (50, 3)
    assert np.isfinite(result["return_levels"].to_numpy()).all()

    ns["load_cheng_checkpoint"] = lambda *args: (ConstantModel([0, 0, 0, 0, -1000]), scaler, {})
    with pytest.raises(ValueError, match="parameter/support checks"):
        ns["estimate_real_sequence"](SimulationConfig().years, observations[0])
