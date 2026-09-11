from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from calibrated_simulation_spatial_cv import (
    build_return_level_recovery,
    summarize_parameter_recovery,
)
import calibrated_simulation_spatial_cv as spatial_cv


def _predictions() -> pd.DataFrame:
    rows = []
    truth = {
        "mu": [30.0, 31.0],
        "log_sigma": [0.0, 0.1],
        "xi": [0.1, 0.1],
    }
    for target, values in truth.items():
        for row_index, value in enumerate(values):
            rows.append(
                {
                    "row_index": row_index,
                    "station": f"g{row_index}",
                    "outer_fold": row_index,
                    "target": target,
                    "true_value": value,
                    "nn_value": value + 0.2,
                    "oof_prediction": value + 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_parameter_recovery_uses_simulated_truth() -> None:
    metrics = summarize_parameter_recovery(_predictions())
    nn = metrics.loc[metrics["estimator"].eq("Frozen NN"), "RMSE"]
    oof = metrics.loc[metrics["estimator"].eq("Nested OOF GP"), "RMSE"]
    assert np.allclose(nn, 0.2)
    assert np.allclose(oof, 0.1)


def test_return_level_recovery_builds_both_periods() -> None:
    predictions, metrics = build_return_level_recovery(_predictions())
    assert set(predictions["return_period"]) == {50, 100}
    assert set(metrics["estimator"]) == {"Frozen NN", "Nested OOF GP"}
    assert np.isfinite(metrics["RMSE"]).all()


def test_nested_spatial_cv_closes_fold_figures(monkeypatch) -> None:
    data = pd.DataFrame(
        {
            "station": [f"g{i}" for i in range(8)],
            "lon": np.linspace(120.0, 121.0, 8),
            "lat": np.linspace(23.0, 24.0, 8),
            "x_km": np.arange(8, dtype=float),
            "y_km": np.arange(8, dtype=float),
            "mu_hat": np.linspace(30.0, 31.0, 8),
            "log_sigma_hat": np.linspace(0.0, 0.1, 8),
            "xi_hat": np.linspace(0.05, 0.10, 8),
            "mu_true": np.linspace(30.0, 31.0, 8),
            "log_sigma_true": np.linspace(0.0, 0.1, 8),
            "xi_true": np.linspace(0.05, 0.10, 8),
        }
    )

    def fake_prepare(frame, n_folds, random_state):
        prepared = frame.copy()
        prepared["spatial_fold"] = np.arange(len(prepared)) % n_folds
        return prepared, plt.figure()

    def fake_selection(frame, target, **kwargs):
        path = pd.DataFrame(
            [{"selected_groups": "none", "predictors": "", "RMSE": 0.0}]
        )
        selected = {
            "predictor_names": [],
            "kernel": "RBF",
            "nu": None,
        }
        return pd.DataFrame(), path, selected

    monkeypatch.setattr(spatial_cv, "prepare_spatial_folds", fake_prepare)
    monkeypatch.setattr(
        spatial_cv,
        "_buffered_training_indices",
        lambda data, candidates, test_indices, buffer_km: candidates,
    )
    monkeypatch.setattr(spatial_cv, "sample_indices", lambda indices, *args: indices)
    monkeypatch.setattr(spatial_cv, "spatial_forward_selection", fake_selection)
    monkeypatch.setattr(
        spatial_cv,
        "_fit_outer_model",
        lambda train, test, target, *args: test[spatial_cv.TARGETS[target]].to_numpy(),
    )

    before = set(plt.get_fignums())
    try:
        spatial_cv.run_nested_buffered_spatial_cv(
            data,
            outer_folds=2,
            inner_folds=2,
            max_train=8,
            min_train=1,
            max_steps=1,
            n_jobs=1,
        )
        assert set(plt.get_fignums()) == before
    finally:
        plt.close("all")
