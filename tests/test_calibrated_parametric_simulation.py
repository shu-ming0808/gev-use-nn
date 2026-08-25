from pathlib import Path
import sys

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from calibrated_parametric_simulation import (  # noqa: E402
    CalibratedSimulationConfig,
    annual_return_level_from_monthly_gev,
    generate_true_parameter_surfaces,
    load_scenario_inputs,
    nn_recovery_metrics,
    simulate_monthly_maxima,
)


def test_load_scenario_inputs_decodes_final_predictors(tmp_path):
    grid = pd.DataFrame(
        {
            "station": ["a", "b"],
            "lon": [120.0, 120.1],
            "lat": [23.0, 23.1],
            "x_km": [100.0, 110.0],
            "y_km": [2500.0, 2510.0],
            "mu_hat": [30.0, 31.0],
            "log_sigma_hat": [0.1, 0.2],
            "xi_hat": [0.05, 0.06],
            "elevation_m": [10.0, 20.0],
            "tpi_m": [1.0, 2.0],
            "wind": [3.0, 4.0],
            "cloud": [0.4, 0.5],
        }
    )
    selected = pd.DataFrame(
        {
            "target": ["mu", "log_sigma", "xi"],
            "predictors": [
                "elevation_m+tpi_m+wind",
                "elevation_m+cloud",
                "elevation_m",
            ],
            "kernel": ["RBF", "Matern", "Matern"],
            "nu": [np.nan, 1.5, 0.5],
        }
    )
    grid_path = tmp_path / "grid.csv"
    selected_path = tmp_path / "selected.csv"
    grid.to_csv(grid_path, index=False)
    selected.to_csv(selected_path, index=False)

    loaded, specifications = load_scenario_inputs(grid_path, selected_path)

    assert len(loaded) == 2
    assert specifications["mu"]["predictors"] == [
        "elevation_m",
        "tpi_m",
        "wind",
    ]
    assert specifications["log_sigma"]["nu"] == 1.5


def test_generated_truth_and_monthly_maxima_are_finite():
    n = 6
    grid = pd.DataFrame(
        {
            "station": [f"g{i}" for i in range(n)],
            "x_km": np.arange(n, dtype=float),
            "y_km": np.arange(n, dtype=float),
        }
    )
    generator = {
        "mu": {"mean": np.full(n, 30.0), "cholesky": np.eye(n) * 0.1},
        "log_sigma": {
            "mean": np.full(n, np.log(1.5)),
            "cholesky": np.eye(n) * 0.01,
        },
        "xi": {"mean": np.full(n, 0.05), "cholesky": np.eye(n) * 0.01},
    }
    config = CalibratedSimulationConfig(n_years=5, xi_lower=-0.2, xi_upper=0.2)
    rng = np.random.default_rng(7)

    truth, clipped = generate_true_parameter_surfaces(grid, generator, config, rng)
    monthly = simulate_monthly_maxima(truth, config, rng)

    assert clipped == 0
    assert monthly.shape == (n, 60)
    assert np.isfinite(monthly).all()
    assert np.isfinite(truth[["RL50_true", "RL100_true"]]).all().all()


def test_annual_return_level_uses_twelve_monthly_blocks() -> None:
    annual_rl = annual_return_level_from_monthly_gev(
        np.array([0.0]), np.array([0.0]), np.array([0.0]), 50, 12
    )
    monthly_50_block_rl = annual_return_level_from_monthly_gev(
        np.array([0.0]), np.array([0.0]), np.array([0.0]), 50, 1
    )
    assert annual_rl[0] > monthly_50_block_rl[0]


def test_nn_recovery_metrics_uses_known_truth_columns():
    frame = pd.DataFrame()
    for outcome in ("mu", "log_sigma", "xi", "RL50", "RL100"):
        frame[f"{outcome}_true"] = [1.0, 2.0, 3.0]
        frame[f"{outcome}_hat"] = [1.0, 2.0, 3.0]

    metrics = nn_recovery_metrics(frame)

    assert set(metrics["outcome"]) == {"mu", "log_sigma", "xi", "RL50", "RL100"}
    assert np.allclose(metrics["RMSE"], 0.0)
