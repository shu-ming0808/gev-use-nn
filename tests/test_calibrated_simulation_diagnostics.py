from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from calibrated_simulation_diagnostics import (
    plot_monthly_maxima_examples,
    nearest_neighbour_roughness,
    normalized_variograms,
)


def _surfaces() -> tuple[pd.DataFrame, pd.DataFrame]:
    x, y = np.meshgrid(np.arange(4.0), np.arange(4.0))
    base = pd.DataFrame(
        {
            "station": np.arange(x.size),
            "x_km": x.ravel(),
            "y_km": y.ravel(),
        }
    )
    real = base.assign(
        mu_hat=(x + y).ravel(),
        log_sigma_hat=(x - y).ravel(),
        xi_hat=np.sin(x).ravel(),
    )
    simulated = base.assign(
        mu_true=(2.0 * (x + y)).ravel(),
        log_sigma_true=(2.0 * (x - y)).ravel(),
        xi_true=(2.0 * np.sin(x)).ravel(),
    )
    return real, simulated


def test_normalized_roughness_is_scale_invariant() -> None:
    real, simulated = _surfaces()
    result = nearest_neighbour_roughness(real, simulated)
    assert np.allclose(result["roughness_ratio_simulated_to_real"], 1.0)


def test_normalized_variogram_contains_both_sources() -> None:
    real, simulated = _surfaces()
    result = normalized_variograms(real, simulated, n_lags=4, maxlag_fraction=1.0)
    assert set(result["parameter"]) == {"mu", "log_sigma", "xi"}
    assert set(result["source"]) == {
        "Real NN-derived",
        "Calibrated simulated truth",
    }
    assert result["normalized_semivariance"].notna().all()


def test_monthly_maxima_examples_use_requested_months() -> None:
    real, _ = _surfaces()
    monthly = pd.DataFrame(
        {
            "station": real["station"],
            "monthly_max_1980_01": np.linspace(25.0, 35.0, len(real)),
            "monthly_max_2002_07": np.linspace(26.0, 36.0, len(real)),
            "monthly_max_2024_12": np.linspace(27.0, 37.0, len(real)),
        }
    )
    figure = plot_monthly_maxima_examples(real, monthly)
    titles = {axis.get_title() for axis in figure.axes if axis.get_title()}
    assert titles == {
        "Simulated 1980-01",
        "Simulated 2002-07",
        "Simulated 2024-12",
    }
