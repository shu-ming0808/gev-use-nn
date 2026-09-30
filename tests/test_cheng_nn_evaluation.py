"""Numerical and support regressions for the time-varying GEV estimator."""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import genextreme

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cheng_nn_evaluation import (
    check_gev_predictions,
    conditional_return_levels,
    gev_parameter_curves,
    return_level_recovery_metrics,
)
from cheng_nn_simulation import ParameterRanges, SimulationConfig, simulate_chunk


def test_support_allows_all_shape_signs_but_rejects_endpoints():
    coefficients = np.array([[0, 0, 0, 0, -0.2], [0, 0, 0, 0, 0.2], [0, 0, 0, 0, 0.0]])
    observations = np.array([[4.9, 5.0, 5.1], [-5.1, -5.0, -4.9], [-100.0, 0.0, 100.0]])
    result = check_gev_predictions(coefficients, observations, [1990, 2000, 2010], reference_year=2000)
    assert result["n_support_violations"].tolist() == [2, 2, 0]
    assert result["gev_valid_all_years"].tolist() == [False, False, True]


def test_support_uses_the_parameter_for_each_observed_year():
    coefficients = np.array([[0, 1, 0, 0, -0.2], [0, 0, 0, 0, 0.8]])
    result = check_gev_predictions(coefficients, [[3.9, 4.9, 5.9], [0, 0, 0]], [1990, 2000, 2010], reference_year=2000)
    assert result["gev_valid_all_years"].all()
    assert result["xi_outside_training_range"].tolist() == [False, True]


def test_near_boundary_is_a_warning_not_a_violation():
    result = check_gev_predictions(np.array([[0, 0, 0, 0, -0.2]]), [[5.0 - 1e-7]], [2000], reference_year=2000)
    assert result.loc[0, "n_near_boundary"] == 1
    assert result.loc[0, "gev_valid_all_years"]


@pytest.mark.parametrize("bad_eta", [np.nan, np.inf, 1000.0, -1000.0])
def test_invalid_scale_or_coefficients_are_flagged(bad_eta):
    coefficients = np.array([[0, 0, bad_eta, 0, 0]])
    diagnostics = check_gev_predictions(coefficients, [[0]], [2000], reference_year=2000)
    assert not diagnostics.loc[0, "parameters_valid_all_years"]
    assert not diagnostics.loc[0, "gev_valid_all_years"]
    assert np.isnan(conditional_return_levels(coefficients, [2000], reference_year=2000)).all()


def test_return_levels_match_scipy_for_negative_zero_and_positive_shape():
    shape = np.array([-0.4, -1e-12, 0, 1e-12, 0.4])
    coefficients = np.column_stack((np.full(5, 30), np.full(5, 0.3), np.full(5, np.log(2)), np.full(5, 0.1), shape))
    years, periods = np.array([1976, 2000, 2025]), np.array([50, 100])
    actual = conditional_return_levels(coefficients, years, periods, reference_year=2000.5)
    mu, sigma, xi, _ = gev_parameter_curves(coefficients, years, reference_year=2000.5)
    expected = genextreme.ppf(1 - 1 / periods, -xi[:, :, None], loc=mu[:, :, None], scale=sigma[:, :, None])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert np.all(actual[:, :, 1] > actual[:, :, 0])


def test_saved_simulation_truth_passes_support_and_rl_checks():
    config = SimulationConfig()
    _, _, coefficients, _, observations = simulate_chunk(np.random.default_rng(2026), 128, config, ParameterRanges())
    validity = check_gev_predictions(coefficients, observations, config.years, reference_year=config.reference_year)
    assert validity["gev_valid_all_years"].all()
    levels = conditional_return_levels(coefficients, config.years, reference_year=config.reference_year)
    assert np.isfinite(levels).all()


def test_rl_metrics_count_numerical_and_support_failures_separately():
    truth = np.zeros((3, 2, 2))
    predicted = np.array([np.full((2, 2), 1.0), np.full((2, 2), 10.0), np.full((2, 2), np.nan)])
    yearly, pooled = return_level_recovery_metrics(predicted, truth, [2000, 2025], [50, 100], [True, False, False])
    numeric = yearly.loc[yearly["subset"].eq("finite_RL")]
    valid = yearly.loc[yearly["subset"].eq("valid_GEV")]
    assert (numeric["n_used"] == 2).all()
    assert (numeric["n_excluded"] == 1).all()
    np.testing.assert_allclose(numeric["RMSE"], np.sqrt(50.5))
    assert (valid["n_used"] == 1).all()
    assert (valid["n_excluded"] == 2).all()
    np.testing.assert_allclose(valid["RMSE"], 1.0)
    assert (pooled["n_total"] == 6).all()


def test_zero_valid_predictions_report_nan_not_zero_error():
    yearly, _ = return_level_recovery_metrics(np.zeros((1, 1, 2)), np.zeros((1, 1, 2)), [2000], [50, 100], [False])
    empty = yearly.loc[yearly["subset"].eq("valid_GEV")]
    assert (empty["n_used"] == 0).all()
    assert empty["RMSE"].isna().all()


@pytest.mark.parametrize("periods", [[1], [0], [np.inf], []])
def test_invalid_return_periods_fail_clearly(periods):
    with pytest.raises(ValueError, match="Return periods"):
        conditional_return_levels(np.zeros((1, 5)), [2000], periods, reference_year=2000)
