from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import genextreme

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cheng_nn_mle import (
    nll_and_gradient, support_constraint, support_jacobian,
    fit_time_varying_gev, original_coefficients, standardized_coefficients,
    fitting_bounds,
)
from cheng_nn_simulation import ParameterRanges, SimulationConfig, simulate_chunk


@pytest.mark.parametrize("xi", [-0.3, -1e-8, 0.0, 1e-8, 0.3])
def test_likelihood_matches_scipy_and_gradient_matches_finite_differences(xi):
    t = np.linspace(-2.45, 2.45, 50)
    y = np.linspace(-0.5, 1.8, 50)
    p = np.array([0.1, 0.13, -0.2, 0.04, xi])
    nll, gradient = nll_and_gradient(p, y, t)
    mu = p[0] + p[1] * np.exp(p[2]) * t
    expected = -genextreme.logpdf(y, c=-xi, loc=mu, scale=np.exp(p[2] + p[3]*t)).sum()
    np.testing.assert_allclose(nll, expected, rtol=1e-11)
    numerical = approx_derivative(lambda x: np.array([nll_and_gradient(x, y, t)[0]]), p).ravel()
    np.testing.assert_allclose(gradient, numerical, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(support_jacobian(p, y, t),
                               approx_derivative(lambda x: support_constraint(x, y, t), p),
                               rtol=1e-5, atol=1e-7)


def test_standardization_roundtrip():
    coefficients = np.array([[30, .3, 1, .1, -.2], [-10, -.5, -1, -.1, .2]])
    transform = np.array([[31, 2], [-9, .5]])
    np.testing.assert_allclose(original_coefficients(standardized_coefficients(coefficients, transform), transform), coefficients)


@pytest.mark.parametrize("domain", ["training_domain", "wide_domain"])
def test_multistart_fit_is_feasible_and_uses_all_five_coefficients(domain):
    config = SimulationConfig()
    _, _, _, _, maxima = simulate_chunk(np.random.default_rng(981), 1, config, ParameterRanges())
    result = fit_time_varying_gev(maxima[0], config.centered_time, domain=domain)
    assert result["success"] and result["successful_starts"] >= 2
    assert result["min_support_margin"] > 0
    coef = result["coefficients"]
    assert len(coef) == 5 and np.isfinite(coef).all()
    expected = -genextreme.logpdf(maxima[0], -coef[4],
        loc=coef[0]+coef[1]*config.centered_time,
        scale=np.exp(coef[2]+coef[3]*config.centered_time)).sum()
    np.testing.assert_allclose(result["nll"], expected, atol=1e-7)


def test_training_bounds_are_exact_transformation_of_sampling_domain():
    ranges = ParameterRanges()
    bounds = fitting_bounds(12, 3, ranges, "training_domain")
    np.testing.assert_allclose(bounds[0]*3+12, ranges.mu0)
    np.testing.assert_allclose(bounds[2]+np.log(3), ranges.eta0)
    np.testing.assert_allclose(bounds[1], ranges.beta_mu_over_sigma0)


def test_invalid_input_is_not_silently_fitted():
    with pytest.raises(ValueError, match="IQR"):
        fit_time_varying_gev(np.ones(50), np.arange(50))


def test_unsuccessful_optimization_is_not_reported_as_zero_error():
    config = SimulationConfig()
    _, _, _, _, maxima = simulate_chunk(np.random.default_rng(983), 1, config, ParameterRanges())
    result = fit_time_varying_gev(maxima[0], config.centered_time, maxiter=1)
    assert not result['success']
    assert np.isnan(result['coefficients']).all()


def test_paired_bootstrap_preserves_sequences_and_sign():
    from compare_cheng_nn_mle import paired_rmse_interval
    low, high = paired_rmse_interval(np.full(50, 4.0), np.full(50, 1.0), repeats=200)
    assert low == high == 1.0
    with pytest.raises(ValueError, match='finite'):
        paired_rmse_interval([float('nan')], [1.0])
