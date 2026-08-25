import numpy as np

from return_level_sensitivity import reduced_variate, return_level_derivatives


def _return_level(mu: float, sigma: float, xi: float, period: float) -> float:
    y_t = float(reduced_variate(np.array([period]))[0])
    if abs(xi) < 1e-10:
        return mu + sigma * y_t
    return mu + sigma * np.expm1(xi * y_t) / xi


def test_analytic_derivatives_match_finite_differences() -> None:
    period = 100.0
    mu = 0.0
    sigma = 1.0
    xi = 0.1
    step = 1e-6
    row = return_level_derivatives(np.array([period]), sigma=sigma, xi=xi).iloc[0]

    numerical_mu = (
        _return_level(mu + step, sigma, xi, period)
        - _return_level(mu - step, sigma, xi, period)
    ) / (2.0 * step)
    numerical_sigma = (
        _return_level(mu, sigma + step, xi, period)
        - _return_level(mu, sigma - step, xi, period)
    ) / (2.0 * step)
    numerical_xi = (
        _return_level(mu, sigma, xi + step, period)
        - _return_level(mu, sigma, xi - step, period)
    ) / (2.0 * step)

    assert np.isclose(row["dz_dmu"], numerical_mu, rtol=1e-7)
    assert np.isclose(row["dz_dsigma"], numerical_sigma, rtol=1e-7)
    assert np.isclose(row["dz_dxi"], numerical_xi, rtol=1e-6)


def test_gumbel_limits_are_finite_and_correct() -> None:
    periods = np.array([2.0, 50.0, 1000.0])
    sigma = 1.7
    result = return_level_derivatives(periods, sigma=sigma, xi=0.0)
    y_t = reduced_variate(periods)

    assert np.allclose(result["dz_dmu"], 1.0)
    assert np.allclose(result["dz_dsigma"], y_t)
    assert np.allclose(result["dz_dxi"], 0.5 * sigma * y_t**2)
