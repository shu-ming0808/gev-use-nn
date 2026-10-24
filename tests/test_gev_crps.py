"""CRPS formula checked against the defining CDF integral, not its own code."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import genextreme

from taiwan_simulation.gev_crps import gev_crps, score_temperature_distribution


@pytest.mark.parametrize('xi', [-.8, -.4, -.1, -1e-7, 0., 1e-7, .1, .4, .8])
@pytest.mark.parametrize('y', [-3., 0., 3.])
def test_closed_form_matches_cdf_integral(xi, y):
    d = genextreme(-xi)
    with np.errstate(over='ignore', under='ignore'):
        expected = quad(lambda x: d.cdf(x)**2, -np.inf, y, epsabs=2e-9, limit=400)[0]
        expected += quad(lambda x: d.sf(x)**2, y, np.inf, epsabs=2e-9, limit=400)[0]
    assert float(gev_crps(y, 0., 1., xi)) == pytest.approx(expected, rel=2e-7, abs=2e-7)


def test_units_broadcast_and_outside_support():
    # negative xi: upper endpoint 2; positive xi: lower endpoint -2.
    y = np.array([4., -4., 0.])
    xi = np.array([-.5, .5, 0.])
    base = gev_crps(y, 0., 1., xi)
    assert np.isfinite(base).all() and (base > 0).all()
    np.testing.assert_allclose(gev_crps(10+3*y, 10., 3., xi), 3*base)
    np.testing.assert_allclose(gev_crps([1000., -1000.], 0., 1., 0.),
        [1000-np.euler_gamma-np.log(2), 1000+np.euler_gamma-np.log(2)])


def test_invalid_predictions_are_reported_without_subset_mean():
    theta = np.array([[0., 0., 0., 0., -.5], [0., 0., 0., 0., 1.]])
    obs = np.full((2, 50), 4.)
    score, overall, yearly = score_temperature_distribution(theta, obs, stage='gp_oof')
    assert np.isfinite(score[0]).all() and np.isnan(score[1]).all()
    row = overall.iloc[0]
    assert row.n_total == 100 and row.n_scored == 50 and row.n_invalid == 50
    assert row.n_outside_support == 50 and row.n_unsupported_shape == 50
    assert np.isnan(row.mean_CRPS_C) and yearly.mean_CRPS_C.isna().all()
    assert np.isnan(gev_crps(1., 0., -1., 0.))


def test_oracle_not_zero_and_full_50_years_required():
    theta = np.zeros((2, 5))
    score, summary, yearly = score_temperature_distribution(theta, np.zeros((2, 50)), stage='truth')
    assert (score > 0).all() and summary.iloc[0].mean_CRPS_C > 0
    assert len(yearly) == 50 and yearly.year.iloc[0] == 1976 and yearly.year.iloc[-1] == 2025
    with pytest.raises(ValueError):
        score_temperature_distribution(theta, np.zeros((2, 10)), stage='truth')
