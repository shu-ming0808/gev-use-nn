"""Utilities for a linked monthly-maximum versus annual-maximum experiment.

The monthly block maxima are simulated first.  Each annual maximum is then the
maximum of the 12 monthly maxima from the same simulated year.  Consequently,
the two samples are linked rather than being two unrelated GEV samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import genextreme as gev


XI_ZERO_TOL = 1e-8


def gev_max_stable_params(mu, sigma, xi, n_blocks: int):
    """Return GEV parameters for the maximum of ``n_blocks`` iid GEV values.

    The input and output use the EVT shape convention ``xi``.  SciPy's
    ``genextreme`` convention is ``c = -xi``.
    """

    if int(n_blocks) < 1:
        raise ValueError("n_blocks must be a positive integer")

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    xi = np.asarray(xi, dtype=float)
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive")

    n_blocks = int(n_blocks)
    near_zero = np.abs(xi) < XI_ZERO_TOL
    n_to_xi = np.exp(xi * np.log(n_blocks))

    out_sigma = sigma * n_to_xi
    out_mu = np.where(
        near_zero,
        mu + sigma * np.log(n_blocks),
        mu + sigma * np.expm1(xi * np.log(n_blocks)) / xi,
    )
    return out_mu, out_sigma, xi.copy()


def simulate_linked_block_maxima(
    monthly_params: pd.DataFrame,
    years,
    rng: np.random.Generator,
    months_per_year: int = 12,
    station_col: str = "station",
):
    """Simulate monthly maxima and derive annual maxima from the same values.

    ``monthly_params`` must contain ``station``, ``true_mu``, ``true_sigma`` and
    ``true_xi``.  The experiment assumes identically distributed monthly block
    maxima within a year; this is a controlled estimator experiment and does
    not model the seasonal cycle of real temperature.
    """

    required = {station_col, "true_mu", "true_sigma", "true_xi"}
    missing = required.difference(monthly_params.columns)
    if missing:
        raise ValueError(f"monthly_params is missing columns: {sorted(missing)}")

    years = np.asarray(list(years))
    months_per_year = int(months_per_year)
    if len(years) == 0:
        raise ValueError("years cannot be empty")
    if months_per_year < 1:
        raise ValueError("months_per_year must be positive")

    block_index = pd.DataFrame(
        {
            "year": np.repeat(years, months_per_year),
            "month": np.tile(
                np.arange(1, months_per_year + 1, dtype=int),
                len(years),
            ),
        }
    )

    simulated_columns = {}
    for row in monthly_params.itertuples(index=False):
        station = getattr(row, station_col)
        simulated_columns[station] = gev.rvs(
            c=-float(row.true_xi),
            loc=float(row.true_mu),
            scale=float(row.true_sigma),
            size=len(block_index),
            random_state=rng,
        )

    monthly = pd.concat(
        [block_index, pd.DataFrame(simulated_columns)],
        axis=1,
    )
    annual = monthly.drop(columns="month").groupby("year", as_index=False).max()
    return monthly, annual


def derive_annual_parameter_frame(
    monthly_params: pd.DataFrame,
    months_per_year: int = 12,
):
    """Transform monthly-block GEV truth into annual-block GEV truth."""

    annual = monthly_params.copy()
    mu, sigma, xi = gev_max_stable_params(
        annual["true_mu"].to_numpy(),
        annual["true_sigma"].to_numpy(),
        annual["true_xi"].to_numpy(),
        n_blocks=months_per_year,
    )
    annual["true_mu"] = mu
    annual["true_sigma"] = sigma
    annual["true_log_sigma"] = np.log(sigma)
    annual["true_xi"] = xi
    return annual


def assert_annual_is_monthly_max(
    monthly: pd.DataFrame,
    annual: pd.DataFrame,
):
    """Raise if any annual value is not the maximum of its 12 monthly values."""

    expected = monthly.drop(columns="month").groupby("year", as_index=False).max()
    pd.testing.assert_frame_equal(
        annual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
