"""Plug-in annual-temperature CRPS, in degrees C (EVT xi, not scipy -xi).

Friederichs & Thorarinsdottir (2012), doi:10.1002/env.2176, equations 9--10.
This implementation deliberately supports xi < 1 (finite first moment).
Unsupported/invalid predictions remain NaN and invalidate the aggregate mean;
they are never dropped to improve a score. An observation outside GEV support
is NOT invalid: the CDF extends to zero/one and its CRPS is still scored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import exp1, gamma, gammainc
from scipy.stats import genextreme

from research_workflows import YEARS, REFERENCE_YEAR, TIME_SCALE_YEARS


def _near_zero_crps(z, xi):
    # CDF-integral fallback avoids cancellation in the closed form near xi=0.
    # Splitting at zero also keeps the important central region visible to quad.
    def f(x):
        with np.errstate(over='ignore', under='ignore'):
            return float(genextreme.cdf(x, -xi)) ** 2
    def s(x):
        with np.errstate(over='ignore', under='ignore'):
            return float(genextreme.sf(x, -xi)) ** 2
    intervals = [(f, -np.inf, min(z, 0.)), (s, max(z, 0.), np.inf)]
    intervals.append((f if z >= 0 else s, min(z, 0.), max(z, 0.)))
    result = 0.
    for fun, lo, hi in intervals:
        if lo == hi:
            continue
        value, error = quad(fun, lo, hi, epsabs=2e-10, epsrel=2e-10, limit=200)
        if not np.isfinite(value) or error > 1e-7 * max(1., abs(value)):
            return np.nan
        result += value
    return result


def gev_crps(observation, location, scale, shape):
    """Broadcast arrays; NaN for invalid values or unsupported xi >= 1."""
    y, mu, sigma, xi = np.broadcast_arrays(*[
        np.asarray(a, dtype=float) for a in (observation, location, scale, shape)])
    score = np.full(y.shape, np.nan)
    valid = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(xi) & (sigma > 0) & (xi < 1)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        z = (y - mu) / sigma
        valid &= np.isfinite(z)
        zero = valid & (xi == 0)
        d = z[zero]
        # exp1(exp(-z)) has a known large-positive-z limit; avoid exp underflow.
        central = d <= 35.
        g = np.empty_like(d)
        g[central] = -d[central] + np.euler_gamma - np.log(2.) + 2 * exp1(np.exp(-d[central]))
        u = np.exp(-d[~central])
        g[~central] = d[~central] - np.euler_gamma - np.log(2.) + 2 * u
        score[zero] = sigma[zero] * g
        regular = valid & (np.abs(xi) >= 1e-5)
        x, d = xi[regular], z[regular]
        logcdf = genextreme.logcdf(d, -x)
        f = np.exp(logcdf)
        term = (2*f-1) - gamma(1-x) * (np.exp2(x) - 2*gammainc(1-x, -logcdf))
        score[regular] = sigma[regular] * (d*(2*f-1) + term/x)
    near = valid & (xi != 0) & (np.abs(xi) < 1e-5)
    for index in zip(*np.nonzero(near)) if near.ndim else ([()] if near else []):
        score[index] = sigma[index] * _near_zero_crps(float(z[index]), float(xi[index]))
    # A negative/nonfinite numeric result is a reported failure, never clipped.
    return np.where(np.isfinite(score) & (score >= 0), score, np.nan)


def score_temperature_distribution(coefficients, observations, *, stage):
    """Score all GRID/year cells and retain invalid and support counts.

    Coefficients are point estimates: no parameter/GP uncertainty is integrated.
    Returns the full CRPS array plus overall/per-year summaries. Means are only
    populated when every requested cell is valid (no nanmean selection bias).
    """
    theta = np.asarray(coefficients, float)
    y = np.asarray(observations, float)
    years = np.asarray(YEARS)
    if theta.ndim != 2 or theta.shape[1] != 5 or y.shape != (len(theta), len(years)):
        raise ValueError('CRPS requires five coefficients and matching full 50-year observations')
    if len(theta) == 0 or not np.isfinite(y).all():
        raise ValueError('CRPS evaluation observations must be finite and nonempty')
    t = (years - REFERENCE_YEAR) / TIME_SCALE_YEARS
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        mu = theta[:, 0, None] + theta[:, 1, None] * t
        sigma = np.exp(theta[:, 2, None] + theta[:, 3, None] * t)
        xi = np.broadcast_to(theta[:, 4, None], y.shape)
        support = 1 + xi * (y-mu)/sigma
    scores = gev_crps(y, mu, sigma, xi)
    parameter_valid = np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0) & np.isfinite(xi)
    outside = parameter_valid & (support <= 0)
    def row(mask, year):
        a = scores[:, mask]
        finite = np.isfinite(a)
        return dict(stage=stage, year=year, n_total=a.size, n_scored=int(finite.sum()),
            n_invalid=int((~finite).sum()),
            n_invalid_parameters=int((~parameter_valid[:, mask]).sum()),
            n_unsupported_shape=int((parameter_valid[:, mask] & (xi[:, mask] >= 1)).sum()),
            n_outside_support=int(outside[:, mask].sum()),
            n_grids_outside_support=int(outside[:, mask].any(axis=1).sum()),
            mean_CRPS_C=float(a.mean()) if finite.all() else np.nan)
    overall = pd.DataFrame([row(np.ones(len(years), bool), 'all')])
    yearly = pd.DataFrame([row(years == year, int(year)) for year in years])
    return scores, overall, yearly
