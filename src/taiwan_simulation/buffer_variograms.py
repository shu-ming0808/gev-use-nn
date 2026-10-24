"""Descriptive variogram ranges for the Taiwan five-coefficient pilot.

These fits are exploratory distance diagnostics, NOT confidence intervals or
automatic CV tuning.  In particular, a response variogram can reflect a spatial
mean trend rather than residual dependence.  Fifty GRID provide only limited
evidence for a sill.  ``range_supported`` is a conservative numerical/plateau
screen; it is not a statistical guarantee that observations beyond it are
independent.  No GP, NN, files or formal CV are run here.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit


def spherical(h, nugget, partial_sill, scale):
    r = np.asarray(h, dtype=float) / scale
    return nugget + partial_sill * np.where(r < 1, 1.5*r - .5*r**3, 1.)


def exponential(h, nugget, partial_sill, scale):
    return nugget + partial_sill * (-np.expm1(-np.asarray(h, dtype=float)/scale))


def gaussian(h, nugget, partial_sill, scale):
    return nugget + partial_sill * (-np.expm1(-(np.asarray(h, dtype=float)/scale)**2))


MODELS = {
    'spherical': (spherical, 1.),
    'exponential': (exponential, -np.log(.05)),
    'gaussian': (gaussian, np.sqrt(-np.log(.05))),
}


def _validate(xy, values, n_bins, max_distance_fraction, min_pairs):
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 4 or not np.isfinite(xy).all():
        raise ValueError('Require at least four finite [x_km, y_km] coordinates.')
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError('Duplicate coordinates are not permitted.')
    if not isinstance(values, pd.DataFrame) or len(values) != len(xy) or values.shape[1] < 1:
        raise ValueError('Provide a coefficient DataFrame with one row per coordinate.')
    if not values.columns.is_unique or any(not isinstance(c, str) for c in values.columns):
        raise ValueError('Coefficient names must be unique strings.')
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite coefficients are not silently dropped.')
    if isinstance(n_bins, bool) or int(n_bins) != n_bins or n_bins < 4:
        raise ValueError('n_bins must be an integer >= 4.')
    if not np.isfinite(max_distance_fraction) or not 0 < max_distance_fraction <= 1:
        raise ValueError('max_distance_fraction must be in (0, 1].')
    if isinstance(min_pairs, bool) or int(min_pairs) != min_pairs or min_pairs < 1:
        raise ValueError('min_pairs must be a positive integer.')
    return xy


def _fit_one(model, h, empirical, counts, variance, max_lag, min_pair_distance):
    fn, factor = MODELS[model]
    # Normalize by empirical coefficient variance so all five targets have the
    # same optimization scales. Parameters are restored to coefficient^2 below.
    normalized = empirical / variance
    lower = np.array([0., 0., min_pair_distance*.05])
    upper = np.array([10., 20., max_lag*20.])
    sigma = 1. / np.sqrt(counts)
    successes = []
    errors = []
    for fraction in (.15, .5, 1.5):
        initial = np.array([min(float(normalized.min()), .2),
                            min(max(float(normalized.max()), .1), 5.),
                            max_lag*fraction])
        initial = np.maximum(lower + 1e-10, np.minimum(initial, upper - 1e-10))
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', OptimizeWarning)
                pars, covariance = curve_fit(fn, h, normalized, p0=initial,
                    sigma=sigma, bounds=(lower, upper), maxfev=15000,
                    ftol=1e-10, xtol=1e-10, gtol=1e-10)
            loss = float(np.average((fn(h, *pars)-normalized)**2, weights=counts))
            if np.isfinite(pars).all() and np.isfinite(loss):
                successes.append((loss, pars, covariance,
                    any(issubclass(w.category, OptimizeWarning) for w in caught)))
        except (ValueError, RuntimeError, FloatingPointError) as error:
            errors.append(type(error).__name__)
    base = dict(model=model, fit_success=False, nugget=np.nan, partial_sill=np.nan,
        sill=np.nan, scale_parameter_km=np.nan, effective_range_km=np.nan,
        weighted_rmse=np.nan, max_lag_km=max_lag,
        min_observed_lag_km=float(h.min()), max_observed_lag_km=float(h.max()), n_bins_used=len(h),
        range_supported=False, bound_hit=False, observed_sill=False,
        covariance_finite=False, warning='all_optimizer_starts_failed:'+','.join(errors))
    if not successes:
        return base
    loss, pars, covariance, cov_warning = min(successes, key=lambda item: item[0])
    nugget, partial, scale = pars
    sill = nugget + partial
    effective = float(factor*scale)
    # A zero nugget is legitimate and does not itself invalidate a range. A
    # vanishing partial sill, however, leaves the range unidentified.
    bound_hit = bool(np.any(pars >= upper*(1-1e-4)) or scale <= lower[2]*(1+1e-4))
    structured = bool(partial > .05*max(sill, 1e-12))
    tail = h >= effective
    # Require two empirical bins around the fitted plateau. This is a disclosed
    # descriptive tolerance, NOT a hypothesis test or a fitted-range CI.
    observed_sill = bool(tail.sum() >= 2 and sill > 0 and
        .75 <= np.average(normalized[tail], weights=counts[tail])/sill <= 1.25)
    messages = []
    if effective > h.max(): messages.append('range_beyond_observed_lags')
    if effective < min_pair_distance: messages.append('range_below_grid_resolution')
    if effective < h.min(): messages.append('unresolved_short_range')
    if bound_hit: messages.append('optimizer_bound_hit')
    if not structured: messages.append('partial_sill_negligible_range_unidentified')
    if not observed_sill: messages.append('no_empirical_sill_support')
    if cov_warning or not np.isfinite(covariance).all():
        messages.append('covariance_not_reliable')
    supported = bool(not bound_hit and structured and observed_sill and
                     max(min_pair_distance, float(h.min())) <= effective <= h.max())
    return dict(base, fit_success=True, nugget=float(nugget*variance),
        partial_sill=float(partial*variance), sill=float(sill*variance),
        scale_parameter_km=float(scale), effective_range_km=effective,
        weighted_rmse=float(np.sqrt(loss)*variance), range_supported=supported,
        bound_hit=bound_hit, observed_sill=observed_sill,
        covariance_finite=bool(np.isfinite(covariance).all() and not cov_warning),
        warning=';'.join(messages))


def diagnose_variograms(xy, coefficient_frame, *, n_bins=8,
                       max_distance_fraction=.6, min_pairs=10):
    """Return ``bins``, ``fits`` and one-row-per-coefficient ``summary`` tables.

    All input rows must be finite and coordinate-aligned. Pairs are binned up
    to ``max_distance_fraction * max(pair_distance)``. Bins with fewer than
    ``min_pairs`` remain in the output but are not fitted. At least four usable
    bins are required for three-parameter fitting. Model selection minimizes
    pair-count-weighted semivariance RMSE, in coefficient^2 units; that score is
    neither prediction RMSE nor comparable across differently scaled targets.

    Spherical range is its exact sill distance. Exponential/Gaussian effective
    ranges are where 95% of their partial sill is reached. A whole-data call is
    exploratory; do not use held-out response values to tune a final CV design.
    """
    xy = _validate(xy, coefficient_frame, n_bins, max_distance_fraction, min_pairs)
    ii, jj = np.triu_indices(len(xy), 1)
    distance = np.linalg.norm(xy[ii]-xy[jj], axis=1)
    max_lag = float(max_distance_fraction*distance.max())
    edges = np.linspace(0, max_lag, int(n_bins)+1)
    assignments = np.minimum(np.searchsorted(edges, distance, side='right')-1, int(n_bins)-1)
    bins, fits, summaries = [], [], []
    for coefficient in coefficient_frame:
        values = coefficient_frame[coefficient].to_numpy(dtype=float)
        variance = float(np.var(values, ddof=1))
        semivar = .5*(values[ii]-values[jj])**2
        local = []
        for index in range(int(n_bins)):
            selected = (assignments == index) & (distance <= max_lag)
            count = int(selected.sum())
            row = dict(coefficient=coefficient, bin=index+1,
                lag_km=float(distance[selected].mean()) if count else float((edges[index]+edges[index+1])/2),
                semivariance=float(semivar[selected].mean()) if count else np.nan,
                n_pairs=count, fit_used=count >= min_pairs, max_lag_km=max_lag)
            local.append(row)
        bins.extend(local)
        used = pd.DataFrame(local).query('fit_used')
        base = dict(coefficient=coefficient, best_model=None, effective_range_km=np.nan,
            weighted_rmse=np.nan, range_supported=False, max_lag_km=max_lag,
            min_observed_lag_km=float(used.lag_km.min()) if len(used) else np.nan,
            max_observed_lag_km=float(used.lag_km.max()) if len(used) else np.nan,
            n_bins_used=len(used), n_points=len(xy), sample_variance=variance,
            min_pair_distance_km=float(distance.min()), warning='')
        # Relative tolerance permits an exactly or effectively constant surface
        # while retaining very small but meaningful variation in slope targets.
        constant = np.ptp(values) <= 32*np.finfo(float).eps*max(1., float(np.max(np.abs(values))))
        if constant or not np.isfinite(variance) or variance <= 0 or len(used) < 4:
            reason = 'constant_surface' if constant or variance <= 0 else 'insufficient_pair_bins'
            summaries.append(dict(base, status=reason, warning=reason))
            for model in MODELS:
                fits.append(dict(coefficient=coefficient, model=model, fit_success=False,
                    nugget=np.nan, partial_sill=np.nan, sill=np.nan, scale_parameter_km=np.nan,
                    effective_range_km=np.nan, weighted_rmse=np.nan, max_lag_km=max_lag,
                    min_observed_lag_km=base['min_observed_lag_km'],
                    max_observed_lag_km=base['max_observed_lag_km'], n_bins_used=len(used),
                    range_supported=False, bound_hit=False, observed_sill=False,
                    covariance_finite=False, warning=reason))
            continue
        coefficient_fits = [dict(coefficient=coefficient, **_fit_one(model,
            used.lag_km.to_numpy(), used.semivariance.to_numpy(), used.n_pairs.to_numpy(),
            variance, max_lag, float(distance.min()))) for model in MODELS]
        fits.extend(coefficient_fits)
        successful = [row for row in coefficient_fits if row['fit_success']]
        if not successful:
            summaries.append(dict(base, status='fit_failed', warning='all_models_failed'))
            continue
        best = min(successful, key=lambda row: row['weighted_rmse'])
        summaries.append(dict(base, status='fitted_descriptive', best_model=best['model'],
            effective_range_km=best['effective_range_km'], weighted_rmse=best['weighted_rmse'],
            range_supported=best['range_supported'], warning=best['warning']))
    return dict(bins=pd.DataFrame(bins), fits=pd.DataFrame(fits), summary=pd.DataFrame(summaries))
