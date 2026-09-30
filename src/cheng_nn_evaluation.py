"""GEV support diagnostics and conditional annual return-level recovery.

Coefficients are (mu0, beta_mu, eta0, beta_sigma, xi0), on the original
temperature scale. The EVT shape convention is used (SciPy c = -xi).
RL_T(t) is the 1 - 1/T quantile of the annual distribution at a fixed year;
it is not an expected waiting time under a changing future climate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def gev_parameter_curves(
    coefficients: np.ndarray,
    years: np.ndarray,
    *,
    reference_year: float,
    time_scale_years: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mu, sigma, xi and numerical parameter validity for each year."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    years = np.asarray(years, dtype=np.float64)
    if coefficients.ndim != 2 or coefficients.shape[1] != 5 or len(coefficients) == 0:
        raise ValueError("Coefficients must have shape (N, 5), with N > 0.")
    if years.ndim != 1 or len(years) == 0 or not np.all(np.isfinite(years)):
        raise ValueError("Years must be a nonempty finite vector.")
    if not np.isfinite(reference_year) or not np.isfinite(time_scale_years) or time_scale_years <= 0:
        raise ValueError("Reference year must be finite and time scale positive and finite.")
    time = (years - reference_year)[None, :] / time_scale_years
    mu0, beta_mu, eta0, beta_sigma, xi0 = coefficients.T
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        mu = mu0[:, None] + beta_mu[:, None] * time
        sigma = np.exp(eta0[:, None] + beta_sigma[:, None] * time)
    xi = np.broadcast_to(xi0[:, None], mu.shape)
    valid = (
        np.all(np.isfinite(coefficients), axis=1)[:, None]
        & np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0.0)
        & np.isfinite(xi)
    )
    return mu, sigma, xi, valid


def _support_margin(values, mu, sigma, xi):
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        margin = 1.0 + xi * ((values - mu) / sigma)
    # The Gumbel distribution has no finite endpoint, including when a
    # standardized value is too large to represent numerically.
    return np.where(xi == 0.0, 1.0, margin)


def check_gev_predictions(
    coefficients: np.ndarray,
    annual_maxima: np.ndarray,
    years: np.ndarray,
    *,
    reference_year: float,
    time_scale_years: float = 10.0,
    xi_training_range: tuple[float, float] = (-0.4, 0.4),
    boundary_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Check strict observed-data support without clipping or forcing xi > 0.

    A shape outside the training range is flagged separately; it does not
    invalidate a GEV distribution. A positive support margin close to zero
    is flagged as near-boundary, but is not treated as a violation.
    """
    mu, sigma, xi, parameter_valid = gev_parameter_curves(
        coefficients, years, reference_year=reference_year,
        time_scale_years=time_scale_years,
    )
    values = np.asarray(annual_maxima, dtype=np.float64)
    if values.shape != mu.shape:
        raise ValueError("Annual maxima must have shape (N, number of years).")
    if not np.isfinite(boundary_tolerance) or boundary_tolerance < 0:
        raise ValueError("Boundary tolerance must be finite and nonnegative.")
    if not np.all(np.isfinite(xi_training_range)) or xi_training_range[0] >= xi_training_range[1]:
        raise ValueError("Shape training range must contain increasing finite bounds.")
    eligible = parameter_valid & np.isfinite(values)
    margin = _support_margin(values, mu, sigma, xi)
    computable = eligible & np.isfinite(margin)
    support_valid = computable & (margin > 0.0)
    minimum = np.min(np.where(computable, margin, np.inf), axis=1)
    minimum[~np.isfinite(minimum)] = np.nan
    xi0 = np.asarray(coefficients)[:, 4]
    return pd.DataFrame({
        "sample_index": np.arange(len(mu)),
        "coefficients_finite": np.all(np.isfinite(coefficients), axis=1),
        "parameters_valid_all_years": np.all(parameter_valid, axis=1),
        "n_invalid_parameter_years": np.sum(~parameter_valid, axis=1),
        "n_nonfinite_observations": np.sum(~np.isfinite(values), axis=1),
        "n_support_violations": np.sum(computable & (margin <= 0.0), axis=1),
        "n_support_numerical_failures": np.sum(eligible & ~np.isfinite(margin), axis=1),
        "n_near_boundary": np.sum(support_valid & (margin <= boundary_tolerance), axis=1),
        "min_support_margin": minimum,
        "gev_valid_all_years": np.all(support_valid, axis=1),
        "xi": xi0,
        "xi_outside_training_range": np.isfinite(xi0) & (
            (xi0 < xi_training_range[0]) | (xi0 > xi_training_range[1])
        ),
    })


def conditional_return_levels(
    coefficients: np.ndarray,
    years: np.ndarray,
    return_periods=(50.0, 100.0),
    *,
    reference_year: float,
    time_scale_years: float = 10.0,
) -> np.ndarray:
    """Return an (N, years, periods) array; invalid numerical RLs are NaN.

    expm1 and the exact xi=0 limit avoid cancellation near zero shape.
    Positive and negative shape values are both permitted.
    """
    mu, sigma, xi, valid = gev_parameter_curves(
        coefficients, years, reference_year=reference_year,
        time_scale_years=time_scale_years,
    )
    periods = np.asarray(return_periods, dtype=np.float64)
    if periods.ndim != 1 or len(periods) == 0 or not np.all(np.isfinite(periods) & (periods > 1.0)):
        raise ValueError("Return periods must be a nonempty finite vector greater than one.")
    y = -np.log(-np.log1p(-1.0 / periods))
    xi = xi[:, :, None]
    factor = np.broadcast_to(y, (*mu.shape, len(periods))).copy()
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        np.divide(np.expm1(xi * y), xi, out=factor, where=xi != 0.0)
        levels = mu[:, :, None] + sigma[:, :, None] * factor
    margin = _support_margin(levels, mu[:, :, None], sigma[:, :, None], xi)
    usable = valid[:, :, None] & np.isfinite(levels) & np.isfinite(margin) & (margin > 0.0)
    return np.where(usable, levels, np.nan)


def _error_summary(predicted, truth, eligible):
    finite = np.isfinite(predicted) & np.isfinite(truth)
    use = finite & eligible
    with np.errstate(over="ignore", invalid="ignore"):
        error = predicted[use] - truth[use]
    # A nonrepresentable error is a numerical failure even if both operands
    # are finite. Keep its exclusion visible in the reported counts.
    error = error[np.isfinite(error)]
    n_used = len(error)
    if n_used:
        scale = np.max(np.abs(error))
        rmse = float(scale * np.sqrt(np.mean((error / scale) ** 2))) if scale else 0.0
        mae = float(scale * np.mean(np.abs(error / scale))) if scale else 0.0
        bias = float(scale * np.mean(error / scale)) if scale else 0.0
    else:
        rmse = mae = bias = np.nan
    return {
        "n_total": predicted.size, "n_used": n_used,
        "n_excluded": predicted.size - n_used,
        "used_fraction": n_used / predicted.size,
        "RMSE": rmse, "MAE": mae, "bias": bias,
    }


def return_level_recovery_metrics(
    predicted: np.ndarray,
    truth: np.ndarray,
    years: np.ndarray,
    return_periods,
    valid_sequences: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-year and pooled-year metrics with explicit exclusion counts.

    finite_RL includes all computable quantile pairs, even if the predicted
    distribution fails observed support. valid_GEV additionally requires all
    observations in that sequence to satisfy the predicted support.
    Pooled counts are sequence-year pairs, not independent observations.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    years = np.asarray(years)
    periods = np.asarray(return_periods)
    valid_sequences = np.asarray(valid_sequences, dtype=bool)
    if predicted.ndim != 3 or predicted.shape != truth.shape or predicted.size == 0:
        raise ValueError("Predicted and true RL arrays must have the same nonempty (N, years, periods) shape.")
    if predicted.shape[1:] != (len(years), len(periods)) or valid_sequences.shape != (len(predicted),):
        raise ValueError("Years, periods, or sequence validity do not match the RL arrays.")
    yearly, pooled = [], []
    for subset, eligible in (
        ("finite_RL", np.ones(len(predicted), dtype=bool)),
        ("valid_GEV", valid_sequences),
    ):
        for period_index, period in enumerate(periods):
            for year_index, year in enumerate(years):
                yearly.append({
                    "year": year, "return_period": period, "subset": subset,
                    **_error_summary(predicted[:, year_index, period_index], truth[:, year_index, period_index], eligible),
                })
            pooled.append({
                "return_period": period, "subset": subset,
                **_error_summary(predicted[:, :, period_index], truth[:, :, period_index], eligible[:, None]),
            })
    return pd.DataFrame(yearly), pd.DataFrame(pooled)
