"""Full-sequence, multi-start time-varying GEV likelihood benchmark.

This estimates the same five coefficients as Cheng NN, not a stationary GEV.
MLE is a finite-sample comparator, NOT a lower bound on achievable MSE.
No known generating coefficients or NN predictions are used as initial values.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.optimize import minimize

from cheng_nn_simulation import ParameterRanges

SUPPORT_EPS = 1e-8


def standardized_coefficients(coefficients, location_scale):
    result = np.asarray(coefficients, dtype=float).copy()
    median, iqr = np.asarray(location_scale, dtype=float).T
    result[:, 0] = (result[:, 0] - median) / iqr
    result[:, 1] /= iqr
    result[:, 2] -= np.log(iqr)
    return result


def original_coefficients(coefficients, location_scale):
    result = np.asarray(coefficients, dtype=float).copy()
    median, iqr = np.asarray(location_scale, dtype=float).T
    result[:, 0] = result[:, 0] * iqr + median
    result[:, 1] *= iqr
    result[:, 2] += np.log(iqr)
    return result


def _terms(parameters, observations, centered_time):
    """Optimizer coordinates are (mu0, beta_mu/sigma0, eta0, beta_sigma, xi)."""
    a, ratio, eta, trend, xi = parameters
    log_scale = eta + trend * centered_time
    inv_scale = np.exp(-log_scale)
    beta = ratio * np.exp(eta)
    z = (observations - a - beta * centered_time) * inv_scale
    dz = np.column_stack((
        -inv_scale,
        -np.exp(eta) * centered_time * inv_scale,
        -(observations - a) * inv_scale,
        -centered_time * z,
    ))
    return log_scale, z, dz, 1.0 + xi * z


def support_constraint(parameters, observations, centered_time):
    return _terms(parameters, observations, centered_time)[3] - SUPPORT_EPS


def support_jacobian(parameters, observations, centered_time):
    _, z, dz, _ = _terms(parameters, observations, centered_time)
    return np.column_stack((parameters[4] * dz, z))


def nll_and_gradient(parameters, observations, centered_time):
    """Stable exact GEV NLL and analytic gradient, including the xi=0 limit.

    Outside support, return a constraint-recovery penalty only for optimizer
    trial steps. Such points are never accepted as fitted distributions.
    """
    log_scale, z, dz, margin = _terms(parameters, observations, centered_time)
    xi = parameters[4]
    if np.any(margin <= 0):
        bad = np.minimum(margin - SUPPORT_EPS, 0.0)
        return (1e10 + 1e6 * float(bad @ bad),
                2e6 * bad @ support_jacobian(parameters, observations, centered_time))
    v = xi * z
    q = np.empty_like(z)
    dq_xi = np.empty_like(z)
    small = np.abs(v) < 1e-4
    # log(1+xi*z)/xi and its xi derivative; fifth-order stable limit.
    zs, vs = z[small], v[small]
    q[small] = zs * (1 - vs/2 + vs**2/3 - vs**3/4 + vs**4/5)
    dq_xi[small] = zs**2 * (-0.5 + 2*vs/3 - 3*vs**2/4 + 4*vs**3/5)
    if np.any(~small):
        log_margin = np.log1p(v[~small])
        q[~small] = log_margin / xi
        dq_xi[~small] = (v[~small] / margin[~small] - log_margin) / xi**2
    # Cap only impossible, extremely bad trial steps to avoid float overflow.
    if np.any(-q > 600):
        return 1e100, np.zeros(5)
    tail = np.exp(-q)
    dq = 1.0 + xi - tail
    loss = np.sum(log_scale + (1.0 + xi) * q + tail)
    gradient = np.empty(5)
    gradient[:4] = (dq / margin) @ dz
    gradient[2] += len(z)
    gradient[3] += np.sum(centered_time)
    gradient[4] = np.sum(q + dq * dq_xi)
    return float(loss), gradient


def fitting_bounds(median, iqr, ranges, domain):
    if domain == "training_domain":
        # Exact same coefficient support as the simulation design, not a prior penalty.
        return np.array([
            (np.asarray(ranges.mu0) - median) / iqr,
            ranges.beta_mu_over_sigma0,
            np.asarray(ranges.eta0) - np.log(iqr),
            ranges.beta_sigma,
            ranges.xi0,
        ])
    if domain == "wide_domain":
        # Explicit numerical/regularity guards, not claimed to be unconstrained MLE.
        return np.array([[-20, 20], [-3, 3], [-7, 4], [-1, 1], [-0.49, 0.8]])
    raise ValueError("domain must be training_domain or wide_domain")


def initial_points(observations, centered_time, bounds):
    """Four deterministic, data-only feasible starts; truth is never available."""
    intercept, slope = np.linalg.lstsq(
        np.column_stack((np.ones(len(observations)), centered_time)), observations,
        rcond=None,
    )[0]
    residual = observations - intercept - slope * centered_time
    q1, q3 = np.quantile(residual, [0.25, 0.75])
    scale = max((q3 - q1) / 1.573, 0.05)
    starts = []
    for use_trend, shape in ((True, 0.0), (True, -0.2), (True, 0.2), (False, 0.0)):
        p = np.array([intercept - np.euler_gamma * scale,
                      slope / scale if use_trend else 0.0,
                      np.log(scale), 0.0, shape])
        p = np.clip(p, bounds[:, 0] + 1e-7, bounds[:, 1] - 1e-7)
        # Shrink only the initial shape until support is feasible.
        for _ in range(60):
            if np.min(support_constraint(p, observations, centered_time)) > 0.05:
                break
            p[4] *= 0.5
        starts.append(p)
    return starts


def fit_time_varying_gev(observations, centered_time, *, ranges=None,
                         domain="training_domain", maxiter=600):
    """Fit one complete sequence, returning convergence and boundary diagnostics."""
    started = time.perf_counter()
    y, t = np.asarray(observations, float), np.asarray(centered_time, float)
    if (y.ndim != 1 or y.shape != t.shape or len(y) < 10
            or not np.isfinite(y).all() or not np.isfinite(t).all()
            or np.ptp(t) <= 0):
        raise ValueError("Need >=10 finite, aligned observations and varying times.")
    median = float(np.median(y))
    iqr = float(np.subtract(*np.quantile(y, [0.75, 0.25])))
    if iqr <= 1e-12:
        raise ValueError("Sequence IQR must be positive.")
    y = (y - median) / iqr
    ranges = ranges or ParameterRanges()
    ranges.validate()
    bounds = fitting_bounds(median, iqr, ranges, domain)
    constraints = [{"type": "ineq", "fun": support_constraint,
                    "jac": support_jacobian, "args": (y, t)}]
    accepted, all_results = [], []
    for start in initial_points(y, t, bounds):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Values in x were outside bounds")
            result = minimize(nll_and_gradient, start, args=(y, t), jac=True,
                              method="SLSQP", bounds=bounds, constraints=constraints,
                              options={"maxiter": maxiter, "ftol": 1e-9})
        loss, _ = nll_and_gradient(result.x, y, t)
        feasible = (np.all(np.isfinite(result.x)) and np.isfinite(loss)
                    and loss < 1e9
                    and np.min(support_constraint(result.x, y, t)) >= -1e-10
                    and np.all(result.x >= bounds[:, 0] - 1e-7)
                    and np.all(result.x <= bounds[:, 1] + 1e-7))
        all_results.append((result, loss, feasible))
        if result.success and feasible:
            accepted.append((result, loss))
    output = {"domain": domain, "success": bool(accepted),
              "successful_starts": len(accepted), "starts": len(all_results)}
    if accepted:
        best, loss = min(accepted, key=lambda pair: pair[1])
        a, ratio, eta, trend, xi = best.x
        standardized = np.array([[a, ratio * np.exp(eta), eta, trend, xi]])
        coefficients = original_coefficients(standardized, [[median, iqr]])[0]
        feasible_losses = [value for _, value, ok in all_results if ok]
        output.update({
            "coefficients": coefficients.tolist(),
            "nll": float(loss + len(y) * np.log(iqr)),
            "min_support_margin": float(np.min(support_constraint(best.x, y, t)) + SUPPORT_EPS),
            "agreeing_starts": sum(abs(value - loss) < 1e-4 for _, value in accepted),
            "lower_nonconverged_nll_gap": max(0.0, float(loss - min(feasible_losses))),
            "on_boundary": bool(np.any(np.minimum(best.x - bounds[:, 0], bounds[:, 1] - best.x) < 1e-4)),
            "boundary_coordinates": ",".join(
                name for name, distance in zip(
                    ("mu0", "beta_mu_over_sigma0", "eta0", "beta_sigma", "xi0"),
                    np.minimum(best.x - bounds[:, 0], bounds[:, 1] - best.x)) if distance < 1e-4),
            "iterations": int(best.nit), "message": str(best.message),
        })
    else:
        output.update({"coefficients": [float("nan")] * 5, "nll": float("nan"),
                       "min_support_margin": float("nan"), "agreeing_starts": 0,
                       "lower_nonconverged_nll_gap": float("nan"),
                       "on_boundary": False, "boundary_coordinates": "",
                       "iterations": max(result.nit for result, _, _ in all_results),
                       "message": " | ".join(str(result.message) for result, _, _ in all_results)})
    output["seconds"] = time.perf_counter() - started
    return output
