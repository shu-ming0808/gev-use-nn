import numpy as np
from scipy.optimize import brentq, minimize_scalar


DEFAULT_PROBS = (0.25, 0.5, 0.75)
DEFAULT_XI_BOUNDS = (-0.5, 1.0)


def gev_quantile_basis(p, xi, eps=1e-8):
    """
    Basis term b(p, xi) in Q(p) = mu + sigma * b(p, xi).
    Here xi is the EVT shape parameter, not scipy.stats.genextreme's c=-xi.
    """
    t = -np.log(p)
    if abs(xi) < eps:
        return -np.log(t)
    return (t ** (-xi) - 1.0) / xi


def theoretical_quantile_ratio(xi, probs=DEFAULT_PROBS):
    p1, p2, p3 = probs
    b1 = gev_quantile_basis(p1, xi)
    b2 = gev_quantile_basis(p2, xi)
    b3 = gev_quantile_basis(p3, xi)
    return (b3 - b2) / (b2 - b1)


def solve_xi_from_ratio(r_data, probs=DEFAULT_PROBS, bounds=DEFAULT_XI_BOUNDS):
    lo, hi = bounds

    def objective(xi):
        return theoretical_quantile_ratio(xi, probs) - r_data

    grid = np.linspace(lo, hi, 301)
    values = np.array([objective(xi) for xi in grid], dtype=np.float64)

    finite = np.isfinite(values)
    grid = grid[finite]
    values = values[finite]
    if len(grid) < 2:
        raise ValueError("No finite values while solving xi.")

    for left_x, right_x, left_y, right_y in zip(
        grid[:-1], grid[1:], values[:-1], values[1:]
    ):
        if left_y == 0:
            return float(left_x), "root"
        if left_y * right_y < 0:
            return float(brentq(objective, left_x, right_x)), "root"

    result = minimize_scalar(
        lambda xi: objective(xi) ** 2,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success:
        raise ValueError("Could not solve xi from quantile ratio.")
    return float(result.x), "bounded_min"


def estimate_gev_quantile_ratio(y, probs=DEFAULT_PROBS, xi_bounds=DEFAULT_XI_BOUNDS):
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if len(y) < 3:
        raise ValueError("At least three observations are required.")

    p1, p2, p3 = probs
    if not (0 < p1 < p2 < p3 < 1):
        raise ValueError("probs must satisfy 0 < p1 < p2 < p3 < 1.")

    q1, q2, q3 = np.quantile(y, probs)
    lower_gap = q2 - q1
    upper_gap = q3 - q2
    if lower_gap <= 0 or upper_gap <= 0:
        raise ValueError("Quantile gaps must be positive.")

    r_data = upper_gap / lower_gap
    xi_hat, solve_status = solve_xi_from_ratio(r_data, probs=probs, bounds=xi_bounds)

    b1 = gev_quantile_basis(p1, xi_hat)
    b2 = gev_quantile_basis(p2, xi_hat)
    basis_gap = b2 - b1
    if basis_gap <= 0:
        raise ValueError("Invalid basis gap while estimating sigma.")

    sigma_hat = lower_gap / basis_gap
    if sigma_hat <= 0 or not np.isfinite(sigma_hat):
        raise ValueError("Invalid sigma estimate.")

    mu_hat = q2 - sigma_hat * b2

    return {
        "mu": float(mu_hat),
        "sigma": float(sigma_hat),
        "log_sigma": float(np.log(sigma_hat)),
        "xi": float(xi_hat),
        "ratio": float(r_data),
        "solve_status": solve_status,
        "p1": float(p1),
        "p2": float(p2),
        "p3": float(p3),
    }
