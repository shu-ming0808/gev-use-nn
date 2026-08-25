"""First-order sensitivity of GEV return levels to parameter error.

The project NN is trained on ``mu``, ``log_sigma`` and ``xi``, but the current
simulation error table reports RMSE on the natural ``sigma`` scale.  This
script therefore uses derivatives with respect to ``mu``, ``sigma`` and
``xi``.  A derivative with respect to ``log_sigma`` is also exported so future
analyses can use a log-scale RMSE without mixing parameter scales.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_paths import FIGURE_DIR, PROCESSED_DATA_DIR, TABLE_DIR


DEFAULT_RETURN_PERIODS = (2, 5, 10, 20, 50, 100, 500, 1000)
DEFAULT_RMSE_PATH = PROCESSED_DATA_DIR / "simulation_error_summary.csv"
DEFAULT_FIGURE_PATH = FIGURE_DIR / "return_level_parameter_sensitivity.png"
DEFAULT_TABLE_PATH = TABLE_DIR / "return_level_parameter_sensitivity.csv"


def reduced_variate(return_period: np.ndarray) -> np.ndarray:
    """Return the GEV reduced variate y_T for return periods T > 1."""
    periods = np.asarray(return_period, dtype=float)
    if np.any(~np.isfinite(periods)) or np.any(periods <= 1.0):
        raise ValueError("Every return period must be finite and greater than 1.")
    return -np.log(-np.log1p(-1.0 / periods))


def return_level_derivatives(
    return_period: np.ndarray,
    sigma: float,
    xi: float,
) -> pd.DataFrame:
    """Calculate stable local derivatives of z_T with respect to GEV parameters."""
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive.")
    if not np.isfinite(xi):
        raise ValueError("xi must be finite.")

    periods = np.asarray(return_period, dtype=float)
    y_t = reduced_variate(periods)
    dz_dmu = np.ones_like(y_t)

    # Use the Gumbel limits to avoid cancellation around xi = 0.
    if abs(xi) < 1e-6:
        dz_dsigma = y_t
        dz_dxi = 0.5 * sigma * y_t**2
    else:
        xi_y = xi * y_t
        expm1_xi_y = np.expm1(xi_y)
        dz_dsigma = expm1_xi_y / xi
        dz_dxi = sigma * (
            xi_y * np.exp(xi_y) - expm1_xi_y
        ) / xi**2

    return pd.DataFrame(
        {
            "return_period": periods,
            "y_T": y_t,
            "dz_dmu": dz_dmu,
            "dz_dsigma": dz_dsigma,
            "dz_dlog_sigma": sigma * dz_dsigma,
            "dz_dxi": dz_dxi,
        }
    )


def load_parameter_rmse(path: Path) -> dict[str, float]:
    """Read natural-scale mu, sigma and xi RMSE values from the NN simulation."""
    table = pd.read_csv(path)
    required_columns = {"param", "rmse"}
    if not required_columns.issubset(table.columns):
        raise ValueError(f"{path} must contain columns {sorted(required_columns)}.")

    values = table.set_index("param")["rmse"].to_dict()
    missing = {"mu", "sigma", "xi"} - set(values)
    if missing:
        raise ValueError(f"Missing RMSE values for: {sorted(missing)}")
    result = {name: float(values[name]) for name in ("mu", "sigma", "xi")}
    if any(not np.isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError("RMSE values must be finite and non-negative.")
    return result


def calculate_contributions(
    return_periods: np.ndarray,
    sigma: float,
    xi: float,
    rmse: dict[str, float],
) -> pd.DataFrame:
    """Calculate absolute first-order one-parameter error contributions."""
    result = return_level_derivatives(return_periods, sigma=sigma, xi=xi)
    result["C_mu"] = np.abs(result["dz_dmu"]) * rmse["mu"]
    result["C_sigma"] = np.abs(result["dz_dsigma"]) * rmse["sigma"]
    result["C_xi"] = np.abs(result["dz_dxi"]) * rmse["xi"]
    result["rss_if_uncorrelated"] = np.sqrt(
        result["C_mu"] ** 2 + result["C_sigma"] ** 2 + result["C_xi"] ** 2
    )
    squared_total = result[["C_mu", "C_sigma", "C_xi"]].pow(2).sum(axis=1)
    for name in ("mu", "sigma", "xi"):
        result[f"variance_share_{name}"] = result[f"C_{name}"] ** 2 / squared_total
    return result


def plot_sensitivity(
    result: pd.DataFrame,
    rmse: dict[str, float],
    sigma: float,
    xi: float,
) -> plt.Figure:
    """Create the derivative and RMSE-weighted contribution panels."""
    colors = {"mu": "#4C78A8", "sigma": "#F58518", "xi": "#E45756"}
    markers = {"mu": "o", "sigma": "s", "xi": "^"}
    labels = {"mu": r"$\mu$", "sigma": r"$\sigma$", "xi": r"$\xi$"}

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.25), constrained_layout=True)
    periods = result["return_period"].to_numpy(float)

    derivative_columns = {
        "mu": "dz_dmu",
        "sigma": "dz_dsigma",
        "xi": "dz_dxi",
    }
    contribution_columns = {name: f"C_{name}" for name in derivative_columns}

    for name, column in derivative_columns.items():
        axes[0].plot(
            periods,
            np.abs(result[column]),
            color=colors[name],
            marker=markers[name],
            linewidth=2.0,
            markersize=5.5,
            label=labels[name],
        )
    axes[0].set_title("A. Local return-level sensitivity")
    axes[0].set_ylabel(r"Absolute derivative $|\partial z_T/\partial\theta|$")

    for name, column in contribution_columns.items():
        axes[1].plot(
            periods,
            result[column],
            color=colors[name],
            marker=markers[name],
            linewidth=2.0,
            markersize=5.5,
            label=rf"{labels[name]}: RMSE={rmse[name]:.3f}",
        )
    axes[1].set_title("B. First-order NN error contribution")
    axes[1].set_ylabel(r"$C_\theta(T)=|\partial z_T/\partial\theta|\,\mathrm{RMSE}_\theta$")

    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Return period T (years)")
        axis.set_xticks(periods)
        axis.set_xticklabels([f"{int(period):,}" for period in periods], rotation=25)
        axis.grid(True, which="major", alpha=0.25, linewidth=0.7)
        axis.legend(frameon=True, fontsize=9)

    # Directly identify the dominant contribution at the two thesis targets.
    for period in (50, 100):
        row = result.loc[np.isclose(result["return_period"], period)]
        if row.empty:
            continue
        contributions = {
            name: float(row[f"C_{name}"].iloc[0]) for name in ("mu", "sigma", "xi")
        }
        dominant = max(contributions, key=contributions.get)
        value = contributions[dominant]
        axes[1].annotate(
            rf"$T={period}$: {labels[dominant]} dominant",
            xy=(period, value),
            xytext=(8, 10 if period == 50 else -18),
            textcoords="offset points",
            fontsize=8.5,
            color=colors[dominant],
            arrowprops={"arrowstyle": "-", "color": colors[dominant], "lw": 0.8},
        )

    figure.suptitle(
        rf"GEV return-level sensitivity ($\sigma={sigma:g}$, $\xi={xi:g}$)",
        fontsize=15,
    )
    figure.text(
        0.5,
        -0.015,
        "Contributions use natural-scale NN simulation RMSE and are local first-order approximations, not an exact error decomposition.",
        ha="center",
        fontsize=8.5,
    )
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--xi", type=float, default=0.1)
    parser.add_argument(
        "--return-periods",
        type=float,
        nargs="+",
        default=DEFAULT_RETURN_PERIODS,
    )
    parser.add_argument("--rmse-path", type=Path, default=DEFAULT_RMSE_PATH)
    parser.add_argument("--output-figure", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--output-table", type=Path, default=DEFAULT_TABLE_PATH)
    parser.add_argument(
        "--copy-figure-to",
        type=Path,
        default=None,
        help="Optional directory used to copy the final PNG for presentation use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rmse = load_parameter_rmse(args.rmse_path)
    result = calculate_contributions(
        np.asarray(args.return_periods, dtype=float),
        sigma=args.sigma,
        xi=args.xi,
        rmse=rmse,
    )
    result.insert(0, "sigma_reference", args.sigma)
    result.insert(1, "xi_reference", args.xi)
    for name, value in rmse.items():
        result[f"RMSE_{name}"] = value

    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_table, index=False, encoding="utf-8-sig")

    figure = plot_sensitivity(result, rmse, sigma=args.sigma, xi=args.xi)
    figure.savefig(args.output_figure, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    if args.copy_figure_to is not None:
        args.copy_figure_to.mkdir(parents=True, exist_ok=True)
        copied_path = args.copy_figure_to / args.output_figure.name
        shutil.copy2(args.output_figure, copied_path)
        print(copied_path)
    print(args.output_figure)
    print(args.output_table)


if __name__ == "__main__":
    main()
