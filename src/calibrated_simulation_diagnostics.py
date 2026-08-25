"""Diagnostics for the calibrated Taiwan-GRID simulation.

The generated parameter surfaces should resemble the spatial scale of the
real NN-derived surfaces without being either artificially smooth or dominated
by cell-to-cell noise.  This module compares maps, marginal distributions,
normalized empirical variograms, and nearest-neighbour roughness on the same
1,385-cell grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from elevation_gp_analysis import empirical_variogram
from project_paths import FIGURE_DIR, PROCESSED_DATA_DIR, SIMULATED_DATA_DIR, TABLE_DIR


REAL_GRID_PATH = PROCESSED_DATA_DIR / "model_ready_grid_parameters.csv"
SIMULATION_PATH = (
    SIMULATED_DATA_DIR
    / "calibrated_final_model"
    / "replicate_000_model_ready.csv"
)
MONTHLY_MAXIMA_PATH = (
    SIMULATED_DATA_DIR
    / "calibrated_final_model"
    / "replicate_000_monthly_maxima.csv"
)

PARAMETERS = {
    "mu": ("mu_hat", "mu_true", r"$\mu$"),
    "log_sigma": ("log_sigma_hat", "log_sigma_true", r"$\log\sigma$"),
    "xi": ("xi_hat", "xi_true", r"$\xi$"),
}


def plot_monthly_maxima_examples(
    model_ready: pd.DataFrame,
    monthly_maxima: pd.DataFrame,
    year_months: tuple[tuple[int, int], ...] = ((1980, 1), (2002, 7), (2024, 12)),
) -> plt.Figure:
    """Plot generated monthly-maximum temperatures on the Taiwan GRID."""
    required_grid = {"station", "x_km", "y_km"}
    missing_grid = required_grid.difference(model_ready.columns)
    if missing_grid:
        raise ValueError(f"Model-ready table missing: {sorted(missing_grid)}")
    columns = [f"monthly_max_{year}_{month:02d}" for year, month in year_months]
    missing_monthly = {"station", *columns}.difference(monthly_maxima.columns)
    if missing_monthly:
        raise ValueError(f"Monthly-maxima table missing: {sorted(missing_monthly)}")
    merged = model_ready[["station", "x_km", "y_km"]].merge(
        monthly_maxima[["station", *columns]],
        on="station",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(model_ready):
        raise ValueError("Monthly maxima do not cover every simulated GRID.")
    all_values = merged[columns].to_numpy(dtype=float).ravel()
    vmin, vmax = np.nanquantile(all_values, [0.01, 0.99])
    figure, axes = plt.subplots(
        1,
        len(columns),
        figsize=(4.25 * len(columns), 5.2),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, (year, month), column in zip(axes[0], year_months, columns):
        points = axis.scatter(
            merged["x_km"],
            merged["y_km"],
            c=merged[column],
            s=11,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(f"Simulated {year}-{month:02d}")
        axis.set_xlabel("Easting (km)")
        axis.set_ylabel("Northing (km)")
        axis.set_aspect("equal")
        figure.colorbar(points, ax=axis, shrink=0.78, label=r"Monthly maximum ($^\circ$C)")
    figure.suptitle("Generated monthly-maximum temperature fields on the Taiwan GRID")
    return figure


def load_comparable_surfaces(
    real_path: str | Path = REAL_GRID_PATH,
    simulation_path: str | Path = SIMULATION_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and align real and simulated surfaces by GRID identifier."""
    real = pd.read_csv(real_path)
    simulated = pd.read_csv(simulation_path)
    required_real = {"station", "x_km", "y_km"} | {
        columns[0] for columns in PARAMETERS.values()
    }
    required_sim = {"station", "x_km", "y_km"} | {
        columns[1] for columns in PARAMETERS.values()
    }
    missing_real = sorted(required_real.difference(real.columns))
    missing_sim = sorted(required_sim.difference(simulated.columns))
    if missing_real or missing_sim:
        raise ValueError(
            f"Missing columns; real={missing_real}, simulated={missing_sim}."
        )
    common = sorted(set(real["station"]).intersection(simulated["station"]))
    if not common:
        raise ValueError("Real and simulated tables have no common GRID identifiers.")
    real = real.loc[real["station"].isin(common)].sort_values("station").reset_index(drop=True)
    simulated = (
        simulated.loc[simulated["station"].isin(common)]
        .sort_values("station")
        .reset_index(drop=True)
    )
    if not np.allclose(real[["x_km", "y_km"]], simulated[["x_km", "y_km"]]):
        raise ValueError("Real and simulated coordinates do not align.")
    return real, simulated


def distribution_summary(real: pd.DataFrame, simulated: pd.DataFrame) -> pd.DataFrame:
    """Summarize the marginal range of each real and simulated surface."""
    rows: list[dict] = []
    for parameter, (real_column, simulated_column, _) in PARAMETERS.items():
        for source, values in (
            ("Real NN-derived", real[real_column]),
            ("Calibrated simulated truth", simulated[simulated_column]),
        ):
            values = np.asarray(values, dtype=float)
            rows.append(
                {
                    "parameter": parameter,
                    "source": source,
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "p05": float(np.quantile(values, 0.05)),
                    "median": float(np.median(values)),
                    "p95": float(np.quantile(values, 0.95)),
                }
            )
    return pd.DataFrame(rows)


def normalized_variograms(
    real: pd.DataFrame,
    simulated: pd.DataFrame,
    n_lags: int = 22,
    maxlag_fraction: float = 0.55,
) -> pd.DataFrame:
    """Calculate variance-normalized variograms for comparable curve shapes."""
    coordinates = real[["x_km", "y_km"]].to_numpy(dtype=float)
    parts: list[pd.DataFrame] = []
    for parameter, (real_column, simulated_column, _) in PARAMETERS.items():
        for source, values in (
            ("Real NN-derived", real[real_column].to_numpy(dtype=float)),
            ("Calibrated simulated truth", simulated[simulated_column].to_numpy(dtype=float)),
        ):
            variance = float(np.var(values, ddof=1))
            curve = empirical_variogram(
                coordinates,
                values,
                n_lags=n_lags,
                maxlag_fraction=maxlag_fraction,
            )
            curve["normalized_semivariance"] = (
                curve["semivariance"] / variance if variance > 0 else np.nan
            )
            curve["parameter"] = parameter
            curve["source"] = source
            parts.append(curve)
    return pd.concat(parts, ignore_index=True)


def nearest_neighbour_roughness(
    real: pd.DataFrame,
    simulated: pd.DataFrame,
) -> pd.DataFrame:
    """Compare local cell-to-cell variation after standardizing by surface SD."""
    coordinates = real[["x_km", "y_km"]].to_numpy(dtype=float)
    _, indices = cKDTree(coordinates).query(coordinates, k=2)
    neighbour = indices[:, 1]
    rows: list[dict] = []
    for parameter, (real_column, simulated_column, _) in PARAMETERS.items():
        metrics: dict[str, float] = {}
        for source, values in (
            ("real", real[real_column].to_numpy(dtype=float)),
            ("simulated", simulated[simulated_column].to_numpy(dtype=float)),
        ):
            differences = values - values[neighbour]
            sd = float(np.std(values, ddof=1))
            rms = float(np.sqrt(np.mean(differences**2)))
            metrics[f"{source}_surface_sd"] = sd
            metrics[f"{source}_nn_rms_difference"] = rms
            metrics[f"{source}_normalized_roughness"] = rms / sd if sd > 0 else np.nan
        ratio = (
            metrics["simulated_normalized_roughness"]
            / metrics["real_normalized_roughness"]
        )
        rows.append(
            {
                "parameter": parameter,
                **metrics,
                "roughness_ratio_simulated_to_real": ratio,
            }
        )
    return pd.DataFrame(rows)


def plot_surface_comparison(real: pd.DataFrame, simulated: pd.DataFrame) -> plt.Figure:
    """Plot real and simulated parameter surfaces using a shared scale per column."""
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), constrained_layout=True)
    for column_index, (parameter, (real_column, simulated_column, label)) in enumerate(PARAMETERS.items()):
        combined = np.concatenate(
            [real[real_column].to_numpy(), simulated[simulated_column].to_numpy()]
        )
        vmin, vmax = np.quantile(combined, [0.01, 0.99])
        for row_index, (data, value_column, row_label) in enumerate(
            (
                (real, real_column, "Real NN-derived"),
                (simulated, simulated_column, "Calibrated simulated truth"),
            )
        ):
            axis = axes[row_index, column_index]
            scatter = axis.scatter(
                data["x_km"], data["y_km"], c=data[value_column], s=10,
                cmap="viridis", vmin=vmin, vmax=vmax, linewidths=0,
            )
            axis.set_title(f"{row_label}: {label}")
            axis.set_aspect("equal")
            axis.set_xlabel("Easting (km)")
            if column_index == 0:
                axis.set_ylabel("Northing (km)")
            figure.colorbar(scatter, ax=axis, shrink=0.78)
    figure.suptitle("Real and calibrated simulated GEV parameter surfaces", fontsize=15)
    return figure


def plot_distribution_comparison(real: pd.DataFrame, simulated: pd.DataFrame) -> plt.Figure:
    """Plot comparable marginal distributions for the three parameter surfaces."""
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    for axis, (_, (real_column, simulated_column, label)) in zip(axes, PARAMETERS.items()):
        axis.hist(real[real_column], bins=30, density=True, alpha=0.55, label="Real NN-derived")
        axis.hist(
            simulated[simulated_column], bins=30, density=True, alpha=0.55,
            label="Calibrated simulated truth",
        )
        axis.set_title(label)
        axis.set_xlabel("Parameter value")
        axis.set_ylabel("Density")
        axis.legend(fontsize=8)
    figure.suptitle("Marginal distribution check")
    return figure


def plot_normalized_variograms(variograms: pd.DataFrame) -> plt.Figure:
    """Plot normalized variograms; short-lag mismatch diagnoses smoothness."""
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    for axis, (parameter, (_, _, label)) in zip(axes, PARAMETERS.items()):
        subset = variograms.loc[variograms["parameter"].eq(parameter)]
        for source, part in subset.groupby("source", sort=False):
            axis.plot(
                part["lag_km"], part["normalized_semivariance"],
                marker="o", markersize=3, linewidth=1.5, label=source,
            )
        axis.axhline(1.0, color="0.55", linestyle="--", linewidth=1)
        axis.set_title(label)
        axis.set_xlabel("Distance (km)")
        axis.set_ylabel("Normalized semivariance")
        axis.legend(fontsize=8)
    figure.suptitle("Smoothness check using variance-normalized variograms")
    return figure


def run_diagnostics(
    real_path: str | Path = REAL_GRID_PATH,
    simulation_path: str | Path = SIMULATION_PATH,
    figure_directory: str | Path = FIGURE_DIR,
    table_directory: str | Path = TABLE_DIR,
) -> dict[str, object]:
    """Run all diagnostics and save presentation-ready outputs."""
    real, simulated = load_comparable_surfaces(real_path, simulation_path)
    figure_directory = Path(figure_directory)
    table_directory = Path(table_directory)
    figure_directory.mkdir(parents=True, exist_ok=True)
    table_directory.mkdir(parents=True, exist_ok=True)

    distributions = distribution_summary(real, simulated)
    variograms = normalized_variograms(real, simulated)
    roughness = nearest_neighbour_roughness(real, simulated)

    distributions.to_csv(
        table_directory / "calibrated_simulation_distribution_summary.csv",
        index=False, encoding="utf-8-sig",
    )
    variograms.to_csv(
        table_directory / "calibrated_simulation_normalized_variograms.csv",
        index=False, encoding="utf-8-sig",
    )
    roughness.to_csv(
        table_directory / "calibrated_simulation_roughness_summary.csv",
        index=False, encoding="utf-8-sig",
    )

    figures = {
        "surfaces": plot_surface_comparison(real, simulated),
        "distributions": plot_distribution_comparison(real, simulated),
        "variograms": plot_normalized_variograms(variograms),
    }
    output_names = {
        "surfaces": "calibrated_simulation_real_vs_truth_surfaces.png",
        "distributions": "calibrated_simulation_parameter_distributions.png",
        "variograms": "calibrated_simulation_normalized_variograms.png",
    }
    for name, figure in figures.items():
        figure.savefig(figure_directory / output_names[name], dpi=220, bbox_inches="tight")

    return {
        "real": real,
        "simulated": simulated,
        "distributions": distributions,
        "variograms": variograms,
        "roughness": roughness,
        "figures": figures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-path", type=Path, default=REAL_GRID_PATH)
    parser.add_argument("--simulation-path", type=Path, default=SIMULATION_PATH)
    parser.add_argument("--figure-directory", type=Path, default=FIGURE_DIR)
    parser.add_argument("--table-directory", type=Path, default=TABLE_DIR)
    args = parser.parse_args()
    outputs = run_diagnostics(
        args.real_path, args.simulation_path, args.figure_directory, args.table_directory
    )
    print(outputs["roughness"].to_string(index=False))


if __name__ == "__main__":
    main()
