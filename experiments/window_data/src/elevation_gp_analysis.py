"""Elevation-adjusted spatial GP analysis for the real TCCIP GRID data.

This module contains reusable data preparation, universal Gaussian-process,
buffered spatial cross-validation, residual-diagnostic, and return-level
functions.  The accompanying notebook should contain only the analysis order,
research explanations, configuration, and presentation of returned results.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    Matern,
    RBF,
    WhiteKernel,
)
from sklearn.neighbors import NearestNeighbors


PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from spatial_coordinates import add_twd97_km_columns  # noqa: E402


RANDOM_STATE = 111
TARGETS = {
    "mu": "mu_hat",
    "log_sigma": "log_sigma_hat",
    "xi": "xi_hat",
}
TARGET_LABELS = {
    "mu": r"$\hat{\mu}$",
    "log_sigma": r"$\widehat{\log\sigma}$",
    "xi": r"$\hat{\xi}$",
}


def resolve_analysis_paths(current_directory: str | Path) -> dict[str, Path]:
    """Resolve input and output paths from the experiment or notebook folder."""
    cwd = Path(current_directory).resolve()
    if (
        cwd
        / "experiments"
        / "window_data"
        / "data"
        / "processed"
    ).exists():
        window_root = cwd / "experiments" / "window_data"
    elif (
        cwd.name == "window_data"
        and (cwd / "data" / "processed").exists()
    ):
        window_root = cwd
    elif (
        cwd.parent.name == "window_data"
        and (cwd.parent / "data" / "processed").exists()
    ):
        window_root = cwd.parent
    else:
        raise FileNotFoundError(
            "請從 experiments/window_data 或其 notebooks 資料夾執行。"
        )

    repository_root = window_root.parents[1]
    workspace_root = repository_root.parent
    figure_directory = window_root / "results" / "figures"
    table_directory = window_root / "results" / "tables"
    figure_directory.mkdir(parents=True, exist_ok=True)
    table_directory.mkdir(parents=True, exist_ok=True)

    return {
        "window_root": window_root,
        "repository_root": repository_root,
        "gev_path": (
            window_root
            / "data"
            / "processed"
            / "grid_station_gev_params_with_loc.csv"
        ),
        "elevation_path": (
            workspace_root
            / "觀測_日資料_臺灣_網格高程"
            / "觀測_日資料_臺灣_網格高程.csv"
        ),
        "figure_directory": figure_directory,
        "table_directory": table_directory,
    }


def load_grid_with_elevation(
    gev_path: str | Path,
    elevation_path: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Read, audit, and one-to-one join GEV estimates with GRID elevation."""
    gev = pd.read_csv(gev_path)
    elevation = pd.read_csv(
        elevation_path,
        encoding="utf-8-sig",
        usecols=[0, 1, 2],
        skipinitialspace=True,
    )
    elevation.columns = [
        str(column).strip().lstrip("\ufeff")
        for column in elevation.columns
    ]
    elevation = elevation.rename(
        columns={"LON": "lon", "LAT": "lat", "Height(m)": "elevation_m"}
    )

    required = {"lon", "lat", "elevation_m"}
    if not required.issubset(elevation.columns):
        raise ValueError(
            f"高程欄位為 {elevation.columns.tolist()}，"
            f"預期至少包含 {sorted(required)}。"
        )

    for frame in (gev, elevation):
        frame["lon_key"] = pd.to_numeric(frame["lon"]).round(2)
        frame["lat_key"] = pd.to_numeric(frame["lat"]).round(2)

    if elevation.duplicated(["lon_key", "lat_key"]).any():
        raise ValueError("高程資料含有重複的經緯度鍵值。")
    if gev.duplicated(["lon_key", "lat_key"]).any():
        raise ValueError("GEV 資料含有重複的經緯度鍵值。")

    data = gev.merge(
        elevation[["lon_key", "lat_key", "elevation_m"]],
        on=["lon_key", "lat_key"],
        how="left",
        validate="one_to_one",
    )
    if data["elevation_m"].isna().any():
        raise ValueError(
            f"有 {int(data['elevation_m'].isna().sum())} 個 GRID 無法配對高程。"
        )
    if (data["elevation_m"] < 0).any():
        raise ValueError("高程出現負值，需先檢查資料定義。")

    data = add_twd97_km_columns(data)
    if output_path is not None:
        data.to_csv(output_path, index=False, encoding="utf-8-sig")
    return data


def exploratory_elevation_analysis(
    data: pd.DataFrame,
    figure_directory: str | Path | None = None,
    table_directory: str | Path | None = None,
) -> tuple[pd.DataFrame, plt.Figure, plt.Figure]:
    """Summarize elevation and visualize its marginal associations."""
    rows: list[dict] = []
    association_figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = data["elevation_m"].to_numpy(float)

    for axis, (target, column) in zip(axes, TARGETS.items()):
        y = data[column].to_numpy(float)
        pearson = pearsonr(x, y)
        spearman = spearmanr(x, y)
        slope, intercept = np.polyfit(x, y, deg=1)
        fitted = intercept + slope * x
        total = np.sum((y - y.mean()) ** 2)
        residual = np.sum((y - fitted) ** 2)
        rows.append(
            {
                "target": target,
                "pearson_r": pearson.statistic,
                "spearman_rho": spearman.statistic,
                "slope_per_1000m": slope * 1000.0,
                "simple_R2": 1.0 - residual / total,
            }
        )
        order = np.argsort(x)
        axis.scatter(x, y, s=10, alpha=0.35)
        axis.plot(x[order], fitted[order], color="crimson", linewidth=2)
        axis.set(
            title=TARGET_LABELS[target],
            xlabel="Elevation (m)",
            ylabel=column,
        )

    association_figure.suptitle(
        "Exploratory association between elevation and NN-derived GEV parameters"
    )
    association_figure.tight_layout()

    map_figure, map_axis = plt.subplots(figsize=(6.2, 7.2))
    points = map_axis.scatter(
        data["x_km"],
        data["y_km"],
        c=data["elevation_m"],
        s=13,
        cmap="terrain",
    )
    map_figure.colorbar(points, ax=map_axis, label="Elevation (m)")
    map_axis.set(
        title="Elevation matched to the TCCIP GRID cells",
        xlabel="TWD97 Easting (km)",
        ylabel="TWD97 Northing (km)",
        aspect="equal",
    )
    map_figure.tight_layout()

    summary = pd.DataFrame(rows)
    if table_directory is not None:
        table_directory = Path(table_directory)
        summary.to_csv(
            table_directory / "elevation_exploratory_associations.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if figure_directory is not None:
        figure_directory = Path(figure_directory)
        association_figure.savefig(
            figure_directory / "elevation_01_parameter_associations.png",
            bbox_inches="tight",
        )
        map_figure.savefig(
            figure_directory / "elevation_02_grid_map.png",
            bbox_inches="tight",
        )
    return summary, association_figure, map_figure


def prepare_spatial_folds(
    data: pd.DataFrame,
    n_folds: int = 5,
    random_state: int = RANDOM_STATE,
    figure_path: str | Path | None = None,
) -> tuple[pd.DataFrame, plt.Figure]:
    """Project coordinates to TWD97 / TM2 and create coordinate K-means folds."""
    prepared = data.copy()
    prepared = add_twd97_km_columns(prepared)
    xy = prepared[["x_km", "y_km"]].to_numpy(float)
    prepared["spatial_fold"] = KMeans(
        n_clusters=n_folds,
        n_init=100,
        random_state=random_state,
    ).fit_predict(xy)

    figure, axis = plt.subplots(figsize=(6.2, 7.2))
    for fold in range(n_folds):
        part = prepared.loc[prepared["spatial_fold"] == fold]
        axis.scatter(
            part["x_km"],
            part["y_km"],
            s=13,
            label=f"Fold {fold}",
        )
    axis.set(
        title="Fixed geographic folds for all model comparisons",
        xlabel="TWD97 Easting (km)",
        ylabel="TWD97 Northing (km)",
        aspect="equal",
    )
    axis.legend()
    figure.tight_layout()
    if figure_path is not None:
        figure.savefig(figure_path, bbox_inches="tight")
    return prepared, figure


def _angular_distance_degrees(
    angles: np.ndarray,
    direction: float,
) -> np.ndarray:
    """Return axial angular distance, where 0 and 180 degrees are identical."""
    return np.abs((angles - direction + 90.0) % 180.0 - 90.0)


def _diagnostic_pair_geometry(
    coordinates: np.ndarray,
) -> dict[str, np.ndarray]:
    """Precompute pair indices, distances, directions, and distance bins."""
    n = len(coordinates)
    pair_i, pair_j = np.triu_indices(n, k=1)
    delta = coordinates[pair_j] - coordinates[pair_i]
    distance = np.sqrt(np.sum(delta**2, axis=1))
    angle = np.degrees(np.arctan2(delta[:, 1], delta[:, 0])) % 180.0
    positive = distance > 0
    pair_i = pair_i[positive]
    pair_j = pair_j[positive]
    distance = distance[positive]
    angle = angle[positive]

    maximum_distance = float(np.quantile(distance, 0.60))
    edges = np.linspace(0.0, maximum_distance, 8)
    distance_bin = np.digitize(distance, edges, right=False) - 1
    in_range = (distance_bin >= 0) & (distance_bin < len(edges) - 1)
    return {
        "i": pair_i,
        "j": pair_j,
        "distance": distance,
        "angle": angle,
        "distance_bin": distance_bin,
        "in_range": in_range,
        "edges": edges,
    }


def _spatial_screening_statistics(
    residual: np.ndarray,
    elevation: np.ndarray,
    geometry: dict[str, np.ndarray],
) -> dict[str, float]:
    """Calculate scale-free statistics used by the diagnostic simulation."""
    residual = np.asarray(residual, dtype=float)
    elevation = np.asarray(elevation, dtype=float)
    residual_variance = max(float(np.var(residual, ddof=1)), 1e-12)
    elevation_scale = max(float(np.std(elevation)), 1e-12)
    z = (elevation - np.mean(elevation)) / elevation_scale

    linear_design = np.column_stack([np.ones(len(z)), z])
    nonlinear_design = np.column_stack(
        [np.ones(len(z)), z, z**2, z**3, z**4]
    )
    linear_beta = np.linalg.lstsq(
        linear_design,
        residual,
        rcond=None,
    )[0]
    nonlinear_beta = np.linalg.lstsq(
        nonlinear_design,
        residual,
        rcond=None,
    )[0]
    nonlinear_component = (
        nonlinear_design @ nonlinear_beta
        - linear_design @ linear_beta
    )
    nonlinear_mean = float(
        np.sqrt(np.mean(nonlinear_component**2))
        / np.sqrt(residual_variance)
    )

    pair_i = geometry["i"]
    pair_j = geometry["j"]
    semivariance = 0.5 * (residual[pair_i] - residual[pair_j]) ** 2
    normalized_semivariance = semivariance / residual_variance
    distance_bin = geometry["distance_bin"]
    in_range = geometry["in_range"]

    directional_values: list[float] = []
    for distance_index in range(len(geometry["edges"]) - 1):
        bin_values: list[float] = []
        for direction in (0.0, 45.0, 90.0, 135.0):
            selected = (
                in_range
                & (distance_bin == distance_index)
                & (
                    _angular_distance_degrees(
                        geometry["angle"],
                        direction,
                    )
                    <= 22.5
                )
            )
            if int(selected.sum()) >= 20:
                bin_values.append(
                    float(np.mean(normalized_semivariance[selected]))
                )
        if len(bin_values) == 4:
            directional_values.append(max(bin_values) - min(bin_values))
    anisotropy = (
        float(np.median(directional_values))
        if directional_values
        else np.nan
    )

    elevation_band = np.asarray(
        pd.qcut(
            elevation,
            q=3,
            labels=False,
            duplicates="drop",
        )
    )
    band_variances = [
        float(np.var(residual[elevation_band == band], ddof=1))
        for band in np.unique(elevation_band)
        if int(np.sum(elevation_band == band)) >= 3
    ]
    positive_variances = [value for value in band_variances if value > 0]
    variance_nonstationarity = (
        float(np.log(max(positive_variances) / min(positive_variances)))
        if len(positive_variances) >= 2
        else np.nan
    )

    short_distance = geometry["distance"] <= np.quantile(
        geometry["distance"],
        0.25,
    )
    local_dependence_values: list[float] = []
    for band in np.unique(elevation_band):
        selected = (
            short_distance
            & (elevation_band[pair_i] == band)
            & (elevation_band[pair_j] == band)
        )
        band_selected = elevation_band == band
        band_variance = float(np.var(residual[band_selected], ddof=1))
        if int(selected.sum()) >= 30 and band_variance > 0:
            local_dependence_values.append(
                float(np.mean(semivariance[selected]) / band_variance)
            )
    range_nonstationarity = (
        max(local_dependence_values) - min(local_dependence_values)
        if len(local_dependence_values) >= 2
        else np.nan
    )

    valid_pairs = in_range & np.isfinite(normalized_semivariance)
    distance_dummies = np.eye(len(geometry["edges"]) - 1)[
        distance_bin[valid_pairs]
    ][:, 1:]
    elevation_gap = np.abs(
        elevation[pair_i[valid_pairs]] - elevation[pair_j[valid_pairs]]
    )
    elevation_gap = (
        elevation_gap - elevation_gap.mean()
    ) / max(float(elevation_gap.std()), 1e-12)
    elevation_distance_design = np.column_stack(
        [
            np.ones(int(valid_pairs.sum())),
            distance_dummies,
            elevation_gap,
        ]
    )
    elevation_distance_beta = np.linalg.lstsq(
        elevation_distance_design,
        normalized_semivariance[valid_pairs],
        rcond=None,
    )[0]
    elevation_distance = abs(float(elevation_distance_beta[-1]))

    return {
        "nonlinear_mean": nonlinear_mean,
        "anisotropy": anisotropy,
        "variance_nonstationarity": variance_nonstationarity,
        "range_nonstationarity": range_nonstationarity,
        "elevation_distance": elevation_distance,
    }


def _plot_screening_diagnostics(
    axes: np.ndarray,
    row: int,
    target: str,
    elevation: np.ndarray,
    residual: np.ndarray,
    coordinates: np.ndarray,
    geometry: dict[str, np.ndarray],
) -> None:
    """Add observed residual, directional, and elevation-band diagnostics."""
    residual_axis, direction_axis, local_axis = axes[row]
    residual_axis.scatter(elevation, residual, s=9, alpha=0.25)
    order = np.argsort(elevation)
    bins = pd.qcut(elevation, q=12, labels=False, duplicates="drop")
    binned = (
        pd.DataFrame(
            {
                "elevation": elevation,
                "residual": residual,
                "bin": bins,
            }
        )
        .groupby("bin", observed=True)
        .mean()
    )
    residual_axis.plot(
        binned["elevation"],
        binned["residual"],
        color="crimson",
        marker="o",
        linewidth=1.5,
    )
    residual_axis.axhline(0.0, color="black", linewidth=0.8)
    residual_axis.set(
        title=f"{TARGET_LABELS[target]}: trend residual vs elevation",
        xlabel="Elevation (m)",
        ylabel="Trend residual",
    )

    pair_i = geometry["i"]
    pair_j = geometry["j"]
    semivariance = 0.5 * (residual[pair_i] - residual[pair_j]) ** 2
    centers = 0.5 * (geometry["edges"][:-1] + geometry["edges"][1:])
    for direction in (0.0, 45.0, 90.0, 135.0):
        values = []
        for distance_index in range(len(centers)):
            selected = (
                geometry["in_range"]
                & (geometry["distance_bin"] == distance_index)
                & (
                    _angular_distance_degrees(
                        geometry["angle"],
                        direction,
                    )
                    <= 22.5
                )
            )
            values.append(
                float(np.mean(semivariance[selected]))
                if int(selected.sum()) >= 20
                else np.nan
            )
        direction_axis.plot(
            centers,
            values,
            marker="o",
            linewidth=1.2,
            label=f"{direction:g}°",
        )
    direction_axis.set(
        title=f"{TARGET_LABELS[target]}: directional variograms",
        xlabel="Distance (km)",
        ylabel="Semivariance",
    )
    if row == 0:
        direction_axis.legend(ncol=2, fontsize=8)

    elevation_band = pd.qcut(
        elevation,
        q=3,
        labels=["low", "middle", "high"],
        duplicates="drop",
    )
    for band in pd.unique(elevation_band):
        membership = np.asarray(elevation_band == band)
        values = []
        for distance_index in range(len(centers)):
            selected = (
                geometry["in_range"]
                & (geometry["distance_bin"] == distance_index)
                & membership[pair_i]
                & membership[pair_j]
            )
            values.append(
                float(np.mean(semivariance[selected]))
                if int(selected.sum()) >= 20
                else np.nan
            )
        local_axis.plot(
            centers,
            values,
            marker="o",
            linewidth=1.2,
            label=str(band),
        )
    local_axis.set(
        title=f"{TARGET_LABELS[target]}: elevation-band variograms",
        xlabel="Distance (km)",
        ylabel="Semivariance",
    )
    if row == 0:
        local_axis.legend(fontsize=8)


def screen_spatial_assumptions(
    data: pd.DataFrame,
    max_train: int = 500,
    n_simulations: int = 199,
    n_restarts: int = 0,
    random_state: int = RANDOM_STATE,
    output_table_path: str | Path | None = None,
    output_figure_path: str | Path | None = None,
) -> tuple[pd.DataFrame, plt.Figure]:
    """Screen baseline assumptions using parametric simulation envelopes.

    The baseline is a linear-elevation mean with a stationary isotropic
    Matérn(1.5) covariance.  The diagnostics are gates for constructing a
    small candidate set; they do not select the final model.
    """
    required = {"x_km", "y_km", "elevation_m"}
    if not required.issubset(data.columns):
        raise ValueError(
            "請先執行 prepare_spatial_folds，使資料包含投影座標。"
        )
    if n_simulations < 19:
        raise ValueError("n_simulations 至少應為 19。")

    candidates = {
        "nonlinear_mean": "加入 nonlinear elevation mean 候選",
        "anisotropy": "加入 geometric anisotropy 候選",
        "variance_nonstationarity": (
            "加入 elevation-dependent variance 候選"
        ),
        "range_nonstationarity": "加入 elevation-dependent range 候選",
        "elevation_distance": "加入 spatial-elevation distance 候選",
    }
    rows: list[dict] = []
    figure, axes = plt.subplots(
        len(TARGETS),
        3,
        figsize=(15.0, 12.0),
        squeeze=False,
    )

    for target_order, (target, column) in enumerate(TARGETS.items()):
        valid_indices = data.index[data[column].notna()].to_numpy()
        fit_indices = sample_indices(
            valid_indices,
            max_train,
            random_state + 20_000 + target_order,
        )
        train = data.loc[fit_indices]
        coordinates = train[["x_km", "y_km"]].to_numpy(float)
        elevation = train["elevation_m"].to_numpy(float)
        response = train[column].to_numpy(float)
        baseline = fit_universal_gp(
            coordinates,
            elevation,
            response,
            trend="T1",
            kernel_name="Matern",
            nu=1.5,
            n_restarts=n_restarts,
            seed=random_state + 21_000 + target_order,
        )
        design = _mean_design(
            elevation,
            "T1",
            baseline["elevation_center"],
            baseline["elevation_scale"],
        )
        trend_residual = response - design @ baseline["beta"]
        geometry = _diagnostic_pair_geometry(coordinates)
        observed = _spatial_screening_statistics(
            trend_residual,
            elevation,
            geometry,
        )
        _plot_screening_diagnostics(
            axes,
            target_order,
            target,
            elevation,
            trend_residual,
            coordinates,
            geometry,
        )

        covariance = baseline["kernel"](baseline["X_train"])
        covariance[np.diag_indices_from(covariance)] += 1e-8
        simulation_cholesky = np.linalg.cholesky(covariance)
        rng = np.random.default_rng(
            random_state + 22_000 + target_order
        )
        simulated_response = (
            design @ baseline["beta"]
        )[:, None] + simulation_cholesky @ rng.standard_normal(
            (len(train), n_simulations)
        )
        inverse_design = cho_solve(
            baseline["chol"],
            design,
            check_finite=False,
        )
        normal = design.T @ inverse_design
        inverse_simulated = cho_solve(
            baseline["chol"],
            simulated_response,
            check_finite=False,
        )
        simulated_beta = np.linalg.solve(
            normal,
            design.T @ inverse_simulated,
        )
        simulated_residual = simulated_response - design @ simulated_beta
        simulated_statistics = {
            diagnostic: []
            for diagnostic in candidates
        }
        for simulation in range(n_simulations):
            statistics = _spatial_screening_statistics(
                simulated_residual[:, simulation],
                elevation,
                geometry,
            )
            for diagnostic, value in statistics.items():
                simulated_statistics[diagnostic].append(value)

        for diagnostic, candidate in candidates.items():
            simulation_values = np.asarray(
                simulated_statistics[diagnostic],
                dtype=float,
            )
            simulation_values = simulation_values[
                np.isfinite(simulation_values)
            ]
            upper = (
                float(np.quantile(simulation_values, 0.95))
                if len(simulation_values)
                else np.nan
            )
            observed_value = float(observed[diagnostic])
            flag = bool(
                np.isfinite(observed_value)
                and np.isfinite(upper)
                and observed_value > upper
            )
            rows.append(
                {
                    "target": target,
                    "baseline": "T1_Matern_1p5",
                    "diagnostic": diagnostic,
                    "observed_statistic": observed_value,
                    "simulation_q95": upper,
                    "flag_candidate": flag,
                    "candidate_action": (
                        candidate if flag else "baseline assumption retained"
                    ),
                    "n_diagnostic_train": len(train),
                    "n_simulations": n_simulations,
                }
            )

    figure.suptitle(
        "Baseline screening: linear elevation mean + stationary isotropic "
        "Matérn(1.5)"
    )
    figure.tight_layout()
    results = pd.DataFrame(rows)
    if output_table_path is not None:
        results.to_csv(
            output_table_path,
            index=False,
            encoding="utf-8-sig",
        )
    if output_figure_path is not None:
        figure.savefig(output_figure_path, bbox_inches="tight")
    return results, figure


def build_model_specs() -> pd.DataFrame:
    """Return the eight trend-by-kernel candidate models."""
    kernel_specs = [
        {"kernel": "RBF", "nu": np.nan},
        {"kernel": "Matern", "nu": 0.5},
        {"kernel": "Matern", "nu": 1.5},
        {"kernel": "Matern", "nu": 2.5},
    ]
    rows: list[dict] = []
    for trend in ("T0", "T1"):
        for kernel_spec in kernel_specs:
            nu = kernel_spec["nu"]
            nu_label = "na" if pd.isna(nu) else str(nu).replace(".", "p")
            rows.append(
                {
                    "model_id": (
                        f"{trend}_{kernel_spec['kernel']}_{nu_label}"
                    ),
                    "trend": trend,
                    "kernel": kernel_spec["kernel"],
                    "nu": nu,
                }
            )
    return pd.DataFrame(rows)


def _center_train_test_km(
    train: np.ndarray,
    test: np.ndarray | None = None,
):
    """Center coordinates while preserving all distances in kilometres."""
    train = np.asarray(train, dtype=float)
    mean = train.mean(axis=0)
    centered_train = train - mean
    if test is None:
        return centered_train, mean
    centered_test = np.asarray(test) - mean
    return centered_train, centered_test, mean


def _mean_design(
    elevation_values: np.ndarray,
    trend: str,
    center: float,
    scale: float,
) -> np.ndarray:
    n = len(elevation_values)
    if trend == "T0":
        return np.ones((n, 1), dtype=float)
    if trend == "T1":
        z = (np.asarray(elevation_values, dtype=float) - center) / scale
        return np.column_stack([np.ones(n), z])
    raise ValueError(f"未知的 mean trend：{trend}")


def _make_covariance(
    kernel_name: str,
    nu: float,
    response_variance: float,
):
    variance = max(float(response_variance), 1e-8)
    amplitude = C(
        constant_value=variance,
        constant_value_bounds=(variance * 1e-4, variance * 1e4),
    )
    if kernel_name == "RBF":
        spatial = RBF(
            length_scale=50.0,
            length_scale_bounds=(1.0, 500.0),
        )
    elif kernel_name == "Matern":
        spatial = Matern(
            length_scale=50.0,
            length_scale_bounds=(1.0, 500.0),
            nu=float(nu),
        )
    else:
        raise ValueError(f"未知的 kernel：{kernel_name}")
    nugget = WhiteKernel(
        noise_level=max(variance * 0.05, 1e-8),
        noise_level_bounds=(max(variance * 1e-8, 1e-12), variance * 2.0),
    )
    return amplitude * spatial + nugget


def fit_universal_gp(
    train_xy: np.ndarray,
    train_elevation: np.ndarray,
    response: np.ndarray,
    trend: str,
    kernel_name: str,
    nu: float = np.nan,
    n_restarts: int = 0,
    seed: int = RANDOM_STATE,
) -> dict:
    """Fit a GP with intercept-only T0 or intercept-plus-elevation T1 mean."""
    train_xy = np.asarray(train_xy, dtype=float)
    train_elevation = np.asarray(train_elevation, dtype=float)
    response = np.asarray(response, dtype=float)
    coordinates, coordinate_origin = _center_train_test_km(train_xy)
    elevation_center = float(train_elevation.mean())
    elevation_scale = float(train_elevation.std())
    if elevation_scale <= 0:
        elevation_scale = 1.0
    design = _mean_design(
        train_elevation,
        trend,
        elevation_center,
        elevation_scale,
    )
    base_kernel = _make_covariance(
        kernel_name,
        nu,
        np.var(response, ddof=1),
    )
    bounds = np.asarray(base_kernel.bounds, dtype=float)
    rng = np.random.default_rng(seed)

    def evaluate(theta, return_components=False):
        kernel = base_kernel.clone_with_theta(theta)
        covariance = kernel(coordinates)
        covariance[np.diag_indices_from(covariance)] += 1e-8
        try:
            chol = cho_factor(covariance, lower=True, check_finite=False)
            inverse_design = cho_solve(chol, design, check_finite=False)
            inverse_response = cho_solve(
                chol,
                response,
                check_finite=False,
            )
            normal = design.T @ inverse_design
            beta = np.linalg.solve(normal, design.T @ inverse_response)
            residual = response - design @ beta
            alpha = cho_solve(chol, residual, check_finite=False)
            log_determinant = 2.0 * np.log(np.diag(chol[0])).sum()
            log_likelihood = -0.5 * (
                residual @ alpha
                + log_determinant
                + len(response) * np.log(2.0 * np.pi)
            )
        except np.linalg.LinAlgError:
            return (np.inf, None) if return_components else np.inf
        components = {
            "kernel": kernel,
            "chol": chol,
            "beta": beta,
            "alpha": alpha,
            "log_likelihood": float(log_likelihood),
        }
        if return_components:
            return -float(log_likelihood), components
        return -float(log_likelihood)

    starts = [base_kernel.theta]
    for _ in range(n_restarts):
        starts.append(rng.uniform(bounds[:, 0], bounds[:, 1]))
    fits = [
        minimize(
            evaluate,
            x0=np.asarray(start),
            method="L-BFGS-B",
            bounds=[tuple(row) for row in bounds],
        )
        for start in starts
    ]
    best = min(fits, key=lambda result: result.fun)
    _, components = evaluate(best.x, return_components=True)
    parameter_count = len(best.x) + design.shape[1]
    sample_size = len(response)
    return {
        **components,
        "trend": trend,
        "kernel_name": kernel_name,
        "nu": nu,
        "X_train": coordinates,
        "xy_origin_km": coordinate_origin,
        "elevation_center": elevation_center,
        "elevation_scale": elevation_scale,
        "n_parameters": parameter_count,
        "AIC": (
            2.0 * parameter_count
            - 2.0 * components["log_likelihood"]
        ),
        "BIC": (
            np.log(sample_size) * parameter_count
            - 2.0 * components["log_likelihood"]
        ),
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
    }


def predict_universal_gp(
    model: dict,
    test_xy: np.ndarray,
    test_elevation: np.ndarray,
    return_std: bool = False,
):
    """Predict from a fitted universal GP."""
    coordinates = (
        np.asarray(test_xy, dtype=float) - model["xy_origin_km"]
    )
    design = _mean_design(
        np.asarray(test_elevation, dtype=float),
        model["trend"],
        model["elevation_center"],
        model["elevation_scale"],
    )
    cross_covariance = model["kernel"](model["X_train"], coordinates)
    prediction = design @ model["beta"] + cross_covariance.T @ model["alpha"]
    if not return_std:
        return prediction
    solved = cho_solve(
        model["chol"],
        cross_covariance,
        check_finite=False,
    )
    variance = (
        model["kernel"].diag(coordinates)
        - np.sum(cross_covariance * solved, axis=0)
    )
    return prediction, np.sqrt(np.maximum(variance, 0.0))


def sample_indices(
    indices: np.ndarray,
    maximum_size: int,
    seed: int,
) -> np.ndarray:
    """Select a reproducible capped training set."""
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= maximum_size:
        return np.sort(indices)
    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(indices, size=maximum_size, replace=False)
    )


def fit_information_criteria(
    data: pd.DataFrame,
    model_specs: pd.DataFrame,
    max_train: int = 800,
    n_restarts: int = 3,
    random_state: int = RANDOM_STATE,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fit all candidate models and calculate AIC, BIC, and Akaike weights."""
    rows: list[dict] = []
    for target_order, (target, column) in enumerate(TARGETS.items()):
        valid_indices = data.index[data[column].notna()].to_numpy()
        fit_indices = sample_indices(
            valid_indices,
            max_train,
            random_state + target_order,
        )
        train = data.loc[fit_indices]
        for model_order, spec in model_specs.iterrows():
            model = fit_universal_gp(
                train[["x_km", "y_km"]].to_numpy(float),
                train["elevation_m"].to_numpy(float),
                train[column].to_numpy(float),
                trend=spec["trend"],
                kernel_name=spec["kernel"],
                nu=spec["nu"],
                n_restarts=n_restarts,
                seed=random_state + target_order * 100 + model_order,
            )
            rows.append(
                {
                    "target": target,
                    "model_id": spec["model_id"],
                    "trend": spec["trend"],
                    "kernel": spec["kernel"],
                    "nu": spec["nu"],
                    "n_train": len(train),
                    "n_parameters": model["n_parameters"],
                    "log_likelihood": model["log_likelihood"],
                    "AIC": model["AIC"],
                    "BIC": model["BIC"],
                    "fitted_kernel": str(model["kernel"]),
                    "optimizer_success": model["optimizer_success"],
                }
            )
    results = pd.DataFrame(rows)
    results["delta_AIC"] = (
        results["AIC"]
        - results.groupby("target")["AIC"].transform("min")
    )
    evidence = np.exp(-0.5 * results["delta_AIC"])
    results["Akaike_weight"] = (
        evidence / evidence.groupby(results["target"]).transform("sum")
    )
    results["AIC_rank"] = results.groupby("target")["AIC"].rank(
        method="min"
    )
    results["BIC_rank"] = results.groupby("target")["BIC"].rank(
        method="min"
    )
    results = results.sort_values(["target", "AIC"]).reset_index(drop=True)
    if output_path is not None:
        results.to_csv(output_path, index=False, encoding="utf-8-sig")
    return results


def _buffered_training_indices(
    data: pd.DataFrame,
    base_indices: np.ndarray,
    test_indices: np.ndarray,
    buffer_km: float,
) -> np.ndarray:
    train_xy = data.loc[
        base_indices,
        ["x_km", "y_km"],
    ].to_numpy(float)
    test_xy = data.loc[
        test_indices,
        ["x_km", "y_km"],
    ].to_numpy(float)
    nearest_test_distance = cdist(train_xy, test_xy).min(axis=1)
    return np.asarray(base_indices)[nearest_test_distance > buffer_km]


def run_buffered_spatial_cv(
    data: pd.DataFrame,
    model_specs: pd.DataFrame,
    buffer_km: dict[str, float],
    n_folds: int = 5,
    max_train: int = 800,
    min_train: int = 100,
    n_restarts: int = 0,
    random_state: int = RANDOM_STATE,
    output_directory: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate all candidate models with fixed buffered geographic folds."""
    base_pool_by_fold = {}
    for fold in range(n_folds):
        candidates = data.index[data["spatial_fold"] != fold].to_numpy()
        base_pool_by_fold[fold] = sample_indices(
            candidates,
            max_train,
            random_state + 10_000 + fold,
        )

    metric_rows: list[dict] = []
    prediction_parts: list[pd.DataFrame] = []
    for target_order, (target, column) in enumerate(TARGETS.items()):
        target_buffer = buffer_km[target]
        for fold in range(n_folds):
            test_indices = data.index[
                data["spatial_fold"] == fold
            ].to_numpy()
            base_indices = base_pool_by_fold[fold]
            train_indices = _buffered_training_indices(
                data,
                base_indices,
                test_indices,
                target_buffer,
            )
            if len(train_indices) < min_train:
                raise ValueError(
                    f"{target}/fold {fold} 套用 {target_buffer:g} km buffer "
                    f"後只剩 {len(train_indices)} 個訓練 GRID。"
                )
            train = data.loc[train_indices]
            test = data.loc[test_indices]
            for model_order, spec in model_specs.iterrows():
                model = fit_universal_gp(
                    train[["x_km", "y_km"]].to_numpy(float),
                    train["elevation_m"].to_numpy(float),
                    train[column].to_numpy(float),
                    trend=spec["trend"],
                    kernel_name=spec["kernel"],
                    nu=spec["nu"],
                    n_restarts=n_restarts,
                    seed=(
                        random_state
                        + target_order * 10_000
                        + fold * 100
                        + model_order
                    ),
                )
                prediction = predict_universal_gp(
                    model,
                    test[["x_km", "y_km"]].to_numpy(float),
                    test["elevation_m"].to_numpy(float),
                )
                truth = test[column].to_numpy(float)
                error = prediction - truth
                metric_rows.append(
                    {
                        "target": target,
                        "model_id": spec["model_id"],
                        "trend": spec["trend"],
                        "kernel": spec["kernel"],
                        "nu": spec["nu"],
                        "fold": fold,
                        "buffer_km": target_buffer,
                        "n_base_train": len(base_indices),
                        "n_retained_train": len(train_indices),
                        "n_test": len(test_indices),
                        "RMSE": float(np.sqrt(np.mean(error**2))),
                        "MAE": float(np.mean(np.abs(error))),
                        "Bias": float(np.mean(error)),
                        "fitted_kernel": str(model["kernel"]),
                    }
                )
                prediction_parts.append(
                    pd.DataFrame(
                        {
                            "target": target,
                            "model_id": spec["model_id"],
                            "trend": spec["trend"],
                            "kernel": spec["kernel"],
                            "nu": spec["nu"],
                            "fold": fold,
                            "row_index": test_indices,
                            "lon": test["lon"].to_numpy(float),
                            "lat": test["lat"].to_numpy(float),
                            "x_km": test["x_km"].to_numpy(float),
                            "y_km": test["y_km"].to_numpy(float),
                            "elevation_m": test[
                                "elevation_m"
                            ].to_numpy(float),
                            "y_true": truth,
                            "y_pred": prediction,
                            "residual": truth - prediction,
                        }
                    )
                )

    fold_metrics = pd.DataFrame(metric_rows)
    oof_predictions = pd.concat(prediction_parts, ignore_index=True)
    summary_rows: list[dict] = []
    for (target, model_id), part in oof_predictions.groupby(
        ["target", "model_id"]
    ):
        fold_part = fold_metrics.query(
            "target == @target and model_id == @model_id"
        )
        error = part["y_pred"].to_numpy() - part["y_true"].to_numpy()
        first = part.iloc[0]
        summary_rows.append(
            {
                "target": target,
                "model_id": model_id,
                "trend": first["trend"],
                "kernel": first["kernel"],
                "nu": first["nu"],
                "RMSE": float(np.sqrt(np.mean(error**2))),
                "MAE": float(np.mean(np.abs(error))),
                "Bias": float(np.mean(error)),
                "fold_RMSE_sd": float(
                    fold_part["RMSE"].std(ddof=1)
                ),
                "mean_retained_train": float(
                    fold_part["n_retained_train"].mean()
                ),
            }
        )
    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["target", "RMSE"])
        .reset_index(drop=True)
    )
    if output_directory is not None:
        output_directory = Path(output_directory)
        fold_metrics.to_csv(
            output_directory / "elevation_gp_spatial_cv_fold_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summary.to_csv(
            output_directory / "elevation_gp_spatial_cv_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        oof_predictions.to_csv(
            output_directory / "elevation_gp_oof_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return fold_metrics, summary, oof_predictions


def select_models(
    cv_summary: pd.DataFrame,
    information_criteria: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select overall and trend-specific CV winners and compare with AIC."""
    best_overall = (
        cv_summary.sort_values(["target", "RMSE"])
        .groupby("target", as_index=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_by_trend = (
        cv_summary.sort_values(["target", "trend", "RMSE"])
        .groupby(
            ["target", "trend"],
            as_index=False,
            group_keys=False,
        )
        .head(1)
        .reset_index(drop=True)
    )
    aic_best = (
        information_criteria.sort_values(["target", "AIC"])
        .groupby("target", as_index=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)[["target", "model_id", "AIC"]]
    )
    comparison = best_overall.merge(
        aic_best,
        on="target",
        how="left",
        suffixes=("_spatial_cv", "_aic"),
    )
    return best_overall, best_by_trend, comparison


def exact_one_sided_sign_flip_test(
    differences: np.ndarray | pd.Series | list[float],
) -> dict[str, float | int]:
    """Test whether the mean paired difference is greater than zero.

    The exact null distribution is obtained by enumerating every sign
    assignment.  This is intended for a small number of paired geographic
    folds, not for treating individual spatially dependent GRID cells as
    independent replicates.
    """
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.ndim != 1 or values.size == 0:
        raise ValueError("At least one finite paired difference is required.")
    if values.size > 20:
        raise ValueError(
            "Exact enumeration is restricted to at most 20 paired units."
        )

    observed_mean = float(np.mean(values))
    permuted_means = np.asarray(
        [
            np.mean(values * np.asarray(signs, dtype=float))
            for signs in product((-1.0, 1.0), repeat=values.size)
        ],
        dtype=float,
    )
    raw_p = float(
        np.mean(permuted_means >= observed_mean - 1e-12)
    )
    return {
        "n_pairs": int(values.size),
        "observed_mean_difference": observed_mean,
        "raw_p": raw_p,
    }


def parameter_wise_elevation_tests(
    fold_metrics: pd.DataFrame,
    best_by_trend: pd.DataFrame,
    alpha: float = 0.05,
    output_directory: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run unadjusted parameter-wise tests of T1 versus T0.

    For every response and geographic fold, the paired difference is

        D_b = MSE(T0, b) - MSE(T1, b).

    Positive values favor the elevation model T1.  The returned p-values are
    deliberately unadjusted because each response is reported as a separate
    parameter-wise exploratory hypothesis.
    """
    required = {
        "target",
        "model_id",
        "trend",
        "fold",
        "RMSE",
    }
    missing = required.difference(fold_metrics.columns)
    if missing:
        raise ValueError(
            f"fold_metrics is missing columns: {sorted(missing)}"
        )

    fold_rows: list[dict] = []
    test_rows: list[dict] = []
    for target in TARGETS:
        selected = best_by_trend.query("target == @target")
        model_ids = dict(zip(selected["trend"], selected["model_id"]))
        if set(model_ids) != {"T0", "T1"}:
            raise ValueError(
                f"{target} must have one selected model for T0 and T1."
            )

        t0 = fold_metrics.query(
            "target == @target and model_id == @model_ids['T0']"
        )[["fold", "RMSE"]].rename(columns={"RMSE": "T0_RMSE"})
        t1 = fold_metrics.query(
            "target == @target and model_id == @model_ids['T1']"
        )[["fold", "RMSE"]].rename(columns={"RMSE": "T1_RMSE"})
        paired = t0.merge(
            t1,
            on="fold",
            how="inner",
            validate="one_to_one",
        ).sort_values("fold")
        if paired.empty:
            raise ValueError(f"No paired fold metrics were found for {target}.")

        paired["T0_MSE"] = paired["T0_RMSE"] ** 2
        paired["T1_MSE"] = paired["T1_RMSE"] ** 2
        paired["D_T0_minus_T1"] = paired["T0_MSE"] - paired["T1_MSE"]
        test = exact_one_sided_sign_flip_test(
            paired["D_T0_minus_T1"]
        )
        folds_favoring_t1 = int(
            (paired["D_T0_minus_T1"] > 0).sum()
        )

        for row in paired.itertuples(index=False):
            fold_rows.append(
                {
                    "target": target,
                    "T0_model_id": model_ids["T0"],
                    "T1_model_id": model_ids["T1"],
                    "fold": int(row.fold),
                    "T0_RMSE": float(row.T0_RMSE),
                    "T1_RMSE": float(row.T1_RMSE),
                    "T0_MSE": float(row.T0_MSE),
                    "T1_MSE": float(row.T1_MSE),
                    "D_T0_minus_T1": float(row.D_T0_minus_T1),
                }
            )
        test_rows.append(
            {
                "target": target,
                "alternative": "T1 has lower fold MSE than T0",
                "T0_model_id": model_ids["T0"],
                "T1_model_id": model_ids["T1"],
                "n_folds": int(test["n_pairs"]),
                "folds_favoring_T1": folds_favoring_t1,
                "mean_D_T0_minus_T1": float(
                    test["observed_mean_difference"]
                ),
                "raw_p": float(test["raw_p"]),
                "alpha": float(alpha),
                "parameter_wise_decision": (
                    "Reject H0"
                    if float(test["raw_p"]) < alpha
                    else "Do not reject H0"
                ),
                "multiplicity_adjustment": "none (parameter-wise)",
            }
        )

    tests = pd.DataFrame(test_rows)
    differences = pd.DataFrame(fold_rows)
    if output_directory is not None:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        tests.to_csv(
            output_directory / "elevation_gp_parameter_wise_tests.csv",
            index=False,
            encoding="utf-8-sig",
        )
        differences.to_csv(
            output_directory
            / "elevation_gp_parameter_wise_fold_differences.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return tests, differences


def plot_cv_comparison(
    cv_summary: pd.DataFrame,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot buffered spatial-CV RMSE for the eight candidate models."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for axis, target in zip(axes, TARGETS):
        part = cv_summary.query("target == @target").copy()
        part["label"] = (
            part["trend"]
            + " "
            + part["kernel"]
            + part["nu"].map(
                lambda value: "" if pd.isna(value) else f" {value:g}"
            )
        )
        part = part.sort_values("RMSE")
        colors = np.where(
            part["trend"].eq("T1"),
            "tab:orange",
            "tab:blue",
        )
        axis.barh(part["label"], part["RMSE"], color=colors)
        axis.invert_yaxis()
        axis.set(title=TARGET_LABELS[target], xlabel="OOF RMSE")
    figure.suptitle(
        "Buffered spatial-CV comparison: baseline T0 versus elevation T1"
    )
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, bbox_inches="tight")
    return figure


def selected_oof(
    oof_predictions: pd.DataFrame,
    best_by_trend: pd.DataFrame,
    target: str,
    trend: str,
) -> pd.DataFrame:
    """Return OOF predictions of the best kernel within a target and trend."""
    model_id = best_by_trend.query(
        "target == @target and trend == @trend"
    ).iloc[0]["model_id"]
    return oof_predictions.query(
        "target == @target and model_id == @model_id"
    ).sort_values("row_index")


def plot_oof_residual_maps(
    oof_predictions: pd.DataFrame,
    best_by_trend: pd.DataFrame,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot OOF residual maps before and after adding elevation."""
    figure, axes = plt.subplots(2, 3, figsize=(14, 9))
    for row, trend in enumerate(("T0", "T1")):
        for column, target in enumerate(TARGETS):
            part = selected_oof(
                oof_predictions,
                best_by_trend,
                target,
                trend,
            )
            limit = np.max(np.abs(part["residual"]))
            points = axes[row, column].scatter(
                part["x_km"],
                part["y_km"],
                c=part["residual"],
                s=12,
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            figure.colorbar(
                points,
                ax=axes[row, column],
                shrink=0.78,
            )
            axes[row, column].set(
                title=f"{trend}: {TARGET_LABELS[target]}",
                xlabel="TWD97 Easting (km)",
                ylabel="TWD97 Northing (km)",
                aspect="equal",
            )
    figure.suptitle(
        "Out-of-fold residual maps before and after adding elevation"
    )
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, bbox_inches="tight")
    return figure


def morans_i_knn(
    xy: np.ndarray,
    values: np.ndarray,
    k: int = 8,
    permutations: int = 999,
    seed: int = RANDOM_STATE,
) -> tuple[float, float, np.ndarray]:
    """Calculate Moran's I and a one-sided permutation p-value."""
    xy = np.asarray(xy, dtype=float)
    values = np.asarray(values, dtype=float)
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    indices = neighbors.kneighbors(return_distance=False)[:, 1:]
    rows = np.repeat(np.arange(len(values)), k)
    columns = indices.reshape(-1)

    def statistic(candidate):
        centered = candidate - candidate.mean()
        numerator = np.sum(centered[rows] * centered[columns]) / k
        denominator = np.sum(centered**2)
        return numerator / denominator

    observed = statistic(values)
    rng = np.random.default_rng(seed)
    permuted = np.array(
        [
            statistic(rng.permutation(values))
            for _ in range(permutations)
        ]
    )
    p_value = (
        1 + np.sum(permuted >= observed)
    ) / (permutations + 1)
    return observed, p_value, permuted


def calculate_residual_morans_i(
    oof_predictions: pd.DataFrame,
    best_by_trend: pd.DataFrame,
    k: int = 8,
    permutations: int = 999,
    seed: int = RANDOM_STATE,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Calculate Moran's I for best T0 and T1 OOF residuals."""
    rows: list[dict] = []
    for target in TARGETS:
        for trend in ("T0", "T1"):
            part = selected_oof(
                oof_predictions,
                best_by_trend,
                target,
                trend,
            )
            observed, p_value, _ = morans_i_knn(
                part[["x_km", "y_km"]].to_numpy(float),
                part["residual"].to_numpy(float),
                k=k,
                permutations=permutations,
                seed=seed,
            )
            rows.append(
                {
                    "target": target,
                    "trend": trend,
                    "Moran_I": observed,
                    "permutation_p": p_value,
                    "k_neighbors": k,
                    "n_permutations": permutations,
                }
            )
    results = pd.DataFrame(rows)
    if output_path is not None:
        results.to_csv(output_path, index=False, encoding="utf-8-sig")
    return results


def empirical_variogram(
    xy: np.ndarray,
    values: np.ndarray,
    n_lags: int = 25,
    maxlag_fraction: float = 0.5,
) -> pd.DataFrame:
    """Calculate a binned empirical semivariogram."""
    distances = pdist(np.asarray(xy, dtype=float))
    semivariance = 0.5 * pdist(
        np.asarray(values, dtype=float).reshape(-1, 1),
        metric="sqeuclidean",
    )
    max_lag = distances.max() * maxlag_fraction
    edges = np.linspace(0.0, max_lag, n_lags + 1)
    bin_id = np.digitize(distances, edges) - 1
    rows: list[dict] = []
    for lag in range(n_lags):
        mask = bin_id == lag
        if mask.any():
            rows.append(
                {
                    "lag_km": float(distances[mask].mean()),
                    "semivariance": float(
                        semivariance[mask].mean()
                    ),
                    "n_pairs": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def calculate_residual_variograms(
    oof_predictions: pd.DataFrame,
    best_by_trend: pd.DataFrame,
    output_table_path: str | Path | None = None,
    output_figure_path: str | Path | None = None,
) -> tuple[pd.DataFrame, plt.Figure]:
    """Compare T0 and T1 empirical OOF residual variograms."""
    parts: list[pd.DataFrame] = []
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for axis, target in zip(axes, TARGETS):
        for trend, color in (
            ("T0", "tab:blue"),
            ("T1", "tab:orange"),
        ):
            selected = selected_oof(
                oof_predictions,
                best_by_trend,
                target,
                trend,
            )
            empirical = empirical_variogram(
                selected[["x_km", "y_km"]].to_numpy(float),
                selected["residual"].to_numpy(float),
            )
            empirical["target"] = target
            empirical["trend"] = trend
            parts.append(empirical)
            axis.plot(
                empirical["lag_km"],
                empirical["semivariance"],
                marker="o",
                markersize=4,
                label=trend,
                color=color,
            )
        axis.set(
            title=TARGET_LABELS[target],
            xlabel="Distance (km)",
            ylabel="Residual semivariance",
        )
        axis.legend()
    results = pd.concat(parts, ignore_index=True)
    figure.suptitle(
        "Residual variograms: coordinate-only T0 versus elevation T1"
    )
    figure.tight_layout()
    if output_table_path is not None:
        results.to_csv(
            output_table_path,
            index=False,
            encoding="utf-8-sig",
        )
    if output_figure_path is not None:
        figure.savefig(output_figure_path, bbox_inches="tight")
    return results, figure


def gev_return_level(
    mu: np.ndarray,
    log_sigma: np.ndarray,
    xi: np.ndarray,
    return_period: int,
) -> np.ndarray:
    """Calculate a GEV return level with a stable Gumbel limit."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.exp(np.asarray(log_sigma, dtype=float))
    xi = np.asarray(xi, dtype=float)
    probability = 1.0 - 1.0 / float(return_period)
    transformed_probability = -np.log(probability)
    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
    ):
        nonzero = (
            mu
            + sigma
            * np.expm1(-xi * np.log(transformed_probability))
            / xi
        )
        gumbel = mu - sigma * np.log(transformed_probability)
    return np.where(np.abs(xi) < 1e-6, gumbel, nonzero)


def _parameter_oof_wide(
    oof_predictions: pd.DataFrame,
    best_by_trend: pd.DataFrame,
    trend: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for target in TARGETS:
        part = selected_oof(
            oof_predictions,
            best_by_trend,
            target,
            trend,
        )[["row_index", "fold", "y_true", "y_pred"]].copy()
        parts.append(
            part.rename(
                columns={
                    "y_true": f"{target}_true",
                    "y_pred": f"{target}_pred",
                }
            )
        )
    wide = parts[0]
    for part in parts[1:]:
        wide = wide.merge(
            part,
            on=["row_index", "fold"],
            how="inner",
            validate="one_to_one",
        )
    return wide


def compare_return_levels(
    data: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    best_by_trend: pd.DataFrame,
    return_periods: tuple[int, ...] = (50, 100),
    output_directory: str | Path | None = None,
    output_figure_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:
    """Compare T0 and T1 OOF return levels derived from all three parameters."""
    metric_rows: list[dict] = []
    prediction_parts: list[pd.DataFrame] = []
    for trend in ("T0", "T1"):
        wide = _parameter_oof_wide(
            oof_predictions,
            best_by_trend,
            trend,
        )
        for return_period in return_periods:
            reference = gev_return_level(
                wide["mu_true"],
                wide["log_sigma_true"],
                wide["xi_true"],
                return_period,
            )
            prediction = gev_return_level(
                wide["mu_pred"],
                wide["log_sigma_pred"],
                wide["xi_pred"],
                return_period,
            )
            error = prediction - reference
            metric_rows.append(
                {
                    "trend": trend,
                    "return_period": return_period,
                    "RMSE_vs_NN_reference": float(
                        np.sqrt(np.mean(error**2))
                    ),
                    "MAE_vs_NN_reference": float(
                        np.mean(np.abs(error))
                    ),
                    "Bias_vs_NN_reference": float(np.mean(error)),
                    "finite_rate": float(
                        np.mean(np.isfinite(prediction))
                    ),
                }
            )
            prediction_parts.append(
                pd.DataFrame(
                    {
                        "row_index": wide["row_index"],
                        "fold": wide["fold"],
                        "trend": trend,
                        "return_period": return_period,
                        "reference_RL": reference,
                        "predicted_RL": prediction,
                        "error": error,
                    }
                )
            )

    metrics = pd.DataFrame(metric_rows)
    oof_return_levels = pd.concat(
        prediction_parts,
        ignore_index=True,
    )
    figure, axes = plt.subplots(
        len(return_periods),
        3,
        figsize=(14, 4.25 * len(return_periods)),
        squeeze=False,
    )
    for row, return_period in enumerate(return_periods):
        t0 = oof_return_levels.query(
            "trend == 'T0' and return_period == @return_period"
        ).sort_values("row_index")
        t1 = oof_return_levels.query(
            "trend == 'T1' and return_period == @return_period"
        ).sort_values("row_index")
        coordinates = data.loc[t0["row_index"], ["x_km", "y_km"]]
        panels = [
            (
                t0["predicted_RL"],
                f"T0 predicted RL{return_period}",
            ),
            (
                t1["predicted_RL"],
                f"T1 predicted RL{return_period}",
            ),
            (
                (
                    t1["predicted_RL"].to_numpy()
                    - t0["predicted_RL"].to_numpy()
                ),
                f"T1 - T0 RL{return_period}",
            ),
        ]
        for column, (values, title) in enumerate(panels):
            points = axes[row, column].scatter(
                coordinates["x_km"],
                coordinates["y_km"],
                c=values,
                s=12,
                cmap="coolwarm" if column == 2 else "turbo",
            )
            figure.colorbar(
                points,
                ax=axes[row, column],
                shrink=0.78,
            )
            axes[row, column].set(
                title=title,
                xlabel="TWD97 Easting (km)",
                ylabel="TWD97 Northing (km)",
                aspect="equal",
            )
    figure.suptitle("Out-of-fold return-level pipeline comparison")
    figure.tight_layout()
    if output_directory is not None:
        output_directory = Path(output_directory)
        metrics.to_csv(
            output_directory / "elevation_gp_return_level_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        oof_return_levels.to_csv(
            output_directory
            / "elevation_gp_return_level_oof_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if output_figure_path is not None:
        figure.savefig(output_figure_path, bbox_inches="tight")
    return metrics, oof_return_levels, figure


def return_level_pipeline_tests(
    oof_return_levels: pd.DataFrame,
    baseline_trend: str = "T0",
    candidate_trend: str = "T1",
    alpha: float = 0.05,
    output_directory: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run unadjusted return-period-wise tests of candidate versus baseline.

    The current notebook uses this function for the preliminary all-T1 versus
    all-T0 comparison.  It can also be reused after final mixed-pipeline OOF
    predictions are available by assigning an appropriate candidate label.

    For every return period and geographic fold,

        D_b = MSE(baseline, b) - MSE(candidate, b).

    Positive differences favor the candidate pipeline.
    """
    required = {
        "row_index",
        "fold",
        "trend",
        "return_period",
        "reference_RL",
        "predicted_RL",
        "error",
    }
    missing = required.difference(oof_return_levels.columns)
    if missing:
        raise ValueError(
            "oof_return_levels is missing columns: "
            f"{sorted(missing)}"
        )

    fold_rows: list[dict] = []
    test_rows: list[dict] = []
    periods = sorted(oof_return_levels["return_period"].unique())
    for return_period in periods:
        baseline = oof_return_levels.query(
            "trend == @baseline_trend "
            "and return_period == @return_period"
        )[
            ["row_index", "fold", "reference_RL", "error"]
        ].rename(
            columns={
                "reference_RL": "baseline_reference_RL",
                "error": "baseline_error",
            }
        )
        candidate = oof_return_levels.query(
            "trend == @candidate_trend "
            "and return_period == @return_period"
        )[
            ["row_index", "fold", "reference_RL", "error"]
        ].rename(
            columns={
                "reference_RL": "candidate_reference_RL",
                "error": "candidate_error",
            }
        )
        paired = baseline.merge(
            candidate,
            on=["row_index", "fold"],
            how="inner",
            validate="one_to_one",
        )
        if paired.empty:
            raise ValueError(
                f"No paired RL{int(return_period)} predictions were found."
            )
        if not np.allclose(
            paired["baseline_reference_RL"],
            paired["candidate_reference_RL"],
            equal_nan=True,
        ):
            raise ValueError(
                f"RL{int(return_period)} reference values do not match."
            )

        period_differences: list[float] = []
        for fold, part in paired.groupby("fold", sort=True):
            baseline_mse = float(np.mean(part["baseline_error"] ** 2))
            candidate_mse = float(np.mean(part["candidate_error"] ** 2))
            difference = baseline_mse - candidate_mse
            period_differences.append(difference)
            fold_rows.append(
                {
                    "return_period": int(return_period),
                    "baseline_trend": baseline_trend,
                    "candidate_trend": candidate_trend,
                    "fold": int(fold),
                    "n_test": int(len(part)),
                    "baseline_RMSE": float(np.sqrt(baseline_mse)),
                    "candidate_RMSE": float(np.sqrt(candidate_mse)),
                    "baseline_MSE": baseline_mse,
                    "candidate_MSE": candidate_mse,
                    "D_baseline_minus_candidate": difference,
                }
            )

        test = exact_one_sided_sign_flip_test(period_differences)
        favorable = int(np.sum(np.asarray(period_differences) > 0))
        test_rows.append(
            {
                "return_period": int(return_period),
                "alternative": (
                    f"{candidate_trend} has lower fold MSE than "
                    f"{baseline_trend}"
                ),
                "baseline_trend": baseline_trend,
                "candidate_trend": candidate_trend,
                "n_folds": int(test["n_pairs"]),
                "folds_favoring_candidate": favorable,
                "mean_D_baseline_minus_candidate": float(
                    test["observed_mean_difference"]
                ),
                "raw_p": float(test["raw_p"]),
                "alpha": float(alpha),
                "return_period_wise_decision": (
                    "Reject H0"
                    if float(test["raw_p"]) < alpha
                    else "Do not reject H0"
                ),
                "multiplicity_adjustment": "none (return-period-wise)",
                "scope": "preliminary all-T1 versus all-T0",
            }
        )

    tests = pd.DataFrame(test_rows)
    differences = pd.DataFrame(fold_rows)
    if output_directory is not None:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        tests.to_csv(
            output_directory / "elevation_gp_return_level_tests.csv",
            index=False,
            encoding="utf-8-sig",
        )
        differences.to_csv(
            output_directory
            / "elevation_gp_return_level_fold_differences.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return tests, differences
