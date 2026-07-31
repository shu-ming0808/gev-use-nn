"""Directional and regional diagnostics for raw surfaces and OOF residuals.

The directional variograms are descriptive isotropy diagnostics.  Regional
variograms and regional mean/variance summaries are descriptive stationarity
diagnostics.  They do not by themselves constitute a formal hypothesis test;
their role is to decide which covariance extensions should enter a later
buffered spatial-CV comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIRECTIONS = (0.0, 45.0, 90.0, 135.0)
DIRECTION_LABELS = {
    0.0: "E--W (0 deg)",
    45.0: "NE--SW (45 deg)",
    90.0: "N--S (90 deg)",
    135.0: "NW--SE (135 deg)",
}
TARGET_LABELS = {
    "mu": r"$\mu$",
    "log_sigma": r"$\log\sigma$",
    "xi": r"$\xi$",
}


def _angular_distance_degrees(
    angles: np.ndarray,
    direction: float,
) -> np.ndarray:
    """Axial angular distance, treating 0 and 180 degrees as identical."""
    return np.abs((angles - direction + 90.0) % 180.0 - 90.0)


def _pair_geometry(
    coordinates: np.ndarray,
    n_bins: int = 8,
    max_distance_quantile: float = 0.60,
) -> dict[str, np.ndarray]:
    """Precompute pair distances, axial directions, and distance bins."""
    coordinates = np.asarray(coordinates, dtype=float)
    pair_i, pair_j = np.triu_indices(len(coordinates), k=1)
    difference = coordinates[pair_j] - coordinates[pair_i]
    distance = np.sqrt(np.sum(difference**2, axis=1))
    angle = np.degrees(
        np.arctan2(difference[:, 1], difference[:, 0])
    ) % 180.0
    positive = distance > 0.0
    pair_i = pair_i[positive]
    pair_j = pair_j[positive]
    distance = distance[positive]
    angle = angle[positive]
    maximum = float(np.quantile(distance, max_distance_quantile))
    edges = np.linspace(0.0, maximum, n_bins + 1)
    bin_index = np.digitize(distance, edges, right=False) - 1
    in_range = (bin_index >= 0) & (bin_index < n_bins)
    return {
        "i": pair_i,
        "j": pair_j,
        "distance": distance,
        "angle": angle,
        "edges": edges,
        "centers": 0.5 * (edges[:-1] + edges[1:]),
        "bin": bin_index,
        "in_range": in_range,
    }


def _directional_variograms(
    values: np.ndarray,
    geometry: dict[str, np.ndarray],
    minimum_pairs: int = 20,
) -> pd.DataFrame:
    """Calculate variance-standardized directional semivariograms."""
    values = np.asarray(values, dtype=float)
    variance = max(float(np.var(values, ddof=1)), 1e-12)
    semivariance = 0.5 * (
        values[geometry["i"]] - values[geometry["j"]]
    ) ** 2
    rows = []
    for direction in DIRECTIONS:
        direction_mask = (
            _angular_distance_degrees(geometry["angle"], direction)
            <= 22.5
        )
        for bin_index, center in enumerate(geometry["centers"]):
            selected = (
                geometry["in_range"]
                & direction_mask
                & (geometry["bin"] == bin_index)
            )
            rows.append(
                {
                    "direction_deg": direction,
                    "distance_km": float(center),
                    "normalized_semivariance": (
                        float(np.mean(semivariance[selected]) / variance)
                        if int(selected.sum()) >= minimum_pairs
                        else np.nan
                    ),
                    "n_pairs": int(selected.sum()),
                }
            )
    return pd.DataFrame(rows)


def _regional_variograms(
    values: np.ndarray,
    region: np.ndarray,
    geometry: dict[str, np.ndarray],
    minimum_pairs: int = 20,
) -> pd.DataFrame:
    """Calculate within-region standardized semivariograms."""
    values = np.asarray(values, dtype=float)
    region = np.asarray(region)
    semivariance = 0.5 * (
        values[geometry["i"]] - values[geometry["j"]]
    ) ** 2
    rows = []
    for label in pd.unique(region):
        membership = region == label
        region_variance = max(
            float(np.var(values[membership], ddof=1)),
            1e-12,
        )
        for bin_index, center in enumerate(geometry["centers"]):
            selected = (
                geometry["in_range"]
                & (geometry["bin"] == bin_index)
                & membership[geometry["i"]]
                & membership[geometry["j"]]
            )
            rows.append(
                {
                    "region": str(label),
                    "distance_km": float(center),
                    "normalized_semivariance": (
                        float(
                            np.mean(semivariance[selected])
                            / region_variance
                        )
                        if int(selected.sum()) >= minimum_pairs
                        else np.nan
                    ),
                    "n_pairs": int(selected.sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_isotropy_stationarity_diagnostics(
    data: pd.DataFrame,
    value_columns: dict[str, str],
    value_kind: str,
    output_figure_path: str | Path | None = None,
    output_table_path: str | Path | None = None,
    n_bins: int = 8,
) -> tuple[pd.DataFrame, plt.Figure]:
    """Plot directional and north/central/south regional variograms.

    ``value_columns`` maps a response label such as ``mu`` to a column in
    ``data``.  Latitude bands are based on thirds of projected northing so
    each region contains observations; they are diagnostic strata rather than
    administrative boundaries.
    """
    required = {"x_km", "y_km", *value_columns.values()}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Spatial diagnostics missing columns: {sorted(missing)}")

    clean = data.dropna(subset=list(required)).copy()
    clean["region"] = pd.qcut(
        clean["y_km"],
        q=3,
        labels=["south", "central", "north"],
        duplicates="drop",
    ).astype(str)
    coordinates = clean[["x_km", "y_km"]].to_numpy(float)
    geometry = _pair_geometry(coordinates, n_bins=n_bins)
    figure, axes = plt.subplots(
        len(value_columns),
        2,
        figsize=(13.0, 3.7 * len(value_columns)),
        squeeze=False,
    )
    summary_rows = []

    for row, (target, column) in enumerate(value_columns.items()):
        values = clean[column].to_numpy(float)
        directional = _directional_variograms(values, geometry)
        regional = _regional_variograms(
            values,
            clean["region"].to_numpy(),
            geometry,
        )
        for direction in DIRECTIONS:
            part = directional.query("direction_deg == @direction")
            axes[row, 0].plot(
                part["distance_km"],
                part["normalized_semivariance"],
                marker="o",
                linewidth=1.3,
                label=DIRECTION_LABELS[direction],
            )
        for region in ("south", "central", "north"):
            part = regional.query("region == @region")
            axes[row, 1].plot(
                part["distance_km"],
                part["normalized_semivariance"],
                marker="o",
                linewidth=1.3,
                label=region,
            )
        label = TARGET_LABELS.get(target, target)
        axes[row, 0].set(
            title=f"{label}: directional variograms",
            xlabel="distance (km)",
            ylabel="normalized semivariance",
        )
        axes[row, 1].set(
            title=f"{label}: regional variograms",
            xlabel="distance (km)",
            ylabel="within-region normalized semivariance",
        )
        axes[row, 0].legend(fontsize=8, ncol=2)
        axes[row, 1].legend(fontsize=8)

        directional_wide = directional.pivot(
            index="distance_km",
            columns="direction_deg",
            values="normalized_semivariance",
        )
        regional_wide = regional.pivot(
            index="distance_km",
            columns="region",
            values="normalized_semivariance",
        )
        regional_groups = clean.groupby("region", observed=True)[column]
        regional_means = regional_groups.mean()
        regional_variances = regional_groups.var(ddof=1)
        positive_variances = regional_variances[regional_variances > 0]
        global_sd = max(float(np.std(values, ddof=1)), 1e-12)
        summary_rows.append(
            {
                "value_kind": value_kind,
                "target": target,
                "n": len(values),
                "directional_variogram_contrast": float(
                    directional_wide.max(axis=1).sub(
                        directional_wide.min(axis=1)
                    ).median()
                ),
                "regional_mean_range_in_sd": float(
                    (regional_means.max() - regional_means.min())
                    / global_sd
                ),
                "regional_log_variance_ratio": (
                    float(
                        np.log(
                            positive_variances.max()
                            / positive_variances.min()
                        )
                    )
                    if len(positive_variances) >= 2
                    else np.nan
                ),
                "regional_variogram_contrast": float(
                    regional_wide.max(axis=1).sub(
                        regional_wide.min(axis=1)
                    ).median()
                ),
            }
        )

    figure.suptitle(
        f"Isotropy and stationarity diagnostics: {value_kind}",
        y=1.0,
    )
    figure.tight_layout()
    summary = pd.DataFrame(summary_rows)
    if output_figure_path is not None:
        output_figure_path = Path(output_figure_path)
        output_figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_figure_path, bbox_inches="tight", dpi=170)
    if output_table_path is not None:
        output_table_path = Path(output_table_path)
        output_table_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_table_path, index=False, encoding="utf-8-sig")
    return summary, figure


def prepare_selected_oof_residuals(
    predictions: pd.DataFrame,
    locations: pd.DataFrame,
) -> pd.DataFrame:
    """Convert long selected-model OOF residuals to one spatial table."""
    required_predictions = {"station", "target", "residual"}
    if not required_predictions.issubset(predictions.columns):
        raise ValueError("OOF predictions need station, target, and residual.")
    required_locations = {"station", "x_km", "y_km"}
    if not required_locations.issubset(locations.columns):
        raise ValueError("Locations need station, x_km, and y_km.")
    wide = predictions.pivot(
        index="station",
        columns="target",
        values="residual",
    ).rename(
        columns={
            "mu": "mu_residual",
            "log_sigma": "log_sigma_residual",
            "xi": "xi_residual",
        }
    )
    result = locations[["station", "x_km", "y_km"]].merge(
        wide.reset_index(),
        on="station",
        how="inner",
        validate="one_to_one",
    )
    expected = {
        "mu_residual",
        "log_sigma_residual",
        "xi_residual",
    }
    if not expected.issubset(result.columns):
        raise ValueError("OOF predictions do not contain all three targets.")
    return result
