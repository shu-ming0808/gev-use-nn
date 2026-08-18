"""Re-evaluate four predictor stages with one common buffered spatial-CV design.

The comparison deliberately fixes the same 1,385 main-island GRID cells,
coordinate K-means folds, target-specific buffers, and capped training samples.
It fits only the missing fixed stage models; the latest spatial-FFS OOF
predictions are read from disk and are not refitted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from elevation_gp_analysis import RANDOM_STATE, TARGETS, empirical_variogram
from project_paths import TABLE_DIR
from spatial_predictor_selection import (
    _fold_contexts,
    evaluate_predictor_set,
    load_predictor_selection_data,
)


TARGET_ORDER = ("mu", "log_sigma", "xi")
TARGET_LABELS = {
    "mu": r"$\hat{\mu}$",
    "log_sigma": r"$\widehat{\log\sigma}$",
    "xi": r"$\hat{\xi}$",
}
STAGE_ORDER = (
    "No predictors",
    "Elevation only",
    "Previous selection",
    "Current selection",
)

# Previous selection means the final pre-atmospheric model reported in the
# preceding analysis, not an outcome selected again from the current results.
PREVIOUS_MODELS = {
    "mu": {
        "predictors": ["elevation_m", "local_relief_m", "agriculture_ratio"],
        "kernel": "RBF",
        "nu": None,
    },
    "log_sigma": {
        "predictors": ["elevation_m", "water_ratio"],
        "kernel": "Matern",
        "nu": 0.5,
    },
    "xi": {"predictors": [], "kernel": "RBF", "nu": None},
}


def _kernel_from_path(path: pd.DataFrame, target: str, step: int) -> dict:
    row = path.loc[path["target"].eq(target) & path["step"].eq(step)]
    if row.empty:
        raise ValueError(f"Selection path has no {target}/step {step}.")
    row = row.iloc[0]
    predictors = [] if row["predictors"] == "intercept" else str(row["predictors"]).split("+")
    return {
        "predictors": predictors,
        "kernel": str(row["kernel"]),
        "nu": None if pd.isna(row["nu"]) else float(row["nu"]),
    }


def _evaluate_stage(
    data: pd.DataFrame,
    contexts: dict[str, list[dict]],
    target: str,
    stage: str,
    specification: dict,
    model_order: int,
) -> pd.DataFrame:
    result = evaluate_predictor_set(
        data=data,
        target=target,
        predictor_names=specification["predictors"],
        contexts=contexts[target],
        kernel_name=specification["kernel"],
        nu=specification["nu"],
        n_restarts=0,
        random_state=RANDOM_STATE,
        model_order=model_order,
    )
    predictions = result["predictions"].copy()
    predictions["stage"] = stage
    predictions["predictors"] = (
        "+".join(specification["predictors"])
        if specification["predictors"]
        else "intercept"
    )
    return predictions


def build_four_stage_oof(
    selected_prediction_path: Path,
    selection_path_path: Path,
    n_jobs: int,
) -> pd.DataFrame:
    data = load_predictor_selection_data()
    current = pd.read_csv(selected_prediction_path).copy()
    fold_map = current[["station", "fold"]].drop_duplicates()
    if fold_map["station"].duplicated().any():
        raise ValueError("Latest selected OOF table assigns a station to multiple folds.")
    data = data.merge(
        fold_map.rename(columns={"fold": "spatial_fold"}),
        on="station",
        how="left",
        validate="one_to_one",
    )
    if data["spatial_fold"].isna().any():
        raise ValueError("Latest selected OOF fold assignments do not cover all GRID cells.")
    data["spatial_fold"] = data["spatial_fold"].astype(int)
    contexts = {
        target: _fold_contexts(
            data,
            target=target,
            n_folds=5,
            max_train=800,
            min_train=100,
            random_state=RANDOM_STATE,
        )
        for target in TARGET_ORDER
    }
    path = pd.read_csv(selection_path_path)

    jobs = []
    reused = []
    model_order = 10_000
    for target in TARGET_ORDER:
        target_path = path.loc[path["target"].eq(target)]
        current_spec = _kernel_from_path(path, target, int(target_path["step"].max()))
        stage_specs = (
            ("No predictors", _kernel_from_path(path, target, 0)),
            ("Elevation only", _kernel_from_path(path, target, 1)),
            ("Previous selection", PREVIOUS_MODELS[target]),
        )
        for stage, specification in stage_specs:
            same_as_current = (
                specification["predictors"] == current_spec["predictors"]
                and specification["kernel"] == current_spec["kernel"]
                and specification["nu"] == current_spec["nu"]
            )
            if same_as_current:
                part = current.loc[current["target"].eq(target)].copy()
                part["stage"] = stage
                part["predictors"] = (
                    "+".join(specification["predictors"])
                    if specification["predictors"]
                    else "intercept"
                )
                reused.append(part)
            else:
                jobs.append((target, stage, specification, model_order))
                model_order += 1

    fitted = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_evaluate_stage)(data, contexts, target, stage, spec, order)
        for target, stage, spec, order in jobs
    )

    current["stage"] = "Current selection"
    current["predictors"] = current["selected_predictors"]
    keep = [
        "target", "fold", "kernel", "nu", "row_index", "station",
        "y_true", "y_pred", "residual", "stage", "predictors",
    ]
    comparison = pd.concat(
        [*fitted, *(part[keep] for part in reused), current[keep]],
        ignore_index=True,
    )
    locations = data[["station", "x_km", "y_km"]]
    comparison = comparison.merge(
        locations,
        on="station",
        how="left",
        validate="many_to_one",
    )
    comparison["residual"] = comparison["y_true"] - comparison["y_pred"]
    comparison["target"] = pd.Categorical(
        comparison["target"], TARGET_ORDER, ordered=True
    )
    comparison["stage"] = pd.Categorical(
        comparison["stage"], STAGE_ORDER, ordered=True
    )
    return comparison.sort_values(["stage", "target", "row_index"]).reset_index(drop=True)


def calculate_metrics(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (stage, target), part in comparison.groupby(
        ["stage", "target"], observed=True, sort=False
    ):
        error = part["y_pred"].to_numpy(float) - part["y_true"].to_numpy(float)
        rows.append(
            {
                "stage": stage,
                "target": target,
                "predictors": part["predictors"].iloc[0],
                "kernel": part["kernel"].iloc[0],
                "nu": part["nu"].iloc[0],
                "n": len(part),
                "RMSE": float(np.sqrt(np.mean(error**2))),
                "MAE": float(np.mean(np.abs(error))),
                "Bias_prediction_minus_reference": float(np.mean(error)),
            }
        )
    result = pd.DataFrame(rows)
    result["stage"] = pd.Categorical(result["stage"], STAGE_ORDER, ordered=True)
    result["target"] = pd.Categorical(result["target"], TARGET_ORDER, ordered=True)
    return result.sort_values(["stage", "target"]).reset_index(drop=True)


def _map_grid(
    comparison: pd.DataFrame,
    value: str,
    residual: bool,
    stages: tuple[str, ...] = STAGE_ORDER,
) -> plt.Figure:
    figure, axes = plt.subplots(
        len(stages), 3,
        figsize=(12.8, 3.55 * len(stages) + 0.4),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    limits = {}
    for target in TARGET_ORDER:
        values = comparison.loc[comparison["target"].eq(target), value].to_numpy(float)
        if residual:
            bound = float(np.nanquantile(np.abs(values), 0.995))
            limits[target] = (-bound, bound)
        else:
            limits[target] = (float(np.nanmin(values)), float(np.nanmax(values)))

    for row, stage in enumerate(stages):
        for column, target in enumerate(TARGET_ORDER):
            axis = axes[row, column]
            part = comparison.loc[
                comparison["stage"].eq(stage) & comparison["target"].eq(target)
            ]
            lower, upper = limits[target]
            points = axis.scatter(
                part["x_km"], part["y_km"], c=part[value], marker="s", s=8,
                linewidths=0, cmap="coolwarm" if residual else "viridis",
                vmin=lower, vmax=upper,
            )
            if row == 0:
                axis.set_title(TARGET_LABELS[target], fontsize=12)
            if column == 0:
                axis.set_ylabel(f"{stage}\nNorthing (km)", fontsize=9.5)
            if row == len(stages) - 1:
                axis.set_xlabel("Easting (km)")
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.12, linewidth=0.4)
            if row == 0:
                figure.colorbar(points, ax=axes[:, column], shrink=0.72, pad=0.012)
    figure.suptitle(
        f"Geographically out-of-fold residual maps across {len(stages)} predictor stages"
        if residual
        else f"Geographically out-of-fold predictions across {len(stages)} predictor stages",
        fontsize=15,
    )
    if residual:
        figure.text(
            0.5, 0.001,
            "Residual = NN-derived reference - OOF GP prediction; each parameter uses a common clipped colour scale.",
            ha="center", fontsize=8.5,
        )
    return figure


def plot_rmse_trends(metrics: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    x = np.arange(len(STAGE_ORDER))
    short_labels = ["None", "Elevation", "Previous", "Current"]
    for axis, target in zip(axes, TARGET_ORDER):
        part = metrics.loc[metrics["target"].eq(target)].sort_values("stage")
        values = part["RMSE"].to_numpy(float)
        axis.plot(x, values, color="#3366AA", marker="o", linewidth=2.0, markersize=6)
        for index, value in enumerate(values):
            axis.annotate(
                f"{value:.4f}" if target == "xi" else f"{value:.3f}",
                (index, value), xytext=(0, 7),
                textcoords="offset points", ha="center", fontsize=8.5,
            )
        best_mask = np.isclose(values, values.min(), rtol=0.0, atol=1e-12)
        axis.scatter(
            x[best_mask], values[best_mask], color="#D62728", s=55,
            zorder=3, label="Lowest RMSE",
        )
        axis.set_xticks(x, short_labels, rotation=18, ha="right")
        axis.set_title(TARGET_LABELS[target], fontsize=12)
        axis.set_ylabel("Pooled OOF RMSE")
        axis.grid(axis="y", alpha=0.25)
        axis.margins(y=0.18)
    axes[0].legend(fontsize=8)
    figure.suptitle("OOF RMSE across predictor-selection stages", fontsize=14)
    return figure


def calculate_residual_variograms(comparison: pd.DataFrame) -> pd.DataFrame:
    """Calculate identically binned empirical OOF residual variograms."""
    rows: list[pd.DataFrame] = []
    for stage in STAGE_ORDER:
        for target in TARGET_ORDER:
            part = comparison.loc[
                comparison["stage"].eq(stage)
                & comparison["target"].eq(target)
            ].sort_values("station")
            variogram = empirical_variogram(
                part[["x_km", "y_km"]].to_numpy(float),
                part["residual"].to_numpy(float),
                n_lags=25,
                maxlag_fraction=0.5,
            )
            variogram["stage"] = stage
            variogram["target"] = target
            rows.append(variogram)
    result = pd.concat(rows, ignore_index=True)
    result["stage"] = pd.Categorical(result["stage"], STAGE_ORDER, ordered=True)
    result["target"] = pd.Categorical(result["target"], TARGET_ORDER, ordered=True)
    return result.sort_values(["stage", "target", "lag_km"]).reset_index(drop=True)


def plot_residual_variogram_grid(variograms: pd.DataFrame) -> plt.Figure:
    """Plot four predictor stages by three GEV responses."""
    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756")
    figure, axes = plt.subplots(
        4, 3, figsize=(13.2, 12.4),
        sharex=True, sharey="col",
    )
    figure.subplots_adjust(
        left=0.09, right=0.985, top=0.94, bottom=0.075,
        hspace=0.13, wspace=0.14,
    )
    for row, (stage, color) in enumerate(zip(STAGE_ORDER, colors)):
        for column, target in enumerate(TARGET_ORDER):
            axis = axes[row, column]
            part = variograms.loc[
                variograms["stage"].eq(stage)
                & variograms["target"].eq(target)
            ]
            axis.plot(
                part["lag_km"], part["semivariance"],
                color=color, marker="o", markersize=3.2, linewidth=1.6,
            )
            axis.fill_between(
                part["lag_km"], 0.0, part["semivariance"],
                color=color, alpha=0.08,
            )
            if row == 0:
                axis.set_title(TARGET_LABELS[target], fontsize=12)
            if column == 0:
                axis.set_ylabel(f"{stage}\nSemivariance", fontsize=9.5)
            if row == len(STAGE_ORDER) - 1:
                axis.set_xlabel("Distance (km)")
            axis.set_ylim(bottom=0.0)
            axis.grid(alpha=0.22, linewidth=0.6)
    figure.suptitle(
        "Out-of-fold residual variograms across four predictor stages",
        fontsize=15, y=0.982,
    )
    figure.text(
        0.5, 0.018,
        "Residual = NN-derived reference - OOF GP prediction; "
        "all panels use identical lag-bin settings and each parameter column shares a y-axis.",
        ha="center", fontsize=8.5,
    )
    return figure


def plot_residual_variogram_overlay(variograms: pd.DataFrame) -> plt.Figure:
    """Overlay all four predictor stages within each GEV-response panel."""
    colors = {
        "No predictors": "#4C78A8",
        "Elevation only": "#F58518",
        "Previous selection": "#54A24B",
        "Current selection": "#E45756",
    }
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.35), constrained_layout=True)
    for axis, target in zip(axes, TARGET_ORDER):
        for stage in STAGE_ORDER:
            part = variograms.loc[
                variograms["stage"].eq(stage)
                & variograms["target"].eq(target)
            ]
            axis.plot(
                part["lag_km"], part["semivariance"],
                color=colors[stage], marker="o", markersize=3.2,
                linewidth=1.55, label=stage,
            )
        axis.set_title(TARGET_LABELS[target], fontsize=12)
        axis.set_xlabel("Distance (km)")
        axis.set_ylabel("OOF residual semivariance")
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.legend(fontsize=7.5, frameon=True)
    figure.suptitle(
        "OOF residual variograms across four predictor stages",
        fontsize=14,
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=-2)
    parser.add_argument(
        "--selected-oof-path",
        type=Path,
        default=TABLE_DIR / "spatial_ffs_selected_oof_predictions.csv",
    )
    parser.add_argument(
        "--selection-path",
        type=Path,
        default=TABLE_DIR / "spatial_ffs_selection_path.csv",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)

    comparison = build_four_stage_oof(
        args.selected_oof_path, args.selection_path, args.n_jobs
    )
    metrics = calculate_metrics(comparison)
    comparison.to_csv(
        args.output_directory / "four_stage_oof_predictions.csv",
        index=False, encoding="utf-8-sig",
    )
    metrics.to_csv(
        args.output_directory / "four_stage_oof_metrics.csv",
        index=False, encoding="utf-8-sig",
    )
    variograms = calculate_residual_variograms(comparison)
    variograms.to_csv(
        args.output_directory / "four_stage_oof_residual_variograms.csv",
        index=False, encoding="utf-8-sig",
    )

    outputs = (
        (
            _map_grid(comparison, "y_pred", residual=False),
            args.output_directory / "13_four_stage_oof_predictions_4x3.png",
        ),
        (
            _map_grid(comparison, "residual", residual=True),
            args.output_directory / "14_four_stage_oof_residual_maps_4x3.png",
        ),
        (
            plot_rmse_trends(metrics),
            args.output_directory / "15_four_stage_oof_rmse_trends_1x3.png",
        ),
        (
            plot_residual_variogram_grid(variograms),
            args.output_directory / "16_four_stage_oof_residual_variograms_4x3.png",
        ),
        (
            plot_residual_variogram_overlay(variograms),
            args.output_directory / "17_four_stage_oof_residual_variograms_1x3.png",
        ),
    )
    for figure, path in outputs:
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        print(path)
    print(args.output_directory / "four_stage_oof_metrics.csv")


if __name__ == "__main__":
    main()
