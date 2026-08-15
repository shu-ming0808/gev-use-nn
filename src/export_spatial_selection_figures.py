"""Export presentation-ready figures for the latest spatial FFS OOF results.

This script is intentionally post-processing only: it reads the OOF tables
already produced by ``real_grid_modeling_pipeline.py`` and does not refit any
Gaussian process.  Every map labelled OOF therefore uses predictions from a
model that excluded the corresponding geographic test fold and its buffer.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from elevation_gp_analysis import empirical_variogram
from plot_selected_oof_parameter_maps import (
    TARGET_LABELS,
    load_selected_oof,
    plot_oof_prediction_surfaces,
    plot_oof_residual_surfaces,
    plot_oof_truth_prediction_residual,
)
from project_paths import FIGURE_DIR, PROCESSED_DATA_DIR, TABLE_DIR
from spatial_diagnostics import (
    plot_isotropy_stationarity_diagnostics,
    prepare_selected_oof_residuals,
)


TARGETS = ("mu", "log_sigma", "xi")
TARGET_PLAIN_LABELS = {
    "mu": r"$\mu$",
    "log_sigma": r"$\log\sigma$",
    "xi": r"$\xi$",
}


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(path)


def plot_fold_map(data: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=True)
    for axis, target in zip(axes, TARGETS):
        part = data.loc[data["target"].eq(target)]
        for fold in sorted(part["fold"].unique()):
            fold_part = part.loc[part["fold"].eq(fold)]
            axis.scatter(
                fold_part["x_km"],
                fold_part["y_km"],
                s=11,
                marker="s",
                linewidths=0,
                label=f"Fold {int(fold)}",
            )
        axis.set_title(TARGET_PLAIN_LABELS[target])
        axis.set_xlabel("Easting (km)")
        axis.set_ylabel("Northing (km)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.15, linewidth=0.5)
    axes[-1].legend(fontsize=8, loc="best")
    figure.suptitle("Geographic test folds used by the selected GP models")
    return figure


def calculate_parameter_metrics(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict] = []
    fold_rows: list[dict] = []
    for target, part in data.groupby("target", sort=False):
        residual = part["y_true"] - part["y_pred"]
        overall_rows.append(
            {
                "target": target,
                "n": len(part),
                "RMSE": float(np.sqrt(np.mean(residual**2))),
                "MAE": float(np.mean(np.abs(residual))),
                "Bias": float(np.mean(-residual)),
                "correlation": float(np.corrcoef(part["y_true"], part["y_pred"])[0, 1]),
            }
        )
        for fold, fold_part in part.groupby("fold"):
            fold_residual = fold_part["y_true"] - fold_part["y_pred"]
            fold_rows.append(
                {
                    "target": target,
                    "fold": int(fold),
                    "n": len(fold_part),
                    "RMSE": float(np.sqrt(np.mean(fold_residual**2))),
                    "MAE": float(np.mean(np.abs(fold_residual))),
                    "Bias": float(np.mean(-fold_residual)),
                }
            )
    return pd.DataFrame(overall_rows), pd.DataFrame(fold_rows)


def plot_observed_predicted(data: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    for axis, target in zip(axes, TARGETS):
        part = data.loc[data["target"].eq(target)]
        lower = float(min(part["y_true"].min(), part["y_pred"].min()))
        upper = float(max(part["y_true"].max(), part["y_pred"].max()))
        residual = part["y_true"] - part["y_pred"]
        rmse = float(np.sqrt(np.mean(residual**2)))
        correlation = float(np.corrcoef(part["y_true"], part["y_pred"])[0, 1])
        axis.scatter(part["y_true"], part["y_pred"], s=8, alpha=0.35)
        axis.plot([lower, upper], [lower, upper], "--", color="black", linewidth=1)
        axis.set(
            title=f"{TARGET_PLAIN_LABELS[target]}: RMSE={rmse:.3f}, r={correlation:.3f}",
            xlabel="NN-derived reference",
            ylabel="OOF GP prediction",
        )
        axis.grid(alpha=0.2)
    figure.suptitle("Selected models: geographically OOF prediction agreement")
    return figure


def plot_fold_rmse(fold_metrics: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
    for axis, target in zip(axes, TARGETS):
        part = fold_metrics.loc[fold_metrics["target"].eq(target)].sort_values("fold")
        axis.plot(part["fold"], part["RMSE"], marker="o", linewidth=1.8)
        axis.axhline(part["RMSE"].mean(), linestyle="--", color="crimson", label="fold mean")
        axis.set(
            title=TARGET_PLAIN_LABELS[target],
            xlabel="Geographic test fold",
            ylabel="Fold RMSE",
            xticks=part["fold"],
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Fold-level stability of the selected GP models")
    return figure


def plot_residual_variograms(data: pd.DataFrame) -> tuple[pd.DataFrame, plt.Figure]:
    rows: list[pd.DataFrame] = []
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for axis, target in zip(axes, TARGETS):
        part = data.loc[data["target"].eq(target)].sort_values("station")
        variogram = empirical_variogram(
            part[["x_km", "y_km"]].to_numpy(float),
            part["residual"].to_numpy(float),
            n_lags=20,
            maxlag_fraction=0.5,
        )
        variogram["target"] = target
        rows.append(variogram)
        axis.plot(
            variogram["lag_km"],
            variogram["semivariance"],
            marker="o",
            markersize=3.5,
            linewidth=1.5,
        )
        axis.set(
            title=TARGET_PLAIN_LABELS[target],
            xlabel="Distance (km)",
            ylabel="OOF residual semivariance",
        )
        axis.grid(alpha=0.25)
    figure.suptitle("Omnidirectional variograms of selected-model OOF residuals")
    return pd.concat(rows, ignore_index=True), figure


def plot_selection_path(paths: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    for axis, target in zip(axes, TARGETS):
        part = paths.loc[paths["target"].eq(target)].sort_values("step")
        labels = [str(value).replace("+", "\n+") for value in part["selected_groups"]]
        axis.plot(part["step"], part["RMSE"], marker="o", linewidth=1.8)
        for step, rmse, label in zip(part["step"], part["RMSE"], labels):
            axis.annotate(label, (step, rmse), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7)
        axis.set(
            title=TARGET_PLAIN_LABELS[target],
            xlabel="Forward-selection step",
            ylabel="Pooled OOF RMSE",
            xticks=part["step"],
        )
        axis.grid(alpha=0.25)
    figure.suptitle("Buffered-spatial forward-selection path")
    return figure


def plot_return_levels(rl: pd.DataFrame) -> tuple[plt.Figure, plt.Figure]:
    locations = pd.read_csv(
        PROCESSED_DATA_DIR / "grid_station_gev_params_with_loc.csv",
        usecols=["station", "x_km", "y_km"],
    )
    data = rl.merge(locations, on="station", how="left", validate="many_to_one")
    map_figure, axes = plt.subplots(2, 3, figsize=(12.8, 9.1), constrained_layout=True)
    scatter_figure, scatter_axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    for row, period in enumerate((50, 100)):
        part = data.loc[data["return_period"].eq(period)].copy()
        joint = np.r_[part["reference_rl"], part["predicted_rl"]]
        lower, upper = np.nanquantile(joint, [0.01, 0.99])
        residual = part["reference_rl"] - part["predicted_rl"]
        limit = float(np.nanquantile(np.abs(residual), 0.99))
        specs = (
            ("reference_rl", f"$RL_{{{period}}}$ NN reference", "viridis", lower, upper),
            ("predicted_rl", f"$RL_{{{period}}}$ OOF prediction", "viridis", lower, upper),
        )
        for column, title, cmap, vmin, vmax in specs:
            points = axes[row, 0 if column == "reference_rl" else 1].scatter(
                part["x_km"], part["y_km"], c=part[column], s=11, marker="s",
                linewidths=0, cmap=cmap, vmin=vmin, vmax=vmax,
            )
            axis = axes[row, 0 if column == "reference_rl" else 1]
            axis.set_title(title)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("Easting (km)")
            axis.set_ylabel("Northing (km)")
            map_figure.colorbar(points, ax=axis, shrink=0.75)
        points = axes[row, 2].scatter(
            part["x_km"], part["y_km"], c=residual, s=11, marker="s",
            linewidths=0, cmap="coolwarm", vmin=-limit, vmax=limit,
        )
        axes[row, 2].set_title(f"$RL_{{{period}}}$ residual")
        axes[row, 2].set_aspect("equal", adjustable="box")
        axes[row, 2].set_xlabel("Easting (km)")
        axes[row, 2].set_ylabel("Northing (km)")
        map_figure.colorbar(points, ax=axes[row, 2], shrink=0.75)

        axis = scatter_axes[row]
        axis.scatter(part["reference_rl"], part["predicted_rl"], s=8, alpha=0.35)
        lo = float(joint.min())
        hi = float(joint.max())
        axis.plot([lo, hi], [lo, hi], "--", color="black")
        rmse = float(np.sqrt(np.mean((part["reference_rl"] - part["predicted_rl"]) ** 2)))
        axis.set(
            title=rf"$RL_{{{period}}}$: RMSE={rmse:.3f} $^\circ$C",
            xlabel="NN-derived return-level reference",
            ylabel="OOF reconstructed return level",
        )
        axis.grid(alpha=0.2)
    map_figure.suptitle("OOF reconstruction of real-data return levels")
    scatter_figure.suptitle("OOF return-level agreement")
    return map_figure, scatter_figure


def plot_model_summary(selected: pd.DataFrame, metrics: pd.DataFrame) -> plt.Figure:
    merged = selected.merge(metrics, on="target", suffixes=("", "_recomputed"))
    rows = []
    for target in TARGETS:
        row = merged.loc[merged["target"].eq(target)].iloc[0]
        kernel = str(row["kernel"])
        if kernel == "Matern":
            kernel = f"Matérn ν={float(row['nu']):g}"
        predictors = str(row["predictors"]).replace("+", " + ")
        rows.append(
            [TARGET_PLAIN_LABELS[target], predictors, kernel, f"{row['RMSE_recomputed']:.4f}", f"{row['MAE_recomputed']:.4f}", f"{row['Bias_recomputed']:.4f}"]
        )
    figure, axis = plt.subplots(figsize=(13.0, 2.8))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["Response", "Selected predictors", "Kernel", "OOF RMSE", "OOF MAE", "Bias (pred-ref)"],
        cellLoc="center",
        loc="center",
        colWidths=[0.08, 0.42, 0.13, 0.11, 0.11, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.65)
    axis.set_title("Final predictor and kernel selection under buffered Spatial CV", pad=18)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)

    prediction_path = TABLE_DIR / "spatial_ffs_selected_oof_predictions.csv"
    selected_path = TABLE_DIR / "spatial_ffs_selected_models.csv"
    path_path = TABLE_DIR / "spatial_ffs_selection_path.csv"
    rl_path = TABLE_DIR / "spatial_ffs_selected_return_level_oof_predictions.csv"
    for path in (prediction_path, selected_path, path_path, rl_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run: python src/real_grid_modeling_pipeline.py --n-jobs -2"
            )

    data = load_selected_oof(
        prediction_path,
        PROCESSED_DATA_DIR / "grid_station_gev_params_with_loc.csv",
    )
    selected = pd.read_csv(selected_path)
    paths = pd.read_csv(path_path)
    rl = pd.read_csv(rl_path)
    metrics, fold_metrics = calculate_parameter_metrics(data)
    metrics.to_csv(args.output_directory / "selected_gp_oof_metrics.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(args.output_directory / "selected_gp_oof_fold_metrics.csv", index=False, encoding="utf-8-sig")

    _save(plot_fold_map(data), args.output_directory / "01_selected_gp_spatial_folds.png")
    _save(plot_selection_path(paths), args.output_directory / "02_selected_gp_ffs_path.png")
    _save(plot_model_summary(selected, metrics), args.output_directory / "03_selected_gp_model_summary.png")
    _save(plot_oof_prediction_surfaces(data), args.output_directory / "04_selected_gp_oof_parameter_surfaces_1x3.png")
    _save(plot_oof_residual_surfaces(data, selected), args.output_directory / "05_selected_gp_oof_residual_maps_1x3.png")
    _save(plot_oof_truth_prediction_residual(data), args.output_directory / "06_selected_gp_reference_prediction_residual_3x3.png")
    _save(plot_observed_predicted(data), args.output_directory / "07_selected_gp_oof_agreement_1x3.png")
    _save(plot_fold_rmse(fold_metrics), args.output_directory / "08_selected_gp_fold_rmse_1x3.png")

    variograms, variogram_figure = plot_residual_variograms(data)
    variograms.to_csv(args.output_directory / "selected_gp_oof_residual_variograms.csv", index=False, encoding="utf-8-sig")
    _save(variogram_figure, args.output_directory / "09_selected_gp_oof_residual_variograms_1x3.png")

    locations = pd.read_csv(
        PROCESSED_DATA_DIR / "grid_station_gev_params_with_loc.csv",
        usecols=["station", "x_km", "y_km"],
    )
    residuals = prepare_selected_oof_residuals(pd.read_csv(prediction_path), locations)
    diagnostic_table, diagnostic_figure = plot_isotropy_stationarity_diagnostics(
        residuals,
        value_columns={
            "mu": "mu_residual",
            "log_sigma": "log_sigma_residual",
            "xi": "xi_residual",
        },
        value_kind="selected multi-predictor OOF residuals",
    )
    diagnostic_table.to_csv(
        args.output_directory / "selected_gp_oof_isotropy_stationarity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _save(diagnostic_figure, args.output_directory / "10_selected_gp_oof_isotropy_stationarity.png")

    rl_maps, rl_agreement = plot_return_levels(rl)
    _save(rl_maps, args.output_directory / "11_selected_gp_oof_return_levels_2x3.png")
    _save(rl_agreement, args.output_directory / "12_selected_gp_oof_return_level_agreement_1x2.png")

    source = FIGURE_DIR / "preprocessing_raw_parameter_isotropy_stationarity.png"
    if source.exists():
        destination = args.output_directory / "00_raw_parameter_isotropy_stationarity.png"
        shutil.copy2(source, destination)
        print(destination)


if __name__ == "__main__":
    main()
