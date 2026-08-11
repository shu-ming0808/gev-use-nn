"""Compare OOF residuals for T0, elevation-only, and selected predictors.

The first two stages use the best spatial kernel within their trend from the
elevation-model comparison.  The third stage uses the latest grouped spatial
forward-selection result.  All metrics are recomputed from cell-level OOF
predictions so the displayed table and residual maps share the same source.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from elevation_gp_analysis import empirical_variogram
from project_paths import PROCESSED_DATA_DIR, TABLE_DIR


TARGETS = ("mu", "log_sigma", "xi")
TARGET_LABELS = {
    "mu": r"$\hat{\mu}$",
    "log_sigma": r"$\widehat{\log\sigma}$",
    "xi": r"$\hat{\xi}$",
}
STAGES = ("No predictors", "Elevation only", "Forward-selected")


def exact_one_sided_sign_flip_p(differences: np.ndarray) -> float:
    """Exact paired randomization p-value for a positive mean improvement.

    ``differences`` is MSE(old model) - MSE(new model), calculated once per
    geographic fold.  Under the null hypothesis of no directional advantage,
    every sign assignment is equally likely.
    """
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return float("nan")
    observed = float(differences.mean())
    statistics = np.asarray(
        [
            np.mean(differences * np.asarray(signs, dtype=float))
            for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
        ]
    )
    return float(np.mean(statistics >= observed - 1e-12))


def calculate_stage_tests(comparison: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare consecutive predictor stages using fold-level paired MSE.

    The forward-selected comparison is exploratory because the same fixed
    folds were used for variable selection and evaluation.  Confirmatory
    inference requires an outer spatial-CV loop that was not used in selection.
    """
    comparisons = (
        (STAGES[0], STAGES[1], "Elevation improves over no predictors", False),
        (STAGES[1], STAGES[2], "Forward selection improves over elevation", True),
    )
    fold_rows: list[dict] = []
    test_rows: list[dict] = []
    for target in TARGETS:
        target_data = comparison.loc[comparison["target"].eq(target)]
        fold_mse = (
            target_data.assign(squared_error=target_data["residual"] ** 2)
            .groupby(["stage", "fold"], observed=True)["squared_error"]
            .mean()
        )
        for old_stage, new_stage, alternative, exploratory in comparisons:
            old_mse = fold_mse.loc[old_stage].sort_index()
            new_mse = fold_mse.loc[new_stage].sort_index()
            common_folds = old_mse.index.intersection(new_mse.index)
            differences = (
                old_mse.loc[common_folds].to_numpy(float)
                - new_mse.loc[common_folds].to_numpy(float)
            )
            raw_p = exact_one_sided_sign_flip_p(differences)
            for fold, mse_old, mse_new, difference in zip(
                common_folds,
                old_mse.loc[common_folds],
                new_mse.loc[common_folds],
                differences,
            ):
                fold_rows.append(
                    {
                        "target": target,
                        "comparison": f"{new_stage} vs {old_stage}",
                        "fold": int(fold),
                        "old_MSE": float(mse_old),
                        "new_MSE": float(mse_new),
                        "D_old_minus_new": float(difference),
                    }
                )
            test_rows.append(
                {
                    "target": target,
                    "comparison": f"{new_stage} vs {old_stage}",
                    "alternative": alternative,
                    "n_folds": int(len(common_folds)),
                    "folds_favoring_new": int((differences > 0).sum()),
                    "mean_D_old_minus_new": float(differences.mean()),
                    "raw_p": raw_p,
                    "decision_at_0.05": (
                        "Reject H0" if raw_p < 0.05 else "Do not reject H0"
                    ),
                    "inference_status": (
                        "exploratory; selection and evaluation reuse folds"
                        if exploratory
                        else "parameter-wise paired fold comparison"
                    ),
                }
            )
    return pd.DataFrame(test_rows), pd.DataFrame(fold_rows)


def load_comparison(
    elevation_oof_path: str | Path,
    best_kernel_path: str | Path,
    selected_oof_path: str | Path,
    selected_model_path: str | Path,
    location_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the three model stages and recompute pooled OOF metrics."""
    elevation_oof = pd.read_csv(elevation_oof_path)
    best_kernels = pd.read_csv(best_kernel_path)
    selected_oof = pd.read_csv(selected_oof_path)
    selected_models = pd.read_csv(selected_model_path)
    locations = pd.read_csv(location_path, usecols=["station", "x_km", "y_km"])

    parts: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    for target in TARGETS:
        for trend, stage in (("T0", STAGES[0]), ("T1", STAGES[1])):
            best = best_kernels.loc[
                best_kernels["target"].eq(target)
                & best_kernels["trend"].eq(trend)
            ].iloc[0]
            part = elevation_oof.loc[
                elevation_oof["target"].eq(target)
                & elevation_oof["model_id"].eq(best["model_id"])
            ].copy()
            part["stage"] = stage
            part["model_spec"] = best["model_id"]
            parts.append(part)

        model = selected_models.loc[selected_models["target"].eq(target)].iloc[0]
        part = selected_oof.loc[selected_oof["target"].eq(target)].copy()
        part["stage"] = STAGES[2]
        part["model_spec"] = model["predictors"]
        parts.append(part)

    comparison = pd.concat(parts, ignore_index=True)
    if not {"x_km", "y_km"}.issubset(comparison.columns):
        comparison = comparison.merge(
            locations,
            on="station",
            how="left",
            validate="many_to_one",
        )
    else:
        missing_coordinates = comparison["x_km"].isna() | comparison["y_km"].isna()
        if missing_coordinates.any():
            replacement = comparison.loc[missing_coordinates].drop(columns=["x_km", "y_km"])
            replacement = replacement.merge(
                locations,
                on="station",
                how="left",
                validate="many_to_one",
            )
            comparison.loc[missing_coordinates, ["x_km", "y_km"]] = replacement[
                ["x_km", "y_km"]
            ].to_numpy()

    comparison["residual"] = comparison["y_true"] - comparison["y_pred"]
    for (target, stage), group in comparison.groupby(["target", "stage"], sort=False):
        prediction_error = group["y_pred"].to_numpy(float) - group["y_true"].to_numpy(float)
        metric_rows.append(
            {
                "target": target,
                "stage": stage,
                "model_spec": group["model_spec"].iloc[0],
                "n": len(group),
                "RMSE": float(np.sqrt(np.mean(prediction_error**2))),
                "MAE": float(np.mean(np.abs(prediction_error))),
                "Bias": float(np.mean(prediction_error)),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics["target"] = pd.Categorical(metrics["target"], TARGETS, ordered=True)
    metrics["stage"] = pd.Categorical(metrics["stage"], STAGES, ordered=True)
    metrics = metrics.sort_values(["target", "stage"]).reset_index(drop=True)
    return comparison, metrics


def plot_residual_grid(comparison: pd.DataFrame) -> plt.Figure:
    """Plot stages by row and GEV responses by column on shared column scales."""
    figure, axes = plt.subplots(3, 3, figsize=(15.0, 11.2), constrained_layout=True)
    limits = {
        target: float(
            comparison.loc[comparison["target"].eq(target), "residual"].abs().max()
        )
        for target in TARGETS
    }
    for row, stage in enumerate(STAGES):
        for column, target in enumerate(TARGETS):
            axis = axes[row, column]
            part = comparison.loc[
                comparison["stage"].eq(stage) & comparison["target"].eq(target)
            ]
            limit = limits[target]
            points = axis.scatter(
                part["x_km"],
                part["y_km"],
                c=part["residual"],
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                marker="s",
                s=9,
                linewidths=0,
            )
            axis.set_title(f"{stage}: {TARGET_LABELS[target]}", fontsize=9.5)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("Easting (km)")
            axis.set_ylabel("Northing (km)")
            axis.grid(alpha=0.15, linewidth=0.5)
            if row == 0:
                figure.colorbar(points, ax=axes[:, column], shrink=0.72, pad=0.015)
    figure.suptitle(
        "Out-of-fold residuals across spatial mean-structure stages",
        fontsize=14,
    )
    figure.text(
        0.5,
        -0.01,
        "Residual = NN-derived reference - geographically out-of-fold GP prediction; "
        "each parameter column uses one common color scale.",
        ha="center",
        fontsize=9,
    )
    return figure


def calculate_stage_variograms(comparison: pd.DataFrame) -> pd.DataFrame:
    """Calculate identically binned empirical OOF residual variograms."""
    parts: list[pd.DataFrame] = []
    for target in TARGETS:
        for stage in STAGES:
            part = comparison.loc[
                comparison["target"].eq(target)
                & comparison["stage"].eq(stage)
            ].sort_values("station")
            empirical = empirical_variogram(
                part[["x_km", "y_km"]].to_numpy(float),
                part["residual"].to_numpy(float),
                n_lags=25,
                maxlag_fraction=0.5,
            )
            empirical["target"] = target
            empirical["stage"] = stage
            parts.append(empirical)
    return pd.concat(parts, ignore_index=True)


def plot_stage_variograms(variograms: pd.DataFrame) -> plt.Figure:
    """Overlay the three mean-structure stages for each GEV response."""
    colors = {
        STAGES[0]: "#4C78A8",
        STAGES[1]: "#F58518",
        STAGES[2]: "#54A24B",
    }
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    for axis, target in zip(axes, TARGETS):
        for stage in STAGES:
            part = variograms.loc[
                variograms["target"].eq(target)
                & variograms["stage"].eq(stage)
            ]
            axis.plot(
                part["lag_km"],
                part["semivariance"],
                marker="o",
                markersize=3.2,
                linewidth=1.5,
                label=stage,
                color=colors[stage],
            )
        axis.set_title(TARGET_LABELS[target], fontsize=12)
        axis.set_xlabel("Distance (km)")
        axis.set_ylabel("Residual semivariance")
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.legend(fontsize=8)
    figure.suptitle(
        "OOF residual variograms across predictor structures",
        fontsize=14,
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--elevation-oof-path",
        type=Path,
        default=TABLE_DIR / "elevation_gp_oof_predictions.csv",
    )
    parser.add_argument(
        "--best-kernel-path",
        type=Path,
        default=TABLE_DIR / "elevation_gp_best_kernel_within_trend.csv",
    )
    parser.add_argument(
        "--selected-oof-path",
        type=Path,
        default=TABLE_DIR / "spatial_ffs_selected_oof_predictions.csv",
    )
    parser.add_argument(
        "--selected-model-path",
        type=Path,
        default=TABLE_DIR / "spatial_ffs_selected_models.csv",
    )
    parser.add_argument(
        "--location-path",
        type=Path,
        default=PROCESSED_DATA_DIR / "grid_station_gev_params_with_loc.csv",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)

    comparison, metrics = load_comparison(
        args.elevation_oof_path,
        args.best_kernel_path,
        args.selected_oof_path,
        args.selected_model_path,
        args.location_path,
    )
    metrics_path = args.output_directory / "predictor_stage_oof_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    tests, fold_differences = calculate_stage_tests(comparison)
    tests_path = args.output_directory / "predictor_stage_oof_tests.csv"
    tests.to_csv(tests_path, index=False, encoding="utf-8-sig")
    fold_differences_path = (
        args.output_directory / "predictor_stage_oof_fold_differences.csv"
    )
    fold_differences.to_csv(
        fold_differences_path,
        index=False,
        encoding="utf-8-sig",
    )
    figure_path = args.output_directory / "predictor_stage_oof_residual_maps_3x3.png"
    figure = plot_residual_grid(comparison)
    figure.savefig(figure_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    variograms = calculate_stage_variograms(comparison)
    variogram_table_path = (
        args.output_directory / "predictor_stage_oof_residual_variograms.csv"
    )
    variograms.to_csv(variogram_table_path, index=False, encoding="utf-8-sig")
    variogram_figure_path = (
        args.output_directory / "predictor_stage_oof_residual_variograms_1x3.png"
    )
    figure = plot_stage_variograms(variograms)
    figure.savefig(
        variogram_figure_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    print(metrics_path)
    print(tests_path)
    print(fold_differences_path)
    print(figure_path)
    print(variogram_table_path)
    print(variogram_figure_path)


if __name__ == "__main__":
    main()
