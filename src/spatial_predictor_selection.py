"""Spatial forward selection of GP mean-structure predictors.

All candidate predictor sets are compared with the same coordinate-based
K-means folds, the same target-specific buffer, and the same capped training
pool.  The spatial kernel is held fixed during this stage so that the
comparison isolates the contribution of the mean-structure predictors.

The resulting CV scores are development-stage model-selection scores.  A
repeated or nested buffered spatial CV is still required for an unbiased final
generalisation estimate after the predictor set and kernel are selected.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from elevation_gp_analysis import (
    RANDOM_STATE,
    TARGETS,
    _buffered_training_indices,
    gev_return_level,
    prepare_spatial_folds,
    sample_indices,
)
from land_cover_gp_analysis import (
    BUFFER_KM,
    SELECTED_KERNELS,
    _exact_one_sided_p,
    fit_gp_with_covariates,
    predict_gp_with_covariates,
)
from project_paths import (
    PROCESSED_DATA_DIR,
    SPATIAL_PREDICTOR_PROCESSED_DIR,
    TABLE_DIR,
)


# Aspect is a circular quantity and must enter as a sine/cosine pair.
CANDIDATE_GROUPS: OrderedDict[str, list[str]] = OrderedDict(
    [
        ("elevation", ["elevation_m"]),
        ("slope", ["slope_deg"]),
        ("aspect", ["northness", "eastness"]),
        ("local_relief", ["local_relief_m"]),
        ("terrain_ruggedness", ["terrain_ruggedness_m"]),
        ("urban_2000", ["urban_ratio"]),
        ("forest_2000", ["forest_ratio"]),
        ("agriculture_2000", ["agriculture_ratio"]),
        ("water_2000", ["water_ratio"]),
        ("coast_distance", ["coast_distance_km"]),
    ]
)

TERRAIN_COLUMNS = [
    "station",
    "x_km",
    "y_km",
    "elevation_m",
    "slope_deg",
    "northness",
    "eastness",
    "local_relief_m",
    "terrain_ruggedness_m",
]
LAND_COVER_COLUMNS = [
    "station",
    "land_cover_year",
    "urban_ratio",
    "forest_ratio",
    "agriculture_ratio",
    "water_ratio",
    "other_ratio",
]


def load_predictor_selection_data(
    gev_path: str | Path = (
        PROCESSED_DATA_DIR / "grid_station_gev_params_with_loc.csv"
    ),
    terrain_path: str | Path = (
        SPATIAL_PREDICTOR_PROCESSED_DIR
        / "tccip_grid_terrain_predictors.csv"
    ),
    land_cover_path: str | Path = (
        SPATIAL_PREDICTOR_PROCESSED_DIR
        / "tccip_grid_land_cover_2000.csv"
    ),
    coast_distance_path: str | Path = (
        SPATIAL_PREDICTOR_PROCESSED_DIR
        / "tccip_grid_coast_distance.csv"
    ),
) -> pd.DataFrame:
    """One-to-one join responses and all candidate spatial predictors."""
    gev = pd.read_csv(gev_path)
    terrain = pd.read_csv(terrain_path)
    land_cover = pd.read_csv(land_cover_path)
    coast = pd.read_csv(coast_distance_path)

    # The canonical preprocessing table already carries projected coordinates
    # and may also carry all predictors.  Merge only columns that are absent so
    # the loader remains compatible with both the old response-only table and
    # the new model-ready table without creating ``*_x``/``*_y`` duplicates.
    data = gev.copy()
    for source, columns in (
        (terrain, TERRAIN_COLUMNS),
        (land_cover, LAND_COVER_COLUMNS),
        (coast, ["station", "coast_distance_km"]),
    ):
        missing_columns = [
            column
            for column in columns
            if column == "station" or column not in data.columns
        ]
        if len(missing_columns) > 1:
            data = data.merge(
                source[missing_columns],
                on="station",
                how="left",
                validate="one_to_one",
            )
    predictor_columns = [
        column
        for columns in CANDIDATE_GROUPS.values()
        for column in columns
    ]
    required = [
        "x_km",
        "y_km",
        *TARGETS.values(),
        *predictor_columns,
    ]
    missing = data[required].isna().sum()
    if (missing > 0).any():
        details = ", ".join(
            f"{column}={int(count)}"
            for column, count in missing[missing > 0].items()
        )
        raise ValueError(f"候選變數資料仍有缺值：{details}")
    if not (data["land_cover_year"] == 2000).all():
        raise ValueError("土地覆蓋資料並非全部使用 2000 年參考面。")
    return data


def predictor_audit(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return predictor summaries and a long-form correlation audit."""
    predictor_columns = [
        column
        for columns in CANDIDATE_GROUPS.values()
        for column in columns
    ]
    summary = (
        data[predictor_columns]
        .describe()
        .T.reset_index()
        .rename(columns={"index": "predictor"})
    )
    correlation = data[predictor_columns].corr()
    correlation_rows = []
    for index, first in enumerate(predictor_columns):
        for second in predictor_columns[index + 1 :]:
            value = float(correlation.loc[first, second])
            correlation_rows.append(
                {
                    "predictor_1": first,
                    "predictor_2": second,
                    "correlation": value,
                    "abs_correlation": abs(value),
                }
            )
    correlation_long = pd.DataFrame(correlation_rows).sort_values(
        "abs_correlation",
        ascending=False,
    )
    return summary, correlation_long


def _fold_contexts(
    data: pd.DataFrame,
    target: str,
    n_folds: int,
    max_train: int,
    min_train: int,
    random_state: int,
) -> list[dict]:
    """Create fixed test and buffered-training indices for one response."""
    contexts = []
    for fold in range(n_folds):
        test_indices = data.index[
            data["spatial_fold"] == fold
        ].to_numpy()
        candidates = data.index[
            data["spatial_fold"] != fold
        ].to_numpy()
        base_indices = sample_indices(
            candidates,
            max_train,
            random_state + 10_000 + fold,
        )
        train_indices = _buffered_training_indices(
            data,
            base_indices,
            test_indices,
            BUFFER_KM[target],
        )
        if len(train_indices) < min_train:
            raise ValueError(
                f"{target}/fold {fold} 套用 buffer 後只剩 "
                f"{len(train_indices)} 個 training GRID。"
            )
        contexts.append(
            {
                "fold": fold,
                "test_indices": test_indices,
                "base_indices": base_indices,
                "train_indices": train_indices,
            }
        )
    return contexts


def evaluate_predictor_set(
    data: pd.DataFrame,
    target: str,
    predictor_names: list[str],
    contexts: list[dict],
    n_restarts: int = 0,
    random_state: int = RANDOM_STATE,
    model_order: int = 0,
) -> dict:
    """Evaluate one mean structure with fixed buffered geographic folds."""
    response_column = TARGETS[target]
    kernel_name, nu = SELECTED_KERNELS[target]
    fold_rows = []
    prediction_parts = []

    for context in contexts:
        fold = context["fold"]
        train = data.loc[context["train_indices"]]
        test = data.loc[context["test_indices"]]
        if predictor_names:
            train_covariates = train[predictor_names].to_numpy(float)
            test_covariates = test[predictor_names].to_numpy(float)
        else:
            train_covariates = np.empty((len(train), 0))
            test_covariates = np.empty((len(test), 0))

        model = fit_gp_with_covariates(
            train[["x_km", "y_km"]].to_numpy(float),
            train_covariates,
            train[response_column].to_numpy(float),
            predictor_names=predictor_names,
            kernel_name=kernel_name,
            nu=nu,
            n_restarts=n_restarts,
            seed=(
                random_state
                + list(TARGETS).index(target) * 100_000
                + model_order * 100
                + fold
            ),
        )
        prediction = predict_gp_with_covariates(
            model,
            test[["x_km", "y_km"]].to_numpy(float),
            test_covariates,
        )
        truth = test[response_column].to_numpy(float)
        error = prediction - truth
        fold_rows.append(
            {
                "fold": fold,
                "MSE": float(np.mean(error**2)),
                "RMSE": float(np.sqrt(np.mean(error**2))),
                "MAE": float(np.mean(np.abs(error))),
                "Bias": float(np.mean(error)),
                "n_base_train": len(context["base_indices"]),
                "n_retained_train": len(context["train_indices"]),
                "n_test": len(context["test_indices"]),
                "optimizer_success": model["optimizer_success"],
            }
        )
        prediction_parts.append(
            pd.DataFrame(
                {
                    "target": target,
                    "fold": fold,
                    "row_index": context["test_indices"],
                    "station": test["station"].to_numpy(),
                    "y_true": truth,
                    "y_pred": prediction,
                    "residual": truth - prediction,
                }
            )
        )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    error = predictions["y_pred"].to_numpy() - predictions["y_true"].to_numpy()
    return {
        "predictor_names": list(predictor_names),
        "fold_metrics": folds,
        "predictions": predictions,
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "Bias": float(np.mean(error)),
        "fold_RMSE_se": float(folds["RMSE"].std(ddof=1) / np.sqrt(len(folds))),
        "optimizer_success": bool(folds["optimizer_success"].all()),
    }


def spatial_forward_selection(
    data: pd.DataFrame,
    target: str,
    n_folds: int = 5,
    max_train: int = 800,
    min_train: int = 100,
    max_steps: int | None = None,
    min_relative_improvement: float = 0.01,
    n_restarts: int = 0,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run grouped forward selection using buffered spatial-CV RMSE.

    Selection starts from an intercept-only mean.  At each step, the candidate
    group with the lowest global OOF RMSE is added.  The search stops when no
    remaining group improves RMSE by ``min_relative_improvement``.  The
    default 1% threshold is a pre-specified practical-improvement rule that
    prevents negligible fixed-partition gains from creating very long models;
    it is not a hypothesis-test significance level.
    """
    if target not in TARGETS:
        raise ValueError(f"未知的 target：{target}")
    if min_relative_improvement < 0:
        raise ValueError("min_relative_improvement 不可為負值。")
    if "spatial_fold" not in data.columns:
        raise ValueError("請先建立固定 spatial_fold。")

    contexts = _fold_contexts(
        data,
        target=target,
        n_folds=n_folds,
        max_train=max_train,
        min_train=min_train,
        random_state=random_state,
    )
    current = evaluate_predictor_set(
        data,
        target,
        predictor_names=[],
        contexts=contexts,
        n_restarts=n_restarts,
        random_state=random_state,
        model_order=0,
    )
    selected_groups: list[str] = []
    selected_predictors: list[str] = []
    remaining = list(CANDIDATE_GROUPS)
    trial_rows: list[dict] = []
    path_rows = [
        {
            "target": target,
            "step": 0,
            "selected_group": "intercept",
            "selected_groups": "intercept",
            "predictors": "intercept",
            "RMSE": current["RMSE"],
            "MAE": current["MAE"],
            "Bias": current["Bias"],
            "fold_RMSE_se": current["fold_RMSE_se"],
            "relative_RMSE_improvement": np.nan,
            "raw_p": np.nan,
        }
    ]
    max_steps = len(remaining) if max_steps is None else min(
        max_steps,
        len(remaining),
    )

    for step in range(1, max_steps + 1):
        step_results = []
        for candidate_order, candidate_group in enumerate(remaining):
            candidate_predictors = [
                *selected_predictors,
                *CANDIDATE_GROUPS[candidate_group],
            ]
            result = evaluate_predictor_set(
                data,
                target,
                predictor_names=candidate_predictors,
                contexts=contexts,
                n_restarts=n_restarts,
                random_state=random_state,
                model_order=step * 100 + candidate_order,
            )
            mse_difference = (
                current["fold_metrics"]["MSE"].to_numpy()
                - result["fold_metrics"]["MSE"].to_numpy()
            )
            relative_improvement = (
                current["RMSE"] - result["RMSE"]
            ) / current["RMSE"]
            trial = {
                "target": target,
                "step": step,
                "candidate_group": candidate_group,
                "candidate_columns": "+".join(
                    CANDIDATE_GROUPS[candidate_group]
                ),
                "predictors_if_added": "+".join(candidate_predictors),
                "RMSE": result["RMSE"],
                "MAE": result["MAE"],
                "Bias": result["Bias"],
                "fold_RMSE_se": result["fold_RMSE_se"],
                "relative_RMSE_improvement": relative_improvement,
                "folds_improved": int((mse_difference > 0).sum()),
                "mean_fold_MSE_improvement": float(mse_difference.mean()),
                "raw_p": _exact_one_sided_p(mse_difference),
                "optimizer_success": result["optimizer_success"],
            }
            trial_rows.append(trial)
            step_results.append((trial, result))

        best_trial, best_result = min(
            step_results,
            key=lambda item: item[1]["RMSE"],
        )
        if (
            best_trial["relative_RMSE_improvement"]
            <= min_relative_improvement
        ):
            break

        selected_group = best_trial["candidate_group"]
        selected_groups.append(selected_group)
        selected_predictors.extend(CANDIDATE_GROUPS[selected_group])
        remaining.remove(selected_group)
        current = best_result
        path_rows.append(
            {
                "target": target,
                "step": step,
                "selected_group": selected_group,
                "selected_groups": "+".join(selected_groups),
                "predictors": "+".join(selected_predictors),
                "RMSE": current["RMSE"],
                "MAE": current["MAE"],
                "Bias": current["Bias"],
                "fold_RMSE_se": current["fold_RMSE_se"],
                "relative_RMSE_improvement": best_trial[
                    "relative_RMSE_improvement"
                ],
                "raw_p": best_trial["raw_p"],
            }
        )

    return pd.DataFrame(trial_rows), pd.DataFrame(path_rows), current


def calculate_selected_oof_return_levels(
    predictions: pd.DataFrame,
    return_periods: tuple[int, ...] = (50, 100),
    output_directory: str | Path | None = TABLE_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combine selected parameter OOF predictions into return levels.

    The reference return level is calculated from the three NN-derived
    parameter surfaces at the same GRID.  It therefore evaluates the GP
    spatial reconstruction stage and is not an external observed return-level
    truth.
    """
    required = {
        "target",
        "fold",
        "row_index",
        "station",
        "y_true",
        "y_pred",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            f"Selected OOF predictions missing columns: {sorted(missing)}"
        )
    keys = ["row_index", "station", "fold"]
    parts = []
    for target in TARGETS:
        part = predictions.loc[
            predictions["target"].eq(target),
            keys + ["y_true", "y_pred"],
        ].rename(
            columns={
                "y_true": f"{target}_true",
                "y_pred": f"{target}_pred",
            }
        )
        if part.empty:
            raise ValueError(f"No selected OOF predictions for {target}.")
        parts.append(part)
    wide = parts[0]
    for part in parts[1:]:
        wide = wide.merge(
            part,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
    prediction_parts = []
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
        part = wide[keys].copy()
        part["return_period"] = int(return_period)
        part["reference_rl"] = reference
        part["predicted_rl"] = prediction
        part["error"] = prediction - reference
        prediction_parts.append(part)
    return_level_predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    def summarize(group: pd.DataFrame) -> pd.Series:
        valid = np.isfinite(group["reference_rl"]) & np.isfinite(
            group["predicted_rl"]
        )
        error = group.loc[valid, "error"].to_numpy(float)
        return pd.Series(
            {
                "n": len(group),
                "RMSE_vs_NN_reference": float(np.sqrt(np.mean(error**2))),
                "MAE_vs_NN_reference": float(np.mean(np.abs(error))),
                "Bias_vs_NN_reference": float(np.mean(error)),
                "finite_rate": float(np.mean(valid)),
            }
        )

    metrics = (
        return_level_predictions.groupby("return_period", as_index=False)
        .apply(summarize, include_groups=False)
        .reset_index(drop=True)
    )
    fold_metrics = (
        return_level_predictions.groupby(
            ["return_period", "fold"],
            as_index=False,
        )
        .apply(summarize, include_groups=False)
        .reset_index(drop=True)
    )
    if output_directory is not None:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        return_level_predictions.to_csv(
            output_directory
            / "spatial_ffs_selected_return_level_oof_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        metrics.to_csv(
            output_directory / "spatial_ffs_selected_return_level_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        fold_metrics.to_csv(
            output_directory
            / "spatial_ffs_selected_return_level_fold_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return metrics, fold_metrics, return_level_predictions


def run_all_targets(
    data: pd.DataFrame | None = None,
    n_folds: int = 5,
    max_train: int = 800,
    min_train: int = 100,
    max_steps: int | None = None,
    min_relative_improvement: float = 0.01,
    n_restarts: int = 0,
    random_state: int = RANDOM_STATE,
    output_directory: str | Path = TABLE_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run predictor audit and spatial FFS for all three GEV responses."""
    if data is None:
        data = load_predictor_selection_data()
    data, _ = prepare_spatial_folds(
        data,
        n_folds=n_folds,
        random_state=random_state,
    )
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    audit, correlations = predictor_audit(data)
    audit.to_csv(
        output_directory / "spatial_predictor_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    correlations.to_csv(
        output_directory / "spatial_predictor_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_trials = []
    all_paths = []
    final_predictions = []
    for target in TARGETS:
        trials, path, final = spatial_forward_selection(
            data,
            target=target,
            n_folds=n_folds,
            max_train=max_train,
            min_train=min_train,
            max_steps=max_steps,
            min_relative_improvement=min_relative_improvement,
            n_restarts=n_restarts,
            random_state=random_state,
        )
        all_trials.append(trials)
        all_paths.append(path)
        prediction = final["predictions"].copy()
        prediction["selected_predictors"] = (
            path.iloc[-1]["predictors"]
        )
        final_predictions.append(prediction)

    trials = pd.concat(all_trials, ignore_index=True)
    paths = pd.concat(all_paths, ignore_index=True)
    predictions = pd.concat(final_predictions, ignore_index=True)
    trials.to_csv(
        output_directory / "spatial_ffs_trials.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paths.to_csv(
        output_directory / "spatial_ffs_selection_path.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        output_directory / "spatial_ffs_selected_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calculate_selected_oof_return_levels(
        predictions,
        return_periods=(50, 100),
        output_directory=output_directory,
    )
    selected = (
        paths.sort_values("step")
        .groupby("target", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    selected["selection_scope"] = (
        "development-stage fixed buffered 5-fold spatial CV"
    )
    selected["kernel"] = selected["target"].map(
        {target: spec[0] for target, spec in SELECTED_KERNELS.items()}
    )
    selected["nu"] = selected["target"].map(
        {target: spec[1] for target, spec in SELECTED_KERNELS.items()}
    )
    selected.to_csv(
        output_directory / "spatial_ffs_selected_models.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return trials, paths, selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Buffered spatial forward selection for GP predictors."
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-train", type=int, default=800)
    parser.add_argument("--min-train", type=int, default=100)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum selected predictor groups; default searches until no gain.",
    )
    parser.add_argument(
        "--min-relative-improvement",
        type=float,
        default=0.01,
        help=(
            "Pre-specified practical RMSE gain required at each step "
            "(default: 0.01, or 1%)."
        ),
    )
    parser.add_argument("--n-restarts", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--output-directory", type=Path, default=TABLE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, paths, selected = run_all_targets(
        n_folds=args.n_folds,
        max_train=args.max_train,
        min_train=args.min_train,
        max_steps=args.max_steps,
        min_relative_improvement=args.min_relative_improvement,
        n_restarts=args.n_restarts,
        random_state=args.random_state,
        output_directory=args.output_directory,
    )
    print("\nSpatial FFS selection path")
    print(paths.to_string(index=False))
    print("\nDevelopment-stage selected models")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
