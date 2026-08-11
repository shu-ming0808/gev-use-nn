"""Joint spatial forward selection of GP predictors and covariance kernels.

All candidate predictor sets are compared with the same coordinate-based
K-means folds, the same target-specific buffer, and the same capped training
pool.  At every forward-selection step, each candidate predictor group is
evaluated with every candidate kernel.  The selected object is therefore a
``predictor set x kernel`` pair rather than a predictor set conditional on a
previous, fixed kernel choice.

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
from joblib import Parallel, delayed

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
    _exact_one_sided_p,
    fit_gp_with_covariates,
    predict_gp_with_covariates,
)


KERNEL_CANDIDATES: tuple[tuple[str, float | None], ...] = (
    ("RBF", None),
    ("Matern", 0.5),
    ("Matern", 1.5),
    ("Matern", 2.5),
)
from project_paths import (
    PROCESSED_DATA_DIR,
    SPATIAL_PREDICTOR_PROCESSED_DIR,
    TABLE_DIR,
)

ATMOSPHERIC_PATH = (
    SPATIAL_PREDICTOR_PROCESSED_DIR
    / "tccip_grid_atmospheric_predictors.csv"
)


# Aspect is a circular quantity and must enter as a sine/cosine pair.
CANDIDATE_GROUPS: OrderedDict[str, list[str]] = OrderedDict(
    [
        ("elevation", ["elevation_m"]),
        ("slope", ["slope_deg"]),
        ("aspect", ["northness", "eastness"]),
        ("local_relief", ["local_relief_m"]),
        ("topographic_position", ["tpi_m"]),
        ("terrain_ruggedness", ["terrain_ruggedness_m"]),
        ("urban_2000", ["urban_ratio"]),
        ("forest_2000", ["forest_ratio"]),
        ("agriculture_2000", ["agriculture_ratio"]),
        ("water_2000", ["water_ratio"]),
        ("coast_distance", ["coast_distance_km"]),
        (
            "rainfall_climatology",
            ["mean_annual_precip_mm", "rain_wet_day_ratio"],
        ),
        (
            "tmax_event_rainfall",
            [
                "tmax_event_rain_mean_mm",
                "tmax_event_rain_wet_ratio",
            ],
        ),
    ]
)

# Atmospheric groups become candidates only after their audited table exists.
# This keeps the historical workflow runnable before CDS-protected data are
# downloaded, while automatically enabling the variables in a fresh process.
if ATMOSPHERIC_PATH.exists():
    CANDIDATE_GROUPS.update(
        [
            ("tmax_event_wind", ["tmax_event_wind_mean_mps"]),
            (
                "tmax_event_solar_radiation",
                ["tmax_event_solar_radiation_mean_mj_m2"],
            ),
            (
                "tmax_event_agera5_cloud_cover",
                ["tmax_event_agera5_cloud_cover_mean_fraction"],
            ),
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
    "tpi_m",
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
    rainfall_path: str | Path = (
        SPATIAL_PREDICTOR_PROCESSED_DIR
        / "tccip_grid_rainfall_predictors.csv"
    ),
    atmospheric_path: str | Path = ATMOSPHERIC_PATH,
) -> pd.DataFrame:
    """One-to-one join responses and all candidate spatial predictors."""
    gev = pd.read_csv(gev_path)
    terrain = pd.read_csv(terrain_path)
    land_cover = pd.read_csv(land_cover_path)
    coast = pd.read_csv(coast_distance_path)
    rainfall = pd.read_csv(rainfall_path)
    atmospheric_path = Path(atmospheric_path)
    atmosphere = (
        pd.read_csv(atmospheric_path)
        if atmospheric_path.exists()
        else None
    )
    if atmosphere is not None:
        coverage_columns = [
            "tmax_event_wind_mean_mps_available_ratio",
            "tmax_event_solar_radiation_mean_mj_m2_available_ratio",
            "tmax_event_agera5_cloud_cover_mean_fraction_available_ratio",
        ]
        missing_coverage = set(coverage_columns).difference(atmosphere.columns)
        if missing_coverage:
            raise ValueError(
                "大氣候選表缺少事件涵蓋率欄位："
                f"{sorted(missing_coverage)}；請用新版 atmospheric_predictors.py 重建。"
            )
        minimum_coverage = atmosphere[coverage_columns].min().min()
        if minimum_coverage < 0.999:
            raise ValueError(
                "大氣事件資料尚未完整下載或配對；最小 GRID-event 涵蓋率為 "
                f"{minimum_coverage:.3f}。完成 1980--2024 後再執行 Spatial FFS。"
            )

    # The canonical preprocessing table already carries projected coordinates
    # and may also carry all predictors.  Merge only columns that are absent so
    # the loader remains compatible with both the old response-only table and
    # the new model-ready table without creating ``*_x``/``*_y`` duplicates.
    data = gev.copy()
    for source, columns in (
        (terrain, TERRAIN_COLUMNS),
        (land_cover, LAND_COVER_COLUMNS),
        (coast, ["station", "coast_distance_km"]),
        (
            rainfall,
            [
                "station",
                "mean_annual_precip_mm",
                "rain_wet_day_ratio",
                "tmax_event_rain_mean_mm",
                "tmax_event_rain_wet_ratio",
            ],
        ),
        (
            atmosphere,
            [
                "station",
                "tmax_event_wind_mean_mps",
                "tmax_event_solar_radiation_mean_mj_m2",
                "tmax_event_agera5_cloud_cover_mean_fraction",
            ],
        ),
    ):
        if source is None:
            continue
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


def predictor_collinearity_audit(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return Spearman pairs and variance-inflation factors.

    This is a diagnostic, not an outcome-based selection rule.  Circular
    aspect columns remain a predefined group in the subsequent spatial FFS.
    """
    columns = [
        column
        for group_columns in CANDIDATE_GROUPS.values()
        for column in group_columns
    ]
    numeric = data[columns].astype(float)
    spearman = numeric.corr(method="spearman")
    rows = []
    for index, first in enumerate(columns):
        for second in columns[index + 1 :]:
            rho = float(spearman.loc[first, second])
            rows.append(
                {
                    "predictor_1": first,
                    "predictor_2": second,
                    "spearman_rho": rho,
                    "abs_spearman_rho": abs(rho),
                    "flag_abs_rho_ge_0p7": abs(rho) >= 0.7,
                }
            )
    pairs = pd.DataFrame(rows).sort_values(
        "abs_spearman_rho", ascending=False
    )

    standardized = (numeric - numeric.mean()) / numeric.std(ddof=0)
    matrix = standardized.to_numpy(float)
    vif_rows = []
    for column_index, column in enumerate(columns):
        response = matrix[:, column_index]
        others = np.delete(matrix, column_index, axis=1)
        design = np.column_stack([np.ones(len(others)), others])
        coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
        residual = response - design @ coefficients
        total_sum_squares = float(np.sum((response - response.mean()) ** 2))
        residual_sum_squares = float(np.sum(residual**2))
        r_squared = 1.0 - residual_sum_squares / total_sum_squares
        vif = np.inf if r_squared >= 1.0 - 1e-12 else 1.0 / (1.0 - r_squared)
        vif_rows.append(
            {
                "predictor": column,
                "r_squared_against_other_predictors": r_squared,
                "vif": vif,
                "flag_vif_ge_5": vif >= 5.0,
            }
        )
    vif_table = pd.DataFrame(vif_rows).sort_values("vif", ascending=False)
    return pairs, vif_table


def _maximum_vif(frame: pd.DataFrame, columns: list[str]) -> float:
    """Maximum VIF for one proposed mean structure."""
    if len(columns) < 2:
        return 1.0
    numeric = frame[columns].astype(float)
    standard_deviation = numeric.std(ddof=0)
    if (standard_deviation <= 1e-12).any():
        return float("inf")
    matrix = ((numeric - numeric.mean()) / standard_deviation).to_numpy()
    maximum = 1.0
    for index in range(matrix.shape[1]):
        response = matrix[:, index]
        others = np.delete(matrix, index, axis=1)
        design = np.column_stack([np.ones(len(others)), others])
        coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
        residual = response - design @ coefficients
        r_squared = 1.0 - np.sum(residual**2) / np.sum(response**2)
        vif = np.inf if r_squared >= 1.0 - 1e-12 else 1.0 / (1.0 - r_squared)
        maximum = max(maximum, float(vif))
    return maximum


def _maximum_training_fold_vif(
    data: pd.DataFrame,
    columns: list[str],
    contexts: list[dict],
) -> float:
    """Worst proposed-model VIF across buffered training folds."""
    return max(
        _maximum_vif(data.loc[context["train_indices"]], columns)
        for context in contexts
    )


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
    kernel_name: str,
    nu: float | None,
    n_restarts: int = 0,
    random_state: int = RANDOM_STATE,
    model_order: int = 0,
) -> dict:
    """Evaluate one predictor-set/kernel pair on fixed buffered folds."""
    response_column = TARGETS[target]
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
                "kernel": kernel_name,
                "nu": nu,
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
                    "kernel": kernel_name,
                    "nu": nu,
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
        "kernel": kernel_name,
        "nu": nu,
        "fold_metrics": folds,
        "predictions": predictions,
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "Bias": float(np.mean(error)),
        "fold_RMSE_se": float(folds["RMSE"].std(ddof=1) / np.sqrt(len(folds))),
        "optimizer_success": bool(folds["optimizer_success"].all()),
    }


def evaluate_candidate_models(
    data: pd.DataFrame,
    target: str,
    predictor_names: list[str],
    contexts: list[dict],
    n_restarts: int,
    random_state: int,
    model_order: int,
    kernel_candidates: tuple[tuple[str, float | None], ...],
    n_jobs: int,
) -> list[dict]:
    """Evaluate all kernels for one predictor set using identical folds."""
    return Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(evaluate_predictor_set)(
            data,
            target,
            predictor_names=predictor_names,
            contexts=contexts,
            kernel_name=kernel_name,
            nu=nu,
            n_restarts=n_restarts,
            random_state=random_state,
            model_order=model_order * len(kernel_candidates) + kernel_order,
        )
        for kernel_order, (kernel_name, nu) in enumerate(kernel_candidates)
    )


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
    kernel_candidates: tuple[tuple[str, float | None], ...] = KERNEL_CANDIDATES,
    n_jobs: int = -2,
    maximum_allowed_vif: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run grouped forward selection using buffered spatial-CV RMSE.

    Selection starts from an intercept-only mean.  At each step, the candidate
    predictor-group/kernel pair with the lowest pooled OOF RMSE is selected.
    The search stops when no remaining group improves RMSE by
    ``min_relative_improvement``.  The
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
    if not kernel_candidates:
        raise ValueError("kernel_candidates 不可為空。")
    if n_jobs == 0:
        raise ValueError("n_jobs 不可為 0；使用 1 代表序列運算，-2 代表保留一核心。")
    if maximum_allowed_vif <= 1:
        raise ValueError("maximum_allowed_vif 必須大於 1。")

    contexts = _fold_contexts(
        data,
        target=target,
        n_folds=n_folds,
        max_train=max_train,
        min_train=min_train,
        random_state=random_state,
    )
    print(
        f"[{target}] baseline: evaluating {len(kernel_candidates)} kernels "
        f"with n_jobs={n_jobs}",
        flush=True,
    )
    baseline_results = evaluate_candidate_models(
        data=data,
        target=target,
        predictor_names=[],
        contexts=contexts,
        n_restarts=n_restarts,
        random_state=random_state,
        model_order=0,
        kernel_candidates=kernel_candidates,
        n_jobs=n_jobs,
    )
    current = min(baseline_results, key=lambda result: result["RMSE"])
    selected_groups: list[str] = []
    selected_predictors: list[str] = []
    remaining = list(CANDIDATE_GROUPS)
    trial_rows: list[dict] = [
        {
            "target": target,
            "step": 0,
            "candidate_group": "intercept",
            "candidate_columns": "intercept",
            "predictors_if_added": "intercept",
            "kernel": result["kernel"],
            "nu": result["nu"],
            "RMSE": result["RMSE"],
            "MAE": result["MAE"],
            "Bias": result["Bias"],
            "fold_RMSE_se": result["fold_RMSE_se"],
            "relative_RMSE_improvement": np.nan,
            "folds_improved": np.nan,
            "mean_fold_MSE_improvement": np.nan,
            "raw_p": np.nan,
            "optimizer_success": result["optimizer_success"],
            "max_training_fold_vif": 1.0,
            "collinearity_eligible": True,
        }
        for result in baseline_results
    ]
    path_rows = [
        {
            "target": target,
            "step": 0,
            "selected_group": "intercept",
            "selected_groups": "intercept",
            "predictors": "intercept",
            "kernel": current["kernel"],
            "nu": current["nu"],
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
        step_specs = []
        for candidate_order, candidate_group in enumerate(remaining):
            candidate_predictors = [
                *selected_predictors,
                *CANDIDATE_GROUPS[candidate_group],
            ]
            maximum_vif = _maximum_training_fold_vif(
                data,
                candidate_predictors,
                contexts,
            )
            if maximum_vif > maximum_allowed_vif:
                trial_rows.append(
                    {
                        "target": target,
                        "step": step,
                        "candidate_group": candidate_group,
                        "candidate_columns": "+".join(
                            CANDIDATE_GROUPS[candidate_group]
                        ),
                        "predictors_if_added": "+".join(candidate_predictors),
                        "kernel": "not_fitted",
                        "nu": np.nan,
                        "RMSE": np.nan,
                        "MAE": np.nan,
                        "Bias": np.nan,
                        "fold_RMSE_se": np.nan,
                        "relative_RMSE_improvement": np.nan,
                        "folds_improved": np.nan,
                        "mean_fold_MSE_improvement": np.nan,
                        "raw_p": np.nan,
                        "optimizer_success": False,
                        "max_training_fold_vif": maximum_vif,
                        "collinearity_eligible": False,
                    }
                )
                continue
            for kernel_order, (kernel_name, nu) in enumerate(kernel_candidates):
                step_specs.append(
                    {
                        "candidate_order": candidate_order,
                        "candidate_group": candidate_group,
                        "candidate_predictors": candidate_predictors,
                        "kernel_order": kernel_order,
                        "kernel_name": kernel_name,
                        "nu": nu,
                        "max_training_fold_vif": maximum_vif,
                    }
                )

        print(
            f"[{target}] step {step}: evaluating {len(step_specs)} "
            f"predictor-kernel models with n_jobs={n_jobs}",
            flush=True,
        )
        if not step_specs:
            print(
                f"[{target}] stop: all remaining predictor groups exceed "
                f"VIF {maximum_allowed_vif:g}",
                flush=True,
            )
            break
        evaluated_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(evaluate_predictor_set)(
                data,
                target,
                predictor_names=spec["candidate_predictors"],
                contexts=contexts,
                kernel_name=spec["kernel_name"],
                nu=spec["nu"],
                n_restarts=n_restarts,
                random_state=random_state,
                model_order=(
                    (step * 100 + spec["candidate_order"])
                    * len(kernel_candidates)
                    + spec["kernel_order"]
                ),
            )
            for spec in step_specs
        )
        for spec, result in zip(step_specs, evaluated_results):
            candidate_group = spec["candidate_group"]
            candidate_predictors = spec["candidate_predictors"]
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
                "kernel": result["kernel"],
                "nu": result["nu"],
                "RMSE": result["RMSE"],
                "MAE": result["MAE"],
                "Bias": result["Bias"],
                "fold_RMSE_se": result["fold_RMSE_se"],
                "relative_RMSE_improvement": relative_improvement,
                "folds_improved": int((mse_difference > 0).sum()),
                "mean_fold_MSE_improvement": float(mse_difference.mean()),
                "raw_p": _exact_one_sided_p(mse_difference),
                "optimizer_success": result["optimizer_success"],
                "max_training_fold_vif": spec["max_training_fold_vif"],
                "collinearity_eligible": True,
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
            print(
                f"[{target}] stop: best relative RMSE improvement="
                f"{best_trial['relative_RMSE_improvement']:.4%}",
                flush=True,
            )
            break

        selected_group = best_trial["candidate_group"]
        selected_groups.append(selected_group)
        selected_predictors.extend(CANDIDATE_GROUPS[selected_group])
        remaining.remove(selected_group)
        current = best_result
        print(
            f"[{target}] selected {selected_group} + {current['kernel']}"
            f"(nu={current['nu']}), RMSE={current['RMSE']:.6f}",
            flush=True,
        )
        path_rows.append(
            {
                "target": target,
                "step": step,
                "selected_group": selected_group,
                "selected_groups": "+".join(selected_groups),
                "predictors": "+".join(selected_predictors),
                "kernel": current["kernel"],
                "nu": current["nu"],
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
    n_jobs: int = -2,
    maximum_allowed_vif: float = 5.0,
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
    spearman_pairs, vif_table = predictor_collinearity_audit(data)
    spearman_pairs.to_csv(
        output_directory / "spatial_predictor_spearman_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    vif_table.to_csv(
        output_directory / "spatial_predictor_vif.csv",
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
            n_jobs=n_jobs,
            maximum_allowed_vif=maximum_allowed_vif,
        )
        all_trials.append(trials)
        all_paths.append(path)
        prediction = final["predictions"].copy()
        prediction["selected_predictors"] = (
            path.iloc[-1]["predictors"]
        )
        prediction["selected_kernel"] = final["kernel"]
        prediction["selected_nu"] = final["nu"]
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
        "development-stage joint predictor-kernel fixed buffered 5-fold spatial CV"
    )
    selected.to_csv(
        output_directory / "spatial_ffs_selected_models.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return trials, paths, selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint buffered spatial selection of GP predictors and kernels."
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
        "--maximum-vif",
        type=float,
        default=5.0,
        help="Reject proposed predictor sets above this training-fold VIF.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Write correlation/VIF diagnostics without fitting any GP.",
    )
    parser.add_argument(
        "--min-relative-improvement",
        type=float,
        default=0.01,
        help=(
            "Pre-specified practical RMSE gain required at each step "
            "(default: 0.01, or 1%%)."
        ),
    )
    parser.add_argument("--n-restarts", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-2,
        help="Parallel workers; default -2 uses all logical CPUs except one.",
    )
    parser.add_argument("--output-directory", type=Path, default=TABLE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_only:
        data = load_predictor_selection_data()
        args.output_directory.mkdir(parents=True, exist_ok=True)
        summary, pearson = predictor_audit(data)
        spearman, vif = predictor_collinearity_audit(data)
        summary.to_csv(
            args.output_directory / "spatial_predictor_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pearson.to_csv(
            args.output_directory / "spatial_predictor_correlations.csv",
            index=False,
            encoding="utf-8-sig",
        )
        spearman.to_csv(
            args.output_directory / "spatial_predictor_spearman_pairs.csv",
            index=False,
            encoding="utf-8-sig",
        )
        vif.to_csv(
            args.output_directory / "spatial_predictor_vif.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("Top absolute Spearman correlations")
        print(spearman.head(20).to_string(index=False))
        print("\nFull candidate-pool VIF diagnostic")
        print(vif.to_string(index=False))
        return
    _, paths, selected = run_all_targets(
        n_folds=args.n_folds,
        max_train=args.max_train,
        min_train=args.min_train,
        max_steps=args.max_steps,
        min_relative_improvement=args.min_relative_improvement,
        n_restarts=args.n_restarts,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        output_directory=args.output_directory,
        maximum_allowed_vif=args.maximum_vif,
    )
    print("\nSpatial FFS selection path")
    print(paths.to_string(index=False))
    print("\nDevelopment-stage selected models")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
