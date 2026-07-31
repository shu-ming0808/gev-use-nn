"""Buffered spatial-CV screening of year-2000 land-cover predictors.

This analysis deliberately keeps the spatial kernel fixed at the winner from
the preceding elevation analysis.  It asks a focused next question: after the
current mean structure is retained, do urban, forest, agriculture, and water
fractions improve geographically held-out prediction?
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from elevation_gp_analysis import (
    RANDOM_STATE,
    TARGETS,
    _buffered_training_indices,
    _center_train_test_km,
    _make_covariance,
    prepare_spatial_folds,
    sample_indices,
)
from project_paths import (
    PROCESSED_DATA_DIR,
    SPATIAL_PREDICTOR_PROCESSED_DIR,
    TABLE_DIR,
)


LAND_COVER_COLUMNS = [
    "urban_ratio",
    "forest_ratio",
    "agriculture_ratio",
    "water_ratio",
]
BASE_PREDICTORS = {
    "mu": ["elevation_m"],
    "log_sigma": ["elevation_m"],
    "xi": [],
}
SELECTED_KERNELS = {
    "mu": ("RBF", np.nan),
    "log_sigma": ("Matern", 0.5),
    "xi": ("RBF", np.nan),
}
BUFFER_KM = {
    "mu": 55.0,
    "log_sigma": 35.0,
    "xi": 30.0,
}


def load_land_cover_model_data(
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
) -> pd.DataFrame:
    """One-to-one join responses, terrain, and land-cover predictors."""
    gev = pd.read_csv(gev_path)
    terrain = pd.read_csv(terrain_path)
    land_cover = pd.read_csv(land_cover_path)

    terrain_columns = ["station", "x_km", "y_km", "elevation_m"]
    land_cover_columns = [
        "station",
        "land_cover_year",
        *LAND_COVER_COLUMNS,
        "other_ratio",
    ]
    data = (
        gev.merge(
            terrain[terrain_columns],
            on="station",
            how="left",
            validate="one_to_one",
        )
        .merge(
            land_cover[land_cover_columns],
            on="station",
            how="left",
            validate="one_to_one",
        )
    )
    required = [
        "x_km",
        "y_km",
        "elevation_m",
        *LAND_COVER_COLUMNS,
    ]
    missing = data[required].isna().sum()
    if (missing > 0).any():
        raise ValueError(
            "模型資料仍有缺值："
            + ", ".join(
                f"{column}={int(count)}"
                for column, count in missing[missing > 0].items()
            )
        )
    if not (data["land_cover_year"] == 2000).all():
        raise ValueError("土地覆蓋資料並非全部來自 2000 年參考面。")
    return data


def _standardized_design(
    covariates: np.ndarray,
    center: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return intercept-plus-standardized-covariate design matrix."""
    covariates = np.asarray(covariates, dtype=float)
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    if covariates.shape[1] == 0:
        return (
            np.ones((len(covariates), 1), dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )
    if center is None:
        center = covariates.mean(axis=0)
    if scale is None:
        scale = covariates.std(axis=0)
    scale = np.asarray(scale, dtype=float)
    if np.any(scale <= 1e-12):
        raise ValueError("至少一個 mean-structure predictor 在訓練資料中為常數。")
    standardized = (covariates - center) / scale
    return (
        np.column_stack([np.ones(len(covariates)), standardized]),
        np.asarray(center, dtype=float),
        scale,
    )


def fit_gp_with_covariates(
    train_xy: np.ndarray,
    train_covariates: np.ndarray,
    response: np.ndarray,
    predictor_names: list[str],
    kernel_name: str,
    nu: float = np.nan,
    n_restarts: int = 0,
    seed: int = RANDOM_STATE,
) -> dict:
    """Fit a universal GP with an arbitrary linear mean structure."""
    response = np.asarray(response, dtype=float)
    coordinates, coordinate_origin = _center_train_test_km(train_xy)
    design, center, scale = _standardized_design(train_covariates)
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
    best = min(fits, key=lambda fit: fit.fun)
    _, components = evaluate(best.x, return_components=True)
    return {
        **components,
        "kernel_name": kernel_name,
        "nu": nu,
        "predictor_names": list(predictor_names),
        "X_train": coordinates,
        "xy_origin_km": coordinate_origin,
        "covariate_center": center,
        "covariate_scale": scale,
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
    }


def predict_gp_with_covariates(
    model: dict,
    test_xy: np.ndarray,
    test_covariates: np.ndarray,
) -> np.ndarray:
    """Predict from a fitted covariate-mean universal GP."""
    coordinates = (
        np.asarray(test_xy, dtype=float) - model["xy_origin_km"]
    )
    design, _, _ = _standardized_design(
        test_covariates,
        model["covariate_center"],
        model["covariate_scale"],
    )
    cross_covariance = model["kernel"](model["X_train"], coordinates)
    return (
        design @ model["beta"]
        + cross_covariance.T @ model["alpha"]
    )


def _candidate_predictors(target: str) -> dict[str, list[str]]:
    """Return the current and land-cover-augmented mean structures."""
    base = BASE_PREDICTORS[target]
    return {
        "current": list(base),
        "land_cover_2000": [*base, *LAND_COVER_COLUMNS],
    }


def run_land_cover_buffered_cv(
    data: pd.DataFrame,
    n_folds: int = 5,
    max_train: int = 800,
    min_train: int = 100,
    n_restarts: int = 0,
    random_state: int = RANDOM_STATE,
    output_directory: str | Path | None = TABLE_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare current and land-cover mean structures under buffered CV."""
    if "spatial_fold" not in data:
        data, _ = prepare_spatial_folds(
            data,
            n_folds=n_folds,
            random_state=random_state,
        )

    base_pool_by_fold: dict[int, np.ndarray] = {}
    for fold in range(n_folds):
        candidates = data.index[data["spatial_fold"] != fold].to_numpy()
        base_pool_by_fold[fold] = sample_indices(
            candidates,
            max_train,
            random_state + 10_000 + fold,
        )

    metric_rows: list[dict] = []
    prediction_parts: list[pd.DataFrame] = []
    for target_order, (target, response_column) in enumerate(TARGETS.items()):
        kernel_name, nu = SELECTED_KERNELS[target]
        candidates = _candidate_predictors(target)
        for fold in range(n_folds):
            test_indices = data.index[
                data["spatial_fold"] == fold
            ].to_numpy()
            base_indices = base_pool_by_fold[fold]
            train_indices = _buffered_training_indices(
                data,
                base_indices,
                test_indices,
                BUFFER_KM[target],
            )
            if len(train_indices) < min_train:
                raise ValueError(
                    f"{target}/fold {fold} 套用 buffer 後只剩"
                    f" {len(train_indices)} 個訓練 GRID。"
                )
            train = data.loc[train_indices]
            test = data.loc[test_indices]

            for candidate_order, (model_id, predictor_names) in enumerate(
                candidates.items()
            ):
                train_covariates = train[predictor_names].to_numpy(float)
                test_covariates = test[predictor_names].to_numpy(float)
                if not predictor_names:
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
                        + target_order * 10_000
                        + fold * 100
                        + candidate_order
                    ),
                )
                prediction = predict_gp_with_covariates(
                    model,
                    test[["x_km", "y_km"]].to_numpy(float),
                    test_covariates,
                )
                truth = test[response_column].to_numpy(float)
                error = prediction - truth
                metric_rows.append(
                    {
                        "target": target,
                        "model_id": model_id,
                        "predictors": "+".join(predictor_names)
                        if predictor_names
                        else "intercept",
                        "kernel": kernel_name,
                        "nu": nu,
                        "fold": fold,
                        "buffer_km": BUFFER_KM[target],
                        "n_base_train": len(base_indices),
                        "n_retained_train": len(train_indices),
                        "n_test": len(test_indices),
                        "RMSE": float(np.sqrt(np.mean(error**2))),
                        "MAE": float(np.mean(np.abs(error))),
                        "Bias": float(np.mean(error)),
                        "optimizer_success": model["optimizer_success"],
                    }
                )
                prediction_parts.append(
                    pd.DataFrame(
                        {
                            "target": target,
                            "model_id": model_id,
                            "fold": fold,
                            "row_index": test_indices,
                            "station": test["station"].to_numpy(),
                            "lon": test["lon"].to_numpy(float),
                            "lat": test["lat"].to_numpy(float),
                            "y_true": truth,
                            "y_pred": prediction,
                            "residual": truth - prediction,
                        }
                    )
                )

    fold_metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    summary_rows: list[dict] = []
    for (target, model_id), part in predictions.groupby(
        ["target", "model_id"]
    ):
        error = part["y_pred"].to_numpy() - part["y_true"].to_numpy()
        matching = fold_metrics.query(
            "target == @target and model_id == @model_id"
        )
        summary_rows.append(
            {
                "target": target,
                "model_id": model_id,
                "predictors": matching.iloc[0]["predictors"],
                "kernel": matching.iloc[0]["kernel"],
                "nu": matching.iloc[0]["nu"],
                "RMSE": float(np.sqrt(np.mean(error**2))),
                "MAE": float(np.mean(np.abs(error))),
                "Bias": float(np.mean(error)),
                "fold_RMSE_sd": float(
                    matching["RMSE"].std(ddof=1)
                ),
                "mean_retained_train": float(
                    matching["n_retained_train"].mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["target", "RMSE"]
    )

    if output_directory is not None:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        fold_metrics.to_csv(
            output_directory / "land_cover_gp_fold_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summary.to_csv(
            output_directory / "land_cover_gp_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        predictions.to_csv(
            output_directory / "land_cover_gp_oof_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return fold_metrics, summary, predictions


def _exact_one_sided_p(differences: np.ndarray) -> float:
    """Exact sign-flip p-value for a positive mean paired improvement."""
    differences = np.asarray(differences, dtype=float)
    observed = float(differences.mean())
    reference = [
        float(np.mean(differences * np.asarray(signs)))
        for signs in product((-1.0, 1.0), repeat=len(differences))
    ]
    return float(np.mean(np.asarray(reference) >= observed - 1e-12))


def land_cover_parameter_tests(
    fold_metrics: pd.DataFrame,
    output_path: str | Path | None = (
        TABLE_DIR / "land_cover_gp_parameter_tests.csv"
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test whether land-cover augmentation lowers fold-level MSE."""
    rows: list[dict] = []
    differences: list[pd.DataFrame] = []
    for target in TARGETS:
        part = fold_metrics.query("target == @target")
        wide = part.pivot(
            index="fold",
            columns="model_id",
            values="RMSE",
        )
        mse_difference = wide["current"] ** 2 - wide[
            "land_cover_2000"
        ] ** 2
        differences.append(
            pd.DataFrame(
                {
                    "target": target,
                    "fold": mse_difference.index,
                    "MSE_current_minus_land_cover": mse_difference.values,
                }
            )
        )
        rows.append(
            {
                "target": target,
                "H1": "land_cover_2000 has lower MSE",
                "n_folds": len(mse_difference),
                "folds_improved": int((mse_difference > 0).sum()),
                "mean_MSE_improvement": float(mse_difference.mean()),
                "raw_p": _exact_one_sided_p(mse_difference.to_numpy()),
            }
        )
    tests = pd.DataFrame(rows)
    tests["decision_raw_0.05"] = np.where(
        tests["raw_p"] < 0.05,
        "Reject H0",
        "Do not reject H0",
    )
    differences_frame = pd.concat(differences, ignore_index=True)
    if output_path is not None:
        output_path = Path(output_path)
        tests.to_csv(output_path, index=False, encoding="utf-8-sig")
        differences_frame.to_csv(
            output_path.with_name(
                "land_cover_gp_parameter_fold_differences.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )
    return tests, differences_frame


def return_level(
    mu: np.ndarray,
    log_sigma: np.ndarray,
    xi: np.ndarray,
    period: int,
) -> np.ndarray:
    """Calculate a GEV return level using the project parameterization."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.exp(np.asarray(log_sigma, dtype=float))
    xi = np.asarray(xi, dtype=float)
    transformed_probability = -np.log1p(-1.0 / float(period))
    near_zero = np.abs(xi) < 1e-6
    result = np.empty_like(mu)
    result[near_zero] = (
        mu[near_zero]
        - sigma[near_zero] * np.log(transformed_probability)
    )
    result[~near_zero] = (
        mu[~near_zero]
        + sigma[~near_zero]
        / xi[~near_zero]
        * (
            transformed_probability ** (-xi[~near_zero])
            - 1.0
        )
    )
    return result


def evaluate_land_cover_return_levels(
    predictions: pd.DataFrame,
    periods: tuple[int, ...] = (50, 100),
    selected_model_by_target: dict[str, str] | None = None,
    output_directory: str | Path | None = TABLE_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare current and land-cover OOF return-level prediction."""
    if selected_model_by_target is not None:
        selected_parts: list[pd.DataFrame] = []
        for target, model_id in selected_model_by_target.items():
            part = predictions.query(
                "target == @target and model_id == @model_id"
            ).copy()
            if part.empty:
                raise ValueError(
                    f"找不到 mixed pipeline 所需的 {target}/{model_id}。"
                )
            part["model_id"] = "mixed_selected"
            selected_parts.append(part)
        predictions = pd.concat(
            [predictions, *selected_parts],
            ignore_index=True,
        )

    wide_true = predictions.pivot_table(
        index=["model_id", "fold", "row_index", "station", "lon", "lat"],
        columns="target",
        values="y_true",
        aggfunc="first",
    ).reset_index()
    wide_pred = predictions.pivot(
        index=["model_id", "fold", "row_index", "station", "lon", "lat"],
        columns="target",
        values="y_pred",
    ).reset_index()
    joined = wide_true.merge(
        wide_pred,
        on=["model_id", "fold", "row_index", "station", "lon", "lat"],
        suffixes=("_true", "_pred"),
        validate="one_to_one",
    )

    rows: list[dict] = []
    fold_rows: list[dict] = []
    prediction_parts: list[pd.DataFrame] = []
    for period in periods:
        for model_id, part in joined.groupby("model_id"):
            truth = return_level(
                part["mu_true"],
                part["log_sigma_true"],
                part["xi_true"],
                period,
            )
            predicted = return_level(
                part["mu_pred"],
                part["log_sigma_pred"],
                part["xi_pred"],
                period,
            )
            error = predicted - truth
            rows.append(
                {
                    "period": period,
                    "model_id": model_id,
                    "RMSE": float(np.sqrt(np.mean(error**2))),
                    "MAE": float(np.mean(np.abs(error))),
                    "Bias": float(np.mean(error)),
                }
            )
            output = part[
                ["fold", "row_index", "station", "lon", "lat"]
            ].copy()
            output["period"] = period
            output["model_id"] = model_id
            output["rl_true"] = truth
            output["rl_pred"] = predicted
            output["residual"] = truth - predicted
            prediction_parts.append(output)
            for fold, fold_part in output.groupby("fold"):
                fold_error = (
                    fold_part["rl_pred"].to_numpy()
                    - fold_part["rl_true"].to_numpy()
                )
                fold_rows.append(
                    {
                        "period": period,
                        "model_id": model_id,
                        "fold": fold,
                        "MSE": float(np.mean(fold_error**2)),
                        "RMSE": float(np.sqrt(np.mean(fold_error**2))),
                    }
                )

    metrics = pd.DataFrame(rows)
    fold_metrics = pd.DataFrame(fold_rows)
    rl_predictions = pd.concat(prediction_parts, ignore_index=True)
    test_rows: list[dict] = []
    alternatives = [
        model_id
        for model_id in fold_metrics["model_id"].unique()
        if model_id != "current"
    ]
    for period in periods:
        wide = fold_metrics.query("period == @period").pivot(
            index="fold",
            columns="model_id",
            values="MSE",
        )
        for model_id in alternatives:
            difference = wide["current"] - wide[model_id]
            test_rows.append(
                {
                    "period": period,
                    "model_id": model_id,
                    "H1": f"{model_id} has lower MSE",
                    "n_folds": len(difference),
                    "folds_improved": int((difference > 0).sum()),
                    "mean_MSE_improvement": float(difference.mean()),
                    "raw_p": _exact_one_sided_p(
                        difference.to_numpy()
                    ),
                }
            )
    tests = pd.DataFrame(test_rows)
    tests["decision_raw_0.05"] = np.where(
        tests["raw_p"] < 0.05,
        "Reject H0",
        "Do not reject H0",
    )

    if output_directory is not None:
        output_directory = Path(output_directory)
        metrics.to_csv(
            output_directory / "land_cover_gp_return_level_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        fold_metrics.to_csv(
            output_directory
            / "land_cover_gp_return_level_fold_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        tests.to_csv(
            output_directory / "land_cover_gp_return_level_tests.csv",
            index=False,
            encoding="utf-8-sig",
        )
        rl_predictions.to_csv(
            output_directory
            / "land_cover_gp_return_level_oof_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return metrics, tests, rl_predictions


def main() -> None:
    data = load_land_cover_model_data()
    fold_metrics, summary, predictions = run_land_cover_buffered_cv(data)
    tests, _ = land_cover_parameter_tests(fold_metrics)
    selected_model_by_target = {
        row.target: (
            "land_cover_2000"
            if row.raw_p < 0.05
            and row.mean_MSE_improvement > 0
            else "current"
        )
        for row in tests.itertuples(index=False)
    }
    selected_models = pd.DataFrame(
        [
            {
                "target": target,
                "selected_model": model_id,
                "selection_rule": (
                    "lower buffered-CV MSE and raw one-sided p < 0.05"
                ),
            }
            for target, model_id in selected_model_by_target.items()
        ]
    )
    selected_models.to_csv(
        TABLE_DIR / "land_cover_gp_selected_models.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rl_metrics, rl_tests, _ = evaluate_land_cover_return_levels(
        predictions,
        selected_model_by_target=selected_model_by_target,
    )
    print("\nParameter model summary")
    print(summary.to_string(index=False))
    print("\nParameter tests")
    print(tests.to_string(index=False))
    print("\nReturn-level summary")
    print(rl_metrics.to_string(index=False))
    print("\nReturn-level tests")
    print(rl_tests.to_string(index=False))
    print("\nSelected parameter-specific pipeline")
    print(selected_models.to_string(index=False))


if __name__ == "__main__":
    main()
