"""Nested buffered spatial-CV evaluation for calibrated GEV simulations.

The model-selection stage sees only frozen-NN responses and candidate spatial
predictors.  Known simulated parameter and return-level surfaces are used only
after each outer-fold prediction has been produced.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from elevation_gp_analysis import (
    RANDOM_STATE,
    TARGETS,
    _buffered_training_indices,
    prepare_spatial_folds,
    sample_indices,
)
from calibrated_parametric_simulation import annual_return_level_from_monthly_gev
from land_cover_gp_analysis import (
    BUFFER_KM,
    fit_gp_with_covariates,
    predict_gp_with_covariates,
)
from project_paths import REPOSITORY_ROOT
from spatial_predictor_selection import spatial_forward_selection


DEFAULT_INPUT = (
    REPOSITORY_ROOT
    / "data"
    / "simulated"
    / "calibrated_final_model"
    / "replicate_000_model_ready.csv"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "data"
    / "simulated"
    / "calibrated_final_model"
    / "nested_spatial_cv_monthly"
)


def _validate_monthly_input(data: pd.DataFrame) -> int:
    """Reject stale annual simulations before expensive nested CV begins."""
    required = {"block_scale", "n_months", "months_per_year"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "拒絕執行：輸入缺少月模擬 provenance 欄位 "
            f"{sorted(missing)}；這通常是舊年度模擬檔。"
        )
    if set(data["block_scale"].astype(str).str.lower()) != {"monthly"}:
        raise ValueError("拒絕執行：block_scale 不是 monthly。")
    n_months = data["n_months"].dropna().astype(int).unique()
    months_per_year = data["months_per_year"].dropna().astype(int).unique()
    if len(n_months) != 1 or int(n_months[0]) != 540:
        raise ValueError(f"拒絕執行：預期 n_months=540，實際為 {n_months}。")
    if len(months_per_year) != 1 or int(months_per_year[0]) != 12:
        raise ValueError(
            f"拒絕執行：預期 months_per_year=12，實際為 {months_per_year}。"
        )
    return int(months_per_year[0])


def _summary(error: np.ndarray) -> dict[str, float]:
    error = np.asarray(error, dtype=float)
    valid = np.isfinite(error)
    error = error[valid]
    if len(error) == 0:
        return {"n": 0, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan}
    return {
        "n": int(len(error)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "Bias": float(np.mean(error)),
    }


def summarize_parameter_recovery(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare frozen NN and nested OOF GP estimates with simulated truth."""
    rows = []
    for target, part in predictions.groupby("target", sort=False):
        for estimator, column in (
            ("Frozen NN", "nn_value"),
            ("Nested OOF GP", "oof_prediction"),
        ):
            metrics = _summary(
                part[column].to_numpy(float)
                - part["true_value"].to_numpy(float)
            )
            rows.append({"target": target, "estimator": estimator, **metrics})
    return pd.DataFrame(rows)


def build_return_level_recovery(
    predictions: pd.DataFrame,
    return_periods: tuple[int, ...] = (50, 100),
    months_per_year: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score annual return levels reconstructed from monthly-GEV parameters."""
    keys = ["row_index", "station", "outer_fold"]
    wide = None
    for target in TARGETS:
        part = predictions.loc[
            predictions["target"].eq(target),
            keys + ["true_value", "nn_value", "oof_prediction"],
        ].rename(
            columns={
                "true_value": f"{target}_true",
                "nn_value": f"{target}_nn",
                "oof_prediction": f"{target}_oof",
            }
        )
        if part.empty:
            raise ValueError(f"缺少 {target} 的 nested OOF prediction。")
        wide = part if wide is None else wide.merge(
            part, on=keys, how="inner", validate="one_to_one"
        )

    prediction_parts = []
    metric_rows = []
    for period in return_periods:
        truth = annual_return_level_from_monthly_gev(
            wide["mu_true"], wide["log_sigma_true"], wide["xi_true"], period,
            months_per_year,
        )
        nn = annual_return_level_from_monthly_gev(
            wide["mu_nn"], wide["log_sigma_nn"], wide["xi_nn"], period,
            months_per_year,
        )
        oof = annual_return_level_from_monthly_gev(
            wide["mu_oof"], wide["log_sigma_oof"], wide["xi_oof"], period,
            months_per_year,
        )
        part = wide[keys].copy()
        part["return_period"] = int(period)
        part["true_return_level"] = truth
        part["nn_return_level"] = nn
        part["oof_return_level"] = oof
        prediction_parts.append(part)
        for estimator, values in (("Frozen NN", nn), ("Nested OOF GP", oof)):
            metric_rows.append(
                {
                    "return_period": int(period),
                    "estimator": estimator,
                    **_summary(np.asarray(values) - np.asarray(truth)),
                }
            )
    return pd.concat(prediction_parts, ignore_index=True), pd.DataFrame(metric_rows)


def _fit_outer_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    predictor_names: list[str],
    kernel: str,
    nu: float | None,
    n_restarts: int,
    seed: int,
) -> np.ndarray:
    train_covariates = (
        train[predictor_names].to_numpy(float)
        if predictor_names
        else np.empty((len(train), 0))
    )
    test_covariates = (
        test[predictor_names].to_numpy(float)
        if predictor_names
        else np.empty((len(test), 0))
    )
    model = fit_gp_with_covariates(
        train[["x_km", "y_km"]].to_numpy(float),
        train_covariates,
        train[TARGETS[target]].to_numpy(float),
        predictor_names=predictor_names,
        kernel_name=kernel,
        nu=np.nan if nu is None else nu,
        n_restarts=n_restarts,
        seed=seed,
    )
    return predict_gp_with_covariates(
        model,
        test[["x_km", "y_km"]].to_numpy(float),
        test_covariates,
    )


def run_nested_buffered_spatial_cv(
    data: pd.DataFrame,
    outer_folds: int = 5,
    inner_folds: int = 4,
    max_train: int = 800,
    min_train: int = 100,
    max_steps: int | None = 3,
    min_relative_improvement: float = 0.01,
    maximum_allowed_vif: float = 5.0,
    n_restarts: int = 0,
    n_jobs: int = -2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select in inner CV and evaluate once in untouched outer spatial folds."""
    required = {
        "station", "lon", "lat", "x_km", "y_km",
        "mu_hat", "log_sigma_hat", "xi_hat",
        "mu_true", "log_sigma_true", "xi_true",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"模擬 model-ready table 缺少欄位：{sorted(missing)}")
    working, figure = prepare_spatial_folds(
        data.reset_index(drop=True),
        n_folds=outer_folds,
        random_state=random_state,
    )
    figure.clear()

    prediction_parts = []
    selection_rows = []
    for target_order, target in enumerate(TARGETS):
        for outer_fold in range(outer_folds):
            print(
                f"[{target}] outer fold {outer_fold + 1}/{outer_folds}: "
                "inner FFS + kernel selection",
                flush=True,
            )
            test_indices = working.index[
                working["spatial_fold"].eq(outer_fold)
            ].to_numpy()
            candidates = working.index[
                ~working["spatial_fold"].eq(outer_fold)
            ].to_numpy()
            base_indices = sample_indices(
                candidates,
                max_train,
                random_state + target_order * 100_000 + outer_fold,
            )
            train_indices = _buffered_training_indices(
                working,
                base_indices,
                test_indices,
                BUFFER_KM[target],
            )
            if len(train_indices) < min_train:
                raise ValueError(
                    f"{target}/outer fold {outer_fold} buffer 後僅剩 "
                    f"{len(train_indices)} 個 training GRID。"
                )

            outer_train = working.loc[train_indices].copy().reset_index(drop=True)
            inner, inner_figure = prepare_spatial_folds(
                outer_train,
                n_folds=inner_folds,
                random_state=random_state + 10_000 + outer_fold,
            )
            inner_figure.clear()
            _, path, selected = spatial_forward_selection(
                inner,
                target=target,
                n_folds=inner_folds,
                max_train=max_train,
                min_train=min_train,
                max_steps=max_steps,
                min_relative_improvement=min_relative_improvement,
                n_restarts=n_restarts,
                random_state=random_state + 20_000 + outer_fold,
                n_jobs=n_jobs,
                maximum_allowed_vif=maximum_allowed_vif,
            )
            predictor_names = list(selected["predictor_names"])
            kernel = str(selected["kernel"])
            nu = selected["nu"]
            test = working.loc[test_indices]
            prediction = _fit_outer_model(
                working.loc[train_indices],
                test,
                target,
                predictor_names,
                kernel,
                nu,
                n_restarts,
                random_state + target_order * 1_000_000 + outer_fold,
            )
            prediction_parts.append(
                pd.DataFrame(
                    {
                        "row_index": test_indices,
                        "station": test["station"].to_numpy(),
                        "outer_fold": outer_fold,
                        "target": target,
                        "true_value": test[f"{target}_true"].to_numpy(float),
                        "nn_value": test[TARGETS[target]].to_numpy(float),
                        "oof_prediction": prediction,
                    }
                )
            )
            last = path.iloc[-1]
            selection_rows.append(
                {
                    "target": target,
                    "outer_fold": outer_fold,
                    "selected_groups": last["selected_groups"],
                    "predictors": last["predictors"],
                    "kernel": kernel,
                    "nu": nu,
                    "inner_RMSE": last["RMSE"],
                    "n_outer_train": len(train_indices),
                    "n_outer_test": len(test_indices),
                    "buffer_km": BUFFER_KM[target],
                }
            )
    return pd.concat(prediction_parts, ignore_index=True), pd.DataFrame(selection_rows)


def run_evaluation(
    input_path: str | Path = DEFAULT_INPUT,
    output_directory: str | Path = DEFAULT_OUTPUT,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Run and persist one calibrated-simulation nested-CV evaluation."""
    input_path = Path(input_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_path)
    months_per_year = _validate_monthly_input(data)
    predictions, selections = run_nested_buffered_spatial_cv(data, **kwargs)
    parameter_metrics = summarize_parameter_recovery(predictions)
    rl_predictions, rl_metrics = build_return_level_recovery(
        predictions, months_per_year=months_per_year
    )
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    metadata = pd.DataFrame(
        [
            {
                "input_path": str(input_path.resolve()),
                "input_sha256": input_hash,
                "input_modified_ns": input_path.stat().st_mtime_ns,
                "block_scale": "monthly",
                "n_months": int(data["n_months"].iloc[0]),
                "months_per_year": months_per_year,
                "n_grid": len(data),
            }
        ]
    )
    outputs = {
        "predictions": predictions,
        "selections": selections,
        "parameter_metrics": parameter_metrics,
        "return_level_predictions": rl_predictions,
        "return_level_metrics": rl_metrics,
        "metadata": metadata,
    }
    for name, frame in outputs.items():
        frame.to_csv(
            output_directory / f"calibrated_nested_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--max-train", type=int, default=800)
    parser.add_argument("--min-train", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--min-relative-improvement", type=float, default=0.01)
    parser.add_argument("--maximum-allowed-vif", type=float, default=5.0)
    parser.add_argument("--n-restarts", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-2)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_evaluation(
        input_path=args.input_path,
        output_directory=args.output_directory,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        max_train=args.max_train,
        min_train=args.min_train,
        max_steps=args.max_steps,
        min_relative_improvement=args.min_relative_improvement,
        maximum_allowed_vif=args.maximum_allowed_vif,
        n_restarts=args.n_restarts,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )
    print("\nNested spatial-CV parameter recovery")
    print(outputs["parameter_metrics"].to_string(index=False))
    print("\nNested spatial-CV return-level recovery")
    print(outputs["return_level_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
