"""Independent 45-annual-maxima validation of the selected spatial GEV flow.

This module deliberately writes to a separate annual output directory.  It
does not alter the calibrated monthly simulation or its 100-replicate output.
For each GRID cell it draws 45 annual block maxima from known calibrated GEV
surfaces, sends those 45 observations through the frozen NN, and evaluates the
complete nested buffered Spatial-CV/FFS/GP pipeline against known truth.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import genextreme

from calibrated_parametric_simulation import (
    DEFAULT_GRID_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SELECTED_MODELS_PATH,
    CalibratedSimulationConfig,
    CalibratedSimulationSetup,
    estimate_with_frozen_nn,
    generate_true_parameter_surfaces,
    nn_recovery_metrics,
    prepare_calibrated_simulation,
)
from calibrated_simulation_spatial_cv import (
    build_return_level_recovery,
    run_nested_buffered_spatial_cv,
    summarize_parameter_recovery,
)
from project_paths import SIMULATED_DATA_DIR


DEFAULT_OUTPUT_DIR = SIMULATED_DATA_DIR / "calibrated_final_model_annual_45"
DEFAULT_INPUT = DEFAULT_OUTPUT_DIR / "replicate_000_model_ready.csv"
DEFAULT_CV_DIR = DEFAULT_OUTPUT_DIR / "nested_spatial_cv_annual" / "replicate_000"

OUTPUT_PREFIX = "calibrated_annual_nested"
CV_OUTPUT_NAMES = (
    "predictions",
    "selections",
    "parameter_metrics",
    "return_level_predictions",
    "return_level_metrics",
    "metadata",
)


def _atomic_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write one CSV atomically so interruption cannot leave a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)
    return path


def annual_config(
    *,
    n_years: int = 45,
    start_year: int = 1980,
    seed: int = 20260907,
    calibration_max_train: int = 800,
) -> CalibratedSimulationConfig:
    """Return a configuration whose one block per year is explicitly annual."""
    return CalibratedSimulationConfig(
        n_years=n_years,
        months_per_year=1,
        start_year=start_year,
        n_replicates=1,
        calibration_max_train=calibration_max_train,
        n_restarts_optimizer=0,
        seed=seed,
    )


def _validate_annual_config(config: CalibratedSimulationConfig) -> None:
    if config.months_per_year != 1:
        raise ValueError(
            "年度模擬要求 months_per_year=1；月尺度模擬請使用原本的模組。"
        )
    if config.n_years <= 1:
        raise ValueError("n_years must exceed 1.")


def simulate_annual_maxima(
    truth: pd.DataFrame,
    n_years: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one annual block maximum per GRID cell and year."""
    required = {"mu_true", "sigma_true", "xi_true"}
    missing = required.difference(truth.columns)
    if missing:
        raise ValueError(f"truth table is missing columns: {sorted(missing)}")
    if n_years <= 1:
        raise ValueError("n_years must exceed 1.")
    return genextreme.rvs(
        c=-truth["xi_true"].to_numpy(float)[:, None],
        loc=truth["mu_true"].to_numpy(float)[:, None],
        scale=truth["sigma_true"].to_numpy(float)[:, None],
        size=(len(truth), n_years),
        random_state=rng,
    )


def generate_calibrated_annual_replicate(
    setup: CalibratedSimulationSetup,
    config: CalibratedSimulationConfig,
    *,
    replicate: int = 0,
    output_directory: str | Path = DEFAULT_OUTPUT_DIR,
    save_annual_maxima: bool = True,
) -> dict[str, Path]:
    """Generate one 45-annual-maxima data set and frozen-NN estimates."""
    _validate_annual_config(config)
    if replicate < 0:
        raise ValueError("replicate must be non-negative.")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed + replicate * 10_000)
    truth, clipped_xi = generate_true_parameter_surfaces(
        setup.grid, setup.generator, config, rng
    )
    annual = simulate_annual_maxima(truth, config.n_years, rng)
    nn = estimate_with_frozen_nn(
        annual,
        setup.nn_model,
        setup.nn_device,
        months_per_year=1,
    )

    model_ready = truth.copy()
    for column in nn.columns:
        model_ready[column] = nn[column].to_numpy()
    model_ready["replicate"] = replicate
    model_ready["block_scale"] = "annual"
    model_ready["n_years"] = config.n_years
    model_ready["blocks_per_year"] = 1
    model_ready["n_blocks"] = config.n_years
    model_ready["start_year"] = config.start_year
    model_ready["xi_clipped"] = (
        (model_ready["xi_true"] <= config.xi_lower)
        | (model_ready["xi_true"] >= config.xi_upper)
    )

    prefix = f"replicate_{replicate:03d}"
    model_ready_path = output_directory / f"{prefix}_model_ready.csv"
    annual_path = output_directory / f"{prefix}_annual_maxima.csv"
    nn_metric_path = output_directory / f"{prefix}_nn_recovery_metrics.csv"
    _atomic_csv(model_ready, model_ready_path)

    if save_annual_maxima:
        years = range(config.start_year, config.start_year + config.n_years)
        annual_table = pd.DataFrame(
            annual,
            columns=[f"annual_max_{year}" for year in years],
        )
        annual_table.insert(0, "station", setup.grid["station"].to_numpy())
        _atomic_csv(annual_table, annual_path)

    _atomic_csv(nn_recovery_metrics(model_ready), nn_metric_path)
    _atomic_csv(pd.DataFrame(
        [
            {
                **asdict(config),
                "block_scale": "annual",
                "blocks_per_year": 1,
                "n_blocks": config.n_years,
                "xi_clipped_grid_count": clipped_xi,
            }
        ]
    ), output_directory / "annual_simulation_config.csv")
    return {
        "model_ready": model_ready_path,
        "annual_maxima": annual_path,
        "nn_metrics": nn_metric_path,
    }


def validate_annual_input(data: pd.DataFrame) -> int:
    """Reject monthly or stale simulation inputs before expensive CV."""
    required = {
        "block_scale",
        "n_years",
        "blocks_per_year",
        "n_blocks",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "拒絕執行：輸入缺少年度模擬 provenance 欄位 "
            f"{sorted(missing)}。"
        )
    if set(data["block_scale"].astype(str).str.lower()) != {"annual"}:
        raise ValueError("拒絕執行：block_scale 不是 annual。")

    n_years = data["n_years"].dropna().astype(int).unique()
    blocks_per_year = data["blocks_per_year"].dropna().astype(int).unique()
    n_blocks = data["n_blocks"].dropna().astype(int).unique()
    if len(n_years) != 1:
        raise ValueError(f"拒絕執行：n_years 不唯一，實際為 {n_years}。")
    if len(blocks_per_year) != 1 or int(blocks_per_year[0]) != 1:
        raise ValueError(
            "拒絕執行：年度資料必須每年一個 block，"
            f"實際為 {blocks_per_year}。"
        )
    if len(n_blocks) != 1 or int(n_blocks[0]) != int(n_years[0]):
        raise ValueError(
            "拒絕執行：年度資料必須滿足 n_blocks=n_years，"
            f"實際 n_blocks={n_blocks}, n_years={n_years}。"
        )
    return int(n_years[0])


def _output_path(output_directory: Path, name: str) -> Path:
    return output_directory / f"{OUTPUT_PREFIX}_{name}.csv"


def annual_generation_outputs_current(
    output_directory: str | Path,
    replicate: int,
    n_years: int,
    *,
    require_annual_maxima: bool = True,
) -> bool:
    """Validate generated annual data instead of trusting file existence."""
    output_directory = Path(output_directory)
    prefix = f"replicate_{replicate:03d}"
    model_ready_path = output_directory / f"{prefix}_model_ready.csv"
    annual_path = output_directory / f"{prefix}_annual_maxima.csv"
    metric_path = output_directory / f"{prefix}_nn_recovery_metrics.csv"
    required = [model_ready_path, metric_path]
    if require_annual_maxima:
        required.append(annual_path)
    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        return False
    try:
        model_ready = pd.read_csv(model_ready_path)
        if validate_annual_input(model_ready) != int(n_years):
            return False
        if model_ready["station"].duplicated().any() or model_ready.empty:
            return False
        metrics = pd.read_csv(metric_path)
        metric_target = "target" if "target" in metrics.columns else "outcome"
        if metric_target not in metrics.columns:
            return False
        required_targets = {"mu", "log_sigma", "xi"}
        if not required_targets.issubset(
            set(metrics[metric_target].astype(str))
        ):
            return False
        parameter_metrics = metrics.loc[
            metrics[metric_target].astype(str).isin(required_targets)
        ]
        if "RMSE" not in parameter_metrics.columns or not np.isfinite(
            parameter_metrics["RMSE"].to_numpy(float)
        ).all():
            return False
        if require_annual_maxima:
            annual = pd.read_csv(annual_path)
            value_columns = [
                column for column in annual.columns if column.startswith("annual_max_")
            ]
            if len(value_columns) != int(n_years) or len(annual) != len(model_ready):
                return False
            if annual["station"].astype(str).tolist() != model_ready["station"].astype(str).tolist():
                return False
            if not np.isfinite(annual[value_columns].to_numpy(float)).all():
                return False
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return False
    return True


def annual_outputs_current(
    input_path: str | Path,
    output_directory: str | Path,
) -> bool:
    """Check that all annual CV outputs exist and match the current input."""
    input_path = Path(input_path)
    output_directory = Path(output_directory)
    if not input_path.exists():
        return False
    paths = [_output_path(output_directory, name) for name in CV_OUTPUT_NAMES]
    if not all(path.exists() and path.stat().st_size > 0 for path in paths):
        return False
    try:
        data = pd.read_csv(input_path)
        n_years = validate_annual_input(data)
        outputs = load_annual_outputs(output_directory)
        metadata = outputs["metadata"]
        predictions = outputs["predictions"]
        parameter_metrics = outputs["parameter_metrics"]
        return_predictions = outputs["return_level_predictions"]
        return_metrics = outputs["return_level_metrics"]
        selections = outputs["selections"]
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return False
    try:
        if (
            len(metadata) != 1
            or str(metadata.loc[0, "block_scale"]).lower() != "annual"
            or int(metadata.loc[0, "n_years"]) != n_years
            or int(metadata.loc[0, "n_grid"]) != len(data)
        ):
            return False
        current_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
        if str(metadata.loc[0, "input_sha256"]) != current_hash:
            return False
        if len(predictions) != len(data) * 3 or set(predictions["target"]) != {
            "mu", "log_sigma", "xi"
        }:
            return False
        if len(return_predictions) != len(data) * 2 or set(
            return_predictions["return_period"].astype(int)
        ) != {50, 100}:
            return False
        if set(parameter_metrics["target"]) != {"mu", "log_sigma", "xi"}:
            return False
        if set(return_metrics["return_period"].astype(int)) != {50, 100}:
            return False
        if set(selections["target"]) != {"mu", "log_sigma", "xi"}:
            return False
        numeric_checks = [
            predictions["oof_prediction"],
            parameter_metrics["RMSE"],
            return_predictions["oof_return_level"],
            return_metrics["RMSE"],
        ]
        return all(
            np.isfinite(series.to_numpy(float)).all()
            for series in numeric_checks
        )
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return False


def load_annual_outputs(
    output_directory: str | Path = DEFAULT_CV_DIR,
) -> dict[str, pd.DataFrame]:
    """Load an already completed annual nested-CV evaluation."""
    output_directory = Path(output_directory)
    return {
        name: pd.read_csv(_output_path(output_directory, name))
        for name in CV_OUTPUT_NAMES
    }


def run_annual_evaluation(
    input_path: str | Path = DEFAULT_INPUT,
    output_directory: str | Path = DEFAULT_CV_DIR,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Run and persist nested buffered Spatial CV for annual simulation."""
    input_path = Path(input_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_path)
    n_years = validate_annual_input(data)

    predictions, selections = run_nested_buffered_spatial_cv(data, **kwargs)
    parameter_metrics = summarize_parameter_recovery(predictions)
    rl_predictions, rl_metrics = build_return_level_recovery(
        predictions,
        return_periods=(50, 100),
        months_per_year=1,
    )
    metadata = pd.DataFrame(
        [
            {
                "input_path": str(input_path.resolve()),
                "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "input_modified_ns": input_path.stat().st_mtime_ns,
                "block_scale": "annual",
                "n_years": n_years,
                "blocks_per_year": 1,
                "n_blocks": n_years,
                "return_level_probability": "1-1/T",
                "n_grid": len(data),
                "n_jobs": kwargs.get("n_jobs", -2),
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
        _atomic_csv(frame, _output_path(output_directory, name))
    return outputs


def run_annual_pilot(
    *,
    output_directory: str | Path = DEFAULT_OUTPUT_DIR,
    cv_directory: str | Path = DEFAULT_CV_DIR,
    grid_path: str | Path = DEFAULT_GRID_PATH,
    selected_models_path: str | Path = DEFAULT_SELECTED_MODELS_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    n_jobs: int = -2,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Generate and evaluate one annual replicate, with safe resume support."""
    output_directory = Path(output_directory)
    cv_directory = Path(cv_directory)
    model_ready_path = output_directory / "replicate_000_model_ready.csv"
    start = time.perf_counter()
    generated_now = False
    evaluated_now = False

    config = annual_config()
    if force or not annual_generation_outputs_current(
        output_directory,
        replicate=0,
        n_years=config.n_years,
    ):
        setup = prepare_calibrated_simulation(
            config,
            grid_path=grid_path,
            selected_models_path=selected_models_path,
            model_path=model_path,
            output_directory=output_directory,
        )
        generate_calibrated_annual_replicate(
            setup,
            config,
            replicate=0,
            output_directory=output_directory,
        )
        generated_now = True
    else:
        validate_annual_input(pd.read_csv(model_ready_path))

    if force or not annual_outputs_current(model_ready_path, cv_directory):
        outputs = run_annual_evaluation(
            input_path=model_ready_path,
            output_directory=cv_directory,
            outer_folds=5,
            inner_folds=4,
            max_train=800,
            min_train=100,
            max_steps=3,
            min_relative_improvement=0.01,
            maximum_allowed_vif=5.0,
            n_restarts=0,
            n_jobs=n_jobs,
        )
        evaluated_now = True
    else:
        outputs = load_annual_outputs(cv_directory)

    elapsed = time.perf_counter() - start
    timing_path = output_directory / "annual_pilot_time.csv"
    # A notebook reload must not replace the real full-run duration with the
    # few seconds needed to read already-current CSV files.
    if generated_now or evaluated_now or not timing_path.exists():
        pd.DataFrame(
            [
                {
                    "replicate": 0,
                    "generated_now": generated_now,
                    "evaluated_now": evaluated_now,
                    "n_jobs": n_jobs,
                    "elapsed_seconds": elapsed,
                    "elapsed_minutes": elapsed / 60.0,
                }
            ]
        ).to_csv(timing_path, index=False, encoding="utf-8-sig")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cv-directory", type=Path, default=DEFAULT_CV_DIR)
    parser.add_argument("--n-jobs", type=int, default=-2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_annual_pilot(
        output_directory=args.output_directory,
        cv_directory=args.cv_directory,
        n_jobs=args.n_jobs,
        force=args.force,
    )
    print("\n45-annual-maxima parameter recovery")
    print(outputs["parameter_metrics"].to_string(index=False))
    print("\n45-annual-maxima return-level recovery")
    print(outputs["return_level_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
