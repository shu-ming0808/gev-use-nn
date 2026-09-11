"""Run the complete calibrated simulation pipeline for many replicates.

One timed replicate contains data generation, frozen-NN estimation, nested
buffered spatial cross-validation, forward predictor selection, GP-kernel
selection, and OOF parameter/return-level scoring.  Timing and metric files
are checkpointed after every completed replicate so a long study can resume.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
from time import perf_counter, sleep

# Long command-line simulation studies must not allocate Windows/Tk GUI objects.
# Notebook imports keep their already-selected inline backend.
if __name__ == "__main__":
    os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd

from calibrated_parametric_simulation import (
    DEFAULT_GRID_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR as DEFAULT_MONTHLY_OUTPUT_DIR,
    DEFAULT_SELECTED_MODELS_PATH,
    CalibratedSimulationConfig,
    generate_calibrated_replicate,
    prepare_calibrated_simulation,
)
from calibrated_simulation_spatial_cv import run_evaluation
from calibrated_annual_simulation import (
    DEFAULT_OUTPUT_DIR as DEFAULT_ANNUAL_OUTPUT_DIR,
    annual_generation_outputs_current,
    annual_outputs_current,
    generate_calibrated_annual_replicate,
    run_annual_evaluation,
)


DEFAULT_OUTPUT_DIR = DEFAULT_ANNUAL_OUTPUT_DIR
DEFAULT_CV_ROOT = DEFAULT_OUTPUT_DIR / "nested_spatial_cv_annual"
DEFAULT_TIME_PATH = DEFAULT_OUTPUT_DIR / "simulation_time.csv"


@dataclass(frozen=True)
class CompleteSimulationStudyConfig:
    """Settings for the complete repeated nested-CV simulation study."""

    n_replicates: int = 100
    block_scale: str = "annual"
    outer_folds: int = 5
    inner_folds: int = 4
    max_train: int = 800
    min_train: int = 100
    max_ffs_steps: int = 3
    min_relative_improvement: float = 0.01
    maximum_allowed_vif: float = 5.0
    n_restarts: int = 0
    n_jobs: int = -2
    cv_random_state: int = 20260721
    save_maxima: bool = True
    resume: bool = True
    max_attempts: int = 3
    retry_delay_seconds: float = 10.0
    max_consecutive_failed_replicates: int = 3


TIME_COLUMNS = [
    "replicate",
    "replicate_label",
    "status",
    "attempts",
    "started_at_utc",
    "finished_at_utc",
    "elapsed_seconds",
    "elapsed_hms",
    "running_average_seconds",
    "running_average_hms",
    "cumulative_seconds",
    "cumulative_hms",
    "error_type",
    "error_message",
]


def _format_seconds(seconds: float) -> str:
    """Format seconds as HH:MM:SS without losing runs longer than one day."""
    if not np.isfinite(seconds):
        return ""
    total = max(int(round(float(seconds))), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _atomic_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a CSV atomically so interruption cannot leave a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)
    return path


def update_simulation_time(
    path: str | Path,
    record: dict,
) -> pd.DataFrame:
    """Insert or replace one timing record and recompute running summaries."""
    path = Path(path)
    if path.exists():
        timing = pd.read_csv(path)
    else:
        timing = pd.DataFrame(columns=TIME_COLUMNS)
    if not timing.empty:
        timing = timing.loc[
            timing["replicate"].astype(int).ne(int(record["replicate"]))
        ].copy()
    new_row = pd.DataFrame([record])
    timing = (
        new_row
        if timing.empty
        else pd.concat([timing, new_row], ignore_index=True)
    )
    timing["replicate"] = timing["replicate"].astype(int)
    timing = timing.sort_values("replicate").reset_index(drop=True)

    completed_elapsed: list[float] = []
    running_average = []
    cumulative = []
    for row in timing.itertuples(index=False):
        if row.status == "completed" and np.isfinite(float(row.elapsed_seconds)):
            completed_elapsed.append(float(row.elapsed_seconds))
        running_average.append(
            float(np.mean(completed_elapsed)) if completed_elapsed else np.nan
        )
        cumulative.append(
            float(np.sum(completed_elapsed)) if completed_elapsed else 0.0
        )
    timing["running_average_seconds"] = running_average
    timing["running_average_hms"] = [
        _format_seconds(value) for value in running_average
    ]
    timing["cumulative_seconds"] = cumulative
    timing["cumulative_hms"] = [_format_seconds(value) for value in cumulative]
    timing = timing.reindex(columns=TIME_COLUMNS)
    _atomic_csv(timing, path)
    return timing


def summarize_replicate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize RMSE, MAE, and Bias distributions across replicates."""
    if metrics.empty:
        return pd.DataFrame()
    id_columns = ["result_type", "outcome", "estimator"]
    rows = []
    for keys, part in metrics.groupby(id_columns, dropna=False, sort=False):
        for metric in ("RMSE", "MAE", "Bias"):
            values = part[metric].dropna().to_numpy(float)
            if not len(values):
                continue
            rows.append(
                {
                    **dict(zip(id_columns, keys)),
                    "metric": metric,
                    "n_replicates": int(len(values)),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                    "median": float(np.median(values)),
                    "q025": float(np.quantile(values, 0.025)),
                    "q975": float(np.quantile(values, 0.975)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def _replicate_paths(
    output_directory: Path,
    cv_root: Path,
    replicate: int,
) -> tuple[Path, Path]:
    label = f"replicate_{replicate:03d}"
    return (
        output_directory / f"{label}_model_ready.csv",
        cv_root / label,
    )


def _normalize_block_scale(block_scale: str) -> str:
    block_scale = str(block_scale).strip().lower()
    if block_scale not in {"annual", "monthly"}:
        raise ValueError("block_scale must be 'annual' or 'monthly'.")
    return block_scale


def _cv_prefix(block_scale: str) -> str:
    return (
        "calibrated_annual_nested"
        if _normalize_block_scale(block_scale) == "annual"
        else "calibrated_nested"
    )


def _replicate_is_complete(
    output_directory: Path,
    cv_root: Path,
    replicate: int,
    block_scale: str = "annual",
    *,
    require_saved_maxima: bool = True,
    n_years: int = 45,
) -> bool:
    """Validate checkpoint contents; file existence alone is insufficient."""
    block_scale = _normalize_block_scale(block_scale)
    model_ready, cv_directory = _replicate_paths(
        output_directory, cv_root, replicate
    )
    if block_scale == "annual":
        return annual_generation_outputs_current(
            output_directory,
            replicate,
            n_years=n_years,
            require_annual_maxima=require_saved_maxima,
        ) and annual_outputs_current(model_ready, cv_directory)

    label = f"replicate_{replicate:03d}"
    generated_paths = [
        model_ready,
        output_directory / f"{label}_nn_recovery_metrics.csv",
    ]
    if require_saved_maxima:
        generated_paths.append(
            output_directory / f"{label}_monthly_maxima.csv"
        )
    prefix = _cv_prefix(block_scale)
    cv_paths = {
        name: cv_directory / f"{prefix}_{name}.csv"
        for name in (
            "predictions",
            "selections",
            "parameter_metrics",
            "return_level_predictions",
            "return_level_metrics",
            "metadata",
        )
    }
    if not all(
        path.exists() and path.stat().st_size > 0
        for path in [*generated_paths, *cv_paths.values()]
    ):
        return False
    try:
        data = pd.read_csv(model_ready)
        if set(data["block_scale"].astype(str).str.lower()) != {"monthly"}:
            return False
        if data["station"].duplicated().any() or data.empty:
            return False
        if int(data["months_per_year"].iloc[0]) != 12:
            return False
        if require_saved_maxima:
            maxima = pd.read_csv(generated_paths[-1])
            value_columns = [
                column
                for column in maxima.columns
                if column.startswith("monthly_max_")
            ]
            if (
                len(maxima) != len(data)
                or len(value_columns) != int(n_years) * 12
            ):
                return False
        outputs = {name: pd.read_csv(path) for name, path in cv_paths.items()}
        metadata = outputs["metadata"]
        current_hash = hashlib.sha256(model_ready.read_bytes()).hexdigest()
        if (
            len(metadata) != 1
            or str(metadata.loc[0, "block_scale"]).lower() != "monthly"
            or int(metadata.loc[0, "n_grid"]) != len(data)
            or str(metadata.loc[0, "input_sha256"]) != current_hash
        ):
            return False
        if len(outputs["predictions"]) != len(data) * 3:
            return False
        if len(outputs["return_level_predictions"]) != len(data) * 2:
            return False
        if set(outputs["parameter_metrics"]["target"]) != {
            "mu", "log_sigma", "xi"
        }:
            return False
        if set(outputs["return_level_metrics"]["return_period"].astype(int)) != {
            50, 100
        }:
            return False
        numeric_checks = [
            outputs["predictions"]["oof_prediction"],
            outputs["parameter_metrics"]["RMSE"],
            outputs["return_level_predictions"]["oof_return_level"],
            outputs["return_level_metrics"]["RMSE"],
        ]
        return all(
            np.isfinite(series.to_numpy(float)).all()
            for series in numeric_checks
        )
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return False


def _timing_marks_completed(path: str | Path, replicate: int) -> bool:
    """Require a valid timing checkpoint before a replicate is resumed over."""
    path = Path(path)
    if not path.exists():
        return False
    timing = pd.read_csv(path)
    required = {"replicate", "status", "elapsed_seconds"}
    if not required.issubset(timing.columns):
        return False
    rows = timing.loc[timing["replicate"].astype(int).eq(int(replicate))]
    if rows.empty:
        return False
    row = rows.iloc[-1]
    return (
        str(row["status"]) == "completed"
        and np.isfinite(float(row["elapsed_seconds"]))
    )


def aggregate_completed_replicates(
    output_directory: str | Path,
    cv_root: str | Path,
    n_replicates: int,
    block_scale: str = "annual",
    *,
    require_saved_maxima: bool = True,
    n_years: int = 45,
) -> dict[str, pd.DataFrame]:
    """Collect checkpointed per-replicate metrics and selection results."""
    output_directory = Path(output_directory)
    cv_root = Path(cv_root)
    block_scale = _normalize_block_scale(block_scale)
    prefix = _cv_prefix(block_scale)
    metric_parts = []
    selection_parts = []
    for replicate in range(n_replicates):
        if not _replicate_is_complete(
            output_directory,
            cv_root,
            replicate,
            block_scale,
            require_saved_maxima=require_saved_maxima,
            n_years=n_years,
        ):
            continue
        label = f"replicate_{replicate:03d}"
        cv_directory = cv_root / label
        parameter = pd.read_csv(
            cv_directory / f"{prefix}_parameter_metrics.csv"
        ).rename(columns={"target": "outcome"})
        parameter.insert(0, "result_type", "parameter")
        parameter.insert(0, "replicate", replicate)
        return_level = pd.read_csv(
            cv_directory / f"{prefix}_return_level_metrics.csv"
        )
        return_level["outcome"] = (
            "RL" + return_level["return_period"].astype(int).astype(str)
        )
        return_level.insert(0, "result_type", "return_level")
        return_level.insert(0, "replicate", replicate)
        metric_parts.extend([parameter, return_level])

        selections = pd.read_csv(
            cv_directory / f"{prefix}_selections.csv"
        )
        selections.insert(0, "replicate", replicate)
        selection_parts.append(selections)

    metrics = (
        pd.concat(metric_parts, ignore_index=True)
        if metric_parts
        else pd.DataFrame()
    )
    selections = (
        pd.concat(selection_parts, ignore_index=True)
        if selection_parts
        else pd.DataFrame()
    )
    summary = summarize_replicate_metrics(metrics)
    if selections.empty:
        selection_frequency = pd.DataFrame()
    else:
        group_columns = [
            "target", "selected_groups", "predictors", "kernel", "nu"
        ]
        selection_frequency = (
            selections.groupby(group_columns, dropna=False)
            .size()
            .rename("outer_folds_selected")
            .reset_index()
        )
        denominators = selections.groupby("target").size().rename("total_outer_folds")
        selection_frequency = selection_frequency.merge(
            denominators, on="target", validate="many_to_one"
        )
        selection_frequency["selection_rate"] = (
            selection_frequency["outer_folds_selected"]
            / selection_frequency["total_outer_folds"]
        )
        selection_frequency = selection_frequency.sort_values(
            ["target", "selection_rate"], ascending=[True, False]
        )

    frames = {
        "replicate_metrics": metrics,
        "metric_summary": summary,
        "selections": selections,
        "selection_frequency": selection_frequency,
    }
    for name, frame in frames.items():
        _atomic_csv(
            frame,
            output_directory / f"simulation_{name}.csv",
        )
    return frames


def run_complete_simulation_study(
    study_config: CompleteSimulationStudyConfig = CompleteSimulationStudyConfig(),
    simulation_config: CalibratedSimulationConfig | None = None,
    grid_path: str | Path = DEFAULT_GRID_PATH,
    selected_models_path: str | Path = DEFAULT_SELECTED_MODELS_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    output_directory: str | Path | None = None,
    cv_root: str | Path | None = None,
    time_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run 100 complete replicates by default, checkpointing after each one."""
    if study_config.n_replicates <= 0:
        raise ValueError("n_replicates must be positive.")
    if study_config.max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")
    if study_config.retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds cannot be negative.")
    if study_config.max_consecutive_failed_replicates <= 0:
        raise ValueError("max_consecutive_failed_replicates must be positive.")

    block_scale = _normalize_block_scale(study_config.block_scale)
    if output_directory is None:
        output_directory = (
            DEFAULT_ANNUAL_OUTPUT_DIR
            if block_scale == "annual"
            else DEFAULT_MONTHLY_OUTPUT_DIR
        )
    output_directory = Path(output_directory)
    default_cv_name = (
        "nested_spatial_cv_annual"
        if block_scale == "annual"
        else "nested_spatial_cv_monthly"
    )
    cv_root = Path(cv_root) if cv_root else output_directory / default_cv_name
    time_path = Path(time_path) if time_path else output_directory / "simulation_time.csv"
    simulation_config = simulation_config or CalibratedSimulationConfig(
        n_replicates=study_config.n_replicates,
        months_per_year=1 if block_scale == "annual" else 12,
    )
    if simulation_config.n_replicates != study_config.n_replicates:
        raise ValueError(
            "simulation_config.n_replicates must equal study_config.n_replicates."
        )
    expected_blocks_per_year = 1 if block_scale == "annual" else 12
    if simulation_config.months_per_year != expected_blocks_per_year:
        raise ValueError(
            f"{block_scale} study requires months_per_year="
            f"{expected_blocks_per_year}."
        )

    setup = prepare_calibrated_simulation(
        config=simulation_config,
        grid_path=grid_path,
        selected_models_path=selected_models_path,
        model_path=model_path,
        output_directory=output_directory,
    )
    consecutive_failures = 0
    for replicate in range(study_config.n_replicates):
        checkpoint_valid = _replicate_is_complete(
            output_directory,
            cv_root,
            replicate,
            block_scale,
            require_saved_maxima=study_config.save_maxima,
            n_years=simulation_config.n_years,
        )
        if study_config.resume and checkpoint_valid:
            if not _timing_marks_completed(time_path, replicate):
                update_simulation_time(
                    time_path,
                    {
                        "replicate": replicate,
                        "replicate_label": f"replicate_{replicate:03d}",
                        "status": "recovered",
                        "attempts": 0,
                        "started_at_utc": "",
                        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": np.nan,
                        "elapsed_hms": "",
                        "error_type": "",
                        "error_message": "valid outputs recovered after interruption",
                    },
                )
            print(
                f"replicate {replicate + 1}/{study_config.n_replicates} "
                "輸出內容與 SHA-256 驗證通過，略過。",
                flush=True,
            )
            continue

        label = f"replicate_{replicate:03d}"
        started_at = datetime.now(timezone.utc)
        started = perf_counter()
        success = False
        last_error: Exception | None = None
        for attempt in range(1, study_config.max_attempts + 1):
            update_simulation_time(
                time_path,
                {
                    "replicate": replicate,
                    "replicate_label": label,
                    "status": "running",
                    "attempts": attempt,
                    "started_at_utc": started_at.isoformat(),
                    "finished_at_utc": "",
                    "elapsed_seconds": np.nan,
                    "elapsed_hms": "",
                    "error_type": "",
                    "error_message": "",
                },
            )
            try:
                if block_scale == "annual":
                    generated = generate_calibrated_annual_replicate(
                        setup=setup,
                        config=simulation_config,
                        replicate=replicate,
                        output_directory=output_directory,
                        save_annual_maxima=study_config.save_maxima,
                    )
                else:
                    generated = generate_calibrated_replicate(
                        setup=setup,
                        config=simulation_config,
                        replicate=replicate,
                        output_directory=output_directory,
                        save_monthly_maxima=study_config.save_maxima,
                    )
                _, cv_directory = _replicate_paths(
                    output_directory, cv_root, replicate
                )
                evaluation = (
                    run_annual_evaluation
                    if block_scale == "annual"
                    else run_evaluation
                )
                evaluation(
                    input_path=generated["model_ready"],
                    output_directory=cv_directory,
                    outer_folds=study_config.outer_folds,
                    inner_folds=study_config.inner_folds,
                    max_train=study_config.max_train,
                    min_train=study_config.min_train,
                    max_steps=study_config.max_ffs_steps,
                    min_relative_improvement=(
                        study_config.min_relative_improvement
                    ),
                    maximum_allowed_vif=study_config.maximum_allowed_vif,
                    n_restarts=study_config.n_restarts,
                    n_jobs=study_config.n_jobs,
                    random_state=study_config.cv_random_state,
                )
                if not _replicate_is_complete(
                    output_directory,
                    cv_root,
                    replicate,
                    block_scale,
                    require_saved_maxima=study_config.save_maxima,
                    n_years=simulation_config.n_years,
                ):
                    raise RuntimeError(
                        "replicate finished but output validation failed"
                    )
                success = True
                break
            except Exception as error:
                last_error = error
                elapsed = perf_counter() - started
                update_simulation_time(
                    time_path,
                    {
                        "replicate": replicate,
                        "replicate_label": label,
                        "status": "failed",
                        "attempts": attempt,
                        "started_at_utc": started_at.isoformat(),
                        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": elapsed,
                        "elapsed_hms": _format_seconds(elapsed),
                        "error_type": type(error).__name__,
                        "error_message": str(error).replace("\n", " ")[:500],
                    },
                )
                if attempt < study_config.max_attempts:
                    print(
                        f"{label} 第 {attempt} 次失敗：{error}；"
                        f"{study_config.retry_delay_seconds:g} 秒後重新執行。",
                        flush=True,
                    )
                    if study_config.retry_delay_seconds:
                        sleep(study_config.retry_delay_seconds)

        if not success:
            consecutive_failures += 1
            print(
                f"{label} 已連續嘗試 {study_config.max_attempts} 次仍失敗；"
                "保留 checkpoint，下一次執行會再嘗試。",
                flush=True,
            )
            if consecutive_failures >= study_config.max_consecutive_failed_replicates:
                raise RuntimeError(
                    "Too many consecutive failed replicates; stopping to avoid "
                    "wasting compute. Rerun the same command after checking the "
                    "last error in simulation_time.csv."
                ) from last_error
            continue

        elapsed = perf_counter() - started
        timing = update_simulation_time(
            time_path,
            {
                "replicate": replicate,
                "replicate_label": label,
                "status": "completed",
                "attempts": attempt,
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": elapsed,
                "elapsed_hms": _format_seconds(elapsed),
                "error_type": "",
                "error_message": "",
            },
        )
        aggregate_completed_replicates(
            output_directory,
            cv_root,
            study_config.n_replicates,
            block_scale,
            require_saved_maxima=study_config.save_maxima,
            n_years=simulation_config.n_years,
        )
        consecutive_failures = 0
        completed = timing["status"].eq("completed").sum()
        latest = timing.loc[timing["replicate"].eq(replicate)].iloc[0]
        print(
            f"完整 replicate {replicate + 1}/{study_config.n_replicates} 完成；"
            f"本次 {latest['elapsed_hms']}；"
            f"目前平均 {latest['running_average_hms']}；"
            f"累積 {latest['cumulative_hms']}；"
            f"已完成 {completed} 次。",
            flush=True,
        )

    frames = aggregate_completed_replicates(
        output_directory,
        cv_root,
        study_config.n_replicates,
        block_scale,
        require_saved_maxima=study_config.save_maxima,
        n_years=simulation_config.n_years,
    )
    incomplete = [
        replicate
        for replicate in range(study_config.n_replicates)
        if not _replicate_is_complete(
            output_directory,
            cv_root,
            replicate,
            block_scale,
            require_saved_maxima=study_config.save_maxima,
            n_years=simulation_config.n_years,
        )
    ]
    if incomplete:
        labels = ", ".join(f"{value:03d}" for value in incomplete[:10])
        suffix = " ..." if len(incomplete) > 10 else ""
        raise RuntimeError(
            f"Study incomplete; rerun the same command. Missing/invalid "
            f"replicates: {labels}{suffix}"
        )
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-replicates", type=int, default=100)
    parser.add_argument(
        "--block-scale", choices=("annual", "monthly"), default="annual"
    )
    parser.add_argument("--n-years", type=int, default=45)
    parser.add_argument("--months-per-year", type=int, default=None)
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--calibration-max-train", type=int, default=800)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--max-train", type=int, default=800)
    parser.add_argument("--min-train", type=int, default=100)
    parser.add_argument("--max-ffs-steps", type=int, default=3)
    parser.add_argument("--n-restarts", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-2)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--cv-random-state", type=int, default=20260721)
    parser.add_argument("--grid-path", type=Path, default=DEFAULT_GRID_PATH)
    parser.add_argument(
        "--selected-models-path",
        type=Path,
        default=DEFAULT_SELECTED_MODELS_PATH,
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--cv-root", type=Path, default=None)
    parser.add_argument("--time-path", type=Path, default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Rerun even when a replicate already has complete checkpoint files.",
    )
    parser.add_argument(
        "--skip-maxima-output",
        "--skip-monthly-output",
        dest="skip_maxima_output",
        action="store_true",
        help="Do not persist the generated annual/monthly maxima.",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=10.0)
    parser.add_argument(
        "--max-consecutive-failed-replicates", type=int, default=3
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study_config = CompleteSimulationStudyConfig(
        n_replicates=args.n_replicates,
        block_scale=args.block_scale,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        max_train=args.max_train,
        min_train=args.min_train,
        max_ffs_steps=args.max_ffs_steps,
        n_restarts=args.n_restarts,
        n_jobs=args.n_jobs,
        cv_random_state=args.cv_random_state,
        save_maxima=not args.skip_maxima_output,
        resume=not args.no_resume,
        max_attempts=args.max_attempts,
        retry_delay_seconds=args.retry_delay_seconds,
        max_consecutive_failed_replicates=(
            args.max_consecutive_failed_replicates
        ),
    )
    months_per_year = args.months_per_year
    if months_per_year is None:
        months_per_year = 1 if args.block_scale == "annual" else 12
    simulation_config = CalibratedSimulationConfig(
        n_years=args.n_years,
        months_per_year=months_per_year,
        start_year=args.start_year,
        n_replicates=args.n_replicates,
        calibration_max_train=args.calibration_max_train,
        n_restarts_optimizer=args.n_restarts,
        seed=args.seed,
    )
    run_complete_simulation_study(
        study_config=study_config,
        simulation_config=simulation_config,
        grid_path=args.grid_path,
        selected_models_path=args.selected_models_path,
        model_path=args.model_path,
        output_directory=args.output_directory,
        cv_root=args.cv_root,
        time_path=args.time_path,
    )


if __name__ == "__main__":
    main()
