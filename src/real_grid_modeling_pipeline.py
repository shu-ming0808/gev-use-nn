"""Run real-GRID atmospheric processing through spatial model selection.

This orchestration layer does not download data.  It first verifies that all
three AgERA5 variables cover every month in the requested analysis period,
then builds Tmax-event atmospheric predictors, rebuilds the canonical
model-ready GRID table, and finally runs joint buffered-spatial forward
selection of predictor groups and GP kernels.
"""

from __future__ import annotations

import argparse

from atmospheric_predictors import (
    DEFAULT_RAW_DIR,
    atmospheric_download_coverage,
    build_atmospheric_predictors,
    require_complete_atmospheric_downloads,
)
from data_preprocessing_pipeline import run_preprocessing_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate/process AgERA5, build the model-ready TCCIP GRID, "
            "and run joint spatial predictor/kernel selection."
        )
    )
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rebuild-temperature", action="store_true")
    parser.add_argument("--rebuild-rainfall", action="store_true")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-train", type=int, default=800)
    parser.add_argument("--min-train", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--minimum-rmse-improvement", type=float, default=0.01)
    parser.add_argument("--maximum-vif", type=float, default=5.0)
    parser.add_argument("--n-restarts", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[1/4] Audit AgERA5 archive coverage", flush=True)
    coverage = atmospheric_download_coverage(
        raw_directory=DEFAULT_RAW_DIR,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    print(coverage.to_string(index=False))
    if args.check_only:
        return
    require_complete_atmospheric_downloads(
        raw_directory=DEFAULT_RAW_DIR,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    print("\n[2/4] Match Tmax dates and build GRID atmospheric predictors", flush=True)
    atmosphere, alignment = build_atmospheric_predictors(
        raw_directory=DEFAULT_RAW_DIR,
        analysis_start_year=args.start_year,
        analysis_end_year=args.end_year,
    )
    coverage_columns = [
        column
        for column in atmosphere.columns
        if column.endswith("_available_ratio")
    ]
    minimum_event_coverage = atmosphere[coverage_columns].min().min()
    print(
        f"Atmospheric GRID rows={len(atmosphere)}, "
        f"minimum event coverage={minimum_event_coverage:.3f}"
    )
    print(alignment.to_string(index=False))

    print("\n[3/4] Build canonical model-ready GRID table", flush=True)
    outputs = run_preprocessing_pipeline(
        rebuild_temperature=args.rebuild_temperature,
        rebuild_rainfall=args.rebuild_rainfall,
    )
    model_ready = outputs["model_ready"]
    print(
        f"Model-ready GRID rows={len(model_ready)}, "
        f"columns={model_ready.shape[1]}"
    )
    if args.prepare_only:
        return

    # Delayed import is intentional: atmospheric_predictors.csv now exists,
    # so the atmospheric groups are included in CANDIDATE_GROUPS.
    from spatial_predictor_selection import run_all_targets

    print("\n[4/4] Run buffered Spatial FFS and GP-kernel selection", flush=True)
    _, paths, selected = run_all_targets(
        data=model_ready,
        n_folds=args.n_folds,
        max_train=args.max_train,
        min_train=args.min_train,
        max_steps=args.max_steps,
        min_relative_improvement=args.minimum_rmse_improvement,
        n_restarts=args.n_restarts,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        maximum_allowed_vif=args.maximum_vif,
    )
    print("\nSpatial selection path")
    print(paths.to_string(index=False))
    print("\nSelected development-stage models")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
