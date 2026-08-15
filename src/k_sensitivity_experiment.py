"""Run the experimental K=3--7 buffered Spatial-CV sensitivity analysis.

This is intentionally separate from the formal K=5 notebook workflow.  Each
candidate K uses the same GRID data, buffers, training cap, FFS procedure, and
GP kernel candidates.  Completed K directories are reusable after interruption.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from elevation_gp_analysis import prepare_spatial_folds
from spatial_predictor_selection import (
    TARGETS,
    _fold_contexts,
    load_predictor_selection_data,
    run_all_targets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "tables" / "experimental_k_sensitivity"
DEFAULT_FIGURE = (
    PROJECT_ROOT / "results" / "figures" / "experimental_k_sensitivity_rl_rmse.png"
)


def _completed_outputs(directory: Path) -> bool:
    required = (
        "spatial_ffs_selected_models.csv",
        "spatial_ffs_selected_return_level_metrics.csv",
        "spatial_ffs_selected_return_level_fold_metrics.csv",
    )
    return all((directory / name).exists() for name in required)


def _summarize_completed_k(
    candidate_k: int,
    candidate_dir: Path,
    retained_counts: list[int],
    retention_rates: list[float],
) -> dict:
    models = pd.read_csv(candidate_dir / "spatial_ffs_selected_models.csv")
    metrics = pd.read_csv(
        candidate_dir / "spatial_ffs_selected_return_level_metrics.csv"
    )
    fold_metrics = pd.read_csv(
        candidate_dir / "spatial_ffs_selected_return_level_fold_metrics.csv"
    )
    model_parts = []
    for row in models.itertuples():
        predictors = "intercept" if pd.isna(row.predictors) else row.predictors
        nu_text = "" if pd.isna(row.nu) else f" nu={row.nu}"
        model_parts.append(f"{row.target}: {predictors} | {row.kernel}{nu_text}")

    result = {
        "K": candidate_k,
        "feasible": True,
        "failure_reason": "",
        "min_retained_train": min(retained_counts),
        "mean_retention_rate": float(np.mean(retention_rates)),
        "selected_models": "; ".join(model_parts),
    }
    for return_period in (50, 100):
        metric = metrics.query("return_period == @return_period").iloc[0]
        by_fold = fold_metrics.query("return_period == @return_period")
        result[f"RL{return_period}_RMSE"] = float(
            metric["RMSE_vs_NN_reference"]
        )
        result[f"RL{return_period}_fold_RMSE_SD"] = float(
            by_fold["RMSE_vs_NN_reference"].std(ddof=1)
        )
    return result


def run_experiment(
    k_values: tuple[int, ...],
    output_directory: Path,
    figure_path: Path,
    max_train: int = 800,
    min_train: int = 100,
    random_state: int = 111,
    n_jobs: int = -2,
    force: bool = False,
) -> pd.DataFrame:
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_predictor_selection_data()
    rows = []

    for position, candidate_k in enumerate(k_values, start=1):
        print(
            f"[{position}/{len(k_values)}] K={candidate_k}: prepare folds and buffers",
            flush=True,
        )
        candidate_dir = output_directory / f"K{candidate_k}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        prepared, fold_figure = prepare_spatial_folds(
            data,
            n_folds=candidate_k,
            random_state=random_state,
        )
        plt.close(fold_figure)
        retained_counts: list[int] = []
        retention_rates: list[float] = []

        try:
            for target in TARGETS:
                contexts = _fold_contexts(
                    prepared,
                    target=target,
                    n_folds=candidate_k,
                    max_train=max_train,
                    min_train=min_train,
                    random_state=random_state,
                )
                retained_counts.extend(
                    len(context["train_indices"]) for context in contexts
                )
                retention_rates.extend(
                    len(context["train_indices"]) / len(context["base_indices"])
                    for context in contexts
                )

            if force or not _completed_outputs(candidate_dir):
                print(
                    f"[{position}/{len(k_values)}] K={candidate_k}: run FFS and GP CV",
                    flush=True,
                )
                run_all_targets(
                    data=data,
                    n_folds=candidate_k,
                    max_train=max_train,
                    min_train=min_train,
                    random_state=random_state,
                    output_directory=candidate_dir,
                    n_jobs=n_jobs,
                )
            else:
                print(
                    f"[{position}/{len(k_values)}] K={candidate_k}: reuse completed outputs",
                    flush=True,
                )

            rows.append(
                _summarize_completed_k(
                    candidate_k,
                    candidate_dir,
                    retained_counts,
                    retention_rates,
                )
            )
            print(f"[{position}/{len(k_values)}] K={candidate_k}: completed", flush=True)
        except ValueError as error:
            rows.append(
                {
                    "K": candidate_k,
                    "feasible": False,
                    "failure_reason": str(error),
                    "min_retained_train": (
                        min(retained_counts) if retained_counts else np.nan
                    ),
                    "mean_retention_rate": (
                        float(np.mean(retention_rates)) if retention_rates else np.nan
                    ),
                    "selected_models": "",
                    "RL50_RMSE": np.nan,
                    "RL50_fold_RMSE_SD": np.nan,
                    "RL100_RMSE": np.nan,
                    "RL100_fold_RMSE_SD": np.nan,
                }
            )
            print(
                f"[{position}/{len(k_values)}] K={candidate_k}: infeasible ({error})",
                flush=True,
            )

    summary = pd.DataFrame(rows).sort_values("K")
    summary_path = output_directory / "experimental_k_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    feasible = summary.query("feasible == True")
    if not feasible.empty:
        figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
        for return_period, color in ((50, "tab:blue"), (100, "tab:orange")):
            axes[0].plot(
                feasible["K"],
                feasible[f"RL{return_period}_RMSE"],
                "o-",
                color=color,
                label=fr"$RL_{{{return_period}}}$",
            )
            axes[1].plot(
                feasible["K"],
                feasible[f"RL{return_period}_fold_RMSE_SD"],
                "o-",
                color=color,
                label=fr"$RL_{{{return_period}}}$",
            )
        axes[0].set(
            title="Pooled OOF prediction error", xlabel="K", ylabel="RMSE"
        )
        axes[1].set(
            title="Fold-level error stability",
            xlabel="K",
            ylabel="SD of fold RMSE",
        )
        for axis in axes:
            axis.set_xticks(k_values)
            axis.legend()
        figure.suptitle("Experimental buffered Spatial-CV sensitivity to K")
        figure.tight_layout()
        figure.savefig(figure_path, bbox_inches="tight", dpi=180)
        plt.close(figure)

    print(f"Summary: {summary_path}", flush=True)
    print(f"Figure:  {figure_path}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-values", nargs="+", type=int, default=[3, 4, 5, 6, 7])
    parser.add_argument("--max-train", type=int, default=800)
    parser.add_argument("--min-train", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=111)
    parser.add_argument("--n-jobs", type=int, default=-2)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute a K even when its required output files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        k_values=tuple(args.k_values),
        output_directory=args.output_directory,
        figure_path=args.figure_path,
        max_train=args.max_train,
        min_train=args.min_train,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        force=args.force,
    )


if __name__ == "__main__":
    main()
