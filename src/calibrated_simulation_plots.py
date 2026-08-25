"""Presentation figures from the fresh monthly calibrated-simulation OOF run."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_paths import REPOSITORY_ROOT


SIMULATION_DIR = (
    REPOSITORY_ROOT / "data" / "simulated" / "calibrated_final_model"
)
DEFAULT_MODEL_READY = SIMULATION_DIR / "replicate_000_model_ready.csv"
DEFAULT_CV_DIR = SIMULATION_DIR / "nested_spatial_cv_monthly"
DEFAULT_FIGURE_DIR = Path.home() / "Desktop" / "picture"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified_monthly_outputs(
    model_ready_path: str | Path = DEFAULT_MODEL_READY,
    cv_directory: str | Path = DEFAULT_CV_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load OOF tables only when metadata matches the current monthly input."""
    model_ready_path = Path(model_ready_path)
    cv_directory = Path(cv_directory)
    model_ready = pd.read_csv(model_ready_path)
    metadata = pd.read_csv(cv_directory / "calibrated_nested_metadata.csv")
    predictions = pd.read_csv(cv_directory / "calibrated_nested_predictions.csv")
    return_levels = pd.read_csv(
        cv_directory / "calibrated_nested_return_level_predictions.csv"
    )
    expected_hash = _sha256(model_ready_path)
    recorded_hash = str(metadata.loc[0, "input_sha256"])
    if recorded_hash != expected_hash:
        raise ValueError("OOF output is stale: input SHA-256 does not match.")
    if str(metadata.loc[0, "block_scale"]).lower() != "monthly":
        raise ValueError("OOF output is not based on monthly maxima.")
    if int(metadata.loc[0, "n_months"]) != 540:
        raise ValueError("OOF output is not based on 540 monthly maxima.")
    return model_ready, predictions, return_levels


def plot_parameter_recovery(
    model_ready: pd.DataFrame,
    predictions: pd.DataFrame,
) -> plt.Figure:
    """Plot truth and nested OOF GP parameter predictions in a 3-by-2 layout."""
    data = predictions.merge(
        model_ready[["station", "x_km", "y_km"]],
        on="station",
        how="left",
        validate="many_to_one",
    )
    targets = (("mu", r"$\mu$"), ("log_sigma", r"$\log\sigma$"), ("xi", r"$\xi$"))
    columns = (
        ("true_value", "Truth"),
        ("oof_prediction", "Nested OOF GP"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(8, 14), constrained_layout=True)
    for row, (target, label) in enumerate(targets):
        part = data.loc[data["target"].eq(target)]
        values = np.concatenate([part[column].to_numpy(float) for column, _ in columns])
        vmin, vmax = np.nanquantile(values, [0.01, 0.99])
        for column_index, (column, title) in enumerate(columns):
            axis = axes[row, column_index]
            points = axis.scatter(
                part["x_km"], part["y_km"], c=part[column], s=10,
                cmap="viridis", vmin=vmin, vmax=vmax,
            )
            axis.set_title(f"{label}: {title}")
            axis.set_aspect("equal")
            figure.colorbar(points, ax=axis, shrink=0.75)
    figure.suptitle("Monthly simulation: parameter recovery", fontsize=15)
    return figure


def plot_return_level_recovery(
    model_ready: pd.DataFrame,
    return_levels: pd.DataFrame,
) -> plt.Figure:
    """Plot truth and OOF annual RL50/RL100 in a 2-by-2 layout."""
    data = return_levels.merge(
        model_ready[["station", "x_km", "y_km"]],
        on="station",
        how="left",
        validate="many_to_one",
    )
    columns = (
        ("true_return_level", "Truth"),
        ("oof_return_level", "Nested OOF GP"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(8, 9), constrained_layout=True)
    for row, period in enumerate((50, 100)):
        part = data.loc[data["return_period"].eq(period)]
        values = np.concatenate([part[column].to_numpy(float) for column, _ in columns])
        vmin, vmax = np.nanquantile(values, [0.01, 0.99])
        for column_index, (column, title) in enumerate(columns):
            axis = axes[row, column_index]
            points = axis.scatter(
                part["x_km"], part["y_km"], c=part[column], s=10,
                cmap="magma", vmin=vmin, vmax=vmax,
            )
            axis.set_title(f"Annual RL{period}: {title}")
            axis.set_aspect("equal")
            figure.colorbar(points, ax=axis, shrink=0.75)
    figure.suptitle("Monthly simulation: annual return-level recovery", fontsize=15)
    return figure


def save_verified_figures(
    figure_directory: str | Path = DEFAULT_FIGURE_DIR,
) -> list[Path]:
    """Verify provenance, then write both OOF figures."""
    figure_directory = Path(figure_directory)
    figure_directory.mkdir(parents=True, exist_ok=True)
    model_ready, predictions, return_levels = load_verified_monthly_outputs()
    figures = {
        "calibrated_simulation_oof_parameter_recovery.png":
            plot_parameter_recovery(model_ready, predictions),
        "calibrated_simulation_oof_return_level_recovery.png":
            plot_return_level_recovery(model_ready, return_levels),
    }
    written = []
    for name, figure in figures.items():
        path = figure_directory / name
        figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-directory", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()
    for path in save_verified_figures(args.figure_directory):
        print(path)


if __name__ == "__main__":
    main()
