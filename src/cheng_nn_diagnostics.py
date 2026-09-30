"""Read-only model/data diagnostics for saved Cheng NN training runs.

Plotting never trains a model, changes simulated data, clips predictions, or
silently falls back to another run. Test predictions are opt-in at the caller.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cheng_nn_evaluation import (
    check_gev_predictions, conditional_return_levels,
    return_level_recovery_metrics,
)
from cheng_nn_simulation import (
    COEFFICIENT_NAMES, DATA_SCHEMA_VERSION, INPUT_FEATURE_NAMES,
)

SPLIT_COLORS = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:green"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_run_directory(project_root, run_directory=None):
    """Resolve an explicit run or the recorded latest run, not directory mtime."""
    project_root = Path(project_root).resolve()
    if run_directory is not None:
        path = Path(run_directory)
        path = path if path.is_absolute() else project_root / path
    else:
        root = project_root / "results" / "cheng_nn_17d"
        pointer = root / "latest_run.json"
        if not pointer.exists():
            return None
        path = Path(json.loads(pointer.read_text(encoding="utf-8"))["run_dir"])
        if not path.is_absolute():
            path = project_root / path
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("latest_run.json points outside results/cheng_nn_17d.")
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"The selected run does not exist: {path}")
    return path


def load_diagnostic_split(data_directory, split, expected_file_hashes=None):
    """Load aligned simulation arrays without changing files or their scale."""
    if split not in SPLIT_COLORS:
        raise ValueError("Unknown split; use train, validation, or test.")
    directory = Path(data_directory) / split
    metadata_bytes = (directory / "metadata.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    if (metadata.get("data_schema_version") != DATA_SCHEMA_VERSION
            or metadata.get("input_columns") != list(INPUT_FEATURE_NAMES)
            or metadata.get("target_columns") != list(COEFFICIENT_NAMES)
            or metadata.get("split") != split):
        raise ValueError(f"{split}: incompatible simulation metadata.")
    n, n_years = metadata["n_samples"], len(metadata["years"])
    files = {
        "inputs": ("inputs.npy", (n, 17)),
        "targets": ("targets_standardized.npy", (n, 5)),
        "coefficients": ("coefficients_original.npy", (n, 5)),
        "location_scale": ("sample_location_scale.npy", (n, 2)),
        "annual_maxima": ("annual_maxima.npy", (n, n_years)),
    }
    data = {"metadata": metadata,
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "directory": directory.resolve()}
    for key, (filename, expected_shape) in files.items():
        if expected_file_hashes is not None:
            if filename not in expected_file_hashes:
                raise ValueError(f"{split}: run has no saved hash for {filename}.")
            if file_sha256(directory / filename) != expected_file_hashes[filename]:
                raise ValueError(f"{split}/{filename}: file hash differs from this training run.")
        array = np.load(directory / filename, mmap_mode="r", allow_pickle=False)
        if array.shape != expected_shape:
            raise ValueError(f"{split}/{filename}: expected {expected_shape}, got {array.shape}.")
        data[key] = array
    return data


def load_saved_predictions(run_directory, split, data):
    """Return predictions only for a committed, matching split; otherwise None.

    prediction_metadata.json is written last by the training runner. This
    marker prevents plots from mixing a half-written prediction pair or a
    different simulated dataset with the current labels.
    """
    directory = Path(run_directory) / split
    marker = directory / "prediction_metadata.json"
    if not marker.exists():
        return None
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if metadata.get("checkpoint") != "best.pt":
        raise ValueError(f"{split}: predictions must refer to the selected best.pt checkpoint.")
    if "checkpoint_sha256" in metadata:
        checkpoint = Path(run_directory) / "best.pt"
        if not checkpoint.exists() or file_sha256(checkpoint) != metadata["checkpoint_sha256"]:
            raise ValueError(f"{split}: best.pt checkpoint hash differs from the prediction marker.")
    if metadata.get("dataset_metadata_sha256") != data["metadata_sha256"]:
        raise ValueError(f"{split}: prediction and simulation metadata hashes differ.")
    if metadata.get("n_samples") != len(data["inputs"]):
        raise ValueError(f"{split}: prediction sample count differs from the dataset.")
    run_metadata_path = Path(run_directory) / "run_metadata.json"
    if run_metadata_path.exists():
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        if ("best_epoch" in run_metadata and "best_epoch" in metadata
                and run_metadata["best_epoch"] != metadata["best_epoch"]):
            raise ValueError(f"{split}: predictions do not use this run's best epoch.")
    predictions = {}
    for scale in ("standardized", "original"):
        path = directory / f"predictions_{scale}.npy"
        if not path.exists():
            raise ValueError(f"{split}: completion marker exists but {path.name} is missing.")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (len(data["inputs"]), 5):
            raise ValueError(f"{split}: wrong {scale} prediction shape.")
        predictions[scale] = array
    return predictions


def error_summary(predicted, truth):
    """Visible finite-pair counts; zero valid predictions never means zero error."""
    predicted, truth = np.broadcast_arrays(
        np.asarray(predicted, dtype=float), np.asarray(truth, dtype=float)
    )
    with np.errstate(over="ignore", invalid="ignore"):
        error = predicted - truth
    use = np.isfinite(predicted) & np.isfinite(truth) & np.isfinite(error)
    error = error[use]
    scale = np.max(np.abs(error)) if error.size else np.nan
    relative = error / scale if scale > 0 else error
    return {
        "n_total": predicted.size, "n_used": len(error),
        "n_excluded": predicted.size - len(error),
        "RMSE": float(scale * np.sqrt(np.mean(relative ** 2))) if error.size else np.nan,
        "MAE": float(scale * np.mean(np.abs(relative))) if error.size else np.nan,
        "bias": float(scale * np.mean(relative)) if error.size else np.nan,
    }


def coefficient_metrics(data_by_split, predictions_by_split):
    rows = []
    for split, prediction in predictions_by_split.items():
        for scale, key in (("standardized", "targets"), ("original", "coefficients")):
            for j, name in enumerate(COEFFICIENT_NAMES):
                rows.append({"split": split, "scale": scale, "coefficient": name,
                             **error_summary(prediction[scale][:, j], data_by_split[split][key][:, j])})
    return pd.DataFrame(rows)


def _sample_indices(n, max_points, seed):
    return np.sort(np.random.default_rng(seed).choice(n, min(n, max_points), replace=False))


def plot_distributions(data_by_split, key="inputs", max_points=10_000):
    names = INPUT_FEATURE_NAMES if key == "inputs" else COEFFICIENT_NAMES
    if key not in ("inputs", "targets", "coefficients"):
        raise ValueError("Use inputs, targets, or coefficients for distributions.")
    ncols = 4 if key == "inputs" else 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.7 * nrows), squeeze=False, constrained_layout=True)
    samples = {split: np.asarray(data[key][_sample_indices(len(data[key]), max_points, 31)])
               for split, data in data_by_split.items()}
    for j, (name, ax) in enumerate(zip(names, axes.flat)):
        values = np.concatenate([sample[:, j] for sample in samples.values()])
        values = values[np.isfinite(values)]
        if key == "inputs" and name == "q_0.5":
            ax.axvline(0, color="black")
            ax.text(0.5, 0.7, "Median = 0 by normalization", ha="center", transform=ax.transAxes)
            ax.set_xlim(-1, 1)
        elif values.size:
            edges = np.histogram_bin_edges(values, bins=40)
            for split, sample in samples.items():
                finite = sample[np.isfinite(sample[:, j]), j]
                if finite.size:
                    ax.hist(finite, bins=edges, density=True, histtype="step", linewidth=1.5,
                            color=SPLIT_COLORS[split], label=f"{split} (n={len(finite):,})")
        ax.set_title(name)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{key}: split distributions (up to {max_points:,} sequences / split)")
    return fig


def plot_learning_curves(history, best_epoch=None):
    required = {"epoch", "train_loss", "validation_loss"}
    if not required.issubset(history.columns):
        raise ValueError(f"History must contain {sorted(required)}.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for column, label in (("train_loss", "Train (online)"), ("validation_loss", "Validation")):
        axes[0].plot(history["epoch"], history[column], label=label)
    if best_epoch is not None:
        axes[0].axvline(best_epoch, color="black", linestyle="--", label="Best validation epoch")
    axes[0].set(xlabel="Epoch", ylabel="Target-scaled MSE", title="Training / validation loss")
    axes[0].legend()
    lr_column = "lr" if "lr" in history else "learning_rate"
    if lr_column in history:
        axes[1].plot(history["epoch"], history[lr_column])
        axes[1].set_yscale("log")
    else:
        axes[1].text(0.5, 0.5, "Learning rate not recorded", ha="center", transform=axes[1].transAxes)
    axes[1].set(xlabel="Epoch", ylabel="Learning rate", title="Adam learning-rate schedule")
    for ax in axes:
        ax.grid(alpha=0.25)
    return fig


def plot_coefficient_comparison(metrics, scale="original"):
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), constrained_layout=True)
    for ax, name in zip(axes, COEFFICIENT_NAMES):
        part = metrics.loc[metrics["scale"].eq(scale) & metrics["coefficient"].eq(name)]
        ax.bar(part["split"], part["RMSE"], color=[SPLIT_COLORS[s] for s in part["split"]])
        ax.set(title=name, ylabel="RMSE")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"Best checkpoint, final evaluation: {scale} coefficient RMSE")
    return fig


def plot_prediction_details(prediction, truth, split="validation", max_points=5000):
    """Scatter and residual panels use an identified random display subsample."""
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    indices = _sample_indices(len(truth), max_points, 42)
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    for j, name in enumerate(COEFFICIENT_NAMES):
        use = indices[np.isfinite(prediction[indices, j]) & np.isfinite(truth[indices, j])]
        x, y = truth[use, j], prediction[use, j]
        ax = axes[0, j]
        ax.scatter(x, y, s=5, alpha=0.15, rasterized=True)
        if len(x):
            low, high = min(x.min(), y.min()), max(x.max(), y.max())
            ax.plot([low, high], [low, high], "k--", linewidth=1)
            axes[1, j].hist(y - x, bins=40, color=SPLIT_COLORS[split], alpha=0.8)
        ax.set(title=name, xlabel="Truth", ylabel="Prediction")
        axes[1, j].axvline(0, color="black", linestyle="--")
        axes[1, j].set(xlabel="Prediction - truth", ylabel="Count")
        axes[1, j].text(0.03, 0.94, f"Displayed n={len(use):,}", va="top", transform=axes[1, j].transAxes)
    fig.suptitle(f"{split}: original-scale coefficient recovery")
    return fig


def evaluate_gev_and_rl(data, predictions):
    """Evaluate all sequences, not the visualization subsample."""
    metadata = data["metadata"]
    years = np.asarray(metadata["years"])
    options = {"reference_year": metadata["reference_year"],
               "time_scale_years": metadata["time_scale_years"]}
    shape_range = tuple(metadata["parameter_ranges"]["xi0"])
    validity = check_gev_predictions(predictions["original"], data["annual_maxima"], years,
                                    xi_training_range=shape_range, **options)
    truth_rl = conditional_return_levels(data["coefficients"], years, **options)
    predicted_rl = conditional_return_levels(predictions["original"], years, **options)
    if not np.isfinite(truth_rl).all():
        raise ValueError("Stored truth produces invalid return levels; inspect simulation data.")
    yearly, overall = return_level_recovery_metrics(
        predicted_rl, truth_rl, years, (50, 100), validity["gev_valid_all_years"].to_numpy()
    )
    return validity, yearly, overall


def _bin_masks(values, edges):
    values, edges = np.asarray(values), np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("Bin edges must be strictly increasing and finite.")
    if np.any(np.isfinite(values) & ((values < edges[0]) | (values > edges[-1]))):
        raise ValueError("Bin edges do not cover all finite values.")
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = np.isfinite(values) & (values >= left) & ((values <= right) if i == len(edges) - 2 else (values < right))
        yield left, right, mask


def validity_by_shape(validity, true_shape, edges):
    rows = []
    for left, right, use in _bin_masks(true_shape, edges):
        part = validity.loc[use]
        n = len(part)
        rows.append({"left": left, "right": right, "midpoint": (left + right) / 2,
                     "n_total": n,
                     "valid_fraction": float(part["gev_valid_all_years"].mean()) if n else np.nan,
                     "support_failure_fraction": float((part["n_support_violations"] > 0).mean()) if n else np.nan,
                     "out_of_range_fraction": float(part["xi_outside_training_range"].mean()) if n else np.nan})
    return pd.DataFrame(rows)


def plot_validity_by_shape(table):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for split, part in table.groupby("split", sort=False):
        axes[0].plot(part["midpoint"], part["valid_fraction"], "o-", color=SPLIT_COLORS[split], label=split)
        axes[1].plot(part["midpoint"], part["n_total"], "o-", color=SPLIT_COLORS[split], label=split)
    axes[0].set(xlabel="True xi0", ylabel="Valid fraction", ylim=(-0.02, 1.02), title="All 50 observations inside predicted support")
    axes[1].set(xlabel="True xi0", ylabel="Sequence count", title="Number of sequences in each shape bin")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
    return fig


def plot_rl_by_year(table):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, period in zip(axes, (50, 100)):
        part = table.loc[table["return_period"].eq(period)]
        for (split, subset), group in part.groupby(["split", "subset"], sort=False):
            ax.plot(group["year"], group["RMSE"], "--" if subset == "valid_GEV" else "-",
                    color=SPLIT_COLORS[split], label=f"{split}: {subset}")
        ax.set(title=f"Conditional RL{period}(t)", xlabel="Year", ylabel="RMSE (degrees C)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    return fig


def binned_rl_errors(data, predictions, n_bins=8):
    """End-year conditional RL errors versus the independently sampled factors."""
    metadata = data["metadata"]
    year = metadata["years"][-1]
    options = {"reference_year": metadata["reference_year"], "time_scale_years": metadata["time_scale_years"]}
    truth = conditional_return_levels(data["coefficients"], [year], **options)[:, 0, :]
    predicted = conditional_return_levels(predictions["original"], [year], **options)[:, 0, :]
    coefficients = np.asarray(data["coefficients"])
    ranges = metadata["parameter_ranges"]
    factors = {"xi0": coefficients[:, 4],
               "beta_mu_over_sigma0": coefficients[:, 1] / np.exp(coefficients[:, 2]),
               "beta_sigma": coefficients[:, 3]}
    rows = []
    for factor, values in factors.items():
        edges = np.linspace(*ranges[factor], n_bins + 1)
        for left, right, use in _bin_masks(values, edges):
            for j, period in enumerate((50, 100)):
                rows.append({"factor": factor, "left": left, "right": right,
                             "midpoint": (left + right) / 2, "year": year,
                             "return_period": period,
                             **error_summary(predicted[use, j], truth[use, j])})
    return pd.DataFrame(rows)


def plot_binned_rl_errors(table):
    factors = ("xi0", "beta_mu_over_sigma0", "beta_sigma")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for row, factor in enumerate(factors):
        for col, period in enumerate((50, 100)):
            ax = axes[row, col]
            part = table.loc[table["factor"].eq(factor) & table["return_period"].eq(period)]
            for split, group in part.groupby("split", sort=False):
                ax.plot(group["midpoint"], group["RMSE"], "o-", color=SPLIT_COLORS[split], label=split)
                for record in group.itertuples():
                    if np.isfinite(record.RMSE):
                        ax.annotate(f"{record.n_used}/{record.n_total}", (record.midpoint, record.RMSE),
                                    textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6)
            ax.set(xlabel=f"True {factor}", ylabel="RMSE (degrees C)", title=f"RL{period} at {int(table['year'].iloc[0])}; labels: used / total")
            ax.margins(x=0.075, y=0.15)
            ax.legend()
            ax.grid(alpha=0.25)
    return fig
