"""Small, deterministic diagnostics tests; no neural-network training."""

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cheng_nn_simulation import generate_all_splits, SimulationConfig
from cheng_nn_diagnostics import (
    resolve_run_directory, load_diagnostic_split, load_saved_predictions,
    error_summary, coefficient_metrics, evaluate_gev_and_rl,
    validity_by_shape, binned_rl_errors, plot_distributions,
    plot_learning_curves, plot_prediction_details, plot_rl_by_year,
    plot_validity_by_shape, plot_binned_rl_errors, plot_coefficient_comparison,
    file_sha256,
)


@pytest.fixture
def diagnostic_run(tmp_path):
    project = tmp_path / "project"
    data_dir = project / "data" / "simulated" / "cheng_nn_17d"
    generate_all_splits(data_dir, n_train=12, n_validation=8, n_test=6,
                        config=SimulationConfig(seed=84))
    run_dir = project / "results" / "cheng_nn_17d" / "example"
    run_dir.mkdir(parents=True)
    hashes = {}
    for split in ("train", "validation", "test"):
        data = load_diagnostic_split(data_dir, split)
        directory = run_dir / split
        directory.mkdir()
        np.save(directory / "predictions_standardized.npy", data["targets"])
        np.save(directory / "predictions_original.npy", data["coefficients"])
        hashes[split] = data["metadata_sha256"]
        (directory / "prediction_metadata.json").write_text(json.dumps({
            "checkpoint": "best.pt", "best_epoch": 2,
            "n_samples": len(data["inputs"]),
            "dataset_metadata_sha256": data["metadata_sha256"],
        }), encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(json.dumps({
        "status": "completed", "data_directory": str(data_dir),
        "best_epoch": 2, "dataset_metadata_sha256": hashes,
    }), encoding="utf-8")
    (run_dir.parent / "latest_run.json").write_text(json.dumps({"run_dir": str(run_dir)}), encoding="utf-8")
    pd.DataFrame({"epoch": [1, 2], "train_loss": [1.0, 0.8],
                  "validation_loss": [1.1, 0.9], "learning_rate": [0.001, 0.001],
                  "elapsed_seconds": [2.0, 4.0]}).to_csv(run_dir / "training_history.csv", index=False)
    return project, data_dir, run_dir


def test_resolves_recorded_run_and_rejects_unrelated_latest_pointer(diagnostic_run):
    project, _, run_dir = diagnostic_run
    assert resolve_run_directory(project) == run_dir.resolve()
    assert resolve_run_directory(project, run_dir) == run_dir.resolve()
    pointer = run_dir.parent / "latest_run.json"
    pointer.write_text(json.dumps({"run_dir": str(project)}), encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        resolve_run_directory(project)


def test_missing_run_is_graceful_but_explicit_missing_path_fails(tmp_path):
    assert resolve_run_directory(tmp_path) is None
    with pytest.raises(FileNotFoundError):
        resolve_run_directory(tmp_path, "not_here")


def test_loader_requires_matching_atomic_marker(diagnostic_run):
    _, data_dir, run_dir = diagnostic_run
    data = load_diagnostic_split(data_dir, "train")
    assert load_saved_predictions(run_dir, "train", data)["original"].shape == (12, 5)
    assert data["metadata_sha256"] == hashlib.sha256((data_dir / "train" / "metadata.json").read_bytes()).hexdigest()
    marker = run_dir / "train" / "prediction_metadata.json"
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    metadata["dataset_metadata_sha256"] = "other-data"
    marker.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="hashes"):
        load_saved_predictions(run_dir, "train", data)
    marker.unlink()
    assert load_saved_predictions(run_dir, "train", data) is None


def test_loader_rejects_mismatched_epoch_and_half_pair(diagnostic_run):
    _, data_dir, run_dir = diagnostic_run
    data = load_diagnostic_split(data_dir, "validation")
    marker = run_dir / "validation" / "prediction_metadata.json"
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    metadata["best_epoch"] = 999
    marker.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="best epoch"):
        load_saved_predictions(run_dir, "validation", data)
    metadata["best_epoch"] = 2
    marker.write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "validation" / "predictions_original.npy").unlink()
    with pytest.raises(ValueError, match="missing"):
        load_saved_predictions(run_dir, "validation", data)


def test_checks_data_bytes_and_checkpoint_hash(diagnostic_run):
    _, data_dir, run_dir = diagnostic_run
    split_dir = data_dir / "validation"
    hashes = {path.name: file_sha256(path) for path in split_dir.glob("*.npy")}
    data = load_diagnostic_split(data_dir, "validation", hashes)
    wrong = dict(hashes)
    wrong["inputs.npy"] = "incorrect"
    with pytest.raises(ValueError, match="file hash"):
        load_diagnostic_split(data_dir, "validation", wrong)
    checkpoint = run_dir / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint bytes")
    marker = run_dir / "validation" / "prediction_metadata.json"
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    metadata["checkpoint_sha256"] = file_sha256(checkpoint)
    marker.write_text(json.dumps(metadata), encoding="utf-8")
    assert load_saved_predictions(run_dir, "validation", data) is not None
    checkpoint.write_bytes(b"different checkpoint bytes")
    with pytest.raises(ValueError, match="checkpoint hash"):
        load_saved_predictions(run_dir, "validation", data)


def test_error_summary_preserves_exclusion_counts():
    result = error_summary([1, 2, np.nan, np.inf], [0, 0, 0, 0])
    assert result["n_total"] == 4 and result["n_used"] == 2
    assert result["n_excluded"] == 2
    assert result["RMSE"] == pytest.approx(np.sqrt(2.5))
    assert result["MAE"] == result["bias"] == 1.5
    assert error_summary([0], [0])["RMSE"] == 0
    assert np.isnan(error_summary([np.nan], [0])["RMSE"])


def test_perfect_predictions_have_zero_error_and_all_bins_counted(diagnostic_run):
    _, data_dir, run_dir = diagnostic_run
    data = load_diagnostic_split(data_dir, "validation")
    prediction = load_saved_predictions(run_dir, "validation", data)
    metrics = coefficient_metrics({"validation": data}, {"validation": prediction})
    assert (metrics[["RMSE", "MAE", "bias"]] == 0).all().all()
    validity, yearly, overall = evaluate_gev_and_rl(data, prediction)
    assert validity["gev_valid_all_years"].all()
    assert (yearly["RMSE"] == 0).all()
    assert (overall["n_total"] == 8 * 50).all()
    shape = validity_by_shape(validity, data["coefficients"][:, 4], np.linspace(-0.4, 0.4, 9))
    assert shape["n_total"].sum() == 8
    bins = binned_rl_errors(data, prediction)
    assert (bins.groupby(["factor", "return_period"])["n_total"].sum() == 8).all()
    assert (bins.loc[bins["n_used"] > 0, "RMSE"] == 0).all()


def test_shape_bin_edges_include_rightmost_endpoint():
    validity = pd.DataFrame({"gev_valid_all_years": [True, False, True],
                             "n_support_violations": [0, 1, 0],
                             "xi_outside_training_range": [False] * 3})
    table = validity_by_shape(validity, [-0.4, 0.0, 0.4], [-0.4, 0.0, 0.4])
    assert table["n_total"].tolist() == [1, 2]
    with pytest.raises(ValueError, match="cover"):
        validity_by_shape(validity, [-0.4, 0, 0.5], [-0.4, 0.0, 0.4])


def test_plots_work_on_small_saved_run(diagnostic_run):
    _, data_dir, run_dir = diagnostic_run
    data = load_diagnostic_split(data_dir, "validation")
    prediction = load_saved_predictions(run_dir, "validation", data)
    validity, yearly, _ = evaluate_gev_and_rl(data, prediction)
    metrics = coefficient_metrics({"validation": data}, {"validation": prediction})
    shape = validity_by_shape(validity, data["coefficients"][:, 4], np.linspace(-0.4, 0.4, 9)).assign(split="validation")
    figures = [
        plot_distributions({"validation": data}),
        plot_distributions({"validation": data}, "targets"),
        plot_learning_curves(pd.read_csv(run_dir / "training_history.csv"), 2),
        plot_prediction_details(prediction["original"], data["coefficients"]),
        plot_coefficient_comparison(metrics),
        plot_validity_by_shape(shape),
        plot_rl_by_year(yearly.assign(split="validation")),
        plot_binned_rl_errors(binned_rl_errors(data, prediction).assign(split="validation")),
    ]
    for figure in figures:
        figure.canvas.draw()
        assert len(figure.axes) >= 2
        plt.close(figure)


def test_actual_notebook_executes_without_reading_test(diagnostic_run, monkeypatch):
    project, _, run_dir = diagnostic_run
    (project / "src").mkdir()
    (project / "src" / "cheng_nn_simulation.py").touch()
    monkeypatch.chdir(project)
    notebook = json.loads((ROOT / "notebooks" / "cheng_NN_diagnostics.ipynb").read_text(encoding="utf-8"))
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(compile(source, cell["id"], "exec"), namespace)
    assert namespace["SHOW_TEST"] is False
    assert set(namespace["data_by_split"]) == {"train", "validation"}
    assert set(namespace["predictions_by_split"]) == {"train", "validation"}
    assert (run_dir / "figures" / "diagnostic_learning_curves.png").exists()
    assert (run_dir / "diagnostics_tables" / "coefficient_metrics.csv").exists()
    assert not (run_dir / "diagnostics_tables" / "test_gev_validity.csv").exists()
