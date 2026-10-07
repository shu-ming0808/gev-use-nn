"""Evaluate the six preselected large/small L2 checkpoints without retraining.

The architecture comparison and checkpoint hashes are frozen before loading
test arrays. This writes a separate test report and never edits source runs,
fits a new scaler, publishes a model, or updates latest_run.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from compare_cheng_nn_architectures import validate_runs
from train_cheng_nn import (
    PROJECT_ROOT, atomic_csv, atomic_json, export_predictions,
    file_sha256, load_notebook_components,
)

PERIODS = (20., 50., 100.)
SPLITS = ("train", "validation", "test")
DESIGN_FIELDS = (
    "data_schema_version", "sampling_scheme", "years", "reference_year",
    "time_scale_years", "input_columns", "parameter_ranges", "target_columns",
    "input_standardization", "target_scale", "input_shape_per_sample",
)


def validate_design(metadata):
    reference = metadata["train"]
    seeds = []
    for split in SPLITS:
        item = metadata[split]
        if item["split"] != split:
            raise ValueError("Dataset split label mismatch")
        if any(item[k] != reference[k] for k in DESIGN_FIELDS):
            raise ValueError("Dataset simulation designs differ")
        seeds.append(item["seed"])
    if len(set(seeds)) != len(SPLITS):
        raise ValueError("Train/validation/test must use distinct generation seeds")


def collect_metrics(directory, model, seed, split):
    """Reject nonfinite/excluded cases rather than improving RMSE by omission."""
    coefficients = pd.read_csv(directory / split / "coefficient_metrics.csv")
    original = coefficients.loc[coefficients.scale.eq("original")]
    scaled = coefficients.loc[coefficients.scale.eq("train_target_z")]
    names = {"mu0", "beta_mu", "eta0", "beta_sigma", "xi0"}
    for part in (original, scaled):
        if len(part) != 5 or set(part.coefficient) != names or not (part.n_used == part.n_total).all():
            raise ValueError("Missing or excluded coefficient predictions")
        if not np.isfinite(part.RMSE).all():
            raise ValueError("Nonfinite coefficient RMSE")
    values = dict(zip(original.coefficient, original.RMSE))
    values["coefficients_train_z"] = float(np.sqrt(np.mean(scaled.RMSE.to_numpy() ** 2)))
    rl = pd.read_csv(directory / split / "return_level_overall.csv")
    for period in PERIODS:
        part = rl.loc[rl.subset.eq("finite_RL") & rl.return_period.eq(period)]
        if len(part) != 1 or not (part.n_used == part.n_total).all() or not np.isfinite(part.RMSE).all():
            raise ValueError("Missing or excluded return-level predictions")
        values[f"RL{int(period)}"] = float(part.RMSE.iloc[0])
    return [dict(model=model, seed=seed, split=split, outcome=k, RMSE=float(v)) for k, v in values.items()]


def summarize(rows):
    frame = pd.DataFrame(rows)
    keys = ["model", "seed", "split", "outcome"]
    if frame.duplicated(keys).any() or not np.isfinite(frame.RMSE).all():
        raise ValueError("Duplicated or nonfinite metrics")
    paired = frame.pivot(index=["model", "seed", "outcome"], columns="split", values="RMSE")
    if any(s not in paired.columns for s in SPLITS) or paired[list(SPLITS)].isna().any().any():
        raise ValueError("Every model/seed/outcome needs all three splits")
    summary = frame.groupby(["model", "split", "outcome"]).RMSE.agg(["mean", "std", "count"]).reset_index()
    summary = summary.rename(columns={"mean": "RMSE_mean", "std": "RMSE_seed_SD", "count": "n_seeds"})
    if (paired[["train", "validation"]] <= 0).any().any():
        raise ValueError("Cannot calculate relative gaps with zero baseline RMSE")
    paired["test_vs_train_percent"] = 100 * (paired.test / paired.train - 1)
    paired["test_vs_validation_percent"] = 100 * (paired.test / paired.validation - 1)
    gaps = paired.reset_index().groupby(["model", "outcome"])[
        ["test_vs_train_percent", "test_vs_validation_percent"]].mean().reset_index()
    return frame, summary, gaps


def render_notebook(output):
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            "# 三層與五層 L2 模型：獨立 test 評估\n\n"
            "固定先前架構比較的 6 個 best.pt（各 3 個 seed），不重訓、不改 scaler、不依 test 選 checkpoint。"
            "共用同一份 10,000 組 test，並非 30,000 組獨立 test。"
            "每個 checkpoint 用 eval 模式重新計算 train／validation／test。\n\n"
            "五係數使用原尺度 RMSE；RL20／RL50／RL100 為 50 個年份的條件式年分布分位數，"
            "單位 °C，合併序列與年份計算 RMSE。GEV support 違反不從主要誤差中刪除。"
            "平均與 SD 表示三個訓練 seed 的變動，不是信賴區間或顯著性檢定。\n\n"
            "Test 可檢查同一模擬設定下的泛化，不能證明完全沒有過擬合，也不是全球 ERA5 的實證驗證。"
            "若看完 test 後繼續調整模型，需另保留新的最終確認資料。"),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display\n"
            f"RESULT = Path({str(output)!r})\n"
            "summary = pd.read_csv(RESULT / 'test_comparison_summary.csv')\n"
            "gaps = pd.read_csv(RESULT / 'generalization_gaps.csv')\n"
            "order = ['mu0','beta_mu','eta0','beta_sigma','xi0','RL20','RL50','RL100','coefficients_train_z']\n"
            "table = summary.pivot(index='outcome', columns=['model','split'], values='RMSE_mean').reindex(order)\n"
            "display(table.round(5))"),
        nbformat.v4.new_markdown_cell("## Test RMSE 與 seed 變動\n\n請比較同一個目標，不跨不同單位的係數比較 RMSE。"),
        nbformat.v4.new_code_cell(
            "part = summary.loc[summary.split.eq('test')].copy()\n"
            "part['mean ± SD'] = part.apply(lambda r: f'{r.RMSE_mean:.5f} ± {r.RMSE_seed_SD:.5f}', axis=1)\n"
            "display(part.pivot(index='outcome', columns='model', values='mean ± SD').reindex(order))\n"
            "display(gaps.round(3))\n"
            "display(pd.read_csv(RESULT / 'validity_by_split.csv').round(6))"),
    ]
    NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(PROJECT_ROOT)}}).execute()
    path = output / "cheng_NN_test_comparison.ipynb"
    nbformat.write(nb, path)
    return path


def evaluate(experiment, output, device="cuda"):
    manifest, runs = validate_runs(experiment)
    if manifest.get("status") != "completed" or len(runs) != 6:
        raise ValueError("Need the completed six-run architecture comparison")
    output = Path(output).resolve()
    root = Path(runs[0][2]["data_directory"]).resolve()
    metadata = {s: json.loads((root / s / "metadata.json").read_text(encoding="utf-8")) for s in SPLITS}
    validate_design(metadata)
    hashes = {s: {p.name: file_sha256(p) for p in sorted((root / s).glob("*.npy"))} for s in SPLITS}
    meta_hashes = {s: file_sha256(root / s / "metadata.json") for s in SPLITS}
    for _, _, meta in runs:
        for split in ("train", "validation"):
            if hashes[split] != meta["dataset_files_sha256"][split] or meta_hashes[split] != meta["dataset_metadata_sha256"][split]:
                raise ValueError("Training/validation data changed since fitting")
        if meta["optimizer"] != "Adam" or meta["weight_decay"] != 1e-4:
            raise ValueError("Expected Adam with L2=1e-4")
    protocol = {
        "status": "locked_before_test_inference", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment": str(Path(experiment).resolve()), "device": device,
        "data_directory": str(root), "data_files_sha256": hashes, "metadata_sha256": meta_hashes,
        "split_sizes": {s: metadata[s]["n_samples"] for s in SPLITS},
        "return_periods": list(PERIODS), "checkpoint_selection": "Original validation coefficient MSE only",
        "test_used_for_training_or_selection": False,
        "runs": [{"model": entry["model"], "seed": entry["seed"], "checkpoint": str(path / "best.pt"),
                  "sha256": file_sha256(path / "best.pt"), "best_epoch": meta["best_epoch"]} for entry, path, meta in runs],
        "evaluation_source_sha256": file_sha256(__file__),
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "evaluation_protocol.json", protocol)
    started = time.perf_counter()
    try:
        torch.set_num_threads(4)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        ns = load_notebook_components()
        ns.update(DATA_DIR=root, N_TRAIN=metadata["train"]["n_samples"],
                  N_VALIDATION=metadata["validation"]["n_samples"], N_TEST=metadata["test"]["n_samples"])
        cfg = ns["SIMULATION_CONFIG"]
        if cfg.reference_year != metadata["test"]["reference_year"] or cfg.time_scale_years != metadata["test"]["time_scale_years"]:
            raise ValueError("Notebook time normalization differs from test data")
        data = {s: ns["load_split"](s) for s in SPLITS}
        rows, validity = [], []
        for entry, path, meta in runs:
            dest = output / f"{entry['model']}_seed_{entry['seed']}"
            dest.mkdir()
            shutil.copy2(path / "best.pt", dest / "best.pt")
            model, scaler, checkpoint = ns["load_cheng_checkpoint"](dest / "best.pt", device)
            if checkpoint["epoch"] != meta["best_epoch"]:
                raise ValueError("Loaded checkpoint is not the preselected best epoch")
            evaluation_meta = {**meta, "dataset_metadata_sha256": meta_hashes, "return_periods": list(PERIODS)}
            for split in SPLITS:
                stats = export_predictions(ns, model, scaler, data[split], split, dest, evaluation_meta, device)
                rows.extend(collect_metrics(dest, entry["model"], entry["seed"], split))
                validity.append(dict(model=entry["model"], seed=entry["seed"], split=split, **stats))
                if split in ("train", "validation"):
                    np.testing.assert_allclose(np.load(dest / split / "predictions_original.npy"),
                                               np.load(path / split / "predictions_original.npy"), rtol=2e-6, atol=2e-6,
                                               err_msg="Re-evaluation differs from the original frozen predictions")
            del model
            print(f"Evaluated {entry['model']} seed={entry['seed']}: train/validation/test", flush=True)
        detail, summary, gaps = summarize(rows)
        atomic_csv(output / "rmse_per_seed.csv", detail)
        atomic_csv(output / "test_comparison_summary.csv", summary)
        atomic_csv(output / "generalization_gaps.csv", gaps)
        atomic_csv(output / "validity_by_split.csv", pd.DataFrame(validity))
        for item in protocol["runs"]:
            if file_sha256(item["checkpoint"]) != item["sha256"]:
                raise ValueError("Source checkpoint changed during evaluation")
        for split in SPLITS:
            if hashes[split] != {p.name: file_sha256(p) for p in sorted((root / split).glob("*.npy"))}:
                raise ValueError("Dataset changed during evaluation")
        report = render_notebook(output)
        protocol.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
                        seconds=time.perf_counter()-started, report=str(report))
        atomic_json(output / "evaluation_protocol.json", protocol)
        print(summary.loc[summary.split.eq("test")].to_string(index=False), flush=True)
    except BaseException as error:
        protocol.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(output / "evaluation_protocol.json", protocol)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, default=PROJECT_ROOT / "results/cheng_nn_17d/architecture_comparison_20260930")
    p.add_argument("--output-directory", type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = p.parse_args()
    evaluate(args.experiment, args.output_directory, args.device)
