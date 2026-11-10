"""Summarize four paired EAS simulations and make coefficient-map slides.

The first, older single-run experiment is diagnostic only. It is never
counted as an additional independent replicate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import t


DESKTOP = Path.home() / "Desktop"
PAIRED = DESKTOP / "模擬資料"
FIRST = (DESKTOP / "simulations" / "latitude_v3_ram16_mem8_cpu20_v1"
         / "mle_comparison_v2_crps" / "replicates" / "replicate_001")
OUTPUT = (Path(__file__).resolve().parents[1] / "output" / "beamer"
          / "eas_paired_simulations_20261110")
COEFS = ("mu0", "beta_mu", "eta0", "beta_sigma", "xi0")
SYMBOLS = {
    "mu0": r"$\mu_0$",
    "beta_mu": r"$\beta_\mu$",
    "eta0": r"$\eta_0=\log\sigma_0$",
    "beta_sigma": r"$\beta_\sigma$",
    "xi0": r"$\xi_0$",
}
YEARS = (1976, 2005, 2025)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def interval(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n != 4 or not np.isfinite(values).all():
        raise ValueError("Exactly four finite paired replicates required")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return mean, sd, mean - half, mean + half


def load_run(method: str, replicate: int) -> dict:
    folder = PAIRED / method / f"replicate_{replicate:03d}" / "attempt_001"
    report = json.loads((folder / "run_report.json").read_text(encoding="utf-8"))
    if not report.get("workflow_completed") or report["estimator"] != method:
        raise ValueError(f"Incomplete or mislabeled run: {folder}")
    direct_name = "nn_coefficients.csv" if method == "nn" else "mle_coefficients.csv"
    direct = pd.read_csv(folder / direct_name, dtype={"grid_id": str}).set_index("grid_id")
    oof = pd.read_csv(folder / "oof_coefficients.csv", dtype={"grid_id": str}).set_index("grid_id")
    truth = np.load(folder / "truth.npy", allow_pickle=False)
    if truth.shape != (7822, 5) or len(direct) != len(oof) or not direct.index.equals(oof.index):
        raise ValueError(f"Grid mismatch: {folder}")
    metrics = pd.read_csv(folder / "truth_parameter_metrics.csv")
    for stage, frame in (("nn" if method == "nn" else "mle", direct), ("gp_oof", oof)):
        for j, coef in enumerate(COEFS):
            reported = metrics.loc[(metrics.stage == stage) & (metrics.coefficient == coef), "RMSE"]
            observed = np.sqrt(np.mean((frame[coef].to_numpy() - truth[:, j]) ** 2))
            if len(reported) != 1 or not np.isclose(reported.iloc[0], observed, rtol=0, atol=2e-8):
                raise ValueError(f"RMSE cannot be reproduced: {folder}, {stage}, {coef}")
    return dict(folder=folder, report=report, direct=direct, oof=oof, truth=truth,
                metrics=metrics, crps=pd.read_csv(folder / "crps_metrics.csv"),
                rl=pd.read_csv(folder / "return_level_metrics.csv"),
                folds=pd.read_csv(folder / "spatial_oof.csv", dtype={"grid_id": str}))


def frame_table(heading: str, stage: str, rows: pd.DataFrame) -> str:
    body = []
    for coef in COEFS:
        row = rows[(rows.stage == stage) & (rows.coefficient == coef)].set_index("method")
        def cell(method: str) -> str:
            v = row.loc[method]
            return f"{v['mean']:.4f} $\\pm$ {v['sd']:.4f} [{v['ci_low']:.4f}, {v['ci_high']:.4f}]"
        body.append(f"{SYMBOLS[coef]} & {cell('nn')} & {cell('mle')} \\\\")
    return "\n".join([
        rf"\begin{{frame}}{{{heading}}}",
        r"\centering\scriptsize",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Coefficient & NN mean $\pm$ SD [95\% CI] & MLE mean $\pm$ SD [95\% CI] \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\medskip",
        r"{\footnotesize Four paired simulation replicates; CI uses $t_{3,0.975}$. "
        + ("Direct five-coefficient estimates." if stage == "direct" else "After buffered spatial CV (OOF).")
        + r" Lower RMSE is better.}",
        r"\end{frame}",
    ])


def normalized_table(rows: pd.DataFrame, truth_sd: dict[str, float]) -> str:
    body = []
    for coef in COEFS:
        stage = rows[(rows.stage == "gp_oof") & (rows.coefficient == coef)].set_index("method")
        vals = []
        for method in ("nn", "mle"):
            v = stage.loc[method]
            vals.append(f"{v['mean']/truth_sd[coef]:.2f} [{v['ci_low']/truth_sd[coef]:.2f}, {v['ci_high']/truth_sd[coef]:.2f}]")
        body.append(f"{SYMBOLS[coef]} & {truth_sd[coef]:.4f} & {vals[0]} & {vals[1]} \\\\")
    return "\n".join([
        r"\begin{frame}{Standardized Spatial RMSE}",
        r"\centering\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Coefficient & Truth spatial SD & NN NRMSE [95\% CI] & MLE NRMSE [95\% CI] \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\medskip",
        r"{\footnotesize NRMSE $=$ OOF RMSE / spatial SD of the fixed generating truth. "
        r"CI is across four paired replicates, not across GRID cells.}",
        r"\end{frame}",
    ])


def make_maps(runs: dict[tuple[str, int], dict], truth: np.ndarray) -> None:
    first = runs["nn", 1]["oof"]
    grids = first[["latitude", "longitude_180"]].reset_index(drop=True)
    if not all(r["oof"].index.equals(first.index) for r in runs.values()):
        raise ValueError("Different GRID order across replicate maps")
    lon = grids.longitude_180.to_numpy()
    lat = grids.latitude.to_numpy()
    for j, coef in enumerate(COEFS):
        nn = np.mean([runs["nn", i]["oof"][coef].to_numpy() for i in range(1, 5)], axis=0)
        mle = np.mean([runs["mle", i]["oof"][coef].to_numpy() for i in range(1, 5)], axis=0)
        surfaces = (truth[:, j], nn, mle)
        all_values = np.concatenate(surfaces)
        lo, hi = np.quantile(all_values, [0.005, 0.995])
        cmap = "coolwarm" if coef in ("beta_mu", "beta_sigma", "xi0") else "viridis"
        if lo < 0 < hi:
            norm = TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi)
        else:
            norm = plt.Normalize(vmin=lo, vmax=hi)
        fig, axes = plt.subplots(3, 3, figsize=(15.0, 8.8), sharex=True, sharey=True)
        for r, (name, values) in enumerate(zip(("Generating truth", "NN + GP OOF", "MLE + GP OOF"), surfaces)):
            for c, year in enumerate(YEARS):
                ax = axes[r, c]
                scatter = ax.scatter(lon, lat, c=values, cmap=cmap, norm=norm,
                                     s=2.2, marker="s", linewidths=0, rasterized=True)
                ax.set_xlim(lon.min() - 1, lon.max() + 1)
                ax.set_ylim(lat.min() - 1, lat.max() + 1)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(alpha=0.12, linewidth=0.4)
                if r == 0:
                    ax.set_title(f"Year {year - 1975} ({year})", fontsize=13)
                if c == 0:
                    ax.set_ylabel(name + "\nLatitude (°N)", fontsize=10)
                if r == 2:
                    ax.set_xlabel("Longitude (°E)", fontsize=10)
        fig.suptitle(f"{SYMBOLS[coef]} spatial surface", fontsize=17)
        fig.colorbar(scatter, ax=axes, fraction=0.018, pad=0.015, extend="both",
                     label=f"{coef} ({'°C' if coef == 'mu0' else '°C/decade' if coef == 'beta_mu' else 'per decade' if coef == 'beta_sigma' else 'dimensionless'})")
        fig.text(0.5, 0.015,
                 "The five GEV coefficients do not change with year. Columns repeat by design; "
                 "only μ(t) and log σ(t) change with time.",
                 ha="center", fontsize=9)
        fig.subplots_adjust(left=0.065, right=0.89, top=0.89, bottom=0.095,
                            wspace=0.08, hspace=0.08)
        fig.savefig(OUTPUT / f"eas_{coef}_truth_nn_mle_3x3.png", dpi=190)
        plt.close(fig)

    nn_avg = np.mean([runs["nn", i]["oof"][list(COEFS)].to_numpy()
                      for i in range(1, 5)], axis=0)
    mle_avg = np.mean([runs["mle", i]["oof"][list(COEFS)].to_numpy()
                       for i in range(1, 5)], axis=0)
    surfaces = (truth, nn_avg, mle_avg)
    vals = np.concatenate([
        arr[:, 0] + (year - 2000.5) / 10 * arr[:, 1]
        for arr in surfaces for year in YEARS
    ])
    lo, hi = np.quantile(vals, [0.005, 0.995])
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 8.8), sharex=True, sharey=True)
    for r, (name, arr) in enumerate(zip(("Generating truth", "NN + GP OOF", "MLE + GP OOF"), surfaces)):
        for c, year in enumerate(YEARS):
            ax = axes[r, c]
            current = arr[:, 0] + (year - 2000.5) / 10 * arr[:, 1]
            scatter = ax.scatter(lon, lat, c=current, cmap="viridis", vmin=lo, vmax=hi,
                                 s=2.2, marker="s", linewidths=0, rasterized=True)
            ax.set_xlim(lon.min() - 1, lon.max() + 1)
            ax.set_ylim(lat.min() - 1, lat.max() + 1)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.12, linewidth=0.4)
            if r == 0:
                ax.set_title(f"Year {year - 1975} ({year})", fontsize=13)
            if c == 0:
                ax.set_ylabel(name + "\nLatitude (°N)", fontsize=10)
            if r == 2:
                ax.set_xlabel("Longitude (°E)", fontsize=10)
    fig.suptitle(r"Time-varying location $\mu(t)=\mu_0+\beta_\mu t_c$", fontsize=17)
    fig.colorbar(scatter, ax=axes, fraction=0.018, pad=0.015, extend="both",
                 label=r"$\mu(t)$ (°C)")
    fig.subplots_adjust(left=0.065, right=0.89, top=0.89, bottom=0.095,
                        wspace=0.08, hspace=0.08)
    fig.savefig(OUTPUT / "eas_mu_t_truth_nn_mle_3x3.png", dpi=190)
    plt.close(fig)

    # Scale, unlike its two coefficients, varies by year. Keep one color
    # scale across all nine panels so the time contrast is interpretable.
    vals = np.concatenate([
        np.exp(arr[:, 2] + (year - 2000.5) / 10 * arr[:, 3])
        for arr in surfaces for year in YEARS
    ])
    if not np.isfinite(vals).all():
        raise ValueError("Nonfinite sigma(t) in the reconstructed surfaces")
    lo, hi = np.quantile(vals, [0.005, 0.995])
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 8.8), sharex=True, sharey=True)
    for r, (name, arr) in enumerate(zip(("Generating truth", "NN + GP OOF", "MLE + GP OOF"), surfaces)):
        for c, year in enumerate(YEARS):
            ax = axes[r, c]
            current = np.exp(arr[:, 2] + (year - 2000.5) / 10 * arr[:, 3])
            scatter = ax.scatter(lon, lat, c=current, cmap="viridis", vmin=lo, vmax=hi,
                                 s=2.2, marker="s", linewidths=0, rasterized=True)
            ax.set_xlim(lon.min() - 1, lon.max() + 1)
            ax.set_ylim(lat.min() - 1, lat.max() + 1)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.12, linewidth=0.4)
            if r == 0:
                ax.set_title(f"Year {year - 1975} ({year})", fontsize=13)
            if c == 0:
                ax.set_ylabel(name + "\nLatitude (°N)", fontsize=10)
            if r == 2:
                ax.set_xlabel("Longitude (°E)", fontsize=10)
    fig.suptitle(r"Time-varying scale $\sigma(t)=\exp(\eta_0+\beta_\sigma t_c)$", fontsize=17)
    fig.colorbar(scatter, ax=axes, fraction=0.018, pad=0.015, extend="both",
                 label=r"$\sigma(t)$ (°C)")
    fig.subplots_adjust(left=0.065, right=0.89, top=0.89, bottom=0.095,
                        wspace=0.08, hspace=0.08)
    fig.savefig(OUTPUT / "eas_sigma_t_truth_nn_mle_3x3.png", dpi=190)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs = {(m, i): load_run(m, i) for m in ("nn", "mle") for i in range(1, 5)}
    sample_ids = []
    truth_hashes = set()
    for i in range(1, 5):
        nn, mle = runs["nn", i], runs["mle", i]
        if nn["report"].get("sample_id") != mle["report"].get("sample_id"):
            raise ValueError(f"Replicate {i} is not paired")
        sample_ids.append(nn["report"]["sample_id"])
        truth_hashes.update((digest(nn["folder"] / "truth.npy"), digest(mle["folder"] / "truth.npy")))
    if len(set(sample_ids)) != 4 or len(truth_hashes) != 1:
        raise ValueError("Four independent sample IDs and one frozen truth required")
    truth = runs["nn", 1]["truth"]
    truth_sd = {coef: float(np.std(truth[:, j], ddof=0)) for j, coef in enumerate(COEFS)}
    if min(truth_sd.values()) <= 0:
        raise ValueError("Cannot standardize constant truth surfaces")

    per_rep = []
    for (method, rep), run in runs.items():
        for stage, label in (("nn" if method == "nn" else "mle", "direct"), ("gp_oof", "gp_oof")):
            for _, metric in run["metrics"].loc[run["metrics"].stage == stage].iterrows():
                per_rep.append(dict(method=method, replicate=rep, stage=label,
                                    coefficient=metric.coefficient, rmse=metric.RMSE,
                                    mae=metric.MAE, bias=metric.Bias))
    per_rep = pd.DataFrame(per_rep)
    per_rep.to_csv(OUTPUT / "coefficient_metrics_by_replicate.csv", index=False)
    summary = []
    for (method, stage, coef), group in per_rep.groupby(["method", "stage", "coefficient"], sort=False):
        mean, sd, low, high = interval(group.sort_values("replicate").rmse.to_numpy())
        summary.append(dict(method=method, stage=stage, coefficient=coef,
                            mean=mean, sd=sd, variance=sd**2, ci_low=low, ci_high=high,
                            truth_spatial_sd=truth_sd[coef], standardized_rmse=mean/truth_sd[coef]))
    summary = pd.DataFrame(summary)
    summary.to_csv(OUTPUT / "coefficient_rmse_summary.csv", index=False)
    paired = []
    for stage in ("direct", "gp_oof"):
        for coef in COEFS:
            one = per_rep[(per_rep.stage == stage) & (per_rep.coefficient == coef)]
            pivot = one.pivot(index="replicate", columns="method", values="rmse").sort_index()
            delta = pivot["nn"].to_numpy() - pivot["mle"].to_numpy()
            mean, sd, low, high = interval(delta)
            paired.append(dict(stage=stage, coefficient=coef, mean_nn_minus_mle=mean,
                               sd=sd, variance=sd**2, ci_low=low, ci_high=high))
    pd.DataFrame(paired).to_csv(OUTPUT / "paired_rmse_difference.csv", index=False)

    quality = []
    for (method, rep), run in runs.items():
        for stage in ("nn" if method == "nn" else "mle", "gp_oof"):
            c = run["crps"].loc[(run["crps"].stage == stage) & (run["crps"].year.astype(str) == "all")].iloc[0]
            for period in (50, 100):
                rl = run["rl"].loc[(run["rl"].stage == stage)
                                   & (run["rl"].return_period == period)].iloc[0]
                quality.append(dict(method=method, replicate=rep, stage=stage,
                                    return_period=period, crps=float(c.mean_CRPS_C),
                                    outside_support=int(c.n_outside_support),
                                    grids_outside_support=int(c.n_grids_outside_support),
                                    rl_rmse=float(rl.RMSE)))
    pd.DataFrame(quality).to_csv(OUTPUT / "quality_metrics_by_replicate.csv", index=False)

    mu = []
    for (method, rep), run in runs.items():
        for stage, frame in (("direct", run["direct"]), ("gp_oof", run["oof"])):
            delta0 = frame.mu0.to_numpy() - truth[:, 0]
            deltab = frame.beta_mu.to_numpy() - truth[:, 1]
            for year in YEARS:
                centered = (year - 2000.5) / 10
                error = delta0 + centered * deltab
                mu.append(dict(method=method, replicate=rep, stage=stage, year=year,
                               rmse_mu_t=float(np.sqrt(np.mean(error**2))),
                               bias_mu_t=float(error.mean()),
                               rmse_mu0=float(np.sqrt(np.mean(delta0**2))),
                               rmse_slope_contribution=float(abs(centered) * np.sqrt(np.mean(deltab**2)))))
    pd.DataFrame(mu).to_csv(OUTPUT / "mu_time_diagnostics.csv", index=False)
    folds = []
    for (method, rep), run in runs.items():
        frame = run["oof"]
        mapping = run["folds"].loc[run["folds"].coefficient == "mu0", ["grid_id", "outer_fold"]]
        joined = frame[["mu0"]].reset_index().merge(mapping, on="grid_id", validate="one_to_one")
        truth_by_grid = pd.Series(truth[:, 0], index=frame.index)
        joined["truth"] = joined.grid_id.map(truth_by_grid)
        joined["error"] = joined.mu0 - joined.truth
        for fold, g in joined.groupby("outer_fold"):
            folds.append(dict(method=method, replicate=rep, outer_fold=fold, n_grids=len(g),
                              mu0_bias=g.error.mean(), mu0_rmse=np.sqrt(np.mean(g.error**2))))
    pd.DataFrame(folds).to_csv(OUTPUT / "mu0_fold_diagnostics.csv", index=False)

    old = {m: FIRST / m / "attempt_001" for m in ("nn", "mle")}
    comparison = {}
    for method in ("nn", "mle"):
        old_report = json.loads((old[method] / "run_report.json").read_text(encoding="utf-8"))
        new_report = runs[method, 1]["report"]
        old_annual = np.load(old[method] / "annual_maxima.npy", allow_pickle=False)
        new_annual = np.load(runs[method, 1]["folder"] / "annual_maxima.npy", allow_pickle=False)
        old_oof = pd.read_csv(old[method] / "oof_coefficients.csv")
        new_oof = runs[method, 1]["oof"].reset_index()
        comparison[method] = dict(old_protocol_id=old_report["protocol_id"],
                                  new_protocol_id=new_report["protocol_id"],
                                  old_sample_id=old_report.get("sample_id"),
                                  new_sample_id=new_report.get("sample_id"),
                                  annual_max_absolute_difference=float(np.max(np.abs(old_annual-new_annual))),
                                  truth_byte_identical=digest(old[method] / "truth.npy") == digest(runs[method,1]["folder"] / "truth.npy"),
                                  gp_oof_max_absolute_difference=float(
                                      (old_oof[list(COEFS)] - new_oof[list(COEFS)]).abs().max().max()))
    (OUTPUT / "first_run_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    tex = "\n\n".join((frame_table("Direct GEV Estimation: Four Simulations", "direct", summary),
                           frame_table("After Buffered Spatial CV: Four Simulations", "gp_oof", summary),
                           normalized_table(summary, truth_sd)))
    (OUTPUT / "eas_simulation_tables.tex").write_text(tex + "\n", encoding="utf-8")
    make_maps(runs, truth)
    map_frames = []
    for coef in COEFS:
        map_frames.extend([
            rf"\begin{{frame}}{{EAS {SYMBOLS[coef]}: Truth, NN and MLE}}",
            r"\centering",
            rf"\includegraphics[width=0.96\textwidth,height=0.82\textheight,keepaspectratio]{{eas_{coef}_truth_nn_mle_3x3.png}}",
            r"\end{frame}",
            "",
        ])
    for name, title in (("mu_t", r"Time-Varying Location $\mu(t)$"),
                        ("sigma_t", r"Time-Varying Scale $\sigma(t)$")):
        map_frames.extend([
            rf"\begin{{frame}}{{{title}: Truth, NN and MLE}}",
            r"\centering",
            rf"\includegraphics[width=0.96\textwidth,height=0.82\textheight,keepaspectratio]{{eas_{name}_truth_nn_mle_3x3.png}}",
            r"\end{frame}",
            "",
        ])
    (OUTPUT / "eas_simulation_maps.tex").write_text("\n".join(map_frames), encoding="utf-8")
    print(f"Wrote tables, diagnostics, five coefficient maps and two time-varying maps: {OUTPUT}")


if __name__ == "__main__":
    main()
