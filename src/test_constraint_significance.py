import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel, wilcoxon

from baseline_train import GEVNet as BaselineNet
from constraint_penalty_train import GEVNet as ConstraintNet
from grid_search_safety_margin import (
    MODEL_DIR,
    PROJECT_ROOT,
    load_constraint_dataset,
    params_from_output,
    predict_array,
    return_level,
)


OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "tables",
    "constraint_significance_tests.csv",
)

MODEL_PATHS = {
    "xi margin (m=0.01)": os.path.join(
        MODEL_DIR,
        "best_constraint_penalty_xi_margin_m0p0100_d0p0000_model.pth",
    ),
    "delta margin (epsilon=0.05)": os.path.join(
        MODEL_DIR,
        "best_constraint_penalty_delta_margin_m0p0000_d0p0500_model.pth",
    ),
}


def parameter_arrays(params):
    return {
        "mu": params["mu"].to_numpy(),
        "sigma": params["sigma"].to_numpy(),
        "xi": params["xi"].to_numpy(),
        "RL10": return_level(params, 10),
        "RL50": return_level(params, 50),
        "RL100": return_level(params, 100),
    }


def paired_bootstrap_rmse_difference(
    y_true,
    baseline_pred,
    candidate_pred,
    n_bootstrap=2000,
    seed=111,
    chunk_size=50,
):
    finite = (
        np.isfinite(y_true)
        & np.isfinite(baseline_pred)
        & np.isfinite(candidate_pred)
    )
    y_true = np.asarray(y_true[finite], dtype=np.float64)
    baseline_pred = np.asarray(baseline_pred[finite], dtype=np.float64)
    candidate_pred = np.asarray(candidate_pred[finite], dtype=np.float64)

    baseline_sq = (baseline_pred - y_true) ** 2
    candidate_sq = (candidate_pred - y_true) ** 2
    n = len(y_true)

    baseline_rmse = float(np.sqrt(np.mean(baseline_sq)))
    candidate_rmse = float(np.sqrt(np.mean(candidate_sq)))
    observed_difference = candidate_rmse - baseline_rmse

    rng = np.random.default_rng(seed)
    differences = np.empty(n_bootstrap, dtype=np.float64)

    completed = 0
    while completed < n_bootstrap:
        size = min(chunk_size, n_bootstrap - completed)
        indices = rng.integers(0, n, size=(size, n), endpoint=False)
        baseline_boot = np.sqrt(np.mean(baseline_sq[indices], axis=1))
        candidate_boot = np.sqrt(np.mean(candidate_sq[indices], axis=1))
        differences[completed:completed + size] = candidate_boot - baseline_boot
        completed += size

    ci_low, ci_high = np.percentile(differences, [2.5, 97.5])

    # H1: candidate squared error is smaller than baseline squared error.
    try:
        statistic, p_value = wilcoxon(
            candidate_sq,
            baseline_sq,
            alternative="less",
            zero_method="wilcox",
            method="approx",
        )
    except ValueError:
        statistic, p_value = np.nan, 1.0

    improvement_pct = (
        100.0 * (baseline_rmse - candidate_rmse) / baseline_rmse
        if baseline_rmse > 0
        else np.nan
    )
    ttest_result = ttest_rel(
        candidate_sq,
        baseline_sq,
        alternative="less",
        nan_policy="omit",
    )

    return {
        "n": n,
        "baseline_rmse": baseline_rmse,
        "candidate_rmse": candidate_rmse,
        "rmse_difference": observed_difference,
        "improvement_pct": improvement_pct,
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "paired_t_statistic": float(ttest_result.statistic),
        "p_value_one_sided": float(ttest_result.pvalue),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p_value_one_sided": float(p_value),
    }


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted_sorted = np.empty(n, dtype=np.float64)
    running_max = 0.0

    for rank, index in enumerate(order):
        adjusted = (n - rank) * p_values[index]
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = min(running_max, 1.0)

    adjusted = np.empty(n, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted


def load_predictions(device):
    x, y, extrema, n_train, n_valid, _ = load_constraint_dataset()
    x_valid = x[n_train:n_train + n_valid]
    y_valid = y[n_train:n_train + n_valid]
    extrema_valid = extrema[n_train:n_train + n_valid]

    true_params = params_from_output(y_valid, extrema_valid)

    baseline_model = BaselineNet().to(device)
    baseline_model.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "best_baseline_model.pth"),
            map_location=device,
        )
    )
    baseline_output = predict_array(baseline_model, x_valid, device=device)
    baseline_params = params_from_output(baseline_output, extrema_valid)

    candidate_params = {}
    for method, model_path in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Missing model for {method}: {model_path}. "
                "Run the corresponding grid search first."
            )
        model = ConstraintNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        output = predict_array(model, x_valid, device=device)
        candidate_params[method] = params_from_output(output, extrema_valid)

    return true_params, baseline_params, candidate_params


def run_significance_tests(n_bootstrap=2000, seed=111):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    true_params, baseline_params, candidate_params = load_predictions(device)
    truth = parameter_arrays(true_params)
    baseline = parameter_arrays(baseline_params)

    rows = []
    for method, params in candidate_params.items():
        candidate = parameter_arrays(params)
        for target in ["mu", "sigma", "xi", "RL10", "RL50", "RL100"]:
            print(f"Testing {method}: {target}")
            result = paired_bootstrap_rmse_difference(
                y_true=truth[target],
                baseline_pred=baseline[target],
                candidate_pred=candidate[target],
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            result.update(
                {
                    "comparison": f"{method} vs Baseline",
                    "target": target,
                    "alternative": "candidate RMSE < baseline RMSE",
                }
            )
            rows.append(result)

    results = pd.DataFrame(rows)
    results["p_value_holm"] = holm_adjust(results["p_value_one_sided"])
    results["bootstrap_supports_improvement"] = (
        results["bootstrap_ci_high"] < 0
    )
    results["significant_improvement_0.05"] = (
        (results["p_value_holm"] < 0.05)
        & results["bootstrap_supports_improvement"]
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    columns = [
        "comparison",
        "target",
        "baseline_rmse",
        "candidate_rmse",
        "rmse_difference",
        "improvement_pct",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "p_value_holm",
        "significant_improvement_0.05",
    ]
    print(results[columns].to_string(index=False))
    print(f"Saved: {OUTPUT_PATH}")
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=111)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_significance_tests(n_bootstrap=args.bootstrap, seed=args.seed)
