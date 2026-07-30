import argparse
import os

import numpy as np
import pandas as pd
import torch

from baseline_train import GEVNet as BaselineNet
from constraint_penalty_train import (
    GEVNet as ConstraintNet,
    MODEL_PATH,
    PROJECT_ROOT,
    load_constraint_dataset,
    train_constraint_penalty,
)
from project_paths import HISTORY_DIR


MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULT_TABLE_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
GRID_SUMMARY_PATH = os.path.join(RESULT_TABLE_DIR, "safety_margin_grid_search_summary.csv")
GRID_METRICS_PATH = os.path.join(RESULT_TABLE_DIR, "safety_margin_grid_search_metrics.csv")


def margin_label(margin: float) -> str:
    return f"m{margin:.4f}".replace(".", "p")


def delta_label(delta_margin: float) -> str:
    return f"d{delta_margin:.4f}".replace(".", "p")


def predict_array(model, x, device, batch_size=4096):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).float().to(device)
            pred = model(xb)
            preds.append(pred.detach().cpu().numpy())
    return np.vstack(preds)


def params_from_output(output, extrema, delta_clip=30.0):
    mu_star = output[:, 0]
    delta = output[:, 1]
    c = output[:, 2]
    z_min = extrema[:, 0]
    z_max = extrema[:, 1]

    positive_part = np.exp(np.clip(delta, -delta_clip, delta_clip))
    sigma_if_c_positive = positive_part + c * (z_max - mu_star)
    sigma_if_c_nonpositive = positive_part + c * (z_min - mu_star)
    sigma_star = np.where(c > 0.0, sigma_if_c_positive, sigma_if_c_nonpositive)
    sigma_star = np.clip(sigma_star, 1e-8, None)
    xi = -c

    return pd.DataFrame({"mu": mu_star, "sigma": sigma_star, "xi": xi})


def return_level(params, period):
    mu = params["mu"].to_numpy()
    sigma = params["sigma"].to_numpy()
    xi = params["xi"].to_numpy()
    a = -np.log(1.0 - 1.0 / period)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gev_rl = mu + sigma / xi * (a ** (-xi) - 1.0)
        gumbel_rl = mu - sigma * np.log(a)

    return np.where(np.abs(xi) < 1e-6, gumbel_rl, gev_rl)


def xi_bounds(params, extrema, safety_margin=0.0, tau=0.0):
    mu_star = params["mu"].to_numpy()
    sigma_star = params["sigma"].to_numpy()
    z_min = extrema[:, 0]
    z_max = extrema[:, 1]

    lower_denominator = np.clip(z_max - mu_star, 1e-6, None)
    upper_denominator = np.clip(mu_star - z_min, 1e-6, None)

    xi_lower = -(sigma_star / lower_denominator) * (1.0 + tau)
    xi_upper = (sigma_star / upper_denominator) * (1.0 + tau)
    return xi_lower + safety_margin, xi_upper - safety_margin


def metric_rows(true_params, pred_params, method, safety_margin, delta_margin):
    rows = []
    for param in ["mu", "sigma", "xi"]:
        y_true = true_params[param].to_numpy()
        y_pred = pred_params[param].to_numpy()
        diff = y_pred - y_true
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        rows.append(
            {
                "method": method,
                "safety_margin": safety_margin,
                "delta_margin": delta_margin,
                "param": param,
                "RMSE": float(np.sqrt(np.mean(diff ** 2))),
                "MAE": float(np.mean(np.abs(diff))),
                "Bias": float(np.mean(diff)),
                "Correlation": float(corr),
            }
        )
    return rows


def return_level_metric_rows(true_params, pred_params, method, safety_margin, delta_margin):
    rows = []
    for period in [10, 50, 100]:
        y_true = return_level(true_params, period)
        y_pred = return_level(pred_params, period)
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[finite]
        y_pred = y_pred[finite]
        diff = y_pred - y_true
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        rows.append(
            {
                "method": method,
                "safety_margin": safety_margin,
                "delta_margin": delta_margin,
                "param": f"RL{period}",
                "RMSE": float(np.sqrt(np.mean(diff ** 2))),
                "MAE": float(np.mean(np.abs(diff))),
                "Bias": float(np.mean(diff)),
                "Correlation": float(corr),
            }
        )
    return rows


def constraint_summary(params, output, extrema, method, safety_margin, delta_margin):
    xi_lower, xi_upper = xi_bounds(params, extrema, safety_margin=safety_margin)
    xi = params["xi"].to_numpy()
    delta = output[:, 1]
    if delta_margin > 0:
        delta_violation = delta < np.log(delta_margin)
    else:
        delta_violation = np.zeros_like(delta, dtype=bool)
    violation = (xi < xi_lower) | (xi > xi_upper) | delta_violation
    return {
        "method": method,
        "safety_margin": safety_margin,
        "delta_margin": delta_margin,
        "violation_rate": float(np.mean(violation)),
        "delta_violation_rate": float(np.mean(delta_violation)),
        "min_lower_margin": float(np.min(xi - xi_lower)),
        "min_upper_margin": float(np.min(xi_upper - xi)),
        "min_delta_slack": float(np.min(np.exp(np.clip(delta, -30.0, 30.0)))),
        "mean_interval_width": float(np.mean(xi_upper - xi_lower)),
    }


def score_metrics(metrics_df, weights=None):
    if weights is None:
        weights = {"mu": 1.0, "sigma": 1.0, "xi": 1.0}

    score = 0.0
    for param, weight in weights.items():
        rmse = metrics_df.loc[metrics_df["param"] == param, "RMSE"].iloc[0]
        score += weight * rmse
    return float(score)


def score_return_levels(metrics_df):
    score = 0.0
    for param in ["RL10", "RL50", "RL100"]:
        rmse = metrics_df.loc[metrics_df["param"] == param, "RMSE"].iloc[0]
        score += rmse
    return float(score)


def build_search_grid(mode, margins, delta_margins):
    if mode == "xi":
        return [("xi_margin", margin, 0.0) for margin in margins]
    if mode == "delta":
        return [("delta_margin", 0.0, delta_margin) for delta_margin in delta_margins]
    if mode == "both":
        return [
            ("combined", margin, delta_margin)
            for margin in margins
            for delta_margin in delta_margins
        ]
    raise ValueError(f"Unknown mode: {mode}")


def run_grid_search(
    margins,
    delta_margins=None,
    mode="xi",
    penalty_weight=0.1,
    epochs=150,
    patience=8,
    seed=111,
):
    if delta_margins is None:
        delta_margins = [0.0]
    search_grid = build_search_grid(mode, margins, delta_margins)

    os.makedirs(RESULT_TABLE_DIR, exist_ok=True)
    summary_path = os.path.join(
        RESULT_TABLE_DIR,
        f"safety_margin_grid_search_{mode}_summary.csv",
    )
    metrics_path = os.path.join(
        RESULT_TABLE_DIR,
        f"safety_margin_grid_search_{mode}_metrics.csv",
    )

    x, y, extrema, n_train, n_valid, _ = load_constraint_dataset()
    x_valid = x[n_train:n_train + n_valid]
    y_valid = y[n_train:n_train + n_valid]
    extrema_valid = extrema[n_train:n_train + n_valid]
    true_params = params_from_output(y_valid, extrema_valid)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    baseline_model = BaselineNet().to(device)
    baseline_model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "best_baseline_model.pth"), map_location=device)
    )
    baseline_output = predict_array(baseline_model, x_valid, device=device)
    baseline_params = params_from_output(baseline_output, extrema_valid)

    all_metrics = []
    all_summary = []

    for experiment, margin, delta_margin in search_grid:
            label = f"{experiment}_{margin_label(margin)}_{delta_label(delta_margin)}"
            model_path = os.path.join(MODEL_DIR, f"best_constraint_penalty_{label}_model.pth")
            history_path = str(
                HISTORY_DIR / f"constraint_penalty_{label}_history.csv"
            )

            print("=" * 72)
            print(
                f"Training experiment={experiment}, "
                f"safety_margin={margin:.4f}, "
                f"delta_margin={delta_margin:.4f}"
            )
            train_constraint_penalty(
                seed=seed,
                penalty_weight=penalty_weight,
                safety_margin=margin,
                delta_margin=delta_margin,
                epochs=epochs,
                patience=patience,
                model_path=model_path,
                history_path=history_path,
            )

            constraint_model = ConstraintNet().to(device)
            constraint_model.load_state_dict(torch.load(model_path, map_location=device))
            constraint_output = predict_array(constraint_model, x_valid, device=device)
            constraint_params = params_from_output(constraint_output, extrema_valid)

            baseline_metric = pd.DataFrame(
                metric_rows(true_params, baseline_params, "Baseline", margin, delta_margin)
                + return_level_metric_rows(
                    true_params, baseline_params, "Baseline", margin, delta_margin
                )
            )
            constraint_metric = pd.DataFrame(
                metric_rows(
                    true_params,
                    constraint_params,
                    "Constraint penalty",
                    margin,
                    delta_margin,
                )
                + return_level_metric_rows(
                    true_params,
                    constraint_params,
                    "Constraint penalty",
                    margin,
                    delta_margin,
                )
            )
            baseline_metric["experiment"] = experiment
            constraint_metric["experiment"] = experiment
            all_metrics.append(baseline_metric)
            all_metrics.append(constraint_metric)

            baseline_summary = constraint_summary(
                baseline_params,
                baseline_output,
                extrema_valid,
                "Baseline",
                safety_margin=margin,
                delta_margin=delta_margin,
            )
            baseline_summary["experiment"] = experiment
            constraint_summary_row = constraint_summary(
                constraint_params,
                constraint_output,
                extrema_valid,
                "Constraint penalty",
                safety_margin=margin,
                delta_margin=delta_margin,
            )
            constraint_summary_row["experiment"] = experiment
            param_metric = constraint_metric[constraint_metric["param"].isin(["mu", "sigma", "xi"])]
            rl_metric = constraint_metric[constraint_metric["param"].str.startswith("RL")]
            constraint_score = score_metrics(param_metric)
            rl_score = score_return_levels(rl_metric)
            xi_rmse = param_metric.loc[param_metric["param"] == "xi", "RMSE"].iloc[0]
            rl100_rmse = rl_metric.loc[rl_metric["param"] == "RL100", "RMSE"].iloc[0]

            all_summary.extend([baseline_summary, constraint_summary_row])
            all_summary[-1]["rmse_score"] = constraint_score
            all_summary[-1]["return_level_rmse_score"] = rl_score
            all_summary[-1]["xi_rmse"] = float(xi_rmse)
            all_summary[-1]["rl100_rmse"] = float(rl100_rmse)
            all_summary[-1]["model_path"] = model_path
            all_summary[-1]["history_path"] = history_path

            print(
                f"m={margin:.4f}, delta_margin={delta_margin:.4f} | "
                f"param_score={constraint_score:.6f} | "
                f"RL_score={rl_score:.6f} | "
                f"xi_RMSE={xi_rmse:.6f} | "
                f"RL100_RMSE={rl100_rmse:.6f} | "
                f"viol={constraint_summary_row['violation_rate']:.4f}"
            )

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    summary_df = pd.DataFrame(all_summary)
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    constraint_summary_df = summary_df[summary_df["method"] == "Constraint penalty"].copy()
    best_row = constraint_summary_df.sort_values(
        ["return_level_rmse_score", "rmse_score", "violation_rate"]
    ).iloc[0]

    print("=" * 72)
    print("Best setting by return-level RMSE score:")
    print(best_row)
    print(f"Saved summary: {summary_path}")
    print(f"Saved metrics: {metrics_path}")
    return metrics_df, summary_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["xi", "delta", "both"],
        default="xi",
        help=(
            "xi: vary only xi safety margin; "
            "delta: vary only exp(delta) margin; "
            "both: full two-parameter grid."
        ),
    )
    parser.add_argument(
        "--margins",
        default="0.0,0.01,0.03,0.05",
        help="Comma-separated safety margins.",
    )
    parser.add_argument(
        "--delta-margins",
        default="0.0",
        help="Comma-separated exp(delta) lower bounds. Use 0.0 to disable.",
    )
    parser.add_argument("--penalty-weight", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=111)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    margins = [float(x.strip()) for x in args.margins.split(",") if x.strip()]
    delta_margins = [float(x.strip()) for x in args.delta_margins.split(",") if x.strip()]
    run_grid_search(
        margins=margins,
        delta_margins=delta_margins,
        mode=args.mode,
        penalty_weight=args.penalty_weight,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )
