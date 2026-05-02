import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.stats import genextreme as gev
from scipy.stats import norm

from baseline_train import GEVNet

P_SET = np.array(
    [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999],
    dtype=np.float64,
)


@dataclass
class SampleStats:
    median: float
    iqr: float
    z_min: float
    z_max: float
    n: int


# =============================
# 基本工具
# =============================
def robust_standardize_with_stats(y: np.ndarray) -> Tuple[np.ndarray, SampleStats]:
    y = np.asarray(y, dtype=np.float64)
    median = float(np.median(y))
    q25, q75 = np.percentile(y, [25, 75])
    iqr = float(q75 - q25)
    if iqr <= 1e-12:
        iqr = 1e-12
    z = (y - median) / iqr
    stats = SampleStats(
        median=median,
        iqr=iqr,
        z_min=float(np.min(z)),
        z_max=float(np.max(z)),
        n=int(len(y)),
    )
    return z, stats


def sample_to_features(y: np.ndarray) -> Tuple[np.ndarray, SampleStats]:
    z, stats = robust_standardize_with_stats(y)
    x_feat = np.percentile(z, P_SET * 100).astype(np.float32)
    return x_feat, stats


def invert_network_output(
    pred: np.ndarray,
    stats: SampleStats,
) -> Dict[str, float]:
    sc_loc, delta, c = [float(v) for v in pred]

    # 作者程式 generate_one_observation 的反推
    if c > np.finfo(float).eps:
        sc_scale = math.exp(delta) + c * (stats.z_max - sc_loc)
    else:
        sc_scale = math.exp(delta) + c * (stats.z_min - sc_loc)

    if sc_scale <= 1e-12:
        sc_scale = 1e-12

    mu = sc_loc * stats.iqr + stats.median
    sigma = sc_scale * stats.iqr
    sigma = max(float(sigma), 1e-12)

    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "c": float(c),
        "sc_loc": float(sc_loc),
        "delta": float(delta),
        "sample_median": float(stats.median),
        "sample_iqr": float(stats.iqr),
        "z_min": float(stats.z_min),
        "z_max": float(stats.z_max),
        "n": int(stats.n),
    }


def load_baseline_model(
    weights_path: str = "best_baseline_model.pth",
    device: Optional[str] = None,
) -> Tuple[GEVNet, str]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GEVNet().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def predict_gev_nn(
    y: np.ndarray,
    model: torch.nn.Module,
    device: Optional[str] = None,
) -> Dict[str, float]:
    if device is None:
        device = next(model.parameters()).device.type

    x_feat, stats = sample_to_features(y)
    x_tensor = torch.from_numpy(x_feat[None, :]).float().to(device)

    with torch.no_grad():
        pred = model(x_tensor).detach().cpu().numpy()[0]

    out = invert_network_output(pred, stats)
    out["x_feat"] = x_feat
    return out


# =============================
# NN parametric bootstrap
# =============================
def bootstrap_gev_nn(
    y: np.ndarray,
    model: torch.nn.Module,
    B: int = 1000,
    alpha: float = 0.05,
    seed: int = 123,
    device: Optional[str] = None,
) -> Dict[str, object]:
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    point = predict_gev_nn(y, model=model, device=device)

    rng = np.random.default_rng(seed)
    boot_rows: List[List[float]] = []

    for _ in range(B):
        y_star = gev.rvs(
            c=point["c"],
            loc=point["mu"],
            scale=point["sigma"],
            size=n,
            random_state=rng,
        )
        pred_star = predict_gev_nn(y_star, model=model, device=device)
        boot_rows.append([pred_star["mu"], pred_star["sigma"], pred_star["c"]])

    boot = np.asarray(boot_rows, dtype=np.float64)
    lower = np.quantile(boot, alpha / 2, axis=0)
    upper = np.quantile(boot, 1 - alpha / 2, axis=0)

    ci_df = pd.DataFrame(
        {
            "param": ["mu", "sigma", "c"],
            "estimate": [point["mu"], point["sigma"], point["c"]],
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
        }
    )

    return {
        "point": point,
        "bootstrap_samples": pd.DataFrame(boot, columns=["mu", "sigma", "c"]),
        "ci": ci_df,
    }


# =============================
# MLE + normal CI
# =============================
def gev_nll_eta(eta: np.ndarray, y: np.ndarray) -> float:
    mu, log_sigma, c = eta
    sigma = math.exp(log_sigma)
    y = np.asarray(y, dtype=np.float64)

    if sigma <= 0:
        return np.inf

    if abs(c) < 1e-8:
        z = (y - mu) / sigma
        return len(y) * log_sigma + np.sum(z) + np.sum(np.exp(-z))

    t = 1.0 + c * (y - mu) / sigma
    if np.any(t <= 0):
        return np.inf

    return len(y) * log_sigma + (1.0 + 1.0 / c) * np.sum(np.log(t)) + np.sum(t ** (-1.0 / c))



def fit_gev_mle(y: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.float64)

    # 先用 scipy 的 fit 當初值，失敗再 fallback
    try:
        c0, mu0, sigma0 = gev.fit(y)
        sigma0 = max(float(sigma0), 1e-6)
        x0 = np.array([mu0, math.log(sigma0), np.clip(c0, -0.49, 0.99)], dtype=np.float64)
    except Exception:
        mu0 = float(np.mean(y))
        sigma0 = max(float(np.std(y, ddof=1)), 1e-2)
        x0 = np.array([mu0, math.log(sigma0), 0.0], dtype=np.float64)

    bounds = [(None, None), (math.log(1e-8), math.log(1e6)), (-0.49, 0.99)]

    res = minimize(
        gev_nll_eta,
        x0=x0,
        args=(y,),
        method="L-BFGS-B",
        bounds=bounds,
    )

    if not res.success:
        res2 = minimize(
            gev_nll_eta,
            x0=x0,
            args=(y,),
            method="Nelder-Mead",
            options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-8},
        )
        if res2.success:
            res = res2

    mu_hat, log_sigma_hat, c_hat = res.x
    sigma_hat = math.exp(log_sigma_hat)

    return {
        "mu": float(mu_hat),
        "sigma": float(sigma_hat),
        "c": float(c_hat),
        "log_sigma": float(log_sigma_hat),
        "success": bool(res.success),
        "fun": float(res.fun),
        "x": np.asarray(res.x, dtype=np.float64),
    }



def numerical_hessian(fun, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = len(x)
    h = np.zeros((p, p), dtype=np.float64)
    f0 = fun(x)

    for i in range(p):
        ei = np.zeros(p)
        ei[i] = eps
        f_ip = fun(x + ei)
        f_im = fun(x - ei)
        h[i, i] = (f_ip - 2.0 * f0 + f_im) / (eps ** 2)

        for j in range(i + 1, p):
            ej = np.zeros(p)
            ej[j] = eps
            f_pp = fun(x + ei + ej)
            f_pm = fun(x + ei - ej)
            f_mp = fun(x - ei + ej)
            f_mm = fun(x - ei - ej)
            h_ij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps ** 2)
            h[i, j] = h_ij
            h[j, i] = h_ij
    return h



def mle_normal_ci_widths(
    y: np.ndarray,
    alpha: float = 0.05,
    eps: float = 1e-4,
) -> Dict[str, float]:
    fit = fit_gev_mle(y)
    xhat = fit["x"]
    y = np.asarray(y, dtype=np.float64)

    fun = lambda eta: gev_nll_eta(eta, y)
    hess = numerical_hessian(fun, xhat, eps=eps)

    try:
        cov_eta = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        cov_eta = np.linalg.pinv(hess)

    if np.any(~np.isfinite(cov_eta)):
        return {
            "mu": np.nan,
            "sigma": np.nan,
            "c": np.nan,
            "mu_se": np.nan,
            "sigma_se": np.nan,
            "c_se": np.nan,
            "mu_hat": fit["mu"],
            "sigma_hat": fit["sigma"],
            "c_hat": fit["c"],
        }

    se_mu = math.sqrt(max(cov_eta[0, 0], 0.0))
    se_log_sigma = math.sqrt(max(cov_eta[1, 1], 0.0))
    se_sigma = fit["sigma"] * se_log_sigma
    se_c = math.sqrt(max(cov_eta[2, 2], 0.0))

    z = norm.ppf(1 - alpha / 2)
    return {
        "mu": 2 * z * se_mu,
        "sigma": 2 * z * se_sigma,
        "c": 2 * z * se_c,
        "mu_se": se_mu,
        "sigma_se": se_sigma,
        "c_se": se_c,
        "mu_hat": fit["mu"],
        "sigma_hat": fit["sigma"],
        "c_hat": fit["c"],
    }


# =============================
# 給 real data.ipynb 用的高階函數
# =============================
def estimate_all_stations(
    annual_max: pd.DataFrame,
    model: torch.nn.Module,
    min_n: int = 20,
    device: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    for col in annual_max.columns:
        y = annual_max[col].dropna().to_numpy(dtype=np.float64)
        if len(y) < min_n:
            continue
        pred = predict_gev_nn(y, model=model, device=device)
        rows.append(
            {
                "station": col,
                "n": len(y),
                "mu": pred["mu"],
                "sigma": pred["sigma"],
                "c": pred["c"],
                "sample_median": pred["sample_median"],
                "sample_iqr": pred["sample_iqr"],
            }
        )
    return pd.DataFrame(rows)



def ci_widths_all_stations(
    annual_max: pd.DataFrame,
    model: torch.nn.Module,
    B: int = 300,
    alpha: float = 0.05,
    min_n: int = 20,
    seed: int = 123,
    device: Optional[str] = None,
) -> pd.DataFrame:
    rows = []

    for i, col in enumerate(annual_max.columns):
        y = annual_max[col].dropna().to_numpy(dtype=np.float64)
        if len(y) < min_n:
            continue

        nn_boot = bootstrap_gev_nn(
            y=y,
            model=model,
            B=B,
            alpha=alpha,
            seed=seed + i,
            device=device,
        )
        nn_ci = nn_boot["ci"].set_index("param")

        ml_ci = mle_normal_ci_widths(y=y, alpha=alpha)

        for param in ["mu", "sigma", "c"]:
            ml_width = ml_ci[param]
            nn_width = float(nn_ci.loc[param, "width"])
            ratio = np.nan
            if np.isfinite(ml_width) and abs(ml_width) > 1e-12:
                ratio = nn_width / ml_width

            rows.append(
                {
                    "station": col,
                    "param": param,
                    "nn_width": nn_width,
                    "ml_width": ml_width,
                    "ratio": ratio,
                    "n": len(y),
                }
            )

    return pd.DataFrame(rows)
