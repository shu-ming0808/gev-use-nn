import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from quantile_ratio_estimator import estimate_gev_quantile_ratio

# =========================
# Project paths
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "annual_max_25stations.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_baseline_model.pth")
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "station_gev_params.csv")

# =========================
# Quantile setting (一定要跟訓練一致)
# =========================
P_SET = np.array([
    0.0001, 0.001, 0.01, 0.1, 0.25,
    0.5, 0.75, 0.9, 0.99, 0.999, 0.9999
])

# =========================
# NN structure（要跟訓練完全一致）
# =========================
class GEVNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# Robust standardization
# =========================
def robust_standardize(y):
    y = np.asarray(y)
    y = y[~np.isnan(y)]

    med = np.median(y)
    q1 = np.quantile(y, 0.25)
    q3 = np.quantile(y, 0.75)
    iqr = q3 - q1

    if iqr == 0:
        raise ValueError("IQR = 0")

    z = (y - med) / iqr
    return z, med, iqr

# =========================
# Convert to NN input
# =========================
def make_input(y):
    z, med, iqr = robust_standardize(y)
    q = np.quantile(z, P_SET)
    return q, med, iqr

# =========================
# Estimate one station
# =========================
def estimate_one(model, y, device):
    q, med, iqr = make_input(y)

    x = torch.tensor(q, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x).cpu().numpy().flatten()

    mu_star, delta_star, c_hat = pred

    # Fast reparameterization predicts the positive distance from the GEV
    # support boundary, not log(sigma_star) directly.  Reconstruct the
    # standardized scale with the same rule used to generate the targets.
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    z = (y - med) / iqr
    positive_part = np.exp(np.clip(delta_star, -30.0, 30.0))
    if c_hat > 0.0:
        sigma_star = positive_part + c_hat * (np.max(z) - mu_star)
    else:
        sigma_star = positive_part + c_hat * (np.min(z) - mu_star)
    sigma_star = max(float(sigma_star), 1e-12)

    # inverse transform
    mu_hat = mu_star * iqr + med
    sigma_hat = sigma_star * iqr

    return mu_hat, sigma_hat, c_hat


def estimate_one_quantile_ratio(y):
    return estimate_gev_quantile_ratio(y)

# =========================
# Main
# =========================
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    df = pd.read_csv(DATA_PATH)

    # 第一欄是日期
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    stations = df.columns
    print("Stations:", len(stations))

    # load model
    model = GEVNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    results = []

    for s in stations:
        y = df[s].dropna().values

        if len(y) < 20:
            continue

        try:
            mu, sigma, c = estimate_one(model, y, device)
            xi = -c
            qr = estimate_one_quantile_ratio(y)

            results.append({
                "station": s,
                "n_obs": len(y),
                "mu_hat": mu,
                "sigma_hat": sigma,
                "log_sigma_hat": np.log(sigma),
                "xi_hat": xi,
                "shape_c_hat": c,
                "qr_mu_hat": qr["mu"],
                "qr_sigma_hat": qr["sigma"],
                "qr_log_sigma_hat": qr["log_sigma"],
                "qr_xi_hat": qr["xi"],
                "qr_ratio": qr["ratio"],
                "qr_solve_status": qr["solve_status"],
            })

        except Exception as e:
            print(f"Skip {s}: {e}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("Saved:", OUT_PATH)
    print(out_df.head())

if __name__ == "__main__":
    main()
