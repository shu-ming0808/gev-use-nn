import os
import numpy as np
import pandas as pd
from scipy.stats import genextreme as gev

# =========================
# 基本設定
# =========================
out_dir = "figure4_samples"
os.makedirs(out_dir, exist_ok=True)

n = 416      # 論文 Figure 4 固定 sample size
R = 100      # 每個 true 值重複幾次
seed = 123

rng = np.random.default_rng(seed)

# 這三組是畫圖 x 軸要用的 true values
mu_vals = np.linspace(1, 50, 25)
sigma_vals = np.linspace(0.1, 40, 25)
xi_vals = np.linspace(-0.4, 0.8, 25)

# scipy 的 genextreme 用 c = -xi
def rgev(mu, sigma, xi, n, rng):
    return gev.rvs(c=-xi, loc=mu, scale=sigma, size=n, random_state=rng)

meta_rows = []

# =========================
# Panel 1: true mu
# 固定 sigma=10, xi=0.2
# =========================
for mu_true in mu_vals:
    for rep in range(R):
        x = rgev(mu_true, 10.0, 0.2, n, rng)

        fname = f"mu_true_{mu_true:.6f}_rep_{rep:03d}.npy"
        fpath = os.path.join(out_dir, fname)
        np.save(fpath, x)

        meta_rows.append({
            "panel": "mu",
            "true_value": mu_true,
            "rep": rep,
            "file_path": fpath
        })

# =========================
# Panel 2: true sigma
# 固定 mu=25, xi=0.2
# =========================
for sigma_true in sigma_vals:
    for rep in range(R):
        x = rgev(25.0, sigma_true, 0.2, n, rng)

        fname = f"sigma_true_{sigma_true:.6f}_rep_{rep:03d}.npy"
        fpath = os.path.join(out_dir, fname)
        np.save(fpath, x)

        meta_rows.append({
            "panel": "sigma",
            "true_value": sigma_true,
            "rep": rep,
            "file_path": fpath
        })

# =========================
# Panel 3: true xi
# 固定 mu=25, sigma=10
# =========================
for xi_true in xi_vals:
    for rep in range(R):
        x = rgev(25.0, 10.0, xi_true, n, rng)

        fname = f"xi_true_{xi_true:.6f}_rep_{rep:03d}.npy"
        fpath = os.path.join(out_dir, fname)
        np.save(fpath, x)

        meta_rows.append({
            "panel": "xi",
            "true_value": xi_true,
            "rep": rep,
            "file_path": fpath
        })

meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv("sample_index.csv", index=False)

print("Done.")
print("Saved sample_index.csv")
print("Saved all .npy samples in folder:", out_dir)
print(meta_df.head())