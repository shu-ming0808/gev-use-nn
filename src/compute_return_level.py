import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "grid_gev_params.csv")
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "grid_return_level.csv")

df = pd.read_csv(IN_PATH)

T = 45  # 你可以改 10, 50, 100

mu = df["mu"]
sigma = df["sigma"]
xi = df["xi"]

# 避免 xi = 0 問題
eps = 1e-6
xi_safe = xi.copy()
xi_safe[np.abs(xi_safe) < eps] = eps

z = mu + (sigma / xi_safe) * ((-np.log(1 - 1/T)) ** (-xi_safe) - 1)

df["z_T"] = z

df.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print(df.head())