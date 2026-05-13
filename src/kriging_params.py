import os
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "station_gev_params_with_loc.csv")
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "grid_gev_params.csv")

df = pd.read_csv(IN_PATH)

X_raw = df[["lon", "lat"]].values
X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
X = (X_raw - X_mean) / X_std

targets = {
    "mu": df["mu_hat"].values,
    "log_sigma": df["log_sigma_hat"].values,
    "xi": df["xi_hat"].values,
}

lon_min, lon_max = df["lon"].min(), df["lon"].max()
lat_min, lat_max = df["lat"].min(), df["lat"].max()

lon_grid = np.linspace(lon_min, lon_max, 80)
lat_grid = np.linspace(lat_min, lat_max, 80)

grid_raw = np.array([[lon, lat] for lat in lat_grid for lon in lon_grid])
grid = (grid_raw - X_mean) / X_std

out = pd.DataFrame(grid_raw, columns=["lon", "lat"])

for name, y in targets.items():
    kernel = (
        C(1.0, (1e-2, 1e2))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10))
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
        )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=123
    )

    gp.fit(X, y)

    pred, std = gp.predict(grid, return_std=True)

    out[name] = pred
    out[f"{name}_std"] = std

out["sigma"] = np.exp(out["log_sigma"])

out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("Saved:", OUT_PATH)
print(out.head())
