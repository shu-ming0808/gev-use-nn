import os

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel

from simulate_spatial_gev import SEED, standardize_coords


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(PROJECT_ROOT, "data", "simulated", "spatial_gev")
OUT_PATH = os.path.join(SIM_DIR, "spatial_kernel_gridsearch_rmse.csv")
MONTHLY_COMPAT_OUT_PATH = os.path.join(SIM_DIR, "spatial_monthly_kernel_gridsearch_rmse.csv")


TARGETS = [
    ("mu", "true_mu", "mu_hat"),
    ("sigma", "true_sigma", "sigma_hat"),
    ("xi", "true_xi", "xi_hat"),
]


def default_length_scales():
    values = np.round(np.arange(0.05, 5.0, 0.1), 2)
    return np.unique(np.append(values, 5.0))


def fixed_kernel(kernel_type, length_scale, nu=None):
    if kernel_type == "RBF":
        spatial_kernel = RBF(length_scale=length_scale, length_scale_bounds="fixed")
    elif kernel_type == "Matern":
        spatial_kernel = Matern(
            length_scale=length_scale,
            length_scale_bounds="fixed",
            nu=nu,
        )
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    return (
        C(1.0, constant_value_bounds="fixed")
        * spatial_kernel
        + WhiteKernel(noise_level=1e-4, noise_level_bounds="fixed")
    )


def load_inputs(scenario, sim_dir=SIM_DIR):
    true_grid = pd.read_csv(os.path.join(sim_dir, f"spatial_{scenario}_grid_true_params.csv"))
    station = pd.read_csv(os.path.join(sim_dir, f"spatial_{scenario}_station_nn_estimates.csv"))
    return true_grid, station


def predict_grid(station, true_grid, source_col, kernel_type, length_scale, nu=None):
    coords_raw = station[["lon", "lat"]].to_numpy(dtype=np.float64)
    coords, coord_mean, coord_std = standardize_coords(coords_raw)
    grid_coords = (true_grid[["lon", "lat"]].to_numpy(dtype=np.float64) - coord_mean) / coord_std

    y = station[source_col].to_numpy(dtype=np.float64)
    gp = GaussianProcessRegressor(
        kernel=fixed_kernel(kernel_type, length_scale, nu=nu),
        optimizer=None,
        normalize_y=True,
        random_state=SEED,
    )
    gp.fit(coords, y)
    return gp.predict(grid_coords)


def rmse(y_true, y_pred):
    err = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sqrt(np.mean(err**2)))


def evaluate_one(scenario, true_grid, station, kernel_type, length_scale, nu=None):
    rows = []
    raw_errors = []
    scaled_errors = []

    for param, true_col, source_col in TARGETS:
        pred = predict_grid(
            station=station,
            true_grid=true_grid,
            source_col=source_col,
            kernel_type=kernel_type,
            length_scale=length_scale,
            nu=nu,
        )
        err = np.asarray(pred) - true_grid[true_col].to_numpy(dtype=np.float64)
        scale = true_grid[true_col].std()
        if scale == 0:
            scale = 1.0
        raw_errors.append(err)
        scaled_errors.append(err / scale)
        rows.append(
            {
                "scenario": scenario,
                "kernel": kernel_type,
                "param": param,
                "length_scale": length_scale,
                "nu": np.nan if nu is None else nu,
                "rmse": float(np.sqrt(np.mean(err**2))),
            }
        )

    rows.append(
        {
            "scenario": scenario,
            "kernel": kernel_type,
            "param": "overall",
            "length_scale": length_scale,
            "nu": np.nan if nu is None else nu,
            "rmse": float(np.sqrt(np.mean(np.concatenate(raw_errors) ** 2))),
        }
    )
    rows.append(
        {
            "scenario": scenario,
            "kernel": kernel_type,
            "param": "overall_standardized",
            "length_scale": length_scale,
            "nu": np.nan if nu is None else nu,
            "rmse": float(np.sqrt(np.mean(np.concatenate(scaled_errors) ** 2))),
        }
    )
    return rows


def run_grid_search(
    length_scales=None,
    matern_nus=(0.5, 2.5),
    scenarios=("annual", "monthly"),
    sim_dir=SIM_DIR,
    out_path=OUT_PATH,
):
    os.makedirs(sim_dir, exist_ok=True)
    if length_scales is None:
        length_scales = default_length_scales()

    rows = []

    for scenario in scenarios:
        true_grid, station = load_inputs(scenario=scenario, sim_dir=sim_dir)

        for length_scale in length_scales:
            rows.extend(
                evaluate_one(
                    scenario=scenario,
                    true_grid=true_grid,
                    station=station,
                    kernel_type="RBF",
                    length_scale=float(length_scale),
                )
            )

        for nu in matern_nus:
            for length_scale in length_scales:
                rows.extend(
                    evaluate_one(
                        scenario=scenario,
                        true_grid=true_grid,
                        station=station,
                        kernel_type="Matern",
                        length_scale=float(length_scale),
                        nu=float(nu),
                    )
                )

    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    result[result["scenario"] == "monthly"].to_csv(
        MONTHLY_COMPAT_OUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    return result


def main():
    result = run_grid_search()
    print("Saved:", OUT_PATH)
    print(result.head())
    print("Best RMSE by kernel and parameter:")
    print(
        result.loc[result.groupby(["scenario", "kernel", "param"])["rmse"].idxmin()]
        .sort_values(["scenario", "kernel", "param"])
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    main()
