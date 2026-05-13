import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import genextreme as gev
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel

from estimate_real_params import GEVNet, estimate_one


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOC_PATH = os.path.join(PROJECT_ROOT, "data", "original_data", "station_location.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_baseline_model.pth")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "simulated", "spatial_gev")

N_YEARS = 45
N_MONTHS = 12
GRID_SIZE = 80
SEED = 123


def standardize_coords(coords):
    mean = coords.mean(axis=0)
    std = coords.std(axis=0)
    std[std == 0] = 1.0
    return (coords - mean) / std, mean, std


def rbf_covariance(coords, variance=1.0, length_scale=1.0, jitter=1e-8):
    diff = coords[:, None, :] - coords[None, :, :]
    sq_dist = np.sum(diff**2, axis=2)
    cov = variance * np.exp(-0.5 * sq_dist / (length_scale**2))
    cov += jitter * np.eye(coords.shape[0])
    return cov


def sample_spatial_effect(coords_std, rng, variance, length_scale):
    cov = rbf_covariance(coords_std, variance=variance, length_scale=length_scale)
    return rng.multivariate_normal(mean=np.zeros(coords_std.shape[0]), cov=cov)


def make_grid(locs):
    lon_grid = np.linspace(locs["lon"].min(), locs["lon"].max(), GRID_SIZE)
    lat_grid = np.linspace(locs["lat"].min(), locs["lat"].max(), GRID_SIZE)
    grid_raw = np.array([[lon, lat] for lat in lat_grid for lon in lon_grid])
    return pd.DataFrame(grid_raw, columns=["lon", "lat"])


def generate_true_station_params(locs, rng):
    coords = locs[["lon", "lat"]].to_numpy(dtype=np.float64)
    coords_std, _, _ = standardize_coords(coords)
    lon_s = coords_std[:, 0]
    lat_s = coords_std[:, 1]

    w_mu = sample_spatial_effect(coords_std, rng, variance=2.5**2, length_scale=0.9)
    w_log_sigma = sample_spatial_effect(coords_std, rng, variance=0.12**2, length_scale=1.1)
    w_xi = sample_spatial_effect(coords_std, rng, variance=0.035**2, length_scale=1.0)

    true_mu = 80.0 + 7.0 * lat_s + 3.0 * lon_s + w_mu
    true_log_sigma = np.log(12.0) + 0.12 * lat_s - 0.08 * lon_s + w_log_sigma
    true_sigma = np.exp(true_log_sigma)
    true_xi = 0.12 + 0.03 * lat_s + w_xi
    true_xi = np.clip(true_xi, -0.25, 0.35)

    out = locs.copy()
    out["true_mu"] = true_mu
    out["true_log_sigma"] = true_log_sigma
    out["true_sigma"] = true_sigma
    out["true_xi"] = true_xi
    return out


def simulate_annual_max(true_params, rng):
    years = np.arange(1, N_YEARS + 1)
    out = pd.DataFrame({"year": years})

    for row in true_params.itertuples(index=False):
        # scipy.stats.genextreme uses c = -xi.
        values = gev.rvs(
            c=-row.true_xi,
            loc=row.true_mu,
            scale=row.true_sigma,
            size=N_YEARS,
            random_state=rng,
        )
        out[row.station] = values

    return out


def simulate_monthly_max(true_params, rng):
    blocks = [
        {"year": year, "month": month}
        for year in range(1, N_YEARS + 1)
        for month in range(1, N_MONTHS + 1)
    ]
    out = pd.DataFrame(blocks)
    n_blocks = len(out)

    for row in true_params.itertuples(index=False):
        # scipy.stats.genextreme uses c = -xi.
        values = gev.rvs(
            c=-row.true_xi,
            loc=row.true_mu,
            scale=row.true_sigma,
            size=n_blocks,
            random_state=rng,
        )
        out[row.station] = values

    return out


def estimate_station_params_with_nn(block_max, locs, id_cols):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GEVNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    rows = []
    station_cols = [col for col in block_max.columns if col not in id_cols]

    for station in station_cols:
        y = block_max[station].dropna().to_numpy(dtype=np.float64)
        mu_hat, sigma_hat, c_hat = estimate_one(model, y, device)

        rows.append(
            {
                "station": station,
                "n_obs": len(y),
                "mu_hat": mu_hat,
                "sigma_hat": sigma_hat,
                "log_sigma_hat": np.log(sigma_hat),
                "xi_hat": -c_hat,
                "shape_c_hat": c_hat,
            }
        )

    est = pd.DataFrame(rows)
    return est.merge(locs, on="station", how="left")


def make_gp_kernel(kernel_type):
    if kernel_type == "rbf":
        spatial_kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10))
    elif kernel_type == "matern":
        spatial_kernel = Matern(
            length_scale=1.0,
            length_scale_bounds=(1e-2, 10),
            nu=1.5,
        )
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    return (
        C(1.0, (1e-2, 1e2))
        * spatial_kernel
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    )


def krige_params(station_params, targets, kernel_type="rbf"):
    coords_raw = station_params[["lon", "lat"]].to_numpy(dtype=np.float64)
    coords, coord_mean, coord_std = standardize_coords(coords_raw)

    grid = make_grid(station_params)
    grid_coords = (grid[["lon", "lat"]].to_numpy(dtype=np.float64) - coord_mean) / coord_std

    out = grid.copy()
    for out_name, source_col in targets.items():
        y = station_params[source_col].to_numpy(dtype=np.float64)
        kernel = make_gp_kernel(kernel_type)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=SEED,
        )
        gp.fit(coords, y)
        pred, std = gp.predict(grid_coords, return_std=True)
        out[out_name] = pred
        out[f"{out_name}_std"] = std

    if "log_sigma" in out.columns:
        out["sigma"] = np.exp(out["log_sigma"])

    return out


def summarize_station_error(compare):
    rows = []
    pairs = [
        ("mu", "true_mu", "mu_hat"),
        ("sigma", "true_sigma", "sigma_hat"),
        ("xi", "true_xi", "xi_hat"),
    ]

    for name, true_col, est_col in pairs:
        err = compare[est_col] - compare[true_col]
        rows.append(
            {
                "param": name,
                "rmse": float(np.sqrt(np.mean(err**2))),
                "mae": float(np.mean(np.abs(err))),
                "correlation": float(compare[[true_col, est_col]].corr().iloc[0, 1]),
            }
        )

    return pd.DataFrame(rows)


def summarize_grid_error(true_grid, estimated_grid, method):
    rows = []
    pairs = [
        ("mu", "true_mu", "mu"),
        ("sigma", "true_sigma", "sigma"),
        ("xi", "true_xi", "xi"),
    ]

    merged = true_grid[["lon", "lat"] + [p[1] for p in pairs]].merge(
        estimated_grid[["lon", "lat"] + [p[2] for p in pairs]],
        on=["lon", "lat"],
        how="inner",
    )

    for name, true_col, est_col in pairs:
        err = merged[est_col] - merged[true_col]
        rows.append(
            {
                "method": method,
                "param": name,
                "rmse": float(np.sqrt(np.mean(err**2))),
                "mae": float(np.mean(np.abs(err))),
                "correlation": float(merged[[true_col, est_col]].corr().iloc[0, 1]),
            }
        )

    return pd.DataFrame(rows)


def save_analysis_outputs(
    scenario,
    block_max,
    true_station,
    nn_station,
    true_grid,
    nn_grid_rbf,
    nn_grid_matern,
    station_error,
    grid_error,
):
    prefix = f"spatial_{scenario}"

    block_max.to_csv(
        os.path.join(OUT_DIR, f"{prefix}_max_25stations.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    nn_station.to_csv(
        os.path.join(OUT_DIR, f"{prefix}_station_nn_estimates.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    true_station.merge(
        nn_station[
            [
                "station",
                "mu_hat",
                "sigma_hat",
                "log_sigma_hat",
                "xi_hat",
                "shape_c_hat",
            ]
        ],
        on="station",
        how="left",
    ).to_csv(
        os.path.join(OUT_DIR, f"{prefix}_station_true_vs_nn.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    true_grid.to_csv(
        os.path.join(OUT_DIR, f"{prefix}_grid_true_params.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    nn_grid_rbf.to_csv(
        os.path.join(OUT_DIR, f"{prefix}_grid_nn_rbf_kriging_params.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    nn_grid_matern.to_csv(
        os.path.join(OUT_DIR, f"{prefix}_grid_nn_matern_kriging_params.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    station_error.to_csv(
        os.path.join(OUT_DIR, f"{prefix}_station_error_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    grid_error.to_csv(
        os.path.join(OUT_DIR, f"{prefix}_grid_error_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def copy_backward_compatible_annual_outputs(
    block_max,
    true_station,
    nn_station,
    true_grid,
    nn_grid_rbf,
    nn_grid_matern,
    station_error,
    grid_error,
):
    true_station.to_csv(
        os.path.join(OUT_DIR, "spatial_station_true_params.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    block_max.to_csv(
        os.path.join(OUT_DIR, "spatial_annual_max_25stations.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    nn_station.to_csv(
        os.path.join(OUT_DIR, "spatial_station_nn_estimates.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    true_station.merge(
        nn_station[
            [
                "station",
                "mu_hat",
                "sigma_hat",
                "log_sigma_hat",
                "xi_hat",
                "shape_c_hat",
            ]
        ],
        on="station",
        how="left",
    ).to_csv(
        os.path.join(OUT_DIR, "spatial_station_true_vs_nn.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    true_grid.to_csv(
        os.path.join(OUT_DIR, "spatial_grid_true_params.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    nn_grid_rbf.to_csv(
        os.path.join(OUT_DIR, "spatial_grid_nn_rbf_kriging_params.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    nn_grid_matern.to_csv(
        os.path.join(OUT_DIR, "spatial_grid_nn_matern_kriging_params.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    nn_grid_rbf.to_csv(
        os.path.join(OUT_DIR, "spatial_grid_nn_kriging_params.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    station_error.to_csv(
        os.path.join(OUT_DIR, "spatial_station_error_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    grid_error.to_csv(
        os.path.join(OUT_DIR, "spatial_grid_error_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def run_analysis(scenario, block_max, true_station, locs, true_grid):
    id_cols = ["year"] if scenario == "annual" else ["year", "month"]
    nn_station = estimate_station_params_with_nn(block_max, locs, id_cols=id_cols)

    compare = true_station.merge(
        nn_station[
            [
                "station",
                "mu_hat",
                "sigma_hat",
                "log_sigma_hat",
                "xi_hat",
                "shape_c_hat",
            ]
        ],
        on="station",
        how="left",
    )

    nn_grid_rbf = krige_params(
        nn_station,
        {
            "mu": "mu_hat",
            "log_sigma": "log_sigma_hat",
            "xi": "xi_hat",
        },
        kernel_type="rbf",
    )

    nn_grid_matern = krige_params(
        nn_station,
        {
            "mu": "mu_hat",
            "log_sigma": "log_sigma_hat",
            "xi": "xi_hat",
        },
        kernel_type="matern",
    )

    station_error = summarize_station_error(compare)
    station_error.insert(0, "scenario", scenario)

    grid_error = pd.concat(
        [
            summarize_grid_error(true_grid, nn_grid_rbf, method="RBF"),
            summarize_grid_error(true_grid, nn_grid_matern, method="Matern"),
        ],
        ignore_index=True,
    )
    grid_error.insert(0, "scenario", scenario)

    return {
        "block_max": block_max,
        "nn_station": nn_station,
        "nn_grid_rbf": nn_grid_rbf,
        "nn_grid_matern": nn_grid_matern,
        "station_error": station_error,
        "grid_error": grid_error,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    locs = pd.read_csv(LOC_PATH)
    locs["station"] = locs["station"].astype(str).str.strip().str.upper()

    true_station = generate_true_station_params(locs, rng)
    annual_max = simulate_annual_max(true_station, rng)
    monthly_max = simulate_monthly_max(true_station, rng)

    true_grid = krige_params(
        true_station,
        {
            "true_mu": "true_mu",
            "true_log_sigma": "true_log_sigma",
            "true_xi": "true_xi",
        },
        kernel_type="rbf",
    )
    true_grid["true_sigma"] = np.exp(true_grid["true_log_sigma"])

    annual = run_analysis("annual", annual_max, true_station, locs, true_grid)
    monthly = run_analysis("monthly", monthly_max, true_station, locs, true_grid)

    true_station.to_csv(
        os.path.join(OUT_DIR, "spatial_station_true_params.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    save_analysis_outputs("annual", true_station=true_station, true_grid=true_grid, **annual)
    save_analysis_outputs("monthly", true_station=true_station, true_grid=true_grid, **monthly)
    copy_backward_compatible_annual_outputs(
        true_station=true_station,
        true_grid=true_grid,
        **annual,
    )

    sensitivity_error = pd.concat(
        [annual["grid_error"], monthly["grid_error"]],
        ignore_index=True,
    )
    sensitivity_error.to_csv(
        os.path.join(OUT_DIR, "spatial_annual_monthly_grid_error_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("Saved simulation outputs to:", OUT_DIR)
    print("Annual station-level NN error:")
    print(annual["station_error"])
    print("Monthly station-level NN error:")
    print(monthly["station_error"])
    print("Annual vs monthly grid-level kriging error:")
    print(sensitivity_error)


if __name__ == "__main__":
    main()
