"""Compare independent annual- and monthly-sample GEV parameter estimates.

This reproduces the project's original sensitivity experiment:

1. Define one true spatial GEV parameter field.
2. Independently simulate 45 annual observations per grid point.
3. Independently simulate 45 x 12 = 540 monthly observations per grid point.
4. Load the original pretrained Fast NN without retraining.
5. Apply the original RBF/Matérn GP spatial-smoothing stage.
6. Plot true, annual-GP and monthly-GP parameter surfaces in a 3 x 3 figure.

This simulated comparison is deliberately different from real-data
preprocessing. Real monthly data must be genuine monthly maxima calculated
from daily maximum temperature before they are passed to the pretrained NN.
"""

from __future__ import annotations

import hashlib
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import griddata
from scipy.stats import genextreme as gev
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    Matern,
    RBF,
    WhiteKernel,
)
from shapely import affinity, contains_xy


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from gev_nn import GEVNet, estimate_one  # noqa: E402
from project_paths import (  # noqa: E402
    FIGURE_DIR,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    SHAPEFILE_DIR,
)
from spatial_coordinates import (  # noqa: E402
    add_twd97_km_columns,
    center_train_test_coordinates,
)


MODEL_PATH = MODEL_DIR / "best_baseline_model.pth"
SHAPEFILE_PATH = (
    SHAPEFILE_DIR
    / "ne_50m_admin_0_countries"
    / "ne_50m_admin_0_countries.shp"
)
FIG_DIR = FIGURE_DIR
TABLE_DIR = PROCESSED_DATA_DIR
FIG_PATH = FIG_DIR / "annual_vs_monthly_block_maxima_comparison.png"
ERROR_PATH = TABLE_DIR / "annual_vs_monthly_block_maxima_error.csv"
LINKAGE_PATH = TABLE_DIR / "annual_vs_monthly_block_maxima_definition.csv"
TRUTH_PATH = TABLE_DIR / "annual_monthly_true_parameters.csv"
ANNUAL_PREDICTION_PATH = TABLE_DIR / "annual_45_nn_predictions.csv"
MONTHLY_PREDICTION_PATH = TABLE_DIR / "monthly_540_nn_predictions.csv"
ANNUAL_GP_PREDICTION_PATH = TABLE_DIR / "annual_45_gp_predictions.csv"
MONTHLY_GP_PREDICTION_PATH = TABLE_DIR / "monthly_540_gp_predictions.csv"
GP_SEARCH_PATH = TABLE_DIR / "annual_monthly_gp_kernel_search.csv"
MODEL_AUDIT_PATH = TABLE_DIR / "annual_monthly_model_audit.csv"

SIM_SEED = 20260525
SIM_YEARS = 45
SIM_MONTHS = 12
SIM_GRID_STEP = 0.50
GP_LENGTH_SCALES = (10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0)
GP_MATERN_NUS = (0.5, 1.5, 2.5)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_taiwan_boundary():
    world = gpd.read_file(SHAPEFILE_PATH)
    name_col = "ADMIN" if "ADMIN" in world.columns else "NAME"
    taiwan = world[
        world[name_col].astype(str).str.contains("Taiwan", case=False, na=False)
    ].to_crs(epsg=4326)
    if taiwan.empty:
        raise ValueError("Taiwan polygon was not found in the configured shapefile")
    return taiwan


def make_simulation_grid(taiwan):
    lon_min, lat_min, lon_max, lat_max = taiwan.total_bounds
    pad = 0.25
    lon_min, lon_max = lon_min - pad, lon_max + pad
    lat_min, lat_max = lat_min - pad, lat_max + pad
    lons = np.arange(
        np.floor(lon_min / SIM_GRID_STEP) * SIM_GRID_STEP,
        np.ceil(lon_max / SIM_GRID_STEP) * SIM_GRID_STEP + SIM_GRID_STEP / 2,
        SIM_GRID_STEP,
    )
    lats = np.arange(
        np.floor(lat_min / SIM_GRID_STEP) * SIM_GRID_STEP,
        np.ceil(lat_max / SIM_GRID_STEP) * SIM_GRID_STEP + SIM_GRID_STEP / 2,
        SIM_GRID_STEP,
    )
    grid = add_twd97_km_columns(
        pd.DataFrame(
            [(lon, lat) for lat in lats for lon in lons],
            columns=["lon", "lat"],
        )
    )
    grid["station"] = (
        "SIM"
        + grid["lon"].map(lambda value: f"{value:.2f}")
        + "_"
        + grid["lat"].map(lambda value: f"{value:.2f}")
    )

    lon_s = (grid["x_km"] - grid["x_km"].mean()) / grid["x_km"].std()
    lat_s = (grid["y_km"] - grid["y_km"].mean()) / grid["y_km"].std()

    # These are the true MONTHLY-block GEV parameters.  A negative xi is used
    # because temperature extremes have a physically bounded upper tail.
    grid["true_mu"] = (
        30.0
        + 1.6 * lat_s
        - 0.7 * lon_s
        + 0.9 * np.sin(np.pi * lon_s)
        + 0.6 * np.cos(np.pi * lat_s / 1.5)
    )
    grid["true_log_sigma"] = (
        np.log(1.25)
        + 0.10 * lat_s
        + 0.08 * np.cos(np.pi * lon_s / 1.8)
    )
    grid["true_sigma"] = np.exp(grid["true_log_sigma"])
    grid["true_xi"] = (
        0.08
        + 0.04 * np.sin(np.pi * lat_s / 1.4)
        - 0.025 * np.cos(np.pi * lon_s / 1.7)
    ).clip(-0.20, 0.30)
    return grid


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GEVNet().to(device)
    try:
        state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def simulate_independent_samples(truth, rng):
    """Draw independent annual-45 and monthly-540 samples from one GEV truth."""

    annual_columns = {"year": np.arange(1, SIM_YEARS + 1)}
    monthly_columns = {
        "year": np.repeat(np.arange(1, SIM_YEARS + 1), SIM_MONTHS),
        "month": np.tile(np.arange(1, SIM_MONTHS + 1), SIM_YEARS),
    }

    for row in truth.itertuples(index=False):
        # scipy.stats.genextreme uses c = -xi.
        annual_columns[row.station] = gev.rvs(
            c=-row.true_xi,
            loc=row.true_mu,
            scale=row.true_sigma,
            size=SIM_YEARS,
            random_state=rng,
        )
        monthly_columns[row.station] = gev.rvs(
            c=-row.true_xi,
            loc=row.true_mu,
            scale=row.true_sigma,
            size=SIM_YEARS * SIM_MONTHS,
            random_state=rng,
        )

    return pd.DataFrame(annual_columns), pd.DataFrame(monthly_columns)


def estimate_grid(block_df, truth, scenario, id_cols, model, device):
    rows = []
    station_cols = [column for column in block_df.columns if column not in id_cols]
    for station in station_cols:
        values = block_df[station].to_numpy(dtype=float)
        mu_hat, sigma_hat, shape_c_hat = estimate_one(
            model,
            values,
            device,
        )
        rows.append(
            {
                "scenario": scenario,
                "station": station,
                "n_obs": len(values),
                "mu_hat": float(mu_hat),
                "sigma_hat": float(sigma_hat),
                "xi_hat": float(-shape_c_hat),
            }
        )
    return pd.DataFrame(rows).merge(
        truth[
            [
                "station",
                "lon",
                "lat",
                "x_km",
                "y_km",
                "true_mu",
                "true_sigma",
                "true_xi",
            ]
        ],
        on="station",
        how="left",
        validate="one_to_one",
    )


def metric_coordinates(frame):
    projected = add_twd97_km_columns(frame)
    coordinates, _, _ = center_train_test_coordinates(
        projected[["x_km", "y_km"]].to_numpy(dtype=float)
    )
    return coordinates


def make_gp_kernel(kernel_name, length_scale, nu=None):
    if kernel_name == "RBF":
        spatial = RBF(
            length_scale=length_scale,
            length_scale_bounds=(1.0, 500.0),
        )
    elif kernel_name == "Matern":
        spatial = Matern(
            length_scale=length_scale,
            length_scale_bounds=(1.0, 500.0),
            nu=nu,
        )
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")
    return (
        C(1.0, constant_value_bounds=(1e-2, 1e2))
        * spatial
        + WhiteKernel(
            noise_level=1e-3,
            noise_level_bounds=(1e-8, 1e-1),
        )
    )


def gp_smooth_predictions(predictions, truth, scenario):
    """Select an RBF/Matérn GP for each parameter using known simulation truth."""

    coordinates = metric_coordinates(predictions)
    output = truth[
        [
            "station",
            "lon",
            "lat",
            "x_km",
            "y_km",
            "true_mu",
            "true_sigma",
            "true_xi",
        ]
    ].copy()
    search_rows = []

    candidates = [
        ("RBF", length_scale, np.nan)
        for length_scale in GP_LENGTH_SCALES
    ]
    candidates.extend(
        ("Matern", length_scale, nu)
        for nu in GP_MATERN_NUS
        for length_scale in GP_LENGTH_SCALES
    )

    for parameter in ("mu", "sigma", "xi"):
        source = predictions[f"{parameter}_hat"].to_numpy(dtype=float)
        target = truth[f"true_{parameter}"].to_numpy(dtype=float)
        target_scale = float(np.std(target))
        if target_scale <= 0:
            target_scale = 1.0

        candidate_results = []
        for kernel_name, length_scale, nu in candidates:
            gp = GaussianProcessRegressor(
                kernel=make_gp_kernel(
                    kernel_name,
                    length_scale,
                    None if pd.isna(nu) else float(nu),
                ),
                n_restarts_optimizer=2,
                normalize_y=True,
                random_state=SIM_SEED,
            )
            gp.fit(coordinates, source)
            smoothed = gp.predict(coordinates)
            error = smoothed - target
            row = {
                "scenario": scenario,
                "parameter": parameter,
                "kernel": kernel_name,
                "initial_length_scale": float(length_scale),
                "nu": nu,
                "rmse": float(np.sqrt(np.mean(error**2))),
                "standardized_rmse": float(
                    np.sqrt(np.mean((error / target_scale) ** 2))
                ),
                "log_marginal_likelihood": float(
                    gp.log_marginal_likelihood(gp.kernel_.theta)
                ),
                "optimized_kernel": str(gp.kernel_),
                "_prediction": smoothed,
            }
            candidate_results.append(row)

        best = min(
            candidate_results,
            key=lambda row: row["standardized_rmse"],
        )
        output[f"{parameter}_hat"] = best["_prediction"]
        output[f"{parameter}_kernel"] = best["kernel"]
        output[f"{parameter}_optimized_kernel"] = best["optimized_kernel"]
        for row in candidate_results:
            saved = {key: value for key, value in row.items() if key != "_prediction"}
            saved["selected"] = row is best
            search_rows.append(saved)

    output["scenario"] = f"{scenario}_gp"
    output["n_obs"] = int(predictions["n_obs"].iloc[0])
    return output, pd.DataFrame(search_rows)


def summarize_error(predictions):
    rows = []
    for scenario, frame in predictions.items():
        for parameter in ("mu", "sigma", "xi"):
            truth = frame[f"true_{parameter}"].to_numpy(dtype=float)
            estimate = frame[f"{parameter}_hat"].to_numpy(dtype=float)
            error = estimate - truth
            truth_sd = float(np.std(truth))
            rows.append(
                {
                    "scenario": scenario,
                    "parameter": parameter,
                    "n_obs_per_grid": int(frame["n_obs"].iloc[0]),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "nrmse_by_spatial_sd": (
                        float(np.sqrt(np.mean(error**2)) / truth_sd)
                        if truth_sd > 0
                        else np.nan
                    ),
                    "mae": float(np.mean(np.abs(error))),
                    "bias": float(np.mean(error)),
                    "correlation": float(np.corrcoef(truth, estimate)[0, 1]),
                }
            )
    return pd.DataFrame(rows)


def interpolate_surface(frame, value_col, x_grid_km, y_grid_km, mask):
    values = griddata(
        frame[["x_km", "y_km"]].to_numpy(),
        frame[value_col].to_numpy(),
        (x_grid_km, y_grid_km),
        method="cubic",
    )
    return np.where(mask, values, np.nan)


def plot_comparison(
    taiwan,
    truth,
    annual_gp,
    monthly_gp,
    error_summary,
):
    taiwan_km = taiwan.to_crs(epsg=3826).copy()
    taiwan_km.geometry = taiwan_km.geometry.apply(
        lambda geometry: affinity.scale(
            geometry,
            xfact=0.001,
            yfact=0.001,
            origin=(0.0, 0.0),
        )
    )
    x_min, y_min, x_max, y_max = taiwan_km.total_bounds
    pad_km = 20.0
    x_min, x_max = x_min - pad_km, x_max + pad_km
    y_min, y_max = y_min - pad_km, y_max + pad_km
    plot_x = np.linspace(x_min, x_max, 220)
    plot_y = np.linspace(y_min, y_max, 260)
    x_grid_km, y_grid_km = np.meshgrid(plot_x, plot_y)
    mask = contains_xy(
        taiwan_km.geometry.union_all(),
        x_grid_km,
        y_grid_km,
    )

    row_specs = [
        ("True parameter surfaces", truth, ["true_mu", "true_sigma", "true_xi"]),
        ("Annual GP prediction (n=45)", annual_gp, ["mu_hat", "sigma_hat", "xi_hat"]),
        ("Monthly GP prediction (n=540)", monthly_gp, ["mu_hat", "sigma_hat", "xi_hat"]),
    ]
    column_names = ["Location μ (°C)", "Scale σ (°C)", "Shape ξ"]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12.2, 10.8),
        dpi=180,
        constrained_layout=True,
    )

    for row_index, (row_name, frame, columns) in enumerate(row_specs):
        for col_index, (column, parameter) in enumerate(
            zip(columns, ("mu", "sigma", "xi"))
        ):
            ax = axes[row_index, col_index]
            panel_values = frame[column].to_numpy(dtype=float)
            vmin = float(np.nanmin(panel_values))
            vmax = float(np.nanmax(panel_values))
            surface = interpolate_surface(
                frame,
                column,
                x_grid_km,
                y_grid_km,
                mask,
            )
            mesh = ax.pcolormesh(
                x_grid_km,
                y_grid_km,
                surface,
                shading="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            taiwan_km.boundary.plot(ax=ax, color="#222222", linewidth=0.55)
            ax.set_aspect("equal")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("TWD97 Easting (km)")
            if col_index == 0:
                ax.set_ylabel(f"{row_name}\nTWD97 Northing (km)")
            else:
                ax.set_ylabel("TWD97 Northing (km)")
            if row_index == 0:
                ax.set_title(column_names[col_index])
            fig.colorbar(mesh, ax=ax, shrink=0.74)

            if row_index in (1, 2):
                scenario = "annual_45_gp" if row_index == 1 else "monthly_540_gp"
                metric = error_summary[
                    (error_summary["scenario"] == scenario)
                    & (error_summary["parameter"] == parameter)
                ].iloc[0]
                ax.text(
                    0.02,
                    0.02,
                    f"RMSE={metric['rmse']:.3f}\nr={metric['correlation']:.3f}",
                    transform=ax.transAxes,
                    va="bottom",
                    ha="left",
                    fontsize=7.5,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.78,
                        "pad": 2,
                    },
                )

    fig.suptitle(
        "True and predicted gridded simulated annual/monthly data\n"
        "(each panel uses its own color scale)",
        fontsize=13,
    )
    fig.savefig(FIG_PATH, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    taiwan = load_taiwan_boundary()
    truth = make_simulation_grid(taiwan)

    rng = np.random.default_rng(SIM_SEED)
    annual, monthly = simulate_independent_samples(truth, rng)

    model, device = load_model()
    annual_pred = estimate_grid(
        annual,
        truth,
        "annual_45",
        ["year"],
        model,
        device,
    )
    monthly_pred = estimate_grid(
        monthly,
        truth,
        "monthly_540",
        ["year", "month"],
        model,
        device,
    )
    annual_gp, annual_search = gp_smooth_predictions(
        annual_pred,
        truth,
        "annual_45",
    )
    monthly_gp, monthly_search = gp_smooth_predictions(
        monthly_pred,
        truth,
        "monthly_540",
    )

    errors = summarize_error(
        {
            "annual_45": annual_pred,
            "monthly_540": monthly_pred,
            "annual_45_gp": annual_gp,
            "monthly_540_gp": monthly_gp,
        }
    )
    errors.to_csv(ERROR_PATH, index=False, encoding="utf-8-sig")
    truth.to_csv(TRUTH_PATH, index=False, encoding="utf-8-sig")
    annual_pred.to_csv(
        ANNUAL_PREDICTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    monthly_pred.to_csv(
        MONTHLY_PREDICTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    annual_gp.to_csv(
        ANNUAL_GP_PREDICTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    monthly_gp.to_csv(
        MONTHLY_GP_PREDICTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(
        [annual_search, monthly_search],
        ignore_index=True,
    ).to_csv(GP_SEARCH_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "weight_path": str(MODEL_PATH),
                "weight_bytes": MODEL_PATH.stat().st_size,
                "weight_sha256": hashlib.sha256(
                    MODEL_PATH.read_bytes()
                ).hexdigest(),
                "network": "11-512-512-512-128-128-3",
                "trainable_parameters": int(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
                "retrained": False,
                "paper_training_parameter_sampling": (
                    "mu~Uniform(1,50); sigma~Uniform(0.1,40); "
                    "xi~Uniform(-0.4,1)"
                ),
                "spatial_truth_note": (
                    "smooth benchmark surfaces inside the paper parameter ranges; "
                    "paper does not define a spatial field"
                ),
            }
        ]
    ).to_csv(MODEL_AUDIT_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "scenario": "annual_45",
                "definition": "45 independent draws from the common true GEV field",
                "n_obs_per_grid": SIM_YEARS,
            },
            {
                "scenario": "monthly_540",
                "definition": "540 independent draws from the common true GEV field",
                "n_obs_per_grid": SIM_YEARS * SIM_MONTHS,
            },
        ]
    ).to_csv(LINKAGE_PATH, index=False, encoding="utf-8-sig")

    plot_comparison(
        taiwan,
        truth,
        annual_gp,
        monthly_gp,
        errors,
    )

    print(errors.to_string(index=False))
    selected = pd.concat(
        [annual_search, monthly_search],
        ignore_index=True,
    )
    print("\nSelected GP kernels:")
    print(
        selected[selected["selected"]][
            [
                "scenario",
                "parameter",
                "kernel",
                "initial_length_scale",
                "nu",
                "standardized_rmse",
                "optimized_kernel",
            ]
        ].to_string(index=False)
    )
    print(f"Figure: {FIG_PATH}")
    print(f"Error table: {ERROR_PATH}")


if __name__ == "__main__":
    main()
