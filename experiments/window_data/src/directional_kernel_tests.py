"""Six pre-specified directional GP-kernel tests for the simulation study.

The outcomes are original-station annual and monthly parameter recovery,
gridded annual and monthly parameter recovery, and gridded monthly RL50 and
RL100 recovery. Annual and monthly errors are never pooled in the confirmatory
table. Parameter recovery uses the mean squared standardized error jointly
over mu, sigma, and xi. Return-level recovery uses squared error in the
temperature unit.

Taiwan is partitioned into ten geographic K-means blocks. One paired loss is
calculated per block, so adjacent grid cells are not treated as independent
replicates. Original-station hypotheses favour Matérn; gridded hypotheses
favour RBF. Holm adjustment is applied across the six one-sided tests.

The gridded-input branch reproduces the original notebook experiment and its
published comparison table.  In particular, it uses the original pretrained
network inverse transform, sigma = exp(delta_star) * IQR, and the notebook's
GEV shape-to-xi sign convention.  Changing that transform changes the
experiment and no longer reproduces the displayed RMSE values.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
import sys
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from scipy.stats import genextreme as gev
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    Matern,
    RBF,
    WhiteKernel,
)


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
WINDOW_ROOT = SCRIPT_PATH.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from estimate_real_params import GEVNet, P_SET  # noqa: E402
from kriging_kernel_gridsearch import load_inputs  # noqa: E402
from spatial_coordinates import (  # noqa: E402
    add_twd97_km_columns,
    center_train_test_coordinates,
)


RANDOM_STATE = 20260525
N_SPATIAL_BLOCKS = 10
SIM_YEARS = 45
SIM_MONTHS = 12
SIM_GRID_STEP = 0.5
MODEL_PATH = PROJECT_ROOT / "models" / "best_baseline_model.pth"
SHAPEFILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "shapefile"
    / "ne_50m_admin_0_countries"
    / "ne_50m_admin_0_countries.shp"
)
SIM_DIR = PROJECT_ROOT / "data" / "simulated" / "spatial_gev"
ORIGINAL_SEARCH_PATH = SIM_DIR / "spatial_kernel_gridsearch_rmse.csv"
TABLE_DIR = WINDOW_ROOT / "results" / "tables"
REPORT_PATH = WINDOW_ROOT / "results" / "directional_kernel_test_report.md"

PRIMARY_PATH = TABLE_DIR / "directional_kernel_primary_tests.csv"
BLOCK_PATH = TABLE_DIR / "directional_kernel_spatial_blocks.csv"
DESCRIPTIVE_PATH = TABLE_DIR / "directional_kernel_descriptive_rmse.csv"

PARAMETERS = ("mu", "sigma", "xi")
SCENARIOS = ("annual", "monthly")
GRID_SCENARIO_LABELS = {
    "annual": "annual_45",
    "monthly": "monthly_540",
}


def metric_coordinates(frame: pd.DataFrame) -> np.ndarray:
    projected = (
        frame
        if {"x_km", "y_km"}.issubset(frame.columns)
        else add_twd97_km_columns(frame)
    )
    coordinates, _, _ = center_train_test_coordinates(
        projected[["x_km", "y_km"]].to_numpy(dtype=np.float64)
    )
    return coordinates


def geographic_block_labels(frame: pd.DataFrame) -> np.ndarray:
    return KMeans(
        n_clusters=N_SPATIAL_BLOCKS,
        random_state=RANDOM_STATE,
        n_init=50,
    ).fit_predict(metric_coordinates(frame))


def exact_one_sided_sign_flip(
    directional_contrasts: np.ndarray,
) -> tuple[float, float]:
    """Test whether the mean pre-oriented block contrast is greater than zero."""
    contrasts = np.asarray(directional_contrasts, dtype=np.float64)
    observed = float(np.mean(contrasts))
    signs = np.asarray(
        list(product((-1.0, 1.0), repeat=len(contrasts))),
        dtype=np.float64,
    )
    randomized = np.mean(signs * contrasts[None, :], axis=1)
    p_value = float(np.mean(randomized >= observed - 1e-12))
    return observed, p_value


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=np.float64)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * values[order]
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return pd.Series(adjusted, index=p_values.index)


def summarize_block_test(
    *,
    experiment: str,
    preferred_kernel: str,
    null_hypothesis: str,
    alternative_hypothesis: str,
    block_table: pd.DataFrame,
    loss_scale: str,
) -> dict:
    pivot = block_table.pivot(
        index="spatial_block",
        columns="kernel",
        values="MSE",
    )
    if set(pivot.columns) != {"RBF", "Matern"}:
        raise ValueError(f"Missing paired kernel losses for {experiment}.")

    if preferred_kernel == "Matern":
        contrast = pivot["RBF"] - pivot["Matern"]
    elif preferred_kernel == "RBF":
        contrast = pivot["Matern"] - pivot["RBF"]
    else:
        raise ValueError(preferred_kernel)

    observed, p_value = exact_one_sided_sign_flip(contrast.to_numpy())
    rbf_rmse = float(np.sqrt(block_table.loc[
        block_table["kernel"].eq("RBF"),
        "weighted_squared_error_sum",
    ].sum() / block_table.loc[
        block_table["kernel"].eq("RBF"),
        "n_cells",
    ].sum()))
    matern_rmse = float(np.sqrt(block_table.loc[
        block_table["kernel"].eq("Matern"),
        "weighted_squared_error_sum",
    ].sum() / block_table.loc[
        block_table["kernel"].eq("Matern"),
        "n_cells",
    ].sum()))

    preferred_rmse = matern_rmse if preferred_kernel == "Matern" else rbf_rmse
    other_rmse = rbf_rmse if preferred_kernel == "Matern" else matern_rmse
    return {
        "experiment": experiment,
        "preferred_kernel_in_H1": preferred_kernel,
        "H0": null_hypothesis,
        "H1": alternative_hypothesis,
        "n_spatial_blocks": len(pivot),
        "loss_scale": loss_scale,
        "RBF_RMSE": rbf_rmse,
        "Matern_RMSE": matern_rmse,
        "preferred_kernel_RMSE_reduction_pct": (
            100.0 * (other_rmse - preferred_rmse) / other_rmse
        ),
        "mean_directional_block_MSE_contrast": observed,
        "blocks_favouring_H1": int((contrast > 0).sum()),
        "one_sided_exact_p_value": p_value,
        "minimum_p_value_resolution": 1.0 / (2 ** len(pivot)),
        "decision_raw_alpha_0.05": (
            f"reject H0: {preferred_kernel} is significantly better"
            if p_value < 0.05 and observed > 0
            else f"do not reject H0: insufficient evidence that "
            f"{preferred_kernel} is better"
        ),
    }


def block_loss_table(
    *,
    experiment: str,
    coordinates: pd.DataFrame,
    cell_losses: dict[str, np.ndarray],
) -> pd.DataFrame:
    labels = geographic_block_labels(coordinates)
    rows = []
    for kernel, losses in cell_losses.items():
        losses = np.asarray(losses, dtype=np.float64)
        if len(losses) != len(coordinates):
            raise ValueError(f"Cell-loss length mismatch for {experiment}/{kernel}.")
        for block in range(N_SPATIAL_BLOCKS):
            mask = labels == block
            block_losses = losses[mask]
            rows.append(
                {
                    "experiment": experiment,
                    "spatial_block": block,
                    "kernel": kernel,
                    "n_cells": int(mask.sum()),
                    "MSE": float(block_losses.mean()),
                    "RMSE": float(np.sqrt(block_losses.mean())),
                    "weighted_squared_error_sum": float(block_losses.sum()),
                }
            )
    return pd.DataFrame(rows)


def original_descriptive_table(search: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        for parameter in (
            "mu",
            "sigma",
            "xi",
            "overall",
            "overall_standardized",
        ):
            part = search.loc[
                search["scenario"].eq(scenario)
                & search["param"].eq(parameter)
            ]
            for kernel, family in part.groupby("kernel"):
                best = family.loc[family["rmse"].idxmin()]
                rows.append(
                    {
                        "experiment": "original_station_input",
                        "scenario": scenario,
                        "parameter": parameter,
                        "kernel": kernel,
                        "RMSE": float(best["rmse"]),
                        "length_scale": float(best["length_scale"]),
                        "nu": best["nu"],
                        "metric": (
                            "standardized_RMSE"
                            if parameter == "overall_standardized"
                            else "RMSE"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def fit_station_to_grid_predictions(
    station: pd.DataFrame,
    truth: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    """Fit pre-specified kernels using station estimates only."""
    station = add_twd97_km_columns(station)
    truth = add_twd97_km_columns(truth)
    x_train, x_test, _ = center_train_test_coordinates(
        station[["x_km", "y_km"]].to_numpy(dtype=np.float64),
        truth[["x_km", "y_km"]].to_numpy(dtype=np.float64),
    )
    predictions = {"RBF": {}, "Matern": {}}

    for kernel, nu in (("RBF", None), ("Matern", 0.5)):
        for parameter in PARAMETERS:
            gp = GaussianProcessRegressor(
                kernel=optimized_kernel(kernel, nu),
                n_restarts_optimizer=5,
                normalize_y=True,
                random_state=RANDOM_STATE,
            )
            gp.fit(
                x_train,
                station[f"{parameter}_hat"].to_numpy(dtype=np.float64),
            )
            predictions[kernel][parameter] = gp.predict(x_test)
    return predictions


def parameter_recovery_losses(
    data: pd.DataFrame,
    predictions: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    losses = {}
    for kernel in ("RBF", "Matern"):
        squared_errors = []
        for parameter in PARAMETERS:
            true_values = data[f"true_{parameter}"].to_numpy(dtype=np.float64)
            scale = float(np.std(true_values))
            if scale <= 0:
                scale = 1.0
            squared_errors.append(
                ((predictions[kernel][parameter] - true_values) / scale) ** 2
            )
        losses[kernel] = np.mean(squared_errors, axis=0)
    return losses


def evaluate_original_station_input() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Evaluate annual and monthly station-input recovery separately."""
    search = pd.read_csv(ORIGINAL_SEARCH_PATH)
    descriptive = original_descriptive_table(search)
    block_tables = {}

    for scenario in SCENARIOS:
        truth, station = load_inputs(scenario, sim_dir=str(SIM_DIR))
        predictions = fit_station_to_grid_predictions(station, truth)
        losses = parameter_recovery_losses(truth, predictions)
        experiment = f"original_station_{scenario}"
        block_tables[experiment] = block_loss_table(
            experiment=experiment,
            coordinates=truth[["lon", "lat"]],
            cell_losses=losses,
        )
    return descriptive, block_tables


def make_gridded_truth() -> pd.DataFrame:
    world = gpd.read_file(SHAPEFILE_PATH)
    name_col = "ADMIN" if "ADMIN" in world.columns else "NAME"
    taiwan = world[
        world[name_col].astype(str).str.contains(
            "Taiwan",
            case=False,
            na=False,
        )
    ].to_crs(epsg=4326)
    if taiwan.empty:
        raise ValueError("Taiwan polygon was not found.")

    lon_min, lat_min, lon_max, lat_max = taiwan.total_bounds
    lon_min -= 0.25
    lon_max += 0.25
    lat_min -= 0.25
    lat_max += 0.25
    lons = np.arange(
        np.floor(lon_min / SIM_GRID_STEP) * SIM_GRID_STEP,
        np.ceil(lon_max / SIM_GRID_STEP) * SIM_GRID_STEP
        + SIM_GRID_STEP / 2,
        SIM_GRID_STEP,
    )
    lats = np.arange(
        np.floor(lat_min / SIM_GRID_STEP) * SIM_GRID_STEP,
        np.ceil(lat_max / SIM_GRID_STEP) * SIM_GRID_STEP
        + SIM_GRID_STEP / 2,
        SIM_GRID_STEP,
    )
    truth = add_twd97_km_columns(
        pd.DataFrame(
            [(lon, lat) for lat in lats for lon in lons],
            columns=["lon", "lat"],
        )
    )
    truth["station"] = (
        "SIM"
        + truth["lon"].map(lambda value: f"{value:.1f}")
        + "_"
        + truth["lat"].map(lambda value: f"{value:.1f}")
    )

    lon_s = (truth["x_km"] - truth["x_km"].mean()) / truth["x_km"].std()
    lat_s = (truth["y_km"] - truth["y_km"].mean()) / truth["y_km"].std()
    truth["true_mu"] = (
        30.0
        + 1.6 * lat_s
        - 0.7 * lon_s
        + 0.9 * np.sin(np.pi * lon_s)
        + 0.6 * np.cos(np.pi * lat_s / 1.5)
    )
    truth["true_sigma"] = np.exp(
        np.log(1.25)
        + 0.10 * lat_s
        + 0.08 * np.cos(np.pi * lon_s / 1.8)
    )
    truth["true_xi"] = (
        0.08
        + 0.04 * np.sin(np.pi * lat_s / 1.4)
        - 0.025 * np.cos(np.pi * lon_s / 1.7)
    ).clip(-0.20, 0.30)
    return truth


def load_pretrained_model() -> tuple[GEVNet, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GEVNet().to(device)
    try:
        state = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def original_notebook_nn_estimate(
    model: GEVNet,
    values: np.ndarray,
    device: str,
) -> tuple[float, float, float]:
    """Apply the exact inverse transform used by the screenshot experiment."""
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = float(q3 - q1)
    if iqr <= 1e-12:
        raise ValueError("IQR is too small.")

    standardized = (values - median) / iqr
    quantiles = np.quantile(standardized, P_SET)
    tensor = torch.tensor(
        quantiles,
        dtype=torch.float32,
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(tensor).cpu().numpy().ravel()
    mu_star, delta_star, shape_c_hat = prediction

    mu_hat = float(mu_star * iqr + median)
    sigma_hat = float(np.exp(delta_star) * iqr)
    xi_hat = float(-shape_c_hat)
    return mu_hat, sigma_hat, xi_hat


def simulate_gridded_nn_estimates(
    truth: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    annual_samples = {}
    monthly_samples = {}

    # Ordering deliberately matches the original notebook: first draw every
    # annual series, then draw every monthly series from the same RNG stream.
    for row in truth.itertuples(index=False):
        annual_samples[row.station] = gev.rvs(
            c=-row.true_xi,
            loc=row.true_mu,
            scale=row.true_sigma,
            size=SIM_YEARS,
            random_state=rng,
        )
    for row in truth.itertuples(index=False):
        monthly_samples[row.station] = gev.rvs(
            c=-row.true_xi,
            loc=row.true_mu,
            scale=row.true_sigma,
            size=SIM_YEARS * SIM_MONTHS,
            random_state=rng,
        )

    model, device = load_pretrained_model()
    output = {}
    for scenario, samples in (
        ("annual", annual_samples),
        ("monthly", monthly_samples),
    ):
        rows = []
        for station, values in samples.items():
            mu_hat, sigma_hat, xi_hat = original_notebook_nn_estimate(
                model,
                values,
                device,
            )
            rows.append(
                {
                    "station": station,
                    "mu_hat": mu_hat,
                    "sigma_hat": sigma_hat,
                    "xi_hat": xi_hat,
                }
            )
        output[scenario] = pd.DataFrame(rows).merge(
            truth,
            on="station",
            how="left",
            validate="one_to_one",
        )
    return output


def optimized_kernel(kernel: str, nu: float | None):
    if kernel == "RBF":
        spatial = RBF(
            length_scale=50.0,
            length_scale_bounds=(1.0, 500.0),
        )
    elif kernel == "Matern":
        spatial = Matern(
            length_scale=50.0,
            length_scale_bounds=(1.0, 500.0),
            nu=float(nu),
        )
    else:
        raise ValueError(kernel)
    return (
        C(1.0, constant_value_bounds=(1e-2, 1e2))
        * spatial
        + WhiteKernel(
            noise_level=1e-4,
            noise_level_bounds=(1e-8, 1e-1),
        )
    )


def fit_gridded_candidates(
    estimates: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[tuple[str, str, float, str], np.ndarray]]:
    rows = []
    predictions = {}
    specifications = (
        ("RBF", np.nan),
        ("Matern", 0.5),
        ("Matern", 1.5),
        ("Matern", 2.5),
    )

    for scenario, data in estimates.items():
        coordinates = metric_coordinates(data)
        for parameter in PARAMETERS:
            source = data[f"{parameter}_hat"].to_numpy(dtype=np.float64)
            truth = data[f"true_{parameter}"].to_numpy(dtype=np.float64)
            truth_scale = float(np.std(truth))
            if truth_scale <= 0:
                truth_scale = 1.0

            for kernel, nu in specifications:
                gp = GaussianProcessRegressor(
                    kernel=optimized_kernel(
                        kernel,
                        None if pd.isna(nu) else float(nu),
                    ),
                    n_restarts_optimizer=10,
                    normalize_y=True,
                    random_state=RANDOM_STATE,
                )
                gp.fit(coordinates, source)
                prediction = gp.predict(coordinates)
                error = prediction - truth
                key = (scenario, kernel, nu, parameter)
                predictions[key] = prediction
                rows.append(
                    {
                        "scenario": scenario,
                        "parameter": parameter,
                        "kernel": kernel,
                        "nu": nu,
                        "RMSE": float(np.sqrt(np.mean(error**2))),
                        "standardized_RMSE": float(
                            np.sqrt(np.mean((error / truth_scale) ** 2))
                        ),
                        "fitted_kernel": str(gp.kernel_),
                    }
                )
    return pd.DataFrame(rows), predictions


def gridded_descriptive_table(candidate_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        scenario_part = candidate_table.loc[
            candidate_table["scenario"].eq(scenario)
        ]
        for parameter in PARAMETERS:
            parameter_part = scenario_part.loc[
                scenario_part["parameter"].eq(parameter)
            ]
            for kernel, family in parameter_part.groupby("kernel"):
                best = family.loc[family["standardized_RMSE"].idxmin()]
                rows.append(
                    {
                        "experiment": "gridded_input",
                        "scenario": GRID_SCENARIO_LABELS[scenario],
                        "parameter": parameter,
                        "kernel": kernel,
                        "RMSE": float(best["standardized_RMSE"]),
                        "length_scale": np.nan,
                        "nu": best["nu"],
                        "metric": "standardized_RMSE",
                    }
                )

        for metric_name, source_metric in (
            ("overall", "RMSE"),
            ("overall_standardized", "standardized_RMSE"),
        ):
            for kernel, family in scenario_part.groupby("kernel"):
                candidate_rows = []
                for nu, candidate in family.groupby("nu", dropna=False):
                    value = float(
                        np.sqrt(np.mean(candidate[source_metric] ** 2))
                    )
                    candidate_rows.append((value, nu))
                value, nu = min(candidate_rows, key=lambda item: item[0])
                rows.append(
                    {
                        "experiment": "gridded_input",
                        "scenario": GRID_SCENARIO_LABELS[scenario],
                        "parameter": metric_name,
                        "kernel": kernel,
                        "RMSE": value,
                        "length_scale": np.nan,
                        "nu": nu,
                        "metric": (
                            "RMSE"
                            if metric_name == "overall"
                            else "standardized_RMSE"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def fit_spatial_out_of_fold_predictions(
    data: pd.DataFrame,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    """Predict every grid point only from the other nine spatial blocks."""
    labels = geographic_block_labels(data)
    predictions = {
        kernel: {
            parameter: np.full(len(data), np.nan, dtype=np.float64)
            for parameter in PARAMETERS
        }
        for kernel in ("RBF", "Matern")
    }

    for block in range(N_SPATIAL_BLOCKS):
        train_mask = labels != block
        test_mask = labels == block
        projected = add_twd97_km_columns(data)
        x_train, x_test, _ = center_train_test_coordinates(
            projected.loc[train_mask, ["x_km", "y_km"]].to_numpy(
                dtype=np.float64
            ),
            projected.loc[test_mask, ["x_km", "y_km"]].to_numpy(
                dtype=np.float64
            ),
        )

        for kernel, nu in (("RBF", None), ("Matern", 0.5)):
            for parameter in PARAMETERS:
                gp = GaussianProcessRegressor(
                    kernel=optimized_kernel(kernel, nu),
                    n_restarts_optimizer=3,
                    normalize_y=True,
                    random_state=RANDOM_STATE,
                )
                gp.fit(
                    x_train,
                    data.loc[
                        train_mask,
                        f"{parameter}_hat",
                    ].to_numpy(dtype=np.float64),
                )
                predictions[kernel][parameter][test_mask] = gp.predict(x_test)

    for kernel in predictions:
        for parameter in predictions[kernel]:
            if not np.all(np.isfinite(predictions[kernel][parameter])):
                raise ValueError(
                    f"Missing out-of-fold prediction: {kernel}/{parameter}"
                )
    return predictions, labels


def monthly_return_level(
    mu: np.ndarray,
    sigma: np.ndarray,
    xi: np.ndarray,
    period_years: int,
) -> np.ndarray:
    """Annual T-year level implied by a monthly-block GEV distribution."""
    block_probability = (1.0 - 1.0 / period_years) ** (1.0 / 12.0)
    a = -np.log(block_probability)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gev_level = mu + sigma * np.expm1(-xi * np.log(a)) / xi
        gumbel_level = mu - sigma * np.log(a)
    return np.where(np.abs(xi) < 1e-6, gumbel_level, gev_level)


def return_level_losses(
    data: pd.DataFrame,
    predictions: dict[str, dict[str, np.ndarray]],
    period_years: int,
) -> dict[str, np.ndarray]:
    true_level = monthly_return_level(
        data["true_mu"].to_numpy(dtype=np.float64),
        data["true_sigma"].to_numpy(dtype=np.float64),
        data["true_xi"].to_numpy(dtype=np.float64),
        period_years,
    )
    losses = {}
    for kernel in ("RBF", "Matern"):
        predicted_level = monthly_return_level(
            predictions[kernel]["mu"],
            predictions[kernel]["sigma"],
            predictions[kernel]["xi"],
            period_years,
        )
        losses[kernel] = (predicted_level - true_level) ** 2
    return losses


def evaluate_gridded_input() -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
]:
    """Evaluate annual, monthly, RL50, and RL100 with spatial OOF predictions."""
    truth = make_gridded_truth()
    estimates = simulate_gridded_nn_estimates(truth)
    candidates, _ = fit_gridded_candidates(estimates)
    descriptive = gridded_descriptive_table(candidates)
    block_tables = {}
    monthly_predictions = None

    for scenario, data in estimates.items():
        predictions, _ = fit_spatial_out_of_fold_predictions(data)
        losses = parameter_recovery_losses(data, predictions)
        experiment = f"gridded_{scenario}"
        block_tables[experiment] = block_loss_table(
            experiment=experiment,
            coordinates=data[["lon", "lat"]],
            cell_losses=losses,
        )
        if scenario == "monthly":
            monthly_predictions = predictions

    monthly_data = estimates["monthly"]
    for period in (50, 100):
        experiment = f"gridded_monthly_RL{period}"
        block_tables[experiment] = block_loss_table(
            experiment=experiment,
            coordinates=monthly_data[["lon", "lat"]],
            cell_losses=return_level_losses(
                monthly_data,
                monthly_predictions,
                period,
            ),
        )
    return descriptive, block_tables


def write_report(
    primary: pd.DataFrame,
    descriptive: pd.DataFrame,
) -> None:
    result_rows = []
    for _, row in primary.iterrows():
        result_rows.append(
            "| "
            + " | ".join(
                [
                    str(row["experiment"]),
                    str(row["loss_scale"]),
                    f"{row['RBF_RMSE']:.6f}",
                    f"{row['Matern_RMSE']:.6f}",
                    str(row["preferred_kernel_in_H1"]),
                    f"{row['one_sided_exact_p_value']:.6f}",
                    f"{row['holm_p_value_six_tests']:.6f}",
                    str(row["decision_Holm_alpha_0.05"]),
                ]
            )
            + " |"
        )

    text = f"""# Six directional RBF–Matérn kernel tests

## Outcomes

- Original stations: annual and monthly parameter recovery are tested
  separately, with Matérn as the pre-specified directional alternative.
- Gridded data: annual and monthly parameter recovery are tested separately,
  with RBF as the pre-specified directional alternative.
- Gridded monthly data: annual 50-year and 100-year return-level recovery are
  tested separately, with RBF as the pre-specified directional alternative.

## Test design

- These are six pre-specified, one-sided exact paired sign-flip tests.
- The unit of inference is one of ten geographic K-means blocks.
- Parameter-recovery loss jointly averages standardized squared errors for
  $\\mu$, $\\sigma$, and $\\xi$ without pooling annual and monthly scenarios.
- Return-level loss is squared prediction error in the temperature unit.
- Individual adjacent grid cells are not treated as independent replicates.
- With ten blocks, the minimum attainable one-sided p-value is
  $1/2^{{10}}=0.0009765625$.
- RBF and Matérn $\\nu=0.5$ are fixed before validation. GP hyperparameters
  are estimated by marginal likelihood from training responses only.
- Gridded predictions are spatial out-of-fold: each block is predicted from
  the other nine blocks.
- Holm adjustment controls family-wise error across all six hypotheses.

## Results

| Outcome | Loss scale | RBF RMSE | Matérn RMSE | Directional $H_1$ | Raw $p$ | Holm $p$ | Decision |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
{chr(10).join(result_rows)}
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    original_descriptive, original_blocks = evaluate_original_station_input()
    gridded_descriptive, gridded_blocks = evaluate_gridded_input()

    blocks = pd.concat(
        list(original_blocks.values()) + list(gridded_blocks.values()),
        ignore_index=True,
    )
    descriptive = pd.concat(
        [original_descriptive, gridded_descriptive],
        ignore_index=True,
    )
    test_specs = [
        ("original_station_annual", "Matern", "standardized parameters"),
        ("original_station_monthly", "Matern", "standardized parameters"),
        ("gridded_annual", "RBF", "standardized parameters"),
        ("gridded_monthly", "RBF", "standardized parameters"),
        ("gridded_monthly_RL50", "RBF", "temperature"),
        ("gridded_monthly_RL100", "RBF", "temperature"),
    ]
    rows = []
    all_block_tables = {**original_blocks, **gridded_blocks}
    for experiment, preferred_kernel, loss_scale in test_specs:
        other_kernel = "RBF" if preferred_kernel == "Matern" else "Matern"
        rows.append(
            summarize_block_test(
                experiment=experiment,
                preferred_kernel=preferred_kernel,
                null_hypothesis=(
                    f"{preferred_kernel} is not better than {other_kernel}"
                ),
                alternative_hypothesis=(
                    f"{preferred_kernel} is better than {other_kernel}"
                ),
                block_table=all_block_tables[experiment],
                loss_scale=loss_scale,
            )
        )
    primary = pd.DataFrame(rows)
    primary["holm_p_value_six_tests"] = holm_adjust(
        primary["one_sided_exact_p_value"]
    )
    primary["decision_Holm_alpha_0.05"] = np.where(
        primary["holm_p_value_six_tests"] < 0.05,
        "reject H0",
        "do not reject H0",
    )

    primary.to_csv(PRIMARY_PATH, index=False, encoding="utf-8-sig")
    blocks.to_csv(BLOCK_PATH, index=False, encoding="utf-8-sig")
    descriptive.to_csv(
        DESCRIPTIVE_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    write_report(primary, descriptive)

    columns = [
        "experiment",
        "preferred_kernel_in_H1",
        "loss_scale",
        "RBF_RMSE",
        "Matern_RMSE",
        "blocks_favouring_H1",
        "n_spatial_blocks",
        "one_sided_exact_p_value",
        "holm_p_value_six_tests",
        "decision_Holm_alpha_0.05",
    ]
    print(primary[columns].to_string(index=False))
    print("\nSaved:", PRIMARY_PATH)
    print("Saved:", BLOCK_PATH)
    print("Saved:", DESCRIPTIVE_PATH)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
