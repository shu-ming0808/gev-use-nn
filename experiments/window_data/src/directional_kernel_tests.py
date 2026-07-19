"""Two pre-specified directional GP-kernel tests for the simulation study.

The program answers exactly two confirmatory questions:

1. Original station-input experiment:
   Is Matérn better than RBF when the 25 original simulated station estimates
   are interpolated to the known dense truth grid?
2. Gridded-input experiment:
   Is RBF better than Matérn when the NN is first applied at every point of
   the 0.5-degree simulated grid and the resulting fields are GP-smoothed?

The primary loss is the mean squared standardized recovery error jointly over
mu, sigma, and xi and over the annual-45 and monthly-540 scenarios.  Taiwan is
partitioned into ten geographic K-means blocks.  One paired loss is calculated
per block, so adjacent grid cells are not treated as independent replicates.

Only the two one-sided alternatives above are tested.  The old generic
two-sided difference test and its AIC section are intentionally not included.

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
from kriging_kernel_gridsearch import (  # noqa: E402
    load_inputs,
    predict_grid,
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


def standardized_coordinates(frame: pd.DataFrame) -> np.ndarray:
    coordinates = frame[["lon", "lat"]].to_numpy(dtype=np.float64)
    mean = coordinates.mean(axis=0)
    scale = coordinates.std(axis=0)
    scale[scale == 0] = 1.0
    return (coordinates - mean) / scale


def geographic_block_labels(frame: pd.DataFrame) -> np.ndarray:
    return KMeans(
        n_clusters=N_SPATIAL_BLOCKS,
        random_state=RANDOM_STATE,
        n_init=50,
    ).fit_predict(standardized_coordinates(frame))


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


def summarize_block_test(
    *,
    experiment: str,
    preferred_kernel: str,
    null_hypothesis: str,
    alternative_hypothesis: str,
    block_table: pd.DataFrame,
) -> dict:
    pivot = block_table.pivot(
        index="spatial_block",
        columns="kernel",
        values="joint_standardized_MSE",
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
        "RBF_joint_standardized_RMSE": rbf_rmse,
        "Matern_joint_standardized_RMSE": matern_rmse,
        "preferred_kernel_RMSE_reduction_pct": (
            100.0 * (other_rmse - preferred_rmse) / other_rmse
        ),
        "mean_directional_block_MSE_contrast": observed,
        "blocks_favouring_H1": int((contrast > 0).sum()),
        "one_sided_exact_p_value": p_value,
        "minimum_p_value_resolution": 1.0 / (2 ** len(pivot)),
        "decision_alpha_0.05": (
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
                    "joint_standardized_MSE": float(block_losses.mean()),
                    "joint_standardized_RMSE": float(
                        np.sqrt(block_losses.mean())
                    ),
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


def evaluate_original_station_input() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the first screenshot and build its combined block losses."""
    search = pd.read_csv(ORIGINAL_SEARCH_PATH)
    descriptive = original_descriptive_table(search)
    scenario_losses: dict[str, dict[str, np.ndarray]] = {}
    reference_grid = None

    for scenario in SCENARIOS:
        truth, station = load_inputs(scenario, sim_dir=str(SIM_DIR))
        if reference_grid is None:
            reference_grid = truth[["lon", "lat"]].copy()
        elif not np.allclose(
            reference_grid.to_numpy(),
            truth[["lon", "lat"]].to_numpy(),
        ):
            raise ValueError("Annual and monthly truth grids are not aligned.")

        scenario_losses[scenario] = {}
        for kernel in ("RBF", "Matern"):
            best = search.loc[
                search["scenario"].eq(scenario)
                & search["kernel"].eq(kernel)
                & search["param"].eq("overall_standardized")
            ].sort_values("rmse").iloc[0]

            squared_errors = []
            for parameter in PARAMETERS:
                true_col = f"true_{parameter}"
                source_col = f"{parameter}_hat"
                prediction = predict_grid(
                    station=station,
                    true_grid=truth,
                    source_col=source_col,
                    kernel_type=kernel,
                    length_scale=float(best["length_scale"]),
                    nu=(
                        None
                        if pd.isna(best["nu"])
                        else float(best["nu"])
                    ),
                )
                true_values = truth[true_col].to_numpy(dtype=np.float64)
                scale = float(np.std(true_values))
                if scale <= 0:
                    scale = 1.0
                squared_errors.append(
                    ((prediction - true_values) / scale) ** 2
                )
            scenario_losses[scenario][kernel] = np.mean(
                squared_errors,
                axis=0,
            )

    combined = {
        kernel: np.mean(
            [scenario_losses[scenario][kernel] for scenario in SCENARIOS],
            axis=0,
        )
        for kernel in ("RBF", "Matern")
    }
    blocks = block_loss_table(
        experiment="original_station_input",
        coordinates=reference_grid,
        cell_losses=combined,
    )
    return descriptive, blocks


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
    truth = pd.DataFrame(
        [(lon, lat) for lat in lats for lon in lons],
        columns=["lon", "lat"],
    )
    truth["station"] = (
        "SIM"
        + truth["lon"].map(lambda value: f"{value:.1f}")
        + "_"
        + truth["lat"].map(lambda value: f"{value:.1f}")
    )

    lon_s = (truth["lon"] - truth["lon"].mean()) / truth["lon"].std()
    lat_s = (truth["lat"] - truth["lat"].mean()) / truth["lat"].std()
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
            length_scale=1.0,
            length_scale_bounds=(1e-2, 10.0),
        )
    elif kernel == "Matern":
        spatial = Matern(
            length_scale=1.0,
            length_scale_bounds=(1e-2, 10.0),
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
        coordinates = standardized_coordinates(data)
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


def best_joint_nu(
    candidate_table: pd.DataFrame,
    scenario: str,
    kernel: str,
) -> float:
    family = candidate_table.loc[
        candidate_table["scenario"].eq(scenario)
        & candidate_table["kernel"].eq(kernel)
    ]
    candidates = []
    for nu, part in family.groupby("nu", dropna=False):
        joint = float(np.sqrt(np.mean(part["standardized_RMSE"] ** 2)))
        candidates.append((joint, nu))
    return min(candidates, key=lambda item: item[0])[1]


def prediction_lookup(
    predictions: dict[tuple[str, str, float, str], np.ndarray],
    scenario: str,
    kernel: str,
    nu: float,
    parameter: str,
) -> np.ndarray:
    if pd.isna(nu):
        for key, prediction in predictions.items():
            key_scenario, key_kernel, key_nu, key_parameter = key
            if (
                key_scenario == scenario
                and key_kernel == kernel
                and pd.isna(key_nu)
                and key_parameter == parameter
            ):
                return prediction
        raise KeyError((scenario, kernel, nu, parameter))
    return predictions[(scenario, kernel, nu, parameter)]


def evaluate_gridded_input() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the second screenshot and build its combined block losses."""
    truth = make_gridded_truth()
    estimates = simulate_gridded_nn_estimates(truth)
    candidates, predictions = fit_gridded_candidates(estimates)
    descriptive = gridded_descriptive_table(candidates)
    scenario_losses: dict[str, dict[str, np.ndarray]] = {}

    for scenario, data in estimates.items():
        scenario_losses[scenario] = {}
        for kernel in ("RBF", "Matern"):
            nu = best_joint_nu(candidates, scenario, kernel)
            squared_errors = []
            for parameter in PARAMETERS:
                prediction = prediction_lookup(
                    predictions,
                    scenario,
                    kernel,
                    nu,
                    parameter,
                )
                true_values = data[f"true_{parameter}"].to_numpy(
                    dtype=np.float64
                )
                scale = float(np.std(true_values))
                if scale <= 0:
                    scale = 1.0
                squared_errors.append(
                    ((prediction - true_values) / scale) ** 2
                )
            scenario_losses[scenario][kernel] = np.mean(
                squared_errors,
                axis=0,
            )

    combined = {
        kernel: np.mean(
            [scenario_losses[scenario][kernel] for scenario in SCENARIOS],
            axis=0,
        )
        for kernel in ("RBF", "Matern")
    }
    blocks = block_loss_table(
        experiment="gridded_input",
        coordinates=truth[["lon", "lat"]],
        cell_losses=combined,
    )
    return descriptive, blocks


def write_report(
    primary: pd.DataFrame,
    descriptive: pd.DataFrame,
) -> None:
    original = primary.loc[
        primary["experiment"].eq("original_station_input")
    ].iloc[0]
    gridded = primary.loc[
        primary["experiment"].eq("gridded_input")
    ].iloc[0]

    def result_line(row: pd.Series) -> str:
        return (
            f"{row['preferred_kernel_in_H1']} RMSE = "
            f"{row[f'{row.preferred_kernel_in_H1}_joint_standardized_RMSE']:.6f}; "
            f"one-sided exact p = {row['one_sided_exact_p_value']:.6f}; "
            f"{row['decision_alpha_0.05']}."
        )

    table_rows = []
    for _, row in descriptive.iterrows():
        table_rows.append(
            "| "
            + " | ".join(
                [
                    str(row["experiment"]),
                    str(row["scenario"]),
                    str(row["parameter"]),
                    str(row["kernel"]),
                    f"{row['RMSE']:.6f}",
                    str(row["metric"]),
                ]
            )
            + " |"
        )

    text = f"""# Directional RBF–Matérn kernel tests

## Confirmatory hypotheses

### Test 1: original station input

For geographic block $b$, define

$$
C_b=L_{{\\mathrm{{RBF}},b}}-L_{{\\mathrm{{Mat\\acute{{e}}rn}},b}}.
$$

$$
H_0:E(C_b)\\leq 0,
\\qquad
H_1:E(C_b)>0.
$$

The one-sided alternative means that Matérn has smaller joint standardized
recovery loss than RBF.

Result: {result_line(original)}

### Test 2: gridded input

For geographic block $b$, define

$$
C_b=L_{{\\mathrm{{Mat\\acute{{e}}rn}},b}}-L_{{\\mathrm{{RBF}},b}}.
$$

$$
H_0:E(C_b)\\leq 0,
\\qquad
H_1:E(C_b)>0.
$$

The one-sided alternative means that RBF has smaller joint standardized
recovery loss than Matérn.

Result: {result_line(gridded)}

## Test design

- These are two pre-specified, one-sided exact paired sign-flip tests.
- The unit of inference is one of ten geographic K-means blocks.
- Each block loss jointly averages standardized squared recovery errors for
  $\\mu$, $\\sigma$, and $\\xi$ in both annual-45 and monthly-540 simulations.
- Individual adjacent grid cells are not treated as independent replicates.
- With ten blocks, the minimum attainable one-sided p-value is
  $1/2^{{10}}=0.0009765625$.
- The original station-input experiment uses the saved exhaustive fixed-kernel
  grid search in `spatial_kernel_gridsearch_rmse.csv`.
- The gridded-input experiment exactly reproduces the notebook table, including
  the original pretrained-network inverse transform and RNG ordering.
- The tests assess the two stated directional claims. They are not two-sided
  generic difference tests, and AIC is not used as a p-value.

## Descriptive RMSE values reproduced from the two tables

| Experiment | Scenario | Parameter | Kernel | RMSE | Metric |
| --- | --- | --- | --- | ---: | --- |
{chr(10).join(table_rows)}
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    original_descriptive, original_blocks = evaluate_original_station_input()
    gridded_descriptive, gridded_blocks = evaluate_gridded_input()

    blocks = pd.concat(
        [original_blocks, gridded_blocks],
        ignore_index=True,
    )
    descriptive = pd.concat(
        [original_descriptive, gridded_descriptive],
        ignore_index=True,
    )
    primary = pd.DataFrame(
        [
            summarize_block_test(
                experiment="original_station_input",
                preferred_kernel="Matern",
                null_hypothesis=(
                    "Matérn is not better than RBF "
                    "(E[L_RBF - L_Matérn] <= 0)"
                ),
                alternative_hypothesis=(
                    "Matérn is better than RBF "
                    "(E[L_RBF - L_Matérn] > 0)"
                ),
                block_table=original_blocks,
            ),
            summarize_block_test(
                experiment="gridded_input",
                preferred_kernel="RBF",
                null_hypothesis=(
                    "RBF is not better than Matérn "
                    "(E[L_Matérn - L_RBF] <= 0)"
                ),
                alternative_hypothesis=(
                    "RBF is better than Matérn "
                    "(E[L_Matérn - L_RBF] > 0)"
                ),
                block_table=gridded_blocks,
            ),
        ]
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
        "RBF_joint_standardized_RMSE",
        "Matern_joint_standardized_RMSE",
        "blocks_favouring_H1",
        "n_spatial_blocks",
        "one_sided_exact_p_value",
        "decision_alpha_0.05",
    ]
    print(primary[columns].to_string(index=False))
    print("\nSaved:", PRIMARY_PATH)
    print("Saved:", BLOCK_PATH)
    print("Saved:", DESCRIPTIVE_PATH)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
