"""Scenario 1: simulation calibrated to the selected real-data GP models.

This module deliberately does not replace ``simulate_spatial_gev.py``.  The
older module supports the original station/annual-monthly experiment, whereas
this module creates an independent downstream validation data set on the real
Taiwan TCCIP grid.

For each GEV response, the selected universal-GP specification is fitted to
the real NN-derived parameter surface.  A new latent spatial surface is then
drawn from the fitted spatial covariance and combined with the fitted linear
mean.  Forty-five years of monthly maxima (540 monthly blocks per GRID) are
generated conditionally on those known surfaces and passed through the frozen
neural network.  The saved
model-ready table can subsequently be supplied to nested buffered spatial CV
to test predictor/kernel recovery and return-level reconstruction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import genextreme

from elevation_gp_analysis import TARGETS, sample_indices
from gev_nn import estimate_one, load_baseline_model
from land_cover_gp_analysis import fit_gp_with_covariates
from project_paths import (
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    SIMULATED_DATA_DIR,
    TABLE_DIR,
)


DEFAULT_GRID_PATH = PROCESSED_DATA_DIR / "model_ready_grid_parameters.csv"
DEFAULT_SELECTED_MODELS_PATH = TABLE_DIR / "spatial_ffs_selected_models.csv"
DEFAULT_MODEL_PATH = MODEL_DIR / "best_baseline_model.pth"
DEFAULT_OUTPUT_DIR = SIMULATED_DATA_DIR / "calibrated_final_model"

TARGET_ORDER = ("mu", "log_sigma", "xi")


@dataclass(frozen=True)
class CalibratedSimulationConfig:
    """Configuration for the fitted-model data-generating experiment."""

    n_years: int = 45
    months_per_year: int = 12
    start_year: int = 1980
    n_replicates: int = 1
    calibration_max_train: int = 800
    n_restarts_optimizer: int = 0
    # Match the existing quantile-ratio estimator's supported shape range.
    # This contains the observed real-data range (-0.381, 0.400) and avoids
    # creating the large artificial point masses caused by [-0.2, 0.2].
    xi_lower: float = -0.50
    xi_upper: float = 0.50
    seed: int = 20260820

    @property
    def n_months(self) -> int:
        """Total number of monthly maxima generated at each GRID cell."""
        return self.n_years * self.months_per_year


@dataclass
class CalibratedSimulationSetup:
    """Objects calibrated once and reused by every simulation replicate."""

    grid: pd.DataFrame
    specifications: dict[str, dict]
    fitted_models: dict[str, dict]
    generator: dict[str, dict[str, np.ndarray]]
    calibration_indices: np.ndarray
    nn_model: object
    nn_device: str


def annual_return_level_from_monthly_gev(
    mu: np.ndarray,
    log_sigma: np.ndarray,
    xi: np.ndarray,
    return_period_years: int,
    months_per_year: int = 12,
) -> np.ndarray:
    """Convert a monthly-maxima GEV to a T-year annual return level.

    If monthly maxima are conditionally IID with CDF ``F_m``, the annual
    maximum has CDF ``F_m(z) ** months_per_year``.  Therefore the T-year
    annual return level satisfies
    ``F_m(z) = (1 - 1/T) ** (1/months_per_year)``.
    """
    if return_period_years <= 1:
        raise ValueError("return_period_years must exceed 1.")
    if months_per_year <= 0:
        raise ValueError("months_per_year must be positive.")
    mu = np.asarray(mu, dtype=float)
    sigma = np.exp(np.asarray(log_sigma, dtype=float))
    xi = np.asarray(xi, dtype=float)
    annual_probability = 1.0 - 1.0 / float(return_period_years)
    monthly_probability = annual_probability ** (1.0 / months_per_year)
    transformed_probability = -np.log(monthly_probability)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        nonzero = (
            mu
            + sigma
            * np.expm1(-xi * np.log(transformed_probability))
            / xi
        )
        gumbel = mu - sigma * np.log(transformed_probability)
    return np.where(np.abs(xi) < 1e-6, gumbel, nonzero)


def _split_predictors(value: object) -> list[str]:
    """Decode the plus-separated predictor list written by Spatial FFS."""
    if pd.isna(value) or not str(value).strip():
        return []
    return [item.strip() for item in str(value).split("+") if item.strip()]


def load_scenario_inputs(
    grid_path: str | Path = DEFAULT_GRID_PATH,
    selected_models_path: str | Path = DEFAULT_SELECTED_MODELS_PATH,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Load the 1,385-cell main-island table and final selected structures."""
    grid = pd.read_csv(grid_path)
    selected = pd.read_csv(selected_models_path)
    if grid["station"].duplicated().any():
        raise ValueError("model-ready GRID contains duplicated station identifiers.")
    if set(selected["target"]) != set(TARGET_ORDER):
        raise ValueError("selected-model table must contain mu, log_sigma, and xi.")

    specifications: dict[str, dict] = {}
    required = {"station", "lon", "lat", "x_km", "y_km"}
    for target in TARGET_ORDER:
        row = selected.loc[selected["target"].eq(target)].iloc[0]
        predictors = _split_predictors(row["predictors"])
        required.update(predictors)
        required.add(TARGETS[target])
        nu = float(row["nu"]) if pd.notna(row["nu"]) else np.nan
        specifications[target] = {
            "predictors": predictors,
            "kernel": str(row["kernel"]),
            "nu": nu,
        }

    missing = sorted(required.difference(grid.columns))
    if missing:
        raise ValueError(f"model-ready GRID is missing required columns: {missing}")
    if grid[list(required)].isna().any().any():
        bad = grid[list(required)].columns[grid[list(required)].isna().any()].tolist()
        raise ValueError(f"scenario inputs contain missing values: {bad}")
    return grid.reset_index(drop=True), specifications


def calibrate_selected_models(
    grid: pd.DataFrame,
    specifications: dict[str, dict],
    config: CalibratedSimulationConfig,
) -> tuple[dict[str, dict], np.ndarray]:
    """Fit each selected GP to one shared, reproducible capped GRID sample."""
    indices = sample_indices(
        grid.index.to_numpy(),
        config.calibration_max_train,
        config.seed,
    )
    train = grid.loc[indices]
    fitted: dict[str, dict] = {}
    for order, target in enumerate(TARGET_ORDER):
        specification = specifications[target]
        predictors = specification["predictors"]
        covariates = (
            train[predictors].to_numpy(float)
            if predictors
            else np.empty((len(train), 0), dtype=float)
        )
        fitted[target] = fit_gp_with_covariates(
            train[["x_km", "y_km"]].to_numpy(float),
            covariates,
            train[TARGETS[target]].to_numpy(float),
            predictor_names=predictors,
            kernel_name=specification["kernel"],
            nu=specification["nu"],
            n_restarts=config.n_restarts_optimizer,
            seed=config.seed + order * 10_000,
        )
    return fitted, np.asarray(indices, dtype=int)


def _full_grid_mean(grid: pd.DataFrame, model: dict) -> np.ndarray:
    """Evaluate the fitted standardized linear mean on every GRID cell."""
    predictors = model["predictor_names"]
    if not predictors:
        design = np.ones((len(grid), 1), dtype=float)
    else:
        values = grid[predictors].to_numpy(float)
        standardized = (
            values - np.asarray(model["covariate_center"], dtype=float)
        ) / np.asarray(model["covariate_scale"], dtype=float)
        design = np.column_stack([np.ones(len(grid)), standardized])
    return design @ np.asarray(model["beta"], dtype=float)


def _latent_cholesky(grid: pd.DataFrame, model: dict) -> np.ndarray:
    """Factor the fitted spatial covariance, excluding the white-noise nugget."""
    coordinates = (
        grid[["x_km", "y_km"]].to_numpy(float)
        - np.asarray(model["xy_origin_km"], dtype=float)
    )
    fitted_kernel = model["kernel"]
    spatial_kernel = getattr(fitted_kernel, "k1", fitted_kernel)
    covariance = np.asarray(spatial_kernel(coordinates), dtype=float)
    diagonal = np.diag_indices_from(covariance)
    jitter = max(float(np.mean(np.diag(covariance))) * 1e-10, 1e-10)
    for _ in range(8):
        try:
            covariance[diagonal] += jitter
            return np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError("unable to factor calibrated spatial covariance")


def prepare_generator(
    grid: pd.DataFrame,
    fitted_models: dict[str, dict],
) -> dict[str, dict[str, np.ndarray]]:
    """Precompute means and covariance factors shared by all replicates."""
    return {
        target: {
            "mean": _full_grid_mean(grid, fitted_models[target]),
            "cholesky": _latent_cholesky(grid, fitted_models[target]),
        }
        for target in TARGET_ORDER
    }


def generate_true_parameter_surfaces(
    grid: pd.DataFrame,
    generator: dict[str, dict[str, np.ndarray]],
    config: CalibratedSimulationConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int]:
    """Draw one known spatial GEV surface from the calibrated final models."""
    truth = grid.copy()
    for target in TARGET_ORDER:
        component = generator[target]
        latent = component["cholesky"] @ rng.standard_normal(len(grid))
        truth[f"{target}_true"] = component["mean"] + latent

    raw_xi = truth["xi_true"].to_numpy(float)
    clipped = (raw_xi < config.xi_lower) | (raw_xi > config.xi_upper)
    truth["xi_true"] = np.clip(raw_xi, config.xi_lower, config.xi_upper)
    truth["sigma_true"] = np.exp(truth["log_sigma_true"])
    for period in (50, 100):
        truth[f"RL{period}_true"] = annual_return_level_from_monthly_gev(
            truth["mu_true"],
            truth["log_sigma_true"],
            truth["xi_true"],
            period,
            config.months_per_year,
        )
    return truth, int(clipped.sum())


def simulate_monthly_maxima(
    truth: pd.DataFrame,
    config: CalibratedSimulationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate monthly block maxima independently conditional on truth."""
    return genextreme.rvs(
        c=-truth["xi_true"].to_numpy(float)[:, None],
        loc=truth["mu_true"].to_numpy(float)[:, None],
        scale=truth["sigma_true"].to_numpy(float)[:, None],
        size=(len(truth), config.n_months),
        random_state=rng,
    )


def estimate_with_frozen_nn(
    monthly_maxima: np.ndarray,
    model,
    device: str,
    months_per_year: int = 12,
) -> pd.DataFrame:
    """Convert each 540-month sample to NN-derived monthly-GEV estimates."""
    rows = []
    for values in monthly_maxima:
        mu, sigma, shape_c = estimate_one(model, values, device)
        rows.append(
            {
                "mu_hat": mu,
                "sigma_hat": sigma,
                "log_sigma_hat": np.log(sigma),
                "shape_c_hat": shape_c,
                "xi_hat": -shape_c,
            }
        )
    result = pd.DataFrame(rows)
    for period in (50, 100):
        result[f"RL{period}_hat"] = annual_return_level_from_monthly_gev(
            result["mu_hat"],
            result["log_sigma_hat"],
            result["xi_hat"],
            period,
            months_per_year,
        )
    return result


def nn_recovery_metrics(model_ready: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the frozen NN against the known simulated parameter/RL truth."""
    rows = []
    for outcome in (*TARGET_ORDER, "RL50", "RL100"):
        truth = model_ready[f"{outcome}_true"].to_numpy(float)
        estimate = model_ready[f"{outcome}_hat"].to_numpy(float)
        error = estimate - truth
        rows.append(
            {
                "outcome": outcome,
                "RMSE": float(np.sqrt(np.mean(error**2))),
                "MAE": float(np.mean(np.abs(error))),
                "Bias": float(np.mean(error)),
                "correlation": float(np.corrcoef(truth, estimate)[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def calibration_table(
    fitted_models: dict[str, dict],
    specifications: dict[str, dict],
    n_calibration: int,
) -> pd.DataFrame:
    """Create an auditable summary of the real-data calibration fit."""
    rows = []
    for target in TARGET_ORDER:
        model = fitted_models[target]
        specification = specifications[target]
        rows.append(
            {
                "target": target,
                "predictors": "+".join(specification["predictors"]),
                "kernel": specification["kernel"],
                "nu": specification["nu"],
                "beta": json.dumps(np.asarray(model["beta"]).tolist()),
                "covariate_center": json.dumps(
                    np.asarray(model["covariate_center"]).tolist()
                ),
                "covariate_scale": json.dumps(
                    np.asarray(model["covariate_scale"]).tolist()
                ),
                "optimized_kernel": str(model["kernel"]),
                "n_calibration": n_calibration,
                "optimizer_success": model["optimizer_success"],
            }
        )
    return pd.DataFrame(rows)


def prepare_calibrated_simulation(
    config: CalibratedSimulationConfig,
    grid_path: str | Path = DEFAULT_GRID_PATH,
    selected_models_path: str | Path = DEFAULT_SELECTED_MODELS_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    output_directory: str | Path = DEFAULT_OUTPUT_DIR,
) -> CalibratedSimulationSetup:
    """Calibrate the data-generating models and load the frozen NN once."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    grid, specifications = load_scenario_inputs(grid_path, selected_models_path)
    print(f"載入 {len(grid):,} 個臺灣本島 GRID。", flush=True)
    fitted_models, calibration_indices = calibrate_selected_models(
        grid, specifications, config
    )
    generator = prepare_generator(grid, fitted_models)
    calibration = calibration_table(
        fitted_models, specifications, len(calibration_indices)
    )
    calibration.to_csv(
        output_directory / "calibrated_gp_models.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([asdict(config)]).to_csv(
        output_directory / "simulation_config.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {
            "row_index": calibration_indices,
            "station": grid.loc[calibration_indices, "station"],
        }
    ).to_csv(
        output_directory / "calibration_grid_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nn_model, nn_device = load_baseline_model(model_path=model_path)
    return CalibratedSimulationSetup(
        grid=grid,
        specifications=specifications,
        fitted_models=fitted_models,
        generator=generator,
        calibration_indices=calibration_indices,
        nn_model=nn_model,
        nn_device=nn_device,
    )


def generate_calibrated_replicate(
    setup: CalibratedSimulationSetup,
    config: CalibratedSimulationConfig,
    replicate: int,
    output_directory: str | Path = DEFAULT_OUTPUT_DIR,
    save_monthly_maxima: bool = True,
) -> dict[str, Path]:
    """Generate, estimate, and persist one independently seeded replicate."""
    if replicate < 0:
        raise ValueError("replicate must be non-negative.")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed + replicate * 10_000)
    truth, clipped_xi = generate_true_parameter_surfaces(
        setup.grid, setup.generator, config, rng
    )
    monthly = simulate_monthly_maxima(truth, config, rng)
    nn = estimate_with_frozen_nn(
        monthly,
        setup.nn_model,
        setup.nn_device,
        config.months_per_year,
    )
    model_ready = truth.copy()
    for column in nn.columns:
        model_ready[column] = nn[column].to_numpy()
    model_ready["replicate"] = replicate
    model_ready["block_scale"] = "monthly"
    model_ready["n_years"] = config.n_years
    model_ready["months_per_year"] = config.months_per_year
    model_ready["n_months"] = config.n_months
    model_ready["start_year"] = config.start_year
    model_ready["xi_clipped"] = (
        (model_ready["xi_true"] <= config.xi_lower)
        | (model_ready["xi_true"] >= config.xi_upper)
    )

    prefix = f"replicate_{replicate:03d}"
    model_ready_path = output_directory / f"{prefix}_model_ready.csv"
    monthly_path = output_directory / f"{prefix}_monthly_maxima.csv"
    nn_metric_path = output_directory / f"{prefix}_nn_recovery_metrics.csv"
    model_ready.to_csv(
        model_ready_path, index=False, encoding="utf-8-sig"
    )
    if save_monthly_maxima:
        month_columns = [
            f"monthly_max_{year}_{month:02d}"
            for year in range(config.start_year, config.start_year + config.n_years)
            for month in range(1, config.months_per_year + 1)
        ]
        monthly_table = pd.DataFrame(monthly, columns=month_columns)
        monthly_table.insert(0, "station", setup.grid["station"].to_numpy())
        monthly_table.to_csv(
            monthly_path, index=False, encoding="utf-8-sig"
        )
    nn_recovery_metrics(model_ready).to_csv(
        nn_metric_path, index=False, encoding="utf-8-sig"
    )
    print(
        f"replicate {replicate + 1}/{config.n_replicates} 資料生成完成；"
        f"xi 邊界裁切 {clipped_xi}/{len(setup.grid)} 個 GRID。",
        flush=True,
    )
    paths = {
        "model_ready": model_ready_path,
        "nn_metrics": nn_metric_path,
    }
    if save_monthly_maxima:
        paths["monthly_maxima"] = monthly_path
    return paths


def run_calibrated_simulation(
    config: CalibratedSimulationConfig,
    grid_path: str | Path = DEFAULT_GRID_PATH,
    selected_models_path: str | Path = DEFAULT_SELECTED_MODELS_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    output_directory: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """Calibrate once, generate independent replicates, and save all evidence."""
    output_directory = Path(output_directory)
    setup = prepare_calibrated_simulation(
        config=config,
        grid_path=grid_path,
        selected_models_path=selected_models_path,
        model_path=model_path,
        output_directory=output_directory,
    )
    written = [output_directory / "calibrated_gp_models.csv"]
    for replicate in range(config.n_replicates):
        paths = generate_calibrated_replicate(
            setup=setup,
            config=config,
            replicate=replicate,
            output_directory=output_directory,
        )
        written.extend(paths.values())
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate scenario 1 from the final real-data GP specifications."
    )
    parser.add_argument("--n-replicates", type=int, default=1)
    parser.add_argument("--n-years", type=int, default=45)
    parser.add_argument("--months-per-year", type=int, default=12)
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--calibration-max-train", type=int, default=800)
    parser.add_argument("--n-restarts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--xi-lower", type=float, default=-0.50)
    parser.add_argument("--xi-upper", type=float, default=0.50)
    parser.add_argument("--grid-path", type=Path, default=DEFAULT_GRID_PATH)
    parser.add_argument(
        "--selected-models-path",
        type=Path,
        default=DEFAULT_SELECTED_MODELS_PATH,
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CalibratedSimulationConfig(
        n_years=args.n_years,
        months_per_year=args.months_per_year,
        start_year=args.start_year,
        n_replicates=args.n_replicates,
        calibration_max_train=args.calibration_max_train,
        n_restarts_optimizer=args.n_restarts,
        xi_lower=args.xi_lower,
        xi_upper=args.xi_upper,
        seed=args.seed,
    )
    paths = run_calibrated_simulation(
        config=config,
        grid_path=args.grid_path,
        selected_models_path=args.selected_models_path,
        model_path=args.model_path,
        output_directory=args.output_directory,
    )
    print("輸出檔案：", flush=True)
    for path in paths:
        print(path, flush=True)


if __name__ == "__main__":
    main()
