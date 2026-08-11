"""Independent downstream simulation for the frozen GEV neural network.

The simulation generates spatially varying, *known* GEV parameters, draws a
new annual-maximum series at every synthetic grid cell, applies the frozen
pretrained neural network, and evaluates GP model selection with nested
buffered spatial cross-validation.  It therefore evaluates the complete
NN -> spatial GP -> return-level pipeline against known truth without reusing
the data that trained the neural network.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs, parallel_backend
from scipy.stats import genextreme
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    Matern,
    RBF,
    WhiteKernel,
)

from gev_nn import estimate_one, load_baseline_model
from project_paths import FIGURE_DIR, TABLE_DIR


TARGETS = ("mu", "log_sigma", "xi")
SCENARIOS = (
    "stationary_rbf",
    "stationary_matern",
    "elevation_anisotropic_matern",
)


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration shared by pilot and final Monte Carlo experiments."""

    grid_side: int = 20
    spacing_km: float = 10.0
    n_years: int = 45
    outer_folds: int = 5
    inner_folds: int = 3
    buffer_km: float = 20.0
    max_train: int = 200
    min_train: int = 20
    n_replicates: int = 20
    seed: int = 20260801
    n_restarts_optimizer: int = 0
    n_jobs: int = -2


def gev_return_level(mu, log_sigma, xi, return_period: int) -> np.ndarray:
    """GEV plug-in return level with a stable Gumbel limit."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.exp(np.asarray(log_sigma, dtype=float))
    xi = np.asarray(xi, dtype=float)
    a = -np.log(1.0 - 1.0 / float(return_period))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        nonzero = mu + sigma * np.expm1(-xi * np.log(a)) / xi
        gumbel = mu - sigma * np.log(a)
    return np.where(np.abs(xi) < 1e-6, gumbel, nonzero)


def make_grid(config: SimulationConfig) -> pd.DataFrame:
    """Create a rectangular metric grid and a reproducible synthetic DEM."""
    axis = np.arange(config.grid_side, dtype=float) * config.spacing_km
    x_km, y_km = np.meshgrid(axis, axis)
    x = x_km.ravel()
    y = y_km.ravel()
    width = max(axis.max(), config.spacing_km)

    ridge = 2600.0 * np.exp(-0.5 * ((x - 0.55 * width) / (0.13 * width)) ** 2)
    north_peak = 900.0 * np.exp(
        -0.5
        * (
            ((x - 0.42 * width) / (0.20 * width)) ** 2
            + ((y - 0.76 * width) / (0.18 * width)) ** 2
        )
    )
    elevation = np.maximum(ridge + north_peak - 150.0, 0.0)
    return pd.DataFrame(
        {
            "cell_id": np.arange(x.size),
            "x_km": x,
            "y_km": y,
            "elevation_m": elevation,
        }
    )


def _spatial_kernel(scenario: str, amplitude: float, length_km: float):
    if scenario == "stationary_rbf":
        spatial = RBF(length_scale=length_km)
    elif scenario == "stationary_matern":
        spatial = Matern(length_scale=length_km, nu=1.5)
    elif scenario == "elevation_anisotropic_matern":
        spatial = Matern(length_scale=[1.8 * length_km, 0.55 * length_km], nu=1.5)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return C(amplitude**2, constant_value_bounds="fixed") * spatial


def _draw_latent(
    xy: np.ndarray,
    scenario: str,
    amplitude: float,
    length_km: float,
    rng: np.random.Generator,
) -> np.ndarray:
    covariance = _spatial_kernel(scenario, amplitude, length_km)(xy)
    covariance[np.diag_indices_from(covariance)] += 1e-8
    return rng.multivariate_normal(np.zeros(len(xy)), covariance)


def generate_true_parameters(
    grid: pd.DataFrame,
    scenario: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate known GEV surfaces under one prespecified spatial scenario."""
    xy = grid[["x_km", "y_km"]].to_numpy(float)
    z_elevation = (
        grid["elevation_m"].to_numpy(float) - grid["elevation_m"].mean()
    ) / max(grid["elevation_m"].std(), 1.0)
    use_elevation = scenario == "elevation_anisotropic_matern"

    mu = 30.0 + _draw_latent(xy, scenario, 1.15, 48.0, rng)
    log_sigma = np.log(1.5) + _draw_latent(xy, scenario, 0.13, 58.0, rng)
    eta_xi = _draw_latent(xy, scenario, 0.030, 65.0, rng)
    if use_elevation:
        mu = mu - 2.8 * z_elevation
        log_sigma = log_sigma + 0.08 * z_elevation
    xi = np.clip(0.05 + eta_xi, -0.18, 0.18)

    result = grid.copy()
    result["mu_true"] = mu
    result["log_sigma_true"] = log_sigma
    result["xi_true"] = xi
    result["RL50_true"] = gev_return_level(mu, log_sigma, xi, 50)
    result["RL100_true"] = gev_return_level(mu, log_sigma, xi, 100)
    result["scenario"] = scenario
    return result


def simulate_annual_maxima(
    truth: pd.DataFrame,
    n_years: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw independent annual maxima conditional on the spatial GEV surfaces."""
    return genextreme.rvs(
        c=-truth["xi_true"].to_numpy(float)[:, None],
        loc=truth["mu_true"].to_numpy(float)[:, None],
        scale=np.exp(truth["log_sigma_true"].to_numpy(float))[:, None],
        size=(len(truth), n_years),
        random_state=rng,
    )


def estimate_grid_with_frozen_nn(
    annual_maxima: np.ndarray,
    model,
    device: str,
) -> pd.DataFrame:
    """Apply the frozen NN independently to every newly simulated time series."""
    rows = []
    for values in annual_maxima:
        mu, sigma, shape_c = estimate_one(model, values, device)
        rows.append(
            {
                "mu_nn": mu,
                "log_sigma_nn": np.log(sigma),
                "xi_nn": -shape_c,
            }
        )
    result = pd.DataFrame(rows)
    result["RL50_nn"] = gev_return_level(
        result["mu_nn"], result["log_sigma_nn"], result["xi_nn"], 50
    )
    result["RL100_nn"] = gev_return_level(
        result["mu_nn"], result["log_sigma_nn"], result["xi_nn"], 100
    )
    return result


def candidate_models() -> list[dict]:
    """Candidate mean/covariance structures selected only in inner CV."""
    covariance_specs = (
        ("RBF_isotropic", "RBF", False),
        ("Matern15_isotropic", "Matern", False),
        ("Matern15_anisotropic", "Matern", True),
    )
    return [
        {
            "model_id": f"{trend}_{label}",
            "trend": trend,
            "kernel": kernel,
            "anisotropic": anisotropic,
        }
        for trend in ("T0", "T1")
        for label, kernel, anisotropic in covariance_specs
    ]


def _design(elevation: np.ndarray, trend: str, center: float, scale: float):
    if trend == "T0":
        return np.ones((len(elevation), 1))
    z = (np.asarray(elevation, float) - center) / scale
    return np.column_stack([np.ones(len(z)), z])


def _candidate_kernel(spec: dict, variance: float):
    amplitude = C(max(variance, 1e-6), (1e-5, max(variance * 1e4, 1e2)))
    length = [50.0, 50.0] if spec["anisotropic"] else 50.0
    bounds = (2.0, 300.0)
    if spec["kernel"] == "RBF":
        spatial = RBF(length_scale=length, length_scale_bounds=bounds)
    else:
        spatial = Matern(length_scale=length, length_scale_bounds=bounds, nu=1.5)
    nugget = WhiteKernel(
        noise_level=max(variance * 0.05, 1e-6),
        noise_level_bounds=(1e-7, max(variance * 2.0, 1e-4)),
    )
    return amplitude * spatial + nugget


def fit_candidate_gp(
    train: pd.DataFrame,
    response_col: str,
    spec: dict,
    seed: int,
    n_restarts_optimizer: int,
) -> dict:
    """Fit a linear mean plus stationary GP residual model."""
    elevation = train["elevation_m"].to_numpy(float)
    center = float(elevation.mean())
    scale = max(float(elevation.std()), 1.0)
    design = _design(elevation, spec["trend"], center, scale)
    response = train[response_col].to_numpy(float)
    beta = np.linalg.lstsq(design, response, rcond=None)[0]
    residual = response - design @ beta
    gp = GaussianProcessRegressor(
        kernel=_candidate_kernel(spec, np.var(residual, ddof=1)),
        normalize_y=False,
        alpha=1e-8,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(train[["x_km", "y_km"]].to_numpy(float), residual)
    return {
        "gp": gp,
        "beta": beta,
        "trend": spec["trend"],
        "elevation_center": center,
        "elevation_scale": scale,
    }


def predict_candidate_gp(model: dict, test: pd.DataFrame) -> np.ndarray:
    design = _design(
        test["elevation_m"].to_numpy(float),
        model["trend"],
        model["elevation_center"],
        model["elevation_scale"],
    )
    residual = model["gp"].predict(test[["x_km", "y_km"]].to_numpy(float))
    return design @ model["beta"] + residual


def _kmeans_folds(frame: pd.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    return KMeans(
        n_clusters=n_folds,
        n_init=30,
        random_state=seed,
    ).fit_predict(frame[["x_km", "y_km"]].to_numpy(float))


def _buffered_train_indices(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    buffer_km: float,
) -> np.ndarray:
    train_xy = frame.loc[train_indices, ["x_km", "y_km"]].to_numpy(float)
    test_xy = frame.loc[test_indices, ["x_km", "y_km"]].to_numpy(float)
    squared = np.sum((train_xy[:, None, :] - test_xy[None, :, :]) ** 2, axis=2)
    keep = np.sqrt(squared.min(axis=1)) >= buffer_km
    return train_indices[keep]


def _cap_indices(indices: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(indices) <= maximum:
        return np.asarray(indices, dtype=int)
    return np.sort(np.random.default_rng(seed).choice(indices, maximum, replace=False))


def _inner_score(
    outer_train: pd.DataFrame,
    response_col: str,
    spec: dict,
    config: SimulationConfig,
    seed: int,
) -> float:
    inner = outer_train.copy().reset_index(drop=True)
    inner["inner_fold"] = _kmeans_folds(inner, config.inner_folds, seed)
    squared_errors = []
    for fold in range(config.inner_folds):
        test_idx = inner.index[inner["inner_fold"] == fold].to_numpy()
        base_idx = inner.index[inner["inner_fold"] != fold].to_numpy()
        train_idx = _buffered_train_indices(
            inner, base_idx, test_idx, config.buffer_km
        )
        train_idx = _cap_indices(train_idx, config.max_train, seed + fold)
        if len(train_idx) < config.min_train:
            raise ValueError(
                "Too few inner training cells after buffering: "
                f"{len(train_idx)} < {config.min_train}."
            )
        model = fit_candidate_gp(
            inner.loc[train_idx], response_col, spec, seed + fold,
            config.n_restarts_optimizer,
        )
        prediction = predict_candidate_gp(model, inner.loc[test_idx])
        squared_errors.extend(
            (prediction - inner.loc[test_idx, response_col].to_numpy(float)) ** 2
        )
    return float(np.mean(squared_errors))


def nested_buffered_spatial_cv(
    data: pd.DataFrame,
    config: SimulationConfig,
    replicate_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select models in inner CV and estimate performance in untouched outer folds."""
    working = data.copy().reset_index(drop=True)
    working["outer_fold"] = _kmeans_folds(
        working, config.outer_folds, replicate_seed
    )
    predictions = []
    selections = []
    specs = candidate_models()

    for target_order, target in enumerate(TARGETS):
        response_col = f"{target}_nn"
        for outer_fold in range(config.outer_folds):
            test_idx = working.index[working["outer_fold"] == outer_fold].to_numpy()
            base_idx = working.index[working["outer_fold"] != outer_fold].to_numpy()
            train_idx = _buffered_train_indices(
                working, base_idx, test_idx, config.buffer_km
            )
            train_idx = _cap_indices(
                train_idx,
                config.max_train,
                replicate_seed + target_order * 1000 + outer_fold,
            )
            outer_train = working.loc[train_idx].copy()
            scores = []
            for model_order, spec in enumerate(specs):
                score = _inner_score(
                    outer_train,
                    response_col,
                    spec,
                    config,
                    replicate_seed
                    + target_order * 100_000
                    + outer_fold * 1_000
                    + model_order * 10,
                )
                scores.append((score, spec))
            best_score, best_spec = min(scores, key=lambda item: item[0])
            final_model = fit_candidate_gp(
                outer_train,
                response_col,
                best_spec,
                replicate_seed + target_order * 10_000 + outer_fold,
                config.n_restarts_optimizer,
            )
            test = working.loc[test_idx]
            prediction = predict_candidate_gp(final_model, test)
            predictions.append(
                pd.DataFrame(
                    {
                        "cell_id": test["cell_id"].to_numpy(),
                        "outer_fold": outer_fold,
                        "target": target,
                        "nn_value": test[response_col].to_numpy(float),
                        "true_value": test[f"{target}_true"].to_numpy(float),
                        "oof_prediction": prediction,
                    }
                )
            )
            selections.append(
                {
                    "target": target,
                    "outer_fold": outer_fold,
                    "selected_model": best_spec["model_id"],
                    "inner_MSE": best_score,
                    "n_outer_train": len(outer_train),
                    "n_outer_test": len(test),
                }
            )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(selections)


def _wide_oof(data: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    wide = data.copy()
    for target in TARGETS:
        part = predictions.loc[
            predictions["target"] == target,
            ["cell_id", "oof_prediction"],
        ].rename(columns={"oof_prediction": f"{target}_oof"})
        wide = wide.merge(part, on="cell_id", how="left", validate="one_to_one")
    for period in (50, 100):
        wide[f"RL{period}_oof"] = gev_return_level(
            wide["mu_oof"], wide["log_sigma_oof"], wide["xi_oof"], period
        )
    return wide


def _metric_rows(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    outcomes = [*TARGETS, "RL50", "RL100"]
    for outcome in outcomes:
        true_col = f"{outcome}_true"
        for stage, estimate_col in (
            ("NN", f"{outcome}_nn"),
            ("NN_plus_GP_OOF", f"{outcome}_oof"),
        ):
            error = wide[estimate_col].to_numpy(float) - wide[true_col].to_numpy(float)
            rows.append(
                {
                    "outcome": outcome,
                    "stage": stage,
                    "RMSE_vs_truth": float(np.sqrt(np.mean(error**2))),
                    "MAE_vs_truth": float(np.mean(np.abs(error))),
                    "Bias_vs_truth": float(np.mean(error)),
                }
            )
        if outcome in TARGETS:
            reconstruction = (
                wide[f"{outcome}_oof"].to_numpy(float)
                - wide[f"{outcome}_nn"].to_numpy(float)
            )
            rows.append(
                {
                    "outcome": outcome,
                    "stage": "GP_reconstruction_of_NN",
                    "RMSE_vs_truth": float(np.sqrt(np.mean(reconstruction**2))),
                    "MAE_vs_truth": float(np.mean(np.abs(reconstruction))),
                    "Bias_vs_truth": float(np.mean(reconstruction)),
                }
            )
    return pd.DataFrame(rows)


def run_one_replicate(
    scenario: str,
    replicate: int,
    config: SimulationConfig,
    model,
    device: str,
) -> dict:
    seed = config.seed + SCENARIOS.index(scenario) * 1_000_000 + replicate * 10_000
    rng = np.random.default_rng(seed)
    truth = generate_true_parameters(make_grid(config), scenario, rng)
    annual_maxima = simulate_annual_maxima(truth, config.n_years, rng)
    estimates = estimate_grid_with_frozen_nn(annual_maxima, model, device)
    data = pd.concat([truth.reset_index(drop=True), estimates], axis=1)
    predictions, selections = nested_buffered_spatial_cv(data, config, seed)
    wide = _wide_oof(data, predictions)
    metrics = _metric_rows(wide)
    for frame in (wide, predictions, selections, metrics):
        frame.insert(0, "replicate", replicate)
        if "scenario" not in frame.columns:
            frame.insert(0, "scenario", scenario)
    return {
        "grid": wide,
        "predictions": predictions,
        "selections": selections,
        "metrics": metrics,
    }


_WORKER_MODEL_CACHE: dict[str, tuple] = {}


def _parallel_replicate_worker(
    scenario: str,
    replicate: int,
    config: SimulationConfig,
    model_path: str | Path | None,
) -> dict:
    """Load one CPU model per worker and execute one independent replicate."""
    cache_key = str(Path(model_path).resolve()) if model_path else "baseline"
    if cache_key not in _WORKER_MODEL_CACHE:
        _WORKER_MODEL_CACHE[cache_key] = load_baseline_model(
            model_path=model_path,
            device="cpu",
        )
    model, device = _WORKER_MODEL_CACHE[cache_key]
    return run_one_replicate(
        scenario,
        replicate,
        config,
        model,
        device,
    )


def run_monte_carlo(
    config: SimulationConfig,
    scenarios=SCENARIOS,
    model_path: str | Path | None = None,
) -> dict:
    """Run independent replicates and print completion progress.

    ``config.n_jobs=-2`` uses all detected logical CPU cores except one.
    Replicates are independent parallel jobs.  Each worker restricts nested
    BLAS/OpenMP work to one thread to avoid CPU oversubscription.
    """
    tasks = [
        (scenario, replicate)
        for scenario in scenarios
        for replicate in range(config.n_replicates)
    ]
    total = len(tasks)
    outputs = []
    workers = effective_n_jobs(config.n_jobs)
    print(
        f"開始 Monte Carlo：共 {total} 次模擬，使用 {workers} 個 CPU worker。",
        flush=True,
    )

    if config.n_jobs == 1:
        model, device = load_baseline_model(model_path=model_path)
        iterator = (
            (
                scenario,
                replicate,
                run_one_replicate(
                    scenario, replicate, config, model, device
                ),
            )
            for scenario, replicate in tasks
        )
        for completed, (scenario, replicate, result) in enumerate(iterator, 1):
            outputs.append(result)
            print(
                f"已完成 {completed}/{total}：{scenario}，"
                f"replicate {replicate + 1}/{config.n_replicates}",
                flush=True,
            )
    else:
        with parallel_backend("loky", inner_max_num_threads=1):
            generator = Parallel(
                n_jobs=config.n_jobs,
                return_as="generator_unordered",
            )(
                delayed(_parallel_replicate_worker)(
                    scenario,
                    replicate,
                    config,
                    model_path,
                )
                for scenario, replicate in tasks
            )
            for completed, result in enumerate(generator, 1):
                scenario = str(result["metrics"]["scenario"].iloc[0])
                replicate = int(result["metrics"]["replicate"].iloc[0])
                outputs.append(result)
                print(
                    f"已完成 {completed}/{total}：{scenario}，"
                    f"replicate {replicate + 1}/{config.n_replicates}",
                    flush=True,
                )

    print(f"全部完成：{total}/{total} 次模擬。", flush=True)
    return {
        key: pd.concat([result[key] for result in outputs], ignore_index=True)
        for key in ("grid", "predictions", "selections", "metrics")
    }


def summarize_monte_carlo(results: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_summary = (
        results["metrics"]
        .groupby(["scenario", "outcome", "stage"], as_index=False)
        .agg(
            mean_RMSE=("RMSE_vs_truth", "mean"),
            sd_RMSE=("RMSE_vs_truth", "std"),
            mean_MAE=("MAE_vs_truth", "mean"),
            mean_Bias=("Bias_vs_truth", "mean"),
        )
    )
    selection_frequency = (
        results["selections"]
        .groupby(["scenario", "target", "selected_model"], as_index=False)
        .size()
        .rename(columns={"size": "selection_count"})
    )
    selection_frequency["selection_rate"] = selection_frequency.groupby(
        ["scenario", "target"]
    )["selection_count"].transform(lambda values: values / values.sum())
    return metric_summary, selection_frequency


def plot_first_replicate_surfaces(results: dict, scenario: str) -> plt.Figure:
    data = results["grid"].query("scenario == @scenario and replicate == 0")
    figure, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for row, target in enumerate(TARGETS):
        columns = [f"{target}_{suffix}" for suffix in ("true", "nn", "oof")]
        shared_values = data[columns].to_numpy(float)
        vmin = float(np.nanmin(shared_values))
        vmax = float(np.nanmax(shared_values))
        for axis, (suffix, title) in zip(
            axes[row], (("true", "Truth"), ("nn", "Frozen NN"), ("oof", "Nested OOF GP"))
        ):
            points = axis.scatter(
                data["x_km"], data["y_km"], c=data[f"{target}_{suffix}"],
                s=18, cmap="viridis", vmin=vmin, vmax=vmax,
            )
            figure.colorbar(points, ax=axis, shrink=0.8)
            axis.set(title=f"{target}: {title}", aspect="equal")
    figure.suptitle(f"Known truth, NN estimates, and OOF GP predictions: {scenario}")
    return figure


def plot_return_level_surfaces(results: dict, scenario: str) -> plt.Figure:
    data = results["grid"].query("scenario == @scenario and replicate == 0")
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row, period in enumerate((50, 100)):
        columns = [f"RL{period}_{suffix}" for suffix in ("true", "nn", "oof")]
        shared_values = data[columns].to_numpy(float)
        vmin = float(np.nanmin(shared_values))
        vmax = float(np.nanmax(shared_values))
        for axis, (suffix, title) in zip(
            axes[row], (("true", "Truth"), ("nn", "Frozen NN"), ("oof", "Nested OOF GP"))
        ):
            points = axis.scatter(
                data["x_km"], data["y_km"], c=data[f"RL{period}_{suffix}"],
                s=18, cmap="magma", vmin=vmin, vmax=vmax,
            )
            figure.colorbar(points, ax=axis, shrink=0.8)
            axis.set(title=f"RL{period}: {title}", aspect="equal")
    figure.suptitle(f"Return-level recovery against known truth: {scenario}")
    return figure


def save_results(
    results: dict,
    config: SimulationConfig,
    prefix: str = "downstream_spatial_simulation",
) -> dict[str, Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary, frequency = summarize_monte_carlo(results)
    paths = {}
    frames = {
        "grid": results["grid"],
        "metrics": results["metrics"],
        "model_selections": results["selections"],
        "metric_summary": summary,
        "selection_frequency": frequency,
        "config": pd.DataFrame([asdict(config)]),
    }
    for name, frame in frames.items():
        path = TABLE_DIR / f"{prefix}_{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = path
    for scenario in SCENARIOS:
        if scenario not in set(results["grid"]["scenario"]):
            continue
        for label, figure in (
            ("parameters", plot_first_replicate_surfaces(results, scenario)),
            ("return_levels", plot_return_level_surfaces(results, scenario)),
        ):
            path = FIGURE_DIR / f"{prefix}_{scenario}_{label}.png"
            figure.savefig(path, dpi=180, bbox_inches="tight")
            paths[f"{scenario}_{label}"] = path
    return paths


if __name__ == "__main__":
    pilot = SimulationConfig(grid_side=12, n_replicates=1, max_train=100)
    simulation_results = run_monte_carlo(pilot)
    save_results(simulation_results, pilot)
