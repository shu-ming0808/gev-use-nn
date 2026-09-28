"""Generate labelled 50-year nonstationary GEV samples for Cheng NN.

Each simulated item contains 17 summaries of a complete annual-maxima
sequence and its known time-varying GEV coefficients.  The module only
generates data; model fitting is kept in ``notebooks/cheng_NN.ipynb``.

The coefficient convention is

    mu(t)        = mu0 + beta_mu * t_centered
    log sigma(t) = eta0 + beta_sigma * t_centered
    xi(t)        = xi0

where centered time is measured in decades.  The saved NN targets are on the
same median/IQR-standardized scale as the saved temperature sequences.  The
original-scale coefficients are saved separately for auditing and evaluation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


COEFFICIENT_NAMES = (
    "mu0",
    "beta_mu",
    "eta0",
    "beta_sigma",
    "xi0",
)

QUANTILE_LEVELS = (
    0.0001,
    0.001,
    0.01,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.99,
    0.999,
    0.9999,
)
SEGMENT_NAMES = ("early", "middle", "late")
INPUT_FEATURE_NAMES = tuple(f"q_{level:g}" for level in QUANTILE_LEVELS) + tuple(
    f"{segment}_{summary}"
    for segment in SEGMENT_NAMES
    for summary in ("median", "iqr")
)


@dataclass(frozen=True)
class ParameterRanges:
    """Uniform sampling ranges used to build the simulation design."""

    mu0: tuple[float, float] = (20.0, 45.0)
    beta_mu: tuple[float, float] = (-1.0, 1.0)
    eta0: tuple[float, float] = (float(np.log(0.5)), float(np.log(5.0)))
    beta_sigma: tuple[float, float] = (-0.20, 0.20)
    xi0: tuple[float, float] = (-0.40, 0.40)

    def validate(self) -> None:
        for name, bounds in asdict(self).items():
            lower, upper = bounds
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError(f"{name} contains a non-finite bound.")
            if lower >= upper:
                raise ValueError(f"{name} must satisfy lower < upper.")


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration shared by the train, validation, and test splits."""

    start_year: int = 1976
    n_years: int = 50
    time_scale_years: float = 10.0
    chunk_size: int = 4096
    seed: int = 20260923

    @property
    def years(self) -> np.ndarray:
        return np.arange(
            self.start_year,
            self.start_year + self.n_years,
            dtype=np.int16,
        )

    @property
    def reference_year(self) -> float:
        return float(np.mean(self.years.astype(np.float64)))

    @property
    def centered_time(self) -> np.ndarray:
        return (
            self.years.astype(np.float64) - self.reference_year
        ) / self.time_scale_years

    def validate(self) -> None:
        if self.n_years < 3:
            raise ValueError("n_years must be at least 3.")
        if self.time_scale_years <= 0:
            raise ValueError("time_scale_years must be positive.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")


def draw_coefficients(
    rng: np.random.Generator,
    n_samples: int,
    ranges: ParameterRanges,
) -> np.ndarray:
    """Draw original-scale coefficients in canonical column order."""

    bounds = [
        ranges.mu0,
        ranges.beta_mu,
        ranges.eta0,
        ranges.beta_sigma,
        ranges.xi0,
    ]
    columns = [
        rng.uniform(lower, upper, size=n_samples)
        for lower, upper in bounds
    ]
    return np.column_stack(columns)


def gev_inverse_cdf(
    uniform: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    xi: np.ndarray,
    *,
    xi_tolerance: float = 1e-6,
) -> np.ndarray:
    """Evaluate the GEV inverse CDF using the EVT shape convention ``xi``."""

    if np.any(sigma <= 0.0):
        raise ValueError("All GEV scale values must be positive.")

    u = np.clip(np.asarray(uniform, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    reduced = -np.log(u)
    xi_array = np.broadcast_to(np.asarray(xi, dtype=np.float64), u.shape)
    mu_array = np.broadcast_to(np.asarray(mu, dtype=np.float64), u.shape)
    sigma_array = np.broadcast_to(np.asarray(sigma, dtype=np.float64), u.shape)

    near_zero = np.abs(xi_array) < xi_tolerance
    result = np.empty_like(u)
    result[near_zero] = (
        mu_array[near_zero]
        - sigma_array[near_zero] * np.log(reduced[near_zero])
    )

    nonzero = ~near_zero
    xi_nonzero = xi_array[nonzero]
    result[nonzero] = mu_array[nonzero] + (
        sigma_array[nonzero] / xi_nonzero
    ) * (np.power(reduced[nonzero], -xi_nonzero) - 1.0)
    return result


def time_segment_slices(n_years: int) -> tuple[slice, slice, slice]:
    """Split ordered years into three consecutive, nearly equal periods."""

    if n_years < 3:
        raise ValueError("At least three years are required for time summaries.")
    boundaries = np.linspace(0, n_years, 4).astype(int)
    return tuple(
        slice(int(start), int(stop))
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
    )


def build_input_features(
    annual_maxima: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 11 pooled quantiles, six period summaries, and sample scaling.

    The whole sequence is standardized once. Periods are not standardized
    separately because doing so would remove the time-varying signals.
    """

    values = np.asarray(annual_maxima, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("Annual maxima must have shape (samples, at least 3 years).")
    if not np.all(np.isfinite(values)):
        raise ValueError("Annual maxima must be finite.")

    median = np.median(values, axis=1)
    q1, q3 = np.quantile(values, [0.25, 0.75], axis=1)
    iqr = q3 - q1
    if np.any(~np.isfinite(iqr)) or np.any(iqr <= 1e-12):
        raise ValueError("An annual-maxima sequence has an invalid IQR.")

    standardized = (values - median[:, None]) / iqr[:, None]
    pooled_quantiles = np.quantile(
        standardized, QUANTILE_LEVELS, axis=1
    ).T
    period_features = []
    for period in time_segment_slices(values.shape[1]):
        period_values = standardized[:, period]
        period_median = np.median(period_values, axis=1)
        period_q1, period_q3 = np.quantile(
            period_values, [0.25, 0.75], axis=1
        )
        period_features.extend((period_median, period_q3 - period_q1))

    features = np.column_stack((pooled_quantiles, *period_features))
    location_scale = np.column_stack((median, iqr))
    return features, location_scale


def simulate_chunk(
    rng: np.random.Generator,
    n_samples: int,
    config: SimulationConfig,
    ranges: ParameterRanges,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate one chunk and return inputs, targets, truths, and transforms."""

    coefficients = draw_coefficients(rng, n_samples, ranges)
    mu0, beta_mu, eta0, beta_sigma, xi0 = coefficients.T
    time = config.centered_time[None, :]

    mu = mu0[:, None] + beta_mu[:, None] * time
    log_sigma = eta0[:, None] + beta_sigma[:, None] * time
    sigma = np.exp(log_sigma)
    xi = np.broadcast_to(xi0[:, None], mu.shape)

    annual_maxima = gev_inverse_cdf(
        rng.random((n_samples, config.n_years)),
        mu,
        sigma,
        xi,
    )

    inputs, location_scale = build_input_features(annual_maxima)
    median, iqr = location_scale.T

    targets = np.column_stack(
        (
            (mu0 - median) / iqr,
            beta_mu / iqr,
            eta0 - np.log(iqr),
            beta_sigma,
            xi0,
        )
    )
    return (
        inputs.astype(np.float32),
        targets.astype(np.float32),
        coefficients.astype(np.float32),
        location_scale.astype(np.float32),
    )


def _open_split_arrays(
    directory: Path,
    n_samples: int,
) -> dict[str, np.memmap]:
    """Create memory-mapped NPY outputs so generation stays memory bounded."""

    return {
        "inputs": np.lib.format.open_memmap(
            directory / "inputs.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_samples, len(INPUT_FEATURE_NAMES)),
        ),
        "targets": np.lib.format.open_memmap(
            directory / "targets_standardized.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_samples, len(COEFFICIENT_NAMES)),
        ),
        "coefficients": np.lib.format.open_memmap(
            directory / "coefficients_original.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_samples, len(COEFFICIENT_NAMES)),
        ),
        "location_scale": np.lib.format.open_memmap(
            directory / "sample_location_scale.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_samples, 2),
        ),
    }


def generate_split(
    output_directory: str | Path,
    split_name: str,
    n_samples: int,
    seed: int,
    config: SimulationConfig,
    ranges: ParameterRanges,
) -> Path:
    """Generate one independent split using an atomic partial directory."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    final_directory = root / split_name
    partial_directory = root / f".{split_name}.partial"

    if final_directory.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing split: {final_directory}"
        )
    if partial_directory.exists():
        raise FileExistsError(
            "An incomplete split already exists. Inspect or remove it before "
            f"retrying: {partial_directory}"
        )

    partial_directory.mkdir(parents=False)
    arrays = _open_split_arrays(
        partial_directory,
        n_samples=n_samples,
    )
    rng = np.random.default_rng(seed)

    for start in range(0, n_samples, config.chunk_size):
        stop = min(start + config.chunk_size, n_samples)
        chunk = simulate_chunk(rng, stop - start, config, ranges)
        arrays["inputs"][start:stop] = chunk[0]
        arrays["targets"][start:stop] = chunk[1]
        arrays["coefficients"][start:stop] = chunk[2]
        arrays["location_scale"][start:stop] = chunk[3]

    for name in arrays:
        arrays[name].flush()
    del arrays

    metadata = {
        "split": split_name,
        "n_samples": n_samples,
        "seed": int(seed),
        "years": config.years.astype(int).tolist(),
        "reference_year": config.reference_year,
        "time_scale_years": config.time_scale_years,
        "centered_time": config.centered_time.tolist(),
        "input_shape_per_sample": [len(INPUT_FEATURE_NAMES)],
        "input_columns": list(INPUT_FEATURE_NAMES),
        "input_quantile_levels": list(QUANTILE_LEVELS),
        "input_periods": {
            name: [int(config.years[period.start]), int(config.years[period.stop - 1])]
            for name, period in zip(
                SEGMENT_NAMES, time_segment_slices(config.n_years), strict=True
            )
        },
        "input_standardization": "one sample median and IQR across all years",
        "target_columns": list(COEFFICIENT_NAMES),
        "target_scale": "sample median/IQR standardized GEV coefficients",
        "original_coefficient_scale": {
            "mu0": "temperature",
            "beta_mu": "temperature per decade",
            "eta0": "log temperature scale",
            "beta_sigma": "log scale per decade",
            "xi0": "dimensionless",
        },
        "parameter_ranges": asdict(ranges),
        "gev_shape_convention": "EVT xi; scipy.stats.genextreme would use c=-xi",
    }
    (partial_directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    partial_directory.replace(final_directory)
    return final_directory


def generate_all_splits(
    output_directory: str | Path,
    *,
    n_train: int = 300_000,
    n_validation: int = 40_000,
    n_test: int = 40_000,
    config: SimulationConfig | None = None,
    ranges: ParameterRanges | None = None,
) -> dict[str, Path]:
    """Generate mutually independent train, validation, and test datasets."""

    selected_config = config or SimulationConfig()
    selected_ranges = ranges or ParameterRanges()
    selected_config.validate()
    selected_ranges.validate()

    split_sizes = {
        "train": n_train,
        "validation": n_validation,
        "test": n_test,
    }
    child_sequences = np.random.SeedSequence(selected_config.seed).spawn(
        len(split_sizes)
    )
    result: dict[str, Path] = {}
    for (split_name, n_samples), child in zip(
        split_sizes.items(), child_sequences, strict=True
    ):
        child_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        result[split_name] = generate_split(
            output_directory,
            split_name,
            n_samples,
            child_seed,
            selected_config,
            selected_ranges,
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate 50-year time-varying GEV samples for Cheng NN."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data") / "simulated" / "cheng_nn_17d",
    )
    parser.add_argument("--n-train", type=int, default=300_000)
    parser.add_argument("--n-validation", type=int, default=40_000)
    parser.add_argument("--n-test", type=int, default=40_000)
    parser.add_argument("--start-year", type=int, default=1976)
    parser.add_argument("--n-years", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260923)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SimulationConfig(
        start_year=args.start_year,
        n_years=args.n_years,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
    generated = generate_all_splits(
        args.output_directory,
        n_train=args.n_train,
        n_validation=args.n_validation,
        n_test=args.n_test,
        config=config,
    )
    for split_name, directory in generated.items():
        print(f"{split_name}: {directory}")


if __name__ == "__main__":
    main()
