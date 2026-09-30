"""Regression checks for the 1976--2025 Cheng NN simulation design."""

import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cheng_nn_simulation import (
    DATA_SCHEMA_VERSION,
    INPUT_FEATURE_NAMES,
    ParameterRanges,
    QUANTILE_LEVELS,
    SAMPLING_SCHEME,
    SimulationConfig,
    build_input_features,
    build_parser,
    draw_coefficients,
    generate_split,
    simulate_chunk,
    time_segment_slices,
)


def test_default_cli_uses_100k_sequence_811_dataset():
    args = build_parser().parse_args([])
    assert (args.n_train, args.n_validation, args.n_test) == (80_000, 10_000, 10_000)
    assert args.output_directory.name == 'cheng_nn_17d'
    assert args.seed == 20260930
    assert args.n_years == 50


def test_global_pilot_draws_scale_relative_location_trend() -> None:
    coefficients = draw_coefficients(np.random.default_rng(123), 512, ParameterRanges())
    mu0, beta_mu, eta0, beta_sigma, xi0 = coefficients.T
    sigma0 = np.exp(eta0)
    ratio = beta_mu / sigma0

    for values, lower, upper in (
        (mu0, -50.0, 60.0),
        (sigma0, 0.2, 8.0),
        (ratio, -0.5, 0.5),
        (beta_sigma, -0.2, 0.2),
        (xi0, -0.4, 0.4),
    ):
        assert np.all((values >= lower) & (values <= upper))
    assert np.any(mu0 < 0.0)
    assert np.any(np.abs(beta_mu) > 1.0)


def test_location_trend_scales_with_reference_sigma() -> None:
    base_ranges = ParameterRanges(beta_mu_over_sigma0=(0.1, 0.3))
    scaled_ranges = ParameterRanges(
        beta_mu_over_sigma0=(0.1, 0.3),
        eta0=tuple(bound + np.log(3.0) for bound in base_ranges.eta0),
    )
    base = draw_coefficients(np.random.default_rng(8), 32, base_ranges)
    scaled = draw_coefficients(np.random.default_rng(8), 32, scaled_ranges)

    np.testing.assert_allclose(scaled[:, 1], 3.0 * base[:, 1])
    np.testing.assert_allclose(np.exp(scaled[:, 2]), 3.0 * np.exp(base[:, 2]))
    np.testing.assert_allclose(scaled[:, [0, 3, 4]], base[:, [0, 3, 4]])
    assert np.all(base[:, 1] > 0.0)


def test_default_simulation_uses_fifty_years() -> None:
    config = SimulationConfig()

    assert config.years[0] == 1976
    assert config.years[-1] == 2025
    assert config.n_years == 50
    assert config.reference_year == 2000.5
    np.testing.assert_allclose(np.mean(config.centered_time), 0.0, atol=1e-15)


def test_seventeen_features_retain_period_order() -> None:
    values = np.arange(1.0, 51.0)
    inputs, location_scale = build_input_features(
        np.stack((values, values[::-1]))
    )

    assert inputs.shape == (2, 17)
    assert len(INPUT_FEATURE_NAMES) == 17
    assert [(part.start, part.stop) for part in time_segment_slices(50)] == [
        (0, 16), (16, 33), (33, 50)
    ]
    np.testing.assert_allclose(location_scale, [[25.5, 24.5], [25.5, 24.5]])
    np.testing.assert_allclose(inputs[0, :11], inputs[1, :11])
    np.testing.assert_allclose(inputs[:, 5], 0.0)
    np.testing.assert_allclose(inputs[:, 6] - inputs[:, 4], 1.0)
    assert inputs[0, 11] < inputs[0, 13] < inputs[0, 15]
    assert inputs[1, 11] > inputs[1, 13] > inputs[1, 15]

    standardized = (values - 25.5) / 24.5
    np.testing.assert_allclose(
        inputs[0, :11], np.quantile(standardized, QUANTILE_LEVELS)
    )
    for period, median_index in zip(time_segment_slices(50), (11, 13, 15)):
        period_values = standardized[period]
        np.testing.assert_allclose(inputs[0, median_index], np.median(period_values))
        q1, q3 = np.quantile(period_values, [0.25, 0.75])
        np.testing.assert_allclose(inputs[0, median_index + 1], q3 - q1)


def test_simulated_inputs_and_targets_use_same_sample_standardization() -> None:
    inputs, targets, coefficients, location_scale, annual_maxima = simulate_chunk(
        np.random.default_rng(123),
        n_samples=3,
        config=SimulationConfig(),
        ranges=ParameterRanges(),
    )

    assert inputs.shape == (3, 17)
    assert targets.shape == (3, 5)
    assert annual_maxima.shape == (3, 50)
    reproduced_inputs, reproduced_scale = build_input_features(annual_maxima)
    np.testing.assert_allclose(inputs, reproduced_inputs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(location_scale, reproduced_scale)
    median, iqr = location_scale.T
    np.testing.assert_allclose(inputs[:, 5], 0.0, atol=1e-6)
    np.testing.assert_allclose(inputs[:, 6] - inputs[:, 4], 1.0, atol=1e-6)
    np.testing.assert_allclose(targets[:, 0] * iqr + median, coefficients[:, 0], atol=1e-5)
    np.testing.assert_allclose(targets[:, 1] * iqr, coefficients[:, 1], atol=1e-6)
    np.testing.assert_allclose(targets[:, 2] + np.log(iqr), coefficients[:, 2], atol=1e-6)
    np.testing.assert_allclose(targets[:, 3:], coefficients[:, 3:])


def test_saved_split_has_seventeen_feature_schema(tmp_path: Path) -> None:
    directory = generate_split(
        tmp_path,
        split_name="train",
        n_samples=2,
        seed=123,
        config=SimulationConfig(chunk_size=1),
        ranges=ParameterRanges(),
    )

    assert np.load(directory / "inputs.npy").shape == (2, 17)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["data_schema_version"] == DATA_SCHEMA_VERSION
    annual_maxima = np.load(directory / "annual_maxima.npy")
    assert annual_maxima.shape == (2, 50)
    assert annual_maxima.dtype == np.float64
    assert np.load(directory / "coefficients_original.npy").dtype == np.float64
    reproduced_inputs, _ = build_input_features(annual_maxima)
    np.testing.assert_allclose(np.load(directory / "inputs.npy"), reproduced_inputs, rtol=1e-6, atol=1e-6)
    assert metadata["sampling_scheme"] == SAMPLING_SCHEME
    assert metadata["parameter_ranges"] == {
        name: list(bounds) for name, bounds in asdict(ParameterRanges()).items()
    }
    assert metadata["coefficient_relationships"]["beta_mu"] == "beta_mu_over_sigma0 * sigma0"
    assert metadata["input_shape_per_sample"] == [17]
    assert metadata["input_columns"] == list(INPUT_FEATURE_NAMES)
    assert metadata["input_periods"] == {
        "early": [1976, 1991],
        "middle": [1992, 2008],
        "late": [2009, 2025],
    }


@pytest.mark.parametrize("stale_field", ["sampling_scheme", "parameter_ranges", "time_scale_years", "data_schema_version", "n_samples"])
def test_notebook_loader_rejects_incompatible_simulation(tmp_path: Path, stale_field: str) -> None:
    config = SimulationConfig()
    ranges = ParameterRanges()
    notebook_path = SRC.parent / "notebooks" / "cheng_NN.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    loader_source = next(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["id"] == "cheng-load"
    )
    namespace = {
        "DATA_DIR": tmp_path,
        "SIMULATION_CONFIG": config,
        "PARAMETER_RANGES": ranges,
        "SAMPLING_SCHEME": SAMPLING_SCHEME,
        "DATA_SCHEMA_VERSION": DATA_SCHEMA_VERSION,
        "N_TRAIN": 2, "N_VALIDATION": 2, "N_TEST": 2,
        "INPUT_FEATURE_NAMES": INPUT_FEATURE_NAMES,
        "asdict": asdict,
        "np": np,
        "json": json,
    }
    exec(compile(loader_source, str(notebook_path), "exec"), namespace)
    directory = generate_split(tmp_path, "train", 2, 123, config, ranges)
    assert namespace["load_split"]("train")["inputs"].shape == (2, 17)

    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if stale_field == "parameter_ranges":
        metadata[stale_field]["mu0"] = [20.0, 45.0]
    elif stale_field == "time_scale_years":
        metadata[stale_field] = 1.0
    else:
        metadata.pop(stale_field)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="sampling scheme|50-year|schema or split size"):
        namespace["load_split"]("train")
