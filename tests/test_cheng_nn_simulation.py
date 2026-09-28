"""Regression checks for the 1976--2025 Cheng NN simulation design."""

import json
from pathlib import Path
import sys

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cheng_nn_simulation import (
    INPUT_FEATURE_NAMES,
    ParameterRanges,
    QUANTILE_LEVELS,
    SimulationConfig,
    build_input_features,
    generate_split,
    simulate_chunk,
    time_segment_slices,
)


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
    inputs, targets, coefficients, location_scale = simulate_chunk(
        np.random.default_rng(123),
        n_samples=3,
        config=SimulationConfig(),
        ranges=ParameterRanges(),
    )

    assert inputs.shape == (3, 17)
    assert targets.shape == (3, 5)
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
    assert metadata["input_shape_per_sample"] == [17]
    assert metadata["input_columns"] == list(INPUT_FEATURE_NAMES)
    assert metadata["input_periods"] == {
        "early": [1976, 1991],
        "middle": [1992, 2008],
        "late": [2009, 2025],
    }
