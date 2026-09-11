from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import calibrated_annual_simulation as annual
from calibrated_parametric_simulation import annual_return_level_from_monthly_gev


def _annual_input() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["g0", "g1"],
            "lon": [121.0, 121.1],
            "lat": [23.0, 23.1],
            "x_km": [0.0, 10.0],
            "y_km": [0.0, 10.0],
            "mu_hat": [30.1, 31.1],
            "log_sigma_hat": [0.1, 0.2],
            "xi_hat": [0.05, 0.10],
            "mu_true": [30.0, 31.0],
            "log_sigma_true": [0.0, 0.1],
            "xi_true": [0.04, 0.09],
            "block_scale": ["annual", "annual"],
            "n_years": [45, 45],
            "blocks_per_year": [1, 1],
            "n_blocks": [45, 45],
        }
    )


def _predictions() -> pd.DataFrame:
    rows = []
    for target in ("mu", "log_sigma", "xi"):
        data = _annual_input()
        for row_index, row in data.iterrows():
            rows.append(
                {
                    "row_index": row_index,
                    "station": row["station"],
                    "outer_fold": row_index,
                    "target": target,
                    "true_value": row[f"{target}_true"],
                    "nn_value": row[f"{target}_hat"],
                    "oof_prediction": row[f"{target}_true"] + 0.02,
                }
            )
    return pd.DataFrame(rows)


def test_simulate_annual_maxima_has_one_column_per_year() -> None:
    truth = pd.DataFrame(
        {
            "mu_true": [30.0, 31.0, 32.0],
            "sigma_true": [1.0, 1.2, 0.8],
            "xi_true": [0.0, 0.1, -0.1],
        }
    )
    values = annual.simulate_annual_maxima(
        truth, n_years=45, rng=np.random.default_rng(7)
    )
    assert values.shape == (3, 45)
    assert np.isfinite(values).all()


def test_validate_annual_input_rejects_monthly_data() -> None:
    data = _annual_input()
    data["block_scale"] = "monthly"
    with pytest.raises(ValueError, match="block_scale"):
        annual.validate_annual_input(data)


def test_annual_return_level_uses_annual_probability() -> None:
    mu = np.array([30.0])
    log_sigma = np.array([0.0])
    xi = np.array([0.1])
    annual_rl = annual_return_level_from_monthly_gev(
        mu, log_sigma, xi, 50, months_per_year=1
    )
    monthly_rl = annual_return_level_from_monthly_gev(
        mu, log_sigma, xi, 50, months_per_year=12
    )
    assert monthly_rl[0] > annual_rl[0]


def test_annual_evaluation_persists_provenance_and_n_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "annual.csv"
    output_directory = tmp_path / "cv"
    _annual_input().to_csv(input_path, index=False)
    predictions = _predictions()

    def fake_nested(data: pd.DataFrame, **kwargs):
        assert kwargs["n_jobs"] == -2
        return predictions, pd.DataFrame(
            {
                "target": ["mu", "log_sigma", "xi"],
                "outer_fold": [0, 0, 0],
                "kernel": ["RBF", "Matern", "Matern"],
            }
        )

    monkeypatch.setattr(annual, "run_nested_buffered_spatial_cv", fake_nested)
    outputs = annual.run_annual_evaluation(
        input_path=input_path,
        output_directory=output_directory,
        n_jobs=-2,
    )

    metadata = outputs["metadata"].iloc[0]
    assert metadata["block_scale"] == "annual"
    assert metadata["n_years"] == 45
    assert metadata["n_jobs"] == -2
    assert set(outputs["return_level_metrics"]["return_period"]) == {50, 100}
    assert all(
        (output_directory / f"calibrated_annual_nested_{name}.csv").exists()
        for name in annual.CV_OUTPUT_NAMES
    )
    assert annual.annual_outputs_current(input_path, output_directory)

    (output_directory / "calibrated_annual_nested_parameter_metrics.csv").write_text(
        "broken", encoding="utf-8"
    )
    assert not annual.annual_outputs_current(input_path, output_directory)


def test_resume_does_not_replace_full_run_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_directory = tmp_path / "annual"
    cv_directory = output_directory / "cv"
    output_directory.mkdir()
    _annual_input().to_csv(
        output_directory / "replicate_000_model_ready.csv", index=False
    )
    timing_path = output_directory / "annual_pilot_time.csv"
    original = "elapsed_seconds\n656.28\n"
    timing_path.write_text(original, encoding="utf-8")

    monkeypatch.setattr(annual, "annual_outputs_current", lambda *args: True)
    monkeypatch.setattr(
        annual, "annual_generation_outputs_current", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        annual,
        "load_annual_outputs",
        lambda *args: {"parameter_metrics": pd.DataFrame()},
    )
    annual.run_annual_pilot(
        output_directory=output_directory,
        cv_directory=cv_directory,
        n_jobs=-2,
    )
    assert timing_path.read_text(encoding="utf-8") == original
