from pathlib import Path
import sys

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from calibrated_parametric_simulation import CalibratedSimulationConfig  # noqa: E402
import calibrated_simulation_study as study  # noqa: E402


def test_update_simulation_time_records_running_average_and_total(tmp_path):
    path = tmp_path / "simulation_time.csv"
    common = {
        "status": "completed",
        "started_at_utc": "2026-08-26T00:00:00+00:00",
        "finished_at_utc": "2026-08-26T00:00:10+00:00",
    }
    study.update_simulation_time(
        path,
        {
            **common,
            "replicate": 0,
            "replicate_label": "replicate_000",
            "elapsed_seconds": 10.0,
            "elapsed_hms": "00:00:10",
        },
    )
    result = study.update_simulation_time(
        path,
        {
            **common,
            "replicate": 1,
            "replicate_label": "replicate_001",
            "elapsed_seconds": 20.0,
            "elapsed_hms": "00:00:20",
        },
    )

    assert np.allclose(result["running_average_seconds"], [10.0, 15.0])
    assert np.allclose(result["cumulative_seconds"], [10.0, 30.0])
    assert result.iloc[-1]["running_average_hms"] == "00:00:15"
    assert result.iloc[-1]["cumulative_hms"] == "00:00:30"
    assert len(pd.read_csv(path)) == 2


def test_summarize_replicate_metrics_reports_requested_statistics():
    metrics = pd.DataFrame(
        {
            "result_type": ["parameter"] * 3,
            "outcome": ["mu"] * 3,
            "estimator": ["Nested OOF GP"] * 3,
            "RMSE": [1.0, 2.0, 3.0],
            "MAE": [0.5, 1.0, 1.5],
            "Bias": [-0.1, 0.0, 0.1],
        }
    )

    summary = study.summarize_replicate_metrics(metrics)
    rmse = summary.loc[summary["metric"].eq("RMSE")].iloc[0]

    assert rmse["n_replicates"] == 3
    assert rmse["mean"] == 2.0
    assert rmse["median"] == 2.0
    assert rmse["minimum"] == 1.0
    assert rmse["maximum"] == 3.0


def test_complete_study_times_every_full_replicate(tmp_path, monkeypatch):
    output = tmp_path / "simulation"
    cv_root = output / "nested"
    time_path = output / "simulation_time.csv"

    monkeypatch.setattr(
        study,
        "prepare_calibrated_simulation",
        lambda **kwargs: object(),
    )

    def fake_generate(**kwargs):
        replicate = kwargs["replicate"]
        path = output / f"replicate_{replicate:03d}_model_ready.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"station": ["g0"]}).to_csv(path, index=False)
        return {"model_ready": path}

    calls = []
    completed = set()

    def fake_evaluation(input_path, output_directory, **kwargs):
        calls.append(Path(input_path).name)
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        completed.add(int(Path(output_directory).name.split("_")[-1]))
        return {}

    monkeypatch.setattr(
        study, "generate_calibrated_annual_replicate", fake_generate
    )
    monkeypatch.setattr(study, "run_annual_evaluation", fake_evaluation)
    monkeypatch.setattr(
        study,
        "_replicate_is_complete",
        lambda output_directory, cv_root, replicate, *args, **kwargs: (
            replicate in completed
        ),
    )
    monkeypatch.setattr(
        study,
        "aggregate_completed_replicates",
        lambda *args, **kwargs: {"metric_summary": pd.DataFrame()},
    )

    result = study.run_complete_simulation_study(
        study_config=study.CompleteSimulationStudyConfig(
            n_replicates=2,
            block_scale="annual",
            n_jobs=1,
            resume=False,
            save_maxima=False,
            retry_delay_seconds=0,
        ),
        simulation_config=CalibratedSimulationConfig(
            n_replicates=2,
            n_years=1,
            months_per_year=1,
        ),
        output_directory=output,
        cv_root=cv_root,
        time_path=time_path,
    )

    timing = pd.read_csv(time_path)
    assert calls == ["replicate_000_model_ready.csv", "replicate_001_model_ready.csv"]
    assert timing["status"].tolist() == ["completed", "completed"]
    assert np.isfinite(timing["elapsed_seconds"]).all()
    assert "metric_summary" in result


def test_transient_failure_is_retried_without_skipping_replicate(
    tmp_path, monkeypatch
):
    output = tmp_path / "simulation"
    cv_root = output / "nested"
    attempts = []
    completed = set()

    monkeypatch.setattr(
        study,
        "prepare_calibrated_simulation",
        lambda **kwargs: object(),
    )

    def fake_generate(**kwargs):
        path = output / "replicate_000_model_ready.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"station": ["g0"]}).to_csv(path, index=False)
        return {"model_ready": path}

    def flaky_evaluation(input_path, output_directory, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("temporary read failure")
        completed.add(0)

    monkeypatch.setattr(
        study, "generate_calibrated_annual_replicate", fake_generate
    )
    monkeypatch.setattr(study, "run_annual_evaluation", flaky_evaluation)
    monkeypatch.setattr(
        study,
        "_replicate_is_complete",
        lambda output_directory, cv_root, replicate, *args, **kwargs: (
            replicate in completed
        ),
    )
    monkeypatch.setattr(
        study,
        "aggregate_completed_replicates",
        lambda *args, **kwargs: {"metric_summary": pd.DataFrame()},
    )

    study.run_complete_simulation_study(
        study_config=study.CompleteSimulationStudyConfig(
            n_replicates=1,
            block_scale="annual",
            save_maxima=False,
            max_attempts=3,
            retry_delay_seconds=0,
        ),
        simulation_config=CalibratedSimulationConfig(
            n_replicates=1,
            n_years=2,
            months_per_year=1,
        ),
        output_directory=output,
        cv_root=cv_root,
    )

    timing = pd.read_csv(output / "simulation_time.csv")
    assert len(attempts) == 2
    assert timing.loc[0, "status"] == "completed"
    assert timing.loc[0, "attempts"] == 2
