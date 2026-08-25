from pathlib import Path
import sys

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from calibrated_simulation_spatial_cv import (
    build_return_level_recovery,
    summarize_parameter_recovery,
)


def _predictions() -> pd.DataFrame:
    rows = []
    truth = {
        "mu": [30.0, 31.0],
        "log_sigma": [0.0, 0.1],
        "xi": [0.1, 0.1],
    }
    for target, values in truth.items():
        for row_index, value in enumerate(values):
            rows.append(
                {
                    "row_index": row_index,
                    "station": f"g{row_index}",
                    "outer_fold": row_index,
                    "target": target,
                    "true_value": value,
                    "nn_value": value + 0.2,
                    "oof_prediction": value + 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_parameter_recovery_uses_simulated_truth() -> None:
    metrics = summarize_parameter_recovery(_predictions())
    nn = metrics.loc[metrics["estimator"].eq("Frozen NN"), "RMSE"]
    oof = metrics.loc[metrics["estimator"].eq("Nested OOF GP"), "RMSE"]
    assert np.allclose(nn, 0.2)
    assert np.allclose(oof, 0.1)


def test_return_level_recovery_builds_both_periods() -> None:
    predictions, metrics = build_return_level_recovery(_predictions())
    assert set(predictions["return_period"]) == {50, 100}
    assert set(metrics["estimator"]) == {"Frozen NN", "Nested OOF GP"}
    assert np.isfinite(metrics["RMSE"]).all()
