from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from benchmark_cheng_21d_test_mle import fit_case, paired_metrics, validate_fit_rows
from cheng_nn_simulation import ParameterRanges, SimulationConfig, simulate_chunk


def test_paired_errors_use_same_cases_and_sign():
    truth = np.zeros((3, 5))
    nn = np.ones((3, 5))
    nn[2] = 100
    mle = np.full((3, 5), 2.)
    mle[2] = np.nan
    result = paired_metrics(truth, nn, mle, [True, True, False])
    assert result.n_paired.eq(2).all() and result.n_excluded.eq(1).all()
    assert result.NN_RMSE.eq(1).all() and result.MLE_RMSE.eq(2).all()
    assert result.NN_improvement_percent.eq(50).all()


def test_nonfinite_nn_is_not_silently_dropped():
    with pytest.raises(ValueError, match='finite'):
        paired_metrics(np.zeros((1, 5)), np.full((1, 5), np.nan), np.ones((1, 5)), [True])


def test_resume_rejects_duplicate_fit_indices():
    frame = pd.DataFrame(dict(sample_index=[0, 0], domain=['training_domain']*2))
    with pytest.raises(ValueError, match='duplicated'):
        validate_fit_rows(frame, 10)


def test_data_only_fit_smoke():
    config = SimulationConfig()
    *_, maxima = simulate_chunk(np.random.default_rng(717), 1, config, ParameterRanges())
    result = fit_case(3, maxima[0], config.centered_time, ParameterRanges())
    assert result['sample_index'] == 3 and result['success']
    assert result['min_support_margin'] > 0
    assert result['seconds'] > 0
