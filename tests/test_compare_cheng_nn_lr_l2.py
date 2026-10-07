from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import compare_cheng_nn_lr_l2 as study


def test_six_cases_three_paired_seeds_and_no_test_or_publication(tmp_path):
    plan = study.make_plan()
    assert len(plan) == 18
    assert {(e['case'], e['seed']) for e in plan} == {(c, s) for c in study.CASES for s in study.SEEDS}
    assert set(study.CASES.values()) == {(lr, l2) for lr in (1e-3, 3e-4) for l2 in (0., 1e-5, 1e-4)}
    for entry in plan:
        command = study.training_command(tmp_path/'data', tmp_path/'out', entry, 'cuda')
        assert '--evaluate-test' not in command and '--publish-model' not in command
        assert '--no-update-latest' in command and '--coefficient-only' in command
        assert command[command.index('--hidden-sizes')+1:command.index('--hidden-sizes')+4] == ['128', '128', '64']
        assert float(command[command.index('--learning-rate')+1]) == study.CASES[entry['case']][0]
        assert float(command[command.index('--weight-decay')+1]) == study.CASES[entry['case']][1]


def rows_for_test():
    return [dict(case=c, seed=s, coefficient=p, train_RMSE=1.,
                 validation_RMSE=2. if c == study.BASELINE else 1.)
            for c in study.CASES for s in study.SEEDS for p in study.COEFFICIENT_NAMES]


def test_improvements_paired_by_seed_without_combining_coefficient_units():
    paired, summary = study.aggregate(rows_for_test())
    assert len(paired) == 90 and len(summary) == 30
    assert (summary.n_seeds == 3).all()
    assert (summary.paired_baseline_count == 3).all()
    np.testing.assert_allclose(summary.loc[summary.case.eq(study.BASELINE), 'improvement_percent_mean'], 0.)
    np.testing.assert_allclose(summary.loc[~summary.case.eq(study.BASELINE), 'improvement_percent_mean'], 50.)
    assert not any('RL' in c or 'test' in c for c in summary.columns)


def test_incomplete_and_duplicate_runs_are_not_reported_as_complete():
    rows = rows_for_test()
    with pytest.raises(ValueError):
        study.aggregate(rows[:-1])
    with pytest.raises(ValueError):
        study.aggregate(rows + rows[:1])
    _, partial = study.aggregate(rows[:5], require_complete=False)
    assert (partial.n_seeds == 1).all()


def test_experiment_lock_releases_after_failure(tmp_path):
    with pytest.raises(RuntimeError):
        with study.experiment_lock(tmp_path):
            with pytest.raises(OSError):
                with study.experiment_lock(tmp_path):
                    pass
            raise RuntimeError('simulated interruption')
    with study.experiment_lock(tmp_path):
        pass
