"""The Windows extension reuses only complete, same-protocol CV cases."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/thesis_run_bundle'))
from portable import resumable_cv
from research_workflows import COEFFICIENTS
from taiwan_simulation import cheng_spatial_cv as cv
from train_cheng_nn import atomic_csv


def test_checkpointed_nested_cv_matches_fresh_and_recomputes_tampered_case(tmp_path, monkeypatch):
    calls = []

    def select(train, target, **kwargs):
        calls.append((target, kwargs['random_state']))
        return (pd.DataFrame([{'target': target, 'RMSE': 1.0}]),
                pd.DataFrame([{'target': target, 'step': 0}]),
                dict(predictor_names=[], kernel='RBF', nu=None, RMSE=1.0))

    monkeypatch.setattr(cv, 'spatial_forward_selection', select)
    monkeypatch.setattr(cv, 'fit_gp_with_covariates',
                        lambda *args, **kwargs: {'optimizer_success': True})
    monkeypatch.setattr(cv, 'predict_gp_with_covariates',
                        lambda fitted, xy, covariates: xy[:, 0] * 0.001)
    n = 16
    frame = pd.DataFrame(dict(
        grid_id=[f'g{i:03d}' for i in range(n)],
        x_km=np.arange(n, dtype=float) * 12,
        y_km=np.array([i % 3 for i in range(n)], dtype=float) * 5,
        dummy=np.arange(n, dtype=float)))
    for col in COEFFICIENTS:
        frame[col] = np.arange(n, dtype=float) * 0.01
    config = dict(candidate_groups={'dummy': ['dummy']},
        buffers_km={col: 0.0 for col in COEFFICIENTS},
        outer_folds=2, inner_folds=2, min_train=2, max_train=n,
        seed=29, n_jobs=1, n_restarts=0)
    expected = cv.nested_cv(frame, **config)
    assert len(calls) == 10

    def progress(output):
        def record(stage, detail, tables):
            if tables is not None:
                sub = output / 'checkpoints' / f"outer_{detail['outer_fold']}_{detail['coefficient']}"
                sub.mkdir(parents=True, exist_ok=True)
                for name, table in tables.items():
                    atomic_csv(sub / f'{name}.csv', table)
        return record

    first = tmp_path / 'attempt_001'
    actual = resumable_cv.nested_cv(frame, resume_attempts=[], checkpoint_output=first,
        checkpoint_token='same-science-and-sample', progress=progress(first), **config)
    pd.testing.assert_frame_equal(actual['oof'], expected['oof'])
    assert len(calls) == 20
    assert len(list(first.rglob(resumable_cv.MARKER))) == 10

    second = tmp_path / 'attempt_002'
    reused = resumable_cv.nested_cv(frame, resume_attempts=[first], checkpoint_output=second,
        checkpoint_token='same-science-and-sample', progress=progress(second), **config)
    pd.testing.assert_frame_equal(reused['oof'], expected['oof'])
    assert len(calls) == 20  # No fresh predictor selection or GP refit.

    changed = first / 'checkpoints' / 'outer_0_mu0' / 'oof.csv'
    changed.write_text(changed.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    third = tmp_path / 'attempt_003'
    replayed = resumable_cv.nested_cv(frame, resume_attempts=[first], checkpoint_output=third,
        checkpoint_token='same-science-and-sample', progress=progress(third), **config)
    pd.testing.assert_frame_equal(replayed['oof'], expected['oof'])
    assert len(calls) == 21  # Just the altered case is recomputed.
