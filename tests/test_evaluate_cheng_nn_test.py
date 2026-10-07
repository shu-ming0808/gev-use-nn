from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_cheng_nn_test import DESIGN_FIELDS, collect_metrics, summarize, validate_design


def test_design_requires_matching_design_and_independent_seed():
    meta = {s: {**{k: "fixed" for k in DESIGN_FIELDS}, "split": s, "seed": i}
            for i, s in enumerate(("train", "validation", "test"))}
    validate_design(meta)
    meta["test"]["seed"] = 0
    with pytest.raises(ValueError, match="distinct"):
        validate_design(meta)
    meta["test"]["seed"] = 2
    meta["test"]["reference_year"] = 1999
    with pytest.raises(ValueError, match="designs"):
        validate_design(meta)


def test_summary_uses_paired_split_gaps_and_rejects_duplicates():
    rows = [dict(model=model, seed=seed, split=split, outcome="RL100", RMSE=value)
            for model in ("small", "large") for seed in (1, 2, 3)
            for split, value in (("train", 2.), ("validation", 3.), ("test", 4.))]
    _, summary, gaps = summarize(rows)
    assert (summary.n_seeds == 3).all()
    assert np.allclose(gaps.test_vs_train_percent, 100.)
    assert np.allclose(gaps.test_vs_validation_percent, 100./3.)
    with pytest.raises(ValueError, match="Duplicated"):
        summarize(rows + [rows[0]])
    with pytest.raises(ValueError, match="all three"):
        summarize(rows[:-1])


def test_rejects_metrics_with_excluded_test_predictions(tmp_path):
    dest = tmp_path / "test"
    dest.mkdir()
    rows = [dict(scale=scale, coefficient=name, RMSE=1., n_total=5, n_used=5)
            for scale in ("original", "train_target_z")
            for name in ("mu0", "beta_mu", "eta0", "beta_sigma", "xi0")]
    pd.DataFrame(rows).to_csv(dest / "coefficient_metrics.csv", index=False)
    rl = pd.DataFrame([dict(return_period=p, subset="finite_RL", RMSE=2., n_total=250, n_used=250)
                       for p in (20, 50, 100)])
    rl.to_csv(dest / "return_level_overall.csv", index=False)
    assert len(collect_metrics(tmp_path, "small", 1, "test")) == 9
    rl.loc[0, "n_used"] = 249
    rl.to_csv(dest / "return_level_overall.csv", index=False)
    with pytest.raises(ValueError, match="excluded return-level"):
        collect_metrics(tmp_path, "small", 1, "test")
