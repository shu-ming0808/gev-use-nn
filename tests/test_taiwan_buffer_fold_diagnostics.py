"""Coordinate-only regression tests; no real NN, GP, downloads or source data."""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from elevation_gp_analysis import _buffered_training_indices
from taiwan_simulation import diagnose_buffer_folds as diagnostics


SEED = 20260929


def grid_fixture():
    """Fifty distinct kilometre coordinates, approximately an ERA5 footprint."""
    row = np.repeat(np.arange(10), 5)
    col = np.tile(np.arange(5), 10)
    return pd.DataFrame({
        "grid_id": [f"G{i:03d}" for i in range(50)],
        "x_km": 200.0 + col * 25.0 + row * 0.6,
        "y_km": 2400.0 + row * 25.0,
    })


def run_geometry(grids, **overrides):
    options = dict(outer_counts=[4], inner_counts=[3], buffers=[0.0, 30.0],
                   min_train=12, seed=SEED)
    options.update(overrides)
    return diagnostics.geometry_diagnostics(grids, **options)


def id_set(value):
    if value is None or pd.isna(value) or value == "":
        return set()
    return set(str(value).split("|"))


def assert_split(row, frame, train_idx, test_idx, baseline_count):
    assert row.n_train == len(train_idx)
    assert row.n_test == len(test_idx)
    assert row.n_removed == baseline_count - len(train_idx)
    assert id_set(row.train_ids) == set(frame.loc[train_idx, "grid_id"])
    assert id_set(row.test_ids) == set(frame.loc[test_idx, "grid_id"])
    assert id_set(row.train_ids).isdisjoint(id_set(row.test_ids))
    assert bool(row.min_train_pass) == (len(train_idx) >= 12)
    if len(train_idx):
        nearest = cdist(frame.loc[train_idx, ["x_km", "y_km"]],
                        frame.loc[test_idx, ["x_km", "y_km"]]).min()
        assert row.nearest_distance_km == pytest.approx(nearest)
        assert nearest > row.buffer_km


def test_nested_geometry_matches_current_cv_exactly():
    frame = grid_fixture()
    before = frame.copy(deep=True)
    result = run_geometry(frame)
    assert {"summary", "splits", "outer_assignments", "fold_quality"}.issubset(result)
    splits = result["splits"]
    outer_labels = KMeans(n_clusters=4, n_init=100, random_state=SEED).fit_predict(
        frame[["x_km", "y_km"]])
    assert len(splits) == 2 * 4 * (1 + 3)
    for buffer in (0.0, 30.0):
        part = splits.loc[splits.buffer_km.eq(buffer)]
        for outer in range(4):
            test = np.flatnonzero(outer_labels == outer)
            candidates = np.flatnonzero(outer_labels != outer)
            retained = _buffered_training_indices(frame, candidates, test, buffer)
            row = part.loc[part.level.eq("outer") & part.outer_fold.eq(outer)].iloc[0]
            assert row.inner_fold == -1
            assert_split(row, frame, retained, test, len(candidates))

            # Refit inner geometry on THIS buffered outer training set, exactly
            # as cheng_spatial_cv.nested_cv does; never use global inner labels.
            train = frame.loc[retained].copy().reset_index(drop=True)
            inner_labels = KMeans(n_clusters=3, n_init=100,
                random_state=SEED + 1000 + outer).fit_predict(train[["x_km", "y_km"]])
            for inner in range(3):
                inner_test = np.flatnonzero(inner_labels == inner)
                inner_candidates = np.flatnonzero(inner_labels != inner)
                inner_train = _buffered_training_indices(
                    train, inner_candidates, inner_test, buffer)
                row = part.loc[part.level.eq("inner") & part.outer_fold.eq(outer)
                               & part.inner_fold.eq(inner)].iloc[0]
                assert_split(row, train, inner_train, inner_test, len(inner_candidates))

        summary = result["summary"].loc[result["summary"].buffer_km.eq(buffer)].iloc[0]
        outer_rows = part.loc[part.level.eq("outer")]
        inner_rows = part.loc[part.level.eq("inner")]
        assert summary.outer_train_min == outer_rows.n_train.min()
        assert summary.outer_train_max == outer_rows.n_train.max()
        assert summary.inner_train_min == inner_rows.n_train.min()
        assert summary.inner_train_max == inner_rows.n_train.max()
        assert summary.outer_test_min == outer_rows.n_test.min()
        assert summary.inner_test_min == inner_rows.n_test.min()
        assert summary.outer_failed_splits == int((outer_rows.n_train < 12).sum())
        assert summary.inner_failed_splits == int((inner_rows.n_train < 12).sum())
        assert bool(summary.all_folds_meet_min_train) == bool(part.min_train_pass.all())
        assert summary.actual_outer_min_distance_km == pytest.approx(
            outer_rows.nearest_distance_km.min())
        assert summary.actual_inner_min_distance_km == pytest.approx(
            inner_rows.nearest_distance_km.min())
    pd.testing.assert_frame_equal(frame, before)


def test_impossible_buffer_remains_failed_not_silently_relaxed():
    result = run_geometry(grid_fixture(), buffers=[10000.0], min_train=20)
    summary = result["summary"].iloc[0]
    assert summary.buffer_km == 10000.0
    assert not bool(summary.all_folds_meet_min_train)
    assert summary.outer_failed_splits == 4
    assert summary.outer_train_min == 0
    outer = result["splits"].loc[result["splits"].level.eq("outer")]
    assert outer.n_train.eq(0).all()
    assert all(not id_set(ids) for ids in outer.train_ids)


def test_same_seed_is_deterministic_and_summary_covers_requested_grid():
    options = dict(outer_counts=[3, 4], inner_counts=[2, 3], buffers=[0.0, 30.0])
    first = run_geometry(grid_fixture(), **options)
    second = run_geometry(grid_fixture(), **options)
    assert len(first["summary"]) == 2 * 2 * 2
    assert not first["summary"].duplicated(["outer_folds", "inner_folds", "buffer_km"]).any()
    for name in ("summary", "splits", "outer_assignments", "fold_quality"):
        pd.testing.assert_frame_equal(first[name], second[name])


@pytest.mark.parametrize("problem", ["duplicate_id", "missing_id", "nonfinite", "duplicate_xy"])
def test_invalid_grid_contract_is_rejected(problem):
    grids = grid_fixture()
    if problem == "duplicate_id":
        grids.loc[1, "grid_id"] = grids.loc[0, "grid_id"]
    elif problem == "missing_id":
        grids.loc[1, "grid_id"] = None
    elif problem == "nonfinite":
        grids.loc[1, "x_km"] = np.nan
    else:
        grids.loc[1, ["x_km", "y_km"]] = grids.loc[0, ["x_km", "y_km"]].to_numpy()
    with pytest.raises(ValueError):
        run_geometry(grids)


@pytest.mark.parametrize("options", [
    {"outer_counts": [1]}, {"outer_counts": [51]}, {"outer_counts": []},
    {"inner_counts": [1]}, {"inner_counts": []},
    {"buffers": [-1.0]}, {"buffers": [np.nan]}, {"buffers": [np.inf]},
    {"buffers": []}, {"min_train": 1},
])
def test_invalid_settings_are_rejected(options):
    with pytest.raises(ValueError):
        run_geometry(grid_fixture(), **options)


def test_elevation_residuals_remove_intercept_and_linear_elevation():
    elevation = np.linspace(100.0, 3100.0, 50)
    slopes = np.array([0.003, 0.0001, -0.0002, 0.00001, 0.00002])
    intercepts = np.array([30.0, 0.2, 0.7, 0.01, -0.1])
    coefficients = pd.DataFrame(intercepts + elevation[:, None] * slopes,
                                columns=diagnostics.COEFFICIENTS,
                                index=np.arange(100, 150))
    residuals = diagnostics.elevation_residuals(coefficients, elevation)
    np.testing.assert_allclose(residuals, 0.0, atol=1e-12)
    assert list(residuals.columns) == list(diagnostics.COEFFICIENTS)
    assert residuals.index.equals(coefficients.index)


def test_elevation_residuals_preserve_residual_units_and_are_affine_invariant():
    elevation = np.linspace(100.0, 3100.0, 50)
    wave = np.sin(np.arange(50) * 0.7)
    amplitudes = np.array([1.0, 0.1, 0.01, 0.001, 0.0001])
    coefficients = pd.DataFrame(25.0 + elevation[:, None] * amplitudes / 1000.0
                                + wave[:, None] * amplitudes,
                                columns=diagnostics.COEFFICIENTS)
    residuals = diagnostics.elevation_residuals(coefficients, elevation)
    converted = diagnostics.elevation_residuals(coefficients, elevation / 1000.0 + 17.0)
    assert np.isfinite(residuals.to_numpy()).all()
    np.testing.assert_allclose(residuals, converted, atol=1e-12)
    np.testing.assert_allclose(residuals.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose((elevation - elevation.mean()) @ residuals.to_numpy(),
                               0.0, atol=1e-8)
    # Residual coefficients keep their own units; they are NOT all normalized
    # to unit variance before the variogram helper sees them.
    np.testing.assert_allclose(residuals.std().to_numpy() / residuals.iloc[:, 0].std(),
                               amplitudes, atol=1e-12)


def test_variogram_training_rows_exclude_outer_test_and_do_not_leak(monkeypatch):
    from taiwan_simulation import buffer_variograms

    grids = grid_fixture()
    labels = np.arange(50) % 3
    assignments = pd.DataFrame({"grid_id": grids.grid_id, "outer_folds": 3,
                                "outer_fold": labels}).sample(frac=1, random_state=7)
    elevation = np.linspace(100.0, 3100.0, 50)
    wave = np.sin(np.arange(50) * 0.7)
    coefficients = pd.DataFrame(
        20.0 + elevation[:, None] * np.arange(1, 6) / 1000.0
        + wave[:, None] * np.linspace(0.1, 1.0, 5), columns=diagnostics.COEFFICIENTS)
    calls = []

    def record(xy, part, **options):
        calls.append((np.asarray(xy).copy(), part.copy(deep=True), options.copy()))
        simple = pd.DataFrame({"coefficient": list(part.columns)})
        return {name: simple.copy() for name in ("bins", "fits", "summary")}

    monkeypatch.setattr(buffer_variograms, "diagnose_variograms", record)
    result = diagnostics.variogram_diagnostics(grids, coefficients, elevation, assignments)
    first_calls = calls[:]
    assert len(first_calls) == 2 + 3
    for fold in range(3):
        expected_index = np.flatnonzero(labels != fold)
        xy, residuals, options = first_calls[2 + fold]
        np.testing.assert_array_equal(residuals.index.to_numpy(), expected_index)
        np.testing.assert_array_equal(xy, grids.iloc[expected_index][["x_km", "y_km"]])
        assert options == dict(n_bins=8, max_distance_fraction=0.6, min_pairs=10)
        assert not set(residuals.index).intersection(np.flatnonzero(labels == fold))
    assert set(result["summary"].kind) == {
        "full_raw_exploratory", "full_elevation_residual_exploratory",
        "outer_training_elevation_residual"}

    # Change ONLY fold zero's held-out responses. Full-data exploratory
    # diagnostics must change, while fold zero's training-only residuals must
    # remain exactly the same (including the fitted elevation mean effect).
    changed = coefficients.copy()
    heldout = labels == 0
    changed.loc[heldout] += np.arange(heldout.sum())[:, None] * 100.0 + 50.0
    calls.clear()
    diagnostics.variogram_diagnostics(grids, changed, elevation, assignments)
    assert not calls[0][1].equals(first_calls[0][1])
    assert not calls[1][1].equals(first_calls[1][1])
    pd.testing.assert_frame_equal(calls[2][1], first_calls[2][1], check_exact=True)
    np.testing.assert_array_equal(calls[2][0], first_calls[2][0])
