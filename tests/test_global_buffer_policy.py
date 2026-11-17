"""Pure policy tests; these do not run a GP or touch EAS data."""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'tools' / 'thesis_run_bundle')]

from portable.buffer_policy import choose_region_buffer
from research_workflows import COEFFICIENTS


def geometry():
    return pd.DataFrame(dict(outer_folds=[5] * 3, inner_folds=[3] * 3,
        buffer_km=[300., 500., 750.], inner_train_min=[300, 200, 90],
        outer_train_min=[400, 300, 120], all_folds_meet_min_train=[True, True, False],
        min_retained_fraction=[.45, .32, .10]))


def fitted_ranges(distance):
    return pd.DataFrame([dict(kind='outer_training_elevation_residual',
        outer_folds=5, excluded_outer_fold=fold, coefficient=name,
        range_supported=True, effective_range_km=distance,
        weighted_rmse=1.) for fold in range(5) for name in COEFFICIENTS])


def test_supported_training_range_returns_smallest_feasible_radius():
    result = choose_region_buffer(geometry(), fitted_ranges(380.))
    assert result['buffer_km'] == 500.
    assert result['basis'] == 'supported_training_residual_ranges'
    assert result['estimated_target_km'] == 380.


def test_unidentified_range_is_explicit_geometry_fallback():
    result = choose_region_buffer(geometry(), fitted_ranges(380.).query('coefficient != "xi0"'))
    assert result['buffer_km'] == 500.
    assert result['basis'] == 'range_unidentified_geometry_fallback'
    assert result['missing_range_coefficients'] == ['xi0']


def test_range_larger_than_feasible_is_flagged():
    result = choose_region_buffer(geometry(), fitted_ranges(700.))
    assert result['buffer_km'] == 500.
    assert result['basis'] == 'range_exceeds_feasible_grid'


def test_no_feasible_buffer_blocks_region():
    data = geometry().assign(inner_train_min=9)
    with pytest.raises(ValueError, match='No diagnosed'):
        choose_region_buffer(data, fitted_ranges(300.))
