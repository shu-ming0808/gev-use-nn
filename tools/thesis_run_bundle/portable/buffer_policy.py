"""Return an auditable regional buffer choice from training-only diagnostics.

This policy is exploratory. An empirical variogram range is not an
independence proof, and geometry-only fallback is not a range estimate.
No outer held-out coefficient, prediction error, or CRPS enters the choice.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from research_workflows import COEFFICIENTS


FOLD_PREFERENCE = ((5, 3), (4, 3), (6, 3), (5, 5))
RANGE_FOLD_FRACTION = .6
MIN_RETAINED_FRACTION = .15
FALLBACK_KM = 500.


def _supported_range_by_coefficient(fits: pd.DataFrame, outer: int) -> dict[str, float]:
    required = {'kind', 'outer_folds', 'excluded_outer_fold', 'coefficient',
                'range_supported', 'effective_range_km', 'weighted_rmse'}
    if not required.issubset(fits.columns):
        raise ValueError(f'Variogram fits missing: {sorted(required - set(fits.columns))}')
    part = fits.loc[(fits.kind == 'outer_training_elevation_residual') &
                    (fits.outer_folds == outer) & (fits.range_supported == True)].copy()
    part = part.loc[np.isfinite(part.effective_range_km) &
                    np.isfinite(part.weighted_rmse) & part.effective_range_km.gt(0)]
    # Multiple variogram families are alternatives, not independent folds.
    part = part.sort_values('weighted_rmse').drop_duplicates(
        ['excluded_outer_fold', 'coefficient'])
    minimum = math.ceil(RANGE_FOLD_FRACTION * outer)
    result = {}
    for name in COEFFICIENTS:
        rows = part.loc[part.coefficient.eq(name)]
        if rows.excluded_outer_fold.nunique() >= minimum:
            result[name] = float(rows.effective_range_km.median())
    return result


def choose_region_buffer(summary: pd.DataFrame, fits: pd.DataFrame,
                         *, min_train: int = 100) -> dict:
    """Return buffer_km and fold counts; caller feeds these to GP settings.

    Choose the smallest feasible diagnosed radius at least as large as the
    largest coefficient median supported range. If any coefficient's range is
    unidentified, use the feasible radius nearest the declared 500-km
    fallback. Never silently call that fallback an estimated correlation range.
    """
    required = {'outer_folds', 'inner_folds', 'buffer_km', 'inner_train_min',
                'outer_train_min', 'all_folds_meet_min_train', 'min_retained_fraction'}
    if not required.issubset(summary.columns):
        raise ValueError(f'Geometry summary missing: {sorted(required - set(summary.columns))}')
    if min_train < 2:
        raise ValueError('min_train must be at least 2')
    for outer, inner in FOLD_PREFERENCE:
        rows = summary.loc[(summary.outer_folds == outer) &
                           (summary.inner_folds == inner) &
                           (summary.all_folds_meet_min_train == True) &
                           summary.inner_train_min.ge(min_train) &
                           summary.outer_train_min.ge(min_train) &
                           summary.buffer_km.ge(0) & np.isfinite(summary.buffer_km)].copy()
        if rows.empty:
            continue
        retained = rows.loc[rows.min_retained_fraction.ge(MIN_RETAINED_FRACTION)]
        retention_relaxed = retained.empty
        if not retention_relaxed:
            rows = retained
        ranges = _supported_range_by_coefficient(fits, outer)
        candidates = sorted(set(float(x) for x in rows.buffer_km))
        if set(ranges) == set(COEFFICIENTS):
            target = max(ranges.values())
            above = [radius for radius in candidates if radius >= target]
            choice = min(above) if above else max(candidates)
            basis = 'supported_training_residual_ranges' if above else 'range_exceeds_feasible_grid'
        else:
            target = None
            choice = min(candidates, key=lambda radius: (abs(radius - FALLBACK_KM), radius))
            basis = 'range_unidentified_geometry_fallback'
        row = rows.loc[rows.buffer_km.eq(choice)].iloc[0]
        return dict(buffer_km=float(choice), outer_folds=int(outer), inner_folds=int(inner),
                    min_train=int(min_train), basis=basis, estimated_target_km=target,
                    supported_median_ranges_km=ranges,
                    missing_range_coefficients=sorted(set(COEFFICIENTS) - set(ranges)),
                    retained_fraction=float(row.min_retained_fraction),
                    retention_threshold_relaxed=bool(retention_relaxed),
                    candidate_buffers_km=candidates,
                    caveat='Training-only descriptive range; not independence proof or test-error optimization.')
    raise ValueError('No diagnosed fold/buffer combination retains the minimum training grids')
