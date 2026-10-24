"""Five-coefficient nested buffered spatial CV using the existing spatial FFS.

This is conditional on retaining both temporal slopes. It does NOT choose a
time structure, infer significance or use AIC. Real-data errors are agreement
with frozen NN estimates, not errors against true climate parameters.
"""
from __future__ import annotations

# Keep direct-script execution independent of the working directory.
import sys as _sys
from pathlib import Path as _SourcePath
_SOURCE_ROOT = _SourcePath(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from research_workflows import COEFFICIENTS, TEMPORAL_POLICY
from spatial_coordinates import project_lonlat_to_twd97_km, project_lonlat_to_km
from spatial_predictor_selection import spatial_forward_selection, KERNEL_CANDIDATES
from elevation_gp_analysis import _buffered_training_indices, sample_indices
from land_cover_gp_analysis import fit_gp_with_covariates, predict_gp_with_covariates
from train_cheng_nn import atomic_csv, atomic_json, file_sha256

TARGET_COLUMNS = {name: name for name in COEFFICIENTS}


def join_predictors(responses, predictors, candidate_groups, *, coordinate_crs='EPSG:3826'):
    """One-to-one, exact GRID join; never reuse 0.05-degree TCCIP by row order."""
    if not candidate_groups or any(not cols for cols in candidate_groups.values()):
        raise ValueError('Specify nonempty, predeclared predictor groups.')
    columns = [c for cols in candidate_groups.values() for c in cols]
    reserved = {*COEFFICIENTS, 'grid_id', 'station', 'x_km', 'y_km', 'latitude',
                'longitude', 'longitude_180', 'spatial_fold'}
    if len(set(columns)) != len(columns) or set(columns) & reserved:
        raise ValueError('Predictor groups must be disjoint and must not contain responses/identifiers.')
    for frame in (responses, predictors):
        if 'grid_id' not in frame or frame.grid_id.isna().any() or frame.grid_id.duplicated().any():
            raise ValueError('Unique, nonmissing grid_id is required.')
    required = ['grid_id','longitude_180','latitude','x_km','y_km',*COEFFICIENTS]
    if set(columns) & set(responses.columns):
        raise ValueError('Predictor/response column collision.')
    result = responses[required].merge(predictors[['grid_id',*columns]],
        how='left', on='grid_id', validate='one_to_one', sort=False)
    numeric = result.drop(columns=['grid_id']).to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError('Incomplete/nonfinite predictors or responses; no silent GRID deletion.')
    expected = (project_lonlat_to_twd97_km(result.longitude_180, result.latitude)
        if coordinate_crs == 'EPSG:3826' else
        project_lonlat_to_km(result.longitude_180, result.latitude, target_crs=coordinate_crs))
    if not np.allclose(result[['x_km','y_km']], expected, atol=1e-6, rtol=0):
        raise ValueError(f'Coordinate contract must be {coordinate_crs} in km.')
    result['station'] = result.grid_id
    return result.reset_index(drop=True)


def nested_cv(data, *, candidate_groups, buffers_km, outer_folds, inner_folds,
              min_train, max_train, seed=20260929, n_jobs=1, n_restarts=0,
              min_relative_improvement=.01, maximum_allowed_vif=5., max_steps=None,
              kernel_candidates=KERNEL_CANDIDATES, buffer_distance_mode='projected',
              progress=None, gp_options=None):
    """Selection is restricted to each target's buffered outer-training GRID.

    Buffers/fold counts are explicit research choices, not copied from the old
    1,385-point Taiwan dataset. Preflight ALL folds before any GP fitting.
    """
    if (set(buffers_km) != set(COEFFICIENTS)
            or any(v is None or not np.isfinite(v) or v < 0 for v in buffers_km.values())):
        raise ValueError('Explicit finite nonnegative buffers for all five coefficients are required.')
    if outer_folds < 2 or inner_folds < 2 or min_train < 2 or max_train < min_train:
        raise ValueError('Invalid fold counts or training retention limits.')
    frame = data.reset_index(drop=True).copy()
    buffer_options = ({'distance_mode': buffer_distance_mode} if buffer_distance_mode != 'projected' else {})
    xy = frame[['x_km','y_km']].to_numpy(float)
    if not np.isfinite(xy).all() or len(np.unique(xy, axis=0)) != len(xy) or len(xy) < outer_folds:
        raise ValueError('Insufficient or duplicate/nonfinite coordinates.')
    frame['outer_fold'] = KMeans(n_clusters=outer_folds, n_init=100, random_state=seed).fit_predict(xy)
    plan = []
    for fold in range(outer_folds):
        test_idx = frame.index[frame.outer_fold.eq(fold)].to_numpy()
        candidates = frame.index[frame.outer_fold.ne(fold)].to_numpy()
        for target in COEFFICIENTS:
            train_idx = _buffered_training_indices(frame, candidates, test_idx, float(buffers_km[target]), **buffer_options)
            train_idx = sample_indices(train_idx, max_train, seed+fold)
            if len(train_idx) < max(min_train, inner_folds):
                raise ValueError(f'{target}/outer {fold}: only {len(train_idx)} training GRID after buffer.')
            train = frame.loc[train_idx].copy().reset_index(drop=True)
            train['spatial_fold'] = KMeans(n_clusters=inner_folds, n_init=100,
                random_state=seed+1000+fold).fit_predict(train[['x_km','y_km']])
            # Validate inner retention before running expensive model selection.
            for inner in range(inner_folds):
                itest = train.index[train.spatial_fold.eq(inner)].to_numpy()
                itrain = train.index[train.spatial_fold.ne(inner)].to_numpy()
                retained = _buffered_training_indices(train, itrain, itest, float(buffers_km[target]), **buffer_options)
                if len(retained) < min_train:
                    raise ValueError(f'{target}/outer {fold}/inner {inner}: insufficient retained GRID ({len(retained)}).')
            plan.append((fold, target, train, frame.loc[test_idx].copy(), len(candidates)))
    if progress is not None:
        progress('preflight_complete', dict(n_cases=len(plan)), None)
    all_oof, selections, trials, paths = [], [], [], []
    for fold, target, train, test, n_candidates in plan:
        if progress is not None:
            progress('case_started', dict(outer_fold=fold, coefficient=target,
                n_train=len(train), n_test=len(test), completed_cases=len(selections), total_cases=len(plan)), None)
        trial, path, winner = spatial_forward_selection(
            train, target, target_columns=TARGET_COLUMNS, candidate_groups=candidate_groups,
            buffer_km=float(buffers_km[target]), n_folds=inner_folds, min_train=min_train,
            max_train=max_train, n_jobs=n_jobs, n_restarts=n_restarts, random_state=seed+fold,
            min_relative_improvement=min_relative_improvement, maximum_allowed_vif=maximum_allowed_vif,
            max_steps=max_steps, kernel_candidates=tuple(tuple(k) for k in kernel_candidates),
            **({'gp_options':gp_options} if gp_options else {}),
            **({'buffer_distance_mode':buffer_distance_mode} if buffer_distance_mode != 'projected' else {}))
        cols = winner['predictor_names']
        fitted = fit_gp_with_covariates(train[['x_km','y_km']].to_numpy(), train[cols].to_numpy(),
            train[target].to_numpy(), predictor_names=cols, kernel_name=winner['kernel'],
            nu=winner['nu'], n_restarts=n_restarts, seed=seed+fold, **(gp_options or {}))
        pred = predict_gp_with_covariates(fitted, test[['x_km','y_km']].to_numpy(), test[cols].to_numpy())
        if not np.isfinite(pred).all():
            raise FloatingPointError('Nonfinite outer prediction; no omitted cases.')
        all_oof.append(pd.DataFrame(dict(grid_id=test.grid_id.to_numpy(), outer_fold=fold,
            coefficient=target, reference_nn=test[target].to_numpy(), prediction=pred)))
        selections.append(dict(outer_fold=fold, coefficient=target, predictors='+'.join(cols) or 'intercept',
            kernel=winner['kernel'], nu=winner['nu'], inner_selection_rmse=winner['RMSE'],
            n_outer_candidates=n_candidates, n_retained_train=len(train), n_test=len(test),
            buffer_km=buffers_km[target], optimizer_success=fitted['optimizer_success']))
        if 'kernel' in fitted:
            spatial_kernel=fitted['kernel'].k1.k2
            length=float(spatial_kernel.length_scale)
            low,high=spatial_kernel.length_scale_bounds
            selections[-1].update(fitted_length_scale_km=length,
                length_scale_near_bound=bool(np.isclose(np.log(length),np.log(low),atol=1e-4)
                    or np.isclose(np.log(length),np.log(high),atol=1e-4)),
                optimizer_message=fitted.get('optimizer_message',''))
        trials.append(trial.assign(outer_fold=fold))
        paths.append(path.assign(outer_fold=fold))
        if progress is not None:
            progress('case_completed', dict(outer_fold=fold, coefficient=target,
                completed_cases=len(selections), total_cases=len(plan)),
                dict(oof=all_oof[-1], selections=pd.DataFrame([selections[-1]]),
                     trials=trials[-1], paths=paths[-1]))
    oof = pd.concat(all_oof, ignore_index=True)
    rows = []
    for coefficient, part in oof.groupby('coefficient', sort=False):
        error = part.prediction.to_numpy()-part.reference_nn.to_numpy()
        rows.append(dict(coefficient=coefficient, n_grid=len(part),
            RMSE_vs_NN_reference=float(np.sqrt(np.mean(error**2))),
            MAE_vs_NN_reference=float(np.mean(np.abs(error))), Bias_vs_NN_reference=float(error.mean())))
    return dict(oof=oof, metrics=pd.DataFrame(rows), selections=pd.DataFrame(selections),
                trials=pd.concat(trials, ignore_index=True), paths=pd.concat(paths, ignore_index=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--predictors', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Use a new output directory.')
    report_path = args.prepared/'run_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('completed_stage') != 'taiwan_preparation_and_nn_only'
            or report.get('coordinate_crs') != 'EPSG:3826'):
        raise ValueError('Expected a completed Cheng Taiwan preparation run.')
    response_path = args.prepared/'spatial_responses.csv'
    if file_sha256(response_path) != report['output_sha256']['spatial_responses.csv']:
        raise ValueError('Prepared responses have changed.')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config.pop('temporal_policy', None) != TEMPORAL_POLICY:
        raise ValueError('Only explicit full_no_selection is supported; temporal selection is disabled.')
    data = join_predictors(pd.read_csv(response_path), pd.read_csv(args.predictors), config['candidate_groups'])
    args.output.mkdir(parents=True)
    atomic_json(args.output/'run_status.json', dict(status='running'))
    try:
        results = nested_cv(data, **config)
        for name, table in results.items():
            atomic_csv(args.output/f'{name}.csv', table)
        atomic_json(args.output/'run_report.json', dict(status='completed',
            completed_stage='five_coefficient_spatial_cv_conditional_on_full_time_structure',
            reference='frozen NN estimates, NOT known climate truth',
            temporal_policy=TEMPORAL_POLICY, selection='FFS; inner buffered CV RMSE; no AIC',
            config=config, input_sha256={str(p):file_sha256(p) for p in
                (report_path, response_path, args.predictors, args.config)},
            warning='Adaptive FFS trial p values are exploratory, not confirmatory trend tests.'))
        atomic_json(args.output/'run_status.json', dict(status='completed'))
    except BaseException as error:
        atomic_json(args.output/'run_status.json', dict(status='failed', error_type=type(error).__name__))
        raise


if __name__ == '__main__':
    main()
