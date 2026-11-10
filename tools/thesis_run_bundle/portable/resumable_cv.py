"""Checkpointed execution of the pinned five-coefficient nested Spatial CV.

The scientific fit/selection calls and fold preflight mirror the frozen
``taiwan_simulation.cheng_spatial_cv.nested_cv``. Only complete, hashed
fold-by-coefficient outputs from a previous attempt may be reused. An
interrupted current case is recomputed; held-out observations never enter fit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import identity, read, sha, write
from taiwan_simulation import cheng_spatial_cv as cv
from research_workflows import COEFFICIENTS


NAMES = ('oof', 'selections', 'trials', 'paths')
MARKER = 'checkpoint_complete.json'


def _checkpoint_path(attempt, fold, target):
    return Path(attempt) / 'checkpoints' / f'outer_{fold}_{target}'


def _read_case(attempts, fold, target, test, train, token):
    """Return only a complete same-protocol case, newest attempt first."""
    for attempt in attempts:
        folder = _checkpoint_path(attempt, fold, target)
        marker_path = folder / MARKER
        if not marker_path.is_file():
            continue
        try:
            marker = read(marker_path)
            if (marker.get('token') != token or marker.get('outer_fold') != fold
                    or marker.get('coefficient') != target
                    or marker.get('n_train') != len(train)
                    or marker.get('n_test') != len(test)
                    or set(marker.get('files', {})) != set(NAMES)):
                continue
            if any(not (folder / f'{name}.csv').is_file()
                   or sha(folder / f'{name}.csv') != marker['files'][name]
                   for name in NAMES):
                continue
            tables = {name: pd.read_csv(folder / f'{name}.csv', dtype={'grid_id': str})
                      for name in NAMES}
            oof, selections = tables['oof'], tables['selections']
            if (len(oof) != len(test) or oof.grid_id.tolist() != test.grid_id.tolist()
                    or not oof.coefficient.eq(target).all()
                    or not oof.outer_fold.eq(fold).all()
                    or not np.isfinite(oof.prediction.to_numpy(float)).all()
                    or not np.allclose(oof.reference_nn.to_numpy(float),
                                       test[target].to_numpy(float), atol=1e-10, rtol=0)
                    or len(selections) != 1
                    or selections.iloc[0]['coefficient'] != target
                    or int(selections.iloc[0]['outer_fold']) != fold
                    or int(selections.iloc[0]['n_retained_train']) != len(train)
                    or int(selections.iloc[0]['n_test']) != len(test)
                    or tables['trials'].empty or tables['paths'].empty):
                continue
            return tables
        except (OSError, ValueError, KeyError, TypeError, AttributeError,
                pd.errors.ParserError):
            # A torn marker/file is not a checkpoint; recompute this case.
            continue
    return None


def _save_marker(output, fold, target, train, test, token):
    folder = _checkpoint_path(output, fold, target)
    files = {name: sha(folder / f'{name}.csv') for name in NAMES}
    write(folder / MARKER, dict(version=1, token=token,
        outer_fold=fold, coefficient=target, n_train=len(train),
        n_test=len(test), files=files))


def nested_cv(data, *, resume_attempts, checkpoint_output, checkpoint_token,
              candidate_groups, buffers_km, outer_folds, inner_folds,
              min_train, max_train, seed=20260929, n_jobs=1, n_restarts=0,
              min_relative_improvement=.01, maximum_allowed_vif=5., max_steps=None,
              kernel_candidates=cv.KERNEL_CANDIDATES, buffer_distance_mode='projected',
              progress=None, gp_options=None):
    if (set(buffers_km) != set(COEFFICIENTS)
            or any(v is None or not np.isfinite(v) or v < 0 for v in buffers_km.values())):
        raise ValueError('Explicit finite nonnegative buffers for five coefficients required')
    if outer_folds < 2 or inner_folds < 2 or min_train < 2 or max_train < min_train:
        raise ValueError('Invalid fold or training settings')
    frame = data.reset_index(drop=True).copy()
    buffer_options = ({'distance_mode': buffer_distance_mode}
                      if buffer_distance_mode != 'projected' else {})
    xy = frame[['x_km', 'y_km']].to_numpy(float)
    if not np.isfinite(xy).all() or len(np.unique(xy, axis=0)) != len(xy) or len(xy) < outer_folds:
        raise ValueError('Insufficient or duplicate/nonfinite coordinates')
    frame['outer_fold'] = cv.KMeans(n_clusters=outer_folds, n_init=100,
                                    random_state=seed).fit_predict(xy)
    plan = []
    for fold in range(outer_folds):
        test_idx = frame.index[frame.outer_fold.eq(fold)].to_numpy()
        candidates = frame.index[frame.outer_fold.ne(fold)].to_numpy()
        for target in COEFFICIENTS:
            train_idx = cv._buffered_training_indices(
                frame, candidates, test_idx, float(buffers_km[target]), **buffer_options)
            train_idx = cv.sample_indices(train_idx, max_train, seed + fold)
            if len(train_idx) < max(min_train, inner_folds):
                raise ValueError(f'{target}/outer {fold}: insufficient training GRID')
            train = frame.loc[train_idx].copy().reset_index(drop=True)
            train['spatial_fold'] = cv.KMeans(n_clusters=inner_folds, n_init=100,
                random_state=seed + 1000 + fold).fit_predict(train[['x_km', 'y_km']])
            for inner in range(inner_folds):
                itest = train.index[train.spatial_fold.eq(inner)].to_numpy()
                itrain = train.index[train.spatial_fold.ne(inner)].to_numpy()
                retained = cv._buffered_training_indices(
                    train, itrain, itest, float(buffers_km[target]), **buffer_options)
                if len(retained) < min_train:
                    raise ValueError(f'{target}/outer {fold}/inner {inner}: insufficient retained GRID')
            plan.append((fold, target, train, frame.loc[test_idx].copy(), len(candidates)))
    if progress is not None:
        progress('preflight_complete', dict(n_cases=len(plan)), None)

    all_oof, selections, trials, paths = [], [], [], []
    for fold, target, train, test, n_candidates in plan:
        if progress is not None:
            progress('case_started', dict(outer_fold=fold, coefficient=target,
                n_train=len(train), n_test=len(test), completed_cases=len(selections),
                total_cases=len(plan)), None)
        cached = _read_case(resume_attempts, fold, target, test, train, checkpoint_token)
        if cached is not None:
            all_oof.append(cached['oof'])
            selections.append(cached['selections'].iloc[0].to_dict())
            trials.append(cached['trials'])
            paths.append(cached['paths'])
            print(f'reused verified checkpoint outer {fold} {target}', flush=True)
        else:
            trial, path, winner = cv.spatial_forward_selection(
                train, target, target_columns=cv.TARGET_COLUMNS,
                candidate_groups=candidate_groups, buffer_km=float(buffers_km[target]),
                n_folds=inner_folds, min_train=min_train, max_train=max_train,
                n_jobs=n_jobs, n_restarts=n_restarts, random_state=seed + fold,
                min_relative_improvement=min_relative_improvement,
                maximum_allowed_vif=maximum_allowed_vif, max_steps=max_steps,
                kernel_candidates=tuple(tuple(k) for k in kernel_candidates),
                **({'gp_options': gp_options} if gp_options else {}),
                **({'buffer_distance_mode': buffer_distance_mode}
                   if buffer_distance_mode != 'projected' else {}))
            cols = winner['predictor_names']
            from gp_execution import refit_context
            with refit_context(f'outer_{fold}_{target}_refit', n_train=len(train), n_test=len(test)):
                fitted = cv.fit_gp_with_covariates(
                    train[['x_km', 'y_km']].to_numpy(), train[cols].to_numpy(),
                    train[target].to_numpy(), predictor_names=cols,
                    kernel_name=winner['kernel'], nu=winner['nu'],
                    n_restarts=n_restarts, seed=seed + fold, **(gp_options or {}))
                pred = cv.predict_gp_with_covariates(
                    fitted, test[['x_km', 'y_km']].to_numpy(), test[cols].to_numpy())
            if not np.isfinite(pred).all():
                raise FloatingPointError('Nonfinite outer prediction')
            all_oof.append(pd.DataFrame(dict(
                grid_id=test.grid_id.to_numpy(), outer_fold=fold,
                coefficient=target, reference_nn=test[target].to_numpy(), prediction=pred)))
            selected = dict(outer_fold=fold, coefficient=target,
                predictors='+'.join(cols) or 'intercept', kernel=winner['kernel'],
                nu=winner['nu'], inner_selection_rmse=winner['RMSE'],
                n_outer_candidates=n_candidates, n_retained_train=len(train),
                n_test=len(test), buffer_km=buffers_km[target],
                optimizer_success=fitted['optimizer_success'])
            if 'kernel' in fitted:
                spatial_kernel = fitted['kernel'].k1.k2
                length = float(spatial_kernel.length_scale)
                low, high = spatial_kernel.length_scale_bounds
                selected.update(fitted_length_scale_km=length,
                    length_scale_near_bound=bool(
                        np.isclose(np.log(length), np.log(low), atol=1e-4)
                        or np.isclose(np.log(length), np.log(high), atol=1e-4)),
                    optimizer_message=fitted.get('optimizer_message', ''))
            del fitted
            selections.append(selected)
            trials.append(trial.assign(outer_fold=fold))
            paths.append(path.assign(outer_fold=fold))
        if progress is not None:
            progress('case_completed', dict(outer_fold=fold, coefficient=target,
                completed_cases=len(selections), total_cases=len(plan)),
                dict(oof=all_oof[-1], selections=pd.DataFrame([selections[-1]]),
                     trials=trials[-1], paths=paths[-1]))
            _save_marker(checkpoint_output, fold, target, train, test, checkpoint_token)

    oof = pd.concat(all_oof, ignore_index=True)
    rows = []
    for coefficient, part in oof.groupby('coefficient', sort=False):
        error = part.prediction.to_numpy() - part.reference_nn.to_numpy()
        rows.append(dict(coefficient=coefficient, n_grid=len(part),
            RMSE_vs_NN_reference=float(np.sqrt(np.mean(error ** 2))),
            MAE_vs_NN_reference=float(np.mean(np.abs(error))),
            Bias_vs_NN_reference=float(error.mean())))
    return dict(oof=oof, metrics=pd.DataFrame(rows),
        selections=pd.DataFrame(selections), trials=pd.concat(trials, ignore_index=True),
        paths=pd.concat(paths, ignore_index=True))


def checkpointed_execute(original_execute):
    """Patch only this process's extension run; frozen bundle sources stay intact."""
    def execute(*args, **kwargs):
        output = Path(args[5])
        token = identity(dict(method_protocol=kwargs['protocol_id'],
                              resume_code_sha256=sha(Path(__file__))))
        older = sorted(output.parent.glob('attempt_*'), reverse=True)
        original_nested = cv.nested_cv

        def patched(data, **settings):
            return nested_cv(data, resume_attempts=older,
                checkpoint_output=output, checkpoint_token=token, **settings)

        cv.nested_cv = patched
        try:
            return original_execute(*args, **kwargs)
        finally:
            cv.nested_cv = original_nested
    return execute
