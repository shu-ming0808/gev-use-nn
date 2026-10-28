"""Add an immutable EAS terrain overlay without changing frozen bundle inputs.

The expanded_v2 profile appends four coarse-terrain columns. The separately
versioned expanded_latitude_v3 profile also adds geographic latitude as an
optional linear-mean candidate. Weather summaries, original terrain, grid
membership, NN observations and spatial-CV settings stay intact.
The cache identity excludes drive letters and execution settings, so moving a
verified bundle between D: and F: does not invalidate the derived features.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .common import ROOT, identity, lock, read, sha, write
from . import workflow
from parameters import regional_terrain
from train_cheng_nn import atomic_csv


PROFILE = 'expanded_v2'
LATITUDE_PROFILE = 'expanded_latitude_v3'
PROFILES = (PROFILE, LATITUDE_PROFILE)
OUTPUT_FILES = ('terrain_extra.csv', 'grid_predictors.csv', 'candidate_groups.json')


def frame_fingerprint(frame):
    """Stable across drive letters/newline conventions, sensitive to order/type."""
    content = frame.to_csv(index=False, float_format='%.17g', lineterminator='\n')
    return identity(dict(columns=list(frame.columns), dtypes=[str(t) for t in frame.dtypes],
                         n_rows=len(frame), sha256=hashlib.sha256(content.encode('utf-8')).hexdigest()))


def extra_candidate_groups(profile=PROFILE):
    """Predeclared candidates; latitude is selected by FFS, not forced in."""
    if profile not in PROFILES:
        raise ValueError(f'Unsupported EAS predictor profile: {profile}')
    groups = regional_terrain.extra_candidate_groups(PROFILE)
    if profile == LATITUDE_PROFILE:
        groups['latitude'] = ['latitude_deg']
    return groups


def _extra_columns(profile=PROFILE):
    return tuple(column for group in extra_candidate_groups(profile).values()
                 for column in group)


def _assemble(inputs, extra, profile=PROFILE):
    values, grids, predictors, config, crs = inputs
    columns = _extra_columns(profile)
    expected = {'grid_id', *columns}
    if set(extra.columns) != expected or len(extra.columns) != len(expected):
        raise ValueError('Terrain overlay must contain exactly GRID IDs and the declared profile columns.')
    if (extra.grid_id.isna().any() or extra.grid_id.duplicated().any()
            or len(extra) != len(grids) or set(extra.grid_id) != set(grids.grid_id)):
        raise ValueError('Terrain overlay GRID coverage differs; no deletion or imputation allowed.')
    if not np.isfinite(extra[list(columns)].to_numpy(float)).all():
        raise ValueError('Nonfinite terrain overlay; no deletion or imputation allowed.')
    if set(columns).intersection(predictors.columns):
        raise ValueError('Baseline predictors already contain expanded terrain columns.')
    additions = extra_candidate_groups(profile)
    if set(additions).intersection(config['candidate_groups']):
        raise ValueError('Baseline candidate groups already contain expanded terrain groups.')
    # Assign by exact identity while preserving every baseline column and row.
    combined = predictors.copy(deep=True)
    aligned = extra.set_index('grid_id').loc[combined.grid_id]
    for name in columns:
        combined[name] = aligned[name].to_numpy(copy=True)
    new_config = deepcopy(config)
    new_config['candidate_groups'].update(additions)
    result = (values, grids, combined, new_config, crs)
    workflow.validate_inputs(*result)
    return result


def _checked_cache(folder, expected_identity, inputs):
    profile = expected_identity['profile']
    report = read(folder/'run_report.json')
    if (report.get('status') != 'completed' or report.get('profile') != profile
            or report.get('input_identity') != expected_identity
            or report.get('protocol_id') != identity(expected_identity)
            or set(report.get('output_sha256', {})) != set(OUTPUT_FILES)):
        raise ValueError('Terrain cache identity/report differs; existing cache was not overwritten.')
    for name, digest in report['output_sha256'].items():
        if not (folder/name).is_file() or sha(folder/name) != digest:
            raise ValueError(f'Terrain cache checksum failed: {name}; existing cache was not overwritten.')
    options = dict(dtype={'grid_id': str}, float_precision='round_trip')
    extra = pd.read_csv(folder/'terrain_extra.csv', **options)
    result = _assemble(inputs, extra, profile)
    stored = pd.read_csv(folder/'grid_predictors.csv', **options)
    try:
        pd.testing.assert_frame_equal(stored.reset_index(drop=True), result[2].reset_index(drop=True),
                                      check_dtype=False, check_exact=True,
                                      check_index_type=False)
    except AssertionError as exc:
        raise ValueError('Cached combined predictors differ from the unchanged baseline plus terrain.') from exc
    if read(folder/'candidate_groups.json') != result[3]['candidate_groups']:
        raise ValueError('Cached terrain candidate groups differ.')
    return result, report


def augment_inputs(inputs, source_dem_path, output_dir, *, base_fingerprints=None, profile=PROFILE):
    """Return ``(augmented_inputs, provenance)`` with an immutable hashed cache.

    ``inputs`` is the usual (50-year values, grids, predictors, config, crs)
    tuple. ``output_dir`` is an output/profile root, never ``inputs/eas``.
    Every distinct source/baseline/helper version receives a new subdirectory.
    Existing cache corruption raises instead of silently rebuilding in place.
    No NN, MLE, GP, weather extraction or model replacement runs here.
    """
    values, grids, predictors, config, crs = inputs
    additions = extra_candidate_groups(profile)
    workflow.validate_inputs(*inputs)
    if (grids.grid_id.isna().any() or grids.grid_id.duplicated().any()
            or predictors.grid_id.isna().any() or predictors.grid_id.duplicated().any()
            or not grids.grid_id.map(lambda value: isinstance(value, str)).all()
            or not predictors.grid_id.map(lambda value: isinstance(value, str)).all()):
        raise ValueError('Terrain overlay requires unique, nonmissing string GRID IDs.')
    source = Path(source_dem_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f'Original ERA5 geopotential file required: {source}')
    root = Path(output_dir).resolve()
    for name in ('inputs', 'models', 'data'):
        frozen = (ROOT/name).resolve()
        if root == frozen or frozen in root.parents:
            raise ValueError('Terrain overlay must not write into frozen bundle inputs/models/data.')
    if source == root or root in source.parents:
        raise ValueError('Terrain output directory cannot contain the original source DEM.')
    helper = Path(regional_terrain.__file__).resolve()
    evidence = json.loads(json.dumps(base_fingerprints or {}, sort_keys=True, allow_nan=False))
    input_identity = dict(schema_version=1, profile=profile,
        source_dem_sha256=sha(source), terrain_helper_sha256=sha(helper),
        overlay_code_sha256=sha(Path(__file__)),
        grid_fingerprint=frame_fingerprint(grids),
        predictor_fingerprint=frame_fingerprint(predictors),
        original_candidate_groups=deepcopy(config['candidate_groups']),
        coordinate_crs=crs, base_fingerprints=evidence)
    key = identity(input_identity)
    folder = root/key
    root.mkdir(parents=True, exist_ok=True)
    with lock(root):
        reused = folder.exists()
        if reused:
            result, report = _checked_cache(folder, input_identity, inputs)
        else:
            extra = regional_terrain.read_extra_terrain(source, grids)
            if profile == LATITUDE_PROFILE:
                # Geography is available at unseen GRID locations. No observed
                # temperature, fitted response or test-fold statistic is used.
                extra['latitude_deg'] = grids.set_index('grid_id').loc[extra.grid_id, 'latitude'].to_numpy()
            result = _assemble(inputs, extra, profile)
            if sha(source) != input_identity['source_dem_sha256'] or sha(helper) != input_identity['terrain_helper_sha256']:
                raise RuntimeError('Terrain source/helper changed during preprocessing.')
            # Publish only a complete, validated generation. Incomplete temporary
            # outputs are removed automatically and never masquerade as a cache.
            with tempfile.TemporaryDirectory(prefix='.terrain_build_', dir=root) as temporary:
                staged = Path(temporary)/'complete'
                staged.mkdir()
                atomic_csv(staged/'terrain_extra.csv', extra)
                atomic_csv(staged/'grid_predictors.csv', result[2])
                write(staged/'candidate_groups.json', result[3]['candidate_groups'])
                definitions = deepcopy(regional_terrain.EXTRA_TERRAIN_DEFINITIONS)
                if profile == LATITUDE_PROFILE:
                    definitions['latitude_deg'] = 'Geographic latitude in degrees north, copied by GRID ID; optional linear-mean FFS candidate. Scaling fitted on each training fold only.'
                report = dict(status='completed', profile=profile, protocol_id=key,
                    input_identity=input_identity, source_dem_at_creation=str(source),
                    n_grids=len(grids), added_columns=list(_extra_columns(profile)),
                    added_candidate_groups=additions,
                    candidate_group_count=len(result[3]['candidate_groups']),
                    candidate_column_count=sum(map(len, result[3]['candidate_groups'].values())),
                    definitions=definitions,
                    frozen_inputs_modified=False, original_predictors_preserved=True,
                    nn_executed=False, gp_executed=False,
                    output_sha256={name: sha(staged/name) for name in OUTPUT_FILES})
                write(staged/'run_report.json', report)
                _checked_cache(staged, input_identity, inputs)
                staged.rename(folder)
        if sha(source) != input_identity['source_dem_sha256'] or sha(helper) != input_identity['terrain_helper_sha256']:
            raise RuntimeError('Terrain source/helper changed during preprocessing.')
    # Runtime location/reuse state do not rewrite the immutable saved report.
    provenance = dict(report, directory=str(folder), cache_reused=reused)
    return result, provenance
