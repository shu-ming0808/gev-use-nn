"""Immutable additive terrain cache, without fitting NN/MLE/GP."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'src'), str(REPO/'tools/thesis_run_bundle')]
from portable import terrain, workflow
from portable.common import EAS_CRS, read, sha, write
from research_workflows import YEARS
from spatial_coordinates import project_lonlat_to_km


def fixture(tmp_path, monkeypatch):
    n = 4
    grids = pd.DataFrame(dict(grid_id=[f'G{i:02d}' for i in range(n)],
        longitude_180=np.linspace(120, 121, n), latitude=np.linspace(23, 24, n), lsm=np.ones(n)))
    grids[['x_km', 'y_km']] = project_lonlat_to_km(grids.longitude_180, grids.latitude, target_crs=EAS_CRS)
    predictors = pd.DataFrame(dict(grid_id=grids.grid_id, elevation_m=np.arange(n)*100.,
        existing_float=[0.12345678901234568, 1.2, 2.3, 4.5], candidate_data_complete=True))
    config = workflow.settings({'elevation': ['elevation_m']}, n, outer=2, inner=2, buffer=0, min_train=2)
    values = np.tile(np.linspace(20, 30, len(YEARS)), (n, 1))
    inputs = (values, grids, predictors, config, EAS_CRS)
    source = tmp_path/'source.nc'
    source.write_bytes(b'fake source for isolated I/O contract test')
    extra = pd.DataFrame({'grid_id': grids.grid_id,
        **{name: np.arange(n, dtype=float)/10 for name in terrain._extra_columns()}})
    calls = []
    def derive(path, registry):
        calls.append((path, registry.grid_id.tolist()))
        return extra.copy()
    monkeypatch.setattr(terrain.regional_terrain, 'read_extra_terrain', derive)
    return inputs, source, extra, calls


def test_additive_roundtrip_preserves_baseline_and_reuses_without_nn(tmp_path, monkeypatch):
    inputs, source, extra, calls = fixture(tmp_path, monkeypatch)
    before = deepcopy(inputs)
    result, report = terrain.augment_inputs(inputs, source, tmp_path/'derived')
    assert report['profile'] == 'expanded_v2' and not report['cache_reused']
    assert report['candidate_group_count'] == 4 and report['candidate_column_count'] == 5
    assert not report['nn_executed'] and not report['gp_executed']
    assert result[0] is inputs[0] and result[1] is inputs[1]
    pd.testing.assert_frame_equal(result[2][inputs[2].columns], inputs[2])
    pd.testing.assert_frame_equal(inputs[2], before[2])
    pd.testing.assert_frame_equal(inputs[1], before[1])
    np.testing.assert_array_equal(inputs[0], before[0])
    assert inputs[3] == before[3] and len(calls) == 1
    folder = Path(report['directory'])
    frozen = {p.name: p.read_bytes() for p in folder.iterdir()}
    again, reused = terrain.augment_inputs(inputs, source, tmp_path/'derived')
    assert reused['cache_reused'] and len(calls) == 1
    pd.testing.assert_frame_equal(again[2], result[2])
    assert frozen == {p.name: p.read_bytes() for p in folder.iterdir()}
    assert not list((tmp_path/'derived').glob('.terrain_build_*'))


def test_changed_source_and_predictors_create_new_cache_not_overwrite(tmp_path, monkeypatch):
    inputs, source, _, _ = fixture(tmp_path, monkeypatch)
    _, first = terrain.augment_inputs(inputs, source, tmp_path/'derived')
    old = Path(first['directory'])/'run_report.json'
    original = old.read_bytes()
    source.write_bytes(b'new source content')
    _, second = terrain.augment_inputs(inputs, source, tmp_path/'derived')
    assert second['protocol_id'] != first['protocol_id']
    inputs[2].loc[0, 'elevation_m'] = 12.5
    _, third = terrain.augment_inputs(inputs, source, tmp_path/'derived')
    assert third['protocol_id'] != second['protocol_id']
    assert old.read_bytes() == original


def test_source_location_and_jobs_do_not_change_cache_identity(tmp_path, monkeypatch):
    inputs, source, _, calls = fixture(tmp_path, monkeypatch)
    _, first = terrain.augment_inputs(inputs, source, tmp_path/'derived')
    moved = tmp_path/'other_drive'/'source.nc'
    moved.parent.mkdir()
    moved.write_bytes(source.read_bytes())
    inputs[3]['n_jobs'] = 8
    result, again = terrain.augment_inputs(inputs, moved, tmp_path/'derived')
    assert again['cache_reused'] and again['protocol_id'] == first['protocol_id']
    assert result[3]['n_jobs'] == 8 and len(calls) == 1


@pytest.mark.parametrize('file', ['terrain_extra.csv', 'grid_predictors.csv', 'candidate_groups.json'])
def test_cache_corruption_fails_without_overwrite(tmp_path, monkeypatch, file):
    inputs, source, _, calls = fixture(tmp_path, monkeypatch)
    _, report = terrain.augment_inputs(inputs, source, tmp_path/'derived')
    path = Path(report['directory'])/file
    path.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='checksum failed'):
        terrain.augment_inputs(inputs, source, tmp_path/'derived')
    assert path.read_text() == 'changed' and len(calls) == 1


def test_cache_report_identity_mismatch_is_not_accepted(tmp_path, monkeypatch):
    inputs, source, _, _ = fixture(tmp_path, monkeypatch)
    _, report = terrain.augment_inputs(inputs, source, tmp_path/'derived')
    path = Path(report['directory'])/'run_report.json'
    saved = read(path); saved['input_identity']['profile'] = 'wrong'
    write(path, saved)
    with pytest.raises(ValueError, match='identity/report differs'):
        terrain.augment_inputs(inputs, source, tmp_path/'derived')


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'nonfinite', 'extra_column'])
def test_invalid_derived_rows_do_not_publish_cache(tmp_path, monkeypatch, kind):
    inputs, source, extra, _ = fixture(tmp_path, monkeypatch)
    if kind == 'missing': extra = extra.iloc[:-1]
    elif kind == 'duplicate': extra.loc[0, 'grid_id'] = extra.loc[1, 'grid_id']
    elif kind == 'nonfinite': extra.loc[0, terrain._extra_columns()[0]] = np.nan
    else: extra['surprise'] = 1.
    monkeypatch.setattr(terrain.regional_terrain, 'read_extra_terrain', lambda *args: extra)
    with pytest.raises(ValueError, match='Terrain overlay|Nonfinite terrain'):
        terrain.augment_inputs(inputs, source, tmp_path/'derived')
    assert not list((tmp_path/'derived').glob('*/run_report.json'))


def test_reordered_derived_rows_align_by_grid_id(tmp_path, monkeypatch):
    inputs, source, extra, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(terrain.regional_terrain, 'read_extra_terrain', lambda *args: extra.iloc[::-1])
    result, _ = terrain.augment_inputs(inputs, source, tmp_path/'derived')
    assert result[2].grid_id.tolist() == inputs[2].grid_id.tolist()
    for name in terrain._extra_columns():
        np.testing.assert_array_equal(result[2][name], extra[name])


def test_dem_changed_during_read_refuses_publish(tmp_path, monkeypatch):
    inputs, source, extra, _ = fixture(tmp_path, monkeypatch)
    def derive(*args):
        source.write_bytes(b'mutated while reading')
        return extra
    monkeypatch.setattr(terrain.regional_terrain, 'read_extra_terrain', derive)
    with pytest.raises(RuntimeError, match='changed during'):
        terrain.augment_inputs(inputs, source, tmp_path/'derived')
    assert not list((tmp_path/'derived').glob('*/run_report.json'))


def test_missing_dem_fails_before_cache_or_workflow(tmp_path, monkeypatch):
    inputs, _, _, calls = fixture(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match='geopotential'):
        terrain.augment_inputs(inputs, tmp_path/'absent.nc', tmp_path/'derived')
    assert not calls and not (tmp_path/'derived').exists()
