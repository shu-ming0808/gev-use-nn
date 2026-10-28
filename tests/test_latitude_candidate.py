"""Latitude is a predeclared mean candidate, never a response or frozen input edit."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'src'), str(REPO/'tools/thesis_run_bundle')]
from portable import comparison, terrain, workflow
from portable.common import EAS_CRS
from research_workflows import COEFFICIENTS, YEARS
from spatial_coordinates import project_lonlat_to_km
from taiwan_simulation import cheng_spatial_cv as cv

PROFILE = 'expanded_latitude_v3'


def fixture(tmp_path, monkeypatch):
    n = 4
    grids = pd.DataFrame(dict(grid_id=[f'G{i:02d}' for i in range(n)],
        longitude_180=np.linspace(120, 121, n), latitude=np.linspace(23, 24, n), lsm=np.ones(n)))
    grids[['x_km', 'y_km']] = project_lonlat_to_km(grids.longitude_180, grids.latitude, target_crs=EAS_CRS)
    predictors = pd.DataFrame(dict(grid_id=grids.grid_id, elevation_m=np.arange(n)*100., candidate_data_complete=True))
    config = workflow.settings({'elevation': ['elevation_m']}, n, outer=2, inner=2, buffer=0, min_train=2)
    values = np.tile(np.linspace(20, 30, len(YEARS)), (n, 1))
    inputs = (values, grids, predictors, config, EAS_CRS)
    source = tmp_path/'source.nc'; source.write_bytes(b'isolated DEM I/O fixture')
    extra = pd.DataFrame({'grid_id': grids.grid_id,
        **{name: np.arange(n, dtype=float)/10 for name in terrain._extra_columns()}})
    monkeypatch.setattr(terrain.regional_terrain, 'read_extra_terrain', lambda *args: extra.copy())
    return inputs, source


def test_latitude_profile_preserves_every_baseline_and_has_distinct_identity(tmp_path, monkeypatch):
    inputs, source = fixture(tmp_path, monkeypatch)
    before = deepcopy(inputs)
    old, old_report = terrain.augment_inputs(inputs, source, tmp_path/'derived')
    new, report = terrain.augment_inputs(inputs, source, tmp_path/'derived', profile=PROFILE)
    assert report['profile'] == PROFILE and report['protocol_id'] != old_report['protocol_id']
    assert new[0] is inputs[0] and new[1] is inputs[1]
    pd.testing.assert_frame_equal(new[2][old[2].columns], old[2])
    pd.testing.assert_frame_equal(inputs[2], before[2])
    assert inputs[3] == before[3]
    assert new[3]['candidate_groups']['latitude'] == ['latitude_deg']
    assert report['candidate_group_count'] == old_report['candidate_group_count'] + 1
    assert report['candidate_column_count'] == old_report['candidate_column_count'] + 1
    np.testing.assert_array_equal(new[2].latitude_deg, inputs[1].latitude)
    again, reused = terrain.augment_inputs(inputs, source, tmp_path/'derived', profile=PROFILE)
    assert reused['cache_reused'] and reused['protocol_id'] == report['protocol_id']
    pd.testing.assert_frame_equal(new[2], again[2])
    assert not report['nn_executed'] and not report['gp_executed']


def test_latitude_alias_aligned_by_grid_and_checked_against_geometry(tmp_path, monkeypatch):
    inputs, source = fixture(tmp_path, monkeypatch)
    new, _ = terrain.augment_inputs(inputs, source, tmp_path/'derived', profile=PROFILE)
    response = pd.concat([new[1], pd.DataFrame(np.zeros((4,5)), columns=COEFFICIENTS)], axis=1)
    joined = cv.join_predictors(response, new[2].iloc[::-1], new[3]['candidate_groups'], coordinate_crs=EAS_CRS)
    np.testing.assert_array_equal(joined.latitude_deg, response.latitude)
    changed = new[2].copy(); changed.loc[0,'latitude_deg'] += 1.
    with pytest.raises(ValueError, match='latitude_deg'):
        cv.join_predictors(response, changed, new[3]['candidate_groups'], coordinate_crs=EAS_CRS)
    for forbidden in ['latitude', 'longitude_180', 'x_km', 'grid_id', *COEFFICIENTS]:
        with pytest.raises(ValueError, match='responses/identifiers'):
            cv.join_predictors(response, changed, {'bad': [forbidden]}, coordinate_crs=EAS_CRS)


def test_latitude_source_cannot_reuse_terrain_only_report(tmp_path, monkeypatch):
    inputs, source = fixture(tmp_path, monkeypatch)
    old, old_provenance = terrain.augment_inputs(inputs, source, tmp_path/'derived')
    new, new_provenance = terrain.augment_inputs(inputs, source, tmp_path/'derived', profile=PROFILE)
    report = comparison.terrain_input_metadata(old, old_provenance)
    with pytest.raises(ValueError, match='source provenance/fingerprints differ'):
        comparison.validate_terrain_source(report, new, new_provenance)
    matching = comparison.terrain_input_metadata(new, new_provenance)
    assert matching['terrain_profile'] == PROFILE
    assert comparison.validate_terrain_source(matching, new, new_provenance) == matching
    with pytest.raises(ValueError, match='requires verified terrain provenance'):
        comparison.terrain_input_metadata(new)
    new[2].loc[0,'latitude_deg'] += .25
    with pytest.raises(ValueError, match='latitude_deg'):
        comparison.terrain_input_metadata(new, new_provenance)


def test_latitude_profile_cli_has_separate_output_and_is_eas_only():
    import importlib.util
    spec = importlib.util.spec_from_file_location('latitude_runner', REPO/'tools/thesis_run_bundle/run.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    assert runner.real_output_name(PROFILE, 'ram16_mem8_v1') == 'eas_terrain_latitude_v3_ram16_mem8_v1'
    with pytest.raises(SystemExit) as exc:
        runner.main(['global', '--terrain-profile', PROFILE])
    assert exc.value.code == 2


def test_latitude_scaling_uses_training_mean_not_heldout_values():
    from land_cover_gp_analysis import _standardized_design
    train = np.array([[20.],[25.],[30.]])
    test = np.array([[60.]])
    _, center, scale = _standardized_design(train)
    design, reused_center, reused_scale = _standardized_design(test, center, scale)
    assert center[0] == 25.
    assert design[0,1] == pytest.approx(35./np.std(train))
    np.testing.assert_array_equal(center, reused_center)
    np.testing.assert_array_equal(scale, reused_scale)


def test_latitude_is_offered_to_inner_ffs_without_outer_test_responses(monkeypatch):
    lon = np.tile(np.linspace(105.,135.,6),4)
    lat = np.repeat(np.linspace(20.,40.,4),6)
    frame = pd.DataFrame(dict(grid_id=[f'G{i}' for i in range(24)], longitude_180=lon, latitude=lat))
    frame[['x_km','y_km']] = project_lonlat_to_km(lon,lat,target_crs=EAS_CRS)
    for i,name in enumerate(COEFFICIENTS): frame[name] = lat*.01+i
    predictors = pd.DataFrame(dict(grid_id=frame.grid_id, latitude_deg=lat))
    groups = {'latitude':['latitude_deg']}
    joined = cv.join_predictors(frame,predictors,groups,coordinate_crs=EAS_CRS)
    seen = []
    def select(train,target,**kwargs):
        seen.append((target,set(train.grid_id)))
        assert kwargs['candidate_groups'] == groups
        assert kwargs['max_steps'] is None
        # FFS may reject latitude. It is not silently forced into final models.
        return pd.DataFrame({'target':[target]}), pd.DataFrame({'target':[target]}), dict(
            predictor_names=[],kernel='RBF',nu=None,RMSE=.2)
    monkeypatch.setattr(cv,'spatial_forward_selection',select)
    monkeypatch.setattr(cv,'fit_gp_with_covariates',lambda xy,x,y,**kwargs:dict(mean=y.mean(),optimizer_success=True))
    monkeypatch.setattr(cv,'predict_gp_with_covariates',lambda model,xy,x:np.repeat(model['mean'],len(xy)))
    result=cv.nested_cv(joined,candidate_groups=groups,buffers_km={name:0 for name in COEFFICIENTS},
        outer_folds=2,inner_folds=2,min_train=2,max_train=24)
    for i,(target,train_ids) in enumerate(seen):
        test_ids=set(result['oof'].loc[(result['oof'].outer_fold==i//5)&(result['oof'].coefficient==target),'grid_id'])
        assert not train_ids.intersection(test_ids) and test_ids
    assert result['selections'].predictors.eq('intercept').all()
