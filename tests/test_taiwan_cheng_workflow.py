from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from taiwan_simulation import prepare_taiwan_cheng as tw
from taiwan_simulation import cheng_spatial_cv as cv
import spatial_predictor_selection as ffs
from test_infer_global_cheng_nn import annual_file, fixture_checkpoint


def boundary_file(path):
    gpd.GeoDataFrame({'ADM0_A3':['TWN']}, geometry=[MultiPolygon([
        box(119,24,122.5,26), box(118,23,118.01,23.01)])], crs='EPSG:4326').to_file(path, driver='GeoJSON')
    return path


def test_geographic_filter_preserves_global_ids_and_rejects_bad_masks(tmp_path):
    p = tmp_path/'annual.nc'
    annual_file(p)
    full, index = tw.shared.read_annual_land(p)
    values, sub = tw.shared.read_annual_land(p, grid_selector=lambda g:g.grid_id.eq(index.grid_id.iloc[0]))
    np.testing.assert_array_equal(values, full[:1])
    pd.testing.assert_frame_equal(sub, index.iloc[:1])
    with pytest.raises(ValueError, match='boolean'):
        tw.shared.read_annual_land(p, grid_selector=lambda g:np.ones(len(g)))
    with pytest.raises(ValueError, match='no land'):
        tw.shared.read_annual_land(p, grid_selector=lambda g:np.zeros(len(g), dtype=bool))


def test_polygon_not_bounding_box_and_explicit_main_island(tmp_path):
    boundary = boundary_file(tmp_path/'country.geojson')
    main = tw.taiwan_geometry(boundary)
    all_islands = tw.taiwan_geometry(boundary, scope='all-islands')
    grids = pd.DataFrame(dict(longitude_180=[120,118.005,118.5,119], latitude=[25,23.005,23.5,24]))
    assert tw.geographic_mask(grids, main).tolist() == [True,False,False,True]
    assert tw.geographic_mask(grids, all_islands).tolist() == [True,True,False,True]


def test_taiwan_end_to_end_prepare_and_no_overwrite(tmp_path, monkeypatch):
    p = tmp_path/'annual.nc'
    annual_file(p)
    with tw.shared.open_nc(p, 'a') as ds:
        ds['latitude'][:] = [25,23]
        ds['longitude'][:] = [120,121,122]
    checkpoint, train = fixture_checkpoint(tmp_path)
    monkeypatch.setitem(tw.MODELS, 'original', (checkpoint, tw.file_sha256(checkpoint)))
    actual_infer = tw.infer_sequences
    monkeypatch.setattr(tw, 'infer_sequences', lambda values, **kw:actual_infer(values, train_directory=train, **kw))
    boundary = boundary_file(tmp_path/'country.geojson')
    output = tmp_path/'prepared'
    before = tw.file_sha256(p)
    report = tw.prepare(p, output, boundary=boundary, device='cpu', plots=False)
    assert report['n_grids'] == 2
    assert report['data_role'] == 'real_data_pilot_not_simulated_truth'
    assert report['completed_stage'] == 'taiwan_preparation_and_nn_only'
    assert report['temporal_policy'] == 'full_no_selection'
    assert 'temporal_selection_rule' not in report['pending']
    assert 'support_failure_grids' in report['summary']
    assert report['model']['formal_model_replaced'] is False
    assert tw.file_sha256(p) == before
    assert np.load(output/'inputs_21d.npy').shape == (2,21)
    assert np.load(output/'coefficients_original.npy').shape == (2,5)
    frame = pd.read_csv(output/'spatial_responses.csv')
    assert list(frame.columns[-5:]) == list(tw.COEFFICIENTS)
    assert frame.grid_id.tolist() == ['G000_0000','G000_0002']
    for name, digest in report['output_sha256'].items():
        assert tw.file_sha256(output/name) == digest
    assert report['timings']['total_wall_seconds_until_report'] >= report['timings']['nn_pipeline_seconds'] > 0
    with pytest.raises(FileExistsError):
        tw.prepare(p, output, boundary=boundary, plots=False)


def spatial_fixture():
    lon = np.tile(np.linspace(120.2,121.6,5), 4)
    lat = np.repeat(np.linspace(22.5,25,4), 5)
    xy = cv.project_lonlat_to_twd97_km(lon,lat)
    frame = pd.DataFrame(dict(grid_id=[f'g{i}' for i in range(20)], longitude_180=lon,
        latitude=lat, x_km=xy[:,0], y_km=xy[:,1]))
    for j, name in enumerate(tw.COEFFICIENTS):
        frame[name] = np.arange(20)*.01 + j*.1
    predictors = pd.DataFrame(dict(grid_id=frame.grid_id, elevation_m=np.arange(20)+100))
    return frame, predictors


def test_join_rejects_missing_or_leaky_predictors():
    frame, predictors = spatial_fixture()
    result = cv.join_predictors(frame, predictors.iloc[::-1], {'elevation':['elevation_m']})
    np.testing.assert_array_equal(result.elevation_m, predictors.elevation_m)
    with pytest.raises(ValueError, match='Incomplete'):
        cv.join_predictors(frame, predictors.iloc[:-1], {'elevation':['elevation_m']})
    with pytest.raises(ValueError, match='responses'):
        cv.join_predictors(frame, predictors, {'bad':['beta_mu']})


def test_nested_selection_never_sees_outer_test(monkeypatch):
    frame, predictors = spatial_fixture()
    groups = {'elevation':['elevation_m']}
    data = cv.join_predictors(frame, predictors, groups)
    seen = []
    def select(train, target, **kwargs):
        seen.append((target, set(train.grid_id)))
        assert kwargs['target_columns'] == cv.TARGET_COLUMNS
        assert kwargs['buffer_km'] == 0
        assert set(train.spatial_fold) == {0,1}
        return pd.DataFrame({'target':[target]}), pd.DataFrame({'target':[target]}), dict(
            predictor_names=[], kernel='RBF', nu=None, RMSE=.2)
    monkeypatch.setattr(cv, 'spatial_forward_selection', select)
    monkeypatch.setattr(cv, 'fit_gp_with_covariates', lambda xy,x,y,**kw:dict(mean=y.mean(),optimizer_success=True))
    monkeypatch.setattr(cv, 'predict_gp_with_covariates', lambda model,xy,x:np.repeat(model['mean'],len(xy)))
    result = cv.nested_cv(data, candidate_groups=groups, buffers_km={c:0. for c in tw.COEFFICIENTS},
        outer_folds=2, inner_folds=2, min_train=2, max_train=100)
    assert result['oof'].groupby('coefficient').size().to_dict() == {c:20 for c in tw.COEFFICIENTS}
    for i, (target, train_ids) in enumerate(seen):
        # Use explicit fold indexing; each outer fold visits five coefficients.
        test_ids = set(result['oof'].loc[(result['oof'].outer_fold == i//5) &
                                        (result['oof'].coefficient == target), 'grid_id'])
        assert test_ids and not train_ids.intersection(test_ids)
    with pytest.raises(ValueError, match='Explicit'):
        cv.nested_cv(data, candidate_groups=groups, buffers_km={c:None for c in tw.COEFFICIENTS},
            outer_folds=2, inner_folds=2, min_train=2, max_train=100)
    with pytest.raises(ValueError, match='training GRID'):
        cv.nested_cv(data, candidate_groups=groups, buffers_km={c:10000 for c in tw.COEFFICIENTS},
            outer_folds=2, inner_folds=2, min_train=2, max_train=100)


def test_real_ffs_accepts_beta_response_without_mutating_legacy_globals():
    frame, predictors = spatial_fixture()
    data = cv.join_predictors(frame, predictors, {'elevation':['elevation_m']})
    data['spatial_fold'] = np.arange(len(data)) % 2
    before_targets, before_buffers = dict(ffs.TARGETS), dict(ffs.BUFFER_KM)
    trial, path, winner = ffs.spatial_forward_selection(data, 'beta_mu',
        target_columns=cv.TARGET_COLUMNS, candidate_groups={'elevation':['elevation_m']},
        buffer_km=0, n_folds=2, min_train=4, max_train=100, n_jobs=1,
        max_steps=1, kernel_candidates=(('RBF',None),))
    assert np.isfinite(winner['RMSE'])
    assert set(winner['predictions'].target) == {'beta_mu'}
    assert len(winner['predictions']) == len(data)
    assert dict(ffs.TARGETS) == before_targets and dict(ffs.BUFFER_KM) == before_buffers
    with pytest.raises(ValueError, match='explicit buffer'):
        ffs.spatial_forward_selection(data, 'beta_mu', target_columns=cv.TARGET_COLUMNS)
