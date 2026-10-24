import numpy as np
import pandas as pd
import pytest
import json
import zipfile
from shapely.geometry import box
from global_pipeline import prepare_eas_predictors as eas


def test_month_reduction_applies_only_approved_tiny_negative_rule():
    values=np.ones((3,2,5),dtype=float)
    quality=np.zeros(values.shape,dtype=np.uint8)
    tp=eas.weather.NAMES.index('tp')
    values[:,0,tp]=[-.0001,-.0001001,2.]
    quality[:2,0,tp]=5
    values[0,1,0]=np.nan  # also reject nonfinite values with a bad producer flag
    original=values.copy(); flags=quality.copy()
    sums,counts,wind,wind_count,clipped,correction=eas.reduce_month(values,quality)
    assert sums[0,tp]==2.
    assert counts[0,tp]==2
    assert clipped.tolist()==[1,0]
    assert correction[0]==pytest.approx(.0001)
    assert counts[1,0]==2 and wind_count[1]==2
    assert wind[1]==pytest.approx(2*np.sqrt(2))
    np.testing.assert_array_equal(values,original)
    np.testing.assert_array_equal(quality,flags)


def test_assembly_rejects_missing_calendar_day(tmp_path,monkeypatch):
    monkeypatch.setattr(eas,'YEARS',(1976,))
    grids=pd.DataFrame({'grid_id':['a']})
    np.savez(tmp_path/'197601.npz',grid_id=np.array(['a']),variables=np.array(eas.weather.NAMES),
        dates=np.array(['1976-01-01']),values=np.ones((1,1,5)),quality=np.zeros((1,1,5),dtype=np.uint8))
    with pytest.raises(ValueError,match='calendar'):
        eas.assemble(grids,tmp_path,pd.DataFrame(),tmp_path)


def test_projected_coast_expands_window_for_inland_grid(tmp_path,monkeypatch):
    path=tmp_path/'coast.zip'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('GSHHS_i_L1.shp','fixture')
    grids=pd.DataFrame({'longitude_180':[100.,140.],'latitude':[35.,35.]})
    coast=eas.gpd.GeoDataFrame(geometry=[box(120,30,121,40)],crs=4326)
    monkeypatch.setattr(eas.gpd,'read_file',lambda *a,**k:coast)
    distance,report=eas.projected_coast_distance(grids,path)
    assert report['initial_window_guard_failures']>0
    assert report['search_margin_degrees']>15
    assert report['window_min_clearance_km']>0
    points=eas.gpd.GeoSeries(eas.gpd.points_from_xy(grids.longitude_180,grids.latitude),crs=4326).to_crs(eas.EAS_CRS)
    expected=points.distance(coast.geometry.boundary.to_crs(eas.EAS_CRS).iloc[0])/1000
    np.testing.assert_allclose(distance,expected)


def test_prior_cache_is_read_only_and_used_only_after_verification(tmp_path,monkeypatch):
    prior=tmp_path/'prior'; prior.mkdir()
    (prior/'197601.npz').write_bytes(b'keep old cache')
    def verify(cache,grids,y,m):
        assert cache==prior
        return dict(year=y,month=m,reused=True,n_days=31)
    monkeypatch.setattr(eas.weather,'verified_cached_month',verify)
    monkeypatch.setattr(eas.weather,'extract_month',lambda *a:pytest.fail('Must reuse verified cache'))
    report=eas.extract_or_reuse_month(tmp_path,pd.DataFrame(),1976,1,tmp_path/'new',prior)
    assert report['cache_origin']=='prior'
    assert report['cache_path']==str(prior/'197601.npz')
    assert (prior/'197601.npz').read_bytes()==b'keep old cache'


def test_bad_prior_cache_is_recomputed_elsewhere_and_reason_retained(tmp_path,monkeypatch):
    prior=tmp_path/'prior'; prior.mkdir()
    (prior/'197601.npz').write_bytes(b'bad but preserved')
    def verify(*args): raise ValueError('Cached hash mismatch')
    def extract(root,grids,y,m,cache):
        assert cache==tmp_path/'new'
        return dict(year=y,month=m,reused=False,n_days=31)
    monkeypatch.setattr(eas.weather,'verified_cached_month',verify)
    monkeypatch.setattr(eas.weather,'extract_month',extract)
    report=eas.extract_or_reuse_month(tmp_path,pd.DataFrame(),1976,1,tmp_path/'new',prior)
    assert report['cache_origin']=='current'
    assert 'hash mismatch' in report['prior_cache_rejection']
    assert (prior/'197601.npz').read_bytes()==b'bad but preserved'


def test_prior_protocol_allows_coast_change_but_not_weather_change(tmp_path):
    protocol=dict(version='eas_mean_trend_predictors_v1',registry_sha256='r',grid_sha256='g',
                  root='data',years=[1976],precipitation_zero_tolerance_mm_day=.0001,
                  source_code_sha256='old',shared_extraction_sha256='extract',matching_sha256='match')
    (tmp_path/'protocol.json').write_text(json.dumps(protocol),encoding='utf-8')
    (tmp_path/'cache_months').mkdir()
    current={**protocol,'version':eas.VERSION,'source_code_sha256':'new'}
    lineage=eas.validate_cache_source(tmp_path,current)
    assert lineage['protocol_sha256']==eas.io.sha256(tmp_path/'protocol.json')
    current['shared_extraction_sha256']='different'
    with pytest.raises(ValueError,match='shared_extraction'):
        eas.validate_cache_source(tmp_path,current)


def test_assembly_reads_exact_verified_cache_mapping(tmp_path,monkeypatch):
    monkeypatch.setattr(eas,'YEARS',(1976,))
    grids=pd.DataFrame({'grid_id':['a']})
    target=tmp_path/'verified_old_month.npz'
    np.savez(target,grid_id=np.array(['a']),variables=np.array(eas.weather.NAMES),
             dates=np.array(['1976-01-01']),values=np.ones((1,1,5)),quality=np.zeros((1,1,5),dtype=np.uint8))
    with pytest.raises(ValueError,match='calendar'):
        eas.assemble(grids,{(1976,1):target},pd.DataFrame(),tmp_path)
