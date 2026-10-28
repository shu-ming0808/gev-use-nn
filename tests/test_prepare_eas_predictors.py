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


def test_expanded_static_keeps_legacy_values_and_sources(tmp_path,monkeypatch):
    import netCDF4 as nc
    from parameters.regional_terrain import EXTRA_TERRAIN_COLUMNS
    root=tmp_path/'source'; root.mkdir()
    # More than two source cells around each target, with nontrivial terrain.
    lat=np.arange(7)*.25+34.25; lon=np.arange(7)*.25+128.25
    dem=100.+np.arange(49).reshape(7,7)**1.2
    with nc.Dataset(root/'era5_surface_geopotential_global_025.nc','w') as ds:
        ds.createDimension('latitude',7); ds.createDimension('longitude',7)
        ds.createVariable('latitude','f8',('latitude',))[:]=lat
        ds.createVariable('longitude','f8',('longitude',))[:]=lon
        z=ds.createVariable('z','f8',('latitude','longitude')); z.units='m**2 s**-2'; z[:]=dem*9.80665
    for name in ('esa_cci_land_cover_2000_global_300m.nc','gshhg-shp-2.3.7.zip'):
        (root/name).write_bytes(b'immutable fixture')
    before={p.name:eas.io.sha256(p) for p in root.iterdir()}
    grids=pd.DataFrame({'grid_id':['b','a'],'latitude':lat[[4,2]],'longitude_180':lon[[3,3]]})
    cover=pd.DataFrame({'grid_id':grids.grid_id,'agriculture_ratio':[.2,.1],
        'forest_ratio':[.3,.4],'urban_ratio':[.1,.1],'water_ratio':[.05,.05],
        'land_cover_valid_fraction':[1.,1.]})
    monkeypatch.setattr(eas,'land_cover',lambda *args:cover)
    monkeypatch.setattr(eas,'projected_coast_distance',lambda *args:(np.array([12.,14.]),{'fixture':True}))
    legacy=tmp_path/'legacy'; legacy.mkdir()
    expanded=tmp_path/'expanded'; expanded.mkdir()
    a=eas.static_predictors(grids,root,legacy)
    b=eas.static_predictors(grids,root,expanded,terrain_schema='expanded_v2')
    pd.testing.assert_frame_equal(a,b[a.columns],check_exact=True)
    assert set(EXTRA_TERRAIN_COLUMNS).issubset(b)
    assert not set(EXTRA_TERRAIN_COLUMNS)&set(a)
    assert b.static_complete.all()
    report=json.loads((expanded/'static_report.json').read_text())
    assert report['terrain_schema']=='expanded_v2'
    assert 'Root-mean-square' in report['extra_terrain_definitions']['ruggedness']
    assert before=={p.name:eas.io.sha256(p) for p in root.iterdir()}


def test_expanded_assembly_adds_three_groups_four_columns_only(tmp_path,monkeypatch):
    import calendar
    from parameters.regional_terrain import EXTRA_TERRAIN_COLUMNS
    monkeypatch.setattr(eas,'YEARS',(2000,2001))
    cache=tmp_path/'cache'; cache.mkdir()
    grids=pd.DataFrame({'grid_id':['a','b']})
    for year in eas.YEARS:
        for month in range(1,13):
            days=calendar.monthrange(year,month)[1]
            dates=pd.date_range(f'{year}-{month:02d}-01',periods=days).strftime('%Y-%m-%d').to_numpy(dtype=str)
            np.savez(cache/f'{year}{month:02d}.npz',grid_id=grids.grid_id.to_numpy(dtype=str),
                variables=np.array(eas.weather.NAMES),dates=dates,values=np.ones((days,2,5))*(1+month/12),
                quality=np.zeros((days,2,5),dtype=np.uint8))
    fixed=pd.DataFrame({'grid_id':['a','b'],'elevation_m':[100.,200.],'coast_distance_km':[1.,2.],
        'agriculture_ratio':[.2,.3],'forest_ratio':[.4,.3],'urban_ratio':[.1,.2],'water_ratio':[.01,.02],
        'terrain_025_tpi_m':[10.,20.],'terrain_025_local_relief_m':[40.,50.],'static_complete':[True,True]})
    extended=fixed.copy()
    for column,value in zip(EXTRA_TERRAIN_COLUMNS,(2.,.6,.8,25.)):
        extended[column]=value
    legacy=tmp_path/'legacy'; legacy.mkdir()
    expanded=tmp_path/'expanded'; expanded.mkdir()
    a,issues_a,clipped_a=eas.assemble(grids,cache,fixed,legacy)
    b,issues_b,clipped_b=eas.assemble(grids,cache,extended,expanded,terrain_schema='expanded_v2')
    pd.testing.assert_frame_equal(a,b[a.columns],check_exact=True)
    assert (issues_a,issues_b,clipped_a,clipped_b)==(0,0,0,0)
    old=json.loads((legacy/'candidate_groups.json').read_text())
    new=json.loads((expanded/'candidate_groups.json').read_text())
    assert len(old)==11 and sum(map(len,old.values()))==20
    assert len(new)==14 and sum(map(len,new.values()))==24
    assert {key:new[key] for key in old}==old
    assert new['terrain_aspect']==['terrain_025_northness','terrain_025_eastness']


def test_raw_weather_cache_can_cross_explicit_terrain_schema(tmp_path):
    protocol=dict(version=eas.VERSION,registry_sha256='r',grid_sha256='g',root='data',years=[1976],
        precipitation_zero_tolerance_mm_day=.0001,source_code_sha256='old',shared_extraction_sha256='extract',
        matching_sha256='match',terrain_schema='legacy_v1')
    (tmp_path/'protocol.json').write_text(json.dumps(protocol),encoding='utf-8')
    (tmp_path/'cache_months').mkdir()
    upgraded={**protocol,'source_code_sha256':'new','terrain_schema':'expanded_v2',
        'extra_terrain_source_sha256':'new_terrain'}
    assert eas.validate_cache_source(tmp_path,upgraded)['policy'].endswith('Static/annual results rebuilt.')
