import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import pytest
import netCDF4 as nc
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import preprocess as app
from bundle import io,weather,static,coast


def grid():
    return pd.DataFrame(dict(grid_id=['G360_0000'],latitude=[0.],longitude=[0.],longitude_180=[0.],lsm=[1.],ar6_region=['TEST'],ar6_region_id=[0]))


def write_cds(path,statistic,year=1976,month=1):
    path.parent.mkdir(parents=True,exist_ok=True)
    previous=Path.cwd()
    try:
        os.chdir(path.parent)
        with nc.Dataset(path.name,'w') as ds:
            days=weather.calendar.monthrange(year,month)[1]
            ds.createDimension('time',days); ds.createDimension('latitude',1); ds.createDimension('longitude',1)
            t=ds.createVariable('time','f8',('time',)); t.units=f'days since {year}-{month:02d}-01'; t[:]=np.arange(days)
            ds.createVariable('latitude','f8',('latitude',))[:]=[0]
            ds.createVariable('longitude','f8',('longitude',))[:]=[0]
            for name,(kind,unit) in io.FIELDS.items():
                if kind!=statistic: continue
                v=ds.createVariable(name,'f8',('time','latitude','longitude')); v.units=unit; v.cell_methods='time: '+kind
                v[:]=.5 if name=='tcc' else 1.
    finally: os.chdir(previous)
    io.write_json(path.with_suffix('.request.json'),dict(request=dict(daily_statistic='daily_'+statistic,time_zone='utc+00:00',frequency='1_hourly')))


def test_packaged_registry_exact_scope():
    g,outside,regions,_=app.load_registry()
    assert len(g)==341743 and len(regions)==45 and 'EAS' not in regions
    assert len(outside)==1439
    eas,_,_,_=app.load_registry('EAS','')
    assert len(eas)==7822
    assert not g.grid_id.duplicated().any() and g.lsm.gt(.5).all()
    with pytest.raises(ValueError,match='Unknown'): app.load_registry('NOT_A_REGION','')


def test_rain_cleaning_exact_boundary_and_no_mutation():
    a=np.ones((3,5)); q=np.zeros((3,5),np.uint8)
    a[:,3]=[-.0001,-.0001001,-2.]; q[:,3]=5
    saved=a.copy(); flags=q.copy()
    vals,good,clean,clipped,delta=weather.clean_day(a,q)
    assert clipped.tolist()==[True,False,False]
    assert good[:,3].tolist()==[True,False,False]
    assert delta.sum()==pytest.approx(.0001)
    np.testing.assert_equal(a,saved); np.testing.assert_equal(q,flags)


def test_monthly_extract_reuse_move_and_missing_data(tmp_path):
    root=tmp_path/'source'; cache=tmp_path/'cache'; cache.mkdir()
    for kind in ('mean','sum'): write_cds(root/'daily_predictors/1976'/f'era5_daily_{kind}_197601_global_025.nc',kind)
    report=weather.extract_month(root,grid(),1976,1,cache,'v1')
    assert not report['reused']
    assert weather.extract_month(root,grid(),1976,1,cache,'v1')['reused']
    with np.load(cache/'197601.npz') as d:
        assert d['sums'][0,3]==31 and d['counts'].min()==31
        assert d['quality_counts'][:,:,1:].sum()==0
    moved=tmp_path/'other_drive'; shutil.copytree(root,moved,copy_function=shutil.copy2)
    assert weather.extract_month(moved,grid(),1976,1,cache,'v1')['reused']
    # Losing a source invalidates cache; keep explicit gaps, not zero-filled rain.
    (moved/'daily_predictors/1976/era5_daily_sum_197601_global_025.nc').unlink()
    assert not weather.extract_month(moved,grid(),1976,1,cache,'v1')['reused']
    with np.load(cache/'197601.npz') as d:
        assert d['counts'][0,3]==0 and d['quality_counts'][0,3,1]==31


def test_cache_checksum_and_code_change_invalidate(tmp_path):
    root=tmp_path/'raw'; root.mkdir(); cache=tmp_path/'cache'; cache.mkdir()
    weather.extract_month(root,grid(),1976,2,cache,'v1')
    assert weather.extract_month(root,grid(),1976,2,cache,'v1')['reused']
    with (cache/'197602.npz').open('ab') as f: f.write(b'corrupt')
    assert not weather.extract_month(root,grid(),1976,2,cache,'v1')['reused']
    assert not weather.extract_month(root,grid(),1976,2,cache,'v2')['reused']


def test_annual_requires_all_days_and_leap_year(tmp_path):
    for m in range(1,13):
        days=weather.calendar.monthrange(1976,m)[1]
        counts=np.full((1,6),days)
        if m==2: counts[0,3]-=1
        weather.save_npz(tmp_path/f'1976{m:02d}.npz',sums=np.full((1,6),days),counts=counts,
            quality_counts=np.zeros((1,5,7),int),clips=np.zeros(1,int),corrections=np.zeros(1),n_days=np.array(days),grid_id=np.array(['G360_0000']))
    annual,issues,a=weather.summarize_year(tmp_path,grid(),1976)
    assert annual.expected_days.iloc[0]==366
    assert np.isnan(a[0,3]) and a[0,0]==1
    assert issues.variable.tolist()==['tp'] and issues.valid_days.iloc[0]==365


def test_full_50_year_mean_and_trend_and_missingness(tmp_path,monkeypatch):
    def year(cache,g,year):
        t=(year-2000.5)/10
        a=np.ones((1,6))*(4+2*t)
        if year==1980: a[0,3]=np.nan
        annual=g[['grid_id','ar6_region']].copy(); annual['year']=year
        return annual,pd.DataFrame(columns=['ar6_region']),a
    monkeypatch.setattr(weather,'summarize_year',year)
    fixed=pd.DataFrame({'grid_id':grid().grid_id,'elevation_m':[1.],'coast_distance_km':[2.],
        'agriculture_ratio':[0.],'forest_ratio':[1.],'urban_ratio':[0.],'water_ratio':[0.],
        'terrain_025_tpi_m':[0.],'terrain_025_local_relief_m':[0.],'static_complete':[True]})
    report=app.output_tables(grid(),tmp_path,fixed,tmp_path,lambda *a,**k:None)
    result=pd.read_csv(tmp_path/'regions/TEST/grid_predictors.csv')
    assert result.u10_annual_mean_mean_50y.iloc[0]==pytest.approx(4)
    assert result.u10_annual_mean_trend_per_decade.iloc[0]==pytest.approx(2)
    assert np.isnan(result.tp_annual_total_mean_50y.iloc[0])
    assert result.tp_annual_total_valid_years.iloc[0]==49 and report['complete_grids']==0


@pytest.mark.parametrize('lat,lon,rows',[(0,0,90),(0,180,90),(0,-180,90),(90,0,45),(-90,0,45),(25,121.25,90)])
def test_cci_footprint_dateline_and_poles(lat,lon,rows):
    rr,cc=static.footprint(lat,lon)
    assert len(rr)==rows and len(cc)==90 and len(np.unique(cc))==90
    assert rr.min()>=0 and rr.max()<64800 and cc.min()>=0 and cc.max()<129600
    if abs(lon)==180: assert cc.min()==0 and cc.max()==129599


def test_spherical_fractions_preserve_unknown_missing():
    rr=np.arange(90); d=np.ones((90,90))*50
    r=static.fractions(d,rr,np.array([0,50,210]))
    assert r['forest_ratio']==pytest.approx(1) and r['land_cover_valid_fraction']==pytest.approx(1)
    d[:,0]=0
    r=static.fractions(d,rr,np.array([0,50,210]))
    assert r['land_cover_valid_fraction']==pytest.approx(89/90)


def test_coast_dateline_short_arc_and_seam_exclusion():
    segments=coast.split_edges([[179.9,0],[-179.9,0]])
    _,_,length=coast.GEOD.inv(segments[:,0,0],segments[:,0,1],segments[:,1,0],segments[:,1,1])
    assert length.max()<=5000.00001
    assert len(coast.split_edges([[180,10],[180,20]]))==0
    assert len(coast.split_edges([[179.999,10],[179.999,20]]))>0
    assert len(coast.split_edges([[0,-90],[0,-80]]))==0
    g=grid(); g['longitude_180']=180.; g['latitude']=.1
    plan=coast.plan_for_region('TEST',g)
    distance,report=coast.regional_distance(segments,g,g,plan)
    assert distance[0]==pytest.approx(11.0574,abs=.005)
    assert report['window_min_clearance_km']>0


def test_output_lock_blocks_second_process_handle(tmp_path):
    with app.output_lock(tmp_path):
        with pytest.raises((RuntimeError,BlockingIOError)):
            with app.output_lock(tmp_path): pass


def test_precise_lookup_handles_unsorted_longitudes(tmp_path):
    path=tmp_path/'a.nc'
    with nc.Dataset(path,'w') as ds:
        ds.createDimension('lat',2); ds.createDimension('lon',3)
        ds.createVariable('lat','f8',('lat',))[:]=[1.,0.]
        ds.createVariable('lon','f8',('lon',))[:]=[0.,180.,359.75]
    with io.open_nc(path) as ds:
        q=pd.DataFrame({'latitude':[0.,1.],'longitude_180':[-.25,-180.]})
        ix=io.spatial_indices(ds,q)
        assert ix['lat'].tolist()==[1,0] and ix['lon'].tolist()==[2,1]
        q.loc[0,'latitude']=.1
        with pytest.raises(io.DataIssue): io.spatial_indices(ds,q)


def test_process_workers_write_independent_months(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    root=tmp_path/'raw'; root.mkdir(); cache=tmp_path/'cache'; cache.mkdir()
    with ProcessPoolExecutor(max_workers=2,initializer=app.initialize_worker,
                             initargs=(root,grid(),cache,'test')) as pool:
        jobs=[pool.submit(app.month_job,1976,m) for m in (1,2)]
        assert [j.result()['month'] for j in jobs]==[1,2]
    assert len(list(cache.glob('*.npz')))==2
    with np.load(cache/'197602.npz') as d:
        assert int(d['n_days'])==29 and d['counts'].sum()==0


def test_coastline_north_south_layers_required(tmp_path,monkeypatch):
    import zipfile
    p=tmp_path/'coast.zip'
    with zipfile.ZipFile(p,'w') as z: z.writestr('GSHHS_i_L1.shp',b'placeholder')
    with pytest.raises(ValueError,match='Antarctic L5'):
        coast.layers(p)


def test_projected_distance_is_to_segment_not_vertices():
    from shapely.geometry import LineString
    g=grid(); _,latitude,_=coast.GEOD.fwd(0,0,0,50000)
    g['latitude']=latitude
    crs='+proj=aeqd +lat_0=0 +lon_0=0 +datum=WGS84 +units=m'
    distance,report=coast.distance_to_lines(g,[LineString([(-100000,0),(100000,0)])],crs)
    assert distance[0]==pytest.approx(50,abs=1e-8)
    assert report['max_endpoint_distance_discrepancy_pct']<1e-6


def test_projection_uses_full_region_and_preserves_eas():
    reference,_,_,_=app.load_registry('all','')
    plan=coast.plan_for_region('EAS',reference)
    assert plan['crs']==coast.EAS_CRS and plan['n_reference_grids']==7822
    for region in ('RAR','EAN','CAF'):
        p=coast.plan_for_region(region,reference)
        assert '+proj=aeqd' in p['crs']
        part=reference.loc[reference.ar6_region.eq(region)]
        t=coast.Transformer.from_crs(4326,p['crs'],always_xy=True)
        assert np.isfinite(t.transform(part.longitude_180.to_numpy(),part.latitude.to_numpy())).all()


def test_regional_coast_cache_requires_same_method_and_reference(tmp_path,monkeypatch):
    g=grid(); calls=[]
    def fake(path,part,plan):
        calls.append(True)
        return np.ones(len(part)),{'projection_review_recommended':False}
    monkeypatch.setattr(coast,'eas_distance',fake)
    g['ar6_region']='EAS'
    first=coast.prepare('unused.zip',g,g,tmp_path,{'v':1},lambda *a,**k:None)
    second=coast.prepare('unused.zip',g,g,tmp_path,{'v':1},lambda *a,**k:None)
    assert len(calls)==1 and first.equals(second)
    coast.prepare('unused.zip',g,g,tmp_path,{'v':2},lambda *a,**k:None)
    assert len(calls)==2


def test_eas_window_expands_without_changing_projection(monkeypatch):
    import geopandas as gpd
    from shapely.geometry import box
    g=grid(); g['longitude_180']=100.25; g['latitude']=35.; g['ar6_region']='EAS'
    monkeypatch.setattr(coast,'layers',lambda p:{1:'unused',5:'unused'})
    monkeypatch.setattr(coast.gpd,'read_file',lambda *a,**k:gpd.GeoDataFrame(geometry=[box(115,30,120,40)],crs=4326))
    used=[]
    def distance(grids,lines,crs):
        used.append(crs)
        return np.array([1600.]),{'projection_review_recommended':False}
    monkeypatch.setattr(coast,'distance_to_lines',distance)
    d,r=coast.eas_distance('unused',g,{'bounds':[85,4.5,162.25,59.75]})
    assert r['initial_window_guard_failures']==1 and r['search_margin_degrees']==25
    assert r['window_min_clearance_km']>0 and d[0]==1600
    assert used==[coast.EAS_CRS,coast.EAS_CRS]
