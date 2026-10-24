"""Prepare all EAS GRID covariates from existing local files, no downloads/NN.

Reuse audited monthly extraction and CCI area-weighting. Regional geometry
replaces Taiwan-only geometry. Monthly raw caches retain the raw values and
quality flags; only the previously approved tiny-negative-precipitation rule
is applied to derived annual statistics. Missing days/years never get filled.
"""
from __future__ import annotations
import sys
from pathlib import Path
SOURCE_ROOT=Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path: sys.path.insert(0,str(SOURCE_ROOT))
import argparse
import calendar
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter
from shapely.geometry import box
from taiwan_simulation import match_annual_events as io
from taiwan_simulation import prepare_predictors as weather
from taiwan_simulation.prepare_static_predictors import land_cover
from global_pipeline.diagnose_eas_buffer_folds import EAS_CRS, DATA_ROOT
from research_workflows import YEARS

VERSION='eas_mean_trend_predictors_v2'


def projected_coast_distance(grids, path):
    """Keep the EAS LCC metric; expand a too-small ocean-coast search window.

    Project the actual coastline boundary, never the edge of a clipped land
    polygon. The window edge is densified before projection for the guard.
    """
    path=Path(path)
    with zipfile.ZipFile(path) as archive:
        match=[n for n in archive.namelist() if n.endswith('GSHHS_i_L1.shp')]
        if len(match)!=1: raise ValueError('Expected one GSHHG ocean coastline layer.')
    points=gpd.GeoSeries(gpd.points_from_xy(grids.longitude_180,grids.latitude),crs=4326).to_crs(EAS_CRS)
    initial_failed=0
    for margin in (15,25,30):
        bounds=(float(grids.longitude_180.min()-margin),float(grids.latitude.min()-margin),
                float(grids.longitude_180.max()+margin),float(grids.latitude.max()+margin))
        if bounds[0]<=-180 or bounds[2]>=180 or bounds[1]<=-90 or bounds[3]>=90:
            raise ValueError('EAS coast search window outside regional projection domain.')
        coast=gpd.read_file(f'zip://{path.as_posix()}!{match[0]}',bbox=bounds)
        if coast.crs is None: raise ValueError('Missing coastline CRS.')
        lines=coast.to_crs(4326).geometry.boundary.intersection(box(*bounds))
        lines=lines.loc[~lines.is_empty]
        if lines.empty: continue
        parts=shapely.get_parts(gpd.GeoSeries(lines,crs=4326).to_crs(EAS_CRS).to_numpy())
        if not np.isfinite(shapely.get_coordinates(parts)).all():
            raise ValueError('Nonfinite projected coastline.')
        tree=shapely.STRtree(parts)
        distance=shapely.distance(points.to_numpy(),parts[tree.nearest(points.to_numpy())])/1000
        edge=gpd.GeoSeries([shapely.segmentize(box(*bounds).boundary,.1)],crs=4326).to_crs(EAS_CRS).iloc[0]
        clearance=points.distance(edge).to_numpy()/1000-distance
        if margin==15: initial_failed=int((clearance<=0).sum())
        if np.isfinite(distance).all() and np.all(clearance>0):
            return distance,dict(crs=EAS_CRS,search_bounds=bounds,search_margin_degrees=margin,
                window_min_clearance_km=float(clearance.min()),initial_window_guard_failures=initial_failed,
                definition='Projected distance to continuous GSHHG L1 ocean boundary; excludes lakes; same EAS LCC metric, expanded search only.')
    raise ValueError('Coastline window too small even after expansion.')


def validate_cache_source(source, protocol):
    """Permit an explicit static-stage upgrade, never a changed weather recipe."""
    source=Path(source).resolve()
    old=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    if old.get('version') not in ('eas_mean_trend_predictors_v1',VERSION):
        raise ValueError('Unsupported prior EAS cache protocol.')
    for key in ('registry_sha256','grid_sha256','root','years','precipitation_zero_tolerance_mm_day',
                'shared_extraction_sha256','matching_sha256'):
        if old.get(key)!=protocol[key]:
            raise ValueError(f'Incompatible prior cache protocol: {key}')
    if not (source/'cache_months').is_dir(): raise ValueError('Missing prior monthly cache directory.')
    return dict(directory=str(source),protocol_sha256=io.sha256(source/'protocol.json'),
                source_code_sha256=old['source_code_sha256'],policy='Verified raw weather cache only; old files read-only. Static/annual results rebuilt.')


def extract_or_reuse_month(root, grids, year, month, cache, prior=None):
    """Overlay verified old months without copying or modifying them."""
    filename=f'{year}{month:02d}.npz'; rejection=''
    if prior is not None and (Path(prior)/filename).is_file():
        try:
            report=weather.verified_cached_month(prior,grids,year,month)
        except (ValueError,KeyError,TypeError,OSError) as error:
            rejection=f'{type(error).__name__}: {error}'
        else:
            return dict(report,cache_origin='prior',cache_path=str(Path(prior)/filename),prior_cache_rejection='')
    report=weather.extract_month(root,grids,year,month,cache)
    return dict(report,cache_origin='current',cache_path=str(Path(cache)/filename),prior_cache_rejection=rejection)


def static_predictors(grids, root, output):
    """Coarse terrain, area-weighted CCI 2000, regional projected ocean coast."""
    root=Path(root)
    names=['era5_surface_geopotential_global_025.nc','esa_cci_land_cover_2000_global_300m.nc','gshhg-shp-2.3.7.zip']
    sources={n:io.signature(root/n) for n in names}
    with io.open_nc(root/names[0]) as ds:
        if getattr(ds['z'],'units','') not in ('m**2 s**-2','m2 s-2','m^2 s^-2'):
            raise ValueError('Unexpected geopotential units.')
        idx=io.spatial_indices(ds,grids)
        rr=np.arange(idx['latitude'].min()-2,idx['latitude'].max()+3)
        cc=np.arange(idx['longitude'].min()-2,idx['longitude'].max()+3)
        dem=io.gather(ds['z'],{'latitude':np.repeat(rr,len(cc)),'longitude':np.tile(cc,len(rr))},len(rr)*len(cc)).reshape(len(rr),len(cc))/9.80665
        if not np.isfinite(dem).all(): raise ValueError('Terrain halo has gaps.')
        row,col=idx['latitude']-rr[0],idx['longitude']-cc[0]
        mean=(uniform_filter(dem,size=5,mode='nearest')*25-dem)/24
        relief=maximum_filter(dem,size=5,mode='nearest')-minimum_filter(dem,size=5,mode='nearest')
        result=pd.DataFrame({'grid_id':grids.grid_id,'elevation_m':dem[row,col],
            'terrain_025_tpi_m':(dem-mean)[row,col], 'terrain_025_local_relief_m':relief[row,col]})
    io.write_csv(output/'terrain_predictors.csv',result)
    cover=land_cover(root/names[1],grids)
    io.write_csv(output/'land_cover_predictors.csv',cover)
    distance,coast_report=projected_coast_distance(grids,root/names[2])
    result=result.merge(cover,on='grid_id',validate='one_to_one')
    result['coast_distance_km']=distance
    result['static_complete']=np.isfinite(result.drop(columns='grid_id').to_numpy()).all(axis=1)&result.land_cover_valid_fraction.ge(1-1e-10)
    if sources!={n:io.signature(root/n) for n in names}: raise RuntimeError('Static source changed.')
    io.write_csv(output/'static_predictors.csv',result)
    io.write_json(output/'static_report.json',dict(sources=sources,crs=EAS_CRS,n_grids=len(grids),coast_search=coast_report,
        complete_grids=int(result.static_complete.sum()),definitions=dict(
            elevation='Geopotential / 9.80665, metres; not a high-resolution DEM',
            terrain='5x5 ERA5 source-cell TPI (24 neighbours) and relief; full two-cell halo',
            land_cover='Fixed CCI 2000, spherical area-weighted exact 90x90 pixel ERA5 footprint',
            coast='GSHHG intermediate L1 ocean boundary, approximate regional LCC distance in km; not lake distance, not exact geodesic distance')))
    return result


def reduce_month(values, quality, tolerance=.0001):
    """Vectorized raw->derived sums/counts; do not mutate cached arrays."""
    values=np.asarray(values,dtype=float).copy(); quality=np.asarray(quality).copy()
    tp=weather.NAMES.index('tp')
    clipped=(quality[:,:,tp]==5)&np.isfinite(values[:,:,tp])&(values[:,:,tp]<0)&(values[:,:,tp]>=-tolerance)
    correction=np.where(clipped,-values[:,:,tp],0).sum(axis=0)
    values[:,:,tp][clipped]=0; quality[:,:,tp][clipped]=0
    valid=(quality==0)&np.isfinite(values)
    sums=np.where(valid,values,0).sum(axis=0)
    counts=valid.sum(axis=0)
    wind_valid=valid[:,:,0]&valid[:,:,1]
    wind_sum=np.where(wind_valid,np.hypot(values[:,:,0],values[:,:,1]),0).sum(axis=0)
    return sums,counts,wind_sum,wind_valid.sum(axis=0),clipped.sum(axis=0),correction


def assemble(grids, cache, fixed, output):
    frames=[]; issues=[]
    for year in YEARS:
        total=np.zeros((len(grids),5)); counts=np.zeros_like(total,dtype=int)
        wind=np.zeros(len(grids)); wind_count=np.zeros(len(grids),dtype=int)
        clipped=np.zeros(len(grids),dtype=int); correction=np.zeros(len(grids)); days=0
        for month in range(1,13):
            path=cache[(year,month)] if isinstance(cache,dict) else cache/f'{year}{month:02d}.npz'
            with np.load(path,allow_pickle=False) as data:
                if data['grid_id'].tolist()!=grids.grid_id.tolist() or tuple(data['variables'])!=weather.NAMES:
                    raise ValueError('EAS monthly identity mismatch.')
                expected=pd.date_range(f'{year}-{month:02d}-01',periods=calendar.monthrange(year,month)[1]).strftime('%Y-%m-%d').tolist()
                if data['dates'].tolist()!=expected:
                    raise ValueError('EAS monthly calendar mismatch.')
                s,c,w,wc,z,delta=reduce_month(data['values'],data['quality'])
                total+=s; counts+=c; wind+=w; wind_count+=wc; clipped+=z; correction+=delta
                days+=len(data['dates'])
        annual=pd.DataFrame({'grid_id':grids.grid_id,'year':year,'expected_days':days,
            'tp_zero_clipped_days':clipped,'tp_zero_correction_total_mm':correction})
        for k,name in enumerate(weather.NAMES):
            column=name+('_annual_total' if name=='tp' else '_annual_mean')
            annual[column]=np.where(counts[:,k]==days,total[:,k] if name=='tp' else total[:,k]/days,np.nan)
            annual[name+'_valid_days']=counts[:,k]
            for j in np.flatnonzero(counts[:,k]!=days):
                issues.append(dict(grid_id=grids.grid_id.iloc[j],year=year,variable=name,valid_days=int(counts[j,k]),expected_days=days))
        annual['wind_vector_magnitude_annual_mean']=np.where(wind_count==days,wind/days,np.nan)
        annual['annual_weather_complete']=(counts==days).all(axis=1)
        frames.append(annual)
    annual=pd.concat(frames,ignore_index=True)
    io.write_csv(output/'annual_weather.csv',annual)
    io.write_csv(output/'annual_missing_report.csv',pd.DataFrame(issues,columns=['grid_id','year','variable','valid_days','expected_days']))
    groups={}; result=grids.copy()
    t=(np.asarray(YEARS)-2000.5)/10
    for name in (*weather.NAMES,'wind_vector_magnitude'):
        metric=name+('_annual_total' if name=='tp' else '_annual_mean')
        a=annual.pivot(index='grid_id',columns='year',values=metric).reindex(index=grids.grid_id,columns=YEARS).to_numpy(float)
        result[metric+'_mean_50y']=a.mean(axis=1)
        result[metric+'_trend_per_decade']=(a@t)/(t@t)
        result[metric+'_valid_years']=np.isfinite(a).sum(axis=1)
        groups[name]=[metric+'_mean_50y',metric+'_trend_per_decade']
    result=result.merge(fixed,on='grid_id',validate='one_to_one')
    groups.update(elevation=['elevation_m'],coast=['coast_distance_km'],
        land_cover=['agriculture_ratio','forest_ratio','urban_ratio','water_ratio'],
        terrain_position=['terrain_025_tpi_m'],terrain_relief=['terrain_025_local_relief_m'])
    columns=[c for group in groups.values() for c in group]
    result['candidate_data_complete']=np.isfinite(result[columns].to_numpy()).all(axis=1)&result.static_complete.eq(True)
    io.write_csv(output/'grid_predictors.csv',result)
    io.write_json(output/'candidate_groups.json',groups)
    return result,len(issues),int(annual.tp_zero_clipped_days.sum())


def run(registry,output,*,root=DATA_ROOT,workers=2,resume=False,cache_source=None):
    registry,output,root=map(lambda p:Path(p).resolve(),(registry,output,root))
    if not 1<=workers<=4: raise ValueError('Use 1--4 workers.')
    grids=pd.read_csv(registry,dtype={'grid_id':str})
    if len(grids)!=7822 or grids.grid_id.duplicated().any() or not grids.lsm.gt(.5).all():
        raise ValueError('Require exact 7,822-GRID diagnosed EAS registry.')
    if output.exists() and not resume: raise FileExistsError('Use new output or --resume.')
    protocol=dict(version=VERSION,registry_sha256=io.sha256(registry),grid_sha256=weather.extraction_identity(grids),
        root=str(root),years=list(YEARS),precipitation_zero_tolerance_mm_day=.0001,
        source_code_sha256=io.sha256(__file__),shared_extraction_sha256=io.sha256(weather.__file__),
        matching_sha256=io.sha256(io.__file__))
    prior=None
    if cache_source is not None:
        cache_source=Path(cache_source).resolve()
        if cache_source==output: raise ValueError('Cache source must be a different, read-only run.')
        protocol['cache_source']=validate_cache_source(cache_source,protocol)
        prior=cache_source/'cache_months'
    if resume and json.loads((output/'protocol.json').read_text(encoding='utf-8'))!=protocol:
        raise ValueError('Changed resume protocol.')
    output.mkdir(parents=True,exist_ok=True)
    io.write_json(output/'protocol.json',protocol)
    io.write_csv(output/'grid_index.csv',grids)
    cache=output/'cache_months'; cache.mkdir(exist_ok=True)
    started=time.perf_counter()
    def status(stage,**extra):
        io.write_json(output/'run_status.json',dict(status='running',stage=stage,elapsed_seconds=time.perf_counter()-started,
            n_grids=len(grids),expected_months=600,full_pipeline_executed=False,**extra))
        print(stage,extra,flush=True)
    try:
        # Run weather before slow CCI aggregation: the missing stage is visible.
        status('daily_extraction',completed_months=0)
        reports=[]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs=[pool.submit(extract_or_reuse_month,root,grids,y,m,cache,prior) for y in YEARS for m in range(1,13)]
            for job in as_completed(jobs):
                reports.append(job.result())
                if len(reports)==1 or len(reports)%6==0:
                    status('daily_extraction',completed_months=len(reports),last_month=[reports[-1]['year'],reports[-1]['month']],
                           reused_prior_months=sum(r['cache_origin']=='prior' for r in reports),
                           newly_extracted_months=sum(not r['reused'] for r in reports),
                           rejected_prior_months=sum(bool(r['prior_cache_rejection']) for r in reports))
        io.write_csv(output/'source_month_inventory.csv',pd.DataFrame(reports).sort_values(['year','month']))
        status('regional_static_predictors',completed_months=600)
        fixed=static_predictors(grids,root/'static',output)
        status('annual_means_and_trends',completed_months=600)
        cache_paths={(r['year'],r['month']):Path(r['cache_path']) for r in reports}
        result,issues,corrected=assemble(grids,cache_paths,fixed,output)
        report=dict(status='completed' if result.candidate_data_complete.all() else 'completed_with_issues',
            version=VERSION,region='EAS',n_grids=len(grids),complete_grids=int(result.candidate_data_complete.sum()),
            annual_missing_rows=issues,precipitation_zero_clipped_values=corrected,
            reused_prior_months=sum(r['cache_origin']=='prior' for r in reports),
            newly_extracted_months=sum(not r['reused'] for r in reports),
            elapsed_seconds=time.perf_counter()-started,full_pipeline_executed=False,nn_changed=False,
            definitions='Annual daily statistics -> 50-year means and per-decade OLS trends; all 50 years required. UTC daily means u/v/tcc, daily totals tp/ssrd; wind is magnitude of daily mean vector. No event-day features are substituted.',
            not_run=['NN','GP/FFS','spatial CV','return levels'],
            output_sha256={p.name:io.sha256(p) for p in output.glob('*.csv')})
        io.write_json(output/'run_report.json',report)
        io.write_json(output/'run_status.json',report)
        return report
    except BaseException as error:
        io.write_json(output/'run_status.json',dict(status='failed',error=str(error),error_type=type(error).__name__,
            elapsed_seconds=time.perf_counter()-started,full_pipeline_executed=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--cache-source',type=Path,help='Read-only previous EAS run; verify and reuse its monthly weather cache in a new run.')
    a=p.parse_args(); run(a.registry,a.output,workers=a.workers,resume=a.resume,cache_source=a.cache_source)
