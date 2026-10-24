"""Portable local-only preprocessing. No downloads, credentials, NN or GP."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
from bundle import io, weather, static, coast

HERE=Path(__file__).resolve().parent
YEARS=tuple(range(1976,2026))
VERSION='portable_global_preprocessing_v2_projected_coast'


def code_hash():
    h=hashlib.sha256()
    for p in [Path(__file__),*sorted((HERE/'bundle').glob('*.py'))]:
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def load_registry(regions='all',exclude='EAS'):
    manifest=json.loads((HERE/'registries/manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files'].items():
        if io.sha256(HERE/'registries'/name)!=digest: raise ValueError(f'Registry checksum mismatch: {name}')
    raw=pd.read_csv(HERE/'registries/land_grid_lookup.csv.gz')
    raw['longitude_180']=(raw.longitude+180)%360-180
    row=np.rint((90-raw.latitude.to_numpy())/.25).astype(int)
    col=np.rint((raw.longitude.to_numpy()%360)/.25).astype(int)
    raw['grid_id']=[f'G{r:03d}_{c:04d}' for r,c in zip(row,col)]
    if not np.array_equal(raw.grid_id_025.to_numpy(),row*1440+col): raise ValueError('Original GRID id mapping mismatch.')
    if raw.grid_id.duplicated().any() or not raw.lsm.gt(.5).all(): raise ValueError('Invalid land registry.')
    eligible=raw.spatial_cv_eligible.eq(1)&raw.ar6_region.notna()
    available=set(raw.loc[eligible,'ar6_region'])
    selected=available if regions.lower()=='all' else {x.strip().upper() for x in regions.split(',') if x.strip()}
    omitted={x.strip().upper() for x in exclude.split(',') if x.strip()}
    if (selected|omitted)-available: raise ValueError(f'Unknown region codes: {(selected|omitted)-available}')
    selected-=omitted
    if not selected: raise ValueError('No regions selected. Use --exclude NONE to include EAS alone.')
    columns=['grid_id','latitude','longitude','longitude_180','lsm','ar6_region','ar6_region_id']
    grids=raw.loc[eligible&raw.ar6_region.isin(selected),columns].sort_values(['latitude','longitude'],ascending=[False,True]).reset_index(drop=True)
    return grids,raw.loc[~eligible].copy(),sorted(selected),manifest


def source_check(root,grids):
    root=Path(root).resolve()
    if not root.is_dir(): raise FileNotFoundError(f'Global daily source directory not found: {root}')
    missing_static=[n for n in static.FILES if not (root/'static'/n).is_file()]
    missing=[]; arco=0; cds=0
    def names_in(folder):
        try:
            with os.scandir(folder) as entries: return {entry.name for entry in entries if entry.is_file()}
        except FileNotFoundError: return set()
    for year in YEARS:
        cds_files=names_in(root/'daily_predictors'/str(year))
        for month in range(1,13):
            arco_files=names_in(root/'arco_daily_predictors'/str(year)/f'{month:02d}')
            for day in pd.date_range(f'{year}-{month:02d}-01',periods=weather.calendar.monthrange(year,month)[1]):
                p=weather.reader.arco_path(root,day)
                if p.name in arco_files:
                    arco+=1
                    if p.with_suffix('.audit.json').name not in arco_files: missing.append(dict(date=str(day.date()),issue='missing_ARCO_audit'))
                else:
                    paths={weather.reader.cds_path(root,year,month,n) for n in weather.NAMES}
                    if all(p.name in cds_files and p.with_suffix('.request.json').name in cds_files for p in paths): cds+=1
                    else: missing.append(dict(date=str(day.date()),issue='missing_CDS_source_or_request'))
    # Conservative uncompressed-cache + annual-output + temporary-space allowance.
    required=int(len(grids)*600*210*1.5+4_000_000_000)
    return dict(n_grids=len(grids),arco_days=arco,cds_fallback_days=cds,
                missing_day_count=len(missing),missing_days=missing,missing_static=missing_static,
                estimated_output_space_GB=round(required/1e9,2),required_bytes=required,
                scope='Existence/sidecar preflight only; values/units/UTC contracts checked during extraction.')


@contextmanager
def output_lock(output):
    path=output/'run.lock'
    with path.open('a+b') as handle:
        handle.seek(0)
        if path.stat().st_size==0: handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name=='nt':
            import msvcrt
            try: msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as e: raise RuntimeError('Another process is using this output directory.') from e
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: yield
        finally:
            handle.seek(0)
            if os.name=='nt': msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def make_protocol(grids,regions,manifest):
    return dict(version=VERSION,years=list(YEARS),regions=regions,
        registry_files=manifest['files'],grid_hash=static.io_hash_grid(grids),code_hash=code_hash(),
        rain_zero_tolerance_mm_day=weather.TOLERANCE,
        daily_cache='compact monthly aggregates; source daily files remain authoritative',
        root_policy='relative source names + sizes/mtimes/sidecar checksums; D/F drive change alone does not invalidate',
        coast_method=coast.METHOD)


def output_tables(grids,cache,fixed,output,progress):
    n=len(grids); total=np.zeros((n,6)); tx=np.zeros((n,6)); counts=np.zeros((n,6),int)
    missing_rows=0; fixed=fixed.set_index('grid_id').loc[grids.grid_id].reset_index()
    # Means/trends require ALL 50 complete annual metrics, never partial-year substitutions.
    t=(np.asarray(YEARS)-2000.5)/10
    for i,year in enumerate(YEARS):
        annual,issues,a=weather.summarize_year(cache,grids,year)
        good=np.isfinite(a); total+=np.where(good,a,0); tx+=np.where(good,a,0)*t[i]; counts+=good
        missing_rows+=len(issues)
        for region,part in annual.groupby('ar6_region',sort=True):
            dest=output/'regions'/region/'annual'; dest.mkdir(parents=True,exist_ok=True)
            io.write_csv(dest/f'{year}.csv',part)
            missing=issues.loc[issues.ar6_region.eq(region)]
            io.write_csv(dest/f'{year}_missing.csv',missing)
        progress('annual_summaries',completed_years=i+1,total_years=50)
    result=grids.copy(); groups={}
    for k,metric in enumerate(weather.METRICS):
        result[metric+'_mean_50y']=np.where(counts[:,k]==50,total[:,k]/50,np.nan)
        result[metric+'_trend_per_decade']=np.where(counts[:,k]==50,tx[:,k]/(t@t),np.nan)
        result[metric+'_valid_years']=counts[:,k]
        groups[(*weather.NAMES,'wind_vector_magnitude')[k]]=[metric+'_mean_50y',metric+'_trend_per_decade']
    result=result.merge(fixed,on='grid_id',validate='one_to_one')
    groups.update(elevation=['elevation_m'],coast=['coast_distance_km'],
        land_cover=['agriculture_ratio','forest_ratio','urban_ratio','water_ratio'],
        terrain_position=['terrain_025_tpi_m'],terrain_relief=['terrain_025_local_relief_m'])
    allcols=[c for group in groups.values() for c in group]
    result['weather_complete']=(counts==50).all(axis=1)
    result['candidate_data_complete']=result.weather_complete&result.static_complete&np.isfinite(result[allcols].to_numpy(float)).all(axis=1)
    summary=[]; outputs={}
    for region,part in result.groupby('ar6_region',sort=True):
        dest=output/'regions'/region; dest.mkdir(parents=True,exist_ok=True)
        io.write_csv(dest/'grid_predictors.csv',part)
        io.write_csv(dest/'grid_index.csv',grids.loc[grids.ar6_region.eq(region)])
        io.write_csv(dest/'incomplete_grids.csv',part.loc[~part.candidate_data_complete])
        io.write_json(dest/'candidate_groups.json',groups)
        outputs[(dest/'grid_predictors.csv').relative_to(output).as_posix()]=io.sha256(dest/'grid_predictors.csv')
        summary.append(dict(region=region,n_grids=len(part),complete_grids=int(part.candidate_data_complete.sum()),
                            weather_complete_grids=int(part.weather_complete.sum()),static_complete_grids=int(part.static_complete.sum())))
    io.write_csv(output/'region_summary.csv',pd.DataFrame(summary))
    io.write_json(output/'candidate_groups.json',groups)
    return dict(complete_grids=int(result.candidate_data_complete.sum()),n_grids=n,
                annual_missing_rows=missing_rows,outputs_sha256=outputs)


def run(args):
    if not 1<=args.workers<=4: raise ValueError('Use 1--4 processes; same disk, start with 2.')
    exclude='' if args.exclude.upper()=='NONE' else args.exclude
    grids,outside,regions,manifest=load_registry(args.regions,exclude)
    root=Path(args.data_root).resolve(); output=Path(args.output_root).resolve()
    if output==root or root in output.parents: raise ValueError('Output must be outside the read-only source tree.')
    if output.exists() and not args.resume: raise FileExistsError('Output exists; use --resume with unchanged scope.')
    output.mkdir(parents=True,exist_ok=True)
    with output_lock(output):
        protocol=make_protocol(grids,regions,manifest)
        pp=output/'protocol.json'
        if pp.is_file() and json.loads(pp.read_text(encoding='utf-8'))!=protocol:
            raise ValueError('Resume configuration/code/GRID changed. Use a different output directory.')
        if any(output.iterdir()) and not pp.is_file() and len(list(output.iterdir()))>1:
            raise ValueError('Existing files without a run protocol; use an empty output directory.')
        io.write_json(pp,protocol)
        started=time.perf_counter()
        def progress(stage,**extra):
            status=dict(status='running',stage=stage,utc=io.utc_now(),elapsed_seconds=time.perf_counter()-started,
                        selected_regions=regions,n_grids=len(grids),nn_executed=False,gp_executed=False,**extra)
            io.write_json(output/'run_status.json',status)
            print(json.dumps(status,ensure_ascii=False),flush=True)
        try:
            progress('preflight')
            check=source_check(root,grids); io.write_json(output/'preflight.json',check)
            if check['missing_static']: raise FileNotFoundError(f"Missing static sources: {check['missing_static']}")
            # Missing daily files become explicit quality flags; never silently dropped.
            free=shutil.disk_usage(output).free
            existing=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if free+existing<check['required_bytes']: raise OSError('Insufficient conservative output-space allowance; see preflight.json.')
            io.write_csv(output/'grid_index.csv',grids)
            io.write_csv(output/'ineligible_grid_replicas.csv',outside)
            shutil.copy2(HERE/'registries/land_outside_ar6_land_regions.csv',output/'land_outside_ar6_land_regions.csv')
            cache=output/'cache_months'; cache.mkdir(exist_ok=True)
            completed=0; reused=0
            progress('monthly_extraction',completed_months=0,total_months=600)
            with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_worker,initargs=(root,grids,cache,protocol['code_hash'])) as pool:
                jobs=[pool.submit(month_job,y,m) for y in YEARS for m in range(1,13)]
                for job in as_completed(jobs):
                    done=job.result(); completed+=1; reused+=int(done['reused'])
                    progress('monthly_extraction',completed_months=completed,total_months=600,reused_months=reused,last_month=[done['year'],done['month']])
            reference,_,_,_=load_registry('all','')
            fixed=static.prepare(root/'static',grids,output/'static',protocol['code_hash'],progress,reference)
            result=output_tables(grids,cache,fixed,output,progress)
            if code_hash()!=protocol['code_hash']: raise RuntimeError('Code changed during run; do not accept outputs.')
            report=dict(status='completed' if result['complete_grids']==len(grids) else 'completed_with_issues',
                        version=VERSION,utc=io.utc_now(),elapsed_seconds=time.perf_counter()-started,
                        reused_months=reused,nn_executed=False,gp_executed=False,downloads_executed=False,
                        selected_regions=regions,**result)
            io.write_json(output/'run_report.json',report); io.write_json(output/'run_status.json',report)
            print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        except BaseException as err:
            io.write_json(output/'run_status.json',dict(status='interrupted' if isinstance(err,KeyboardInterrupt) else 'failed',
                error_type=type(err).__name__,error=str(err),utc=io.utc_now(),elapsed_seconds=time.perf_counter()-started))
            raise


def initialize_worker(root,grids,cache,code):
    global WORKER
    WORKER=(root,grids,cache,code)


def month_job(year,month):
    root,grids,cache,code=WORKER
    return weather.extract_month(root,grids,year,month,cache,code)


def main():
    drive=HERE.anchor or 'F:/'
    default_data=Path(drive)/'論文資料/ERA5/global_predictors_1976_2025'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','run','status','regions'])
    p.add_argument('--data-root',type=Path,default=default_data)
    p.add_argument('--output-root',type=Path,default=Path(drive)/'論文資料/ERA5/global_preprocessed_1976_2025/run_v1')
    p.add_argument('--regions',default='all',help='AR6 comma-separated codes, or all.')
    p.add_argument('--exclude',default='EAS',help='Default skips EAS; NONE disables exclusions.')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.command=='status':
        print((args.output_root/'run_status.json').read_text(encoding='utf-8')); return
    if args.command=='run': run(args); return
    grids,outside,regions,manifest=load_registry(args.regions,'' if args.exclude.upper()=='NONE' else args.exclude)
    if args.command=='regions':
        print(grids.groupby('ar6_region').size().to_string()); print('Ineligible replicated GRID:',len(outside)); return
    result=source_check(args.data_root,grids)
    report=dict(data_root=str(args.data_root.resolve()),output_root=str(args.output_root.resolve()),
                regions=regions,ineligible_replicated_grids=len(outside),
                outside_ar6_land_grids=len(pd.read_csv(HERE/'registries/land_outside_ar6_land_regions.csv')),**result)
    report_dir=HERE/'reports'; report_dir.mkdir(exist_ok=True)
    io.write_json(report_dir/'preflight.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='missing_days'},ensure_ascii=False,indent=2))
    print('Detailed preflight: reports/preflight.json. No full preprocessing was started.')
    if result['missing_static'] or result['missing_day_count']: sys.exit(2)


if __name__=='__main__':
    main()
