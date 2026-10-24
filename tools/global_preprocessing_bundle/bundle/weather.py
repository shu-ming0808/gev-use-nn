"""One read per selected source day, shared across all requested AR6 regions.

Store compact monthly sufficient aggregates, NOT another copy of daily grids.
Raw source files are read-only. Tiny-negative rain correction is derived only.
"""
from pathlib import Path
import calendar
import hashlib
import json
import os
import numpy as np
import pandas as pd
from bundle import io, reader

NAMES = tuple(reader.NAMES)
METRICS = tuple(n+('_annual_total' if n=='tp' else '_annual_mean') for n in NAMES) + ('wind_vector_magnitude_annual_mean',)
VERSION = 'global_monthly_aggregates_v1'
TOLERANCE = .0001


def relative_signature(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    result = {'path': path.relative_to(root).as_posix()}
    if not path.is_file():
        return dict(result, missing=True)
    stat = path.stat()
    result.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    # Sidecars are small: checksum the processing contract too.
    if path.suffix == '.json':
        result['sha256'] = io.sha256(path)
    return result


def month_identity(root, grids, year, month, code_hash):
    dates = pd.date_range(f'{year}-{month:02d}-01', periods=calendar.monthrange(year,month)[1])
    watched = set()
    for day in dates:
        p = reader.arco_path(root, day)
        watched.update([p, p.with_suffix('.audit.json')])
        if not p.is_file():
            for n in NAMES:
                q = reader.cds_path(root,year,month,n)
                watched.update([q,q.with_suffix('.request.json')])
    return dict(version=VERSION, year=year, month=month, code_hash=code_hash,
                grid_hash=hashlib.sha256(grids.to_csv(index=False).encode()).hexdigest(),
                sources=[relative_signature(p,root) for p in sorted(watched)])


def cached(path, identity):
    path = Path(path)
    meta = path.with_suffix('.json')
    if not path.is_file() or not meta.is_file():
        return False
    try:
        m = json.loads(meta.read_text(encoding='utf-8'))
        return m['identity']==identity and m['sha256']==io.sha256(path)
    except (ValueError,KeyError,OSError):
        return False


def save_npz(path, **arrays):
    path=Path(path); part=path.with_suffix('.part')
    with part.open('wb') as handle:
        np.savez_compressed(handle,**arrays)
    os.replace(part,path)


def clean_day(values, quality):
    a=np.array(values,dtype=float,copy=True); q=np.array(quality,copy=True)
    rain=NAMES.index('tp')
    mask=(q[:,rain]==5)&np.isfinite(a[:,rain])&(a[:,rain]<0)&(a[:,rain]>=-TOLERANCE)
    correction=np.where(mask,-a[:,rain],0)
    a[mask,rain]=0; q[mask,rain]=0
    valid=(q==0)&np.isfinite(a)
    wind_good=valid[:,0]&valid[:,1]
    metrics=np.column_stack([a,np.hypot(a[:,0],a[:,1])])
    valid=np.column_stack([valid,wind_good])
    return np.where(valid,metrics,0),valid,q,mask,correction


def extract_month(root, grids, year, month, output, code_hash):
    root,output=Path(root),Path(output)
    target=output/f'{year}{month:02d}.npz'
    identity=month_identity(root,grids,year,month,code_hash)
    if cached(target,identity):
        return dict(year=year,month=month,reused=True)
    days=pd.date_range(f'{year}-{month:02d}-01',periods=calendar.monthrange(year,month)[1])
    n=len(grids)
    sums=np.zeros((n,6)); counts=np.zeros((n,6),np.uint16)
    quality_counts=np.zeros((n,5,7),np.uint16)
    coverage_counts=np.zeros((5,3),np.uint32)
    clips=np.zeros(n,np.uint16); corrections=np.zeros(n)
    errors=[]
    for day in days:
        raw=np.full((n,5),np.nan); quality=np.ones((n,5),np.uint8)
        arco=reader.arco_path(root,day)
        groups={}
        for k,name in enumerate(NAMES):
            kind,path=('arco',arco) if arco.is_file() else ('cds',reader.cds_path(root,year,month,name))
            groups.setdefault((kind,path),[]).append(k)
        for (kind,path),cols in groups.items():
            if not path.is_file():
                errors.append(dict(date=str(day.date()),file=path.relative_to(root).as_posix(),error='missing_file'))
                continue
            try:
                a,q,c,e=reader.read_fields(path,grids,[str(day.date())],[NAMES[k] for k in cols],kind)
                raw[:,cols]=a[0]; quality[:,cols]=q[0]
                for j,k in enumerate(cols):
                    for code in (0,1,2): coverage_counts[k,code]+=np.sum(c[0,:,j]==code)
                errors.extend(dict(date=str(day.date()),file=path.relative_to(root).as_posix(),**err) for err in e)
            except (ValueError,KeyError,OSError,RuntimeError,IndexError,AttributeError) as err:
                quality[:,cols]=2
                errors.append(dict(date=str(day.date()),file=path.relative_to(root).as_posix(),error=str(err)))
        vals,valid,q,clipped,delta=clean_day(raw,quality)
        sums+=vals; counts+=valid.astype(np.uint16); clips+=clipped.astype(np.uint16); corrections+=delta
        for code in range(7): quality_counts[:,:,code]+=(q==code).astype(np.uint16)
    if identity!=month_identity(root,grids,year,month,code_hash):
        raise RuntimeError(f'Source changed during {year}-{month:02d}; no new cache committed.')
    output.mkdir(parents=True,exist_ok=True)
    save_npz(target,sums=sums,counts=counts,quality_counts=quality_counts,
             clips=clips,corrections=corrections,coverage_counts=coverage_counts,
             n_days=np.array(len(days)),grid_id=grids.grid_id.to_numpy(dtype=str))
    io.write_json(target.with_suffix('.json'),dict(identity=identity,sha256=io.sha256(target),
                  errors=errors,unresolved_value_count=int(quality_counts[:,:,1:].sum()),
                  corrected_rain_values=int(clips.sum()),coverage_code=reader.COVERAGE))
    return dict(year=year,month=month,reused=False)


def summarize_year(cache, grids, year):
    n=len(grids); sums=np.zeros((n,6)); counts=np.zeros((n,6),int)
    bad=np.zeros((n,5,7),int); clips=np.zeros(n,int); correction=np.zeros(n); days=0
    for month in range(1,13):
        p=Path(cache)/f'{year}{month:02d}.npz'
        with np.load(p,allow_pickle=False) as d:
            if d['grid_id'].tolist()!=grids.grid_id.tolist(): raise ValueError('Cache grid ordering mismatch.')
            if int(d['n_days'])!=calendar.monthrange(year,month)[1]: raise ValueError('Cache calendar mismatch.')
            sums+=d['sums']; counts+=d['counts']; bad+=d['quality_counts']
            clips+=d['clips']; correction+=d['corrections']; days+=int(d['n_days'])
    values=sums/days; values[:,NAMES.index('tp')]=sums[:,NAMES.index('tp')]
    values[counts!=days]=np.nan
    annual=grids[['grid_id','ar6_region']].copy(); annual['year']=year; annual['expected_days']=days
    for k,name in enumerate(METRICS):
        annual[name]=values[:,k]; annual[name+'_valid_days']=counts[:,k]
    annual['tp_zero_clipped_days']=clips; annual['tp_zero_correction_total_mm']=correction
    annual['annual_weather_complete']=np.isfinite(values).all(axis=1)
    records=[]
    for k,name in enumerate(NAMES):
        ii=np.flatnonzero(counts[:,k]!=days)
        if len(ii):
            frame=grids.iloc[ii][['grid_id','ar6_region']].copy()
            frame['year']=year; frame['variable']=name; frame['valid_days']=counts[ii,k]; frame['expected_days']=days
            for code,label in reader.QUALITY.items():
                if code: frame[label+'_days']=bad[ii,k,code]
            records.append(frame)
    cols=['grid_id','ar6_region','year','variable','valid_days','expected_days',*[v+'_days' for k,v in reader.QUALITY.items() if k]]
    issues=pd.concat(records,ignore_index=True) if records else pd.DataFrame(columns=cols)
    return annual,issues,values
