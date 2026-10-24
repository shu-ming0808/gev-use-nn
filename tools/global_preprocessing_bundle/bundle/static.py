"""Global static predictors; regional projected distances match the EAS approach."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.ndimage import convolve, maximum_filter, minimum_filter
from bundle import io, weather, coast

CLASS_GROUPS={'agriculture':{10,11,12,20,30},
              'forest':{50,60,61,62,70,71,72,80,81,82,90,160,170},
              'urban':{190},'water':{210}}
FILES=('era5_land_sea_mask_global_025.nc','era5_surface_geopotential_global_025.nc',
       'esa_cci_land_cover_2000_global_300m.nc','gshhg-shp-2.3.7.zip')


def terrain(root,grids):
    with io.open_nc(Path(root)/FILES[0]) as ds:
        lsm=io.gather(ds['lsm'],io.spatial_indices(ds,grids),len(grids))
        if not np.allclose(lsm,grids.lsm,rtol=0,atol=1e-6): raise ValueError('Registry/source LSM mismatch.')
    with io.open_nc(Path(root)/FILES[1]) as ds:
        if getattr(ds['z'],'units','') not in ('m**2 s**-2','m2 s-2','m^2 s^-2'): raise ValueError('Geopotential units.')
        lat=io.numeric(ds['latitude'][:]); lon=io.numeric(ds['longitude'][:])
        if not np.allclose(lat,np.linspace(90,-90,721)) or not np.allclose(lon,np.arange(1440)*.25):
            raise ValueError('Terrain requires the complete canonical 0.25-degree ERA5 grid.')
        z=io.numeric(ds['z'][:]).squeeze()/9.80665
        if z.shape!=(721,1440): raise ValueError('Terrain array shape.')
    # Longitude wraps. Latitude does not wrap/reflect across a pole.
    valid=np.isfinite(z)
    padded=np.pad(np.where(valid,z,0),((2,2),(0,0)),constant_values=0)
    padded=np.pad(padded,((0,0),(2,2)),mode='wrap')
    vp=np.pad(valid.astype(int),((2,2),(0,0)),constant_values=0)
    vp=np.pad(vp,((0,0),(2,2)),mode='wrap')
    kernel=np.ones((5,5)); kernel[2,2]=0
    sums=convolve(padded,kernel,mode='constant')[2:-2,2:-2]
    count=convolve(vp,kernel,mode='constant')[2:-2,2:-2]
    filled=np.pad(z,((2,2),(0,0)),constant_values=np.nan)
    filled=np.pad(filled,((0,0),(2,2)),mode='wrap')
    hi=maximum_filter(np.nan_to_num(filled,nan=-np.inf),size=5,mode='constant',cval=-np.inf)[2:-2,2:-2]
    lo=minimum_filter(np.nan_to_num(filled,nan=np.inf),size=5,mode='constant',cval=np.inf)[2:-2,2:-2]
    r=np.rint((90-grids.latitude.to_numpy())/.25).astype(int)
    c=np.rint(grids.longitude.to_numpy()/.25).astype(int)%1440
    ok=valid[r,c]&(count[r,c]==24)
    return pd.DataFrame(dict(grid_id=grids.grid_id,elevation_m=z[r,c],
        terrain_025_tpi_m=np.where(ok,z[r,c]-sums[r,c]/24,np.nan),
        terrain_025_local_relief_m=np.where(ok,hi[r,c]-lo[r,c],np.nan),
        terrain_complete=ok))


def footprint(latitude,longitude):
    # At poles clip the geographical footprint to +/-90; longitude is periodic.
    top=min(90.,latitude+.125); bottom=max(-90.,latitude-.125)
    first=int(round((90-top)*360)); last=int(round((90-bottom)*360))
    rr=np.arange(first,last,dtype=int)
    left=longitude-.125
    start=int(round((left+180)*360))%129600
    cc=(start+np.arange(90))%129600
    return rr,cc


def fractions(data,rr,known):
    north=90-rr/360; south=90-(rr+1)/360
    weight=np.broadcast_to((np.sin(np.deg2rad(north))-np.sin(np.deg2rad(south)))[:,None],data.shape)
    valid=np.isin(data,known)&(data!=0)
    total=weight.sum(); covered=weight[valid].sum()
    result={'land_cover_valid_fraction':float(covered/total), 'land_cover_pixel_count':int(valid.sum())}
    used=set()
    for name,codes in CLASS_GROUPS.items():
        result[name+'_ratio']=float(weight[valid&np.isin(data,list(codes))].sum()/covered) if covered else np.nan
        used.update(codes)
    result['other_ratio']=float(weight[valid&~np.isin(data,list(used))].sum()/covered) if covered else np.nan
    return result


def land_cover(path,grids,parts,identity,progress):
    parts=Path(parts); parts.mkdir(parents=True,exist_ok=True)
    plan={}
    for j,g in enumerate(grids.itertuples()):
        rr,cc=footprint(g.latitude,g.longitude_180)
        plan.setdefault((int(rr[0]//2025),int(cc[0]//2025)),[]).append((j,rr,cc))
    result=[]
    with io.open_nc(path) as ds:
        if str(getattr(ds,'time_coverage_start',''))!='20000101': raise ValueError('CCI must be year 2000.')
        lat=io.numeric(ds['lat'][:]); lon=io.numeric(ds['lon'][:])
        if len(lat)!=64800 or len(lon)!=129600 or not np.allclose(lat,90-(np.arange(64800)+.5)/360,atol=1e-8):
            raise ValueError('Unexpected CCI global latitude grid.')
        if not np.allclose(lon,-180+(np.arange(129600)+.5)/360,atol=1e-8): raise ValueError('Unexpected CCI longitude grid.')
        var=ds['lccs_class']; known=np.asarray(var.flag_values)
        var.set_var_chunk_cache(128*1024*1024,1009,.75)
        for number,(key,items) in enumerate(sorted(plan.items()),1):
            target=parts/f'cci_{key[0]:02d}_{key[1]:02d}.csv'; meta=target.with_suffix('.json')
            ident=dict(base=identity,grid_ids=grids.iloc[[j for j,_,_ in items]].grid_id.tolist())
            old=json.loads(meta.read_text(encoding='utf-8')) if meta.is_file() else {}
            if target.is_file() and old.get('identity')==ident and old.get('sha256')==io.sha256(target):
                part=pd.read_csv(target)
            else:
                rr=np.unique(np.concatenate([r for _,r,_ in items])); cc=np.unique(np.concatenate([c for _,_,c in items]))
                raw=io.numeric(var[0,rr,cc]); rows=[]
                for j,r,c in items:
                    data=raw[np.ix_(np.searchsorted(rr,r),np.searchsorted(cc,c))]
                    rows.append(dict(grid_id=grids.grid_id.iloc[j],**fractions(data,r,known)))
                part=pd.DataFrame(rows); io.write_csv(target,part)
                io.write_json(meta,dict(identity=ident,sha256=io.sha256(target)))
            result.append(part)
            progress('land_cover',completed_tiles=number,total_tiles=len(plan))
    return pd.concat(result,ignore_index=True).set_index('grid_id').loc[grids.grid_id].reset_index()


def prepare(root,grids,output,code_hash,progress,reference):
    root,output=Path(root),Path(output); output.mkdir(parents=True,exist_ok=True)
    # Independent content hashes once per static file; no hard-coded D/F identity.
    sources={name:io.sha256(root/name) for name in FILES}
    identity=dict(version='global_static_v2',source_hashes=sources,code_hash=code_hash,
                  grid_hash=io_hash_grid(grids),coast_method=coast.METHOD)
    target=output/'static_predictors.csv'; meta=output/'static_report.json'
    old=json.loads(meta.read_text(encoding='utf-8')) if meta.is_file() else {}
    if target.is_file() and old.get('identity')==identity and old.get('sha256')==io.sha256(target):
        return pd.read_csv(target)
    progress('terrain')
    result=terrain(root,grids)
    result=result.merge(land_cover(root/FILES[2],grids,output/'land_cover_parts',identity,progress),on='grid_id',validate='one_to_one')
    distances=coast.prepare(root/FILES[3],grids,reference,output/'coast_regions',
        dict(source=sources[FILES[3]],code_hash=code_hash),progress)
    result=result.merge(distances,on='grid_id',validate='one_to_one')
    numerical=result.drop(columns=['grid_id','terrain_complete']).to_numpy(float)
    result['static_complete']=np.isfinite(numerical).all(axis=1)&result.terrain_complete&result.land_cover_valid_fraction.ge(1-1e-10)
    if sources!={name:io.sha256(root/name) for name in FILES}: raise RuntimeError('Static sources changed.')
    io.write_csv(target,result)
    issues=result.loc[~result.static_complete].copy()
    issues['issue']=np.where(~issues.terrain_complete,'incomplete_terrain_neighbourhood','CCI_missing_or_unknown_class')
    io.write_csv(output/'static_missing_report.csv',issues)
    io.write_json(meta,dict(identity=identity,sha256=io.sha256(target),n_grids=len(grids),complete_grids=int(result.static_complete.sum()),
        definitions=dict(terrain='0.25-degree geopotential / 9.80665, 5x5 source-cell TPI/relief; longitude wrap; incomplete polar halos flagged',
        land_cover='Fixed CCI 2000 spherical-area weighted 0.25-degree footprint; dateline wrap; polar footprint clipped to +/-90',
        coast='GSHHG intermediate L1 plus Antarctic L5 ice-front (not L6); regional WGS84 projection, shortest point-to-continuous-polyline km, not nearest coast sample point; not exact geodesic',
        comparison='Same projected-distance workflow as EAS. EAS preserves its original LCC and clipping; other regions use local AEQD. See coast_regions/projection_report.json for distortion checks. No GP projection/folds created.')))
    return result


def io_hash_grid(grids):
    import hashlib
    return hashlib.sha256(grids.to_csv(index=False).encode()).hexdigest()
