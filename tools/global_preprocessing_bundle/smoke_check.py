"""Small local-data smoke test only; never starts the full preprocessing run."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import numpy as np
import pandas as pd
import preprocess as app
from bundle import io,reader,weather,static,coast


def run(root):
    grids,_,_,_=app.load_registry('all','')
    indices=[]
    # Real cells in several continents, both sides of the date line and polar region.
    for lat,lon in [(25,121.25),(40,-100),(0,25),(65,179.75),(65,-179.75),(-80,20)]:
        d=(grids.latitude-lat)**2+(((grids.longitude_180-lon+180)%360)-180)**2
        if abs(lon)>179:
            d=d.where(grids.longitude_180.gt(0) if lon>0 else grids.longitude_180.lt(0),np.inf)
        indices.append(int(d.idxmin()))
    sample=grids.iloc[indices].drop_duplicates('grid_id').reset_index(drop=True)
    results={'n_grids':len(sample),'grid_ids':sample.grid_id.tolist(),'months':[]}
    with TemporaryDirectory(prefix='era5_global_smoke_') as temp:
        temp=Path(temp)
        # Validate one local month against independent per-day summation and reuse.
        weather.extract_month(root,sample,1976,1,temp,app.code_hash())
        assert weather.extract_month(root,sample,1976,1,temp,app.code_hash())['reused']
        with np.load(temp/'197601.npz') as d:
            assert np.all(d['counts'][:,0]==31)
            assert np.all(d['counts'][:,1]==31)
            results['months'].append({'year':1976,'month':1,'valid_days_min_by_metric':d['counts'].min(axis=0).tolist(),
                'corrected_rain_values':int(d['clips'].sum()),'unresolved_values':int(d['quality_counts'][:,:,1:].sum())})
        # Newer ARCO source: one-day units/UTC/full-value-audit validation.
        day=pd.Timestamp('2025-01-01'); p=reader.arco_path(root,day)
        a,q,c,e=reader.read_fields(p,sample,[str(day.date())],list(weather.NAMES),'arco')
        assert np.isfinite(a).all() and not e and np.all(c==1)
        results['arco_day']='2025-01-01'; results['arco_quality_counts']={str(k):int((q==k).sum()) for k in np.unique(q)}
        terrain=static.terrain(Path(root)/'static',sample)
        assert terrain.terrain_complete.all()
        cover=static.land_cover(Path(root)/'static'/static.FILES[2],sample,temp/'cover',{'smoke':True},lambda *a,**k:None)
        assert cover.land_cover_valid_fraction.between(0,1).all()
        coastal=coast.prepare(Path(root)/'static'/static.FILES[3],sample,grids,temp/'coast',
            {'smoke':True,'code_hash':app.code_hash()},lambda *a,**k:None)
        distance=coastal.coast_distance_km.to_numpy()
        assert np.isfinite(distance).all() and (distance>=0).all()
        assert distance[sample.latitude.to_numpy() < -70].max()<1500, 'Missing Antarctic ice-front coastline'
        results['static']=sample[['grid_id','ar6_region','latitude','longitude_180']].assign(
            elevation_m=terrain.elevation_m,land_cover_valid_fraction=cover.land_cover_valid_fraction,coast_distance_km=distance).to_dict('records')
        results['coast_method']=coast.METHOD
        results['coast_projection_report']=json.loads((temp/'coast/projection_report.json').read_text(encoding='utf-8'))
    results.update(status='passed',code_hash=app.code_hash(),full_preprocessing_started=False,source_files_modified=False)
    report=app.HERE/'reports'; report.mkdir(exist_ok=True)
    io.write_json(report/'smoke_test.json',results)
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data-root',required=True,type=Path)
    run(p.parse_args().data_root.resolve())
