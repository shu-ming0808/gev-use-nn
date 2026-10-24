"""Coast-only QA: all EAS cells versus legacy, plus 16 cells per other region."""
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import preprocess as app
from bundle import io,coast,static


def run(root):
    reference,_,_,_=app.load_registry('all','')
    parts=[]
    for name,part in reference.groupby('ar6_region',sort=True):
        parts.append(part if name=='EAS' else part.iloc[np.unique(np.linspace(0,len(part)-1,16).astype(int))])
    sample=pd.concat(parts,ignore_index=True)
    source=Path(root)/'static'/static.FILES[3]
    digest=io.sha256(source)
    with TemporaryDirectory(prefix='era5_projected_coast_qa_') as folder:
        def progress(stage,**values):
            if stage=='coastline_distance': print(json.dumps(dict(stage=stage,**values)),flush=True)
        result=coast.prepare(source,sample,reference,folder,{'code_hash':app.code_hash(),'source_sha256':digest},progress)
        report=json.loads((Path(folder)/'projection_report.json').read_text(encoding='utf-8'))
        eas=reference.loc[reference.ar6_region.eq('EAS')]
        # Independent expression copied verbatim in substance from the active
        # EAS routine: union of clipped boundaries, then GeoSeries.distance.
        bounds=(float(eas.longitude_180.min()-15),float(eas.latitude.min()-15),
                float(eas.longitude_180.max()+15),float(eas.latitude.max()+15))
        shape=gpd.read_file(coast.layers(source)[1],bbox=bounds)
        lines=shape.to_crs(4326).geometry.boundary.intersection(box(*bounds))
        lines=lines.loc[~lines.is_empty]
        union=gpd.GeoSeries(lines,crs=4326).to_crs(coast.EAS_CRS).union_all()
        points=gpd.GeoSeries(gpd.points_from_xy(eas.longitude_180,eas.latitude),crs=4326).to_crs(coast.EAS_CRS)
        legacy=points.distance(union).to_numpy()/1000
        candidate=result.set_index('grid_id').loc[eas.grid_id,'coast_distance_km'].to_numpy()
        delta=np.abs(candidate-legacy)
        if not np.allclose(candidate,legacy,rtol=0,atol=1e-8): raise AssertionError('EAS compatibility failure')
        if not np.isfinite(result.coast_distance_km).all() or result.coast_distance_km.lt(0).any():
            raise AssertionError('Invalid projected coast distance')
        report.update(status='passed',code_hash=app.code_hash(),source_sha256=digest,
            verified_utc=io.utc_now(),n_regions=sample.ar6_region.nunique(),n_grids=len(sample),
            eas_compatibility=dict(n_grids=len(eas),max_absolute_difference_km=float(delta.max()),
                                   tolerance_km=1e-8,passed=True),
            scope='All 7,822 EAS grids; 16 lat/lon-ordered sample grids per other region. Coast-only, no daily extraction or NN/GP. Distortion diagnostics outside EAS are sampled, not all-grid guarantees.',
            full_preprocessing_started=False,source_files_modified=False)
        io.write_json(app.HERE/'reports/coast_projection_validation.json',report)
        print(json.dumps({k:v for k,v in report.items() if k!='regions'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    run(p.parse_args().data_root)
