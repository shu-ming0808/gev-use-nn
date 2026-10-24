"""Regional projected point-to-ocean-coast distances, in kilometres.

EAS deliberately reproduces the existing EAS LCC implementation. Other
regions use a WGS84 azimuthal-equidistant (AEQD) map centred on the full
registered region. AEQD preserves centre-to-point distances, NOT arbitrary
point-to-point distances; the measured distortion is reported, not hidden.
No nearest-sampled-coast-point distances are used as the final predictor.
"""
from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import box
from pyproj import Geod, Transformer
from bundle import io

METHOD='regional_projected_ocean_polyline_v2'
EAS_CRS='+proj=lcc +lat_1=25 +lat_2=40 +lat_0=32.5 +lon_0=120 +datum=WGS84 +units=m +no_defs'
GEOD=Geod(ellps='WGS84')
MAX_SEGMENT_KM=5.0


def layers(path):
    with zipfile.ZipFile(path) as archive:
        found={level:[n for n in archive.namelist() if n.endswith(f'GSHHS_i_L{level}.shp')]
               for level in (1,5)}
    if any(len(names)!=1 for names in found.values()):
        raise ValueError('Require L1 plus Antarctic L5 ice-front layer, not L6 grounding line.')
    return {level:f'zip://{Path(path).as_posix()}!{names[0]}' for level,names in found.items()}


def plan_for_region(region, reference):
    """Use the entire frozen region, never a smoke-test/subset-dependent CRS."""
    part=reference.loc[reference.ar6_region.eq(region)]
    if part.empty: raise ValueError(f'Missing complete projection reference for {region}.')
    if region=='EAS':
        return dict(region=region,crs=EAS_CRS,projection='EAS legacy LCC',
            bounds=[float(part.longitude_180.min()-15),float(part.latitude.min()-15),
                    float(part.longitude_180.max()+15),float(part.latitude.max()+15)],
            n_reference_grids=len(part))
    lon=np.deg2rad(part.longitude_180.to_numpy()); lat=np.deg2rad(part.latitude.to_numpy())
    centre=np.array([np.mean(np.cos(lat)*np.cos(lon)),np.mean(np.cos(lat)*np.sin(lon)),np.mean(np.sin(lat))])
    if np.linalg.norm(centre)<.1: raise ValueError(f'Region too broad for one regional map: {region}')
    lon0=float(np.rad2deg(np.arctan2(centre[1],centre[0])))
    lat0=float(np.rad2deg(np.arctan2(centre[2],np.hypot(centre[0],centre[1]))))
    # A land-density-weighted centroid can sit too far west in an archipelago.
    # Also consider the centre of the smallest wrapped longitude interval.
    # Pick the smaller worst centre-to-GRID radius, using geography alone.
    angles=np.sort(np.unique(part.longitude.to_numpy()%360))
    gaps=np.diff(np.r_[angles,angles[0]+360]); cut=int(np.argmax(gaps))
    start=float(angles[(cut+1)%len(angles)]); width=360-float(gaps[cut])
    box_lon=(start+width/2+180)%360-180
    box_lat=float((part.latitude.min()+part.latitude.max())/2)
    candidates=[(lon0,lat0),(box_lon,box_lat)]
    radii=[]
    for lo,la in candidates:
        _,_,d=GEOD.inv(np.full(len(part),lo),np.full(len(part),la),
                      part.longitude_180.to_numpy(),part.latitude.to_numpy())
        radii.append(float(np.max(d)))
    lon0,lat0=candidates[int(np.argmin(radii))]
    crs=f'+proj=aeqd +lat_0={lat0:.12f} +lon_0={lon0:.12f} +datum=WGS84 +units=m +no_defs'
    return dict(region=region,crs=crs,projection='Regional WGS84 AEQD',
                longitude_origin=lon0,latitude_origin=lat0,n_reference_grids=len(part),
                origin_rule='Smaller maximum GRID radius of spherical-vector centroid and wrapped geographic-bounds centre; full frozen registry, no weather/model fitting')


def distance_to_lines(grids, lines, crs):
    """Nearest continuous projected line segment, not nearest vertex."""
    parts=shapely.get_parts(np.asarray(lines,dtype=object))
    parts=parts[~shapely.is_empty(parts)]
    if not len(parts): raise ValueError('Empty projected coastline.')
    if not np.isfinite(shapely.get_coordinates(parts)).all(): raise ValueError('Nonfinite projected coast.')
    forward=Transformer.from_crs(4326,crs,always_xy=True)
    inverse=Transformer.from_crs(crs,4326,always_xy=True)
    x,y=forward.transform(grids.longitude_180.to_numpy(),grids.latitude.to_numpy())
    if not np.isfinite([x,y]).all(): raise ValueError('Nonfinite projected GRID.')
    points=shapely.points(x,y)
    tree=shapely.STRtree(parts)
    nearest=parts[tree.nearest(points)]
    distances=shapely.distance(points,nearest)/1000
    links=shapely.shortest_line(points,nearest)
    foot=shapely.get_point(links,1)
    lon,lat=inverse.transform(shapely.get_x(foot),shapely.get_y(foot))
    _,_,metres=GEOD.inv(grids.longitude_180.to_numpy(),grids.latitude.to_numpy(),lon,lat)
    geodesic=np.asarray(metres)/1000
    relative=np.zeros(len(grids)); meaningful=geodesic>1e-6
    relative[meaningful]=100*np.abs(distances[meaningful]/geodesic[meaningful]-1)
    diagnostics=dict(max_endpoint_distance_discrepancy_pct=float(relative.max()),
        p95_endpoint_distance_discrepancy_pct=float(np.quantile(relative,.95)),
        max_endpoint_distance_discrepancy_km=float(np.abs(distances-geodesic).max()),
        distortion_check='Projected distance versus WGS84 geodesic to the SAME inverse-projected nearest foot; not a proof of the geodesic-nearest coast or all-direction distortion.',
        projection_review_recommended=bool(relative.max()>5),
        review_threshold_pct=5.0)
    return distances,diagnostics


def eas_distance(path, grids, plan):
    """Same EAS metric; expand the search window if inland cells need it."""
    source=layers(path)[1]; initial=plan['bounds']
    points=gpd.GeoSeries(gpd.points_from_xy(grids.longitude_180,grids.latitude),crs=4326).to_crs(EAS_CRS)
    initial_failed=0
    for extra in (0,10,15):
        bounds=[initial[0]-extra,initial[1]-extra,initial[2]+extra,initial[3]+extra]
        if bounds[0]<=-180 or bounds[2]>=180 or bounds[1]<=-90 or bounds[3]>=90:
            raise ValueError('EAS search window outside its regional domain.')
        shape=gpd.read_file(source,bbox=tuple(bounds))
        if shape.crs is None: raise ValueError('Missing coastline CRS.')
        lines=shape.to_crs(4326).geometry.boundary.intersection(box(*bounds))
        lines=lines.loc[~lines.is_empty]
        projected=gpd.GeoSeries(lines,crs=4326).to_crs(EAS_CRS)
        distance,report=distance_to_lines(grids,projected,EAS_CRS)
        # Densify the search boundary so its projection follows the geographic
        # window, rather than replacing long projected parallels by chords.
        edge=gpd.GeoSeries([shapely.segmentize(box(*bounds).boundary,.1)],crs=4326).to_crs(EAS_CRS).iloc[0]
        clearance=points.distance(edge).to_numpy()/1000-distance
        if extra==0: initial_failed=int((clearance<=0).sum())
        if np.isfinite(distance).all() and np.all(clearance>0):
            report.update(window_min_clearance_km=float(clearance.min()),source_layers=[1],
                search_bounds=bounds,search_margin_degrees=15+extra,
                initial_window_guard_failures=initial_failed,
                compatibility='Existing EAS LCC metric; full-region initial bounds; window expands when required, not a change of distance units or projection.')
            return distance,report
    raise ValueError('EAS coastline window too small even after expansion.')


def ocean_segments(path, progress):
    """Load real ocean edges; no lakes or artificial date-line/pole closures."""
    frames=[]
    for level,source in layers(path).items():
        frame=gpd.read_file(source)
        if frame.crs is None: raise ValueError('Coastline CRS missing.')
        frames.append(frame.to_crs(4326))
    repaired=int((~frames[1].geometry.is_valid).sum())
    frames[1]['geometry']=frames[1].geometry.make_valid()
    ice=frames[1].geometry.union_all()
    south=frames[0].geometry.bounds.miny.lt(-60)
    repaired+=int((~frames[0].loc[south].geometry.is_valid).sum())
    frames[0].loc[south,'geometry']=frames[0].loc[south].geometry.make_valid().difference(ice)
    frame=pd.concat(frames,ignore_index=True)
    def polygons(g):
        if g is None or g.is_empty: return
        if g.geom_type=='Polygon': yield g
        elif hasattr(g,'geoms'):
            for child in g.geoms: yield from polygons(child)
    all_segments=[]
    for j,g in enumerate(frame.geometry):
        for poly in polygons(g):
            segments=split_edges(np.asarray(poly.exterior.coords)[:,:2])
            if len(segments): all_segments.append(segments)
        if (j+1)%2000==0:
            progress('coastline_loading',completed_polygons=j+1,total_polygons=len(frame))
    segments=np.concatenate(all_segments)
    return segments,dict(topology_repaired_polygons=repaired,source_layers=[1,5],
        antarctic_coast='L5 ice front, not L6 grounding line; enclosed L1 rock removed',
        raw_source_modified=False,max_geodesic_segment_km=MAX_SEGMENT_KM)


def split_edges(coords):
    """Densify WGS84 edges before projection; return separate real segments."""
    a=np.asarray(coords,float)
    if len(a)<2: return np.empty((0,2,2))
    p,q=a[:-1],a[1:]
    seam=np.isclose(abs(p[:,0]),180,rtol=0,atol=1e-10)&np.isclose(abs(q[:,0]),180,rtol=0,atol=1e-10)
    pole=(abs(p[:,1])>89.999999)|(abs(q[:,1])>89.999999)
    p,q=p[~seam&~pole],q[~seam&~pole]
    if not len(p): return np.empty((0,2,2))
    az,_,length=GEOD.inv(p[:,0],p[:,1],q[:,0],q[:,1])
    count=np.maximum(1,np.ceil(length/(MAX_SEGMENT_KM*1000)).astype(int))
    pieces=[]
    for n in np.unique(count):
        take=np.flatnonzero(count==n)
        if n==1:
            pieces.append(np.stack([p[take],q[take]],axis=1)); continue
        f=np.arange(n+1)/n
        lon,lat,_=GEOD.fwd(np.repeat(p[take,0],n+1),np.repeat(p[take,1],n+1),
                          np.repeat(az[take],n+1),(length[take,None]*f).ravel())
        vertices=np.stack([lon,lat],axis=-1).reshape(-1,n+1,2)
        pieces.append(np.stack([vertices[:,:-1],vertices[:,1:]],axis=2).reshape(-1,2,2))
    return np.concatenate(pieces)


def regional_distance(segments, grids, reference, plan):
    """Project an ample radial neighbourhood; fail if its edge could matter."""
    forward=Transformer.from_crs(4326,plan['crs'],always_xy=True)
    # AEQD radius equals ellipsoidal distance from its map centre. Limit the
    # map to a regional cap, excluding the antipodal singularity and its chords.
    x,y=forward.transform(reference.longitude_180.to_numpy(),reference.latitude.to_numpy())
    radius=float(np.hypot(x,y).max()/1000)
    centre_lon=plan['longitude_origin']; centre_lat=plan['latitude_origin']
    flat=segments.reshape(-1,2)
    _,_,r=GEOD.inv(np.full(len(flat),centre_lon),np.full(len(flat),centre_lat),flat[:,0],flat[:,1])
    endpoint_radius=np.asarray(r).reshape(-1,2)/1000
    gx,gy=forward.transform(grids.longitude_180.to_numpy(),grids.latitude.to_numpy())
    grid_radius=np.hypot(gx,gy)/1000
    for margin in (2000.,3500.,5000.):
        cap=radius+margin
        if cap>12000: raise ValueError('Region too broad for safe regional coast projection.')
        # Every geodesic edge <=5 km. Edges beyond cap+5 at both ends cannot
        # enter the cap. Include the boundary band; do not manufacture a coast
        # along the circle that clips the search neighbourhood.
        take=endpoint_radius.min(axis=1)<=cap+MAX_SEGMENT_KM
        selected=segments[take]
        if not len(selected): continue
        px,py=forward.transform(selected[:,:,0],selected[:,:,1])
        lines=shapely.linestrings(np.stack([px,py],axis=-1))
        distance,report=distance_to_lines(grids,lines,plan['crs'])
        clearance=cap-grid_radius-distance-4*MAX_SEGMENT_KM
        if np.isfinite(distance).all() and np.all(clearance>0):
            report.update(search_cap_radius_km=cap,window_min_clearance_km=float(clearance.min()),
                n_coast_segments=int(take.sum()),n_reference_grids=len(reference))
            return distance,report
    raise ValueError('Coastline search window too small; do not accept distances.')


def prepare(path, grids, reference, output, identity, progress):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    segments=None; topology=None; results=[]; reports={}
    for region,part in grids.groupby('ar6_region',sort=True):
        plan=plan_for_region(region,reference)
        ids=dict(base=identity,method=METHOD,plan=plan,grid_ids=part.grid_id.tolist())
        target=output/f'{region}.csv'; meta=output/f'{region}.json'
        old=json.loads(meta.read_text(encoding='utf-8')) if meta.is_file() else {}
        if target.is_file() and old.get('identity')==ids and old.get('sha256')==io.sha256(target):
            result=pd.read_csv(target); report=old['diagnostics']
        else:
            progress('coastline_projection',region=region,n_grids=len(part))
            if region=='EAS': distance,report=eas_distance(path,part,plan)
            else:
                if segments is None: segments,topology=ocean_segments(path,progress)
                distance,report=regional_distance(segments,part,reference.loc[reference.ar6_region.eq(region)],plan)
                report.update(topology)
            result=pd.DataFrame(dict(grid_id=part.grid_id.to_numpy(),coast_distance_km=distance))
            io.write_csv(target,result)
            io.write_json(meta,dict(identity=ids,sha256=io.sha256(target),diagnostics=report))
        results.append(result); reports[region]={**plan,**report}
        progress('coastline_distance',completed_regions=len(results),total_regions=grids.ar6_region.nunique())
    io.write_json(output/'projection_report.json',dict(method=METHOD,regions=reports,
        review_regions=[r for r,v in reports.items() if v['projection_review_recommended']],
        projection_warning='Same projected point-to-polyline workflow; regional CRSs differ. No global exact-geodesic claim. Does not configure GP/folds/buffer.'))
    return pd.concat(results,ignore_index=True).set_index('grid_id').loc[grids.grid_id].reset_index()
