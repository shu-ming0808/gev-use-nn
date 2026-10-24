"""All AR6 land regions: preprocess -> diagnostics -> explicit regional approval."""
from __future__ import annotations
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from pyproj import Geod, Proj
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from threadpoolctl import threadpool_limits
from .common import ROOT, OUTPUT, SEED, read, write, sha, identity, latest_completed, attempt_folder
from .workflow import settings, execute, validate_inputs
from .inference import infer
from research_workflows import YEARS, COEFFICIENTS
from spatial_coordinates import project_lonlat_to_km
from train_cheng_nn import atomic_csv, atomic_array
from global_pipeline.diagnose_eas_buffer_folds import variograms

sys.path.insert(0,str(ROOT/'global_preprocessing'))
from preprocess import load_registry
from bundle.coast import plan_for_region

GEOD=Geod(ellps='WGS84')

def retain(grids, candidates, test, buffer):
    """Conservative spherical neighbour screen, then exact WGS84 threshold.

    6335 km is below WGS84's minimum meridional curvature radius, so a
    geodesic within buffer cannot fall outside this angular search radius.
    This avoids an N x N distance matrix. Only the final Geod distance decides.
    """
    if buffer==0 or not len(candidates) or not len(test): return candidates.copy()
    lon=grids.longitude_180.to_numpy(); lat=grids.latitude.to_numpy()
    tree=BallTree(np.deg2rad(np.column_stack([lat[test],lon[test]])),metric='haversine')
    neighbours=tree.query_radius(np.deg2rad(np.column_stack([lat[candidates],lon[candidates]])),
                                 r=float(buffer)/6335.+1e-12)
    keep=np.ones(len(candidates),bool)
    for j,near in enumerate(neighbours):
        if len(near):
            i=candidates[j]; points=test[near]
            distances=GEOD.inv(np.full(len(points),lon[i]),np.full(len(points),lat[i]),lon[points],lat[points])[2]/1000
            keep[j]=bool(np.all(distances>buffer))
    return candidates[keep]

def geometry(grids, combinations, *, seed=SEED, min_train=100):
    xy=grids[['x_km','y_km']].to_numpy(); rows=[]; assignments=[]; outer_cache={}
    for outer,inner,buffer in combinations:
        if outer not in outer_cache:
            label=KMeans(n_clusters=outer,n_init=100,random_state=seed).fit_predict(xy)
            outer_cache[outer]=label
            assignments.extend(dict(grid_id=g,outer_folds=outer,outer_fold=int(k)) for g,k in zip(grids.grid_id,label))
        labels=outer_cache[outer]
        for fold in range(outer):
            test=np.flatnonzero(labels==fold); base=np.flatnonzero(labels!=fold)
            train=retain(grids,base,test,buffer)
            parts=[('outer',-1,train,test,len(base))]
            if len(train)>=inner:
                ilabel=KMeans(n_clusters=inner,n_init=100,random_state=seed+1000+fold).fit_predict(xy[train])
                for k in range(inner):
                    itest=train[ilabel==k]; ibase=train[ilabel!=k]
                    parts.append(('inner',k,retain(grids,ibase,itest,buffer),itest,len(ibase)))
            else:
                parts.extend(('inner',k,np.array([],int),np.array([],int),len(train)) for k in range(inner))
            for level,k,a,b,n in parts:
                rows.append(dict(outer_folds=outer,inner_folds=inner,buffer_km=buffer,
                    level=level,outer_fold=fold,inner_fold=k,n_train=len(a),n_test=len(b),
                    n_removed=n-len(a),retained_fraction=len(a)/n if n else 0.,
                    train_ids_hash=identity(grids.iloc[a].grid_id.tolist()),test_ids_hash=identity(grids.iloc[b].grid_id.tolist()),
                    pass_min_train=bool(len(a)>=min_train and len(b)>=2)))
    splits=pd.DataFrame(rows); summaries=[]
    for keys,p in splits.groupby(['outer_folds','inner_folds','buffer_km'],sort=False):
        a=p[p.level.eq('outer')]; b=p[p.level.eq('inner')]
        summaries.append(dict(zip(['outer_folds','inner_folds','buffer_km'],keys),
            outer_train_min=int(a.n_train.min()),outer_train_max=int(a.n_train.max()),
            inner_train_min=int(b.n_train.min()),inner_train_max=int(b.n_train.max()),
            min_retained_fraction=float(p.retained_fraction.min()),min_train=min_train,
            all_folds_meet_min_train=bool(p.pass_min_train.all()),
            gp_memory_screening_GB=(8*float(a.n_train.max())**2*10+2e9)/1e9))
    return pd.DataFrame(summaries),splits,pd.DataFrame(assignments)

def audit_projection(grids,crs,seed=SEED):
    xy=grids[['x_km','y_km']].to_numpy(); n=len(grids)
    rng=np.random.default_rng(seed); a,b=rng.integers(0,n,size=(2,20000)); take=a!=b; a,b=a[take],b[take]
    lon=grids.longitude_180.to_numpy();lat=grids.latitude.to_numpy()
    d=GEOD.inv(lon[a],lat[a],lon[b],lat[b])[2]/1000
    relative=np.abs(np.linalg.norm(xy[a]-xy[b],axis=1)/d-1)
    f=Proj(crs).get_factors(lon,lat)
    local=np.abs(np.r_[np.asarray(f.meridional_scale)-1,np.asarray(f.parallel_scale)-1])
    pair=float(np.max(relative));scale=float(np.max(local))
    return dict(crs=crs,pair_sample_count=len(relative),sample_max_relative_error=pair,
        all_grid_max_local_scale_error=scale,tolerance=.05,passed=bool(max(pair,scale)<=.05),
        scope='Sampled pair distances plus all-GRID local scale; not proof of all-pair distortion bound.')

def verify_eas_geometry(grids, output, threads=4):
    """Check actual portable-machine membership against the saved EAS diagnosis."""
    with threadpool_limits(limits=threads):
        summary,splits,assignments=geometry(grids,[(5,3,500.)])
    output.mkdir(parents=True,exist_ok=True)
    atomic_csv(output/'actual_splits.csv',splits)
    atomic_csv(output/'actual_summary.csv',summary)
    old=pd.read_csv(ROOT/'inputs/eas/diagnostic_splits.csv').query('outer_folds==5 and inner_folds==3 and buffer_km==500')
    keys=['level','outer_fold','inner_fold']
    old=old.set_index(keys).sort_index(); actual=splits.set_index(keys).sort_index()
    expected=old[['train_ids','test_ids']].map(lambda value:identity(str(value).split('|')))
    if (not summary.all_folds_meet_min_train.all() or
        not np.array_equal(expected.to_numpy(),actual[['train_ids_hash','test_ids_hash']].to_numpy())):
        raise ValueError('EAS fold membership differs on this machine. Actual splits saved; re-diagnose before fitting, not silently change folds.')
    write(output/'run_report.json',dict(status='passed',threads=threads,seed=SEED,
        original_memberships_matched=True,buffer_distance='WGS84 geodesic km'))

def region_inputs(region,registry,preprocessed):
    folder=preprocessed/'regions'/region
    report=read(preprocessed/'run_report.json')
    path=folder/'grid_predictors.csv'; key=path.relative_to(preprocessed).as_posix()
    if sha(path)!=report['outputs_sha256'][key]: raise ValueError('Changed global predictors')
    predictors=pd.read_csv(path,dtype={'grid_id':str})
    grids=registry.loc[registry.ar6_region.eq(region)].reset_index(drop=True).copy()
    source=pd.read_csv(ROOT/'inputs/global_annual/grid_index.csv',dtype={'grid_id':str}).set_index('grid_id')
    index=source.index.get_indexer(grids.grid_id)
    if (index<0).any(): raise ValueError('Region not covered by annual temperatures')
    paired=source.iloc[index]
    if not np.allclose(paired[['longitude_180','latitude','lsm']],grids[['longitude_180','latitude','lsm']],atol=1e-6,rtol=0):
        raise ValueError('Regional annual/predictor coordinate mismatch')
    values=np.load(ROOT/'inputs/global_annual/annual_maxima_land.npy',mmap_mode='r')[index]
    crs=plan_for_region(region,registry)['crs']
    grids[['x_km','y_km']]=project_lonlat_to_km(grids.longitude_180,grids.latitude,target_crs=crs)
    config=settings(read(folder/'candidate_groups.json'),len(grids))
    validate_inputs(values,grids,predictors,config,crs)
    return values,grids,predictors,config,crs

def diagnose(region,inputs,folder,*,threads=4):
    values,grids,predictors,config,crs=inputs
    folder.mkdir(parents=True,exist_ok=True)
    protocol=identity(dict(grids=grids.grid_id.tolist(),crs=crs,groups=config['candidate_groups'],
        values_sha256=identity(values.tolist()),predictors_sha256=identity(predictors.to_json()),version=1))
    rp=folder/'run_report.json'
    if rp.exists():
        report=read(rp)
        if report['protocol_id']!=protocol: raise ValueError('Regional diagnostics input changed')
        for name,digest in report['output_sha256'].items():
            if sha(folder/name)!=digest: raise ValueError('Regional diagnostics output changed')
        return report
    audit=audit_projection(grids,crs)
    write(folder/'projection_audit.json',audit)
    # Geometry feasibility alone is not an estimate of the correlation range.
    combinations=[(o,i,b) for o,i in ((4,3),(5,3),(5,5),(6,3)) for b in (0.,100.,300.,500.,750.,1000.)]
    with threadpool_limits(limits=threads):
        write(folder/'run_status.json',dict(status='running',stage='geometry'))
        summary,splits,assignments=geometry(grids,combinations)
        atomic_csv(folder/'summary.csv',summary);atomic_csv(folder/'splits.csv',splits)
        atomic_csv(folder/'outer_assignments.csv',assignments)
        write(folder/'run_status.json',dict(status='running',stage='five_coefficient_variograms'))
        result=infer(values);coeff=pd.DataFrame(result['prediction'],columns=COEFFICIENTS)
        elevation=predictors.set_index('grid_id').loc[grids.grid_id,'elevation_m'].to_numpy()
        vg=variograms(grids,coeff,elevation,assignments,seed=SEED,maximum=750)
        for name,table in vg.items():atomic_csv(folder/f'variogram_{name}.csv',table)
    atomic_csv(folder/'grid_index.csv',grids)
    report=dict(status='completed_diagnostics_only',protocol_id=protocol,region=region,n_grids=len(grids),
        projection=audit,approval_required=True,
        notes='Feasibility + exploratory/outer-training elevation-residual variograms; not independence proof. All retained GRID, no 800 cap.',
        output_sha256={p.name:sha(p) for p in folder.iterdir() if p.suffix=='.csv' or p.name=='projection_audit.json'})
    write(rp,report);write(folder/'run_status.json',dict(status='completed_diagnostics_only'))
    return report

def global_workflow(data_root, *, threads=4, workers=2, settings_path=None, skip_preprocess=False, n_jobs=1):
    base=OUTPUT/'global';base.mkdir(parents=True,exist_ok=True);prep=base/'preprocessing'
    if not skip_preprocess:
        write(base/'run_status.json',dict(status='running',stage='global_predictor_conversion'))
        subprocess.run([sys.executable,'-B','-u',str(ROOT/'global_preprocessing/preprocess.py'),'run',
            '--data-root',str(data_root),'--output-root',str(prep),'--regions','all','--exclude','EAS',
            '--workers',str(workers),'--resume'],check=True,cwd=ROOT/'global_preprocessing')
    if not (prep/'run_report.json').is_file():raise FileNotFoundError('Run global preprocessing first')
    registry,_,regions,_=load_registry('all','')
    approvals=read(settings_path) if settings_path else {}
    template={};states=[]
    # EAS is already prepared and diagnosed, and uses the separate eas command.
    for region in regions:
        if region=='EAS':
            states.append(dict(region=region,status='use_eas_command_prepared_and_approved'))
            continue
        try:
            write(base/'run_status.json',dict(status='running',stage='regional_diagnostics',region=region))
            inputs=region_inputs(region,registry,prep)
            diagnostic=diagnose(region,inputs,base/'diagnostics'/region,threads=threads)
            template[region]=dict(approved=False,diagnostic_protocol_id=diagnostic['protocol_id'],
                outer_folds=None,inner_folds=None,buffer_km=None,min_train=100)
            choice=approvals.get(region,{})
            if not choice.get('approved'):
                states.append(dict(region=region,status='awaiting_buffer_fold_approval'));continue
            if choice.get('diagnostic_protocol_id')!=diagnostic['protocol_id'] or not diagnostic['projection']['passed']:
                raise ValueError('Projection audit or diagnostic identity failed; do not override automatically')
            values,grids,predictors,config,crs=inputs
            table=pd.read_csv(base/'diagnostics'/region/'summary.csv')
            take=table[(table.outer_folds==choice['outer_folds'])&(table.inner_folds==choice['inner_folds'])&
                       (table.buffer_km==choice['buffer_km'])]
            if len(take)!=1 or not bool(take.iloc[0].all_folds_meet_min_train) or choice.get('min_train')!=100:
                raise ValueError('Choose an actually diagnosed feasible setting')
            config=settings(config['candidate_groups'],len(grids),outer=choice['outer_folds'],
                inner=choice['inner_folds'],buffer=choice['buffer_km'])
            config['n_jobs']=n_jobs
            pid=identity(dict(diagnostic=diagnostic['protocol_id'],config=config))
            parent=base/'runs'/region;prior=latest_completed(parent,pid)
            if prior:folder,report=prior
            else:
                folder=attempt_folder(parent)
                report=execute(values,grids,predictors,config,crs,folder,region=region,protocol_id=pid,threads=threads)
            states.append(dict(region=region,status=report['status'],output=str(folder)))
        except Exception as exc:
            states.append(dict(region=region,status='blocked',error=str(exc)))
        finally:
            atomic_csv(base/'region_status.csv',pd.DataFrame(states))
            write(base/'region_settings.template.json',template)
    write(base/'run_status.json',dict(status='review_required' if any(s['status'] in ('blocked','awaiting_buffer_fold_approval') for s in states)
        else 'regional_runs_completed_EAS_separate',regions=states,
        excluded='Ocean-only AR6 regions, polar aliases and land outside AR6 land regions are not reassigned; see registries.'))
