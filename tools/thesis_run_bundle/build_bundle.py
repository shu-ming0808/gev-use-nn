"""Build an independent bundle; do not edit or copy active workflow outputs."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone

SOURCE=Path(__file__).resolve().parent
REPO=SOURCE.parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def put(src,dst):
    if not src.is_file(): raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.is_file() or sha(src)!=sha(dst): shutil.copy2(src,dst)
def build(dest, uv):
    dest=dest.resolve()
    if dest==REPO or REPO in dest.parents: raise ValueError('Bundle must be outside original checkout')
    dest.mkdir(parents=True,exist_ok=True)
    protected={p:sha(p) for p in (REPO/'src').rglob('*.py')}
    for p in SOURCE.rglob('*'):
        if p.is_file() and not set(p.relative_to(SOURCE).parts)&{'__pycache__','.venv','.pytest_cache'}:
            put(p,dest/p.relative_to(SOURCE))
    for p in protected: put(p,dest/p.relative_to(REPO))
    put(REPO/'notebooks/cheng_NN.ipynb',dest/'notebooks/cheng_NN.ipynb')
    put(REPO/'simulation/taiwan_validation/nn_mle_full_workflow_comparison.ipynb',
        dest/'notebooks/nn_mle_full_workflow_comparison.ipynb')
    put(REPO/'results/cheng_nn_21d/reparameterization_100k_20261012/reparameterized/best.pt',dest/'models/bound.pt')
    for n in ('annual_maxima.npy','sample_location_scale.npy'):
        p=Path('data/simulated/cheng_nn_mixed_100k_20261006/train')/n
        put(REPO/p,dest/p)
    diag=REPO/'results/global/eas/buffer_fold_diagnostics_20261021_v1'
    for n in ('grid_index.csv','annual_maxima.npy','years.npy','summary.csv','splits.csv','projection_audit.json','run_report.json'):
        put(diag/n,dest/'inputs/eas'/('diagnostic_'+n if n in ('run_report.json','summary.csv','splits.csv') else n))
    prep=Path('D:/論文資料/ERA5/eas_predictors_1976_2025/run_20261021_projected_v2')
    report=json.loads((prep/'run_report.json').read_text(encoding='utf-8'))
    if report['status']!='completed' or report['complete_grids']!=7822 or report['annual_missing_rows']!=0:
        raise ValueError('EAS predictor preparation is not complete')
    for n in ('grid_predictors.csv','candidate_groups.json','run_report.json','protocol.json','static_report.json'):
        if n in report['output_sha256'] and sha(prep/n)!=report['output_sha256'][n]:raise ValueError(n)
        put(prep/n,dest/'inputs/eas'/('predictor_'+n if n in ('run_report.json','protocol.json','static_report.json') else n))
    global_cache=Path('D:/論文資料/ERA5/cheng_nn_21d_1976_2025/run_20261011_01')
    r=json.loads((global_cache/'run_report.json').read_text(encoding='utf-8'))
    for n in ('annual_maxima_land.npy','grid_index.csv'):
        if sha(global_cache/n)!=r['output_sha256'][n]:raise ValueError('Global annual cache changed: '+n)
        put(global_cache/n,dest/'inputs/global_annual'/n)
    put(global_cache/'run_report.json',dest/'inputs/global_annual/run_report.json')
    preprocessor=Path('D:/全球前處理')
    for sub in ('bundle','registries'):
        for p in (preprocessor/sub).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:put(p,dest/'global_preprocessing'/p.relative_to(preprocessor))
    for name in ('preprocess.py','progress_fix_compatibility.json'):
        put(REPO/'tools/global_preprocessing_bundle'/name,dest/'global_preprocessing'/name)
    put(uv,dest/'bin/uv.exe')
    for p,digest in protected.items():
        if sha(p)!=digest:raise RuntimeError('Original source changed during packaging: '+str(p))
    provenance=dict(created_utc=datetime.now(timezone.utc).isoformat(),original_repo=str(REPO),
        original_sources_unchanged=True,current_running_outputs_copied=False,
        uv_source=str(uv),uv_sha256=sha(uv),
        dependencies='CPU lock; no .venv, Python interpreter, credentials or raw daily files copied',
        external_data_relative='../論文資料/ERA5/global_predictors_1976_2025')
    (dest/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*')
        if p.is_file() and p!=dest/'manifest.json' and
        not set(p.relative_to(dest).parts)&{'outputs','outputs_wsl','archives','qa','__pycache__','.venv','.pytest_cache'}}
    (dest/'manifest.json').write_text(json.dumps(dict(version=1,files=manifest),indent=2),encoding='utf-8')
    print(json.dumps(dict(destination=str(dest),files=len(manifest),bytes=sum((dest/n).stat().st_size for n in manifest)),ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dest',type=Path,required=True);p.add_argument('--uv',type=Path,required=True)
    a=p.parse_args();build(a.dest,a.uv)
