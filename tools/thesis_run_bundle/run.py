"""Supported entry point. All paths derive from this bundle, never old C:/D:."""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','eas','simulate','compare-mle','global','status','import-eas'])
    p.add_argument('--replicates',type=int,choices=[1,100],default=1)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--jobs',type=int,default=-2,help='GP candidate workers: -2 leaves one available CPU; 1 is serial')
    p.add_argument('--mle-jobs',type=int,default=1,help='Only the MLE estimator workers; distinct from GP --jobs')
    p.add_argument('--mle-domain',choices=['wide_domain','training_domain'],default='wide_domain')
    p.add_argument('--workers',type=int,choices=[1,2,3,4],default=2)
    p.add_argument('--experiment',default='eas_calibrated_v1')
    p.add_argument('--data-root',type=Path)
    p.add_argument('--settings',type=Path)
    p.add_argument('--skip-preprocess',action='store_true')
    p.add_argument('--source',type=Path)
    args=p.parse_args(argv)
    if not 1<=args.threads<=32:p.error('threads must be 1..32')
    if args.jobs==0:p.error('jobs cannot be zero')
    if args.mle_jobs==0:p.error('mle-jobs cannot be zero')
    if not args.experiment.replace('_','').replace('-','').isalnum():p.error('Use letters/numbers/_/- for experiment')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[k]=str(args.threads)
    from portable.common import ROOT,OUTPUT,read,write,sha,identity,verify_bundle,lock,latest_completed,attempt_folder,raw_root
    if args.command=='status':
        for f in sorted(OUTPUT.rglob('run_status.json')):
            print(f.relative_to(ROOT),read(f),flush=True)
        return 0
    manifest=verify_bundle()
    import torch
    if torch.version.cuda is not None:raise RuntimeError('Install the CPU-only environment with setup.ps1; CUDA build detected')
    torch.set_num_threads(args.threads)
    from portable.workflow import eas_inputs,execute,memory_plan,memory_guard
    inputs=eas_inputs()
    inputs[3]['n_jobs']=args.jobs
    pid=identity(dict(config=inputs[3],bundle={k:v for k,v in manifest['files'].items()
        if k.startswith(('src/','portable/','models/','inputs/eas/'))}))
    if args.command=='check':
        from portable.inference import infer
        result=infer(inputs[0][:8])
        report=dict(status='passed',bundle_files=len(manifest['files']),python=sys.version,torch=torch.__version__,
            torch_cuda_build=torch.version.cuda,nn_rows=len(result['prediction']),nn_finite=bool(__import__('numpy').isfinite(result['prediction']).all()),
            eas_grids=len(inputs[1]),external_daily_root=str(args.data_root or raw_root()),
            external_daily_root_exists=(args.data_root or raw_root()).is_dir(),
            global_grid_registry_present=(ROOT/'global_preprocessing/registries/manifest.json').is_file(),
            threads_per_worker=args.threads,memory_screen_enforced=False,
            parallel_memory_screen=memory_plan(len(inputs[1]),args.jobs))
        write(OUTPUT/'environment_check.json',report);print(report);return 0
    # Prevent competing heavyweight runs in this independent bundle only.
    with lock(OUTPUT):
        if args.command=='import-eas':
            if args.source is None:p.error('--source required')
            import shutil
            from portable.inference import DIGEST
            import pandas as pd
            source=args.source.resolve();report=read(source/'run_report.json')
            if (not report.get('workflow_completed') or report.get('region')!='EAS' or report.get('n_grids')!=7822
                or report.get('model',{}).get('sha256')!=DIGEST or report.get('config')!=inputs[3]):
                raise ValueError('Expected a completed compatible EAS BOUND full run; active partial runs cannot be imported')
            for name,digest in report['output_sha256'].items():
                if Path(name).name!=name or sha(source/name)!=digest:raise ValueError('Changed source run')
            grids=pd.read_csv(source/'nn_coefficients.csv',dtype={'grid_id':str})
            if grids.grid_id.tolist()!=inputs[1].grid_id.tolist():raise ValueError('Imported GRID order differs')
            target=attempt_folder(OUTPUT/'eas');target.mkdir()
            for name in report['output_sha256']:shutil.copy2(source/name,target/name)
            shutil.copy2(source/'run_report.json',target/'original_run_report.json')
            report.update(protocol_id=pid,data_role='real_data',imported_completed_run=True)
            write(target/'run_report.json',report);print('Imported completed EAS run:',target);return 0
        if args.command=='eas':
            prior=latest_completed(OUTPUT/'eas',pid)
            if prior:print('Already completed and verified:',prior[0]);return 0
            memory_guard(len(inputs[1]),args.jobs)
            from portable.global_run import verify_eas_geometry
            verify_eas_geometry(inputs[1],OUTPUT/'eas_geometry_check',args.threads)
            execute(*inputs,attempt_folder(OUTPUT/'eas'),protocol_id=pid,threads=args.threads)
        elif args.command=='simulate':
            from portable.simulation import simulate
            real=latest_completed(OUTPUT/'eas',pid)
            if not real:raise RuntimeError('First run eas (or import a completed compatible EAS result). Calibration needs completed real data.')
            from portable.global_run import verify_eas_geometry
            verify_eas_geometry(inputs[1],OUTPUT/'eas_geometry_check',args.threads)
            simulate(real[0],inputs,OUTPUT/'simulations'/args.experiment,args.replicates,threads=args.threads)
        elif args.command=='compare-mle':
            from portable.comparison import compare_simulations
            # Old completed EAS is a calibration source, not paired timing data.
            # The comparison validates its hashes/model/settings explicitly.
            real=latest_completed(OUTPUT/'eas')
            if not real:raise RuntimeError('First complete the EAS real workflow; no calibration source is available.')
            compare_simulations(real[0],inputs,OUTPUT/'simulations'/args.experiment,args.replicates,
                threads=args.threads,mle_options=dict(domain=args.mle_domain,jobs=args.mle_jobs))
        elif args.command=='global':
            from portable.global_run import global_workflow
            global_workflow(args.data_root or raw_root(),threads=args.threads,workers=args.workers,
                settings_path=args.settings,skip_preprocess=args.skip_preprocess,n_jobs=args.jobs)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
