"""Supported entry point. All paths derive from this bundle, never old C:/D:."""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

def native_initial_threads(args):
    """Allocate BLAS capacity before importing libraries; runtime may only lower it.

    The pinned Linux OpenBLAS builds can crash if initialized at one thread
    and later raised for a large Cholesky factorization. Fixed mode is unchanged.
    """
    if args.parallel_mode == 'fixed':
        return args.threads
    return max(args.threads, args.worker_threads_max, args.refit_threads,
               4 if args.parallel_mode == 'adaptive' else 1)

def real_output_name(terrain_profile, run_label=''):
    """Keep different execution/timing profiles out of historical EAS attempts."""
    if run_label and (len(run_label) > 64 or not run_label.replace('_','').replace('-','').isalnum()
                      or not run_label.isascii()):
        raise ValueError('run-label must be 1..64 ASCII letters/numbers/_/- with an alphanumeric character')
    names = {'legacy_v1': 'eas', 'expanded_v2': 'eas_terrain_v2',
             'expanded_latitude_v3': 'eas_terrain_latitude_v3'}
    if terrain_profile not in names:
        raise ValueError('Unknown EAS predictor profile')
    base = names[terrain_profile]
    return base + ('_' + run_label if run_label else '')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','prepare-terrain','eas','simulate','compare-mle','replay-nn','global','status','import-eas'])
    p.add_argument('--terrain-profile',choices=['legacy_v1','expanded_v2','expanded_latitude_v3'],default='legacy_v1',
                   help='EAS only: terrain overlay, optionally geographic latitude as an FFS mean candidate; separate from frozen inputs and old results')
    p.add_argument('--run-label',default='',help='Optional isolated EAS execution/timing output namespace; does not change the scientific model')
    p.add_argument('--replicates',type=int,choices=[1,100],default=1)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--jobs',type=int,default=-2,help='GP candidate workers: -2 leaves one available CPU; 1 is serial')
    p.add_argument('--parallel-mode',choices=['staged','fixed','adaptive'],default='staged')
    p.add_argument('--memory-budget-gib',type=float,default=26.,help='Adaptive process-tree scheduling budget; not a hard OS allocation cap')
    p.add_argument('--memory-reserve-gib',type=float,default=2.,help='RAM left available inside Linux/Windows')
    p.add_argument('--method-order',choices=['alternating','mle-first','nn-first'],default='alternating')
    p.add_argument('--cpu-budget',type=int,help='Total GP CPU slots; default available CPUs minus one')
    p.add_argument('--worker-threads-max',type=int,default=4)
    p.add_argument('--refit-threads',type=int,default=4)
    p.add_argument('--new-run',action='store_true',help='EAS: retain prior results and create a fresh attempt')
    p.add_argument('--mle-jobs',type=int,default=1,help='Only the MLE estimator workers; distinct from GP --jobs')
    p.add_argument('--mle-domain',choices=['wide_domain','training_domain'],default='wide_domain')
    p.add_argument('--workers',type=int,choices=[1,2,3,4],default=2)
    p.add_argument('--experiment',default='eas_calibrated_v1')
    p.add_argument('--data-root',type=Path)
    p.add_argument('--settings',type=Path)
    p.add_argument('--skip-preprocess',action='store_true')
    p.add_argument('--source',type=Path)
    args=p.parse_args(argv)
    try: real_name=real_output_name(args.terrain_profile,args.run_label)
    except ValueError as error: p.error(str(error))
    if args.command=='prepare-terrain' and args.terrain_profile=='legacy_v1':
        p.error('prepare-terrain requires --terrain-profile expanded_v2 or expanded_latitude_v3')
    if args.command=='global' and args.terrain_profile!='legacy_v1':
        p.error('Expanded terrain/latitude profiles are EAS-only; global weather/cache contracts are unchanged')
    if not 1<=args.threads<=32:p.error('threads must be 1..32')
    if args.jobs==0:p.error('jobs cannot be zero')
    if args.mle_jobs==0:p.error('mle-jobs cannot be zero')
    if args.cpu_budget is not None and args.cpu_budget<1:p.error('cpu-budget must be positive')
    if not 1<=args.worker_threads_max<=32 or not 1<=args.refit_threads<=32:p.error('GP threads must be 1..32')
    if not args.experiment.replace('_','').replace('-','').isalnum():p.error('Use letters/numbers/_/- for experiment')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    initial_threads=native_initial_threads(args)
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[k]=str(initial_threads)
    from portable.common import ROOT,OUTPUT,read,write,sha,identity,verify_bundle,lock,latest_completed,attempt_folder,raw_root
    if args.command=='status':
        for f in sorted(OUTPUT.rglob('run_status.json')):
            print(f.relative_to(ROOT),read(f),flush=True)
        return 0
    manifest=verify_bundle()
    import torch
    if torch.version.cuda is not None:raise RuntimeError('Install the CPU-only environment with setup.ps1; CUDA build detected')
    torch.set_num_threads(args.threads)
    from gp_execution import configure,describe
    configure(args.parallel_mode,cpu_budget=args.cpu_budget,
              max_worker_threads=args.worker_threads_max,refit_threads=args.refit_threads,
              memory_budget_gib=args.memory_budget_gib,memory_reserve_gib=args.memory_reserve_gib)
    from portable.workflow import eas_inputs,execute,memory_plan,memory_guard
    inputs=eas_inputs()
    inputs[3]['n_jobs']=args.jobs
    terrain_provenance=None
    if args.terrain_profile in ('expanded_v2','expanded_latitude_v3'):
        from portable.terrain import augment_inputs
        with lock(OUTPUT):
            inputs,terrain_provenance=augment_inputs(inputs,
                (args.data_root or raw_root())/'static/era5_surface_geopotential_global_025.nc',
                OUTPUT/'prepared_inputs'/real_output_name(args.terrain_profile),
                profile=args.terrain_profile,
                base_fingerprints={k:v for k,v in manifest['files'].items() if k.startswith('inputs/eas/')})
        if args.command=='prepare-terrain':
            print(terrain_provenance,flush=True)
            return 0
    real_parent=OUTPUT/real_name
    pid=identity(dict(config=inputs[3],execution_policy=describe(),
        terrain_profile=args.terrain_profile,
        terrain_protocol_id=terrain_provenance.get('protocol_id') if terrain_provenance else None,
        bundle={k:v for k,v in manifest['files'].items()
        if k.startswith(('src/','portable/','models/','inputs/eas/'))}))
    if args.command=='check':
        from portable.inference import infer
        result=infer(inputs[0][:8])
        from gp_execution import memory_snapshot, memory_headroom
        report=dict(status='passed',bundle_files=len(manifest['files']),python=sys.version,torch=torch.__version__,
            torch_cuda_build=torch.version.cuda,nn_rows=len(result['prediction']),nn_finite=bool(__import__('numpy').isfinite(result['prediction']).all()),
            eas_grids=len(inputs[1]),external_daily_root=str(args.data_root or raw_root()),
            external_daily_root_exists=(args.data_root or raw_root()).is_dir(),
            global_grid_registry_present=(ROOT/'global_preprocessing/registries/manifest.json').is_file(),
            threads_base=args.threads,blas_initial_threads=initial_threads,
            run_label=args.run_label,real_output_parent=str(real_parent),
            terrain_profile=args.terrain_profile,terrain_provenance=terrain_provenance,
            gp_execution_policy=describe(),memory_snapshot=memory_snapshot(),memory_screen_enforced=False,
            parallel_memory_screen=(dict(mode='adaptive',legacy_full_grid_screen_applied=False,
                bounded_rolling_actual_fold_scheduling=True,**memory_headroom()) if args.parallel_mode=='adaptive'
                else memory_plan(len(inputs[1]),args.jobs)))
        write(OUTPUT/'environment_check.json',report);print(report);return 0
    # Prevent competing heavyweight runs in this independent bundle only.
    with lock(OUTPUT):
        if args.command=='import-eas':
            if args.source is None:p.error('--source required')
            import shutil
            from portable.inference import DIGEST
            from portable.comparison import calibration_configs_match, validate_terrain_source
            import pandas as pd
            source=args.source.resolve();report=read(source/'run_report.json')
            if (not report.get('workflow_completed') or report.get('region')!='EAS' or report.get('n_grids')!=7822
                or report.get('model',{}).get('sha256')!=DIGEST
                or not calibration_configs_match(report.get('config'),inputs[3])):
                raise ValueError('Expected a completed compatible EAS BOUND full run; active partial runs cannot be imported')
            for name,digest in report['output_sha256'].items():
                if Path(name).name!=name or sha(source/name)!=digest:raise ValueError('Changed source run')
            grids=pd.read_csv(source/'nn_coefficients.csv',dtype={'grid_id':str})
            if grids.grid_id.tolist()!=inputs[1].grid_id.tolist():raise ValueError('Imported GRID order differs')
            validate_terrain_source(report,inputs,terrain_provenance)
            target=attempt_folder(real_parent);target.mkdir()
            for name in report['output_sha256']:shutil.copy2(source/name,target/name)
            shutil.copy2(source/'run_report.json',target/'original_run_report.json')
            report.update(protocol_id=pid,data_role='real_data',imported_completed_run=True)
            write(target/'run_report.json',report);print('Imported completed EAS run:',target);return 0
        if args.command=='eas':
            prior=None if args.new_run else latest_completed(real_parent,pid)
            if prior:
                from portable.comparison import validate_terrain_source
                validate_terrain_source(prior[1],inputs,terrain_provenance)
                print('Already completed and verified:',prior[0]);return 0
            memory_guard(len(inputs[1]),args.jobs)
            from portable.global_run import verify_eas_geometry
            verify_eas_geometry(inputs[1],OUTPUT/'eas_geometry_check',args.threads)
            folder=attempt_folder(real_parent)
            report=execute(*inputs,folder,protocol_id=pid,threads=args.threads)
            if terrain_provenance is not None:
                from portable.comparison import terrain_input_metadata
                report.update(terrain_input_metadata(inputs,terrain_provenance))
                write(folder/'run_report.json',report)
        elif args.command=='simulate':
            from portable.simulation import simulate
            real=latest_completed(real_parent)
            if not real:raise RuntimeError('First run eas (or import a completed compatible EAS result). Calibration needs completed real data.')
            from portable.comparison import validate_real_source
            validate_real_source(real[0],inputs,terrain_provenance=terrain_provenance)
            from portable.global_run import verify_eas_geometry
            verify_eas_geometry(inputs[1],OUTPUT/'eas_geometry_check',args.threads)
            simulate(real[0],inputs,OUTPUT/'simulations'/args.experiment,args.replicates,
                     threads=args.threads,terrain_provenance=terrain_provenance)
        elif args.command=='compare-mle':
            from portable.comparison import compare_simulations
            # Old completed EAS is a calibration source, not paired timing data.
            # The comparison validates its hashes/model/settings explicitly.
            real=latest_completed(real_parent)
            if not real:raise RuntimeError('First complete the EAS real workflow; no calibration source is available.')
            compare_simulations(real[0],inputs,OUTPUT/'simulations'/args.experiment,args.replicates,
                threads=args.threads,mle_options=dict(domain=args.mle_domain,jobs=args.mle_jobs),
                method_order=args.method_order,terrain_provenance=terrain_provenance)
        elif args.command=='replay-nn':
            if args.source is None:p.error('--source paired comparison folder is required')
            from portable.global_run import verify_eas_geometry
            from portable.platform_replay import replay
            verify_eas_geometry(inputs[1],OUTPUT/'eas_geometry_check',args.threads)
            replay(args.source,inputs,OUTPUT/'simulations'/args.experiment/'platform_nn_replay',threads=args.threads)
        elif args.command=='global':
            from portable.global_run import global_workflow
            global_workflow(args.data_root or raw_root(),threads=args.threads,workers=args.workers,
                settings_path=args.settings,skip_preprocess=args.skip_preprocess,n_jobs=args.jobs)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
