"""Windows CPU continuation: prepare or run EAS replicates 005--010."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=('prepare', 'nn', 'mle', 'status'))
    parser.add_argument('--threads', type=int, default=20)
    parser.add_argument('--jobs', type=int, default=-1)
    parser.add_argument('--mle-jobs', type=int, default=20)
    parser.add_argument('--cpu-budget', type=int, default=20)
    parser.add_argument('--worker-threads-max', type=int, default=20)
    parser.add_argument('--refit-threads', type=int, default=20)
    args = parser.parse_args(argv)
    if min(args.threads, args.mle_jobs, args.cpu_budget,
           args.worker_threads_max, args.refit_threads) < 1:
        parser.error('Thread and worker counts must be positive')
    if args.jobs not in (-1, -2) and args.jobs < 1:
        parser.error('Jobs must be -1, -2 or positive')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    initial_threads = max(args.threads, args.worker_threads_max, args.refit_threads)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(initial_threads)

    from portable.common import ROOT, latest_completed, lock, verify_bundle
    root = ROOT / 'outputs_wsl' / 'simulations' / 'eas_four_pair_rl_v1' / 'split_comparison_v1'
    if args.method == 'status':
        for method in ('nn', 'mle'):
            path = root / 'arms' / method / 'extension_status.json'
            print(method, path.read_text(encoding='utf-8') if path.is_file() else 'not started')
        return 0
    manifest = verify_bundle()  # Protect all frozen model/science inputs and code.
    import torch
    if torch.version.cuda is not None:
        raise RuntimeError('Expected the Windows CPU-only PyTorch environment')
    torch.set_num_threads(args.threads)
    from gp_execution import configure
    configure('staged', cpu_budget=args.cpu_budget,
              max_worker_threads=args.worker_threads_max,
              refit_threads=args.refit_threads)
    from portable.workflow import eas_inputs
    from portable.terrain import augment_inputs
    from portable.split_extension import prepare_extension, run_extension
    from portable.resumable_cv import checkpointed_execute
    from portable import workflow

    inputs = eas_inputs()
    inputs[3]['n_jobs'] = args.jobs
    static = ROOT / 'inputs' / 'external_era5' / 'static' / 'era5_surface_geopotential_global_025.nc'
    if not static.is_file():
        raise FileNotFoundError(f'Missing bundled terrain source: {static}')
    output_root = ROOT / 'outputs_wsl'
    with lock(output_root):
        inputs, terrain_provenance = augment_inputs(
            inputs, static, output_root / 'prepared_inputs' / 'eas_terrain_latitude_v3',
            profile='expanded_latitude_v3',
            base_fingerprints={key: value for key, value in manifest['files'].items()
                               if key.startswith('inputs/eas/')})
        real_parent = output_root / 'eas_terrain_latitude_v3_ram16_mem8_v1'
        real = latest_completed(real_parent)
        if real is None:
            raise RuntimeError(f'No completed EAS source under {real_parent}')
        calibration = output_root / 'simulations' / 'latitude_v3_ram16_mem8_cpu20_v1' / 'calibration'
        options = dict(domain='wide_domain', jobs=args.mle_jobs)
        if args.method == 'prepare':
            path = prepare_extension(real[0], calibration, inputs, root,
                mle_options=options, terrain_provenance=terrain_provenance)
            print(f'Prepared/verified six new paired samples: {path}', flush=True)
        else:
            original_execute = workflow.execute
            workflow.execute = checkpointed_execute(original_execute)
            try:
                path = run_extension(real[0], calibration, inputs, root, args.method,
                    threads=1 if args.method == 'mle' else args.threads, mle_options=options,
                    terrain_provenance=terrain_provenance)
            finally:
                workflow.execute = original_execute
            print(f'Completed extension: {path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
