"""D-drive sidecar: opt-in automatic global workflow, no frozen-file edits."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import psutil


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check', 'run', 'status'), default='run')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--jobs', type=int, default=-2)
    parser.add_argument('--cpu-budget', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--worker-threads-max', type=int)
    parser.add_argument('--refit-threads', type=int)
    parser.add_argument('--memory-budget-gib', type=float)
    parser.add_argument('--memory-reserve-gib', type=float, default=2.0)
    parser.add_argument('--skip-preprocess', action='store_true')
    args = parser.parse_args(argv)

    from portable.common import ROOT, OUTPUT, raw_root, verify_bundle, sha
    data_root = args.data_root or raw_root()
    if args.mode == 'status':
        path = OUTPUT / 'global' / 'run_status.json'
        print(path.read_text(encoding='utf-8') if path.is_file() else 'Global run not started')
        region_status = OUTPUT / 'global' / 'region_status.csv'
        print(f'Region status: {region_status}')
        return 0
    if not 1 <= args.workers <= 4 or not 1 <= args.threads <= 32 or args.jobs == 0:
        parser.error('workers must be 1--4, threads 1--32, and jobs nonzero')
    cpu_budget = min(args.cpu_budget, os.cpu_count() or args.cpu_budget)
    worker_threads = args.worker_threads_max or max(1, min(32, cpu_budget // 4))
    refit_threads = args.refit_threads or min(32, cpu_budget)
    memory_budget = args.memory_budget_gib or min(20.0, max(4.0,
        psutil.virtual_memory().total / 1024**3 * 0.55))
    if (cpu_budget < 1 or not 1 <= worker_threads <= 32 or not 1 <= refit_threads <= 32
            or memory_budget <= 0 or args.memory_reserve_gib < 0):
        parser.error('Invalid CPU/thread/memory resource plan')
    resources = dict(preprocessing_workers=args.workers, gp_jobs=args.jobs,
        cpu_budget=cpu_budget, candidate_worker_threads_max=worker_threads,
        refit_threads=refit_threads, memory_budget_gib=round(memory_budget, 2),
        memory_reserve_gib=args.memory_reserve_gib,
        note='Adaptive scheduling is not an OS memory cap or a CPU-utilization guarantee')
    manifest = verify_bundle()  # Original D-drive files remain frozen.
    if not data_root.is_dir():
        raise FileNotFoundError(f'Missing original daily predictors: {data_root}')
    from portable import buffer_policy, global_run_auto
    source = {name: sha(Path(module.__file__)) for name, module in (
        ('buffer_policy', buffer_policy), ('global_run_auto', global_run_auto))}
    if args.mode == 'check':
        report = dict(status='ready_for_auto_global', data_root=str(data_root),
            frozen_bundle_files=len(manifest['files']), auto_source_sha256=source,
            preprocessing_cache=str(OUTPUT / 'global' / 'preprocessing'),
            cached_months=sum(1 for _ in (OUTPUT / 'global' / 'preprocessing' / 'cache_months').glob('*.npz')),
            resource_plan=resources,
            EAS_policy='completed formal EAS run remains separate with 500-km buffer',
            note='Read-only check; no preprocessing, NN or GP started')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    # Reuse the verified bundle's environment, locking and execution policy.
    # Patch only this Python process's import binding; no D-drive frozen file
    # or manifest is overwritten.
    from portable import global_run as frozen_global_run
    frozen_global_run.global_workflow = global_run_auto.global_workflow
    import run as bundled_runner
    command = ['global', '--data-root', str(data_root), '--workers', str(args.workers),
               '--threads', str(args.threads), '--jobs', str(args.jobs),
               '--parallel-mode', 'adaptive', '--cpu-budget', str(cpu_budget),
               '--worker-threads-max', str(worker_threads),
               '--refit-threads', str(refit_threads),
               '--memory-budget-gib', str(memory_budget),
               '--memory-reserve-gib', str(args.memory_reserve_gib)]
    if args.skip_preprocess:
        command.append('--skip-preprocess')
    return bundled_runner.main(command)


if __name__ == '__main__':
    raise SystemExit(main())
