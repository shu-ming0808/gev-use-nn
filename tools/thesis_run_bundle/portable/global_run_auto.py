"""Opt-in global real-data loop with per-region buffered-CV settings.

The frozen D-drive global_run.py remains unchanged. Each AR6 land region gets
its own projected KMeans folds, WGS84 buffer diagnosis, training-only residual
variogram, and explicit buffer decision before its NN -> GP workflow. EAS is
excluded because its completed formal 500-km result is a separate run.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import psutil

from . import buffer_policy, global_run as base
from .common import ROOT, OUTPUT, read, write, sha, identity, latest_completed, attempt_folder
from .workflow import settings, execute
from train_cheng_nn import atomic_csv


def global_workflow(data_root, *, threads=4, workers=2, settings_path=None,
                    skip_preprocess=False, n_jobs=1):
    """Return status after continuing through all feasible regions.

    A manual --settings JSON, when supplied, still overrides the automatic
    decision only for approved regions. Failed/infeasible regions are recorded
    and do not prevent later regions from being examined.
    """
    output = OUTPUT / 'global'
    output.mkdir(parents=True, exist_ok=True)
    prep = output / 'preprocessing'
    if not skip_preprocess:
        write(output / 'run_status.json', dict(status='running', stage='global_predictor_conversion'))
        subprocess.run([sys.executable, '-B', '-u', str(ROOT / 'global_preprocessing' / 'preprocess.py'),
            'run', '--data-root', str(data_root), '--output-root', str(prep),
            '--regions', 'all', '--exclude', 'EAS', '--workers', str(workers), '--resume'],
            check=True, cwd=ROOT / 'global_preprocessing')
    report_path = prep / 'run_report.json'
    if not report_path.is_file():
        raise FileNotFoundError('Global preprocessing is not complete; resume it before --SkipPreprocess')
    report = read(report_path)
    if report.get('status') != 'completed' or report.get('complete_grids') != report.get('n_grids'):
        raise ValueError('Global preprocessing has missing/abnormal rows; inspect run_report.json before spatial modeling')
    registry, _, regions, _ = base.load_registry('all', '')
    approvals = read(settings_path) if settings_path else {}
    policy_id = identity(dict(auto_runner_sha256=sha(Path(__file__)),
        buffer_policy_sha256=sha(Path(buffer_policy.__file__)), version=1))
    template, states = {}, []
    for region in regions:
        if region == 'EAS':
            states.append(dict(region=region, status='existing_EAS_separate_500_km'))
            continue
        try:
            write(output / 'run_status.json', dict(status='running', stage='regional_diagnostics',
                 region=region, policy_id=policy_id))
            inputs = base.region_inputs(region, registry, prep)
            diagnostic = base.diagnose(region, inputs, output / 'diagnostics' / region,
                                       threads=threads)
            folder = output / 'diagnostics' / region
            geometry = pd.read_csv(folder / 'summary.csv')
            fits = pd.read_csv(folder / 'variogram_fits.csv')
            decision = buffer_policy.choose_region_buffer(geometry, fits, min_train=100)
            decision.update(policy_id=policy_id, diagnostic_protocol_id=diagnostic['protocol_id'])
            write(folder / 'buffer_decision.json', decision)
            template[region] = dict(approved=False,
                diagnostic_protocol_id=diagnostic['protocol_id'],
                outer_folds=decision['outer_folds'], inner_folds=decision['inner_folds'],
                buffer_km=decision['buffer_km'], min_train=100, basis=decision['basis'])
            choice = (approvals.get(region, {}) if settings_path else dict(
                approved=True, diagnostic_protocol_id=diagnostic['protocol_id'],
                outer_folds=decision['outer_folds'], inner_folds=decision['inner_folds'],
                buffer_km=decision['buffer_km'], min_train=100))
            if not choice.get('approved'):
                states.append(dict(region=region, status='awaiting_explicit_override'))
                continue
            if (choice.get('diagnostic_protocol_id') != diagnostic['protocol_id']
                    or not diagnostic['projection']['passed']):
                raise ValueError('Changed diagnostic protocol or failed projection audit')
            selected = geometry.loc[(geometry.outer_folds == choice['outer_folds']) &
                (geometry.inner_folds == choice['inner_folds']) &
                (geometry.buffer_km == choice['buffer_km'])]
            if (len(selected) != 1 or not bool(selected.iloc[0].all_folds_meet_min_train)
                    or choice.get('min_train') != 100):
                raise ValueError('Selected buffer/folds were not diagnosed feasible')
            # Exact GP remains full-rank. Do not cap GRID or silently fall back
            # to a different model when a region exceeds available physical RAM.
            estimated_gb = float(selected.iloc[0].gp_memory_screening_GB)
            available_gb = psutil.virtual_memory().available / 1e9
            if not np.isfinite(estimated_gb) or estimated_gb > max(0., available_gb - 1.5):
                raise MemoryError(f'Exact GP screening estimate {estimated_gb:.1f} GB; '
                    f'available {available_gb:.1f} GB. Region needs more RAM or an approved model change.')
            values, grids, predictors, config, crs = inputs
            config = settings(config['candidate_groups'], len(grids),
                outer=choice['outer_folds'], inner=choice['inner_folds'],
                buffer=choice['buffer_km'])
            config['n_jobs'] = n_jobs
            from gp_execution import describe
            protocol_id = identity(dict(diagnostic=diagnostic['protocol_id'],
                config=config, execution_policy=describe(), policy_id=policy_id))
            parent = output / 'runs' / region
            prior = latest_completed(parent, protocol_id)
            if prior:
                run_folder, result = prior
            else:
                run_folder = attempt_folder(parent)
                result = execute(values, grids, predictors, config, crs, run_folder,
                    region=region, protocol_id=protocol_id, threads=threads)
            states.append(dict(region=region, status=result['status'],
                buffer_km=float(choice['buffer_km']),
                outer_folds=int(choice['outer_folds']), inner_folds=int(choice['inner_folds']),
                output=str(run_folder)))
        except Exception as error:
            states.append(dict(region=region, status='blocked',
                error_type=type(error).__name__, error=str(error)))
        finally:
            atomic_csv(output / 'region_status.csv', pd.DataFrame(states))
            write(output / 'region_settings.template.json', template)
    final = dict(status='regional_runs_completed_EAS_separate' if all(
        row['status'] not in ('blocked', 'awaiting_explicit_override') for row in states)
        else 'review_required', policy_id=policy_id, regions=states,
        note='EAS 500-km result is separate. Unidentified ranges use a labeled geometric fallback; no independence guarantee.')
    write(output / 'run_status.json', final)
    return final
