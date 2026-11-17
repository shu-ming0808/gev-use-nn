"""Separate-host EAS simulation continuation, replicates 011--020 only.

This does not edit the 001--004 or 005--010 manifests or outputs. Both NN
and MLE arms must use the same generated shared samples on their own disks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import comparison, simulation, workflow
from .common import attempt_folder, completed, identity, read, sha, write
from .split_extension import _context, _records as previous_records
from train_cheng_nn import atomic_array
from research_workflows import YEARS


FIRST = 11
LAST = 20
MANIFEST = 'extension_manifest_011_020.json'
STATUS = 'extension_status_011_020.json'


def arm_folder(root, method):
    """Keep this disk's existing arm layout; never move historical results."""
    standard = root / 'arms' / method
    nested = standard / method
    if method == 'mle' and (nested / 'replicate_001').is_dir():
        if (standard / 'replicate_001').exists():
            raise ValueError('MLE historical arms exist in both layouts; resolve manually')
        return nested
    return standard


def _verify_prior_arm(arm, method, protocol_id, truth):
    for index in range(1, 5):
        parent = arm / f'replicate_{index:03d}'
        branches = []
        for folder in sorted(parent.glob('attempt_*')):
            report = completed(folder)
            if report:
                if (report.get('replicate') != index or report.get('estimator') != method
                        or report.get('split_science_protocol_id') != protocol_id
                        or not np.array_equal(np.load(folder / 'truth.npy', allow_pickle=False), truth)):
                    raise ValueError(f'Incompatible historical {method} arm: {folder}')
                branches.append(folder)
        if len(branches) != 1:
            raise ValueError(f'Expected one verified {method} replicate {index} in {parent}')


def _samples(root, truth, protocol_id):
    manifest_path = root / 'shared' / MANIFEST
    manifest = read(manifest_path)
    if (manifest.get('version') != 1
            or manifest.get('original_protocol_id') != protocol_id
            or manifest.get('extension_code_sha256') != sha(Path(__file__))
            or manifest.get('n_grids') != len(truth)
            or manifest.get('n_years') != len(YEARS)
            or len(manifest.get('replicates', [])) != LAST - FIRST + 1):
        raise ValueError('11--20 manifest/protocol/code differs')
    prior = read(root / 'shared' / 'sample_manifest.json')['replicates']
    prior += read(root / 'shared' / 'extension_manifest_005_010.json')['replicates']
    seen = {record['annual_sha256'] for record in prior}
    if len(seen) != 10:
        raise ValueError('The 001--010 fit samples are not distinct')
    samples = []
    for index, record in enumerate(manifest['replicates'], FIRST):
        folder = root / 'shared' / f'replicate_{index:03d}'
        fit, evaluation = folder / 'annual_maxima.npy', folder / 'crps_evaluation_maxima.npy'
        if (record.get('replicate') != index or not fit.is_file() or not evaluation.is_file()
                or sha(fit) != record.get('annual_sha256')
                or sha(evaluation) != record.get('evaluation_sha256')
                or record['annual_sha256'] in seen):
            raise ValueError(f'Shared replicate {index} is missing, changed or duplicated')
        values = np.load(fit, allow_pickle=False)
        heldout = np.load(evaluation, allow_pickle=False)
        if (values.shape != (len(truth), len(YEARS)) or heldout.shape != values.shape
                or not np.isfinite(values).all() or not np.isfinite(heldout).all()
                or np.array_equal(values, heldout)):
            raise ValueError(f'Shared replicate {index} has invalid data')
        seen.add(record['annual_sha256'])
        samples.append((values, heldout, record))
    return samples


def prepare_extension(real_folder, calibration_dir, inputs, root, *,
                      mle_options=None, terrain_provenance=None):
    truth, _, pid = _context(real_folder, calibration_dir, inputs, root,
                             mle_options, terrain_provenance)
    previous_records(root, truth, pid)  # Verify prepared 005--010 samples.
    shared = root / 'shared'
    manifest_path = shared / MANIFEST
    if manifest_path.exists():
        _samples(root, truth, pid)
        return manifest_path
    records = []
    for index in range(FIRST, LAST + 1):
        folder = shared / f'replicate_{index:03d}'
        fit, evaluation = folder / 'annual_maxima.npy', folder / 'crps_evaluation_maxima.npy'
        values = simulation.draw_annual(truth, index - 1)
        heldout = simulation.draw_annual(truth, index - 1, purpose='crps_evaluation')
        if folder.exists():
            if not fit.is_file() or not evaluation.is_file():
                raise ValueError(f'Partial unmanifested replicate {index}; inspect it before continuing')
            if (not np.array_equal(np.load(fit, allow_pickle=False), values)
                    or not np.array_equal(np.load(evaluation, allow_pickle=False), heldout)):
                raise ValueError(f'Existing unmanifested replicate {index} differs')
        else:
            folder.mkdir()
            atomic_array(fit, values)
            atomic_array(evaluation, heldout)
        records.append(dict(replicate=index, annual_sha256=sha(fit),
                            evaluation_sha256=sha(evaluation)))
    write(manifest_path, dict(version=1, original_protocol_id=pid,
        extension_code_sha256=sha(Path(__file__)), n_grids=len(truth),
        n_years=len(YEARS), replicates=records))
    _samples(root, truth, pid)
    return manifest_path


def run_extension(real_folder, calibration_dir, inputs, root, method, *, threads,
                  mle_options=None, terrain_provenance=None):
    if method not in ('nn', 'mle'):
        raise ValueError('Method must be nn or mle')
    truth, options, pid = _context(real_folder, calibration_dir, inputs, root,
                                   mle_options, terrain_provenance)
    previous_records(root, truth, pid)
    samples = _samples(root, truth, pid)
    # Original four completed arms anchor the protocol. 005--010 may live on
    # another machine; their shared samples are required, their results are not.
    arm = arm_folder(root, method)
    _verify_prior_arm(arm, method, pid, truth)
    _, grids, predictors, config, crs = inputs
    status_path = arm / STATUS
    for offset, (values, evaluation, record) in enumerate(samples):
        index = FIRST + offset
        sample_id = identity(dict(protocol_id=pid, replicate=index,
            fit_sha256=record['annual_sha256'], evaluation_sha256=record['evaluation_sha256']))
        method_id = identity(dict(sample_id=sample_id, estimator=method))
        parent = arm / f'replicate_{index:03d}'
        done = []
        for folder in sorted(parent.glob('attempt_*')):
            report = completed(folder)
            if report:
                if (report.get('protocol_id') != method_id
                        or not np.array_equal(np.load(folder / 'annual_maxima.npy', allow_pickle=False), values)
                        or not np.array_equal(np.load(folder / 'truth.npy', allow_pickle=False), truth)
                        or not np.array_equal(np.load(folder / 'crps_evaluation_maxima.npy', allow_pickle=False), evaluation)):
                    raise ValueError(f'Completed replicate {index} is incompatible: {folder}')
                done.append((folder, report))
        if len(done) > 1:
            raise ValueError(f'Multiple completed attempts for replicate {index}')
        if done:
            folder, report = done[0]
            if (report.get('replicate') != index or report.get('sample_id') != sample_id
                    or report.get('estimator') != method):
                raise ValueError(f'Completed replicate {index} metadata differs')
            print(f'{method} replicate {index}/{LAST}: verified existing {folder}', flush=True)
            continue
        folder = attempt_folder(parent)
        write(status_path, dict(status='running', method=method,
            current_replicate=index, completed_extension_replicates=offset,
            total_extension_replicates=10, output=str(folder)))
        print(f'{method} replicate {index}/{LAST}: starting {folder}', flush=True)
        report = workflow.execute(values, grids, predictors, config, crs, folder,
            truth=truth, protocol_id=method_id, threads=threads, estimator=method,
            mle_options=options, evaluation_values=evaluation)
        report.update(replicate=index, sample_id=sample_id,
            split_science_protocol_id=pid, generation_seconds=0.,
            extension_manifest_sha256=sha(root / 'shared' / MANIFEST),
            run_environment=comparison.environment(threads, config['n_jobs'], options['jobs']))
        write(folder / 'run_report.json', report)
        write(status_path, dict(status='running', method=method,
            completed_extension_replicates=offset + 1,
            total_extension_replicates=10, last_output=str(folder)))
    write(status_path, dict(status='completed', method=method,
        completed_replicates=LAST, total=LAST, completed_extension_replicates=10,
        original_protocol_id=pid, extension_manifest_sha256=sha(root / 'shared' / MANIFEST)))
    return status_path
