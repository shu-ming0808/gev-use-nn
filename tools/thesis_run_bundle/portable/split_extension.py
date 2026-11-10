"""Append paired simulation replicates 005--010 without changing the frozen four-run protocol.

The first four samples and their results remain untouched. This extension has
its own manifest; both estimator arms must use the same prepared six arrays.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import comparison, simulation, workflow
from .common import attempt_folder, completed, identity, read, sha, write
from .split_comparison import (
    _calibration, _load_shared, _method_options, _science_protocol,
    _write_or_check_protocol,
)
from train_cheng_nn import atomic_array
from research_workflows import YEARS


FIRST = 5
LAST = 10
MANIFEST = 'extension_manifest_005_010.json'


def _context(real_folder, calibration_dir, inputs, root, mle_options, terrain_provenance):
    truth, calibration_report = _calibration(
        calibration_dir, real_folder, inputs, terrain_provenance)
    options = _method_options(mle_options)
    protocol = _science_protocol(inputs, truth, calibration_report, options)
    original = root / 'split_protocol.json'
    if not original.is_file() or read(original) != protocol:
        raise ValueError('Original four-run science protocol differs; refusing to append')
    pid = _write_or_check_protocol(root, protocol)
    _load_shared(root, pid, truth)  # All four original samples must still verify.
    return truth, options, pid


def _records(root, truth, pid):
    shared = root / 'shared'
    manifest_path = shared / MANIFEST
    manifest = read(manifest_path)
    if (manifest.get('version') != 1 or manifest.get('original_protocol_id') != pid
            or manifest.get('extension_code_sha256') != sha(Path(__file__))
            or manifest.get('n_grids') != len(truth)
            or manifest.get('n_years') != len(YEARS)
            or len(manifest.get('replicates', [])) != LAST - FIRST + 1):
        raise ValueError('Six-run extension manifest/protocol differs')
    seen = set()
    result = []
    for index, record in enumerate(manifest['replicates'], FIRST):
        folder = shared / f'replicate_{index:03d}'
        fit = folder / 'annual_maxima.npy'
        evaluation = folder / 'crps_evaluation_maxima.npy'
        if (record.get('replicate') != index or not fit.is_file() or not evaluation.is_file()
                or sha(fit) != record.get('annual_sha256')
                or sha(evaluation) != record.get('evaluation_sha256')):
            raise ValueError(f'Extension sample {index} is missing or modified')
        values = np.load(fit, allow_pickle=False)
        heldout = np.load(evaluation, allow_pickle=False)
        expected = (len(truth), len(YEARS))
        if (values.shape != expected or heldout.shape != expected
                or not np.isfinite(values).all() or not np.isfinite(heldout).all()
                or np.array_equal(values, heldout)
                or record['annual_sha256'] in seen):
            raise ValueError(f'Extension sample {index} failed shape/independence checks')
        seen.add(record['annual_sha256'])
        result.append((values, heldout, record))
    return result


def prepare_extension(real_folder, calibration_dir, inputs, root, *,
                      mle_options=None, terrain_provenance=None):
    """Generate six deterministic new streams, preserving replicate 001--004."""
    truth, _, pid = _context(real_folder, calibration_dir, inputs, root,
                            mle_options, terrain_provenance)
    shared = root / 'shared'
    manifest_path = shared / MANIFEST
    if manifest_path.exists():
        _records(root, truth, pid)
        return manifest_path
    for index in range(FIRST, LAST + 1):
        if (shared / f'replicate_{index:03d}').exists():
            raise ValueError(f'Unmanifested replicate {index} exists; preserve it for inspection')
    records = []
    for index in range(FIRST, LAST + 1):
        folder = shared / f'replicate_{index:03d}'
        folder.mkdir()
        values = simulation.draw_annual(truth, index - 1)
        heldout = simulation.draw_annual(truth, index - 1, purpose='crps_evaluation')
        atomic_array(folder / 'annual_maxima.npy', values)
        atomic_array(folder / 'crps_evaluation_maxima.npy', heldout)
        records.append(dict(
            replicate=index, annual_sha256=sha(folder / 'annual_maxima.npy'),
            evaluation_sha256=sha(folder / 'crps_evaluation_maxima.npy')))
    write(manifest_path, dict(version=1, original_protocol_id=pid,
        extension_code_sha256=sha(Path(__file__)), n_grids=len(truth),
        n_years=len(YEARS), replicates=records))
    _records(root, truth, pid)
    return manifest_path


def run_extension(real_folder, calibration_dir, inputs, root, method, *, threads,
                  mle_options=None, terrain_provenance=None):
    """Resume after four verified results; append six attempts for one method."""
    if method not in ('nn', 'mle'):
        raise ValueError('Method must be nn or mle')
    truth, options, pid = _context(real_folder, calibration_dir, inputs, root,
                                   mle_options, terrain_provenance)
    samples = _records(root, truth, pid)
    arm = root / 'arms' / method
    # Do not call the four-run runner: its fixed count and manifest must stay frozen.
    for index in range(1, FIRST):
        from .split_comparison import _branch
        _branch(root, method, index, pid, truth)
    status_path = arm / 'extension_status.json'
    _, grids, predictors, config, crs = inputs
    for offset, (values, evaluation, record) in enumerate(samples):
        index = FIRST + offset
        sample_id = identity(dict(protocol_id=pid, replicate=index,
            fit_sha256=record['annual_sha256'],
            evaluation_sha256=record['evaluation_sha256']))
        method_id = identity(dict(sample_id=sample_id, estimator=method))
        parent = arm / f'replicate_{index:03d}'
        done = []
        for folder in sorted(parent.glob('attempt_*')):
            report = completed(folder)
            if report:
                if (report.get('protocol_id') != method_id
                        or not np.array_equal(np.load(folder / 'annual_maxima.npy', allow_pickle=False), values)
                        or not np.array_equal(np.load(folder / 'truth.npy', allow_pickle=False), truth)):
                    raise ValueError(f'Completed replicate {index} is incompatible: {folder}')
                done.append((folder, report))
        if len(done) > 1:
            raise ValueError(f'Multiple completed attempts for replicate {index}')
        if done:
            folder, report = done[0]
            if 'sample_id' not in report:
                if not np.array_equal(np.load(folder / 'crps_evaluation_maxima.npy', allow_pickle=False), evaluation):
                    raise ValueError(f'Interrupted replicate {index} evaluation sample differs')
                report.update(replicate=index, sample_id=sample_id,
                    split_science_protocol_id=pid,
                    extension_manifest_sha256=sha(root / 'shared' / MANIFEST),
                    run_environment=dict(host='unknown_after_metadata_recovery', timing_comparable=False))
                write(folder / 'run_report.json', report)
            if (report.get('replicate') != index or report.get('sample_id') != sample_id
                    or report.get('estimator') != method):
                raise ValueError(f'Completed replicate {index} metadata differs')
            print(f'{method} replicate {index}/{LAST}: verified existing {folder}', flush=True)
            continue
        folder = attempt_folder(parent)
        write(status_path, dict(status='running', method=method,
            completed_extension_replicates=offset, total_extension_replicates=6,
            current_replicate=index, output=str(folder)))
        print(f'{method} replicate {index}/{LAST}: starting {folder}', flush=True)
        report = workflow.execute(values, grids, predictors, config, crs, folder,
            truth=truth, protocol_id=method_id, threads=threads, estimator=method,
            mle_options=options, evaluation_values=evaluation)
        report.update(replicate=index, sample_id=sample_id,
            split_science_protocol_id=pid, generation_seconds=0.0,
            extension_manifest_sha256=sha(root / 'shared' / MANIFEST),
            run_environment=comparison.environment(threads, config['n_jobs'], options['jobs']))
        write(folder / 'run_report.json', report)
        write(status_path, dict(status='running', method=method,
            completed_extension_replicates=offset + 1, total_extension_replicates=6,
            last_output=str(folder)))
    write(status_path, dict(status='completed', method=method,
        completed_replicates=LAST, total=LAST, completed_extension_replicates=6,
        original_protocol_id=pid, extension_manifest_sha256=sha(root / 'shared' / MANIFEST)))
    return status_path
