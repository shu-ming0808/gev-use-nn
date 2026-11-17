"""Small synthetic manifest checks; no EAS workflow is run."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'tools' / 'thesis_run_bundle')]

from portable.common import sha, write
from portable import split_extension_11_20 as extension
from research_workflows import YEARS


def make_shared(tmp_path):
    shared = tmp_path / 'shared'
    shared.mkdir()
    prior = [dict(replicate=i, annual_sha256=f'prior{i}') for i in range(1, 11)]
    write(shared / 'sample_manifest.json', dict(replicates=prior[:4]))
    write(shared / 'extension_manifest_005_010.json', dict(replicates=prior[4:]))
    records = []
    for index in range(11, 21):
        folder = shared / f'replicate_{index:03d}'
        folder.mkdir()
        np.save(folder / 'annual_maxima.npy', np.full((2, len(YEARS)), index, dtype=float))
        np.save(folder / 'crps_evaluation_maxima.npy', np.full((2, len(YEARS)), -index, dtype=float))
        records.append(dict(replicate=index,
            annual_sha256=sha(folder / 'annual_maxima.npy'),
            evaluation_sha256=sha(folder / 'crps_evaluation_maxima.npy')))
    write(shared / extension.MANIFEST, dict(version=1, original_protocol_id='protocol',
        extension_code_sha256=sha(Path(extension.__file__)), n_grids=2,
        n_years=len(YEARS), replicates=records))
    return shared


def test_reads_exactly_ten_new_paired_samples(tmp_path):
    make_shared(tmp_path)
    samples = extension._samples(tmp_path, np.zeros((2, 5)), 'protocol')
    assert len(samples) == 10
    assert samples[0][2]['replicate'] == 11
    assert samples[-1][2]['replicate'] == 20


def test_rejects_changed_shared_sample(tmp_path):
    shared = make_shared(tmp_path)
    np.save(shared / 'replicate_011' / 'annual_maxima.npy', np.zeros((2, len(YEARS))))
    with pytest.raises(ValueError, match='changed'):
        extension._samples(tmp_path, np.zeros((2, 5)), 'protocol')


def test_mle_continues_existing_nested_layout_without_moving(tmp_path):
    nested = tmp_path / 'arms' / 'mle' / 'mle' / 'replicate_001'
    nested.mkdir(parents=True)
    assert extension.arm_folder(tmp_path, 'mle') == nested.parent
    assert extension.arm_folder(tmp_path, 'nn') == tmp_path / 'arms' / 'nn'
