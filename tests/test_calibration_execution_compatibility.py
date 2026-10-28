"""Only the historical calibration source may differ in its worker count."""
from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'src'), str(REPO/'tools/thesis_run_bundle')]
from portable import comparison, inference
from portable.common import read, write, sha
from research_workflows import YEARS


def fixture(tmp_path):
    config = dict(n_jobs=4, random_state=20260929, outer_splits=5, inner_splits=3,
                  buffer_km=500., min_train=25, max_train=None,
                  kernel_candidates=[['RBF', None], ['Matern', 1.5]],
                  candidate_groups={'terrain': ['elevation_m']}, max_steps=11)
    grids = pd.DataFrame({'grid_id': ['EAS_1', 'EAS_2']})
    folder = tmp_path/'historical'; folder.mkdir()
    grids.to_csv(folder/'nn_coefficients.csv', index=False)
    report = dict(workflow_completed=True, region='EAS', data_role='real_data',
                  model={'sha256': inference.DIGEST}, config={**config, 'n_jobs': -2},
                  years=list(YEARS), n_grids=2,
                  output_sha256={'nn_coefficients.csv': sha(folder/'nn_coefficients.csv')})
    write(folder/'run_report.json', report)
    return folder, (None, grids, None, config, None), report


def test_njobs_only_difference_accepts_old_calibration_without_rewriting_source(tmp_path):
    folder, inputs, report = fixture(tmp_path)
    before = (folder/'run_report.json').read_bytes()
    checked = comparison.validate_real_source(folder, inputs)
    assert checked == report and checked['config']['n_jobs'] == -2
    assert inputs[3]['n_jobs'] == 4
    assert (folder/'run_report.json').read_bytes() == before


def test_config_helper_accepts_only_njobs_and_does_not_mutate():
    left = dict(n_jobs=-2, seed=1, nested={'value': [1, 2]})
    right = dict(n_jobs=4, seed=1, nested={'value': [1, 2]})
    saved = deepcopy((left, right))
    assert comparison.calibration_configs_match(left, right)
    assert (left, right) == saved
    assert comparison.calibration_configs_match(left, deepcopy(left))
    for bad in (None, [], 'config'):
        assert not comparison.calibration_configs_match(bad, right)
        assert not comparison.calibration_configs_match(left, bad)
    for changed in ({'n_jobs': 4, 'seed': 1},
                    {**right, 'unknown': None},
                    {'seed': 1, 'nested': {'value': [1, 2]}},
                    {**right, 'nested': {'value': [2, 1]}}):
        assert not comparison.calibration_configs_match(left, changed)


@pytest.mark.parametrize('key,value', [
    ('random_state', 42), ('outer_splits', 6), ('inner_splits', 5),
    ('buffer_km', 30.), ('min_train', 2), ('max_train', 800),
    ('kernel_candidates', [['Matern', .5]]),
    ('candidate_groups', {'terrain': ['latitude']}), ('max_steps', 1),
])
def test_scientific_config_changes_still_reject(tmp_path, key, value):
    folder, inputs, _ = fixture(tmp_path)
    inputs[3][key] = value
    with pytest.raises(ValueError, match='compatible real EAS'):
        comparison.validate_real_source(folder, inputs)


@pytest.mark.parametrize('key,value', [
    ('workflow_completed', False), ('region', 'WNA'), ('data_role', 'simulation'),
    ('model', {'sha256': 'not-the-frozen-model'}), ('n_grids', 3),
    ('years', list(YEARS)[1:]),
])
def test_historical_identity_checks_still_reject(tmp_path, key, value):
    folder, inputs, report = fixture(tmp_path)
    report[key] = value
    write(folder/'run_report.json', report)
    with pytest.raises(ValueError, match='compatible real EAS'):
        comparison.validate_real_source(folder, inputs)


def test_modified_historical_coefficients_fail_checksum(tmp_path):
    folder, inputs, _ = fixture(tmp_path)
    (folder/'nn_coefficients.csv').write_text('grid_id\nchanged\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Completed output changed'):
        comparison.validate_real_source(folder, inputs)


def test_valid_checksum_but_reordered_grids_still_rejects(tmp_path):
    folder, inputs, report = fixture(tmp_path)
    inputs[1].iloc[::-1].to_csv(folder/'nn_coefficients.csv', index=False)
    report['output_sha256']['nn_coefficients.csv'] = sha(folder/'nn_coefficients.csv')
    write(folder/'run_report.json', report)
    with pytest.raises(ValueError, match='GRID order differs'):
        comparison.validate_real_source(folder, inputs)
