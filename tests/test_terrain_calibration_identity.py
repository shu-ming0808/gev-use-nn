"""Expanded terrain values/provenance must reach real-run and truth identities."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'src'), str(REPO/'tools/thesis_run_bundle')]
from portable import comparison, inference, simulation, terrain
from portable.common import identity, read, sha, write
from parameters.regional_terrain import extra_candidate_groups
from research_workflows import YEARS


def fixture(tmp_path, *, expanded=True):
    grids = pd.DataFrame({'grid_id': ['G01', 'G02'], 'latitude': [23., 24.], 'longitude_180': [120., 121.]})
    base = pd.DataFrame({'grid_id': grids.grid_id, 'elevation_m': [10., 200.], 'candidate_data_complete': True})
    base_groups = {'elevation': ['elevation_m']}
    config = {'candidate_groups': deepcopy(base_groups), 'n_jobs': 4, 'seed': 20260929}
    predictors = base.copy()
    provenance = None
    if expanded:
        additions = extra_candidate_groups('expanded_v2')
        for i, name in enumerate(column for group in additions.values() for column in group):
            predictors[name] = [i+0.1, i+0.2]
        config['candidate_groups'].update(additions)
        evidence = dict(profile='expanded_v2', source_dem_sha256='a'*64,
            terrain_helper_sha256='b'*64, overlay_code_sha256='c'*64,
            grid_fingerprint=terrain.frame_fingerprint(grids),
            predictor_fingerprint=terrain.frame_fingerprint(base),
            original_candidate_groups=base_groups, coordinate_crs='regional')
        provenance = dict(profile='expanded_v2', input_identity=evidence, protocol_id=identity(evidence))
    inputs = (None, grids, predictors, config, 'regional')
    folder = tmp_path/'real'; folder.mkdir()
    grids.to_csv(folder/'nn_coefficients.csv', index=False)
    report = dict(workflow_completed=True, region='EAS', data_role='real_data',
        model={'sha256': inference.DIGEST}, config={**config, 'n_jobs': -2},
        years=list(YEARS), n_grids=2,
        output_sha256={'nn_coefficients.csv': sha(folder/'nn_coefficients.csv')})
    if expanded:
        report.update(comparison.terrain_input_metadata(inputs, provenance))
    write(folder/'run_report.json', report)
    return folder, inputs, provenance, report


def test_expanded_real_source_accepts_matching_values_and_njobs_exception(tmp_path):
    folder, inputs, provenance, report = fixture(tmp_path)
    assert comparison.validate_real_source(folder, inputs, terrain_provenance=provenance) == report
    inputs[3]['n_jobs'] = 8
    assert comparison.validate_real_source(folder, inputs, terrain_provenance=provenance) == report


def test_expanded_real_without_new_metadata_cannot_silently_continue(tmp_path):
    folder, inputs, provenance, report = fixture(tmp_path)
    for key in list(report):
        if key.startswith('terrain_'): del report[key]
    write(folder/'run_report.json', report)
    with pytest.raises(ValueError, match='source provenance/fingerprints differ'):
        comparison.validate_real_source(folder, inputs, terrain_provenance=provenance)


def test_expanded_requires_overlay_provenance_even_if_report_has_it(tmp_path):
    folder, inputs, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='requires verified terrain provenance'):
        comparison.validate_real_source(folder, inputs)


def test_actual_new_terrain_values_reject_previous_real_source(tmp_path):
    folder, inputs, provenance, _ = fixture(tmp_path)
    inputs[2].loc[0, 'terrain_025_slope_deg'] += 1.
    with pytest.raises(ValueError, match='source provenance/fingerprints differ'):
        comparison.validate_real_source(folder, inputs, terrain_provenance=provenance)


def test_new_dem_identity_rejects_previous_real_source_even_when_values_equal(tmp_path):
    folder, inputs, provenance, _ = fixture(tmp_path)
    provenance['input_identity']['source_dem_sha256'] = 'd'*64
    provenance['protocol_id'] = identity(provenance['input_identity'])
    with pytest.raises(ValueError, match='source provenance/fingerprints differ'):
        comparison.validate_real_source(folder, inputs, terrain_provenance=provenance)


@pytest.mark.parametrize('changed', ['grid', 'baseline', 'schema'])
def test_overlay_metadata_checks_actual_baseline_and_candidate_schema(tmp_path, changed):
    _, inputs, provenance, _ = fixture(tmp_path)
    if changed == 'grid': inputs[1].loc[0, 'latitude'] += .25
    elif changed == 'baseline': inputs[2].loc[0, 'elevation_m'] += 1.
    else: inputs[3]['candidate_groups']['terrain_aspect'] = ['terrain_025_northness']
    with pytest.raises(ValueError, match='provenance'):
        comparison.terrain_input_metadata(inputs, provenance)


def test_legacy_source_without_new_fields_remains_compatible(tmp_path):
    folder, inputs, provenance, report = fixture(tmp_path, expanded=False)
    assert provenance is None
    assert comparison.terrain_input_metadata(inputs) == {}
    assert comparison.validate_real_source(folder, inputs) == report


class StopBeforeFit(RuntimeError):
    pass


def test_expanded_calibration_protocol_hashes_values_and_rejects_stale_truth(tmp_path, monkeypatch):
    folder, inputs, provenance, report = fixture(tmp_path)
    calls = []
    def stop(*args, **kwargs):
        calls.append(True)
        raise StopBeforeFit()
    monkeypatch.setattr(simulation, 'memory_guard', stop)
    output = tmp_path/'calibration'
    with pytest.raises(StopBeforeFit):
        simulation.calibrate(folder, inputs[1], inputs[2], inputs[3], inputs[4], output,
                             threads=1, terrain_provenance=provenance)
    before = read(output/'protocol.json')
    assert before['version'] == 2 and before['terrain_profile'] == 'expanded_v2'
    assert before['terrain_input_fingerprints']['predictors'] == terrain.frame_fingerprint(inputs[2])
    assert before['terrain_protocol_id'] == provenance['protocol_id']
    assert before['calibration_code_sha256']
    # A new compatible real run exists, but this calibration directory still
    # belongs to the old predictor values. It must fail before fitting/reuse.
    inputs[2].loc[0, 'terrain_025_slope_deg'] += 2.
    report.update(comparison.terrain_input_metadata(inputs, provenance))
    write(folder/'run_report.json', report)
    with pytest.raises(ValueError, match='Calibration inputs changed'):
        simulation.calibrate(folder, inputs[1], inputs[2], inputs[3], inputs[4], output,
                             threads=1, terrain_provenance=provenance)
    assert len(calls) == 1 and read(output/'protocol.json') == before


def test_legacy_calibration_retains_original_protocol_shape(tmp_path, monkeypatch):
    folder, inputs, _, _ = fixture(tmp_path, expanded=False)
    monkeypatch.setattr(simulation, 'memory_guard', lambda *args: (_ for _ in ()).throw(StopBeforeFit()))
    output = tmp_path/'legacy_calibration'
    with pytest.raises(StopBeforeFit):
        simulation.calibrate(folder, inputs[1], inputs[2], inputs[3], inputs[4], output, threads=1)
    protocol = read(output/'protocol.json')
    assert protocol['version'] == 1
    assert not any(key.startswith('terrain_') for key in protocol)
    assert 'calibration_code_sha256' not in protocol


def test_calibrate_directly_rejects_expanded_source_without_provenance(tmp_path):
    folder, inputs, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='requires verified terrain provenance'):
        simulation.calibrate(folder, inputs[1], inputs[2], inputs[3], inputs[4], tmp_path/'calibration')
    assert not (tmp_path/'calibration').exists()
