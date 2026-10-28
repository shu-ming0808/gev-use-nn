"""Expanded terrain entrypoints stay separate from historical experiments."""
from pathlib import Path
import importlib.util

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'tools/thesis_run_bundle'


def runner():
    spec = importlib.util.spec_from_file_location('terrain_profile_runner', BUNDLE/'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('args', [
    ['prepare-terrain'],
    ['global', '--terrain-profile', 'expanded_v2'],
])
def test_wrong_profile_command_fails_before_loading_data(args):
    with pytest.raises(SystemExit) as error:
        runner().main(args)
    assert error.value.code == 2


def test_expanded_launcher_eas_precedes_calibrated_comparison():
    text = (BUNDLE/'run-eas-terrain-linux.sh').read_text(encoding='utf-8')
    assert '--terrain-profile expanded_v2' in text
    assert text.index('"$ROOT" eas') < text.index('"$ROOT" compare-mle')
    assert 'prepare-terrain "${COMMON[@]}" "$@"' in text
    assert '--method-order mle-first' in text
    assert '--new-run' not in text


def test_legacy_launcher_does_not_silently_change_candidate_profile():
    text = (BUNDLE/'run-linux.sh').read_text(encoding='utf-8')
    assert 'expanded_v2' not in text


def test_terrain_profile_has_separate_real_data_parent():
    text = (BUNDLE/'run.py').read_text(encoding='utf-8')
    name = runner().real_output_name
    assert name('expanded_v2') == 'eas_terrain_v2'
    assert name('legacy_v1') == 'eas'
    assert name('expanded_v2','ram16_v1') == 'eas_terrain_v2_ram16_v1'
    assert 'real_parent=OUTPUT/real_name' in text
    assert "terrain_protocol_id=terrain_provenance.get('protocol_id')" in text
    assert "latest_completed(OUTPUT/'eas')" not in text


def test_benchmark_entry_does_not_start_formal_workflow():
    text = (BUNDLE/'benchmark-linux.sh').read_text(encoding='utf-8')
    assert 'benchmark_gp_execution.py' in text
    assert 'compare-mle' not in text and 'run.py eas' not in text
    assert 'OMP_NUM_THREADS=8' in text


@pytest.mark.parametrize('name', ['run.ps1','run-wsl.ps1'])
def test_powershell_wrappers_forward_explicit_profile(name):
    text = (BUNDLE/name).read_text(encoding='utf-8')
    assert "[string]$TerrainProfile = 'legacy_v1'" in text
    assert "'--terrain-profile',$TerrainProfile" in text
    assert "'prepare-terrain'" in text
