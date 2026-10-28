"""Only temporary configs are touched; no real WSL settings or VM changes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/thesis_run_bundle/configure-wsl-16gb.ps1'


def runner():
    spec = importlib.util.spec_from_file_location('namespace_runner', ROOT / 'tools/thesis_run_bundle/run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_names_are_isolated_and_default_is_unchanged():
    name = runner().real_output_name
    assert name('expanded_v2') == 'eas_terrain_v2'
    assert name('legacy_v1') == 'eas'
    assert name('expanded_v2', 'ram16_v1') == 'eas_terrain_v2_ram16_v1'


@pytest.mark.parametrize('label', ['../oops', '/tmp', 'C:\\oops', '---', 'a b', 'a'*65, '測試'])
def test_unsafe_output_names_are_rejected(label):
    with pytest.raises(ValueError):
        runner().real_output_name('expanded_v2', label)


def invoke(path, apply=False, processors=None):
    args = ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(SCRIPT), '-ConfigPath', str(path)]
    if apply:
        args.append('-Apply')
    if processors is not None:
        args.extend(['-Processors', str(processors)])
    return subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)


@pytest.mark.skipif(sys.platform != 'win32', reason='PowerShell 5.1 configuration utility')
def test_dry_run_then_apply_preserves_other_settings_and_creates_backup(tmp_path):
    path = tmp_path / '.wslconfig'
    original = b'# Keep comment\r\n[wsl2]\r\nmemory=22GB\r\nprocessors=18\r\nswap=8GB\r\n[experimental]\r\nautoMemoryReclaim=gradual\r\n'
    path.write_bytes(original)
    result = invoke(path)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert not plan['applied'] and plan['changed'] and not plan['wsl_stopped']
    assert path.read_bytes() == original
    result = invoke(path, True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert Path(plan['backup']).read_bytes() == original
    text = path.read_text(encoding='utf-8')
    assert 'memory=10GB' in text and 'swap=4GB' in text
    assert '# Keep comment' in text and 'processors=18' in text
    assert '[experimental]\nautoMemoryReclaim=gradual' in text
    result = invoke(path, True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['changed'] is False
    assert len(list(tmp_path.glob('.wslconfig.backup-*'))) == 1


@pytest.mark.skipif(sys.platform != 'win32', reason='PowerShell 5.1 configuration utility')
@pytest.mark.parametrize('initial', ['', '[experimental]\nautoMemoryReclaim=gradual\n', '[wsl2]\nprocessors=8\n'])
def test_new_config_or_missing_keys(tmp_path, initial):
    path = tmp_path / '.wslconfig'
    if initial:
        path.write_text(initial, encoding='utf-8')
    result = invoke(path, True)
    assert result.returncode == 0, result.stderr
    text = path.read_text(encoding='utf-8')
    assert text.count('[wsl2]') == 1
    assert text.count('memory=10GB') == 1 and text.count('swap=4GB') == 1
    assert json.loads(result.stdout)['workflow_started'] is False


@pytest.mark.skipif(sys.platform != 'win32', reason='PowerShell 5.1 configuration utility')
@pytest.mark.parametrize('initial', ['[wsl2]\nmemory=22GB\nmemory=4GB\n', '[wsl2]\nswap=2GB\nswap=4GB\n', '[wsl2]\n[wsl2]\n'])
def test_ambiguous_config_is_unchanged_on_failure(tmp_path, initial):
    path = tmp_path / '.wslconfig'
    path.write_text(initial, encoding='utf-8')
    before = path.read_bytes()
    result = invoke(path, True)
    assert result.returncode != 0
    assert path.read_bytes() == before
    assert not list(tmp_path.glob('*.backup-*'))


@pytest.mark.skipif(sys.platform != 'win32', reason='PowerShell 5.1 configuration utility')
def test_explicit_processor_budget_replaces_old_cap_only_when_applied(tmp_path):
    path = tmp_path / '.wslconfig'
    original = '[wsl2]\nprocessors=8\nmemory=8GB\nswap=4GB\n[experimental]\nfoo=bar\n'
    path.write_text(original, encoding='utf-8')
    before = path.read_bytes()
    result = invoke(path, processors=20)
    assert result.returncode == 0, result.stderr
    assert path.read_bytes() == before
    assert json.loads(result.stdout)['processors'] == 20
    result = invoke(path, apply=True, processors=20)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert Path(plan['backup']).read_bytes() == before
    text = path.read_text(encoding='utf-8')
    assert 'processors=20' in text and 'processors=8' not in text
    assert '[experimental]\nfoo=bar' in text
    assert 'memory=10GB' in text
    assert not plan['wsl_stopped'] and not plan['workflow_started']


@pytest.mark.skipif(sys.platform != 'win32', reason='PowerShell 5.1 configuration utility')
def test_duplicate_processors_rejected_before_writing(tmp_path):
    path = tmp_path / '.wslconfig'
    path.write_text('[wsl2]\nprocessors=8\nprocessors=20\n', encoding='utf-8')
    before = path.read_bytes()
    result = invoke(path, apply=True, processors=20)
    assert result.returncode != 0
    assert path.read_bytes() == before
    assert not list(tmp_path.glob('*.backup-*'))
