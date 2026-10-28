"""Exercise the isolated 16 GiB launcher without setup or scientific workloads."""
from pathlib import Path
import os
import shutil
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / 'tools/thesis_run_bundle/run-eas-16gb-linux.sh'


@pytest.fixture(scope='module')
def bash():
    # Windows' System32/bash.exe starts WSL: never select it for these tests.
    # Git Bash exercises the shell itself without starting a Linux VM/setup.
    if os.name == 'nt':
        candidates = [Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git/bin/bash.exe',
                      Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git/usr/bin/bash.exe']
        executable = next((str(path) for path in candidates if path.is_file()), None)
    else:
        executable = shutil.which('bash')
    if executable is None:
        pytest.skip('A native Bash shell is required; tests never start WSL automatically')
    return executable


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / 'bundle with 空間'
    root.mkdir()
    script = root / LAUNCHER.name
    shutil.copyfile(LAUNCHER, script)
    return root, script


def invoke(bash, script, *args, env=None):
    process_env = dict(os.environ, **(env or {}))
    # Test wrappers are pure Bash; use no inherited shell startup hooks.
    process_env.pop('BASH_ENV', None)
    return subprocess.run([bash, str(script).replace('\\', '/'), *args],
                          capture_output=True, text=True, encoding='utf-8',
                          errors='strict', env=process_env, timeout=20)


def install_stubs(root, monkeypatch):
    folder = root / 'wsl'
    folder.mkdir()
    capture = root / 'captured arguments.bin'
    monkeypatch.setenv('CAPTURE_FILE', capture.as_posix())
    (folder / 'setup.sh').write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        'printf "%s\\0" setup "$@" __END_COMMAND__ >> "$CAPTURE_FILE"\n'
        'exit "${STUB_SETUP_EXIT:-0}"\n', encoding='utf-8', newline='\n')
    (folder / 'run.sh').write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        'printf "%s\\0" run "$@" __END_COMMAND__ >> "$CAPTURE_FILE"\n'
        'if [[ "$2" == eas ]]; then exit "${STUB_EAS_EXIT:-0}"; fi\n'
        'if [[ "$2" == compare-mle ]]; then exit "${STUB_COMPARE_EXIT:-0}"; fi\n',
        encoding='utf-8', newline='\n')
    return capture


def captured_calls(path):
    calls, current = [], []
    for part in path.read_bytes().decode('utf-8').split('\0'):
        if part == '__END_COMMAND__':
            calls.append(current)
            current = []
        elif part:
            current.append(part)
    assert not current
    return calls


def option_value(args, name):
    """argparse keeps the final occurrence, including explicit user overrides."""
    index = max(i for i, arg in enumerate(args) if arg == name)
    return args[index + 1]


def test_launcher_bash_syntax(bash):
    result = subprocess.run([bash, '-n', LAUNCHER.as_posix()], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(('mode', 'commands'), [
    ('prepare', ['prepare-terrain']), ('check', ['check']),
    ('eas', ['eas']), ('compare', ['eas', 'compare-mle']),
    ('compare-existing', ['compare-mle']),
])
def test_plan_only_needs_no_runtime_and_displays_every_command(bash, bundle, mode, commands):
    root, script = bundle
    result = invoke(bash, script, mode, '--plan-only')
    assert result.returncode == 0, result.stderr
    assert not (root / 'wsl').exists()
    assert list(root.iterdir()) == [script]
    lines = [line for line in result.stdout.splitlines() if line.startswith('  bash ')]
    assert len(lines) == 1 + len(commands)
    assert '/wsl/setup.sh ' in lines[0] and lines[0].endswith('--skip-check ')
    for line, command in zip(lines[1:], commands):
        assert '/wsl/run.sh ' in line and f' {command} --terrain-profile expanded_latitude_v3 ' in line
        assert '--run-label ram16_mem8_v1 --parallel-mode adaptive' in line
        assert '--memory-budget-gib 8 --memory-reserve-gib 1.5 --cpu-budget 20' in line
        assert '--jobs 2 --worker-threads-max 20 --refit-threads 20 --threads 4 --mle-jobs 20' in line
        assert '--plan-only' not in line
    assert 'no setup, staging or computation' in result.stdout
    if mode in ('compare', 'compare-existing'):
        assert '--method-order mle-first --experiment latitude_v3_ram16_mem8_cpu20_' in lines[-1]
        assert '--replicates 1 ' in lines[-1]


@pytest.mark.parametrize(('mode', 'commands'), [
    ('prepare', ['prepare-terrain']), ('check', ['check']),
    ('eas', ['eas']), ('compare', ['eas', 'compare-mle']),
    ('compare-existing', ['compare-mle']),
])
def test_real_dispatch_defaults_and_completed_protocol_reuse(bash, bundle, monkeypatch, mode, commands):
    root, script = bundle
    capture = install_stubs(root, monkeypatch)
    result = invoke(bash, script, mode)
    assert result.returncode == 0, result.stderr
    calls = captured_calls(capture)
    assert calls[0][0] == 'setup' and calls[0][2:] == ['--skip-check']
    assert [call[2] for call in calls[1:]] == commands
    for call in calls[1:]:
        assert call[:2] == ['run', calls[0][1]]
        args = call[3:]
        expected = {
            '--terrain-profile': 'expanded_latitude_v3', '--run-label': 'ram16_mem8_v1', '--parallel-mode': 'adaptive',
            '--memory-budget-gib': '8', '--memory-reserve-gib': '1.5',
            '--cpu-budget': '20', '--jobs': '2', '--worker-threads-max': '20',
            '--refit-threads': '20', '--threads': '4', '--mle-jobs': '20',
        }
        for name, value in expected.items():
            assert option_value(args, name) == value
        # No forced new attempt or alternate input/cache paths bypass validation.
        assert not {'--new-run', '--source', '--settings', '--skip-preprocess'} & set(args)
    if mode in ('compare', 'compare-existing'):
        args = calls[-1][3:]
        assert option_value(args, '--method-order') == 'mle-first'
        assert option_value(args, '--replicates') == '1'
        assert option_value(args, '--experiment').startswith('latitude_v3_ram16_mem8_cpu20_')


@pytest.mark.parametrize('mode', ['compare', 'compare-existing'])
def test_user_overrides_are_last_and_paths_are_not_split(bash, bundle, monkeypatch, mode):
    root, script = bundle
    capture = install_stubs(root, monkeypatch)
    overrides = ['--jobs', '1', '--memory-budget-gib', '4', '--threads', '3',
                 '--experiment', 'terrain_v2_16gb_resume', '--replicates', '100',
                 '--method-order', 'nn-first', '--data-root', '/mnt/f/論文資料 with spaces']
    result = invoke(bash, script, mode, *overrides)
    assert result.returncode == 0, result.stderr
    calls = captured_calls(capture)
    assert len(calls) == (3 if mode == 'compare' else 2)
    for call in calls[1:]:
        assert call[-len(overrides):] == overrides
        for name in ['--jobs', '--memory-budget-gib', '--threads', '--experiment',
                     '--replicates', '--method-order', '--data-root']:
            assert option_value(call, name) == option_value(overrides, name)


def test_plan_flag_is_removed_even_after_overrides(bash, bundle):
    root, script = bundle
    result = invoke(bash, script, 'compare', '--jobs', '1', '--plan-only', '--experiment', 'resume_16gb')
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.startswith('  bash ')]
    assert all('--plan-only' not in line for line in lines)
    assert lines[-1].endswith('--jobs 1 --experiment resume_16gb ')
    assert not (root / 'wsl').exists()


@pytest.mark.parametrize(('environment', 'expected_calls'), [
    ({'STUB_SETUP_EXIT': '7'}, ['setup']),
    ({'STUB_EAS_EXIT': '7'}, ['setup', 'eas']),
])
def test_failure_stops_before_comparison(bash, bundle, monkeypatch, environment, expected_calls):
    root, script = bundle
    capture = install_stubs(root, monkeypatch)
    result = invoke(bash, script, 'compare', env=environment)
    assert result.returncode == 7
    calls = captured_calls(capture)
    assert [call[0] if call[0] == 'setup' else call[2] for call in calls] == expected_calls


@pytest.mark.parametrize(('environment', 'expected_calls'), [
    ({'STUB_SETUP_EXIT': '7'}, ['setup']),
    ({'STUB_COMPARE_EXIT': '9'}, ['setup', 'compare-mle']),
])
def test_compare_existing_failure_never_falls_back_to_real_eas(
        bash, bundle, monkeypatch, environment, expected_calls):
    root, script = bundle
    capture = install_stubs(root, monkeypatch)
    result = invoke(bash, script, 'compare-existing', env=environment)
    assert result.returncode == int(next(iter(environment.values())))
    calls = captured_calls(capture)
    assert [call[0] if call[0] == 'setup' else call[2] for call in calls] == expected_calls


def test_invalid_mode_stops_before_setup(bash, bundle):
    root, script = bundle
    result = invoke(bash, script, 'global', '--plan-only')
    assert result.returncode == 2
    assert 'Use prepare, eas, compare, compare-existing or check' in result.stderr
    assert not (root / 'wsl').exists()


def test_no_arguments_preserves_eas_default(bash, bundle, monkeypatch):
    root, script = bundle
    capture = install_stubs(root, monkeypatch)
    result = invoke(bash, script)
    assert result.returncode == 0, result.stderr
    assert [call[2] for call in captured_calls(capture)[1:]] == ['eas']


def test_primary_instructions_run_new_latitude_profile_before_comparison():
    instructions = (LAUNCHER.parent / 'PowerShell指令.txt').read_text(encoding='utf-8-sig')
    run_commands = [line for line in instructions.splitlines()
                    if line.startswith('wsl ') and 'run-eas-16gb-linux.sh' in line]
    assert len(run_commands) == 1
    assert ' compare --experiment latitude_v3_ram16_mem8_cpu20_v1 --replicates 1' in run_commands[0]
    assert 'compare-existing' not in run_commands[0]
    assert 'eas_terrain_latitude_v3_ram16_mem8_v1' in instructions
    assert '排程預算8GiB' in instructions and 'WSL最多10GB' in instructions
    assert '不保證不會OOM' in instructions
