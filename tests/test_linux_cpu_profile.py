"""Portable CPU defaults and early BLAS initialization; no formal workflow starts."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / 'tools/thesis_run_bundle'


def runner():
    spec = importlib.util.spec_from_file_location('cpu_profile_runner', BUNDLE / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('mode,base,worker,refit,expected', [
    ('fixed', 1, 4, 8, 1),
    ('staged', 1, 4, 8, 8),
    ('staged', 1, 1, 1, 1),
    ('adaptive', 1, 1, 1, 4),
    ('adaptive', 8, 4, 8, 8),
    ('adaptive', 2, 16, 8, 16),
])
def test_native_initial_threads_covers_all_later_limits(mode, base, worker, refit, expected):
    args = SimpleNamespace(parallel_mode=mode, threads=base,
                           worker_threads_max=worker, refit_threads=refit)
    assert runner().native_initial_threads(args) == expected


def test_importing_runner_does_not_initialize_numeric_libraries():
    code = (
        'import runpy,sys; '
        f'runpy.run_path({str(BUNDLE / "run.py")!r},run_name="profile_import_check"); '
        'assert not ({"numpy","scipy","torch"} & set(sys.modules))'
    )
    result = subprocess.run([sys.executable, '-B', '-c', code],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_bash_entry_uses_ultra265_profile_and_keeps_cli_overrides_last():
    text = (BUNDLE / 'run-linux.sh').read_text(encoding='utf-8')
    start = text.index('bash "$ROOT/wsl/run.sh"')
    words = shlex.split(text[start:].replace('\\\n', ' '), comments=True)
    expected = {'--threads': '8', '--worker-threads-max': '4', '--refit-threads': '8',
                '--cpu-budget': '18', '--jobs': '4', '--mle-jobs': '16',
                '--memory-budget-gib': '20', '--memory-reserve-gib': '2',
                '--parallel-mode': 'adaptive', '--method-order': 'mle-first',
                '--replicates': '1'}
    for flag, value in expected.items():
        assert words[words.index(flag) + 1] == value, flag
    assert words[-1] == '$@'


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows launcher')
def test_powershell_plan_only_exposes_same_profile_without_running():
    result = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                             '-File', str(BUNDLE / 'run-linux.ps1'), '-PlanOnly'],
                            capture_output=True, text=True, encoding='utf-8', errors='replace',
                            timeout=20)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['order'] == ['MLE + GP', 'NN + GP']
    assert plan['starts_workflow'] is False
    assert plan['parallel_mode'] == 'adaptive'
    for name, expected in {'threads': 8, 'worker_threads_max': 4, 'refit_threads': 8,
                           'cpu_budget': 18, 'jobs': 4, 'mle_jobs': 16,
                           'memory_budget_gib': 20, 'memory_reserve_gib': 2}.items():
        assert plan[name] == expected, name


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows launcher')
def test_wsl_wrapper_forwards_unicode_path_and_explicit_thread_limits():
    # Mock WSL rather than launch a distro or touch a prepared workspace.
    wrapper = str(BUNDLE / 'run-wsl.ps1').replace("'", "''")
    code = f"""
    $ErrorActionPreference = 'Stop'
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    function global:wsl.exe {{
        $global:LASTEXITCODE = 0
        if ($args -contains 'wslpath') {{ '/mnt/f/論文跑' }}
        else {{ $global:WslArguments = @($args) }}
    }}
    & '{wrapper}' -Mode check -Distro Ubuntu-24.04 -Threads 8 -CpuBudget 18 `
        -Jobs 4 -MleJobs 16 -WorkerThreadsMax 4 -RefitThreads 8 `
        -ParallelMode adaptive -MemoryBudgetGiB 20 -MemoryReserveGiB 2
    $global:WslArguments | ConvertTo-Json -Compress
    """
    result = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                             '-Command', code], capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=20)
    assert result.returncode == 0, result.stderr
    arguments = json.loads(result.stdout)
    assert arguments[:2] == ['-d', 'Ubuntu-24.04']
    assert '/mnt/f/論文跑/wsl/run.sh' in arguments
    assert '/mnt/f/論文跑' in arguments
    for flag, value in {'--threads': '8', '--jobs': '4', '--mle-jobs': '16',
                        '--worker-threads-max': '4', '--refit-threads': '8',
                        '--cpu-budget': '18'}.items():
        assert arguments[arguments.index(flag) + 1] == value
