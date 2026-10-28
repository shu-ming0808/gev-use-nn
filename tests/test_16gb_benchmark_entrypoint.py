"""16 GiB layout benchmark launcher: no installs, WSL or actual GP in tests."""
from pathlib import Path
import os
import shlex
import shutil
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO/'tools/thesis_run_bundle/benchmark-16gb-linux.sh'


@pytest.fixture(scope='module')
def bash():
    # System32/bash.exe launches WSL; use native Git Bash on Windows instead.
    if os.name == 'nt':
        program_files = Path(os.environ.get('ProgramFiles', 'C:/Program Files'))
        candidates = [program_files/'Git/bin/bash.exe', program_files/'Git/usr/bin/bash.exe']
        executable = next((str(path) for path in candidates if path.is_file()), None)
    else:
        executable = shutil.which('bash')
    if executable is None:
        pytest.skip('Native Bash unavailable; tests never start/install WSL')
    return executable


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path/'bundle with 空間'
    root.mkdir()
    script = root/LAUNCHER.name
    shutil.copyfile(LAUNCHER, script)
    return root, script


def invoke(bash, script, *args, env=None):
    process_env = dict(os.environ, **(env or {}))
    process_env.pop('BASH_ENV', None)
    return subprocess.run([bash, script.as_posix(), *args], capture_output=True,
        text=True, encoding='utf-8', errors='strict', env=process_env, timeout=20)


def install_stubs(root):
    wsl = root/'wsl'
    wsl.mkdir()
    data = root/'xdg data 空間'
    runtime = data/'thesis-run-cpu-v1'
    (runtime/'env/bin').mkdir(parents=True)
    (runtime/'workspace').mkdir()
    capture = root/'captured arguments.bin'
    (wsl/'setup.sh').write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        'printf "%s\\0" setup "$@" __END_COMMAND__ >> "$CAPTURE_FILE"\n'
        'printf "%s\\0" setup_env "$OMP_NUM_THREADS" "$OPENBLAS_NUM_THREADS" '
        '"$MKL_NUM_THREADS" "$NUMEXPR_NUM_THREADS" __END_COMMAND__ >> "$CAPTURE_FILE"\n'
        'exit "${STUB_SETUP_EXIT:-0}"\n', encoding='utf-8', newline='\n')
    python = runtime/'env/bin/python'
    python.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        'printf "%s\\0" python "$@" __END_COMMAND__ >> "$CAPTURE_FILE"\n'
        'printf "%s\\0" python_env "$OMP_NUM_THREADS" "$OPENBLAS_NUM_THREADS" '
        '"$MKL_NUM_THREADS" "$NUMEXPR_NUM_THREADS" "$PYTHONDONTWRITEBYTECODE" '
        '"$CUDA_VISIBLE_DEVICES" __END_COMMAND__ >> "$CAPTURE_FILE"\n'
        'exit "${STUB_PYTHON_EXIT:-0}"\n', encoding='utf-8', newline='\n')
    python.chmod(0o755)
    env = {'XDG_DATA_HOME': data.as_posix(), 'CAPTURE_FILE': capture.as_posix()}
    return runtime, capture, env


def captured_calls(path):
    calls, current = [], []
    for value in path.read_bytes().decode('utf-8').split('\0'):
        if value == '__END_COMMAND__':
            calls.append(current)
            current=[]
        elif value or current:
            current.append(value)
    assert not current or current == ['']
    return calls


def option_value(arguments, name):
    index = max(i for i, value in enumerate(arguments) if value == name)
    return arguments[index+1]


def test_benchmark_shell_syntax(bash):
    result = subprocess.run([bash, '-n', LAUNCHER.as_posix()], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_plan_only_needs_no_environment_and_does_not_create_files(bash,bundle):
    root,script=bundle
    data=root/'must not be created'
    result=invoke(bash,script,'--plan-only',env={'XDG_DATA_HOME':data.as_posix()})
    assert result.returncode == 0, result.stderr
    assert list(root.iterdir()) == [script]
    assert 'no setup, staging, directory creation or computation' in result.stdout
    assert 'benchmark RAM reserve: 2 GiB' in result.stdout
    assert 'Native library initialization: 20 threads' in result.stdout
    assert 'No winning layout is applied' in result.stdout
    line=next(line for line in result.stdout.splitlines() if ' -B -u benchmark_gp_execution.py ' in line)
    args=shlex.split(line)
    for name,value in {'--profiles':'2x4,2x10,1x20','--train-grids':'2000','--tasks':'4','--repeats':'2'}.items():
        assert option_value(args,name)==value
    assert '--cpu-budget 20 --memory-budget-gib 8' in result.stdout
    assert '/wsl/setup.sh ' in result.stdout and '--skip-check ' in result.stdout


def test_default_dispatch_initializes_native_capacity_before_setup(bash,bundle):
    root,script=bundle
    runtime,capture,env=install_stubs(root)
    result=invoke(bash,script,env=env)
    assert result.returncode == 0, result.stderr
    calls=captured_calls(capture)
    assert [row[0] for row in calls] == ['setup','setup_env','python','python_env']
    assert calls[0][2:] == ['--skip-check']
    assert calls[1][1:] == ['20']*4
    assert calls[3][1:] == ['20']*4+['1','']
    args=calls[2][1:]
    assert args[:3] == ['-B','-u','benchmark_gp_execution.py']
    expected={'--profiles':'2x4,2x10,1x20','--train-grids':'2000','--tasks':'4',
              '--repeats':'2','--cpu-budget':'20','--memory-budget-gib':'8'}
    for name,value in expected.items():
        assert option_value(args,name)==value
    assert option_value(args,'--output').startswith('outputs/benchmarks/gp_ram16_layout_')
    assert option_value(args,'--output').endswith('.json')
    assert (runtime/'workspace/outputs/benchmarks').is_dir()
    assert not {'--memory-reserve-gib','--new-run','--replicates'} & set(args)


def test_user_overrides_stay_last_and_unicode_path_is_one_argument(bash,bundle):
    root,script=bundle
    _,capture,env=install_stubs(root)
    overrides=['--profiles','1x4','--train-grids','256','--tasks','1','--repeats','1',
               '--cpu-budget','4','--memory-budget-gib','3',
               '--output','outputs/benchmarks/自己的 測試.json']
    result=invoke(bash,script,*overrides,env=env)
    assert result.returncode == 0, result.stderr
    args=next(row[1:] for row in captured_calls(capture) if row[0]=='python')
    assert args[-len(overrides):] == overrides
    for name in overrides[::2]:
        assert option_value(args,name) == option_value(overrides,name)


def test_plan_only_is_consumed_at_any_position(bash,bundle):
    root,script=bundle
    result=invoke(bash,script,'--profiles','1x4','--plan-only','--train-grids','256')
    assert result.returncode == 0, result.stderr
    command=next(line for line in result.stdout.splitlines() if ' -B -u benchmark_gp_execution.py ' in line)
    assert '--plan-only' not in command
    assert command.endswith('--profiles 1x4 --train-grids 256 ')
    assert list(root.iterdir()) == [script]


@pytest.mark.parametrize('failure,code,python_expected', [
    ('STUB_SETUP_EXIT',7,False), ('STUB_PYTHON_EXIT',9,True),
    ('STUB_PYTHON_EXIT',2,True),
])
def test_setup_or_benchmark_failure_propagates(bash,bundle,failure,code,python_expected):
    root,script=bundle
    runtime,capture,env=install_stubs(root)
    env[failure]=str(code)
    result=invoke(bash,script,env=env)
    assert result.returncode==code
    calls=captured_calls(capture)
    assert any(row[0]=='python' for row in calls)==python_expected
    assert (runtime/'workspace/outputs/benchmarks').exists()==python_expected
