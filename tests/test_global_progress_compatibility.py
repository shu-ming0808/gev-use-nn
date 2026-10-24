import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'tools/global_preprocessing_bundle'
sys.path.insert(0,str(FOLDER))
spec=importlib.util.spec_from_file_location('progress_preprocess',FOLDER/'preprocess.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_progress_only_compatibility_is_exact():
    old={'code_hash':'old','grid_hash':'same'};new={'code_hash':'new','grid_hash':'same'}
    approved={'legacy_code_sha256':'old','current_code_sha256':'new'}
    assert module.compatible_progress_protocol(new,old,approved)==old
    for changed in ({**new,'grid_hash':'different'},{**new,'code_hash':'unapproved'}):
        with pytest.raises(ValueError): module.compatible_progress_protocol(changed,old,approved)


def test_packaged_compatibility_matches_current_code():
    compatibility=json.loads((FOLDER/'progress_fix_compatibility.json').read_text())
    assert compatibility['current_code_sha256']==module.code_hash()


def test_static_progress_duplicate_count_not_crash():
    import ast
    source=ast.parse((FOLDER/'preprocess.py').read_text(encoding='utf-8'))
    function=next(n for n in ast.walk(source) if isinstance(n,ast.FunctionDef) and n.name=='progress')
    tree=ast.Module(body=[function],type_ignores=[])
    calls=[]
    from types import SimpleNamespace
    namespace=dict(time=SimpleNamespace(perf_counter=lambda:1),started=0,regions=['EAS'],grids=[1,2],
        io=SimpleNamespace(utc_now=lambda:'now',write_json=lambda path,value:calls.append(value)),
        output=Path('.'),json=json)
    exec(compile(ast.fix_missing_locations(tree),'<progress-test>','exec'),namespace)
    namespace['progress']('static',n_grids=2,completed=1)
    assert calls[-1]['n_grids']==2 and calls[-1]['nn_executed'] is False
    with pytest.raises(ValueError): namespace['progress']('static',n_grids=3)
