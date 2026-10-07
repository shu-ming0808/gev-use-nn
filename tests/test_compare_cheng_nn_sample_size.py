from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import compare_cheng_nn_sample_size as study
from cheng_nn_simulation import ParameterRanges, SimulationConfig
from train_cheng_nn import atomic_json, file_sha256


def test_nested_training_and_common_validation(tmp_path):
    source, target = tmp_path/'old',tmp_path/'new'
    for split in ('train','validation'):
        study.mixed.generate_split(source,split,16,42,SimulationConfig(),ParameterRanges())
    meta = study.extend_training(source/'train',target/'train',64,99)
    assert meta['n_reused_train']==16 and meta['n_new_train']==48
    for filename in [*study.mixed.ARRAY_FILES.values(),'scenario_id.npy']:
        np.testing.assert_array_equal(np.load(target/'train'/filename)[:16],np.load(source/'train'/filename))
    labels = np.load(target/'train/scenario_id.npy')
    np.testing.assert_array_equal(np.bincount(labels),[16]*4)
    c = np.load(target/'train/coefficients_original.npy')
    assert (c[np.isin(labels,[0,2]),1]==0).all()
    assert (c[np.isin(labels,[0,1]),3]==0).all()
    assert study.extend_training(source/'train',target/'train',64,99)==meta
    with pytest.raises(ValueError):
        study.extend_training(source/'train',target/'train',64,100)
    study.copy_common_validation(source/'validation',target/'validation')
    assert file_sha256(source/'validation/metadata.json')==file_sha256(target/'validation/metadata.json')


def test_single_seed_training_and_report_never_read_reserves(tmp_path,monkeypatch):
    base, olddata = tmp_path/'base',tmp_path/'olddata'
    old = study.mixed.prepare(base,olddata,sizes=(16,8,8))
    old['fixed'].update(max_epochs=2,patience=2)
    atomic_json(base/'experiment.json',old)
    entry = next(r for r in old['runs'] if r['dimension']==21 and r['seed']==study.TRAINING_SEED)
    study.mixed.train_one(base,21,entry['seed'],Path(entry['path']),device='cpu')
    entry.update(status='completed',checkpoint_sha256=file_sha256(Path(entry['path'])/'best.pt'))
    atomic_json(base/'experiment.json',old)
    output, newdata = tmp_path/'output',tmp_path/'newdata'
    m = study.prepare(output,newdata,base,sizes=(64,8,8))
    assert len(m['runs'])==1 and m['runs'][0]['seed']==study.TRAINING_SEED
    assert m['nominal_total']==88
    assert set(m['features']['21'])=={'train','validation'}
    study.validate_baseline(m)
    original_load = np.load
    def guarded(path,*args,**kwargs):
        parts = Path(path).relative_to(tmp_path).parts
        if 'test' in parts or 'validation_reserve' in parts:
            raise AssertionError('Sealed data opened')
        return original_load(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',guarded)
    new = m['runs'][0]
    study.mixed.train_one(output,21,new['seed'],Path(new['path']),device='cpu')
    new.update(status='completed',checkpoint_sha256=file_sha256(Path(new['path'])/'best.pt'))
    m['status']='completed'
    atomic_json(output/'experiment.json',m)
    result = study.report(output)
    assert result.shape[0]==25
    assert (result.loc[result.scenario=='All','n_cases']==8).all()
    assert (output/'training_diagnostics.csv').exists()
    assert not (Path(new['path'])/'test').exists()
    assert not (Path(new['path'])/'validation/return_level_overall.csv').exists()
