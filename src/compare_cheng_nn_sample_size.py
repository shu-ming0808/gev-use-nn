"""One-seed 21D learning curve: nested 80k versus 320k training cases.

The nominal 400k pool has 320k train, 40k validation and 40k sealed test.
Only the original 10k validation subset selects checkpoints in BOTH arms.
The other 30k validation cases are reserved, not used for tuning/reporting.
Existing 100k results, models and data are never replaced.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

import compare_cheng_nn_mixed_sampling as mixed
from cheng_nn_simulation import COEFFICIENT_NAMES, ParameterRanges, SimulationConfig, _open_split_arrays
from compare_cheng_nn_inputs import make_features, now, read_json
from compare_cheng_nn_lr_l2 import experiment_lock
from train_cheng_nn import PROJECT_ROOT, atomic_array, atomic_csv, atomic_json, file_sha256

DEFAULT_BASELINE = mixed.DEFAULT_OUTPUT
DEFAULT_DATA = PROJECT_ROOT / 'data/simulated/cheng_nn_mixed_400k_20261006'
DEFAULT_OUTPUT = PROJECT_ROOT / 'results/cheng_nn_17d/sample_size_21d_400k_20261006'
TRAINING_SEED = 20260929
SIMULATION_SEED = 202610064


def new_partial(root, name):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'.{name}.partial'
    i = 1
    while path.exists():
        i += 1
        path = root / f'.{name}.partial_attempt{i}'
    path.mkdir()
    return path


def copy_common_validation(source, target):
    """Byte-identical copy, including metadata, without touching test."""
    metadata = read_json(source / 'metadata.json')
    mixed.check_hashes(source, metadata['files_sha256'])
    if target.exists():
        if file_sha256(source/'metadata.json') != file_sha256(target/'metadata.json'):
            raise ValueError('Existing common validation metadata differs.')
        mixed.check_hashes(target, metadata['files_sha256'])
        return metadata
    partial = new_partial(target.parent, target.name)
    for name in [*metadata['files_sha256'], 'metadata.json']:
        shutil.copy2(source/name, partial/name)
    mixed.check_hashes(partial, metadata['files_sha256'])
    partial.replace(target)
    return metadata


def extend_training(source, target, n_total, seed):
    """Keep every original training case, then append independent balanced cases."""
    metadata = read_json(source/'metadata.json')
    mixed.check_hashes(source, metadata['files_sha256'])
    n_base = metadata['n_samples']
    n_extra = n_total - n_base
    if n_base % 4 or n_extra < 4 or n_extra % 4:
        raise ValueError('Expansion must add a positive multiple of four cases.')
    identity = dict(n_samples=n_total, parent_metadata_sha256=file_sha256(source/'metadata.json'),
                    extension_seed=int(seed), extension_source_sha256=file_sha256(Path(__file__)))
    if target.exists():
        saved = read_json(target/'metadata.json')
        if any(saved.get(k) != v for k, v in identity.items()):
            raise ValueError('Existing expanded training data differ.')
        mixed.check_hashes(target, saved['files_sha256'])
        return saved
    config, ranges = SimulationConfig(), ParameterRanges()
    rng = np.random.default_rng(seed)
    original_labels = np.load(source/'scenario_id.npy')
    np.testing.assert_array_equal(np.bincount(original_labels, minlength=4), [n_base//4]*4)
    extra_labels = mixed.balanced_scenarios(n_extra, rng)
    labels = np.concatenate([original_labels, extra_labels])
    partial = new_partial(target.parent, target.name)
    arrays = _open_split_arrays(partial, n_total, config.n_years)
    for key, filename in mixed.ARRAY_FILES.items():
        original = np.load(source/filename, mmap_mode='r')
        arrays[key][:n_base] = original
        del original
    for start in range(0, n_extra, config.chunk_size):
        stop = min(start + config.chunk_size, n_extra)
        values = mixed.mixed_chunk(rng, extra_labels[start:stop], config, ranges)
        for key, value in zip(mixed.ARRAY_FILES, values):
            arrays[key][n_base+start:n_base+stop] = value
    for array in arrays.values():
        array.flush()
    del array, arrays
    atomic_array(partial/'scenario_id.npy', labels)
    hashes = {p.name: file_sha256(p) for p in sorted(partial.glob('*.npy'))}
    result = {**metadata, **identity, 'seed': int(seed), 'created_utc': now(),
              'parent_directory': str(source.resolve()), 'n_reused_train': n_base,
              'n_new_train': n_extra, 'scenario_counts': {s:n_total//4 for s in mixed.SCENARIOS},
              'files_sha256': hashes,
              'nested_sampling': 'First n_reused_train rows equal the original training arrays exactly.'}
    atomic_json(partial/'metadata.json', result)
    partial.replace(target)
    return result


def prepare(output, data_dir, baseline, *, training_seed=TRAINING_SEED,
            simulation_seed=SIMULATION_SEED, sizes=(320000, 30000, 40000)):
    """sizes = expanded train, RESERVED validation, sealed test."""
    output, data_dir, baseline = (Path(p).resolve() for p in (output, data_dir, baseline))
    if (output/'experiment.json').exists():
        raise FileExistsError('Existing experiment: use --resume.')
    old = read_json(baseline/'experiment.json')
    mixed.validate_manifest(old)
    matches = [r for r in old['runs'] if r['dimension']==21 and r['seed']==training_seed]
    if len(matches) != 1 or matches[0]['status'] != 'completed':
        raise ValueError('Need exactly one completed, matching 21D baseline seed.')
    baseline_entry = matches[0]
    mixed.validate_run(baseline_entry, old)
    root = Path(old['data_directory'])
    output.mkdir(parents=True, exist_ok=True)
    children = [int(c.generate_state(1, dtype=np.uint32)[0])
                for c in np.random.SeedSequence(simulation_seed).spawn(3)]
    train_meta = extend_training(root/'train', data_dir/'train', sizes[0], children[0])
    validation_meta = copy_common_validation(root/'validation', data_dir/'validation')
    config, ranges = SimulationConfig(), ParameterRanges()
    for split, count, seed in [('validation_reserve',sizes[1],children[1]), ('test',sizes[2],children[2])]:
        mixed.generate_split(data_dir, split, count, seed, config, ranges)
    print(f'Data ready: {sizes[0]:,} train; {validation_meta["n_samples"]:,} common validation; '
          f'{sizes[1]:,} reserved validation; {sizes[2]:,} sealed test.', flush=True)
    _, data = mixed.load_data(data_dir)
    features = {'21': {}}
    for split, d in data.items():
        # Chunk feature computation to avoid a full 320k x 50 temporary matrix.
        path = output/f'features_21_{split}.npy'
        temp = path.with_suffix('.npy.partial')
        x = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float32, shape=(len(d['inputs']),21))
        for start in range(0, len(x), config.chunk_size):
            stop = min(start+config.chunk_size, len(x))
            x[start:stop] = make_features(d['annual_maxima'][start:stop],d['location_scale'][start:stop],21)
        x.flush(); del x
        temp.replace(path)
        features['21'][split] = dict(path=str(path),sha256=file_sha256(path))
    # Validation inputs must also be identical, not merely drawn from the same law.
    np.testing.assert_array_equal(np.load(features['21']['validation']['path']),
                                  np.load(old['features']['21']['validation']['path']))
    sources = {**old['source_hashes'], str(Path(__file__).resolve()):file_sha256(Path(__file__))}
    metas = {'train':train_meta, 'validation':validation_meta}
    manifest = dict(status='prepared',created_utc=now(),fixed=dict(old['fixed']),
        runs=[dict(dimension=21,seed=training_seed,status='pending',path=str(output/f'input21_seed{training_seed}'))],
        data_directory=str(data_dir),baseline_directory=str(baseline),baseline_entry=baseline_entry,
        baseline_experiment_sha256=file_sha256(baseline/'experiment.json'),
        simulation_seed=simulation_seed,scenario_names=list(mixed.SCENARIOS),
        source_hashes=sources,features=features,
        split_sizes=dict(train=sizes[0],validation=validation_meta['n_samples'],
                         validation_reserve=sizes[1],test=sizes[2]),
        nominal_total=sizes[0]+validation_meta['n_samples']+sizes[1]+sizes[2],
        dataset_files_sha256={s:v['files_sha256'] for s,v in metas.items()},
        dataset_metadata_sha256={s:file_sha256(data_dir/s/'metadata.json') for s in metas},
        checkpoint_selection='SAME original 10k validation subset, coefficient MSE; no test access',
        evaluation_note='One fixed seed. Remaining validation and all test cases stay sealed. '
                        'Equal epochs imply more gradient updates for the larger training set.')
    atomic_json(output/'experiment.json', manifest)
    return manifest


def validate_baseline(manifest):
    base = Path(manifest['baseline_directory'])
    if file_sha256(base/'experiment.json') != manifest['baseline_experiment_sha256']:
        raise ValueError('Baseline experiment changed.')
    old = read_json(base/'experiment.json')
    mixed.validate_manifest(old)
    mixed.validate_run(manifest['baseline_entry'], old)
    if old['fixed'] != manifest['fixed']:
        raise ValueError('Training settings differ between sample-size arms.')
    if old['dataset_files_sha256']['validation'] != manifest['dataset_files_sha256']['validation']:
        raise ValueError('Validation differs between arms.')
    return old


def report(output):
    output = Path(output)
    m = read_json(output/'experiment.json')
    mixed.validate_manifest(m)
    old = validate_baseline(m)
    entry = m['runs'][0]
    if entry['status'] != 'completed':
        raise ValueError('Wait for the 400k model to finish.')
    mixed.validate_run(entry, m)
    d = Path(m['data_directory'])/'validation'
    truth, labels = np.load(d/'coefficients_original.npy'), np.load(d/'scenario_id.npy')
    predictions, diagnostics = {}, []
    for arm, e, source in [('100k',m['baseline_entry'],old),('400k',entry,m)]:
        path = Path(e['path'])
        pred = np.load(path/'validation/predictions_original.npy')
        if pred.shape != truth.shape or not np.isfinite(pred).all():
            raise ValueError('Invalid predictions: no cases may be silently excluded.')
        predictions[arm] = pred
        meta = read_json(path/'run_metadata.json')
        validity = pd.read_csv(path/'validation/gev_validity.csv')
        scaled = {}
        for split in ('train','validation'):
            metrics = pd.read_csv(path/split/'coefficient_metrics.csv')
            if not (metrics.n_used == metrics.n_total).all():
                raise ValueError('Metrics excluded cases.')
            values = metrics.loc[metrics.scale.eq('train_target_z'),'RMSE'].to_numpy()
            scaled[split] = float(np.sqrt(np.mean(values**2)))
        diagnostics.append(dict(pool=arm,training_cases=source['split_sizes']['train'],
            seed=e['seed'],best_epoch=meta['best_epoch'],epochs_completed=meta['epochs_completed'],
            training_seconds=meta['training_seconds'],
            train_validation_gap_percent=100*(scaled['validation']/scaled['train']-1),
            invalid_parameters=int((~validity.parameters_valid_all_years).sum()),
            support_failed_cases=int((validity.n_support_violations>0).sum())))
    rows = []
    for label, mask in [('All',np.ones(len(labels),dtype=bool))]+[
            (s,labels==i) for i,s in enumerate(mixed.SCENARIOS)]:
        a = np.sqrt(np.mean((predictions['100k'][mask]-truth[mask])**2,axis=0))
        b = np.sqrt(np.mean((predictions['400k'][mask]-truth[mask])**2,axis=0))
        for j,name in enumerate(COEFFICIENT_NAMES):
            rows.append(dict(scenario=label,coefficient=name,n_cases=int(mask.sum()),
                RMSE_100k=a[j],RMSE_400k=b[j],reduction_percent=100*(1-b[j]/a[j])))
    table = pd.DataFrame(rows)
    atomic_csv(output/'coefficient_comparison.csv',table[table.scenario.eq('All')])
    atomic_csv(output/'scenario_comparison.csv',table[~table.scenario.eq('All')])
    atomic_csv(output/'training_diagnostics.csv',pd.DataFrame(diagnostics))
    atomic_json(output/'comparison_protocol.json',dict(completed_utc=now(),input_dimension=21,
        training_seed=entry['seed'],common_validation_cases=len(truth),test_used=False,
        baseline_checkpoint_sha256=m['baseline_entry']['checkpoint_sha256'],
        expanded_checkpoint_sha256=entry['checkpoint_sha256'],
        aggregation='Single seed, RMSE across cases, all five original-scale coefficients.',
        interpretation='Exploratory common-validation comparison, not an independent test. '
                       'Target scalers fitted separately on each training set. No RL or formal model replacement.'))
    print(table[table.scenario.eq('All')].to_string(index=False),flush=True)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=DEFAULT_BASELINE)
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.report_only:
        report(output)
        return
    output.mkdir(parents=True,exist_ok=True)
    with experiment_lock(output):
        marker = output/'experiment.json'
        if marker.exists():
            if not args.resume:
                raise FileExistsError('Use --resume; existing results are never replaced.')
            m = read_json(marker)
            if Path(m['data_directory']) != args.data.resolve() or Path(m['baseline_directory']) != args.baseline.resolve():
                raise ValueError('Resume paths differ.')
            mixed.validate_manifest(m)
            validate_baseline(m)
        else:
            m = prepare(output,args.data,args.baseline)
        if args.prepare_only:
            return
        entry = m['runs'][0]
        path = Path(entry['path'])
        if (path/'run_metadata.json').exists() and read_json(path/'run_metadata.json').get('status')=='completed':
            mixed.validate_run(entry,m)
            entry.update(status='completed',checkpoint_sha256=file_sha256(path/'best.pt'))
        if entry['status'] != 'completed':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; data preserved, no training started.')
            if path.exists():
                i = 2
                while path.with_name(path.name+f'_attempt{i}').exists():
                    i += 1
                path = path.with_name(path.name+f'_attempt{i}')
                entry['path'] = str(path)
            entry.update(status='running',started_utc=now())
            m.update(status='running',started_utc=now())
            m.pop('error',None)
            atomic_json(marker,m)
            try:
                print(f'START 21D 400k pool, seed={entry["seed"]}',flush=True)
                with path.with_suffix('.log').open('x',encoding='utf-8') as log:
                    result = subprocess.run([sys.executable,'-B','-u',str(Path(mixed.__file__)),
                        '--output',str(output),'--train-one','21','--seed',str(entry['seed']),
                        '--run-dir',str(path)],stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'Training failed; inspect {path.with_suffix(".log")}')
                mixed.validate_run(entry,m)
                entry.update(status='completed',completed_utc=now(),checkpoint_sha256=file_sha256(path/'best.pt'))
                m.update(status='completed',completed_utc=now())
            except BaseException as exc:
                m.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc))
                entry['status']=m['status']
                raise
            finally:
                atomic_json(marker,m)
        else:
            m.update(status='completed')
            atomic_json(marker,m)
        report(output)
        print('One-seed 100k/400k comparison complete. Test remains sealed.',flush=True)


if __name__ == '__main__':
    main()
