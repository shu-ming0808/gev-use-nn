"""Locked 21D / 100k / one-seed MSE versus MSE + observed-support study.

Reuses the frozen notebook architecture and training loop without editing them.
Only the training objective changes. All checkpoints, scheduler decisions and
early stopping still use validation coefficient MSE. No production pointers
or weights are written. Test is loaded only after every run has finished.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from cheng_nn_evaluation import check_gev_predictions
from cheng_nn_simulation import COEFFICIENT_NAMES
from compare_cheng_nn_inputs import FIXED, make_features, now, read_json
from compare_cheng_nn_mixed_sampling import DEFAULT_DATA, SCHEME, SCENARIOS, check_hashes
from train_cheng_nn import (PROJECT_ROOT, atomic_array, atomic_csv, atomic_json,
                           atomic_torch_save, file_sha256, load_notebook_components)

SEED = 20260929
WEIGHTS = (0.0, 0.1, 1.0)
MARGIN = 1e-3  # Dimensionless training safety margin, NOT a temperature cutoff.
SIZES = {'train': 80000, 'validation': 10000, 'test': 10000}
OFFICIAL = PROJECT_ROOT / 'results/cheng_nn_17d/mixed_sampling_20261006/input21_seed20260929/best.pt'
OFFICIAL_SHA256 = '534e3877a15826f5b443f71d74804edc8628821a3f0fec4b32b192c2162134b2'
DEFAULT_OUTPUT = PROJECT_ROOT / 'results/cheng_nn_21d/support_penalty_100k_20261012'
FILES = {'targets': 'targets_standardized.npy', 'coefficients': 'coefficients_original.npy',
         'location_scale': 'sample_location_scale.npy', 'annual_maxima': 'annual_maxima.npy',
         'scenario': 'scenario_id.npy'}


def support_margin(coefficients, observations, centered_time):
    """Differentiable g = 1 + xi (z - mu_z(t)) / sigma_z(t), in float64.

    Coefficients must ALREADY be inverse-transformed from training target-z
    scale to sample-standardized observation scale. Both signs of xi work;
    xi == 0 gives g == 1. No endpoint, coefficient, or observation clipping.
    """
    c, z, t = coefficients.double(), observations.double(), centered_time.double()
    mu = c[:, 0:1] + c[:, 1:2] * t[None, :]
    eta = c[:, 2:3] + c[:, 3:4] * t[None, :]
    return 1.0 + c[:, 4:5] * (z - mu) * torch.exp(-eta)


def support_penalty(coefficients, observations, centered_time, margin=MARGIN):
    """Mean per-sequence squared hinge on its worst yearly support margin.

    Each 50-year sequence contributes once; sparse violations are not diluted
    by averaging over 50 years. A soft penalty is not a feasibility guarantee.
    """
    if not np.isfinite(margin) or not 0 <= margin < 1:
        raise ValueError('Training support margin must be finite and in [0, 1).')
    g = support_margin(coefficients, observations, centered_time)
    return torch.relu(margin - g.amin(dim=1)).square().mean()


class SupportDataset(Dataset):
    """Keep input 21D; original observations are used only to compute loss."""
    def __init__(self, data, scaler):
        self.inputs = data['inputs']
        self.targets = data['targets']
        self.scaler = scaler
        scaling = data['location_scale']
        self.z = (data['annual_maxima'] - scaling[:, :1]) / scaling[:, 1:2]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        x = np.array(self.inputs[index], dtype=np.float32, copy=True)
        y = self.scaler.transform(self.targets[index]).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(self.z[index].copy())


class SupportObjective:
    """Drop-in mean_batch_loss; return pure MSE to the unchanged selection loop."""
    def __init__(self, scaler, centered_time, weight, device, margin=MARGIN):
        if not np.isfinite(weight) or weight < 0:
            raise ValueError('Penalty weight must be finite and nonnegative.')
        self.mean = torch.as_tensor(scaler.mean, dtype=torch.float64, device=device)
        self.std = torch.as_tensor(scaler.std, dtype=torch.float64, device=device)
        self.time = torch.as_tensor(centered_time, dtype=torch.float64, device=device)
        self.weight, self.margin = float(weight), float(margin)
        self.last_training = {}

    def __call__(self, model, loader, device, optimizer=None, l1_lambda=0.0):
        if l1_lambda != 0:
            raise ValueError('This study fixes L1 = 0.')
        training = optimizer is not None
        model.train(training)
        total_mse, total_penalty, count = 0.0, 0.0, 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for inputs, targets, observations in loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                predicted = model(inputs)
                mse = torch.mean((predicted - targets) ** 2)
                objective = mse
                penalty = None
                if training and self.weight:
                    c = predicted.double() * self.std + self.mean
                    penalty = support_penalty(c, observations.to(device, non_blocking=True),
                                              self.time, self.margin)
                    objective = mse + self.weight * penalty
                if not torch.isfinite(objective):
                    raise FloatingPointError('Nonfinite objective: stop; do not clip or omit cases.')
                if training:
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0,
                                                   error_if_nonfinite=True)
                    optimizer.step()
                n = len(inputs)
                total_mse += float(mse.detach()) * n
                if penalty is not None:
                    total_penalty += float(penalty.detach()) * n
                count += n
        if training:
            self.last_training = dict(train_support_penalty_online=(total_penalty/count if self.weight else None),
                                      train_objective_online=(total_mse+self.weight*total_penalty)/count)
        return total_mse / count


def load_split(root, split):
    """Verify existing data; never regenerate, mutate, or fit a scaler here."""
    directory = Path(root) / split
    meta = read_json(directory / 'metadata.json')
    if (meta['sampling_scheme'] != SCHEME or meta['scenario_names'] != list(SCENARIOS)
            or meta['n_samples'] != SIZES[split] or meta['years'] != list(range(1976, 2026))
            or meta['reference_year'] != 2000.5 or meta['time_scale_years'] != 10):
        raise ValueError(f'Dataset design mismatch: {split}')
    check_hashes(directory, meta['files_sha256'])
    data = {k: np.load(directory/v, mmap_mode='r') for k,v in FILES.items()}
    if not np.array_equal(np.bincount(data['scenario'], minlength=4), [SIZES[split]//4]*4):
        raise ValueError(f'Scenarios not equally balanced: {split}')
    if any(not np.isfinite(a).all() for a in data.values()):
        raise ValueError(f'Nonfinite data: {split}')
    data['metadata'] = meta
    data['inputs'] = make_features(data['annual_maxima'], data['location_scale'], 21)
    return data


def model_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_one(directory, data, weight, device, protocol):
    directory.mkdir(exist_ok=False)
    ns = load_notebook_components()
    ns['INPUT_FEATURE_NAMES'] = tuple(f'input_{i}' for i in range(21))
    ns['LEARNING_RATE'] = FIXED['learning_rate']
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if device == 'cuda':
        torch.cuda.manual_seed_all(SEED)
    scaler = ns['TargetScaler'].fit(data['train']['targets'])
    original_factory = ns['ChengGEVNet']
    initial = {}

    def tracked_factory(*args, **kwargs):
        model = original_factory(*args, **kwargs)
        initial['model_sha256'] = model_digest(model)
        return model

    ns['ChengGEVNet'] = tracked_factory
    objective = SupportObjective(scaler, data['train']['metadata']['centered_time'], weight, device)
    ns['mean_batch_loss'] = objective
    generator = torch.Generator().manual_seed(SEED)
    loaders = {s: DataLoader(SupportDataset(d, scaler), batch_size=FIXED['batch_size'],
                  shuffle=s=='train', num_workers=0, pin_memory=device=='cuda',
                  generator=generator if s=='train' else None) for s,d in data.items()}
    train_eval = DataLoader(loaders['train'].dataset, batch_size=2048, shuffle=False,
                  num_workers=0, pin_memory=device=='cuda', generator=torch.Generator().manual_seed(SEED))
    started = time.perf_counter()
    metadata = dict(status='running', penalty_weight=weight, margin=MARGIN, seed=SEED,
                    fixed=FIXED, device=device, created_utc=now(), best_epoch=None,
                    checkpoint_monitor='validation coefficient MSE (no penalty)',
                    protocol_sha256=file_sha256(directory.parent/'protocol.json'))

    def checkpoint(model, optimizer, scheduler, history, improved, stale):
        history[-1].update(objective.last_training)
        if improved:
            metadata.update(best_epoch=history[-1]['epoch'], best_validation_loss=history[-1]['validation_loss'])
        metadata.update(last_epoch=history[-1]['epoch'], elapsed_seconds=time.perf_counter()-started,
                        initial_model_sha256=initial['model_sha256'])
        if improved:
            payload = dict(model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                target_scaler=scaler.state_dict(), coefficient_names=list(COEFFICIENT_NAMES),
                input_shape=[21], hidden_sizes=FIXED['hidden_sizes'], dropout_p=0.,
                architecture=[21,128,128,64,5], epoch=history[-1]['epoch'], best_epoch=metadata['best_epoch'],
                simulation_metadata=data['train']['metadata'], input_representation='21D',
                source_hashes=protocol['source_hashes'], penalty_weight=weight, support_margin=MARGIN,
                experimental_only=True)
            atomic_torch_save(directory/'best.pt', payload)
        atomic_csv(directory/'training_history.csv', pd.DataFrame(history))
        atomic_json(directory/'run_metadata.json', metadata)

    atomic_json(directory/'run_metadata.json', metadata)
    try:
        model, history = ns['train_cheng_model'](loaders['train'], loaders['validation'],
            FIXED['max_epochs'], FIXED['patience'], device, epoch_callback=checkpoint,
            weight_decay=FIXED['weight_decay'], hidden_sizes=tuple(FIXED['hidden_sizes']),
            dropout_p=0., optimizer_name='Adam', l1_lambda=0., train_eval_loader=train_eval)
        if device == 'cuda':
            torch.cuda.synchronize()
        metadata.update(status='completed', completed_utc=now(), training_seconds=time.perf_counter()-started,
                        best_model_sha256=model_digest(model), checkpoint_sha256=file_sha256(directory/'best.pt'))
    except BaseException as exc:
        metadata.update(status='failed', error=repr(exc))
        raise
    finally:
        atomic_json(directory/'run_metadata.json', metadata)
    return metadata


def evaluate(directory, data, weight, device):
    ns = load_notebook_components()
    ns['INPUT_FEATURE_NAMES'] = tuple(f'input_{i}' for i in range(21))
    # Only this locally generated checkpoint is unpickled, never downloaded weights.
    payload = torch.load(directory/'best.pt', map_location='cpu', weights_only=False)
    model = ns['ChengGEVNet'](hidden_sizes=(128,128,64), dropout_p=0.).to(device)
    model.load_state_dict(payload['model_state'])
    scaler = ns['TargetScaler'](**payload['target_scaler'])
    rmse_rows, support_rows = [], []
    for split, d in data.items():
        standardized = ns['predict_standardized_coefficients'](model, d['inputs'], scaler, device, batch_size=2048)
        original = ns['inverse_sample_standardization'](standardized, d['location_scale'])
        if not np.isfinite(original).all():
            raise FloatingPointError('Nonfinite predictions: no omission allowed.')
        errors = original - d['coefficients']
        zerrors = scaler.transform(standardized) - scaler.transform(d['targets'])
        checks = check_gev_predictions(original, d['annual_maxima'], np.arange(1976,2026), reference_year=2000.5)
        dest = directory/split
        dest.mkdir(exist_ok=False)
        atomic_array(dest/'predictions_original.npy', original)
        atomic_csv(dest/'gev_validity.csv', checks)
        for scenario, selected in [('all', np.ones(len(original),dtype=bool))] + [
                (name, d['scenario']==i) for i,name in enumerate(SCENARIOS)]:
            n = int(selected.sum())
            se = errors[selected]
            for j,name in enumerate(COEFFICIENT_NAMES):
                rmse_rows.append(dict(weight=weight, split=split, scenario=scenario, coefficient=name,
                    n=n, RMSE=float(np.sqrt(np.mean(se[:,j]**2))), bias=float(np.mean(se[:,j]))))
            ch = checks.loc[selected]
            failed = int((ch['n_support_violations']>0).sum())
            support_rows.append(dict(weight=weight, split=split, scenario=scenario, n=n,
                support_failed_sequences=failed, support_failure_pct=100*failed/n,
                support_violating_years=int(ch['n_support_violations'].sum()),
                invalid_parameter_sequences=int((~ch['parameters_valid_all_years']).sum()),
                numerical_failure_sequences=int((ch['n_support_numerical_failures']>0).sum()),
                combined_failed_sequences=int((~ch['gev_valid_all_years']).sum()),
                coefficient_RMSE_train_target_z=float(np.sqrt(np.mean(zerrors[selected].astype(float)**2)))))
    return rmse_rows, support_rows


def run(output, data_root, device):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; do not silently change device.')
    torch.set_num_threads(FIXED['threads'])
    if file_sha256(OFFICIAL) != OFFICIAL_SHA256:
        raise ValueError('Official checkpoint identity changed; stop before running.')
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(OFFICIAL, map_location='cpu', weights_only=False)
    sources = {str(Path(__file__).resolve()): file_sha256(__file__)}
    for path, digest in checkpoint['source_hashes'].items():
        if file_sha256(path) != digest:
            raise ValueError(f'Frozen source changed: {path}')
        sources[path] = digest
    metadata = {s: read_json(data_root/s/'metadata.json') for s in SIZES}
    protocol = dict(created_utc=now(), fixed=FIXED, architecture=[21,128,128,64,5],
        trainable_parameters=27909, training_seed=SEED, split_sizes=SIZES, weights=list(WEIGHTS),
        data_directory=str(data_root), data_sha256={s:m['files_sha256'] for s,m in metadata.items()},
        source_hashes=sources, official_checkpoint=str(OFFICIAL), official_sha256=OFFICIAL_SHA256,
        margin=MARGIN, penalty='mean_i max_t relu(epsilon - (1+xi_i*(z_it-mu_it)/sigma_it))^2',
        loss='training-target-z coefficient MSE + lambda * penalty; Adam L2 weight decay unchanged',
        precision='float32 network/MSE; float64 support calculation, with differentiable inverse target scaling',
        penalty_weight_rationale='Exploratory fixed 0/0.1/1 grid, not a literature optimum.',
        checkpoint_selection='Pure validation coefficient MSE, original tolerance 1e-7',
        scheduler='ReduceLROnPlateau on validation MSE, factor 0.5, patience 6, min_lr 1e-6',
        gradient_norm_clip=5.,
        test_protocol='Evaluate all three predeclared arms once, after all training and validation export; no tuning from test.',
        test_limitation='Existing test has been used in previous research; not a new untouched confirmatory test.',
        failure_definition='Sequence has >=1 observed year with g<=0; numeric/parameter failures reported separately.',
        final_model_replacement=False, real_global_data_evaluated=False, return_levels_evaluated=False)
    atomic_json(output/'protocol.json', protocol)
    state = dict(status='loading_train_validation', created_utc=now())
    atomic_json(output/'status.json', state)
    try:
        data = {s:load_split(data_root,s) for s in ('train','validation')}
        # Affine transformations and target scalers must match the official model.
        ns = load_notebook_components()
        scaler = ns['TargetScaler'].fit(data['train']['targets'])
        for key in ('mean','std'):
            np.testing.assert_array_equal(getattr(scaler,key),checkpoint['target_scaler'][key])
        runs = []
        for weight in WEIGHTS:
            name = f'lambda_{weight:g}'
            state.update(status='training', current_arm=name)
            atomic_json(output/'status.json', state)
            print(f'ARM {name}', flush=True)
            runs.append(train_one(output/name, data, weight, device, protocol))
        if len({r['initial_model_sha256'] for r in runs}) != 1:
            raise RuntimeError('Paired initialization mismatch.')
        # Freeze validation results before any test arrays are loaded.
        all_rmse, all_support = [], []
        for weight in WEIGHTS:
            a,b = evaluate(output/f'lambda_{weight:g}', data, weight, device)
            all_rmse += a; all_support += b
        atomic_csv(output/'pretest_coefficient_rmse.csv',pd.DataFrame(all_rmse))
        atomic_csv(output/'pretest_support_summary.csv',pd.DataFrame(all_support))
        atomic_json(output/'pretest_lock.json',dict(locked_utc=now(),
            runs=runs, weights=list(WEIGHTS), no_additional_arms_allowed=True))
        state.update(status='final_test_evaluation')
        atomic_json(output/'status.json', state)
        test = {'test':load_split(data_root,'test')}
        for weight in WEIGHTS:
            a,b = evaluate(output/f'lambda_{weight:g}',test,weight,device)
            all_rmse += a; all_support += b
        rmse, support = pd.DataFrame(all_rmse),pd.DataFrame(all_support)
        atomic_csv(output/'coefficient_rmse.csv',rmse)
        atomic_csv(output/'support_summary.csv',support)
        table = rmse[rmse.scenario.eq('all')].pivot(index=['split','weight'],columns='coefficient',values='RMSE').reset_index()
        table = table.merge(support[support.scenario.eq('all')].drop(columns='scenario'),on=['split','weight'])
        atomic_csv(output/'comparison.csv',table)
        base = torch.load(output/'lambda_0/best.pt',map_location='cpu',weights_only=False)
        baseline_delta = max(float((base['model_state'][k]-v).abs().max()) for k,v in checkpoint['model_state'].items())
        if file_sha256(OFFICIAL) != OFFICIAL_SHA256 or any(file_sha256(p)!=h for p,h in sources.items()):
            raise RuntimeError('Protected checkpoint/source changed during experiment.')
        state.update(status='completed',completed_utc=now(),runs=runs,
            official_checkpoint_unchanged=True,original_sources_unchanged=True,
            control_vs_official_max_weight_difference=baseline_delta,
            control_best_epoch=runs[0]['best_epoch'], official_best_epoch=checkpoint['best_epoch'])
        print(table[table.split.eq('test')].to_string(index=False),flush=True)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc))
        raise
    finally:
        atomic_json(output/'status.json',state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--data-directory',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args = parser.parse_args()
    run(args.output.resolve(),args.data_directory.resolve(),args.device)
