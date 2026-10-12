"""Experimental support-constrained decoder for five time-varying GEV coefficients.

The trainable MLP still takes 21 summaries. A parameter-free decoder ALSO uses
the same observed 50-year sequence during training AND inference. It never uses
true coefficients at inference. This is not a summary-only estimator and is not
the stationary transformation of Rai et al. verbatim.

For sample-standardized z_t, mu_t = mu0 + beta_mu*t and b = beta_sigma, set
    A = max(0, max_t xi*(mu_t-z_t)*exp(-b*t)) / (1-epsilon)
    sigma0 = A + exp(delta), eta0 = log(sigma0).
Then sigma_t = sigma0*exp(b*t)>0 and all observed g_t>epsilon (0<epsilon<1),
in exact arithmetic. Both signs of xi, including zero, are allowed. The single
eta0 is decoded jointly, preserving the linear log-scale curve. Future/unseen
observations are NOT covered by this guarantee, nor is statistical accuracy.

Loss, scheduler and checkpoint selection use the decoded five coefficients in
the ORIGINAL training-target-z metric. Baseline is the hash-locked official
checkpoint, independently reproduced by the earlier lambda=0 control. No
production artifact is written. Test is evaluated only after a pretest lock.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cheng_nn_evaluation import check_gev_predictions
from cheng_nn_simulation import COEFFICIENT_NAMES
from compare_cheng_nn_inputs import FIXED, now, read_json
from compare_cheng_nn_mixed_sampling import DEFAULT_DATA, SCENARIOS
from compare_cheng_nn_support_penalty import (
    MARGIN, OFFICIAL, OFFICIAL_SHA256, SEED, SIZES,
    SupportDataset, load_split, model_digest, support_margin,
)
from train_cheng_nn import (
    PROJECT_ROOT, atomic_array, atomic_csv, atomic_json, atomic_torch_save,
    file_sha256, load_notebook_components,
)

DEFAULT_OUTPUT = PROJECT_ROOT / 'results/cheng_nn_21d/reparameterization_100k_20261012'
CONTROL = PROJECT_ROOT / 'results/cheng_nn_21d/support_penalty_100k_20261012/lambda_0'
DECODER_ID = 'five_coefficient_observed_support_scale_v1'


def decode_coefficients(latent, observations, centered_time, margin=MARGIN):
    """Map (mu0,beta_mu,delta,beta_sigma,xi) to five sample-scale coefficients.

    Float64 and logaddexp avoid exp(delta) overflow. Numerical overflow in the
    time multiplier is not silently clipped: callers reject nonfinite output.
    This map has no trainable parameters. At xi=0 the decoded eta0 is delta.

    Proof: let a_t=xi*(mu_t-z_t)*exp(-b*t). Then g_t=1-a_t/sigma0.
    If a_t<=0, g_t>=1. Otherwise sigma0>A>=a_t/(1-margin), hence
    g_t>margin. This includes all 50 years without changing beta_sigma.
    Every curve whose observed g_t>margin is representable by choosing
    delta=log(exp(eta0)-A). The boundary g_t=margin is approached as
    delta tends to minus infinity. Numerical results are checked separately.
    """
    if not np.isfinite(margin) or not 0 < margin < 1:
        raise ValueError('Support safety margin must be in (0, 1).')
    c, z, t = latent.double(), observations.double(), centered_time.double()
    if (c.ndim != 2 or c.shape[1] != 5 or t.ndim != 1
            or z.shape != (len(c), len(t))):
        raise ValueError('Need latent (N,5), observations (N,T), and time (T,).')
    mu = c[:, 0:1] + c[:, 1:2] * t[None, :]
    requirements = c[:, 4:5] * (mu-z) * torch.exp(-c[:, 3:4]*t[None, :])
    bound = requirements.amax(dim=1).clamp_min(0.0) / (1.0-margin)
    # log(0) is avoided even in the unselected branch to prevent NaN gradients.
    log_bound = torch.log(bound.clamp_min(torch.finfo(torch.float64).tiny))
    eta = torch.where(bound > 0, torch.logaddexp(c[:, 2], log_bound), c[:, 2])
    return torch.stack((c[:, 0], c[:, 1], eta, c[:, 3], c[:, 4]), dim=1)


class ReparameterizedObjective:
    """Drop-in loss for the unchanged frozen notebook training loop."""
    def __init__(self, scaler, centered_time, device, margin=MARGIN):
        self.mean = torch.as_tensor(scaler.mean, dtype=torch.float64, device=device)
        self.std = torch.as_tensor(scaler.std, dtype=torch.float64, device=device)
        self.time = torch.as_tensor(centered_time, dtype=torch.float64, device=device)
        self.margin = margin

    def __call__(self, model, loader, device, optimizer=None, l1_lambda=0.0):
        if l1_lambda != 0:
            raise ValueError('L1 is fixed to zero in this experiment.')
        training = optimizer is not None
        model.train(training)
        total, count = 0.0, 0
        with torch.enable_grad() if training else torch.no_grad():
            for inputs, targets, observations in loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                observations = observations.to(device, non_blocking=True)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                raw = model(inputs)
                latent = raw.double()*self.std + self.mean
                coefficients = decode_coefficients(latent, observations, self.time, self.margin)
                decoded_z = ((coefficients-self.mean)/self.std).float()
                loss = torch.mean((decoded_z-targets)**2)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite decoded loss; no clipping or omission.')
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                    optimizer.step()
                total += float(loss.detach())*len(inputs)
                count += len(inputs)
        return total/count


def train_candidate(directory, data, device, protocol):
    directory.mkdir(exist_ok=False)
    ns = load_notebook_components()
    ns['INPUT_FEATURE_NAMES'] = tuple(f'input_{i}' for i in range(21))
    ns['LEARNING_RATE'] = FIXED['learning_rate']
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device == 'cuda':
        torch.cuda.manual_seed_all(SEED)
    scaler = ns['TargetScaler'].fit(data['train']['targets'])
    initial = {}
    factory = ns['ChengGEVNet']

    def tracked_factory(*args, **kwargs):
        model = factory(*args, **kwargs)
        initial['sha256'] = model_digest(model)
        expected = protocol.get('baseline_initial_model_sha256')
        if expected is not None and initial['sha256'] != expected:
            raise RuntimeError('Initialization differs from the paired baseline.')
        return model

    ns['ChengGEVNet'] = tracked_factory
    ns['mean_batch_loss'] = ReparameterizedObjective(
        scaler, data['train']['metadata']['centered_time'], device)
    generator = torch.Generator().manual_seed(SEED)
    loaders = {s: DataLoader(SupportDataset(d, scaler), batch_size=FIXED['batch_size'],
               shuffle=s=='train', num_workers=0, pin_memory=device=='cuda',
               generator=generator if s=='train' else None) for s,d in data.items()}
    train_eval = DataLoader(loaders['train'].dataset, batch_size=2048, shuffle=False,
                 num_workers=0, pin_memory=device=='cuda',
                 generator=torch.Generator().manual_seed(SEED))
    started = time.perf_counter()
    metadata = dict(status='training', decoder=DECODER_ID, margin=MARGIN, seed=SEED,
        fixed=FIXED, device=device, created_utc=now(), best_epoch=None,
        protocol_sha256=file_sha256(directory.parent/'protocol.json'))

    def checkpoint(model, optimizer, scheduler, history, improved, stale):
        if improved:
            metadata.update(best_epoch=history[-1]['epoch'],
                            best_validation_loss=history[-1]['validation_loss'])
        metadata.update(last_epoch=history[-1]['epoch'],
                        initial_model_sha256=initial['sha256'],
                        elapsed_seconds=time.perf_counter()-started)
        if improved:
            payload = dict(model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                target_scaler=scaler.state_dict(), coefficient_names=list(COEFFICIENT_NAMES),
                latent_coefficient_names=['mu0','beta_mu','delta','beta_sigma','xi0'],
                input_shape=[21], hidden_sizes=[128,128,64], dropout_p=0.,
                architecture=[21,128,128,64,5], epoch=history[-1]['epoch'],
                best_epoch=metadata['best_epoch'], decoder=DECODER_ID, support_margin=MARGIN,
                requires_observed_sequence_at_inference=True,
                simulation_metadata=data['train']['metadata'],
                input_representation='21D summaries + observed sequence for parameter-free decoder',
                source_hashes=protocol['source_hashes'], experimental_only=True)
            atomic_torch_save(directory/'best.pt', payload)
        atomic_csv(directory/'training_history.csv', pd.DataFrame(history))
        atomic_json(directory/'run_metadata.json', metadata)

    atomic_json(directory/'run_metadata.json', metadata)
    try:
        model, history = ns['train_cheng_model'](loaders['train'], loaders['validation'],
            FIXED['max_epochs'], FIXED['patience'], device, epoch_callback=checkpoint,
            weight_decay=FIXED['weight_decay'], hidden_sizes=(128,128,64), dropout_p=0.,
            optimizer_name='Adam', l1_lambda=0., train_eval_loader=train_eval)
        if device == 'cuda':
            torch.cuda.synchronize()
        metadata.update(status='completed', completed_utc=now(),
            training_seconds=time.perf_counter()-started, best_model_sha256=model_digest(model),
            checkpoint_sha256=file_sha256(directory/'best.pt'))
    except BaseException as exc:
        metadata.update(status='failed', error=repr(exc))
        raise
    finally:
        atomic_json(directory/'run_metadata.json', metadata)
    return metadata


def predict_sample_coefficients(model, inputs, scaler, observations, centered_time,
                               device, reparameterized, margin=MARGIN, batch_size=2048):
    """Inference never receives true coefficient labels. Return float64 arrays."""
    mean = torch.as_tensor(scaler.mean, dtype=torch.float64, device=device)
    std = torch.as_tensor(scaler.std, dtype=torch.float64, device=device)
    t = torch.as_tensor(centered_time, dtype=torch.float64, device=device)
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            x = torch.as_tensor(np.array(inputs[start:start+batch_size], copy=True),
                                dtype=torch.float32, device=device)
            latent = model(x).double()*std + mean
            if reparameterized:
                z = torch.as_tensor(np.array(observations[start:start+batch_size], copy=True),
                                    dtype=torch.float64, device=device)
                latent = decode_coefficients(latent, z, t, margin)
            if not torch.isfinite(latent).all():
                raise FloatingPointError('Nonfinite coefficients; no omitted cases.')
            out.append(latent.cpu().numpy())
    return np.concatenate(out)


def evaluate(checkpoint_path, directory, data, arm, device):
    ns = load_notebook_components()
    ns['INPUT_FEATURE_NAMES'] = tuple(f'input_{i}' for i in range(21))
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    constrained = arm == 'reparameterized'
    if constrained and payload.get('decoder') != DECODER_ID:
        raise ValueError('Wrong decoder identity.')
    if not constrained and 'decoder' in payload:
        raise ValueError('Baseline must not use a constrained decoder.')
    model = ns['ChengGEVNet'](hidden_sizes=(128,128,64), dropout_p=0.).to(device)
    model.load_state_dict(payload['model_state'])
    scaler = ns['TargetScaler'](**payload['target_scaler'])
    rows, summary = [], []
    for split, d in data.items():
        scaling = d['location_scale']
        z = (d['annual_maxima']-scaling[:, :1])/scaling[:, 1:2]
        if constrained:
            standardized = predict_sample_coefficients(model, d['inputs'], scaler, z,
                d['metadata']['centered_time'], device, True)
        else:
            # Preserve the official inference path, including its float32
            # inverse target scaling, rather than subtly changing the baseline.
            standardized = ns['predict_standardized_coefficients'](
                model,d['inputs'],scaler,device,batch_size=2048)
        original = ns['inverse_sample_standardization'](standardized, scaling)
        if not np.isfinite(original).all():
            raise FloatingPointError('Nonfinite restored coefficients; no omissions.')
        checks = check_gev_predictions(original, d['annual_maxima'], np.arange(1976,2026),
                                       reference_year=2000.5)
        # Independent arithmetic in the shared checker runs on original units.
        if constrained and (not checks['gev_valid_all_years'].all()
                or (checks.min_support_margin < MARGIN-1e-10).any()):
            raise RuntimeError('Decoder failed the original-scale support audit.')
        dest = directory/split
        dest.mkdir(parents=True, exist_ok=False)
        atomic_array(dest/'predictions_standardized.npy', standardized)
        atomic_array(dest/'predictions_original.npy', original)
        atomic_csv(dest/'gev_validity.csv', checks)
        errors = original-d['coefficients']
        ze = (standardized-d['targets'])/np.asarray(scaler.std)
        for scenario, use in [('all',np.ones(len(original),dtype=bool))] + [
                (name,d['scenario']==i) for i,name in enumerate(SCENARIOS)]:
            n = int(use.sum())
            for j,name in enumerate(COEFFICIENT_NAMES):
                e = errors[use,j]
                rows.append(dict(arm=arm, split=split, scenario=scenario, coefficient=name,
                    n=n, RMSE=float(np.sqrt(np.mean(e*e))), bias=float(e.mean())))
            ch = checks.loc[use]
            failures = int((ch.n_support_violations > 0).sum())
            summary.append(dict(arm=arm,split=split,scenario=scenario,n=n,
                support_failed_sequences=failures, support_failure_pct=100*failures/n,
                support_violating_years=int(ch.n_support_violations.sum()),
                invalid_parameter_sequences=int((~ch.parameters_valid_all_years).sum()),
                numerical_failure_sequences=int((ch.n_support_numerical_failures>0).sum()),
                combined_failed_sequences=int((~ch.gev_valid_all_years).sum()),
                minimum_support_margin=float(ch.min_support_margin.min()),
                margin_shortfall_sequences=int((ch.min_support_margin<MARGIN-1e-10).sum()),
                coefficient_RMSE_train_target_z=float(np.sqrt(np.mean(ze[use]**2)))))
    return rows, summary


def validate_control(official):
    metadata = read_json(CONTROL/'run_metadata.json')
    prior = read_json(CONTROL.parent/'protocol.json')
    reproduced = torch.load(CONTROL/'best.pt', map_location='cpu', weights_only=False)
    if (metadata['status'] != 'completed' or metadata['seed'] != SEED
            or metadata['fixed'] != FIXED or metadata['penalty_weight'] != 0
            or prior['official_sha256'] != OFFICIAL_SHA256):
        raise RuntimeError('Reproduced baseline protocol does not match.')
    if any(not torch.equal(v,reproduced['model_state'][k]) for k,v in official['model_state'].items()):
        raise RuntimeError('Earlier control no longer matches official weights exactly.')
    return metadata, prior


def run(output, data_root, device):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU substitution.')
    torch.set_num_threads(FIXED['threads'])
    if file_sha256(OFFICIAL) != OFFICIAL_SHA256:
        raise RuntimeError('Official checkpoint identity changed.')
    official = torch.load(OFFICIAL, map_location='cpu', weights_only=False)
    control, prior = validate_control(official)
    sources = {str(Path(__file__).resolve()):file_sha256(__file__)}
    # Include the helpers this experiment imports, in addition to frozen sources.
    sources[str(PROJECT_ROOT/'src/compare_cheng_nn_support_penalty.py')] = file_sha256(
        PROJECT_ROOT/'src/compare_cheng_nn_support_penalty.py')
    for p,h in official['source_hashes'].items():
        if file_sha256(p) != h:
            raise RuntimeError(f'Frozen source changed: {p}')
        sources[p] = h
    metadata = {s:read_json(data_root/s/'metadata.json') for s in SIZES}
    for s in SIZES:
        if metadata[s]['files_sha256'] != prior['data_sha256'][s]:
            raise ValueError('Data differs from the paired baseline.')
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(created_utc=now(), architecture=[21,128,128,64,5], trainable_parameters=27909,
        fixed=FIXED, seed=SEED, split_sizes=SIZES, scenarios=list(SCENARIOS),
        arms=['original','reparameterized'], decoder=DECODER_ID, margin=MARGIN,
        formula='A=max(0,max_t xi*(mu0+beta_mu*t-z_t)*exp(-beta_sigma*t))/(1-epsilon); eta0=log(A+exp(delta))',
        guarantee='Finite, representable decoded parameters satisfy observed g_t>epsilon for all 50 times; not future observations.',
        inference_information='21 features into MLP; same observed standardized 50-year sequence additionally used by nontrainable decoder.',
        label_information='True coefficients only in supervised loss and error evaluation, never in decoder/inference.',
        loss='Pure coefficient MSE on the original training-target-z scale after decoding; Adam L2 unchanged.',
        latent_scaling='Original scaler used as a fixed affine preconditioner of the five raw outputs; the third latent is delta, NOT eta0.',
        precision='Float32 MLP and coefficient loss; float64 constrained decoder. Baseline retains exact original inference/scaling arithmetic.',
        selection='Decoded validation coefficient MSE; original scheduler, early stopping and gradient norm clip=5.',
        data_directory=str(data_root), data_sha256={s:m['files_sha256'] for s,m in metadata.items()},
        source_hashes=sources, official_checkpoint=str(OFFICIAL), official_sha256=OFFICIAL_SHA256,
        baseline_reused=True, baseline_reproduction=str(CONTROL),
        baseline_initial_model_sha256=control['initial_model_sha256'],
        baseline_reproduction_metadata=control,
        test_protocol='Load test arrays only after checkpoint and train/validation reports are locked. No tuning on test.',
        test_limitation='Existing test previously inspected; exploratory paired comparison, not a new confirmatory test.',
        research_status='Our time-varying extension motivated by Rai et al. (2024), not a published reproduction.',
        final_model_replacement=False, real_global_data_evaluated=False, return_levels_evaluated=False)
    atomic_json(output/'protocol.json', protocol)
    state = dict(status='loading_train_validation',created_utc=now())
    atomic_json(output/'status.json', state)
    try:
        data = {s:load_split(data_root,s) for s in ('train','validation')}
        ns = load_notebook_components()
        scaler = ns['TargetScaler'].fit(data['train']['targets'])
        for key in ('mean','std'):
            np.testing.assert_array_equal(getattr(scaler,key),official['target_scaler'][key])
        truth_audit = {}
        for split,d in data.items():
            z = (d['annual_maxima']-d['location_scale'][:,:1])/d['location_scale'][:,1:2]
            g = support_margin(torch.as_tensor(np.array(d['targets'])), torch.as_tensor(z),
                               torch.as_tensor(d['metadata']['centered_time']))
            truth_audit[split] = dict(minimum_margin=float(g.min()),
                sequences_below_decoder_margin=int((g.amin(dim=1)<=MARGIN).sum()))
        atomic_json(output/'truth_feasibility.json',truth_audit)
        state.update(status='training',current_arm='reparameterized')
        atomic_json(output/'status.json',state)
        print('Training reparameterized candidate; official baseline reused unchanged.',flush=True)
        candidate = train_candidate(output/'reparameterized',data,device,protocol)
        rows, summaries = [], []
        paths = {'original':OFFICIAL,'reparameterized':output/'reparameterized/best.pt'}
        for arm,p in paths.items():
            a,b = evaluate(p,output/arm,data,arm,device)
            rows += a; summaries += b
        atomic_csv(output/'pretest_coefficient_rmse.csv',pd.DataFrame(rows))
        atomic_csv(output/'pretest_support_summary.csv',pd.DataFrame(summaries))
        atomic_json(output/'pretest_lock.json',dict(locked_utc=now(),candidate=candidate,
            checkpoint_sha256={a:file_sha256(p) for a,p in paths.items()},
            no_additional_arms_or_tuning=True))
        state.update(status='test_evaluation')
        atomic_json(output/'status.json',state)
        test = {'test':load_split(data_root,'test')}
        for arm,p in paths.items():
            a,b = evaluate(p,output/arm,test,arm,device)
            rows += a; summaries += b
        rmse, support = pd.DataFrame(rows),pd.DataFrame(summaries)
        atomic_csv(output/'coefficient_rmse.csv',rmse)
        atomic_csv(output/'support_summary.csv',support)
        table = rmse[rmse.scenario.eq('all')].pivot(index=['split','coefficient'],columns='arm',values='RMSE').reset_index()
        table['RMSE_improvement_pct'] = 100*(1-table.reparameterized/table.original)
        atomic_csv(output/'comparison.csv',table)
        if file_sha256(OFFICIAL)!=OFFICIAL_SHA256 or any(file_sha256(p)!=h for p,h in sources.items()):
            raise RuntimeError('Protected checkpoint or experiment source changed.')
        state.update(status='completed',completed_utc=now(),candidate=candidate,
                     official_checkpoint_unchanged=True,original_sources_unchanged=True)
        print(table[table.split.eq('test')].to_string(index=False),flush=True)
        print(support[support.split.eq('test') & support.scenario.eq('all')].to_string(index=False),flush=True)
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
