"""Same frozen BOUND computation, relocating source-hash checks, not weights."""
from pathlib import PureWindowsPath
import numpy as np
from .common import ROOT, sha
from global_pipeline import infer_global_cheng_nn as shared
from compare_cheng_nn_reparameterization import DECODER_ID, MARGIN, predict_sample_coefficients
from research_workflows import YEARS, REFERENCE_YEAR, TIME_SCALE_YEARS

CHECKPOINT = ROOT / 'models/bound.pt'
DIGEST = '03890b4877a40cf93822877d09231c003c920f34c68ed911b1da481ce396318f'

def infer(values, batch_size=2048):
    from gp_execution import nn_plan, measured_phase
    plan = nn_plan(batch_size)
    if plan is None:
        return _infer(values, batch_size)
    import torch
    from threadpoolctl import threadpool_limits
    previous = torch.get_num_threads()
    print(f"[resources] NN: batch {plan['batch_size']}, {plan['native_threads']} CPU threads", flush=True)
    try:
        torch.set_num_threads(plan['native_threads'])
        with measured_phase('nn_estimation', **plan), threadpool_limits(limits=plan['native_threads']):
            result = _infer(values, plan['batch_size'])
        result['model']['resource_plan'] = plan
        return result
    finally:
        torch.set_num_threads(previous)


def _infer(values, batch_size):
    timing = {}
    with shared.timed(timing, 'model_load_seconds'):
        ns, model, scaler, payload = shared.load_frozen_model(CHECKPOINT, DIGEST, 'cpu')
        if (payload.get('decoder') != DECODER_ID or payload.get('support_margin') != MARGIN
            or payload.get('requires_observed_sequence_at_inference') is not True
            or payload.get('latent_coefficient_names') != ['mu0','beta_mu','delta','beta_sigma','xi0']):
            raise ValueError('BOUND decoder contract mismatch')
        # Check EVERY frozen implementation hash, but never read old C: paths.
        for source, digest in payload['source_hashes'].items():
            name = PureWindowsPath(source).name
            local = ROOT / ('notebooks' if name.endswith('.ipynb') else 'src') / name
            if sha(local) != digest:
                raise ValueError(f'Frozen implementation changed: {name}')
    with shared.timed(timing, 'feature_transform_seconds'):
        x, scaling = shared.transform_sequences(values)
        z = (values - scaling[:, :1]) / scaling[:, 1:2]
    with shared.timed(timing, 'nn_and_decoder_seconds'):
        standardized = predict_sample_coefficients(model, x, scaler, z,
            (np.asarray(YEARS)-REFERENCE_YEAR)/TIME_SCALE_YEARS,
            'cpu', True, margin=MARGIN, batch_size=batch_size)
    with shared.timed(timing, 'inverse_scaling_seconds'):
        prediction = ns['inverse_sample_standardization'](standardized, scaling)
    with shared.timed(timing, 'diagnostic_seconds'):
        low, high = shared.training_feature_envelope(shared.TRAIN_DIRECTORY, payload['simulation_metadata'])
        checks = shared.diagnostic_table(prediction, values, x, low, high,
                                         payload['simulation_metadata']['parameter_ranges'])
    return dict(prediction=prediction, checks=checks, timings=timing,
                model=dict(kind='bound', sha256=DIGEST, retrained=False, support_margin=MARGIN))
