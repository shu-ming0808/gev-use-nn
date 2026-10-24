"""Replay one verified paired simulation with NN on another CPU platform.

Use the exact saved fit/evaluation observations and truth. Never recalibrate or
draw another sample just to compare Windows and WSL execution times.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from .common import read, write, sha, identity, latest_completed, attempt_folder
from . import comparison, inference, workflow
from research_workflows import YEARS, validate_annual_sequences
from train_cheng_nn import atomic_csv
import gp_execution


def paired_source(folder, inputs):
    folder = Path(folder)
    report = read(folder/'run_report.json')
    if report.get('completed_pairs') != 1 or report.get('total') != 1:
        raise ValueError('This three-method benchmark requires exactly one completed pair')
    for name, checksum in report['output_sha256'].items():
        if Path(name).name != name or sha(folder/name) != checksum:
            raise ValueError('Changed paired comparison output')
    protocol = read(folder/'comparison_protocol.json')
    if identity(protocol) != report['protocol_id']:
        raise ValueError('Changed comparison protocol')
    _, grids, predictors, config, crs = inputs
    if (protocol['config'] != config or protocol['coordinate_crs'] != crs
        or protocol['grids_sha256'] != identity(grids.to_json())
        or protocol['predictors_sha256'] != identity(predictors.to_json())
        or protocol['years'] != list(YEARS)
        or protocol['bound_model_sha256'] != inference.DIGEST):
        raise ValueError('Replay data/model/spatial settings differ from the Windows pair')
    # Copying the bundle preserves source bytes, including Windows newlines.
    if protocol['source_sha256'] != comparison.code_identity():
        raise ValueError('Replay source code differs; do not mix code versions in timings')
    branches = {}
    for method in ('nn', 'mle'):
        item = latest_completed(folder/'replicates/replicate_001'/method)
        if item is None or item[1].get('estimator') != method:
            raise ValueError('Missing completed paired estimator')
        branches[method] = item
    nn, saved = branches['nn']
    required = ('annual_maxima.npy','truth.npy','crps_evaluation_maxima.npy','nn_coefficients.csv','spatial_oof.csv')
    if any(name not in saved['output_sha256'] for name in required):
        raise ValueError('Replay inputs are not covered by the completed-output hashes')
    if (saved.get('model',{}).get('sha256') != inference.DIGEST
        or saved.get('data_role') != 'calibrated_simulation' or saved.get('config') != config
        or saved.get('sample_id') != branches['mle'][1].get('sample_id')):
        raise ValueError('Invalid NN simulation source')
    ids = pd.read_csv(nn/'nn_coefficients.csv', dtype={'grid_id':str}).grid_id.tolist()
    if ids != grids.grid_id.tolist(): raise ValueError('Replay GRID order differs')
    values = validate_annual_sequences(np.asarray(YEARS), np.load(nn/'annual_maxima.npy', allow_pickle=False))
    evaluation = validate_annual_sequences(np.asarray(YEARS), np.load(nn/'crps_evaluation_maxima.npy', allow_pickle=False))
    truth = np.load(nn/'truth.npy', allow_pickle=False)
    if (values.shape != (len(grids),50) or evaluation.shape != values.shape
        or truth.shape != (len(grids),5) or not np.isfinite(truth).all()
        or np.array_equal(values,evaluation)):
        raise ValueError('Invalid simulation arrays or non-independent CRPS sample')
    for name, array in (('annual_maxima.npy',values),('truth.npy',truth),('crps_evaluation_maxima.npy',evaluation)):
        if not np.array_equal(np.load(branches['mle'][0]/name,allow_pickle=False),array):
            raise ValueError('The paired methods did not use identical arrays')
    return branches, values, truth, evaluation


def replay(folder, inputs, output, *, threads=1):
    branches, values, truth, evaluation = paired_source(folder, inputs)
    _, grids, predictors, config, crs = inputs
    # Paired summary CSVs may be regenerated on resume (generation timings can
    # change). Bind to the immutable experiment identity and estimator report,
    # not to that mutable aggregate report.
    protocol = dict(version=1,source_pair_protocol_id=read(Path(folder)/'run_report.json')['protocol_id'],
        source_nn_report_sha256=sha(branches['nn'][0]/'run_report.json'),
        arrays={name:comparison.array_hash(array) for name,array in
            (('fit',values),('truth',truth),('evaluation',evaluation))},
        config=config,coordinate_crs=crs,threads=threads,execution_policy=gp_execution.describe(),
        source_sha256=comparison.code_identity(),replay_code_sha256=sha(Path(__file__)))
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    path = output/'replay_protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Replay protocol changed; use a new experiment name')
    write(path,protocol); pid = identity(protocol)
    prior = latest_completed(output,pid)
    if prior and prior[1].get('platform_replay_verified'):
        print('Verified replay already completed:',prior[0],flush=True)
        return prior
    target = attempt_folder(output)
    report = workflow.execute(values,grids,predictors,config,crs,target,truth=truth,
        protocol_id=pid,threads=threads,estimator='nn',evaluation_values=evaluation)
    keys = ['grid_id','coefficient','outer_fold']
    def folds(directory):
        return pd.read_csv(directory/'spatial_oof.csv',dtype={'grid_id':str})[keys].sort_values(keys).reset_index(drop=True)
    if not folds(target).equals(folds(branches['nn'][0])):
        write(target/'platform_validation.json',dict(status='failed',reason='Outer fold membership differs'))
        raise ValueError('Replay outer folds differ; do not compare timing as equivalent tasks')
    report.update(platform_replay_verified=True,source_pair=str(Path(folder)),
        source_sample_id=branches['nn'][1]['sample_id'],
        run_environment=comparison.environment(threads,config['n_jobs'],1),
        same_fit_truth_evaluation_arrays=True,outer_fold_membership_matched=True)
    write(target/'run_report.json',report)
    return target,report


def summarize(windows, linux, inputs, output):
    branches, values, truth, evaluation = paired_source(windows,inputs)
    replayed = latest_completed(linux)
    if not replayed or not replayed[1].get('platform_replay_verified'):
        raise ValueError('No verified platform replay')
    rpath = Path(linux)/'replay_protocol.json'
    rp = read(rpath)
    if (identity(rp) != replayed[1]['protocol_id'] or
        rp['source_pair_protocol_id'] != read(Path(windows)/'run_report.json')['protocol_id'] or
        rp['source_nn_report_sha256'] != sha(branches['nn'][0]/'run_report.json')):
        raise ValueError('WSL replay belongs to a different source pair')
    arms = {'Windows_NN':branches['nn'],'Windows_MLE':branches['mle'],'WSL_NN':replayed}
    timings=[]; metrics=[]
    for label,(folder,report) in arms.items():
        for name,array in (('annual_maxima.npy',values),('truth.npy',truth),('crps_evaluation_maxima.npy',evaluation)):
            if not np.array_equal(np.load(folder/name,allow_pickle=False),array):
                raise ValueError('Three-method inputs differ')
        estimator = report['estimator']
        row = dict(method=label,n_grids=len(inputs[1]),
            estimator_seconds=report['timings']['estimator_seconds'],
            spatial_cv_seconds=report['timings']['join_preflight_nested_cv_seconds'],
            total_seconds=report['timings']['total_from_estimator_start_until_outputs_seconds'],
            estimator_support_failures=report['support_failure_grids'][estimator],
            gp_support_failures=report['support_failure_grids']['gp_oof'],
            cpu_budget=report['gp_execution_policy']['cpu_budget'])
        timings.append(row)
        table=pd.read_csv(folder/'truth_parameter_metrics.csv')
        table['stage']=table.stage.replace({estimator:'estimation'})
        table.insert(0,'method',label); metrics.append(table)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    atomic_csv(output/'timing_and_support.csv',pd.DataFrame(timings))
    atomic_csv(output/'parameter_rmse.csv',pd.concat(metrics,ignore_index=True))
    write(output/'run_report.json',dict(status='completed',methods=list(arms),replicates=1,
        same_fit_truth_evaluation_arrays=True,
        same_execution_policy=len({identity(r['gp_execution_policy']) for _,r in arms.values()})==1,
        environments={name:r['run_environment'] for name,(_,r) in arms.items()},
        sources={name:str(p) for name,(p,_) in arms.items()},
        timing_scope='Fresh estimator through final workflow outputs; excludes installation, calibration, generation and platform staging.',
        caveat='One sequential run per method, not a statistically established platform speedup; no significance test.',
        output_sha256={name:sha(output/name) for name in ('timing_and_support.csv','parameter_rmse.csv')}))
    print('Three-method summary:',output,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--windows',type=Path,required=True);p.add_argument('--linux',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--jobs',type=int,default=-2)
    args=p.parse_args()
    inputs=workflow.eas_inputs();inputs[3]['n_jobs']=args.jobs
    summarize(args.windows,args.linux,inputs,args.output)
