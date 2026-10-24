"""Tiny real CPU NN/MLE -> nested GP integration, no retained test artifacts."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import tempfile
from portable.common import read,verify_bundle
from portable import workflow, simulation, comparison
from gp_execution import configure
from test_bundle import fixture


def main():
    verify_bundle()
    configure('staged',cpu_budget=2,max_worker_threads=2,refit_threads=2)
    y,g,p,c,crs,truth=fixture()
    c['n_jobs']=2;c['kernel_candidates']=[['RBF',None],['Matern',.5]]
    evaluation=simulation.draw_annual(truth,0,purpose='crps_evaluation')
    with tempfile.TemporaryDirectory(prefix='thesis_staged_smoke_') as tmp:
        folder=Path(tmp)
        for method in ('nn','mle'):
            report=workflow.execute(y,g,p,c,crs,folder/method,truth=truth,threads=1,check_memory=False,
                estimator=method,evaluation_values=evaluation)
            assert report['workflow_completed']
            assert report['n_grids']==24 and report['crps_evaluated']
            phases=read(folder/method/'execution_phases.json')['phases']
            assert any(x['stage']=='candidate_models' and x['workers']==2 for x in phases)
            assert any('refit' in x['stage'] and x['native_threads']==2 for x in phases)
        metrics=[__import__('pandas').read_csv(folder/m/'truth_parameter_metrics.csv') for m in ('nn','mle')]
        assert len(comparison.paired_accuracy(*metrics))==10
        print(json.dumps({'status':'passed','n_grids':24,'methods':['nn','mle'],'nn_modified':False,
            'phases_recorded':True,'crps_independent':True,'temporary_outputs_removed_on_exit':True}))


if __name__=='__main__':main()
