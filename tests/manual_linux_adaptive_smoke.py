"""CPU-only, 24-GRID integration QA against a staged portable bundle; temp outputs."""
import argparse
from pathlib import Path
import json
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--ram16', action='store_true', help='Exercise the 16GB execution policy on the same temporary 24-GRID fixture')
    args = parser.parse_args()
    root = args.bundle.resolve()
    sys.path[:0] = [str(root), str(root/'src')]
    import numpy as np
    import pandas as pd
    import torch
    from portable import comparison, workflow, simulation
    from portable.common import EAS_CRS, read, verify_bundle
    from spatial_coordinates import project_lonlat_to_km
    import gp_execution as resources

    assert sys.platform == 'linux' and torch.version.cuda is None
    verify_bundle()
    if args.ram16:
        resources.configure('adaptive', cpu_budget=8, max_worker_threads=4, refit_threads=4,
                            memory_budget_gib=5., memory_reserve_gib=1.5)
    else:
        resources.configure('adaptive', cpu_budget=2, max_worker_threads=1, refit_threads=1)
    threads = 4 if args.ram16 else 1
    mle_jobs = 4 if args.ram16 else 2
    torch.set_num_threads(threads)
    n = 24
    lon=120.+np.linspace(0,2,n); lat=23.+.2*np.sin(np.arange(n))
    grids=pd.DataFrame(dict(grid_id=[f'X{i}' for i in range(n)],longitude_180=lon,
                           latitude=lat,lsm=np.ones(n)))
    grids[['x_km','y_km']]=project_lonlat_to_km(lon,lat,target_crs=EAS_CRS)
    predictors=pd.DataFrame(dict(grid_id=grids.grid_id,elevation_m=np.linspace(0,1000,n),
                                candidate_data_complete=True))
    truth=np.column_stack([25.+np.sin(np.arange(n)/5),np.full(n,.12),np.full(n,np.log(1.2)),
                           np.full(n,.02),np.full(n,-.12)])
    config=workflow.settings({'elevation':['elevation_m']},n,outer=2,inner=2,buffer=0,min_train=4)
    config.update(kernel_candidates=[['RBF',None],['Matern',1.5]],max_steps=0,n_jobs=2)
    inputs=(simulation.draw_annual(truth,0),grids,predictors,config,EAS_CRS)
    order=[]; original=workflow.execute
    def track(*positional, **keywords):
        order.append(keywords['estimator'])
        return original(*positional,**keywords)
    workflow.execute=track
    with tempfile.TemporaryDirectory(prefix='linux_adaptive_qa_') as temp:
        output=Path(temp)
        report=comparison.run_replicates(inputs,truth,output,1,calibration_id='linux_qa',
            threads=threads,mle_options={'jobs':mle_jobs},method_order='mle-first')
        assert report['completed_pairs']==1 and report['all_grids_retained']
        assert order==['mle','nn']
        for method in order:
            folder=output/f'replicates/replicate_001/{method}/attempt_001'
            saved=read(folder/'run_report.json')
            assert saved['method_order']==['mle','nn']
            assert saved['gp_execution_policy']['mode']=='adaptive'
            phases=read(folder/'execution_phases.json')['phases']
            assert any('schedulable_bytes' in phase for phase in phases)
            assert any(phase['stage']=='candidate_models' and phase['workers']==2 for phase in phases)
        scores=pd.read_csv(output/'crps_comparison.csv')
        assert len(scores)==2 and scores.n_total.eq(n*50).all()
        comparison.run_replicates(inputs,truth,output,1,calibration_id='linux_qa',
            threads=threads,mle_options={'jobs':mle_jobs},method_order='mle-first')
        assert order==['mle','nn']
        print(json.dumps(dict(status='passed',n_grids=n,order=order,platform=sys.platform,
            torch=torch.__version__,memory=resources.memory_snapshot(),execution_policy=resources.describe(),
            full_EAS_started=False,temporary_outputs='automatically removed'),indent=2))


if __name__=='__main__':
    main()
