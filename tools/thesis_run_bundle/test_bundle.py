"""Offline, tiny-fixture integration tests; never start a full EAS/global run."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.stats import genextreme
from threadpoolctl import threadpool_limits
from portable.common import ROOT, read, sha, write, completed, latest_completed
from portable.workflow import settings, execute, errors, eas_inputs
from portable.simulation import draw_annual, calibrate, simulate
from portable.global_run import geometry, retain
from spatial_coordinates import project_lonlat_to_km
from research_workflows import COEFFICIENTS, YEARS
from elevation_gp_analysis import _buffered_training_indices
from train_cheng_nn import atomic_csv
from portable.common import EAS_CRS

def fixture(n=24):
    longitude=120.+np.linspace(0,2,n);latitude=23.+.2*np.sin(np.arange(n))
    grids=pd.DataFrame(dict(grid_id=[f'X{i}' for i in range(n)],longitude_180=longitude,
        latitude=latitude,lsm=np.ones(n)))
    grids[['x_km','y_km']]=project_lonlat_to_km(longitude,latitude,target_crs=EAS_CRS)
    predictors=pd.DataFrame(dict(grid_id=grids.grid_id,elevation_m=np.linspace(0,1000,n),candidate_data_complete=True))
    truth=np.column_stack([25.+np.sin(np.arange(n)/5),np.full(n,.12),np.full(n,np.log(1.2)),
        np.full(n,.02),np.full(n,-.12)])
    config=settings({'elevation':['elevation_m']},n,outer=2,inner=2,buffer=0,min_train=4)
    config['kernel_candidates']=[['RBF',None]];config['max_steps']=0
    return draw_annual(truth,0),grids,predictors,config,EAS_CRS,truth

class BundleTests(unittest.TestCase):
    def test_generation_reproducible_and_distinct_100(self):
        truth=np.array([[20.,.2,np.log(2.),.03,-.1],[15.,0.,0.,0.,0.],[30.,-.1,.4,-.02,.2]])
        first=draw_annual(truth,0)
        np.testing.assert_array_equal(first,draw_annual(truth,0))
        self.assertEqual(len({draw_annual(truth,i).tobytes() for i in range(100)}),100)
        t=(np.asarray(YEARS)-2000.5)/10
        mu=truth[:,0,None]+truth[:,1,None]*t;scale=np.exp(truth[:,2,None]+truth[:,3,None]*t)
        u=np.random.default_rng(np.random.SeedSequence(20260929,spawn_key=(0,))).random((3,50))
        np.testing.assert_allclose(first,genextreme.ppf(u,c=-truth[:,4,None],loc=mu,scale=scale),rtol=1e-13)

    def test_geodesic_buffer_screen_matches_original(self):
        _,g,_,_,_,_=fixture(40);a=np.arange(25);b=np.arange(25,40)
        for radius in (0,10,50,100,300):
            np.testing.assert_array_equal(retain(g,a,b,radius),
                _buffered_training_indices(g,a,b,float(radius),distance_mode='wgs84'))
        # Dateline/high-latitude screen must also match.
        g.longitude_180=np.linspace(170,190,40)%360;g.longitude_180=(g.longitude_180+180)%360-180
        g.latitude=75.+np.sin(np.arange(40))
        for radius in (10,100,500,1000):
            np.testing.assert_array_equal(retain(g,a,b,radius),
                _buffered_training_indices(g,a,b,float(radius),distance_mode='wgs84'))

    def test_metric_stage_reference(self):
        t=np.zeros((4,5));p=np.ones((4,5))*2
        m=errors(p,t,'gp_oof');self.assertTrue(m.RMSE.eq(2).all());self.assertTrue(m.Bias.eq(2).all())

    def test_hash_checked_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'attempt_001';d.mkdir();(d/'x.csv').write_text('1')
            write(d/'run_report.json',dict(workflow_completed=True,protocol_id='a',output_sha256={'x.csv':sha(d/'x.csv')}))
            self.assertEqual(latest_completed(Path(tmp),'a')[0],d)
            with self.assertRaises(ValueError): completed(d,'b')
            (d/'x.csv').write_text('2')
            with self.assertRaises(ValueError): completed(d,'a')

    def test_eas_data_contract(self):
        y,g,p,c,crs=eas_inputs()
        self.assertEqual(y.shape,(7822,50));self.assertEqual(c['max_train'],7822)
        self.assertEqual(set(g.grid_id),set(p.grid_id))

    def test_tiny_complete_nn_nested_gp_truth_and_calibration(self):
        y,g,p,c,crs,truth=fixture()
        with tempfile.TemporaryDirectory() as tmp, threadpool_limits(limits=1):
            root=Path(tmp)
            report=execute(y,g,p,c,crs,root/'real',threads=1,check_memory=False)
            self.assertTrue(report['workflow_completed'])
            calibrated,creport=calibrate(root/'real',g,p,c,crs,root/'calibration',threads=1)
            self.assertEqual(calibrated.shape,(24,5))
            again,_=calibrate(root/'real',g,p,c,crs,root/'calibration',threads=1)
            np.testing.assert_array_equal(again,calibrated)
            fake=draw_annual(calibrated,1)
            sim=execute(fake,g,p,c,crs,root/'simulation',truth=calibrated,threads=1,check_memory=False)
            self.assertEqual(sim['metric_reference'],'known_simulated_truth')
            metric=pd.read_csv(root/'simulation/truth_parameter_metrics.csv')
            self.assertEqual(len(metric),10)
            self.assertEqual(set(metric.stage),{'nn','gp_oof'})
            self.assertEqual(pd.read_csv(root/'simulation/spatial_oof.csv').shape[0],24*5)
            self.assertEqual(sim['support_failure_grids']['truth'],0)
            self.assertTrue(completed(root/'simulation'))

    def test_global_region_coverage(self):
        from portable.global_run import load_registry
        g,_,codes,_=load_registry('all','')
        self.assertEqual(len(codes),46);self.assertEqual(len(g),349565)
        self.assertEqual(int(g.ar6_region.eq('EAS').sum()),7822)

    def test_relocated_bundle_no_old_project_paths(self):
        import shutil
        import subprocess
        from portable.common import raw_root
        with tempfile.TemporaryDirectory(prefix='thesis_portable_') as tmp:
            target=Path(tmp)/'moved bundle';target.mkdir()
            manifest=read(ROOT/'manifest.json')
            for name in manifest['files']:
                p=target/name;p.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(ROOT/name,p)
            shutil.copy2(ROOT/'manifest.json',target/'manifest.json')
            proc=subprocess.run([sys.executable,'-B',str(target/'run.py'),'check',
                '--threads','1','--data-root',str(raw_root())],cwd=tmp,capture_output=True,text=True,encoding='utf-8',errors='replace')
            self.assertEqual(proc.returncode,0,proc.stderr)
            report=read(target/'outputs/environment_check.json')
            self.assertIsNone(report['torch_cuda_build']);self.assertEqual(report['status'],'passed')

    def test_simulate_one_then_100_resumes_99_only(self):
        y,g,p,c,crs,truth=fixture(8)
        seen=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'calibration').mkdir()
            np.save(root/'calibration/truth.npy',truth)
            def fake_execute(values,grids,predictors,config,crs,folder,**kwargs):
                seen.append(values.tobytes());folder.mkdir(parents=True)
                metric=pd.concat([errors(truth,truth,'nn'),errors(truth,truth,'gp_oof')])
                atomic_csv(folder/'truth_parameter_metrics.csv',metric)
                report=dict(workflow_completed=True,protocol_id=kwargs['protocol_id'],
                    timings={'total_from_nn_start_until_outputs_seconds':1.},support_failure_grids={'nn':0,'gp_oof':0},
                    output_sha256={'truth_parameter_metrics.csv':sha(folder/'truth_parameter_metrics.csv')})
                write(folder/'run_report.json',report);return report
            with patch('portable.simulation.calibrate',return_value=(truth,{'protocol_id':'cal'})), \
                 patch('portable.simulation.execute',side_effect=fake_execute):
                simulate(root/'real',(y,g,p,c,crs),root,1,threads=1)
                self.assertEqual(len(seen),1)
                simulate(root/'real',(y,g,p,c,crs),root,100,threads=1)
                self.assertEqual(len(seen),100);self.assertEqual(len(set(seen)),100)
                self.assertEqual(len(pd.read_csv(root/'parameter_metrics_all_replicates.csv')),1000)

if __name__=='__main__':
    unittest.main(verbosity=2)
