"""Small resource/configuration checks; never launch the full EAS workflow."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from portable.common import ROOT
from portable.workflow import memory_guard, memory_plan


def worker_threads():
    import numpy as np
    from threadpoolctl import threadpool_info
    np.ones((32, 32)) @ np.ones((32, 32))
    return [entry['num_threads'] for entry in threadpool_info()]


class ParallelSettingsTests(unittest.TestCase):
    def test_memory_guard_accounts_for_all_requested_workers(self):
        with patch('portable.workflow.psutil.virtual_memory', return_value=SimpleNamespace(available=10_000_000_000)), \
             patch('portable.workflow.effective_n_jobs', return_value=19):
            plan = memory_plan(7822, -2)
            self.assertEqual(plan['effective_workers'], 19)
            self.assertEqual(plan['required_bytes'], 19 * 6_894_694_720)
            self.assertFalse(plan['passes'])
            with self.assertRaisesRegex(MemoryError, 'effective workers=19'):
                memory_guard(7822, -2, enforce=True)

    def test_low_memory_default_only_warns_without_changing_workers(self):
        with patch('portable.workflow.psutil.virtual_memory', return_value=SimpleNamespace(available=1)), \
             patch('portable.workflow.effective_n_jobs', return_value=19), \
             patch('builtins.print') as printed:
            self.assertEqual(memory_guard(7822, -2), 19 * 6_894_694_720)
            self.assertIn('MEMORY ADVISORY ONLY', printed.call_args.args[0])
            self.assertIn('Continuing as requested', printed.call_args.args[0])

    def test_real_allocation_error_still_propagates_and_is_recorded(self):
        import tempfile
        from test_bundle import fixture
        from portable.common import read
        from portable.workflow import execute
        y, grids, predictors, config, crs, _ = fixture(8)
        with tempfile.TemporaryDirectory() as temp, \
             patch('portable.workflow.psutil.virtual_memory', return_value=SimpleNamespace(available=1)), \
             patch('portable.workflow.infer', side_effect=MemoryError('actual allocation failed')) as infer:
            output = Path(temp) / 'failed_run'
            with self.assertRaisesRegex(MemoryError, 'actual allocation failed'):
                execute(y, grids, predictors, config, crs, output, threads=1)
            infer.assert_called_once()
            report = read(output / 'run_status.json')
            self.assertEqual(report['status'], 'failed')
            self.assertFalse(report['workflow_completed'])
            self.assertIn('actual allocation failed', report['error'])

    def test_single_worker_original_screen_is_preserved(self):
        with patch('portable.workflow.psutil.virtual_memory', return_value=SimpleNamespace(available=10_000_000_000)), \
             patch('portable.workflow.effective_n_jobs', return_value=1):
            self.assertEqual(memory_guard(7822, 1), 6_894_694_720)
            self.assertTrue(memory_plan(7822, 1)['passes'])

    def test_zero_workers_rejected(self):
        with self.assertRaises(ValueError):
            memory_plan(7822, 0)

    def test_cli_defaults_and_calibration_worker_propagation(self):
        tree = ast.parse((ROOT / 'run.py').read_text(encoding='utf-8'))
        defaults = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
                if node.args and isinstance(node.args[0], ast.Constant):
                    for keyword in node.keywords:
                        if keyword.arg == 'default':
                            defaults[node.args[0].value] = ast.literal_eval(keyword.value)
        self.assertEqual(defaults['--threads'], 1)
        self.assertEqual(defaults['--jobs'], -2)
        ps = (ROOT / 'run.ps1').read_text(encoding='utf-8')
        self.assertIn('[int]$Jobs = -2', ps)
        self.assertIn('[int]$Threads = 1', ps)
        self.assertIn("'--jobs'", ps)
        tree = ast.parse((ROOT / 'portable/simulation.py').read_text(encoding='utf-8'))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == 'spatial_forward_selection']
        self.assertEqual(len(calls), 1)
        jobs = next(k.value for k in calls[0].keywords if k.arg == 'n_jobs')
        self.assertEqual(ast.unparse(jobs), "config['n_jobs']")

    def test_real_loky_workers_inherit_one_native_thread(self):
        from spatial_predictor_selection import _parallel_executor
        from joblib import delayed
        from joblib.externals.loky import get_reusable_executor
        env = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
        try:
            with patch.dict(os.environ, env):
                results = _parallel_executor(2)(delayed(worker_threads)() for _ in range(2))
            self.assertTrue(all(result and all(n == 1 for n in result) for result in results), results)
        finally:
            get_reusable_executor().shutdown(wait=True, kill_workers=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
