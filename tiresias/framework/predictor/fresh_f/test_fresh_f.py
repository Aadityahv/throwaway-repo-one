"""CPU tests of the unseen machine-learning kernel set (fresh sets F and G): cell definitions, driver argument construction, graph-batch rule, windows parsing, and the L2 band rule."""
import json, os, sys, tempfile
from pathlib import Path
import unittest
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'fresh_g'))
import cells_f, run_f  # noqa: E402
import importlib.util
spec = importlib.util.spec_from_file_location('cells_g', HERE.parent / 'fresh_g' / 'cells_g.py'); cells_g = importlib.util.module_from_spec(spec); spec.loader.exec_module(cells_g)
HW = dict(l2_bytes=134217728)


class Sets(unittest.TestCase):
    def test_cells_cover_the_grid_and_avoid_the_marginal_band(self):
        f = cells_f.define_cells(HW); g = cells_g.define_cells(HW)
        self.assertEqual(len(f), 40); self.assertEqual(len(g), 8)
        for c in f + g:
            self.assertFalse(0.4 <= c['l2_ratio'] <= 1.5, c['cell_id']); self.assertEqual(c['tier'], 'L2' if c['l2_ratio'] < 1 else 'DRAM')
        self.assertEqual(len({c['cell_id'] for c in f + g}), 48)
        for fam in ('gelu', 'swiglu', 'rmsnorm', 'rope', 'sgemm'):
            cs = [c for c in f if c['family'] == fam]; self.assertEqual(sorted({c['regime'] for c in cs}), sorted(cells_f.REGIMES)); self.assertEqual(sorted({c['candidate_id'] for c in cs}), ['c1', 'c2'])

    def test_candidates_do_equal_work(self):
        by = {}
        for c in cells_f.define_cells(HW) + cells_g.define_cells(HW): by.setdefault((c['family'], c['regime']), []).append(c)
        for k, cs in by.items():
            self.assertEqual(len(cs), 2); self.assertEqual(cs[0]['footprint_bytes'], cs[1]['footprint_bytes'], k); self.assertEqual(cs[0]['logical_bytes_per_launch'], cs[1]['logical_bytes_per_launch'], k)

    def test_launch_geometry_tiles_the_problem_exactly(self):
        for c in cells_f.define_cells(HW):
            k, ctl = c['kernels'][0], c['controls']; g = k['grid'][0] * k['grid'][1]; thr = k['block'][0]
            if c['family'] in ('gelu', 'swiglu'): self.assertEqual(g * thr * (1 if k['kid'].endswith('_s') else 4), ctl['n'])
            if c['family'] == 'rmsnorm': self.assertEqual(g, ctl['rows']); self.assertEqual(ctl['cols'] % (thr * (4 if k['kid'] == 'rms_v4' else 1)), 0)
            if c['family'] == 'rope': self.assertEqual(k['grid'][0], ctl['seq']); self.assertEqual(k['grid'][1], ctl['batch'] if k['kid'] == 'rope_all' else ctl['batch'] * ctl['heads'])
            if c['family'] == 'sgemm': t = 64 if k['kid'] == 'sg64' else 128; self.assertEqual(g * t * t, ctl['M'] * ctl['N'])
        for c in cells_g.define_cells(HW):
            k, ctl = c['kernels'][0], c['controls']; t = 128 if k['kid'] == 'tc128' else 64; self.assertEqual(k['grid'][0] * k['grid'][1] * t * t, ctl['M'] * ctl['N']); self.assertEqual(ctl['K'] % 32, 0)

    def test_driver_arguments_and_batch_rule(self):
        os.environ['FRESH_SET'] = 'f'
        c = {'family': 'rmsnorm', 'candidate_id': 'c2', 'controls': dict(rows=1024, cols=1024)}
        self.assertEqual(run_f.argv_of(c), ['rmsnorm', '1', '1024', '1024'])
        self.assertEqual(run_f.argv_of({'family': 'tcgemm', 'candidate_id': 'c1', 'controls': dict(M=512, N=512, K=512)}), ['tcgemm', '0', '512', '512', '512'])
        self.assertEqual(run_f.choose_graph_batch(1e-6), 2048 if 5e-3 / 1e-6 > 2048 else 5000); self.assertEqual(run_f.choose_graph_batch(1.0), 1); self.assertEqual(run_f.choose_graph_batch(1e-3), 5)

    def test_windows_parser_and_check_reader(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'windows.csv'; p.write_text('block,launches,host_begin_monotonic_ns,host_end_monotonic_ns,cuda_seconds\n1,100,0,1,0.5\n')
            w = run_f.parse_windows(p); self.assertEqual(w['launches'], 100); self.assertAlmostEqual(w['per_launch_s'], 0.005)
            p.write_text('block,launches,host_begin_monotonic_ns,host_end_monotonic_ns,cuda_seconds\n1,100,0,1,0\n')
            with self.assertRaises(ValueError): run_f.parse_windows(p)
            o = Path(d) / 'out.bin'; self.assertEqual(run_f.read_check(o), (False, 'CHECK_MISSING')); Path(str(o) + '.check').write_text('CHECK_OK x\n'); self.assertTrue(run_f.read_check(o)[0])
            Path(str(o) + '.check').write_text('CHECK_FAIL x\n'); self.assertFalse(run_f.read_check(o)[0])


if __name__ == '__main__':
    unittest.main()
