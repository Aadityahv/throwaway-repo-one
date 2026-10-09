"""CPU-only tests for the power-sensor update test: the analysis on synthetic traces with a known lag and update period, and energy_harness/run_power_sensor_test_cluster.sh in DRY_RUN mode
(accept and refuse paths, canned nvidia-smi rows, no GPU, no remote machine)."""
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
import analyze_power_sensor as A  # noqa: E402

CAL = REPO / 'tiresias' / 'framework' / 'predictor' / 'calibrate'
BASH = shutil.which('bash')
PLAN = '0.1 0.25 0.5 1 2 5 10 20 40 0.1 0.25 0.5 1 2 5 40'


def synth(on_list, idle_s=6.0, idle_w=60.0, plat_w=400.0, update_s=0.25, tau_s=0.4, counter_update_s=0.1, rate_hz=1000.0, seed=1, counter_every_sample=False):
    """True power is a square wave. The board-power reading is held for update_s and moves towards the true power with time constant tau_s at every update; the counter is the exact
    integral of the true power, updated every counter_update_s. Returns (samples array, edges dict). Known values: update period update_s, step lag about tau_s*ln2 (+ up to update_s)."""
    rng = np.random.default_rng(seed)
    t_on, t_off, t = [], [], idle_s
    for d in on_list:
        t_on.append(t); t_off.append(t + d); t += d + idle_s
    total = t
    ts = np.arange(0, total, 1.0 / rate_hz)

    def true_power(x):
        v = np.full_like(x, idle_w)
        for a, b in zip(t_on, t_off):
            v[(x >= a) & (x < b)] = plat_w
        return v
    uk = np.arange(0, total + update_s, update_s)
    tp = true_power(uk); alpha = 1 - np.exp(-update_s / tau_s)
    rd = np.empty_like(uk); rd[0] = idle_w
    for i in range(1, len(uk)):
        rd[i] = rd[i - 1] + alpha * (tp[i] - rd[i - 1]) + rng.normal(0, 0.4)
    reading = rd[np.minimum((ts / update_s).astype(int), len(uk) - 1)]
    fine = np.arange(0, total + 0.001, 0.001); cum = np.concatenate([[0], np.cumsum(true_power(fine[:-1]) * 0.001)])

    def counter(x):
        q = np.floor(np.asarray(x) / counter_update_s) * counter_update_s
        return np.interp(q, fine, cum) * 1e3 + 1e6
    base = 1_000_000_000_000_000   # arbitrary monotonic clock offset (ns)
    ns = (base + ts * 1e9).astype(np.float64)
    ctr = np.zeros_like(ts)
    if counter_every_sample:
        ctr = counter(ts)
    else:
        idx = np.unique(np.searchsorted(ts, np.arange(0, total, 1.0)))
        ctr[idx] = counter(ts[idx])
    S = np.column_stack([ns, ns + 2e6, np.round(reading * 1e3), np.round(ctr)])
    edges = []
    for d, a, b in zip(on_list, t_on, t_off):
        edges.append(dict(on_s=d, t_on_ns=int(base + a * 1e9), t_off_ns=int(base + b * 1e9), e_before_mj=int(counter(a - 1e-4)), e_on_mj=int(counter(a)), e_off_mj=int(counter(b)), e_after_mj=int(counter(b + idle_s))))
    return S, dict(device='synthetic', power_limit_mw=700000, idle_s=idle_s, kernel_trips=1, blocks=1, nvml_ok=True, edges=edges)


class SyntheticAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = [0.5, 1, 2, 5, 10, 20, 0.5, 2, 20]
        cls.S, cls.E = synth(cls.plan, rate_hz=500.0)
        cls.res = A.analyze_run(cls.S, cls.E, 3.0)

    def test_power_update_period_recovered(self):
        u = self.res['power_value_updates']
        self.assertAlmostEqual(u['median_interval_ms'], 250.0, delta=3.0)
        self.assertGreater(u['share_of_intervals_within_5ms_of_median'], 0.8)

    def test_step_lag_recovered_within_tolerance(self):
        # smoothing with time constant tau and updates every U: 50% after about tau*ln2, 90% after about tau*ln10, each plus up to one update period
        tau, U = 0.4, 0.25
        lag = self.res['step_lag_s']
        for key, expect in (('rise_to_50pct_s', tau * np.log(2)), ('rise_to_90pct_s', tau * np.log(10)), ('fall_to_50pct_s', tau * np.log(2))):
            self.assertGreater(lag[key]['n'], 0)
            self.assertLess(abs(lag[key]['median'] - expect), U + 0.1, (key, lag[key], expect))
        self.assertEqual(lag['rise_to_50pct_s']['n_level_not_reached'], 0)
        self.assertGreater(lag['rise_to_90pct_s']['median'], lag['rise_to_50pct_s']['median'])

    def test_window_error_shrinks_with_padding_and_length(self):
        by = {}
        for r in self.res['steps']:
            by.setdefault(r['on_s'], []).append(r)
        short, long_ = by[0.5][0], by[20][0]
        self.assertLess(short['padded_0s_vs_reference_pct'], -40)             # edge-to-edge window on a short load loses most of the energy to the lag
        self.assertLess(abs(short['padded_2s_vs_reference_pct']), 5)
        self.assertLess(abs(long_['padded_0s_vs_reference_pct']), 6)
        self.assertLess(abs(long_['padded_2s_vs_reference_pct']), 1.5)
        for pad_key in ('padded_0s_vs_reference_pct', 'padded_1s_vs_reference_pct', 'padded_2s_vs_reference_pct'):
            self.assertLess(abs(long_[pad_key]), abs(short['padded_0s_vs_reference_pct']))

    def test_counter_edge_window_and_counter_slope(self):
        for r in self.res['steps']:
            if r['on_s'] >= 2:
                self.assertLess(abs(r['counter_edge_window_vs_reference_pct']), 100 * 0.1 / r['on_s'] * 1.5 + 1.0, r)   # one 100 ms counter update of quantisation
        plat = self.res['counter_vs_power_on_long_plateaus']
        self.assertTrue(plat)
        for p in plat:
            self.assertAlmostEqual(p['counter_slope_w'], 400.0, delta=2.0); self.assertLess(abs(p['counter_over_power_mean_pct']), 2.0)

    def test_steady_window_agreement_improves_with_length(self):
        a = self.res['window_inside_steady_load_error_vs_long_mean_power_pct']
        self.assertIn('3', a); self.assertLess(a['3']['max'], a['0.1']['max'] + 1e-9); self.assertLess(a['3']['p90'], 1.0)

    def test_decision_block(self):
        d = A.decision([self.res], 3.0)
        L0, L2 = d['padding_0s']['min_load_s'], d['padding_2s']['min_load_s']
        self.assertTrue(L0 is None or L0 >= 10, L0)                    # a 5 s load edge to edge is still more than 3% off with this lag
        self.assertIsNotNone(L2); self.assertLessEqual(L2, 5)
        self.assertEqual(d['padding_2s']['total_window_s'], L2 + 4)
        self.assertEqual(A.min_load_within([(1, 10.0), (5, 2.0), (10, -1.0)], 3.0), 5)
        self.assertIsNone(A.min_load_within([(1, 10.0), (5, 2.0), (10, -4.0)], 3.0))
        self.assertIsNone(A.min_load_within([(1, None)], 3.0))
        worse = json.loads(json.dumps(self.res))
        for r in worse['steps']:
            r['padded_2s_vs_reference_pct'] = 50.0
        self.assertIsNone(A.decision([self.res, worse], 3.0)['padding_2s']['min_load_s'])          # a run that never gets there makes the decision None

    def test_plateau_fraction_of_limit(self):
        self.assertAlmostEqual(self.res['max_plateau_fraction_of_power_limit'], 400 / 700, delta=0.01)
        self.assertFalse(self.res['plateau_near_power_limit_warning'])

    def test_other_update_period_and_lag_are_distinguished(self):
        S, E = synth([5, 20], update_s=0.5, tau_s=0.8, seed=3)
        r = A.analyze_run(S, E, 3.0)
        self.assertAlmostEqual(r['power_value_updates']['median_interval_ms'], 500.0, delta=3.0)
        self.assertGreater(r['step_lag_s']['rise_to_50pct_s']['median'], self.res['step_lag_s']['rise_to_50pct_s']['median'])

    def test_level_not_reached_is_none_not_zero(self):
        self.assertIsNone(A._crossing(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]), 10.0, True))
        self.assertIsNone(A._crossing(np.array([0.0, 1.0, 2.0]), np.array([3.0, 2.0, 1.0]), 0.5, False))
        self.assertEqual(A._crossing(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]), 2.0, True), 1.0)
        self.assertIsNone(A._delta(None, 1.0))


class FilesAndCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.tmp, True)

    def write_run(self, d, S, E):
        d.mkdir(parents=True)
        with gzip.open(d / 'samples.csv.gz', 'wt') as f:
            f.write('monotonic_ns,power_call_end_ns,board_power_mw,energy_counter_mj\n')
            for row in S:
                f.write('%d,%d,%d,%d\n' % tuple(row))
        (d / 'edges.json').write_text(json.dumps(E))

    def test_end_to_end_with_counter_update_diagnostic(self):
        S, E = synth([0.5, 2, 5, 20], rate_hz=500.0); self.write_run(self.tmp / 'run1', S, E)
        S2, E2 = synth([2], rate_hz=500.0, counter_every_sample=True, counter_update_s=0.1); self.write_run(self.tmp / 'diag', S2, E2)
        out = self.tmp / 'res.json'
        self.assertEqual(A.main(['--raw', str(self.tmp / 'run1'), '--diag', str(self.tmp / 'diag'), '--out', str(out)]), 0)
        res = json.loads(out.read_text())
        self.assertTrue(res['energy_counter_update_period']['measured'])
        self.assertAlmostEqual(res['energy_counter_update_period']['median_interval_ms'], 100.0, delta=3.0)
        self.assertAlmostEqual(res['power_reading_update_period_ms'][0], 250.0, delta=3.0)
        for k in ('padding_0s', 'padding_1s', 'padding_2s', 'energy_counter_edge_window'):
            self.assertIn(k, res['decision'])
        self.assertEqual(A.main(['--raw', str(self.tmp / 'run1'), '--out', str(out)]), 2)           # never overwrites
        out2 = self.tmp / 'res2.json'; A.main(['--raw', str(self.tmp / 'run1'), '--out', str(out2)])
        self.assertFalse(json.loads(out2.read_text())['energy_counter_update_period']['measured'])  # no diagnostic: reported as not measured, never guessed

    def test_bad_header_refused(self):
        d = self.tmp / 'bad'; d.mkdir(); (d / 'samples.csv').write_text('a,b,c,d\n1,2,3,4\n'); (d / 'edges.json').write_text('{}')
        with self.assertRaises(ValueError):
            A.load_samples(d / 'samples.csv')


FQDN = 'node-9.cluster.example.org'
NEWU = 'GPU-cafe0001-62f3-809c-7bc4-68233240d03d'
SCRIPT = REPO / 'energy_harness' / 'run_power_sensor_test_cluster.sh'


@unittest.skipUnless(BASH and SCRIPT.is_file(), 'needs bash and energy_harness/run_power_sensor_test_cluster.sh')
class DryRunScript(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.tmp, True)
        self.tree = self.tmp / 'tree'; (self.tree / 'energy_harness').mkdir(parents=True)
        for f in ('cluster_resolve_toolchain.sh', 'cluster_cal_lib.sh'):
            shutil.copy(REPO / 'energy_harness' / f, self.tree / 'energy_harness' / f)
        shutil.copy(REPO / 'HARDWARE_GROUND_TRUTH.md', self.tree / 'HARDWARE_GROUND_TRUTH.md')
        shutil.copytree(CAL, self.tree / 'tiresias/framework/predictor/calibrate', ignore=shutil.ignore_patterns('runs', 'tests', '__pycache__', 'data'))
        shutil.copytree(HERE, self.tree / 'tiresias/framework/predictor/h100_power_sensor_test', ignore=shutil.ignore_patterns('tests', '__pycache__'))
        (self.tree / '.archived_commit').write_text('a' * 40)
        self.rel_allow = 'tiresias/framework/predictor/calibrate/approved_devices.json'
        self.real_hash = hashlib.sha256((CAL / 'approved_devices.json').read_bytes()).hexdigest()
        self.n = 0

    def base_env(self, root, uuid, name, part='partition_h100', host=FQDN, **kw):
        env = dict(os.environ, DRY_RUN='1', ARCH_LABEL='h100', ARCH_SM='sm_90', HIPC_COMMIT='a' * 40, BOOKING_REF='booking log test booking entry', SOURCE_TREE=str(self.tree), OUT_ROOT=str(root),
                   HOME=str(self.tmp), CAL_PYTHON=sys.executable, DRY_HOSTNAME=host, DRY_SMI_CSV='0, %s, 00000000:4A:00.0, %s, 1650000001' % (uuid, name), SLURM_JOB_ID='888', SLURM_JOB_PARTITION=part)
        for k in ('APPROVE_ALLOCATED', 'TARGET_UUID', 'CUDA_VISIBLE_DEVICES', 'RUNS', 'COUNTER_PERIOD_DIAG'):
            env.pop(k, None)
        env.update(kw)
        return env

    def run_job(self, uuid=NEWU, name='NVIDIA H100 80GB HBM3', arch='h100', sm='sm_90', opt=False, host=FQDN, part='partition_h100', extra=None):
        self.n += 1; root = self.tmp / ('o%d' % self.n)
        env = self.base_env(root, uuid, name, part, host, ARCH_LABEL=arch, ARCH_SM=sm)
        if opt:
            env['APPROVE_ALLOCATED'] = '1'
        env.update(extra or {})
        r = subprocess.run([BASH, str(SCRIPT)], env=env, capture_output=True, text=True)
        outs = list(root.glob('*_*'))
        return r, (outs[0] if outs else None)

    def staged(self, out):
        return [e['uuid'] for e in json.loads((out / 'src' / self.rel_allow).read_text())['devices']]

    def approved_h100(self):
        return [e['uuid'] for e in json.loads((CAL / 'approved_devices.json').read_text())['devices'] if e.get('ground_truth_section') == 'H100' and not e.get('pending')][0]

    def test_approved_h100_dry_run_prints_compile_run_and_analysis_and_writes_status_files(self):
        u = self.approved_h100()
        r, out = self.run_job(uuid=u)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for s in ('micro_stepload.cu', '-arch=sm_90', '-ldl', 'CUDA_VISIBLE_DEVICES=%s' % u, 'CAL_EXPECT_UUID=%s' % u, 'COUNTER_POLL_MS=1000', 'analyze_power_sensor.py'):
            self.assertIn(s, r.stdout)
        self.assertNotIn('COUNTER_POLL_MS=0', r.stdout)                         # the counter is never polled at high rate in the main run
        for f in ('STATUS', 'summary.txt', 'COMPLETE', 'env.txt', 'gpu_line.txt', 'allowlist_check.txt', 'nvidia_smi_before.csv', 'nvidia_smi_after.csv'):
            self.assertTrue((out / f).exists(), f)
        self.assertIn('dry run', (out / 'STATUS').read_text())
        self.assertIn(PLAN, (out / 'env.txt').read_text()); self.assertIn(PLAN, r.stdout.replace('\\ ', ' '))
        self.assertFalse((out / 'approved_allocated.txt').exists())

    def test_unapproved_uuid_refused_exit_4_with_status(self):
        r, out = self.run_job()
        self.assertEqual(r.returncode, 4, r.stdout + r.stderr); self.assertNotIn(NEWU, self.staged(out))
        self.assertNotIn('CAL_EXPECT_UUID', r.stdout)                                            # no GPU command was printed
        self.assertIn('exit_code=4', (out / 'STATUS').read_text()); self.assertTrue((out / 'COMPLETE').exists())

    def test_opt_in_adds_one_staged_entry_only(self):
        before = [e['uuid'] for e in json.loads((self.tree / self.rel_allow).read_text())['devices']]
        r, out = self.run_job(opt=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(sorted(self.staged(out)), sorted(before + [NEWU]))
        self.assertEqual([e['uuid'] for e in json.loads((self.tree / self.rel_allow).read_text())['devices']], before)
        self.assertEqual(hashlib.sha256((CAL / 'approved_devices.json').read_bytes()).hexdigest(), self.real_hash)
        self.assertIn('APPROVE_ALLOCATED', (out / 'STATUS').read_text()); self.assertTrue((out / 'approved_allocated.txt').exists())

    def test_opt_in_conditions_refused_add_nothing(self):
        for label, kw in dict(wrong_partition=dict(part='partition_a100'), not_cluster=dict(host='ant.example.org')).items():
            with self.subTest(label):
                r, out = self.run_job(opt=True, **kw)
                self.assertEqual(r.returncode, 4, r.stdout + r.stderr); self.assertNotIn(NEWU, self.staged(out)); self.assertIn('REFUSED (APPROVE_ALLOCATED)', r.stderr)

    def test_target_uuid_pin_and_wrong_model_and_pair(self):
        r, out = self.run_job(opt=True, extra=dict(TARGET_UUID='GPU-deadbeef-94b3-99c5-6b61-dc31fd15b231'))
        self.assertEqual(r.returncode, 6); self.assertNotIn(NEWU, self.staged(out))
        r, out = self.run_job(name='NVIDIA A100-SXM4-80GB')
        self.assertEqual(r.returncode, 2)                                                        # h100 job on an A100
        r, out = self.run_job(sm='sm_80')
        self.assertEqual(r.returncode, 2); self.assertIn('unsupported architecture pair', r.stderr)

    def test_a100_dry_run_uses_sm_80(self):
        r, out = self.run_job(uuid=NEWU, name='NVIDIA A100-SXM4-80GB', arch='a100', sm='sm_80', part='partition_a100', opt=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr); self.assertIn('-arch=sm_80', r.stdout)

    def test_runs_and_diag_options(self):
        u = self.approved_h100()
        r, out = self.run_job(uuid=u, extra=dict(RUNS='3', COUNTER_PERIOD_DIAG='1'))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for d in ('/run1', '/run2', '/run3'):
            self.assertIn(d, r.stdout)
        self.assertIn('COUNTER_POLL_MS=0', r.stdout); self.assertIn('counter_diag', r.stdout)
        for bad in ('0', '4', 'x'):
            r, out = self.run_job(uuid=u, extra=dict(RUNS=bad)); self.assertEqual(r.returncode, 2, bad)
        r, out = self.run_job(uuid=u, extra=dict(COUNTER_PERIOD_DIAG='2'))
        self.assertEqual(r.returncode, 2)

    def test_never_overwrites_and_requires_booking_and_stamped_tree(self):
        u = self.approved_h100()
        r, out = self.run_job(uuid=u)
        self.assertEqual(r.returncode, 0)
        env = self.base_env(out.parent, u, 'NVIDIA H100 80GB HBM3', SLURM_JOB_ID=out.name.split('_')[1])
        r2 = subprocess.run([BASH, str(SCRIPT)], env=env, capture_output=True, text=True)
        self.assertEqual(r2.returncode, 2); self.assertIn('refusing to overwrite', r2.stderr)
        r3, _ = self.run_job(uuid=u, extra=dict(BOOKING_REF='short'))
        self.assertEqual(r3.returncode, 2)
        r4, _ = self.run_job(uuid=u, extra=dict(HIPC_COMMIT='b' * 40))
        self.assertEqual(r4.returncode, 2)

    def test_script_declares_the_required_sbatch_and_safety_properties(self):
        s = SCRIPT.read_text()
        for needle in ('#SBATCH --time=1-00:00:00', '#SBATCH --gres=gpu:1', '#SBATCH -c 2', '#SBATCH --mem=16G'):
            self.assertIn(needle, s)
        code = '\n'.join(l for l in s.splitlines() if not l.lstrip().startswith('#'))
        for banned in ('sudo', ' -lgc', ' -rgc', 'nvidia-smi -pm', 'nvidia-smi -pl', ' ncu ', 'nsys', 'kill '):
            self.assertNotIn(banned, code)


if __name__ == '__main__':
    unittest.main()
