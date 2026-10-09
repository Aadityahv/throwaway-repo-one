"""Derivations of the runtime model's traffic rates (`store_legacy`): legacy reproduces every archived document exactly, the stream-rate and guarded derivations behave by their stated rules,
and the offline re-derivation tool never overwrites and records its provenance."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'tools'))
from cal import fits  # noqa: E402
import rederive_store as R  # noqa: E402

RUNS = HERE / 'runs'
BLACKWELL = ['energy_20261002/run_full', 'energy_repeat_20261002/run_full']
H100 = ['cluster_h100_20261002/h100_j13/run', 'cluster_h100_20261002/h100_j13/run_r2', 'cluster_h100_20261002/h100_j14/run']
A100 = ['cluster_a100_20261002/a100_j15/run', 'cluster_a100_20261002/a100_j15/run_r2']
ALL = BLACKWELL + H100 + A100


def doc_path(rel):
    d = RUNS / rel; return [x for x in sorted(d.glob('calibration_sm_*.json')) if 'application' not in x.name][0]


def raw(rel):
    d = RUNS / rel; rd = lambda n: R.parse_jsonl((d / 'stages' / n / 'stdout.jsonl').read_bytes().decode())
    doc = json.loads(doc_path(rel).read_text()); return doc, rd('store'), rd('stream')


def derive(rel, method, **kw):
    doc, store, stream = raw(rel); c = doc['constants']
    return fits.fit_store_detailed(store, doc['device']['sm_count'], c['chase_latency_ns'], c['stream_curves'], method, doc['device']['l2_bytes'], **kw)


def original_fit_store(rows, sm, latency_ns):
    """The derivation exactly as it was before the `method` argument existed (frozen copy for the regression test)."""
    import numpy as np
    pick = lambda **kw: next(x for x in rows if all(x.get(k) == v for k, v in kw.items()))
    t0 = float(pick(fixture='tiny')['us'])
    tri = pick(fixture='triad', tier='below'); s1 = pick(fixture='store', tier='below', stride=1); s7 = pick(fixture='store', tier='below', stride=7); s33 = pick(fixture='store', tier='below', stride=33)
    A, y = [], []
    tri_c = dict(rs=tri['bytes_read'] / 32, ws=tri['bytes_written'] / 32, lines=(tri['bytes_read'] + tri['bytes_written']) / 128)
    A.append([tri_c['rs'] * 32, tri_c['ws'] * 32, tri_c['lines']]); y.append(tri['us'] - t0)
    for st, row in ((1, s1), (7, s7), (33, s33)):
        sec, lin = fits.store_pattern(st); req = row['bytes_written'] / 128
        A.append([0.0, req * sec * 32, req * lin]); y.append(row['us'] - t0)
    coef, cond = fits.nnls3(np.array(A), np.array(y))
    inv_r, inv_w, c_line = coef
    sa = pick(fixture='store', tier='above', stride=1); ta = pick(fixture='triad', tier='above')
    inv_dw = (sa['us'] - t0) / sa['bytes_written']
    inv_dr = (ta['us'] - t0 - ta['bytes_written'] * inv_dw) / ta['bytes_read']
    return dict(t0_us=t0, L2_read_sector_TBps=1 / inv_r / 1e6, L2_write_sector_TBps=1 / inv_w / 1e6, c_line_ns=c_line * 1e3,
                DRAM_read_TBps=1 / inv_dr / 1e6, DRAM_write_TBps=1 / inv_dw / 1e6, latency_ns=latency_ns, active_sms_microbench=sm)


have = all((RUNS / r).exists() for r in ALL)


def synthetic(rates, t0=1.0, cline=0.01):
    """Store-stage rows generated from known rates (TB/s) so that the refit has an exact answer."""
    rows = [dict(fixture='tiny', us=t0)]
    def t(rb, wb, lines): return t0 + rb / (rates['L2_read_sector_TBps'] * 1e6) + wb / (rates['L2_write_sector_TBps'] * 1e6) + lines * cline
    tri_r, tri_w = 12582912.0, 4194304.0
    rows.append(dict(fixture='triad', tier='below', bytes_read=tri_r, bytes_written=tri_w, us=t(tri_r, tri_w, (tri_r + tri_w) / 128)))
    for st in (1, 7, 33):
        sec, lin = fits.store_pattern(st); req = 4194304 / 128
        rows.append(dict(fixture='store', tier='below', stride=st, bytes_written=4194304, us=t(0.0, req * sec * 32, req * lin)))
    rows.append(dict(fixture='store', tier='above', stride=1, bytes_written=1073741824, us=t0 + 1073741824 / (rates['DRAM_write_TBps'] * 1e6)))
    rows.append(dict(fixture='triad', tier='above', bytes_read=805306368, bytes_written=268435456,
                     us=t0 + 805306368 / (rates['DRAM_read_TBps'] * 1e6) + 268435456 / (rates['DRAM_write_TBps'] * 1e6)))
    return rows


def curves(l2r, l2w, dr, dw):
    return dict(l2_total_bandwidth_TBps=dict(read_fraction=[0.0, 0.5, 1.0], value=[l2w, 1.5 * l2r, l2r]), dram_total_bandwidth_TBps=dict(read_fraction=[0.0, 0.5, 1.0], value=[dw, 1.0, dr]))


class Legacy(unittest.TestCase):
    @unittest.skipUnless(have, 'archived calibration runs not present')
    def test_legacy_reproduces_every_archived_document_exactly(self):
        for rel in ALL:
            doc, store, stream = raw(rel); got, _, der = derive(rel, 'legacy')
            ref = doc['constants']['store_legacy']; self.assertEqual(set(got), set(ref), rel)
            for k, v in ref.items():   # the archived documents were fitted on Linux/BLAS; a different platform's least squares may differ in the last bit
                if isinstance(v, dict): self.assertEqual(got[k], v)
                else: self.assertLessEqual(abs(got[k] - v), 1e-12 * max(1.0, abs(v)), (rel, k))
            self.assertEqual(got, original_fit_store(store, doc['device']['sm_count'], doc['constants']['chase_latency_ns']), rel)   # bit-identical to the original derivation on this platform
            self.assertEqual(fits.fit_stream(stream), doc['constants']['stream_curves'], rel)
            self.assertEqual(der['method_used'], 'legacy')

    @unittest.skipUnless(have, 'archived calibration runs not present')
    def test_legacy_warns_loudly_when_unidentified(self):
        for rel in H100:
            _, w, der = derive(rel, 'legacy')
            self.assertFalse(der['identification']['passes'], rel); self.assertTrue(any(x.startswith('UNIDENTIFIED TRAFFIC RATES') for x in w), rel)
        for rel in A100:
            _, w, der = derive(rel, 'legacy'); self.assertTrue(der['identification']['passes'], rel); self.assertFalse(any('UNIDENTIFIED' in x for x in w))
        for rel in BLACKWELL:   # the line coefficient sits at the nonnegativity bound
            _, w, der = derive(rel, 'legacy'); self.assertEqual(der['identification']['coefficient_at_bound'], [False, False, True]); self.assertFalse(der['identification']['passes'])

    @unittest.skipUnless(have, 'archived calibration runs not present')
    def test_identification_numbers(self):
        got = {rel: derive(rel, 'legacy')[2]['identification'] for rel in ALL}
        self.assertAlmostEqual(got[H100[0]]['read_increment_us'], 0.0214, places=3); self.assertLess(got[H100[2]]['read_increment_us'], 0)
        self.assertGreater(got[A100[0]]['read_increment_fraction'], 0.5); self.assertAlmostEqual(got[BLACKWELL[0]]['read_increment_fraction'], 0.082, places=2)


class StreamRates(unittest.TestCase):
    def test_units_and_exact_refit_on_synthetic_windows(self):
        rates = dict(L2_read_sector_TBps=8.0, L2_write_sector_TBps=5.0, DRAM_read_TBps=3.0, DRAM_write_TBps=2.0)
        out, w, der = fits.fit_store_detailed(synthetic(rates, t0=1.0, cline=0.01), 100, {}, curves(8.0, 5.0, 3.0, 2.0), 'stream_rates')
        for k, v in rates.items(): self.assertEqual(out[k], v)           # endpoints are used as given: TB/s of bytes of that kind
        self.assertAlmostEqual(out['t0_us'], 1.0, places=6); self.assertAlmostEqual(out['c_line_ns'], 10.0, places=4)   # 0.01 us per line = 10 ns
        self.assertEqual(w, []); self.assertEqual(der['method_used'], 'stream_rates')

    def test_launch_floor_is_the_measured_empty_launch_not_refitted(self):
        # recorded decision 2026-10-04: windows inconsistent with the stream rates must not move t0; only the line cost absorbs them
        rates = dict(L2_read_sector_TBps=8.0, L2_write_sector_TBps=5.0, DRAM_read_TBps=3.0, DRAM_write_TBps=2.0)
        rows = synthetic(rates, t0=1.0, cline=0.01)
        out, w, der = fits.fit_store_detailed(rows, 100, {}, curves(4.0, 2.5, 3.0, 2.0), 'stream_rates')   # stream rates half the true ones
        self.assertEqual(out['t0_us'], 1.0); self.assertEqual(der['stream_rates_fit']['tiny_launch_floor_us'], 1.0)
        self.assertEqual(out['c_line_ns'] >= 0, True)

    def test_bound_is_reported(self):
        rates = dict(L2_read_sector_TBps=8.0, L2_write_sector_TBps=5.0, DRAM_read_TBps=3.0, DRAM_write_TBps=2.0)
        out, w, der = fits.fit_store_detailed(synthetic(rates, t0=1.0, cline=0.0), 100, {}, curves(8.0, 5.0, 3.0, 2.0), 'stream_rates')
        self.assertTrue(any('nonnegativity' in x for x in w)); self.assertAlmostEqual(out['c_line_ns'], 0.0, places=6)

    def test_missing_curves_or_endpoints_refuse(self):
        rows = synthetic(dict(L2_read_sector_TBps=8.0, L2_write_sector_TBps=5.0, DRAM_read_TBps=3.0, DRAM_write_TBps=2.0))
        for m in ('stream_rates', 'guarded'):
            with self.assertRaises(ValueError): fits.fit_store_detailed(rows, 100, {}, None, m)
        bad = curves(8.0, 5.0, 3.0, 2.0); bad['l2_total_bandwidth_TBps'] = dict(read_fraction=[0.25, 1.0], value=[1.0, 2.0])
        with self.assertRaises(ValueError): fits.fit_store_detailed(rows, 100, {}, bad, 'stream_rates')
        with self.assertRaises(ValueError): fits.fit_store_detailed(rows, 100, {}, curves(8, 5, 3, 2), 'nonsense')

    @unittest.skipUnless(have, 'archived calibration runs not present')
    def test_repeat_runs_agree_within_5_percent(self):
        for pair in ((H100[0], H100[1]), (A100[0], A100[1]), (BLACKWELL[0], BLACKWELL[1])):
            a, b = (derive(r, 'stream_rates')[0] for r in pair)
            for k in ('L2_read_sector_TBps', 'L2_write_sector_TBps', 'DRAM_read_TBps', 'DRAM_write_TBps', 'c_line_ns', 't0_us'):
                self.assertLessEqual(abs(a[k] - b[k]), 0.05 * max(abs(a[k]), abs(b[k]), 1e-9), (pair, k, a[k], b[k]))

    @unittest.skipUnless(have, 'archived calibration runs not present')
    def test_legacy_l2_read_rate_is_not_repeatable_on_h100(self):
        a, b = (derive(r, 'legacy')[0]['L2_read_sector_TBps'] for r in H100[:2]); self.assertGreater(abs(a - b) / max(a, b), 0.3)


class Guarded(unittest.TestCase):
    def test_well_identified_fit_is_kept(self):
        # a clean synthetic board: legacy passes the identification check and the DRAM check, so guarded equals legacy
        rates = dict(L2_read_sector_TBps=2.0, L2_write_sector_TBps=1.5, DRAM_read_TBps=1.0, DRAM_write_TBps=0.9)
        rows = synthetic(rates, t0=1.0, cline=0.002)
        leg, _, dl = fits.fit_store_detailed(rows, 100, {}, curves(2.0, 1.5, 1.0, 0.9), 'legacy', 1 << 20)
        out, _, dg = fits.fit_store_detailed(rows, 100, {}, curves(2.0, 1.5, 1.0, 0.9), 'guarded', 1 << 20)
        self.assertTrue(dl['identification']['passes'], dl['identification'])
        self.assertTrue(dl['dram_plausibility']['passes'], dl['dram_plausibility'])
        self.assertEqual(out, leg); self.assertEqual(dg['method_used'], 'legacy')

    @unittest.skipUnless(have, 'archived calibration runs not present')
    def test_guarded_per_board(self):
        used = {rel: derive(rel, 'guarded')[2]['method_used'] for rel in ALL}
        for rel in H100: self.assertEqual(used[rel], 'stream_rates')                                                   # unidentified reads and implausible DRAM
        for rel in A100: self.assertEqual(used[rel], 'mixed(l2=legacy,dram=stream_rates)')                              # L2 identified; DRAM algebra disagrees with stream
        for rel in BLACKWELL: self.assertEqual(used[rel], 'mixed(l2=stream_rates,dram=legacy)')                         # line coefficient at the bound; DRAM algebra within 10%
        out, _, _ = derive(BLACKWELL[1], 'guarded'); leg, _, _ = derive(BLACKWELL[1], 'legacy'); self.assertEqual(out['DRAM_read_TBps'], leg['DRAM_read_TBps'])

    @unittest.skipUnless(have, 'archived calibration runs not present')
    def test_dram_plausibility_checks(self):
        _, _, d = derive(H100[0], 'legacy'); p = d['dram_plausibility']; self.assertFalse(p['passes'])
        names = {c['check']: c for c in p['checks']}
        self.assertTrue(names['footprint_over_L2']['ok']); self.assertGreater(names['footprint_over_L2']['value'], 4)
        self.assertFalse(names['agrees_with_stream_read_endpoint']['ok']); self.assertIn('not_evaluated', names['below_verified_peak'])   # no peak supplied: not evaluated, not passed
        _, _, d = derive(BLACKWELL[0], 'legacy', dram_peak_TBps=1.0); self.assertFalse(d['dram_plausibility']['passes'])                    # a peak below the rate fails the check
        _, _, d = derive(BLACKWELL[0], 'legacy', dram_peak_TBps=5.0); self.assertTrue(d['dram_plausibility']['passes'])


class Tool(unittest.TestCase):
    @unittest.skipUnless(have, 'archived calibration runs not present')
    def test_rederive_writes_new_document_with_provenance_and_never_overwrites(self):
        src = doc_path(H100[0]); before = hashlib.sha256(src.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / 'new.json'; p, new = R.rederive(src, 'stream_rates', out)
            self.assertEqual(p, out); self.assertEqual(new['rederived_from']['original_document_sha256'], before); self.assertEqual(new['store_derivation']['method_requested'], 'stream_rates')
            self.assertEqual(new['constants']['store_legacy']['L2_read_sector_TBps'], derive(H100[0], 'stream_rates')[0]['L2_read_sector_TBps'])
            self.assertEqual(new['constants']['pipes'], json.loads(src.read_text())['constants']['pipes'])
            self.assertFalse(any(w.startswith('UNIDENTIFIED') for w in new['warnings']))
            with self.assertRaises(SystemExit): R.rederive(src, 'legacy', out)       # exists: refused
            self.assertTrue(json.loads(out.read_text())['rederived_from']['original_warnings'] == json.loads(src.read_text())['warnings'])
            R.rederive(src, 'legacy', Path(t) / 'legacy.json', Path(t) / 'leg')      # legacy re-derivation exports the legacy constants files
            self.assertEqual(json.loads((Path(t) / 'leg' / 'stream_constants.json').read_text())['constants'], derive(H100[0], 'legacy')[0])
        self.assertEqual(hashlib.sha256(src.read_bytes()).hexdigest(), before)       # the original is untouched


if __name__ == '__main__':
    unittest.main()
