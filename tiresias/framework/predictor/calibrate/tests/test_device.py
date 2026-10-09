import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cal import device as D

GT = """
## Ada (RTX 5000 Ada Generation) — fully verified
| Compute capability | sm_89 (major=8, minor=9) | x |
| SM count | 100 | x |
| L2 cache size | 67,108,864 bytes = exactly 64.0 MB | x |
## Blackwell (RTX PRO 6000) — partially verified
| Compute capability | 12.0 (sm_120, matches) | x |
| SM count | 188 | x |
| L2 cache size | 134,217,728 bytes (128.0 MiB) | x |
## A100 (A100-SXM4-80GB) — verified
| Compute capability | sm_80 (8.0) | x |
| SM count | 108 | x |
| L2 cache size | 41,943,040 bytes = 40.0 MiB | x |
## H200 — not attempted
"""
BW = 'GPU-0e63baea-0bb1-e90c-f19d-8aa5f2995894'


class DeviceTests(unittest.TestCase):
    def test_parse_four_known_gpus(self):
        self.assertEqual(D.parse_ground_truth('Ada', GT), dict(sm_count=100, l2_bytes=67108864, compute_capability='8.9'))
        self.assertEqual(D.parse_ground_truth('Blackwell', GT), dict(sm_count=188, l2_bytes=134217728, compute_capability='12.0'))
        self.assertEqual(D.parse_ground_truth('A100', GT), dict(sm_count=108, l2_bytes=41943040, compute_capability='8.0'))

    def test_missing_or_unverified_section_refuses(self):
        with self.assertRaises(D.Refusal): D.parse_ground_truth('H100', GT)
        with self.assertRaises(D.Refusal): D.parse_ground_truth('H200', GT)

    def test_unlisted_and_malformed_uuid_refuse(self):
        allow = D.load_allow_list()
        self.assertIn(BW, allow)
        with self.assertRaises(D.Refusal): D.require_approved('GPU-00000000-12b9-377c-be7e-5c94e8a70d9d', allow)
        with self.assertRaises(D.Refusal): D.require_approved('1', allow)
        with self.assertRaises(D.Refusal): D.require_approved('', allow)
        self.assertEqual(D.require_approved(BW, allow)['machine'], 'blackwell')

    def test_placeholder_entries_are_not_approvals(self):
        self.assertTrue(all(not u.startswith(D.PLACEHOLDER) for u in D.load_allow_list()))

    def test_booking_required(self):
        with self.assertRaises(D.Refusal): D.require_booking('')
        with self.assertRaises(D.Refusal): D.require_booking('x')
        self.assertTrue(D.require_booking('booking log 2026-10-02 BOOKING calibrate'))

    def test_cross_check_mismatch_refuses(self):
        gt = D.parse_ground_truth('Blackwell', GT); entry = D.load_allow_list()[BW]
        good = dict(sm_count=188, l2_bytes=134217728, compute_capability='12.0')
        D.cross_check(good, entry, gt)
        for k, v in (('sm_count', 132), ('l2_bytes', 50 << 20), ('compute_capability', '9.0')):
            bad = dict(good); bad[k] = v
            with self.assertRaises(D.Refusal): D.cross_check(bad, entry, gt)

    def test_idle_guard_with_stubbed_smi(self):
        class R:
            def __init__(s, out, rc=0): s.stdout, s.stderr, s.returncode = out, '', rc
        def run(cmd, **k):
            if '--query-gpu=uuid,utilization.gpu,memory.used' in cmd[1:2] + cmd[1:]: return R(BW + ', 0, 15\nGPU-other, 0, 240\n')
            return R(self.apps)
        self.apps = ''
        self.assertEqual(D.require_idle(BW, run)['utilization_pct'], 0)
        self.apps = BW + ', 123, python\n'
        with self.assertRaises(D.Refusal): D.require_idle(BW, run)
        self.apps = 'GPU-other, 5, vllm\n'                      # a process on ANOTHER GPU does not block this one
        self.assertEqual(D.require_idle(BW, run)['memory_used_mib'], 15)

    def test_wait_idle_settles_after_lagging_utilisation_and_refuses_on_processes(self):
        class R:
            def __init__(s, out, rc=0): s.stdout, s.stderr, s.returncode = out, '', rc
        util = iter([99, 40, 0, 0, 0, 0])
        def run(cmd, **k):
            if '--query-compute-apps=gpu_uuid,pid,process_name' in cmd: return R('')
            return R('%s, %d, 15\n' % (BW, next(util)))
        r = D.wait_idle(BW, run, sleep=lambda s: None)
        self.assertEqual(r['utilization_pct'], 0); self.assertGreater(r['settled_after_s'], 0)
        def stuck(cmd, **k):
            if '--query-compute-apps=gpu_uuid,pid,process_name' in cmd: return R('')
            return R('%s, 99, 15\n' % BW)
        with self.assertRaises(D.Refusal): D.wait_idle(BW, stuck, sleep=lambda s: None, timeout_s=6.0)
        def proc(cmd, **k):
            if '--query-compute-apps=gpu_uuid,pid,process_name' in cmd: return R('%s, 7, x\n' % BW)
            return R('%s, 0, 15\n' % BW)
        with self.assertRaises(D.Refusal): D.wait_idle(BW, proc, sleep=lambda s: None)


if __name__ == '__main__':
    unittest.main()
