import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_runs as CR  # noqa: E402


class CompareRuns(unittest.TestCase):
    def test_identifies_constants_outside_the_bound(self):
        a = dict(device=dict(uuid='u'), constants=dict(pipes=dict(issue=dict(fp32=1.0, mufu=2.0)), overlap=dict(alpha=[0.5, 0.6])))
        b = dict(device=dict(uuid='u'), constants=dict(pipes=dict(issue=dict(fp32=1.02, mufu=2.4)), overlap=dict(alpha=[0.5, 0.6])))
        rows, bad = CR.compare(a, b, 0.05)
        self.assertEqual(len(rows), 4); self.assertEqual([r[0] for r in bad], ['/pipes/issue/mufu'])

    def test_exact_zero_pairs_are_not_counted_and_lists_of_numbers_are_flattened(self):
        a = dict(constants=dict(x=dict(v=[0.0, 1.0]))); b = dict(constants=dict(x=dict(v=[0.0, 1.0])))
        rows, bad = CR.compare(a, b); self.assertEqual(len(rows), 1); self.assertFalse(bad)


if __name__ == '__main__':
    unittest.main()
