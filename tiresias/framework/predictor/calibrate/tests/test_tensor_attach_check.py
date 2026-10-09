"""CPU tests of tools/check_tensor_attach.py: the cross-check that the tensor constants of a full calibration run equal the ones attach_tensor.py re-derives from the same run."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
import check_tensor_attach as C  # noqa: E402

TENSOR = dict(status='ok', issue=dict(issue_cycles_per_warp_instruction_per_sm=3.99, dependent_latency_cycles=29.8),
              energy=dict(status='ok', rate_pJ_per_lane_instruction=88.9, admitted=['tc_w1', 'tc_w4', 'tc_w16'], not_admitted=[]))


class Check(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp()); self.addCleanup(__import__('shutil').rmtree, self.d, True)

    def write(self, name, tensor):
        p = self.d / name; p.write_text(json.dumps(dict(constants=dict(tensor=tensor) if tensor is not None else dict()))); return p

    def run_main(self, doc, att, *extra):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = C.main(['--doc', str(doc), '--attached', str(att), *extra])
        return rc, buf.getvalue()

    def test_agreeing_derivations_pass(self):
        rc, out = self.run_main(self.write('a.json', TENSOR), self.write('b.json', TENSOR), '--require-ok')
        self.assertEqual(rc, 0); self.assertIn('attach=AGREES', out); self.assertIn('rate_pJ=88.9', out)

    def test_differing_derivations_fail(self):
        other = json.loads(json.dumps(TENSOR)); other['energy']['rate_pJ_per_lane_instruction'] = 90.0
        rc, out = self.run_main(self.write('a.json', TENSOR), self.write('b.json', other))
        self.assertEqual(rc, 1); self.assertIn('attach=DIFFERS', out)

    def test_missing_stage_or_missing_attached_document_fails(self):
        rc, out = self.run_main(self.write('a.json', None), self.write('b.json', TENSOR))
        self.assertEqual(rc, 1); self.assertIn('missing', out)
        rc, out = self.run_main(self.write('c.json', TENSOR), self.d / 'nothing.json')
        self.assertEqual(rc, 1); self.assertIn('DIFFERS', out)

    def test_require_ok_rejects_an_unidentified_rate(self):
        bad = json.loads(json.dumps(TENSOR)); bad['energy'] = dict(status='UNIDENTIFIED', rate_pJ_per_lane_instruction=None, admitted=['tc_w1'], not_admitted=['tc_w4'])
        rc, out = self.run_main(self.write('a.json', bad), self.write('b.json', bad), '--require-ok')
        self.assertEqual(rc, 3)


if __name__ == '__main__':
    unittest.main()
