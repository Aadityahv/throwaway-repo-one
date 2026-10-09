import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cal import archive, fits, stages  # noqa: E402


class Slim(unittest.TestCase):
    def test_slim_summarises_then_removes_raw_and_summary_equals_direct_fit_inputs(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t); cells = run / 'stages' / 'pipes' / 'cells'; grid = stages.pipes_grid(dict(sm_count=4, l2_bytes=4 << 20))[:3] + [r for r in stages.pipes_grid(dict(sm_count=4, l2_bytes=4 << 20)) if r['kind'] == 'memory'][:1]
            (run / 'stages' / 'pipes').mkdir(parents=True); (run / 'stages' / 'pipes' / 'grid.json').write_text(json.dumps(grid))
            expect = {}
            for r in grid:
                d = cells / r['slot']; d.mkdir(parents=True); lanes = r['blocks'] * r['threads']; warps = lanes // 32
                cyc = np.arange(warps, dtype='<u8') + 1000
                for s in range(3): (d / ('sample%d.bin' % s)).write_bytes(b'\x00' * (lanes * r['streams'] * 4) + cyc.tobytes() + b'\x00' * (r['blocks'] * 4))
                (d / 'result.json').write_text(json.dumps(dict(event_ms=[1.0, 1.1, 1.2])))
                (d / 'input_pattern.bin').write_bytes(b'x')
                expect[r['slot']] = fits.load_cell(d, r)[0] if r['kind'] == 'compute' else fits.chase_cycles_per_step(d, r)
            n = archive.slim(run)
            self.assertEqual(n, len(grid))
            self.assertFalse(list(cells.rglob('*.bin')))
            summ = json.loads((run / 'stages' / 'pipes' / 'cells_summary.json').read_text())
            for r in grid:
                got = summ[r['slot']].get('device_cycle_span', summ[r['slot']].get('cycles_per_step')); self.assertAlmostEqual(got, expect[r['slot']])
            man = json.loads((run / 'stages' / 'pipes' / 'raw_files_sha256.json').read_text()); self.assertEqual(len(next(iter(man.values()))), 4)


if __name__ == '__main__':
    unittest.main()
