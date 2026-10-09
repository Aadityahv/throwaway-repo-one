#!/usr/bin/env python3
"""Regenerate the bank-conflict tables of set E and of the 32 unseen cells WITH the per-phase request-wavefront histograms, as NEW files (the frozen tables, whose hashes
are recorded in frozen prediction files, are not touched). The analysis is bank/bank_unseen.py's, unchanged, with its phase summation extended to keep the histogram.
    python3 calibrate/tools/regen_bank_wavefront.py e        -> bank/bank_conflicts_fresh_e_wavefront.json
    python3 calibrate/tools/regen_bank_wavefront.py unseen   -> bank/bank_conflicts_unseen_wavefront.json
Gate: every cell's cost recomputed from the histogram with floor 2.0 and slope 1.0 must equal the cost stored in the frozen table (checked in tests/test_regen_bank.py)."""
import collections
import json
import multiprocessing
import sys
from pathlib import Path

SR = Path(__file__).resolve().parents[2]
which = sys.argv[1] if len(sys.argv) > 1 else ''
if which not in ('e', 'unseen'): raise SystemExit('usage: regen_bank_wavefront.py e|unseen')
sys.path.insert(0, str(SR / 'bank'))
if which == 'e':
    sys.path.insert(0, str(SR / 'fresh_e')); import fresh_e_lib as L  # registers the set-E kernels with the unseen pipeline (imported, never edited)
    import cells_e as CE
else:
    sys.path.insert(0, str(SR / 'unseen_kernels'))
import bank_unseen as BU  # noqa: E402

_orig = BU.sum_rows

def sum_rows_w(rows):
    out = _orig(rows); h = collections.Counter()
    for r in rows: h.update(r['shared_request_wavefront_histogram'])
    out['shared_request_wavefront_histogram'] = {str(k): v for k, v in sorted(h.items(), key=lambda kv: int(kv[0]))}
    return out

BU.sum_rows = sum_rows_w

def main():
    if which == 'e':
        hw = L.UP.X.load_hardware(L.UP.X.read_text(L.UP.X.GROUND_TRUTH)); frozen = json.loads((SR / 'fresh_e' / 'phases_fresh_e.json').read_text())['rows']; cells = CE.define_cells(hw)
        out = SR / 'bank' / 'bank_conflicts_fresh_e_wavefront.json'; name = 'fresh_e'
    else:
        import cells as CELLS
        hw = BU.X.load_hardware(BU.X.read_text(BU.X.GROUND_TRUTH)); frozen = json.loads((SR / 'unseen_kernels' / 'frozen' / 'phases_unseen.json').read_text())['rows']; cells = CELLS.define_cells(hw)
        out = SR / 'bank' / 'bank_conflicts_unseen_wavefront.json'; name = 'unseen_exposed'
    cells = [c for c in cells if c['cell_id'] in frozen and frozen[c['cell_id']].get('kernels')]; rows = {}
    with multiprocessing.get_context('fork').Pool(6) as pool:
        for cid, row in pool.imap_unordered(BU.one, [(c, frozen[c['cell_id']]) for c in cells]):
            rows[cid] = row; print(cid, row.get('status', 'gated ok'), str(row.get('reason', ''))[:120], flush=True)
    out.write_text(json.dumps(dict(set=name, schema='bank_conflicts/1', note='same analysis as the frozen table plus the per-phase request-wavefront histogram', rows=rows), indent=1, sort_keys=True) + '\n')

if __name__ == '__main__':
    main()
