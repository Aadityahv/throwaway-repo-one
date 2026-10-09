"""Tests of predict_runtime_v3k (CPU only): (1) with all three rules off the model equals v3j (the shared-traffic model) bit for bit on every evaluation and validation cell;
(2) with the rules on it equals the retrospective composition code of the diagnosis (mech_lib.compose with Mech(beta_tc, stage, tail)) to 1e-9 relative; (3) every cell with no tensor-core
instruction and at most one block per SM is bit-identical to v3j with the rules on.   python3 test_runtime_v3k.py"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'prospective_test')); sys.path.insert(0, str(HERE))
import prosp_common as C
import mech_lib as ML
from mechanisms import Mech
from cal import traffic as T
import predict_runtime_v3k as K3


def main():
    mcells = {c['cid']: c for c in ML.load_cells()}; n = bad1 = bad2 = bad3 = moved = 0; mm = Mech(beta_tc=True, stage=True, tail=True)
    for s in C.iter_sets(('eval', 'validation')):
        j = T.predict_candidate(s['feats'], s['ph'], s['un'], s['bk'], s['C'], 188)
        off = K3.predict_portable(s['feats'], s['ph'], s['un'], s['bk'], s['C'], 188, pair=False, stage=False, busiest=False)
        on = K3.predict_portable(s['feats'], s['ph'], s['un'], s['bk'], s['C'], 188)
        for r in s['feats']['rows']:
            c = r['cell_id']; n += 1
            if j[c].get('primary_s') != off[c].get('primary_s') or j[c].get('traffic') != off[c].get('traffic'): bad1 += 1; print('RULES-OFF MISMATCH', c)
            if j[c].get('primary_s') is None:
                if on[c].get('primary_s') is not None: bad2 += 1; print('REFUSAL CHANGED', c)
                continue
            ref = ML.compose(mcells[c], mm)
            if abs(on[c]['primary_s'] / ref - 1) > 1e-9: bad2 += 1; print('COMPOSE MISMATCH', c, on[c]['primary_s'], ref)
            has_mma = any(op.startswith(K3.TENSOR_PREFIXES) for k in s['ph'][c]['kernels'] for p in k['phases'] for op in p['issue_warp_instructions'])
            blocks = [(r['geometry'].get('grid_blocks') or r['geometry'].get('grid_blocks_dispatch')) for r in [r] + r.get('secondary_kernels', [])]
            if not has_mma and max(blocks) <= 188 and on[c]['primary_s'] != j[c]['primary_s']: bad3 += 1; print('UNTOUCHED CELL CHANGED', c)
            moved += on[c]['primary_s'] != j[c]['primary_s']
    print('%d cells: rules-off vs v3j mismatches %d; composition mismatches %d; untouched-cell changes %d; cells whose prediction the rules move: %d' % (n, bad1, bad2, bad3, moved))
    assert bad1 == bad2 == bad3 == 0
    print('PASS')


if __name__ == '__main__': main()
