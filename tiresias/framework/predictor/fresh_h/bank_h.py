"""Bank-conflict tables for the tensor-core matrix multiply set (fresh set H) (CPU only). Uses bank/bank_unseen.py's analysis (imported, never edited) with this set's kernel registry."""
import json, multiprocessing, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; SR = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(SR / 'bank'))
import fresh_h_lib as L
import bank_unseen as BU
import cells_h as CE

def main():
    hw = L.UP.X.load_hardware(L.UP.X.read_text(L.UP.X.GROUND_TRUTH)); frozen = json.loads((HERE / 'phases_fresh_h.json').read_text())['rows']
    cells = [c for c in CE.define_cells(hw) if c['cell_id'] in frozen and frozen[c['cell_id']].get('kernels')]
    rows = {}
    with multiprocessing.get_context('fork').Pool(6) as pool:
        for cid, row in pool.imap_unordered(BU.one, [(c, frozen[c['cell_id']]) for c in cells]):
            rows[cid] = row; print(cid, row.get('status', 'gated ok'), str(row.get('reason', ''))[:200], flush=True)
    (HERE / 'bank_conflicts_fresh_h.json').write_text(json.dumps(dict(set='fresh_h', schema='bank_conflicts/1', rows=rows), indent=1, sort_keys=True) + '\n')
if __name__ == '__main__': main()
