"""Static tables of the 24 prospective cells (no GPU, no label): features, barrier phases, first-touch tables, bank-conflict tables and grid footprints, by the same pipeline calls the evaluation and
validation sets use (UP.build_cell, bank_unseen.one, run_footprints.cell_up).  python3 build_prosp.py [--only SUBSTR]  -> tables/prosp.json"""
import argparse, json, multiprocessing as mp, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; SR = HERE.parent; ST = SR / 'shared_traffic'
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ST)); sys.path.insert(0, str(SR / 'bank'))
import prosp_lib as L, cells_prosp as V
G = {}


def one(cell):
    import run_footprints as RF, bank_unseen as BU
    UP = L.UP; hw = G['hw']; t = time.time()
    try:
        rec, prow, urow = UP.build_cell(cell, hw)
        cid, brow = BU.one((cell, prow)) if prow.get('kernels') else (cell['cell_id'], dict(status='refused', reason='no phases'))
        fps = RF.cell_up(UP, cell, hw)
        return cell['cell_id'], dict(cell={k: v for k, v in cell.items() if k != '_dev'}, features=rec, phases=prow, unique=urow, bank=brow, footprints=dict(cell_id=cell['cell_id'], status='ok', kernels=fps), seconds=round(time.time() - t, 1))
    except Exception as ex:
        import traceback
        return cell['cell_id'], dict(cell={k: v for k, v in cell.items() if k != '_dev'}, refused='%s: %s' % (type(ex).__name__, ex), trace=traceback.format_exc()[-1500:], seconds=round(time.time() - t, 1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=4); ap.add_argument('--only', default=''); a = ap.parse_args()
    hw = L.UP.X.load_hardware(L.UP.X.read_text(L.UP.X.GROUND_TRUTH)); cells = [c for c in V.define_cells(hw) if a.only in c['cell_id']]; G.update(hw=hw)
    out = {}
    with mp.get_context('fork').Pool(a.jobs) as pool:
        for cid, r in pool.imap_unordered(one, cells, chunksize=1): out[cid] = r; print(cid, r.get('refused') or 'ok', r['seconds'], flush=True)
    p = HERE / 'tables' / ('prosp%s.json' % ('_' + a.only.replace('/', '_') if a.only else '')); p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + '\n')


if __name__ == '__main__': main()
