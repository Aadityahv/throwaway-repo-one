"""Static tables of the prospective validation cells (no GPU, no label): features, barrier phases, first-touch tables, bank-conflict tables and grid footprints, by the same pipeline calls
the evaluation sets use (UP.build_cell, bank_unseen.one, run_footprints.cell_up), per kernel family in its own process.  python3 build_validation.py --lib f|g|h|e|classic"""
import argparse, json, multiprocessing as mp, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; ST = HERE.parent; SR = ST.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ST)); sys.path.insert(0, str(SR / 'bank'))
import cells_validation as V
G = {}


def one(cell):
    import run_footprints as RF, bank_unseen as BU
    UP = G['UP']; hw = G['hw']; t = time.time()
    try:
        rec, prow, urow = UP.build_cell(cell, hw)
        cid, brow = BU.one((cell, prow)) if prow.get('kernels') else (cell['cell_id'], dict(status='refused', reason='no phases'))
        fps = RF.cell_up(UP, cell, hw)
        return cell['cell_id'], dict(cell={k: v for k, v in cell.items() if k != '_dev'}, features=rec, phases=prow, unique=urow, bank=brow, footprints=dict(cell_id=cell['cell_id'], status='ok', kernels=fps),
                                     seconds=round(time.time() - t, 1))
    except Exception as ex:
        return cell['cell_id'], dict(cell={k: v for k, v in cell.items() if k != '_dev'}, refused='%s: %s' % (type(ex).__name__, ex), seconds=round(time.time() - t, 1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--lib', required=True); ap.add_argument('--jobs', type=int, default=2); a = ap.parse_args()
    if V.LIBS[a.lib]:
        sys.path.insert(0, str(SR / {'f': 'fresh_f', 'g': 'fresh_g', 'h': 'fresh_h', 'e': 'fresh_e'}[a.lib])); UP = __import__(V.LIBS[a.lib]).UP
    else:
        sys.path.insert(0, str(SR / 'unseen_kernels')); import unseen_pipeline as UP
    hw = UP.X.load_hardware(UP.X.read_text(UP.X.GROUND_TRUTH)); cells = V.define(a.lib, hw); G.update(UP=UP, hw=hw)
    out = {}
    with mp.get_context('fork').Pool(a.jobs) as pool:
        for cid, r in pool.imap_unordered(one, cells, chunksize=1): out[cid] = r; print(cid, r.get('refused') or 'ok', r['seconds'], flush=True)
    (HERE / 'tables' / ('%s.json' % a.lib)).write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + '\n')


if __name__ == '__main__': main()
