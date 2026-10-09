"""Build the static tables of fresh set E (CPU only; reads no measured value). One JSON per cell in build/cells, then merged tables."""
import json, multiprocessing, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fresh_e_lib as L
import cells_e as CE
UP = L.UP; C = UP.C; P = UP.P
OUT = HERE / 'build/cells'

def one(cell):
    out = OUT / (cell['cell_id'].replace('/', '__') + '.json')
    if out.exists(): return cell['cell_id'], 'skip'
    hw = UP.X.load_hardware(UP.X.read_text(UP.X.GROUND_TRUTH))
    try:
        rec, prow, urow = UP.build_cell(cell, hw)
        doc = dict(features=rec, phases=prow, unique=urow, cell={k: v for k, v in cell.items() if k != '_dev'})
        msg = '%s %s %s %.0fs' % (rec['status'], prow['status'], urow['status'], rec['build_seconds'])
    except (C.Refusal, P.Refusal) as ex:
        doc = dict(refused=str(ex), cell={k: v for k, v in cell.items() if k != '_dev'}); msg = 'REFUSED ' + str(ex)[:200]
    out.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + '\n')
    return cell['cell_id'], msg

def merge():
    feats, phases, uniq, cells = [], {}, {}, {}
    for p in sorted(OUT.glob('*.json')):
        d = json.loads(p.read_text()); c = d['cell']; cid = c['cell_id']; cells[cid] = c
        if 'refused' in d:
            feats.append(dict(cell_id=cid, status='missing_features', missing_features=[dict(feature='static pipeline', reason=d['refused'])]))
            phases[cid] = dict(status='unsupported', reason=d['refused'], kernels=[]); uniq[cid] = dict(status='unsupported', reason=d['refused'], kernels=[])
        else: feats.append(d['features']); phases[cid] = d['phases']; uniq[cid] = d['unique']
    hw = UP.X.load_hardware(UP.X.read_text(UP.X.GROUND_TRUTH))
    (HERE / 'features_fresh_e.json').write_text(json.dumps({'schema': 'static_runtime_features_blackwell/1', 'hardware_from_ground_truth': hw, 'rows': sorted(feats, key=lambda r: r['cell_id']),
        'note': 'Label-free static features of fresh set E (pinned cuda-samples). No runtime, energy or power value was read.'}, indent=1, sort_keys=True) + '\n')
    (HERE / 'phases_fresh_e.json').write_text(json.dumps(dict(schema='static_dynamic_barrier_phases/1', rows=phases), indent=1, sort_keys=True) + '\n')
    (HERE / 'phases_unique_fresh_e.json').write_text(json.dumps(dict(schema='first_touch_unique_sectors/1', rows=uniq), indent=1, sort_keys=True) + '\n')
    (HERE / 'fresh_cells_e.json').write_text(json.dumps(dict(cells=[cells[k] for k in sorted(cells)]), indent=1, sort_keys=True) + '\n')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hw = UP.X.load_hardware(UP.X.read_text(UP.X.GROUND_TRUTH)); cells = CE.define_cells(hw)
    if '--only' in sys.argv: cells = [c for c in cells if sys.argv[sys.argv.index('--only') + 1] in c['cell_id']]
    t0 = time.time()
    with multiprocessing.get_context('fork').Pool(6) as pool:
        for cid, msg in pool.imap_unordered(one, cells): print('%-70s %s (%.0fs)' % (cid, msg, time.time() - t0), flush=True)
    merge()
if __name__ == '__main__': main()
