"""Footprints of the DRAM-tier cells of the Blackwell development corpus (132 cells; L2-tier cells need no DRAM footprint), by the same loops as reuse/phases_unique.py main()
(called unchanged; its output file is not written). CPU only.  python3 run_footprints_dev.py"""
import json, sys, time, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent; SR = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(SR / 'reuse'))
import footprint as FP
import phases_unique as PUD
C, A, P, PH = PUD.C, PUD.A, PUD.P, PUD.PH
out = HERE / 'footprints/dev'; out.mkdir(parents=True, exist_ok=True)
frozen = json.loads((SR / 'phases_blackwell.json').read_text())['rows']
feats = {r['cell_id']: r for r in json.loads((SR / 'features_blackwell.json').read_text())['rows']}
fz = {r['cell_id']: r for r in json.loads((SR / 'coalescing/static_sectors_frozen.json').read_text())['rows']}
dram = {c for c, r in feats.items() if (r.get('memory') or {}).get('tier') == 'DRAM'}
print(len(dram), 'DRAM-tier development cells', flush=True)


def save(cid, kernels=None, reason=None):
    d = dict(cell_id=cid, status='ok' if kernels else 'refused', reason=reason, kernels=kernels or [])
    (out / (cid.replace('/', '__') + '.json')).write_text(json.dumps(d, indent=1, sort_keys=True, default=str) + '\n'); print(cid, d['status'], reason or '', flush=True)


for corpus, root in C.D.CORPORA.items():
    for row in json.loads((root / 'retention_manifest.json').read_text())['rows']:
        cid = 'blackwell/' + row['operator_id'] + '/' + row['cell']
        if cid not in dram: continue
        if frozen[cid]['status'] != 'conditional_static_phases': save(cid, reason='frozen phase table unsupported: ' + str(frozen[cid].get('reason'))); continue
        del PUD.CAPTURED[:]
        try:
            PH.retained(corpus, row, root, fz[cid]); obs = list(PUD.CAPTURED)
            fp = FP.kernel_footprint(PUD, obs, obs[0].blocks); fp['kernel_id'] = 'retained'; save(cid, [fp])
        except Exception as ex: save(cid, reason='%s: %s' % (type(ex).__name__, ex))
dispatch = json.loads((SR / 'pytorch_dispatch/dispatch_trace.json').read_text()); cache = {}
for cell in dispatch['cells']:
    cid = cell['cell_id']
    if cid not in dram: continue
    if frozen[cid]['status'] != 'conditional_static_phases': save(cid, reason='frozen phase table unsupported: ' + str(frozen[cid].get('reason'))); continue
    try:
        res = []
        for kern, fk in zip(cell['kernels'], frozen[cid]['kernels']):
            kid = A.kernel_id(kern); key = json.dumps([kid, kern, cell['input'] if kid == 'k1' else None], sort_keys=True)
            if key not in cache:
                del PUD.CAPTURED[:]; PH.pytorch(kid, kern, cell); cache[key] = list(PUD.CAPTURED)
            fp = FP.kernel_footprint(PUD, cache[key], cache[key][0].blocks); fp['kernel_id'] = kid; res.append(fp)
        save(cid, res)
    except Exception as ex: save(cid, reason='%s: %s' % (type(ex).__name__, ex))
