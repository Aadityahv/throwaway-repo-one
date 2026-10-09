"""Shared loader: predictions of the current model (v3j, wave rule) for the tensor-core matrix multiply and fused attention cells (evaluation sets g, h and the 4 validation cells),
with the per-phase terms and the static tables, CPU only. Nothing here changes a model file."""
import csv, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; SR = HERE.parent; PW = SR.parents[1]; ST = SR / 'shared_traffic'
sys.path.insert(0, str(SR)); sys.path.insert(0, str(SR / 'calibrate')); sys.path.insert(0, str(ST))
import predict as P  # noqa
from cal import portable_predict as PP, traffic as T  # noqa
import predict_runtime_v3j as J  # noqa
import evaluate as EV  # noqa
g = lambda p: json.loads(Path(p).read_text())
L2 = EV.l2_capacity_bytes()
CONST = SR / 'fresh_g/constants_tensor'


def load_all():
    """-> {cid: dict(set, feat, ph, un, bank, pred, measured_s)} for eval sets g,h and validation g,h cells."""
    out = {}
    lab = {r['cell_id']: r for r in csv.DictReader(open(ST / 'cells_scored.csv'))}   # committed labels of the 131 evaluation cells (not the working-tree evaluation files, which other agents edit)
    for name in ('g', 'h'):
        d = SR / ('fresh_' + name)
        feats = g(d / ('features_fresh_%s.json' % name)); ph = g(d / ('phases_fresh_%s.json' % name)); ph = ph.get('rows', ph)
        un = g(d / ('phases_unique_fresh_%s.json' % name)); un = un.get('rows', un); bk = g(d / ('bank_conflicts_fresh_%s.json' % name))['rows']
        fps = EV.load_footprints(name); un_fp = J.attach_footprints(un, fps, L2)
        C = PP.load_constants(CONST); C['smem'] = None
        pred = T.predict_candidate(feats, ph, un_fp, bk, C, 188)
        for r in feats['rows']:
            c = r['cell_id']
            out[c] = dict(set='eval_' + name, feat=r, ph=ph[c], un=un_fp[c], bank=bk[c], pred=pred[c], measured_s=float(lab[c]['runtime_measured_s']) if c in lab else None,
                          energy_j=float(lab[c]['energy_measured_j']) if c in lab else None, power_w=float(lab[c]['window_power_w']) if c in lab else None)
    V = ST / 'validation'
    rt = {}
    for f in ('g', 'h'):
        for c in g(V / 'raw' / ('timing_%s.json' % f))['cells']: rt[c['cell_id']] = c['per_launch_runtime_s']
    for name in ('g', 'h'):
        t = g(V / 'tables' / ('%s.json' % name))
        ok = {c: v for c, v in t.items() if 'refused' not in v}
        feats = {'hardware_from_ground_truth': g(SR / 'fresh_g/features_fresh_g.json')['hardware_from_ground_truth'], 'rows': [v['features'] for v in ok.values()]}
        ph = {c: v['phases'] for c, v in ok.items()}; un = {c: v['unique'] for c, v in ok.items()}; bk = {c: v['bank'] for c, v in ok.items()}
        un_fp = J.attach_footprints(un, {c: v['footprints'] for c, v in ok.items()}, L2)
        C = PP.load_constants(CONST); C['smem'] = None
        pred = T.predict_candidate(feats, ph, un_fp, bk, C, 188)
        for r in feats['rows']:
            c = r['cell_id']
            out[c] = dict(set='validation_' + name, feat=r, ph=ph[c], un=un_fp[c], bank=bk[c], pred=pred[c], measured_s=rt.get(c), energy_j=None, power_w=None)
    return out
