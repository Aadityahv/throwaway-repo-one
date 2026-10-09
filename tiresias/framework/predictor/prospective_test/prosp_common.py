"""Input loaders shared by the freeze, scoring and test code: yields every set of static tables (the 131 evaluation cells, the 12 validation cells, the 24 prospective cells) with its constants
directory and calibration document, built exactly as attention_diagnosis/mech_lib.py builds them. CPU only; no label is read here."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; SR = HERE.parent; ST = SR / 'shared_traffic'
sys.path.insert(0, str(SR)); sys.path.insert(0, str(SR / 'calibrate')); sys.path.insert(0, str(ST)); sys.path.insert(0, str(SR / 'attention_diagnosis'))
import predict as P  # noqa
from cal import portable_predict as PP, traffic as T  # noqa
import predict_runtime_v3j as J  # noqa
import evaluate as EV  # noqa
g = lambda p: json.loads(Path(p).read_text())
L2 = EV.l2_capacity_bytes()
HW = None


def _hw():
    return g(SR / 'fresh_g/features_fresh_g.json')['hardware_from_ground_truth']


def iter_sets(which=('eval', 'validation', 'prosp')):
    """yield dict(name, feats, ph, un, bk, C, docp, constants_dir, kind) per set; un has the grid footprints attached."""
    if 'eval' in which:
        for name, d, ff, pf, uf, bf, cd, docp in EV.SETS:
            feats = g(d / ff); ph = g(d / pf); ph = ph.get('rows', ph); un = g(d / uf); un = un.get('rows', un); bk = g(bf)['rows']
            C = PP.load_constants(cd); C['smem'] = None
            yield dict(name=name, kind='eval', feats=feats, ph=ph, un=J.attach_footprints(un, EV.load_footprints(name), L2), bk=bk, C=C, docp=docp)
    CONST = SR / 'fresh_g/constants_tensor'
    def from_tables(tp, cd, docp, kind, name):
        t = g(tp); ok = {c: v for c, v in t.items() if 'refused' not in v}
        feats = {'hardware_from_ground_truth': _hw(), 'rows': [v['features'] for v in ok.values()]}
        ph = {c: v['phases'] for c, v in ok.items()}; un = {c: v['unique'] for c, v in ok.items()}; bk = {c: v['bank'] for c, v in ok.items()}
        un_fp = J.attach_footprints(un, {c: v['footprints'] for c, v in ok.items()}, L2)
        C = PP.load_constants(cd); C['smem'] = None
        return dict(name=name, kind=kind, feats=feats, ph=ph, un=un_fp, bk=bk, C=C, docp=docp, refused={c: v['refused'] for c, v in t.items() if 'refused' in v}, cells={c: v.get('cell') for c, v in t.items()})
    if 'validation' in which:
        V = ST / 'validation'
        for lib, cd, docp in (('f', SR / 'constants', EV.RUN_DOC), ('e', SR / 'constants', EV.RUN_DOC), ('classic', SR / 'constants', EV.RUN_DOC), ('g', CONST, EV.TENSOR_DOC), ('h', CONST, EV.TENSOR_DOC)):
            yield from_tables(V / 'tables' / ('%s.json' % lib), cd, docp, 'validation', 'validation_' + lib)
    if 'prosp' in which:
        yield from_tables(HERE / 'tables' / 'prosp.json', CONST, EV.TENSOR_DOC, 'prosp', 'prosp')
