"""Loader and composer for mechanism tests. Loads every cell of the 131 evaluation cells and the 12 validation cells, runs the CURRENT model (v3j, per-wave rule) unchanged, and keeps for each
kernel launch the per-phase terms, the per-phase pipe times (recomputed with the current model's own classes, bank-conflict shared cost, tensor class and constants) and the composition
parameters (barrier, launch, dispatch, overlap alpha). compose() re-derives the model's runtime from these (verified equal to the model on every cell) and lets a mechanism change one rule.
Nothing here edits a model file; constants come only from the committed microbenchmark documents."""
import collections, csv, json, math
from common import *
import predict_runtime_v2 as V2
import predict_runtime_v3 as V3

DISP = V3.DISPATCH_CYCLES
SETS_ALL = EV.SETS


def tensor_classify(op):
    if op.startswith(('HMMA', 'IMMA', 'QMMA', 'DMMA')): return 'tensor_mma', 1.0
    return V2.classify(op)


def phase_pipes(ph, bank_ph, K):
    """cycles per SM-work unit (before dividing by active SMs x clock) per pipe class of one phase, as the current model computes them (issue_cycles_v3h + tensor class)."""
    pipes = collections.Counter(); total = 0
    for op, n in ph['issue_warp_instructions'].items():
        total += n
        cls, mult = tensor_classify(op)
        if cls == 'shared_load': continue
        if cls: pipes[cls] += n * mult * K['issue'][cls]
    pipes['shared_load'] = float(bank_ph['shared']['shared_cost_cycles'] or 0.0)
    return pipes, DISP * total


def load_cells():
    """-> list of dict(cid, set, kernel_family, validation, measured_s, kernels=[dict(...)], pred_primary_s)"""
    cells = []
    lab = {r['cell_id']: r for r in csv.DictReader(open(ST / 'cells_scored.csv'))}
    sm_total = 188
    def add(cid, setname, feat, ph, un_fp, bank, pred, C, measured, validation, kernel_label, tier, docp=None, energy_j=None):
        K = V2.make_constants(C['stream'], C['micro']); sm = sm_total
        ks = []
        secondary = {k['kernel_id']: k for k in feat.get('secondary_kernels', [])}
        if pred.get('primary_s') is None:
            cells.append(dict(cid=cid, set=setname, kernel=kernel_label, validation=validation, measured_s=measured, refused=pred.get('unsupported_reason'), kernels=[], pred_primary_s=None, tier=tier, feat=feat, traffic=None, docp=docp, energy_j=energy_j)); return
        for kk, bk, pk in zip(ph['kernels'], bank['kernels'], pred['kernels']):
            meta = secondary.get(kk.get('kernel_id'), feat); occ = meta['occupancy']; active = sm * occ['active_sm_fraction']; clock = K['clock']
            phs = []
            for p, b, t in zip(kk['phases'], bk['phases'], pk['phase_terms']):
                pipes, disp = phase_pipes(p, b, K)
                ops = p['issue_warp_instructions']
                kind = 'compute_tensor' if any(o.startswith('HMMA') for o in ops) else ('load' if any(o.startswith('LDG') for o in ops) else ('store' if any(o.startswith(('STG', 'RED', 'ATOM')) for o in ops) else 'other'))
                phs.append(dict(t=t, pipes_s={k_: v / (active * clock) for k_, v in pipes.items()}, disp_s=disp / (active * clock), kind=kind, depth=p['dependent_global_load_depth'], barriers=sum(n for o, n in ops.items() if o.startswith('BAR')) / max(1, (meta['geometry'].get('grid_blocks') or meta['geometry'].get('grid_blocks_dispatch') or 1) * (occ.get('warps_per_block') or 1)),
                                shared_store_s=0.0))
                issue_chk = max(disp / (active * clock), max(pipes.values(), default=0) / (active * clock))
                assert abs(issue_chk - t['issue_s']) <= 1e-9 * max(1, t['issue_s']) + 1e-15, (cid, issue_chk, t['issue_s'])
            geo = meta['geometry']; blocks = geo.get('grid_blocks') or geo.get('grid_blocks_dispatch')
            resident = min(occ['blocks_per_sm'], math.ceil(blocks / sm)); w = occ.get('warps_per_block') or 1
            ks.append(dict(phases=phs, barrier_s=pk['barrier_s'], launch_s=pk['launch_s'], dispatch_s=pk['dispatch_s'], alpha=pk['overlap_alpha'], primary_s=pk['primary_s'], resident_blocks=resident,
                           warps_per_block=w, resident_warps=resident * w, waves=occ['waves'], tier=feat['memory']['tier'], active=active, clock=clock, K=K,
                           barrier_releases=(meta.get('structure') or {}).get('barrier_releases_per_block_estimate') or 0, grid_blocks=blocks))
        cells.append(dict(cid=cid, set=setname, kernel=kernel_label, validation=validation, measured_s=measured, kernels=ks, pred_primary_s=pred['primary_s'], tier=tier, feat=feat, traffic=pred.get('traffic'), docp=docp, energy_j=energy_j))
    for name, d, ff, pf, uf, bf, cd, docp in SETS_ALL:
        feats = g(d / ff); ph = g(d / pf); ph = ph.get('rows', ph); un = g(d / uf); un = un.get('rows', un); bk = g(bf)['rows']
        C = PP.load_constants(cd); C['smem'] = None
        un_fp = J.attach_footprints(un, EV.load_footprints(name), L2)
        pred = T.predict_candidate(feats, ph, un_fp, bk, C, 188)
        for r in feats['rows']:
            c = r['cell_id']
            if c not in lab: continue
            add(c, name, r, ph[c], un_fp[c], bk[c], pred[c], C, float(lab[c]['runtime_measured_s']), False, lab[c]['kernel'], r['memory']['tier'], docp, float(lab[c]['energy_measured_j']))
    V = ST / 'validation'; valrows = {r['cell_id']: r for r in json.loads((V / 'scores_validation_cells.json').read_text())}
    LIBS = {'f': (SR / 'constants'), 'e': (SR / 'constants'), 'classic': (SR / 'constants'), 'g': CONST, 'h': CONST}
    DOCS = {'f': EV.RUN_DOC, 'e': EV.RUN_DOC, 'classic': EV.RUN_DOC, 'g': EV.TENSOR_DOC, 'h': EV.TENSOR_DOC}
    for lib, cd in LIBS.items():
        t = g(V / 'tables' / ('%s.json' % lib)); ok = {c: v for c, v in t.items() if 'refused' not in v}
        feats = {'hardware_from_ground_truth': g(SR / 'fresh_g/features_fresh_g.json')['hardware_from_ground_truth'], 'rows': [v['features'] for v in ok.values()]}
        ph = {c: v['phases'] for c, v in ok.items()}; un = {c: v['unique'] for c, v in ok.items()}; bk = {c: v['bank'] for c, v in ok.items()}
        un_fp = J.attach_footprints(un, {c: v['footprints'] for c, v in ok.items()}, L2)
        C = PP.load_constants(cd); C['smem'] = None; pred = T.predict_candidate(feats, ph, un_fp, bk, C, 188)
        for r in feats['rows']:
            c = r['cell_id']; vr = valrows[c]
            add(c, 'validation_' + lib, r, ph[c], un_fp[c], bk[c], pred[c], C, vr['rt_meas'], True, vr['kernel'], r['memory']['tier'], DOCS[lib], vr['E'])
    return cells


def compose(cell, mech=None):
    """Runtime of a cell from the stored terms; mech(kernel, phase_index_phase) hooks: mech.phase_s(k, ph) -> serial phase time (default max(stream, latency, issue)),
    mech.full_totals(k) -> dict of whole-kernel totals for the full-overlap leg (default stream, latency, issue sums), mech.extra_s(k) -> added seconds (serial leg and full leg), mech.alpha(k)."""
    tot = 0.0
    for k in cell['kernels']:
        body = 0.0; sums = collections.Counter()
        for p in k['phases']:
            t = p['t']; rep = t['repetitions']
            s = mech.phase_s(k, p) if mech and hasattr(mech, 'phase_s') else max(t['stream_s'], t['latency_s'], t['issue_s'])
            body += s * rep
            if mech and hasattr(mech, 'full_terms'):
                for n, v in mech.full_terms(k, p).items(): sums[n] += v * rep
            else:
                sums['stream'] += t['stream_s'] * rep; sums['latency'] += t['latency_s'] * rep; sums['issue'] += t['issue_s'] * rep
        extra = mech.extra_s(k) if mech and hasattr(mech, 'extra_s') else 0.0
        bar = k['barrier_s']; t0 = k['launch_s']; disp = k['dispatch_s']
        serial = t0 + max(body + bar + extra, disp); full = t0 + max(max(sums.values()) + bar + extra, disp)
        a = mech.alpha(k) if mech and hasattr(mech, 'alpha') else k['alpha']
        tot += (1 - a) * serial + a * full
    return tot


if __name__ == '__main__':
    cells = load_cells(); bad = 0
    for c in cells:
        if c['pred_primary_s'] is None: continue
        v = compose(c)
        if abs(v / c['pred_primary_s'] - 1) > 1e-9: bad += 1; print('MISMATCH', c['cid'], v, c['pred_primary_s'])
    print(len(cells), 'cells,', sum(1 for c in cells if c['pred_primary_s'] is None), 'refused,', bad, 'mismatches of compose() against the model')


_docs = {}
def energy_static(cell, rt):
    """Energy (J) of the component model with traffic columns for a given runtime (the cell's own traffic, unchanged); None if unsupported."""
    if rt is None or cell.get('traffic') is None: return None
    d = _docs.setdefault(cell['docp'], P.load_calibration(cell['docp'], allow_incomplete=True))
    x = T.energy_rows_traffic(d, {'rows': [cell['feat']]}, {cell['cid']: rt}, {cell['cid']: cell['traffic']})[cell['cid']]
    return x['energy_j'] if x['status'] == 'ok' else None
