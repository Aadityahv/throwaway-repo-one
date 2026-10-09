"""Bank-conflict tables for the exposed unseen-kernel cells (32 cells, four cuda-samples kernels). CPU only; reads no measured value.

Uses the unseen pipeline's own binding, interpreter extension and block-class sampling (imported, never edited) with this package's
SharedObserver. Gate: per kernel and phase, the shared request counts per opcode equal the frozen phase table's warp-instruction counts
of the shared opcodes, and the phase count equals the frozen one; representatives of a block class must agree exactly.
"""
import collections, json, math, multiprocessing, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; SR = HERE.parent; UK = SR / 'unseen_kernels'
sys.path.insert(0, str(UK)); sys.path.insert(0, str(HERE))
import unseen_pipeline as UP
import bank_conflicts as BC
import cells as CELLS
P, C, X = UP.P, UP.C, UP.X

def sum_rows(rows):
    out = dict(shared_requests=0, shared_requests_unknown=0, shared_cost_cycles_known=0.0, shared_requests_degree_gt2=0, shared_requests_all_lanes_predicated_off=0)
    by_op, hist, reasons = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rows:
        out['shared_requests'] += r['shared_requests']; out['shared_requests_unknown'] += r['shared_requests_unknown']
        out['shared_cost_cycles_known'] += r['shared_cost_cycles_known_requests_only']; out['shared_requests_degree_gt2'] += r['shared_requests_degree_gt2']
        out['shared_requests_all_lanes_predicated_off'] += r['shared_requests_all_lanes_predicated_off']
        by_op.update(r['shared_requests_by_opcode']); hist.update(r['shared_conflict_degree_histogram']); reasons.update(r['shared_unknown_reasons'])
    out['shared_cost_cycles'] = out['shared_cost_cycles_known'] if not out['shared_requests_unknown'] else None
    out.update(shared_requests_by_opcode=dict(by_op), shared_conflict_degree_histogram=dict(hist), shared_unknown_reasons=dict(reasons))
    return out

def analyse_kernel(kid, kern, frozen_kernel):
    a = UP.kernel_assets(kid); consts, grid, block, _ = UP.binding(kid, kern); threads = math.prod(block); nblocks = math.prod(grid)
    classes = UP.class_list(grid) if kern.get('sampling') == 'classes' else [dict(name='all', weight=nblocks, reps=UP.sample_blocks(grid, kern))]
    total = None
    for cl in classes:
        sigs, per = [], []
        for b in cl['reps']:
            obs = BC.SharedObserver(threads); obs.block_x = block[0]
            interp = UP.INTERP(C.D, ext=True, fchk_fast_path=True, trace=obs.pytorch)
            def coords(l, b=b):
                return {k: P.V.exact(v) for k, v in {'SR_CTAID.X': b % grid[0], 'SR_CTAID.Y': (b // grid[0]) % grid[1], 'SR_CTAID.Z': b // (grid[0] * grid[1]), 'SR_CgaCtaId': 0,
                    'SR_TID.X': l % block[0], 'SR_TID.Y': (l // block[0]) % block[1], 'SR_TID.Z': l // (block[0] * block[1]), 'SR_LANEID': l % 32}.items()}
            interp.run_block(a['sites'], {k: P.V.exact(v) for k, v in consts.items()}, coords, threads)
            n = len(obs.finish(nblocks)['phases']); sigs.append(BC.scale_free_signature(obs, n)); per.append(BC.shared_phases(obs, cl['weight'], n))
        if any(s != sigs[0] for s in sigs[1:]): raise C.Refusal('class %s: representatives have different shared-memory signatures' % cl['name'])
        per0 = per[0]
        total = [[r] for r in per0] if total is None else [t + [r] for t, r in zip(total, per0)]
    phases = [sum_rows(rows) for rows in total]
    for i, (mine, fz) in enumerate(zip(phases, frozen_kernel['phases'])):
        want = {op: n for op, n in fz['issue_warp_instructions'].items() if BC.shared_info(op) is not None and BC.shared_info(op)['kind'] != 'async_write'}
        if mine['shared_requests_by_opcode'] != want: raise C.Refusal('%s phase %d: shared request counts differ from the frozen warp-instruction counts (%s vs %s)' % (kid, i, mine['shared_requests_by_opcode'], want))
    if len(phases) != len(frozen_kernel['phases']): raise C.Refusal('%s: phase count differs from the frozen table' % kid)
    return dict(kernel_id=kid, phases=[dict(index=i, shared=p) for i, p in enumerate(phases)], gate='passed: shared request counts equal frozen shared warp counts per phase')

def one(args):
    cell, frozen = args
    try:
        return cell['cell_id'], dict(kernels=[analyse_kernel(k['kid'], k, fk) for k, fk in zip(cell['kernels'], frozen['kernels'])])
    except (C.Refusal, P.Refusal) as ex:
        return cell['cell_id'], dict(status='refused', reason=str(ex))

def main():
    hw = X.load_hardware(X.read_text(X.GROUND_TRUTH)); frozen = json.loads((UK / 'frozen/phases_unseen.json').read_text())['rows']
    cells = [c for c in CELLS.define_cells(hw) if c['cell_id'] in frozen]
    rows = {}
    with multiprocessing.get_context('fork').Pool(6) as pool:
        for cid, row in pool.imap_unordered(one, [(c, frozen[c['cell_id']]) for c in cells]):
            rows[cid] = row; print(cid, row.get('status', 'gated ok'), str(row.get('reason', ''))[:150], flush=True)
    (HERE / 'bank_conflicts_unseen.json').write_text(json.dumps(dict(set='unseen_exposed', schema='bank_conflicts/1', rows=rows), indent=1, sort_keys=True) + '\n')
if __name__ == '__main__': main()
