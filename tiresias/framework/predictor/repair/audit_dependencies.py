"""Audit every retained opcode and all 25 calibration paths, CPU only.

No operator runtime/energy label or profiler report is read. Calibration timings
are reported only to diagnose identifiability; their transfer flag stays false.
"""
import collections
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import phases as F
from dependencies import decode, Graph, Refusal


def compare(s):
    try:
        roles = decode(s)
        old_defs, old_uses = F.X.sass_def_use(s)
        delta = dict(missing_definitions=sorted(roles.definitions - old_defs),
                     spurious_definitions=sorted(old_defs - roles.definitions),
                     missing_uses=sorted(roles.uses - old_uses), spurious_uses=sorted(old_uses - roles.uses))
        return dict(status='decoded', opcode=s.op, pc=hex(s.pc), operands=list(s.a),
                    changed=any(delta.values()), differences=delta, unresolved_edges=list(roles.unresolved))
    except Refusal as ex:
        return dict(status='refused', opcode=s.op, pc=hex(s.pc), reason=str(ex))


def inventory():
    out, hashes = [], {}
    for corpus, root in F.C.D.CORPORA.items():
        manifest_path = root / 'retention_manifest.json'
        hashes[str(manifest_path.relative_to(F.X.REPO))] = F.sha(manifest_path)
        for row in json.loads(manifest_path.read_text())['rows']:
            F.C.D.verify(root, row)
            path = root / row['disassembly_path']
            hashes[str(path.relative_to(F.X.REPO))] = F.sha(path)
            sites = F.C.D.parse(path.read_text())
            checks = [compare(s) for s in sites]
            out.append(dict(cell_id='blackwell/' + row['operator_id'] + '/' + row['cell'],
                static_sites=len(checks), changed_sites=sum(c.get('changed', False) for c in checks),
                refused_sites=[c for c in checks if c['status']=='refused'],
                differences=[c for c in checks if c.get('changed')],
                unresolved_edges=sorted({e for c in checks for e in c.get('unresolved_edges', [])})))
    dispatch_path = F.HERE / 'pytorch_dispatch/dispatch_trace.json'
    hashes[str(dispatch_path.relative_to(F.X.REPO))] = F.sha(dispatch_path)
    cache = {}
    for cell in json.loads(dispatch_path.read_text())['cells']:
        checks = []
        for kernel in cell['kernels']:
            kid = F.A.kernel_id(kernel)
            if kid not in cache:
                path = F.HERE / 'libtorch_sm120' / (kid + '.isolated.sass')
                hashes[str(path.relative_to(F.X.REPO))] = F.sha(path)
                cache[kid] = [compare(s) for s in F.P.parse(path.read_text())]
            checks.extend(cache[kid])
        out.append(dict(cell_id=cell['cell_id'], static_sites=len(checks),
            changed_sites=sum(c.get('changed', False) for c in checks),
            refused_sites=[c for c in checks if c['status']=='refused'],
            differences=[c for c in checks if c.get('changed')],
            unresolved_edges=sorted({e for c in checks for e in c.get('unresolved_edges', [])})))
    return out, hashes


def calibration():
    R = F.C._load('repair_calibration_paths', F.X.MI / 'revised_count/derive.py')
    packet_path = R.ARCHIVE / 'prepared_packet.json'
    sass_path = R.ARCHIVE / 'output/full_sass.stdout'
    candidates_path = F.X.MI / 'revised_count/compiled_candidates.json'
    packet = json.loads(packet_path.read_text())
    candidates = {c['slot']: c for c in json.loads(candidates_path.read_text())['cells']}
    funcs = R.functions(sass_path.read_text())
    source = inspect.getsource(R.helper.execute)
    needle = '        visits[(pc,execute_instruction)]+=1\n'
    if source.count(needle) != 1: raise Refusal('calibration observer insertion point changed')
    env = dict(R.helper.execute.__globals__)
    exec(compile(source.replace(needle, needle+'        observer(s, execute_instruction)\n'),
                 '<calibration-producer-audit>', 'exec'), env)
    hashes = {str(p.relative_to(F.X.REPO)): F.sha(p) for p in [packet_path, sass_path,
        candidates_path, Path(R.__file__), Path(R.helper.__file__)]}
    result = []
    for row in packet['correctness_rows']:
        graph = Graph()
        symbol, bindings, coords, threads, blocks, module, fixture = R.bind(row, funcs)
        def observer(s, guard):
            graph.add(SimpleNamespace(pc=s.pc, op=s.op, a=s.args, pred=s.pred), guard=guard)
        env['observer'] = observer
        path = env['execute'](funcs[symbol], bindings, coords)
        if R.helper.counts(funcs[symbol], path, threads, blocks)['reached_warp_issues_by_full_opcode'] != candidates[row['slot']]['counts']['reached_warp_issues_by_full_opcode']:
            raise Refusal('audited calibration path differs from frozen counts')
        native_path = R.ARCHIVE / 'output/correctness' / row['slot'].replace('/', '_') / 'native_result.json'
        native = json.loads(native_path.read_text())
        hashes[str(native_path.relative_to(F.X.REPO))] = F.sha(native_path)
        result.append(dict(slot=row['slot'], tier=row['tier'], role=row['role'], controls=row['controls'],
            path_counts_unchanged=True, register_graph=graph.summary(),
            runtime_s_per_launch=native['runtime_s_per_launch'],
            timing_definition=native['timing_definition'], timing_transfer_admitted=native['timing_transfer_admitted']))
    contrasts = []
    for tier in ('l2','dram'):
        null = next(r for r in result if r['tier']==tier and r['controls'].get('mask')==0)
        for mask, name in [(1,'warp shuffle'),(2,'shared load'),(4,'shared store'),(8,'exponential')]:
            doses = {r['controls']['k']:r for r in result if r['tier']==tier and r['controls'].get('mask')==mask}
            if set(doses) != {4,16}: raise Refusal('incomplete calibration dose pair')
            low, high = doses[4], doses[16]
            delta = high['runtime_s_per_launch']-low['runtime_s_per_launch']
            contrasts.append(dict(tier=tier, instruction_family=name,
                low_runtime_us=low['runtime_s_per_launch']*1e6, high_runtime_us=high['runtime_s_per_launch']*1e6,
                dose_runtime_change_pct=delta/low['runtime_s_per_launch']*100,
                high_vs_null_runtime_change_pct=(high['runtime_s_per_launch']/null['runtime_s_per_launch']-1)*100,
                standalone_compute_latency_identified=False))
    return result, contrasts, hashes


def main():
    rows, hashes = inventory()
    cal, contrasts, h = calibration(); hashes.update(h)
    nulls=[r for r in cal if r['controls'].get('mask')==0]
    if len(nulls)!=2: raise Refusal('unexpected null-design population')
    signatures=[[r['register_graph']['register_path_instructions'],
                 r['register_graph']['register_path_global_loads']] for r in nulls]
    rank=2 if signatures[0][0]*signatures[1][1]!=signatures[0][1]*signatures[1][0] else int(any(any(r) for r in signatures))
    null_check=dict(register_summary_columns=['path_instructions','path_global_loads'],
        register_summary_rows=signatures,register_summary_rank=rank,
        interpretation='Structural summaries only; not an admitted linear timing design or physical compute latency.',
        runtime_gap_us=abs(nulls[0]['runtime_s_per_launch']-nulls[1]['runtime_s_per_launch'])*1e6,
        standalone_compute_latency_identified=False)
    for p in [Path(__file__), HERE/'dependencies.py', F.HERE/'extract_features.py',
              F.HERE/'phases.py', F.HERE/'pytorch_features/pt_interp.py']:
        hashes[str(p.relative_to(F.X.REPO))] = F.sha(p)
    output = dict(schema='static_dependency_repair_audit/1', operator_binary_cells=len(rows),
        missing_binary_cells=12, changed_operand_role_cells=sum(r['changed_sites']>0 for r in rows),
        decoder_refused_cells=sum(bool(r['refused_sites']) for r in rows),
        rows=rows, calibration_rows=cal, calibration_dose_contrasts=contrasts,
        null_design_identifiability=null_check,
        source_sha256=hashes, compute_constants_admitted=False, operator_runtime_labels_read=False,
        new_operator_predictions=False,
        limitations=['Static inventory includes unreachable sites; differences are decoder findings, not automatically timing changes.',
            'Calibration graph is register-only over the uniform proven path; shared, peer-lane, control/scheduler and synchronization edges are not fully bound.',
            'No physical instruction latency or operator prediction is supplied.'])
    (HERE/'dependency_audit.json').write_text(json.dumps(output, indent=1, sort_keys=True)+'\n')
    print(json.dumps({k:output[k] for k in ['operator_binary_cells','changed_operand_role_cells','decoder_refused_cells']}))


if __name__=='__main__': main()
