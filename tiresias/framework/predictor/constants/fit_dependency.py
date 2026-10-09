"""Fit the dependent-instruction cost from retained calibration paths, CPU only.

Only the two null scaffolds fit the nonnegative cost. Other latency-bound
instruction doses and mixed-fill designs are residual checks, never fit inputs.
Correctness-event timing is retained as an unadmitted transfer assumption.
"""
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import phases as F
R = F.C._load('runtime_dependency_revised', F.X.MI / 'revised_count/derive.py')


def nonnegative_cost(rows):
    numerator = sum(r['compute_depth'] * (r['measured_s'] - r['fixed_s']) for r in rows)
    denominator = sum(r['compute_depth'] ** 2 for r in rows)
    if not denominator:
        raise ValueError('dependency cost is not identifiable: zero chain depth')
    return max(0.0, numerator / denominator)


def main():
    packet_path = R.ARCHIVE / 'prepared_packet.json'
    sass_path = R.ARCHIVE / 'output/full_sass.stdout'
    packet = json.loads(packet_path.read_text())
    inventory = R.functions(sass_path.read_text())
    source = inspect.getsource(R.helper.execute)
    needle = '        visits[(pc,execute_instruction)]+=1\n'
    if source.count(needle) != 1:
        raise ValueError('calibration trace insertion point changed')
    source = source.replace(needle, needle + '        observer(s, execute_instruction)\n')
    env = dict(R.helper.execute.__globals__)
    exec(compile(source, '<calibration-read-only-observer>', 'exec'), env)
    constants = json.loads((HERE / 'stream_constants.json').read_text())['constants']
    rows = []
    hashes = {str(p.relative_to(F.X.REPO)): F.sha(p) for p in [Path(__file__), packet_path, sass_path,
        Path(R.__file__), Path(R.helper.__file__), HERE / 'stream_constants.json', HERE.parent / 'phases.py',
        HERE.parent / 'extract_features.py']}
    for row in packet['correctness_rows']:
        if 'mask' not in row['controls']:
            continue
        symbol, bindings, coords, threads, blocks, module, fixture = R.bind(row, inventory)
        obs = F.Observer(1)
        def observer(s, guard):
            # Transaction addresses are irrelevant to the dependency fit.
            t = SimpleNamespace(pc=s.pc, op=s.op, a=s.args, pred=s.pred)
            obs.event(0, t, guard)
        env['observer'] = observer
        path = env['execute'](inventory[symbol], bindings, coords)
        if dict(R.helper.counts(inventory[symbol], path, threads, blocks)['reached_warp_issues_by_full_opcode']) != next(
            c['counts']['reached_warp_issues_by_full_opcode'] for c in json.loads((F.X.MI / 'revised_count/compiled_candidates.json').read_text())['cells'] if c['slot'] == row['slot']):
            raise ValueError('instrumented calibration path changed')
        ld = sum(v[1] + obs.completed_chains[k][1] for k, v in obs.chains.items())
        cp = sum(v[0] + obs.completed_chains[k][0] for k, v in obs.chains.items())
        native_path = R.ARCHIVE / 'output/correctness' / row['slot'].replace('/', '_') / 'native_result.json'
        native = json.loads(native_path.read_text())
        hashes[str(native_path.relative_to(F.X.REPO))] = F.sha(native_path)
        # These calibration launches have <= one block per SM and one wave.
        fixed = constants['t0_us'] * 1e-6 + ld * constants['latency_ns'][row['tier'].upper()] * 1e-9
        rows.append(dict(slot=row['slot'], tier=row['tier'], mask=row['controls']['mask'],
            compute_depth=cp, load_depth=ld, measured_s=native['runtime_s_per_launch'], fixed_s=fixed,
            role='fit_null_scaffold' if row['controls']['mask'] == 0 else 'residual_check',
            timing_transfer_admitted=native['timing_transfer_admitted']))
    cost = nonnegative_cost([r for r in rows if r['role'] == 'fit_null_scaffold'])
    for row in rows:
        row['predicted_s'] = row['fixed_s'] + row['compute_depth'] * cost
        row['signed_residual_pct'] = 100 * (row['predicted_s'] / row['measured_s'] - 1)
    result = dict(c_dep_s_per_instruction=cost, c_dep_cycles=cost * 2.617e9,
        fit='Nonnegative least squares on both null scaffolds after fixed launch and global-load latency.',
        timing_transfer_admitted=False,
        limitation='Correctness timing includes host gaps; this fit does not admit it as an inference timing reference. Register chains omit shared-memory and cross-lane edges.',
        input_sha256=hashes, rows=rows)
    (HERE / 'dependency_constants.json').write_text(json.dumps(result, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(c_dep_cycles=result['c_dep_cycles'], rows=len(rows), fit_rows=2)))


if __name__ == '__main__': main()
