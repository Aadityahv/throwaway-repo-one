"""Label-free predictions for the frozen phase-wise static runtime model.

Reads only static features/phases and microbenchmark constants. No operator
runtime, energy, profiler metric or fitted operator coefficient is opened.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def issue_cost(op):
    base = op.split('.')[0]
    if base == 'BAR': return 0.0
    if base == 'MUFU' or base.startswith(('F2I', 'I2F', 'I2FP', 'UI2FP')): return 2.0
    if base == 'SHFL': return 1.0
    if base in ('LDS', 'STS'):
        width = next((int(x) for x in op.split('.')[1:] if x.isdigit()), 32)
        return width / 32
    if base in ('FADD', 'FMUL', 'FFMA', 'UFADD', 'UFMUL', 'UFFMA'): return 0.25
    if base.startswith(('IADD', 'UIADD', 'LOP', 'ULOP', 'ISETP', 'UISETP', 'FSETP', 'UFSETP',
                        'IMNMX', 'FMNMX', 'SHF', 'USHF')): return 0.5
    return 0.25


def kernel_terms(phases, occupancy, tier, constants, dep_s, sm_count, clock, logical_scale=1.0):
    if not occupancy.get('blocks_per_sm') or not occupancy.get('waves'):
        raise ValueError('static occupancy or waves unavailable')
    active = sm_count * occupancy['active_sm_fraction']
    micro_sms = constants['active_sms_microbench']
    scale = micro_sms / active
    terms = []
    for ph in phases:
        required = ['read_sectors', 'write_sectors', 'lines', 'read_bytes', 'write_bytes',
                    'dependent_global_load_depth', 'critical_path_compute_instructions']
        if any(ph.get(k) is None for k in required):
            raise ValueError('unknown phase transaction or dependency input')
        l2 = (ph['read_sectors'] * 32 / (constants['L2_read_sector_TBps'] * 1e12)
            + ph['write_sectors'] * 32 / (constants['L2_write_sector_TBps'] * 1e12)
            + ph['lines'] * constants['c_line_ns'] * 1e-9) * scale
        dram = ((ph['read_bytes'] * logical_scale / (constants['DRAM_read_TBps'] * 1e12)
            + ph['write_bytes'] * logical_scale / (constants['DRAM_write_TBps'] * 1e12)) * scale) if tier == 'DRAM' else 0.0
        latency = occupancy['waves'] * (ph['dependent_global_load_depth'] * constants['latency_ns'][tier] * 1e-9
            + ph['critical_path_compute_instructions'] * dep_s)
        issue = sum(n * issue_cost(op) for op, n in ph['issue_warp_instructions'].items()) / (active * clock)
        terms.append(dict(stream_s=max(l2, dram), l2_s=l2, dram_s=dram, latency_s=latency,
            issue_s=issue, repetitions=ph['repetitions']))
    total = {k: sum(p[k] * p['repetitions'] for p in terms) for k in ['stream_s', 'latency_s', 'issue_s']}
    t0 = constants['t0_us'] * 1e-6
    return dict(primary_s=t0 + sum(max(p['stream_s'], p['latency_s'], p['issue_s']) * p['repetitions'] for p in terms),
        max_s=t0 + max(total.values()), sum_s=t0 + sum(total.values()), phase_terms=terms)


def build(features, phase_rows, constants, dep_s, clock):
    out = {}
    for f in features['rows']:
        cid = f['cell_id']; ph = phase_rows[cid]; predictions = []
        try:
            if ph['status'] == 'unsupported': raise ValueError(ph['reason'])
            tier = f['memory']['tier']
            secondary = {k['kernel_id']: k for k in f.get('secondary_kernels', [])}
            executed_bytes = sum(p['read_bytes'] + p['write_bytes'] for k in ph['kernels'] for p in k['phases'])
            if not executed_bytes: raise ValueError('no static bytes to partition the logical-byte input')
            # The frozen plan uses the development-table logical total. Allocate it
            # across phases/directions in proportion to statically executed bytes.
            logical_scale = f['memory']['logical_bytes_per_launch'] / executed_bytes
            for k in ph['kernels']:
                meta = secondary.get(k.get('kernel_id'), f)
                predictions.append(kernel_terms(k['phases'], meta['occupancy'], tier, constants, dep_s,
                    features['hardware_from_ground_truth']['sm_count'], clock, logical_scale))
            if not predictions: raise ValueError('no static kernel phases')
            out[cid] = {key: sum(k[key] for k in predictions) for key in ['primary_s', 'max_s', 'sum_s']}
            out[cid].update(unsupported_reason=None, kernels=predictions, logical_byte_allocation_scale=logical_scale)
        except (ValueError, KeyError) as ex:
            out[cid] = dict(primary_s=None, max_s=None, sum_s=None, unsupported_reason=str(ex))
    return out


def main():
    features = json.loads((HERE / 'features_blackwell.json').read_text())
    phases = json.loads((HERE / 'phases_blackwell.json').read_text())['rows']
    constants = json.loads((HERE / 'constants/stream_constants.json').read_text())['constants']
    dep = json.loads((HERE / 'constants/dependency_constants.json').read_text())['c_dep_s_per_instruction']
    # Ground truth is authoritative; the plan uses its instantaneous boost reading.
    ground = (HERE.parents[2] / 'HARDWARE_GROUND_TRUTH.md').read_text()
    section = ground.split('## Blackwell', 1)[1].split('\n## ', 1)[0]
    if '| SM clock (at time of probe) | 2,617,000 kHz' not in section:
        raise ValueError('preregistered clock differs from current hardware ground truth')
    output = build(features, phases, constants, dep, 2.617e9)
    (HERE / 'predictions_blackwell.json').write_text(json.dumps(output, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(cells=len(output), supported=sum(v['primary_s'] is not None for v in output.values()))))


if __name__ == '__main__': main()
