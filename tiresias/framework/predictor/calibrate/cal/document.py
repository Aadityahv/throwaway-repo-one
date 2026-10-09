"""Calibration document: build, validate completeness, and export the legacy constants files the existing predictors read."""
import json
from pathlib import Path

from . import SCHEMA, TOOL_VERSION

REQUIRED_CONSTANTS = ('launch_reuse', 'mlp', 'stream_curves', 'smem', 'overlap', 'store_legacy', 'pipes', 'chase_latency_ns', 'energy', 'tensor')


def build(device, entry, ground_truth, booking, toolchain, stages, constants, warnings, created_utc, complete, store_derivation=None):
    doc = dict(schema=SCHEMA, tool_version=TOOL_VERSION, created_utc=created_utc, complete=complete, device=device, approved_entry=entry, ground_truth_checked=ground_truth,
                booking_ref=booking, toolchain=toolchain, stages=stages, constants=constants, warnings=warnings,
                note='All constants come from synthetic microbenchmarks (timing and energy windows); no operator kernel, application runtime or application energy value was read. Failed or missing stages are listed in `stages` and make `complete` false.')
    if store_derivation is not None: doc['store_derivation'] = store_derivation  # method, identification diagnostics of the traffic-rate (`store_legacy`) derivation
    return doc


def validate(doc):
    missing = [k for k in REQUIRED_CONSTANTS if k not in doc['constants']]
    if missing: return False, 'missing constants: ' + ', '.join(missing)
    return True, ''


def _pipes_with_tensor(c):
    """The pipe constants of the runtime model, plus the instruction class `tensor_mma` when the tensor stage measured it (a class that is absent makes the runtime model report a tensor-core kernel unsupported)."""
    import copy
    pipes = copy.deepcopy(c['pipes']); iss = (c.get('tensor') or {}).get('issue') or {}
    if iss.get('issue_cycles_per_warp_instruction_per_sm'): pipes['issue_cycles_per_warp_instruction_per_sm']['tensor_mma'] = iss['issue_cycles_per_warp_instruction_per_sm']
    if iss.get('dependent_latency_cycles'): pipes['dependent_latency_cycles']['tensor_mma'] = iss['dependent_latency_cycles']
    return pipes


def export_legacy(doc, out_dir):
    """Write the constants files in the shapes read by predict_runtime_v2/v3/v3f/v3h/v3i. Returns the list of files."""
    c = doc['constants']; out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    w = lambda name, obj: (out / name).write_text(json.dumps(obj, indent=1, sort_keys=True) + '\n') or name
    lr = c['launch_reuse']
    files = [
        w('stream_constants.json', {'constants': c['store_legacy'], 'source': 'calibration document %s' % doc['created_utc']}),
        w('microbench_constants_v2.json', _pipes_with_tensor(c)),
        w('v3_constants.json', dict(l1_reread_l2_fraction_curve=lr['l1_reread_l2_fraction_curve'], launch_us_per_kernel=lr['launch_us_per_kernel'], l1=lr['l1'], composition_rule=lr['composition_rule'])),
        w('v3c_constants.json', c['mlp']),
        w('v3e_constants.json', c['stream_curves']),
        w('overlap_constants.json', c['overlap']),
        w('smem_constants.json', c['smem']),
    ]
    return files
