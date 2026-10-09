"""Portable wrapper of the current runtime model (bank-conflict cost + partial phase overlap): the SM count and the shared-memory cost come from the calibration, not from
literals. The existing predictors are imported unchanged (their hashes are recorded in frozen prediction files); the two places that hard-code 188 SMs
(`mlp_seconds` in predict_runtime_v3 and predict_runtime_v3f) are replaced for the duration of a call, and the shared-memory cost of each phase is recomputed from the stored
request-wavefront histograms with the calibrated floor and slope (cost per request = max(floor, slope * wavefronts); with floor 2.0 and slope 1.0 this is exactly the
cost stored in the bank tables)."""
import copy
import json
import math
import sys
from pathlib import Path

SR = Path(__file__).resolve().parents[2]
if str(SR) not in sys.path: sys.path.insert(0, str(SR))
import predict_runtime_v2 as V2  # noqa: E402
import predict_runtime_v3 as V3  # noqa: E402
import predict_runtime_v3f as V3F  # noqa: E402
import predict_runtime_v3h as V3H  # noqa: E402
import predict_runtime_v3i as V3I  # noqa: E402


def load_constants(constants_dir):
    d = Path(constants_dir); rd = lambda n: json.loads((d / n).read_text())
    return dict(stream=rd('stream_constants.json')['constants'], micro=rd('microbench_constants_v2.json'), v3=rd('v3_constants.json'), v3c=rd('v3c_constants.json'),
                v3e=rd('v3e_constants.json'), overlap=rd('overlap_constants.json'), smem=rd('smem_constants.json') if (d / 'smem_constants.json').exists() else None)


def recost_bank(bank_rows, floor, slope):
    """Copy of the bank tables with every phase's shared_cost_cycles recomputed from its wavefront histogram."""
    out = copy.deepcopy(bank_rows)
    for row in out.values():
        for k in row.get('kernels', []):
            for ph in k['phases']:
                s = ph['shared']
                if s.get('shared_requests_unknown'): continue
                s['shared_cost_cycles'] = float(sum(n * max(floor, slope * int(w)) for w, n in s['shared_request_wavefront_histogram'].items()))
    return out


_classify_original = V2.classify


def _classify_with_tensor(op):
    """The runtime model's instruction classes plus the tensor class: HMMA/IMMA/QMMA/DMMA issue on the tensor pipe (`tensor_mma`, measured by the optional tensor stage). The frozen
    predictor files are not edited; without a measured `tensor_mma` cost the model raises KeyError for such a kernel and reports it unsupported (no silent integer-class cost)."""
    if op.startswith(('HMMA', 'IMMA', 'QMMA', 'DMMA')): return 'tensor_mma', 1.0
    return _classify_original(op)


def predict(features, phase_rows, unique_rows, bank_rows, constants, sm_count):
    """Predictions of the current model for every cell of a set; `constants` from load_constants()."""
    K = V2.make_constants(constants['stream'], constants['micro'])
    if constants['smem']:
        bank_rows = recost_bank(bank_rows, constants['smem']['floor_cycles'], constants['smem']['degree_slope_cycles'])
    def mlp_seconds(*args, **kw):
        # identical to the originals but with the device's SM count
        return _mlp(sm_count, *args, **kw)
    orig3, orig3f = V3.mlp_seconds, V3F.mlp_seconds
    V2.classify = _classify_with_tensor
    V3.mlp_seconds = lambda ph, occ, blocks, tier, V3C, active: _mlp_v3(sm_count, ph, occ, blocks, tier, V3C, active)
    V3F.mlp_seconds = lambda served_bytes, ph, occ, blocks, tier, V3C, active: _mlp_v3f(sm_count, served_bytes, ph, occ, blocks, tier, V3C, active)
    try:
        return V3I.build(features, phase_rows, unique_rows, bank_rows, K, constants['v3'], constants['v3c'], constants['v3e'], constants['overlap'])
    finally:
        V3.mlp_seconds, V3F.mlp_seconds = orig3, orig3f
        V2.classify = _classify_original


def _mlp_v3(sm, ph, occ, blocks, tier, V3C, active):
    requests = sum(n for op, n in ph['issue_warp_instructions'].items() if op.startswith('LDG'))
    if not requests or not ph['read_sectors']: return 0.0
    warps_block = occ.get('warps_per_block') or 1
    resident_warps = min(occ['blocks_per_sm'], math.ceil(blocks / sm)) * warps_block
    bytes_per_request = ph['read_sectors'] * 32 / requests
    loads_per_warp = requests / max(1, blocks * warps_block)
    mlp = max(1.0, loads_per_warp / max(1, ph['dependent_global_load_depth']))
    inflight = resident_warps * bytes_per_request * mlp
    lam = V3C['service_latency_us'][tier] * 1e-6
    return (ph['read_sectors'] * 32) / (active * inflight / lam)


def _mlp_v3f(sm, served_bytes, ph, occ, blocks, tier, V3C, active):
    requests = sum(n for op, n in ph['issue_warp_instructions'].items() if op.startswith('LDG'))
    if not requests or not ph['read_sectors'] or not served_bytes: return 0.0
    warps_block = occ.get('warps_per_block') or 1
    resident_warps = min(occ['blocks_per_sm'], math.ceil(blocks / sm)) * warps_block
    bytes_per_request = ph['read_sectors'] * 32 / requests
    mlp = max(1.0, requests / max(1, blocks * warps_block) / max(1, ph['dependent_global_load_depth']))
    return served_bytes / (active * resident_warps * bytes_per_request * mlp / (V3C['service_latency_us'][tier] * 1e-6))
