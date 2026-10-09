"""Tables from the ablation run (raw/ablation.jsonl) next to the model's terms. python3 ablation_tables.py > ablation_tables.md
F0 (fixed part present in every variant: launch, epilogue store, one tile load) is estimated from the MODEL (launch + epilogue phase + one load phase); it is used only for the in situ coefficients
(beta, alpha), whose sensitivity to F0 is small for the medium/small cells shown and is not meaningful for the large cells (epilogue write of 128-268 MB overlaps the loop), which are excluded from them."""
import json, collections
from common import *
from mech_lib import *
abl = {}
for l in open(HERE / 'raw/ablation.jsonl'):
    r = json.loads(l); abl.setdefault((r['kernel'], r['cell']), {})[r['variant']] = r['us_median']
cells = {c['cid']: c for c in load_cells()}
def model_parts(c):
    k = c['kernels'][0]; load = comp = epi = 0.0; first_load = None
    for p in k['phases']:
        t = p['t']; v = max(t['stream_s'], t['latency_s'], t['issue_s']) * 1e6
        if p['kind'] == 'load':
            load += v; first_load = first_load if first_load is not None else v
        elif p['kind'] == 'compute_tensor': comp += v
        else: epi += v
    return dict(load=load, comp=comp, epi=epi, first_load=first_load or 0.0, launch=k['launch_s'] * 1e6)
print('## Ablation (median per launch, us; control = unchanged kernel through the same harness)\n')
print('| Kernel cell | resident blocks/SM | original (header) | full copy | no load | no compute | no barriers | V plain (attention) | project harness (eval) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for (kern, cell), a in abl.items():
    cid = 'blackwell/' + ('fresh_g_ml_tensor_matmul/' if kern == 'tc_gemm' else 'fresh_h_ml_fused_attention/') + cell
    m = cells[cid]['measured_s'] * 1e6
    print('| %s %s | %s | %.1f | %.1f | %.1f | %.1f | %.1f | %s | %.1f |' % (kern, cell, {'tc_gemm': {'c1': 2, 'c2': 8}, 'attn_fwd': {'c1': 4, 'c2': 2}}[kern][cell[-2:]], a['original_header'], a['full'], a['noload'], a['nocomp'], a['nobar'], ('%.1f' % a['vplain']) if 'vplain' in a else '-', m))
print('\n## In situ coefficients (medium and small cells)\n')
print('| Kernel cell | warps/SM | compute-only (no load) | LDS-only | HMMA-only | beta in situ = (both - max)/(min) | alpha in situ (load/compute overlap) | model alpha |\n|---|---:|---:|---:|---:|---:|---:|---:|')
for (kern, cell), a in abl.items():
    if cell.startswith('large'): continue
    cid = 'blackwell/' + ('fresh_g_ml_tensor_matmul/' if kern == 'tc_gemm' else 'fresh_h_ml_fused_attention/') + cell; c = cells[cid]; k = c['kernels'][0]; mp = model_parts(c)
    F0 = mp['launch'] + mp['epi'] + mp['first_load']
    if kern == 'tc_gemm':
        l, h, b = a['noload_ldsonly'] - F0, a['noload_hmmaonly'] - F0, a['noload'] - F0; beta = (b - max(l, h)) / min(l, h); lds_s = '%.1f' % a['noload_ldsonly']; hm_s = '%.1f' % a['noload_hmmaonly']
    else:
        beta = None; lds_s = '-'; hm_s = '-'
    Tm = a['nocomp'] - F0; Tc = a['noload'] - F0; T = a['full'] - F0
    alpha = (Tm + Tc - T) / (Tm + Tc - max(Tm, Tc))
    print('| %s %s | %d | %.1f | %s | %s | %s | %.2f | %.2f |' % (kern, cell, k['resident_warps'], a['noload'], lds_s, hm_s, ('%.2f' % beta) if beta is not None else '-', alpha, k['alpha']))
print('\n## Fused attention compute stages (compute phase alone, resident blocks as the original)\n')
print('| Cell | QK only | QK + softmax | P V only | all three (no load) | F0 (model) | QK | softmax | P V | sum of stages | all three | model compute phase | model: pair + rest |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for (kern, cell), a in abl.items():
    if kern != 'attn_fwd' or cell.startswith('large'): continue
    cid = 'blackwell/fresh_h_ml_fused_attention/' + cell; c = cells[cid]; mp = model_parts(c); F0 = mp['launch'] + mp['epi'] + mp['first_load']
    qk, sm, pv, al = a['noload_qkonly'] - F0, a['noload_qksm'] - a['noload_qkonly'], a['noload_pvonly'] - F0, a['noload'] - F0
    from mechanisms import Mech
    k = c['kernels'][0]; m2 = Mech(beta_tc=True, stage=True); m0 = Mech()
    comp_m2 = sum(m2.phase_s(k, p) * p['t']['repetitions'] for p in k['phases'] if p['kind'] == 'compute_tensor') * 1e6
    print('| %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |' % (cell, a['noload_qkonly'], a['noload_qksm'], a['noload_pvonly'], a['noload'], F0, qk, sm, pv, qk + sm + pv, al, mp['comp'], comp_m2))
print('\n## Shared-store conflict check (attention): saving of replacing the 8 STS.U16 (8-way bank conflicts) by one STS.128 per tile row\n')
print('| Cell | full | V plain | measured saving | share of runtime | model prediction of the saving (load phase is stream-bound, issue term hidden) |\n|---|---:|---:|---:|---:|---|')
for (kern, cell), a in abl.items():
    if kern != 'attn_fwd': continue
    print('| %s | %.1f | %.1f | %.1f | %.0f%% | 0 (shared issue below the stream term in every load phase) |' % (cell, a['full'], a['vplain'], a['full'] - a['vplain'], 100 * (a['full'] - a['vplain']) / a['full']))
print('\n## Barrier removal (data race, timing only): saving of dropping both __syncthreads per iteration\n')
print('| Kernel cell | full | no barriers | saving | model barrier term (releases x measured barrier latency) |\n|---|---:|---:|---:|---:|')
for (kern, cell), a in abl.items():
    cid = 'blackwell/' + ('fresh_g_ml_tensor_matmul/' if kern == 'tc_gemm' else 'fresh_h_ml_fused_attention/') + cell; k = cells[cid]['kernels'][0]
    print('| %s %s | %.1f | %.1f | %.1f (%.0f%%) | %.2f us |' % (kern, cell, a['full'], a['nobar'], a['full'] - a['nobar'], 100 * (a['full'] - a['nobar']) / a['full'], k['barrier_s'] * 1e6))
