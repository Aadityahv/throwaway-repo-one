"""Compiler-backed exact lane counts for the structured synthetic calibration loop.

Refuses extra active backedges, predicated body work, unknown guards, differing
block traces or nonlinear visits. Does not read timing, energy or profiler data.
"""
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SR = HERE.parents[1]
sys.path.insert(0, str(SR / 'port_common'))
import port_ext as PX
PX.install('sm_120')
UP, P, A, C = PX.UP, PX.P, PX.A, PX.C
X = UP.X


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source_mix(row):
    d = row['dose']; name = row['mix']
    return dict(memory=(d,0,0,0),arithmetic=(1,d,0,0),special_function=(1,0,d,0),
                other_shared=(1,0,0,d),mixed=(1,8,8,8),mixed_altered=(1,24,4,12))[name]


def constants(trips, grid, block):
    lanes = grid[0] * block[0]
    out = {o:v for o,v in zip(A.IMPLICIT.values(), block + grid)}
    values = dict(input=C.PTR_BASE0,out=C.PTR_BASE0+C.PTR_STRIDE,n=lanes*trips)
    fields = A.layout([('input','ptr'),('out','ptr'),('n','i32')])
    for name, off, size in fields:
        out[off]=values[name]&0xffffffff
        if size == 8: out[off+4]=(values[name]>>32)&0xffffffff
    return out


def trace(sites, trips, block_index, grid, block):
    interp = UP.INTERP(C.D,ext=True,fchk_fast_path=False)
    # CUDA 13.2 lowers explicit ld.global.cg to a GPU-scope strong global
    # load. Existing LDG interpretation (unknown payload, exact instruction
    # visit) applies; cache/order modifiers do not change lane work or ABI.
    interp.KNOWN_OPS.add('LDG.E.STRONG.GPU')
    # New sink uses this conversion only as data, never control or addressing.
    # Existing unknown-value handling applies; exact visits are counted.
    interp.KNOWN_OPS.add('F2I.U32.NTZ')
    cm = {k:P.V.exact(v) for k,v in constants(trips,grid,block).items()}
    coords = lambda lane:{'SR_CTAID.X':P.V.exact(block_index),'SR_CTAID.Y':P.V.exact(0),
        'SR_CTAID.Z':P.V.exact(0),'SR_CgaCtaId':P.V.exact(0),'SR_TID.X':P.V.exact(lane),
        'SR_TID.Y':P.V.exact(0),'SR_TID.Z':P.V.exact(0),'SR_LANEID':P.V.exact(lane%32)}
    res = interp.run_block(sites,cm,coords,block[0],max_steps=100000)
    if interp.unknown_pcs or interp.assumption_log:
        raise ValueError('REFUSED: unknown predicates or interpreter assumptions')
    visits = [v for _,v,_ in res]
    if any(v != visits[0] for v in visits[1:]):
        raise ValueError('REFUSED: calibration lanes do not have uniform exact visits')
    return visits[0]


def certify(sites, grid, block):
    observations = {}
    for trips in (1,2,3,4):
        samples = [trace(sites,trips,b,grid,block) for b in (0,grid[0]//2,grid[0]-1)]
        if any(v != samples[0] for v in samples[1:]):
            raise ValueError('REFUSED: block-dependent calibration count')
        observations[trips]=samples[0]
    active = set(observations[4])
    back = [s for s in sites if s.pc in active and s.op in ('BRA','BRA.U') and int(s.a[-1],0)<s.pc]
    if len(back)!=1:
        raise ValueError('REFUSED: require one active structured backedge')
    branch=back[0];start=int(branch.a[-1],0);stop=branch.pc
    # This is the induction certificate, beyond checking four small traces:
    # straight-line unpredicated body and one backedge; no data-dependent body
    # work/control. The source/ABI contract binds index=lane+k*launch_width,
    # n=trip_count*launch_width, so every lane executes exactly trip_count passes.
    for s in sites:
        if start<=s.pc<stop:
            if s.pred is not None or s.op.startswith(('BRA','EXIT','CALL','RET','JMP')):
                raise ValueError('REFUSED: predicated/branching work inside loop body')
    slope = collections.Counter(); intercept = collections.Counter()
    for pc in active:
        a=observations[2][pc]-observations[1][pc];b=observations[1][pc]-a
        if a not in (0,1) or b not in (-1,0,1):
            raise ValueError('REFUSED: unexpected instruction multiplicity')
        if any(observations[k][pc] != a*k+b for k in (1,2,3,4)):
            raise ValueError('REFUSED: nonlinear loop visits')
        if start<=pc<stop and (a,b)!=(1,0):
            raise ValueError('REFUSED: loop body does not execute exactly once per trip')
        if (pc<start or pc>stop) and a!=0:
            raise ValueError('REFUSED: loop-dependent work outside certified body')
        slope[pc]=a;intercept[pc]=b
    return slope,intercept,dict(loop_start_pc=hex(start),backedge_pc=hex(stop),
        source_induction='index=lane+k*(188*256); n=(188*256)*trips; 0<=lane<188*256; all lanes execute trips iterations',
        audited_trips=[1,2,3,4],audited_blocks=[0,grid[0]//2,grid[0]-1],audited_lanes=block[0],
        conditional_pointer_binding='distinct 4-GiB-aligned synthetic pointer bases, as in retained static interpreter',
        pc_slope={hex(k):v for k,v in slope.items()},pc_intercept={hex(k):v for k,v in intercept.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet',required=True,type=Path);p.add_argument('--binary',required=True,type=Path)
    p.add_argument('--sass',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();packet=json.loads(a.packet.read_text());txt=a.sass.read_text()
    sections=re.split(r'\n\s*Function : ',txt)[1:];cache={};rows=[];certificates={}
    for row in packet['rows']:
        m=source_mix(row);needle='energy_fixture'+''.join(('ILi' if i==0 else 'Li')+str(v)+'E' for i,v in enumerate(m))+'E'
        hits=[s for s in sections if needle in s.splitlines()[0]]
        if len(hits)!=1:raise ValueError(f'REFUSED: kernel template isolation {m}: {len(hits)} matches')
        if m not in cache:
            text='Function : '+hits[0];sites=P.parse(text)
            kid='energy_calibration';A.PARAM_FIELDS[kid]=[('input','ptr'),('out','ptr'),('n','i32')]
            ok,problems=A.abi_check(kid,sites)
            if not ok:raise ValueError('REFUSED: ABI '+str(problems))
            slope,intercept,proof=certify(sites,row['grid'],row['block'])
            proof['isolated_sass_sha256']=hashlib.sha256(text.encode()).hexdigest()
            cache[m]=(sites,slope,intercept,proof)
        sites,slope,intercept,proof=cache[m]
        lanes=row['grid'][0]*row['block'][0]
        if row['n']%lanes:raise ValueError('REFUSED: partial calibration iteration')
        trips=row['n']//lanes;visits=collections.Counter({pc:slope[pc]*trips+intercept[pc] for pc in slope})
        work,_=X.work_features(sites,[visits]*row['block'][0],set(),row['block'][0],row['grid'][0])
        if not work['all_counts_exact']:raise ValueError('REFUSED: inexact compiled count')
        families=work['families']
        got=lambda name:families.get(name,{}).get('lane_instructions',0)
        if got('fp32_fma')!=row['n']*m[0]*m[1] or got('special_function')!=row['n']*m[0]*m[2]:
            raise ValueError('REFUSED: native arithmetic/SFU dose differs from source design')
        if work['executed_global_load_bytes']!=row['n']*m[0]*4 or work['executed_global_store_bytes']!=lanes*12:
            raise ValueError('REFUSED: compiled global traffic differs from logical activity convention')
        proof_sha=hashlib.sha256(json.dumps(proof,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        certificates[proof_sha]=proof
        rows.append(dict(row,work=work,count_status='exact',abi_status='verified',
            source_sha256=sha(HERE/'calibration.cu'),binary_sha256=sha(a.binary),count_evidence_sha256=proof_sha))
        print(row['design_id'],work['total_lane_instructions'],flush=True)
    out=dict(schema='energy_component_calibration_design/1',status='compiled_counts_frozen_correctness_pending',
        cap_w=packet['cap_w'],gpu_index=1,gpu_uuid=packet['gpu_uuid'],rows=rows,certificates=certificates,
        qualification='Exact under declared source induction, straight-line compiled CFG and synthetic-pointer binding. No physical cache-traffic claim.',
        inputs_sha256={str(f):sha(f) for f in [a.packet,a.binary,a.sass,Path(__file__),HERE/'calibration.cu',HERE/'oracle.hpp',
            SR/'pytorch_features/pt_interp.py',SR/'pytorch_features/adapter.py',SR/'port_common/port_ext.py',SR/'extract_features.py']})
    with a.out.open('x') as f:json.dump(out,f,indent=2,sort_keys=True);f.write('\n')


if __name__=='__main__':main()
