"""Collective-reduction extension of the CPU operator derivation (BRA.DIV).

derive.py is untouched. run_lane_div below is a copy of derive.run_lane whose only
semantic change is the BRA.DIV handling (see the marked block). Rationale:

* Lanes are enumerated individually, so BSSY/BSYNC reconvergence needs no explicit
  stack: every lane follows its own path and each priced site is counted per lane.
* BRA.DIV Ux, target branches to the warp-collective fallback only when the warp is
  not converged on mask Ux at that instruction. The interpreter takes the fall-through
  (converged fast path) and records an arrival (pc, mask) for the lane. The cell-level
  check then requires, for every warp, mask == 0xffffffff, all 32 lanes present and
  every lane arriving the same number of times at that site; otherwise the whole cell
  is refused. The fallback block (WARPSYNC.COLLECTIVE .. ENDCOLLECTIVE) is therefore
  never entered, and still refuses if some lane reaches it.
* A BRA.DIV mask that is not an exact value, or an unknown predicate on BRA.DIV,
  refuses. Every other rule (unknown priced predicate/control refuses, poison, no
  zero-fill) is inherited verbatim from derive.py.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, sys
from pathlib import Path
import derive as D
from derive import (Refusal, require, sha, V, parse, val, pred, calc, cmp, target, KNOWN_OPS, CORPORA, BASE,
                    verify, parameter_layout, binding)
import re, math

def run_lane_div(sites,constants,coords,max_steps=100000,arrivals=None):
    r={};p={};events=collections.Counter();visits=collections.Counter();unknown_unpriced=collections.Counter();by={s.pc:s for s in sites};pc=0
    for step in range(max_steps):
        require(pc in by,'control outside function');s=by[pc];a=s.a;o=s.op;t=target(o)
        require(o in KNOWN_OPS,'unsupported opcode '+o)
        # Validate every declared output shape before predication. A skipped or
        # unknown operation must not leave an unmodelled carry/predicate alive.
        if o in ('IADD3','UIADD3'):require(a[1:3] in (('PT','PT'),('UPT','UPT')),'unsupported add carry outputs')
        if o.startswith(('ISETP.','UISETP.')) or o=='PLOP3.LUT':require(a[1] in ('PT','UPT'),'unsupported second predicate output')
        if o in ('LOP3.LUT','ULOP3.LUT'):
            require(a[-1] in ('!PT','!UPT'),'unsupported LOP3 predicate combiner')
            if a[0].startswith(('P','UP')):require(a[5] in ('0xc0','0xfc'),'unsupported LOP3 predicate LUT')
        do=pred(s.pred,p) if s.pred else True
        if t:require(do is not None,'unknown target predicate at '+hex(pc))
        if o in ('EXIT','BRA','BRA.U','BRA.DIV'):require(do is not None,'unknown control at '+hex(pc)+' '+str(s.pred))
        if do is False:pc+=16;continue
        if t:events[(s.pc,t[0],t[1])]+=1
        visits[s.pc]+=1
        if o=='EXIT':return events,visits,unknown_unpriced
        if o in ('BRA','BRA.U'):
            take=True if o=='BRA' else pred(a[0],p);require(take is not None,'unknown uniform branch')
            pc=int(a[-1],0) if take else pc+16;continue
        # --- collective extension (the only semantic change versus derive.run_lane) ---
        if o=='BRA.DIV':
            mask=val(a[0],r);require(mask is not None and mask.exact_value is not None,'unknown BRA.DIV mask at '+hex(pc))
            require(len(a)==2 and arrivals is not None,'BRA.DIV needs arrival recording')
            arrivals[(pc,mask.lo)]+=1;pc+=16;continue   # converged fast path; verified per warp by the caller
        # --- end extension ---
        require(not o.startswith(('CALL','RET','JMP','BRX','WARPSYNC','BRA.')),'unsupported control '+o)
        # Unmodelled synchronization addresses do not modify scalar state.
        if o.startswith(('STG','STS','BAR','BSSY','BSYNC','DEPBAR','LDGDEPBAR','LDGSTS','NOP')):pc+=16;continue
        dest=a[1] if o.startswith('SHFL') or (o=='LOP3.LUT' and a[0].startswith(('P','UP'))) else a[0] if a else None;result=None
        if do is None:
            unknown_unpriced[o]+=1
            if o=='LOP3.LUT' and a[0].startswith(('P','UP')):p[a[0]]=None
        elif o.startswith(('LDC','LDCU')):
            m=re.fullmatch(r'c\[0x0\]\[(0x[0-9a-f]+)\]',a[1]);off=int(m[1],0) if m else None
            result=constants.get(off)
            if '.64' in o and re.fullmatch(r'U?R\d+',dest):r[dest[0:-len(re.search(r'\d+$',dest)[0])]+str(int(re.search(r'\d+$',dest)[0])+1)]=constants.get(off+4) if off else None
        elif o in ('S2R','S2UR'):result=coords.get(a[1])
        elif o=='CS2R' and a[1]=='SRZ':result=V.exact(0)
        elif o=='HFMA2' and a[1:3]==('-RZ','RZ') and a[3:]==('0','0'):result=V.exact(0)
        elif o=='P2R' and a[1:3]==('PR','RZ'):
            mask=int(a[3],0);bits=[i for i in range(7) if mask>>i&1]
            if all(p.get('P'+str(i)) is not None for i in bits):result=V.exact(sum(int(p['P'+str(i)])<<i for i in bits))
        elif o.startswith(('MOV','UMOV')):result=val(a[1],r)
        elif o in ('IMAD','UIMAD','IMAD.SHL.U32'):result=calc([val(x,r) for x in a[1:4]],lambda x,y,z:x*y+z)
        elif o=='IADD':result=calc([val(x,r) for x in a[1:3]],lambda x,y:x+y)
        elif o in ('IADD3','UIADD3'):
            require(a[1:3] in (('PT','PT'),('UPT','UPT')),'unsupported add carry outputs');result=calc([val(x,r) for x in a[3:6]],lambda x,y,z:x+y+z)
        elif o in ('LEA','ULEA') and len(a)==4:
            result=calc([val(a[1],r),val(a[2],r),val(a[3],r)],lambda x,y,z:(x<<z)+y)
        elif o.startswith(('ISETP.','UISETP.')):
            require(a[1] in ('PT','UPT'),'unsupported second predicate output');rel=o.split('.')[1];c=cmp(val(a[2],r),val(a[3],r),rel,'U32' in o);q=pred(a[4],p)
            comb=o.split('.')[-1]
            out=(False if c is False or q is False else True if c is True and q is True else None) if comb=='AND' else (True if c is True or q is True else False if c is False and q is False else None) if comb=='OR' else None
            p[a[0]]=out if do is True else None
            pc+=16;continue
        elif o in ('LOP3.LUT','ULOP3.LUT'):
            has_pred=a[0].startswith(('P','UP'));start=2 if has_pred else 1
            require(a[-1] in ('!PT','!UPT'),'unsupported LOP3 predicate combiner')
            if has_pred:require(a[start+3] in ('0xc0','0xfc'),'unsupported LOP3 predicate LUT')
            vs=[val(x,r) for x in a[start:start+3]];lut=int(a[start+3],0)
            if all(v and v.exact_value is not None for v in vs):
                x,y,z=[v.lo for v in vs];n=0
                for bit in range(32):n|=((lut>>(((x>>bit&1)<<2)|((y>>bit&1)<<1)|(z>>bit&1)))&1)<<bit
                result=V.exact(n)
            if has_pred:p[a[0]]=bool(result.lo) if result else None;dest=a[1]
        elif o=='PLOP3.LUT':
            require(a[1] in ('PT','UPT'),'unsupported second predicate output');vs=[pred(x,p) for x in a[2:5]]
            if all(v is not None for v in vs):idx=(int(vs[0])<<2)|(int(vs[1])<<1)|int(vs[2]);p[a[0]]=bool(int(a[5],0)>>idx&1)
            else:p[a[0]]=None
            pc+=16;continue
        elif o in ('SHF.L.U32','USHF.L.U32'):
            vs=[val(x,r) for x in a[1:4]]
            if all(v and v.exact_value is not None for v in vs) and vs[2].lo==0:result=V.exact(vs[0].lo<<(vs[1].lo&31))
        elif o=='SHF.R.U32.HI':
            vs=[val(x,r) for x in a[1:4]]
            require(vs[0] == V.exact(0),'unsupported nonzero-low funnel shift')
            if all(v and v.exact_value is not None for v in vs):result=V.exact(((vs[2].lo<<32)|vs[0].lo)>>(vs[1].lo&31)>>32)
            # .HI returns the upper word after shifting the (hi,lo) pair.
        elif o=='SEL':
            q=pred(a[3],p);x,y=val(a[1],r),val(a[2],r);result=x if q is True else y if q is False else x if x==y else None
        elif o.startswith('SHFL'):
            if a[0] not in ('PT','UPT'):p[a[0]]=None
        elif o.startswith(('FSETP','P2R','R2P')):
            if a and a[0].startswith(('P','UP')):p[a[0]]=None
        if dest and re.fullmatch(r'U?R\d+',dest):
            r[dest]=result
            width=4 if '.128' in o else 2 if '.64' in o or '.WIDE' in o or o=='CS2R' else 1
            if not o.startswith(('LDC','LDCU')) or do is None:
                pref='UR' if dest.startswith('UR') else 'R';idx=int(dest[len(pref):])
                for j in range(1,width):r[pref+str(idx+j)]=V.exact(0) if o=='CS2R' and a[1]=='SRZ' else None
        elif dest and re.fullmatch(r'U?P\d+',dest):p[dest]=None
        if o.startswith(('LEA','IMAD','IADD3')):
            for output in a[:2]:
                if re.fullmatch(r'U?P\d+',output):p[output]=None
        pc+=16
    raise Refusal('step limit')


WARP=32
CLASSES=['shuffle','shared_load','shared_store','exponential','fma','barrier']
def check_warp_convergence(arrivals_by_lane,threads):
    """Every warp that reaches a BRA.DIV site must arrive as one full converged warp on mask 0xffffffff."""
    for w in range(math.ceil(threads/WARP)):
        lanes=range(w*WARP,min(threads,(w+1)*WARP));keys=set()
        for l in lanes:keys|=set(arrivals_by_lane[l])
        for key in keys:
            require(key[1]==0xffffffff,'BRA.DIV mask %#x is not the full-warp mask at %s'%(key[1],hex(key[0])))
            require(len(lanes)==WARP,'partial warp %d reaches BRA.DIV at %s'%(w,hex(key[0])))
            counts={arrivals_by_lane[l].get(key,0) for l in lanes}
            require(len(counts)==1 and 0 not in counts,'warp %d diverges at BRA.DIV %s (per-lane arrivals %s)'%(w,hex(key[0]),sorted(counts)))

def derive_collective(corpus,row,root):
    proof=verify(root,row);proof['cubin_parameter_layout']=parameter_layout(root,row);proof['explicit_scalar_parameter_layout_verified']=True
    constants,coords,threads,blocks,launch=binding(corpus,row,root);sites=parse((root/row['disassembly_path']).read_text())
    total=collections.Counter();per=collections.Counter();unknown=collections.Counter();arrivals_by_lane=[]
    for lane in range(threads):
        c={**coords,'SR_TID.X':V.exact(lane%launch['block'][0]),'SR_TID.Y':V.exact(lane//launch['block'][0]),'SR_TID.Z':V.exact(0),'SR_LANEID':V.exact(lane%32)}
        arr=collections.Counter();e,v,u=run_lane_div(sites,constants,c,arrivals=arr);total.update(e);per.update(v);unknown.update(u);arrivals_by_lane.append(arr)
    check_warp_convergence(arrivals_by_lane,threads)
    op_at={s.pc:s.op for s in sites}
    summaries={name:{'predicate_true_thread_instruction':0,'warp_issued_instruction':None,'width_bits_histogram':{},'status':'candidate_cpu_path_derivation'} for name in CLASSES}
    for (_,name,width),n in total.items():
        q=summaries[name];q['predicate_true_thread_instruction']+=n*blocks
        if width is not None:q['width_bits_histogram'][str(width)]=q['width_bits_histogram'].get(str(width),0)+n*blocks
    missing={}
    for name,prefix in [('shared_matrix_load','LDSM'),('async_global_to_shared','LDGSTS')]:
        if any(s.op.startswith(prefix) for s in sites):missing[name]={'predicate_true_thread_instruction':None,'status':'unsupported_ownership_or_mapping','lexical_sites':sum(s.op.startswith(prefix) for s in sites)}
    shuffle_modes=collections.Counter()
    for (pc,name,_),n in total.items():
        if name=='shuffle':shuffle_modes[op_at[pc]]+=n*blocks
    div_sites=sorted({k for arr in arrivals_by_lane for k in arr})
    return {'status':'candidate_class_counts','classes':summaries,'missing_instruction_families':missing,'constant_binding':{hex(k):[v.lo,v.hi] for k,v in constants.items()},'launch':launch,'evidence':proof,
            'sites':[{'pc':hex(pc),'op':op_at[pc],'class':name,'width_bits':width,'predicate_true_thread_instruction':n*blocks} for (pc,name,width),n in sorted(total.items())],
            'shuffle_modes_lane_invocations':dict(shuffle_modes),
            'divergence_model':{'bra_div_sites':[{'pc':hex(pc),'mask':hex(m)} for pc,m in div_sites],'treatment':'converged fast path; each warp verified to arrive with all 32 lanes, equal per-lane arrival count, full mask','fallback_block_entered':False},
            'unknown_unpriced_predicate_visits_per_block':dict(unknown),'all_opcode_counts_complete':False,'profiler_validated':False,'scientifically_admitted':False}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path(__file__).parent/'collective_counts.json');args=ap.parse_args();rows=[];base=json.loads((Path(__file__).parent/'candidate_counts.json').read_text())
    refused={(r['operator_id'],r['cell']) for r in base['rows'] if r['status']!='candidate_class_counts'}
    for corpus,root in CORPORA.items():
        for row in json.loads((root/'retention_manifest.json').read_text())['rows']:
            if (row['operator_id'],row['cell']) not in refused:continue
            try:res=derive_collective(corpus,row,root)
            except Refusal as e:res={'status':'refused','reason':str(e),'scientifically_admitted':False}
            rows.append({'operator_id':row['operator_id'],'cell':row['cell'],'corpus':corpus,**res})
    out={'schema':'operator_collective_candidate_class_counts/1','derive_py_sha256':sha(Path(D.__file__)),'collective_py_sha256':sha(__file__),
         'candidate_counts_json_sha256':sha(Path(__file__).parent/'candidate_counts.json'),
         'note':'Frozen BEFORE any profiler report for these cells was opened. CPU derivation candidates only; no model admission.',
         'total_cells':len(rows),'rows':rows}
    args.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(collections.Counter(r['status'] for r in rows),[r.get('reason') for r in rows if r['status']=='refused'])
if __name__=='__main__':main()
