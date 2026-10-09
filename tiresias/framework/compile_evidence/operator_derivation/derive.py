"""Conservative CPU target-class path derivation; no profiler admission.

Every lane is interpreted; block coordinates remain bounded intervals. Unknown
unpriced predicates poison their outputs. Unknown target predicates/control
refuse the cell. Warp-issued counts are deliberately not inferred from lanes.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, re, math, importlib.util, sys
from dataclasses import dataclass
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
RUNS = BASE / 'acquisition_runs'
CORPORA = {'triton': RUNS/'operator_compile_blackwell_20261001_76cdbada/cpu_recovery/triton', 'cuda': RUNS/'cuda_compile_blackwell_20261001_51afb2f3/output/nvcc'}
class Refusal(ValueError): pass
def require(ok,msg):
    if not ok: raise Refusal(msg)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@dataclass(frozen=True)
class V:
    lo:int; hi:int
    @classmethod
    def exact(cls,n): return cls(n & 0xffffffff,n & 0xffffffff)
    @property
    def exact_value(self): return self.lo if self.lo==self.hi else None
@dataclass(frozen=True)
class Site:
    pc:int; pred:str|None; op:str; a:tuple
LINE=re.compile(r'\s*/\*([0-9a-f]+)\*/\s+(?:@(!?(?:U?P\d+|U?PT))\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;')
def parse(text):
    sites=[]
    for line in text.splitlines():
        if not re.match(r'\s*/\*[0-9a-f]+\*/\s+[A-Z@]',line):continue
        m=LINE.match(line);require(m is not None,'unparsed instruction')
        pc,p,o,a=m.groups();sites.append(Site(int(pc,16),p,o,tuple(x.strip() for x in a.split(','))))
    require(sites and [s.pc for s in sites]==list(range(0,len(sites)*16,16)),'PC gap')
    return sites

def val(t,r):
    t=t.replace('.reuse','');neg=t.startswith('-');t=t[1:] if neg else t
    if t in ('RZ','URZ'):v=V.exact(0)
    elif re.fullmatch(r'U?R\d+',t):v=r.get(t)
    else:
        try:v=V.exact(int(t,0))
        except ValueError:return None
    if neg:return V.exact(-v.lo) if v and v.exact_value is not None else None
    return v

def pred(t,p):
    inv=t.startswith('!');t=t.lstrip('!');v=True if t in ('PT','UPT') else p.get(t)
    return not v if inv and v is not None else v

def calc(vs,fun):
    if any(v is None for v in vs):return None
    if all(v.exact_value is not None for v in vs):return V.exact(fun(*(v.lo for v in vs)))
    # Safe range arithmetic for nonnegative multiplication/addition only.
    lo=fun(*(v.lo for v in vs));hi=fun(*(v.hi for v in vs))
    return V(lo,hi) if 0<=lo<=hi<=0xffffffff else None

def cmp(x,y,rel,unsigned):
    if x is None or y is None:return None
    def signed(v):
        if unsigned:return v
        if v.hi<0x80000000:return v
        if v.lo>=0x80000000:return V(v.lo-2**32,v.hi-2**32)
        return None
    x,y=signed(x),signed(y)
    if x is None or y is None:return None
    if rel=='EQ':return not cmp(x,y,'NE',True) if cmp(x,y,'NE',True) is not None else None
    if rel=='NE':
        if x.hi<y.lo or y.hi<x.lo:return True
        if x.lo==x.hi and y.lo==y.hi:return x.lo!=y.lo
    if rel=='GE':
        if x.lo>=y.hi:return True
        if x.hi<y.lo:return False
    if rel=='GT':
        if x.lo>y.hi:return True
        if x.hi<=y.lo:return False
    if rel=='LT':return cmp(y,x,'GT',True)
    if rel=='LE':return cmp(y,x,'GE',True)
    return None

def target(op):
    if op.startswith('SHFL.'):return 'shuffle',32
    if op.startswith(('LDS.','STS.')) or op in ('LDS','STS'):
        require(op.split('.')[0] in ('LDS','STS'),'unknown shared operation')
        widths=[int(t) for t in op.split('.')[1:] if t.isdigit()]
        require(len(widths)<=1,'ambiguous shared width')
        return ('shared_load' if op.startswith('LDS') else 'shared_store'), widths[0] if widths else 32
    if op=='MUFU.EX2':return 'exponential',32
    if op=='FFMA':return 'fma',32
    if op.startswith('BAR.SYNC'):return 'barrier',None
    return None

KNOWN_OPS=frozenset('BAR.SYNC.DEFER_BLOCKING BRA BRA.DIV BRA.U BSSY BSSY.RECONVERGENT BSYNC BSYNC.RECONVERGENT CS2R DEPBAR.LE ENDCOLLECTIVE EXIT F2I.FTZ.U32.TRUNC.NTZ FADD FFMA FMNMX FMUL FSEL FSETP.GEU.AND FSETP.GT.AND HFMA2 I2F.U32.RP I2FP.F32.S32 IADD IADD.64 IADD3 IMAD IMAD.HI IMAD.HI.U32 IMAD.SHL.U32 IMAD.WIDE IMAD.WIDE.U32 ISETP.GE.AND ISETP.GE.OR ISETP.GE.U32.AND ISETP.GT.AND ISETP.GT.OR ISETP.GT.U32.AND ISETP.LE.AND ISETP.LT.AND ISETP.LT.OR ISETP.LT.U32.AND ISETP.NE.AND ISETP.NE.U32.AND LDC LDC.64 LDCU LDCU.64 LDG.E LDG.E.128 LDG.E.64 LDGDEPBAR LDGSTS.E LDGSTS.E.64 LDGSTS.E.BYPASS.128 LDS LDS.128 LDS.64 LDSM.16.M88 LEA LEA.HI.X LOP3.LUT MOV MOV.64 MUFU.EX2 MUFU.RCP MUFU.SQRT NOP P2R PLOP3.LUT R2UR S2R S2UR SEL SGXT.U32 SHF.L.U32 SHF.L.U64.HI SHF.R.U32.HI SHFL.BFLY SHFL.DOWN STG.E STG.E.128 STG.E.64 STS UFFMA UFMUL UFSEL UFSETP.GEU.AND UFSETP.GT.AND UI2FP.F32.S32 UIADD3 UIADD3.64 UIMAD UIMAD.WIDE UIMAD.WIDE.U32 UISETP.GE.AND UISETP.GE.U32.AND ULEA UMOV USHF.L.U32 USHF.R.S32.HI WARPSYNC.COLLECTIVE'.split())
def run_lane(sites,constants,coords,max_steps=100000):
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
        if o in ('EXIT','BRA','BRA.U'):require(do is not None,'unknown control at '+hex(pc)+' '+str(s.pred))
        if do is False:pc+=16;continue
        if t:events[(s.pc,t[0],t[1])]+=1
        visits[s.pc]+=1
        if o=='EXIT':return events,visits,unknown_unpriced
        if o in ('BRA','BRA.U'):
            take=True if o=='BRA' else pred(a[0],p);require(take is not None,'unknown uniform branch')
            pc=int(a[-1],0) if take else pc+16;continue
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

# Reuse the reviewed retained ELF metadata decoder under a distinct module name.
_spec=importlib.util.spec_from_file_location('operator_retained_abi',BASE/'retained_count/derive.py')
_abi=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_abi;_spec.loader.exec_module(_abi)

def parameter_layout(root,row):
    try:meta=_abi.cubin_parameters(root/row['cubin_path'],row['isolated_function_section'])
    except _abi.Refusal as exc:raise Refusal('cubin ABI: '+str(exc)) from exc
    require(meta['parameter_base']==0x380,'parameter bank drift')
    oid=row['operator_id']
    expected={'final_triton_layer_norm':[(6,48),(7,52)],'dev_triton_softmax':[(2,16),(3,20),(4,24),(5,28)],'train_triton_vector_add':[(3,24)]}
    if oid not in expected:
        expected[oid]=[(2,16),(3,20)] if 'transpose' in oid or oid=='alt_cuda_samples_copy' else [(2,16)] if 'reduction' in oid else [(3,24)]
    require(set(expected[oid])<=set(meta['ordinal_offsets']),'scalar parameter offsets mismatch')
    return meta

CONFIG=BASE.parents[0]/'evaluation_data/measured/cell_manifests/cell_manifest_blackwell.json'
def binding(corpus,row,root):
    oid=row['operator_id'];cell=row['cell'];geom=json.loads(CONFIG.read_text()).get(oid,{}).get('cells',{}).get(cell)
    if geom is None and oid in ('alt_cuda_samples_copy','alt_cuda_samples_transposefine','alt_cuda_samples_reduction'):
        adapters=json.loads((BASE.parents[0]/'adapters/adapters.json').read_text())['adapters'];adapter=next(a for a in adapters if a['parent_id']==oid);regime,candidate=cell.split('/');n=adapter['regimes'][regime]['n']
        if oid=='alt_cuda_samples_reduction':geom={'threads':256,'blocks':64 if candidate=='c3' else n//256,'n':n}
        else:
            edge=n if oid=='alt_cuda_samples_transposefine' else math.isqrt(n);require(oid!='alt_cuda_samples_copy' or edge*edge==n,'copy shape not square');geom={'block_x':32,'block_y':16,'grid_x':edge//32,'grid_y':edge//32}
    const={};coords={};launch={};event=json.loads((root/row['retained_compiler_events'][0]['path']).read_text())
    if corpus=='triton' and oid=='train_triton_vector_add':
        matches=[]
        for ref in row['retained_compiler_events']:
            ev=json.loads((root/ref['path']).read_text());pos=ev['actual_positional_arguments']['items'];kw=ev['actual_keyword_arguments']['items']
            if pos[3].get('value')==geom['n'] and [x['value'] for x in kw['grid']['items']]==row['grid'] and kw['num_warps']['value']==row['num_warps']:matches.append(ev)
        require(len(matches)==1,'missing/ambiguous matching vector-add compilation event');event=matches[0]
    if corpus=='triton':
        ints=[x['value'] if x['type'] in ('int','float') else None for x in event['actual_positional_arguments']['items']]
        kw=event['actual_keyword_arguments']['items'];threads=32*kw['num_warps']['value'];blocks=kw['grid']['items'][0]['value']
        blocks=row['grid'][0];require(row['num_warps']*32==threads,'Triton geometry mismatch')
        require(geom and geom['num_warps']*32==threads,'declared warp mismatch')
        if oid=='final_triton_layer_norm':
            require(ints[6:8]==[geom['cols'],geom['cols']] and blocks==geom['rows'],'layer norm args mismatch');const={0x3b0:ints[6],0x3b4:ints[7]}
        elif oid=='dev_triton_softmax':
            require(ints[2:]==[geom['cols'],geom['cols'],geom['rows'],geom['cols']] and blocks==geom['rows'],'softmax args mismatch');const={0x390:ints[2],0x394:ints[3],0x398:ints[4],0x39c:ints[5],0x370:blocks}
        elif oid=='train_triton_vector_add':
            require(ints[3]==geom['n'],'vector-add scalar drift');const={0x398:ints[3]}
        else:raise Refusal('unknown Triton family')
        launch={'grid':[blocks,1,1],'block':[threads,1,1],'recorded_compile_arguments':ints,'constexpr':kw}
    elif geom and 'threads' in geom:
        threads=geom['threads'];blocks=geom['blocks'];n=geom.get('n') or {'small':65536,'medium':1048576,'large':16777216}[cell.split('/')[0]] if 'reduction' in oid else {'small':1048576,'medium':8388608,'large':67108864}[cell.split('/')[0]]
        const={0x360:threads,0x364:1,0x368:1,0x370:blocks,(0x390 if 'reduction' in oid else 0x398):n};launch={'grid':[blocks,1,1],'block':[threads,1,1],'n':n}
    elif geom and 'block_x' in geom:
        threads=geom['block_x']*geom['block_y'];blocks=geom['grid_x']*geom['grid_y'];const={0x390:geom['grid_x']*32,0x394:geom['grid_y']*32};launch={'grid':[geom['grid_x'],geom['grid_y'],1],'block':[geom['block_x'],geom['block_y'],1]}
    else:raise Refusal('missing exact declared launch adapter')
    const={k:V.exact(v) for k,v in const.items()}
    coords={'SR_CTAID.X':V(0,launch['grid'][0]-1),'SR_CTAID.Y':V(0,launch['grid'][1]-1),'SR_CTAID.Z':V.exact(0),'SR_CgaCtaId':V.exact(0)}
    return const,coords,threads,blocks,launch

def verify(root,row):
    require(row['architecture']=='sm_120','architecture mismatch')
    for key,hkey in [('retained_source_path','source_sha256'),('build_path','build_sha256'),('cubin_path','cubin_sha256'),('disassembly_path','disassembly_sha256'),('full_disassembly_path','full_disassembly_sha256')]:require(sha(root/row[key])==row[hkey],key+' hash mismatch')
    build=json.loads((root/row['build_path']).read_text());require(build['cell_metadata']['operator_id']==row['operator_id'] and build['cell_metadata']['cell']==row['cell'],'build cell mismatch')
    events=[]
    for ref in row['retained_compiler_events']:
        require(sha(root/ref['path'])==ref['sha256'],'event hash mismatch');ev=json.loads((root/ref['path']).read_text());require(ev['cubin_sha256']==row['cubin_sha256'] and ev['status']=='completed','event cubin mismatch')
        artifacts=list(ev.get('asm',{}).values())+ev.get('intermediates',[])
        ptx_records=[]
        for a in artifacts:
            require(sha(root/a['path'])==a['sha256'],'event artifact hash mismatch')
            if a['path'].endswith('.ptx'):
                ptx=(root/a['path']).read_text();require(re.search(r'\.entry\s+'+re.escape(row['isolated_function_section'])+r'\s*\(',ptx),'PTX exact entry missing')
                ptx_records.append({'path':a['path'],'sha256':a['sha256'],'target':re.search(r'\.target\s+([^\n]+)',ptx)[1].strip(),'version':re.search(r'\.version\s+([^\n]+)',ptx)[1].strip()})
        require(ptx_records,'event PTX unavailable')
        events.append({'path':ref['path'],'sha256':ref['sha256'],'actual_argv':ev.get('actual_argv'),'kind':ev['kind'],'ptx':ptx_records})
    return {'retention_manifest_sha256':sha(root/'retention_manifest.json'),'geometry_document_sha256':sha(CONFIG),'alternate_adapter_document_sha256':sha(BASE.parents[0]/'adapters/adapters.json'),'declared_launch_source_hashes':{str(p.relative_to(BASE.parents[2])):sha(p) for p in [BASE.parents[1]/'app_runners/softmax_runner.py',BASE.parents[1]/'app_runners/layernorm_runner.py',BASE.parents[1]/'app_runners/vector_add_runner.py',BASE.parents[1]/'app_runners/reduction_runner.py',BASE.parents[0]/'unseen_operators/runners/tile_family.py',BASE.parents[0]/'unseen_operators/runners/alt_reduction_runner.py']},'abi_decoder_sha256':sha(BASE/'retained_count/derive.py'),'source_sha256':row['source_sha256'],'cubin_sha256':row['cubin_sha256'],'build_sha256':row['build_sha256'],'sass_sha256':row['disassembly_sha256'],'compiler_events':events,'source_closure_verified':False,'actual_dispatch_verified':False,'constant_bank_abi_verified':False,'binding_status':'explicit scalar offsets checked against cubin metadata; implicit geometry ABI and actual driver dispatch remain candidates'}

def derive(corpus,row,root):
    proof=verify(root,row);proof['cubin_parameter_layout']=parameter_layout(root,row);proof['explicit_scalar_parameter_layout_verified']=True;constants,coords,threads,blocks,launch=binding(corpus,row,root);sites=parse((root/row['disassembly_path']).read_text());total=collections.Counter();per=collections.Counter();unknown=collections.Counter()
    for lane in range(threads):
        c={**coords,'SR_TID.X':V.exact(lane%launch['block'][0]),'SR_TID.Y':V.exact(lane//launch['block'][0]),'SR_TID.Z':V.exact(0),'SR_LANEID':V.exact(lane%32)}
        e,v,u=run_lane(sites,constants,c);total.update(e);per.update(v);unknown.update(u)
    summaries={name:{'predicate_true_thread_instruction':0,'warp_issued_instruction':None,'width_bits_histogram':{},'status':'candidate_cpu_path_derivation'} for name in ['shuffle','shared_load','shared_store','exponential','fma','barrier']}
    for (_,name,width),n in total.items():
        q=summaries[name];q['predicate_true_thread_instruction']+=n*blocks
        if width is not None:q['width_bits_histogram'][str(width)]=q['width_bits_histogram'].get(str(width),0)+n*blocks
    missing={}
    for name,prefix in [('shared_matrix_load','LDSM'),('async_global_to_shared','LDGSTS')]:
        if any(s.op.startswith(prefix) for s in sites):missing[name]={'predicate_true_thread_instruction':None,'status':'unsupported_ownership_or_mapping','lexical_sites':sum(s.op.startswith(prefix) for s in sites)}
    return {'status':'candidate_class_counts','classes':summaries,'missing_instruction_families':missing,'constant_binding':{hex(k):[v.lo,v.hi] for k,v in constants.items()},'class_definitions':{'shuffle':'SHFL.* lane invocation','shared_load':'LDS scalar/vector lane invocation; LDSM excluded','shared_store':'STS scalar/vector lane invocation','exponential':'MUFU.EX2 only','fma':'FFMA only; packed/uniform operations excluded','barrier':'BAR.SYNC participating thread invocation; not CTA release count'},'launch':launch,'evidence':proof,'sites':[{'pc':hex(pc),'class':name,'width_bits':width,'predicate_true_thread_instruction':n*blocks} for (pc,name,width),n in sorted(total.items())],'unknown_unpriced_predicate_visits_per_block':dict(unknown),'all_opcode_counts_complete':False,'profiler_validated':False,'scientifically_admitted':False}

def write_features(path,rows):
    fields=['operator_id','cell','status','shuffle_lane_invocations','exponential_lane_invocations','scalar_vector_fma_lane_invocations','shared_load32_lane_invocations','shared_load64_lane_invocations','shared_load128_lane_invocations','shared_store32_lane_invocations','shared_store64_lane_invocations','shared_store128_lane_invocations','barrier_participant_invocations','missing_instruction_families','scientifically_admitted']
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in rows:
            out={'operator_id':r['operator_id'],'cell':r['cell'],'status':r['status'],'missing_instruction_families':';'.join(r.get('missing_instruction_families',{})),'scientifically_admitted':False}
            if r['status']=='candidate_class_counts':
                c=r['classes'];out.update(shuffle_lane_invocations=c['shuffle']['predicate_true_thread_instruction'],exponential_lane_invocations=c['exponential']['predicate_true_thread_instruction'],scalar_vector_fma_lane_invocations=c['fma']['predicate_true_thread_instruction'],barrier_participant_invocations=c['barrier']['predicate_true_thread_instruction'])
                for cls in ['shared_load','shared_store']:
                    for width in [32,64,128]:out[f'{cls}{width}_lane_invocations']=c[cls]['width_bits_histogram'].get(str(width),0)
            w.writerow(out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path(__file__).parent/'candidate_counts.json');args=ap.parse_args();rows=[]
    for corpus,root in CORPORA.items():
        m=json.loads((root/'retention_manifest.json').read_text())
        for row in m['rows']:
            try:result=derive(corpus,row,root)
            except Refusal as e:result={'status':'refused','reason':str(e),'scientifically_admitted':False}
            rows.append({'operator_id':row['operator_id'],'cell':row['cell'],'corpus':corpus,**result})
    summary={}
    for row in rows:
        s=summary.setdefault(row['operator_id'],{'cells':0,'candidate_class_counts':0,'refused':0});s['cells']+=1;s['candidate_class_counts' if row['status']=='candidate_class_counts' else 'refused']+=1
    output={'schema':'operator_candidate_class_counts/1','implementation_sha256':sha(__file__),'note':'CPU derivation candidates only. Full opcode inventory, source closure, actual dispatch and compatible profiler verification remain missing. No model admission.','total_cells':len(rows),'summary':summary,'rows':rows};args.out.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n');write_features(args.out.with_name('candidate_features.csv'),rows);print(json.dumps(summary,indent=2));print(collections.Counter(r.get('reason') for r in rows if r['status']=='refused'))
if __name__=='__main__':main()
