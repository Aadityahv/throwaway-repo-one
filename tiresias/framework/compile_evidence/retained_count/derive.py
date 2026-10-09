"""CPU-only, hash-bound SASS path derivation for the retained calibration packet.

No GPU/profiler/reference or scientific admission. The abstract interpreter
derives branch decisions from compiled arithmetic and recorded launch inputs.
It never accepts supplied visits/masks. Unknown control refuses the entire cell.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import struct

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ARCHIVE = BASE / "acquisition_runs/blackwell_20260930_0cbc61ec/acquisition_attempt"
PIN_BINARY = "c655840200f033ba9dc074e8f5a6d015e27951819ec3f716e51a4201c32abdb5"
PIN_PTX = {
    "alternative_bridge": "57d949e955341fcd45a783e906e083987dd685ed937a4f5a9014cdd4d9401ae1",
    "store_bridge": "66687ef6a8a83be881dd33f8df4b58b097f70f3c8247256442f970e2c770c846",
    "floor_bridge": "9a0b38fb665be9f722e0785f45c5e8740089bd32fa6b76b2bcab5385f7097983",
}
# These are hashes of real complete function text, frozen after inspecting the
# retained dump. They stop the deliberately narrow semantics from generalizing
# to arbitrary instructions or a future compiler's code.
PIN_SASS = "4c569f59117fdab399fe197e60a87c20b3007c3732ea005b344373c2c4ea8aa6"
PIN_ARCHIVE = "71686cbba096e6fc64e3028cac854262d63c99743dded5e4c8c91c5f5856e794"
PIN_SOURCE = {
    'calibration_alternative/primitive.cu':'1ec02acac30249f9ee1ba741256d7ac144e4711b64f1f3942e3b0868b58566a1',
    'graph_calibration/fixture_api.hpp':'1ce0e0d68270d65010a6fc0ed8dcb4f37af46dc6b0b0bc5e48716e5946c506f2',
    'graph_calibration/store_bridge.cu':'3c304893a1a2c46cba7f83df6344c887b584d6cf7bfa81bb9abc4026abd77d51',
    'graph_calibration/floor_bridge.cu':'d6863111a4106b94c2478a7cacb5fe72241ecd9dc3a6e74caeace567cc6ef510',
    'graph_calibration/pinned_sources/predictor_store.cu':'aac0fc23836798ad0cb1a28fbea1eb64e86cb184b9bbc336d23b80757e2e5b09',
    'graph_calibration/pinned_sources/e2e_tile_triad_calibration.cu':'acb59dee00f8a51ed21c2f99657cbe9d284bbb619e20b060c7988fb380ab2ad0',
}
KNOWN_OPS = frozenset('BRA BRA.U CS2R EXIT FADD FFMA FMUL HFMA2 I2FP.F32.U32 IADD IADD.64 IMAD IMAD.SHL.U32 IMAD.WIDE.U32 ISETP.GE.U32.AND ISETP.GE.U64.AND ISETP.NE.U32.AND LDC LDC.U8 LDCU LDCU.128 LDCU.64 LDG.E LDG.E.STRONG.GPU LDS LEA LEA.HI.X LOP3.LUT MOV MUFU.EX2 NOP PRMT S2R S2UR SEL SHF.L.U64.HI SHF.R.U32.HI SHFL.BFLY STG.E STG.E.STRONG.SYS STS UIADD3 UIMAD UIMAD.WIDE.U32 UISETP.NE.U32.AND ULEA ULOP3.LUT UMOV VIMNMX.U32'.split())
TARGETS = ("SHFL.BFLY", "LDS", "STS", "MUFU.EX2")
class Refusal(ValueError): pass

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok, message):
    if not ok: raise Refusal(message)

@dataclass(frozen=True)
class Range:
    lo: int
    hi: int
    @classmethod
    def exact(cls, n): return cls(n & 0xffffffff, n & 0xffffffff)
    @property
    def singleton(self): return self.lo == self.hi

@dataclass(frozen=True)
class Site:
    pc: int
    pred: str | None
    op: str
    args: tuple[str, ...]
    text: str

def parse_sass(text):
    functions = {}
    for section in text.split("Function : ")[1:]:
        name = section.splitlines()[0].strip()
        require(name not in functions, "duplicate function")
        require('EF_CUDA_SM120' in section, "wrong architecture")
        sites = []
        for line in section.splitlines():
            if not re.match(r"\s*/\*[0-9a-f]+\*/", line): continue
            m = re.match(r"\s*/\*([0-9a-f]+)\*/\s+(?:@(!?(?:P|UP)\d+)\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;", line)
            require(m is not None, "dropped instruction: " + line)
            pc, pred, op, operands = m.groups()
            sites.append(Site(int(pc,16), pred, op, tuple(x.strip() for x in operands.split(',')) if operands else (), line.split(';')[0].strip()))
        require(sites and sites[0].pc == 0, "missing entry")
        require([s.pc for s in sites] == list(range(0, len(sites)*16,16)), "PC gap/duplicate")
        functions[name] = sites
    require(len(functions) == 9, "incomplete retained function inventory")
    return functions

def val(token, regs):
    token=token.replace('.reuse','')
    negative=token.startswith('-'); token=token[1:] if negative else token
    if token in ('RZ','URZ'): result=Range.exact(0)
    elif re.fullmatch(r'(?:UR|R)\d+',token): result=regs.get(token)
    else:
        try: result=Range.exact(int(token,0))
        except ValueError: return None
    if negative:
        return Range.exact(-result.lo) if result and result.singleton else None
    return result

def arithmetic(args, regs, kind):
    values=[val(x,regs) for x in args]
    if any(x is None for x in values): return None
    if all(x.singleton for x in values):
        n=[x.lo for x in values]
        return Range.exact(sum(n) if kind=='add' else n[0]*n[1]+n[2])
    # Only nonnegative, no-overflow interval arithmetic is supported. No modular
    # wrapped interval can prove a branch.
    lo=sum(x.lo for x in values) if kind=='add' else values[0].lo*values[1].lo+values[2].lo
    hi=sum(x.hi for x in values) if kind=='add' else values[0].hi*values[1].hi+values[2].hi
    return Range(lo,hi) if hi <= 0xffffffff else None

def compare(a,b,relation):
    if a is None or b is None: return None
    if relation=='NE':
        if a.hi < b.lo or b.hi < a.lo: return True
        if a.singleton and b.singleton: return a.lo != b.lo
    elif relation=='GE':
        if a.lo >= b.hi: return True
        if a.hi < b.lo: return False
    return None

def execute(sites, constants, coordinates, max_steps=200000):
    """One uniform path over all supplied block/thread coordinate intervals.

    Exact control values prove loop trips. If an interval predicate overlaps,
    control is unknown and refuses. Full warps and uniform predicates are the
    only supported case; memory data is unknown and cannot decide control.
    """
    regs={}; predicates={}; visits=Counter(); taken=Counter(); not_taken=Counter()
    by_pc={x.pc:x for x in sites}; pc=0
    for _ in range(max_steps):
        require(pc in by_pc, "branch outside retained function")
        s=by_pc[pc]; a=s.args; op=s.op
        require(op in KNOWN_OPS, 'unsupported opcode: ' + op)
        execute_instruction=True
        if s.pred:
            p=predicates.get(s.pred.lstrip('!'))
            require(p is not None,"unknown/divergent instruction predicate at " + hex(pc))
            execute_instruction=not p if s.pred.startswith('!') else p
        # Issued-site visits are separate from predicate-true events. All reached
        # sites issue per warp even when an instruction predicate is false.
        visits[(pc,execute_instruction)]+=1
        if not execute_instruction:
            if op in ('EXIT','BRA'): not_taken[pc]+=1
            pc+=16; continue
        if op=='EXIT':
            return {'visits':visits,'branch_taken':taken,'branch_not_taken':not_taken}
        if op in ('CALL','RET','JMP','BRX') or op.startswith(('CALL.','RET.','JMP.','BRX.')):
            raise Refusal('unsupported indirect/call control')
        if op in ('BRA','BRA.U'):
            if op=='BRA.U':
                require(len(a)==2, 'bad uniform branch')
                p=predicates.get(a[0].lstrip('!'))
                require(p is not None, 'unknown uniform branch at ' + hex(pc))
                take=not p if a[0].startswith('!') else p
                dest=int(a[1],0)
            else: take=True; dest=int(a[0],0)
            (taken if take else not_taken)[pc]+=1
            pc=dest if take else pc+16; continue
        # All opcodes are recognized by the exact-artifact lock. Unknown value
        # semantics invalidate outputs. Never reuse stale control registers.
        output=a[1] if op=='SHFL.BFLY' else a[0] if a else None
        if op=='SHFL.BFLY' and a[0] != 'PT': predicates[a[0]]=None
        result=None
        if op in ('LDC','LDCU','LDC.U8'):
            m=re.fullmatch(r'c\[0x0\]\[(0x[0-9a-f]+)\]',a[1])
            result=Range.exact(constants[int(m[1],0)]) if m and int(m[1],0) in constants else None
        elif op in ('S2R','S2UR'): result=coordinates.get(a[1])
        elif op in ('UMOV','MOV'): result=val(a[1],regs)
        elif op=='UIADD3': result=arithmetic(a[3:6],regs,'add')
        elif op in ('IMAD','UIMAD'): result=arithmetic(a[1:4],regs,'mad')
        elif op=='IADD': result=arithmetic(a[1:3],regs,'add')
        elif op in ('ISETP.NE.U32.AND','UISETP.NE.U32.AND','ISETP.GE.U32.AND'):
            require(a[1] in ('PT','UPT') and a[4] in ('PT','UPT'), 'unsupported predicate combination')
            predicates[a[0]]=compare(val(a[2],regs),val(a[3],regs),op.split('.')[1]);
            pc+=16; continue
        elif op=='ULOP3.LUT':
            # Both retained control masks are AND with zero third operand.
            x,y,z=map(lambda t:val(t,regs),a[1:4])
            if x and y and z and x.singleton and y.singleton and z.lo==z.hi==0 and a[4:] == ('0xc0','!UPT'):
                result=Range.exact(x.lo & y.lo)
        elif op=='VIMNMX.U32':
            # The branch needs only a containing range; hull contains either
            # min or max for every lane and avoids relying on selector choice.
            x,y=val(a[1],regs),val(a[2],regs)
            if x and y: result=Range(min(x.lo,y.lo),max(x.hi,y.hi))
        if output and re.fullmatch(r'(?:UR|R)\d+',output):
            regs[output]=result
            # Invalidate other halves of wider loads/arithmetic destinations.
            width=4 if '.128' in op else 2 if '.64' in op or '.WIDE' in op or op=='CS2R' else 1
            prefix='UR' if output.startswith('UR') else 'R'
            for shift in range(1,width): regs[prefix+str(int(output[len(prefix):])+shift)]=None
        elif output and re.fullmatch(r'(?:P|UP)\d+',output): predicates[output]=None
        # Carry predicates are written by address arithmetic and SHFL. They
        # cannot retain a previous known value if read by later control.
        if op.startswith(('LEA','IADD3','IMAD.X','LOP3')):
            for t in a[:2]:
                if re.fullmatch(r'(?:P|UP)\d+',t): predicates[t]=None
        pc+=16
    raise Refusal('nontermination/bounded-step limit')

def argv_values(command):
    args=command['argv']; require(len(args)%2==1, 'unpaired command option')
    require(args[0]=='/home/user/ipdps_model_20260930_0cbc61ec/acquisition_attempt/calibration_blackwell','wrong recorded executable')
    options=dict(zip(args[1::2],args[2::2]))
    require(len(options)==(len(args)-1)//2,'duplicate command option')
    require(options.get('--mode')=='correctness', 'not correctness evidence')
    require(options.get('--expected-uuid')=='GPU-0e63baea-0bb1-e90c-f19d-8aa5f2995894','wrong UUID')
    require(options.get('--graph-batch')=='1000' and options.get('--graph-replays')=='2','wrong replay regime')
    return options

def binding(options, functions):
    fixture=options['--fixture']; tier=options['--tier']
    n=lambda key:int(options['--'+key])
    if fixture=='alternative':
        mask,k=n('mask'),n('k')
        require((mask,k) in ((0,0),(1,16),(2,16),(4,16),(8,16),(15,4)), 'unsupported specialization')
        require([n(x) for x in ('blocks','threads','iterations','stride','step-warps','tile-elements')] == [94,1024,100,33,8,131072 if tier=='l2' else 524288], 'alternative shape drift')
        symbol=f'_Z18alternative_kernelILj{mask}ELj{k}EEvPKjPjjjjj'
        constants={0x360:1024,0x370:94,0x390:n('tile-elements'),0x394:100,0x398:33,0x39c:8}
        coords={'SR_TID.X':Range(0,1023),'SR_CTAID.X':Range(0,93)}
        threads,blocks=1024,94; module='alternative_bridge'
    elif fixture=='store':
        require([n(x) for x in ('blocks','threads','iterations','stride','step-warps','tile-elements','batches')] == [188,256,16 if tier=='l2' else 4096,1,1,8192 if tier=='l2' else 1048576,1], 'store shape drift')
        symbol=next(s for s in functions if 'store_kernelEPV' in s)
        constants={0x360:256,0x388:n('tile-elements'),0x38c:n('iterations'),0x390:1,0x394:1,0x398:1,0x39c:0}
        coords={'SR_TID.X':Range(0,255),'SR_CTAID.X':Range(0,187)}
        threads,blocks=256,188; module='store_bridge'
    elif fixture=='floor':
        require(tier=='l2' and n('n')==32,'floor shape drift')
        symbol=next(s for s in functions if 'triadEPKf' in s)
        constants={0x3a0:32}
        coords={'SR_TID.X':Range(0,15),'SR_TID.Y':Range(0,15),'SR_CTAID.X':Range(0,1),'SR_CTAID.Y':Range(0,1)}
        threads,blocks=256,4; module='floor_bridge'
    else: raise Refusal('unsupported fixture')
    require(tier in ('l2','dram'), 'unsupported tier')
    return symbol,constants,coords,threads,blocks,module

def counts(sites,path,threads,blocks):
    require(threads%32==0,'partial warps unsupported')
    warps=threads//32*blocks; lanes=threads*blocks
    issues=Counter(); thread_events=Counter(); uniform_events=Counter(); per_site=[]
    for site in sites:
        reached=sum(path['visits'][(site.pc,p)] for p in (True,False))
        effective=path['visits'][(site.pc,True)]
        # Uniform/control instructions execute once per warp. No thread-event
        # claim for those opcodes. Vector operations have all-lane proof.
        uniform=site.op.startswith(('U','LDCU')) or site.op in ('S2UR','BRA','BRA.U','EXIT','NOP')
        issues[site.op]+=reached*warps
        (uniform_events if uniform else thread_events)[site.op]+=effective*(warps if uniform else lanes)
        per_site.append({'pc':hex(site.pc),'opcode':site.op,'predicate':site.pred,'operands':list(site.args),'visits_per_warp':reached,'predicate_true_visits_per_warp':effective,'reached_warp_issues':reached*warps,'predicate_true_thread_events':None if uniform else effective*lanes,'uniform_warp_events':effective*warps if uniform else None})
    return {'reached_warp_issues_by_full_opcode':dict(sorted(issues.items())), 'predicate_true_thread_events_by_full_opcode':dict(sorted(thread_events.items())), 'uniform_warp_events_by_full_opcode':dict(sorted(uniform_events.items())), 'sites':per_site}

def rank(rows):
    a=[[Fraction(x) for x in row] for row in rows]; r=0
    if not a: return 0
    for c in range(len(a[0])):
        pivot=next((i for i in range(r,len(a)) if a[i][c]),None)
        if pivot is None: continue
        a[r],a[pivot]=a[pivot],a[r]; d=a[r][c]; a[r]=[x/d for x in a[r]]
        for i in range(len(a)):
            if i!=r:
                d=a[i][c]; a[i]=[x-d*y for x,y in zip(a[i],a[r])]
        r+=1
        if r==len(a): break
    return r

def verify_abi(archive, module, symbol):
    """Verify actual compiler-generated argument order/offset and PTX types.

    Constant-bank parameter metadata is decoded only from the exact pinned
    ELF64LE cubin. Pointer values are unknown, never used as controls.
    Special launch dimensions are hardware c-bank values, not kernel params.
    """
    ptx=(archive/'compiler_intermediates'/f'{module}.ptx').read_text()
    parameter_text=ptx.split(f'.entry {symbol}(',1)[1].split(')',1)[0]
    types=re.findall(r'\.param\s+\.(u\d+)\b',parameter_text)
    expected={'alternative_bridge':(['u64','u64','u32','u32','u32','u32'],[0,8,16,20,24,28]),'store_bridge':(['u64','u32','u32','u32','u32','u32','u8'],[0,8,12,16,20,24,28]),'floor_bridge':(['u64','u64','u64','u64','u32'],[0,8,16,24,32])}[module]
    require(types==expected[0],'PTX parameter types/order drift')
    stub=(archive/'compiler_intermediates'/f'{module}.cudafe1.stub.c').read_text()
    name='__device_stub_'+symbol
    lines=[line for line in stub.splitlines() if name+'(' in line and '__cudaLaunchPrologue' in line]
    require(len(lines)==1,'missing/ambiguous actual launch stub')
    offsets=[int(x) for x in re.findall(r'__cudaSetupArgSimple\(__par\d+,\s*(\d+)UL\)',lines[0])]
    require(offsets==expected[1],'host argument offset drift')
    ordinals=[int(x) for x in re.findall(r'__cudaSetupArgSimple\(__par(\d+),',lines[0])]
    require(ordinals==list(range(len(types))), 'host argument order drift')
    metadata=cubin_parameters(archive/'compiler_intermediates'/f'{module}.sm_120.cubin',symbol)
    require(metadata['parameter_base']==0x380,'compiled constant-bank base drift')
    require(metadata['ordinal_offsets']==list(enumerate(offsets)), 'cubin/stub argument offsets disagree')
    return {'parameter_types':types,'stub_offsets':offsets,'constant_bank_parameter_base':hex(metadata['parameter_base']),'cubin_parameter_metadata':metadata,'stub_sha256':sha(archive/'compiler_intermediates'/f'{module}.cudafe1.stub.c'),'launch_geometry_origin':'hash-bound native fixture source plus recorded command; complete thread/block coordinate ranges','runtime_dispatch_independently_traced':False}

def cubin_parameters(path,symbol):
    """Read the retained NVIDIA .nv.info parameter records, not executable code.

    Narrow ELF64LE and sm_120 only. EIATTR_PARAM_CBANK (format4/tag0x0a)
    contains an opaque u32 identifier, u16 base, u16 length; KPARAM_INFO (tag0x17)
    contains index, u16 ordinal, u16 parameter offset and opaque flags. The
    flags are retained, not interpreted as width/energy/class information.
    Formats1/2 carry their value in the record header; format3 is u16.
    """
    data=path.read_bytes(); require(data[:6]==b'\x7fELF\x02\x01','unsupported cubin ELF')
    header=struct.unpack_from('<HHIQQQIHHHHHH',data,16)
    shoff,entsize,num,string_index=header[5],header[10],header[11],header[12]
    require(entsize==64 and 0<string_index<num and shoff+entsize*num<=len(data),'invalid ELF section table')
    sections=[struct.unpack_from('<IIQQQQIIQQ',data,shoff+i*entsize) for i in range(num)]
    ss=sections[string_index];strings=data[ss[4]:ss[4]+ss[5]]
    matches=[]
    for section in sections:
        require(section[0]<len(strings),'invalid section name')
        name=strings[section[0]:].split(b'\0',1)[0].decode()
        if name=='.nv.info.'+symbol:
            require(section[4]+section[5]<=len(data),'truncated info section')
            matches.append(data[section[4]:section[4]+section[5]])
    require(len(matches)==1,'missing/ambiguous cubin parameter section')
    raw=matches[0]; position=0; bank=[];parameters=[]
    while position<len(raw):
        require(position+4<=len(raw),'truncated info header')
        fmt,tag,size=struct.unpack_from('<BBH',raw,position);position+=4
        if fmt==4:
            require(position+size<=len(raw),'truncated info payload')
            payload=raw[position:position+size];position+=size
        else:
            require(fmt in (1,2,3),'unsupported info format');payload=b''
        if tag==0x0a:
            require(fmt==4 and size==8,'unsupported parameter bank format')
            bank.append(struct.unpack('<IHH',payload))
        if tag==0x17:
            require(fmt==4 and size==12,'unsupported kernel parameter format')
            index,ordinal,offset,flags=struct.unpack('<IHHI',payload)
            parameters.append((ordinal,offset,flags))
    require(len(bank)==1 and parameters,'missing bank/parameter metadata')
    return {'opaque_parameter_bank_identifier':bank[0][0],'parameter_base':bank[0][1],'parameter_bytes':bank[0][2],'ordinal_offsets':sorted((o,p) for o,p,f in parameters),'opaque_parameter_flags':{str(o):hex(f) for o,p,f in parameters},'nv_info_sha256':hashlib.sha256(raw).hexdigest(),'cubin_sha256':sha(path)}

def diagnostics(cells):
    result={}
    for tier in ('l2','dram'):
        selected=[c for c in cells if c['fixture']=='alternative' and c['tier']==tier]
        null=next(c for c in selected if c['mask']==0)
        opcodes=sorted(set().union(*(set(c['counts']['reached_warp_issues_by_full_opcode']) for c in selected)))
        contrasts=[]
        for c in selected:
            if c is null: continue
            delta={op:c['counts']['reached_warp_issues_by_full_opcode'].get(op,0)-null['counts']['reached_warp_issues_by_full_opcode'].get(op,0) for op in opcodes}
            contrasts.append({'design_id':c['design_id'],'all_opcode_deltas_warp_issues':{k:v for k,v in delta.items() if v},'nuisance_opcode_deltas_warp_issues':{k:v for k,v in delta.items() if v and k not in TARGETS}})
        varying=sorted(set().union(*(set(c['all_opcode_deltas_warp_issues']) for c in contrasts)))
        matrix=[[c['all_opcode_deltas_warp_issues'].get(op,0) for op in varying] for c in contrasts]
        target_matrix=[[c['all_opcode_deltas_warp_issues'].get(op,0) for op in TARGETS] for c in contrasts]
        nuisance=[op for op in varying if op not in TARGETS]
        nuisance_matrix=[[c['all_opcode_deltas_warp_issues'].get(op,0) for op in nuisance] for c in contrasts]
        joint_rank=rank(matrix); nuisance_rank=rank(nuisance_matrix)
        result[tier]={'contrasts':contrasts,'target_columns':list(TARGETS),'target_rank':rank(target_matrix),'varying_full_opcode_columns':varying,'joint_full_opcode_rank':joint_rank,'nuisance_rank':nuisance_rank,'target_rank_after_allowing_free_nuisance_coefficients':joint_rank-nuisance_rank,'all_nuisance_equal_to_null':not nuisance,'generic_per_opcode_prices_identified':False,'runtime_column_known':False,'measured_design_conditioning_known':False,'reason':'Actual compiled nuisance deltas and absent energy/runtime design prevent a physical class-price claim. Static target rank alone is not empirical identification.'}
    return result

def derive(archive=ARCHIVE):
    require(sha(archive/'calibration_blackwell')==PIN_BINARY,'binary drift')
    require(sha(archive/'retained_full_sass.stdout')==PIN_SASS,'SASS drift')
    require(sha(archive/'engineering_archive.json')==PIN_ARCHIVE,'archive manifest drift')
    manifest=json.loads((archive/'engineering_archive.json').read_text())
    require(manifest['binary_sha256']==PIN_BINARY and len(manifest['numeric_cells'])==15,'archive identity/grid drift')
    for module,h in PIN_PTX.items(): require(sha(archive/'compiler_intermediates'/f'{module}.ptx')==h,'PTX drift')
    # Current source packet must still agree with frozen reviewed build inputs.
    root=BASE.parents[2]
    for rel,h in PIN_SOURCE.items():
        full='tiresias/framework/compile_evidence/'+rel
        require(sha(root/full)==h,'source drift: '+rel)
    for rel,h in manifest['compiler_intermediate_sha256'].items(): require(sha(archive/rel)==h,'compiler evidence drift: '+rel)
    functions=parse_sass((archive/'retained_full_sass.stdout').read_text())
    cells=[]; seen=set()
    for numeric in manifest['numeric_cells']:
        design=numeric['design_id']; require(design not in seen,'duplicate cell');seen.add(design)
        file_id=design.replace('/','_')
        command_path=archive/f'correctness_{file_id}_command.json'
        options=argv_values(json.loads(command_path.read_text()))
        require(design.split('/')[1]==options['--tier'],'tier mismatch')
        label=design.split('/')[2]
        expected_labels={'null':('alternative',0,0),'shuffle_bfly32':('alternative',1,16),'shared_load32':('alternative',2,16),'shared_store32':('alternative',4,16),'ex2_approx_ftz':('alternative',8,16),'mixed-all-four':('alternative',15,4),'coalesced-store':('store',None,None),'tiny-floor':('floor',None,None)}
        require(label in expected_labels,'unknown design label')
        fixture,mask,k=expected_labels[label]
        require(options['--fixture']==fixture,'fixture/design mismatch')
        if fixture=='alternative': require(int(options['--mask'])==mask and int(options['--k'])==k,'specialization/design mismatch')
        generic={'--mode','--fixture','--tier','--expected-uuid','--deadline-ns','--graph-batch','--graph-replays','--trace-dir'}
        extra={'--n'} if fixture=='floor' else {'--blocks','--threads','--iterations','--stride','--step-warps','--tile-elements'} | ({'--mask','--k'} if fixture=='alternative' else {'--batches'})
        require(set(options)==generic|extra,'unknown or missing launch option')
        require(options['--trace-dir']=='/home/user/ipdps_model_20260930_0cbc61ec/acquisition_attempt/correctness/'+file_id,'native trace/design mismatch')
        native_path=archive/'correctness'/file_id/'native_result.json'
        require(sha(native_path)==numeric['native_result_sha256'],'native result drift')
        native=json.loads(native_path.read_text())
        require(native['launch_count']==2000 and native['graph_batch']==1000 and native['priming_launches']==1000,'native replay mismatch')
        symbol,constants,coords,threads,blocks,module=binding(options,functions)
        ptx=(archive/'compiler_intermediates'/f'{module}.ptx').read_text()
        require('.version 9.2' in ptx and '.target sm_120' in ptx and f'.entry {symbol}(' in ptx,'PTX exact entry mismatch')
        abi=verify_abi(archive,module,symbol)
        path=execute(functions[symbol],constants,coords)
        cells.append({'design_id':design,'fixture':options['--fixture'],'tier':options['--tier'],'mask':int(options['--mask']) if '--mask' in options else None,'function':symbol,'threads_per_block':threads,'blocks':blocks,'constant_parameter_bindings':{hex(k):v for k,v in constants.items()},'abi_linkage':abi,'coordinate_ranges':{k:[v.lo,v.hi] for k,v in coords.items()},'command_sha256':sha(command_path),'native_result_sha256':sha(native_path),'ptx_sha256':PIN_PTX[module],'numeric_full_exact':native['numeric_full_exact'],'path_status':'derived_uniform_compiled_path_candidate','profiler_validated':False,'scientifically_admitted':False,'branch_taken_per_warp':{hex(k):v for k,v in path['branch_taken'].items()},'branch_not_taken_per_warp':{hex(k):v for k,v in path['branch_not_taken'].items()},'counts':counts(functions[symbol],path,threads,blocks)})
    require(len(cells)==15,'missing grid')
    return {'schema':'retained_calibration_compiled_path_candidate_v1','source_commit':'0cbc61ec254ba3ec2160228233101867845b95ec','binary_sha256':PIN_BINARY,'sass_sha256':PIN_SASS,'scope':'One retained Blackwell calibration executable only. Per launch; priming/fill/setup excluded from counted workload and remain separately charged acquisition cost.','class_unit':'predicate-true thread events for SHFL.BFLY, LDS32, STS32 and MUFU.EX2; separate uniform/control warp events','declared_cells':15,'derived_cells':len(cells),'production_target_operator_coverage':0,'profiler_validated_cells':0,'scientific_admission':False,'energy_admitted':False,'policy_registration':None,'evidence_limit':'Recorded command, host dispatch source/PTX ABI and native result bind the intended launch. No independently traced runtime kernel dispatch or compatible profiler count references were acquired. Cubin-to-SASS association relies on the retained cuobjdump command/output, not a new local disassembly.','cells':cells,'compiled_design_diagnostics':diagnostics(cells)}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=HERE/'compiled_counts.json'); args=parser.parse_args()
    report=derive(); args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'derived_cells':report['derived_cells'],'production_target_operator_coverage':0,'energy_admitted':False,'output':str(args.output)}))
if __name__=='__main__': main()
