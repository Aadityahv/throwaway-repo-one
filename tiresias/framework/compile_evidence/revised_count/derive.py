"""CPU-only candidate paths for one retained revised calibration executable.

No profiler, runtime dispatch trace, label access or scientific admission.
The old interpreter is reused as a hash-bound arithmetic helper, never as an
old artifact/geometry/admission adapter. Every new row is archive-bound.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ARCHIVE = BASE/'acquisition_runs/calibration_correctness_blackwell_20261001_0005d887'
PIN_SASS = '7b183144b5605f528e146b0042b3ad1c3617a746e2ea1528f1ba6235b56d45d9'
PIN_HELPER = '8f9e65865863b3ad623e220255a6a0ca7ee22b36dfae928f2cc3a3c416556876'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module

if sha(BASE/'retained_count/derive.py') != PIN_HELPER:
    raise ValueError('Reviewed arithmetic/ELF helper changed')
helper=load_module('revised_calibration_count_helpers',BASE/'retained_count/derive.py')
require=helper.require

def functions(text):
    require(hashlib.sha256(text.encode()).hexdigest()==PIN_SASS,'Revised complete SASS changed')
    result={}
    for section in text.split('Function : ')[1:]:
        name=section.splitlines()[0].strip()
        require(name not in result and 'EF_CUDA_SM120' in section,'Duplicate/wrong architecture function')
        sites=[]
        for line in section.splitlines():
            if not re.match(r'\s*/\*[0-9a-f]+\*/',line):continue
            m=re.match(r'\s*/\*([0-9a-f]+)\*/\s+(?:@(!?(?:P|UP)\d+)\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;',line)
            require(m is not None,'Dropped SASS instruction')
            pc,pred,op,args=m.groups()
            sites.append(helper.Site(int(pc,16),pred,op,tuple(x.strip() for x in args.split(',')) if args else (),line.split(';')[0].strip()))
        require(sites and [s.pc for s in sites]==list(range(0,len(sites)*16,16)),'Missing/duplicate PCs')
        result[name]=sites
    expected={f'_Z18alternative_kernelILj{mask}ELj{k}EEvPKjPjjjjj'
              for mask,k in [(0,0),(15,4)]+[(m,k) for m in [1,2,4,8] for k in [4,16]]}
    require(len(result)==13 and expected.issubset(result),'Full revised function inventory differs')
    require(sum('store_kernelEPV' in s for s in result)==1
            and sum('triadEPKf' in s for s in result)==1
            and '_Z22fill_alternative_inputPjy' in result,'Missing store/floor/fill function')
    return result

def bind(row, inventory):
    c=row['controls'];tier=row['tier']
    if 'mask' in c:
        symbol=f"_Z18alternative_kernelILj{c['mask']}ELj{c['k']}EEvPKjPjjjjj"
        constants={0x360:c['threads'],0x370:c['blocks'],0x390:c['tile_elements'],
                   0x394:c['iterations'],0x398:c['stride'],0x39c:c['step_warps']}
        coords={'SR_TID.X':helper.Range(0,c['threads']-1),'SR_CTAID.X':helper.Range(0,c['blocks']-1)}
        threads,blocks=c['threads'],c['blocks'];module='native_alternative_bridge';fixture='alternative'
    elif 'n' in c:
        symbol=next(s for s in inventory if 'triadEPKf' in s)
        constants={0x3a0:c['n']}
        coords={'SR_TID.X':helper.Range(0,15),'SR_TID.Y':helper.Range(0,15),
                'SR_CTAID.X':helper.Range(0,1),'SR_CTAID.Y':helper.Range(0,1)}
        threads,blocks=256,4;module='floor_bridge';fixture='floor'
    else:
        symbol=next(s for s in inventory if 'store_kernelEPV' in s)
        constants={0x360:c['threads'],0x388:c['tile_elements'],0x38c:c['iterations'],
                   0x390:c['stride'],0x394:c['step_warps'],0x398:c['batches'],0x39c:0}
        coords={'SR_TID.X':helper.Range(0,c['threads']-1),'SR_CTAID.X':helper.Range(0,c['blocks']-1)}
        threads,blocks=c['threads'],c['blocks'];module='store_bridge';fixture='store'
    require(symbol in inventory,'Unretained specialization')
    return symbol,constants,coords,threads,blocks,module,fixture

def abi(raw,module,symbol):
    ptx_path=raw/'compiler_intermediates'/f'{module}.ptx'
    ptx=ptx_path.read_text()
    require('.version 9.2' in ptx and '.target sm_120' in ptx,'PTX target/version changed')
    param=ptx.split(f'.entry {symbol}(',1)[1].split(')',1)[0]
    types=re.findall(r'\.param\s+\.(u\d+)\b',param)
    expected={'native_alternative_bridge':(['u64','u64','u32','u32','u32','u32'],[0,8,16,20,24,28]),
              'store_bridge':(['u64','u32','u32','u32','u32','u32','u8'],[0,8,12,16,20,24,28]),
              'floor_bridge':(['u64','u64','u64','u64','u32'],[0,8,16,24,32])}[module]
    require(types==expected[0],'PTX scalar/pointer ABI changed')
    stub_path=raw/'compiler_intermediates'/f'{module}.cudafe1.stub.c'
    stub=stub_path.read_text();name='__device_stub_'+symbol
    lines=[l for l in stub.splitlines() if name+'(' in l and '__cudaLaunchPrologue' in l]
    require(len(lines)==1,'Missing/ambiguous launch stub')
    offsets=[int(x) for x in re.findall(r'__cudaSetupArgSimple\(__par\d+,\s*(\d+)UL\)',lines[0])]
    ordinals=[int(x) for x in re.findall(r'__cudaSetupArgSimple\(__par(\d+),',lines[0])]
    require(offsets==expected[1] and ordinals==list(range(len(types))),'Stub argument mapping differs')
    metadata=helper.cubin_parameters(raw/'compiler_intermediates'/f'{module}.sm_120.cubin',symbol)
    require(metadata['parameter_base']==0x380 and metadata['ordinal_offsets']==list(enumerate(offsets)),
            'Actual cubin/stub argument disagreement')
    return dict(ptx_sha256=sha(ptx_path),stub_sha256=sha(stub_path),parameter_types=types,
                offsets=offsets,cubin_parameter_metadata=metadata,
                runtime_dispatch_independently_traced=False,implicit_geometry_ABI_independently_validated=False)

def diagnostics(cells):
    result={}
    for tier in ['l2','dram']:
        selected=[c for c in cells if c['fixture']=='alternative' and c['tier']==tier]
        null=next(c for c in selected if c['controls']['mask']==0)
        axes=[c for c in selected if c['controls']['mask'] in [1,2,4,8]]
        base=null['counts']['predicate_true_thread_events_by_full_opcode']
        deltas=[]
        for c in axes:
            values=c['counts']['predicate_true_thread_events_by_full_opcode']
            delta={op:values.get(op,0)-base.get(op,0) for op in sorted(set(values)|set(base))}
            deltas.append(dict(slot=c['slot'],full_opcode_thread_deltas=delta))
        nuisance=sorted({op for d in deltas for op in d['full_opcode_thread_deltas']} - set(helper.TARGETS))
        result[tier]=dict(null_fadd_thread_events=base.get('FADD',0),axis_contrasts=deltas,
                         target_contrast_rank=helper.rank([[d['full_opcode_thread_deltas'].get(t,0) for t in helper.TARGETS] for d in deltas]),
                         nuisance_contrast_rank=helper.rank([[d['full_opcode_thread_deltas'].get(t,0) for t in nuisance] for d in deltas]),
                         axis_FADD_equal_to_null=all(d['full_opcode_thread_deltas'].get('FADD',0)==0 for d in deltas),
                         physical_prices_identified=False,runtime_or_energy_identification_test=False)
    return result

def derive(archive=ARCHIVE):
    archive=Path(archive).resolve()
    auditor=load_module('revised_calibration_archive_verifier',BASE/'completion_review/audit_engineering_archive.py')
    audit=auditor.audit(archive);raw=archive/'output'
    packet=json.loads((archive/'prepared_packet.json').read_text())
    inventory=functions((raw/'full_sass.stdout').read_text());cells=[]
    for row in packet['correctness_rows']:
        symbol,constants,coords,threads,blocks,module,fixture=bind(row,inventory)
        path=helper.execute(inventory[symbol],constants,coords)
        c=helper.counts(inventory[symbol],path,threads,blocks)
        cells.append(dict(slot=row['slot'],tier=row['tier'],role=row['role'],controls=row['controls'],
                          fixture=fixture,symbol=symbol,counts=c,scalar_abi=abi(raw,module,symbol),
                          branch_taken={hex(k):v for k,v in path['branch_taken'].items()},
                          branch_not_taken={hex(k):v for k,v in path['branch_not_taken'].items()},
                          source_expectations_agree=all(c['predicate_true_thread_events_by_full_opcode'].get(op,0)==row['source_expected_counts'][key]
                            for op,key in [('SHFL.BFLY','shuffle_bfly32_thread'),('LDS','shared_load32_thread'),('STS','shared_store32_thread'),('MUFU.EX2','ex2_approx_ftz_thread')]),
                          scientific_admission=False,profiler_validated=False))
    require(len(cells)==25,'Incomplete revised count denominator')
    require(all(c['source_expectations_agree'] for c in cells),'Actual target-class/source count disagreement')
    return dict(schema='revised_calibration_compiled_path_candidates/1',source_commit=packet['source_commit'],
                binary_sha256=audit and sha(raw/'calibration_blackwell'),sass_sha256=PIN_SASS,
                outcome_sha256=sha(raw/'outcome.json'),decoder_sha256=PIN_HELPER,
                evaluator_sha256=sha(Path(__file__)),archive_auditor_sha256=sha(Path(auditor.__file__)),
                declared_cells=25,derived_cells=25,profiler_validated_cells=0,scientific_admission=False,
                actual_runtime_dispatch_traced=False,toolchain_closure_verified=False,energy_admitted=False,
                count_units='predicate-true thread events; modeled reached warp sites and uniform warp events remain separate',
                setup_fill_priming_included_in_per_launch_features=False,cells=cells,
                compiled_design_diagnostics=diagnostics(cells))

if __name__=='__main__':
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--archive',type=Path,default=ARCHIVE)
    p.add_argument('--out',type=Path,default=HERE/'compiled_candidates.json');args=p.parse_args()
    report=derive(args.archive);args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(derived_cells=25,scientific_admission=False,diagnostics={k:{x:v for x,v in d.items() if x!='axis_contrasts'} for k,d in report['compiled_design_diagnostics'].items()})))
