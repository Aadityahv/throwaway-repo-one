"""CPU compilation/disassembly of every unique Blackwell CUDA compilation unit."""
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

BASE=Path('/home/user/tiresias_review_cpu_20261005');ROOT=BASE/'source'
SR=ROOT/'tiresias/framework/predictor'; ORIGINAL=Path('/home/user/cudasamples_pinned_5443602d')
STAGE=BASE/'compile_source';OUT=BASE/'compile_results';NV=Path('/usr/local/cuda-13.2/bin')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert subprocess.check_output(['git','-C',str(ORIGINAL),'rev-parse','HEAD'],text=True).strip()=='5443602d89ed99aede2e4b7bf329daddeadb320e'
assert not OUT.exists(),'Preserve previous attempts and choose a fresh output root'
OUT.mkdir();os.sched_setaffinity(0,set(json.loads((BASE/'results/host.json').read_text())['selected_logical_cpus'][:8]))
SAMPLES=STAGE/'cuda_samples'
SAMPLES.mkdir(exist_ok=True)
shutil.copytree(ORIGINAL/'Common',SAMPLES/'Common',dirs_exist_ok=True)
units=[]
def add_sample(name,relative,setname,flags=None,expected=None,sass=None):
    d=ORIGINAL/relative;dst=SAMPLES/relative;shutil.copytree(d,dst,dirs_exist_ok=True)
    src=dst/(name+'.cu');assert src.exists()
    flags=flags or ['-arch=sm_120','-Dmain='+name+'_main_disabled','-I',str(SAMPLES/'Common')]
    flags=[f.replace(str(ORIGINAL),str(SAMPLES)) for f in flags]
    if expected is None:expected=SR/setname/'compiled/cubin'/(name+'.cubin')
    if sass is None:sass=SR/setname/'compiled/sass'/(name+'.sass')
    units.append({'name':name,'source':str(src),'flags':flags,'expected_cubin':str(expected),'expected_sass':str(sass)})
for name,rel in [('matrixMul','cpp/0_Introduction/matrixMul'),('BlackScholes','cpp/5_Domain_Specific/BlackScholes'),('scan','cpp/2_Concepts_and_Techniques/scan'),('convolutionSeparable','cpp/2_Concepts_and_Techniques/convolutionSeparable')]:
    add_sample(name,rel,'unseen_kernels')
for name,rel in [('scalarProd','cpp/2_Concepts_and_Techniques/scalarProd'),('fastWalshTransform','cpp/5_Domain_Specific/fastWalshTransform')]:add_sample(name,rel,'fresh_e')
corpus=ROOT/'tiresias/framework/compile_evidence/acquisition_runs/cuda_compile_blackwell_20261001_51afb2f3/output/nvcc'
retained=json.loads((corpus/'retention_manifest.json').read_text())['rows']
for name,rel in [('transpose','cpp/6_Performance/transpose'),('reduction_kernel','cpp/2_Concepts_and_Techniques/reduction'),('vectorAdd','cpp/0_Introduction/vectorAdd')]:
    row=next(r for r in retained if Path(r['retained_source_path']).name==name+'.cu')
    build=json.loads((corpus/row['build_path']).read_text());flags=build.get('compile_argv_prefix') or build['compile_flags']
    flags=[f for f in flags if f!='-cubin']
    if '-I' not in flags:flags+=['-I',str(SAMPLES/'Common')]
    add_sample(name,rel,'fresh_d',flags,corpus/row['cubin_path'],corpus/row['full_disassembly_path'])
    assert hashlib.sha256(Path(units[-1]['source']).read_bytes()).hexdigest()==row['source_sha256']
ML=STAGE/'tiresias/framework/predictor'
for dirname,name,filename in [('fresh_f','mlk','compile_unit.cu'),('fresh_g','tck','compile_unit_tc.cu'),('fresh_h','atk','compile_unit_attn.cu'),('prospective_test','prosp','compile_unit_prosp.cu')]:
    flags=['-arch=sm_120','-Xptxas','-v']
    for include in [ML/dirname/'src',ML/'fresh_f/src',ML/'fresh_g/src',ML/'fresh_h/src']:flags+=['-I',str(include)]
    units.append({'name':name,'source':str(ML/dirname/'src'/filename),'flags':flags,'expected_cubin':str(SR/dirname/'compiled/cubin'/(name+'.cubin')),'expected_sass':str(SR/dirname/'compiled/sass'/(name+'.sass'))})
assert len(units)==13

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def code(text):
    # Compare each function name plus exact encoded instruction lines, ignoring dump headers.
    return '\n'.join(x.strip() for x in text.splitlines() if 'Function :' in x or re.match(r'\s*/\*[0-9a-f]+\*/',x))
manifest={'units':units,'source_sha256':{str(p.relative_to(STAGE)):sha(p) for p in STAGE.rglob('*') if p.is_file()},
          'toolkit':subprocess.check_output([str(NV/'nvcc'),'--version'],text=True),'scope':'Compilation units are shared across launch configurations; compile each unique unit once per worker-count run. All 167 configuration analyses use these 13 units. No target execution. OS caches are not flushed. Source and toolchain matching is checked; exact cubin and encoded SASS equality are recorded.'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

def command(cmd,dest,log):
    timing=dest.with_suffix('.time.json'); start=time.perf_counter()
    with log.open('w') as f:
        r=subprocess.run(['/usr/bin/time','-f','{"user_s":%U,"system_s":%S,"max_rss_kib":%M}','-o',str(timing)]+cmd,stdout=f,stderr=f)
    if r.returncode:raise RuntimeError('Compile/disassembly failed: '+str(log))
    return {'wall_s':time.perf_counter()-start,**json.loads(timing.read_text()),'argv':cmd}
def one(unit,run):
    name=unit['name'];cubin=run/(name+'.cubin');sass=run/(name+'.sass');resources=run/(name+'.res')
    compile_info=command([str(NV/'nvcc')]+unit['flags']+['-cubin',unit['source'],'-o',str(cubin)],cubin,run/(name+'.compile.log'))
    sass_info=command([str(NV/'cuobjdump'),'--dump-sass',str(cubin)],sass,sass)
    resource_info=command([str(NV/'cuobjdump'),'--dump-resource-usage',str(cubin)],resources,resources)
    expected_code=code(Path(unit['expected_sass']).read_text());actual_code=code(sass.read_text());matching=actual_code==expected_code
    row={'name':name,'compile':compile_info,'disassemble':sass_info,'resource_usage':resource_info,'cubin_sha256':sha(cubin),
         'original_cubin_sha256':sha(unit['expected_cubin']),'cubin_exact':sha(cubin)==sha(unit['expected_cubin']),
         'encoded_sass_exact':matching,'encoded_sass_sha256':hashlib.sha256(actual_code.encode()).hexdigest()}
    (run/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n');return row
for workers in [1,4,8]:
    run=OUT/('workers_'+str(workers));run.mkdir();start=time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(lambda u:one(u,run),units))
    report={'workers':workers,'units':len(rows),'wall_s':time.perf_counter()-start,'cpu_s':sum(x[k]['user_s']+x[k]['system_s'] for x in rows for k in ['compile','disassemble','resource_usage']),
            'all_encoded_sass_exact':all(x['encoded_sass_exact'] for x in rows),'all_cubins_exact':all(x['cubin_exact'] for x in rows),'rows':rows}
    (run/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COMPLETE',workers,report['wall_s'],'SASS exact',report['all_encoded_sass_exact'],flush=True)
    if not report['all_encoded_sass_exact']:raise RuntimeError('Recompiled code differs; preserve this attempt and investigate before using its cost as identical compilation.')
