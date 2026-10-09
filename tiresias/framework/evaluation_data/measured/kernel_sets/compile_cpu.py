"""CPU-only nvcc/PTX/resource capture. CUDA visibility MUST be empty; no binaries are executed."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
FAMILIES={
 'sp':('fresh_e','cpp/2_Concepts_and_Techniques/scalarProd','scalarProd_kernel.cuh'),
 'fwt':('fresh_e','cpp/5_Domain_Specific/fastWalshTransform','fastWalshTransform_kernel.cuh'),
 'matmul':('unseen_kernels','cpp/0_Introduction/matrixMul','matrixMul.cu'),
 'bs':('unseen_kernels','cpp/5_Domain_Specific/BlackScholes','BlackScholes_kernel.cuh'),
 'scan':('unseen_kernels','cpp/2_Concepts_and_Techniques/scan','scan.cu'),
 'conv':('unseen_kernels','cpp/2_Concepts_and_Techniques/convolutionSeparable','convolutionSeparable.cu')}
REV='5443602d89ed99aede2e4b7bf329daddeadb320e'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--samples',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise SystemExit('REFUSED: CUDA visibility must explicitly be empty')
 if a.out.exists():raise SystemExit('REFUSED: output already exists')
 rev=subprocess.check_output(['git','-C',str(a.samples),'rev-parse','HEAD'],text=True).strip()
 if rev!=REV:raise SystemExit('REFUSED: cuda-samples revision differs')
 a.out.mkdir(parents=True);nvcc='/usr/local/cuda-13.2/bin/nvcc'; cuobj='/usr/local/cuda-13.2/bin/cuobjdump'
 log=dict(schema='published_fresh_compile/1',samples_revision=rev,nvcc=subprocess.check_output([nvcc,'--version'],text=True),cuda_visible_devices='',gpu_contexts=0,kernel_launches=0,commands=[],inputs_sha256={})
 for fam,(section,sdir,sfile) in FAMILIES.items():
  drv=a.repo/'tiresias/framework/predictor'/section/'gpu/drivers';src=drv/('driver_'+fam+'.cu')
  pinned=a.samples/sdir/sfile
  status=subprocess.check_output(['git','-C',str(a.samples),'status','--porcelain','--',str(pinned.relative_to(a.samples))],text=True)
  if status.strip():raise SystemExit('REFUSED: sample source modified')
  log['inputs_sha256'][str(src)]=sha(src);log['inputs_sha256'][str(pinned)]=sha(pinned)
  flags=[nvcc,'-arch=sm_120','-I',str(a.samples/'Common'),'-I',str(a.samples/sdir),'-I',str(drv),str(src)]
  for mode,suffix in [('-ptx','.ptx'),('-cubin','.cubin')]:
   cmd=flags+[mode,'-Xptxas=-v','-o',str(a.out/(fam+suffix))];p=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
   log['commands'].append(dict(argv=cmd,exit_code=p.returncode));(a.out/(fam+suffix+'.stdout')).write_text(p.stdout);(a.out/(fam+suffix+'.stderr')).write_text(p.stderr)
   if p.returncode:raise SystemExit('REFUSED: compilation failed '+fam+mode)
  p=subprocess.run([cuobj,'--dump-sass',str(a.out/(fam+'.cubin'))],capture_output=True,text=True,check=True)
  (a.out/(fam+'.sass')).write_text(p.stdout)
 log['output_sha256']={p.name:sha(p) for p in a.out.iterdir()};(a.out/'manifest.json').write_text(json.dumps(log,indent=2)+'\n')
 print('Six families compiled and disassembled; no CUDA context or target launch.')
if __name__=='__main__':main()
