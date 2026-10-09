"""Serial CPU-only native compilation in a new private directory; no CUDA context."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet',type=Path,default=HERE/'calibration_packet_current_port.json')
    p.add_argument('--out-dir',required=True,type=Path);p.add_argument('--booking-ref',required=True)
    p.add_argument('--i-have-a-booking',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    packet=json.loads(a.packet.read_text());nvcc=Path(packet['compile']['nvcc']);cuobjdump=nvcc.parent/'cuobjdump'
    for name,digest in packet['inputs_sha256'].items():
        path=HERE.parents[4]/name
        if sha(path)!=digest:raise ValueError('REFUSED: source packet drift: '+name)
    commands=[('compile',['nice','-n','19',str(nvcc),*packet['compile']['flags'],str(HERE/'calibration.cu'),'-o',str(a.out_dir/'calibration')]),
              ('sass',[str(cuobjdump),'--dump-sass',str(a.out_dir/'calibration')]),
              ('ptx',[str(cuobjdump),'--dump-ptx',str(a.out_dir/'calibration')]),
              ('resources',[str(cuobjdump),'--dump-resource-usage',str(a.out_dir/'calibration')])]
    if a.dry_run:print(json.dumps(dict(env={'CUDA_VISIBLE_DEVICES':''},commands=commands),indent=2));return
    if not a.i_have_a_booking or not a.booking_ref.strip():raise ValueError('REFUSED: shared-machine booking required, even for the new serial compiler scope')
    a.out_dir.mkdir(parents=True,exist_ok=False);env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    records=[];started=time.monotonic()
    for name,argv in commands:
        before=time.monotonic();child=subprocess.run(argv,capture_output=True,text=True,timeout=900,env=env)
        (a.out_dir/(name+'.stdout')).write_text(child.stdout);(a.out_dir/(name+'.stderr')).write_text(child.stderr)
        records.append(dict(stage=name,argv=argv,returncode=child.returncode,wall_s=time.monotonic()-before))
        with (a.out_dir/'compile_ledger.json').open('w') as f:json.dump(records,f,indent=2);f.write('\n')
        if child.returncode:raise SystemExit('REFUSED: native compilation/dump failure, retained in '+str(a.out_dir))
    out=dict(schema='energy_component_native_build/1',status='compiled_count_and_correctness_pending',booking_ref=a.booking_ref,
        cpu_only=True,elapsed_wall_s=time.monotonic()-started,packet_sha256=sha(a.packet),binary_sha256=sha(a.out_dir/'calibration'),
        compiler_sha256=sha(nvcc),source_sha256=sha(HERE/'calibration.cu'),oracle_code_sha256=sha(HERE/'oracle.hpp'),
        raw_sha256={f.name:sha(f) for f in sorted(a.out_dir.iterdir()) if f.is_file()})
    with (a.out_dir/'build_manifest.json').open('x') as f:json.dump(out,f,indent=2,sort_keys=True);f.write('\n')
    print('Compiled serially without visible CUDA devices; count/correctness gates pending')


if __name__=='__main__':main()
