"""Compare the FROZEN collective-reduction static counts with profiler per-opcode counts.

Run only after collective_counts.json was frozen. Refuses if derive.py / collective.py no longer
hash to the values recorded in the frozen file (so any post-hoc interpreter change is visible).
Reuses profiler_reference/inspect_capture.py helpers (sha, load, classifier); the per-cell
instances.json files are the retained typed extraction of the raw .ncu-rep reports, and the
retained report hash binding is re-verified here.
"""
import argparse, collections, importlib.util, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('inspect_capture',BASE/'profiler_reference/inspect_capture.py');IC=importlib.util.module_from_spec(spec);sys.modules[spec.name]=IC;spec.loader.exec_module(IC)
ARCHIVE=BASE/'acquisition_runs/profiler_blackwell_20261001_de268265'
CLASSES=['shuffle','shared_load','shared_store','exponential','fma','barrier']

def compare(frozen_path=HERE/'collective_counts.json',archive=ARCHIVE):
    fz=IC.load(frozen_path)
    IC.require(IC.sha(HERE/'derive.py')==fz['derive_py_sha256'],'derive.py changed since freeze')
    IC.require(IC.sha(HERE/'collective.py')==fz['collective_py_sha256'],'collective.py changed since freeze (retrospective repair must be labelled and the pre-fix file kept)')
    packet=IC.load(archive/'prepared_packet.json');raw=archive/'output';target=IC.classifier()
    slots={(r['corpus'],r['operator_id'],r['cell']):(i,r) for i,r in enumerate(packet['rows']) if r.get('family','operator')!='calibration'}
    out=[]
    for c in fz['rows']:
        i,row=slots[(c['corpus'],c['operator_id'],c['cell'])];folder=raw/('cell_'+str(i).zfill(3))
        ins=IC.load(folder/'instances.json');IC.require(ins['report_sha256']==IC.sha(folder/'profile.ncu-rep'),'raw report binding differs')
        drv=IC.load(folder/'driver_result.json');IC.require(drv['cubin_sha256']==c['evidence']['cubin_sha256']==row['cubin']['sha256'],'binary differs')
        actual={r['opcode']:r['count'] for r in ins['instances']}
        sums=collections.Counter();widths=collections.defaultdict(collections.Counter)
        for op,n in actual.items():
            t=target(op)
            if t:
                sums[t[0]]+=n
                if n and t[1] is not None:widths[t[0]][str(t[1])]+=n
        diffs=[];table={}
        for name in CLASSES:
            st=c['classes'][name];sw={k:v for k,v in st['width_bits_histogram'].items() if v}
            table[name]={'static':st['predicate_true_thread_instruction'],'profiler':sums[name],'static_widths':sw,'profiler_widths':dict(widths[name])}
            if st['predicate_true_thread_instruction']!=sums[name] or sw!=dict(widths[name]):diffs.append(name)
        st_modes=c['shuffle_modes_lane_invocations'];pr_modes={k:v for k,v in actual.items() if k.startswith('SHFL')}
        if st_modes!=pr_modes:diffs.append('shuffle_mode')
        out.append({'operator_id':c['operator_id'],'cell':c['cell'],'slot':row['slot'],'classes':table,'shuffle_modes':{'static':st_modes,'profiler':pr_modes},'differing':diffs,
                    'profiler_barrier_opcodes':{k:v for k,v in actual.items() if k.startswith('BAR')}})
    return {'schema':'collective_static_vs_profiler/1','frozen_file_sha256':IC.sha(frozen_path),'derive_py_sha256':fz['derive_py_sha256'],'collective_py_sha256':fz['collective_py_sha256'],
            'cells':len(out),'cells_with_any_difference':sum(bool(r['differing']) for r in out),'rows':out,'scientific_admission':False}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=None);a=ap.parse_args();r=compare()
    for x in r['rows']:
        print(x['operator_id'],x['cell'],'DIFF:'+','.join(x['differing']) if x['differing'] else 'match')
        for n,v in x['classes'].items():print('   ',n,v['static'],v['profiler'],v['static_widths'],v['profiler_widths'])
        print('    shuffle modes',x['shuffle_modes'],'barrier opcodes',x['profiler_barrier_opcodes'])
    print({k:v for k,v in r.items() if k!='rows'})
    if a.out:
        with a.out.open('x') as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\n')
