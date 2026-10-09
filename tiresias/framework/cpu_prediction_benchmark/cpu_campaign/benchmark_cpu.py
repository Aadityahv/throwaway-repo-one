"""Full inventory cold analysis and cached prediction; CPU-only isolated outputs."""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SR=ROOT/'tiresias/framework/predictor'
for p in [HERE,SR,SR/'prospective_test',SR/'calibrate']:
    sys.path.insert(0,str(p))


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False,default=str)+'\n');tmp.replace(path)


def prepare(out):
    import prosp_common as C
    import predict_runtime_v3k as K
    import predict as P
    from cal import traffic as T
    tasks=[];seen=set();sources={}
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    oracle={}
    for filename in ['eval_cells.csv','eval_validation.csv','eval_prospective.csv']:
        path=ROOT/'tiresias/evaluation/results'/filename
        sources[str(path.relative_to(ROOT))]=sha(path)
        for r in csv.DictReader(path.open()):
            refused=r.get('refused','False').lower()=='true'
            oracle[r['cell_id']]={'runtime_s':None if refused else float(r['runtime_ours_s']),
                                  'energy_j':None if refused else float(r['e_ours_static'])}
    for s in C.iter_sets():
        sm=s['feats']['hardware_from_ground_truth']['sm_count']
        preds=K.predict_portable(s['feats'],s['ph'],s['un'],s['bk'],s['C'],sm)
        doc=P.load_calibration(s['docp'],allow_incomplete=True)
        sources[str(s['docp'].relative_to(ROOT))]=sha(s['docp'])
        for row in s['feats']['rows']:
            cid=row['cell_id']
            if cid not in oracle: continue  # Exclude the archived correctness failure.
            assert cid not in seen;seen.add(cid)
            pred=preds[cid];rt=pred.get('primary_s');tr=pred.get('traffic')
            energy=T.energy_rows_traffic(doc,{'rows':[row]},{cid:rt} if rt else {},{cid:tr} if tr else {})[cid].get('energy_j')
            expected={'runtime_s':rt,'energy_j':energy}
            from pipeline_worker import verify
            verify({'cell_id':cid,'expected':oracle[cid]},expected)
            task={'cell_id':cid,'set_name':s['name'],'sm_count':sm,
                  'constants_dir':str(s['constants_dir'].relative_to(ROOT)) if 'constants_dir' in s else str((SR/'constants' if s['name'] in ['classic','e','d','f','validation_classic','validation_e','validation_f'] else SR/'fresh_g/constants_tensor').relative_to(ROOT)),
                  'calibration_path':str(s['docp'].relative_to(ROOT)),'expected':expected,
                  'cached_record':{'features':row,'phases':s['ph'][cid],'unique':s['un'][cid],'bank':s['bk'][cid],'hardware':s['feats']['hardware_from_ground_truth']}}
            tasks.append(task)
            write(out/'tasks'/(cid.replace('/','__')+'.json'),task)
    assert seen==set(oracle) and len(tasks)==167
    write(out/'inventory.json',{'configurations':len(tasks),'refused_predictions':sum(t['expected']['runtime_s'] is None for t in tasks),'tasks':[str((out/'tasks'/(t['cell_id'].replace('/','__')+'.json')).resolve()) for t in tasks],'sources_sha256':sources,'oracle_reproduced':True})
    print('PREPARED full inventory:',len(tasks),'configurations, frozen predictions reproduced',flush=True)


def percentile(xs,p):
    xs=sorted(xs);pos=(len(xs)-1)*p;l=math.floor(pos);h=math.ceil(pos)
    return xs[l]+(pos-l)*(xs[h]-xs[l])


def cached_one(path):
    from pipeline_worker import prediction,verify
    start=time.perf_counter();cpu=time.process_time()
    task=json.loads(Path(path).read_text());got=prediction(task,task['cached_record']);verify(task,got)
    return {'cell_id':task['cell_id'],'wall_s':time.perf_counter()-start,'cpu_s':time.process_time()-cpu,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'prediction':got}


def cold_one(path,run):
    task=json.loads(Path(path).read_text());key=task['cell_id'].replace('/','__')
    output=run/'cells'/(key+'.json');log=run/'logs'/(key+'.txt');log.parent.mkdir(parents=True,exist_ok=True)
    assert not output.exists(),'Cold cache was not empty'
    start=time.perf_counter()
    # GNU time measures the whole fresh interpreter process including imports.
    timing=run/'times'/(key+'.json');timing.parent.mkdir(parents=True,exist_ok=True)
    cmd=['/usr/bin/time','-f','{"cpu_user_s":%U,"cpu_system_s":%S,"max_rss_kib":%M,"exit_code":%x}','-o',str(timing),sys.executable,str(HERE/'pipeline_worker.py'),path,str(output)]
    with log.open('w') as f: result=subprocess.run(cmd,stdout=f,stderr=f)
    if result.returncode:
        raise RuntimeError(task['cell_id']+': worker failed; '+str(log))
    timed=json.loads(timing.read_text());doc=json.loads(output.read_text())
    return {'cell_id':task['cell_id'],'wall_s':time.perf_counter()-start,'cpu_s':timed['cpu_user_s']+timed['cpu_system_s'],'max_rss_kib':timed['max_rss_kib'],'analysis_s':doc['analysis_s'],'prediction_s':doc['prediction_s'],'prediction':doc['prediction']}


def monitor(stop,samples):
    import psutil
    proc=psutil.Process()
    while not stop.wait(.2):
        try:
            tree=[proc]+proc.children(recursive=True)
            samples.append({'epoch_s':time.time(),'aggregate_rss_bytes':sum(p.memory_info().rss for p in tree if p.is_running()),'host_load':os.getloadavg()})
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass


def benchmark(out,mode,jobs):
    inv=json.loads((out/'inventory.json').read_text());paths=inv['tasks'];run=out/(mode+'_workers_'+str(jobs))
    assert not run.exists(),'Run already exists; preserve it and use a new output root'
    run.mkdir();samples=[];stop=threading.Event();mon=threading.Thread(target=monitor,args=(stop,samples),daemon=True);mon.start()
    start=time.perf_counter();ru0=resource.getrusage(resource.RUSAGE_CHILDREN);rows=[]
    # Across the entire inventory, use heaviest shapes first to reduce end tails.
    paths.sort(key=lambda p: (0 if any(x in p for x in ['prosp','attention','matmul','matrix_multiply','reduce6','xlarge','dram']) else 1,p))
    try:
        pool=cf.ThreadPoolExecutor(max_workers=jobs) if mode=='cold' else cf.ProcessPoolExecutor(max_workers=jobs)
        with pool:
            futs={pool.submit(cold_one,p,run) if mode=='cold' else pool.submit(cached_one,p):p for p in paths}
            for fut in cf.as_completed(futs):
                rows.append(fut.result());write(run/'progress.json',{'completed':len(rows),'total':len(paths),'elapsed_s':time.perf_counter()-start,'rows':rows})
                print(mode,jobs,len(rows),'/',len(paths),rows[-1]['cell_id'],round(rows[-1]['wall_s'],2),flush=True)
        elapsed=time.perf_counter()-start;ru1=resource.getrusage(resource.RUSAGE_CHILDREN)
    finally:stop.set();mon.join()
    assert len(rows)==167 and len({r['cell_id'] for r in rows})==167
    wall=[r['wall_s'] for r in rows]
    report={'mode':mode,'workers':jobs,'configurations':len(rows),'wall_s':elapsed,'throughput_cells_per_min':len(rows)*60/elapsed,
            'worker_cpu_s':sum(r['cpu_s'] for r in rows),'child_cpu_s':(ru1.ru_utime+ru1.ru_stime)-(ru0.ru_utime+ru0.ru_stime),
            'peak_aggregate_rss_bytes':max((s['aggregate_rss_bytes'] for s in samples),default=0),
            'max_cell_rss_kib':max(r['max_rss_kib'] for r in rows),
            'cell_wall_s':{'median':statistics.median(wall),'mean':statistics.mean(wall),'p90':percentile(wall,.9),'max':max(wall)},
            'predictions_match_frozen':True,'rows':rows,'memory_samples':samples}
    write(run/'report.json',report)
    print('COMPLETE',mode,jobs,round(elapsed,2),'seconds',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--prepare',action='store_true');ap.add_argument('--mode',choices=['cold','cached']);ap.add_argument('--jobs',type=int,default=1);a=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','GPU visibility must be disabled'
    assert 1<=a.jobs<=56
    if a.prepare:prepare(a.out)
    else:benchmark(a.out,a.mode,a.jobs)

if __name__=='__main__':main()
