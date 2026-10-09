"""Complete-grid dependent-hop response reconstruction, CPU only.

No operator labels or predictions. All raw repeats are retained; held-out folds
exclude complete stride/thread groups, rather than individual repeated samples.
"""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
GRID = REPO/'calibration/work_time_grid'
GROUND = REPO/'HARDWARE_GROUND_TRUTH.md'
TIERS = ['L1','L2','DRAM']
STRIDES = [1,2,4,5,7,8,16,23,32,64]
THREADS = [32,64,128,256,384,512,768,1024]
GATE = {'median_max_pct':15, 'p90_max_pct':30}
ORIGINAL = HERE.parent/'constants/stream_constants.json'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sector_count(stride, threads, subtable_bytes):
    # The pointer-chase kernel reads uint64 pointers and advances by 32 B/hop.
    if threads%32: raise ValueError('whole warp required')
    sectors=[]
    elements=subtable_bytes//8
    for warp in range(threads//32):
        addresses=[((warp*32+lane)*stride%elements)*8 for lane in range(32)]
        sectors.append(len({byte//32 for a in addresses for byte in (a,a+7)}))
    return float(np.mean(sectors))


def read_grid():
    import sys
    sys.path.insert(0,str(HERE.parent))
    import extract_features as X
    hw=X.load_hardware(GROUND.read_text())
    sms, warp = hw['sm_count'],hw['warp_size']
    if sms is None or warp!=32: raise ValueError('ground-truth geometry missing/unsupported')
    hashes={str(GROUND.relative_to(REPO)):sha(GROUND)}
    raw=[]
    for path in sorted(GRID.glob('blackwell_*_unlocked/raw/energy_samples.csv')):
        manifest_path=path.parent/'manifest.txt'
        if not (path.parent/'COMPLETE').exists(): raise ValueError('incomplete retained slice '+str(path))
        manifest=dict(line.split('=',1) for line in manifest_path.read_text().splitlines() if '=' in line)
        if manifest['architecture']!='blackwell' or manifest['clock_mode']!='unlocked': raise ValueError('wrong slice')
        for source in [path,manifest_path]: hashes[str(source.relative_to(REPO))]=sha(source)
        for r in csv.DictReader(path.open()):
            if r['phase']!='main': continue
            if r['mode']!='opt' or r['clock_lock_status']!='unlocked' or r['architecture']!='blackwell':
                raise ValueError('unexpected main row')
            tier,stride,threads=r['memory_level'],int(r['stride']),int(r['threads'])
            blocks,size=int(r['blocks']),int(r['size_bytes'])
            if (threads!=int(manifest['threads']) or r['gpu_uuid']!=manifest['gpu_uuid']
                or r['run_id']!=manifest['run_id'] or r['commit']!=manifest['commit']):
                raise ValueError('row/manifest run identity mismatch')
            if tier not in TIERS or stride not in STRIDES or threads not in THREADS:
                raise ValueError('unexpected parameter axes')
            spec=dict(x.split(':',1) for x in manifest['level_'+tier].split(','))
            if blocks!=int(spec['blocks']) or size!=int(spec['size_bytes']) or r['binary_sha256']!=spec['binary_sha256']:
                raise ValueError('row/manifest geometry or binary mismatch')
            accesses=int(r['total_logical_accesses']);duration=float(r['kernel_time_s'])
            if accesses<=0 or not math.isfinite(duration) or duration<=0: raise ValueError('invalid timing')
            hops=accesses/(threads*blocks)
            if not hops.is_integer(): raise ValueError('nonintegral per-lane dependent hop count')
            sectors=sector_count(stride,threads,size)
            raw.append(dict(tier=tier,stride=stride,threads=threads,blocks=blocks,subtable_bytes=size,
                source=str(path.relative_to(REPO)),round=int(r['round']),hop_count_per_lane=int(hops),
                effective_hop_ns=duration/hops*1e9, sectors_per_warp=sectors,
                launched_warps_per_sm=blocks*threads/(sms*warp),
                binary_sha256=r['binary_sha256'], footprint_bytes=blocks*size))
    grouped=collections.defaultdict(list)
    for r in raw: grouped[r['tier'],r['stride'],r['threads']].append(r)
    expected={(l,s,t) for l in TIERS for s in STRIDES for t in THREADS}
    if set(grouped)!=expected: raise ValueError('incomplete full grid: '+repr(sorted(expected-set(grouped))))
    cells=[]
    for (tier,stride,threads),values in sorted(grouped.items()):
        signatures={(r['blocks'],r['subtable_bytes'],r['sectors_per_warp'],r['launched_warps_per_sm']) for r in values}
        if len(signatures)!=1: raise ValueError('geometry changed within a repeated configuration')
        row=dict(values[0]);row.pop('source');row.pop('round');row.pop('binary_sha256')
        row.update(effective_hop_ns=float(np.median([r['effective_hop_ns'] for r in values])),
            repeats=len(values), min_hop_ns=min(r['effective_hop_ns'] for r in values),
            max_hop_ns=max(r['effective_hop_ns'] for r in values))
        cells.append(row)
    return raw,cells,hashes


def response(coefficients, cells):
    c=np.asarray(coefficients)
    sectors=np.array([r['sectors_per_warp'] for r in cells])
    pressure=sectors*np.array([r['launched_warps_per_sm'] for r in cells])
    return np.maximum(c[0]+c[1]*(sectors-8),c[2]*pressure)


def fit(cells):
    """Deterministic bounded coordinate minimization; NumPy only, three costs.

    Multiple initial floors guard against different max-term active sets.
    Golden-section substeps minimize squared log-relative calibration residuals.
    """
    if not cells: raise ValueError('no calibration cells')
    y=np.array([r['effective_hop_ns'] for r in cells])
    extra=np.array([r['sectors_per_warp']-8 for r in cells])
    pressure=np.array([r['sectors_per_warp']*r['launched_warps_per_sm'] for r in cells])
    bounds=[max(y)*2,max(y)*2/max(1,max(extra)),max(y)*2/min(pressure)]
    def loss(p):
        pred=np.maximum(p[0]+p[1]*extra,p[2]*pressure)
        if np.any(pred<=0): return float('inf')
        return float(np.mean(np.log(pred/y)**2))
    best=None
    ratio=(math.sqrt(5)-1)/2
    for fraction in [.75,1,1.25]:
        p=np.array([min(y)*fraction,.1,float(np.median(y/pressure))])
        previous=loss(p)
        for sweep in range(40):
            for axis in range(3):
                left,right=0.0,bounds[axis]
                def objective(value):
                    q=p.copy();q[axis]=value
                    return loss(q)
                c=right-ratio*(right-left);d=left+ratio*(right-left)
                fc,fd=objective(c),objective(d)
                for _ in range(40):
                    if fc<fd:
                        right,d,fd=d,c,fc;c=right-ratio*(right-left);fc=objective(c)
                    else:
                        left,c,fc=c,d,fd;d=left+ratio*(right-left);fd=objective(d)
                candidate=(left+right)/2
                if objective(candidate)<loss(p):p[axis]=candidate
            current=loss(p)
            if previous-current<1e-12:break
            previous=current
        value=loss(p)
        if best is None or value<best[0]:best=value,p
    if best is None or not math.isfinite(best[0]):raise ValueError('calibration fit failed')
    return best[1].tolist()


def stats(errors):
    return dict(n=len(errors),median_pct=float(np.median(errors)),p90_pct=float(np.percentile(errors,90)),
                max_pct=float(max(errors)))


def reconstruct(cells):
    result={}
    baselines=json.loads(ORIGINAL.read_text())['constants']['latency_ns']
    for tier in TIERS:
        rows=[r for r in cells if r['tier']==tier]
        coef=fit(rows)
        predictions=response(coef,rows)
        report={'coefficients':dict(base_ns=coef[0],extra_sector_ns=coef[1],pressure_request_ns=coef[2]),
                'fit_error':stats([abs(p/r['effective_hop_ns']-1)*100 for p,r in zip(predictions,rows)]),
                'rows':[{**r,'predicted_hop_ns':float(p)} for r,p in zip(rows,predictions)],'heldout':{}}
        baseline=baselines[tier]
        report['original_constant_hop_error']=stats([abs(baseline/r['effective_hop_ns']-1)*100 for r in rows])
        indistinguishable=collections.defaultdict(list)
        for r in rows:
            indistinguishable[r['sectors_per_warp'],r['launched_warps_per_sm']].append(r)
        report['same_feature_spreads']=[dict(sectors_per_warp=s,launched_warps_per_sm=w,
            min_measured_hop_ns=min(r['effective_hop_ns'] for r in group),
            max_measured_hop_ns=max(r['effective_hop_ns'] for r in group),
            max_to_min_ratio=max(r['effective_hop_ns'] for r in group)/min(r['effective_hop_ns'] for r in group),
            configurations=[dict(stride=r['stride'],threads=r['threads']) for r in group])
            for (s,w),group in sorted(indistinguishable.items()) if len(group)>1]
        for axis in ['stride','threads']:
            heldout=[];folds=[]
            for value in sorted({r[axis] for r in rows}):
                train=[r for r in rows if r[axis]!=value]
                test=[r for r in rows if r[axis]==value]
                fitted=fit(train)
                predicted=response(fitted,test)
                errors=[abs(p/r['effective_hop_ns']-1)*100 for p,r in zip(predicted,test)]
                heldout.extend(errors)
                folds.append(dict(heldout_value=value,fit_cells=len(train),test_cells=len(test),
                    coefficients=fitted,error=stats(errors),
                    rows=[dict(stride=r['stride'],threads=r['threads'],measured_hop_ns=r['effective_hop_ns'],
                               predicted_hop_ns=float(p)) for p,r in zip(predicted,test)]))
            summary=stats(heldout)
            report['heldout'][axis]=dict(error=summary,folds=folds,
                passes=summary['median_pct']<=GATE['median_max_pct'] and summary['p90_pct']<=GATE['p90_max_pct'])
        result[tier]=report
    return result


def main():
    raw,cells,hashes=read_grid()
    for p in [Path(__file__),HERE/'PLAN.md',ORIGINAL,REPO/'calibration/benchmark_optimizer.cu',REPO/'energy_harness/wp9_grid_common.sh',
              REPO/'energy_harness/run_blackwell_work_time_grid.sh',HERE.parent/'extract_features.py']:
        hashes[str(p.relative_to(REPO))]=sha(p)
    results=reconstruct(cells)
    output=dict(schema='dependent_hop_calibration_reconstruction/1',raw_rows=len(raw),configuration_cells=len(cells),
        axes=dict(memory_candidates=TIERS,strides=STRIDES,threads=THREADS),clock_mode='unlocked',
        gate=GATE,passes_grid_reconstruction=all(v['heldout'][axis]['passes'] for v in results.values() for axis in ['stride','threads']),
        universal_memory_latency_admitted=False,compute_latency_identified=False,new_operator_predictions=False,
        raw_rows_retained=raw,results=results,source_sha256=hashes,
        assumptions=['Effective hop time applies to the retained eight-byte dependent pointer-chase family.',
          'Launched warps per SM describes this retained grid, not inferred resource occupancy on arbitrary operators.',
          'Source-pattern sector counts assume aligned allocations; no profiler transaction validation.',
          'Candidate memory tiers remain effective residency labels; no physical DRAM bandwidth is inferred.',
          'Kernel durations include pointer warmup, loop/control work and the initial barrier; hop time is an effective response.',
          'Current source and historical manifests are retained; exact binary-source closure is not independently re-established.'])
    (HERE/'calibration_reconstruction.json').write_text(json.dumps(output,indent=1,sort_keys=True)+'\n')
    print(json.dumps({k:output[k] for k in ['raw_rows','configuration_cells','passes_grid_reconstruction']},indent=1))
    print(json.dumps({k:{a:v['heldout'][a]['error'] for a in ['stride','threads']} for k,v in results.items()},indent=1))


if __name__=='__main__': main()
