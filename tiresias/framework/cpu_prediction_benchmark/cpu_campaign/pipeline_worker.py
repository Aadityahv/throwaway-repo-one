"""CPU-only reconstruction with the original Blackwell pipeline functions."""
import collections
import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SR = ROOT / 'tiresias/framework/predictor'
for p in [SR,SR/'bank',SR/'shared_traffic',SR/'prospective_test',SR/'calibrate']:
    sys.path.insert(0,str(p))


def reconstruct(task):
    name, cid = task['set_name'], task['cell_id']
    import run_footprints as RF
    if name == 'd':
        sys.path.insert(0,str(SR/'fresh_d'))
        import fresh_d_lib as L, build_fresh_d as BD, bank_conflicts as BC
        hw,l2=L.l2_bytes_from_ground_truth()
        cell=next(c for c in L.define_cells(l2) if c['cell_id']==cid)
        spec=BD.cell_spec(cell,hw['sm_count'])
        out=L.analyse(spec)
        # Same retained bank derivation, bound to this fresh geometry.
        origin=L.retained_rows()[tuple(spec['origin'])]
        row=L.marked_row(origin,spec['family'],spec['geometry'])
        if not getattr(BC.D.binding,'_fresh_wrapper',False): BC.D.binding=L._wrap(BC.D.binding)
        base,shared,_=BC.run_retained('cuda',row,L.CUDA_ROOT)
        fk=out['phases']['kernels'][0]
        BC.gate_kernel(base,fk,shared)
        krow={'barrier_sequence':fk['barrier_sequence'],'phases':BC._merge_phase_ids(fk['phases'],shared),'sampled_blocks':base['sampled_blocks']}
        bank={'status':'static_shared_conflicts','kernels':[krow]}
        fps=RF.cell_d((L,BD),cell,hw)
        return {'features':out['features'],'phases':out['phases'],'unique':out['unique'],'bank':bank,
                'footprints':{'cell_id':cid,'status':'ok','kernels':fps},'hardware':hw}
    lib=name.removeprefix('validation_')
    if name=='prosp':
        sys.path.insert(0,str(SR/'prospective_test'))
        import prosp_lib as L, cells_prosp as V
        UP=L.UP
        hw=UP.X.load_hardware(UP.X.read_text(UP.X.GROUND_TRUTH))
        cells=V.define_cells(hw)
    else:
        if lib=='classic':
            sys.path.insert(0,str(SR/'unseen_kernels'))
            import unseen_pipeline as UP
        else:
            sys.path.insert(0,str(SR/('fresh_'+lib)))
            UP=__import__('fresh_'+lib+'_lib').UP
        hw=UP.X.load_hardware(UP.X.read_text(UP.X.GROUND_TRUTH))
        if name.startswith('validation_'):
            sys.path.insert(0,str(SR/'shared_traffic/validation'))
            import cells_validation as V
            cells=V.define(lib,hw)
        elif lib=='classic':
            import cells as V
            cells=V.define_cells(hw)
        else:
            cells=__import__('cells_'+lib).define_cells(hw)
    cell=next(c for c in cells if c['cell_id']==cid)
    # Derived SASS belongs to this run, never to frozen source directories.
    import os
    UP.ISO_DIR=Path(os.environ['CPU_CAMPAIGN_SCRATCH'])/cid.replace('/','__')/'isolated'
    UP.ISO_DIR.mkdir(parents=True,exist_ok=True)
    import bank_unseen as BU
    # Preserve the modern request-wavefront histogram when summing block classes.
    original=BU.sum_rows
    def sum_rows(rows):
        out=original(rows); h=collections.Counter()
        for row in rows: h.update(row.get('shared_request_wavefront_histogram',{}))
        out['shared_request_wavefront_histogram']={str(k):v for k,v in sorted(h.items(),key=lambda x:int(x[0]))}
        return out
    BU.sum_rows=sum_rows
    rec,ph,un=UP.build_cell(cell,hw)
    _,bank=BU.one((cell,ph))
    fps=RF.cell_up(UP,cell,hw)
    return {'features':rec,'phases':ph,'unique':un,'bank':bank,
            'footprints':{'cell_id':cid,'status':'ok','kernels':fps},'hardware':hw}


@lru_cache(None)
def model_inputs(constants_dir, calibration_path):
    from cal import portable_predict as PP
    import predict as P
    constants=PP.load_constants(ROOT/constants_dir)
    constants['smem']=None
    return constants,P.load_calibration(ROOT/calibration_path,allow_incomplete=True)


def prediction(task,record):
    import predict_runtime_v3j as J, predict_runtime_v3k as K
    from cal import traffic as T
    cid=task['cell_id']; r=record['features']
    un=record['unique']
    if 'footprints' in record:
        un=J.attach_footprints({cid:un},{cid:record['footprints']},int(record['hardware']['l2_bytes']))[cid]
    constants,calibration=model_inputs(task['constants_dir'],task['calibration_path'])
    p=K.predict_portable({'rows':[r],'hardware_from_ground_truth':record['hardware']},{cid:record['phases']},{cid:un},{cid:record['bank']},constants,task['sm_count'])[cid]
    rt=p.get('primary_s'); tr=p.get('traffic')
    e=T.energy_rows_traffic(calibration,{'rows':[r]},{cid:rt} if rt else {},{cid:tr} if tr else {})[cid]
    return {'runtime_s':rt,'energy_j':e.get('energy_j'),'reason':p.get('unsupported_reason')}


def verify(task,got):
    for key in ['runtime_s','energy_j']:
        want=task['expected'][key]; actual=got[key]
        assert (want is None and actual is None) or (want is not None and actual is not None and math.isclose(want,actual,rel_tol=1e-9,abs_tol=1e-14)),(task['cell_id'],key,want,actual)


def main():
    task=json.loads(Path(sys.argv[1]).read_text()); out=Path(sys.argv[2])
    start=time.perf_counter(); record=reconstruct(task); analysis_s=time.perf_counter()-start
    start=time.perf_counter(); got=prediction(task,record); prediction_s=time.perf_counter()-start
    verify(task,got)
    record['features'].pop('build_seconds',None)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'cell_id':task['cell_id'],'record':record,'prediction':got,'analysis_s':analysis_s,'prediction_s':prediction_s},sort_keys=True,default=str)+'\n')

if __name__=='__main__': main()
