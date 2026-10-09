"""Fit stream-term constants on the streaming calibration suite only (Amendment 3).

Microbenchmark windows: tiresias/app_runners/iteration4_fit_score_blackwell.json (both sessions,
mean runtime per configuration). Sector/line counts are derived from each fixture's address formula.
No operator data is read.
"""
import json
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[4]
SRC=REPO/'tiresias/app_runners/iteration4_fit_score_blackwell.json'

def store_pattern(stride,threads=256):
    """Mean sectors and 128-B lines per 32-lane warp request for index=(tid*stride)&mask, 4-byte stores."""
    s=l=0;warps=threads//32
    for w in range(warps):
        addrs=[((32*w+i)*stride)*4 for i in range(32)]
        s+=len({a//32 for a in addrs});l+=len({a//128 for a in addrs})
    return s/warps,l/warps

def windows():
    d=json.loads(SRC.read_text());acc={}
    for w in d['calibration']['per_window']:
        acc.setdefault((w['fixture'],w['candidate_id'],w['regime']),[]).append(w['runtime_us'])
    return {k:float(np.mean(v)) for k,v in acc.items()}

def counts(fixture,cand,regime,W):
    """Return dict(read_bytes, write_bytes, read_sectors, write_sectors, lines)."""
    if fixture=='triad':
        rb={'below_l2':12582912,'above_l2':805306368,'tiny':12288}[regime];wb=rb//3
        return dict(read_bytes=rb,write_bytes=wb,read_sectors=rb/32,write_sectors=wb/32,lines=(rb+wb)/128)
    if fixture=='store':
        wb={'below_l2':3080192,'above_l2':788529152}[regime]*(2 if cand=='coalesced_dense' else 1)
        stride={'coalesced_s1':1,'coalesced_dense':1,'strided_s7':7,'strided_s33':33}[cand]
        sec,lin=store_pattern(stride);requests=wb/128
        return dict(read_bytes=0,write_bytes=wb,read_sectors=0,write_sectors=requests*sec,lines=requests*lin)
    raise KeyError(fixture)

def main():
    W=windows()
    t0=float(np.median([W[('triad','triad','tiny')],W[('transpose','transpose','tiny')],0.70]))  # 0.70: floor design per-launch (revised correctness run)
    fit_keys=[('triad','c1','below_l2'),('store','coalesced_s1','below_l2'),('store','strided_s7','below_l2'),('store','strided_s33','below_l2')]
    A=[];y=[]
    for k in fit_keys:
        c=counts(*k,W);A.append([c['read_sectors']*32,c['write_sectors']*32,c['lines']]);y.append(W[k]-t0)
    A=np.array(A);y=np.array(y)
    coef,*_=np.linalg.lstsq(A,y,rcond=None)  # us per byte (read sector), us per byte (write sector), us per line
    inv_r,inv_w,c_line=coef
    if min(coef)<=0:print('WARNING: nonpositive fitted coefficient',coef)
    # DRAM stage: store above (write only) then triad above (read + write).
    cs=counts('store','coalesced_s1','above_l2',W);inv_dw=(W[('store','coalesced_s1','above_l2')]-t0)/cs['write_bytes']
    ct=counts('triad','c1','above_l2',W);inv_dr=(W[('triad','c1','above_l2')]-t0-ct['write_bytes']*inv_dw)/ct['read_bytes']
    const=dict(t0_us=t0,L2_read_sector_TBps=1/inv_r/1e6,L2_write_sector_TBps=1/inv_w/1e6,c_line_ns=c_line*1e3,
        DRAM_read_TBps=1/inv_dr/1e6,DRAM_write_TBps=1/inv_dw/1e6,latency_ns={'L1':16,'L2':134,'DRAM':311},
        source=str(SRC.relative_to(REPO)),active_sms_microbench=188)
    def pred(k):
        c=counts(*k,W);l2=c['read_sectors']*32*inv_r+c['write_sectors']*32*inv_w+c['lines']*c_line
        dram=c['read_bytes']*inv_dr+c['write_bytes']*inv_dw if k[2]=='above_l2' else 0
        return t0+max(l2,dram)
    report={}
    for k in sorted(W):
        try:p=pred(k)
        except KeyError:continue
        report['/'.join(k)]={'measured_us':round(W[k],3),'pred_us':round(p,3),'err_pct':round((p/W[k]-1)*100,1),'role':'fit' if k in fit_keys or k in [('store','coalesced_s1','above_l2'),('triad','c1','above_l2')] else 'heldout_check'}
    out={'constants':const,'microbenchmark_fit':report}
    Path(__file__).with_name('stream_constants.json').write_text(json.dumps(out,indent=1)+'\n');return out

if __name__=='__main__':
    o=main();print(json.dumps(o,indent=1))
