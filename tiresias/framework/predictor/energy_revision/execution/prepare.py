"""Emit a SOURCE-ONLY calibration packet. Does not compile, execute or admit a fit."""
import hashlib
import json
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
MIXES = ('memory', 'arithmetic', 'special_function', 'other_shared')
FOOTPRINTS = ('small_candidate', 'l2_candidate', 'dram_candidate')
FIT_IDS = tuple(f'{mix}/{dose}/{foot}' for mix in MIXES for dose in ('low', 'high') for foot in FOOTPRINTS)
HOLD_IDS = ('mixed/small_candidate', 'mixed/l2_candidate', 'mixed/dram_candidate',
            'mixed_altered/dram_candidate', 'anchor_memory/l2_candidate', 'anchor_arithmetic/small_candidate')


def packet():
    sys.path.insert(0,str(HERE.parents[1]))
    import extract_features as hardware
    text=(ROOT/'HARDWARE_GROUND_TRUTH.md').read_text()
    facts=hardware.load_hardware(text)
    section=text.split('## Blackwell',1)[1].split('\n## ',1)[0]
    if facts['sm_count']!=188 or facts['l2_bytes']!=134217728 or '| Power limit (current/default/max) | 600 W |' not in section:
        raise ValueError('REFUSED: source constants no longer match authoritative Blackwell ground truth')
    rows = []
    for did in FIT_IDS + HOLD_IDS:
        parts = did.split('/'); mix = parts[0]; foot = parts[-1]
        dose = 64 if len(parts) == 3 and parts[1] == 'high' else 16
        if mix == 'anchor_memory': mix = 'memory'
        if mix == 'anchor_arithmetic': mix = 'arithmetic'
        lanes = 188 * 256; l2 = 134217728
        target = dict(small_candidate=lanes*4,l2_candidate=l2//4,dram_candidate=l2*4)[foot]
        n = ((target+lanes*4-1)//(lanes*4) if foot == 'dram_candidate' else target//(lanes*4))*lanes
        repeat=dose if mix=='memory' else 1
        rows.append(dict(design_id=did,role='fit' if did in FIT_IDS else 'heldout',mix=mix,dose=dose,
            footprint_candidate=foot,input_bytes=(n+repeat-1)*4,n=n,grid=[188,1,1],block=[256,1,1],cache_policy='ld.global.cg',
            logical_bytes=n*4*repeat+12*lanes,touched_bytes=(n+repeat-1)*4+12*lanes,
            tier='DRAM' if foot == 'dram_candidate' else 'L2',argv_prefix=[mix,str(dose),foot,'CELL_OUTPUT.bin'],
            count_status='PENDING_COMPILE_AND_ABI_AUDIT',abi_status='PENDING',work=None))
    out = dict(schema='energy_component_calibration_packet/1',status='SOURCE_READY_NOT_COMPILED_OR_ADMITTED',
        gpu_index=1,gpu_uuid='GPU-0e63baea-0bb1-e90c-f19d-8aa5f2995894',cap_w=600,rows=rows,
        energy_windows_maximum=30,attempts_per_slot=1,counted_target_s=15,precondition_target_s=90,planned_device_floor_s=3150,
        compile=dict(nvcc='/usr/local/cuda-13.2/bin/nvcc',flags=['-O3','-std=c++17','-arch=sm_120','-Xptxas=-v'],cpu_only_env={'CUDA_VISIBLE_DEVICES':''},priority='nice -n 19, one compiler at a time'),
        count_gate='Complete actual compiled lane-instruction census and ABI/trip proof; six-column rank/conditioning after real calibration time is known; no transplanted hand counts',
        correctness_gate='All 30 configurations, full output census, immutable inputs, exact uint/checksum and FP32 FMA oracle; dependent bounded SFU recurrence with 2 device +1 CPU ulp per step, geometrically propagated bound of 10 ulp',
        small_candidate_qualification='Not verified L1. cg bypasses L1; small and L2 rows share lookup coefficient. No L1 inference or physical L1 coefficient claimed.',
        traffic_qualification='Declared logical tier/byte proxy from footprint/cache policy; not observed physical hierarchy traffic',
        booking='Wait for every open GPU 1 booking to close; exact packet booking in the booking log pushed; fresh idle/UUID check; serial full grid; no GPU 0 or settings changes',
        sfu_reference='https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-ex2',sfu_reference_checked_date='2026-10-02',
        cost_qualification='52.5 minutes is a device-time floor only. Full CPU oracles, probes, idle brackets, compilation and transfers are additional and presently unmeasured.')
    paths = [Path(__file__), HERE/'compile.py',HERE/'calibration.cu',HERE/'oracle.hpp',HERE/'count.py',HERE/'acquire.py',HERE/'fit.py',HERE/'fit_policy.json',HERE.parent/'component_candidate.py',HERE.parents[1]/'fresh_e/gpu/drivers/driver_common.h',HERE.parents[1]/'extract_features.py',ROOT/'HARDWARE_GROUND_TRUTH.md']
    out['inputs_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'calibration_packet.json')
    args=parser.parse_args()
    out = packet()
    with args.out.open('x') as f:
        json.dump(out,f,indent=2,sort_keys=True); f.write('\n')
    print('30 explicit source-ready slots; compilation/count/ABI/correctness gates pending')
