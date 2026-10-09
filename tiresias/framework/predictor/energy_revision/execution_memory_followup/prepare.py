"""Source-only two-window proposal; no execution or replacement fitting."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
FIT_IDS=('memory/low/dram_candidate','memory/high/dram_candidate')
HOLD_IDS=()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    source=HERE.parent/'execution/calibration_packet_sfu_sink.json'
    base=json.loads(source.read_text())
    rows=[dict(r,dependent_checksum_steps_per_memory_load=9) for r in base['rows'] if r['design_id'] in FIT_IDS]
    files=[HERE/name for name in ('prepare.py','calibration.cu','oracle.hpp','compile.py','count.py','acquire.py','fit_followup.py','fit_policy.json')]
    files += [ROOT/'HARDWARE_GROUND_TRUTH.md',ROOT/'energy_harness/application_energy_harness.py',
              ROOT/'energy_harness/measurement_runner.py',ROOT/'energy_harness/nvml_sampler.py',ROOT/'energy_harness/verify_b_stabilization_trace.py',
              HERE.parent/'execution/fit.py',HERE.parent/'component_candidate.py',
              HERE.parents[1]/'fresh_e/gpu/drivers/driver_common.h',HERE.parents[1]/'port_common/port_ext.py',
              HERE.parents[1]/'extract_features.py']
    # Same already-verified footprints, GPU, ABI and doses as the initial grid;
    # hardware file must still yield the same Blackwell constants.
    import sys
    sys.path.insert(0,str(HERE.parents[1]))
    import extract_features as X
    text=(ROOT/'HARDWARE_GROUND_TRUTH.md').read_text();hw=X.load_hardware(text)
    section=text.split('## Blackwell',1)[1].split('\n## ',1)[0]
    if hw['sm_count']!=188 or hw['l2_bytes']!=134217728 or '| Power limit (current/default/max) | 600 W |' not in section:
        raise ValueError('REFUSED: geometry/cap truth changed')
    doc=dict(base,rows=rows,status='SOURCE_READY_NOT_COMPILED_OR_ADMITTED',
             schema='energy_component_calibration_packet/1',max_energy_windows=2,energy_windows_maximum=2,
             planned_device_floor_s=210,correctness_gate='Both replacement configurations, full output arrays and versioned CPU oracle',
             cost_qualification='3.5-minute new device-time floor, plus serial compilation/counts, two CPU oracle gates, probes and cooldown. Original 30 windows remain charged.',
             inputs_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
             dose_change='Memory checksum folds nine times per load instead of once; all added native work is counted. No machine-setting change.',
             prerequisite='Original 30 windows complete and accepted; exactly these two controls fail the 95%-cap fit gate; all other 28 pass it. Otherwise refuse this proposal.',
             retained_initial_windows=28,total_acquired_if_completed=32,fit_windows=24,heldout_windows=6,
             excluded_initial_controls=list(FIT_IDS),exclusion_reason='Explicit retirement of two near-cap calibration controls; never a silent exclusion or target-driven fit.',
             window_budget=dict(counted_s_per_slot=15,precondition_s_per_slot=90,device_time_floor_s=210),
             approval='NEW shared-machine approval required after initial booking closes; compile/count/two correctness gates then maximum two energy windows.')
    with a.out.open('x') as f:json.dump(doc,f,indent=2,sort_keys=True);f.write('\n')
    print('Two source-only replacement controls; new approval and native gates required')


if __name__=='__main__':main()
