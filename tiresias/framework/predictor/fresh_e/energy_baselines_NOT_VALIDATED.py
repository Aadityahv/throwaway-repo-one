"""Published-method energy baselines applied to new cells with their FROZEN constants (nothing refitted). Forms from
tiresias/app_runners/BASELINE_HEADTOHEAD_PROTOCOL_2026-09-26.md (operator forms); constants from reset/baseline_headtohead_results_blackwell.json (X2c fit on 22
Blackwell calibration windows) and reset/baseline_headtohead_constants.json (X3 literature constants). Static inputs: logical bytes and tier of the cell and the exact
static lane-instruction count of the launch. NOT the published tools: X2c is an AccelWattch-style component model re-implemented from its published structure
(never called AccelWattch); X3 is bytes x literature pJ per bit (O'Connor 2017 for DRAM, Keckler 2011 on chip).
 X2c:  P = min(cap, P_const + e_inst * inst / t + e_B * bytes / t + e_DRAM * dram_bytes / t),  E = P * t   (t: measured runtime, as in the baseline's own protocol, or
       our predicted runtime; both are reported)
 X3:   E = bytes * 8 * pJ_per_bit[tier] * 1e-12   (no time term, no cap)"""
import json
from pathlib import Path
RESET = Path(__file__).resolve().parents[3] / 'reset'
_P = json.loads((RESET / 'baseline_headtohead_results_blackwell.json').read_text())['part1_stride_grid']['x2c_fit']['params']
_X3 = json.loads((RESET / 'baseline_headtohead_constants.json').read_text())['X3']
CAP_W = 600.0
def x2c_energy(inst, bytes_, tier, t):
    dram = bytes_ if tier == 'DRAM' else 0.0
    return min(CAP_W, _P['P_const'] + _P['e_inst'] * inst / t + _P['e_B'] * bytes_ / t + _P['e_DRAM'] * dram / t) * t
def x3_energy(bytes_, tier):
    return bytes_ * 8 * _X3[tier] * 1e-12
