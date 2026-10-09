"""Energy stage of the packaged calibrator: window designs, SASS-based instruction counting, power-trace integration, admission, the calibration-only fit and the energy predictor.

Everything here is CPU-only and testable without a GPU (the GPU part is csrc/micro_energy.cu). Design principles:
 * The rates are fitted from synthetic windows only. No application label is read.
 * Instruction counts are taken from the SASS of the binary that actually ran, grouped by the same opcode families the application feature extractor uses
   (`extract_features.op_family`, copied below with a test that keeps them identical), so a calibration window and an application kernel are counted the same way.
 * A window that fails any check is kept in the document and excluded from the fit; a rate with fewer than two admitted supporting windows is reported UNIDENTIFIED and
   makes the document incomplete. Nothing is defaulted, and no other board's number is ever substituted.
"""
import math
import re

import numpy as np
from scipy.optimize import nnls

from . import capsafe as CS

# ---------------------------------------------------------------- design table (mirrors csrc/micro_energy.cu; a test keeps the two identical)
# name: (R loads, F fma, A fp add/sub, I int, S sfu, H shared st+ld, B shuffle, Z barrier, W stores, input tier, output tier, light geometry)
DESIGNS = {
    'fma16': (1, 16, 0, 0, 0, 0, 0, 0, 0, 'small', 'none', 0), 'fma64': (1, 64, 0, 0, 0, 0, 0, 0, 0, 'small', 'none', 0),
    'fpadd16': (1, 0, 16, 0, 0, 0, 0, 0, 0, 'small', 'none', 0), 'fpadd64': (1, 0, 64, 0, 0, 0, 0, 0, 0, 'small', 'none', 0),
    'int16': (1, 0, 0, 16, 0, 0, 0, 0, 0, 'small', 'none', 0), 'int64': (1, 0, 0, 64, 0, 0, 0, 0, 0, 'small', 'none', 0),
    'sfu16': (1, 0, 0, 0, 16, 0, 0, 0, 0, 'small', 'none', 0), 'sfu64': (1, 0, 0, 0, 64, 0, 0, 0, 0, 'small', 'none', 0),
    'shared16': (1, 0, 0, 0, 0, 16, 0, 0, 0, 'small', 'none', 0), 'shared64': (1, 0, 0, 0, 0, 64, 0, 0, 0, 'small', 'none', 0),
    'shfl16': (1, 0, 0, 0, 0, 0, 16, 0, 0, 'small', 'none', 0), 'shfl64': (1, 0, 0, 0, 0, 0, 64, 0, 0, 'small', 'none', 0),
    'bar16': (1, 0, 0, 0, 0, 0, 0, 16, 0, 'small', 'none', 0), 'bar64': (1, 0, 0, 0, 0, 0, 0, 64, 0, 'small', 'none', 0),
    'rd_l2_8': (8, 0, 0, 0, 0, 0, 0, 0, 0, 'l2', 'none', 0), 'rd_l2_16': (16, 0, 0, 0, 0, 0, 0, 0, 0, 'l2', 'none', 0),
    'rd_dram_8': (8, 0, 0, 0, 0, 0, 0, 0, 0, 'dram', 'none', 0), 'rd_dram_16': (16, 0, 0, 0, 0, 0, 0, 0, 0, 'dram', 'none', 0),
    'wr_l2_8': (1, 0, 0, 0, 0, 0, 0, 0, 8, 'small', 'l2', 0), 'wr_dram_8': (1, 0, 0, 0, 0, 0, 0, 0, 8, 'small', 'dram', 0),
    'wr_dram_16': (1, 0, 0, 0, 0, 0, 0, 0, 16, 'small', 'dram', 0), 'copy_dram_8': (8, 0, 0, 0, 0, 0, 0, 0, 8, 'dram', 'dram', 0),
    'light_int64': (1, 0, 0, 64, 0, 0, 0, 0, 0, 'small', 'none', 1),
    'mixed_a': (1, 8, 0, 0, 8, 8, 0, 0, 0, 'small', 'none', 0), 'mixed_b': (1, 24, 0, 0, 4, 12, 0, 0, 0, 'small', 'none', 0),
    'mixed_app': (8, 0, 8, 8, 0, 0, 4, 4, 0, 'dram', 'none', 0),
}
# Window plan: (window id, design, role). Order runs from cool to hot work; the repeated anchor at the end measures thermal carry-over directly.
WINDOWS = [(d, d, 'fit') for d in ('fma16', 'fpadd16', 'int16', 'sfu16', 'shared16', 'shfl16', 'bar16', 'fma64', 'fpadd64', 'int64', 'sfu64', 'shared64', 'shfl64', 'bar64', 'light_int64')] + \
          [('mixed_a', 'mixed_a', 'heldout'), ('mixed_b', 'mixed_b', 'heldout')] + \
          [(d, d, 'fit') for d in ('rd_l2_8', 'rd_l2_16', 'wr_l2_8', 'rd_dram_8', 'rd_dram_16', 'wr_dram_8', 'wr_dram_16', 'copy_dram_8')] + \
          [('mixed_app', 'mixed_app', 'heldout'), ('fma64_repeat', 'fma64', 'heldout')]
WARMUP_S, WINDOW_S = 60.0, 20.0
ADMIT_FRACTION_OF_CAP = 0.985
CAP_SAFE_FRACTION = CS.CAP_SAFE_FRACTION   # window power target (calibrate/CAP_SAFE_WINDOWS.md); the admission rule above is unchanged
COLUMNS = ['B_l2', 'B_dr', 'B_wr', 'ffma', 'fpo', 'intc', 'sfu', 'shm', 'shfl', 'bar']


# ---------------------------------------------------------------- SASS
def op_family(op):
    """Copy of predictor/extract_features.op_family (family only); test_energy.py checks it stays identical."""
    if op.startswith('LDGSTS'): return 'async_copy_global_to_shared'
    if op.startswith('LDGDEPBAR') or op.startswith('DEPBAR'): return 'dependency_barrier'
    if op.startswith('LDG'): return 'global_load'
    if op.startswith('STG'): return 'global_store'
    if op.startswith(('HMMA', 'IMMA', 'QMMA', 'DMMA')): return 'tensor_core'   # warp-level matrix multiply-accumulate; priced only through the optional tensor stage
    if op.startswith('LDSM'): return 'shared_matrix_load'
    if op.startswith('LDS'): return 'shared_load'
    if op.startswith('STS'): return 'shared_store'
    if op.startswith('SHFL'): return 'shuffle'
    if op.startswith('MUFU'): return 'special_function'
    if op.startswith('BAR'): return 'barrier'
    if op == 'FFMA' or op == 'UFFMA': return 'fp32_fma'
    if op in ('FADD', 'UFADD'): return 'fp32_add'
    if op in ('FMUL', 'UFMUL'): return 'fp32_mul'
    if op.startswith(('FMNMX', 'FSEL', 'FSETP', 'UFSEL', 'UFSETP', 'HFMA2', 'F2I', 'I2F', 'I2FP', 'UI2FP')): return 'fp_other'
    if op.startswith(('IADD', 'UIADD', 'IMAD', 'UIMAD', 'LEA', 'ULEA', 'LOP3', 'ULOP3', 'SHF', 'USHF', 'ISETP', 'UISETP', 'SEL', 'SGXT', 'IMNMX', 'POPC', 'FLO', 'PRMT')): return 'integer_alu'
    if op.startswith(('BRA', 'BSSY', 'BSYNC', 'WARPSYNC', 'ENDCOLLECTIVE', 'EXIT', 'RET', 'CALL', 'JMP')): return 'control'
    if op.startswith(('LDC', 'LDCU', 'S2R', 'S2UR', 'CS2R', 'R2UR', 'MOV', 'UMOV', 'P2R', 'R2P', 'PLOP3', 'UPLOP3', 'NOP')): return 'move_const_special'
    return 'other'


FUNC_RE = re.compile(r'^\s*Function\s*:\s*(\S+)')
INSTR_RE = re.compile(r'^\s*/\*([0-9a-fA-F]{4,})\*/\s+(@!?U?P[T0-9]\s+)?([A-Z][A-Z0-9_]*(?:\.[A-Za-z0-9_]+)*)\b([^;]*);')
TARGET_RE = re.compile(r'\b(0x[0-9a-fA-F]+)\b')
TEMPLATE_RE = re.compile(r'energy_fixtureI((?:Li\d+E){9})E')


def split_functions(sass_text):
    funcs, cur = {}, None
    for line in sass_text.splitlines():
        m = FUNC_RE.match(line)
        if m: cur = m.group(1); funcs[cur] = []; continue
        if cur is not None: funcs[cur].append(line)
    return funcs


def design_of_symbol(symbol):
    m = TEMPLATE_RE.search(symbol)
    if not m: return None
    return tuple(int(x) for x in re.findall(r'Li(\d+)E', m.group(1)))


def parse_kernel(lines):
    """Instructions of one function as (address, guard, opcode, operand text). cuobjdump prints branch targets as absolute addresses."""
    instrs = []
    for line in lines:
        im = INSTR_RE.match(line)
        if im: instrs.append((int(im.group(1), 16), (im.group(2) or '').strip(), im.group(3), im.group(4)))
    return instrs


def count_kernel(sass_lines, trips):
    """Dynamic lane-instruction counts PER THREAD for one launch: instructions outside the loop once, inside the single backward loop `trips` times.
    Refuses unless the kernel has exactly one backward branch, no spill (LDL/STL), and no predicate-guarded instruction inside the loop other than the loop branch."""
    instrs = parse_kernel(sass_lines)
    if not instrs: raise ValueError('no SASS instructions parsed')
    back = []
    for addr, guard, op, rest in instrs:
        if op.startswith('BRA'):
            t = TARGET_RE.search(rest)
            if t and int(t.group(1), 16) < addr: back.append((int(t.group(1), 16), addr))
    if len(back) != 1: raise ValueError('expected exactly one backward branch (the trip loop), found %d' % len(back))
    start, end = back[0]
    if any(op.startswith(('LDL', 'STL')) for _, _, op, _ in instrs): raise ValueError('register spill (LDL/STL) in kernel')
    inloop = {}; outloop = {}; opcodes = {}
    for addr, guard, op, rest in instrs:
        if guard in ('@!PT', '@!UPT'): continue   # a never-true guard: the instruction is never executed (a compiler artifact), so it is not counted
        fam = op_family(op)
        inside = start <= addr <= end
        if inside and guard and not (op.startswith('BRA') and addr == end): raise ValueError('predicate-guarded instruction inside the loop: %s %s' % (guard, op))
        (inloop if inside else outloop)[fam] = (inloop if inside else outloop).get(fam, 0) + 1
        if inside: opcodes[op] = opcodes.get(op, 0) + 1
    fams = sorted(set(inloop) | set(outloop))
    return dict(per_thread={f: outloop.get(f, 0) + trips * inloop.get(f, 0) for f in fams}, in_loop=inloop, out_of_loop=outloop, loop_opcodes=opcodes)


def expected_in_loop(p):
    """What the design promises per loop iteration (exact for the priced classes, lower bound for integer work)."""
    R, F, A, I, S, H, B, Z, W = p[:9]
    return dict(fp32_fma=R * F, fp32_add=R * A, special_function=R * S, shared_store=R * H, shared_load=R * H, shuffle=R * B, barrier=R * Z, global_load=R, global_store=W)


def verify_design(name, in_loop):
    """Return a list of mismatches between the real SASS loop body and the design (empty means verified)."""
    p = DESIGNS[name]; exp = expected_in_loop(p); bad = []
    for fam, n in exp.items():
        have = in_loop.get(fam, 0)
        if have != n: bad.append('%s: SASS %d per iteration, design %d' % (fam, have, n))
    for fam in ('fp32_fma', 'fp32_add', 'special_function', 'shared_store', 'shared_load', 'shuffle', 'barrier', 'global_store'):
        if exp[fam] == 0 and in_loop.get(fam, 0): bad.append('%s: SASS %d per iteration but the design has none' % (fam, in_loop[fam]))
    R, F, A, I = p[:4]
    if in_loop.get('integer_alu', 0) < R * I: bad.append('integer_alu: SASS %d per iteration, design needs at least %d' % (in_loop.get('integer_alu', 0), R * I))
    return bad


def columns_from_counts(per_thread, lanes, read_bytes, write_bytes, in_tier, out_tier):
    """Fit columns of one window per launch from the per-thread dynamic counts and the design bytes."""
    g = lambda *fs: float(sum(per_thread.get(f, 0) for f in fs)) * lanes
    tier = lambda t: 'DRAM' if t == 'dram' else 'L2'
    c = dict(B_l2=0.0, B_dr=0.0, B_wr=float(write_bytes if out_tier != 'none' else 0))
    for b, t in ((read_bytes, in_tier), (write_bytes, out_tier)):
        if t == 'none' or not b: continue
        c['B_dr' if tier(t) == 'DRAM' else 'B_l2'] += float(b)
    c.update(ffma=g('fp32_fma', 'fp32_mul'), fpo=g('fp32_add', 'fp_other'), intc=g('integer_alu', 'move_const_special', 'control', 'other'), sfu=g('special_function'),
             shm=g('shared_load', 'shared_store', 'shared_matrix_load'), shfl=g('shuffle'), bar=g('barrier'))
    c['tc'] = g('tensor_core')     # not a column of the main fit: the tensor rate is derived by the optional tensor stage (cal/tensor.py)
    return c


# ---------------------------------------------------------------- power trace
def integrate_energy(samples, t_begin_ns, t_end_ns):
    """Trapezoid integral of board power (W) over [t_begin, t_end] with linear interpolation at both ends. samples: array of (ns, mW). Returns (joules, info)."""
    s = np.asarray(samples, dtype=float)
    if len(s) < 20: raise ValueError('fewer than 20 power samples')
    t = s[:, 0]; p = s[:, 1] / 1000.0
    if t[0] > t_begin_ns or t[-1] < t_end_ns: raise ValueError('power trace does not cover the window')
    inside = (t >= t_begin_ns) & (t <= t_end_ns)
    pb = float(np.interp(t_begin_ns, t, p)); pe = float(np.interp(t_end_ns, t, p))
    tt = np.concatenate([[t_begin_ns], t[inside], [t_end_ns]]) * 1e-9; pp = np.concatenate([[pb], p[inside], [pe]])
    joules = float(np.sum((pp[1:] + pp[:-1]) / 2 * np.diff(tt)))
    gaps = np.diff(t[inside]) * 1e-9 if inside.sum() > 1 else np.array([1e9])
    return joules, dict(mean_power_w=joules / ((t_end_ns - t_begin_ns) * 1e-9), power_sd_w=float(p[inside].std()), max_gap_s=float(gaps.max()), samples_in_window=int(inside.sum()))


def analyse_window(window_id, row, samples_arr, temps_clocks, counts, cap_w):
    """Window record with energy per launch and every admission check. temps_clocks: dict(temp_start, temp_end, clk_min, clk_med, clk_max_run) from the trace."""
    joules, info = integrate_energy(samples_arr, row['t_begin_ns'], row['t_end_ns'])
    launches = row['launches']; checks = {}
    checks['sampler_ok'] = bool(row['nvml_sampler_ok']); checks['output_deterministic'] = bool(row['output_deterministic']); checks['output_nonzero'] = bool(row['output_nonzero'])
    checks['sample_gap_below_0.1s'] = info['max_gap_s'] < 0.1
    checks['power_below_%.1f_percent_of_cap' % (100 * ADMIT_FRACTION_OF_CAP)] = info['mean_power_w'] < ADMIT_FRACTION_OF_CAP * cap_w
    checks['clock_not_throttled'] = temps_clocks['clk_med'] >= 0.9 * temps_clocks['clk_max_run']
    cs = row.get('cap_safe'); cs_problems = []
    if cs and cs.get('enabled'):
        cs_problems = CS.verify_record(cs, cap_w)
        if row['blocks'] != cs['blocks_chosen']: cs_problems.append('launched %d blocks, the cap-safe record says %d' % (row['blocks'], cs['blocks_chosen']))
        checks['cap_safe_choice_reproduced'] = not cs_problems
    if row.get('energy_counter_ok') and row['energy_counter_mj'] > 0:
        ratio = row['energy_counter_mj'] / 1000.0 / joules; checks['counter_agrees_within_5pct'] = abs(ratio - 1) < 0.05; info['energy_counter_ratio'] = round(ratio, 4)
    rec = dict(id=window_id, design=row['design'], energy_j_per_launch=joules / launches, runtime_s_per_launch=row['window_seconds'] / launches, power_w=info['mean_power_w'], **info,
               temp_start_c=temps_clocks['temp_start'], temp_end_c=temps_clocks['temp_end'], clock_min_mhz=temps_clocks['clk_min'], clock_median_mhz=temps_clocks['clk_med'],
               launches=launches, checks=checks, admitted=all(checks.values()), columns=counts)
    if cs and cs.get('enabled'): rec['cap_safe'] = cs; rec['blocks'] = row['blocks']; rec['lanes'] = row['lanes']
    if cs_problems: rec['cap_safe_problems'] = cs_problems
    return rec


# ---------------------------------------------------------------- fit
def fit_rates(windows, base_w=None):
    """windows: admitted fit windows (dicts with energy_j_per_launch, runtime_s_per_launch, columns). Non-negative least squares on relative error; free base power unless fixed."""
    rows = [w for w in windows if w['admitted']]
    support = {c: sum(1 for w in rows if w['columns'].get(c, 0) > 0) for c in COLUMNS}
    keep = [c for c in COLUMNS if support[c] >= 2]
    if len(rows) < len(keep) + 1: raise ValueError('fewer admitted windows (%d) than parameters (%d)' % (len(rows), len(keep) + 1))
    X = np.array([[w['columns'][c] * 1e-12 for c in keep] for w in rows]); E = np.array([w['energy_j_per_launch'] for w in rows]); t = np.array([w['runtime_s_per_launch'] for w in rows]); wt = 1 / E
    if base_w is None:
        A = np.hstack([X, t[:, None]]); sol, _ = nnls(A * wt[:, None], E * wt); base = float(sol[-1]); sol = sol[:-1]
    else:
        sol, _ = nnls(X * wt[:, None], (E - base_w * t) * wt); base = float(base_w)
    rates = {c: float(v) for c, v in zip(keep, sol)}
    status = {c: ('UNIDENTIFIED (fewer than 2 admitted windows)' if support[c] < 2 else ('fitted to zero' if rates[c] == 0 else 'ok')) for c in COLUMNS}
    pred = X @ sol + base * t
    return dict(base_power_w=base, rates_pJ=rates, status=status, support_windows=support, n_windows=len(rows), condition_number=float(np.linalg.cond(np.hstack([X, t[:, None]]) * wt[:, None])),
                residual_pct={w['design']: round(float((p / w['energy_j_per_launch'] - 1) * 100), 2) for w, p in zip(rows, pred)})


def predict_energy(profile, columns, runtime_s, cap_w):
    """Energy (J) of one launch sequence: min(cap * t, base * t + sum(rate * count)). columns: dict over COLUMNS (counts and bytes per launch, 0 if absent)."""
    dyn = sum(profile['rates_pJ'].get(c, 0.0) * columns.get(c, 0.0) * 1e-12 for c in COLUMNS)
    return min(cap_w * runtime_s, profile['base_power_w'] * runtime_s + dyn)


def heldout_errors(profile, windows, cap_w):
    out = {}
    for w in windows:
        if not w['admitted']: continue
        p = predict_energy(profile, w['columns'], w['runtime_s_per_launch'], cap_w); out[w['id']] = round(float((p / w['energy_j_per_launch'] - 1) * 100), 2)
    return out


def columns_from_feature_row(work, logical_bytes, tier, store_bytes):
    """Fit columns of an APPLICATION kernel from an extract_features row (`work['families']` lane counts) and its memory facts: used by the predictor and the evaluation."""
    F = work['families']; g = lambda *fs: float(sum(F.get(f, {}).get('lane_instructions', 0) for f in fs))
    c = dict(B_l2=0.0 if tier == 'DRAM' else float(logical_bytes), B_dr=float(logical_bytes) if tier == 'DRAM' else 0.0, B_wr=float(store_bytes or 0))
    c.update(ffma=g('fp32_fma', 'fp32_mul'), fpo=g('fp32_add', 'fp_other'), intc=g('integer_alu', 'move_const_special', 'control', 'other'), sfu=g('special_function'),
             shm=g('shared_load', 'shared_store', 'shared_matrix_load'), shfl=g('shuffle'), bar=g('barrier'))
    c['tc'] = g('tensor_core')   # only features built with a tensor_core family (fresh_g_lib.py) carry it; the frozen extractor files HMMA under `other`
    return c
