"""Orchestration of the energy stage: build the SASS count table once, run every window once (no retry), analyse each window, fit, and report.
`run` and `wait_idle` are injectable so the whole stage is testable on a CPU with stubs."""
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import capsafe as CS
from . import energy as En
from . import runner


def find_cuobjdump(nvcc):
    for c in (Path(nvcc).parent / 'cuobjdump', shutil.which('cuobjdump')):
        if c and Path(c).is_file(): return str(c)
    raise RuntimeError('cuobjdump not found next to nvcc or on PATH (needed to count the real SASS instructions of the energy kernels)')


def sass_table(binary, nvcc, run=subprocess.run):
    """cuobjdump the binary once; return {design params tuple: function lines}. Raises if any instantiation of the design table is missing."""
    r = run([find_cuobjdump(nvcc), '-sass', str(binary)], capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError('cuobjdump failed: %s' % (r.stderr or '')[-300:])
    table = {}
    for sym, lines in En.split_functions(r.stdout).items():
        p = En.design_of_symbol(sym)
        if p: table[p] = lines
    missing = sorted({n for n, d in En.DESIGNS.items() if d[:9] not in table})
    if missing: raise RuntimeError('SASS of design(s) not found in the binary: ' + ', '.join(missing))
    return table, r.stdout


def read_samples(path):
    rows = list(csv.DictReader(open(path)))
    arr = np.array([[int(r['monotonic_ns']), int(r['board_power_mw'])] for r in rows], dtype=float)
    clk = np.array([int(r['graphics_clock_mhz']) for r in rows]); temp = np.array([int(r['temperature_c']) for r in rows]); t = arr[:, 0]
    return arr, clk, temp, t


def trace_summary(row, clk, temp, t):
    inside = (t >= row['t_begin_ns']) & (t <= row['t_end_ns'])
    if not inside.any(): raise ValueError('no samples inside the window')
    return dict(temp_start=int(temp[0]), temp_end=int(temp[inside][-1]), clk_min=int(clk[inside].min()), clk_med=float(np.median(clk[inside])), clk_max_run=int(clk.max()))


def run_energy_stage(binary, nvcc, uuid, booking, out_dir, wait_idle, run_program, run=subprocess.run, warmup_s=En.WARMUP_S, window_s=En.WINDOW_S, only=None, timeout_s=400, cap_fraction=CS.CAP_SAFE_FRACTION):
    """Returns (stage_record, constants_or_None). Every window runs once; a failed window is recorded and excluded, never retried.
    cap_fraction: the cap-safe rule of CAP_SAFE_WINDOWS.md (probe, and reduce the grid so the window stays below this fraction of the enforced limit); None = previous full-grid behaviour (tests only)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    table, sass_text = sass_table(binary, nvcc, run)
    (out_dir / 'sass.txt').write_text(sass_text)
    windows, failures, verified = [], [], {}
    cap_w = None; chosen_blocks = {}
    for wid, design, role in En.WINDOWS:
        if only and wid not in only: continue
        wdir = out_dir / wid; wdir.mkdir(parents=True, exist_ok=True)
        rec = dict(id=wid, design=design, role=role)
        try:
            idle = wait_idle(uuid)
            forced = chosen_blocks.get(design) if wid != design else None   # the repeated anchor reuses the grid chosen for its first window
            r = run_program(binary, [design, warmup_s, window_s, str(wdir)] + CS.window_args(cap_fraction, forced), uuid, booking, timeout_s)
            (wdir / 'stdout.txt').write_text(r['stdout']); (wdir / 'stderr.txt').write_text(r['stderr'])
            rows = [x for x in runner.parse_jsonl(r['stdout']) if x.get('mode') == 'energy_window']
            if not rows: raise ValueError('no window row (return code %s): %s' % (r['returncode'], (r['stderr'] or '').strip()[-300:]))
            row = rows[0]
            if tuple(row[k] for k in 'RFAIS') + tuple(row[k] for k in ('H', 'B', 'Z', 'W')) != En.DESIGNS[design][:9]: raise ValueError('program parameters differ from the design table')
            params = En.DESIGNS[design][:9]
            cnt = En.count_kernel(table[params], row['trips_per_launch'])
            bad = En.verify_design(design, cnt['in_loop'])
            if bad: raise ValueError('SASS does not match the design: ' + '; '.join(bad))
            verified[design] = dict(in_loop=cnt['in_loop'], out_of_loop=cnt['out_of_loop'], loop_opcodes=cnt['loop_opcodes'])
            arr, clk, temp, t = read_samples(wdir / 'samples.csv')
            cols = En.columns_from_counts(cnt['per_thread'], row['lanes'], row['read_bytes_per_launch'], row['write_bytes_per_launch'], row['in_tier'], row['out_tier'])
            cap_w = row['power_limit_mw'] / 1000.0 if row['power_limit_mw'] else cap_w
            if not cap_w: raise ValueError('NVML reported no enforced power limit')
            a = En.analyse_window(wid, row, arr, trace_summary(row, clk, temp, t), cols, cap_w)
            if (row.get('cap_safe') or {}).get('enabled') and wid == design: chosen_blocks[design] = row['blocks']
            a.update(role=role, idle_before=idle, program_return_code=r['returncode'], seconds=r['seconds']); windows.append(a); rec.update(status='ok' if a['admitted'] else 'not_admitted', checks=a['checks'])
        except Exception as ex:  # noqa: BLE001 - every window failure is recorded, never raised past the stage
            rec.update(status='failed', reason=str(ex)[:400]); failures.append(rec)
        (wdir / 'record.json').write_text(json.dumps(rec, indent=1))
    (out_dir / 'windows.json').write_text(json.dumps(windows, indent=1))
    fit_w = [w for w in windows if w['role'] == 'fit']; held = [w for w in windows if w['role'] == 'heldout']
    stage = dict(name='energy', windows_total=len(En.WINDOWS) if not only else len(only), windows_ran=len(windows) + len(failures), failed=failures,
                 not_admitted=[w['id'] for w in windows if not w['admitted']], status='ok')
    expected_fit = sum(1 for w in En.WINDOWS if w[2] == 'fit' and (not only or w[0] in only))
    constants = None
    try:
        profile = En.fit_rates(fit_w)
        unident = [c for c, s in profile['status'].items() if s.startswith('UNIDENTIFIED')]
        stage['unidentified'] = unident
        if unident or len(fit_w) < expected_fit or any(not w['admitted'] for w in fit_w): stage['status'] = 'incomplete'
        anchor = {}
        first = next((w for w in windows if w['id'] == 'fma64'), None); rep = next((w for w in windows if w['id'] == 'fma64_repeat'), None)
        if first and rep: anchor = dict(first_energy_j=first['energy_j_per_launch'], repeat_energy_j=rep['energy_j_per_launch'], ratio_repeat_to_first=rep['energy_j_per_launch'] / first['energy_j_per_launch'],
                                         temp_first_end_c=first['temp_end_c'], temp_repeat_end_c=rep['temp_end_c'])
        constants = dict(cap_w=cap_w, profile=profile, heldout_error_pct=En.heldout_errors(profile, held, cap_w), thermal_anchor=anchor, sass_verified=verified,
                         windows=[{k: w[k] for k in ('id', 'design', 'role', 'energy_j_per_launch', 'runtime_s_per_launch', 'power_w', 'temp_start_c', 'temp_end_c', 'clock_median_mhz', 'admitted', 'columns') + (('cap_safe',) if 'cap_safe' in w else ())} for w in windows])
    except ValueError as ex:
        stage['status'] = 'failed'; stage['reason'] = str(ex)
    if failures and stage['status'] == 'ok': stage['status'] = 'incomplete'
    return stage, constants
