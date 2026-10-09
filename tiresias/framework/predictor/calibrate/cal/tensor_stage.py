"""Orchestration of the optional tensor stage: the `issue` cells of micro_tensor, then three energy windows (1, 4, 16 warps per SM), each run once (no retry), analysed with the
same window checks as the energy stage. `run` and `wait_idle` are injectable so the stage is testable on a CPU with stubs."""
import json
import subprocess
from pathlib import Path

from . import capsafe as CS
from . import energy as En
from . import runner
from . import tensor as Tn
from .energy_stage import find_cuobjdump, read_samples, trace_summary


def run_tensor_stage(binary, nvcc, uuid, booking, out_dir, wait_idle, run_program, run=subprocess.run, warmup_s=En.WARMUP_S, window_s=En.WINDOW_S, only=None, timeout_s=400, cap_fraction=CS.CAP_SAFE_FRACTION):
    """Returns (stage_record, raw). raw = dict(issue=<issue.json>, windows=[analysed windows], sass_verified=...)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stage = dict(name='tensor', status='ok', failed=[], not_admitted=[]); raw = dict(issue=None, windows=[], sass_verified={})
    r = run([find_cuobjdump(nvcc), '-sass', str(binary)], capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError('cuobjdump failed: %s' % (r.stderr or '')[-300:])
    (out_dir / 'sass.txt').write_text(r.stdout); funcs = En.split_functions(r.stdout)
    sym = next((s for s in funcs if s.startswith('_Z10mma_energyILi%dE' % Tn.M_PER_TRIP)), None)
    if sym is None: raise RuntimeError('SASS of the tensor energy kernel not found in the binary')
    # ---- issue cells
    idir = out_dir / 'issue'; idir.mkdir(parents=True, exist_ok=True)
    try:
        wait_idle(uuid); p = run_program(binary, ['issue', str(idir)], uuid, booking, timeout_s)
        (idir / 'stdout.txt').write_text(p['stdout']); (idir / 'stderr.txt').write_text(p['stderr'])
        if p['returncode'] != 0 or not (idir / 'issue.json').exists(): raise ValueError('issue run failed (return code %s): %s' % (p['returncode'], (p['stderr'] or '').strip()[-300:]))
        raw['issue'] = json.loads((idir / 'issue.json').read_text())
    except Exception as ex:  # noqa: BLE001
        stage['failed'].append(dict(id='issue', reason=str(ex)[:400]))
    # ---- energy windows
    cap_w = None
    for wid in Tn.WINDOWS:
        if only and wid not in only: continue
        wdir = out_dir / wid; wdir.mkdir(parents=True, exist_ok=True); rec = dict(id=wid)
        try:
            idle = wait_idle(uuid); p = run_program(binary, ['energy', wid, warmup_s, window_s, str(wdir)] + CS.window_args(cap_fraction), uuid, booking, timeout_s)
            (wdir / 'stdout.txt').write_text(p['stdout']); (wdir / 'stderr.txt').write_text(p['stderr'])
            rows = [x for x in runner.parse_jsonl(p['stdout']) if x.get('mode') == 'tensor_energy_window']
            if not rows: raise ValueError('no window row (return code %s): %s' % (p['returncode'], (p['stderr'] or '').strip()[-300:]))
            row = rows[0]; cnt = En.count_kernel(funcs[sym], row['trips_per_launch']); bad = Tn.verify_design(cnt['in_loop'], row['M'])
            if bad: raise ValueError('SASS does not match the design: ' + '; '.join(bad))
            raw['sass_verified'][wid] = dict(in_loop=cnt['in_loop'], out_of_loop=cnt['out_of_loop'], loop_opcodes=cnt['loop_opcodes'])
            arr, clk, temp, t = read_samples(wdir / 'samples.csv')
            cols = En.columns_from_counts(cnt['per_thread'], row['lanes'], row['read_bytes_per_launch'], row['write_bytes_per_launch'], 'small', 'none')
            cap_w = row['power_limit_mw'] / 1000.0 if row['power_limit_mw'] else cap_w
            if not cap_w: raise ValueError('NVML reported no enforced power limit')
            a = En.analyse_window(wid, row, arr, trace_summary(row, clk, temp, t), cols, cap_w)
            a.update(role='tensor', idle_before=idle, program_return_code=p['returncode'], seconds=p['seconds']); raw['windows'].append(a); rec.update(status='ok' if a['admitted'] else 'not_admitted', checks=a['checks'])
            if not a['admitted']: stage['not_admitted'].append(wid)
        except Exception as ex:  # noqa: BLE001
            rec.update(status='failed', reason=str(ex)[:400]); stage['failed'].append(rec)
        (wdir / 'record.json').write_text(json.dumps(rec, indent=1))
    (out_dir / 'windows.json').write_text(json.dumps(raw['windows'], indent=1))
    if stage['failed'] or stage['not_admitted']: stage['status'] = 'incomplete'
    return stage, raw
