#!/usr/bin/env python3
"""One command per GPU: calibrate the static runtime model.

    python3 calibrate/run_calibration.py --device-uuid GPU-... --booking-ref "<the booking log booking entry>" [--plan] [--facts-json f.json] [--stages a,b] [--max-minutes 120]

`--plan` prints the device facts used, every stage with its expected duration, and the files that will be written; with --facts-json it touches no GPU at all, without it
it compiles and runs `device_facts` (a CUDA context only, no kernel). Without --plan it runs every stage once (no retry), writes the raw outputs with hashes and
`calibration_<arch>_<uuid8>.json`, and exports the legacy constants files. Exit codes: 0 complete, 1 refused, 3 ran but incomplete.
"""
import argparse
import datetime
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cal import device as D, document, energy_stage, fits, runner, stages, tensor, tensor_stage  # noqa: E402
from cal import TOOL_VERSION  # noqa: E402


def find_nvcc(arg):
    for c in (arg, os.environ.get('HARNESS_NVCC'), shutil.which('nvcc')):
        if c and Path(c).is_file(): return c
    raise D.Refusal('no nvcc found: pass --nvcc (the toolkit must support this GPU architecture)')


def live_facts(tool, entry_uuid, booking, out_dir, facts_hint, run=None):
    """Compile and run device_facts. facts_hint supplies the compute capability for the -arch flag only (taken from the allow-list entry)."""
    info = tool.compile('device_facts', facts_hint, out_dir / 'build')
    r = runner.run_program(info['binary'], [], entry_uuid, booking, 120)
    if r['returncode'] != 0: raise D.Refusal('device_facts refused: ' + r['stderr'].strip()[-300:])
    row = [x for x in runner.parse_jsonl(r['stdout']) if x.get('mode') == 'device_facts'][0]
    return row, info


def plan_text(facts, selected, tables):
    lines = ['Device facts used: %s, %s, compute capability %s, %d SMs, L2 %d bytes, shared memory per SM %d, max threads per SM %d' % (
        facts['name'], facts['uuid'], facts['compute_capability'], facts['sm_count'], facts['l2_bytes'], facts['shared_per_sm'], facts['max_threads_per_sm'])]
    lines.append('Pointer-chase tables (bytes per block): ' + json.dumps(tables))
    tot = 0
    for name, prog, args, minutes, what in stages.STAGES:
        if name not in selected: continue
        extra = '%d cells' % len(stages.pipes_grid(facts)) if name == 'pipes' else ' '.join(args or [])
        lines.append('  stage %-14s program %-14s %-14s ~%.1f min (Blackwell)  %s' % (name, prog, extra, minutes, what)); tot += minutes
    lines.append('Expected total about %.0f minutes (measured 11 on Blackwell; scales with the GPU), idle GPU required before every stage, no retries.' % tot)
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device-uuid', required=True); ap.add_argument('--booking-ref', required=True); ap.add_argument('--plan', action='store_true')
    ap.add_argument('--facts-json', type=Path, help='device_facts output to use instead of querying the device (CPU-only planning and tests)')
    ap.add_argument('--stages', default=','.join(s[0] for s in stages.STAGES)); ap.add_argument('--nvcc'); ap.add_argument('--out-dir', type=Path)
    ap.add_argument('--store-method', default='legacy', choices=fits.STORE_METHODS, help='derivation of the runtime model traffic rates (store_legacy): legacy (default, unchanged), stream_rates or guarded; see RATE_FIX_20261003.md')
    ap.add_argument('--dram-peak-tbps', type=float, help='verified DRAM peak of this board in TB/s from HARDWARE_GROUND_TRUTH.md (enables the peak check of the DRAM plausibility test; never computed by the tool)')
    ap.add_argument('--allow-passive-display-context', action='store_true', help='OPT-IN, default off: accept foreign display-server/compositor processes (Xorg, gnome-shell, ...; at most %d MiB each) on the approved GPU while its utilisation is 0%%%%; any other process refuses (see CAP_SAFE_WINDOWS.md)' % D.MAX_DISPLAY_CONTEXT_MIB)
    ap.add_argument('--max-minutes', type=float, default=120.0); ap.add_argument('--timeout-s', type=float, default=1800.0)
    a = ap.parse_args(argv)
    try:
        allow = D.load_allow_list(); entry = D.require_approved(a.device_uuid, allow); booking = D.require_booking(a.booking_ref)
        selected = [s for s in a.stages.split(',') if s]
        unknown = [s for s in selected if s not in {x[0] for x in stages.STAGES}]
        if unknown: raise D.Refusal('unknown stage(s): ' + ', '.join(unknown))
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out_dir = a.out_dir or HERE / 'runs' / ('%s_%s_%s' % (entry['machine'], a.device_uuid[4:12], stamp)); out_dir = out_dir.resolve()
        if a.facts_json:
            facts = json.loads(a.facts_json.read_text()); tool = None; facts_info = None
            if facts['uuid'] != a.device_uuid: raise D.Refusal('--facts-json describes %s, not %s' % (facts['uuid'], a.device_uuid))
        else:
            nvcc = find_nvcc(a.nvcc); tool = runner.Toolchain(nvcc); out_dir.mkdir(parents=True, exist_ok=True)
            hint = dict(compute_capability=entry['compute_capability'])
            facts, facts_info = live_facts(tool, a.device_uuid, booking, out_dir, hint)
        gt = D.parse_ground_truth(entry['ground_truth_section'])
        facts_c = dict(sm_count=facts['sm_count'], l2_bytes=facts['l2_bytes'], compute_capability=facts['compute_capability'])
        D.cross_check(facts_c, entry, gt)
        tables = stages.tier_tables(facts_c)
        if a.plan:
            print(plan_text(facts, selected, tables)); return 0
        if tool is None: raise D.Refusal('a real run needs the live device: drop --facts-json')
        return execute(a, entry, booking, facts, facts_c, gt, tool, out_dir, selected, tables)
    except D.Refusal as ex:
        print('REFUSED: %s' % ex, file=sys.stderr); return 1


def execute(a, entry, booking, facts, facts_c, gt, tool, out_dir, selected, tables):
    t_start = time.time(); results = {}; stage_records = []; warnings = []; build = out_dir / 'build'; sm = facts['sm_count']
    passive = bool(a.allow_passive_display_context); wait_idle = lambda u: D.wait_idle(u, allow_passive_display=passive)
    if passive: warnings.append('--allow-passive-display-context was set: foreign display-server processes of at most %d MiB were tolerated on the GPU while utilisation was 0%%; accepted processes are in each stage idle_before' % D.MAX_DISPLAY_CONTEXT_MIB)
    for name, prog, args, minutes, what in stages.STAGES:
        if name not in selected: continue
        if (time.time() - t_start) / 60 + minutes > a.max_minutes:
            warnings.append('stage %s not started: %.0f of %.0f minutes used' % (name, (time.time() - t_start) / 60, a.max_minutes)); stage_records.append(dict(name=name, status='not_started')); continue
        idle = wait_idle(a.device_uuid)
        info = tool.compile(prog, facts_c, build); sdir = out_dir / 'stages' / name; sdir.mkdir(parents=True, exist_ok=True)
        rec = dict(name=name, program=prog, source_sha256=info['source_sha256'], binary_sha256=info['binary_sha256'], compile_command=info['command'], idle_before=idle, status='ok')
        if name == 'energy':
            stage, econst = energy_stage.run_energy_stage(info['binary'], tool.nvcc, a.device_uuid, booking, sdir, wait_idle, runner.run_program)
            rec.update({k: v for k, v in stage.items() if k != 'name'}); rec['status'] = stage['status']
            if econst is not None: results['energy'] = econst
        elif name == 'tensor':
            stage, traw = tensor_stage.run_tensor_stage(info['binary'], tool.nvcc, a.device_uuid, booking, sdir, wait_idle, runner.run_program)
            rec.update({k: v for k, v in stage.items() if k != 'name'}); rec['status'] = stage['status']; results['tensor'] = traw
        elif name != 'pipes':
            r = runner.run_program(info['binary'], args, a.device_uuid, booking, a.timeout_s)
            (sdir / 'stdout.jsonl').write_text(r['stdout']); (sdir / 'stderr.txt').write_text(r['stderr'])
            rec.update(args=args, returncode=r['returncode'], seconds=r['seconds'], stdout_sha256=runner.hashlib.sha256(r['stdout'].encode()).hexdigest())
            rows = runner.parse_jsonl(r['stdout']); rec['rows'] = len(rows)
            if r['returncode'] != 0 or not rows: rec['status'] = 'failed'; rec['reason'] = (r['stderr'] or 'no output').strip()[-300:]
            else: results[name] = rows
        else:
            grid = stages.pipes_grid(facts_c); cells = out_dir / 'stages' / 'pipes' / 'cells'; ok = 0; failures = []
            t0 = time.time()
            for row in grid:
                cdir = cells / row['slot']
                r = runner.run_program(info['binary'], stages.pipes_args(str(cdir), row), a.device_uuid, booking, 300)
                if r['returncode'] == 0 and (cdir / 'result.json').exists(): ok += 1
                else: failures.append(dict(slot=row['slot'], reason=(r['stderr'] or '')[-200:]))
            (sdir / 'grid.json').write_text(json.dumps(grid, indent=1))
            rec.update(cells=len(grid), cells_ok=ok, cells_failed=failures, seconds=round(time.time() - t0, 1))
            if failures: rec['status'] = 'failed'; rec['reason'] = '%d of %d cells failed' % (len(failures), len(grid))
            else: results['pipes'] = grid
        stage_records.append(rec)
    constants = {}; store_derivation = None; complete = all(s.get('status') == 'ok' for s in stage_records) and len(stage_records) == len(stages.STAGES)
    def need(*names): return all(n in results for n in names)
    try:
        if need('launch_chain', 'launch_reuse'): constants['launch_reuse'] = fits.fit_launch_reuse(results['launch_chain'], results['launch_reuse'], sm)
        if need('mlp'): constants['mlp'] = fits.fit_mlp(results['mlp'], sm)
        if need('stream'): constants['stream_curves'] = fits.fit_stream(results['stream'])
        if need('smem_volatile', 'smem_plain'): constants['smem'] = fits.fit_smem(results['smem_volatile'], results['smem_plain'])
        if need('overlap'): constants['overlap'] = fits.fit_overlap(results['overlap'])
        if need('pipes'):
            grid = results['pipes']; by_slot = {r['slot']: r for r in grid}; cells = out_dir / 'stages' / 'pipes' / 'cells'
            table = {r['slot']: fits.load_cell(cells / r['slot'], r) for r in grid if r['kind'] == 'compute'}
            pipes = fits.fit_pipes(table, by_slot, sm); constants['pipes'] = pipes
            cyc = {r['tier']: fits.chase_cycles_per_step(cells / r['slot'], r) for r in grid if r['kind'] == 'memory'}
            constants['chase_latency_ns'] = fits.fit_chase(cyc, pipes['effective_sm_clock_hz'])
        if need('store') and 'chase_latency_ns' in constants:
            legacy, w, store_derivation = fits.fit_store_detailed(results['store'], sm, constants['chase_latency_ns'], constants.get('stream_curves'), a.store_method, facts.get('l2_bytes'), a.dram_peak_tbps); constants['store_legacy'] = legacy; warnings += w
        if need('energy'): constants['energy'] = results['energy']
        if need('tensor'): constants['tensor'] = tensor.finalize(results['tensor'], results.get('energy'), sm)
    except (KeyError, ValueError, StopIteration, IndexError) as ex:
        warnings.append('fit failed: %r' % (ex,)); complete = False
    doc = document.build(device=facts, entry=entry, ground_truth=gt, booking=booking, toolchain=dict(nvcc=tool.version, arch=runner.arch_flag(facts_c)), stages=stage_records,
                         constants=constants, warnings=warnings, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), complete=complete, store_derivation=store_derivation)
    ok, why = document.validate(doc)
    if not ok: doc['complete'] = False; doc['warnings'].append(why)
    path = out_dir / ('calibration_%s_%s.json' % (runner.arch_flag(facts_c), a.device_uuid[4:12]))
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    if doc['complete']: document.export_legacy(doc, out_dir / 'legacy_constants')
    print('calibration %s: %s (%s)' % ('COMPLETE' if doc['complete'] else 'INCOMPLETE', path, '; '.join(doc['warnings']) or 'no warnings'))
    return 0 if doc['complete'] else 3


if __name__ == '__main__':
    sys.exit(main())
