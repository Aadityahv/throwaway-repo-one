"""CPU-only tests of the energy stage: design table, SASS counting, trace integration, admission, fit, and a full stubbed run of the stage (synthetic physics with known rates)."""
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from cal import energy as En, energy_stage as ES  # noqa: E402

SR = HERE.parent


def synth_sass(name, with_spill=False, guard=False, second_loop=False, drop=None):
    """Fake cuobjdump SASS of one design: a prologue, one backward loop whose body has exactly the design's priced instructions plus integer/control overhead, an epilogue."""
    R, F, A, I, S, H, B, Z, W = En.DESIGNS[name][:9]
    sym = '_Z14energy_fixtureI' + ''.join('Li%dE' % x for x in (R, F, A, I, S, H, B, Z, W)) + 'EvPKjPjS2_jjj'
    body = []
    exp = En.expected_in_loop(En.DESIGNS[name])
    ops = dict(fp32_fma='FFMA R4, R4, R0, R5', fp32_add='FADD R6, R6, R1', special_function='MUFU.EX2 R7, -R7', shared_store='STS [R2], R3', shared_load='LDS R3, [R2]',
               shuffle='SHFL.BFLY PT, R3, R3, 0x1, 0x1f', barrier='BAR.SYNC.DEFER_BLOCKING 0x0', global_load='LDG.E.STRONG.GPU R8, desc[UR4][R10.64]', global_store='STG.E desc[UR4][R12.64], R3')
    for fam, n in exp.items():
        if drop == fam: n = max(0, n - 1)
        body += [ops[fam]] * n
    body += ['LOP3.LUT R9, R9, R8, RZ, 0x3c, !PT'] * (R * I // 2) + ['IADD3 R9, R9, R8, RZ'] * (R * I // 2) + ['IADD3 R14, R14, 0x1, RZ', 'ISETP.GE.U32.AND P0, PT, R14, UR8, PT']
    if with_spill: body.append('STL [R1], R3')
    if guard: body.append('@P1 IADD3 R15, R15, 0x1, RZ')
    lines = ['\t\tFunction : ' + sym]; addr = [0]
    def emit(text):
        lines.append('        /*%04x*/                   %s ;                                  /* 0x0000000000000000 */' % (addr[0], text)); addr[0] += 16
    emit('LDC R1, c[0x0][0x37c]'); emit('S2R R7, SR_TID.X'); start = addr[0]
    for b in body: emit(b)
    if second_loop: emit('@P2 BRA 0x%x' % start)
    emit('@P0 BRA 0x%x' % start); emit('EXIT')
    return sym, lines


def test_op_family_identical_to_the_feature_extractor():
    spec = importlib.util.spec_from_file_location('xf', SR / 'extract_features.py'); xf = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SR))
    try: spec.loader.exec_module(xf)
    except Exception as ex: pytest.skip('extract_features not importable here: %r' % (ex,))
    ops = ['LDG.E', 'LDG.E.STRONG.GPU', 'STG.E', 'LDGSTS.E', 'LDGDEPBAR', 'DEPBAR.LE', 'LDSM.16.M88', 'LDS', 'LDS.U.128', 'STS', 'SHFL.BFLY', 'MUFU.EX2', 'BAR.SYNC.DEFER_BLOCKING',
           'FFMA', 'UFFMA', 'FADD', 'FMUL', 'FSETP.GE.AND', 'HFMA2', 'F2I.TRUNC', 'IADD3', 'IMAD.WIDE', 'LEA', 'LOP3.LUT', 'ISETP.GE.U32.AND', 'SEL', 'BRA', 'BSSY.RECONVERGENT', 'EXIT',
           'LDC', 'LDCU.64', 'S2R', 'MOV', 'CS2R', 'NOP', 'FOO.BAR', 'IMNMX', 'POPC']
    for op in ops: assert En.op_family(op) == xf.op_family(op)[0], op


def test_design_table_mirrors_the_cuda_program():
    src = (HERE / 'csrc' / 'micro_energy.cu').read_text()
    rows = re.findall(r'\{"(\w+)", (\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+), "(\w+)","(\w+)",(\d+),', src)
    cu = {r[0]: tuple(int(x) for x in r[1:10]) + (r[10], r[11], int(r[12])) for r in rows}
    assert cu == En.DESIGNS


def test_window_plan_is_consistent():
    ids = [w[0] for w in En.WINDOWS]; assert len(ids) == len(set(ids)) == 27
    assert all(d in En.DESIGNS for _, d, _ in En.WINDOWS)
    assert sum(1 for w in En.WINDOWS if w[2] == 'fit') == 23 and sum(1 for w in En.WINDOWS if w[2] == 'heldout') == 4
    # every fit design appears once; the repeated anchor reuses fma64
    fit_designs = [d for _, d, r in En.WINDOWS if r == 'fit']; assert len(fit_designs) == len(set(fit_designs))
    # every cost class has at least two fit windows so a rate can be identified
    cols = {}
    for _, d, r in En.WINDOWS:
        if r != 'fit': continue
        R, F, A, I, S, H, B, Z, W = En.DESIGNS[d][:9]
        for c, on in dict(ffma=F, fpo=A, intc=I, sfu=S, shm=H, shfl=B, bar=Z, B_wr=W).items(): cols[c] = cols.get(c, 0) + (1 if on else 0)
    assert all(v >= 2 for v in cols.values()), cols


def test_count_kernel_counts_loop_and_prologue():
    sym, lines = synth_sass('fma16'); c = En.count_kernel(lines, trips=1000)
    assert c['in_loop']['fp32_fma'] == 16 and c['per_thread']['fp32_fma'] == 16000 and c['out_of_loop']['move_const_special'] >= 1
    assert En.verify_design('fma16', c['in_loop']) == []
    assert En.design_of_symbol(sym) == En.DESIGNS['fma16'][:9]


@pytest.mark.parametrize('kw,msg', [(dict(with_spill=True), 'spill'), (dict(guard=True), 'predicate-guarded'), (dict(second_loop=True), 'exactly one backward branch')])
def test_count_kernel_refuses(kw, msg):
    _, lines = synth_sass('shared16', **kw)
    with pytest.raises(ValueError, match=msg): En.count_kernel(lines, 10)


def test_verify_design_catches_a_deleted_instruction():
    _, lines = synth_sass('sfu16', drop='special_function'); c = En.count_kernel(lines, 10)
    assert any('special_function' in b for b in En.verify_design('sfu16', c['in_loop']))


def test_real_retained_sass_parses():
    p = SR / 'energy_revision' / 'execution' / 'runs' / 'native_5a5e881d' / 'native' / 'sass.stdout'
    if not p.exists(): pytest.skip('retained SASS not present')
    funcs = En.split_functions(p.read_text()); assert funcs
    for name, lines in funcs.items():
        c = En.count_kernel(lines, 100)
        assert c['in_loop'].get('global_load', 0) >= 1 and c['in_loop'].get('control', 0) >= 1


def test_integrate_energy_constant_power():
    t = np.arange(0, 30e9, 5e6); s = np.stack([t, np.full_like(t, 400000.0)], 1)  # 400 W
    j, info = En.integrate_energy(s, 5e9, 25e9); assert abs(j - 400 * 20) < 1e-6 and abs(info['mean_power_w'] - 400) < 1e-9 and info['max_gap_s'] < 0.006


def test_integrate_energy_refuses_uncovered_window():
    t = np.arange(10e9, 20e9, 5e6); s = np.stack([t, np.full_like(t, 400000.0)], 1)
    with pytest.raises(ValueError): En.integrate_energy(s, 5e9, 15e9)


def window(i, cols, E, t, admitted=True):
    return dict(id=str(i), design=str(i), energy_j_per_launch=E, runtime_s_per_launch=t, columns=cols, admitted=admitted)


def test_fit_recovers_known_rates_and_flags_unidentified():
    rng = np.random.default_rng(1); true = dict(B_l2=60., B_dr=170., B_wr=40., ffma=9., fpo=8., intc=4., sfu=12., shm=38., shfl=70., bar=50.); base = 150.; ws = []
    for i in range(60):
        cols = {c: float(rng.uniform(0, 1e9) * (rng.random() < 0.5)) for c in En.COLUMNS}; t = float(rng.uniform(1e-5, 1e-3))
        ws.append(window(i, cols, base * t + sum(true[c] * cols[c] * 1e-12 for c in En.COLUMNS), t))
    p = En.fit_rates(ws)
    for c in En.COLUMNS: assert abs(p['rates_pJ'][c] - true[c]) < 1e-3
    assert abs(p['base_power_w'] - base) < 1e-3
    for w in ws: w['columns']['shfl'] = 0.0
    assert En.fit_rates(ws)['status']['shfl'].startswith('UNIDENTIFIED')


def test_inadmissible_windows_do_not_support_a_rate():
    ws = [window(i, {c: (1e9 if c == 'sfu' else 0.0) for c in En.COLUMNS}, 1e-3 * (i + 1), 1e-5, admitted=(i > 0)) for i in range(5)]
    ws += [window(10 + i, {c: (1e9 if c == 'ffma' else 0.0) for c in En.COLUMNS}, 2e-3 * (i + 1), 2e-5) for i in range(5)]
    assert En.fit_rates(ws)['support_windows']['sfu'] == 4


def test_predict_energy_applies_cap():
    p = dict(base_power_w=100., rates_pJ=dict(B_dr=1000.))
    assert En.predict_energy(p, dict(B_dr=1e12), 1e-3, 600.) == pytest.approx(0.6)       # 1 J uncapped, capped at 600 W * 1 ms
    assert En.predict_energy(p, dict(B_dr=1e8), 1e-3, 600.) == pytest.approx(0.1 + 0.1)


# ------------------------------------------------------------------ the full stage with stubs and synthetic physics
TRUE = dict(B_l2=55., B_dr=150., B_wr=30., ffma=8., fpo=7., intc=3., sfu=10., shm=36., shfl=60., bar=45.); BASE = 140.


def stub_run_factory(sass_by_design, fma_drift=1.0):
    state = dict(n=0)
    def run(cmd, **kw):
        class R: returncode = 0; stdout = '\n'.join('\n'.join(l) for l in sass_by_design.values()); stderr = ''
        return R()
    def run_program(binary, args, uuid, booking, timeout):
        design, warm, win, wdir = args[0], float(args[1]), float(args[2]), Path(args[3]); p = En.DESIGNS[design]; R, F, A, I, S, H, B, Z, W = p[:9]
        lanes = 188 * 1024 if not p[11] else 188 * 32; T = 1000
        read = float(lanes * T * R * 4); write = float(lanes * T * W * 4)
        sym, lines = synth_sass(design); cnt = En.count_kernel(lines, T)
        cols = En.columns_from_counts(cnt['per_thread'], lanes, read, write, p[9], p[10])
        dyn = sum(TRUE[c] * cols[c] * 1e-12 for c in En.COLUMNS); runtime = max(1e-5, dyn / (100.0 + 20.0 * (hash(design) % 7))); E = BASE * runtime + dyn; power = E / runtime  # dynamic power 100-220 W: below the cap, varied
        if design == 'fma64': state['n'] += 1
        if state['n'] == 2 and design == 'fma64': power *= fma_drift  # the repeated anchor, to test the thermal ratio report
        t0 = 10_000_000_000; t_begin = t0 + 300_000_000; t_end = t_begin + int(win * 1e9); ts = np.arange(t0, t_end + 400_000_000, 5_000_000)
        pw = power * (1 + 0.003 * np.sin(np.arange(len(ts)) / 7.0))
        wdir.mkdir(parents=True, exist_ok=True)
        with open(wdir / 'samples.csv', 'w') as f:
            f.write('monotonic_ns,board_power_mw,temperature_c,graphics_clock_mhz,memory_clock_mhz,utilization_percent\n')
            for a, b in zip(ts, pw): f.write('%d,%d,60,2800,14000,100\n' % (a, int(b * 1000)))
        launches = int(win / runtime)
        row = dict(mode='energy_window', design=design, R=R, F=F, A=A, I=I, S=S, H=H, B=B, Z=Z, W=W, in_tier=p[9], out_tier=p[10], lanes=lanes, trips_per_launch=T, read_bytes_per_launch=read,
                   write_bytes_per_launch=write, launches=launches, t_begin_ns=t_begin, t_end_ns=t_end, window_seconds=win, nvml_sampler_ok=True, energy_counter_mj=-1, energy_counter_ok=False,
                   power_limit_mw=600000, output_deterministic=True, output_nonzero=True)
        return dict(returncode=0, stdout=json.dumps(row) + '\n', stderr='', seconds=1.0)
    return run, run_program


def test_stage_end_to_end_recovers_rates_with_synthetic_physics(tmp_path):
    sass = dict(synth_sass(n) for n in En.DESIGNS); run, run_program = stub_run_factory(sass)
    (tmp_path / 'bin').write_text('x'); cud = tmp_path / 'cuobjdump'; cud.write_text('')
    stage, const = ES.run_energy_stage(tmp_path / 'bin', str(tmp_path / 'nvcc'), 'GPU-x', 'booking-ref-12345', tmp_path / 'out', lambda u: dict(utilization_pct=0), run_program, run=run, warmup_s=0, window_s=2)
    assert stage['status'] == 'ok', stage
    prof = const['profile']
    for c in En.COLUMNS: assert prof['rates_pJ'][c] == pytest.approx(TRUE[c], rel=2e-2), c
    assert prof['base_power_w'] == pytest.approx(BASE, rel=3e-2)
    assert max(abs(v) for v in const['heldout_error_pct'].values()) < 3.0
    assert const['thermal_anchor']['ratio_repeat_to_first'] == pytest.approx(1.0, abs=0.02)
    assert (tmp_path / 'out' / 'windows.json').exists() and len(const['sass_verified']) == len(En.DESIGNS)


def test_stage_reports_a_throttled_window_as_not_admitted_and_incomplete(tmp_path):
    sass = dict(synth_sass(n) for n in En.DESIGNS); run, run_program = stub_run_factory(sass)
    def cap_hit(binary, args, uuid, booking, timeout):
        r = run_program(binary, args, uuid, booking, timeout)
        if args[0] == 'rd_dram_16':
            row = json.loads(r['stdout']); row['power_limit_mw'] = 100000; r['stdout'] = json.dumps(row) + '\n'  # a 100 W cap makes every window inadmissible by the cap rule
        return r
    (tmp_path / 'bin').write_text('x'); (tmp_path / 'cuobjdump').write_text('')
    stage, const = ES.run_energy_stage(tmp_path / 'bin', str(tmp_path / 'nvcc'), 'GPU-x', 'booking-ref-12345', tmp_path / 'out', lambda u: {}, cap_hit, run=run, warmup_s=0, window_s=2)
    assert stage['status'] == 'incomplete' and 'rd_dram_16' in stage['not_admitted']


def test_stage_records_a_failed_window_and_continues(tmp_path):
    sass = dict(synth_sass(n) for n in En.DESIGNS); run, run_program = stub_run_factory(sass)
    def boom(binary, args, uuid, booking, timeout):
        if args[0] == 'bar16': return dict(returncode=2, stdout='', stderr='REFUSED: test', seconds=0.1)
        return run_program(binary, args, uuid, booking, timeout)
    (tmp_path / 'bin').write_text('x'); (tmp_path / 'cuobjdump').write_text('')
    stage, const = ES.run_energy_stage(tmp_path / 'bin', str(tmp_path / 'nvcc'), 'GPU-x', 'booking-ref-12345', tmp_path / 'out', lambda u: {}, boom, run=run, warmup_s=0, window_s=2)
    assert stage['status'] == 'incomplete' and stage['failed'][0]['id'] == 'bar16' and stage['windows_ran'] == 27


def test_verifier_tool_accepts_synthetic_sass_and_flags_a_deleted_load():
    sys.path.insert(0, str(HERE / 'tools'))
    import verify_energy_sass as V
    text = '\n'.join('\n'.join(synth_sass(n)[1]) for n in En.DESIGNS)
    # distinct instantiations only: designs sharing parameters appear once
    problems, _ = V.verify('\n'.join('\n'.join(l) for l in {En.design_of_symbol(synth_sass(n)[0]): synth_sass(n)[1] for n in En.DESIGNS}.values()))
    assert problems == []
    broken = synth_sass('rd_dram_8', drop='global_load')[1]
    problems, _ = V.verify('\n'.join(broken))
    assert any('global_load' in x for x in problems)
