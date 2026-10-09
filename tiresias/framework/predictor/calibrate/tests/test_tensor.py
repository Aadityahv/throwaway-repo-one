"""CPU-only tests of the optional tensor stage (cal/tensor.py, cal/tensor_stage.py) and of its use by the predictors."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from cal import energy as En, tensor as Tn, tensor_stage as TS, document  # noqa: E402

TRUE_TC = 80.0; BASE = 140.0
RATES = dict(B_l2=55., B_dr=150., B_wr=30., ffma=8., fpo=7., intc=3., sfu=10., shm=36., shfl=60., bar=45.)


def mma_sass(m=32, guard_never=False, extra=None):
    lines = ['Function : _Z10mma_energyILi%dEEvPKjPjjj' % m, '        /*0000*/                   MOV R1, c[0x0][0x28] ;']
    pc = 0x10
    def add(op, g=''):
        nonlocal pc
        lines.append('        /*%04x*/                   %s%s ;' % (pc, (g + ' ') if g else '', op)); pc += 0x10
    add('IMAD.MOV.U32 R2, RZ, RZ, 0x0')
    loop = pc
    add('LDG.E.CG R4, desc[UR4][R2.64]')
    for _ in range(13): add('IADD3 R5, R5, R4, RZ')
    if guard_never: add('UIADD3 UR4, UR4, 0x1, URZ', '@!UPT')
    for i in range(m): add('HMMA.16816.F32.BF16 R%d, R8, R12, R%d' % (20 + 4 * (i % 4), 20 + 4 * (i % 4)))
    for x in (extra or []): add(x)
    add('BRA 0x%x' % loop, '@P0')
    add('STG.E desc[UR4][R6.64], R5'); add('EXIT')
    return lines


def test_op_family_and_columns_know_tensor_core():
    assert En.op_family('HMMA.16816.F32.BF16') == 'tensor_core' and En.op_family('IMMA.16832.S8.S8') == 'tensor_core' and En.op_family('LDSM.16.M88.4') == 'shared_matrix_load'
    c = En.columns_from_feature_row(dict(families={'tensor_core': dict(lane_instructions=1000), 'integer_alu': dict(lane_instructions=10)}), 1024, 'L2', 0)
    assert c['tc'] == 1000.0 and c['intc'] == 10.0
    cc = En.columns_from_counts({'tensor_core': 32, 'integer_alu': 3}, 100, 0, 0, 'small', 'none'); assert cc['tc'] == 3200.0 and cc['intc'] == 300.0


def test_count_kernel_skips_never_true_guards_and_verify_design():
    c = En.count_kernel(mma_sass(guard_never=True), 10)
    assert c['in_loop']['tensor_core'] == 32 and Tn.verify_design(c['in_loop']) == []
    lines = mma_sass(); lines.insert(-3, '        /*0200*/                   @P1 IADD3 R1, R1, R1, RZ ;')   # a data-dependent guard inside the loop is still refused
    with pytest.raises(ValueError, match='predicate-guarded'): En.count_kernel(lines, 10)
    bad = Tn.verify_design(En.count_kernel(mma_sass(m=16), 10)['in_loop']); assert any('tensor_core' in b for b in bad)


def test_fit_issue_recovers_slopes_and_refuses_non_linear_cells():
    cells = []
    for S, t, b, slope in ((1, 32, 1, 30.0), (4, 1024, 188, 512.0)):
        for loops, unroll in ((127, 1), (127, 4), (127, 16), (509, 1), (509, 4), (509, 16)): cells.append(dict(streams=S, threads=t, blocks=b, loops=loops, unroll=unroll, cycles_median=slope * loops * unroll + 7 * loops + 100))
    r = Tn.fit_issue(dict(cells=cells), 188)
    assert r['dependent_latency_cycles'] == pytest.approx(30.0) and r['issue_cycles_per_warp_instruction_per_sm'] == pytest.approx(512.0 / 128)
    bad = [dict(c) for c in cells]; bad[8]['cycles_median'] *= 3
    with pytest.raises(ValueError, match='loop model'): Tn.fit_issue(dict(cells=bad), 188)
    lat = [dict(c) for c in cells]; lat[2]['cycles_median'] *= 1.5; ok = Tn.fit_issue(dict(cells=lat), 188)    # a poor latency fit is recorded, not rejected (as in the pipe stage)
    assert ok['fit_quality_max_relative_residual']['S1_t32_b1'] > 0.05 and ok['issue_cycles_per_warp_instruction_per_sm'] == pytest.approx(512.0 / 128)
    with pytest.raises(ValueError, match='missing'): Tn.fit_issue(dict(cells=cells[:6]), 188)


def synth_window(i, warps, rate=TRUE_TC, noise=0.0, admitted=True):
    lanes = 188 * warps * 32; trips = 2048; per = {'tensor_core': 32, 'integer_alu': 13, 'global_load': 1}
    cols = En.columns_from_counts(En.count_kernel(mma_sass(), trips)['per_thread'], lanes, lanes * trips * 4.0, 0, 'small', 'none')
    t = 6e-4 * (1 + 2 * (warps == 16)); other = sum(RATES[c] * cols[c] * 1e-12 for c in En.COLUMNS)
    E = BASE * t + other + rate * cols['tc'] * 1e-12 * (1 + noise)
    return dict(id=i, design=i, energy_j_per_launch=E, runtime_s_per_launch=t, columns=cols, admitted=admitted)


ENERGY = dict(profile=dict(base_power_w=BASE, rates_pJ=RATES, status={c: 'ok' for c in En.COLUMNS}))


def test_derive_rate_recovers_the_true_rate_and_flags_unstable_or_unidentified_windows():
    r = Tn.derive_rate([synth_window('tc_w1', 1), synth_window('tc_w4', 4), synth_window('tc_w16', 16)], ENERGY)
    assert r['status'] == 'ok' and r['rate_pJ_per_lane_instruction'] == pytest.approx(TRUE_TC, rel=1e-6) and r['max_window_error_pct'] < 1e-6
    # one window with a 40% higher tensor energy: the fitted rate sits between, the window error exceeds 10%?  (reported as UNSTABLE only when it does)
    u = Tn.derive_rate([synth_window('tc_w1', 1, rate=30.0), synth_window('tc_w4', 4), synth_window('tc_w16', 16, rate=160.0)], ENERGY)
    assert u['status'] in ('ok', 'UNSTABLE') and u['rate_pJ_per_lane_instruction'] > 0
    one = Tn.derive_rate([synth_window('tc_w1', 1), synth_window('tc_w4', 4, admitted=False), synth_window('tc_w16', 16, admitted=False)], ENERGY)
    assert one['status'] == 'UNIDENTIFIED' and 'fewer than two' in one['reason']
    short = dict(profile=dict(base_power_w=BASE, rates_pJ={k: v for k, v in RATES.items() if k != 'B_l2'}, status={}))
    assert Tn.derive_rate([synth_window('a', 1), synth_window('b', 4)], short)['status'] == 'UNIDENTIFIED'
    assert Tn.derive_rate([synth_window('a', 1), synth_window('b', 4)], None)['status'] == 'UNIDENTIFIED'


def stub_factory():
    sass_text = '\n'.join(mma_sass())
    def run(cmd, **kw):
        class R: returncode = 0; stdout = sass_text; stderr = ''
        return R()
    def run_program(binary, args, uuid, booking, timeout):
        if args[0] == 'issue':
            d = Path(args[1]); d.mkdir(parents=True, exist_ok=True)
            cells = [dict(streams=S, threads=t, blocks=b, loops=l, unroll=u, cycles_median=sl * l * u + 5 * l + 90) for S, t, b, sl in ((1, 32, 1, 30.0), (4, 1024, 188, 512.0)) for l, u in ((127, 1), (127, 4), (127, 16), (509, 1), (509, 4), (509, 16))]
            (d / 'issue.json').write_text(json.dumps(dict(schema='tensor_issue/1', sm_count=188, cells=cells))); return dict(returncode=0, stdout='{"mode":"tensor_issue","ok":true}\n', stderr='', seconds=1.0)
        _, wid, warm, win, wdir = args[:5]; wdir = Path(wdir); warps = {'tc_w1': 1, 'tc_w4': 4, 'tc_w16': 16}[wid]; w = synth_window(wid, warps)
        lanes = 188 * warps * 32; T = 2048; t = w['runtime_s_per_launch']; E = w['energy_j_per_launch']; power = E / t
        t0 = 10_000_000_000; t_begin = t0 + 300_000_000; t_end = t_begin + int(float(win) * 1e9); ts = np.arange(t0, t_end + 400_000_000, 5_000_000); wdir.mkdir(parents=True, exist_ok=True)
        with open(wdir / 'samples.csv', 'w') as f:
            f.write('monotonic_ns,board_power_mw,temperature_c,graphics_clock_mhz,memory_clock_mhz,utilization_percent\n')
            for a in ts: f.write('%d,%d,60,2800,14000,0\n' % (a, int(power * 1000)))
        row = dict(mode='tensor_energy_window', design=wid, M=32, warps_per_sm=warps, blocks=188, threads=warps * 32, lanes=lanes, trips_per_launch=T, read_bytes_per_launch=float(lanes * T * 4), write_bytes_per_launch=0.0,
                   launches=int(float(win) / t), t_begin_ns=t_begin, t_end_ns=t_end, window_seconds=float(win), nvml_sampler_ok=True, energy_counter_mj=-1, energy_counter_ok=False, power_limit_mw=600000, output_deterministic=True, output_nonzero=True)
        return dict(returncode=0, stdout=json.dumps(row) + '\n', stderr='', seconds=1.0)
    return run, run_program


def test_stage_end_to_end_with_stubs(tmp_path):
    run, run_program = stub_factory(); (tmp_path / 'bin').write_text('x'); (tmp_path / 'cuobjdump').write_text('')
    stage, raw = TS.run_tensor_stage(tmp_path / 'bin', str(tmp_path / 'nvcc'), 'GPU-x', 'booking-ref-12345', tmp_path / 'out', lambda u: {}, run_program, run=run, warmup_s=0, window_s=2)
    assert stage['status'] == 'ok', stage and len(raw['windows']) == 3
    c = Tn.finalize(raw, ENERGY, 188)
    assert c['status'] == 'ok' and c['energy']['rate_pJ_per_lane_instruction'] == pytest.approx(TRUE_TC, rel=2e-2) and c['issue']['issue_cycles_per_warp_instruction_per_sm'] == pytest.approx(4.0)
    assert set(c['sass_verified']) == {'tc_w1', 'tc_w4', 'tc_w16'}


def test_stage_records_a_failed_window_and_a_sass_mismatch(tmp_path):
    run, run_program = stub_factory(); (tmp_path / 'bin').write_text('x'); (tmp_path / 'cuobjdump').write_text('')
    def boom(binary, args, uuid, booking, timeout):
        if args[0] == 'energy' and args[1] == 'tc_w4': raise RuntimeError('driver crashed')
        return run_program(binary, args, uuid, booking, timeout)
    stage, raw = TS.run_tensor_stage(tmp_path / 'bin', str(tmp_path / 'nvcc'), 'GPU-x', 'booking-ref-12345', tmp_path / 'out', lambda u: {}, boom, run=run, warmup_s=0, window_s=2)
    assert stage['status'] == 'incomplete' and any(f['id'] == 'tc_w4' for f in stage['failed']) and len(raw['windows']) == 2
    c = Tn.finalize(raw, ENERGY, 188); assert c['status'] == 'ok'     # two admitted windows still identify the rate
    def wrong_sass(cmd, **kw):
        class R: returncode = 0; stdout = '\n'.join(mma_sass(m=16)); stderr = ''
        return R()
    with pytest.raises(RuntimeError, match='not found'): TS.run_tensor_stage(tmp_path / 'bin', str(tmp_path / 'nvcc'), 'GPU-x', 'booking-ref-12345', tmp_path / 'out2', lambda u: {}, run_program, run=wrong_sass, warmup_s=0, window_s=2)


def test_energy_rows_price_tensor_instructions_only_with_a_calibrated_rate():
    import predict as PR
    row = dict(cell_id='c', status='supported', work=dict(families={'tensor_core': dict(lane_instructions=1e6), 'integer_alu': dict(lane_instructions=1e5)}), memory=dict(logical_bytes_per_launch=1e6, tier='L2'))
    doc = dict(constants=dict(energy=dict(cap_w=600.0, profile=dict(base_power_w=BASE, rates_pJ=RATES))))
    r = PR.energy_rows(doc, dict(rows=[row]), {'c': 1e-4})['c']; assert r['status'] == 'unsupported' and 'tensor' in r['reason']
    doc['constants']['tensor'] = dict(energy=dict(status='UNIDENTIFIED')); assert PR.energy_rows(doc, dict(rows=[row]), {'c': 1e-4})['c']['status'] == 'unsupported'
    doc['constants']['tensor'] = dict(energy=dict(status='ok', rate_pJ_per_lane_instruction=80.0)); r = PR.energy_rows(doc, dict(rows=[row]), {'c': 1e-4})['c']
    assert r['status'] == 'ok' and r['term_j']['tc'] == pytest.approx(80.0 * 1e6 * 1e-12)
    nt = dict(row, cell_id='n', work=dict(families={'integer_alu': dict(lane_instructions=1e5)})); assert PR.energy_rows(doc, dict(rows=[nt]), {'n': 1e-4})['n']['status'] == 'ok'


def test_runtime_model_classifies_mma_as_tensor_class_and_refuses_without_a_measured_cost():
    from cal import portable_predict as PP
    assert PP._classify_with_tensor('HMMA.16816.F32.BF16') == ('tensor_mma', 1.0) and PP._classify_with_tensor('IMAD') == PP._classify_original('IMAD') and PP._classify_with_tensor('FFMA') == ('fp32_fma', 1.0)
    pipes = dict(issue_cycles_per_warp_instruction_per_sm={'mufu_ex2': 1.0}, dependent_latency_cycles={})
    c = dict(pipes=pipes, tensor=dict(issue=dict(issue_cycles_per_warp_instruction_per_sm=4.0, dependent_latency_cycles=30.0)))
    out = document._pipes_with_tensor(c); assert out['issue_cycles_per_warp_instruction_per_sm']['tensor_mma'] == 4.0 and out['dependent_latency_cycles']['tensor_mma'] == 30.0 and 'tensor_mma' not in pipes['issue_cycles_per_warp_instruction_per_sm']
    assert 'tensor_mma' not in document._pipes_with_tensor(dict(pipes=pipes))['issue_cycles_per_warp_instruction_per_sm']
