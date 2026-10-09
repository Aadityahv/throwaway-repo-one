"""CPU-only tests of the cap-safe window rule (calibrate/CAP_SAFE_WINDOWS.md): the scaling choice, the replay check of recorded probe steps, the stage plumbing with
stubbed window programs (counts scale with the launched grid, unscaled designs identical to the previous behaviour), the opt-in device guard, and the unchanged fit of the
archived Blackwell windows."""
import copy
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'tests'))
from cal import capsafe as CS, device as D, energy as En, energy_stage as ES, tensor_stage as TS  # noqa: E402
import test_energy as TE  # noqa: E402
import test_tensor as TT  # noqa: E402

CAP_MW = 250000; THR = CS.threshold_mw(0.90, CAP_MW); IDLE = 30000.0   # Ada-like: 250 W limit, 30 W idle


def cs_rec(steps, chosen, full=100, forced=False, idle=IDLE, cap=CAP_MW, fraction=0.90, converged=True):
    return dict(enabled=True, fraction=fraction, limit_mw=cap, threshold_mw=CS.threshold_mw(fraction, cap), idle_mw=idle, blocks_full=full, blocks_chosen=chosen, scaled=chosen != full,
                forced=forced, converged=converged, probe_elapsed_s=30.0, probe_steps=[dict(blocks=b, mean_mw=p, samples=500) for b, p in steps])


# ------------------------------------------------------------ the choice
def test_threshold_is_ninety_percent_of_the_enforced_limit():
    assert THR == 225000.0 and CS.CAP_SAFE_FRACTION == 0.90 and En.CAP_SAFE_FRACTION == 0.90 and En.ADMIT_FRACTION_OF_CAP == 0.985   # admission rule unchanged


def test_design_below_the_threshold_keeps_the_full_grid():
    assert CS.next_blocks(100, 224999.0, IDLE, THR) == 100 and CS.next_blocks(100, THR, IDLE, THR) == 100 and CS.next_blocks(7, 31000.0, IDLE, THR) == 7


def test_design_above_the_threshold_is_scaled_linearly_in_dynamic_power():
    # full grid reads 280 W: dynamic 250 W; the target leaves 195 W of dynamic power, so 78 of 100 blocks
    assert CS.next_blocks(100, 280000.0, IDLE, THR) == 78
    assert CS.next_blocks(188, 580000.0, 60000.0, CS.threshold_mw(0.90, 600000)) == int((188 * (540000.0 - 60000.0)) // (580000.0 - 60000.0)) == 173


def test_choice_is_a_strict_decrease_of_at_least_one_block():
    assert CS.next_blocks(10, 225001.0, IDLE, THR) == 9        # barely above: still one block less
    assert CS.next_blocks(100, 400000.0, 300000.0, THR) == 1   # idle above the threshold: the floor is one block
    assert CS.next_blocks(1, 400000.0, IDLE, THR) == 1
    assert CS.next_blocks(50, 260000.0, 260000.0, THR) == 49   # no measurable dynamic power: nothing to extrapolate, shrink by one


def test_chord_lower_than_the_curve_is_corrected_by_the_next_probe():
    # power concave in blocks: a 78-block proposal still reads above the threshold, the second probe shrinks again from the measured point
    b1 = CS.next_blocks(100, 280000.0, IDLE, THR); b2 = CS.next_blocks(b1, 240000.0, IDLE, THR)
    assert b1 == 78 and b2 == int(78 * (THR - IDLE) // (240000.0 - IDLE)) and b2 < b1


def test_settle_and_measure_seconds():
    assert CS.settle_measure_s(60) == 6.0 and CS.settle_measure_s(20) == 2.0 and CS.settle_measure_s(0) == 1.0


def test_window_args_empty_when_off_and_forced_block_count_appended():
    assert CS.window_args(None) == [] and CS.window_args(0.9) == ['0.9'] and CS.window_args(0.9, 78) == ['0.9', '78']


# ------------------------------------------------------------ replay of recorded probes
def test_replay_accepts_unscaled_scaled_four_step_floor_and_forced_records():
    assert CS.verify_record(cs_rec([(100, 200000.0)], 100)) == []
    assert CS.verify_record(cs_rec([(100, 280000.0), (78, 224000.0)], 78)) == []
    chain, b = [], 100
    for p in (400000.0, 390000.0, 380000.0, 370000.0): chain.append((b, p)); b = CS.next_blocks(b, p, IDLE, THR)     # four probes, all still above the threshold
    assert len(chain) == CS.MAX_PROBES_V1 and CS.verify_record(cs_rec(chain, b, converged=False)) == []
    assert CS.verify_record(cs_rec([(1, 400000.0)], 1, full=1, converged=False)) == []                 # cannot go below one block
    assert CS.verify_record(cs_rec([], 78, forced=True)) == [] and CS.verify_record(None) == [] and CS.verify_record(dict(enabled=False)) == []


@pytest.mark.parametrize('mutate, text', [
    (lambda r: r.update(blocks_chosen=90, scaled=True), 'the rule gives'),
    (lambda r: r['probe_steps'][1].update(blocks=70), 'ran at 70 blocks'),
    (lambda r: r['probe_steps'].pop(1), 'stopped after 1 steps'),
    (lambda r: r.update(threshold_mw=250000.0), 'threshold'),
    (lambda r: r.update(converged=False), 'converged flag'),
    (lambda r: r['probe_steps'][0].update(blocks=99), 'not the full grid'),
    (lambda r: r.update(scaled=False), 'scaled flag'),
])
def test_replay_refuses_a_choice_the_rule_would_not_make(mutate, text):
    r = cs_rec([(100, 280000.0), (78, 224000.0)], 78); mutate(r)
    assert any(text in p for p in CS.verify_record(r)), CS.verify_record(r)


def test_replay_checks_the_limit_against_the_stage_cap():
    assert any('differs from the stage cap' in p for p in CS.verify_record(cs_rec([(100, 200000.0)], 100), cap_w=600.0))


# ------------------------------------------------------------ stage plumbing (stub window programs)
FULL = 188 * 4          # the stub's full grid (188 SMs x 4 blocks of 256 threads; the light design has 188 blocks of 32 threads)
def capsafe_stub(sass, scale_of, log):
    """Stub of micro_energy honouring the cap-safe arguments: a design in scale_of runs on that fraction of its full grid, with a probe record the rule reproduces; others stay full."""
    run, base = TE.stub_run_factory(sass); thr = CS.threshold_mw(0.9, 600000)
    def run_program(binary, args, uuid, booking, timeout):
        log.append(list(args)); design = args[0]; extra = args[4:]
        r = base(binary, args[:4], uuid, booking, timeout); row = json.loads(r['stdout'])
        light = En.DESIGNS[design][11]; full = 188 if light else FULL; threads = 32 if light else 256
        forced = int(extra[1]) if len(extra) > 1 else None
        blocks = forced or max(1, int(full * scale_of.get(design, 1.0)))
        scale = (blocks * threads) / row['lanes']; row.update(lanes=blocks * threads, blocks=blocks, threads=threads)
        row['read_bytes_per_launch'] *= scale; row['write_bytes_per_launch'] *= scale
        if extra:
            if forced: row['cap_safe'] = cs_rec([], blocks, full=full, forced=True, cap=600000)
            elif blocks == full: row['cap_safe'] = cs_rec([(full, 200000.0)], full, full=full, cap=600000)
            else:                                   # a hot full-grid probe whose linear proposal is exactly `blocks`, then a cool probe at `blocks`
                hot = CS.q(IDLE + full * (thr - IDLE) / (blocks + 0.5)); assert CS.next_blocks(full, hot, IDLE, thr) == blocks
                row['cap_safe'] = cs_rec([(full, hot), (blocks, 400000.0)], blocks, full=full, cap=600000)
        r['stdout'] = json.dumps(row) + '\n'; return r
    return run, run_program


def run_stage(tmp_path, scale_of, cap_fraction=CS.CAP_SAFE_FRACTION, name='out'):
    sass = dict(TE.synth_sass(n) for n in En.DESIGNS); log = []; run, rp = capsafe_stub(sass, scale_of, log)
    (tmp_path / 'bin').write_text('x'); (tmp_path / 'cuobjdump').write_text('')
    stage, const = ES.run_energy_stage(tmp_path / 'bin', str(tmp_path / 'nvcc'), 'GPU-x', 'booking-ref-12345', tmp_path / name, lambda u: {}, rp, run=run, warmup_s=0, window_s=2, cap_fraction=cap_fraction)
    return stage, const, log


def test_stage_passes_the_fraction_and_forces_the_repeated_anchor_to_the_first_grid(tmp_path):
    stage, const, log = run_stage(tmp_path, {'fma64': 0.5})
    assert all(a[4] == '0.9' for a in log)
    fma = [a for a in log if a[0] == 'fma64']; chosen = next(w for w in const['windows'] if w['id'] == 'fma64')['cap_safe']['blocks_chosen']
    assert chosen == FULL // 2 and len(fma) == 2 and len(fma[0]) == 5 and fma[1][4:] == ['0.9', str(chosen)]    # first window probes; the repeat is forced to its grid
    assert [a for a in log if a[0] == 'fma16'] == [a for a in log if a[0] == 'fma16' and len(a) == 5]
    rep = next(w for w in const['windows'] if w['id'] == 'fma64_repeat'); assert rep['cap_safe']['forced'] and rep['cap_safe']['blocks_chosen'] == chosen


def test_stage_without_cap_fraction_is_the_previous_behaviour(tmp_path):
    sass = dict(TE.synth_sass(n) for n in En.DESIGNS); run, rp = TE.stub_run_factory(sass); seen = []
    def spy(binary, args, uuid, booking, timeout): seen.append(list(args)); return rp(binary, args, uuid, booking, timeout)
    (tmp_path / 'bin').write_text('x'); (tmp_path / 'cuobjdump').write_text('')
    stage, const = ES.run_energy_stage(tmp_path / 'bin', str(tmp_path / 'nvcc'), 'GPU-x', 'booking-ref-12345', tmp_path / 'out', lambda u: {}, spy, run=run, warmup_s=0, window_s=2, cap_fraction=None)
    assert stage['status'] == 'ok' and all(len(a) == 4 for a in seen) and all('cap_safe' not in w for w in const['windows'])


def test_counts_scale_with_the_launched_grid_and_unscaled_windows_are_identical(tmp_path):
    _, base, _ = run_stage(tmp_path, {}, name='a')
    _, scaled, _ = run_stage(tmp_path, {'fma64': 0.5, 'rd_dram_8': 0.25}, name='b')
    w0 = {w['id']: w for w in base['windows']}; w1 = {w['id']: w for w in scaled['windows']}
    for wid in w0:
        if wid in ('fma64', 'fma64_repeat', 'rd_dram_8'): continue          # the repeated anchor follows the first fma64 window's grid
        assert w1[wid]['columns'] == w0[wid]['columns'], wid                              # below-threshold designs: same grid, same counts
    for wid, frac in (('fma64', 0.5), ('rd_dram_8', 0.25)):
        c0, c1 = w0[wid]['columns'], w1[wid]['columns']; k = {k: c1[k] / c0[k] for k in ('ffma', 'B_dr') if c0[k]}
        assert all(v == pytest.approx(frac, rel=1e-9) for v in k.values()) and k, (wid, k)
    cs = w1['fma64']['cap_safe']; assert cs['scaled'] and cs['blocks_chosen'] == FULL // 2 and not w1['int64']['cap_safe']['scaled']


def test_cap_safe_record_reaches_windows_json_and_the_document_windows(tmp_path):
    stage, const, _ = run_stage(tmp_path, {'fma64': 0.5})
    on_disk = json.loads((tmp_path / 'out' / 'windows.json').read_text()); w = next(x for x in on_disk if x['id'] == 'fma64')
    assert w['cap_safe']['enabled'] and w['checks']['cap_safe_choice_reproduced'] and w['blocks'] == w['cap_safe']['blocks_chosen']
    assert next(x for x in const['windows'] if x['id'] == 'fma64')['cap_safe']['probe_steps']
    assert 'cap_safe' in json.loads((tmp_path / 'out' / 'fma64' / 'stdout.txt').read_text().splitlines()[0])


def test_a_recorded_choice_the_rule_would_not_make_refuses_the_window(tmp_path):
    sass = dict(TE.synth_sass(n) for n in En.DESIGNS); log = []; run, rp = capsafe_stub(sass, {'fma64': 0.5}, log)
    def tamper(binary, args, uuid, booking, timeout):
        r = rp(binary, args, uuid, booking, timeout)
        if args[0] == 'int64':
            row = json.loads(r['stdout']); row['cap_safe']['probe_steps'][0]['mean_mw'] = 700000.0; r['stdout'] = json.dumps(row) + '\n'   # the probe said "too hot" yet the grid stayed full
        return r
    (tmp_path / 'bin').write_text('x'); (tmp_path / 'cuobjdump').write_text('')
    stage, const = ES.run_energy_stage(tmp_path / 'bin', str(tmp_path / 'nvcc'), 'GPU-x', 'booking-ref-12345', tmp_path / 'out', lambda u: {}, tamper, run=run, warmup_s=0, window_s=2)
    assert 'int64' in stage['not_admitted'] and stage['status'] == 'incomplete'


def test_tensor_stage_passes_the_fraction(tmp_path):
    run, rp = TT.stub_factory(); seen = []
    def spy(binary, args, uuid, booking, timeout): seen.append(list(args)); return rp(binary, args, uuid, booking, timeout)
    (tmp_path / 'bin').write_text('x'); (tmp_path / 'cuobjdump').write_text('')
    TS.run_tensor_stage(tmp_path / 'bin', str(tmp_path / 'nvcc'), 'GPU-x', 'booking-ref-12345', tmp_path / 'out', lambda u: {}, spy, run=run, warmup_s=0, window_s=2)
    assert all(a[5] == '0.9' for a in seen if a[0] == 'energy') and len([a for a in seen if a[0] == 'energy']) == 3
    seen.clear(); TS.run_tensor_stage(tmp_path / 'bin', str(tmp_path / 'nvcc'), 'GPU-x', 'booking-ref-12345', tmp_path / 'out2', lambda u: {}, spy, run=run, warmup_s=0, window_s=2, cap_fraction=None)
    assert all(len(a) == 5 for a in seen if a[0] == 'energy')


def test_csrc_and_python_agree_on_the_constants():
    src = (HERE / 'csrc' / 'cap_safe.h').read_text()
    assert 'CAP_MAX_PROBES = %d' % CS.MAX_PROBES in src and 'std::min(6.0, std::max(1.0, 0.1 * warmup_s))' in src and CS.RULE_VERSION == 2 and 'rule_version = 2' in src
    assert 'std::floor(sm * (threshold_mw - idle_mw) / dyn)' in src and 'if (b > sm) return sm;' in src and 'std::max(1, b / 2)' in src and 'std::floor(x * 1000.0 + 0.5) / 1000.0' in src
    for f in ('micro_energy.cu', 'micro_tensor.cu'): assert '#include "cap_safe.h"' in (HERE / 'csrc' / f).read_text()


# ------------------------------------------------------------ rule version 2 (Amendment 1): whole SMs idle
SM = 100


def ada_like_power(b, sm=SM, idle=IDLE, per_sm=220000.0 / 1.0):
    """Power flat at the limit while every SM is busy (clock-held), linear in active SMs below that."""
    return 249000.0 if b >= sm else idle + (249000.0 - idle) * b / sm


def run_ladder(power_fn, full=400, sm=SM, idle=IDLE, thr=THR):
    steps = []; b = full
    while True:
        steps.append((b, power_fn(b))); nb = CS.next_blocks_v2(steps, sm, idle, thr)
        if nb == b or len(steps) == CS.MAX_PROBES: return steps, b
        b = nb


def rec2(steps, chosen, full=400, converged=True, sm=SM):
    r = cs_rec(steps, chosen, full=full, converged=converged); r.update(rule_version=2, sm_count=sm); return r


def test_v2_ada_like_flat_power_converges_within_the_budget():
    steps, b = run_ladder(ada_like_power)
    assert [x[0] for x in steps][:2] == [400, 100] and len(steps) <= CS.MAX_PROBES and steps[-1][1] <= THR and b < SM
    assert CS.verify_record(rec2(steps, b)) == []


def test_v2_concave_curve_halves_after_a_short_proposal():
    steps, b = run_ladder(lambda x: 249000.0 if x >= SM else IDLE + 219000.0 * (x / SM) ** 0.5)   # concave: the linear proposal falls short, halving follows
    assert len(steps) <= CS.MAX_PROBES and steps[-1][1] <= THR and steps[2][0] < SM and steps[3][0] == steps[2][0] // 2
    assert CS.verify_record(rec2(steps, b)) == []


def test_v2_below_threshold_designs_are_unchanged():
    steps, b = run_ladder(lambda x: 200000.0); assert steps == [(400, 200000.0)] and b == 400 and CS.verify_record(rec2(steps, 400)) == []
    assert CS.next_blocks_v2([(SM, 224999.0)], SM, IDLE, THR) == SM


def test_v2_step_rules():
    assert CS.next_blocks_v2([(400, 249000.0)], SM, IDLE, THR) == SM
    assert CS.next_blocks_v2([(SM, 249000.0)], SM, IDLE, THR) == int(SM * (THR - IDLE) // (249000.0 - IDLE)) == 89
    assert CS.next_blocks_v2([(60, 240000.0)], SM, IDLE, THR) == 30 and CS.next_blocks_v2([(1, 400000.0)], SM, IDLE, THR) == 1
    assert CS.next_blocks_v2([(SM, 400000.0)], SM, 400000.0, THR) == SM - 1      # no dynamic power to extrapolate


def test_v2_budget_spent_uses_the_last_measured_grid_not_converged():
    steps = [(400, 249000.0), (100, 249000.0)]; b = 100
    while len(steps) < CS.MAX_PROBES: b = CS.next_blocks_v2(steps, SM, IDLE, THR); steps.append((b, 249000.0))
    r = rec2(steps, steps[-1][0], converged=False); assert CS.verify_record(r) == []
    r2 = rec2(steps, steps[-1][0] // 2, converged=False); assert any('the rule gives' in p for p in CS.verify_record(r2))


@pytest.mark.parametrize('mutate, text', [
    (lambda r: r['probe_steps'][1].update(blocks=300), 'the rule gives'),
    (lambda r: r.update(blocks_chosen=50, scaled=True), 'the rule gives'),
    (lambda r: r.pop('sm_count'), 'sm_count'),
    (lambda r: r.update(rule_version=3), 'unknown'),
])
def test_v2_replay_refuses_a_choice_the_rule_would_not_make(mutate, text):
    steps, b = run_ladder(ada_like_power); r = rec2(steps, b); mutate(r)
    assert any(text in p for p in CS.verify_record(r)), CS.verify_record(r)


def test_archived_ada_document_replays_with_rule_version_1():
    base = HERE / 'runs' / 'ada_capsafe_20261004' / 'run_full'; doc = next(base.glob('calibration_sm_89_*.json'))
    d = json.loads(doc.read_text()); wins = [w for w in json.dumps(d) and _find_cap_safe(d)]
    assert wins and all('rule_version' not in c for c in wins)
    for c in wins: assert CS.verify_record(c, cap_w=250.0) == [], c


def _find_cap_safe(o):
    out = []
    if isinstance(o, dict):
        if isinstance(o.get('cap_safe'), dict) and o['cap_safe'].get('enabled'): out.append(o['cap_safe'])
        for v in o.values(): out += _find_cap_safe(v)
    elif isinstance(o, list):
        for v in o: out += _find_cap_safe(v)
    return out


# ------------------------------------------------------------ unchanged fit of archived windows
@pytest.mark.parametrize('run', ['energy_20261002', 'energy_repeat_20261002'])
def test_refit_of_archived_blackwell_windows_reproduces_the_committed_rates(run):
    base = HERE / 'runs' / run / 'run_full'; doc = json.loads((base / 'calibration_sm_120_0e63baea.json').read_text())
    windows = json.loads((base / 'stages' / 'energy' / 'windows.json').read_text()); prof = doc['constants']['energy']['profile']
    refit = En.fit_rates([w for w in windows if w['role'] == 'fit'])
    assert refit['base_power_w'] == pytest.approx(prof['base_power_w'], rel=1e-9)
    assert refit['rates_pJ'] == pytest.approx(prof['rates_pJ'], rel=1e-9) and refit['status'] == prof['status']
    assert all('cap_safe' not in w for w in windows)


# ------------------------------------------------------------ opt-in device guard
UUID = 'GPU-0e63baea-0bb1-e90c-f19d-8aa5f2995894'
class R:
    def __init__(s, out, rc=0): s.stdout, s.stderr, s.returncode = out, '', rc
def fake_smi(util=0, apps='', detail=''):
    def run(cmd, **k):
        if '--query-gpu=uuid,utilization.gpu,memory.used' in cmd: return R('%s, %d, 40\n' % (UUID, util))
        if '--query-compute-apps=gpu_uuid,pid,process_name' in cmd: return R(apps)
        if '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory' in cmd: return R(detail)
        raise AssertionError(cmd)
    return run
XORG = '%s, 1234, /usr/lib/xorg/Xorg, 24\n' % UUID


def test_foreign_display_context_refuses_by_default():
    with pytest.raises(D.Refusal, match='compute processes'): D.require_idle(UUID, fake_smi(0, XORG, XORG))
    with pytest.raises(D.Refusal, match='compute processes'): D.wait_idle(UUID, fake_smi(0, XORG, XORG), sleep=lambda s: None)


def test_opt_in_accepts_a_small_passive_display_context_and_records_it():
    r = D.require_idle(UUID, fake_smi(0, XORG, XORG), allow_passive_display=True)
    assert r['utilization_pct'] == 0 and r['foreign_display_processes_allowed'] == [dict(pid='1234', name='Xorg', used_mib=24)]
    w = D.wait_idle(UUID, fake_smi(0, XORG, XORG), sleep=lambda s: None, allow_passive_display=True)
    assert w['foreign_display_processes_allowed'][0]['name'] == 'Xorg'


def test_opt_in_without_foreign_processes_is_unchanged():
    r = D.require_idle(UUID, fake_smi(0, '', ''), allow_passive_display=True)
    assert 'foreign_display_processes_allowed' not in r and r['utilization_pct'] == 0


@pytest.mark.parametrize('apps, why', [
    ('%s, 77, python3, 500\n' % UUID, 'not a display-server'),
    (XORG + '%s, 78, ./train, 100\n' % UUID, 'not a display-server'),
    ('%s, 1234, /usr/lib/xorg/Xorg, 65\n' % UUID, '65 MiB'),
    ('%s, 1234, Xorg, [N/A]\n' % UUID, 'None'.replace('None', 'uses')),
])
def test_opt_in_still_refuses_anything_that_is_not_a_small_display_context(apps, why):
    with pytest.raises(D.Refusal, match=why): D.require_idle(UUID, fake_smi(0, apps, apps), allow_passive_display=True)
    with pytest.raises(D.Refusal): D.wait_idle(UUID, fake_smi(0, apps, apps), sleep=lambda s: None, allow_passive_display=True)


def test_opt_in_requires_zero_utilisation_not_the_usual_five_percent():
    with pytest.raises(D.Refusal, match='1% utilisation'): D.require_idle(UUID, fake_smi(1, XORG, XORG), allow_passive_display=True)
    with pytest.raises(D.Refusal, match='utilisation'): D.wait_idle(UUID, fake_smi(3, XORG, XORG), sleep=lambda s: None, timeout_s=4.0, allow_passive_display=True)
    assert D.require_idle(UUID, fake_smi(3, '', ''), allow_passive_display=True)['utilization_pct'] == 3   # with no foreign process the old 5% rule applies


def test_opt_in_is_a_cli_flag_default_off():
    import run_calibration as RC
    src = (HERE / 'run_calibration.py').read_text()
    assert "'--allow-passive-display-context', action='store_true'" in src and 'wait_idle = lambda u: D.wait_idle(u, allow_passive_display=passive)' in src
