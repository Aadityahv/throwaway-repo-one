"""Cluster (H100 / A100) readiness, CPU only: several allow-list entries per GPU model, no host-name check, the plan path for H100 and A100 from device facts,
and the helpers that turn a UUID job's output into allow-list entries. No GPU, no remote machine."""
import contextlib
import io
import hashlib
import json
import os
import subprocess
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'tools'))
import run_calibration as RC  # noqa: E402
import add_cluster_uuids as ADD  # noqa: E402
import check_uuid_approved as CHK  # noqa: E402
from cal import device as D, runner  # noqa: E402

H1, H2, H3 = ('GPU-%s-2222-3333-4444-555555555555' % x for x in ('11111111', '22222222', '33333333'))
A1 = 'GPU-aaaaaaaa-de0f-b017-49aa-7e54a889eda9'
REAL_ALLOW = HERE / 'approved_devices.json'


def entry(uuid, section, node, cc, sm):
    return {"uuid": uuid, "machine": "cluster", "gpu": section + " test", "ground_truth_section": section, "compute_capability": cc, "sm_count": sm, "node": node}


def facts_for(section, uuid, name):
    gt = D.parse_ground_truth(section)       # the verified HARDWARE_GROUND_TRUTH.md values
    return dict(mode='device_facts', uuid=uuid, name=name, compute_capability=gt['compute_capability'], sm_count=gt['sm_count'], l2_bytes=gt['l2_bytes'],
                shared_per_sm=233472, max_threads_per_sm=2048)


class AllowListSeveralPerModel(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.tmp, True)
        devs = [entry(H1, 'H100', 'node-5.cluster.example.org', '9.0', 132), entry(H2, 'H100', 'node-6.cluster.example.org', '9.0', 132),
                entry(H3, 'H100', 'node-6.cluster.example.org', '9.0', 132), entry(A1, 'A100', 'node1', '8.0', 108),
                {"uuid": "TO-BE-ADDED-BY-the authors-AFTER-LIVE-CHECK", "machine": "cluster", "ground_truth_section": "H100", "compute_capability": "9.0", "sm_count": 132, "pending": True}]
        self.path = self.tmp / 'approved.json'; self.path.write_text(json.dumps(dict(schema='x', devices=devs)))

    def test_several_entries_with_one_section_all_approved_placeholder_ignored(self):
        allow = D.load_allow_list(self.path)
        self.assertEqual(sorted(allow), sorted([H1, H2, H3, A1]))
        for u in (H1, H2, H3): self.assertEqual(D.require_approved(u, allow)['ground_truth_section'], 'H100')

    def test_every_h100_entry_passes_the_cross_check_against_ground_truth(self):
        gt = D.parse_ground_truth('H100'); allow = D.load_allow_list(self.path)
        for u in (H1, H2, H3): D.cross_check(dict(sm_count=gt['sm_count'], l2_bytes=gt['l2_bytes'], compute_capability=gt['compute_capability']), allow[u], gt)

    def test_machine_is_a_label_not_compared_with_the_host_name(self):
        src = (HERE / 'cal' / 'device.py').read_text() + (HERE / 'run_calibration.py').read_text()
        for word in ('gethostname', 'socket', 'platform.node', 'hostname'): self.assertNotIn(word, src)

    def test_check_uuid_approved_paths(self):
        entry_, idle = CHK.check(H2, 'H100', self.path); self.assertEqual(entry_['node'], 'node-6.cluster.example.org'); self.assertIsNone(idle)
        with self.assertRaises(D.Refusal) as cm: CHK.check('GPU-99999999-5149-2f1d-2d0e-edffe525ec25', 'H100', self.path)
        for u in (H1, H2, H3): self.assertIn(u, str(cm.exception))                 # a refusal names the approved UUIDs so the job can be pinned to a node
        with self.assertRaises(D.Refusal): CHK.check(A1, 'H100', self.path)         # approved, but for another section
        with self.assertRaises(D.Refusal): CHK.check('1', 'H100', self.path)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(CHK.main(['--uuid', H1, '--section', 'H100', '--allow-list', str(self.path)]), 0)
            self.assertEqual(CHK.main(['--uuid', A1, '--section', 'H100', '--allow-list', str(self.path)]), 1)

    def test_check_uuid_approved_idle_guard_uses_the_calibrator_guard(self):
        with mock.patch.object(D, 'wait_idle', side_effect=D.Refusal('the approved GPU has compute processes')):
            with self.assertRaises(D.Refusal): CHK.check(H1, 'H100', self.path, require_idle=True)
        with mock.patch.object(D, 'wait_idle', return_value=dict(utilization_pct=0, memory_used_mib=1)):
            self.assertEqual(CHK.check(H1, 'H100', self.path, require_idle=True)[1]['utilization_pct'], 0)


class PlanPathFromFacts(unittest.TestCase):
    """run_calibration.py --plan --facts-json builds the plan on a CPU for the H100 (sm_90) and the A100 (sm_80), from the verified ground-truth values."""
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.tmp, True)

    def plan(self, section, uuid, name, mutate=None):
        facts = facts_for(section, uuid, name)
        if mutate: facts.update(mutate)
        fj = self.tmp / 'facts.json'; fj.write_text(json.dumps(facts))
        allow = {uuid: entry(uuid, section, 'node-6.cluster.example.org', facts_for(section, uuid, name)['compute_capability'], D.parse_ground_truth(section)['sm_count'])}
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(D, 'load_allow_list', return_value=allow), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = RC.main(['--device-uuid', uuid, '--booking-ref', 'booking log 2026-10-02 booking test', '--plan', '--facts-json', str(fj), '--out-dir', str(self.tmp / 'run')])
        return rc, out.getvalue(), err.getvalue(), facts

    def test_h100_plan_builds_for_sm_90(self):
        rc, out, err, facts = self.plan('H100', H1, 'NVIDIA H100 80GB HBM3')
        self.assertEqual(rc, 0, err); self.assertEqual(runner.arch_flag(facts), 'sm_90')
        self.assertIn('132 SMs', out); self.assertIn('compute capability 9.0', out); self.assertIn('stage energy', out); self.assertIn('Pointer-chase tables', out)

    def test_a100_plan_builds_for_sm_80(self):
        rc, out, err, facts = self.plan('A100', A1, 'NVIDIA A100-SXM4-80GB')
        self.assertEqual(rc, 0, err); self.assertEqual(runner.arch_flag(facts), 'sm_80')
        self.assertIn('108 SMs', out); self.assertIn('compute capability 8.0', out)

    def test_plan_refuses_a_wrong_board(self):
        rc, out, err, _ = self.plan('H100', H1, 'NVIDIA H100 80GB HBM3', mutate=dict(sm_count=108))
        self.assertEqual(rc, 1); self.assertIn('device identity mismatch', err)
        rc, out, err, _ = self.plan('H100', H1, 'NVIDIA H100 80GB HBM3', mutate=dict(compute_capability='8.0'))
        self.assertEqual(rc, 1)

    def test_tier_tables_fit_the_limits_on_both_boards(self):
        from cal import stages
        for section in ('H100', 'A100'):
            gt = D.parse_ground_truth(section); t = stages.tier_tables(gt)
            self.assertLessEqual(t['DRAM'], 8 * 1024 * 1024); self.assertGreater(t['DRAM'] * gt['sm_count'], 3 * gt['l2_bytes'] - 1)


class UuidJobHelpers(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.tmp, True)
        self.allow = self.tmp / 'approved.json'; shutil.copy(REAL_ALLOW, self.allow)
        self.gt = self.tmp / 'gt.md'
        txt = D.GROUND_TRUTH.read_text(encoding='utf-8').replace('\r\n', '\n')
        self.gt.write_bytes(txt.replace('\n', '\r\n').encode('utf-8'))           # the working copy on Windows has CRLF; the helper must keep it

    def job_dir(self, name, arch, rows, state='ok: recorded', node='node-6.cluster.example.org', job='4242'):
        d = self.tmp / name; d.mkdir()
        (d / 'meta.txt').write_text('arch_label=%s\nhostname=%s\nslurm_job_id=%s\n' % (arch, node, job))
        hdr = 'index, uuid, pci.bus_id, name, serial\n'
        (d / 'gpu_query.csv').write_text(hdr + '\n'.join(rows) + '\n'); (d / 'allocated_gpu.csv').write_text(hdr + rows[0] + '\n')
        (d / 'COMPLETE').write_text(state + '\nfinished\n')
        return d

    def rows(self, n, name='NVIDIA H100 80GB HBM3'):
        return ['%d, GPU-%08x-2222-3333-4444-555555555555, 00000000:%02X:00.0, %s, 16500000%02d' % (i, 0xabc00000 + i, 0x40 + i, name, i) for i in range(n)]

    def test_entries_use_the_section_values_and_one_entry_per_uuid(self):
        d = self.job_dir('h', 'h100', self.rows(3))
        entries, md, skipped = ADD.build_entries([d], None, self.allow)
        self.assertEqual(len(entries), 3); self.assertEqual(skipped, [])
        for e in entries:
            self.assertEqual((e['machine'], e['ground_truth_section'], e['compute_capability'], e['sm_count']), ('cluster', 'H100', '9.0', 132))
            self.assertEqual(e['source_job'], '4242')
        self.assertEqual(len(md), 3); self.assertIn('node-6.cluster.example.org', md[0])

    def test_allocated_only(self):
        d = self.job_dir('h', 'h100', self.rows(3))
        self.assertEqual(len(ADD.build_entries([d], None, self.allow, allocated_only=True)[0]), 1)

    def test_name_not_matching_the_section_is_refused(self):
        d = self.job_dir('a', 'a100', self.rows(1, 'NVIDIA H100 80GB HBM3'))
        with self.assertRaises(D.Refusal): ADD.build_entries([d], None, self.allow)
        d2 = self.job_dir('b', 'h100', self.rows(1, 'NVIDIA A100-SXM4-80GB'))
        with self.assertRaises(D.Refusal): ADD.build_entries([d2], None, self.allow)

    def test_unfinished_job_malformed_uuid_and_unknown_section_refused(self):
        with self.assertRaises(D.Refusal): ADD.build_entries([self.job_dir('u', 'h100', self.rows(1), state='failed (script stopped)')], None, self.allow)
        with self.assertRaises(D.Refusal): ADD.build_entries([self.job_dir('m', 'h100', ['0, GPU-1234, 00:00, NVIDIA H100 80GB HBM3, 1'])], None, self.allow)
        with self.assertRaises(D.Refusal): ADD.build_entries([self.job_dir('s', 'h200', self.rows(1))], None, self.allow)

    def test_write_adds_entries_and_table_rows_keeps_files_valid_and_idempotent(self):
        d = self.job_dir('h', 'h100', self.rows(2)); gt_before = self.gt.read_bytes()
        h100_before = sum(1 for e in D.load_allow_list(self.allow).values() if e['ground_truth_section'] == 'H100')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ADD.main([str(d), '--write', '--allow-list', str(self.allow), '--ground-truth', str(self.gt)]), 0)
        allow = D.load_allow_list(self.allow)
        self.assertEqual(sum(1 for e in allow.values() if e['ground_truth_section'] == 'H100'), h100_before + 2)
        self.assertIn('GPU-0e63baea-0bb1-e90c-f19d-8aa5f2995894', allow)                    # Blackwell entry untouched
        text = self.allow.read_text(); self.assertEqual(sum(1 for l in text.splitlines() if l.lstrip().startswith('{"uuid"')), len(json.loads(text)['devices']))
        gt_after = self.gt.read_bytes(); self.assertIn(b'\r\n', gt_after); self.assertNotIn(b'\n', gt_after.replace(b'\r\n', b''))
        self.assertIn(b'GPU-abc00000-2222', gt_after); self.assertNotIn(b'| pending: filled from energy_harness/cluster_list_gpu_uuids.sh', gt_after)
        self.assertEqual(D.parse_ground_truth('H100', self.gt.read_text(encoding='utf-8'))['sm_count'], 132)   # sections still parse
        self.assertEqual(D.parse_ground_truth('A100', self.gt.read_text(encoding='utf-8'))['sm_count'], 108)
        self.assertGreater(len(gt_after), len(gt_before))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):                                                 # second run: nothing new
            self.assertEqual(ADD.main([str(d), '--write', '--allow-list', str(self.allow), '--ground-truth', str(self.gt)]), 0)
        self.assertIn('nothing to add', out.getvalue()); self.assertEqual(self.gt.read_bytes(), gt_after)

    def test_real_files_have_the_uuid_table_and_an_entry_for_each_section(self):
        gt = D.GROUND_TRUTH.read_text(encoding='utf-8')
        self.assertIn('## Cluster GPU UUIDs', gt); self.assertIn('| Node | Index | UUID | PCI bus | GPU name | Source |', gt)   # skeleton, filled or not
        devs = json.loads(REAL_ALLOW.read_text())['devices']
        for sec in ('H100', 'A100'): self.assertTrue([x for x in devs if x['ground_truth_section'] == sec and x['machine'] == 'cluster'])


REPO = HERE.parents[3]
BASH = shutil.which('bash')
NEWU = 'GPU-cafe0001-62f3-809c-7bc4-68233240d03d'
FQDN = 'node-9.cluster.example.org'


@unittest.skipUnless(BASH and (REPO / 'energy_harness' / 'run_calibration_cluster.sh').is_file(), 'needs bash and the repo energy_harness/ scripts')
class ApproveAllocatedDryRun(unittest.TestCase):
    """energy_harness/run_calibration_cluster.sh in DRY_RUN mode (canned nvidia-smi rows, no GPU): the APPROVE_ALLOCATED opt-in on a staged copy of the tree."""
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.tmp, True)
        self.tree = self.tmp / 'tree'; (self.tree / 'energy_harness').mkdir(parents=True)
        for f in ('cluster_resolve_toolchain.sh', 'cluster_cal_lib.sh'): shutil.copy(REPO / 'energy_harness' / f, self.tree / 'energy_harness' / f)
        shutil.copy(REPO / 'HARDWARE_GROUND_TRUTH.md', self.tree / 'HARDWARE_GROUND_TRUTH.md')
        shutil.copytree(HERE, self.tree / 'tiresias/framework/predictor/calibrate', ignore=shutil.ignore_patterns('runs', 'tests', '__pycache__', 'data'))
        (self.tree / '.archived_commit').write_text('a' * 40)
        self.staged_allow = self.tree / 'tiresias/framework/predictor/calibrate/approved_devices.json'
        self.orig_hash = hashlib.sha256(REAL_ALLOW.read_bytes()).hexdigest()
        self.njobs = 0

    def run_job(self, uuid=NEWU, name='NVIDIA A100-SXM4-80GB', arch='a100', sm='sm_80', opt=True, host=FQDN, part='partition_a100', scheduler=True, extra=None):
        self.njobs += 1; root = self.tmp / ('o%d' % self.njobs)
        env = dict(os.environ, DRY_RUN='1', ARCH_LABEL=arch, ARCH_SM=sm, HIPC_COMMIT='a' * 40, BOOKING_REF='booking log test booking entry', SOURCE_TREE=str(self.tree),
                   OUT_ROOT=str(root), HOME=str(self.tmp), CAL_PYTHON=sys.executable, DRY_HOSTNAME=host,
                   DRY_SMI_CSV='0, %s, 00000000:4A:00.0, %s, 1650000001' % (uuid, name))
        for k in ('SLURM_JOB_ID', 'SLURM_JOB_PARTITION', 'APPROVE_ALLOCATED', 'TARGET_UUID', 'CUDA_VISIBLE_DEVICES'): env.pop(k, None)
        if opt: env['APPROVE_ALLOCATED'] = '1'
        if scheduler: env['SLURM_JOB_ID'] = '777'
        if part: env['SLURM_JOB_PARTITION'] = part
        env.update(extra or {})
        r = subprocess.run([BASH, str(REPO / 'energy_harness' / 'run_calibration_cluster.sh')], env=env, capture_output=True, text=True)
        outs = list(root.glob('*_*'))
        return r, (outs[0] if outs else None)

    REL = 'tiresias/framework/predictor/calibrate/approved_devices.json'

    def staged_uuids(self, out=None):
        """UUIDs in the JOB's staged copy ($out/src/...), or (no out) in the SOURCE_TREE copy, which the job must never touch."""
        path = (out / 'src' / self.REL) if out else self.staged_allow
        return [e['uuid'] for e in json.loads(path.read_text())['devices']]

    def assertUntouched(self):
        self.assertEqual(hashlib.sha256(REAL_ALLOW.read_bytes()).hexdigest(), self.orig_hash)

    def test_off_and_unknown_uuid_exits_4(self):
        r, out = self.run_job(opt=False)
        self.assertEqual(r.returncode, 4, r.stderr); self.assertNotIn(NEWU, self.staged_uuids(out)); self.assertFalse((out / 'approved_allocated.txt').exists()); self.assertUntouched()
        self.assertNotIn('APPROVE_ALLOCATED', (out / 'STATUS').read_text())

    def test_on_and_good_conditions_adds_one_staged_entry_only(self):
        before = self.staged_uuids(); gt_hash = hashlib.sha256((self.tree / 'HARDWARE_GROUND_TRUTH.md').read_bytes()).hexdigest()
        r, out = self.run_job()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(sorted(self.staged_uuids(out)), sorted(before + [NEWU])); self.assertEqual(self.staged_uuids(), before); self.assertUntouched()
        e = [x for x in json.loads((out / 'src' / self.REL).read_text())['devices'] if x['uuid'] == NEWU][0]
        self.assertEqual((e['machine'], e['ground_truth_section'], e['compute_capability'], e['sm_count'], e['node'], e['source_job']), ('cluster', 'A100', '8.0', 108, FQDN, '777'))
        self.assertEqual(e['approved_by'], 'APPROVE_ALLOCATED opt-in in cluster job 777; user approval 2026-10-02')
        ev = (out / 'approved_allocated.txt').read_text(); self.assertIn(NEWU, ev); self.assertIn('UNCHANGED', ev)
        self.assertTrue((out / 'uuid_evidence' / 'COMPLETE').read_text().startswith('ok'))
        self.assertIn('APPROVE_ALLOCATED', (out / 'STATUS').read_text()); self.assertIn('APPROVE_ALLOCATED', (out / 'summary.txt').read_text())
        self.assertEqual(hashlib.sha256((out / 'src' / 'HARDWARE_GROUND_TRUTH.md').read_bytes()).hexdigest(), gt_hash)    # staged ground truth not edited
        self.assertIn('micro_pipes.cu', r.stdout)                                                                           # compile preflight printed in the dry run

    def test_h100_good_conditions(self):
        u = 'GPU-cafe0002-6d1b-eef7-8937-e4a7cb1250ad'
        r, out = self.run_job(uuid=u, name='NVIDIA H100 80GB HBM3', arch='h100', sm='sm_90', part='partition_h100')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr); self.assertIn(u, self.staged_uuids(out))

    def test_refusals_add_nothing(self):
        cases = dict(wrong_partition=dict(part='partition_h100'), no_partition=dict(part=''), not_cluster=dict(host='ant.example.org'), no_slurm_job=dict(scheduler=False))
        for label, kw in cases.items():
            with self.subTest(label):
                r, out = self.run_job(**kw)
                self.assertEqual(r.returncode, 4, r.stdout + r.stderr); self.assertNotIn(NEWU, self.staged_uuids(out)); self.assertUntouched()
                self.assertFalse((out / 'approved_allocated.txt').exists()); self.assertIn('REFUSED (APPROVE_ALLOCATED)', r.stderr)

    def test_wrong_gpu_name_stopped_before_the_opt_in_with_nothing_added(self):
        r, out = self.run_job(name='NVIDIA H100 80GB HBM3')          # a100 job on an H100: the existing model check stops it (exit 2, before any opt-in)
        self.assertEqual(r.returncode, 2); self.assertNotIn(NEWU, self.staged_uuids(out)); self.assertUntouched()

    def test_malformed_uuid_refused(self):
        r, out = self.run_job(uuid='GPU-1234')
        self.assertEqual(r.returncode, 4, r.stdout + r.stderr); self.assertUntouched()

    def test_already_approved_uuid_needs_no_opt_in(self):
        known = [u for u, e in D.load_allow_list(REAL_ALLOW).items() if e['ground_truth_section'] == 'H100'][0]
        n = len(self.staged_uuids())
        r, out = self.run_job(uuid=known, name='NVIDIA H100 80GB HBM3', arch='h100', sm='sm_90', part='partition_h100')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr); self.assertIn('opt-in not needed', r.stdout); self.assertEqual(len(self.staged_uuids(out)), n)
        self.assertFalse((out / 'approved_allocated.txt').exists())

    def test_target_uuid_pin_takes_precedence(self):
        r, out = self.run_job(extra=dict(TARGET_UUID='GPU-deadbeef-94b3-99c5-6b61-dc31fd15b231'))
        self.assertEqual(r.returncode, 6); self.assertNotIn(NEWU, self.staged_uuids(out))

    def test_host_compiler_note_recorded(self):
        r, out = self.run_job()
        self.assertIn('host_compiler:', (out / 'env.txt').read_text())

    def test_dry_run_runs_the_tensor_sass_gate_and_the_default_stage_list_with_the_tensor_stage(self):
        for arch, name, sm, part in (('h100', 'NVIDIA H100 80GB HBM3', 'sm_90', 'partition_h100'), ('a100', 'NVIDIA A100-SXM4-80GB', 'sm_80', 'partition_a100')):
            r, out = self.run_job(arch=arch, name=name, sm=sm, part=part)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('verify_tensor_sass.py', r.stdout); self.assertIn('--arch %s' % sm, r.stdout.split('verify_tensor_sass.py')[1].splitlines()[0])
            self.assertLess(r.stdout.index('verify_energy_sass.py'), r.stdout.index('verify_tensor_sass.py')); self.assertLess(r.stdout.index('verify_tensor_sass.py'), r.stdout.index('--plan'))
            self.assertNotIn('--stages', r.stdout)            # no stage list given: the calibrator's default list, which includes tensor and energy
        sys.path.insert(0, str(HERE.parent))
        from cal import stages
        self.assertIn('tensor', [s[0] for s in stages.STAGES]); self.assertIn('energy', [s[0] for s in stages.STAGES])

    def test_a_staged_tree_without_the_tensor_files_is_refused_before_any_gpu_work(self):
        (self.tree / 'tiresias/framework/predictor/calibrate/csrc/micro_tensor.cu').unlink()
        r, out = self.run_job(arch='h100', name='NVIDIA H100 80GB HBM3', sm='sm_90', part='partition_h100')
        self.assertNotEqual(r.returncode, 0); self.assertIn('micro_tensor.cu', r.stderr + r.stdout)

    def test_repeats_dry_run_prints_every_repeat_and_keeps_the_single_run_layout(self):
        r, out = self.run_job(extra=dict(REPEATS='3'))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for d in ('/run ', '/run_r2', '/run_r3'): self.assertIn(d, r.stdout)
        self.assertEqual(r.stdout.count('[dry-run] repeat'), 2)
        r1, out1 = self.run_job()
        self.assertNotIn('run_r2', r1.stdout); self.assertNotIn('repeat_', (out1 / 'STATUS').read_text())

    def test_repeats_out_of_range_refused(self):
        for bad in ('0', '4', 'x'):
            r, out = self.run_job(extra=dict(REPEATS=bad)); self.assertEqual(r.returncode, 2, bad)

    def test_aggregate_rc(self):
        lib = str(REPO / 'energy_harness' / 'cluster_cal_lib.sh')
        def agg(*codes): return subprocess.run([BASH, '-c', 'source "%s"; cal_aggregate_rc %s' % (lib.replace(os.sep, '/'), ' '.join(codes))], capture_output=True, text=True).stdout.strip()
        self.assertEqual(agg('0'), '0'); self.assertEqual(agg('0', '0', '0'), '0'); self.assertEqual(agg('1', '1'), '1'); self.assertEqual(agg('0', '1'), '3'); self.assertEqual(agg('0', '3', '0'), '3')

    def test_add_tool_approved_by_field(self):
        d = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, d, True)
        (d / 'meta.txt').write_text('arch_label=a100\nhostname=%s\nslurm_job_id=9\n' % FQDN)
        (d / 'gpu_query.csv').write_text('index, uuid, pci.bus_id, name, serial\n0, %s, 00:4A, NVIDIA A100-SXM4-80GB, 1\n' % NEWU); (d / 'allocated_gpu.csv').write_text((d / 'gpu_query.csv').read_text()); (d / 'COMPLETE').write_text('ok\n')
        e = ADD.build_entries([d], None, REAL_ALLOW, True, 'because')[0][0]; self.assertEqual(e['approved_by'], 'because')
        self.assertNotIn('approved_by', ADD.build_entries([d], None, REAL_ALLOW, True)[0][0])


if __name__ == '__main__':
    unittest.main()
