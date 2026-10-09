"""CPU-only tests of the H100 measurement orchestrator and its shell wrapper (energy_harness/run_h100_eval_measure_cluster.sh in DRY_RUN mode). No GPU, no remote machine.

    python -m pytest tiresias/framework/predictor/h100_eval_freeze/test_measure_h100.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import measure_h100 as M  # noqa: E402

REPO = M.REPO
BASH = shutil.which("bash")
FZ_REL = "tiresias/framework/predictor/h100_eval_freeze"
H100_NAME = "NVIDIA H100 80GB HBM3"


def power_doc(tol=3.0, device=H100_NAME, min_load=20.0, schema=M.POWER_SCHEMA, runs=2):
    return dict(schema=schema, tolerance_pct=tol, runs=[dict(device=dict(name=device)) for _ in range(runs)],
                decision={"padding_0s": dict(min_load_s=min_load, total_window_s=min_load), "padding_1s": dict(min_load_s=10.0, total_window_s=12.0)},
                power_reading_update_period_ms=[100.0, 100.0])


class CheckWindow(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.tmp, True)
        self.p = self.tmp / "power_sensor_results.json"

    def write(self, **kw):
        self.p.write_text(json.dumps(power_doc(**kw)))

    def test_accepts_a_window_the_validation_admits(self):
        self.write()
        r = M.check_window(self.p, 20, 0, tree_root=self.tmp)
        self.assertEqual((r["window_s"], r["padding_s"], r["min_load_s_at_padding_0"]), (20.0, 0.0, 20.0))
        self.assertTrue(M.check_window(self.p, 45, 0, tree_root=self.tmp))

    def refuses(self, window=20, padding=0, tree=True, **kw):
        self.write(**kw)
        with self.assertRaises(SystemExit) as cm:
            M.check_window(self.p, window, padding, tree_root=self.tmp if tree else self.tmp / "elsewhere")
        return str(cm.exception)

    def test_refusals(self):
        self.assertIn("shorter than the shortest window", self.refuses(window=19.9))
        self.assertIn("looser than the 3.0% criterion", self.refuses(tol=5.0))
        self.assertIn("not an H100", self.refuses(device="NVIDIA RTX PRO 6000 Blackwell Workstation Edition"))
        self.assertIn("schema", self.refuses(schema="other"))
        self.assertIn("no padding support", self.refuses(padding=1))
        self.assertIn("no energy window length is admitted", self.refuses(min_load=None))
        self.assertIn("not inside the committed tree", self.refuses(tree=False))
        with self.assertRaises(SystemExit):
            M.check_window(self.tmp / "missing.json", 20, 0)

    def test_window_must_be_positive_number(self):
        self.write()
        with self.assertRaises(SystemExit):
            M.check_window(self.p, 0, 0)
        with self.assertRaises(SystemExit):
            M.check_window(self.p, "x", 0)


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo)] + list(args), capture_output=True, text=True, check=True).stdout.strip()


@unittest.skipUnless(shutil.which("git"), "needs git")
class FreezeGate(unittest.TestCase):
    """check_freeze and write_freeze_record on a throw-away git repository laid out like this one."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.tmp, True)
        self.repo = self.tmp / "r"; self.d = self.repo / FZ_REL; self.d.mkdir(parents=True)
        git(self.repo, "init", "-q"); git(self.repo, "config", "user.email", "t@example.org"); git(self.repo, "config", "user.name", "t"); git(self.repo, "config", "core.autocrlf", "false")
        (self.d / "README").write_text("x")
        git(self.repo, "add", "-A"); git(self.repo, "commit", "-qm", "base")
        self.base = git(self.repo, "rev-parse", "HEAD")
        (self.d / M.PRED_NAME).write_bytes((json.dumps(dict(kind="h100_frozen_predictions", warning=None, calibration=dict(sha256="c" * 64, uuid="GPU-x"), cells={})) + "\n").encode())
        for n in M.FROZEN_INPUTS:
            (self.d / n).parent.mkdir(parents=True, exist_ok=True)
            (self.d / n).write_bytes(b"{}\n")
        (self.repo / M.SAMPLES_MANIFEST_REL).parent.mkdir(parents=True, exist_ok=True); (self.repo / M.SAMPLES_MANIFEST_REL).write_bytes(("0" * 64 + "  ./a.cu\n").encode())
        git(self.repo, "add", "-A"); git(self.repo, "commit", "-qm", "freeze")
        self.freeze = git(self.repo, "rev-parse", "HEAD")

    def record(self):
        self.orig = (M.HERE, M.REPO)
        M.HERE, M.REPO = self.d, self.repo
        try:
            return M.write_freeze_record(self.repo)
        finally:
            M.HERE, M.REPO = self.orig

    def measurement_commit(self):
        (self.d / "measure.txt").write_text("m"); git(self.repo, "add", "-A"); git(self.repo, "commit", "-qm", "measurement code")
        return git(self.repo, "rev-parse", "HEAD")

    def test_tensor_profile_has_its_own_names_inputs_and_needs_no_samples_manifest(self):
        import cell_sets as CS
        prof = CS.profile("tensor")
        self.assertNotEqual(prof["predictions"], M.PRED_NAME)
        self.assertTrue(all("tensor" in n or n.startswith("baselines") for n in CS.frozen_inputs("tensor")))
        (self.d / prof["predictions"]).write_bytes((json.dumps(dict(kind="h100_frozen_predictions", warning=None, calibration=dict(sha256="d" * 64, uuid="GPU-x"), cells={})) + "\n").encode())
        for n in CS.frozen_inputs("tensor"):
            (self.d / n).parent.mkdir(parents=True, exist_ok=True); (self.d / n).write_bytes(b"{}\n")
        git(self.repo, "add", "-A"); git(self.repo, "commit", "-qm", "tensor freeze")
        tfreeze = git(self.repo, "rev-parse", "HEAD")
        self.orig = (M.HERE, M.REPO)
        M.HERE, M.REPO = self.d, self.repo
        try:
            r = M.write_freeze_record(self.repo, profile="tensor")
        finally:
            M.HERE, M.REPO = self.orig
        self.assertEqual((r["profile"], r["freeze_commit"], r["samples_manifest_sha256"]), ("tensor", tfreeze, None))
        git(self.repo, "add", "-A"); git(self.repo, "commit", "-qm", "tensor record")
        hipc = self.measurement_commit()
        got = M.check_freeze(self.repo, hipc, profile="tensor")
        self.assertEqual((got["profile"], got["freeze_commit"]), ("tensor", tfreeze))
        with self.assertRaises(SystemExit):
            M.check_freeze(self.repo, hipc, profile="main")             # the main profile's record does not exist in this repository

    def test_record_then_accept_with_git_ancestry(self):
        r = self.record()
        self.assertEqual(r["freeze_commit"], self.freeze)
        git(self.repo, "add", "-A"); git(self.repo, "commit", "-qm", "record")
        hipc = self.measurement_commit()
        got = M.check_freeze(self.repo, hipc)
        self.assertIn("proper ancestor", got["ancestry"]); self.assertEqual(got["freeze_commit"], self.freeze)

    def test_no_git_mode_relies_on_the_record_and_says_so(self):
        self.record(); hipc = "e" * 40
        got = M.check_freeze(self.repo, hipc, use_git=False)
        self.assertIn("not verified by git", got["ancestry"])

    def test_refusals(self):
        with self.assertRaises(SystemExit) as cm: M.check_freeze(self.repo, "a" * 40)
        self.assertIn("freeze_record_h100.json is missing", str(cm.exception))
        self.record(); hipc = "a" * 40
        # predictions changed after the freeze
        pred = self.d / M.PRED_NAME; saved = pred.read_bytes(); pred.write_bytes(saved + b" ")
        with self.assertRaises(SystemExit) as cm: M.check_freeze(self.repo, hipc, use_git=False)
        self.assertIn("changed after the freeze", str(cm.exception)); pred.write_bytes(saved)
        # another static table changed after the freeze
        (self.d / "cells_h100.json").write_bytes(b'{"a":1}\n')
        with self.assertRaises(SystemExit) as cm: M.check_freeze(self.repo, hipc, use_git=False)
        self.assertIn("cells_h100.json differs", str(cm.exception)); (self.d / "cells_h100.json").write_bytes(b"{}\n")
        # freeze commit equal to HIPC_COMMIT
        with self.assertRaises(SystemExit) as cm: M.check_freeze(self.repo, self.freeze, use_git=False)
        self.assertIn("predates", str(cm.exception))
        # plumbing-test file
        pred.write_bytes((json.dumps(dict(kind="plumbing_test_not_h100", warning="PLUMBING")) + "\n").encode())
        with self.assertRaises(SystemExit) as cm: M.check_freeze(self.repo, hipc, use_git=False)
        self.assertIn("not an H100 frozen-predictions file", str(cm.exception))
        pred.write_bytes(saved)
        # missing predictions
        pred.unlink()
        with self.assertRaises(SystemExit) as cm: M.check_freeze(self.repo, hipc, use_git=False)
        self.assertIn("must be frozen and committed", str(cm.exception))

    def test_not_an_ancestor_is_refused(self):
        self.record(); git(self.repo, "add", "-A"); git(self.repo, "commit", "-qm", "record")
        git(self.repo, "checkout", "-q", "-b", "side", self.base)
        (self.d / "other.txt").write_text("o"); git(self.repo, "add", "-A"); git(self.repo, "commit", "-qm", "side")
        # the record file only exists on the first branch: carry it over untracked so the tree has it, but HIPC_COMMIT (side) does not contain the freeze commit
        side = git(self.repo, "rev-parse", "HEAD")
        git(self.repo, "checkout", "-q", "-f", "-")  # back to the first branch
        with self.assertRaises(SystemExit) as cm: M.check_freeze(self.repo, side)
        self.assertIn("not an ancestor", str(cm.exception))

    def test_record_refuses_when_the_working_tree_differs_from_the_committed_blob(self):
        (self.d / M.PRED_NAME).write_bytes(b'{"kind": "h100_frozen_predictions"}\n')
        with self.assertRaises(SystemExit) as cm: self.record()
        self.assertIn("differs from its blob", str(cm.exception))


class GpuIdentity(unittest.TestCase):
    def smi(self, rows, rc=0):
        class R:
            returncode = rc; stdout = "\n".join(rows); stderr = ""
        return lambda *a, **k: R()

    U1, U2 = "GPU-11111111-666f-f6cc-aa5b-3c07feaa3a95", "GPU-22222222-b802-e130-0a97-d347bcba8646"

    def test_one_visible_gpu(self):
        g = M.gpu_identity(self.smi(["0, %s, %s" % (self.U1, H100_NAME)]), environ={})
        self.assertEqual((g["index"], g["uuid"]), (0, self.U1))

    def test_several_need_an_integer_index_and_must_be_an_h100(self):
        rows = ["0, %s, %s" % (self.U1, H100_NAME), "1, %s, %s" % (self.U2, H100_NAME)]
        self.assertEqual(M.gpu_identity(self.smi(rows), environ={"CUDA_VISIBLE_DEVICES": "1"})["uuid"], self.U2)
        with self.assertRaises(SystemExit): M.gpu_identity(self.smi(rows), environ={"CUDA_VISIBLE_DEVICES": self.U2})
        with self.assertRaises(SystemExit): M.gpu_identity(self.smi(["0, %s, NVIDIA A100-SXM4-80GB" % self.U1]), environ={})
        with self.assertRaises(SystemExit): M.gpu_identity(self.smi([], rc=1), environ={})


class SassGate(unittest.TestCase):
    SASS = "        /*0000*/                   MOV R1, c[0x0][0x28] ;                        /* 0x00000a0000017a02 */\n" \
           "                                                                                /* 0x000fc40000000f00 */\n" \
           "        /*0010*/              @P0  EXIT ;                                        /* 0x000000000000794d */\n"

    def test_equal_and_different_sequences(self):
        want = M.parse_sass_text(self.SASS)
        manifest = dict(kernels=dict(k1=dict(symbol="_Z1kv", instruction_sequence_sha256=want), k2=dict(symbol="_Z2kv", instruction_sequence_sha256="0" * 64)))

        class R:
            returncode = 0; stderr = ""
            stdout = SassGate.SASS
        gate = M.sass_gate(dict(k1="bin", k2="bin", k3="bin"), "nvcc", manifest=manifest, runner=lambda *a, **k: R())
        self.assertTrue(gate["k1"]["ok"]); self.assertFalse(gate["k2"]["ok"]); self.assertFalse(gate["k3"]["ok"])
        cells = [dict(cell_id="a", kernels=[dict(kid="k1")]), dict(cell_id="b", kernels=[dict(kid="k1"), dict(kid="k2")])]
        ok, refused = M.cells_without_failed_kernels(cells, gate)
        self.assertEqual([c["cell_id"] for c in ok], ["a"]); self.assertEqual(refused, [dict(cell_id="b", failed_kernels=["k2"])])

    def test_every_cell_kernel_is_in_the_committed_manifest(self):
        import cell_sets as CS
        for profile in ("main", "tensor"):
            man = CS.merged_sass_manifest(profile)
            _, cells = M.load_cells(profile=profile)
            self.assertEqual({k["kid"] for c in cells for k in c["kernels"]} - set(man), set(), profile)


class PlanAndDryRun(unittest.TestCase):
    def test_plan_counts_and_estimate(self):
        p = M.plan(20.0)
        self.assertEqual(p["cells"], 112)                                    # 72 CUDA-samples cells + 40 machine-learning-kernel cells
        self.assertEqual(sum(p["by_set_family_tier"].values()), 112)
        self.assertAlmostEqual(p["gpu_seconds"]["energy"], 112 * (90 + 20 + 46))
        p40 = M.plan()                                                       # default window: the H100's 40 s, not Blackwell's 15 s
        self.assertEqual(p40["window_s"], 40.0); self.assertAlmostEqual(p40["gpu_seconds"]["energy"], 112 * (90 + 40 + 46))
        pt = M.plan(40.0, profile="tensor")
        self.assertEqual(pt["cells"], 16); self.assertAlmostEqual(pt["gpu_seconds"]["energy"], 16 * (90 + 40 + 46))

    def run_main(self, *argv):
        import io, contextlib
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = M.main(list(argv))
            except SystemExit as ex:
                rc = ex.code
        return rc, out.getvalue(), err.getvalue()

    def test_energy_without_a_window_validation_is_refused(self):
        rc, out, err = self.run_main("run", "--stage", "all", "--measure", "energy", "--dry-run")
        self.assertIn("REFUSED", str(rc)); self.assertIn("power-sensor update test", str(rc))
        rc, out, err = self.run_main("run", "--stage", "all", "--measure", "both", "--dry-run")
        self.assertIn("REFUSED", str(rc))

    def test_runtime_only_dry_run_lists_every_cell(self):
        rc, out, err = self.run_main("run", "--stage", "all", "--measure", "runtime", "--dry-run")
        self.assertEqual(rc, 0)
        self.assertEqual(out.count("\nh100/"), 112)
        rc, out, err = self.run_main("run", "--stage", "all", "--measure", "runtime", "--dry-run", "--profile", "tensor")
        self.assertEqual(rc, 0)
        self.assertEqual(out.count("\nh100/"), 16)


# ------------------------------------------------------------------------------------------------ the shell wrapper
@unittest.skipUnless(BASH and (REPO / "energy_harness" / "run_h100_eval_measure_cluster.sh").is_file(), "needs bash and the repo energy_harness/ scripts")
class ShellDryRun(unittest.TestCase):
    UUID = "GPU-cafe0003-c01c-6fbe-8308-7fbade573b5d"
    HIPC = "a" * 40
    FREEZE_COMMIT = "b" * 40
    FILE_TEXT = "int main() { return 0; }\n"
    MANIFEST_TEXT = hashlib.sha256(FILE_TEXT.encode()).hexdigest() + "  ./cpp/a.cu\n"

    def make_samples(self, name="cuda-samples-5443602", manifest_text=None):
        """A stand-in for the Cluster copy: one source file, its MANIFEST.sha256 (the committed text by default) and .source_rev."""
        root = self.tmp / name
        (root / "cpp").mkdir(parents=True)
        (root / "cpp" / "a.cu").write_bytes(self.FILE_TEXT.encode())
        (root / "MANIFEST.sha256").write_bytes((manifest_text or self.MANIFEST_TEXT).encode())
        (root / ".source_rev").write_text(M.SOURCES_REVISION + "\n")
        return root

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.tmp, True)
        self.tree = self.tmp / "tree"
        (self.tree / "energy_harness").mkdir(parents=True)
        for f in ("cluster_resolve_toolchain.sh", "cluster_cal_lib.sh", "application_energy_harness.py", "measurement_runner.py", "nvml_sampler.py", "verify_b_stabilization_trace.py"):
            shutil.copy(REPO / "energy_harness" / f, self.tree / "energy_harness" / f)
        shutil.copy(REPO / "HARDWARE_GROUND_TRUTH.md", self.tree / "HARDWARE_GROUND_TRUTH.md")
        sr = self.tree / "tiresias/framework/predictor"
        shutil.copytree(sr_src := REPO / "tiresias/framework/predictor/calibrate", sr / "calibrate", ignore=shutil.ignore_patterns("runs", "tests", "__pycache__", "data"))
        self.fz = self.tree / FZ_REL; self.fz.mkdir(parents=True)
        for f in ("measure_h100.py", "cells_h100.json", "sass_manifest_h100.json", "make_sass_manifest_h100.py", "cell_sets.py", "window_gate.py"):
            shutil.copy(HERE / f, self.fz / f)
        (self.fz / "ml_sets").mkdir()
        for f in ("cells_ml_h100.json", "cells_tensor_h100.json", "sass_manifest_ml_h100.json", "sass_manifest_tensor_h100.json"):
            shutil.copy(HERE / "ml_sets" / f, self.fz / "ml_sets" / f)
        for rel in ("fresh_f/run_f.py", "fresh_f/gpu/drivers/driver_ml.cu", "fresh_f/gpu/drivers/driver_common.h", "fresh_f/src/ml_kernels.cuh", "fresh_g/src/tc_kernels.cuh", "fresh_h/src/attn_kernels.cuh"):
            (sr / rel).parent.mkdir(parents=True, exist_ok=True); (sr / rel).write_text("# placeholder\n")
        for rel in ("fresh_e/gpu/run_e.py", "unseen_kernels/gpu/run_unseen.py", "fresh_d/timing_fresh_d.py"):
            (sr / rel).parent.mkdir(parents=True, exist_ok=True); (sr / rel).write_text("# placeholder\n")
        (self.tree / "tiresias/app_runners").mkdir(parents=True); (self.tree / "tiresias/app_runners/copy_runner.py").write_text("# placeholder\n")
        self.write_freeze()
        (self.tree / ".archived_commit").write_text(self.HIPC)
        self.n = 0

    def write_freeze(self, kind="h100_frozen_predictions"):
        pred = self.fz / M.PRED_NAME
        pred.write_bytes((json.dumps(dict(kind=kind, warning=None, calibration=dict(sha256="c" * 64, uuid=self.UUID), cells={})) + "\n").encode())
        for n in M.FROZEN_INPUTS:
            if not (self.fz / n).is_file():
                (self.fz / n).parent.mkdir(parents=True, exist_ok=True)
                (self.fz / n).write_bytes(b"{}\n")
        mf = self.tree / M.SAMPLES_MANIFEST_REL
        mf.parent.mkdir(parents=True, exist_ok=True)
        if not mf.exists():
            mf.write_bytes(self.MANIFEST_TEXT.encode())
        rec = dict(schema="h100_freeze_record/1", freeze_commit=self.FREEZE_COMMIT, predictions_sha256=hashlib.sha256(pred.read_bytes()).hexdigest(),
                   sha256={n: hashlib.sha256((self.fz / n).read_bytes()).hexdigest() for n in M.FROZEN_INPUTS}, samples_manifest_sha256=hashlib.sha256(mf.read_bytes()).hexdigest())
        (self.fz / M.RECORD_NAME).write_text(json.dumps(rec))

    def write_validation(self, **kw):
        p = self.fz.parent / "h100_power_sensor_test" / "power_sensor_results.json"
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(power_doc(**kw)))
        return p

    def job(self, name=H100_NAME, **extra):
        self.n += 1; root = self.tmp / ("o%d" % self.n)
        env = dict(os.environ, DRY_RUN="1", HIPC_COMMIT=self.HIPC, BOOKING_REF="booking log test booking entry", SOURCE_TREE=self.tree.as_posix(), OUT_ROOT=str(root), HOME=str(self.tmp),
                   CAL_PYTHON=sys.executable, DRY_HOSTNAME="node-9.cluster.example.org", DRY_SMI_CSV="0, %s, 00000000:4A:00.0, %s, 1650000001" % (self.UUID, name))
        for k in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "APPROVE_ALLOCATED", "TARGET_UUID", "CUDA_VISIBLE_DEVICES", "WINDOW_VALIDATION", "WINDOW_S", "PADDING_S", "MEASURE", "ONLY", "STAGE", "PROFILE"):
            env.pop(k, None)
        env.update({k: str(v) for k, v in extra.items()})
        r = subprocess.run([BASH, str(REPO / "energy_harness" / "run_h100_eval_measure_cluster.sh")], env=env, capture_output=True, text=True)
        outs = list(root.glob("h100_eval_*"))
        return r, (outs[0] if outs else None)

    def approve_uuid(self):
        """Put the test UUID into the staged tree's allow-list (a real allow-list entry shape)."""
        p = self.tree / "tiresias/framework/predictor/calibrate/approved_devices.json"
        d = json.loads(p.read_text())
        d["devices"].append({"uuid": self.UUID, "machine": "cluster", "gpu": "H100 test", "ground_truth_section": "H100", "compute_capability": "9.0", "sm_count": 132, "node": "x"})
        p.write_text(json.dumps(d))

    def test_runtime_only_accepted_without_a_window_validation(self):
        self.approve_uuid()
        r, out = self.job(MEASURE="runtime")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("dry run", (out / "STATUS").read_text()); self.assertIn("--measure runtime", r.stdout)
        self.assertTrue((out / "freeze_check.json").is_file()); self.assertIn("not verified by git", (out / "freeze_check.json").read_text())
        self.assertNotIn("--window-validation", r.stdout)

    def write_tensor_freeze(self):
        import cell_sets as CS
        prof = CS.profile("tensor")
        pred = self.fz / prof["predictions"]
        pred.write_bytes((json.dumps(dict(kind="h100_frozen_predictions", warning=None, calibration=dict(sha256="d" * 64, uuid=self.UUID), cells={})) + "\n").encode())
        for n in CS.frozen_inputs("tensor"):
            if not (self.fz / n).is_file():
                (self.fz / n).parent.mkdir(parents=True, exist_ok=True); (self.fz / n).write_bytes(b"{}\n")
        rec = dict(schema="h100_freeze_record/1", profile="tensor", freeze_commit=self.FREEZE_COMMIT, predictions_sha256=hashlib.sha256(pred.read_bytes()).hexdigest(),
                   sha256={n: hashlib.sha256((self.fz / n).read_bytes()).hexdigest() for n in CS.frozen_inputs("tensor")}, samples_manifest_sha256=None)
        (self.fz / prof["record"]).write_text(json.dumps(rec))

    def test_tensor_profile_job_checks_its_own_freeze_and_needs_no_samples_copy(self):
        self.approve_uuid()
        r, out = self.job(MEASURE="runtime", PROFILE="tensor")                        # no tensor freeze in the tree yet: refused by the freeze gate (7), not by a missing file
        self.assertEqual(r.returncode, 7, r.stdout + r.stderr); self.assertIn("predictions_h100_tensor.json", r.stderr)
        self.write_tensor_freeze()
        r, out = self.job(MEASURE="runtime", PROFILE="tensor", SAMPLES_ROOT=self.tmp / "no_such_samples_copy")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("PROFILE=tensor uses no cuda-samples copy", r.stdout)
        self.assertIn("--profile tensor", r.stdout)
        plan = (out / "dry_run_plan.txt").read_text()
        self.assertEqual(plan.count("\nh100/"), 16); self.assertNotIn("\nh100/unseen", plan)
        r, _ = self.job(MEASURE="runtime", PROFILE="nonsense")
        self.assertEqual(r.returncode, 2)
        r, out = self.job(MEASURE="runtime")                                         # the main profile still has its sources copy gate and finds its own freeze
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual((out / "dry_run_plan.txt").read_text().count("\nh100/"), 112)

    def test_the_default_energy_window_in_the_plan_is_the_h100s_40_s(self):
        self.approve_uuid()
        r, out = self.job(MEASURE="runtime")
        plan = json.loads((out / "plan.json").read_text())
        self.assertEqual(plan["window_s"], 40.0); self.assertEqual(plan["cells"], 112)

    def test_energy_default_is_refused_without_a_validation(self):
        self.approve_uuid()
        r, out = self.job()                                     # MEASURE defaults to both
        self.assertEqual(r.returncode, 7, r.stdout + r.stderr); self.assertIn("power-sensor update test", r.stderr)
        self.assertIn("exit_code=7", (out / "STATUS").read_text())

    def test_energy_accepted_with_an_admitting_validation_and_refused_otherwise(self):
        self.approve_uuid()
        p = self.write_validation()
        rel = str(p.relative_to(self.tree)).replace("\\", "/")
        r, out = self.job(MEASURE="both", WINDOW_VALIDATION=rel, WINDOW_S=20)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr); self.assertIn("window_check", " ".join(x.name for x in out.iterdir()))
        self.assertIn("--window-validation", r.stdout)
        r, _ = self.job(MEASURE="both", WINDOW_VALIDATION=p.as_posix(), WINDOW_S=20)                         # absolute path inside SOURCE_TREE is accepted too
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        r, _ = self.job(MEASURE="energy", WINDOW_VALIDATION=rel, WINDOW_S=10)                          # shorter than the validation admits
        self.assertEqual(r.returncode, 7); self.assertIn("shorter than the shortest window", r.stderr)
        r, _ = self.job(MEASURE="energy", WINDOW_VALIDATION=rel, WINDOW_S=20, PADDING_S=1)             # harness has no padding support
        self.assertEqual(r.returncode, 7); self.assertIn("no padding support", r.stderr)
        r, _ = self.job(MEASURE="energy", WINDOW_VALIDATION="/etc/hostname", WINDOW_S=20)
        self.assertEqual(r.returncode, 7)
        self.write_validation(device="NVIDIA RTX PRO 6000 Blackwell Workstation Edition")
        r, _ = self.job(MEASURE="energy", WINDOW_VALIDATION=rel, WINDOW_S=20)
        self.assertEqual(r.returncode, 7); self.assertIn("not an H100", r.stderr)

    def test_missing_or_altered_predictions_are_refused(self):
        self.approve_uuid()
        (self.fz / M.PRED_NAME).write_bytes((self.fz / M.PRED_NAME).read_bytes() + b" ")
        r, _ = self.job(MEASURE="runtime")
        self.assertEqual(r.returncode, 7); self.assertIn("changed after the freeze", r.stderr)
        self.write_freeze(kind="plumbing_test_not_h100")
        r, _ = self.job(MEASURE="runtime")
        self.assertEqual(r.returncode, 7); self.assertIn("not an H100 frozen-predictions file", r.stderr)
        (self.fz / M.PRED_NAME).unlink()
        r, _ = self.job(MEASURE="runtime")
        self.assertEqual(r.returncode, 7)
        self.write_freeze(); (self.fz / M.RECORD_NAME).unlink()
        r, _ = self.job(MEASURE="runtime")
        self.assertEqual(r.returncode, 7); self.assertIn("freeze_record_h100.json is missing", r.stderr)

    def test_freeze_commit_equal_to_the_tree_commit_is_refused(self):
        self.approve_uuid(); self.FREEZE_COMMIT = self.HIPC; self.write_freeze()
        r, _ = self.job(MEASURE="runtime")
        self.assertEqual(r.returncode, 7); self.assertIn("predates", r.stderr)

    def test_gpu_gates(self):
        r, _ = self.job(MEASURE="runtime")                                                        # not in the allow-list
        self.assertEqual(r.returncode, 4, r.stdout + r.stderr)
        self.approve_uuid()
        r, _ = self.job(MEASURE="runtime", TARGET_UUID="GPU-deadbeef-94b3-99c5-6b61-dc31fd15b231")
        self.assertEqual(r.returncode, 6)
        r, _ = self.job(name="NVIDIA A100-SXM4-80GB", MEASURE="runtime")
        self.assertNotEqual(r.returncode, 0)

    def test_approve_allocated_adds_one_staged_entry_only(self):
        before = (self.tree / "tiresias/framework/predictor/calibrate/approved_devices.json").read_bytes()
        r, out = self.job(MEASURE="runtime", APPROVE_ALLOCATED=1, SLURM_JOB_ID=777, SLURM_JOB_PARTITION="partition_h100")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual((self.tree / "tiresias/framework/predictor/calibrate/approved_devices.json").read_bytes(), before)
        staged = (out / "src/tiresias/framework/predictor/calibrate/approved_devices.json").read_text()
        self.assertIn(self.UUID, staged); self.assertIn("APPROVE_ALLOCATED", (out / "STATUS").read_text())
        r, out = self.job(MEASURE="runtime", APPROVE_ALLOCATED=1, SLURM_JOB_ID=778, SLURM_JOB_PARTITION="partition_a100")        # wrong partition for an H100 job
        self.assertEqual(r.returncode, 4)

    def test_scratch_sources_and_output_are_refused(self):
        self.approve_uuid()
        r, _ = self.job(MEASURE="runtime", SAMPLES_ROOT="/scratch/shareduser/app_sources/cuda-samples-5443602")
        self.assertEqual(r.returncode, 2); self.assertIn("/scratch", r.stderr)
        r, _ = self.job(MEASURE="runtime", OUT_ROOT="/scratch/shareduser/h100_eval_runs")
        self.assertEqual(r.returncode, 2); self.assertIn("/scratch", r.stderr)

    def test_default_sources_path_is_the_project_copy(self):
        self.approve_uuid()
        r, out = self.job(MEASURE="runtime")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("samples_root=/home/user/tiresias/app_sources/cuda-samples-5443602", r.stdout)
        self.assertIn("does not exist on this machine", r.stdout); self.assertIn("sha256sum -c --quiet MANIFEST.sha256", r.stdout)

    def test_sources_gate_accepts_a_matching_copy_and_refuses_any_difference_with_exit_2(self):
        self.approve_uuid()
        root = self.make_samples()
        r, out = self.job(MEASURE="runtime", SAMPLES_ROOT=root.as_posix())
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr); self.assertIn("matches the committed manifest", r.stdout)
        (root / "MANIFEST.sha256").write_bytes((self.MANIFEST_TEXT + "# extra\n").encode())                      # manifest not byte-identical
        r, out = self.job(MEASURE="runtime", SAMPLES_ROOT=root.as_posix())
        self.assertEqual(r.returncode, 2); self.assertIn("not byte-identical", r.stderr); self.assertIn("exit_code=2", (out / "STATUS").read_text())
        root2 = self.make_samples("copy2"); (root2 / "cpp" / "a.cu").write_bytes(b"tampered\n")          # file differs from the manifest: sha256sum -c fails
        r, _ = self.job(MEASURE="runtime", SAMPLES_ROOT=root2.as_posix())
        self.assertEqual(r.returncode, 2); self.assertIn("sha256sum -c", r.stderr)
        r, _ = self.job(MEASURE="runtime", SAMPLES_ROOT=(self.tmp / "nowhere").as_posix())                # dry run only: the real job refuses a missing copy (exit 2), a dry run on another machine cannot see it
        self.assertEqual(r.returncode, 0); self.assertIn("does not exist on this machine", r.stdout)
        root3 = self.make_samples("copy3"); (root3 / "MANIFEST.sha256").unlink()                          # the copy without its manifest
        r, _ = self.job(MEASURE="runtime", SAMPLES_ROOT=root3.as_posix())
        self.assertEqual(r.returncode, 2); self.assertIn("MANIFEST.sha256 does not exist", r.stderr)
        (self.tree / M.SAMPLES_MANIFEST_REL).unlink()                                                    # the committed manifest missing from the tree
        r, _ = self.job(MEASURE="runtime", SAMPLES_ROOT=root.as_posix())
        self.assertNotEqual(r.returncode, 0)

    def test_a_tree_without_the_baselines_record_is_refused(self):
        self.approve_uuid()
        rec = json.loads((self.fz / M.RECORD_NAME).read_text()); del rec["sha256"][M.BASELINES_NAME]; (self.fz / M.RECORD_NAME).write_text(json.dumps(rec))
        r, _ = self.job(MEASURE="runtime")
        self.assertEqual(r.returncode, 7); self.assertIn("baselines", r.stderr)

    def test_variables_are_validated_before_anything_else(self):
        r, _ = self.job(MEASURE="sometimes")
        self.assertEqual(r.returncode, 2)
        r, _ = self.job(MEASURE="runtime", STAGE="nothing")
        self.assertEqual(r.returncode, 2)
        r, _ = self.job(MEASURE="runtime", HIPC_COMMIT="abc")
        self.assertEqual(r.returncode, 2)

    def test_stamp_must_match(self):
        self.approve_uuid()
        (self.tree / ".archived_commit").write_text("c" * 40)
        r, _ = self.job(MEASURE="runtime")
        self.assertEqual(r.returncode, 2); self.assertIn("not stamped", r.stderr)

    def test_never_overwrites(self):
        self.approve_uuid()
        r, out = self.job(MEASURE="runtime", SLURM_JOB_ID=55, OUT_ROOT=str(self.tmp / "same"))
        self.assertEqual(r.returncode, 0, r.stderr)
        r, _ = self.job(MEASURE="runtime", SLURM_JOB_ID=55, OUT_ROOT=str(self.tmp / "same"))
        self.assertNotEqual(r.returncode, 0); self.assertIn("refusing to overwrite", r.stderr)


class EngineWiring(unittest.TestCase):
    """The Blackwell engines can be loaded and pointed at sm_90 and an H100 GPU; their argv for every H100 cell is consistent with the cell's own kernel launches (cell_argv_check)."""

    def test_every_engine_cell_has_a_consistent_driver_argv(self):
        gpu = dict(index=0, uuid="GPU-11111111-666f-f6cc-aa5b-3c07feaa3a95", name=H100_NAME)
        _, cells = M.load_cells()
        by = {s: [c for c in cells if c["set"] == s] for s in M.SET_FAMILY}
        n = 0
        for kind, s in (("unseen", "unseen"), ("fresh_e", "set_e")):
            eng = M.patch_engine(M.load_engine(kind), kind, gpu, by[s], "/tmp/x")
            self.assertEqual((eng.ARCH, eng.PLATFORM, eng.APPROVED_UUID), ("sm_90", "h100", gpu["uuid"]))
            for c in eng.load_cells():
                eng.cell_argv_check(c)       # raises SystemExit if the driver geometry differs from the cell's kernels
                n += 1
        self.assertEqual(n, 48)

    def test_fresh_engine_is_pointed_at_the_h100_and_every_ml_cell_has_a_consistent_argv(self):
        import cell_sets as CS
        gpu = dict(index=0, uuid="GPU-11111111-666f-f6cc-aa5b-3c07feaa3a95", name=H100_NAME)
        n = 0
        for profile, expect in (("main", 40), ("tensor", 16)):
            _, cells = M.load_cells(profile=profile)
            fresh = [c for c in cells if c["set"] in CS.ML_SETS]
            eng = M.patch_fresh_engine(M.load_fresh_engine(), gpu, fresh)
            self.assertEqual((eng.ARCH, eng.PLATFORM, eng.APPROVED_UUID, eng.GPU_INDEX), ("sm_90", "h100", gpu["uuid"], 0))
            self.assertIn("release 12.1", eng.NVCC_RELEASE)
            self.assertEqual(len(eng.load_cells()), expect)
            for c in eng.load_cells():
                argv = eng.argv_of(c)
                self.assertEqual(argv[0], c["family"])
                self.assertTrue(all(isinstance(x, str) for x in argv))
                n += 1
            gate = {k: dict(ok=True, detail="x", symbol="s") for k in M.fresh_cell_kids(fresh)}
            tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, tmp, True)
            rec = M.write_fresh_gate(tmp, gate, M.fresh_cell_kids(fresh))
            self.assertTrue(all(r["sass_equal"] for r in rec)); self.assertTrue((tmp / "gate.json").is_file())
        self.assertEqual(n, 56)

    def test_fresh_engine_time_loop_runs_on_a_fake_driver_and_refuses_a_failed_gate(self):
        """The engine's own timing loop, pointed at the H100 by patch_fresh_engine, with nvidia-smi and the driver replaced: proves the guard, the manifest and gate files, the argv and the result schema."""
        import cell_sets as CS
        gpu = dict(index=0, uuid="GPU-11111111-666f-f6cc-aa5b-3c07feaa3a95", name=H100_NAME)
        _, cells = M.load_cells(profile="tensor")
        eng = M.patch_fresh_engine(M.load_fresh_engine(), gpu, cells)
        tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, tmp, True)
        binary = tmp / "driver_ml"; binary.write_bytes(b"not a real binary")
        (tmp / "manifest.json").write_text(json.dumps(dict(binary=str(binary), binary_sha256=eng.sha256_file(binary))))
        eng.smi = lambda args, timeout=30: gpu["uuid"] if "--query-gpu=uuid" in " ".join(args) else ""
        seen = []

        def fake_driver(binary_path, pos, out_path, repeat, batch, d, env, timeout=3600):
            seen.append((pos[0], env.get("CUDA_VISIBLE_DEVICES")))
            return dict(rc=0, check_ok=True, check="CHECK_OK fake", windows=dict(launches=repeat, cuda_seconds=repeat * 2e-5, per_launch_s=2e-5))
        eng.run_driver = fake_driver
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        try:
            kids = M.fresh_cell_kids(cells)
            M.write_fresh_gate(tmp, {k: dict(ok=False, detail="differs", symbol="s") for k in kids}, kids)
            args = M.engine_args(workdir=str(tmp), only="", booking_ref="booking log test booking", out=str(tmp / "timing.json"))
            with self.assertRaises(SystemExit) as cm:
                eng.cmd_time(args)
            self.assertIn("hash gate failed", str(cm.exception))
            M.write_fresh_gate(tmp, {k: dict(ok=True, detail="identical", symbol="s") for k in kids}, kids)
            self.assertEqual(eng.cmd_time(args), 0)
        finally:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        res = json.loads((tmp / "timing.json").read_text())
        self.assertEqual(len(res["cells"]), 16); self.assertTrue(res["all_correct"])
        self.assertEqual({r["cell_id"] for r in res["cells"]}, {c["cell_id"] for c in cells})
        self.assertTrue(all(abs(r["per_launch_runtime_s"] - 2e-5) < 1e-12 for r in res["cells"]))
        self.assertEqual({fam for fam, _ in seen}, {"tcgemm", "attn"}); self.assertEqual({u for _, u in seen}, {gpu["uuid"]})

    def test_set_d_cells_carry_complete_timing_specs(self):
        _, cells = M.load_cells()
        d = [c for c in cells if c["set"] == "set_d"]
        self.assertEqual(len(d), 24)
        for c in d:
            self.assertEqual(set(c["timing"]), {"runner", "numeric_args", "input_kind", "elements", "oracle", "file_args"})


class ProtocolConstants(unittest.TestCase):
    def test_precondition_matches_the_harness(self):
        import re
        text = (REPO / "energy_harness" / "measurement_runner.py").read_text()
        self.assertEqual(float(re.search(r"^RAW_TRACE_PRECONDITION_SECONDS = ([0-9.]+)", text, re.M).group(1)), M.PRECONDITION_S)

    def test_arch_and_platform_are_the_h100_ones(self):
        self.assertEqual((M.ARCH, M.PLATFORM), ("sm_90", "h100"))


class RunRefusals(unittest.TestCase):
    """cmd_run refuses, in order, before touching a GPU: freeze gate, window gate, booking."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.tmp, True)
        self.fz = self.tmp / FZ_REL; self.fz.mkdir(parents=True)
        pred = self.fz / M.PRED_NAME
        pred.write_bytes((json.dumps(dict(kind="h100_frozen_predictions", warning=None, calibration=dict(sha256="c" * 64, uuid="GPU-x"), cells={})) + "\n").encode())
        for n in M.FROZEN_INPUTS:
            (self.fz / n).parent.mkdir(parents=True, exist_ok=True)
            (self.fz / n).write_bytes(b"{}\n")
        mf = self.tmp / M.SAMPLES_MANIFEST_REL; mf.parent.mkdir(parents=True, exist_ok=True); mf.write_bytes(("0" * 64 + "  ./a.cu\n").encode())
        (self.fz / M.RECORD_NAME).write_text(json.dumps(dict(schema="h100_freeze_record/1", freeze_commit="b" * 40, predictions_sha256=hashlib.sha256(pred.read_bytes()).hexdigest(),
                                                              sha256={n: hashlib.sha256((self.fz / n).read_bytes()).hexdigest() for n in M.FROZEN_INPUTS},
                                                              samples_manifest_sha256=hashlib.sha256(mf.read_bytes()).hexdigest())))

    def run_main(self, *argv):
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                return M.main(list(argv))
            except SystemExit as ex:
                return str(ex.code)

    def base(self, *extra):
        return ["run", "--stage", "build", "--measure", "runtime", "--tree-root", str(self.tmp), "--no-git", "--hipc-commit", "a" * 40, *extra]

    def test_order_of_gates(self):
        rc = self.run_main(*self.base())
        self.assertIn("booking-ref", rc)                                                     # freeze gate passed, then the booking is required
        (self.fz / M.RECORD_NAME).unlink()
        self.assertIn("freeze_record_h100.json is missing", self.run_main(*self.base("--booking-ref", "booking log booking text")))

    def test_energy_stage_needs_the_validation_before_anything_else(self):
        rc = self.run_main("run", "--stage", "all", "--measure", "both", "--tree-root", str(self.tmp), "--no-git", "--hipc-commit", "a" * 40, "--booking-ref", "booking log booking text")
        self.assertIn("power-sensor update test", rc)

    def test_missing_directories_are_refused(self):
        rc = self.run_main(*self.base("--booking-ref", "booking log booking text"))
        self.assertIn("--samples-root", rc)


class SamplesManifestScheme(unittest.TestCase):
    """verify_samples: the Cluster copy's MANIFEST.sha256 must equal the committed one byte for byte and every listed file must hash as listed."""
    TEXT = "int main() { return 0; }\n"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.tmp, True)
        self.manifest_text = hashlib.sha256(self.TEXT.encode()).hexdigest() + "  ./cpp/a.cu\n"
        self.tree = self.tmp / "tree"; mf = self.tree / M.SAMPLES_MANIFEST_REL; mf.parent.mkdir(parents=True); mf.write_bytes(self.manifest_text.encode())
        self.root = self.tmp / "copy"; (self.root / "cpp").mkdir(parents=True)
        (self.root / "cpp" / "a.cu").write_bytes(self.TEXT.encode()); (self.root / "MANIFEST.sha256").write_bytes(self.manifest_text.encode()); (self.root / ".source_rev").write_text(M.SOURCES_REVISION + "\n")

    def refuses(self, fragment):
        with self.assertRaises(SystemExit) as cm:
            M.verify_samples(self.root, self.tree)
        self.assertIn(fragment, str(cm.exception))

    def test_matching_copy_is_accepted(self):
        r = M.verify_samples(self.root, self.tree)
        self.assertEqual((r["files_verified"], r["source_rev"]), (1, M.SOURCES_REVISION))

    def test_manifest_not_byte_identical(self):
        (self.root / "MANIFEST.sha256").write_bytes((self.manifest_text + "\n").encode()); self.refuses("not byte-identical")

    def test_changed_or_missing_file(self):
        (self.root / "cpp" / "a.cu").write_bytes(b"changed\n"); self.refuses("1 of 1 files")
        (self.root / "cpp" / "a.cu").unlink(); self.refuses("1 of 1 files")

    def test_missing_manifest_in_the_copy_or_the_tree(self):
        (self.root / "MANIFEST.sha256").unlink(); self.refuses("must carry its MANIFEST.sha256")
        (self.root / "MANIFEST.sha256").write_bytes(self.manifest_text.encode()); (self.tree / M.SAMPLES_MANIFEST_REL).unlink(); self.refuses("missing from the tree")

    def test_wrong_pinned_revision(self):
        (self.root / ".source_rev").write_text("0" * 40 + "\n"); self.refuses("not the pinned revision")

    def test_the_real_committed_manifest_is_the_parents(self):
        mf = REPO / M.SAMPLES_MANIFEST_REL
        if not mf.is_file():
            self.skipTest("the committed manifest is not in this checkout")
        self.assertEqual(hashlib.sha256(mf.read_bytes()).hexdigest(), "23ad042ec36f50dd271b05037cbcd2a1fda3e78c1f5e9edbb019f29d63fb7a10")
        self.assertEqual(sum(1 for l in mf.read_text().splitlines() if l.strip()), 1443)

    def test_scratch_paths_are_refused(self):
        with self.assertRaises(SystemExit) as cm: M.refuse_scratch("/scratch/shareduser/x", "--workdir")
        self.assertIn("purges after 15 days", str(cm.exception))
        M.refuse_scratch("/home/user/x", "--workdir")


if __name__ == "__main__":
    unittest.main()
