"""CPU-only tests of the Ada measurement orchestrator (measure_ada.py). No GPU, no ssh; the run stage is never executed here.

    python -m pytest tiresias/framework/predictor/replication_ada_rtx5000/test_measure_ada.py
"""
import contextlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = HERE.parent
sys.path.insert(0, str(HERE))
import measure_ada as M  # noqa: E402


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = M.main(list(argv))
        except SystemExit as ex:
            rc = ex.code if isinstance(ex.code, int) else 2
            if rc == 0:
                rc = 2
    return rc, out.getvalue(), err.getvalue()


def stub_smi(gpu_rows, apps_rows=(), rc=0):
    def run(cmd, **kw):
        if "query-compute-apps" in " ".join(cmd):
            out = "\n".join(apps_rows)
        else:
            out = "\n".join(gpu_rows)
        return subprocess.CompletedProcess(cmd, rc, out, "")
    return run


ADA_GPU = ["0, GPU-4640d904-c754-5243-df67-37761e91b400, NVIDIA RTX 5000 Ada Generation"]
FOREIGN = ["GPU-4640d904-c754-5243-df67-37761e91b400, 1919421, /snap/snapd-desktop-integration"]


class Plan(unittest.TestCase):
    def test_plan_counts_and_window(self):
        for prof, n in (("main", 124), ("tensor", 44)):
            p = M.plan(15.0, profile=prof)
            self.assertEqual(p["cells"], n)
            self.assertEqual(p["window_s"], 15.0)
            self.assertGreater(p["gpu_seconds"]["energy"], 0)

    def test_plan_cli(self):
        rc, out, _ = run("plan", "--profile", "tensor")
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out)["cells"], 44)

    def test_load_cells_carries_the_source_revision(self):
        doc, cells = M.load_cells(profile="main")
        self.assertEqual(doc, dict(source_revision=M.SOURCES_REVISION))
        self.assertEqual(len(cells), 124)


class FreezeGates(unittest.TestCase):
    def test_check_freeze_refuses_without_predictions(self):
        import tempfile, shutil
        tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, tmp, True)
        with self.assertRaises(M.Refusal) as cm:
            M.check_freeze(str(tmp), "0" * 40, use_git=False, profile="main")
        self.assertIn("predictions_ada.json", str(cm.exception))

    def test_check_freeze_needs_full_commit_sha(self):
        import tempfile, shutil
        tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "predictions_ada.json").write_text(json.dumps(dict(kind="ada_frozen_predictions")))
        with self.assertRaises(M.Refusal) as cm:
            M.check_freeze(str(tmp), "abc", use_git=False, profile="main")
        self.assertIn("HIPC_COMMIT", str(cm.exception))

    def test_verify_samples_refuses_without_committed_manifest(self):
        with self.assertRaises(M.Refusal) as cm:
            M.verify_samples(HERE, None)
        self.assertIn("MANIFEST", str(cm.exception))

    def test_check_window_refuses_without_validation(self):
        with self.assertRaises(M.Refusal) as cm:
            M.check_window("/nonexistent/power_sensor_results.json", 15, 0, tree_root=None, use_git=False)
        self.assertIn("15", str(cm.exception))

    def test_non_default_window_refused_without_validation(self):
        rc, _, err = run("run", "--stage", "build", "--measure", "both", "--window-s", "40",
                         "--samples-root", "x", "--workdir", "x", "--out-dir", "x", "--booking-ref", "test booking ref")
        self.assertEqual(rc, 2)
        self.assertIn("written decision", err)


class GpuGates(unittest.TestCase):
    def test_gpu_identity_single_ada_gpu(self):
        g = M.gpu_identity(smi_runner=stub_smi(ADA_GPU))
        self.assertEqual((g["index"], g["uuid"], g["visible_gpus"]), (0, "GPU-4640d904-c754-5243-df67-37761e91b400", 1))

    def test_gpu_identity_refuses_non_ada(self):
        with self.assertRaises(M.Refusal):
            M.gpu_identity(smi_runner=stub_smi(["0, GPU-aaaa, NVIDIA H100 80GB HBM3"]))

    def test_foreign_processes_listed_never_killed(self):
        found = M.foreign_compute_processes(smi_runner=stub_smi(ADA_GPU, FOREIGN), uuid="GPU-4640d904-c754-5243-df67-37761e91b400")
        self.assertEqual(found, FOREIGN)
        self.assertEqual(M.foreign_compute_processes(smi_runner=stub_smi(ADA_GPU), uuid="GPU-4640d904-c754-5243-df67-37761e91b400"), [])
        benign, hostile = M.split_foreign(FOREIGN + ["GPU-4640d904-c754-5243-df67-37761e91b400, 999, vllm"])
        self.assertEqual(benign, FOREIGN)
        self.assertEqual(hostile, ["GPU-4640d904-c754-5243-df67-37761e91b400, 999, vllm"])
        text = (HERE / "measure_ada.py").read_text()
        for forbidden in ("os.kill", "signal.", "SIGKILL", "SIGTERM", "pkill", "pynvml"):
            self.assertNotIn(forbidden, text)

    def test_engine_allow_lists_match_the_canonical_hpc_list(self):
        import re
        text = (HERE.parents[3] / "energy_harness" / "measurement_runner.py").read_text()
        m = re.search(r"ADA_BENIGN_COMPUTE_APP_SUBSTRINGS\s*=\s*\((.*?)\)", text, re.S)
        self.assertIsNotNone(m)
        canonical = tuple(sorted(re.findall(r'"([^"]+)"', m.group(1))))
        for rel in ("../fresh_f/run_f.py", "../prospective_test/run_prosp.py",
                    "../unseen_kernels/gpu/run_unseen.py", "../fresh_e/gpu/run_e.py",
                    "../fresh_d/timing_fresh_d.py"):
            text = (HERE / rel).read_text()
            m = re.search(r"_BENIGN_DISPLAY_PROCS\s*=\s*\((.*?)\)", text, re.S)
            self.assertIsNotNone(m, rel)
            self.assertEqual(tuple(sorted(re.findall(r'"([^"]+)"', m.group(1)))), canonical, rel)
            self.assertIn("HARNESS_ALLOW_PASSIVE_DISPLAY_CONTEXT", text, rel)

    def test_nvcc_release_refuses_wrong_toolchain(self):
        with self.assertRaises(M.Refusal):
            M.nvcc_release("/bin/false")

    def test_refuse_scratch(self):
        with self.assertRaises(M.Refusal):
            M.refuse_scratch("/scratch/foo", "test paths")
        M.refuse_scratch(str(HERE / "x"), "test paths")


class EngineWiring(unittest.TestCase):
    def test_sass_manifest_covers_every_profile_kid(self):
        import cell_sets_ada as CS
        man = M.merged_manifest()
        kids = {k["kid"] for prof in ("main", "tensor") for c in CS.load_cells(prof) for k in c["kernels"]}
        self.assertEqual(set(man) & kids, kids)

    def test_orchestrator_never_scores(self):
        # the orchestrator writes timing records and checks gates; scoring lives in score_ada.py, which this file must not touch
        text = (HERE / "measure_ada.py").read_text()
        for forbidden in ("score_ada", "application_energy_raw/"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
