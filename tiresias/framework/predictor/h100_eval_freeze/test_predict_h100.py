"""CPU-only tests of predict_h100.py: refuses without an H100 calibration document, no fallback to another board, plumbing-test mode, determinism.

    python -m pytest tiresias/framework/predictor/h100_eval_freeze/test_predict_h100.py

The plumbing tests use a BLACKWELL calibration document from calibrate/runs/ (a stand-in that only proves the plumbing); their outputs are written to a temporary directory and are not predictions.
"""
import contextlib
import copy
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = HERE.parent
sys.path.insert(0, str(HERE))
import predict_h100 as PH  # noqa: E402
import cell_sets as CS  # noqa: E402

BLACKWELL_DOC = SR / "calibrate" / "runs" / "energy_20261002" / "run_full" / "calibration_sm_120_0e63baea.json"
TABLES = [HERE / n for n in CS.static_files("main")]
CELL = "matrix_multiply/small/c1"
need = unittest.skipUnless(BLACKWELL_DOC.exists() and all(p.exists() for p in TABLES), "needs the committed Blackwell document and the built H100 static tables")


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = PH.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


class Refusals(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_no_document_is_refused_with_no_fallback(self):
        rc, out, err = run("--out", str(self.tmp / "p.json"))
        self.assertEqual(rc, 2); self.assertIn("no calibration document", err); self.assertIn("no fallback to Blackwell", err)
        self.assertFalse((self.tmp / "p.json").exists())

    def test_missing_document(self):
        rc, _, err = run("--calibration", str(self.tmp / "none.json"), "--out", str(self.tmp / "p.json"))
        self.assertEqual(rc, 2); self.assertIn("does not exist", err)

    @need
    def test_a_blackwell_document_is_not_accepted_as_h100(self):
        rc, _, err = run("--calibration", str(BLACKWELL_DOC), "--calibration-source", "unit test", "--out", str(self.tmp / "p.json"), "--allow-incomplete")
        self.assertEqual(rc, 2); self.assertIn("not an H100", err)
        self.assertFalse((self.tmp / "p.json").exists())

    @need
    def test_plumbing_mode_rules(self):
        base = ["--calibration", str(BLACKWELL_DOC), "--allow-incomplete", "--plumbing-test"]
        rc, _, err = run(*base, "--out", str(self.tmp / PH.FROZEN_NAME), "--only", CELL)
        self.assertEqual(rc, 2); self.assertIn("must not be called", err)
        rc, _, err = run(*base, "--out", str(self.tmp / "p.json"))
        self.assertEqual(rc, 2); self.assertIn("needs --only", err)
        rc, _, err = run(*base, "--out", str(self.tmp / "p.json"), "--only", "h100/")
        self.assertEqual(rc, 2); self.assertIn("must select 1 to 4", err)
        rc, _, err = run("--calibration", str(BLACKWELL_DOC), "--allow-incomplete", "--out", str(self.tmp / "p.json"), "--only", CELL)     # --only without --plumbing-test
        self.assertEqual(rc, 2)

    @need
    def test_incomplete_document_needs_the_explicit_flag(self):
        doc = json.loads(BLACKWELL_DOC.read_text())
        self.assertFalse(doc["complete"])                       # the committed Blackwell energy run is incomplete (windows near the cap were not admitted)
        rc, _, err = run("--calibration", str(BLACKWELL_DOC), "--plumbing-test", "--only", CELL, "--out", str(self.tmp / "p.json"))
        self.assertEqual(rc, 2); self.assertIn("incomplete", err)


@need
class Plumbing(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.tmp, True)

    def plumbing(self, name):
        out = self.tmp / name
        rc, stdout, err = run("--calibration", str(BLACKWELL_DOC), "--allow-incomplete", "--plumbing-test", "--only", CELL, "--out", str(out))
        self.assertEqual(rc, 0, err)
        return json.loads(out.read_text())

    def test_marked_output_with_sensible_numbers(self):
        r = self.plumbing("a.json")
        self.assertEqual(r["kind"], "plumbing_test_not_h100"); self.assertIn("PLUMBING TEST ONLY", r["warning"])
        self.assertEqual(list(r["cells"]), ["h100/unseen_cuda_samples_matrix_multiply/small/c1"])
        c = r["cells"]["h100/unseen_cuda_samples_matrix_multiply/small/c1"]
        self.assertEqual(c["status"], "predicted")
        self.assertGreater(c["runtime_s"], 0); self.assertGreater(c["energy_j"], 0); self.assertLessEqual(c["mean_power_w"], r["calibration"]["energy_cap_w"] + 1e-9)
        self.assertFalse(r["calibration"]["complete"]); self.assertTrue(r["calibration"]["allow_incomplete_used"])
        self.assertIn("roofline_runtime_s", r)
        for name in CS.static_files("main"):
            self.assertEqual(r["inputs_sha256"][name], PH.sha(HERE / name))

    def test_deterministic_and_never_overwrites(self):
        a, b = self.plumbing("a.json"), self.plumbing("b.json")
        self.assertEqual(a, b)
        rc, _, err = run("--calibration", str(BLACKWELL_DOC), "--allow-incomplete", "--plumbing-test", "--only", CELL, "--out", str(self.tmp / "a.json"))
        self.assertEqual(rc, 2); self.assertIn("never overwritten", err)

    def test_energy_follows_the_documents_formula(self):
        r = self.plumbing("a.json")
        doc = json.loads(BLACKWELL_DOC.read_text()); prof = doc["constants"]["energy"]["profile"]; cap = doc["constants"]["energy"]["cap_w"]
        c = r["cells"]["h100/unseen_cuda_samples_matrix_multiply/small/c1"]
        want = min(cap * c["runtime_s"], prof["base_power_w"] * c["runtime_s"] + sum(c["term_j"].values()))
        self.assertAlmostEqual(c["energy_j"], want, places=12)
        self.assertAlmostEqual(c["base_term_j"], prof["base_power_w"] * c["runtime_s"], places=12)

    def test_the_blackwell_l2_size_appears_nowhere_in_the_h100_prediction_path(self):
        for f in ("predict_h100.py", "footprints_h100.py", "cell_sets.py"):
            text = (HERE / f).read_text()
            for bad in ("188", "128 MiB", "2**27", "<< 27"):
                self.assertNotIn(bad, text, f)
        text = (HERE / "predict_h100.py").read_text(); self.assertEqual(text.count("134217728"), 1)           # only the refusal condition names it

    def test_a_document_that_matches_the_h100_rows_is_accepted_as_h100(self):
        """Acceptance path of the H100 check, with Blackwell's constants dressed up as an H100 document. Output goes to a temporary file; it is a test of the checker, not a prediction."""
        import make_cells_h100 as MC
        hw, _ = MC.read_h100_hardware()
        doc = json.loads(BLACKWELL_DOC.read_text())
        doc["device"].update(name="NVIDIA H100 80GB HBM3", sm_count=hw["sm_count"], l2_bytes=hw["l2_bytes"], compute_capability="9.0")
        p = self.tmp / "fake_h100_doc.json"; p.write_text(json.dumps(doc))
        self.assertEqual(PH.check_h100_document(doc, hw), [])
        bad = copy.deepcopy(doc); bad["device"]["sm_count"] = 188
        self.assertIn("SM count", " ".join(PH.check_h100_document(bad, hw)))
        # full run over every cell, only to prove coverage and the unsupported bookkeeping
        out = self.tmp / "all.json"
        rc, stdout, err = run("--calibration", str(p), "--allow-incomplete", "--out", str(out))
        self.assertEqual(rc, 2); self.assertIn("--calibration-source is required", err)                      # which run it is must be stated
        rc, stdout, err = run("--calibration", str(p), "--allow-incomplete", "--calibration-source", "unit test, not a real run", "--out", str(out))
        self.assertEqual(rc, 0, err)
        r = json.loads(out.read_text())
        self.assertEqual(r["kind"], "h100_frozen_predictions"); self.assertIsNone(r["warning"]); self.assertEqual(r["calibration"]["source"], "unit test, not a real run")
        self.assertEqual(r["calibration"]["energy_windows_not_admitted"], ["int16", "fpadd64", "int64", "rd_l2_8", "rd_l2_16", "fma64_repeat"])
        self.assertEqual(len(r["cells"]), 112)
        support = CS.load_static("main")[4]
        self.assertEqual(sorted(c for c, v in r["cells"].items() if v["status"] == "unsupported"), sorted(c for c, v in support.items() if not v["supported"]))
        self.assertEqual(r["schema"], "h100_predictions/2"); self.assertIn("shared-traffic", r["runtime_model"]); self.assertIn("traffic-based byte columns", r["energy_model"])
        self.assertEqual(r["l2_capacity_bytes_used_by_the_capacity_rule"], hw["l2_bytes"]); self.assertEqual(hw["l2_bytes"], 52428800)
        refused = [c for c, v in r["cells"].items() if v["status"] == "not_predicted"]
        for cid in refused:                                                                                    # refusals stay in the list with their reason and count as failures
            self.assertTrue(r["cells"][cid]["counted_as_failure"]); self.assertIn("L2 capacity", r["cells"][cid]["reason"])
            self.assertIn("%.1f MiB" % (hw["l2_bytes"] / 2**20), r["cells"][cid]["reason"])                      # the H100 capacity (50.0 MiB), never the Blackwell 128.0 MiB
        for cid, v in r["cells"].items():
            if v["status"] == "predicted": self.assertGreater(v["traffic_bytes"]["l2_served"], 0)
        self.assertEqual(r["coverage"]["unsupported"], sorted(c for c, v in r["cells"].items() if v["status"] != "predicted"))


if __name__ == "__main__":
    unittest.main()
