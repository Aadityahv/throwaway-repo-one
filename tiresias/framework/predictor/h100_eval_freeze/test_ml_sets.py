"""CPU-only tests of the machine-learning, tensor-core and attention sets of the H100 (and, prepared, A100) evaluation: cell generator, profiles, SASS manifests, port extension, window gate,
static tables, prediction and baseline plumbing.

    python -m pytest tiresias/framework/predictor/h100_eval_freeze/test_ml_sets.py
"""
import contextlib
import csv
import io
import json
import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = HERE.parent
REPO = SR.parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "ml_sets"))
import cell_sets as CS  # noqa: E402
import make_cells_ml as MM  # noqa: E402
import make_sass_manifest_ml as MSM  # noqa: E402
import window_gate as WG  # noqa: E402

ML = HERE / "ml_sets"


def hgt_section(arch):
    text = (REPO / "HARDWARE_GROUND_TRUTH.md").read_text(encoding="utf-8")
    sys.path.insert(0, str(SR))
    import extract_features as X
    return X.load_hardware(text, MM.SECTION[arch])


class CellGenerator(unittest.TestCase):
    def test_committed_files_reproduce_byte_for_byte(self):
        for arch in ("h100", "a100"):
            self.assertEqual(MM.main(["--arch", arch, "--check"]), 0, arch)

    def test_l2_is_read_from_the_ground_truth_and_every_size_derives_from_it(self):
        for arch, expect_l2 in (("h100", 52428800), ("a100", 41943040)):
            hw = hgt_section(arch)
            self.assertEqual(hw["l2_bytes"], expect_l2)
            docs = MM.build(arch)
            self.assertEqual(docs["ml"]["l2_bytes"], hw["l2_bytes"])
            for key, n in (("ml", 40), ("tensor", 16)):
                cells = docs[key]["cells"]
                self.assertEqual(len(cells), n)
                for c in cells:
                    r = c["footprint_bytes"] / hw["l2_bytes"]
                    self.assertAlmostEqual(r, c["footprint_over_l2"], places=9)
                    self.assertFalse(0.4 <= r < 1.5, c["cell_id"])
                    self.assertEqual(c["tier"], "L2" if r < 1 else "DRAM")
                    self.assertTrue(c["cell_id"].startswith(arch + "/"))

    def test_regimes_are_the_blackwell_ratios_and_two_tiers_of_each(self):
        t = MM.blackwell_targets()
        sys.path.insert(0, str(SR / "fresh_f"))
        docs = MM.build("h100")
        for c in docs["ml"]["cells"] + docs["tensor"]["cells"]:
            tgt = t[(c["family"], c["regime"])]
            # the scaled dimension is a multiple of its tile unit (128 for the matrix multiplies, a few elements or rows otherwise): the ratio stays close to the Blackwell regime ratio
            self.assertLess(abs(math.log(c["footprint_over_l2"] / tgt)), 0.5 if c["family"] in ("sgemm", "tcgemm") else 0.25, c["cell_id"])
            self.assertEqual(c["tier"], "L2" if c["regime"] in ("small", "medium") else "DRAM")
        for fam in ("gelu", "swiglu", "rmsnorm", "rope", "sgemm", "tcgemm", "attn"):
            cs = [c for c in docs["ml"]["cells"] + docs["tensor"]["cells"] if c["family"] == fam]
            self.assertEqual(sum(c["tier"] == "L2" for c in cs), 4); self.assertEqual(sum(c["tier"] == "DRAM" for c in cs), 4)
            self.assertEqual(sorted({c["candidate_id"] for c in cs}), ["c1", "c2"])

    def test_blackwell_regime_targets_equal_the_committed_blackwell_cells(self):
        for s in "fgh":
            doc = json.loads((SR / ("fresh_%s" % s) / ("fresh_cells_%s.json" % s)).read_text())
            t = MM.blackwell_targets()
            for c in doc["cells"]:
                self.assertAlmostEqual(t[(c["family"], c["regime"])], c["footprint_bytes"] / 134217728, places=9, msg=c["cell_id"])

    def test_no_blackwell_shape_is_copied(self):
        blackwell = []
        for s in "fgh":
            blackwell += [json.dumps(c["controls"], sort_keys=True) for c in json.loads((SR / ("fresh_%s" % s) / ("fresh_cells_%s.json" % s)).read_text())["cells"]]
        docs = MM.build("h100")
        for c in docs["ml"]["cells"] + docs["tensor"]["cells"]:
            self.assertNotIn(json.dumps(c["controls"], sort_keys=True), blackwell, c["cell_id"])

    def test_footprints_recomputed_independently_from_the_launch_arguments(self):
        for c in MM.build("h100")["ml"]["cells"] + MM.build("h100")["tensor"]["cells"]:
            ctl, k = c["controls"], c["kernels"][0]
            f = c["family"]
            if f == "gelu": fp = 2 * ctl["n"] * 4
            elif f == "swiglu": fp = 3 * ctl["n"] * 4
            elif f == "rmsnorm": fp = (2 * ctl["rows"] * ctl["cols"] + ctl["cols"]) * 4
            elif f == "rope": fp = 2 * ctl["batch"] * ctl["seq"] * ctl["heads"] * 128 * 4 + 2 * ctl["seq"] * 64 * 4
            elif f == "sgemm": fp = (ctl["M"] * ctl["K"] + ctl["K"] * ctl["N"] + ctl["M"] * ctl["N"]) * 4
            elif f == "tcgemm": fp = (ctl["M"] * ctl["K"] + ctl["N"] * ctl["K"]) * 2 + ctl["M"] * ctl["N"] * 4
            else: fp = 3 * ctl["BH"] * ctl["S"] * 64 * 2 + ctl["BH"] * ctl["S"] * 64 * 4
            self.assertEqual(fp, c["footprint_bytes"], c["cell_id"])
            # the launch tiles the problem exactly
            blocks = k["grid"][0] * k["grid"][1] * k["grid"][2]
            if f in ("gelu", "swiglu"): self.assertEqual(blocks * k["block"][0] * (1 if k["kid"].endswith("_s") else 4), ctl["n"])
            if f == "rmsnorm": self.assertEqual(blocks, ctl["rows"]); self.assertEqual(ctl["cols"] % 512, 0)
            if f == "rope": self.assertEqual(k["grid"][0], ctl["seq"])
            if f in ("sgemm", "tcgemm"):
                t = {"sg64": 64, "sg128": 128, "tc128": 128, "tc64": 64}[k["kid"]]; self.assertEqual(blocks * t * t, ctl["M"] * ctl["N"]); self.assertEqual(ctl["K"] % 32, 0)
            if f == "attn": self.assertEqual(blocks * (k["block"][0] // 32) * 16, ctl["S"] * ctl["BH"])

    def test_candidates_do_equal_work(self):
        by = {}
        for c in MM.build("h100")["ml"]["cells"] + MM.build("h100")["tensor"]["cells"]:
            by.setdefault((c["family"], c["regime"]), []).append(c)
        for k, cs in by.items():
            self.assertEqual(len(cs), 2); self.assertEqual(cs[0]["footprint_bytes"], cs[1]["footprint_bytes"], k); self.assertEqual(cs[0]["logical_bytes_per_launch"], cs[1]["logical_bytes_per_launch"], k)

    def test_a_missing_hardware_row_stops_the_generator(self):
        text = (REPO / "HARDWARE_GROUND_TRUTH.md").read_text(encoding="utf-8").replace("| L2 cache size | 52,428,800 bytes = 50.0 MiB |", "| L2 cache size | pending |")
        old = MM.GROUND_TRUTH
        tmp = Path(tempfile.mkdtemp()); (tmp / "g.md").write_text(text, encoding="utf-8")
        MM.GROUND_TRUTH = tmp / "g.md"
        try:
            with self.assertRaises(SystemExit):
                MM.read_hardware("h100")
        finally:
            MM.GROUND_TRUTH = old


class Profiles(unittest.TestCase):
    def test_main_is_the_72_plus_40_and_tensor_the_16(self):
        main, _ = CS.load_cells("main"); tensor, _ = CS.load_cells("tensor")
        self.assertEqual((len(main), len(tensor)), (112, 16))
        self.assertEqual(len({c["cell_id"] for c in main + tensor}), 128)
        self.assertEqual(sorted({c["set"] for c in main}), ["fresh_f", "set_d", "set_e", "unseen"]); self.assertEqual(sorted({c["set"] for c in tensor}), ["fresh_g", "fresh_h"])

    def test_names_and_inputs_do_not_collide(self):
        a, b = CS.profile("main"), CS.profile("tensor")
        for k in ("predictions", "baselines", "record"):
            self.assertNotEqual(a[k], b[k])
        self.assertFalse(set(CS.frozen_inputs("main")) & set(CS.frozen_inputs("tensor")))

    def test_every_set_family_is_covered_by_a_cell_family(self):
        fams = {c["family"] for p in ("main", "tensor") for c in CS.load_cells(p)[0]}
        self.assertEqual(fams, {f for v in CS.SET_FAMILY.values() for f in v})


class SassManifests(unittest.TestCase):
    def test_committed_manifests_reproduce(self):
        for arch in ("h100", "a100"):
            self.assertEqual(MSM.main(["--arch", arch, "--check"]), 0)

    def test_every_ml_kernel_has_one_symbol_and_a_hash_of_the_committed_dump(self):
        for arch in ("h100", "a100"):
            docs = MSM.build(arch)
            self.assertEqual((len(docs["ml"]["kernels"]), len(docs["tensor"]["kernels"])), (10, 4))
            for d in docs.values():
                for kid, row in d["kernels"].items():
                    self.assertRegex(row["instruction_sequence_sha256"], r"^[0-9a-f]{64}$"); self.assertGreater(row["instructions"], 10)
                    self.assertIn("_Z", row["symbol"])
        main = CS.merged_sass_manifest("main"); tensor = CS.merged_sass_manifest("tensor")
        kids = {k["kid"] for p in ("main", "tensor") for c in CS.load_cells(p)[0] for k in c["kernels"]}
        self.assertEqual(kids - set(main) - set(tensor), set())


class WindowGate(unittest.TestCase):
    def make(self, tmp, name, power_mw, clocks, gap_ns=100_000_000, n=40):
        tr = Path(tmp) / "raw_attempts" / name / "x-s1-a1"; tr.mkdir(parents=True)
        with (tr / "samples.csv").open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["monotonic_ns", "board_power_mw", "temperature_c", "graphics_clock_mhz", "memory_clock_mhz", "utilization_percent"])
            for i in range(n):
                w.writerow([1_000_000_000 + i * gap_ns, power_mw, 60, clocks[i % len(clocks)], 2619, 100])
        with (tr / "windows.csv").open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["block", "launches", "host_begin_monotonic_ns", "host_end_monotonic_ns", "cuda_seconds"]); w.writerow([1, 100, 1_000_000_000, 1_000_000_000 + (n - 1) * gap_ns, 3.9])
        return tr

    def raw(self, tmp, rows):
        p = Path(tmp) / "application_energy_raw.csv"
        fields = ["run_id", "parent_id", "regime", "candidate_id", "board_energy_j_total", "counted_launch_interval_s", "trace_dir"]
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    def test_bands_clock_and_cap_rule(self):
        tmp = tempfile.mkdtemp(); self.addCleanup(__import__("shutil").rmtree, tmp, True)
        specs = [("below", 400_000, [1980]), ("near", 670_000, [1980]), ("above", 695_000, [1980]), ("throttled", 500_000, [1980, 1200, 1200])]
        rows = []
        for i, (name, mw, clk) in enumerate(specs):
            tr = self.make(tmp, name, mw, clk)
            rows.append(dict(run_id=name, parent_id="fresh_g_ml_tensor_matmul", regime="small", candidate_id="c%d" % i, board_energy_j_total=str(mw / 1000.0 * 3.9), counted_launch_interval_s="3.9", trace_dir=str(tr)))
        self.raw(tmp, rows)
        doc = WG.gate([tmp], 700.0)
        band = {r["run_id"]: r for r in doc["rows"]}
        self.assertEqual(band["below"]["band"], "below_cap"); self.assertEqual(band["near"]["band"], "near_cap"); self.assertEqual(band["above"]["band"], "above_cap_rule")
        self.assertTrue(band["above"]["excluded_from_energy_scoring"]); self.assertFalse(band["near"]["excluded_from_energy_scoring"])
        self.assertFalse(band["below"]["clock_throttled"]); self.assertTrue(band["throttled"]["clock_throttled"])
        self.assertEqual(doc["above_cap_rule"], ["h100/fresh_g_ml_tensor_matmul/small/c2"])
        self.assertEqual(doc["rules"]["above_cap_rule_fraction"], 0.985); self.assertEqual(doc["rules"]["below_cap_fraction"], 0.95)

    def test_missing_trace_is_listed_never_admitted_silently(self):
        tmp = tempfile.mkdtemp(); self.addCleanup(__import__("shutil").rmtree, tmp, True)
        self.raw(tmp, [dict(run_id="r", parent_id="p", regime="small", candidate_id="c1", board_energy_j_total="500", counted_launch_interval_s="1", trace_dir="/nonexistent")])
        doc = WG.gate([tmp], 700.0)
        self.assertEqual(doc["no_trace"], ["h100/p/small/c1"]); self.assertEqual(doc["rows"][0]["band"], "near_cap" if False else "below_cap")

    def test_rule_constants_match_the_calibrator_and_the_scorer(self):
        import score_h100 as S
        sys.path.insert(0, str(SR / "calibrate"))
        from cal import energy as En
        self.assertEqual(WG.ADMIT_FRACTION_OF_CAP, En.ADMIT_FRACTION_OF_CAP); self.assertEqual(WG.ADMIT_FRACTION_OF_CAP, S.ADMIT_FRACTION_OF_CAP)
        self.assertEqual(WG.BELOW_CAP_FRACTION, S.BELOW_CAP_FRACTION)

    def test_the_sensor_result_was_taken_far_below_the_cap(self):
        """The reason this gate exists: the committed power-sensor result covers a 282 W plateau, about 40% of the 700 W limit."""
        p = SR / "h100_power_sensor_test" / "runs" / "h100_j16" / "power_sensor_results.json"
        if not p.is_file():
            self.skipTest("power-sensor result not present")
        d = json.loads(p.read_text())
        for r in d["runs"]:
            self.assertLess(r["max_plateau_fraction_of_power_limit"], 0.5)


class PortExtension(unittest.TestCase):
    def run_py(self, code):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(HERE), env=dict(__import__("os").environ, PYTHONPATH=str(SR / "port_common")))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout

    def test_blackwell_kernel_registry_and_interpreter_are_unchanged_by_importing_the_extension(self):
        out = self.run_py("import port_ext as PX; before=(dict(PX.UP.KERNELS), PX.UP.INTERP, PX.A.PARAM_BASE);\nimport port_ml_ext as PM\n"
                          "assert (dict(PX.UP.KERNELS), PX.UP.INTERP, PX.A.PARAM_BASE)==before, 'import changed global state'\nprint('unchanged')")
        self.assertIn("unchanged", out)

    def test_install_registers_14_kernels_and_the_assets_resolve_on_both_architectures(self):
        for arch, folder in (("sm_90", "port_h100"), ("sm_80", "port_a100")):
            out = self.run_py("import port_ml_ext as PM\nfrom pathlib import Path\nPM.install(%r, Path(%r))\nUP=PM.UP\nn=0\n"
                              "for kid in PM.KERNELS:\n    a=UP.kernel_assets(kid); assert a['abi_ok'], (kid, a['abi_problems']); assert a['cubin_sha'] and a.get('cubin_sha_is_sass_dump_hash'); n+=1\nprint(n)"
                              % (arch, str(SR / folder / "compiled_cluster_cuda12.1_ml")))
            self.assertEqual(out.strip(), "14")

    def test_kernel_table_matches_the_sass_manifest_generator(self):
        import port_ml_ext as PM
        self.assertEqual({k: v[0] for k, v in PM.KERNELS.items()}, {k: v[0] for k, v in MSM.KERNELS.items()})

    def test_tensor_family_is_added_to_the_instruction_families(self):
        out = self.run_py("import port_ml_ext as PM\nfrom pathlib import Path\nPM.install('sm_90', Path(%r))\nprint(PM.UP.X.op_family('HMMA.16816.F32.BF16')[0], PM.UP.X.op_family('FFMA')[0])"
                          % str(SR / "port_h100" / "compiled_cluster_cuda12.1_ml"))
        self.assertTrue(out.startswith("tensor_core"))

    def test_the_only_tensor_instruction_form_in_the_h100_and_a100_builds_is_the_one_the_tensor_stage_measures(self):
        """The tensor stage measures the bf16 m16n8k16 MMA (HMMA.16816.F32.BF16). Every tensor instruction of every tensor/attention kernel in the CUDA 12.1 builds must be that form."""
        for arch, folder in (("sm_90", "port_h100"), ("sm_80", "port_a100")):
            text = (SR / folder / "compiled_cluster_cuda12.1_ml" / "driver_ml.sass").read_text(encoding="utf-8", errors="replace")
            funcs = MSM.functions(text)
            for kid in ("tc128", "tc64", "at4", "at8"):
                sym = next(s for s in funcs if MSM.KERNELS[kid][0] in s)
                ops = re.findall(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?((?:HMMA|IMMA|QMMA|DMMA|BMMA|WGMMA)[A-Z0-9_.]*)", funcs[sym])
                self.assertTrue(ops, (arch, kid)); self.assertEqual(set(ops), {"HMMA.16816.F32.BF16"}, (arch, kid))
            for kid in ("gelu_s", "sg64", "rms_s", "rope_all"):
                sym = next(s for s in funcs if MSM.KERNELS[kid][0] in s)
                self.assertFalse(re.search(r"\bHMMA|\bIMMA|\bWGMMA", funcs[sym]), (arch, kid))


STATIC = [ML / ("%s_%s_h100.json" % (k, s)) for s in ("ml", "tensor") for k in ("features", "static_support")]


class StaticTables(unittest.TestCase):
    def table(self, which):
        sup = json.loads((ML / ("static_support_%s_h100.json" % which)).read_text())
        feats = json.loads((ML / ("features_%s_h100.json" % which)).read_text())
        cells = json.loads((ML / ("cells_%s_h100.json" % which)).read_text())["cells"]
        return sup, feats, cells

    @unittest.skipUnless(all(p.exists() for p in STATIC[:2]), "static tables of the machine-learning cells not built")
    def test_every_ml_cell_is_listed_with_support_and_a_reason_when_unsupported(self):
        sup, feats, cells = self.table("ml")
        self.assertEqual(set(sup["rows"]), {c["cell_id"] for c in cells}); self.assertEqual(sup["cells"], 40)
        self.assertEqual({r["cell_id"] for r in feats["rows"]}, {c["cell_id"] for c in cells})
        for cid, v in sup["rows"].items():
            if not v["supported"]:
                self.assertTrue(v["reason"], cid)
        self.assertEqual(sorted(c for c, v in sup["rows"].items() if not v["supported"]), sup["unsupported"])
        for r in feats["rows"]:
            fam = (r.get("per_launch_totals") or r.get("work") or {})
            self.assertNotIn("tensor_core", json.dumps(fam.get("issue_by_family", fam)) if r["status"] == "supported" else "")

    @unittest.skipUnless(all(p.exists() for p in STATIC[2:]), "static tables of the tensor cells not built")
    def test_tensor_cells_carry_tensor_core_instructions_in_their_counts(self):
        sup, feats, cells = self.table("tensor")
        self.assertEqual(set(sup["rows"]), {c["cell_id"] for c in cells})
        for r in feats["rows"]:
            if sup["rows"][r["cell_id"]]["supported"]:
                self.assertIn("tensor_core", json.dumps(r["work"]), r["cell_id"])


class PredictAndBaselinePlumbing(unittest.TestCase):
    """Plumbing only, on the committed BLACKWELL documents; nothing written here is an H100 prediction or baseline."""
    BW = SR / "calibrate" / "runs" / "energy_20261002" / "run_full" / "calibration_sm_120_0e63baea.json"
    BW_TENSOR = SR / "fresh_g" / "calibration_with_tensor.json"
    need_main = unittest.skipUnless(all((HERE / n).exists() for n in CS.static_files("main")) and BW.exists(), "needs the built main-profile tables and the Blackwell document")
    need_tensor = unittest.skipUnless(all((HERE / n).exists() for n in CS.static_files("tensor")) and BW_TENSOR.exists(), "needs the built tensor-profile tables and the Blackwell tensor document")

    def predict(self, *argv):
        import predict_h100 as PH
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = PH.main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def test_frozen_names_are_refused_in_plumbing_mode_for_both_profiles(self):
        import predict_h100 as PH
        for prof in ("main", "tensor"):
            rc, _, err = self.predict("--calibration", str(self.BW), "--allow-incomplete", "--plumbing-test", "--only", "gelu", "--profile", prof, "--out", str(Path(tempfile.mkdtemp()) / CS.profile(prof)["predictions"]))
            self.assertEqual(rc, 2); self.assertIn("must not be called", err)

    @need_main
    def test_ml_cells_are_predicted_through_the_main_profile(self):
        tmp = Path(tempfile.mkdtemp()); self.addCleanup(__import__("shutil").rmtree, tmp, True)
        rc, _, err = self.predict("--calibration", str(self.BW), "--allow-incomplete", "--plumbing-test", "--only", "fresh_f_ml_gelu/small", "--out", str(tmp / "p.json"))
        self.assertEqual(rc, 0, err)
        r = json.loads((tmp / "p.json").read_text())
        self.assertEqual(r["kind"], "plumbing_test_not_h100"); self.assertEqual(r["profile"], "main")
        self.assertEqual(sorted(r["cells"]), ["h100/fresh_f_ml_gelu/small/c1", "h100/fresh_f_ml_gelu/small/c2"])
        for c in r["cells"].values():
            if c["status"] == "predicted":
                self.assertGreater(c["runtime_s"], 0); self.assertGreater(c["energy_j"], 0)
        self.assertEqual(sorted(r["inputs_sha256"]), sorted(CS.static_files("main")))

    @need_tensor
    def test_tensor_cells_need_tensor_constants_and_are_unsupported_without_them(self):
        tmp = Path(tempfile.mkdtemp()); self.addCleanup(__import__("shutil").rmtree, tmp, True)
        rc, _, err = self.predict("--calibration", str(self.BW_TENSOR), "--allow-incomplete", "--plumbing-test", "--only", "tensor_matmul/small", "--profile", "tensor", "--out", str(tmp / "t.json"))
        self.assertEqual(rc, 0, err)
        r = json.loads((tmp / "t.json").read_text())
        self.assertTrue(r["tensor_stage"]["present"]); self.assertIsNotNone(r["tensor_stage"]["energy_rate_pJ_per_lane_instruction"])
        # a document without the tensor stage: the cells are unsupported or not predicted, never priced at another class's rate
        rc, _, err = self.predict("--calibration", str(self.BW), "--allow-incomplete", "--plumbing-test", "--only", "tensor_matmul/small", "--profile", "tensor", "--out", str(tmp / "n.json"))
        self.assertEqual(rc, 0, err); self.assertIn("no usable tensor constants", err)
        r = json.loads((tmp / "n.json").read_text())
        self.assertFalse(r["tensor_stage"]["present"])
        for c in r["cells"].values():
            self.assertNotEqual(c["status"], "predicted"); self.assertTrue(c["counted_as_failure"])


if __name__ == "__main__":
    unittest.main()
