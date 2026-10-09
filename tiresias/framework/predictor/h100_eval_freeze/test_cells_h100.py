"""CPU-only tests of the H100 cell list: reproducibility from HARDWARE_GROUND_TRUTH.md, tier of every cell, geometry equal to the Blackwell cell files' rules.

    python -m pytest tiresias/framework/predictor/h100_eval_freeze/test_cells_h100.py
"""
import json
import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SR / "unseen_kernels"))
sys.path.insert(0, str(SR / "fresh_e"))
import make_cells_h100 as MC  # noqa: E402

FLOAT = 4
doc = json.loads(MC.OUT.read_text(encoding="utf-8"))
cells = doc["cells"]
L2_REGIMES = ("small", "medium", "large", "l2")


def independent_footprint(c):
    """Footprint recomputed from the kernel launch arguments, not from the generator's own formula."""
    ks, a = c["kernels"], c["kernels"][0]["args"]
    f = c["family"]
    if f == "matmul":
        return 3 * a["wA"] * a["wB"] * FLOAT
    if f == "bs":
        return 5 * a["optN"] * FLOAT
    if f == "scan":
        n = ks[0]["grid"][0] * 1024
        return 2 * n * FLOAT + ks[0]["grid"][0] * FLOAT
    if f == "conv":
        return 3 * a["imageW"] * a["imageH"] * FLOAT
    if f == "sp":
        return (2 * a["vectorN"] * a["elementN"] + a["vectorN"]) * FLOAT
    if f == "fwt":
        return (1 << c["controls"]["log2N"]) * c["controls"]["batches"] * FLOAT
    if f == "tile":
        return 2 * a["width"] * a["height"] * FLOAT
    if f == "reduction":
        return 4 * a["n"] + 4 * ks[0]["grid"][0]
    raise AssertionError(f)


class Reproducible(unittest.TestCase):
    def test_regenerated_file_is_byte_identical(self):
        self.assertEqual(MC.render(MC.build_document()).encode("utf-8"), MC.OUT.read_bytes())

    def test_generation_is_deterministic(self):
        self.assertEqual(MC.render(MC.build_document()), MC.render(MC.build_document()))

    def test_hardware_values_come_from_the_ground_truth_file(self):
        hw, quoted = MC.read_h100_hardware()
        h = doc["hardware_from_ground_truth"]
        self.assertEqual(h["values"], {k: hw[k] for k in sorted(hw)})
        text = MC.GROUND_TRUTH.read_text(encoding="utf-8")
        for label, row in quoted.items():
            self.assertIn(row.split("|")[1].strip(), text)
        self.assertEqual((hw["sm_count"], hw["l2_bytes"]), (132, 52428800))      # the file's rows at the time of the freeze; a change of the file changes cells_h100.json (next test)

    def test_sizes_are_derived_not_typed(self):
        text = MC.GROUND_TRUTH.read_text(encoding="utf-8")
        self.assertIn("52,428,800", text)
        other = text.replace("52,428,800 bytes = 50.0 MiB", "41,943,040 bytes = 40.0 MiB", 1)       # pretend the L2 were 40 MiB
        self.assertNotEqual(other, text)
        hw2, q2 = MC.read_h100_hardware(other)
        self.assertEqual(hw2["l2_bytes"], 41943040)
        doc2 = MC.build_document(hw2, q2)
        by1 = {c["cell_id"]: c for c in cells}
        changed = [c["cell_id"] for c in doc2["cells"] if by1[c["cell_id"]]["footprint_bytes"] != c["footprint_bytes"]]
        self.assertGreater(len(changed), 40)
        for c in doc2["cells"]:
            self.assertTrue(MC.allowed(c["footprint_bytes"] / 41943040, c["regime"]), c["cell_id"])

    def test_refuses_an_incomplete_ground_truth_section(self):
        text = MC.GROUND_TRUTH.read_text(encoding="utf-8")
        broken = "\n".join(l for l in text.splitlines() if not l.startswith("| L2 cache size") or "52,428,800" not in l)
        with self.assertRaises(SystemExit):
            MC.read_h100_hardware(broken)


class Tiers(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(cells), 72)
        by_set = {}
        for c in cells:
            by_set[c["set"]] = by_set.get(c["set"], 0) + 1
        self.assertEqual(by_set, {"unseen": 32, "set_e": 16, "set_d": 24})
        self.assertEqual(len({c["cell_id"] for c in cells}), 72)

    def test_every_cell_is_in_its_intended_tier_with_no_marginal_cell(self):
        l2 = doc["hardware_from_ground_truth"]["values"]["l2_bytes"]
        for c in cells:
            fp = independent_footprint(c)
            self.assertEqual(fp, c["footprint_bytes"], c["cell_id"])
            ratio = fp / l2
            self.assertFalse(0.4 <= ratio < 1.5, c["cell_id"])
            self.assertEqual(c["tier"], "L2" if ratio < 1 else "DRAM", c["cell_id"])
            if c["regime"] in L2_REGIMES:
                self.assertLess(ratio, 0.4, c["cell_id"]); self.assertEqual(c["tier"], "L2")
            else:
                self.assertGreaterEqual(ratio, 1.5, c["cell_id"]); self.assertEqual(c["tier"], "DRAM")
            self.assertAlmostEqual(c["l2_ratio"], ratio)

    def test_tiers_are_well_separated_and_nearest_to_the_targets(self):
        """Each regime's footprint is within a factor of 3 of target x L2 (the coarsest family, the power-of-two Walsh transform, is the widest) and regimes keep their order."""
        l2 = doc["hardware_from_ground_truth"]["values"]["l2_bytes"]
        order = {}
        for c in cells:
            t = MC.TARGET_RATIO[c["regime"]]
            self.assertLess(abs(math.log((c["footprint_bytes"] / l2) / t)), math.log(3.0), c["cell_id"])
            order.setdefault((c["operator_id"], c["candidate_id"]), []).append((c["footprint_bytes"], c["regime"]))
        for k, v in order.items():
            fps = [fp for fp, _ in sorted(v)]
            self.assertEqual(len(fps), len(set(fps)), k)

    def test_both_tiers_present_in_every_family(self):
        for fam in {c["family"] for c in cells}:
            self.assertEqual({c["tier"] for c in cells if c["family"] == fam}, {"L2", "DRAM"}, fam)

    def test_candidates_use_the_grid_rule(self):
        sm = doc["hardware_from_ground_truth"]["values"]["sm_count"]
        g1, g2 = MC.grid_candidates(sm)
        self.assertEqual((g1, g2), (128, 256))
        for c in cells:
            if c["family"] == "sp":
                self.assertEqual(c["kernels"][0]["grid"][0], g1 if c["candidate_id"] == "c1" else g2)

    def test_vector_add_is_listed_as_excluded_before_the_freeze(self):
        ex = doc["excluded_before_freeze"]
        self.assertEqual(len(ex), 1); self.assertEqual((ex[0]["kernel"], ex[0]["count"]), ("vecAdd", 4)); self.assertFalse(ex[0]["counted_as_failure"])
        self.assertEqual([c for c in cells if "vector_add" in c["operator_id"]], [])

    def test_no_l1_tier(self):
        self.assertIn("none", doc["derivation"]["l1_tier"])
        self.assertEqual({c["tier"] for c in cells}, {"L2", "DRAM"})


class GeometryMatchesTheBlackwellCellFiles(unittest.TestCase):
    """The kernel-launch builders reproduce the committed Blackwell cell lists when given Blackwell's own dimensions."""

    def test_unseen_kernels(self):
        import cells as CB
        hw = dict(l2_bytes=134217728)
        n = 0
        for b in CB.define_cells(hw):
            c = b["controls"]
            f = b["family"]
            if f == "matmul":
                mine = MC.mm_kernels(c["N"], c["tile"])
            elif f == "bs":
                mine = MC.bs_kernels(c["optN"], c["block"])
            elif f == "scan":
                mine = MC.scan_kernels(c["n"], c["array_length"])
            else:
                mine = MC.conv_kernels(c["imageW"], c["imageH"])
            self.assertEqual(mine, b["kernels"], b["cell_id"]); n += 1
        self.assertEqual(n, 32)

    def test_set_e(self):
        import cells_e as CE
        n = 0
        for b in CE.define_cells(dict(l2_bytes=134217728)):
            c = b["controls"]
            if b["family"] == "sp":
                mine = MC.sp_kernels(c["vectorN"], c["elementN"], c["grid"])
            else:
                mine = MC.fwt_kernels(c["log2N"], c["batches"])
            self.assertEqual(mine, b["kernels"], b["cell_id"]); n += 1
        self.assertEqual(n, 16)

    def test_set_d_geometry(self):
        d = json.loads((SR / "fresh_d" / "fresh_cells_d.json").read_text())["cells"]
        by = {(c["kernel"], c["regime"]): c for c in d}
        for c in cells:
            if c["set"] != "set_d":
                continue
            b = by[(c["kernel"], c["regime"])]
            if c["family"] == "tile":
                self.assertEqual((c["geometry"]["block"], c["geometry"]["grid"][2]), (b["geometry"]["block"], 1))
                self.assertEqual(c["geometry"]["grid"][0], c["geometry"]["dim_x"] // 32)
            else:
                self.assertEqual((c["geometry"]["threads"], c["geometry"]["block"]), (b["geometry"]["threads"], b["geometry"]["block"]))
                self.assertEqual(c["geometry"]["blocks"], 64 if c["kernel"] == "reduce6" else c["geometry"]["n"] // 256)
            self.assertEqual(c["timing"]["runner"], b["timing"]["runner"]); self.assertEqual(c["timing"]["input_kind"], b["timing"]["input_kind"])
            self.assertEqual(c["timing"]["numeric_args"][-1], b["timing"]["numeric_args"][-1])       # the kernel selector


class PickRule(unittest.TestCase):
    def test_pick_takes_the_nearest_allowed_footprint(self):
        l2 = 1000                                                          # medium target = 125
        self.assertEqual(MC.pick("medium", l2, [(1, 10), (2, 100), (3, 130), (4, 500)])[0], 3)       # 130 is nearest; 500 is in the forbidden band
        self.assertEqual(MC.pick("medium", l2, [(1, 10), (2, 110), (3, 160)])[0], 2)                  # 110 is 12.8% below, 160 is 24.7% above in log distance
        with self.assertRaises(SystemExit):
            MC.pick("xlarge", l2, [(1, 10), (2, 1400)])                    # 1.4 x L2 is in the forbidden band
        self.assertEqual(MC.pick("xlarge", l2, [(1, 10), (2, 1400), (3, 1600)])[0], 3)
        self.assertEqual(MC.pick("large", l2, [(5, 300), (4, 300)])[0], 4)                            # ties go to the smaller dimension


if __name__ == "__main__":
    unittest.main()
