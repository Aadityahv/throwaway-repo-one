"""CPU tests for the unseen-kernel static pipeline: every built cell is checked against closed-form counts derived by hand
from the samples' source (not from the pipeline), plus ABI validity, tier rules and the frozen energy coefficients.

Run after the cells are built:  python3 test_unseen.py
"""
import json
import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cells as CELLS  # noqa: E402

CELL_DIR = HERE / "build/cells"


def load_all():
    out = {}
    for p in sorted(CELL_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        out[d["cell"]["cell_id"]] = d
    return out


ALL = load_all()


def fam(f, k):
    return f["per_launch_totals"]["families"].get(k, {}).get("lane_instructions", 0)


class TestCells(unittest.TestCase):
    def test_all_32_cells_built_and_supported(self):
        self.assertEqual(len(ALL), 32)
        bad = {k: v.get("refused") or v["features"]["status"] for k, v in ALL.items() if "refused" in v or v["features"]["status"] == "missing_features"}
        self.assertFalse(bad, bad)

    def test_tier_rule_and_no_marginal_cells(self):
        for cid, d in ALL.items():
            c = d["cell"]
            self.assertEqual(c["tier"], "L2" if c["footprint_bytes"] / 134217728 < 1 else "DRAM", cid)
            self.assertFalse(0.4 <= c["l2_ratio"] <= 1.5, cid)

    def test_abi_checks_pass_for_every_kernel(self):
        for cid, d in ALL.items():
            f = d["features"]
            for k in [f["main_kernel"]] + [s for s in f["secondary_kernels"]]:
                abi = k["abi"]
                self.assertTrue(abi["all_constant_bank_accesses_on_declared_fields"], (cid, abi["problems"]))

    def test_matrix_multiply_closed_forms(self):
        for cid, d in ALL.items():
            c = d["cell"]
            if c["operator_id"] != "unseen_cuda_samples_matrix_multiply":
                continue
            n, tile = c["controls"]["N"], c["controls"]["tile"]
            f = d["features"]
            trips = n // tile
            w = f["work"]
            self.assertEqual(w["six_class_counts"]["fma"], n ** 3, cid)
            self.assertEqual(fam(f, "global_load"), 2 * trips * n * n, cid)
            self.assertEqual(fam(f, "global_store"), n * n, cid)
            self.assertEqual(w["six_class_counts"]["barrier"], 2 * trips * n * n, cid)
            self.assertEqual(w["six_class_counts"]["shared_store"], 2 * trips * n * n, cid)
            self.assertEqual(len(d["phases"]["kernels"][0]["phases"]), 2 * trips + 1, cid)
            self.assertEqual(d["unique"]["kernels"][0]["unique_read_sectors_per_block"], n * tile // 4, cid)
            self.assertTrue(w["all_counts_exact"], cid)

    def test_black_scholes_closed_forms(self):
        for cid, d in ALL.items():
            c = d["cell"]
            if c["operator_id"] != "unseen_cuda_samples_black_scholes":
                continue
            threads = c["controls"]["optN"] // 2
            f = d["features"]
            oc = f["per_launch_totals"]["opcode_lane_counts"]
            self.assertEqual(fam(f, "global_load"), 3 * threads, cid)
            self.assertEqual(fam(f, "global_store"), 2 * threads, cid)
            for op, per_thread in (("MUFU.EX2", 6), ("MUFU.LG2", 2), ("MUFU.RCP", 10), ("MUFU.RSQ", 2), ("FCHK", 2)):
                self.assertEqual(oc.get(op), per_thread * threads, (cid, op))

    def test_scan_closed_forms(self):
        for cid, d in ALL.items():
            c = d["cell"]
            if c["operator_id"] != "unseen_cuda_samples_scan":
                continue
            n = c["controls"]["n"]
            nb = n // 1024
            f = d["features"]
            self.assertEqual(f["per_launch_totals"]["kernels_per_launch"], 3, cid)
            self.assertEqual(fam(f, "global_load"), n // 4 + 2 * nb + n // 4 + nb, cid)
            self.assertEqual(fam(f, "global_store"), n // 4 + nb + n // 4, cid)
            self.assertTrue(f["per_launch_totals"]["all_counts_exact"], cid)

    def test_convolution_closed_forms(self):
        for cid, d in ALL.items():
            c = d["cell"]
            if c["operator_id"] != "unseen_cuda_samples_separable_convolution":
                continue
            w, h = c["controls"]["imageW"], c["controls"]["imageH"]
            gx, gy = w // 128, h // 64
            f = d["features"]
            rows_loads = h * (w + 32 * (gx - 1))
            cols_loads = w * (h + 16 * (gy - 1))
            self.assertEqual(fam(f, "global_load"), rows_loads + cols_loads, cid)
            self.assertEqual(fam(f, "global_store"), 2 * w * h, cid)
            self.assertEqual(f["per_launch_totals"]["opcode_lane_counts"].get("FFMA"), 2 * 17 * w * h, cid)

    def test_phase_bytes_equal_global_instruction_bytes(self):
        """Executed bytes of the phase table equal the lane-level global bytes of the feature table, for every kernel."""
        for cid, d in ALL.items():
            f = d["features"]
            t = f["per_launch_totals"]
            ph = sum(p["read_bytes"] + p["write_bytes"] for k in d["phases"]["kernels"] for p in k["phases"])
            self.assertEqual(ph, t["executed_global_load_bytes_lane_level"] + t["executed_global_store_bytes_lane_level"], cid)


class TestEnergyModel(unittest.TestCase):
    def test_frozen_coefficients_match_exploratory_analysis(self):
        import energy_model as EM
        m = json.loads(EM.FROZEN.read_text())
        r = m["rates_pJ_per_unit"]
        self.assertAlmostEqual(r["bytes_L2"], 57.96, 1)
        self.assertAlmostEqual(r["bytes_DRAM"], 172.26, 1)
        self.assertAlmostEqual(r["nonmem_lane_instructions"], 10.32, 1)
        self.assertAlmostEqual(r["sfu_lane_instructions"], 85.70, 1)
        self.assertEqual(m["fitted_cells"], 73)

    def test_predict_is_capped_and_decomposes(self):
        import energy_model as EM
        m = json.loads(EM.FROZEN.read_text())
        terms = dict(bytes_L2=1e6, bytes_DRAM=0.0, nonmem_lane_instructions=1e6, sfu_lane_instructions=0)
        p = EM.predict(terms, 1e-3, m)
        self.assertAlmostEqual(p["uncapped_j"], p["runtime_term_j"] + p["memory_term_j"] + p["compute_term_j"] + p["sfu_term_j"])
        self.assertLessEqual(p["energy_j"], m["cap_w"] * 1e-3 + 1e-12)
        huge = EM.predict(dict(terms, nonmem_lane_instructions=1e15), 1e-3, m)
        self.assertTrue(huge["capped"])
        self.assertAlmostEqual(huge["energy_j"], m["cap_w"] * 1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
