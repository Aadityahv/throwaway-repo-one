#!/usr/bin/env python3
"""CPU-only tests for the fresh-shape static pipeline. Run:  python3 test_fresh.py

Needs the outputs of build_fresh.py (fresh_cells.json, features_fresh.json, phases_fresh.json,
phases_fresh_divider_corrected.json, dispatch_trace_fresh.json). No GPU, no ssh, no measured value is read; the
development table is opened only for its non-measured columns (ids, shapes, tier, ratio, logical bytes).
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATIC = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STATIC))
import build_fresh as B  # noqa: E402
import divider_fix  # noqa: E402

REPO = B.X.REPO
DEV_CSV = REPO / "tiresias/framework/compile_evidence/development/development_cells.csv"
DEV_COLS = ("gpu", "operator_id", "regime", "candidate_id", "shape_a", "shape_b", "tier", "l2_ratio", "actual_logical_bytes_per_launch")


def J(name):
    p = HERE / name
    if not p.exists():
        raise AssertionError("%s missing: run build_fresh.py first" % name)
    return json.loads(p.read_text())


def dev_table_rows():
    out = {}
    with DEV_CSV.open() as fh:
        rows_ = list(csv.DictReader(fh))
    for r in rows_:
        if r["gpu"] == "blackwell" and r["operator_id"] in (B.DEV_SOFTMAX_OP, B.DEV_LAYERNORM_OP):
            out[(r["operator_id"], r["regime"], r["candidate_id"])] = {k: r[k] for k in DEV_COLS}  # whitelist: no measured column
    return out


class Definitions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = J("fresh_cells.json")
        cls.cells = cls.doc["cells"]

    def test_count_and_ids_complete(self):
        self.assertEqual(len(self.cells), 24)
        want = {"blackwell/%s/%s/%s" % (op, r, c) for op in (B.SOFTMAX_OP, B.LAYERNORM_OP) for r in B.REGIMES for c in B.CANDS}
        self.assertEqual({c["cell_id"] for c in self.cells}, want)
        self.assertEqual(len(want), 24)
        for op in (B.SOFTMAX_OP, B.LAYERNORM_OP):
            self.assertEqual(sum(c["operator_id"] == op for c in self.cells), 12)

    def test_shapes_differ_from_development(self):
        dev = {(c["controls"].get("outer_size", c["controls"].get("rows")), c["controls"].get("dim_size", c["controls"].get("cols")))
               for c in json.loads((STATIC / "pytorch_dispatch/dispatch_trace.json").read_text())["cells"]
               if c["operator_id"] in (B.DEV_SOFTMAX_OP, B.DEV_LAYERNORM_OP)}
        self.assertEqual(dev, set(B.DEV_SHAPES.values()))
        for c in self.cells:
            self.assertNotIn((c["rows"], c["cols"]), dev)
            self.assertNotIn(c["cols"], {128, 512, 1024})

    def test_rows_cols_rules(self):
        for c in self.cells:
            self.assertEqual(c["rows"] % 8, 0)
            self.assertEqual(c["rows"] % 4, 0)
            self.assertIn(c["cols"], (96, 384, 768))
            self.assertEqual((c["rows"] * c["cols"]) % 256, 0)
            ratio = c["rows"] * c["cols"] * 4 / B.DEV_FOOTPRINT_TARGET_BYTES[c["regime"]]
            self.assertTrue(1 / 1.5 <= ratio <= 1.5, (c["cell_id"], ratio))

    def test_padding_rule_matches_development(self):
        pads = {"c1": 0, "c2": 1, "c3": 3, "c4": 7}
        for c in self.cells:
            self.assertEqual(c["row_stride"], c["cols"] + pads[c["candidate_id"]])
            self.assertEqual(c["controls"]["row_stride"], c["row_stride"])
        self.assertEqual(self.doc["padding_rule"]["pad_by_candidate"], pads)

    def test_byte_and_tier_rules_reproduce_development_table(self):
        """The rules used for fresh cells, applied to the development shapes, reproduce the development table's
        tier, L2 ratio and logical bytes for all 24 development cells."""
        l2 = B.l2_bytes_from_ground_truth()[1]
        self.assertEqual(l2, 134217728)
        dev = dev_table_rows()
        self.assertEqual(len(dev), 24)
        for (op, regime, cand), row in dev.items():
            rows, cols = B.DEV_SHAPES[regime]
            stride = cols + {"c1": 0, "c2": 1, "c3": 3, "c4": 7}[cand]
            self.assertEqual((int(row["shape_a"]), int(row["shape_b"])), (rows, cols))
            if op == B.DEV_SOFTMAX_OP:
                rb, wb = B.softmax_logical_bytes(rows, cols)
                fp = B.softmax_footprint(rows, cols, stride)
            else:
                rb, wb = B.layernorm_logical_bytes(rows, cols)
                fp = B.layernorm_footprint(rows, cols, stride)
            tier, ratio = B.tier_for(fp, l2)
            self.assertEqual(rb + wb, int(row["actual_logical_bytes_per_launch"]), (op, regime, cand))
            self.assertEqual(tier, row["tier"])
            self.assertAlmostEqual(ratio, float(row["l2_ratio"]), places=12)

    def test_fresh_tier_and_bytes(self):
        l2 = self.doc["hardware_l2_bytes_from_ground_truth"]
        for c in self.cells:
            rows, cols, stride = c["rows"], c["cols"], c["row_stride"]
            if c["operator_id"] == B.SOFTMAX_OP:
                self.assertEqual(c["logical_bytes_per_launch"], 8 * rows * cols)
                self.assertEqual(c["footprint_bytes"], 8 * rows * cols)
            else:
                self.assertEqual(c["logical_bytes_per_launch"], 16 * rows * cols + 16 * rows)
                self.assertEqual(c["footprint_bytes"], 4 * rows * stride + 8 * cols + 4 * (rows * cols + 2 * rows))
            self.assertEqual(c["tier"], "L2" if c["footprint_bytes"] / l2 < 1 else "DRAM")
            self.assertEqual((c["shape_a"], c["shape_b"]), (rows, cols))


class Dispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells = J("fresh_cells.json")["cells"]
        cls.disp = {c["cell_id"]: c for c in J("dispatch_trace_fresh.json")["cells"]}
        cls.idx = json.loads((STATIC / "libtorch_sm120/isolated_index.json").read_text())

    def test_kernel_identity_geometry_and_copy(self):
        kid_for_log2 = {7: "k5", 9: "k6", 10: "k4"}
        for c in self.cells:
            d = self.disp[c["cell_id"]]
            rows, cols, stride = c["rows"], c["cols"], c["row_stride"]
            ks = d["kernels"]
            self.assertEqual(len(ks), 1 if stride == cols else 2)
            self.assertEqual(c["kernels_per_launch"], len(ks))
            if stride != cols:
                copy = ks[0]
                self.assertRegex(copy["demangled_name_regex"], "elementwise_kernel<128, 2")
                self.assertEqual(copy["grid"], [rows * cols // 256, 1, 1])
                self.assertEqual(copy["block"], [128, 1, 1])
            main = ks[-1]
            if c["operator_id"] == B.SOFTMAX_OP:
                l2e = math.ceil(math.log2(cols))
                self.assertEqual(main["template_args"]["log2_elements"], l2e)
                self.assertEqual(l2e, {96: 7, 384: 9, 768: 10}[cols])
                self.assertIn("softmax_warp_forwardIfffLi%dELb0ELb0E" % l2e, self.idx[kid_for_log2[l2e]]["mangled"])
                npot = 1 << l2e
                warp = min(npot, 32)
                bpw = 2 if npot <= 128 else 1
                self.assertEqual(main["block"], [warp, 128 // warp, 1])
                self.assertEqual(main["grid"][0], math.ceil(rows / ((128 // warp) * bpw)))
                self.assertEqual(rows % ((128 // warp) * bpw), 0)  # no partial block
                self.assertEqual(main["args"]["element_count"], cols)
                self.assertEqual(main["args"]["stride"], cols)  # dense after contiguous()
            else:
                self.assertEqual(cols % 4, 0)  # vectorized path condition
                self.assertEqual(main["grid"], [rows, 1, 1])
                self.assertEqual(main["block"], [32, 4, 1])
                self.assertEqual(main["dynamic_smem_bytes"], 24)
                self.assertRegex(main["demangled_name_regex"], "vectorized_layer_norm_kernel<float, float, false>")
                self.assertIn("vectorized_layer_norm_kernelIffLb0E", self.idx["k3"]["mangled"])

    def test_retained_sass_covers_every_kernel(self):
        import pytorch_features.adapter as A_  # noqa: F401  (only to prove importability of the committed adapter)
        for c in self.cells:
            feats_kids = [k["kernel_id"] for k in c["kernels"]]
            for kid in feats_kids:
                self.assertTrue((STATIC / "libtorch_sm120" / (kid + ".isolated.sass")).exists(), kid)
            self.assertIn(feats_kids[-1], ("k3", "k4", "k5", "k6"))

    def test_dispatch_rules_are_called_not_copied(self):
        """Regenerating the dispatch cells from the controls gives exactly the file on disk."""
        l2 = B.l2_bytes_from_ground_truth()[1]
        cells, _ = B.define_cells(l2)
        again = {d["cell_id"]: d for d in B.dispatch_cells(cells)}
        self.assertEqual(json.dumps(again, sort_keys=True), json.dumps(self.disp, sort_keys=True))


def shape(d, depth):
    if not isinstance(d, dict) or depth == 0:
        return None
    return {k: shape(v, depth - 1) for k, v in d.items()}


class Schemas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = J("features_fresh.json")
        cls.ph = J("phases_fresh.json")
        cls.phc = J("phases_fresh_divider_corrected.json")
        cls.F = json.loads((STATIC / "features_blackwell.json").read_text())
        cls.P = json.loads((STATIC / "phases_blackwell.json").read_text())
        cls.frow = {r["cell_id"]: r for r in cls.F["rows"]}

    def test_top_level_keys(self):
        self.assertEqual(set(self.f), set(self.F))
        self.assertEqual(self.f["schema"], self.F["schema"])
        self.assertEqual(self.f["hardware_from_ground_truth"], self.F["hardware_from_ground_truth"])
        for p in (self.ph, self.phc):
            self.assertEqual(set(p), set(self.P))
            self.assertEqual(p["schema"], self.P["schema"])

    def test_row_schema_equals_frozen_pytorch_rows(self):
        for r in self.f["rows"]:
            kind = "dev_pytorch_rowwise_softmax" if r["operator_id"] == B.SOFTMAX_OP else "final_pytorch_layer_norm"
            ref = self.frow["blackwell/%s/%s" % (kind, r["cell"])]
            self.assertEqual(set(r), set(ref), r["cell_id"])
            for key in ("resources", "occupancy", "work", "structure", "memory", "geometry", "main_kernel", "inputs"):
                self.assertEqual(set(r[key]), set(ref[key]), (r["cell_id"], key))
            self.assertEqual(len(r["secondary_kernels"]), len(ref["secondary_kernels"]))
            for a, b in zip(r["secondary_kernels"], ref["secondary_kernels"]):
                self.assertEqual(set(a), set(b))
                for key in ("resources", "occupancy", "work", "structure"):
                    self.assertEqual(set(a[key]), set(b[key]), (r["cell_id"], "secondary", key))
            self.assertEqual(set(r["per_launch_totals"]), set(ref["per_launch_totals"]))
            self.assertEqual(r["status"], "supported_with_assumptions", r["cell_id"])
            self.assertEqual(r["corpus"], "libtorch_sm120")

    def test_phase_schema_equals_frozen(self):
        for table in (self.ph, self.phc):
            for cid, row in table["rows"].items():
                kind = "dev_pytorch_rowwise_softmax" if "fresh_pytorch_rowwise_softmax" in cid else "final_pytorch_layer_norm"
                ref = self.P["rows"]["blackwell/%s/%s" % (kind, cid.split("/", 2)[2])]
                self.assertEqual(set(row), set(ref), cid)
                if row["status"] == "conditional_static_phases":
                    self.assertEqual(len(row["kernels"]), len(ref["kernels"]), cid)
                    for k, kr in zip(row["kernels"], ref["kernels"]):
                        self.assertTrue(set(kr) <= set(k), cid)  # corrected variant adds `block_classes` for the copy kernel
                        self.assertEqual(k["kernel_id"], kr["kernel_id"])
                        for p, pr in zip(k["phases"], kr["phases"]):
                            self.assertEqual(set(p), set(pr))

    def test_features_and_phases_cover_the_same_cells(self):
        ids = {c["cell_id"] for c in J("fresh_cells.json")["cells"]}
        self.assertEqual({r["cell_id"] for r in self.f["rows"]}, ids)
        self.assertEqual(set(self.ph["rows"]), ids)
        self.assertEqual(set(self.phc["rows"]), ids)

    def test_no_zero_fill(self):
        for r in self.f["rows"]:
            self.assertEqual(r["missing_features"], [], r["cell_id"])
            self.assertGreater(r["work"]["total_lane_instructions"], 0)
            self.assertTrue(r["work"]["six_class_cross_check_all_equal"], r["cell_id"])
            self.assertIsNotNone(r["occupancy"]["blocks_per_sm"])
            self.assertIsNotNone(r["occupancy"]["waves"])
            self.assertEqual(r["memory"]["tier"], "L2")
            for s in r["secondary_kernels"]:
                self.assertGreater(s["work"]["total_lane_instructions"], 0)
        for table in (self.ph, self.phc):
            for cid, row in table["rows"].items():
                if row["status"] == "unsupported":
                    self.assertTrue(row["reason"] and row["kernels"] == [], cid)  # exact reason, no placeholder numbers
                else:
                    self.assertEqual(row["status"], "conditional_static_phases")
                    for k in row["kernels"]:
                        for p in k["phases"]:
                            for key in ("read_sectors", "write_sectors", "lines", "read_bytes", "write_bytes",
                                        "critical_path_compute_instructions", "dependent_global_load_depth"):
                                self.assertIsNotNone(p[key], (cid, key))
                            self.assertGreater(sum(p["issue_warp_instructions"].values()), 0)
                        self.assertGreater(sum(p["read_sectors"] for p in k["phases"]), 0, cid)
                        self.assertGreater(sum(p["write_sectors"] for p in k["phases"]), 0, cid)

    def test_frozen_pipeline_refusals_are_exact(self):
        """The frozen pipeline derives the six contiguous cells and refuses the padded ones with its own message."""
        for cid, row in self.ph["rows"].items():
            padded = not cid.endswith("/c1")
            if padded:
                self.assertEqual(row["status"], "unsupported", cid)
                self.assertEqual(row["reason"], "PyTorch boundary block phase signatures differ", cid)
            else:
                self.assertEqual(row["status"], "conditional_static_phases", cid)

    def test_phase_bytes_match_logical_bytes_structure(self):
        """Main-kernel static bytes: softmax reads and writes rows*cols*4; copy kernel the same."""
        cells = {c["cell_id"]: c for c in J("fresh_cells.json")["cells"]}
        for cid, row in self.phc["rows"].items():
            c = cells[cid]
            n4 = c["rows"] * c["cols"] * 4
            for k in row["kernels"]:
                rb = sum(p["read_bytes"] for p in k["phases"])
                wb = sum(p["write_bytes"] for p in k["phases"])
                if k["kernel_id"] == "k1" or c["operator_id"] == B.SOFTMAX_OP:
                    self.assertEqual((rb, wb), (n4, n4), (cid, k["kernel_id"]))


class CorrectedCopyKernel(unittest.TestCase):
    """The copy-kernel sector totals of the corrected variant equal a closed form that does not use the interpreter."""

    @classmethod
    def setUpClass(cls):
        cls.cells = {c["cell_id"]: c for c in J("fresh_cells.json")["cells"]}
        cls.phc = J("phases_fresh_divider_corrected.json")["rows"]

    @staticmethod
    def closed_form(rows, cols, stride):
        reads = lines = 0
        for r in range(rows):
            off = (4 * stride * r) % 128
            reads += (cols // 32) * (4 if off % 32 == 0 else 5)
            lines += (cols // 32) * (1 if off == 0 else 2)
        return reads, lines

    def test_copy_sectors_and_lines_match_closed_form(self):
        n = 0
        for cid, c in self.cells.items():
            if c["row_stride"] == c["cols"]:
                continue
            k1 = next(k for k in self.phc[cid]["kernels"] if k["kernel_id"] == "k1")
            ph = k1["phases"][0]
            reads, lines_r = self.closed_form(c["rows"], c["cols"], c["row_stride"])
            writes = c["rows"] * c["cols"] // 8  # dense aligned output: 32 B sectors
            self.assertEqual(ph["read_sectors"], reads, cid)
            self.assertEqual(ph["write_sectors"], writes, cid)
            self.assertEqual(ph["lines"], lines_r + c["rows"] * c["cols"] // 32, cid)  # read lines + write lines (1 per request)
            n += 1
        self.assertEqual(n, 18)

    def test_corrected_interpreter_reproduces_offset_calculator_addresses(self):
        PH, A, C = B.PH, B.A, B.C
        Fixed = divider_fix.corrected_interp_class(A.P)
        orig = A.P.Interp
        A.P.Interp = Fixed
        try:
            disp = {d["cell_id"]: d for d in J("dispatch_trace_fresh.json")["cells"]}
            for cid in ("blackwell/fresh_pytorch_rowwise_softmax/small/c3", "blackwell/fresh_pytorch_layer_norm/medium/c4"):
                cell = disp[cid]
                kern = cell["kernels"][0]
                constants, grid, block = PH.bind_pytorch("k1", kern, cell)
                rows, cols = cell["input"]["shape"]
                stride = cell["input"]["strides"][0]
                sites = A.P.parse((STATIC / "libtorch_sm120/k1.isolated.sass").read_text())
                for b in (0, 1, 5, 11, grid[0] - 1):
                    obs = PH.Observer(128)
                    obs.block_x = 128
                    interp = A.P.Interp(C.D, ext=True, fchk_fast_path=True, trace=obs.pytorch)

                    def coords(l, b=b):
                        return {k: A.P.V.exact(v) for k, v in {"SR_CTAID.X": b, "SR_CTAID.Y": 0, "SR_CTAID.Z": 0, "SR_CgaCtaId": 0, "SR_TID.X": l,
                                                               "SR_TID.Y": 0, "SR_TID.Z": 0, "SR_LANEID": l % 32}.items()}
                    interp.run_block(sites, {k: A.P.V.exact(v) for k, v in constants.items()}, coords, 128)
                    base = C.PTR_BASE0 + C.PTR_STRIDE  # data_1 = input
                    seen = 0
                    for (w, ph, pc, k, direction, width), addrs in obs.memory.items():
                        if direction != "read":
                            continue
                        second = pc > 6000  # second element of the thread's two (+128)
                        for lane, a in enumerate(addrs):
                            e = b * 256 + w * 32 + lane + (128 if second else 0)
                            self.assertEqual(a - base, 4 * (e % cols) + 4 * stride * (e // cols), (cid, b, w, lane))
                            seen += 1
                    self.assertEqual(seen, 256)
        finally:
            A.P.Interp = orig

    def test_features_identical_under_correction(self):
        self.assertTrue(J("fresh_cells.json")["features_identical_under_corrected_imad_hi"])


class Prediction(unittest.TestCase):
    """predict_runtime_v2.build consumes the fresh tables as it consumes the frozen ones (smoke test only)."""

    def test_build_consumes_fresh_tables(self):
        import predict_runtime_v2 as V2
        features = J("features_fresh.json")
        K = V2.make_constants(json.loads((STATIC / "constants/stream_constants.json").read_text())["constants"],
                              json.loads((STATIC / "constants/microbench_constants_v2.json").read_text()))
        for fname, expect_supported in (("phases_fresh.json", 6), ("phases_fresh_divider_corrected.json", 24)):
            out = V2.build(features, J(fname)["rows"], K)
            self.assertEqual(len(out), 24)
            ok = [v for v in out.values() if v["primary_s"] is not None]
            self.assertEqual(len(ok), expect_supported, fname)
            for v in out.values():
                if v["primary_s"] is None:
                    self.assertTrue(v["unsupported_reason"])
                else:
                    self.assertTrue(math.isfinite(v["primary_s"]) and v["primary_s"] > 0)


class TimingScript(unittest.TestCase):
    def test_dry_run_lists_cells_without_torch(self):
        code = ("import sys; sys.modules['torch'] = None; sys.argv = ['timing_fresh.py', '--dry-run']; "
                "import runpy; runpy.run_path(%r, run_name='__main__')" % str(HERE / "timing_fresh.py"))
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("fresh cells: 24", p.stdout)
        self.assertEqual(len(re.findall(r"^blackwell/fresh_pytorch_", p.stdout, re.M)), 24)
        self.assertIn("F.softmax(view, dim=1)", p.stdout)
        self.assertIn("F.layer_norm(view, [", p.stdout)

    def test_refuses_without_single_device_and_booking(self):
        env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        out = HERE / "never_written.json"
        p = subprocess.run([sys.executable, str(HERE / "timing_fresh.py"), "--booking-ref", "x", "--out", str(out)], capture_output=True, text=True, env=env)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("REFUSED", p.stderr + p.stdout)
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
        p = subprocess.run([sys.executable, str(HERE / "timing_fresh.py"), "--booking-ref", "x", "--out", str(out)], capture_output=True, text=True, env=env)
        self.assertIn("REFUSED", p.stderr + p.stdout)
        p = subprocess.run([sys.executable, str(HERE / "timing_fresh.py"), "--out", str(out)], capture_output=True, text=True,
                           env=dict(env, CUDA_VISIBLE_DEVICES="1"))
        self.assertIn("REFUSED", p.stderr + p.stdout)
        self.assertFalse(out.exists())

    def test_script_matches_runner_calls(self):
        src = (HERE / "timing_fresh.py").read_text()
        sm = (REPO / "tiresias/app_runners/pytorch_softmax_runner.py").read_text()
        ln = (REPO / "tiresias/app_runners/pytorch_layernorm_runner.py").read_text()
        for needle in ("F.softmax(view, dim=1)",):
            self.assertIn(needle.replace("F.", "torch.nn.functional."), sm)
            self.assertIn(needle, src)
        self.assertIn("torch.nn.functional.layer_norm(view, [cols], weight, bias, eps=1e-5)", ln)
        self.assertIn("F.layer_norm(view, [cols], weight, bias, eps=1e-5)", src)
        self.assertIn("padded[:, :cols]", sm + ln)
        self.assertIn("view = padded[:, :cols]", src)
        self.assertIn("SEED = 20260914", src)
        self.assertIn("SEED = 20260914", sm)


if __name__ == "__main__":
    unittest.main(verbosity=1)
