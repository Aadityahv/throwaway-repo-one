#!/usr/bin/env python3
"""CPU-only tests for fresh set B. Run:  python3 test_fresh_b.py

Needs build_fresh_b.py and phases_unique_fresh_b.py outputs. No GPU, no ssh; no measured runtime or energy value is read
(the development table is not opened at all; the development shapes come from the dispatch trace controls).
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FRESH = HERE.parent / "fresh"
STATIC = HERE.parent
sys.path.insert(0, str(HERE))
import build_fresh_b as BB  # noqa: E402  (also patches the imported set A builder; read-only use here)

B = BB.B
SOFT, LN = BB.SOFTMAX_OP, BB.LAYERNORM_OP
CANDS = ("c1", "c2", "c3", "c4")
PADS = {"c1": 0, "c2": 1, "c3": 3, "c4": 7}


def J(path):
    p = Path(path) if Path(path).is_absolute() else HERE / path
    if not p.exists():
        raise AssertionError("%s missing: run build_fresh_b.py / phases_unique_fresh_b.py first" % p)
    return json.loads(p.read_text())


def shape(d, depth):
    if not isinstance(d, dict) or depth == 0:
        return None
    return {k: shape(v, depth - 1) for k, v in d.items()}


class Definitions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = J("fresh_cells_b.json")
        cls.cells = cls.doc["cells"]
        cls.l2 = cls.doc["hardware_l2_bytes_from_ground_truth"]

    def test_l2_is_ground_truth(self):
        self.assertEqual(self.l2, B.l2_bytes_from_ground_truth()[1])
        self.assertEqual(self.l2, 134217728)

    def test_count_and_ids_complete(self):
        want = {"blackwell/%s/%s/%s" % (op, r, c) for op in (SOFT, LN) for r in BB.REGIMES for c in CANDS}
        self.assertEqual(len(want), 32)
        self.assertEqual([c["cell_id"] for c in self.cells].__len__(), 32)
        self.assertEqual({c["cell_id"] for c in self.cells}, want)
        self.assertEqual(self.doc["set"], "B")

    def test_shapes_differ_from_development_and_set_a(self):
        dev = {(c["controls"].get("outer_size", c["controls"].get("rows")), c["controls"].get("dim_size", c["controls"].get("cols")))
               for c in json.loads((STATIC / "pytorch_dispatch/dispatch_trace.json").read_text())["cells"]
               if c["operator_id"] in (B.DEV_SOFTMAX_OP, B.DEV_LAYERNORM_OP)}
        a = {(c["rows"], c["cols"]) for c in J(FRESH / "fresh_cells.json")["cells"]}
        self.assertEqual(dev, set(B.DEV_SHAPES.values()))
        self.assertEqual(a, set(BB.SET_A_SHAPES.values()))
        for c in self.cells:
            self.assertNotIn((c["rows"], c["cols"]), dev | a)
            self.assertNotIn(c["cols"], {128, 512, 1024, 96, 384, 768})
            self.assertTrue(c["cell_id"].split("/")[1].startswith("fresh_b_"))

    def test_rows_cols_rules(self):
        log2 = {112: 7, 320: 9, 640: 10, 960: 10}
        for c in self.cells:
            self.assertEqual(c["rows"] % 16, 0)
            self.assertEqual((c["rows"] * c["cols"]) % 256, 0)
            self.assertIn(c["cols"], log2)
            self.assertEqual(math.ceil(math.log2(c["cols"])), log2[c["cols"]])
            self.assertEqual(c["cols"] % 4, 0)

    def test_padding_rule(self):
        for c in self.cells:
            self.assertEqual(c["row_stride"], c["cols"] + PADS[c["candidate_id"]])
            self.assertEqual(c["controls"]["row_stride"], c["row_stride"])
        self.assertEqual(self.doc["padding_rule"]["pad_by_candidate"], PADS)

    def test_tier_rule_per_regime(self):
        for c in self.cells:
            rows, cols, stride = c["rows"], c["cols"], c["row_stride"]
            if c["operator_id"] == SOFT:
                self.assertEqual(c["footprint_bytes"], 8 * rows * cols)
                self.assertEqual(c["logical_bytes_per_launch"], 8 * rows * cols)
            else:
                self.assertEqual(c["footprint_bytes"], 4 * rows * stride + 8 * cols + 4 * (rows * cols + 2 * rows))
                self.assertEqual(c["logical_bytes_per_launch"], 16 * rows * cols + 16 * rows)
            ratio = c["footprint_bytes"] / self.l2
            self.assertAlmostEqual(ratio, c["l2_ratio"], places=12)
            self.assertEqual(c["tier"], "L2" if ratio < 1 else "DRAM")
            if c["regime"] == "xlarge":
                self.assertGreaterEqual(ratio, 1.15, c["cell_id"])
                self.assertEqual(c["tier"], "DRAM")
            elif c["regime"] == "large":
                self.assertLess(ratio, 0.6, c["cell_id"])
                self.assertEqual(c["tier"], "L2")
            else:
                self.assertLess(ratio, 0.6, c["cell_id"])
                self.assertEqual(c["tier"], "L2")


class Dispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells = J("fresh_cells_b.json")["cells"]
        cls.disp = {c["cell_id"]: c for c in J("dispatch_trace_fresh_b.json")["cells"]}
        cls.idx = json.loads((STATIC / "libtorch_sm120/isolated_index.json").read_text())

    def test_paths_and_geometry(self):
        kid_for_log2 = {7: "k5", 9: "k6", 10: "k4"}
        for c in self.cells:
            d = self.disp[c["cell_id"]]
            rows, cols, stride = c["rows"], c["cols"], c["row_stride"]
            ks = d["kernels"]
            self.assertEqual(len(ks), 1 if stride == cols else 2)
            if stride != cols:
                self.assertEqual(ks[0]["grid"], [rows * cols // 256, 1, 1])
                self.assertEqual(ks[0]["block"], [128, 1, 1])
            main = ks[-1]
            if c["operator_id"] == SOFT:
                # persistent-warp path (SoftMax.cu:1097) and an instantiation retained in libtorch_sm120
                self.assertTrue(cols <= 1024 and cols <= 2048 and cols * 4 <= 8192)
                l2e = math.ceil(math.log2(cols))
                self.assertEqual(main["template_args"]["log2_elements"], l2e)
                self.assertIn("softmax_warp_forwardIfffLi%dELb0ELb0E" % l2e, self.idx[kid_for_log2[l2e]]["mangled"])
                warp = min(1 << l2e, 32)
                bpw = 2 if (1 << l2e) <= 128 else 1
                self.assertEqual(main["block"], [warp, 128 // warp, 1])
                self.assertEqual(main["grid"][0], rows // ((128 // warp) * bpw))
                self.assertEqual(rows % ((128 // warp) * bpw), 0)  # no partial block
                self.assertEqual(main["args"]["element_count"], cols)
                self.assertEqual(main["args"]["stride"], cols)
                self.assertIn("masks columns >= element_count", main["extra_control_flow"])  # all four cols are below the next power of two
            else:
                # vectorized path (layer_norm_kernel.cu:1103-1111): N % 4 == 0, 16 B aligned rows
                self.assertEqual(cols % 4, 0)
                self.assertEqual((cols * 4) % 16, 0)
                self.assertEqual(main["grid"], [rows, 1, 1])
                self.assertEqual(main["block"], [32, 4, 1])
                self.assertRegex(main["demangled_name_regex"], "vectorized_layer_norm_kernel<float, float, false>")

    def test_dispatch_rules_are_called_not_copied(self):
        cells, _ = B.define_cells(B.l2_bytes_from_ground_truth()[1])
        again = {d["cell_id"]: d for d in B.dispatch_cells(cells)}
        self.assertEqual(json.dumps(again, sort_keys=True), json.dumps(self.disp, sort_keys=True))


class Schemas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = J("features_fresh_b.json")
        cls.p = J("phases_fresh_b.json")
        cls.u = J("phases_unique_fresh_b.json")
        cls.fa = J(FRESH / "features_fresh.json")
        cls.pa = J(FRESH / "phases_fresh_divider_corrected.json")
        cls.ua = J(FRESH / "phases_unique_fresh.json")
        cls.cells = {c["cell_id"]: c for c in J("fresh_cells_b.json")["cells"]}

    def test_top_level_and_cover_same_cells(self):
        self.assertEqual(set(self.f), set(self.fa))
        self.assertEqual(set(self.p), set(self.pa))
        self.assertEqual(set(self.u), set(self.ua))
        self.assertEqual(self.p["schema"], self.pa["schema"])
        self.assertEqual(self.u["schema"], self.ua["schema"])
        ids = set(self.cells)
        self.assertEqual({r["cell_id"] for r in self.f["rows"]}, ids)
        self.assertEqual(set(self.p["rows"]), ids)
        self.assertEqual(set(self.u["rows"]), ids)

    def test_row_schema_equals_set_a(self):
        ref = {r["operator_id"].replace("fresh_", ""): r for r in self.fa["rows"]}
        for r in self.f["rows"]:
            a = next(x for x in self.fa["rows"] if x["operator_id"] == r["operator_id"].replace("fresh_b_", "fresh_") and len(x["secondary_kernels"]) == len(r["secondary_kernels"]))
            self.assertEqual(set(r), set(a), r["cell_id"])
            for key in ("resources", "occupancy", "work", "structure", "memory", "geometry", "main_kernel", "inputs"):
                self.assertEqual(set(r[key]), set(a[key]), (r["cell_id"], key))
            for s, t in zip(r["secondary_kernels"], a["secondary_kernels"]):
                self.assertEqual(set(s), set(t))
            self.assertEqual(r["status"], "supported_with_assumptions")
            self.assertEqual(r["memory"]["tier"], self.cells[r["cell_id"]]["tier"])

    def test_phase_schema_equals_set_a(self):
        refs = {}
        for cid, row in self.pa["rows"].items():
            refs[(("soft" if "softmax" in cid else "ln"), len(row["kernels"]))] = row
        for cid, row in self.p["rows"].items():
            ref = refs[(("soft" if "softmax" in cid else "ln"), len(row["kernels"]))]
            self.assertEqual(set(row), set(ref), cid)
            self.assertEqual(row["status"], "conditional_static_phases", cid)
            for k, kr in zip(row["kernels"], ref["kernels"]):
                self.assertEqual(set(k), set(kr))
                if k["kernel_id"] in ("k1", "k3"):
                    self.assertEqual(k["kernel_id"], kr["kernel_id"])
                else:
                    self.assertIn(k["kernel_id"], ("k4", "k5", "k6"))
                for ph, pr in zip(k["phases"], kr["phases"]):
                    self.assertEqual(set(ph), set(pr))

    def test_first_touch_schema_equals_set_a(self):
        refs = {}
        for cid, row in self.ua["rows"].items():
            for k in row["kernels"]:
                refs[(("soft" if "softmax" in cid else "ln"), k["kernel_id"])] = k
        for cid, row in self.u["rows"].items():
            self.assertEqual(row["status"], "ok", (cid, row.get("reason")))
            for k in row["kernels"]:
                ref = refs[(("soft" if "softmax" in cid else "ln"), k["kernel_id"])]
                self.assertEqual(set(k), set(ref), cid)
                for ph, pr in zip(k["phases"], ref["phases"]):
                    self.assertEqual(set(ph), set(pr))

    def test_no_zero_fill(self):
        for r in self.f["rows"]:
            self.assertEqual(r["missing_features"], [], r["cell_id"])
            self.assertGreater(r["work"]["total_lane_instructions"], 0)
            self.assertIsNotNone(r["occupancy"]["blocks_per_sm"])
            self.assertIsNotNone(r["occupancy"]["waves"])
            for s in r["secondary_kernels"]:
                self.assertGreater(s["work"]["total_lane_instructions"], 0)
        for cid, row in self.p["rows"].items():
            for k in row["kernels"]:
                for ph in k["phases"]:
                    for key in ("read_sectors", "write_sectors", "lines", "read_bytes", "write_bytes", "critical_path_compute_instructions",
                                "dependent_global_load_depth"):
                        self.assertIsNotNone(ph[key], (cid, key))
                    self.assertGreater(sum(ph["issue_warp_instructions"].values()), 0)
                self.assertGreater(sum(ph["read_sectors"] for ph in k["phases"]), 0)
                self.assertGreater(sum(ph["write_sectors"] for ph in k["phases"]), 0)

    def test_totals_equal_between_phase_table_and_first_touch_table(self):
        for cid, o in self.u["rows"].items():
            for ko, kc in zip(o["kernels"], self.p["rows"][cid]["kernels"]):
                self.assertEqual(ko["kernel_id"], kc["kernel_id"])
                self.assertEqual(len(ko["phases"]), len(kc["phases"]))
                for po, pc in zip(ko["phases"], kc["phases"]):
                    for f in ("read_sectors", "write_sectors", "lines"):
                        self.assertEqual(po[f], pc[f], (cid, f))
                    self.assertEqual(po["read_sectors_total_requested"], pc["read_sectors"])

    def test_first_touch_not_above_total(self):
        for cid, o in self.u["rows"].items():
            for k in o["kernels"]:
                for ph in k["phases"]:
                    self.assertLessEqual(ph["read_sectors_first_touch"], ph["read_sectors"], cid)
                    self.assertGreaterEqual(ph["read_sectors_first_touch"], 0)
                self.assertGreater(sum(ph["read_sectors_first_touch"] for ph in k["phases"]), 0, cid)
                self.assertLessEqual(k["unique_read_sectors_per_block"], k["total_requested_read_sectors_per_block"] + 1e-9, cid)
                self.assertLessEqual(k["first_touch_read_sectors_per_block"], k["unique_read_sectors_per_block"] + 1e-9, cid)
                self.assertEqual(k["blocks_per_sm"], next(r for r in self.f["rows"] if r["cell_id"] == cid)["occupancy"]["blocks_per_sm"])

    def test_main_kernel_static_bytes(self):
        for cid, row in self.p["rows"].items():
            c = self.cells[cid]
            n4 = c["rows"] * c["cols"] * 4
            for k in row["kernels"]:
                rb = sum(p["read_bytes"] for p in k["phases"])
                wb = sum(p["write_bytes"] for p in k["phases"])
                if k["kernel_id"] == "k1" or c["operator_id"] == SOFT:
                    self.assertEqual((rb, wb), (n4, n4), (cid, k["kernel_id"]))

    def test_padded_cells_use_corrected_divider(self):
        self.assertIn("IMAD.HI.U32", " ".join(self.p["assumptions"]))
        n = 0
        for cid, row in self.p["rows"].items():
            if self.cells[cid]["row_stride"] != self.cells[cid]["cols"]:
                k1 = next(k for k in row["kernels"] if k["kernel_id"] == "k1")
                self.assertIn("block_classes", k1)
                n += 1
        self.assertEqual(n, 24)


class CopyKernelClosedForm(unittest.TestCase):
    """Interpreter-free count for ANY cols (warp requests may straddle rows): address = 4*(e mod cols) + 4*stride*(e div cols)."""

    @staticmethod
    def closed_form(rows, cols, stride):
        n = rows * cols
        reads = lines = 0
        step = 1 << 15  # requests per chunk
        lane = np.arange(32, dtype=np.int64)
        for j0 in range(0, n // 32, step):
            j = np.arange(j0, min(j0 + step, n // 32), dtype=np.int64)
            e = j[:, None] * 32 + lane[None, :]
            addr = 4 * (e % cols) + 4 * stride * (e // cols)
            for div, tot in ((32, "r"), (128, "l")):
                u = np.sort(addr // div, axis=1)
                cnt = int((u[:, 1:] != u[:, :-1]).sum()) + len(j)
                if tot == "r":
                    reads += cnt
                else:
                    lines += cnt
        return reads, lines

    def test_copy_sectors_and_lines(self):
        cells = {c["cell_id"]: c for c in J("fresh_cells_b.json")["cells"]}
        phases = J("phases_fresh_b.json")["rows"]
        cache, n = {}, 0
        for cid, c in cells.items():
            if c["row_stride"] == c["cols"]:
                continue
            key = (c["rows"], c["cols"], c["row_stride"])
            if key not in cache:
                cache[key] = self.closed_form(*key)
            reads, lines_r = cache[key]
            ph = next(k for k in phases[cid]["kernels"] if k["kernel_id"] == "k1")["phases"][0]
            self.assertEqual(ph["read_sectors"], reads, cid)
            self.assertEqual(ph["write_sectors"], c["rows"] * c["cols"] // 8, cid)
            self.assertEqual(ph["lines"], lines_r + c["rows"] * c["cols"] // 32, cid)
            n += 1
        self.assertEqual(n, 24)


class TimingScript(unittest.TestCase):
    def test_dry_run_without_torch(self):
        code = ("import sys; sys.modules['torch'] = None; sys.argv = ['timing_fresh_b.py', '--dry-run']; "
                "import runpy; runpy.run_path(%r, run_name='__main__')" % str(HERE / "timing_fresh_b.py"))
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("fresh cells: 32", p.stdout)
        self.assertEqual(len(re.findall(r"^blackwell/fresh_b_pytorch_", p.stdout, re.M)), 32)
        self.assertIn("F.softmax(view, dim=1)", p.stdout)
        self.assertIn("F.layer_norm(view, [", p.stdout)
        self.assertIn("rows=54608", p.stdout)

    def test_refusals(self):
        env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        out = HERE / "never_written.json"
        script = str(HERE / "timing_fresh_b.py")
        p = subprocess.run([sys.executable, script, "--booking-ref", "x", "--out", str(out)], capture_output=True, text=True, env=env)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("REFUSED", p.stderr + p.stdout)
        p = subprocess.run([sys.executable, script, "--booking-ref", "x", "--out", str(out)], capture_output=True, text=True, env=dict(env, CUDA_VISIBLE_DEVICES="0,1"))
        self.assertIn("REFUSED", p.stderr + p.stdout)
        p = subprocess.run([sys.executable, script, "--out", str(out)], capture_output=True, text=True, env=dict(env, CUDA_VISIBLE_DEVICES="1"))
        self.assertIn("REFUSED", p.stderr + p.stdout)
        existing = HERE / "phases_fresh_b.json"  # an existing output path is refused before anything else
        p = subprocess.run([sys.executable, script, "--booking-ref", "x", "--out", str(existing)], capture_output=True, text=True, env=dict(env, CUDA_VISIBLE_DEVICES="1"))
        self.assertIn("exists", p.stderr + p.stdout)
        self.assertFalse(out.exists())

    def test_script_matches_runner_calls_and_defaults(self):
        src = (HERE / "timing_fresh_b.py").read_text()
        ref = (FRESH / "timing_fresh.py").read_text()
        reset = HERE.parents[2] / "app_runners"
        sm = (reset / "pytorch_softmax_runner.py").read_text()
        ln = (reset / "pytorch_layernorm_runner.py").read_text()
        for needle in ("F.softmax(view, dim=1)", "F.layer_norm(view, [cols], weight, bias, eps=1e-5)", "SEED = 20260914", "view = padded[:, :cols]",
                       "GRAPH_LAUNCHES = 1000", "WARMUP_LAUNCHES = 3", "TARGET_WINDOW_S = 0.100", 'EXPECTED_UUID = "GPU-0e63baea-0bb1-e90c-f19d-8aa5f2995894"',
                       "refuse_if_busy(EXPECTED_UUID)"):
            self.assertIn(needle, src)
            self.assertIn(needle, ref)
        self.assertIn("torch.nn.functional.layer_norm(view, [cols], weight, bias, eps=1e-5)", ln)
        self.assertIn("torch.nn.functional.softmax(view, dim=1)", sm)
        self.assertRegex(src, r'"--skip-reference-above-mib", type=float, default=None')
        self.assertNotIn("--skip-reference-above-mib\"]", src)
        self.assertNotRegex(src, r"MemoryError|psutil|virtual_memory|total_memory")


if __name__ == "__main__":
    unittest.main(verbosity=1)
