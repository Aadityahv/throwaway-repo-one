#!/usr/bin/env python3
"""CPU tests of fresh set D (cell table, binding, regression gate record, output tables, timing script guards). No GPU."""
from __future__ import annotations

import csv
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fresh_d_lib as L  # noqa: E402
import timing_fresh_d as T  # noqa: E402

SR = L.SR
X = L.X
OUT = Path(os.environ.get("FRESH_D_OUTDIR", str(HERE)))   # env override: test a partial build written elsewhere
CELLS = json.loads((OUT / "fresh_cells_d.json").read_text()) if (OUT / "fresh_cells_d.json").exists() else None
HW, L2 = L.l2_bytes_from_ground_truth()


def need_outputs(test):
    return unittest.skipIf(CELLS is None, "run build_fresh_d.py first")(test)


class CellTable(unittest.TestCase):
    def setUp(self):
        self.cells = L.define_cells(L2)

    def test_counts_and_ids(self):
        self.assertEqual(len(self.cells), 28)
        per_op = {}
        for c in self.cells:
            per_op[c["operator_id"]] = per_op.get(c["operator_id"], 0) + 1
            parts = c["cell_id"].split("/")
            self.assertEqual(parts[0], "blackwell")
            self.assertTrue(parts[1].startswith("fresh_d_"))
            self.assertEqual(parts[2:], [c["regime"], c["candidate_id"]])
        self.assertEqual(per_op, {"fresh_d_cuda_samples_vector_add": 4, "fresh_d_cuda_samples_transpose": 8, "fresh_d_cuda_samples_copy": 4,
                                  "fresh_d_cuda_samples_transposefine": 4, "fresh_d_cuda_samples_reduction": 6, "fresh_d_cuda_samples_reduce2": 2})
        self.assertEqual(len({c["cell_id"] for c in self.cells}), 28)

    def test_ground_truth_l2(self):
        self.assertEqual(L2, 134217728)   # HARDWARE_GROUND_TRUTH.md Blackwell section, read by the extractor's loader

    def test_tier_is_recomputed_from_footprint(self):
        for c in self.cells:
            self.assertEqual(c["tier"], "L2" if c["footprint_bytes"] / L2 < 1 else "DRAM", c["cell_id"])
            self.assertEqual(c["tier"], "L2" if c["regime"] == "l2" else "DRAM", c["cell_id"])
            self.assertAlmostEqual(c["footprint_over_l2"], c["footprint_bytes"] / L2)
        ratios = {c["cell_id"]: c["footprint_over_l2"] for c in self.cells}
        self.assertAlmostEqual(ratios["blackwell/fresh_d_cuda_samples_transpose/l2/c1"], 2 * 1536 * 1536 * 4 / L2)
        self.assertGreater(min(r for c, r in ratios.items() if "/dram/" in c), 1.0 + 1e-3)   # no marginal DRAM cell
        self.assertLess(max(r for c, r in ratios.items() if "/l2/" in c), 0.4)

    def test_geometries_differ_from_development_cells(self):
        """Per origin (development) operator: no fresh cell has the shape, grid or block of a development cell of that operator.
        (reduce6 at n = 8,388,608 shares only the NUMBER with the development vecAdd medium cell, a different kernel and grid.)"""
        dev = X.load_dev_rows()
        dev_cuda = [d for d in dev.values() if "cuda" in d["operator_id"]]
        self.assertEqual(len(dev_cuda), 57)
        by_op = {}
        for d in dev_cuda:
            by_op.setdefault(d["operator_id"], set()).add((X.num(d["shape_a"]), X.num(d["shape_b"]), X.num(d["grid_blocks"]), X.num(d["block_threads"])))
        for c in self.cells:
            mine = (c["shape_a"], c["shape_b"], c["grid_blocks"], c["block_threads"])
            for dev_shape in by_op[c["origin_operator_id"]]:
                self.assertNotEqual(mine[0], dev_shape[0], c["cell_id"])          # the shape never repeats within the family
                self.assertNotEqual(mine[:2] + mine[2:3], dev_shape[:2] + dev_shape[2:3], c["cell_id"])
            if c["kernel_family"] == "tile":
                self.assertNotIn(c["geometry"]["dim_x"], (1024, 2048, 8192))
            # and no fresh grid size equals a development grid size of the same origin operator with the same block shape
            self.assertFalse(any(d[2] == mine[2] and d[3] == mine[3] and d[0] == mine[0] for d in by_op[c["origin_operator_id"]]), c["cell_id"])

    def test_reduce6_uses_the_retained_power_of_two_instantiation(self):
        for c in self.cells:
            n = c["geometry"].get("n")
            if c["kernel"] == "reduce6":
                self.assertEqual(n & (n - 1), 0, "retained reduce6<float,256,true> assumes a power-of-two n")
                self.assertEqual(c["geometry"]["blocks"], 64)
            elif c["kernel_family"] == "reduction":
                self.assertEqual(n % 256, 0)
                self.assertEqual(c["geometry"]["blocks"], n // 256)
                self.assertEqual(c["origin_operator_id"].startswith("alt_") or c["kernel"] == "reduce2", True)

    def test_bytes_rules(self):
        for c in self.cells:
            g = c["geometry"]
            if c["kernel_family"] == "vecadd":
                self.assertEqual(c["logical_bytes_per_launch"], 12 * g["n"])
                self.assertEqual(g["blocks"] * g["threads"], g["n"])
            elif c["kernel_family"] == "tile":
                self.assertEqual(c["logical_bytes_per_launch"], 8 * g["dim_x"] ** 2)
                self.assertEqual(g["grid"], [g["dim_x"] // 32, g["dim_x"] // 32, 1])
            else:
                self.assertEqual(c["logical_bytes_per_launch"], 4 * g["n"] + 4 * g["blocks"])

    def test_every_kernel_has_a_retained_origin_with_matching_symbol(self):
        rows = L.retained_rows()
        for c in self.cells:
            o = rows[(c["origin_operator_id"], c["origin_cell"])]
            self.assertTrue(c["kernel"].lower() in o["isolated_function_section"].lower() or c["kernel"] == "vecAdd" and "vecAdd" in o["isolated_function_section"], c["cell_id"])
        self.assertEqual({L.retained_rows()[(c["origin_operator_id"], c["origin_cell"])]["isolated_function_section"] for c in self.cells}.__len__(), 13)


class Binding(unittest.TestCase):
    def test_fresh_binding_reproduces_frozen_binding_on_all_57_cuda_development_cells(self):
        dev = X.load_dev_rows()
        rows = L.retained_rows()
        n_checked = 0
        for (oid, cell), row in rows.items():
            cid = "blackwell/%s/%s" % (oid, cell)
            spec = L.dev_spec(cid, dev)
            frozen = L.D.binding("cuda", row, L.CUDA_ROOT)               # row has no marker: frozen function
            fresh = L.D.binding("cuda", L.marked_row(row, spec["family"], spec["geometry"]), L.CUDA_ROOT)
            self.assertEqual(frozen, fresh, cid)
            n_checked += 1
        self.assertEqual(n_checked, 57)

    def test_n_is_explicit_not_read_from_the_regime_name(self):
        row = L.retained_rows()[("final_cuda_samples_copy", "small/c1")]
        g = dict(n=3145728, threads=128, blocks=24576, grid=[24576, 1, 1], block=[128, 1, 1])
        const = L.D.binding("cuda", L.marked_row(row, "vecadd", g), L.CUDA_ROOT)[0]
        self.assertEqual(const[0x398].exact_value, 3145728)
        # the frozen function, for the same row, ignores any n and returns the regime's n (the quirk the fresh binding avoids)
        self.assertEqual(L.D.binding("cuda", row, L.CUDA_ROOT)[0][0x398].exact_value, 1048576)

    def test_binding_installed_in_every_loaded_derivation_copy(self):
        for m in (L.D, L.COL, L.D_X, L.COL_X, sys.modules["derive"]):
            self.assertTrue(getattr(m.binding, "_fresh_wrapper", False), m.__name__)

    def test_unmarked_rows_fall_through_and_unknown_corpus_refuses(self):
        row = L.marked_row(L.retained_rows()[("final_cuda_samples_copy", "small/c1")], "vecadd", {})
        with self.assertRaises(L.Refusal):
            L.D.binding("triton", row, L.CUDA_ROOT)

    def test_no_measured_artifact_is_referenced(self):
        for name in ("fresh_d_lib.py", "build_fresh_d.py"):
            text = (HERE / name).read_text()
            for token in ("timing_fresh_result", "score_", "predictions_", "RESULT_", "energy_raw", "runtime_s", "timing_fresh_d_result"):
                self.assertNotIn(token, text.replace("Result", ""), (name, token))
        self.assertEqual(set(X.DEV_COLUMNS_ALLOWED) & {"runtime_s", "energy_j", "power_w", "mean_power_w"}, set())


class GateRecord(unittest.TestCase):
    @need_outputs
    def test_gate_passed_for_every_operator_and_covers_every_kernel(self):
        gate = json.loads((OUT / "gate_fresh_d.json").read_text())
        self.assertEqual(gate["fresh_d_lib_py_sha256"], L.sha(HERE / "fresh_d_lib.py"), "gate was run on a different lib")
        for r in gate["cells"]:
            self.assertTrue(r["all_equal"], r["cell"])
        self.assertTrue(all(v["passed"] for v in gate["operators"].values()))
        gate_symbols = {r["kernel_symbol"] for r in gate["cells"]}
        for c in CELLS["cells"]:
            self.assertIn(c["kernel_symbol"], gate_symbols, c["cell_id"])
        # the frozen refusals are reproduced byte-exactly too
        refused = [r for r in gate["cells"] if r["phase_status"] == "unsupported"]
        self.assertGreaterEqual(len(refused), 2)


class Tables(unittest.TestCase):
    def load(self):
        f = json.loads((OUT / "features_fresh_d.json").read_text())
        p = json.loads((OUT / "phases_fresh_d.json").read_text())
        u = json.loads((OUT / "phases_unique_fresh_d.json").read_text())
        return f, p, u

    @need_outputs
    def test_schemas_equal_the_development_tables(self):
        f, p, u = self.load()
        df = json.loads((SR / "features_blackwell.json").read_text())
        dp = json.loads((SR / "phases_blackwell.json").read_text())
        du = json.loads((SR / "reuse/phases_unique_blackwell.json").read_text())
        self.assertEqual(f["schema"], df["schema"])
        self.assertEqual(set(f), set(df))
        self.assertEqual(set(p), set(dp))
        self.assertEqual(set(u), set(du))
        dev_feat_keys = {tuple(sorted(r)) for r in df["rows"] if r["status"] == "supported" and r["corpus"] == "cuda"}
        self.assertEqual(len(dev_feat_keys), 1)
        for r in f["rows"]:
            if r["status"] == "supported":
                self.assertEqual(tuple(sorted(r)), next(iter(dev_feat_keys)))
        dev_ok = next(r for r in dp["rows"].values() if r["status"] == "conditional_static_phases")
        dev_un = next(r for r in du["rows"].values() if r["status"] == "ok")
        for r in p["rows"].values():
            if r["status"] == "conditional_static_phases":
                self.assertEqual(set(r), set(dev_ok))
                self.assertEqual(set(r["kernels"][0]), set(dev_ok["kernels"][0]))
        for r in u["rows"].values():
            if r["status"] == "ok":
                self.assertEqual(set(r), set(dev_un))
                self.assertEqual(set(r["kernels"][0]), set(dev_un["kernels"][0]))

    @need_outputs
    def test_every_cell_is_in_every_table_and_denominator_is_28(self):
        f, p, u = self.load()
        ids = {c["cell_id"] for c in CELLS["cells"]}
        self.assertEqual(len(ids), 28)
        self.assertEqual({r["cell_id"] for r in f["rows"]}, ids)
        self.assertEqual(set(p["rows"]), ids)
        self.assertEqual(set(u["rows"]), ids)
        for c in CELLS["cells"]:
            ok = (p["rows"][c["cell_id"]]["status"] == "conditional_static_phases" and u["rows"][c["cell_id"]]["status"] == "ok"
                  and next(r for r in f["rows"] if r["cell_id"] == c["cell_id"])["status"] == "supported")
            self.assertEqual(c["derived_fully"], ok, c["cell_id"])
            self.assertEqual(c["in_timing_list"], ok)
            if not ok:   # exact refusal reason, never zero-filled
                self.assertTrue(p["rows"][c["cell_id"]].get("reason"), c["cell_id"])
                self.assertEqual(p["rows"][c["cell_id"]]["kernels"], [])

    @need_outputs
    def test_features_match_cell_table(self):
        f, _, _ = self.load()
        by = {r["cell_id"]: r for r in f["rows"]}
        for c in CELLS["cells"]:
            r = by[c["cell_id"]]
            if r["status"] != "supported":
                continue
            self.assertEqual(r["memory"]["tier"], c["tier"])
            self.assertEqual(r["memory"]["logical_bytes_per_launch"], c["logical_bytes_per_launch"])
            self.assertTrue(r["geometry"]["geometry_matches_dev_table"], c["cell_id"])   # derivation grid == explicit fresh grid
            self.assertEqual(r["geometry"]["grid_blocks_derivation"], c["grid_blocks"])
            self.assertEqual(r["geometry"]["block_threads_derivation"], c["block_threads"])
            self.assertTrue(r["work"]["six_class_cross_check_all_equal"])
            self.assertEqual(r["inputs"]["cubin_sha256"], c["retained_cubin_sha256"])
            self.assertIsNotNone(r["occupancy"]["blocks_per_sm"])

    @need_outputs
    def test_first_touch_totals_equal_phase_totals(self):
        _, p, u = self.load()
        for cid, row in u["rows"].items():
            if row["status"] != "ok":
                continue
            for ph, up in zip(p["rows"][cid]["kernels"][0]["phases"], row["kernels"][0]["phases"]):
                self.assertEqual((ph["read_sectors"], ph["write_sectors"], ph["lines"]), (up["read_sectors"], up["write_sectors"], up["lines"]), cid)
                self.assertLessEqual(up["read_sectors_first_touch"], up["read_sectors"])

    @need_outputs
    def test_sector_totals_scale_like_the_frozen_development_cells(self):
        """Per-element sector counts of fresh tile and vecAdd cells equal those of the frozen development cell of the same kernel."""
        _, p, _ = self.load()
        dp = json.loads((SR / "phases_blackwell.json").read_text())["rows"]
        dev = X.load_dev_rows()

        def per_element(row, n):
            ph = row["kernels"][0]["phases"]
            return (sum(x["read_sectors"] for x in ph) / n, sum(x["write_sectors"] for x in ph) / n)

        for c in CELLS["cells"]:
            if c["kernel_family"] == "reduction" or not c["derived_fully"]:
                continue
            n = c["geometry"]["n"] if c["kernel_family"] == "vecadd" else c["geometry"]["dim_x"] ** 2
            devid = "blackwell/%s/%s" % (c["origin_operator_id"], c["origin_cell"])
            ndev = X.num(dev[devid]["shape_a"]) if c["kernel_family"] == "vecadd" or c["operator_id"].endswith("_copy") else X.num(dev[devid]["shape_a"]) ** 2
            if c["operator_id"].endswith("transposefine"):
                ndev = X.num(dev[devid]["shape_a"]) ** 2
            self.assertEqual(per_element(p["rows"][c["cell_id"]], n), per_element(dp[devid], ndev), c["cell_id"])

    @need_outputs
    def test_closed_form_sector_counts(self):
        _, p, _ = self.load()
        for c in CELLS["cells"]:
            if not c["derived_fully"]:
                continue
            ph = p["rows"][c["cell_id"]]["kernels"][0]["phases"]
            r, w = sum(x["read_sectors"] for x in ph), sum(x["write_sectors"] for x in ph)
            if c["kernel_family"] == "vecadd":
                n = c["geometry"]["n"]
                self.assertEqual((r, w), (n // 4, n // 8), c["cell_id"])          # two input and one output stream, 8 floats per 32 B sector
            elif c["kernel_family"] == "tile":
                n = c["geometry"]["dim_x"] ** 2
                if c["kernel"] in ("copy", "copySharedMem", "transposeCoalesced", "transposeNoBankConflicts", "transposeDiagonal"):
                    self.assertEqual((r, w), (n // 8, n // 8), c["cell_id"])
                if c["kernel"] == "transposeNaive":
                    self.assertEqual((r, w), (n // 8, n), c["cell_id"])           # every transposed store is its own sector
            else:
                n, blocks = c["geometry"]["n"], c["geometry"]["blocks"]
                self.assertEqual(r, n // 8, c["cell_id"])                         # one pass over the input
                self.assertEqual(w, blocks, c["cell_id"])                         # one partial sum per block, one sector each


class TimingScript(unittest.TestCase):
    def test_parse_sass_of_every_retained_kernel_and_identity(self):
        rows = {r["isolated_function_section"]: r for r in json.loads((T.RETENTION / "retention_manifest.json").read_text())["rows"]}
        self.assertEqual(len(rows), 13 + 3)   # 13 kernels of the set plus reduce3/4/5
        for sym, r in rows.items():
            text = (T.RETENTION / r["disassembly_path"]).read_text()
            ok, why = T.sass_identical(text, text)
            self.assertTrue(ok, sym)
            self.assertGreater(len(T.parse_sass(text)), 5)

    def test_sass_mutation_is_detected(self):
        r = next(r for r in json.loads((T.RETENTION / "retention_manifest.json").read_text())["rows"] if r["isolated_function_section"] == "_Z4copyPfS_ii")
        text = (T.RETENTION / r["disassembly_path"]).read_text()
        mutated = text.replace("LDG.E", "LDG.E.64", 1)
        ok, why = T.sass_identical(text, mutated)
        self.assertFalse(ok)
        self.assertIn("differs", why)
        shorter = "\n".join(text.splitlines()[:-6])
        self.assertFalse(T.sass_identical(text, shorter)[0])

    def test_window_selection_and_median(self):
        repeat, per_launch = T.window_launches_for(0.010)      # 1000 launches took 10 ms -> 10 us per launch
        self.assertEqual(repeat, 10_000)
        self.assertAlmostEqual(per_launch, 1e-5)
        self.assertEqual(T.window_launches_for(0.5)[0], 1000)   # a 0.5 s replay is already longer than one window: one replay
        self.assertEqual(T.window_launches_for(0.0001)[0] % 1000, 0)
        self.assertEqual(T.median([3, 1, 2]), 2)
        self.assertEqual(T.median([4, 1, 2, 3, 10]), 3)

    def test_windows_csv_parse_and_launch_check(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "windows.csv"
            p.write_text("block,launches,host_begin_monotonic_ns,host_end_monotonic_ns,cuda_seconds\n1,5000,1,2,0.123456789\n")
            self.assertAlmostEqual(T.parse_windows_csv(p, 5000), 0.123456789)
            with self.assertRaises(RuntimeError):
                T.parse_windows_csv(p, 4000)

    def test_driver_argv_matches_the_runners(self):
        spec = {"numeric_args": [3145728, 128]}
        self.assertEqual(T.driver_argv("/b", spec, ["x", "y", "o"], 10000, "/t"),
                         ["/b", "3145728", "128", "x", "y", "o", "--repeat", "10000", "--graph-batch", "1000", "--trace-dir", "/t"])
        self.assertEqual(T.driver_argv("/b", {"numeric_args": [5120, 5120, 7]}, ["i", "o"]), ["/b", "5120", "5120", "7", "i", "o"])

    @need_outputs
    def test_timing_specs_match_runner_conventions(self):
        for c in CELLS["cells"]:
            t = c["timing"]
            if t["runner"] == "reduction_runner":
                self.assertEqual(len(t["numeric_args"]), 4)
                self.assertEqual(t["numeric_args"][1], 256)
            elif t["runner"] == "copy_runner":
                self.assertEqual(t["numeric_args"][0] % t["numeric_args"][1], 0)
            else:
                self.assertEqual(t["numeric_args"][:2], [c["geometry"]["dim_x"]] * 2)
                self.assertIn(t["numeric_args"][2], (0, 1, 2, 3, 4, 5, 6, 7))

    @need_outputs
    def test_dry_run_lists_timing_cells_without_gpu(self):
        out = subprocess.run([sys.executable, str(HERE / "timing_fresh_d.py"), "--dry-run"], capture_output=True, text=True, timeout=60, env=dict(os.environ, FRESH_D_CELLS_JSON=str(OUT / "fresh_cells_d.json")))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(sum(1 for ln in out.stdout.splitlines() if ln.startswith("blackwell/")), sum(c["in_timing_list"] for c in CELLS["cells"]))
        self.assertIn("hash gate", out.stdout)

    @need_outputs
    def test_refuses_existing_output_partial_and_missing_booking(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", FRESH_D_CELLS_JSON=str(OUT / "fresh_cells_d.json"))
        with tempfile.TemporaryDirectory() as d:
            existing = Path(d) / "r.json"
            existing.write_text("{}")
            r = subprocess.run([sys.executable, str(HERE / "timing_fresh_d.py"), "--out", str(existing), "--booking-ref", "x"], capture_output=True, text=True, env=env)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("exists", r.stderr)
            new = Path(d) / "n.json"
            Path(str(new) + ".partial").write_text("{}")
            r = subprocess.run([sys.executable, str(HERE / "timing_fresh_d.py"), "--out", str(new), "--booking-ref", "x"], capture_output=True, text=True, env=env)
            self.assertIn("partial", r.stderr)
            Path(str(new) + ".partial").unlink()
            r = subprocess.run([sys.executable, str(HERE / "timing_fresh_d.py"), "--out", str(new)], capture_output=True, text=True, env=env)
            self.assertIn("booking-ref", r.stderr)

    def test_uuid_normalisation(self):
        self.assertEqual(T.norm_uuid("GPU-0E63BAEA-0bb1-e90c-f19d-8aa5f2995894"), T.norm_uuid(T.EXPECTED_UUID))

    def _fake_toolchain(self, d, version_line, sass_for):
        nvcc = Path(d) / "nvcc"
        nvcc.write_text("#!/bin/sh\necho '%s'\n" % version_line)
        cuo = Path(d) / "cuobjdump"
        cuo.write_text("#!/bin/sh\nif [ \"$1\" = \"-sass\" ]; then cat '%s'; fi\n" % sass_for)
        for p in (nvcc, cuo):
            p.chmod(p.stat().st_mode | stat.S_IEXEC)
        return nvcc

    def _cell(self):
        row = next(r for r in json.loads((T.RETENTION / "retention_manifest.json").read_text())["rows"]
                   if r["operator_id"] == "alt_cuda_samples_copy" and r["cell"] == "small/c1")
        return {"cell_id": "blackwell/x/l2/c1", "origin_operator_id": row["operator_id"], "origin_cell": row["cell"],
                "retained_cubin_sha256": row["cubin_sha256"], "kernel_symbol": row["isolated_function_section"], "timing": {"runner": "tile_family"}}, row

    def test_hash_gate_passes_for_identical_sass_and_refuses_otherwise(self):
        cell, row = self._cell()
        good = T.RETENTION / row["disassembly_path"]
        with tempfile.TemporaryDirectory() as d:
            nvcc = self._fake_toolchain(d, T.RETAINED_NVCC_RELEASE, good)
            rec = T.hash_gate([cell], {"tile_family": Path(d) / "bin"}, str(nvcc), d)
            self.assertTrue(rec[0]["sass_equal"])
            bad = Path(d) / "bad.sass"
            bad.write_text(good.read_text().replace("LDG.E", "LDG.E.64", 1))
            nvcc = self._fake_toolchain(d, T.RETAINED_NVCC_RELEASE, bad)
            with self.assertRaises(SystemExit) as cm:
                T.hash_gate([cell], {"tile_family": Path(d) / "bin"}, str(nvcc), d)
            self.assertIn("What must be built", str(cm.exception))
            nvcc = self._fake_toolchain(d, "Cuda compilation tools, release 12.8, V12.8.61", good)
            with self.assertRaises(SystemExit) as cm:
                T.hash_gate([cell], {"tile_family": Path(d) / "bin"}, str(nvcc), d)
            self.assertIn("13.2.78", str(cm.exception))

    def test_retained_row_refuses_disagreeing_cell_table(self):
        cell, _ = self._cell()
        cell["retained_cubin_sha256"] = "0" * 64
        with self.assertRaises(SystemExit):
            T.retained_row_for(cell)


if __name__ == "__main__":
    unittest.main(verbosity=2)
