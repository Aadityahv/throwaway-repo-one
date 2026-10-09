#!/usr/bin/env python3
"""Tests for extract_features.py (CPU only).

Run from the repository root:
    python3 tiresias/framework/predictor/test_extract_features.py
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extract_features as X  # noqa: E402

SYNTH_GT = """
## Blackwell (synthetic copy for tests)

| Field | Value | Verified via |
|---|---|---|
| SM count | 188 | probe |
| L2 cache size | 134,217,728 bytes (128.0 MiB) | probe |
| Warp size | 32 threads | probe |
| Max threads per block / per SM | 1,024 / 1,536 | probe |
| Registers per block / per SM | 65,536 / 65,536 | probe |
| Shared mem per block | 49,152 bytes | probe |
| Shared mem per SM | 102,400 bytes | probe |
| Max resident blocks per SM | 24 | probe |
| Shared-memory allocation granularity | 128 bytes | header |
| Register allocation granularity | 256 (registers/warp) | header |
| Max registers per thread | 256 | header |
| Reserved shared memory per block (driver carveout) | 1,024 bytes | live |
| Max warps per SM | 48 | arithmetic |
| Register sub-partitions per SM | 4 | header |

## Next section

| Field | Value |
|---|---|
| SM count | 999 |
"""


def hw():
    return X.load_hardware(SYNTH_GT)


class HardwareParsing(unittest.TestCase):
    def test_parse_synthetic(self):
        h = hw()
        self.assertEqual(h["sm_count"], 188)
        self.assertEqual(h["l2_bytes"], 134217728)
        self.assertEqual((h["max_threads_per_block"], h["max_threads_per_sm"]), (1024, 1536))
        self.assertEqual(h["shared_per_sm_bytes"], 102400)
        self.assertEqual(h["reserved_shared_per_block_bytes"], 1024)
        self.assertTrue(all(v is not None for v in h.values()))

    def test_stops_at_next_section(self):
        self.assertEqual(hw()["sm_count"], 188)

    def test_missing_row_is_none_not_guessed(self):
        text = SYNTH_GT.replace("| Register allocation granularity | 256 (registers/warp) | header |\n", "")
        h = X.load_hardware(text)
        self.assertIsNone(h["register_alloc_granularity_regs_per_warp"])
        self.assertEqual(h["sm_count"], 188)

    def test_live_ground_truth_has_every_constant_the_occupancy_model_uses(self):
        h = X.load_hardware((X.GROUND_TRUTH).read_text())
        self.assertEqual([k for k, v in h.items() if v is None], [])


class OccupancyArithmetic(unittest.TestCase):
    def test_thread_limited_example(self):
        # 32 regs, 256 threads (8 warps): regs/warp=1024, per block 8192 -> 8 blocks;
        # threads 1536/256=6, warps 48/8=6, shared alloc 1024 -> 100, max blocks 24 -> 6.
        o = X.occupancy(hw(), 32, 256, 1000, 0, 0)
        self.assertEqual(o["blocks_per_sm"], 6)
        self.assertEqual(o["limiter"], ["threads", "warps"])
        self.assertEqual(o["max_resident_warps_per_sm"], 48)
        self.assertEqual(o["occupancy_fraction_of_max_warps"], 1.0)
        self.assertEqual(o["waves"], 1)  # 1000 <= 188*6 = 1128
        self.assertAlmostEqual(o["last_wave_fill_fraction"], 1000 / 1128)
        self.assertAlmostEqual(o["active_sm_fraction"], 1.0)

    def test_two_waves_and_tail(self):
        o = X.occupancy(hw(), 32, 256, 2000, 0, 0)
        self.assertEqual(o["waves"], 2)
        self.assertAlmostEqual(o["last_wave_fill_fraction"], (2000 - 1128) / 1128)
        self.assertAlmostEqual(o["tail_idle_fraction_of_last_wave"], 1 - 872 / 1128)

    def test_register_limited_example_with_dynamic_shared(self):
        # 64 regs, 512 threads (16 warps): regs/warp=2048, block=32768 -> 2 blocks (registers).
        # shared: (0 + 20000 + 1024) = 21024 -> round up to 128 -> 21120; 102400//21120 = 4.
        o = X.occupancy(hw(), 64, 512, 5000, 0, 20000)
        self.assertEqual(o["shared_allocated_bytes_per_block"], 21120)
        self.assertEqual(o["limit_by_shared"], 4)
        self.assertEqual(o["blocks_per_sm"], 2)
        self.assertEqual(o["limiter"], ["registers"])
        self.assertEqual(o["waves"], -(-5000 // (188 * 2)))

    def test_shared_limited_example(self):
        # 16 regs, 64 threads: regs fine; shared 40000 static -> (40000+1024)=41024 -> 41088; 102400//41088 = 2.
        o = X.occupancy(hw(), 16, 64, 100, 40000, 0)
        self.assertEqual(o["limit_by_shared"], 2)
        self.assertEqual(o["blocks_per_sm"], 2)
        self.assertEqual(o["limiter"], ["shared"])
        self.assertAlmostEqual(o["active_sm_fraction"], 100 / 188)
        self.assertEqual(o["waves"], 1)

    def test_register_limit_follows_cuda_occupancy_header(self):
        # 128 regs, 64 threads (2 warps): 4096 regs/warp; 16384/4096 = 4 warps per sub-partition,
        # x4 sub-partitions = 16 warps/SM -> 8 blocks (cudaOccMaxBlocksPerSMRegsLimit).
        o = X.occupancy(hw(), 128, 64, 1000, 0, 0)
        self.assertEqual(o["limit_by_registers_if_multiple_1"], 8)
        self.assertEqual(o["blocks_per_sm"], 8)
        # 40 regs -> 1280 regs/warp; 16384//1280 = 12 warps per sub-partition -> 48 warps/SM.
        o = X.occupancy(hw(), 40, 32, 1000, 0, 0)
        self.assertEqual(o["limit_by_registers_if_multiple_1"], 48)

    def test_unknown_dynamic_shared_gives_null_with_nonbinding_bound(self):
        o = X.occupancy(hw(), 32, 256, 1000, 0, None)
        self.assertIsNone(o["blocks_per_sm"])
        self.assertIn("dynamic shared", o["reasons"][0])
        # 6 blocks -> 102400//6 = 17066 -> floor to 128 multiple = 133*128 = 17024; minus 1024 reserve = 16000.
        self.assertEqual(o["shared_nonbinding_up_to_dynamic_bytes"], 16000)

    def test_missing_ground_truth_constant_refuses(self):
        h = hw()
        h["register_alloc_granularity_regs_per_warp"] = None
        o = X.occupancy(h, 32, 256, 1000, 0, 0)
        self.assertIsNone(o["blocks_per_sm"])
        self.assertIn("register_alloc_granularity_regs_per_warp", o["reasons"][0])
        self.assertNotIn("waves", o)

    def test_missing_register_count_refuses(self):
        o = X.occupancy(hw(), None, 256, 1000, 0, 0)
        self.assertIsNone(o["blocks_per_sm"])
        self.assertIn("registers_per_thread", o["reasons"][0])

    def test_over_limit_launch_refuses(self):
        o = X.occupancy(hw(), 32, 2048, 10, 0, 0)
        self.assertIsNone(o["blocks_per_sm"])


class ElfParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D, cls.C = X.load_derivation()

    def _cubin(self, corpus, operator_id, cell):
        root = cls_root = self.D.CORPORA[corpus]
        m = json.loads((root / "retention_manifest.json").read_text())
        row = next(r for r in m["rows"] if r["operator_id"] == operator_id and r["cell"] == cell)
        data = (root / row["cubin_path"]).read_bytes()
        self.assertEqual(X.sha_bytes(data), row["cubin_sha256"])
        return data, row

    def test_triton_vector_add_registers_match_prior_cuobjdump_record(self):
        # 03_triton_vector_add.txt (cuobjdump on the Triton cache cubins): REG 14/16/22/28 for
        # BLOCK_SIZE 128/256/512/1024, shared 0, 128 threads (num_warps=4).
        for cand, reg in (("c1", 14), ("c2", 16), ("c3", 22), ("c4", 28)):
            data, row = self._cubin("triton", "train_triton_vector_add", "large/" + cand)
            r = X.parse_cubin_resources(data, row["isolated_function_section"], 1024)
            self.assertEqual(r["registers_per_thread"], reg, cand)
            self.assertEqual(r["static_shared_bytes_per_block"], 0)
            self.assertEqual(r["reqntid"], [128, 1, 1])
            self.assertEqual(r["local_memory_bytes_per_thread"], 0)

    def test_triton_softmax_registers_and_reserve_only_shared_section(self):
        # 04_triton_softmax.txt: small c1 REG=22; cuobjdump SHARED:1024 == reserve only (dynamic kernel).
        data, row = self._cubin("triton", "dev_triton_softmax", "small/c1")
        r = X.parse_cubin_resources(data, row["isolated_function_section"], 1024)
        self.assertEqual(r["registers_per_thread"], 22)
        self.assertEqual(r["shared_section_bytes_raw"], 1024)
        self.assertEqual(r["static_shared_bytes_per_block"], 0)
        data, row = self._cubin("triton", "dev_triton_softmax", "large/c1")
        self.assertEqual(X.parse_cubin_resources(data, row["isolated_function_section"], 1024)["registers_per_thread"], 37)

    def test_cuda_transpose_static_shared_subtracts_reserve(self):
        # transposeCoalesced declares tile[32][32] floats = 4096 B; section is 5120 B (4096 + 1024 reserve).
        data, row = self._cubin("cuda", "train_cuda_samples_transpose", "large/c2")
        self.assertEqual(row["isolated_function_section"], "_Z18transposeCoalescedPfS_ii")
        r = X.parse_cubin_resources(data, row["isolated_function_section"], 1024)
        self.assertEqual(r["shared_section_bytes_raw"], 5120)
        self.assertEqual(r["static_shared_bytes_per_block"], 4096)
        self.assertEqual(r["registers_per_thread"], 18)
        # transposeNoBankConflicts: tile[32][33] = 4224 B.
        data, row = self._cubin("cuda", "train_cuda_samples_transpose", "large/c3")
        r = X.parse_cubin_resources(data, row["isolated_function_section"], 1024)
        self.assertEqual(r["static_shared_bytes_per_block"], 4224)
        # no shared section at all -> 0
        data, row = self._cubin("cuda", "train_cuda_samples_transpose", "large/c1")
        r = X.parse_cubin_resources(data, row["isolated_function_section"], 1024)
        self.assertEqual(r["shared_section_bytes_raw"], None)
        self.assertEqual(r["static_shared_bytes_per_block"], 0)

    def test_multi_kernel_cubin_selects_the_requested_symbol(self):
        data, row = self._cubin("cuda", "train_cuda_samples_transpose", "large/c4")
        self.assertEqual(X.parse_cubin_resources(data, "_Z17transposeDiagonalPfS_ii", 1024)["registers_per_thread"], 22)
        self.assertEqual(X.parse_cubin_resources(data, "_Z14transposeNaivePfS_ii", 1024)["registers_per_thread"], 16)

    def test_reserve_unknown_gives_null_static_shared(self):
        data, row = self._cubin("cuda", "train_cuda_samples_transpose", "large/c2")
        r = X.parse_cubin_resources(data, row["isolated_function_section"], None)
        self.assertIsNone(r["static_shared_bytes_per_block"])

    def test_bad_inputs_raise(self):
        with self.assertRaises(X.ElfError):
            X.parse_cubin_resources(b"not an elf at all" * 10, "x", 1024)
        data, row = self._cubin("triton", "dev_triton_softmax", "small/c1")
        with self.assertRaises(X.ElfError):
            X.parse_cubin_resources(data, "no_such_kernel", 1024)


class ChainHeuristic(unittest.TestCase):
    def test_load_use_chain(self):
        D, _ = X.load_derivation()
        text = """
        /*0000*/ LDG.E R4, desc[UR4][R2.64] ;
        /*0010*/ FADD R5, R4, 1 ;
        /*0020*/ FMUL R6, R5, R5 ;
        /*0030*/ STG.E desc[UR4][R8.64], R6 ;
        /*0040*/ LDG.E R7, desc[UR4][R10.64] ;
        /*0050*/ EXIT ;
        """
        sites = D.parse(text)
        m = X.chain_metrics(sites)
        self.assertEqual(m["global_loads"], 2)
        self.assertEqual(m["global_loads_with_a_consumer"], 1)
        self.assertEqual(m["global_load_to_use_edges"], 1)
        self.assertEqual(m["critical_path_instructions"], 4)  # LDG -> FADD -> FMUL -> STG
        self.assertEqual(m["dependent_global_load_depth"], 1)


class FullTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X.READ_HASHES.clear()
        cls.hw = X.load_hardware(X.GROUND_TRUTH.read_text())
        cls.rows = X.build_rows(cls.hw)

    def test_denominator_is_132(self):
        self.assertEqual(len(self.rows), 132)
        self.assertEqual(len({r["cell_id"] for r in self.rows}), 132)
        st = {}
        for r in self.rows:
            st[r["status"]] = st.get(r["status"], 0) + 1
        # 27 PyTorch cells now carry features from the libtorch sm_120 SASS (pytorch_features/); only the 12
        # xFormers cells (a Triton JIT kernel with no byte-verified compile) remain without a retained binary.
        self.assertEqual(st.get("no_retained_binary"), 12)
        self.assertEqual(sum(1 for r in self.rows if r.get("corpus") == "libtorch_sm120"), 27)
        self.assertEqual(sum(1 for r in self.rows if r["status"] in ("supported", "missing_features") and r.get("corpus") != "libtorch_sm120"), 93)

    def test_no_measured_column_is_whitelisted(self):
        for c in X.DEV_COLUMNS_ALLOWED:
            for bad in ("runtime", "energy", "power", "pred", "epsilon", "calibration"):
                self.assertNotIn(bad, c)

    def test_frozen_class_counts_reproduced_for_all_93(self):
        retained = [r for r in self.rows if r["status"] != "no_retained_binary" and r.get("corpus") != "libtorch_sm120"]
        self.assertEqual(len(retained), 93)
        for r in retained:
            self.assertTrue(r["work"]["six_class_cross_check_all_equal"], r["cell_id"])

    def test_unretained_rows_carry_only_geometry_and_bytes(self):
        for r in self.rows:
            if r["status"] == "no_retained_binary":
                self.assertEqual(set(r) - {"cell_id", "operator_id", "cell", "corpus", "status", "missing_features", "geometry", "memory"}, set())
                self.assertIn("logical_bytes_per_launch", r["memory"])

    def test_unknowns_are_null_with_reason_never_zero(self):
        for r in self.rows:
            if r["status"] == "no_retained_binary":
                continue
            if r["resources"]["dynamic_shared_bytes_per_launch"] is None:
                self.assertTrue(any(m["feature"] == "dynamic_shared_bytes_per_launch" for m in r["missing_features"]))
                self.assertIsNone(r["occupancy"]["blocks_per_sm"])

    def test_register_count_not_below_sass_use(self):
        for r in self.rows:
            if r["status"] != "no_retained_binary":
                self.assertTrue(r["resources"]["register_count_consistent_with_sass"], r["cell_id"])

    def test_output_file_matches_fresh_build(self):
        path = X.OUT_JSON
        if not path.exists():
            self.skipTest("features_blackwell.json not generated yet")
        on_disk = json.loads(path.read_text())
        fresh = json.loads(json.dumps(self.rows, sort_keys=True))
        self.assertEqual(on_disk["rows"], fresh)
        self.assertEqual(on_disk["extract_features_py_sha256"], X.sha_bytes(Path(X.__file__).read_bytes()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
