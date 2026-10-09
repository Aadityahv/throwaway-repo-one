#!/usr/bin/env python3
"""Tests for the PyTorch static-feature adapter and its extended interpreter (CPU only).

Run from the repository root:
    python3 tiresias/framework/predictor/pytorch_features/test_pytorch_features.py
"""
from __future__ import annotations

import collections
import json
import random
import struct
import sys
import unittest
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATIC = HERE.parent
sys.path.insert(0, str(STATIC))
sys.path.insert(0, str(HERE))
import extract_features as X  # noqa: E402
import adapter as A  # noqa: E402
import pt_interp as P  # noqa: E402

D, _C = X.load_derivation()
_ROWS: dict = {}


def get_rows():
    """Build the 27 rows once per test process."""
    if not _ROWS:
        X.READ_HASHES.clear()
        _ROWS.update(A.build_rows(X, X.load_hardware(X.GROUND_TRUTH.read_text()), X.load_dev_rows()))
    return _ROWS


def f32bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


class Binary32(unittest.TestCase):
    def test_round_matches_hardware_float(self):
        rnd = random.Random(7)
        # reference: Python double -> float rounds-to-nearest-even; a Fraction of a double is exact in both
        for _ in range(5000):
            d = rnd.uniform(-1e6, 1e6) * 10 ** rnd.randint(-20, 20)
            if d == 0 or abs(d) < 1e-37 or abs(d) > 1e37:
                continue
            self.assertEqual(P.f32_round(Fraction(d)), f32bits(d), d)

    def test_directed_rounding(self):
        third = Fraction(1, 3)
        rn, rp = P.f32_round(third), P.f32_round(third, "rp")
        self.assertEqual(rp, rn + 1 if P.f32_value(rn) < third else rn)
        self.assertEqual(P.f32_round(Fraction(1), "rp"), 0x3F800000)
        self.assertEqual(P.f32_round(Fraction(0)), 0)

    def test_denormal_and_overflow_are_unknown_not_zero(self):
        self.assertIsNone(P.f32_round(Fraction(1, 2 ** 130)))
        self.assertIsNone(P.f32_round(Fraction(2 ** 130)))
        self.assertIsNone(P.f32_value(0x00000001))
        self.assertIsNone(P.f32_value(0x7F800000))


class InterpreterFidelity(unittest.TestCase):
    """With ext=False the copy must reproduce derive.run_lane exactly; with ext=True it must give the same
    six-class events wherever the frozen derivation succeeds."""

    def test_copy_matches_frozen_interpreter_on_retained_cells(self):
        checked = 0
        for corpus, root in D.CORPORA.items():
            rows = json.loads((root / "retention_manifest.json").read_text())["rows"]
            for row in rows[::5]:
                consts, coords, threads, blocks, launch = D.binding(corpus, row, root)
                text = (root / row["disassembly_path"]).read_text()
                ds, ps = D.parse(text), P.parse(text)

                def lane_ctx(mod, l):
                    return {"SR_TID.X": mod.V.exact(l % launch["block"][0]), "SR_TID.Y": mod.V.exact(l // launch["block"][0]),
                            "SR_TID.Z": mod.V.exact(0), "SR_LANEID": mod.V.exact(l % 32)}
                dt = collections.Counter()
                try:
                    for l in range(threads):
                        e, _v, _u = D.run_lane(ds, consts, {**coords, **lane_ctx(D, l)})
                        dt.update(e)
                except D.Refusal:
                    continue
                pconsts = {o: P.V(v.lo, v.hi) for o, v in consts.items()}
                pcoords = {k: P.V(v.lo, v.hi) for k, v in coords.items()}
                for ext in (False, True):
                    I = P.Interp(D, ext=ext)
                    tot = collections.Counter()
                    for l in range(threads):
                        e, _v, _u = I.run_lane(ps, pconsts, {**pcoords, **lane_ctx(P, l)})
                        tot.update(e)
                    self.assertEqual(tot, dt, (row["operator_id"], row["cell"], ext))
                checked += 1
        self.assertGreater(checked, 12)

    def test_unsupported_opcode_still_refuses_in_compat_mode(self):
        sites = P.parse("        /*0000*/  FCHK P0, R1, R2 ;\n        /*0010*/  EXIT ;")
        with self.assertRaises(P.Refusal):
            P.Interp(D, ext=False).run_lane(sites, {}, {})

    def test_fchk_refuses_unless_assumption_enabled_and_logs_when_enabled(self):
        text = "        /*0000*/  FCHK P0, R1, R2 ;\n        /*0010*/ @!P0 BRA 0x30 ;\n        /*0020*/  EXIT ;\n        /*0030*/  EXIT ;"
        sites = P.parse(text)
        with self.assertRaises(P.Refusal):
            P.Interp(D, ext=True).run_lane(sites, {}, {})
        I = P.Interp(D, ext=True, fchk_fast_path=True)
        I.run_lane(sites, {}, {})
        self.assertEqual(dict(I.assumption_log), {("fchk_fast_path", 0): 1})

    def test_unknown_guard_on_branch_and_exit_refuses(self):
        sites = P.parse("        /*0000*/  LDG.E R1, desc[UR4][R2.64] ;\n        /*0010*/  ISETP.GT.AND P0, PT, R1, 0x0, PT ;\n"
                        "        /*0020*/ @P0 EXIT ;\n        /*0030*/  EXIT ;")
        with self.assertRaises(P.Refusal):
            P.Interp(D, ext=True).run_lane(sites, {}, {})

    def test_interval_negation_is_decided_over_the_block_index(self):
        # R18 = batch - 4*ctaid for ctaid in [0, 127] with batch = 1024: always >= 1
        text = ("        /*0000*/  S2R R0, SR_CTAID.X ;\n        /*0010*/  LDCU UR8, c[0x0][0x390] ;\n"
                "        /*0020*/  IMAD R13, R0, 0x4, RZ ;\n        /*0030*/  IADD3 R18, PT, PT, -R13, UR8, RZ ;\n"
                "        /*0040*/  ISETP.GE.AND P0, PT, R18, 0x1, PT ;\n        /*0050*/ @!P0 EXIT ;\n        /*0060*/  EXIT ;")
        sites = P.parse(text)
        I = P.Interp(D, ext=True)
        e, v, u = I.run_lane(sites, {0x390: P.V.exact(1024)}, {"SR_CTAID.X": P.V(0, 127)})
        self.assertEqual(v[0x60], 1)
        self.assertNotIn(0x50, v)       # the `@!P0 EXIT` guard was decided False over the whole interval
        with self.assertRaises(P.Refusal):   # batch too small: not decidable over the interval
            P.Interp(D, ext=True).run_lane(sites, {0x390: P.V.exact(300)}, {"SR_CTAID.X": P.V(0, 127)})

    def test_or_of_thread_id_and_shifted_block_index(self):
        text = ("        /*0000*/  S2R R3, SR_TID.X ;\n        /*0010*/  S2UR UR4, SR_CTAID.X ;\n"
                "        /*0020*/  USHF.L.U32 UR4, UR4, 0x8, URZ ;\n        /*0030*/  LOP3.LUT R3, R3, UR4, RZ, 0xfc, !PT ;\n"
                "        /*0040*/  ISETP.GE.AND P0, PT, R3, 0x4000, PT ;\n        /*0050*/ @P0 EXIT ;\n        /*0060*/  EXIT ;")
        sites = P.parse(text)
        coords = {"SR_CTAID.X": P.V(0, 63), "SR_TID.X": P.V.exact(77)}
        e, v, u = P.Interp(D, ext=True).run_lane(sites, {}, coords)
        self.assertNotIn(0x50, v)       # max index 63*256+77 < 0x4000: guard decided False
        self.assertEqual(v[0x60], 1)


class ParameterAbi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sites = {k: P.parse((A.SASS_DIR / (k + ".isolated.sass")).read_text()) for k in ("k1", "k2", "k3", "k4", "k5", "k6")}

    def test_every_constant_bank_access_lands_on_a_declared_field(self):
        for k, s in self.sites.items():
            ok, problems = A.abi_check(k, s)
            self.assertTrue(ok, (k, problems))

    def test_offsets_follow_natural_alignment(self):
        off = {n: o for n, o, _ in A.abi_layout("k2")}
        self.assertEqual((off["out"], off["inp"], off["idx"], off["num_ind"], off["slice_size"], off["allow_neg_indices"]),
                         (0x380, 0x388, 0x390, 0x398, 0x3A0, 0x3C0))
        off = {n: o for n, o, _ in A.abi_layout("k4")}
        self.assertEqual((off["batch_size"], off["stride"], off["element_count"], off["mask"], off["head_chunk_size"]),
                         (0x390, 0x394, 0x398, 0x3A0, 0x3A8))
        off = {n: o for n, o, _ in A.abi_layout("k3")}
        self.assertEqual((off["N"], off["eps"], off["gamma"], off["beta"], off["Y"]), (0x380, 0x384, 0x390, 0x398, 0x3B0))
        off = {n: o for n, o, _ in A.abi_layout("k1")}
        self.assertEqual((off["N"], off["dims"], off["sizes_0.divisor"], off["strides_0_0"], off["data_0"]),
                         (0x380, 0x388, 0x38C, 0x4B8, 0x580))

    def test_wrong_layout_is_detected(self):
        saved = A.PARAM_FIELDS["k3"]
        try:
            A.PARAM_FIELDS["k3"] = [("N", "i32"), ("X", "ptr"), ("gamma", "ptr")]   # drops eps, shifts pointers
            ok, problems = A.abi_check("k3", self.sites["k3"])
            self.assertFalse(ok)
            self.assertTrue(problems)
        finally:
            A.PARAM_FIELDS["k3"] = saved


class SharedMemoryDecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sites = {k: P.parse((A.SASS_DIR / (k + ".isolated.sass")).read_text()) for k in ("k1", "k2", "k3", "k4", "k5", "k6")}

    def test_softmax_and_copy_kernels_contain_no_shared_memory_instruction(self):
        for k in ("k1", "k2", "k4", "k5", "k6"):
            ops = {s.op.split(".")[0] for s in self.sites[k]}
            self.assertFalse(ops & {"LDS", "STS", "LDSM", "ATOMS", "REDS", "LDGSTS", "ATOMG"}, k)

    def test_layer_norm_shared_addresses_stay_inside_reserve_plus_dynamic(self):
        # dynamic 24 B starts right after the 1,024 B reserve: addresses 0x400..0x417 only
        rows = get_rows()
        r = rows["blackwell/final_pytorch_layer_norm/large/c1"]
        lo, hi = r["main_kernel"]["interpreter_outcomes"]["extended_interpreter"]["shared_addresses_touched_min_max"]
        self.assertEqual(lo, 1024)
        self.assertLess(hi, 1024 + 24)

    def test_decision_does_not_change_any_occupancy(self):
        for r in get_rows().values():
            for b in [r] + r["secondary_kernels"]:
                self.assertFalse(b["occupancy"]["shared_decision_changes_occupancy"], r["cell_id"])
                self.assertEqual(b["resources"]["static_shared_bytes_per_block"], 0)


class TestRows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = get_rows()
        cls.disp = {c["cell_id"]: c for c in json.loads(DISPATCH.read_text())["cells"]}

    def test_all_27_cells_are_built_and_supported_with_assumptions(self):
        self.assertEqual(len(self.rows), 27)
        self.assertEqual({r["status"] for r in self.rows.values()}, {"supported_with_assumptions"})
        for r in self.rows.values():
            self.assertEqual(r["missing_features"], [])
            self.assertTrue(r["assumptions"])

    def test_geometry_and_kernel_count_come_from_the_dispatch_trace(self):
        for cid, r in self.rows.items():
            kernels = self.disp[cid]["kernels"]
            self.assertEqual(r["kernels_per_launch"], len(kernels), cid)
            self.assertEqual(len(r["secondary_kernels"]), len(kernels) - 1)
            main = kernels[-1]
            self.assertEqual(r["geometry"]["grid_blocks_dispatch"], main["grid"][0] * main["grid"][1] * main["grid"][2])
            self.assertEqual(r["geometry"]["block_threads_dispatch"], main["block"][0] * main["block"][1] * main["block"][2])
            for sec, k in zip(r["secondary_kernels"], kernels[:-1]):
                self.assertEqual(sec["geometry"]["grid"], k["grid"])
                self.assertEqual(sec["geometry"]["block"], k["block"])
                self.assertEqual(sec["role"], "copy")

    def test_copy_kernel_only_for_padded_candidates(self):
        for cid, r in self.rows.items():
            cand = cid.rsplit("/", 1)[1]
            if "embedding" in cid:
                self.assertEqual(r["kernels_per_launch"], 1)
            else:
                self.assertEqual(r["kernels_per_launch"], 1 if cand == "c1" else 2, cid)

    def test_main_kernel_is_the_one_of_the_schema_and_totals_are_sums(self):
        for cid, r in self.rows.items():
            tot = r["per_launch_totals"]
            parts = [r["work"]] + [s["work"] for s in r["secondary_kernels"]]
            self.assertEqual(tot["kernels_per_launch"], r["kernels_per_launch"])
            self.assertEqual(tot["total_lane_instructions"], sum(p["total_lane_instructions"] for p in parts))
            for c in A.SIX_CLASSES:
                self.assertEqual(tot["six_class_counts"][c], sum(p["six_class_counts"][c] for p in parts))
            self.assertEqual(sum(tot["opcode_lane_counts"].values()), tot["total_lane_instructions"])
            for fam, v in tot["families"].items():
                self.assertEqual(v["lane_instructions"], sum(p["families"].get(fam, {}).get("lane_instructions", 0) for p in parts))

    def test_occupancy_stays_per_kernel(self):
        for cid, r in self.rows.items():
            self.assertIn("waves", r["occupancy"])
            for s in r["secondary_kernels"]:
                self.assertIn("waves", s["occupancy"])
                self.assertNotEqual(s["occupancy"], r["occupancy"])
            self.assertNotIn("occupancy", r["per_launch_totals"])

    def test_main_kernel_register_counts_match_cuobjdump(self):
        want = {"k1": 20, "k2": 24, "k3": 44, "k4": 72, "k5": 34, "k6": 48}
        for cid, r in self.rows.items():
            self.assertEqual(r["resources"]["registers_per_thread"], want[r["main_kernel"]["kernel_id"]], cid)
            self.assertTrue(r["resources"]["register_count_consistent_with_sass"], cid)
            for s in r["secondary_kernels"]:
                self.assertEqual(s["resources"]["registers_per_thread"], 20)

    def test_every_kernel_records_the_frozen_interpreter_refusal(self):
        for r in self.rows.values():
            for io in [r["main_kernel"]["interpreter_outcomes"]] + [s["interpreter_outcomes"] for s in r["secondary_kernels"]]:
                self.assertEqual(io["frozen_derive_run_lane_unmodified"]["status"], "refused")
                self.assertIn("unsupported opcode", io["frozen_derive_run_lane_unmodified"]["reason"])

    def test_assumptions_are_logged_exactly_where_used(self):
        for cid, r in self.rows.items():
            ids = {a["id"] for a in r["assumptions"]}
            self.assertIn("dispatch_trace", ids)
            kid = r["main_kernel"]["kernel_id"]
            applied = set(r["work"]["assumptions_applied"])
            if kid in ("k3", "k4", "k5", "k6"):
                self.assertEqual(applied, {"division_fast_path"}, cid)
            if kid == "k2":
                self.assertEqual(applied, {"bounds_assert_not_taken"}, cid)
                self.assertIn("embedding_fast_path_eligibility", ids)
            for s in r["secondary_kernels"]:
                self.assertEqual(s["work"]["assumptions_applied"], [])

    def test_without_assumptions_the_extended_interpreter_still_refuses_the_data_dependent_kernels(self):
        for r in self.rows.values():
            io = r["main_kernel"]["interpreter_outcomes"]["extended_interpreter_without_assumptions"]
            self.assertEqual(io["status"], "refused")
            for s in r["secondary_kernels"]:
                self.assertEqual(s["interpreter_outcomes"]["extended_interpreter_without_assumptions"]["status"], "ok")

    def test_six_class_events_equal_visit_recomputation(self):
        for r in self.rows.values():
            self.assertTrue(r["work"]["six_class_cross_check_all_equal"])
            for s in r["secondary_kernels"]:
                self.assertTrue(s["work"]["six_class_cross_check_all_equal"])


DISPATCH = A.DISPATCH_JSON


class AnalyticCounts(unittest.TestCase):
    """Counts that follow directly from the source (not from the interpreter): a wrong opcode semantics or
    parameter offset would break these."""

    @classmethod
    def setUpClass(cls):
        cls.rows = get_rows()

    def cells(self, op):
        return [r for r in self.rows.values() if r["operator_id"] == op]

    def test_softmax(self):
        for r in self.cells("dev_pytorch_rowwise_softmax"):
            rows, cols = r["memory"]["shape_a"], r["memory"]["shape_b"]
            w = r["work"]
            elems = rows * cols
            self.assertEqual(w["six_class_counts"]["exponential"], elems)                 # one MUFU.EX2 per element
            wb = 2 if cols <= 128 else 1                                                  # batches per warp
            self.assertEqual(w["six_class_counts"]["shuffle"], rows // wb * 32 * 10 * wb)  # 2 x 5 butterfly steps per batch row
            self.assertEqual(w["executed_global_load_bytes"], elems * 4)
            self.assertEqual(w["executed_global_store_bytes"], elems * 4)
            self.assertEqual(w["six_class_counts"]["barrier"], 0)
            self.assertEqual(w["six_class_counts"]["shared_load"] + w["six_class_counts"]["shared_store"], 0)

    def test_layer_norm(self):
        for r in self.cells("final_pytorch_layer_norm"):
            rows, cols = r["memory"]["shape_a"], r["memory"]["shape_b"]
            w = r["work"]
            n_vec = cols // 4
            fam = w["families"]
            self.assertEqual(fam["global_load"]["width_bits_histogram"]["128"], rows * 2 * n_vec)      # X read twice (stats, normalize)
            self.assertEqual(fam["global_load"]["width_bits_histogram"]["32"], rows * 8 * n_vec)       # gamma and beta, 4 scalars per vector each
            self.assertEqual(fam["global_store"]["width_bits_histogram"]["128"], rows * n_vec)
            self.assertEqual(fam["global_store"]["width_bits_histogram"]["32"], rows * 2)              # mean and rstd
            self.assertEqual(w["six_class_counts"]["shuffle"], rows * 128 * 15)                        # 5 steps x 3 values per lane
            self.assertEqual(w["six_class_counts"]["barrier"], rows * 128 * 5)

    def test_embedding(self):
        for r in self.cells("final_pytorch_embedding"):
            n_idx = self.disp_arg(r, "num_ind")
            slice_b = self.disp_arg(r, "slice_size_bytes")
            fam = r["work"]["families"]
            threads = r["geometry"]["block_threads_dispatch"]
            self.assertEqual(slice_b, threads * 16)                       # one 16-byte vector per thread
            self.assertEqual(fam["global_load"]["width_bits_histogram"]["128"], n_idx * threads)
            self.assertEqual(fam["global_store"]["width_bits_histogram"]["128"], n_idx * threads)
            self.assertEqual(fam["global_load"]["width_bits_histogram"]["64"], n_idx * threads)        # every thread loads the index

    def disp_arg(self, r, key):
        cell = next(c for c in json.loads(DISPATCH.read_text())["cells"] if c["cell_id"] == r["cell_id"])
        return cell["kernels"][-1]["args"][key]

    def test_copy_kernel_moves_every_element_once_and_walks_two_dimensions(self):
        for r in self.rows.values():
            for s in r["secondary_kernels"]:
                n = s["geometry"]["grid_blocks"] * 256
                w = s["work"]
                self.assertEqual(w["executed_global_load_bytes"], n * 4)
                self.assertEqual(w["executed_global_store_bytes"], n * 4)
                self.assertEqual(w["opcode_lane_counts"]["IMAD.HI.U32"], n * 2)   # one magic-number division per dimension (dims = 2)
                self.assertEqual(s["abi"]["bound_values"]["dims"]["value"], 2)


class Robustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D = D
        cls.sites = P.parse((A.SASS_DIR / "k3.isolated.sass").read_text())

    def _k3(self, placeholders=None, rcp=0, cols=1024):
        old = A.POINTER_PLACEHOLDERS
        if placeholders:
            A.POINTER_PLACEHOLDERS = placeholders
        try:
            kern = {"args": {"N": cols, "eps": 1e-5, "gamma_defined": True, "beta_defined": True},
                    "grid": [64, 1, 1], "block": [32, 4, 1]}
            consts, grid, block, _p, _prov = A.binding("k3", kern)
        finally:
            A.POINTER_PLACEHOLDERS = old
        run = A.run_kernel(D, "k3", self.sites, consts, grid, block, extra={"rcp": rcp})
        self.assertEqual(run["status"], "ok")
        return run["events"], [sorted(v.items()) for v in run["visits"]]

    def test_pointer_placeholder_values_do_not_change_counts(self):
        base = self._k3()
        for ph in ((0x7E0000040000, 0x7E0000080010), (0x1000, 0x2004), (0x7F0000001001, 0x7F0000101003)):
            self.assertEqual(self._k3(placeholders=ph), base, ph)

    def test_reciprocal_approximation_error_does_not_change_counts(self):
        base = self._k3()
        for d in (-1, 1):
            self.assertEqual(self._k3(rcp=d), base)

    def test_null_gamma_would_change_the_path(self):
        kern = {"args": {"N": 1024, "eps": 1e-5, "gamma_defined": True, "beta_defined": True}, "grid": [64, 1, 1], "block": [32, 4, 1]}
        consts, grid, block, _p, _prov = A.binding("k3", kern)
        off = {n: o for n, o, _ in A.abi_layout("k3")}
        consts[off["gamma"]] = 0
        consts[off["gamma"] + 4] = 0
        base = A.run_kernel(D, "k3", self.sites, *A.binding("k3", kern)[:3])
        run = A.run_kernel(D, "k3", self.sites, consts, grid, block)
        self.assertEqual(run["status"], "ok")
        self.assertNotEqual(sum(sum(v.values()) for v in run["visits"]), sum(sum(v.values()) for v in base["visits"]))


class NoZeroFill(unittest.TestCase):
    def test_refusal_gives_null_work_and_a_missing_feature(self):
        saved = A.run_kernel

        def refusing(D_, kid, *a, **k):
            if kid == "k3":
                return {"status": "refused", "reason": "injected refusal for the test"}
            return saved(D_, kid, *a, **k)
        A.run_kernel = refusing
        try:
            rows = A.build_rows(X, X.load_hardware(X.GROUND_TRUTH.read_text()), X.load_dev_rows())
        finally:
            A.run_kernel = saved
        ln = [r for r in rows.values() if r["operator_id"] == "final_pytorch_layer_norm"]
        self.assertEqual(len(ln), 12)
        for r in ln:
            self.assertEqual(r["status"], "missing_features")
            self.assertIsNone(r["work"])
            self.assertIsNone(r["structure"])
            self.assertIsNone(r["per_launch_totals"])
            reasons = [m["reason"] for m in r["missing_features"]]
            self.assertTrue(any("injected refusal" in x for x in reasons))
            self.assertEqual(r["main_kernel"]["interpreter_outcomes"]["extended_interpreter"]["status"], "refused")
            self.assertNotIn("six_class_counts", r["resources"])
        for r in rows.values():
            if r["operator_id"] != "final_pytorch_layer_norm":
                self.assertEqual(r["status"], "supported_with_assumptions")


class OutputFile(unittest.TestCase):
    def test_xformers_cells_stay_without_a_binary_and_pytorch_cells_have_features(self):
        doc = json.loads(X.OUT_JSON.read_text())
        self.assertEqual(len(doc["rows"]), 132)
        by = collections.Counter((r["operator_id"], r["status"]) for r in doc["rows"])
        self.assertEqual(by[("final_xformers_indexed_select", "no_retained_binary")], 12)
        self.assertEqual(by[("dev_pytorch_rowwise_softmax", "supported_with_assumptions")], 12)
        self.assertEqual(by[("final_pytorch_layer_norm", "supported_with_assumptions")], 12)
        self.assertEqual(by[("final_pytorch_embedding", "supported_with_assumptions")], 3)
        self.assertEqual(sum(1 for r in doc["rows"] if r["status"] == "supported"), 93)


if __name__ == "__main__":
    unittest.main(verbosity=2)
