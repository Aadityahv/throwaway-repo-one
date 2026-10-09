"""CPU-only tests of score_h100.py on synthetic predictions and measurements (no H100 value of any kind is involved).

    python -m pytest tiresias/framework/predictor/h100_eval_freeze/test_score_h100.py
"""
import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_h100 as S  # noqa: E402

CELLS = json.loads((HERE / "cells_h100.json").read_text())["cells"]
IDS = [c["cell_id"] for c in CELLS]
CAP = 700.0
RAW_FIELDS = ["parent_id", "regime", "candidate_id", "board_energy_j_total", "counted_launch_interval_s", "board_energy_j_per_launch", "launch_count"]


class Synthetic(unittest.TestCase):
    runtime_err = 0.10           # every prediction is 10% high
    measure_uuid = "GPU-aaaaaaaa-f4ce-0015-2700-937f98b06266"
    n_unsupported = 4

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.tmp, True)
        self.unsupported = set(IDS[:self.n_unsupported])
        cells, roof = {}, {}
        for i, cid in enumerate(IDS):
            t = 1e-5 * (1 + i)
            terms = {"B_l2": 0.001 * (1 + i % 5), "ffma": 0.002}
            base = 100.0
            if cid in self.unsupported:
                cells[cid] = dict(status="unsupported", reason="test", counted_as_failure=True)
            else:
                tp = t * (1 + self.runtime_err)
                cells[cid] = dict(status="predicted", runtime_s=tp, base_power_w=base, term_j=terms, energy_j=min(CAP * tp, base * tp + sum(terms.values())), mean_power_w=1.0, capped=False,
                                  base_term_j=base * tp)
            roof[cid] = t * 3.0
        self.pred = dict(kind="h100_frozen_predictions", warning=None, calibration=dict(energy_cap_w=CAP, board_seconds_recorded=3000.0, stages_without_recorded_seconds=[], uuid="GPU-aaaaaaaa-f4ce-0015-2700-937f98b06266",
                                                                                          source="job 1 repeat 1", energy_windows_not_admitted=[]),
                         cells=cells, roofline_runtime_s=roof)
        self.pred_path = self.tmp / "predictions_h100.json"; self.pred_path.write_text(json.dumps(self.pred))
        self.sha = hashlib.sha256(self.pred_path.read_bytes()).hexdigest()
        self.rec_path = self.tmp / "freeze_record_h100.json"; self.rec_path.write_text(json.dumps(dict(schema="h100_freeze_record/1", freeze_commit="b" * 40, predictions_sha256=self.sha)))
        self.timing = dict(meta=dict(freeze=dict(predictions_sha256=self.sha), gpu=dict(uuid=self.measure_uuid)), engines={"set_d": dict(wall_s=100.0)}, gate_refused_cells=[],
                           cells=[dict(cell_id=cid, per_launch_runtime_s=1e-5 * (1 + i), correct=True) for i, cid in enumerate(IDS)])
        self.timing_path = self.tmp / "timing.json"; self.timing_path.write_text(json.dumps(self.timing))
        # energy: measured energy = 5% below the model at the measured runtime; power 300 W (below cap)
        self.edir = self.tmp / "energy"; self.edir.mkdir()
        with (self.edir / "application_energy_raw.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RAW_FIELDS); w.writeheader()
            for i, c in enumerate(CELLS):
                t = 1e-5 * (1 + i); terms = {"B_l2": 0.001 * (1 + i % 5), "ffma": 0.002}
                e = min(CAP * t, 100.0 * t + sum(terms.values())) / 1.05
                launches = 1000
                w.writerow(dict(parent_id=c["operator_id"], regime=c["regime"], candidate_id=c["candidate_id"], board_energy_j_total=e * launches, counted_launch_interval_s=e * launches / 300.0,
                                board_energy_j_per_launch=e, launch_count=launches))

    def score(self, **kw):
        return S.score(self.pred_path, self.rec_path, self.timing_path, kw.get("energy", ()), kw.get("baselines"), cells_path=HERE / "cells_h100.json")


class RuntimeScores(Synthetic):
    def test_known_errors_and_criteria(self):
        r = S.score(self.pred_path, self.rec_path, self.timing_path, cells_path=HERE / "cells_h100.json")
        rt = r["runtime"]
        self.assertAlmostEqual(rt["supported_cells"]["median_pct"], 10.0, places=1); self.assertEqual(rt["supported_cells"]["n"], 68)
        self.assertEqual(rt["unsupported_as_failure"]["n"], 72); self.assertEqual(rt["unsupported_as_failure"]["failures"], 4)
        self.assertAlmostEqual(rt["unsupported_as_failure"]["median_pct"], 10.0, places=1)       # 4 infinite errors of 72 do not move the median
        self.assertAlmostEqual(rt["signed_median_pct"], 10.0, places=1)
        self.assertTrue(r["criteria"]["runtime_median_le_15_supported"]); self.assertTrue(r["criteria"]["runtime_median_le_15_unsupported_as_failure"])
        self.assertTrue(r["criteria"]["runtime_beats_roofline_median"])                          # roofline is 200% off in this synthetic data
        self.assertEqual(r["runtime"]["per_cell"][IDS[0]]["predicted_s"], None)

    def test_many_unsupported_cells_fail_the_failure_counting_criterion_only(self):
        self.n_unsupported = 40
        self.setUp()
        r = S.score(self.pred_path, self.rec_path, self.timing_path, cells_path=HERE / "cells_h100.json")
        self.assertTrue(r["criteria"]["runtime_median_le_15_supported"])
        self.assertFalse(r["criteria"]["runtime_median_le_15_unsupported_as_failure"])          # the median now reaches an infinite error: reported null, criterion false
        self.assertIsNone(r["runtime"]["unsupported_as_failure"]["median_pct"])

    def test_a_miss_is_reported_as_measured(self):
        self.runtime_err = 0.30
        self.setUp()
        r = S.score(self.pred_path, self.rec_path, self.timing_path, cells_path=HERE / "cells_h100.json")
        self.assertFalse(r["criteria"]["runtime_median_le_15_supported"]); self.assertIn("stop_rule", r)

    def test_correctness_failed_cells_are_excluded_and_listed(self):
        t = json.loads(self.timing_path.read_text()); t["cells"][10]["correct"] = False; t["cells"][11]["correct"] = False
        self.timing_path.write_text(json.dumps(t))
        r = S.score(self.pred_path, self.rec_path, self.timing_path, cells_path=HERE / "cells_h100.json")
        self.assertEqual(sorted(r["runtime"]["excluded_correctness_failed"]), sorted([IDS[10], IDS[11]]))
        self.assertEqual(r["runtime"]["measured_correct"], 70)

    def test_per_group_breakdown_present(self):
        r = S.score(self.pred_path, self.rec_path, self.timing_path, cells_path=HERE / "cells_h100.json")
        self.assertEqual(set(r["runtime"]["per_tier"]), {"L2", "DRAM"}); self.assertEqual(set(r["runtime"]["per_set"]), {"unseen", "set_e", "set_d"})


class Refusals(Synthetic):
    def refuses(self, fragment):
        with self.assertRaises(SystemExit) as cm:
            S.score(self.pred_path, self.rec_path, self.timing_path, cells_path=HERE / "cells_h100.json")
        self.assertIn(fragment, str(cm.exception))

    def test_predictions_changed_after_the_freeze(self):
        self.pred_path.write_text(self.pred_path.read_text() + " ")
        self.refuses("differs from the sha256 recorded")

    def test_plumbing_file_is_refused(self):
        d = dict(self.pred, kind="plumbing_test_not_h100", warning="PLUMBING"); self.pred_path.write_text(json.dumps(d))
        self.refuses("not an H100 frozen-predictions file")

    def test_timing_taken_against_other_predictions(self):
        t = json.loads(self.timing_path.read_text()); t["meta"]["freeze"]["predictions_sha256"] = "0" * 64; self.timing_path.write_text(json.dumps(t))
        self.refuses("not taken against the frozen predictions")

    def test_missing_record_and_incomplete_coverage(self):
        self.rec_path.unlink()
        self.refuses("is missing")

    def test_output_is_never_overwritten(self):
        out = self.tmp / "score.json"; out.write_text("{}")
        with self.assertRaises(SystemExit):
            S.main(["--predictions", str(self.pred_path), "--record", str(self.rec_path), "--timing", str(self.timing_path), "--out", str(out)])


class EnergyScores(Synthetic):
    def test_energy_error_and_criteria(self):
        r = S.score(self.pred_path, self.rec_path, self.timing_path, [self.edir], cells_path=HERE / "cells_h100.json")
        e = r["energy"]
        self.assertEqual(e["cells_with_energy"], 72); self.assertEqual(e["cells_without_energy"], [])
        # at the measured runtime the model is 5% above the measurement; at the predicted (10% higher) runtime the error is larger but both are finite and ordered
        self.assertAlmostEqual(e["ours_measured_runtime"]["below_cap"]["median_pct"], 5.0, places=1)
        self.assertGreater(e["ours_static_runtime"]["below_cap"]["median_pct"], e["ours_measured_runtime"]["below_cap"]["median_pct"])
        self.assertEqual(e["ours_static_runtime"]["below_cap_unsupported_as_failure"]["failures"], 4)
        self.assertIn("not scored", e["published_methods"])                                     # no frozen baselines file given
        self.assertIn("energy_median_within_5pp_of_measured_runtime_below_cap", r["criteria"])

    def test_published_rows_need_a_frozen_file_and_count_missing_as_failures(self):
        cells = {cid: dict(kind="constant_plus_variable", constant_power_w=100.0, variable_j=0.003) for cid in IDS[2:]}          # two cells have no entry: failures
        ala = {cid: dict(kind="native_power", power_w=300.0) for cid in IDS}
        b = self.tmp / "baselines_h100.json"
        b.write_text(json.dumps(dict(methods={"accelwattch_style": dict(status="run", label="x", cells=cells), "alavani_refit": dict(status="run", label="y", cells=ala),
                                              "flipflop": dict(status="not_run_on_h100", reason="no calibration from the packaged run", blockers=["power_alpha"])})))
        r = S.score(self.pred_path, self.rec_path, self.timing_path, [self.edir], b, cells_path=HERE / "cells_h100.json")
        pm = r["energy"]["published_methods"]
        self.assertEqual(set(pm["methods_run"]), {"accelwattch_style", "alavani_refit"})
        self.assertEqual(pm["methods_not_run"]["flipflop"]["blockers"], ["power_alpha"])
        aw = pm["methods_run"]["accelwattch_style"]
        self.assertEqual(aw["measured_runtime"]["all_scored_cells"]["failures"], 2)
        self.assertEqual(aw["our_static_runtime"]["all_scored_cells"]["failures"], 4)             # the two cells without an entry plus the four unsupported cells (they overlap in two): no static runtime = failure
        self.assertEqual(pm["methods_run"]["alavani_refit"]["measured_runtime"]["all_scored_cells"]["failures"], 0)
        self.assertEqual(S.form_energy(dict(kind="native_power", power_w=300.0), 2.0, CAP), 600.0)
        self.assertEqual(S.form_energy(dict(kind="constant_plus_variable", constant_power_w=100.0, variable_j=1.0), 0.001, CAP), CAP * 0.001)       # capped
        self.assertAlmostEqual(S.form_energy(dict(kind="constant_plus_variable", constant_power_w=100.0, variable_j=0.0001), 0.001, CAP), 0.1 + 0.0001)

    def test_cells_above_the_cap_rule_are_excluded_for_every_method(self):
        rows = list(csv.DictReader((self.edir / "application_energy_raw.csv").open(newline="")))
        hot = rows[40]
        for r in rows:
            if r is hot:
                r["counted_launch_interval_s"] = float(r["board_energy_j_total"]) / (0.99 * CAP)        # 99% of the limit: above the 98.5% rule
        with (self.edir / "application_energy_raw.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RAW_FIELDS); w.writeheader(); w.writerows(rows)
        r = S.score(self.pred_path, self.rec_path, self.timing_path, [self.edir], cells_path=HERE / "cells_h100.json")
        e = r["energy"]
        self.assertEqual(e["excluded_above_the_cap_rule"]["count"], 1)
        self.assertEqual(e["cells_scored"], 71); self.assertEqual(e["cells_with_energy"], 72)
        self.assertEqual(e["excluded_above_the_cap_rule"]["cells"], ["h100/%s/%s/%s" % (hot["parent_id"], hot["regime"], hot["candidate_id"])])

    def test_gpu_comparison_is_reported(self):
        self.assertTrue(S.score(self.pred_path, self.rec_path, self.timing_path, cells_path=HERE / "cells_h100.json")["gpu"]["same_gpu"])
        t = json.loads(self.timing_path.read_text()); t["meta"]["gpu"]["uuid"] = "GPU-bbbbbbbb-9cd2-e827-7632-3ad7b19aaf8a"; self.timing_path.write_text(json.dumps(t))
        g = S.score(self.pred_path, self.rec_path, self.timing_path, cells_path=HERE / "cells_h100.json")["gpu"]
        self.assertFalse(g["same_gpu"]); self.assertIn("GPU-to-GPU variation", g["statement"])

    def test_decision_utility_picks_the_lowest_energy_candidate(self):
        r = S.score(self.pred_path, self.rec_path, self.timing_path, [self.edir], cells_path=HERE / "cells_h100.json")
        d = r["decision_utility"]
        self.assertGreater(d["groups"], 20)
        for row in d["rows"]:
            self.assertGreaterEqual(row["fastest_regret_pct"], 0)
        self.assertIn("different claim", d["note"])

    def test_cost_uses_the_log_and_the_calibration_time(self):
        with (self.edir / "h100_set_d_energy_log.jsonl").open("w") as f:
            for i in range(5):
                f.write(json.dumps(dict(cell_id="x%d" % i, status="raw", utc="2026-10-03T00:%02d:%02dZ" % ((i * 156) // 60, (i * 156) % 60))) + "\n")
        r = S.score(self.pred_path, self.rec_path, self.timing_path, [self.edir], cells_path=HERE / "cells_h100.json")
        c = r["cost"]
        self.assertAlmostEqual(c["energy_window_service_time_s_median"], 156.0); self.assertEqual(c["valid_windows_timed"], 4)
        self.assertAlmostEqual(c["break_even_cells"], 3000.0 / 156.0, places=1)


class Statistics(unittest.TestCase):
    def test_percentile_with_infinite_errors(self):
        self.assertIsNone(S.percentile([1, 2, math.inf, math.inf], 0.9))
        self.assertEqual(S.percentile([1, 2, 3], 0.5), 2)
        self.assertEqual(S.stats([1.0, math.inf])["failures"], 1)


if __name__ == "__main__":
    unittest.main()
