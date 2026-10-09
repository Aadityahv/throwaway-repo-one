"""CPU test of score_ada.py's multi-file / joint scoring on SYNTHETIC measurements derived from the frozen Ada predictions (never real data): structure, counts, refusals.
Run: python -m pytest -q test_score_ada_joint.py"""
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_sets_ada as CS  # noqa: E402
import score_ada as SC  # noqa: E402

pytestmark = pytest.mark.skipif(not all((HERE / CS.PROFILES[p]["predictions"]).is_file() for p in CS.PROFILES), reason="frozen Ada predictions not in this tree")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture()
def fx(tmp_path):
    rnd = random.Random(11)
    tree = tmp_path / "tree"
    tree.mkdir()
    timing_files = {"main": [], "tensor": []}
    rows = []
    for prof in ("main", "tensor"):
        pr = CS.PROFILES[prof]
        (tree / pr["predictions"]).write_bytes((HERE / pr["predictions"]).read_bytes())
        (tree / pr["baselines"]).write_bytes((HERE / pr["baselines"]).read_bytes())
        rec = dict(schema="x", profile=prof, freeze_commit="c" * 40, predictions_sha256=sha(HERE / pr["predictions"]))
        (tree / pr["record"]).write_text(json.dumps(rec))
        pred = json.loads((HERE / pr["predictions"]).read_text())
        uuid = pred["calibration"]["uuid"]
        cells = []
        for cid, v in pred["cells"].items():
            base_t = v["runtime_s"] if v["status"] == "predicted" else pred["roofline_runtime_s"][cid]
            t = base_t * (1 + rnd.uniform(-0.12, 0.12))
            cells.append(dict(cell_id=cid, correct=True, per_launch_runtime_s=t))
            e = (v["energy_j"] if v["status"] == "predicted" else base_t * 150) * (1 + rnd.uniform(-0.1, 0.1))
            _, op, regime, cand = cid.split("/")
            rows.append(dict(parent_id=op, regime=regime, candidate_id=cand, board_energy_j_per_launch=e, board_energy_j_total=e * 1000, counted_launch_interval_s=t * 1000, launch_count=1000, gpu_uuid=uuid, correctness_check="True"))
        meta = lambda: dict(profile=prof, freeze=dict(predictions_sha256=rec["predictions_sha256"]), gpu=dict(uuid=uuid))
        if prof == "main":     # first run lacks the last 8 cells, a resume run holds them
            parts = [cells[:-8], cells[-8:]]
        else:
            parts = [cells]
        for i, part in enumerate(parts):
            f = tmp_path / ("timing_%s_%d.json" % (prof, i))
            f.write_text(json.dumps(dict(meta=meta(), cells=part, gate_refused_cells=[], engines={})))
            timing_files[prof].append(f)
    dirs = []
    for i in range(3):
        d = tmp_path / ("energy%d" % i)
        d.mkdir()
        part = rows[i::3]
        with (d / "application_energy_raw.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(part)
        dirs.append(d)
    return tree, timing_files, dirs, rows, tmp_path


def test_joint_scores_168_cells_from_several_files(fx):
    tree, tf, dirs, rows, tmp = fx
    rep = SC.score_joint(tf["main"], tf["tensor"], dirs, pred_dir=tree)
    rt = rep["runtime"]
    assert rt["cells_in_list"] == 168 and rt["measured_correct"] == 168
    assert rt["unsupported_as_failure"]["n"] == 168 and rt["supported_cells"]["n"] + rt["unsupported_as_failure"]["failures"] == 168 and rt["unsupported_as_failure"]["failures"] > 0
    assert rt["supported_cells"]["median_pct"] < 15 and rep["criteria"]["runtime_median_le_15_supported"]
    en = rep["energy"]
    assert en["cells_with_energy"] == 168 and "controls" in en and rep["gpu"]["same_gpu"] is True
    assert rep["held_out"]["prospective"]["cells"] == 24 and rep["held_out"]["all_other_cells"]["cells"] == 144 and rep["held_out"]["validation"]["cells"] == 12
    assert rep["decision_utility"]["summary"]["groups"] > 20


def test_single_profile_score_accepts_two_timing_files(fx):
    tree, tf, dirs, rows, tmp = fx
    pr = CS.PROFILES["main"]
    rep = SC.score(tree / pr["predictions"], tree / pr["record"], tf["main"], dirs, None, "main")
    assert rep["runtime"]["measured_correct"] == 124 and len(rep["inputs"]["timing_sha256"]) == 2


def test_refuses_a_cell_timed_twice(fx):
    tree, tf, dirs, rows, tmp = fx
    t = json.loads(tf["main"][0].read_text())
    t["cells"] = t["cells"] + json.loads(tf["main"][1].read_text())["cells"][:1]
    (tmp / "dup.json").write_text(json.dumps(t))
    with pytest.raises(SystemExit, match="timed twice"):
        SC.score_joint([tmp / "dup.json", tf["main"][1]], tf["tensor"], dirs, pred_dir=tree)


def test_refuses_missing_cells(fx):
    tree, tf, dirs, rows, tmp = fx
    with pytest.raises(SystemExit, match="not exactly"):
        SC.score_joint(tf["main"][:1], tf["tensor"], dirs, pred_dir=tree)


def test_refuses_timing_against_other_predictions(fx):
    tree, tf, dirs, rows, tmp = fx
    t = json.loads(tf["tensor"][0].read_text())
    t["meta"]["freeze"]["predictions_sha256"] = "0" * 64
    (tmp / "bad.json").write_text(json.dumps(t))
    with pytest.raises(SystemExit, match="not taken against the frozen"):
        SC.score_joint(tf["main"], [tmp / "bad.json"], dirs, pred_dir=tree)


def test_refuses_two_energy_rows_for_a_cell(fx):
    tree, tf, dirs, rows, tmp = fx
    d = tmp / "energy_dup"
    d.mkdir()
    (d / "application_energy_raw.csv").write_text((dirs[0] / "application_energy_raw.csv").read_text())
    with pytest.raises(SystemExit, match="two energy rows"):
        SC.score_joint(tf["main"], tf["tensor"], dirs + [d], pred_dir=tree)


def test_cap_rule_exclusion_and_throttle_report(fx):
    tree, tf, dirs, rows, tmp = fx
    r = list(csv.DictReader((dirs[0] / "application_energy_raw.csv").open()))
    r[0]["board_energy_j_total"] = str(float(r[0]["counted_launch_interval_s"]) * 249.0)    # above 98.5 percent of 250 W
    with (dirs[0] / "application_energy_raw.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r[0]))
        w.writeheader()
        w.writerows(r)
    r[1]["board_energy_j_total"] = str(float(r[1]["counted_launch_interval_s"]) * 100.0)
    with (dirs[0] / "application_energy_raw.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r[0]))
        w.writeheader()
        w.writerows(r)
    cid = "h100/%s/%s/%s" % (r[1]["parent_id"], r[1]["regime"], r[1]["candidate_id"])
    g = tmp / "gate.json"
    g.write_text(json.dumps(dict(clock_throttled=[cid])))
    rep = SC.score_joint(tf["main"], tf["tensor"], dirs, [g], pred_dir=tree)
    assert rep["energy"]["excluded_above_the_cap_rule"]["count"] >= 1
    assert rep["clock_throttled_windows"]["among_scored_cells"] == 1


def test_refuses_to_overwrite_a_score(fx, tmp_path):
    out = tmp_path / "s.json"
    out.write_text("{}")
    with pytest.raises(SystemExit, match="never overwritten"):
        SC.main(["--joint", "--timing-main", "a", "--timing-tensor", "b", "--out", str(out)])
