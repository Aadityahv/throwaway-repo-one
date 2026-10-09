"""CPU test of score_board.py on SYNTHETIC measurements derived from the frozen predictions (never real data): structure, counts, refusals. Run: python -m pytest -q test_score_board.py"""
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_sets_board as CS  # noqa: E402
import score_board as SB  # noqa: E402

BOARD = "h100"
BD = CS.board_dir(BOARD)
pytestmark = pytest.mark.skipif(not all((BD / CS.names(BOARD, p)["predictions"]).is_file() for p in CS.PROFILES), reason="frozen predictions not in this tree")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture()
def fixture(tmp_path):
    rnd = random.Random(7)
    tree = tmp_path / "tree"
    tree.mkdir()
    timings, rows = {}, []
    for prof in CS.PROFILES:
        n = CS.names(BOARD, prof)
        (tree / n["predictions"]).write_bytes((BD / n["predictions"]).read_bytes())
        rec = dict(schema="cluster_freeze_record/1", board=BOARD, profile=prof, freeze_commit="c" * 40, predictions_sha256=sha(BD / n["predictions"]), sha256={})
        (tree / n["record"]).write_text(json.dumps(rec))
        pred = json.loads((BD / n["predictions"]).read_text())
        cells = []
        for cid, v in pred["cells"].items():
            base_t = v["runtime_s"] if v["status"] == "predicted" else pred["roofline_runtime_s"][cid]
            t = base_t * (1 + rnd.uniform(-0.12, 0.12))
            cells.append(dict(cell_id=cid, correct=True, per_launch_runtime_s=t, set="x"))
            e = (v["energy_j"] if v["status"] == "predicted" else base_t * 300) * (1 + rnd.uniform(-0.1, 0.1))
            _, operator, regime, cand = cid.split("/")
            rows.append(dict(parent_id=operator, regime=regime, candidate_id=cand, board_energy_j_per_launch=e, board_energy_j_total=e * 1000, counted_launch_interval_s=t * 1000, launch_count=1000))
        timings[prof] = dict(meta=dict(profile=prof, freeze=dict(predictions_sha256=rec["predictions_sha256"]), gpu=dict(uuid="GPU-29b9a2c0-8d11-0ab9-3fd6-cd57f364246b")), cells=cells,
                             gate_refused_cells=[], engines={})
        (tmp_path / ("timing_%s.json" % prof)).write_text(json.dumps(timings[prof]))
    edir = tmp_path / "energy"
    edir.mkdir()
    with (edir / "application_energy_raw.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return tree, tmp_path, edir


def test_scores_all_168_cells_with_failures_counted(fixture):
    tree, tmp, edir = fixture
    rep = SB.score(BOARD, tmp / "timing_main.json", tmp / "timing_tensor.json", [edir], tree=tree)
    rt = rep["runtime"]
    assert rt["cells_in_list"] == 168 and rt["measured_correct"] == 168
    assert rt["supported_cells"]["n"] == 164 and rt["unsupported_as_failure"]["n"] == 168 and rt["unsupported_as_failure"]["failures"] == 4
    assert rt["supported_cells"]["median_pct"] < 15 and rep["criteria"]["runtime_median_le_15_supported"]
    assert set(rt["per_profile"]) == {"main", "tensor"}
    en = rep["energy"]
    assert en["cells_with_energy"] == 168 and en["ours_static_runtime"]["all"]["n"] + en["excluded_above_the_cap_rule"]["count"] >= 160
    assert "controls" in en and en["controls"]["constant_power_times_measured_runtime"]["constant_power_w"] > 100
    assert "not run on this board" in en["published_methods"]
    assert rep["gpu"]["same_gpu"] is True
    assert rep["decision_utility"]["groups"] > 20


def test_runtime_only_scoring_and_a_profile_not_measured(fixture):
    tree, tmp, _ = fixture
    rep = SB.score(BOARD, tmp / "timing_main.json", None, [], tree=tree)
    assert rep["scope"]["profiles_not_measured"] == ["tensor"] and rep["runtime"]["cells_in_list"] == 124 and "energy" not in rep


def test_refuses_timing_not_taken_against_the_frozen_predictions(fixture):
    tree, tmp, _ = fixture
    t = json.loads((tmp / "timing_main.json").read_text())
    t["meta"]["freeze"]["predictions_sha256"] = "0" * 64
    (tmp / "timing_bad.json").write_text(json.dumps(t))
    with pytest.raises(SystemExit):
        SB.score(BOARD, tmp / "timing_bad.json", None, [], tree=tree)


def test_refuses_predictions_changed_after_the_freeze(fixture):
    tree, tmp, _ = fixture
    p = tree / CS.names(BOARD, "main")["predictions"]
    p.write_text(p.read_text().replace('"board": "h100"', '"board": "h100" ', 1))
    with pytest.raises(SystemExit):
        SB.score(BOARD, tmp / "timing_main.json", None, [], tree=tree)


def test_a_cell_above_the_cap_rule_is_excluded_for_energy(fixture):
    tree, tmp, edir = fixture
    rows = list(csv.DictReader((edir / "application_energy_raw.csv").open()))
    rows[0]["board_energy_j_total"] = str(float(rows[0]["counted_launch_interval_s"]) * 695.0)     # 695 W > 98.5 percent of 700 W
    with (edir / "application_energy_raw.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rep = SB.score(BOARD, tmp / "timing_main.json", tmp / "timing_tensor.json", [edir], tree=tree)
    assert rep["energy"]["excluded_above_the_cap_rule"]["count"] >= 1


def test_energy_rows_are_keyed_by_the_boards_own_name_and_h100_is_unchanged(tmp_path):
    """The shared reader hard-codes "h100/"; on the A100 every energy row then failed to match its cell (zero energy cells scored, 7 October 2026). The board-aware reader must key by the board, and for h100
    must equal the shared reader row for row."""
    import score_h100 as S
    d = tmp_path / "e"
    d.mkdir()
    cols = ["parent_id", "regime", "candidate_id", "board_energy_j_total", "counted_launch_interval_s", "board_energy_j_per_launch", "launch_count"]
    with (d / "application_energy_raw.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerow(["parentX", "small", "c1", "800.0", "20.0", "0.5", "1600"])
    a = SB.read_energy_board("a100", [d])
    assert list(a) == ["a100/parentX/small/c1"] and abs(a["a100/parentX/small/c1"]["mean_power_w"] - 40.0) < 1e-9
    assert SB.read_energy_board("h100", [d]) == S.read_energy([d])
    assert not S.read_energy([d]).get("a100/parentX/small/c1")        # the shared reader would not have matched an a100 cell
