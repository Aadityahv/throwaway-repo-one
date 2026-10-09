"""Integration test of the in-job freeze (injob_freeze.py) on the H100's real archived calibration, pair-overlap run and power-sensor result, in a mirror of the tree (nothing is written into the repo):
it must reproduce the committed H100 predictions cell by cell, write in-job freeze records that verify, and be accepted by the measurement wrapper's freeze gate. CPU only, about a minute.
Run: python -m pytest -q test_injob_freeze.py"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SR = HERE.parent
sys.path.insert(0, str(HERE))
import cell_sets_board as CS  # noqa: E402
import freeze_board as FZ  # noqa: E402
import measure_board as M  # noqa: E402

BOARD = "h100"
RUNS = SR / "calibrate" / "runs" / "cluster_h100_replication_20261004"
CAL_RUN = RUNS / "h100_j20" / "run"
PAIR = RUNS / "pair_overlap_29b9a2c0" / "dep_frag.jsonl"
SENSOR = SR / "h100_power_sensor_test" / "runs" / "h100_j16" / "power_sensor_results.json"
CODE = "a" * 40
need = [CAL_RUN / "calibration_sm_90_29b9a2c0.json", PAIR, SENSOR] + [HERE / BOARD / CS.names(BOARD, p)["predictions"] for p in CS.PROFILES]
pytestmark = pytest.mark.skipif(not all(p.is_file() for p in need), reason="archived H100 inputs are not in this tree")


@pytest.fixture(scope="module")
def mirror(tmp_path_factory):
    root = tmp_path_factory.mktemp("mirror")
    sr = root / "tiresias" / "framework" / "predictor"
    sr.mkdir(parents=True)
    for child in SR.iterdir():                                           # siblings are linked (read only use); the board directory is copied; calibrate gets its own empty runs/
        if child.name in ("replication_h100_a100", "calibrate", "constants"):
            continue
        (sr / child.name).symlink_to(child)
    shutil.copytree(SR / "constants", sr / "constants")                    # a real copy: a symlinked script resolves back into the repo and would read and write there
    cal = sr / "calibrate"
    cal.mkdir()
    for child in (SR / "calibrate").iterdir():
        if child.name != "runs":
            (cal / child.name).symlink_to(child)
    (cal / "runs").mkdir()
    skip = shutil.ignore_patterns("build", "__pycache__", "predictions_replication_*", "freeze_record_*", "pair_overlap_constants_29b9a2c0.json", "power_sensor_results_29b9a2c0.json", "compile_evidence")
    shutil.copytree(HERE, sr / "replication_h100_a100", ignore=lambda d, names: skip(d, names) if Path(d).name in ("replication_h100_a100", BOARD) else [n for n in names if n == "__pycache__"])
    (root / "tiresias" / "app_runners").symlink_to(SR.parents[1] / "app_runners")
    (root / "HARDWARE_GROUND_TRUTH.md").symlink_to(SR.parents[2] / "HARDWARE_GROUND_TRUTH.md")
    calrun = root / "calrun" / "run"
    (calrun / "stages").mkdir(parents=True)
    shutil.copy(CAL_RUN / "calibration_sm_90_29b9a2c0.json", calrun)
    for st in ("store", "stream"):
        (calrun / "stages" / st).mkdir()
        shutil.copy(CAL_RUN / "stages" / st / "stdout.jsonl", calrun / "stages" / st)
    pairrun = root / "pairrun"
    pairrun.mkdir()
    shutil.copy(PAIR, pairrun)
    sensor = root / "sensorrun" / "h100_1"
    sensor.mkdir(parents=True)
    shutil.copy(SENSOR, sensor)
    # the real allocation's output directories also hold a private copy of the source tree (src/), with OLDER results of other runs inside it (A100 job j24, 6 October 2026: a stale archived
    # power-sensor result inside sensor/<run>/src/ made the freeze stop with "found 2"). The mirror reproduces that layout, so the freeze must pick only its own run's files.
    stale_sensor = sensor / "src" / "tiresias" / "framework" / "predictor" / "h100_power_sensor_test" / "runs" / "h100_old"
    stale_sensor.mkdir(parents=True)
    shutil.copy(SENSOR, stale_sensor)
    stale_cal = root / "calrun" / "src" / "tiresias" / "framework" / "predictor" / "calibrate" / "runs" / "old" / "run"
    stale_cal.mkdir(parents=True)
    shutil.copy(CAL_RUN / "calibration_sm_90_29b9a2c0.json", stale_cal / "calibration_sm_80_00000000.json")
    board_dir = sr / "replication_h100_a100"
    r = subprocess.run([sys.executable, str(board_dir / "injob_freeze.py"), "--board", BOARD, "--calibration-run", str(root / "calrun"), "--pair-run", str(pairrun), "--sensor-run", str(root / "sensorrun"),
                        "--code-commit", CODE, "--env-out", str(root / "injob.env")], capture_output=True, text=True, cwd=str(board_dir))
    return root, board_dir, r


def test_the_in_job_freeze_runs(mirror):
    root, board_dir, r = mirror
    assert r.returncode == 0, r.stderr[-1500:]
    assert (root / "injob.env").is_file() and "WINDOW_S=40.0" in (root / "injob.env").read_text()


def test_predictions_equal_the_committed_freeze_cell_by_cell(mirror):
    root, board_dir, r = mirror
    assert r.returncode == 0, r.stderr[-800:]
    for prof in CS.PROFILES:
        n = CS.names(BOARD, prof)
        new = json.loads((board_dir / BOARD / n["predictions"]).read_text())
        old = json.loads((HERE / BOARD / n["predictions"]).read_text())
        assert set(new["cells"]) == set(old["cells"])
        for cid, v in old["cells"].items():
            w = new["cells"][cid]
            assert w["status"] == v["status"], cid
            if v["status"] == "predicted":
                assert w["runtime_s"] == pytest.approx(v["runtime_s"], rel=1e-9) and w["energy_j"] == pytest.approx(v["energy_j"], rel=1e-9), cid
        mine = json.loads(next((root / "calrun" / "run").glob("*_store_stream_rates_store_stream_rates.json")).read_text())
        theirs = json.loads((CAL_RUN / "calibration_sm_90_29b9a2c0_store_stream_rates_store_stream_rates.json").read_text())
        assert mine["constants"] == theirs["constants"]                           # the document differs only by its derivation time stamps: the constants are the committed ones
        assert new["pair_overlap_constants"]["beta_hmma"] == old["pair_overlap_constants"]["beta_hmma"]


def test_records_verify_and_the_wrapper_gate_accepts_them(mirror):
    root, board_dir, r = mirror
    assert r.returncode == 0, r.stderr[-800:]
    for prof in CS.PROFILES:
        rec = FZ.check(BOARD, prof, tree=board_dir / BOARD)
        assert rec["mode"] == "in_job" and rec["freeze_commit"] == CODE and rec["created_utc"]
        info = M.check_freeze(BOARD, prof, root, CODE, use_git=False)
        assert info["mode"] == "in_job" and "no git ancestry applies" in info["ancestry"]


def test_a_record_of_another_code_commit_is_refused_by_the_gate(mirror):
    root, board_dir, r = mirror
    with pytest.raises(SystemExit):
        M.check_freeze(BOARD, "main", root, "b" * 40, use_git=False)


def test_the_in_job_freeze_refuses_to_overwrite(mirror):
    root, board_dir, r = mirror
    again = subprocess.run([sys.executable, str(board_dir / "injob_freeze.py"), "--board", BOARD, "--calibration-run", str(root / "calrun"), "--pair-run", str(root / "pairrun"),
                            "--sensor-run", str(root / "sensorrun"), "--code-commit", CODE, "--env-out", str(root / "again.env")], capture_output=True, text=True, cwd=str(board_dir))
    assert again.returncode != 0 and "exists" in again.stderr


A100_DOC = SR / "calibrate" / "runs" / "cluster_a100_replication_20261004" / "a100_j17" / "run" / "calibration_sm_80_be460c50_store_stream_rates_store_stream_rates.json"


@pytest.mark.skipif(not A100_DOC.is_file(), reason="archived A100 calibration is not in this tree")
def test_a100_hardware_path_runs_through_the_predictor_plumbing_only(mirror):
    """PLUMBING ONLY: the A100 cells, static tables, hardware rows and calibration document go through predict_board with a STAND-IN pair table (the H100's, copied in the mirror). The values are not used for
    anything; this only shows that nothing in the A100 path refuses for a reason other than the missing A100 pair table (which the in-job run supplies)."""
    root, board_dir, r = mirror
    assert r.returncode == 0
    a100 = board_dir / "a100"
    shutil.copyfile(board_dir / "h100" / CS.pair_file("h100"), a100 / CS.pair_file("a100"))
    for prof, want in (("main", 124), ("tensor", 44)):
        out = root / ("plumb_a100_%s.json" % prof)
        p = subprocess.run([sys.executable, str(board_dir / "predict_board.py"), "--board", "a100", "--profile", prof, "--calibration", str(A100_DOC), "--calibration-source", "plumbing", "--plumbing-test",
                            "--only", "/", "--out", str(out)], capture_output=True, text=True, cwd=str(board_dir))
        assert p.returncode != 0 and "selects" in p.stderr            # --only is limited to 1 to 4 cells in plumbing mode: the real path is exercised below without --only
    # the real path without the plumbing flag: a frozen-name prediction file in the MIRROR (never the repo), from the stand-in table
    for prof, want in (("main", 124), ("tensor", 44)):
        p = subprocess.run([sys.executable, str(board_dir / "predict_board.py"), "--board", "a100", "--profile", prof, "--calibration", str(A100_DOC), "--calibration-source", "plumbing in a mirror with a stand-in pair table"],
                           capture_output=True, text=True, cwd=str(board_dir))
        assert p.returncode == 0, p.stderr[-1200:]
        d = json.loads((a100 / CS.names("a100", prof)["predictions"]).read_text())
        assert d["coverage"]["cells"] == want and d["coverage"]["predicted"] >= want - 6
        assert d["calibration"]["uuid"].startswith("GPU-be460c50") and d["l2_capacity_bytes_used_by_the_capacity_rule"] == 41943040 and d["sm_count_used"] == 108


def test_stale_copies_inside_the_run_directories_are_ignored_and_ambiguity_still_refuses(mirror, tmp_path):
    root, board_dir, r = mirror
    assert r.returncode == 0, r.stderr[-1500:]
    sys.path.insert(0, str(board_dir))
    import importlib
    IJ = importlib.import_module("injob_freeze")
    assert IJ.find_calibration_document(root / "calrun").parent.name == "run"
    assert IJ.window_decision(BOARD, root / "sensorrun")[0].parent.name == "h100_1"
    assert IJ.window_decision(BOARD, root / "sensorrun" / "h100_1")[0].name == "power_sensor_results.json"      # the run directory itself is accepted too
    two = tmp_path / "two"
    for d in ("a", "b"):
        (two / d).mkdir(parents=True)
        shutil.copy(SENSOR, two / d)
    with pytest.raises(SystemExit, match="found 2"):                                                          # two real run directories are still an ambiguity, not silently resolved
        IJ.window_decision(BOARD, two)
    with pytest.raises(SystemExit, match="found 0"):
        IJ.window_decision(BOARD, tmp_path / "none")
