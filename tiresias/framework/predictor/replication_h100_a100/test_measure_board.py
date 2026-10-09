"""CPU tests of the replication's profile split, engine routing, freeze/window/SASS gates and refusals (no GPU, no network). Run: python -m pytest -q test_measure_board.py"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_sets_board as CS  # noqa: E402
import freeze_board as FZ  # noqa: E402
import measure_board as M  # noqa: E402
import predict_board as PB  # noqa: E402

BOARD = "h100"
POWER = HERE.parent / "h100_power_sensor_test/runs/h100_j16/power_sensor_results.json"


def all_cells():
    return CS.load_cells(BOARD, "main") + CS.load_cells(BOARD, "tensor")


def test_profiles_partition_the_168_cells():
    main, tensor = CS.load_cells(BOARD, "main"), CS.load_cells(BOARD, "tensor")
    ids = [c["cell_id"] for c in main + tensor]
    assert (len(main), len(tensor)) == (124, 44)
    assert len(ids) == len(set(ids)) == 168
    assert not {k["kid"] for c in main for k in c["kernels"]} & CS.TENSOR_KIDS


def test_every_cell_has_an_engine_and_validation_cells_follow_their_family():
    engines = {c["cell_id"]: CS.engine_of(c) for c in all_cells()}
    assert set(engines.values()) <= set(M.ENGINES)
    val = {c["cell_id"]: CS.engine_of(c) for c in all_cells() if c["set"] == "validation"}
    assert len(val) == 12
    for c in all_cells():
        if c["set"] == "validation":
            assert CS.engine_of(c) == {"sp": "set_e", "conv": "unseen"}.get(c["family"], "fresh")
    assert sum(1 for c in all_cells() if CS.engine_of(c) == "prosp") == 24


def test_by_engine_loses_no_cell():
    cells = all_cells()
    assert sum(len(v) for v in M.by_engine(cells).values()) == len(cells)


def test_unknown_family_is_refused():
    with pytest.raises(SystemExit):
        CS.engine_of(dict(cell_id="x", family="nonsense"))


def test_names_are_per_board_and_profile():
    a, b = CS.names("h100", "main"), CS.names("h100", "tensor")
    assert a["predictions"] != b["predictions"] and a["record"] != b["record"] and a["sass_manifest"] == b["sass_manifest"]
    assert "h100" in a["predictions"] and "a100" in CS.names("a100", "main")["predictions"]


def test_window_needs_the_committed_h100_result_and_no_padding():
    ok = M.check_window(POWER, 40, 0, board="h100", use_git=False)
    assert ok["min_load_s_at_padding_0"] == 40.0
    with pytest.raises(SystemExit):
        M.check_window(POWER, 15, 0, board="h100", use_git=False)          # shorter than admitted
    with pytest.raises(SystemExit):
        M.check_window(POWER, 40, 5, board="h100", use_git=False)          # padding unsupported
    with pytest.raises(SystemExit):
        M.check_window(POWER, 40, 0, board="a100", use_git=False)          # an H100 result is not an A100 one
    with pytest.raises(SystemExit):
        M.check_window(HERE / "nope.json", 40, 0, board="h100", use_git=False)


def test_energy_run_without_window_validation_refuses():
    r = subprocess.run([sys.executable, str(HERE / "measure_board.py"), "run", "--board", BOARD, "--profile", "main", "--stage", "all", "--measure", "both", "--dry-run"], capture_output=True, text=True)
    assert r.returncode == 2 and "window-validation" in r.stderr


def test_run_refuses_without_frozen_predictions(tmp_path):
    r = subprocess.run([sys.executable, str(HERE / "measure_board.py"), "check-freeze", "--board", BOARD, "--profile", "main", "--tree", str(tmp_path), "--hipc-commit", "a" * 40], capture_output=True, text=True)
    assert r.returncode == 2 and "predictions must be frozen" in r.stderr


def test_freeze_record_check_detects_a_changed_input(tmp_path):
    n = CS.names(BOARD, "main")
    (tmp_path / n["predictions"]).write_text("{}")
    (tmp_path / "other.json").write_text("x")
    rec = dict(schema=FZ.SCHEMA, board=BOARD, profile="main", freeze_commit="b" * 40, predictions_sha256=FZ.sha256_file(tmp_path / n["predictions"]), sha256={"other.json": FZ.sha256_file(tmp_path / "other.json")})
    (tmp_path / n["record"]).write_text(json.dumps(rec))
    assert FZ.check(BOARD, "main", tree=tmp_path)["freeze_commit"] == "b" * 40
    (tmp_path / "other.json").write_text("changed")
    with pytest.raises(SystemExit):
        FZ.check(BOARD, "main", tree=tmp_path)


def test_freeze_record_is_never_written_over_an_existing_file(tmp_path):
    existing = tmp_path / "x.json"
    existing.write_text("{}")
    with pytest.raises(SystemExit):
        FZ.write_record(BOARD, "tensor", out=existing)      # refused whether or not the predictions are committed yet: an existing output is never overwritten
    assert existing.read_text() == "{}"


def test_sass_gate_accepts_identical_and_refuses_differing_sequences(monkeypatch):
    man = {"k1": dict(symbol="_Z2k1v", instruction_sequence_sha256="aa" * 32), "k2": dict(symbol="_Z2k2v", instruction_sequence_sha256="bb" * 32)}
    monkeypatch.setattr(M, "parse_sass_text", lambda text: {"same": "aa" * 32, "diff": "cc" * 32}[text.split()[0]])
    runs = {"_Z2k1v": "same /* ok */", "_Z2k2v": "diff /* ok */"}

    def runner(cmd, **kw):
        sym = cmd[cmd.index("-fun") + 1]
        return subprocess.CompletedProcess(cmd, 0, stdout=runs[sym], stderr="")
    out = M.sass_gate(BOARD, {"k1": "b1", "k2": "b2", "k3": "b3"}, "nvcc", manifest=man, runner=runner)
    assert out["k1"]["ok"] and not out["k2"]["ok"] and not out["k3"]["ok"] and out["k3"]["detail"] == "no manifest entry"
    ok, refused = M.cells_without_failed_kernels([dict(cell_id="c1", kernels=[dict(kid="k1")]), dict(cell_id="c2", kernels=[dict(kid="k2")])], out)
    assert [c["cell_id"] for c in ok] == ["c1"] and refused[0]["cell_id"] == "c2"


def test_gpu_identity_refuses_wrong_board_and_ambiguous_selection():
    def smi(rows):
        return lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="\n".join(rows) + "\n", stderr="")
    uuid = "GPU-6132e1b7-a6ba-9eea-9084-bcb0ec15009a"
    assert M.gpu_identity("h100", smi(["0, %s, NVIDIA H100 80GB HBM3" % uuid]), environ={})["uuid"] == uuid
    with pytest.raises(SystemExit):
        M.gpu_identity("h100", smi(["0, %s, NVIDIA A100-SXM4-80GB" % uuid]), environ={})
    with pytest.raises(SystemExit):
        M.gpu_identity("h100", smi(["0, %s, NVIDIA H100 80GB HBM3" % uuid, "1, %s, NVIDIA H100 80GB HBM3" % uuid.replace("6132", "6133")]), environ={})


def test_scratch_paths_are_refused():
    with pytest.raises(SystemExit):
        M.refuse_scratch("/scratch/shareduser/x", "--workdir")
    M.refuse_scratch("/home/user/x", "--workdir")


def test_predictor_refuses_blackwell_pair_table_and_missing_table(tmp_path):
    with pytest.raises(PB.Refusal):
        PB.load_pair_constants(PB.SR / "constants" / "pair_overlap_constants.json")
    with pytest.raises(PB.Refusal):
        PB.load_pair_constants(tmp_path / "missing.json")
    d = PB.load_pair_constants(HERE / BOARD / "pair_overlap_constants.json")
    assert d["resident_warps_per_sm"] == [4, 8, 16, 32] and len(d["beta_hmma"]) == 4


def test_predictor_refuses_a_document_of_another_board():
    hw = {"sm_count": 132, "l2_bytes": 52428800}
    doc = dict(device=dict(name="NVIDIA RTX PRO 6000 Blackwell", sm_count=188, l2_bytes=134217728, compute_capability="12.0"))
    assert len(PB.check_document(doc, hw, "h100")) == 4


def test_sass_manifest_covers_every_kernel_of_every_cell():
    man = json.loads((HERE / BOARD / CS.names(BOARD, "main")["sass_manifest"]).read_text())
    kids = {k["kid"] for c in all_cells() for k in c["kernels"]}
    assert kids <= set(man["kernels"]) and not man["pending_build"]


def test_check_imports_passes_in_the_repo_and_reports_a_missing_file(monkeypatch):
    assert set(M.check_imports(BOARD).values()) == {"ok"}
    monkeypatch.setattr(M, "load_prosp_engine", lambda: (_ for _ in ()).throw(ModuleNotFoundError("tile_family")))
    with pytest.raises(SystemExit) as ex:
        M.check_imports(BOARD)
    assert "prospective engine" in str(ex.value) and "tile_family" in str(ex.value)


def test_vector_header_matches_the_validated_one_and_is_scoped_to_the_copy_runner_build(tmp_path, monkeypatch):
    import build_drivers as BDV
    assert M.VECTOR_HOST_HEADER == BDV.VECTOR_HOST_HEADER
    seen = {}

    class Runner:
        def __init__(self, name):
            self.name = name

        def find_nvcc(self):
            return "nvcc"

        def build_binary(self, root, wd, nvcc):
            seen[self.name] = (__import__("os").environ.get("NVCC_PREPEND_FLAGS"), (wd / "host_compat" / "cuda" / "cmath").is_file())
            return wd / "bin"

    class T:
        RUNNER_BINARY = {"copy_runner": "copy_driver", "transpose_runner": "transpose_driver"}
    cells = [dict(timing=dict(runner="copy_runner")), dict(timing=dict(runner="transpose_runner"))]
    monkeypatch.delenv("NVCC_PREPEND_FLAGS", raising=False)
    M.set_d_build(T, {"copy_runner": Runner("copy_runner"), "transpose_runner": Runner("transpose_runner")}, tmp_path, tmp_path / "w", cells)
    assert "-I" in seen["copy_runner"][0] and seen["copy_runner"][1]
    assert seen["transpose_runner"] == (None, False)                         # other runners get no extra flags
    assert "NVCC_PREPEND_FLAGS" not in __import__("os").environ               # restored afterwards


def test_energy_stage_hands_the_fresh_and_prospective_engines_a_correct_flag():
    src = (HERE / "measure_board.py").read_text()
    assert "dict(cell_id=k, per_launch_runtime_s=v, correct=True)" in src and "dict(cell_id=k, per_launch_runtime_s=v)" not in src.replace("correct=True)", "")


def test_energy_coverage_flags_a_cell_that_was_timed_but_skipped(tmp_path):
    for sub, rows in (("energy_fresh", []), ("energy_unseen", [("op_a", "small", "c1")])):
        d = tmp_path / sub
        d.mkdir()
        with (d / "application_energy_raw.csv").open("w", newline="") as f:
            import csv
            w = csv.DictWriter(f, fieldnames=["parent_id", "regime", "candidate_id"])
            w.writeheader()
            for op, rg, cd in rows:
                w.writerow(dict(parent_id=op, regime=rg, candidate_id=cd))
    rej = tmp_path / "energy_unseen" / "application_energy_rejected.csv"
    rej.write_text("parent_id,regime,candidate_id\nop_b,small,c2\n")
    expected = ["h100/op_a/small/c1", "h100/op_b/small/c2", "h100/op_c/large/c1"]
    missing, n = M.energy_coverage(tmp_path, expected, "h100")
    assert missing == ["h100/op_c/large/c1"] and n == 2                      # a recorded rejection counts as covered; the silently skipped cell does not


def test_engines_filter_and_reuse_timing_options_exist():
    p = M.build_parser()
    a = p.parse_args(["run", "--board", "h100", "--profile", "main", "--stage", "all", "--measure", "both", "--engines", "fresh", "--reuse-timing", "x.json"])
    assert a.engines == "fresh" and a.reuse_timing == "x.json"
