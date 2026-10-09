"""CPU-only tests of the Ada replication plumbing (no GPU, no ssh, no measured value). Run: python -m pytest replication_ada_rtx5000 -x -q
Tests that need the static tables (built by static_ada.py) skip until they exist."""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import board_ada as B  # noqa: E402
import cell_sets_ada as CS  # noqa: E402
import make_cells_ada as MA  # noqa: E402

PY = sys.executable
STATIC_READY = all((HERE / "static" / ("static_support_%s.json" % g)).is_file() for g in MA.GROUPS)
BW_DOC = B.SR / "calibrate" / "runs" / "energy_repeat_20261002" / "run_full" / "calibration_sm_120_0e63baea.json"
ADA_SOURCES = ["board_ada.py", "make_cells_ada.py", "cell_sets_ada.py", "static_ada.py", "predict_ada.py", "score_ada.py", "freeze_ada.py", "make_build_manifest.py", "build_ada_eval_sass.py"]


def run(*args):
    return subprocess.run([PY] + list(args), capture_output=True, text=True, cwd=str(HERE))


# ------------------------------------------------------------------------------------------------ cells
def test_cell_files_reproduce_byte_for_byte():
    r = run("make_cells_ada.py", "--check")
    assert r.returncode == 0, r.stderr


def test_counts_per_group_and_set():
    n = {g: len(MA.load_group(g)["cells"]) for g in MA.GROUPS}
    assert n == dict(samples=76, ml=40, tensor=16, validation=12, prospective=24)
    s = {}
    for c in MA.load_cells():
        s[c["set"]] = s.get(c["set"], 0) + 1
    assert s == dict(unseen=32, set_e=16, set_d=28, fresh_f=40, fresh_g=8, fresh_h=8, validation=12, prospective=24)
    assert all(not MA.load_group(g)["unplaceable"] for g in MA.GROUPS)


def test_hardware_values_are_adas_and_never_blackwells():
    hw, _ = B.read_ada_hardware()
    assert (hw["sm_count"], hw["l2_bytes"], hw["max_blocks_per_sm"]) == (100, 67108864, 24)
    bw_l2, bw_sm = B.forbidden_blackwell_values()
    for g in MA.GROUPS:
        d = MA.load_group(g)
        v = d["hardware_from_ground_truth"]["values"]
        assert v["l2_bytes"] == hw["l2_bytes"] != bw_l2 and v["sm_count"] == hw["sm_count"] != bw_sm
        assert d["derivation"]["grid_candidates"] == dict(c1=64, c2=128)
    for f in ADA_SOURCES:
        text = (HERE / f).read_text(encoding="utf-8")
        assert not re.search(r"\b134217728\b|\b188\b", text), "%s contains a Blackwell hardware literal" % f
    assert B.power_limit_w() == 250.0


def test_every_footprint_is_in_its_intended_tier():
    hw, _ = B.read_ada_hardware()
    l2 = hw["l2_bytes"]
    for c in MA.load_cells():
        ratio = c["footprint_bytes"] / l2
        assert not (0.4 <= ratio < 1.5), c["cell_id"]
        assert c["tier"] == ("L2" if ratio < 1 else "DRAM"), c["cell_id"]
        assert abs(c["footprint_over_l2"] - ratio) < 1e-12
    # intended tier = the Blackwell cell's tier of the same operator and regime (validation, prospective, machine-learning, tensor); samples: regime rule
    bw_val, bw_prosp = MA._blackwell_cells()
    want = {(c["operator_id"], c["regime"]): c["tier"] for c in bw_val + bw_prosp}
    for c in MA.load_cells(("validation", "prospective")):
        assert c["tier"] == want[(c["operator_id"], c["regime"])], c["cell_id"]
    for c in MA.load_cells(("ml", "tensor")):
        assert c["tier"] == ("L2" if c["regime"] in ("small", "medium") else "DRAM"), c["cell_id"]
    for c in MA.load_group("samples")["cells"]:
        assert c["tier"] == ("DRAM" if c["regime"] in ("xlarge", "dram") else "L2"), c["cell_id"]


def test_cell_ids_are_ada_and_unique():
    ids = [c["cell_id"] for c in MA.load_cells()]
    assert len(ids) == len(set(ids)) == 168 and all(i.startswith("ada/") for i in ids)


def test_vector_add_is_included_and_has_a_timing_entry():
    va = [c for c in MA.load_group("samples")["cells"] if c["family"] == "vecadd"]
    assert len(va) == 4 and {c["tier"] for c in va} == {"L2", "DRAM"}
    assert all(c["footprint_bytes"] == 12 * c["geometry"]["n"] and c["kernels"][0]["kid"] == "d_vecAdd" for c in va)


def test_tensor_profile_split():
    main, tens = CS.load_cells("main"), CS.load_cells("tensor")
    assert (len(main), len(tens)) == (124, 44) and not ({c["cell_id"] for c in main} & {c["cell_id"] for c in tens})


# ------------------------------------------------------------------------------------------------ build manifest
def test_build_manifest_reproduces_and_changes_only_arch_and_nvcc():
    r = run("make_build_manifest.py", "--check")
    assert r.returncode == 0, r.stderr
    m = json.loads((HERE / "build_manifest_ada.json").read_text(encoding="utf-8"))
    assert len(m["builds"]) == 12
    for b in m["builds"]:
        for s in b["steps"]:
            cmd = s["cmd"]
            assert cmd[0] == "/usr/local/cuda-13.2/bin/nvcc", b["id"]
            assert "-arch=sm_89" in cmd and not any("sm_120" in x for x in cmd), b["id"]
            assert not any("cuda-12" in x or "cuda-13.1" in x for x in cmd)
    ids = {b["id"] for b in m["builds"]}
    assert {"driver_ml", "driver_prosp", "run_e_sp", "run_e_fwt", "copy_runner", "transpose_runner", "reduction_runner", "tile_family"} <= ids
    assert sum(1 for b in m["builds"] if b["static_input"]) == 2
    assert "driver_ml.sass" in m["static_pending"]


def test_build_script_dry_run_lists_every_build():
    r = run("build_ada_eval_sass.py", "--samples", "/S", "--work", "/W", "--out", "/O", "--dry-run")
    assert r.returncode == 0 and sum(1 for l in r.stdout.splitlines() if re.match(r"# \w+ \(", l)) == 12 and "-arch=sm_89" in r.stdout


def test_build_script_refuses_without_the_ada_nvcc():
    r = run("build_ada_eval_sass.py", "--samples", "/S", "--work", "/W", "--out", "/O")
    assert r.returncode != 0 and "REFUSED" in (r.stderr + r.stdout)


# ------------------------------------------------------------------------------------------------ static
def test_prospective_kernel_table_matches_the_prospective_library():
    code = ("import sys; sys.path.insert(0, r'%s'); import static_ada as S; sys.path.insert(0, r'%s'); import prosp_lib as L; "
            "ok = all(L.UP.KERNELS[k]['needle'] == S.PROSP_KERNELS[k][0] and [tuple(x) for x in L.UP.KERNELS[k]['params']] == [tuple(x) for x in S.PROSP_KERNELS[k][1]] for k in S.PROSP_KERNELS) and set(L.UP.KERNELS) == set(S.PROSP_KERNELS); print(ok)"
            % (HERE, B.SR / "prospective_test"))
    r = run("-c", code)
    assert r.stdout.strip().endswith("True"), r.stderr[-500:]


def test_pending_cells_are_marked_not_dropped():
    if not STATIC_READY:
        pytest.skip("static tables not built")
    sup = json.loads((HERE / "static" / "static_support_ml.json").read_text(encoding="utf-8"))
    state = json.loads(json.dumps({}))
    import static_ada as SA
    if SA.sass_state()["ml"][0]:
        pytest.skip("ml SASS exists: nothing pending")
    assert sup["cells"] == 40 and len(sup["pending_build"]) == 40 and not sup["unsupported"]
    assert all(v["pending_build"] and not v["supported"] for v in sup["rows"].values())


def test_base_cells_supported_or_listed():
    if not STATIC_READY:
        pytest.skip("static tables not built")
    sup = json.loads((HERE / "static" / "static_support_samples.json").read_text(encoding="utf-8"))
    assert sup["cells"] == 76 and not sup["pending_build"]
    assert sup["supported"] + len(sup["unsupported"]) == 76


# ------------------------------------------------------------------------------------------------ prediction plumbing
def test_class_mapped_opcodes_are_priced_not_refused():
    # Delegation decision (reversible until freeze): F2FP/FMNMX go through the shared
    # class mapping with Ada's class costs instead of being refused, keeping Ada no
    # stricter than Blackwell. The helper therefore reports no unmeasured opcodes...
    row = dict(kernels=[dict(phases=[dict(issue_warp_instructions={"FFMA": 10, "FMNMX": 4, "F2FP.BF16.F32.PACK_AB": 0, "F2FP.BF16.PACK_AB": 3}), dict(issue_warp_instructions={"IADD3": 1})])])
    assert CS.unmeasured_opcodes(row) == {}
    assert CS.unmeasured_opcodes(dict(kernels=[dict(phases=[dict(issue_warp_instructions={"FFMA": 2})])])) == {}
    # ...while the shared mapping prices them: FMNMX in the floating-point-other
    # family, F2FP forms in the catch-all bucket (same mapping as Blackwell).
    sys.path.insert(0, str(HERE.parent))
    import extract_features as X
    import predict_runtime as PR
    assert X.op_family("FMNMX")[0] == "fp_other"
    assert X.op_family("F2FP.BF16.F32.PACK_AB")[0] == "other"
    assert PR.issue_cost("FMNMX") == 0.5 and PR.issue_cost("F2FP.BF16.F32.PACK_AB") == 0.25


def test_predict_refuses_without_calibration_or_with_wrong_inputs(tmp_path):
    if not STATIC_READY:
        pytest.skip("static tables not built")
    r = run("predict_ada.py", "--out", str(tmp_path / "x.json"))
    assert r.returncode == 2 and "no calibration document" in r.stderr
    r = run("predict_ada.py", "--calibration", str(tmp_path / "missing.json"), "--out", str(tmp_path / "x.json"))
    assert r.returncode == 2 and "does not exist" in r.stderr
    if BW_DOC.is_file():
        r = run("predict_ada.py", "--calibration", str(BW_DOC), "--calibration-source", "t", "--out", str(tmp_path / "x.json"))
        assert r.returncode == 2 and ("not an Ada one" in r.stderr or "pending build" in r.stderr), r.stderr
        r = run("predict_ada.py", "--calibration", str(BW_DOC), "--plumbing-test", "--only", "ada/unseen_cuda_samples_scan/small/c1", "--out", str(tmp_path / "predictions_ada.json"))
        assert r.returncode == 2 and "must not be called" in r.stderr
        r = run("predict_ada.py", "--calibration", str(BW_DOC), "--plumbing-test", "--out", str(tmp_path / "x.json"))
        assert r.returncode == 2 and "--only" in r.stderr


def test_predict_refuses_without_ada_pair_constants(monkeypatch, tmp_path):
    import predict_ada as PA
    monkeypatch.setattr(B, "PAIR_CONSTANTS", tmp_path / "nope.json")
    with pytest.raises(PA.Refusal, match="no fallback to the Blackwell file"):
        PA.load_pair_constants()
    bw = B.SR / "constants" / "pair_overlap_constants.json"
    monkeypatch.setattr(B, "PAIR_CONSTANTS", bw)
    with pytest.raises(PA.Refusal):
        PA.load_pair_constants()          # byte-identical to itself / names no Ada source
    monkeypatch.setattr(B, "PAIR_CONSTANTS", HERE.parent / "constants" / "pair_overlap_constants_ada.json")
    if B.PAIR_CONSTANTS.is_file():
        assert PA.load_pair_constants()["schema"] == "pair_overlap_constants/1"


def test_predict_never_reads_the_blackwell_pair_file():
    text = (HERE / "predict_ada.py").read_text(encoding="utf-8")
    assert "pair_overlap_constants.json" in text and text.count("B.PAIR_CONSTANTS") >= 3
    assert B.PAIR_CONSTANTS.name == "pair_overlap_constants_ada.json"


def test_plumbing_prediction_on_one_base_cell_with_a_blackwell_document(tmp_path):
    cell = "ada/unseen_cuda_samples_scan/small/c1"
    if not STATIC_READY or not BW_DOC.is_file() or not (B.SR / "constants" / "pair_overlap_constants_ada.json").is_file():
        pytest.skip("static tables, Blackwell document or Ada pair file missing")
    sup = json.loads((HERE / "static" / "static_support_samples.json").read_text(encoding="utf-8"))["rows"]
    if not sup[cell]["supported"]:
        pytest.skip("cell not supported")
    out = tmp_path / "plumbing.json"
    r = run("predict_ada.py", "--calibration", str(BW_DOC), "--plumbing-test", "--only", cell, "--out", str(out), "--allow-incomplete")
    if r.returncode != 0 and "pending build" in r.stderr:
        pytest.skip("profile has pending cells: " + r.stderr[:80])
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["kind"] == "plumbing_test_not_ada" and d["warning"]
    assert d["pair_overlap_constants"]["file"].endswith("pair_overlap_constants_ada.json")
    assert d["l2_capacity_bytes_used_by_the_capacity_rule"] == 67108864 and d["sm_count_used"] == 100
    assert d["cells"][cell]["status"] == "predicted" and d["cells"][cell]["runtime_s"] > 0


# ------------------------------------------------------------------------------------------------ scoring and freeze
def _synthetic(tmp_path, cells, preds, timing_rows, uuid="GPU-4640d904-c754-5243-df67-37761e91b400"):
    import hashlib
    pred = dict(kind="ada_frozen_predictions", profile="main", warning=None, calibration=dict(uuid=uuid, energy_cap_w=250.0, source="t", file="f", sha256="0"), roofline_runtime_s={c: 1e-4 for c in preds}, cells=preds)
    p = tmp_path / "predictions_ada.json"
    p.write_text(json.dumps(pred, sort_keys=True))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    rec = tmp_path / "freeze_record_ada.json"
    rec.write_text(json.dumps(dict(predictions_sha256=h, freeze_commit="a" * 40)))
    t = tmp_path / "timing.json"
    t.write_text(json.dumps(dict(meta=dict(freeze=dict(predictions_sha256=h), gpu=dict(uuid=uuid)), cells=timing_rows)))
    return p, rec, t


def test_score_counts_unsupported_and_refused_as_failures(tmp_path):
    import score_ada as SC
    cells = [dict(cell_id="ada/x/%d" % i, set="unseen", group="samples", family="matmul", tier="L2") for i in range(3)]
    preds = {"ada/x/0": dict(status="predicted", runtime_s=1.1e-4), "ada/x/1": dict(status="predicted", runtime_s=0.8e-4),
             "ada/x/2": dict(status="not_predicted", reason="refused: per-wave L2 working set exceeds 64 MiB", counted_as_failure=True)}
    timing = [dict(cell_id=c["cell_id"], correct=True, per_launch_runtime_s=1e-4) for c in cells]
    p, rec, t = _synthetic(tmp_path, cells, preds, timing)
    rep = SC.score(p, rec, t, cells_list=cells)
    rt = rep["runtime"]
    assert rt["supported_cells"]["n"] == 2 and abs(rt["supported_cells"]["median_pct"] - 15.0) < 1e-6
    assert rt["unsupported_as_failure"]["failures"] == 1 and rt["unsupported_as_failure"]["median_pct"] == 20.0
    assert rep["power_limit_w"] == 250.0 and rep["criteria"]["runtime_median_le_15_supported"] is True


def test_score_refuses_mismatched_freeze(tmp_path):
    import score_ada as SC
    cells = [dict(cell_id="ada/x/0", set="unseen", group="samples", family="matmul", tier="L2")]
    p, rec, t = _synthetic(tmp_path, cells, {"ada/x/0": dict(status="predicted", runtime_s=1e-4)}, [dict(cell_id="ada/x/0", correct=True, per_launch_runtime_s=1e-4)])
    p.write_text(p.read_text() + " ")
    with pytest.raises(SystemExit, match="differs from the sha256"):
        SC.score(p, rec, t, cells_list=cells)


def test_score_below_limit_rule_uses_95_percent_of_250w():
    import score_h100 as S
    assert S.BELOW_CAP_FRACTION == 0.95 and 0.95 * B.power_limit_w() == 237.5


def test_freeze_record_refuses_overwrite_and_missing(tmp_path):
    import freeze_ada as FZ
    with pytest.raises(SystemExit, match="never overwritten"):
        FZ.write_record("tensor")
    assert set(CS.FROZEN_NAMES) >= {"predictions_ada.json", "freeze_record_ada.json", "baselines_ada.json"}


def test_sass_manifest_of_existing_ada_sass_reproduces_and_lists_pending():
    r = run("make_sass_manifest_ada.py", "--check")
    assert r.returncode == 0, r.stderr
    m = json.loads((HERE / "sass_manifest_ada.json").read_text(encoding="utf-8"))
    assert m["arch"] == "sm_89" and "d_vecAdd" in m["kernels"] and "mm16" in m["kernels"]
    import static_ada as SA
    if not SA.sass_state()["ml"][0]:
        assert "at4" in m["pending_build"] and "tc128" in m["pending_build"]
