#!/usr/bin/env python3
"""Frozen A100/H100 predictions for the complete 168-cell Blackwell-suite replication. CPU only: no kernel is executed and no measured value of any evaluation cell is read.

    python predict_board.py --board h100 --profile main|tensor --calibration <document .json> --calibration-source "<job>" [--out <name>] [--allow-incomplete]
    python predict_board.py --board h100 --calibration <document .json> --plumbing-test --only <substring>     # 1-4 cells, never a frozen name

Runtime: the CURRENT runtime model of CURRENT_MODELS.md (predict_runtime_v3k.py: shared-traffic model with grid-wide DRAM footprint and per-wave L2-capacity refusal, bank-conflict and partial
overlap stack, pair-overlap / stage-serialisation / busiest-SM rules for tensor-core phases), run through calibrate/cal/portable_predict.py with the constants of the calibration document.
Pair-overlap constants: EXACTLY <board>/pair_overlap_constants.json (the board's own dependent fragment-load microbenchmark). They replace the Blackwell table in memory for the duration of the
call only; the Blackwell file is never read for a prediction here, nothing in predict_runtime_v3k.py is edited, and the predictor refuses a missing, malformed or Blackwell-identical table.
Energy: current component energy model with traffic-based byte columns (calibrate/cal/traffic.py), calibrator-only profile rates of the same document.
Refusals (unsupported static support, per-wave L2 capacity, DRAM tier without footprint) stay in the list and count as failures; nothing is dropped, shrunk or priced at another rate.

Refuses (exit 2, nothing written) when: no document or it does not exist; its device is not this board (name, SM count, L2 size, compute capability against HARDWARE_GROUND_TRUTH.md); the document is
incomplete (unless --allow-incomplete, marked); any cell is pending build; the pair table is missing/invalid; the output exists; --calibration-source is empty (frozen run).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import board as BD
import cell_sets_board as CS

HERE = BD.HERE
SR = BD.SR
CAL = SR / "calibrate"
for _p in (CAL, SR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SCHEMA = "cluster_predictions/1"
GROUPS = ("samples", "ml", "tensor", "validation", "prospective")
STATIC_KINDS = (("features", "features"), ("phases", "phases"), ("phases_unique", "unique"), ("bank_conflicts", "bank"), ("static_support", "support"), ("footprints", "footprints"))
MODEL_FILES = ["predict_runtime_v2.py", "predict_runtime_v3.py", "predict_runtime_v3f.py", "predict_runtime_v3h.py", "predict_runtime_v3i.py", "predict_runtime_v3j.py", "predict_runtime_v3k.py",
               "calibrate/predict.py", "calibrate/cal/portable_predict.py", "calibrate/cal/document.py", "calibrate/cal/energy.py", "calibrate/cal/traffic.py",
               "port_common/port_ext.py", "port_common/port_ml_ext.py", "port_common/set_d_port.py", "shared_traffic/footprint.py"]


class Refusal(RuntimeError):
    pass


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_name(board, profile="main"):
    return CS.names(board, profile)["predictions"]


def load_pair_constants(path):
    path = Path(path)
    if not path.is_file():
        raise Refusal("%s is missing: the pair-overlap table comes from this board's own dependent fragment-load microbenchmark; there is no fallback to the Blackwell file" % path)
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("schema") != "pair_overlap_constants/1" or not (len(d.get("resident_warps_per_sm", [])) == len(d.get("beta_hmma", [])) > 1):
        raise Refusal("%s is not a pair_overlap_constants/1 document" % path.name)
    bw = SR / "constants" / "pair_overlap_constants.json"
    if bw.is_file() and sha(path) == sha(bw):
        raise Refusal("%s is byte-identical to the Blackwell pair-overlap table" % path.name)
    if "replication_20261004" not in str(d.get("source", "")):
        raise Refusal("the source of %s (%r) does not name a replication-campaign microbenchmark" % (path.name, d.get("source")))
    return d


def check_document(doc, hw, board):
    d = doc.get("device") or {}
    want_name, want_cc = {"h100": ("H100", "9.0"), "a100": ("A100", "8.0")}[board]
    problems = []
    if want_name not in str(d.get("name", "")):
        problems.append("device name %r is not an %s" % (d.get("name"), want_name))
    if d.get("sm_count") != hw["sm_count"]:
        problems.append("SM count %r differs from HARDWARE_GROUND_TRUTH.md (%d)" % (d.get("sm_count"), hw["sm_count"]))
    if d.get("l2_bytes") != hw["l2_bytes"]:
        problems.append("L2 size %r differs from HARDWARE_GROUND_TRUTH.md (%d)" % (d.get("l2_bytes"), hw["l2_bytes"]))
    if str(d.get("compute_capability")) != want_cc:
        problems.append("compute capability %r is not %s" % (d.get("compute_capability"), want_cc))
    return problems


def load_static(board):
    """-> cells (list), and per-kind dicts over every group. Pending-build cells are reported by the caller."""
    b = BD.binding(board)
    cells, tabs = [], {k: {} for _, k in STATIC_KINDS}
    feats = None
    for g in GROUPS:
        cells += json.loads((b.HERE / ("cells_%s.json" % g)).read_text(encoding="utf-8"))["cells"]
        for fname, kind in STATIC_KINDS:
            f = b.HERE / "static" / ("%s_%s.json" % (fname, g))
            if not f.is_file():
                raise Refusal("%s is missing: run static_tables.py --board %s to completion first" % (f.relative_to(b.HERE), board))
            d = json.loads(f.read_text(encoding="utf-8"))
            if kind == "features":
                feats = dict(d, rows=(feats["rows"] if feats else []) + d["rows"])
            else:
                tabs[kind].update(d["rows"])
    return b, cells, feats, tabs


def run(a):
    import predict as P
    from cal import document as D, portable_predict as PP, traffic as T
    import predict_runtime_v3j as J
    import predict_runtime_v3k as K3
    board = a.board
    b, all_cells, feats_all, tabs = load_static(board)
    all_cells = [c for c in all_cells if CS.needs_tensor(c) == (a.profile == "tensor")]
    if a.calibration is None or not Path(a.calibration).is_file():
        raise Refusal("a calibration document is required (and must exist); there is no default and no fallback to another board")
    cal_path = Path(a.calibration)
    out = Path(a.out) if a.out else b.HERE / frozen_name(board, a.profile)
    if not a.plumbing_test and not (a.calibration_source or "").strip():
        raise Refusal("--calibration-source is required: name the cluster job of the FIRST complete calibration of this campaign")
    if out.exists():
        raise Refusal("%s exists; never overwritten" % out)
    if a.plumbing_test:
        if out.name in {frozen_name(board, p) for p in CS.PROFILES}:
            raise Refusal("--plumbing-test output must not be called %s" % out.name)
        if not a.only:
            raise Refusal("--plumbing-test needs --only <substring>")
    elif a.only:
        raise Refusal("--only is for --plumbing-test")
    pair_path = b.HERE / CS.pair_file(board)
    pair = load_pair_constants(pair_path)
    hw, _ = b.read_ada_hardware()
    doc = json.loads(cal_path.read_text(encoding="utf-8"))
    problems = check_document(doc, hw, board)
    if problems and not a.plumbing_test:
        raise Refusal("the calibration document is not this board's: " + "; ".join(problems))
    try:
        P.load_calibration(cal_path, allow_incomplete=a.allow_incomplete)
    except P.PredictRefusal as ex:
        raise Refusal(str(ex))
    support_all, fps_all = tabs["support"], tabs["footprints"]
    cell_ids = [c["cell_id"] for c in all_cells]
    if a.only:
        cell_ids = [c for c in cell_ids if a.only in c]
        if not 0 < len(cell_ids) <= 4:
            raise Refusal("--plumbing-test --only %r selects %d cells; it must select 1 to 4" % (a.only, len(cell_ids)))
    missing = [c for c in cell_ids if c not in support_all]
    pending = [c for c in cell_ids if c in support_all and support_all[c].get("pending_build")]
    if missing or pending:
        raise Refusal("%d cells have no static row and %d are pending build (first: %s)" % (len(missing), len(pending), (missing or pending)[0]))
    keep = set(cell_ids)
    features = dict(feats_all, rows=[r for r in feats_all["rows"] if r["cell_id"] in keep])
    phases, unique, bank, support, fps = ({k: v for k, v in tabs[kind].items() if k in keep} for kind in ("phases", "unique", "bank", "support", "footprints"))
    l2_capacity = int(hw["l2_bytes"])
    bw_l2, bw_sm = b.forbidden_blackwell_values()
    if l2_capacity == bw_l2 or hw["sm_count"] == bw_sm:
        raise Refusal("the hardware rows equal Blackwell's; refusing")
    if not a.plumbing_test and int(doc["device"]["l2_bytes"]) != l2_capacity:
        raise Refusal("the document's L2 differs from the hardware row")
    ok_ids = {c for c in cell_ids if support[c]["supported"]}
    f_ok = dict(features, rows=[r for r in features["rows"] if r["cell_id"] in ok_ids])
    with tempfile.TemporaryDirectory(prefix="cluster_legacy_") as tmp:
        D.export_legacy(doc, tmp)
        consts = PP.load_constants(tmp)
        consts_sha = {n: sha(Path(tmp) / n) for n in sorted(p.name for p in Path(tmp).iterdir())}
        unique_fp = J.attach_footprints({k: unique[k] for k in ok_ids}, {k: fps[k] for k in ok_ids}, l2_capacity)
        old_pair = K3._PAIR
        K3._PAIR = pair                      # this board's own table for the duration of the call; restored below
        try:
            rt_raw = K3.predict_portable(f_ok, {k: phases[k] for k in ok_ids}, unique_fp, {k: bank[k] for k in ok_ids}, consts, hw["sm_count"])
        finally:
            K3._PAIR = old_pair
    runtime = {cid: (v.get("primary_s") if isinstance(v, dict) else v) for cid, v in rt_raw.items()}
    traffic = {cid: v["traffic"] for cid, v in rt_raw.items() if isinstance(v, dict) and v.get("primary_s") and v.get("traffic")}
    en = T.energy_rows_traffic(doc, f_ok, runtime, traffic)
    energy_profile = doc["constants"]["energy"]["profile"]
    sl = doc["constants"].get("store_legacy") or {}
    need = ("t0_us", "L2_read_sector_TBps", "DRAM_read_TBps")
    if any(k not in sl for k in need):
        raise Refusal("the document lacks the store constants the roofline reference needs: %s" % ", ".join(k for k in need if k not in sl))
    cell_by_id = {c["cell_id"]: c for c in all_cells}
    roofline = {cid: sl["t0_us"] * 1e-6 + cell_by_id[cid]["logical_bytes_per_launch"] / ((sl["L2_read_sector_TBps"] if cell_by_id[cid]["tier"] == "L2" else sl["DRAM_read_TBps"]) * 1e12)
                for cid in cell_ids}
    tdoc = doc["constants"].get("tensor") or {}
    te, ti = tdoc.get("energy") or {}, tdoc.get("issue") or {}
    tensor_record = dict(present=bool(tdoc), status=tdoc.get("status"), energy_status=te.get("status"), energy_rate_pJ_per_lane_instruction=te.get("rate_pJ_per_lane_instruction"),
                         issue_cycles_per_warp_instruction_per_sm=ti.get("issue_cycles_per_warp_instruction_per_sm"), dependent_latency_cycles=ti.get("dependent_latency_cycles"),
                         rule="every tensor constant comes from the document's tensor stage only; a cell with tensor instructions is unsupported without them")
    if not (tdoc and te.get("status") == "ok" and ti.get("issue_cycles_per_warp_instruction_per_sm")):
        print("WARNING: the document has no usable tensor constants: tensor cells will be unsupported (failures)", file=sys.stderr)
    rows = {}
    for cid in cell_ids:
        if not support[cid]["supported"]:
            rows[cid] = dict(status="unsupported", reason=support[cid].get("reason"), counted_as_failure=True)
        elif en.get(cid, {}).get("status") != "ok":
            why = rt_raw[cid].get("unsupported_reason") if isinstance(rt_raw.get(cid), dict) else None
            rows[cid] = dict(status="not_predicted", reason=why or en.get(cid, {}).get("reason", "no runtime prediction"), runtime_s=runtime.get(cid), counted_as_failure=True)
        else:
            e = en[cid]
            rows[cid] = dict(status="predicted", runtime_s=e["runtime_s"], base_power_w=energy_profile["base_power_w"], energy_j=e["energy_j"], mean_power_w=e["mean_power_w"], capped=bool(e["capped"]),
                             base_term_j=e["base_term_j"], term_j=e["term_j"],
                             traffic_bytes=dict(l2_served=traffic[cid]["l2_read_bytes"] + traffic[cid]["l2_write_bytes"], dram=traffic[cid]["dram_read_bytes"] + traffic[cid]["dram_write_bytes"]))
    static_names = sorted("static/%s_%s.json" % (f, g) for f, _ in STATIC_KINDS for g in GROUPS) + ["cells_%s.json" % g for g in GROUPS]
    res = dict(
        schema=SCHEMA, board=board, profile=a.profile, kind="plumbing_test_not_frozen" if a.plumbing_test else "cluster_frozen_predictions",
        warning=("PLUMBING TEST ONLY on %d cells; never to be committed as %s" % (len(cell_ids), frozen_name(board, a.profile))) if a.plumbing_test else None,
        calibration=dict(source=(a.calibration_source or "").strip() or None, file=cal_path.name, sha256=sha(cal_path), device=doc["device"].get("name"), uuid=doc["device"].get("uuid"),
                         sm_count=doc["device"].get("sm_count"), complete=bool(doc.get("complete")), allow_incomplete_used=bool(a.allow_incomplete), created_utc=doc.get("created_utc"),
                         warnings=doc.get("warnings"), energy_profile_status=energy_profile.get("status"), energy_profile_base_power_w=energy_profile.get("base_power_w"),
                         energy_cap_w=doc["constants"]["energy"].get("cap_w"), legacy_constants_sha256=consts_sha),
        pair_overlap_constants=dict(file="%s/%s" % (board, CS.pair_file(board)), sha256=sha(pair_path), source=pair.get("source"), source_sha256=pair.get("source_sha256"),
                                    resident_warps_per_sm=pair["resident_warps_per_sm"], beta_hmma=pair["beta_hmma"]),
        runtime_model="current runtime model (CURRENT_MODELS.md): stage-rule model (predict_runtime_v3k.py) on the shared-traffic model with per-wave L2 refusal, this board's own pair-overlap table",
        energy_model="current component energy model with traffic-based byte columns: E = min(cap x t, base x t + sum(rate x column)), calibrator-only profile rates of the calibration document (calibrate/cal/traffic.py)",
        l2_capacity_bytes_used_by_the_capacity_rule=l2_capacity, sm_count_used=hw["sm_count"], tensor_stage=tensor_record,
        inputs_sha256={n: sha(b.HERE / n) for n in static_names},
        code_sha256={p: sha(SR / p) for p in MODEL_FILES} | {"replication_h100_a100/predict_board.py": sha(Path(__file__)), "replication_h100_a100/board.py": sha(Path(__file__).with_name("board.py"))},
        coverage=dict(cells=len(cell_ids), predicted=sum(1 for v in rows.values() if v["status"] == "predicted"),
                      not_predicted=sorted(c for c, v in rows.items() if v["status"] != "predicted")),
        note="No kernel was executed and no runtime, energy or power value of any evaluation cell was read. Refused and unsupported cells stay in the list and count as failures.",
        roofline_runtime_s=roofline, roofline_constants={k: sl[k] for k in need}, cells=rows)
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(dict(kind=res["kind"], board=board, cells=res["coverage"]["cells"], predicted=res["coverage"]["predicted"], out=str(out))))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", required=True, choices=sorted(BD.CONFIG))
    ap.add_argument("--profile", required=True, choices=CS.PROFILES)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--calibration-source", default="")
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--plumbing-test", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    try:
        return run(a)
    except Refusal as ex:
        print("REFUSED: %s" % ex, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
