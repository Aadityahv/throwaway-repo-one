#!/usr/bin/env python3
"""Frozen Ada predictions: static runtime and energy of every cell of a profile from an Ada calibration document. No kernel is executed; no measured value of an Ada cell is read.

    python predict_ada.py --calibration <Ada calibration document .json> --calibration-source "<booking and run>" [--profile main|tensor] [--out <name>] [--allow-incomplete]
    python predict_ada.py --list-unmeasured                       # cells whose kernels contain an opcode of unmeasured throughput (needs only the static tables)
    python predict_ada.py --refusal-preview --calibration <any document>   # which cells the model's refusal rules (per-wave L2 capacity, DRAM tier without footprint) refuse on Ada; writes nothing

Runtime: the CURRENT runtime model of CURRENT_MODELS.md, the stage-rule model (predict_runtime_v3k.py: shared-traffic model, grid-wide DRAM footprint, per-wave L2-capacity refusal, pair overlap, stage
serialisation, busiest-SM rule), run through calibrate/cal/portable_predict.py with the constants of the Ada calibration document. Pair-overlap constants: EXACTLY constants/pair_overlap_constants_ada.json
(Ada's own dependent fragment-load microbenchmark; path and sha256 recorded in the output). They are passed to the model in memory for the duration of the call; the Blackwell file is never read for
an Ada prediction, and the predictor refuses if the Ada file is missing, is not a pair_overlap_constants/1 document, or equals the Blackwell file. Nothing in predict_runtime_v3k.py is edited.
The per-wave L2 refusal uses Ada's own L2 capacity (HARDWARE_GROUND_TRUTH.md Ada section = the calibration document's device); refused cells are failures.
Energy: the current component energy model with traffic-based byte columns (calibrate/cal/traffic.py), rates from the Ada calibration document's energy stage; a cell whose energy needs an
unidentified rate is "not_predicted" (a failure), never priced at another rate.
Class-mapped opcodes (NOT refused): F2FP and FMNMX throughput was never measured per opcode on Ada or on Blackwell. Ada prices them exactly as the
Blackwell model does, through the shared instruction-class mapping with Ada's own calibrated class costs (F2FP: catch-all issue bucket, integer energy family;
FMNMX: floating-point-other issue bucket and energy family). This keeps Ada no stricter than Blackwell (refusing would drop the attention family and fail the
five-sixths rule); it is listed as a known weakness in FREEZE.md and stays reversible until the freeze. Cells whose kernels execute either are predicted, never
given a guessed per-opcode cost.

Refuses (exit 2, nothing written) when: no calibration document is given or it does not exist; its device is not the Ada board of HARDWARE_GROUND_TRUTH.md (name, SM count, L2, compute capability, approved
UUID); the document is incomplete (unless --allow-incomplete, marked); any cell of the profile is pending build (SASS missing) ; the Ada pair-overlap file is missing or invalid; the output exists.
There is NO fallback to Blackwell or H100 constants and no default document.

Plumbing test only: `--plumbing-test --only <substring>` accepts a calibration document of ANOTHER board for at most four cells and marks the output `kind: plumbing_test_not_ada`; the frozen names are refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import board_ada as B
import cell_sets_ada as CS
import make_cells_ada as MA

HERE = B.HERE
SR = B.SR
CAL = SR / "calibrate"
for p in (CAL, SR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SCHEMA = "ada_predictions/1"
CODE_FILES = [HERE / f for f in ("predict_ada.py", "cell_sets_ada.py", "board_ada.py", "make_cells_ada.py", "static_ada.py")] + [
    SR / "port_common" / "port_ext.py", SR / "port_common" / "port_ml_ext.py", SR / "port_common" / "set_d_port.py", CAL / "predict.py", CAL / "cal" / "portable_predict.py", CAL / "cal" / "document.py",
    CAL / "cal" / "energy.py", CAL / "cal" / "traffic.py", SR / "predict_runtime_v2.py", SR / "predict_runtime_v3.py", SR / "predict_runtime_v3f.py", SR / "predict_runtime_v3h.py",
    SR / "predict_runtime_v3i.py", SR / "predict_runtime_v3j.py", SR / "predict_runtime_v3k.py", SR / "shared_traffic" / "footprint.py"]


class Refusal(RuntimeError):
    pass


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pair_constants():
    path = B.PAIR_CONSTANTS
    if not path.is_file():
        raise Refusal("%s is missing: the Ada pair-overlap constants come from Ada's own dependent fragment-load microbenchmark and there is no fallback to the Blackwell file" % path)
    d = json.loads(path.read_text(encoding="utf-8"))
    bw = SR / "constants" / "pair_overlap_constants.json"
    if d.get("schema") != "pair_overlap_constants/1" or not (len(d.get("resident_warps_per_sm", [])) == len(d.get("beta_hmma", [])) > 1):
        raise Refusal("%s is not a pair_overlap_constants/1 document" % path.name)
    if bw.is_file() and sha(path) == sha(bw):
        raise Refusal("%s is byte-identical to the Blackwell pair-overlap file; Ada needs its own measurement" % path.name)
    if "ada" not in str(d.get("source", "")).lower():
        raise Refusal("the source of %s (%r) does not name an Ada microbenchmark" % (path.name, d.get("source")))
    return d


def check_ada_document(doc, hw):
    d = doc.get("device") or {}
    problems = []
    if "5000 Ada" not in str(d.get("name", "")):
        problems.append("device name %r is not the RTX 5000 Ada" % d.get("name"))
    if d.get("sm_count") != hw["sm_count"]:
        problems.append("SM count %r differs from HARDWARE_GROUND_TRUTH.md (%d)" % (d.get("sm_count"), hw["sm_count"]))
    if d.get("l2_bytes") != hw["l2_bytes"]:
        problems.append("L2 size %r differs from HARDWARE_GROUND_TRUTH.md (%d)" % (d.get("l2_bytes"), hw["l2_bytes"]))
    if str(d.get("compute_capability")) != "8.9":
        problems.append("compute capability %r is not 8.9" % d.get("compute_capability"))
    approved = json.loads((CAL / "approved_devices.json").read_text(encoding="utf-8"))["devices"]
    ada_uuids = {x["uuid"] for x in approved if x.get("ground_truth_section") == "Ada"}
    if d.get("uuid") not in ada_uuids:
        problems.append("device UUID %r is not the approved Ada GPU in calibrate/approved_devices.json" % d.get("uuid"))
    return problems


def list_unmeasured(cells, phases):
    return {c["cell_id"]: CS.unmeasured_opcodes(phases[c["cell_id"]]) for c in cells if c["cell_id"] in phases and CS.unmeasured_opcodes(phases[c["cell_id"]])}


def refusal_preview(a, all_cells, feats_all, phases_all, unique_all, bank_all, support_all, fp_all):
    """Which cells the runtime model's refusal rules refuse on Ada's 64 MiB L2. The refusal rules depend on footprints, the L2 capacity and occupancy, not on the rates of the document used here; every runtime in
    the model's result is discarded. Another board's document may be given; nothing is written and nothing here is a prediction."""
    from cal import document as D, portable_predict as PP
    import predict_runtime_v3j as J
    import predict_runtime_v3k as K3
    if a.calibration is None or not Path(a.calibration).is_file():
        raise Refusal("--refusal-preview needs --calibration <any calibration document> (its rates are not used for the refusal rules)")
    hw, _ = B.read_ada_hardware()
    doc = json.loads(Path(a.calibration).read_text(encoding="utf-8"))
    pair = load_pair_constants()
    ok = {c["cell_id"] for c in all_cells if support_all[c["cell_id"]]["supported"] and not CS.unmeasured_opcodes(phases_all[c["cell_id"]])}
    f_ok = dict(feats_all, rows=[r for r in feats_all["rows"] if r["cell_id"] in ok])
    with tempfile.TemporaryDirectory(prefix="ada_legacy_") as tmp:
        D.export_legacy(doc, tmp)
        consts = PP.load_constants(tmp)
        unique_fp = J.attach_footprints({k: unique_all[k] for k in ok}, {k: fp_all[k] for k in ok}, int(hw["l2_bytes"]))
        old = K3._PAIR
        K3._PAIR = pair
        try:
            rt = K3.predict_portable(f_ok, {k: phases_all[k] for k in ok}, unique_fp, {k: bank_all[k] for k in ok}, consts, hw["sm_count"])
        finally:
            K3._PAIR = old
    refused = {c: v.get("unsupported_reason") for c, v in rt.items() if isinstance(v, dict) and not v.get("primary_s")}
    pend = [c["cell_id"] for c in all_cells if support_all[c["cell_id"]].get("pending_build")]
    print(json.dumps(dict(profile=a.profile, preview="board-independent refusal rules only; no runtime here is a prediction", l2_capacity_bytes=hw["l2_bytes"], evaluated=len(ok), pending_build=len(pend),
                          refused_by_model=refused, refused_for_unmeasured_opcodes=list_unmeasured(all_cells, phases_all)), indent=1, sort_keys=True))
    return 0


def run(a):
    prof = CS.profile(a.profile)
    all_cells = CS.load_cells(a.profile)
    feats_all, phases_all, unique_all, bank_all, support_all, fp_all = CS.load_static(a.profile)
    if a.list_unmeasured:
        found = list_unmeasured(all_cells, phases_all)
        pend = [c["cell_id"] for c in all_cells if support_all[c["cell_id"]].get("pending_build")]
        print(json.dumps(dict(profile=a.profile, cells=len(all_cells), pending_build=len(pend), refused_for_unmeasured_opcodes=found), indent=1, sort_keys=True))
        return 0
    import predict as P
    from cal import document as D, portable_predict as PP, traffic as T
    import predict_runtime_v3j as J
    import predict_runtime_v3k as K3
    if a.refusal_preview:
        return refusal_preview(a, all_cells, feats_all, phases_all, unique_all, bank_all, support_all, fp_all)
    if a.calibration is None:
        raise Refusal("no calibration document given: the Ada predictions need the Ada calibration (calibrate/run_calibration.py on Ada). There is no default and no fallback to Blackwell constants.")
    cal_path = Path(a.calibration)
    if not cal_path.is_file():
        raise Refusal("calibration document %s does not exist" % cal_path)
    out = Path(a.out) if a.out else HERE / prof["predictions"]
    if not a.plumbing_test and not (a.calibration_source or "").strip():
        raise Refusal("--calibration-source is required: say which Ada calibration run this is (booking and repeat)")
    if out.exists():
        raise Refusal("%s exists; never overwritten" % out)
    if a.plumbing_test:
        if out.name in CS.FROZEN_NAMES:
            raise Refusal("--plumbing-test output must not be called %s" % out.name)
        if not a.only:
            raise Refusal("--plumbing-test needs --only <substring> (a couple of cells)")
    elif a.only:
        raise Refusal("--only is for --plumbing-test; the frozen predictions cover every cell of the profile")
    pair = load_pair_constants()
    hw, _ = B.read_ada_hardware()
    doc = json.loads(cal_path.read_text(encoding="utf-8"))
    problems = check_ada_document(doc, hw)
    if problems and not a.plumbing_test:
        raise Refusal("the calibration document is not an Ada one: " + "; ".join(problems))
    try:
        P.load_calibration(cal_path, allow_incomplete=a.allow_incomplete)
    except P.PredictRefusal as ex:
        raise Refusal(str(ex))
    cell_ids = [c["cell_id"] for c in all_cells]
    if a.only:
        cell_ids = [c for c in cell_ids if a.only in c]
        if not 0 < len(cell_ids) <= 4:
            raise Refusal("--plumbing-test --only %r selects %d cells; it must select 1 to 4" % (a.only, len(cell_ids)))
    pending = [c for c in cell_ids if support_all[c].get("pending_build")]
    if pending:
        raise Refusal("%d cells are pending build (SASS missing; BUILD_MANIFEST.md), first: %s" % (len(pending), pending[0]))
    keep = set(cell_ids)
    features = dict(feats_all, rows=[r for r in feats_all["rows"] if r["cell_id"] in keep])
    phases, unique, bank, support, fps = ({k: v for k, v in t.items() if k in keep} for t in (phases_all, unique_all, bank_all, support_all, fp_all))
    l2_capacity = int(hw["l2_bytes"])
    bw_l2, bw_sm = B.forbidden_blackwell_values()
    if l2_capacity == bw_l2 or hw["sm_count"] == bw_sm:
        raise Refusal("the Ada hardware rows equal Blackwell's; refusing")
    if not a.plumbing_test and int(doc["device"]["l2_bytes"]) != l2_capacity:
        raise Refusal("the document's L2 differs from the Ada row")
    unmeasured = list_unmeasured([c for c in all_cells if c["cell_id"] in keep], phases)
    ok_ids = {c for c in cell_ids if support[c]["supported"] and c not in unmeasured}
    f_ok = dict(features, rows=[r for r in features["rows"] if r["cell_id"] in ok_ids])
    consts_sha = {}
    with tempfile.TemporaryDirectory(prefix="ada_legacy_") as tmp:
        D.export_legacy(doc, tmp)
        consts = PP.load_constants(tmp)
        consts_sha = {n: sha(Path(tmp) / n) for n in sorted(p.name for p in Path(tmp).iterdir())}
        unique_fp = J.attach_footprints({k: unique[k] for k in ok_ids}, {k: fps[k] for k in ok_ids}, l2_capacity)
        old_pair = K3._PAIR
        K3._PAIR = pair                      # Ada's own constants for the duration of the call; restored below
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
        raise Refusal("the calibration document lacks the store constants the roofline reference needs: %s" % ", ".join(k for k in need if k not in sl))
    cell_by_id = {c["cell_id"]: c for c in all_cells}
    roofline = {cid: sl["t0_us"] * 1e-6 + cell_by_id[cid]["logical_bytes_per_launch"] / ((sl["L2_read_sector_TBps"] if cell_by_id[cid]["tier"] == "L2" else sl["DRAM_read_TBps"]) * 1e12) for cid in cell_ids}
    tdoc = doc["constants"].get("tensor") or None
    tensor_record = None
    if prof["needs_tensor"] or tdoc:
        te = (tdoc or {}).get("energy") or {}
        ti = (tdoc or {}).get("issue") or {}
        tensor_record = dict(present=bool(tdoc), status=(tdoc or {}).get("status"), energy_status=te.get("status"), energy_rate_pJ_per_lane_instruction=te.get("rate_pJ_per_lane_instruction"),
                             issue_cycles_per_warp_instruction_per_sm=ti.get("issue_cycles_per_warp_instruction_per_sm"), dependent_latency_cycles=ti.get("dependent_latency_cycles"),
                             rule="every tensor constant comes from the calibration document's tensor stage only; a cell with tensor instructions is unsupported without them")
        if prof["needs_tensor"] and not (tdoc and te.get("status") == "ok" and ti.get("issue_cycles_per_warp_instruction_per_sm")):
            print("WARNING: the calibration document has no usable tensor constants: tensor cells will be unsupported (failures)", file=sys.stderr)
    rows = {}
    for cid in cell_ids:
        if cid in unmeasured:
            rows[cid] = dict(status="not_predicted", reason="refused: executes opcode(s) of unmeasured throughput on Ada (also unmeasured on Blackwell): %s; no cost is guessed" % ", ".join("%s x%d" % kv for kv in sorted(unmeasured[cid].items())),
                             unmeasured_opcodes=unmeasured[cid], counted_as_failure=True)
        elif not support[cid]["supported"]:
            rows[cid] = dict(status="unsupported", reason=support[cid]["reason"], counted_as_failure=True)
        elif en.get(cid, {}).get("status") != "ok":
            why = rt_raw[cid].get("unsupported_reason") if isinstance(rt_raw.get(cid), dict) else None
            rows[cid] = dict(status="not_predicted", reason=why or en.get(cid, {}).get("reason", "no runtime prediction"), runtime_s=runtime.get(cid), counted_as_failure=True)
        else:
            e = en[cid]
            rows[cid] = dict(status="predicted", runtime_s=e["runtime_s"], base_power_w=energy_profile["base_power_w"], energy_j=e["energy_j"], mean_power_w=e["mean_power_w"], capped=bool(e["capped"]),
                             base_term_j=e["base_term_j"], term_j=e["term_j"],
                             traffic_bytes=dict(l2_served=traffic[cid]["l2_read_bytes"] + traffic[cid]["l2_write_bytes"], dram=traffic[cid]["dram_read_bytes"] + traffic[cid]["dram_write_bytes"]))
    res = dict(
        schema=SCHEMA, kind="plumbing_test_not_ada" if a.plumbing_test else "ada_frozen_predictions",
        warning=("PLUMBING TEST ONLY: made with the document of %r on %d cells; not Ada predictions and never to be committed as %s" % (doc["device"].get("name"), len(cell_ids), prof["predictions"])) if a.plumbing_test else None,
        calibration=dict(source=(a.calibration_source or "").strip() or None, file=cal_path.name, sha256=sha(cal_path), device=doc["device"].get("name"), uuid=doc["device"].get("uuid"),
                         sm_count=doc["device"].get("sm_count"), complete=bool(doc.get("complete")), allow_incomplete_used=bool(a.allow_incomplete), created_utc=doc.get("created_utc"), warnings=doc.get("warnings"),
                         energy_profile_status=energy_profile.get("status"), energy_profile_base_power_w=energy_profile.get("base_power_w"), energy_cap_w=doc["constants"]["energy"].get("cap_w"),
                         legacy_constants_sha256=consts_sha),
        pair_overlap_constants=dict(file=str(B.PAIR_CONSTANTS.relative_to(SR)).replace("\\", "/"), sha256=sha(B.PAIR_CONSTANTS), source=pair.get("source"), source_sha256=pair.get("source_sha256"),
                                    resident_warps_per_sm=pair["resident_warps_per_sm"], beta_hmma=pair["beta_hmma"]),
        runtime_model="current runtime model (CURRENT_MODELS.md): stage-rule model (predict_runtime_v3k.py) on the shared-traffic model with per-wave L2 refusal, Ada's own pair-overlap constants",
        energy_model="current component energy model with traffic-based byte columns: E = min(cap x t, base x t + sum(rate x column)), calibrator-only profile rates of the Ada calibration document (calibrate/cal/traffic.py)",
        l2_capacity_bytes_used_by_the_capacity_rule=l2_capacity, sm_count_used=hw["sm_count"], profile=a.profile, tensor_stage=tensor_record,
        inputs_sha256={n: sha(HERE / n) for n in CS.static_files()},
        code_sha256={str(p.relative_to(SR)).replace("\\", "/"): sha(p) for p in CODE_FILES},
        coverage=dict(cells=len(cell_ids), predicted=sum(1 for v in rows.values() if v["status"] == "predicted"), refused_unmeasured_opcodes=sorted(unmeasured),
                      not_predicted=sorted(c for c, v in rows.items() if v["status"] != "predicted")),
        note="No kernel was executed and no runtime, energy or power value of any Ada cell was read. Cells the static side or the model refuses stay in the list and count as failures.",
        roofline_runtime_s=roofline, roofline_constants={k: sl[k] for k in need}, cells=rows)
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(dict(kind=res["kind"], cells=res["coverage"]["cells"], predicted=res["coverage"]["predicted"], out=str(out))))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--profile", choices=sorted(CS.PROFILES), default=CS.DEFAULT_PROFILE)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--calibration-source", default="")
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--plumbing-test", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--list-unmeasured", action="store_true")
    ap.add_argument("--refusal-preview", action="store_true")
    ap.add_argument("--committed-ada-calibration", action="store_true", help="use board_ada.ADA_CALIBRATION (the committed first complete Ada calibration) as --calibration; --calibration-source is still required for a frozen run")
    a = ap.parse_args(argv)
    if a.committed_ada_calibration:
        if a.calibration is not None:
            print("REFUSED: give either --calibration or --committed-ada-calibration", file=sys.stderr)
            return 2
        a.calibration = B.ADA_CALIBRATION
    try:
        return run(a)
    except Refusal as ex:
        print("REFUSED: %s" % ex, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
