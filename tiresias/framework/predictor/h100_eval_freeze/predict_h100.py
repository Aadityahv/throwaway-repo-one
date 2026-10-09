#!/usr/bin/env python3
"""Frozen H100 predictions: static runtime and energy of every cell of cells_h100.json from an H100 calibration document. No kernel is executed; no measured value is read.

    python predict_h100.py --calibration <H100 calibration document .json> --calibration-source "<job and repeat>" [--profile main|tensor] [--out <name>] [--allow-incomplete]

Profiles (cell_sets.py): `main` = the 72 CUDA-samples cells and the 40 machine-learning-kernel cells, predicted from ONE document into predictions_h100.json; `tensor` = the 16 tensor-core matrix
multiply and fused attention cells into predictions_h100_tensor.json, from the first complete calibration run that contains the calibrator's tensor stage (a document without usable tensor constants
makes every tensor cell "unsupported", counted as a failure, never priced at another class's rate).

Refuses (exit 2, nothing written) when: no calibration document is given; the document does not exist; the device in it is not an H100 whose SM count, L2 size and compute capability
equal the H100 rows of HARDWARE_GROUND_TRUTH.md; the document is incomplete or has an unidentified energy rate (unless --allow-incomplete, which marks the output); the output exists.
There is NO fallback to Blackwell constants, to a default document or to any other board's numbers.

Runtime: the CURRENT model of CURRENT_MODELS.md, the shared-traffic model (predict_runtime_v3j.py on top of the bank-conflict shared-memory cost and partial phase overlap, run through
calibrate/cal/portable_predict.py): DRAM reads are the grid-wide unique footprint of every kernel (footprints_h100.json, derived through the sm_90 port by footprints_h100.py), and a kernel whose
per-wave inter-block reuse working set exceeds the L2 capacity of THIS board (the H100 row of HARDWARE_GROUND_TRUTH.md, checked against the document) is refused; a DRAM-tier kernel
without a footprint is refused. The refused cells stay in the list and count as failures. Constants come from the document (exported with calibrate/cal/document.export_legacy) and the static
tables of this directory.
Energy: the current component energy model with traffic-based byte columns (calibrate/cal/traffic.py): min(cap x t, base x t + sum(rate x column)), the document's own calibrator-only profile
rates, at the predicted runtime; B_dr and B_l2 are the bytes the runtime model itself serves. Roofline reference: the document's measured empty-launch time plus logical bytes over the document's
measured peak read bandwidth of the cell's tier. A cell the static side could not handle is listed with
its reason and counts as a failure in the scorer; it is never dropped.

Plumbing test only: `--plumbing-test --only <substring>` accepts a calibration document of ANOTHER board (for example a Blackwell document in calibrate/runs/), for at most four cells, and
marks the output `kind: plumbing_test_not_h100`. Such an output must never be committed as predictions_h100.json (the name is refused in this mode).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = HERE.parent
CAL = SR / "calibrate"
sys.path.insert(0, str(CAL))
sys.path.insert(0, str(SR))
sys.path.insert(0, str(HERE))

import cell_sets as CS  # noqa: E402

FROZEN_NAMES = tuple(CS.PROFILES[p]["predictions"] for p in CS.PROFILES)
FROZEN_NAME = CS.PROFILES[CS.DEFAULT_PROFILE]["predictions"]
SCHEMA = "h100_predictions/2"
CODE_FILES = [HERE / "predict_h100.py", HERE / "cell_sets.py", HERE / "make_cells_h100.py", HERE / "build_static_h100.py", HERE / "ml_sets" / "make_cells_ml.py", HERE / "ml_sets" / "build_static_ml.py",
              SR / "port_common" / "port_ext.py", SR / "port_common" / "port_ml_ext.py", CAL / "predict.py", CAL / "cal" / "portable_predict.py", CAL / "cal" / "document.py",
              CAL / "cal" / "energy.py", SR / "predict_runtime_v2.py", SR / "predict_runtime_v3.py", SR / "predict_runtime_v3f.py", SR / "predict_runtime_v3h.py", SR / "predict_runtime_v3i.py",
              SR / "predict_runtime_v3j.py", SR / "shared_traffic" / "footprint.py", CAL / "cal" / "traffic.py", HERE / "footprints_h100.py"]


class Refusal(RuntimeError):
    pass


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_h100_document(doc, hw):
    d = doc.get("device") or {}
    problems = []
    if "H100" not in str(d.get("name", "")):
        problems.append("device name %r is not an H100" % d.get("name"))
    if d.get("sm_count") != hw["sm_count"]:
        problems.append("SM count %r differs from HARDWARE_GROUND_TRUTH.md (%d)" % (d.get("sm_count"), hw["sm_count"]))
    if d.get("l2_bytes") != hw["l2_bytes"]:
        problems.append("L2 size %r differs from HARDWARE_GROUND_TRUTH.md (%d)" % (d.get("l2_bytes"), hw["l2_bytes"]))
    if str(d.get("compute_capability")) != "9.0":
        problems.append("compute capability %r is not 9.0" % d.get("compute_capability"))
    return problems


def run(a):
    import make_cells_h100 as MC
    import predict as P          # calibrate/predict.py (reused unchanged)
    prof = CS.profile(a.profile)
    from cal import document as D, portable_predict as PP, traffic as T
    import predict_runtime_v3j as J
    if a.calibration is None:
        raise Refusal("no calibration document given: the H100 predictions need the H100 calibration (calibrate/run_calibration.py on an H100). There is no default and no fallback to Blackwell constants.")
    cal_path = Path(a.calibration)
    if not cal_path.is_file():
        raise Refusal("calibration document %s does not exist" % cal_path)
    out = Path(a.out) if a.out else HERE / prof["predictions"]
    if not a.plumbing_test and not (a.calibration_source or "").strip():
        raise Refusal("--calibration-source is required: say which run this is (cluster job id and repeat). The frozen document is the FIRST complete calibration run by job order and timestamp, "
                      "never the one that looks best (FREEZE.md section 9)")
    if out.exists():
        raise Refusal("%s exists; never overwritten" % out)
    if a.plumbing_test:
        if out.name in FROZEN_NAMES:
            raise Refusal("--plumbing-test output must not be called %s" % out.name)
        if not a.only:
            raise Refusal("--plumbing-test needs --only <substring> (a couple of cells)")
    elif a.only:
        raise Refusal("--only is for --plumbing-test; the frozen predictions cover every cell")
    hw, _ = MC.read_h100_hardware()
    doc = json.loads(cal_path.read_text(encoding="utf-8"))
    problems = check_h100_document(doc, hw)
    if problems and not a.plumbing_test:
        raise Refusal("the calibration document is not an H100 one: " + "; ".join(problems))
    try:
        P.load_calibration(cal_path, allow_incomplete=a.allow_incomplete)
    except P.PredictRefusal as ex:
        raise Refusal(str(ex))
    for name in CS.static_files(a.profile):
        if not (HERE / name).is_file():
            raise Refusal("%s is missing: run build_static_h100.py / ml_sets/build_static_ml.py / footprints_h100.py (and the cell generators) first" % name)
    all_cells, _ = CS.load_cells(a.profile)
    cells_doc = dict(cells=all_cells)
    cell_ids = [c["cell_id"] for c in all_cells]
    if a.only:
        cell_ids = [c for c in cell_ids if a.only in c]
        if not 0 < len(cell_ids) <= 4:
            raise Refusal("--plumbing-test --only %r selects %d cells; it must select 1 to 4" % (a.only, len(cell_ids)))
    features, phases, unique, bank, support = CS.load_static(a.profile)
    keep = set(cell_ids)
    features = dict(features, rows=[r for r in features["rows"] if r["cell_id"] in keep])
    phases, unique, bank = ({k: v for k, v in t.items() if k in keep} for t in (phases, unique, bank))

    # runtime: only the cells the static side supports (an unsupported cell has no tables), from the document's own constants
    fp_doc = json.loads((HERE / prof["footprints"][0]).read_text(encoding="utf-8"))
    if fp_doc.get("schema") != "h100_footprints/1" or fp_doc.get("arch") != "sm_90":
        raise Refusal("%s is not an sm_90 h100_footprints/1 file" % prof["footprints"][0])
    l2_capacity = int(hw["l2_bytes"])
    if (not a.plumbing_test and int(doc["device"]["l2_bytes"]) != l2_capacity) or l2_capacity == 134217728:
        raise Refusal("the L2 capacity of the capacity rule must be this board's own (%d bytes, HARDWARE_GROUND_TRUTH.md); the Blackwell L2 size must never reach an H100 prediction" % l2_capacity)
    missing_fp = [c for c in cell_ids if c not in fp_doc["rows"]]
    if missing_fp:
        raise Refusal("%s lacks %d cells (first: %s); run footprints_h100.py" % (prof["footprints"][0], len(missing_fp), missing_fp[0]))
    ok_ids = {c for c in cell_ids if support[c]["supported"]}
    f_ok = dict(features, rows=[r for r in features["rows"] if r["cell_id"] in ok_ids])
    with tempfile.TemporaryDirectory(prefix="h100_legacy_") as tmp:
        D.export_legacy(doc, tmp)
        consts = PP.load_constants(tmp)
        legacy_sha = {n: sha(Path(tmp) / n) for n in sorted(p.name for p in Path(tmp).iterdir())}
        unique_fp = J.attach_footprints({k: unique[k] for k in ok_ids}, fp_doc["rows"], l2_capacity)
        rt_raw = T.predict_candidate(f_ok, {k: phases[k] for k in ok_ids}, unique_fp, {k: bank[k] for k in ok_ids}, consts, doc["device"]["sm_count"], read_footprint=True, l2_rule="wave")
    runtime = {cid: (v.get("primary_s") if isinstance(v, dict) else v) for cid, v in rt_raw.items()}
    traffic = {cid: v["traffic"] for cid, v in rt_raw.items() if isinstance(v, dict) and v.get("primary_s") and v.get("traffic")}
    en = T.energy_rows_traffic(doc, f_ok, runtime, traffic)

    energy_profile = doc["constants"]["energy"]["profile"]
    energy_profile_base = energy_profile["base_power_w"]
    # plain roofline runtime (t0 + logical bytes / measured peak tier bandwidth), from the same document: the runtime reference the criteria compare with
    sl = doc["constants"].get("store_legacy") or {}
    need = ("t0_us", "L2_read_sector_TBps", "DRAM_read_TBps")
    if any(k not in sl for k in need):
        raise Refusal("the calibration document lacks the store constants the roofline reference needs: %s" % ", ".join(k for k in need if k not in sl))
    cell_by_id = {c["cell_id"]: c for c in cells_doc["cells"]}
    roofline = {cid: sl["t0_us"] * 1e-6 + cell_by_id[cid]["logical_bytes_per_launch"] / ((sl["L2_read_sector_TBps"] if cell_by_id[cid]["tier"] == "L2" else sl["DRAM_read_TBps"]) * 1e12)
                for cid in cell_ids}
    tensor_record = None
    tdoc = doc["constants"].get("tensor") or None
    if prof["needs_tensor"] or tdoc:
        te = (tdoc or {}).get("energy") or {}
        ti = (tdoc or {}).get("issue") or {}
        tensor_record = dict(present=bool(tdoc), status=(tdoc or {}).get("status"), energy_status=te.get("status"), energy_rate_pJ_per_lane_instruction=te.get("rate_pJ_per_lane_instruction"),
                             energy_windows_admitted=te.get("admitted"), energy_windows_not_admitted=te.get("not_admitted"),
                             issue_cycles_per_warp_instruction_per_sm=ti.get("issue_cycles_per_warp_instruction_per_sm"), dependent_latency_cycles=ti.get("dependent_latency_cycles"),
                             attachment=doc.get("tensor_attachment"),
                             rule="every tensor constant comes from the calibration document's tensor stage (and its energy stage) only; a cell with tensor instructions is unsupported without them")
        if prof["needs_tensor"] and not (tdoc and te.get("status") == "ok" and ti.get("issue_cycles_per_warp_instruction_per_sm")):
            print("WARNING: the calibration document has no usable tensor constants: every cell with tensor instructions will be unsupported (a failure in the scores)", file=sys.stderr)
    rows = {}
    for cid in cell_ids:
        if not support[cid]["supported"]:
            rows[cid] = dict(status="unsupported", reason=support[cid]["reason"], counted_as_failure=True)
        elif en.get(cid, {}).get("status") != "ok":
            why = rt_raw[cid].get("unsupported_reason") if isinstance(rt_raw.get(cid), dict) else None
            rows[cid] = dict(status="not_predicted", reason=why or en.get(cid, {}).get("reason", "no runtime prediction"), runtime_s=runtime.get(cid), counted_as_failure=True)
        else:
            e = en[cid]
            rows[cid] = dict(status="predicted", runtime_s=e["runtime_s"], base_power_w=energy_profile_base, energy_j=e["energy_j"], mean_power_w=e["mean_power_w"], capped=bool(e["capped"]),
                             base_term_j=e["base_term_j"], term_j=e["term_j"],
                             traffic_bytes=dict(l2_served=traffic[cid]["l2_read_bytes"] + traffic[cid]["l2_write_bytes"], dram=traffic[cid]["dram_read_bytes"] + traffic[cid]["dram_write_bytes"]))
    res = dict(
        schema=SCHEMA, kind="plumbing_test_not_h100" if a.plumbing_test else "h100_frozen_predictions",
        warning=("PLUMBING TEST ONLY: made with the document of %r on %d cells; not H100 predictions and never to be committed as %s" % (doc["device"].get("name"), len(cell_ids), FROZEN_NAME)) if a.plumbing_test else None,
        calibration=dict(source=(a.calibration_source or "").strip() or None, file=cal_path.name, sha256=sha(cal_path), device=doc["device"].get("name"), uuid=doc["device"].get("uuid"), sm_count=doc["device"].get("sm_count"),
                         complete=bool(doc.get("complete")), allow_incomplete_used=bool(a.allow_incomplete), created_utc=doc.get("created_utc"), warnings=doc.get("warnings"),
                         energy_profile_status=energy_profile.get("status"), energy_profile_base_power_w=energy_profile.get("base_power_w"), energy_cap_w=doc["constants"]["energy"].get("cap_w"),
                         energy_windows_not_admitted=next((st.get("not_admitted") for st in doc.get("stages", []) if st.get("program") == "micro_energy" or st.get("name") == "energy"), None),
                         legacy_constants_sha256=legacy_sha, board_seconds_recorded=round(sum(float(st.get("seconds") or 0) for st in doc.get("stages", [])), 2),
                         stages_without_recorded_seconds=[st.get("name") or st.get("program") for st in doc.get("stages", []) if st.get("seconds") is None]),
        runtime_model="current runtime model (CURRENT_MODELS.md): shared-traffic model, grid-wide DRAM read footprint and per-wave L2-capacity refusal (predict_runtime_v3j.py via calibrate/cal/traffic.py and portable_predict.py)",
        energy_model="current component energy model with traffic-based byte columns: E = min(cap x t, base x t + sum(rate x column)), calibrator-only profile rates of the calibration document's energy stage, at the predicted runtime (calibrate/cal/traffic.py)",
        l2_capacity_bytes_used_by_the_capacity_rule=l2_capacity, footprints_sha256=sha(HERE / prof["footprints"][0]),
        profile=a.profile, tensor_stage=tensor_record,
        inputs_sha256={n: sha(HERE / n) for n in CS.static_files(a.profile)},
        code_sha256={str(p.relative_to(SR)).replace("\\", "/"): sha(p) for p in CODE_FILES},
        coverage=dict(cells=len(cell_ids), predicted=sum(1 for v in rows.values() if v["status"] == "predicted"),
                      unsupported=sorted(c for c, v in rows.items() if v["status"] != "predicted")),
        note="No kernel was executed and no runtime, energy or power value of any H100 cell was read. Cells the static side could not handle stay in the list and count as failures.",
        roofline_runtime_s=roofline, roofline_constants={k: sl[k] for k in need},
        cells=rows)
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(dict(kind=res["kind"], cells=res["coverage"]["cells"], predicted=res["coverage"]["predicted"], out=str(out))))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", type=Path, default=None, help="H100 calibration document (required; no default)")
    ap.add_argument("--profile", choices=sorted(CS.PROFILES), default=CS.DEFAULT_PROFILE)
    ap.add_argument("--out", type=Path, default=None, help="default: the frozen name of the profile (predictions_h100.json, predictions_h100_tensor.json)")
    ap.add_argument("--calibration-source", default="", help="which calibration run this document is, e.g. 'cluster job j13, repeat 1' (required unless --plumbing-test)")
    ap.add_argument("--allow-incomplete", action="store_true", help="accept an incomplete calibration; the output is marked and the user must decide whether it may be frozen")
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
