#!/usr/bin/env python3
"""In-job freeze for a board whose queue is waited for once (A100): derive everything the predictions need from the allocation's OWN calibration, pair-overlap and power-sensor outputs, make the predictions for
both profiles and write their freeze records, all BEFORE any evaluation cell is measured (CPU only; no kernel is run here).

    python injob_freeze.py --board a100 --calibration-run <calibration output dir> --pair-run <pair-overlap output dir> --sensor-run <power-sensor output dir> --code-commit <40-hex> --env-out <file>

Runs inside the job's private copy of the source tree (the module's own location decides which tree it writes into). Steps, each of which refuses (exit 2, nothing further written) on failure:
  1. the calibration document of the run (complete, this board's device, one document);
  2. the traffic rates re-derived with the stream-rate method and the directly measured launch floor, in the two steps used for every other board (the tool's own checks of the raw stage hashes), the DRAM
     peak for its plausibility check read from HARDWARE_GROUND_TRUTH.md, never typed;
  3. the board's pair-overlap table from the dependent fragment-load run (the median rule of constants/make_pair_overlap_constants.py), written under the tagged name of the board;
  4. the energy-window decision from the power-sensor result (at most 3 percent tolerance, padding 0, device of this board): WINDOW_S = decision.padding_0s.min_load_s; no window means no energy run (exit 7);
  5. predict_board.py for both profiles from the re-derived document;
  6. freeze_board.write_record_injob for both profiles.
The environment file it writes (WINDOW_VALIDATION, WINDOW_S, PADDING_S, CALIBRATION_UUID) is sourced by the job before the measurement stages. A summary JSON next to it lists every hash and the coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import board as BD
import cell_sets_board as CS
import freeze_board as FZ

HERE = BD.HERE
SR = BD.SR
REPO = BD.REPO
GPU_NAME = {"h100": "H100", "a100": "A100"}


class Refusal(SystemExit):
    def __init__(self, m):
        super().__init__("REFUSED: " + m)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(cmd, what, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        raise Refusal("%s failed (rc=%d): %s" % (what, r.returncode, (r.stderr or r.stdout)[-600:]))
    return r.stdout


def dram_peak_tbps(board):
    """The verified theoretical peak DRAM bandwidth of the board, read from HARDWARE_GROUND_TRUTH.md (its section for the board)."""
    text = (REPO / "HARDWARE_GROUND_TRUTH.md").read_text(encoding="utf-8").splitlines()
    sec = "## " + GPU_NAME[board]
    start = next((i for i, l in enumerate(text) if l.startswith(sec)), None)
    if start is None:
        raise Refusal("HARDWARE_GROUND_TRUTH.md has no section %r" % sec)
    end = next((i for i in range(start + 1, len(text)) if text[i].startswith("## ")), len(text))
    for l in text[start:end]:
        m = re.match(r"\|\s*Theoretical peak DRAM bandwidth \(derived\)\s*\|\s*([0-9.]+)\s*TB/s", l)
        if m:
            return float(m.group(1))
    raise Refusal("no 'Theoretical peak DRAM bandwidth (derived)' row in the %s section" % GPU_NAME[board])


def find_calibration_document(cal_run):
    """The run's own calibration document: a direct child of <cal_run>/run (or run_r1). Never a recursive search: the run directory also holds a private copy of the source tree (src/), which carries
    older calibration documents of other boards (the 6 October 2026 A100 job stopped on exactly this for the power-sensor result)."""
    docs = [p for d in ("run", "run_r1") for p in sorted((Path(cal_run) / d).glob("calibration_sm_*.json")) if "_store_" not in p.name]
    if len(docs) != 1:
        raise Refusal("expected exactly one original calibration document directly under %s/run, found %d: %s" % (cal_run, len(docs), [str(d) for d in docs]))
    return docs[0]


def derive_rates(doc, board):
    peak = dram_peak_tbps(board)
    tool = SR / "calibrate" / "tools" / "rederive_store.py"
    run([sys.executable, str(tool), "--doc", str(doc), "--method", "stream_rates", "--dram-peak-TBps", repr(peak)], "rederive_store (step 1)", cwd=str(SR))
    first = doc.with_name(doc.stem + "_store_stream_rates.json")
    run([sys.executable, str(tool), "--doc", str(first), "--method", "stream_rates", "--dram-peak-TBps", repr(peak)], "rederive_store (step 2)", cwd=str(SR))
    final = first.with_name(first.stem + "_store_stream_rates.json")
    if not final.is_file():
        raise Refusal("the re-derived document %s was not written" % final)
    d = json.loads(final.read_text(encoding="utf-8"))
    if d.get("warnings"):
        raise Refusal("the re-derived document carries warnings: %s" % d["warnings"])
    return final, peak


def derive_pair_table(board, pair_run):
    raw = Path(pair_run) / "dep_frag.jsonl"
    if not raw.is_file():
        raise Refusal("%s is missing" % raw)
    rel_dir = Path("calibrate/runs/cluster_%s_replication_20261004/pair_overlap_injob" % board)
    dest = SR / rel_dir
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw, dest / "dep_frag.jsonl")
    out_name = "pair_overlap_constants_%s_injob_tmp.json" % board
    run([sys.executable, str(SR / "constants" / "make_pair_overlap_constants.py"), "--raw", str(rel_dir / "dep_frag.jsonl"), "--out", out_name], "make_pair_overlap_constants", cwd=str(SR / "constants"))
    target = CS.board_dir(board) / CS.pair_file(board)
    if target.exists():
        raise Refusal("%s exists; never overwritten" % target)
    shutil.move(str(SR / "constants" / out_name), str(target))
    d = json.loads(target.read_text(encoding="utf-8"))
    if len(d["resident_warps_per_sm"]) < 2:
        raise Refusal("the pair-overlap run covers fewer than two resident-warp points")
    return target, d


def window_decision(board, sensor_run):
    # the result is the file directly in the sensor output directory or in its single run directory (<sensor_run>/<arch>_<job>/); never a recursive search: that directory also holds a private copy of the
    # source tree (src/) with the archived H100 result in it, which made the 6 October 2026 A100 job stop here with "found 2"
    cands = sorted(Path(sensor_run).glob("power_sensor_results.json")) + sorted(Path(sensor_run).glob("*/power_sensor_results.json"))
    if len(cands) != 1:
        raise Refusal("expected one power_sensor_results.json directly in %s or one level below it, found %d: %s" % (sensor_run, len(cands), [str(c) for c in cands]))
    d = json.loads(cands[0].read_text(encoding="utf-8"))
    if d.get("schema") != "power_sensor_results_v1":
        raise Refusal("%s has schema %r" % (cands[0], d.get("schema")))
    tol = d.get("tolerance_pct")
    if not isinstance(tol, (int, float)) or not 0 < tol <= 3.0:
        raise Refusal("the sensor test was run with tolerance %r" % tol)
    if not d.get("runs") or any(GPU_NAME[board] not in json.dumps(r.get("device")) for r in d["runs"]):
        raise Refusal("the power-sensor result is not an %s result" % GPU_NAME[board])
    dec = (d.get("decision") or {}).get("padding_0s")
    if not dec or dec.get("min_load_s") is None:
        raise Refusal("the sensor test admits no energy window at padding 0 within %.1f percent: no energy run is possible (exit 7)" % tol)
    return cands[0], float(dec["min_load_s"]), tol


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", required=True, choices=sorted(GPU_NAME))
    ap.add_argument("--calibration-run", required=True, type=Path)
    ap.add_argument("--pair-run", required=True, type=Path)
    ap.add_argument("--sensor-run", required=True, type=Path)
    ap.add_argument("--code-commit", required=True)
    ap.add_argument("--env-out", required=True, type=Path)
    a = ap.parse_args(argv)
    board = a.board
    try:
        doc_path = find_calibration_document(a.calibration_run)
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        if not doc.get("complete"):
            raise Refusal("the calibration document is not complete")
        final_doc, peak = derive_rates(doc_path, board)
        pair_path, pair = derive_pair_table(board, a.pair_run)
        sensor_path, window_s, tol = window_decision(board, a.sensor_run)
        window_copy = CS.board_dir(board) / ("power_sensor_results_%s.json" % CS.tag(board).lstrip("_"))
        shutil.copyfile(sensor_path, window_copy)
        label = "in-job calibration of the measurement allocation (%s), stream-rate re-derived document; made before any evaluation cell was measured" % doc["device"]["uuid"]
        summary = dict(board=board, calibration_document=final_doc.name, calibration_sha256=sha(final_doc), device_uuid=doc["device"]["uuid"], dram_peak_TBps=peak,
                       pair_table=dict(file=pair_path.name, sha256=sha(pair_path), beta_hmma=pair["beta_hmma"], resident_warps_per_sm=pair["resident_warps_per_sm"]),
                       window=dict(file=window_copy.name, sha256=sha(window_copy), window_s=window_s, tolerance_pct=tol, padding_s=0), profiles={})
        for prof in CS.PROFILES:
            run([sys.executable, str(HERE / "predict_board.py"), "--board", board, "--profile", prof, "--calibration", str(final_doc), "--calibration-source", label], "predict_board %s" % prof)
            n = CS.names(board, prof)
            pred = json.loads((CS.board_dir(board) / n["predictions"]).read_text(encoding="utf-8"))
            FZ.write_record_injob(board, prof, a.code_commit)
            summary["profiles"][prof] = dict(predictions=n["predictions"], predictions_sha256=sha(CS.board_dir(board) / n["predictions"]), record=n["record"], cells=pred["coverage"]["cells"],
                                             predicted=pred["coverage"]["predicted"], not_predicted=pred["coverage"]["not_predicted"])
        a.env_out.write_text("WINDOW_VALIDATION=%s\nWINDOW_S=%s\nPADDING_S=0\nCALIBRATION_UUID=%s\n" % (window_copy, repr(window_s), doc["device"]["uuid"]), encoding="utf-8")
        (a.env_out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=1, sort_keys=True))
        return 0
    except Refusal as ex:
        print(ex, file=sys.stderr)
        return 7 if "exit 7" in str(ex) else 2


if __name__ == "__main__":
    sys.exit(main())
