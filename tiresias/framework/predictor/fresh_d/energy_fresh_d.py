#!/usr/bin/env python3
"""Energy of the 27 set-D CUDA-samples cells (copy, transposes, vector add, reductions at new shapes) on Blackwell GPU 1, through the application-energy harness (energy_harness/application_energy_harness.py,
run_binary_energy: cooldown, 90 s precondition, 15 s counted window, NVML trace) -- the same protocol that measured the unseen-kernel and set-E energies. Set D was timed (runtime) earlier and never
had energy measured; this adds the labels. Reuses timing_fresh_d.py unchanged for: build (the runners' own build_binary), the SASS hash gate against the retained kernels, input files and the correctness oracles.

  CUDA_VISIBLE_DEVICES=1 python3 energy_fresh_d.py --dry-run
  CUDA_VISIBLE_DEVICES=1 python3 energy_fresh_d.py --gate-only --source-root <pinned samples>            (build + hash gate; no kernel, no energy window)
  CUDA_VISIBLE_DEVICES=1 python3 energy_fresh_d.py --booking-ref "<booking log booking>" --source-root <pinned samples> --out-dir energy_raw_d

Guards: a booking log booking reference is required; exactly Blackwell GPU 1, idle, before every cell; the hash gate must pass; one attempt per cell (rejections are recorded, never retried); no clocks, persistence
mode or power limit is touched; no profiler. The per-launch runtime of each cell comes from the committed timing (timing_fresh_d_result.json), only to choose the graph batch.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import timing_fresh_d as T  # noqa: E402

GPU_INDEX = 1
PLATFORM = "blackwell"
TARGET_GRAPH_S = 5e-3
MAX_GRAPH_BATCH = 2048
PINNED_REVISION = "5443602d89ed99aede2e4b7bf329daddeadb320e"


def choose_graph_batch(t_call_s):
    if not (t_call_s > 0 and math.isfinite(t_call_s)):
        raise ValueError("non-positive per-call time")
    return int(min(MAX_GRAPH_BATCH, max(1, math.ceil(TARGET_GRAPH_S / t_call_s))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--booking-ref", default=""); ap.add_argument("--source-root", default=T.DEFAULT_SOURCE_ROOT); ap.add_argument("--workdir", default="")
    ap.add_argument("--out-dir", default=str(HERE / "energy_raw_d")); ap.add_argument("--timing-json", default=str(HERE / "timing_fresh_d_result.json"))
    ap.add_argument("--window-target-seconds", type=float, default=15.0); ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--only", default=""); ap.add_argument("--run-id-prefix", default="fresh_d_energy_20261002")
    a = ap.parse_args()
    doc, cells, bad = T.load_cells()
    timing = json.loads(Path(a.timing_json).read_text())
    ok_cells = {r["cell_id"]: r for r in timing["cells"] if r.get("correct") and r.get("per_launch_runtime_s")}
    cells = [c for c in cells if c["cell_id"] in ok_cells]  # the cell whose correctness check failed in timing (reduce6, DRAM) is not measured
    if a.only:
        cells = [c for c in cells if a.only in c["cell_id"]]
    if not cells:
        raise SystemExit("no cells selected")
    if a.dry_run:
        print("%d cells; window %.0f s, session %d; graph batch per cell from the committed timing (>= %.0f ms graph)" % (len(cells), a.window_target_seconds, a.session, TARGET_GRAPH_S * 1e3))
        for c in cells:
            print("%-66s per-launch %.3f us  graph batch %d" % (c["cell_id"], ok_cells[c["cell_id"]]["per_launch_runtime_s"] * 1e6, choose_graph_batch(ok_cells[c["cell_id"]]["per_launch_runtime_s"])))
        return 0
    if not a.gate_only and not a.booking_ref.strip():
        raise SystemExit("REFUSED: --booking-ref (the the booking log booking entry for GPU 1) is required")
    cvd, uuid = T.check_device()
    if cvd != str(GPU_INDEX):
        raise SystemExit("REFUSED: energy needs CUDA_VISIBLE_DEVICES=%d exactly (energy_harness/measurement_runner contract), got %r" % (GPU_INDEX, cvd))
    T.refuse_if_busy()
    R = T.import_runners()
    out_dir = Path(a.out_dir).expanduser(); source_root = Path(a.source_root).expanduser()
    workdir = Path(a.workdir).expanduser() if a.workdir else Path(os.path.expanduser("~/fresh_d_energy_work"))
    workdir.mkdir(parents=True, exist_ok=True)
    binaries, nvcc = T.build_binaries(R, source_root, workdir, {c["timing"]["runner"] for c in cells})
    gate = T.hash_gate(cells, binaries, nvcc, workdir)
    print("hash gate passed for %d kernel/binary pairs" % len(gate), flush=True)
    if a.gate_only:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(T.REPO / "energy_harness"))
    import application_energy_harness as H  # noqa: E402
    log = out_dir / "fresh_d_energy_run_log.jsonl"; any_rejected = False
    for i, c in enumerate(cells):
        t = c["timing"]; cid = c["cell_id"]
        run_id = "%s-%s" % (a.run_id_prefix, cid.replace("/", "-"))
        print("cell %d/%d %s run_id=%s" % (i + 1, len(cells), cid, run_id), flush=True)
        batch = choose_graph_batch(ok_cells[cid]["per_launch_runtime_s"])
        try:
            gpu_uuid = H.preflight_before_context(GPU_INDEX, PLATFORM)
            if T.norm_uuid(gpu_uuid) != T.norm_uuid(T.EXPECTED_UUID):
                raise H.RunnerError("uuid_mismatch: nvidia-smi index %d reports %s, not the approved %s" % (GPU_INDEX, gpu_uuid, T.EXPECTED_UUID))
            cd = workdir / ("cell_%02d" % i); files = T.make_inputs(R, c, cd)
            ctx = H.BinaryEnergyContext(
                parent_id=c["operator_id"], regime=c["regime"], candidate_id=c["candidate_id"], source_revision=PINNED_REVISION,
                source_sha256=c["retained_source_sha256"], source_path=c["kernel"],
                controls=dict(family=c["kernel_family"], kernel=c["kernel"], tier=c["tier"], argv_positional=[str(x) for x in t["numeric_args"]] + [str(f) for f in files]),
                runtime_info=dict(nvcc=nvcc, arch="sm_120", driver=t["runner"], binary_sha256=hashlib.sha256(Path(binaries[t["runner"]]).read_bytes()).hexdigest(), host_opt=True),
                binary=Path(binaries[t["runner"]]), argv_prefix=[str(x) for x in t["numeric_args"]] + [str(f) for f in files],
                check=lambda c=c, files=files: bool(T.check_output(R, c, files)))
            result = H.run_binary_energy(ctx, window_target_seconds=a.window_target_seconds, session=a.session, run_id=run_id, out_dir=out_dir, gpu_index=GPU_INDEX,
                                         platform=PLATFORM, runner_name="fresh_d_%s" % t["runner"], gpu_uuid=gpu_uuid, graph_batch=batch)
        except H.RunnerError as exc:
            result = dict(status="rejected", row=dict(timestamp_utc=H.iso_now(), run_id=run_id, session=a.session, runner="fresh_d_%s" % t["runner"], parent_id=c["operator_id"], regime=c["regime"],
                                                      candidate_id=c["candidate_id"], window_target_seconds=a.window_target_seconds, rejection_reason="harness_gate: %s" % str(exc)[:200], raw_stderr_tail=str(exc)[-1000:]))
            H.append_raw_or_rejected(out_dir, result)
            print("STOP: harness gate failed for %s: %s" % (cid, exc), file=sys.stderr)
            return 3
        H.append_raw_or_rejected(out_dir, result)
        row = result["row"]
        with open(log, "a") as f:
            f.write(json.dumps(dict(cell_id=cid, run_id=run_id, status=result["status"], graph_batch=batch, board_energy_j_per_launch=row.get("board_energy_j_per_launch"), rejection_reason=row.get("rejection_reason"),
                                    utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))) + "\n")
        print("  %s %s" % (result["status"].upper(), row.get("rejection_reason", "")), flush=True)
        any_rejected |= result["status"] != "raw"
        shutil.rmtree(cd, ignore_errors=True)
    return 3 if any_rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
