#!/usr/bin/env python3
"""Short runtime timing of the fresh set C PyTorch softmax / layer-norm cells (40 cells, five regimes) on Blackwell GPU 1.

Same protocol as ../fresh/timing_fresh.py (set A) and ../fresh_b/timing_fresh_b.py (set B). Differences: set C cells file and operator ids; large, dram1 and dram2 cells
(dram1 and dram2 are DRAM tier, about 90 and 260 MiB logical per tensor) are never refused for host memory or time; cells run in the order of
fresh_cells_c.json with progress printed per stage; the Python reference check is kept (it takes minutes for the dram2 cell);
an opt-in --skip-reference-above-mib N (OFF by default) skips only the Python reference comparison for cells whose padded
input is larger than N MiB and records that in the result (all_correct is then false). A partial result file
(<out>.partial) is rewritten after every cell so a crash loses nothing; the final --out file is written once.

NOT RUN by the session that wrote it. It uses the GPU, so before running it: check the booking log and nvidia-smi,
and make a booking entry for GPU 1 (pass its reference with --booking-ref). It sets no clocks, no persistence
mode and no power limit; it only allocates tensors, captures CUDA graphs and replays them.

Per cell it builds exactly the call of the development runner
(tiresias/app_runners/pytorch_softmax_runner.py and pytorch_layernorm_runner.py, prepare_energy_context):
    values  = correctness_reference.softmax_input(rows, stride, seed=20260914)
    padded  = torch.tensor(values, float32, cuda).reshape(rows, stride);  view = padded[:, :cols]
    softmax:    F.softmax(view, dim=1)
    layer norm: F.layer_norm(view, [cols], ones(cols), zeros(cols), eps=1e-5)
captures 1000 back-to-back calls in one CUDA graph (the recipe of energy_harness/torch_graph_capture.make_graph_replayer: 3
warm-up calls on a side stream, then capture), does one discarded replay, calibrates the replay time with a
single event-timed replay, picks N so that N replays last about 100 ms, runs one discarded warm window and then
`--windows` timed windows. Each timed window is N replays between two CUDA events:
    per-launch runtime = elapsed_seconds / (N * 1000).
After the last window the output of the last replay is compared with the same reference the runners use
(pytorch_softmax_output_matches / pytorch_layer_norm_output_matches; full output, same tolerances). A failed check
is recorded for the cell, the remaining cells still run, and the script exits 2 at the end (all_correct is false).

Usage:
    python3 timing_fresh_c.py --dry-run
    CUDA_VISIBLE_DEVICES=1 python3 timing_fresh_c.py --booking-ref "<booking log entry>" --out timing_fresh_c_result.json
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RESET = REPO / "tiresias" / "app_runners"
CELLS_JSON = HERE / "fresh_cells_c.json"
EXPECTED_UUID = "GPU-0e63baea-0bb1-e90c-f19d-8aa5f2995894"
SEED = 20260914  # SEED of both runners
GRAPH_LAUNCHES = 1000
WARMUP_LAUNCHES = 3  # energy_harness/torch_graph_capture.WARMUP_LAUNCHES
TARGET_WINDOW_S = 0.100
SOFTMAX_OP = "fresh_c_pytorch_rowwise_softmax"
LAYERNORM_OP = "fresh_c_pytorch_layer_norm"


def norm_uuid(s: str) -> str:
    s = str(s).lower().replace("gpu-", "")
    return re.sub(r"[^0-9a-f]", "", s)


def load_cells():
    doc = json.loads(CELLS_JSON.read_text())
    return doc, doc["cells"]


def call_text(c) -> str:
    rows, cols, stride = c["rows"], c["cols"], c["row_stride"]
    head = ("padded = torch.tensor(softmax_input(%d, %d, seed=%d), dtype=torch.float32, device='cuda').reshape(%d, %d); "
            "view = padded[:, :%d]; " % (rows, stride, SEED, rows, stride, cols))
    if c["operator_id"] == SOFTMAX_OP:
        return head + "launch = lambda: F.softmax(view, dim=1)"
    return head + ("weight = torch.ones(%d); bias = torch.zeros(%d); "
                   "launch = lambda: F.layer_norm(view, [%d], weight, bias, eps=1e-05)" % (cols, cols, cols))


def dry_run(cells) -> int:
    print("fresh cells: %d (no torch import, no GPU)" % len(cells))
    print("graph: %d launches per capture, %d warm-up launches, discarded replay, ~%.0f ms timed windows"
          % (GRAPH_LAUNCHES, WARMUP_LAUNCHES, TARGET_WINDOW_S * 1e3))
    for c in cells:
        print("%-62s rows=%-6d cols=%-4d stride=%-4d kernels/launch=%d tier=%s" % (
            c["cell_id"], c["rows"], c["cols"], c["row_stride"], c["kernels_per_launch"], c["tier"]))
        print("    " + call_text(c))
    return 0


def smi_snapshot(uuid):
    try:
        out = subprocess.run(["nvidia-smi", "-i", uuid, "--query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else "nvidia-smi rc=%d" % out.returncode
    except (OSError, subprocess.TimeoutExpired) as ex:
        return "nvidia-smi unavailable: %s" % ex


def refuse_if_busy(uuid):
    """Never time on a GPU that another process is using, and never proceed if that cannot be checked."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as ex:
        raise SystemExit("REFUSED: cannot run nvidia-smi to check that GPU 1 is idle (%s)" % ex)
    if out.returncode != 0:
        raise SystemExit("REFUSED: nvidia-smi compute-apps query failed (rc=%d)" % out.returncode)
    busy = [ln for ln in out.stdout.splitlines() if norm_uuid(ln.split(",")[0]) == norm_uuid(uuid)]
    if busy:
        raise SystemExit("REFUSED: GPU 1 has compute processes (check the booking log; never kill a process this project did not start):\n"
                         + "\n".join(busy))


def check_device(torch):
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd or "," in cvd:
        raise SystemExit("REFUSED: CUDA_VISIBLE_DEVICES must name exactly one device (got %r)" % cvd)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("REFUSED: expected exactly one visible CUDA device, found %d" % (torch.cuda.device_count() if torch.cuda.is_available() else 0))
    props = torch.cuda.get_device_properties(0)
    uuid = getattr(props, "uuid", None)
    if uuid is None or norm_uuid(uuid) != norm_uuid(EXPECTED_UUID):
        raise SystemExit("REFUSED: visible device UUID %r is not Blackwell GPU 1 (%s)" % (uuid, EXPECTED_UUID))
    return props


def make_launch(torch, c, values):
    F = torch.nn.functional
    rows, cols, stride = c["rows"], c["cols"], c["row_stride"]
    padded = torch.tensor(values, dtype=torch.float32, device="cuda").reshape(rows, stride)
    view = padded[:, :cols]
    assert view.stride(0) == stride and view.stride(1) == 1, "strided view lost the catalog layout"
    holder = {}
    if c["operator_id"] == SOFTMAX_OP:
        def launch_once():
            holder["out"] = F.softmax(view, dim=1)
    else:
        weight = torch.ones(cols, dtype=torch.float32, device="cuda")
        bias = torch.zeros(cols, dtype=torch.float32, device="cuda")

        def launch_once():
            holder["out"] = F.layer_norm(view, [cols], weight, bias, eps=1e-5)
    return launch_once, holder, (padded, view)


def capture(torch, launch_once):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(WARMUP_LAUNCHES):
            launch_once()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(GRAPH_LAUNCHES):
            launch_once()
    torch.cuda.synchronize()
    return graph


def timed(torch, fn, reps):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0  # seconds


def run(args, doc, cells) -> int:
    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit("REFUSED: %s exists; choose a new --out" % out_path)
    partial_path = Path(str(out_path) + ".partial")
    if partial_path.exists():
        raise SystemExit("REFUSED: %s exists (partial result of an earlier run); move it aside first" % partial_path)
    if not args.booking_ref.strip():
        raise SystemExit("REFUSED: --booking-ref (the the booking log booking entry for GPU 1) is required")
    # Fail before importing torch if the environment is obviously wrong.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd or "," in cvd:
        raise SystemExit("REFUSED: CUDA_VISIBLE_DEVICES must name exactly one device (got %r)" % cvd)
    import torch  # noqa: E402  (deliberately late: --dry-run never imports it)
    props = check_device(torch)
    refuse_if_busy(EXPECTED_UUID)
    sys.path.insert(0, str(RESET))
    from correctness_reference import (pytorch_layer_norm_output_matches, pytorch_softmax_output_matches,  # noqa: E402
                                       softmax_input)
    result = {
        "schema": "fresh_c_timing_raw/1",
        "booking_ref": args.booking_ref,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": {"name": props.name, "uuid": str(props.uuid), "cuda_visible_devices": cvd,
                   "torch": torch.__version__, "torch_cuda": torch.version.cuda},
        "fresh_cells_b_json_sha256": hashlib.sha256(CELLS_JSON.read_bytes()).hexdigest(),
        "method": {"graph_launches": GRAPH_LAUNCHES, "warmup_launches": WARMUP_LAUNCHES, "target_window_s": TARGET_WINDOW_S,
                   "windows": args.windows, "per_launch_runtime_s": "median over windows of elapsed_s / (replays_per_window * graph_launches)",
                   "clocks_set": False},
        "cells": [],
    }
    failed = False
    skipped = []
    t_start = time.time()

    def say(msg):
        print("[%7.1fs] %s" % (time.time() - t_start, msg), flush=True)

    for idx, c in enumerate(cells):
        rows, cols, stride = c["rows"], c["cols"], c["row_stride"]
        say("cell %d/%d %s: generating input (%d values)" % (idx + 1, len(cells), c["cell_id"], rows * stride))
        values = softmax_input(rows, stride, seed=SEED)
        say("  uploading tensor")
        launch_once, holder, _keep = make_launch(torch, c, values)
        rec = {"cell_id": c["cell_id"], "rows": rows, "cols": cols, "row_stride": stride, "kernels_per_launch": c["kernels_per_launch"],
               "smi_before": smi_snapshot(EXPECTED_UUID)}
        say("  capturing graph (%d launches)" % GRAPH_LAUNCHES)
        graph = capture(torch, launch_once)
        say("  timing")
        graph.replay()
        torch.cuda.synchronize()  # one discarded replay
        calib_s = timed(torch, graph.replay, 1)
        n = max(1, round(TARGET_WINDOW_S / calib_s))
        rec["calibration_replay_s"] = calib_s
        rec["replays_per_window"] = n
        timed(torch, graph.replay, n)  # discarded warm window
        raw = [timed(torch, graph.replay, n) for _ in range(args.windows)]
        rec["window_elapsed_s_raw"] = raw
        rec["window_per_launch_s"] = [t / (n * GRAPH_LAUNCHES) for t in raw]
        srt = sorted(rec["window_per_launch_s"])
        rec["per_launch_runtime_s"] = srt[len(srt) // 2] if len(srt) % 2 else 0.5 * (srt[len(srt) // 2 - 1] + srt[len(srt) // 2])
        rec["per_launch_min_s"], rec["per_launch_max_s"] = srt[0], srt[-1]
        rec["window_seconds_mean"] = sum(raw) / len(raw)
        rec["smi_after"] = smi_snapshot(EXPECTED_UUID)
        padded_mib = rows * stride * 4 / 2**20
        if args.skip_reference_above_mib is not None and padded_mib > args.skip_reference_above_mib:
            rec["correct"] = None
            rec["reference_skipped"] = True
            skipped.append(c["cell_id"])
            say("  Python reference check SKIPPED (padded input %.1f MiB > --skip-reference-above-mib %s)" % (padded_mib, args.skip_reference_above_mib))
        else:
            say("  reference check (pure Python, full output; slow for large cells)")
            output = holder["out"].detach().cpu().reshape(-1).tolist()
            match = (pytorch_softmax_output_matches if c["operator_id"] == SOFTMAX_OP else pytorch_layer_norm_output_matches)
            rec["correct"] = bool(match(rows, cols, stride, output, values))
            rec["reference_skipped"] = False
            failed |= not rec["correct"]
            del output
        result["cells"].append(rec)
        say("%-62s N=%-5d %.3f us/launch  correct=%s" % (c["cell_id"], n, rec["per_launch_runtime_s"] * 1e6, rec["correct"]))
        partial_path.write_text(json.dumps(result, indent=1) + "\n")
        del graph, holder, launch_once, _keep, values
        gc.collect()
        torch.cuda.empty_cache()
    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result["all_correct"] = (not failed) and not skipped
    result["reference_skipped_cells"] = skipped
    out_path.write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", out_path)
    if failed:
        print("CORRECTNESS FAILURE in at least one cell; do not use this file", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="list cells and calls; imports no torch, touches no GPU")
    ap.add_argument("--out", default=str(HERE / "timing_fresh_c_result.json"))
    ap.add_argument("--booking-ref", default="", help="the booking log entry that books GPU 1 for this run (required to run)")
    ap.add_argument("--windows", type=int, default=5, help="timed windows per cell (each ~100 ms)")
    ap.add_argument("--only", default="", help="substring filter on cell ids (default: all 40 cells)")
    ap.add_argument("--skip-reference-above-mib", type=float, default=None,
                    help="OFF by default. If set, skip the pure-Python reference comparison for cells whose padded input is larger than N MiB "
                         "(recorded as correct=null and reference_skipped; all_correct becomes false). Do not use unless the full check is infeasible.")
    args = ap.parse_args()
    doc, cells = load_cells()
    if args.only:
        cells = [c for c in cells if args.only in c["cell_id"]]
    if not cells:
        raise SystemExit("no cells selected")
    if args.dry_run:
        return dry_run(cells)
    if args.windows < 3:
        raise SystemExit("--windows must be at least 3")
    return run(args, doc, cells)


if __name__ == "__main__":
    raise SystemExit(main())
