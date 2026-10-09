#!/usr/bin/env python3
"""Short runtime timing of the fresh set D CUDA Samples cells (28 cells) on Blackwell GPU 1.  NOT RUN by the session that wrote it.

Same protocol and guards as ../fresh_b/timing_fresh_b.py, with the existing C++ harness drivers in place of PyTorch calls:
  * per cell the driver of the development runner is invoked with the NEW shape (copy_runner, transpose_runner and
    reduction_runner under tiresias/app_runners/, tile_family under tiresias/framework/unseen_operators/runners/); their
    own build_binary(), seeded input generators and oracles are imported, nothing is edited or re-derived;
  * each driver process captures `--graph-batch 1000` launches into one CUDA graph, does one discarded replay, then replays
    `--repeat` launches between two CUDA events and writes windows.csv (block, launches, host ns, cuda_seconds);
  * calibration run (repeat = 1000, one replay) -> N replays per window so that a window lasts about 100 ms (at least one replay);
    one discarded warm run; `--windows` (default 5) timed runs; per-launch runtime = cuda_seconds / repeat; the cell value is the
    median of the windows. One driver process is one window (each re-uploads its input outside the timed region).
  * after the last window the output is compared with the runner's own oracle (exact for copies and transposes, tolerance of
    reduction_runner for sums). A failed check is recorded; the script exits 2 at the end.

HASH GATE (runs before anything is timed; `--gate-only` runs just this, builds the binaries and launches no kernel): the
kernel actually linked into each freshly built harness binary must be the retained build. For every kernel symbol used, the SASS of
that function is dumped from the binary (cuobjdump -sass -fun <symbol> -arch sm_120) and its instruction sequence (pc, predicate,
opcode, operands) must equal that of the retained isolated SASS in
compile_evidence/acquisition_runs/cuda_compile_blackwell_20261001_51afb2f3 (hash-checked against the retention manifest). The
cubin container sha256 is reported when it is byte-identical (informational). nvcc must report CUDA 13.2.78, the version of the
retention. On any mismatch the script REFUSES and prints what must be built: the pinned sources at revision 5443602d with
`nvcc -O2 -arch=sm_120 -Dmain=<name>_sample_main_disabled -I <Common>` from CUDA 13.2.78 (the retained compile arguments).

Guards: --booking-ref (the booking log booking of GPU 1) required; CUDA_VISIBLE_DEVICES one numeric index whose nvidia-smi UUID is the
approved one (children get the UUID itself); GPU idle (no compute process) at start and before every cell; refuses if --out or
<out>.partial exists. No clocks, persistence mode or power limit is touched; no profiler; no energy window.

Usage (from this directory):
    python3 timing_fresh_d.py --dry-run
    CUDA_VISIBLE_DEVICES=1 python3 timing_fresh_d.py --gate-only --source-root /home/user/cudasamples_pinned_5443602d
    CUDA_VISIBLE_DEVICES=1 python3 timing_fresh_d.py --booking-ref "<booking log entry>" --source-root <pinned checkout> --out timing_fresh_d_result.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RESET = REPO / "tiresias" / "app_runners"
RUNNERS = REPO / "tiresias" / "framework" / "unseen_operators" / "runners"
CELLS_JSON = Path(os.environ.get("FRESH_D_CELLS_JSON", str(HERE / "fresh_cells_d.json")))   # env override: CPU tests on a partial build
RETENTION = (REPO / "tiresias/framework/compile_evidence/acquisition_runs/cuda_compile_blackwell_20261001_51afb2f3/output/nvcc")
EXPECTED_UUID = "GPU-0e63baea-0bb1-e90c-f19d-8aa5f2995894"
# Substring list for the HARNESS_ALLOW_PASSIVE_DISPLAY_CONTEXT opt-in (default off): must equal
# energy_harness/measurement_runner.py ADA_BENIGN_COMPUTE_APP_SUBSTRINGS (checked by test); anything else refuses.
_BENIGN_DISPLAY_PROCS = ("snapd-desktop-integration",)
GRAPH_BATCH = 1000
TARGET_WINDOW_S = 0.100
RETAINED_NVCC_RELEASE = "Cuda compilation tools, release 13.2, V13.2.78"
DEFAULT_SOURCE_ROOT = "/home/user/cudasamples_pinned_5443602d"
RUNNER_BINARY = {"copy_runner": "vectorAdd harness (copy_runner.build_binary)", "transpose_runner": "transpose harness (transpose_runner.build_binary)",
                 "tile_family": "tile-family harness (tile_family.build_binary)", "reduction_runner": "reduction harness (reduction_runner.build_binary)"}

LINE = re.compile(r"\s*/\*([0-9a-f]+)\*/\s+(?:@(!?(?:U?P\d+|U?PT))\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;")


# ------------------------------------------------------------------------------------------------ pure helpers (CPU tests)
def norm_uuid(s):
    return re.sub(r"[^0-9a-f]", "", str(s).lower().replace("gpu-", ""))


def parse_sass(text):
    """Instruction sequence of a cuobjdump -sass listing: (pc, predicate, opcode, operands). Encoding lines are ignored."""
    out = []
    for line in text.splitlines():
        if not re.match(r"\s*/\*[0-9a-f]+\*/\s+[A-Z@]", line):
            continue
        m = LINE.match(line)
        if m is None:
            raise ValueError("unparsed SASS line: " + line)
        pc, pred, op, args = m.groups()
        out.append((int(pc, 16), pred, op, tuple(a.strip() for a in args.split(","))))
    if not out:
        raise ValueError("no SASS instructions found")
    return out


def sass_identical(retained_text, built_text):
    a, b = parse_sass(retained_text), parse_sass(built_text)
    if a == b:
        return True, "identical (%d instructions)" % len(a)
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return False, "differs at instruction %d (retained %s, built %s); lengths %d vs %d" % (
        n, a[n] if n < len(a) else None, b[n] if n < len(b) else None, len(a), len(b))


def window_launches_for(calibration_cuda_seconds, calibration_launches=GRAPH_BATCH, target_s=TARGET_WINDOW_S, batch=GRAPH_BATCH):
    """repeat (a multiple of the graph batch) so that one window lasts about target_s; at least one full replay."""
    per_launch = calibration_cuda_seconds / calibration_launches
    replays = max(1, round(target_s / (per_launch * batch)))
    return replays * batch, per_launch


def median(xs):
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def parse_windows_csv(path, expect_launches):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError("windows.csv has %d rows, expected 1" % len(rows))
    r = rows[0]
    if int(r["launches"]) != expect_launches:
        raise RuntimeError("windows.csv launches %s != requested %d" % (r["launches"], expect_launches))
    sec = float(r["cuda_seconds"])
    if not sec > 0:
        raise RuntimeError("non-positive cuda_seconds")
    return sec


def driver_argv(binary, spec, files, repeat=None, trace_dir=None):
    argv = [str(binary)] + [str(a) for a in spec["numeric_args"]] + [str(f) for f in files]
    if repeat is not None:
        argv += ["--repeat", str(repeat), "--graph-batch", str(GRAPH_BATCH), "--trace-dir", str(trace_dir)]
    return argv


def load_cells():
    doc = json.loads(CELLS_JSON.read_text())
    bad = [c["cell_id"] for c in doc["cells"] if not c.get("in_timing_list")]
    return doc, [c for c in doc["cells"] if c.get("in_timing_list")], bad


def retained_row_for(cell):
    m = json.loads((RETENTION / "retention_manifest.json").read_text())
    rows = [r for r in m["rows"] if r["operator_id"] == cell["origin_operator_id"] and r["cell"] == cell["origin_cell"]]
    if len(rows) != 1:
        raise SystemExit("REFUSED: retained row for %s not unique" % cell["cell_id"])
    row = rows[0]
    if hashlib.sha256((RETENTION / row["disassembly_path"]).read_bytes()).hexdigest() != row["disassembly_sha256"]:
        raise SystemExit("REFUSED: retained isolated SASS hash mismatch for " + cell["cell_id"])
    if row["cubin_sha256"] != cell["retained_cubin_sha256"] or row["isolated_function_section"] != cell["kernel_symbol"]:
        raise SystemExit("REFUSED: fresh_cells_d.json disagrees with the retention manifest for " + cell["cell_id"])
    return row


def dry_run(cells, bad):
    print("fresh cells D: %d in timing list (%d excluded as underived: %s); no import of runners, no GPU" % (len(cells), len(bad), bad))
    print("protocol: graph batch %d, 1 calibration run, 1 discarded warm run, windows of about %.0f ms (>= 1 replay), median of 5 timed runs"
          % (GRAPH_BATCH, TARGET_WINDOW_S * 1e3))
    print("hash gate: dump SASS of each used symbol from the freshly built binary and require equality with the retained isolated SASS; nvcc must be CUDA 13.2.78")
    for c in cells:
        t = c["timing"]
        print("%-66s %-16s tier=%-4s %s" % (c["cell_id"], t["runner"], c["tier"], driver_argv("<%s>" % t["runner"], t, ["<in>"] * len(t["file_args"]), "<N*1000>", "<dir>")))
        print("    kernel %s (symbol %s) oracle=%s input=%s elements=%d" % (c["kernel"], c["kernel_symbol"], t["oracle"], t["input_kind"], t["elements"]))
    return 0


# ------------------------------------------------------------------------------------------------ GPU side
def smi(args, timeout=30):
    try:
        out = subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as ex:
        raise SystemExit("REFUSED: cannot run nvidia-smi (%s)" % ex)
    if out.returncode != 0:
        raise SystemExit("REFUSED: nvidia-smi %s failed (rc=%d)" % (" ".join(args), out.returncode))
    return out.stdout


def check_device():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not re.fullmatch(r"\d+", cvd):
        raise SystemExit("REFUSED: CUDA_VISIBLE_DEVICES must be exactly one numeric GPU index (got %r)" % cvd)
    uuid = smi(["-i", cvd, "--query-gpu=uuid", "--format=csv,noheader"]).strip()
    if norm_uuid(uuid) != norm_uuid(EXPECTED_UUID):
        raise SystemExit("REFUSED: GPU index %s has UUID %s, not Blackwell GPU 1 (%s)" % (cvd, uuid, EXPECTED_UUID))
    return cvd, uuid


def refuse_if_busy():
    out = smi(["--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"])
    busy = [ln for ln in out.splitlines() if ln.strip() and norm_uuid(ln.split(",")[0]) == norm_uuid(EXPECTED_UUID)]
    if busy and os.environ.get("HARNESS_ALLOW_PASSIVE_DISPLAY_CONTEXT") == "1":
        # Opt-in (default off): tolerate a passive desktop context (same named list as energy_harness/measurement_runner.py
        # ADA_BENIGN_COMPUTE_APP_SUBSTRINGS, M376). Anything else refuses exactly as without the opt-in.
        busy = [ln for ln in busy if not any(s in ln for s in _BENIGN_DISPLAY_PROCS)]
    if busy:
        raise SystemExit("REFUSED: GPU 1 has compute processes (check the booking log; never kill a process this project did not start):\n" + "\n".join(busy))


def smi_snapshot():
    try:
        return smi(["-i", os.environ.get("CUDA_VISIBLE_DEVICES", "1"), "--query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu",
                    "--format=csv,noheader,nounits"], timeout=20).strip()
    except SystemExit as ex:
        return str(ex)


def find_cuobjdump(nvcc):
    explicit = os.environ.get("HARNESS_CUOBJDUMP", "").strip()
    for cand in (explicit, str(Path(nvcc).with_name("cuobjdump")), shutil.which("cuobjdump") or ""):
        if cand and Path(cand).is_file():
            return cand
    raise SystemExit("REFUSED: cuobjdump not found next to nvcc, on PATH, or in HARNESS_CUOBJDUMP")


def import_runners():
    for p in (RUNNERS, RESET, REPO / "energy_harness"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import copy_runner, transpose_runner, reduction_runner, tile_family, correctness_reference  # noqa: E401
    return dict(copy_runner=copy_runner, transpose_runner=transpose_runner, reduction_runner=reduction_runner, tile_family=tile_family,
                correctness_reference=correctness_reference)


def build_binaries(R, source_root, workdir, runners_needed):
    """Existing build_binary() of each runner, unchanged. Returns {runner name: binary path} and the nvcc path."""
    nvcc = R["copy_runner"].find_nvcc()
    out = {}
    for name in sorted(runners_needed):
        wd = Path(workdir) / name
        wd.mkdir(parents=True, exist_ok=True)
        if name == "copy_runner":
            R[name].verify_source(source_root)
        elif name == "reduction_runner":
            R[name].verify_source(source_root)
        elif name == "transpose_runner":
            R[name].verify_source(source_root)
        else:
            import unseen_common as uc
            uc.verify_source(source_root, (uc.TRANSPOSE_REL,))
        out[name] = R[name].build_binary(Path(source_root).expanduser().resolve(), wd, nvcc)
        print("built %s -> %s" % (RUNNER_BINARY[name], out[name]), flush=True)
    return out, nvcc


def hash_gate(cells, binaries, nvcc, workdir):
    """Refuses (SystemExit) unless every used kernel in every built binary has the retained SASS. Returns the record."""
    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    if RETAINED_NVCC_RELEASE not in ver:
        raise SystemExit("REFUSED: nvcc is not CUDA 13.2.78 (retained build). Found:\n%s\nBuild with the retained toolchain "
                         "(/usr/local/cuda-13.2/bin/nvcc, V13.2.78) so the kernel bytes can match." % ver.strip())
    cuobjdump = find_cuobjdump(nvcc)
    record, failures = [], []
    seen = set()
    for c in cells:
        key = (c["timing"]["runner"], c["kernel_symbol"])
        if key in seen:
            continue
        seen.add(key)
        row = retained_row_for(c)
        dump = subprocess.run([cuobjdump, "-sass", "-fun", c["kernel_symbol"], "-arch", "sm_120", str(binaries[key[0]])],
                              capture_output=True, text=True)
        if dump.returncode != 0 or "/*" not in dump.stdout:
            ok, why = False, "cuobjdump failed or found no function (rc=%d): %s" % (dump.returncode, (dump.stderr or dump.stdout)[-300:])
        else:
            ok, why = sass_identical((RETENTION / row["disassembly_path"]).read_text(), dump.stdout)
        rec = {"runner": key[0], "kernel_symbol": key[1], "retained_cubin_sha256": row["cubin_sha256"], "sass_equal": ok, "detail": why}
        record.append(rec)
        print("hash gate %-16s %-40s %s" % (key[0], key[1], "PASS " + why if ok else "FAIL " + why), flush=True)
        if not ok:
            failures.append(rec)
    # informational: is the embedded cubin byte-identical to the retained one?
    for name, binary in binaries.items():
        xdir = Path(workdir) / ("xelf_" + name)
        xdir.mkdir(exist_ok=True)
        subprocess.run([cuobjdump, "-xelf", "all", str(binary)], cwd=str(xdir), capture_output=True, text=True)
        shas = sorted(hashlib.sha256(p.read_bytes()).hexdigest() for p in xdir.glob("*.cubin"))
        retained = {c["retained_cubin_sha256"] for c in cells if c["timing"]["runner"] == name}
        for r in record:
            if r["runner"] == name:
                r["cubin_container_byte_identical_to_retained"] = bool(retained & set(shas))
    if failures:
        raise SystemExit(
            "REFUSED: the freshly built binary does not contain the retained kernel for %d symbol(s): %s\n"
            "What must be built: the pinned CUDA Samples revision 5443602d89ed99aede2e4b7bf329daddeadb320e compiled exactly as retained "
            "(nvcc -O2 -arch=sm_120 -Dmain=<transpose|vectorAdd|reduction>_sample_main_disabled -I <Common>, CUDA 13.2.78; see "
            "compile_argv_prefix in the retention manifest), e.g. through the runners' build_binary() on a host with that toolchain. "
            "Do not time cells of a kernel whose gate failed." % (len(failures), [f["kernel_symbol"] for f in failures]))
    return record


def make_inputs(R, c, d):
    """Write the runner's own seeded input file(s) for this cell into d; returns the list of file paths (inputs..., output)."""
    t = c["timing"]
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    n = t["elements"]
    if t["input_kind"] == "vecadd_uniform_pair":
        x, y = R["copy_runner"].deterministic_inputs(n)
        with open(d / "x.bin", "wb") as f:
            x.tofile(f)
        with open(d / "y.bin", "wb") as f:
            y.tofile(f)
        return [d / "x.bin", d / "y.bin", d / "out.bin"]
    if t["input_kind"] == "reduction_uniform":
        a = R["reduction_runner"].deterministic_inputs(n)
    elif t["input_kind"] == "transpose_mod8192":
        dim = t["numeric_args"][0]
        a = R["transpose_runner"].deterministic_inputs(dim, dim)
    elif t["input_kind"] == "copy_uniform":
        a = R["tile_family"].copy_input(n)
    elif t["input_kind"] == "transpose_mod2p24":
        a = R["tile_family"].transpose_input(n)
    else:
        raise SystemExit("REFUSED: unknown input kind " + t["input_kind"])
    with open(d / "in.bin", "wb") as f:
        a.tofile(f)
    del a
    return [d / "in.bin", d / "out.bin"]


def read_array(path, count):
    import array
    a = array.array("f")
    with open(path, "rb") as f:
        a.fromfile(f, count)
    return a


def check_output(R, c, files):
    """Runner oracle for the output left by the last window."""
    t = c["timing"]
    n = t["elements"]
    TF = R["tile_family"]
    if t["oracle"] == "vector_add":
        x, y, out = read_array(files[0], n), read_array(files[1], n), read_array(files[2], n)
        expected = R["correctness_reference"].ref_cuda_samples_vector_add(list(x), list(y))
        return R["copy_runner"].first_mismatch(list(out), expected) is None
    if t["oracle"] == "sum_double":
        values = read_array(files[0], n)
        out = TF._read_output(files[1], 1) if hasattr(TF, "_read_output") else read_array(files[1], 1)
        got = out[0] if out is not None else float("nan")
        return bool(R["reduction_runner"].scalar_matches(got, sum(values)))
    inp, out = read_array(files[0], n), read_array(files[1], n)
    dim = t["numeric_args"][0]
    if t["oracle"] == "full_transpose":
        return bool(TF.matches_full_transpose(inp, out, dim))
    if t["oracle"] == "identity":
        return bool(TF.matches_identity(inp, out))
    if t["oracle"].startswith("kernel_semantics:"):
        return bool(TF.matches_kernel_semantics(t["oracle"].split(":", 1)[1], inp, out, dim))
    raise SystemExit("REFUSED: unknown oracle " + t["oracle"])


def run_driver(binary, spec, files, repeat, trace_dir, env):
    Path(trace_dir).mkdir(parents=True, exist_ok=True)
    wcsv = Path(trace_dir) / "windows.csv"
    if wcsv.exists():
        wcsv.unlink()
    p = subprocess.run(driver_argv(binary, spec, files, repeat, trace_dir), capture_output=True, text=True, env=env, timeout=1800)
    if p.returncode != 0:
        raise SystemExit("REFUSED: harness failed (rc=%d): %s" % (p.returncode, p.stderr[-500:]))
    return parse_windows_csv(wcsv, repeat)


def run(args, doc, cells, bad):
    out_path = Path(args.out)
    partial = Path(str(out_path) + ".partial")
    if not args.gate_only:
        if out_path.exists():
            raise SystemExit("REFUSED: %s exists; choose a new --out" % out_path)
        if partial.exists():
            raise SystemExit("REFUSED: %s exists (partial result of an earlier run); move it aside first" % partial)
        if not args.booking_ref.strip():
            raise SystemExit("REFUSED: --booking-ref (the the booking log booking entry for GPU 1) is required")
    cvd, uuid = check_device()
    refuse_if_busy()
    R = import_runners()
    source_root = Path(args.source_root).expanduser()
    t_start = time.time()

    def say(msg):
        print("[%7.1fs] %s" % (time.time() - t_start, msg), flush=True)

    with tempfile.TemporaryDirectory(prefix="fresh_d_") as tmp:
        workdir = Path(args.workdir) if args.workdir else Path(tmp)
        workdir.mkdir(parents=True, exist_ok=True)
        binaries, nvcc = build_binaries(R, source_root, workdir, {c["timing"]["runner"] for c in cells})
        gate = hash_gate(cells, binaries, nvcc, workdir)
        say("hash gate passed for %d kernel/binary pairs" % len(gate))
        if args.gate_only:
            print(json.dumps(gate, indent=1))
            return 0
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=EXPECTED_UUID)
        result = {"schema": "fresh_d_timing_raw/1", "booking_ref": args.booking_ref, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "device": {"uuid": uuid, "cuda_visible_devices_at_launch": cvd, "children_cuda_visible_devices": EXPECTED_UUID},
                  "fresh_cells_d_json_sha256": hashlib.sha256(CELLS_JSON.read_bytes()).hexdigest(), "nvcc": nvcc,
                  "hash_gate": gate, "excluded_underived_cells": bad,
                  "method": {"graph_batch": GRAPH_BATCH, "target_window_s": TARGET_WINDOW_S, "windows": args.windows,
                             "per_launch_runtime_s": "median over windows of cuda_seconds / repeat (repeat = replays * graph_batch)",
                             "one_driver_process_per_window": True, "clocks_set": False},
                  "cells": []}
        failed = False
        for idx, c in enumerate(cells):
            refuse_if_busy()
            t = c["timing"]
            say("cell %d/%d %s: generating input (%d elements)" % (idx + 1, len(cells), c["cell_id"], t["elements"]))
            cd = workdir / ("cell_%02d" % idx)
            files = make_inputs(R, c, cd)
            binary = binaries[t["runner"]]
            rec = {"cell_id": c["cell_id"], "kernel": c["kernel"], "runner": t["runner"], "numeric_args": t["numeric_args"], "smi_before": smi_snapshot()}
            cal = run_driver(binary, t, files, GRAPH_BATCH, cd / "t_cal", env)
            repeat, per_launch0 = window_launches_for(cal)
            rec.update(calibration_cuda_seconds=cal, calibration_per_launch_s=per_launch0, repeat_per_window=repeat, replays_per_window=repeat // GRAPH_BATCH)
            run_driver(binary, t, files, repeat, cd / "t_warm", env)   # discarded warm window
            raw = [run_driver(binary, t, files, repeat, cd / ("t_%d" % w), env) for w in range(args.windows)]
            rec["window_cuda_seconds_raw"] = raw
            rec["window_per_launch_s"] = [x / repeat for x in raw]
            rec["per_launch_runtime_s"] = median(rec["window_per_launch_s"])
            rec["per_launch_min_s"], rec["per_launch_max_s"] = min(rec["window_per_launch_s"]), max(rec["window_per_launch_s"])
            rec["smi_after"] = smi_snapshot()
            say("  reference check")
            rec["correct"] = bool(check_output(R, c, files))
            failed |= not rec["correct"]
            result["cells"].append(rec)
            say("%-66s repeat=%-7d %.3f us/launch correct=%s" % (c["cell_id"], repeat, rec["per_launch_runtime_s"] * 1e6, rec["correct"]))
            partial.write_text(json.dumps(result, indent=1) + "\n")
            shutil.rmtree(cd, ignore_errors=True)
        result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result["all_correct"] = not failed
        out_path.write_text(json.dumps(result, indent=1) + "\n")
        print("wrote", out_path)
        if failed:
            print("CORRECTNESS FAILURE in at least one cell; do not use this file", file=sys.stderr)
            return 2
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gate-only", action="store_true", help="build the binaries and run the SASS hash gate; launch no kernel")
    ap.add_argument("--out", default=str(HERE / "timing_fresh_d_result.json"))
    ap.add_argument("--booking-ref", default="")
    ap.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT, help="pinned CUDA Samples checkout at revision 5443602d")
    ap.add_argument("--workdir", default="")
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--only", default="", help="substring filter on cell ids")
    args = ap.parse_args()
    doc, cells, bad = load_cells()
    if args.only:
        cells = [c for c in cells if args.only in c["cell_id"]]
    if not cells:
        raise SystemExit("no cells selected")
    if args.dry_run:
        return dry_run(cells, bad)
    if args.windows < 3:
        raise SystemExit("--windows must be at least 3")
    return run(args, doc, cells, bad)


if __name__ == "__main__":
    raise SystemExit(main())
