#!/usr/bin/env python3
"""Build, SASS hash gate, runtime timing and energy of the unseen machine-learning kernel set on Blackwell GPU 1 (fresh set F). One driver binary (gpu/drivers/driver_ml.cu + src/ml_kernels.cuh).

  python3 run_f.py list                                                   cells and the driver argv of each; runs nothing
  python3 run_f.py build  --workdir W                                     compile the driver with CUDA 13.2.78 (CPU only)
  python3 run_f.py gate   --workdir W                                     compare the SASS of every kernel in the built binary with the analysed (isolated) SASS (CPU only)
  python3 run_f.py time   --workdir W --booking-ref REF --out timing.json one probe, one calibration run, one discarded warm run and 5 timed windows of ~100 ms per cell
  python3 run_f.py energy --workdir W --booking-ref REF --timing-json timing.json --out-dir D   one 15 s window per cell through energy_harness/application_energy_harness.run_binary_energy

Guards (time, energy): a booking log booking reference; CUDA_VISIBLE_DEVICES = the approved GPU 1 UUID (energy needs index 1 for the harness); GPU idle before every cell; the hash gate must pass;
one attempt per cell, no retry; no clock, persistence or power-limit change; no profiler. Nothing here changes a prediction: frozen predictions are committed before `time` is ever run."""
from __future__ import annotations

import argparse, hashlib, json, math, os, re, shutil, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
APPROVED_UUID = "GPU-0e63baea-0bb1-e90c-f19d-8aa5f2995894"
GPU_INDEX, PLATFORM, ARCH = 1, "blackwell", "sm_120"
# Substring list for the HARNESS_ALLOW_PASSIVE_DISPLAY_CONTEXT opt-in (default off): must equal
# energy_harness/measurement_runner.py ADA_BENIGN_COMPUTE_APP_SUBSTRINGS (checked by test); anything else refuses.
_BENIGN_DISPLAY_PROCS = ("snapd-desktop-integration",)
NVCC_RELEASE = "Cuda compilation tools, release 13.2, V13.2.78"
TARGET_GRAPH_S, MAX_GRAPH_BATCH, TARGET_WINDOW_S, WINDOWS = 5e-3, 2048, 0.100, 5
FAM = {"gelu": "gelu", "swiglu": "swiglu", "rmsnorm": "rmsnorm", "rope": "rope", "sgemm": "sgemm"}
SYMBOL = {"gelu_s": "_Z6gelu_sPfS_", "gelu_v4": None}  # filled from the isolated SASS headers below


def norm_uuid(s): return re.sub(r"[^0-9a-f]", "", str(s).lower().replace("gpu-", ""))
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


SET = os.environ.get("FRESH_SET", "f")                       # "f": machine-learning kernels, "g": tensor-core matrix multiply, "h": fused attention
SETDIR = HERE if SET == "f" else HERE.parent / ("fresh_" + SET)


def load_cells():
    return json.loads((SETDIR / ("fresh_cells_%s.json" % SET)).read_text())["cells"]


def argv_of(cell):
    """Driver positional arguments (before out.bin) of a cell; variant 0 = c1, 1 = c2."""
    c, v = cell["controls"], 0 if cell["candidate_id"] == "c1" else 1
    f = cell["family"]
    if f in ("gelu", "swiglu"): return [f, str(v), str(c["n"])]
    if f == "rmsnorm": return [f, str(v), str(c["rows"]), str(c["cols"])]
    if f == "attn": return [f, str(v), str(c["BH"]), str(c["S"])]
    if f == "rope": return [f, str(v), str(c["seq"]), str(c["batch"])]
    if f in ("sgemm", "tcgemm"): return [f, str(v), str(c["M"]), str(c["N"]), str(c["K"])]
    raise ValueError(f)


def choose_graph_batch(t):
    return int(min(MAX_GRAPH_BATCH, max(1, math.ceil(TARGET_GRAPH_S / t))))


def trailer(repeat, batch, d): return ["--repeat", str(repeat), "--graph-batch", str(batch), "--trace-dir", str(d)]


def read_check(out_path):
    p = Path(str(out_path) + ".check")
    if not p.is_file(): return False, "CHECK_MISSING"
    line = p.read_text().strip()
    return line.startswith("CHECK_OK"), line


def parse_windows(p):
    import csv
    rows = list(csv.DictReader(open(p, newline="")))
    if len(rows) != 1: raise ValueError("windows.csv has %d rows" % len(rows))
    r = rows[0]; sec = float(r["cuda_seconds"])
    if not sec > 0: raise ValueError("non-positive cuda_seconds")
    return dict(launches=int(r["launches"]), cuda_seconds=sec, per_launch_s=sec / int(r["launches"]))


def smi(args, timeout=30):
    out = subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0: raise SystemExit("REFUSED: nvidia-smi %s failed" % " ".join(args))
    return out.stdout


def guard(mode, booking):
    if not booking.strip(): raise SystemExit("REFUSED: --booking-ref (the the booking log booking entry for GPU 1) is required")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not re.fullmatch(r"\d+", cvd): raise SystemExit("REFUSED: CUDA_VISIBLE_DEVICES must be one numeric GPU index, got %r" % cvd)
    if mode == "energy" and cvd != str(GPU_INDEX): raise SystemExit("REFUSED: energy needs CUDA_VISIBLE_DEVICES=%d" % GPU_INDEX)
    uuid = smi(["-i", cvd, "--query-gpu=uuid", "--format=csv,noheader"]).strip()
    if norm_uuid(uuid) != norm_uuid(APPROVED_UUID): raise SystemExit("REFUSED: GPU index %s is %s, not Blackwell GPU 1" % (cvd, uuid))
    idle()
    return uuid


def idle():
    out = smi(["--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"])
    busy = [l for l in out.splitlines() if l.strip() and norm_uuid(l.split(",")[0]) == norm_uuid(APPROVED_UUID)]
    if busy and os.environ.get("HARNESS_ALLOW_PASSIVE_DISPLAY_CONTEXT") == "1":
        # Opt-in (default off): tolerate a passive desktop context (same named list as energy_harness/measurement_runner.py
        # ADA_BENIGN_COMPUTE_APP_SUBSTRINGS, M376). Anything else refuses exactly as without the opt-in.
        busy = [l for l in busy if not any(s in l for s in _BENIGN_DISPLAY_PROCS)]
    if busy: raise SystemExit("REFUSED: GPU 1 has compute processes (never kill a process this project did not start):\n" + "\n".join(busy))


def find_nvcc():
    n = os.environ.get("HARNESS_NVCC", "").strip() or "/usr/local/cuda-13.2/bin/nvcc"
    if not Path(n).is_file(): raise SystemExit("BUILD_BLOCKED: nvcc %s not found" % n)
    return n


def cmd_list(a):
    for c in load_cells(): print("%-62s %-4s %s" % (c["cell_id"], c["tier"], " ".join(argv_of(c) + ["<out.bin>"])))
    return 0


def cmd_build(a):
    nvcc = find_nvcc(); ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    if NVCC_RELEASE not in ver: raise SystemExit("BUILD_BLOCKED: nvcc is not CUDA 13.2.78:\n" + ver)
    w = Path(a.workdir).expanduser().resolve(); w.mkdir(parents=True, exist_ok=True)
    binary = w / "driver_ml"; cmd = [nvcc, "-arch=" + ARCH, "-Xcompiler", "-O2", "-I", str(HERE / "src"), "-I", str(HERE.parent / "fresh_g" / "src"), "-I", str(HERE.parent / "fresh_h" / "src"), "-I", str(HERE / "gpu/drivers"), str(HERE / "gpu/drivers/driver_ml.cu"), "-o", str(binary)]
    p = subprocess.run(cmd, capture_output=True, text=True); (w / "build.log").write_text("$ %s\n%s%s" % (" ".join(cmd), p.stdout, p.stderr))
    if p.returncode or not binary.is_file(): raise SystemExit("BUILD FAILED, see %s" % (w / "build.log"))
    man = dict(schema="fresh_f_build/1", nvcc=ver.strip(), command=cmd, driver_ml_cu_sha256=sha256_file(HERE / "gpu/drivers/driver_ml.cu"), driver_common_h_sha256=sha256_file(HERE / "gpu/drivers/driver_common.h"),
               ml_kernels_cuh_sha256=sha256_file(HERE / "src/ml_kernels.cuh"), tc_kernels_cuh_sha256=sha256_file(HERE.parent / "fresh_g/src/tc_kernels.cuh"), attn_kernels_cuh_sha256=sha256_file(HERE.parent / "fresh_h/src/attn_kernels.cuh"), binary=str(binary), binary_sha256=sha256_file(binary))
    (w / "manifest.json").write_text(json.dumps(man, indent=1) + "\n"); print("built", binary, man["binary_sha256"][:16]); return 0


def manifest(w):
    m = json.loads((Path(w).expanduser().resolve() / "manifest.json").read_text())
    if sha256_file(m["binary"]) != m["binary_sha256"]: raise SystemExit("binary changed since the build; rebuild")
    return m


def cmd_gate(a):
    m = manifest(a.workdir); cuobjdump = str(Path(find_nvcc()).with_name("cuobjdump")); sys.path.insert(0, str(HERE.parent / "fresh_d"))
    import timing_fresh_d as T
    ok_all = True; rec = []
    for p in sorted(q for q in (SETDIR / "isolated").glob("*.isolated.sass") if not q.name.startswith("._")):   # skip macOS AppleDouble files that tar can add
        text = p.read_text(); sym = re.match(r"Function : (\S+)", text).group(1)
        d = subprocess.run([cuobjdump, "-sass", "-fun", sym, "-arch", ARCH, m["binary"]], capture_output=True, text=True)
        if d.returncode or "/*" not in d.stdout: ok, why = False, "cuobjdump found no function %s" % sym
        else: ok, why = T.sass_identical(text, d.stdout)
        print("hash gate %-12s %-52s %s" % (p.name.split(".")[0], sym[:52], ("PASS " if ok else "FAIL ") + why)); ok_all &= ok; rec.append(dict(kid=p.name.split(".")[0], symbol=sym, sass_equal=ok, detail=why))
    (Path(a.workdir).expanduser().resolve() / "gate.json").write_text(json.dumps(rec, indent=1)); return 0 if ok_all else 2


def run_driver(binary, pos, out_path, repeat, batch, d, env, timeout=3600):
    Path(d).mkdir(parents=True, exist_ok=True)
    argv = [binary] + pos + [str(out_path)] + trailer(repeat, batch, d); wcsv = Path(d) / "windows.csv"
    if wcsv.exists(): wcsv.unlink()
    p = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=timeout)
    ok, line = read_check(out_path)
    if p.returncode not in (0, 1): raise SystemExit("REFUSED: driver exit %d: %s" % (p.returncode, p.stderr[-400:]))
    return dict(rc=p.returncode, check_ok=ok, check=line, windows=parse_windows(wcsv))


def median(x): return statistics.median(x)


def cmd_time(a):
    uuid = guard("time", a.booking_ref); m = manifest(a.workdir)
    if not (Path(a.workdir).expanduser().resolve() / "gate.json").is_file(): raise SystemExit("REFUSED: run the gate first")
    if not all(r["sass_equal"] for r in json.loads((Path(a.workdir).expanduser().resolve() / "gate.json").read_text())): raise SystemExit("REFUSED: the hash gate failed")
    out = Path(a.out); part = Path(str(out) + ".partial")
    if out.exists() or part.exists(): raise SystemExit("REFUSED: %s or its .partial exists" % out)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=APPROVED_UUID, UNSEEN_EXPECT_UUID=APPROVED_UUID); w = Path(a.workdir).expanduser().resolve()
    cells = [c for c in load_cells() if not a.only or a.only in c["cell_id"]]; res = dict(schema="fresh_f_timing/1", booking_ref=a.booking_ref, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                                                           manifest=m, cells=[]); failed = False; t0 = time.time()
    for i, c in enumerate(cells):
        idle(); pos = argv_of(c); out_bin = w / "t_out.bin"
        pr = run_driver(m["binary"], pos, out_bin, 8, 8, w / "t_probe", env)             # probe: per-launch estimate
        batch = choose_graph_batch(pr["windows"]["per_launch_s"])
        cal = run_driver(m["binary"], pos, out_bin, batch, batch, w / "t_cal", env)
        t1 = cal["windows"]["per_launch_s"]; replays = max(1, math.ceil(TARGET_WINDOW_S / (batch * t1))); repeat = replays * batch
        run_driver(m["binary"], pos, out_bin, repeat, batch, w / "t_warm", env)          # discarded warm window
        raw = [run_driver(m["binary"], pos, out_bin, repeat, batch, w / ("t_%d" % k), env) for k in range(WINDOWS)]
        per = [r["windows"]["per_launch_s"] for r in raw]; correct = all(r["check_ok"] for r in raw) and pr["check_ok"] and cal["check_ok"]; failed |= not correct
        rec = dict(cell_id=c["cell_id"], argv=pos, graph_batch=batch, repeat_per_window=repeat, probe_per_launch_s=pr["windows"]["per_launch_s"], calibration_per_launch_s=t1, window_per_launch_s=per,
                   per_launch_runtime_s=median(per), per_launch_min_s=min(per), per_launch_max_s=max(per), correct=correct, check=raw[-1]["check"])
        res["cells"].append(rec); part.write_text(json.dumps(res, indent=1) + "\n")
        print("[%7.1fs] %d/%d %-62s batch=%-5d repeat=%-7d %.3f us/launch correct=%s" % (time.time() - t0, i + 1, len(cells), c["cell_id"], batch, repeat, rec["per_launch_runtime_s"] * 1e6, correct), flush=True)
    res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); res["all_correct"] = not failed
    out.write_text(json.dumps(res, indent=1) + "\n"); print("wrote", out); return 2 if failed else 0


def cmd_energy(a):
    uuid = guard("energy", a.booking_ref); m = manifest(a.workdir); w = Path(a.workdir).expanduser().resolve()
    if not all(r["sass_equal"] for r in json.loads((w / "gate.json").read_text())): raise SystemExit("REFUSED: the hash gate failed")
    timing = json.loads(Path(a.timing_json).read_text()); ok = {r["cell_id"]: r for r in timing["cells"] if r.get("correct") and r.get("per_launch_runtime_s")}
    cells = [c for c in load_cells() if c["cell_id"] in ok and (not a.only or a.only in c["cell_id"])]
    os.environ["UNSEEN_EXPECT_UUID"] = APPROVED_UUID      # the driver's device guard (driver_common.h) requires it; the harness passes the environment on
    sys.path.insert(0, str(REPO / "energy_harness")); import application_energy_harness as H
    out_dir = Path(a.out_dir).expanduser(); out_dir.mkdir(parents=True, exist_ok=True); log = out_dir / "fresh_f_energy_run_log.jsonl"; any_rej = False
    for i, c in enumerate(cells):
        cid = c["cell_id"]; run_id = "%s-%s" % (a.run_id_prefix, cid.replace("/", "-")); pos = argv_of(c); batch = choose_graph_batch(ok[cid]["per_launch_runtime_s"])
        print("cell %d/%d %s run_id=%s batch=%d" % (i + 1, len(cells), cid, run_id, batch), flush=True)
        try:
            gpu_uuid = H.preflight_before_context(GPU_INDEX, PLATFORM)
            if norm_uuid(gpu_uuid) != norm_uuid(APPROVED_UUID): raise H.RunnerError("uuid_mismatch: %s" % gpu_uuid)
            outb = w / "e_out.bin"
            ctx = H.BinaryEnergyContext(parent_id=c["operator_id"], regime=c["regime"], candidate_id=c["candidate_id"], source_revision="own-source", source_sha256=m["ml_kernels_cuh_sha256"], source_path="src/ml_kernels.cuh",
                                        controls=dict(family=c["family"], tier=c["tier"], argv_positional=pos), runtime_info=dict(nvcc=m["nvcc"], arch=ARCH, driver="driver_ml", binary_sha256=m["binary_sha256"], host_opt=True),
                                        binary=Path(m["binary"]), argv_prefix=pos + [str(outb)], check=lambda p=outb: read_check(p)[0])
            result = H.run_binary_energy(ctx, window_target_seconds=a.window_target_seconds, session=a.session, run_id=run_id, out_dir=out_dir, gpu_index=GPU_INDEX, platform=PLATFORM,
                                         runner_name="fresh_f_driver_ml", gpu_uuid=gpu_uuid, graph_batch=batch)
        except H.RunnerError as exc:
            result = dict(status="rejected", row=dict(timestamp_utc=H.iso_now(), run_id=run_id, session=a.session, runner="fresh_f_driver_ml", parent_id=c["operator_id"], regime=c["regime"], candidate_id=c["candidate_id"],
                                                      window_target_seconds=a.window_target_seconds, rejection_reason="harness_gate: %s" % str(exc)[:200], raw_stderr_tail=str(exc)[-1000:]))
            H.append_raw_or_rejected(out_dir, result); print("STOP: harness gate failed for %s: %s" % (cid, exc), file=sys.stderr); return 3
        H.append_raw_or_rejected(out_dir, result); row = result["row"]
        with open(log, "a") as f:
            f.write(json.dumps(dict(cell_id=cid, run_id=run_id, status=result["status"], graph_batch=batch, board_energy_j_per_launch=row.get("board_energy_j_per_launch"), rejection_reason=row.get("rejection_reason"), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))) + "\n")
        print("  %s %s" % (result["status"].upper(), row.get("rejection_reason", "")), flush=True); any_rej |= result["status"] != "raw"
    return 3 if any_rej else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("list", "build", "gate", "time", "energy"):
        p = sp.add_parser(n); p.add_argument("--workdir", default=os.path.expanduser("~/fresh_f_work")); p.add_argument("--only", default="")
        if n in ("time", "energy"): p.add_argument("--booking-ref", default="")
        if n == "time": p.add_argument("--out", default=str(HERE / "timing_fresh_f_result.json"))
        if n == "energy": p.add_argument("--timing-json", default=str(HERE / "timing_fresh_f_result.json")); p.add_argument("--out-dir", default=str(HERE / "energy_raw_f")); p.add_argument("--window-target-seconds", type=float, default=15.0); p.add_argument("--session", type=int, default=1); p.add_argument("--run-id-prefix", default="fresh_f_energy_20261003")
    a = ap.parse_args(); return dict(list=cmd_list, build=cmd_build, gate=cmd_gate, time=cmd_time, energy=cmd_energy)[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
