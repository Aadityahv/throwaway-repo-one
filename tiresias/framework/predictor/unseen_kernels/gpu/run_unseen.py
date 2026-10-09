#!/usr/bin/env python3
"""Orchestrator of the unseen-kernel prospective test on Blackwell GPU 1 (32 cells of tiresias/framework/
predictor/unseen_kernels/cells.py: pinned NVIDIA cuda-samples matrix multiply, Black-Scholes, scan and
separable convolution).

Subcommands
    build    compile the four drivers in drivers/ against the pinned sample checkout (CPU only, no GPU is touched);
             write <workdir>/manifest.json with SHA-256 of every driver, sample file, header and binary.
    time     per cell: calibrate a graph batch (graph >= ~5 ms) and a repeat count (window >= ~30 ms), run the
             driver with the trailer `--repeat N --graph-batch B --trace-dir DIR` for a few windows, write a JSON
             {schema, cells:[{cell_id, per_launch_runtime_s, launches, repeat, graph_batch, check, ...}]}.
    energy   per cell: build a BinaryEnergyContext and call energy_harness/application_energy_harness.run_binary_energy (after
             preflight_before_context), exactly like energy_harness/run_application_energy.py does for binary runners, but
             directly (that CLI restricts regimes to small/medium/large and candidates to c1..c4).
    list     print the argv every cell would run; runs nothing (same as --dry-run of time/energy).

Safety (every subcommand that could touch a GPU: time, energy; not build, not list, not --dry-run)
    * refuses to run at all unless --i-have-a-booking is given (a the booking log booking of GPU 1 exists);
    * refuses unless CUDA_VISIBLE_DEVICES names exactly one device and that device is the approved Blackwell GPU 1
      (UUID below), found live through nvidia-smi, and that GPU has no compute process;
    * the drivers repeat the device check themselves (exactly one visible device whose UUID equals the
      UNSEEN_EXPECT_UUID that this script sets), so a mapping error between nvidia-smi and CUDA cannot run on
      another GPU;
    * sets no clocks, no persistence mode, no power limit; makes no retry loop (one attempt per cell, failures are
      recorded as results); never runs nvidia-smi with a setting flag.
The only environment it changes is process-local: CUDA_DEVICE_ORDER=PCI_BUS_ID (so CUDA indices agree with
nvidia-smi / NVML indices) and UNSEEN_EXPECT_UUID.

NOT RUN by the session that wrote it (no GPU, no nvcc). See README.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
UNSEEN = HERE.parent
REPO = HERE.parents[4]
DRIVERS = HERE / "drivers"

PINNED_REVISION = "5443602d89ed99aede2e4b7bf329daddeadb320e"
APPROVED_UUID = "GPU-0e63baea-0bb1-e90c-f19d-8aa5f2995894"  # Blackwell GPU 1 (timing_fresh_b.py EXPECTED_UUID)
GPU_INDEX = 1
# Substring list for the HARNESS_ALLOW_PASSIVE_DISPLAY_CONTEXT opt-in (default off): must equal
# energy_harness/measurement_runner.py ADA_BENIGN_COMPUTE_APP_SUBSTRINGS (checked by test); anything else refuses.
_BENIGN_DISPLAY_PROCS = ("snapd-desktop-integration",)
PLATFORM = "blackwell"
ARCH = "sm_120"
L2_BYTES = 134217728  # cells.py asserts this value; the stub hardware dict is all it needs
DEFAULT_SAMPLES_ROOT = "~/cudasamples_pinned_5443602d"
DEFAULT_WORKDIR = "~/unseen_gpu_work"
TIMING_SCHEMA = "unseen_timing_raw/1"
MANIFEST_SCHEMA = "unseen_gpu_build/1"
WINDOWS_HEADER = ["block", "launches", "host_begin_monotonic_ns", "host_end_monotonic_ns", "cuda_seconds"]
TARGET_GRAPH_S = 5e-3
TARGET_WINDOW_S = 30e-3
MAX_GRAPH_BATCH = 2048
PROBE_BATCH = 16

# family -> driver source, sample directory (relative to the samples root), the pinned file(s) the driver includes
FAMILIES = {
    "matmul": dict(driver="driver_matmul.cu", sample_dir="cpp/0_Introduction/matrixMul",
                   sample_file="matrixMul.cu", extra=[]),
    "bs": dict(driver="driver_bs.cu", sample_dir="cpp/5_Domain_Specific/BlackScholes",
               sample_file="BlackScholes_kernel.cuh", extra=[]),
    "scan": dict(driver="driver_scan.cu", sample_dir="cpp/2_Concepts_and_Techniques/scan",
                 sample_file="scan.cu", extra=["scan_common.h"]),
    "conv": dict(driver="driver_conv.cu", sample_dir="cpp/2_Concepts_and_Techniques/convolutionSeparable",
                 sample_file="convolutionSeparable.cu", extra=["convolutionSeparable_common.h"]),
}
COMMON_HEADERS = ["helper_cuda.h", "helper_string.h", "helper_functions.h", "helper_image.h", "helper_timer.h",
                  "exception.h"]
LEGACY_NVCC = ["/usr/local/cuda-13.2/bin/nvcc", "/usr/local/cuda-12.8/bin/nvcc", "/usr/local/cuda/bin/nvcc"]

# Indirection so the unit tests can run the guards without a GPU or nvidia-smi.
SMI_RUN = subprocess.run


# ----------------------------------------------------------------------------------------------------------------
# cells and argv
# ----------------------------------------------------------------------------------------------------------------
def load_cells():
    if str(UNSEEN) not in sys.path:
        sys.path.insert(0, str(UNSEEN))
    import cells as cells_mod  # noqa: E402  (unseen_kernels/cells.py, read-only import)
    return cells_mod.define_cells({"l2_bytes": L2_BYTES})


def select_cells(cells, only: str):
    if only:
        cells = [c for c in cells if only in c["cell_id"]]
    if not cells:
        raise SystemExit("no cells selected")
    return cells


def safe_id(cell_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", cell_id)


def positional_args(cell) -> list[str]:
    """The two driver arguments before out.bin, derived from the cell's kernel list (cells.py)."""
    fam, ks = cell["family"], cell["kernels"]
    if fam == "matmul":
        k = ks[0]
        assert k["args"]["wA"] == k["args"]["wB"], "driver supports square matrices only"
        return [str(k["args"]["wA"]), str(k["block"][0])]
    if fam == "bs":
        k = ks[0]
        return [str(k["args"]["optN"]), str(k["block"][0])]
    if fam == "scan":
        bottom, top = ks[0], ks[1]
        return [str(bottom["grid"][0] * 1024), str(top["args"]["arrayLength"] * 1024)]
    if fam == "conv":
        k = ks[0]
        return [str(k["args"]["imageW"]), str(k["args"]["imageH"])]
    raise ValueError("unknown family %r" % fam)


def launch_geometry(family: str, pos: list[str]) -> list[dict]:
    """Kernel grid/block of ONE operator call as the driver launches it, derived from the driver arguments."""
    a, b = int(pos[0]), int(pos[1])
    if family == "matmul":  # a = N, b = tile
        return [dict(grid=[a // b, a // b, 1], block=[b, b, 1])]
    if family == "bs":  # a = optN, b = block
        return [dict(grid=[a // 2 // b, 1, 1], block=[b, 1, 1])]
    if family == "scan":  # a = N, b = arrayLength; THREADBLOCK_SIZE 256
        nb = a // (4 * 256)
        return [dict(grid=[nb, 1, 1], block=[256, 1, 1]),
                dict(grid=[-(-nb // 256), 1, 1], block=[256, 1, 1]),
                dict(grid=[nb, 1, 1], block=[256, 1, 1])]
    if family == "conv":  # a = W, b = H
        return [dict(grid=[a // 128, b // 4, 1], block=[16, 4, 1]),
                dict(grid=[a // 16, b // 64, 1], block=[16, 8, 1])]
    raise ValueError("unknown family %r" % family)


def cell_argv_check(cell) -> list[str]:
    """Positional args of the cell; fails loudly if their derived launch geometry differs from cells.py."""
    pos = positional_args(cell)
    got = launch_geometry(cell["family"], pos)
    want = [dict(grid=list(k["grid"]), block=list(k["block"])) for k in cell["kernels"]]
    if got != want:
        raise SystemExit("INTERNAL: argv geometry for %s (%s) differs from cells.py (%s)" % (cell["cell_id"], got, want))
    return pos


def cell_out_dir(workdir: Path, cell) -> Path:
    return workdir / "runs" / safe_id(cell["cell_id"])


def trailer(repeat, graph_batch, trace_dir) -> list[str]:
    return ["--repeat", str(repeat), "--graph-batch", str(graph_batch), "--trace-dir", str(trace_dir)]


# ----------------------------------------------------------------------------------------------------------------
# batch / repeat selection and result parsing
# ----------------------------------------------------------------------------------------------------------------
def choose_graph_batch(t_call_s: float, target_graph_s: float = TARGET_GRAPH_S, max_batch: int = MAX_GRAPH_BATCH) -> int:
    """Smallest operator-call count whose graph lasts >= target_graph_s, capped (graph size)."""
    if not (t_call_s > 0 and math.isfinite(t_call_s)):
        raise ValueError("non-positive per-call time %r" % (t_call_s,))
    return int(min(max_batch, max(1, math.ceil(target_graph_s / t_call_s))))


def choose_repeat(t_call_s: float, graph_batch: int, target_window_s: float = TARGET_WINDOW_S) -> int:
    """Operator calls in the timed window: a whole number of graph replays lasting >= target_window_s."""
    if not (t_call_s > 0 and math.isfinite(t_call_s)) or graph_batch < 1:
        raise ValueError("bad per-call time / graph batch")
    replays = max(1, math.ceil(target_window_s / (graph_batch * t_call_s)))
    return replays * graph_batch


def parse_windows_csv(path: Path) -> dict:
    """Read the single-row windows.csv a driver writes (format of copy_runner.py's driver)."""
    try:
        with Path(path).open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = [r for r in reader if r]
    except OSError as exc:
        raise ValueError("cannot read windows.csv: %s" % exc)
    if header != WINDOWS_HEADER:
        raise ValueError("windows.csv header %r != %r" % (header, WINDOWS_HEADER))
    if len(rows) != 1:
        raise ValueError("windows.csv must have exactly one data row, found %d" % len(rows))
    r = rows[0]
    if len(r) != len(WINDOWS_HEADER):
        raise ValueError("windows.csv row has %d fields" % len(r))
    try:
        out = dict(block=int(r[0]), launches=int(r[1]), host_begin_monotonic_ns=int(r[2]),
                   host_end_monotonic_ns=int(r[3]), cuda_seconds=float(r[4]))
    except ValueError as exc:
        raise ValueError("windows.csv row not numeric: %s" % exc)
    if out["launches"] <= 0 or not (out["cuda_seconds"] > 0 and math.isfinite(out["cuda_seconds"])):
        raise ValueError("windows.csv has non-positive launches or cuda_seconds: %r" % (out,))
    out["per_launch_s"] = out["cuda_seconds"] / out["launches"]
    return out


def read_check(out_path: Path) -> tuple[bool, str]:
    """Verdict line the driver wrote to <out>.check; a missing file counts as a failed check."""
    p = Path(str(out_path) + ".check")
    try:
        line = p.read_text().strip()
    except OSError:
        return False, "CHECK_MISSING"
    return line.startswith("CHECK_OK"), line


# ----------------------------------------------------------------------------------------------------------------
# guards (GPU-1-only; nothing here ever sets a clock, persistence mode or power limit)
# ----------------------------------------------------------------------------------------------------------------
def norm_uuid(s: str) -> str:
    return re.sub(r"[^0-9a-f]", "", str(s).lower().replace("gpu-", ""))


def _smi(args: list[str]):
    try:
        return SMI_RUN(["nvidia-smi"] + args, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SystemExit("REFUSED: cannot run nvidia-smi (%s); cannot verify the approved GPU is the visible one and idle" % exc)


def require_booking(args) -> None:
    if not getattr(args, "i_have_a_booking", False):
        raise SystemExit("REFUSED: pass --i-have-a-booking to confirm a the booking log booking of Blackwell GPU 1 exists "
                         "(check the booking log first; this tool never books for you)")


def require_approved_device(mode: str, env=None) -> dict:
    """Refuse unless exactly the approved GPU 1 is the single visible device and it is idle. mode in {time, energy}.

    energy additionally needs CUDA_VISIBLE_DEVICES == "1" because energy_harness/measurement_runner.assert_env_authorized
    requires exactly that string."""
    env = os.environ if env is None else env
    cvd = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd or "," in cvd:
        raise SystemExit("REFUSED: CUDA_VISIBLE_DEVICES must name exactly one device (got %r)" % cvd)
    if mode == "energy" and cvd != str(GPU_INDEX):
        raise SystemExit("REFUSED: energy needs CUDA_VISIBLE_DEVICES=%d exactly (energy_harness/measurement_runner contract), got %r"
                         % (GPU_INDEX, cvd))
    out = _smi(["--query-gpu=index,uuid", "--format=csv,noheader"])
    if out.returncode != 0:
        raise SystemExit("REFUSED: nvidia-smi GPU query failed (rc=%d)" % out.returncode)
    rows = []
    for ln in out.stdout.splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) >= 2:
            rows.append((parts[0], parts[1]))
    approved = [idx for idx, u in rows if norm_uuid(u) == norm_uuid(APPROVED_UUID)]
    if not approved:
        raise SystemExit("REFUSED: approved GPU %s is not present according to nvidia-smi" % APPROVED_UUID)
    if cvd.isdigit():
        visible_uuid = next((u for idx, u in rows if idx == cvd), None)
    else:
        visible_uuid = cvd if any(norm_uuid(u) == norm_uuid(cvd) for _, u in rows) else None
    if visible_uuid is None or norm_uuid(visible_uuid) != norm_uuid(APPROVED_UUID):
        raise SystemExit("REFUSED: CUDA_VISIBLE_DEVICES=%r is not the approved Blackwell GPU 1 (%s)" % (cvd, APPROVED_UUID))
    if mode == "energy" and approved[0] != str(GPU_INDEX):
        raise SystemExit("REFUSED: approved GPU has nvidia-smi index %s, not %d" % (approved[0], GPU_INDEX))
    busy_q = _smi(["--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"])
    if busy_q.returncode != 0:
        raise SystemExit("REFUSED: nvidia-smi compute-apps query failed (rc=%d); cannot show the GPU is idle" % busy_q.returncode)
    busy = [ln for ln in busy_q.stdout.splitlines() if ln.strip() and norm_uuid(ln.split(",")[0]) == norm_uuid(APPROVED_UUID)]
    if busy and os.environ.get("HARNESS_ALLOW_PASSIVE_DISPLAY_CONTEXT") == "1":
        # Opt-in (default off): tolerate a passive desktop context (same named list as energy_harness/measurement_runner.py
        # ADA_BENIGN_COMPUTE_APP_SUBSTRINGS, M376). Anything else refuses exactly as without the opt-in.
        busy = [ln for ln in busy if not any(s in ln for s in _BENIGN_DISPLAY_PROCS)]
    if busy:
        raise SystemExit("REFUSED: the approved GPU has compute processes (check the booking log; never kill a process this "
                         "project did not start):\n" + "\n".join(busy))
    return dict(cuda_visible_devices=cvd, uuid=APPROVED_UUID, nvidia_smi_index=approved[0])


def apply_process_env() -> dict:
    """Process-local environment shared by every subprocess (driver and harness): index order agrees with
    nvidia-smi/NVML, and the driver verifies the device UUID itself."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["UNSEEN_EXPECT_UUID"] = APPROVED_UUID
    return dict(os.environ)


# ----------------------------------------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_nvcc() -> str:
    """HARNESS_NVCC, else the CUDA 13.2 nvcc that produced the analysed cubins, else the repo's usual search."""
    explicit = os.environ.get("HARNESS_NVCC", "").strip()
    if explicit:
        if not Path(explicit).is_file():
            raise SystemExit("BUILD_BLOCKED: HARNESS_NVCC=%s does not exist" % explicit)
        return explicit
    for cand in LEGACY_NVCC:
        if Path(cand).is_file():
            return cand
    found = shutil.which("nvcc")
    if found:
        return found
    raise SystemExit("BUILD_BLOCKED: no nvcc found (set HARNESS_NVCC; the analysed cubins used CUDA 13.2.78)")


def sample_files(root: Path, family: str) -> list[Path]:
    fam = FAMILIES[family]
    base = Path(root) / fam["sample_dir"]
    return [base / fam["sample_file"]] + [base / e for e in fam["extra"]]


def build_command(nvcc: str, root: Path, family: str, binary: Path, host_opt: bool = True) -> list[str]:
    """nvcc default optimisation (no -O flag), as the analysed cubins were built. -Xcompiler -O2 optimises only the
    HOST code (the CPU reference checks of the drivers); it does not change the device code."""
    fam = FAMILIES[family]
    cmd = [nvcc, "-arch=%s" % ARCH]
    if host_opt:
        cmd += ["-Xcompiler", "-O2"]
    cmd += ["-I", str(Path(root) / "Common"), "-I", str(Path(root) / fam["sample_dir"]), "-I", str(DRIVERS),
            str(DRIVERS / fam["driver"]), "-o", str(binary)]
    return cmd


def verify_samples_root(root: Path) -> str:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit("BUILD_BLOCKED: samples root %s does not exist" % root)
    try:
        rev = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        raise SystemExit("BUILD_BLOCKED: %s is not a readable git checkout" % root)
    if rev != PINNED_REVISION:
        raise SystemExit("BUILD_BLOCKED: expected cuda-samples revision %s, got %s" % (PINNED_REVISION, rev))
    for family in FAMILIES:
        for f in sample_files(root, family):
            if not f.is_file():
                raise SystemExit("BUILD_BLOCKED: missing pinned sample file %s" % f)
            rel = str(f.relative_to(root))
            st = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", rel], capture_output=True, text=True)
            if st.returncode != 0 or st.stdout.strip():
                raise SystemExit("BUILD_BLOCKED: pinned file %s is modified or git status failed: %r" % (rel, st.stdout.strip()))
    return rev


def cmd_build(args) -> int:
    root = Path(args.samples_root).expanduser().resolve()
    revision = verify_samples_root(root)
    nvcc = find_nvcc()
    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
    if ver.returncode != 0:
        raise SystemExit("BUILD_BLOCKED: %s --version failed" % nvcc)
    workdir = Path(args.workdir).expanduser().resolve()
    (workdir / "bin").mkdir(parents=True, exist_ok=True)
    (workdir / "build").mkdir(parents=True, exist_ok=True)
    manifest = dict(schema=MANIFEST_SCHEMA, built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    samples_root=str(root), samples_revision=revision, nvcc=dict(path=nvcc, version=ver.stdout.strip()),
                    arch=ARCH, host_opt=bool(args.host_opt),
                    driver_common_h_sha256=sha256_file(DRIVERS / "driver_common.h"),
                    common_headers={h: (sha256_file(root / "Common" / h) if (root / "Common" / h).is_file() else None)
                                    for h in COMMON_HEADERS},
                    families={})
    failed = False
    for family in FAMILIES:
        fam = FAMILIES[family]
        binary = workdir / "bin" / ("unseen_%s" % family)
        if binary.exists():
            binary.unlink()
        cmd = build_command(nvcc, root, family, binary, args.host_opt)
        print("[build] %s: %s" % (family, " ".join(cmd)), flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        (workdir / "build" / ("%s.log" % family)).write_text("$ %s\n%s\n%s" % (" ".join(cmd), proc.stdout, proc.stderr))
        entry = dict(driver=str(DRIVERS / fam["driver"]), driver_sha256=sha256_file(DRIVERS / fam["driver"]),
                     sample_files={str(f.relative_to(root)): sha256_file(f) for f in sample_files(root, family)},
                     command=cmd, returncode=proc.returncode)
        if proc.returncode != 0 or not binary.is_file():
            failed = True
            entry["error"] = proc.stderr[-2000:]
            print("[build] %s FAILED (see %s)" % (family, workdir / "build" / ("%s.log" % family)), file=sys.stderr)
        else:
            entry.update(binary=str(binary), binary_sha256=sha256_file(binary))
        manifest["families"][family] = entry
    (workdir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("wrote", workdir / "manifest.json")
    return 3 if failed else 0


def load_manifest(workdir: Path) -> dict:
    p = Path(workdir) / "manifest.json"
    if not p.is_file():
        raise SystemExit("no %s; run the build subcommand first" % p)
    m = json.loads(p.read_text())
    if m.get("schema") != MANIFEST_SCHEMA:
        raise SystemExit("manifest schema %r is not %s" % (m.get("schema"), MANIFEST_SCHEMA))
    for fam, e in m["families"].items():
        b = e.get("binary")
        if not b or not Path(b).is_file():
            raise SystemExit("family %s has no built binary in the manifest (build failed?)" % fam)
        if sha256_file(Path(b)) != e["binary_sha256"]:
            raise SystemExit("binary of %s changed since the build (SHA-256 mismatch); rebuild" % fam)
    return m


# ----------------------------------------------------------------------------------------------------------------
# running a driver
# ----------------------------------------------------------------------------------------------------------------
def run_driver(binary: str, pos: list[str], out_path: Path, repeat: int, graph_batch: int, trace_dir: Path,
               env: dict, timeout_s: float) -> dict:
    """One driver invocation with the trailer. Returns dict(rc, stdout, stderr, windows, check_ok, check_line)."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    argv = [binary] + pos + [str(out_path)] + trailer(repeat, graph_batch, trace_dir)
    res = dict(argv=argv, rc=None, stdout="", stderr="", windows=None, check_ok=False, check_line="CHECK_MISSING", error=None)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        res["error"] = "timeout after %.0f s" % timeout_s
        return res
    except OSError as exc:
        res["error"] = "cannot start driver: %s" % exc
        return res
    res.update(rc=proc.returncode, stdout=proc.stdout[-2000:], stderr=proc.stderr[-2000:])
    res["check_ok"], res["check_line"] = read_check(out_path)
    if proc.returncode in (0, 1):  # 1 = failed check; windows.csv is written before the check
        try:
            res["windows"] = parse_windows_csv(trace_dir / "windows.csv")
        except ValueError as exc:
            res["error"] = str(exc)
    else:
        res["error"] = "driver exit code %d: %s" % (proc.returncode, proc.stderr.strip()[-500:])
    return res


def time_one_cell(cell, manifest, workdir: Path, env: dict, windows: int, timeout_s: float) -> dict:
    pos = cell_argv_check(cell)
    fam = cell["family"]
    binary = manifest["families"][fam]["binary"]
    d = cell_out_dir(workdir, cell)
    d.mkdir(parents=True, exist_ok=True)
    out_path = d / "out.bin"
    rec = dict(cell_id=cell["cell_id"], family=fam, operator_id=cell["operator_id"], regime=cell["regime"],
               candidate_id=cell["candidate_id"], tier=cell["tier"], argv=pos, probes=[], windows=[],
               per_launch_runtime_s=None, launches=None, repeat=None, graph_batch=None, check="CHECK_NOT_RUN", correct=False)
    # fixed two-step calibration (not a retry): 16-call graph -> batch estimate -> one replay of that batch
    p1 = run_driver(binary, pos, out_path, PROBE_BATCH, PROBE_BATCH, d / "probe1", env, timeout_s)
    rec["probes"].append(dict(repeat=PROBE_BATCH, graph_batch=PROBE_BATCH, rc=p1["rc"], error=p1["error"],
                              per_launch_s=(p1["windows"] or {}).get("per_launch_s")))
    if p1["windows"] is None or p1["rc"] != 0:
        rec["error"] = "probe1 failed: %s" % (p1["error"] or p1["check_line"] or p1["stderr"][-300:])
        rec["check"] = p1["check_line"]
        return rec
    b1 = choose_graph_batch(p1["windows"]["per_launch_s"])
    p2 = run_driver(binary, pos, out_path, b1, b1, d / "probe2", env, timeout_s)
    rec["probes"].append(dict(repeat=b1, graph_batch=b1, rc=p2["rc"], error=p2["error"],
                              per_launch_s=(p2["windows"] or {}).get("per_launch_s")))
    if p2["windows"] is None or p2["rc"] != 0:
        rec["error"] = "probe2 failed: %s" % (p2["error"] or p2["check_line"] or p2["stderr"][-300:])
        rec["check"] = p2["check_line"]
        return rec
    t2 = p2["windows"]["per_launch_s"]
    batch = choose_graph_batch(t2)
    repeat = choose_repeat(t2, batch)
    rec.update(graph_batch=batch, repeat=repeat)
    all_ok = True
    for w in range(windows):
        r = run_driver(binary, pos, out_path, repeat, batch, d / ("window%d" % (w + 1)), env, timeout_s)
        entry = dict(rc=r["rc"], error=r["error"], check=r["check_line"], check_ok=r["check_ok"])
        if r["windows"] is not None:
            entry.update(launches=r["windows"]["launches"], cuda_seconds=r["windows"]["cuda_seconds"],
                         per_launch_s=r["windows"]["per_launch_s"],
                         host_begin_monotonic_ns=r["windows"]["host_begin_monotonic_ns"],
                         host_end_monotonic_ns=r["windows"]["host_end_monotonic_ns"])
        rec["windows"].append(entry)
        rec["check"] = r["check_line"]
        all_ok &= bool(r["check_ok"]) and r["rc"] == 0 and r["windows"] is not None
    good = [w for w in rec["windows"] if "per_launch_s" in w]
    if good:
        rec["per_launch_runtime_s"] = statistics.median(w["per_launch_s"] for w in good)
        rec["per_launch_min_s"] = min(w["per_launch_s"] for w in good)
        rec["per_launch_max_s"] = max(w["per_launch_s"] for w in good)
        rec["launches"] = good[-1]["launches"]
    rec["correct"] = bool(all_ok and good)
    return rec


def cmd_time(args) -> int:
    cells = select_cells(load_cells(), args.only)
    workdir = Path(args.workdir).expanduser().resolve()
    if args.dry_run:
        return dry_run(cells, workdir, "time")
    require_booking(args)
    info = require_approved_device("time")
    env = apply_process_env()
    manifest = load_manifest(workdir)
    out_path = Path(args.out).expanduser() if args.out else workdir / "timing_unseen_result.json"
    partial = Path(str(out_path) + ".partial")
    if out_path.exists() or partial.exists():
        raise SystemExit("REFUSED: %s or its .partial exists; choose a new --out" % out_path)
    result = dict(schema=TIMING_SCHEMA, booking_ref=args.booking_ref, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  device=info, cell_definitions="tiresias/framework/predictor/unseen_kernels/cells.py",
                  cells_py_sha256=sha256_file(UNSEEN / "cells.py"), manifest_binaries={f: e["binary_sha256"] for f, e in manifest["families"].items()},
                  nvcc=manifest["nvcc"]["version"].splitlines()[-1] if manifest["nvcc"]["version"] else "",
                  method=dict(probe_batch=PROBE_BATCH, target_graph_s=TARGET_GRAPH_S, target_window_s=TARGET_WINDOW_S,
                              max_graph_batch=MAX_GRAPH_BATCH, windows=args.windows,
                              per_launch_runtime_s="median over windows of cuda_seconds / launches (cudaEvents around repeat operator calls)",
                              clocks_set=False, persistence_set=False, power_limit_set=False),
                  cells=[])
    t0 = time.time()
    failed = False
    for i, cell in enumerate(cells):
        print("[%7.1fs] cell %d/%d %s" % (time.time() - t0, i + 1, len(cells), cell["cell_id"]), flush=True)
        rec = time_one_cell(cell, manifest, workdir, env, args.windows, args.run_timeout_s)
        result["cells"].append(rec)
        failed |= not rec["correct"]
        pl = rec["per_launch_runtime_s"]
        print("          %s per_launch=%s repeat=%s graph_batch=%s check=%s" % (
            "OK" if rec["correct"] else "FAILED", "%.3f us" % (pl * 1e6) if pl else "n/a", rec["repeat"], rec["graph_batch"], rec["check"][:80]), flush=True)
        partial.write_text(json.dumps(result, indent=1) + "\n")
    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result["all_correct"] = not failed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", out_path)
    if failed:
        print("at least one cell failed its correctness check or its run; do not use this file as is", file=sys.stderr)
        return 2
    return 0


# ----------------------------------------------------------------------------------------------------------------
# energy
# ----------------------------------------------------------------------------------------------------------------
def energy_graph_batch(cell, manifest, workdir, env, args, timing_by_cell) -> tuple[int | None, str]:
    """Graph batch for the energy harness: fixed (--graph-batch), from a timing JSON, or from the same two-step
    probe the time subcommand uses. Returns (batch, note); batch None means the probe failed."""
    if args.graph_batch:
        return int(args.graph_batch), "fixed by --graph-batch"
    t = timing_by_cell.get(cell["cell_id"])
    if t:
        return choose_graph_batch(t), "from --timing-json"
    pos = cell_argv_check(cell)
    d = cell_out_dir(workdir, cell) / "energy_probe"
    d.mkdir(parents=True, exist_ok=True)
    binary = manifest["families"][cell["family"]]["binary"]
    out_path = d / "out.bin"
    p1 = run_driver(binary, pos, out_path, PROBE_BATCH, PROBE_BATCH, d / "probe1", env, args.run_timeout_s)
    if p1["windows"] is None or p1["rc"] != 0:
        return None, "probe1 failed: %s" % (p1["error"] or p1["check_line"] or p1["stderr"][-300:])
    b1 = choose_graph_batch(p1["windows"]["per_launch_s"])
    p2 = run_driver(binary, pos, out_path, b1, b1, d / "probe2", env, args.run_timeout_s)
    if p2["windows"] is None or p2["rc"] != 0:
        return None, "probe2 failed: %s" % (p2["error"] or p2["check_line"] or p2["stderr"][-300:])
    return choose_graph_batch(p2["windows"]["per_launch_s"]), "from probe"


def make_energy_context(cell, manifest, workdir: Path, BinaryEnergyContext):
    fam = cell["family"]
    pos = cell_argv_check(cell)
    entry = manifest["families"][fam]
    d = cell_out_dir(workdir, cell) / "energy"
    d.mkdir(parents=True, exist_ok=True)
    out_path = d / "out.bin"
    main_rel, main_sha = next(iter(entry["sample_files"].items()))
    controls = dict(cell["controls"], family=fam, tier=cell["tier"], argv_positional=pos)
    return BinaryEnergyContext(
        parent_id=cell["operator_id"], regime=cell["regime"], candidate_id=cell["candidate_id"],
        source_revision=manifest["samples_revision"], source_sha256=main_sha, source_path=main_rel,
        controls=controls,
        runtime_info=dict(nvcc=manifest["nvcc"]["version"].splitlines()[-1] if manifest["nvcc"]["version"] else "",
                          arch=ARCH, driver=Path(entry["driver"]).name, driver_sha256=entry["driver_sha256"],
                          binary_sha256=entry["binary_sha256"], host_opt=manifest["host_opt"]),
        binary=Path(entry["binary"]), argv_prefix=pos + [str(out_path)],
        check=lambda p=out_path: read_check(p)[0])


def cmd_energy(args) -> int:
    cells = select_cells(load_cells(), args.only)
    workdir = Path(args.workdir).expanduser().resolve()
    if args.dry_run:
        return dry_run(cells, workdir, "energy", args)
    require_booking(args)
    require_approved_device("energy")
    env = apply_process_env()
    manifest = load_manifest(workdir)
    timing_by_cell = {}
    if args.timing_json:
        doc = json.loads(Path(args.timing_json).read_text())
        timing_by_cell = {c["cell_id"]: c["per_launch_runtime_s"] for c in doc["cells"] if c.get("per_launch_runtime_s")}
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else workdir / "energy_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Imported only now: the harness pulls in measurement_runner and nvml_sampler (Linux/NVML host).
    if str(REPO / "energy_harness") not in sys.path:
        sys.path.insert(0, str(REPO / "energy_harness"))
    import application_energy_harness as H  # noqa: E402
    log = out_dir / "unseen_energy_run_log.jsonl"
    any_rejected = False
    for i, cell in enumerate(cells):
        run_id = "%s-%s-%s-%s" % (args.run_id_prefix, cell["family"], cell["regime"], cell["candidate_id"])  # unique per cell
        runner_name = "unseen_%s" % cell["family"]
        print("cell %d/%d %s run_id=%s" % (i + 1, len(cells), cell["cell_id"], run_id), flush=True)
        batch, note = energy_graph_batch(cell, manifest, workdir, env, args, timing_by_cell)
        if batch is None:
            result = dict(status="rejected", row=dict(
                timestamp_utc=H.iso_now(), run_id=run_id, session=args.session, runner=runner_name,
                parent_id=cell["operator_id"], regime=cell["regime"], candidate_id=cell["candidate_id"],
                window_target_seconds=args.window_target_seconds, rejection_reason="graph_batch_probe_failed",
                raw_stderr_tail=note[-1000:]))
            H.append_raw_or_rejected(out_dir, result)
            _log(log, cell, run_id, result, None, note)
            any_rejected = True
            continue
        try:
            # preflight_before_context must precede any context that could hold GPU memory (here none does, but the
            # order is kept exactly as energy_harness/run_application_energy.py has it)
            gpu_uuid = H.preflight_before_context(GPU_INDEX, PLATFORM)
            if norm_uuid(gpu_uuid) != norm_uuid(APPROVED_UUID):
                raise H.RunnerError("uuid_mismatch: nvidia-smi index %d reports %s, not the approved %s"
                                    % (GPU_INDEX, gpu_uuid, APPROVED_UUID))
            ctx = make_energy_context(cell, manifest, workdir, H.BinaryEnergyContext)
            result = H.run_binary_energy(ctx, window_target_seconds=args.window_target_seconds, session=args.session,
                                         run_id=run_id, out_dir=out_dir, gpu_index=GPU_INDEX, platform=PLATFORM,
                                         runner_name=runner_name, gpu_uuid=gpu_uuid, graph_batch=batch)
        except H.RunnerError as exc:
            # Gate failures (GPU shared, cooldown timeout, wrong env) can mean another user appeared: record and stop.
            result = dict(status="rejected", row=dict(
                timestamp_utc=H.iso_now(), run_id=run_id, session=args.session, runner=runner_name,
                parent_id=cell["operator_id"], regime=cell["regime"], candidate_id=cell["candidate_id"],
                window_target_seconds=args.window_target_seconds, rejection_reason="harness_gate: %s" % str(exc)[:200],
                raw_stderr_tail=str(exc)[-1000:]))
            H.append_raw_or_rejected(out_dir, result)
            _log(log, cell, run_id, result, batch, note)
            print("STOP: harness gate failed for %s: %s" % (cell["cell_id"], exc), file=sys.stderr)
            return 3
        H.append_raw_or_rejected(out_dir, result)
        _log(log, cell, run_id, result, batch, note)
        print("  %s %s" % (result["status"].upper(), result["row"].get("rejection_reason", "")), flush=True)
        any_rejected |= result["status"] != "raw"
    return 3 if any_rejected else 0


def _log(path: Path, cell, run_id, result, batch, note) -> None:
    row = result["row"]
    entry = dict(cell_id=cell["cell_id"], run_id=run_id, status=result["status"], graph_batch=batch, graph_batch_note=note,
                 rejection_reason=row.get("rejection_reason"), launch_count=row.get("launch_count"),
                 board_energy_j_per_launch=row.get("board_energy_j_per_launch"), trace_dir=row.get("trace_dir"),
                 logged_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ----------------------------------------------------------------------------------------------------------------
# list / dry-run
# ----------------------------------------------------------------------------------------------------------------
def dry_run(cells, workdir: Path, mode: str, args=None) -> int:
    """Print what would run. Touches no GPU, runs no subprocess, needs neither a build nor a booking."""
    print("%d cells (%s dry run; nothing is executed)" % (len(cells), mode))
    for cell in cells:
        pos = cell_argv_check(cell)
        binary = workdir / "bin" / ("unseen_%s" % cell["family"])
        out_path = cell_out_dir(workdir, cell) / "out.bin"
        print("%-62s %-4s %s" % (cell["cell_id"], cell["tier"], " ".join([str(binary)] + pos + [str(out_path)])))
        geo = launch_geometry(cell["family"], pos)
        print("    kernels per operator call: " + "; ".join("%s grid=%s block=%s" % (k["kid"], g["grid"], g["block"])
                                                           for k, g in zip(cell["kernels"], geo)))
        if mode == "time":
            print("    steps: probe(16 calls) -> probe(1 graph) -> N windows of `... --repeat R --graph-batch B --trace-dir DIR`; "
                  "graph >= %.0f ms, window >= %.0f ms" % (TARGET_GRAPH_S * 1e3, TARGET_WINDOW_S * 1e3))
        elif mode == "energy":
            wt = getattr(args, "window_target_seconds", None)
            print("    steps: graph-batch probe -> preflight_before_context -> run_binary_energy(window_target_seconds=%s, "
                  "graph_batch=<auto>) -> append_raw_or_rejected" % wt)
    return 0


def cmd_list(args) -> int:
    cells = select_cells(load_cells(), args.only)
    return dry_run(cells, Path(args.workdir).expanduser().resolve(), "list")


# ----------------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--samples-root", default=DEFAULT_SAMPLES_ROOT, help="pinned cuda-samples checkout (revision %s)" % PINNED_REVISION[:8])
    common.add_argument("--workdir", default=DEFAULT_WORKDIR, help="binaries, manifest and per-cell run directories")
    common.add_argument("--only", default="", help="substring filter on cell ids (default: all 32 cells)")
    gpu = argparse.ArgumentParser(add_help=False)
    gpu.add_argument("--dry-run", action="store_true", help="print the argv of every selected cell; run nothing, needs no booking")
    gpu.add_argument("--i-have-a-booking", action="store_true", help="REQUIRED to run: confirms a the booking log booking of Blackwell GPU 1")
    gpu.add_argument("--booking-ref", default="", help="the booking log entry of the booking (recorded in the result)")
    gpu.add_argument("--run-timeout-s", type=float, default=1800.0, help="kill one driver invocation after this many seconds")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", parents=[common], help="compile the four drivers (CPU only)")
    b.add_argument("--no-host-opt", dest="host_opt", action="store_false",
                   help="drop -Xcompiler -O2 (host code only) so the nvcc command line has no optimisation flag at all")
    t = sub.add_parser("time", parents=[common, gpu], help="runtime of every cell")
    t.add_argument("--out", default="", help="result JSON (default <workdir>/timing_unseen_result.json)")
    t.add_argument("--windows", type=int, default=3, help="timed windows (separate driver runs) per cell; median reported")
    e = sub.add_parser("energy", parents=[common, gpu], help="energy of every cell through the application energy harness")
    e.add_argument("--window-target-seconds", type=float, default=15.0)
    e.add_argument("--session", type=int, default=1)
    e.add_argument("--run-id-prefix", default="UNSEEN-ENERGY")
    e.add_argument("--out-dir", default="", help="harness output directory (default <workdir>/energy_raw)")
    e.add_argument("--graph-batch", type=int, default=0, help="fixed graph batch for every cell (default: derived per cell)")
    e.add_argument("--timing-json", default="", help="result of the time subcommand: derive each cell's graph batch from it")
    sub.add_parser("list", parents=[common], help="print the argv of every cell; run nothing")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "time" and args.windows < 1:
        raise SystemExit("--windows must be at least 1")
    return {"build": cmd_build, "time": cmd_time, "energy": cmd_energy, "list": cmd_list}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
