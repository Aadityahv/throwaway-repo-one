#!/usr/bin/env python3
"""Generic energy-measurement harness for Task C's Python-callable application
runners (tiresias/app_runners/*_runner.py's Triton/PyTorch/xFormers family).

Reuses Task B's already-validated protocol instead of reinventing it: the same
cooldown gate, 90s +/-10% precondition, counted-window +/-10% duration gate, and
retarget-from-observed-CUDA-event-time retry loop live in
``energy_harness/measurement_runner.py`` and are imported here unchanged. Raw traces are
written in exactly the ``samples.csv``/``windows.csv``/``session.csv`` schema
``energy_harness/verify_b_stabilization_trace.py`` already finalizes for B's fixtures, so
that finalizer runs against these traces unmodified too. See
``tiresias/planning/B_GRAPH_REPLAY_HANDOFF_2026-09-20.md`` section 6 for
why this harness needs to exist at all: the common runner only knows B's five
local fixtures, and C's runners check correctness only, with no NVML sampling
or raw trace path.

Hard rule carried over unchanged: never silently produce a plausible-but-
unmeasured energy number. Every failure path returns a rejected record with a
stated reason; nothing falls through to a fabricated value.

Scope note (2026-09-20, M-numbered per this repo's convention -- see CHANGELOG):
wired for the seven Python-callable runners only (vector_add, triton_layernorm,
softmax, pytorch_softmax, pytorch_layernorm, embedding, xformers). The five
CUDA-driver runners (cub_block, cub_device, transpose, reduction, copy) need
their generated C++ DRIVER_TEMPLATE extended with an in-process repeat-loop
mode first -- a subprocess-per-launch loop would make process-spawn overhead
dominate the measurement, the same failure mode B hit with host-enqueue
submission on its own sub-3us kernels (M296/M297) before it adopted CUDA-graph
replay. That C++ is not written here: this session has no free GPU/nvcc host to
compile-check it against the pinned CUB/CUDA-Samples/CCCL checkouts, and
shipping uncompiled measurement-path C++ on this project's own broken-tools
rule is not acceptable. See APPLICATION_ENERGY_HARNESS_RUNBOOK.md section 4 for
the exact patch spec to apply once a GPU/toolchain host is free.

No GPU/SSH/shared-machine action is taken by writing or testing this module.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_BATCH = 1000  # matches B's fixtures and the M310 diagnostic
HPC_DIR = Path(__file__).resolve().parent

if str(HPC_DIR) not in sys.path:
    sys.path.insert(0, str(HPC_DIR))
from measurement_runner import (  # noqa: E402
    RunnerError,
    RAW_TRACE_PRECONDITION_SECONDS,
    MAX_DURATION_ATTEMPTS,
    wait_for_cooldown,
    check_target_gpu_idle_and_unshared,
    assert_env_authorized,
    query_target_gpu_uuid,
    duration_within_tolerance,
    retarget_launches_from_event_time,
    iso_now,
    append_csv,
)
from nvml_sampler import NvmlSampler  # noqa: E402

APPLICATION_RAW_FIELDS = [
    "timestamp_utc", "run_id", "session", "runner", "family", "parent_id", "regime", "candidate_id",
    "cuda_visible_devices", "nvml_gpu_id", "gpu_uuid", "host",
    "source_revision", "source_sha256", "source_path",
    "window_target_seconds", "launch_count", "trace_dir",
    "precondition_cuda_seconds", "precondition_target_seconds",
    "counted_launch_interval_s", "board_energy_j_total", "board_energy_j_per_launch",
    "active_sample_count", "correctness_check", "runtime_json", "controls_json",
    "launch_model", "graph_batch",
]
APPLICATION_REJECTED_FIELDS = [
    "timestamp_utc", "run_id", "session", "runner", "parent_id", "regime", "candidate_id",
    "window_target_seconds", "rejection_reason", "raw_stderr_tail",
]


@dataclass
class EnergyContext:
    """What each ``*_runner.py``'s ``prepare_energy_context()`` must return.

    Built once per candidate (inputs allocated on device, source verified,
    kernel imported); the harness then calls ``launch_once`` many times inside
    a timed, NVML-sampled window, and ``check`` exactly once afterward against
    whatever the last launch left in the persistent output buffer -- the same
    single-shot correctness check ``run_candidate()`` already does, just moved
    to after the counted repeats instead of after a single call.
    """

    parent_id: str
    regime: str
    candidate_id: str
    source_revision: str
    source_sha256: str
    source_path: str
    controls: dict
    runtime_info: dict
    launch_once: Callable[[], None]
    sync: Callable[[], None]
    check: Callable[[], bool]
    # cuda_event_seconds(fn): runs fn() bracketed by CUDA events on the runner's
    # own stream and returns elapsed seconds. Kept as an injected callable (not
    # raw Event objects) so this module never imports torch directly -- every
    # runner already knows how to time its own launches on its own runtime.
    cuda_event_seconds: Callable[[Callable[[], None]], float]
    # capture_graph(batch): captures ``batch`` back-to-back launch_once calls into one CUDA graph
    # and returns a callable that replays it (energy_harness/torch_graph_capture.py). Optional so a context
    # without one can still run the declared host-loop cross-check; the graph model rejects it.
    capture_graph: Callable[[int], Callable[[], None]] | None = None


def calibrate_in_process(ctx: EnergyContext, target_seconds: float, unit=None) -> int:
    """Two-point host-time doubling calibration, done in-process.

    B's ``energy_harness/duration_probe.sh calibrate`` exists specifically to cancel out a
    *separate process's* startup cost by differencing two probe sizes
    (measurement_runner.py's own comment: "necessarily based on host elapsed
    time"). A Python-callable candidate has no such subprocess boundary --
    ``launch_once`` is an in-process call -- so a plain doubling probe against a
    noise floor is sufficient here; this is a deliberate, documented
    simplification, not a silent shortcut. The retarget-from-observed-CUDA-
    event-time correction below (shared with B, via
    ``retarget_launches_from_event_time``) is what actually controls final
    accuracy, exactly as it does for B's fixtures.
    """
    unit = unit or ctx.launch_once  # a graph replay when running under the graph launch model
    k = 16
    while True:
        t0 = time.perf_counter()
        for _ in range(k):
            unit()
        ctx.sync()
        dt = time.perf_counter() - t0
        if dt >= 0.2:
            rate = k / dt
            return max(1, round(rate * target_seconds))
        k *= 4
        if k > 1 << 30:
            raise RunnerError(
                "duration_did_not_converge: in-process calibration probe never cleared the noise floor"
            )


def _run_n(fn: Callable[[], None], n: int) -> None:
    for _ in range(n):
        fn()


def _write_session_csv(trace_dir: Path, precondition_cuda_seconds: float) -> None:
    with (trace_dir / "session.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["precondition_cuda_seconds"])
        writer.writeheader()
        writer.writerow({"precondition_cuda_seconds": precondition_cuda_seconds})


def _write_windows_csv(trace_dir: Path, launches: int, begin_ns: int, end_ns: int, cuda_seconds: float) -> None:
    with (trace_dir / "windows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["block", "launches", "host_begin_monotonic_ns",
                                                "host_end_monotonic_ns", "cuda_seconds"])
        writer.writeheader()
        writer.writerow({"block": 1, "launches": launches, "host_begin_monotonic_ns": begin_ns,
                          "host_end_monotonic_ns": end_ns, "cuda_seconds": cuda_seconds})


def finalize_raw_trace(trace_dir: Path) -> dict:
    """Delegates to energy_harness/verify_b_stabilization_trace.py, unmodified -- the raw
    schema this module writes is exactly what that finalizer already expects."""
    import subprocess
    proc = subprocess.run(
        [sys.executable, str(HPC_DIR / "verify_b_stabilization_trace.py"), str(trace_dir)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RunnerError(f"raw_trace_finalization_failed: {proc.stderr.strip()[-1000:]}")
    try:
        rows = list(csv.DictReader((trace_dir / "blocks.csv").open()))
    except OSError as exc:
        raise RunnerError(f"raw_trace_missing_block: {exc}")
    if len(rows) != 1:
        raise RunnerError(f"raw_trace_missing_block: expected one canonical block, got {len(rows)}")
    row = rows[0]
    return dict(
        counted_launch_interval_s=float(row["cuda_seconds"]),
        board_energy_j_total=float(row["board_joules"]),
        board_energy_j_per_launch=float(row["board_joules_per_launch"]),
        active_sample_count=int(row["samples_including_boundaries"]),
    )


def _compute_process_pids(gpu_index: int) -> set[int]:
    import subprocess
    from measurement_runner import ADA_BENIGN_COMPUTE_APP_SUBSTRINGS
    out = subprocess.run(
        ["nvidia-smi", "-i", str(gpu_index), "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RunnerError(
            f"gpu_occupancy_unknown: GPU{gpu_index} process query failed "
            f"(exit {out.returncode}): {out.stderr.strip()}")
    if not out.stdout.strip():
        return set()
    pids = set()
    for line in out.stdout.strip().splitlines():
        line = line.strip()
        # M376: Ada's permanent desktop helper is not a workload (same named whitelist as
        # measurement_runner.check_target_gpu_idle_and_unshared); any other process still counts.
        if line and not any(s in line for s in ADA_BENIGN_COMPUTE_APP_SUBSTRINGS):
            pids.add(int(line.split(",")[0]))
    return pids


def check_gpu_unshared_excluding_self(gpu_index: int) -> None:
    """Same rejection as measurement_runner.check_target_gpu_idle_and_unshared,
    but excludes this process's own PID.

    B's fixtures are separate binaries launched *after* the idle check, so the
    checking process never appears in --query-compute-apps itself. A Python-
    callable EnergyContext's tensors keep a CUDA context open in *this* process
    for the whole measurement (see prepare_energy_context() in each runner) --
    check_target_gpu_idle_and_unshared alone cannot tell that context apart
    from another user's process, so it would reject on the harness's own
    memory every time. This still catches a genuinely new process appearing
    during the run.
    """
    other_pids = _compute_process_pids(gpu_index) - {os.getpid()}
    if other_pids:
        raise RunnerError(
            f"gpu1_shared_during_run: GPU{gpu_index} has other active compute processes: {sorted(other_pids)}"
        )


def preflight_before_context(gpu_index: int, platform: str) -> str:
    """Every hardware gate that must run BEFORE any GPU memory is touched by
    this process. Callers run this, then prepare_energy_context() (which
    allocates device tensors), then run_python_callable_energy() with the
    returned gpu_uuid -- never the other way around, or check_target_gpu_
    idle_and_unshared would see this process's own soon-to-exist CUDA context
    and reject it as another user's process. Returns gpu_uuid.
    """
    assert_env_authorized(gpu_index)
    check_target_gpu_idle_and_unshared(gpu_index, platform)
    gpu_uuid = query_target_gpu_uuid(gpu_index)
    wait_for_cooldown(gpu_index)
    return gpu_uuid


def run_python_callable_energy(
    ctx: EnergyContext, *, window_target_seconds: float, session: int, run_id: str,
    out_dir: Path, gpu_index: int, platform: str, runner_name: str, gpu_uuid: str,
    dry_run: bool = False, sampler_factory=NvmlSampler,
    launch_model: str = "graph", graph_batch: int = DEFAULT_GRAPH_BATCH,
) -> dict:
    """Runs ``ctx`` through B's cooldown/precondition/window-gate protocol.

    ``gpu_uuid`` must come from ``preflight_before_context()``, called BEFORE
    ``ctx`` was built -- see that function's docstring for why the ordering
    matters. This function itself only re-checks for *new* processes via
    ``check_gpu_unshared_excluding_self`` (this process's own context is
    already expected here) and re-applies the cooldown gate before the timed
    loop, matching B's own two-cooldown convention (once before calibration,
    once before recording).

    Returns ``{"status": "raw", "row": {...}}`` or
    ``{"status": "rejected", "row": {...}}`` (never raises RunnerError itself;
    callers that want fail-loud behavior should check ``status``).

    ``launch_model="graph"`` (default, M311) captures ``graph_batch`` launches into one CUDA graph
    and replays it, so the counted window measures kernels rather than Python submission, which
    M310 measured at 53-87% of the host-loop window.  ``launch_model="host"`` is the plain Python
    loop, kept only as the declared cross-check; its rows must never be pooled with graph rows.
    A context without ``capture_graph`` is rejected under the graph model, never silently run on
    the host path.  In graph mode all calibration/retarget counts are in replays; the recorded
    ``launch_count`` is replays * ``graph_batch``.
    """
    if launch_model not in ("graph", "host"):
        raise ValueError(f"launch_model must be 'graph' or 'host', got {launch_model!r}")
    base_row = dict(run_id=run_id, session=session, runner=runner_name, parent_id=ctx.parent_id,
                     regime=ctx.regime, candidate_id=ctx.candidate_id,
                     window_target_seconds=window_target_seconds,
                     launch_model=launch_model,
                     graph_batch=graph_batch if launch_model == "graph" else 1)

    if dry_run:
        return dict(status="dry_run", row=dict(base_row, controls=ctx.controls, runtime=ctx.runtime_info))

    check_gpu_unshared_excluding_self(gpu_index)
    wait_for_cooldown(gpu_index)

    ctx.launch_once()
    ctx.sync()  # warm-up call, discarded -- matches every runner's own single already-verified launch

    if launch_model == "graph":
        if ctx.capture_graph is None:
            return dict(status="rejected", row=dict(
                timestamp_utc=iso_now(), rejection_reason="graph_capture_unavailable",
                raw_stderr_tail="context has no capture_graph; refusing to fall back to the host loop",
                **base_row))
        unit = ctx.capture_graph(graph_batch)
        per_unit = graph_batch
    else:
        unit, per_unit = ctx.launch_once, 1
    ctx.sync()

    # From here every count is in units (one graph replay, or one host launch).
    precondition = calibrate_in_process(ctx, RAW_TRACE_PRECONDITION_SECONDS, unit)
    launches = calibrate_in_process(ctx, window_target_seconds, unit)
    wait_for_cooldown(gpu_index)

    cache_pre = f"app:{launch_model}:{runner_name}:{ctx.candidate_id}:pre"
    cache_win = f"app:{launch_model}:{runner_name}:{ctx.candidate_id}:win"

    for attempt in range(1, MAX_DURATION_ATTEMPTS + 1):
        trace_dir = out_dir / "raw_attempts" / run_id / f"{runner_name}-s{session}-a{attempt}"
        trace_dir.mkdir(parents=True, exist_ok=False)
        sampler = sampler_factory(gpu_index)
        sampler.start()
        try:
            precondition_seconds = ctx.cuda_event_seconds(lambda: _run_n(unit, precondition))
            ctx.sync()
            block_begin_ns = time.monotonic_ns()
            block_seconds = ctx.cuda_event_seconds(lambda: _run_n(unit, launches))
            ctx.sync()
            block_end_ns = time.monotonic_ns()
            # Guarantee at least one sample strictly after block_end_ns: the sampler polls on a
            # fixed interval independent of when the timed block actually ends, so stopping it
            # immediately can leave the boundary timestamp outside the sampled range (verify_b_
            # stabilization_trace.py's power_at() requires both boundaries to be interpolable).
            time.sleep(max(2 * getattr(sampler, "_interval", 0.01), 0.05))
        finally:
            sampler.stop()
        sampler.write_csv(trace_dir / "samples.csv")
        _write_session_csv(trace_dir, precondition_seconds)
        _write_windows_csv(trace_dir, launches * per_unit, block_begin_ns, block_end_ns, block_seconds)

        precondition_ok = duration_within_tolerance(precondition_seconds, RAW_TRACE_PRECONDITION_SECONDS)
        window_ok = duration_within_tolerance(block_seconds, window_target_seconds)

        if precondition_ok and window_ok:
            parsed = finalize_raw_trace(trace_dir)
            if not ctx.check():
                return dict(status="rejected", row=dict(
                    timestamp_utc=iso_now(), rejection_reason="correctness_check_failed",
                    raw_stderr_tail="", **base_row,
                ))
            row = dict(
                timestamp_utc=iso_now(), family="application",
                cuda_visible_devices=str(gpu_index), nvml_gpu_id=gpu_index, gpu_uuid=gpu_uuid,
                host=os.uname().nodename if hasattr(os, "uname") else "unknown",
                source_revision=ctx.source_revision, source_sha256=ctx.source_sha256,
                source_path=ctx.source_path, launch_count=launches * per_unit, trace_dir=str(trace_dir),
                precondition_cuda_seconds=precondition_seconds,
                precondition_target_seconds=RAW_TRACE_PRECONDITION_SECONDS,
                correctness_check=True, runtime_json=json.dumps(ctx.runtime_info, sort_keys=True),
                controls_json=json.dumps(ctx.controls, sort_keys=True),
                **base_row,
            )
            row.update(parsed)
            return dict(status="raw", row=row)

        if attempt == MAX_DURATION_ATTEMPTS:
            return dict(status="rejected", row=dict(
                timestamp_utc=iso_now(), rejection_reason="duration_did_not_converge",
                raw_stderr_tail=(f"after {attempt} attempts: precondition {precondition_seconds}s "
                                  f"(target {RAW_TRACE_PRECONDITION_SECONDS}s), block {block_seconds}s "
                                  f"(target {window_target_seconds}s)"),
                **base_row,
            ))

        if not precondition_ok:
            precondition = retarget_launches_from_event_time(
                cache_pre, precondition, precondition_seconds, RAW_TRACE_PRECONDITION_SECONDS)
        if not window_ok:
            launches = retarget_launches_from_event_time(
                cache_win, launches, block_seconds, window_target_seconds)

    raise AssertionError("unreachable: loop always returns or raises within MAX_DURATION_ATTEMPTS")


@dataclass
class BinaryEnergyContext:
    """What each CUDA-driver ``*_runner.py``'s ``prepare_binary_energy_context()``
    must return -- the binary-invocation counterpart to ``EnergyContext``.

    Unlike the Python-callable family, the timed launch loop lives inside the
    compiled binary itself (native ``cudaGraph_t`` capture, per
    ``APPLICATION_ENERGY_HARNESS_RUNBOOK.md`` section 4): the binary already
    writes its own ``windows.csv`` into ``--trace-dir`` when given
    ``--repeat N --graph-batch B --trace-dir DIR``. This harness only supplies
    the NVML sampler, the cooldown/precondition/retry protocol, and the
    calibration that picks ``N``; it never re-implements the timed loop.
    """

    parent_id: str
    regime: str
    candidate_id: str
    source_revision: str
    source_sha256: str
    source_path: str
    controls: dict
    runtime_info: dict
    binary: Path
    # Required positional args the driver takes before the optional
    # --repeat/--graph-batch/--trace-dir trailer (already-built input/output
    # file paths, geometry, etc. -- exactly what a correctness-only
    # single-shot invocation would pass).
    argv_prefix: list[str]
    # Reads whatever output file the binary just wrote (same file every
    # invocation -- every repeat/replay is the identical deterministic op on
    # identical inputs) and compares it to the reference. Called once, after
    # the counted block invocation, never per-launch.
    check: Callable[[], bool]


def _binary_windows_cuda_seconds(trace_dir: Path) -> float:
    try:
        row = next(csv.DictReader((trace_dir / "windows.csv").open()))
        return float(row["cuda_seconds"])
    except (OSError, StopIteration, KeyError, ValueError) as exc:
        raise RunnerError(f"raw_trace_missing_block: cannot read binary's own windows.csv: {exc}")


def calibrate_binary_repeat(binary: Path, argv_prefix: list[str], graph_batch: int,
                            target_seconds: float, workdir: Path) -> int:
    """One real invocation at a graph_batch-sized repeat count, then rate-
    extrapolated to the target -- cheap because the driver reports real CUDA-
    event time directly (unlike B's host-time probes, which exist specifically
    to cancel out a separate process's startup cost before a real CUDA-event
    measurement is affordable). Retargeting after each real attempt (below)
    is what actually controls final accuracy, exactly as it does for B and
    for the Python-callable family."""
    probe_dir = workdir / "calibrate"
    probe_dir.mkdir(parents=True, exist_ok=True)
    proc = _run_binary(binary, argv_prefix, graph_batch, graph_batch, probe_dir)
    if proc.returncode != 0:
        raise RunnerError(f"duration_did_not_converge: calibration probe failed: {proc.stderr[-1000:]}")
    observed = _binary_windows_cuda_seconds(probe_dir)
    if observed <= 0:
        raise RunnerError(f"duration_did_not_converge: nonpositive calibration duration {observed}")
    rate = graph_batch / observed
    return max(graph_batch, round(rate * target_seconds))


def _run_binary(binary: Path, argv_prefix: list[str], repeat: int, graph_batch: int, trace_dir: Path):
    import subprocess
    argv = [str(binary)] + [str(a) for a in argv_prefix] + [
        "--repeat", str(repeat), "--graph-batch", str(graph_batch), "--trace-dir", str(trace_dir),
    ]
    return subprocess.run(argv, capture_output=True, text=True)


def run_binary_energy(
    ctx: BinaryEnergyContext, *, window_target_seconds: float, session: int, run_id: str,
    out_dir: Path, gpu_index: int, platform: str, runner_name: str, gpu_uuid: str,
    dry_run: bool = False, sampler_factory=NvmlSampler, graph_batch: int = DEFAULT_GRAPH_BATCH,
) -> dict:
    """Binary-invocation counterpart to ``run_python_callable_energy``.

    ``gpu_uuid`` must come from ``preflight_before_context()``, called before
    anything in ``ctx`` was built (mirrors the Python-callable path exactly,
    for the same reason -- see that function's docstring). Always
    ``launch_model="graph"``: these drivers only implement native CUDA-graph
    capture (APPLICATION_ENERGY_HARNESS_RUNBOOK.md section 4), never a host
    loop, so there is no cross-check mode to select here.

    Returns ``{"status": "raw"|"rejected", "row": {...}}``, never raises.
    """
    base_row = dict(run_id=run_id, session=session, runner=runner_name, parent_id=ctx.parent_id,
                     regime=ctx.regime, candidate_id=ctx.candidate_id,
                     window_target_seconds=window_target_seconds,
                     launch_model="graph", graph_batch=graph_batch)

    if dry_run:
        return dict(status="dry_run", row=dict(base_row, controls=ctx.controls, runtime=ctx.runtime_info))

    check_gpu_unshared_excluding_self(gpu_index)
    wait_for_cooldown(gpu_index)

    calibrate_workdir = out_dir / "raw_attempts" / run_id / f"{runner_name}-s{session}-calibrate"
    calibrate_workdir.mkdir(parents=True, exist_ok=True)
    try:
        precondition = calibrate_binary_repeat(ctx.binary, ctx.argv_prefix, graph_batch,
                                               RAW_TRACE_PRECONDITION_SECONDS, calibrate_workdir)
        launches = calibrate_binary_repeat(ctx.binary, ctx.argv_prefix, graph_batch,
                                           window_target_seconds, calibrate_workdir)
    except RunnerError as exc:
        # A calibration failure (e.g. the M319 reduction capture error) is a real result for this cell:
        # record it as a rejection like every other failure path instead of raising past the caller.
        return dict(status="rejected", row=dict(
            timestamp_utc=iso_now(), rejection_reason="calibration_failed",
            raw_stderr_tail=str(exc)[-1000:], **base_row))
    wait_for_cooldown(gpu_index)

    cache_pre = f"appbin:{runner_name}:{ctx.candidate_id}:pre"
    cache_win = f"appbin:{runner_name}:{ctx.candidate_id}:win"

    for attempt in range(1, MAX_DURATION_ATTEMPTS + 1):
        trace_dir = out_dir / "raw_attempts" / run_id / f"{runner_name}-s{session}-a{attempt}"
        trace_dir.mkdir(parents=True, exist_ok=False)
        pre_dir = trace_dir / "precondition"
        pre_dir.mkdir()
        sampler = sampler_factory(gpu_index)
        sampler.start()
        try:
            proc_pre = _run_binary(ctx.binary, ctx.argv_prefix, precondition, graph_batch, pre_dir)
            if proc_pre.returncode != 0:
                sampler.stop()
                return dict(status="rejected", row=dict(
                    timestamp_utc=iso_now(), rejection_reason="nonzero_exit",
                    raw_stderr_tail=proc_pre.stderr[-2000:], **base_row))
            precondition_seconds = _binary_windows_cuda_seconds(pre_dir)

            proc_block = _run_binary(ctx.binary, ctx.argv_prefix, launches, graph_batch, trace_dir)
            if proc_block.returncode != 0:
                sampler.stop()
                return dict(status="rejected", row=dict(
                    timestamp_utc=iso_now(), rejection_reason="nonzero_exit",
                    raw_stderr_tail=proc_block.stderr[-2000:], **base_row))
            block_seconds = _binary_windows_cuda_seconds(trace_dir)
            # Guarantee at least one sample strictly after the block invocation exits: see the
            # identical comment in run_python_callable_energy for why this matters.
            time.sleep(max(2 * getattr(sampler, "_interval", 0.01), 0.05))
        finally:
            sampler.stop()
        sampler.write_csv(trace_dir / "samples.csv")
        _write_session_csv(trace_dir, precondition_seconds)

        precondition_ok = duration_within_tolerance(precondition_seconds, RAW_TRACE_PRECONDITION_SECONDS)
        window_ok = duration_within_tolerance(block_seconds, window_target_seconds)

        if precondition_ok and window_ok:
            parsed = finalize_raw_trace(trace_dir)
            if not ctx.check():
                return dict(status="rejected", row=dict(
                    timestamp_utc=iso_now(), rejection_reason="correctness_check_failed",
                    raw_stderr_tail="", **base_row,
                ))
            row = dict(
                timestamp_utc=iso_now(), family="application",
                cuda_visible_devices=str(gpu_index), nvml_gpu_id=gpu_index, gpu_uuid=gpu_uuid,
                host=os.uname().nodename if hasattr(os, "uname") else "unknown",
                source_revision=ctx.source_revision, source_sha256=ctx.source_sha256,
                source_path=ctx.source_path, launch_count=launches, trace_dir=str(trace_dir),
                precondition_cuda_seconds=precondition_seconds,
                precondition_target_seconds=RAW_TRACE_PRECONDITION_SECONDS,
                correctness_check=True, runtime_json=json.dumps(ctx.runtime_info, sort_keys=True),
                controls_json=json.dumps(ctx.controls, sort_keys=True),
                **base_row,
            )
            row.update(parsed)
            return dict(status="raw", row=row)

        if attempt == MAX_DURATION_ATTEMPTS:
            return dict(status="rejected", row=dict(
                timestamp_utc=iso_now(), rejection_reason="duration_did_not_converge",
                raw_stderr_tail=(f"after {attempt} attempts: precondition {precondition_seconds}s "
                                  f"(target {RAW_TRACE_PRECONDITION_SECONDS}s), block {block_seconds}s "
                                  f"(target {window_target_seconds}s)"),
                **base_row,
            ))

        if not precondition_ok:
            precondition = retarget_launches_from_event_time(
                cache_pre, precondition, precondition_seconds, RAW_TRACE_PRECONDITION_SECONDS)
        if not window_ok:
            launches = retarget_launches_from_event_time(
                cache_win, launches, block_seconds, window_target_seconds)

    raise AssertionError("unreachable: loop always returns or raises within MAX_DURATION_ATTEMPTS")


def _require_matching_header(path: Path, fields: list[str]) -> None:
    """Refuse to append to a CSV whose header differs from ``fields``.

    ``append_csv`` writes a header only for a new file and silently drops unknown columns, so
    appending graph rows (M311, which added ``launch_model``/``graph_batch``) to a pre-M311 file
    would lose exactly the columns that stop graph and host rows being pooled.
    """
    if not path.exists():
        return
    with path.open(newline="") as f:
        header = next(csv.reader(f), [])
    if header != fields:
        raise RunnerError(
            f"csv_header_mismatch: {path} has columns {header}, expected {fields}; "
            "use a fresh --out-dir instead of appending (rows would silently lose columns)")


def append_raw_or_rejected(out_dir: Path, result: dict) -> None:
    if result["status"] == "raw":
        _require_matching_header(out_dir / "application_energy_raw.csv", APPLICATION_RAW_FIELDS)
        append_csv(out_dir / "application_energy_raw.csv", APPLICATION_RAW_FIELDS, result["row"])
    elif result["status"] == "rejected":
        _require_matching_header(out_dir / "application_energy_rejected.csv", APPLICATION_REJECTED_FIELDS)
        append_csv(out_dir / "application_energy_rejected.csv", APPLICATION_REJECTED_FIELDS, result["row"])
