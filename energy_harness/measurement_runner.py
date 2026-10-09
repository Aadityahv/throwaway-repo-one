#!/usr/bin/env python3
"""Common runner for Task B's measurement contract (tiresias/app_runners/measurement_contract.md).

Turns a fixture + catalog-cell + window target into a single verified GPU measurement, or a
rejected record with a stated reason. Never silently produces a plausible-but-unmeasured number
(AGENTS.md's broken-tools rule): every failure path writes to rejected_records.csv with a reason,
none fall through to a fabricated energy value.

Hard constraints enforced here, not just documented:
  - CUDA_VISIBLE_DEVICES must be exactly "1" in the runner's own environment before any build or
    launch happens (contract 2.1). GPU 0 is never a valid target.
  - -DGPU_ID=1 is always passed explicitly at build time (contract 2.2) -- never left at a
    fixture's own default, which differs per fixture and is wrong for two of the three fixtures
    if left unset.
  - UUID verification (contract 2.3) must pass before any timed measurement.

Usage:
  measurement_runner.py --dry-run --fixture triad --controls '{"n":1024,"launches":1}' \
      --window-target-seconds 15 --session 1
  measurement_runner.py --fixture triad --controls '{...}' --window-target-seconds 15 \
      --session 1 --run-id R1 --out-dir tiresias/app_runners/sentinel_raw

--dry-run performs no GPU/build action; it only prints the resolved argv/env/schema-row skeleton
so device/build/environment/launch/window controls can be reviewed without touching any GPU.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO / "build" / "reset"

REJECTION_REASONS = {
    "nonzero_exit", "insufficient_idle_samples", "uuid_mismatch",
    "cuda_visible_devices_unset", "correctness_check_failed", "build_failed",
    "gpu0_targeted_refused", "gpu1_shared_during_run", "parse_failed",
    "duration_did_not_converge", "incomplete_contract_telemetry",
    "raw_trace_finalization_failed", "raw_trace_missing_block",
    "d_identity_unsupported", "d_identity_source_mismatch", "d_identity_incomplete",
}

RAW_TRACE_PRECONDITION_SECONDS = 90.0
SESSION_START_MAX_TEMPERATURE_C = 70
COOLDOWN_POLL_SECONDS = 10
COOLDOWN_REQUIRED_POLLS = 2
COOLDOWN_MAX_WAIT_SECONDS = 600

DURATION_PROBE = Path(__file__).resolve().parent / "duration_probe.sh"
MAX_DURATION_ATTEMPTS = 3  # matches M214/M217/M221's "retarget twice, then fail loud" pattern

# --- Platform/GPU-index parameterization (Task F, M246) --------------------------------------
# The contract (measurement_contract.md Sec.2.1/2.2) was written against Blackwell's dual-GPU
# situation: GPU0 drives a real person's desktop and must never be targeted, so GPU1 is the only
# valid physical index and CUDA_VISIBLE_DEVICES=1 is load-bearing to remap logical device 0 (which
# every fixture's cudaSetDevice(0) call always selects) onto it. That is a Blackwell-specific
# fact, not a property of this runner's own logic -- lib_duration_control.sh/duration_probe.sh
# take a gpu_selector as a plain argument and assume nothing about which index is safe. Ada is a
# single-GPU box (confirmed live via `nvidia-smi -L`, 2026-09-16: exactly one GPU, index 0) so its
# only valid target is index 0, with no remap needed. Rather than hardcoding either number, both
# are threaded through explicitly as --platform/--gpu-index; defaults below reproduce the exact
# prior hardcoded Blackwell behavior for every existing caller that does not pass these flags.
PLATFORM_DEFAULT_GPU_INDEX = {"blackwell": 1}  # ada has no default -- must be passed explicitly,
# verified live rather than assumed (AGENTS.md's hardware-verification rule)

RAW_FIELDS = [
    "timestamp_utc", "run_id", "session", "fixture", "parent_id", "regime", "candidate_id",
    "cuda_visible_devices", "nvml_gpu_id", "gpu_uuid", "arch", "host", "commit",
    "source_git_blob", "source_sha256", "binary_sha256", "nvcc_version", "window_target_seconds", "launch_count",
    "trace_dir", "integration_interval_s", "counted_launch_interval_s",
    "board_energy_j_total", "board_energy_j_per_launch", "idle_power_w_before",
    "idle_power_w_after", "idle_power_w_baseline", "active_power_w", "dynamic_power_w",
    "dynamic_energy_j_total", "dynamic_energy_j_per_launch", "idle_sample_count_before",
    "idle_sample_count_after", "active_sample_count", "correctness_check", "launch_argv_json",
    "precondition_cuda_seconds", "precondition_target_seconds", "graph_batch",
    "cell_id", "operation_id", "context_id", "context_json",
]
REJECTED_FIELDS = [
    "timestamp_utc", "run_id", "session", "fixture", "parent_id", "regime", "candidate_id",
    "window_target_seconds", "rejection_reason", "raw_stderr_tail",
]

FIXTURES = {
    "triad": dict(
        # The retired aggregate fixture is intentionally not a primary-label source.  The
        # stabilization recorder is the source-native raw implementation used for triad labels.
        source="energy_harness/triad_stabilization.cu",
        needs_predictor_energy_flag=False,   # sampler is unconditional, not gated
        needs_pthread=True,
        role="streaming_load",
        stdout_is_csv_row=True,
    ),
    "transpose": dict(
        source="workloads/e2e_transpose.cu",
        needs_predictor_energy_flag=True,
        needs_pthread=False,
        role="streaming_load_store",
        stdout_is_csv_row=True,
    ),
    "shared_stage": dict(
        source="workloads/predictor_shared_stage_load.cu",
        needs_predictor_energy_flag=True,
        needs_pthread=True,
        role="shared_reuse",
        stdout_is_csv_row=True,
    ),
    "store": dict(
        source="workloads/predictor_store.cu",
        needs_predictor_energy_flag=True,
        needs_pthread=False,
        role="store",
        stdout_is_csv_row=True,
    ),
    "reduction": dict(
        source="workloads/predictor_reduction.cu",
        needs_predictor_energy_flag=True,
        needs_pthread=False,
        role="shared_reduction",
        stdout_is_csv_row=True,
    ),
}

class RunnerError(RuntimeError):
    pass


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def git_blob_hash(path: Path) -> str:
    out = sh(["git", "-C", str(REPO), "hash-object", str(path)])
    if out.returncode != 0:
        raise RunnerError(f"git hash-object failed for {path}: {out.stderr}")
    return out.stdout.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def attach_d_identity(row: dict, *, platform_profile: dict, measurement_protocol: dict) -> dict:
    """Attach D's exact identity/context to an admitted B row or fail loudly.

    This uses D's catalog/source formula to make the cell identity, but first
    proves that the source B measured is the same source D described.  It is
    intentionally opt-in: old rows remain historical raw evidence, never
    silently upgraded to joinable labels.
    """
    reset_dir = REPO / "tiresias" / "app_runners"
    sys.path.insert(0, str(reset_dir))
    try:
        from static_features import extract
    finally:
        sys.path.pop(0)
    build = {
        "commit": row["commit"],
        "binary_sha256": row["binary_sha256"],
        "nvcc_version": row["nvcc_version"],
    }
    compiler = {"nvcc_version": row["nvcc_version"]}
    record = extract(row["parent_id"], row["regime"], row["candidate_id"],
                     compiler=compiler, platform=platform_profile, build=build,
                     measurement_protocol=measurement_protocol)
    if record.get("status") != "supported":
        raise RunnerError("d_identity_unsupported: no D extractor for this measured parent")
    if record["identity"]["source_sha256"] != row["source_sha256"]:
        raise RunnerError(
            "d_identity_source_mismatch: B measured source does not equal D catalog source; "
            "do not join this label to that descriptor"
        )
    if record["context"]["status"] != "complete" or not record["context_id"]:
        raise RunnerError("d_identity_incomplete: platform/build/protocol context is not joinable")
    row.update({"cell_id": record["cell_id"], "operation_id": record["operation_id"],
                "context_id": record["context_id"],
                "context_json": json.dumps(record["context"], sort_keys=True)})
    return row


def nvcc_version(nvcc_path: str) -> str:
    out = sh([nvcc_path, "--version"])
    if out.returncode != 0:
        return "unknown"
    m = re.search(r"release ([0-9.]+)", out.stdout)
    return m.group(1) if m else out.stdout.strip().splitlines()[-1]


def assert_env_authorized(gpu_index: int) -> None:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    expected = str(gpu_index)
    if cvd != expected:
        raise RunnerError(
            f"cuda_visible_devices_unset: CUDA_VISIBLE_DEVICES must be exactly '{expected}', got '{cvd}'"
        )


def query_target_gpu_uuid(gpu_index: int) -> str:
    out = sh(["nvidia-smi", "-i", str(gpu_index), "--query-gpu=uuid",
              "--format=csv,noheader"])
    if out.returncode != 0:
        raise RunnerError(f"uuid_mismatch: nvidia-smi query failed: {out.stderr}")
    return out.stdout.strip()


def wait_for_cooldown(gpu_index: int) -> dict:
    """Require an idle thermal entry state; return immutable manifest fields."""
    readings: list[int] = []
    consecutive = 0
    waited = 0
    while waited <= COOLDOWN_MAX_WAIT_SECONDS:
        out = sh(["nvidia-smi", "-i", str(gpu_index), "--query-gpu=temperature.gpu",
                  "--format=csv,noheader,nounits"])
        try:
            temperature = int(out.stdout.strip())
        except ValueError:
            temperature = SESSION_START_MAX_TEMPERATURE_C + 1
        readings.append(temperature)
        consecutive = consecutive + 1 if temperature <= SESSION_START_MAX_TEMPERATURE_C else 0
        if consecutive >= COOLDOWN_REQUIRED_POLLS:
            return {"cooldown_waited_seconds": waited, "cooldown_initial_temperature_c": readings[0],
                    "cooldown_final_temperature_c": temperature}
        time.sleep(COOLDOWN_POLL_SECONDS)
        waited += COOLDOWN_POLL_SECONDS
    raise RunnerError("cooldown_timeout: target GPU did not reach the recorded entry temperature")


# Ada is a desktop workstation (unlike the headless Cluster/Blackwell boxes this check was
# written for) and always has a small, permanent, non-GPU-compute helper resident -- discovered
# live 2026-09-16 running energy_harness/run_ada_device_smoke.sh. It shows up in --query-compute-apps but is
# not a real workload and never clears, so treating it as "shared" would permanently refuse every
# Ada launch. Named explicitly (not a wildcard) so an unexpected *different* process still FATALs.
ADA_BENIGN_COMPUTE_APP_SUBSTRINGS = ("snapd-desktop-integration",)


def check_target_gpu_idle_and_unshared(gpu_index: int, platform: str) -> None:
    procs = sh(["nvidia-smi", "-i", str(gpu_index),
                "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    if procs.returncode != 0:
        raise RunnerError(
            f"gpu_occupancy_unknown: GPU{gpu_index} process query failed "
            f"(exit {procs.returncode}): {procs.stderr.strip()}")
    if not procs.stdout.strip():
        return
    lines = [l for l in procs.stdout.strip().splitlines() if l.strip()]
    if platform == "ada":
        unexpected = [l for l in lines
                      if not any(s in l for s in ADA_BENIGN_COMPUTE_APP_SUBSTRINGS)]
        if not unexpected:
            print(f"NOTE: GPU{gpu_index} --query-compute-apps lists only the known resident "
                  f"desktop helper, not a real workload:\n{procs.stdout}", file=sys.stderr)
            return
        raise RunnerError(
            f"gpu1_shared_during_run: GPU{gpu_index} has unexpected active compute processes:\n"
            + "\n".join(unexpected))
    raise RunnerError(
        f"gpu1_shared_during_run: GPU{gpu_index} has active compute processes:\n{procs.stdout}")


def build_argv(fixture: str, arch: str, gpu_index: int, nvcc_path: str) -> tuple[list[str], Path]:
    spec = FIXTURES[fixture]
    source = REPO / spec["source"]
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    output = BUILD_DIR / f"{fixture}_energy_{arch}"
    argv = [nvcc_path, "-O3", "-std=c++17", f"-arch={arch}"]
    if spec["needs_predictor_energy_flag"]:
        argv.append("-DPREDICTOR_ENERGY")
    # -DGPU_ID=<gpu_index> always explicit -- contract 2.2. Never omitted, never left at a
    # fixture default. gpu_index is the runner's own verified physical target (contract 2.1/2.2
    # collapse to the same number for every platform this runner has been run on so far: Blackwell
    # remaps CUDA_VISIBLE_DEVICES=1 onto physical GPU1 while NVML opens physical index 1 directly;
    # Ada has one GPU at physical/logical index 0 and needs no remap. Both are just "gpu_index".)
    argv.append(f"-DGPU_ID={gpu_index}")
    argv += [str(source), "-o", str(output), "-lnvidia-ml"]
    if spec["needs_pthread"]:
        argv += ["-Xcompiler", "-pthread"]
    return argv, output


# M240 fix: the sentinel proof failed its <=5% window-doubling gate (5.62%/6.53%) because launch
# counts were chosen by a naive single small-scale probe, not this project's converged two-point
# method (M197/M214/M221). That method already lives in energy_harness/lib_duration_control.sh and is reused
# here (not reimplemented) through energy_harness/duration_probe.sh -- a thin bash CLI adapter around the
# library's own calibrated_launches()/retarget_launches()/duration_within_tolerance(), so the
# arithmetic and the noise-floor/probe-ceiling/retry-cap logic are identical to every other
# Blackwell campaign runner in this repo (P0/P1/P2/E-RED and the five core single-axis campaigns).
#
# triad/transpose have no --energy flag (their NVML sampler is unconditional, verified in
# workloads/e2e_tile_triad_calibration.cu / e2e_transpose.cu); their cheap, no-NVML timing path is
# --probe instead, which is what makes the calibration probe inexpensive for them, mirroring why
# lib_duration_control.sh's own calibrated_launches() omits --energy for the M2xx family binaries.
FIXTURE_CALIBRATION_EXTRA_ARGS = {
    "triad": ["--probe"],
    "transpose": ["--probe"],
}


def calibration_kernel_argv(fixture: str, controls: dict, graph_batch: int = 0) -> list[str]:
    """Kernel args for the calibration probe: the same native controls the real measurement will
    use, minus --launches (the probe supplies its own) and minus --energy (calibration must never
    pay the NVML idle-bracket cost -- matches lib_duration_control.sh's convention)."""
    keys = FIXTURE_CONTROL_KEYS.get(fixture)
    if keys is None:
        raise RunnerError(f"parse_failed: no launch-argv mapping declared for fixture={fixture}")
    argv: list[str] = list(FIXTURE_CALIBRATION_EXTRA_ARGS.get(fixture, []))
    for key in keys:
        if key == "launches":
            continue
        underscored = key.replace("-", "_")
        if key in controls:
            value = controls[key]
        elif underscored in controls:
            value = controls[underscored]
        else:
            raise RunnerError(f"parse_failed: control '{key}' missing for fixture={fixture}")
        argv += [f"--{key}", str(value)]
    # M309: the probe must measure the submission path the counted window will actually use.
    # Calibrating host-enqueue launches and then measuring graph replays biased every count low by
    # exactly the host overhead replay removes (Blackwell -8.6%, H100 -19.6%), which is what forced
    # the M301-M304 retry chain.
    if graph_batch:
        argv += ["--graph-batch", str(graph_batch)]
    return argv


def calibrate_launches(cache_key: str, binary: Path, fixture: str, controls: dict,
                        window_target_seconds: float, gpu_selector: str, graph_batch: int = 0) -> int:
    """Initial host-time two-point estimate, via energy_harness/lib_duration_control.sh's
    calibrated_launches() -- probes P and 2P launches, differences them to cancel fixed host cost,
    doubles the probe until the difference clears a noise floor, and extrapolates to the target.
    This is only a starting point (the library's own comment: 'necessarily based on host elapsed
    time'); run_one() below corrects it against real CUDA-event time before accepting a row."""
    kernel_argv = calibration_kernel_argv(fixture, controls, graph_batch)
    out = sh(["bash", str(DURATION_PROBE), "calibrate", cache_key, str(binary), gpu_selector,
              str(window_target_seconds), "--"] + kernel_argv)
    if out.returncode != 0:
        raise RunnerError(f"duration_did_not_converge: calibration failed: {out.stderr.strip()[-1000:]}")
    try:
        return int(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise RunnerError(
            f"duration_did_not_converge: could not parse calibrated launch count from: {out.stdout!r}")


def retarget_launches_from_event_time(cache_key: str, current_launches: int, observed_seconds: float,
                                       window_target_seconds: float) -> int:
    """Corrects the host-time estimate against the CUDA-event duration actually measured in a real
    row, via lib_duration_control.sh's retarget_launches() -- this is the M214 fix: host time does
    not guarantee CUDA-event label duration on a launch-heavy generator."""
    out = sh(["bash", str(DURATION_PROBE), "retarget", cache_key, str(current_launches),
              str(observed_seconds), str(window_target_seconds)])
    if out.returncode != 0:
        raise RunnerError(f"duration_did_not_converge: retarget failed: {out.stderr.strip()[-1000:]}")
    try:
        return int(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise RunnerError(
            f"duration_did_not_converge: could not parse retargeted launch count from: {out.stdout!r}")


def duration_within_tolerance(observed_seconds: float, window_target_seconds: float) -> bool:
    """+/-10% CUDA-event acceptance check, via lib_duration_control.sh's own
    duration_within_tolerance() -- identical tolerance to every other duration-controlled campaign
    in this repo. Distinct from the sentinel reliability gate's separate <=5% window-doubling
    check, which compares two already-accepted rows at W and 2W."""
    out = sh(["bash", str(DURATION_PROBE), "tolerance", str(observed_seconds), str(window_target_seconds)])
    return out.returncode == 0


def raw_kernel_argv(fixture: str, controls: dict) -> list[str]:
    """Native controls for a raw recorder, excluding the free repetition count."""
    args: list[str] = []
    for key in FIXTURE_CONTROL_KEYS[fixture]:
        if key == "launches":
            continue
        value = controls.get(key, controls.get(key.replace("-", "_")))
        if value is None:
            raise RunnerError(f"parse_failed: control '{key}' missing for fixture={fixture}")
        args += [f"--{key}", str(value)]
    return args


def read_precondition_seconds(trace_dir: Path) -> float:
    """The uncounted precondition's CUDA-event duration, as the fixture itself recorded it in
    session.csv. M289: non-triad fixtures derived the precondition launch count from a host-time
    probe and nothing ever checked the realised duration (0.18-160 s against a 90 s target)."""
    try:
        value = float(next(csv.DictReader((trace_dir / "session.csv").open()))["precondition_cuda_seconds"])
    except (OSError, StopIteration, KeyError, TypeError, ValueError, csv.Error) as exc:
        raise RunnerError(f"raw_trace_missing_block: cannot read precondition duration: {exc}")
    if value <= 0:
        raise RunnerError(f"raw_trace_missing_block: nonpositive precondition duration {value}")
    return value


def finalize_raw_trace(trace_dir: Path) -> dict:
    finalized = sh([sys.executable, str(REPO / "energy_harness" / "verify_b_stabilization_trace.py"), str(trace_dir)])
    if finalized.returncode != 0:
        raise RunnerError(f"raw_trace_finalization_failed: {finalized.stderr.strip()[-1000:]}")
    try:
        rows = list(csv.DictReader((trace_dir / "blocks.csv").open()))
    except OSError as exc:
        raise RunnerError(f"raw_trace_missing_block: {exc}")
    if len(rows) != 1:
        raise RunnerError(f"raw_trace_missing_block: expected one canonical block, got {len(rows)}")
    row = rows[0]
    return dict(
        counted_launch_interval_s=float(row["cuda_seconds"]),
        integration_interval_s=(int(row["host_end_monotonic_ns"]) - int(row["host_begin_monotonic_ns"])) / 1e9,
        board_energy_j_total=float(row["board_joules"]),
        board_energy_j_per_launch=float(row["board_joules_per_launch"]),
        active_sample_count=int(row["samples_including_boundaries"]),
        correctness_check=True,
    )


def raw_trace_invocation(fixture: str, binary: Path, controls: dict, precondition: int,
                         launches: int, target_seconds: float, trace_dir: Path,
                         graph_batch: int = 0) -> list[str]:
    base = raw_kernel_argv(fixture, controls)
    if fixture == "triad":
        argv = [str(binary)] + base + ["--precondition-launches", str(precondition),
            "--block-launches", str(launches), "--blocks", "1", "--trace-dir", str(trace_dir),
            "--precondition-target-seconds", str(RAW_TRACE_PRECONDITION_SECONDS),
            "--block-target-seconds", str(target_seconds), "--duration-tolerance-percent", "10",
            "--session-start-max-temperature-c", "70"]
        if graph_batch:
            argv += ["--graph-batch", str(graph_batch)]
        return argv
    argv = [str(binary)] + base + ["--trace-dir", str(trace_dir),
        "--trace-precondition-launches", str(precondition), "--trace-block-launches", str(launches),
        "--trace-blocks", "1"]
    # M296/M300: opt-in CUDA-graph replay, implemented by every raw-trace fixture.
    if graph_batch:
        argv += ["--graph-batch", str(graph_batch)]
    return argv


def parse_stdout_csv_row(fixture: str, stdout: str) -> dict:
    """Each fixture prints exactly one CSV data row to stdout on success (verified per-source).
    This does not reimplement their measurement; it trusts their own idle-bracket/NVML/CUDA-event
    numbers and only reshapes column names into the common schema."""
    lines = [l for l in stdout.strip().splitlines() if l.strip()]
    if not lines:
        raise RunnerError("parse_failed: no stdout produced")
    row = lines[-1].split(",")

    # Every layout below is read verbatim from the fixture's own std::printf() call in source
    # (workloads/*.cu) at the commit this runner ships with -- not guessed or inferred from a
    # header comment. If a source's printf changes, this must be re-verified, not patched blind.
    if fixture == "triad":
        # triad,n,tile,grid_size,threads_per_block,sec,integration_sec,idle_before,idle_after,
        # idle_base,active,dyn,joule,joule_per_launch,idle_n_before,idle_n_after,active_n,
        # launches,correct (19 cols; e2e_tile_triad_calibration.cu main())
        # NOTE (found 2026-09-15 during the sentinel proof run, see sentinel_grid_2026-09-15.md):
        # this parser originally read row[4] as "sec", forgetting that BOTH grid_size (row[3]) and
        # threads_per_block (row[4]) are separate numeric fields before "sec" at row[5] -- an
        # off-by-one that silently mislabeled every column from "sec" onward (e.g. reported
        # "dynamic_energy_j_per_launch" was actually the fixture's *total* joule figure, and
        # "counted_launch_interval_s" was actually the constant threads_per_block=256, not a
        # duration). Fixed here; do not revert without re-verifying against the real printf in
        # workloads/e2e_tile_triad_calibration.cu.
        if len(row) != 19:
            raise RunnerError(f"parse_failed: triad expected 19 columns, got {len(row)}: {row}")
        launches = int(row[17])
        joule_total = float(row[12])
        return dict(
            counted_launch_interval_s=float(row[5]), integration_interval_s=float(row[6]),
            idle_power_w_before=float(row[7]), idle_power_w_after=float(row[8]),
            idle_power_w_baseline=float(row[9]), active_power_w=float(row[10]),
            dynamic_power_w=float(row[11]), dynamic_energy_j_total=joule_total,
            dynamic_energy_j_per_launch=float(row[13]), idle_sample_count_before=int(row[14]),
            idle_sample_count_after=int(row[15]), active_sample_count=int(row[16]),
            launch_count=launches, correctness_check=bool(int(row[18])),
        )
    if fixture == "transpose":
        # e2e_transpose,n,tile,grid_size,threads_per_block,sec,integration_sec,idle_before,
        # idle_after,idle_base,active,dyn,joule,joule_per_launch,idle_before_samples,
        # idle_after_samples,active_samples,launches,correct  (19 cols; e2e_transpose.cu, fixed
        # M246 -- the fixture never emitted integration_sec/sample counts at all before this, so
        # every prior invocation was correctly rejected by require_contract_telemetry() as
        # incomplete_contract_telemetry, not a parser bug. Do not revert without re-verifying
        # against the real printf in workloads/e2e_transpose.cu.
        if len(row) != 19:
            raise RunnerError(f"parse_failed: transpose expected 19 columns, got {len(row)}: {row}")
        launches = int(row[17])
        joule_total = float(row[12])
        return dict(
            counted_launch_interval_s=float(row[5]), integration_interval_s=float(row[6]),
            idle_power_w_before=float(row[7]), idle_power_w_after=float(row[8]),
            idle_power_w_baseline=float(row[9]), active_power_w=float(row[10]),
            dynamic_power_w=float(row[11]), dynamic_energy_j_total=joule_total,
            dynamic_energy_j_per_launch=float(row[13]), idle_sample_count_before=int(row[14]),
            idle_sample_count_after=int(row[15]), active_sample_count=int(row[16]),
            launch_count=launches, correctness_check=bool(int(row[18])),
        )
    if fixture == "shared_stage":
        # sharedstage_energy,layout,tile_elements,stage_elements,chunks,reuse,blocks,threads,
        # stride,batches,launches,sec,integration_sec,idle_before,idle_after,idle_base,active,dyn,
        # joule,samples,idle_before_samples,idle_after_samples,correct,executed_loads,shared_reads
        # (25 cols; predictor_shared_stage_load.cu, fixed M246 -- the fixture never emitted
        # integration_sec/idle sample counts at all before this, so every prior invocation was
        # correctly rejected by require_contract_telemetry() as incomplete_contract_telemetry, not
        # a parser bug. Do not revert without re-verifying against the real printf in
        # workloads/predictor_shared_stage_load.cu.
        if len(row) != 25:
            raise RunnerError(f"parse_failed: shared_stage expected 25 columns, got {len(row)}: {row}")
        launches = int(row[10])
        joule = float(row[18])
        return dict(
            counted_launch_interval_s=float(row[11]), integration_interval_s=float(row[12]),
            idle_power_w_before=float(row[13]), idle_power_w_after=float(row[14]),
            idle_power_w_baseline=float(row[15]), active_power_w=float(row[16]),
            dynamic_power_w=float(row[17]), dynamic_energy_j_total=joule,
            dynamic_energy_j_per_launch=joule / launches if launches else 0.0,
            active_sample_count=int(row[19]), idle_sample_count_before=int(row[20]),
            idle_sample_count_after=int(row[21]), launch_count=launches,
            correctness_check=bool(int(row[22])),
        )
    if fixture == "store":
        # store_energy,...,launches,sec,integration_sec,idle_before,idle_after,idle_base,active,
        # dyn,joule,samples,idle_before_samples,idle_after_samples,correct,executed_stores (23 cols)
        if len(row) != 23:
            raise RunnerError(f"parse_failed: store expected 23 columns, got {len(row)}: {row}")
        launches = int(row[9])
        joule = float(row[16])
        return dict(
            counted_launch_interval_s=float(row[10]), integration_interval_s=float(row[11]),
            idle_power_w_before=float(row[12]), idle_power_w_after=float(row[13]),
            idle_power_w_baseline=float(row[14]), active_power_w=float(row[15]), dynamic_power_w=float(row[16]),
            dynamic_energy_j_total=joule, dynamic_energy_j_per_launch=joule / launches if launches else 0.0,
            active_sample_count=int(row[18]), idle_sample_count_before=int(row[19]),
            idle_sample_count_after=int(row[20]), launch_count=launches, correctness_check=bool(int(row[21])),
        )
    if fixture == "reduction":
        # reduction_energy,...,launches,sec,integration_sec,idle_before,idle_after,idle_base,active,
        # dyn,joule,samples,idle_before_samples,idle_after_samples,correct,executed_loads (24 cols)
        if len(row) != 24:
            raise RunnerError(f"parse_failed: reduction expected 24 columns, got {len(row)}: {row}")
        launches = int(row[10])
        joule = float(row[17])
        return dict(
            counted_launch_interval_s=float(row[11]), integration_interval_s=float(row[12]),
            idle_power_w_before=float(row[13]), idle_power_w_after=float(row[14]),
            idle_power_w_baseline=float(row[15]), active_power_w=float(row[16]), dynamic_power_w=float(row[17]),
            dynamic_energy_j_total=joule, dynamic_energy_j_per_launch=joule / launches if launches else 0.0,
            active_sample_count=int(row[19]), idle_sample_count_before=int(row[20]),
            idle_sample_count_after=int(row[21]), launch_count=launches, correctness_check=bool(int(row[22])),
        )
    raise RunnerError(
        f"parse_failed: no committed column layout yet for fixture={fixture}; "
        f"add its exact stdout schema here before trusting its output (do not guess)."
    )


def require_contract_telemetry(parsed: dict) -> None:
    """Reject, rather than silently blanking, fields the contract declares mandatory.

    Some legacy fixture CSV layouts predate the common schema.  They cannot be
    treated as new-contract measurements until their sources emit the integration
    duration and sample counts; empty columns would otherwise look like valid
    energy records to downstream analysis.
    """
    required = (
        "integration_interval_s", "idle_sample_count_before",
        "idle_sample_count_after", "active_sample_count",
    )
    missing = [field for field in required if field not in parsed or parsed[field] in (None, "")]
    if missing:
        raise RunnerError(
            "incomplete_contract_telemetry: fixture CSV omits required "
            + ", ".join(missing)
        )


# Every fixture except triad requires an explicit --energy flag to take the PREDICTOR_ENERGY
# printf path (verified per-source: transpose/shared_stage/store/reduction all gate on
# options.energy; triad's energy sampler is unconditional, verified in e2e_tile_triad_calibration.cu).
FIXTURE_REQUIRES_ENERGY_FLAG = {"transpose", "shared_stage", "store", "reduction"}
# Store and reduction now emit complete aggregate telemetry, but they have not yet been migrated
# to B's timestamp-integrated primary-label protocol.  Keep the runner fail-closed so a well-shaped
# legacy CSV row cannot be admitted merely because its fields are populated.
LEGACY_ONLY_FIXTURES = {"store", "reduction"}

# Native control keys accepted per fixture, verified against each source's argv parser above.
FIXTURE_CONTROL_KEYS = {
    "triad": ["n", "launches"],
    "transpose": ["n", "launches"],
    "shared_stage": ["layout", "tile-elements", "stage-elements", "chunks", "reuse", "blocks",
                      "threads", "stride", "batches", "launches"],
    "store": ["layout", "tile-elements", "blocks", "threads", "iterations", "stride",
              "step-warps", "batches", "launches"],
    "reduction": ["layout", "reduction-scope", "tile-elements", "blocks", "threads", "iterations",
                  "stride", "step-warps", "batches", "launches"],
}


def fixture_launch_argv(fixture: str, controls: dict) -> list[str]:
    """Build the exact CLI argv for a fixture from its catalog-cell controls dict. Keys in
    `controls` may use either hyphens or underscores (catalog cells commonly use underscores);
    both are accepted and normalized to the fixture's real hyphenated flag name."""
    keys = FIXTURE_CONTROL_KEYS.get(fixture)
    if keys is None:
        raise RunnerError(f"parse_failed: no launch-argv mapping declared for fixture={fixture}")
    argv: list[str] = []
    if fixture in FIXTURE_REQUIRES_ENERGY_FLAG:
        argv.append("--energy")
    for key in keys:
        underscored = key.replace("-", "_")
        if key in controls:
            value = controls[key]
        elif underscored in controls:
            value = controls[underscored]
        else:
            raise RunnerError(f"parse_failed: control '{key}' missing for fixture={fixture}")
        argv += [f"--{key}", str(value)]
    return argv


@dataclass
class RunResult:
    status: str  # "raw" or "rejected"
    row: dict = field(default_factory=dict)


def run_one(fixture: str, parent_id: str, regime: str, candidate_id: str, controls: dict,
            arch: str, window_target_seconds: float, session: int, run_id: str,
            commit: str, dry_run: bool, gpu_index: int, platform: str, nvcc_path: str,
            graph_batch: int = 0) -> RunResult:
    spec = FIXTURES[fixture]
    source = REPO / spec["source"]
    argv, binary = build_argv(fixture, arch, gpu_index, nvcc_path)
    # "launches" is never taken from the caller's controls in live mode -- it is the runner's own
    # calibrated free repetition parameter (contract 1.4), resolved below via the converged
    # two-point method, never a caller-supplied guess.
    base_controls = {k: v for k, v in controls.items() if k not in ("launches", "launch_count")}

    if dry_run:
        # No GPU action in dry-run mode, so the calibrated launch count cannot be computed here --
        # shown as an explicit placeholder rather than a guessed number (fixture_launch_argv just
        # str()s whatever value it is given, so the placeholder passes through unchanged).
        launch_argv = raw_kernel_argv(fixture, base_controls)
        if fixture == "triad":
            launch_argv += ["--precondition-launches", "<calibrated at runtime>",
                            "--block-launches", "<calibrated at runtime>", "--blocks", "1",
                            "--trace-dir", "<immutable attempt directory>"]
        else:
            launch_argv += ["--trace-dir", "<immutable attempt directory>",
                            "--trace-precondition-launches", "<calibrated at runtime>",
                            "--trace-block-launches", "<calibrated at runtime>", "--trace-blocks", "1"]
        return RunResult("raw", dict(
            fixture=fixture, parent_id=parent_id, regime=regime, candidate_id=candidate_id,
            cuda_visible_devices=str(gpu_index), nvml_gpu_id=gpu_index, arch=arch, commit=commit,
            source_git_blob="<dry-run: not computed>", window_target_seconds=window_target_seconds,
            session=session, run_id=run_id, build_argv=argv, launch_argv=launch_argv,
            role=spec["role"],
        ))

    assert_env_authorized(gpu_index)
    check_target_gpu_idle_and_unshared(gpu_index, platform)
    gpu_uuid = query_target_gpu_uuid(gpu_index)
    cooldown_before_calibration = wait_for_cooldown(gpu_index)

    build = sh(argv)
    if build.returncode != 0:
        return RunResult("rejected", dict(
            timestamp_utc=iso_now(), run_id=run_id, session=session, fixture=fixture,
            parent_id=parent_id, regime=regime, candidate_id=candidate_id,
            window_target_seconds=window_target_seconds, rejection_reason="build_failed",
            raw_stderr_tail=build.stderr[-2000:],
        ))

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    # Primary labels are never parsed from the fixtures' retired dynamic-above-idle stdout.
    # Each attempt writes immutable raw samples/boundaries, finalizes blocks.csv, then admits the
    # single canonical block only after its CUDA duration passes the unchanged +/-10% gate.
    cache_key = f"raw:{fixture}:{candidate_id}:{json.dumps(base_controls, sort_keys=True)}"
    if fixture == "triad":
        # M309: the triad binary's --calibrate-launches mode times through the same time_launches()
        # path the recording run uses, so it honours --graph-batch; pass it or the estimate is biased.
        graph_argv = ["--graph-batch", str(graph_batch)] if graph_batch else []
        probe = sh(["bash", str(DURATION_PROBE), "calibrate-event", cache_key, str(binary), str(gpu_index),
                    str(window_target_seconds), "--", "--n", str(base_controls.get("n"))] + graph_argv)
        pre_probe = sh(["bash", str(DURATION_PROBE), "calibrate-event", cache_key + ":pre", str(binary), str(gpu_index),
                        str(RAW_TRACE_PRECONDITION_SECONDS), "--", "--n", str(base_controls.get("n"))] + graph_argv)
        if probe.returncode or pre_probe.returncode:
            raise RunnerError("duration_did_not_converge: triad raw calibration failed")
        launches, precondition = int(probe.stdout.strip().splitlines()[-1]), int(pre_probe.stdout.strip().splitlines()[-1])
    else:
        launches = calibrate_launches(cache_key, binary, fixture, base_controls, window_target_seconds,
                                       str(gpu_index), graph_batch)
        precondition = calibrate_launches(cache_key + ":pre", binary, fixture, base_controls,
                                          RAW_TRACE_PRECONDITION_SECONDS, str(gpu_index), graph_batch)
    cooldown_before_recording = wait_for_cooldown(gpu_index)
    for attempt in range(1, MAX_DURATION_ATTEMPTS + 1):
        trace_dir = REPO / "tiresias" / "app_runners" / "raw_attempts" / run_id / f"{fixture}-s{session}-a{attempt}"
        trace_dir.mkdir(parents=True, exist_ok=False)
        (trace_dir / "runner_manifest.json").write_text(json.dumps({
            "run_id": run_id, "fixture": fixture, "session": session, "gpu_uuid": gpu_uuid,
            "commit": commit, "window_target_seconds": window_target_seconds,
            "precondition_target_seconds": RAW_TRACE_PRECONDITION_SECONDS,
            "cooldown_before_calibration": cooldown_before_calibration,
            "cooldown_before_recording": cooldown_before_recording,
        }, indent=2) + "\n")
        launch_argv = raw_trace_invocation(fixture, binary, base_controls, precondition, launches,
                                           window_target_seconds, trace_dir, graph_batch)
        proc = sh(launch_argv, env=env)
        # The same exclusive-use rule applies after recording: a process appearing during the
        # window invalidates this attempt even if its raw power trace is internally consistent.
        check_target_gpu_idle_and_unshared(gpu_index, platform)
        if proc.returncode:
            # triad itself rejects (exit 7) if its separately calibrated, uncounted
            # precondition misses the same +/-10% duration contract as the counted block.
            # Preserve this failed immutable attempt, retarget from its recorded CUDA
            # duration, and retry within the existing bounded attempt budget.  Previously
            # the common runner returned immediately, making a normal duration-control
            # correction look like an unexplained nonzero exit.
            if fixture == "triad" and proc.returncode == 7 and attempt < MAX_DURATION_ATTEMPTS:
                try:
                    session_row = next(csv.DictReader((trace_dir / "session.csv").open()))
                    observed_precondition = float(session_row["precondition_cuda_seconds"])
                    if observed_precondition <= 0:
                        raise ValueError("nonpositive precondition duration")
                    precondition = retarget_launches_from_event_time(
                        cache_key + ":pre", precondition, observed_precondition,
                        RAW_TRACE_PRECONDITION_SECONDS)
                    # M303: the triad binary validates its precondition first and its counted block
                    # second, and a block rejection leaves windows.csv with only a header -- so the
                    # block's own duration is often unavailable.  Retargeting the precondition alone
                    # (M264's behaviour) can never fix a biased block count, which is how H100
                    # sentinel 354546 burned all three attempts with a converged 89.89 s
                    # precondition and a still-rejected block.  Both windows run the same kernel at
                    # the same per-launch cost, so once the precondition count is correct for 90 s
                    # the block count follows by proportion.  If the block did record a duration,
                    # prefer retargeting from that measurement.
                    try:
                        window_row = next(csv.DictReader((trace_dir / "windows.csv").open()))
                        observed_block = float(window_row["cuda_seconds"])
                    except (OSError, StopIteration, KeyError, TypeError, ValueError, csv.Error):
                        observed_block = 0.0
                    if observed_block > 0:
                        launches = retarget_launches_from_event_time(
                            cache_key, launches, observed_block, window_target_seconds)
                    else:
                        launches = max(1, round(precondition * window_target_seconds
                                                / RAW_TRACE_PRECONDITION_SECONDS))
                    continue
                except (OSError, StopIteration, KeyError, TypeError, ValueError, csv.Error) as exc:
                    return RunResult("rejected", dict(timestamp_utc=iso_now(), run_id=run_id,
                        session=session, fixture=fixture, parent_id=parent_id, regime=regime,
                        candidate_id=candidate_id, window_target_seconds=window_target_seconds,
                        rejection_reason="duration_did_not_converge",
                        raw_stderr_tail=f"triad precondition retry could not read raw duration: {exc}"))
            # M304: a nonzero exit whose attempt still recorded a counted window that missed the
            # +/-10% contract is a duration failure wearing another exit code, not a fixture fault.
            # H100 sentinel 354842 died at c058 this way: one bad calibration asked for 30,482
            # launches where the same cell's session-1 twin used 11,417,811, so the window lasted
            # 0.0397 s, the recorder had too few samples to integrate, and the fixture exited 5
            # ("raw trace telemetry or correctness failed").  Retarget from the recorded duration
            # and retry within the unchanged attempt budget; the rejected attempt stays archived.
            # An attempt with no recorded window is still a hard rejection -- correctness failures
            # must never be retried into a pass.
            recorded_window = 0.0
            try:
                window_row = next(csv.DictReader((trace_dir / "windows.csv").open()))
                recorded_window = float(window_row["cuda_seconds"])
            except (OSError, StopIteration, KeyError, TypeError, ValueError, csv.Error):
                recorded_window = 0.0
            if (recorded_window > 0 and attempt < MAX_DURATION_ATTEMPTS
                    and not duration_within_tolerance(recorded_window, window_target_seconds)):
                launches = retarget_launches_from_event_time(
                    cache_key, launches, recorded_window, window_target_seconds)
                continue
            return RunResult("rejected", dict(timestamp_utc=iso_now(), run_id=run_id, session=session,
                fixture=fixture, parent_id=parent_id, regime=regime, candidate_id=candidate_id,
                window_target_seconds=window_target_seconds, rejection_reason="nonzero_exit",
                raw_stderr_tail=(proc.stderr or f"fixture exit code {proc.returncode}")[-2000:]))
        parsed = finalize_raw_trace(trace_dir)
        observed = parsed["counted_launch_interval_s"]
        observed_precondition = read_precondition_seconds(trace_dir)
        window_ok = duration_within_tolerance(observed, window_target_seconds)
        precondition_ok = duration_within_tolerance(observed_precondition, RAW_TRACE_PRECONDITION_SECONDS)
        if window_ok and precondition_ok:
            row = dict(timestamp_utc=iso_now(), run_id=run_id, session=session, fixture=fixture,
                parent_id=parent_id, regime=regime, candidate_id=candidate_id,
                cuda_visible_devices=str(gpu_index), nvml_gpu_id=gpu_index, gpu_uuid=gpu_uuid, arch=arch,
                host=os.uname().nodename if hasattr(os, "uname") else "unknown", commit=commit,
                source_git_blob=git_blob_hash(source), source_sha256=sha256_file(source), binary_sha256=sha256_file(binary),
                nvcc_version=nvcc_version(nvcc_path), window_target_seconds=window_target_seconds,
                launch_count=launches, trace_dir=str(trace_dir), launch_argv_json=json.dumps(launch_argv),
                precondition_cuda_seconds=observed_precondition,
                precondition_target_seconds=RAW_TRACE_PRECONDITION_SECONDS,
                graph_batch=graph_batch)
            row.update(parsed)
            return RunResult("raw", row)
        if attempt == MAX_DURATION_ATTEMPTS:
            return RunResult("rejected", dict(timestamp_utc=iso_now(), run_id=run_id, session=session,
                fixture=fixture, parent_id=parent_id, regime=regime, candidate_id=candidate_id,
                window_target_seconds=window_target_seconds, rejection_reason="duration_did_not_converge",
                raw_stderr_tail=(f"after {attempt} attempts: counted window {observed}s "
                                 f"(target {window_target_seconds}s), precondition {observed_precondition}s "
                                 f"(target {RAW_TRACE_PRECONDITION_SECONDS}s)")))
        # Each failed attempt stays archived.  Retarget whichever duration missed from its own
        # recorded CUDA-event time; both use the unchanged +/-10% tolerance.
        if not window_ok:
            launches = retarget_launches_from_event_time(cache_key, launches, observed, window_target_seconds)
        if not precondition_ok:
            precondition = retarget_launches_from_event_time(
                cache_key + ":pre", precondition, observed_precondition, RAW_TRACE_PRECONDITION_SECONDS)

    # Cache key mirrors the M2xx campaigns' convention (candidate/scope-keyed, not fixture-wide):
    # every native control that affects per-launch cost must be part of the key, or one count
    # would be asked to fit configurations whose per-launch time differs.
    cache_key = f"{fixture}:{candidate_id}:{json.dumps(base_controls, sort_keys=True)}"

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    launches = calibrate_launches(cache_key, binary, fixture, base_controls, window_target_seconds,
                                   str(gpu_index))
    attempt = 1
    while True:
        result_controls = dict(base_controls, launches=launches)
        launch_argv = fixture_launch_argv(fixture, result_controls)
        proc = sh([str(binary)] + launch_argv, env=env)
        if proc.returncode != 0:
            return RunResult("rejected", dict(
                timestamp_utc=iso_now(), run_id=run_id, session=session, fixture=fixture,
                parent_id=parent_id, regime=regime, candidate_id=candidate_id,
                window_target_seconds=window_target_seconds, rejection_reason="nonzero_exit",
                raw_stderr_tail=proc.stderr[-2000:],
            ))

        try:
            parsed = parse_stdout_csv_row(fixture, proc.stdout)
            require_contract_telemetry(parsed)
        except RunnerError as e:
            reason = str(e).split(":")[0]
            if reason not in REJECTION_REASONS:
                reason = "parse_failed"
            return RunResult("rejected", dict(
                timestamp_utc=iso_now(), run_id=run_id, session=session, fixture=fixture,
                parent_id=parent_id, regime=regime, candidate_id=candidate_id,
                window_target_seconds=window_target_seconds, rejection_reason=reason,
                raw_stderr_tail=str(e)[-2000:],
            ))

        if not parsed.get("correctness_check", True):
            return RunResult("rejected", dict(
                timestamp_utc=iso_now(), run_id=run_id, session=session, fixture=fixture,
                parent_id=parent_id, regime=regime, candidate_id=candidate_id,
                window_target_seconds=window_target_seconds,
                rejection_reason="correctness_check_failed", raw_stderr_tail="",
            ))

        observed = parsed["counted_launch_interval_s"]
        if duration_within_tolerance(observed, window_target_seconds):
            break

        if attempt >= MAX_DURATION_ATTEMPTS:
            # Per M214/M217/M221: the mismatched attempt is discarded, never written, and the
            # runner fails loud rather than silently pooling an off-target row.
            return RunResult("rejected", dict(
                timestamp_utc=iso_now(), run_id=run_id, session=session, fixture=fixture,
                parent_id=parent_id, regime=regime, candidate_id=candidate_id,
                window_target_seconds=window_target_seconds,
                rejection_reason="duration_did_not_converge",
                raw_stderr_tail=(
                    f"CUDA-event duration {observed}s did not converge to {window_target_seconds}s "
                    f"+/-10% after {attempt} attempts (last launches={launches})"
                )[-2000:],
            ))

        # Correct the host-time estimate against the real CUDA-event duration this row measured
        # (M214's fix) and retry -- this attempt's row is intentionally not written.
        launches = retarget_launches_from_event_time(cache_key, launches, observed, window_target_seconds)
        attempt += 1

    row = dict(
        timestamp_utc=iso_now(), run_id=run_id, session=session, fixture=fixture,
        parent_id=parent_id, regime=regime, candidate_id=candidate_id,
        cuda_visible_devices=str(gpu_index), nvml_gpu_id=gpu_index, gpu_uuid=gpu_uuid, arch=arch,
        host=os.uname().nodename if hasattr(os, "uname") else "unknown", commit=commit,
        source_git_blob=git_blob_hash(source), source_sha256=sha256_file(source), binary_sha256=sha256_file(binary),
        nvcc_version=nvcc_version(nvcc_path), window_target_seconds=window_target_seconds,
        launch_argv_json=json.dumps(launch_argv),
    )
    row.update(parsed)
    return RunResult("raw", row)


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append_csv(path: Path, fields: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def append_jsonl(path: Path, row: dict) -> None:
    """Append a new admitted label record without altering historical CSVs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True, choices=sorted(FIXTURES))
    ap.add_argument("--parent-id", default="")
    ap.add_argument("--regime", default="sentinel")
    ap.add_argument("--candidate-id", default="default")
    ap.add_argument("--controls", default="{}", help="JSON dict of native controls")
    ap.add_argument("--arch", default="sm_120")
    ap.add_argument("--window-target-seconds", type=float, required=True)
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--out-dir", default=str(REPO / "tiresias" / "app_runners" / "sentinel_raw"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--graph-batch", type=int, default=0,
                    help="M296: launches captured per CUDA graph replay (store fixture only). "
                         "0 keeps the unchanged host-enqueue submission path.")
    ap.add_argument("--platform", default="blackwell", choices=["blackwell", "ada", "a100", "h100"],
                     help="Governs the hard GPU-index safety check below. Default 'blackwell' "
                          "reproduces this runner's exact original behavior for every existing "
                          "caller that does not pass --platform/--gpu-index. 'a100' is Task F's "
                          "Cluster scheduler port (energy_harness/run_measurement_a100.sh): a `--gres=gpu:1` "
                          "scheduler allocation, not a manually-managed desktop box, so its GPU index "
                          "is whatever scheduler's own cgroup/gres binding resolves inside the job --"
                          "no default assumed here either, same reasoning as ada.")
    ap.add_argument("--gpu-index", type=int, default=None,
                     help="Physical GPU index for nvidia-smi queries, -DGPU_ID, and "
                          "CUDA_VISIBLE_DEVICES. Defaults to 1 only for --platform blackwell "
                          "(GPU0 there drives a human desktop and is never valid). No default is "
                          "assumed for --platform ada or a100 -- pass it explicitly, verified live "
                          "via `nvidia-smi -L` (ada) or the job's own `nvidia-smi`/scheduler env (a100) "
                          "first (AGENTS.md's hardware-verification rule).")
    ap.add_argument("--nvcc-path", default="nvcc",
                     help="Full path to the nvcc to build with. Default 'nvcc' (bare PATH lookup) "
                          "reproduces this runner's exact original behavior for every existing "
                          "caller. Discovered live 2026-09-16 on Ada that a bare 'nvcc' resolves "
                          "/usr/bin/nvcc (CUDA 12.0), not the HARDWARE_GROUND_TRUTH.md-documented "
                          "/usr/local/cuda-13.1/bin/nvcc (CUDA 13.1) -- multiple CUDA installs on "
                          "PATH is a known issue on these boxes (see git history c6733a8: "
                          "'Blackwell has CUDA 12.8, Ada has 13.1'). Pass an explicit path rather "
                          "than assume PATH order resolves to the documented toolchain.")
    ap.add_argument("--emit-d-identity", action="store_true",
                    help="Emit a joinable B label only after exact source/context equality is proven.")
    ap.add_argument("--platform-profile-json", type=Path,
                    help="Immutable platform profile for D's context hash; required with --emit-d-identity.")
    ap.add_argument("--measurement-protocol-json", type=Path,
                    help="Versioned protocol payload for D's context hash; required with --emit-d-identity.")
    args = ap.parse_args()

    gpu_index = args.gpu_index
    if gpu_index is None:
        gpu_index = PLATFORM_DEFAULT_GPU_INDEX.get(args.platform)
    if gpu_index is None:
        print(f"FATAL: --gpu-index is required for --platform {args.platform} "
              "(no default assumed; verify live first).", file=sys.stderr)
        return 1
    if args.platform == "blackwell" and gpu_index != 1:
        print("FATAL: --platform blackwell only ever supports --gpu-index 1 "
              "(GPU 0 drives a human desktop and is never a valid target). Refusing.",
              file=sys.stderr)
        return 1

    commit = sh(["git", "-C", str(REPO), "rev-parse", "HEAD"]).stdout.strip()
    controls = json.loads(args.controls)
    run_id = args.run_id or f"B-SENTINEL-{commit[:7]}"
    if args.emit_d_identity and (args.platform_profile_json is None or args.measurement_protocol_json is None):
        ap.error("--emit-d-identity requires --platform-profile-json and --measurement-protocol-json")

    try:
        result = run_one(args.fixture, args.parent_id or args.fixture, args.regime,
                          args.candidate_id, controls, args.arch, args.window_target_seconds,
                          args.session, run_id, commit, args.dry_run, gpu_index, args.platform,
                          args.nvcc_path, args.graph_batch)
    except RunnerError as e:
        reason = str(e).split(":")[0]
        if reason not in REJECTION_REASONS:
            reason = "parse_failed"
        result = RunResult("rejected", dict(
            timestamp_utc=iso_now(), run_id=run_id, session=args.session, fixture=args.fixture,
            parent_id=args.parent_id, regime=args.regime, candidate_id=args.candidate_id,
            window_target_seconds=args.window_target_seconds, rejection_reason=reason,
            raw_stderr_tail=str(e)[-2000:],
        ))

    if args.dry_run:
        print(json.dumps(result.row, indent=2, default=str))
        return 0

    out_dir = Path(args.out_dir)
    if result.status == "raw" and args.emit_d_identity:
        try:
            attach_d_identity(result.row,
                              platform_profile=json.loads(args.platform_profile_json.read_text()),
                              measurement_protocol=json.loads(args.measurement_protocol_json.read_text()))
        except (RunnerError, json.JSONDecodeError, OSError) as e:
            reason = str(e).split(":")[0]
            if reason not in REJECTION_REASONS:
                reason = "d_identity_incomplete"
            result = RunResult("rejected", dict(
                timestamp_utc=iso_now(), run_id=run_id, session=args.session, fixture=args.fixture,
                parent_id=args.parent_id, regime=args.regime, candidate_id=args.candidate_id,
                window_target_seconds=args.window_target_seconds, rejection_reason=reason,
                raw_stderr_tail=str(e)[-2000:],
            ))
    if result.status == "raw":
        append_csv(out_dir / "raw_records.csv", RAW_FIELDS, result.row)
        if args.emit_d_identity:
            label = dict(result.row)
            label["context"] = json.loads(label.pop("context_json"))
            label["energy_j"] = label["board_energy_j_per_launch"]
            label["label_name"] = "board_energy_j_per_launch"
            append_jsonl(out_dir / "admitted_labels.jsonl", label)
        print(f"RAW {json.dumps(result.row, default=str)}")
    else:
        append_csv(out_dir / "rejected_records.csv", REJECTED_FIELDS, result.row)
        print(f"REJECTED {json.dumps(result.row, default=str)}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
