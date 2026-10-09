#!/usr/bin/env python3
"""Measurement orchestrator of the H100 evaluation: runtime and energy of the frozen cells on ONE H100 (WRITTEN AND TESTED ON THE CPU ONLY; never run on a GPU by its author).

Reuses, unchanged except for module constants set at run time, the three Blackwell measurement engines and the application-energy harness:
  unseen kernels  unseen_kernels/gpu/run_unseen.py   (build / time / energy of matrix multiply, Black-Scholes, scan, separable convolution)
  set E           fresh_e/gpu/run_e.py               (scalar product, fast Walsh transform)
  set D           fresh_d/timing_fresh_d.py helpers  (transposes, copies, reductions through the reset runners; the loops below are the H100 form of its run() and of
                  energy_fresh_d.py's main())
  energy          energy_harness/application_energy_harness.py  (idle check, cooldown gate, 90 s precondition, counted window, NVML trace, correctness check, one attempt per cell)
  fresh sets      fresh_f/run_f.py                   (the machine-learning kernels GELU, SwiGLU gate, RMSNorm, rotary embedding, FP32 matrix multiply; the tensor-core matrix multiply and the fused attention:
                  ONE driver binary, fresh_f/gpu/drivers/driver_ml.cu, built on the node with CUDA 12.1 and gated against the committed sm_90 SASS of the combined driver)
What is set instead of Blackwell's values: ARCH sm_90, the UUID/index of the GPU this job owns, PLATFORM h100, the cell list (cells_h100.json), nvcc = the job's CUDA 12.1.

Gates, all checked before any GPU work and again here (the shell wrapper energy_harness/run_h100_eval_measure_cluster.sh checks them first):
  * predictions_h100.json exists in the tree, is an H100 (not plumbing-test) file, equals the sha256 recorded in freeze_record_h100.json, and its freeze commit is a proper ancestor of
    HIPC_COMMIT (git; without git the recorded commit and hash are required and the missing ancestry check is stated in the output);
  * energy windows only with a committed power_sensor_results.json for the H100 whose decision block admits the chosen window and padding within its 3% tolerance (check-window);
    the harness integrates exactly the counted window, so only padding 0 is supported (padding > 0 is refused, not approximated);
  * the cuda-samples copy (a git archive on Cluster with its own MANIFEST.sha256) must carry a manifest byte-identical to the one committed in the tree (pinned by sha256 in the freeze record) and
    every listed file must hash as listed (verify-samples; the wrapper also runs `sha256sum -c --quiet`); nothing is read from or written to /scratch;
  * after the build, every kernel in every binary must have the instruction sequence of the SASS the static analysis used (sass_manifest_h100.json); a cell whose kernel differs is
    refused and recorded, not timed.
No clock, persistence-mode or power-limit change, no sudo, no profiler. One attempt per cell; a rejected cell is recorded, never retried.

    python measure_h100.py plan [--window-s 40] [--profile main|tensor]
    python measure_h100.py check-freeze --tree <tree> --hipc-commit <sha> [--no-git] [--profile main|tensor]
    python measure_h100.py write-freeze-record --repo <git checkout> [--freeze-commit <sha>] [--profile main|tensor]     (the human, after committing the profile's predictions)
    python measure_h100.py verify-samples --root <cuda-samples copy> [--tree <staged tree>]
    python measure_h100.py check-window --window-validation <power_sensor_results.json> --window-s 20 --padding-s 0
    python measure_h100.py run --stage build|time|energy|all --measure runtime|energy|both --samples-root <pinned cuda-samples> --workdir <dir> --out-dir <dir> --booking-ref "<booking log booking>" \\
        --hipc-commit <sha> [--window-validation F --window-s N --padding-s 0] [--only substr] [--dry-run] [--profile main|tensor]

Profiles (cell_sets.py): `main` = the 72 CUDA-samples cells + the 40 machine-learning-kernel cells (ONE job; predictions_h100.json); `tensor` = the 16 tensor-core matrix multiply and fused attention
cells (a later job; predictions_h100_tensor.json, frozen from the first complete calibration run that contains the tensor stage). Energy windows of the H100: 40 s unpadded (the power-sensor result's
shortest admitted window at padding 0), never Blackwell's 15 s. After every energy stage the window gate (window_gate.py) records, per window, the mean power against the enforced limit, the
cap rule (above 98.5% of the limit: excluded for every method alike), clocks, sample gaps and temperature; its rules are written there before any measurement and are not changed afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
SR = HERE.parent
REPO = SR.parents[2]
sys.path.insert(0, str(HERE))
import cell_sets as CS  # noqa: E402
CELLS = HERE / "cells_h100.json"
PRED_NAME = CS.profile("main")["predictions"]
RECORD_NAME = CS.profile("main")["record"]
SAMPLES_MANIFEST_REL = "tiresias/app_runners/app_source_manifests/cuda-samples-5443602_cluster_MANIFEST.sha256"      # committed copy of MANIFEST.sha256 of the Cluster sources copy
SOURCES_REVISION = "5443602d89ed99aede2e4b7bf329daddeadb320e"
BASELINES_NAME = CS.profile("main")["baselines"]
# every input the freeze commit must contain besides the predictions: hashed in the record, re-checked by the job and the scorer (the `main` profile: the 72 CUDA-samples cells and the 40
# machine-learning-kernel cells; the `tensor` profile has its own list, cell_sets.frozen_inputs("tensor"))
FZ_REL_IN_TREE = "tiresias/framework/predictor/h100_eval_freeze"
FROZEN_INPUTS = CS.frozen_inputs("main")
MANIFEST = HERE / "sass_manifest_h100.json"
DEFAULT_WINDOW_S = 40.0                                  # H100: the shortest window the power-sensor result admits at padding 0 (Blackwell's 15 s is not used)
FRESH_NVCC_RELEASE = "Cuda compilation tools, release 12.1, V12.1.66"
POWER_SCHEMA = "power_sensor_results_v1"
CRITERION_PCT = 3.0
ARCH = "sm_90"
PLATFORM = "h100"
SET_FAMILY = CS.SET_FAMILY
SAMPLE_SETS = ("unseen", "set_e", "set_d")               # the sets built from the pinned cuda-samples copy
LAUNCH_TARGET_PER_CELL_S = dict(timing=15.0)            # planning assumption: seconds of board time per cell for the timing stage (probes, windows, input upload)
PRECONDITION_S = 90.0                                    # energy_harness/measurement_runner.RAW_TRACE_PRECONDITION_SECONDS
PER_WINDOW_OVERHEAD_S = 46.0                             # cooldown polls, two calibration probes, process starts, trace finalisation: 151 s median service time at a 15 s window on Blackwell minus 90 + 15
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Refusal(SystemExit):
    def __init__(self, msg):
        super().__init__("REFUSED: " + msg)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_cells(only: str = "", profile: str = "main"):
    """(first cell document of the profile, merged cells of the profile). The document is cells_h100.json for `main` (it carries source_revision)."""
    cells, docs = CS.load_cells(profile)
    doc = docs[0]
    if only:
        cells = [c for c in cells if only in c["cell_id"]]
    return doc, cells


# ------------------------------------------------------------------------------------------------ the three gates
def check_window(path, window_s, padding_s, tree_root=None, use_git=True) -> dict:
    """Refuse unless `path` is a committed power-sensor result for an H100 that admits (window_s, padding_s) within its stated tolerance (at most CRITERION_PCT)."""
    p = Path(path)
    if not p.is_file():
        raise Refusal("WINDOW_VALIDATION %s does not exist: the energy windows need the H100 power-sensor update test (power_sensor_results.json); run MEASURE=runtime without it" % p)
    if tree_root is not None:
        try:
            p.resolve().relative_to(Path(tree_root).resolve())
        except ValueError:
            raise Refusal("WINDOW_VALIDATION %s is not inside the committed tree %s" % (p, tree_root))
    if use_git and tree_root is not None and (Path(tree_root) / ".git").exists():
        r = subprocess.run(["git", "-C", str(tree_root), "ls-files", "--error-unmatch", str(p.resolve())], capture_output=True, text=True)
        if r.returncode != 0:
            raise Refusal("WINDOW_VALIDATION %s is not tracked by git: commit it first" % p)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as ex:
        raise Refusal("WINDOW_VALIDATION is not JSON: %s" % ex)
    if doc.get("schema") != POWER_SCHEMA:
        raise Refusal("WINDOW_VALIDATION schema %r is not %s" % (doc.get("schema"), POWER_SCHEMA))
    tol = doc.get("tolerance_pct")
    if not isinstance(tol, (int, float)) or not 0 < tol <= CRITERION_PCT:
        raise Refusal("the validation was run with tolerance %r%%, looser than the %.1f%% criterion" % (tol, CRITERION_PCT))
    runs = doc.get("runs") or []
    if not runs:
        raise Refusal("WINDOW_VALIDATION has no runs")
    bad = [str(r.get("device")) for r in runs if "H100" not in json.dumps(r.get("device"))]
    if bad:
        raise Refusal("WINDOW_VALIDATION is not an H100 result (device of at least one run: %s)" % bad[0])
    try:
        window_s, padding_s = float(window_s), float(padding_s)
    except (TypeError, ValueError):
        raise Refusal("WINDOW_S and PADDING_S must be numbers")
    if not window_s > 0:
        raise Refusal("WINDOW_S must be positive")
    if padding_s != 0:
        raise Refusal("PADDING_S=%g: energy_harness/application_energy_harness.py integrates exactly the counted window [begin, end] and has no padding support; only PADDING_S=0 is accepted "
                      "(use the window the validation admits at padding 0)" % padding_s)
    dec = (doc.get("decision") or {}).get("padding_0s")
    if not dec or dec.get("min_load_s") is None:
        raise Refusal("the validation has no window that meets %.1f%% at padding 0 (decision.padding_0s.min_load_s is null): no energy window length is admitted; MEASURE=runtime only" % tol)
    if window_s < float(dec["min_load_s"]):
        raise Refusal("WINDOW_S=%g is shorter than the shortest window the validation admits at padding 0 (%g s, tolerance %.1f%%)" % (window_s, dec["min_load_s"], tol))
    return dict(file=p.name, sha256=sha256_file(p), tolerance_pct=tol, window_s=window_s, padding_s=padding_s, min_load_s_at_padding_0=dec["min_load_s"], runs=len(runs),
                power_reading_update_period_ms=doc.get("power_reading_update_period_ms"))


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo)] + list(args), capture_output=True, text=True)


def write_freeze_record(repo, freeze_commit=None, out=None, profile="main"):
    """The human runs this after committing the profile's predictions (the commit hash cannot be inside the commit it names). Reads git; never overwrites."""
    repo = Path(repo)
    prof = CS.profile(profile)
    PRED_NAME = prof["predictions"]
    rel = "tiresias/framework/predictor/h100_eval_freeze/" + PRED_NAME
    if freeze_commit is None:
        r = _git(repo, "log", "--diff-filter=A", "--format=%H", "--", rel)
        commits = [x for x in r.stdout.split() if x]
        if r.returncode != 0 or len(commits) != 1:
            raise Refusal("cannot find the single commit that added %s (git log said %r); pass --freeze-commit" % (rel, r.stdout.strip()))
        freeze_commit = commits[0]
    if not SHA1.match(freeze_commit):
        raise Refusal("freeze commit must be a full 40-character sha")
    blob = subprocess.run(["git", "-C", str(repo), "show", "%s:%s" % (freeze_commit, rel)], capture_output=True)      # bytes: no newline conversion
    if blob.returncode != 0:
        raise Refusal("%s is not in commit %s" % (rel, freeze_commit))
    committed = hashlib.sha256(blob.stdout).hexdigest()
    disk = sha256_file(HERE / PRED_NAME)
    shas = {}
    for name in CS.frozen_inputs(profile) + (PRED_NAME,):
        if not (HERE / name).is_file():
            raise Refusal("%s is missing: every frozen input must exist before the record is written (the baselines file may record a baseline as not run, but it must exist)" % name)
        shas[name] = sha256_file(HERE / name)
    out = Path(out) if out else HERE / prof["record"]
    if out.exists():
        raise Refusal("%s exists; never overwritten" % out)
    manifest_file = REPO / SAMPLES_MANIFEST_REL
    needs_samples = profile == "main"
    if needs_samples and not manifest_file.is_file():
        raise Refusal("%s is missing: the committed manifest of the cuda-samples copy is part of the frozen inputs" % SAMPLES_MANIFEST_REL)
    doc = dict(schema="h100_freeze_record/1", profile=profile, freeze_commit=freeze_commit, predictions_sha256=disk, sha256=shas,
               samples_manifest_sha256=sha256_file(manifest_file) if needs_samples else None,
               note="Written after the freeze commit. Every file hash is that of the working-tree file when this record was written; predictions_sha256 equals the blob committed at freeze_commit "
                    "(checked: %s)." % ("yes" if committed == disk else "NO - the working tree differs from the committed blob; do not use"))
    if committed != disk:
        raise Refusal("the working-tree %s differs from its blob at %s; nothing written" % (PRED_NAME, freeze_commit))
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", out)
    return doc


def check_freeze(tree, hipc_commit, use_git=True, profile="main") -> dict:
    """Refuse unless the predictions were frozen before this tree's commit. Returns what was verified (recorded in the result)."""
    tree = Path(tree)
    prof = CS.profile(profile)
    PRED_NAME, RECORD_NAME = prof["predictions"], prof["record"]
    d = tree / "tiresias/framework/predictor/h100_eval_freeze" if (tree / "tiresias").is_dir() else tree
    pred, rec = d / PRED_NAME, d / RECORD_NAME
    if not pred.is_file():
        raise Refusal("%s is missing from the staged tree: the predictions must be frozen and committed before any H100 timing" % PRED_NAME)
    if not SHA1.match(hipc_commit or ""):
        raise Refusal("HIPC_COMMIT must be the full 40-character commit the tree was archived from")
    doc = json.loads(pred.read_text(encoding="utf-8"))
    if doc.get("kind") != "h100_frozen_predictions" or doc.get("warning"):
        raise Refusal("%s is not an H100 frozen-predictions file (kind %r)" % (PRED_NAME, doc.get("kind")))
    if not rec.is_file():
        raise Refusal("%s is missing: after committing %s run `measure_h100.py write-freeze-record` and commit the record before the measurement commit" % (RECORD_NAME, PRED_NAME))
    r = json.loads(rec.read_text(encoding="utf-8"))
    if r.get("schema") != "h100_freeze_record/1" or not SHA1.match(r.get("freeze_commit", "")) or not SHA256.match(r.get("predictions_sha256", "")):
        raise Refusal("%s is malformed" % RECORD_NAME)
    if sha256_file(pred) != r["predictions_sha256"]:
        raise Refusal("%s differs from the sha256 recorded at the freeze (%s): the predictions were changed after the freeze" % (PRED_NAME, r["predictions_sha256"][:12]))
    recorded = r.get("sha256") or {}
    for name in CS.frozen_inputs(profile):
        if name not in recorded:
            raise Refusal("%s does not record %s: the baselines and the cuda-samples manifest are frozen with the predictions" % (RECORD_NAME, name))
        if not (d / name).is_file():
            raise Refusal("%s is missing from the staged tree" % name)
        if sha256_file(d / name) != recorded[name]:
            raise Refusal("%s differs from the sha256 recorded at the freeze" % name)
    tree_root = tree if (tree / "tiresias").is_dir() else None
    if tree_root is not None and profile == "main":
        mf = tree_root / SAMPLES_MANIFEST_REL
        if not r.get("samples_manifest_sha256"):
            raise Refusal("%s does not record the sha256 of the cuda-samples manifest" % RECORD_NAME)
        if not mf.is_file() or sha256_file(mf) != r["samples_manifest_sha256"]:
            raise Refusal("the committed cuda-samples manifest %s is missing or differs from the sha256 recorded at the freeze" % SAMPLES_MANIFEST_REL)
    if r["freeze_commit"] == hipc_commit:
        raise Refusal("the freeze commit equals HIPC_COMMIT: the predictions must be committed in a commit that predates the measurement code's commit")
    note = "ancestry not verified by git (no git here): relied on the recorded freeze commit %s and the matching sha256" % r["freeze_commit"][:12]
    if use_git:
        repo = tree if (tree / ".git").exists() else None
        if repo is not None:
            anc = _git(repo, "merge-base", "--is-ancestor", r["freeze_commit"], hipc_commit)
            if anc.returncode == 1:
                raise Refusal("freeze commit %s is not an ancestor of HIPC_COMMIT %s" % (r["freeze_commit"][:12], hipc_commit[:12]))
            if anc.returncode != 0:
                raise Refusal("git could not check the ancestry of the freeze commit (rc=%d): %s" % (anc.returncode, anc.stderr.strip()[:200]))
            note = "git: freeze commit %s is a proper ancestor of HIPC_COMMIT %s" % (r["freeze_commit"][:12], hipc_commit[:12])
    return dict(profile=profile, freeze_commit=r["freeze_commit"], predictions_sha256=r["predictions_sha256"], hipc_commit=hipc_commit, ancestry=note,
                calibration=doc.get("calibration", {}).get("sha256"), calibration_uuid=doc.get("calibration", {}).get("uuid"))


# ------------------------------------------------------------------------------------------------ plan and cost estimate
def plan(window_s=DEFAULT_WINDOW_S, cells=None, profile="main"):
    _, cells = load_cells(profile=profile) if cells is None else (None, cells)
    by = {}
    for c in cells:
        by.setdefault((c["set"], c["family"], c["tier"]), 0)
        by[(c["set"], c["family"], c["tier"])] += 1
    n = len(cells)
    service = PRECONDITION_S + window_s + PER_WINDOW_OVERHEAD_S
    energy_s = n * service
    timing_s = n * LAUNCH_TARGET_PER_CELL_S["timing"]
    build_s = 600.0
    return dict(cells=n, by_set_family_tier={"%s/%s/%s" % k: v for k, v in sorted(by.items())}, window_s=window_s,
                assumptions=dict(per_energy_window_service_s="%g = %g precondition + %g counted window + %g gates, probes, process starts and finalisation (Blackwell: 151 s median at a 15 s window)"
                                 % (service, PRECONDITION_S, window_s, PER_WINDOW_OVERHEAD_S),
                                 per_cell_timing_s=LAUNCH_TARGET_PER_CELL_S["timing"], build_and_sass_gate_s=build_s,
                                 rejected_windows="up to 3 attempts per cell are possible inside the harness (duration retargeting); 10% extra is added"),
                gpu_seconds=dict(build_and_gate=build_s, timing=timing_s, energy=energy_s, energy_with_10pct_retarget_margin=energy_s * 1.1),
                gpu_hours_total_runtime_only=round((build_s + timing_s) / 3600, 2), gpu_hours_total_with_energy=round((build_s + timing_s + energy_s * 1.1) / 3600, 2))


# ------------------------------------------------------------------------------------------------ paths and sources
def refuse_scratch(path, what):
    """Everything the job reads or writes lives under $HOME or the project directory: /scratch purges after 15 days."""
    if str(Path(path).expanduser()).replace("\\", "/").startswith("/scratch"):
        raise Refusal("%s %s is under /scratch, which purges after 15 days; use a path under $HOME (or the project directory)" % (what, path))


def verify_samples(root, tree=None):
    """The cuda-samples copy on Cluster carries MANIFEST.sha256 (sha256sum list, './path' form). Refuse unless (1) that file is byte-identical to the manifest committed in the staged tree, (2) every
    listed file hashes as listed (what `sha256sum -c --quiet` checks; the wrapper runs that too), (3) .source_rev, when present, is the pinned revision. Extra files are not an error (as with sha256sum -c)."""
    root = Path(root)
    committed = (Path(tree) if tree else REPO) / SAMPLES_MANIFEST_REL
    if not committed.is_file():
        raise Refusal("the committed manifest %s is missing from the tree" % SAMPLES_MANIFEST_REL)
    copy_manifest = root / "MANIFEST.sha256"
    if not copy_manifest.is_file():
        raise Refusal("%s is missing: the cuda-samples copy must carry its MANIFEST.sha256" % copy_manifest)
    if copy_manifest.read_bytes() != committed.read_bytes():
        raise Refusal("%s is not byte-identical to the committed manifest %s" % (copy_manifest, SAMPLES_MANIFEST_REL))
    rev = root / ".source_rev"
    if rev.is_file() and rev.read_text().strip() != SOURCES_REVISION:
        raise Refusal("%s says %r, not the pinned revision %s" % (rev, rev.read_text().strip(), SOURCES_REVISION))
    n, bad = 0, []
    for line in committed.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        name = name.lstrip("*")
        f = root / name[2:] if name.startswith("./") else root / name
        n += 1
        if not f.is_file() or sha256_file(f) != digest:
            bad.append(name)
    if bad:
        raise Refusal("%d of %d files of the cuda-samples copy at %s differ from the manifest or are missing (first: %s)" % (len(bad), n, root, ", ".join(bad[:3])))
    return dict(root=str(root), manifest=SAMPLES_MANIFEST_REL, manifest_sha256=sha256_file(committed), files_verified=n, source_rev=rev.read_text().strip() if rev.is_file() else None)


# ------------------------------------------------------------------------------------------------ GPU identity
def gpu_identity(smi_runner=subprocess.run, environ=None):
    """The one H100 this job owns: (nvidia-smi index, uuid, name). Refuses unless it can say which GPU the job's CUDA_VISIBLE_DEVICES selects."""
    environ = os.environ if environ is None else environ
    r = smi_runner(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], capture_output=True, text=True)
    if r.returncode != 0:
        raise Refusal("nvidia-smi failed: %s" % r.stderr.strip()[:200])
    rows = [[x.strip() for x in ln.split(",")] for ln in r.stdout.splitlines() if ln.strip()]
    if not rows:
        raise Refusal("nvidia-smi lists no GPU")
    if len(rows) == 1:
        row = rows[0]
    else:
        cvd = environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        match = [x for x in rows if x[0] == cvd] if cvd.isdigit() else []
        if len(match) != 1:
            raise Refusal("nvidia-smi lists %d GPUs and CUDA_VISIBLE_DEVICES=%r is not one integer index: cannot tell which GPU this job owns" % (len(rows), cvd))
        row = match[0]
    idx, uuid, name = row[0], row[1], row[2]
    if "H100" not in name:
        raise Refusal("the GPU is %r, not an H100" % name)
    if not re.match(r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", uuid):
        raise Refusal("malformed UUID %r" % uuid)
    return dict(index=int(idx), uuid=uuid, name=name, visible_gpus=len(rows))


def power_limit_w(smi_runner=subprocess.run):
    """The enforced power limit of this job's GPU in W (nvidia-smi power.limit), or None when it cannot be read; recorded in measure_meta.json and used by the window gate."""
    try:
        r = smi_runner(["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0")], capture_output=True, text=True)
        return float(r.stdout.strip().splitlines()[0]) if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, ValueError, IndexError):
        return None


# ------------------------------------------------------------------------------------------------ engines (module constants only)
def load_engine(kind):
    path = SR / ("unseen_kernels/gpu/run_unseen.py" if kind == "unseen" else "fresh_e/gpu/run_e.py")
    spec = importlib.util.spec_from_file_location("h100_engine_" + kind, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patch_engine(mod, kind, gpu, cells, workdir_default):
    """Blackwell constants -> this job's H100. Everything else of the engine (guards, build, probes, windows, energy harness calls) is used as is."""
    mod.ARCH = ARCH
    mod.PLATFORM = PLATFORM
    mod.GPU_INDEX = gpu["index"]
    mod.APPROVED_UUID = gpu["uuid"]
    mod.load_cells = lambda: [c for c in cells if c["family"] in mod.FAMILIES]
    mod.verify_samples_root = lambda root: mod.PINNED_REVISION      # the manifest check (verify_samples) has already passed for this copy, which is a git archive without .git
    mod.DEFAULT_WORKDIR = workdir_default
    return mod


def engine_args(**kw):
    return argparse.Namespace(**kw)


# ------------------------------------------------------------------------------------------------ SASS gate
def parse_sass_text(text):
    sys.path.insert(0, str(HERE))
    import make_sass_manifest_h100 as MS
    return MS.sass_hash(text)


def find_cuobjdump(nvcc):
    for cand in (os.environ.get("HARNESS_CUOBJDUMP", ""), str(Path(nvcc).with_name("cuobjdump")), shutil.which("cuobjdump") or ""):
        if cand and Path(cand).is_file():
            return cand
    raise Refusal("cuobjdump not found next to nvcc, on PATH, or in HARNESS_CUOBJDUMP")


def sass_gate(binaries_by_kid, nvcc, manifest=None, runner=subprocess.run, profile="main"):
    """binaries_by_kid: {kernel id: binary path}. -> {kid: dict(ok, detail)}. A kernel whose normalised instruction sequence differs from the manifest fails."""
    manifest = manifest or dict(kernels=CS.merged_sass_manifest(profile))
    cuobjdump = find_cuobjdump(nvcc) if runner is subprocess.run else "cuobjdump"
    out = {}
    for kid, binary in sorted(binaries_by_kid.items()):
        want = manifest["kernels"].get(kid)
        if want is None:
            out[kid] = dict(ok=False, detail="no manifest entry")
            continue
        d = runner([cuobjdump, "-sass", "-fun", want["symbol"], "-arch", ARCH, str(binary)], capture_output=True, text=True)
        if d.returncode != 0 or "/*" not in d.stdout:
            out[kid] = dict(ok=False, detail="cuobjdump failed or found no function %s (rc=%s): %s" % (want["symbol"], d.returncode, (d.stderr or d.stdout)[-200:]))
            continue
        try:
            got = parse_sass_text(d.stdout)
        except ValueError as ex:
            out[kid] = dict(ok=False, detail="cannot parse the dump: %s" % ex)
            continue
        out[kid] = dict(ok=got == want["instruction_sequence_sha256"], symbol=want["symbol"],
                        detail="identical to the analysed SASS" if got == want["instruction_sequence_sha256"] else "instruction sequence differs from the analysed SASS (%s vs %s)" % (got[:12], want["instruction_sequence_sha256"][:12]))
    return out


def cells_without_failed_kernels(cells, gate):
    ok, refused = [], []
    for c in cells:
        bad = [k["kid"] for k in c["kernels"] if not gate.get(k["kid"], {}).get("ok")]
        (refused if bad else ok).append((c, bad))
    return [c for c, _ in ok], [dict(cell_id=c["cell_id"], failed_kernels=b) for c, b in refused]


# ------------------------------------------------------------------------------------------------ fresh sets (machine-learning kernels, tensor-core matrix multiply, fused attention)
def load_fresh_engine():
    """fresh_f/run_f.py as a module (Blackwell build, SASS gate, timing and energy loops). One driver binary serves all three sets; FRESH_SET only chooses its default cell file, which is replaced."""
    path = SR / "fresh_f" / "run_f.py"
    spec = importlib.util.spec_from_file_location("h100_engine_fresh", path)
    mod = importlib.util.module_from_spec(spec)
    old = os.environ.get("FRESH_SET")
    os.environ["FRESH_SET"] = "f"
    try:
        spec.loader.exec_module(mod)
    finally:
        if old is None:
            os.environ.pop("FRESH_SET", None)
        else:
            os.environ["FRESH_SET"] = old
    return mod


def patch_fresh_engine(mod, gpu, cells):
    """Blackwell constants -> this job's H100 and the CUDA 12.1 toolchain. The build command is the engine's own (nvcc -arch=<ARCH> -Xcompiler -O2 with the three kernel headers), which is the command the
    committed sm_90 build record names; its hash gate against Blackwell's isolated SASS is not used (the H100 gate is sass_gate against the committed sm_90 manifest)."""
    mod.ARCH = ARCH
    mod.PLATFORM = PLATFORM
    mod.GPU_INDEX = gpu["index"]
    mod.APPROVED_UUID = gpu["uuid"]
    mod.NVCC_RELEASE = FRESH_NVCC_RELEASE
    mod.load_cells = lambda: list(cells)
    return mod


def fresh_cell_kids(cells):
    return sorted({k["kid"] for c in cells for k in c["kernels"]})


def write_fresh_gate(workdir, gate, kids):
    """gate.json of the engine (list of dict(kid, symbol, sass_equal, detail)) for the kernels of the cells it will run; the engine refuses to time unless every listed kernel passed."""
    rec = [dict(kid=k, symbol=gate[k].get("symbol"), sass_equal=bool(gate[k]["ok"]), detail=gate[k]["detail"]) for k in kids if k in gate]
    Path(workdir).mkdir(parents=True, exist_ok=True)
    (Path(workdir) / "gate.json").write_text(json.dumps(rec, indent=1))
    return rec


# ------------------------------------------------------------------------------------------------ set D (reset runners)
def set_d_module():
    sys.path.insert(0, str(SR / "fresh_d"))
    import timing_fresh_d as T
    return T


def set_d_prepare(T, gpu):
    T.EXPECTED_UUID = gpu["uuid"]
    os.environ["HARNESS_CUDA_ARCH"] = ARCH
    return T


def set_d_build(T, R, source_root, workdir, cells):
    """The runners' own build_binary() on the manifest-verified copy. Their verify_source() (git HEAD of the checkout) is replaced by the manifest check done before (the copy is a git archive)."""
    runners = {c["timing"]["runner"] for c in cells}
    nvcc = R["copy_runner"].find_nvcc()
    out = {}
    for name in sorted(runners):
        wd = Path(workdir) / name
        wd.mkdir(parents=True, exist_ok=True)
        out[name] = R[name].build_binary(Path(source_root).expanduser().resolve(), wd, nvcc)
        print("built %s -> %s" % (T.RUNNER_BINARY[name], out[name]), flush=True)
    return out, nvcc


def set_d_binaries_by_kid(cells, binaries):
    out = {}
    for c in cells:
        out[c["kernels"][0]["kid"]] = binaries[c["timing"]["runner"]]
    return out


def set_d_time(T, R, cells, binaries, workdir, windows, say):
    """The H100 form of fresh_d/timing_fresh_d.py run(): calibration run, one discarded warm run, `windows` timed runs, median; output check after the last window."""
    results = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=T.EXPECTED_UUID)
    for idx, c in enumerate(cells):
        T.refuse_if_busy()
        t = c["timing"]
        cd = Path(workdir) / ("cell_%02d" % idx)
        t0 = time.time()
        files = T.make_inputs(R, c, cd)
        binary = binaries[t["runner"]]
        rec = dict(cell_id=c["cell_id"], set="set_d", kernel=c["kernel"], runner=t["runner"], numeric_args=t["numeric_args"])
        cal = T.run_driver(binary, t, files, T.GRAPH_BATCH, cd / "t_cal", env)
        repeat, per_launch0 = T.window_launches_for(cal)
        T.run_driver(binary, t, files, repeat, cd / "t_warm", env)
        raw = [T.run_driver(binary, t, files, repeat, cd / ("t_%d" % w), env) for w in range(windows)]
        per = [x / repeat for x in raw]
        rec.update(repeat=repeat, graph_batch=T.GRAPH_BATCH, per_launch_runtime_s=T.median(per), per_launch_min_s=min(per), per_launch_max_s=max(per),
                   correct=bool(T.check_output(R, c, files)), wall_s=round(time.time() - t0, 1))
        results.append(rec)
        say("%-66s repeat=%-7d %.3f us/launch correct=%s" % (c["cell_id"], repeat, rec["per_launch_runtime_s"] * 1e6, rec["correct"]))
        shutil.rmtree(cd, ignore_errors=True)
    return results


def set_d_energy(T, R, cells, binaries, workdir, out_dir, timing_by_cell, window_s, session, run_prefix, gpu, say, samples_revision, nvcc):
    sys.path.insert(0, str(REPO / "energy_harness"))
    import application_energy_harness as H
    sys.path.insert(0, str(SR / "fresh_d"))
    import energy_fresh_d as ED
    out_dir.mkdir(parents=True, exist_ok=True)
    any_rejected = False
    for i, c in enumerate(cells):
        t = c["timing"]
        cid = c["cell_id"]
        run_id = "%s-%s" % (run_prefix, cid.replace("/", "-"))
        batch = ED.choose_graph_batch(timing_by_cell[cid])
        cd = Path(workdir) / ("energy_cell_%02d" % i)
        try:
            gpu_uuid = H.preflight_before_context(gpu["index"], PLATFORM)
            if T.norm_uuid(gpu_uuid) != T.norm_uuid(gpu["uuid"]):
                raise H.RunnerError("uuid_mismatch: nvidia-smi index %d reports %s, not this job's %s" % (gpu["index"], gpu_uuid, gpu["uuid"]))
            files = T.make_inputs(R, c, cd)
            ctx = H.BinaryEnergyContext(
                parent_id=c["operator_id"], regime=c["regime"], candidate_id=c["candidate_id"], source_revision=samples_revision, source_sha256="see the pinned revision",
                source_path=c["kernel"], controls=dict(family=c["family"], kernel=c["kernel"], tier=c["tier"], argv_positional=[str(x) for x in t["numeric_args"]] + [str(f) for f in files]),
                runtime_info=dict(nvcc=nvcc, arch=ARCH, driver=t["runner"], binary_sha256=sha256_file(binaries[t["runner"]]), host_opt=True),
                binary=Path(binaries[t["runner"]]), argv_prefix=[str(x) for x in t["numeric_args"]] + [str(f) for f in files],
                check=lambda c=c, files=files: bool(T.check_output(R, c, files)))
            result = H.run_binary_energy(ctx, window_target_seconds=window_s, session=session, run_id=run_id, out_dir=out_dir, gpu_index=gpu["index"], platform=PLATFORM,
                                         runner_name="h100_eval_set_d_%s" % t["runner"], gpu_uuid=gpu_uuid, graph_batch=batch)
        except H.RunnerError as exc:
            result = dict(status="rejected", row=dict(timestamp_utc=H.iso_now(), run_id=run_id, session=session, runner="h100_eval_set_d_%s" % t["runner"], parent_id=c["operator_id"],
                                                      regime=c["regime"], candidate_id=c["candidate_id"], window_target_seconds=window_s, rejection_reason="harness_gate: %s" % str(exc)[:200],
                                                      raw_stderr_tail=str(exc)[-1000:]))
            H.append_raw_or_rejected(out_dir, result)
            say("STOP: harness gate failed for %s: %s" % (cid, exc))
            return 3
        H.append_raw_or_rejected(out_dir, result)
        with open(out_dir / "h100_set_d_energy_log.jsonl", "a") as f:
            f.write(json.dumps(dict(cell_id=cid, run_id=run_id, status=result["status"], graph_batch=batch, board_energy_j_per_launch=result["row"].get("board_energy_j_per_launch"),
                                    rejection_reason=result["row"].get("rejection_reason"), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))) + "\n")
        say("  %s %s" % (result["status"].upper(), result["row"].get("rejection_reason", "")))
        any_rejected |= result["status"] != "raw"
        shutil.rmtree(cd, ignore_errors=True)
    return 3 if any_rejected else 0


# ------------------------------------------------------------------------------------------------ run
def cmd_run(a) -> int:
    profile = getattr(a, "profile", "main")
    doc, cells = load_cells(a.only, profile)
    if not cells:
        raise Refusal("no cells selected")
    stages = ["build", "time", "energy"] if a.stage == "all" else [a.stage]
    want_energy = a.measure in ("energy", "both") and "energy" in stages
    if a.measure == "energy" and a.stage in ("time",):
        raise Refusal("--measure energy needs the timing stage's result (graph batches); use --stage all or run time first")
    tree = Path(a.tree_root).resolve()
    window = None
    if want_energy or a.dry_run:
        if a.window_validation:
            if a.window_s is None or a.padding_s is None:
                raise Refusal("--window-validation needs --window-s and --padding-s")
            window = check_window(a.window_validation, a.window_s, a.padding_s, tree_root=tree, use_git=not a.no_git)
        elif want_energy:
            raise Refusal("energy windows need --window-validation <committed H100 power_sensor_results.json> (and --window-s, --padding-s): the window length and padding come from the "
                          "H100 power-sensor update test, not from a Blackwell default. Use --measure runtime to time without energy.")
    freeze = None
    if not a.dry_run or a.hipc_commit:
        freeze = check_freeze(tree, a.hipc_commit, use_git=not a.no_git, profile=profile)
    if a.dry_run:
        p = plan(a.window_s if a.window_s else DEFAULT_WINDOW_S, cells)
        print(json.dumps(dict(stage=stages, measure=a.measure, freeze=freeze, window_validation=window, plan=p), indent=1))
        for c in cells:
            print("%-72s %-6s %-6s %-4s kernels=%s" % (c["cell_id"], c["set"], c["family"], c["tier"], ",".join(k["kid"] for k in c["kernels"])))
        return 0
    if len(a.booking_ref.strip()) < 8:
        raise Refusal("--booking-ref (the the booking log booking entry) is required")
    for v in ("samples_root", "workdir", "out_dir"):
        if not getattr(a, v):
            raise Refusal("--%s is required" % v.replace("_", "-"))
    for v, what in (("samples_root", "--samples-root"), ("workdir", "--workdir"), ("out_dir", "--out-dir")):
        refuse_scratch(getattr(a, v), what)
    uses_samples = any(c["set"] in SAMPLE_SETS for c in cells)
    samples_check = verify_samples(Path(a.samples_root).expanduser(), tree) if uses_samples else None
    gpu = gpu_identity()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    workdir, out_dir = Path(a.workdir).expanduser(), Path(a.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(schema="h100_eval_measure/1", profile=profile, gpu=gpu, freeze=freeze, window_validation=window, samples=samples_check, measure=a.measure, booking_ref=a.booking_ref, hipc_commit=a.hipc_commit,
                cells_sha256={n: sha256_file(HERE / n) for n in CS.profile(profile)["cells"]}, sass_manifest_sha256={n: sha256_file(HERE / n) for n in CS.profile(profile)["sass_manifest"]},
                enforced_power_limit_w=power_limit_w(), board_matches_calibration=(None if not freeze else
                                                                                                                         (freeze.get("calibration_uuid") in (None, gpu["uuid"]))))
    (out_dir / "measure_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    if freeze and freeze.get("calibration_uuid") not in (None, gpu["uuid"]):
        print("NOTE: this GPU (%s) is not the board of the calibration (%s); recorded in measure_meta.json, not a refusal" % (gpu["uuid"], freeze["calibration_uuid"]), flush=True)
    t_start = time.time()
    say = lambda m: print("[%7.1fs] %s" % (time.time() - t_start, m), flush=True)
    samples = Path(a.samples_root).expanduser()
    nvcc = os.environ.get("HARNESS_NVCC", "")
    if not nvcc or not Path(nvcc).is_file():
        raise Refusal("HARNESS_NVCC must name the CUDA 12.1 nvcc of the job")
    by_set = {s: [c for c in cells if c["set"] in (s,)] for s in SET_FAMILY}
    engines = {}
    for kind, s in (("unseen", "unseen"), ("fresh_e", "set_e")):
        eng = load_engine("unseen" if kind == "unseen" else "fresh_e")
        engines[s] = patch_engine(eng, kind, gpu, by_set[s], str(workdir / s))
    fresh_cells = [c for c in cells if c["set"] in CS.ML_SETS]
    FE = patch_fresh_engine(load_fresh_engine(), gpu, fresh_cells) if fresh_cells else None
    T = set_d_prepare(set_d_module(), gpu)
    R = T.import_runners() if by_set["set_d"] else None
    gate_path = workdir / "sass_gate_h100.json"
    rc_total = 0
    timing_name = "timing_h100_result.json" if profile == "main" else "timing_h100_%s_result.json" % profile

    if "build" in stages:
        binaries_by_kid = {}
        for s in ("unseen", "set_e"):
            if not by_set[s]:
                continue
            eng = engines[s]
            rc = eng.cmd_build(engine_args(samples_root=str(samples), workdir=str(workdir / s), only="", host_opt=True))
            if rc != 0:
                raise Refusal("the %s build failed (rc=%s)" % (s, rc))
            man = eng.load_manifest(workdir / s)
            fam_bin = {f: e["binary"] for f, e in man["families"].items()}
            for c in by_set[s]:
                for k in c["kernels"]:
                    binaries_by_kid[k["kid"]] = fam_bin[c["family"]]
        d_bins = {}
        if by_set["set_d"]:
            d_bins, _ = set_d_build(T, R, samples, workdir / "set_d", by_set["set_d"])
            binaries_by_kid.update(set_d_binaries_by_kid(by_set["set_d"], d_bins))
            (workdir / "set_d" / "binaries.json").write_text(json.dumps({k: str(v) for k, v in d_bins.items()}, indent=1))
        if FE is not None:
            rc = FE.cmd_build(engine_args(workdir=str(workdir / "fresh"), only=""))
            if rc != 0:
                raise Refusal("the build of the combined machine-learning driver failed (rc=%s)" % rc)
            fresh_binary = FE.manifest(workdir / "fresh")["binary"]
            for kid in fresh_cell_kids(fresh_cells):
                binaries_by_kid[kid] = fresh_binary
        gate = sass_gate(binaries_by_kid, nvcc, profile=profile)
        if FE is not None:
            write_fresh_gate(workdir / "fresh", gate, fresh_cell_kids(fresh_cells))
        gate_path.write_text(json.dumps(dict(schema="h100_sass_gate/1", nvcc=nvcc, kernels=gate, passed=all(g["ok"] for g in gate.values())), indent=1, sort_keys=True) + "\n")
        for kid, g in sorted(gate.items()):
            say("sass gate %-26s %s %s" % (kid, "PASS" if g["ok"] else "FAIL", g["detail"]))
        if not all(g["ok"] for g in gate.values()):
            say("some kernels differ from the analysed SASS: their cells are refused (listed in the result), the others continue")

    if "time" in stages or "energy" in stages:
        if not gate_path.is_file():
            raise Refusal("%s is missing: run the build stage (it records the SASS gate) first" % gate_path)
        gate = json.loads(gate_path.read_text())["kernels"]
        ok_cells, refused = cells_without_failed_kernels(cells, gate)
        by_ok = {s: [c for c in ok_cells if c["set"] == s] for s in SET_FAMILY}
        for s in ("unseen", "set_e"):
            engines[s].load_cells = (lambda s=s: by_ok[s])
        fresh_ok = [c for c in ok_cells if c["set"] in CS.ML_SETS]
        if FE is not None:
            FE.load_cells = lambda: list(fresh_ok)
        timing_path = out_dir / timing_name
        if FE is not None:                                  # the engine reads gate.json of its workdir: rewrite it for the kernels that still run (a failed kernel's cells are refused above)
            write_fresh_gate(workdir / "fresh", gate, fresh_cell_kids([c for c in ok_cells if c["set"] in CS.ML_SETS]))

    if "time" in stages:
        if timing_path.exists():
            raise Refusal("%s exists; never overwritten" % timing_path)
        records, parts = [], {}
        for s in ("unseen", "set_e"):
            if not by_ok[s]:
                continue
            part = out_dir / ("timing_%s_engine.json" % s)
            t0 = time.time()
            rc = engines[s].cmd_time(engine_args(samples_root=str(samples), workdir=str(workdir / s), only="", dry_run=False, i_have_a_booking=True, booking_ref=a.booking_ref,
                                                 run_timeout_s=1800.0, out=str(part), windows=3))
            rc_total = rc_total or rc
            parts[s] = dict(rc=rc, wall_s=round(time.time() - t0, 1), file=part.name)
            if part.exists():
                for r in json.loads(part.read_text())["cells"]:
                    records.append(dict(r, set=s))
        if by_ok["set_d"]:
            d_bins = {k: Path(v) for k, v in json.loads((workdir / "set_d" / "binaries.json").read_text()).items()}
            t0 = time.time()
            T.refuse_if_busy()
            recs = set_d_time(T, R, by_ok["set_d"], d_bins, workdir / "set_d_run", 5, say)
            parts["set_d"] = dict(rc=0 if all(r["correct"] for r in recs) else 2, wall_s=round(time.time() - t0, 1))
            rc_total = rc_total or parts["set_d"]["rc"]
            records += recs
        if fresh_ok:
            part = out_dir / "timing_fresh_engine.json"
            t0 = time.time()
            rc = FE.cmd_time(engine_args(workdir=str(workdir / "fresh"), only="", booking_ref=a.booking_ref, out=str(part)))
            rc_total = rc_total or rc
            parts["fresh"] = dict(rc=rc, wall_s=round(time.time() - t0, 1), file=part.name)
            if part.exists():
                for r in json.loads(part.read_text())["cells"]:
                    records.append(dict(r, set=next(c["set"] for c in fresh_ok if c["cell_id"] == r["cell_id"])))
        timing = dict(schema="h100_timing/1", meta=meta, engines=parts, gate_refused_cells=refused, cells=sorted(records, key=lambda r: r["cell_id"]),
                      note="per_launch_runtime_s: median over windows of cuda_seconds / launches; a cell with correct=false is excluded from the criteria and listed")
        timing_path.write_text(json.dumps(timing, indent=1, sort_keys=True) + "\n")
        say("wrote %s (%d cells timed, %d refused by the SASS gate)" % (timing_path, len(records), len(refused)))

    if "energy" in stages and a.measure in ("energy", "both"):
        if not timing_path.is_file():
            raise Refusal("%s is missing: the energy stage takes its graph batches from the timing stage's result" % timing_path)
        timing = json.loads(timing_path.read_text())
        tb = {r["cell_id"]: r["per_launch_runtime_s"] for r in timing["cells"] if r.get("correct") and r.get("per_launch_runtime_s")}
        for s in ("unseen", "set_e"):
            ec = [c for c in by_ok[s] if c["cell_id"] in tb]
            if not ec:
                continue
            engines[s].load_cells = (lambda ec=ec: ec)
            tj = out_dir / ("timing_for_energy_%s.json" % s)
            tj.write_text(json.dumps(dict(cells=[dict(cell_id=k, per_launch_runtime_s=v) for k, v in tb.items()])))
            rc = engines[s].cmd_energy(engine_args(samples_root=str(samples), workdir=str(workdir / s), only="", dry_run=False, i_have_a_booking=True, booking_ref=a.booking_ref,
                                                   run_timeout_s=1800.0, window_target_seconds=float(window["window_s"]), session=a.session, run_id_prefix="H100-EVAL-%s" % s.upper(),
                                                   out_dir=str(out_dir / ("energy_%s" % s)), graph_batch=0, timing_json=str(tj)))
            rc_total = rc_total or rc
        fc = [c for c in fresh_ok if c["cell_id"] in tb]
        if fc:
            FE.load_cells = lambda fc=fc: fc
            tj = out_dir / "timing_for_energy_fresh.json"
            tj.write_text(json.dumps(dict(cells=[dict(cell_id=k, per_launch_runtime_s=v) for k, v in tb.items()])))
            rc = FE.cmd_energy(engine_args(workdir=str(workdir / "fresh"), only="", booking_ref=a.booking_ref, timing_json=str(tj), out_dir=str(out_dir / "energy_fresh"),
                                           window_target_seconds=float(window["window_s"]), session=a.session, run_id_prefix="H100-EVAL-FRESH"))
            rc_total = rc_total or rc
        dc = [c for c in by_ok["set_d"] if c["cell_id"] in tb]
        if dc:
            d_bins = {k: Path(v) for k, v in json.loads((workdir / "set_d" / "binaries.json").read_text()).items()}
            rc = set_d_energy(T, R, dc, d_bins, workdir / "set_d_run", out_dir / "energy_set_d", tb, float(window["window_s"]), a.session, "H100-EVAL-SET_D", gpu, say,
                              doc["source_revision"], nvcc)
            rc_total = rc_total or rc
        gate_dirs = [out_dir / n for n in ("energy_unseen", "energy_set_e", "energy_set_d", "energy_fresh") if (out_dir / n / "application_energy_raw.csv").is_file()]
        if gate_dirs and meta.get("enforced_power_limit_w"):
            import window_gate as WG
            wg = WG.gate(gate_dirs, float(meta["enforced_power_limit_w"]))
            (out_dir / ("window_gate_%s.json" % profile)).write_text(json.dumps(wg, indent=1, sort_keys=True) + "\n")
            say("window gate: %d windows, %d below cap, %d near cap, %d above the cap rule, %d clock-throttled, %d without trace"
                % (wg["windows"], wg["below_cap"], len(wg["near_cap"]), len(wg["above_cap_rule"]), len(wg["clock_throttled"]), len(wg["no_trace"])))
    print("done; exit code %d" % rc_total)
    return rc_total


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S); p.add_argument("--profile", choices=sorted(CS.PROFILES), default="main")
    f = sub.add_parser("check-freeze"); f.add_argument("--tree", required=True); f.add_argument("--hipc-commit", required=True); f.add_argument("--no-git", action="store_true")
    f.add_argument("--profile", choices=sorted(CS.PROFILES), default="main")
    w = sub.add_parser("write-freeze-record"); w.add_argument("--repo", required=True); w.add_argument("--freeze-commit", default=None); w.add_argument("--out", default=None)
    w.add_argument("--profile", choices=sorted(CS.PROFILES), default="main")
    v = sub.add_parser("verify-samples"); v.add_argument("--root", required=True); v.add_argument("--tree", default=None)
    c = sub.add_parser("check-window"); c.add_argument("--window-validation", required=True); c.add_argument("--window-s", required=True); c.add_argument("--padding-s", required=True)
    c.add_argument("--tree", default=None); c.add_argument("--no-git", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--stage", choices=["build", "time", "energy", "all"], required=True)
    r.add_argument("--measure", choices=["runtime", "energy", "both"], required=True)
    r.add_argument("--tree-root", default=str(REPO)); r.add_argument("--hipc-commit", default=""); r.add_argument("--no-git", action="store_true")
    r.add_argument("--samples-root", default=""); r.add_argument("--workdir", default=""); r.add_argument("--out-dir", default=""); r.add_argument("--booking-ref", default="")
    r.add_argument("--window-validation", default=""); r.add_argument("--window-s", type=float, default=None); r.add_argument("--padding-s", type=float, default=None)
    r.add_argument("--session", type=int, default=1); r.add_argument("--only", default=""); r.add_argument("--dry-run", action="store_true")
    r.add_argument("--profile", choices=sorted(CS.PROFILES), default="main")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "plan":
        print(json.dumps(plan(a.window_s, profile=a.profile), indent=1))
        return 0
    if a.cmd == "check-freeze":
        print(json.dumps(check_freeze(a.tree, a.hipc_commit, use_git=not a.no_git, profile=a.profile), indent=1))
        return 0
    if a.cmd == "write-freeze-record":
        write_freeze_record(a.repo, a.freeze_commit, a.out, profile=a.profile)
        return 0
    if a.cmd == "verify-samples":
        refuse_scratch(a.root, "--root")
        print(json.dumps(verify_samples(Path(a.root).expanduser(), a.tree), indent=1))
        return 0
    if a.cmd == "check-window":
        print(json.dumps(check_window(a.window_validation, a.window_s, a.padding_s, tree_root=a.tree, use_git=not a.no_git), indent=1))
        return 0
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
