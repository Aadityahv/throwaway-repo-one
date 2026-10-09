#!/usr/bin/env python3
"""Grid-wide memory footprints of every H100 cell, derived through the sm_90 port (CPU only; reads no measured value).

    python footprints_h100.py --profile main [--jobs 6] [--only <substring>]      # footprints_h100.json (72 CUDA-samples cells + 40 machine-learning-kernel cells)
    python footprints_h100.py --profile tensor [--jobs 3]                         # ml_sets/footprints_tensor_h100.json (16 tensor-core and attention cells)
    python footprints_h100.py --profile main --merge-only                         # merge the cached per-cell files

The current runtime model (predict_runtime_v3j.py, shared-traffic model) charges DRAM reads from the grid-wide unique footprint of every kernel and refuses a kernel whose per-wave
inter-block reuse working set exceeds the L2. The footprint comes from the same interpreted address traces the static analysis uses, re-run here for the sampled blocks of every kernel through the
port (port_ext.py for the CUDA-samples cells, port_ml_ext.py for the machine-learning, tensor and attention cells), with the footprint code of shared_traffic/footprint.py imported
unchanged. The L2 capacity attached to each cell is NOT read here: predict_h100.py takes it from the H100 row of HARDWARE_GROUND_TRUTH.md (via the calibration document check). A cell whose trace
is data-dependent is recorded as `refused` with the reason (never given a default footprint); the model then refuses it if it needs a footprint (DRAM tier).
Per-cell cache: build/footprints/<profile>/ (git-ignored, resumable).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = HERE.parent
sys.path.insert(0, str(SR / "port_common"))
sys.path.insert(0, str(SR / "shared_traffic"))
sys.path.insert(0, str(HERE))
import cell_sets as CS  # noqa: E402

SCHEMA = "h100_footprints/1"
OUT = {"main": HERE / "footprints_h100.json", "tensor": HERE / "ml_sets" / "footprints_tensor_h100.json"}
ML_FILES = {"ml_sets/cells_ml_h100.json", "ml_sets/cells_tensor_h100.json"}
SASS_BASE = SR / "port_h100" / "compiled_cluster_cuda12.1"
SASS_ML = SR / "port_h100" / "compiled_cluster_cuda12.1_ml"


def cache_dir(profile):
    return HERE / "build" / "footprints" / profile


def corners(grid):
    ids = set()
    for xyz in itertools.product(*[sorted({0, g - 1}) for g in grid]):
        ids.add(xyz[0] + xyz[1] * grid[0] + xyz[2] * grid[0] * grid[1])
    return sorted(ids)


def _cell_origin(profile):
    """{cell_id: 'ml' | 'base'}: which static setup (and SASS folder) builds the cell."""
    out = {}
    for rel in CS.profile(profile)["cells"]:
        for c in json.loads((HERE / rel).read_text(encoding="utf-8"))["cells"]:
            out[c["cell_id"]] = "ml" if rel in ML_FILES else "base"
    return out


def _setup(origin):
    if origin == "ml":
        import port_ml_ext as PM
        PM.install("sm_90", SASS_ML)
        return PM.UP
    import port_ext as PX
    import set_d_port as SD
    PX.install("sm_90", SASS_BASE)
    SD.register(blackwell=False)
    return PX.UP


def one(args):
    profile, origin, cell_id = args
    import footprint as FP
    UP = _setup(origin)
    cells, _ = CS.load_cells(profile)
    cell = next(c for c in cells if c["cell_id"] == cell_id)
    path = cache_dir(profile) / (cell_id.replace("/", "__") + ".json")
    t = time.time()
    try:
        import make_cells_h100 as MC
        hw, _ = MC.read_h100_hardware()
        kernels = []
        for kern in cell["kernels"]:
            kid = kern["kid"]
            UP.KW["hw"] = hw
            consts, grid, block, prov = UP.binding(kid, kern)
            nblocks = math.prod(grid)
            sub = dict(kern, sample_blocks=corners(grid)) if kern.get("sampling") == "classes" else kern
            sigs, obs = UP.trace_kernel(kid, sub, consts, grid, block)
            fp = FP.kernel_footprint(UP.U, obs, nblocks)
            fp["kernel_id"] = kid
            fp["sampled_block_ids"] = UP.sample_blocks(grid, sub)
            kernels.append(fp)
        doc = dict(cell_id=cell_id, status="ok", kernels=kernels)
    except Exception as ex:   # a refusal is recorded with its reason, never replaced by a default
        doc = dict(cell_id=cell_id, status="refused", reason="%s: %s" % (type(ex).__name__, ex), kernels=[])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return cell_id, "%s %.0fs" % (doc["status"], time.time() - t)


def merge(profile):
    ids = [c["cell_id"] for c in CS.load_cells(profile)[0]]
    rows = {}
    for cid in ids:
        p = cache_dir(profile) / (cid.replace("/", "__") + ".json")
        if not p.exists():
            raise SystemExit("missing cached footprint for %s; run without --merge-only" % cid)
        rows[cid] = json.loads(p.read_text(encoding="utf-8"))
    refused = sorted(c for c, v in rows.items() if v["status"] != "ok")
    out = dict(schema=SCHEMA, profile=profile, arch="sm_90", cells=len(rows), refused=refused, rows=rows,
               note="Grid-wide unique footprints per pointer argument from the interpreted address traces of the sampled blocks (shared_traffic/footprint.py), sm_90 port. No measured value was read.")
    OUT[profile].write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("merged %s: %d cells, %d without a footprint%s" % (profile, len(rows), len(refused), (": " + ", ".join(refused)) if refused else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(CS.PROFILES), default="main")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args(argv)
    origin = _cell_origin(a.profile)
    if not a.merge_only:
        todo = [(a.profile, origin[c], c) for c in origin if a.only in c and not (cache_dir(a.profile) / (c.replace("/", "__") + ".json")).exists()]
        print("%d cells to derive" % len(todo), flush=True)
        todo.sort(key=lambda t: t[2])
        with cf.ProcessPoolExecutor(max_workers=a.jobs) as pool:
            for cid, msg in pool.map(one, todo):
                print("%-72s %s" % (cid, msg), flush=True)
    if a.only and not a.merge_only:
        return 0
    merge(a.profile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
