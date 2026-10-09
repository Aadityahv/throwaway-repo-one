#!/usr/bin/env python3
"""Static tables of every H100 evaluation cell, through the cross-GPU port (CPU only; reads no measured value).

For each cell of cells_h100.json: the static pipeline (port_common/port_ext.py installed for sm_90 on the CUDA 12.1 builds of port_h100/compiled_cluster_cuda12.1/) builds the
feature row (instruction-family counts, memory traffic, resources and occupancy from the H100 rows of HARDWARE_GROUND_TRUTH.md), the barrier-phase table, the first-touch table and
the shared-memory bank-conflict table (with request-wavefront histograms, as the calibrated runtime model needs). A cell the static side refuses is NOT dropped: it is kept with its
exact reason and counted as a failure by the scorer.

    python build_static_h100.py [--jobs 4] [--only <substring>]        # writes build/cells/*.json then the merged tables
    python build_static_h100.py --merge-only                          # merge the cached per-cell files

Outputs (all deterministic, no timestamps): features_h100.json, phases_h100.json, phases_unique_h100.json, bank_conflicts_h100.json, static_support_h100.json.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = HERE.parent
sys.path.insert(0, str(SR / "port_common"))
sys.path.insert(0, str(HERE))

BUILD = HERE / "build" / "cells"
SASS_ROOT = SR / "port_h100" / "compiled_cluster_cuda12.1"
ARCH = "sm_90"
SUPPORTED = ("supported", "supported_with_assumptions")


def _setup():
    """Install the port for sm_90 on the CUDA 12.1 builds and register all three kernel sets (idempotent; called in every worker)."""
    import port_ext as PX
    import set_d_port as SD
    PX.install(ARCH, SASS_ROOT)
    SD.register(blackwell=False)
    sys.path.insert(0, str(SR / "bank"))
    import bank_unseen as BU
    import make_cells_h100 as MC
    # the wavefront histogram is part of every phase's shared summary (as calibrate/tools/regen_bank_wavefront.py does)
    if not getattr(BU.sum_rows, "_with_histogram", False):
        orig = BU.sum_rows

        def sum_rows_w(rows):
            out = orig(rows)
            h = collections.Counter()
            for r in rows:
                h.update(r["shared_request_wavefront_histogram"])
            out["shared_request_wavefront_histogram"] = {str(k): v for k, v in sorted(h.items(), key=lambda kv: int(kv[0]))}
            return out
        sum_rows_w._with_histogram = True
        BU.sum_rows = sum_rows_w
    return PX, BU, MC


def build_one(cell_id: str):
    PX, BU, MC = _setup()
    UP = PX.UP
    C, P = UP.C, UP.P
    doc = json.loads((HERE / "cells_h100.json").read_text(encoding="utf-8"))
    cell = next(c for c in doc["cells"] if c["cell_id"] == cell_id)
    hw, _ = MC.read_h100_hardware()
    out = BUILD / (cell_id.replace("/", "__") + ".json")
    try:
        rec, prow, urow = UP.build_cell(cell, hw)
        rec = dict(rec)
        rec.pop("build_seconds", None)
        bank = None
        if prow["status"] == "conditional_static_phases" and urow["status"] == "ok":
            try:
                bank = dict(kernels=[BU.analyse_kernel(k["kid"], k, fk) for k, fk in zip(cell["kernels"], prow["kernels"])])
            except (C.Refusal, P.Refusal) as ex:
                bank = dict(status="refused", reason=str(ex))
        else:
            bank = dict(status="refused", reason="no phase or first-touch table for this cell (%s / %s)" % (prow["status"], urow["status"]))
        res = dict(cell=cell, features=rec, phases=prow, unique=urow, bank=bank)
        msg = "%s %s %s %s" % (rec["status"], prow["status"], urow["status"], "bank ok" if "kernels" in bank else "bank refused")
    except (C.Refusal, P.Refusal) as ex:
        res = dict(cell=cell, refused=str(ex))
        msg = "REFUSED " + str(ex)[:200]
    BUILD.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return cell_id, msg


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def merge(cell_ids):
    PX, BU, MC = _setup()
    hw, quoted = MC.read_h100_hardware()
    feats, phases, uniq, bank, support = [], {}, {}, {}, {}
    for cid in cell_ids:
        p = BUILD / (cid.replace("/", "__") + ".json")
        if not p.exists():
            raise SystemExit("missing build file for %s; run without --merge-only" % cid)
        d = json.loads(p.read_text(encoding="utf-8"))
        if "refused" in d:
            reason = d["refused"]
            feats.append(dict(cell_id=cid, status="missing_features", missing_features=[dict(feature="static pipeline", reason=reason)]))
            phases[cid] = dict(status="unsupported", reason=reason, kernels=[])
            uniq[cid] = dict(status="unsupported", reason=reason, kernels=[])
            bank[cid] = dict(status="refused", reason=reason)
            support[cid] = dict(supported=False, reason=reason)
            continue
        feats.append(d["features"]); phases[cid] = d["phases"]; uniq[cid] = d["unique"]; bank[cid] = d["bank"]
        ok = (d["features"]["status"] in SUPPORTED and d["phases"]["status"] == "conditional_static_phases" and d["unique"]["status"] == "ok" and "kernels" in d["bank"])
        why = [] if ok else ["features %s%s" % (d["features"]["status"], (": " + "; ".join(str(m.get("reason", m))[:120] for m in d["features"].get("missing_features", [])[:2])) if d["features"].get("missing_features") else ""),
                               "phases %s %s" % (d["phases"]["status"], d["phases"].get("reason") or ""), "first-touch %s %s" % (d["unique"]["status"], d["unique"].get("reason") or ""),
                               "bank %s" % (d["bank"].get("reason") or "ok")]
        support[cid] = dict(supported=ok, status=d["features"]["status"], reason=None if ok else " | ".join(why))
    feats.sort(key=lambda r: r["cell_id"])
    w = lambda name, obj: (HERE / name).write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    w("features_h100.json", dict(schema="static_runtime_features_h100/1", hardware_from_ground_truth=hw, rows=feats, toolchain="CUDA 12.1.66 builds of the pinned cuda-samples (port_h100/compiled_cluster_cuda12.1/)",
                                 note="Label-free static features of the H100 evaluation cells. No runtime, energy or power value was read."))
    w("phases_h100.json", dict(schema="static_dynamic_barrier_phases/1", rows=phases))
    w("phases_unique_h100.json", dict(schema="first_touch_unique_sectors/1", rows=uniq))
    w("bank_conflicts_h100.json", dict(set="h100_eval", schema="bank_conflicts/1", note="shared-memory bank-conflict tables with per-phase request-wavefront histograms", rows=bank))
    n_ok = sum(1 for v in support.values() if v["supported"])
    w("static_support_h100.json", dict(schema="h100_static_support/1", cells=len(support), supported=n_ok, unsupported=sorted(c for c, v in support.items() if not v["supported"]), rows=support,
                                       note="Unsupported cells stay in the list and are counted as failures by the scorer; none is dropped."))
    print("merged: %d cells, %d supported, %d unsupported" % (len(support), n_ok, len(support) - n_ok))
    for c, v in support.items():
        if not v["supported"]:
            print("  UNSUPPORTED %s: %s" % (c, (v["reason"] or "")[:200]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true", help="rebuild cells that already have a cached file")
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args(argv)
    doc = json.loads((HERE / "cells_h100.json").read_text(encoding="utf-8"))
    ids = [c["cell_id"] for c in doc["cells"]]
    if not a.merge_only:
        todo = [c for c in ids if (not a.only or a.only in c) and (a.force or not (BUILD / (c.replace("/", "__") + ".json")).exists())]
        print("%d cells to build" % len(todo), flush=True)
        if a.jobs > 1 and len(todo) > 1:
            with cf.ProcessPoolExecutor(max_workers=a.jobs) as pool:
                for cid, msg in pool.map(build_one, todo):
                    print("%-72s %s" % (cid, msg), flush=True)
        else:
            for cid in todo:
                print("%-72s %s" % build_one(cid), flush=True)
    if a.only and not a.merge_only:
        return 0
    merge(ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
