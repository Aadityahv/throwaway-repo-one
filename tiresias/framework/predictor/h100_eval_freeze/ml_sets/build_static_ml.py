#!/usr/bin/env python3
"""Static tables of the machine-learning kernel cells and of the tensor-core matrix multiply and fused attention cells, through the cross-GPU port (CPU only; reads no measured value).

For each cell of cells_ml_<arch>.json (or cells_tensor_<arch>.json): the static pipeline (port_common/port_ml_ext.py: port_ext.py plus the machine-learning extensions, installed for sm_90 or sm_80 on
the CUDA 12.1 build of the combined driver, port_<arch>/compiled_cluster_cuda12.1_ml/) builds the feature row (instruction-family counts incl. the tensor_core family, memory traffic, resources and
occupancy from the board's rows of HARDWARE_GROUND_TRUTH.md), the barrier-phase table, the first-touch table and the shared-memory bank-conflict table (with request-wavefront histograms). A cell
the static side refuses is NOT dropped: it is kept with its exact reason (the opcode or form the port cannot interpret) and is a failure in every score. The tensor cells' static counts need no
tensor constant; only their predictions do (predict_h100.py --profile tensor).

    python build_static_ml.py --arch h100 --set ml [--jobs 4] [--only <substring>] [--force]
    python build_static_ml.py --arch h100 --set tensor
    python build_static_ml.py --arch h100 --set ml --merge-only        # merge the cached per-cell files (deterministic)

Outputs in this directory: features_<set>_<arch>.json, phases_<set>_<arch>.json, phases_unique_<set>_<arch>.json, bank_conflicts_<set>_<arch>.json, static_support_<set>_<arch>.json
(no timestamps). Per-cell cache: build/<arch>_<set>/ (git-ignored).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FZ = HERE.parent
SR = FZ.parent
sys.path.insert(0, str(SR / "port_common"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FZ))

ARCH_SM = {"h100": "sm_90", "a100": "sm_80"}
SUPPORTED = ("supported", "supported_with_assumptions")
SETS = ("ml", "tensor")


def sass_root(arch):
    return SR / ("port_%s" % arch) / "compiled_cluster_cuda12.1_ml"


def _setup(arch):
    import port_ml_ext as PM
    PM.install(ARCH_SM[arch], sass_root(arch))
    sys.path.insert(0, str(SR / "bank"))
    import bank_unseen as BU
    import make_cells_ml as MM
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
    return PM, BU, MM


def cells_file(arch, which):
    return HERE / ("cells_%s_%s.json" % (which, arch))


def build_dir(arch, which):
    return HERE / "build" / ("%s_%s" % (arch, which))


def out_name(kind, which, arch):
    return HERE / ("%s_%s_%s.json" % (kind, which, arch))


def build_one(args):
    arch, which, cell_id = args
    PM, BU, MM = _setup(arch)
    UP = PM.UP
    C, P = UP.C, UP.P
    doc = json.loads(cells_file(arch, which).read_text(encoding="utf-8"))
    cell = next(c for c in doc["cells"] if c["cell_id"] == cell_id)
    hw = MM.read_hardware(arch)
    out = build_dir(arch, which) / (cell_id.replace("/", "__") + ".json")
    try:
        rec, prow, urow = UP.build_cell(cell, hw)
        rec = dict(rec)
        rec.pop("build_seconds", None)
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
    except SystemExit as ex:                                # the pipeline's own hard stops (an unknown opcode is raised as SystemExit in places)
        res = dict(cell=cell, refused="static pipeline stopped: %s" % ex)
        msg = "REFUSED " + str(ex)[:200]
    build_dir(arch, which).mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return cell_id, msg


def merge(arch, which, cell_ids):
    PM, BU, MM = _setup(arch)
    hw = MM.read_hardware(arch)
    feats, phases, uniq, bank, support = [], {}, {}, {}, {}
    cells_by_id = {c["cell_id"]: c for c in json.loads(cells_file(arch, which).read_text(encoding="utf-8"))["cells"]}
    for cid in cell_ids:
        p = build_dir(arch, which) / (cid.replace("/", "__") + ".json")
        if not p.exists():
            raise SystemExit("missing build file for %s; run without --merge-only" % cid)
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("cell") != json.loads(json.dumps(cells_by_id[cid], sort_keys=True, default=str)):
            raise SystemExit("the cached build of %s was made from a different cell than the cell file holds: rebuild it (run without --merge-only)" % cid)
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
        why = [] if ok else ["features %s%s" % (d["features"]["status"], (": " + "; ".join(str(m.get("reason", m))[:160] for m in d["features"].get("missing_features", [])[:2])) if d["features"].get("missing_features") else ""),
                               "phases %s %s" % (d["phases"]["status"], d["phases"].get("reason") or ""), "first-touch %s %s" % (d["unique"]["status"], d["unique"].get("reason") or ""),
                               "bank %s" % (d["bank"].get("reason") or "ok")]
        support[cid] = dict(supported=ok, status=d["features"]["status"], reason=None if ok else " | ".join(why))
    feats.sort(key=lambda r: r["cell_id"])
    w = lambda kind, obj: out_name(kind, which, arch).write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    w("features", dict(schema="static_runtime_features_ml/1", arch=arch, hardware_from_ground_truth=hw, rows=feats,
                       toolchain="CUDA 12.1.66 build of the combined machine-learning, tensor-core and attention driver (port_%s/compiled_cluster_cuda12.1_ml/)" % arch,
                       note="Label-free static features. No runtime, energy or power value was read."))
    w("phases", dict(schema="static_dynamic_barrier_phases/1", rows=phases))
    w("phases_unique", dict(schema="first_touch_unique_sectors/1", rows=uniq))
    w("bank_conflicts", dict(set="ml_eval_%s" % which, schema="bank_conflicts/1", note="shared-memory bank-conflict tables with per-phase request-wavefront histograms", rows=bank))
    n_ok = sum(1 for v in support.values() if v["supported"])
    w("static_support", dict(schema="ml_static_support/1", arch=arch, cells=len(support), supported=n_ok, unsupported=sorted(c for c, v in support.items() if not v["supported"]), rows=support,
                             note="Unsupported cells stay in the list and are counted as failures by the scorer; none is dropped."))
    print("merged %s/%s: %d cells, %d supported, %d unsupported" % (arch, which, len(support), n_ok, len(support) - n_ok))
    for c, v in support.items():
        if not v["supported"]:
            print("  UNSUPPORTED %s: %s" % (c, (v["reason"] or "")[:300]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=sorted(ARCH_SM), required=True)
    ap.add_argument("--set", dest="which", choices=SETS, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args(argv)
    doc = json.loads(cells_file(a.arch, a.which).read_text(encoding="utf-8"))
    ids = [c["cell_id"] for c in doc["cells"]]
    by_id = {c["cell_id"]: c for c in doc["cells"]}

    def cached_current(cid):
        """A cached per-cell file counts only if it was built from exactly the cell the cell file holds now (a changed shape is rebuilt)."""
        p = build_dir(a.arch, a.which) / (cid.replace("/", "__") + ".json")
        if not p.exists():
            return False
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("cell") == json.loads(json.dumps(by_id[cid], sort_keys=True, default=str))
        except ValueError:
            return False
    if not a.merge_only:
        todo = [c for c in ids if (not a.only or a.only in c) and (a.force or not cached_current(c))]
        print("%d cells to build" % len(todo), flush=True)
        jobs = [(a.arch, a.which, c) for c in todo]
        if a.jobs > 1 and len(jobs) > 1:
            with cf.ProcessPoolExecutor(max_workers=a.jobs) as pool:
                for cid, msg in pool.map(build_one, jobs):
                    print("%-72s %s" % (cid, msg), flush=True)
        else:
            for j in jobs:
                print("%-72s %s" % build_one(j), flush=True)
    if a.only and not a.merge_only:
        return 0
    merge(a.arch, a.which, ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
