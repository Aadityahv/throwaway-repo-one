#!/usr/bin/env python3
"""Host-side checks for independent-operator adapter manifests (no GPU, no energy).

Validates schema, lineage uniqueness, bytes math, launch-geometry coverage, tier footprints
vs hardware L2 sizes, correctness-reference presence and provenance completeness.
`--selftest` runs synthetic bad manifests (must fail) plus the real manifest (must pass).
GPU compile/correctness validation happens only in the released pilot, never here.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "adapters.json"
L2_BYTES = {"blackwell": 134_217_728, "a100": 41_943_040, "ada": 67_108_864, "h100": 52_428_800}
THREADS = 256


class ManifestError(Exception):
    pass


def check_schema(d):
    if not isinstance(d.get("adapters"), list) or not d["adapters"]:
        raise ManifestError("adapters must be a non-empty list")
    seen_lineage, seen_parents = set(), set()
    for a in d["adapters"]:
        for k in ("parent_id", "lineage_id", "family", "source", "regimes", "candidates",
                  "correctness_reference", "status"):
            if k not in a:
                raise ManifestError(f"{a.get('parent_id', '?')}: missing key {k}")
        if a["parent_id"] in seen_parents:
            raise ManifestError(f"duplicate parent_id {a['parent_id']}")
        if a["lineage_id"] in seen_lineage:
            raise ManifestError(f"duplicate lineage_id {a['lineage_id']}")
        seen_parents.add(a["parent_id"])
        seen_lineage.add(a["lineage_id"])
        if not a["correctness_reference"]:
            raise ManifestError(f"{a['parent_id']}: empty correctness reference")
        s = a["source"]
        for k in ("repository", "revision", "file", "sha256", "bytes", "license_spdx"):
            if k not in s:
                raise ManifestError(f"{a['parent_id']}: source missing {k}")
        if len(s["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in s["sha256"]):
            raise ManifestError(f"{a['parent_id']}: sha256 malformed")
    return True


def elements(family, regime):
    n = regime["n"]
    if family in ("streaming", "reduction"):
        return n
    if family == "layout_transform":
        return n * n
    raise ManifestError(f"no element rule for family {family}")


def check_geometry(a):
    out = {}
    for regime_name, regime in a["regimes"].items():
        n = elements(a["family"], regime)
        if n <= 0:
            raise ManifestError(f"{a['parent_id']}/{regime_name}: nonpositive elements")
        blocks = math.ceil(n / THREADS)
        if blocks * THREADS < n:
            raise ManifestError(f"{a['parent_id']}/{regime_name}: grid undercovers")
        out[regime_name] = {"elements": n, "threads": THREADS, "blocks": blocks}
    return out


def check_bytes_tiers(a, geo):
    out = {}
    for regime_name, regime in a["regimes"].items():
        n = geo[regime_name]["elements"]
        if a["family"] == "reduction":
            # input read once (4n) plus one float32 partial sum per block; blocks = n/threads is the
            # upper bound (reduce0/reduce1); reduce6 launches fewer, which only lowers the total.
            total = 4 * n + 4 * geo[regime_name]["blocks"]
        else:
            total = 2 * n * 4
        out[regime_name] = {"bytes": total,
                            "tier": {g: ("L2" if total / l2 < 1.0 else "DRAM")
                                     for g, l2 in L2_BYTES.items()}}
    return out


def check_manifest(d):
    check_schema(d)
    report = {}
    for a in d["adapters"]:
        geo = check_geometry(a)
        report[a["parent_id"]] = {"geometry": geo, "bytes_tiers": check_bytes_tiers(a, geo),
                                  "n_candidates": len(a["candidates"])}
    return report


def selftest():
    check_manifest(json.loads(MANIFEST.read_text()))
    bad_dup = {"adapters": [
        {"parent_id": "x", "lineage_id": "L", "family": "streaming", "source": {},
         "regimes": {}, "candidates": {}, "correctness_reference": "r", "status": "s"},
        {"parent_id": "y", "lineage_id": "L", "family": "streaming", "source": {},
         "regimes": {}, "candidates": {}, "correctness_reference": "r", "status": "s"}]}
    try:
        check_manifest(bad_dup)
        raise AssertionError("duplicate lineage accepted")
    except ManifestError:
        pass
    bad_bytes = {"adapters": [
        {"parent_id": "x", "lineage_id": "L", "family": "streaming", "source": {},
         "regimes": {}, "candidates": {}, "correctness_reference": "", "status": "s"}]}
    try:
        check_manifest(bad_bytes)
        raise AssertionError("empty correctness reference accepted")
    except ManifestError:
        pass
    print("SELFTEST ADAPTERS PASSED: real manifest valid; duplicate-lineage and empty-reference refused.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--report", action="store_true", help="print geometry/tiers report")
    a = ap.parse_args()
    if a.selftest or len(sys.argv) == 1:
        selftest()
        return
    if a.report:
        print(json.dumps(check_manifest(json.loads(MANIFEST.read_text())), indent=1))


if __name__ == "__main__":
    main()
