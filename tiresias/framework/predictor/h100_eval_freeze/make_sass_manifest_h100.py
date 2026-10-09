#!/usr/bin/env python3
"""SASS identity manifest of the H100 evaluation kernels (CPU only).

For every kernel launched by a cell of cells_h100.json: the mangled symbol and the hash of the normalised instruction sequence (pc, predicate, opcode, operands; encoding
lines ignored) of the SASS the static analysis used (port_h100/compiled_cluster_cuda12.1/sass/, isolated per function by the pipeline). The measurement job dumps each
symbol from the binary it built on Cluster (`cuobjdump -sass -fun <symbol> -arch sm_90`) and refuses to time a cell whose kernel hashes differently: the kernel that is measured
must be the kernel that was analysed.

    python make_sass_manifest_h100.py            # writes sass_manifest_h100.json
    python make_sass_manifest_h100.py --check    # exit 1 unless the committed file equals the regenerated one
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = HERE.parent
sys.path.insert(0, str(SR / "port_common"))
sys.path.insert(0, str(HERE))
OUT = HERE / "sass_manifest_h100.json"
SASS_ROOT = SR / "port_h100" / "compiled_cluster_cuda12.1"

LINE = re.compile(r"\s*/\*([0-9a-f]+)\*/\s+(?:@(!?(?:U?P\d+|U?PT))\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;")


def parse_sass(text):
    """Instruction sequence of a cuobjdump -sass listing: (pc, predicate, opcode, operands). Same grammar as fresh_d/timing_fresh_d.py."""
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


def sass_hash(text) -> str:
    return hashlib.sha256(repr(parse_sass(text)).encode()).hexdigest()


def build_manifest():
    import port_ext as PX
    import set_d_port as SD
    import make_cells_h100 as MC
    PX.install("sm_90", SASS_ROOT)
    SD.register(blackwell=False)
    UP = PX.UP
    doc = json.loads((HERE / "cells_h100.json").read_text(encoding="utf-8"))
    kids = sorted({k["kid"] for c in doc["cells"] for k in c["kernels"]})
    rows = {}
    for kid in kids:
        a = UP.kernel_assets(kid)
        rows[kid] = dict(symbol=a["idx"]["mangled"], instruction_sequence_sha256=sass_hash(a["text"]), instructions=len(parse_sass(a["text"])),
                         source_sass_file=UP.KERNELS[kid]["sass"] + ".sass")
    cubins = {}
    for line in (SASS_ROOT / "sha256.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].endswith(".cubin"):
            cubins[Path(parts[1]).name] = parts[0]
    return dict(schema="h100_sass_manifest/1", arch="sm_90", toolchain="CUDA 12.1.66 (port_h100/compiled_cluster_cuda12.1/build.txt)", kernels=rows, analysed_cubin_sha256=cubins,
                note="The job compares sha256 of the normalised instruction sequence of each symbol dumped from its own binary; a mismatch refuses the cells of that kernel.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = json.dumps(build_manifest(), indent=1, sort_keys=True) + "\n"
    if a.check:
        if not OUT.exists() or OUT.read_bytes() != text.encode("utf-8"):
            print("sass_manifest_h100.json differs from the regenerated manifest", file=sys.stderr)
            return 1
        print("sass_manifest_h100.json reproduced")
        return 0
    OUT.write_bytes(text.encode("utf-8"))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
