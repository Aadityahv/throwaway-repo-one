#!/usr/bin/env python3
"""SASS identity manifest of the Ada evaluation kernels (CPU only): for every kernel of every cell, the mangled symbol and the hash of the normalised instruction sequence of the SASS the static analysis
used, so that a measurement job can dump each symbol from the binary it built and refuse to time a cell whose kernel differs (as h100_eval_freeze/make_sass_manifest_h100.py).

    python make_sass_manifest_ada.py            # writes sass_manifest_ada.json (kernels whose SASS exists; the others listed as pending_build)
    python make_sass_manifest_ada.py --check    # exit 1 unless the committed file equals the regenerated one
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import board_ada as B
import make_cells_ada as MA
import static_ada as SA

sys.path.insert(0, str(B.FZ))
import make_sass_manifest_h100 as MH  # noqa: E402  (parse_sass, sass_hash reused unchanged)

OUT = B.HERE / "sass_manifest_ada.json"


def build_manifest():
    state = SA.sass_state()
    by_origin = {}
    for c in MA.load_cells():
        by_origin.setdefault(SA.origin_of(c), {}).update({k["kid"]: 1 for k in c["kernels"]})
    rows, pending = {}, {}
    for origin, kids in sorted(by_origin.items()):
        if not state[origin][0]:
            for kid in sorted(kids):
                pending[kid] = dict(origin=origin, reason="SASS missing: %s" % ", ".join(Path(f).name for f in state[origin][1]))
            continue
        UP, _ = SA._setup(origin)
        for kid in sorted(kids):
            a = UP.kernel_assets(kid)
            rows[kid] = dict(symbol=a["idx"]["mangled"], instruction_sequence_sha256=MH.sass_hash(a["text"]), instructions=len(MH.parse_sass(a["text"])), source_sass_file=UP.KERNELS[kid]["sass"] + ".sass", origin=origin)
    return dict(schema="ada_sass_manifest/1", arch=B.ARCH, toolchain="Ada's own CUDA 13.2 (port_ada/compiled_ada_cuda13.2 and compiled_ada_cuda13.2_eval)", kernels=rows, pending_build=pending,
                note="The job compares sha256 of the normalised instruction sequence of each symbol dumped from its own binary; a mismatch refuses the cells of that kernel. The freeze refuses while pending_build is non-empty.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = json.dumps(build_manifest(), indent=1, sort_keys=True) + "\n"
    if a.check:
        if not OUT.exists() or OUT.read_bytes() != text.encode("utf-8"):
            print("sass_manifest_ada.json differs from the regenerated manifest", file=sys.stderr)
            return 1
        print("sass_manifest_ada.json reproduced")
        return 0
    OUT.write_bytes(text.encode("utf-8"))
    print("wrote", OUT.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
