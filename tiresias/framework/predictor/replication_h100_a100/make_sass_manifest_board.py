#!/usr/bin/env python3
"""SASS identity manifest of the replication kernels on one Cluster board (CPU only): for every kernel of every intended cell, the mangled symbol and the hash of the normalised instruction
sequence of the SASS the static analysis used. A measurement job dumps each symbol from the binary it built and refuses to time a cell whose kernel differs.

    python make_sass_manifest_board.py --board h100            # writes sass_manifest_replication_h100.json (kernels whose SASS exists; the others under pending_build)
    python make_sass_manifest_board.py --board h100 --check    # exit 1 unless the file equals the regenerated one

Reuses the static analyser's own asset loading (static_tables.setup -> static_ada.sass_state/_setup) so the manifest and the analysis read the same files, and the unchanged parse/hash of
h100_eval_freeze/make_sass_manifest_h100.py. The freeze refuses while pending_build is non-empty.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import board as BD
import cell_sets_board as CS
import static_tables as ST


def build_manifest(board):
    mod = ST.setup(board)
    sys.path.insert(0, str(mod.B.FZ))
    import make_sass_manifest_h100 as MH
    state = mod.sass_state()
    by_origin = {}
    for c in CS.load_cells(board, "main") + CS.load_cells(board, "tensor"):
        by_origin.setdefault(mod.origin_of(c), {}).update({k["kid"]: 1 for k in c["kernels"]})
    rows, pending = {}, {}
    for origin, kids in sorted(by_origin.items()):
        if not state[origin][0]:
            for kid in sorted(kids):
                pending[kid] = dict(origin=origin, reason="SASS missing: %s" % ", ".join(Path(f).name for f in state[origin][1]))
            continue
        UP, _ = mod._setup(origin)
        for kid in sorted(kids):
            a = UP.kernel_assets(kid)
            rows[kid] = dict(symbol=a["idx"]["mangled"], instruction_sequence_sha256=MH.sass_hash(a["text"]), instructions=len(MH.parse_sass(a["text"])),
                             source_sass_file=UP.KERNELS[kid]["sass"] + ".sass", origin=origin)
    return dict(schema="cluster_sass_manifest/1", board=board, arch=mod.B.ARCH,
                toolchain="Cluster CUDA 12.1 (%s and %s)" % (mod.B.SASS_BASE.name, mod.B.SASS_EVAL.name), kernels=rows, pending_build=pending,
                note="The job compares sha256 of the normalised instruction sequence of each symbol dumped from its own binary; a mismatch refuses the cells of that kernel. The freeze refuses while pending_build is non-empty.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", required=True, choices=sorted(BD.CONFIG))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    out = CS.board_dir(a.board) / CS.names(a.board, "main")["sass_manifest"]
    text = json.dumps(build_manifest(a.board), indent=1, sort_keys=True) + "\n"
    if a.check:
        if not out.exists() or out.read_bytes() != text.encode("utf-8"):
            print("%s differs from the regenerated manifest" % out.name, file=sys.stderr)
            return 1
        print("%s reproduced" % out.name)
        return 0
    out.write_bytes(text.encode("utf-8"))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
