#!/usr/bin/env python3
"""SASS identity manifests of the machine-learning kernels (sass_manifest_ml_<arch>.json: 10 kernels) and of the tensor-core matrix multiply and fused attention kernels
(sass_manifest_tensor_<arch>.json: 4 kernels), from the committed CUDA 12.1 SASS dump of the combined driver (port_<arch>/compiled_cluster_cuda12.1_ml/driver_ml.sass). CPU only.

Same contract as make_sass_manifest_h100.py: per kernel, the mangled symbol and the sha256 of the normalised instruction sequence of the SASS the static analysis used. The measurement job builds
the combined driver on the node with the same command and CUDA 12.1, dumps each symbol (`cuobjdump -sass -fun <symbol> -arch sm_90 <driver_ml>`) and refuses the cells of any kernel whose
sequence differs. The manifest also records the sha256 of the committed dump the sequences were taken from.

    python make_sass_manifest_ml.py --arch h100            # writes both manifests
    python make_sass_manifest_ml.py --arch h100 --check    # exit 1 unless both equal the regenerated ones
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FZ = HERE.parent
SR = FZ.parent
sys.path.insert(0, str(FZ))
import make_sass_manifest_h100 as MS  # noqa: E402

# kernel id -> (needle in the mangled symbol, set)
KERNELS = {
    "gelu_s": ("gelu_s", "ml"), "gelu_v4": ("gelu_v4", "ml"), "swiglu_s": ("swiglu_s", "ml"), "swiglu_v4": ("swiglu_v4", "ml"), "rms_s": ("rmsnorm_sILi256E", "ml"),
    "rms_v4": ("rmsnorm_v4ILi128E", "ml"), "rope_all": ("rope_allILi8E", "ml"), "rope_one": ("rope_oneILi8E", "ml"), "sg64": ("sgemmILi64E", "ml"), "sg128": ("sgemmILi128E", "ml"),
    "tc128": ("tc_gemmILi128E", "tensor"), "tc64": ("tc_gemmILi64E", "tensor"), "at4": ("attn_fwdILi4E", "tensor"), "at8": ("attn_fwdILi8E", "tensor"),
}
ARCH = {"h100": ("sm_90", "port_h100"), "a100": ("sm_80", "port_a100")}


def dump_path(arch):
    return SR / ARCH[arch][1] / "compiled_cluster_cuda12.1_ml" / "driver_ml.sass"


def functions(text):
    out = {}
    for f in re.split(r"\n\s*Function : ", text)[1:]:
        out[f.split("\n")[0].strip()] = "Function : " + f
    return out


def build(arch):
    p = dump_path(arch)
    text = p.read_text(encoding="utf-8", errors="replace")
    funcs = functions(text)
    docs = {}
    for which in ("ml", "tensor"):
        rows = {}
        for kid, (needle, w) in sorted(KERNELS.items()):
            if w != which:
                continue
            hits = [s for s in funcs if needle in s]
            if len(hits) != 1:
                raise SystemExit("kernel %s (%s) matched %d functions in %s" % (kid, needle, len(hits), p))
            body = funcs[hits[0]]
            rows[kid] = dict(symbol=hits[0], instruction_sequence_sha256=MS.sass_hash(body), instructions=len(MS.parse_sass(body)), source_sass_file="driver_ml.sass")
        docs[which] = dict(schema="ml_sass_manifest/1", arch=ARCH[arch][0], toolchain="CUDA 12.1.66 (port_%s/compiled_cluster_cuda12.1_ml/build.txt)" % arch, kernels=rows,
                           analysed_sass_dump_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                           note="The job compares sha256 of the normalised instruction sequence of each symbol dumped from its own binary; a mismatch refuses the cells of that kernel.")
    return docs


def out_path(which, arch):
    return HERE / ("sass_manifest_%s_%s.json" % (which, arch))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=sorted(ARCH), required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    docs = build(a.arch)
    bad = []
    for which, d in docs.items():
        text = (json.dumps(d, indent=1, sort_keys=True) + "\n").encode("utf-8")
        p = out_path(which, a.arch)
        if a.check:
            if not p.exists() or p.read_bytes() != text:
                bad.append(p.name)
        else:
            p.write_bytes(text)
            print("wrote", p, len(d["kernels"]), "kernels")
    if bad:
        print("differs from the regenerated manifest: " + ", ".join(bad), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
