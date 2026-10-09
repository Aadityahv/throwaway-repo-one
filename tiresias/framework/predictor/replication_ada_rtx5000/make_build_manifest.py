#!/usr/bin/env python3
"""Build manifest of the Ada evaluation (CPU only; nothing is compiled or run here).

    python make_build_manifest.py            # writes build_manifest_ada.json and BUILD_MANIFEST.md
    python make_build_manifest.py --check    # exit 1 unless both files equal the regenerated ones

Every command is the Blackwell command, taken from the Blackwell harness code itself (the build functions of the committed runners are executed against a recording stand-in for subprocess, so the text
is theirs, not a re-typed copy), with exactly two substitutions: the architecture (-arch=sm_89 instead of sm_120) and Ada's CUDA 13.2 nvcc (/usr/local/cuda-13.2/bin/nvcc, HARDWARE_GROUND_TRUTH.md).
Paths: <REPO> the repository checkout on the Ada host, <SAMPLES> the pinned cuda-samples checkout (revision 5443602d), <WORK> a scratch directory.
Execution on the Ada host (compile only, no GPU): build_ada_eval_sass.py reads build_manifest_ada.json, runs each build, dumps `cuobjdump -sass` and `-res-usage` and writes the files named under `outputs`.
Version note: Ada's nvcc is CUDA 13.2.86; the Blackwell runners (run_f.py, run_prosp.py, timing_fresh_d.py) refuse any nvcc that is not 13.2.78. build_ada_eval_sass.py does not call those guards (it records the
version); an Ada measurement harness needs the guard made board-aware.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import board_ada as B

HERE = B.HERE
SR = B.SR
REPO = B.REPO
OUT_JSON = HERE / "build_manifest_ada.json"
OUT_MD = HERE / "BUILD_MANIFEST.md"
SAMPLES_SENTINEL = "C:/__SAMPLES__"
NVCC = B.CUDA_NVCC
SCHEMA = "ada_build_manifest/1"


class _Done:
    returncode = 0
    stderr = ""

    def __init__(self, stdout=""):
        self.stdout = stdout


def _norm(arg, work_map):
    s = str(arg).replace("\\", "/")
    for real, ph in work_map:
        s = s.replace(real.replace("\\", "/"), ph)
    s = s.replace(str(REPO).replace("\\", "/"), "<REPO>").replace(SAMPLES_SENTINEL, "<SAMPLES>")
    return s


class Recorder:
    def __init__(self, work_map):
        self.cmds = []
        self.work_map = work_map

    def run(self, cmd, **kw):
        if len(cmd) == 2 and cmd[1] == "--version":
            return _Done("Cuda compilation tools, release 13.2, V13.2.78")   # satisfies the Blackwell guards while the command is being recorded
        self.cmds.append([_norm(a, self.work_map) for a in cmd])
        return _Done()


def _record(fn, work_map):
    rec = Recorder(work_map)
    real = subprocess.run
    subprocess.run = rec.run
    try:
        fn()
    finally:
        subprocess.run = real
    return rec.cmds


def _add_paths(*rels):
    for r in rels:
        p = str(REPO / r)
        if p not in sys.path:
            sys.path.insert(0, p)


def entries():
    os.environ["HARNESS_CUDA_ARCH"] = B.ARCH
    os.environ["HARNESS_NVCC"] = NVCC
    ST = "tiresias/framework/predictor"
    _add_paths("tiresias/app_runners", "tiresias/framework/unseen_operators/runners", "energy_harness", ST + "/fresh_e/gpu", ST + "/unseen_kernels/gpu", ST + "/fresh_f", ST + "/prospective_test")
    out = []
    samples = Path(SAMPLES_SENTINEL)

    # ---- combined machine-learning driver (sets F, G, H and the validation F, G, H cells) and prospective driver: the commands of run_f.py / run_prosp.py cmd_build
    import importlib
    for name, mod, stem, kernels, covers in (
            ("driver_ml", "run_f", "driver_ml", ["gelu_s", "gelu_v4", "swiglu_s", "swiglu_v4", "rms_s", "rms_v4", "rope_all", "rope_one", "sg64", "sg128", "tc128", "tc64", "at4", "at8"],
             "cells_ada_ml.json (40), cells_ada_tensor.json (16), 8 validation cells (FP32 matrix multiply, RMSNorm, tensor-core matrix multiply, fused attention)"),
            ("driver_prosp", "run_prosp", "driver_prosp", ["tc128x64", "tc256x128", "at2", "at16", "nm4", "nm8", "tcr128", "tcr64"], "cells_ada_prospective.json (24)")):
        m = importlib.import_module(mod)
        m.ARCH = B.ARCH
        m.find_nvcc = lambda: NVCC
        wd = Path(tempfile.mkdtemp(prefix="ada_manifest_"))
        (wd / stem).write_bytes(b"recording stand-in")
        m.sha256_file = lambda p, _o=m.sha256_file: _o(p) if Path(p).exists() and "recording" not in str(p) else "0" * 64
        cmds = _record(lambda: m.cmd_build(types.SimpleNamespace(workdir=str(wd))), [(str(wd), "<WORK>/" + name)])
        assert len(cmds) == 1, (name, cmds)
        out.append(dict(id=name, role="static_and_measurement_driver", source_of_command="%s.cmd_build" % mod, covers=covers, kernels=kernels, steps=[dict(cmd=cmds[0], cwd="<WORK>/%s" % name)],
                        binary="<WORK>/%s/%s" % (name, stem),
                        outputs=dict(sass="%s.sass" % stem, res="%s.res" % stem, dump_sass="cuobjdump -sass -arch sm_89 <binary>", dump_res="cuobjdump -res-usage -arch sm_89 <binary>",
                                     deposit="port_ada/compiled_ada_cuda13.2_eval/ (flat folder; the static side reads exactly these two files)"),
                        static_input=True, ada_sass_exists=False))

    # ---- measurement drivers of the CUDA-samples sets (static SASS of these kernels already exists in port_ada/compiled_ada_cuda13.2/; the driver binaries are needed for the SASS identity gate)
    for mod, label in (("run_e", "set E driver (scalar product, fast Walsh transform; also the validation scalar-product cells)"), ("run_unseen", "classic-kernel drivers (matrix multiply, Black-Scholes, scan, separable convolution; also the validation convolution cells)")):
        m = importlib.import_module(mod)
        m.ARCH = B.ARCH
        for fam in sorted(m.FAMILIES):
            cmd = m.build_command(NVCC, Path(SAMPLES_SENTINEL), fam, Path("C:/__WORK__/driver_%s" % fam), True)
            cmd = [_norm(a, [("C:/__WORK__", "<WORK>/%s_%s" % (mod, fam))]) for a in cmd]
            out.append(dict(id="%s_%s" % (mod, fam), role="measurement_driver", source_of_command="%s.build_command(family=%r)" % (mod, fam), covers=label, kernels=[fam], steps=[dict(cmd=cmd, cwd="<REPO>")],
                            binary="<WORK>/%s_%s/driver_%s" % (mod, fam, fam),
                            outputs=dict(sass="meas/%s_%s.sass" % (mod, fam), res="meas/%s_%s.res" % (mod, fam), dump_sass="cuobjdump -sass -arch sm_89 <binary>", dump_res="cuobjdump -res-usage -arch sm_89 <binary>",
                                         deposit="port_ada/compiled_ada_cuda13.2_eval/meas/"),
                            static_input=False, ada_sass_exists=True))

    # ---- set D harness binaries (build_binary of the committed runners; they write their driver source first, from their own template)
    for name, label in (("copy_runner", "vector add (set D)"), ("transpose_runner", "transposes (set D)"), ("reduction_runner", "reductions (set D)"), ("tile_family", "copies and fine/coarse transposes (set D)")):
        m = importlib.import_module(name)
        wd = Path(tempfile.mkdtemp(prefix="ada_manifest_"))
        for n in ("copy_harness", "transpose_harness", "reduction_harness", "tile_family_harness"):
            (wd / n).write_text("x")
        import pathlib
        orig = pathlib.Path.is_dir
        pathlib.Path.is_dir = lambda s: True
        try:
            cmds = _record(lambda: m.build_binary(samples, wd, NVCC), [(str(wd), "<WORK>/" + name)])
        finally:
            pathlib.Path.is_dir = orig
        binary = {"copy_runner": "copy_harness", "transpose_runner": "transpose_harness", "reduction_runner": "reduction_harness", "tile_family": "tile_family_harness"}[name]
        out.append(dict(id=name, role="measurement_driver", source_of_command="%s.build_binary (HARNESS_CUDA_ARCH=sm_89)" % name, covers=label, kernels=[], steps=[dict(cmd=c, cwd="<WORK>/%s" % name) for c in cmds],
                        binary="<WORK>/%s/%s" % (name, binary), writes_driver_source="<WORK>/%s/*_driver.cu from the runner's DRIVER_TEMPLATE (build_ada_eval_sass.py calls build_binary itself)",
                        outputs=dict(sass="meas/%s.sass" % name, res="meas/%s.res" % name, dump_sass="cuobjdump -sass -arch sm_89 <binary>", dump_res="cuobjdump -res-usage -arch sm_89 <binary>",
                                     deposit="port_ada/compiled_ada_cuda13.2_eval/meas/"),
                        static_input=False, ada_sass_exists=True, python_entry="%s.build_binary(<SAMPLES>, <WORK>/%s, nvcc)" % (name, name)))
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_hashes():
    ST = SR
    files = ["fresh_f/gpu/drivers/driver_ml.cu", "fresh_f/gpu/drivers/driver_common.h", "fresh_f/src/ml_kernels.cuh", "fresh_g/src/tc_kernels.cuh", "fresh_h/src/attn_kernels.cuh",
             "prospective_test/gpu/driver_prosp.cu", "prospective_test/src/prosp_kernels.cuh", "fresh_e/gpu/drivers/driver_sp.cu", "fresh_e/gpu/drivers/driver_fwt.cu"]
    files += ["unseen_kernels/gpu/drivers/" + f for f in sorted(os.listdir(SR / "unseen_kernels/gpu/drivers")) if f.endswith(".cu")]
    return {f: sha(ST / f) for f in files}


def existing_sass():
    d = B.SASS_BASE
    return dict(folder="port_ada/compiled_ada_cuda13.2/", files=sorted(p.name for p in (d / "sass").glob("*.sass")), command_log="build.txt (`nvcc -std=c++17 -arch=sm_89 -cubin -Dmain=<sample>_main_disabled ...`, CUDA 13.2.86)",
                covers="static analysis of every kernel of the CUDA-samples sets: classic kernels, scalar product, fast Walsh transform, set D (incl. vector add), validation scalar product and convolution",
                sha256_of_log=sha(d / "build.txt"))


def build_doc():
    es = entries()
    return dict(schema=SCHEMA, board="ada", arch=B.ARCH, nvcc=NVCC, nvcc_note="HARDWARE_GROUND_TRUTH.md Ada section: CUDA 13.2 at /usr/local/cuda-13.2 (re-checked 2026-10-02); the version string of the build host is recorded by build_ada_eval_sass.py",
                substitutions=["-arch=sm_120 -> -arch=sm_89", "nvcc -> " + NVCC],
                source_sha256=source_hashes(), existing_ada_sass=existing_sass(), builds=es,
                eval_sass_folder="port_ada/compiled_ada_cuda13.2_eval/",
                static_pending=["driver_ml.sass", "driver_ml.res", "driver_prosp.sass", "driver_prosp.res"])


def markdown(doc):
    L = ["# Ada build manifest (sm_89, Ada's CUDA 13.2; compile only, no GPU)", "",
         "Generated by `make_build_manifest.py` from the Blackwell harness code; `build_manifest_ada.json` is the machine-readable form. Two substitutions only: `-arch=sm_89` and `%s`." % NVCC, "",
         "Placeholders: `<REPO>` repository checkout on the Ada host, `<SAMPLES>` pinned cuda-samples (revision 5443602d), `<WORK>` scratch directory. One command on the Ada host builds everything and dumps the SASS:", "",
         "```", "python3 <REPO>/tiresias/framework/predictor/replication_ada_rtx5000/build_ada_eval_sass.py --samples <SAMPLES> --work <WORK> --out <REPO>/tiresias/framework/predictor/port_ada/compiled_ada_cuda13.2_eval", "```", "",
         "(`--dry-run` prints the commands only.) It uses `nice -n 19`, runs no kernel and refuses a dirty or wrong-revision samples checkout. Then run `python static_ada.py` here: it builds the static tables of the pending cells.", "",
         "## What exists already", "", "Ada SASS in `%s` (%s): %s." % (doc["existing_ada_sass"]["folder"], doc["existing_ada_sass"]["command_log"], ", ".join(doc["existing_ada_sass"]["files"])), "It covers: " + doc["existing_ada_sass"]["covers"] + ".", "",
         "## What must be built (static side, blocks 88 cells: the 40 machine-learning, 16 tensor-core and attention and 24 prospective cells and 8 validation cells)", ""]
    for b in doc["builds"]:
        if b["static_input"]:
            L += ["### `%s` (covers %s)" % (b["id"], b["covers"]), "", "Source of the command: `%s`. Kernels: %s." % (b["source_of_command"], ", ".join(b["kernels"])), "", "```"]
            for s in b["steps"]:
                L.append(" ".join(s["cmd"]))
            L += ["```", "", "Outputs: binary `%s`; `%s` = `%s`; `%s` = `%s`; deposit in `%s`." % (b["binary"], b["outputs"]["sass"], b["outputs"]["dump_sass"], b["outputs"]["res"], b["outputs"]["dump_res"], b["outputs"]["deposit"]), ""]
    L += ["## Measurement drivers (needed for the SASS identity gate and the later timing; their kernels already have static SASS)", ""]
    for b in doc["builds"]:
        if not b["static_input"]:
            L += ["### `%s` (%s)" % (b["id"], b["covers"]), "", "Source of the command: `%s`." % b["source_of_command"], "", "```"]
            for s in b["steps"]:
                L.append(" ".join(s["cmd"]))
            L += ["```", "", "Outputs: `%s`, `%s` in `%s`." % (b["outputs"]["sass"], b["outputs"]["res"], b["outputs"]["deposit"]), ""]
    L += ["## Notes", "", "- Ada's nvcc is CUDA 13.2.86; Blackwell's was 13.2.78. `run_f.py`, `run_prosp.py` and `timing_fresh_d.py` refuse any nvcc that is not 13.2.78 and hard-code `sm_120` (set D runners read `HARNESS_CUDA_ARCH`), so an Ada measurement job needs those guards made board-aware; the build script here does not call them.",
          "- Set D runners also call `assert_live_arch` (nvidia-smi) when `find_nvcc` is used; `build_ada_eval_sass.py` calls `build_binary` directly and so touches no GPU.",
          "- Source hashes are in `build_manifest_ada.json` (`source_sha256`)."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    doc = build_doc()
    j, m = json.dumps(doc, indent=1, sort_keys=True) + "\n", markdown(doc)
    if a.check:
        if not OUT_JSON.is_file() or OUT_JSON.read_bytes() != j.encode() or not OUT_MD.is_file() or OUT_MD.read_bytes() != m.encode():
            print("manifest files differ from the regenerated ones", file=sys.stderr)
            return 1
        print("manifest reproduced: %d builds" % len(doc["builds"]))
        return 0
    OUT_JSON.write_bytes(j.encode())
    OUT_MD.write_bytes(m.encode())
    print("wrote", OUT_JSON.name, OUT_MD.name, len(doc["builds"]), "builds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
