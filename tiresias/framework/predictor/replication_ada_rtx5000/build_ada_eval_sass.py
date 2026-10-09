#!/usr/bin/env python3
"""Run on the Ada host (COMPILE ONLY: no kernel is launched, no GPU is used): build every driver of build_manifest_ada.json with Ada's CUDA 13.2 nvcc and dump the SASS and resource usage.

    python3 build_ada_eval_sass.py --samples <pinned cuda-samples> --work <scratch dir> --out <.../port_ada/compiled_ada_cuda13.2_eval> [--dry-run] [--only <build id>]

Refuses when: the samples checkout is not at the pinned revision or has modified files; nvcc is missing; the output folder exists and is not empty (never overwrites). Writes, per build, the SASS and
resource files named in the manifest (static inputs go to the top of --out, measurement drivers to --out/meas/), plus build.txt (nvcc version, every command, source hashes) and sha256.txt.
The commands are the Blackwell commands with -arch=sm_89 and Ada's nvcc (make_build_manifest.py); the set D harness binaries are built by the committed runners' own build_binary with
HARNESS_CUDA_ARCH=sm_89 (their driver sources come from their own templates). The Blackwell nvcc-version guards (13.2.78) are not called; the version found is recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PINNED = "5443602d89ed99aede2e4b7bf329daddeadb320e"
MANIFEST = HERE / "build_manifest_ada.json"
RUNNER_DIRS = ["tiresias/app_runners", "tiresias/framework/unseen_operators/runners", "energy_harness"]


def sub(s, samples, work, bid):
    return s.replace("<REPO>", str(REPO)).replace("<SAMPLES>", str(samples)).replace("<WORK>", str(work))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_samples(root):
    rev = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if rev != PINNED:
        raise SystemExit("REFUSED: samples revision %r is not the pinned %s" % (rev, PINNED))
    st = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    if st:
        raise SystemExit("REFUSED: the samples checkout has modified or untracked files:\n" + st[:500])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    nvcc = man["nvcc"]
    builds = [b for b in man["builds"] if not a.only or a.only == b["id"]]
    if a.dry_run:
        for b in builds:
            print("# %s (%s)" % (b["id"], b["role"]))
            if b.get("python_entry"):
                print("#   python entry: %s with HARNESS_CUDA_ARCH=sm_89" % b["python_entry"])
            for s in b["steps"]:
                print(" ".join(sub(x, a.samples, a.work, b["id"]) for x in s["cmd"]))
            print("%s -sass -arch sm_89 %s > %s" % (Path(nvcc).with_name("cuobjdump"), sub(b["binary"], a.samples, a.work, b["id"]), b["outputs"]["sass"]))
        return 0
    if not Path(nvcc).is_file():
        raise SystemExit("REFUSED: %s not found (this script is for the Ada host)" % nvcc)
    check_samples(a.samples)
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit("REFUSED: %s is not empty; never overwritten" % a.out)
    try:
        os.nice(19)
    except (AttributeError, OSError):
        pass
    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip()
    if "release 13.2" not in ver:
        raise SystemExit("REFUSED: nvcc is not CUDA 13.2:\n" + ver)
    cuobjdump = str(Path(nvcc).with_name("cuobjdump"))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "meas").mkdir(exist_ok=True)
    log = ["nvcc: " + ver.replace("\n", " | "), "arch: sm_89", "samples_rev: " + PINNED, "built_on: Ada host, compile only, nice 19", ""]
    hashes = []
    for b in builds:
        work = a.work / b["id"]
        work.mkdir(parents=True, exist_ok=True)
        if b.get("python_entry"):
            os.environ["HARNESS_CUDA_ARCH"] = "sm_89"
            os.environ["HARNESS_NVCC"] = nvcc
            for d in RUNNER_DIRS:
                sys.path.insert(0, str(REPO / d))
            mod = __import__(b["id"])
            mod.build_binary(a.samples, work, nvcc)
            log.append("CMD %s: python build_binary(%s) HARNESS_CUDA_ARCH=sm_89 (commands in build_manifest_ada.json)" % (b["id"], b["id"]))
        else:
            for s in b["steps"]:
                cmd = [sub(x, a.samples, a.work, b["id"]) for x in s["cmd"]]
                cmd[0] = nvcc
                log.append("CMD %s: %s" % (b["id"], " ".join(cmd)))
                p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(work))
                (work / "build.log").write_text("$ %s\n%s%s" % (" ".join(cmd), p.stdout, p.stderr), encoding="utf-8")
                if p.returncode:
                    raise SystemExit("BUILD FAILED %s, see %s" % (b["id"], work / "build.log"))
        binary = Path(sub(b["binary"], a.samples, a.work, b["id"]))
        if not binary.is_file():
            raise SystemExit("binary missing after build: %s" % binary)
        for kind, flag in (("sass", "-sass"), ("res", "-res-usage")):
            dst = a.out / b["outputs"][kind]
            dst.parent.mkdir(parents=True, exist_ok=True)
            r = subprocess.run([cuobjdump, flag, "-arch", "sm_89", str(binary)], capture_output=True, text=True)
            if r.returncode:
                raise SystemExit("cuobjdump failed for %s: %s" % (b["id"], r.stderr[:300]))
            dst.write_text(r.stdout, encoding="utf-8")
            hashes.append("%s  %s" % (sha(dst), dst.relative_to(a.out)))
        hashes.append("%s  binary:%s" % (sha(binary), b["id"]))
    (a.out / "build.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    (a.out / "sha256.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    print("built %d drivers into %s (compile only)" % (len(builds), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
