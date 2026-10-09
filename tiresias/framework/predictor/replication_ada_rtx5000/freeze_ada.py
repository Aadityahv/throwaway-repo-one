#!/usr/bin/env python3
"""Freeze record of the Ada evaluation, mirroring `measure_h100.py write-freeze-record` (CPU only; reads git; never overwrites).

    python freeze_ada.py write-record --profile main|tensor [--freeze-commit <40-hex>] [--out <name>]
    python freeze_ada.py check --profile main|tensor            # verify the record against the working tree and git

Procedure (FREEZE.md section 9): produce predictions_ada*.json (predict_ada.py) and baselines_ada*.json, COMMIT them (and every static input), then run write-record: it finds the single commit that added
the predictions file, checks the working-tree file equals that commit's blob, hashes every frozen input (cells, static tables, footprints, SASS manifest of the Ada build, Ada pair-overlap constants,
the Ada calibration document named in the predictions) and writes freeze_record_ada*.json. The record is committed before any Ada timing. Plumbing tests never write these names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import board_ada as B
import cell_sets_ada as CS

HERE = B.HERE
REPO = B.REPO
SHA1 = re.compile(r"^[0-9a-f]{40}$")
REL = "tiresias/framework/predictor/replication_ada_rtx5000/"


class Refusal(SystemExit):
    def __init__(self, m):
        super().__init__("REFUSED: " + m)


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _git(*args):
    return subprocess.run(["git", "-C", str(REPO)] + list(args), capture_output=True, text=True)


def frozen_inputs(profile):
    names = list(CS.static_files()) + [CS.profile(profile)["baselines"]]
    extra = {str(B.PAIR_CONSTANTS.relative_to(B.SR)).replace("\\", "/"): B.PAIR_CONSTANTS}
    return names, extra


def write_record(profile, freeze_commit=None, out=None):
    prof = CS.profile(profile)
    pred_name = prof["predictions"]
    rel = REL + pred_name
    if not (HERE / pred_name).is_file():
        raise Refusal("%s is missing" % pred_name)
    pred = json.loads((HERE / pred_name).read_text(encoding="utf-8"))
    if pred.get("kind") != "ada_frozen_predictions" or pred.get("warning"):
        raise Refusal("%s is not an Ada frozen-predictions file (kind %r)" % (pred_name, pred.get("kind")))
    if freeze_commit is None:
        r = _git("log", "--diff-filter=A", "--format=%H", "--", rel)
        commits = [x for x in r.stdout.split() if x]
        if r.returncode != 0 or len(commits) != 1:
            raise Refusal("cannot find the single commit that added %s (git said %r); pass --freeze-commit" % (rel, r.stdout.strip()))
        freeze_commit = commits[0]
    if not SHA1.match(freeze_commit):
        raise Refusal("freeze commit must be a full 40-character sha")
    blob = subprocess.run(["git", "-C", str(REPO), "show", "%s:%s" % (freeze_commit, rel)], capture_output=True)
    if blob.returncode != 0:
        raise Refusal("%s is not in commit %s" % (rel, freeze_commit))
    committed = hashlib.sha256(blob.stdout).hexdigest()
    disk = sha256_file(HERE / pred_name)
    if committed != disk:
        raise Refusal("the working-tree %s differs from its blob at %s; nothing written" % (pred_name, freeze_commit))
    names, extra = frozen_inputs(profile)
    shas = {}
    for n in names + [pred_name]:
        if not (HERE / n).is_file():
            raise Refusal("%s is missing: every frozen input must exist before the record is written" % n)
        shas[n] = sha256_file(HERE / n)
    for n, p in extra.items():
        if not p.is_file():
            raise Refusal("%s is missing" % n)
        shas["../" + n] = sha256_file(p)
    if pred["pair_overlap_constants"]["sha256"] != sha256_file(B.PAIR_CONSTANTS):
        raise Refusal("the Ada pair-overlap constants changed since the predictions were made")
    out = Path(out) if out else HERE / prof["record"]
    if out.exists():
        raise Refusal("%s exists; never overwritten" % out)
    doc = dict(schema="ada_freeze_record/1", profile=profile, freeze_commit=freeze_commit, predictions_sha256=disk, sha256=shas,
               calibration=dict(source=pred["calibration"]["source"], file=pred["calibration"]["file"], sha256=pred["calibration"]["sha256"], uuid=pred["calibration"]["uuid"]),
               pair_overlap_constants=pred["pair_overlap_constants"],
               note="Written after the freeze commit. Every file hash is that of the working-tree file when this record was written; predictions_sha256 equals the blob committed at freeze_commit (checked: yes).")
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", out)
    return doc


def check(profile):
    prof = CS.profile(profile)
    rec = json.loads((HERE / prof["record"]).read_text(encoding="utf-8"))
    if rec.get("schema") != "ada_freeze_record/1" or not SHA1.match(rec.get("freeze_commit", "")):
        raise Refusal("%s is malformed" % prof["record"])
    bad = [n for n, h in rec["sha256"].items() if sha256_file((HERE / n).resolve()) != h]
    if bad:
        raise Refusal("frozen inputs changed since the record: %s" % ", ".join(bad))
    print("record verified: %d files" % len(rec["sha256"]))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-record")
    w.add_argument("--profile", choices=sorted(CS.PROFILES), default="main")
    w.add_argument("--freeze-commit", default=None)
    w.add_argument("--out", default=None)
    c = sub.add_parser("check")
    c.add_argument("--profile", choices=sorted(CS.PROFILES), default="main")
    a = ap.parse_args(argv)
    if a.cmd == "write-record":
        write_record(a.profile, a.freeze_commit, a.out)
    else:
        check(a.profile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
