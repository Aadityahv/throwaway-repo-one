#!/usr/bin/env python3
"""Freeze record of the A100/H100 replication (CPU only; reads git; never overwrites).

    python freeze_board.py write-record --board h100 --profile main|tensor [--freeze-commit <40-hex>] [--out <name>]
    python freeze_board.py check --board h100 --profile main|tensor

Procedure: produce predictions (predict_board.py) and the SASS manifest (published-method rows were dropped by recorded decision, 4 October 2026: no baselines file), COMMIT them together with every static input, then run write-record: it finds the single
commit that added the predictions file, checks the working-tree file equals that commit's blob, hashes every frozen input (cells, static tables, footprints, SASS manifest, the board's
pair-overlap table, the calibration document named in the predictions, the committed cuda-samples manifest) and writes the record. The record is committed before any timing.
`check` re-verifies the record against the working tree; the measurement wrapper does the same, plus the git-ancestry rule, inside the job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import board as BD
import cell_sets_board as CS

REPO = BD.REPO
SHA1 = re.compile(r"^[0-9a-f]{40}$")
REL = "tiresias/framework/predictor/replication_h100_a100/"
SAMPLES_MANIFEST_REL = "tiresias/app_runners/app_source_manifests/cuda-samples-5443602_cluster_MANIFEST.sha256"
SCHEMA = "cluster_freeze_record/1"


class Refusal(SystemExit):
    def __init__(self, m):
        super().__init__("REFUSED: " + m)


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _git(*args):
    return subprocess.run(["git", "-C", str(REPO)] + list(args), capture_output=True, text=True)


def write_record(board, profile, freeze_commit=None, out=None):
    d = CS.board_dir(board)
    n = CS.names(board, profile)
    pred_name = n["predictions"]
    rel = REL + board + "/" + pred_name
    if not (d / pred_name).is_file():
        raise Refusal("%s is missing" % pred_name)
    pred = json.loads((d / pred_name).read_text(encoding="utf-8"))
    if pred.get("kind") != "cluster_frozen_predictions" or pred.get("warning") or pred.get("profile") != profile or pred.get("board") != board:
        raise Refusal("%s is not a frozen-predictions file of board %s profile %s (kind %r)" % (pred_name, board, profile, pred.get("kind")))
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
    committed, disk = hashlib.sha256(blob.stdout).hexdigest(), sha256_file(d / pred_name)
    if committed != disk:
        raise Refusal("the working-tree %s differs from its blob at %s; nothing written" % (pred_name, freeze_commit))
    shas = {}
    for name in CS.frozen_inputs(board, profile) + (pred_name,):
        if not (d / name).is_file():
            raise Refusal("%s is missing: every frozen input must exist before the record is written " % name)
        shas[name] = sha256_file(d / name)
    if pred["pair_overlap_constants"]["sha256"] != shas[CS.pair_file(board)]:
        raise Refusal("the pair-overlap table changed since the predictions were made")
    manifest = REPO / SAMPLES_MANIFEST_REL
    if not manifest.is_file():
        raise Refusal("%s is missing: the committed manifest of the cuda-samples copy is part of the frozen inputs" % SAMPLES_MANIFEST_REL)
    sm = json.loads((d / n["sass_manifest"]).read_text(encoding="utf-8"))
    if sm.get("pending_build"):
        raise Refusal("%s lists %d kernels pending build: the freeze refuses while any kernel's SASS is missing" % (n["sass_manifest"], len(sm["pending_build"])))
    out = Path(out) if out else d / n["record"]
    if out.exists():
        raise Refusal("%s exists; never overwritten" % out)
    doc = dict(schema=SCHEMA, board=board, profile=profile, freeze_commit=freeze_commit, predictions_sha256=disk, sha256=shas, samples_manifest_sha256=sha256_file(manifest),
               calibration=dict(source=pred["calibration"]["source"], file=pred["calibration"]["file"], sha256=pred["calibration"]["sha256"], uuid=pred["calibration"]["uuid"]),
               pair_overlap_constants=pred["pair_overlap_constants"],
               note="Written after the freeze commit. Every file hash is that of the working-tree file when this record was written; predictions_sha256 equals the blob committed at freeze_commit (checked: yes).")
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", out)
    return doc


def write_record_injob(board, profile, code_commit):
    """Freeze record for a prediction made INSIDE the measurement allocation (recorded decision of 5 October 2026: the A100 queue is waited for once, so calibration, pair overlap, window test, prediction and measurement
    share one allocation). No git is involved: the predictions file and this record are written, hash-bound, before any measurement of any evaluation cell, from code committed beforehand (`freeze_commit` here is
    the commit of that code, not of the predictions) and static tables, cells and SASS manifest committed beforehand; every input is hashed and the record carries its creation time. The wrapper accepts such a record
    (mode `in_job`) and re-checks every hash before it runs anything; the output directory is committed afterwards."""
    import datetime
    import socket
    d = CS.board_dir(board)
    n = CS.names(board, profile)
    if not SHA1.match(code_commit or ""):
        raise Refusal("the code commit must be a full 40-character sha")
    pred_name = n["predictions"]
    if not (d / pred_name).is_file():
        raise Refusal("%s is missing" % pred_name)
    pred = json.loads((d / pred_name).read_text(encoding="utf-8"))
    if pred.get("kind") != "cluster_frozen_predictions" or pred.get("warning") or pred.get("profile") != profile or pred.get("board") != board:
        raise Refusal("%s is not a frozen-predictions file of board %s profile %s" % (pred_name, board, profile))
    shas = {}
    for name in CS.frozen_inputs(board, profile) + (pred_name,):
        if not (d / name).is_file():
            raise Refusal("%s is missing: every frozen input must exist before the record is written" % name)
        shas[name] = sha256_file(d / name)
    if pred["pair_overlap_constants"]["sha256"] != shas[CS.pair_file(board)]:
        raise Refusal("the pair-overlap table changed since the predictions were made")
    manifest = REPO / SAMPLES_MANIFEST_REL
    if not manifest.is_file():
        raise Refusal("%s is missing" % SAMPLES_MANIFEST_REL)
    sm = json.loads((d / n["sass_manifest"]).read_text(encoding="utf-8"))
    if sm.get("pending_build"):
        raise Refusal("%s lists kernels pending build" % n["sass_manifest"])
    out = d / n["record"]
    if out.exists():
        raise Refusal("%s exists; never overwritten" % out)
    doc = dict(schema=SCHEMA, mode="in_job", board=board, profile=profile, freeze_commit=code_commit, predictions_sha256=shas[pred_name], sha256=shas, samples_manifest_sha256=sha256_file(manifest),
               created_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), host=socket.gethostname(),
               calibration=dict(source=pred["calibration"]["source"], file=pred["calibration"]["file"], sha256=pred["calibration"]["sha256"], uuid=pred["calibration"]["uuid"]),
               pair_overlap_constants=pred["pair_overlap_constants"],
               note="In-job freeze: written inside the measurement allocation before any measurement of any evaluation cell; freeze_commit is the commit of the CODE that made it (no git ancestry applies); "
                    "every input file hash is that of the file when this record was written.")
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", out)
    return doc


def check(board, profile, tree=None):
    """Verify the record against a directory holding the board's files (the working tree by default)."""
    d = Path(tree) if tree else CS.board_dir(board)
    n = CS.names(board, profile)
    rec_path = d / n["record"]
    if not rec_path.is_file():
        raise Refusal("%s is missing" % n["record"])
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    if rec.get("schema") != SCHEMA or not SHA1.match(rec.get("freeze_commit", "")) or rec.get("board") != board or rec.get("profile") != profile:
        raise Refusal("%s is malformed or for another board/profile" % n["record"])
    bad = [k for k, h in rec["sha256"].items() if not (d / k).is_file() or sha256_file(d / k) != h]
    if bad:
        raise Refusal("frozen inputs missing or changed since the record: %s" % ", ".join(bad[:5]))
    if sha256_file(d / n["predictions"]) != rec["predictions_sha256"]:
        raise Refusal("the predictions changed after the freeze")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("write-record", "check"):
        p = sub.add_parser(name)
        p.add_argument("--board", required=True, choices=sorted(BD.CONFIG))
        p.add_argument("--profile", required=True, choices=CS.PROFILES)
        if name == "write-record":
            p.add_argument("--freeze-commit", default=None)
            p.add_argument("--out", default=None)
            p.add_argument("--in-job", action="store_true", help="in-job freeze (no git); --freeze-commit is the commit of the code that made the predictions")
    a = ap.parse_args(argv)
    if a.cmd == "write-record":
        if getattr(a, "in_job", False):
            write_record_injob(a.board, a.profile, a.freeze_commit)
        else:
            write_record(a.board, a.profile, a.freeze_commit, a.out)
    else:
        print("record verified: %d files" % len(check(a.board, a.profile)["sha256"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
