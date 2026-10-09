#!/usr/bin/env python3
"""Cross-check of the tensor stage of ONE calibration run (CPU only).

    python3 calibrate/tools/check_tensor_attach.py --doc <calibration_*.json of a run> --attached <document written by attach_tensor.py from the same run> [--require-ok]

A full calibration run (`run_calibration.py` with the default stage list, which includes `tensor` and `energy`) derives `constants.tensor` itself from the tensor windows and the energy stage of the SAME
run. `tools/attach_tensor.py` derives it again from the archived tensor windows and the energy document of that run. The two derivations must agree exactly (same code path, same inputs); this tool
checks that and prints one status line:

    TENSOR <status> energy=<status> rate_pJ=<rate> issue_cycles=<c> latency_cycles=<l> admitted=<ids> not_admitted=<ids> attach=AGREES|DIFFERS

Exit 0 when the tensor stage is present, the two derivations agree (and, with --require-ok, the tensor energy status is `ok` and the issue cost is measured); 3 when the stage is present but
not usable; 1 when the stage is missing or the derivations differ. A run without usable tensor constants is the signal that the tensor cells cannot be predicted from it."""
import argparse
import json
import sys
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text())


def summarize(doc, attached):
    t = (doc.get("constants") or {}).get("tensor")
    if not t:
        return 1, "TENSOR missing (the run has no tensor stage constants)"
    te, ti = t.get("energy") or {}, t.get("issue") or {}
    same = attached is not None and (attached.get("constants") or {}).get("tensor") == t
    line = "TENSOR %s energy=%s rate_pJ=%s issue_cycles=%s latency_cycles=%s admitted=%s not_admitted=%s attach=%s" % (
        t.get("status"), te.get("status"), te.get("rate_pJ_per_lane_instruction"), ti.get("issue_cycles_per_warp_instruction_per_sm"), ti.get("dependent_latency_cycles"),
        ",".join(te.get("admitted") or []) or "-", ",".join(te.get("not_admitted") or []) or "-", "AGREES" if same else "DIFFERS")
    return (0 if same else 1), line


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", type=Path, required=True)
    ap.add_argument("--attached", type=Path, required=True)
    ap.add_argument("--require-ok", action="store_true")
    a = ap.parse_args(argv)
    doc = load(a.doc)
    attached = load(a.attached) if a.attached.is_file() else None
    rc, line = summarize(doc, attached)
    print(line)
    if rc == 0 and a.require_ok:
        t = doc["constants"]["tensor"]
        if (t.get("energy") or {}).get("status") != "ok" or not (t.get("issue") or {}).get("issue_cycles_per_warp_instruction_per_sm"):
            return 3
    return rc


if __name__ == "__main__":
    sys.exit(main())
