#!/usr/bin/env python3
"""Runtime-only scoring of the unseen-kernel test (energy is measured in a later, separately approved booking).

Calls the frozen score_unseen.score() unchanged with no energy rows, so criterion 1 and every runtime statistic are computed
exactly as pre-registered; the energy criteria are not evaluable here and are left out. Written and committed before the
timing result was opened. Run once:  python3 score_unseen_runtime_only.py --timing measured/timing_unseen_result.json --out RESULT_runtime.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import score_unseen as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit("REFUSED: %s exists; the runtime score is written once" % a.out)
    rep = S.score(json.loads(a.timing.read_text()), [])
    keep = {k: rep[k] for k in ("schema", "cells_total", "cells_timed", "models", "criterion_1_runtime")}
    keep["energy_criteria"] = "not evaluated: energy is measured in a later booking"
    keep["inputs_sha256"] = {"timing": hashlib.sha256(a.timing.read_bytes()).hexdigest(),
                             "score_unseen.py": S.sha(S.__file__), "freeze": S.sha(S.FROZEN / "PREDICTION_FREEZE_UNSEEN.json")}
    a.out.write_text(json.dumps(keep, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"criterion_1_runtime": rep["criterion_1_runtime"]}, indent=1))


if __name__ == "__main__":
    main()
