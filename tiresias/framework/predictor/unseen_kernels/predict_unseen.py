#!/usr/bin/env python3
"""Merge the per-cell static results and write the frozen predictions for the unseen-kernel test (Blackwell, CPU only).

Runtime: the committed predictors v2 and v3d, and v3e (the model under test), called unchanged on the new tables.
Energy: the frozen decomposed energy model (energy_model.py) evaluated at each predicted runtime, with its runtime,
memory and compute terms reported separately. Cells the static pipeline refused are kept as unsupported rows (a
coverage failure, never dropped). No runtime, energy or power value is read; nothing here touches a GPU.

Run after every cell has been built (unseen_pipeline.py), and commit the outputs before any timing or energy run.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SR = HERE.parent
sys.path.insert(0, str(SR))
sys.path.insert(0, str(HERE))
import predict_runtime_v2 as V2  # noqa: E402  (imported, never edited)
import predict_runtime_v3 as V3  # noqa: E402
import energy_model as EM  # noqa: E402

CELL_DIR = HERE / "build/cells"
OUT = HERE / "frozen"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def merge():
    feats, phases, uniq, cells = [], {}, {}, {}
    for p in sorted(CELL_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        c = d["cell"]
        cid = c["cell_id"]
        cells[cid] = c
        if "refused" in d:
            feats.append(dict(cell_id=cid, status="missing_features", missing_features=[dict(feature="static pipeline", reason=d["refused"])]))
            phases[cid] = dict(status="unsupported", reason=d["refused"], kernels=[])
            uniq[cid] = dict(status="unsupported", reason=d["refused"], kernels=[])
        else:
            feats.append(d["features"])
            phases[cid] = d["phases"]
            uniq[cid] = d["unique"]
    return sorted(feats, key=lambda r: r["cell_id"]), phases, uniq, cells


def development_ranges():
    """Range of instruction density per logical byte over the development cells (for the in/out-of-range flag)."""
    cells, _, _ = EM.load_development()
    nm = [c[4]["nonmem_lane_instructions"] / (c[4]["bytes_L2"] + c[4]["bytes_DRAM"]) for c in cells]
    sf = [c[4]["sfu_lane_instructions"] / (c[4]["bytes_L2"] + c[4]["bytes_DRAM"]) for c in cells]
    return dict(nonmem_per_byte=[min(nm), max(nm)], sfu_per_byte=[min(sf), max(sf)])


def main():
    OUT.mkdir(exist_ok=True)
    feats, phases, uniq, cells = merge()
    import unseen_pipeline as UP  # noqa: E402
    hw = UP.X.load_hardware(UP.X.read_text(UP.X.GROUND_TRUTH))
    features = {"schema": "static_runtime_features_blackwell/1", "hardware_from_ground_truth": hw, "rows": feats,
                "note": "Label-free static features of the unseen-kernel cells (pinned cuda-samples). No runtime, energy or power value was read."}
    (OUT / "features_unseen.json").write_text(json.dumps(features, indent=1, sort_keys=True) + "\n")
    (OUT / "phases_unseen.json").write_text(json.dumps(dict(schema="static_dynamic_barrier_phases/1", rows=phases), indent=1, sort_keys=True) + "\n")
    (OUT / "phases_unique_unseen.json").write_text(json.dumps(dict(schema="first_touch_unique_sectors/1", rows=uniq), indent=1, sort_keys=True) + "\n")

    stream = json.loads((SR / "constants/stream_constants.json").read_text())["constants"]
    K = V2.make_constants(stream, json.loads((SR / "constants/microbench_constants_v2.json").read_text()))
    v3c = json.loads((SR / "constants/v3c_constants.json").read_text())
    v3 = json.loads((SR / "constants/v3_constants.json").read_text())
    v3e = json.loads((SR / "constants/v3e_constants.json").read_text())
    pred = {"v2": V2.build(features, phases, K),
            "v3d": V3.build(features, phases, uniq, K, v3, v3c, True),
            "v3e": V3.build(features, phases, uniq, K, v3, v3c, True, v3e)}
    for name, p in pred.items():
        (OUT / ("predictions_unseen_%s.json" % name)).write_text(json.dumps(p, indent=1, sort_keys=True) + "\n")

    model = json.loads(EM.FROZEN.read_text())
    ranges = development_ranges()
    energy = {}
    for f in feats:
        cid = f["cell_id"]
        c = cells[cid]
        t = pred["v3e"][cid].get("primary_s")
        if f["status"] == "missing_features" or not f.get("per_launch_totals"):
            energy[cid] = dict(status="unsupported", reason="static features unavailable", energy_j=None)
            continue
        terms = EM.terms_from_totals(f["per_launch_totals"], c["logical_bytes_per_launch"], c["tier"])
        nb = c["logical_bytes_per_launch"]
        flags = dict(nonmem_per_byte=terms["nonmem_lane_instructions"] / nb, sfu_per_byte=terms["sfu_lane_instructions"] / nb)
        flags["nonmem_outside_development_range"] = not (ranges["nonmem_per_byte"][0] <= flags["nonmem_per_byte"] <= ranges["nonmem_per_byte"][1])
        flags["sfu_outside_development_range"] = not (ranges["sfu_per_byte"][0] <= flags["sfu_per_byte"] <= ranges["sfu_per_byte"][1])
        e = dict(status="ok" if t else "runtime_unsupported", terms=terms, flags=flags, counts_exact=bool(f["per_launch_totals"]["all_counts_exact"]))
        if t:
            e["with_v3e_runtime"] = EM.predict(terms, t, model)
            e["v3e_runtime_s"] = t
        energy[cid] = e
    (OUT / "energy_predictions_unseen.json").write_text(json.dumps(dict(model=model, development_ranges=ranges, cells=energy), indent=1, sort_keys=True) + "\n")

    cell_doc = dict(cells=[{k: v for k, v in c.items() if k != "kernels"} | dict(kernel_launches=[dict(kid=k["kid"], grid=k["grid"], block=k["block"], args=k["args"]) for k in c["kernels"]]) for c in cells.values()])
    (OUT / "cells_unseen.json").write_text(json.dumps(cell_doc, indent=1, sort_keys=True) + "\n")
    cov = {m: dict(supported=sum(v.get("primary_s") is not None for v in p.values()), cells=len(p)) for m, p in pred.items()}
    freeze = dict(schema="unseen_kernel_prediction_freeze/1", frozen_before_timing=True, coverage=cov,
                  inputs_sha256={n: sha(OUT / n) for n in ("features_unseen.json", "phases_unseen.json", "phases_unique_unseen.json", "cells_unseen.json")},
                  code_sha256={n: sha(p) for n, p in (("predict_unseen.py", __file__), ("energy_model.py", HERE / "energy_model.py"), ("unseen_pipeline.py", HERE / "unseen_pipeline.py"),
                                                      ("cells.py", HERE / "cells.py"), ("predict_runtime_v2.py", SR / "predict_runtime_v2.py"), ("predict_runtime_v3.py", SR / "predict_runtime_v3.py"))},
                  constants_sha256={n: sha(SR / "constants" / n) for n in ("stream_constants.json", "microbench_constants_v2.json", "v3_constants.json", "v3c_constants.json", "v3e_constants.json")},
                  energy_model_sha256=sha(EM.FROZEN),
                  outputs_sha256={n: sha(OUT / n) for n in ("predictions_unseen_v2.json", "predictions_unseen_v3d.json", "predictions_unseen_v3e.json", "energy_predictions_unseen.json")},
                  note="Written and committed before any unseen-kernel cell was timed or energy-measured. No runtime, energy or power value was read.")
    (OUT / "PREDICTION_FREEZE_UNSEEN.json").write_text(json.dumps(freeze, indent=1, sort_keys=True) + "\n")
    print(json.dumps(cov))


if __name__ == "__main__":
    main()
