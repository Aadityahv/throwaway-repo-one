"""Which cells and static files belong to which freeze profile of the A100/H100 replication (CPU only; reads no measured value).

  main    every cell whose kernels need no tensor-core constant: CUDA-sample sets (76), machine-learning kernels (40) and the validation cells that are not tensor-core kernels (8) = 124
  tensor  every cell with tensor-core instructions: tensor-core matrix multiply and fused attention (16), the later tensor/attention test cells (24) and the tensor validation cells (4) = 44

The two profiles partition the 168 intended cells. Predictions, baselines and freeze record are per profile (names built from the board). Freezing both profiles in one commit before any cell runs is
the default; freezing the tensor profile later is a separate recorded decision (see README, "freeze order").
"""
from __future__ import annotations

import json
from pathlib import Path

import board as BD

GROUPS = ("samples", "ml", "tensor", "validation", "prospective")
STATIC_KINDS = ("features", "phases", "phases_unique", "bank_conflicts", "static_support", "footprints")
TENSOR_KIDS = frozenset(["tc128", "tc64", "at4", "at8", "tc128x64", "tc256x128", "at2", "at16", "nm4", "nm8", "tcr128", "tcr64"])
PROFILES = ("main", "tensor")
# routing of a cell to a measurement engine, by kernel family (the validation cells have set "validation" but belong to the engine of their family)
ENGINE_OF_FAMILY = {"matmul": "unseen", "bs": "unseen", "scan": "unseen", "conv": "unseen", "sp": "set_e", "fwt": "set_e", "tile": "set_d", "reduction": "set_d", "vecadd": "set_d",
                    "gelu": "fresh", "swiglu": "fresh", "rmsnorm": "fresh", "rope": "fresh", "sgemm": "fresh", "tcgemm": "fresh", "attn": "fresh",
                    "tcgemm2": "prosp", "tcrelu": "prosp", "attn2": "prosp", "attnnm": "prosp"}


def needs_tensor(cell) -> bool:
    return any(k["kid"] in TENSOR_KIDS for k in cell["kernels"])


# Freeze generation of each board: appended to every frozen file name so a re-freeze never reuses a name (the freeze record finds "the single commit that added" the predictions file, and a
# replaced file under an old name would point at the superseded freeze). h100: re-frozen on 4 October 2026 after a recalibration on the board that the measurement runs on, before any evaluation
# cell was measured; the first freeze (912e8e26, calibration of GPU-6132e1b7, untagged names) is superseded and stays in git history.
ACTIVE_TAG = {"h100": "29b9a2c0", "a100": "injob"}   # a100: in-job freeze (injob_freeze.py): the predictions are made inside the measurement allocation from its own calibration


def tag(board):
    t = ACTIVE_TAG.get(board, "")
    return "_" + t if t else ""


def pair_file(board):
    return "pair_overlap_constants%s.json" % tag(board)


def names(board, profile):
    if profile not in PROFILES:
        raise SystemExit("unknown profile %r (known: %s)" % (profile, ", ".join(PROFILES)))
    suffix = "" if profile == "main" else "_tensor"
    return dict(predictions="predictions_replication_%s%s%s.json" % (board, tag(board), suffix),
                record="freeze_record_replication_%s%s%s.json" % (board, tag(board), suffix), sass_manifest="sass_manifest_replication_%s.json" % board)


def board_dir(board):
    return BD.binding(board).HERE


def load_cells(board, profile, groups=GROUPS):
    d = board_dir(board)
    cells = []
    for g in groups:
        cells += json.loads((d / ("cells_%s.json" % g)).read_text(encoding="utf-8"))["cells"]
    return [c for c in cells if needs_tensor(c) == (profile == "tensor")]


def static_files(board):
    """Relative to the board directory: the cell documents and every static table, the inputs of a frozen prediction."""
    return tuple(sorted("static/%s_%s.json" % (k, g) for k in STATIC_KINDS for g in GROUPS)) + tuple("cells_%s.json" % g for g in GROUPS)


def engine_of(cell) -> str:
    try:
        return ENGINE_OF_FAMILY[cell["family"]]
    except KeyError:
        raise SystemExit("cell %s: family %r has no measurement engine" % (cell["cell_id"], cell["family"]))


def frozen_inputs(board, profile):
    n = names(board, profile)
    return static_files(board) + (n["sass_manifest"], pair_file(board))
