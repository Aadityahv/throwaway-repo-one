"""Which Ada cells and static files belong to which freeze profile (CPU only; reads no measured value). Mirrors h100_eval_freeze/cell_sets.py.

  main    every cell whose kernels need no tensor-core constant: sets of CUDA samples (76), machine-learning kernels (40), and the validation cells that are not tensor-core kernels (8)
  tensor  every cell with tensor-core instructions (tensor-core matrix multiply and fused attention, prospective cells, validation tensor matmul and attention): 16 + 24 + 4 = 44 cells. They need the
          calibrator's tensor stage, so they are predicted from a calibration document that contains it.
Own frozen names: predictions_ada.json / predictions_ada_tensor.json, baselines_ada*.json, freeze_record_ada*.json (never written by plumbing tests).
"""
from __future__ import annotations

import json
from pathlib import Path

import board_ada as B
import make_cells_ada as MA

HERE = B.HERE
STATIC = HERE / "static"
TENSOR_KIDS = frozenset(["tc128", "tc64", "at4", "at8", "tc128x64", "tc256x128", "at2", "at16", "nm4", "nm8", "tcr128", "tcr64"])
PROFILES = {
    "main": dict(predictions="predictions_ada.json", baselines="baselines_ada.json", record="freeze_record_ada.json", needs_tensor=False),
    "tensor": dict(predictions="predictions_ada_tensor.json", baselines="baselines_ada_tensor.json", record="freeze_record_ada_tensor.json", needs_tensor=True),
}
DEFAULT_PROFILE = "main"
FROZEN_NAMES = tuple(v[k] for v in PROFILES.values() for k in ("predictions", "baselines", "record"))
# scoring families per set (as h100 cell_sets.SET_FAMILY, plus vector add and the validation and prospective families)
SET_FAMILY = {"unseen": ("matmul", "bs", "scan", "conv"), "set_e": ("sp", "fwt"), "set_d": ("tile", "reduction", "vecadd"), "fresh_f": ("gelu", "swiglu", "rmsnorm", "rope", "sgemm"),
              "fresh_g": ("tcgemm",), "fresh_h": ("attn",), "validation": ("sgemm", "tcgemm", "attn", "sp", "conv", "rmsnorm"), "prospective": ("tcgemm2", "attn2", "attnnm", "tcrelu")}


def needs_tensor(cell) -> bool:
    return any(k["kid"] in TENSOR_KIDS for k in cell["kernels"])


def profile(name):
    if name not in PROFILES:
        raise SystemExit("unknown freeze profile %r (known: %s)" % (name, ", ".join(sorted(PROFILES))))
    return PROFILES[name]


def load_cells(name):
    p = profile(name)
    return [c for c in MA.load_cells() if needs_tensor(c) == p["needs_tensor"]]


def _read(kind, group):
    f = STATIC / ("%s_%s.json" % (kind, group))
    if not f.is_file():
        raise SystemExit("%s is missing: run static_ada.py first" % f.name)
    return json.loads(f.read_text(encoding="utf-8"))


def load_static(name):
    """(features doc with rows for the profile's cells, phases, unique, bank, support, footprint rows) over all groups; only the profile's cells are kept."""
    keep = {c["cell_id"] for c in load_cells(name)}
    feats, phases, unique, bank, support, fps = None, {}, {}, {}, {}, {}
    for g in MA.GROUPS:
        d = _read("features", g)
        rows = [r for r in d["rows"] if r["cell_id"] in keep]
        feats = dict(d, rows=(feats["rows"] if feats else []) + rows)
        for store, kind in ((phases, "phases"), (unique, "phases_unique"), (bank, "bank_conflicts"), (support, "static_support"), (fps, "footprints")):
            store.update({k: v for k, v in _read(kind, g)["rows"].items() if k in keep})
    return feats, phases, unique, bank, support, fps


def static_files():
    return tuple(sorted("static/%s_%s.json" % (k, g) for k in ("features", "phases", "phases_unique", "bank_conflicts", "static_support", "footprints") for g in MA.GROUPS)) + tuple(MA.FILES[g] for g in MA.GROUPS)


def unmeasured_opcodes(phase_row):
    """{opcode: executed warp instructions} of the opcodes whose Ada throughput is unmeasured (board_ada.UNMEASURED_OPCODE_PREFIXES) in a cell's phase table."""
    found = {}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "issue_warp_instructions" and isinstance(v, dict):
                    for op, n in v.items():
                        if op.startswith(B.UNMEASURED_OPCODE_PREFIXES) and n:
                            found[op] = found.get(op, 0) + n
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(phase_row)
    return found
