"""Which cells and which files belong to which H100 freeze (CPU only; reads no measured value).

Two freeze profiles share every script of this directory (predict_h100.py, baselines_h100.py, measure_h100.py, score_h100.py):

  main    the 72 CUDA-samples cells (cells_h100.json) plus the 40 machine-learning-kernel cells (ml_sets/cells_ml_h100.json), predicted from ONE calibration document, measured in ONE job
          and frozen together. File names are the frozen names of the first H100 freeze (predictions_h100.json, baselines_h100.json, freeze_record_h100.json).
  tensor  the 16 tensor-core matrix multiply and fused attention cells (ml_sets/cells_tensor_h100.json). They need the calibrator's tensor stage, which the first H100 calibration run
          does not contain, so they are predicted later, from the first complete calibration run that contains that stage (FREEZE.md section 12). Own file names
          (predictions_h100_tensor.json, baselines_h100_tensor.json, freeze_record_h100_tensor.json).

A profile lists, per kind of static table, the files to read and merge (base files in this directory, machine-learning files in ml_sets/). Paths are relative to this directory and use
forward slashes. Nothing here changes the meaning of the 72-cell files.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

PROFILES = {
    "main": dict(
        cells=("cells_h100.json", "ml_sets/cells_ml_h100.json"),
        features=("features_h100.json", "ml_sets/features_ml_h100.json"),
        phases=("phases_h100.json", "ml_sets/phases_ml_h100.json"),
        unique=("phases_unique_h100.json", "ml_sets/phases_unique_ml_h100.json"),
        bank=("bank_conflicts_h100.json", "ml_sets/bank_conflicts_ml_h100.json"),
        support=("static_support_h100.json", "ml_sets/static_support_ml_h100.json"),
        sass_manifest=("sass_manifest_h100.json", "ml_sets/sass_manifest_ml_h100.json"),
        footprints=("footprints_h100.json",),
        predictions="predictions_h100.json", baselines="baselines_h100.json", record="freeze_record_h100.json", needs_tensor=False),
    "tensor": dict(
        cells=("ml_sets/cells_tensor_h100.json",),
        features=("ml_sets/features_tensor_h100.json",),
        phases=("ml_sets/phases_tensor_h100.json",),
        unique=("ml_sets/phases_unique_tensor_h100.json",),
        bank=("ml_sets/bank_conflicts_tensor_h100.json",),
        support=("ml_sets/static_support_tensor_h100.json",),
        sass_manifest=("ml_sets/sass_manifest_tensor_h100.json",),
        footprints=("ml_sets/footprints_tensor_h100.json",),
        predictions="predictions_h100_tensor.json", baselines="baselines_h100_tensor.json", record="freeze_record_h100_tensor.json", needs_tensor=True),
}
DEFAULT_PROFILE = "main"
SET_FAMILY = {"unseen": ("matmul", "bs", "scan", "conv"), "set_e": ("sp", "fwt"), "set_d": ("tile", "reduction"),
              "fresh_f": ("gelu", "swiglu", "rmsnorm", "rope", "sgemm"), "fresh_g": ("tcgemm",), "fresh_h": ("attn",)}
ML_SETS = ("fresh_f", "fresh_g", "fresh_h")


def profile(name):
    if name not in PROFILES:
        raise SystemExit("unknown freeze profile %r (known: %s)" % (name, ", ".join(sorted(PROFILES))))
    return PROFILES[name]


def _read(rel):
    return json.loads((HERE / rel).read_text(encoding="utf-8"))


def static_files(name):
    """Every static input of a profile (relative paths): cells, tables, SASS manifest."""
    p = profile(name)
    out = []
    for k in ("cells", "features", "phases", "unique", "bank", "support", "sass_manifest", "footprints"):
        out.extend(p[k])
    return tuple(out)


def frozen_inputs(name):
    """Files hashed in the freeze record besides the predictions: the static inputs and the baselines."""
    return static_files(name) + (profile(name)["baselines"],)


def load_cells(name):
    """-> (list of cell dicts in file order, list of per-file docs). A cell id may appear once only."""
    cells, docs = [], []
    for rel in profile(name)["cells"]:
        d = _read(rel)
        docs.append(d)
        cells.extend(d["cells"])
    ids = [c["cell_id"] for c in cells]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate cell ids in the cell files of profile %s" % name)
    return cells, docs


def load_static(name):
    """Merged static tables of a profile: (features doc, phases rows, unique rows, bank rows, support rows). Missing files are a refusal naming the generator."""
    p = profile(name)
    feats, phases, unique, bank, support = None, {}, {}, {}, {}
    for rel in p["features"] + p["phases"] + p["unique"] + p["bank"] + p["support"]:
        if not (HERE / rel).is_file():
            raise SystemExit("%s is missing: run build_static_h100.py (base cells) or ml_sets/build_static_ml.py (machine-learning and tensor cells) first" % rel)
    for rel in p["features"]:
        d = _read(rel)
        if feats is None:
            feats = dict(d, rows=list(d["rows"]))
        else:
            feats["rows"] = feats["rows"] + d["rows"]
    for rel in p["phases"]:
        phases.update(_read(rel)["rows"])
    for rel in p["unique"]:
        unique.update(_read(rel)["rows"])
    for rel in p["bank"]:
        bank.update(_read(rel)["rows"])
    for rel in p["support"]:
        support.update(_read(rel)["rows"])
    return feats, phases, unique, bank, support


def merged_sass_manifest(name):
    """{kid: manifest row} over the profile's SASS manifests (kernel ids are unique across the sets)."""
    out = {}
    for rel in profile(name)["sass_manifest"]:
        d = _read(rel)
        for kid, row in d["kernels"].items():
            if kid in out:
                raise SystemExit("kernel id %s appears in two SASS manifests" % kid)
            out[kid] = row
    return out
