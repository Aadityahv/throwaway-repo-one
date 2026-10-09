"""Ada (RTX 5000 Ada Generation, sm_89) board constants for the Ada replication of the Blackwell/H100 evaluation. CPU only; no measured value is read anywhere here.

Every hardware number comes from the Ada section of HARDWARE_GROUND_TRUTH.md at run time (read_hardware); nothing is typed. No Blackwell hardware value (its L2 size or SM count) may reach an Ada path.
"""
from __future__ import annotations

import contextlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = HERE.parent
FZ = SR / "h100_eval_freeze"
REPO = SR.parents[2]
GROUND_TRUTH = REPO / "HARDWARE_GROUND_TRUTH.md"
SECTION = "## Ada"
BLACKWELL_SECTION = "## Blackwell"
ARCH = "sm_89"
PREFIX = "ada"
GPU_NAME = "NVIDIA RTX 5000 Ada Generation (sm_89)"
CUDA_NVCC = "/usr/local/cuda-13.2/bin/nvcc"
SASS_BASE = SR / "port_ada" / "compiled_ada_cuda13.2"            # exists: nine CUDA-samples builds (the authors, Phase A)
SASS_EVAL = SR / "port_ada" / "compiled_ada_cuda13.2_eval"       # to be filled from the Ada host by the parent (BUILD_MANIFEST.md)
PAIR_CONSTANTS = SR / "constants" / "pair_overlap_constants_ada.json"
ADA_CALIBRATION = SR / "calibrate" / "runs" / "ada_capsafe_v2_20261004" / "run_full" / "calibration_sm_89_4640d904_store_stream_rates.json"   # Ada's first complete calibration (commit 1e86f68d), tensor stage included
# opcodes priced by shared class mapping, not refused: F2FP falls into the
# catch-all issue bucket (0.25 default cost) and the integer energy family,
# FMNMX into the floating-point-other bucket and family (same mapping as on
# Blackwell; Ada's own calibrated class costs; no per-opcode rate was ever
# measured on either board). Listed as a known weakness in FREEZE.md;
# reversible until freeze by restoring the prefixes below.
UNMEASURED_OPCODE_PREFIXES = ()
LABEL_ALIAS = ("| Maximum resident blocks per SM |", "| Max resident blocks per SM |")   # the Ada row uses the long label; extract_features.load_hardware parses the short one

for p in (SR, FZ):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _section_text(section: str, text: str | None = None) -> str:
    return text if text is not None else GROUND_TRUTH.read_text(encoding="utf-8")


def read_hardware(section: str = SECTION, text: str | None = None):
    """-> (hw dict as extract_features.load_hardware, quoted rows). Reads HARDWARE_GROUND_TRUTH.md; the one in-memory change is the label alias for the resident-blocks row
    (same value, same row; the file itself is not edited)."""
    import extract_features as X
    import make_cells_h100 as MC
    text = _section_text(section, text)
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(section))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    if not any(l.startswith(LABEL_ALIAS[1]) for l in lines[start:end]):
        for i in range(start, end):
            if lines[i].startswith(LABEL_ALIAS[0]):
                lines[i] = lines[i].replace(LABEL_ALIAS[0], LABEL_ALIAS[1], 1)
    text = "\n".join(lines)
    old = MC.SECTION
    MC.SECTION = section
    try:
        return MC.read_h100_hardware(text)
    finally:
        MC.SECTION = old


def read_ada_hardware():
    hw, quoted = read_hardware(SECTION)
    return hw, quoted


def forbidden_blackwell_values():
    """The Blackwell L2 size and SM count, read from the Blackwell section (never typed), for the 'no Blackwell value in any Ada path' tests."""
    hw, _ = read_hardware(BLACKWELL_SECTION)
    return hw["l2_bytes"], hw["sm_count"]


def power_limit_w(text: str | None = None) -> float:
    """The enforced power limit of Ada from the Ada section of HARDWARE_GROUND_TRUTH.md (row 'Power limit (current/default/max)'); the below-limit rule uses 95% of it."""
    text = _section_text(SECTION, text)
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(SECTION))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    for l in lines[start:end]:
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if l.startswith("|") and len(cells) >= 2 and cells[0].startswith("Power limit (current/default/max)"):
            m = re.match(r"(\d+(?:\.\d+)?)\s*W", cells[1])
            if m:
                return float(m.group(1))
    raise SystemExit("HARDWARE_GROUND_TRUTH.md Ada section has no parsable 'Power limit (current/default/max)' row")
