#!/usr/bin/env python3
"""Vendored copy of the Alavani-style SASS opcode classifier and function-section splitter.

Copied verbatim (logic unchanged) from `tiresias/app_runners/extract_sass_mix_remote.py`'s
`sass_instruction_mix()` and `function_sections()`, per this task's instruction to reuse the
existing clean-room classification rather than re-derive it, and per the "no repo checkout on
Ada" constraint (Ada's disk is ~100% full -- see STAGING.md) this file is a small standalone
module staged next to `emit_sass.py`, not an import from `tiresias/app_runners/`.

Do not edit `tiresias/app_runners/extract_sass_mix_remote.py` to keep this in sync by import;
these are independent copies, same as the FlipFlop task's own `flipflop_common.py` treats
`analysis/r2_flipflop_score.py`. If the original classifier changes, update both.

Extends the original with `isolate_function_section()`, a name-substring isolator over
`function_sections()` output that refuses loudly (ValueError, listing every function name seen)
on zero or multiple matches -- the September extraction script never needed this for Triton dumps
(each Triton cubin held exactly one kernel, so it fed the whole SASS text straight to
`sass_instruction_mix()`), but the CUDA Samples families' `cuobjdump --dump-sass` output holds one
`Function :` block per template instantiation (e.g. reduction's `reduce2..reduce5` are all present
in one file), so isolating the one instantiation a given (parent, regime, candidate) cell actually
uses is required there. Reused here for every family (including Triton) for the same reason
`emit_sass.py`'s task description asks for it: never silently guess which function section the
mix was computed from.
"""
from __future__ import annotations

import re

LOAD_OPS = {"LDG", "LDS", "LDC", "LDCU", "LD"}
STORE_OPS = {"STG", "STS", "ST"}
BRANCH_OPS = {"BRA", "EXIT", "BRX", "BSSY", "BSYNC", "CALL", "RET"}
ARITH_OPS = {"FFMA", "FADD", "FMUL", "DFMA", "DADD", "DMUL", "HFMA", "HADD", "HMUL",
             "IMAD", "IADD", "IADD3", "LOP3", "LOP"}
SASS_LINE = re.compile(r"/\*[0-9a-fA-F]{4}\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9]*)")
FUNC_HEADER = re.compile(r"^\s*Function\s*:\s*(\S+)\s*$")


def sass_instruction_mix(text: str) -> dict:
    """Identical logic to `extract_sass_mix_remote.py`'s function of the same name."""
    opcodes = SASS_LINE.findall(text)
    if not opcodes:
        raise ValueError("no SASS instruction lines matched")
    total = len(opcodes)
    loads = sum(1 for op in opcodes if op in LOAD_OPS)
    stores = sum(1 for op in opcodes if op in STORE_OPS)
    branches = sum(1 for op in opcodes if op in BRANCH_OPS)
    arith = sum(1 for op in opcodes if op in ARITH_OPS)
    return {
        "static_total_instructions": total,
        "static_unique_opcodes": len(set(opcodes)),
        "static_load_count": loads,
        "static_store_count": stores,
        "static_branch_count": branches,
        "static_arith_count": arith,
        "static_branch_density": branches / total,
        "static_arith_density": arith / total,
        "static_load_density": loads / total,
        "static_store_density": stores / total,
    }


def function_sections(text: str) -> dict[str, str]:
    """Split a `cuobjdump --dump-sass` text dump into {mangled_name: sass_text}.

    Identical logic to `extract_sass_mix_remote.py`'s function of the same name, except this
    version takes the already-read text directly (the original took a Path and read it itself);
    callers here always have the cuobjdump stdout as an in-memory string already.
    """
    sections: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        m = FUNC_HEADER.match(line)
        if m:
            current = m.group(1)
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


def isolate_function_section(text: str, entry_substring: str) -> tuple[str, str]:
    """Return (function_name, sass_text) for the exactly-one function section in `text`
    whose mangled name contains `entry_substring`. Refuses loudly (ValueError, listing every
    function name seen) on zero or multiple matches, rather than guessing or aggregating.
    """
    sections = function_sections(text)
    matches = [(name, body) for name, body in sections.items() if entry_substring in name]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one SASS function containing {entry_substring!r}, "
            f"found {len(matches)}. All function names seen: {sorted(sections)}"
        )
    return matches[0]
