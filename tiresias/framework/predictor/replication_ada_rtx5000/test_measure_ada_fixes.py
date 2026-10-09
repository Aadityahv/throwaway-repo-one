"""Tests of the three measure_ada.py fixes of 5 October 2026 (same as the replication wrapper's): routing by kernel family, the `correct` flag for the fresh/prospective energy engines, and the energy-coverage check.
Run: python -m pytest -q test_measure_ada_fixes.py"""
import csv
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_sets_ada as CS  # noqa: E402
import measure_ada as M  # noqa: E402


def test_every_cell_of_both_profiles_has_an_engine_and_validation_cells_are_routed_by_family():
    cells = CS.load_cells("main") + CS.load_cells("tensor")
    assert len(cells) == 168
    engines = {c["cell_id"]: M.engine_of(c) for c in cells}
    assert set(engines.values()) <= set(M.ENGINES)
    val = [c for c in cells if c["set"] == "validation"]
    assert len(val) == 12
    for c in val:
        assert M.engine_of(c) == {"sp": "set_e", "conv": "unseen"}.get(c["family"], "fresh")
    assert sum(1 for c in cells if M.engine_of(c) == "prosp") == 24


def test_unknown_family_is_refused():
    with pytest.raises(SystemExit):
        M.engine_of(dict(cell_id="x", family="nonsense"))


def test_energy_stage_hands_the_engines_a_correct_flag():
    src = (HERE / "measure_ada.py").read_text()
    assert src.count("dict(cell_id=k, per_launch_runtime_s=v, correct=True)") == 3
    assert "dict(cell_id=k, per_launch_runtime_s=v)" not in src.replace("correct=True)", "")


def test_energy_coverage_flags_a_timed_cell_that_was_skipped(tmp_path):
    for sub, rows in (("energy_fresh", []), ("energy_unseen", [("op_a", "small", "c1")])):
        d = tmp_path / sub
        d.mkdir()
        with (d / "application_energy_raw.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["parent_id", "regime", "candidate_id"])
            w.writeheader()
            for op, rg, cd in rows:
                w.writerow(dict(parent_id=op, regime=rg, candidate_id=cd))
    (tmp_path / "energy_unseen" / "application_energy_rejected.csv").write_text("parent_id,regime,candidate_id\nop_b,small,c2\n")
    missing, n = M.energy_coverage(tmp_path, ["ada/op_a/small/c1", "ada/op_b/small/c2", "ada/op_c/large/c1"])
    assert missing == ["ada/op_c/large/c1"] and n == 2


def test_engines_filter_and_reuse_timing_options_exist():
    a = M.build_parser().parse_args(["run", "--stage", "all", "--measure", "both", "--engines", "fresh", "--reuse-timing", "x.json"])
    assert a.engines == "fresh" and a.reuse_timing == "x.json"


def test_skip_attempted_option_exists_and_energy_recorded_ids_reads_raw_and_rejected(tmp_path):
    a = M.build_parser().parse_args(["run", "--stage", "all", "--measure", "both", "--skip-attempted", "d1", "d2"])
    assert a.skip_attempted == ["d1", "d2"]
    d = tmp_path / "energy_fresh"
    d.mkdir()
    (d / "application_energy_raw.csv").write_text("parent_id,regime,candidate_id\nop_a,small,c1\n")
    (d / "application_energy_rejected.csv").write_text("parent_id,regime,candidate_id\nop_b,small,c2\n")
    assert M.energy_recorded_ids(tmp_path, "ada") == {"ada/op_a/small/c1", "ada/op_b/small/c2"}
