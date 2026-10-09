#!/usr/bin/env python3
"""CPU-only tests for the unseen-operator runners.  No GPU, no nvcc, no NVML, no energy, no labels.

What these prove: cell enumeration equals adapters.json (9 + 6 + 6 = 21), geometry and byte arithmetic, that the
source-hash check refuses a modified or missing file, that a missing CUDA toolchain refuses with a clear message,
that the correctness oracles accept the right permutation and reject corrupt output, and that the CLI registers the
three runners with the committed harness.  What they cannot prove: that anything compiles or is correct on a GPU.
"""
from __future__ import annotations

import array
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alt_copy_runner  # noqa: E402
import alt_reduction_runner  # noqa: E402
import alt_transposefine_runner  # noqa: E402
import run_unseen_operator_energy as cli  # noqa: E402
import tile_family as tf  # noqa: E402
import unseen_common as uc  # noqa: E402

REPO = uc.REPO


# ---------------------------------------------------------------------------- cells and manifest
def test_cells_equal_manifest_21():
    cells = uc.enumerate_cells()
    assert len(cells) == 21
    manifest = json.loads(uc.ADAPTERS_JSON.read_text())
    expected = {(a["parent_id"], regime, cand) for a in manifest["adapters"] for regime in a["regimes"] for cand in a["candidates"]}
    assert {(c["parent_id"], c["regime"], c["candidate_id"]) for c in cells} == expected
    per_parent = {p: sum(1 for c in cells if c["parent_id"] == p) for p in uc.PARENTS}
    assert per_parent == {"alt_cuda_samples_reduction": 9, "alt_cuda_samples_copy": 6, "alt_cuda_samples_transposefine": 6}
    assert len({(c["parent_id"], c["regime"], c["candidate_id"]) for c in cells}) == 21


def test_manifest_shapes_and_kernels_verbatim():
    by = {(c["parent_id"], c["regime"], c["candidate_id"]): c for c in uc.enumerate_cells()}
    n = {"small": 65536, "medium": 1048576, "large": 16777216}
    for regime, size in n.items():
        assert by[("alt_cuda_samples_reduction", regime, "c1")]["n"] == size
    assert by[("alt_cuda_samples_copy", "small", "c1")]["n"] == 1048576
    assert by[("alt_cuda_samples_copy", "medium", "c1")]["n"] == 4194304
    assert by[("alt_cuda_samples_copy", "large", "c1")]["n"] == 67108864
    assert [by[("alt_cuda_samples_transposefine", r, "c1")]["n"] for r in uc.REGIME_ORDER] == [1024, 2048, 8192]
    assert [by[("alt_cuda_samples_reduction", "small", c)]["kernel_name"] for c in ("c1", "c2", "c3")] == ["reduce0", "reduce1", "reduce6"]
    assert [by[("alt_cuda_samples_reduction", "small", c)]["selector"] for c in ("c1", "c2", "c3")] == [0, 1, 6]
    assert [by[("alt_cuda_samples_copy", "small", c)]["kernel_name"] for c in ("c1", "c2")] == ["copy", "copySharedMem"]
    assert [by[("alt_cuda_samples_transposefine", "small", c)]["kernel_name"] for c in ("c1", "c2")] == [
        "transposeFineGrained", "transposeCoarseGrained"]


def test_manifest_consistency_and_pins_match_existing_runners():
    uc.check_manifest_consistency()
    import reduction_runner
    import transpose_runner
    assert uc.PINNED_REVISION == reduction_runner.PINNED_REVISION == transpose_runner.PINNED_REVISION
    assert uc.PINNED_SHA256[uc.REDUCTION_KERNEL_REL] == reduction_runner.PINNED_KERNEL_SHA256
    assert uc.PINNED_SHA256[uc.REDUCTION_UTIL_REL] == reduction_runner.PINNED_UTIL_SHA256
    assert uc.PINNED_SHA256[uc.TRANSPOSE_REL] == transpose_runner.PINNED_SHA256
    assert (uc.RTOL, uc.ATOL, uc.SEED) == (reduction_runner.RTOL, reduction_runner.ATOL, reduction_runner.SEED)


def test_manifest_consistency_detects_drift():
    manifest = json.loads(uc.ADAPTERS_JSON.read_text())
    manifest["adapters"][0]["source"]["sha256"] = "0" * 64
    with pytest.raises(uc.RunnerBlocked):
        uc.check_manifest_consistency(manifest)
    manifest = json.loads(uc.ADAPTERS_JSON.read_text())
    manifest["adapters"][1]["candidates"]["c2"]["kernel"] = "somethingElse"
    with pytest.raises(uc.RunnerBlocked):
        uc.check_manifest_consistency(manifest)


def test_nonexistent_cells_raise_config_error_named_for_exit_4():
    assert uc.UnseenOperatorConfigError.__name__.endswith("ConfigError")
    for parent, regime, cand in (("alt_cuda_samples_copy", "small", "c3"), ("alt_cuda_samples_reduction", "small", "c4"),
                                 ("alt_cuda_samples_transposefine", "tiny", "c1"), ("nope", "small", "c1")):
        with pytest.raises(uc.UnseenOperatorConfigError):
            uc.cell_config(parent, regime, cand)


# ---------------------------------------------------------------------------- geometry and bytes
EXPECTED_REDUCTION_BLOCKS = {
    ("small", "c1"): 256, ("small", "c2"): 256, ("small", "c3"): 64,
    ("medium", "c1"): 4096, ("medium", "c2"): 4096, ("medium", "c3"): 64,
    ("large", "c1"): 65536, ("large", "c2"): 65536, ("large", "c3"): 64,
}


def test_reduction_blocks_and_bytes_against_scorer_upper_bound():
    import importlib.util
    spec = importlib.util.spec_from_file_location("score_unseen_operators", uc.HERE.parent / "score_unseen_operators.py")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)  # imports only; no labels are read at import
    for (regime, cand), blocks in EXPECTED_REDUCTION_BLOCKS.items():
        plan = alt_reduction_runner.cell_plan(regime, cand)
        assert plan["blocks"] == blocks and plan["threads"] == 256
        assert plan["bytes_model"] == 4 * plan["n"] + 4 * blocks
        bound = scorer.new_code_bytes("alt_cuda_samples_reduction", plan["n"])  # 4n + 4*ceil(n/256)
        assert plan["bytes_model"] <= bound
        assert (plan["bytes_model"] == bound) == (cand in ("c1", "c2"))


def test_tile_family_geometry_and_bytes():
    for regime, dims in (("small", (1024, 1024)), ("medium", (2048, 2048)), ("large", (8192, 8192))):
        for cand in ("c1", "c2"):
            plan = alt_copy_runner.cell_plan(regime, cand)
            assert (plan["dim_x"], plan["dim_y"]) == dims
            assert plan["dim_x"] * plan["dim_y"] == plan["n"]
            assert plan["threads_per_block"] == 512 and plan["block_shape"] == [32, 16]
            assert plan["blocks"] == plan["n"] // 1024
            assert plan["bytes_model"] == 2 * 4 * plan["n"]
    for regime, side in (("small", 1024), ("medium", 2048), ("large", 8192)):
        for cand in ("c1", "c2"):
            plan = alt_transposefine_runner.cell_plan(regime, cand)
            assert (plan["dim_x"], plan["dim_y"]) == (side, side) and plan["blocks"] == (side // 32) ** 2
            assert plan["bytes_model"] == 2 * 4 * side * side
    assert alt_transposefine_runner.cell_plan("small", "c1")["variant"] == 6
    assert alt_transposefine_runner.cell_plan("small", "c2")["variant"] == 5
    assert alt_copy_runner.cell_plan("small", "c1")["variant"] == 0
    assert alt_copy_runner.cell_plan("small", "c2")["variant"] == 1
    with pytest.raises(uc.UnseenOperatorConfigError):
        tf.tile_dims(tf.COPY_PARENT, 3 * 1024 * 1024)      # neither square nor power of two
    with pytest.raises(uc.UnseenOperatorConfigError):
        uc.reduction_blocks(2, 65536)                        # catalog kernels are not part of this test
    with pytest.raises(uc.UnseenOperatorConfigError):
        uc.reduction_blocks(0, 1000)


def test_oracle_env_selection(monkeypatch):
    monkeypatch.delenv(tf.ORACLE_ENV, raising=False)
    assert tf.cell_plan(tf.TRANSPOSEFINE_PARENT, "small", "c1")["oracle"] == "kernel_semantics"
    assert tf.cell_plan(tf.TRANSPOSEFINE_PARENT, "small", "c1", {tf.ORACLE_ENV: "full_transpose"})["oracle"] == "full_transpose"
    assert tf.cell_plan(tf.COPY_PARENT, "small", "c1", {tf.ORACLE_ENV: "kernel_semantics"})["oracle"] == "identity"
    with pytest.raises(uc.RunnerBlocked):
        tf.cell_plan(tf.TRANSPOSEFINE_PARENT, "small", "c1", {tf.ORACLE_ENV: "anything_that_passes"})


# ---------------------------------------------------------------------------- source verification and toolchain
def _fake_git_head(monkeypatch, head: str):
    real_run = subprocess.run

    def run(argv, *a, **k):
        if argv[:1] == ["git"] and "rev-parse" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=head + "\n", stderr="")
        return real_run(argv, *a, **k)

    monkeypatch.setattr(uc.subprocess, "run", run)


def _fake_checkout(tmp_path: Path, rel_to_content: dict) -> Path:
    for rel, content in rel_to_content.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return tmp_path


def test_source_hash_check_refuses_modified_and_missing_files(tmp_path, monkeypatch):
    _fake_git_head(monkeypatch, uc.PINNED_REVISION)
    root = _fake_checkout(tmp_path, {uc.TRANSPOSE_REL: b"not the pinned bytes"})
    with pytest.raises(uc.RunnerBlocked, match="SHA-256 mismatch"):
        uc.verify_source(root, (uc.TRANSPOSE_REL,))
    with pytest.raises(uc.RunnerBlocked, match="missing pinned source"):
        uc.verify_source(tmp_path / "empty", (uc.TRANSPOSE_REL,))
    # a matching hash passes: pin a synthetic file and confirm the same code accepts the right bytes
    content = b"synthetic pinned bytes"
    monkeypatch.setitem(uc.PINNED_SHA256, "x/pinned.cu", uc.hashlib.sha256(content).hexdigest())
    root2 = _fake_checkout(tmp_path / "ok", {"x/pinned.cu": content})
    _, revision, hashes = uc.verify_source(root2, ("x/pinned.cu",))
    assert revision == uc.PINNED_REVISION and hashes["x/pinned.cu"] == uc.PINNED_SHA256["x/pinned.cu"]
    (root2 / "x/pinned.cu").write_bytes(content + b"\n")
    with pytest.raises(uc.RunnerBlocked, match="SHA-256 mismatch"):
        uc.verify_source(root2, ("x/pinned.cu",))


def test_wrong_revision_and_non_git_source_refused(tmp_path, monkeypatch):
    _fake_git_head(monkeypatch, "0" * 40)
    root = _fake_checkout(tmp_path, {uc.TRANSPOSE_REL: b"x"})
    with pytest.raises(uc.RunnerBlocked, match="expected revision"):
        uc.verify_source(root, (uc.TRANSPOSE_REL,))
    monkeypatch.undo()
    with pytest.raises(uc.RunnerBlocked, match="not a readable Git checkout|expected revision"):
        uc.check_revision(tmp_path)  # tmp_path is not a git repo (or a repo at another revision)


def test_kernel_symbol_checks():
    good = "\n".join(f"__global__ void {n}(float *odata, float *idata, int width, int height)\n{{" for n in tf.KERNEL_NAMES)
    uc.check_four_arg_kernels(good, tf.KERNEL_NAMES)
    with pytest.raises(uc.RunnerBlocked, match="transposeFineGrained"):
        uc.check_four_arg_kernels(good.replace("transposeFineGrained", "transposeFine"), tf.KERNEL_NAMES)
    with pytest.raises(uc.RunnerBlocked, match="copy"):   # an extra nreps parameter would not link
        uc.check_four_arg_kernels(good.replace("void copy(float *odata, float *idata, int width, int height)",
                                               "void copy(float *odata, float *idata, int width, int height, int nreps)"),
                                  tf.KERNEL_NAMES)
    reduce_src = "template <class T>\n__global__ void reduce0(T *g_idata, T *g_odata, unsigned int n) {}\n" \
                 "template <class T>\n__global__ void\nreduce1(T *a) {}\n"
    with pytest.raises(uc.RunnerBlocked):   # reduce1 split across lines and reduce6 absent
        uc.check_reduce_kernels(reduce_src, ("reduce0", "reduce1", "reduce6"))
    uc.check_reduce_kernels(reduce_src + "__global__ void reduce6(T *g) {}\n__global__ void reduce1(T *g) {}\n",
                            ("reduce0", "reduce1", "reduce6"))


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.delenv("HARNESS_NVCC", raising=False)
    monkeypatch.delenv("HARNESS_CUDA_ARCH", raising=False)
    monkeypatch.setattr(uc.cuda_build_target, "LEGACY_NVCC_CANDIDATES", ())
    monkeypatch.setattr(uc.cuda_build_target.shutil, "which", lambda name: None)


def _fake_verified_source(monkeypatch, module_source_rels, text):
    def fake_verify(source_root, rels):
        return Path(source_root), uc.PINNED_REVISION, {r: uc.PINNED_SHA256[r] for r in rels}
    monkeypatch.setattr(uc, "verify_source", fake_verify)
    monkeypatch.setattr(uc, "read_source_text", lambda root, rel: text)


@pytest.mark.parametrize("module,text", [
    (alt_reduction_runner, "__global__ void reduce0(a) {} __global__ void reduce1(a) {} __global__ void reduce6(a) {}"),
    (alt_copy_runner, "\n".join(f"__global__ void {n}(float *o, float *i, int w, int h)" for n in tf.KERNEL_NAMES)),
    (alt_transposefine_runner, "\n".join(f"__global__ void {n}(float *o, float *i, int w, int h)" for n in tf.KERNEL_NAMES)),
])
def test_runner_refuses_without_cuda_toolchain(module, text, no_cuda, monkeypatch, tmp_path):
    _fake_verified_source(monkeypatch, None, text)
    with pytest.raises(uc.RunnerBlocked, match="no nvcc found"):
        module.prepare_binary_energy_context(tmp_path, "small", "c1", tmp_path)
    with pytest.raises(uc.RunnerBlocked, match="no nvcc found"):
        module.run_candidate(tmp_path, "small", "c1", tmp_path)


def test_runner_refuses_wrong_arch_build(monkeypatch, tmp_path):
    fake_nvcc = tmp_path / "nvcc"
    fake_nvcc.write_text("")
    monkeypatch.setenv("HARNESS_NVCC", str(fake_nvcc))
    monkeypatch.setenv("HARNESS_CUDA_ARCH", "sm_120")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(uc.cuda_build_target, "live_compute_arch", lambda blocked: "sm_89")
    with pytest.raises(uc.RunnerBlocked, match="does not match the target GPU"):
        uc.find_nvcc()


# ---------------------------------------------------------------------------- oracles versus literal kernel index math
def emulate_tile_kernel(name: str, inp, n_x: int, n_y: int):
    """Thread-by-thread emulation of the sample's index arithmetic (TILE_DIM 32, BLOCK_ROWS 16, blocks (32,16)).
    Written from the sample's documented kernels; this checks the oracles implement that arithmetic, it is not
    evidence about the pinned bytes."""
    tile, rows = 32, 16
    out = array.array("f", [float("nan")] * (n_x * n_y))
    for by in range(n_y // tile):
        for bx in range(n_x // tile):
            block = [[0.0] * (tile + 1) for _ in range(tile)]
            for ty in range(rows):
                for tx in range(tile):
                    x, y = bx * tile + tx, by * tile + ty
                    index = x + n_x * y
                    if name == "copy":
                        for i in range(0, tile, rows):
                            out[index + i * n_x] = inp[index + i * n_x]
                    elif name == "transposeFineGrained":
                        for i in range(0, tile, rows):
                            block[ty + i][tx] = inp[index + i * n_x]
                    elif name == "transposeCoarseGrained":
                        for i in range(0, tile, rows):
                            block[ty + i][tx] = inp[index + i * n_x]
            if name == "copy":
                continue
            for ty in range(rows):
                for tx in range(tile):
                    if name == "transposeFineGrained":
                        index = bx * tile + tx + n_x * (by * tile + ty)
                        for i in range(0, tile, rows):
                            out[index + i * n_y] = block[tx][ty + i]
                    else:
                        index_out = (by * tile + tx) + n_y * (bx * tile + ty)
                        for i in range(0, tile, rows):
                            out[index_out + i * n_y] = block[ty + i][tx]
    return out


def test_oracles_match_emulated_kernels_and_reject_corruption():
    n = 96                                   # 3 x 3 tiles: catches tile-index mix-ups a 2 x 2 grid could hide
    inp = tf.transpose_input(n * n)
    fine = emulate_tile_kernel("transposeFineGrained", inp, n, n)
    coarse = emulate_tile_kernel("transposeCoarseGrained", inp, n, n)
    copied = emulate_tile_kernel("copy", inp, n, n)
    full = array.array("f", [inp[c * n + r] for r in range(n) for c in range(n)])
    assert tf.matches_fine_grained_partial(inp, fine, n) and not tf.matches_fine_grained_partial(inp, coarse, n)
    assert tf.matches_coarse_grained_partial(inp, coarse, n) and not tf.matches_coarse_grained_partial(inp, fine, n)
    assert tf.matches_full_transpose(inp, full, n)
    assert not tf.matches_full_transpose(inp, fine, n) and not tf.matches_full_transpose(inp, coarse, n)
    assert tf.matches_identity(inp, copied) and not tf.matches_identity(inp, fine)
    # the default oracle accepts each kernel only its own permutation; the diagnostic full-transpose oracle rejects both
    plan_fine = dict(tf.cell_plan(tf.TRANSPOSEFINE_PARENT, "small", "c1"), dim_x=n, dim_y=n)
    plan_coarse = dict(tf.cell_plan(tf.TRANSPOSEFINE_PARENT, "small", "c2"), dim_x=n, dim_y=n)
    assert plan_fine["oracle"] == plan_coarse["oracle"] == "kernel_semantics"
    assert tf.oracle_passes(plan_fine, inp, fine) and tf.oracle_passes(plan_coarse, inp, coarse)
    assert not tf.oracle_passes(plan_fine, inp, coarse) and not tf.oracle_passes(plan_coarse, inp, fine)
    plan_fine["oracle"] = plan_coarse["oracle"] = "full_transpose"
    assert not tf.oracle_passes(plan_fine, inp, fine) and not tf.oracle_passes(plan_coarse, inp, coarse)
    off_by_one = array.array("f", fine)
    off_by_one[5] += 1.0
    assert not tf.matches_fine_grained_partial(inp, off_by_one, n)
    assert not tf.matches_coarse_grained_partial(inp, off_by_one, n)
    corrupt = array.array("f", full)
    corrupt[n * n - 1] += 1.0
    assert not tf.matches_full_transpose(inp, corrupt, n)
    assert not tf.matches_full_transpose(inp, full[:-1], n)


def test_sentinel_statistics_expose_partial_writes():
    n = 64
    inp = tf.copy_input(n * n)
    half = array.array("f", inp)
    nan_word = array.array("I", [tf.SENTINEL_WORD]).tobytes()
    nan_value = array.array("f")
    nan_value.frombytes(nan_word)
    for r in range(n):
        if r % 32 >= 16:                       # what a copy that only covers BLOCK_ROWS rows of each tile would leave
            half[r * n:(r + 1) * n] = array.array("f", [nan_value[0]] * n)
    plan = dict(tf.cell_plan(tf.COPY_PARENT, "small", "c2"), dim_x=n, dim_y=n)
    diag = tf.diagnostics(plan, inp, half)
    assert diag["matches_identity"] is False
    assert diag["sentinel"]["written_fraction"] == 0.5
    assert diag["sentinel"]["rows_with_unwritten_by_row_mod_32"][16:] == [2] * 16
    assert diag["sentinel"]["rows_with_unwritten_by_row_mod_32"][:16] == [0] * 16
    full = tf.diagnostics(plan, inp, array.array("f", inp))
    assert full["matches_identity"] and full["sentinel"]["written_fraction"] == 1.0


def test_input_builders_are_deterministic_and_exact():
    assert list(tf.copy_input(100)) == list(tf.copy_input(100))
    assert all(-1.0 <= v <= 1.0 for v in tf.copy_input(100))
    m = tf.TRANSPOSE_INPUT_MODULUS
    x = tf.transpose_input(m + 5)
    assert len(x) == m + 5 and x[m - 1] == float(m - 1) and x[m] == 0.0 and x[m + 4] == 4.0
    assert tf.transpose_input(10)[9] == 9.0


# ---------------------------------------------------------------------------- energy contexts (fake build, real oracle)
def _fake_build(monkeypatch, module, text):
    _fake_verified_source(monkeypatch, None, text)
    monkeypatch.setattr(uc, "find_nvcc", lambda: "/fake/nvcc")
    monkeypatch.setattr(uc, "build_arch", lambda: "sm_120")
    target = tf if module is not alt_reduction_runner else alt_reduction_runner
    monkeypatch.setattr(target, "build_binary", lambda root, workdir, nvcc: workdir / "fake_binary")


def test_reduction_context_check_and_accumulation_metrics(monkeypatch, tmp_path):
    _fake_build(monkeypatch, alt_reduction_runner,
                "__global__ void reduce0(a) {} __global__ void reduce1(a) {} __global__ void reduce6(a) {}")
    for regime, cand in (("small", "c1"), ("small", "c3")):
        work = tmp_path / f"{regime}{cand}"
        work.mkdir()
        ctx = alt_reduction_runner.prepare_binary_energy_context(tmp_path, regime, cand, work)
        assert ctx.parent_id == "alt_cuda_samples_reduction" and ctx.candidate_id == cand
        assert ctx.controls["blocks"] == EXPECTED_REDUCTION_BLOCKS[(regime, cand)] and ctx.controls["threads"] == 256
        assert ctx.argv_prefix[:4] == ["65536", "256", str(ctx.controls["blocks"]), str(ctx.controls["which_kernel"])]
        values = array.array("f")
        with open(ctx.argv_prefix[4], "rb") as f:
            values.fromfile(f, 65536)
        want = sum(values)
        out = Path(ctx.argv_prefix[5])
        out.write_bytes(array.array("f", [want]).tobytes())
        assert ctx.check() is True
        acc = ctx.controls["accumulation_check"]           # lands in controls_json (harness serialises after check())
        assert acc["abs_error"] < acc["tolerance_abs"] and acc["rel_error"] < 1e-6 and acc["l1_norm"] > 0
        json.dumps(ctx.controls)
        out.write_bytes(array.array("f", [want + 1.0]).tobytes())
        assert ctx.check() is False
        out.write_bytes(array.array("f", [float("nan")]).tobytes())
        assert ctx.check() is False
        out.write_bytes(b"")
        assert ctx.check() is False


@pytest.mark.parametrize("module,parent", [(alt_copy_runner, "alt_cuda_samples_copy"),
                                           (alt_transposefine_runner, "alt_cuda_samples_transposefine")])
def test_tile_context_check(module, parent, monkeypatch, tmp_path):
    monkeypatch.delenv(tf.ORACLE_ENV, raising=False)
    text = "\n".join(f"__global__ void {n}(float *o, float *i, int w, int h)" for n in tf.KERNEL_NAMES)
    _fake_build(monkeypatch, module, text)
    ctx = module.prepare_binary_energy_context(tmp_path, "small", "c1", tmp_path)
    assert ctx.parent_id == parent and ctx.controls["threads_per_block"] == 512 and ctx.controls["blocks"] == 1024
    dim_x, dim_y, variant = (int(v) for v in ctx.argv_prefix[:3])
    inp = array.array("f")
    with open(ctx.argv_prefix[3], "rb") as f:
        inp.fromfile(f, dim_x * dim_y)
    out = Path(ctx.argv_prefix[4])
    right = (inp if parent.endswith("copy") else
             emulate_tile_kernel("transposeFineGrained", inp, dim_x, dim_y))   # c1 = fine-grained: its own permutation
    out.write_bytes(right.tobytes())
    assert ctx.check() is True
    wrong = array.array("f", right)
    wrong[3] += 1.0
    out.write_bytes(wrong.tobytes())
    assert ctx.check() is False
    out.write_bytes(right.tobytes()[:-4])
    assert ctx.check() is False
    if parent.endswith("copy"):                     # the identity oracle rejects a transposed layout and vice versa
        out.write_bytes(array.array("f", [inp[c * dim_x + r] for r in range(dim_x) for c in range(dim_x)]).tobytes())
        assert ctx.check() is False


# ---------------------------------------------------------------------------- drivers
def test_driver_template_contracts():
    d = tf.DRIVER_TEMPLATE
    assert "int main(int argc" in d and "-Dmain=" not in d
    assert "cudaStreamBeginCapture(stream" in d and "cudaStreamBeginCapture(0" not in d
    assert "cudaMemset(d_out, 0xFF" in d
    assert d.count("__global__ void") == 4
    for name in tf.KERNEL_NAMES:
        assert f"__global__ void {name}(float *odata, float *idata, int width, int height);" in d
    for case in ("case 0: copy<<<", "case 1: copySharedMem<<<", "case 5: transposeCoarseGrained<<<", "case 6: transposeFineGrained<<<"):
        assert case in d
    assert 'argc != 6 && argc != 12' in d and "windows.csv" in d
    assert "cudaStreamDestroy(stream)" in d          # private stream created here, so destroying it is correct
    import reduction_runner
    assert "cudaStreamDestroy" not in reduction_runner.DRIVER_TEMPLATE   # reused as-is: per-thread stream, never destroyed


# ---------------------------------------------------------------------------- CLI and registration
def _run_cli(monkeypatch, capsys, argv):
    monkeypatch.setenv("HARNESS_CUDA_ARCH", "")   # the harness overwrites it; monkeypatch restores it afterwards
    monkeypatch.setattr(sys, "argv", ["run_unseen_operator_energy.py"] + argv)
    try:
        rc = cli.main(argv)
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
        return rc, capsys.readouterr().err + str(exc.code)
    return rc, capsys.readouterr().err


def test_cli_registers_runners_and_maps_exit_codes(monkeypatch, capsys, tmp_path):
    common = ["--dry-run", "--source-root", str(tmp_path / "nonexistent"), "--window-target-seconds", "15",
              "--session", "1", "--out-dir", str(tmp_path / "out")]
    rc, err = _run_cli(monkeypatch, capsys, ["--runner", "alt_copy", "--regime", "small", "--candidate", "c1"] + common)
    assert rc == 3 and "RUNNER_BLOCKED: missing pinned source" in err                # blocked, not a traceback
    rc, err = _run_cli(monkeypatch, capsys, ["--runner", "alt_copy", "--regime", "small", "--candidate", "c3"] + common)
    assert rc == 4 and "NO_SUCH_CELL" in err
    for runner in ("alt_reduction", "alt_transposefine"):
        rc, err = _run_cli(monkeypatch, capsys, ["--runner", runner, "--regime", "small", "--candidate", "c1"] + common)
        assert rc == 3 and "RUNNER_BLOCKED" in err, (runner, rc, err)
    import run_application_energy as harness_cli
    for runner, module in uc.RUNNER_MODULES.items():
        assert harness_cli.BINARY_RUNNER_MODULES[runner] == module
    assert {"cub_block", "cub_device", "transpose", "reduction", "copy"} <= set(harness_cli.BINARY_RUNNER_MODULES)


def test_cli_requires_out_dir_and_refuses_dataset_and_sealed_dirs(monkeypatch, capsys, tmp_path):
    base = ["--runner", "alt_copy", "--dry-run", "--source-root", "x", "--regime", "small", "--candidate", "c1",
            "--window-target-seconds", "15", "--session", "1"]
    rc, err = _run_cli(monkeypatch, capsys, base)
    assert rc != 0 and "--out-dir is required" in err
    rc, err = _run_cli(monkeypatch, capsys, base + ["--out-dir", str(uc.RESET_DIR / "application_energy_raw")])
    assert rc != 0 and "development dataset" in err
    rc, err = _run_cli(monkeypatch, capsys, base + ["--out-dir", str(uc.RESET_DIR / "sealed_labels" / "x")])
    assert rc != 0 and "sealed" in err
    rc, err = _run_cli(monkeypatch, capsys, base + ["--out-dir", str(tmp_path / "FINAL-thing")])
    assert rc != 0 and "sealed" in err


def test_print_plan_lists_all_21_cells_and_touches_nothing(monkeypatch, capsys, tmp_path):
    rc, _ = _run_cli(monkeypatch, capsys, ["--print-plan", "--source-root", "/cs", "--out-dir", str(tmp_path / "o"),
                                           "--platform", "blackwell", "--session", "1"])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert rc == 0 and not (tmp_path / "o").exists()
    # the first _run_cli call consumed stdout; run again capturing it
    monkeypatch.setattr(sys, "argv", ["x"])
    cli.main(["--print-plan", "--source-root", "/cs", "--out-dir", str(tmp_path / "o"), "--platform", "blackwell",
              "--session", "2", "--reverse"])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 21 and len(set(lines)) == 21
    assert all("--gpu-index 1" in ln and "--platform blackwell" in ln and "--session 2" in ln for ln in lines)
    assert "alt_transposefine" in lines[0] and "alt_reduction" in lines[-1]      # reversed order
    runners = {ln.split("--runner ")[1].split()[0] for ln in lines}
    assert runners == set(uc.RUNNER_MODULES)


def test_no_committed_shared_file_is_imported_with_side_effects_on_path():
    # the runners reuse, never edit: the new files live only under tiresias/framework/unseen_operators/runners
    for module in (alt_copy_runner, alt_reduction_runner, alt_transposefine_runner, tf, uc):
        assert Path(module.__file__).resolve().parent == uc.HERE
