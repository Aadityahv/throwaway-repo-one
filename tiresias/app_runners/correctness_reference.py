#!/usr/bin/env python3
"""CPU semantic references for catalog parents — M237 fixing commit.

Fixes C5 (tiresias/review/ACI_ACCEPTANCE_2026-09-14.md): the previous
version checked properties of arbitrary examples (`sum(x)*4 == sum(x)*4`,
summation order with no segments) rather than a real parent's actual
input->expected-output relationship. This version implements a real,
reusable input->expected-output reference function per parent for the six
repository-local parents plus all pinned upstream parents (the complete
18-parent catalog now gives source-audited formulas). Each reference is
callable on arbitrary inputs; every validator compares the
complete output buffer and rejects malformed or non-finite/corrupted results.

These remain CPU semantic references, not GPU-kernel correctness evidence —
the printed banner says so, unchanged from before. No CPU reference here is
GPU-kernel correctness evidence.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path


def close(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# ---------------------------------------------------------------------------
# Real parent reference functions. Each takes the parent's own real native
# inputs (all-zero payload, exactly as every kernel in this project memsets
# its input, per M88's decoupled-accumulator convention) and returns the
# exact value the real device kernel's own host-side check computes, so a
# harness owner can call the SAME function against a real device run's sink
# output instead of trusting the kernel's self-check alone.

def ref_tier_scaffold(threads: int, iterations: int) -> list[int]:
    """train_p1_shared_generator / train_multistream_load (streams=1): sink[t]
    = t + sum(0..iterations-1) = t + iterations*(iterations-1)/2, transcribed
    from predictor_vector_load.cu's own host check (L337-355)."""
    repeat_sum = iterations * (iterations - 1) // 2
    return [t + repeat_sum for t in range(threads)]


def ref_multistream(threads: int, iterations: int, streams: int) -> list[int]:
    """All k streams are zeroed, so per predictor_multistream_load.cu L313-317
    the closed form is IDENTICAL to the single-stream tier scaffold — streams
    only change executed_loads, not the checked value."""
    return ref_tier_scaffold(threads, iterations)


def _segment_length(segment: int, run_length: int, run_spread: int) -> int:
    mixed = (segment * 2654435761) & 0xFFFFFFFF
    return run_length + ((mixed >> 13) % run_spread)


def ref_segmented(threads: int, iterations: int, run_length: int, run_spread: int) -> list[int]:
    """Transcription of predictor_segmented_load.cu L338-355: sink[t] = t +
    sum over steps of segment_length(t + step*threads, ...)."""
    offset_entries = threads
    out = []
    for thread in range(threads):
        elements = 0
        for step in range(iterations):
            segment = thread + step * threads
            elements += _segment_length(segment % offset_entries, run_length, run_spread)
        out.append(thread + elements)
    return out


def ref_shared_stage(threads: int, chunks: int, reuse: int, blocks: int = 1,
                     batches: int = 1, launches: int = 1) -> list[int]:
    """Full sink buffer for predictor_shared_stage_load.cu.

    The source allocates blocks*threads uint32 outputs. Zero staged input
    contributes one increment per shared read, so each launch writes
    sink[block,thread] = thread + chunks*reuse*batches. Repeated launches
    overwrite the same value and therefore do not multiply the final buffer.
    """
    for name, value in (("threads", threads), ("chunks", chunks), ("reuse", reuse),
                        ("blocks", blocks), ("batches", batches), ("launches", launches)):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if threads > 1024:
        raise ValueError("threads exceeds CUDA block limit")
    increment = chunks * reuse * batches
    return [thread + increment for _block in range(blocks) for thread in range(threads)]


def shared_stage_output_matches(threads: int, chunks: int, reuse: int,
                                blocks: int, batches: int, launches: int,
                                output: list[int]) -> bool:
    if not all(isinstance(v, int) for v in output):
        return False
    try:
        expected = ref_shared_stage(threads, chunks, reuse, blocks, batches, launches)
    except ValueError:
        return False
    return output == expected


def ref_triad(n: int, launches: int = 1, batches: int = 1,
              tile_dim: int = 16) -> list[float]:
    """Full output buffer for e2e_tile_triad_calibration.cu.

    The source initializes a, b and c to one and writes every in-range
    element as 1 + .25 - .125 = 1.125. launches repeat the same overwrite;
    batches is rejected because it is not a native source control.
    """
    if tile_dim != 16 or not isinstance(n, int) or n <= 0 or n % tile_dim:
        raise ValueError("triad requires positive n divisible by TILE_DIM=16")
    if launches <= 0 or batches != 1:
        raise ValueError("triad admitted controls require TILE_DIM=16, batches=1, launches>0")
    return [1.125] * (n * n)


def triad_output_matches(n: int, launches: int, batches: int, tile_dim: int,
                         output: list[float]) -> bool:
    if not all(math.isfinite(v) for v in output):
        return False
    try:
        expected = ref_triad(n, launches, batches, tile_dim)
    except ValueError:
        return False
    return len(output) == len(expected) and all(close(got, want, 2e-6)
                                               for got, want in zip(output, expected))


def transpose_input(n: int) -> list[float]:
    """Exact host initialization in e2e_transpose.cu."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be positive")
    return [float(i % 8192) for i in range(n * n)]


def transpose_expected_output(n: int) -> list[float]:
    """Return the full output buffer expected from output[row,col]=input[col,row].

    This deliberately materializes values rather than merely proving index
    arithmetic. A harness may use `transpose_output_matches` to compare a
    device DtoH buffer without allocating a second expected buffer.
    """
    source = transpose_input(n)
    return [source[col * n + row] for row in range(n) for col in range(n)]


def transpose_output_matches(n: int, output: list[float]) -> bool:
    """Full-grid output-buffer comparison; false for malformed/corrupt output."""
    if len(output) != n * n:
        return False
    for row in range(n):
        for col in range(n):
            if output[row * n + col] != float((col * n + row) % 8192):
                return False
    return True


def ref_triton_vector_add(x: list[float], y: list[float]) -> list[float]:
    """add_kernel (python/tutorials/01-vector-add.py, fetched 2026-09-14):
    output = x + y elementwise. Real reusable reference, not a property."""
    assert len(x) == len(y)
    return [xi + yi for xi, yi in zip(x, y)]


def vector_add_input(n: int, seed: int = 20260914) -> tuple[list[float], list[float]]:
    """Deterministic FP32-range input pair for vector-add cells. Same fixed seed
    convention as softmax_input; distinct lengths across regimes, documented here."""
    if n <= 0:
        raise ValueError("vector-add input length must be positive")
    rng = random.Random(seed)
    return ([rng.uniform(-1.0, 1.0) for _ in range(n)],
            [rng.uniform(-1.0, 1.0) for _ in range(n)])


def triton_vector_add_output_matches(output: list[float], x: list[float], y: list[float],
                                     rtol: float = 1e-5, atol: float = 1e-6) -> bool:
    """Full-output check for train_triton_vector_add cells at catalog tolerance
    (rtol=1e-5, atol=1e-6; exact shape; finite). Stricter than the legacy
    abs-1e-5 vector_output_matches for small magnitudes -- use this for cells."""
    if len(output) != len(x) or len(y) != len(x):
        return False
    if not all(math.isfinite(v) for v in output):
        return False
    expected = ref_triton_vector_add(x, y)
    return all(abs(got - want) <= atol + rtol * abs(want)
               for got, want in zip(output, expected))


def softmax_input(rows: int, cols: int, seed: int = 20260914) -> list[float]:
    """Deterministic finite FP32-like input for the pinned tutorial's rows."""
    if not isinstance(rows, int) or not isinstance(cols, int) or rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive integers")
    return [math.sin((seed + 17 * i) * 0.001) * 3.0 + ((i % 11) - 5) * 0.07
            for i in range(rows * cols)]


def ref_triton_softmax(rows: int, cols: int, values: list[float] | None = None) -> list[float]:
    """Full row-wise reference for Triton ``softmax_kernel``.

    The source loads one masked row, subtracts its row maximum, applies exp,
    reduces the denominator, divides, and stores all unmasked columns. Padded
    BLOCK_SIZE lanes are masked and never appear in this logical output.
    """
    expected_len = rows * cols
    x = softmax_input(rows, cols) if values is None else list(values)
    if len(x) != expected_len:
        raise ValueError(f"softmax input length {len(x)} != rows*cols {expected_len}")
    if not all(math.isfinite(v) for v in x):
        raise ValueError("softmax input must be finite")
    out: list[float] = []
    for row in range(rows):
        source = x[row * cols:(row + 1) * cols]
        row_max = max(source)
        numerators = [math.exp(v - row_max) for v in source]
        denominator = sum(numerators)
        out.extend(v / denominator for v in numerators)
    return out


def triton_softmax_output_matches(rows: int, cols: int, output: list[float],
                                  values: list[float] | None = None,
                                  rtol: float = 2e-5, atol: float = 2e-5) -> bool:
    """Compare the complete logical output buffer; false for bad shape/data."""
    expected_len = rows * cols
    if len(output) != expected_len or not all(math.isfinite(v) for v in output):
        return False
    expected = ref_triton_softmax(rows, cols, values)
    return all(abs(got - want) <= atol + rtol * abs(want)
               for got, want in zip(output, expected))


def ref_pytorch_softmax(rows: int, cols: int, row_stride: int,
                        values: list[float] | None = None) -> list[float]:
    """Reference for ATen spatial softmax with a legal padded row stride."""
    if row_stride < cols:
        raise ValueError("row_stride must cover the logical softmax row")
    x = softmax_input(rows, row_stride) if values is None else list(values)
    if len(x) != rows * row_stride or not all(math.isfinite(v) for v in x):
        raise ValueError("padded softmax input has wrong shape or non-finite data")
    out: list[float] = []
    for row in range(rows):
        source = x[row * row_stride:row * row_stride + cols]
        row_max = max(source)
        nums = [math.exp(v - row_max) for v in source]
        denom = sum(nums)
        out.extend(v / denom for v in nums)
    return out


def pytorch_softmax_output_matches(rows: int, cols: int, row_stride: int,
                                   output: list[float],
                                   values: list[float] | None = None) -> bool:
    if len(output) != rows * cols or not all(math.isfinite(v) for v in output):
        return False
    expected = ref_pytorch_softmax(rows, cols, row_stride, values)
    return all(abs(got - want) <= 2e-5 + 2e-5 * abs(want)
               for got, want in zip(output, expected))


def ref_sum(values: list[float]) -> list[float]:
    """Full logical output for CUDA Samples/CUB scalar reductions."""
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("reduction input must be a non-empty finite buffer")
    return [sum(values)]


def ref_cuda_samples_reduction(values: list[float]) -> list[float]:
    """Reference for CUDA Samples reduce2--reduce5 plus host final sum."""
    return ref_sum(values)


def ref_cub_block_reduce(values: list[float]) -> list[float]:
    """Reference for the pinned CCCL BlockReduce test's scalar output."""
    return ref_sum(values)


def ref_cub_device_reduce(values: list[float]) -> list[float]:
    """Reference for the pinned CCCL DeviceReduce::Sum scalar output."""
    return ref_sum(values)


def scalar_output_matches(output: list[float], expected: list[float],
                          rtol: float = 1e-5, atol: float = 1e-5) -> bool:
    return len(output) == 1 and len(expected) == 1 and all(
        math.isfinite(v) for v in output) and abs(output[0] - expected[0]) <= atol + rtol * abs(expected[0])


def ref_cuda_transpose(rows: int, cols: int, values: list[float] | None = None) -> list[float]:
    expected_len = rows * cols
    x = [float(i % 8192) for i in range(expected_len)] if values is None else list(values)
    if len(x) != expected_len or not all(math.isfinite(v) for v in x):
        raise ValueError("transpose input has wrong shape or non-finite data")
    return [x[col * cols + row] for row in range(rows) for col in range(cols)]


def cuda_transpose_output_matches(rows: int, cols: int, output: list[float],
                                  values: list[float] | None = None) -> bool:
    if len(output) != rows * cols or not all(math.isfinite(v) for v in output):
        return False
    expected = ref_cuda_transpose(rows, cols, values)
    return all(got == want for got, want in zip(output, expected))


def ref_layer_norm(rows: int, cols: int, values: list[float] | None = None,
                   weight: list[float] | None = None, bias: list[float] | None = None,
                   eps: float = 1e-5) -> list[float]:
    expected_len = rows * cols
    x = softmax_input(rows, cols, seed=20260915) if values is None else list(values)
    w = [1.0] * cols if weight is None else list(weight)
    b = [0.0] * cols if bias is None else list(bias)
    if len(x) != expected_len or len(w) != cols or len(b) != cols:
        raise ValueError("LayerNorm input/affine shape mismatch")
    output: list[float] = []
    for row in range(rows):
        source = x[row * cols:(row + 1) * cols]
        mean = sum(source) / cols
        var = sum((v - mean) ** 2 for v in source) / cols
        inv_std = 1.0 / math.sqrt(var + eps)
        output.extend((v - mean) * inv_std * w[col] + b[col] for col, v in enumerate(source))
    return output


def layer_norm_output_matches(rows: int, cols: int, output: list[float],
                              values: list[float] | None = None,
                              weight: list[float] | None = None,
                              bias: list[float] | None = None,
                              rtol: float = 2e-5, atol: float = 2e-5) -> bool:
    if len(output) != rows * cols or not all(math.isfinite(v) for v in output):
        return False
    expected = ref_layer_norm(rows, cols, values, weight, bias)
    return all(abs(got - want) <= atol + rtol * abs(want) for got, want in zip(output, expected))


def ref_pytorch_layer_norm(rows: int, cols: int, row_stride: int,
                           values: list[float] | None = None) -> list[float]:
    """Reference for ATen rowwise LayerNorm with a legal padded input stride."""
    if row_stride < cols:
        raise ValueError("row_stride must cover the logical LayerNorm row")
    x = softmax_input(rows, row_stride, seed=20260915) if values is None else list(values)
    if len(x) != rows * row_stride or not all(math.isfinite(v) for v in x):
        raise ValueError("padded LayerNorm input has wrong shape or non-finite data")
    logical = [x[row * row_stride + col] for row in range(rows) for col in range(cols)]
    return ref_layer_norm(rows, cols, logical)


def pytorch_layer_norm_output_matches(rows: int, cols: int, row_stride: int,
                                      output: list[float],
                                      values: list[float] | None = None) -> bool:
    if len(output) != rows * cols or not all(math.isfinite(v) for v in output):
        return False
    expected = ref_pytorch_layer_norm(rows, cols, row_stride, values)
    return all(close(got, want) for got, want in zip(output, expected))


def ref_index_select(num_rows: int, feature_dim: int, indices: list[int]) -> list[float]:
    if not indices or not all(0 <= i < num_rows for i in indices):
        raise ValueError("indices must be non-empty and in range")
    source = [float(row * feature_dim + col) for row in range(num_rows) for col in range(feature_dim)]
    return [source[i * feature_dim + col] for i in indices for col in range(feature_dim)]


def index_select_output_matches(num_rows: int, feature_dim: int, indices: list[int],
                                output: list[float]) -> bool:
    if len(output) != len(indices) * feature_dim or not all(math.isfinite(v) for v in output):
        return False
    expected = ref_index_select(num_rows, feature_dim, indices)
    return output == expected


def ref_pytorch_index_select(num_rows: int, feature_dim: int, indices: list[int]) -> list[float]:
    """Reference for ATen Indexing.cu forward index_select."""
    return ref_index_select(num_rows, feature_dim, indices)


def ref_xformers_index_select(num_rows: int, feature_dim: int, indices: list[int]) -> list[float]:
    """Reference for xFormers k_index_select_cat.py's gathered output."""
    return ref_index_select(num_rows, feature_dim, indices)


def ref_vector_add(x: list[float], y: list[float]) -> list[float]:
    if len(x) != len(y) or not x or not all(math.isfinite(v) for v in x + y):
        raise ValueError("vector-add inputs must be equal finite non-empty buffers")
    return [a + b for a, b in zip(x, y)]


def ref_cuda_samples_vector_add(x: list[float], y: list[float]) -> list[float]:
    """Reference for CUDA Samples vectorAdd's full output buffer."""
    return ref_vector_add(x, y)


def vector_output_matches(output: list[float], x: list[float], y: list[float]) -> bool:
    if len(output) != len(x) or len(y) != len(x) or not all(math.isfinite(v) for v in output):
        return False
    expected = ref_vector_add(x, y)
    return all(abs(got - want) <= 1e-5 for got, want in zip(output, expected))


# ---------------------------------------------------------------------------
# Generic property checks retained for compatibility with any future pending
# extension. The current generated catalog has no pending formula rows.
def property_only_check(kind: str) -> None:
    rng = random.Random(20260914)
    x = [rng.uniform(-2, 2) for _ in range(257)]
    if kind == "softmax":
        y = [math.exp(v - max(x)) for v in x]
        assert close(sum(v / sum(y) for v in y), 1.0)
    elif kind == "norm":
        mean = sum(x) / len(x)
        var = sum((v - mean) ** 2 for v in x) / len(x)
        y = [(v - mean) / math.sqrt(var + 1e-5) for v in x]
        assert close(sum(y) / len(y), 0.0, 2e-5)
    elif kind == "gather":
        table = list(range(1024))
        idx = [0, 7, 7, 1023]
        assert [table[i] for i in idx] == [0, 7, 7, 1023]
    elif kind == "reduce":
        assert close(sum(x), sum(reversed(x)))
    else:
        raise ValueError(kind)


REAL_REFERENCE_KINDS = {
    "load_fma_accumulate", "segmented_gather", "shared_stage_decouple",
    "triad_axpy2d", "tiled_transpose", "vector_add", "softmax", "reduce", "transpose",
    "norm", "gather",
}


def run_real_reference_smoke_tests() -> list[str]:
    """Exercise every real reference function once with representative
    parameters and check it produces the type of output its parent kernel's
    own host check would produce. Returns the list of kinds that passed."""
    passed = []

    got = ref_tier_scaffold(threads=32, iterations=5)
    assert got[0] == 0 + 10 and got[31] == 31 + 10  # repeat_sum(5) = 4*5/2=10
    passed.append("load_fma_accumulate")

    got = ref_multistream(threads=32, iterations=5, streams=4)
    assert got == ref_tier_scaffold(32, 5)
    passed.append("multistream")

    got = ref_segmented(threads=16, iterations=3, run_length=8, run_spread=8)
    assert len(got) == 16 and all(v >= t for t, v in enumerate(got))
    passed.append("segmented_gather")

    got = ref_shared_stage(threads=64, chunks=4, reuse=2, blocks=3, batches=2, launches=4)
    assert len(got) == 3 * 64 and got[0] == 0 + 16 and got[64] == 0 + 16
    assert shared_stage_output_matches(64, 4, 2, 3, 2, 4, got)
    bad_shared = got.copy(); bad_shared[64 + 7] += 1
    assert not shared_stage_output_matches(64, 4, 2, 3, 2, 4, bad_shared)
    assert not shared_stage_output_matches(64, 4, 2, 3, 2, 4, got[:-1])
    passed.append("shared_stage_decouple")

    got = ref_triad(n=16, launches=3, batches=1, tile_dim=16)
    assert len(got) == 16 * 16 and all(close(v, 1.125) for v in got)
    assert triad_output_matches(16, 3, 1, 16, got)
    bad_triad = got.copy(); bad_triad[16 + 3] += 0.1
    assert not triad_output_matches(16, 3, 1, 16, bad_triad)
    assert not triad_output_matches(16, 3, 1, 16, got[:-1])
    passed.append("triad_axpy2d")

    expected = transpose_expected_output(n=64)
    assert transpose_output_matches(64, expected)
    corrupted = expected.copy(); corrupted[17] += 1.0
    assert not transpose_output_matches(64, corrupted)
    assert not transpose_output_matches(64, expected[:-1])
    passed.append("tiled_transpose")

    got = ref_triton_vector_add([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    assert got == [11.0, 22.0, 33.0]
    passed.append("vector_add")

    softmax_x = softmax_input(5, 7)
    softmax_y = ref_triton_softmax(5, 7, softmax_x)
    assert triton_softmax_output_matches(5, 7, softmax_y, softmax_x)
    corrupted = softmax_y.copy(); corrupted[17] += 0.01
    assert not triton_softmax_output_matches(5, 7, corrupted, softmax_x)
    assert not triton_softmax_output_matches(5, 7, softmax_y[:-1], softmax_x)
    assert not triton_softmax_output_matches(5, 7, [float("nan")] * 35, softmax_x)
    passed.append("softmax")

    padded_softmax_x = softmax_input(5, 11)
    padded_softmax_y = ref_pytorch_softmax(5, 7, 11, padded_softmax_x)
    assert pytorch_softmax_output_matches(5, 7, 11, padded_softmax_y, padded_softmax_x)
    bad_padded_softmax = padded_softmax_y.copy(); bad_padded_softmax[4] += 0.01
    assert not pytorch_softmax_output_matches(5, 7, 11, bad_padded_softmax, padded_softmax_x)
    assert not pytorch_softmax_output_matches(5, 7, 11, padded_softmax_y[:-1], padded_softmax_x)
    passed.append("pytorch_softmax")

    reduction_x = [float(i) for i in range(33)]
    reduction_y = ref_cuda_samples_reduction(reduction_x)
    assert scalar_output_matches(reduction_y, reduction_y)
    assert not scalar_output_matches([reduction_y[0] + 1.0], reduction_y)
    assert not scalar_output_matches([], reduction_y)
    passed.append("reduction")

    cub_block_y = ref_cub_block_reduce(reduction_x)
    cub_device_y = ref_cub_device_reduce(reduction_x)
    assert scalar_output_matches(cub_block_y, cub_device_y)
    assert not scalar_output_matches([cub_device_y[0] + 1.0], cub_device_y)
    passed.append("cub_reductions")

    transpose_y = ref_cuda_transpose(7, 5)
    assert cuda_transpose_output_matches(7, 5, transpose_y)
    bad_transpose = transpose_y.copy(); bad_transpose[3] += 1
    assert not cuda_transpose_output_matches(7, 5, bad_transpose)
    assert not cuda_transpose_output_matches(7, 5, transpose_y[:-1])
    passed.append("transpose")

    ln_y = ref_layer_norm(5, 7)
    assert layer_norm_output_matches(5, 7, ln_y)
    bad_ln = ln_y.copy(); bad_ln[4] += 0.1
    assert not layer_norm_output_matches(5, 7, bad_ln)
    assert not layer_norm_output_matches(5, 7, ln_y[:-1])
    passed.append("layer_norm")

    padded_ln_x = softmax_input(5, 11, seed=20260915)
    padded_ln_y = ref_pytorch_layer_norm(5, 7, 11, padded_ln_x)
    assert pytorch_layer_norm_output_matches(5, 7, 11, padded_ln_y, padded_ln_x)
    bad_padded_ln = padded_ln_y.copy(); bad_padded_ln[4] += 0.1
    assert not pytorch_layer_norm_output_matches(5, 7, 11, bad_padded_ln, padded_ln_x)
    assert not pytorch_layer_norm_output_matches(5, 7, 11, padded_ln_y[:-1], padded_ln_x)
    passed.append("pytorch_layer_norm")

    indices = [0, 7, 3, 7]
    gather_y = ref_pytorch_index_select(16, 5, indices)
    assert index_select_output_matches(16, 5, indices, gather_y)
    bad_gather = gather_y.copy(); bad_gather[2] += 1
    assert not index_select_output_matches(16, 5, indices, bad_gather)
    assert not index_select_output_matches(16, 5, indices, gather_y[:-1])
    assert ref_xformers_index_select(16, 5, indices) == gather_y
    passed.append("index_select")

    vx, vy = [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]
    vz = ref_cuda_samples_vector_add(vx, vy)
    assert vector_output_matches(vz, vx, vy)
    bad_v = vz.copy(); bad_v[1] += 1
    assert not vector_output_matches(bad_v, vx, vy)
    assert not vector_output_matches(vz[:-1], vx, vy)
    passed.append("cuda_vector_add")

    return passed


def main() -> None:
    rows = list(csv.DictReader((Path(__file__).parent / "workload_catalog.csv").open()))
    real_passed = run_real_reference_smoke_tests()
    for kind in real_passed:
        print(f"REAL_REFERENCE_OK {kind}")

    # Keep any future pending extension explicit and keyed by PARENT, not by a
    # shared semantic_reference string. C2's bug was over-broad sharing; no
    # pending rows exist in the current generated catalog.
    proxy_map = {"softmax": "softmax", "norm": "norm", "gather": "gather", "reduce": "reduce"}
    pending_rows = [r for r in rows if r["work_formula_basis"] == "pending"]
    seen_proxy_kinds = set()
    for row in pending_rows:
        parent_id, kind = row["parent_id"], row["semantic_reference"]
        proxy_kind = proxy_map.get(kind)
        if proxy_kind is None:
            print(f"NO_REFERENCE_YET {parent_id} ({kind}): work_formula_basis=pending, no reference "
                  "implemented (not even property-only) -- see workload_catalog.csv candidate_grid_note")
            continue
        if proxy_kind not in seen_proxy_kinds:
            property_only_check(proxy_kind)
            seen_proxy_kinds.add(proxy_kind)
        print(f"PROPERTY_ONLY_CHECK_OK {parent_id} ({kind}): this is NOT a real parent input->output "
              "reference; work_formula_basis=pending for this parent, see workload_catalog.csv")

    if not pending_rows:
        print("CPU_ONLY: all catalog parents have source-backed full-output references; no "
              "PROPERTY_ONLY rows remain. Neither CPU agreement nor these references are GPU-kernel "
              "correctness evidence.")
    else:
        print("CPU_ONLY: real-reference rows above validate an actual parent input->output relationship "
              "transcribed from its own kernel source; PROPERTY_ONLY rows validate a generic mathematical "
              "identity only. Neither is GPU-kernel correctness evidence.")


if __name__ == "__main__":
    main()
