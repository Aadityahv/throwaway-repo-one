# PyTorch dispatch trace for the measured Blackwell cells

Static host-code analysis of ATen at commit `70d99e998b4955e0049d13a98d77ae1b14db1f45` (the commit torch
2.11.0+cu128 was built from, CHANGELOG M443; the catalog pin 67faf385 is not what ran). No binary inspected, no
GPU or ssh used. Sources and sha256 are in `src/`. Machine-readable result: `dispatch_trace.json` (27 cells);
rules and builder: `dispatch_rules.py`; check: `python3 test_dispatch.py`.

Measured unit: one `launch_once()` call; the harness captures 1000 calls in one CUDA graph and divides energy and
time by the launch count, so per-launch figures include every kernel listed below, including any copy kernel.

| Operator (cells) | Python call | Kernels per call | Kernel and geometry rule |
|---|---|---|---|
| Row softmax (12), fp32, dim 1 | `F.softmax(padded[:, :cols], dim=1)` | stride == cols (candidate 1): 1. Padded stride (candidates 2-4): 2 | Optional copy: `at::native::elementwise_kernel<128, 2, gpu_kernel_impl_nocast<direct_copy_kernel_cuda ... lambda(float)>>`, grid = rows*cols/256, block 128 (scalar strided copy, `contiguous()` of the non-contiguous view). Then `softmax_warp_forward<float, float, float, L, false, false>` with L = ceil(log2(cols)) (7, 9, 10): block (32, 4), no shared memory, grid = ceil(rows / (4 * (2 if cols<=128 else 1))) = 128 / 1024 / 2048 for small / medium / large |
| Layer norm (12), fp32, affine, eps 1e-5 | `F.layer_norm(padded[:, :cols], [cols], ones, zeros)` | stride == cols: 1. Padded: 2 | Optional copy: same copy kernel, grid rows*cols/256. Then `vectorized_layer_norm_kernel<float, float, false>`: grid = rows, block (32, 4), dynamic shared memory 24 B, float4 loads. The non-vectorized pair (`RowwiseMomentsCUDAKernel` + `LayerNormForwardCUDAKernel`) is not launched in any cell |
| Embedding (3), fp32 table, int64 index | `torch.index_select(table, 0, index)` (not `nn.Embedding`) | 1 | `index_select_cuda` takes the gather route because num_indices > 16 (the small-index kernel needs <= 16): `at::gather_out` -> `vectorized_gather_kernel<16, long>`, grid (num_indices, 1), block min(256, round_up32(row_bytes/16)) = 32 / 64 / 128 for feature dim 128 / 256 / 512, no shared memory |

Cell-by-cell geometry (copy kernel only for padded-stride candidates):

| Operator, regime | rows x cols | main kernel grid, block | copy kernel grid, block |
|---|---|---|---|
| softmax small / medium / large | 1024x128 / 4096x512 / 8192x1024 | 128 / 1024 / 2048, (32,4) | 512 / 8192 / 32768, 128 |
| layer norm small / medium / large | same | 1024 / 4096 / 8192, (32,4), 24 B smem | same as softmax |
| embedding small / medium / large | 1024 idx x 128, 4096 x 256, 16384 x 512 (table 4096 / 65536 / 1048576 rows) | 1024 / 4096 / 16384, 32 / 64 / 128 threads | none |

Device values used: warp size 32 (HARDWARE_GROUND_TRUTH.md). No taken branch reads SM count or shared memory per
block. The embedding launch reads `maxGridSize[1]` (not in HARDWARE_GROUND_TRUTH.md); the result is the same for any
value >= 1 because the y grid is min(1, maxGridSize[1]).

## Ambiguity and confidence

No cell has an ambiguous kernel choice at the host-code level for softmax or layer norm. Embedding is the least
certain: the gather route is unconditional, but taking the vectorized fast path rather than the generic
`_scatter_gather_elementwise_kernel<128, 8>` rests on a hand derivation of TensorIterator strides.

## Assumptions that need a runtime trace (or a symbol-table grep) to confirm

1. Allocator alignment: fresh tensors are at least 16-byte aligned (512-byte rounding, c10/core/AllocatorConfig.h:18).
   This decides the layer norm vectorized path and the embedding fast path. Allocator config env was not recorded.
2. The binary's code is the code in these files (M443 asserts the git version; the demangled names in the table
   should be grepped in the real `libtorch_cuda.so`).
3. The copy kernel's lambda ordinal in the demangled name is not derivable statically; match by regex.
4. Embedding fast-path eligibility (TensorIterator stays 2-D, index strides (0, 8), src row stride 0). Alternative
   if false: `_scatter_gather_elementwise_kernel<128, 8, ...>`, grid ceil(num_idx*dim/1024), block 128.
5. `use_deterministic_algorithms` off (otherwise `empty` adds fill kernels); the runners never enable it.
6. The 1000-call CUDA graph replays the same kernels; graph capture does not change the selected kernels.

## Disagreement with the development table

`tiresias/framework/compile_evidence/development/development_cells.csv` has `grid_blocks`, `block_threads`,
`blocks_per_sm`, `active_sm_fraction` and `geometry_source` empty for all 27 of these Blackwell cells, so there is
nothing to disagree with numerically. `shape_a`/`shape_b` hold rows/cols (softmax, layer norm) and vocabulary/feature
dimension (embedding). They are not launch geometry: using them as grid/block would be wrong (true grids above are
rows/4 or rows/8 for softmax, rows for layer norm, num_indices for embedding; blocks are 128 threads, not cols).
The table also does not record the extra copy kernel for the padded-stride candidates.
