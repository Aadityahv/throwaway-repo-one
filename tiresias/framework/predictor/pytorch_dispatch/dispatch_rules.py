#!/usr/bin/env python3
"""Encoded ATen host-dispatch rules (PyTorch commit 70d99e99) for the measured
Blackwell PyTorch cells, and the builder for dispatch_trace.json.

CPU-only, static. Every rule below was read from the source files saved under
src/ (sha256 in SHA256SUMS); file:line references are into those exact files
(local name = repo path with '/' replaced by '_', see SRC_NAME).

Device facts: only warpSize = 32 (HARDWARE_GROUND_TRUTH.md, Blackwell section,
"Warp size") is consumed by the taken paths. maxGridSize[1] is consumed by the
embedding launch but the result is insensitive to its value (min(1, x) = 1 for
any x >= 1); it is not in HARDWARE_GROUND_TRUTH.md and is stated as such.

Run: python3 dispatch_rules.py   (writes dispatch_trace.json next to this file)
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SRC = HERE / "src"
REVISION = "70d99e998b4955e0049d13a98d77ae1b14db1f45"
WARP_SIZE = 32  # HARDWARE_GROUND_TRUTH.md, Blackwell "Warp size"
REGIMES = ("small", "medium", "large")
CANDS = ("c1", "c2", "c3", "c4")


def src_name(path: str) -> str:
    return path.replace("/", "_")


def ref(path: str, line: str | int, note: str) -> str:
    return f"{path}:{line} @70d99e99 -- {note}"


# ---------------------------------------------------------------- shared rules
def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def log2_ceil(v: int) -> int:  # PersistentSoftmax.cuh:13-17
    k = 0
    while (1 << k) < v:
        k += 1
    return k


def softmax_persistent_geometry(batch: int, dim: int) -> dict:
    """PersistentSoftmax.cuh:302-346 dispatch_softmax_forward (fp32, no mask)."""
    l2e = log2_ceil(dim)
    npot = 1 << l2e
    warp = min(npot, WARP_SIZE)
    bpw = 2 if npot <= 128 else 1
    tpb = 128
    warps_per_block = tpb // warp
    bpb = warps_per_block * bpw
    blocks = ceil_div(batch, bpb)
    return {"log2_elements": l2e, "grid": [blocks, 1, 1],
            "block": [warp, warps_per_block, 1], "batches_per_warp": bpw,
            "warp_iterations": npot // warp}


def copy_geometry(numel: int) -> dict:
    """CUDALoops.cuh:661-667 -> launch_legacy_kernel<128, 2> (540-550)."""
    return {"grid": [ceil_div(numel, 128 * 2), 1, 1], "block": [128, 1, 1]}


def layernorm_geometry(rows: int, cols: int) -> dict:
    """layer_norm_kernel.cu:1020-1049 launch_vectorized_layer_norm_kernel."""
    num_threads = WARP_SIZE * 4  # thread_constants.h:num_threads() (non-ROCm)
    ty = num_threads // WARP_SIZE
    nshared = (ty * 3 // 2) * 4 if ty > 1 else 0  # sizeof(float)
    n_vec = cols // 4
    return {"grid": [rows, 1, 1], "block": [WARP_SIZE, ty, 1], "smem": nshared,
            "vec_loads_per_row": n_vec,
            "max_vec_iterations_per_thread": ceil_div(n_vec, WARP_SIZE * ty)}


def gather_geometry(num_idx: int, dim: int, max_grid_y: int = 65535) -> dict:
    """IndexKernelUtils.cu:26-39 vectorized_gather_kernel_launch<16, int64>."""
    slice_bytes = dim * 4
    nthreads = ceil_div(slice_bytes, 16)
    nthreads = ceil_div(nthreads, WARP_SIZE) * WARP_SIZE  # at::round_up
    grid_y = min(ceil_div(slice_bytes, 256 * 16), max_grid_y)
    return {"grid": [num_idx, grid_y, 1], "block": [min(256, nthreads), 1, 1],
            "slice_bytes": slice_bytes}


# ----------------------------------------------------------------- cell tables
def load_controls() -> dict:
    """Controls straight from the measured raw CSVs (read-only)."""
    out: dict = {}
    base = REPO / "tiresias/app_runners/application_energy_raw"
    for d, parents in (("SOFTMAX-DEV-20260922T185912Z", None),
                       ("FINAL-PYTORCH-20260922T195700Z", None),
                       ("FINAL-PYTORCH-S2-20260922T204600Z", None)):
        f = base / d / "application_energy_raw.csv"
        for r in csv.DictReader(open(f)):
            if not r["parent_id"].endswith(("softmax", "layer_norm", "embedding")) or "triton" in r["parent_id"]:
                continue
            key = (r["parent_id"], r["regime"], r["candidate_id"])
            out.setdefault(key, []).append((d, json.loads(r["controls_json"])))
    return out


def merged_controls() -> dict:
    res = {}
    for key, lst in load_controls().items():
        first = lst[0][1]
        assert all(c == first for _, c in lst), f"controls differ between sessions for {key}"
        res[key] = (first, [d for d, _ in lst])
    return res


def build() -> dict:
    controls = merged_controls()
    cells = []
    for (op, regime, cand), (ctl, dirs) in sorted(controls.items()):
        cid = f"blackwell/{op}/{regime}/{cand}"
        if op == "dev_pytorch_rowwise_softmax":
            cells.append(softmax_cell(cid, op, regime, cand, ctl, dirs))
        elif op == "final_pytorch_layer_norm":
            cells.append(layernorm_cell(cid, op, regime, cand, ctl, dirs))
        elif op == "final_pytorch_embedding":
            cells.append(embedding_cell(cid, op, regime, cand, ctl, dirs))
    sums = {}
    for line in (SRC / "SHA256SUMS").read_text().splitlines():
        h, n = line.split()
        sums[n] = h
    return {
        "schema": "pytorch_dispatch_trace_v1",
        "pytorch_commit": REVISION,
        "provenance_note": "Measured cells ran torch 2.11.0+cu128 built from this commit (CHANGELOG M443); the catalog pin 67faf385 is not the source that ran. No binary was inspected: this is static host-code analysis only.",
        "device": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition (sm_120)",
        "device_facts_used": {"warp_size": {"value": WARP_SIZE, "source": "HARDWARE_GROUND_TRUTH.md Blackwell: Warp size"},
                              "maxGridSize[1]": {"value": "not in HARDWARE_GROUND_TRUTH.md", "effect": "none: embedding grid_y = min(ceil(slice_bytes/4096)=1, maxGridSize[1]) = 1 for any maxGridSize[1] >= 1"},
                              "sharedMemPerBlock/multiProcessorCount": "read only on branches NOT taken by any of these cells"},
        "source_files_sha256": sums,
        "source_dir": "src/ (local filename = repo path with / replaced by _)",
        "measured_unit": "One launch_once() = one Python call (softmax / layer_norm / index_select). Energy harness captures 1000 launch_once calls in one CUDA graph (energy_harness/torch_graph_capture.py make_graph_replayer; launch_count = replays*1000), board_energy_j_per_launch = total / launch_count, so per-launch energy and runtime cover ALL kernels of a call, including the contiguous() copy kernel.",
        "cells": cells,
    }


COMMON_ASSUMPTIONS = [
    "Tensors come from the CUDA caching allocator: block sizes are rounded to multiples of 512 B (c10/core/AllocatorConfig.h:18 kMinBlockSize=512; c10/cuda/CUDACachingAllocator.cpp:2600-2609 round_size), so freshly allocated tensors are >= 16 B aligned. Assumes no PYTORCH_CUDA_ALLOC_CONF that changes this and no cudaMallocAsync backend difference (not recorded in runtime_json).",
    "torch.use_deterministic_algorithms is off (runners never enable it); otherwise at::empty would add a fill kernel.",
    "Binary is the official-style cu128 build of this commit with USE_ROCM off; the host paths read are the non-ROCm branches.",
    "Source-vs-binary identity (that libtorch_cuda.so's text was compiled from these files) is taken from M443 (torch.version.git_version) and is not re-verified here; kernels' demangled names should be grepped in the real binary to confirm.",
]


def softmax_cell(cid, op, regime, cand, ctl, dirs):
    rows, cols, stride = ctl["outer_size"], ctl["dim_size"], ctl["row_stride"]
    assert ctl["inner_size"] == 1
    contiguous = stride == cols
    pg = softmax_persistent_geometry(rows, cols)
    kernels = []
    path = [
        ref("torch/nn/functional.py", 2159, "F.softmax(x, dim=1): dtype None -> input.softmax(dim)"),
        ref("aten/src/ATen/native/native_functions.yaml", 5716, "softmax.int has no dispatch entry -> CompositeImplicitAutograd C++ softmax()"),
        ref("aten/src/ATen/native/SoftMax.cpp", "411-418", "softmax(): not (CUDA half->float) -> at::_softmax(input, dim, half_to_float=false)"),
        ref("aten/src/ATen/native/native_functions.yaml", "5727-5740", "_softmax structured_delegate _softmax.out; CUDA: softmax_cuda_out"),
        ref("aten/src/ATen/native/cuda/SoftMax.cu", "1435", "softmax_cuda_out -> host_softmax<SoftMaxForwardEpilogue, SoftMaxForwardWithMulEpilogue, is_log_softmax=false, use_fast_softmax=false>"),
        ref("aten/src/ATen/native/cuda/SoftMax.cu", 1072, "host_softmax: input = input_.contiguous()"),
    ]
    if not contiguous:
        path += [
            ref("aten/src/ATen/native/TensorProperties.cpp", 127, f"view strides ({stride},1) != contiguous strides ({cols},1): contiguous() -> clone(Contiguous)"),
            ref("aten/src/ATen/native/TensorFactories.cpp", "2284,2290", "clone: empty_like(contiguous) then self.copy_(src)"),
            ref("aten/src/ATen/native/cuda/Copy.cu", "264,314", "copy_kernel_cuda -> copy_device_to_device: iter.is_contiguous() false (2-D, src row stride != cols, cannot coalesce) -> direct_copy_kernel_cuda"),
            ref("aten/src/ATen/native/cuda/Copy.cu", 240, "float: gpu_kernel(iter, [](float x){return x;})"),
            ref("aten/src/ATen/native/cuda/CUDALoops.cuh", "960,659-667", "no dynamic cast -> gpu_kernel_impl_nocast; not contiguous -> launch_legacy_kernel<128, unroll=2 (sizeof(float)>=4)>"),
        ]
        kernels.append(copy_kernel(rows * cols, "contiguous() materialisation of the padded view into a dense rows x cols float tensor (precedes softmax)"))
    else:
        path.append(ref("aten/src/ATen/core/TensorBase.h", "137-141", f"view strides ({stride},1) == contiguous: contiguous() returns self, NO copy kernel"))
    path += [
        ref("aten/src/ATen/native/cuda/SoftMax.cu", 1090, "inner_size == 1 (dim=1 of a 2-D tensor)"),
        ref("aten/src/ATen/native/cuda/SoftMax.cu", 1097, f"!half_to_float, dim_size={cols} <= 2048 && {cols*4} B <= 8192 B -> persistent warp softmax (dispatch_softmax_forward); block-softmax / smem / reg variants not reached"),
        ref("aten/src/ATen/native/cuda/PersistentSoftmax.cuh", "309-324", f"log2_ceil({cols})={pg['log2_elements']}; warp={pg['block'][0]}; batches_per_warp={pg['batches_per_warp']}; 128 threads/block; blocks=ceil({rows}/{pg['block'][1]*pg['batches_per_warp']})={pg['grid'][0]}"),
        ref("aten/src/ATen/native/cuda/PersistentSoftmax.cuh", f"{335+pg['log2_elements']}", f"LAUNCH_SOFTMAX_WARP_FORWARD({pg['log2_elements']}) single launch (chunk_size 2^30/dim >= batch)"),
    ]
    l2e = pg["log2_elements"]
    kernels.append({
        "order": len(kernels) + 1,
        "role": "softmax",
        "source": {"file": "aten/src/ATen/native/cuda/PersistentSoftmax.cuh", "function": "softmax_warp_forward", "line": 68},
        "template_args": {"input_t": "float", "output_t": "float", "acc_t": "float", "log2_elements": l2e, "is_log_softmax": False, "is_masked": False},
        "demangled_name_pattern": f"(anonymous namespace)::softmax_warp_forward<float, float, float, {l2e}, false, false>(float*, float const*, int, int, int, bool const*, int, bool)",
        "demangled_name_regex": rf"softmax_warp_forward<float, float, float, {l2e}, false, false>",
        "grid": pg["grid"], "block": pg["block"], "dynamic_smem_bytes": 0,
        "args": {"batch_size": rows, "stride": cols, "element_count": cols, "mask": None, "head_chunk_size": -1, "is_transformer_mask": False,
                 "WARP_BATCH": pg["batches_per_warp"], "WARP_ITERATIONS": pg["warp_iterations"], "WARP_SIZE": pg["block"][0]},
        "extra_control_flow": "rows beyond batch_size masked via local_batches; element_count == next_power_of_two so no tail predication on columns",
    })
    return {
        "cell_id": cid, "operator_id": op, "regime": regime, "candidate_id": cand,
        "call": "torch.nn.functional.softmax(view, dim=1) where view = padded[:, :cols], padded = (rows, row_stride) float32 cuda tensor",
        "dtype": "float32",
        "input": {"shape": [rows, cols], "strides": [stride, 1], "contiguous": contiguous, "storage_offset": 0},
        "controls": ctl, "controls_from_sessions": dirs,
        "kernels": kernels, "kernels_per_call": len(kernels),
        "measured_unit_covers_all_kernels": True,
        "decision_path": path,
        "confidence": "high",
        "assumptions": COMMON_ASSUMPTIONS + ["Lambda ordinal in the copy kernel's demangled name ({lambda(float)#N}) is not determinable statically; match with the regex, not the literal."] if not contiguous else COMMON_ASSUMPTIONS,
    }


def copy_kernel(numel: int, role: str) -> dict:
    g = copy_geometry(numel)
    return {
        "order": 1, "role": role,
        "source": {"file": "aten/src/ATen/native/cuda/CUDALoops.cuh", "function": "elementwise_kernel<128, 2, ...> via gpu_kernel_impl_nocast (instantiated from direct_copy_kernel_cuda, Copy.cu:240)", "line": 526},
        "template_args": {"nt": 128, "vt": 2, "func_t": "gpu_kernel_impl_nocast<direct_copy_kernel_cuda(TensorIteratorBase&)::{lambda(float)#N}>::{lambda(int)#1}"},
        "demangled_name_pattern": "at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#N}>(at::TensorIteratorBase&, ...)::{lambda(int)#1}>(int, ...)",
        "demangled_name_regex": r"elementwise_kernel<128, 2, .*gpu_kernel_impl_nocast<.*direct_copy_kernel_cuda.*\{lambda\(float\)#\d+\}.*\{lambda\(int\)#1\}>",
        "grid": g["grid"], "block": g["block"], "dynamic_smem_bytes": 0,
        "args": {"N": numel, "offset_calculator": "OffsetCalculator<2> over dims [cols, rows]"},
        "extra_control_flow": "each thread handles vt=2 elements 128 apart (idx += nt); offset calculator does 2-D div/mod per element (non-vectorized scalar float loads/stores)",
    }


def layernorm_cell(cid, op, regime, cand, ctl, dirs):
    rows, cols, stride = ctl["rows"], ctl["cols"], ctl["row_stride"]
    contiguous = stride == cols
    lg = layernorm_geometry(rows, cols)
    kernels = []
    path = [
        ref("torch/nn/functional.py", 2935, "F.layer_norm -> torch.layer_norm(input, [cols], weight, bias, eps=1e-5, cudnn_enabled)"),
        ref("aten/src/ATen/native/native_functions.yaml", "3322-3324", "layer_norm: CompositeImplicitAutograd layer_norm_symint"),
        ref("aten/src/ATen/native/layer_norm.cpp", 198, "layer_norm_symint -> std::get<0>(at::native_layer_norm_symint(...))"),
        ref("aten/src/ATen/native/native_functions.yaml", "3326-3329", "native_layer_norm CUDA: layer_norm_cuda"),
        ref("aten/src/ATen/native/cuda/layer_norm_kernel.cu", 1725, "layer_norm_cuda: X = input.expect_contiguous(); gamma/beta expect_contiguous"),
    ]
    if not contiguous:
        path += [
            ref("aten/src/ATen/core/TensorBase.h", "1056-1064", "expect_contiguous on non-contiguous view -> owned __dispatch_contiguous -> clone(Contiguous)"),
            ref("aten/src/ATen/native/TensorFactories.cpp", "2284,2290", "clone: empty_like + copy_"),
            ref("aten/src/ATen/native/cuda/Copy.cu", "264,314,240", "non-contiguous same-dtype copy -> direct_copy_kernel_cuda float lambda"),
            ref("aten/src/ATen/native/cuda/CUDALoops.cuh", "960,659-667", "gpu_kernel_impl_nocast, not contiguous -> launch_legacy_kernel<128, 2>"),
        ]
        kernels.append(copy_kernel(rows * cols, "expect_contiguous() materialisation of the padded view (precedes layer norm)"))
    else:
        path.append(ref("aten/src/ATen/native/cuda/layer_norm_kernel.cu", 1725, f"view strides ({stride},1) == contiguous: expect_contiguous borrows, NO copy kernel"))
    path += [
        ref("aten/src/ATen/native/cuda/layer_norm_kernel.cu", "1730-1737", "Y = empty_like(X); mean, rstd = at::empty({M}) (no fill kernels)"),
        ref("aten/src/ATen/native/cuda/layer_norm_kernel.cu", 1144, "LayerNormKernelImpl dispatch float -> LayerNormKernelImplInternal<float, float(acc)>"),
        ref("aten/src/ATen/native/cuda/layer_norm_kernel.cu", "1104-1111", f"fast-path test: T=float, N={cols} <= 2^24, N % 4 == 0 ({cols % 4 == 0}), X/Y/gamma/beta all 16-B aligned (allocator-fresh, assumed) -> TRUE"),
        ref("aten/src/ATen/native/cuda/layer_norm_kernel.cu", 1114, "launch_vectorized_layer_norm_kernel<float, float, false> (RowwiseMoments + LayerNormForward fallback at 1117-1122 NOT taken)"),
        ref("aten/src/ATen/native/cuda/layer_norm_kernel.cu", "1034-1048", f"threads=({WARP_SIZE}, num_threads()/{WARP_SIZE}=4, 1) (thread_constants.h num_threads()=32*4=128); blocks=M={rows}; nshared = 4*3/2*sizeof(float) = {lg['smem']} B"),
    ]
    kernels.append({
        "order": len(kernels) + 1,
        "role": "layer_norm_forward (single fused moments+normalize kernel)",
        "source": {"file": "aten/src/ATen/native/cuda/layer_norm_kernel.cu", "function": "vectorized_layer_norm_kernel -> vectorized_layer_norm_kernel_impl", "line": 343},
        "template_args": {"T": "float", "T_ACC": "float", "rms_norm": False},
        "demangled_name_pattern": "at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float, false>(int, float, float const*, float const*, float const*, float*, float*, float*)",
        "demangled_name_regex": r"vectorized_layer_norm_kernel<float, float, false>",
        "grid": lg["grid"], "block": lg["block"], "dynamic_smem_bytes": lg["smem"],
        "args": {"N": cols, "eps": ctl["eps"], "gamma_defined": ctl["affine"], "beta_defined": ctl["affine"],
                 "n_vec_to_read": lg["vec_loads_per_row"], "vec_size": 4,
                 "max_vec_iterations_per_thread": lg["max_vec_iterations_per_thread"],
                 "block_threads_total": 128},
        "extra_control_flow": "compute_stats: float4 loads strided by 128 threads (Welford), intra-warp shuffle reduce over 5 steps, inter-warp (blockDim.y=4) 2-step smem reduce with __syncthreads; then normalize loop with gamma and beta both defined",
    })
    return {
        "cell_id": cid, "operator_id": op, "regime": regime, "candidate_id": cand,
        "call": "torch.nn.functional.layer_norm(view, [cols], weight=ones(cols), bias=zeros(cols), eps=1e-5) where view = padded[:, :cols], padded = (rows, row_stride) float32",
        "dtype": "float32",
        "input": {"shape": [rows, cols], "strides": [stride, 1], "contiguous": contiguous, "storage_offset": 0},
        "controls": ctl, "controls_from_sessions": dirs,
        "kernels": kernels, "kernels_per_call": len(kernels),
        "measured_unit_covers_all_kernels": True,
        "decision_path": path,
        "confidence": "high",
        "assumptions": COMMON_ASSUMPTIONS + (["Lambda ordinal in the copy kernel's demangled name is not determinable statically; use the regex."] if not contiguous else []) + [
            "The non-vectorized path (RowwiseMomentsCUDAKernel then LayerNormForwardCUDAKernel, 2 kernels) would be taken only if X/Y/gamma/beta were not 16-B aligned or N%4 != 0; neither holds here, so it is NOT launched for any cell.",
        ],
    }


def embedding_cell(cid, op, regime, cand, ctl, dirs):
    vocab, dim, nidx = ctl["num_weights"], ctl["feature_dim"], ctl["num_indices"]
    gg = gather_geometry(nidx, dim)
    path = [
        ref("tiresias/app_runners/embedding_runner.py", "launch_once", "torch.index_select(source, 0, index): NOT nn.Embedding / F.embedding / aten::embedding; source = arange(vocab*dim).reshape(vocab, dim) float32 contiguous, index int64 1-D contiguous"),
        ref("aten/src/ATen/native/native_functions.yaml", "9509-9515", "index_select CUDA: index_select_cuda"),
        ref("aten/src/ATen/native/cuda/Indexing.cu", 1699, "index_select_cuda: out = at::empty({0}); index_select_out_cuda -> index_select_out_cuda_impl<float>"),
        ref("aten/src/ATen/native/cuda/Indexing.cu", 1618, f"numIndices={nidx} > 16, so the indexSelectSmallIndex branch (needs numIndices <= 16) is NOT taken"),
        ref("aten/src/ATen/native/cuda/Indexing.cu", 1649, "else: at::gather_out(out, self, dim=0, index.view({n,1}).expand({n,dim})) -- no kernel for view/expand"),
        ref("aten/src/ATen/native/TensorAdvancedIndexing.cpp", "2069-2083", "gather_out: can_use_expanded_index_path false (returns false: FBGEMM-only/CPU-only, line 1995-2010) -> gather_stub (CUDA: gather_cuda_kernel)"),
        ref("aten/src/ATen/native/cuda/ScatterGatherKernel.cu", "518-521,199-210", "gather_cuda_kernel -> cuda_scatter_gather_base_kernel<false>(TensorAssign): iter(out as_strided, src restrided with stride[0]=0, index expanded); AT_DISPATCH float, index_t=int64"),
        ref("aten/src/ATen/native/cuda/ScatterGatherKernel.cu", "144-155", "!is_scatter_like: fast_gather_kernel_eligible<16> TRUE -> vectorized_gather_kernel_launch<16, int64_t>; generic _scatter_gather_elementwise_kernel (line 174) NOT launched"),
        ref("aten/src/ATen/native/cuda/IndexKernelUtils.h", "20-27", f"eligible: iter.ndim()==2 (src stride-0 dim prevents coalescing), index strides (0,8), out/src inner stride 4 B, out/src row-base ptrs 16-B aligned, slice {dim*4} B, index_stride {dim*4} B, out row stride {dim*4} B all multiples of 16"),
        ref("aten/src/ATen/native/cuda/IndexKernelUtils.cu", "26-39", f"num_threads=round_up(ceil({dim*4}/16), 32)={ceil_div(dim*4,16)}->{gg['block'][0]}; block=min(256, that)={gg['block'][0]}; grid=({nidx}, grid_y=min(ceil({dim*4}/4096)=1, maxGridSize[1]) = 1, 1)"),
    ]
    kernels = [{
        "order": 1, "role": "row gather (index_select over dim 0)",
        "source": {"file": "aten/src/ATen/native/cuda/IndexKernelUtils.cu", "function": "vectorized_gather_kernel", "line": 11},
        "template_args": {"Alignment": 16, "index_t": "int64_t"},
        "demangled_name_pattern": "void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)",
        "demangled_name_regex": r"vectorized_gather_kernel<16, long>",
        "grid": gg["grid"], "block": gg["block"], "dynamic_smem_bytes": 0,
        "args": {"num_ind": nidx, "slice_size_bytes": dim * 4, "ind_dim_size": vocab, "inp_stride_bytes": dim * 4, "out_stride_bytes": dim * 4, "allow_neg_indices": False},
        "extra_control_flow": f"one block per output row (blockIdx.x indexes idx[]); inner loop off=(blockDim.x*blockIdx.y+tid)*16 stepping blockDim.x*gridDim.y*16: exactly 1 iteration per thread since slice {dim*4} B == block {gg['block'][0]} x 16 B; each thread one 16-B ld/st; random 16-B-aligned row gather, data-dependent index loads (int64) one per block",
    }]
    return {
        "cell_id": cid, "operator_id": op, "regime": regime, "candidate_id": cand,
        "call": "torch.index_select(source, 0, index)",
        "dtype": "float32 (source/out), int64 (index)",
        "input": {"shape": [vocab, dim], "strides": [dim, 1], "contiguous": True, "storage_offset": 0, "index_shape": [nidx], "index_dtype": "int64"},
        "controls": ctl, "controls_from_sessions": dirs,
        "kernels": kernels, "kernels_per_call": 1,
        "measured_unit_covers_all_kernels": True,
        "decision_path": path,
        "confidence": "high for the gather route (numIndices > 16 is unconditional); medium-high that the vectorized fast path (not the generic scatter-gather elementwise kernel) is taken, because it rests on TensorIterator keeping a 2-D, uncoalesced layout with the stated strides (derived by hand from TensorIterator.cpp reorder/coalesce rules, not observed)",
        "assumptions": COMMON_ASSUMPTIONS + [
            "If the fast-path predicate were false, the launched kernel would instead be at::native::_scatter_gather_elementwise_kernel<128, 8, ...> with grid=ceil(num_idx*dim/1024), block 128 (ScatterGatherKernel.cu:174, 94-112). Check by grepping the real symbol list / a one-call profile.",
            "Indexing.cu is NOT where the kernel lives at this commit for >16 indices: the catalog's source_path (Indexing.cu) holds only the host function; the device code is IndexKernelUtils.cu and ScatterGatherKernel.cu.",
        ],
    }


def main() -> None:
    trace = build()
    (HERE / "dispatch_trace.json").write_text(json.dumps(trace, indent=1, sort_keys=False) + "\n")
    print(f"wrote dispatch_trace.json with {len(trace['cells'])} cells")


if __name__ == "__main__":
    main()
