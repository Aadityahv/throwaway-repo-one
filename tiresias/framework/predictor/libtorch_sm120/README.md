# Blackwell (sm_120) machine code of the traced PyTorch kernels

The machine code (SASS) for these kernels was extracted on October 1, 2026, CPU only and at the lowest
priority, from the host's installed `libtorch_cuda.so`. That library is torch 2.11.0+cu128, built from
commit `70d99e99`, and its SHA256 is in `libtorch_cuda.sha256`. Extraction used `cuobjdump` from CUDA
13.2 (`-arch sm_120 -sass -fun <mangled>`). No GPU was used.

These are the kernels named by the static dispatch trace (`../pytorch_dispatch/`). Every one of them
exists in the sm_120 code, which supports the trace's assumption that the binary matches source
`70d99e99`.

| File | Kernel | Registers per thread | Shared (as reported by cuobjdump) |
| --- | --- | ---: | ---: |
| k1 | `elementwise_kernel<128,2>` float direct copy, the extra copy kernel for padded candidates | 20 | 0 |
| k2 | `vectorized_gather_kernel<16,long>` (embedding via index_select) | 24 | 0 |
| k3 | `vectorized_layer_norm_kernel<float,float,false>` | 44 | 1024 |
| k4 | `softmax_warp_forward<float,float,float,10,false,false>` (large) | 72 | 1024 |
| k5 | `softmax_warp_forward<...,7,...>` (small) | 34 | 1024 |
| k6 | `softmax_warp_forward<...,9,...>` (medium) | 48 | 1024 |

To check: the 1024 B shown as shared may be the per-block reserve rather than static shared memory.
Confirm this before computing occupancy.

`isolated_index.json` records the mangled names, line counts and hashes.
