# Hardware ground truth for the supported GPUs

Every value was read live from the device (`nvidia-smi`, `cudaGetDeviceProperties()`). The calibrator (`framework/predictor/calibrate/cal/device.py`) refuses to run when a live SM count, L2 size or compute capability disagrees with the section for its GPU, and `extract_features.py` reads the occupancy rows. Add a section for any new GPU before calibrating it.

## Ada (RTX 5000 Ada Generation) — fully verified 2026-08-27

| Field | Value | Verified via |
|---|---|---|
| Name | NVIDIA RTX 5000 Ada Generation | `nvidia-smi --query-gpu=name` |
| Driver version | 595.84 | `nvidia-smi --query-gpu=driver_version` (was documented 595.71.05 -- wrong) |
| Compute capability | sm_89 (major=8, minor=9) | `cudaGetDeviceProperties()` |
| SM count | 100 | `cudaDeviceProp.multiProcessorCount` |
| L2 cache size | 67,108,864 bytes = exactly 64.0 MB | `cudaDeviceProp.l2CacheSize` |
| Total memory | 33,796,456,448 bytes = 31.5 GB (32760 MiB per `nvidia-smi`) | `cudaDeviceProp.totalGlobalMem` + `nvidia-smi` |
| Memory bus width | 256 bits | `cudaDeviceProp.memoryBusWidth` |
| Memory type | GDDR6 (not GDDR6X) | Cross-checked: 9001 MHz base clock x 256-bit bus x 2 (DDR) = 576 GB/s, matches NVIDIA's published spec for this card exactly |
| Power limit (current/default/max) | 250 W | `nvidia-smi --query-gpu=power.limit,power.default_limit,power.max_limit` (was documented 140W -- wrong by 110W/44%) |
| Power limit (min throttle floor) | 100 W | `nvidia-smi --query-gpu=power.min_limit` |
| Max graphics clock | 3105 MHz | `nvidia-smi -q -d SUPPORTED_CLOCKS` |
| Clock-lock target used (1065 MHz) | Confirmed a real, selectable clock step | `nvidia-smi -q -d SUPPORTED_CLOCKS \| grep 1065` |
| Normal idle power (empirical, not a spec) | ~60-70 W | Hundreds of already-validated rows in `calibration/paired_fixed_coeff/ada_*/raw/energy_samples.csv` |
| Shared mem per block | 49,152 bytes | `cudaDeviceProp.sharedMemPerBlock` |
| Shared mem per SM | 102,400 bytes | `cudaDeviceProp.sharedMemPerMultiprocessor` |
| Registers per SM | 65,536 | `cudaDeviceProp.regsPerMultiprocessor` |
| Max threads per SM | 1,536 | `cudaDeviceProp.maxThreadsPerMultiProcessor` |
| CUDA toolkit installed/used for compilation | 13.1 (`/usr/local/cuda-13.1/bin/nvcc`) | Direct check -- no 13.2 toolkit is installed, despite the driver supporting up to CUDA 13.2 API level |
| CUDA toolkits installed (re-checked 2026-10-02) | 12.8, 13.1 and **13.2** (`/usr/local/cuda-13.2`, build 37953736; `/usr/local/cuda-13` points to it). The row above is out of date on "no 13.2"; which toolkit Ada timing binaries use must be stated per campaign | `ls /usr/local/cuda-*/bin/nvcc` + `--version`, read-only over `jump-host`, 2026-10-02 (the authors) |

Occupancy-model rows (added 2026-10-03, labels as parsed by `predictor/extract_features.py`):

| Field | Value | Verified via |
|---|---|---|
| Warp size | 32 | live `cudaGetDeviceProperties()`, Ada booking `ada_hardware_inventory_20261002` (no kernel), 2026-10-03; raw `calibration/hardware_inventory/ada_20261002/` |
| Max threads per block / per SM | 1,024 / 1,536 | same live query as above |
| Registers per block / per SM | 65,536 / 65,536 | same live query as above |
| Maximum resident blocks per SM | 24 | same live query (`maxBlocksPerMultiProcessor`); agrees with Ada's own CUDA 13.2 `cuda_occupancy.h` (`case 8: minor == 9 → 24`) |
| Max resident blocks per SM | 24 | Same verified value as the row above (live `maxBlocksPerMultiProcessor`, Ada booking `ada_hardware_inventory_20261002`), repeated under the label `predictor/extract_features.py` parses, as in the H100 and A100 sections (added 2026-10-04, the authors) |
| Max warps per SM | 48 | arithmetic: 1,536 threads/SM / 32 |
| Reserved shared memory per block (driver carveout) | 1,024 bytes | same live query (`reservedSharedMemPerBlock`) |
| Shared memory per block (opt-in maximum) | 101,376 bytes | same live query (`sharedMemPerBlockOptin`) |
| Shared-memory allocation granularity | 128 bytes | Ada's own CUDA 13.2 `cuda_occupancy.h` (sha256 `4c8d0f75…`), `case 8/9/10/11/12: 128` |
| Register allocation granularity | 256 (registers/warp) | Ada's own CUDA 13.2 header, all majors: 256 |
| Max registers per thread | 256 | Ada's own CUDA 13.2 header, `case 7/8/9/10/11/12: 256` (informational, as for Blackwell) |
| Register sub-partitions per SM | 4 | Ada's own CUDA 13.2 header, `case 8: 4` |
| `cuobjdump SHARED` includes the reserve | no (declared size reported exactly; e.g. 32-wide matrix multiply reports 8,192 = two 32x32 float tiles) | `port_ada/compiled_ada_cuda13.2/log/*.res`, consistent with the M460 per-architecture finding |

Toolchain caveat (2026-10-03): the committed `calibration/device_inventory.cu`
does **not** compile under CUDA 13.2 — 13.2 removed
`cudaDeviceProp::clockRate`/`memoryClockRate`, which that source prints. The
Ada query above used a private variant with those two lines dropped (clocks
are not occupancy inputs); the shared file was left untouched for its owner
to fix. Any future inventory build under 13.2 needs the same treatment.

## Blackwell (RTX PRO 6000 Blackwell Workstation Edition, GPU 1) — partially verified 2026-08-27

| Field | Value | Verified via |
|---|---|---|
| Name | NVIDIA RTX PRO 6000 Blackwell Workstation Edition | `nvidia-smi -i 1 --query-gpu=name` |
| Driver version | 595.84 | Re-verified live on GPU 1, 2026-09-18 via `nvidia-smi -i 1 --query-gpu=driver_version`; supersedes the stale 570.211.01 entry |
| Compute capability | 12.0 (sm_120, matches `arch_sm=sm_120` already used in scripts) | `nvidia-smi --query-gpu=compute_cap` |
| Total memory | 97,887 MiB (~95.6 GiB) | `nvidia-smi --query-gpu=memory.total` |
| Power limit (current/default/max) | 600 W | `nvidia-smi --query-gpu=power.limit,power.default_limit,power.max_limit` |
| Power limit (min throttle floor) | 150 W | `nvidia-smi --query-gpu=power.min_limit` |
| SM count | 188 | `cudaGetDeviceProperties(&p, 1).multiProcessorCount` (small compiled probe, 2026-09-03) |
| L2 cache size | 134,217,728 bytes (128.0 MiB) | `cudaGetDeviceProperties(&p, 1).l2CacheSize`, same probe -- much larger than assumed while tuning `predictor_gather_load.cu`'s DRAM candidate (M101/M102 treated a 512KB-per-chunk footprint as safely DRAM-forcing; it is not, once multiplied across concurrently-resident blocks) |
| Total memory (re-verified) | 101,973,491,712 bytes (~95.0 GiB) | `cudaGetDeviceProperties(&p, 1).totalGlobalMem`, `calibration/device_inventory.cu` run under `CUDA_VISIBLE_DEVICES=1` on GPU1, 2026-09-15 (Task B sentinel-prep verification, see `calibration/hardware_inventory/blackwell_gpu1_20260915/`) |
| Memory bus width | 512 bits | same probe/run |
| DRAM read bandwidth (measured) | 1.64 TB/s (1,642 GB/s with SM locked at 1065 MHz; 1,649 GB/s at default clock); NCU `dram__throughput` 95.7-96.2% of peak | Streaming read, 4 GiB x 5 passes, `calibration/dram_validity_stream.cu` via `energy_harness/run_blackwell_dram_validity.sh` on GPU 1, 2026-09-25 (`calibration/dram_validity/blackwell_20260925_152201`, CHANGELOG M373). `dram__bytes_read`, `dram__sectors_read` x 32 and `fbpa__dram_read_bytes` all match the known 21,474,836,480 B within 0.1%. |
| Memory clock (at time of probe) | 14,001,000 kHz reported by CUDA (raw `memoryClockRate`) | same probe/run |
| Theoretical peak DRAM bandwidth (derived) | 1.792 TB/s = 2 x 14,001,000 kHz x 512 bit / 8 | 2 transfers per memory clock x memoryClockRate x memoryBusWidth / 8 (the CUDA deviceQuery convention) from the probe values above, added 2026-10-03 (the authors). Cross-check of the convention: the measured streaming read above (1.64 TB/s, NCU 95.7-96.2% of peak) is 91.6% of it |
| SM clock (at time of probe) | 2,617,000 kHz reported by CUDA (raw `clockRate`) -- an instantaneous boost reading, not a fixed spec | same probe/run |
| Warp size | 32 threads | same probe/run |
| Max threads per block / per SM | 1,024 / 1,536 | same probe/run |
| Registers per block / per SM | 65,536 / 65,536 | same probe/run |
| Shared mem per block | 49,152 bytes | same probe/run |
| Shared mem per SM | 102,400 bytes | same probe/run |
| Max resident blocks per SM | 24 | same probe/run |
| Shared-memory allocation granularity | 128 bytes | `/usr/local/cuda-12.8/targets/x86_64-linux/include/cuda_occupancy.h`, `cudaOccSMemAllocationGranularity()`: `case 8/9/10/12: value = 128;` (M328) |
| Register allocation granularity | 256 (registers/warp) | same header, `cudaOccRegAllocationGranularity()`: `case 3/5/6/7/8/9/10/12: value = 256;` (M328) |
| Max registers per thread | 256 | same header, `cudaOccRegAllocationMaxPerThread()`: `case 7/8/9/10/12: value = 256;` (M328; informational, not consumed by the project's occupancy formula) |
| Register sub-partitions per SM | 4 | `/usr/local/cuda-13.2/include/cuda_occupancy.h` (sha256 `4c8d0f75…`), `cudaOccSubPartitionsPerMultiprocessor()`: `case 3/5/7/8/9/10/11/12: value = 4;`, read on the Blackwell host 2026-10-01. Register-limited blocks per SM (`cudaOccMaxBlocksPerSMRegsLimit`, partitioned global caching off): regs/warp = roundup(regs/thread × 32, 256); warps/SM = floor((65,536 / 4) / regs-per-warp) × 4; blocks = floor(warps/SM ÷ warps/block). Hardware per-block check: regs-per-warp × roundup(warps/block, 4) ≤ registers per block. There is no other warp-allocation multiple (M444) |
| Reserved shared memory per block (driver carveout) | 1,024 bytes | Live `cudaGetDeviceProperties(&p, 1).reservedSharedMemPerBlock` on GPU1, read-only metadata query (no kernel launch, no allocation), 2026-09-22 -- not exposed by `cuda_occupancy.h`'s per-architecture tables, only by the live driver (M328) |
| Max warps per SM | 48 | Arithmetic: `max_threads_per_sm / warp_size` = 1536/32, cross-confirmed live by the same probe as the reserved-shared-memory row (M328) |
| CUDA toolkit installed/used for compilation | 13.2 (`/usr/local/cuda-13.2/bin/nvcc`) -- 12.8 and stock 12.0 also present, no 13.1 toolkit exists on this box | `dpkg -l \| grep -i "cuda-toolkit\|cuda-nvcc"` + `nvcc --version` per install path, live 2026-09-22 (M340). This table had no Blackwell CUDA-toolkit row before; `energy_harness/run_calibration_suite.py` had pinned "13.1" here by misattributing Ada's own 13.1 row (line 46 above) to Blackwell -- `energy_harness/measurement_runner.py`'s own comment already recorded the real split ("Blackwell has CUDA 12.8, Ada has 13.1") but the calibration-suite script's comment cited this file incorrectly anyway. Caught by A-GPU-BW's pre-launch smoke check before any real data was collected under the wrong pin. |

**Cross-checked finding (M328):** `cuobjdump --dump-resource-usage`'s `SHARED` field is **not** a kernel's
true declared shared-memory size on this architecture -- for a kernel with any static shared memory it
reports `declared_static_bytes + 1024` (the reserve above, added silently); for a kernel using only
*dynamic* shared memory (`extern __shared__`, size set at launch, e.g. Triton kernels and CUDA-Samples'
`reduction_kernel.cu`) it reports only `1024` regardless of the real launch-time size, since the dynamic
amount isn't in the compiled ELF at all. Verified two ways: (1) `workloads/e2e_transpose.cu`'s
`transpose_tiled` kernel declares `tile[32][33]` floats = 4,224 bytes; `cuobjdump` reports `SHARED:5248`
(4224+1024), while `nvcc -Xptxas -v` (which has no reserve concept) reports exactly `4224 bytes smem`.
(2) A Triton `softmax_kernel` cubin whose own compile-cache metadata JSON records `"shared": 520` (the
real per-launch dynamic size) shows `cuobjdump SHARED:1024`, not 1544. **Never use `cuobjdump`'s `SHARED`
value directly as a kernel's `shared_bytes_per_block`** -- subtract the reserve for static-shared kernels,
or use `ptxas -v` / the compiler's own metadata for dynamic-shared kernels. See
`tiresias/app_runners/compiler_resources_raw/blackwell_2026-09-22/00_platform_limits.txt` for the full
evidence and CHANGELOG M328.

**No verified L1 data-cache capacity for Blackwell as of this update (2026-09-15).** CUDA's
`cudaDeviceProp` does not expose an L1-cache-size field at all (only shared-memory-per-SM/block,
above) -- this matches the same limitation already noted for A100/H100 in this file: shared memory
capacity is **not** a verified L1 capacity, and no size/timing constant should be chosen to land a
workload in the L1 tier from this table alone. An architecture-specific residency gate (as A100/H100
already require) is the only way to confirm an actual L1-resident footprint on Blackwell; none has
been run yet. Task B's sentinel grid (`tiresias/app_runners/measurement_budget.csv`) therefore
treats its "L1" candidate tier as "small, within the 102,400-byte shared-memory-per-SM budget and
well under the 128 MiB L2 size" rather than as a confirmed L1-cache-verified footprint, and this gap
is recorded as an open dependency in `sentinel_grid_2026-09-15.md`, not silently assumed past.

## A100 (A100-SXM4-80GB, Cluster GPU 4) — verified 2026-09-08

Source: non-NCU scheduler inventory job `j02`, pinned commit
`6c71bb50543e59d1c71878e0f358fa34f512e7fe`. Raw output:
`calibration/hardware_inventory/a100_j02/`.

| Field | Value | Verified via |
|---|---|---|
| Name | NVIDIA A100-SXM4-80GB | `nvidia-smi` + `cudaGetDeviceProperties()` |
| Driver version | 580.126.20 | `nvidia-smi` |
| Compute capability | sm_80 (8.0) | `cudaGetDeviceProperties()` |
| SM count | 108 | `cudaDeviceProp.multiProcessorCount` |
| L2 cache size | 41,943,040 bytes = 40.0 MiB | `cudaDeviceProp.l2CacheSize` |
| Total memory | 85,093,777,408 bytes; 81,920 MiB | CUDA + `nvidia-smi` |
| Memory bus width | 5,120 bits | `cudaDeviceProp.memoryBusWidth` |
| Power limit (current/default/max/min) | 400 / 400 / 400 / 100 W | `nvidia-smi` |
| SM / memory clocks reported at inventory | 1410 / 1593 MHz | `cudaDeviceProp` |
| Theoretical peak DRAM bandwidth (derived) | 2.039 TB/s = 2 x 1,593,000 kHz x 5,120 bit / 8 | 2 transfers per memory clock x memoryClockRate x memoryBusWidth / 8 (the CUDA deviceQuery convention), from the live `memory_clock_khz=1593000` and `memory_bus_width_bits=5120` of cluster job j06 (2026-10-02, no kernel; raw `calibration/hardware_inventory/a100_j06/cuda_properties.txt`), added 2026-10-03 (the authors) |
| Warp / thread limits | 32 threads/warp; 1,024 threads/block; 2,048 threads/SM | CUDA properties |
| Registers per block / SM | 65,536 / 65,536 | CUDA properties |
| Shared memory per block / SM | 49,152 / 167,936 bytes | CUDA properties |
| Maximum resident blocks per SM | 32 | CUDA property |
| CUDA toolkit used for compilation (Cluster) | 12.1, through the spack package pinned in `energy_harness/cluster_resolve_toolchain.sh` (refuses any other release). The login node lists spack CUDA modules 9.2 to 12.1 only; `/usr/local/cuda-12.3` exists but has no usable `nvcc` on the login node | `module -t avail`, `ls /usr/local/cuda*`, read-only on `hpc01`, 2026-10-02 (the authors) |

The CUDA shared-memory field is **not** a verified L1 capacity. Do not choose an L1/L2/DRAM
footprint from it alone; first run an architecture-specific residency gate.

Occupancy-model rows (added 2026-10-02, labels as parsed by `predictor/extract_features.py`):

| Field | Value | Verified via |
|---|---|---|
| Warp size | 32 | live `cudaGetDeviceProperties()`, cluster job j06 (commit `b9f3a97e`, no kernel), 2026-10-02; raw `calibration/hardware_inventory/a100_j06/` |
| Max threads per block / per SM | 1,024 / 2,048 | live `cudaGetDeviceProperties()`, cluster job j06 (commit `b9f3a97e`, no kernel), 2026-10-02; raw `calibration/hardware_inventory/a100_j06/` |
| Registers per block / per SM | 65,536 / 65,536 | live `cudaGetDeviceProperties()`, cluster job j06 (commit `b9f3a97e`, no kernel), 2026-10-02; raw `calibration/hardware_inventory/a100_j06/` |
| Shared mem per block | 49,152 bytes | live `cudaGetDeviceProperties()`, cluster job j06 (commit `b9f3a97e`, no kernel), 2026-10-02; raw `calibration/hardware_inventory/a100_j06/` |
| Shared mem per SM | 167,936 bytes | live `cudaGetDeviceProperties()`, cluster job j06 (commit `b9f3a97e`, no kernel), 2026-10-02; raw `calibration/hardware_inventory/a100_j06/` |
| Opt-in shared memory per block (cudaFuncSetAttribute limit) | 166,912 bytes | live `cudaGetDeviceProperties()`, cluster job j06 (commit `b9f3a97e`, no kernel), 2026-10-02; raw `calibration/hardware_inventory/a100_j06/` |
| Max resident blocks per SM | 32 | live `cudaGetDeviceProperties()`, cluster job j06 (commit `b9f3a97e`, no kernel), 2026-10-02; raw `calibration/hardware_inventory/a100_j06/` |
| Reserved shared memory per block (driver carveout) | 1,024 bytes | live `cudaGetDeviceProperties()`, cluster job j06 (commit `b9f3a97e`, no kernel), 2026-10-02; raw `calibration/hardware_inventory/a100_j06/` (`reservedSharedMemPerBlock`) |
| Max warps per SM | 64 | Arithmetic: 2,048 threads per SM / 32 |
| Shared-memory allocation granularity | 128 bytes | `cuda_occupancy.h` (local CUDA 13.0 copy, sha256 `24ec2676...`; same case tables as the 13.2 header cited for Blackwell), `cudaOccSMemAllocationGranularity()`: `case 8/9: 128` |
| Register allocation granularity | 256 (registers/warp) | `cuda_occupancy.h` (local CUDA 13.0 copy, sha256 `24ec2676...`; same case tables as the 13.2 header cited for Blackwell), `cudaOccRegAllocationGranularity()`: `case 8/9: 256` |
| Max registers per thread | 256 | `cuda_occupancy.h` (local CUDA 13.0 copy, sha256 `24ec2676...`; same case tables as the 13.2 header cited for Blackwell), `cudaOccRegAllocationMaxPerThread()`: `case 8/9: 256` (informational, as for Blackwell) |
| Register sub-partitions per SM | 4 | `cuda_occupancy.h` (local CUDA 13.0 copy, sha256 `24ec2676...`; same case tables as the 13.2 header cited for Blackwell), `cudaOccSubPartitionsPerMultiprocessor()`: `case 8/9: 4` |

## H100 (H100 80GB HBM3, Cluster GPU 5) — verified 2026-09-08

Source: non-NCU scheduler inventory job `j03`, same pinned commit. Raw output:
`calibration/hardware_inventory/h100_j03/`.

| Field | Value | Verified via |
|---|---|---|
| Name | NVIDIA H100 80GB HBM3 | `nvidia-smi` + `cudaGetDeviceProperties()` |
| Driver version | 580.126.20 | `nvidia-smi` |
| Compute capability | sm_90 (9.0) | `cudaGetDeviceProperties()` |
| SM count | 132 | `cudaDeviceProp.multiProcessorCount` |
| L2 cache size | 52,428,800 bytes = 50.0 MiB | `cudaDeviceProp.l2CacheSize` |
| Total memory | 85,017,493,504 bytes; 81,559 MiB | CUDA + `nvidia-smi` |
| Memory bus width | 5,120 bits | `cudaDeviceProp.memoryBusWidth` |
| Power limit (current/default/max/min) | 700 / 700 / 700 / 200 W | `nvidia-smi` |
| SM / memory clocks reported at inventory | 1980 / 2619 MHz | `cudaDeviceProp` |
| Theoretical peak DRAM bandwidth (derived) | 3.352 TB/s = 2 x 2,619,000 kHz x 5,120 bit / 8 | 2 transfers per memory clock x memoryClockRate x memoryBusWidth / 8 (the CUDA deviceQuery convention), from the live `memory_clock_khz=2619000` and `memory_bus_width_bits=5120` of cluster job j07 (2026-10-02, no kernel; raw `calibration/hardware_inventory/h100_j07/cuda_properties.txt`), added 2026-10-03 (the authors). Any measured or fitted H100 DRAM rate above this is not a DRAM rate (CHANGELOG M477: the calibrator's 4.36 TB/s) |
| Warp / thread limits | 32 threads/warp; 1,024 threads/block; 2,048 threads/SM | CUDA properties |
| Registers per block / SM | 65,536 / 65,536 | CUDA properties |
| Shared memory per block / SM | 49,152 / 233,472 bytes | CUDA properties |
| Maximum resident blocks per SM | 32 | CUDA property |
| CUDA toolkit used for compilation (Cluster) | 12.1, same spack package as A100 (`energy_harness/cluster_resolve_toolchain.sh`) | read-only check on `hpc01`, 2026-10-02 (the authors) |

The CUDA shared-memory field is **not** a verified L1 capacity. Do not choose an L1/L2/DRAM
footprint from it alone; first run an architecture-specific residency gate.

Occupancy-model rows (added 2026-10-02, labels as parsed by `predictor/extract_features.py`):

| Field | Value | Verified via |
|---|---|---|
| Warp size | 32 | live `cudaGetDeviceProperties()`, cluster job j07 (commit `b9f3a97e`, no kernel), 2026-10-02; raw `calibration/hardware_inventory/h100_j07/` |
| Max threads per block / per SM | 1,024 / 2,048 | live `cudaGetDeviceProperties()`, cluster job j07 (commit `b9f3a97e`, no kernel), 2026-10-02; raw `calibration/hardware_inventory/h100_j07/` |
| Registers per block / per SM | 65,536 / 65,536 | live `cudaGetDeviceProperties()`, cluster job j07 (commit `b9f3a97e`, no kernel), 2026-10-02; raw `calibration/hardware_inventory/h100_j07/` |
| Shared mem per block | 49,152 bytes | live `cudaGetDeviceProperties()`, cluster job j07 (commit `b9f3a97e`, no kernel), 2026-10-02; raw `calibration/hardware_inventory/h100_j07/` |
| Shared mem per SM | 233,472 bytes | live `cudaGetDeviceProperties()`, cluster job j07 (commit `b9f3a97e`, no kernel), 2026-10-02; raw `calibration/hardware_inventory/h100_j07/` |
| Opt-in shared memory per block (cudaFuncSetAttribute limit) | 232,448 bytes | live `cudaGetDeviceProperties()`, cluster job j07 (commit `b9f3a97e`, no kernel), 2026-10-02; raw `calibration/hardware_inventory/h100_j07/` |
| Max resident blocks per SM | 32 | live `cudaGetDeviceProperties()`, cluster job j07 (commit `b9f3a97e`, no kernel), 2026-10-02; raw `calibration/hardware_inventory/h100_j07/` |
| Reserved shared memory per block (driver carveout) | 1,024 bytes | live `cudaGetDeviceProperties()`, cluster job j07 (commit `b9f3a97e`, no kernel), 2026-10-02; raw `calibration/hardware_inventory/h100_j07/` (`reservedSharedMemPerBlock`) |
| Max warps per SM | 64 | Arithmetic: 2,048 threads per SM / 32 |
| Shared-memory allocation granularity | 128 bytes | `cuda_occupancy.h` (local CUDA 13.0 copy, sha256 `24ec2676...`; same case tables as the 13.2 header cited for Blackwell), `cudaOccSMemAllocationGranularity()`: `case 8/9: 128` |
| Register allocation granularity | 256 (registers/warp) | `cuda_occupancy.h` (local CUDA 13.0 copy, sha256 `24ec2676...`; same case tables as the 13.2 header cited for Blackwell), `cudaOccRegAllocationGranularity()`: `case 8/9: 256` |
| Max registers per thread | 256 | `cuda_occupancy.h` (local CUDA 13.0 copy, sha256 `24ec2676...`; same case tables as the 13.2 header cited for Blackwell), `cudaOccRegAllocationMaxPerThread()`: `case 8/9: 256` (informational, as for Blackwell) |
| Register sub-partitions per SM | 4 | `cuda_occupancy.h` (local CUDA 13.0 copy, sha256 `24ec2676...`; same case tables as the 13.2 header cited for Blackwell), `cudaOccSubPartitionsPerMultiprocessor()`: `case 8/9: 4` |

## Cluster GPU UUIDs

Which physical GPUs Cluster's H100 and A100 nodes have, by full UUID. A cluster job does not know in advance which GPU it gets, and the packaged calibrator
(`tiresias/framework/predictor/calibrate/`) refuses any UUID that is not in `calibrate/approved_devices.json`; this table is the human-readable copy of
those entries. Status: **partial (2026-10-03): 3 of the 8 H100 GPUs, 1 of the 8 A100 GPUs** (the A100 one was identified twice, by UUID job j10 and by the approve-allocated evidence of calibration job j15, which landed on the same GPU); filled from `energy_harness/cluster_list_gpu_uuids.sh` (a no-kernel cluster job that records `nvidia-smi` identity; then
`calibrate/tools/add_cluster_uuids.py <job output dir> --write` adds the rows below and the allow-list entries together). Do not type UUIDs in by hand and do not copy
them from anywhere but a job's own output. The allow-list is keyed by UUID, so several GPUs of one model are listed side by side. A GPU approved through the `APPROVE_ALLOCATED=1` opt-in of `energy_harness/run_calibration_cluster.sh` is added afterwards, from that job's `uuid_evidence/` directory, with `add_cluster_uuids.py` (`--allocated-only --write`).

Cluster's GPU cgroups show a job only its own GPU, so `Index` is always 0 inside the job; the PCI bus id identifies the physical GPU. A job landing on a GPU not listed here is refused by the calibrator before any GPU work; add that GPU from the job's own output and resubmit.

| Node | Index | UUID | PCI bus | GPU name | Source |
|---|---|---|---|---|---|
| node-5.cluster.example.org | 0 | GPU-74e7b308-137d-cdb1-3c89-ba3186450254 | 00000000:04:00.0 | NVIDIA H100 80GB HBM3 | cluster job j08 (energy_harness/cluster_list_gpu_uuids.sh) |
| node-6.cluster.example.org | 0 | GPU-90376b5d-6fd8-1ffa-43fa-d42fd95159f1 | 00000000:03:00.0 | NVIDIA H100 80GB HBM3 | cluster job j09 (energy_harness/cluster_list_gpu_uuids.sh) |
| node-6.cluster.example.org | 0 | GPU-6132e1b7-a6ba-9eea-9084-bcb0ec15009a | 00000000:E4:00.0 | NVIDIA H100 80GB HBM3 | cluster job j11 (nvidia-smi query run as a pair with 377052 so two free node6 GPUs were held at once) |
| node-4.cluster.example.org | 0 | GPU-f593bab2-a5ab-0f04-b0ad-f9ace54dfff3 | 00000000:4D:00.0 | NVIDIA A100-SXM4-80GB | cluster job j10 (energy_harness/cluster_list_gpu_uuids.sh) |

