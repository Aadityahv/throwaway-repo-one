# Tiresias: microbenchmark-calibrated static prediction of GPU kernel runtime and energy

Tiresias predicts the runtime and energy of a CUDA kernel **without executing or profiling the target kernel**.
The work a kernel performs is fixed by its machine code (SASS) and launch arguments; how that work overlaps on the
hardware is a property of the GPU. Tiresias therefore

1. **reads the work from the SASS** (instruction mix, per-phase memory traffic, shared-memory bank conflicts,
   cache-tier footprints) by interpreting a few sample blocks on the CPU,
2. **measures the hardware once per GPU** with microbenchmarks (the calibration; no application kernel is run), and
3. **composes** the two with a phase-based runtime model and a component energy model.

This repository contains the framework, the calibrator and its CUDA microbenchmarks, the evaluated kernels with their
frozen predictions and measured outputs, and a CPU benchmark that re-predicts every evaluated configuration.
`ARTIFACT.md` describes the data, what can be reproduced without a GPU, and the known limits.

## Layout

```
HARDWARE_GROUND_TRUTH.md           live-verified SM counts, L2 sizes, compute capability and occupancy limits per GPU
requirements-cpu.txt               Python packages for the CPU-only analysis
reproduce.py                       entry point: `predict` (re-predict all configurations) and `test`
energy_harness/                    energy-measurement harness: graph-replay launcher, NVML power sampler, window runner
analysis/                          paired-comparison statistics used by the kernel-set scorer
tiresias/
  framework/
    predictor/                     THE FRAMEWORK
      extract_features.py          SASS interpreter and feature extraction (work per kernel)
      phases.py                    barrier-delimited phase construction
      predict_runtime_v2.py ... predict_runtime_v3k.py
                                   runtime model, layered: each version wraps and reuses the one below it; v3k is final
      shared_traffic/              grid-wide DRAM footprint, per-wave L2 rule, refusal rules
      constants/                   pair-overlap constants for tensor-core kernels and their generator
      calibrate/                   one-command per-GPU calibration (run_calibration.py), the energy model (predict.py),
                                   CUDA microbenchmarks (csrc/), fitting code (cal/), tests (tests/), calibration
                                   documents of every GPU (runs/)
      prospective_test/            24 configurations predicted before measurement: frozen predictions and scores
      fresh, fresh_b ... fresh_h   evaluated kernel sets (CUDA-samples kernels, machine-learning kernels,
      unseen_kernels               tensor-core matrix multiply, fused attention): CUDA sources, compiled SASS, cells,
                                   frozen predictions, measured timing and energy
      bank/, coalescing/, pytorch_dispatch/, libtorch_sm120/
                                   shared-memory bank-conflict, coalescing and library-dispatch static analysis
      port_a100/, port_h100/, port_ada/, port_common/
                                   the evaluated kernels compiled for another GPU (SASS per architecture)
      replication_h100_a100/, replication_ada_rtx5000/, h100_eval_freeze/, h100_power_sensor_test/
                                   cross-GPU runs: cells, frozen predictions, measurements, scores
      micro_v3/, repair/, reuse/, energy_revision/, blackwell_experiments/
                                   earlier calibration programs and fixtures for the calibrator tests
    evaluation_data/measured/kernel_sets/
                                   labels and frozen predictions of the kernel sets
    compile_evidence/              compile evidence and instruction-count derivations read by extract_features.py
    cpu_prediction_benchmark/      CPU re-prediction benchmark of all evaluated configurations
    adapters/, unseen_operators/   operator adapters and runners
  evaluation/results/              eval_cells.csv, eval_validation.csv, eval_prospective.csv: every evaluated
                                   configuration with Tiresias's prediction and the measured value
  app_runners/                     application runners and adapters (vector add, softmax, layer norm, copy, reduction,
                                   transpose, ...), correctness references and small input tables
```

### Model files

`predict_runtime_v3k.py` (the final model) wraps and reuses `v3j`, which reuses `v3i`, `v3h`, `v3f`, `v3` and `v2`; all
are required. Files named `predictions_v3*_dev_blackwell.json` and `score_v3*_dev_blackwell.json` are development-time
outputs of intermediate versions and are not used for any reported result. `micro_v3/` supplies constants that `v3h` and
`v3i` read, and `repair/` holds earlier calibration programs that `calibrate/` replaces, with the tests that pin them down.

Directory names are historical labels kept because scripts and recorded provenance refer to them by path. A date or
`jNN` token in a file name is a run identifier.

## Quick start (no GPU)

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements-cpu.txt
export CUDA_VISIBLE_DEVICES=
.venv/bin/python reproduce.py predict --jobs 4   # re-predict all 167 configurations and compare with the frozen record
.venv/bin/python reproduce.py test               # unit tests of the framework and calibrator
```

`predict` recomputes every runtime and energy prediction from the retained SASS and the calibration documents and checks
it against the frozen prediction or refusal (relative tolerance 1e-9). It writes only to `predict_out/`.

## Running on a new GPU

1. Add a verified section for the GPU to `HARDWARE_GROUND_TRUTH.md` (SM count, L2 size, compute capability).
2. Add the GPU to `tiresias/framework/predictor/calibrate/approved_devices.json`. The calibrator refuses any device that
   is not listed or whose live properties disagree with the ground-truth file.
3. `python3 tiresias/framework/predictor/calibrate/run_calibration.py --device-uuid GPU-<uuid> --booking-ref <text> --nvcc <path> [--plan]`
   (`--plan` prints the stages and runs nothing). It needs an idle GPU and CUDA >= 12.8 for Blackwell (`sm_120`);
   it never changes clocks, persistence mode or the power limit.
4. `calibrate/predict.py` turns the resulting calibration document and the static features of a kernel into runtime and energy.
