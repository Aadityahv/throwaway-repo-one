# Artifact description

This file states what is in the artifact, what can be reproduced without a GPU, and the limits of the package.
`README.md` describes the layout and how to run things.

## 1. Contents

| Component | Where | Needs a GPU? |
|---|---|---|
| SASS interpreter and static feature extraction | `tiresias/framework/predictor/extract_features.py`, `phases.py` | no (reads retained `.sass`/`.cubin`) |
| Runtime model (layered v2 ... v3k; v3k is final) | `tiresias/framework/predictor/predict_runtime_v*.py`, `shared_traffic/`, `constants/` | no |
| Per-GPU calibrator, CUDA microbenchmarks, energy model | `tiresias/framework/predictor/calibrate/` | calibration: yes; prediction: no |
| Evaluated kernels (sources, compiled SASS, cells, frozen predictions, measurements) | `fresh*`, `unseen_kernels`, `prospective_test`, `port_*`, `replication_*` under `predictor/` | no to read; yes to re-measure |
| Evaluation results table | `tiresias/evaluation/results/*.csv` | no |
| Energy-measurement harness | `energy_harness/`, `tiresias/app_runners/` | yes |
| CPU re-prediction benchmark | `tiresias/framework/cpu_prediction_benchmark/` | no |

## 2. Predictions are frozen before measurement

For the prospective evidence, predictions were written to a frozen record (with a SHA-256 of the prediction file) before
any timing or energy window of those cells was taken; the scoring scripts read only the frozen file.
`predictor/prospective_test/` holds the 24 prospective cells, `PREDICTIONS_FROZEN.json` and the scoring code; the
replication directories hold the corresponding freeze records for the other GPUs.

**Which predictor is the system's.** There is one predictor: the runtime model in `predict_runtime_v3k.py`
(`calibrate/predict.py` for energy). `PREDICTIONS_FROZEN.json` records two sets of predictions made before any timing or
energy window of the 24 prospective cells: `models.current` is the previous runtime model (`predict_runtime_v3j.py`, the
shared-traffic model that was current when the file was written), and `models.proposed` is the model that adds the
pair-overlap, stage-serialisation and busiest-SM rules (`v3k`). Both were frozen so that the rules could be tested against
the model they extend. The `proposed` model passed its pre-registered criteria on those cells and is the only model
reported as the system's result; the file's label `current` is historical. On the 24 prospective cells it gives a median
static-runtime energy error of 12.2% against 24.5% for the previous model, and a median runtime error of 6.2% against 21.5%.

**Note on recorded hashes.** This package was anonymised: author, host, account and path identifiers were rewritten,
including inside archives and JSON provenance, and directories were renamed. Every SHA-256 value pinned in a freeze
record, manifest, gate file or test was then re-derived mechanically (for each file whose bytes changed, its original
hash was replaced everywhere by the hash of the anonymised file, repeating until the chain of records that hash other
records was stable). The integrity checks therefore attest the files as released. The numerical content (predictions,
measurements, scores) is unchanged.

## 3. Reproducing without a GPU

`python3 reproduce.py predict` recomputes every prediction from the retained compiled kernels and checks it against the
frozen prediction or refusal (relative tolerance 1e-9, identical refusal status). `reproduce.py test` runs the unit
tests. Cold analysis of the largest matrix-multiply kernels (not run by `predict`) is slow and memory hungry (more than
12 GiB for one worker).

## 4. Reproducing with a GPU

* **Calibration** (once per GPU, no application kernel): `calibrate/run_calibration.py` (see `README.md`). The stages are:
  `launch_chain`, `launch_reuse` (kernel gaps; per-SM re-read footprint and L1 capacity), `stream` (L2 and DRAM
  bandwidth), `mlp` (memory-level-parallelism latency per tier), `smem_volatile`/`smem_plain` (shared-request cost versus
  bank-conflict degree), `overlap` (overlap of other blocks' phases), `store`, `pipes` (per-pipe issue cost, latencies,
  barrier cost, pointer-chase latencies), `tensor` (tensor-core MMA) and `energy` (27 windows run, energy per byte by tier
  and per instruction class, and base power). Windows too close to the board power limit are rejected: on the RTX PRO 6000,
  6 of the 27 are rejected, leaving 21 admitted windows, 18 used to fit the rates and 3 held out for validation; H100 and
  RTX 5000 use 23 fit and 4 held-out windows. Sources are `calibrate/csrc/*.cu`. Requires CUDA >= 12.8 on Blackwell, an idle
  GPU, and an entry for the GPU in `approved_devices.json` and `HARDWARE_GROUND_TRUTH.md`. The calibrator refuses to run
  otherwise and never changes clocks or power limits.
* **Energy windows**: `energy_harness/run_application_energy.py` with the adapters in `tiresias/app_runners/`. Kernels are
  replayed from a CUDA graph (1000 launches per replay) so that a window measures the device, not host launch overhead.
  Power is sampled in-process via NVML. Windows whose power is too close to the board power limit are rejected.
* GPUs used: RTX PRO 6000 Blackwell (primary), H100, A100, RTX 5000 Ada. Their verified properties are in
  `HARDWARE_GROUND_TRUTH.md`. CUDA 13.2 (Blackwell, Ada) and 12.1 (H100, A100) toolchains were used.

## 5. Raw data and archives

Per-window raw sample files and calibration stage outputs are stored as `.tar.gz` next to the directory they belong to
(`raw_attempts.tar.gz`, `stages/<stage>.tar.gz`). Scripts read the summary tables, not these archives; extract them in
place to audit individual windows. GPU UUIDs are replaced by pseudonyms that keep the first eight hex characters.
Cluster job numbers, partition and node names were replaced by neutral tokens (`jNN`, `partition_*`, `node*`).

## 6. Known limits

* **Checks.** The unit tests and `reproduce.py predict` were run on a Linux host (Python 3.12, CUDA 13.2 command-line
  tools on `PATH`; `cuobjdump` must be from CUDA 13.x to read `sm_120` cubins). Open items: `replication_ada_rtx5000/test_ada.py`
  (the regenerated build manifest differs from the shipped one) and one assertion in `test_extract_features.py`
  (the 36 Triton rows fall back to the launch-configuration note instead of the cubin-hash match; cause not isolated).
  Neither affects a reported number. The multi-process cached prediction run stalled once at 165/167 under load;
  re-running completes.
* **Development-time inputs not included.** A few freeze and exploration scripts read inputs from earlier development
  stages. They are not needed for the CPU re-prediction and are omitted, as are tests of the superseded first calibration
  iteration. Some scoring pipelines still read frozen input files for the prior methods that the evaluation compared
  against; the code that produced those inputs is not part of this artifact.
* **Third-party code is not redistributed.** The CUDA Samples (commit `5443602d89ed99aede2e4b7bf329daddeadb320e`) must be
  fetched from their upstream repository.
* **Job scripts omitted.** Cluster submission scripts are not included.
* **Licence.** MIT (see `LICENSE`); third-party material keeps its own notices.
