#!/usr/bin/env python3
"""Checks dispatch_trace.json against (a) hand-derived literals for two cells per
operator and (b) the encoded rules for all cells. CPU-only. Run: python3 test_dispatch.py"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dispatch_rules as R  # noqa: E402

T = json.loads((HERE / "dispatch_trace.json").read_text())
CELLS = {c["cell_id"]: c for c in T["cells"]}


def kern(cid, role_prefix):
    return [k for k in CELLS[cid]["kernels"] if k["role"].startswith(role_prefix)][0]


def test_cell_count():
    assert len(CELLS) == 27, len(CELLS)
    ops = {}
    for c in T["cells"]:
        ops[c["operator_id"]] = ops.get(c["operator_id"], 0) + 1
    assert ops == {"dev_pytorch_rowwise_softmax": 12, "final_pytorch_layer_norm": 12, "final_pytorch_embedding": 3}, ops


def test_source_hashes():
    for line in (HERE / "src/SHA256SUMS").read_text().splitlines():
        h, n = line.split()
        assert hashlib.sha256((HERE / "src" / n).read_bytes()).hexdigest() == h, n


def test_softmax_literals():
    # medium/c1: 4096x512 contiguous. log2_ceil(512)=9, warp 32, 1 batch/warp, 4 warps, 4 rows/block -> 1024 blocks.
    c = CELLS["blackwell/dev_pytorch_rowwise_softmax/medium/c1"]
    assert c["kernels_per_call"] == 1
    k = c["kernels"][0]
    assert k["grid"] == [1024, 1, 1] and k["block"] == [32, 4, 1] and k["dynamic_smem_bytes"] == 0
    assert k["template_args"]["log2_elements"] == 9
    assert re.search(k["demangled_name_regex"], "softmax_warp_forward<float, float, float, 9, false, false>")
    # small/c3: 1024x128 padded stride 131 -> copy kernel (131072/256 = 512 blocks) + warp softmax L2E=7,
    # 2 batches/warp, 8 rows/block -> 128 blocks.
    c = CELLS["blackwell/dev_pytorch_rowwise_softmax/small/c3"]
    assert c["kernels_per_call"] == 2
    cp, sm = c["kernels"]
    assert cp["grid"] == [512, 1, 1] and cp["block"] == [128, 1, 1]
    assert sm["grid"] == [128, 1, 1] and sm["block"] == [32, 4, 1] and sm["template_args"]["log2_elements"] == 7
    # large/c2: 8192x1024 stride 1025 -> copy 8388608/256=32768 blocks; softmax L2E=10, 4 rows/block -> 2048 blocks
    c = CELLS["blackwell/dev_pytorch_rowwise_softmax/large/c2"]
    assert [k["grid"][0] for k in c["kernels"]] == [32768, 2048]


def test_layernorm_literals():
    # medium/c1: 4096x512 contiguous: single vectorized kernel, grid 4096, block (32,4), smem 4*3/2*4=24 B.
    c = CELLS["blackwell/final_pytorch_layer_norm/medium/c1"]
    assert c["kernels_per_call"] == 1
    k = c["kernels"][0]
    assert k["grid"] == [4096, 1, 1] and k["block"] == [32, 4, 1] and k["dynamic_smem_bytes"] == 24
    assert k["args"]["n_vec_to_read"] == 128 and k["args"]["max_vec_iterations_per_thread"] == 1
    # large/c4: 8192x1024 stride 1031 -> copy (32768 blocks) + vectorized kernel (8192 blocks)
    c = CELLS["blackwell/final_pytorch_layer_norm/large/c4"]
    assert c["kernels_per_call"] == 2
    assert c["kernels"][0]["grid"] == [32768, 1, 1]
    k = c["kernels"][1]
    assert k["grid"] == [8192, 1, 1] and k["args"]["n_vec_to_read"] == 256 and k["args"]["max_vec_iterations_per_thread"] == 2
    for cc in CELLS.values():
        if cc["operator_id"] == "final_pytorch_layer_norm":
            assert not any("RowwiseMoments" in k["demangled_name_pattern"] for k in cc["kernels"])


def test_embedding_literals():
    # small: 1024 idx x 128 floats = 512 B rows -> 32 threads, grid (1024,1,1)
    k = CELLS["blackwell/final_pytorch_embedding/small/c1"]["kernels"][0]
    assert k["grid"] == [1024, 1, 1] and k["block"] == [32, 1, 1]
    assert k["template_args"] == {"Alignment": 16, "index_t": "int64_t"}
    # large: 16384 idx x 512 floats = 2048 B rows -> 128 threads, grid (16384,1,1)
    k = CELLS["blackwell/final_pytorch_embedding/large/c1"]["kernels"][0]
    assert k["grid"] == [16384, 1, 1] and k["block"] == [128, 1, 1]
    assert k["args"]["ind_dim_size"] == 1048576 and k["args"]["slice_size_bytes"] == 2048


def test_all_cells_match_rules():
    for cid, c in CELLS.items():
        ctl = c["controls"]
        op = c["operator_id"]
        if op == "dev_pytorch_rowwise_softmax":
            rows, cols, stride = ctl["outer_size"], ctl["dim_size"], ctl["row_stride"]
            g = R.softmax_persistent_geometry(rows, cols)
            assert c["kernels"][-1]["grid"] == g["grid"] and c["kernels"][-1]["block"] == g["block"], cid
            assert c["kernels_per_call"] == (1 if stride == cols else 2), cid
            if stride != cols:
                assert c["kernels"][0]["grid"] == R.copy_geometry(rows * cols)["grid"], cid
        elif op == "final_pytorch_layer_norm":
            rows, cols, stride = ctl["rows"], ctl["cols"], ctl["row_stride"]
            g = R.layernorm_geometry(rows, cols)
            k = c["kernels"][-1]
            assert k["grid"] == g["grid"] and k["block"] == g["block"] and k["dynamic_smem_bytes"] == g["smem"], cid
            assert c["kernels_per_call"] == (1 if stride == cols else 2), cid
        else:
            g = R.gather_geometry(ctl["num_indices"], ctl["feature_dim"])
            k = c["kernels"][0]
            assert k["grid"] == g["grid"] and k["block"] == g["block"] and c["kernels_per_call"] == 1, cid


def test_controls_match_raw_csv():
    base = HERE.parents[2] / "app_runners/application_energy_raw"
    f = base / "FINAL-PYTORCH-S2-20260922T204600Z/application_energy_raw.csv"
    if not f.exists():
        return
    seen = 0
    for r in csv.DictReader(open(f)):
        cid = f"blackwell/{r['parent_id']}/{r['regime']}/{r['candidate_id']}"
        if cid in CELLS:
            assert CELLS[cid]["controls"] == json.loads(r["controls_json"]), cid
            seen += 1
    assert seen == 15, seen


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("ok", name)
    print(f"DISPATCH_TESTS_OK: {n} tests")
