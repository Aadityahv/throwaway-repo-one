"""Parallel form of static_tables.py: the same per-cell analysis (static_ada.build_one with the board binding), run by a pool of worker processes, heaviest cells first.

    python static_parallel.py --board h100 --jobs 8 [--only substr] [--list]

Same cache files as static_tables.py (<board>/build/<group>/<cell>.json), so cells already analysed by either tool are skipped and the outputs are interchangeable. It builds cells only; run
`python static_tables.py --board <board> --merge-only` afterwards to write the merged static tables. CPU only, no GPU, no measured value. Needs the `fork` start method (Linux).
"""
import argparse
import concurrent.futures as cf
import multiprocessing as mp
import sys
import time

import board as B
import static_tables as ST

HEAVY = ("matrix_multiply", "fp32_matmul", "tensor", "attention", "attn", "tcgemm", "tc_", "prosp")


def weight(cell_id):
    return 0 if any(h in cell_id for h in HEAVY) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--board", required=True, choices=B.CONFIG)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true", help="print the cells that would be built and exit")
    a = ap.parse_args()
    mod = ST.setup(a.board)
    todo = []
    for g in mod.MA.GROUPS:
        for c in mod.MA.load_group(g)["cells"]:
            if a.only and a.only not in c["cell_id"]:
                continue
            if ST.cell_sass_ready(mod, c) and not mod.cache_path(g, c["cell_id"]).exists():
                todo.append((g, c["cell_id"]))
    todo.sort(key=lambda t: (weight(t[1]), t[1]))
    print("%s: %d cells to build, %d workers" % (a.board, len(todo), a.jobs), flush=True)
    if a.list:
        for g, cid in todo:
            print(g, cid)
        return 0
    if not todo:
        return 0
    t0, done = time.time(), 0
    with cf.ProcessPoolExecutor(max_workers=a.jobs, mp_context=mp.get_context("fork")) as pool:
        futs = {pool.submit(mod.build_one, t): t for t in todo}
        for f in cf.as_completed(futs):
            cid, msg = f.result()
            done += 1
            print("[%d/%d %6.0fs] %-72s %s" % (done, len(todo), time.time() - t0, cid, msg), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
