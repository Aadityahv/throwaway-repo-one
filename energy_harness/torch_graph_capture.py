#!/usr/bin/env python3
"""CUDA-graph capture of a Python-callable runner's ``launch_once`` (M311).

M310 measured Python submission at 53-87% of the application-energy harness window, the same defect
class B fixed with graph replay (M296/M299).  ``make_graph_replayer`` captures ``batch`` back-to-back
``launch_once`` calls into one ``torch.cuda.CUDAGraph`` and returns a callable that replays it, so a
counted window measures kernels, not Python enqueue.  The same recipe ``diagnose_python_launch_overhead.py``
validated on vector_add / softmax / triton_layernorm.

Fails loudly: a capture error propagates; nothing falls back to the host loop silently.
"""
from __future__ import annotations

from typing import Callable

WARMUP_LAUNCHES = 3


def make_graph_replayer(torch, launch_once: Callable[[], None], batch: int) -> Callable[[], None]:
    if batch < 1:
        raise ValueError(f"graph batch must be >= 1, got {batch}")
    # Warm up on a side stream: capturing an uninitialised kernel (Triton JIT/autotune, lazy
    # allocator state) fails or captures the wrong work.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(WARMUP_LAUNCHES):
            launch_once()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(batch):
            launch_once()
    torch.cuda.synchronize()
    return graph.replay
