#!/usr/bin/env python3
"""Background NVML sampler for Task C's application-energy raw traces.

Produces ``samples.csv`` rows in exactly the schema
``energy_harness/verify_b_stabilization_trace.py`` already validates for Task B's fixtures
(``monotonic_ns, board_power_mw, temperature_c, graphics_clock_mhz,
memory_clock_mhz, utilization_percent``), so the same finalizer reconstructs
application-parent energy labels unchanged -- see
``tiresias/planning/B_GRAPH_REPLAY_HANDOFF_2026-09-20.md`` section 6.

Fails loud if libnvidia-ml cannot be loaded, the target GPU index cannot be
opened, or any single NVML call errors -- never silently drops a sample or
produces a plausible-but-fake one (AGENTS.md's broken-tools rule). The thread
records its own exception and re-raises it from ``stop()`` rather than dying
silently.
"""
from __future__ import annotations

import csv
import ctypes
import ctypes.util
import threading
import time
from pathlib import Path

# NVML enum values (nvml.h) -- pinned here because this module intentionally
# has no pynvml/nvidia-ml-py dependency, matching the project's zero-extra-
# dependency convention for these runners.
NVML_TEMPERATURE_GPU = 0
NVML_CLOCK_GRAPHICS = 0
NVML_CLOCK_MEM = 2


class SamplerError(RuntimeError):
    """A required NVML call failed; never silently fall back to a fake sample."""


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class NvmlHandle:
    """Thin ctypes wrapper around libnvidia-ml. Injectable via ``lib=`` for tests
    that must run without a real driver present (this project has no GPU on the
    machine most sessions edit this file from)."""

    def __init__(self, gpu_index: int, lib=None):
        if lib is None:
            path = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
            try:
                lib = ctypes.CDLL(path)
            except OSError as exc:
                raise SamplerError(f"BROKEN: cannot load libnvidia-ml ({path}): {exc}") from exc
        self.lib = lib
        rc = self.lib.nvmlInit_v2()
        if rc != 0:
            raise SamplerError(f"BROKEN: nvmlInit_v2 failed (rc={rc})")
        handle = ctypes.c_void_p()
        rc = self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(gpu_index), ctypes.byref(handle))
        if rc != 0:
            raise SamplerError(f"BROKEN: nvmlDeviceGetHandleByIndex_v2 failed for index {gpu_index} (rc={rc})")
        self.handle = handle

    def sample(self) -> tuple[int, int, int, int, int]:
        """Returns (board_power_mw, temperature_c, graphics_clock_mhz, memory_clock_mhz,
        utilization_percent), the same fields B's fixtures sample every ~5-10ms."""
        power_mw = ctypes.c_uint()
        rc = self.lib.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(power_mw))
        if rc != 0:
            raise SamplerError(f"BROKEN: nvmlDeviceGetPowerUsage failed (rc={rc})")
        temp_c = ctypes.c_uint()
        rc = self.lib.nvmlDeviceGetTemperature(self.handle, ctypes.c_uint(NVML_TEMPERATURE_GPU), ctypes.byref(temp_c))
        if rc != 0:
            raise SamplerError(f"BROKEN: nvmlDeviceGetTemperature failed (rc={rc})")
        gclk = ctypes.c_uint()
        rc = self.lib.nvmlDeviceGetClockInfo(self.handle, ctypes.c_uint(NVML_CLOCK_GRAPHICS), ctypes.byref(gclk))
        if rc != 0:
            raise SamplerError(f"BROKEN: nvmlDeviceGetClockInfo(graphics) failed (rc={rc})")
        mclk = ctypes.c_uint()
        rc = self.lib.nvmlDeviceGetClockInfo(self.handle, ctypes.c_uint(NVML_CLOCK_MEM), ctypes.byref(mclk))
        if rc != 0:
            raise SamplerError(f"BROKEN: nvmlDeviceGetClockInfo(memory) failed (rc={rc})")
        util = _NvmlUtilization()
        rc = self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(util))
        if rc != 0:
            raise SamplerError(f"BROKEN: nvmlDeviceGetUtilizationRates failed (rc={rc})")
        return power_mw.value, temp_c.value, gclk.value, mclk.value, util.gpu


class NvmlSampler:
    """Background thread sampling at a fixed interval into memory, flushed to a
    B-schema-compatible ``samples.csv`` only after ``stop()``."""

    def __init__(self, gpu_index: int, interval_seconds: float = 0.01, handle_factory=NvmlHandle):
        self._gpu_index = gpu_index
        self._interval = interval_seconds
        self._handle_factory = handle_factory
        self._rows: list[tuple[int, int, int, int, int, int]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            handle = self._handle_factory(self._gpu_index)
            while not self._stop_event.is_set():
                power_mw, temp_c, gclk, mclk, util = handle.sample()
                self._rows.append((time.monotonic_ns(), power_mw, temp_c, gclk, mclk, util))
                time.sleep(self._interval)
        except BaseException as exc:  # noqa: BLE001 -- surfaced by stop(), never swallowed
            self._error = exc

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        if self._error is not None:
            raise SamplerError(f"BROKEN: NVML sampler thread failed: {self._error}") from self._error

    def write_csv(self, path: Path) -> None:
        if len(self._rows) < 20:
            raise SamplerError(
                f"BROKEN: only {len(self._rows)} NVML samples captured; refusing a trace too thin "
                "for energy_harness/verify_b_stabilization_trace.py's own minimum (matches B's fixtures)."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["monotonic_ns", "board_power_mw", "temperature_c",
                              "graphics_clock_mhz", "memory_clock_mhz", "utilization_percent"])
            writer.writerows(self._rows)
