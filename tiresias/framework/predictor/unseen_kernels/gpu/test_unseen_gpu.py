#!/usr/bin/env python3
"""CPU-only tests of the unseen-kernel GPU tooling. Run with `python3 test_unseen_gpu.py` or `pytest`.
No GPU, no nvcc, no nvidia-smi and no network are used: every subprocess call that could reach one is replaced
by a function that fails the test if it is called.
"""
from __future__ import annotations

import contextlib
import io
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_unseen as ru  # noqa: E402

DRIVER_FILES = {"matmul": "driver_matmul.cu", "bs": "driver_bs.cu", "scan": "driver_scan.cu", "conv": "driver_conv.cu"}
OTHER_UUID = "GPU-11111111-666f-f6cc-aa5b-3c07feaa3a95"


class Forbidden:
    """Stand-in for subprocess.run that fails the test if anything tries to execute a command."""

    def __init__(self):
        self.calls = []

    def __call__(self, *a, **k):
        self.calls.append(a)
        raise AssertionError("a subprocess was started: %r" % (a,))


@contextlib.contextmanager
def patched(forbid_smi=True, forbid_subprocess=True, smi=None):
    saved = (ru.SMI_RUN, subprocess.run)
    f = Forbidden()
    ru.SMI_RUN = smi if smi is not None else (f if forbid_smi else saved[0])
    if forbid_subprocess:
        subprocess.run = f
    try:
        yield f
    finally:
        ru.SMI_RUN, subprocess.run = saved


def fake_smi(gpus, apps="", gpu_rc=0, apps_rc=0):
    """gpus: list of (index, uuid). apps: text of --query-compute-apps."""
    def run(argv, **kwargs):
        assert argv[0] == "nvidia-smi"
        joined = " ".join(argv)
        for bad in ("-pm", "-lgc", "-rgc", "-pl", "-ac", "-lmc", "--persistence-mode", "--power-limit", "--applications-clocks"):
            assert bad not in argv, "a setting flag was passed to nvidia-smi: %s" % bad
        if "--query-gpu=index,uuid" in joined:
            return types.SimpleNamespace(returncode=gpu_rc, stdout="".join("%s, %s\n" % g for g in gpus), stderr="")
        if "--query-compute-apps" in joined:
            return types.SimpleNamespace(returncode=apps_rc, stdout=apps, stderr="")
        raise AssertionError("unexpected nvidia-smi query: %s" % joined)
    return run


def run_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = ru.main(argv)
        except SystemExit as exc:
            rc = exc.code
    return rc, out.getvalue(), err.getvalue()


def expect_refused(argv, env=None, smi=None, needle="REFUSED"):
    old = dict(ru.os.environ)
    try:
        for k in ("CUDA_VISIBLE_DEVICES",):
            ru.os.environ.pop(k, None)
        ru.os.environ.update(env or {})
        with patched(smi=smi, forbid_smi=smi is None) as forb:
            rc, out, err = run_main(argv)
    finally:
        ru.os.environ.clear()
        ru.os.environ.update(old)
    assert isinstance(rc, str) and needle in rc, "expected a REFUSED exit, got rc=%r out=%r err=%r" % (rc, out[-200:], err[-200:])
    return rc


# ---------------------------------------------------------------------------------------------------------------
# cells and argv
# ---------------------------------------------------------------------------------------------------------------
def expected_kernels_from_argv(family, pos):
    """Independent re-derivation from the facts of the brief (the sample host code), written separately from
    run_unseen.launch_geometry."""
    a, b = int(pos[0]), int(pos[1])
    if family == "matmul":
        return [([a // b, a // b, 1], [b, b, 1])]
    if family == "bs":
        return [([(a // 2) // b, 1, 1], [b, 1, 1])]
    if family == "scan":
        nb = a // (4 * 256)
        return [([nb, 1, 1], [256, 1, 1]), ([(nb + 255) // 256, 1, 1], [256, 1, 1]), ([nb, 1, 1], [256, 1, 1])]
    if family == "conv":
        return [([a // 128, b // 4, 1], [16, 4, 1]), ([a // 16, b // 64, 1], [16, 8, 1])]
    raise AssertionError(family)


def test_32_cells_and_unique_ids():
    cells = ru.load_cells()
    assert len(cells) == 32
    assert len({c["cell_id"] for c in cells}) == 32
    assert {c["family"] for c in cells} == set(DRIVER_FILES)
    assert len({ru.safe_id(c["cell_id"]) for c in cells}) == 32


def test_argv_geometry_matches_cells_for_all_32_cells():
    for c in ru.load_cells():
        pos = ru.positional_args(c)
        assert len(pos) == 2 and all(re.fullmatch(r"\d+", p) for p in pos), pos
        want = [(list(k["grid"]), list(k["block"])) for k in c["kernels"]]
        assert expected_kernels_from_argv(c["family"], pos) == want, (c["cell_id"], pos)
        assert [(g["grid"], g["block"]) for g in ru.launch_geometry(c["family"], pos)] == want, c["cell_id"]
        assert ru.cell_argv_check(c) == pos


def test_known_argv_values():
    by = {c["cell_id"]: c for c in ru.load_cells()}
    p = lambda op, reg, cand: ru.positional_args(by["blackwell/unseen_cuda_samples_%s/%s/%s" % (op, reg, cand)])
    assert p("matrix_multiply", "small", "c1") == ["352", "16"]
    assert p("matrix_multiply", "xlarge", "c2") == ["4256", "32"]
    assert p("black_scholes", "xlarge", "c2") == ["12582912", "64"]
    assert p("scan", "large", "c1") == [str(3 << 21), "4096"]
    assert p("scan", "xlarge", "c2") == [str(1 << 25), str(1 << 18)]
    assert p("separable_convolution", "medium", "c2") == ["1024", "1280"]  # candidate 2 swaps width and height


def test_geometry_mismatch_is_fatal():
    c = dict(ru.load_cells()[0])
    c["kernels"] = [dict(c["kernels"][0], grid=[1, 1, 1])]
    try:
        ru.cell_argv_check(c)
    except SystemExit as exc:
        assert "differs from cells.py" in str(exc)
    else:
        raise AssertionError("a geometry mismatch must be fatal")


# ---------------------------------------------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------------------------------------------
def test_refuses_without_booking_flag_before_anything_runs():
    for cmd in ("time", "energy"):
        with tempfile.TemporaryDirectory() as d:
            msg = expect_refused([cmd, "--workdir", d, "--only", "small/c1"], env={"CUDA_VISIBLE_DEVICES": "1"})
            assert "--i-have-a-booking" in msg


def test_refuses_without_approved_device():
    good = [("0", OTHER_UUID), ("1", ru.APPROVED_UUID)]
    base = ["time", "--i-have-a-booking", "--workdir", "/nonexistent_unseen_workdir"]
    # no CUDA_VISIBLE_DEVICES
    expect_refused(base, env={}, smi=fake_smi(good))
    # several devices visible
    expect_refused(base, env={"CUDA_VISIBLE_DEVICES": "0,1"}, smi=fake_smi(good))
    # the other GPU (by index and by UUID)
    expect_refused(base, env={"CUDA_VISIBLE_DEVICES": "0"}, smi=fake_smi(good))
    expect_refused(base, env={"CUDA_VISIBLE_DEVICES": OTHER_UUID}, smi=fake_smi(good))
    # approved GPU absent from the machine
    expect_refused(base, env={"CUDA_VISIBLE_DEVICES": "1"}, smi=fake_smi([("0", OTHER_UUID), ("1", OTHER_UUID)]))
    # nvidia-smi missing (a Mac) or failing: cannot verify -> refuse
    def missing(argv, **kw):
        raise FileNotFoundError("nvidia-smi")
    expect_refused(base, env={"CUDA_VISIBLE_DEVICES": "1"}, smi=missing)
    expect_refused(base, env={"CUDA_VISIBLE_DEVICES": "1"}, smi=fake_smi(good, gpu_rc=9))
    # approved GPU busy
    busy = "%s, 4242, python\n" % ru.APPROVED_UUID
    expect_refused(base, env={"CUDA_VISIBLE_DEVICES": "1"}, smi=fake_smi(good, apps=busy))
    expect_refused(base, env={"CUDA_VISIBLE_DEVICES": "1"}, smi=fake_smi(good, apps_rc=9))
    # energy needs CUDA_VISIBLE_DEVICES == "1" even when the UUID form names the right GPU
    expect_refused(["energy", "--i-have-a-booking", "--workdir", "/nonexistent_unseen_workdir"],
                   env={"CUDA_VISIBLE_DEVICES": ru.APPROVED_UUID}, smi=fake_smi(good))


def test_approved_idle_device_is_accepted_by_the_guard():
    good = [("0", OTHER_UUID), ("1", ru.APPROVED_UUID)]
    other_busy = "%s, 77, python\n" % OTHER_UUID  # another GPU busy is not our business
    with patched(smi=fake_smi(good, apps=other_busy)):
        info = ru.require_approved_device("time", env={"CUDA_VISIBLE_DEVICES": "1"})
        assert info["uuid"] == ru.APPROVED_UUID
        info = ru.require_approved_device("time", env={"CUDA_VISIBLE_DEVICES": ru.APPROVED_UUID.replace("GPU-", "")})
        assert info["nvidia_smi_index"] == "1"
        ru.require_approved_device("energy", env={"CUDA_VISIBLE_DEVICES": "1"})


def test_dry_run_and_list_run_nothing():
    for argv in (["time", "--dry-run"], ["energy", "--dry-run"], ["list"], ["time", "--dry-run", "--only", "scan"]):
        with patched() as forb:  # any nvidia-smi or subprocess call raises
            rc, out, err = run_main(argv)
        assert rc == 0, (argv, rc, err)
        assert not forb.calls
        n_cells = len([ln for ln in out.splitlines() if ln.startswith("blackwell/")])
        assert n_cells == (8 if "scan" in argv else 32), (argv, n_cells)
        assert "unseen_scan" in out or "scan" not in argv


def test_build_command_has_no_optimisation_flag_for_device_code():
    root = Path("/x/cuda-samples")
    cmd = ru.build_command("/usr/local/cuda-13.2/bin/nvcc", root, "scan", Path("/w/bin/unseen_scan"), host_opt=True)
    assert "-arch=sm_120" in cmd
    i = cmd.index("-Xcompiler")
    assert cmd[i + 1] == "-O2"  # host only
    rest = cmd[:i] + cmd[i + 2:]
    assert not any(a in rest for a in ("-O", "-O1", "-O2", "-O3", "-G", "-g", "-lineinfo", "--dlink-time-opt"))
    assert str(root / "Common") in cmd and str(root / "cpp/2_Concepts_and_Techniques/scan") in cmd
    assert str(ru.DRIVERS / "driver_scan.cu") in cmd and cmd[-2:] == ["-o", "/w/bin/unseen_scan"]
    cmd0 = ru.build_command("nvcc", root, "bs", Path("/w/b"), host_opt=False)
    assert "-Xcompiler" not in cmd0
    for fam in ru.FAMILIES:
        assert (ru.DRIVERS / ru.FAMILIES[fam]["driver"]).is_file()


# ---------------------------------------------------------------------------------------------------------------
# parsing and selection helpers
# ---------------------------------------------------------------------------------------------------------------
def write(path, text):
    Path(path).write_text(text)
    return Path(path)


def test_parse_windows_csv():
    with tempfile.TemporaryDirectory() as d:
        hdr = "block,launches,host_begin_monotonic_ns,host_end_monotonic_ns,cuda_seconds\n"
        w = ru.parse_windows_csv(write(Path(d) / "a.csv", hdr + "1,4000,100,900,0.012000000\n"))
        assert w["launches"] == 4000 and abs(w["per_launch_s"] - 3e-6) < 1e-15 and w["host_end_monotonic_ns"] == 900
        for bad in (hdr + "", hdr + "1,4000,100,900,0.01\n1,1,1,1,1\n", "x,y\n1,2\n", hdr + "1,0,1,2,0.1\n",
                    hdr + "1,10,1,2,abc\n", hdr + "1,10,1,2,-0.5\n", hdr + "1,10,1,2\n", hdr + "1,10,1,2,nan\n"):
            try:
                ru.parse_windows_csv(write(Path(d) / "b.csv", bad))
            except ValueError:
                continue
            raise AssertionError("accepted a bad windows.csv: %r" % bad)
        try:
            ru.parse_windows_csv(Path(d) / "missing.csv")
        except ValueError:
            pass
        else:
            raise AssertionError("missing windows.csv accepted")


def test_windows_header_matches_driver_source():
    text = (ru.DRIVERS / "driver_common.h").read_text()
    assert ",".join(ru.WINDOWS_HEADER) in text
    assert '"1,%lld,%lld,%lld,%.9f\\n"' in text  # same row format as copy_runner.py's driver


def test_read_check():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "out.bin"
        assert ru.read_check(out) == (False, "CHECK_MISSING")
        write(str(out) + ".check", "CHECK_OK matmul N=352\n")
        assert ru.read_check(out) == (True, "CHECK_OK matmul N=352")
        write(str(out) + ".check", "CHECK_FAIL scan mismatches=3\n")
        assert ru.read_check(out)[0] is False


def test_batch_and_repeat_selection():
    for t in (1e-6, 3e-6, 20e-6, 1e-4, 5e-4, 2e-3, 0.02):
        b = ru.choose_graph_batch(t)
        assert 1 <= b <= ru.MAX_GRAPH_BATCH
        assert b * t >= ru.TARGET_GRAPH_S or b == ru.MAX_GRAPH_BATCH
        r = ru.choose_repeat(t, b)
        assert r % b == 0 and r // b >= 1 and r * t >= ru.TARGET_WINDOW_S * 0.999
    assert ru.choose_graph_batch(0.5) == 1
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        try:
            ru.choose_graph_batch(bad)
        except ValueError:
            continue
        raise AssertionError("accepted per-call time %r" % bad)


# ---------------------------------------------------------------------------------------------------------------
# static checks of the C++ sources (no compiler available here)
# ---------------------------------------------------------------------------------------------------------------
def sources():
    return {name: (ru.DRIVERS / name).read_text() for name in list(DRIVER_FILES.values()) + ["driver_common.h"]}


def test_drivers_include_pinned_sample_unmodified_via_main_rename():
    pat = {"matmul": "matrixMul.cu", "bs": "BlackScholes_kernel.cuh", "scan": "scan.cu", "conv": "convolutionSeparable.cu"}
    for fam, fname in DRIVER_FILES.items():
        t = (ru.DRIVERS / fname).read_text()
        i_def = t.index("#define main SAMPLE_main_disabled")
        i_inc = t.index('#include "%s"' % pat[fam])
        i_und = t.index("#undef main")
        assert i_def < i_inc < i_und, fname
        assert t.index('#include "driver_common.h"') < i_def  # our headers come before the rename
        assert len(re.findall(r"^int main\(", t, re.M)) == 1
        assert t.index("int main(") > i_und
        assert t.count("#include \"") == 2  # only the common header and the pinned sample


def test_drivers_use_device_zero_only_and_never_touch_clocks_or_persistence():
    forbidden = ["nvml", "nvidia-smi", "persistence", "cudaDeviceSetLimit", "cudaDeviceReset", "system(", "popen(", "lgc",
                 "power_limit", "powerlimit", "cudaSetValidDevices", "cudaChooseDevice", "setenv", "putenv", "fork("]
    for name, text in sources().items():
        text = re.sub(r"//[^\n]*", "", text)  # comments may name what the code never does
        low = text.lower()
        for token in forbidden:
            assert token.lower() not in low, "%s contains %r" % (name, token)
        for m in re.finditer(r"cudaSetDevice\s*\(([^)]*)\)", text):
            assert m.group(1).strip() == "0", "%s: cudaSetDevice(%s)" % (name, m.group(1))
    common = sources()["driver_common.h"]
    assert "cudaGetDeviceCount" in common and "UNSEEN_EXPECT_UUID" in common
    assert "cudaStreamCaptureModeGlobal" in common and "cudaGraphLaunch" in common


def test_drivers_launch_on_private_stream_with_sample_geometry():
    t = sources()
    assert "grid((unsigned)(N / tile), (unsigned)(N / tile))" in t["driver_matmul.cu"]
    assert "threads((unsigned)tile, (unsigned)tile)" in t["driver_matmul.cu"]
    assert "MatrixMulCUDA<16><<<grid, threads, 0, s>>>" in t["driver_matmul.cu"]
    assert "MatrixMulCUDA<32><<<grid, threads, 0, s>>>" in t["driver_matmul.cu"]
    assert "BlackScholesGPU<<<blocks, (unsigned)block, 0, s>>>" in t["driver_bs.cu"]
    assert "(optN / 2) / block" in t["driver_bs.cu"] and "0.02f" in t["driver_bs.cu"] and "0.30f" in t["driver_bs.cu"]
    sc = t["driver_scan.cu"]
    assert "scanExclusiveShared<<<nb," in sc and "scanExclusiveShared2<<<top_blocks," in sc and "uniformUpdate<<<nb," in sc
    assert "scanExclusiveLarge" not in re.sub(r"//.*", "", sc)  # only mentioned in comments, never called
    assert "L / (4 * kTB)" in sc and "N / (4 * kTB)" in sc
    cv = t["driver_conv.cu"]
    assert "rows_blocks((unsigned)(W / 128), (unsigned)(H / 4))" in cv and "cols_blocks((unsigned)(W / 16), (unsigned)(H / 64))" in cv
    assert "rows_threads(16, 4)" in cv and "cols_threads(16, 8)" in cv
    assert cv.index("setConvolutionKernel(hKernel)") < cv.index("execute(stream")
    for name in DRIVER_FILES.values():
        body = t[name]
        assert "execute(stream, tr, launch_all)" in body
        assert body.index("execute(stream, tr, launch_all)") < body.index("cudaMemcpy(h", body.index("execute(stream"))  # check after the timed loop
        assert "finish(out_path" in body and ".check" in t["driver_common.h"]


def test_common_cli_matches_harness_trailer():
    c = sources()["driver_common.h"]
    for flag in ("--repeat", "--graph-batch", "--trace-dir"):
        assert '"%s"' % flag in c
    argv_tail = ru.trailer(7, 3, "/d")
    assert argv_tail == ["--repeat", "7", "--graph-batch", "3", "--trace-dir", "/d"]
    # the harness appends exactly this trailer after argv_prefix (energy_harness/application_energy_harness._run_binary)
    h = (ru.REPO / "energy_harness" / "application_energy_harness.py").read_text()
    assert '"--repeat", str(repeat), "--graph-batch", str(graph_batch), "--trace-dir", str(trace_dir)' in h


# ---------------------------------------------------------------------------------------------------------------
# orchestration logic with the GPU, the drivers and the harness replaced by fakes
# ---------------------------------------------------------------------------------------------------------------
def fake_workdir(d):
    import json
    d = Path(d)
    (d / "bin").mkdir()
    fams = {}
    for fam in ru.FAMILIES:
        b = d / "bin" / ("unseen_%s" % fam)
        b.write_bytes(b"fake binary " + fam.encode())
        fams[fam] = dict(driver=str(ru.DRIVERS / ru.FAMILIES[fam]["driver"]), driver_sha256="d" * 64,
                         sample_files={"cpp/x/%s" % ru.FAMILIES[fam]["sample_file"]: "s" * 64}, command=["nvcc"], returncode=0,
                         binary=str(b), binary_sha256=ru.sha256_file(b))
    (d / "manifest.json").write_text(json.dumps(dict(schema=ru.MANIFEST_SCHEMA, samples_revision=ru.PINNED_REVISION,
                                                     nvcc=dict(path="nvcc", version="nvcc\nBuild fake"), arch="sm_120",
                                                     host_opt=True, families=fams)))
    return d


def test_time_one_cell_calibration_and_median_with_fake_driver():
    cell = next(c for c in ru.load_cells() if c["family"] == "scan" and c["regime"] == "small")
    calls = []

    def fake_run_driver(binary, pos, out_path, repeat, graph_batch, trace_dir, env, timeout_s):
        calls.append((repeat, graph_batch))
        t_call = 4e-6 if graph_batch == ru.PROBE_BATCH else 3e-6  # the second probe sees less launch overhead
        win = dict(block=1, launches=repeat, host_begin_monotonic_ns=1, host_end_monotonic_ns=2,
                   cuda_seconds=repeat * t_call, per_launch_s=t_call)
        Path(str(out_path) + ".check").write_text("CHECK_OK fake\n")
        return dict(argv=[], rc=0, stdout="", stderr="", windows=win, check_ok=True, check_line="CHECK_OK fake", error=None)

    with tempfile.TemporaryDirectory() as d:
        wd = fake_workdir(d)
        manifest = ru.load_manifest(wd)
        saved = ru.run_driver
        ru.run_driver = fake_run_driver
        try:
            rec = ru.time_one_cell(cell, manifest, wd, {}, 3, 10.0)
        finally:
            ru.run_driver = saved
    assert rec["correct"] and len(rec["windows"]) == 3
    assert calls[0] == (16, 16)
    b1 = ru.choose_graph_batch(4e-6)
    assert calls[1] == (b1, b1)
    batch = ru.choose_graph_batch(3e-6)
    assert rec["graph_batch"] == batch and rec["repeat"] == ru.choose_repeat(3e-6, batch) and rec["repeat"] % batch == 0
    assert all(c == (rec["repeat"], batch) for c in calls[2:]) and len(calls) == 5
    assert abs(rec["per_launch_runtime_s"] - 3e-6) < 1e-15 and rec["launches"] == rec["repeat"]
    assert rec["check"].startswith("CHECK_OK")


def test_time_one_cell_records_failed_check_without_retry():
    cell = ru.load_cells()[0]
    calls = []

    def fake_run_driver(binary, pos, out_path, repeat, graph_batch, trace_dir, env, timeout_s):
        calls.append(repeat)
        win = dict(block=1, launches=repeat, host_begin_monotonic_ns=1, host_end_monotonic_ns=2, cuda_seconds=repeat * 1e-5, per_launch_s=1e-5)
        return dict(argv=[], rc=1, stdout="", stderr="CHECK_FAIL x", windows=win, check_ok=False, check_line="CHECK_FAIL x", error=None)

    with tempfile.TemporaryDirectory() as d:
        wd = fake_workdir(d)
        saved = ru.run_driver
        ru.run_driver = fake_run_driver
        try:
            rec = ru.time_one_cell(cell, ru.load_manifest(wd), wd, {}, 3, 10.0)
        finally:
            ru.run_driver = saved
    assert rec["correct"] is False and calls == [16] and "CHECK_FAIL" in rec["check"]  # stopped after the first probe


def test_energy_flow_with_fake_harness():
    """Context fields, per-cell unique run ids, preflight-before-run order, harness called directly (not its CLI)."""
    events = []

    class FakeError(Exception):
        pass

    class Ctx:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    fake = types.ModuleType("application_energy_harness")
    fake.RunnerError = FakeError
    fake.iso_now = lambda: "now"
    fake.BinaryEnergyContext = Ctx
    fake.preflight_before_context = lambda gpu_index, platform: (events.append(("preflight", gpu_index, platform)) or ru.APPROVED_UUID.replace("GPU-", "GPU-"))
    seen = []

    def run_binary_energy(ctx, **kw):
        events.append(("run", kw["run_id"]))
        seen.append((ctx, kw))
        Path(str(ctx.argv_prefix[-1]) + ".check").write_text("CHECK_OK fake\n")
        assert ctx.check() is True
        return dict(status="raw", row=dict(launch_count=1, board_energy_j_per_launch=1.0, trace_dir="t"))

    fake.run_binary_energy = run_binary_energy
    fake.append_raw_or_rejected = lambda out_dir, result: events.append(("append", str(out_dir), result["status"]))
    saved_mod = sys.modules.get("application_energy_harness")
    sys.modules["application_energy_harness"] = fake
    saved_guard, saved_booking = ru.require_approved_device, ru.require_booking
    ru.require_approved_device = lambda mode, env=None: dict(uuid=ru.APPROVED_UUID)
    try:
        with tempfile.TemporaryDirectory() as d:
            wd = fake_workdir(d)
            rc, out, err = run_main(["energy", "--i-have-a-booking", "--workdir", str(wd), "--only", "/small/", "--graph-batch", "64",
                                     "--window-target-seconds", "15", "--session", "2"])
            out_dir = wd / "energy_raw"
            assert (out_dir / "unseen_energy_run_log.jsonl").is_file()
    finally:
        ru.require_approved_device, ru.require_booking = saved_guard, saved_booking
        if saved_mod is None:
            sys.modules.pop("application_energy_harness", None)
        else:
            sys.modules["application_energy_harness"] = saved_mod
    assert rc == 0, (rc, out, err)
    assert len(seen) == 8  # small regime: 4 families x 2 candidates
    run_ids = [kw["run_id"] for _, kw in seen]
    assert len(set(run_ids)) == 8  # raw_attempts/<run_id>/<runner>-s<session>-aN must not collide across cells
    order = [e[0] for e in events]
    assert order[:3] == ["preflight", "run", "append"]
    for ctx, kw in seen:
        assert ctx.parent_id.startswith("unseen_cuda_samples_") and ctx.regime == "small" and ctx.candidate_id in ("c1", "c2")
        assert ctx.argv_prefix[-1].endswith("out.bin") and len(ctx.argv_prefix) == 3
        assert kw["gpu_index"] == 1 and kw["platform"] == "blackwell" and kw["graph_batch"] == 64 and kw["session"] == 2
        assert kw["window_target_seconds"] == 15.0 and kw["runner_name"].startswith("unseen_")
        assert ctx.source_revision == ru.PINNED_REVISION and Path(ctx.binary).name.startswith("unseen_")


def run_all():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("FAIL %s: %s: %s" % (name, type(exc).__name__, exc))
        else:
            print("ok   %s" % name)
    print("%d tests, %d failed" % (len(tests), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run_all())
