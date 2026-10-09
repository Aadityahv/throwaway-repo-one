// Shared host-side code of the four unseen-kernel drivers (matrix multiply, Black-Scholes, scan, separable
// convolution). Included BEFORE the pinned sample source in every driver; it defines no main() and no
// __global__ function, so it never clashes with the sample (all names are in namespace unseen / prefixed UD_).
//
// Contract of every driver (positional arguments differ per driver, the trailer is common):
//   <driver> <3 positional args ending in out.bin> [--repeat N --graph-batch B --trace-dir DIR]
//   - no trailer: one launch of the whole operator call on a private non-default stream, sync, then check.
//   - trailer:    capture a CUDA graph of B operator calls, one discarded warm-up replay, then time `repeat`
//                 operator calls (repeat/B full replays plus repeat%B direct operator calls) between two
//                 cudaEvents; write <DIR>/windows.csv in the format of tiresias/app_runners/copy_runner.py.
//   - after the timed loop and a final sync (never inside the timed region) the output is copied back and
//     compared with a CPU reference; the one-line verdict goes to <out.bin>.check ("CHECK_OK ..." or
//     "CHECK_FAIL ..."), and the process exits 0 on success, 1 on a failed check, 2 on any fatal error.
//
// Safety: the driver uses CUDA device 0 only (it never calls cudaSetDevice with another index), refuses unless
// exactly one device is visible AND that device's UUID equals the environment variable UNSEEN_EXPECT_UUID
// (set by run_unseen.py to the approved Blackwell GPU 1). It sets no clocks, no persistence mode, no power
// limit, and never links NVML.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#define UD_CUDA_CHECK(call)                                                                      \
    do {                                                                                         \
        cudaError_t ud_status_ = (call);                                                         \
        if (ud_status_ != cudaSuccess) {                                                         \
            std::fprintf(stderr, "FATAL CUDA %s:%d: %s\n", __FILE__, __LINE__,                   \
                         cudaGetErrorString(ud_status_));                                        \
            std::exit(2);                                                                        \
        }                                                                                        \
    } while (0)

namespace unseen {

constexpr uint64_t kSeed = 20260914ULL;              // same project seed as the copy driver / runners
constexpr size_t kMaxOutputFileBytes = 64ULL << 20;  // output array is written to out.bin only up to 64 MiB

[[noreturn]] inline void fatal(const char *msg) {
    std::fprintf(stderr, "FATAL: %s\n", msg);
    std::exit(2);
}

inline long long monotonic_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// splitmix64: a fixed, specified generator (platform independent). Each driver seeds one stream per input
// buffer as kSeed + buffer_id.
struct SplitMix {
    uint64_t s;
    explicit SplitMix(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    float uniform01() { return (float)(next() >> 40) * (1.0f / 16777216.0f); }  // 24 bits -> [0,1)
    float uniform(float lo, float hi) { return lo + (hi - lo) * uniform01(); }
    unsigned small_uint() { return (unsigned)((next() >> 32) % 16ULL); }          // value % 16
};

inline long long parse_ll(const char *s, const char *what) {
    char *end = nullptr;
    long long v = std::strtoll(s, &end, 10);
    if (end == s || *end != '\0') {
        std::fprintf(stderr, "FATAL: argument %s is not an integer: '%s'\n", what, s);
        std::exit(2);
    }
    return v;
}

// Hex digits of a UUID string, lower case, without "GPU-" prefix and dashes.
inline std::string normalise_uuid(const char *s) {
    std::string out;
    for (const char *p = s; *p; ++p)
        if (std::isxdigit((unsigned char)*p)) out.push_back((char)std::tolower((unsigned char)*p));
    return out;
}

// Exactly one visible device, and it must be the approved one. Uses device 0 (the only visible device).
inline void guard_single_approved_device() {
    int count = 0;
    UD_CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count != 1) {
        std::fprintf(stderr,
                     "FATAL: exactly one visible CUDA device is required, found %d "
                     "(set CUDA_VISIBLE_DEVICES to the single approved GPU)\n",
                     count);
        std::exit(2);
    }
    cudaDeviceProp prop;
    UD_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    char hex[40];
    for (int i = 0; i < 16; ++i) std::snprintf(hex + 2 * i, 3, "%02x", (unsigned char)prop.uuid.bytes[i]);
    const char *expect = std::getenv("UNSEEN_EXPECT_UUID");
    if (!expect || !*expect) fatal("UNSEEN_EXPECT_UUID is not set; refusing to run on an unverified device");
    if (normalise_uuid(expect) != std::string(hex)) {
        std::fprintf(stderr, "FATAL: visible device uuid %s does not match UNSEEN_EXPECT_UUID=%s\n", hex, expect);
        std::exit(2);
    }
    std::printf("UNSEEN_DEVICE name=%s uuid=%s\n", prop.name, hex);
}

struct Trailer {
    bool timed = false;
    long long repeat = 0;
    long long graph_batch = 0;
    const char *trace_dir = nullptr;
};

// `first` is the index of the first argument after the positionals; the trailer is either absent or exactly
// "--repeat N --graph-batch B --trace-dir DIR" (the form used by energy_harness/application_energy_harness.py).
inline Trailer parse_trailer(int argc, char **argv, int first) {
    Trailer t;
    const int extra = argc - first;
    if (extra == 0) return t;
    if (extra != 6) fatal("expected either no trailer or exactly: --repeat N --graph-batch B --trace-dir DIR");
    if (std::strcmp(argv[first], "--repeat") != 0 || std::strcmp(argv[first + 2], "--graph-batch") != 0 ||
        std::strcmp(argv[first + 4], "--trace-dir") != 0)
        fatal("expected --repeat N --graph-batch B --trace-dir DIR");
    t.timed = true;
    t.repeat = parse_ll(argv[first + 1], "--repeat");
    t.graph_batch = parse_ll(argv[first + 3], "--graph-batch");
    t.trace_dir = argv[first + 5];
    if (t.repeat <= 0 || t.graph_batch <= 0) fatal("bad repeat/graph-batch");
    return t;
}

inline std::string sidecar_path(const char *out_path) { return std::string(out_path) + ".check"; }

// A stale verdict or output from an earlier invocation must never be mistaken for this run's.
inline void remove_stale_outputs(const char *out_path) {
    std::remove(out_path);
    std::remove(sidecar_path(out_path).c_str());
}

// Writes the output array only when it is small; the .check file is the contract.
inline void maybe_write_output(const char *out_path, const void *data, size_t bytes) {
    if (bytes > kMaxOutputFileBytes) return;
    FILE *fo = std::fopen(out_path, "wb");
    if (!fo) fatal("cannot open output file");
    if (std::fwrite(data, 1, bytes, fo) != bytes) fatal("short output write");
    std::fclose(fo);
}

inline int finish(const char *out_path, bool ok, const std::string &msg) {
    const std::string path = sidecar_path(out_path);
    FILE *fc = std::fopen(path.c_str(), "w");
    if (!fc) fatal("cannot open .check file");
    std::fprintf(fc, "%s %s\n", ok ? "CHECK_OK" : "CHECK_FAIL", msg.c_str());
    std::fclose(fc);
    if (ok) {
        std::printf("CHECK_OK %s\n", msg.c_str());
        return 0;
    }
    std::fprintf(stderr, "CHECK_FAIL %s\n", msg.c_str());
    return 1;
}

inline std::string fmt(const char *format, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, format);
    std::vsnprintf(buf, sizeof(buf), format, ap);
    va_end(ap);
    return std::string(buf);
}

inline long long ceil_div_ll(long long a, long long b) { return (a + b - 1) / b; }

// launch_all(stream) must enqueue ALL kernels of ONE operator call, in order, on `stream`, with no allocation
// and no synchronisation (it is called inside stream capture).
template <class LaunchAll>
inline void execute(cudaStream_t stream, const Trailer &t, LaunchAll launch_all) {
    if (!t.timed) {
        launch_all(stream);
        UD_CUDA_CHECK(cudaGetLastError());
        UD_CUDA_CHECK(cudaStreamSynchronize(stream));
        return;
    }
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    UD_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (long long i = 0; i < t.graph_batch; ++i) launch_all(stream);
    UD_CUDA_CHECK(cudaGetLastError());
    UD_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    // cudaGraphInstantiateWithFlags exists with the same signature from CUDA 11.4 to 13; the copy driver's
    // five-argument cudaGraphInstantiate form is avoided on purpose.
    UD_CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec, graph, 0));
    UD_CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));  // warm-up replay, discarded
    UD_CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t ev_begin, ev_end;
    UD_CUDA_CHECK(cudaEventCreate(&ev_begin));
    UD_CUDA_CHECK(cudaEventCreate(&ev_end));
    const long long full_replays = t.repeat / t.graph_batch;
    const long long remainder = t.repeat % t.graph_batch;
    const long long host_begin_ns = monotonic_ns();
    UD_CUDA_CHECK(cudaEventRecord(ev_begin, stream));
    for (long long r = 0; r < full_replays; ++r) UD_CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    for (long long i = 0; i < remainder; ++i) launch_all(stream);
    UD_CUDA_CHECK(cudaEventRecord(ev_end, stream));
    UD_CUDA_CHECK(cudaEventSynchronize(ev_end));
    const long long host_end_ns = monotonic_ns();
    float elapsed_ms = 0.0f;
    UD_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_begin, ev_end));

    char windows_path[4096];
    std::snprintf(windows_path, sizeof(windows_path), "%s/windows.csv", t.trace_dir);
    FILE *fw = std::fopen(windows_path, "w");
    if (!fw) fatal("cannot open windows.csv");
    std::fprintf(fw, "block,launches,host_begin_monotonic_ns,host_end_monotonic_ns,cuda_seconds\n");
    std::fprintf(fw, "1,%lld,%lld,%lld,%.9f\n", t.repeat, host_begin_ns, host_end_ns, (double)elapsed_ms / 1000.0);
    std::fclose(fw);

    UD_CUDA_CHECK(cudaGetLastError());
    UD_CUDA_CHECK(cudaStreamSynchronize(stream));
    UD_CUDA_CHECK(cudaEventDestroy(ev_begin));
    UD_CUDA_CHECK(cudaEventDestroy(ev_end));
    UD_CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    UD_CUDA_CHECK(cudaGraphDestroy(graph));
}

}  // namespace unseen
