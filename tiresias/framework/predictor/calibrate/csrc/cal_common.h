// Common layer of the portable calibration tool: device guard and device facts. Replaces the per-file Blackwell-only literals of micro_v3/*.cu.
// A program refuses unless (1) exactly one device is visible, (2) CUDA_VISIBLE_DEVICES equals the UUID in CAL_EXPECT_UUID (set by run_calibration.py from the
// approved-devices allow-list), (3) the live device UUID equals it, and (4) CAL_BOOKING_REF is non-empty. No other identity is hard-coded: SM count,
// L2 size, shared memory and thread limits come from cudaGetDeviceProperties at run time.
#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void cal_check(cudaError_t e, const char* w) { if (e != cudaSuccess) { fprintf(stderr, "CUDA error at %s: %s\n", w, cudaGetErrorString(e)); exit(2); } }
#define CK(x) cal_check((x), #x)

struct DeviceInfo {
  std::string uuid, name;
  int sm = 0, major = 0, minor = 0, warp = 32, max_threads_sm = 0, max_blocks_sm = 0, max_threads_block = 0, regs_sm = 0, clock_khz = 0, mem_clock_khz = 0, bus_bits = 0;
  size_t l2 = 0, smem_block = 0, smem_optin = 0, smem_sm = 0, smem_reserved_per_block = 0, total_mem = 0;
};

static std::string cal_uuid_string(const cudaUUID_t& id) {
  char live[41]; const unsigned char* u = reinterpret_cast<const unsigned char*>(id.bytes);
  snprintf(live, sizeof live, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
  return live;
}

static DeviceInfo cal_init_device() {
  const char* vis = getenv("CUDA_VISIBLE_DEVICES"); const char* want = getenv("CAL_EXPECT_UUID"); const char* booking = getenv("CAL_BOOKING_REF");
  if (!want || !*want) { fprintf(stderr, "REFUSED: CAL_EXPECT_UUID is not set (run through run_calibration.py)\n"); exit(1); }
  if (!booking || !*booking) { fprintf(stderr, "REFUSED: CAL_BOOKING_REF is not set\n"); exit(1); }
  if (!vis || std::string(vis) != want) { fprintf(stderr, "REFUSED: CUDA_VISIBLE_DEVICES must be exactly %s\n", want); exit(1); }
  int n = 0; CK(cudaGetDeviceCount(&n)); if (n != 1) { fprintf(stderr, "REFUSED: exactly one device required\n"); exit(1); }
  CK(cudaSetDevice(0)); cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  DeviceInfo d; d.uuid = cal_uuid_string(p.uuid); d.name = p.name;
  if (d.uuid != want) { fprintf(stderr, "REFUSED: live device %s is not the approved %s\n", d.uuid.c_str(), want); exit(1); }
  d.sm = p.multiProcessorCount; d.major = p.major; d.minor = p.minor; d.warp = p.warpSize; d.max_threads_sm = p.maxThreadsPerMultiProcessor; d.max_blocks_sm = p.maxBlocksPerMultiProcessor;
  d.max_threads_block = p.maxThreadsPerBlock; d.regs_sm = p.regsPerMultiprocessor; d.l2 = p.l2CacheSize; d.smem_block = p.sharedMemPerBlock; d.smem_optin = p.sharedMemPerBlockOptin;
  d.smem_sm = p.sharedMemPerMultiprocessor; d.smem_reserved_per_block = p.reservedSharedMemPerBlock; d.total_mem = p.totalGlobalMem; d.bus_bits = p.memoryBusWidth;
  int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrClockRate, 0); d.clock_khz = v; v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrMemoryClockRate, 0); d.mem_clock_khz = v;
  if (d.warp != 32) { fprintf(stderr, "REFUSED: warp size %d\n", d.warp); exit(1); }
  return d;
}

static void cal_print_device(const DeviceInfo& d) {
  printf("{\"mode\":\"device_facts\",\"uuid\":\"%s\",\"name\":\"%s\",\"compute_capability\":\"%d.%d\",\"sm_count\":%d,\"l2_bytes\":%zu,\"shared_per_block\":%zu,\"shared_per_block_optin\":%zu,"
         "\"shared_per_sm\":%zu,\"shared_reserved_per_block\":%zu,\"max_threads_per_sm\":%d,\"max_blocks_per_sm\":%d,\"max_threads_per_block\":%d,\"regs_per_sm\":%d,"
         "\"sm_clock_khz_reported\":%d,\"mem_clock_khz_reported\":%d,\"memory_bus_bits\":%d,\"total_memory_bytes\":%zu}\n",
         d.uuid.c_str(), d.name.c_str(), d.major, d.minor, d.sm, d.l2, d.smem_block, d.smem_optin, d.smem_sm, d.smem_reserved_per_block, d.max_threads_sm, d.max_blocks_sm,
         d.max_threads_block, d.regs_sm, d.clock_khz, d.mem_clock_khz, d.bus_bits, d.total_mem);
}
