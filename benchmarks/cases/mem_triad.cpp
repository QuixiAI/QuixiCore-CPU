// mem_triad: STREAM-triad memory bandwidth probe, a[i] = b[i] + s * c[i].
//
// System-characterization probe, not a contract kernel: it establishes the
// machine's memory roofline that memory-bound kernels (quantized decode
// GEMV/GEMM) are judged against. Single-threaded, f32, baseline arch flags.

#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "harness/alloc.h"
#include "harness/case.h"
#include "harness/donotopt.h"
#include "harness/shapes.h"

namespace qcb {
namespace {

constexpr long long kKiB = 1024;
constexpr long long kMiB = 1024 * 1024;
constexpr float kScale = 3.0f;

std::string ws_name(long long bytes) {
  if (bytes % kMiB == 0) {
    return "ws_" + std::to_string(bytes / kMiB) + "MiB";
  }
  return "ws_" + std::to_string(bytes / kKiB) + "KiB";
}

struct Buffers {
  AlignedBuffer<float> a;
  AlignedBuffer<float> b;
  AlignedBuffer<float> c;
  long long n = 0;
};

CaseDecl make_triad_decl(long long ws_bytes) {
  // Three arrays share the working set; round n down to a 1024 multiple.
  long long n = ws_bytes / (3 * 4);
  n -= n % 1024;

  CaseDecl decl;
  decl.kernel = "mem_triad";
  decl.variant = ws_name(ws_bytes);
  decl.shape = {{"n", n}};  // working set = 3*4*n bytes, named in variant
  decl.dtype = "f32";
  decl.notes = "single-thread system probe; not a contract kernel";
  // STREAM counting convention: read b, read c, write a. Write-allocate
  // (RFO) traffic for a is not counted.
  decl.bytes_moved = 3.0 * 4.0 * static_cast<double>(n);
  decl.make = [n]() {
    auto bufs = std::make_shared<Buffers>();
    bufs->a = aligned_alloc_array<float>(n);
    bufs->b = aligned_alloc_array<float>(n);
    bufs->c = aligned_alloc_array<float>(n);
    bufs->n = n;
    // Deterministic init that writes every element of every array: pages are
    // faulted in (and first-touch placed) before any check or warmup runs.
    float* a = bufs->a.get();
    float* b = bufs->b.get();
    float* c = bufs->c.get();
    for (long long i = 0; i < n; ++i) {
      a[i] = 0.0f;
      b[i] = static_cast<float>(i % 512) * 0.25f;
      c[i] = static_cast<float>(i % 257) * 0.5f;
    }

    CaseBody body;
    body.target = [bufs]() {
      float* a = bufs->a.get();
      const float* b = bufs->b.get();
      const float* c = bufs->c.get();
      const long long n = bufs->n;
      for (long long i = 0; i < n; ++i) {
        a[i] = b[i] + kScale * c[i];
      }
      do_not_optimize(a);
    };
    // memcpy moves 2*n*4 bytes (read b, write a) versus triad's 3*n*4;
    // convert accordingly when comparing bandwidths.
    body.baselines.emplace_back("memcpy", [bufs]() {
      std::memcpy(bufs->a.get(), bufs->b.get(),
                  static_cast<size_t>(bufs->n) * sizeof(float));
      do_not_optimize(bufs->a.get());
    });
    body.check = [bufs]() {
      float* a = bufs->a.get();
      const float* b = bufs->b.get();
      const float* c = bufs->c.get();
      const long long n = bufs->n;
      for (long long i = 0; i < n; ++i) {
        a[i] = b[i] + kScale * c[i];
      }
      do_not_optimize(a);
      double max_abs = 0.0;
      double max_ref = 0.0;
      for (long long i = 0; i < n; ++i) {
        const double ref = static_cast<double>(b[i]) +
                           static_cast<double>(kScale) * c[i];
        max_abs = std::max(max_abs, std::fabs(a[i] - ref));
        max_ref = std::max(max_ref, std::fabs(ref));
      }
      return CheckResult{max_abs, max_abs / (max_ref + 1e-9)};
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_mem_triad_cases(const BuildCtx& ctx, std::vector<CaseDecl>& out) {
  // Working-set ladder crossing typical L1 (64-192 KiB), L2 (1-16 MiB),
  // last-level/SLC, and DRAM on current x86_64 and Apple silicon parts.
  static const std::vector<long long> kSmoke = {96 * kKiB, 24 * kMiB};
  static const std::vector<long long> kQuick = {96 * kKiB, 1536 * kKiB,
                                                24 * kMiB, 192 * kMiB};
  static const std::vector<long long> kComprehensive = {
      96 * kKiB,  384 * kKiB, 1536 * kKiB, 6 * kMiB,
      24 * kMiB,  96 * kMiB,  384 * kMiB,  768 * kMiB};

  for (const long long ws : pick(ctx.preset, kSmoke, kQuick, kComprehensive)) {
    out.push_back(make_triad_decl(ws));
  }
}

}  // namespace qcb
