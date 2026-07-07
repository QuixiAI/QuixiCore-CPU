// sgemv_naive: y = W x, row-major W (n x k), f32, plain loops with an f32
// accumulator, no intrinsics, no pragmas.
//
// System-characterization probe, not a contract kernel. This is the scalar
// reference semantics future quantized/ISA GEMV variants must beat; whatever
// auto-vectorization the baseline Release flags produce is part of the
// definition ("naive at Release flags").
//
// Shape provenance: n/k grids from the umbrella quant_matmul family taken at
// m = 1 (decode); 2048 from the decode_small hidden sizes.

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "harness/alloc.h"
#include "harness/case.h"
#include "harness/donotopt.h"
#include "harness/shapes.h"

namespace qcb {
namespace {

struct Buffers {
  AlignedBuffer<float> w;
  AlignedBuffer<float> x;
  AlignedBuffer<float> y;
  long long n = 0;
  long long k = 0;
};

// Deterministic xorshift32 mapped to [-0.5, 0.5): avoids subnormals and
// keeps cancellation mild so the f64 oracle comparison stays meaningful.
class Rng {
 public:
  float next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 17;
    state_ ^= state_ << 5;
    return static_cast<float>(state_ >> 8) * (1.0f / 16777216.0f) - 0.5f;
  }

 private:
  uint32_t state_ = 0x12345678u;
};

void run_sgemv(const Buffers& bufs) {
  const float* w = bufs.w.get();
  const float* x = bufs.x.get();
  float* y = bufs.y.get();
  const long long n = bufs.n;
  const long long k = bufs.k;
  for (long long i = 0; i < n; ++i) {
    const float* row = w + i * k;
    float acc = 0.0f;
    for (long long j = 0; j < k; ++j) {
      acc += row[j] * x[j];
    }
    y[i] = acc;
  }
  do_not_optimize(y);
}

CaseDecl make_sgemv_decl(long long n, long long k) {
  CaseDecl decl;
  decl.kernel = "sgemv_naive";
  decl.variant = "N" + std::to_string(n) + "_K" + std::to_string(k);
  decl.shape = {{"m", 1}, {"n", n}, {"k", k}};
  decl.dtype = "f32";
  decl.notes =
      "naive scalar reference at baseline flags; system probe, not a "
      "contract kernel";
  decl.flops = 2.0 * static_cast<double>(n) * static_cast<double>(k);
  decl.bytes_moved =
      4.0 * (static_cast<double>(n) * static_cast<double>(k) +
             static_cast<double>(k) + static_cast<double>(n));
  decl.make = [n, k]() {
    auto bufs = std::make_shared<Buffers>();
    bufs->w = aligned_alloc_array<float>(n * k);
    bufs->x = aligned_alloc_array<float>(k);
    bufs->y = aligned_alloc_array<float>(n);
    bufs->n = n;
    bufs->k = k;
    Rng rng;
    float* w = bufs->w.get();
    for (long long i = 0; i < n * k; ++i) {
      w[i] = rng.next();
    }
    float* x = bufs->x.get();
    for (long long j = 0; j < k; ++j) {
      x[j] = rng.next();
    }
    float* y = bufs->y.get();
    for (long long i = 0; i < n; ++i) {
      y[i] = 0.0f;
    }

    CaseBody body;
    body.target = [bufs]() { run_sgemv(*bufs); };
    // No baselines: this case IS the naive baseline future variants beat.
    body.check = [bufs]() {
      run_sgemv(*bufs);
      const float* w = bufs->w.get();
      const float* x = bufs->x.get();
      const float* y = bufs->y.get();
      const long long n = bufs->n;
      const long long k = bufs->k;
      double max_abs = 0.0;
      double max_ref = 0.0;
      for (long long i = 0; i < n; ++i) {
        const float* row = w + i * k;
        double acc = 0.0;
        for (long long j = 0; j < k; ++j) {
          acc += static_cast<double>(row[j]) * static_cast<double>(x[j]);
        }
        max_abs = std::max(max_abs, std::fabs(y[i] - acc));
        max_ref = std::max(max_ref, std::fabs(acc));
      }
      return CheckResult{max_abs, max_abs / (max_ref + 1e-9)};
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_sgemv_naive_cases(const BuildCtx& ctx, std::vector<CaseDecl>& out) {
  using Shape = std::pair<long long, long long>;  // (n, k)
  static const std::vector<Shape> kSmoke = {{4096, 4096}};
  static const std::vector<Shape> kQuick = {
      {2048, 2048}, {4096, 4096}, {8192, 8192}, {16384, 4096}};
  static const std::vector<Shape> kComprehensive = [] {
    std::vector<Shape> shapes = {{2048, 2048}};
    for (const long long n : kQuantMatmulN) {
      for (const long long k : kQuantMatmulK) {
        shapes.emplace_back(n, k);  // 16384 x 16384 weight is 1 GiB
      }
    }
    return shapes;
  }();

  for (const auto& [n, k] : pick(ctx.preset, kSmoke, kQuick, kComprehensive)) {
    out.push_back(make_sgemv_decl(n, k));
  }
}

}  // namespace qcb
