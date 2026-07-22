// qgemv: quantized GEMV (GGUF q8_0) through the public qgemv entry point —
// family contract out = dequantize(wq) @ x, f32 activations.
//
// Baselines per the three-baseline rule (perf/perf.md):
//   ref_scalar      — the library's scalar reference called directly (the
//                     prior implementation ISA variants must beat),
//   scalar_multiacc — manual 4-accumulator scalar variant, rejected in the
//                     2026-07-07 optimization run (2-3% slower than the
//                     auto-vectorized plain loop); kept here so the A/B
//                     stays reproducible,
//   dequant_sgemv   — naive decomposed path: unpack to f32, then scalar GEMV,
//   dotprod_i8      — the activation-quantizing int8 SDOT path used by the
//                     separate qgemv_w8a8 operation; retained here as context,
//                     never selected by weight-only qgemv dispatch.
// The roofline comparison is weight_gbps vs mem_triad DRAM bandwidth.
//
// The in-harness oracle is the family oracle (dequantized weights x f32
// activations); the contract path should report ~1e-7 here. Per-variant
// oracles live in tests/correctness/test_qgemv.cpp.
//
// Shape provenance: umbrella quant_matmul family at m = 1 (decode), plus
// 2048 from decode_small hidden sizes.

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "harness/alloc.h"
#include "harness/case.h"
#include "harness/donotopt.h"
#include "harness/shapes.h"
#include "kernels/common/fp16.h"
#include "kernels/quantization/qgemv.h"
#include "quixicore_cpu/cpu_features.h"
#include "quixicore_cpu/qgemv.h"

namespace qcb {
namespace {

using quixicore_cpu::QuantFormat;
using quixicore_cpu::Status;
using quixicore_cpu::quant::BlockQ8_0;
using quixicore_cpu::quant::kQ8_0BlockSize;

// qgemv's benchmark oracle starts from exactly dequantized packed weights, so
// storage quantization error is absent. This is stricter than the umbrella
// quantized tolerance and isolates the public f32-activation accumulation.
constexpr Tolerance kQgemvAccumTolerance{1e-4, 1e-4};

class Rng {
 public:
  float next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 17;
    state_ ^= state_ << 5;
    return static_cast<float>(state_ >> 8) * (1.0f / 16777216.0f) - 0.5f;
  }

 private:
  uint32_t state_ = 0x2545F491u;
};

struct Buffers {
  AlignedBuffer<uint8_t> packed;
  AlignedBuffer<float> x;
  AlignedBuffer<float> y;
  AlignedBuffer<float> scratch;  // n*k f32 for the decomposed baseline
  long long n = 0;
  long long k = 0;
};

// Rejected candidate from the 2026-07-07 optimization run: manual 4-way
// accumulator split. Measured 2-3% slower than the plain loop the compiler
// auto-vectorizes; kept as a baseline so the comparison stays reproducible.
void scalar_multiacc_gemv(const BlockQ8_0* packed, const float* x, float* y,
                          long long n, long long k) {
  const long long blocks_per_row = k / kQ8_0BlockSize;
  for (long long i = 0; i < n; ++i) {
    const BlockQ8_0* row = packed + i * blocks_per_row;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (long long b = 0; b < blocks_per_row; ++b) {
      const float d = quixicore_cpu::fp16_to_fp32(row[b].d);
      const int8_t* q = row[b].qs;
      const float* xb = x + b * kQ8_0BlockSize;
      float s0 = 0.0f;
      float s1 = 0.0f;
      float s2 = 0.0f;
      float s3 = 0.0f;
      for (int j = 0; j < 8; ++j) {
        s0 += static_cast<float>(q[j]) * xb[j];
        s1 += static_cast<float>(q[j + 8]) * xb[j + 8];
        s2 += static_cast<float>(q[j + 16]) * xb[j + 16];
        s3 += static_cast<float>(q[j + 24]) * xb[j + 24];
      }
      acc0 += d * s0;
      acc1 += d * s1;
      acc2 += d * s2;
      acc3 += d * s3;
    }
    y[i] = (acc0 + acc1) + (acc2 + acc3);
  }
}

CaseDecl make_qgemv_decl(long long n, long long k) {
  CaseDecl decl;
  decl.kernel = "qgemv";
  decl.variant = "N" + std::to_string(n) + "_K" + std::to_string(k);
  decl.shape = {{"m", 1}, {"n", n}, {"k", k}};
  decl.dtype = "f32";
  decl.format = "q8_0";
  decl.notes = std::string("public qgemv, variant ") +
               quixicore_cpu::qgemv_variant(QuantFormat::kQ8_0);
  decl.flops = 2.0 * static_cast<double>(n) * static_cast<double>(k);
  const double weight_bytes = static_cast<double>(n) *
                              static_cast<double>(k / kQ8_0BlockSize) *
                              sizeof(BlockQ8_0);
  decl.weight_bytes = weight_bytes;
  decl.bytes_moved = weight_bytes + 4.0 * static_cast<double>(n + k);
  decl.make = [n, k]() {
    size_t packed_size = 0;
    if (quixicore_cpu::qgemv_packed_size(QuantFormat::kQ8_0, n, k,
                                              &packed_size) != Status::kOk) {
      throw std::runtime_error("qgemv_packed_size failed");
    }

    auto bufs = std::make_shared<Buffers>();
    bufs->n = n;
    bufs->k = k;
    bufs->packed =
        aligned_alloc_array<uint8_t>(static_cast<long long>(packed_size));
    bufs->x = aligned_alloc_array<float>(k);
    bufs->y = aligned_alloc_array<float>(n);
    bufs->scratch = aligned_alloc_array<float>(n * k);

    {
      // Quantize from a temporary f32 weight matrix, then release it so a
      // live case holds only packed weights + activations + scratch.
      auto weights = aligned_alloc_array<float>(n * k);
      Rng rng;
      for (long long i = 0; i < n * k; ++i) {
        weights.get()[i] = rng.next();
      }
      if (quixicore_cpu::qgemv_pack(QuantFormat::kQ8_0, weights.get(), n,
                                         k, bufs->packed.get()) !=
          Status::kOk) {
        throw std::runtime_error("qgemv_pack failed");
      }
    }
    Rng rng_x;
    for (long long j = 0; j < k; ++j) {
      bufs->x.get()[j] = rng_x.next();
    }
    for (long long i = 0; i < n; ++i) {
      bufs->y.get()[i] = 0.0f;
    }
    for (long long i = 0; i < n * k; ++i) {
      bufs->scratch.get()[i] = 0.0f;  // first-touch before timing
    }

    CaseBody body;
    body.target = [bufs]() {
      if (quixicore_cpu::qgemv(QuantFormat::kQ8_0, bufs->packed.get(),
                                    bufs->x.get(), bufs->y.get(), bufs->n,
                                    bufs->k) != Status::kOk) {
        throw std::runtime_error("qgemv failed");
      }
      do_not_optimize(bufs->y.get());
    };
    body.baselines.emplace_back("ref_scalar", [bufs]() {
      quixicore_cpu::quant::q8_0_gemv_ref(
          reinterpret_cast<const BlockQ8_0*>(bufs->packed.get()),
          bufs->x.get(), bufs->y.get(), bufs->n, bufs->k);
      do_not_optimize(bufs->y.get());
    });
    body.baselines.emplace_back("scalar_multiacc", [bufs]() {
      scalar_multiacc_gemv(
          reinterpret_cast<const BlockQ8_0*>(bufs->packed.get()),
          bufs->x.get(), bufs->y.get(), bufs->n, bufs->k);
      do_not_optimize(bufs->y.get());
    });
#if defined(QUIXICORE_CPU_HAVE_QGEMV_DOTPROD)
    if (quixicore_cpu::cpu_features().dotprod) {
      body.baselines.emplace_back("dotprod_i8", [bufs]() {
        quixicore_cpu::quant::q8_0_gemv_dotprod(
            reinterpret_cast<const BlockQ8_0*>(bufs->packed.get()),
            bufs->x.get(), bufs->y.get(), bufs->n, bufs->k);
        do_not_optimize(bufs->y.get());
      });
    }
#endif
    body.baselines.emplace_back("dequant_sgemv", [bufs]() {
      if (quixicore_cpu::qgemv_unpack(QuantFormat::kQ8_0,
                                           bufs->packed.get(), bufs->n,
                                           bufs->k, bufs->scratch.get()) !=
          Status::kOk) {
        throw std::runtime_error("qgemv_unpack failed");
      }
      const float* w = bufs->scratch.get();
      const float* x = bufs->x.get();
      float* y = bufs->y.get();
      for (long long i = 0; i < bufs->n; ++i) {
        const float* row = w + i * bufs->k;
        float acc = 0.0f;
        for (long long j = 0; j < bufs->k; ++j) {
          acc += row[j] * x[j];
        }
        y[i] = acc;
      }
      do_not_optimize(y);
    });
    body.check = [bufs]() {
      if (quixicore_cpu::qgemv(QuantFormat::kQ8_0, bufs->packed.get(),
                                    bufs->x.get(), bufs->y.get(), bufs->n,
                                    bufs->k) != Status::kOk) {
        throw std::runtime_error("qgemv failed");
      }
      if (quixicore_cpu::qgemv_unpack(QuantFormat::kQ8_0,
                                           bufs->packed.get(), bufs->n,
                                           bufs->k, bufs->scratch.get()) !=
          Status::kOk) {
        throw std::runtime_error("qgemv_unpack failed");
      }
      const float* dq = bufs->scratch.get();
      const float* x = bufs->x.get();
      const float* y = bufs->y.get();
      CheckResult check;
      for (long long i = 0; i < bufs->n; ++i) {
        const float* row = dq + i * bufs->k;
        double acc = 0.0;
        for (long long j = 0; j < bufs->k; ++j) {
          acc += static_cast<double>(row[j]) * static_cast<double>(x[j]);
        }
        check_value(check, y[i], acc, kQgemvAccumTolerance);
      }
      return check;
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_qgemv_cases(const BuildCtx& ctx, std::vector<CaseDecl>& out) {
  using Shape = std::pair<long long, long long>;  // (n, k)
  static const std::vector<Shape> kSmoke = {{4096, 4096}};
  static const std::vector<Shape> kQuick = {
      {4096, 4096}, {8192, 8192}, {16384, 4096}};
  static const std::vector<Shape> kComprehensive = [] {
    std::vector<Shape> shapes = {{2048, 2048}};
    for (const long long n : kQuantMatmulN) {
      for (const long long k : kQuantMatmulK) {
        shapes.emplace_back(n, k);
      }
    }
    return shapes;
  }();

  for (const auto& [n, k] : pick(ctx.preset, kSmoke, kQuick, kComprehensive)) {
    out.push_back(make_qgemv_decl(n, k));
  }
}

}  // namespace qcb
