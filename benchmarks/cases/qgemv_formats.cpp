// Continuation cases for q4_0 weight-only GEMV and q4_0/q8_0 W8A8 GEMV.
// Every public target carries its direct portable reference, a decomposed
// dequantized-f32 baseline, and an independent f64 correctness oracle.

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness/alloc.h"
#include "harness/case.h"
#include "harness/donotopt.h"
#include "kernels/quantization/qgemv.h"
#include "kernels/quantization/qgemv_w8a8.h"
#include "quixicore_cpu/qgemv.h"
#include "quixicore_cpu/qgemv_w8a8.h"

namespace qcb {
namespace {

using quixicore_cpu::QuantFormat;
using quixicore_cpu::Status;

class Rng {
 public:
  float next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 17;
    state_ ^= state_ << 5;
    return static_cast<float>(state_ >> 8) * (2.0f / 16777216.0f) - 1.0f;
  }

 private:
  std::uint32_t state_ = 0xB5297A4Du;
};

struct Buffers {
  AlignedBuffer<std::uint8_t> packed;
  AlignedBuffer<float> x;
  AlignedBuffer<float> y;
  AlignedBuffer<float> dequantized;
  QuantFormat format = QuantFormat::kQ4_0;
  long long n = 0;
  long long k = 0;
};

void decomposed_gemv(Buffers& buffers) {
  if (quixicore_cpu::qgemv_unpack(
          buffers.format, buffers.packed.get(), buffers.n, buffers.k,
          buffers.dequantized.get()) != Status::kOk) {
    throw std::runtime_error("qgemv_unpack failed");
  }
  for (long long row = 0; row < buffers.n; ++row) {
    const float* weights = buffers.dequantized.get() + row * buffers.k;
    float sum = 0.0f;
    for (long long column = 0; column < buffers.k; ++column) {
      sum += weights[column] * buffers.x.get()[column];
    }
    buffers.y.get()[row] = sum;
  }
}

CheckResult check_against_dequantized(Buffers& buffers,
                                      const Tolerance& tolerance) {
  if (quixicore_cpu::qgemv_unpack(
          buffers.format, buffers.packed.get(), buffers.n, buffers.k,
          buffers.dequantized.get()) != Status::kOk) {
    throw std::runtime_error("qgemv_unpack failed");
  }
  CheckResult check;
  for (long long row = 0; row < buffers.n; ++row) {
    double sum = 0.0;
    for (long long column = 0; column < buffers.k; ++column) {
      sum += static_cast<double>(
                 buffers.dequantized.get()[row * buffers.k + column]) *
             buffers.x.get()[column];
    }
    check_value(check, buffers.y.get()[row], sum, tolerance);
  }
  return check;
}

CheckResult check_w8a8(Buffers& buffers) {
  if (quixicore_cpu::qgemv_unpack(
          buffers.format, buffers.packed.get(), buffers.n, buffers.k,
          buffers.dequantized.get()) != Status::kOk) {
    throw std::runtime_error("qgemv_unpack failed");
  }
  std::vector<double> quantized_input(static_cast<std::size_t>(buffers.k));
  for (long long block = 0; block < buffers.k / 32; ++block) {
    float maximum = 0.0f;
    for (long long element = 0; element < 32; ++element) {
      maximum = std::fmax(
          maximum,
          std::fabs(buffers.x.get()[block * 32 + element]));
    }
    const float scale = maximum / 127.0f;
    const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
    for (long long element = 0; element < 32; ++element) {
      const float rounded = std::nearbyint(
          buffers.x.get()[block * 32 + element] * inverse);
      const float clamped = rounded < -127.0f
                                ? -127.0f
                                : (rounded > 127.0f ? 127.0f : rounded);
      quantized_input[static_cast<std::size_t>(block * 32 + element)] =
          static_cast<double>(scale) * clamped;
    }
  }
  CheckResult check;
  for (long long row = 0; row < buffers.n; ++row) {
    double sum = 0.0;
    for (long long column = 0; column < buffers.k; ++column) {
      sum += static_cast<double>(
                 buffers.dequantized.get()[row * buffers.k + column]) *
             quantized_input[static_cast<std::size_t>(column)];
    }
    check_value(check, buffers.y.get()[row], sum,
                Tolerance{1e-4, 1e-4});
  }
  return check;
}

std::shared_ptr<Buffers> make_buffers(QuantFormat format, long long n,
                                      long long k) {
  std::size_t packed_size = 0;
  if (quixicore_cpu::qgemv_packed_size(format, n, k, &packed_size) !=
      Status::kOk) {
    throw std::runtime_error("qgemv_packed_size failed");
  }
  auto buffers = std::make_shared<Buffers>();
  buffers->format = format;
  buffers->n = n;
  buffers->k = k;
  buffers->packed = aligned_alloc_array<std::uint8_t>(
      static_cast<long long>(packed_size));
  buffers->x = aligned_alloc_array<float>(k);
  buffers->y = aligned_alloc_array<float>(n);
  buffers->dequantized = aligned_alloc_array<float>(n * k);
  auto weights = aligned_alloc_array<float>(n * k);
  Rng rng;
  for (long long index = 0; index < n * k; ++index) {
    weights.get()[index] = rng.next();
  }
  for (long long index = 0; index < k; ++index) {
    buffers->x.get()[index] = rng.next();
  }
  if (quixicore_cpu::qgemv_pack(format, weights.get(), n, k,
                                buffers->packed.get()) != Status::kOk) {
    throw std::runtime_error("qgemv_pack failed");
  }
  return buffers;
}

CaseDecl make_q4_weight_only(long long n, long long k) {
  CaseDecl decl;
  decl.kernel = "qgemv";
  decl.variant = "q4_0_N" + std::to_string(n) + "_K" + std::to_string(k);
  decl.shape = {{"m", 1}, {"n", n}, {"k", k}};
  decl.dtype = "f32";
  decl.format = "q4_0";
  decl.notes = "public q4_0 weight-only qgemv, portable reference";
  decl.flops = 2.0 * n * k;
  decl.weight_bytes = static_cast<double>(n) * (k / 32) * 18.0;
  decl.make = [n, k]() {
    auto buffers = make_buffers(QuantFormat::kQ4_0, n, k);
    CaseBody body;
    body.target = [buffers]() {
      if (quixicore_cpu::qgemv(
              buffers->format, buffers->packed.get(), buffers->x.get(),
              buffers->y.get(), buffers->n, buffers->k) != Status::kOk) {
        throw std::runtime_error("q4_0 qgemv failed");
      }
      do_not_optimize(buffers->y.get());
    };
    body.baselines.emplace_back("ref_scalar", [buffers]() {
      quixicore_cpu::quant::q4_0_gemv_ref(
          reinterpret_cast<const quixicore_cpu::quant::BlockQ4_0*>(
              buffers->packed.get()),
          buffers->x.get(), buffers->y.get(), buffers->n, buffers->k);
      do_not_optimize(buffers->y.get());
    });
    body.baselines.emplace_back("dequant_scalar", [buffers]() {
      decomposed_gemv(*buffers);
      do_not_optimize(buffers->y.get());
    });
    body.check = [buffers]() {
      if (quixicore_cpu::qgemv(
              buffers->format, buffers->packed.get(), buffers->x.get(),
              buffers->y.get(), buffers->n, buffers->k) != Status::kOk) {
        throw std::runtime_error("q4_0 qgemv failed");
      }
      return check_against_dequantized(*buffers, Tolerance{1e-4, 1e-4});
    };
    return body;
  };
  return decl;
}

CaseDecl make_w8a8(QuantFormat format, long long n, long long k) {
  const bool q8 = format == QuantFormat::kQ8_0;
  const std::string format_name = q8 ? "q8_0" : "q4_0";
  CaseDecl decl;
  decl.kernel = "qgemv_w8a8";
  decl.variant = format_name + "_N" + std::to_string(n) + "_K" +
                 std::to_string(k);
  decl.shape = {{"m", 1}, {"n", n}, {"k", k}};
  decl.dtype = "f32_to_i8";
  decl.format = format_name;
  decl.notes = std::string("public W8A8 qgemv, variant ") +
               quixicore_cpu::qgemv_w8a8_variant(format);
  decl.flops = 2.0 * n * k;
  decl.weight_bytes = static_cast<double>(n) * (k / 32) * (q8 ? 34.0 : 18.0);
  decl.make = [format, n, k, q8]() {
    auto buffers = make_buffers(format, n, k);
    CaseBody body;
    body.target = [buffers]() {
      if (quixicore_cpu::qgemv_w8a8(
              buffers->format, buffers->packed.get(), buffers->x.get(),
              buffers->y.get(), buffers->n, buffers->k) != Status::kOk) {
        throw std::runtime_error("qgemv_w8a8 failed");
      }
      do_not_optimize(buffers->y.get());
    };
    body.baselines.emplace_back("ref_scalar", [buffers, q8]() {
      if (q8) {
        quixicore_cpu::quant::q8_0_gemv_w8a8_ref(
            reinterpret_cast<const quixicore_cpu::quant::BlockQ8_0*>(
                buffers->packed.get()),
            buffers->x.get(), buffers->y.get(), buffers->n, buffers->k);
      } else {
        quixicore_cpu::quant::q4_0_gemv_w8a8_ref(
            reinterpret_cast<const quixicore_cpu::quant::BlockQ4_0*>(
                buffers->packed.get()),
            buffers->x.get(), buffers->y.get(), buffers->n, buffers->k);
      }
      do_not_optimize(buffers->y.get());
    });
    body.baselines.emplace_back("dequant_scalar", [buffers]() {
      decomposed_gemv(*buffers);
      do_not_optimize(buffers->y.get());
    });
    body.check = [buffers]() {
      if (quixicore_cpu::qgemv_w8a8(
              buffers->format, buffers->packed.get(), buffers->x.get(),
              buffers->y.get(), buffers->n, buffers->k) != Status::kOk) {
        throw std::runtime_error("qgemv_w8a8 failed");
      }
      return check_w8a8(*buffers);
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_qgemv_formats_cases(const BuildCtx& ctx,
                               std::vector<CaseDecl>& out) {
  const long long n = ctx.preset == Preset::kSmoke ? 256 : 4096;
  const long long k = ctx.preset == Preset::kSmoke ? 256 : 4096;
  out.push_back(make_q4_weight_only(n, k));
  out.push_back(make_w8a8(QuantFormat::kQ4_0, n, k));
  out.push_back(make_w8a8(QuantFormat::kQ8_0, n, k));
}

}  // namespace qcb
