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
#include "kernels/quantization/gguf_ref.h"
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

const char* format_name(QuantFormat format) {
  switch (format) {
    case QuantFormat::kQ1_0: return "q1_0";
    case QuantFormat::kQ2_0: return "q2_0";
    case QuantFormat::kQ4_0: return "q4_0";
    case QuantFormat::kQ4_1: return "q4_1";
    case QuantFormat::kQ5_0: return "q5_0";
    case QuantFormat::kQ5_1: return "q5_1";
    case QuantFormat::kQ8_0: return "q8_0";
    case QuantFormat::kMXFP4: return "mxfp4";
    case QuantFormat::kNVFP4: return "nvfp4";
    case QuantFormat::kQ2_K: return "q2_k";
    case QuantFormat::kQ3_K: return "q3_k";
    case QuantFormat::kQ4_K: return "q4_k";
    case QuantFormat::kQ5_K: return "q5_k";
    case QuantFormat::kQ6_K: return "q6_k";
    case QuantFormat::kIQ4_NL: return "iq4_nl";
    case QuantFormat::kIQ4_XS: return "iq4_xs";
    case QuantFormat::kIQ2_XXS: return "iq2_xxs";
    case QuantFormat::kIQ2_XS: return "iq2_xs";
    case QuantFormat::kIQ3_XXS: return "iq3_xxs";
    case QuantFormat::kIQ3_S: return "iq3_s";
    case QuantFormat::kIQ2_S: return "iq2_s";
    case QuantFormat::kIQ1_S: return "iq1_s";
    case QuantFormat::kIQ1_M: return "iq1_m";
    case QuantFormat::kTQ1_0: return "tq1_0";
    case QuantFormat::kTQ2_0: return "tq2_0";
    default: return "gguf";
  }
}

void set_half(std::uint8_t* bytes, std::size_t offset, float one = 1.0f) {
  const std::uint16_t bits = one == 0.0f ? 0x0000u : 0x3c00u;
  bytes[offset] = static_cast<std::uint8_t>(bits & 0xffu);
  bytes[offset + 1] = static_cast<std::uint8_t>(bits >> 8);
}

void normalize_block_scales(QuantFormat format, std::uint8_t* block) {
  switch (format) {
    case QuantFormat::kQ4_1:
    case QuantFormat::kQ5_1:
      set_half(block, 2);
      [[fallthrough]];
    case QuantFormat::kQ1_0:
    case QuantFormat::kQ2_0:
    case QuantFormat::kQ5_0:
    case QuantFormat::kIQ4_NL:
    case QuantFormat::kIQ4_XS:
    case QuantFormat::kIQ2_XXS:
    case QuantFormat::kIQ2_XS:
    case QuantFormat::kIQ3_XXS:
    case QuantFormat::kIQ3_S:
    case QuantFormat::kIQ2_S:
    case QuantFormat::kIQ1_S:
      set_half(block, 0);
      break;
    case QuantFormat::kQ2_K:
      set_half(block, 80);
      set_half(block, 82);
      break;
    case QuantFormat::kQ3_K:
      set_half(block, 108);
      break;
    case QuantFormat::kQ4_K:
    case QuantFormat::kQ5_K:
      set_half(block, 0);
      set_half(block, 2, 0.0f);
      break;
    case QuantFormat::kQ6_K:
      set_half(block, 208);
      break;
    case QuantFormat::kMXFP4:
      block[0] = 127;
      break;
    case QuantFormat::kNVFP4:
      for (int index = 0; index < 4; ++index) block[index] = 0x38;
      break;
    case QuantFormat::kIQ1_M:
      for (int index = 48; index < 56; ++index) block[index] = 0;
      block[53] = 0xc0;
      block[55] = 0x30;
      break;
    case QuantFormat::kTQ1_0:
      set_half(block, 52);
      break;
    case QuantFormat::kTQ2_0:
      set_half(block, 64);
      break;
    default:
      break;
  }
}

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
  decl.notes = std::string("public q4_0 weight-only qgemv, variant ") +
               quixicore_cpu::qgemv_variant(QuantFormat::kQ4_0);
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

CaseDecl make_gguf_weight_only(QuantFormat format, long long n, long long k) {
  const std::string name = format_name(format);
  CaseDecl decl;
  decl.kernel = "qgemv_formats";
  decl.variant = name + "_N" + std::to_string(n) + "_K" + std::to_string(k);
  decl.shape = {{"m", 1}, {"n", n}, {"k", k}};
  decl.dtype = "f32";
  decl.format = name;
  decl.notes = std::string("GGUF direct packed block-dot GEMV, variant ") +
               quixicore_cpu::qgemv_variant(format);
  decl.flops = 2.0 * n * k;
  std::size_t packed_size = 0;
  if (quixicore_cpu::qgemv_packed_size(format, n, k, &packed_size) !=
      Status::kOk) {
    throw std::runtime_error("GGUF benchmark packed size failed");
  }
  decl.weight_bytes = static_cast<double>(packed_size);
  decl.make = [format, n, k, packed_size]() {
    long long block_size = 0;
    std::size_t block_bytes = 0;
    if (!quixicore_cpu::quant::gguf_format_info(
            format, &block_size, &block_bytes)) {
      throw std::runtime_error("GGUF format info failed");
    }
    auto buffers = std::make_shared<Buffers>();
    buffers->format = format;
    buffers->n = n;
    buffers->k = k;
    buffers->packed = aligned_alloc_array<std::uint8_t>(
        static_cast<long long>(packed_size));
    buffers->x = aligned_alloc_array<float>(k);
    buffers->y = aligned_alloc_array<float>(n);
    buffers->dequantized = aligned_alloc_array<float>(n);
    Rng rng;
    for (std::size_t i = 0; i < packed_size; ++i) {
      buffers->packed.get()[i] =
          static_cast<std::uint8_t>((i * 73u + 19u) & 0xffu);
    }
    for (long long i = 0; i < k; ++i) buffers->x.get()[i] = rng.next();
    for (std::size_t offset = 0; offset < packed_size; offset += block_bytes) {
      normalize_block_scales(format, buffers->packed.get() + offset);
    }
    CaseBody body;
    body.target = [buffers]() {
      if (quixicore_cpu::qgemv(
              buffers->format, buffers->packed.get(), buffers->x.get(),
              buffers->y.get(), buffers->n, buffers->k) != Status::kOk) {
        throw std::runtime_error("GGUF qgemv failed");
      }
      do_not_optimize(buffers->y.get());
    };
    body.baselines.emplace_back("element_decode_ref", [buffers]() {
      quixicore_cpu::quant::gguf_gemv_ref(
          buffers->format, buffers->packed.get(), buffers->x.get(),
          buffers->dequantized.get(), buffers->n, buffers->k);
      do_not_optimize(buffers->dequantized.get());
    });
    body.check = [buffers]() {
      if (quixicore_cpu::qgemv(
              buffers->format, buffers->packed.get(), buffers->x.get(),
              buffers->y.get(), buffers->n, buffers->k) != Status::kOk) {
        throw std::runtime_error("GGUF qgemv failed");
      }
      quixicore_cpu::quant::gguf_gemv_ref(
          buffers->format, buffers->packed.get(), buffers->x.get(),
          buffers->dequantized.get(), buffers->n, buffers->k);
      CheckResult check;
      for (long long row = 0; row < buffers->n; ++row) {
        check_value(check, buffers->y.get()[row],
                    buffers->dequantized.get()[row],
                    Tolerance{2e-3, 2e-4});
      }
      return check;
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
  const long long generic_n = ctx.preset == Preset::kSmoke ? 64 : 1024;
  for (QuantFormat format : {
           QuantFormat::kQ1_0, QuantFormat::kQ2_0,
           QuantFormat::kQ4_1, QuantFormat::kQ5_0,
           QuantFormat::kQ5_1, QuantFormat::kMXFP4,
           QuantFormat::kNVFP4, QuantFormat::kQ2_K,
           QuantFormat::kQ3_K, QuantFormat::kQ4_K,
           QuantFormat::kQ5_K, QuantFormat::kQ6_K,
           QuantFormat::kIQ4_NL, QuantFormat::kIQ4_XS,
           QuantFormat::kIQ2_XXS, QuantFormat::kIQ2_XS,
           QuantFormat::kIQ3_XXS, QuantFormat::kIQ3_S,
           QuantFormat::kIQ2_S, QuantFormat::kIQ1_S,
           QuantFormat::kIQ1_M, QuantFormat::kTQ1_0,
           QuantFormat::kTQ2_0}) {
    out.push_back(make_gguf_weight_only(format, generic_n, k));
  }
}

}  // namespace qcb
