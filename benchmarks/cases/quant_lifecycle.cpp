// Focused conversion-time and intermediate-activation benchmarks. Packing is
// intentionally inside the timed target; GEMV/GEMM cases elsewhere measure
// reusable packed weights outside their hot regions.

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness/case.h"
#include "harness/donotopt.h"
#include "quixicore_cpu/qgemv.h"

namespace qcb {
namespace {

using quixicore_cpu::QuantActivationFormat;
using quixicore_cpu::QuantFormat;
using quixicore_cpu::Status;

const char* quant_name(QuantFormat format) {
  switch (format) {
    case QuantFormat::kIQ2_XXS: return "iq2_xxs";
    case QuantFormat::kIQ2_XS: return "iq2_xs";
    case QuantFormat::kIQ3_XXS: return "iq3_xxs";
    case QuantFormat::kIQ3_S: return "iq3_s";
    case QuantFormat::kIQ2_S: return "iq2_s";
    case QuantFormat::kIQ1_S: return "iq1_s";
    case QuantFormat::kIQ1_M: return "iq1_m";
    default: return "quant";
  }
}

const char* activation_name(QuantActivationFormat format) {
  switch (format) {
    case QuantActivationFormat::kQ8_0: return "q8_0";
    case QuantActivationFormat::kQ8_1: return "q8_1";
    case QuantActivationFormat::kQ8_K: return "q8_k";
  }
  return "activation";
}

struct PackBuffers {
  std::vector<float> input;
  std::vector<float> importance;
  std::vector<float> unpacked;
  std::vector<std::uint8_t> packed;
};

CaseDecl make_iq_pack(QuantFormat format, long long rows, long long k) {
  CaseDecl decl;
  decl.kernel = "quant_lifecycle";
  decl.variant = std::string("weighted_pack_") + quant_name(format) + "_R" +
                 std::to_string(rows) + "_K" + std::to_string(k);
  decl.shape = {{"rows", rows}, {"k", k}};
  decl.format = quant_name(format);
  decl.notes = "importance-aware GGUF IQ authoring; conversion is timed";
  decl.bytes_moved = static_cast<double>(rows * k) * 2.0 * sizeof(float);
  decl.make = [format, rows, k]() {
    std::size_t packed_bytes = 0;
    if (quixicore_cpu::qgemv_packed_size(format, rows, k, &packed_bytes) !=
        Status::kOk) {
      throw std::runtime_error("IQ packed size failed");
    }
    auto buffers = std::make_shared<PackBuffers>();
    buffers->input.resize(static_cast<std::size_t>(rows * k));
    buffers->importance.resize(buffers->input.size());
    buffers->unpacked.resize(buffers->input.size());
    buffers->packed.resize(packed_bytes);
    for (std::size_t i = 0; i < buffers->input.size(); ++i) {
      buffers->input[i] = std::sin(0.0071f * static_cast<float>(i)) +
                          0.3f * std::cos(0.0019f * static_cast<float>(i));
      buffers->importance[i] =
          0.25f + static_cast<float>((i * 17) % 29) / 11.0f;
    }
    CaseBody body;
    body.target = [buffers, format, rows, k]() {
      if (quixicore_cpu::qgemv_pack_weighted(
              format, buffers->input.data(), buffers->importance.data(), rows,
              k, buffers->packed.data()) != Status::kOk) {
        throw std::runtime_error("weighted IQ pack failed");
      }
      do_not_optimize(buffers->packed.data());
    };
    body.baselines.emplace_back("uniform_importance", [buffers, format, rows, k]() {
      if (quixicore_cpu::qgemv_pack(format, buffers->input.data(), rows, k,
                                    buffers->packed.data()) != Status::kOk) {
        throw std::runtime_error("uniform IQ pack failed");
      }
      do_not_optimize(buffers->packed.data());
    });
    body.check = [buffers, format, rows, k]() {
      if (quixicore_cpu::qgemv_pack_weighted(
              format, buffers->input.data(), buffers->importance.data(), rows,
              k, buffers->packed.data()) != Status::kOk ||
          quixicore_cpu::qgemv_unpack(format, buffers->packed.data(), rows, k,
                                      buffers->unpacked.data()) != Status::kOk) {
        throw std::runtime_error("IQ lifecycle check failed");
      }
      CheckResult check;
      double squared_error = 0.0;
      double squared_reference = 0.0;
      for (std::size_t i = 0; i < buffers->input.size(); ++i) {
        if (!std::isfinite(buffers->unpacked[i])) {
          check.passed = false;
          check.finite = false;
          return check;
        }
        const double error = buffers->input[i] - buffers->unpacked[i];
        squared_error += error * error;
        squared_reference +=
            static_cast<double>(buffers->input[i]) * buffers->input[i];
      }
      check.max_rel_err = squared_error / std::max(1e-30, squared_reference);
      check.max_abs_err = std::sqrt(squared_error / buffers->input.size());
      check.passed = check.max_rel_err < 0.15;
      return check;
    };
    return body;
  };
  return decl;
}

CaseDecl make_activation_pack(QuantActivationFormat format, long long rows,
                              long long k) {
  CaseDecl decl;
  decl.kernel = "quant_lifecycle";
  decl.variant = std::string("activation_pack_") + activation_name(format) +
                 "_R" + std::to_string(rows) + "_K" + std::to_string(k);
  decl.shape = {{"rows", rows}, {"k", k}};
  decl.format = activation_name(format);
  decl.notes = "canonical llama activation/intermediate quantization";
  decl.bytes_moved = static_cast<double>(rows * k) * sizeof(float);
  decl.make = [format, rows, k]() {
    std::size_t packed_bytes = 0;
    if (quixicore_cpu::quant_activation_packed_size(format, rows, k,
                                                     &packed_bytes) !=
        Status::kOk) {
      throw std::runtime_error("activation packed size failed");
    }
    auto buffers = std::make_shared<PackBuffers>();
    buffers->input.resize(static_cast<std::size_t>(rows * k));
    buffers->unpacked.resize(buffers->input.size());
    buffers->packed.resize(packed_bytes);
    for (std::size_t i = 0; i < buffers->input.size(); ++i) {
      buffers->input[i] = std::sin(0.0031f * static_cast<float>(i));
    }
    CaseBody body;
    body.target = [buffers, format, rows, k]() {
      if (quixicore_cpu::quant_activation_pack(
              format, buffers->input.data(), rows, k,
              buffers->packed.data()) != Status::kOk) {
        throw std::runtime_error("activation pack failed");
      }
      do_not_optimize(buffers->packed.data());
    };
    body.check = [buffers, format, rows, k]() {
      if (quixicore_cpu::quant_activation_pack(
              format, buffers->input.data(), rows, k,
              buffers->packed.data()) != Status::kOk ||
          quixicore_cpu::quant_activation_unpack(
              format, buffers->packed.data(), rows, k,
              buffers->unpacked.data()) != Status::kOk) {
        throw std::runtime_error("activation lifecycle check failed");
      }
      CheckResult check;
      for (std::size_t i = 0; i < buffers->input.size(); ++i) {
        check_value(check, buffers->unpacked[i], buffers->input[i],
                    Tolerance{0.01, 0.01});
      }
      return check;
    };
    return body;
  };
  return decl;
}

struct QuantizedGemvBuffers {
  std::vector<std::uint8_t> weights;
  std::vector<std::uint8_t> activation;
  std::vector<float> activation_f32;
  std::vector<float> output;
  std::vector<float> expected;
};

CaseDecl make_quantized_activation_gemv(long long n, long long k) {
  CaseDecl decl;
  decl.kernel = "quant_lifecycle";
  decl.variant = "q4_k_x_q8_k_N" + std::to_string(n) + "_K" +
                 std::to_string(k);
  decl.shape = {{"m", 1}, {"n", n}, {"k", k}};
  decl.format = "q4_k_x_q8_k";
  decl.flops = 2.0 * n * k;
  decl.notes = "quantized-activation contract route";
  decl.make = [n, k]() {
    std::size_t weight_bytes = 0;
    std::size_t activation_bytes = 0;
    quixicore_cpu::qgemv_packed_size(QuantFormat::kQ4_K, n, k,
                                     &weight_bytes);
    quixicore_cpu::quant_activation_packed_size(
        QuantActivationFormat::kQ8_K, 1, k, &activation_bytes);
    auto buffers = std::make_shared<QuantizedGemvBuffers>();
    buffers->weights.resize(weight_bytes);
    buffers->activation.resize(activation_bytes);
    buffers->activation_f32.resize(static_cast<std::size_t>(k));
    buffers->output.resize(static_cast<std::size_t>(n));
    buffers->expected.resize(static_cast<std::size_t>(n));
    std::vector<float> dense_weights(static_cast<std::size_t>(n * k));
    std::vector<float> dense_activation(static_cast<std::size_t>(k));
    for (std::size_t i = 0; i < dense_weights.size(); ++i) {
      dense_weights[i] = std::sin(0.00031f * static_cast<float>(i));
    }
    for (long long i = 0; i < k; ++i) {
      dense_activation[static_cast<std::size_t>(i)] =
          std::cos(0.0037f * static_cast<float>(i));
    }
    if (quixicore_cpu::qgemv_pack(QuantFormat::kQ4_K, dense_weights.data(), n,
                                  k, buffers->weights.data()) != Status::kOk ||
        quixicore_cpu::quant_activation_pack(
            QuantActivationFormat::kQ8_K, dense_activation.data(), 1, k,
            buffers->activation.data()) != Status::kOk) {
      throw std::runtime_error("quantized GEMV setup failed");
    }
    CaseBody body;
    body.target = [buffers, n, k]() {
      if (quixicore_cpu::qgemv_quantized_activation(
              QuantFormat::kQ4_K, buffers->weights.data(),
              QuantActivationFormat::kQ8_K, buffers->activation.data(),
              buffers->output.data(), n, k) != Status::kOk) {
        throw std::runtime_error("quantized activation GEMV failed");
      }
      do_not_optimize(buffers->output.data());
    };
    body.baselines.emplace_back("explicit_unpack", [buffers, n, k]() {
      quixicore_cpu::quant_activation_unpack(
          QuantActivationFormat::kQ8_K, buffers->activation.data(), 1, k,
          buffers->activation_f32.data());
      quixicore_cpu::qgemv(QuantFormat::kQ4_K, buffers->weights.data(),
                           buffers->activation_f32.data(),
                           buffers->expected.data(), n, k);
      do_not_optimize(buffers->expected.data());
    });
    body.check = [buffers, n, k]() {
      quixicore_cpu::qgemv_quantized_activation(
          QuantFormat::kQ4_K, buffers->weights.data(),
          QuantActivationFormat::kQ8_K, buffers->activation.data(),
          buffers->output.data(), n, k);
      quixicore_cpu::quant_activation_unpack(
          QuantActivationFormat::kQ8_K, buffers->activation.data(), 1, k,
          buffers->activation_f32.data());
      quixicore_cpu::qgemv(QuantFormat::kQ4_K, buffers->weights.data(),
                           buffers->activation_f32.data(),
                           buffers->expected.data(), n, k);
      CheckResult check;
      for (long long i = 0; i < n; ++i) {
        check_value(check, buffers->output[static_cast<std::size_t>(i)],
                    buffers->expected[static_cast<std::size_t>(i)],
                    Tolerance{0.0, 0.0});
      }
      return check;
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_quant_lifecycle_cases(const BuildCtx& ctx,
                                 std::vector<CaseDecl>& out) {
  const long long k = ctx.preset == Preset::kSmoke ? 256 : 4096;
  const long long rows = ctx.preset == Preset::kComprehensive ? 8 : 1;
  for (QuantFormat format : {
           QuantFormat::kIQ2_XXS, QuantFormat::kIQ2_XS,
           QuantFormat::kIQ3_XXS, QuantFormat::kIQ3_S,
           QuantFormat::kIQ2_S, QuantFormat::kIQ1_S,
           QuantFormat::kIQ1_M}) {
    out.push_back(make_iq_pack(format, rows, k));
  }
  const long long activation_rows = ctx.preset == Preset::kSmoke ? 4 : 64;
  for (QuantActivationFormat format : {
           QuantActivationFormat::kQ8_0, QuantActivationFormat::kQ8_1,
           QuantActivationFormat::kQ8_K}) {
    out.push_back(make_activation_pack(format, activation_rows, k));
  }
  const long long n = ctx.preset == Preset::kSmoke ? 64 : 1024;
  out.push_back(make_quantized_activation_gemv(n, k));
}

}  // namespace qcb
