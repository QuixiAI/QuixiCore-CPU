#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness/case.h"
#include "harness/donotopt.h"
#include "kernels/quantization/int8_gemm.h"
#include "kernels/quantization/lowbit.h"
#include "kernels/quantization/w8a32_gemm.h"
#include "quixicore_cpu/lowbit.h"
#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/qgemm.h"
#include "quixicore_cpu/qgemv.h"
#include "quixicore_cpu/quantization.h"

namespace qcb {
namespace {

using quixicore_cpu::LowBitFormat;
using quixicore_cpu::QuantFormat;
using quixicore_cpu::Status;

struct Rng {
  std::uint32_t state = 0x91E10DA5u;
  std::uint32_t bits() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  }
  float f32() {
    return static_cast<float>(bits() >> 8) * (2.0f / 16777216.0f) - 1.0f;
  }
};

CheckResult check_arrays(const float* actual, const float* expected,
                         long long count, Tolerance tolerance) {
  CheckResult check;
  for (long long i = 0; i < count; ++i) {
    check_value(check, actual[i], expected[i], tolerance);
  }
  return check;
}

struct W8A32Buffers {
  std::vector<std::int8_t> weights;
  std::vector<float> scales, x, target, reference;
  long long m, n, k;
};

CaseDecl make_w8a32(long long m, long long n, long long k) {
  CaseDecl decl;
  decl.kernel = "colibri_ops";
  decl.variant = "w8a32_M" + std::to_string(m) + "_N" +
                 std::to_string(n) + "_K" + std::to_string(k);
  decl.shape = {{"m", m}, {"n", n}, {"k", k}};
  decl.format = "row_int8_f32_activation";
  decl.notes = "Colibri row-scale W8A32 route versus scalar reference";
  decl.flops = 2.0 * m * n * k;
  decl.make = [=]() {
    auto b = std::make_shared<W8A32Buffers>();
    b->m = m; b->n = n; b->k = k;
    b->weights.resize(static_cast<std::size_t>(n * k));
    b->scales.resize(static_cast<std::size_t>(n));
    b->x.resize(static_cast<std::size_t>(m * k));
    b->target.resize(static_cast<std::size_t>(m * n));
    b->reference.resize(static_cast<std::size_t>(m * n));
    Rng rng;
    for (auto& value : b->weights) {
      value = static_cast<std::int8_t>(static_cast<int>(rng.bits() % 255) - 127);
    }
    for (auto& value : b->scales) value = 0.001f + 0.01f * std::fabs(rng.f32());
    for (auto& value : b->x) value = rng.f32();
    auto target = [b]() {
      if (quixicore_cpu::qgemm_w8a32(
              b->weights.data(), b->scales.data(), b->x.data(),
              b->target.data(), b->m, b->n, b->k) != Status::kOk) {
        throw std::runtime_error("qgemm_w8a32 failed");
      }
      do_not_optimize(b->target.data());
    };
    auto baseline = [b]() {
      quixicore_cpu::quant::w8a32_gemm_ref(
          b->weights.data(), b->scales.data(), b->x.data(),
          b->reference.data(), b->m, b->n, b->k);
      do_not_optimize(b->reference.data());
    };
    CaseBody body;
    body.target = target;
    body.baselines.emplace_back("scalar_ref", baseline);
    body.check = [b, target, baseline]() {
      target(); baseline();
      return check_arrays(b->target.data(), b->reference.data(), b->m * b->n,
                          Tolerance{3e-5, 3e-5});
    };
    return body;
  };
  return decl;
}

struct Int8Buffers {
  std::vector<std::int8_t> weights, x;
  std::vector<float> weight_scales, activation_scales, target, reference;
  long long m, n, k;
};

CaseDecl make_int8(long long m, long long n, long long k) {
  CaseDecl decl;
  decl.kernel = "colibri_ops";
  decl.variant = "int8_idot_M" + std::to_string(m) + "_N" +
                 std::to_string(n) + "_K" + std::to_string(k);
  decl.shape = {{"m", m}, {"n", n}, {"k", k}};
  decl.format = "w8a8_row_scale";
  decl.notes = "prequantized IDOT dispatch versus exact scalar integer dot";
  decl.flops = 2.0 * m * n * k;
  decl.make = [=]() {
    auto b = std::make_shared<Int8Buffers>();
    b->m = m; b->n = n; b->k = k;
    b->weights.resize(static_cast<std::size_t>(n * k));
    b->x.resize(static_cast<std::size_t>(m * k));
    b->weight_scales.resize(static_cast<std::size_t>(n));
    b->activation_scales.resize(static_cast<std::size_t>(m));
    b->target.resize(static_cast<std::size_t>(m * n));
    b->reference.resize(static_cast<std::size_t>(m * n));
    Rng rng;
    for (auto& value : b->weights) value = static_cast<std::int8_t>(int(rng.bits() % 255) - 127);
    for (auto& value : b->x) value = static_cast<std::int8_t>(int(rng.bits() % 255) - 127);
    for (auto& value : b->weight_scales) value = 0.001f + 0.01f * std::fabs(rng.f32());
    for (auto& value : b->activation_scales) value = 0.001f + 0.01f * std::fabs(rng.f32());
    auto target = [b]() {
      if (quixicore_cpu::int8_gemm(
              b->weights.data(), b->x.data(), b->weight_scales.data(),
              b->activation_scales.data(), nullptr, nullptr, b->target.data(),
              b->m, b->n, b->k, false) != Status::kOk) {
        throw std::runtime_error("int8_gemm failed");
      }
      do_not_optimize(b->target.data());
    };
    auto baseline = [b]() {
      quixicore_cpu::quant::int8_gemm_ref_kernel(
          b->weights.data(), b->x.data(), b->weight_scales.data(),
          b->activation_scales.data(), nullptr, nullptr, b->reference.data(),
          b->m, b->n, b->k, false);
      do_not_optimize(b->reference.data());
    };
    CaseBody body;
    body.target = target;
    body.baselines.emplace_back("scalar_ref", baseline);
    body.check = [b, target, baseline]() {
      target(); baseline();
      return check_arrays(b->target.data(), b->reference.data(), b->m * b->n,
                          Tolerance{0.0, 0.0});
    };
    return body;
  };
  return decl;
}

struct LowBitBuffers {
  std::vector<float> weights, scales, x, target, reference;
  std::vector<std::uint8_t> packed;
  std::vector<std::int8_t> xq;
  std::vector<float> xscale;
  long long m, n, k;
  bool idot;
};

CaseDecl make_lowbit(long long m, long long n, long long k, bool idot) {
  CaseDecl decl;
  decl.kernel = "colibri_ops";
  decl.variant = std::string(idot ? "w4a8_" : "int4_f32_") + "M" +
                 std::to_string(m) + "_N" + std::to_string(n) + "_K" +
                 std::to_string(k);
  decl.shape = {{"m", m}, {"n", n}, {"k", k}};
  decl.format = idot ? "int4_row_dynamic_int8" : "int4_row_f32";
  decl.notes = idot ? "dynamic W4A8 dispatch versus quantize plus scalar IDOT"
                    : "packed int4 SIMD route versus scalar unpack-on-use";
  decl.flops = 2.0 * m * n * k;
  decl.make = [=]() {
    auto b = std::make_shared<LowBitBuffers>();
    b->m = m; b->n = n; b->k = k; b->idot = idot;
    b->weights.resize(static_cast<std::size_t>(n * k));
    b->x.resize(static_cast<std::size_t>(m * k));
    b->target.resize(static_cast<std::size_t>(m * n));
    b->reference.resize(static_cast<std::size_t>(m * n));
    std::size_t bytes = 0, scales = 0;
    if (quixicore_cpu::lowbit_packed_size(
            LowBitFormat::kInt4Row, n, k, 0, &bytes, &scales) != Status::kOk) {
      throw std::runtime_error("lowbit size failed");
    }
    b->packed.resize(bytes); b->scales.resize(scales);
    b->xq.resize(static_cast<std::size_t>(m * k));
    b->xscale.resize(static_cast<std::size_t>(m));
    Rng rng;
    for (auto& value : b->weights) value = 0.2f * rng.f32();
    for (auto& value : b->x) value = rng.f32();
    if (quixicore_cpu::lowbit_pack(
            LowBitFormat::kInt4Row, b->weights.data(), b->packed.data(),
            b->scales.data(), n, k) != Status::kOk) {
      throw std::runtime_error("lowbit pack failed");
    }
    auto target = [b]() {
      const Status status = b->idot
          ? quixicore_cpu::lowbit_gemm_w8a8(
                b->packed.data(), b->scales.data(), b->x.data(),
                b->target.data(), b->m, b->n, b->k)
          : quixicore_cpu::lowbit_gemm(
                LowBitFormat::kInt4Row, b->packed.data(), b->scales.data(),
                b->x.data(), b->target.data(), b->m, b->n, b->k);
      if (status != Status::kOk) throw std::runtime_error("lowbit GEMM failed");
      do_not_optimize(b->target.data());
    };
    auto baseline = [b]() {
      if (b->idot) {
        if (quixicore_cpu::quantize_int8(
                b->x.data(), b->xq.data(), b->xscale.data(), b->m, b->k,
                b->k) != Status::kOk) {
          throw std::runtime_error("lowbit baseline quantize failed");
        }
        quixicore_cpu::quant::lowbit_w8a8_ref(
            b->packed.data(), b->scales.data(), b->xq.data(),
            b->xscale.data(), b->reference.data(), b->m, b->n, b->k);
      } else {
        quixicore_cpu::quant::lowbit_gemm_ref(
            LowBitFormat::kInt4Row, b->packed.data(), b->scales.data(),
            b->x.data(), b->reference.data(), b->m, b->n, b->k, 0);
      }
      do_not_optimize(b->reference.data());
    };
    CaseBody body;
    body.target = target;
    body.baselines.emplace_back("scalar_ref", baseline);
    body.check = [b, target, baseline]() {
      target(); baseline();
      return check_arrays(b->target.data(), b->reference.data(), b->m * b->n,
                          b->idot ? Tolerance{0.0, 0.0}
                                  : Tolerance{3e-5, 3e-5});
    };
    return body;
  };
  return decl;
}

double uniform01(std::uint32_t seed) {
  std::uint64_t z = (static_cast<std::uint64_t>(seed) << 32) ^
                    UINT64_C(0x9E3779B97F4A7C15);
  z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
  z ^= z >> 31;
  return (static_cast<double>(z >> 11) + 0.5) *
         (1.0 / 9007199254740992.0);
}

int sorted_top_p(const std::vector<float>& logits, float p, float temperature,
                 std::uint32_t seed) {
  std::vector<int> ids(logits.size());
  std::iota(ids.begin(), ids.end(), 0);
  std::stable_sort(ids.begin(), ids.end(), [&](int lhs, int rhs) {
    return logits[lhs] == logits[rhs] ? lhs < rhs : logits[lhs] > logits[rhs];
  });
  const float maximum = logits[ids.front()];
  std::vector<double> weights(logits.size());
  double total = 0.0;
  for (std::size_t token = 0; token < logits.size(); ++token) {
    weights[token] = std::exp((double(logits[token]) - maximum) / temperature);
    total += weights[token];
  }
  double cumulative = 0.0;
  std::size_t keep = 0;
  do { cumulative += weights[ids[keep++]]; }
  while (keep < ids.size() && cumulative < p * total);
  ids.resize(keep);
  double kept = 0.0;
  for (int id : ids) kept += weights[id];
  const double target = uniform01(seed) * kept;
  cumulative = 0.0;
  for (int id : ids) {
    cumulative += weights[id];
    if (target < cumulative) return id;
  }
  return ids.back();
}

CaseDecl make_top_p(long long vocab) {
  CaseDecl decl;
  decl.kernel = "colibri_ops";
  decl.variant = "heap_top_p_V" + std::to_string(vocab);
  decl.shape = {{"rows", 1}, {"vocab", vocab}};
  decl.notes = "partial max-heap nucleus sampler versus full stable sort";
  decl.make = [=]() {
    auto logits = std::make_shared<std::vector<float>>(static_cast<std::size_t>(vocab));
    auto target_out = std::make_shared<int>(-1);
    auto reference = std::make_shared<int>(-1);
    Rng rng;
    for (float& value : *logits) value = 4.0f * rng.f32();
    constexpr float p = 0.9f, temperature = 0.8f;
    constexpr std::uint32_t seed = 173u;
    auto target = [=]() {
      if (quixicore_cpu::top_p_sample(logits->data(), target_out.get(), 1,
                                      vocab, p, temperature, seed) != Status::kOk) {
        throw std::runtime_error("top_p_sample failed");
      }
      do_not_optimize(target_out.get());
    };
    auto baseline = [=]() {
      *reference = sorted_top_p(*logits, p, temperature, seed);
      do_not_optimize(reference.get());
    };
    CaseBody body;
    body.target = target;
    body.baselines.emplace_back("full_stable_sort", baseline);
    body.check = [=]() {
      target(); baseline();
      CheckResult check;
      check.passed = *target_out == *reference;
      return check;
    };
    return body;
  };
  return decl;
}

CaseDecl make_threshold(long long rows, long long width, int k) {
  CaseDecl decl;
  decl.kernel = "colibri_ops";
  decl.variant = "threshold_topk_R" + std::to_string(rows) + "_W" +
                 std::to_string(width) + "_K" + std::to_string(k);
  decl.shape = {{"rows", rows}, {"width", width}, {"k", k}};
  decl.notes = "median-of-three threshold selection versus full sort";
  decl.make = [=]() {
    auto scores = std::make_shared<std::vector<float>>(static_cast<std::size_t>(rows * width));
    auto target_out = std::make_shared<std::vector<int>>(static_cast<std::size_t>(rows * k));
    auto reference = std::make_shared<std::vector<int>>(static_cast<std::size_t>(rows * k));
    Rng rng;
    for (float& value : *scores) value = rng.f32();
    auto target = [=]() {
      if (quixicore_cpu::threshold_topk_indices(
              scores->data(), target_out->data(), rows, width, k) != Status::kOk) {
        throw std::runtime_error("threshold_topk_indices failed");
      }
      do_not_optimize(target_out->data());
    };
    auto baseline = [=]() {
      std::vector<float> sorted(static_cast<std::size_t>(width));
      for (long long row = 0; row < rows; ++row) {
        std::copy_n(scores->data() + row * width, width, sorted.data());
        std::sort(sorted.begin(), sorted.end(), std::greater<float>());
        const float threshold = sorted[static_cast<std::size_t>(k - 1)];
        int count = 0;
        for (long long column = 0; column < width && count < k; ++column) {
          if ((*scores)[static_cast<std::size_t>(row * width + column)] > threshold)
            (*reference)[static_cast<std::size_t>(row * k + count++)] = int(column);
        }
        for (long long column = 0; column < width && count < k; ++column) {
          if ((*scores)[static_cast<std::size_t>(row * width + column)] == threshold)
            (*reference)[static_cast<std::size_t>(row * k + count++)] = int(column);
        }
      }
      do_not_optimize(reference->data());
    };
    CaseBody body;
    body.target = target;
    body.baselines.emplace_back("full_sort", baseline);
    body.check = [=]() {
      target(); baseline();
      CheckResult check;
      check.passed = *target_out == *reference;
      return check;
    };
    return body;
  };
  return decl;
}

struct MoeBuffers {
  std::vector<float> weights, x, target, reference;
  std::vector<std::uint8_t> packed;
  std::vector<int> experts_for_row;
  long long rows, experts, input_dim, output_dim;
  std::size_t expert_bytes;
};

CaseDecl make_moe_union(long long rows, long long experts,
                        long long input_dim, long long output_dim) {
  CaseDecl decl;
  decl.kernel = "colibri_ops";
  decl.variant = "moe_union_R" + std::to_string(rows) + "_E" +
                 std::to_string(experts) + "_K" + std::to_string(input_dim) +
                 "_N" + std::to_string(output_dim);
  decl.shape = {{"rows", rows}, {"experts", experts},
                {"input_dim", input_dim}, {"output_dim", output_dim}};
  decl.format = "q4_0";
  decl.notes = "expert-union packed projection versus one dispatched GEMV per row";
  decl.make = [=]() {
    auto b = std::make_shared<MoeBuffers>();
    b->rows = rows; b->experts = experts; b->input_dim = input_dim;
    b->output_dim = output_dim;
    b->weights.resize(static_cast<std::size_t>(experts * output_dim * input_dim));
    b->x.resize(static_cast<std::size_t>(rows * input_dim));
    b->target.resize(static_cast<std::size_t>(rows * output_dim));
    b->reference.resize(static_cast<std::size_t>(rows * output_dim));
    b->experts_for_row.resize(static_cast<std::size_t>(rows));
    if (quixicore_cpu::qgemv_packed_size(
            QuantFormat::kQ4_0, output_dim, input_dim, &b->expert_bytes) != Status::kOk) {
      throw std::runtime_error("MoE packed size failed");
    }
    b->packed.resize(static_cast<std::size_t>(experts) * b->expert_bytes);
    Rng rng;
    for (float& value : b->weights) value = 0.2f * rng.f32();
    for (float& value : b->x) value = rng.f32();
    const long long active = std::max(1LL, experts / 2);
    for (long long row = 0; row < rows; ++row) b->experts_for_row[row] = int(row % active);
    for (long long expert = 0; expert < experts; ++expert) {
      if (quixicore_cpu::qgemv_pack(
              QuantFormat::kQ4_0,
              b->weights.data() + expert * output_dim * input_dim,
              output_dim, input_dim,
              b->packed.data() + expert * b->expert_bytes) != Status::kOk) {
        throw std::runtime_error("MoE pack failed");
      }
    }
    auto target = [b]() {
      if (quixicore_cpu::moe_grouped_qgemm(
              b->x.data(), b->packed.data(), b->experts_for_row.data(),
              nullptr, b->target.data(), b->rows, b->experts, b->input_dim,
              b->output_dim, QuantFormat::kQ4_0, false) != Status::kOk) {
        throw std::runtime_error("moe_grouped_qgemm failed");
      }
      do_not_optimize(b->target.data());
    };
    auto baseline = [b]() {
      for (long long row = 0; row < b->rows; ++row) {
        if (quixicore_cpu::qgemv(
                QuantFormat::kQ4_0,
                b->packed.data() + b->experts_for_row[row] * b->expert_bytes,
                b->x.data() + row * b->input_dim,
                b->reference.data() + row * b->output_dim, b->output_dim,
                b->input_dim) != Status::kOk) {
          throw std::runtime_error("MoE row GEMV baseline failed");
        }
      }
      do_not_optimize(b->reference.data());
    };
    CaseBody body;
    body.target = target;
    body.baselines.emplace_back("per_row_qgemv", baseline);
    body.check = [b, target, baseline]() {
      target(); baseline();
      return check_arrays(b->target.data(), b->reference.data(),
                          b->rows * b->output_dim, Tolerance{2e-5, 2e-5});
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_colibri_ops_cases(const BuildCtx& ctx, std::vector<CaseDecl>& out) {
  if (ctx.preset == Preset::kSmoke) {
    out.push_back(make_w8a32(1, 16, 64));
    out.push_back(make_top_p(256));
    return;
  }
  out.push_back(make_w8a32(4, 1024, 1408));
  out.push_back(make_int8(4, 1024, 1408));
  out.push_back(make_lowbit(4, 1024, 1408, false));
  out.push_back(make_lowbit(4, 1024, 1408, true));
  out.push_back(make_top_p(65536));
  out.push_back(make_threshold(16, 8192, 2048));
  out.push_back(make_moe_union(32, 8, 256, 512));
  if (ctx.preset == Preset::kComprehensive) {
    out.push_back(make_w8a32(16, 4096, 4096));
    out.push_back(make_top_p(131072));
  }
}

}  // namespace qcb
