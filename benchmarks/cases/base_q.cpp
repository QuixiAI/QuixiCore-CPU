#include "quixicore_cpu/base_q.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness/alloc.h"
#include "harness/case.h"
#include "harness/donotopt.h"
#include "quixicore_cpu/float_storage.h"

namespace qcb {
namespace {

using quixicore_cpu::Status;

struct BaseQBuffers {
  std::vector<std::uint8_t> codes;
  std::vector<std::uint16_t> scales;
  std::vector<std::uint16_t> biases;
  std::vector<float> dequantized;
  AlignedBuffer<float> input;
  AlignedBuffer<float> output;
  AlignedBuffer<float> baseline;
  AlignedBuffer<float> reference;
  quixicore_cpu::BaseQTensorView view;
  long long m = 0;
  long long n = 0;
  long long k = 0;
  std::vector<int> tokens;
  std::vector<int> reference_tokens;
  std::vector<int> expert_of_tile;
};

void insert_code(std::vector<std::uint8_t>& bytes, std::size_t row_offset,
                 long long column, int bits, std::uint32_t code) {
  const int bit = static_cast<int>(column * bits);
  const std::size_t byte = row_offset + static_cast<std::size_t>(bit >> 3);
  const int shift = bit & 7;
  const std::uint16_t word = static_cast<std::uint16_t>(code << shift);
  bytes[byte] |= static_cast<std::uint8_t>(word);
  if (shift + bits > 8) bytes[byte + 1] |= static_cast<std::uint8_t>(word >> 8);
}

CaseDecl base_q_projection(int bits, long long m, long long n, long long k) {
  CaseDecl decl;
  decl.kernel = "base_q";
  decl.variant = "q" + std::to_string(bits) + "_M" + std::to_string(m) + "_N" +
                 std::to_string(n) + "_K" + std::to_string(k);
  decl.shape = {{"m", m}, {"n", n}, {"k", k}};
  decl.dtype = "f32";
  decl.format = "base_q" + std::to_string(bits);
  decl.notes = "Metal BaseQN sibling contract, BF16 affine group scales";
  decl.flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
               static_cast<double>(k);
  decl.make = [bits, m, n, k]() {
    auto buffers = std::make_shared<BaseQBuffers>();
    buffers->m = m;
    buffers->n = n;
    buffers->k = k;
    const int group_size = bits <= 3 ? 32 : (bits == 8 ? 128 : 64);
    const std::size_t row_bytes = static_cast<std::size_t>(k * bits / 8);
    buffers->codes.assign(static_cast<std::size_t>(n) * row_bytes, 0);
    const long long groups_per_row = k / group_size;
    const std::size_t groups = static_cast<std::size_t>(n * groups_per_row);
    buffers->scales.resize(groups);
    buffers->biases.resize(groups);
    buffers->dequantized.resize(static_cast<std::size_t>(n * k));
    for (std::size_t group = 0; group < groups; ++group) {
      buffers->scales[group] = quixicore_cpu::float_to_bf16(
          0.015625f * static_cast<float>((group % 7) + 1));
      buffers->biases[group] = quixicore_cpu::float_to_bf16(
          -0.125f + 0.015625f * static_cast<float>(group % 9));
    }
    const std::uint32_t mask = (1u << bits) - 1u;
    for (long long row = 0; row < n; ++row) {
      for (long long column = 0; column < k; ++column) {
        const std::uint32_t code =
            static_cast<std::uint32_t>((row * 19 + column * 5 + 3) & mask);
        insert_code(buffers->codes, static_cast<std::size_t>(row) * row_bytes,
                    column, bits, code);
        const std::size_t group = static_cast<std::size_t>(
            row * groups_per_row + column / group_size);
        buffers->dequantized[static_cast<std::size_t>(row * k + column)] =
            static_cast<float>(code) *
                quixicore_cpu::bf16_to_float(buffers->scales[group]) +
            quixicore_cpu::bf16_to_float(buffers->biases[group]);
      }
    }
    buffers->view = {buffers->codes.data(),
                     buffers->codes.size(),
                     buffers->scales.data(),
                     groups,
                     buffers->biases.data(),
                     groups,
                     n,
                     k,
                     bits,
                     group_size,
                     quixicore_cpu::BaseQScaleType::kBF16,
                     false};
    buffers->input = aligned_alloc_array<float>(m * k);
    buffers->output = aligned_alloc_array<float>(m * n);
    buffers->baseline = aligned_alloc_array<float>(m * n);
    buffers->reference = aligned_alloc_array<float>(m * n);
    for (long long index = 0; index < m * k; ++index) {
      buffers->input.get()[index] =
          static_cast<float>(static_cast<int>((index * 13 + 7) % 41) - 20) /
          23.0f;
    }
    for (long long input_row = 0; input_row < m; ++input_row) {
      for (long long weight_row = 0; weight_row < n; ++weight_row) {
        double sum = 0.0;
        for (long long column = 0; column < k; ++column) {
          sum +=
              static_cast<double>(buffers->dequantized[static_cast<std::size_t>(
                  weight_row * k + column)]) *
              buffers->input.get()[input_row * k + column];
        }
        buffers->reference.get()[input_row * n + weight_row] =
            static_cast<float>(sum);
      }
    }
    auto target = [buffers]() {
      if (quixicore_cpu::base_q_gemm(
              buffers->view,
              {buffers->input.get(), quixicore_cpu::FloatStorageType::kF32,
               buffers->m * buffers->k},
              {buffers->output.get(), quixicore_cpu::FloatStorageType::kF32,
               buffers->m * buffers->n},
              buffers->m) != Status::kOk) {
        throw std::runtime_error("BaseQN projection failed");
      }
      do_not_optimize(buffers->output.get());
    };
    auto baseline = [buffers]() {
      for (long long input_row = 0; input_row < buffers->m; ++input_row) {
        for (long long weight_row = 0; weight_row < buffers->n; ++weight_row) {
          float sums[4] = {};
          for (long long column = 0; column < buffers->k; ++column) {
            sums[column & 3] +=
                buffers->dequantized[static_cast<std::size_t>(
                    weight_row * buffers->k + column)] *
                buffers->input.get()[input_row * buffers->k + column];
          }
          buffers->baseline.get()[input_row * buffers->n + weight_row] =
              (sums[0] + sums[1]) + (sums[2] + sums[3]);
        }
      }
      do_not_optimize(buffers->baseline.get());
    };
    CaseBody body;
    body.target = target;
    body.baselines.emplace_back("dequantized_scalar_gemm", baseline);
    body.check = [buffers, target]() {
      target();
      CheckResult check;
      for (long long index = 0; index < buffers->m * buffers->n; ++index) {
        check_value(check, buffers->output.get()[index],
                    buffers->reference.get()[index], Tolerance{2e-4, 2e-4});
      }
      return check;
    };
    return body;
  };
  return decl;
}

CaseDecl base_q_lm_head(int bits, long long batch, long long vocab,
                        long long hidden) {
  CaseDecl decl;
  decl.kernel = "base_q";
  decl.variant = "lm_head_argmax_q" + std::to_string(bits) + "_B" +
                 std::to_string(batch) + "_V" + std::to_string(vocab) + "_K" +
                 std::to_string(hidden);
  decl.shape = {{"batch", batch}, {"vocab", vocab}, {"hidden", hidden}};
  decl.dtype = "f32";
  decl.format = "base_q" + std::to_string(bits);
  decl.notes = "packed BaseQN greedy LM head; ties choose the lower token";
  decl.flops = 2.0 * static_cast<double>(batch) * static_cast<double>(vocab) *
               static_cast<double>(hidden);
  decl.make = [bits, batch, vocab, hidden]() {
    auto buffers = std::make_shared<BaseQBuffers>();
    buffers->m = batch;
    buffers->n = vocab;
    buffers->k = hidden;
    const int group_size = bits <= 3 ? 32 : (bits == 8 ? 128 : 64);
    const std::size_t row_bytes = static_cast<std::size_t>(hidden * bits / 8);
    buffers->codes.assign(static_cast<std::size_t>(vocab) * row_bytes, 0);
    const long long groups_per_row = hidden / group_size;
    const std::size_t groups = static_cast<std::size_t>(vocab * groups_per_row);
    buffers->scales.resize(groups);
    buffers->biases.resize(groups);
    for (std::size_t group = 0; group < groups; ++group) {
      buffers->scales[group] = quixicore_cpu::float_to_bf16(
          0.015625f * static_cast<float>((group % 7) + 1));
      buffers->biases[group] = quixicore_cpu::float_to_bf16(
          -0.125f + 0.015625f * static_cast<float>(group % 9));
    }
    const std::uint32_t mask = (1u << bits) - 1u;
    for (long long row = 0; row < vocab; ++row) {
      for (long long column = 0; column < hidden; ++column) {
        const std::uint32_t code =
            static_cast<std::uint32_t>((row * 19 + column * 5 + 3) & mask);
        insert_code(buffers->codes, static_cast<std::size_t>(row) * row_bytes,
                    column, bits, code);
      }
    }
    buffers->view = {buffers->codes.data(),
                     buffers->codes.size(),
                     buffers->scales.data(),
                     groups,
                     buffers->biases.data(),
                     groups,
                     vocab,
                     hidden,
                     bits,
                     group_size,
                     quixicore_cpu::BaseQScaleType::kBF16,
                     false};
    buffers->input = aligned_alloc_array<float>(batch * hidden);
    buffers->output = aligned_alloc_array<float>(batch * vocab);
    buffers->tokens.resize(static_cast<std::size_t>(batch));
    buffers->reference_tokens.resize(static_cast<std::size_t>(batch));
    for (long long index = 0; index < batch * hidden; ++index) {
      buffers->input.get()[index] =
          static_cast<float>(static_cast<int>((index * 13 + 7) % 41) - 20) /
          23.0f;
    }
    auto target = [buffers]() {
      if (quixicore_cpu::base_q_lm_head_argmax(
              buffers->view,
              {buffers->input.get(), quixicore_cpu::FloatStorageType::kF32,
               buffers->m * buffers->k},
              buffers->tokens.data(), buffers->m) != Status::kOk) {
        throw std::runtime_error("BaseQN LM-head argmax failed");
      }
      do_not_optimize(buffers->tokens.data());
    };
    auto materialized = [buffers]() {
      if (quixicore_cpu::base_q_gemm(
              buffers->view,
              {buffers->input.get(), quixicore_cpu::FloatStorageType::kF32,
               buffers->m * buffers->k},
              {buffers->output.get(), quixicore_cpu::FloatStorageType::kF32,
               buffers->m * buffers->n},
              buffers->m) != Status::kOk) {
        throw std::runtime_error("BaseQN materialized LM head failed");
      }
      for (long long input_row = 0; input_row < buffers->m; ++input_row) {
        int best = 0;
        for (long long token = 1; token < buffers->n; ++token) {
          if (buffers->output.get()[input_row * buffers->n + token] >
              buffers->output.get()[input_row * buffers->n + best]) {
            best = static_cast<int>(token);
          }
        }
        buffers->reference_tokens[static_cast<std::size_t>(input_row)] = best;
      }
      do_not_optimize(buffers->reference_tokens.data());
    };
    CaseBody body;
    body.target = target;
    body.baselines.emplace_back("base_q_gemm_plus_argmax", materialized);
    body.check = [buffers, target, materialized]() {
      target();
      materialized();
      CheckResult check;
      check.passed = buffers->tokens == buffers->reference_tokens;
      return check;
    };
    return body;
  };
  return decl;
}

CaseDecl base_q_moe(int bits, bool swiglu, long long experts,
                    long long total_rows, long long hidden,
                    long long output_rows) {
  CaseDecl decl;
  decl.kernel = "base_q";
  decl.variant = std::string(swiglu ? "moe_swiglu_q" : "moe_gemm_q") +
                 std::to_string(bits) + "_E" + std::to_string(experts) + "_R" +
                 std::to_string(total_rows) + "_K" + std::to_string(hidden) +
                 (swiglu ? "_I" : "_N") + std::to_string(output_rows);
  decl.shape = {{"experts", experts},
                {"rows", total_rows},
                {"hidden", hidden},
                {swiglu ? "intermediate" : "output", output_rows}};
  decl.dtype = "f32";
  decl.format = "base_q" + std::to_string(bits);
  decl.notes = swiglu ? "direct BaseQN grouped expert gate/up plus SwiGLU"
                      : "direct BaseQN grouped expert projection";
  decl.flops = (swiglu ? 4.0 : 2.0) * static_cast<double>(total_rows) *
               static_cast<double>(hidden) * static_cast<double>(output_rows);
  decl.make = [=]() {
    auto buffers = std::make_shared<BaseQBuffers>();
    buffers->m = total_rows;
    buffers->n = output_rows;
    buffers->k = hidden;
    const int group_size = bits <= 3 ? 32 : (bits == 8 ? 128 : 64);
    const long long packed_rows = (swiglu ? 2 : 1) * output_rows;
    const long long weight_rows = experts * packed_rows;
    const std::size_t row_bytes = static_cast<std::size_t>(hidden * bits / 8);
    buffers->codes.assign(static_cast<std::size_t>(weight_rows) * row_bytes, 0);
    const long long groups_per_row = hidden / group_size;
    const std::size_t groups =
        static_cast<std::size_t>(weight_rows * groups_per_row);
    buffers->scales.resize(groups);
    buffers->biases.resize(groups);
    buffers->dequantized.resize(static_cast<std::size_t>(weight_rows * hidden));
    for (std::size_t group = 0; group < groups; ++group) {
      buffers->scales[group] = quixicore_cpu::float_to_bf16(
          0.00390625f * static_cast<float>((group % 7) + 1));
      buffers->biases[group] = quixicore_cpu::float_to_bf16(
          -0.03125f + 0.00390625f * static_cast<float>(group % 9));
    }
    const std::uint32_t mask = (1u << bits) - 1u;
    for (long long row = 0; row < weight_rows; ++row) {
      for (long long column = 0; column < hidden; ++column) {
        const std::uint32_t code =
            static_cast<std::uint32_t>((row * 19 + column * 5 + 3) & mask);
        insert_code(buffers->codes, static_cast<std::size_t>(row) * row_bytes,
                    column, bits, code);
        const std::size_t group = static_cast<std::size_t>(
            row * groups_per_row + column / group_size);
        buffers->dequantized[static_cast<std::size_t>(row * hidden + column)] =
            static_cast<float>(code) *
                quixicore_cpu::bf16_to_float(buffers->scales[group]) +
            quixicore_cpu::bf16_to_float(buffers->biases[group]);
      }
    }
    buffers->view = {buffers->codes.data(),
                     buffers->codes.size(),
                     buffers->scales.data(),
                     groups,
                     buffers->biases.data(),
                     groups,
                     weight_rows,
                     hidden,
                     bits,
                     group_size,
                     quixicore_cpu::BaseQScaleType::kBF16,
                     false};
    buffers->input = aligned_alloc_array<float>(total_rows * hidden);
    buffers->output = aligned_alloc_array<float>(total_rows * output_rows);
    buffers->baseline = aligned_alloc_array<float>(total_rows * output_rows);
    buffers->expert_of_tile.resize(static_cast<std::size_t>(total_rows / 32));
    for (long long tile = 0; tile < total_rows / 32; ++tile)
      buffers->expert_of_tile[static_cast<std::size_t>(tile)] =
          static_cast<int>(tile % experts);
    for (long long index = 0; index < total_rows * hidden; ++index) {
      buffers->input.get()[index] =
          static_cast<float>(static_cast<int>((index * 13 + 7) % 41) - 20) /
          47.0f;
    }
    auto target = [buffers, experts, swiglu]() {
      const auto input = quixicore_cpu::FloatStorageInput{
          buffers->input.get(), quixicore_cpu::FloatStorageType::kF32,
          buffers->m * buffers->k};
      const auto output = quixicore_cpu::FloatStorageOutput{
          buffers->output.get(), quixicore_cpu::FloatStorageType::kF32,
          buffers->m * buffers->n};
      const Status status =
          swiglu
              ? quixicore_cpu::base_q_moe_swiglu(buffers->view, experts, input,
                                                 buffers->expert_of_tile.data(),
                                                 buffers->m, output)
              : quixicore_cpu::base_q_moe_gemm(buffers->view, experts, input,
                                               buffers->expert_of_tile.data(),
                                               buffers->m, output);
      if (status != Status::kOk)
        throw std::runtime_error("BaseQN grouped expert projection failed");
      do_not_optimize(buffers->output.get());
    };
    auto baseline = [buffers, swiglu]() {
      const long long packed_rows = (swiglu ? 2 : 1) * buffers->n;
      for (long long row = 0; row < buffers->m; ++row) {
        const long long expert =
            buffers->expert_of_tile[static_cast<std::size_t>(row / 32)];
        for (long long output_row = 0; output_row < buffers->n; ++output_row) {
          float gate_sums[4] = {};
          float up_sums[4] = {};
          const long long gate_row = expert * packed_rows + output_row;
          const long long up_row = gate_row + buffers->n;
          for (long long column = 0; column < buffers->k; ++column) {
            const float activation =
                buffers->input.get()[row * buffers->k + column];
            gate_sums[column & 3] +=
                activation * buffers->dequantized[static_cast<std::size_t>(
                                 gate_row * buffers->k + column)];
            if (swiglu) {
              up_sums[column & 3] +=
                  activation * buffers->dequantized[static_cast<std::size_t>(
                                   up_row * buffers->k + column)];
            }
          }
          const float gate =
              (gate_sums[0] + gate_sums[1]) + (gate_sums[2] + gate_sums[3]);
          buffers->baseline.get()[row * buffers->n + output_row] =
              swiglu
                  ? gate / (1.0f + std::exp(-gate)) *
                        ((up_sums[0] + up_sums[1]) + (up_sums[2] + up_sums[3]))
                  : gate;
        }
      }
      do_not_optimize(buffers->baseline.get());
    };
    CaseBody body;
    body.target = target;
    body.baselines.emplace_back("dequantized_grouped_reference", baseline);
    body.check = [buffers, target, baseline]() {
      target();
      baseline();
      CheckResult check;
      for (long long index = 0; index < buffers->m * buffers->n; ++index) {
        check_value(check, buffers->output.get()[index],
                    buffers->baseline.get()[index], Tolerance{3e-4, 3e-4});
      }
      return check;
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_base_q_cases(const BuildCtx& ctx, std::vector<CaseDecl>& out) {
  const long long n = ctx.preset == Preset::kSmoke
                          ? 16
                          : (ctx.preset == Preset::kQuick ? 512 : 4096);
  const long long k = ctx.preset == Preset::kSmoke
                          ? 128
                          : (ctx.preset == Preset::kQuick ? 1024 : 4096);
  for (int bits : {2, 3, 4, 5, 6, 8})
    out.push_back(base_q_projection(bits, 1, n, k));
  if (ctx.preset != Preset::kSmoke) {
    for (int bits : {2, 3, 4, 5, 6, 8})
      out.push_back(base_q_projection(bits, 16, n, k));
    out.push_back(base_q_lm_head(4, 1, 4096, 1024));
    out.push_back(base_q_lm_head(4, 4, 4096, 1024));
    out.push_back(base_q_lm_head(6, 1, 4096, 1024));
  } else {
    out.push_back(base_q_lm_head(4, 2, 257, 128));
  }
  if (ctx.preset == Preset::kSmoke) {
    out.push_back(base_q_moe(4, false, 2, 32, 128, 32));
    out.push_back(base_q_moe(4, true, 2, 32, 128, 32));
  } else {
    out.push_back(base_q_moe(4, false, 4, 128, 1024, 512));
    out.push_back(base_q_moe(4, true, 4, 128, 1024, 512));
  }
}

}  // namespace qcb
