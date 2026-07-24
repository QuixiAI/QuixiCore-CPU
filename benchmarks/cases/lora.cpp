// Direct F16-adapter LoRA versus an independent two-stage scalar composition.

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness/case.h"
#include "harness/donotopt.h"
#include "quixicore_cpu/float_storage.h"

namespace qcb {
namespace {

using quixicore_cpu::FloatStorageInput;
using quixicore_cpu::FloatStorageOutput;
using quixicore_cpu::FloatStorageType;
using quixicore_cpu::Status;

struct LoraBuffers {
  std::vector<std::uint16_t> x, a, b, base, out, reference, low;
  quixicore_cpu::FloatStorageWorkspace workspace;
  long long rows = 0, input_dim = 0, output_dim = 0, rank = 0;
};

void lora_reference(LoraBuffers& buffers) {
  for (long long row = 0; row < buffers.rows; ++row) {
    for (long long rank = 0; rank < buffers.rank; ++rank) {
      float sum = 0.0f;
      for (long long input = 0; input < buffers.input_dim; ++input) {
        sum += quixicore_cpu::bf16_to_float(
                   buffers.x[row * buffers.input_dim + input]) *
               quixicore_cpu::f16_to_float(
                   buffers.a[rank * buffers.input_dim + input]);
      }
      buffers.low[row * buffers.rank + rank] =
          quixicore_cpu::float_to_f16(sum);
    }
    for (long long output = 0; output < buffers.output_dim; ++output) {
      float sum = 0.0f;
      for (long long rank = 0; rank < buffers.rank; ++rank) {
        sum += quixicore_cpu::f16_to_float(
                   buffers.low[row * buffers.rank + rank]) *
               quixicore_cpu::f16_to_float(
                   buffers.b[output * buffers.rank + rank]);
      }
      const float delta = quixicore_cpu::f16_to_float(
          quixicore_cpu::float_to_f16(sum));
      const long long index = row * buffers.output_dim + output;
      const float value = quixicore_cpu::bf16_to_float(buffers.base[index]) +
                          0.75f * delta;
      buffers.reference[index] = quixicore_cpu::float_to_bf16(value);
    }
  }
}

CaseDecl make_lora(long long rows, long long input_dim, long long output_dim,
                   long long rank) {
  CaseDecl decl;
  decl.kernel = "lora";
  decl.variant = "direct_M" + std::to_string(rows) + "_K" +
                 std::to_string(input_dim) + "_N" +
                 std::to_string(output_dim) + "_R" + std::to_string(rank);
  decl.shape = {{"M", rows}, {"K", input_dim}, {"N", output_dim},
                {"R", rank}};
  decl.dtype = "bf16";
  decl.format = "f16_adapter";
  decl.notes = "two-stage FP16-rounded LoRA with fused scale/base add";
  decl.flops = 2.0 * rows * rank * (input_dim + output_dim);
  decl.bytes_moved = 2.0 *
                     (rows * input_dim + rank * input_dim +
                      output_dim * rank + 2 * rows * output_dim);
  decl.make = [=]() {
    auto buffers = std::make_shared<LoraBuffers>();
    buffers->rows = rows;
    buffers->input_dim = input_dim;
    buffers->output_dim = output_dim;
    buffers->rank = rank;
    buffers->x.resize(rows * input_dim);
    buffers->a.resize(rank * input_dim);
    buffers->b.resize(output_dim * rank);
    buffers->base.resize(rows * output_dim);
    buffers->out.resize(rows * output_dim);
    buffers->reference.resize(rows * output_dim);
    buffers->low.resize(rows * rank);
    for (std::size_t i = 0; i < buffers->x.size(); ++i) {
      buffers->x[i] = quixicore_cpu::float_to_bf16(
          0.15f * std::sin(0.013f * static_cast<float>(i + 1)));
    }
    for (std::size_t i = 0; i < buffers->a.size(); ++i) {
      buffers->a[i] = quixicore_cpu::float_to_f16(
          0.10f * std::sin(0.007f * static_cast<float>(i + 3)));
    }
    for (std::size_t i = 0; i < buffers->b.size(); ++i) {
      buffers->b[i] = quixicore_cpu::float_to_f16(
          0.10f * std::cos(0.011f * static_cast<float>(i + 5)));
    }
    for (std::size_t i = 0; i < buffers->base.size(); ++i) {
      buffers->base[i] = quixicore_cpu::float_to_bf16(
          0.20f * std::cos(0.017f * static_cast<float>(i + 7)));
    }
    const std::size_t scratch = static_cast<std::size_t>(
        rows * input_dim + 2 * rows * output_dim);
    if (buffers->workspace.reserve(scratch) != Status::kOk) {
      throw std::runtime_error("LoRA workspace allocation failed");
    }
    CaseBody body;
    body.target = [buffers]() {
      if (quixicore_cpu::lora_apply_direct_f16_storage(
              FloatStorageInput{buffers->x.data(), FloatStorageType::kBF16,
                                static_cast<long long>(buffers->x.size())},
              buffers->a.data(), buffers->b.data(),
              FloatStorageInput{
                  buffers->base.data(), FloatStorageType::kBF16,
                  static_cast<long long>(buffers->base.size())},
              FloatStorageOutput{
                  buffers->out.data(), FloatStorageType::kBF16,
                  static_cast<long long>(buffers->out.size())},
              buffers->rows, buffers->input_dim, buffers->output_dim,
              buffers->rank, 0.75f, &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("LoRA target failed");
      }
      do_not_optimize(buffers->out.data());
    };
    body.baselines.emplace_back("scalar_f16_matmul_pair", [buffers]() {
      lora_reference(*buffers);
      do_not_optimize(buffers->reference.data());
    });
    body.check = [buffers]() {
      if (quixicore_cpu::lora_apply_direct_f16_storage(
              FloatStorageInput{buffers->x.data(), FloatStorageType::kBF16,
                                static_cast<long long>(buffers->x.size())},
              buffers->a.data(), buffers->b.data(),
              FloatStorageInput{
                  buffers->base.data(), FloatStorageType::kBF16,
                  static_cast<long long>(buffers->base.size())},
              FloatStorageOutput{
                  buffers->out.data(), FloatStorageType::kBF16,
                  static_cast<long long>(buffers->out.size())},
              buffers->rows, buffers->input_dim, buffers->output_dim,
              buffers->rank, 0.75f, &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("LoRA check failed");
      }
      lora_reference(*buffers);
      CheckResult check;
      for (std::size_t i = 0; i < buffers->out.size(); ++i) {
        check_value(check, quixicore_cpu::bf16_to_float(buffers->out[i]),
                    quixicore_cpu::bf16_to_float(buffers->reference[i]),
                    {0.03, 0.03});
      }
      return check;
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_lora_cases(const BuildCtx& ctx, std::vector<CaseDecl>& out) {
  if (ctx.preset == Preset::kSmoke) {
    out.push_back(make_lora(1, 512, 512, 8));
    return;
  }
  out.push_back(make_lora(1, 4096, 4096, 16));
  out.push_back(make_lora(4, 4096, 4096, 16));
  out.push_back(make_lora(8, 4096, 4096, 16));
  out.push_back(make_lora(64, 4096, 4096, 16));
  if (ctx.preset == Preset::kComprehensive) {
    out.push_back(make_lora(1, 4096, 4096, 8));
    out.push_back(make_lora(1, 4096, 4096, 32));
    out.push_back(make_lora(1, 4096, 4096, 64));
    out.push_back(make_lora(1, 4096, 4096, 128));
    out.push_back(make_lora(64, 4096, 4096, 64));
    out.push_back(make_lora(512, 4096, 4096, 16));
  }
}

}  // namespace qcb
