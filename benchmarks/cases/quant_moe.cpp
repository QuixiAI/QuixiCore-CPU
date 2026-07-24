#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness/alloc.h"
#include "harness/case.h"
#include "harness/donotopt.h"
#include "quixicore_cpu/qgemm.h"
#include "quixicore_cpu/quant_import.h"

namespace qcb {
namespace {

using quixicore_cpu::CanonicalQuantLayout;
using quixicore_cpu::CanonicalQuantTensor;
using quixicore_cpu::CpuPackedWeights;
using quixicore_cpu::FloatStorageType;
using quixicore_cpu::LinearActivation;
using quixicore_cpu::Status;

struct MoeFormat {
  CanonicalQuantLayout layout;
  long long group;
  bool scale_2d;
  const char* name;
};

struct MoeBuffers {
  std::vector<CanonicalQuantTensor> gate_tensors;
  std::vector<CanonicalQuantTensor> up_tensors;
  std::vector<CpuPackedWeights> gate;
  std::vector<CpuPackedWeights> up;
  CanonicalQuantTensor activation_tensor;
  CanonicalQuantTensor output_tensor;
  CanonicalQuantTensor reference_tensor;
  AlignedBuffer<float> input;
  AlignedBuffer<float> output;
  AlignedBuffer<float> reference;
  AlignedBuffer<float> gate_row;
  AlignedBuffer<float> up_row;
  std::vector<int> expert_ids;
  long long experts = 0;
  long long rows = 0;
  long long n = 0;
  long long k = 0;
  bool swiglu = false;
  bool dual = false;
  bool quant_swiglu = false;
  CanonicalQuantLayout output_layout = CanonicalQuantLayout::kInt8Symmetric;
  long long output_group = 0;
  bool output_scale_2d = false;
};

void require_ok(Status status, const char* message) {
  if (status != Status::kOk) throw std::runtime_error(message);
}

void run_target(MoeBuffers& b) {
  const Status status =
      b.quant_swiglu
          ? quixicore_cpu::moe_grouped_prepacked_swiglu_quantized(
                b.gate.data(), b.up.data(), b.experts,
                {b.input.get(), FloatStorageType::kF32, b.rows * b.k},
                b.expert_ids.data(), b.output_layout, b.output_group,
                &b.output_tensor, b.rows, b.output_scale_2d)
      : b.dual ? quixicore_cpu::moe_grouped_prepacked_quantized(
                     b.gate.data(), b.experts, b.activation_tensor,
                     b.expert_ids.data(), b.output.get())
      : b.swiglu
          ? quixicore_cpu::moe_grouped_prepacked_swiglu_storage(
                b.gate.data(), b.up.data(), b.experts,
                {b.input.get(), FloatStorageType::kF32, b.rows * b.k},
                b.expert_ids.data(),
                {b.output.get(), FloatStorageType::kF32, b.rows * b.n}, b.rows)
          : quixicore_cpu::moe_grouped_prepacked_storage(
                b.gate.data(), b.experts,
                {b.input.get(), FloatStorageType::kF32, b.rows * b.k},
                b.expert_ids.data(), nullptr,
                {b.output.get(), FloatStorageType::kF32, b.rows * b.n}, b.rows,
                LinearActivation::kNone);
  require_ok(status, "grouped canonical MoE failed");
  if (b.quant_swiglu)
    do_not_optimize(b.output_tensor.data.data());
  else
    do_not_optimize(b.output.get());
}

void run_baseline(MoeBuffers& b) {
  if (b.quant_swiglu) {
    require_ok(
        quixicore_cpu::moe_grouped_prepacked_swiglu_storage(
            b.gate.data(), b.up.data(), b.experts,
            {b.input.get(), FloatStorageType::kF32, b.rows * b.k},
            b.expert_ids.data(),
            {b.reference.get(), FloatStorageType::kF32, b.rows * b.n}, b.rows),
        "grouped SwiGLU baseline failed");
    require_ok(quixicore_cpu::quantize_canonical(
                   {b.reference.get(), FloatStorageType::kF32, b.rows * b.n},
                   b.rows, b.n, b.output_layout, b.output_group,
                   &b.reference_tensor, b.output_scale_2d),
               "grouped SwiGLU baseline quantization failed");
    do_not_optimize(b.reference_tensor.data.data());
    return;
  }
  if (b.dual) {
    require_ok(quixicore_cpu::dequantize_canonical(b.activation_tensor,
                                                   b.input.get(), b.rows * b.k),
               "MoE activation dequantization failed");
    require_ok(quixicore_cpu::moe_grouped_prepacked_storage(
                   b.gate.data(), b.experts,
                   {b.input.get(), FloatStorageType::kF32, b.rows * b.k},
                   b.expert_ids.data(), nullptr,
                   {b.reference.get(), FloatStorageType::kF32, b.rows * b.n},
                   b.rows, LinearActivation::kNone),
               "dequantized grouped MoE failed");
    do_not_optimize(b.reference.get());
    return;
  }
  for (long long row = 0; row < b.rows; ++row) {
    const int expert = b.expert_ids[static_cast<std::size_t>(row)];
    const quixicore_cpu::FloatStorageInput input{b.input.get() + row * b.k,
                                                 FloatStorageType::kF32, b.k};
    if (!b.swiglu) {
      require_ok(
          quixicore_cpu::qgemv_prepacked_storage(
              b.gate[expert], input,
              {b.reference.get() + row * b.n, FloatStorageType::kF32, b.n}),
          "per-token expert GEMV failed");
      continue;
    }
    require_ok(quixicore_cpu::qgemv_prepacked_storage(
                   b.gate[expert], input,
                   {b.gate_row.get(), FloatStorageType::kF32, b.n}),
               "per-token gate GEMV failed");
    require_ok(
        quixicore_cpu::qgemv_prepacked_storage(
            b.up[expert], input, {b.up_row.get(), FloatStorageType::kF32, b.n}),
        "per-token up GEMV failed");
    for (long long column = 0; column < b.n; ++column) {
      const float gate = b.gate_row.get()[column];
      b.reference.get()[row * b.n + column] =
          gate / (1.0f + std::exp(-gate)) * b.up_row.get()[column];
    }
  }
  do_not_optimize(b.reference.get());
}

CaseDecl make_moe_case(
    MoeFormat format, bool swiglu, bool sorted, long long experts,
    long long rows, long long n, long long k, bool dual = false,
    CanonicalQuantLayout activation_layout =
        CanonicalQuantLayout::kInt8Symmetric,
    long long activation_group = 32, bool activation_scale_2d = false,
    std::string pair_name = {}, bool quant_swiglu = false,
    CanonicalQuantLayout output_layout = CanonicalQuantLayout::kInt8Symmetric,
    long long output_group = 0, bool output_scale_2d = false) {
  CaseDecl decl;
  decl.kernel = "quant_moe";
  decl.variant = std::string(quant_swiglu ? "swiglu_quant_"
                             : dual       ? "dual_"
                             : swiglu     ? "swiglu_"
                                          : "projection_") +
                 (sorted ? "sorted_" : "unsorted_") + format.name + "_E" +
                 std::to_string(experts) + "_M" + std::to_string(rows) + "_N" +
                 std::to_string(n) + "_K" + std::to_string(k);
  decl.shape = {{"experts", experts}, {"m", rows}, {"n", n}, {"k", k}};
  decl.format =
      (dual || quant_swiglu) && !pair_name.empty() ? pair_name : format.name;
  decl.notes =
      quant_swiglu ? "M3 S6 grouped SwiGLU with direct activation packing"
      : dual       ? "M3 S5 direct grouped quantized-activation projection"
      : swiglu ? "M3 S6 prepared grouped expert projection with fused SwiGLU"
               : "M3 S5/S7 prepared grouped expert projection";
  decl.flops = ((swiglu || quant_swiglu) ? 4.0 : 2.0) *
               static_cast<double>(rows) * static_cast<double>(n) *
               static_cast<double>(k);
  decl.make = [=]() {
    auto b = std::make_shared<MoeBuffers>();
    b->experts = experts;
    b->rows = rows;
    b->n = n;
    b->k = k;
    b->swiglu = swiglu;
    b->dual = dual;
    b->quant_swiglu = quant_swiglu;
    b->output_layout = output_layout;
    b->output_group = output_group;
    b->output_scale_2d = output_scale_2d;
    b->gate_tensors.resize(static_cast<std::size_t>(experts));
    b->gate.resize(static_cast<std::size_t>(experts));
    if (swiglu || quant_swiglu) {
      b->up_tensors.resize(static_cast<std::size_t>(experts));
      b->up.resize(static_cast<std::size_t>(experts));
    }
    for (long long expert = 0; expert < experts; ++expert) {
      std::vector<float> source(static_cast<std::size_t>(n * k));
      for (long long item = 0; item < n * k; ++item) {
        source[static_cast<std::size_t>(item)] =
            static_cast<float>(
                static_cast<int>((item * 29 + expert * 17 + 11) % 101) - 50) /
            127.0f;
      }
      require_ok(quixicore_cpu::quantize_canonical(
                     {source.data(), FloatStorageType::kF32, n * k}, n, k,
                     format.layout, format.group, &b->gate_tensors[expert],
                     format.scale_2d),
                 "MoE gate quantization failed");
      require_ok(b->gate[expert].prepare(b->gate_tensors[expert]),
                 "MoE gate preparation failed");
      if (swiglu || quant_swiglu) {
        for (long long item = 0; item < n * k; ++item) {
          source[static_cast<std::size_t>(item)] =
              static_cast<float>(
                  static_cast<int>((item * 31 + expert * 23 + 7) % 103) - 51) /
              131.0f;
        }
        require_ok(quixicore_cpu::quantize_canonical(
                       {source.data(), FloatStorageType::kF32, n * k}, n, k,
                       format.layout, format.group, &b->up_tensors[expert],
                       format.scale_2d),
                   "MoE up quantization failed");
        require_ok(b->up[expert].prepare(b->up_tensors[expert]),
                   "MoE up preparation failed");
      }
    }
    b->input = aligned_alloc_array<float>(rows * k);
    b->output = aligned_alloc_array<float>(rows * n);
    b->reference = aligned_alloc_array<float>(rows * n);
    b->gate_row = aligned_alloc_array<float>(n);
    b->up_row = aligned_alloc_array<float>(n);
    b->expert_ids.resize(static_cast<std::size_t>(rows));
    for (long long item = 0; item < rows * k; ++item) {
      b->input.get()[item] =
          static_cast<float>(static_cast<int>((item * 37 + 13) % 97) - 48) /
          113.0f;
    }
    if (dual) {
      require_ok(quixicore_cpu::quantize_canonical(
                     {b->input.get(), FloatStorageType::kF32, rows * k}, rows,
                     k, activation_layout, activation_group,
                     &b->activation_tensor, activation_scale_2d),
                 "MoE activation quantization failed");
    }
    for (long long row = 0; row < rows; ++row) {
      b->expert_ids[static_cast<std::size_t>(row)] =
          sorted ? static_cast<int>(row * experts / rows)
                 : static_cast<int>((row * 5 + 3) % experts);
    }
    CaseBody body;
    body.target = [b]() { run_target(*b); };
    body.baselines.emplace_back(quant_swiglu ? "grouped_swiglu_then_quantize"
                                : dual ? "dequantize_activation_then_grouped"
                                       : "per_token_expert_gemv",
                                [b]() { run_baseline(*b); });
    body.check = [b]() {
      run_target(*b);
      run_baseline(*b);
      CheckResult check;
      if (b->quant_swiglu) {
        require_ok(quixicore_cpu::dequantize_canonical(
                       b->output_tensor, b->output.get(), b->rows * b->n),
                   "target activation decode failed");
        require_ok(quixicore_cpu::dequantize_canonical(
                       b->reference_tensor, b->reference.get(), b->rows * b->n),
                   "baseline activation decode failed");
      }
      for (long long item = 0; item < b->rows * b->n; ++item) {
        check_value(
            check, b->output.get()[item], b->reference.get()[item],
            b->quant_swiglu ? kQuantizedTolerance : Tolerance{2e-4, 2e-4});
      }
      return check;
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_quant_moe_cases(const BuildCtx& ctx, std::vector<CaseDecl>& out) {
  const MoeFormat formats[] = {
      {CanonicalQuantLayout::kInt4Symmetric, 128, false, "int4"},
      {CanonicalQuantLayout::kFP8E4M3FN, 128, false, "fp8_e4m3"},
      {CanonicalQuantLayout::kMXFP4E2M1E8M0, 32, false, "mxfp4"},
      {CanonicalQuantLayout::kNVFP4E2M1E4M3, 16, true, "nvfp4"},
  };
  const long long experts = ctx.preset == Preset::kSmoke ? 2 : 8;
  const long long rows = ctx.preset == Preset::kSmoke ? 8 : 128;
  const long long n = ctx.preset == Preset::kSmoke ? 64 : 512;
  const long long k = ctx.preset == Preset::kSmoke ? 128 : 1024;
  const std::size_t count = ctx.preset == Preset::kSmoke ? 1 : 4;
  for (std::size_t index = 0; index < count; ++index) {
    out.push_back(
        make_moe_case(formats[index], false, true, experts, rows, n, k));
    out.push_back(
        make_moe_case(formats[index], false, false, experts, rows, n, k));
    out.push_back(
        make_moe_case(formats[index], true, true, experts, rows, n, k));
  }
  if (ctx.preset != Preset::kSmoke) {
    out.push_back(make_moe_case(formats[0], false, true, experts, rows, n, k,
                                true, CanonicalQuantLayout::kInt8Symmetric, 128,
                                false, "w4a8"));
    const MoeFormat dual_fp8{CanonicalQuantLayout::kFP8E4M3FN, 32, false,
                             "fp8_e4m3"};
    out.push_back(make_moe_case(dual_fp8, false, true, experts, rows, n, k,
                                true, CanonicalQuantLayout::kFP8E4M3FN, 32,
                                false, "fp8_e4m3_e4m3"));
    const MoeFormat mxfp8{CanonicalQuantLayout::kMXFP8E4M3E8M0, 32, false,
                          "mxfp8"};
    out.push_back(make_moe_case(mxfp8, false, true, experts, rows, n, k, true,
                                CanonicalQuantLayout::kMXFP8E4M3E8M0, 32, false,
                                "mxfp8_mxfp8"));
    out.push_back(make_moe_case(formats[2], false, true, experts, rows, n, k,
                                true, CanonicalQuantLayout::kMXFP4E2M1E8M0, 32,
                                false, "mxfp4_mxfp4"));
    out.push_back(make_moe_case(formats[3], false, true, experts, rows, n, k,
                                true, CanonicalQuantLayout::kNVFP4E2M1E4M3, 16,
                                true, "nvfp4_nvfp4"));
    for (std::size_t index = 0; index < 4; ++index) {
      out.push_back(
          make_moe_case(formats[index], false, true, experts, rows, n, k, false,
                        CanonicalQuantLayout::kInt8Symmetric, 32, false,
                        std::string(formats[index].name) + "_a8", true,
                        CanonicalQuantLayout::kInt8Symmetric, 128));
    }
  }
}

}  // namespace qcb
