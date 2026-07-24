// M0 pass-0 registrations for the full quantization matrix. These entries
// reuse already-correct reference cases until later milestones replace each
// row with a format-specialized kernel. Relabeling keeps the families
// executable without duplicating setup or changing kernel semantics.

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "harness/alloc.h"
#include "harness/case.h"
#include "harness/donotopt.h"
#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/qgemm.h"
#include "quixicore_cpu/qgemv.h"
#include "quixicore_cpu/quant_import.h"
#include "quixicore_cpu/quantization.h"

namespace qcb {

void build_qgemv_formats_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_lifecycle_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_ported_ops_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_prerequisites_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_optimization_plan_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_norm_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_embedding_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_lm_head_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_moe_cases(const BuildCtx&, std::vector<CaseDecl>&);

namespace {

using quixicore_cpu::Status;

bool matches(const CaseDecl& decl,
             std::initializer_list<std::string_view> needles) {
  for (std::string_view needle : needles) {
    if (decl.variant.find(needle) != std::string::npos ||
        decl.format.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

void append_cases(const BuildCtx& ctx, CaseBuilder builder, const char* family,
                  std::initializer_list<std::string_view> needles,
                  std::vector<CaseDecl>& out) {
  std::vector<CaseDecl> source;
  builder(ctx, source);
  for (CaseDecl& decl : source) {
    if (!matches(decl, needles)) continue;
    decl.variant = decl.kernel + "_" + decl.variant;
    decl.kernel = family;
    decl.notes = "M0 pass-0 existing reference: " + decl.notes;
    out.push_back(std::move(decl));
  }
}

void append_family_cases(const BuildCtx& ctx, CaseBuilder builder,
                         const char* family, std::vector<CaseDecl>& out) {
  std::vector<CaseDecl> source;
  builder(ctx, source);
  for (CaseDecl& decl : source) {
    decl.variant = decl.kernel + "_" + decl.variant;
    decl.kernel = family;
    out.push_back(std::move(decl));
  }
}

struct BitNetBuffers {
  AlignedBuffer<float> weights;
  AlignedBuffer<float> dequantized;
  AlignedBuffer<float> activation;
  AlignedBuffer<float> output;
  AlignedBuffer<float> reference;
  AlignedBuffer<std::uint8_t> packed;
  long long n = 0;
  long long k = 0;
};

struct CanonicalProjectionBuffers {
  quixicore_cpu::CanonicalQuantTensor tensor;
  quixicore_cpu::CpuPackedWeights prepared;
  AlignedBuffer<float> activation;
  AlignedBuffer<std::uint16_t> activation_16;
  AlignedBuffer<float> output;
  AlignedBuffer<float> baseline;
  AlignedBuffer<float> reference;
  std::vector<float> dequantized;
  long long m = 0;
  long long n = 0;
  long long k = 0;
  quixicore_cpu::FloatStorageType input_type =
      quixicore_cpu::FloatStorageType::kF32;
};

struct CanonicalDualProjectionBuffers {
  quixicore_cpu::CanonicalQuantTensor weight_tensor;
  quixicore_cpu::CanonicalQuantTensor activation_tensor;
  quixicore_cpu::CpuPackedWeights prepared;
  AlignedBuffer<float> output;
  AlignedBuffer<float> baseline;
  AlignedBuffer<float> reference;
  std::vector<float> weights;
  std::vector<float> activations;
  long long m = 0;
  long long n = 0;
  long long k = 0;
};

struct CanonicalEpilogueBuffers {
  quixicore_cpu::CanonicalQuantTensor tensor;
  quixicore_cpu::CpuPackedWeights prepared;
  AlignedBuffer<float> activation;
  AlignedBuffer<float> bias;
  AlignedBuffer<float> output;
  AlignedBuffer<float> composed;
  AlignedBuffer<float> reference;
  AlignedBuffer<std::uint16_t> output_16;
  AlignedBuffer<std::uint16_t> composed_16;
  std::vector<float> dequantized;
  long long m = 0;
  long long n = 0;
  long long k = 0;
  quixicore_cpu::FloatStorageType output_type =
      quixicore_cpu::FloatStorageType::kF32;
};

struct CanonicalGateUpBuffers {
  quixicore_cpu::CanonicalQuantTensor gate_tensor;
  quixicore_cpu::CanonicalQuantTensor up_tensor;
  quixicore_cpu::CanonicalQuantTensor quantized_swiglu;
  quixicore_cpu::CanonicalQuantTensor composed_quantized_swiglu;
  quixicore_cpu::CpuPackedWeights gate_prepared;
  quixicore_cpu::CpuPackedWeights up_prepared;
  quixicore_cpu::Workspace workspace;
  AlignedBuffer<float> activation;
  AlignedBuffer<float> gate;
  AlignedBuffer<float> up;
  AlignedBuffer<float> composed_gate;
  AlignedBuffer<float> composed_up;
  AlignedBuffer<float> swiglu;
  AlignedBuffer<float> composed_swiglu;
  AlignedBuffer<float> reference_gate;
  AlignedBuffer<float> reference_up;
  AlignedBuffer<float> reference_swiglu;
  std::vector<float> gate_weights;
  std::vector<float> up_weights;
  long long m = 0;
  long long n = 0;
  long long k = 0;
};

struct CanonicalQKVBuffers {
  quixicore_cpu::CanonicalQuantTensor tensors[3];
  quixicore_cpu::CpuPackedWeights prepared[3];
  AlignedBuffer<float> activation;
  AlignedBuffer<std::uint16_t> activation_16;
  AlignedBuffer<float> output[3];
  AlignedBuffer<float> baseline[3];
  long long m = 0;
  long long rows[3] = {};
  long long k = 0;
  quixicore_cpu::FloatStorageType input_type =
      quixicore_cpu::FloatStorageType::kF32;
};

struct CanonicalQKVRopeBuffers {
  quixicore_cpu::CanonicalQuantTensor tensors[3];
  quixicore_cpu::CpuPackedWeights prepared[3];
  AlignedBuffer<float> activation;
  AlignedBuffer<std::uint16_t> activation_16;
  AlignedBuffer<float> q;
  AlignedBuffer<float> key_cache;
  AlignedBuffer<float> value_cache;
  AlignedBuffer<float> raw_q;
  AlignedBuffer<float> raw_k;
  AlignedBuffer<float> raw_v;
  AlignedBuffer<float> baseline_q;
  AlignedBuffer<float> baseline_key;
  AlignedBuffer<float> baseline_value;
  std::vector<float> cosine;
  std::vector<float> sine;
  long long query_heads = 0;
  long long kv_heads = 0;
  long long head_dim = 0;
  long long k = 0;
  long long slots = 0;
  long long max_position = 0;
  int position = 0;
  int slot = 0;
  quixicore_cpu::FloatStorageType input_type =
      quixicore_cpu::FloatStorageType::kF32;
};

struct ProjectionFormat {
  quixicore_cpu::CanonicalQuantLayout layout;
  long long group_size;
  const char* name;
};

struct DualProjectionFormat {
  ProjectionFormat weight;
  ProjectionFormat activation;
  const char* name;
};

CaseDecl canonical_projection(ProjectionFormat format, long long m, long long n,
                              long long k,
                              quixicore_cpu::FloatStorageType input_type =
                                  quixicore_cpu::FloatStorageType::kF32) {
  CaseDecl decl;
  decl.kernel = m == 1 ? "quant_gemv_matrix" : "quant_gemm_matrix";
  decl.variant = "canonical_panel_" + std::string(format.name) + "_M" +
                 std::to_string(m) + "_N" + std::to_string(n) + "_K" +
                 std::to_string(k);
  if (input_type == quixicore_cpu::FloatStorageType::kF16)
    decl.variant += "_f16";
  else if (input_type == quixicore_cpu::FloatStorageType::kBF16)
    decl.variant += "_bf16";
  decl.shape = {{"m", m}, {"n", n}, {"k", k}};
  decl.format = format.name;
  decl.notes =
      input_type == quixicore_cpu::FloatStorageType::kF32
          ? (m == 1 ? "M2 canonical row GEMV with format block dots"
                    : "M2 canonical row-panel projection with M32 reuse tile")
          : (m == 1 ? "M2 canonical typed-activation row GEMV"
                    : "M2 canonical typed-activation row-panel GEMM");
  decl.flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
               static_cast<double>(k);
  decl.make = [format, m, n, k, input_type]() {
    auto buffers = std::make_shared<CanonicalProjectionBuffers>();
    buffers->m = m;
    buffers->n = n;
    buffers->k = k;
    buffers->input_type = input_type;
    std::vector<float> weights(static_cast<std::size_t>(n * k));
    for (long long index = 0; index < n * k; ++index) {
      weights[static_cast<std::size_t>(index)] =
          static_cast<float>(static_cast<int>((index * 17 + 11) % 61) - 30) /
          19.0f;
    }
    if (quixicore_cpu::quantize_canonical(
            {weights.data(), quixicore_cpu::FloatStorageType::kF32, n * k}, n,
            k, format.layout, format.group_size, &buffers->tensor,
            format.layout ==
                quixicore_cpu::CanonicalQuantLayout::kNVFP4E2M1E4M3) !=
        Status::kOk) {
      throw std::runtime_error("canonical projection packing failed");
    }
    if (buffers->prepared.prepare(buffers->tensor) != Status::kOk) {
      throw std::runtime_error("canonical projection preparation failed");
    }
    buffers->dequantized.resize(static_cast<std::size_t>(n * k));
    if (quixicore_cpu::dequantize_canonical(buffers->tensor,
                                            buffers->dequantized.data(),
                                            n * k) != Status::kOk) {
      throw std::runtime_error("canonical projection decode failed");
    }
    buffers->activation = aligned_alloc_array<float>(m * k);
    if (input_type != quixicore_cpu::FloatStorageType::kF32)
      buffers->activation_16 = aligned_alloc_array<std::uint16_t>(m * k);
    buffers->output = aligned_alloc_array<float>(m * n);
    buffers->baseline = aligned_alloc_array<float>(m * n);
    buffers->reference = aligned_alloc_array<float>(m * n);
    for (long long index = 0; index < m * k; ++index) {
      const float value =
          static_cast<float>(static_cast<int>((index * 29 + 7) % 67) - 33) /
          23.0f;
      if (input_type == quixicore_cpu::FloatStorageType::kF16) {
        const std::uint16_t bits = quixicore_cpu::float_to_f16(value);
        buffers->activation_16.get()[index] = bits;
        buffers->activation.get()[index] = quixicore_cpu::f16_to_float(bits);
      } else if (input_type == quixicore_cpu::FloatStorageType::kBF16) {
        const std::uint16_t bits = quixicore_cpu::float_to_bf16(value);
        buffers->activation_16.get()[index] = bits;
        buffers->activation.get()[index] = quixicore_cpu::bf16_to_float(bits);
      } else {
        buffers->activation.get()[index] = value;
      }
    }
    for (long long row = 0; row < m; ++row) {
      for (long long output = 0; output < n; ++output) {
        double sum = 0.0;
        for (long long input = 0; input < k; ++input) {
          sum += static_cast<double>(buffers->dequantized[output * k + input]) *
                 buffers->activation.get()[row * k + input];
        }
        buffers->reference.get()[row * n + output] = static_cast<float>(sum);
      }
    }
    auto target = [buffers]() {
      const void* input =
          buffers->input_type == quixicore_cpu::FloatStorageType::kF32
              ? static_cast<const void*>(buffers->activation.get())
              : static_cast<const void*>(buffers->activation_16.get());
      if (quixicore_cpu::qgemm_prepacked_storage(
              buffers->prepared,
              {input, buffers->input_type, buffers->m * buffers->k},
              {buffers->output.get(), quixicore_cpu::FloatStorageType::kF32,
               buffers->m * buffers->n},
              buffers->m) != Status::kOk) {
        throw std::runtime_error("canonical panel projection failed");
      }
      do_not_optimize(buffers->output.get());
    };
    auto baseline = [buffers]() {
      const long long m_local = buffers->m;
      const long long n_local = buffers->n;
      const long long k_local = buffers->k;
      const float* weights_local = buffers->dequantized.data();
      const float* activation_local = buffers->activation.get();
      float* baseline_local = buffers->baseline.get();
      for (long long row = 0; row < m_local; ++row) {
        for (long long output = 0; output < n_local; ++output) {
          float sum = 0.0f;
          for (long long input = 0; input < k_local; ++input) {
            sum += weights_local[output * k_local + input] *
                   activation_local[row * k_local + input];
          }
          baseline_local[row * n_local + output] = sum;
        }
      }
      do_not_optimize(baseline_local);
    };
    CaseBody body;
    body.target = target;
    body.baselines.emplace_back("dequantized_scalar_gemm", baseline);
    body.check = [buffers, target]() {
      target();
      CheckResult check;
      for (long long index = 0; index < buffers->m * buffers->n; ++index) {
        // The candidate uses four independent FP32 reduction chains while the
        // oracle is accumulated in FP64. This is still 300x tighter than the
        // umbrella quantized tolerance and catches decoder/scale errors
        // without requiring a serial FP64 hot path.
        check_value(check, buffers->output.get()[index],
                    buffers->reference.get()[index], Tolerance{1e-4, 1e-4});
      }
      return check;
    };
    return body;
  };
  return decl;
}

void append_canonical_projection_cases(const BuildCtx& ctx, long long m,
                                       std::vector<CaseDecl>& out) {
  const ProjectionFormat formats[] = {
      {quixicore_cpu::CanonicalQuantLayout::kInt4Symmetric, 32, "int4"},
      {quixicore_cpu::CanonicalQuantLayout::kUInt4Affine, 32, "uint4_affine"},
      {quixicore_cpu::CanonicalQuantLayout::kInt8Symmetric, 32, "int8"},
      {quixicore_cpu::CanonicalQuantLayout::kInt8Affine, 0, "int8_affine"},
      {quixicore_cpu::CanonicalQuantLayout::kFP8E4M3FN, 32, "fp8_e4m3"},
      {quixicore_cpu::CanonicalQuantLayout::kFP8E5M2, 32, "fp8_e5m2"},
      {quixicore_cpu::CanonicalQuantLayout::kFP4E2M1, 32, "fp4_e2m1"},
      {quixicore_cpu::CanonicalQuantLayout::kMXFP8E4M3E8M0, 32, "mxfp8"},
      {quixicore_cpu::CanonicalQuantLayout::kMXFP4E2M1E8M0, 32, "mxfp4"},
      {quixicore_cpu::CanonicalQuantLayout::kNVFP4E2M1E4M3, 16, "nvfp4"},
      {quixicore_cpu::CanonicalQuantLayout::kBitNetTernary, 32, "bitnet"},
  };
  const long long n = ctx.preset == Preset::kSmoke
                          ? 33
                          : (ctx.preset == Preset::kQuick ? 512 : 4096);
  const long long k = ctx.preset == Preset::kSmoke
                          ? 32
                          : (ctx.preset == Preset::kQuick ? 1024 : 4096);
  const std::size_t count =
      ctx.preset == Preset::kSmoke
          ? 1
          : (ctx.preset == Preset::kComprehensive && m != 1 ? 4 : 11);
  for (std::size_t index = 0; index < count; ++index) {
    out.push_back(canonical_projection(formats[index], m, n, k));
  }
}

void append_canonical_typed_projection_cases(const BuildCtx& ctx, long long m,
                                             std::vector<CaseDecl>& out) {
  const ProjectionFormat formats[] = {
      {quixicore_cpu::CanonicalQuantLayout::kInt4Symmetric, 32, "int4"},
      {quixicore_cpu::CanonicalQuantLayout::kUInt4Affine, 32, "uint4_affine"},
      {quixicore_cpu::CanonicalQuantLayout::kInt8Symmetric, 32, "int8"},
      {quixicore_cpu::CanonicalQuantLayout::kInt8Affine, 0, "int8_affine"},
      {quixicore_cpu::CanonicalQuantLayout::kFP8E4M3FN, 32, "fp8_e4m3"},
      {quixicore_cpu::CanonicalQuantLayout::kFP8E5M2, 32, "fp8_e5m2"},
      {quixicore_cpu::CanonicalQuantLayout::kFP4E2M1, 32, "fp4_e2m1"},
      {quixicore_cpu::CanonicalQuantLayout::kMXFP8E4M3E8M0, 32, "mxfp8"},
      {quixicore_cpu::CanonicalQuantLayout::kMXFP4E2M1E8M0, 32, "mxfp4"},
      {quixicore_cpu::CanonicalQuantLayout::kNVFP4E2M1E4M3, 16, "nvfp4"},
      {quixicore_cpu::CanonicalQuantLayout::kBitNetTernary, 32, "bitnet"},
  };
  const long long n = ctx.preset == Preset::kSmoke
                          ? 33
                          : (ctx.preset == Preset::kQuick ? 512 : 4096);
  const long long k = ctx.preset == Preset::kSmoke
                          ? 32
                          : (ctx.preset == Preset::kQuick ? 1024 : 4096);
  const std::size_t count =
      ctx.preset == Preset::kSmoke
          ? 1
          : (m == 128 || (ctx.preset == Preset::kComprehensive && m != 1) ? 4
                                                                          : 11);
  const bool representative_set = count == 4;
  constexpr std::size_t representative_indices[] = {0, 4, 8, 10};
  for (quixicore_cpu::FloatStorageType input_type :
       {quixicore_cpu::FloatStorageType::kF16,
        quixicore_cpu::FloatStorageType::kBF16}) {
    for (std::size_t slot = 0; slot < count; ++slot) {
      const std::size_t index =
          representative_set ? representative_indices[slot] : slot;
      out.push_back(canonical_projection(formats[index], m, n, k, input_type));
    }
  }
}

float benchmark_activation(float value,
                           quixicore_cpu::LinearActivation activation) {
  if (activation == quixicore_cpu::LinearActivation::kSilu)
    return value / (1.0f + std::exp(-value));
  const float positive = value > 0.0f ? value : 0.0f;
  return positive * positive;
}

CaseDecl canonical_epilogue(ProjectionFormat format, long long m, long long n,
                            long long k,
                            quixicore_cpu::LinearActivation activation,
                            const char* activation_name,
                            quixicore_cpu::FloatStorageType output_type =
                                quixicore_cpu::FloatStorageType::kF32) {
  CaseDecl decl;
  decl.kernel = "quant_fusions";
  decl.variant = "canonical_epilogue_" + std::string(format.name) + "_" +
                 activation_name + "_M" + std::to_string(m) + "_N" +
                 std::to_string(n) + "_K" + std::to_string(k);
  if (output_type == quixicore_cpu::FloatStorageType::kF16)
    decl.variant += "_out_f16";
  else if (output_type == quixicore_cpu::FloatStorageType::kBF16)
    decl.variant += "_out_bf16";
  decl.shape = {{"m", m}, {"n", n}, {"k", k}};
  decl.format = format.name;
  decl.notes = "M3 canonical projection with fused bias and activation store";
  decl.flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
               static_cast<double>(k);
  decl.make = [format, m, n, k, activation, output_type]() {
    auto buffers = std::make_shared<CanonicalEpilogueBuffers>();
    buffers->m = m;
    buffers->n = n;
    buffers->k = k;
    buffers->output_type = output_type;
    std::vector<float> weights(static_cast<std::size_t>(n * k));
    for (long long index = 0; index < n * k; ++index) {
      weights[static_cast<std::size_t>(index)] =
          static_cast<float>(static_cast<int>((index * 17 + 11) % 61) - 30) /
          19.0f;
    }
    if (quixicore_cpu::quantize_canonical(
            {weights.data(), quixicore_cpu::FloatStorageType::kF32, n * k}, n,
            k, format.layout, format.group_size, &buffers->tensor,
            format.layout ==
                quixicore_cpu::CanonicalQuantLayout::kNVFP4E2M1E4M3) !=
            Status::kOk ||
        buffers->prepared.prepare(buffers->tensor) != Status::kOk) {
      throw std::runtime_error("canonical epilogue preparation failed");
    }
    buffers->dequantized.resize(static_cast<std::size_t>(n * k));
    if (quixicore_cpu::dequantize_canonical(buffers->tensor,
                                            buffers->dequantized.data(),
                                            n * k) != Status::kOk) {
      throw std::runtime_error("canonical epilogue decode failed");
    }
    buffers->activation = aligned_alloc_array<float>(m * k);
    buffers->bias = aligned_alloc_array<float>(n);
    buffers->output = aligned_alloc_array<float>(m * n);
    buffers->composed = aligned_alloc_array<float>(m * n);
    buffers->reference = aligned_alloc_array<float>(m * n);
    if (output_type != quixicore_cpu::FloatStorageType::kF32) {
      buffers->output_16 = aligned_alloc_array<std::uint16_t>(m * n);
      buffers->composed_16 = aligned_alloc_array<std::uint16_t>(m * n);
    }
    for (long long index = 0; index < m * k; ++index) {
      buffers->activation.get()[index] =
          static_cast<float>(static_cast<int>((index * 29 + 7) % 67) - 33) /
          23.0f;
    }
    for (long long output = 0; output < n; ++output) {
      buffers->bias.get()[output] =
          static_cast<float>(static_cast<int>((output * 13 + 5) % 29) - 14) /
          37.0f;
    }
    for (long long row = 0; row < m; ++row) {
      for (long long output = 0; output < n; ++output) {
        double sum = 0.0;
        for (long long input = 0; input < k; ++input) {
          sum += static_cast<double>(buffers->dequantized[output * k + input]) *
                 buffers->activation.get()[row * k + input];
        }
        buffers->reference.get()[row * n + output] = benchmark_activation(
            static_cast<float>(sum) + buffers->bias.get()[output], activation);
      }
    }
    auto target = [buffers, activation]() {
      void* output =
          buffers->output_type == quixicore_cpu::FloatStorageType::kF32
              ? static_cast<void*>(buffers->output.get())
              : static_cast<void*>(buffers->output_16.get());
      if (quixicore_cpu::qgemm_prepacked_epilogue_storage(
              buffers->prepared,
              {buffers->activation.get(), quixicore_cpu::FloatStorageType::kF32,
               buffers->m * buffers->k},
              buffers->bias.get(),
              {output, buffers->output_type, buffers->m * buffers->n},
              buffers->m, activation) != Status::kOk) {
        throw std::runtime_error("canonical fused epilogue failed");
      }
      do_not_optimize(output);
    };
    auto composed = [buffers, activation]() {
      if (quixicore_cpu::qgemm_prepacked_storage(
              buffers->prepared,
              {buffers->activation.get(), quixicore_cpu::FloatStorageType::kF32,
               buffers->m * buffers->k},
              {buffers->composed.get(), quixicore_cpu::FloatStorageType::kF32,
               buffers->m * buffers->n},
              buffers->m) != Status::kOk) {
        throw std::runtime_error("canonical composed projection failed");
      }
      for (long long row = 0; row < buffers->m; ++row) {
        for (long long output = 0; output < buffers->n; ++output) {
          const long long index = row * buffers->n + output;
          buffers->composed.get()[index] = benchmark_activation(
              buffers->composed.get()[index] + buffers->bias.get()[output],
              activation);
          if (buffers->output_type == quixicore_cpu::FloatStorageType::kF16) {
            buffers->composed_16.get()[index] =
                quixicore_cpu::float_to_f16(buffers->composed.get()[index]);
          } else if (buffers->output_type ==
                     quixicore_cpu::FloatStorageType::kBF16) {
            buffers->composed_16.get()[index] =
                quixicore_cpu::float_to_bf16(buffers->composed.get()[index]);
          }
        }
      }
      if (buffers->output_type == quixicore_cpu::FloatStorageType::kF32)
        do_not_optimize(buffers->composed.get());
      else
        do_not_optimize(buffers->composed_16.get());
    };
    CaseBody body;
    body.target = target;
    body.baselines.emplace_back("projection_then_bias_activation", composed);
    body.check = [buffers, target]() {
      target();
      CheckResult check;
      for (long long index = 0; index < buffers->m * buffers->n; ++index) {
        float actual = buffers->output.get()[index];
        if (buffers->output_type == quixicore_cpu::FloatStorageType::kF16) {
          actual = quixicore_cpu::f16_to_float(buffers->output_16.get()[index]);
        } else if (buffers->output_type ==
                   quixicore_cpu::FloatStorageType::kBF16) {
          actual =
              quixicore_cpu::bf16_to_float(buffers->output_16.get()[index]);
        }
        check_value(check, actual, buffers->reference.get()[index],
                    Tolerance{2e-2, 2e-2});
      }
      return check;
    };
    return body;
  };
  return decl;
}

void append_canonical_epilogue_cases(const BuildCtx& ctx, long long m,
                                     std::vector<CaseDecl>& out) {
  const ProjectionFormat formats[] = {
      {quixicore_cpu::CanonicalQuantLayout::kInt4Symmetric, 32, "int4"},
      {quixicore_cpu::CanonicalQuantLayout::kFP8E4M3FN, 32, "fp8_e4m3"},
      {quixicore_cpu::CanonicalQuantLayout::kMXFP4E2M1E8M0, 32, "mxfp4"},
      {quixicore_cpu::CanonicalQuantLayout::kBitNetTernary, 32, "bitnet"},
  };
  const long long n = ctx.preset == Preset::kSmoke
                          ? 33
                          : (ctx.preset == Preset::kQuick ? 512 : 4096);
  const long long k = ctx.preset == Preset::kSmoke
                          ? 32
                          : (ctx.preset == Preset::kQuick ? 1024 : 4096);
  const std::size_t count = ctx.preset == Preset::kSmoke ? 1 : 4;
  for (std::size_t index = 0; index < count; ++index) {
    out.push_back(canonical_epilogue(formats[index], m, n, k,
                                     quixicore_cpu::LinearActivation::kSilu,
                                     "silu"));
    out.push_back(canonical_epilogue(formats[index], m, n, k,
                                     quixicore_cpu::LinearActivation::kRelu2,
                                     "relu2"));
    if (index == 0 || index == 2) {
      out.push_back(canonical_epilogue(
          formats[index], m, n, k, quixicore_cpu::LinearActivation::kSilu,
          "silu", quixicore_cpu::FloatStorageType::kF16));
      out.push_back(canonical_epilogue(
          formats[index], m, n, k, quixicore_cpu::LinearActivation::kSilu,
          "silu", quixicore_cpu::FloatStorageType::kBF16));
    }
  }
}

CaseDecl canonical_gate_up(ProjectionFormat format, long long m, long long n,
                           long long k, const char* family,
                           bool fuse_swiglu = false,
                           bool quantize_swiglu = false) {
  CaseDecl decl;
  decl.kernel = family;
  decl.variant =
      std::string(quantize_swiglu ? "canonical_swiglu_quant_a8_"
                                  : (fuse_swiglu ? "canonical_swiglu_"
                                                 : "canonical_gate_up_")) +
      format.name + "_M" + std::to_string(m) + "_N" + std::to_string(n) + "_K" +
      std::to_string(k);
  decl.shape = {{"m", m}, {"n", n}, {"k", k}};
  decl.format = format.name;
  decl.notes =
      quantize_swiglu
          ? "M3 F3 direct paired projection, SwiGLU, and group-A8 pack"
          : (fuse_swiglu
                 ? "M3 F3 paired projection with direct SwiGLU store"
                 : "M3 F2 paired prepared projection under one CPU schedule");
  decl.flops = 4.0 * static_cast<double>(m) * static_cast<double>(n) *
               static_cast<double>(k);
  decl.make = [format, m, n, k, fuse_swiglu, quantize_swiglu]() {
    auto buffers = std::make_shared<CanonicalGateUpBuffers>();
    buffers->m = m;
    buffers->n = n;
    buffers->k = k;
    std::vector<float> gate_source(static_cast<std::size_t>(n * k));
    std::vector<float> up_source(static_cast<std::size_t>(n * k));
    for (long long index = 0; index < n * k; ++index) {
      gate_source[static_cast<std::size_t>(index)] =
          static_cast<float>(static_cast<int>((index * 17 + 11) % 61) - 30) /
          19.0f;
      up_source[static_cast<std::size_t>(index)] =
          static_cast<float>(static_cast<int>((index * 23 + 5) % 67) - 33) /
          21.0f;
    }
    const bool scale_2d =
        format.layout == quixicore_cpu::CanonicalQuantLayout::kNVFP4E2M1E4M3;
    if (quixicore_cpu::quantize_canonical(
            {gate_source.data(), quixicore_cpu::FloatStorageType::kF32, n * k},
            n, k, format.layout, format.group_size, &buffers->gate_tensor,
            scale_2d) != Status::kOk ||
        quixicore_cpu::quantize_canonical(
            {up_source.data(), quixicore_cpu::FloatStorageType::kF32, n * k}, n,
            k, format.layout, format.group_size, &buffers->up_tensor,
            scale_2d) != Status::kOk ||
        buffers->gate_prepared.prepare(buffers->gate_tensor) != Status::kOk ||
        buffers->up_prepared.prepare(buffers->up_tensor) != Status::kOk) {
      throw std::runtime_error("canonical gate/up preparation failed");
    }
    buffers->gate_weights.resize(static_cast<std::size_t>(n * k));
    buffers->up_weights.resize(static_cast<std::size_t>(n * k));
    if (quixicore_cpu::dequantize_canonical(buffers->gate_tensor,
                                            buffers->gate_weights.data(),
                                            n * k) != Status::kOk ||
        quixicore_cpu::dequantize_canonical(buffers->up_tensor,
                                            buffers->up_weights.data(),
                                            n * k) != Status::kOk) {
      throw std::runtime_error("canonical gate/up decode failed");
    }
    buffers->activation = aligned_alloc_array<float>(m * k);
    buffers->gate = aligned_alloc_array<float>(m * n);
    buffers->up = aligned_alloc_array<float>(m * n);
    buffers->composed_gate = aligned_alloc_array<float>(m * n);
    buffers->composed_up = aligned_alloc_array<float>(m * n);
    buffers->swiglu = aligned_alloc_array<float>(m * n);
    buffers->composed_swiglu = aligned_alloc_array<float>(m * n);
    buffers->reference_gate = aligned_alloc_array<float>(m * n);
    buffers->reference_up = aligned_alloc_array<float>(m * n);
    buffers->reference_swiglu = aligned_alloc_array<float>(m * n);
    for (long long index = 0; index < m * k; ++index) {
      buffers->activation.get()[index] =
          static_cast<float>(static_cast<int>((index * 29 + 7) % 67) - 33) /
          23.0f;
    }
    for (long long row = 0; row < m; ++row) {
      for (long long output = 0; output < n; ++output) {
        double gate_sum = 0.0;
        double up_sum = 0.0;
        for (long long input = 0; input < k; ++input) {
          const float activation = buffers->activation.get()[row * k + input];
          gate_sum +=
              static_cast<double>(buffers->gate_weights[output * k + input]) *
              activation;
          up_sum +=
              static_cast<double>(buffers->up_weights[output * k + input]) *
              activation;
        }
        buffers->reference_gate.get()[row * n + output] =
            static_cast<float>(gate_sum);
        buffers->reference_up.get()[row * n + output] =
            static_cast<float>(up_sum);
        const float gate = static_cast<float>(gate_sum);
        buffers->reference_swiglu.get()[row * n + output] =
            gate / (1.0f + std::exp(-gate)) * static_cast<float>(up_sum);
      }
    }
    auto target = [buffers, fuse_swiglu, quantize_swiglu]() {
      const quixicore_cpu::FloatStorageInput input{
          buffers->activation.get(), quixicore_cpu::FloatStorageType::kF32,
          buffers->m * buffers->k};
      const Status status =
          quantize_swiglu
              ? quixicore_cpu::qgemm_prepacked_swiglu_quantized(
                    buffers->gate_prepared, buffers->up_prepared, input,
                    quixicore_cpu::CanonicalQuantLayout::kInt8Symmetric, 32,
                    &buffers->quantized_swiglu, buffers->m, false,
                    &buffers->workspace)
              : (fuse_swiglu
                     ? quixicore_cpu::qgemm_prepacked_swiglu_storage(
                           buffers->gate_prepared, buffers->up_prepared, input,
                           {buffers->swiglu.get(),
                            quixicore_cpu::FloatStorageType::kF32,
                            buffers->m * buffers->n},
                           buffers->m)
                     : quixicore_cpu::qgemm_prepacked_gate_up_storage(
                           buffers->gate_prepared, buffers->up_prepared, input,
                           {buffers->gate.get(),
                            quixicore_cpu::FloatStorageType::kF32,
                            buffers->m * buffers->n},
                           {buffers->up.get(),
                            quixicore_cpu::FloatStorageType::kF32,
                            buffers->m * buffers->n},
                           buffers->m));
      if (status != Status::kOk) {
        throw std::runtime_error("canonical paired projection failed");
      }
      if (quantize_swiglu)
        do_not_optimize(buffers->quantized_swiglu.data.data());
      else if (fuse_swiglu)
        do_not_optimize(buffers->swiglu.get());
      else {
        do_not_optimize(buffers->gate.get());
        do_not_optimize(buffers->up.get());
      }
    };
    auto composed = [buffers, fuse_swiglu, quantize_swiglu]() {
      const quixicore_cpu::FloatStorageInput input{
          buffers->activation.get(), quixicore_cpu::FloatStorageType::kF32,
          buffers->m * buffers->k};
      const quixicore_cpu::FloatStorageOutput gate_output{
          buffers->composed_gate.get(), quixicore_cpu::FloatStorageType::kF32,
          buffers->m * buffers->n};
      const quixicore_cpu::FloatStorageOutput up_output{
          buffers->composed_up.get(), quixicore_cpu::FloatStorageType::kF32,
          buffers->m * buffers->n};
      const Status status =
          quantize_swiglu
              ? quixicore_cpu::qgemm_prepacked_swiglu_storage(
                    buffers->gate_prepared, buffers->up_prepared, input,
                    {buffers->composed_swiglu.get(),
                     quixicore_cpu::FloatStorageType::kF32,
                     buffers->m * buffers->n},
                    buffers->m)
              : (fuse_swiglu
                     ? quixicore_cpu::qgemm_prepacked_gate_up_storage(
                           buffers->gate_prepared, buffers->up_prepared, input,
                           gate_output, up_output, buffers->m)
                     : (quixicore_cpu::qgemm_prepacked_storage(
                            buffers->gate_prepared, input, gate_output,
                            buffers->m) == Status::kOk &&
                                quixicore_cpu::qgemm_prepacked_storage(
                                    buffers->up_prepared, input, up_output,
                                    buffers->m) == Status::kOk
                            ? Status::kOk
                            : Status::kInvalidArgument));
      if (status != Status::kOk) {
        throw std::runtime_error("canonical composed gate/up failed");
      }
      if (fuse_swiglu) {
        for (long long index = 0; index < buffers->m * buffers->n; ++index) {
          const float gate = buffers->composed_gate.get()[index];
          buffers->composed_swiglu.get()[index] =
              gate / (1.0f + std::exp(-gate)) *
              buffers->composed_up.get()[index];
        }
        do_not_optimize(buffers->composed_swiglu.get());
      } else if (quantize_swiglu) {
        if (quixicore_cpu::quantize_canonical(
                {buffers->composed_swiglu.get(),
                 quixicore_cpu::FloatStorageType::kF32,
                 buffers->m * buffers->n},
                buffers->m, buffers->n,
                quixicore_cpu::CanonicalQuantLayout::kInt8Symmetric, 32,
                &buffers->composed_quantized_swiglu) != Status::kOk) {
          throw std::runtime_error(
              "canonical composed SwiGLU quantization failed");
        }
        do_not_optimize(buffers->composed_quantized_swiglu.data.data());
      } else {
        do_not_optimize(buffers->composed_gate.get());
        do_not_optimize(buffers->composed_up.get());
      }
    };
    CaseBody body;
    body.target = target;
    body.baselines.emplace_back(
        quantize_swiglu ? "fused_swiglu_then_group_a8"
                        : (fuse_swiglu ? "paired_projection_then_swiglu"
                                       : "two_prepared_projections"),
        composed);
    body.check = [buffers, target, composed, fuse_swiglu, quantize_swiglu]() {
      target();
      CheckResult check;
      if (quantize_swiglu) {
        composed();
        std::vector<float> decoded(
            static_cast<std::size_t>(buffers->m * buffers->n));
        std::vector<float> reference(
            static_cast<std::size_t>(buffers->m * buffers->n));
        if (quixicore_cpu::dequantize_canonical(
                buffers->quantized_swiglu, decoded.data(),
                buffers->m * buffers->n) != Status::kOk ||
            quixicore_cpu::dequantize_canonical(
                buffers->composed_quantized_swiglu, reference.data(),
                buffers->m * buffers->n) != Status::kOk) {
          check.passed = false;
          return check;
        }
        for (long long index = 0; index < buffers->m * buffers->n; ++index) {
          check_value(check, decoded[static_cast<std::size_t>(index)],
                      reference[static_cast<std::size_t>(index)],
                      Tolerance{3e-2, 3e-2});
        }
        return check;
      }
      for (long long index = 0; index < buffers->m * buffers->n; ++index) {
        if (fuse_swiglu) {
          check_value(check, buffers->swiglu.get()[index],
                      buffers->reference_swiglu.get()[index],
                      Tolerance{2e-2, 2e-2});
        } else {
          check_value(check, buffers->gate.get()[index],
                      buffers->reference_gate.get()[index],
                      Tolerance{1e-4, 1e-4});
          check_value(check, buffers->up.get()[index],
                      buffers->reference_up.get()[index],
                      Tolerance{1e-4, 1e-4});
        }
      }
      return check;
    };
    return body;
  };
  return decl;
}

void append_canonical_gate_up_cases(const BuildCtx& ctx, const char* family,
                                    std::vector<CaseDecl>& out) {
  const ProjectionFormat formats[] = {
      {quixicore_cpu::CanonicalQuantLayout::kInt4Symmetric, 32, "int4"},
      {quixicore_cpu::CanonicalQuantLayout::kFP8E4M3FN, 32, "fp8_e4m3"},
      {quixicore_cpu::CanonicalQuantLayout::kMXFP4E2M1E8M0, 32, "mxfp4"},
      {quixicore_cpu::CanonicalQuantLayout::kBitNetTernary, 32, "bitnet"},
  };
  const long long n = ctx.preset == Preset::kSmoke
                          ? 33
                          : (ctx.preset == Preset::kQuick ? 512 : 4096);
  const long long k = ctx.preset == Preset::kSmoke
                          ? 32
                          : (ctx.preset == Preset::kQuick ? 1024 : 4096);
  const std::size_t count = ctx.preset == Preset::kSmoke ? 1 : 4;
  for (long long m : {1LL, 16LL, 128LL}) {
    for (std::size_t index = 0; index < count; ++index)
      out.push_back(canonical_gate_up(formats[index], m, n, k, family));
  }
}

CaseDecl canonical_qkv(ProjectionFormat format, long long m, long long nq,
                       long long nk, long long nv, long long k,
                       const char* family,
                       quixicore_cpu::FloatStorageType input_type =
                           quixicore_cpu::FloatStorageType::kF32) {
  CaseDecl decl;
  decl.kernel = family;
  decl.variant = "canonical_qkv_" + std::string(format.name) + "_M" +
                 std::to_string(m) + "_Nq" + std::to_string(nq) + "_Nk" +
                 std::to_string(nk) + "_Nv" + std::to_string(nv) + "_K" +
                 std::to_string(k);
  if (input_type == quixicore_cpu::FloatStorageType::kF16)
    decl.variant += "_f16";
  else if (input_type == quixicore_cpu::FloatStorageType::kBF16)
    decl.variant += "_bf16";
  decl.shape = {{"m", m}, {"nq", nq}, {"nk", nk}, {"nv", nv}, {"k", k}};
  decl.format = format.name;
  decl.dtype =
      input_type == quixicore_cpu::FloatStorageType::kF16
          ? "f16"
          : (input_type == quixicore_cpu::FloatStorageType::kBF16 ? "bf16"
                                                                  : "f32");
  decl.notes =
      "M3 F4 unequal-head Q/K/V prepared projection under one CPU schedule";
  decl.flops = 2.0 * static_cast<double>(m) *
               static_cast<double>(nq + nk + nv) * static_cast<double>(k);
  decl.make = [format, m, nq, nk, nv, k, input_type]() {
    auto buffers = std::make_shared<CanonicalQKVBuffers>();
    buffers->m = m;
    buffers->rows[0] = nq;
    buffers->rows[1] = nk;
    buffers->rows[2] = nv;
    buffers->k = k;
    buffers->input_type = input_type;
    const bool scale_2d =
        format.layout == quixicore_cpu::CanonicalQuantLayout::kNVFP4E2M1E4M3;
    for (int plane = 0; plane < 3; ++plane) {
      const long long rows = buffers->rows[plane];
      std::vector<float> source(static_cast<std::size_t>(rows * k));
      for (long long index = 0; index < rows * k; ++index) {
        source[static_cast<std::size_t>(index)] =
            static_cast<float>(
                static_cast<int>((index * (17 + plane * 6) + 11 + plane * 5) %
                                 67) -
                33) /
            static_cast<float>(19 + plane * 2);
      }
      if (quixicore_cpu::quantize_canonical(
              {source.data(), quixicore_cpu::FloatStorageType::kF32, rows * k},
              rows, k, format.layout, format.group_size,
              &buffers->tensors[plane], scale_2d) != Status::kOk ||
          buffers->prepared[plane].prepare(buffers->tensors[plane]) !=
              Status::kOk) {
        throw std::runtime_error("canonical Q/K/V preparation failed");
      }
      buffers->output[plane] = aligned_alloc_array<float>(m * rows);
      buffers->baseline[plane] = aligned_alloc_array<float>(m * rows);
    }
    buffers->activation = aligned_alloc_array<float>(m * k);
    if (input_type != quixicore_cpu::FloatStorageType::kF32)
      buffers->activation_16 = aligned_alloc_array<std::uint16_t>(m * k);
    for (long long index = 0; index < m * k; ++index) {
      const float value =
          static_cast<float>(static_cast<int>((index * 29 + 7) % 71) - 35) /
          23.0f;
      buffers->activation.get()[index] = value;
      if (input_type == quixicore_cpu::FloatStorageType::kF16) {
        buffers->activation_16.get()[index] =
            quixicore_cpu::float_to_f16(value);
      } else if (input_type == quixicore_cpu::FloatStorageType::kBF16) {
        buffers->activation_16.get()[index] =
            quixicore_cpu::float_to_bf16(value);
      }
    }
    auto target = [buffers]() {
      const quixicore_cpu::FloatStorageInput input{
          buffers->input_type == quixicore_cpu::FloatStorageType::kF32
              ? static_cast<const void*>(buffers->activation.get())
              : static_cast<const void*>(buffers->activation_16.get()),
          buffers->input_type, buffers->m * buffers->k};
      const Status status = quixicore_cpu::qgemm_prepacked_qkv_storage(
          buffers->prepared[0], buffers->prepared[1], buffers->prepared[2],
          input,
          {buffers->output[0].get(), quixicore_cpu::FloatStorageType::kF32,
           buffers->m * buffers->rows[0]},
          {buffers->output[1].get(), quixicore_cpu::FloatStorageType::kF32,
           buffers->m * buffers->rows[1]},
          {buffers->output[2].get(), quixicore_cpu::FloatStorageType::kF32,
           buffers->m * buffers->rows[2]},
          buffers->m);
      if (status != Status::kOk)
        throw std::runtime_error("canonical Q/K/V projection failed");
      for (int plane = 0; plane < 3; ++plane)
        do_not_optimize(buffers->output[plane].get());
    };
    auto composed = [buffers]() {
      const quixicore_cpu::FloatStorageInput input{
          buffers->input_type == quixicore_cpu::FloatStorageType::kF32
              ? static_cast<const void*>(buffers->activation.get())
              : static_cast<const void*>(buffers->activation_16.get()),
          buffers->input_type, buffers->m * buffers->k};
      for (int plane = 0; plane < 3; ++plane) {
        if (quixicore_cpu::qgemm_prepacked_storage(
                buffers->prepared[plane], input,
                {buffers->baseline[plane].get(),
                 quixicore_cpu::FloatStorageType::kF32,
                 buffers->m * buffers->rows[plane]},
                buffers->m) != Status::kOk) {
          throw std::runtime_error("composed Q/K/V projection failed");
        }
        do_not_optimize(buffers->baseline[plane].get());
      }
    };
    CaseBody body;
    body.target = target;
    body.baselines.emplace_back("three_prepared_projections", composed);
    body.check = [buffers, target, composed]() {
      target();
      composed();
      CheckResult check;
      for (int plane = 0; plane < 3; ++plane) {
        for (long long index = 0; index < buffers->m * buffers->rows[plane];
             ++index) {
          check_value(check, buffers->output[plane].get()[index],
                      buffers->baseline[plane].get()[index],
                      Tolerance{1e-4, 1e-4});
        }
      }
      return check;
    };
    return body;
  };
  return decl;
}

void append_canonical_qkv_cases(const BuildCtx& ctx, const char* family,
                                std::vector<CaseDecl>& out) {
  const ProjectionFormat formats[] = {
      {quixicore_cpu::CanonicalQuantLayout::kInt4Symmetric, 32, "int4"},
      {quixicore_cpu::CanonicalQuantLayout::kFP8E4M3FN, 32, "fp8_e4m3"},
      {quixicore_cpu::CanonicalQuantLayout::kMXFP4E2M1E8M0, 32, "mxfp4"},
      {quixicore_cpu::CanonicalQuantLayout::kBitNetTernary, 32, "bitnet"},
  };
  const long long nq = ctx.preset == Preset::kSmoke
                           ? 33
                           : (ctx.preset == Preset::kQuick ? 512 : 4096);
  const long long nk = ctx.preset == Preset::kSmoke
                           ? 17
                           : (ctx.preset == Preset::kQuick ? 128 : 1024);
  const long long nv = ctx.preset == Preset::kSmoke
                           ? 19
                           : (ctx.preset == Preset::kQuick ? 128 : 1024);
  const long long k = ctx.preset == Preset::kSmoke
                          ? 32
                          : (ctx.preset == Preset::kQuick ? 1024 : 4096);
  const std::size_t count = ctx.preset == Preset::kSmoke ? 1 : 4;
  for (long long m : {1LL, 16LL, 128LL}) {
    for (std::size_t index = 0; index < count; ++index)
      out.push_back(canonical_qkv(formats[index], m, nq, nk, nv, k, family));
    const std::size_t typed_count = ctx.preset == Preset::kSmoke ? 1 : 2;
    for (std::size_t index = 0; index < typed_count; ++index) {
      out.push_back(canonical_qkv(formats[index], m, nq, nk, nv, k, family,
                                  quixicore_cpu::FloatStorageType::kF16));
      out.push_back(canonical_qkv(formats[index], m, nq, nk, nv, k, family,
                                  quixicore_cpu::FloatStorageType::kBF16));
    }
  }
}

CaseDecl canonical_qkv_rope_kv(ProjectionFormat format, long long query_heads,
                               long long kv_heads, long long head_dim,
                               long long k, const char* family,
                               quixicore_cpu::FloatStorageType input_type =
                                   quixicore_cpu::FloatStorageType::kF32) {
  const long long query_dim = query_heads * head_dim;
  const long long kv_dim = kv_heads * head_dim;
  CaseDecl decl;
  decl.kernel = family;
  decl.variant = "canonical_qkv_rope_kv_" + std::string(format.name) + "_Hq" +
                 std::to_string(query_heads) + "_Hkv" +
                 std::to_string(kv_heads) + "_D" + std::to_string(head_dim) +
                 "_K" + std::to_string(k);
  if (input_type == quixicore_cpu::FloatStorageType::kF16)
    decl.variant += "_f16";
  else if (input_type == quixicore_cpu::FloatStorageType::kBF16)
    decl.variant += "_bf16";
  decl.shape = {{"query_heads", query_heads},
                {"kv_heads", kv_heads},
                {"head_dim", head_dim},
                {"k", k}};
  decl.dtype =
      input_type == quixicore_cpu::FloatStorageType::kF16
          ? "f16"
          : (input_type == quixicore_cpu::FloatStorageType::kBF16 ? "bf16"
                                                                  : "f32");
  decl.format = format.name;
  decl.notes =
      "M3 F5 direct canonical QKV, split-half RoPE, and typed KV write";
  decl.flops = 2.0 * static_cast<double>(query_dim + 2 * kv_dim) *
               static_cast<double>(k);
  decl.make = [=]() {
    auto buffers = std::make_shared<CanonicalQKVRopeBuffers>();
    buffers->query_heads = query_heads;
    buffers->kv_heads = kv_heads;
    buffers->head_dim = head_dim;
    buffers->k = k;
    buffers->slots = 4;
    buffers->max_position = 4;
    buffers->position = 2;
    buffers->slot = 1;
    buffers->input_type = input_type;
    const long long rows[3] = {query_dim, kv_dim, kv_dim};
    const bool scale_2d =
        format.layout == quixicore_cpu::CanonicalQuantLayout::kNVFP4E2M1E4M3;
    for (int plane = 0; plane < 3; ++plane) {
      std::vector<float> source(static_cast<std::size_t>(rows[plane] * k));
      for (long long index = 0; index < rows[plane] * k; ++index) {
        source[static_cast<std::size_t>(index)] =
            static_cast<float>(
                static_cast<int>((index * (17 + plane * 6) + 7 + plane * 3) %
                                 67) -
                33) /
            static_cast<float>(19 + plane * 2);
      }
      if (quixicore_cpu::quantize_canonical(
              {source.data(), quixicore_cpu::FloatStorageType::kF32,
               rows[plane] * k},
              rows[plane], k, format.layout, format.group_size,
              &buffers->tensors[plane], scale_2d) != Status::kOk ||
          buffers->prepared[plane].prepare(buffers->tensors[plane]) !=
              Status::kOk) {
        throw std::runtime_error("canonical QKV RoPE preparation failed");
      }
    }
    buffers->activation = aligned_alloc_array<float>(k);
    if (input_type != quixicore_cpu::FloatStorageType::kF32)
      buffers->activation_16 = aligned_alloc_array<std::uint16_t>(k);
    for (long long index = 0; index < k; ++index) {
      const float value =
          static_cast<float>(static_cast<int>((index * 29 + 7) % 71) - 35) /
          23.0f;
      buffers->activation.get()[index] = value;
      if (input_type == quixicore_cpu::FloatStorageType::kF16)
        buffers->activation_16.get()[index] =
            quixicore_cpu::float_to_f16(value);
      else if (input_type == quixicore_cpu::FloatStorageType::kBF16)
        buffers->activation_16.get()[index] =
            quixicore_cpu::float_to_bf16(value);
    }
    const long long cache_count = buffers->slots * kv_dim;
    buffers->q = aligned_alloc_array<float>(query_dim);
    buffers->key_cache = aligned_alloc_array<float>(cache_count);
    buffers->value_cache = aligned_alloc_array<float>(cache_count);
    buffers->raw_q = aligned_alloc_array<float>(query_dim);
    buffers->raw_k = aligned_alloc_array<float>(kv_dim);
    buffers->raw_v = aligned_alloc_array<float>(kv_dim);
    buffers->baseline_q = aligned_alloc_array<float>(query_dim);
    buffers->baseline_key = aligned_alloc_array<float>(cache_count);
    buffers->baseline_value = aligned_alloc_array<float>(cache_count);
    buffers->cosine.resize(
        static_cast<std::size_t>(buffers->max_position * head_dim / 2));
    buffers->sine.resize(buffers->cosine.size());
    for (long long pos = 0; pos < buffers->max_position; ++pos) {
      for (long long dim = 0; dim < head_dim / 2; ++dim) {
        const float angle = 0.01f * static_cast<float>((pos + 1) * (dim + 1));
        buffers->cosine[static_cast<std::size_t>(pos * head_dim / 2 + dim)] =
            std::cos(angle);
        buffers->sine[static_cast<std::size_t>(pos * head_dim / 2 + dim)] =
            std::sin(angle);
      }
    }
    auto input_view = [buffers]() {
      return quixicore_cpu::FloatStorageInput{
          buffers->input_type == quixicore_cpu::FloatStorageType::kF32
              ? static_cast<const void*>(buffers->activation.get())
              : static_cast<const void*>(buffers->activation_16.get()),
          buffers->input_type, buffers->k};
    };
    auto target = [buffers, input_view, query_dim, kv_dim]() {
      if (quixicore_cpu::qgemv_prepacked_qkv_rope_kv_storage(
              buffers->prepared[0], buffers->prepared[1], buffers->prepared[2],
              input_view(), buffers->cosine.data(), buffers->sine.data(),
              {buffers->q.get(), quixicore_cpu::FloatStorageType::kF32,
               query_dim},
              {buffers->key_cache.get(), quixicore_cpu::FloatStorageType::kF32,
               buffers->slots * kv_dim},
              {buffers->value_cache.get(),
               quixicore_cpu::FloatStorageType::kF32, buffers->slots * kv_dim},
              buffers->query_heads, buffers->kv_heads, buffers->head_dim,
              buffers->slots, buffers->max_position, buffers->position,
              buffers->slot) != Status::kOk) {
        throw std::runtime_error("canonical fused QKV RoPE KV failed");
      }
      do_not_optimize(buffers->q.get());
      do_not_optimize(buffers->key_cache.get());
      do_not_optimize(buffers->value_cache.get());
    };
    auto composed = [buffers, input_view, query_dim, kv_dim]() {
      if (quixicore_cpu::qgemv_prepacked_qkv_storage(
              buffers->prepared[0], buffers->prepared[1], buffers->prepared[2],
              input_view(),
              {buffers->raw_q.get(), quixicore_cpu::FloatStorageType::kF32,
               query_dim},
              {buffers->raw_k.get(), quixicore_cpu::FloatStorageType::kF32,
               kv_dim},
              {buffers->raw_v.get(), quixicore_cpu::FloatStorageType::kF32,
               kv_dim}) != Status::kOk) {
        throw std::runtime_error("composed QKV projection failed");
      }
      const long long half = buffers->head_dim / 2;
      const float* cosine = buffers->cosine.data() + buffers->position * half;
      const float* sine = buffers->sine.data() + buffers->position * half;
      for (long long head = 0; head < buffers->query_heads; ++head) {
        for (long long dim = 0; dim < half; ++dim) {
          const long long row0 = head * buffers->head_dim + dim;
          const long long row1 = row0 + half;
          const float first = buffers->raw_q.get()[row0];
          const float second = buffers->raw_q.get()[row1];
          buffers->baseline_q.get()[row0] =
              first * cosine[dim] - second * sine[dim];
          buffers->baseline_q.get()[row1] =
              second * cosine[dim] + first * sine[dim];
        }
      }
      const long long cache_base = buffers->slot * kv_dim;
      for (long long head = 0; head < buffers->kv_heads; ++head) {
        for (long long dim = 0; dim < half; ++dim) {
          const long long row0 = head * buffers->head_dim + dim;
          const long long row1 = row0 + half;
          const float first = buffers->raw_k.get()[row0];
          const float second = buffers->raw_k.get()[row1];
          buffers->baseline_key.get()[cache_base + row0] =
              first * cosine[dim] - second * sine[dim];
          buffers->baseline_key.get()[cache_base + row1] =
              second * cosine[dim] + first * sine[dim];
        }
      }
      for (long long row = 0; row < kv_dim; ++row)
        buffers->baseline_value.get()[cache_base + row] =
            buffers->raw_v.get()[row];
      do_not_optimize(buffers->baseline_q.get());
      do_not_optimize(buffers->baseline_key.get());
      do_not_optimize(buffers->baseline_value.get());
    };
    CaseBody body;
    body.target = target;
    body.baselines.emplace_back("qkv_then_rope_kv_write", composed);
    body.check = [buffers, target, composed, query_dim, kv_dim]() {
      std::fill_n(buffers->key_cache.get(), buffers->slots * kv_dim, 0.0f);
      std::fill_n(buffers->value_cache.get(), buffers->slots * kv_dim, 0.0f);
      std::fill_n(buffers->baseline_key.get(), buffers->slots * kv_dim, 0.0f);
      std::fill_n(buffers->baseline_value.get(), buffers->slots * kv_dim, 0.0f);
      target();
      composed();
      CheckResult check;
      for (long long index = 0; index < query_dim; ++index)
        check_value(check, buffers->q.get()[index],
                    buffers->baseline_q.get()[index], Tolerance{2e-4, 2e-4});
      for (long long index = 0; index < buffers->slots * kv_dim; ++index) {
        check_value(check, buffers->key_cache.get()[index],
                    buffers->baseline_key.get()[index], Tolerance{2e-4, 2e-4});
        check_value(check, buffers->value_cache.get()[index],
                    buffers->baseline_value.get()[index],
                    Tolerance{2e-4, 2e-4});
      }
      return check;
    };
    return body;
  };
  return decl;
}

void append_canonical_qkv_rope_kv_cases(const BuildCtx& ctx, const char* family,
                                        std::vector<CaseDecl>& out) {
  const ProjectionFormat formats[] = {
      {quixicore_cpu::CanonicalQuantLayout::kInt4Symmetric, 32, "int4"},
      {quixicore_cpu::CanonicalQuantLayout::kFP8E4M3FN, 32, "fp8_e4m3"},
      {quixicore_cpu::CanonicalQuantLayout::kMXFP4E2M1E8M0, 32, "mxfp4"},
      {quixicore_cpu::CanonicalQuantLayout::kBitNetTernary, 32, "bitnet"},
  };
  const long long query_heads = ctx.preset == Preset::kSmoke ? 2 : 8;
  const long long kv_heads = ctx.preset == Preset::kSmoke ? 1 : 2;
  const long long head_dim = ctx.preset == Preset::kSmoke ? 16 : 64;
  const long long k = ctx.preset == Preset::kSmoke
                          ? 32
                          : (ctx.preset == Preset::kQuick ? 1024 : 4096);
  const std::size_t count = ctx.preset == Preset::kSmoke ? 1 : 4;
  for (std::size_t index = 0; index < count; ++index) {
    out.push_back(canonical_qkv_rope_kv(formats[index], query_heads, kv_heads,
                                        head_dim, k, family));
  }
  const std::size_t typed_count = ctx.preset == Preset::kSmoke ? 1 : 2;
  for (std::size_t index = 0; index < typed_count; ++index) {
    out.push_back(canonical_qkv_rope_kv(formats[index], query_heads, kv_heads,
                                        head_dim, k, family,
                                        quixicore_cpu::FloatStorageType::kF16));
    out.push_back(canonical_qkv_rope_kv(
        formats[index], query_heads, kv_heads, head_dim, k, family,
        quixicore_cpu::FloatStorageType::kBF16));
  }
}

void append_canonical_swiglu_cases(const BuildCtx& ctx, const char* family,
                                   std::vector<CaseDecl>& out) {
  const ProjectionFormat formats[] = {
      {quixicore_cpu::CanonicalQuantLayout::kInt4Symmetric, 32, "int4"},
      {quixicore_cpu::CanonicalQuantLayout::kFP8E4M3FN, 32, "fp8_e4m3"},
      {quixicore_cpu::CanonicalQuantLayout::kMXFP4E2M1E8M0, 32, "mxfp4"},
      {quixicore_cpu::CanonicalQuantLayout::kBitNetTernary, 32, "bitnet"},
  };
  const long long n = ctx.preset == Preset::kSmoke
                          ? 33
                          : (ctx.preset == Preset::kQuick ? 512 : 4096);
  const long long k = ctx.preset == Preset::kSmoke
                          ? 32
                          : (ctx.preset == Preset::kQuick ? 1024 : 4096);
  const std::size_t count = ctx.preset == Preset::kSmoke ? 1 : 4;
  for (long long m : {1LL, 16LL, 128LL}) {
    for (std::size_t index = 0; index < count; ++index) {
      out.push_back(canonical_gate_up(formats[index], m, n, k, family, true));
    }
  }
}

void append_canonical_swiglu_quant_cases(const BuildCtx& ctx,
                                         const char* family,
                                         std::vector<CaseDecl>& out) {
  const ProjectionFormat formats[] = {
      {quixicore_cpu::CanonicalQuantLayout::kInt4Symmetric, 32, "int4"},
      {quixicore_cpu::CanonicalQuantLayout::kFP8E4M3FN, 32, "fp8_e4m3"},
      {quixicore_cpu::CanonicalQuantLayout::kMXFP4E2M1E8M0, 32, "mxfp4"},
      {quixicore_cpu::CanonicalQuantLayout::kBitNetTernary, 32, "bitnet"},
  };
  const long long n = ctx.preset == Preset::kSmoke
                          ? 32
                          : (ctx.preset == Preset::kQuick ? 512 : 4096);
  const long long k = ctx.preset == Preset::kSmoke
                          ? 32
                          : (ctx.preset == Preset::kQuick ? 1024 : 4096);
  const std::size_t count = ctx.preset == Preset::kSmoke ? 1 : 4;
  for (long long m : {1LL, 16LL, 128LL}) {
    for (std::size_t index = 0; index < count; ++index) {
      out.push_back(
          canonical_gate_up(formats[index], m, n, k, family, false, true));
    }
  }
}

CaseDecl canonical_dual_projection(DualProjectionFormat format, long long m,
                                   long long n, long long k) {
  CaseDecl decl;
  decl.kernel = m == 1 ? "quant_gemv_matrix" : "quant_gemm_matrix";
  decl.variant = "canonical_dual_" + std::string(format.name) + "_M" +
                 std::to_string(m) + "_N" + std::to_string(n) + "_K" +
                 std::to_string(k);
  decl.shape = {{"m", m}, {"n", n}, {"k", k}};
  decl.format = format.name;
  decl.notes = m == 1 ? "M2 canonical packed weight and activation GEMV"
                      : "M2 canonical packed weight and activation GEMM";
  decl.flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
               static_cast<double>(k);
  decl.make = [format, m, n, k]() {
    auto buffers = std::make_shared<CanonicalDualProjectionBuffers>();
    buffers->m = m;
    buffers->n = n;
    buffers->k = k;
    std::vector<float> weight_source(static_cast<std::size_t>(n * k));
    std::vector<float> activation_source(static_cast<std::size_t>(m * k));
    for (long long index = 0; index < n * k; ++index) {
      weight_source[static_cast<std::size_t>(index)] =
          static_cast<float>(static_cast<int>((index * 17 + 11) % 61) - 30) /
          19.0f;
    }
    for (long long index = 0; index < m * k; ++index) {
      activation_source[static_cast<std::size_t>(index)] =
          static_cast<float>(static_cast<int>((index * 29 + 7) % 67) - 33) /
          23.0f;
    }
    const bool weight_scale_2d =
        format.weight.layout ==
        quixicore_cpu::CanonicalQuantLayout::kNVFP4E2M1E4M3;
    const bool activation_scale_2d =
        format.activation.layout ==
        quixicore_cpu::CanonicalQuantLayout::kNVFP4E2M1E4M3;
    if (quixicore_cpu::quantize_canonical(
            {weight_source.data(), quixicore_cpu::FloatStorageType::kF32,
             n * k},
            n, k, format.weight.layout, format.weight.group_size,
            &buffers->weight_tensor, weight_scale_2d) != Status::kOk ||
        quixicore_cpu::quantize_canonical(
            {activation_source.data(), quixicore_cpu::FloatStorageType::kF32,
             m * k},
            m, k, format.activation.layout, format.activation.group_size,
            &buffers->activation_tensor, activation_scale_2d) != Status::kOk) {
      throw std::runtime_error("canonical dual projection packing failed");
    }
    if (buffers->prepared.prepare(buffers->weight_tensor) != Status::kOk) {
      throw std::runtime_error("canonical dual weight preparation failed");
    }
    buffers->weights.resize(static_cast<std::size_t>(n * k));
    buffers->activations.resize(static_cast<std::size_t>(m * k));
    if (quixicore_cpu::dequantize_canonical(buffers->weight_tensor,
                                            buffers->weights.data(),
                                            n * k) != Status::kOk ||
        quixicore_cpu::dequantize_canonical(buffers->activation_tensor,
                                            buffers->activations.data(),
                                            m * k) != Status::kOk) {
      throw std::runtime_error("canonical dual projection decode failed");
    }
    buffers->output = aligned_alloc_array<float>(m * n);
    buffers->baseline = aligned_alloc_array<float>(m * n);
    buffers->reference = aligned_alloc_array<float>(m * n);
    for (long long input_row = 0; input_row < m; ++input_row) {
      for (long long output_row = 0; output_row < n; ++output_row) {
        double sum = 0.0;
        for (long long column = 0; column < k; ++column) {
          sum +=
              static_cast<double>(buffers->weights[output_row * k + column]) *
              buffers->activations[static_cast<std::size_t>(input_row * k +
                                                            column)];
        }
        buffers->reference.get()[input_row * n + output_row] =
            static_cast<float>(sum);
      }
    }
    auto target = [buffers]() {
      if (quixicore_cpu::qgemm_prepacked_quantized(
              buffers->prepared, buffers->activation_tensor,
              buffers->output.get()) != Status::kOk) {
        throw std::runtime_error("canonical dual projection failed");
      }
      do_not_optimize(buffers->output.get());
    };
    auto baseline = [buffers]() {
      for (long long input_row = 0; input_row < buffers->m; ++input_row) {
        for (long long output_row = 0; output_row < buffers->n; ++output_row) {
          float sum = 0.0f;
          for (long long column = 0; column < buffers->k; ++column) {
            sum += buffers->weights[output_row * buffers->k + column] *
                   buffers->activations[static_cast<std::size_t>(
                       input_row * buffers->k + column)];
          }
          buffers->baseline.get()[input_row * buffers->n + output_row] = sum;
        }
      }
      do_not_optimize(buffers->baseline.get());
    };
    CaseBody body;
    body.target = target;
    body.baselines.emplace_back(
        buffers->m == 1 ? "dequantized_scalar_gemv" : "dequantized_scalar_gemm",
        baseline);
    body.check = [buffers, target]() {
      target();
      CheckResult check;
      for (long long index = 0; index < buffers->m * buffers->n; ++index) {
        check_value(check, buffers->output.get()[index],
                    buffers->reference.get()[index], Tolerance{1e-4, 1e-4});
      }
      return check;
    };
    return body;
  };
  return decl;
}

void append_canonical_dual_projection_cases(const BuildCtx& ctx, long long m,
                                            std::vector<CaseDecl>& out) {
  const ProjectionFormat int4 = {
      quixicore_cpu::CanonicalQuantLayout::kInt4Symmetric, 32, "int4"};
  const ProjectionFormat uint4 = {
      quixicore_cpu::CanonicalQuantLayout::kUInt4Affine, 32, "uint4"};
  const ProjectionFormat int8 = {
      quixicore_cpu::CanonicalQuantLayout::kInt8Symmetric, 32, "int8"};
  const ProjectionFormat int8_affine = {
      quixicore_cpu::CanonicalQuantLayout::kInt8Affine, 0, "int8_affine"};
  const ProjectionFormat fp8_e4 = {
      quixicore_cpu::CanonicalQuantLayout::kFP8E4M3FN, 32, "fp8_e4m3"};
  const ProjectionFormat fp8_e5 = {
      quixicore_cpu::CanonicalQuantLayout::kFP8E5M2, 32, "fp8_e5m2"};
  const ProjectionFormat fp4 = {quixicore_cpu::CanonicalQuantLayout::kFP4E2M1,
                                32, "fp4"};
  const ProjectionFormat mxfp8 = {
      quixicore_cpu::CanonicalQuantLayout::kMXFP8E4M3E8M0, 32, "mxfp8"};
  const ProjectionFormat mxfp4 = {
      quixicore_cpu::CanonicalQuantLayout::kMXFP4E2M1E8M0, 32, "mxfp4"};
  const ProjectionFormat nvfp4 = {
      quixicore_cpu::CanonicalQuantLayout::kNVFP4E2M1E4M3, 16, "nvfp4"};
  const ProjectionFormat bitnet = {
      quixicore_cpu::CanonicalQuantLayout::kBitNetTernary, 32, "bitnet"};
  const DualProjectionFormat formats[] = {
      {int4, int4, "w4a4"},         {uint4, int8, "w4a8"},
      {int8, int8_affine, "w8a8"},  {fp8_e4, fp8_e4, "fp8_e4e4"},
      {fp8_e5, fp8_e4, "fp8_e5e4"}, {mxfp8, mxfp8, "mxfp8"},
      {mxfp4, mxfp4, "mxfp4"},      {nvfp4, nvfp4, "nvfp4"},
      {bitnet, int8, "bitnet_a8"},  {bitnet, fp4, "bitnet_a4"},
  };
  const long long n = ctx.preset == Preset::kSmoke
                          ? 33
                          : (ctx.preset == Preset::kQuick ? 512 : 4096);
  const long long k = ctx.preset == Preset::kSmoke
                          ? 32
                          : (ctx.preset == Preset::kQuick ? 1024 : 4096);
  const std::size_t representative_indices[] = {0, 3, 6, 8};
  const std::size_t count =
      ctx.preset == Preset::kSmoke ? 1 : (m == 128 ? 4 : 10);
  for (std::size_t item = 0; item < count; ++item) {
    const std::size_t index = m == 128 && ctx.preset != Preset::kSmoke
                                  ? representative_indices[item]
                                  : item;
    out.push_back(canonical_dual_projection(formats[index], m, n, k));
  }
}

CaseDecl bitnet_pass0(long long n, long long k) {
  CaseDecl decl;
  decl.kernel = "bitnet_matrix";
  decl.variant = "b1_58_qgemv_N" + std::to_string(n) + "_K" + std::to_string(k);
  decl.shape = {{"m", 1}, {"n", n}, {"k", k}};
  decl.format = "bitnet_b1_58";
  decl.notes = "M0 pass-0 canonical ternary pack plus portable qgemv";
  decl.flops = 2.0 * static_cast<double>(n) * static_cast<double>(k);
  decl.make = [n, k]() {
    auto buffers = std::make_shared<BitNetBuffers>();
    buffers->n = n;
    buffers->k = k;
    buffers->weights = aligned_alloc_array<float>(n * k);
    buffers->dequantized = aligned_alloc_array<float>(n * k);
    buffers->activation = aligned_alloc_array<float>(k);
    buffers->output = aligned_alloc_array<float>(n);
    buffers->reference = aligned_alloc_array<float>(n);
    buffers->packed = aligned_alloc_array<std::uint8_t>(n * (k / 32) * 10);
    for (long long i = 0; i < n * k; ++i) {
      buffers->weights.get()[i] =
          static_cast<float>(static_cast<int>(i % 3) - 1) *
          (0.25f + 0.03125f * static_cast<float>((i / k) % 7));
    }
    for (long long i = 0; i < k; ++i) {
      buffers->activation.get()[i] =
          static_cast<float>(static_cast<int>((i * 17) % 31) - 15) / 16.0f;
    }
    if (quixicore_cpu::ternary_pack(
            buffers->weights.get(), buffers->packed.get(),
            buffers->dequantized.get(), n, k, k) != Status::kOk) {
      throw std::runtime_error("BitNet pass-0 packing failed");
    }
    auto target = [buffers]() {
      if (quixicore_cpu::qgemv(quixicore_cpu::QuantFormat::kBitnet,
                               buffers->packed.get(), buffers->activation.get(),
                               buffers->output.get(), buffers->n,
                               buffers->k) != Status::kOk) {
        throw std::runtime_error("BitNet pass-0 qgemv failed");
      }
      do_not_optimize(buffers->output.get());
    };
    auto baseline = [buffers]() {
      for (long long row = 0; row < buffers->n; ++row) {
        double sum = 0.0;
        for (long long column = 0; column < buffers->k; ++column) {
          sum += buffers->dequantized.get()[row * buffers->k + column] *
                 buffers->activation.get()[column];
        }
        buffers->reference.get()[row] = static_cast<float>(sum);
      }
      do_not_optimize(buffers->reference.get());
    };
    CaseBody body;
    body.target = target;
    body.baselines.emplace_back("dequantized_f64_dot", baseline);
    body.check = [buffers, target, baseline]() {
      target();
      baseline();
      CheckResult check;
      for (long long row = 0; row < buffers->n; ++row) {
        check_value(check, buffers->output.get()[row],
                    buffers->reference.get()[row], Tolerance{2e-4, 2e-4});
      }
      return check;
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_quant_formats_matrix_cases(const BuildCtx& ctx,
                                      std::vector<CaseDecl>& out) {
  append_cases(ctx, &build_qgemv_formats_cases, "quant_formats",
               {"fp8_e4m3", "fp8_e5m2", "mxfp4", "nvfp4", "bitnet"}, out);
}

void build_quant_activation_matrix_cases(const BuildCtx& ctx,
                                         std::vector<CaseDecl>& out) {
  append_cases(ctx, &build_quant_lifecycle_cases, "quant_activation",
               {"activation_pack_"}, out);
  append_cases(ctx, &build_ported_ops_cases, "quant_activation", {"rms_add_i8"},
               out);
}

void build_quant_gemv_matrix_cases(const BuildCtx& ctx,
                                   std::vector<CaseDecl>& out) {
  append_canonical_projection_cases(ctx, 1, out);
  append_canonical_typed_projection_cases(ctx, 1, out);
  append_canonical_dual_projection_cases(ctx, 1, out);
  append_cases(ctx, &build_qgemv_formats_cases, "quant_gemv_matrix",
               {"q4_0", "mxfp4", "nvfp4", "bitnet"}, out);
}

void build_quant_gemm_matrix_cases(const BuildCtx& ctx,
                                   std::vector<CaseDecl>& out) {
  append_canonical_projection_cases(ctx, 16, out);
  append_canonical_projection_cases(ctx, 128, out);
  append_canonical_typed_projection_cases(ctx, 16, out);
  append_canonical_typed_projection_cases(ctx, 128, out);
  append_canonical_dual_projection_cases(ctx, 16, out);
  append_canonical_dual_projection_cases(ctx, 128, out);
  append_cases(ctx, &build_prerequisites_cases, "quant_gemm_matrix",
               {"prepacked_q4_0", "prepacked_q4_k", "prepacked_q6_k"}, out);
  append_cases(ctx, &build_ported_ops_cases, "quant_gemm_matrix", {"mxfp8_"},
               out);
}

void build_quant_fusions_matrix_cases(const BuildCtx& ctx,
                                      std::vector<CaseDecl>& out) {
  append_canonical_epilogue_cases(ctx, 1, out);
  append_canonical_epilogue_cases(ctx, 16, out);
  append_canonical_epilogue_cases(ctx, 128, out);
  append_canonical_gate_up_cases(ctx, "quant_fusions", out);
  append_canonical_swiglu_cases(ctx, "quant_fusions", out);
  append_canonical_swiglu_quant_cases(ctx, "quant_fusions", out);
  append_canonical_qkv_cases(ctx, "quant_fusions", out);
  append_canonical_qkv_rope_kv_cases(ctx, "quant_fusions", out);
  append_family_cases(ctx, &build_quant_norm_cases, "quant_fusions", out);
  append_cases(ctx, &build_optimization_plan_cases, "quant_fusions",
               {"fused_swiglu", "fused_qkv_rope_kv"}, out);
}

void build_quant_gate_up_cases(const BuildCtx& ctx,
                               std::vector<CaseDecl>& out) {
  append_canonical_gate_up_cases(ctx, "quant_gate_up", out);
}

void build_quant_swiglu_cases(const BuildCtx& ctx, std::vector<CaseDecl>& out) {
  append_canonical_swiglu_cases(ctx, "quant_swiglu", out);
}

void build_quant_swiglu_quant_cases(const BuildCtx& ctx,
                                    std::vector<CaseDecl>& out) {
  append_canonical_swiglu_quant_cases(ctx, "quant_swiglu_quant", out);
}

void build_quant_qkv_cases(const BuildCtx& ctx, std::vector<CaseDecl>& out) {
  append_canonical_qkv_cases(ctx, "quant_qkv", out);
}

void build_quant_qkv_rope_kv_cases(const BuildCtx& ctx,
                                   std::vector<CaseDecl>& out) {
  append_canonical_qkv_rope_kv_cases(ctx, "quant_qkv_rope_kv", out);
}

void build_quant_serving_matrix_cases(const BuildCtx& ctx,
                                      std::vector<CaseDecl>& out) {
  append_family_cases(ctx, &build_quant_embedding_cases, "quant_serving", out);
  append_family_cases(ctx, &build_quant_lm_head_cases, "quant_serving", out);
  append_family_cases(ctx, &build_quant_moe_cases, "quant_serving", out);
  append_cases(ctx, &build_optimization_plan_cases, "quant_serving",
               {"streaming_lm_head", "tiled_moe"}, out);
}

void build_quant_kv_matrix_cases(const BuildCtx& ctx,
                                 std::vector<CaseDecl>& out) {
  append_cases(ctx, &build_optimization_plan_cases, "quant_kv",
               {"online_paged", "online_mla"}, out);
}

void build_bitnet_matrix_cases(const BuildCtx& ctx,
                               std::vector<CaseDecl>& out) {
  const long long n = ctx.preset == Preset::kSmoke ? 64 : 1024;
  const long long k = ctx.preset == Preset::kSmoke ? 256 : 4096;
  out.push_back(bitnet_pass0(n, k));
  if (ctx.preset == Preset::kComprehensive) {
    out.push_back(bitnet_pass0(4096, 4096));
  }
}

}  // namespace qcb
