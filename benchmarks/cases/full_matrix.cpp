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
#include "quixicore_cpu/qgemv.h"
#include "quixicore_cpu/quantization.h"

namespace qcb {

void build_qgemv_formats_cases(const BuildCtx &, std::vector<CaseDecl> &);
void build_quant_lifecycle_cases(const BuildCtx &, std::vector<CaseDecl> &);
void build_ported_ops_cases(const BuildCtx &, std::vector<CaseDecl> &);
void build_prerequisites_cases(const BuildCtx &, std::vector<CaseDecl> &);
void build_optimization_plan_cases(const BuildCtx &, std::vector<CaseDecl> &);

namespace {

using quixicore_cpu::Status;

bool matches(const CaseDecl &decl,
             std::initializer_list<std::string_view> needles) {
  for (std::string_view needle : needles) {
    if (decl.variant.find(needle) != std::string::npos ||
        decl.format.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

void append_cases(const BuildCtx &ctx, CaseBuilder builder, const char *family,
                  std::initializer_list<std::string_view> needles,
                  std::vector<CaseDecl> &out) {
  std::vector<CaseDecl> source;
  builder(ctx, source);
  for (CaseDecl &decl : source) {
    if (!matches(decl, needles))
      continue;
    decl.variant = decl.kernel + "_" + decl.variant;
    decl.kernel = family;
    decl.notes = "M0 pass-0 existing reference: " + decl.notes;
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

} // namespace

void build_quant_formats_matrix_cases(const BuildCtx &ctx,
                                      std::vector<CaseDecl> &out) {
  append_cases(ctx, &build_qgemv_formats_cases, "quant_formats",
               {"fp8_e4m3", "fp8_e5m2", "mxfp4", "nvfp4", "bitnet"}, out);
}

void build_quant_activation_matrix_cases(const BuildCtx &ctx,
                                         std::vector<CaseDecl> &out) {
  append_cases(ctx, &build_quant_lifecycle_cases, "quant_activation",
               {"activation_pack_"}, out);
  append_cases(ctx, &build_ported_ops_cases, "quant_activation", {"rms_add_i8"},
               out);
}

void build_quant_gemv_matrix_cases(const BuildCtx &ctx,
                                   std::vector<CaseDecl> &out) {
  append_cases(ctx, &build_qgemv_formats_cases, "quant_gemv_matrix",
               {"q4_0", "mxfp4", "nvfp4", "bitnet"}, out);
}

void build_quant_gemm_matrix_cases(const BuildCtx &ctx,
                                   std::vector<CaseDecl> &out) {
  append_cases(ctx, &build_prerequisites_cases, "quant_gemm_matrix",
               {"prepacked_q4_0", "prepacked_q4_k", "prepacked_q6_k"}, out);
  append_cases(ctx, &build_ported_ops_cases, "quant_gemm_matrix", {"mxfp8_"},
               out);
}

void build_quant_fusions_matrix_cases(const BuildCtx &ctx,
                                      std::vector<CaseDecl> &out) {
  append_cases(ctx, &build_ported_ops_cases, "quant_fusions", {"rms_add_i8"},
               out);
  append_cases(ctx, &build_optimization_plan_cases, "quant_fusions",
               {"fused_swiglu", "fused_qkv_rope_kv"}, out);
}

void build_quant_serving_matrix_cases(const BuildCtx &ctx,
                                      std::vector<CaseDecl> &out) {
  append_cases(ctx, &build_optimization_plan_cases, "quant_serving",
               {"streaming_lm_head", "tiled_moe"}, out);
}

void build_quant_kv_matrix_cases(const BuildCtx &ctx,
                                 std::vector<CaseDecl> &out) {
  append_cases(ctx, &build_optimization_plan_cases, "quant_kv",
               {"online_paged", "online_mla"}, out);
}

void build_bitnet_matrix_cases(const BuildCtx &ctx,
                               std::vector<CaseDecl> &out) {
  const long long n = ctx.preset == Preset::kSmoke ? 64 : 1024;
  const long long k = ctx.preset == Preset::kSmoke ? 256 : 4096;
  out.push_back(bitnet_pass0(n, k));
  if (ctx.preset == Preset::kComprehensive) {
    out.push_back(bitnet_pass0(4096, 4096));
  }
}

} // namespace qcb
