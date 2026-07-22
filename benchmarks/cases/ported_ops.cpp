// Focused performance evidence for newly ported fused CPU semantics.
// The target calls the fused public API; the baseline composes the same
// contract operations with preallocated intermediate storage.

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness/alloc.h"
#include "harness/case.h"
#include "harness/donotopt.h"
#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/qgemm.h"
#include "quixicore_cpu/quantization.h"

namespace qcb {
namespace {

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
  std::uint32_t state_ = 0x517CC1B7u;
};

struct NormQuantBuffers {
  AlignedBuffer<float> x, residual, weight, residual_out, scales;
  AlignedBuffer<float> normalized, reference_residual, reference_scales;
  AlignedBuffer<std::int8_t> codes, reference_codes;
  long long rows, hidden, group_size, groups;
};

void run_decomposed(NormQuantBuffers& buffers) {
  if (quixicore_cpu::rms_norm_add(
          buffers.x.get(), buffers.residual.get(), buffers.weight.get(),
          buffers.normalized.get(), buffers.reference_residual.get(),
          buffers.rows, buffers.hidden) != Status::kOk ||
      quixicore_cpu::quantize_int8(
          buffers.normalized.get(), buffers.reference_codes.get(),
          buffers.reference_scales.get(), buffers.rows, buffers.hidden,
          buffers.group_size) != Status::kOk) {
    throw std::runtime_error("decomposed norm quant failed");
  }
}

CaseDecl make_norm_quant(long long rows, long long hidden,
                         long long group_size) {
  CaseDecl decl;
  decl.kernel = "ported_ops";
  decl.variant = "rms_add_i8_R" + std::to_string(rows) + "_H" +
                 std::to_string(hidden) + "_G" +
                 std::to_string(group_size);
  decl.shape = {{"rows", rows}, {"hidden", hidden},
                {"group_size", group_size}};
  decl.format = "int8_group";
  decl.notes = "fused RMSNorm-add plus dynamic group-int8 quantization";
  decl.make = [rows, hidden, group_size]() {
    auto buffers = std::make_shared<NormQuantBuffers>();
    buffers->rows = rows;
    buffers->hidden = hidden;
    buffers->group_size = group_size;
    buffers->groups = hidden / group_size;
    const long long elements = rows * hidden;
    const long long scale_count = rows * buffers->groups;
    buffers->x = aligned_alloc_array<float>(elements);
    buffers->residual = aligned_alloc_array<float>(elements);
    buffers->weight = aligned_alloc_array<float>(hidden);
    buffers->residual_out = aligned_alloc_array<float>(elements);
    buffers->scales = aligned_alloc_array<float>(scale_count);
    buffers->normalized = aligned_alloc_array<float>(elements);
    buffers->reference_residual = aligned_alloc_array<float>(elements);
    buffers->reference_scales = aligned_alloc_array<float>(scale_count);
    buffers->codes = aligned_alloc_array<std::int8_t>(elements);
    buffers->reference_codes = aligned_alloc_array<std::int8_t>(elements);
    Rng rng;
    for (long long i = 0; i < elements; ++i) {
      buffers->x.get()[i] = rng.next();
      buffers->residual.get()[i] = 0.25f * rng.next();
    }
    for (long long i = 0; i < hidden; ++i) {
      buffers->weight.get()[i] = 0.5f + 0.5f * std::fabs(rng.next());
    }

    CaseBody body;
    body.target = [buffers]() {
      if (quixicore_cpu::rms_norm_add_quant_int8(
              buffers->x.get(), buffers->residual.get(),
              buffers->weight.get(), buffers->codes.get(),
              buffers->residual_out.get(), buffers->scales.get(),
              buffers->rows, buffers->hidden, 1e-5f,
              buffers->group_size) != Status::kOk) {
        throw std::runtime_error("rms_norm_add_quant_int8 failed");
      }
      do_not_optimize(buffers->codes.get());
      do_not_optimize(buffers->scales.get());
    };
    body.baselines.emplace_back("decomposed_preallocated", [buffers]() {
      run_decomposed(*buffers);
      do_not_optimize(buffers->reference_codes.get());
      do_not_optimize(buffers->reference_scales.get());
    });
    body.check = [buffers]() {
      if (quixicore_cpu::rms_norm_add_quant_int8(
              buffers->x.get(), buffers->residual.get(),
              buffers->weight.get(), buffers->codes.get(),
              buffers->residual_out.get(), buffers->scales.get(),
              buffers->rows, buffers->hidden, 1e-5f,
              buffers->group_size) != Status::kOk) {
        throw std::runtime_error("rms_norm_add_quant_int8 failed");
      }
      run_decomposed(*buffers);
      CheckResult check;
      const long long elements = buffers->rows * buffers->hidden;
      for (long long i = 0; i < elements; ++i) {
        const long long row = i / buffers->hidden;
        const long long column = i % buffers->hidden;
        const long long scale_index =
            row * buffers->groups + column / buffers->group_size;
        const float actual_dequant =
            buffers->codes.get()[i] * buffers->scales.get()[scale_index];
        const float expected_dequant = buffers->reference_codes.get()[i] *
                                       buffers->reference_scales.get()[scale_index];
        check_value(check, actual_dequant, expected_dequant,
                    Tolerance{0.02, 0.02});
        check_value(check, buffers->residual_out.get()[i],
                    buffers->reference_residual.get()[i], kFp32Tolerance);
      }
      for (long long i = 0; i < buffers->rows * buffers->groups; ++i) {
        check_value(check, buffers->scales.get()[i],
                    buffers->reference_scales.get()[i], kFp32Tolerance);
      }
      return check;
    };
    return body;
  };
  return decl;
}

struct MicroscaleBuffers {
  AlignedBuffer<float> a, b, a_scales, b_scales, out, reference;
  AlignedBuffer<float> a_dequant, b_dequant, b_transposed;
  AlignedBuffer<std::uint8_t> a_codes, b_codes;
  long long m, n, k;
};

void run_predecoded_dense(MicroscaleBuffers& buffers) {
  if (quixicore_cpu::dense_gemm(
          buffers.a_dequant.get(), buffers.b_transposed.get(),
          buffers.reference.get(), buffers.m, buffers.n,
          buffers.k) != Status::kOk) {
    throw std::runtime_error("predecoded dense MXFP8 baseline failed");
  }
}

CaseDecl make_mxfp8_gemm(long long m, long long n, long long k) {
  CaseDecl decl;
  decl.kernel = "ported_ops";
  decl.variant = "mxfp8_M" + std::to_string(m) + "_N" +
                 std::to_string(n) + "_K" + std::to_string(k);
  decl.shape = {{"m", m}, {"n", n}, {"k", k}, {"group_size", 32}};
  decl.format = "mxfp8_e4m3_e8m0";
  decl.notes = "logical-scale MXFP8 GEMM versus predecoded dense GEMM";
  decl.flops = 2.0 * m * n * k;
  decl.make = [m, n, k]() {
    auto buffers = std::make_shared<MicroscaleBuffers>();
    buffers->m = m;
    buffers->n = n;
    buffers->k = k;
    buffers->a = aligned_alloc_array<float>(m * k);
    buffers->b = aligned_alloc_array<float>(n * k);
    buffers->a_codes = aligned_alloc_array<std::uint8_t>(m * k);
    buffers->b_codes = aligned_alloc_array<std::uint8_t>(n * k);
    buffers->a_scales = aligned_alloc_array<float>(m * k / 32);
    buffers->b_scales = aligned_alloc_array<float>(n * k / 32);
    buffers->a_dequant = aligned_alloc_array<float>(m * k);
    buffers->b_dequant = aligned_alloc_array<float>(n * k);
    buffers->b_transposed = aligned_alloc_array<float>(k * n);
    buffers->out = aligned_alloc_array<float>(m * n);
    buffers->reference = aligned_alloc_array<float>(m * n);
    Rng rng;
    for (long long i = 0; i < m * k; ++i) buffers->a.get()[i] = rng.next();
    for (long long i = 0; i < n * k; ++i) buffers->b.get()[i] = rng.next();
    if (quixicore_cpu::mxfp8_quantize(
            buffers->a.get(), buffers->a_codes.get(),
            buffers->a_scales.get(), m, k) != Status::kOk ||
        quixicore_cpu::mxfp8_quantize(
            buffers->b.get(), buffers->b_codes.get(),
            buffers->b_scales.get(), n, k) != Status::kOk ||
        quixicore_cpu::dequantize_float8(
            buffers->a_codes.get(), buffers->a_scales.get(),
            buffers->a_dequant.get(), m, k, 32) != Status::kOk ||
        quixicore_cpu::dequantize_float8(
            buffers->b_codes.get(), buffers->b_scales.get(),
            buffers->b_dequant.get(), n, k, 32) != Status::kOk) {
      throw std::runtime_error("MXFP8 benchmark setup failed");
    }
    for (long long row = 0; row < n; ++row) {
      for (long long input = 0; input < k; ++input) {
        buffers->b_transposed.get()[input * n + row] =
            buffers->b_dequant.get()[row * k + input];
      }
    }
    CaseBody body;
    body.target = [buffers]() {
      if (quixicore_cpu::mxfp8_gemm(
              buffers->a_codes.get(), buffers->a_scales.get(),
              buffers->b_codes.get(), buffers->b_scales.get(),
              buffers->out.get(), buffers->m, buffers->n,
              buffers->k) != Status::kOk) {
        throw std::runtime_error("mxfp8_gemm failed");
      }
      do_not_optimize(buffers->out.get());
    };
    body.baselines.emplace_back("predecoded_dense", [buffers]() {
      run_predecoded_dense(*buffers);
      do_not_optimize(buffers->reference.get());
    });
    body.check = [buffers]() {
      if (quixicore_cpu::mxfp8_gemm(
              buffers->a_codes.get(), buffers->a_scales.get(),
              buffers->b_codes.get(), buffers->b_scales.get(),
              buffers->out.get(), buffers->m, buffers->n,
              buffers->k) != Status::kOk) {
        throw std::runtime_error("mxfp8_gemm failed");
      }
      run_predecoded_dense(*buffers);
      CheckResult check;
      for (long long i = 0; i < buffers->m * buffers->n; ++i) {
        check_value(check, buffers->out.get()[i], buffers->reference.get()[i],
                    Tolerance{2e-4, 2e-4});
      }
      return check;
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_ported_ops_cases(const BuildCtx& ctx, std::vector<CaseDecl>& out) {
  if (ctx.preset == Preset::kSmoke) {
    out.push_back(make_norm_quant(4, 256, 64));
    out.push_back(make_mxfp8_gemm(2, 8, 32));
    return;
  }
  out.push_back(make_norm_quant(512, 4096, 128));
  out.push_back(make_mxfp8_gemm(16, 128, 256));
  if (ctx.preset == Preset::kComprehensive) {
    out.push_back(make_norm_quant(1024, 8192, 128));
    out.push_back(make_mxfp8_gemm(64, 512, 1024));
  }
}

}  // namespace qcb
