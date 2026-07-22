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
        if (buffers->codes.get()[i] != buffers->reference_codes.get()[i]) {
          check.passed = false;
        }
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

}  // namespace

void build_ported_ops_cases(const BuildCtx& ctx, std::vector<CaseDecl>& out) {
  if (ctx.preset == Preset::kSmoke) {
    out.push_back(make_norm_quant(4, 256, 64));
    return;
  }
  out.push_back(make_norm_quant(512, 4096, 128));
  if (ctx.preset == Preset::kComprehensive) {
    out.push_back(make_norm_quant(1024, 8192, 128));
  }
}

}  // namespace qcb
