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
#include "quixicore_cpu/quant_import.h"
#include "quixicore_cpu/quantization.h"

namespace qcb {
namespace {

using quixicore_cpu::CanonicalQuantLayout;
using quixicore_cpu::CanonicalQuantTensor;
using quixicore_cpu::FloatStorageType;
using quixicore_cpu::Status;

struct NormFormat {
  CanonicalQuantLayout layout;
  long long group_size;
  bool scale_2d;
  const char* name;
};

struct NormBuffers {
  AlignedBuffer<float> x;
  AlignedBuffer<float> residual;
  AlignedBuffer<float> weight;
  AlignedBuffer<float> bias;
  AlignedBuffer<float> residual_out;
  AlignedBuffer<float> reference_residual;
  AlignedBuffer<float> normalized;
  CanonicalQuantTensor output;
  CanonicalQuantTensor reference;
  long long rows = 0;
  long long hidden = 0;
  NormFormat format{};
  bool layer_norm = false;
};

void fill_inputs(NormBuffers& buffers) {
  const long long count = buffers.rows * buffers.hidden;
  for (long long index = 0; index < count; ++index) {
    const int x_code = static_cast<int>((index * 37 + 11) % 101) - 50;
    const int r_code = static_cast<int>((index * 19 + 7) % 89) - 44;
    buffers.x.get()[index] = static_cast<float>(x_code) / 47.0f;
    buffers.residual.get()[index] = static_cast<float>(r_code) / 211.0f;
  }
  for (long long item = 0; item < buffers.hidden; ++item) {
    buffers.weight.get()[item] =
        0.75f + 0.2f * std::sin(static_cast<float>(item) * 0.03125f);
    buffers.bias.get()[item] =
        0.03f * std::cos(static_cast<float>(item) * 0.0625f);
  }
}

void run_target(NormBuffers& buffers) {
  const long long count = buffers.rows * buffers.hidden;
  const Status status =
      buffers.layer_norm
          ? quixicore_cpu::layer_norm_add_quantized_storage(
                {buffers.x.get(), FloatStorageType::kF32, count},
                {buffers.residual.get(), FloatStorageType::kF32, count},
                {buffers.weight.get(), FloatStorageType::kF32, buffers.hidden},
                {buffers.bias.get(), FloatStorageType::kF32, buffers.hidden},
                {buffers.residual_out.get(), FloatStorageType::kF32, count},
                buffers.format.layout, buffers.format.group_size,
                &buffers.output, buffers.rows, buffers.hidden, 1e-5f,
                buffers.format.scale_2d)
          : quixicore_cpu::rms_norm_add_quantized_storage(
                {buffers.x.get(), FloatStorageType::kF32, count},
                {buffers.residual.get(), FloatStorageType::kF32, count},
                {buffers.weight.get(), FloatStorageType::kF32, buffers.hidden},
                {buffers.residual_out.get(), FloatStorageType::kF32, count},
                buffers.format.layout, buffers.format.group_size,
                &buffers.output, buffers.rows, buffers.hidden, 1e-5f,
                buffers.format.scale_2d);
  if (status != Status::kOk) {
    throw std::runtime_error("canonical fused norm-add-quant failed");
  }
  do_not_optimize(buffers.output.data.data());
  do_not_optimize(buffers.residual_out.get());
}

void run_composed(NormBuffers& buffers) {
  const long long count = buffers.rows * buffers.hidden;
  const Status norm_status =
      buffers.layer_norm
          ? quixicore_cpu::layer_norm_add(
                buffers.x.get(), buffers.residual.get(), buffers.weight.get(),
                buffers.bias.get(), buffers.normalized.get(),
                buffers.reference_residual.get(), buffers.rows, buffers.hidden)
          : quixicore_cpu::rms_norm_add(
                buffers.x.get(), buffers.residual.get(), buffers.weight.get(),
                buffers.normalized.get(), buffers.reference_residual.get(),
                buffers.rows, buffers.hidden);
  if (norm_status != Status::kOk ||
      quixicore_cpu::quantize_canonical(
          {buffers.normalized.get(), FloatStorageType::kF32, count},
          buffers.rows, buffers.hidden, buffers.format.layout,
          buffers.format.group_size, &buffers.reference,
          buffers.format.scale_2d) != Status::kOk) {
    throw std::runtime_error("composed norm-add then canonical quant failed");
  }
  do_not_optimize(buffers.reference.data.data());
  do_not_optimize(buffers.reference_residual.get());
}

CaseDecl make_norm_case(NormFormat format, bool layer_norm, long long rows,
                        long long hidden) {
  CaseDecl decl;
  decl.kernel = "quant_norm";
  decl.variant = std::string(layer_norm ? "layer_add_" : "rms_add_") +
                 format.name + "_R" + std::to_string(rows) + "_H" +
                 std::to_string(hidden);
  decl.shape = {
      {"rows", rows}, {"hidden", hidden}, {"group_size", format.group_size}};
  decl.dtype = "f32";
  decl.format = format.name;
  decl.notes =
      "M3 F6/F7 row-local norm-add with direct canonical activation packing";
  decl.bytes_moved =
      static_cast<double>(rows * hidden) * (layer_norm ? 22.0 : 18.0);
  decl.make = [format, layer_norm, rows, hidden]() {
    auto buffers = std::make_shared<NormBuffers>();
    buffers->rows = rows;
    buffers->hidden = hidden;
    buffers->format = format;
    buffers->layer_norm = layer_norm;
    const long long count = rows * hidden;
    buffers->x = aligned_alloc_array<float>(count);
    buffers->residual = aligned_alloc_array<float>(count);
    buffers->weight = aligned_alloc_array<float>(hidden);
    buffers->bias = aligned_alloc_array<float>(hidden);
    buffers->residual_out = aligned_alloc_array<float>(count);
    buffers->reference_residual = aligned_alloc_array<float>(count);
    buffers->normalized = aligned_alloc_array<float>(count);
    fill_inputs(*buffers);

    CaseBody body;
    body.target = [buffers]() { run_target(*buffers); };
    body.baselines.emplace_back("norm_then_quant_preallocated",
                                [buffers]() { run_composed(*buffers); });
    body.check = [buffers]() {
      run_target(*buffers);
      run_composed(*buffers);
      CheckResult check;
      const long long count = buffers->rows * buffers->hidden;
      std::vector<float> actual(static_cast<std::size_t>(count));
      std::vector<float> expected(static_cast<std::size_t>(count));
      if (quixicore_cpu::dequantize_canonical(buffers->output, actual.data(),
                                              count) != Status::kOk ||
          quixicore_cpu::dequantize_canonical(
              buffers->reference, expected.data(), count) != Status::kOk) {
        check.passed = false;
        return check;
      }
      for (long long index = 0; index < count; ++index) {
        check_value(check, actual[static_cast<std::size_t>(index)],
                    expected[static_cast<std::size_t>(index)],
                    Tolerance{4e-2, 4e-2});
        check_value(check, buffers->residual_out.get()[index],
                    buffers->reference_residual.get()[index], kFp32Tolerance);
      }
      return check;
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_quant_norm_cases(const BuildCtx& ctx, std::vector<CaseDecl>& out) {
  const NormFormat int4{CanonicalQuantLayout::kInt4Symmetric, 128, false,
                        "int4"};
  const NormFormat int8{CanonicalQuantLayout::kInt8Symmetric, 128, false,
                        "int8"};
  const NormFormat fp8{CanonicalQuantLayout::kFP8E4M3FN, 128, false,
                       "fp8_e4m3"};
  const NormFormat fp4{CanonicalQuantLayout::kFP4E2M1, 128, false, "fp4"};
  const NormFormat mxfp8{CanonicalQuantLayout::kMXFP8E4M3E8M0, 32, false,
                         "mxfp8"};
  const NormFormat mxfp4{CanonicalQuantLayout::kMXFP4E2M1E8M0, 32, false,
                         "mxfp4"};
  const NormFormat nvfp4{CanonicalQuantLayout::kNVFP4E2M1E4M3, 16, true,
                         "nvfp4_2d"};
  if (ctx.preset == Preset::kSmoke) {
    out.push_back(make_norm_case(int8, false, 4, 256));
    out.push_back(make_norm_case(fp8, true, 4, 256));
    return;
  }
  const long long rows = ctx.preset == Preset::kQuick ? 128 : 512;
  const long long hidden = ctx.preset == Preset::kQuick ? 2048 : 4096;
  for (NormFormat format : {int4, int8, fp8, fp4, mxfp8, mxfp4, nvfp4}) {
    out.push_back(make_norm_case(format, false, rows, hidden));
  }
  for (NormFormat format : {int4, int8, fp8, fp4, mxfp4, nvfp4}) {
    out.push_back(make_norm_case(format, true, rows, hidden));
  }
}

}  // namespace qcb
