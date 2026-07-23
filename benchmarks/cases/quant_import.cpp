// Canonical quant lifecycle evidence. Conversion and checkpoint ingestion are
// intentionally timed; prepared weights are constructed inside their case so
// conversion cost and memory amplification remain visible.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness/case.h"
#include "harness/donotopt.h"
#include "quixicore_cpu/packed_weights.h"
#include "quixicore_cpu/quant_import.h"

namespace qcb {
namespace {

using quixicore_cpu::CanonicalQuantLayout;
using quixicore_cpu::CanonicalQuantTensor;
using quixicore_cpu::Status;

const char* layout_name(CanonicalQuantLayout layout) {
  return quixicore_cpu::canonical_quant_layout_name(layout);
}

long long layout_group(CanonicalQuantLayout layout, long long columns) {
  switch (layout) {
    case CanonicalQuantLayout::kFP8E4M3FN:
    case CanonicalQuantLayout::kFP8E5M2:
      return 0;
    case CanonicalQuantLayout::kInt8Affine:
      return columns;
    case CanonicalQuantLayout::kNVFP4E2M1E4M3:
      return 16;
    case CanonicalQuantLayout::kMXFP8E4M3E8M0:
    case CanonicalQuantLayout::kMXFP4E2M1E8M0:
    case CanonicalQuantLayout::kBitNetTernary:
      return 32;
    default:
      return std::min<long long>(128, columns);
  }
}

double layout_rmse_limit(CanonicalQuantLayout layout) {
  switch (layout) {
    case CanonicalQuantLayout::kInt8Symmetric:
    case CanonicalQuantLayout::kInt8Affine:
      return 0.02;
    case CanonicalQuantLayout::kFP8E4M3FN:
    case CanonicalQuantLayout::kFP8E5M2:
    case CanonicalQuantLayout::kMXFP8E4M3E8M0:
      return 0.08;
    case CanonicalQuantLayout::kBitNetTernary:
      return 0.8;
    default:
      return 0.25;
  }
}

struct CanonicalBuffers {
  std::vector<float> input;
  std::vector<float> decoded;
  CanonicalQuantTensor tensor;
  long long rows = 0;
  long long columns = 0;
  long long group_size = 0;
  CanonicalQuantLayout layout = CanonicalQuantLayout::kInt4Symmetric;
};

CaseDecl make_canonical_pack(CanonicalQuantLayout layout, long long rows,
                             long long columns) {
  CaseDecl decl;
  decl.kernel = "quant_import";
  decl.variant = std::string("canonical_pack_") + layout_name(layout) +
                 "_R" + std::to_string(rows) + "_K" +
                 std::to_string(columns);
  decl.shape = {{"rows", rows}, {"k", columns}};
  decl.format = layout_name(layout);
  decl.dtype = "f32";
  decl.notes = "owned canonical pack including logical side metadata";
  decl.bytes_moved = static_cast<double>(rows) * columns * sizeof(float);
  decl.make = [layout, rows, columns]() {
    auto buffers = std::make_shared<CanonicalBuffers>();
    buffers->rows = rows;
    buffers->columns = columns;
    buffers->group_size = layout_group(layout, columns);
    buffers->layout = layout;
    buffers->input.resize(static_cast<std::size_t>(rows * columns));
    buffers->decoded.resize(buffers->input.size());
    for (std::size_t index = 0; index < buffers->input.size(); ++index) {
      buffers->input[index] =
          std::sin(0.0031f * static_cast<float>(index)) +
          0.35f * std::cos(0.00073f * static_cast<float>(index));
    }
    CaseBody body;
    body.target = [buffers]() {
      if (quixicore_cpu::quantize_canonical(
              {buffers->input.data(), quixicore_cpu::FloatStorageType::kF32,
               static_cast<long long>(buffers->input.size())},
              buffers->rows, buffers->columns, buffers->layout,
              buffers->group_size, &buffers->tensor) != Status::kOk) {
        throw std::runtime_error("canonical quant pack failed");
      }
      do_not_optimize(buffers->tensor.data.data());
    };
    body.check = [buffers]() {
      const Status pack_status = quixicore_cpu::quantize_canonical(
              {buffers->input.data(), quixicore_cpu::FloatStorageType::kF32,
               static_cast<long long>(buffers->input.size())},
              buffers->rows, buffers->columns, buffers->layout,
              buffers->group_size, &buffers->tensor);
      if (pack_status != Status::kOk) {
        throw std::runtime_error("canonical quant lifecycle pack check failed: " +
                                 std::to_string(static_cast<int>(pack_status)));
      }
      const Status unpack_status = quixicore_cpu::dequantize_canonical(
              buffers->tensor, buffers->decoded.data(),
              static_cast<long long>(buffers->decoded.size()));
      if (unpack_status != Status::kOk) {
        throw std::runtime_error(
            "canonical quant lifecycle unpack check failed: " +
            std::to_string(static_cast<int>(unpack_status)));
      }
      CheckResult check;
      double squared_error = 0.0;
      for (std::size_t index = 0; index < buffers->input.size(); ++index) {
        if (!std::isfinite(buffers->decoded[index])) {
          check.passed = false;
          check.finite = false;
          return check;
        }
        const double error = buffers->decoded[index] - buffers->input[index];
        squared_error += error * error;
        check.max_abs_err = std::max(check.max_abs_err, std::fabs(error));
      }
      const double rmse =
          std::sqrt(squared_error / static_cast<double>(buffers->input.size()));
      check.max_rel_err = rmse;
      check.passed = rmse <= layout_rmse_limit(buffers->layout);
      return check;
    };
    return body;
  };
  return decl;
}

struct ImportBuffers {
  std::vector<std::uint32_t> qweight;
  std::vector<std::uint32_t> qzeros;
  std::vector<float> scales;
  std::vector<std::int8_t> int8_weights;
  std::vector<float> activation_scales;
  CanonicalQuantTensor tensor;
  long long n = 0;
  long long k = 0;
  long long group_size = 0;
};

CaseDecl make_awq_import(long long n, long long k) {
  CaseDecl decl;
  decl.kernel = "quant_import";
  decl.variant = "awq_u4_import_N" + std::to_string(n) + "_K" +
                 std::to_string(k);
  decl.shape = {{"n", n}, {"k", k}, {"group", 128}};
  decl.format = "awq_u4";
  decl.notes = "AWQ int32 checkpoint fragment to canonical row-major U4";
  decl.bytes_moved = static_cast<double>(n) * k / 2.0;
  decl.make = [n, k]() {
    auto buffers = std::make_shared<ImportBuffers>();
    buffers->n = n;
    buffers->k = k;
    buffers->group_size = 128;
    buffers->qweight.resize(static_cast<std::size_t>(k * (n / 8)));
    buffers->qzeros.resize(
        static_cast<std::size_t>((k / buffers->group_size) * (n / 8)));
    buffers->scales.resize(
        static_cast<std::size_t>((k / buffers->group_size) * n));
    for (std::size_t index = 0; index < buffers->qweight.size(); ++index)
      buffers->qweight[index] = static_cast<std::uint32_t>(0x76543210U ^ index);
    std::fill(buffers->qzeros.begin(), buffers->qzeros.end(), 0x33333333U);
    std::fill(buffers->scales.begin(), buffers->scales.end(), 0.03125f);
    auto invoke = [buffers]() {
      quixicore_cpu::AwqU4Source source;
      source.qweight = buffers->qweight.data();
      source.qweight_words = buffers->qweight.size();
      source.qzeros = buffers->qzeros.data();
      source.qzero_words = buffers->qzeros.size();
      source.scales = buffers->scales.data();
      source.scale_count = buffers->scales.size();
      source.input_features = buffers->k;
      source.output_features = buffers->n;
      source.group_size = buffers->group_size;
      return quixicore_cpu::import_awq_u4(source, &buffers->tensor);
    };
    CaseBody body;
    body.target = [buffers, invoke]() {
      if (invoke() != Status::kOk) throw std::runtime_error("AWQ import failed");
      do_not_optimize(buffers->tensor.data.data());
    };
    body.check = [buffers, invoke]() {
      CheckResult check;
      check.passed = invoke() == Status::kOk &&
                     quixicore_cpu::validate_canonical_quant_tensor(
                         buffers->tensor) == Status::kOk;
      return check;
    };
    return body;
  };
  return decl;
}

CaseDecl make_gptq_import(long long n, long long k) {
  CaseDecl decl;
  decl.kernel = "quant_import";
  decl.variant = "gptq_v1_u4_import_N" + std::to_string(n) + "_K" +
                 std::to_string(k);
  decl.shape = {{"n", n}, {"k", k}, {"group", 128}};
  decl.format = "gptq_u4";
  decl.notes = "GPTQ v1 checkpoint fragment with stored zero+1 conversion";
  decl.bytes_moved = static_cast<double>(n) * k / 2.0;
  decl.make = [n, k]() {
    auto buffers = std::make_shared<ImportBuffers>();
    buffers->n = n;
    buffers->k = k;
    buffers->group_size = 128;
    buffers->qweight.resize(static_cast<std::size_t>((k / 8) * n));
    buffers->qzeros.resize(
        static_cast<std::size_t>((k / buffers->group_size) * (n / 8)));
    buffers->scales.resize(
        static_cast<std::size_t>((k / buffers->group_size) * n));
    for (std::size_t index = 0; index < buffers->qweight.size(); ++index)
      buffers->qweight[index] = static_cast<std::uint32_t>(0x76543210U ^ index);
    std::fill(buffers->qzeros.begin(), buffers->qzeros.end(), 0x22222222U);
    std::fill(buffers->scales.begin(), buffers->scales.end(), 0.03125f);
    auto invoke = [buffers]() {
      quixicore_cpu::GptqU4Source source;
      source.qweight = buffers->qweight.data();
      source.qweight_words = buffers->qweight.size();
      source.qzeros = buffers->qzeros.data();
      source.qzero_words = buffers->qzeros.size();
      source.scales = buffers->scales.data();
      source.scale_count = buffers->scales.size();
      source.input_features = buffers->k;
      source.output_features = buffers->n;
      source.group_size = buffers->group_size;
      return quixicore_cpu::import_gptq_u4(source, &buffers->tensor);
    };
    CaseBody body;
    body.target = [buffers, invoke]() {
      if (invoke() != Status::kOk)
        throw std::runtime_error("GPTQ import failed");
      do_not_optimize(buffers->tensor.data.data());
    };
    body.check = [buffers, invoke]() {
      CheckResult check;
      check.passed = invoke() == Status::kOk &&
                     buffers->tensor.zero_points.front() == 3.0f;
      return check;
    };
    return body;
  };
  return decl;
}

CaseDecl make_smoothquant_import(long long n, long long k) {
  CaseDecl decl;
  decl.kernel = "quant_import";
  decl.variant = "smoothquant_w8a8_import_N" + std::to_string(n) + "_K" +
                 std::to_string(k);
  decl.shape = {{"n", n}, {"k", k}};
  decl.format = "smoothquant_w8a8_azp";
  decl.notes = "SmoothQuant weights, activation metadata, and row-sum prepare";
  decl.bytes_moved = static_cast<double>(n) * k;
  decl.make = [n, k]() {
    auto buffers = std::make_shared<ImportBuffers>();
    buffers->n = n;
    buffers->k = k;
    buffers->int8_weights.resize(static_cast<std::size_t>(n * k));
    buffers->scales.resize(static_cast<std::size_t>(n));
    buffers->activation_scales = {0.015625f};
    for (std::size_t index = 0; index < buffers->int8_weights.size(); ++index)
      buffers->int8_weights[index] =
          static_cast<std::int8_t>(static_cast<int>(index % 251) - 125);
    std::fill(buffers->scales.begin(), buffers->scales.end(), 0.0078125f);
    auto invoke = [buffers]() {
      quixicore_cpu::SmoothQuantW8A8Source source;
      source.weights = buffers->int8_weights.data();
      source.weight_count = buffers->int8_weights.size();
      source.weight_scales = buffers->scales.data();
      source.weight_scale_count = buffers->scales.size();
      source.activation_scales = buffers->activation_scales.data();
      source.activation_scale_count = buffers->activation_scales.size();
      source.rows = buffers->n;
      source.columns = buffers->k;
      return quixicore_cpu::import_smoothquant_w8a8(source, &buffers->tensor);
    };
    CaseBody body;
    body.target = [buffers, invoke]() {
      if (invoke() != Status::kOk)
        throw std::runtime_error("SmoothQuant import failed");
      do_not_optimize(buffers->tensor.row_sums.data());
    };
    body.check = [buffers, invoke]() {
      CheckResult check;
      check.passed = invoke() == Status::kOk &&
                     buffers->tensor.row_sums.size() ==
                         static_cast<std::size_t>(buffers->n);
      return check;
    };
    return body;
  };
  return decl;
}

struct PrepareBuffers {
  CanonicalQuantTensor tensor;
  quixicore_cpu::CpuPackedWeights prepared;
};

CaseDecl make_canonical_prepare(long long rows, long long columns) {
  CaseDecl decl;
  decl.kernel = "quant_import";
  decl.variant = "canonical_prepare_mxfp4_R" + std::to_string(rows) + "_K" +
                 std::to_string(columns);
  decl.shape = {{"rows", rows}, {"k", columns}};
  decl.format = "mxfp4_e2m1_e8m0";
  decl.notes = "canonical bytes to reusable CPU-private row panels";
  decl.make = [rows, columns]() {
    auto buffers = std::make_shared<PrepareBuffers>();
    std::vector<float> input(static_cast<std::size_t>(rows * columns));
    for (std::size_t index = 0; index < input.size(); ++index)
      input[index] = std::sin(0.001f * static_cast<float>(index));
    if (quixicore_cpu::quantize_canonical(
            {input.data(), quixicore_cpu::FloatStorageType::kF32,
             static_cast<long long>(input.size())},
            rows, columns, CanonicalQuantLayout::kMXFP4E2M1E8M0, 32,
            &buffers->tensor) != Status::kOk) {
      throw std::runtime_error("prepare benchmark setup failed");
    }
    CaseBody body;
    body.target = [buffers]() {
      if (buffers->prepared.prepare(buffers->tensor) != Status::kOk)
        throw std::runtime_error("canonical prepare failed");
      do_not_optimize(buffers->prepared.panel_data());
    };
    body.check = [buffers]() {
      CheckResult check;
      if (buffers->prepared.prepare(buffers->tensor) != Status::kOk) {
        check.passed = false;
        return check;
      }
      const auto info = buffers->prepared.info();
      check.passed = info.has_canonical_layout &&
                     info.canonical_layout ==
                         CanonicalQuantLayout::kMXFP4E2M1E8M0 &&
                     std::memcmp(buffers->prepared.contract_data(),
                                 buffers->tensor.data.data(),
                                 buffers->tensor.data.size()) == 0;
      check.max_abs_err = info.memory_amplification;
      return check;
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_quant_import_cases(const BuildCtx& ctx,
                              std::vector<CaseDecl>& out) {
  const long long rows = ctx.preset == Preset::kSmoke ? 4 : 128;
  const long long columns = ctx.preset == Preset::kSmoke ? 256 : 1024;
  for (CanonicalQuantLayout layout : {
           CanonicalQuantLayout::kInt4Symmetric,
           CanonicalQuantLayout::kUInt4Affine,
           CanonicalQuantLayout::kInt8Symmetric,
           CanonicalQuantLayout::kInt8Affine,
           CanonicalQuantLayout::kFP8E4M3FN,
           CanonicalQuantLayout::kFP8E5M2,
           CanonicalQuantLayout::kFP4E2M1,
           CanonicalQuantLayout::kMXFP8E4M3E8M0,
           CanonicalQuantLayout::kMXFP4E2M1E8M0,
           CanonicalQuantLayout::kNVFP4E2M1E4M3,
           CanonicalQuantLayout::kBitNetTernary,
       }) {
    out.push_back(make_canonical_pack(layout, rows, columns));
  }
  const long long import_rows = std::max<long long>(8, rows);
  out.push_back(make_awq_import(import_rows, columns));
  out.push_back(make_gptq_import(import_rows, columns));
  out.push_back(make_smoothquant_import(rows, columns));
  out.push_back(make_canonical_prepare(rows, columns));
}

}  // namespace qcb
