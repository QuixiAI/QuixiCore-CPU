// Focused optimization evidence for universal FP16/BF16 storage fallback.

#include "quixicore_cpu/float_storage.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "harness/alloc.h"
#include "harness/case.h"
#include "harness/donotopt.h"
#include "kernels/common/fp16.h"
#include "quixicore_cpu/ops.h"

namespace qcb {
namespace {

using quixicore_cpu::FloatStorageType;
using quixicore_cpu::Status;

float local_bf16_to_float(std::uint16_t bits) {
  const std::uint32_t value = static_cast<std::uint32_t>(bits) << 16;
  float out;
  std::memcpy(&out, &value, sizeof out);
  return out;
}

std::uint16_t local_float_to_bf16(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof bits);
  if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0) {
    return static_cast<std::uint16_t>((bits >> 16) | 0x0040u);
  }
  return static_cast<std::uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >>
                                    16);
}

std::uint16_t encode_one(FloatStorageType type, float value) {
  return type == FloatStorageType::kF16 ? quixicore_cpu::fp32_to_fp16(value)
                                        : local_float_to_bf16(value);
}

struct ConvertBuffers {
  AlignedBuffer<std::uint16_t> input;
  AlignedBuffer<std::uint16_t> output;
  AlignedBuffer<float> scratch;
  long long count = 0;
  FloatStorageType type = FloatStorageType::kF16;
};

void scalar_roundtrip(const ConvertBuffers& buffers) {
  auto run = [&]<class Decode, class Encode>(Decode decode, Encode encode) {
    for (long long i = 0; i < buffers.count; ++i) {
      buffers.scratch.get()[i] = decode(buffers.input.get()[i]);
    }
    for (long long i = 0; i < buffers.count; ++i) {
      buffers.output.get()[i] = encode(buffers.scratch.get()[i]);
    }
  };
  if (buffers.type == FloatStorageType::kF16) {
    run([](std::uint16_t bits) { return quixicore_cpu::fp16_to_fp32(bits); },
        [](float value) { return quixicore_cpu::fp32_to_fp16(value); });
  } else {
    run(local_bf16_to_float, local_float_to_bf16);
  }
}

void unrolled_roundtrip(const ConvertBuffers& buffers) {
  auto run = [&]<class Decode, class Encode>(Decode decode, Encode encode) {
    long long i = 0;
    for (; i + 3 < buffers.count; i += 4) {
      buffers.scratch.get()[i] = decode(buffers.input.get()[i]);
      buffers.scratch.get()[i + 1] = decode(buffers.input.get()[i + 1]);
      buffers.scratch.get()[i + 2] = decode(buffers.input.get()[i + 2]);
      buffers.scratch.get()[i + 3] = decode(buffers.input.get()[i + 3]);
    }
    for (; i < buffers.count; ++i) {
      buffers.scratch.get()[i] = decode(buffers.input.get()[i]);
    }
    i = 0;
    for (; i + 3 < buffers.count; i += 4) {
      buffers.output.get()[i] = encode(buffers.scratch.get()[i]);
      buffers.output.get()[i + 1] = encode(buffers.scratch.get()[i + 1]);
      buffers.output.get()[i + 2] = encode(buffers.scratch.get()[i + 2]);
      buffers.output.get()[i + 3] = encode(buffers.scratch.get()[i + 3]);
    }
    for (; i < buffers.count; ++i) {
      buffers.output.get()[i] = encode(buffers.scratch.get()[i]);
    }
  };
  if (buffers.type == FloatStorageType::kF16) {
    run([](std::uint16_t bits) { return quixicore_cpu::fp16_to_fp32(bits); },
        [](float value) { return quixicore_cpu::fp32_to_fp16(value); });
  } else {
    run(local_bf16_to_float, local_float_to_bf16);
  }
}

CaseDecl make_conversion(FloatStorageType type, long long count) {
  const std::string name = type == FloatStorageType::kF16 ? "f16" : "bf16";
  CaseDecl decl;
  decl.kernel = "float_storage";
  decl.variant = name + "_roundtrip_N" + std::to_string(count);
  decl.shape = {{"elements", count}};
  decl.dtype = name + "/f32/" + name;
  decl.notes =
      "parallel portable conversion; baselines preserve scalar and manual "
      "four-way optimization passes";
  decl.make = [type, count]() {
    auto buffers = std::make_shared<ConvertBuffers>();
    buffers->count = count;
    buffers->type = type;
    buffers->input = aligned_alloc_array<std::uint16_t>(count);
    buffers->output = aligned_alloc_array<std::uint16_t>(count);
    buffers->scratch = aligned_alloc_array<float>(count);
    for (long long i = 0; i < count; ++i) {
      const float value =
          8.0f * std::sin(static_cast<float>(i) * 0.0009765625f);
      buffers->input.get()[i] = encode_one(type, value);
    }
    CaseBody body;
    body.target = [buffers]() {
      if (quixicore_cpu::float_storage_to_f32(
              buffers->type, buffers->input.get(), buffers->scratch.get(),
              buffers->count) != Status::kOk ||
          quixicore_cpu::float_storage_from_f32(
              buffers->type, buffers->scratch.get(), buffers->output.get(),
              buffers->count) != Status::kOk) {
        throw std::runtime_error("float storage conversion failed");
      }
      do_not_optimize(buffers->output.get());
    };
    body.baselines.emplace_back("scalar_pass1", [buffers]() {
      scalar_roundtrip(*buffers);
      do_not_optimize(buffers->output.get());
    });
    body.baselines.emplace_back("unrolled4_pass2", [buffers]() {
      unrolled_roundtrip(*buffers);
      do_not_optimize(buffers->output.get());
    });
    body.check = [buffers]() {
      if (quixicore_cpu::float_storage_to_f32(
              buffers->type, buffers->input.get(), buffers->scratch.get(),
              buffers->count) != Status::kOk ||
          quixicore_cpu::float_storage_from_f32(
              buffers->type, buffers->scratch.get(), buffers->output.get(),
              buffers->count) != Status::kOk) {
        throw std::runtime_error("float storage conversion failed");
      }
      CheckResult result;
      for (long long i = 0; i < buffers->count; ++i) {
        if (buffers->output.get()[i] != buffers->input.get()[i]) {
          result.passed = false;
          result.max_abs_err = 1.0;
          result.max_rel_err = 1.0;
          break;
        }
      }
      return result;
    };
    return body;
  };
  return decl;
}

struct SoftmaxBuffers {
  AlignedBuffer<std::uint16_t> input;
  AlignedBuffer<std::uint16_t> output;
  AlignedBuffer<float> decoded;
  AlignedBuffer<float> result;
  long long rows = 0;
  long long dim = 0;
};

CaseDecl make_typed_softmax(long long rows, long long dim) {
  const long long count = rows * dim;
  CaseDecl decl;
  decl.kernel = "float_storage";
  decl.variant =
      "bf16_softmax_R" + std::to_string(rows) + "_D" + std::to_string(dim);
  decl.shape = {{"rows", rows}, {"hidden", dim}};
  decl.dtype = "bf16/f32/bf16";
  decl.notes = "generic typed dispatch versus explicitly staged conversion";
  decl.bytes_moved = 12.0 * static_cast<double>(count);
  decl.make = [rows, dim, count]() {
    auto buffers = std::make_shared<SoftmaxBuffers>();
    buffers->rows = rows;
    buffers->dim = dim;
    buffers->input = aligned_alloc_array<std::uint16_t>(count);
    buffers->output = aligned_alloc_array<std::uint16_t>(count);
    buffers->decoded = aligned_alloc_array<float>(count);
    buffers->result = aligned_alloc_array<float>(count);
    for (long long i = 0; i < count; ++i) {
      buffers->input.get()[i] =
          local_float_to_bf16(std::sin(static_cast<float>(i) * 0.01f));
    }
    CaseBody body;
    body.target = [buffers, count]() {
      if (quixicore_cpu::softmax_storage(
              {buffers->input.get(), FloatStorageType::kBF16, count},
              {buffers->output.get(), FloatStorageType::kBF16, count},
              buffers->rows, buffers->dim) != Status::kOk) {
        throw std::runtime_error("typed softmax failed");
      }
      do_not_optimize(buffers->output.get());
    };
    body.baselines.emplace_back("explicit_staging", [buffers, count]() {
      if (quixicore_cpu::float_storage_to_f32(
              FloatStorageType::kBF16, buffers->input.get(),
              buffers->decoded.get(), count) != Status::kOk ||
          quixicore_cpu::softmax(buffers->decoded.get(), buffers->result.get(),
                                 buffers->rows, buffers->dim) != Status::kOk ||
          quixicore_cpu::float_storage_from_f32(
              FloatStorageType::kBF16, buffers->result.get(),
              buffers->output.get(), count) != Status::kOk) {
        throw std::runtime_error("explicit softmax staging failed");
      }
      do_not_optimize(buffers->output.get());
    });
    body.check = [buffers, count]() {
      if (quixicore_cpu::softmax_storage(
              {buffers->input.get(), FloatStorageType::kBF16, count},
              {buffers->output.get(), FloatStorageType::kBF16, count},
              buffers->rows, buffers->dim) != Status::kOk) {
        throw std::runtime_error("typed softmax failed");
      }
      CheckResult result;
      for (long long row = 0; row < buffers->rows; ++row) {
        double sum = 0.0;
        for (long long column = 0; column < buffers->dim; ++column) {
          sum += local_bf16_to_float(
              buffers->output.get()[row * buffers->dim + column]);
        }
        check_value(result, sum, 1.0, Tolerance{0.002, 0.002});
      }
      return result;
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_float_storage_cases(const BuildCtx& ctx,
                               std::vector<CaseDecl>& out) {
  const long long count = ctx.preset == Preset::kSmoke   ? (1LL << 16)
                          : ctx.preset == Preset::kQuick ? (1LL << 20)
                                                         : (1LL << 24);
  out.push_back(make_conversion(FloatStorageType::kF16, count));
  out.push_back(make_conversion(FloatStorageType::kBF16, count));
  if (ctx.preset == Preset::kSmoke) {
    out.push_back(make_typed_softmax(4, 256));
  } else if (ctx.preset == Preset::kQuick) {
    out.push_back(make_typed_softmax(256, 4096));
  } else {
    out.push_back(make_typed_softmax(1024, 4096));
  }
}

}  // namespace qcb
