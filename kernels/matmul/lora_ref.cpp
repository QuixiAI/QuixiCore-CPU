#include <cmath>
#include <cstdint>

#include "kernels/common/fp16.h"
#include "kernels/common/validation.h"
#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/threading.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

#if defined(__x86_64__) || defined(_M_X64)
constexpr bool kUseX86Accumulators = true;
#else
constexpr bool kUseX86Accumulators = false;
#endif

inline void lora_low(const float* x, const std::uint16_t* adapter_a,
                     std::uint16_t* low, long long input_dim, long long rank) {
  for (long long adapter_row = 0; adapter_row < rank; ++adapter_row) {
#if defined(__x86_64__) || defined(_M_X64)
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    const std::uint16_t* adapter = adapter_a + adapter_row * input_dim;
    long long input = 0;
    for (; input + 3 < input_dim; input += 4) {
      sum0 += x[input] * fp16_to_fp32(adapter[input]);
      sum1 += x[input + 1] * fp16_to_fp32(adapter[input + 1]);
      sum2 += x[input + 2] * fp16_to_fp32(adapter[input + 2]);
      sum3 += x[input + 3] * fp16_to_fp32(adapter[input + 3]);
    }
    float sum = (sum0 + sum1) + (sum2 + sum3);
    for (; input < input_dim; ++input) {
      sum += x[input] * fp16_to_fp32(adapter[input]);
    }
#else
    float sum = 0.0f;
    const std::uint16_t* adapter = adapter_a + adapter_row * input_dim;
    for (long long input = 0; input < input_dim; ++input) {
      sum += x[input] * fp16_to_fp32(adapter[input]);
    }
#endif
    low[adapter_row] = fp32_to_fp16(sum);
  }
}

inline void lora_outputs(const std::uint16_t* low,
                         const std::uint16_t* adapter_b, const float* base,
                         float* out, long long rank, float scale,
                         long long begin, long long end) {
  for (long long output = begin; output < end; ++output) {
    const std::uint16_t* adapter = adapter_b + output * rank;
    float sum = 0.0f;
    if (rank == 16 && kUseX86Accumulators) {
      float sum0 = 0.0f;
      float sum1 = 0.0f;
      float sum2 = 0.0f;
      float sum3 = 0.0f;
      for (long long adapter_row = 0; adapter_row < 16; adapter_row += 4) {
        sum0 += fp16_to_fp32(low[adapter_row]) *
                fp16_to_fp32(adapter[adapter_row]);
        sum1 += fp16_to_fp32(low[adapter_row + 1]) *
                fp16_to_fp32(adapter[adapter_row + 1]);
        sum2 += fp16_to_fp32(low[adapter_row + 2]) *
                fp16_to_fp32(adapter[adapter_row + 2]);
        sum3 += fp16_to_fp32(low[adapter_row + 3]) *
                fp16_to_fp32(adapter[adapter_row + 3]);
      }
      sum = (sum0 + sum1) + (sum2 + sum3);
    } else {
      for (long long adapter_row = 0; adapter_row < rank; ++adapter_row) {
        sum += fp16_to_fp32(low[adapter_row]) *
               fp16_to_fp32(adapter[adapter_row]);
      }
    }
    const float delta = fp16_to_fp32(fp32_to_fp16(sum));
    out[output] = (base == nullptr ? 0.0f : base[output]) + scale * delta;
  }
}

inline void lora_row(const float* x, const std::uint16_t* adapter_a,
                     const std::uint16_t* adapter_b, const float* base,
                     float* out, long long input_dim, long long output_dim,
                     long long rank, float scale) {
  std::uint16_t low[256];
  lora_low(x, adapter_a, low, input_dim, rank);
  lora_outputs(low, adapter_b, base, out, rank, scale, 0, output_dim);
}

}  // namespace

Status lora_apply_direct_f16(const float* x,
                             const std::uint16_t* adapter_a,
                             const std::uint16_t* adapter_b,
                             const float* base, float* out, long long rows,
                             long long input_dim, long long output_dim,
                             long long rank, float scale) {
  if (!detail::valid_product({rows, input_dim, output_dim}) || rank < 1 ||
      rank > 256 || !detail::valid_product({rank, input_dim}) ||
      !detail::valid_product({output_dim, rank})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, adapter_a, adapter_b, out) ||
      !std::isfinite(scale)) {
    return Status::kInvalidArgument;
  }
  if (num_threads() == 1) {
    for (long long row = 0; row < rows; ++row) {
      lora_row(x + row * input_dim, adapter_a, adapter_b,
               base == nullptr ? nullptr : base + row * output_dim,
               out + row * output_dim, input_dim, output_dim, rank, scale);
    }
  } else if (rows == 1) {
    std::uint16_t low[256];
    lora_low(x, adapter_a, low, input_dim, rank);
    threading::parallel_ranges(
        output_dim, 128, [&](long long begin, long long end, int) {
          lora_outputs(low, adapter_b, base, out, rank, scale, begin, end);
        });
  } else {
    threading::parallel_ranges(rows, 1,
                               [&](long long begin, long long end, int) {
      for (long long row = begin; row < end; ++row) {
        lora_row(x + row * input_dim, adapter_a, adapter_b,
                 base == nullptr ? nullptr : base + row * output_dim,
                 out + row * output_dim, input_dim, output_dim, rank, scale);
      }
    });
  }
  return Status::kOk;
}

Status lora_apply_direct_f16_storage(
    FloatStorageInput x, const std::uint16_t* adapter_a,
    const std::uint16_t* adapter_b, FloatStorageInput base,
    FloatStorageOutput out, long long rows, long long input_dim,
    long long output_dim, long long rank, float scale,
    FloatStorageWorkspace* workspace) {
  if (!detail::valid_product({rows, input_dim, output_dim}) ||
      x.count != rows * input_dim || out.count != rows * output_dim ||
      (base.data != nullptr && base.count != out.count) ||
      (base.data == nullptr && base.count != 0)) {
    return Status::kInvalidShape;
  }
  if (base.data == nullptr) {
    const FloatStorageInput inputs[] = {x};
    const FloatStorageOutput outputs[] = {out};
    return with_float_storage(
        inputs, 1, outputs, 1,
        [&](const float* const* f32_inputs, float* const* f32_outputs) {
          return lora_apply_direct_f16(
              f32_inputs[0], adapter_a, adapter_b, nullptr, f32_outputs[0],
              rows, input_dim, output_dim, rank, scale);
        },
        workspace);
  }
  const FloatStorageInput inputs[] = {x, base};
  const FloatStorageOutput outputs[] = {out};
  return with_float_storage(
      inputs, 2, outputs, 1,
      [&](const float* const* f32_inputs, float* const* f32_outputs) {
        return lora_apply_direct_f16(
            f32_inputs[0], adapter_a, adapter_b, f32_inputs[1], f32_outputs[0],
            rows, input_dim, output_dim, rank, scale);
      },
      workspace);
}

}  // namespace quixicore_cpu
