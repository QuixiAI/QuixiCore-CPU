#include "kernels/quantization/activation_quant.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "kernels/common/fp16.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu::quant {
namespace {

struct BlockQ8_0Activation {
  std::uint16_t d;
  std::int8_t qs[32];
};
static_assert(sizeof(BlockQ8_0Activation) == 34);

struct BlockQ8_1Activation {
  std::uint16_t d;
  std::uint16_t s;
  std::int8_t qs[32];
};
static_assert(sizeof(BlockQ8_1Activation) == 36);

struct BlockQ8KActivation {
  float d;
  std::int8_t qs[256];
  std::int16_t bsums[16];
};
static_assert(sizeof(BlockQ8KActivation) == 292);

int nearest_int(float value) {
  // llama.cpp's exact round-to-nearest-even bit construction for Q8_K.
  value += 12582912.0f;
  int bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x007fffff) - 0x00400000;
}

bool finite_block(const float* input, long long count) {
  for (long long i = 0; i < count; ++i) {
    if (!std::isfinite(input[i])) return false;
  }
  return true;
}

template <typename Block>
bool pack_q8_0_or_q8_1(const float* input, long long blocks, Block* output) {
  for (long long block = 0; block < blocks; ++block) {
    const float* values = input + block * 32;
    if (!finite_block(values, 32)) return false;
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
      amax = std::max(amax, std::fabs(values[i]));
    }
    const float scale = amax / 127.0f;
    const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
    output[block].d = fp32_to_fp16(scale);
    int sum = 0;
    for (int i = 0; i < 32; ++i) {
      const int quant = static_cast<int>(std::round(values[i] * inverse));
      output[block].qs[i] = static_cast<std::int8_t>(quant);
      sum += quant;
    }
    if constexpr (sizeof(Block) == sizeof(BlockQ8_1Activation)) {
      output[block].s = fp32_to_fp16(static_cast<float>(sum) * scale);
    }
  }
  return true;
}

bool pack_q8_k(const float* input, long long blocks,
               BlockQ8KActivation* output) {
  for (long long block = 0; block < blocks; ++block) {
    const float* values = input + block * 256;
    if (!finite_block(values, 256)) return false;
    float signed_maximum = 0.0f;
    float absolute_maximum = 0.0f;
    for (int i = 0; i < 256; ++i) {
      const float magnitude = std::fabs(values[i]);
      if (magnitude > absolute_maximum) {
        absolute_maximum = magnitude;
        signed_maximum = values[i];
      }
    }
    if (absolute_maximum == 0.0f) {
      output[block].d = 0.0f;
      std::fill_n(output[block].qs, 256, std::int8_t{0});
      std::fill_n(output[block].bsums, 16, std::int16_t{0});
      continue;
    }
    const float inverse = -127.0f / signed_maximum;
    for (int i = 0; i < 256; ++i) {
      output[block].qs[i] = static_cast<std::int8_t>(
          std::min(127, nearest_int(inverse * values[i])));
    }
    for (int group = 0; group < 16; ++group) {
      int sum = 0;
      for (int i = 0; i < 16; ++i) {
        sum += output[block].qs[group * 16 + i];
      }
      output[block].bsums[group] = static_cast<std::int16_t>(sum);
    }
    output[block].d = 1.0f / inverse;
  }
  return true;
}

template <typename Block>
void unpack_q8_half(const Block* input, long long blocks, float* output) {
  for (long long block = 0; block < blocks; ++block) {
    const float scale = fp16_to_fp32(input[block].d);
    for (int i = 0; i < 32; ++i) {
      output[block * 32 + i] = scale * input[block].qs[i];
    }
  }
}

}  // namespace

bool activation_format_info(QuantActivationFormat format,
                            long long* block_size,
                            std::size_t* block_bytes) {
  switch (format) {
    case QuantActivationFormat::kQ8_0:
      *block_size = 32;
      *block_bytes = sizeof(BlockQ8_0Activation);
      return true;
    case QuantActivationFormat::kQ8_1:
      *block_size = 32;
      *block_bytes = sizeof(BlockQ8_1Activation);
      return true;
    case QuantActivationFormat::kQ8_K:
      *block_size = 256;
      *block_bytes = sizeof(BlockQ8KActivation);
      return true;
  }
  return false;
}

bool activation_pack_ref(QuantActivationFormat format, const float* input,
                         long long n, long long k, void* packed) {
  long long block_size = 0;
  std::size_t ignored = 0;
  if (!activation_format_info(format, &block_size, &ignored)) return false;
  const long long blocks_per_row = k / block_size;
  std::atomic<bool> valid{true};
  threading::parallel_ranges(n, 16, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      bool row_valid = false;
      switch (format) {
        case QuantActivationFormat::kQ8_0:
          row_valid = pack_q8_0_or_q8_1(
              input + row * k, blocks_per_row,
              static_cast<BlockQ8_0Activation*>(packed) +
                  row * blocks_per_row);
          break;
        case QuantActivationFormat::kQ8_1:
          row_valid = pack_q8_0_or_q8_1(
              input + row * k, blocks_per_row,
              static_cast<BlockQ8_1Activation*>(packed) +
                  row * blocks_per_row);
          break;
        case QuantActivationFormat::kQ8_K:
          row_valid = pack_q8_k(
              input + row * k, blocks_per_row,
              static_cast<BlockQ8KActivation*>(packed) +
                  row * blocks_per_row);
          break;
      }
      if (!row_valid) valid.store(false, std::memory_order_relaxed);
    }
  });
  return valid.load(std::memory_order_relaxed);
}

void activation_unpack_ref(QuantActivationFormat format, const void* packed,
                           long long n, long long k, float* output) {
  long long block_size = 0;
  std::size_t ignored = 0;
  if (!activation_format_info(format, &block_size, &ignored)) return;
  const long long blocks = n * (k / block_size);
  switch (format) {
    case QuantActivationFormat::kQ8_0:
      unpack_q8_half(static_cast<const BlockQ8_0Activation*>(packed), blocks,
                     output);
      return;
    case QuantActivationFormat::kQ8_1:
      unpack_q8_half(static_cast<const BlockQ8_1Activation*>(packed), blocks,
                     output);
      return;
    case QuantActivationFormat::kQ8_K: {
      const auto* input = static_cast<const BlockQ8KActivation*>(packed);
      for (long long block = 0; block < blocks; ++block) {
        for (int i = 0; i < 256; ++i) {
          output[block * 256 + i] = input[block].d * input[block].qs[i];
        }
      }
      return;
    }
  }
}

}  // namespace quixicore_cpu::quant
