#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#include "kernels/common/validation.h"
#include "quixicore_cpu/base_q.h"
#include "quixicore_cpu/quantization.h"
#include "quixicore_cpu/threading.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

bool known_storage(FloatStorageType type) {
  return type == FloatStorageType::kF32 || type == FloatStorageType::kF16 ||
         type == FloatStorageType::kBF16;
}

bool known_bits(int bits) {
  return bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
         bits == 8;
}

bool known_group_size(int group_size) {
  return group_size == 32 || group_size == 64 || group_size == 128;
}

bool known_scale_type(BaseQScaleType type) {
  return type == BaseQScaleType::kBF16 || type == BaseQScaleType::kF16 ||
         type == BaseQScaleType::kE8M0 || type == BaseQScaleType::kE4M3;
}

std::size_t scale_bytes(BaseQScaleType type) {
  return type == BaseQScaleType::kBF16 || type == BaseQScaleType::kF16 ? 2 : 1;
}

Status validate_view(const BaseQTensorView& view) {
  if (!known_scale_type(view.scale_type)) return Status::kUnsupportedFormat;
  if (!known_bits(view.bits) || !known_group_size(view.group_size) ||
      view.rows <= 0 || view.columns <= 0 ||
      view.columns % view.group_size != 0 ||
      !detail::valid_product({view.rows, view.columns})) {
    return Status::kInvalidShape;
  }
  if (view.scale_type == BaseQScaleType::kE4M3 && view.bits != 8)
    return Status::kUnsupportedFormat;
  const auto packed_bits = static_cast<unsigned long long>(view.columns) *
                           static_cast<unsigned>(view.bits);
  if ((packed_bits & 7u) != 0u) return Status::kInvalidShape;
  const auto row_bytes = packed_bits / 8u;
  const auto rows = static_cast<unsigned long long>(view.rows);
  if (row_bytes > std::numeric_limits<std::size_t>::max() / rows)
    return Status::kInvalidShape;
  const std::size_t expected_code_bytes =
      static_cast<std::size_t>(row_bytes * rows);
  const std::size_t expected_scales =
      static_cast<std::size_t>(view.rows * (view.columns / view.group_size));
  if (view.codes == nullptr || view.scales == nullptr ||
      view.code_bytes != expected_code_bytes ||
      view.scale_count != expected_scales) {
    return Status::kInvalidArgument;
  }
  if (view.symmetric) {
    if (view.biases != nullptr || view.bias_count != 0)
      return Status::kInvalidArgument;
  } else if (view.biases == nullptr || view.bias_count != expected_scales) {
    return Status::kInvalidArgument;
  }
  (void)scale_bytes(view.scale_type);
  return Status::kOk;
}

float load_storage(FloatStorageInput input, std::size_t index) {
  switch (input.type) {
    case FloatStorageType::kF32:
      return static_cast<const float*>(input.data)[index];
    case FloatStorageType::kF16:
      return f16_to_float(static_cast<const std::uint16_t*>(input.data)[index]);
    case FloatStorageType::kBF16:
      return bf16_to_float(
          static_cast<const std::uint16_t*>(input.data)[index]);
  }
  return 0.0f;
}

void store_storage(FloatStorageOutput output, std::size_t index, float value) {
  switch (output.type) {
    case FloatStorageType::kF32:
      static_cast<float*>(output.data)[index] = value;
      break;
    case FloatStorageType::kF16:
      static_cast<std::uint16_t*>(output.data)[index] = float_to_f16(value);
      break;
    case FloatStorageType::kBF16:
      static_cast<std::uint16_t*>(output.data)[index] = float_to_bf16(value);
      break;
  }
}

float load_scale(const void* storage, std::size_t index, BaseQScaleType type) {
  switch (type) {
    case BaseQScaleType::kBF16:
      return bf16_to_float(static_cast<const std::uint16_t*>(storage)[index]);
    case BaseQScaleType::kF16:
      return f16_to_float(static_cast<const std::uint16_t*>(storage)[index]);
    case BaseQScaleType::kE8M0:
      return std::ldexp(
          1.0f,
          static_cast<int>(static_cast<const std::uint8_t*>(storage)[index]) -
              127);
    case BaseQScaleType::kE4M3:
      return float8_decode(static_cast<const std::uint8_t*>(storage)[index],
                           Float8Format::kE4M3FN);
  }
  return 0.0f;
}

std::uint32_t load_code(const BaseQTensorView& view, long long row,
                        long long column) {
  const std::size_t row_bytes =
      static_cast<std::size_t>(view.columns * view.bits / 8);
  const int bit_index = static_cast<int>(column * view.bits);
  const std::size_t byte_index = static_cast<std::size_t>(bit_index >> 3);
  const int shift = bit_index & 7;
  const auto* row_codes =
      view.codes + static_cast<std::size_t>(row) * row_bytes;
  std::uint32_t word = row_codes[byte_index];
  if (shift + view.bits > 8)
    word |= static_cast<std::uint32_t>(row_codes[byte_index + 1]) << 8;
  return (word >> shift) & ((1u << view.bits) - 1u);
}

float load_weight(const BaseQTensorView& view, long long row,
                  long long column) {
  const long long groups_per_row = view.columns / view.group_size;
  const std::size_t group =
      static_cast<std::size_t>(row * groups_per_row + column / view.group_size);
  const float scale = load_scale(view.scales, group, view.scale_type);
  const std::uint32_t code = load_code(view, row, column);
  if (view.symmetric) {
    return static_cast<float>(static_cast<int>(code) - (1 << (view.bits - 1))) *
           scale;
  }
  return static_cast<float>(code) * scale +
         load_scale(view.biases, group, view.scale_type);
}

Status validate_projection(BaseQTensorView weights, FloatStorageInput x,
                           FloatStorageOutput y, long long m) {
  const Status status = validate_view(weights);
  if (status != Status::kOk) return status;
  if (!known_storage(x.type) || !known_storage(y.type))
    return Status::kUnsupportedFormat;
  if (!detail::valid_product({m, weights.columns}) ||
      !detail::valid_product({m, weights.rows}) ||
      x.count != m * weights.columns || y.count != m * weights.rows) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x.data, y.data)) return Status::kInvalidArgument;
  return Status::kOk;
}

bool compatible_views(const BaseQTensorView& first,
                      const BaseQTensorView& second) {
  return first.rows == second.rows && first.columns == second.columns &&
         first.bits == second.bits && first.group_size == second.group_size &&
         first.scale_type == second.scale_type &&
         first.symmetric == second.symmetric;
}

bool compatible_descriptors(const BaseQTensorView& first,
                            const BaseQTensorView& second) {
  return first.columns == second.columns && first.bits == second.bits &&
         first.group_size == second.group_size &&
         first.scale_type == second.scale_type &&
         first.symmetric == second.symmetric;
}

float project_row(BaseQTensorView weights, FloatStorageInput x,
                  long long weight_row) {
  float sums[4] = {};
  const auto* input_f32 = x.type == FloatStorageType::kF32
                              ? static_cast<const float*>(x.data)
                              : nullptr;
  const long long groups_per_row = weights.columns / weights.group_size;
  for (long long group_lane = 0; group_lane < groups_per_row; ++group_lane) {
    const std::size_t group =
        static_cast<std::size_t>(weight_row * groups_per_row + group_lane);
    const float scale = load_scale(weights.scales, group, weights.scale_type);
    const float bias = weights.symmetric ? 0.0f
                                         : load_scale(weights.biases, group,
                                                      weights.scale_type);
    const long long first_column = group_lane * weights.group_size;
    const long long end_column = first_column + weights.group_size;
    for (long long column = first_column; column < end_column; ++column) {
      const std::uint32_t code = load_code(weights, weight_row, column);
      const float value = weights.symmetric
                              ? static_cast<float>(static_cast<int>(code) -
                                                   (1 << (weights.bits - 1))) *
                                    scale
                              : static_cast<float>(code) * scale + bias;
      const std::size_t input_index = static_cast<std::size_t>(column);
      sums[column & 3] +=
          value * (input_f32 != nullptr ? input_f32[input_index]
                                        : load_storage(x, input_index));
    }
  }
  return (sums[0] + sums[1]) + (sums[2] + sums[3]);
}

FloatStorageInput storage_row(FloatStorageInput input, long long row,
                              long long columns) {
  const std::size_t bytes = input.type == FloatStorageType::kF32
                                ? sizeof(float)
                                : sizeof(std::uint16_t);
  return {static_cast<const std::uint8_t*>(input.data) +
              static_cast<std::size_t>(row * columns) * bytes,
          input.type, columns};
}

void project_expert_output_tile(
    BaseQTensorView weights, FloatStorageInput input, long long input_row_begin,
    long long gate_weight_row, long long up_weight_row,
    FloatStorageOutput output, long long output_columns,
    long long output_column) {
  constexpr long long kRows = 32;
  float gate_sums[kRows][4] = {};
  float up_sums[kRows][4] = {};
  const bool fused_swiglu = up_weight_row >= 0;
  const long long groups_per_row = weights.columns / weights.group_size;
  const auto* input_f32 = input.type == FloatStorageType::kF32
                              ? static_cast<const float*>(input.data)
                              : nullptr;
  for (long long group_lane = 0; group_lane < groups_per_row; ++group_lane) {
    const std::size_t gate_group =
        static_cast<std::size_t>(gate_weight_row * groups_per_row + group_lane);
    const float gate_scale =
        load_scale(weights.scales, gate_group, weights.scale_type);
    const float gate_bias =
        weights.symmetric
            ? 0.0f
            : load_scale(weights.biases, gate_group, weights.scale_type);
    float up_scale = 0.0f;
    float up_bias = 0.0f;
    if (fused_swiglu) {
      const std::size_t up_group =
          static_cast<std::size_t>(up_weight_row * groups_per_row + group_lane);
      up_scale = load_scale(weights.scales, up_group, weights.scale_type);
      up_bias = weights.symmetric
                    ? 0.0f
                    : load_scale(weights.biases, up_group, weights.scale_type);
    }
    float decoded_gate[128];
    float decoded_up[128];
    const long long first_column = group_lane * weights.group_size;
    for (int lane = 0; lane < weights.group_size; ++lane) {
      const long long column = first_column + lane;
      const std::uint32_t gate_code =
          load_code(weights, gate_weight_row, column);
      decoded_gate[lane] =
          weights.symmetric
              ? static_cast<float>(static_cast<int>(gate_code) -
                                   (1 << (weights.bits - 1))) *
                    gate_scale
              : static_cast<float>(gate_code) * gate_scale + gate_bias;
      if (fused_swiglu) {
        const std::uint32_t up_code = load_code(weights, up_weight_row, column);
        decoded_up[lane] =
            weights.symmetric
                ? static_cast<float>(static_cast<int>(up_code) -
                                     (1 << (weights.bits - 1))) *
                      up_scale
                : static_cast<float>(up_code) * up_scale + up_bias;
      }
    }
    for (long long input_lane = 0; input_lane < kRows; ++input_lane) {
      const std::size_t input_base = static_cast<std::size_t>(
          (input_row_begin + input_lane) * weights.columns + first_column);
      for (int lane = 0; lane < weights.group_size; ++lane) {
        const float activation = input_f32 != nullptr
                                     ? input_f32[input_base + lane]
                                     : load_storage(input, input_base + lane);
        gate_sums[input_lane][lane & 3] += decoded_gate[lane] * activation;
        if (fused_swiglu)
          up_sums[input_lane][lane & 3] += decoded_up[lane] * activation;
      }
    }
  }
  for (long long input_lane = 0; input_lane < kRows; ++input_lane) {
    const float gate = (gate_sums[input_lane][0] + gate_sums[input_lane][1]) +
                       (gate_sums[input_lane][2] + gate_sums[input_lane][3]);
    float value = gate;
    if (fused_swiglu) {
      const float up = (up_sums[input_lane][0] + up_sums[input_lane][1]) +
                       (up_sums[input_lane][2] + up_sums[input_lane][3]);
      value = gate / (1.0f + std::exp(-gate)) * up;
    }
    store_storage(
        output,
        static_cast<std::size_t>(
            (input_row_begin + input_lane) * output_columns + output_column),
        value);
  }
}

float round_storage(FloatStorageType type, float value) {
  switch (type) {
    case FloatStorageType::kF32:
      return value;
    case FloatStorageType::kF16:
      return f16_to_float(float_to_f16(value));
    case FloatStorageType::kBF16:
      return bf16_to_float(float_to_bf16(value));
  }
  return value;
}

struct ArgmaxWinner {
  float value;
  int token;
};

std::uint32_t load_row_code(const std::uint8_t* row_codes, long long column,
                            int bits) {
  const long long bit_index = column * bits;
  const std::size_t byte_index = static_cast<std::size_t>(bit_index >> 3);
  const int shift = static_cast<int>(bit_index & 7);
  std::uint32_t word = row_codes[byte_index];
  if (shift + bits > 8)
    word |= static_cast<std::uint32_t>(row_codes[byte_index + 1]) << 8;
  return (word >> shift) & ((1u << bits) - 1u);
}

ArgmaxWinner project_argmax_chunk(BaseQTensorView weights, FloatStorageInput x,
                                  long long begin, long long end) {
  const long long groups_per_row = weights.columns / weights.group_size;
  const std::size_t row_bytes =
      static_cast<std::size_t>(weights.columns * weights.bits / 8);
  const auto* input_f32 = x.type == FloatStorageType::kF32
                              ? static_cast<const float*>(x.data)
                              : nullptr;
  ArgmaxWinner winner{0.0f, static_cast<int>(begin)};
  bool have_winner = false;
  long long weight_row = begin;
  for (; weight_row + 4 <= end; weight_row += 4) {
    float sums[4][4] = {};
    const std::uint8_t* row_codes[4];
    for (int row_lane = 0; row_lane < 4; ++row_lane) {
      row_codes[row_lane] =
          weights.codes +
          static_cast<std::size_t>(weight_row + row_lane) * row_bytes;
    }
    for (long long group_lane = 0; group_lane < groups_per_row; ++group_lane) {
      float scales[4];
      float biases[4];
      for (int row_lane = 0; row_lane < 4; ++row_lane) {
        const std::size_t group = static_cast<std::size_t>(
            (weight_row + row_lane) * groups_per_row + group_lane);
        scales[row_lane] =
            load_scale(weights.scales, group, weights.scale_type);
        biases[row_lane] = weights.symmetric ? 0.0f
                                             : load_scale(weights.biases, group,
                                                          weights.scale_type);
      }
      const long long first_column = group_lane * weights.group_size;
      const long long end_column = first_column + weights.group_size;
      for (long long column = first_column; column < end_column; ++column) {
        const std::size_t input_index = static_cast<std::size_t>(column);
        const float activation = input_f32 != nullptr
                                     ? input_f32[input_index]
                                     : load_storage(x, input_index);
        for (int row_lane = 0; row_lane < 4; ++row_lane) {
          const std::uint32_t code =
              load_row_code(row_codes[row_lane], column, weights.bits);
          const float value =
              weights.symmetric
                  ? static_cast<float>(static_cast<int>(code) -
                                       (1 << (weights.bits - 1))) *
                        scales[row_lane]
                  : static_cast<float>(code) * scales[row_lane] +
                        biases[row_lane];
          sums[row_lane][column & 3] += value * activation;
        }
      }
    }
    for (int row_lane = 0; row_lane < 4; ++row_lane) {
      const float value =
          round_storage(x.type, (sums[row_lane][0] + sums[row_lane][1]) +
                                    (sums[row_lane][2] + sums[row_lane][3]));
      if (!have_winner || value > winner.value) {
        winner = {value, static_cast<int>(weight_row + row_lane)};
        have_winner = true;
      }
    }
  }
  for (; weight_row < end; ++weight_row) {
    const float value =
        round_storage(x.type, project_row(weights, x, weight_row));
    if (!have_winner || value > winner.value) {
      winner = {value, static_cast<int>(weight_row)};
      have_winner = true;
    }
  }
  return winner;
}

}  // namespace

Status base_q_dequant(BaseQTensorView weights, FloatStorageOutput output) {
  const Status status = validate_view(weights);
  if (status != Status::kOk) return status;
  if (!known_storage(output.type)) return Status::kUnsupportedFormat;
  if (output.count != weights.rows * weights.columns)
    return Status::kInvalidShape;
  if (output.data == nullptr) return Status::kInvalidArgument;
  threading::parallel_ranges(
      weights.rows * weights.columns, 4096,
      [&](long long begin, long long end, int) {
        for (long long index = begin; index < end; ++index) {
          const long long row = index / weights.columns;
          const long long column = index - row * weights.columns;
          store_storage(output, static_cast<std::size_t>(index),
                        load_weight(weights, row, column));
        }
      });
  return Status::kOk;
}

Status base_q_gemm(BaseQTensorView weights, FloatStorageInput x,
                   FloatStorageOutput y, long long m) {
  const Status status = validate_projection(weights, x, y, m);
  if (status != Status::kOk) return status;
  const long long groups_per_row = weights.columns / weights.group_size;
  const auto* input_f32 = x.type == FloatStorageType::kF32
                              ? static_cast<const float*>(x.data)
                              : nullptr;
  auto* output_f32 =
      y.type == FloatStorageType::kF32 ? static_cast<float*>(y.data) : nullptr;
  if (m > 1) {
    // Decode each packed group once for a tile of activation rows. Prefill
    // shapes otherwise repeat the bit extraction and scale conversion M times.
    constexpr long long kInputRowTile = 16;
    threading::parallel_ranges(
        weights.rows, 4, [&](long long begin, long long end, int) {
          for (long long weight_row = begin; weight_row < end; ++weight_row) {
            for (long long input_row_begin = 0; input_row_begin < m;
                 input_row_begin += kInputRowTile) {
              const long long input_rows =
                  std::min(kInputRowTile, m - input_row_begin);
              float sums[kInputRowTile][4] = {};
              for (long long group_lane = 0; group_lane < groups_per_row;
                   ++group_lane) {
                const std::size_t group = static_cast<std::size_t>(
                    weight_row * groups_per_row + group_lane);
                const float scale =
                    load_scale(weights.scales, group, weights.scale_type);
                const float bias =
                    weights.symmetric
                        ? 0.0f
                        : load_scale(weights.biases, group, weights.scale_type);
                float decoded[128];
                const long long first_column = group_lane * weights.group_size;
                for (int lane = 0; lane < weights.group_size; ++lane) {
                  const std::uint32_t code =
                      load_code(weights, weight_row, first_column + lane);
                  decoded[lane] =
                      weights.symmetric
                          ? static_cast<float>(static_cast<int>(code) -
                                               (1 << (weights.bits - 1))) *
                                scale
                          : static_cast<float>(code) * scale + bias;
                }
                for (long long input_lane = 0; input_lane < input_rows;
                     ++input_lane) {
                  const long long input_row = input_row_begin + input_lane;
                  const std::size_t input_base = static_cast<std::size_t>(
                      input_row * weights.columns + first_column);
                  if (input_f32 != nullptr) {
                    for (int lane = 0; lane < weights.group_size; ++lane) {
                      sums[input_lane][lane & 3] +=
                          decoded[lane] * input_f32[input_base + lane];
                    }
                  } else {
                    for (int lane = 0; lane < weights.group_size; ++lane) {
                      sums[input_lane][lane & 3] +=
                          decoded[lane] * load_storage(x, input_base + lane);
                    }
                  }
                }
              }
              for (long long input_lane = 0; input_lane < input_rows;
                   ++input_lane) {
                const float* row_sums = sums[input_lane];
                const float value =
                    (row_sums[0] + row_sums[1]) + (row_sums[2] + row_sums[3]);
                const std::size_t output_index = static_cast<std::size_t>(
                    (input_row_begin + input_lane) * weights.rows + weight_row);
                if (output_f32 != nullptr)
                  output_f32[output_index] = value;
                else
                  store_storage(y, output_index, value);
              }
            }
          }
        });
    return Status::kOk;
  }
  threading::parallel_ranges(
      m * weights.rows, 16, [&](long long begin, long long end, int) {
        for (long long index = begin; index < end; ++index) {
          const long long weight_row = index;
          const float value = project_row(weights, x, weight_row);
          if (output_f32 != nullptr)
            output_f32[static_cast<std::size_t>(index)] = value;
          else
            store_storage(y, static_cast<std::size_t>(index), value);
        }
      });
  return Status::kOk;
}

Status base_q_gemv(BaseQTensorView weights, FloatStorageInput x,
                   FloatStorageOutput y) {
  return base_q_gemm(weights, x, y, 1);
}

Status base_q_embedding(BaseQTensorView weights, const int* ids,
                        long long tokens, FloatStorageOutput output) {
  const Status status = validate_view(weights);
  if (status != Status::kOk) return status;
  if (!known_storage(output.type)) return Status::kUnsupportedFormat;
  if (!detail::valid_product({tokens, weights.columns}) ||
      output.count != tokens * weights.columns)
    return Status::kInvalidShape;
  if ((tokens > 0 && ids == nullptr) || output.data == nullptr)
    return Status::kInvalidArgument;
  threading::parallel_ranges(
      tokens * weights.columns, 4096, [&](long long begin, long long end, int) {
        for (long long index = begin; index < end; ++index) {
          const long long token = index / weights.columns;
          const long long column = index - token * weights.columns;
          const int row = ids[token];
          const float value = row < 0 || row >= weights.rows
                                  ? 0.0f
                                  : load_weight(weights, row, column);
          store_storage(output, static_cast<std::size_t>(index), value);
        }
      });
  return Status::kOk;
}

Status base_q_gemv_qkv(BaseQTensorView q_weights, BaseQTensorView k_weights,
                       BaseQTensorView v_weights, FloatStorageInput x,
                       FloatStorageOutput q_output, FloatStorageOutput k_output,
                       FloatStorageOutput v_output) {
  if (!compatible_descriptors(q_weights, k_weights) ||
      !compatible_descriptors(q_weights, v_weights))
    return Status::kInvalidShape;
  Status status = validate_projection(q_weights, x, q_output, 1);
  if (status != Status::kOk) return status;
  status = validate_projection(k_weights, x, k_output, 1);
  if (status != Status::kOk) return status;
  status = validate_projection(v_weights, x, v_output, 1);
  if (status != Status::kOk) return status;

  const long long total_rows = q_weights.rows + k_weights.rows + v_weights.rows;
  threading::parallel_ranges(
      total_rows, 16, [&](long long begin, long long end, int) {
        for (long long index = begin; index < end; ++index) {
          const BaseQTensorView* weights = &q_weights;
          FloatStorageOutput* output = &q_output;
          long long row = index;
          if (row >= q_weights.rows) {
            row -= q_weights.rows;
            weights = &k_weights;
            output = &k_output;
            if (row >= k_weights.rows) {
              row -= k_weights.rows;
              weights = &v_weights;
              output = &v_output;
            }
          }
          store_storage(*output, static_cast<std::size_t>(row),
                        project_row(*weights, x, row));
        }
      });
  return Status::kOk;
}

Status base_q_gemv_swiglu(BaseQTensorView gate_weights,
                          BaseQTensorView up_weights, FloatStorageInput x,
                          FloatStorageOutput output) {
  if (!compatible_views(gate_weights, up_weights)) return Status::kInvalidShape;
  const Status status = validate_projection(gate_weights, x, output, 1);
  if (status != Status::kOk) return status;
  const Status up_status = validate_view(up_weights);
  if (up_status != Status::kOk) return up_status;
  threading::parallel_ranges(
      gate_weights.rows, 16, [&](long long begin, long long end, int) {
        for (long long row = begin; row < end; ++row) {
          float gate_sums[4] = {};
          float up_sums[4] = {};
          const long long groups_per_row =
              gate_weights.columns / gate_weights.group_size;
          for (long long group_lane = 0; group_lane < groups_per_row;
               ++group_lane) {
            const std::size_t group =
                static_cast<std::size_t>(row * groups_per_row + group_lane);
            const float gate_scale =
                load_scale(gate_weights.scales, group, gate_weights.scale_type);
            const float up_scale =
                load_scale(up_weights.scales, group, up_weights.scale_type);
            const float gate_bias = gate_weights.symmetric
                                        ? 0.0f
                                        : load_scale(gate_weights.biases, group,
                                                     gate_weights.scale_type);
            const float up_bias = up_weights.symmetric
                                      ? 0.0f
                                      : load_scale(up_weights.biases, group,
                                                   up_weights.scale_type);
            const long long first_column = group_lane * gate_weights.group_size;
            const long long end_column = first_column + gate_weights.group_size;
            for (long long column = first_column; column < end_column;
                 ++column) {
              const float activation =
                  load_storage(x, static_cast<std::size_t>(column));
              const std::uint32_t gate_code =
                  load_code(gate_weights, row, column);
              const std::uint32_t up_code = load_code(up_weights, row, column);
              const float gate_value =
                  gate_weights.symmetric
                      ? static_cast<float>(static_cast<int>(gate_code) -
                                           (1 << (gate_weights.bits - 1))) *
                            gate_scale
                      : static_cast<float>(gate_code) * gate_scale + gate_bias;
              const float up_value =
                  up_weights.symmetric
                      ? static_cast<float>(static_cast<int>(up_code) -
                                           (1 << (up_weights.bits - 1))) *
                            up_scale
                      : static_cast<float>(up_code) * up_scale + up_bias;
              gate_sums[column & 3] += gate_value * activation;
              up_sums[column & 3] += up_value * activation;
            }
          }
          const float gate =
              (gate_sums[0] + gate_sums[1]) + (gate_sums[2] + gate_sums[3]);
          const float up =
              (up_sums[0] + up_sums[1]) + (up_sums[2] + up_sums[3]);
          const float silu = gate / (1.0f + std::exp(-gate));
          store_storage(output, static_cast<std::size_t>(row), silu * up);
        }
      });
  return Status::kOk;
}

Status base_q_lm_head_argmax(BaseQTensorView weights, FloatStorageInput x,
                             int* token_ids, long long batch) {
  const Status view_status = validate_view(weights);
  if (view_status != Status::kOk) return view_status;
  if (!known_storage(x.type)) return Status::kUnsupportedFormat;
  if (!detail::valid_product({batch, weights.columns}) ||
      !detail::valid_product({batch, weights.rows}) ||
      x.count != batch * weights.columns || weights.rows > INT_MAX) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x.data, token_ids)) return Status::kInvalidArgument;

  try {
    const int threads = num_threads();
#if defined(__x86_64__) || defined(_M_X64)
    // The x86 multi-row projection reuses each decoded weight group across
    // activation rows. It also wins threaded Q4 decode on the measured AVX2
    // host, while the streaming four-row vocabulary tile wins the other
    // single-row cells. Materialize only those measured-winning shapes.
    if (batch > 1 || (threads > 1 && weights.bits == 4)) {
      const std::size_t score_count =
          static_cast<std::size_t>(batch * weights.rows);
      thread_local std::vector<float> scores_f32;
      thread_local std::vector<std::uint16_t> scores_f16;
      void* scores = nullptr;
      if (x.type == FloatStorageType::kF32) {
        scores_f32.resize(score_count);
        scores = scores_f32.data();
      } else {
        scores_f16.resize(score_count);
        scores = scores_f16.data();
      }
      const FloatStorageOutput score_output = {
          scores, x.type, static_cast<long long>(score_count)};
      const Status projection_status =
          base_q_gemm(weights, x, score_output, batch);
      if (projection_status != Status::kOk) return projection_status;
      const FloatStorageInput score_input = {
          scores, x.type, static_cast<long long>(score_count)};
      for (long long input_row = 0; input_row < batch; ++input_row) {
        int best = 0;
        float best_value = load_storage(
            score_input, static_cast<std::size_t>(input_row * weights.rows));
        for (long long token = 1; token < weights.rows; ++token) {
          const float value = load_storage(
              score_input,
              static_cast<std::size_t>(input_row * weights.rows + token));
          if (value > best_value) {
            best_value = value;
            best = static_cast<int>(token);
          }
        }
        token_ids[input_row] = best;
      }
      return Status::kOk;
    }
#endif
    const long long chunks =
        std::min(weights.rows, static_cast<long long>(threads) * 4);
    thread_local std::vector<ArgmaxWinner> partial;
    partial.resize(static_cast<std::size_t>(batch * chunks));
    ArgmaxWinner* partial_data = partial.data();
    threading::parallel_ranges(
        batch * chunks, 1, [&](long long begin, long long end, int) {
          for (long long task = begin; task < end; ++task) {
            const long long input_row = task / chunks;
            const long long chunk = task - input_row * chunks;
            const long long chunk_base = weights.rows / chunks;
            const long long chunk_remainder = weights.rows % chunks;
            const long long token_begin =
                chunk * chunk_base + std::min(chunk, chunk_remainder);
            const long long token_end =
                token_begin + chunk_base + (chunk < chunk_remainder ? 1 : 0);
            partial_data[static_cast<std::size_t>(task)] = project_argmax_chunk(
                weights, storage_row(x, input_row, weights.columns),
                token_begin, token_end);
          }
        });
    for (long long input_row = 0; input_row < batch; ++input_row) {
      ArgmaxWinner winner =
          partial[static_cast<std::size_t>(input_row * chunks)];
      for (long long chunk = 1; chunk < chunks; ++chunk) {
        const ArgmaxWinner candidate =
            partial[static_cast<std::size_t>(input_row * chunks + chunk)];
        if (candidate.value > winner.value) winner = candidate;
      }
      token_ids[input_row] = winner.token;
    }
  } catch (const std::bad_alloc&) {
    return Status::kOutOfMemory;
  } catch (const std::length_error&) {
    return Status::kInvalidShape;
  }
  return Status::kOk;
}

Status base_q_moe_gemm(BaseQTensorView weights, long long experts,
                       FloatStorageInput input, const int* expert_of_tile,
                       long long total_rows, FloatStorageOutput output) {
  const Status view_status = validate_view(weights);
  if (view_status != Status::kOk) return view_status;
  if (!known_storage(input.type) || !known_storage(output.type))
    return Status::kUnsupportedFormat;
  if (experts <= 0 || total_rows <= 0 || total_rows % 32 != 0 ||
      weights.columns % 32 != 0 || weights.rows % experts != 0) {
    return Status::kInvalidShape;
  }
  const long long output_rows = weights.rows / experts;
  if (output_rows <= 0 || output_rows % 32 != 0 ||
      !detail::valid_product({total_rows, weights.columns}) ||
      !detail::valid_product({total_rows, output_rows}) ||
      input.count != total_rows * weights.columns ||
      output.count != total_rows * output_rows || output.type != input.type) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input.data, expert_of_tile, output.data))
    return Status::kInvalidArgument;
  for (long long tile = 0; tile < total_rows / 32; ++tile) {
    if (expert_of_tile[tile] < 0 || expert_of_tile[tile] >= experts)
      return Status::kInvalidArgument;
  }
  threading::parallel_ranges(
      total_rows / 32 * output_rows, 4,
      [&](long long begin, long long end, int) {
        for (long long task = begin; task < end; ++task) {
          const long long tile = task / output_rows;
          const long long output_row = task - tile * output_rows;
          const long long expert = expert_of_tile[tile];
          project_expert_output_tile(weights, input, tile * 32,
                                     expert * output_rows + output_row, -1,
                                     output, output_rows, output_row);
        }
      });
  return Status::kOk;
}

Status base_q_moe_swiglu(BaseQTensorView weights, long long experts,
                         FloatStorageInput input, const int* expert_of_tile,
                         long long total_rows, FloatStorageOutput output) {
  const Status view_status = validate_view(weights);
  if (view_status != Status::kOk) return view_status;
  if (!known_storage(input.type) || !known_storage(output.type))
    return Status::kUnsupportedFormat;
  if (experts <= 0 || total_rows <= 0 || total_rows % 32 != 0 ||
      weights.columns % 32 != 0 || weights.rows % experts != 0 ||
      (weights.rows / experts) % 2 != 0) {
    return Status::kInvalidShape;
  }
  const long long intermediate = weights.rows / experts / 2;
  if (intermediate <= 0 || intermediate % 32 != 0 ||
      !detail::valid_product({total_rows, weights.columns}) ||
      !detail::valid_product({total_rows, intermediate}) ||
      input.count != total_rows * weights.columns ||
      output.count != total_rows * intermediate || output.type != input.type) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(input.data, expert_of_tile, output.data))
    return Status::kInvalidArgument;
  for (long long tile = 0; tile < total_rows / 32; ++tile) {
    if (expert_of_tile[tile] < 0 || expert_of_tile[tile] >= experts)
      return Status::kInvalidArgument;
  }
  threading::parallel_ranges(
      total_rows / 32 * intermediate, 4,
      [&](long long begin, long long end, int) {
        for (long long task = begin; task < end; ++task) {
          const long long tile = task / intermediate;
          const long long output_row = task - tile * intermediate;
          const long long expert = expert_of_tile[tile];
          const long long expert_base = expert * 2 * intermediate;
          project_expert_output_tile(weights, input, tile * 32,
                                     expert_base + output_row,
                                     expert_base + intermediate + output_row,
                                     output, intermediate, output_row);
        }
      });
  return Status::kOk;
}

}  // namespace quixicore_cpu
