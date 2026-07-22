#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#include "kernels/common/validation.h"
#include "kernels/common/fp16.h"
#include "kernels/quantization/gguf_ref.h"
#include "kernels/quantization/qgemv_w8a8.h"
#include "quixicore_cpu/qgemm.h"
#include "quixicore_cpu/qgemv.h"
#include "quixicore_cpu/quantization.h"
#include "quixicore_cpu/threading.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

double softplus(double x) {
  if (x > 20.0) return x;
  if (x < -20.0) return std::exp(x);
  return std::log1p(std::exp(x));
}

float decoded_block_dot(const float* weights, const float* input,
                        long long count) {
#if defined(__aarch64__) || defined(_M_ARM64)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  long long item = 0;
  for (; item + 7 < count; item += 8) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(weights + item),
                     vld1q_f32(input + item));
    acc1 = vfmaq_f32(acc1, vld1q_f32(weights + item + 4),
                     vld1q_f32(input + item + 4));
  }
  float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
#else
  float sum = 0.0f;
  long long item = 0;
#endif
  for (; item < count; ++item) sum += weights[item] * input[item];
  return sum;
}

#if defined(__aarch64__) || defined(_M_ARM64)
float int8x16_dot(int8x16_t weights, const float* input) {
  const int16x8_t low = vmovl_s8(vget_low_s8(weights));
  const int16x8_t high = vmovl_s8(vget_high_s8(weights));
  float32x4_t sum = vmulq_f32(
      vcvtq_f32_s32(vmovl_s16(vget_low_s16(low))), vld1q_f32(input));
  sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_high_s16(low))),
                  vld1q_f32(input + 4));
  sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_low_s16(high))),
                  vld1q_f32(input + 8));
  sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_high_s16(high))),
                  vld1q_f32(input + 12));
  return vaddvq_f32(sum);
}
#endif

float q4_block_dot(const quant::BlockQ4_0& block, const float* input) {
#if defined(__aarch64__) || defined(_M_ARM64)
  const uint8x16_t codes = vld1q_u8(block.qs);
  const uint8x16_t mask = vdupq_n_u8(15);
  const uint8x16_t offset = vdupq_n_u8(8);
  const int8x16_t low = vreinterpretq_s8_u8(
      vsubq_u8(vandq_u8(codes, mask), offset));
  const int8x16_t high = vreinterpretq_s8_u8(
      vsubq_u8(vshrq_n_u8(codes, 4), offset));
  const float dot =
      int8x16_dot(low, input) + int8x16_dot(high, input + 16);
#else
  float dot = 0.0f;
  for (int item = 0; item < 16; ++item) {
    dot += static_cast<float>((block.qs[item] & 15) - 8) * input[item];
    dot += static_cast<float>((block.qs[item] >> 4) - 8) * input[item + 16];
  }
#endif
  return fp16_to_fp32(block.d) * dot;
}

Status grouped_qprojection(const float* x, const void* packed_weights,
                           const int* expert_ids, const float* bias,
                           float* out, long long rows, long long experts,
                           long long input_dim, long long output_dim,
                           QuantFormat format, bool use_bias) {
  std::size_t expert_bytes = 0;
  Status status =
      qgemv_packed_size(format, output_dim, input_dim, &expert_bytes);
  if (status != Status::kOk) return status;
  long long block_size = 0;
  std::size_t block_bytes = 0;
  if (!quant::gguf_format_info(format, &block_size, &block_bytes)) {
    return Status::kUnsupportedFormat;
  }
  const long long blocks_per_row = input_dim / block_size;
  const std::size_t weight_row_bytes =
      static_cast<std::size_t>(blocks_per_row) * block_bytes;

  std::vector<long long> offsets(static_cast<std::size_t>(experts + 1), 0);
  for (long long row = 0; row < rows; ++row) {
    const int expert = expert_ids[row];
    if (expert < 0 || expert >= experts) return Status::kInvalidArgument;
    ++offsets[static_cast<std::size_t>(expert + 1)];
  }
  for (long long expert = 0; expert < experts; ++expert) {
    offsets[static_cast<std::size_t>(expert + 1)] +=
        offsets[static_cast<std::size_t>(expert)];
  }
  std::vector<long long> cursor = offsets;
  std::vector<long long> order(static_cast<std::size_t>(rows));
  for (long long row = 0; row < rows; ++row) {
    order[static_cast<std::size_t>(
        cursor[static_cast<std::size_t>(expert_ids[row])]++)] = row;
  }
  std::vector<int> active;
  active.reserve(static_cast<std::size_t>(std::min(rows, experts)));
  for (long long expert = 0; expert < experts; ++expert) {
    if (offsets[static_cast<std::size_t>(expert)] !=
        offsets[static_cast<std::size_t>(expert + 1)]) {
      active.push_back(static_cast<int>(expert));
    }
  }

  // Batch-union only saves weight traffic when at least one expert is shared.
  // Preserve the ISA-dispatched GEMV path for decode/all-unique routing.
  if (num_threads() == 1 || static_cast<long long>(active.size()) == rows) {
    const auto* weights = static_cast<const std::uint8_t*>(packed_weights);
    for (long long row = 0; row < rows; ++row) {
      const int expert = expert_ids[row];
      status = qgemv(format, weights + expert * expert_bytes,
                     x + row * input_dim, out + row * output_dim, output_dim,
                     input_dim);
      if (status != Status::kOk) return status;
      if (use_bias) {
        for (long long output = 0; output < output_dim; ++output) {
          out[row * output_dim + output] +=
              bias[static_cast<long long>(expert) * output_dim + output];
        }
      }
    }
    return Status::kOk;
  }

  const auto* all_weights =
      static_cast<const std::uint8_t*>(packed_weights);
  const long long tasks = static_cast<long long>(active.size()) * output_dim;
  threading::parallel_ranges(tasks, 8,
                             [&](long long begin, long long end, int) {
    thread_local std::vector<double> accumulators;
    for (long long task = begin; task < end; ++task) {
      const int expert = active[static_cast<std::size_t>(task / output_dim)];
      const long long output = task % output_dim;
      const long long first = offsets[static_cast<std::size_t>(expert)];
      const long long last = offsets[static_cast<std::size_t>(expert + 1)];
      const long long count = last - first;
      accumulators.assign(static_cast<std::size_t>(count), 0.0);
      const std::uint8_t* weight_row =
          all_weights + static_cast<std::size_t>(expert) * expert_bytes +
          static_cast<std::size_t>(output) * weight_row_bytes;
      for (long long block = 0; block < blocks_per_row; ++block) {
        const std::uint8_t* packed_block =
            weight_row + static_cast<std::size_t>(block) * block_bytes;
        for (int column = 0; column < block_size; ++column) {
          const float weight =
              quant::gguf_dequant_element(format, packed_block, column);
          const long long input = block * block_size + column;
          for (long long item = 0; item < count; ++item) {
            const long long row = order[static_cast<std::size_t>(first + item)];
            accumulators[static_cast<std::size_t>(item)] +=
                static_cast<double>(weight) * x[row * input_dim + input];
          }
        }
      }
      for (long long item = 0; item < count; ++item) {
        const long long row = order[static_cast<std::size_t>(first + item)];
        double value = accumulators[static_cast<std::size_t>(item)];
        if (use_bias) {
          value += bias[static_cast<long long>(expert) * output_dim + output];
        }
        out[row * output_dim + output] = static_cast<float>(value);
      }
    }
  });
  return Status::kOk;
}

}  // namespace

Status moe_route_grouped(const float* router_logits, const float* bias,
                         int* expert_ids, float* expert_weights,
                         long long tokens, long long experts, int top_k,
                         int groups, int top_groups, bool renormalize,
                         float routed_scale, MoeScoring scoring) {
  if (!detail::valid_product({tokens, experts}) || groups <= 0 ||
      experts % groups != 0 || top_groups <= 0 || top_groups > groups ||
      top_k <= 0 || top_k > experts || !std::isfinite(routed_scale)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(router_logits, expert_ids, expert_weights)) {
    return Status::kInvalidArgument;
  }
  const long long group_size = experts / groups;
  std::vector<double> scores(static_cast<std::size_t>(experts));
  std::vector<double> selection(static_cast<std::size_t>(experts));
  std::vector<int> group_ids(static_cast<std::size_t>(groups));
  std::vector<int> candidates;
  for (long long token = 0; token < tokens; ++token) {
    const float* row = router_logits + token * experts;
    if (scoring == MoeScoring::kSoftmax) {
      const float maximum = *std::max_element(row, row + experts);
      double denominator = 0.0;
      for (long long expert = 0; expert < experts; ++expert) {
        scores[expert] = std::exp(static_cast<double>(row[expert] - maximum));
        denominator += scores[expert];
      }
      for (double& score : scores) score /= denominator;
    } else {
      for (long long expert = 0; expert < experts; ++expert) {
        scores[expert] =
            scoring == MoeScoring::kSigmoid
                ? 1.0 / (1.0 + std::exp(-static_cast<double>(row[expert])))
                : std::sqrt(softplus(row[expert]));
      }
    }
    for (long long expert = 0; expert < experts; ++expert) {
      selection[expert] = scores[expert] + (bias == nullptr ? 0.0 : bias[expert]);
    }
    std::iota(group_ids.begin(), group_ids.end(), 0);
    std::stable_sort(group_ids.begin(), group_ids.end(), [&](int lhs, int rhs) {
      auto top_two = [&](int group) {
        double first = -std::numeric_limits<double>::infinity();
        double second = first;
        for (long long i = 0; i < group_size; ++i) {
          const double value = selection[group * group_size + i];
          if (value > first) {
            second = first;
            first = value;
          } else if (value > second) {
            second = value;
          }
        }
        return first + (group_size > 1 ? second : 0.0);
      };
      const double ls = top_two(lhs);
      const double rs = top_two(rhs);
      return ls == rs ? lhs < rhs : ls > rs;
    });
    candidates.clear();
    for (int rank = 0; rank < top_groups; ++rank) {
      const int group = group_ids[rank];
      for (long long i = 0; i < group_size; ++i) {
        candidates.push_back(static_cast<int>(group * group_size + i));
      }
    }
    std::partial_sort(candidates.begin(), candidates.begin() + top_k,
                      candidates.end(), [&](int lhs, int rhs) {
      return selection[lhs] == selection[rhs] ? lhs < rhs
                                               : selection[lhs] > selection[rhs];
    });
    double denominator = 0.0;
    for (int rank = 0; rank < top_k; ++rank) {
      const int expert = candidates[rank];
      expert_ids[token * top_k + rank] = expert;
      expert_weights[token * top_k + rank] = static_cast<float>(scores[expert]);
      denominator += scores[expert];
    }
    const double factor = renormalize ? routed_scale / denominator : routed_scale;
    for (int rank = 0; rank < top_k; ++rank) {
      expert_weights[token * top_k + rank] =
          static_cast<float>(expert_weights[token * top_k + rank] * factor);
    }
  }
  return Status::kOk;
}

Status moe_route_scored(const float* router_logits, int* expert_ids,
                        float* expert_weights, long long tokens,
                        long long experts, int top_k, int mode,
                        bool renormalize, float scaling) {
  if (mode < 0 || mode > 1) return Status::kInvalidArgument;
  return moe_route_grouped(
      router_logits, nullptr, expert_ids, expert_weights, tokens, experts,
      top_k, 1, 1, renormalize, scaling,
      mode == 0 ? MoeScoring::kSigmoid : MoeScoring::kSqrtSoftplus);
}

Status moe_permute(const int* expert_ids, int* sorted_rows, int* offsets,
                   int* inverse, long long tokens, long long experts,
                   int top_k) {
  if (!detail::valid_product({tokens, experts}) || top_k <= 0 ||
      !detail::valid_product({tokens, top_k})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(expert_ids, sorted_rows, offsets, inverse)) {
    return Status::kInvalidArgument;
  }
  const long long routes = tokens * top_k;
  std::vector<int> counts(static_cast<std::size_t>(experts));
  for (long long row = 0; row < routes; ++row) {
    if (expert_ids[row] < 0 || expert_ids[row] >= experts) {
      return Status::kInvalidArgument;
    }
    ++counts[expert_ids[row]];
  }
  offsets[0] = 0;
  for (long long expert = 0; expert < experts; ++expert) {
    offsets[expert + 1] = offsets[expert] + counts[expert];
  }
  std::vector<int> cursor(offsets, offsets + experts);
  for (long long row = 0; row < routes; ++row) {
    const int destination = cursor[expert_ids[row]]++;
    sorted_rows[destination] = static_cast<int>(row / top_k);
    inverse[row] = destination;
  }
  return Status::kOk;
}

Status moe_pad_schedule(const int* sorted_rows, const int* offsets,
                        const int* inverse,
                        int* expert_of_tile, int* gather_rows,
                        int* inverse_padded, int* padded_offsets,
                        long long tokens, long long experts, int top_k,
                        int tile_rows) {
  if (!detail::valid_product({tokens, experts, top_k}) || tile_rows <= 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(sorted_rows, offsets, inverse, expert_of_tile,
                           gather_rows, inverse_padded, padded_offsets)) {
    return Status::kInvalidArgument;
  }
  const long long routes = tokens * top_k;
  if (offsets[0] != 0 || offsets[experts] != routes) {
    return Status::kInvalidArgument;
  }
  padded_offsets[0] = 0;
  for (long long expert = 0; expert < experts; ++expert) {
    if (offsets[expert] > offsets[expert + 1]) return Status::kInvalidArgument;
    const int count = offsets[expert + 1] - offsets[expert];
    const int padded = (count + tile_rows - 1) / tile_rows * tile_rows;
    padded_offsets[expert + 1] = padded_offsets[expert] + padded;
  }
  std::fill_n(gather_rows, padded_offsets[experts], -1);
  for (long long expert = 0; expert < experts; ++expert) {
    for (int route = offsets[expert]; route < offsets[expert + 1]; ++route) {
      const int destination = padded_offsets[expert] + route - offsets[expert];
      gather_rows[destination] = sorted_rows[route];
    }
    for (int row = padded_offsets[expert]; row < padded_offsets[expert + 1];
         row += tile_rows) {
      expert_of_tile[row / tile_rows] = static_cast<int>(expert);
    }
  }
  for (long long route = 0; route < routes; ++route) {
    const int compact = inverse[route];
    if (compact < 0 || compact >= routes) return Status::kInvalidArgument;
    const long long expert =
        std::upper_bound(offsets, offsets + experts + 1, compact) - offsets - 1;
    inverse_padded[route] =
        padded_offsets[expert] + compact - offsets[expert];
  }
  return Status::kOk;
}

Status moe_lora_align(
    const int* topk_ids, const int* token_lora_mapping,
    const int* lora_ids, const std::uint8_t* adapter_enabled,
    int* sorted_token_ids, int* expert_ids, int* tokens_post_pad,
    long long tokens, long long max_loras, long long num_experts,
    int top_k, int block_size, long long sorted_capacity_per_lora,
    long long expert_capacity_per_lora) {
  if (!detail::valid_product({tokens, max_loras, num_experts, top_k}) ||
      block_size <= 0 || sorted_capacity_per_lora <= 0 ||
      expert_capacity_per_lora <= 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(topk_ids, token_lora_mapping, lora_ids,
                           adapter_enabled, sorted_token_ids, expert_ids,
                           tokens_post_pad)) {
    return Status::kInvalidArgument;
  }
  const long long assignments = tokens * top_k;
  std::fill_n(sorted_token_ids, max_loras * sorted_capacity_per_lora,
              static_cast<int>(assignments));
  std::fill_n(expert_ids, max_loras * expert_capacity_per_lora, -1);
  std::fill_n(tokens_post_pad, max_loras, 0);
  std::vector<std::uint8_t> seen_lora(static_cast<std::size_t>(max_loras), 0);
  for (long long item = 0; item < max_loras; ++item) {
    const int lora = lora_ids[item];
    if (lora < 0) continue;
    if (lora >= max_loras || seen_lora[static_cast<std::size_t>(lora)]++) {
      return Status::kInvalidArgument;
    }
    if (adapter_enabled[lora] == 0) continue;
    std::vector<int> counts(static_cast<std::size_t>(num_experts), 0);
    for (long long route = 0; route < assignments; ++route) {
      if (token_lora_mapping[route / top_k] != lora) continue;
      if (topk_ids[route] >= 0 && topk_ids[route] < num_experts) {
        ++counts[static_cast<std::size_t>(topk_ids[route])];
      }
    }
    std::vector<int> offsets(static_cast<std::size_t>(num_experts + 1), 0);
    for (long long expert = 0; expert < num_experts; ++expert) {
      offsets[static_cast<std::size_t>(expert + 1)] =
          offsets[static_cast<std::size_t>(expert)] +
          (counts[static_cast<std::size_t>(expert)] + block_size - 1) /
              block_size * block_size;
    }
    const int padded = offsets.back();
    if (padded > sorted_capacity_per_lora ||
        (padded + block_size - 1) / block_size >
            expert_capacity_per_lora) {
      return Status::kInvalidShape;
    }
    tokens_post_pad[lora] = padded;
    std::vector<int> cursor(offsets.begin(), offsets.end() - 1);
    int* sorted = sorted_token_ids + lora * sorted_capacity_per_lora;
    int* tile_expert = expert_ids + lora * expert_capacity_per_lora;
    for (long long route = 0; route < assignments; ++route) {
      if (token_lora_mapping[route / top_k] != lora) continue;
      const int expert = topk_ids[route];
      if (expert < 0 || expert >= num_experts) continue;
      sorted[cursor[static_cast<std::size_t>(expert)]++] =
          static_cast<int>(route);
    }
    for (long long expert = 0; expert < num_experts; ++expert) {
      for (int row = offsets[static_cast<std::size_t>(expert)];
           row < offsets[static_cast<std::size_t>(expert + 1)];
           row += block_size) {
        tile_expert[row / block_size] = static_cast<int>(expert);
      }
    }
  }
  return Status::kOk;
}

Status moe_gather(const float* x, const int* gather_rows, float* out,
                  long long tokens, long long gathered_rows, long long dim) {
  if (!detail::valid_product({tokens, dim}) ||
      !detail::valid_product({gathered_rows, dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, gather_rows, out)) return Status::kInvalidArgument;
  for (long long row = 0; row < gathered_rows; ++row) {
    if (gather_rows[row] < -1 || gather_rows[row] >= tokens) {
      return Status::kInvalidArgument;
    }
  }
  threading::parallel_ranges(gathered_rows, 8,
                             [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      if (gather_rows[row] < 0) {
        std::fill_n(out + row * dim, dim, 0.0f);
      } else {
        std::copy_n(x + static_cast<long long>(gather_rows[row]) * dim, dim,
                    out + row * dim);
      }
    }
  });
  return Status::kOk;
}

Status moe_finalize(const float* expert_out, const int* inverse,
                    const float* expert_weights, float* out,
                    long long tokens, int top_k, long long dim) {
  if (!detail::valid_product({tokens, top_k, dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(expert_out, inverse, expert_weights, out)) {
    return Status::kInvalidArgument;
  }
  const long long routes = tokens * top_k;
  for (long long route = 0; route < routes; ++route) {
    if (inverse[route] < 0 || inverse[route] >= routes) {
      return Status::kInvalidArgument;
    }
  }
  threading::parallel_ranges(tokens, 8,
                             [&](long long begin, long long end, int) {
    for (long long token = begin; token < end; ++token) {
      float* destination = out + token * dim;
      std::fill_n(destination, dim, 0.0f);
      for (int rank = 0; rank < top_k; ++rank) {
        const long long route = token * top_k + rank;
        const float* source =
            expert_out + static_cast<long long>(inverse[route]) * dim;
        for (long long feature = 0; feature < dim; ++feature) {
          destination[feature] += expert_weights[route] * source[feature];
        }
      }
    }
  });
  return Status::kOk;
}

Status moe_finalize_backward(
    const float* grad_out, const float* expert_out, const int* inverse,
    const float* expert_weights, float* grad_expert_out,
    float* grad_expert_weights, long long tokens, int top_k, long long dim) {
  if (!detail::valid_product({tokens, top_k, dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(grad_out, expert_out, inverse, expert_weights,
                           grad_expert_out, grad_expert_weights)) {
    return Status::kInvalidArgument;
  }
  const long long routes = tokens * top_k;
  std::fill_n(grad_expert_out, routes * dim, 0.0f);
  for (long long token = 0; token < tokens; ++token) {
    for (int rank = 0; rank < top_k; ++rank) {
      const long long route = token * top_k + rank;
      const int permuted = inverse[route];
      if (permuted < 0 || permuted >= routes) return Status::kInvalidArgument;
      double weight_gradient = 0.0;
      for (long long item = 0; item < dim; ++item) {
        const float gradient = grad_out[token * dim + item];
        grad_expert_out[static_cast<long long>(permuted) * dim + item] +=
            expert_weights[route] * gradient;
        weight_gradient +=
            gradient * expert_out[static_cast<long long>(permuted) * dim + item];
      }
      grad_expert_weights[route] = static_cast<float>(weight_gradient);
    }
  }
  return Status::kOk;
}

Status moe_gather_backward(const float* grad_gathered,
                           const int* gather_rows, float* grad_input,
                           long long tokens, long long gathered_rows,
                           long long dim) {
  if (!detail::valid_product({tokens, dim}) ||
      !detail::valid_product({gathered_rows, dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(grad_gathered, gather_rows, grad_input)) {
    return Status::kInvalidArgument;
  }
  std::fill_n(grad_input, tokens * dim, 0.0f);
  for (long long row = 0; row < gathered_rows; ++row) {
    if (gather_rows[row] < 0) continue;
    if (gather_rows[row] >= tokens) return Status::kInvalidArgument;
    for (long long item = 0; item < dim; ++item) {
      grad_input[static_cast<long long>(gather_rows[row]) * dim + item] +=
          grad_gathered[row * dim + item];
    }
  }
  return Status::kOk;
}

Status moe_grouped_gemm(const float* x, const float* weights,
                        const int* expert_ids, float* out, long long rows,
                        long long experts, long long input_dim,
                        long long output_dim) {
  if (!detail::valid_product({rows, input_dim}) ||
      !detail::valid_product({experts, input_dim, output_dim}) ||
      !detail::valid_product({rows, output_dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, weights, expert_ids, out)) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < rows; ++row) {
    if (expert_ids[row] < 0 || expert_ids[row] >= experts) {
      return Status::kInvalidArgument;
    }
  }
  threading::parallel_ranges(rows, 1, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      const float* weight =
          weights + static_cast<long long>(expert_ids[row]) * input_dim * output_dim;
      for (long long output = 0; output < output_dim; ++output) {
        double accumulator = 0.0;
        for (long long input = 0; input < input_dim; ++input) {
          accumulator += double(x[row * input_dim + input]) *
                         weight[input * output_dim + output];
        }
        out[row * output_dim + output] = static_cast<float>(accumulator);
      }
    }
  });
  return Status::kOk;
}

Status moe_grouped_swiglu(const float* x, const float* weights,
                          const int* expert_ids, float* out, long long rows,
                          long long experts, long long input_dim,
                          long long intermediate_dim) {
  if (intermediate_dim > LLONG_MAX / 2 ||
      !detail::valid_product({rows, input_dim}) ||
      !detail::valid_product({experts, input_dim, 2 * intermediate_dim}) ||
      !detail::valid_product({rows, intermediate_dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, weights, expert_ids, out)) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < rows; ++row) {
    if (expert_ids[row] < 0 || expert_ids[row] >= experts) {
      return Status::kInvalidArgument;
    }
  }
  threading::parallel_ranges(rows, 1, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      const float* weight = weights +
                            static_cast<long long>(expert_ids[row]) * input_dim *
                                2 * intermediate_dim;
      for (long long output = 0; output < intermediate_dim; ++output) {
        double gate = 0.0;
        double up = 0.0;
        for (long long input = 0; input < input_dim; ++input) {
          const float value = x[row * input_dim + input];
          gate += double(value) * weight[input * 2 * intermediate_dim + output];
          up += double(value) *
                weight[input * 2 * intermediate_dim + intermediate_dim + output];
        }
        out[row * intermediate_dim + output] =
            static_cast<float>((gate / (1.0 + std::exp(-gate))) * up);
      }
    }
  });
  return Status::kOk;
}

Status moe_grouped_gemm_backward_input(
    const float* grad_out, const float* weights, const int* expert_ids,
    float* grad_input, long long rows, long long experts,
    long long input_dim, long long output_dim) {
  if (!detail::valid_product({rows, input_dim, output_dim}) ||
      !detail::valid_product({experts, input_dim, output_dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(grad_out, weights, expert_ids, grad_input)) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < rows; ++row) {
    const int expert = expert_ids[row];
    if (expert < 0 || expert >= experts) return Status::kInvalidArgument;
    const float* weight =
        weights + static_cast<long long>(expert) * input_dim * output_dim;
    for (long long input = 0; input < input_dim; ++input) {
      double sum = 0.0;
      for (long long output = 0; output < output_dim; ++output) {
        sum += grad_out[row * output_dim + output] *
               weight[input * output_dim + output];
      }
      grad_input[row * input_dim + input] = static_cast<float>(sum);
    }
  }
  return Status::kOk;
}

Status moe_grouped_gemm_backward_weight(
    const float* x, const float* grad_out, const int* expert_ids,
    float* grad_weights, long long rows, long long experts,
    long long input_dim, long long output_dim) {
  if (!detail::valid_product({rows, input_dim, output_dim}) ||
      !detail::valid_product({experts, input_dim, output_dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, grad_out, expert_ids, grad_weights)) {
    return Status::kInvalidArgument;
  }
  std::fill_n(grad_weights, experts * input_dim * output_dim, 0.0f);
  for (long long row = 0; row < rows; ++row) {
    const int expert = expert_ids[row];
    if (expert < 0 || expert >= experts) return Status::kInvalidArgument;
    float* gradient = grad_weights +
                      static_cast<long long>(expert) * input_dim * output_dim;
    for (long long input = 0; input < input_dim; ++input) {
      for (long long output = 0; output < output_dim; ++output) {
        gradient[input * output_dim + output] +=
            x[row * input_dim + input] * grad_out[row * output_dim + output];
      }
    }
  }
  return Status::kOk;
}

Status moe_grouped_qgemm(
    const float* x, const void* packed_weights, const int* expert_ids,
    const float* bias, float* out, long long rows, long long experts,
    long long input_dim, long long output_dim, QuantFormat format,
    bool use_bias) {
  if (!detail::valid_product({rows, experts, input_dim, output_dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, packed_weights, expert_ids, out) ||
      (use_bias && bias == nullptr)) {
    return Status::kInvalidArgument;
  }
  return grouped_qprojection(x, packed_weights, expert_ids, bias, out, rows,
                             experts, input_dim, output_dim, format, use_bias);
}

Status moe_grouped_qswiglu(
    const float* x, const void* packed_weights, const int* expert_ids,
    const float* bias, float* out, long long rows, long long experts,
    long long input_dim, long long intermediate_dim, QuantFormat format,
    bool use_bias, bool oai_mode, float alpha, float limit) {
  if (!detail::valid_product({rows, experts, input_dim, intermediate_dim}) ||
      intermediate_dim > LLONG_MAX / 2 || !std::isfinite(alpha) ||
      alpha <= 0.0f || !std::isfinite(limit) || limit < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, packed_weights, expert_ids, out) ||
      (use_bias && bias == nullptr)) {
    return Status::kInvalidArgument;
  }
  const long long outputs = 2 * intermediate_dim;
  if (!detail::valid_product({rows, outputs})) return Status::kInvalidShape;
  std::size_t expert_bytes = 0;
  Status status =
      qgemv_packed_size(format, outputs, input_dim, &expert_bytes);
  if (status != Status::kOk) return status;
  long long block_size = 0;
  std::size_t block_bytes = 0;
  if (!quant::gguf_format_info(format, &block_size, &block_bytes)) {
    return Status::kUnsupportedFormat;
  }
  const long long blocks_per_row = input_dim / block_size;
  const std::size_t weight_row_bytes =
      static_cast<std::size_t>(blocks_per_row) * block_bytes;

  std::vector<long long> offsets(static_cast<std::size_t>(experts + 1), 0);
  for (long long row = 0; row < rows; ++row) {
    const int expert = expert_ids[row];
    if (expert < 0 || expert >= experts) return Status::kInvalidArgument;
    ++offsets[static_cast<std::size_t>(expert + 1)];
  }
  for (long long expert = 0; expert < experts; ++expert) {
    offsets[static_cast<std::size_t>(expert + 1)] +=
        offsets[static_cast<std::size_t>(expert)];
  }
  std::vector<long long> cursor = offsets;
  std::vector<long long> order(static_cast<std::size_t>(rows));
  for (long long row = 0; row < rows; ++row) {
    order[static_cast<std::size_t>(
        cursor[static_cast<std::size_t>(expert_ids[row])]++)] = row;
  }
  std::vector<int> active;
  for (long long expert = 0; expert < experts; ++expert) {
    if (offsets[static_cast<std::size_t>(expert)] !=
        offsets[static_cast<std::size_t>(expert + 1)]) {
      active.push_back(static_cast<int>(expert));
    }
  }

  const auto* weights = static_cast<const std::uint8_t*>(packed_weights);
  const long long tasks =
      static_cast<long long>(active.size()) * intermediate_dim;
  threading::parallel_ranges(tasks, 8,
                             [&](long long begin, long long end, int) {
    thread_local std::vector<float> gate_accumulators;
    thread_local std::vector<float> up_accumulators;
    alignas(64) float gate_weights[256];
    alignas(64) float up_weights[256];
    for (long long task = begin; task < end; ++task) {
      const int expert =
          active[static_cast<std::size_t>(task / intermediate_dim)];
      const long long output = task % intermediate_dim;
      const long long first = offsets[static_cast<std::size_t>(expert)];
      const long long last = offsets[static_cast<std::size_t>(expert + 1)];
      const long long count = last - first;
      gate_accumulators.assign(static_cast<std::size_t>(count), 0.0f);
      up_accumulators.assign(static_cast<std::size_t>(count), 0.0f);
      const std::uint8_t* expert_weights =
          weights + static_cast<std::size_t>(expert) * expert_bytes;
      const std::uint8_t* gate_row =
          expert_weights + static_cast<std::size_t>(output) * weight_row_bytes;
      const std::uint8_t* up_row =
          expert_weights +
          static_cast<std::size_t>(intermediate_dim + output) *
              weight_row_bytes;
      for (long long block = 0; block < blocks_per_row; ++block) {
        const std::uint8_t* gate_block =
            gate_row + static_cast<std::size_t>(block) * block_bytes;
        const std::uint8_t* up_block =
            up_row + static_cast<std::size_t>(block) * block_bytes;
        if (format != QuantFormat::kQ4_0) {
          quant::gguf_dequant_block_ref(format, gate_block, gate_weights);
          quant::gguf_dequant_block_ref(format, up_block, up_weights);
        }
        const long long input_base = block * block_size;
        for (long long routed = 0; routed < count; ++routed) {
          const long long row =
              order[static_cast<std::size_t>(first + routed)];
          const float* input = x + row * input_dim + input_base;
          if (format == QuantFormat::kQ4_0) {
            gate_accumulators[static_cast<std::size_t>(routed)] += q4_block_dot(
                *reinterpret_cast<const quant::BlockQ4_0*>(gate_block), input);
            up_accumulators[static_cast<std::size_t>(routed)] += q4_block_dot(
                *reinterpret_cast<const quant::BlockQ4_0*>(up_block), input);
          } else {
            gate_accumulators[static_cast<std::size_t>(routed)] +=
                decoded_block_dot(gate_weights, input, block_size);
            up_accumulators[static_cast<std::size_t>(routed)] +=
                decoded_block_dot(up_weights, input, block_size);
          }
        }
      }
      for (long long routed = 0; routed < count; ++routed) {
        const long long row = order[static_cast<std::size_t>(first + routed)];
        float gate = gate_accumulators[static_cast<std::size_t>(routed)];
        float up = up_accumulators[static_cast<std::size_t>(routed)];
        if (use_bias) {
          const long long bias_base = static_cast<long long>(expert) * outputs;
          gate += bias[bias_base + output];
          up += bias[bias_base + intermediate_dim + output];
        }
        if (oai_mode) {
          gate = std::min(gate, limit);
          up = std::clamp(up, -limit, limit);
          out[row * intermediate_dim + output] =
              gate / (1.0f + std::exp(-alpha * gate)) * (1.0f + up);
        } else {
          out[row * intermediate_dim + output] =
              gate / (1.0f + std::exp(-gate)) * up;
        }
      }
    }
  });
  return Status::kOk;
}

Status moe_gemm_fp8(const float* x, const std::uint8_t* weight_codes,
                    const float* weight_scales, const int* expert_ids,
                    float* out, long long rows, long long experts,
                    long long input_dim, long long output_dim) {
  if (!detail::valid_product({rows, experts, input_dim, output_dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, weight_codes, weight_scales, expert_ids, out)) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < rows; ++row) {
    if (expert_ids[row] < -1 || expert_ids[row] >= experts) {
      return Status::kInvalidArgument;
    }
  }
  for (long long i = 0; i < experts * output_dim; ++i) {
    if (!std::isfinite(weight_scales[i])) return Status::kInvalidArgument;
  }
  threading::parallel_ranges(rows * output_dim, 16,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long row = item / output_dim;
      const long long output = item % output_dim;
      const int expert = expert_ids[row];
      if (expert < 0) {
        out[item] = 0.0f;
        continue;
      }
      const long long weight_row =
          (static_cast<long long>(expert) * output_dim + output) * input_dim;
      double sum = 0.0;
      for (long long input = 0; input < input_dim; ++input) {
        sum += x[row * input_dim + input] *
               float8_decode(weight_codes[weight_row + input],
                             Float8Format::kE4M3FN);
      }
      out[item] = static_cast<float>(
          sum * weight_scales[static_cast<long long>(expert) * output_dim +
                              output]);
    }
  });
  return Status::kOk;
}

Status moe_gemm_wna16(const float* x, const std::uint32_t* packed_weights,
                      const float* scales,
                      const std::uint8_t* zero_points,
                      const int* expert_ids, float* out, long long rows,
                      long long experts, long long input_dim,
                      long long output_dim, long long group_size, int bits) {
  if (!detail::valid_product({rows, experts, input_dim, output_dim,
                              group_size}) ||
      input_dim % group_size != 0 || (bits != 4 && bits != 8) ||
      input_dim % (32 / bits) != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, packed_weights, scales, expert_ids, out)) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < rows; ++row) {
    if (expert_ids[row] < -1 || expert_ids[row] >= experts) {
      return Status::kInvalidArgument;
    }
  }
  const int pack = 32 / bits;
  const long long groups = input_dim / group_size;
  const long long packed_k = input_dim / pack;
  for (long long i = 0; i < experts * output_dim * groups; ++i) {
    if (!std::isfinite(scales[i])) return Status::kInvalidArgument;
  }
  threading::parallel_ranges(rows * output_dim, 16,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long row = item / output_dim;
      const long long output = item % output_dim;
      const int expert = expert_ids[row];
      if (expert < 0) {
        out[item] = 0.0f;
        continue;
      }
      const long long weight_base =
          (static_cast<long long>(expert) * output_dim + output) * packed_k;
      double sum = 0.0;
      for (long long input = 0; input < input_dim; ++input) {
        const int inner = static_cast<int>(input % pack);
        const int local =
            bits == 4
                ? (inner < 4 ? 2 * inner : 2 * (inner - 4) + 1)
                : (inner < 2 ? 2 * inner : 2 * (inner - 2) + 1);
        const std::uint32_t packed =
            packed_weights[weight_base + input / pack];
        const unsigned mask = bits == 4 ? 0x0fu : 0xffu;
        const float code = static_cast<float>((packed >> (local * bits)) & mask);
        const long long group = input / group_size;
        const long long scale_index =
            (static_cast<long long>(expert) * output_dim + output) * groups +
            group;
        float zero = bits == 4 ? 8.0f : 128.0f;
        if (zero_points != nullptr) {
          if (bits == 4) {
            const long long zero_base =
                static_cast<long long>(expert) * ((output_dim + 1) / 2) *
                groups;
            const std::uint8_t byte =
                zero_points[zero_base + (output / 2) * groups + group];
            zero = static_cast<float>((byte >> (4 * (output & 1))) & 15);
          } else {
            zero = zero_points[scale_index];
          }
        }
        sum += x[row * input_dim + input] *
               (code - zero) * scales[scale_index];
      }
      out[item] = static_cast<float>(sum);
    }
  });
  return Status::kOk;
}

Status moe_gemm_nvfp4(
    const std::uint8_t* activation_codes,
    const std::uint8_t* weight_codes,
    const std::uint8_t* activation_scale_codes,
    const std::uint8_t* weight_scale_codes, const float* expert_scale,
    const int* expert_ids, float* out, long long rows, long long experts,
    long long input_dim, long long output_dim) {
  if (!detail::valid_product({rows, experts, input_dim, output_dim}) ||
      input_dim % 16 != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(activation_codes, weight_codes,
                           activation_scale_codes, weight_scale_codes,
                           expert_scale, expert_ids, out)) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < rows; ++row) {
    if (expert_ids[row] < -1 || expert_ids[row] >= experts) {
      return Status::kInvalidArgument;
    }
  }
  for (long long expert = 0; expert < experts; ++expert) {
    if (!std::isfinite(expert_scale[expert]) || expert_scale[expert] < 0.0f) {
      return Status::kInvalidArgument;
    }
  }
  const long long groups = input_dim / 16;
  for (long long i = 0; i < rows * groups; ++i) {
    if (!std::isfinite(float8_decode(activation_scale_codes[i],
                                     Float8Format::kE4M3FN))) {
      return Status::kInvalidArgument;
    }
  }
  for (long long i = 0; i < experts * output_dim * groups; ++i) {
    if (!std::isfinite(float8_decode(weight_scale_codes[i],
                                     Float8Format::kE4M3FN))) {
      return Status::kInvalidArgument;
    }
  }
  const long long weight_bytes = output_dim * input_dim / 2;
  const long long weight_scales = output_dim * (input_dim / 16);
  threading::parallel_ranges(rows, 1, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      const int expert = expert_ids[row];
      if (expert < 0) {
        std::fill_n(out + row * output_dim, output_dim, 0.0f);
        continue;
      }
      const Status status = nvfp4_gemm(
          activation_codes + row * (input_dim / 2),
          activation_scale_codes + row * (input_dim / 16), 1.0f,
          weight_codes + static_cast<long long>(expert) * weight_bytes,
          weight_scale_codes +
              static_cast<long long>(expert) * weight_scales,
          expert_scale[expert], out + row * output_dim, 1, output_dim,
          input_dim);
      if (status != Status::kOk) {
        std::fill_n(out + row * output_dim, output_dim,
                    std::numeric_limits<float>::quiet_NaN());
      }
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
