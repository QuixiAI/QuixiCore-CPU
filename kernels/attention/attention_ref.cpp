#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

float default_scale(long long width, float scale) {
  return scale == 0.0f
             ? static_cast<float>(1.0 / std::sqrt(static_cast<double>(width)))
             : scale;
}

double dot_f32_precise(const float* lhs, const float* rhs, long long count) {
  double total = 0.0;
  long long item = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
  float64x2_t sum0 = vdupq_n_f64(0.0);
  float64x2_t sum1 = vdupq_n_f64(0.0);
  for (; item + 3 < count; item += 4) {
    const float32x4_t left = vld1q_f32(lhs + item);
    const float32x4_t right = vld1q_f32(rhs + item);
    sum0 = vfmaq_f64(sum0, vcvt_f64_f32(vget_low_f32(left)),
                     vcvt_f64_f32(vget_low_f32(right)));
    sum1 = vfmaq_f64(sum1, vcvt_f64_f32(vget_high_f32(left)),
                     vcvt_f64_f32(vget_high_f32(right)));
  }
  total = vaddvq_f64(vaddq_f64(sum0, sum1));
#endif
  for (; item < count; ++item) {
    total += static_cast<double>(lhs[item]) * rhs[item];
  }
  return total;
}

void replace_weighted(float* destination, const float* values,
                      double old_weight, double new_weight,
                      long long count) {
  long long item = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
  const float64x2_t old_broadcast = vdupq_n_f64(old_weight);
  const float64x2_t new_broadcast = vdupq_n_f64(new_weight);
  for (; item + 1 < count; item += 2) {
    const float64x2_t previous =
        vmulq_f64(vcvt_f64_f32(vld1_f32(destination + item)), old_broadcast);
    const float64x2_t added =
        vmulq_f64(vcvt_f64_f32(vld1_f32(values + item)), new_broadcast);
    vst1_f32(destination + item, vcvt_f32_f64(vaddq_f64(previous, added)));
  }
#endif
  for (; item < count; ++item) {
    destination[item] = static_cast<float>(destination[item] * old_weight +
                                            values[item] * new_weight);
  }
}

void add_weighted(float* destination, const float* values, double weight,
                  long long count) {
  long long item = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
  const float64x2_t broadcast = vdupq_n_f64(weight);
  for (; item + 3 < count; item += 4) {
    const float64x2_t product0 = vmulq_f64(
        vcvt_f64_f32(vld1_f32(values + item)), broadcast);
    const float64x2_t product1 = vmulq_f64(
        vcvt_f64_f32(vld1_f32(values + item + 2)), broadcast);
    vst1q_f32(destination + item,
              vaddq_f32(vld1q_f32(destination + item),
                        vcombine_f32(vcvt_f32_f64(product0),
                                     vcvt_f32_f64(product1))));
  }
#endif
  for (; item < count; ++item) {
    destination[item] += static_cast<float>(values[item] * weight);
  }
}

void scale_f32(float* values, float scale, long long count) {
  long long item = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
  const float32x4_t broadcast = vdupq_n_f32(scale);
  for (; item + 3 < count; item += 4) {
    vst1q_f32(values + item,
              vmulq_f32(vld1q_f32(values + item), broadcast));
  }
#endif
  for (; item < count; ++item) values[item] *= scale;
}

}  // namespace

Status rope(const float* x, float* y, long long tokens, long long heads,
            long long head_dim, float base, long long pos0) {
  if (!detail::valid_product({tokens, heads, head_dim}) ||
      head_dim % 2 != 0 || pos0 < 0 || pos0 > LLONG_MAX - tokens ||
      !std::isfinite(base) || base <= 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, y)) {
    return Status::kInvalidArgument;
  }
  const long long half = head_dim / 2;
  threading::parallel_ranges(tokens * heads, 8,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long token = item / heads;
      const long long offset = item * head_dim;
      const double position = static_cast<double>(pos0 + token);
      for (long long i = 0; i < half; ++i) {
        const double frequency =
            std::pow(static_cast<double>(base),
                     -2.0 * static_cast<double>(i) /
                         static_cast<double>(head_dim));
        const double angle = position * frequency;
        const float cosine = static_cast<float>(std::cos(angle));
        const float sine = static_cast<float>(std::sin(angle));
        const float first = x[offset + i];
        const float second = x[offset + half + i];
        y[offset + i] = first * cosine - second * sine;
        y[offset + half + i] = second * cosine + first * sine;
      }
    }
  });
  return Status::kOk;
}

Status rope_interleaved_to_split(const float* x, float* y, long long tokens,
                                 long long heads, long long head_dim,
                                 float base, long long pos0) {
  if (!detail::valid_product({tokens, heads, head_dim}) ||
      head_dim % 2 != 0 || pos0 < 0 || pos0 > LLONG_MAX - tokens ||
      !std::isfinite(base) || base <= 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, y)) return Status::kInvalidArgument;
  const long long half = head_dim / 2;
  threading::parallel_ranges(tokens * heads, 8,
                             [&](long long begin, long long end, int) {
    thread_local std::vector<float> scratch;
    if (x == y && static_cast<long long>(scratch.size()) < head_dim) {
      scratch.resize(static_cast<std::size_t>(head_dim));
    }
    for (long long item = begin; item < end; ++item) {
      const long long token = item / heads;
      const long long offset = item * head_dim;
      const float* source = x + offset;
      if (x == y) {
        std::copy_n(source, head_dim, scratch.data());
        source = scratch.data();
      }
      const double position = static_cast<double>(pos0 + token);
      for (long long pair = 0; pair < half; ++pair) {
        const double frequency =
            std::pow(static_cast<double>(base),
                     -2.0 * static_cast<double>(pair) /
                         static_cast<double>(head_dim));
        const double angle = position * frequency;
        const float cosine = static_cast<float>(std::cos(angle));
        const float sine = static_cast<float>(std::sin(angle));
        const float first = source[2 * pair];
        const float second = source[2 * pair + 1];
        y[offset + pair] = first * cosine - second * sine;
        y[offset + half + pair] = second * cosine + first * sine;
      }
    }
  });
  return Status::kOk;
}

Status attention(const float* q, const float* k, const float* v, float* out,
                 long long query_heads, long long kv_heads,
                 long long query_length, long long kv_length,
                 long long head_dim, bool causal) {
  if (!detail::valid_product({query_heads, query_length, head_dim}) ||
      !detail::valid_product({kv_heads, kv_length, head_dim}) ||
      query_heads % kv_heads != 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, k, v, out)) {
    return Status::kInvalidArgument;
  }
  const float scale = default_scale(head_dim, 0.0f);
  const long long queries = query_heads * query_length;
  threading::parallel_ranges(queries, 1,
                             [&](long long begin, long long end, int) {
    for (long long query_index = begin; query_index < end; ++query_index) {
      const long long head = query_index / query_length;
      const long long position = query_index % query_length;
      const long long kv_head = head / (query_heads / kv_heads);
      const float* query = q + query_index * head_dim;
      float* destination = out + query_index * head_dim;
      std::fill(destination, destination + head_dim, 0.0f);

      double maximum = -std::numeric_limits<double>::infinity();
      double denominator = 0.0;
      const long long causal_limit = position + (kv_length - query_length);
      for (long long key_position = 0; key_position < kv_length;
           ++key_position) {
        if (causal && key_position > causal_limit) {
          continue;
        }
        const float* key =
            k + (kv_head * kv_length + key_position) * head_dim;
        double score = dot_f32_precise(query, key, head_dim);
        score *= scale;
        const double next_maximum = std::max(maximum, score);
        const double old_weight = std::exp(maximum - next_maximum);
        const double new_weight = std::exp(score - next_maximum);
        denominator = denominator * old_weight + new_weight;
        const float* value =
            v + (kv_head * kv_length + key_position) * head_dim;
        replace_weighted(destination, value, old_weight, new_weight, head_dim);
        maximum = next_maximum;
      }
      if (denominator > 0.0) {
        const float inverse = static_cast<float>(1.0 / denominator);
        scale_f32(destination, inverse, head_dim);
      }
    }
  });
  return Status::kOk;
}

Status paged_attention(const float* q, const float* key_cache,
                       const float* value_cache, const int* block_table,
                       const int* context_lens, float* out,
                       long long cache_blocks, long long batch,
                       long long query_heads, long long kv_heads,
                       long long head_dim, long long page_size,
                       long long max_blocks, float scale, long long window) {
  if (!detail::valid_product({cache_blocks, page_size, kv_heads, head_dim}) ||
      !detail::valid_product({batch, query_heads, head_dim}) ||
      !detail::valid_product({batch, max_blocks}) ||
      query_heads % kv_heads != 0 || !std::isfinite(scale) || window < 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, key_cache, value_cache, block_table,
                           context_lens, out)) {
    return Status::kInvalidArgument;
  }
  const float score_scale = default_scale(head_dim, scale);
  for (long long request = 0; request < batch; ++request) {
    const int context = context_lens[request];
    if (context < 0 ||
        static_cast<long long>(context) > max_blocks * page_size) {
      return Status::kInvalidArgument;
    }
    const long long first =
        window > 0 ? std::max(0LL, static_cast<long long>(context) - window)
                   : 0;
    for (long long head = 0; head < query_heads; ++head) {
      const long long kv_head = head / (query_heads / kv_heads);
      const float* query =
          q + (request * query_heads + head) * head_dim;
      float* destination =
          out + (request * query_heads + head) * head_dim;
      std::fill(destination, destination + head_dim, 0.0f);
      double maximum = -std::numeric_limits<double>::infinity();
      double denominator = 0.0;
      for (long long position = first; position < context; ++position) {
        const long long table_index = request * max_blocks + position / page_size;
        const int block = block_table[table_index];
        if (block < 0 || block >= cache_blocks) {
          return Status::kInvalidArgument;
        }
        const long long cache_offset =
            ((static_cast<long long>(block) * page_size + position % page_size) *
                 kv_heads +
             kv_head) *
            head_dim;
        const float* key = key_cache + cache_offset;
        double score = dot_f32_precise(query, key, head_dim);
        score *= score_scale;
        const float* value = value_cache + cache_offset;
        if (score > maximum) {
          const double old_weight = std::exp(maximum - score);
          denominator = denominator * old_weight + 1.0;
          replace_weighted(destination, value, old_weight, 1.0, head_dim);
          maximum = score;
        } else {
          const double weight = std::exp(score - maximum);
          denominator += weight;
          add_weighted(destination, value, weight, head_dim);
        }
      }
      if (denominator > 0.0) {
        const float inverse = static_cast<float>(1.0 / denominator);
        scale_f32(destination, inverse, head_dim);
      }
    }
  }
  return Status::kOk;
}

Status mla_decode(const float* q, const float* kv_cache,
                  const int* block_table, const int* context_lens, float* out,
                  long long cache_blocks, long long batch, long long heads,
                  long long latent_dim, long long rope_dim, long long page_size,
                  long long max_blocks, float scale) {
  const long long width = latent_dim + rope_dim;
  if (latent_dim <= 0 || rope_dim < 0 || width <= 0 ||
      !detail::valid_product({cache_blocks, page_size, width}) ||
      !detail::valid_product({batch, heads, width}) ||
      !detail::valid_product({batch, heads, latent_dim}) ||
      !detail::valid_product({batch, max_blocks}) || !std::isfinite(scale)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(q, kv_cache, block_table, context_lens, out)) {
    return Status::kInvalidArgument;
  }
  const float score_scale = default_scale(width, scale);
  for (long long request = 0; request < batch; ++request) {
    const int context = context_lens[request];
    if (context < 0 ||
        static_cast<long long>(context) > max_blocks * page_size) {
      return Status::kInvalidArgument;
    }
    for (long long position = 0; position < context; ++position) {
      const int block =
          block_table[request * max_blocks + position / page_size];
      if (block < 0 || block >= cache_blocks) {
        return Status::kInvalidArgument;
      }
    }
  }
  threading::parallel_ranges(batch * heads, 1,
                             [&](long long begin, long long end, int) {
    for (long long query_index = begin; query_index < end; ++query_index) {
      const long long request = query_index / heads;
      const long long head = query_index % heads;
      const int context = context_lens[request];
      const float* query = q + (request * heads + head) * width;
      float* destination = out + (request * heads + head) * latent_dim;
      std::fill(destination, destination + latent_dim, 0.0f);
      double maximum = -std::numeric_limits<double>::infinity();
      double denominator = 0.0;
      for (long long position = 0; position < context; ++position) {
        const int block =
            block_table[request * max_blocks + position / page_size];
        const float* item =
            kv_cache +
            (static_cast<long long>(block) * page_size + position % page_size) *
                width;
        double score = dot_f32_precise(query, item, width);
        score *= score_scale;
        if (score > maximum) {
          const double old_weight = std::exp(maximum - score);
          denominator = denominator * old_weight + 1.0;
          replace_weighted(destination, item, old_weight, 1.0, latent_dim);
          maximum = score;
        } else {
          const double weight = std::exp(score - maximum);
          denominator += weight;
          add_weighted(destination, item, weight, latent_dim);
        }
      }
      if (denominator > 0.0) {
        const float inverse = static_cast<float>(1.0 / denominator);
        scale_f32(destination, inverse, latent_dim);
      }
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
