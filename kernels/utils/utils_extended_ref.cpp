#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

double sample_uniform(std::uint32_t seed, std::uint64_t row) {
  std::uint64_t z = (static_cast<std::uint64_t>(seed) << 32) ^ row ^
                    0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return (static_cast<double>(z >> 11) + 0.5) *
         (1.0 / 9007199254740992.0);
}

double transformed_logit(float value, float softcap) {
  return softcap > 0.0f
             ? static_cast<double>(softcap) *
                   std::tanh(static_cast<double>(value) / softcap)
             : value;
}

double transformed_derivative(float value, float softcap) {
  if (softcap <= 0.0f) return 1.0;
  const double t = std::tanh(static_cast<double>(value) / softcap);
  return 1.0 - t * t;
}

}  // namespace

Status add(const float* x, const float* y, float* out, long long count) {
  if (!detail::valid_product({count})) return Status::kInvalidShape;
  if (!detail::all_nonnull(x, y, out)) return Status::kInvalidArgument;
  threading::parallel_ranges(count, 16384,
                             [&](long long begin, long long end, int) {
    for (long long i = begin; i < end; ++i) out[i] = x[i] + y[i];
  });
  return Status::kOk;
}

Status cross_entropy_forward(const float* logits, const int* target,
                             float* loss, float* logsumexp, long long rows,
                             long long vocab, int ignore_index,
                             float label_smoothing, float z_loss,
                             float softcap) {
  if (!detail::valid_product({rows, vocab}) ||
      !std::isfinite(label_smoothing) || label_smoothing < 0.0f ||
      label_smoothing > 1.0f || !std::isfinite(z_loss) || z_loss < 0.0f ||
      !std::isfinite(softcap) || softcap < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(logits, target, loss, logsumexp)) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < rows; ++row) {
    if (target[row] != ignore_index &&
        (target[row] < 0 || target[row] >= vocab)) {
      return Status::kInvalidArgument;
    }
  }
  threading::parallel_ranges(rows, 4,
                             [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      if (target[row] == ignore_index) {
        loss[row] = 0.0f;
        logsumexp[row] = 0.0f;
        continue;
      }
      const float* lr = logits + row * vocab;
      double maximum = -std::numeric_limits<double>::infinity();
      double sum_logits = 0.0;
      for (long long token = 0; token < vocab; ++token) {
        const double value = transformed_logit(lr[token], softcap);
        maximum = std::max(maximum, value);
        sum_logits += value;
      }
      double sum_exp = 0.0;
      for (long long token = 0; token < vocab; ++token) {
        sum_exp += std::exp(transformed_logit(lr[token], softcap) - maximum);
      }
      const double lse = maximum + std::log(sum_exp);
      const double nll = lse - transformed_logit(lr[target[row]], softcap);
      const double smooth = lse - sum_logits / vocab;
      loss[row] = static_cast<float>((1.0 - label_smoothing) * nll +
                                     label_smoothing * smooth +
                                     z_loss * lse * lse);
      logsumexp[row] = static_cast<float>(lse);
    }
  });
  return Status::kOk;
}

Status cross_entropy_backward(const float* logits, const int* target,
                              const float* grad_out, float* grad_logits,
                              long long rows, long long vocab,
                              int ignore_index, float label_smoothing,
                              float z_loss, float softcap) {
  if (!detail::valid_product({rows, vocab}) ||
      !std::isfinite(label_smoothing) || label_smoothing < 0.0f ||
      label_smoothing > 1.0f || !std::isfinite(z_loss) || z_loss < 0.0f ||
      !std::isfinite(softcap) || softcap < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(logits, target, grad_out, grad_logits)) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < rows; ++row) {
    if (target[row] != ignore_index &&
        (target[row] < 0 || target[row] >= vocab)) {
      return Status::kInvalidArgument;
    }
  }
  threading::parallel_ranges(rows, 4,
                             [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      float* gr = grad_logits + row * vocab;
      if (target[row] == ignore_index) {
        std::fill_n(gr, vocab, 0.0f);
        continue;
      }
      const float* lr = logits + row * vocab;
      double maximum = -std::numeric_limits<double>::infinity();
      for (long long token = 0; token < vocab; ++token) {
        maximum = std::max(maximum, transformed_logit(lr[token], softcap));
      }
      double sum_exp = 0.0;
      for (long long token = 0; token < vocab; ++token) {
        sum_exp += std::exp(transformed_logit(lr[token], softcap) - maximum);
      }
      const double lse = maximum + std::log(sum_exp);
      for (long long token = 0; token < vocab; ++token) {
        const double probability =
            std::exp(transformed_logit(lr[token], softcap) - lse);
        const double target_probability =
            label_smoothing / vocab +
            (token == target[row] ? 1.0 - label_smoothing : 0.0);
        const double derivative = probability - target_probability +
                                  2.0 * z_loss * lse * probability;
        gr[token] = static_cast<float>(
            grad_out[row] * derivative *
            transformed_derivative(lr[token], softcap));
      }
    }
  });
  return Status::kOk;
}

Status typical_p_sample(const float* logits, int* out, long long rows,
                        long long vocab, float p, float temperature,
                        std::uint32_t seed) {
  if (!detail::valid_product({rows, vocab}) || !std::isfinite(p) || p <= 0.0f ||
      p > 1.0f || !std::isfinite(temperature) || temperature < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(logits, out)) return Status::kInvalidArgument;
  if (temperature == 0.0f) return argmax_sample(logits, out, rows, vocab);
  for (long long row = 0; row < rows; ++row) {
    const float* lr = logits + row * vocab;
    float maximum = -std::numeric_limits<float>::infinity();
    for (long long token = 0; token < vocab; ++token) {
      if (std::isnan(lr[token]) || lr[token] == INFINITY) {
        return Status::kInvalidArgument;
      }
      maximum = std::max(maximum, lr[token]);
    }
    if (maximum == -std::numeric_limits<float>::infinity()) {
      out[row] = 0;
      continue;
    }
    std::vector<double> probabilities(static_cast<std::size_t>(vocab));
    double total = 0.0;
    for (long long token = 0; token < vocab; ++token) {
      probabilities[token] =
          std::exp((static_cast<double>(lr[token]) - maximum) / temperature);
      total += probabilities[token];
    }
    double entropy = 0.0;
    for (double& probability : probabilities) {
      probability /= total;
      if (probability > 0.0) entropy -= probability * std::log(probability);
    }
    std::vector<int> ids(static_cast<std::size_t>(vocab));
    std::iota(ids.begin(), ids.end(), 0);
    std::stable_sort(ids.begin(), ids.end(), [&](int lhs, int rhs) {
      const double ld = std::fabs(-std::log(probabilities[lhs]) - entropy);
      const double rd = std::fabs(-std::log(probabilities[rhs]) - entropy);
      return ld == rd ? lhs < rhs : ld < rd;
    });
    double kept = 0.0;
    std::size_t keep = 0;
    do {
      kept += probabilities[ids[keep++]];
    } while (keep < ids.size() && kept < p);
    const double threshold = sample_uniform(seed, row) * kept;
    double cumulative = 0.0;
    out[row] = ids[keep - 1];
    for (std::size_t i = 0; i < keep; ++i) {
      cumulative += probabilities[ids[i]];
      if (threshold < cumulative) {
        out[row] = ids[i];
        break;
      }
    }
  }
  return Status::kOk;
}

Status apply_token_bitmask(const float* logits, const std::uint8_t* bitmask,
                           float* out, long long rows, long long vocab) {
  if (!detail::valid_product({rows, vocab})) return Status::kInvalidShape;
  if (!detail::all_nonnull(logits, bitmask, out)) return Status::kInvalidArgument;
  const long long stride = (vocab + 7) / 8;
  for (long long row = 0; row < rows; ++row) {
    for (long long token = 0; token < vocab; ++token) {
      const bool allowed =
          (bitmask[row * stride + token / 8] & (0x80u >> (token % 8))) != 0;
      out[row * vocab + token] =
          allowed ? logits[row * vocab + token]
                  : -std::numeric_limits<float>::infinity();
    }
  }
  return Status::kOk;
}

Status apply_bad_words(const float* logits, const int* bad_ids, float* out,
                       long long rows, long long vocab, long long bad_count) {
  if (!detail::valid_product({rows, vocab}) || bad_count < 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(logits, bad_ids, out)) return Status::kInvalidArgument;
  for (long long i = 0; i < bad_count; ++i) {
    if (bad_ids[i] < 0 || bad_ids[i] >= vocab) return Status::kInvalidArgument;
  }
  std::copy_n(logits, rows * vocab, out);
  for (long long row = 0; row < rows; ++row) {
    for (long long i = 0; i < bad_count; ++i) {
      out[row * vocab + bad_ids[i]] = -std::numeric_limits<float>::infinity();
    }
  }
  return Status::kOk;
}

Status apply_repetition_penalty(const float* logits, const int* previous,
                                const int* lengths, float* out,
                                long long rows, long long vocab,
                                long long max_previous,
                                float repetition_penalty,
                                float presence_penalty,
                                float frequency_penalty) {
  if (!detail::valid_product({rows, vocab}) || max_previous < 0 ||
      !std::isfinite(repetition_penalty) || repetition_penalty <= 0.0f ||
      !std::isfinite(presence_penalty) || !std::isfinite(frequency_penalty)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(logits, previous, lengths, out)) {
    return Status::kInvalidArgument;
  }
  std::copy_n(logits, rows * vocab, out);
  std::vector<int> counts(static_cast<std::size_t>(vocab));
  for (long long row = 0; row < rows; ++row) {
    if (lengths[row] < 0 || lengths[row] > max_previous) {
      return Status::kInvalidArgument;
    }
    std::fill(counts.begin(), counts.end(), 0);
    for (int i = 0; i < lengths[row]; ++i) {
      const int id = previous[row * max_previous + i];
      if (id < 0 || id >= vocab) return Status::kInvalidArgument;
      ++counts[id];
    }
    for (long long token = 0; token < vocab; ++token) {
      if (counts[token] == 0) continue;
      float& value = out[row * vocab + token];
      value = value < 0.0f ? value * repetition_penalty
                           : value / repetition_penalty;
      value -= presence_penalty + frequency_penalty * counts[token];
    }
  }
  return Status::kOk;
}

Status top_k_renorm(const float* probabilities, float* out, long long rows,
                    long long vocab, int k) {
  if (!detail::valid_product({rows, vocab}) || k <= 0 || k > vocab) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(probabilities, out)) return Status::kInvalidArgument;
  for (long long row = 0; row < rows; ++row) {
    std::vector<int> ids(static_cast<std::size_t>(vocab));
    std::iota(ids.begin(), ids.end(), 0);
    const float* pr = probabilities + row * vocab;
    std::vector<float> alias_copy;
    if (probabilities == out) {
      alias_copy.assign(pr, pr + vocab);
      pr = alias_copy.data();
    }
    std::stable_sort(ids.begin(), ids.end(), [&](int lhs, int rhs) {
      return pr[lhs] == pr[rhs] ? lhs < rhs : pr[lhs] > pr[rhs];
    });
    float* destination = out + row * vocab;
    std::fill_n(destination, vocab, 0.0f);
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
      if (!std::isfinite(pr[ids[i]]) || pr[ids[i]] < 0.0f) {
        return Status::kInvalidArgument;
      }
      sum += pr[ids[i]];
    }
    if (!(sum > 0.0)) return Status::kInvalidArgument;
    for (int i = 0; i < k; ++i) destination[ids[i]] = pr[ids[i]] / sum;
  }
  return Status::kOk;
}

Status top_p_renorm(const float* probabilities, float* out, long long rows,
                    long long vocab, float p) {
  if (!detail::valid_product({rows, vocab}) || !std::isfinite(p) || p <= 0.0f ||
      p > 1.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(probabilities, out)) return Status::kInvalidArgument;
  for (long long row = 0; row < rows; ++row) {
    std::vector<int> ids(static_cast<std::size_t>(vocab));
    std::iota(ids.begin(), ids.end(), 0);
    const float* pr = probabilities + row * vocab;
    std::vector<float> alias_copy;
    if (probabilities == out) {
      alias_copy.assign(pr, pr + vocab);
      pr = alias_copy.data();
    }
    double total = 0.0;
    for (long long i = 0; i < vocab; ++i) {
      if (!std::isfinite(pr[i]) || pr[i] < 0.0f) return Status::kInvalidArgument;
      total += pr[i];
    }
    if (!(total > 0.0)) return Status::kInvalidArgument;
    std::stable_sort(ids.begin(), ids.end(), [&](int lhs, int rhs) {
      return pr[lhs] == pr[rhs] ? lhs < rhs : pr[lhs] > pr[rhs];
    });
    double sum = 0.0;
    std::size_t keep = 0;
    do {
      sum += pr[ids[keep++]];
    } while (keep < ids.size() && sum < p * total);
    float* destination = out + row * vocab;
    std::fill_n(destination, vocab, 0.0f);
    for (std::size_t i = 0; i < keep; ++i) {
      destination[ids[i]] = static_cast<float>(pr[ids[i]] / sum);
    }
  }
  return Status::kOk;
}

Status packbits(const std::uint8_t* x, std::uint8_t* out, long long count,
                bool bit_order_big) {
  if (!detail::valid_product({count})) return Status::kInvalidShape;
  if (!detail::all_nonnull(x, out)) return Status::kInvalidArgument;
  const long long bytes = (count + 7) / 8;
  std::fill_n(out, bytes, std::uint8_t{0});
  for (long long i = 0; i < count; ++i) {
    if (x[i] != 0) {
      const int bit = static_cast<int>(i % 8);
      out[i / 8] |= static_cast<std::uint8_t>(
          bit_order_big ? (0x80u >> bit) : (1u << bit));
    }
  }
  return Status::kOk;
}

Status segment_packbits(const std::uint8_t* x, const long long* input_offsets,
                        const long long* output_offsets, std::uint8_t* out,
                        long long segments, long long input_count,
                        long long output_bytes, bool bit_order_big) {
  if (!detail::valid_product({segments}) || input_count < 0 || output_bytes < 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, input_offsets, output_offsets, out)) {
    return Status::kInvalidArgument;
  }
  if (input_offsets[0] != 0 || output_offsets[0] != 0 ||
      input_offsets[segments] != input_count ||
      output_offsets[segments] != output_bytes) {
    return Status::kInvalidArgument;
  }
  std::fill_n(out, output_bytes, std::uint8_t{0});
  for (long long segment = 0; segment < segments; ++segment) {
    const long long begin = input_offsets[segment];
    const long long end = input_offsets[segment + 1];
    const long long obegin = output_offsets[segment];
    const long long oend = output_offsets[segment + 1];
    if (begin < 0 || end < begin || end > input_count || obegin < 0 ||
        oend < obegin || oend > output_bytes ||
        oend - obegin != (end - begin + 7) / 8) {
      return Status::kInvalidArgument;
    }
    if (begin == end) continue;
    const Status status = packbits(x + begin, out + obegin, end - begin,
                                   bit_order_big);
    if (status != Status::kOk) return status;
  }
  return Status::kOk;
}

Status permute_cols(const float* x, const int* permutation, float* out,
                    long long rows, long long cols) {
  if (!detail::valid_product({rows, cols})) return Status::kInvalidShape;
  if (!detail::all_nonnull(x, permutation, out)) return Status::kInvalidArgument;
  for (long long col = 0; col < cols; ++col) {
    if (permutation[col] < 0 || permutation[col] >= cols) {
      return Status::kInvalidArgument;
    }
  }
  for (long long row = 0; row < rows; ++row) {
    for (long long col = 0; col < cols; ++col) {
      out[row * cols + col] = x[row * cols + permutation[col]];
    }
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
