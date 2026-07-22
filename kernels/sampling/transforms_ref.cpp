#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "kernels/common/validation.h"

namespace quixicore_cpu {
namespace {

constexpr float kNegativeInfinity =
    -std::numeric_limits<float>::infinity();

bool valid_transform(const float* input, float* output, long long rows,
                     long long vocab, float temperature) {
  return detail::valid_product({rows, vocab}) && input != nullptr &&
         output != nullptr && std::isfinite(temperature) && temperature > 0.0f;
}

double transform_uniform(std::uint32_t seed, std::uint64_t row) {
  std::uint64_t z = (static_cast<std::uint64_t>(seed) << 32) ^ row ^
                    0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return (static_cast<double>(z >> 11) + 0.5) /
         9007199254740992.0;
}

bool contains(const int* values, long long count, int needle) {
  for (long long i = 0; i < count; ++i) {
    if (values[i] == needle) return true;
  }
  return false;
}

template <typename Function>
Status probability_mask(const float* logits, float* out, long long rows,
                        long long vocab, float parameter, float temperature,
                        Function threshold_function) {
  if (!valid_transform(logits, out, rows, vocab, temperature) ||
      !std::isfinite(parameter)) {
    return Status::kInvalidShape;
  }
  const float inv_temperature = 1.0f / temperature;
  for (long long row_index = 0; row_index < rows; ++row_index) {
    const float* source = logits + row_index * vocab;
    std::vector<float> tempered(static_cast<std::size_t>(vocab));
    float maximum = kNegativeInfinity;
    for (long long token = 0; token < vocab; ++token) {
      if (std::isnan(source[token]) || source[token] ==
                                           std::numeric_limits<float>::infinity()) {
        return Status::kInvalidArgument;
      }
      tempered[token] = source[token] * inv_temperature;
      maximum = std::max(maximum, tempered[token]);
    }
    if (!std::isfinite(maximum)) {
      std::copy(tempered.begin(), tempered.end(), out + row_index * vocab);
      continue;
    }
    double z = 0.0, weighted = 0.0;
    for (float value : tempered) {
      const double probability = std::exp(value - maximum);
      z += probability;
      weighted += probability * value;
    }
    const double threshold = threshold_function(z, weighted, maximum, parameter);
    for (long long token = 0; token < vocab; ++token) {
      const double probability = std::exp(tempered[token] - maximum) / z;
      out[row_index * vocab + token] =
          tempered[token] < maximum && probability < threshold
              ? kNegativeInfinity
              : tempered[token];
    }
  }
  return Status::kOk;
}

}  // namespace

Status quadratic_transform(const float* logits, float* out, long long rows,
                           long long vocab, float factor, float curve,
                           float temperature) {
  if (!valid_transform(logits, out, rows, vocab, temperature) ||
      !std::isfinite(factor) || !std::isfinite(curve)) {
    return Status::kInvalidShape;
  }
  const float inv_temperature = 1.0f / temperature;
  const float k = factor * (3.0f - curve) * 0.5f;
  const float s = factor * (curve - 1.0f) * 0.5f;
  for (long long row = 0; row < rows; ++row) {
    std::vector<float> tempered(static_cast<std::size_t>(vocab));
    float maximum = kNegativeInfinity;
    for (long long token = 0; token < vocab; ++token) {
      tempered[token] = logits[row * vocab + token] * inv_temperature;
      if (std::isnan(tempered[token])) return Status::kInvalidArgument;
      maximum = std::max(maximum, tempered[token]);
    }
    for (long long token = 0; token < vocab; ++token) {
      float difference = tempered[token] - maximum;
      difference -= difference * difference * (s * difference - k);
      out[row * vocab + token] =
          std::isfinite(difference) ? maximum + difference : tempered[token];
    }
  }
  return Status::kOk;
}

Status top_nsigma_mask(const float* logits, float* out, long long rows,
                       long long vocab, float nsigma, float temperature) {
  if (!valid_transform(logits, out, rows, vocab, temperature) || vocab < 2 ||
      !std::isfinite(nsigma) || nsigma < 0.0f) {
    return Status::kInvalidShape;
  }
  const float inv_temperature = 1.0f / temperature;
  for (long long row = 0; row < rows; ++row) {
    std::vector<float> values(static_cast<std::size_t>(vocab));
    double sum = 0.0, square_sum = 0.0;
    float maximum = kNegativeInfinity;
    for (long long token = 0; token < vocab; ++token) {
      const float value = logits[row * vocab + token] * inv_temperature;
      if (!std::isfinite(value)) return Status::kInvalidArgument;
      values[token] = value;
      maximum = std::max(maximum, value);
      sum += value;
      square_sum += static_cast<double>(value) * value;
    }
    const double variance = std::max(
        0.0, (square_sum - sum * sum / vocab) / static_cast<double>(vocab - 1));
    const double threshold = maximum - nsigma * std::sqrt(variance);
    for (long long token = 0; token < vocab; ++token) {
      out[row * vocab + token] =
          values[token] < threshold ? kNegativeInfinity : values[token];
    }
  }
  return Status::kOk;
}

Status top_a_mask(const float* logits, float* out, long long rows,
                  long long vocab, float top_a, float temperature) {
  return probability_mask(
      logits, out, rows, vocab, top_a, temperature,
      [](double z, double, double, float parameter) {
        return parameter > 0.0f ? parameter / (z * z) : 0.0;
      });
}

Status epsilon_cutoff_mask(const float* logits, float* out, long long rows,
                           long long vocab, float epsilon,
                           float temperature) {
  return probability_mask(logits, out, rows, vocab, epsilon, temperature,
                          [](double, double, double, float parameter) {
                            return parameter;
                          });
}

Status eta_cutoff_mask(const float* logits, float* out, long long rows,
                       long long vocab, float eta, float temperature) {
  if (eta < 0.0f) return Status::kInvalidShape;
  return probability_mask(
      logits, out, rows, vocab, eta, temperature,
      [](double z, double weighted, double maximum, float parameter) {
        const double sum_plogp = weighted / z - maximum - std::log(z);
        return std::min<double>(parameter,
                                std::sqrt(parameter) * std::exp(sum_plogp));
      });
}

Status xtc_mask(const float* logits, float* out, long long rows,
                long long vocab, float threshold, float probability,
                std::uint32_t seed, float temperature) {
  if (!valid_transform(logits, out, rows, vocab, temperature) ||
      !std::isfinite(threshold) || threshold < 0.0f || threshold > 1.0f ||
      !std::isfinite(probability) || probability < 0.0f || probability > 1.0f) {
    return Status::kInvalidShape;
  }
  const float inv_temperature = 1.0f / temperature;
  for (long long row = 0; row < rows; ++row) {
    std::vector<float> values(static_cast<std::size_t>(vocab));
    float maximum = kNegativeInfinity;
    for (long long token = 0; token < vocab; ++token) {
      values[token] = logits[row * vocab + token] * inv_temperature;
      if (std::isnan(values[token])) return Status::kInvalidArgument;
      maximum = std::max(maximum, values[token]);
    }
    if (transform_uniform(seed, row) >= probability || !std::isfinite(maximum)) {
      std::copy(values.begin(), values.end(), out + row * vocab);
      continue;
    }
    double z = 0.0;
    for (float value : values) z += std::exp(value - maximum);
    const double eligible = threshold * z;
    int count = 0;
    double least = std::numeric_limits<double>::infinity();
    for (float value : values) {
      const double e = std::exp(value - maximum);
      if (e >= eligible) {
        ++count;
        least = std::min(least, e);
      }
    }
    for (long long token = 0; token < vocab; ++token) {
      const double e = std::exp(values[token] - maximum);
      out[row * vocab + token] =
          count > 1 && e >= eligible && e > least ? kNegativeInfinity
                                                  : values[token];
    }
  }
  return Status::kOk;
}

Status skew_transform(const float* probabilities, float* out, long long rows,
                      long long vocab, float skew) {
  if (!detail::valid_product({rows, vocab}) ||
      !detail::all_nonnull(probabilities, out) || !std::isfinite(skew)) {
    return Status::kInvalidShape;
  }
  const double exponent = std::exp(skew);
  for (long long row = 0; row < rows; ++row) {
    std::vector<float> values(probabilities + row * vocab,
                              probabilities + (row + 1) * vocab);
    double cumulative = 0.0, previous = 0.0;
    for (long long token = 0; token < vocab; ++token) {
      if (!std::isfinite(values[token]) || values[token] < 0.0f) {
        return Status::kInvalidArgument;
      }
      cumulative += values[token];
      const double transformed = std::pow(std::max(cumulative, 0.0), exponent);
      out[row * vocab + token] = static_cast<float>(transformed - previous);
      previous = transformed;
    }
  }
  return Status::kOk;
}

Status no_repeat_ngram_mask(const float* logits, const int* previous,
                            const int* lengths, float* out, long long rows,
                            long long vocab, long long history,
                            int ngram_size, float temperature) {
  if (!valid_transform(logits, out, rows, vocab, temperature) || history <= 0 ||
      ngram_size < 2 || !detail::all_nonnull(previous, lengths)) {
    return Status::kInvalidShape;
  }
  const float inv_temperature = 1.0f / temperature;
  for (long long row = 0; row < rows; ++row) {
    std::vector<float> values(logits + row * vocab, logits + (row + 1) * vocab);
    for (long long token = 0; token < vocab; ++token) {
      out[row * vocab + token] = values[token] * inv_temperature;
    }
    if (lengths[row] < 0 || lengths[row] > history) return Status::kInvalidArgument;
    const int length = lengths[row];
    if (length < ngram_size) continue;
    const int* tokens = previous + row * history;
    for (int start = 0; start + ngram_size - 1 < length; ++start) {
      bool match = true;
      for (int item = 0; item < ngram_size - 1; ++item) {
        match &= tokens[start + item] ==
                 tokens[length - (ngram_size - 1) + item];
      }
      const int banned = tokens[start + ngram_size - 1];
      if (match && banned >= 0 && banned < vocab) {
        out[row * vocab + banned] = kNegativeInfinity;
      }
    }
  }
  return Status::kOk;
}

Status dry_penalty(const float* logits, const int* previous,
                   const int* lengths, const int* breakers, float* out,
                   long long rows, long long vocab, long long history,
                   long long breaker_count, float multiplier, float base,
                   int allowed_length, int range, int max_ngram,
                   int max_occurrences, int early_exit_match_length,
                   float temperature) {
  if (!valid_transform(logits, out, rows, vocab, temperature) || history <= 0 ||
      breaker_count < 0 || !detail::all_nonnull(previous, lengths) ||
      (breaker_count > 0 && breakers == nullptr) || !std::isfinite(multiplier) ||
      !std::isfinite(base) || base <= 0.0f || allowed_length < 0 || range < 0 ||
      max_ngram < 1 || max_occurrences < 1 || early_exit_match_length < 1) {
    return Status::kInvalidShape;
  }
  const float inv_temperature = 1.0f / temperature;
  for (long long row = 0; row < rows; ++row) {
    std::vector<float> source(logits + row * vocab, logits + (row + 1) * vocab);
    for (long long token = 0; token < vocab; ++token) {
      out[row * vocab + token] = source[token] * inv_temperature;
    }
    if (lengths[row] < 0 || lengths[row] > history) return Status::kInvalidArgument;
    const int length = lengths[row];
    if (multiplier == 0.0f || length < 2) continue;
    const int* tokens = previous + row * history;
    const int last = tokens[length - 1];
    if (contains(breakers, breaker_count, last)) continue;
    const int start_index = range > 0 ? std::max(0, length - range) : 0;
    int current_max_ngram = -1;
    const int cap = std::min(length - start_index, max_ngram + 1);
    for (int offset = 0; offset < cap; ++offset) {
      if (contains(breakers, breaker_count, tokens[length - offset - 1])) break;
      current_max_ngram = offset;
    }
    if (current_max_ngram <= allowed_length) continue;
    int seen = 0;
    for (int index = length - 2; index >= start_index; --index) {
      if (tokens[index] != last) continue;
      if (seen++ >= max_occurrences) break;
      const int maximum_unwind =
          std::min(index - start_index, current_max_ngram);
      int match_length = 0;
      for (int offset = 1; offset <= maximum_unwind; ++offset) {
        if (tokens[index - offset] != tokens[length - offset - 1] ||
            contains(breakers, breaker_count, tokens[index - offset])) {
          break;
        }
        match_length = offset;
      }
      if (match_length <= 0) continue;
      const int next = tokens[index + 1];
      if (next >= 0 && next < vocab) {
        const int new_length = match_length + 1;
        const float penalty = multiplier *
                              std::pow(base, new_length - allowed_length);
        out[row * vocab + next] =
            std::min(out[row * vocab + next],
                     source[next] * inv_temperature - penalty);
        if (new_length >= early_exit_match_length) break;
      }
    }
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
