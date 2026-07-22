#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "kernels/common/validation.h"

namespace quixicore_cpu {
namespace {

double uniform01(std::uint32_t seed, std::uint64_t row) {
  std::uint64_t z = (static_cast<std::uint64_t>(seed) << 32) ^ row ^
                    0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return (static_cast<double>(z >> 11) + 0.5) *
         (1.0 / 9007199254740992.0);
}

bool valid_logit(float value) {
  return !std::isnan(value) && value != std::numeric_limits<float>::infinity();
}

int row_argmax(const float* row, long long vocab, bool* valid) {
  int best = 0;
  float best_value = row[0];
  *valid = valid_logit(best_value);
  for (long long token = 1; token < vocab; ++token) {
    *valid = *valid && valid_logit(row[token]);
    if (row[token] > best_value) {
      best_value = row[token];
      best = static_cast<int>(token);
    }
  }
  return best;
}

int sample_weighted(const std::vector<int>& ids,
                    const std::vector<double>& weights, double uniform) {
  double total = std::accumulate(weights.begin(), weights.end(), 0.0);
  if (!(total > 0.0)) {
    return ids.front();
  }
  const double target = uniform * total;
  double cumulative = 0.0;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    cumulative += weights[i];
    if (target < cumulative || i + 1 == ids.size()) {
      return ids[i];
    }
  }
  return ids.back();
}

Status validate_sampling(const float* logits, int* out, long long rows,
                         long long vocab, float temperature) {
  if (!detail::valid_product({rows, vocab}) ||
      !std::isfinite(temperature) || temperature < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(logits, out)) {
    return Status::kInvalidArgument;
  }
  return Status::kOk;
}

std::vector<int> sorted_tokens(const float* row, long long vocab,
                               bool* valid) {
  std::vector<int> ids(static_cast<std::size_t>(vocab));
  std::iota(ids.begin(), ids.end(), 0);
  *valid = true;
  for (long long token = 0; token < vocab; ++token) {
    *valid = *valid && valid_logit(row[token]);
  }
  std::stable_sort(ids.begin(), ids.end(), [&](int lhs, int rhs) {
    if (row[lhs] == row[rhs]) {
      return lhs < rhs;
    }
    return row[lhs] > row[rhs];
  });
  return ids;
}

}  // namespace

Status argmax_sample(const float* logits, int* out, long long rows,
                     long long vocab) {
  const Status status = validate_sampling(logits, out, rows, vocab, 0.0f);
  if (status != Status::kOk) {
    return status;
  }
  for (long long row = 0; row < rows; ++row) {
    bool valid = true;
    out[row] = row_argmax(logits + row * vocab, vocab, &valid);
    if (!valid) {
      return Status::kInvalidArgument;
    }
  }
  return Status::kOk;
}

Status sample_categorical(const float* logits, int* out, long long rows,
                          long long vocab, float temperature,
                          std::uint32_t seed) {
  const Status status = validate_sampling(logits, out, rows, vocab, temperature);
  if (status != Status::kOk) {
    return status;
  }
  if (temperature == 0.0f) {
    return argmax_sample(logits, out, rows, vocab);
  }
  std::vector<int> ids(static_cast<std::size_t>(vocab));
  std::iota(ids.begin(), ids.end(), 0);
  std::vector<double> weights(static_cast<std::size_t>(vocab));
  for (long long row_index = 0; row_index < rows; ++row_index) {
    const float* row = logits + row_index * vocab;
    bool valid = true;
    const int best = row_argmax(row, vocab, &valid);
    if (!valid) {
      return Status::kInvalidArgument;
    }
    const float maximum = row[best];
    if (!std::isfinite(maximum)) {
      out[row_index] = best;
      continue;
    }
    for (long long token = 0; token < vocab; ++token) {
      weights[static_cast<std::size_t>(token)] =
          std::exp((static_cast<double>(row[token]) - maximum) / temperature);
    }
    out[row_index] = sample_weighted(
        ids, weights, uniform01(seed, static_cast<std::uint64_t>(row_index)));
  }
  return Status::kOk;
}

Status top_k_sample(const float* logits, int* out, long long rows,
                    long long vocab, int k, float temperature,
                    std::uint32_t seed) {
  const Status status = validate_sampling(logits, out, rows, vocab, temperature);
  if (status != Status::kOk || k <= 0 || k > vocab) {
    return status != Status::kOk ? status : Status::kInvalidShape;
  }
  if (temperature == 0.0f) {
    return argmax_sample(logits, out, rows, vocab);
  }
  for (long long row_index = 0; row_index < rows; ++row_index) {
    const float* row = logits + row_index * vocab;
    bool valid = true;
    std::vector<int> ids = sorted_tokens(row, vocab, &valid);
    if (!valid) {
      return Status::kInvalidArgument;
    }
    ids.resize(static_cast<std::size_t>(k));
    const float maximum = row[ids.front()];
    std::vector<double> weights(ids.size());
    if (std::isfinite(maximum)) {
      for (std::size_t i = 0; i < ids.size(); ++i) {
        weights[i] = std::exp(
            (static_cast<double>(row[ids[i]]) - maximum) / temperature);
      }
    } else {
      weights.front() = 1.0;
    }
    out[row_index] = sample_weighted(
        ids, weights, uniform01(seed, static_cast<std::uint64_t>(row_index)));
  }
  return Status::kOk;
}

Status top_p_sample(const float* logits, int* out, long long rows,
                    long long vocab, float p, float temperature,
                    std::uint32_t seed) {
  const Status status = validate_sampling(logits, out, rows, vocab, temperature);
  if (status != Status::kOk || !std::isfinite(p) || p <= 0.0f || p > 1.0f) {
    return status != Status::kOk ? status : Status::kInvalidShape;
  }
  if (temperature == 0.0f) {
    return argmax_sample(logits, out, rows, vocab);
  }
  for (long long row_index = 0; row_index < rows; ++row_index) {
    const float* row = logits + row_index * vocab;
    bool valid = true;
    std::vector<int> ids = sorted_tokens(row, vocab, &valid);
    if (!valid) {
      return Status::kInvalidArgument;
    }
    const float maximum = row[ids.front()];
    std::vector<double> weights(ids.size(), 0.0);
    if (!std::isfinite(maximum)) {
      out[row_index] = ids.front();
      continue;
    }
    double total = 0.0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      weights[i] = std::exp(
          (static_cast<double>(row[ids[i]]) - maximum) / temperature);
      total += weights[i];
    }
    double cumulative = 0.0;
    std::size_t keep = 0;
    do {
      cumulative += weights[keep++];
    } while (keep < weights.size() && cumulative < p * total);
    ids.resize(keep);
    weights.resize(keep);
    out[row_index] = sample_weighted(
        ids, weights, uniform01(seed, static_cast<std::uint64_t>(row_index)));
  }
  return Status::kOk;
}

Status min_p_sample(const float* logits, int* out, long long rows,
                    long long vocab, float min_p, float temperature,
                    std::uint32_t seed) {
  const Status status = validate_sampling(logits, out, rows, vocab, temperature);
  if (status != Status::kOk || !std::isfinite(min_p) || min_p <= 0.0f ||
      min_p > 1.0f) {
    return status != Status::kOk ? status : Status::kInvalidShape;
  }
  if (temperature == 0.0f) {
    return argmax_sample(logits, out, rows, vocab);
  }
  for (long long row_index = 0; row_index < rows; ++row_index) {
    const float* row = logits + row_index * vocab;
    bool valid = true;
    std::vector<int> sorted = sorted_tokens(row, vocab, &valid);
    if (!valid) {
      return Status::kInvalidArgument;
    }
    const float maximum = row[sorted.front()];
    if (!std::isfinite(maximum)) {
      out[row_index] = sorted.front();
      continue;
    }
    std::vector<int> ids;
    std::vector<double> weights;
    for (const int token : sorted) {
      const double weight = std::exp(
          (static_cast<double>(row[token]) - maximum) / temperature);
      if (weight + std::numeric_limits<double>::epsilon() < min_p) {
        break;
      }
      ids.push_back(token);
      weights.push_back(weight);
    }
    out[row_index] = sample_weighted(
        ids, weights, uniform01(seed, static_cast<std::uint64_t>(row_index)));
  }
  return Status::kOk;
}

Status beam_search_step(const float* logits, const float* cumulative,
                        int* next_token, int* parent_beam,
                        float* next_cumulative, long long batch,
                        long long beam, long long vocab) {
  if (!detail::valid_product({batch, beam, vocab}) || beam > INT_MAX ||
      vocab > INT_MAX) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(logits, cumulative, next_token, parent_beam,
                           next_cumulative)) {
    return Status::kInvalidArgument;
  }
  struct Candidate {
    float score;
    int parent;
    int token;
  };
  std::vector<Candidate> candidates(static_cast<std::size_t>(beam * vocab));
  for (long long request = 0; request < batch; ++request) {
    std::size_t candidate_index = 0;
    for (long long parent = 0; parent < beam; ++parent) {
      const float* row = logits + (request * beam + parent) * vocab;
      bool valid = true;
      const int best = row_argmax(row, vocab, &valid);
      if (!valid || std::isnan(cumulative[request * beam + parent]) ||
          cumulative[request * beam + parent] ==
              std::numeric_limits<float>::infinity()) {
        return Status::kInvalidArgument;
      }
      const float maximum = row[best];
      if (!std::isfinite(maximum)) {
        return Status::kInvalidArgument;
      }
      double sum = 0.0;
      for (long long token = 0; token < vocab; ++token) {
        sum += std::exp(static_cast<double>(row[token] - maximum));
      }
      const double logsum = static_cast<double>(maximum) + std::log(sum);
      for (long long token = 0; token < vocab; ++token) {
        candidates[candidate_index++] = {
            static_cast<float>(cumulative[request * beam + parent] +
                               row[token] - logsum),
            static_cast<int>(parent), static_cast<int>(token)};
      }
    }
    std::partial_sort(candidates.begin(), candidates.begin() + beam,
                      candidates.end(), [](const Candidate& lhs,
                                           const Candidate& rhs) {
      if (lhs.score != rhs.score) {
        return lhs.score > rhs.score;
      }
      if (lhs.parent != rhs.parent) {
        return lhs.parent < rhs.parent;
      }
      return lhs.token < rhs.token;
    });
    for (long long slot = 0; slot < beam; ++slot) {
      const Candidate& candidate = candidates[static_cast<std::size_t>(slot)];
      const long long index = request * beam + slot;
      next_token[index] = candidate.token;
      parent_beam[index] = candidate.parent;
      next_cumulative[index] = candidate.score;
    }
  }
  return Status::kOk;
}

Status speculative_verify(const int* draft_tokens, const float* draft_probs,
                          const float* target_probs, const int* bonus_tokens,
                          const float* accept_uniforms, int* out_tokens,
                          int* accepted_count, long long batch,
                          long long draft_length, long long vocab,
                          std::uint32_t seed) {
  if (!detail::valid_product({batch, draft_length, vocab}) ||
      !detail::valid_product({batch, draft_length + 1, vocab}) ||
      vocab > INT_MAX) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(draft_tokens, draft_probs, target_probs,
                           bonus_tokens, accept_uniforms, out_tokens,
                           accepted_count)) {
    return Status::kInvalidArgument;
  }
  std::vector<int> ids(static_cast<std::size_t>(vocab));
  std::iota(ids.begin(), ids.end(), 0);
  std::vector<double> weights(static_cast<std::size_t>(vocab));
  for (long long request = 0; request < batch; ++request) {
    int accepted = 0;
    std::fill(out_tokens + request * (draft_length + 1),
              out_tokens + (request + 1) * (draft_length + 1), -1);
    bool rejected = false;
    for (long long step = 0; step < draft_length; ++step) {
      const int token = draft_tokens[request * draft_length + step];
      const float uniform = accept_uniforms[request * draft_length + step];
      if (token < 0 || token >= vocab || !std::isfinite(uniform) ||
          uniform < 0.0f || uniform > 1.0f) {
        return Status::kInvalidArgument;
      }
      const long long probability_offset =
          (request * draft_length + step) * vocab;
      const long long target_offset =
          (request * (draft_length + 1) + step) * vocab;
      const float draft_probability = draft_probs[probability_offset + token];
      const float target_probability = target_probs[target_offset + token];
      if (!std::isfinite(draft_probability) || draft_probability < 0.0f ||
          !std::isfinite(target_probability) || target_probability < 0.0f) {
        return Status::kInvalidArgument;
      }
      const double ratio =
          draft_probability > 0.0f
              ? std::min(1.0, static_cast<double>(target_probability) /
                                  draft_probability)
              : (target_probability > 0.0f ? 1.0 : 0.0);
      if (uniform <= ratio) {
        out_tokens[request * (draft_length + 1) + step] = token;
        ++accepted;
        continue;
      }
      double residual_sum = 0.0;
      double target_sum = 0.0;
      for (long long candidate = 0; candidate < vocab; ++candidate) {
        const float draft_p = draft_probs[probability_offset + candidate];
        const float target_p = target_probs[target_offset + candidate];
        if (!std::isfinite(draft_p) || draft_p < 0.0f ||
            !std::isfinite(target_p) || target_p < 0.0f) {
          return Status::kInvalidArgument;
        }
        weights[static_cast<std::size_t>(candidate)] =
            std::max(0.0, static_cast<double>(target_p - draft_p));
        residual_sum += weights[static_cast<std::size_t>(candidate)];
        target_sum += target_p;
      }
      if (!(residual_sum > 0.0)) {
        for (long long candidate = 0; candidate < vocab; ++candidate) {
          weights[static_cast<std::size_t>(candidate)] =
              target_probs[target_offset + candidate];
        }
        if (!(target_sum > 0.0)) {
          return Status::kInvalidArgument;
        }
      }
      const int recovered = sample_weighted(
          ids, weights,
          uniform01(seed, static_cast<std::uint64_t>(request * draft_length +
                                                     step)));
      out_tokens[request * (draft_length + 1) + step] = recovered;
      rejected = true;
      break;
    }
    if (!rejected) {
      if (bonus_tokens[request] < 0 || bonus_tokens[request] >= vocab) {
        return Status::kInvalidArgument;
      }
      out_tokens[request * (draft_length + 1) + draft_length] =
          bonus_tokens[request];
    }
    accepted_count[request] = accepted;
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
