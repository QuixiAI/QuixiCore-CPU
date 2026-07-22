#include "quixicore_cpu/qgemm.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "kernels/common/validation.h"
#include "quixicore_cpu/ops.h"
#include "src/memory/workspace_internal.h"

namespace quixicore_cpu {
namespace {

void add_bias(float* logits, const float* bias, long long rows,
              long long vocab) {
  if (bias == nullptr) return;
  for (long long row = 0; row < rows; ++row) {
    for (long long token = 0; token < vocab; ++token) {
      logits[row * vocab + token] += bias[token];
    }
  }
}

double logsumexp_selected(const float* logits, const std::vector<int>& ids) {
  double maximum = -std::numeric_limits<double>::infinity();
  for (int id : ids) maximum = std::max(maximum, static_cast<double>(logits[id]));
  double sum = 0.0;
  for (int id : ids) sum += std::exp(logits[id] - maximum);
  return maximum + std::log(sum);
}

void top_ids(const float* logits, std::vector<int>* ids, int count) {
  std::stable_sort(ids->begin(), ids->end(), [&](int lhs, int rhs) {
    return logits[lhs] == logits[rhs] ? lhs < rhs : logits[lhs] > logits[rhs];
  });
  if (static_cast<int>(ids->size()) > count) ids->resize(count);
}

Status dense_logits(const float* hidden_states, const float* weights,
                    const float* bias, float* logits, long long rows,
                    long long vocab, long long hidden) {
  return linear_epilogue(hidden_states, weights, bias, nullptr, logits, rows,
                         hidden, vocab, LinearActivation::kNone);
}

bool better_logit(float value, int token, float incumbent,
                  int incumbent_token) {
  return value > incumbent ||
         (value == incumbent && token < incumbent_token);
}

// Projects vocabulary tiles and retains only the final selection set. The
// packed row layout is contiguous for every GGUF format, so a tile is itself
// a valid smaller GEMV without repacking weights.
Status streaming_quantized_select(
    QuantFormat format, const void* packed_weights,
    const float* hidden_states, const float* bias, int* token_ids,
    long long rows, long long vocab, long long hidden, int retained,
    float temperature, std::uint32_t seed, bool sample_top_k) {
  if (retained <= 0 || retained > vocab || !std::isfinite(temperature) ||
      temperature < 0.0f) {
    return Status::kInvalidShape;
  }
  std::size_t packed_bytes = 0;
  const Status packed_status =
      qgemv_packed_size(format, vocab, hidden, &packed_bytes);
  if (packed_status != Status::kOk) return packed_status;
  const std::size_t row_bytes = packed_bytes / static_cast<std::size_t>(vocab);
  constexpr long long kVocabularyTile = 4096;
  detail::WorkspaceFrame workspace;
  float* tile_logits = workspace.allocate<float>(kVocabularyTile);
  float* retained_logits = workspace.allocate<float>(
      static_cast<std::size_t>(rows * retained));
  int* retained_ids = workspace.allocate<int>(
      static_cast<std::size_t>(rows * retained));
  int* sampled = sample_top_k
                     ? workspace.allocate<int>(static_cast<std::size_t>(rows))
                     : nullptr;
  if (tile_logits == nullptr || retained_logits == nullptr ||
      retained_ids == nullptr || (sample_top_k && sampled == nullptr)) {
    return Status::kOutOfMemory;
  }
  const auto* packed = static_cast<const std::uint8_t*>(packed_weights);
  for (long long row = 0; row < rows; ++row) {
    float* row_logits = retained_logits + row * retained;
    int* row_ids = retained_ids + row * retained;
    int count = 0;
    for (long long token_base = 0; token_base < vocab;
         token_base += kVocabularyTile) {
      const long long tile = std::min(kVocabularyTile, vocab - token_base);
      const Status status = qgemv(
          format, packed + static_cast<std::size_t>(token_base) * row_bytes,
          hidden_states + row * hidden, tile_logits, tile, hidden);
      if (status != Status::kOk) return status;
      for (long long local = 0; local < tile; ++local) {
        const int token = static_cast<int>(token_base + local);
        const float value = tile_logits[local] +
                            (bias != nullptr ? bias[token] : 0.0f);
        if (std::isnan(value) || value == std::numeric_limits<float>::infinity()) {
          return Status::kInvalidArgument;
        }
        int position = count;
        while (position > 0 &&
               better_logit(value, token, row_logits[position - 1],
                            row_ids[position - 1])) {
          --position;
        }
        if (position >= retained) continue;
        const int upper = std::min(count, retained - 1);
        for (int item = upper; item > position; --item) {
          row_logits[item] = row_logits[item - 1];
          row_ids[item] = row_ids[item - 1];
        }
        row_logits[position] = value;
        row_ids[position] = token;
        if (count < retained) ++count;
      }
    }
    if (count != retained) return Status::kInvalidArgument;
    if (!sample_top_k) token_ids[row] = row_ids[0];
  }
  if (!sample_top_k) return Status::kOk;
  const Status sample_status = top_k_sample(
      retained_logits, sampled, rows, retained, retained, temperature, seed);
  if (sample_status != Status::kOk) return sample_status;
  for (long long row = 0; row < rows; ++row) {
    token_ids[row] = retained_ids[row * retained + sampled[row]];
  }
  return Status::kOk;
}

Status streaming_quantized_masked_topk(
    QuantFormat format, const void* packed_weights,
    const float* hidden_states, const float* bias,
    const std::uint8_t* allow_mask, int* token_ids,
    float* log_probabilities, long long rows, long long vocab,
    long long hidden, int top_k, bool normalize_allowed) {
  std::size_t packed_bytes = 0;
  const Status packed_status =
      qgemv_packed_size(format, vocab, hidden, &packed_bytes);
  if (packed_status != Status::kOk) return packed_status;
  const std::size_t row_bytes = packed_bytes / static_cast<std::size_t>(vocab);
  constexpr long long kVocabularyTile = 4096;
  detail::WorkspaceFrame workspace;
  float* tile_logits = workspace.allocate<float>(kVocabularyTile);
  float* best_logits =
      workspace.allocate<float>(static_cast<std::size_t>(top_k));
  int* best_ids = workspace.allocate<int>(static_cast<std::size_t>(top_k));
  if (tile_logits == nullptr || best_logits == nullptr || best_ids == nullptr) {
    return Status::kOutOfMemory;
  }
  const auto* packed = static_cast<const std::uint8_t*>(packed_weights);
  const long long mask_stride = (vocab + 7) / 8;
  for (long long row = 0; row < rows; ++row) {
    int count = 0;
    double maximum = -std::numeric_limits<double>::infinity();
    double denominator = 0.0;
    for (long long token_base = 0; token_base < vocab;
         token_base += kVocabularyTile) {
      const long long tile = std::min(kVocabularyTile, vocab - token_base);
      const Status status = qgemv(
          format, packed + static_cast<std::size_t>(token_base) * row_bytes,
          hidden_states + row * hidden, tile_logits, tile, hidden);
      if (status != Status::kOk) return status;
      for (long long local = 0; local < tile; ++local) {
        const int token = static_cast<int>(token_base + local);
        const float value = tile_logits[local] +
                            (bias != nullptr ? bias[token] : 0.0f);
        const bool allowed =
            (allow_mask[row * mask_stride + token / 8] &
             static_cast<std::uint8_t>(0x80u >> (token % 8))) != 0;
        if (!normalize_allowed || allowed) {
          if (static_cast<double>(value) > maximum) {
            denominator = denominator * std::exp(maximum - value) + 1.0;
            maximum = value;
          } else {
            denominator += std::exp(static_cast<double>(value) - maximum);
          }
        }
        if (!allowed) continue;
        int position = count;
        while (position > 0 &&
               better_logit(value, token, best_logits[position - 1],
                            best_ids[position - 1])) {
          --position;
        }
        if (position >= top_k) continue;
        const int upper = std::min(count, top_k - 1);
        for (int item = upper; item > position; --item) {
          best_logits[item] = best_logits[item - 1];
          best_ids[item] = best_ids[item - 1];
        }
        best_logits[position] = value;
        best_ids[position] = token;
        if (count < top_k) ++count;
      }
    }
    if (count < top_k || !(denominator > 0.0)) {
      return Status::kInvalidArgument;
    }
    const double lse = maximum + std::log(denominator);
    for (int item = 0; item < top_k; ++item) {
      token_ids[row * top_k + item] = best_ids[item];
      log_probabilities[row * top_k + item] =
          static_cast<float>(best_logits[item] - lse);
    }
  }
  return Status::kOk;
}

}  // namespace

Status quantized_lm_head_sample(
    QuantFormat format, const void* packed_weights,
    const float* hidden_states, const float* bias, int* token_ids,
    long long rows, long long vocab, long long hidden, LmHeadSampling mode,
    int k, float top_p, float temperature, std::uint32_t seed) {
  if (!detail::valid_product({rows, vocab, hidden})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(packed_weights, hidden_states, token_ids)) {
    return Status::kInvalidArgument;
  }
  if (mode == LmHeadSampling::kArgmax) {
    return streaming_quantized_select(
        format, packed_weights, hidden_states, bias, token_ids, rows, vocab,
        hidden, 1, 0.0f, seed, false);
  }
  if (mode == LmHeadSampling::kTopK) {
    return streaming_quantized_select(
        format, packed_weights, hidden_states, bias, token_ids, rows, vocab,
        hidden, k, temperature, seed, true);
  }
  std::vector<float> logits(static_cast<std::size_t>(rows * vocab));
  Status status = qgemm(format, packed_weights, hidden_states, logits.data(),
                        rows, vocab, hidden);
  if (status != Status::kOk) return status;
  add_bias(logits.data(), bias, rows, vocab);
  switch (mode) {
    case LmHeadSampling::kArgmax:
      break;
    case LmHeadSampling::kCategorical:
      return sample_categorical(logits.data(), token_ids, rows, vocab,
                                temperature, seed);
    case LmHeadSampling::kTopK:
      break;
    case LmHeadSampling::kTopP:
      return top_p_sample(logits.data(), token_ids, rows, vocab, top_p,
                          temperature, seed);
  }
  return Status::kInvalidArgument;
}

Status lm_head_sample(const float* hidden_states, const float* weights,
                      const float* bias, int* token_ids, long long rows,
                      long long vocab, long long hidden, LmHeadSampling mode,
                      int k, float top_p, float temperature,
                      std::uint32_t seed) {
  if (!detail::valid_product({rows, vocab, hidden})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(hidden_states, weights, token_ids)) {
    return Status::kInvalidArgument;
  }
  std::vector<float> logits(static_cast<std::size_t>(rows * vocab));
  Status status = dense_logits(hidden_states, weights, bias, logits.data(),
                               rows, vocab, hidden);
  if (status != Status::kOk) return status;
  switch (mode) {
    case LmHeadSampling::kArgmax:
      return argmax_sample(logits.data(), token_ids, rows, vocab);
    case LmHeadSampling::kCategorical:
      return sample_categorical(logits.data(), token_ids, rows, vocab,
                                temperature, seed);
    case LmHeadSampling::kTopK:
      return top_k_sample(logits.data(), token_ids, rows, vocab, k,
                          temperature, seed);
    case LmHeadSampling::kTopP:
      return top_p_sample(logits.data(), token_ids, rows, vocab, top_p,
                          temperature, seed);
  }
  return Status::kInvalidArgument;
}

Status quantized_lm_head_masked_topk(
    QuantFormat format, const void* packed_weights,
    const float* hidden_states, const float* bias,
    const std::uint8_t* allow_mask, int* token_ids, float* log_probabilities,
    long long rows, long long vocab, long long hidden, int top_k,
    bool normalize_allowed) {
  if (!detail::valid_product({rows, vocab, hidden}) || top_k <= 0 ||
      top_k > vocab) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(packed_weights, hidden_states, allow_mask,
                           token_ids, log_probabilities)) {
    return Status::kInvalidArgument;
  }
  return streaming_quantized_masked_topk(
      format, packed_weights, hidden_states, bias, allow_mask, token_ids,
      log_probabilities, rows, vocab, hidden, top_k, normalize_allowed);
}

Status lm_head_masked_topk(
    const float* hidden_states, const float* weights, const float* bias,
    const std::uint8_t* allow_mask, int* token_ids, float* log_probabilities,
    long long rows, long long vocab, long long hidden, int top_k,
    bool normalize_allowed) {
  if (!detail::valid_product({rows, vocab, hidden}) || top_k <= 0 ||
      top_k > vocab) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(hidden_states, weights, allow_mask, token_ids,
                           log_probabilities)) {
    return Status::kInvalidArgument;
  }
  std::vector<float> logits(static_cast<std::size_t>(rows * vocab));
  Status status = dense_logits(hidden_states, weights, bias, logits.data(),
                               rows, vocab, hidden);
  if (status != Status::kOk) return status;
  const long long mask_stride = (vocab + 7) / 8;
  for (long long row = 0; row < rows; ++row) {
    std::vector<int> allowed;
    std::vector<int> all(static_cast<std::size_t>(vocab));
    std::iota(all.begin(), all.end(), 0);
    for (int token : all) {
      if (allow_mask[row * mask_stride + token / 8] &
          static_cast<std::uint8_t>(0x80u >> (token % 8))) {
        allowed.push_back(token);
      }
    }
    if (static_cast<int>(allowed.size()) < top_k) {
      return Status::kInvalidArgument;
    }
    const float* row_logits = logits.data() + row * vocab;
    const double lse = logsumexp_selected(row_logits,
                                          normalize_allowed ? allowed : all);
    top_ids(row_logits, &allowed, top_k);
    for (int item = 0; item < top_k; ++item) {
      token_ids[row * top_k + item] = allowed[item];
      log_probabilities[row * top_k + item] =
          static_cast<float>(row_logits[allowed[item]] - lse);
    }
  }
  return Status::kOk;
}

Status quantized_lm_head_candidates(
    QuantFormat format, const void* packed_weights,
    const float* hidden_states, const float* bias, const int* candidate_ids,
    const long long* offsets, int* token_ids, float* log_probabilities,
    long long rows, long long vocab, long long hidden, long long candidates,
    int top_k) {
  if (!detail::valid_product({rows, vocab, hidden}) || candidates < 0 ||
      top_k <= 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(packed_weights, hidden_states, candidate_ids,
                           offsets, token_ids, log_probabilities)) {
    return Status::kInvalidArgument;
  }
  if (offsets[0] != 0 || offsets[rows] != candidates) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < rows; ++row) {
    if (offsets[row] > offsets[row + 1] ||
        offsets[row + 1] - offsets[row] < top_k) {
      return Status::kInvalidArgument;
    }
    std::vector<int> ids;
    for (long long item = offsets[row]; item < offsets[row + 1]; ++item) {
      if (candidate_ids[item] < 0 || candidate_ids[item] >= vocab ||
          std::find(ids.begin(), ids.end(), candidate_ids[item]) != ids.end()) {
        return Status::kInvalidArgument;
      }
      ids.push_back(candidate_ids[item]);
    }
    std::vector<float> logits(ids.size());
    Status status = qgemv_rows(
        format, packed_weights, hidden_states + row * hidden, ids.data(),
        logits.data(), static_cast<long long>(ids.size()), vocab, hidden);
    if (status != Status::kOk) return status;
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t item = 0; item < ids.size(); ++item) {
      if (bias != nullptr) logits[item] += bias[ids[item]];
      maximum = std::max(maximum, static_cast<double>(logits[item]));
    }
    double denominator = 0.0;
    for (float value : logits) denominator += std::exp(value - maximum);
    const double lse = maximum + std::log(denominator);
    std::vector<int> order(ids.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
      return logits[lhs] == logits[rhs] ? ids[lhs] < ids[rhs]
                                        : logits[lhs] > logits[rhs];
    });
    for (int item = 0; item < top_k; ++item) {
      const int selected = order[item];
      token_ids[row * top_k + item] = ids[selected];
      log_probabilities[row * top_k + item] =
          static_cast<float>(logits[selected] - lse);
    }
  }
  return Status::kOk;
}

Status lm_head_candidates(
    const float* hidden_states, const float* weights, const float* bias,
    const int* candidate_ids, const long long* offsets, int* token_ids,
    float* log_probabilities, long long rows, long long vocab,
    long long hidden, long long candidates, int top_k) {
  if (!detail::valid_product({rows, vocab, hidden}) || candidates < 0 ||
      top_k <= 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(hidden_states, weights, candidate_ids, offsets,
                           token_ids, log_probabilities)) {
    return Status::kInvalidArgument;
  }
  if (offsets[0] != 0 || offsets[rows] != candidates) {
    return Status::kInvalidArgument;
  }
  std::vector<float> logits(static_cast<std::size_t>(rows * vocab));
  Status status = dense_logits(hidden_states, weights, bias, logits.data(),
                               rows, vocab, hidden);
  if (status != Status::kOk) return status;
  for (long long row = 0; row < rows; ++row) {
    if (offsets[row] > offsets[row + 1] ||
        offsets[row + 1] - offsets[row] < top_k) {
      return Status::kInvalidArgument;
    }
    std::vector<int> ids;
    for (long long item = offsets[row]; item < offsets[row + 1]; ++item) {
      if (candidate_ids[item] < 0 || candidate_ids[item] >= vocab ||
          std::find(ids.begin(), ids.end(), candidate_ids[item]) != ids.end()) {
        return Status::kInvalidArgument;
      }
      ids.push_back(candidate_ids[item]);
    }
    const float* row_logits = logits.data() + row * vocab;
    const double lse = logsumexp_selected(row_logits, ids);
    top_ids(row_logits, &ids, top_k);
    for (int item = 0; item < top_k; ++item) {
      token_ids[row * top_k + item] = ids[item];
      log_probabilities[row * top_k + item] =
          static_cast<float>(row_logits[ids[item]] - lse);
    }
  }
  return Status::kOk;
}

Status quantized_lm_head_beam_advance(
    QuantFormat format, const void* packed_weights,
    const float* hidden_states, const float* bias,
    const float* cumulative_log_probabilities, int* next_token,
    int* parent_beam, float* next_cumulative, long long batch,
    long long beam_width, long long vocab, long long hidden) {
  if (!detail::valid_product({batch, beam_width, vocab, hidden})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(packed_weights, hidden_states,
                           cumulative_log_probabilities, next_token,
                           parent_beam, next_cumulative)) {
    return Status::kInvalidArgument;
  }
  const long long rows = batch * beam_width;
  std::vector<float> logits(static_cast<std::size_t>(rows * vocab));
  Status status = qgemm(format, packed_weights, hidden_states, logits.data(),
                        rows, vocab, hidden);
  if (status != Status::kOk) return status;
  add_bias(logits.data(), bias, rows, vocab);
  return beam_search_step(logits.data(), cumulative_log_probabilities,
                          next_token, parent_beam, next_cumulative, batch,
                          beam_width, vocab);
}

Status lm_head_constrained(const float* hidden_states, const float* weights,
                           const float* bias, const std::uint8_t* forbidden,
                           const int* previous_token, int* selected_token,
                           float* selected_log_probability, long long rows,
                           long long vocab, long long hidden, int eos_id,
                           bool forbid_eos) {
  if (!detail::valid_product({rows, vocab, hidden}) ||
      (eos_id < -1 || eos_id >= vocab)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(hidden_states, weights, forbidden, previous_token,
                           selected_token, selected_log_probability)) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < rows; ++row) {
    if (previous_token[row] < 0 || previous_token[row] >= vocab) {
      return Status::kInvalidArgument;
    }
    std::vector<float> logits(static_cast<std::size_t>(vocab));
    std::vector<int> allowed;
    for (long long token = 0; token < vocab; ++token) {
      double sum = bias != nullptr ? bias[token] : 0.0;
      for (long long input = 0; input < hidden; ++input) {
        sum += hidden_states[row * hidden + input] *
               weights[token * hidden + input];
      }
      logits[token] = static_cast<float>(sum);
      if (forbidden[static_cast<long long>(previous_token[row]) * vocab + token] == 0 &&
          !(forbid_eos && token == eos_id)) {
        allowed.push_back(static_cast<int>(token));
      }
    }
    if (allowed.empty()) return Status::kInvalidArgument;
    const double lse = logsumexp_selected(logits.data(), allowed);
    top_ids(logits.data(), &allowed, 1);
    selected_token[row] = allowed[0];
    selected_log_probability[row] =
        static_cast<float>(logits[allowed[0]] - lse);
  }
  return Status::kOk;
}

}  // namespace quixicore_cpu
