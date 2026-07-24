#include <algorithm>
#include <cmath>
#include <vector>

#include "kernels/common/float_storage_access.h"
#include "kernels/common/validation.h"
#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/ops.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

template <FloatStorageType Type>
Status calibration_absmax_typed(const void* x, const float* running,
                                float* out, long long tokens,
                                long long channels) {
  if (x == nullptr || out == nullptr) return Status::kInvalidArgument;
  threading::parallel_ranges(
      channels, 64, [&](long long begin, long long end, int) {
        std::vector<float> maxima(static_cast<std::size_t>(end - begin));
        for (long long channel = begin; channel < end; ++channel) {
          maxima[channel - begin] =
              running == nullptr ? 0.0f : std::fabs(running[channel]);
        }
        for (long long token = 0; token < tokens; ++token) {
          const long long row = token * channels;
          for (long long channel = begin; channel < end; ++channel) {
            float& maximum = maxima[channel - begin];
            if (std::isnan(maximum)) continue;
            const float value = std::fabs(
                detail::load_storage<Type>(x, row + channel));
            if (std::isnan(value)) {
              maximum = value;
            } else {
              maximum = std::max(maximum, value);
            }
          }
        }
        for (long long channel = begin; channel < end; ++channel) {
          out[channel] = maxima[channel - begin];
        }
      });
  return Status::kOk;
}

template <FloatStorageType Type>
Status logits_softcap_typed(const void* logits, void* out, long long count,
                            float cap) {
  if (logits == nullptr || out == nullptr || !std::isfinite(cap) ||
      cap <= 0.0f) {
    return Status::kInvalidArgument;
  }
  const float inverse_cap = 1.0f / cap;
  threading::parallel_ranges(count, 4096,
                             [&](long long begin, long long end, int) {
    for (long long index = begin; index < end; ++index) {
      detail::store_storage<Type>(
          out, index,
          cap * std::tanh(detail::load_storage<Type>(logits, index) *
                          inverse_cap));
    }
  });
  return Status::kOk;
}

template <FloatStorageType Type>
Status embedding_lookup_types_typed(
    const int* token_ids, const int* type_ids, const void* token_table,
    const void* type_table, void* out, long long token_vocab,
    long long type_vocab, long long count, long long dim, float token_scale) {
  if (!detail::all_nonnull(token_ids, type_ids, token_table, type_table, out) ||
      !std::isfinite(token_scale)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(count, 1,
                             [&](long long begin, long long end, int) {
    for (long long token = begin; token < end; ++token) {
      const int token_id = token_ids[token];
      const int type_id = type_ids[token];
      const bool token_valid = token_id >= 0 && token_id < token_vocab;
      const bool type_valid = type_id >= 0 && type_id < type_vocab;
      for (long long feature = 0; feature < dim; ++feature) {
        const float token_value =
            token_valid
                ? detail::load_storage<Type>(
                      token_table,
                      static_cast<long long>(token_id) * dim + feature)
                : 0.0f;
        const float type_value =
            type_valid
                ? detail::load_storage<Type>(
                      type_table,
                      static_cast<long long>(type_id) * dim + feature)
                : 0.0f;
        detail::store_storage<Type>(out, token * dim + feature,
                                    token_scale * token_value + type_value);
      }
    }
  });
  return Status::kOk;
}

template <FloatStorageType Type>
Status masked_mean_pool_rms_l2_typed(
    const void* x, const int* mask, const void* weight, void* out,
    long long batch, long long sequence, long long hidden, float eps) {
  if (!detail::all_nonnull(x, mask, weight, out) || !std::isfinite(eps) ||
      eps < 0.0f) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(batch, 1,
                             [&](long long begin, long long end, int) {
    std::vector<float> pooled(static_cast<std::size_t>(hidden));
    for (long long item = begin; item < end; ++item) {
      long long selected = 0;
      for (long long token = 0; token < sequence; ++token) {
        selected += mask[item * sequence + token] != 0;
      }
      if (selected == 0) {
        for (long long feature = 0; feature < hidden; ++feature) {
          detail::store_storage<Type>(out, item * hidden + feature, 0.0f);
        }
        continue;
      }
      const double inverse_selected = 1.0 / static_cast<double>(selected);
      double mean_square = 0.0;
      for (long long feature = 0; feature < hidden; ++feature) {
        double sum = 0.0;
        for (long long token = 0; token < sequence; ++token) {
          if (mask[item * sequence + token] != 0) {
            sum += detail::load_storage<Type>(
                x, (item * sequence + token) * hidden + feature);
          }
        }
        pooled[feature] = static_cast<float>(sum * inverse_selected);
        mean_square += static_cast<double>(pooled[feature]) * pooled[feature];
      }
      const double inverse_rms =
          1.0 / std::sqrt(mean_square / static_cast<double>(hidden) + eps);
      double l2_square = 0.0;
      for (long long feature = 0; feature < hidden; ++feature) {
        pooled[feature] = static_cast<float>(
            pooled[feature] * inverse_rms *
            detail::load_storage<Type>(weight, feature));
        l2_square += static_cast<double>(pooled[feature]) * pooled[feature];
      }
      const double inverse_l2 = 1.0 / std::sqrt(l2_square + 1.0e-12);
      for (long long feature = 0; feature < hidden; ++feature) {
        detail::store_storage<Type>(
            out, item * hidden + feature,
            static_cast<float>(pooled[feature] * inverse_l2));
      }
    }
  });
  return Status::kOk;
}

}  // namespace

Status calibration_absmax(const float* x, const float* running, float* out,
                          long long tokens, long long channels) {
  if (!detail::valid_product({tokens, channels})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, out)) return Status::kInvalidArgument;
  threading::parallel_ranges(
      channels, 64, [&](long long begin, long long end, int) {
        for (long long channel = begin; channel < end; ++channel) {
          float maximum = running == nullptr ? 0.0f : running[channel];
          if (!std::isnan(maximum)) {
            maximum = std::fabs(maximum);
            for (long long token = 0; token < tokens; ++token) {
              const float value = std::fabs(x[token * channels + channel]);
              if (std::isnan(value)) {
                maximum = value;
                break;
              }
              maximum = std::max(maximum, value);
            }
          }
          out[channel] = maximum;
        }
      });
  return Status::kOk;
}

Status logits_softcap(const float* logits, float* out, long long count,
                      float cap) {
  if (count <= 0) return Status::kInvalidShape;
  if (!detail::all_nonnull(logits, out) || !std::isfinite(cap) || cap <= 0.0f) {
    return Status::kInvalidArgument;
  }
  const float inverse_cap = 1.0f / cap;
  threading::parallel_ranges(count, 4096,
                             [&](long long begin, long long end, int) {
    for (long long index = begin; index < end; ++index) {
      out[index] = cap * std::tanh(logits[index] * inverse_cap);
    }
  });
  return Status::kOk;
}

Status embedding_lookup_types(const int* token_ids, const int* type_ids,
                              const float* token_table,
                              const float* type_table, float* out,
                              long long token_vocab, long long type_vocab,
                              long long count, long long dim,
                              float token_scale) {
  if (!detail::valid_product({token_vocab, type_vocab, count, dim})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(token_ids, type_ids, token_table, type_table, out) ||
      !std::isfinite(token_scale)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(count, 1,
                             [&](long long begin, long long end, int) {
    for (long long token = begin; token < end; ++token) {
      const int token_id = token_ids[token];
      const int type_id = type_ids[token];
      const float* token_row = token_id >= 0 && token_id < token_vocab
                                   ? token_table + static_cast<long long>(token_id) * dim
                                   : nullptr;
      const float* type_row = type_id >= 0 && type_id < type_vocab
                                  ? type_table + static_cast<long long>(type_id) * dim
                                  : nullptr;
      float* destination = out + token * dim;
      for (long long feature = 0; feature < dim; ++feature) {
        destination[feature] =
            (token_row == nullptr ? 0.0f : token_scale * token_row[feature]) +
            (type_row == nullptr ? 0.0f : type_row[feature]);
      }
    }
  });
  return Status::kOk;
}

Status masked_mean_pool_rms_l2(const float* x, const int* mask,
                               const float* weight, float* out,
                               long long batch, long long sequence,
                               long long hidden, float eps) {
  if (!detail::valid_product({batch, sequence, hidden}) ||
      !std::isfinite(eps) || eps < 0.0f) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, mask, weight, out)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(batch, 1,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      float* destination = out + item * hidden;
      long long selected = 0;
      for (long long token = 0; token < sequence; ++token) {
        selected += mask[item * sequence + token] != 0;
      }
      if (selected == 0) {
        std::fill_n(destination, hidden, 0.0f);
        continue;
      }
      std::fill_n(destination, hidden, 0.0f);
      for (long long token = 0; token < sequence; ++token) {
        if (mask[item * sequence + token] == 0) continue;
        const float* source = x + (item * sequence + token) * hidden;
        for (long long feature = 0; feature < hidden; ++feature) {
          destination[feature] += source[feature];
        }
      }
      const float inverse_selected = 1.0f / static_cast<float>(selected);
      double mean_square = 0.0;
      for (long long feature = 0; feature < hidden; ++feature) {
        destination[feature] *= inverse_selected;
        mean_square += static_cast<double>(destination[feature]) *
                       destination[feature];
      }
      const double inverse_rms =
          1.0 / std::sqrt(mean_square / static_cast<double>(hidden) + eps);
      double l2_square = 0.0;
      for (long long feature = 0; feature < hidden; ++feature) {
        destination[feature] = static_cast<float>(
            destination[feature] * inverse_rms * weight[feature]);
        l2_square += static_cast<double>(destination[feature]) *
                     destination[feature];
      }
      const double inverse_l2 = 1.0 / std::sqrt(l2_square + 1.0e-12);
      for (long long feature = 0; feature < hidden; ++feature) {
        destination[feature] =
            static_cast<float>(destination[feature] * inverse_l2);
      }
    }
  });
  return Status::kOk;
}

Status calibration_absmax_storage(FloatStorageInput x, const float* running,
                                  float* out, long long tokens,
                                  long long channels,
                                  FloatStorageWorkspace* workspace) {
  if (!detail::valid_product({tokens, channels}) ||
      x.count != tokens * channels) {
    return Status::kInvalidShape;
  }
  switch (x.type) {
    case FloatStorageType::kF32:
      return calibration_absmax_typed<FloatStorageType::kF32>(
          x.data, running, out, tokens, channels);
    case FloatStorageType::kF16:
      return calibration_absmax_typed<FloatStorageType::kF16>(
          x.data, running, out, tokens, channels);
    case FloatStorageType::kBF16:
      return calibration_absmax_typed<FloatStorageType::kBF16>(
          x.data, running, out, tokens, channels);
  }
  const FloatStorageOutput output{out, FloatStorageType::kF32, channels};
  return with_float_storage(
      &x, 1, &output, 1,
      [&](const float* const* inputs, float* const* outputs) {
        return calibration_absmax(inputs[0], running, outputs[0], tokens,
                                  channels);
      },
      workspace);
}

Status logits_softcap_storage(FloatStorageInput logits, FloatStorageOutput out,
                              float cap, FloatStorageWorkspace* workspace) {
  if (logits.count <= 0 || logits.count != out.count) {
    return Status::kInvalidShape;
  }
  return with_float_storage(
      &logits, 1, &out, 1,
      [&](const float* const* inputs, float* const* outputs) {
        return logits_softcap(inputs[0], outputs[0], logits.count, cap);
      },
      workspace);
}

Status embedding_lookup_types_storage(
    const int* token_ids, const int* type_ids, FloatStorageInput token_table,
    FloatStorageInput type_table, FloatStorageOutput out,
    long long token_vocab, long long type_vocab, long long count,
    long long dim, float token_scale, FloatStorageWorkspace* workspace) {
  if (!detail::valid_product({token_vocab, type_vocab, count, dim}) ||
      token_table.count != token_vocab * dim ||
      type_table.count != type_vocab * dim || out.count != count * dim) {
    return Status::kInvalidShape;
  }
  if (token_table.type == type_table.type && token_table.type == out.type) {
    switch (out.type) {
      case FloatStorageType::kF32:
        return embedding_lookup_types_typed<FloatStorageType::kF32>(
            token_ids, type_ids, token_table.data, type_table.data, out.data,
            token_vocab, type_vocab, count, dim, token_scale);
      case FloatStorageType::kF16:
        return embedding_lookup_types_typed<FloatStorageType::kF16>(
            token_ids, type_ids, token_table.data, type_table.data, out.data,
            token_vocab, type_vocab, count, dim, token_scale);
      case FloatStorageType::kBF16:
        return embedding_lookup_types_typed<FloatStorageType::kBF16>(
            token_ids, type_ids, token_table.data, type_table.data, out.data,
            token_vocab, type_vocab, count, dim, token_scale);
    }
  }
  const FloatStorageInput inputs[] = {token_table, type_table};
  return with_float_storage(
      inputs, 2, &out, 1,
      [&](const float* const* values, float* const* outputs) {
        return embedding_lookup_types(token_ids, type_ids, values[0], values[1],
                                      outputs[0], token_vocab, type_vocab,
                                      count, dim, token_scale);
      },
      workspace);
}

Status masked_mean_pool_rms_l2_storage(
    FloatStorageInput x, const int* mask, FloatStorageInput weight,
    FloatStorageOutput out, long long batch, long long sequence,
    long long hidden, float eps, FloatStorageWorkspace* workspace) {
  if (!detail::valid_product({batch, sequence, hidden}) ||
      x.count != batch * sequence * hidden || weight.count != hidden ||
      out.count != batch * hidden) {
    return Status::kInvalidShape;
  }
  const FloatStorageInput inputs[] = {x, weight};
  return with_float_storage(
      inputs, 2, &out, 1,
      [&](const float* const* values, float* const* outputs) {
        return masked_mean_pool_rms_l2(values[0], mask, values[1], outputs[0],
                                       batch, sequence, hidden, eps);
      },
      workspace);
}

}  // namespace quixicore_cpu
