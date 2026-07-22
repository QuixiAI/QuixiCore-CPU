#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "kernels/common/validation.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

double row_lse(const float* values, long long count, float multiplier) {
  double maximum = -std::numeric_limits<double>::infinity();
  for (long long i = 0; i < count; ++i) {
    maximum = std::max(maximum, static_cast<double>(values[i]) * multiplier);
  }
  double sum = 0.0;
  for (long long i = 0; i < count; ++i) {
    sum += std::exp(static_cast<double>(values[i]) * multiplier - maximum);
  }
  return maximum + std::log(sum);
}

bool valid_kd_shape(long long rows, long long vocab, float inverse_temperature) {
  return detail::valid_product({rows, vocab}) &&
         std::isfinite(inverse_temperature) && inverse_temperature > 0.0f;
}

}  // namespace

Status kd_kl_dense_forward(const float* teacher, const float* student,
                           float* loss, float* teacher_lse,
                           float* student_lse, long long rows,
                           long long vocab, float inverse_temperature) {
  if (!valid_kd_shape(rows, vocab, inverse_temperature)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(teacher, student, loss, teacher_lse, student_lse)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(rows, 4, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      const float* tr = teacher + row * vocab;
      const float* sr = student + row * vocab;
      const double tlse = row_lse(tr, vocab, inverse_temperature);
      const double slse = row_lse(sr, vocab, inverse_temperature);
      double value = 0.0;
      for (long long token = 0; token < vocab; ++token) {
        const double teacher_log = tr[token] * inverse_temperature - tlse;
        const double student_log = sr[token] * inverse_temperature - slse;
        value += std::exp(teacher_log) * (teacher_log - student_log);
      }
      loss[row] = static_cast<float>(value);
      teacher_lse[row] = static_cast<float>(tlse);
      student_lse[row] = static_cast<float>(slse);
    }
  });
  return Status::kOk;
}

Status kd_kl_dense_backward(const float* teacher, const float* student,
                            const float* teacher_lse,
                            const float* student_lse,
                            const float* grad_out, float* grad_student,
                            long long rows, long long vocab,
                            float inverse_temperature) {
  if (!valid_kd_shape(rows, vocab, inverse_temperature)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(teacher, student, teacher_lse, student_lse,
                           grad_out, grad_student)) {
    return Status::kInvalidArgument;
  }
  threading::parallel_ranges(rows, 4, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      for (long long token = 0; token < vocab; ++token) {
        const double q = std::exp(student[row * vocab + token] *
                                      inverse_temperature - student_lse[row]);
        const double p = std::exp(teacher[row * vocab + token] *
                                      inverse_temperature - teacher_lse[row]);
        grad_student[row * vocab + token] = static_cast<float>(
            grad_out[row] * inverse_temperature * (q - p));
      }
    }
  });
  return Status::kOk;
}

Status kd_kl_topk_forward(const float* student, const int* teacher_indices,
                          const float* teacher_probabilities, float* loss,
                          float* student_lse, long long rows,
                          long long vocab, long long top_k,
                          float inverse_temperature, bool include_tail) {
  if (!valid_kd_shape(rows, vocab, inverse_temperature) || top_k <= 0 ||
      !detail::valid_product({rows, top_k})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(student, teacher_indices, teacher_probabilities,
                           loss, student_lse)) {
    return Status::kInvalidArgument;
  }
  for (long long i = 0; i < rows * top_k; ++i) {
    if (teacher_indices[i] < -1 || teacher_indices[i] >= vocab ||
        !std::isfinite(teacher_probabilities[i]) ||
        teacher_probabilities[i] < 0.0f) {
      return Status::kInvalidArgument;
    }
  }
  constexpr double kTiny = 1e-30;
  threading::parallel_ranges(rows, 4, [&](long long begin, long long end, int) {
    for (long long row = begin; row < end; ++row) {
      const float* sr = student + row * vocab;
      const double lse = row_lse(sr, vocab, inverse_temperature);
      double probability_sum = 0.0;
      double student_selected = 0.0;
      for (long long item = 0; item < top_k; ++item) {
        const long long index = row * top_k + item;
        const int token = teacher_indices[index];
        if (token >= 0) {
          probability_sum += teacher_probabilities[index];
          student_selected +=
              std::exp(sr[token] * inverse_temperature - lse);
        }
      }
      double value = 0.0;
      const double inverse_sum = 1.0 / std::max(probability_sum, kTiny);
      for (long long item = 0; item < top_k; ++item) {
        const long long index = row * top_k + item;
        const int token = teacher_indices[index];
        if (token < 0) continue;
        double p = teacher_probabilities[index];
        if (!include_tail) p *= inverse_sum;
        if (p > 0.0) {
          const double log_q = sr[token] * inverse_temperature - lse;
          value += p * (std::log(std::max(p, kTiny)) - log_q);
        }
      }
      if (include_tail) {
        const double tail = std::max(1.0 - probability_sum, 0.0);
        if (tail > 0.0) {
          value += tail * (std::log(std::max(tail, kTiny)) -
                           std::log(std::max(1.0 - student_selected, kTiny)));
        }
      }
      loss[row] = static_cast<float>(value);
      student_lse[row] = static_cast<float>(lse);
    }
  });
  return Status::kOk;
}

Status kd_kl_topk_backward(const float* student,
                           const int* teacher_indices,
                           const float* teacher_probabilities,
                           const float* student_lse,
                           const float* grad_out, float* grad_student,
                           long long rows, long long vocab, long long top_k,
                           float inverse_temperature, bool include_tail) {
  if (!valid_kd_shape(rows, vocab, inverse_temperature) || top_k <= 0 ||
      !detail::valid_product({rows, top_k})) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(student, teacher_indices, teacher_probabilities,
                           student_lse, grad_out, grad_student)) {
    return Status::kInvalidArgument;
  }
  constexpr double kTiny = 1e-30;
  for (long long row = 0; row < rows; ++row) {
    double probability_sum = 0.0;
    double student_selected = 0.0;
    for (long long item = 0; item < top_k; ++item) {
      const long long index = row * top_k + item;
      const int token = teacher_indices[index];
      if (token < -1 || token >= vocab ||
          !std::isfinite(teacher_probabilities[index]) ||
          teacher_probabilities[index] < 0.0f) {
        return Status::kInvalidArgument;
      }
      if (token >= 0) {
        probability_sum += teacher_probabilities[index];
        student_selected += std::exp(student[row * vocab + token] *
                                         inverse_temperature - student_lse[row]);
      }
    }
    const double tail = std::max(1.0 - probability_sum, 0.0);
    const double tail_c = include_tail && tail > 0.0
                              ? tail / std::max(1.0 - student_selected, kTiny)
                              : 0.0;
    const double q_coefficient = include_tail
                                     ? probability_sum - tail_c * student_selected
                                     : 1.0;
    const double inverse_sum = 1.0 / std::max(probability_sum, kTiny);
    const double go = grad_out[row] * inverse_temperature;
    for (long long token = 0; token < vocab; ++token) {
      const double q = std::exp(student[row * vocab + token] *
                                    inverse_temperature - student_lse[row]);
      grad_student[row * vocab + token] =
          static_cast<float>(q_coefficient * q * go);
    }
    for (long long item = 0; item < top_k; ++item) {
      const long long index = row * top_k + item;
      const int token = teacher_indices[index];
      if (token < 0) continue;
      const double p = teacher_probabilities[index];
      const double q = std::exp(student[row * vocab + token] *
                                    inverse_temperature - student_lse[row]);
      const double correction = include_tail ? -p + tail_c * q
                                             : -p * inverse_sum;
      grad_student[row * vocab + token] +=
          static_cast<float>(correction * go);
    }
  }
  return Status::kOk;
}

Status kd_ce_fused_forward(
    const float* teacher, const float* student, const int* targets,
    float* cross_entropy_loss, float* kd_loss, float* raw_student_lse,
    float* tempered_student_lse, float* teacher_lse, long long rows,
    long long vocab, float inverse_temperature, int ignore_index) {
  if (!valid_kd_shape(rows, vocab, inverse_temperature)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(teacher, student, targets, cross_entropy_loss,
                           kd_loss, raw_student_lse, tempered_student_lse,
                           teacher_lse)) {
    return Status::kInvalidArgument;
  }
  Status status = kd_kl_dense_forward(
      teacher, student, kd_loss, teacher_lse, tempered_student_lse, rows,
      vocab, inverse_temperature);
  if (status != Status::kOk) return status;
  for (long long row = 0; row < rows; ++row) {
    const int target = targets[row];
    if (target == ignore_index) {
      cross_entropy_loss[row] = 0.0f;
      raw_student_lse[row] = static_cast<float>(
          row_lse(student + row * vocab, vocab, 1.0f));
      continue;
    }
    if (target < 0 || target >= vocab) return Status::kInvalidArgument;
    const double lse = row_lse(student + row * vocab, vocab, 1.0f);
    raw_student_lse[row] = static_cast<float>(lse);
    cross_entropy_loss[row] =
        static_cast<float>(lse - student[row * vocab + target]);
  }
  return Status::kOk;
}

Status kd_ce_fused_backward(
    const float* teacher, const float* student, const int* targets,
    const float* raw_student_lse, const float* tempered_student_lse,
    const float* teacher_lse, const float* grad_cross_entropy,
    const float* grad_kd, float* grad_student, long long rows,
    long long vocab, float inverse_temperature, int ignore_index) {
  if (!valid_kd_shape(rows, vocab, inverse_temperature)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(teacher, student, targets, raw_student_lse,
                           tempered_student_lse, teacher_lse,
                           grad_cross_entropy, grad_kd, grad_student)) {
    return Status::kInvalidArgument;
  }
  for (long long row = 0; row < rows; ++row) {
    const int target = targets[row];
    if (target != ignore_index && (target < 0 || target >= vocab)) {
      return Status::kInvalidArgument;
    }
    for (long long token = 0; token < vocab; ++token) {
      const double student_probability =
          std::exp(student[row * vocab + token] * inverse_temperature -
                   tempered_student_lse[row]);
      const double teacher_probability =
          std::exp(teacher[row * vocab + token] * inverse_temperature -
                   teacher_lse[row]);
      double gradient = grad_kd[row] * inverse_temperature *
                        (student_probability - teacher_probability);
      if (target != ignore_index) {
        gradient += grad_cross_entropy[row] *
                    (std::exp(student[row * vocab + token] -
                              raw_student_lse[row]) -
                     (token == target ? 1.0 : 0.0));
      }
      grad_student[row * vocab + token] = static_cast<float>(gradient);
    }
  }
  return Status::kOk;
}

Status tau_tail(const float* qkv, const float* token_qv_linear,
                const float* position_table, const int* positions, float* out,
                long long tokens, long long heads, long long head_dim,
                long long max_position) {
  if (!detail::valid_product({tokens, heads, head_dim, 3}) ||
      max_position <= 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(qkv, token_qv_linear, position_table, positions,
                           out)) {
    return Status::kInvalidArgument;
  }
  for (long long token = 0; token < tokens; ++token) {
    if (positions[token] < 0 || positions[token] >= max_position) {
      return Status::kInvalidArgument;
    }
  }
  const long long qdim = heads * head_dim;
  if (out != qkv) std::copy_n(qkv, tokens * 3 * qdim, out);
  threading::parallel_ranges(tokens * heads, 32,
                             [&](long long begin, long long end, int) {
    for (long long item = begin; item < end; ++item) {
      const long long token = item / heads;
      const long long head = item % heads;
      const float positional =
          position_table[positions[token] * heads + head];
      const float q_scale =
          std::tanh(token_qv_linear[token * 2 * heads + head]) + positional;
      const float v_scale = std::tanh(
          token_qv_linear[token * 2 * heads + heads + head]) + positional;
      for (long long dim = 0; dim < head_dim; ++dim) {
        out[token * 3 * qdim + head * head_dim + dim] *= q_scale;
        out[token * 3 * qdim + 2 * qdim + head * head_dim + dim] *= v_scale;
      }
    }
  });
  return Status::kOk;
}

}  // namespace quixicore_cpu
