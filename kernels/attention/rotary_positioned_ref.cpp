#include <algorithm>

#include "kernels/common/validation.h"
#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/threading.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

long long resolved_rotary_dim(long long head_dim, long long rotary_dim) {
  return rotary_dim == 0 ? head_dim : rotary_dim;
}

bool valid_rotary(long long head_dim, long long rotary_dim,
                  long long max_position) {
  return head_dim > 0 && rotary_dim > 0 && rotary_dim <= head_dim &&
         rotary_dim % 2 == 0 && max_position > 0;
}

bool valid_sections(const int* sections, long long pairs, bool interleaved) {
  if (sections == nullptr || sections[0] < 0 || sections[1] < 0 ||
      sections[2] < 0 ||
      static_cast<long long>(sections[0]) + sections[1] + sections[2] !=
          pairs) {
    return false;
  }
  return !interleaved ||
         (sections[0] == (pairs + 2) / 3 && sections[1] == (pairs + 1) / 3 &&
          sections[2] == pairs / 3);
}

int pair_axis(long long pair, const int* sections, bool interleaved) {
  if (interleaved) return static_cast<int>(pair % 3);
  if (pair < sections[0]) return 0;
  return pair < static_cast<long long>(sections[0]) + sections[1] ? 1 : 2;
}

void rotate_row(const float* x, const float* cosine, const float* sine,
                float* y, long long head_dim, long long rotary_dim,
                bool interleaved) {
  const long long pairs = rotary_dim / 2;
  for (long long pair = 0; pair < pairs; ++pair) {
    const long long first_index = interleaved ? 2 * pair : pair;
    const long long second_index = interleaved ? 2 * pair + 1 : pairs + pair;
    const float first = x[first_index];
    const float second = x[second_index];
    y[first_index] = first * cosine[pair] - second * sine[pair];
    y[second_index] = second * cosine[pair] + first * sine[pair];
  }
  std::copy_n(x + rotary_dim, head_dim - rotary_dim, y + rotary_dim);
}

void rotate_mrope_row(const float* x, const float* cosine, const float* sine,
                      const int* positions, const int* sections, float* y,
                      long long head_dim, long long rotary_dim,
                      bool section_interleaved) {
  const long long pairs = rotary_dim / 2;
  if (section_interleaved) {
    for (long long first_pair = 0; first_pair < pairs; first_pair += 3) {
      const long long group = std::min(3LL, pairs - first_pair);
      for (long long axis = 0; axis < group; ++axis) {
        const long long pair = first_pair + axis;
        const long long table =
            static_cast<long long>(positions[axis]) * pairs + pair;
        const float first = x[pair];
        const float second = x[pairs + pair];
        y[pair] = first * cosine[table] - second * sine[table];
        y[pairs + pair] = second * cosine[table] + first * sine[table];
      }
    }
  } else {
    for (long long pair = 0; pair < pairs; ++pair) {
      const int axis = pair_axis(pair, sections, false);
      const long long table =
          static_cast<long long>(positions[axis]) * pairs + pair;
      const float first = x[pair];
      const float second = x[pairs + pair];
      y[pair] = first * cosine[table] - second * sine[table];
      y[pairs + pair] = second * cosine[table] + first * sine[table];
    }
  }
  std::copy_n(x + rotary_dim, head_dim - rotary_dim, y + rotary_dim);
}

}  // namespace

Status rotary_positioned(const float* x, const float* cosine, const float* sine,
                         const int* positions, float* y, long long batch,
                         long long heads, long long tokens, long long head_dim,
                         long long rotary_dim, long long max_position,
                         bool interleaved, bool positions_per_batch) {
  const long long rd = resolved_rotary_dim(head_dim, rotary_dim);
  if (!detail::valid_product({batch, heads, tokens, head_dim}) ||
      !valid_rotary(head_dim, rd, max_position)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, cosine, sine, positions, y)) {
    return Status::kInvalidArgument;
  }
  const long long position_count =
      positions_per_batch ? batch * tokens : tokens;
  for (long long index = 0; index < position_count; ++index) {
    if (positions[index] < 0 || positions[index] >= max_position) {
      return Status::kInvalidArgument;
    }
  }
  const long long pairs = rd / 2;
  if (num_threads() == 1) {
    for (long long item = 0; item < batch; ++item) {
      for (long long head = 0; head < heads; ++head) {
        for (long long token = 0; token < tokens; ++token) {
          const long long position_index =
              (positions_per_batch ? item * tokens : 0) + token;
          const long long table =
              static_cast<long long>(positions[position_index]) * pairs;
          const long long row =
              ((item * heads + head) * tokens + token) * head_dim;
          rotate_row(x + row, cosine + table, sine + table, y + row, head_dim,
                     rd, interleaved);
        }
      }
    }
    return Status::kOk;
  }
  threading::parallel_ranges(
      batch * heads * tokens, 16, [&](long long begin, long long end, int) {
        for (long long row_index = begin; row_index < end; ++row_index) {
          const long long token = row_index % tokens;
          const long long item = row_index / (heads * tokens);
          const long long position_index =
              (positions_per_batch ? item * tokens : 0) + token;
          const long long table =
              static_cast<long long>(positions[position_index]) * pairs;
          const long long row = row_index * head_dim;
          rotate_row(x + row, cosine + table, sine + table, y + row, head_dim,
                     rd, interleaved);
        }
      });
  return Status::kOk;
}

Status mrope(const float* x, const float* cosine, const float* sine,
             const int* positions, const int* sections, float* y,
             long long batch, long long heads, long long tokens,
             long long head_dim, long long rotary_dim, long long max_position,
             bool section_interleaved, bool positions_per_batch) {
  const long long rd = resolved_rotary_dim(head_dim, rotary_dim);
  if (!detail::valid_product({batch, heads, tokens, head_dim}) ||
      !valid_rotary(head_dim, rd, max_position) ||
      !valid_sections(sections, rd / 2, section_interleaved)) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, cosine, sine, positions, sections, y)) {
    return Status::kInvalidArgument;
  }
  const long long position_count =
      (positions_per_batch ? batch : 1) * 3 * tokens;
  for (long long index = 0; index < position_count; ++index) {
    if (positions[index] < 0 || positions[index] >= max_position) {
      return Status::kInvalidArgument;
    }
  }
  if (num_threads() == 1) {
    for (long long item = 0; item < batch; ++item) {
      const int* item_positions =
          positions + (positions_per_batch ? item * 3 * tokens : 0);
      for (long long head = 0; head < heads; ++head) {
        for (long long token = 0; token < tokens; ++token) {
          const int selected[3] = {item_positions[token],
                                   item_positions[tokens + token],
                                   item_positions[2 * tokens + token]};
          const long long row =
              ((item * heads + head) * tokens + token) * head_dim;
          rotate_mrope_row(x + row, cosine, sine, selected, sections, y + row,
                           head_dim, rd, section_interleaved);
        }
      }
    }
    return Status::kOk;
  }
  threading::parallel_ranges(
      batch * heads * tokens, 16, [&](long long begin, long long end, int) {
        for (long long row_index = begin; row_index < end; ++row_index) {
          const long long token = row_index % tokens;
          const long long item = row_index / (heads * tokens);
          const int* item_positions =
              positions + (positions_per_batch ? item * 3 * tokens : 0);
          const int selected[3] = {item_positions[token],
                                   item_positions[tokens + token],
                                   item_positions[2 * tokens + token]};
          const long long row = row_index * head_dim;
          rotate_mrope_row(x + row, cosine, sine, selected, sections, y + row,
                           head_dim, rd, section_interleaved);
        }
      });
  return Status::kOk;
}

}  // namespace quixicore_cpu
