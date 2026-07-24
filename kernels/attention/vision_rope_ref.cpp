#include <algorithm>

#include "kernels/common/validation.h"
#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/ops.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

Status vision_rope_2d_impl(
    const float* x, const float* cosine, const float* sine,
    const int* positions, float* out, long long batch, long long heads,
    long long tokens, long long head_dim, long long max_position,
    bool global_split) {
  if (!detail::valid_product({batch, heads, tokens, head_dim}) ||
      !(head_dim == 64 || head_dim == 128 || head_dim == 256 ||
        head_dim == 512) ||
      max_position <= 0) {
    return Status::kInvalidShape;
  }
  if (!detail::all_nonnull(x, cosine, sine, positions, out)) {
    return Status::kInvalidArgument;
  }
  const long long pairs = head_dim / 4;
  const long long rows_per_batch = heads * tokens;
  threading::parallel_ranges(
      batch * rows_per_batch, 8,
      [&](long long begin, long long end, int) {
        for (long long row = begin; row < end; ++row) {
          const long long item = row / rows_per_batch;
          const long long token = row % tokens;
          const int position_x = std::clamp(
              positions[(item * tokens + token) * 2], 0,
              static_cast<int>(max_position - 1));
          const int position_y = std::clamp(
              positions[(item * tokens + token) * 2 + 1], 0,
              static_cast<int>(max_position - 1));
          const float* cos_x = cosine + position_x * pairs;
          const float* sin_x = sine + position_x * pairs;
          const float* cos_y = cosine + position_y * pairs;
          const float* sin_y = sine + position_y * pairs;
          const float* source = x + row * head_dim;
          float* destination = out + row * head_dim;
          if (global_split) {
            for (long long pair = 0; pair < pairs; ++pair) {
              const float x0 = source[pair];
              const float y0 = source[pairs + pair];
              const float x1 = source[2 * pairs + pair];
              const float y1 = source[3 * pairs + pair];
              destination[pair] =
                  x0 * cos_x[pair] - x1 * sin_x[pair];
              destination[pairs + pair] =
                  y0 * cos_y[pair] - y1 * sin_y[pair];
              destination[2 * pairs + pair] =
                  x0 * sin_x[pair] + x1 * cos_x[pair];
              destination[3 * pairs + pair] =
                  y0 * sin_y[pair] + y1 * cos_y[pair];
            }
          } else {
            for (long long pair = 0; pair < pairs; ++pair) {
              const float x0 = source[pair];
              const float x1 = source[pairs + pair];
              const float y0 = source[2 * pairs + pair];
              const float y1 = source[3 * pairs + pair];
              destination[pair] =
                  x0 * cos_x[pair] - x1 * sin_x[pair];
              destination[pairs + pair] =
                  x0 * sin_x[pair] + x1 * cos_x[pair];
              destination[2 * pairs + pair] =
                  y0 * cos_y[pair] - y1 * sin_y[pair];
              destination[3 * pairs + pair] =
                  y0 * sin_y[pair] + y1 * cos_y[pair];
            }
          }
        }
      });
  return Status::kOk;
}

}  // namespace

Status vision_rope_2d(const float* x, const float* cosine, const float* sine,
                      const int* positions, float* out, long long batch,
                      long long heads, long long tokens, long long head_dim,
                      long long max_position) {
  return vision_rope_2d_impl(x, cosine, sine, positions, out, batch, heads,
                             tokens, head_dim, max_position, false);
}

Status qwen_vision_rope_2d(
    const float* x, const float* cosine, const float* sine,
    const int* positions, float* out, long long batch, long long heads,
    long long tokens, long long head_dim, long long max_position) {
  return vision_rope_2d_impl(x, cosine, sine, positions, out, batch, heads,
                             tokens, head_dim, max_position, true);
}

Status vision_rope_2d_storage(
    FloatStorageInput x, FloatStorageInput cosine, FloatStorageInput sine,
    const int* positions, FloatStorageOutput out, long long batch,
    long long heads, long long tokens, long long head_dim,
    long long max_position, FloatStorageWorkspace* workspace) {
  if (!detail::valid_product({batch, heads, tokens, head_dim}) ||
      !detail::valid_product({max_position, head_dim / 4}) ||
      x.count != batch * heads * tokens * head_dim || out.count != x.count ||
      cosine.count != max_position * (head_dim / 4) ||
      sine.count != cosine.count) {
    return Status::kInvalidShape;
  }
  const FloatStorageInput inputs[] = {x, cosine, sine};
  return with_float_storage(
      inputs, 3, &out, 1,
      [&](const float* const* values, float* const* outputs) {
        return vision_rope_2d(values[0], values[1], values[2], positions,
                              outputs[0], batch, heads, tokens, head_dim,
                              max_position);
      },
      workspace);
}

Status qwen_vision_rope_2d_storage(
    FloatStorageInput x, FloatStorageInput cosine, FloatStorageInput sine,
    const int* positions, FloatStorageOutput out, long long batch,
    long long heads, long long tokens, long long head_dim,
    long long max_position, FloatStorageWorkspace* workspace) {
  if (!detail::valid_product({batch, heads, tokens, head_dim}) ||
      !detail::valid_product({max_position, head_dim / 4}) ||
      x.count != batch * heads * tokens * head_dim || out.count != x.count ||
      cosine.count != max_position * (head_dim / 4) ||
      sine.count != cosine.count) {
    return Status::kInvalidShape;
  }
  const FloatStorageInput inputs[] = {x, cosine, sine};
  return with_float_storage(
      inputs, 3, &out, 1,
      [&](const float* const* values, float* const* outputs) {
        return qwen_vision_rope_2d(
            values[0], values[1], values[2], positions, outputs[0], batch,
            heads, tokens, head_dim, max_position);
      },
      workspace);
}

}  // namespace quixicore_cpu
