#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#include "kernels/common/fp16.h"
#include "kernels/utils/float_storage_isa.h"
#include "quixicore_cpu/cpu_features.h"
#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/qgemm.h"
#include "quixicore_cpu/qgemv.h"
#include "quixicore_cpu/rms_norm.h"
#include "src/threading/thread_pool.h"

namespace quixicore_cpu {
namespace {

constexpr long long kParallelElements = 1 << 16;

std::size_t storage_size(FloatStorageType type) {
  return type == FloatStorageType::kF32 ? sizeof(float) : sizeof(std::uint16_t);
}

bool known_type(FloatStorageType type) {
  return type == FloatStorageType::kF32 || type == FloatStorageType::kF16 ||
         type == FloatStorageType::kBF16;
}

bool bytes_for(FloatStorageType type, long long count, std::size_t* bytes) {
  if (!known_type(type) || count < 0) return false;
  const auto size = storage_size(type);
  const auto n = static_cast<unsigned long long>(count);
  if (n > std::numeric_limits<std::size_t>::max() / size) return false;
  *bytes = static_cast<std::size_t>(n) * size;
  return true;
}

bool matches_product(long long count,
                     std::initializer_list<long long> dimensions) {
  if (count < 0) return false;
  long long product = 1;
  for (const long long dimension : dimensions) {
    if (dimension <= 0 ||
        product > std::numeric_limits<long long>::max() / dimension) {
      return false;
    }
    product *= dimension;
  }
  return count == product;
}

bool distinct_storage(const void* input, std::size_t input_bytes,
                      const void* output, std::size_t output_bytes) {
  const auto in = reinterpret_cast<std::uintptr_t>(input);
  const auto out = reinterpret_cast<std::uintptr_t>(output);
  if (in > std::numeric_limits<std::uintptr_t>::max() - input_bytes ||
      out > std::numeric_limits<std::uintptr_t>::max() - output_bytes) {
    return false;
  }
  return in + input_bytes <= out || out + output_bytes <= in;
}

struct Region {
  const void* data;
  FloatStorageType type;
  long long count;
  std::size_t bytes;
};

bool same_region(const Region& a, const Region& b) {
  return a.data == b.data && a.type == b.type && a.count == b.count;
}

bool overlaps(const Region& a, const Region& b) {
  if (a.bytes == 0 || b.bytes == 0) return false;
  const auto ab = reinterpret_cast<std::uintptr_t>(a.data);
  const auto bb = reinterpret_cast<std::uintptr_t>(b.data);
  if (ab > std::numeric_limits<std::uintptr_t>::max() - a.bytes ||
      bb > std::numeric_limits<std::uintptr_t>::max() - b.bytes) {
    return true;
  }
  return ab < bb + b.bytes && bb < ab + a.bytes;
}

template <class Fn>
void parallel_convert(long long count, Fn&& fn) {
  threading::parallel_ranges(
      count, kParallelElements,
      [&](long long begin, long long end, int) { fn(begin, end); });
}

struct Group {
  Region region;
  bool has_input = false;
  bool has_output = false;
  std::size_t offset = 0;
};

struct UnaryContext {
  long long count;
  UnaryOp op;
  XiEluParams params;
};

Status call_unary(const float* const* in, float* const* out, void* opaque) {
  const auto& ctx = *static_cast<UnaryContext*>(opaque);
  if (static_cast<int>(ctx.op) < static_cast<int>(UnaryOp::kAbs) ||
      static_cast<int>(ctx.op) > static_cast<int>(UnaryOp::kTrunc)) {
    return Status::kInvalidArgument;
  }
  return unary(in[0], out[0], ctx.count, ctx.op, ctx.params);
}

struct SoftmaxContext {
  long long rows, dim;
};
Status call_softmax(const float* const* in, float* const* out, void* opaque) {
  const auto& ctx = *static_cast<SoftmaxContext*>(opaque);
  return softmax(in[0], out[0], ctx.rows, ctx.dim);
}

struct RmsContext {
  long long rows, hidden;
  float eps;
};
Status call_rms(const float* const* in, float* const* out, void* opaque) {
  const auto& ctx = *static_cast<RmsContext*>(opaque);
  return rms_norm(in[0], in[1], out[0], ctx.rows, ctx.hidden, ctx.eps);
}

struct GemmContext {
  long long m, n, k;
};
Status call_gemm(const float* const* in, float* const* out, void* opaque) {
  const auto& ctx = *static_cast<GemmContext*>(opaque);
  return dense_gemm(in[0], in[1], out[0], ctx.m, ctx.n, ctx.k);
}

struct AttentionContext {
  long long hq, hkv, sq, sk, dim;
  bool causal;
};
Status call_attention(const float* const* in, float* const* out, void* opaque) {
  const auto& ctx = *static_cast<AttentionContext*>(opaque);
  return attention(in[0], in[1], in[2], out[0], ctx.hq, ctx.hkv, ctx.sq, ctx.sk,
                   ctx.dim, ctx.causal);
}

struct QGemmContext {
  QuantFormat format;
  const void* packed;
  long long m, n, k;
};
Status call_qgemv(const float* const* in, float* const* out, void* opaque) {
  const auto& ctx = *static_cast<QGemmContext*>(opaque);
  return qgemv(ctx.format, ctx.packed, in[0], out[0], ctx.n, ctx.k);
}
Status call_qgemm(const float* const* in, float* const* out, void* opaque) {
  const auto& ctx = *static_cast<QGemmContext*>(opaque);
  return qgemm(ctx.format, ctx.packed, in[0], out[0], ctx.m, ctx.n, ctx.k);
}

}  // namespace

Status FloatStorageWorkspace::reserve(std::size_t float_elements) {
  try {
    scratch_.reserve(float_elements);
    return Status::kOk;
  } catch (const std::bad_alloc&) {
    return Status::kOutOfMemory;
  } catch (const std::length_error&) {
    return Status::kInvalidShape;
  }
}

std::uint16_t float_to_f16(float value) noexcept { return fp32_to_fp16(value); }
float f16_to_float(std::uint16_t bits) noexcept { return fp16_to_fp32(bits); }

std::uint16_t float_to_bf16(float value) noexcept {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof bits);
  if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0) {
    // Keep NaNs as NaNs even when all payload bits live below bit 16.
    return static_cast<std::uint16_t>((bits >> 16) | 0x0040u);
  }
  const std::uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>((bits + rounding_bias) >> 16);
}

float bf16_to_float(std::uint16_t bits) noexcept {
  const std::uint32_t value = static_cast<std::uint32_t>(bits) << 16;
  float out;
  std::memcpy(&out, &value, sizeof out);
  return out;
}

Status float_storage_to_f32(FloatStorageType type, const void* input,
                            float* output, long long count) {
  if (!known_type(type)) return Status::kUnsupportedFormat;
  if (count < 0) return Status::kInvalidShape;
  if (count == 0) return Status::kOk;
  if (input == nullptr || output == nullptr) return Status::kInvalidArgument;
  std::size_t input_bytes = 0;
  if (!bytes_for(type, count, &input_bytes) ||
      static_cast<unsigned long long>(count) >
          std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    return Status::kInvalidShape;
  }
  const std::size_t output_bytes =
      static_cast<std::size_t>(count) * sizeof(float);
  if (type == FloatStorageType::kF32) {
    std::memmove(output, input, output_bytes);
    return Status::kOk;
  }
  if (!distinct_storage(input, input_bytes, output, output_bytes)) {
    return Status::kInvalidArgument;
  }
  const auto* src = static_cast<const std::uint16_t*>(input);
  if (type == FloatStorageType::kF16) {
    parallel_convert(count, [&](long long begin, long long end) {
#if defined(QUIXICORE_CPU_HAVE_FLOAT_STORAGE_NEON_FP16)
      if (cpu_features().fp16) {
        float_storage_detail::f16_to_f32_neon(src, output, begin, end);
        return;
      }
#endif
#if defined(QUIXICORE_CPU_HAVE_FLOAT_STORAGE_F16C)
      if (cpu_features().f16c) {
        float_storage_detail::f16_to_f32_f16c(src, output, begin, end);
        return;
      }
#endif
      for (long long i = begin; i < end; ++i) output[i] = fp16_to_fp32(src[i]);
    });
  } else {
    parallel_convert(count, [&](long long begin, long long end) {
#if defined(QUIXICORE_CPU_HAVE_FLOAT_STORAGE_AVX2)
      if (cpu_features().avx2) {
        float_storage_detail::bf16_to_f32_avx2(src, output, begin, end);
        return;
      }
#endif
      for (long long i = begin; i < end; ++i) output[i] = bf16_to_float(src[i]);
    });
  }
  return Status::kOk;
}

Status float_storage_from_f32(FloatStorageType type, const float* input,
                              void* output, long long count) {
  if (!known_type(type)) return Status::kUnsupportedFormat;
  if (count < 0) return Status::kInvalidShape;
  if (count == 0) return Status::kOk;
  if (input == nullptr || output == nullptr) return Status::kInvalidArgument;
  std::size_t output_bytes = 0;
  if (!bytes_for(type, count, &output_bytes) ||
      static_cast<unsigned long long>(count) >
          std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    return Status::kInvalidShape;
  }
  const std::size_t input_bytes =
      static_cast<std::size_t>(count) * sizeof(float);
  if (type == FloatStorageType::kF32) {
    std::memmove(output, input, input_bytes);
    return Status::kOk;
  }
  if (!distinct_storage(input, input_bytes, output, output_bytes)) {
    return Status::kInvalidArgument;
  }
  auto* dst = static_cast<std::uint16_t*>(output);
  if (type == FloatStorageType::kF16) {
    parallel_convert(count, [&](long long begin, long long end) {
#if defined(QUIXICORE_CPU_HAVE_FLOAT_STORAGE_NEON_FP16)
      if (cpu_features().fp16) {
        float_storage_detail::f32_to_f16_neon(input, dst, begin, end);
        return;
      }
#endif
#if defined(QUIXICORE_CPU_HAVE_FLOAT_STORAGE_F16C)
      if (cpu_features().f16c) {
        float_storage_detail::f32_to_f16_f16c(input, dst, begin, end);
        return;
      }
#endif
      for (long long i = begin; i < end; ++i) dst[i] = fp32_to_fp16(input[i]);
    });
  } else {
    parallel_convert(count, [&](long long begin, long long end) {
      for (long long i = begin; i < end; ++i) dst[i] = float_to_bf16(input[i]);
    });
  }
  return Status::kOk;
}

const char* float_storage_variant(FloatStorageType type) noexcept {
  if (type == FloatStorageType::kF32) return "zero_copy";
  if (type == FloatStorageType::kF16) {
#if defined(QUIXICORE_CPU_HAVE_FLOAT_STORAGE_NEON_FP16)
    if (cpu_features().fp16) return "neon_fp16";
#endif
#if defined(QUIXICORE_CPU_HAVE_FLOAT_STORAGE_F16C)
    if (cpu_features().f16c) return "f16c";
#endif
  }
  if (type == FloatStorageType::kBF16) {
#if defined(QUIXICORE_CPU_HAVE_FLOAT_STORAGE_AVX2)
    if (cpu_features().avx2) return "avx2";
#endif
  }
  return "parallel_f32";
}

Status dispatch_float_storage(const FloatStorageInput* inputs,
                              long long input_count,
                              const FloatStorageOutput* outputs,
                              long long output_count,
                              Float32StorageKernel kernel, void* context,
                              FloatStorageWorkspace* workspace) {
  if (input_count < 0 || output_count < 0) return Status::kInvalidShape;
  if (kernel == nullptr || (input_count > 0 && inputs == nullptr) ||
      (output_count > 0 && outputs == nullptr)) {
    return Status::kInvalidArgument;
  }

  if (static_cast<unsigned long long>(input_count) >
          std::numeric_limits<std::size_t>::max() ||
      static_cast<unsigned long long>(output_count) >
          std::numeric_limits<std::size_t>::max() ||
      input_count > std::numeric_limits<long long>::max() - output_count) {
    return Status::kInvalidShape;
  }
  std::vector<Region> regions;
  std::vector<Group> groups;
  std::vector<long long> input_group;
  std::vector<long long> output_group;
  try {
    input_group.resize(static_cast<std::size_t>(input_count));
    output_group.resize(static_cast<std::size_t>(output_count));
    regions.reserve(static_cast<std::size_t>(input_count + output_count));
    groups.reserve(static_cast<std::size_t>(input_count + output_count));
  } catch (const std::bad_alloc&) {
    return Status::kOutOfMemory;
  } catch (const std::length_error&) {
    return Status::kInvalidShape;
  }

  auto add_region = [&](const void* data, FloatStorageType type,
                        long long count, bool input,
                        long long* group_index) -> Status {
    std::size_t bytes = 0;
    if (!bytes_for(type, count, &bytes)) {
      return known_type(type) ? Status::kInvalidShape
                              : Status::kUnsupportedFormat;
    }
    if (count > 0 && data == nullptr) return Status::kInvalidArgument;
    const Region region{data, type, count, bytes};
    for (std::size_t i = 0; i < regions.size(); ++i) {
      if (!overlaps(region, regions[i])) continue;
      if (!same_region(region, regions[i])) return Status::kInvalidArgument;
    }
    for (std::size_t i = 0; i < groups.size(); ++i) {
      if (same_region(region, groups[i].region)) {
        groups[i].has_input |= input;
        groups[i].has_output |= !input;
        *group_index = static_cast<long long>(i);
        regions.push_back(region);
        return Status::kOk;
      }
    }
    groups.push_back(Group{region, input, !input, 0});
    *group_index = static_cast<long long>(groups.size() - 1);
    regions.push_back(region);
    return Status::kOk;
  };

  try {
    for (long long i = 0; i < input_count; ++i) {
      const Status status =
          add_region(inputs[i].data, inputs[i].type, inputs[i].count, true,
                     &input_group[static_cast<std::size_t>(i)]);
      if (status != Status::kOk) return status;
    }
    for (long long i = 0; i < output_count; ++i) {
      const Status status =
          add_region(outputs[i].data, outputs[i].type, outputs[i].count, false,
                     &output_group[static_cast<std::size_t>(i)]);
      if (status != Status::kOk) return status;
    }
  } catch (const std::bad_alloc&) {
    return Status::kOutOfMemory;
  } catch (const std::length_error&) {
    return Status::kInvalidShape;
  }

  std::size_t scratch_count = 0;
  for (auto& group : groups) {
    if (group.region.type == FloatStorageType::kF32) continue;
    const auto n = static_cast<std::size_t>(group.region.count);
    if (n > std::numeric_limits<std::size_t>::max() - scratch_count) {
      return Status::kInvalidShape;
    }
    group.offset = scratch_count;
    scratch_count += n;
  }

  // A small TLS stack keeps the common route allocation-free after warmup and
  // remains correct if a wrapped kernel itself invokes another typed kernel.
  thread_local std::vector<std::unique_ptr<FloatStorageWorkspace>> tls_arenas;
  thread_local std::size_t tls_depth = 0;
  struct DepthGuard {
    std::size_t* depth = nullptr;
    ~DepthGuard() {
      if (depth != nullptr) --*depth;
    }
  } depth_guard;
  FloatStorageWorkspace* arena_ptr = workspace;
  if (arena_ptr == nullptr) {
    try {
      if (tls_depth == tls_arenas.size()) {
        tls_arenas.push_back(std::make_unique<FloatStorageWorkspace>());
      }
    } catch (const std::bad_alloc&) {
      return Status::kOutOfMemory;
    } catch (const std::length_error&) {
      return Status::kInvalidShape;
    }
    arena_ptr = tls_arenas[tls_depth++].get();
    depth_guard.depth = &tls_depth;
  }
  FloatStorageWorkspace& arena = *arena_ptr;
  try {
    arena.scratch_.resize(scratch_count);
  } catch (const std::bad_alloc&) {
    return Status::kOutOfMemory;
  } catch (const std::length_error&) {
    return Status::kInvalidShape;
  }

  for (auto& group : groups) {
    if (!group.has_input || group.region.type == FloatStorageType::kF32 ||
        group.region.count == 0) {
      continue;
    }
    const Status status = float_storage_to_f32(
        group.region.type, group.region.data,
        arena.scratch_.data() + group.offset, group.region.count);
    if (status != Status::kOk) return status;
  }

  std::vector<const float*> f32_inputs;
  std::vector<float*> f32_outputs;
  try {
    f32_inputs.resize(static_cast<std::size_t>(input_count));
    f32_outputs.resize(static_cast<std::size_t>(output_count));
    for (long long i = 0; i < input_count; ++i) {
      const auto& group = groups[static_cast<std::size_t>(
          input_group[static_cast<std::size_t>(i)])];
      f32_inputs[static_cast<std::size_t>(i)] =
          group.region.type == FloatStorageType::kF32
              ? static_cast<const float*>(inputs[i].data)
          : group.region.count == 0 ? nullptr
                                    : arena.scratch_.data() + group.offset;
    }
    for (long long i = 0; i < output_count; ++i) {
      const auto& group = groups[static_cast<std::size_t>(
          output_group[static_cast<std::size_t>(i)])];
      f32_outputs[static_cast<std::size_t>(i)] =
          group.region.type == FloatStorageType::kF32
              ? static_cast<float*>(outputs[i].data)
          : group.region.count == 0 ? nullptr
                                    : arena.scratch_.data() + group.offset;
    }
  } catch (const std::bad_alloc&) {
    return Status::kOutOfMemory;
  } catch (const std::length_error&) {
    return Status::kInvalidShape;
  }

  const Status kernel_status =
      kernel(f32_inputs.data(), f32_outputs.data(), context);
  if (kernel_status != Status::kOk) return kernel_status;

  for (const auto& group : groups) {
    if (!group.has_output || group.region.type == FloatStorageType::kF32 ||
        group.region.count == 0) {
      continue;
    }
    const Status status = float_storage_from_f32(
        group.region.type, arena.scratch_.data() + group.offset,
        const_cast<void*>(group.region.data), group.region.count);
    if (status != Status::kOk) return status;
  }
  return Status::kOk;
}

Status unary_storage(FloatStorageInput x, FloatStorageOutput y, UnaryOp op,
                     XiEluParams xielu) {
  if (x.count != y.count) return Status::kInvalidShape;
  UnaryContext ctx{x.count, op, xielu};
  return dispatch_float_storage(&x, 1, &y, 1, call_unary, &ctx);
}

Status softmax_storage(FloatStorageInput x, FloatStorageOutput y,
                       long long rows, long long dim) {
  if (!matches_product(x.count, {rows, dim}) || y.count != x.count) {
    return Status::kInvalidShape;
  }
  SoftmaxContext ctx{rows, dim};
  return dispatch_float_storage(&x, 1, &y, 1, call_softmax, &ctx);
}

Status rms_norm_storage(FloatStorageInput x, FloatStorageInput weight,
                        FloatStorageOutput y, long long rows, long long hidden,
                        float eps) {
  if (!matches_product(x.count, {rows, hidden}) || weight.count != hidden ||
      y.count != x.count) {
    return Status::kInvalidShape;
  }
  const FloatStorageInput inputs[] = {x, weight};
  RmsContext ctx{rows, hidden, eps};
  return dispatch_float_storage(inputs, 2, &y, 1, call_rms, &ctx);
}

Status dense_gemm_storage(FloatStorageInput a, FloatStorageInput b,
                          FloatStorageOutput c, long long m, long long n,
                          long long k) {
  if (!matches_product(a.count, {m, k}) || !matches_product(b.count, {k, n}) ||
      !matches_product(c.count, {m, n})) {
    return Status::kInvalidShape;
  }
  const FloatStorageInput inputs[] = {a, b};
  GemmContext ctx{m, n, k};
  return dispatch_float_storage(inputs, 2, &c, 1, call_gemm, &ctx);
}

Status attention_storage(FloatStorageInput q, FloatStorageInput k,
                         FloatStorageInput v, FloatStorageOutput out,
                         long long query_heads, long long kv_heads,
                         long long query_length, long long kv_length,
                         long long head_dim, bool causal) {
  if (!matches_product(q.count, {query_heads, query_length, head_dim}) ||
      !matches_product(k.count, {kv_heads, kv_length, head_dim}) ||
      v.count != k.count || out.count != q.count) {
    return Status::kInvalidShape;
  }
  const FloatStorageInput inputs[] = {q, k, v};
  AttentionContext ctx{query_heads, kv_heads, query_length,
                       kv_length,   head_dim, causal};
  return dispatch_float_storage(inputs, 3, &out, 1, call_attention, &ctx);
}

Status qgemv_storage(QuantFormat format, const void* packed,
                     FloatStorageInput x, FloatStorageOutput y, long long n,
                     long long k) {
  if (n <= 0 || k <= 0 || x.count != k || y.count != n) {
    return Status::kInvalidShape;
  }
  QGemmContext ctx{format, packed, 1, n, k};
  return dispatch_float_storage(&x, 1, &y, 1, call_qgemv, &ctx);
}

Status qgemm_storage(QuantFormat format, const void* packed,
                     FloatStorageInput x, FloatStorageOutput y, long long m,
                     long long n, long long k) {
  if (!matches_product(x.count, {m, k}) || !matches_product(y.count, {m, n}) ||
      n <= 0) {
    return Status::kInvalidShape;
  }
  QGemmContext ctx{format, packed, m, n, k};
  return dispatch_float_storage(&x, 1, &y, 1, call_qgemm, &ctx);
}

}  // namespace quixicore_cpu
