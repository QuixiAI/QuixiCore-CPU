// Direct compressed-cache attention evidence. The baseline explicitly
// materializes both FP8 caches before calling the FP32 paged kernel.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness/case.h"
#include "harness/donotopt.h"
#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/quantization.h"

namespace qcb {
namespace {

using quixicore_cpu::Float8Format;
using quixicore_cpu::FloatStorageInput;
using quixicore_cpu::FloatStorageOutput;
using quixicore_cpu::FloatStorageType;
using quixicore_cpu::Status;

class CacheRng {
 public:
  float next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 17;
    state_ ^= state_ << 5;
    return static_cast<float>(state_ >> 8) * (2.0f / 16777216.0f) - 1.0f;
  }

 private:
  std::uint32_t state_ = 0xA341316Cu;
};

struct Fp8AttentionBuffers {
  std::vector<float> q, source_key, source_value, decoded_key, decoded_value;
  std::vector<float> key_scale, value_scale, target, reference;
  std::vector<std::uint8_t> key_codes, value_codes;
  std::vector<int> block_table, context_lens;
  long long cache_blocks = 0;
  long long batch = 0;
  long long query_heads = 0;
  long long kv_heads = 0;
  long long context = 0;
  long long dim = 0;
  long long page = 0;
  long long max_blocks = 0;
  Float8Format format = Float8Format::kE4M3FN;
};

void materialized_fp8_attention(Fp8AttentionBuffers& b) {
  const long long cache_rows = b.cache_blocks * b.page;
  for (long long row = 0; row < cache_rows; ++row) {
    for (long long head = 0; head < b.kv_heads; ++head) {
      const long long base = (row * b.kv_heads + head) * b.dim;
      for (long long dim = 0; dim < b.dim; ++dim) {
        b.decoded_key[base + dim] =
            quixicore_cpu::float8_decode(b.key_codes[base + dim], b.format) *
            b.key_scale[head];
        b.decoded_value[base + dim] =
            quixicore_cpu::float8_decode(b.value_codes[base + dim], b.format) *
            b.value_scale[head];
      }
    }
  }
  if (quixicore_cpu::paged_attention(
          b.q.data(), b.decoded_key.data(), b.decoded_value.data(),
          b.block_table.data(), b.context_lens.data(), b.reference.data(),
          b.cache_blocks, b.batch, b.query_heads, b.kv_heads, b.dim, b.page,
          b.max_blocks) != Status::kOk) {
    throw std::runtime_error("materialized FP8 paged attention failed");
  }
}

CaseDecl make_fp8_attention(Float8Format format, long long batch,
                            long long query_heads, long long kv_heads,
                            long long context, long long dim, long long page) {
  const long long max_blocks = (context + page - 1) / page;
  const long long cache_blocks = batch * max_blocks;
  const char* format_name = format == Float8Format::kE5M2 ? "e5m2" : "e4m3fn";
  CaseDecl decl;
  decl.kernel = "quant_cache_attention";
  decl.variant = std::string("direct_") + format_name + "_B" +
                 std::to_string(batch) + "_HQ" + std::to_string(query_heads) +
                 "_HKV" + std::to_string(kv_heads) + "_S" +
                 std::to_string(context) + "_D" + std::to_string(dim);
  decl.shape = {{"batch", batch},
                {"query_heads", query_heads},
                {"kv_heads", kv_heads},
                {"context", context},
                {"dim", dim}};
  decl.format = format_name;
  decl.notes =
      "direct online FP8 K/V decode versus full FP32 cache materialization";
  decl.make = [=]() {
    auto b = std::make_shared<Fp8AttentionBuffers>();
    b->cache_blocks = cache_blocks;
    b->batch = batch;
    b->query_heads = query_heads;
    b->kv_heads = kv_heads;
    b->context = context;
    b->dim = dim;
    b->page = page;
    b->max_blocks = max_blocks;
    b->format = format;
    const long long cache_elements = cache_blocks * page * kv_heads * dim;
    b->q.resize(static_cast<std::size_t>(batch * query_heads * dim));
    b->source_key.resize(static_cast<std::size_t>(cache_elements));
    b->source_value.resize(static_cast<std::size_t>(cache_elements));
    b->decoded_key.resize(static_cast<std::size_t>(cache_elements));
    b->decoded_value.resize(static_cast<std::size_t>(cache_elements));
    b->key_codes.resize(static_cast<std::size_t>(cache_elements));
    b->value_codes.resize(static_cast<std::size_t>(cache_elements));
    b->key_scale.resize(static_cast<std::size_t>(kv_heads));
    b->value_scale.resize(static_cast<std::size_t>(kv_heads));
    b->target.resize(b->q.size());
    b->reference.resize(b->q.size());
    b->block_table.resize(static_cast<std::size_t>(batch * max_blocks));
    b->context_lens.assign(static_cast<std::size_t>(batch),
                           static_cast<int>(context));

    CacheRng rng;
    for (float& value : b->q) value = 0.2f * rng.next();
    for (float& value : b->source_key) value = rng.next();
    for (float& value : b->source_value) value = rng.next();
    const float maximum = format == Float8Format::kE5M2 ? 57344.0f : 448.0f;
    for (long long head = 0; head < kv_heads; ++head) {
      float key_max = 0.0f;
      float value_max = 0.0f;
      for (long long row = 0; row < cache_blocks * page; ++row) {
        const long long base = (row * kv_heads + head) * dim;
        for (long long d = 0; d < dim; ++d) {
          key_max = std::max(key_max, std::fabs(b->source_key[base + d]));
          value_max = std::max(value_max, std::fabs(b->source_value[base + d]));
        }
      }
      b->key_scale[head] = key_max / maximum;
      b->value_scale[head] = value_max / maximum;
      for (long long row = 0; row < cache_blocks * page; ++row) {
        const long long base = (row * kv_heads + head) * dim;
        for (long long d = 0; d < dim; ++d) {
          b->key_codes[base + d] = quixicore_cpu::float8_encode(
              b->source_key[base + d] / b->key_scale[head], format);
          b->value_codes[base + d] = quixicore_cpu::float8_encode(
              b->source_value[base + d] / b->value_scale[head], format);
        }
      }
    }
    for (long long request = 0; request < batch; ++request) {
      for (long long block = 0; block < max_blocks; ++block) {
        b->block_table[request * max_blocks + block] =
            static_cast<int>(request * max_blocks + block);
      }
    }

    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::paged_attention_fp8(
              b->q.data(), b->key_codes.data(), b->value_codes.data(),
              b->block_table.data(), b->context_lens.data(),
              b->key_scale.data(), b->value_scale.data(), b->target.data(),
              b->cache_blocks, b->batch, b->query_heads, b->kv_heads, b->dim,
              b->page, b->max_blocks, b->format) != Status::kOk) {
        throw std::runtime_error("direct FP8 paged attention failed");
      }
      do_not_optimize(b->target.data());
    };
    body.baselines.emplace_back("materialize_f32_cache", [b]() {
      materialized_fp8_attention(*b);
      do_not_optimize(b->reference.data());
    });
    body.check = [b]() {
      if (quixicore_cpu::paged_attention_fp8(
              b->q.data(), b->key_codes.data(), b->value_codes.data(),
              b->block_table.data(), b->context_lens.data(),
              b->key_scale.data(), b->value_scale.data(), b->target.data(),
              b->cache_blocks, b->batch, b->query_heads, b->kv_heads, b->dim,
              b->page, b->max_blocks, b->format) != Status::kOk) {
        throw std::runtime_error("direct FP8 paged attention failed");
      }
      materialized_fp8_attention(*b);
      CheckResult result;
      for (std::size_t i = 0; i < b->target.size(); ++i) {
        check_value(result, b->target[i], b->reference[i],
                    Tolerance{3e-5, 3e-4});
      }
      return result;
    };
    return body;
  };
  return decl;
}

struct Fp8CacheIoBuffers {
  std::vector<float> source, staged, gathered;
  std::vector<std::uint16_t> source_16, target_16, reference_16;
  std::vector<std::uint8_t> key_codes, value_codes, reference_key_codes,
      reference_value_codes;
  std::vector<float> scales;
  std::vector<int> slots;
  long long tokens = 0;
  long long heads = 0;
  long long dim = 0;
  FloatStorageType type = FloatStorageType::kBF16;
  Float8Format format = Float8Format::kE4M3FN;
  bool scatter = true;
};

CaseDecl make_fp8_cache_io(bool scatter, FloatStorageType type,
                           Float8Format format, long long tokens,
                           long long heads, long long dim) {
  const long long elements = tokens * heads * dim;
  const char* type_name = type == FloatStorageType::kF16 ? "f16" : "bf16";
  const char* format_name = format == Float8Format::kE5M2 ? "e5m2" : "e4m3fn";
  CaseDecl decl;
  decl.kernel = "quant_cache_attention";
  decl.variant = std::string(scatter ? "scatter_" : "gather_") + type_name +
                 "_" + format_name + "_T" + std::to_string(tokens) + "_H" +
                 std::to_string(heads) + "_D" + std::to_string(dim);
  decl.shape = {{"tokens", tokens}, {"heads", heads}, {"dim", dim}};
  decl.dtype = type_name;
  decl.format = format_name;
  decl.notes = scatter ? "typed cache insert versus FP32 staging"
                       : "typed cache gather versus FP32 staging";
  decl.make = [=]() {
    auto b = std::make_shared<Fp8CacheIoBuffers>();
    b->tokens = tokens;
    b->heads = heads;
    b->dim = dim;
    b->type = type;
    b->format = format;
    b->scatter = scatter;
    b->source.resize(static_cast<std::size_t>(elements));
    b->staged.resize(b->source.size());
    b->gathered.resize(b->source.size());
    b->source_16.resize(b->source.size());
    b->target_16.resize(b->source.size());
    b->reference_16.resize(b->source.size());
    b->key_codes.resize(b->source.size());
    b->value_codes.resize(b->source.size());
    b->reference_key_codes.resize(b->source.size());
    b->reference_value_codes.resize(b->source.size());
    b->scales.resize(static_cast<std::size_t>(heads), 1.0f / 448.0f);
    b->slots.resize(static_cast<std::size_t>(tokens));
    CacheRng rng;
    for (long long i = 0; i < elements; ++i) {
      b->source[i] = rng.next();
      b->source_16[i] = type == FloatStorageType::kF16
                            ? quixicore_cpu::float_to_f16(b->source[i])
                            : quixicore_cpu::float_to_bf16(b->source[i]);
    }
    for (long long token = 0; token < tokens; ++token) {
      b->slots[token] = static_cast<int>(token);
    }
    if (quixicore_cpu::float_storage_to_f32(type, b->source_16.data(),
                                            b->staged.data(),
                                            elements) != Status::kOk ||
        quixicore_cpu::kv_cache_scatter_fp8(
            b->staged.data(), b->staged.data(), b->slots.data(),
            b->scales.data(), b->scales.data(), b->key_codes.data(),
            b->value_codes.data(), tokens, tokens, heads, dim,
            format) != Status::kOk) {
      throw std::runtime_error("FP8 cache I/O setup failed");
    }
    CaseBody body;
    if (scatter) {
      body.target = [b, elements]() {
        if (quixicore_cpu::kv_cache_scatter_fp8_storage(
                FloatStorageInput{b->source_16.data(), b->type, elements},
                FloatStorageInput{b->source_16.data(), b->type, elements},
                b->slots.data(), b->scales.data(), b->scales.data(),
                b->key_codes.data(), b->value_codes.data(), b->tokens,
                b->tokens, b->heads, b->dim, b->format) != Status::kOk) {
          throw std::runtime_error("typed FP8 scatter failed");
        }
        do_not_optimize(b->key_codes.data());
      };
      body.baselines.emplace_back("stage_f32_then_scatter", [b, elements]() {
        if (quixicore_cpu::float_storage_to_f32(b->type, b->source_16.data(),
                                                b->staged.data(),
                                                elements) != Status::kOk ||
            quixicore_cpu::kv_cache_scatter_fp8(
                b->staged.data(), b->staged.data(), b->slots.data(),
                b->scales.data(), b->scales.data(),
                b->reference_key_codes.data(), b->reference_value_codes.data(),
                b->tokens, b->tokens, b->heads, b->dim,
                b->format) != Status::kOk) {
          throw std::runtime_error("staged FP8 scatter failed");
        }
        do_not_optimize(b->reference_key_codes.data());
      });
    } else {
      body.target = [b, elements]() {
        if (quixicore_cpu::kv_cache_gather_fp8_storage(
                b->key_codes.data(), b->value_codes.data(), b->slots.data(),
                b->scales.data(), b->scales.data(),
                FloatStorageOutput{b->target_16.data(), b->type, elements},
                FloatStorageOutput{b->target_16.data(), b->type, elements},
                b->tokens, b->tokens, b->heads, b->dim,
                b->format) != Status::kOk) {
          throw std::runtime_error("typed FP8 gather failed");
        }
        do_not_optimize(b->target_16.data());
      };
      body.baselines.emplace_back("gather_f32_then_convert", [b, elements]() {
        if (quixicore_cpu::kv_cache_gather_fp8(
                b->key_codes.data(), b->value_codes.data(), b->slots.data(),
                b->scales.data(), b->scales.data(), b->gathered.data(),
                b->staged.data(), b->tokens, b->tokens, b->heads, b->dim,
                b->format) != Status::kOk ||
            quixicore_cpu::float_storage_from_f32(b->type, b->gathered.data(),
                                                  b->reference_16.data(),
                                                  elements) != Status::kOk) {
          throw std::runtime_error("staged FP8 gather failed");
        }
        do_not_optimize(b->reference_16.data());
      });
    }
    body.check = [b, elements]() {
      CheckResult result;
      if (b->scatter) {
        if (quixicore_cpu::kv_cache_scatter_fp8_storage(
                FloatStorageInput{b->source_16.data(), b->type, elements},
                FloatStorageInput{b->source_16.data(), b->type, elements},
                b->slots.data(), b->scales.data(), b->scales.data(),
                b->key_codes.data(), b->value_codes.data(), b->tokens,
                b->tokens, b->heads, b->dim, b->format) != Status::kOk ||
            quixicore_cpu::kv_cache_scatter_fp8(
                b->staged.data(), b->staged.data(), b->slots.data(),
                b->scales.data(), b->scales.data(),
                b->reference_key_codes.data(), b->reference_value_codes.data(),
                b->tokens, b->tokens, b->heads, b->dim,
                b->format) != Status::kOk) {
          throw std::runtime_error("FP8 scatter check failed");
        }
        result.passed = b->key_codes == b->reference_key_codes &&
                        b->value_codes == b->reference_value_codes;
      } else {
        if (quixicore_cpu::kv_cache_gather_fp8_storage(
                b->key_codes.data(), b->value_codes.data(), b->slots.data(),
                b->scales.data(), b->scales.data(),
                FloatStorageOutput{b->target_16.data(), b->type, elements},
                FloatStorageOutput{b->target_16.data(), b->type, elements},
                b->tokens, b->tokens, b->heads, b->dim,
                b->format) != Status::kOk ||
            quixicore_cpu::kv_cache_gather_fp8(
                b->key_codes.data(), b->value_codes.data(), b->slots.data(),
                b->scales.data(), b->scales.data(), b->gathered.data(),
                b->staged.data(), b->tokens, b->tokens, b->heads, b->dim,
                b->format) != Status::kOk ||
            quixicore_cpu::float_storage_from_f32(b->type, b->gathered.data(),
                                                  b->reference_16.data(),
                                                  elements) != Status::kOk) {
          throw std::runtime_error("FP8 gather check failed");
        }
        result.passed = b->target_16 == b->reference_16;
      }
      return result;
    };
    return body;
  };
  return decl;
}

struct MxAttentionBuffers {
  std::vector<float> q, key, value, decoded_key, decoded_value, target,
      reference;
  std::vector<std::uint8_t> key_cache, value_cache;
  std::vector<int> slots, block_table, context_lens;
  long long cache_blocks = 0;
  long long batch = 0;
  long long query_heads = 0;
  long long kv_heads = 0;
  long long context = 0;
  long long dim = 0;
  long long page = 0;
  long long max_blocks = 0;
};

void materialized_mxfp8_attention(MxAttentionBuffers& b) {
  const long long rows = b.cache_blocks * b.page;
  if (quixicore_cpu::kv_cache_gather_mxfp8(
          b.key_cache.data(), b.value_cache.data(), b.slots.data(),
          b.decoded_key.data(), b.decoded_value.data(), rows, rows, b.kv_heads,
          b.dim) != Status::kOk ||
      quixicore_cpu::paged_attention(
          b.q.data(), b.decoded_key.data(), b.decoded_value.data(),
          b.block_table.data(), b.context_lens.data(), b.reference.data(),
          b.cache_blocks, b.batch, b.query_heads, b.kv_heads, b.dim, b.page,
          b.max_blocks) != Status::kOk) {
    throw std::runtime_error("materialized MXFP8 attention failed");
  }
}

CaseDecl make_mxfp8_attention(long long batch, long long query_heads,
                              long long kv_heads, long long context,
                              long long dim, long long page) {
  const long long max_blocks = (context + page - 1) / page;
  const long long cache_blocks = batch * max_blocks;
  CaseDecl decl;
  decl.kernel = "quant_cache_attention";
  decl.variant = "direct_mxfp8_B" + std::to_string(batch) + "_HQ" +
                 std::to_string(query_heads) + "_HKV" +
                 std::to_string(kv_heads) + "_S" + std::to_string(context) +
                 "_D" + std::to_string(dim);
  decl.shape = {{"batch", batch},
                {"query_heads", query_heads},
                {"kv_heads", kv_heads},
                {"context", context},
                {"dim", dim}};
  decl.format = "mxfp8";
  decl.notes =
      "direct online MXFP8 blocks versus full FP32 cache materialization";
  decl.make = [=]() {
    auto b = std::make_shared<MxAttentionBuffers>();
    b->cache_blocks = cache_blocks;
    b->batch = batch;
    b->query_heads = query_heads;
    b->kv_heads = kv_heads;
    b->context = context;
    b->dim = dim;
    b->page = page;
    b->max_blocks = max_blocks;
    const long long rows = cache_blocks * page;
    const long long cache_elements = rows * kv_heads * dim;
    const long long cache_bytes = rows * kv_heads * (dim / 32) * 33;
    b->q.resize(static_cast<std::size_t>(batch * query_heads * dim));
    b->key.resize(static_cast<std::size_t>(cache_elements));
    b->value.resize(b->key.size());
    b->decoded_key.resize(b->key.size());
    b->decoded_value.resize(b->key.size());
    b->target.resize(b->q.size());
    b->reference.resize(b->q.size());
    b->key_cache.resize(static_cast<std::size_t>(cache_bytes));
    b->value_cache.resize(b->key_cache.size());
    b->slots.resize(static_cast<std::size_t>(rows));
    b->block_table.resize(static_cast<std::size_t>(batch * max_blocks));
    b->context_lens.assign(static_cast<std::size_t>(batch),
                           static_cast<int>(context));
    CacheRng rng;
    for (float& v : b->q) v = 0.2f * rng.next();
    for (float& v : b->key) v = rng.next();
    for (float& v : b->value) v = rng.next();
    for (long long row = 0; row < rows; ++row) {
      b->slots[row] = static_cast<int>(row);
    }
    for (long long request = 0; request < batch; ++request) {
      for (long long block = 0; block < max_blocks; ++block) {
        b->block_table[request * max_blocks + block] =
            static_cast<int>(request * max_blocks + block);
      }
    }
    if (quixicore_cpu::kv_cache_scatter_mxfp8(
            b->key.data(), b->value.data(), b->slots.data(),
            b->key_cache.data(), b->value_cache.data(), rows, rows, kv_heads,
            dim) != Status::kOk) {
      throw std::runtime_error("MXFP8 attention setup failed");
    }
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::paged_attention_mxfp8(
              b->q.data(), b->key_cache.data(), b->value_cache.data(),
              b->block_table.data(), b->context_lens.data(), b->target.data(),
              b->cache_blocks, b->batch, b->query_heads, b->kv_heads, b->dim,
              b->page, b->max_blocks) != Status::kOk) {
        throw std::runtime_error("direct MXFP8 attention failed");
      }
      do_not_optimize(b->target.data());
    };
    body.baselines.emplace_back("materialize_f32_cache", [b]() {
      materialized_mxfp8_attention(*b);
      do_not_optimize(b->reference.data());
    });
    body.check = [b]() {
      if (quixicore_cpu::paged_attention_mxfp8(
              b->q.data(), b->key_cache.data(), b->value_cache.data(),
              b->block_table.data(), b->context_lens.data(), b->target.data(),
              b->cache_blocks, b->batch, b->query_heads, b->kv_heads, b->dim,
              b->page, b->max_blocks) != Status::kOk) {
        throw std::runtime_error("direct MXFP8 attention failed");
      }
      materialized_mxfp8_attention(*b);
      CheckResult result;
      for (std::size_t i = 0; i < b->target.size(); ++i) {
        check_value(result, b->target[i], b->reference[i],
                    Tolerance{3e-5, 3e-4});
      }
      return result;
    };
    return body;
  };
  return decl;
}

struct MxCacheIoBuffers {
  std::vector<float> source, staged, gathered_key, gathered_value;
  std::vector<std::uint16_t> source_16, target_key, target_value, reference_key,
      reference_value;
  std::vector<std::uint8_t> key_cache, value_cache, reference_key_cache,
      reference_value_cache;
  std::vector<int> slots;
  long long tokens = 0;
  long long heads = 0;
  long long dim = 0;
  bool scatter = true;
};

CaseDecl make_mxfp8_cache_io(bool scatter, long long tokens, long long heads,
                             long long dim) {
  const long long elements = tokens * heads * dim;
  const long long cache_bytes = tokens * heads * (dim / 32) * 33;
  CaseDecl decl;
  decl.kernel = "quant_cache_attention";
  decl.variant = std::string(scatter ? "scatter_" : "gather_") +
                 "bf16_mxfp8_T" + std::to_string(tokens) + "_H" +
                 std::to_string(heads) + "_D" + std::to_string(dim);
  decl.shape = {{"tokens", tokens}, {"heads", heads}, {"dim", dim}};
  decl.dtype = "bf16";
  decl.format = "mxfp8";
  decl.notes = scatter ? "typed MXFP8 insert versus FP32 staging"
                       : "typed MXFP8 gather versus FP32 staging";
  decl.make = [=]() {
    auto b = std::make_shared<MxCacheIoBuffers>();
    b->tokens = tokens;
    b->heads = heads;
    b->dim = dim;
    b->scatter = scatter;
    b->source.resize(static_cast<std::size_t>(elements));
    b->staged.resize(b->source.size());
    b->gathered_key.resize(b->source.size());
    b->gathered_value.resize(b->source.size());
    b->source_16.resize(b->source.size());
    b->target_key.resize(b->source.size());
    b->target_value.resize(b->source.size());
    b->reference_key.resize(b->source.size());
    b->reference_value.resize(b->source.size());
    b->key_cache.resize(static_cast<std::size_t>(cache_bytes));
    b->value_cache.resize(b->key_cache.size());
    b->reference_key_cache.resize(b->key_cache.size());
    b->reference_value_cache.resize(b->key_cache.size());
    b->slots.resize(static_cast<std::size_t>(tokens));
    CacheRng rng;
    for (long long i = 0; i < elements; ++i) {
      b->source[i] = rng.next();
      b->source_16[i] = quixicore_cpu::float_to_bf16(b->source[i]);
    }
    for (long long token = 0; token < tokens; ++token) {
      b->slots[token] = static_cast<int>(token);
    }
    if (quixicore_cpu::float_storage_to_f32(
            FloatStorageType::kBF16, b->source_16.data(), b->staged.data(),
            elements) != Status::kOk ||
        quixicore_cpu::kv_cache_scatter_mxfp8(
            b->staged.data(), b->staged.data(), b->slots.data(),
            b->key_cache.data(), b->value_cache.data(), tokens, tokens, heads,
            dim) != Status::kOk) {
      throw std::runtime_error("MXFP8 cache I/O setup failed");
    }
    CaseBody body;
    if (scatter) {
      body.target = [b, elements]() {
        if (quixicore_cpu::kv_cache_scatter_mxfp8_storage(
                FloatStorageInput{b->source_16.data(), FloatStorageType::kBF16,
                                  elements},
                FloatStorageInput{b->source_16.data(), FloatStorageType::kBF16,
                                  elements},
                b->slots.data(), b->key_cache.data(), b->value_cache.data(),
                b->tokens, b->tokens, b->heads, b->dim) != Status::kOk) {
          throw std::runtime_error("typed MXFP8 scatter failed");
        }
        do_not_optimize(b->key_cache.data());
      };
      body.baselines.emplace_back("stage_f32_then_scatter", [b, elements]() {
        if (quixicore_cpu::float_storage_to_f32(
                FloatStorageType::kBF16, b->source_16.data(), b->staged.data(),
                elements) != Status::kOk ||
            quixicore_cpu::kv_cache_scatter_mxfp8(
                b->staged.data(), b->staged.data(), b->slots.data(),
                b->reference_key_cache.data(), b->reference_value_cache.data(),
                b->tokens, b->tokens, b->heads, b->dim) != Status::kOk) {
          throw std::runtime_error("staged MXFP8 scatter failed");
        }
        do_not_optimize(b->reference_key_cache.data());
      });
    } else {
      body.target = [b, elements]() {
        if (quixicore_cpu::kv_cache_gather_mxfp8_storage(
                b->key_cache.data(), b->value_cache.data(), b->slots.data(),
                FloatStorageOutput{b->target_key.data(),
                                   FloatStorageType::kBF16, elements},
                FloatStorageOutput{b->target_value.data(),
                                   FloatStorageType::kBF16, elements},
                b->tokens, b->tokens, b->heads, b->dim) != Status::kOk) {
          throw std::runtime_error("typed MXFP8 gather failed");
        }
        do_not_optimize(b->target_key.data());
      };
      body.baselines.emplace_back("gather_f32_then_convert", [b, elements]() {
        if (quixicore_cpu::kv_cache_gather_mxfp8(
                b->key_cache.data(), b->value_cache.data(), b->slots.data(),
                b->gathered_key.data(), b->gathered_value.data(), b->tokens,
                b->tokens, b->heads, b->dim) != Status::kOk ||
            quixicore_cpu::float_storage_from_f32(
                FloatStorageType::kBF16, b->gathered_key.data(),
                b->reference_key.data(), elements) != Status::kOk ||
            quixicore_cpu::float_storage_from_f32(
                FloatStorageType::kBF16, b->gathered_value.data(),
                b->reference_value.data(), elements) != Status::kOk) {
          throw std::runtime_error("staged MXFP8 gather failed");
        }
        do_not_optimize(b->reference_key.data());
      });
    }
    body.check = [b, elements]() {
      CheckResult result;
      if (b->scatter) {
        if (quixicore_cpu::kv_cache_scatter_mxfp8_storage(
                FloatStorageInput{b->source_16.data(), FloatStorageType::kBF16,
                                  elements},
                FloatStorageInput{b->source_16.data(), FloatStorageType::kBF16,
                                  elements},
                b->slots.data(), b->key_cache.data(), b->value_cache.data(),
                b->tokens, b->tokens, b->heads, b->dim) != Status::kOk ||
            quixicore_cpu::kv_cache_scatter_mxfp8(
                b->staged.data(), b->staged.data(), b->slots.data(),
                b->reference_key_cache.data(), b->reference_value_cache.data(),
                b->tokens, b->tokens, b->heads, b->dim) != Status::kOk) {
          throw std::runtime_error("MXFP8 scatter check failed");
        }
        result.passed = b->key_cache == b->reference_key_cache &&
                        b->value_cache == b->reference_value_cache;
      } else {
        if (quixicore_cpu::kv_cache_gather_mxfp8_storage(
                b->key_cache.data(), b->value_cache.data(), b->slots.data(),
                FloatStorageOutput{b->target_key.data(),
                                   FloatStorageType::kBF16, elements},
                FloatStorageOutput{b->target_value.data(),
                                   FloatStorageType::kBF16, elements},
                b->tokens, b->tokens, b->heads, b->dim) != Status::kOk ||
            quixicore_cpu::kv_cache_gather_mxfp8(
                b->key_cache.data(), b->value_cache.data(), b->slots.data(),
                b->gathered_key.data(), b->gathered_value.data(), b->tokens,
                b->tokens, b->heads, b->dim) != Status::kOk ||
            quixicore_cpu::float_storage_from_f32(
                FloatStorageType::kBF16, b->gathered_key.data(),
                b->reference_key.data(), elements) != Status::kOk ||
            quixicore_cpu::float_storage_from_f32(
                FloatStorageType::kBF16, b->gathered_value.data(),
                b->reference_value.data(), elements) != Status::kOk) {
          throw std::runtime_error("MXFP8 gather check failed");
        }
        result.passed = b->target_key == b->reference_key &&
                        b->target_value == b->reference_value;
      }
      return result;
    };
    return body;
  };
  return decl;
}

struct TurboAttentionBuffers {
  std::vector<float> q, key, value, decoded_key, decoded_value, target,
      reference, key_scale, value_scale, key_zero, centroids, signs;
  std::vector<std::uint8_t> key_cache, value_cache;
  std::vector<int> slots, block_table, context_lens;
  long long cache_blocks = 0;
  long long batch = 0;
  long long query_heads = 0;
  long long kv_heads = 0;
  long long context = 0;
  long long dim = 0;
  long long page = 0;
  long long max_blocks = 0;
  int key_bits = 0;
  int value_bits = 0;
};

void materialized_turboquant_attention(TurboAttentionBuffers& b) {
  const long long rows = b.cache_blocks * b.page;
  if (quixicore_cpu::turboquant_decode(
          b.key_cache.data(), b.value_cache.data(), b.key_scale.data(),
          b.value_scale.data(), b.key_zero.data(), b.slots.data(),
          b.centroids.data(), b.signs.data(), b.decoded_key.data(),
          b.decoded_value.data(), rows, rows, b.kv_heads, b.dim, b.key_bits,
          false, b.value_bits) != Status::kOk ||
      quixicore_cpu::paged_attention(
          b.q.data(), b.decoded_key.data(), b.decoded_value.data(),
          b.block_table.data(), b.context_lens.data(), b.reference.data(),
          b.cache_blocks, b.batch, b.query_heads, b.kv_heads, b.dim, b.page,
          b.max_blocks) != Status::kOk) {
    throw std::runtime_error("materialized TurboQuant attention failed");
  }
}

CaseDecl make_turboquant_attention(long long batch, long long query_heads,
                                   long long kv_heads, long long context,
                                   long long dim, long long page, int key_bits,
                                   int value_bits) {
  const long long max_blocks = (context + page - 1) / page;
  const long long cache_blocks = batch * max_blocks;
  CaseDecl decl;
  decl.kernel = "quant_cache_attention";
  decl.variant = "direct_turboquant_k" + std::to_string(key_bits) + "v" +
                 std::to_string(value_bits) + "_B" + std::to_string(batch) +
                 "_HQ" + std::to_string(query_heads) + "_HKV" +
                 std::to_string(kv_heads) + "_S" + std::to_string(context) +
                 "_D" + std::to_string(dim);
  decl.shape = {{"batch", batch},
                {"query_heads", query_heads},
                {"kv_heads", kv_heads},
                {"context", context},
                {"dim", dim},
                {"key_bits", key_bits},
                {"value_bits", value_bits}};
  decl.format = "turboquant";
  decl.notes =
      "direct packed-key and rotated-value online attention versus "
      "full FP32 cache materialization";
  decl.make = [=]() {
    auto b = std::make_shared<TurboAttentionBuffers>();
    b->cache_blocks = cache_blocks;
    b->batch = batch;
    b->query_heads = query_heads;
    b->kv_heads = kv_heads;
    b->context = context;
    b->dim = dim;
    b->page = page;
    b->max_blocks = max_blocks;
    b->key_bits = key_bits;
    b->value_bits = value_bits;
    const long long rows = cache_blocks * page;
    const long long cache_elements = rows * kv_heads * dim;
    const long long groups = dim / 32;
    const long long key_bytes = (dim * key_bits + 7) / 8;
    const long long value_bytes = (dim * value_bits + 7) / 8;
    b->q.resize(static_cast<std::size_t>(batch * query_heads * dim));
    b->key.resize(static_cast<std::size_t>(cache_elements));
    b->value.resize(b->key.size());
    b->decoded_key.resize(b->key.size());
    b->decoded_value.resize(b->key.size());
    b->target.resize(b->q.size());
    b->reference.resize(b->q.size());
    b->key_cache.resize(static_cast<std::size_t>(rows * kv_heads * key_bytes));
    b->value_cache.resize(
        static_cast<std::size_t>(rows * kv_heads * value_bytes));
    b->key_scale.resize(static_cast<std::size_t>(rows * kv_heads * groups));
    b->value_scale.resize(b->key_scale.size());
    b->key_zero.resize(b->key_scale.size());
    b->centroids.resize(static_cast<std::size_t>(1 << value_bits));
    b->signs.resize(static_cast<std::size_t>(dim));
    b->slots.resize(static_cast<std::size_t>(rows));
    b->block_table.resize(static_cast<std::size_t>(batch * max_blocks));
    b->context_lens.assign(static_cast<std::size_t>(batch),
                           static_cast<int>(context));
    CacheRng rng;
    for (float& v : b->q) v = 0.2f * rng.next();
    for (float& v : b->key) v = rng.next();
    for (float& v : b->value) v = rng.next();
    for (long long i = 0; i < dim; ++i) {
      b->signs[i] = (i * 17 + 3) % 5 < 2 ? -1.0f : 1.0f;
    }
    for (std::size_t i = 0; i < b->centroids.size(); ++i) {
      b->centroids[i] = -2.5f + 5.0f * static_cast<float>(i) /
                                    static_cast<float>(b->centroids.size() - 1);
    }
    for (long long row = 0; row < rows; ++row) {
      b->slots[row] = static_cast<int>(row);
    }
    for (long long request = 0; request < batch; ++request) {
      for (long long block = 0; block < max_blocks; ++block) {
        b->block_table[request * max_blocks + block] =
            static_cast<int>(request * max_blocks + block);
      }
    }
    if (quixicore_cpu::turboquant_encode(
            b->key.data(), b->value.data(), b->slots.data(),
            b->centroids.data(), b->signs.data(), b->key_cache.data(),
            b->value_cache.data(), b->key_scale.data(), b->value_scale.data(),
            b->key_zero.data(), rows, rows, kv_heads, dim, key_bits, false,
            value_bits) != Status::kOk) {
      throw std::runtime_error("TurboQuant attention setup failed");
    }
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::paged_attention_turboquant(
              b->q.data(), b->key_cache.data(), b->value_cache.data(),
              b->key_scale.data(), b->value_scale.data(), b->key_zero.data(),
              b->centroids.data(), b->signs.data(), b->block_table.data(),
              b->context_lens.data(), b->target.data(), b->cache_blocks,
              b->batch, b->query_heads, b->kv_heads, b->dim, b->page,
              b->max_blocks, b->key_bits, false,
              b->value_bits) != Status::kOk) {
        throw std::runtime_error("direct TurboQuant attention failed");
      }
      do_not_optimize(b->target.data());
    };
    body.baselines.emplace_back("materialize_f32_cache", [b]() {
      materialized_turboquant_attention(*b);
      do_not_optimize(b->reference.data());
    });
    body.check = [b]() {
      if (quixicore_cpu::paged_attention_turboquant(
              b->q.data(), b->key_cache.data(), b->value_cache.data(),
              b->key_scale.data(), b->value_scale.data(), b->key_zero.data(),
              b->centroids.data(), b->signs.data(), b->block_table.data(),
              b->context_lens.data(), b->target.data(), b->cache_blocks,
              b->batch, b->query_heads, b->kv_heads, b->dim, b->page,
              b->max_blocks, b->key_bits, false,
              b->value_bits) != Status::kOk) {
        throw std::runtime_error("direct TurboQuant attention check failed");
      }
      materialized_turboquant_attention(*b);
      CheckResult result;
      for (std::size_t i = 0; i < b->target.size(); ++i) {
        check_value(result, b->target[i], b->reference[i],
                    Tolerance{3e-5, 3e-4});
      }
      return result;
    };
    return body;
  };
  return decl;
}

struct TurboCodecBuffers {
  std::vector<float> source_key, source_value, staged_key, staged_value,
      decoded_key, decoded_value, key_scale, value_scale, key_zero,
      reference_key_scale, reference_value_scale, reference_key_zero, centroids,
      signs;
  std::vector<std::uint16_t> source_key_bf16, source_value_bf16,
      decoded_key_bf16, decoded_value_bf16, reference_key_bf16,
      reference_value_bf16;
  std::vector<std::uint8_t> key_cache, value_cache, reference_key_cache,
      reference_value_cache;
  std::vector<int> slots;
  long long tokens = 0;
  long long heads = 0;
  long long dim = 0;
  int key_bits = 0;
  int value_bits = 0;
  bool encode = true;
};

CaseDecl make_turboquant_codec(bool encode, long long tokens, long long heads,
                               long long dim, int key_bits, int value_bits) {
  const long long elements = tokens * heads * dim;
  const long long groups = dim / 32;
  const long long metadata_elements = tokens * heads * groups;
  const long long key_bytes = (dim * key_bits + 7) / 8;
  const long long value_bytes = (dim * value_bits + 7) / 8;
  CaseDecl decl;
  decl.kernel = "quant_cache_attention";
  decl.variant = std::string(encode ? "encode_" : "decode_") +
                 "bf16_turboquant_k" + std::to_string(key_bits) + "v" +
                 std::to_string(value_bits) + "_T" + std::to_string(tokens) +
                 "_H" + std::to_string(heads) + "_D" + std::to_string(dim);
  decl.shape = {{"tokens", tokens},
                {"heads", heads},
                {"dim", dim},
                {"key_bits", key_bits},
                {"value_bits", value_bits}};
  decl.dtype = "bf16";
  decl.format = "turboquant";
  decl.notes = encode ? "typed TurboQuant insertion versus FP32 staging"
                      : "typed TurboQuant read versus FP32 staging";
  decl.make = [=]() {
    auto b = std::make_shared<TurboCodecBuffers>();
    b->tokens = tokens;
    b->heads = heads;
    b->dim = dim;
    b->key_bits = key_bits;
    b->value_bits = value_bits;
    b->encode = encode;
    b->source_key.resize(static_cast<std::size_t>(elements));
    b->source_value.resize(b->source_key.size());
    b->staged_key.resize(b->source_key.size());
    b->staged_value.resize(b->source_value.size());
    b->decoded_key.resize(b->source_key.size());
    b->decoded_value.resize(b->source_value.size());
    b->source_key_bf16.resize(b->source_key.size());
    b->source_value_bf16.resize(b->source_value.size());
    b->decoded_key_bf16.resize(b->source_key.size());
    b->decoded_value_bf16.resize(b->source_value.size());
    b->reference_key_bf16.resize(b->source_key.size());
    b->reference_value_bf16.resize(b->source_value.size());
    b->key_cache.resize(static_cast<std::size_t>(tokens * heads * key_bytes));
    b->value_cache.resize(
        static_cast<std::size_t>(tokens * heads * value_bytes));
    b->reference_key_cache.resize(b->key_cache.size());
    b->reference_value_cache.resize(b->value_cache.size());
    b->key_scale.resize(static_cast<std::size_t>(metadata_elements));
    b->value_scale.resize(b->key_scale.size());
    b->key_zero.resize(b->key_scale.size());
    b->reference_key_scale.resize(b->key_scale.size());
    b->reference_value_scale.resize(b->key_scale.size());
    b->reference_key_zero.resize(b->key_scale.size());
    b->centroids.resize(static_cast<std::size_t>(1 << value_bits));
    b->signs.resize(static_cast<std::size_t>(dim));
    b->slots.resize(static_cast<std::size_t>(tokens));
    CacheRng rng;
    for (long long i = 0; i < elements; ++i) {
      b->source_key[i] = rng.next();
      b->source_value[i] = rng.next();
      b->source_key_bf16[i] = quixicore_cpu::float_to_bf16(b->source_key[i]);
      b->source_value_bf16[i] =
          quixicore_cpu::float_to_bf16(b->source_value[i]);
    }
    for (long long i = 0; i < dim; ++i) {
      b->signs[i] = (i * 17 + 3) % 5 < 2 ? -1.0f : 1.0f;
    }
    for (std::size_t i = 0; i < b->centroids.size(); ++i) {
      b->centroids[i] = -2.5f + 5.0f * static_cast<float>(i) /
                                    static_cast<float>(b->centroids.size() - 1);
    }
    for (long long token = 0; token < tokens; ++token) {
      b->slots[token] = static_cast<int>(token);
    }
    if (quixicore_cpu::float_storage_to_f32(
            FloatStorageType::kBF16, b->source_key_bf16.data(),
            b->staged_key.data(), elements) != Status::kOk ||
        quixicore_cpu::float_storage_to_f32(
            FloatStorageType::kBF16, b->source_value_bf16.data(),
            b->staged_value.data(), elements) != Status::kOk ||
        quixicore_cpu::turboquant_encode(
            b->staged_key.data(), b->staged_value.data(), b->slots.data(),
            b->centroids.data(), b->signs.data(), b->key_cache.data(),
            b->value_cache.data(), b->key_scale.data(), b->value_scale.data(),
            b->key_zero.data(), tokens, tokens, heads, dim, key_bits, false,
            value_bits) != Status::kOk) {
      throw std::runtime_error("TurboQuant codec setup failed");
    }
    CaseBody body;
    if (encode) {
      body.target = [b, elements]() {
        if (quixicore_cpu::turboquant_encode_storage(
                FloatStorageInput{b->source_key_bf16.data(),
                                  FloatStorageType::kBF16, elements},
                FloatStorageInput{b->source_value_bf16.data(),
                                  FloatStorageType::kBF16, elements},
                b->slots.data(), b->centroids.data(), b->signs.data(),
                b->key_cache.data(), b->value_cache.data(), b->key_scale.data(),
                b->value_scale.data(), b->key_zero.data(), b->tokens, b->tokens,
                b->heads, b->dim, b->key_bits, false,
                b->value_bits) != Status::kOk) {
          throw std::runtime_error("typed TurboQuant encode failed");
        }
        do_not_optimize(b->key_cache.data());
      };
      body.baselines.emplace_back("stage_f32_then_encode", [b, elements]() {
        if (quixicore_cpu::float_storage_to_f32(
                FloatStorageType::kBF16, b->source_key_bf16.data(),
                b->staged_key.data(), elements) != Status::kOk ||
            quixicore_cpu::float_storage_to_f32(
                FloatStorageType::kBF16, b->source_value_bf16.data(),
                b->staged_value.data(), elements) != Status::kOk ||
            quixicore_cpu::turboquant_encode(
                b->staged_key.data(), b->staged_value.data(), b->slots.data(),
                b->centroids.data(), b->signs.data(),
                b->reference_key_cache.data(), b->reference_value_cache.data(),
                b->reference_key_scale.data(), b->reference_value_scale.data(),
                b->reference_key_zero.data(), b->tokens, b->tokens, b->heads,
                b->dim, b->key_bits, false, b->value_bits) != Status::kOk) {
          throw std::runtime_error("staged TurboQuant encode failed");
        }
        do_not_optimize(b->reference_key_cache.data());
      });
    } else {
      body.target = [b, elements]() {
        if (quixicore_cpu::turboquant_decode_storage(
                b->key_cache.data(), b->value_cache.data(), b->key_scale.data(),
                b->value_scale.data(), b->key_zero.data(), b->slots.data(),
                b->centroids.data(), b->signs.data(),
                FloatStorageOutput{b->decoded_key_bf16.data(),
                                   FloatStorageType::kBF16, elements},
                FloatStorageOutput{b->decoded_value_bf16.data(),
                                   FloatStorageType::kBF16, elements},
                b->tokens, b->tokens, b->heads, b->dim, b->key_bits, false,
                b->value_bits) != Status::kOk) {
          throw std::runtime_error("typed TurboQuant decode failed");
        }
        do_not_optimize(b->decoded_key_bf16.data());
      };
      body.baselines.emplace_back("decode_f32_then_convert", [b, elements]() {
        if (quixicore_cpu::turboquant_decode(
                b->key_cache.data(), b->value_cache.data(), b->key_scale.data(),
                b->value_scale.data(), b->key_zero.data(), b->slots.data(),
                b->centroids.data(), b->signs.data(), b->decoded_key.data(),
                b->decoded_value.data(), b->tokens, b->tokens, b->heads, b->dim,
                b->key_bits, false, b->value_bits) != Status::kOk ||
            quixicore_cpu::float_storage_from_f32(
                FloatStorageType::kBF16, b->decoded_key.data(),
                b->reference_key_bf16.data(), elements) != Status::kOk ||
            quixicore_cpu::float_storage_from_f32(
                FloatStorageType::kBF16, b->decoded_value.data(),
                b->reference_value_bf16.data(), elements) != Status::kOk) {
          throw std::runtime_error("staged TurboQuant decode failed");
        }
        do_not_optimize(b->reference_key_bf16.data());
      });
    }
    body.check = [b, elements]() {
      CheckResult result;
      if (b->encode) {
        if (quixicore_cpu::turboquant_encode_storage(
                FloatStorageInput{b->source_key_bf16.data(),
                                  FloatStorageType::kBF16, elements},
                FloatStorageInput{b->source_value_bf16.data(),
                                  FloatStorageType::kBF16, elements},
                b->slots.data(), b->centroids.data(), b->signs.data(),
                b->key_cache.data(), b->value_cache.data(), b->key_scale.data(),
                b->value_scale.data(), b->key_zero.data(), b->tokens, b->tokens,
                b->heads, b->dim, b->key_bits, false,
                b->value_bits) != Status::kOk ||
            quixicore_cpu::turboquant_encode(
                b->staged_key.data(), b->staged_value.data(), b->slots.data(),
                b->centroids.data(), b->signs.data(),
                b->reference_key_cache.data(), b->reference_value_cache.data(),
                b->reference_key_scale.data(), b->reference_value_scale.data(),
                b->reference_key_zero.data(), b->tokens, b->tokens, b->heads,
                b->dim, b->key_bits, false, b->value_bits) != Status::kOk) {
          throw std::runtime_error("TurboQuant encode check failed");
        }
        result.passed = b->key_cache == b->reference_key_cache &&
                        b->value_cache == b->reference_value_cache &&
                        b->key_scale == b->reference_key_scale &&
                        b->value_scale == b->reference_value_scale &&
                        b->key_zero == b->reference_key_zero;
      } else {
        if (quixicore_cpu::turboquant_decode_storage(
                b->key_cache.data(), b->value_cache.data(), b->key_scale.data(),
                b->value_scale.data(), b->key_zero.data(), b->slots.data(),
                b->centroids.data(), b->signs.data(),
                FloatStorageOutput{b->decoded_key_bf16.data(),
                                   FloatStorageType::kBF16, elements},
                FloatStorageOutput{b->decoded_value_bf16.data(),
                                   FloatStorageType::kBF16, elements},
                b->tokens, b->tokens, b->heads, b->dim, b->key_bits, false,
                b->value_bits) != Status::kOk ||
            quixicore_cpu::turboquant_decode(
                b->key_cache.data(), b->value_cache.data(), b->key_scale.data(),
                b->value_scale.data(), b->key_zero.data(), b->slots.data(),
                b->centroids.data(), b->signs.data(), b->decoded_key.data(),
                b->decoded_value.data(), b->tokens, b->tokens, b->heads, b->dim,
                b->key_bits, false, b->value_bits) != Status::kOk ||
            quixicore_cpu::float_storage_from_f32(
                FloatStorageType::kBF16, b->decoded_key.data(),
                b->reference_key_bf16.data(), elements) != Status::kOk ||
            quixicore_cpu::float_storage_from_f32(
                FloatStorageType::kBF16, b->decoded_value.data(),
                b->reference_value_bf16.data(), elements) != Status::kOk) {
          throw std::runtime_error("TurboQuant decode check failed");
        }
        result.passed = b->decoded_key_bf16 == b->reference_key_bf16 &&
                        b->decoded_value_bf16 == b->reference_value_bf16;
      }
      return result;
    };
    return body;
  };
  return decl;
}

struct Kv3AttentionBuffers {
  std::vector<float> q, key, value, decoded_key, decoded_value, target,
      reference, key_scale_f32, value_scale_f32;
  std::vector<std::uint16_t> key_scale_f16, value_scale_f16;
  std::vector<std::uint8_t> key_cache, value_cache;
  std::vector<int> key_zero, value_zero, slots, block_table, context_lens;
  quixicore_cpu::BitNetKv3Config config;
  long long cache_blocks = 0;
  long long batch = 0;
  long long query_heads = 0;
  long long kv_heads = 0;
  long long context = 0;
  long long dim = 0;
  long long page = 0;
  long long max_blocks = 0;

  void* key_scales() {
    return config.scale_type == quixicore_cpu::BitNetKv3ScaleType::kFP32
               ? static_cast<void*>(key_scale_f32.data())
               : static_cast<void*>(key_scale_f16.data());
  }
  void* value_scales() {
    return config.scale_type == quixicore_cpu::BitNetKv3ScaleType::kFP32
               ? static_cast<void*>(value_scale_f32.data())
               : static_cast<void*>(value_scale_f16.data());
  }
  int* key_zeros() {
    return config.zero_point_mode ==
                   quixicore_cpu::BitNetKv3ZeroPointMode::kInteger
               ? key_zero.data()
               : nullptr;
  }
  int* value_zeros() {
    return config.zero_point_mode ==
                   quixicore_cpu::BitNetKv3ZeroPointMode::kInteger
               ? value_zero.data()
               : nullptr;
  }
};

void materialized_kv3_attention(Kv3AttentionBuffers& b) {
  const long long rows = b.cache_blocks * b.page;
  if (quixicore_cpu::kv_cache_gather_bitnet_kv3(
          b.key_cache.data(), b.value_cache.data(), b.slots.data(),
          b.key_scales(), b.value_scales(), b.key_zeros(), b.value_zeros(),
          b.decoded_key.data(), b.decoded_value.data(), rows, rows, b.kv_heads,
          b.dim, b.config) != Status::kOk ||
      quixicore_cpu::paged_attention(
          b.q.data(), b.decoded_key.data(), b.decoded_value.data(),
          b.block_table.data(), b.context_lens.data(), b.reference.data(),
          b.cache_blocks, b.batch, b.query_heads, b.kv_heads, b.dim, b.page,
          b.max_blocks) != Status::kOk) {
    throw std::runtime_error("materialized BitNet KV3 attention failed");
  }
}

CaseDecl make_kv3_attention(long long batch, long long query_heads,
                            long long kv_heads, long long context,
                            long long dim, long long page,
                            quixicore_cpu::BitNetKv3Config config) {
  const long long max_blocks = (context + page - 1) / page;
  const long long cache_blocks = batch * max_blocks;
  const bool affine =
      config.zero_point_mode == quixicore_cpu::BitNetKv3ZeroPointMode::kInteger;
  const bool signed_codes =
      config.signedness == quixicore_cpu::BitNetKv3Signedness::kSigned;
  const bool fp16_scale =
      config.scale_type == quixicore_cpu::BitNetKv3ScaleType::kFP16;
  CaseDecl decl;
  decl.kernel = "quant_cache_attention";
  decl.variant = std::string("direct_bitnet_kv3_") +
                 (signed_codes ? "signed_" : "unsigned_") +
                 (affine ? "affine_" : "symmetric_") +
                 (fp16_scale ? "f16scale" : "f32scale") + "_B" +
                 std::to_string(batch) + "_HQ" + std::to_string(query_heads) +
                 "_HKV" + std::to_string(kv_heads) + "_S" +
                 std::to_string(context) + "_D" + std::to_string(dim) + "_G" +
                 std::to_string(config.group_size);
  decl.shape = {{"batch", batch},       {"query_heads", query_heads},
                {"kv_heads", kv_heads}, {"context", context},
                {"dim", dim},           {"group", config.group_size}};
  decl.format = "bitnet_kv3";
  decl.notes =
      "direct low-bit-first KV3 online attention versus full FP32 "
      "cache materialization";
  decl.make = [=]() {
    auto b = std::make_shared<Kv3AttentionBuffers>();
    b->config = config;
    b->cache_blocks = cache_blocks;
    b->batch = batch;
    b->query_heads = query_heads;
    b->kv_heads = kv_heads;
    b->context = context;
    b->dim = dim;
    b->page = page;
    b->max_blocks = max_blocks;
    const long long rows = cache_blocks * page;
    const long long cache_elements = rows * kv_heads * dim;
    const long long metadata = rows * kv_heads * (dim / config.group_size);
    const long long bytes = rows * kv_heads * ((dim * 3 + 7) / 8);
    b->q.resize(static_cast<std::size_t>(batch * query_heads * dim));
    b->key.resize(static_cast<std::size_t>(cache_elements));
    b->value.resize(b->key.size());
    b->decoded_key.resize(b->key.size());
    b->decoded_value.resize(b->key.size());
    b->target.resize(b->q.size());
    b->reference.resize(b->q.size());
    b->key_cache.resize(static_cast<std::size_t>(bytes));
    b->value_cache.resize(b->key_cache.size());
    b->key_scale_f32.resize(static_cast<std::size_t>(metadata));
    b->value_scale_f32.resize(b->key_scale_f32.size());
    b->key_scale_f16.resize(b->key_scale_f32.size());
    b->value_scale_f16.resize(b->key_scale_f32.size());
    b->key_zero.resize(b->key_scale_f32.size());
    b->value_zero.resize(b->key_scale_f32.size());
    b->slots.resize(static_cast<std::size_t>(rows));
    b->block_table.resize(static_cast<std::size_t>(batch * max_blocks));
    b->context_lens.assign(static_cast<std::size_t>(batch),
                           static_cast<int>(context));
    CacheRng rng;
    for (float& v : b->q) v = 0.2f * rng.next();
    for (float& v : b->key) v = rng.next();
    for (float& v : b->value) v = rng.next();
    if (!signed_codes && !affine) {
      for (float& v : b->key) v = std::fabs(v);
      for (float& v : b->value) v = std::fabs(v);
    }
    for (long long row = 0; row < rows; ++row) {
      b->slots[row] = static_cast<int>(row);
    }
    for (long long request = 0; request < batch; ++request) {
      for (long long block = 0; block < max_blocks; ++block) {
        b->block_table[request * max_blocks + block] =
            static_cast<int>(request * max_blocks + block);
      }
    }
    if (quixicore_cpu::kv_cache_scatter_bitnet_kv3(
            b->key.data(), b->value.data(), b->slots.data(),
            b->key_cache.data(), b->value_cache.data(), b->key_scales(),
            b->value_scales(), b->key_zeros(), b->value_zeros(), rows, rows,
            kv_heads, dim, config) != Status::kOk) {
      throw std::runtime_error("BitNet KV3 attention setup failed");
    }
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::paged_attention_bitnet_kv3(
              b->q.data(), b->key_cache.data(), b->value_cache.data(),
              b->key_scales(), b->value_scales(), b->key_zeros(),
              b->value_zeros(), b->block_table.data(), b->context_lens.data(),
              b->target.data(), b->cache_blocks, b->batch, b->query_heads,
              b->kv_heads, b->dim, b->page, b->max_blocks,
              b->config) != Status::kOk) {
        throw std::runtime_error("direct BitNet KV3 attention failed");
      }
      do_not_optimize(b->target.data());
    };
    body.baselines.emplace_back("materialize_f32_cache", [b]() {
      materialized_kv3_attention(*b);
      do_not_optimize(b->reference.data());
    });
    body.check = [b]() {
      if (quixicore_cpu::paged_attention_bitnet_kv3(
              b->q.data(), b->key_cache.data(), b->value_cache.data(),
              b->key_scales(), b->value_scales(), b->key_zeros(),
              b->value_zeros(), b->block_table.data(), b->context_lens.data(),
              b->target.data(), b->cache_blocks, b->batch, b->query_heads,
              b->kv_heads, b->dim, b->page, b->max_blocks,
              b->config) != Status::kOk) {
        throw std::runtime_error("direct BitNet KV3 attention check failed");
      }
      materialized_kv3_attention(*b);
      CheckResult result;
      for (std::size_t i = 0; i < b->target.size(); ++i) {
        check_value(result, b->target[i], b->reference[i],
                    Tolerance{3e-5, 3e-4});
      }
      return result;
    };
    return body;
  };
  return decl;
}

struct Kv3CodecBuffers {
  std::vector<float> source_key, source_value, staged_key, staged_value,
      decoded_key, decoded_value;
  std::vector<std::uint16_t> source_key_bf16, source_value_bf16,
      output_key_bf16, output_value_bf16, reference_key_bf16,
      reference_value_bf16, key_scale, value_scale, reference_key_scale,
      reference_value_scale;
  std::vector<std::uint8_t> key_cache, value_cache, reference_key_cache,
      reference_value_cache;
  std::vector<int> slots;
  quixicore_cpu::BitNetKv3Config config;
  long long tokens = 0;
  long long heads = 0;
  long long dim = 0;
  bool scatter = true;
};

CaseDecl make_kv3_codec(bool scatter, long long tokens, long long heads,
                        long long dim) {
  const long long elements = tokens * heads * dim;
  const long long metadata = tokens * heads * (dim / 32);
  const long long bytes = tokens * heads * ((dim * 3 + 7) / 8);
  CaseDecl decl;
  decl.kernel = "quant_cache_attention";
  decl.variant = std::string(scatter ? "scatter_" : "gather_") +
                 "bf16_bitnet_kv3_T" + std::to_string(tokens) + "_H" +
                 std::to_string(heads) + "_D" + std::to_string(dim);
  decl.shape = {{"tokens", tokens}, {"heads", heads}, {"dim", dim}};
  decl.dtype = "bf16";
  decl.format = "bitnet_kv3";
  decl.notes = scatter ? "typed KV3 insertion versus FP32 staging"
                       : "typed KV3 gather versus FP32 staging";
  decl.make = [=]() {
    auto b = std::make_shared<Kv3CodecBuffers>();
    b->config = {32, quixicore_cpu::BitNetKv3ScaleType::kFP16,
                 quixicore_cpu::BitNetKv3Signedness::kSigned,
                 quixicore_cpu::BitNetKv3ZeroPointMode::kNone};
    b->tokens = tokens;
    b->heads = heads;
    b->dim = dim;
    b->scatter = scatter;
    b->source_key.resize(static_cast<std::size_t>(elements));
    b->source_value.resize(b->source_key.size());
    b->staged_key.resize(b->source_key.size());
    b->staged_value.resize(b->source_value.size());
    b->decoded_key.resize(b->source_key.size());
    b->decoded_value.resize(b->source_value.size());
    b->source_key_bf16.resize(b->source_key.size());
    b->source_value_bf16.resize(b->source_value.size());
    b->output_key_bf16.resize(b->source_key.size());
    b->output_value_bf16.resize(b->source_value.size());
    b->reference_key_bf16.resize(b->source_key.size());
    b->reference_value_bf16.resize(b->source_value.size());
    b->key_cache.resize(static_cast<std::size_t>(bytes));
    b->value_cache.resize(b->key_cache.size());
    b->reference_key_cache.resize(b->key_cache.size());
    b->reference_value_cache.resize(b->value_cache.size());
    b->key_scale.resize(static_cast<std::size_t>(metadata));
    b->value_scale.resize(b->key_scale.size());
    b->reference_key_scale.resize(b->key_scale.size());
    b->reference_value_scale.resize(b->value_scale.size());
    b->slots.resize(static_cast<std::size_t>(tokens));
    CacheRng rng;
    for (long long i = 0; i < elements; ++i) {
      b->source_key[i] = rng.next();
      b->source_value[i] = rng.next();
      b->source_key_bf16[i] = quixicore_cpu::float_to_bf16(b->source_key[i]);
      b->source_value_bf16[i] =
          quixicore_cpu::float_to_bf16(b->source_value[i]);
    }
    for (long long token = 0; token < tokens; ++token) {
      b->slots[token] = static_cast<int>(token);
    }
    if (quixicore_cpu::float_storage_to_f32(
            FloatStorageType::kBF16, b->source_key_bf16.data(),
            b->staged_key.data(), elements) != Status::kOk ||
        quixicore_cpu::float_storage_to_f32(
            FloatStorageType::kBF16, b->source_value_bf16.data(),
            b->staged_value.data(), elements) != Status::kOk ||
        quixicore_cpu::kv_cache_scatter_bitnet_kv3(
            b->staged_key.data(), b->staged_value.data(), b->slots.data(),
            b->key_cache.data(), b->value_cache.data(), b->key_scale.data(),
            b->value_scale.data(), nullptr, nullptr, tokens, tokens, heads, dim,
            b->config) != Status::kOk) {
      throw std::runtime_error("BitNet KV3 codec setup failed");
    }
    CaseBody body;
    if (scatter) {
      body.target = [b, elements]() {
        if (quixicore_cpu::kv_cache_scatter_bitnet_kv3_storage(
                FloatStorageInput{b->source_key_bf16.data(),
                                  FloatStorageType::kBF16, elements},
                FloatStorageInput{b->source_value_bf16.data(),
                                  FloatStorageType::kBF16, elements},
                b->slots.data(), b->key_cache.data(), b->value_cache.data(),
                b->key_scale.data(), b->value_scale.data(), nullptr, nullptr,
                b->tokens, b->tokens, b->heads, b->dim,
                b->config) != Status::kOk) {
          throw std::runtime_error("typed BitNet KV3 scatter failed");
        }
        do_not_optimize(b->key_cache.data());
      };
      body.baselines.emplace_back("stage_f32_then_scatter", [b, elements]() {
        if (quixicore_cpu::float_storage_to_f32(
                FloatStorageType::kBF16, b->source_key_bf16.data(),
                b->staged_key.data(), elements) != Status::kOk ||
            quixicore_cpu::float_storage_to_f32(
                FloatStorageType::kBF16, b->source_value_bf16.data(),
                b->staged_value.data(), elements) != Status::kOk ||
            quixicore_cpu::kv_cache_scatter_bitnet_kv3(
                b->staged_key.data(), b->staged_value.data(), b->slots.data(),
                b->reference_key_cache.data(), b->reference_value_cache.data(),
                b->reference_key_scale.data(), b->reference_value_scale.data(),
                nullptr, nullptr, b->tokens, b->tokens, b->heads, b->dim,
                b->config) != Status::kOk) {
          throw std::runtime_error("staged BitNet KV3 scatter failed");
        }
        do_not_optimize(b->reference_key_cache.data());
      });
    } else {
      body.target = [b, elements]() {
        if (quixicore_cpu::kv_cache_gather_bitnet_kv3_storage(
                b->key_cache.data(), b->value_cache.data(), b->slots.data(),
                b->key_scale.data(), b->value_scale.data(), nullptr, nullptr,
                FloatStorageOutput{b->output_key_bf16.data(),
                                   FloatStorageType::kBF16, elements},
                FloatStorageOutput{b->output_value_bf16.data(),
                                   FloatStorageType::kBF16, elements},
                b->tokens, b->tokens, b->heads, b->dim,
                b->config) != Status::kOk) {
          throw std::runtime_error("typed BitNet KV3 gather failed");
        }
        do_not_optimize(b->output_key_bf16.data());
      };
      body.baselines.emplace_back("gather_f32_then_convert", [b, elements]() {
        if (quixicore_cpu::kv_cache_gather_bitnet_kv3(
                b->key_cache.data(), b->value_cache.data(), b->slots.data(),
                b->key_scale.data(), b->value_scale.data(), nullptr, nullptr,
                b->decoded_key.data(), b->decoded_value.data(), b->tokens,
                b->tokens, b->heads, b->dim, b->config) != Status::kOk ||
            quixicore_cpu::float_storage_from_f32(
                FloatStorageType::kBF16, b->decoded_key.data(),
                b->reference_key_bf16.data(), elements) != Status::kOk ||
            quixicore_cpu::float_storage_from_f32(
                FloatStorageType::kBF16, b->decoded_value.data(),
                b->reference_value_bf16.data(), elements) != Status::kOk) {
          throw std::runtime_error("staged BitNet KV3 gather failed");
        }
        do_not_optimize(b->reference_key_bf16.data());
      });
    }
    body.check = [b, elements]() {
      CheckResult result;
      if (b->scatter) {
        if (quixicore_cpu::kv_cache_scatter_bitnet_kv3_storage(
                FloatStorageInput{b->source_key_bf16.data(),
                                  FloatStorageType::kBF16, elements},
                FloatStorageInput{b->source_value_bf16.data(),
                                  FloatStorageType::kBF16, elements},
                b->slots.data(), b->key_cache.data(), b->value_cache.data(),
                b->key_scale.data(), b->value_scale.data(), nullptr, nullptr,
                b->tokens, b->tokens, b->heads, b->dim,
                b->config) != Status::kOk ||
            quixicore_cpu::kv_cache_scatter_bitnet_kv3(
                b->staged_key.data(), b->staged_value.data(), b->slots.data(),
                b->reference_key_cache.data(), b->reference_value_cache.data(),
                b->reference_key_scale.data(), b->reference_value_scale.data(),
                nullptr, nullptr, b->tokens, b->tokens, b->heads, b->dim,
                b->config) != Status::kOk) {
          throw std::runtime_error("BitNet KV3 scatter check failed");
        }
        result.passed = b->key_cache == b->reference_key_cache &&
                        b->value_cache == b->reference_value_cache &&
                        b->key_scale == b->reference_key_scale &&
                        b->value_scale == b->reference_value_scale;
      } else {
        if (quixicore_cpu::kv_cache_gather_bitnet_kv3_storage(
                b->key_cache.data(), b->value_cache.data(), b->slots.data(),
                b->key_scale.data(), b->value_scale.data(), nullptr, nullptr,
                FloatStorageOutput{b->output_key_bf16.data(),
                                   FloatStorageType::kBF16, elements},
                FloatStorageOutput{b->output_value_bf16.data(),
                                   FloatStorageType::kBF16, elements},
                b->tokens, b->tokens, b->heads, b->dim,
                b->config) != Status::kOk ||
            quixicore_cpu::kv_cache_gather_bitnet_kv3(
                b->key_cache.data(), b->value_cache.data(), b->slots.data(),
                b->key_scale.data(), b->value_scale.data(), nullptr, nullptr,
                b->decoded_key.data(), b->decoded_value.data(), b->tokens,
                b->tokens, b->heads, b->dim, b->config) != Status::kOk ||
            quixicore_cpu::float_storage_from_f32(
                FloatStorageType::kBF16, b->decoded_key.data(),
                b->reference_key_bf16.data(), elements) != Status::kOk ||
            quixicore_cpu::float_storage_from_f32(
                FloatStorageType::kBF16, b->decoded_value.data(),
                b->reference_value_bf16.data(), elements) != Status::kOk) {
          throw std::runtime_error("BitNet KV3 gather check failed");
        }
        result.passed = b->output_key_bf16 == b->reference_key_bf16 &&
                        b->output_value_bf16 == b->reference_value_bf16;
      }
      return result;
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_quant_cache_attention_cases(const BuildCtx& ctx,
                                       std::vector<CaseDecl>& out) {
  if (ctx.preset == Preset::kSmoke) {
    out.push_back(
        make_fp8_attention(Float8Format::kE4M3FN, 1, 2, 1, 32, 64, 16));
    out.push_back(make_fp8_attention(Float8Format::kE5M2, 1, 2, 1, 32, 64, 16));
    out.push_back(make_fp8_cache_io(true, FloatStorageType::kBF16,
                                    Float8Format::kE4M3FN, 8, 2, 64));
    out.push_back(make_fp8_cache_io(false, FloatStorageType::kBF16,
                                    Float8Format::kE4M3FN, 8, 2, 64));
    out.push_back(make_mxfp8_attention(1, 2, 1, 32, 64, 16));
    out.push_back(make_mxfp8_cache_io(true, 8, 2, 64));
    out.push_back(make_mxfp8_cache_io(false, 8, 2, 64));
    out.push_back(make_turboquant_attention(1, 2, 1, 32, 64, 16, 4, 4));
    out.push_back(make_turboquant_codec(true, 8, 2, 64, 4, 4));
    out.push_back(make_turboquant_codec(false, 8, 2, 64, 4, 4));
    out.push_back(
        make_kv3_attention(1, 2, 1, 32, 64, 16,
                           {32, quixicore_cpu::BitNetKv3ScaleType::kFP16,
                            quixicore_cpu::BitNetKv3Signedness::kSigned,
                            quixicore_cpu::BitNetKv3ZeroPointMode::kNone}));
    out.push_back(make_kv3_codec(true, 8, 2, 64));
    out.push_back(make_kv3_codec(false, 8, 2, 64));
    return;
  }
  out.push_back(
      make_fp8_attention(Float8Format::kE4M3FN, 2, 8, 2, 512, 64, 16));
  out.push_back(make_fp8_attention(Float8Format::kE5M2, 2, 8, 2, 512, 64, 16));
  out.push_back(
      make_fp8_attention(Float8Format::kE4M3FN, 2, 8, 2, 512, 128, 16));
  out.push_back(make_fp8_attention(Float8Format::kE5M2, 2, 8, 2, 512, 128, 16));
  for (FloatStorageType type :
       {FloatStorageType::kF16, FloatStorageType::kBF16}) {
    out.push_back(
        make_fp8_cache_io(true, type, Float8Format::kE4M3FN, 512, 8, 128));
    out.push_back(
        make_fp8_cache_io(false, type, Float8Format::kE4M3FN, 512, 8, 128));
  }
  out.push_back(make_mxfp8_attention(2, 8, 2, 512, 64, 16));
  out.push_back(make_mxfp8_attention(2, 8, 2, 512, 128, 16));
  out.push_back(make_mxfp8_cache_io(true, 512, 8, 128));
  out.push_back(make_mxfp8_cache_io(false, 512, 8, 128));
  out.push_back(make_turboquant_attention(2, 8, 2, 512, 64, 16, 2, 2));
  out.push_back(make_turboquant_attention(2, 8, 2, 512, 128, 16, 4, 4));
  out.push_back(make_turboquant_attention(2, 8, 2, 512, 128, 16, 8, 8));
  out.push_back(make_turboquant_codec(true, 512, 8, 128, 4, 4));
  out.push_back(make_turboquant_codec(false, 512, 8, 128, 4, 4));
  out.push_back(
      make_kv3_attention(2, 8, 2, 512, 64, 16,
                         {32, quixicore_cpu::BitNetKv3ScaleType::kFP16,
                          quixicore_cpu::BitNetKv3Signedness::kSigned,
                          quixicore_cpu::BitNetKv3ZeroPointMode::kNone}));
  out.push_back(
      make_kv3_attention(2, 8, 2, 512, 128, 16,
                         {32, quixicore_cpu::BitNetKv3ScaleType::kFP32,
                          quixicore_cpu::BitNetKv3Signedness::kUnsigned,
                          quixicore_cpu::BitNetKv3ZeroPointMode::kInteger}));
  out.push_back(make_kv3_codec(true, 512, 8, 128));
  out.push_back(make_kv3_codec(false, 512, 8, 128));
  if (ctx.preset == Preset::kComprehensive) {
    out.push_back(
        make_fp8_attention(Float8Format::kE4M3FN, 4, 32, 8, 4096, 128, 16));
    out.push_back(
        make_fp8_attention(Float8Format::kE5M2, 4, 32, 8, 4096, 128, 16));
    out.push_back(make_turboquant_attention(4, 32, 8, 4096, 256, 16, 4, 4));
    out.push_back(
        make_kv3_attention(4, 32, 8, 4096, 256, 16,
                           {64, quixicore_cpu::BitNetKv3ScaleType::kFP16,
                            quixicore_cpu::BitNetKv3Signedness::kSigned,
                            quixicore_cpu::BitNetKv3ZeroPointMode::kInteger}));
  }
}

}  // namespace qcb
