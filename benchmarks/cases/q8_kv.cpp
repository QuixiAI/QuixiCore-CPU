// Focused evidence for Metal-compatible Q8_0 KV-cache codecs, functional block
// copies, and direct compressed-cache paged attention.

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

namespace qcb {
namespace {

using quixicore_cpu::Status;

class Q8Rng {
 public:
  float next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 17;
    state_ ^= state_ << 5;
    return static_cast<float>(state_ >> 8) * (2.0f / 16777216.0f) - 1.0f;
  }

 private:
  std::uint32_t state_ = 0x510E527Fu;
};

std::int8_t encode_q8(float value, float inverse) {
  const float rounded =
      std::copysign(std::floor(std::fabs(value * inverse) + 0.5f), value);
  return static_cast<std::int8_t>(std::clamp(rounded, -127.0f, 127.0f));
}

void scatter_oracle(const std::vector<float>& key,
                    const std::vector<float>& value,
                    const std::vector<int>& slots,
                    std::vector<std::int8_t>& key_codes,
                    std::vector<std::uint16_t>& key_scales,
                    std::vector<std::int8_t>& value_codes,
                    std::vector<std::uint16_t>& value_scales, long long count,
                    long long heads, long long dim) {
  const long long groups = dim / 32;
  std::fill(key_codes.begin(), key_codes.end(), std::int8_t{0});
  std::fill(value_codes.begin(), value_codes.end(), std::int8_t{0});
  std::fill(key_scales.begin(), key_scales.end(), std::uint16_t{0});
  std::fill(value_scales.begin(), value_scales.end(), std::uint16_t{0});
  for (long long token = 0; token < count; ++token) {
    const int slot = slots[static_cast<std::size_t>(token)];
    if (slot < 0) continue;
    for (long long head = 0; head < heads; ++head) {
      const long long source = (token * heads + head) * dim;
      const long long codes = (slot * heads + head) * dim;
      const long long scales = (slot * heads + head) * groups;
      for (long long group = 0; group < groups; ++group) {
        float key_amax = 0.0f;
        float value_amax = 0.0f;
        for (long long lane = 0; lane < 32; ++lane) {
          const long long index = source + group * 32 + lane;
          key_amax = std::max(key_amax, std::fabs(key[index]));
          value_amax = std::max(value_amax, std::fabs(value[index]));
        }
        const float key_scale = key_amax / 127.0f;
        const float value_scale = value_amax / 127.0f;
        key_scales[scales + group] = quixicore_cpu::float_to_f16(key_scale);
        value_scales[scales + group] = quixicore_cpu::float_to_f16(value_scale);
        const float key_inverse = key_scale > 0.0f ? 1.0f / key_scale : 0.0f;
        const float value_inverse =
            value_scale > 0.0f ? 1.0f / value_scale : 0.0f;
        for (long long lane = 0; lane < 32; ++lane) {
          const long long source_index = source + group * 32 + lane;
          const long long code_index = codes + group * 32 + lane;
          key_codes[code_index] = encode_q8(key[source_index], key_inverse);
          value_codes[code_index] =
              encode_q8(value[source_index], value_inverse);
        }
      }
    }
  }
}

void gather_oracle(const std::vector<std::int8_t>& key_codes,
                   const std::vector<std::uint16_t>& key_scales,
                   const std::vector<std::int8_t>& value_codes,
                   const std::vector<std::uint16_t>& value_scales,
                   const std::vector<int>& block_table,
                   const std::vector<int>& cumulative,
                   std::vector<float>& key_out, std::vector<float>& value_out,
                   long long tokens, long long sequences, long long heads,
                   long long dim, long long page, long long max_blocks) {
  const long long groups = dim / 32;
  long long sequence = 0;
  for (long long token = 0; token < tokens; ++token) {
    while (sequence + 1 < sequences && token >= cumulative[sequence + 1]) {
      ++sequence;
    }
    const long long local = token - cumulative[sequence];
    const int block = block_table[sequence * max_blocks + local / page];
    const long long output_base = token * heads * dim;
    if (block < 0) {
      std::fill_n(key_out.data() + output_base, heads * dim, 0.0f);
      std::fill_n(value_out.data() + output_base, heads * dim, 0.0f);
      continue;
    }
    const long long slot = static_cast<long long>(block) * page + local % page;
    for (long long head = 0; head < heads; ++head) {
      const long long codes = (slot * heads + head) * dim;
      const long long scales = (slot * heads + head) * groups;
      for (long long group = 0; group < groups; ++group) {
        const float key_scale =
            quixicore_cpu::f16_to_float(key_scales[scales + group]);
        const float value_scale =
            quixicore_cpu::f16_to_float(value_scales[scales + group]);
        for (long long lane = 0; lane < 32; ++lane) {
          const long long dim_index = group * 32 + lane;
          const long long output = output_base + head * dim + dim_index;
          key_out[output] = key_codes[codes + dim_index] * key_scale;
          value_out[output] = value_codes[codes + dim_index] * value_scale;
        }
      }
    }
  }
}

struct CodecBuffers {
  std::vector<float> key, value, key_out, value_out, reference_key,
      reference_value;
  std::vector<std::int8_t> key_codes, value_codes, reference_key_codes,
      reference_value_codes;
  std::vector<std::uint16_t> key_scales, value_scales, reference_key_scales,
      reference_value_scales;
  std::vector<int> slots, block_table, cumulative;
  long long cache_blocks = 0;
  long long tokens = 0;
  long long heads = 0;
  long long dim = 0;
  long long page = 0;
  long long max_blocks = 0;
  bool scatter = false;
};

CaseDecl make_codec(bool scatter, long long tokens, long long heads,
                    long long dim, long long page) {
  const long long cache_blocks = (tokens + page - 1) / page;
  const long long elements = tokens * heads * dim;
  const long long cache_elements = cache_blocks * page * heads * dim;
  const long long scale_elements = cache_elements / 32;
  CaseDecl decl;
  decl.kernel = "q8_kv";
  decl.variant = std::string(scatter ? "scatter" : "gather") + "_T" +
                 std::to_string(tokens) + "_H" + std::to_string(heads) + "_D" +
                 std::to_string(dim);
  decl.shape = {{"tokens", tokens}, {"heads", heads}, {"dim", dim}};
  decl.format = "q8_0";
  decl.notes = scatter ? "per-32 FP16-scale Q8_0 cache scatter"
                       : "packed-sequence Q8_0 cache gather";
  decl.bytes_moved =
      static_cast<double>(elements) *
      (scatter ? sizeof(float) * 2.0 + 2.125 : 2.125 + sizeof(float) * 2.0);
  decl.make = [=]() {
    auto b = std::make_shared<CodecBuffers>();
    b->cache_blocks = cache_blocks;
    b->tokens = tokens;
    b->heads = heads;
    b->dim = dim;
    b->page = page;
    b->max_blocks = cache_blocks;
    b->scatter = scatter;
    b->key.resize(elements);
    b->value.resize(elements);
    b->key_out.resize(elements);
    b->value_out.resize(elements);
    b->reference_key.resize(elements);
    b->reference_value.resize(elements);
    b->key_codes.resize(cache_elements);
    b->value_codes.resize(cache_elements);
    b->reference_key_codes.resize(cache_elements);
    b->reference_value_codes.resize(cache_elements);
    b->key_scales.resize(scale_elements);
    b->value_scales.resize(scale_elements);
    b->reference_key_scales.resize(scale_elements);
    b->reference_value_scales.resize(scale_elements);
    b->slots.resize(tokens);
    b->block_table.resize(cache_blocks);
    b->cumulative = {0, static_cast<int>(tokens)};
    Q8Rng rng;
    for (long long index = 0; index < elements; ++index) {
      b->key[index] = rng.next();
      b->value[index] = rng.next();
    }
    for (long long token = 0; token < tokens; ++token) b->slots[token] = token;
    for (long long block = 0; block < cache_blocks; ++block) {
      b->block_table[block] = static_cast<int>(block);
    }
    if (quixicore_cpu::kv_cache_scatter_q8_0(
            b->key.data(), b->value.data(), b->slots.data(),
            b->key_codes.data(), b->key_scales.data(), b->value_codes.data(),
            b->value_scales.data(), cache_blocks, tokens, heads, dim,
            page) != Status::kOk) {
      throw std::runtime_error("Q8_0 codec setup failed");
    }
    CaseBody body;
    if (scatter) {
      body.target = [b]() {
        if (quixicore_cpu::kv_cache_scatter_q8_0(
                b->key.data(), b->value.data(), b->slots.data(),
                b->key_codes.data(), b->key_scales.data(),
                b->value_codes.data(), b->value_scales.data(), b->cache_blocks,
                b->tokens, b->heads, b->dim, b->page) != Status::kOk) {
          throw std::runtime_error("Q8_0 scatter failed");
        }
        do_not_optimize(b->key_codes.data());
      };
      body.baselines.emplace_back("scalar_contract", [b]() {
        scatter_oracle(b->key, b->value, b->slots, b->reference_key_codes,
                       b->reference_key_scales, b->reference_value_codes,
                       b->reference_value_scales, b->tokens, b->heads, b->dim);
        do_not_optimize(b->reference_key_codes.data());
      });
    } else {
      body.target = [b]() {
        if (quixicore_cpu::kv_cache_gather_q8_0(
                b->key_codes.data(), b->key_scales.data(),
                b->value_codes.data(), b->value_scales.data(),
                b->block_table.data(), b->cumulative.data(), b->key_out.data(),
                b->value_out.data(), b->cache_blocks, b->tokens, 1, b->heads,
                b->dim, b->page, b->max_blocks) != Status::kOk) {
          throw std::runtime_error("Q8_0 gather failed");
        }
        do_not_optimize(b->key_out.data());
      };
      body.baselines.emplace_back("scalar_contract", [b]() {
        gather_oracle(b->key_codes, b->key_scales, b->value_codes,
                      b->value_scales, b->block_table, b->cumulative,
                      b->reference_key, b->reference_value, b->tokens, 1,
                      b->heads, b->dim, b->page, b->max_blocks);
        do_not_optimize(b->reference_key.data());
      });
    }
    body.check = [b]() {
      CheckResult result;
      if (b->scatter) {
        b->key_codes.assign(b->key_codes.size(), 1);
        b->value_codes.assign(b->value_codes.size(), 1);
        if (quixicore_cpu::kv_cache_scatter_q8_0(
                b->key.data(), b->value.data(), b->slots.data(),
                b->key_codes.data(), b->key_scales.data(),
                b->value_codes.data(), b->value_scales.data(), b->cache_blocks,
                b->tokens, b->heads, b->dim, b->page) != Status::kOk) {
          throw std::runtime_error("Q8_0 scatter check failed");
        }
        scatter_oracle(b->key, b->value, b->slots, b->reference_key_codes,
                       b->reference_key_scales, b->reference_value_codes,
                       b->reference_value_scales, b->tokens, b->heads, b->dim);
        result.passed = b->key_codes == b->reference_key_codes &&
                        b->value_codes == b->reference_value_codes &&
                        b->key_scales == b->reference_key_scales &&
                        b->value_scales == b->reference_value_scales;
      } else {
        if (quixicore_cpu::kv_cache_gather_q8_0(
                b->key_codes.data(), b->key_scales.data(),
                b->value_codes.data(), b->value_scales.data(),
                b->block_table.data(), b->cumulative.data(), b->key_out.data(),
                b->value_out.data(), b->cache_blocks, b->tokens, 1, b->heads,
                b->dim, b->page, b->max_blocks) != Status::kOk) {
          throw std::runtime_error("Q8_0 gather check failed");
        }
        gather_oracle(b->key_codes, b->key_scales, b->value_codes,
                      b->value_scales, b->block_table, b->cumulative,
                      b->reference_key, b->reference_value, b->tokens, 1,
                      b->heads, b->dim, b->page, b->max_blocks);
        result.passed = b->key_out == b->reference_key &&
                        b->value_out == b->reference_value;
      }
      return result;
    };
    return body;
  };
  return decl;
}

struct CopyBuffers {
  std::vector<std::int8_t> key, value, key_out, value_out, reference_key,
      reference_value;
  std::vector<std::uint16_t> key_scales, value_scales, key_scales_out,
      value_scales_out, reference_key_scales, reference_value_scales;
  std::vector<long long> pairs;
  long long blocks = 0;
  long long page = 0;
  long long heads = 0;
  long long dim = 0;
};

void copy_oracle(CopyBuffers& b) {
  b.reference_key = b.key;
  b.reference_value = b.value;
  b.reference_key_scales = b.key_scales;
  b.reference_value_scales = b.value_scales;
  const long long codes_per_block = b.page * b.heads * b.dim;
  const long long scales_per_block = codes_per_block / 32;
  for (std::size_t pair = 0; pair < b.pairs.size() / 2; ++pair) {
    const long long source = b.pairs[2 * pair];
    const long long destination = b.pairs[2 * pair + 1];
    std::copy_n(b.key.data() + source * codes_per_block, codes_per_block,
                b.reference_key.data() + destination * codes_per_block);
    std::copy_n(b.value.data() + source * codes_per_block, codes_per_block,
                b.reference_value.data() + destination * codes_per_block);
    std::copy_n(b.key_scales.data() + source * scales_per_block,
                scales_per_block,
                b.reference_key_scales.data() + destination * scales_per_block);
    std::copy_n(
        b.value_scales.data() + source * scales_per_block, scales_per_block,
        b.reference_value_scales.data() + destination * scales_per_block);
  }
}

CaseDecl make_copy(long long blocks, long long mappings, long long page,
                   long long heads, long long dim) {
  const long long code_count = blocks * page * heads * dim;
  const long long scale_count = code_count / 32;
  CaseDecl decl;
  decl.kernel = "q8_kv";
  decl.variant = "copy_blocks_B" + std::to_string(blocks) + "_M" +
                 std::to_string(mappings) + "_H" + std::to_string(heads) +
                 "_D" + std::to_string(dim);
  decl.shape = {{"blocks", blocks}, {"mappings", mappings}, {"dim", dim}};
  decl.format = "q8_0";
  decl.notes = "functional Q8_0 cache block clone and remap";
  decl.bytes_moved = static_cast<double>(code_count * 4 + scale_count * 8);
  decl.make = [=]() {
    auto b = std::make_shared<CopyBuffers>();
    b->blocks = blocks;
    b->page = page;
    b->heads = heads;
    b->dim = dim;
    b->key.resize(code_count);
    b->value.resize(code_count);
    b->key_out.resize(code_count);
    b->value_out.resize(code_count);
    b->key_scales.resize(scale_count);
    b->value_scales.resize(scale_count);
    b->key_scales_out.resize(scale_count);
    b->value_scales_out.resize(scale_count);
    b->pairs.resize(2 * mappings);
    for (long long index = 0; index < code_count; ++index) {
      b->key[index] = static_cast<std::int8_t>((index * 17) % 255 - 127);
      b->value[index] = static_cast<std::int8_t>((index * 29) % 255 - 127);
    }
    for (long long index = 0; index < scale_count; ++index) {
      b->key_scales[index] = static_cast<std::uint16_t>(0x1800 + index % 1024);
      b->value_scales[index] =
          static_cast<std::uint16_t>(0x1C00 + index % 1024);
    }
    for (long long pair = 0; pair < mappings; ++pair) {
      b->pairs[2 * pair] = pair % (blocks / 2);
      b->pairs[2 * pair + 1] = blocks - 1 - pair % (blocks / 2);
    }
    CaseBody body;
    body.target = [b, mappings]() {
      if (quixicore_cpu::kv_cache_copy_blocks_q8_0(
              b->key.data(), b->key_scales.data(), b->value.data(),
              b->value_scales.data(), b->key_out.data(),
              b->key_scales_out.data(), b->value_out.data(),
              b->value_scales_out.data(), b->pairs.data(), mappings, b->blocks,
              b->page, b->heads, b->dim) != Status::kOk) {
        throw std::runtime_error("Q8_0 block copy failed");
      }
      do_not_optimize(b->key_out.data());
    };
    body.baselines.emplace_back("std_copy_contract", [b]() {
      copy_oracle(*b);
      do_not_optimize(b->reference_key.data());
    });
    body.check = [b, mappings]() {
      if (quixicore_cpu::kv_cache_copy_blocks_q8_0(
              b->key.data(), b->key_scales.data(), b->value.data(),
              b->value_scales.data(), b->key_out.data(),
              b->key_scales_out.data(), b->value_out.data(),
              b->value_scales_out.data(), b->pairs.data(), mappings, b->blocks,
              b->page, b->heads, b->dim) != Status::kOk) {
        throw std::runtime_error("Q8_0 block copy check failed");
      }
      copy_oracle(*b);
      CheckResult result;
      result.passed = b->key_out == b->reference_key &&
                      b->value_out == b->reference_value &&
                      b->key_scales_out == b->reference_key_scales &&
                      b->value_scales_out == b->reference_value_scales;
      return result;
    };
    return body;
  };
  return decl;
}

struct AttentionBuffers {
  std::vector<float> q, key, value, decoded_key, decoded_value, target,
      reference;
  std::vector<std::int8_t> key_codes, value_codes;
  std::vector<std::uint16_t> key_scales, value_scales;
  std::vector<int> slots, block_table, context_lens, all_blocks, all_cumulative;
  long long cache_blocks = 0;
  long long batch = 0;
  long long query_heads = 0;
  long long kv_heads = 0;
  long long context = 0;
  long long dim = 0;
  long long page = 0;
  long long max_blocks = 0;
};

void materialized_attention(AttentionBuffers& b) {
  const long long cache_tokens = b.cache_blocks * b.page;
  if (quixicore_cpu::kv_cache_gather_q8_0(
          b.key_codes.data(), b.key_scales.data(), b.value_codes.data(),
          b.value_scales.data(), b.all_blocks.data(), b.all_cumulative.data(),
          b.decoded_key.data(), b.decoded_value.data(), b.cache_blocks,
          cache_tokens, 1, b.kv_heads, b.dim, b.page,
          b.cache_blocks) != Status::kOk ||
      quixicore_cpu::paged_attention(
          b.q.data(), b.decoded_key.data(), b.decoded_value.data(),
          b.block_table.data(), b.context_lens.data(), b.reference.data(),
          b.cache_blocks, b.batch, b.query_heads, b.kv_heads, b.dim, b.page,
          b.max_blocks) != Status::kOk) {
    throw std::runtime_error("materialized Q8_0 attention failed");
  }
}

CaseDecl make_attention(long long batch, long long query_heads,
                        long long kv_heads, long long context, long long dim,
                        long long page) {
  const long long max_blocks = (context + page - 1) / page;
  const long long cache_blocks = batch * max_blocks;
  const long long cache_tokens = cache_blocks * page;
  const long long cache_elements = cache_tokens * kv_heads * dim;
  CaseDecl decl;
  decl.kernel = "q8_kv";
  decl.variant = "paged_attention_B" + std::to_string(batch) + "_HQ" +
                 std::to_string(query_heads) + "_HKV" +
                 std::to_string(kv_heads) + "_S" + std::to_string(context) +
                 "_D" + std::to_string(dim);
  decl.shape = {{"batch", batch},
                {"query_heads", query_heads},
                {"kv_heads", kv_heads},
                {"context", context},
                {"dim", dim}};
  decl.format = "q8_0";
  decl.notes = "direct Q8_0 paged attention versus materialized FP32 cache";
  decl.flops = static_cast<double>(batch * query_heads * context * dim) * 4.0;
  decl.make = [=]() {
    auto b = std::make_shared<AttentionBuffers>();
    b->cache_blocks = cache_blocks;
    b->batch = batch;
    b->query_heads = query_heads;
    b->kv_heads = kv_heads;
    b->context = context;
    b->dim = dim;
    b->page = page;
    b->max_blocks = max_blocks;
    b->q.resize(batch * query_heads * dim);
    b->key.resize(cache_elements);
    b->value.resize(cache_elements);
    b->decoded_key.resize(cache_elements);
    b->decoded_value.resize(cache_elements);
    b->key_codes.resize(cache_elements);
    b->value_codes.resize(cache_elements);
    b->key_scales.resize(cache_elements / 32);
    b->value_scales.resize(cache_elements / 32);
    b->target.resize(batch * query_heads * dim);
    b->reference.resize(batch * query_heads * dim);
    b->slots.resize(cache_tokens);
    b->block_table.resize(batch * max_blocks);
    b->context_lens.resize(batch, static_cast<int>(context));
    b->all_blocks.resize(cache_blocks);
    b->all_cumulative = {0, static_cast<int>(cache_tokens)};
    Q8Rng rng;
    for (float& value : b->q) value = rng.next();
    for (float& value : b->key) value = rng.next();
    for (float& value : b->value) value = rng.next();
    for (long long token = 0; token < cache_tokens; ++token) {
      b->slots[token] = static_cast<int>(token);
    }
    for (long long request = 0; request < batch; ++request) {
      for (long long block = 0; block < max_blocks; ++block) {
        b->block_table[request * max_blocks + block] =
            static_cast<int>(request * max_blocks + block);
      }
    }
    for (long long block = 0; block < cache_blocks; ++block) {
      b->all_blocks[block] = static_cast<int>(block);
    }
    if (quixicore_cpu::kv_cache_scatter_q8_0(
            b->key.data(), b->value.data(), b->slots.data(),
            b->key_codes.data(), b->key_scales.data(), b->value_codes.data(),
            b->value_scales.data(), cache_blocks, cache_tokens, kv_heads, dim,
            page) != Status::kOk) {
      throw std::runtime_error("Q8_0 attention setup failed");
    }
    CaseBody body;
    body.target = [b]() {
      if (quixicore_cpu::paged_attention_q8_0(
              b->q.data(), b->key_codes.data(), b->key_scales.data(),
              b->value_codes.data(), b->value_scales.data(),
              b->block_table.data(), b->context_lens.data(), b->target.data(),
              b->cache_blocks, b->batch, b->query_heads, b->kv_heads, b->dim,
              b->page, b->max_blocks, 0.0f, 0) != Status::kOk) {
        throw std::runtime_error("direct Q8_0 attention failed");
      }
      do_not_optimize(b->target.data());
    };
    body.baselines.emplace_back("materialize_f32_then_attention", [b]() {
      materialized_attention(*b);
      do_not_optimize(b->reference.data());
    });
    body.check = [b]() {
      if (quixicore_cpu::paged_attention_q8_0(
              b->q.data(), b->key_codes.data(), b->key_scales.data(),
              b->value_codes.data(), b->value_scales.data(),
              b->block_table.data(), b->context_lens.data(), b->target.data(),
              b->cache_blocks, b->batch, b->query_heads, b->kv_heads, b->dim,
              b->page, b->max_blocks, 0.0f, 0) != Status::kOk) {
        throw std::runtime_error("direct Q8_0 attention check failed");
      }
      materialized_attention(*b);
      CheckResult result;
      for (std::size_t index = 0; index < b->target.size(); ++index) {
        check_value(result, b->target[index], b->reference[index],
                    Tolerance{1e-5, 2e-4});
      }
      return result;
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_q8_kv_cases(const BuildCtx& ctx, std::vector<CaseDecl>& out) {
  if (ctx.preset == Preset::kSmoke) {
    out.push_back(make_codec(true, 8, 2, 64, 16));
    out.push_back(make_codec(false, 8, 2, 64, 16));
    out.push_back(make_copy(4, 2, 16, 2, 64));
    out.push_back(make_attention(1, 2, 1, 32, 64, 16));
    return;
  }
  out.push_back(make_codec(true, 512, 8, 128, 16));
  out.push_back(make_codec(false, 512, 8, 128, 16));
  out.push_back(make_copy(32, 8, 16, 8, 128));
  out.push_back(make_attention(2, 8, 2, 512, 64, 16));
  out.push_back(make_attention(2, 8, 2, 512, 128, 16));
  if (ctx.preset == Preset::kComprehensive) {
    out.push_back(make_attention(4, 32, 8, 4096, 128, 16));
  }
}

}  // namespace qcb
