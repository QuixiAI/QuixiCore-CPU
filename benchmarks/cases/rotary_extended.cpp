// Focused evidence for explicit-position, multimodal, and fused norm/RoPE
// kernels. Baselines are independent scalar compositions of the public
// mathematical contract and retain preallocated storage.

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness/alloc.h"
#include "harness/case.h"
#include "harness/donotopt.h"
#include "quixicore_cpu/ops.h"

namespace qcb {
namespace {

using quixicore_cpu::Status;

class Rng {
 public:
  float next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 17;
    state_ ^= state_ << 5;
    return static_cast<float>(state_ >> 8) * (2.0f / 16777216.0f) - 1.0f;
  }

 private:
  std::uint32_t state_ = 0x6A09E667u;
};

struct RotaryBuffers {
  AlignedBuffer<float> x, cosine, sine, out, reference;
  AlignedBuffer<int> positions;
  long long batch = 0;
  long long heads = 0;
  long long tokens = 0;
  long long head_dim = 0;
  long long rotary_dim = 0;
  long long max_position = 0;
  int sections[3] = {};
  bool interleaved = false;
  bool multimodal = false;
  bool positions_per_batch = false;
};

void rotary_reference(RotaryBuffers& buffers) {
  const long long pairs = buffers.rotary_dim / 2;
  const long long position_batch_stride =
      buffers.multimodal ? 3 * buffers.tokens : buffers.tokens;
  for (long long item = 0; item < buffers.batch; ++item) {
    const long long position_base =
        buffers.positions_per_batch ? item * position_batch_stride : 0;
    for (long long head = 0; head < buffers.heads; ++head) {
      for (long long token = 0; token < buffers.tokens; ++token) {
        const long long row =
            ((item * buffers.heads + head) * buffers.tokens + token) *
            buffers.head_dim;
        for (long long pair = 0; pair < pairs; ++pair) {
          int axis = 0;
          if (buffers.multimodal) {
            if (buffers.interleaved) {
              axis = static_cast<int>(pair % 3);
            } else if (pair >= buffers.sections[0]) {
              axis = pair < buffers.sections[0] + buffers.sections[1] ? 1 : 2;
            }
          }
          const int position =
              buffers.positions
                  .get()[position_base + axis * buffers.tokens + token];
          const long long table = position * pairs + pair;
          const long long first_index =
              !buffers.multimodal && buffers.interleaved ? 2 * pair : pair;
          const long long second_index =
              !buffers.multimodal && buffers.interleaved ? 2 * pair + 1
                                                         : pairs + pair;
          const float a = buffers.x.get()[row + first_index];
          const float b = buffers.x.get()[row + second_index];
          const float c = buffers.cosine.get()[table];
          const float s = buffers.sine.get()[table];
          buffers.reference.get()[row + first_index] = a * c - b * s;
          buffers.reference.get()[row + second_index] = b * c + a * s;
        }
        for (long long d = buffers.rotary_dim; d < buffers.head_dim; ++d) {
          buffers.reference.get()[row + d] = buffers.x.get()[row + d];
        }
      }
    }
  }
}

CaseDecl make_rotary(long long batch, long long heads, long long tokens,
                     long long head_dim, long long rotary_dim, bool multimodal,
                     bool interleaved, bool positions_per_batch) {
  CaseDecl decl;
  decl.kernel = "rotary_extended";
  decl.variant = (multimodal ? "mrope" : "positioned") +
                 std::string(interleaved ? "_interleaved" : "_split") + "_B" +
                 std::to_string(batch) + "_H" + std::to_string(heads) + "_N" +
                 std::to_string(tokens) + "_D" + std::to_string(head_dim) +
                 "_R" + std::to_string(rotary_dim);
  decl.shape = {{"batch", batch},
                {"heads", heads},
                {"tokens", tokens},
                {"head_dim", head_dim},
                {"rotary_dim", rotary_dim}};
  decl.notes = multimodal
                   ? "three-axis table gather and split-half partial rotation"
                   : "explicit-position partial table rotation";
  decl.bytes_moved = static_cast<double>(batch * heads * tokens * head_dim) *
                     sizeof(float) * 2.0;
  decl.make = [=]() {
    auto buffers = std::make_shared<RotaryBuffers>();
    buffers->batch = batch;
    buffers->heads = heads;
    buffers->tokens = tokens;
    buffers->head_dim = head_dim;
    buffers->rotary_dim = rotary_dim;
    buffers->max_position = tokens + 17;
    buffers->multimodal = multimodal;
    buffers->interleaved = interleaved;
    buffers->positions_per_batch = positions_per_batch;
    const long long pairs = rotary_dim / 2;
    if (multimodal && interleaved) {
      buffers->sections[0] = static_cast<int>((pairs + 2) / 3);
      buffers->sections[1] = static_cast<int>((pairs + 1) / 3);
      buffers->sections[2] = static_cast<int>(pairs / 3);
    } else if (multimodal) {
      buffers->sections[0] = static_cast<int>(pairs / 2);
      buffers->sections[1] = static_cast<int>(pairs / 3);
      buffers->sections[2] =
          static_cast<int>(pairs - buffers->sections[0] - buffers->sections[1]);
    }
    const long long elements = batch * heads * tokens * head_dim;
    const long long position_count =
        (positions_per_batch ? batch : 1) * (multimodal ? 3 : 1) * tokens;
    buffers->x = aligned_alloc_array<float>(elements);
    buffers->cosine = aligned_alloc_array<float>(buffers->max_position * pairs);
    buffers->sine = aligned_alloc_array<float>(buffers->max_position * pairs);
    buffers->out = aligned_alloc_array<float>(elements);
    buffers->reference = aligned_alloc_array<float>(elements);
    buffers->positions = aligned_alloc_array<int>(position_count);
    Rng rng;
    for (long long i = 0; i < elements; ++i) buffers->x.get()[i] = rng.next();
    for (long long position = 0; position < buffers->max_position; ++position) {
      for (long long pair = 0; pair < pairs; ++pair) {
        const double angle = 0.0007 * (position + 1) * (pair + 1);
        buffers->cosine.get()[position * pairs + pair] =
            static_cast<float>(std::cos(angle));
        buffers->sine.get()[position * pairs + pair] =
            static_cast<float>(std::sin(angle));
      }
    }
    for (long long i = 0; i < position_count; ++i) {
      buffers->positions.get()[i] =
          static_cast<int>((i * 17 + 3) % buffers->max_position);
    }

    CaseBody body;
    body.target = [buffers]() {
      const Status status =
          buffers->multimodal
              ? quixicore_cpu::mrope(
                    buffers->x.get(), buffers->cosine.get(),
                    buffers->sine.get(), buffers->positions.get(),
                    buffers->sections, buffers->out.get(), buffers->batch,
                    buffers->heads, buffers->tokens, buffers->head_dim,
                    buffers->rotary_dim, buffers->max_position,
                    buffers->interleaved, buffers->positions_per_batch)
              : quixicore_cpu::rotary_positioned(
                    buffers->x.get(), buffers->cosine.get(),
                    buffers->sine.get(), buffers->positions.get(),
                    buffers->out.get(), buffers->batch, buffers->heads,
                    buffers->tokens, buffers->head_dim, buffers->rotary_dim,
                    buffers->max_position, buffers->interleaved,
                    buffers->positions_per_batch);
      if (status != Status::kOk) throw std::runtime_error("rotary failed");
      do_not_optimize(buffers->out.get());
    };
    body.baselines.emplace_back("scalar_composed", [buffers]() {
      rotary_reference(*buffers);
      do_not_optimize(buffers->reference.get());
    });
    body.check = [buffers]() {
      if (buffers->multimodal) {
        if (quixicore_cpu::mrope(
                buffers->x.get(), buffers->cosine.get(), buffers->sine.get(),
                buffers->positions.get(), buffers->sections, buffers->out.get(),
                buffers->batch, buffers->heads, buffers->tokens,
                buffers->head_dim, buffers->rotary_dim, buffers->max_position,
                buffers->interleaved,
                buffers->positions_per_batch) != Status::kOk) {
          throw std::runtime_error("mrope check failed");
        }
      } else if (quixicore_cpu::rotary_positioned(
                     buffers->x.get(), buffers->cosine.get(),
                     buffers->sine.get(), buffers->positions.get(),
                     buffers->out.get(), buffers->batch, buffers->heads,
                     buffers->tokens, buffers->head_dim, buffers->rotary_dim,
                     buffers->max_position, buffers->interleaved,
                     buffers->positions_per_batch) != Status::kOk) {
        throw std::runtime_error("positioned RoPE check failed");
      }
      rotary_reference(*buffers);
      CheckResult check;
      const long long count =
          buffers->batch * buffers->heads * buffers->tokens * buffers->head_dim;
      for (long long i = 0; i < count; ++i) {
        check_value(check, buffers->out.get()[i], buffers->reference.get()[i],
                    kFp32Tolerance);
      }
      return check;
    };
    return body;
  };
  return decl;
}

struct QkBuffers {
  AlignedBuffer<float> qkv, q_weight, k_weight, cosine, sine, out, reference;
  AlignedBuffer<int> positions;
  long long tokens = 0;
  long long query_heads = 0;
  long long key_heads = 0;
  long long value_heads = 0;
  long long head_dim = 0;
  long long rotary_dim = 0;
  long long max_position = 0;
  int sections[3] = {};
  bool multimodal = false;
};

void qk_reference(QkBuffers& buffers) {
  const long long total_heads =
      buffers.query_heads + buffers.key_heads + buffers.value_heads;
  const long long pairs = buffers.rotary_dim / 2;
  for (long long token = 0; token < buffers.tokens; ++token) {
    for (long long head = 0; head < buffers.query_heads + buffers.key_heads;
         ++head) {
      const long long row = (token * total_heads + head) * buffers.head_dim;
      const float* weight = head < buffers.query_heads ? buffers.q_weight.get()
                                                       : buffers.k_weight.get();
      double squares = 0.0;
      for (long long d = 0; d < buffers.head_dim; ++d) {
        const double value = buffers.qkv.get()[row + d];
        squares += value * value;
      }
      const double inverse = 1.0 / std::sqrt(squares / buffers.head_dim + 1e-5);
      for (long long pair = 0; pair < pairs; ++pair) {
        const int axis = buffers.multimodal ? static_cast<int>(pair % 3) : 0;
        const int position =
            buffers.positions.get()[axis * buffers.tokens + token];
        const long long table = position * pairs + pair;
        const double a =
            buffers.qkv.get()[row + pair] * inverse * (weight[pair] + 1.0);
        const double b = buffers.qkv.get()[row + pairs + pair] * inverse *
                         (weight[pairs + pair] + 1.0);
        buffers.reference.get()[row + pair] = static_cast<float>(
            a * buffers.cosine.get()[table] - b * buffers.sine.get()[table]);
        buffers.reference.get()[row + pairs + pair] = static_cast<float>(
            b * buffers.cosine.get()[table] + a * buffers.sine.get()[table]);
      }
      for (long long d = buffers.rotary_dim; d < buffers.head_dim; ++d) {
        buffers.reference.get()[row + d] = static_cast<float>(
            buffers.qkv.get()[row + d] * inverse * (weight[d] + 1.0));
      }
    }
    const long long value_row =
        (token * total_heads + buffers.query_heads + buffers.key_heads) *
        buffers.head_dim;
    for (long long d = 0; d < buffers.value_heads * buffers.head_dim; ++d) {
      buffers.reference.get()[value_row + d] = buffers.qkv.get()[value_row + d];
    }
  }
}

CaseDecl make_qk(long long tokens, long long query_heads, long long key_heads,
                 long long value_heads, long long head_dim,
                 long long rotary_dim, bool multimodal) {
  CaseDecl decl;
  decl.kernel = "rotary_extended";
  decl.variant =
      std::string(multimodal ? "qk_norm_mrope" : "qk_norm_positioned") + "_T" +
      std::to_string(tokens) + "_HQ" + std::to_string(query_heads) + "_HK" +
      std::to_string(key_heads) + "_HV" + std::to_string(value_heads) + "_D" +
      std::to_string(head_dim) + "_R" + std::to_string(rotary_dim);
  decl.shape = {{"tokens", tokens},       {"query_heads", query_heads},
                {"key_heads", key_heads}, {"value_heads", value_heads},
                {"head_dim", head_dim},   {"rotary_dim", rotary_dim}};
  decl.notes = "fused Q/K RMSNorm, positioned partial rotation, and V copy";
  decl.make = [=]() {
    auto buffers = std::make_shared<QkBuffers>();
    buffers->tokens = tokens;
    buffers->query_heads = query_heads;
    buffers->key_heads = key_heads;
    buffers->value_heads = value_heads;
    buffers->head_dim = head_dim;
    buffers->rotary_dim = rotary_dim;
    buffers->max_position = tokens + 13;
    buffers->multimodal = multimodal;
    const long long pairs = rotary_dim / 2;
    buffers->sections[0] = static_cast<int>((pairs + 2) / 3);
    buffers->sections[1] = static_cast<int>((pairs + 1) / 3);
    buffers->sections[2] = static_cast<int>(pairs / 3);
    const long long total_heads = query_heads + key_heads + value_heads;
    const long long elements = tokens * total_heads * head_dim;
    buffers->qkv = aligned_alloc_array<float>(elements);
    buffers->q_weight = aligned_alloc_array<float>(head_dim);
    buffers->k_weight = aligned_alloc_array<float>(head_dim);
    buffers->cosine = aligned_alloc_array<float>(buffers->max_position * pairs);
    buffers->sine = aligned_alloc_array<float>(buffers->max_position * pairs);
    buffers->positions =
        aligned_alloc_array<int>((multimodal ? 3 : 1) * tokens);
    buffers->out = aligned_alloc_array<float>(elements);
    buffers->reference = aligned_alloc_array<float>(elements);
    Rng rng;
    for (long long i = 0; i < elements; ++i) buffers->qkv.get()[i] = rng.next();
    for (long long d = 0; d < head_dim; ++d) {
      buffers->q_weight.get()[d] = 0.25f * rng.next();
      buffers->k_weight.get()[d] = 0.25f * rng.next();
    }
    for (long long position = 0; position < buffers->max_position; ++position) {
      for (long long pair = 0; pair < pairs; ++pair) {
        const double angle = 0.0009 * (position + 1) * (pair + 1);
        buffers->cosine.get()[position * pairs + pair] =
            static_cast<float>(std::cos(angle));
        buffers->sine.get()[position * pairs + pair] =
            static_cast<float>(std::sin(angle));
      }
    }
    for (long long i = 0; i < (multimodal ? 3 : 1) * tokens; ++i) {
      buffers->positions.get()[i] =
          static_cast<int>((i * 19 + 5) % buffers->max_position);
    }
    CaseBody body;
    body.target = [buffers]() {
      if (quixicore_cpu::qk_norm_rope_positioned(
              buffers->qkv.get(), buffers->q_weight.get(),
              buffers->k_weight.get(), buffers->cosine.get(),
              buffers->sine.get(), buffers->positions.get(), buffers->out.get(),
              buffers->tokens, buffers->query_heads, buffers->key_heads,
              buffers->value_heads, buffers->head_dim, buffers->rotary_dim,
              buffers->max_position, 1e-5f, false, 1.0f,
              buffers->multimodal ? buffers->sections : nullptr,
              buffers->multimodal) != Status::kOk) {
        throw std::runtime_error("qk_norm_rope_positioned failed");
      }
      do_not_optimize(buffers->out.get());
    };
    body.baselines.emplace_back("scalar_norm_rotate_pack", [buffers]() {
      qk_reference(*buffers);
      do_not_optimize(buffers->reference.get());
    });
    body.check = [buffers]() {
      if (quixicore_cpu::qk_norm_rope_positioned(
              buffers->qkv.get(), buffers->q_weight.get(),
              buffers->k_weight.get(), buffers->cosine.get(),
              buffers->sine.get(), buffers->positions.get(), buffers->out.get(),
              buffers->tokens, buffers->query_heads, buffers->key_heads,
              buffers->value_heads, buffers->head_dim, buffers->rotary_dim,
              buffers->max_position, 1e-5f, false, 1.0f,
              buffers->multimodal ? buffers->sections : nullptr,
              buffers->multimodal) != Status::kOk) {
        throw std::runtime_error("qk norm check failed");
      }
      qk_reference(*buffers);
      CheckResult check;
      const long long count =
          buffers->tokens *
          (buffers->query_heads + buffers->key_heads + buffers->value_heads) *
          buffers->head_dim;
      for (long long i = 0; i < count; ++i) {
        check_value(check, buffers->out.get()[i], buffers->reference.get()[i],
                    Tolerance{3e-5, 3e-4});
      }
      return check;
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_rotary_extended_cases(const BuildCtx& ctx,
                                 std::vector<CaseDecl>& out) {
  if (ctx.preset == Preset::kSmoke) {
    out.push_back(make_rotary(1, 2, 8, 64, 32, false, false, false));
    out.push_back(make_rotary(1, 2, 8, 64, 48, true, true, false));
    out.push_back(make_qk(8, 2, 1, 1, 64, 32, false));
    return;
  }
  out.push_back(make_rotary(1, 32, 512, 128, 64, false, false, false));
  out.push_back(make_rotary(2, 8, 256, 256, 128, false, true, true));
  out.push_back(make_rotary(1, 16, 512, 64, 48, true, true, false));
  out.push_back(make_qk(512, 16, 4, 4, 128, 64, false));
  out.push_back(make_qk(512, 16, 4, 4, 64, 48, true));
  if (ctx.preset == Preset::kComprehensive) {
    out.push_back(make_rotary(1, 8, 1024, 512, 128, false, true, false));
    out.push_back(make_qk(2048, 16, 4, 4, 256, 128, false));
  }
}

}  // namespace qcb
