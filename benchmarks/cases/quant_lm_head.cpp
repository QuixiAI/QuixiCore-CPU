#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness/alloc.h"
#include "harness/case.h"
#include "harness/donotopt.h"
#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/qgemm.h"
#include "quixicore_cpu/quant_import.h"

namespace qcb {
namespace {

using quixicore_cpu::CanonicalQuantLayout;
using quixicore_cpu::CanonicalQuantTensor;
using quixicore_cpu::FloatStorageType;
using quixicore_cpu::LmHeadSampling;
using quixicore_cpu::Status;

enum class LmKind { kArgmax, kTopK, kTopP, kMasked, kCandidates, kBeam };

struct LmFormat {
  CanonicalQuantLayout layout;
  long long group;
  const char* name;
};

struct LmBuffers {
  CanonicalQuantTensor tensor;
  quixicore_cpu::CpuPackedWeights prepared;
  AlignedBuffer<float> decoded;
  AlignedBuffer<float> hidden;
  AlignedBuffer<float> bias;
  AlignedBuffer<float> logits;
  std::vector<int> target_ids;
  std::vector<int> reference_ids;
  std::vector<float> target_scores;
  std::vector<float> reference_scores;
  std::vector<std::uint8_t> mask;
  std::vector<int> candidates;
  std::vector<long long> offsets;
  std::vector<float> cumulative;
  std::vector<int> target_parents;
  std::vector<int> reference_parents;
  long long rows = 0;
  long long vocab = 0;
  long long hidden_size = 0;
  int top_k = 0;
  LmKind kind = LmKind::kArgmax;
};

void require_ok(Status status, const char* message) {
  if (status != Status::kOk) throw std::runtime_error(message);
}

void run_target(LmBuffers& b) {
  const quixicore_cpu::FloatStorageInput hidden{
      b.hidden.get(), FloatStorageType::kF32, b.rows * b.hidden_size};
  switch (b.kind) {
    case LmKind::kArgmax:
      require_ok(quixicore_cpu::qgemm_prepacked_lm_head_sample_storage(
                     b.prepared, hidden, b.bias.get(), b.target_ids.data(),
                     b.rows, LmHeadSampling::kArgmax, 0, 0.0f, 0.0f, 71),
                 "canonical LM-head argmax failed");
      break;
    case LmKind::kTopK:
      require_ok(quixicore_cpu::qgemm_prepacked_lm_head_sample_storage(
                     b.prepared, hidden, b.bias.get(), b.target_ids.data(),
                     b.rows, LmHeadSampling::kTopK, b.top_k, 0.0f, 0.8f, 73),
                 "canonical LM-head top-k failed");
      break;
    case LmKind::kTopP:
      require_ok(quixicore_cpu::qgemm_prepacked_lm_head_sample_storage(
                     b.prepared, hidden, b.bias.get(), b.target_ids.data(),
                     b.rows, LmHeadSampling::kTopP, 0, 0.72f, 0.9f, 79),
                 "canonical LM-head top-p failed");
      break;
    case LmKind::kMasked:
      require_ok(quixicore_cpu::qgemm_prepacked_lm_head_masked_topk_storage(
                     b.prepared, hidden, b.bias.get(), b.mask.data(),
                     b.target_ids.data(), b.target_scores.data(), b.rows,
                     b.top_k, true),
                 "canonical masked LM-head failed");
      break;
    case LmKind::kCandidates:
      require_ok(
          quixicore_cpu::qgemm_prepacked_lm_head_candidates_storage(
              b.prepared, hidden, b.bias.get(), b.candidates.data(),
              b.offsets.data(), b.target_ids.data(), b.target_scores.data(),
              b.rows, static_cast<long long>(b.candidates.size()), b.top_k),
          "canonical candidate LM-head failed");
      break;
    case LmKind::kBeam:
      require_ok(quixicore_cpu::qgemm_prepacked_lm_head_beam_advance_storage(
                     b.prepared, hidden, b.bias.get(), b.cumulative.data(),
                     b.target_ids.data(), b.target_parents.data(),
                     b.target_scores.data(), 1, b.rows),
                 "canonical beam LM-head failed");
      break;
  }
  do_not_optimize(b.target_ids.data());
}

void run_baseline(LmBuffers& b) {
  if (b.kind == LmKind::kMasked) {
    require_ok(quixicore_cpu::lm_head_masked_topk(
                   b.hidden.get(), b.decoded.get(), b.bias.get(), b.mask.data(),
                   b.reference_ids.data(), b.reference_scores.data(), b.rows,
                   b.vocab, b.hidden_size, b.top_k, true),
               "materialized masked LM-head failed");
  } else if (b.kind == LmKind::kCandidates) {
    require_ok(
        quixicore_cpu::lm_head_candidates(
            b.hidden.get(), b.decoded.get(), b.bias.get(), b.candidates.data(),
            b.offsets.data(), b.reference_ids.data(), b.reference_scores.data(),
            b.rows, b.vocab, b.hidden_size,
            static_cast<long long>(b.candidates.size()), b.top_k),
        "materialized candidate LM-head failed");
  } else {
    require_ok(
        quixicore_cpu::qgemm_prepacked_storage(
            b.prepared,
            {b.hidden.get(), FloatStorageType::kF32, b.rows * b.hidden_size},
            {b.logits.get(), FloatStorageType::kF32, b.rows * b.vocab}, b.rows),
        "materialized canonical projection failed");
    for (long long row = 0; row < b.rows; ++row) {
      for (long long token = 0; token < b.vocab; ++token) {
        b.logits.get()[row * b.vocab + token] += b.bias.get()[token];
      }
    }
    switch (b.kind) {
      case LmKind::kArgmax:
        require_ok(quixicore_cpu::argmax_sample(
                       b.logits.get(), b.reference_ids.data(), b.rows, b.vocab),
                   "materialized argmax failed");
        break;
      case LmKind::kTopK:
        require_ok(
            quixicore_cpu::top_k_sample(b.logits.get(), b.reference_ids.data(),
                                        b.rows, b.vocab, b.top_k, 0.8f, 73),
            "materialized top-k failed");
        break;
      case LmKind::kTopP:
        require_ok(
            quixicore_cpu::top_p_sample(b.logits.get(), b.reference_ids.data(),
                                        b.rows, b.vocab, 0.72f, 0.9f, 79),
            "materialized top-p failed");
        break;
      case LmKind::kBeam:
        require_ok(quixicore_cpu::beam_search_step(
                       b.logits.get(), b.cumulative.data(),
                       b.reference_ids.data(), b.reference_parents.data(),
                       b.reference_scores.data(), 1, b.rows, b.vocab),
                   "materialized beam failed");
        break;
      case LmKind::kMasked:
      case LmKind::kCandidates:
        break;
    }
  }
  do_not_optimize(b.reference_ids.data());
}

CaseDecl make_lm_case(LmFormat format, LmKind kind, long long rows,
                      long long vocab, long long hidden_size) {
  const char* kind_name = kind == LmKind::kArgmax       ? "argmax"
                          : kind == LmKind::kTopK       ? "topk"
                          : kind == LmKind::kTopP       ? "topp"
                          : kind == LmKind::kMasked     ? "masked"
                          : kind == LmKind::kCandidates ? "candidates"
                                                        : "beam";
  CaseDecl decl;
  decl.kernel = "quant_lm_head";
  decl.variant = std::string(kind_name) + "_" + format.name + "_R" +
                 std::to_string(rows) + "_V" + std::to_string(vocab) + "_H" +
                 std::to_string(hidden_size);
  decl.shape = {{"rows", rows}, {"vocab", vocab}, {"hidden", hidden_size}};
  decl.format = format.name;
  decl.notes = "M3 S3/S4 prepared canonical streaming selection";
  decl.flops = 2.0 * rows * vocab * hidden_size;
  decl.make = [=]() {
    auto b = std::make_shared<LmBuffers>();
    b->rows = rows;
    b->vocab = vocab;
    b->hidden_size = hidden_size;
    b->kind = kind;
    b->top_k = kind == LmKind::kBeam ? static_cast<int>(rows) : 4;
    std::vector<float> source(static_cast<std::size_t>(vocab * hidden_size));
    for (long long item = 0; item < vocab * hidden_size; ++item) {
      source[static_cast<std::size_t>(item)] =
          static_cast<float>(static_cast<int>((item * 29 + 17) % 89) - 44) /
          127.0f;
    }
    require_ok(quixicore_cpu::quantize_canonical(
                   {source.data(), FloatStorageType::kF32, vocab * hidden_size},
                   vocab, hidden_size, format.layout, format.group, &b->tensor),
               "LM-head quantization failed");
    require_ok(b->prepared.prepare(b->tensor), "LM-head preparation failed");
    b->decoded = aligned_alloc_array<float>(vocab * hidden_size);
    require_ok(quixicore_cpu::dequantize_canonical(b->tensor, b->decoded.get(),
                                                   vocab * hidden_size),
               "LM-head decode failed");
    b->hidden = aligned_alloc_array<float>(rows * hidden_size);
    b->bias = aligned_alloc_array<float>(vocab);
    b->logits = aligned_alloc_array<float>(rows * vocab);
    for (long long item = 0; item < rows * hidden_size; ++item) {
      b->hidden.get()[item] =
          static_cast<float>(static_cast<int>((item * 37 + 11) % 79) - 39) /
          97.0f;
    }
    for (long long token = 0; token < vocab; ++token) {
      b->bias.get()[token] =
          static_cast<float>(static_cast<int>((token * 13) % 31) - 15) / 257.0f;
    }
    const long long output =
        kind == LmKind::kMasked || kind == LmKind::kCandidates ? rows * b->top_k
                                                               : rows;
    b->target_ids.resize(static_cast<std::size_t>(output));
    b->reference_ids.resize(static_cast<std::size_t>(output));
    b->target_scores.resize(static_cast<std::size_t>(output));
    b->reference_scores.resize(static_cast<std::size_t>(output));
    if (kind == LmKind::kMasked) {
      const long long stride = (vocab + 7) / 8;
      b->mask.resize(static_cast<std::size_t>(rows * stride));
      for (long long row = 0; row < rows; ++row) {
        for (long long token = row & 3; token < vocab; token += 16) {
          b->mask[row * stride + token / 8] |=
              static_cast<std::uint8_t>(0x80u >> (token & 7));
        }
      }
    } else if (kind == LmKind::kCandidates) {
      constexpr long long count = 128;
      b->offsets.resize(static_cast<std::size_t>(rows + 1));
      b->candidates.resize(static_cast<std::size_t>(rows * count));
      for (long long row = 0; row <= rows; ++row) b->offsets[row] = row * count;
      for (long long row = 0; row < rows; ++row) {
        for (long long item = 0; item < count; ++item) {
          b->candidates[row * count + item] =
              static_cast<int>((item * 31 + row * 7) % vocab);
        }
      }
    } else if (kind == LmKind::kBeam) {
      b->cumulative.resize(static_cast<std::size_t>(rows));
      b->target_parents.resize(static_cast<std::size_t>(rows));
      b->reference_parents.resize(static_cast<std::size_t>(rows));
      for (long long row = 0; row < rows; ++row)
        b->cumulative[row] = -0.125f * static_cast<float>(row);
    }
    CaseBody body;
    body.target = [b]() { run_target(*b); };
    body.baselines.emplace_back("materialized_logits_then_select",
                                [b]() { run_baseline(*b); });
    body.check = [b]() {
      run_target(*b);
      run_baseline(*b);
      CheckResult check;
      check.passed = b->target_ids == b->reference_ids;
      if (b->kind == LmKind::kBeam)
        check.passed =
            check.passed && b->target_parents == b->reference_parents;
      if (b->kind == LmKind::kMasked || b->kind == LmKind::kCandidates ||
          b->kind == LmKind::kBeam) {
        for (std::size_t item = 0; item < b->target_scores.size(); ++item) {
          check_value(check, b->target_scores[item], b->reference_scores[item],
                      Tolerance{2e-4, 2e-4});
        }
      }
      return check;
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_quant_lm_head_cases(const BuildCtx& ctx,
                               std::vector<CaseDecl>& out) {
  const LmFormat formats[] = {
      {CanonicalQuantLayout::kInt4Symmetric, 128, "int4"},
      {CanonicalQuantLayout::kFP8E4M3FN, 128, "fp8_e4m3"},
      {CanonicalQuantLayout::kMXFP4E2M1E8M0, 32, "mxfp4"},
      {CanonicalQuantLayout::kBitNetTernary, 32, "bitnet"},
  };
  const long long vocab = ctx.preset == Preset::kSmoke ? 128 : 4096;
  const long long hidden = ctx.preset == Preset::kSmoke ? 128 : 1024;
  const long long rows = ctx.preset == Preset::kSmoke ? 2 : 8;
  const std::size_t format_count = ctx.preset == Preset::kSmoke ? 1 : 4;
  for (std::size_t item = 0; item < format_count; ++item) {
    out.push_back(
        make_lm_case(formats[item], LmKind::kArgmax, 1, vocab, hidden));
    out.push_back(
        make_lm_case(formats[item], LmKind::kTopK, rows, vocab, hidden));
  }
  out.push_back(make_lm_case(formats[0], LmKind::kTopP, 1, vocab, hidden));
  out.push_back(make_lm_case(formats[0], LmKind::kMasked, rows, vocab, hidden));
  out.push_back(
      make_lm_case(formats[0], LmKind::kCandidates, rows, vocab, hidden));
  out.push_back(make_lm_case(formats[0], LmKind::kBeam, 4, vocab, hidden));
}

}  // namespace qcb
