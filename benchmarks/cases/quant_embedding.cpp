#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness/alloc.h"
#include "harness/case.h"
#include "harness/donotopt.h"
#include "quixicore_cpu/qgemm.h"
#include "quixicore_cpu/quant_import.h"

namespace qcb {
namespace {

using quixicore_cpu::CanonicalQuantLayout;
using quixicore_cpu::CanonicalQuantTensor;
using quixicore_cpu::FloatStorageType;
using quixicore_cpu::Status;

struct EmbeddingFormat {
  CanonicalQuantLayout layout;
  long long group;
  bool scale_2d;
  const char* name;
};

struct EmbeddingBuffers {
  CanonicalQuantTensor table;
  AlignedBuffer<float> decoded;
  AlignedBuffer<float> output;
  AlignedBuffer<float> reference;
  std::vector<int> ids;
  std::vector<long long> offsets;
  std::vector<float> weights;
  long long vocab = 0;
  long long dim = 0;
  long long count = 0;
  long long bags = 0;
  bool bag = false;
};

CaseDecl make_embedding_case(EmbeddingFormat format, bool bag, long long vocab,
                             long long count, long long dim) {
  CaseDecl decl;
  decl.kernel = "quant_embedding";
  decl.variant = std::string(bag ? "bag_" : "gather_") + format.name + "_V" +
                 std::to_string(vocab) + "_C" + std::to_string(count) + "_D" +
                 std::to_string(dim);
  const long long bags = bag ? count / 4 : 0;
  decl.shape = {{"vocab", vocab}, {"count", count}, {"dim", dim}};
  if (bag) decl.shape.emplace_back("bags", bags);
  decl.format = format.name;
  decl.notes = bag ? "M3 S2 direct canonical embedding-bag reduction"
                   : "M3 S1 selected canonical embedding rows";
  decl.bytes_moved = static_cast<double>(count * dim) * 4.5;
  decl.make = [format, bag, vocab, count, dim, bags]() {
    auto buffers = std::make_shared<EmbeddingBuffers>();
    buffers->vocab = vocab;
    buffers->dim = dim;
    buffers->count = count;
    buffers->bags = bags;
    buffers->bag = bag;
    std::vector<float> source(static_cast<std::size_t>(vocab * dim));
    for (long long index = 0; index < vocab * dim; ++index) {
      source[static_cast<std::size_t>(index)] =
          static_cast<float>(static_cast<int>((index * 31 + 17) % 83) - 41) /
          23.0f;
    }
    if (quixicore_cpu::quantize_canonical(
            {source.data(), FloatStorageType::kF32, vocab * dim}, vocab, dim,
            format.layout, format.group, &buffers->table,
            format.scale_2d) != Status::kOk) {
      throw std::runtime_error("embedding table quantization failed");
    }
    buffers->decoded = aligned_alloc_array<float>(vocab * dim);
    buffers->ids.resize(static_cast<std::size_t>(count));
    buffers->weights.resize(static_cast<std::size_t>(count));
    for (long long item = 0; item < count; ++item) {
      buffers->ids[static_cast<std::size_t>(item)] =
          static_cast<int>((item * 997 + 13) % vocab);
      buffers->weights[static_cast<std::size_t>(item)] =
          0.5f + static_cast<float>((item * 7) % 11) / 16.0f;
    }
    if (bag) {
      buffers->offsets.resize(static_cast<std::size_t>(bags + 1));
      for (long long item = 0; item <= bags; ++item) {
        buffers->offsets[static_cast<std::size_t>(item)] = item * 4;
      }
    }
    const long long output_rows = bag ? bags : count;
    buffers->output = aligned_alloc_array<float>(output_rows * dim);
    buffers->reference = aligned_alloc_array<float>(output_rows * dim);

    auto target = [buffers]() {
      const Status status =
          buffers->bag
              ? quixicore_cpu::canonical_quantized_embedding_bag_storage(
                    buffers->table, buffers->ids.data(),
                    buffers->offsets.data(), buffers->weights.data(),
                    {buffers->output.get(), FloatStorageType::kF32,
                     buffers->bags * buffers->dim},
                    buffers->count, buffers->bags, 1.0f, true, false)
              : quixicore_cpu::canonical_quantized_embedding_storage(
                    buffers->table, buffers->ids.data(), {},
                    {buffers->output.get(), FloatStorageType::kF32,
                     buffers->count * buffers->dim},
                    buffers->count);
      if (status != Status::kOk) {
        throw std::runtime_error("direct canonical embedding failed");
      }
      do_not_optimize(buffers->output.get());
    };
    auto baseline = [buffers]() {
      if (quixicore_cpu::dequantize_canonical(
              buffers->table, buffers->decoded.get(),
              buffers->vocab * buffers->dim) != Status::kOk) {
        throw std::runtime_error("embedding table decode failed");
      }
      if (buffers->bag) {
        for (long long bag_index = 0; bag_index < buffers->bags; ++bag_index) {
          for (long long column = 0; column < buffers->dim; ++column) {
            double sum = 0.0;
            for (long long item = buffers->offsets[bag_index];
                 item < buffers->offsets[bag_index + 1]; ++item) {
              sum += buffers->decoded
                         .get()[buffers->ids[static_cast<std::size_t>(item)] *
                                    buffers->dim +
                                column] *
                     buffers->weights[static_cast<std::size_t>(item)];
            }
            buffers->reference.get()[bag_index * buffers->dim + column] =
                static_cast<float>(sum);
          }
        }
      } else {
        for (long long item = 0; item < buffers->count; ++item) {
          for (long long column = 0; column < buffers->dim; ++column) {
            buffers->reference.get()[item * buffers->dim + column] =
                buffers->decoded
                    .get()[buffers->ids[static_cast<std::size_t>(item)] *
                               buffers->dim +
                           column];
          }
        }
      }
      do_not_optimize(buffers->reference.get());
    };
    CaseBody body;
    body.target = target;
    body.baselines.emplace_back("full_table_decode_then_select", baseline);
    body.check = [buffers, target, baseline]() {
      target();
      baseline();
      CheckResult check;
      const long long size =
          (buffers->bag ? buffers->bags : buffers->count) * buffers->dim;
      for (long long index = 0; index < size; ++index) {
        check_value(check, buffers->output.get()[index],
                    buffers->reference.get()[index], Tolerance{2e-5, 2e-5});
      }
      return check;
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_quant_embedding_cases(const BuildCtx& ctx,
                                 std::vector<CaseDecl>& out) {
  const EmbeddingFormat formats[] = {
      {CanonicalQuantLayout::kInt4Symmetric, 128, false, "int4"},
      {CanonicalQuantLayout::kFP8E4M3FN, 128, false, "fp8_e4m3"},
      {CanonicalQuantLayout::kMXFP4E2M1E8M0, 32, false, "mxfp4"},
      {CanonicalQuantLayout::kBitNetTernary, 32, false, "bitnet"},
  };
  const long long vocab = ctx.preset == Preset::kSmoke ? 64 : 4096;
  const long long count = ctx.preset == Preset::kSmoke ? 8 : 128;
  const long long dim = ctx.preset == Preset::kSmoke ? 128 : 1024;
  const std::size_t format_count = ctx.preset == Preset::kSmoke ? 1 : 4;
  for (std::size_t index = 0; index < format_count; ++index) {
    out.push_back(
        make_embedding_case(formats[index], false, vocab, count, dim));
    out.push_back(make_embedding_case(formats[index], true, vocab, count, dim));
  }
}

}  // namespace qcb
