#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/qgemm.h"
#include "quixicore_cpu/quant_import.h"
#include "quixicore_cpu/threading.h"

#define REQUIRE(condition)                                                     \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "FAILED: " #condition " at " << __FILE__ << ':' << __LINE__ \
                << '\n';                                                       \
      return false;                                                            \
    }                                                                          \
  } while (0)

namespace {

using quixicore_cpu::CanonicalQuantLayout;
using quixicore_cpu::CanonicalQuantTensor;
using quixicore_cpu::FloatStorageType;
using quixicore_cpu::Status;

struct LayoutCase {
  CanonicalQuantLayout layout;
  long long group;
  bool scale_2d;
  const char* name;
};

constexpr LayoutCase kLayouts[] = {
    {CanonicalQuantLayout::kInt4Symmetric, 32, false, "int4"},
    {CanonicalQuantLayout::kUInt4Affine, 32, false, "u4"},
    {CanonicalQuantLayout::kInt8Symmetric, 32, false, "int8"},
    {CanonicalQuantLayout::kInt8Affine, 0, false, "int8_affine"},
    {CanonicalQuantLayout::kFP8E4M3FN, 32, false, "fp8_e4"},
    {CanonicalQuantLayout::kFP8E5M2, 0, false, "fp8_e5"},
    {CanonicalQuantLayout::kFP4E2M1, 32, false, "fp4"},
    {CanonicalQuantLayout::kMXFP8E4M3E8M0, 32, false, "mxfp8"},
    {CanonicalQuantLayout::kMXFP4E2M1E8M0, 32, false, "mxfp4"},
    {CanonicalQuantLayout::kNVFP4E2M1E4M3, 16, true, "nvfp4"},
    {CanonicalQuantLayout::kBitNetTernary, 32, false, "bitnet"},
};

std::vector<float> source_values(long long count, int salt) {
  std::vector<float> values(static_cast<std::size_t>(count));
  for (long long index = 0; index < count; ++index) {
    values[static_cast<std::size_t>(index)] =
        static_cast<float>(static_cast<int>((index * 29 + salt * 13) % 73) -
                           36) /
        17.0f;
  }
  return values;
}

float rounded(float value, FloatStorageType type) {
  if (type == FloatStorageType::kF32) return value;
  return type == FloatStorageType::kF16
             ? quixicore_cpu::f16_to_float(quixicore_cpu::float_to_f16(value))
             : quixicore_cpu::bf16_to_float(
                   quixicore_cpu::float_to_bf16(value));
}

bool close(float actual, float expected) {
  return std::fabs(actual - expected) <= 2e-5f * (1.0f + std::fabs(expected));
}

bool test_serving_matrix() {
  constexpr long long vocab = 7;
  constexpr long long dim = 64;
  constexpr long long count = 5;
  const int ids[count] = {6, 1, 4, 1, 0};
  const std::vector<float> source = source_values(vocab * dim, 3);
  const std::vector<float> add_source = source_values(count * dim, 7);

  for (const LayoutCase& layout : kLayouts) {
    CanonicalQuantTensor table;
    REQUIRE(quixicore_cpu::quantize_canonical(
                {source.data(), FloatStorageType::kF32, vocab * dim}, vocab,
                dim, layout.layout, layout.group, &table,
                layout.scale_2d) == Status::kOk);
    std::vector<float> decoded(static_cast<std::size_t>(vocab * dim));
    REQUIRE(quixicore_cpu::dequantize_canonical(table, decoded.data(),
                                                vocab * dim) == Status::kOk);
    for (FloatStorageType type :
         {FloatStorageType::kF32, FloatStorageType::kF16,
          FloatStorageType::kBF16}) {
      std::vector<float> output_f32;
      std::vector<std::uint16_t> output_u16;
      std::vector<float> add_f32;
      std::vector<std::uint16_t> add_u16;
      void* output;
      const void* add;
      if (type == FloatStorageType::kF32) {
        output_f32.resize(count * dim);
        add_f32 = add_source;
        output = output_f32.data();
        add = add_f32.data();
      } else {
        output_u16.resize(count * dim);
        add_u16.resize(count * dim);
        for (long long index = 0; index < count * dim; ++index) {
          add_u16[static_cast<std::size_t>(index)] =
              type == FloatStorageType::kF16
                  ? quixicore_cpu::float_to_f16(add_source[index])
                  : quixicore_cpu::float_to_bf16(add_source[index]);
        }
        output = output_u16.data();
        add = add_u16.data();
      }
      REQUIRE(quixicore_cpu::canonical_quantized_embedding_storage(
                  table, ids, {add, type, count * dim},
                  {output, type, count * dim}, count, 0.75f) == Status::kOk);
      for (long long item = 0; item < count; ++item) {
        for (long long column = 0; column < dim; ++column) {
          const long long flat = item * dim + column;
          const float add_value = rounded(add_source[flat], type);
          const float expected = rounded(
              0.75f * decoded[ids[item] * dim + column] + add_value, type);
          const float actual =
              type == FloatStorageType::kF32
                  ? output_f32[flat]
                  : (type == FloatStorageType::kF16
                         ? quixicore_cpu::f16_to_float(output_u16[flat])
                         : quixicore_cpu::bf16_to_float(output_u16[flat]));
          if (!close(actual, expected)) {
            std::cerr << "gather " << layout.name
                      << " storage=" << static_cast<int>(type)
                      << " index=" << flat << '\n';
            return false;
          }
        }
      }
    }

    const long long offsets[] = {0, 2, 5};
    const float weights[] = {0.5f, -0.25f, 1.0f, 0.75f, -0.5f};
    std::vector<float> bag(2 * dim);
    REQUIRE(quixicore_cpu::canonical_quantized_embedding_bag_storage(
                table, ids, offsets, weights,
                {bag.data(), FloatStorageType::kF32, 2 * dim}, count, 2, 1.25f,
                true, true) == Status::kOk);
    for (long long bag_index = 0; bag_index < 2; ++bag_index) {
      const long long start = offsets[bag_index];
      const long long stop = offsets[bag_index + 1];
      for (long long column = 0; column < dim; ++column) {
        double sum = 0.0;
        for (long long item = start; item < stop; ++item) {
          sum += decoded[ids[item] * dim + column] * weights[item];
        }
        const float expected =
            static_cast<float>(1.25 * sum / static_cast<double>(stop - start));
        REQUIRE(close(bag[bag_index * dim + column], expected));
      }
    }
  }
  return true;
}

bool test_validation() {
  constexpr long long vocab = 2;
  constexpr long long dim = 32;
  std::vector<float> source(vocab * dim, 1.0f);
  CanonicalQuantTensor table;
  REQUIRE(quixicore_cpu::quantize_canonical(
              {source.data(), FloatStorageType::kF32, vocab * dim}, vocab, dim,
              CanonicalQuantLayout::kInt4Symmetric, 32, &table) == Status::kOk);
  const int invalid_id = 2;
  std::vector<float> output(dim, 17.0f);
  REQUIRE(quixicore_cpu::canonical_quantized_embedding_storage(
              table, &invalid_id, {},
              {output.data(), FloatStorageType::kF32, dim},
              1) == Status::kInvalidArgument);
  for (float value : output) REQUIRE(value == 17.0f);
  return true;
}

std::vector<float> lm_logits(const std::vector<float>& hidden,
                             FloatStorageType type,
                             const std::vector<float>& weights,
                             const std::vector<float>& bias, long long rows,
                             long long vocab, long long dim) {
  std::vector<float> logits(static_cast<std::size_t>(rows * vocab));
  for (long long row = 0; row < rows; ++row) {
    for (long long token = 0; token < vocab; ++token) {
      float sum = 0.0f;
      for (long long column = 0; column < dim; ++column) {
        sum += rounded(hidden[row * dim + column], type) *
               weights[token * dim + column];
      }
      logits[row * vocab + token] = sum + bias[token];
    }
  }
  return logits;
}

bool verify_structured_lm_layout(
    const quixicore_cpu::CpuPackedWeights& prepared,
    const std::vector<float>& hidden, const std::vector<float>& decoded,
    const std::vector<float>& bias, long long rows, long long vocab,
    long long dim) {
  constexpr int top_k = 3;
  const std::vector<float> logits = lm_logits(hidden, FloatStorageType::kF32,
                                              decoded, bias, rows, vocab, dim);
  const long long mask_stride = (vocab + 7) / 8;
  std::vector<std::uint8_t> mask(static_cast<std::size_t>(rows * mask_stride));
  std::vector<int> candidates(static_cast<std::size_t>(rows * 6));
  std::vector<long long> offsets(static_cast<std::size_t>(rows + 1));
  for (long long row = 0; row < rows; ++row) {
    offsets[row] = row * 6;
    for (long long item = 0; item < 6; ++item) {
      const int token = static_cast<int>((row * 5 + item * 3) % vocab);
      candidates[row * 6 + item] = token;
      mask[row * mask_stride + token / 8] |=
          static_cast<std::uint8_t>(0x80u >> (token & 7));
    }
  }
  offsets[rows] = rows * 6;
  std::vector<int> expected_ids(rows * top_k), actual_ids(rows * top_k);
  std::vector<float> expected_lp(rows * top_k), actual_lp(rows * top_k);
  REQUIRE(quixicore_cpu::lm_head_masked_topk(
              hidden.data(), decoded.data(), bias.data(), mask.data(),
              expected_ids.data(), expected_lp.data(), rows, vocab, dim, top_k,
              true) == Status::kOk);
  REQUIRE(quixicore_cpu::qgemm_prepacked_lm_head_masked_topk_storage(
              prepared, {hidden.data(), FloatStorageType::kF32, rows * dim},
              bias.data(), mask.data(), actual_ids.data(), actual_lp.data(),
              rows, top_k, true) == Status::kOk);
  REQUIRE(actual_ids == expected_ids);
  for (std::size_t item = 0; item < actual_lp.size(); ++item)
    REQUIRE(close(actual_lp[item], expected_lp[item]));

  REQUIRE(quixicore_cpu::lm_head_candidates(
              hidden.data(), decoded.data(), bias.data(), candidates.data(),
              offsets.data(), expected_ids.data(), expected_lp.data(), rows,
              vocab, dim, rows * 6, top_k) == Status::kOk);
  REQUIRE(quixicore_cpu::qgemm_prepacked_lm_head_candidates_storage(
              prepared, {hidden.data(), FloatStorageType::kF32, rows * dim},
              bias.data(), candidates.data(), offsets.data(), actual_ids.data(),
              actual_lp.data(), rows, rows * 6, top_k) == Status::kOk);
  REQUIRE(actual_ids == expected_ids);
  for (std::size_t item = 0; item < actual_lp.size(); ++item)
    REQUIRE(close(actual_lp[item], expected_lp[item]));

  std::vector<float> cumulative(static_cast<std::size_t>(rows));
  for (long long row = 0; row < rows; ++row)
    cumulative[row] = -0.125f * static_cast<float>(row);
  std::vector<int> expected_token(rows), expected_parent(rows);
  std::vector<int> actual_token(rows), actual_parent(rows);
  std::vector<float> expected_score(rows), actual_score(rows);
  REQUIRE(quixicore_cpu::beam_search_step(
              logits.data(), cumulative.data(), expected_token.data(),
              expected_parent.data(), expected_score.data(), 1, rows,
              vocab) == Status::kOk);
  REQUIRE(quixicore_cpu::qgemm_prepacked_lm_head_beam_advance_storage(
              prepared, {hidden.data(), FloatStorageType::kF32, rows * dim},
              bias.data(), cumulative.data(), actual_token.data(),
              actual_parent.data(), actual_score.data(), 1,
              rows) == Status::kOk);
  REQUIRE(actual_token == expected_token);
  REQUIRE(actual_parent == expected_parent);
  for (long long item = 0; item < rows; ++item)
    REQUIRE(close(actual_score[item], expected_score[item]));
  return true;
}

bool test_lm_head_matrix() {
  constexpr long long rows = 3;
  constexpr long long vocab = 19;
  constexpr long long dim = 64;
  constexpr int top_k = 4;
  const std::vector<float> source = source_values(vocab * dim, 11);
  const std::vector<float> hidden = source_values(rows * dim, 17);
  std::vector<float> bias(static_cast<std::size_t>(vocab));
  for (long long token = 0; token < vocab; ++token) {
    bias[token] = static_cast<float>((token * 7) % 13 - 6) / 31.0f;
  }
  for (const LayoutCase& layout : kLayouts) {
    CanonicalQuantTensor tensor;
    REQUIRE(quixicore_cpu::quantize_canonical(
                {source.data(), FloatStorageType::kF32, vocab * dim}, vocab,
                dim, layout.layout, layout.group, &tensor,
                layout.scale_2d) == Status::kOk);
    std::vector<float> decoded(static_cast<std::size_t>(vocab * dim));
    REQUIRE(quixicore_cpu::dequantize_canonical(tensor, decoded.data(),
                                                vocab * dim) == Status::kOk);
    quixicore_cpu::CpuPackedWeights prepared;
    REQUIRE(prepared.prepare(tensor) == Status::kOk);
    for (FloatStorageType type :
         {FloatStorageType::kF32, FloatStorageType::kF16,
          FloatStorageType::kBF16}) {
      std::vector<std::uint16_t> typed;
      const void* input = hidden.data();
      if (type != FloatStorageType::kF32) {
        typed.resize(hidden.size());
        for (std::size_t item = 0; item < hidden.size(); ++item) {
          typed[item] = type == FloatStorageType::kF16
                            ? quixicore_cpu::float_to_f16(hidden[item])
                            : quixicore_cpu::float_to_bf16(hidden[item]);
        }
        input = typed.data();
      }
      const std::vector<float> logits =
          lm_logits(hidden, type, decoded, bias, rows, vocab, dim);
      std::vector<int> expected(static_cast<std::size_t>(rows));
      std::vector<int> actual(static_cast<std::size_t>(rows), -1);
      REQUIRE(quixicore_cpu::argmax_sample(logits.data(), expected.data(), rows,
                                           vocab) == Status::kOk);
      REQUIRE(quixicore_cpu::qgemm_prepacked_lm_head_sample_storage(
                  prepared, {input, type, rows * dim}, bias.data(),
                  actual.data(), rows, quixicore_cpu::LmHeadSampling::kArgmax,
                  0, 0.0f, 0.0f, 41) == Status::kOk);
      REQUIRE(actual == expected);

      REQUIRE(quixicore_cpu::top_k_sample(logits.data(), expected.data(), rows,
                                          vocab, top_k, 0.8f,
                                          43) == Status::kOk);
      REQUIRE(quixicore_cpu::qgemm_prepacked_lm_head_sample_storage(
                  prepared, {input, type, rows * dim}, bias.data(),
                  actual.data(), rows, quixicore_cpu::LmHeadSampling::kTopK,
                  top_k, 0.0f, 0.8f, 43) == Status::kOk);
      REQUIRE(actual == expected);

      REQUIRE(quixicore_cpu::top_p_sample(logits.data(), expected.data(), rows,
                                          vocab, 0.72f, 0.9f,
                                          47) == Status::kOk);
      REQUIRE(quixicore_cpu::qgemm_prepacked_lm_head_sample_storage(
                  prepared, {input, type, rows * dim}, bias.data(),
                  actual.data(), rows, quixicore_cpu::LmHeadSampling::kTopP, 0,
                  0.72f, 0.9f, 47) == Status::kOk);
      REQUIRE(actual == expected);
    }
    REQUIRE(verify_structured_lm_layout(prepared, hidden, decoded, bias, rows,
                                        vocab, dim));
  }
  return true;
}

bool test_lm_head_structured() {
  constexpr long long rows = 2;
  constexpr long long vocab = 23;
  constexpr long long dim = 64;
  constexpr int top_k = 3;
  const std::vector<float> source = source_values(vocab * dim, 23);
  const std::vector<float> hidden = source_values(rows * dim, 29);
  std::vector<float> bias(static_cast<std::size_t>(vocab));
  for (long long token = 0; token < vocab; ++token) {
    bias[token] = static_cast<float>((token * 5) % 17 - 8) / 37.0f;
  }
  CanonicalQuantTensor tensor;
  REQUIRE(quixicore_cpu::quantize_canonical(
              {source.data(), FloatStorageType::kF32, vocab * dim}, vocab, dim,
              CanonicalQuantLayout::kMXFP4E2M1E8M0, 32,
              &tensor) == Status::kOk);
  std::vector<float> decoded(static_cast<std::size_t>(vocab * dim));
  REQUIRE(quixicore_cpu::dequantize_canonical(tensor, decoded.data(),
                                              vocab * dim) == Status::kOk);
  quixicore_cpu::CpuPackedWeights prepared;
  REQUIRE(prepared.prepare(tensor) == Status::kOk);
  const std::vector<float> logits = lm_logits(hidden, FloatStorageType::kF32,
                                              decoded, bias, rows, vocab, dim);

  const long long mask_stride = (vocab + 7) / 8;
  std::vector<std::uint8_t> mask(static_cast<std::size_t>(rows * mask_stride));
  const int allowed[rows][6] = {{1, 4, 7, 11, 17, 22}, {0, 3, 8, 14, 19, 21}};
  for (long long row = 0; row < rows; ++row) {
    for (int token : allowed[row]) {
      mask[row * mask_stride + token / 8] |=
          static_cast<std::uint8_t>(0x80u >> (token & 7));
    }
  }
  std::vector<int> expected_ids(rows * top_k), actual_ids(rows * top_k);
  std::vector<float> expected_lp(rows * top_k), actual_lp(rows * top_k);
  REQUIRE(quixicore_cpu::lm_head_masked_topk(
              hidden.data(), decoded.data(), bias.data(), mask.data(),
              expected_ids.data(), expected_lp.data(), rows, vocab, dim, top_k,
              true) == Status::kOk);
  REQUIRE(quixicore_cpu::qgemm_prepacked_lm_head_masked_topk_storage(
              prepared, {hidden.data(), FloatStorageType::kF32, rows * dim},
              bias.data(), mask.data(), actual_ids.data(), actual_lp.data(),
              rows, top_k, true) == Status::kOk);
  REQUIRE(actual_ids == expected_ids);
  for (std::size_t item = 0; item < actual_lp.size(); ++item) {
    REQUIRE(close(actual_lp[item], expected_lp[item]));
  }

  const int candidates[] = {1, 4, 7, 11, 17, 22, 0, 3, 8, 14, 19, 21};
  const long long offsets[] = {0, 6, 12};
  REQUIRE(quixicore_cpu::lm_head_candidates(
              hidden.data(), decoded.data(), bias.data(), candidates, offsets,
              expected_ids.data(), expected_lp.data(), rows, vocab, dim, 12,
              top_k) == Status::kOk);
  REQUIRE(quixicore_cpu::qgemm_prepacked_lm_head_candidates_storage(
              prepared, {hidden.data(), FloatStorageType::kF32, rows * dim},
              bias.data(), candidates, offsets, actual_ids.data(),
              actual_lp.data(), rows, 12, top_k) == Status::kOk);
  REQUIRE(actual_ids == expected_ids);
  for (std::size_t item = 0; item < actual_lp.size(); ++item) {
    REQUIRE(close(actual_lp[item], expected_lp[item]));
  }

  constexpr long long batch = 1;
  constexpr long long beam = rows;
  const float cumulative[] = {-0.1f, -0.3f};
  std::vector<int> expected_token(beam), expected_parent(beam);
  std::vector<float> expected_score(beam);
  REQUIRE(quixicore_cpu::beam_search_step(
              logits.data(), cumulative, expected_token.data(),
              expected_parent.data(), expected_score.data(), batch, beam,
              vocab) == Status::kOk);
  std::vector<int> actual_token(beam), actual_parent(beam);
  std::vector<float> actual_score(beam);
  REQUIRE(quixicore_cpu::qgemm_prepacked_lm_head_beam_advance_storage(
              prepared, {hidden.data(), FloatStorageType::kF32, rows * dim},
              bias.data(), cumulative, actual_token.data(),
              actual_parent.data(), actual_score.data(), batch,
              beam) == Status::kOk);
  REQUIRE(actual_token == expected_token);
  REQUIRE(actual_parent == expected_parent);
  for (std::size_t item = 0; item < actual_score.size(); ++item) {
    REQUIRE(close(actual_score[item], expected_score[item]));
  }

  const int duplicate_candidates[] = {1, 1, 7, 11, 0, 3, 8, 14};
  const long long duplicate_offsets[] = {0, 4, 8};
  std::fill(actual_ids.begin(), actual_ids.end(), 91);
  REQUIRE(quixicore_cpu::qgemm_prepacked_lm_head_candidates_storage(
              prepared, {hidden.data(), FloatStorageType::kF32, rows * dim},
              bias.data(), duplicate_candidates, duplicate_offsets,
              actual_ids.data(), actual_lp.data(), rows, 8,
              top_k) == Status::kInvalidArgument);
  for (int value : actual_ids) REQUIRE(value == 91);
  return true;
}

bool test_lm_head_ties() {
  constexpr long long vocab = 5;
  constexpr long long dim = 32;
  std::vector<float> source(vocab * dim, 0.0f);
  std::vector<float> hidden(dim, 1.0f);
  CanonicalQuantTensor tensor;
  REQUIRE(quixicore_cpu::quantize_canonical(
              {source.data(), FloatStorageType::kF32, vocab * dim}, vocab, dim,
              CanonicalQuantLayout::kInt4Symmetric, dim,
              &tensor) == Status::kOk);
  quixicore_cpu::CpuPackedWeights prepared;
  REQUIRE(prepared.prepare(tensor) == Status::kOk);
  int token = -1;
  REQUIRE(quixicore_cpu::qgemm_prepacked_lm_head_sample_storage(
              prepared, {hidden.data(), FloatStorageType::kF32, dim}, nullptr,
              &token, 1, quixicore_cpu::LmHeadSampling::kArgmax, 0, 0.0f, 0.0f,
              0) == Status::kOk);
  REQUIRE(token == 0);

  std::uint8_t mask = 0;
  for (int id : {4, 2, 1}) mask |= static_cast<std::uint8_t>(0x80u >> id);
  int ids[3] = {-1, -1, -1};
  float log_probabilities[3] = {};
  REQUIRE(quixicore_cpu::qgemm_prepacked_lm_head_masked_topk_storage(
              prepared, {hidden.data(), FloatStorageType::kF32, dim}, nullptr,
              &mask, ids, log_probabilities, 1, 3, true) == Status::kOk);
  REQUIRE(ids[0] == 1 && ids[1] == 2 && ids[2] == 4);
  for (float value : log_probabilities) REQUIRE(close(value, -std::log(3.0f)));
  return true;
}

float storage_value(const std::vector<float>& f32,
                    const std::vector<std::uint16_t>& u16,
                    FloatStorageType type, std::size_t index) {
  if (type == FloatStorageType::kF32) return f32[index];
  return type == FloatStorageType::kF16
             ? quixicore_cpu::f16_to_float(u16[index])
             : quixicore_cpu::bf16_to_float(u16[index]);
}

bool test_grouped_moe_matrix() {
  constexpr long long experts = 3;
  constexpr long long rows = 7;
  constexpr long long input_columns = 64;
  constexpr long long output_columns = 7;
  const int expert_ids[rows] = {2, 0, -1, 1, 2, 0, 1};
  const std::vector<float> source = source_values(rows * input_columns, 37);
  std::vector<float> bias(experts * output_columns);
  for (long long index = 0; index < experts * output_columns; ++index)
    bias[index] = static_cast<float>((index * 7) % 19 - 9) / 29.0f;

  for (const LayoutCase& layout : kLayouts) {
    std::vector<CanonicalQuantTensor> gate_tensors(experts);
    std::vector<CanonicalQuantTensor> up_tensors(experts);
    std::vector<quixicore_cpu::CpuPackedWeights> gate(experts);
    std::vector<quixicore_cpu::CpuPackedWeights> up(experts);
    std::vector<std::vector<float>> decoded_gate(experts);
    std::vector<std::vector<float>> decoded_up(experts);
    for (long long expert = 0; expert < experts; ++expert) {
      const std::vector<float> gate_source = source_values(
          output_columns * input_columns, 41 + static_cast<int>(expert) * 3);
      const std::vector<float> up_source = source_values(
          output_columns * input_columns, 53 + static_cast<int>(expert) * 5);
      REQUIRE(quixicore_cpu::quantize_canonical(
                  {gate_source.data(), FloatStorageType::kF32,
                   output_columns * input_columns},
                  output_columns, input_columns, layout.layout, layout.group,
                  &gate_tensors[expert], layout.scale_2d) == Status::kOk);
      REQUIRE(quixicore_cpu::quantize_canonical(
                  {up_source.data(), FloatStorageType::kF32,
                   output_columns * input_columns},
                  output_columns, input_columns, layout.layout, layout.group,
                  &up_tensors[expert], layout.scale_2d) == Status::kOk);
      decoded_gate[expert].resize(output_columns * input_columns);
      decoded_up[expert].resize(output_columns * input_columns);
      REQUIRE(quixicore_cpu::dequantize_canonical(
                  gate_tensors[expert], decoded_gate[expert].data(),
                  output_columns * input_columns) == Status::kOk);
      REQUIRE(quixicore_cpu::dequantize_canonical(
                  up_tensors[expert], decoded_up[expert].data(),
                  output_columns * input_columns) == Status::kOk);
      REQUIRE(gate[expert].prepare(gate_tensors[expert]) == Status::kOk);
      REQUIRE(up[expert].prepare(up_tensors[expert]) == Status::kOk);
    }

    for (FloatStorageType type :
         {FloatStorageType::kF32, FloatStorageType::kF16,
          FloatStorageType::kBF16}) {
      std::vector<std::uint16_t> encoded;
      const void* input = source.data();
      if (type != FloatStorageType::kF32) {
        encoded.resize(source.size());
        for (std::size_t index = 0; index < source.size(); ++index) {
          encoded[index] = type == FloatStorageType::kF16
                               ? quixicore_cpu::float_to_f16(source[index])
                               : quixicore_cpu::float_to_bf16(source[index]);
        }
        input = encoded.data();
      }
      std::vector<float> projected_f32;
      std::vector<std::uint16_t> projected_u16;
      std::vector<float> swiglu_f32;
      std::vector<std::uint16_t> swiglu_u16;
      void* projected;
      void* swiglu;
      if (type == FloatStorageType::kF32) {
        projected_f32.resize(rows * output_columns);
        swiglu_f32.resize(rows * output_columns);
        projected = projected_f32.data();
        swiglu = swiglu_f32.data();
      } else {
        projected_u16.resize(rows * output_columns);
        swiglu_u16.resize(rows * output_columns);
        projected = projected_u16.data();
        swiglu = swiglu_u16.data();
      }
      REQUIRE(quixicore_cpu::moe_grouped_prepacked_storage(
                  gate.data(), experts, {input, type, rows * input_columns},
                  expert_ids, bias.data(),
                  {projected, type, rows * output_columns}, rows,
                  quixicore_cpu::LinearActivation::kSilu) == Status::kOk);
      REQUIRE(quixicore_cpu::moe_grouped_prepacked_swiglu_storage(
                  gate.data(), up.data(), experts,
                  {input, type, rows * input_columns}, expert_ids,
                  {swiglu, type, rows * output_columns}, rows) == Status::kOk);
      for (long long row = 0; row < rows; ++row) {
        for (long long column = 0; column < output_columns; ++column) {
          const std::size_t flat =
              static_cast<std::size_t>(row * output_columns + column);
          if (expert_ids[row] < 0) {
            REQUIRE(storage_value(projected_f32, projected_u16, type, flat) ==
                    0.0f);
            REQUIRE(storage_value(swiglu_f32, swiglu_u16, type, flat) == 0.0f);
            continue;
          }
          const int expert = expert_ids[row];
          float gate_sum = 0.0f;
          float up_sum = 0.0f;
          for (long long inner = 0; inner < input_columns; ++inner) {
            const float activation =
                rounded(source[row * input_columns + inner], type);
            gate_sum += decoded_gate[expert][column * input_columns + inner] *
                        activation;
            up_sum +=
                decoded_up[expert][column * input_columns + inner] * activation;
          }
          const float projected_expected = rounded(
              (gate_sum + bias[expert * output_columns + column]) /
                  (1.0f + std::exp(-(gate_sum +
                                     bias[expert * output_columns + column]))),
              type);
          const float swiglu_expected =
              rounded(gate_sum / (1.0f + std::exp(-gate_sum)) * up_sum, type);
          const float actual_projected =
              storage_value(projected_f32, projected_u16, type, flat);
          const float actual_swiglu =
              storage_value(swiglu_f32, swiglu_u16, type, flat);
          if (!close(actual_projected, projected_expected) ||
              !close(actual_swiglu, swiglu_expected)) {
            std::cerr << "moe " << layout.name
                      << " storage=" << static_cast<int>(type) << " row=" << row
                      << " column=" << column << '\n';
            return false;
          }
        }
      }
    }

    std::vector<float> wrapper_output(rows * output_columns);
    Status wrapper = Status::kUnsupportedFormat;
    if (layout.layout == CanonicalQuantLayout::kFP8E4M3FN ||
        layout.layout == CanonicalQuantLayout::kFP8E5M2 ||
        layout.layout == CanonicalQuantLayout::kMXFP8E4M3E8M0) {
      wrapper = quixicore_cpu::moe_grouped_fp8_prepacked_storage(
          gate.data(), experts,
          {source.data(), FloatStorageType::kF32, rows * input_columns},
          expert_ids,
          {wrapper_output.data(), FloatStorageType::kF32,
           rows * output_columns},
          rows);
    } else if (layout.layout == CanonicalQuantLayout::kInt4Symmetric ||
               layout.layout == CanonicalQuantLayout::kUInt4Affine ||
               layout.layout == CanonicalQuantLayout::kFP4E2M1 ||
               layout.layout == CanonicalQuantLayout::kMXFP4E2M1E8M0) {
      wrapper = quixicore_cpu::moe_grouped_wna16_prepacked_storage(
          gate.data(), experts,
          {source.data(), FloatStorageType::kF32, rows * input_columns},
          expert_ids,
          {wrapper_output.data(), FloatStorageType::kF32,
           rows * output_columns},
          rows);
    } else if (layout.layout == CanonicalQuantLayout::kNVFP4E2M1E4M3) {
      wrapper = quixicore_cpu::moe_grouped_nvfp4_prepacked_storage(
          gate.data(), experts,
          {source.data(), FloatStorageType::kF32, rows * input_columns},
          expert_ids,
          {wrapper_output.data(), FloatStorageType::kF32,
           rows * output_columns},
          rows);
    }
    if (wrapper != Status::kUnsupportedFormat) REQUIRE(wrapper == Status::kOk);

    if (layout.layout == CanonicalQuantLayout::kInt4Symmetric) {
      const int sorted_ids[6] = {0, 0, 1, 1, 2, 2};
      std::vector<float> sorted_output(6 * output_columns);
      REQUIRE(quixicore_cpu::moe_grouped_prepacked_storage(
                  gate.data(), experts,
                  {source.data(), FloatStorageType::kF32, 6 * input_columns},
                  sorted_ids, nullptr,
                  {sorted_output.data(), FloatStorageType::kF32,
                   6 * output_columns},
                  6, quixicore_cpu::LinearActivation::kNone) == Status::kOk);
      std::vector<float> untouched(rows * output_columns, 91.0f);
      int invalid_ids[rows] = {2, 0, -1, 1, 3, 0, 1};
      REQUIRE(
          quixicore_cpu::moe_grouped_prepacked_storage(
              gate.data(), experts,
              {source.data(), FloatStorageType::kF32, rows * input_columns},
              invalid_ids, nullptr,
              {untouched.data(), FloatStorageType::kF32, rows * output_columns},
              rows, quixicore_cpu::LinearActivation::kNone) ==
          Status::kInvalidArgument);
      for (float value : untouched) REQUIRE(value == 91.0f);
      const int padded_ids[rows] = {-1, -1, -1, -1, -1, -1, -1};
      REQUIRE(
          quixicore_cpu::moe_grouped_prepacked_storage(
              gate.data(), experts,
              {source.data(), FloatStorageType::kF32, rows * input_columns},
              padded_ids, nullptr,
              {untouched.data(), FloatStorageType::kF32, rows * output_columns},
              rows, quixicore_cpu::LinearActivation::kNone) == Status::kOk);
      for (float value : untouched) REQUIRE(value == 0.0f);
      REQUIRE(
          quixicore_cpu::moe_grouped_fp8_prepacked_storage(
              gate.data(), experts,
              {source.data(), FloatStorageType::kF32, rows * input_columns},
              expert_ids,
              {untouched.data(), FloatStorageType::kF32, rows * output_columns},
              rows) == Status::kUnsupportedFormat);
    }
  }
  return true;
}

bool test_grouped_moe_quantized_matrix() {
  struct PairCase {
    CanonicalQuantLayout weight;
    long long weight_group;
    bool weight_scale_2d;
    CanonicalQuantLayout activation;
    long long activation_group;
    bool activation_scale_2d;
    const char* name;
  };
  constexpr PairCase pairs[] = {
      {CanonicalQuantLayout::kInt4Symmetric, 32, false,
       CanonicalQuantLayout::kInt8Symmetric, 32, false, "w4a8"},
      {CanonicalQuantLayout::kUInt4Affine, 32, false,
       CanonicalQuantLayout::kInt8Symmetric, 32, false, "u4a8"},
      {CanonicalQuantLayout::kInt8Symmetric, 32, false,
       CanonicalQuantLayout::kInt8Affine, 0, false, "w8a8_affine"},
      {CanonicalQuantLayout::kInt8Affine, 0, false,
       CanonicalQuantLayout::kInt8Symmetric, 32, false, "affine_w8a8"},
      {CanonicalQuantLayout::kFP8E4M3FN, 32, false,
       CanonicalQuantLayout::kFP8E4M3FN, 32, false, "fp8_e4"},
      {CanonicalQuantLayout::kFP8E5M2, 0, false,
       CanonicalQuantLayout::kFP8E4M3FN, 32, false, "fp8_mixed"},
      {CanonicalQuantLayout::kFP4E2M1, 32, false,
       CanonicalQuantLayout::kFP4E2M1, 32, false, "fp4"},
      {CanonicalQuantLayout::kMXFP8E4M3E8M0, 32, false,
       CanonicalQuantLayout::kMXFP8E4M3E8M0, 32, false, "mxfp8"},
      {CanonicalQuantLayout::kMXFP4E2M1E8M0, 32, false,
       CanonicalQuantLayout::kMXFP4E2M1E8M0, 32, false, "mxfp4"},
      {CanonicalQuantLayout::kNVFP4E2M1E4M3, 16, true,
       CanonicalQuantLayout::kNVFP4E2M1E4M3, 16, true, "nvfp4"},
      {CanonicalQuantLayout::kBitNetTernary, 32, false,
       CanonicalQuantLayout::kInt8Symmetric, 32, false, "bitnet_a8"},
  };
  constexpr long long experts = 3;
  constexpr long long rows = 7;
  constexpr long long n = 7;
  constexpr long long k = 64;
  const int expert_ids[rows] = {2, 0, -1, 1, 2, 0, 1};
  const std::vector<float> activation_source = source_values(rows * k, 71);
  for (const PairCase& pair : pairs) {
    CanonicalQuantTensor activation;
    REQUIRE(quixicore_cpu::quantize_canonical(
                {activation_source.data(), FloatStorageType::kF32, rows * k},
                rows, k, pair.activation, pair.activation_group, &activation,
                pair.activation_scale_2d) == Status::kOk);
    std::vector<float> decoded_activation(rows * k);
    REQUIRE(quixicore_cpu::dequantize_canonical(activation,
                                                decoded_activation.data(),
                                                rows * k) == Status::kOk);
    std::vector<CanonicalQuantTensor> tensors(experts);
    std::vector<quixicore_cpu::CpuPackedWeights> prepared(experts);
    std::vector<std::vector<float>> decoded(experts);
    for (long long expert = 0; expert < experts; ++expert) {
      const std::vector<float> source =
          source_values(n * k, 73 + static_cast<int>(expert) * 7);
      REQUIRE(quixicore_cpu::quantize_canonical(
                  {source.data(), FloatStorageType::kF32, n * k}, n, k,
                  pair.weight, pair.weight_group, &tensors[expert],
                  pair.weight_scale_2d) == Status::kOk);
      decoded[expert].resize(n * k);
      REQUIRE(quixicore_cpu::dequantize_canonical(tensors[expert],
                                                  decoded[expert].data(),
                                                  n * k) == Status::kOk);
      REQUIRE(prepared[expert].prepare(tensors[expert]) == Status::kOk);
    }
    std::vector<float> output(rows * n, 19.0f);
    REQUIRE(quixicore_cpu::moe_grouped_prepacked_quantized(
                prepared.data(), experts, activation, expert_ids,
                output.data()) == Status::kOk);
    for (long long row = 0; row < rows; ++row) {
      for (long long column = 0; column < n; ++column) {
        if (expert_ids[row] < 0) {
          REQUIRE(output[row * n + column] == 0.0f);
          continue;
        }
        float expected = 0.0f;
        for (long long inner = 0; inner < k; ++inner) {
          expected += decoded[expert_ids[row]][column * k + inner] *
                      decoded_activation[row * k + inner];
        }
        if (!close(output[row * n + column], expected)) {
          std::cerr << "quantized moe " << pair.name << " row=" << row
                    << " column=" << column << '\n';
          return false;
        }
      }
    }
  }
  return true;
}

bool same_quant_packet(const CanonicalQuantTensor& actual,
                       const CanonicalQuantTensor& expected) {
  if (actual.metadata.layout != expected.metadata.layout ||
      actual.metadata.logical_rows != expected.metadata.logical_rows ||
      actual.metadata.logical_columns != expected.metadata.logical_columns ||
      actual.metadata.group_size != expected.metadata.group_size ||
      actual.metadata.scale_mode != expected.metadata.scale_mode ||
      actual.metadata.scale_2d != expected.metadata.scale_2d ||
      actual.metadata.scale_domain_rows !=
          expected.metadata.scale_domain_rows) {
    std::cerr << "packet metadata mismatch layout="
              << static_cast<int>(actual.metadata.layout) << '/'
              << static_cast<int>(expected.metadata.layout)
              << " group=" << actual.metadata.group_size << '/'
              << expected.metadata.group_size
              << " mode=" << static_cast<int>(actual.metadata.scale_mode) << '/'
              << static_cast<int>(expected.metadata.scale_mode)
              << " domain=" << actual.metadata.scale_domain_rows << '/'
              << expected.metadata.scale_domain_rows << '\n';
    return false;
  }
  if (actual.metadata.global_scale == expected.metadata.global_scale &&
      actual.data == expected.data &&
      actual.scale_codes == expected.scale_codes &&
      actual.scales == expected.scales &&
      actual.zero_points == expected.zero_points) {
    return true;
  }
  const long long count =
      actual.metadata.logical_rows * actual.metadata.logical_columns;
  std::vector<float> actual_values(count);
  std::vector<float> expected_values(count);
  const Status actual_status =
      quixicore_cpu::dequantize_canonical(actual, actual_values.data(), count);
  const Status expected_status = quixicore_cpu::dequantize_canonical(
      expected, expected_values.data(), count);
  if (actual_status != Status::kOk || expected_status != Status::kOk) {
    std::cerr << "packet decode status=" << static_cast<int>(actual_status)
              << '/' << static_cast<int>(expected_status) << '\n';
    return false;
  }
  for (long long item = 0; item < count; ++item) {
    if (std::fabs(actual_values[item] - expected_values[item]) >
        0.03f * (1.0f + std::fabs(expected_values[item]))) {
      std::cerr << "packet mismatch item=" << item
                << " actual=" << actual_values[item]
                << " expected=" << expected_values[item] << '\n';
      return false;
    }
  }
  return true;
}

bool test_grouped_moe_swiglu_quantized_matrix() {
  constexpr long long experts = 2;
  constexpr long long rows = 17;
  constexpr long long n = 64;
  constexpr long long k = 64;
  std::vector<int> expert_ids(rows);
  for (long long row = 0; row < rows; ++row)
    expert_ids[row] = row == 5 ? -1 : static_cast<int>((row * 3 + 1) % 2);
  const std::vector<float> source = source_values(rows * k, 89);

  for (const LayoutCase& weight_layout : kLayouts) {
    std::vector<CanonicalQuantTensor> gate_tensors(experts);
    std::vector<CanonicalQuantTensor> up_tensors(experts);
    std::vector<quixicore_cpu::CpuPackedWeights> gate(experts);
    std::vector<quixicore_cpu::CpuPackedWeights> up(experts);
    for (long long expert = 0; expert < experts; ++expert) {
      const std::vector<float> gate_source =
          source_values(n * k, 91 + static_cast<int>(expert) * 3);
      const std::vector<float> up_source =
          source_values(n * k, 97 + static_cast<int>(expert) * 5);
      REQUIRE(quixicore_cpu::quantize_canonical(
                  {gate_source.data(), FloatStorageType::kF32, n * k}, n, k,
                  weight_layout.layout, weight_layout.group,
                  &gate_tensors[expert],
                  weight_layout.scale_2d) == Status::kOk);
      REQUIRE(quixicore_cpu::quantize_canonical(
                  {up_source.data(), FloatStorageType::kF32, n * k}, n, k,
                  weight_layout.layout, weight_layout.group,
                  &up_tensors[expert], weight_layout.scale_2d) == Status::kOk);
      REQUIRE(gate[expert].prepare(gate_tensors[expert]) == Status::kOk);
      REQUIRE(up[expert].prepare(up_tensors[expert]) == Status::kOk);
    }

    for (FloatStorageType type :
         {FloatStorageType::kF32, FloatStorageType::kF16,
          FloatStorageType::kBF16}) {
      std::vector<std::uint16_t> typed;
      const void* input = source.data();
      if (type != FloatStorageType::kF32) {
        typed.resize(source.size());
        for (std::size_t item = 0; item < source.size(); ++item) {
          typed[item] = type == FloatStorageType::kF16
                            ? quixicore_cpu::float_to_f16(source[item])
                            : quixicore_cpu::float_to_bf16(source[item]);
        }
        input = typed.data();
      }
      std::vector<float> reference_values(rows * n);
      REQUIRE(quixicore_cpu::moe_grouped_prepacked_swiglu_storage(
                  gate.data(), up.data(), experts, {input, type, rows * k},
                  expert_ids.data(),
                  {reference_values.data(), FloatStorageType::kF32, rows * n},
                  rows) == Status::kOk);
      CanonicalQuantTensor expected;
      CanonicalQuantTensor actual;
      REQUIRE(quixicore_cpu::quantize_canonical(
                  {reference_values.data(), FloatStorageType::kF32, rows * n},
                  rows, n, CanonicalQuantLayout::kInt8Symmetric, 32,
                  &expected) == Status::kOk);
      REQUIRE(quixicore_cpu::moe_grouped_prepacked_swiglu_quantized(
                  gate.data(), up.data(), experts, {input, type, rows * k},
                  expert_ids.data(), CanonicalQuantLayout::kInt8Symmetric, 32,
                  &actual, rows) == Status::kOk);
      if (!same_quant_packet(actual, expected)) {
        std::cerr << "quantized swiglu " << weight_layout.name
                  << " storage=" << static_cast<int>(type) << '\n';
        return false;
      }
    }

    std::vector<int> sorted_ids(rows);
    for (long long row = 0; row < rows; ++row)
      sorted_ids[row] = static_cast<int>(row * experts / rows);
    std::vector<float> sorted_reference_values(rows * n);
    CanonicalQuantTensor sorted_expected;
    CanonicalQuantTensor sorted_actual;
    quixicore_cpu::set_num_threads(4);
    REQUIRE(
        quixicore_cpu::moe_grouped_prepacked_swiglu_storage(
            gate.data(), up.data(), experts,
            {source.data(), FloatStorageType::kF32, rows * k},
            sorted_ids.data(),
            {sorted_reference_values.data(), FloatStorageType::kF32, rows * n},
            rows) == Status::kOk);
    REQUIRE(
        quixicore_cpu::quantize_canonical(
            {sorted_reference_values.data(), FloatStorageType::kF32, rows * n},
            rows, n, CanonicalQuantLayout::kInt8Symmetric, 32,
            &sorted_expected) == Status::kOk);
    REQUIRE(quixicore_cpu::moe_grouped_prepacked_swiglu_quantized(
                gate.data(), up.data(), experts,
                {source.data(), FloatStorageType::kF32, rows * k},
                sorted_ids.data(), CanonicalQuantLayout::kInt8Symmetric, 32,
                &sorted_actual, rows) == Status::kOk);
    quixicore_cpu::set_num_threads(1);
    if (!same_quant_packet(sorted_actual, sorted_expected)) {
      std::cerr << "threaded sorted quantized swiglu " << weight_layout.name
                << '\n';
      return false;
    }

    if (weight_layout.layout == CanonicalQuantLayout::kInt4Symmetric) {
      std::vector<float> reference_values(rows * n);
      REQUIRE(quixicore_cpu::moe_grouped_prepacked_swiglu_storage(
                  gate.data(), up.data(), experts,
                  {source.data(), FloatStorageType::kF32, rows * k},
                  expert_ids.data(),
                  {reference_values.data(), FloatStorageType::kF32, rows * n},
                  rows) == Status::kOk);
      for (const LayoutCase& output_layout : kLayouts) {
        if (output_layout.layout == CanonicalQuantLayout::kBitNetTernary)
          continue;
        CanonicalQuantTensor expected;
        CanonicalQuantTensor actual;
        REQUIRE(quixicore_cpu::quantize_canonical(
                    {reference_values.data(), FloatStorageType::kF32, rows * n},
                    rows, n, output_layout.layout, output_layout.group,
                    &expected, output_layout.scale_2d) == Status::kOk);
        REQUIRE(quixicore_cpu::moe_grouped_prepacked_swiglu_quantized(
                    gate.data(), up.data(), experts,
                    {source.data(), FloatStorageType::kF32, rows * k},
                    expert_ids.data(), output_layout.layout,
                    output_layout.group, &actual, rows,
                    output_layout.scale_2d) == Status::kOk);
        if (!same_quant_packet(actual, expected)) {
          std::cerr << "quantized swiglu output " << output_layout.name
                    << " scales=" << actual.scales.size() << '/'
                    << expected.scales.size() << " first="
                    << (actual.scales.empty() ? 0.0f : actual.scales[0]) << '/'
                    << (expected.scales.empty() ? 0.0f : expected.scales[0])
                    << " bytes=" << actual.data.size() << '/'
                    << expected.data.size() << '\n';
          return false;
        }
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!test_serving_matrix() || !test_validation() || !test_lm_head_matrix() ||
      !test_lm_head_structured() || !test_lm_head_ties() ||
      !test_grouped_moe_matrix() || !test_grouped_moe_quantized_matrix())
    return 1;
  if (!test_grouped_moe_swiglu_quantized_matrix()) return 1;
  std::cout << "canonical quantized serving matrix tests passed\n";
  return 0;
}
