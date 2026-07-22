#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/qgemv.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {

bool close(float actual, float expected) {
  return std::fabs(actual - expected) <=
         3e-4f * std::max(1.0f, std::fabs(expected));
}

bool run(bool sparse) {
  using namespace quixicore_cpu;
  constexpr long long cache_blocks = 3;
  constexpr long long batch = 2;
  constexpr long long heads = 2;
  constexpr long long latent_dim = 32;
  constexpr long long nope_dim = 3;
  constexpr long long rope_dim = 2;
  constexpr long long value_dim = 4;
  constexpr long long page_size = 4;
  constexpr long long max_blocks = 2;
  constexpr long long max_topk = 3;
  constexpr long long weight_rows = heads * (nope_dim + value_dim);
  const int block_table[] = {2, 0, 1, 2};
  const int context_lengths[] = {5, 3};
  const int token_indices[] = {0, 3, 4, 0, 1, 2};
  const int topk_lengths[] = {3, 2};

  std::vector<float> weights(weight_rows * latent_dim);
  std::vector<float> q(batch * heads * (nope_dim + rope_dim));
  std::vector<float> latent(cache_blocks * page_size * latent_dim);
  std::vector<float> rope(cache_blocks * page_size * rope_dim);
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weights[i] = static_cast<float>(static_cast<int>((i * 17 + 3) % 43) - 21) /
                 29.0f;
  }
  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = static_cast<float>(static_cast<int>((i * 13 + 5) % 31) - 15) /
           17.0f;
  }
  for (std::size_t i = 0; i < latent.size(); ++i) {
    latent[i] = static_cast<float>(static_cast<int>((i * 7 + 11) % 37) - 18) /
                23.0f;
  }
  for (std::size_t i = 0; i < rope.size(); ++i) {
    rope[i] = static_cast<float>(static_cast<int>((i * 5 + 1) % 19) - 9) /
              13.0f;
  }
  std::size_t packed_size = 0;
  if (qgemv_packed_size(QuantFormat::kQ4_0, weight_rows, latent_dim,
                        &packed_size) != Status::kOk) {
    return false;
  }
  std::vector<std::uint8_t> packed(packed_size);
  std::vector<float> dequant(weights.size());
  if (qgemv_pack(QuantFormat::kQ4_0, weights.data(), weight_rows, latent_dim,
                 packed.data()) != Status::kOk ||
      qgemv_unpack(QuantFormat::kQ4_0, packed.data(), weight_rows, latent_dim,
                   dequant.data()) != Status::kOk) {
    return false;
  }
  std::vector<float> output(batch * heads * value_dim);
  const Status status = sparse
      ? quantized_mla_decode_absorbed_sparse(
            QuantFormat::kQ4_0, packed.data(), q.data(), latent.data(),
            rope.data(), block_table, token_indices, topk_lengths,
            output.data(), cache_blocks, batch, heads, latent_dim, nope_dim,
            rope_dim, value_dim, page_size, max_blocks, max_topk)
      : quantized_mla_decode_absorbed(
            QuantFormat::kQ4_0, packed.data(), q.data(), latent.data(),
            rope.data(), block_table, context_lengths, output.data(),
            cache_blocks, batch, heads, latent_dim, nope_dim, rope_dim,
            value_dim, page_size, max_blocks);
  if (status != Status::kOk) return false;

  const float factor = 1.0f / std::sqrt(float(nope_dim + rope_dim));
  for (long long request = 0; request < batch; ++request) {
    const int count = sparse ? topk_lengths[request] : context_lengths[request];
    for (long long head = 0; head < heads; ++head) {
      const float* query =
          q.data() + (request * heads + head) * (nope_dim + rope_dim);
      const long long row_base = head * (nope_dim + value_dim);
      std::vector<float> query_latent(latent_dim, 0.0f);
      for (long long d = 0; d < nope_dim; ++d) {
        for (long long l = 0; l < latent_dim; ++l) {
          query_latent[l] += query[d] * dequant[(row_base + d) * latent_dim + l];
        }
      }
      std::vector<double> scores(count);
      double maximum = -std::numeric_limits<double>::infinity();
      for (int item = 0; item < count; ++item) {
        const int position = sparse
            ? token_indices[request * max_topk + item]
            : item;
        const int block = block_table[request * max_blocks + position / page_size];
        const long long cache_row = block * page_size + position % page_size;
        double score = 0.0;
        for (long long l = 0; l < latent_dim; ++l) {
          score += query_latent[l] * latent[cache_row * latent_dim + l];
        }
        for (long long d = 0; d < rope_dim; ++d) {
          score += query[nope_dim + d] * rope[cache_row * rope_dim + d];
        }
        scores[item] = score * factor;
        maximum = std::max(maximum, scores[item]);
      }
      double denominator = 0.0;
      for (double& score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
      }
      std::vector<float> mixed(latent_dim, 0.0f);
      for (int item = 0; item < count; ++item) {
        const int position = sparse
            ? token_indices[request * max_topk + item]
            : item;
        const int block = block_table[request * max_blocks + position / page_size];
        const long long cache_row = block * page_size + position % page_size;
        const float probability = static_cast<float>(scores[item] / denominator);
        for (long long l = 0; l < latent_dim; ++l) {
          mixed[l] += probability * latent[cache_row * latent_dim + l];
        }
      }
      for (long long value = 0; value < value_dim; ++value) {
        float expected = 0.0f;
        for (long long l = 0; l < latent_dim; ++l) {
          expected += dequant[(row_base + nope_dim + value) * latent_dim + l] *
                      mixed[l];
        }
        if (!close(output[(request * heads + head) * value_dim + value],
                   expected)) {
          return false;
        }
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!run(false) || !run(true)) {
    std::cerr << "FAIL: quantized MLA absorption\n";
    return 1;
  }
  float value = 0.0f;
  int index = 0;
  if (quixicore_cpu::quantized_mla_decode_absorbed(
          quixicore_cpu::QuantFormat::kQ4_0, &value, &value, &value, &value,
          &index, &index, &value, 1, 1, 1, 1,
          std::numeric_limits<long long>::max(), 1, 1, 1, 1) !=
      quixicore_cpu::Status::kInvalidShape) {
    std::cerr << "FAIL: MLA derived-dimension overflow guard\n";
    return 1;
  }
  std::cout << "quantized MLA absorption tests passed\n";
  return 0;
}
