#include "quixicore_cpu/collectives.h"
#include "quixicore_cpu/ops.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace {

bool require(bool condition, const char* message) {
  if (!condition) std::cerr << "FAIL: " << message << '\n';
  return condition;
}

bool close(float lhs, float rhs, float tolerance = 1e-5f) {
  return std::fabs(lhs - rhs) <= tolerance;
}

}  // namespace

int main() {
  using namespace quixicore_cpu;
  bool ok = true;

  const float ranks[] = {1, 2, 3, 4};
  float reduced[4] = {};
  ok &= require(all_reduce_sum(ranks, reduced, 2, 2) == Status::kOk &&
                    close(reduced[0], 4) && close(reduced[1], 6) &&
                    close(reduced[2], 4) && close(reduced[3], 6),
                "all reduce");
  float gathered[8] = {};
  ok &= require(all_gather(ranks, gathered, 2, 2) == Status::kOk &&
                    close(gathered[0], 1) && close(gathered[3], 4) &&
                    close(gathered[4], 1) && close(gathered[7], 4),
                "all gather");
  const float full[] = {1, 2, 3, 4, 10, 20, 30, 40};
  ok &= require(reduce_scatter_sum(full, reduced, 2, 2) == Status::kOk &&
                    close(reduced[0], 11) && close(reduced[1], 22) &&
                    close(reduced[2], 33) && close(reduced[3], 44),
                "reduce scatter");
  ok &= require(all_to_all(full, gathered, 2, 2) == Status::kOk &&
                    close(gathered[0], 1) && close(gathered[2], 10) &&
                    close(gathered[4], 3) && close(gathered[6], 30),
                "all to all");
  ok &= require(broadcast(ranks, reduced, 2, 2, 1) == Status::kOk &&
                    close(reduced[0], 3) && close(reduced[1], 4) &&
                    close(reduced[2], 3) && close(reduced[3], 4),
                "broadcast");
  std::fill_n(reduced, 4, -1.0f);
  ok &= require(reduce_sum(ranks, reduced, 2, 2, 1) == Status::kOk &&
                    close(reduced[2], 4) && close(reduced[3], 6),
                "reduce");

  const float a[] = {1, 2, 3, 4};  // two ranks, 1x2
  const float b[] = {1, 0, 0, 1, 2, 0, 0, 2};  // rank-local 2x2
  float gemm[4] = {};
  ok &= require(gemm_all_reduce(a, b, gemm, 2, 1, 2, 2) == Status::kOk &&
                    close(gemm[0], 7) && close(gemm[1], 10) &&
                    close(gemm[2], 7) && close(gemm[3], 10),
                "gemm all reduce");
  ok &= require(all_gather_gemm(a, b, gathered, 2, 1, 2, 2) == Status::kOk &&
                    close(gathered[0], 1) && close(gathered[1], 2) &&
                    close(gathered[2], 3) && close(gathered[3], 4) &&
                    close(gathered[4], 2) && close(gathered[7], 8),
                "all gather gemm");
  const float rs_a[] = {1, 2, 3, 4};
  const float rs_b[] = {10, 100};
  float rs_out[2];
  ok &= require(gemm_reduce_scatter(rs_a, rs_b, rs_out, 2, 1, 1, 1) ==
                        Status::kOk &&
                    close(rs_out[0], 310) && close(rs_out[1], 420),
                "gemm reduce scatter");

  const float add_x[] = {1, 2};
  const float add_y[] = {3, 4};
  float add_out[2];
  ok &= require(add(add_x, add_y, add_out, 2) == Status::kOk &&
                    close(add_out[0], 4) && close(add_out[1], 6),
                "elementwise add");

  const float logits[] = {0, 1, 2, 3};
  float transformed[4] = {};
  ok &= require(quadratic_transform(logits, transformed, 1, 4, 0.0f) ==
                        Status::kOk &&
                    close(transformed[0], 0) && close(transformed[3], 3),
                "quadratic identity");
  ok &= require(top_nsigma_mask(logits, transformed, 1, 4, 0.0f) ==
                        Status::kOk &&
                    std::isinf(transformed[0]) && close(transformed[3], 3),
                "top nsigma");
  ok &= require(top_a_mask(logits, transformed, 1, 4, 0.5f) == Status::kOk &&
                    close(transformed[3], 3),
                "top a");
  ok &= require(epsilon_cutoff_mask(logits, transformed, 1, 4, 0.2f) ==
                        Status::kOk &&
                    std::isinf(transformed[0]) && close(transformed[3], 3),
                "epsilon cutoff");
  ok &= require(eta_cutoff_mask(logits, transformed, 1, 4, 0.2f) ==
                    Status::kOk,
                "eta cutoff");
  ok &= require(xtc_mask(logits, transformed, 1, 4, 0.1f, 1.0f, 3) ==
                        Status::kOk &&
                    std::isinf(transformed[3]) &&
                    std::isfinite(transformed[1]),
                "xtc mask");
  const float probabilities[] = {0.25f, 0.75f};
  ok &= require(skew_transform(probabilities, transformed, 1, 2, 0.0f) ==
                        Status::kOk &&
                    close(transformed[0], 0.25f) &&
                    close(transformed[1], 0.75f),
                "skew identity");
  const int previous[] = {1, 2, 1};
  const int lengths[] = {3};
  ok &= require(no_repeat_ngram_mask(logits, previous, lengths, transformed,
                                     1, 4, 3, 2) == Status::kOk &&
                    std::isinf(transformed[2]),
                "no repeat ngram");
  const int breakers[] = {99};
  ok &= require(dry_penalty(logits, previous, lengths, breakers, transformed,
                            1, 4, 3, 1, 0.5f) == Status::kOk,
                "dry penalty");

  if (!ok) return 1;
  std::cout << "collective and sampler transform tests passed\n";
  return 0;
}
