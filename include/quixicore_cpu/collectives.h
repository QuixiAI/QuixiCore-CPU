#pragma once

#include "quixicore_cpu/status.h"

namespace quixicore_cpu {

// Host-reference collective contracts. Every rank participates in one call;
// rank-major buffers make the cross-rank data movement explicit and allow the
// CPU backend to validate distributed algorithms without a transport runtime.
// all_reduce input/output: [world,count], replicated sum on every rank.
Status all_reduce_sum(const float* input, float* output, long long world,
                      long long count);
// all_gather input: [world,count], output: [world,world,count].
Status all_gather(const float* input, float* output, long long world,
                  long long count);
// reduce_scatter input: [source_rank,destination_rank,count], output:
// [destination_rank,count].
Status reduce_scatter_sum(const float* input, float* output, long long world,
                          long long count);
// all_to_all transposes rank axes from [source,destination,count] to
// [destination,source,count].
Status all_to_all(const float* input, float* output, long long world,
                  long long count);
Status broadcast(const float* input, float* output, long long world,
                 long long count, int root);
Status reduce_sum(const float* input, float* output, long long world,
                  long long count, int root);

// Fused reference seams corresponding to the CUDA/ROCm parallel kernels.
// GEMM-AR: A[world,m,k], B[world,k,n] -> replicated C[world,m,n].
Status gemm_all_reduce(const float* a, const float* b, float* output,
                       long long world, long long m, long long n,
                       long long k);
// AG-GEMM: local A[world,m,k], B[world,k,n] ->
// output[destination,source*m+row,n].
Status all_gather_gemm(const float* a, const float* b, float* output,
                       long long world, long long m, long long n,
                       long long k);
// GEMM-RS: A[source,world*m,k], B[source,k,n] -> output[rank,m,n].
Status gemm_reduce_scatter(const float* a, const float* b, float* output,
                           long long world, long long m, long long n,
                           long long k);

}  // namespace quixicore_cpu
