#pragma once

#include <cstdint>

#include "quixicore_cpu/status.h"

namespace quixicore_cpu {

// Portable f32 operation surface shared by the sibling backends. Layouts are
// row-major and all reductions accumulate in f32 or better.

enum class GeluApprox { kErf, kTanh };
enum class GluMode { kSwiGlu, kGeGlu, kReGlu, kGlu };

Status gelu(const float* x, float* y, long long count,
            GeluApprox approx = GeluApprox::kErf);
Status gelu_backward(const float* grad_out, const float* x, float* grad_in,
                     long long count,
                     GeluApprox approx = GeluApprox::kErf);
Status silu(const float* x, float* y, long long count);

// x is [rows, 2*dim] (gate half followed by value half); y is [rows, dim].
Status glu(const float* x, float* y, long long rows, long long dim,
           GluMode mode = GluMode::kSwiGlu);

// Numerically stable row-wise softmax over [rows, dim]. x and y may alias.
Status softmax(const float* x, float* y, long long rows, long long dim);

// Row-wise LayerNorm over [rows, hidden]. bias may be null.
Status layer_norm(const float* x, const float* weight, const float* bias,
                  float* y, long long rows, long long hidden,
                  float eps = 1e-5f);

// C[M,N] = A[M,K] @ B[K,N]. Inputs and output are row-major.
Status dense_gemm(const float* a, const float* b, float* c, long long m,
                  long long n, long long k);

// NeoX half-split RoPE over [tokens, heads, head_dim].
Status rope(const float* x, float* y, long long tokens, long long heads,
            long long head_dim, float base, long long pos0);

// Scaled dot-product GQA. Q [Hq,Sq,D], K/V [Hkv,Sk,D], O [Hq,Sq,D].
Status attention(const float* q, const float* k, const float* v, float* out,
                 long long query_heads, long long kv_heads,
                 long long query_length, long long kv_length,
                 long long head_dim, bool causal);

// Paged decode attention. q/out [B,Hq,D], cache [blocks,page,Hkv,D],
// block_table [B,max_blocks], context_lens [B]. window<=0 means full context.
Status paged_attention(const float* q, const float* key_cache,
                       const float* value_cache, const int* block_table,
                       const int* context_lens, float* out,
                       long long cache_blocks, long long batch,
                       long long query_heads, long long kv_heads,
                       long long head_dim, long long page_size,
                       long long max_blocks, float scale = 0.0f,
                       long long window = 0);

// MLA latent decode. q [B,H,W], cache [blocks,page,W], out [B,H,value_dim].
// Scores use all W=latent_dim+rope_dim values; values use the latent prefix.
Status mla_decode(const float* q, const float* kv_cache,
                  const int* block_table, const int* context_lens, float* out,
                  long long cache_blocks, long long batch, long long heads,
                  long long latent_dim, long long rope_dim, long long page_size,
                  long long max_blocks, float scale = 0.0f);

// Sampling is exactly reproducible for a fixed seed and row.
Status argmax_sample(const float* logits, int* out, long long rows,
                     long long vocab);
Status sample_categorical(const float* logits, int* out, long long rows,
                          long long vocab, float temperature = 1.0f,
                          std::uint32_t seed = 0);
Status top_k_sample(const float* logits, int* out, long long rows,
                    long long vocab, int k, float temperature = 1.0f,
                    std::uint32_t seed = 0);
Status top_p_sample(const float* logits, int* out, long long rows,
                    long long vocab, float p, float temperature = 1.0f,
                    std::uint32_t seed = 0);
Status min_p_sample(const float* logits, int* out, long long rows,
                    long long vocab, float min_p, float temperature = 1.0f,
                    std::uint32_t seed = 0);

// One beam-search step. logits [B*beam,V], cumulative [B,beam].
Status beam_search_step(const float* logits, const float* cumulative,
                        int* next_token, int* parent_beam,
                        float* next_cumulative, long long batch,
                        long long beam, long long vocab);

// vLLM-style linear speculative verification. out_tokens [B,S+1] is -1
// padded; accepted_count [B] excludes the recovered/bonus token.
Status speculative_verify(const int* draft_tokens, const float* draft_probs,
                          const float* target_probs, const int* bonus_tokens,
                          const float* accept_uniforms, int* out_tokens,
                          int* accepted_count, long long batch,
                          long long draft_length, long long vocab,
                          std::uint32_t seed = 0);

Status embedding_lookup(const float* table, const int* ids, float* out,
                        long long vocab, long long count, long long dim);
Status kv_cache_scatter(float* cache, const float* src, const int* slots,
                        long long max_slots, long long count,
                        long long row_width);
Status kv_cache_gather(const float* cache, const int* indices, float* out,
                       long long max_slots, long long count,
                       long long row_width);

Status dropout(const float* x, float* y, long long count, float probability,
               std::uint32_t seed);
Status cross_entropy(const float* logits, const int* target, float* loss,
                     long long rows, long long vocab);
Status hadamard(const float* x, float* y, long long rows, long long dim);

// MoE top-k uses lowest expert id to break score ties and normalizes selected
// logits with softmax. Outputs are [tokens, top_k].
Status moe_route_topk(const float* router_logits, int* expert_ids,
                      float* expert_weights, long long tokens,
                      long long experts, int top_k);

// A [groups,M,K], B [groups,K,N], C [groups,M,N], all row-major.
Status grouped_gemm(const float* a, const float* b, float* c,
                    long long groups, long long m, long long n, long long k);

// Non-causal linear attention, tensors [heads,seq,dim].
Status linear_attention(const float* q, const float* k, const float* v,
                        float* out, long long heads, long long sequence,
                        long long dim, float eps = 1e-6f);

// Mamba S6 selective scan. u/delta/y [channels,seq], A [channels,state],
// B/C [seq,state], D [channels].
Status selective_scan(const float* u, const float* delta, const float* a,
                      const float* b, const float* c, const float* d, float* y,
                      long long channels, long long sequence,
                      long long state_size);

// Fused AdamW, in-place, with a 1-based step and bias correction.
Status adamw(float* parameters, const float* gradients, float* first_moment,
             float* second_moment, long long count, float learning_rate,
             float beta1, float beta2, float eps, float weight_decay,
             long long step);

}  // namespace quixicore_cpu
