#pragma once

#include <cstddef>
#include <cstdint>

#include "quixicore_cpu/status.h"

namespace quixicore_cpu {

// Compact row-major weight layouts used by the CPU low-batch kernels. These
// are deliberately separate from QuantFormat: they are not GGUF containers.
// INT2/INT4 store adjacent values in increasing nibble/bit order with an
// implicit -2/-8 offset. INT3 uses one 24-byte dual-plane block and one f32
// scale per 64 inputs.
enum class LowBitFormat {
  kInt2Row,
  kInt3Group64,
  kInt4Row,
  kInt4Group,
};

// Returns the weight payload size and number of f32 scales. group_size is used
// only by kInt4Group; zero selects one group per row for the row formats.
Status lowbit_packed_size(LowBitFormat format, long long rows, long long dim,
                          long long group_size, std::size_t* weight_bytes,
                          std::size_t* scale_count);

Status lowbit_pack(LowBitFormat format, const float* weights,
                   std::uint8_t* packed, float* scales, long long rows,
                   long long dim, long long group_size = 0);
Status lowbit_unpack(LowBitFormat format, const std::uint8_t* packed,
                     const float* scales, float* weights, long long rows,
                     long long dim, long long group_size = 0);

// Y[M,N] = X[M,K] @ dequantize(W[N,K])^T. Accumulation and output are f32.
Status lowbit_gemm(LowBitFormat format, const std::uint8_t* packed_weights,
                   const float* weight_scales, const float* x, float* y,
                   long long m, long long n, long long k,
                   long long group_size = 0);

// Decode-oriented W4A8 route for kInt4Row. Activations are dynamically
// quantized once per input row to symmetric INT8, then multiplied with an
// exact integer dot and combined row scales.
Status lowbit_gemm_w8a8(const std::uint8_t* packed_weights,
                        const float* weight_scales, const float* x, float* y,
                        long long m, long long n, long long k);
const char* lowbit_gemm_w8a8_variant();

// One scheduling pass over two projections. This is the gate/up seam used by
// decode and MoE MLPs; both outputs are [M,N].
Status lowbit_gemm_pair(
    LowBitFormat format, const std::uint8_t* packed_gate,
    const float* gate_scales, const std::uint8_t* packed_up,
    const float* up_scales, const float* x, float* gate, float* up,
    long long m, long long n, long long k, long long group_size = 0);

// Packed-row primitives avoid materializing a complete dequantized matrix.
// lowbit_axpy_row performs out[K] += coefficient * W[row,:].
Status lowbit_axpy_row(LowBitFormat format,
                       const std::uint8_t* packed_weights,
                       const float* weight_scales, long long row,
                       float coefficient, float* out, long long n,
                       long long k, long long group_size = 0);
Status lowbit_gemv_rows(LowBitFormat format,
                        const std::uint8_t* packed_weights,
                        const float* weight_scales, const float* x,
                        const int* rows, float* y, long long row_count,
                        long long n, long long k,
                        long long group_size = 0);

// Runtime-selected implementation name for diagnostics and benchmarks.
const char* lowbit_gemm_variant(LowBitFormat format);

// E8/IQ3 lattice blocks: 256 weights in 98 bytes. This is the rotated E8
// layout ([64 grid indices][8 sign/scale words][fp16 super-scale]), not the
// byte-incompatible GGUF IQ3_XXS layout. Packing is a conversion-time kernel
// and requires dim to be a multiple of 256; decode remains tail-tolerant for
// defensive consumption of externally produced containers.
Status e8iq3_packed_size(long long rows, long long dim,
                         std::size_t* weight_bytes);
Status e8iq3_pack(const float* weights, std::uint8_t* packed,
                  long long rows, long long dim);
Status e8iq3_unpack(const std::uint8_t* packed, float* weights,
                    long long rows, long long dim);
Status e8iq3_gemm(const std::uint8_t* packed_weights, const float* x,
                  float* y, long long m, long long n, long long k);

// Deterministic block-diagonal signed FWHT used by E8 weights and activations.
// Forward is Q^T x (sign then FWHT); inverse is Q x (FWHT then sign). The
// operation supports arbitrary positive dimensions through power-of-two
// blocks and permits exact in-place use.
Status e8iq3_rotate(const float* x, float* y, long long rows, long long dim,
                    bool inverse = false);

}  // namespace quixicore_cpu
