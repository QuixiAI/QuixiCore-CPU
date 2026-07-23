#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "quixicore_cpu/ops.h"
#include "quixicore_cpu/status.h"

namespace quixicore_cpu {

// Universal floating-point storage accepted by the CPU fallback layer.
// FP16/BF16 tensors are decoded once, accumulated by the existing FP32
// kernels, and rounded once when an output is committed. kF32 is zero-copy.
// The 16-bit representations are the raw IEEE binary16/bfloat16 bits.
enum class FloatStorageType { kF32, kF16, kBF16 };

struct FloatStorageInput {
  const void* data = nullptr;
  FloatStorageType type = FloatStorageType::kF32;
  long long count = 0;
};

struct FloatStorageOutput {
  void* data = nullptr;
  FloatStorageType type = FloatStorageType::kF32;
  long long count = 0;
};

// Reusable conversion arena. One workspace may be used by one invocation at
// a time. Keeping it at the model/executor level removes allocation after the
// largest observed operation has warmed the arena.
class FloatStorageWorkspace {
 public:
  Status reserve(std::size_t float_elements);
  std::size_t capacity() const noexcept { return scratch_.capacity(); }

 private:
  std::vector<float> scratch_;
  friend Status dispatch_float_storage(const FloatStorageInput*, long long,
                                       const FloatStorageOutput*, long long,
                                       Status (*)(const float* const*,
                                                  float* const*, void*),
                                       void*, FloatStorageWorkspace*);
};

using Float32StorageKernel = Status (*)(const float* const* inputs,
                                        float* const* outputs,
                                        void* context);

// Adapt arbitrary floating storage to any FP32 kernel. All inputs are decoded
// before the callback and outputs are encoded only when it returns kOk.
// Exact input/output aliases with identical type and count are in-place. Any
// partial overlap, or an exact alias described with different metadata, is
// rejected because it cannot preserve the FP32 kernel's alias contract. FP16
// and BF16 outputs commit only after success; zero-copy FP32 outputs retain the
// wrapped kernel's normal failure/alias behavior.
Status dispatch_float_storage(const FloatStorageInput* inputs,
                              long long input_count,
                              const FloatStorageOutput* outputs,
                              long long output_count,
                              Float32StorageKernel kernel, void* context,
                              FloatStorageWorkspace* workspace = nullptr);

// C++ convenience for wrapping any existing FP32 operation without declaring
// a separate context struct. The callable receives (inputs, outputs) and must
// return Status. Dispatch is synchronous, so temporary lambdas are safe.
template <class Kernel>
Status with_float_storage(const FloatStorageInput* inputs,
                          long long input_count,
                          const FloatStorageOutput* outputs,
                          long long output_count, Kernel&& kernel,
                          FloatStorageWorkspace* workspace = nullptr) {
  using Callable = std::remove_reference_t<Kernel>;
  return dispatch_float_storage(
      inputs, input_count, outputs, output_count,
      [](const float* const* f32_inputs, float* const* f32_outputs,
         void* opaque) -> Status {
        return (*static_cast<Callable*>(opaque))(f32_inputs, f32_outputs);
      },
      &kernel, workspace);
}

std::uint16_t float_to_f16(float value) noexcept;
float f16_to_float(std::uint16_t bits) noexcept;
std::uint16_t float_to_bf16(float value) noexcept;
float bf16_to_float(std::uint16_t bits) noexcept;

Status float_storage_to_f32(FloatStorageType type, const void* input,
                            float* output, long long count);
Status float_storage_from_f32(FloatStorageType type, const float* input,
                              void* output, long long count);
const char* float_storage_variant(FloatStorageType type) noexcept;

// Typed convenience routes for the dominant model kernels. Individual inputs
// and outputs may use different storage types; all reductions remain FP32.
Status unary_storage(FloatStorageInput x, FloatStorageOutput y, UnaryOp op,
                     XiEluParams xielu = {});
Status softmax_storage(FloatStorageInput x, FloatStorageOutput y,
                       long long rows, long long dim);
Status rms_norm_storage(FloatStorageInput x, FloatStorageInput weight,
                        FloatStorageOutput y, long long rows,
                        long long hidden, float eps = 1e-5f);
Status dense_gemm_storage(FloatStorageInput a, FloatStorageInput b,
                          FloatStorageOutput c, long long m, long long n,
                          long long k);
Status attention_storage(FloatStorageInput q, FloatStorageInput k,
                         FloatStorageInput v, FloatStorageOutput out,
                         long long query_heads, long long kv_heads,
                         long long query_length, long long kv_length,
                         long long head_dim, bool causal);

enum class QuantFormat;
Status qgemv_storage(QuantFormat format, const void* packed,
                     FloatStorageInput x, FloatStorageOutput y, long long n,
                     long long k);
Status qgemm_storage(QuantFormat format, const void* packed,
                     FloatStorageInput x, FloatStorageOutput y, long long m,
                     long long n, long long k);

}  // namespace quixicore_cpu
