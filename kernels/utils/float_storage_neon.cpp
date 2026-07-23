#if (defined(__aarch64__) || defined(_M_ARM64)) && \
    defined(QUIXICORE_CPU_ISA_FP16)

#include <arm_neon.h>

#include "kernels/common/fp16.h"
#include "kernels/utils/float_storage_isa.h"

namespace quixicore_cpu::float_storage_detail {

void f16_to_f32_neon(const std::uint16_t* input, float* output,
                     long long begin, long long end) {
  long long i = begin;
  for (; i + 7 < end; i += 8) {
    const float16x8_t half = vreinterpretq_f16_u16(vld1q_u16(input + i));
    vst1q_f32(output + i, vcvt_f32_f16(vget_low_f16(half)));
    vst1q_f32(output + i + 4, vcvt_f32_f16(vget_high_f16(half)));
  }
  for (; i < end; ++i) output[i] = fp16_to_fp32(input[i]);
}

void f32_to_f16_neon(const float* input, std::uint16_t* output,
                     long long begin, long long end) {
  long long i = begin;
  for (; i + 7 < end; i += 8) {
    const float16x4_t low = vcvt_f16_f32(vld1q_f32(input + i));
    const float16x4_t high = vcvt_f16_f32(vld1q_f32(input + i + 4));
    vst1q_u16(output + i, vreinterpretq_u16_f16(vcombine_f16(low, high)));
  }
  for (; i < end; ++i) output[i] = fp32_to_fp16(input[i]);
}

}  // namespace quixicore_cpu::float_storage_detail

#endif
