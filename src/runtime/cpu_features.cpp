#include "quixicore_cpu/cpu_features.h"

namespace quixicore_cpu {

namespace detail {
CpuFeatures detect_cpu_features();
}  // namespace detail

#if !(defined(__x86_64__) || defined(_M_X64)) && \
    !(defined(__aarch64__) || defined(_M_ARM64))
namespace detail {
CpuFeatures detect_cpu_features() { return {}; }
}  // namespace detail
#endif

const CpuFeatures& cpu_features() {
  static const CpuFeatures features = detail::detect_cpu_features();
  return features;
}

std::vector<FeatureHint> runtime_feature_hints() {
  const CpuFeatures& f = cpu_features();
  return {
      {"fma", f.fma, "runtime detection"},
      {"f16c", f.f16c, "runtime detection"},
      {"avx2", f.avx2, "runtime detection"},
      {"avx512f", f.avx512f, "runtime detection"},
      {"avx512_vnni", f.avx512_vnni, "runtime detection"},
      {"amx_tile", f.amx_tile, "runtime detection"},
      {"amx_int8", f.amx_int8, "runtime detection"},
      {"amx_bf16", f.amx_bf16, "runtime detection"},
      {"neon", f.neon, "runtime detection"},
      {"arm_fp16", f.fp16, "runtime detection"},
      {"arm_dotprod", f.dotprod, "runtime detection"},
      {"arm_i8mm", f.i8mm, "runtime detection"},
      {"sve", f.sve, "runtime detection"},
      {"sve2", f.sve2, "runtime detection"},
      {"sme", f.sme, "runtime detection"},
      {"sme2", f.sme2, "runtime detection"},
  };
}

}  // namespace quixicore_cpu
