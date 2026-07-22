#include "quixicore_cpu/backend.h"

#include <algorithm>
#include <array>

namespace quixicore_cpu {
namespace {

constexpr bool kTargetX86_64 =
#if defined(__x86_64__) || defined(_M_X64)
    true;
#else
    false;
#endif

constexpr bool kTargetAArch64 =
#if defined(__aarch64__) || defined(_M_ARM64)
    true;
#else
    false;
#endif

constexpr bool kFeatureAvx2 =
#if defined(__AVX2__)
    true;
#else
    false;
#endif

constexpr bool kFeatureAvx512f =
#if defined(__AVX512F__)
    true;
#else
    false;
#endif

constexpr bool kFeatureAvx512Vnni =
#if defined(__AVX512VNNI__)
    true;
#else
    false;
#endif

constexpr bool kFeatureAmxTile =
#if defined(__AMX_TILE__)
    true;
#else
    false;
#endif

constexpr bool kFeatureAmxInt8 =
#if defined(__AMX_INT8__)
    true;
#else
    false;
#endif

constexpr bool kFeatureAmxBf16 =
#if defined(__AMX_BF16__)
    true;
#else
    false;
#endif

constexpr bool kFeatureNeon =
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    true;
#else
    false;
#endif

constexpr bool kFeatureArmDotprod =
#if defined(__ARM_FEATURE_DOTPROD)
    true;
#else
    false;
#endif

constexpr bool kFeatureArmI8mm =
#if defined(__ARM_FEATURE_MATMUL_INT8)
    true;
#else
    false;
#endif

constexpr bool kFeatureSve =
#if defined(__ARM_FEATURE_SVE)
    true;
#else
    false;
#endif

constexpr std::array<std::string_view, 16> kPlannedKernelFamilies = {
    "norms",
    "softmax",
    "activations",
    "causal_attention",
    "paged_attention",
    "mla_decode",
    "quant_gemv",
    "quant_gemm",
    "quantized_lm_head",
    "sampling",
    "beam_search",
    "speculative_decode",
    "mamba_ssd",
    "moe_routing",
    "grouped_moe_gemm",
    "optimizers",
};

}  // namespace

BackendMetadata backend_metadata() {
  return {
      "cpu",
      "QuixiCore CPU",
      "QuixiAI/QuixiCore-CPU",
      "QuixiAI/QuixiCore",
      "v0.1",
      "experimental",
      {"x86_64", "aarch64"},
      {"SIMD", "threading"},
  };
}

std::vector<std::string_view> supported_kernel_families() { return {}; }

std::vector<std::string_view> planned_kernel_families() {
  return {kPlannedKernelFamilies.begin(), kPlannedKernelFamilies.end()};
}

std::vector<FeatureHint> compile_time_feature_hints() {
  return {
      {"x86_64", kTargetX86_64, "compiler target"},
      {"aarch64", kTargetAArch64, "compiler target"},
      {"avx2", kFeatureAvx2, "compiler target macro"},
      {"avx512f", kFeatureAvx512f, "compiler target macro"},
      {"avx512_vnni", kFeatureAvx512Vnni, "compiler target macro"},
      {"amx_tile", kFeatureAmxTile, "compiler target macro"},
      {"amx_int8", kFeatureAmxInt8, "compiler target macro"},
      {"amx_bf16", kFeatureAmxBf16, "compiler target macro"},
      {"neon", kFeatureNeon, "compiler target macro"},
      {"arm_dotprod", kFeatureArmDotprod, "compiler target macro"},
      {"arm_i8mm", kFeatureArmI8mm, "compiler target macro"},
      {"sve", kFeatureSve, "compiler target macro"},
  };
}

bool is_kernel_family_supported(std::string_view family) {
  const auto supported = supported_kernel_families();
  return std::find(supported.begin(), supported.end(), family) !=
         supported.end();
}

}  // namespace quixicore_cpu
