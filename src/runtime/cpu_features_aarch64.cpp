// AArch64 runtime feature detection.
//
// Each OS exposes CPU features differently:
//   macOS   sysctlbyname("hw.optional.arm.FEAT_*")
//   Linux   getauxval(AT_HWCAP / AT_HWCAP2)
//   Windows IsProcessorFeaturePresent(PF_ARM_*)
//
// This file is preprocessor-guarded rather than excluded by the build system
// so that multi-architecture builds (e.g. macOS universal binaries) compile
// every translation unit for every slice without error.

#if defined(__aarch64__) || defined(_M_ARM64)

#include "quixicore_cpu/cpu_features.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#if __has_include(<asm/hwcap.h>)
#include <asm/hwcap.h>
#endif
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace quixicore_cpu::detail {
namespace {

#if defined(__APPLE__)
bool sysctl_flag(const char* name) {
  int value = 0;
  size_t size = sizeof(value);
  if (sysctlbyname(name, &value, &size, nullptr, 0) != 0) {
    return false;  // Absent key means the feature is not implemented.
  }
  return value != 0;
}
#endif

}  // namespace

CpuFeatures detect_cpu_features() {
  CpuFeatures f;

  // Advanced SIMD is architecturally baseline for AArch64 on every target OS.
  f.neon = true;

#if defined(__APPLE__)
  f.fp16 = sysctl_flag("hw.optional.arm.FEAT_FP16");
  f.dotprod = sysctl_flag("hw.optional.arm.FEAT_DotProd");
  f.i8mm = sysctl_flag("hw.optional.arm.FEAT_I8MM");
  f.sve = sysctl_flag("hw.optional.arm.FEAT_SVE");
  f.sme = sysctl_flag("hw.optional.arm.FEAT_SME");
  f.sme2 = sysctl_flag("hw.optional.arm.FEAT_SME2");
#elif defined(__linux__)
  const unsigned long hwcap = getauxval(AT_HWCAP);
  const unsigned long hwcap2 = getauxval(AT_HWCAP2);
  (void)hwcap;
  (void)hwcap2;
  // Guarded per-macro: old libc headers lack newer HWCAP bits, and a feature
  // the header cannot name is reported unavailable rather than guessed.
#if defined(HWCAP_ASIMDDP)
  f.dotprod = (hwcap & HWCAP_ASIMDDP) != 0;
#endif
#if defined(HWCAP_ASIMDHP)
  f.fp16 = (hwcap & HWCAP_ASIMDHP) != 0;
#endif
#if defined(HWCAP_SVE)
  f.sve = (hwcap & HWCAP_SVE) != 0;
#endif
#if defined(HWCAP2_SVE2)
  f.sve2 = (hwcap2 & HWCAP2_SVE2) != 0;
#endif
#if defined(HWCAP2_I8MM)
  f.i8mm = (hwcap2 & HWCAP2_I8MM) != 0;
#endif
#if defined(HWCAP2_SME)
  f.sme = (hwcap2 & HWCAP2_SME) != 0;
#endif
#if defined(HWCAP2_SME2)
  f.sme2 = (hwcap2 & HWCAP2_SME2) != 0;
#endif
#elif defined(_WIN32)
#if defined(PF_ARM_V82_FP16_INSTRUCTIONS_AVAILABLE)
  f.fp16 =
      IsProcessorFeaturePresent(PF_ARM_V82_FP16_INSTRUCTIONS_AVAILABLE) != 0;
#endif
#if defined(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)
  f.dotprod =
      IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) != 0;
#endif
#if defined(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE)
  f.sve = IsProcessorFeaturePresent(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE) != 0;
#endif
#if defined(PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE)
  f.sve2 = IsProcessorFeaturePresent(PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE) != 0;
#endif
  // Windows has no query for NEON-only I8MM or SME yet. SVE-I8MM implies
  // FEAT_I8MM, so use it when the SDK names it; otherwise under-report.
#if defined(PF_ARM_SVE_I8MM_INSTRUCTIONS_AVAILABLE)
  f.i8mm =
      IsProcessorFeaturePresent(PF_ARM_SVE_I8MM_INSTRUCTIONS_AVAILABLE) != 0;
#endif
#endif

  return f;
}

}  // namespace quixicore_cpu::detail

#endif  // aarch64
