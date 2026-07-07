// x86_64 runtime feature detection via CPUID and XGETBV.
//
// This file is preprocessor-guarded rather than excluded by the build system
// so that multi-architecture builds (e.g. macOS universal binaries) compile
// every translation unit for every slice without error.

#if defined(__x86_64__) || defined(_M_X64)

#include <cstdint>

#include "quixicore_cpu/cpu_features.h"

#if defined(_MSC_VER)
#include <immintrin.h>
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace quixicore_cpu::detail {
namespace {

struct CpuidRegs {
  uint32_t eax = 0;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;
};

CpuidRegs cpuid(uint32_t leaf, uint32_t subleaf) {
  CpuidRegs r;
#if defined(_MSC_VER)
  int regs[4];
  __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
  r.eax = static_cast<uint32_t>(regs[0]);
  r.ebx = static_cast<uint32_t>(regs[1]);
  r.ecx = static_cast<uint32_t>(regs[2]);
  r.edx = static_cast<uint32_t>(regs[3]);
#else
  __get_cpuid_count(leaf, subleaf, &r.eax, &r.ebx, &r.ecx, &r.edx);
#endif
  return r;
}

uint64_t xgetbv0() {
#if defined(_MSC_VER)
  return _xgetbv(0);
#else
  // Inline asm instead of _xgetbv() so this file does not need -mxsave.
  uint32_t eax = 0;
  uint32_t edx = 0;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
}

bool bit(uint32_t value, uint32_t index) { return (value >> index) & 1u; }

// XCR0 state components the OS must enable before the ISA is usable.
constexpr uint64_t kXcr0Avx = 0x6;      // XMM + YMM
constexpr uint64_t kXcr0Avx512 = 0xE6;  // + opmask, ZMM_Hi256, Hi16_ZMM
constexpr uint64_t kXcr0Amx = 0x60000;  // XTILECFG + XTILEDATA

}  // namespace

CpuFeatures detect_cpu_features() {
  CpuFeatures f;

  const CpuidRegs leaf0 = cpuid(0, 0);
  if (leaf0.eax < 1) {
    return f;
  }

  const CpuidRegs leaf1 = cpuid(1, 0);
  const bool osxsave = bit(leaf1.ecx, 27);
  const uint64_t xcr0 = osxsave ? xgetbv0() : 0;
  const bool os_avx = (xcr0 & kXcr0Avx) == kXcr0Avx;
  const bool os_avx512 = (xcr0 & kXcr0Avx512) == kXcr0Avx512;
  const bool os_amx = (xcr0 & kXcr0Amx) == kXcr0Amx;

  f.fma = os_avx && bit(leaf1.ecx, 12);

  if (leaf0.eax < 7) {
    return f;
  }

  const CpuidRegs leaf7 = cpuid(7, 0);
  f.avx2 = os_avx && bit(leaf7.ebx, 5);
  f.avx512f = os_avx512 && bit(leaf7.ebx, 16);
  f.avx512_vnni = f.avx512f && bit(leaf7.ecx, 11);

  // AMX here means CPUID + XSAVE support. On Linux a process must still
  // request XTILE_DATA permission (arch_prctl ARCH_REQ_XCOMP_PERM) before
  // executing tile instructions; the dispatch layer owns that request.
  f.amx_bf16 = os_amx && bit(leaf7.edx, 22);
  f.amx_tile = os_amx && bit(leaf7.edx, 24);
  f.amx_int8 = os_amx && bit(leaf7.edx, 25);

  return f;
}

}  // namespace quixicore_cpu::detail

#endif  // x86_64
