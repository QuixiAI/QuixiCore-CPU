#include "harness/case.h"

namespace qcb {

// Case builders, one per benchmarked kernel; defined under cases/.
void build_mem_triad_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_sgemv_naive_cases(const BuildCtx&, std::vector<CaseDecl>&);

// Explicit table instead of static-initializer self-registration: no
// init-order or dead-stripping hazards, and --list ordering is stable.
const std::vector<KernelEntry>& kernel_registry() {
  static const std::vector<KernelEntry> registry = {
      {"mem_triad", &build_mem_triad_cases},
      {"sgemv_naive", &build_sgemv_naive_cases},
  };
  return registry;
}

}  // namespace qcb
