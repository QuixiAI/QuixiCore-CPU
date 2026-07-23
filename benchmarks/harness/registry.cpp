#include "harness/case.h"

namespace qcb {

// Case builders, one per benchmarked kernel; defined under cases/.
void build_mem_triad_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_sgemv_naive_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_qgemv_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_qgemv_formats_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_lifecycle_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_import_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_llama_parity_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_rms_norm_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_contract_ops_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_float_storage_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_ported_ops_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_colibri_ops_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_prerequisites_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_optimization_plan_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_formats_matrix_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_activation_matrix_cases(const BuildCtx&,
                                         std::vector<CaseDecl>&);
void build_quant_gemv_matrix_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_gemm_matrix_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_fusions_matrix_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_serving_matrix_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_kv_matrix_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_bitnet_matrix_cases(const BuildCtx&, std::vector<CaseDecl>&);

// Explicit table instead of static-initializer self-registration: no
// init-order or dead-stripping hazards, and --list ordering is stable.
const std::vector<KernelEntry>& kernel_registry() {
  static const std::vector<KernelEntry> registry = {
      {"mem_triad", &build_mem_triad_cases},
      {"sgemv_naive", &build_sgemv_naive_cases},
      {"qgemv", &build_qgemv_cases},
      {"qgemv_formats", &build_qgemv_formats_cases},
      {"quant_lifecycle", &build_quant_lifecycle_cases},
      {"quant_import", &build_quant_import_cases},
      {"llama_parity", &build_llama_parity_cases},
      {"rms_norm", &build_rms_norm_cases},
      {"contract_ops", &build_contract_ops_cases},
      {"float_storage", &build_float_storage_cases},
      {"ported_ops", &build_ported_ops_cases},
      {"colibri_ops", &build_colibri_ops_cases},
      {"prerequisites", &build_prerequisites_cases},
      {"optimization_plan", &build_optimization_plan_cases},
      {"quant_formats", &build_quant_formats_matrix_cases},
      {"quant_activation", &build_quant_activation_matrix_cases},
      {"quant_gemv_matrix", &build_quant_gemv_matrix_cases},
      {"quant_gemm_matrix", &build_quant_gemm_matrix_cases},
      {"quant_fusions", &build_quant_fusions_matrix_cases},
      {"quant_serving", &build_quant_serving_matrix_cases},
      {"quant_kv", &build_quant_kv_matrix_cases},
      {"bitnet_matrix", &build_bitnet_matrix_cases},
  };
  return registry;
}

}  // namespace qcb
