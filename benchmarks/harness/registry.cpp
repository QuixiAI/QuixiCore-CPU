#include "harness/case.h"

namespace qcb {

// Case builders, one per benchmarked kernel; defined under cases/.
void build_mem_triad_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_norm_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_embedding_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_lm_head_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_moe_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_base_q_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_basert_aux_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_basert_embedding_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_basert_vision_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_basert_audio_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_sgemv_naive_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_qgemv_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_qgemv_formats_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_lifecycle_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_import_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_cache_attention_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_llama_parity_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_rms_norm_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_contract_ops_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_float_storage_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_gdn_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_lora_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_ported_ops_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_colibri_ops_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_prerequisites_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_q8_kv_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_rotary_extended_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_optimization_plan_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_formats_matrix_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_activation_matrix_cases(const BuildCtx&,
                                         std::vector<CaseDecl>&);
void build_quant_gemv_matrix_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_gemm_matrix_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_fusions_matrix_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_gate_up_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_swiglu_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_swiglu_quant_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_qkv_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_qkv_rope_kv_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_serving_matrix_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_quant_kv_matrix_cases(const BuildCtx&, std::vector<CaseDecl>&);
void build_bitnet_matrix_cases(const BuildCtx&, std::vector<CaseDecl>&);

// Explicit table instead of static-initializer self-registration: no
// init-order or dead-stripping hazards, and --list ordering is stable.
const std::vector<KernelEntry>& kernel_registry() {
  static const std::vector<KernelEntry> registry = {
      {"mem_triad", &build_mem_triad_cases},
      {"quant_norm", &build_quant_norm_cases},
      {"quant_embedding", &build_quant_embedding_cases},
      {"quant_lm_head", &build_quant_lm_head_cases},
      {"quant_moe", &build_quant_moe_cases},
      {"base_q", &build_base_q_cases},
      {"basert_aux", &build_basert_aux_cases},
      {"basert_embedding", &build_basert_embedding_cases},
      {"basert_vision", &build_basert_vision_cases},
      {"basert_audio", &build_basert_audio_cases},
      {"sgemv_naive", &build_sgemv_naive_cases},
      {"qgemv", &build_qgemv_cases},
      {"qgemv_formats", &build_qgemv_formats_cases},
      {"quant_lifecycle", &build_quant_lifecycle_cases},
      {"quant_import", &build_quant_import_cases},
      {"quant_cache_attention", &build_quant_cache_attention_cases},
      {"llama_parity", &build_llama_parity_cases},
      {"rms_norm", &build_rms_norm_cases},
      {"contract_ops", &build_contract_ops_cases},
      {"float_storage", &build_float_storage_cases},
      {"gdn", &build_gdn_cases},
      {"lora", &build_lora_cases},
      {"ported_ops", &build_ported_ops_cases},
      {"colibri_ops", &build_colibri_ops_cases},
      {"prerequisites", &build_prerequisites_cases},
      {"q8_kv", &build_q8_kv_cases},
      {"rotary_extended", &build_rotary_extended_cases},
      {"optimization_plan", &build_optimization_plan_cases},
      {"quant_formats", &build_quant_formats_matrix_cases},
      {"quant_activation", &build_quant_activation_matrix_cases},
      {"quant_gemv_matrix", &build_quant_gemv_matrix_cases},
      {"quant_gemm_matrix", &build_quant_gemm_matrix_cases},
      {"quant_fusions", &build_quant_fusions_matrix_cases},
      {"quant_gate_up", &build_quant_gate_up_cases},
      {"quant_swiglu", &build_quant_swiglu_cases},
      {"quant_swiglu_quant", &build_quant_swiglu_quant_cases},
      {"quant_qkv", &build_quant_qkv_cases},
      {"quant_qkv_rope_kv", &build_quant_qkv_rope_kv_cases},
      {"quant_serving", &build_quant_serving_matrix_cases},
      {"quant_kv", &build_quant_kv_matrix_cases},
      {"bitnet_matrix", &build_bitnet_matrix_cases},
  };
  return registry;
}

}  // namespace qcb
