#pragma once

// Generated canonical contract include for the cpu backend.
// Native optimized APIs remain backend-owned behind this adapter surface.

#include "quixicore/contract/kernel_abi.hpp"
#include "quixicore/contract/operations.hpp"
#include "quixicore/cpu/contract_stubs.hpp"

namespace quixicore::cpu::contract_api {

[[nodiscard]] inline quixicore::contract::Status dispatch(
    quixicore::contract::OperationId operation,
    const quixicore::contract::KernelCall& call) noexcept {
  switch (operation) {
    case quixicore::contract::OperationId::adaptive_norm_modulation:
      return quixicore::cpu::contract_stubs::adaptive_norm_modulation(call);
    case quixicore::contract::OperationId::adaptive_norm_modulation_qkv:
      return quixicore::cpu::contract_stubs::adaptive_norm_modulation_qkv(call);
    case quixicore::contract::OperationId::adaptive_rms_norm_modulation:
      return quixicore::cpu::contract_stubs::adaptive_rms_norm_modulation(call);
    case quixicore::contract::OperationId::adaptive_rms_norm_modulation_qkv:
      return quixicore::cpu::contract_stubs::adaptive_rms_norm_modulation_qkv(call);
    case quixicore::contract::OperationId::all_gather_gemm:
      return quixicore::cpu::contract_stubs::all_gather_gemm(call);
    case quixicore::contract::OperationId::all_reduce_rms_norm:
      return quixicore::cpu::contract_stubs::all_reduce_rms_norm(call);
    case quixicore::contract::OperationId::attention_gate_residual:
      return quixicore::cpu::contract_stubs::attention_gate_residual(call);
    case quixicore::contract::OperationId::audio_resample:
      return quixicore::cpu::contract_stubs::audio_resample(call);
    case quixicore::contract::OperationId::causal_conv3d_cache_update:
      return quixicore::cpu::contract_stubs::causal_conv3d_cache_update(call);
    case quixicore::contract::OperationId::causal_conv_gate_residual:
      return quixicore::cpu::contract_stubs::causal_conv_gate_residual(call);
    case quixicore::contract::OperationId::classifier_free_guidance_combine:
      return quixicore::cpu::contract_stubs::classifier_free_guidance_combine(call);
    case quixicore::contract::OperationId::classifier_free_guidance_scheduler_step:
      return quixicore::cpu::contract_stubs::classifier_free_guidance_scheduler_step(call);
    case quixicore::contract::OperationId::conv_transpose_overlap_add:
      return quixicore::cpu::contract_stubs::conv_transpose_overlap_add(call);
    case quixicore::contract::OperationId::ctc_decode:
      return quixicore::cpu::contract_stubs::ctc_decode(call);
    case quixicore::contract::OperationId::deformable_attention:
      return quixicore::cpu::contract_stubs::deformable_attention(call);
    case quixicore::contract::OperationId::deterministic_noise:
      return quixicore::cpu::contract_stubs::deterministic_noise(call);
    case quixicore::contract::OperationId::diffusion_scheduler_step:
      return quixicore::cpu::contract_stubs::diffusion_scheduler_step(call);
    case quixicore::contract::OperationId::down_projection_residual:
      return quixicore::cpu::contract_stubs::down_projection_residual(call);
    case quixicore::contract::OperationId::downsample_3d:
      return quixicore::cpu::contract_stubs::downsample_3d(call);
    case quixicore::contract::OperationId::downsample_conv2d:
      return quixicore::cpu::contract_stubs::downsample_conv2d(call);
    case quixicore::contract::OperationId::flow_euler_step:
      return quixicore::cpu::contract_stubs::flow_euler_step(call);
    case quixicore::contract::OperationId::flow_step_action_clamp_denormalize:
      return quixicore::cpu::contract_stubs::flow_step_action_clamp_denormalize(call);
    case quixicore::contract::OperationId::gate_up_projection_swiglu:
      return quixicore::cpu::contract_stubs::gate_up_projection_swiglu(call);
    case quixicore::contract::OperationId::geglu_projection:
      return quixicore::cpu::contract_stubs::geglu_projection(call);
    case quixicore::contract::OperationId::grammar_mask:
      return quixicore::cpu::contract_stubs::grammar_mask(call);
    case quixicore::contract::OperationId::grid_sample:
      return quixicore::cpu::contract_stubs::grid_sample(call);
    case quixicore::contract::OperationId::group_norm_silu:
      return quixicore::cpu::contract_stubs::group_norm_silu(call);
    case quixicore::contract::OperationId::istft:
      return quixicore::cpu::contract_stubs::istft(call);
    case quixicore::contract::OperationId::joint_multimodal_attention:
      return quixicore::cpu::contract_stubs::joint_multimodal_attention(call);
    case quixicore::contract::OperationId::kv_cache_compact:
      return quixicore::cpu::contract_stubs::kv_cache_compact(call);
    case quixicore::contract::OperationId::kv_cache_offload:
      return quixicore::cpu::contract_stubs::kv_cache_offload(call);
    case quixicore::contract::OperationId::log_mel_spectrogram:
      return quixicore::cpu::contract_stubs::log_mel_spectrogram(call);
    case quixicore::contract::OperationId::logits_bias_penalties_top_k_top_p_sample:
      return quixicore::cpu::contract_stubs::logits_bias_penalties_top_k_top_p_sample(call);
    case quixicore::contract::OperationId::mixed_prefill_decode_attention:
      return quixicore::cpu::contract_stubs::mixed_prefill_decode_attention(call);
    case quixicore::contract::OperationId::moe_all_to_all_dispatch:
      return quixicore::cpu::contract_stubs::moe_all_to_all_dispatch(call);
    case quixicore::contract::OperationId::moe_grouped_gemm_swiglu:
      return quixicore::cpu::contract_stubs::moe_grouped_gemm_swiglu(call);
    case quixicore::contract::OperationId::moe_route_top_k_prefix_sum_permute:
      return quixicore::cpu::contract_stubs::moe_route_top_k_prefix_sum_permute(call);
    case quixicore::contract::OperationId::moe_unpermute_weighted_reduce:
      return quixicore::cpu::contract_stubs::moe_unpermute_weighted_reduce(call);
    case quixicore::contract::OperationId::multi_codebook_sample:
      return quixicore::cpu::contract_stubs::multi_codebook_sample(call);
    case quixicore::contract::OperationId::multimodal_rope_nd:
      return quixicore::cpu::contract_stubs::multimodal_rope_nd(call);
    case quixicore::contract::OperationId::multimodal_stream_deinterleave:
      return quixicore::cpu::contract_stubs::multimodal_stream_deinterleave(call);
    case quixicore::contract::OperationId::multimodal_stream_interleave:
      return quixicore::cpu::contract_stubs::multimodal_stream_interleave(call);
    case quixicore::contract::OperationId::multistream_logits_codebook_sample:
      return quixicore::cpu::contract_stubs::multistream_logits_codebook_sample(call);
    case quixicore::contract::OperationId::non_max_suppression:
      return quixicore::cpu::contract_stubs::non_max_suppression(call);
    case quixicore::contract::OperationId::overlap_add:
      return quixicore::cpu::contract_stubs::overlap_add(call);
    case quixicore::contract::OperationId::paged_attention_output_projection_residual:
      return quixicore::cpu::contract_stubs::paged_attention_output_projection_residual(call);
    case quixicore::contract::OperationId::patchify_nd:
      return quixicore::cpu::contract_stubs::patchify_nd(call);
    case quixicore::contract::OperationId::patchify_projection_position:
      return quixicore::cpu::contract_stubs::patchify_projection_position(call);
    case quixicore::contract::OperationId::pixel_shuffle:
      return quixicore::cpu::contract_stubs::pixel_shuffle(call);
    case quixicore::contract::OperationId::prefix_append_attention:
      return quixicore::cpu::contract_stubs::prefix_append_attention(call);
    case quixicore::contract::OperationId::qk_norm_rope_attention:
      return quixicore::cpu::contract_stubs::qk_norm_rope_attention(call);
    case quixicore::contract::OperationId::qkv_projection_qk_norm_rope_kv_append:
      return quixicore::cpu::contract_stubs::qkv_projection_qk_norm_rope_kv_append(call);
    case quixicore::contract::OperationId::ragged_pack:
      return quixicore::cpu::contract_stubs::ragged_pack(call);
    case quixicore::contract::OperationId::ragged_unpack:
      return quixicore::cpu::contract_stubs::ragged_unpack(call);
    case quixicore::contract::OperationId::reduce_scatter_gemm:
      return quixicore::cpu::contract_stubs::reduce_scatter_gemm(call);
    case quixicore::contract::OperationId::residual_rms_norm_quant:
      return quixicore::cpu::contract_stubs::residual_rms_norm_quant(call);
    case quixicore::contract::OperationId::residual_vector_quantize_decode:
      return quixicore::cpu::contract_stubs::residual_vector_quantize_decode(call);
    case quixicore::contract::OperationId::residual_vector_quantize_encode:
      return quixicore::cpu::contract_stubs::residual_vector_quantize_encode(call);
    case quixicore::contract::OperationId::resize_bicubic:
      return quixicore::cpu::contract_stubs::resize_bicubic(call);
    case quixicore::contract::OperationId::resize_bilinear:
      return quixicore::cpu::contract_stubs::resize_bilinear(call);
    case quixicore::contract::OperationId::resize_normalize_layout_quant:
      return quixicore::cpu::contract_stubs::resize_normalize_layout_quant(call);
    case quixicore::contract::OperationId::rnnt_beam_step:
      return quixicore::cpu::contract_stubs::rnnt_beam_step(call);
    case quixicore::contract::OperationId::rnnt_joiner:
      return quixicore::cpu::contract_stubs::rnnt_joiner(call);
    case quixicore::contract::OperationId::roi_align:
      return quixicore::cpu::contract_stubs::roi_align(call);
    case quixicore::contract::OperationId::rope_3d:
      return quixicore::cpu::contract_stubs::rope_3d(call);
    case quixicore::contract::OperationId::rvq_distance_argmin_residual_update:
      return quixicore::cpu::contract_stubs::rvq_distance_argmin_residual_update(call);
    case quixicore::contract::OperationId::speculative_verify_accept_compact_kv:
      return quixicore::cpu::contract_stubs::speculative_verify_accept_compact_kv(call);
    case quixicore::contract::OperationId::stft:
      return quixicore::cpu::contract_stubs::stft(call);
    case quixicore::contract::OperationId::streaming_conv1d_state:
      return quixicore::cpu::contract_stubs::streaming_conv1d_state(call);
    case quixicore::contract::OperationId::temporal_upsample_conv3d:
      return quixicore::cpu::contract_stubs::temporal_upsample_conv3d(call);
    case quixicore::contract::OperationId::unpatchify_nd:
      return quixicore::cpu::contract_stubs::unpatchify_nd(call);
    case quixicore::contract::OperationId::upsample_3d:
      return quixicore::cpu::contract_stubs::upsample_3d(call);
    case quixicore::contract::OperationId::upsample_conv2d:
      return quixicore::cpu::contract_stubs::upsample_conv2d(call);
    case quixicore::contract::OperationId::window_rfft_power_mel_log:
      return quixicore::cpu::contract_stubs::window_rfft_power_mel_log(call);
    default:
      break;
  }
  const auto name = quixicore::contract::operation_name(operation);
  return quixicore::contract::adapter_not_wired(
      name.empty() ? "unknown_operation" : name.data());
}

}  // namespace quixicore::cpu::contract_api
