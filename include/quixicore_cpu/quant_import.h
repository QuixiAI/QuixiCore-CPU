#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "quixicore_cpu/float_storage.h"
#include "quixicore_cpu/quant_contract.h"

namespace quixicore_cpu {

enum class QuantImportProvenance {
  kCanonical,
  kAWQ,
  kGPTQ,
  kAutoRound,
  kSmoothQuant,
  kBitNet,
};

// Owned, portable quantized tensor. `data` always contains the canonical
// contract bytes. Side tables are logical arrays rather than CPU-private
// panel data, so the object remains serializable and ISA independent.
struct CanonicalQuantTensor {
  QuantTensorMetadata metadata;
  std::vector<std::uint8_t> data;
  std::vector<std::uint8_t> scale_codes;
  std::vector<float> scales;
  std::vector<float> zero_points;
  std::vector<float> activation_scales;
  std::vector<int> activation_zero_points;
  std::vector<int> group_index;
  std::vector<int> act_order;
  std::vector<std::int32_t> row_sums;
  QuantImportProvenance provenance = QuantImportProvenance::kCanonical;
};

Status validate_canonical_quant_tensor(const CanonicalQuantTensor& tensor);

// Canonical pack/unpack entrypoints for every non-cache M0 layout. FP16 and
// BF16 inputs are accepted through FloatStorageInput and accumulate in FP32.
Status quantize_canonical(FloatStorageInput input, long long rows,
                          long long columns, CanonicalQuantLayout layout,
                          long long group_size, CanonicalQuantTensor* tensor,
                          bool scale_2d = false);
// Explicit scale topology for standalone FP8/FP4. For block mode,
// group_size is the K tile and scale_domain_rows is the row tile.
Status quantize_canonical_fp_scaled(
    FloatStorageInput input, long long rows, long long columns,
    CanonicalQuantLayout layout, QuantScaleMode scale_mode,
    long long group_size, int scale_domain_rows,
    CanonicalQuantTensor* tensor);
Status dequantize_canonical(const CanonicalQuantTensor& tensor, float* output,
                            long long output_count);

// AWQ GEMM layout: qweight[K,N/8], qzeros[K/group,N/8], scales[K/group,N].
// Packed lanes use AWQ_REVERSE_ORDER={0,4,1,5,2,6,3,7}.
struct AwqU4Source {
  const std::uint32_t* qweight = nullptr;
  std::size_t qweight_words = 0;
  const std::uint32_t* qzeros = nullptr;
  std::size_t qzero_words = 0;
  const float* scales = nullptr;
  std::size_t scale_count = 0;
  long long input_features = 0;
  long long output_features = 0;
  long long group_size = 0;
};
Status import_awq_u4(const AwqU4Source& source,
                     CanonicalQuantTensor* tensor);

// GPTQ layout: qweight[K/8,N], qzeros[G,N/8], scales[G,N]. g_idx maps each
// packed K lane to its scale group. act_order, when present, maps packed K to
// logical K. V1 stores zero-1; V2 stores the actual zero.
struct GptqU4Source {
  const std::uint32_t* qweight = nullptr;
  std::size_t qweight_words = 0;
  const std::uint32_t* qzeros = nullptr;
  std::size_t qzero_words = 0;
  const float* scales = nullptr;
  std::size_t scale_count = 0;
  const int* group_index = nullptr;
  std::size_t group_index_count = 0;
  const int* act_order = nullptr;
  std::size_t act_order_count = 0;
  long long input_features = 0;
  long long output_features = 0;
  long long group_size = 0;
  bool symmetric = false;
  bool gptq_v2 = false;
};
Status import_gptq_u4(const GptqU4Source& source,
                      CanonicalQuantTensor* tensor);
Status import_autoround_gptq_u4(const GptqU4Source& source,
                                CanonicalQuantTensor* tensor);

// AutoRound can export already-logical codes for FP8/MX/NV/BitNet as well as
// U4. This view copies those fields into the owned canonical representation.
struct AutoRoundCanonicalSource {
  QuantTensorMetadata metadata;
  const std::uint8_t* data = nullptr;
  std::size_t data_bytes = 0;
  const std::uint8_t* scale_codes = nullptr;
  std::size_t scale_code_count = 0;
  const float* scales = nullptr;
  std::size_t scale_count = 0;
  const float* zero_points = nullptr;
  std::size_t zero_point_count = 0;
  const int* group_index = nullptr;
  std::size_t group_index_count = 0;
  const int* act_order = nullptr;
  std::size_t act_order_count = 0;
};
Status import_autoround_canonical(const AutoRoundCanonicalSource& source,
                                  CanonicalQuantTensor* tensor);

// SmoothQuant exports signed int8 weights [N,K], per-channel weight scales,
// and activation quantization metadata. Activation zero points enable the AZP
// W8A8 route; row sums are computed during import when not supplied.
struct SmoothQuantW8A8Source {
  const std::int8_t* weights = nullptr;
  std::size_t weight_count = 0;
  const float* weight_scales = nullptr;
  std::size_t weight_scale_count = 0;
  const float* activation_scales = nullptr;
  std::size_t activation_scale_count = 0;
  const int* activation_zero_points = nullptr;
  std::size_t activation_zero_point_count = 0;
  const std::int32_t* row_sums = nullptr;
  std::size_t row_sum_count = 0;
  long long rows = 0;
  long long columns = 0;
};
Status import_smoothquant_w8a8(const SmoothQuantW8A8Source& source,
                               CanonicalQuantTensor* tensor);

// BitNet I2_S-compatible canonical blocks. Checked import rejects reserved
// code 3 and non-finite/negative FP16 scales before CPU preparation.
struct BitNetI2Source {
  const std::uint8_t* packed = nullptr;
  std::size_t packed_bytes = 0;
  long long rows = 0;
  long long columns = 0;
  long long group_size = 32;
};
Status import_bitnet_i2_s(const BitNetI2Source& source,
                          CanonicalQuantTensor* tensor);

const char* quant_import_provenance_name(QuantImportProvenance provenance);

}  // namespace quixicore_cpu
