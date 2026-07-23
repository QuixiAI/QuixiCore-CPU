cmake_minimum_required(VERSION 3.20)

project(
  QuixiCoreCPU
  VERSION 0.1.0
  DESCRIPTION "QuixiCore CPU backend"
  LANGUAGES CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(QuixiCoreCPUFeatures)

option(QUIXICORE_CPU_BUILD_TESTS "Build QuixiCore CPU tests" ON)
option(QUIXICORE_CPU_BUILD_TOOLS "Build QuixiCore CPU command-line tools" ON)
option(QUIXICORE_CPU_BUILD_BENCHMARKS "Build QuixiCore CPU benchmark harness" ON)

# Baseline sources build with no ISA-specific flags so one binary per
# architecture runs on any CPU of that architecture. ISA kernel variants are
# added per file via quixicore_cpu_add_isa_sources(); see
# cmake/QuixiCoreCPUFeatures.cmake and kernels/README.md.
add_library(quixicore_cpu STATIC
  src/backend.cpp
  src/dispatch/qgemv.cpp
  src/dispatch/qgemv_w8a8.cpp
  src/dispatch/int8_gemm.cpp
  src/dispatch/w8a32_gemm.cpp
  src/dispatch/lowbit_w8a8.cpp
  src/dispatch/rms_norm.cpp
  src/runtime/cpu_features.cpp
  src/runtime/cpu_features_x86_64.cpp
  src/runtime/cpu_features_aarch64.cpp
  src/memory/workspace.cpp
  src/memory/quant_contract.cpp
  src/memory/packed_weights.cpp
  src/threading/thread_pool.cpp
  kernels/activations/activations_ref.cpp
  kernels/attention/attention_ref.cpp
  kernels/attention/attention_extended_ref.cpp
  kernels/attention/attention_composites_ref.cpp
  kernels/attention/attention_serving_ref.cpp
  kernels/attention/mla_fp8_ref.cpp
  kernels/attention/mla_absorb_ref.cpp
  kernels/attention/varlen_metadata_ref.cpp
  kernels/collectives/collectives_ref.cpp
  kernels/linear_attention/linear_attention_ref.cpp
  kernels/linear_attention/linear_attention_extended_ref.cpp
  kernels/linear_attention/llama_recurrent_ref.cpp
  kernels/matmul/dense_gemm_ref.cpp
  kernels/matmul/matmul_extended_ref.cpp
  kernels/moe/moe_ref.cpp
  kernels/moe/moe_extended_ref.cpp
  kernels/norms/layer_norm_ref.cpp
  kernels/norms/norms_extended_ref.cpp
  kernels/norms/norm_quant_ref.cpp
  kernels/norms/rms_norm_ref.cpp
  kernels/norms/rms_norm_neon.cpp
  kernels/optimizers/adamw_ref.cpp
  kernels/quantization/qgemv_ref.cpp
  kernels/quantization/activation_quant_ref.cpp
  kernels/quantization/iq_pack_ref.cpp
  kernels/quantization/qgemv_fused_ref.cpp
  kernels/quantization/qgemv_rows_ref.cpp
  kernels/quantization/gguf_ref.cpp
  kernels/quantization/gguf_pack_ref.cpp
  kernels/quantization/qgemv_gguf_blocked.cpp
  kernels/quantization/qgemv_gguf_neon.cpp
  kernels/quantization/quantization_ref.cpp
  kernels/quantization/quant_import_ref.cpp
  kernels/quantization/microscale_ref.cpp
  kernels/quantization/turboquant_ref.cpp
  kernels/quantization/qgemv_neon.cpp
  kernels/quantization/qgemv_w8a8_ref.cpp
  kernels/quantization/qgemm_ref.cpp
  kernels/quantization/qgemm_panel.cpp
  kernels/quantization/qgemm_extended_ref.cpp
  kernels/quantization/lm_head_ref.cpp
  kernels/quantization/int8_gemm_ref.cpp
  kernels/quantization/w8a32_gemm_ref.cpp
  kernels/quantization/w8a32_gemm_neon.cpp
  kernels/quantization/lowbit_ref.cpp
  kernels/quantization/lowbit_neon.cpp
  kernels/quantization/lowbit_w8a8_ref.cpp
  kernels/sampling/sampling_ref.cpp
  kernels/sampling/selection_ref.cpp
  kernels/sampling/speculative_ref.cpp
  kernels/sampling/transforms_ref.cpp
  kernels/serving/serving_ref.cpp
  kernels/serving/serving_extended_ref.cpp
  kernels/serving/kv_cache_extended_ref.cpp
  kernels/serving/serving_quant_ref.cpp
  kernels/serving/sparse_metadata_ref.cpp
  kernels/softmax/softmax_ref.cpp
  kernels/ssm/selective_scan_ref.cpp
  kernels/ssm/ssm_extended_ref.cpp
  kernels/utils/utils_ref.cpp
  kernels/utils/utils_extended_ref.cpp
  kernels/utils/tensor_ops_ref.cpp
  kernels/utils/float_storage_ref.cpp
  kernels/utils/kd_ref.cpp
  kernels/vision/vision_ref.cpp
  kernels/vision/conv_pool_ref.cpp
)
add_library(QuixiCoreCPU::quixicore_cpu ALIAS quixicore_cpu)

target_compile_features(quixicore_cpu PUBLIC cxx_std_20)
target_include_directories(
  quixicore_cpu
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # kernel-internal headers (kernels/...)
)

if(MSVC)
  target_compile_options(quixicore_cpu PRIVATE /W4)
else()
  target_compile_options(quixicore_cpu PRIVATE -Wall -Wextra -Wpedantic)
endif()

quixicore_cpu_report_isa_support()
quixicore_cpu_target_arch(QUIXICORE_CPU_TARGET_ARCH)

quixicore_cpu_add_isa_sources(quixicore_cpu ISA fp16
  SOURCES kernels/utils/float_storage_neon.cpp)
if(QUIXICORE_CPU_ISA_FP16_SUPPORTED)
  target_compile_definitions(quixicore_cpu
    PRIVATE QUIXICORE_CPU_HAVE_FLOAT_STORAGE_NEON_FP16=1)
endif()

quixicore_cpu_add_isa_sources(quixicore_cpu ISA f16c
  SOURCES kernels/utils/float_storage_f16c.cpp)
if(QUIXICORE_CPU_ISA_F16C_SUPPORTED)
  target_compile_definitions(quixicore_cpu
    PRIVATE QUIXICORE_CPU_HAVE_FLOAT_STORAGE_F16C=1)
endif()

quixicore_cpu_add_isa_sources(quixicore_cpu ISA dotprod
  SOURCES kernels/quantization/qgemv_dotprod.cpp
          kernels/quantization/int8_gemm_dotprod.cpp
          kernels/quantization/lowbit_w8a8_dotprod.cpp)
if(QUIXICORE_CPU_ISA_DOTPROD_SUPPORTED)
  target_compile_definitions(quixicore_cpu
    PRIVATE QUIXICORE_CPU_HAVE_QGEMV_DOTPROD=1
            QUIXICORE_CPU_HAVE_INT8_GEMM_DOTPROD=1)
  target_compile_definitions(quixicore_cpu
    PRIVATE QUIXICORE_CPU_HAVE_LOWBIT_W8A8_DOTPROD=1)
endif()

quixicore_cpu_add_isa_sources(quixicore_cpu ISA i8mm
  SOURCES kernels/quantization/int8_gemm_i8mm.cpp
          kernels/quantization/lowbit_w8a8_i8mm.cpp)
if(QUIXICORE_CPU_ISA_I8MM_SUPPORTED)
  target_compile_definitions(quixicore_cpu
    PRIVATE QUIXICORE_CPU_HAVE_INT8_GEMM_I8MM=1)
  target_compile_definitions(quixicore_cpu
    PRIVATE QUIXICORE_CPU_HAVE_LOWBIT_W8A8_I8MM=1)
endif()

quixicore_cpu_add_isa_sources(quixicore_cpu ISA avx2
  SOURCES kernels/quantization/int8_gemm_avx2.cpp
          kernels/quantization/w8a32_gemm_avx2.cpp
          kernels/quantization/lowbit_w8a8_avx2.cpp
          kernels/quantization/lowbit_avx2.cpp
          kernels/quantization/qgemv_gguf_avx2.cpp)
if(QUIXICORE_CPU_ISA_AVX2_SUPPORTED)
  target_compile_definitions(quixicore_cpu
    PRIVATE QUIXICORE_CPU_HAVE_INT8_GEMM_AVX2=1)
  target_compile_definitions(quixicore_cpu
    PRIVATE QUIXICORE_CPU_HAVE_W8A32_AVX2=1)
  target_compile_definitions(quixicore_cpu
    PRIVATE QUIXICORE_CPU_HAVE_LOWBIT_W8A8_AVX2=1)
  target_compile_definitions(quixicore_cpu
    PRIVATE QUIXICORE_CPU_HAVE_LOWBIT_AVX2=1)
  target_compile_definitions(quixicore_cpu
    PRIVATE QUIXICORE_CPU_HAVE_GGUF_GEMV_AVX2=1)
endif()

quixicore_cpu_add_isa_sources(quixicore_cpu ISA avx512
  SOURCES kernels/quantization/lowbit_avx512.cpp
          kernels/quantization/w8a32_gemm_avx512.cpp
          kernels/quantization/qgemv_gguf_avx512.cpp)
if(QUIXICORE_CPU_ISA_AVX512_SUPPORTED)
  target_compile_definitions(quixicore_cpu
    PRIVATE QUIXICORE_CPU_HAVE_LOWBIT_AVX512=1)
  target_compile_definitions(quixicore_cpu
    PRIVATE QUIXICORE_CPU_HAVE_W8A32_AVX512=1)
  target_compile_definitions(quixicore_cpu
    PRIVATE QUIXICORE_CPU_HAVE_GGUF_GEMV_AVX512=1)
endif()

quixicore_cpu_add_isa_sources(quixicore_cpu ISA avx512_vnni
  SOURCES kernels/quantization/int8_gemm_avx512.cpp
          kernels/quantization/lowbit_w8a8_avx512.cpp)
if(QUIXICORE_CPU_ISA_AVX512_VNNI_SUPPORTED)
  target_compile_definitions(quixicore_cpu
    PRIVATE QUIXICORE_CPU_HAVE_INT8_GEMM_AVX512_VNNI=1)
  target_compile_definitions(quixicore_cpu
    PRIVATE QUIXICORE_CPU_HAVE_LOWBIT_W8A8_AVX512_VNNI=1)
endif()

if(QUIXICORE_CPU_BUILD_TOOLS)
  add_executable(quixicore_cpu_info tools/quixicore_cpu_info.cpp)
  target_link_libraries(quixicore_cpu_info PRIVATE QuixiCoreCPU::quixicore_cpu)
endif()

if(QUIXICORE_CPU_BUILD_TESTS)
  enable_testing()

  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME parity_manifest
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_parity_manifest.py)
  endif()

  add_executable(test_backend_status tests/smoke/test_backend_status.cpp)
  target_link_libraries(test_backend_status PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME backend_status COMMAND test_backend_status)

  add_executable(test_cpu_features tests/unit/test_cpu_features.cpp)
  target_link_libraries(test_cpu_features PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME cpu_features COMMAND test_cpu_features)

  add_executable(test_workspace tests/unit/test_workspace.cpp)
  target_link_libraries(test_workspace PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME workspace COMMAND test_workspace)

  add_executable(test_float_storage tests/correctness/test_float_storage.cpp)
  target_link_libraries(test_float_storage PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME float_storage COMMAND test_float_storage)

  add_executable(test_qgemv tests/correctness/test_qgemv.cpp)
  target_link_libraries(test_qgemv PRIVATE QuixiCoreCPU::quixicore_cpu)
  target_include_directories(test_qgemv
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})  # kernel-internal headers
  if(QUIXICORE_CPU_ISA_DOTPROD_SUPPORTED)
    target_compile_definitions(test_qgemv
      PRIVATE QUIXICORE_CPU_HAVE_QGEMV_DOTPROD=1)
  endif()
  add_test(NAME qgemv COMMAND test_qgemv)
  add_test(NAME qgemv_forced_dotprod_contract COMMAND test_qgemv)
  set_tests_properties(qgemv_forced_dotprod_contract PROPERTIES
    ENVIRONMENT "QUIXICORE_CPU_QGEMV_VARIANT=dotprod_i8")

  add_executable(test_qgemv_w8a8 tests/correctness/test_qgemv_w8a8.cpp)
  target_link_libraries(test_qgemv_w8a8 PRIVATE QuixiCoreCPU::quixicore_cpu)
  target_include_directories(test_qgemv_w8a8
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})  # kernel-internal headers
  add_test(NAME qgemv_w8a8 COMMAND test_qgemv_w8a8)
  add_test(NAME qgemv_w8a8_forced_dotprod COMMAND test_qgemv_w8a8)
  set_tests_properties(qgemv_w8a8_forced_dotprod PROPERTIES
    ENVIRONMENT "QUIXICORE_CPU_QGEMV_W8A8_VARIANT=dotprod")

  add_executable(test_gguf_formats tests/correctness/test_gguf_formats.cpp)
  target_link_libraries(test_gguf_formats PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME gguf_formats COMMAND test_gguf_formats)
  add_test(NAME gguf_formats_forced_ref COMMAND test_gguf_formats)
  set_tests_properties(gguf_formats_forced_ref PROPERTIES
    ENVIRONMENT "QUIXICORE_CPU_GGUF_GEMV_VARIANT=ref")

  add_executable(test_quant_activation
    tests/correctness/test_quant_activation.cpp)
  target_link_libraries(test_quant_activation
    PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME quant_activation COMMAND test_quant_activation)

  add_executable(test_quantization tests/correctness/test_quantization.cpp)
  target_link_libraries(test_quantization PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME quantization COMMAND test_quantization)

  add_executable(test_quant_contract
    tests/correctness/test_quant_contract.cpp)
  target_link_libraries(test_quant_contract
    PRIVATE QuixiCoreCPU::quixicore_cpu)
  target_compile_definitions(test_quant_contract PRIVATE
    QUIXICORE_CPU_TESTDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/testdata")
  target_include_directories(test_quant_contract
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME quant_contract COMMAND test_quant_contract)

  add_executable(test_quant_import
    tests/correctness/test_quant_import.cpp)
  target_link_libraries(test_quant_import
    PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME quant_import COMMAND test_quant_import)

  add_executable(test_tensor_ops tests/correctness/test_tensor_ops.cpp)
  target_link_libraries(test_tensor_ops PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME tensor_ops COMMAND test_tensor_ops)

  add_executable(test_conv_pool tests/correctness/test_conv_pool.cpp)
  target_link_libraries(test_conv_pool PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME conv_pool COMMAND test_conv_pool)

  add_executable(test_llama_recurrent
    tests/correctness/test_llama_recurrent.cpp)
  target_link_libraries(test_llama_recurrent
    PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME llama_recurrent COMMAND test_llama_recurrent)

  add_executable(test_lowbit tests/correctness/test_lowbit.cpp)
  target_link_libraries(test_lowbit PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME lowbit COMMAND test_lowbit)

  add_executable(test_packed_weights
    tests/correctness/test_packed_weights.cpp)
  target_link_libraries(test_packed_weights PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME packed_weights COMMAND test_packed_weights)

  add_executable(test_int8_gemm tests/correctness/test_int8_gemm.cpp)
  target_link_libraries(test_int8_gemm PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME int8_gemm COMMAND test_int8_gemm)
  add_test(NAME int8_gemm_forced_ref COMMAND test_int8_gemm)
  set_tests_properties(int8_gemm_forced_ref PROPERTIES
    ENVIRONMENT "QUIXICORE_CPU_INT8_GEMM_VARIANT=ref")
  if(QUIXICORE_CPU_ISA_DOTPROD_SUPPORTED)
    add_test(NAME int8_gemm_forced_dotprod COMMAND test_int8_gemm)
    set_tests_properties(int8_gemm_forced_dotprod PROPERTIES
      ENVIRONMENT "QUIXICORE_CPU_INT8_GEMM_VARIANT=dotprod")
  endif()

  add_executable(test_w8a32_gemm tests/correctness/test_w8a32_gemm.cpp)
  target_link_libraries(test_w8a32_gemm PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME w8a32_gemm COMMAND test_w8a32_gemm)
  add_test(NAME w8a32_gemm_forced_ref COMMAND test_w8a32_gemm)
  set_tests_properties(w8a32_gemm_forced_ref PROPERTIES
    ENVIRONMENT "QUIXICORE_CPU_W8A32_VARIANT=ref")
  if(QUIXICORE_CPU_TARGET_ARCH STREQUAL "aarch64")
    add_test(NAME w8a32_gemm_forced_neon COMMAND test_w8a32_gemm)
    set_tests_properties(w8a32_gemm_forced_neon PROPERTIES
      ENVIRONMENT "QUIXICORE_CPU_W8A32_VARIANT=neon")
  endif()
  if(QUIXICORE_CPU_ISA_I8MM_SUPPORTED)
    add_test(NAME int8_gemm_forced_i8mm COMMAND test_int8_gemm)
    set_tests_properties(int8_gemm_forced_i8mm PROPERTIES
      ENVIRONMENT "QUIXICORE_CPU_INT8_GEMM_VARIANT=i8mm")
  endif()

  add_executable(test_lowbit_w8a8
    tests/correctness/test_lowbit_w8a8.cpp)
  target_link_libraries(test_lowbit_w8a8
    PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME lowbit_w8a8 COMMAND test_lowbit_w8a8)
  add_test(NAME lowbit_w8a8_forced_ref COMMAND test_lowbit_w8a8)
  set_tests_properties(lowbit_w8a8_forced_ref PROPERTIES
    ENVIRONMENT "QUIXICORE_CPU_LOWBIT_W8A8_VARIANT=ref")
  if(QUIXICORE_CPU_ISA_DOTPROD_SUPPORTED)
    add_test(NAME lowbit_w8a8_forced_dotprod COMMAND test_lowbit_w8a8)
    set_tests_properties(lowbit_w8a8_forced_dotprod PROPERTIES
      ENVIRONMENT "QUIXICORE_CPU_LOWBIT_W8A8_VARIANT=dotprod")
  endif()
  if(QUIXICORE_CPU_ISA_I8MM_SUPPORTED)
    add_test(NAME lowbit_w8a8_forced_i8mm COMMAND test_lowbit_w8a8)
    set_tests_properties(lowbit_w8a8_forced_i8mm PROPERTIES
      ENVIRONMENT "QUIXICORE_CPU_LOWBIT_W8A8_VARIANT=i8mm")
  endif()

  add_executable(test_projection_ops tests/correctness/test_projection_ops.cpp)
  target_link_libraries(test_projection_ops PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME projection_ops COMMAND test_projection_ops)

  add_executable(test_parallel_sampling tests/correctness/test_parallel_sampling.cpp)
  target_link_libraries(test_parallel_sampling PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME parallel_sampling COMMAND test_parallel_sampling)

  add_executable(test_colibri_algorithms
    tests/correctness/test_colibri_algorithms.cpp)
  target_link_libraries(test_colibri_algorithms
    PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME colibri_algorithms COMMAND test_colibri_algorithms)

  add_executable(test_mla_absorb tests/correctness/test_mla_absorb.cpp)
  target_link_libraries(test_mla_absorb PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME mla_absorb COMMAND test_mla_absorb)

  add_executable(test_moe_batch_union
    tests/correctness/test_moe_batch_union.cpp)
  target_link_libraries(test_moe_batch_union
    PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME moe_batch_union COMMAND test_moe_batch_union)

  add_executable(test_serving_metadata tests/correctness/test_serving_metadata.cpp)
  target_link_libraries(test_serving_metadata PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME serving_metadata COMMAND test_serving_metadata)

  add_executable(test_remaining_ops tests/correctness/test_remaining_ops.cpp)
  target_link_libraries(test_remaining_ops PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME remaining_ops COMMAND test_remaining_ops)

  add_executable(test_rms_norm tests/correctness/test_rms_norm.cpp)
  target_link_libraries(test_rms_norm PRIVATE QuixiCoreCPU::quixicore_cpu)
  target_include_directories(test_rms_norm
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})  # kernel-internal headers
  add_test(NAME rms_norm COMMAND test_rms_norm)

  add_executable(test_ops tests/correctness/test_ops.cpp)
  target_link_libraries(test_ops PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME ops COMMAND test_ops)

  add_executable(test_extended_ops tests/correctness/test_extended_ops.cpp)
  target_link_libraries(test_extended_ops PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME extended_ops COMMAND test_extended_ops)

  add_executable(test_sibling_entrypoints
    tests/correctness/test_sibling_entrypoints.cpp)
  target_link_libraries(test_sibling_entrypoints
    PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME sibling_entrypoints COMMAND test_sibling_entrypoints)

  add_executable(test_threading tests/unit/test_threading.cpp)
  target_link_libraries(test_threading PRIVATE QuixiCoreCPU::quixicore_cpu)
  target_include_directories(test_threading
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})  # kernel-internal headers
  add_test(NAME threading COMMAND test_threading)

  add_executable(test_benchmark_check tests/unit/test_benchmark_check.cpp)
  target_compile_features(test_benchmark_check PRIVATE cxx_std_20)
  target_include_directories(test_benchmark_check
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
  add_test(NAME benchmark_check COMMAND test_benchmark_check)

  if(QUIXICORE_CPU_BUILD_TOOLS)
    add_test(NAME cpu_info_tool COMMAND quixicore_cpu_info)
  endif()
endif()

if(QUIXICORE_CPU_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
