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
  src/dispatch/rms_norm.cpp
  src/runtime/cpu_features.cpp
  src/runtime/cpu_features_x86_64.cpp
  src/runtime/cpu_features_aarch64.cpp
  src/threading/thread_pool.cpp
  kernels/activations/activations_ref.cpp
  kernels/attention/attention_ref.cpp
  kernels/attention/attention_extended_ref.cpp
  kernels/attention/attention_composites_ref.cpp
  kernels/attention/attention_serving_ref.cpp
  kernels/attention/mla_fp8_ref.cpp
  kernels/attention/varlen_metadata_ref.cpp
  kernels/collectives/collectives_ref.cpp
  kernels/linear_attention/linear_attention_ref.cpp
  kernels/linear_attention/linear_attention_extended_ref.cpp
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
  kernels/quantization/qgemv_fused_ref.cpp
  kernels/quantization/gguf_ref.cpp
  kernels/quantization/quantization_ref.cpp
  kernels/quantization/microscale_ref.cpp
  kernels/quantization/turboquant_ref.cpp
  kernels/quantization/qgemv_neon.cpp
  kernels/quantization/qgemv_w8a8_ref.cpp
  kernels/quantization/qgemm_ref.cpp
  kernels/quantization/qgemm_extended_ref.cpp
  kernels/quantization/lm_head_ref.cpp
  kernels/sampling/sampling_ref.cpp
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
  kernels/utils/kd_ref.cpp
  kernels/vision/vision_ref.cpp
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

quixicore_cpu_add_isa_sources(quixicore_cpu ISA dotprod
  SOURCES kernels/quantization/qgemv_dotprod.cpp)
if(QUIXICORE_CPU_ISA_DOTPROD_SUPPORTED)
  target_compile_definitions(quixicore_cpu
    PRIVATE QUIXICORE_CPU_HAVE_QGEMV_DOTPROD=1)
endif()

if(QUIXICORE_CPU_BUILD_TOOLS)
  add_executable(quixicore_cpu_info tools/quixicore_cpu_info.cpp)
  target_link_libraries(quixicore_cpu_info PRIVATE QuixiCoreCPU::quixicore_cpu)
endif()

if(QUIXICORE_CPU_BUILD_TESTS)
  enable_testing()

  add_executable(test_backend_status tests/smoke/test_backend_status.cpp)
  target_link_libraries(test_backend_status PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME backend_status COMMAND test_backend_status)

  add_executable(test_cpu_features tests/unit/test_cpu_features.cpp)
  target_link_libraries(test_cpu_features PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME cpu_features COMMAND test_cpu_features)

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

  add_executable(test_quantization tests/correctness/test_quantization.cpp)
  target_link_libraries(test_quantization PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME quantization COMMAND test_quantization)

  add_executable(test_projection_ops tests/correctness/test_projection_ops.cpp)
  target_link_libraries(test_projection_ops PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME projection_ops COMMAND test_projection_ops)

  add_executable(test_parallel_sampling tests/correctness/test_parallel_sampling.cpp)
  target_link_libraries(test_parallel_sampling PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME parallel_sampling COMMAND test_parallel_sampling)

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
