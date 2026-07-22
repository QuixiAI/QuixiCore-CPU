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
  kernels/linear_attention/linear_attention_ref.cpp
  kernels/matmul/dense_gemm_ref.cpp
  kernels/moe/moe_ref.cpp
  kernels/norms/layer_norm_ref.cpp
  kernels/norms/rms_norm_ref.cpp
  kernels/norms/rms_norm_neon.cpp
  kernels/optimizers/adamw_ref.cpp
  kernels/quantization/qgemv_ref.cpp
  kernels/quantization/qgemv_neon.cpp
  kernels/quantization/qgemv_w8a8_ref.cpp
  kernels/quantization/qgemm_ref.cpp
  kernels/sampling/sampling_ref.cpp
  kernels/serving/serving_ref.cpp
  kernels/softmax/softmax_ref.cpp
  kernels/ssm/selective_scan_ref.cpp
  kernels/utils/utils_ref.cpp
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

  add_executable(test_rms_norm tests/correctness/test_rms_norm.cpp)
  target_link_libraries(test_rms_norm PRIVATE QuixiCoreCPU::quixicore_cpu)
  target_include_directories(test_rms_norm
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})  # kernel-internal headers
  add_test(NAME rms_norm COMMAND test_rms_norm)

  add_executable(test_ops tests/correctness/test_ops.cpp)
  target_link_libraries(test_ops PRIVATE QuixiCoreCPU::quixicore_cpu)
  add_test(NAME ops COMMAND test_ops)

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
