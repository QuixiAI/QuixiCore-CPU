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
  src/dispatch/quant_gemv.cpp
  src/runtime/cpu_features.cpp
  src/runtime/cpu_features_x86_64.cpp
  src/runtime/cpu_features_aarch64.cpp
  kernels/quantization/qgemv_ref.cpp
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

  add_executable(test_quant_gemv tests/correctness/test_quant_gemv.cpp)
  target_link_libraries(test_quant_gemv PRIVATE QuixiCoreCPU::quixicore_cpu)
  target_include_directories(test_quant_gemv
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})  # kernel-internal headers
  if(QUIXICORE_CPU_ISA_DOTPROD_SUPPORTED)
    target_compile_definitions(test_quant_gemv
      PRIVATE QUIXICORE_CPU_HAVE_QGEMV_DOTPROD=1)
  endif()
  add_test(NAME quant_gemv COMMAND test_quant_gemv)

  if(QUIXICORE_CPU_BUILD_TOOLS)
    add_test(NAME cpu_info_tool COMMAND quixicore_cpu_info)
  endif()
endif()

if(QUIXICORE_CPU_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
