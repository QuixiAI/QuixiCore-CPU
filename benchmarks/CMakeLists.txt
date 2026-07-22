# QuixiCore CPU benchmark harness.
#
# Built with baseline (non-ISA) flags only: never apply
# quixicore_cpu_add_isa_sources() to harness or naive-case sources — probes
# and _ref baselines must not depend on ISA-specific compilation.

add_executable(quixicore_cpu_bench
  harness/bench_main.cpp
  harness/donotopt_sink.cpp
  harness/env_info.cpp
  harness/registry.cpp
  harness/report.cpp
  harness/timing.cpp
  cases/contract_ops.cpp
  cases/mem_triad.cpp
  cases/qgemv.cpp
  cases/qgemv_formats.cpp
  cases/rms_norm.cpp
  cases/sgemv_naive.cpp
)
target_link_libraries(quixicore_cpu_bench PRIVATE QuixiCoreCPU::quixicore_cpu)
target_include_directories(quixicore_cpu_bench
  PRIVATE ${QuixiCoreCPU_SOURCE_DIR})  # kernel-internal headers for cases
if(QUIXICORE_CPU_ISA_DOTPROD_SUPPORTED)
  target_compile_definitions(quixicore_cpu_bench
    PRIVATE QUIXICORE_CPU_HAVE_QGEMV_DOTPROD=1)
endif()
# Emit at the build root (plus per-config subdir on multi-config generators)
# so the binary lands in a predictable place for scripts/bench and users.
set_target_properties(quixicore_cpu_bench PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_include_directories(quixicore_cpu_bench
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(quixicore_cpu_bench PRIVATE
  QUIXICORE_CPU_BENCH_REPO_DIR="${QuixiCoreCPU_SOURCE_DIR}"
  QUIXICORE_CPU_BENCH_BUILD_TYPE="$<CONFIG>")

if(MSVC)
  target_compile_options(quixicore_cpu_bench PRIVATE /W4)
else()
  target_compile_options(quixicore_cpu_bench PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(QUIXICORE_CPU_BUILD_TESTS)
  add_test(NAME bench_smoke COMMAND ${CMAKE_COMMAND}
    -DBENCH_EXE=$<TARGET_FILE:quixicore_cpu_bench>
    -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/bench_smoke_out
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunBenchSmoke.cmake)
  set_tests_properties(bench_smoke PROPERTIES TIMEOUT 300)
endif()
