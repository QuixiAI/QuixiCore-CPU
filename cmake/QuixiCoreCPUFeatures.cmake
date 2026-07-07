# Per-ISA compile policy for QuixiCore CPU.
#
# The library is built once per architecture and must run on any CPU of that
# architecture, so the global build never uses -march=native or a global
# MSVC /arch flag. ISA-specific kernel variants opt in per source file:
#
#   quixicore_cpu_add_isa_sources(quixicore_cpu ISA avx2
#     SOURCES kernels/quantization/qgemv_avx2.cpp)
#
# On compilers that can target the ISA, the sources are added to the target,
# compiled with that ISA's flags, and given the QUIXICORE_CPU_ISA_<ISA>=1
# definition. Otherwise the sources are skipped with a status message and
# runtime dispatch simply never sees the variant. Availability is reported in
# the cache variable QUIXICORE_CPU_ISA_<ISA>_SUPPORTED.
#
# Multi-architecture builds (CMAKE_OSX_ARCHITECTURES with more than one arch)
# cannot use per-file ISA flags safely; build each slice separately and lipo
# them together instead.

include(CheckCXXCompilerFlag)

# Resolves the target architecture as "x86_64", "aarch64", or "" (unknown).
function(quixicore_cpu_target_arch out_var)
  set(arch "")
  set(processor "${CMAKE_SYSTEM_PROCESSOR}")
  if(CMAKE_OSX_ARCHITECTURES)
    list(LENGTH CMAKE_OSX_ARCHITECTURES _count)
    if(_count GREATER 1)
      set(${out_var} "" PARENT_SCOPE)
      return()
    endif()
    set(processor "${CMAKE_OSX_ARCHITECTURES}")
  endif()
  if(processor MATCHES "^(x86_64|AMD64|amd64)$")
    set(arch "x86_64")
  elseif(processor MATCHES "^(aarch64|arm64|ARM64|arm64e)$")
    set(arch "aarch64")
  endif()
  set(${out_var} "${arch}" PARENT_SCOPE)
endfunction()

# Internal: maps an ISA name to compile flags for the current compiler.
# Sets <out_arch> to the architecture the ISA belongs to, <out_flags> to the
# flag list (may be empty when the compiler needs none), and <out_known> to
# TRUE when the ISA name is recognized.
function(_quixicore_cpu_isa_definition isa out_arch out_flags out_known)
  set(arch "")
  set(flags "")
  set(known TRUE)

  if(isa STREQUAL "avx2")
    set(arch "x86_64")
    if(MSVC)
      set(flags /arch:AVX2)
    else()
      set(flags -mavx2 -mfma -mf16c)
    endif()
  elseif(isa STREQUAL "avx512")
    set(arch "x86_64")
    if(MSVC)
      set(flags /arch:AVX512)
    else()
      set(flags -mavx512f -mavx512bw -mavx512vl -mavx512dq)
    endif()
  elseif(isa STREQUAL "avx512_vnni")
    set(arch "x86_64")
    if(MSVC)
      set(flags /arch:AVX512)
    else()
      set(flags -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni)
    endif()
  elseif(isa STREQUAL "amx")
    set(arch "x86_64")
    if(MSVC)
      set(flags /arch:AVX512)
    else()
      set(flags -mavx512f -mavx512bw -mavx512vl -mavx512dq
                -mamx-tile -mamx-int8 -mamx-bf16)
    endif()
  elseif(isa STREQUAL "dotprod")
    set(arch "aarch64")
    if(NOT MSVC)
      set(flags -march=armv8.2-a+dotprod)
    endif()
  elseif(isa STREQUAL "i8mm")
    set(arch "aarch64")
    if(MSVC)
      set(known FALSE) # No MSVC support for NEON I8MM intrinsics yet.
    else()
      set(flags -march=armv8.2-a+dotprod+i8mm)
    endif()
  elseif(isa STREQUAL "sve")
    set(arch "aarch64")
    if(MSVC)
      set(known FALSE) # No MSVC SVE codegen yet.
    else()
      set(flags -march=armv8.2-a+sve)
    endif()
  elseif(isa STREQUAL "sve2")
    set(arch "aarch64")
    if(MSVC)
      set(known FALSE)
    else()
      set(flags -march=armv9-a+sve2)
    endif()
  elseif(isa STREQUAL "sme2")
    set(arch "aarch64")
    if(MSVC)
      set(known FALSE)
    else()
      set(flags -march=armv9.2-a+sme2)
    endif()
  else()
    set(known FALSE)
  endif()

  set(${out_arch} "${arch}" PARENT_SCOPE)
  set(${out_flags} "${flags}" PARENT_SCOPE)
  set(${out_known} "${known}" PARENT_SCOPE)
endfunction()

# Internal: TRUE when the current build can compile sources for <isa>:
# the ISA belongs to the target architecture and the compiler accepts its
# flags. Result cached in QUIXICORE_CPU_ISA_<ISA>_SUPPORTED.
function(_quixicore_cpu_isa_supported isa out_supported out_flags)
  string(TOUPPER "${isa}" isa_upper)
  _quixicore_cpu_isa_definition("${isa}" isa_arch flags known)
  quixicore_cpu_target_arch(target_arch)

  set(supported FALSE)
  if(known AND target_arch STREQUAL "${isa_arch}")
    if(flags)
      set(CMAKE_REQUIRED_QUIET TRUE)
      list(JOIN flags " " probe_flags)
      check_cxx_compiler_flag("${probe_flags}" _quixicore_cpu_flag_${isa})
      set(supported ${_quixicore_cpu_flag_${isa}})
    else()
      set(supported TRUE)
    endif()
  endif()

  set(QUIXICORE_CPU_ISA_${isa_upper}_SUPPORTED "${supported}"
      CACHE INTERNAL "Build can compile ${isa} kernel variants")
  set(${out_supported} "${supported}" PARENT_SCOPE)
  set(${out_flags} "${flags}" PARENT_SCOPE)
endfunction()

# Adds ISA-variant sources to <target> when the build can compile them.
#
#   quixicore_cpu_add_isa_sources(<target> ISA <isa> SOURCES <files...>)
function(quixicore_cpu_add_isa_sources target)
  cmake_parse_arguments(ARG "" "ISA" "SOURCES" ${ARGN})
  if(NOT ARG_ISA OR NOT ARG_SOURCES)
    message(FATAL_ERROR
        "quixicore_cpu_add_isa_sources requires ISA and SOURCES")
  endif()

  _quixicore_cpu_isa_supported("${ARG_ISA}" supported flags)
  if(NOT supported)
    message(STATUS
        "QuixiCore CPU: skipping ${ARG_ISA} sources (not buildable here)")
    return()
  endif()

  string(TOUPPER "${ARG_ISA}" isa_upper)
  target_sources(${target} PRIVATE ${ARG_SOURCES})
  set_source_files_properties(${ARG_SOURCES} PROPERTIES
      COMPILE_OPTIONS "${flags}")
  set_property(SOURCE ${ARG_SOURCES} APPEND PROPERTY
      COMPILE_DEFINITIONS QUIXICORE_CPU_ISA_${isa_upper}=1)
endfunction()

# Probes every known ISA for the current build and prints a summary. Called
# from the top-level CMakeLists so flag probing runs on every configure.
function(quixicore_cpu_report_isa_support)
  quixicore_cpu_target_arch(target_arch)
  if(NOT target_arch)
    message(STATUS
        "QuixiCore CPU: unknown or multi-arch target; ISA variants disabled")
    return()
  endif()

  if(target_arch STREQUAL "x86_64")
    set(isas avx2 avx512 avx512_vnni amx)
  else()
    set(isas dotprod i8mm sve sve2 sme2)
  endif()

  set(buildable "")
  foreach(isa IN LISTS isas)
    _quixicore_cpu_isa_supported("${isa}" supported _flags)
    if(supported)
      list(APPEND buildable "${isa}")
    endif()
  endforeach()

  message(STATUS "QuixiCore CPU: target arch ${target_arch}")
  message(STATUS "QuixiCore CPU: buildable ISA variants: ${buildable}")
endfunction()
