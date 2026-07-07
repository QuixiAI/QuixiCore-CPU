# CTest driver: run the bench harness on the smoke preset and validate that
# it produces well-formed outputs. Pure output-shape validation — no
# performance assertion of any kind.

file(REMOVE_RECURSE "${OUT_DIR}")

execute_process(
  COMMAND "${BENCH_EXE}" --preset smoke --warmup 1 --iters 3
          --out-dir "${OUT_DIR}"
  RESULT_VARIABLE rc
)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "quixicore_cpu_bench exited with ${rc}")
endif()

foreach(f run.json results.jsonl summary.md)
  if(NOT EXISTS "${OUT_DIR}/${f}")
    message(FATAL_ERROR "missing output file: ${f}")
  endif()
endforeach()

file(STRINGS "${OUT_DIR}/results.jsonl" result_lines)
list(LENGTH result_lines line_count)
if(line_count LESS 2)
  message(FATAL_ERROR "too few result rows: ${line_count}")
endif()

file(READ "${OUT_DIR}/results.jsonl" results_blob)
foreach(needle
    "\"schema\": 1"
    "\"kernel\": \"mem_triad\""
    "\"kernel\": \"sgemv_naive\""
    "\"target_ms\"")
  string(FIND "${results_blob}" "${needle}" pos)
  if(pos EQUAL -1)
    message(FATAL_ERROR "results.jsonl missing ${needle}")
  endif()
endforeach()

file(READ "${OUT_DIR}/run.json" run_blob)
string(FIND "${run_blob}" "\"cpu_features\"" pos)
if(pos EQUAL -1)
  message(FATAL_ERROR "run.json missing cpu_features")
endif()
