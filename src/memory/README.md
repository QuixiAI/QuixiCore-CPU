# Memory

CPU-local storage and layout utilities. These facilities do not alter the
QuixiCore tensor or quantization contracts.

## Workspace arenas

`quixicore_cpu::Workspace` is a public, move-only segmented scratch arena. Its
allocations are 64-byte aligned by default and remain pointer-stable if a later
allocation grows the arena. `reset()` releases scratch logically but retains
every backing block for reuse. A workspace belongs to one concurrent
invocation and is intentionally not thread-safe.

Internal `WorkspaceFrame` scopes mark and rewind an arena, which makes nested
kernel composition safe. A null public workspace selects a persistent arena
local to the invoking application or pool thread. Parallel kernels reserve
disjoint worker slices in that arena before dispatch; workers do not allocate.

The fused norm/quantization and paired QGEMV paths use the persistent internal
arena. `qgemm_prepacked` also accepts an explicit caller-owned workspace so a
runtime can control lifetime and prewarm capacity.

## Packed weights

`quixicore_cpu::CpuPackedWeights` owns both a copy of the canonical packed
quant-format bytes and a private 64-byte-aligned CPU panel. The panel order is
`[row_panel, k_block, row_lane, block_bytes]`; the final row panel is zero
padded. Automatic preparation currently chooses 4-row NEON, 8-row AVX2, or
16-row AVX-512 panels, with a 1-row portable fallback. Forced layouts exist for
cross-architecture correctness tests.

Panel preparation is a model-load/cache operation and must remain outside the
timed kernel path. Canonical bytes remain available for contract-compatible
fallbacks such as `m == 1` QGEMV. The initial consumer is Q4_0/Q8_0
`qgemm_prepacked`; the generic panel builder accepts every registered format
whose packed size and block geometry are known, allowing later microkernels to
reuse the same ownership and layout boundary.

Correctness coverage lives in `tests/unit/test_workspace.cpp` and
`tests/correctness/test_packed_weights.cpp`. Focused measurements are recorded
under the `prerequisites` benchmark case and in `perf/optimization_status.md`.
