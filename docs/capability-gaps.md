# CPU Capability Gaps

Inventory date: 2026-07-27.

This file compares the CPU backend with the **263-operation semantic
union** and **17 exact quant-format IDs** recorded in the
[QuixiCore umbrella capability map](https://github.com/QuixiAI/QuixiCore/blob/main/matrices/capability-map.md).

Backend snapshot: `main` @ `0159223979db`.

Normalized adapter stubs for the planned practical inference and fused
operations are indexed in `.quixicore/kernel-stubs.yaml` and declared in
`include/quixicore/cpu/contract_stubs.hpp`. The CPU has no gaps against the
observed union below; the generated stubs cover only the expanded planned
contract.

Union source revisions: CUDA d959679b0163; Metal a6d984377288; ROCm 636ae5ae983f; XPU 67c70fe4dc0c; CPU 0159223979db.

## How gaps are classified

- **family-only metadata**: this backend marks the family implemented but
  does not publish the exact operation ID. This is an enumeration/evidence
  gap, not proof that the semantic kernel is missing.
- **partial-family coverage**: the exact ID is absent and the backend marks
  the family partial.
- **capability-gated**: the family or operation depends on hardware/runtime
  conditions and is not an unconditional capability.
- **planned family**, **no family claim**, **partial operation**, and
  **experimental operation** are implementation or maturity gaps relative
  to a fully evidenced union capability.

Exact accelerator stage/layout aliases remain separate because the umbrella
map preserves published operation IDs. A backend may close a metadata gap
by documenting a proven semantic collapse instead of adding duplicate code.

## Summary

| Measure | Count |
|---|---:|
| Union operation capabilities | 263 |
| Fully implemented or semantically mapped | 263 |
| Operation gaps or enumeration gaps | 0 |
| Union quant-format IDs | 17 |
| Fully declared quant-format IDs | 17 |
| Quant-format gaps or missing declarations | 0 |

## Operation gap list

No semantic operation gaps are recorded against this union snapshot.

## Quant-format gap list

No quant-format gaps are recorded against the exact union identifiers.

## Evidence rule

Removing an implementation or maturity gap requires the backend's native
path, correctness coverage, focused performance evidence, and an updated
manifest/status entry. Removing a family-only metadata gap requires an exact
operation entry or a documented semantic alias backed by the existing tests
and performance notebook. Directory presence alone is not sufficient.

Evidence remains backend-owned in `perf/optimization_status.md`,
`perf/baseline_status.md`, `perf/results/`, and the
backend correctness tests.
