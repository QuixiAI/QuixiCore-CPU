# Norms

Contract family: `norms` (RMSNorm / LayerNorm). Spec:
umbrella `specs/kernels/norms.md`.

Status: `rms_norm` in progress — scalar reference (float64 accumulation,
`rms_norm_ref.cpp`) plus NEON variant (`rms_norm_neon.cpp`, 4.3-4.6x over
ref on M4 Max), dispatch via `src/dispatch/rms_norm.cpp`, correctness in
`tests/correctness/test_rms_norm.cpp`, evidence in
`perf/optimization_status.md` (2026-07-07). Not claimed supported.
LayerNorm: planned.
