# CPU Performance Guide

QuixiCore CPU benchmarks use host-side timing and must document enough system
state to make results reproducible.

## Required Report Fields

- Repository and git commit or working-tree label.
- QuixiCore contract version.
- Kernel family and operation.
- Input dtype, output dtype, and quant format when applicable.
- Shape family and concrete dimensions.
- CPU model, microarchitecture if known, core count, SMT state, and memory
  configuration.
- ISA target and compiler flags.
- Operating system and compiler version.
- Thread count, affinity policy, NUMA policy, and frequency governor or power
  mode when applicable.
- Warmup iterations, measurement iterations, median, and variance or min/max.
- Correctness command and result.

## Measurement Policy

- Use a monotonic clock for host-side timing.
- Separate warmup iterations from measured iterations.
- Pin or document thread placement for threaded kernels.
- Document frequency scaling state when it can affect measurements.
- Report scalar/reference baselines separately from ISA-tuned candidates.
- Do not compare results across machines unless hardware, memory, compiler, and
  thread policy are clearly identified.

## Minimum Optimization Run

A valid optimization run includes:

```text
kernel:
operation:
dtype_or_format:
shape_set:
correctness_command:
baseline_command:
candidate_command:
hardware:
os:
compiler:
thread_policy:
warmups:
iterations:
median:
variance_or_min_max:
decision:
artifact:
```

Record completed runs in `perf/optimization_status.md`.

## Benchmark Harness

`quixicore_cpu_bench` (see `benchmarks/README.md`) is the measurement tool of
record. It emits `run.json` + `results.jsonl` + `summary.md` per run into
`perf/results/YYYY-MM-DD/<run-id>/` (contents git-ignored). Anything that
will be recorded in the status docs must come from the harness, a `Release`
build, and an otherwise idle machine.

## Three-Baseline Rule (CPU)

Every optimized kernel result is compared against, at minimum:

1. **The naive scalar reference** (`_ref` implementation, or the harness's
   naive case) built at baseline Release flags — proves the variant beats
   honest portable code, not a strawman.
2. **The machine roofline** from `mem_triad`: memory-bound kernels
   (quantized decode GEMV/GEMM) are judged as achieved `weight_gbps`
   against the triad DRAM bandwidth on the same machine — proves the
   variant approaches what the memory system allows.
3. **The current QuixiCore implementation** before the change — proves the
   change is a win, not a shuffle.

A framework/library baseline slot (Accelerate, oneDNN, BLAS) is reserved for
context comparisons; n/a in v1.

## Timing Discipline

- Warm up by call count AND wall time (≥ warmup calls and ≥ 50 ms).
- Batch calls per sample so each timed sample spans ≥ 2 ms
  (`min_sample_ms`), dividing by the batch for per-call latency. This
  defeats clock-read and loop overhead for microsecond-scale kernels.
- Collect ≥ 20 samples; report median, p20/p80, CV, and batch.
- Treat CV > 0.10 as unstable: rerun on a quieter machine state before
  recording, or record the instability explicitly.

## Throughput Formulas

- GEMM/GEMV: `flops = 2*M*N*K` (GEMV is M=1) → GFLOP/s.
- Quantized decode: `effective_GBps = packed_weight_bytes / time` — the
  primary metric for memory-bound quant kernels.
- Elementwise/norms: conservative read+write bytes moved → GB/s.
- STREAM convention: write-allocate (RFO) traffic is not counted.

## Shape Strategy

Cover, per kernel: small-edge shapes, blocking-aligned shapes,
blocking-ragged shapes (odd K, non-multiple rows), real-model shapes
(registry families), and at least one stress shape. Record skipped shapes
with a reason (unsupported format, allocation failure, known gap).

## Per-Kernel Optimization Loop

1. Inventory the current implementation and its public entry point.
2. Read references (llama.cpp, KleidiAI, oneDNN equivalents) for the op.
3. Baseline with the harness on the registry shape set.
4. Classify the bottleneck: memory-bound, latency-bound (accumulation
   chains), or compute-bound — back it with bytes/FLOPs arithmetic.
5. Define a small set of named experiments.
6. Execute, measuring each against the three baselines.
7. Decide keep/reject per the decision rules.
8. Record in `perf/optimization_status.md` using the entry template below.

## Decision Rules

A change is a win only if it passes correctness at contract tolerances,
beats the priority shapes by ≥ 3% (simple change) or ≥ 8-10% (added
complexity), does not regress required shapes, and has an explanation backed
by bytes/FLOPs or microarchitectural reasoning. Reject wins that are within
noise (< CV), appear only on toy shapes, or add complexity disproportionate
to the gain.

## Recording Format

Each optimization pass appends an entry to `perf/optimization_status.md`:

```text
## YYYY-MM-DD: <kernel or pass name>

Status: not started | baselining | experimenting | candidate | landed | deferred.
Current implementation:
Current public route:
References inspected:
Correctness:
Baseline:
Experiments:
Decision:
Open questions:
Raw results:
```

Raw outputs stay in `perf/results/` (git-ignored); the curated table with a
decision column is copied into the status doc.

## CPU Variance Sources

What the harness records versus controls:

- **Turbo/thermal drift** — recorded (`frequency_policy`, CV), not
  controlled. Benchmark on mains power, thermally settled.
- **Hybrid P/E scheduling** — macOS runs request user-interactive QoS;
  otherwise recorded, not pinned. Affinity pinning arrives with threaded
  kernels.
- **SMT siblings** — not controlled; keep the machine idle.
- **Transparent huge pages, NUMA placement (Linux servers)** — documented
  per-run when relevant; first-touch init places pages before timing.
- **Allocator/layout variance** — mitigated with 64-byte-aligned buffers
  and deterministic fills.

