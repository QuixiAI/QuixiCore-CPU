# QuixiCore CPU Performance Handbook

This is the operating guide for optimizing every CPU kernel under `kernels/`.
The goal is a repeatable optimization loop: preserve the QuixiCore contract,
identify a specific bottleneck, measure clean baselines, run controlled
experiments, keep only verified wins, and leave enough evidence for another
developer to reproduce the decision.

The running optimization notebook is `perf/optimization_status.md`. Accepted
baseline snapshots live in `perf/baseline_status.md`. Raw harness output belongs
under `perf/results/` and is git-ignored.

## Principles

Correctness and measurement come before optimization. A change is not a win
until it:

- preserves the public QuixiCore operation semantics and error behavior;
- passes the contract tolerance on aligned, ragged, edge, and realistic shapes;
- improves the intended metric on the intended CPU class and thread count;
- does not introduce a material regression on other required shapes or routes;
- remains safe on CPUs that do not support the new ISA; and
- has an explanation supported by bytes, operations, scaling, code generation,
  or hardware-counter evidence.

Optimize a diagnosed bottleneck rather than collecting isolated tricks:

- **DRAM-bandwidth-bound:** reduce packed bytes, avoid redundant passes, improve
  streaming access, and scale threads only until memory bandwidth saturates.
- **Cache- or TLB-bound:** improve locality, change blocking or packing, reduce
  the working set, and avoid pathological strides or page walks.
- **Compute-throughput-bound:** use appropriate SIMD or matrix instructions,
  increase useful work per instruction, and reduce conversion overhead.
- **Dependency-latency-bound:** use multiple accumulators, shorten serial
  reductions, and expose independent work to the out-of-order core.
- **Front-end- or branch-bound:** specialize stable format decisions, reduce hot
  loop code size, remove unpredictable branches, and simplify address math.
- **Synchronization- or scheduling-bound:** increase grain size, reduce wakeups
  and barriers, improve load balance, and avoid false sharing.
- **Dispatch- or setup-bound:** route tiny shapes to a simpler path, cache
  resolved variants and reusable packing, and keep allocation outside hot calls.

A mechanism that works on one microarchitecture is a hypothesis elsewhere.
AVX-512 can reduce frequency on some x86 CPUs, SMT can help or hurt, prefetch can
pollute a cache, and extra threads can slow a memory-bound kernel. Measure on
each CPU class for which a performance tier is claimed.

## Contract And Repository Facts To Preserve

The umbrella QuixiCore repository owns the public contract. Before changing a
contract-facing kernel, read the relevant entries in:

```text
registry/kernels.yaml
registry/quant-formats.yaml
registry/benchmark-shapes.yaml
registry/tolerances.yaml
matrices/
```

Also read the corresponding kernel or format specification and check
`docs/sibling-port-matrix.md` for CPU-specific semantic mappings. CPU packing,
ISA dispatch, thread scheduling, and cache blocking stay behind the shared API.

Repository roles:

- `include/quixicore_cpu/` contains the public CPU API.
- `kernels/<family>/*_ref.cpp` contains portable correctness anchors.
- `kernels/<family>/*_<isa>.cpp` contains optional ISA variants.
- `src/dispatch/` selects supported variants behind public entry points.
- `src/runtime/` detects executable CPU and OS feature state.
- `src/threading/` owns the persistent fork-join pool.
- `cmake/QuixiCoreCPUFeatures.cmake` applies ISA flags per source file.
- `tests/correctness/` validates semantics and tolerances.
- `benchmarks/` contains the native measurement harness.

The library is deliberately not built globally with `-march=native` or a broad
MSVC `/arch` option. Portable sources use baseline architecture flags. ISA
sources are compiled separately and may run only after runtime feature checks.
Do not turn a local benchmark win into an illegal-instruction failure on an
older CPU.

Current model-kernel families include:

- activations, norms, softmax, dense and complex matmul;
- dense, windowed, variable-length, paged, cascade, quantized, and MLA
  attention, including backward and cache paths;
- GGUF, integer, FP8, MXFP8, MXFP4, NVFP4, BitNet, TurboQuant, and fused
  quantization paths;
- MoE routing, permutation, grouped GEMM, and finalize operations;
- sampling, beam search, speculative decoding, EAGLE, embeddings, and KV-cache
  utilities;
- linear attention, Mamba/SSD, FFT convolution, vision, optimizer, utility,
  knowledge-distillation, and host-reference collective operations.

Most of this surface is currently portable reference code. Existing tuned paths
do not imply an optimized tier for sibling operations, other dtypes, formats, or
architectures. Performance support remains operation- and evidence-specific.

## Reference Search Protocol

Start each kernel pass by finding comparable CPU implementations and benchmark
methodology. Record exact repository revisions and file paths in
`perf/optimization_status.md`; a project name alone is not reproducible.

Useful reference families include:

- llama.cpp/ggml for quantized packing, decode GEMV, row kernels, attention, and
  portable CPU dispatch patterns;
- KleidiAI for Arm NEON, DotProd, I8MM, SME, and packed microkernels;
- oneDNN for x86 and Arm primitive dispatch, threading, post-ops, and matmul;
- BLIS, OpenBLAS, or platform BLAS for dense GEMM blocking and baselines;
- XNNPACK for small tensors, elementwise kernels, packing, and mobile CPUs;
- the QuixiCore Metal, XPU, CUDA, and ROCm siblings for operation semantics,
  shapes, fusion boundaries, and benchmark ideas—not CPU mechanisms.

Use targeted searches rather than reading entire mirrors:

```sh
rg -n "operation_name|format_name|qgemv|softmax|rope|paged" \
  .reference kernels ../QuixiCore-* 2>/dev/null
rg -n "avx|vnni|amx|neon|dotprod|i8mm|sve|sme" \
  .reference 2>/dev/null
```

`.reference/` is optional and git-ignored; record when a reference was inspected
from another location. Separate findings into four buckets:

1. **Portable algorithm:** applicable across CPU architectures.
2. **ISA mechanism:** useful only where the corresponding instructions and OS
   state are available.
3. **Layout or packing idea:** valuable only if its conversion, cache, and reuse
   costs are included in the intended workload.
4. **Benchmark or shape idea:** often reusable even when the implementation is
   not.

Do not copy an accelerator tile, warp decomposition, scale swizzle, or launch
pipeline merely because it exists in a sibling. Translate the operation and
bottleneck to CPU cache, SIMD, and threading behavior.

## Build And Harness

`quixicore_cpu_bench` is the measurement tool of record. It calls the compiled
library directly, checks results before timing, adaptively batches short calls,
and writes machine-readable and Markdown output. See `benchmarks/README.md` for
the complete case registry and CLI.

Create a clean Release build:

```sh
cmake -S . -B build-perf \
  -DCMAKE_BUILD_TYPE=Release \
  -DQUIXICORE_CPU_BUILD_TESTS=ON \
  -DQUIXICORE_CPU_BUILD_BENCHMARKS=ON
cmake --build build-perf --config Release
ctest --test-dir build-perf --build-config Release --output-on-failure
QUIXICORE_CPU_BUILD_DIR="$PWD/build-perf" scripts/bench --list
```

On a multi-config generator, the executable is normally under
`build-perf/Release/`; `scripts/bench` handles that layout. Always preserve the
configure command and relevant cache/compiler flags with the run evidence.

Typical runs:

```sh
QUIXICORE_CPU_BUILD_DIR="$PWD/build-perf" \
  scripts/bench --preset smoke --kernel all --threads 1

QUIXICORE_CPU_BUILD_DIR="$PWD/build-perf" \
  scripts/bench --preset quick --kernel qgemv,rms_norm --threads 1 \
  --warmup 5 --iters 30 --min-sample-ms 5

QUIXICORE_CPU_BUILD_DIR="$PWD/build-perf" \
  scripts/bench --preset comprehensive --kernel qgemv --threads 8 \
  --warmup 5 --iters 30 --min-sample-ms 5
```

Each run writes:

```text
perf/results/YYYY-MM-DD/<run-id>/run.json
perf/results/YYYY-MM-DD/<run-id>/results.jsonl
perf/results/YYYY-MM-DD/<run-id>/summary.md
```

Anything copied into a status document must come from a Release harness run on
an otherwise idle machine. `--no-check` is for profiler experiments after an
identical checked run; it is not valid correctness or landing evidence.

## Required Report Fields

Every accepted result records:

- repository, git commit or explicit dirty-tree label, and QuixiCore contract;
- kernel family, public operation, selected variant, and baseline variants;
- input/output/accumulator dtype and quant format;
- shape family, concrete dimensions, layout, flags, and packed block sizes;
- CPU vendor/model, microarchitecture when known, architecture, and runtime ISA
  features;
- physical/logical core count, SMT state, hybrid-core topology, and selected
  cores;
- cache hierarchy when relevant, memory capacity/configuration, and NUMA node;
- operating system/kernel, compiler version, build type, compile flags, and LTO
  state;
- thread count, affinity/core-selection policy, NUMA policy, and pool/grain-size
  policy;
- frequency governor or power mode, turbo state when controlled, and whether the
  host was thermally settled and on mains power;
- cache state policy: steady-state/hot, cache-cold, or rotating corpus;
- warmup calls and wall time, sample count, minimum sample duration, batch size,
  median, p20/p80, CV, and independent repeat count;
- correctness command, tolerance, maximum absolute/relative error, and result;
- baseline and candidate commands, derived metrics, keep/reject decision, and
  raw artifact paths.

The harness captures many, but not all, of these fields in `run.json`. Its
current `affinity_policy` and `frequency_policy` values describe defaults; they
do not prove pinning, NUMA binding, fixed frequency, SMT state, detailed memory
topology, or compiler flag state. Add missing facts to the status entry. On
macOS the harness requests high QoS as a scheduling hint, but that is not CPU
affinity and does not guarantee performance-core placement.

## Machine Preparation And Reproducibility

Use a physically idle machine. Stop background builds, indexing, backups, and
other sustained CPU or memory traffic. Do not record a run during active thermal
throttling, immediately after a cold boot, or while switching power modes.

Before a new machine becomes an evidence host:

1. Record CPU model, topology, caches, memory, OS, compiler, and available ISA.
2. Run `quixicore_cpu_info` and retain its output.
3. Run the `mem_triad` working-set ladder at the intended thread counts.
4. Record stable single-thread and aggregate DRAM bandwidth baselines.
5. Measure thread scaling through physical cores and, separately, SMT siblings
   when they exist.
6. Repeat a stable calibration kernel to establish ordinary CV and thermal
   drift for that host.

Platform notes:

- **Linux:** record `lscpu`, NUMA topology, governor/turbo state, and any cgroup
  CPU or memory limits. Use `taskset` or `numactl` only after choosing exact
  physical cores; bind memory as well as threads for NUMA experiments. First
  touch buffers on the node that will execute the kernel.
- **macOS:** record the hardware/power mode and performance/efficiency core
  counts. QoS biases scheduling but is not pinning. Treat P/E migration as a
  variance source and prefer thread counts that remain stable in repeated runs.
- **Windows:** record the power plan, processor groups, efficiency classes, and
  any process-affinity policy. Ensure a run intended for one NUMA node or one
  processor group actually remains there.
- **Virtual machines/containers:** record the host/guest relationship, vCPU
  topology, quotas, and PMU availability. Do not compare bare-metal and virtual
  results as if they came from the same machine class.

Never hide uncontrolled state. “OS default; not pinned” is valid metadata;
silently assuming a fixed frequency or placement is not.

## Cache State And What Is Timed

The native harness allocates and deterministically initializes buffers before
timing, then repeatedly invokes the same thunk. Its default result is therefore
a steady-state measurement. A buffer larger than last-level cache is normally a
DRAM workload, while a small repeatedly used buffer can remain cache-resident.

Every case must state whether it intends to measure:

- **kernel-only steady state:** setup, packing, allocation, and validation are
  outside the timed thunk;
- **end-to-end operation:** required packing, allocation, or format conversion
  is deliberately timed; or
- **cold/streaming behavior:** data is rotated or evicted using an explicit,
  documented method.

Do not call a cache-resident GB/s result DRAM bandwidth. Do not flush caches with
an unmeasured ad hoc loop and then compare it with a steady-state baseline.
Packing may be excluded only when the deployment can reuse the packed object;
otherwise report amortization across the expected reuse count and also provide
an end-to-end result.

Keep page faults, first allocation, thread-pool construction, and one-time
dispatch resolution out of steady-state kernel timing. Measure them separately
when startup latency is relevant.

## Measurement Discipline

- Use `std::chrono::steady_clock` or another monotonic clock.
- Warm up by call count **and** wall time: at least the requested calls and at
  least 50 ms.
- Batch calls so each timed sample spans at least 2 ms; use 5 ms when the host is
  noisy or the expected difference is small.
- Collect at least 20 samples; 30 or more is preferred for landing evidence.
- Report median, p20, p80, CV, and batch size. Never report only the best sample.
- Treat CV above 0.10 as unstable. Rerun after fixing machine state or record the
  instability and defer the decision.
- Use at least three fresh-process repeats for marginal wins, new routing
  thresholds, or comparisons affected by thermal/frequency drift.
- For a close A/B, alternate fresh-process ordering (A/B then B/A) or rerun the
  same binary in both directions. Do not let candidate always benefit from a
  warmer system state.
- Do not discard outliers without documenting the objective rule and retaining
  raw data.
- Make the output observable through `qcb::do_not_optimize()` or `clobber()` so
  the compiler cannot delete or hoist measured work.
- Use identical inputs, thread count, placement, cache policy, compiler, and
  harness settings for candidate and baseline.

Host-clock time is the user-visible kernel latency, including thread-pool and
dispatch overhead inside the public call. Profiler collection perturbs timing;
use counters to explain a clean timing result, not as the primary latency record.

## Baselines And Rooflines

Every optimized result has at least three baselines:

1. **Portable reference:** the `_ref` implementation or an independent naive
   implementation compiled with baseline Release flags.
2. **Machine limit:** a same-machine bandwidth or compute reference appropriate
   to the bottleneck.
3. **Current QuixiCore route:** the public implementation before the change.

Add a fourth contextual baseline when a mature library exposes the same
semantics:

- Accelerate/BLAS on Apple platforms;
- oneDNN, BLIS, OpenBLAS, or another named BLAS on supported systems;
- a framework primitive only when its dtype, layout, fusion, and setup costs are
  stated clearly.

Library baselines do not replace the portable reference or current route. Record
library version, thread settings, and whether packing or primitive creation is
included. Avoid nested parallelism between QuixiCore and a threaded library.

### Memory Roofline

Use `mem_triad` at the same `--threads` value and comparable NUMA placement as
the kernel. Select the largest stable working set that is genuinely beyond the
last-level cache. Compare single-thread kernels with the single-thread roofline
and threaded kernels with aggregate bandwidth.

The STREAM-style convention counts required reads and writes but not
write-allocate traffic. State a different convention explicitly. Report both
achieved GB/s and the fraction of the measured roofline:

```text
memory_fraction = kernel_effective_GBps / same_thread_triad_GBps
```

### Compute Roofline

The current harness does not provide a universal peak-compute probe. Until one
exists for the relevant dtype and ISA, report achieved operations/s and compare
with a mature dense-library baseline; do not label a theoretical instruction
count as a measured roofline. If a compute probe is added, it must use the same
ISA, accumulation type, thread placement, and sustained frequency as the kernel.

### Thread Scaling

Measure at least one thread and the intended production thread count. For
threaded performance claims, sweep through physical cores and optionally SMT:

```text
speedup(T)             = time(1) / time(T)
parallel_efficiency(T) = speedup(T) / T
```

Interpret the curve. Flattening can mean bandwidth saturation, too little work,
serial sections, pool overhead, load imbalance, migration, or NUMA traffic.
More threads are not automatically a better route.

## Derived Metrics

Use explicit, operation-appropriate formulas:

- Dense GEMM/GEMV: `FLOPs = 2*M*N*K` (GEMV uses `M=1`).
- Quantized decode: `effective_weight_GBps = packed_weight_bytes / time`.
- Elementwise/norm/serving copies: conservative required read+write bytes divided
  by time.
- Attention forward, when reported as an approximation:
  `FLOPs ~= 4*B*H*Q*K*D`, adjusted for causal/window sparsity.
- Token serving paths: tokens/s and latency/token, with batch and context stated.
- Sampling/routing: rows/s or tokens/s plus vocabulary/expert dimensions.

For fused operations, count bytes that the fused implementation actually must
move and show the avoided intermediate traffic separately. For packed formats,
include codes, scales, zero points, metadata, and activation bytes as appropriate.
Do not mix decimal GB with binary GiB without labeling it.

Arithmetic intensity is:

```text
arithmetic_intensity = useful_operations / required_bytes_moved
```

It is a model, not a counter. Explain cache reuse and ignored traffic. A result
can exceed the DRAM roofline when its buffers remain in cache; that is evidence
of cache residency, not impossible bandwidth.

## Shape Strategy

The umbrella `registry/benchmark-shapes.yaml` owns contract-compatible shape
families. Preserve those values in `benchmarks/harness/shapes.h`. Exploratory
shapes are welcome, but label them local and do not substitute them for registry
coverage.

Every kernel pass should cover:

- smallest supported and zero/invalid cases in correctness tests;
- tile- or vector-aligned fast paths;
- vector-, block-, and row-ragged tails;
- non-power-of-two and odd dimensions where the API supports them;
- realistic decode, prefill, training, or serving shapes;
- working sets on both sides of important cache boundaries;
- a stress shape and a thread-count sweep;
- every claimed dtype, quant format, mode, and routing branch.

Useful exploratory sweeps, subject to the contract registry, include:

- GEMV/GEMM: `M in {1,2,4,8,16,32,64,128}` with rectangular LLM projection
  dimensions and both aligned/ragged K and N.
- Attention: head dimensions 64/128, short-to-long context, GQA ratios, causal
  and non-causal modes, and page/block boundary cases.
- Norms/softmax/activations: small rows through throughput batches, hidden sizes
  spanning SIMD tails and common model dimensions.
- Quantization: every format/block size, scale layout, zero-point mode, and both
  cache-resident and DRAM-sized packed weights.
- Serving/sampling/MoE: batch/token sweeps, vocabulary/expert counts, skewed and
  balanced routing, sparse metadata, and realistic context lengths.

Record skipped shapes with an explicit reason: outside the contract, unsupported
format/ISA, allocation failure, unavailable matching hardware, or known gap.

## Profiling And Counter Workflow

Timing says whether a change helped. Profiles, counters, compiler reports, and
assembly help explain why. Start with low-overhead evidence, then escalate:

1. Confirm a stable timing difference in the native harness.
2. Use a sampling profiler to verify the intended function is hot.
3. Inspect compiler vectorization reports and generated assembly.
4. Collect a small, architecture-appropriate counter set.
5. Use vendor microarchitecture analysis only when simpler evidence is
   insufficient.

### Linux

`perf` is the default profiler when PMU access is available. For example, to
profile the qgemv case while preserving the counter output:

```sh
PERF_OUT=perf/results/YYYY-MM-DD/qgemv-perf
mkdir -p "$PERF_OUT"

perf stat -o "$PERF_OUT/perf-stat.txt" \
  -e cycles,instructions,branches,branch-misses,cache-references,cache-misses \
  ./build-perf/quixicore_cpu_bench --preset quick --kernel qgemv --threads 1 \
  --out-dir "$PERF_OUT/harness-stat"

perf record -o "$PERF_OUT/perf.data" -g --call-graph dwarf -- \
  ./build-perf/quixicore_cpu_bench --preset quick --kernel qgemv --threads 1 \
  --out-dir "$PERF_OUT/harness-record"
perf report -i "$PERF_OUT/perf.data"
```

Event names, permissions, multiplexing, and counter semantics vary by kernel and
CPU. Record `perf stat` output, event availability, and any multiplexing ratio.
Use model-specific events only with the matching vendor event reference.

### macOS

Use Instruments Time Profiler for hotspots and CPU Counters when the executing
hardware and OS expose the needed events. Use `sample`, `spindump`, or command-
line Instruments tooling for automation where appropriate. Record the template,
sampling interval, OS version, and whether collection changed scheduling or
thermal behavior. `powermetrics` can provide frequency/power context when
available and authorized; it is not a latency timer.

### Windows

Use Visual Studio CPU Usage for a first hotspot pass and Windows Performance
Recorder/Analyzer for scheduling, CPU usage, and system interference. Vendor
tools can add PMU analysis. Record the power plan, affinity/processor-group
configuration, collection profile, and symbol state.

### Vendor Tools

- Intel VTune can analyze hotspots, threading, memory access, and
  microarchitecture behavior on supported targets.
- AMD uProf can provide sampling, IBS, PMC, cache, data-fabric, and power data
  on supported AMD systems.
- Arm Streamline or current Arm server profiling tools can interpret Arm PMU
  and system behavior on supported targets.

Profiler availability never broadens the support claim. Counter names and
ratios are microarchitecture-specific; do not compare raw events between
unrelated CPUs.

### Compiler And Assembly Evidence

For a focused experiment, create a separate diagnostic build rather than adding
report flags to production sources. Useful compiler diagnostics include:

```text
Clang: -Rpass=loop-vectorize -Rpass-missed=loop-vectorize
       -Rpass-analysis=loop-vectorize
GCC:   -fopt-info-vec-optimized -fopt-info-vec-missed
MSVC:  /Qvec-report:2
```

Inspect the final library or object with the platform's `llvm-objdump`,
`objdump`, `otool`, or debugger disassembly. Verify that:

- the intended ISA instructions are present in the variant;
- baseline objects did not acquire unsupported instructions;
- loops are not dominated by spills, scalar conversions, or tail branches;
- loads/stores and widening/accumulation match the packed layout; and
- a source-level “vectorized” loop did not merely move the bottleneck elsewhere.

Counter interpretation starts with context:

- cycles and instructions describe work but IPC alone is not performance;
- branch misses identify unpredictable control flow, not all front-end stalls;
- cache misses need level, request type, and traffic context;
- TLB/page-walk evidence matters for large or sparse working sets;
- low CPU utilization can mean insufficient parallel work or waiting;
- high utilization can still hide bandwidth saturation or spinning.

## CPU Experiment Catalogue

Use these as experiment templates. Change one meaningful factor per experiment
and write the hypothesis before editing.

### ISA And Dispatch

- Compare reference, auto-vectorized baseline, and explicit ISA variants.
- Sweep AVX2, AVX-512, VNNI, and AMX only on matching x86 hardware.
- Sweep NEON, DotProd, I8MM, SVE/SVE2, and SME/SME2 only on matching Arm
  hardware and OS state.
- Verify dispatcher selection and force each supported route in a fresh process.
- Measure dispatch/setup overhead on tiny shapes and steady-state overhead on
  normal shapes.
- Check whether a wider vector ISA loses sustained frequency or increases tail
  and downclock costs enough to change the crossover.

An ISA source is compiled through `quixicore_cpu_add_isa_sources()`; never apply
its target flags globally. A forced route is valid only if feature detection says
the executing CPU supports it.

### Vectorization And Accumulation

- Sweep vector width, unroll factor, and number of independent accumulators.
- Compare intrinsic and compiler-vectorized implementations when both are clear.
- Specialize common aligned lengths while retaining a correct generic tail.
- Test horizontal reduction structures and the placement of widening/conversion.
- Check register pressure and spills after every unroll or fusion increase.
- Preserve required accumulation precision and special-value behavior.

### Packing And Data Layout

- Compare native row-major access with block/interleaved layouts.
- Align hot arrays and packed blocks to cache-line and ISA requirements.
- Test decode-on-load, predecoded tables, and reusable prepacking.
- Include code, scale, zero-point, and permutation traffic in bandwidth math.
- Measure pack-once and end-to-end pack-per-call scenarios separately.
- Avoid layouts whose theoretical SIMD convenience is erased by repacking or
  poor consumer locality.

### Cache Blocking, Prefetch, And Pages

- Sweep M/N/K blocks against L1, L2, and last-level cache capacity.
- Compare loop orders and output tiles for reuse of weights, activations, and
  accumulators.
- Test hardware-prefetch-friendly contiguous walks before adding software
  prefetch.
- Sweep prefetch distance only with a stable long-running workload.
- Check large-page or TLB hypotheses on supported systems, keeping page policy
  identical between A and B.
- Separate cache-resident decode shapes from DRAM-streaming shapes.

### Threading, Grain Size, And NUMA

- Sweep `--threads` from one through physical cores, then SMT if present.
- Sweep outer work partition and minimum chunk size.
- Compare static contiguous partitions with alternate schedules only when load
  imbalance is measured.
- Keep thread-local scratch outside cache lines shared by other workers.
- Check false sharing in outputs, partial reductions, counters, and queue state.
- On NUMA systems, compare local-node execution with deliberate interleaving or
  cross-node placement; never mix these accidentally.
- Avoid nested pools and account for caller-thread participation.

### Fusion And Allocation

- Fuse bias, residual, activation, gate, normalization, quantization, or sampling
  only when the avoided intermediate traffic exceeds added register/code cost.
- Split a fusion when it causes spills, prevents vectorization, or reduces useful
  parallelism.
- Reuse scratch and packed objects when the API/lifetime permits it.
- Keep allocator cost out of kernel-only tests and include it in end-to-end
  tests when the public call must allocate.
- Measure tiny-shape routing where call, validation, and pool overhead dominate.

### Branches, Metadata, And Scalar Side Work

- Hoist format and shape decisions out of inner loops through dispatch or
  templates.
- Replace unpredictable per-element branches only when the branch is measured.
- Precompute stable offsets, masks, and decode tables.
- Reduce pointer arithmetic and integer division in hot loops.
- Balance specialization against instruction-cache pressure and maintenance
  cost.

### Reductions And Numerics

- Compare linear, tree, pairwise, and blocked reductions.
- Use multiple partial accumulators to break dependency chains.
- Preserve fp32 or wider accumulation where the tolerance or long reduction
  requires it.
- Treat exact integer results, deterministic tie-breaking, NaN/Inf handling, and
  stable ordering as part of the contract where specified.
- Re-test ragged tails, extreme values, and every thread count after changing
  reduction order.

### Routing And Shape Specialization

- Find GEMV-to-GEMM, serial-to-threaded, and generic-to-packed crossovers using a
  shape sweep rather than a single point.
- Route tiny rows away from the pool if wakeup/synchronization dominates.
- Route memory-bound kernels only up to the thread count that saturates bandwidth.
- Add aligned fast paths only when the generic edge path remains tested.
- Store routing thresholds with the CPU/ISA evidence that justified them and
  provide a conservative fallback elsewhere.

## Kernel-Specific Starting Hypotheses

These are starting points, not performance claims. Replace them with measured
facts in the optimization notebook.

### Quantized GEMV And Decode Projections

Decode GEMV is usually dominated by packed-weight traffic plus block decode.
The primary metrics are latency and effective packed-weight GB/s. Compare q4,
q8, ternary, FP8, and microscale formats at the same logical N/K while including
all scale and metadata bytes.

Start with packed-load vectorization, multiple output rows per worker, several
independent dot accumulators, branchless block decode, activation reuse, and the
thread count at which DRAM bandwidth saturates. If fewer-bit formats do not beat
wider formats, inspect decode instructions, scale conversions, address math,
and load layout before blaming bandwidth.

### Quantized GEMM, Fused Projections, And LM Head

As M grows, the path can move from memory/decode-bound to compute-bound. Sweep M
to find the GEMV/GEMM crossover. Compare dequantize-then-dense, fused decode and
dot, reusable prepacking, output tiling, activation/scale reuse, and fused
epilogues.

For LM-head operations, compare full-logit materialization against fused argmax,
candidate, masked-top-k, beam, and sampling paths. Report both projection cost
and avoided vocabulary-sized writes.

### Dense, Complex, And Fused Matmul

Dense GEMM needs a mature BLAS context baseline. Start with cache blocking,
microkernel dimensions, packed panels, loop order, vector accumulators, ragged
edges, and thread partition. For small M or N, packing and pool overhead can
dominate.

Complex GEMM and fused gate/GELU/residual paths trade avoided writes against
register pressure and code size. Compare fused operations with the same dense
primitive plus separate epilogues.

### Norms, Softmax, Activations, And Row Utilities

These are usually bandwidth- or reduction-bound. Start with vectorized
loads/stores, several accumulators, one-pass versus blocked reductions, rows per
worker, common hidden-size specializations, and serial routing for tiny rows.

For softmax and norm, keep numerically stable max/variance logic and required
accumulation precision. For GELU/SILU and other transcendental paths, compare
approximations only when the public mode and tolerance allow them.

### Dense, Windowed, And Variable-Length Attention

Attention can be cache-, compute-, or reduction-bound depending on context and
head dimension. Start with head/query partitioning, sequence blocking, Q reuse,
K/V layout, online softmax, causal/window branch placement, and head-dimension
specialization.

Variable-length work needs token-balanced scheduling rather than merely equal
request counts. Backward needs separate dQ and dK/dV hypotheses, reduction
ownership, scratch strategy, and recompute-versus-storage analysis.

### Paged, Cascade, Quantized, And MLA Attention

Decode attention is sensitive to scattered cache access, page-table metadata,
GQA reuse, dequantization, and short per-head work. Start with block-table
prefetch, contiguous page runs, head/query partitioning, cache format, K/V
decode layout, and context-length routing.

Compare FP8/quantized cache reads with an otherwise equivalent f32 path and
include scale metadata. Sparse or cascade paths must count selected tokens and
metadata work, not compare different logical workloads.

### Embeddings, KV Cache, RoPE, And Serving Metadata

Embedding and cache operations are often irregular-memory workloads. Measure
index locality, row width, duplicate ids, scatter/gather balance, vectorized
row copies, cache-line tails, scale conversion, and NUMA placement.

RoPE and QK normalization can benefit from fusion when it avoids full tensor
round trips. Metadata-only EAGLE, sparse-index, bitmask, packbits, and remap
operations may be latency/front-end-bound; report requests or tokens per second
instead of misleading FLOPs.

### MoE Routing And Grouped GEMM

Routing is driven by token/expert counts, selection algorithm, stable ordering,
and softmax work. Compare full sort, partial selection, and specialized small-k
paths with identical tie behavior.

Grouped GEMM depends on expert-size skew. Sweep balanced and highly skewed token
distributions, per-expert dispatch versus grouped execution, packing reuse,
thread allocation, and empty/padded experts. Include route/permute/finalize
traffic when claiming an end-to-end MoE win.

### Sampling, Beam Search, And Speculative Decode

Sampling mixes vocabulary scans, reductions, sorting/selection, RNG, and
metadata. Sweep rows, vocabulary, k/p thresholds, masks, and accepted-token
rates. Preserve deterministic seeds, stable ties, and distribution correctness.

Speculative and EAGLE paths should report tokens accepted/processed and separate
model-independent metadata cost. Fuse projection and selection only when the
same logits and sampling semantics are preserved.

### Linear Attention, Mamba/SSD, And FFT Convolution

These combine recurrence/scan dependencies with parallel outer dimensions.
Start with channel/head partitioning, chunk size, state layout, vectorized state
updates, exp/decay evaluation, and cache residency. Measure decode and prefill
separately.

FFT convolution and complex transforms need radix/block-size, transpose/layout,
temporary traffic, and pointwise-fusion experiments. Compare with a library only
when transform convention and setup/amortization match.

### Vision, Training Utilities, And Optimizers

Patch/space transforms and small fixed MLPs may be dominated by layout and copy
traffic. Start with fused normalization/projection, contiguous output layout,
and vectorized fixed dimensions.

Backward, dropout, cross-entropy, KD, embedding-gradient, and AdamW paths must
include deterministic/correct reduction semantics and realistic batch/parameter
sizes. Optimizer results should state whether parameters, gradients, and states
are cache-resident or streaming.

### Host-Reference Collectives

Current collectives are host-reference semantics, not a transport performance
tier. If optimized, distinguish in-process memory movement from real multi-rank
communication. Report world size, process/thread topology, message bytes,
latency, algorithm, transport, and NUMA placement. Do not call a local memory
copy an interconnect result.

## Per-Kernel Optimization Loop

Run this loop for every public operation being optimized:

1. **Inventory.** Identify the public entry, reference, dispatch route, dtypes,
   formats, layouts, shape contract, tests, benchmark cases, and current status.
2. **Read the contract.** Confirm registry, spec, tolerance, and matrix entries.
3. **Find references.** Record exact files/revisions and portable ideas.
4. **Establish correctness.** Run focused tests before performance work.
5. **Establish baselines.** Measure reference, machine limit, current route, and
   a relevant library on the agreed shape/thread matrix.
6. **Classify the bottleneck.** Use bytes, operations, scaling, profiles,
   compiler output, and counters as appropriate.
7. **Write experiments.** State the expected effect and success criterion.
8. **Execute one factor at a time.** Rebuild, run focused correctness, benchmark
   the same matrix, and retain raw results.
9. **Decide.** Keep, reject, narrow the route, or defer using the rules below.
10. **Verify broadly.** Run full correctness and cross-route/platform checks.
11. **Record.** Update optimization and baseline status with commands, numbers,
    rationale, rejected alternatives, and artifacts.

## Decision Rules

A candidate may land only when it:

- passes focused and broad correctness at contract tolerances;
- improves median performance on priority realistic shapes by at least 3% for a
  simple low-risk change, or roughly 8-10% for added complexity;
- exceeds noise and survives independent repeats;
- does not materially regress required edge or secondary shapes;
- remains dispatch-safe and correct on the baseline architecture;
- has a plausible microarchitectural or traffic explanation; and
- has complexity proportional to a durable real-workload gain.

Reject, narrow, or defer when:

- the apparent win is within CV or reverses with A/B order;
- it appears only on a toy, cache-artificial, or unsupported shape;
- it improves one ISA but is accidentally selected on unsupported hardware;
- it trades unacceptable numerical behavior for speed;
- it shifts packing/allocation work outside the timer without valid reuse;
- it helps one thread count while regressing the production route without a
  justified threshold;
- it requires uncontrolled machine state to reproduce; or
- the implementation and maintenance cost exceed the measured benefit.

Performance evidence is scoped. A win on one Apple M-series CPU does not prove
an Arm server tier; AVX2 evidence does not prove AVX-512; f32 does not prove bf16;
one quant format does not prove another.

## Recording Format

Append every focused pass to `perf/optimization_status.md`:

```text
## YYYY-MM-DD: <kernel or pass name>

Status: not started | baselining | experimenting | candidate | landed | deferred
Kernel / public route:
Dtype / format / layout:
Shape and thread matrix:
Current implementation:
References inspected:
Correctness command and result:
Machine / OS / compiler / flags:
Affinity / NUMA / frequency / cache policy:
Baseline commands and artifacts:
Experiments and hypotheses:
Results (median, p20/p80, CV, derived metrics):
Counter / codegen evidence:
Decision and rationale:
Regressions / rejected alternatives:
Open questions:
Raw results:
```

The minimum valid optimization-run summary is:

```text
kernel:
operation:
dtype_or_format:
shape_set:
correctness_command:
baseline_command:
candidate_command:
hardware_and_isa:
os_and_compiler:
thread_affinity_numa_policy:
frequency_and_cache_policy:
warmups_samples_repeats:
median_p20_p80_cv:
derived_metric:
decision:
artifact:
```

Copy a curated result table with an explicit decision into the status document.
Keep raw JSON/JSONL/Markdown outputs under `perf/results/`. Do not commit large
profiler traces; record their local or external artifact location, collection
command, machine, and a concise finding.

## Final Verification Before Landing A Win

Run the focused test and benchmark first:

```sh
ctest --test-dir build-perf --build-config Release \
  -R 'qgemv|qgemv_w8a8' --output-on-failure

QUIXICORE_CPU_BUILD_DIR="$PWD/build-perf" \
  scripts/bench --preset comprehensive --kernel qgemv \
  --threads 1 --warmup 5 --iters 30 --min-sample-ms 5
```

Replace the example test, case, and thread count with the route being landed.

Then run the full Release suite:

```sh
ctest --test-dir build-perf --build-config Release --output-on-failure
```

For a new ISA or dispatch route, additionally verify:

- direct comparison with the portable reference across aligned and ragged
  shapes;
- forced-reference and forced-candidate public routes in fresh processes;
- the reported `*_variant()` value where available;
- unsupported-feature fallback on another CPU, emulator, or CI architecture;
- baseline objects remain free of the new ISA instructions;
- all supported thread counts produce contract-correct results; and
- a matching-hardware focused benchmark is recorded.

Current forced A/B hooks include:

```sh
QUIXICORE_CPU_QGEMV_VARIANT=ref scripts/bench --kernel qgemv --preset quick
QUIXICORE_CPU_QGEMV_W8A8_VARIANT=ref \
  scripts/bench --kernel qgemv_formats --preset quick
QUIXICORE_CPU_RMS_NORM_VARIANT=ref \
  scripts/bench --kernel rms_norm --preset quick
```

Only force a non-reference name that is present in that dispatcher and supported
by the executing CPU. Dispatch resolution is cached per process, so use a fresh
process for each forced variant.

Run sanitizer coverage for memory-sensitive, packed-format, threading, or broad
substrate changes using compiler-supported AddressSanitizer and
UndefinedBehaviorSanitizer settings. Run another supported architecture or the
CI platform matrix for portable/shared changes. Sanitizer and profiler builds
are correctness/diagnostic evidence, not performance evidence.

Before publishing, update both status documents, ensure `git diff --check`
passes, and include the performance table and scoped claim in the change notes.

## CPU Variance And Validity Checklist

Before accepting a result, explicitly review:

- turbo, sustained frequency, power mode, and thermal drift;
- performance/efficiency core migration on hybrid CPUs;
- SMT sibling placement and unrelated system work;
- NUMA placement, first touch, and cross-node traffic;
- transparent/explicit huge pages and TLB state;
- cache residency caused by adaptive batching;
- allocator reuse, page faults, and thread-pool startup;
- compiler/LTO changes and unintended auto-vectorization differences;
- library nested threading and environment variables;
- counter multiplexing or profiler perturbation;
- virtualization, cgroup quotas, and shared-host interference; and
- output observability and dead-code elimination.

The harness mitigates allocator and data variation with aligned buffers and
deterministic fills. It does not control all machine state. Stable raw data plus
honest metadata is more valuable than a cleaner number obtained by omitting
variance sources.

## External References

- Linux kernel documentation for `perf` and workload tracing:
  <https://docs.kernel.org/admin-guide/workload-tracing.html>
- Intel 64 and IA-32 optimization manuals and architecture references:
  <https://www.intel.com/content/www/us/en/developer/articles/technical/intel64-and-ia32-architectures-optimization.html>
- Intel VTune Profiler documentation:
  <https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler-documentation.html>
- AMD uProf user guide:
  <https://docs.amd.com/r/en-US/57368-uProf-user-guide/uProf-User-Guide>
- Arm C Language Extensions, including SIMD/SVE/SME programming interfaces:
  <https://arm-software.github.io/acle/main/acle.html>
- Arm KleidiAI reference microkernels:
  <https://github.com/ARM-software/kleidiai>
- oneDNN CPU primitives and dispatch reference:
  <https://github.com/uxlfoundation/oneDNN>
- llama.cpp/ggml CPU and quantized kernel references:
  <https://github.com/ggml-org/llama.cpp>

Use the documentation for the exact processor, OS, compiler, and tool version
that produced the evidence. Vendor event names, instruction costs, and profiler
capabilities change across microarchitectures and releases.
