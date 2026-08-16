# CPU Optimization Backlog

The beam: 3-5 active idea families, best first. Pick from the top. Update
after every concluded experiment. Kill criteria are binding — when one fires,
record the kill in `perf/findings.md` and remove the family.

Where measurable, each family carries a quantitative target derived from
recorded data — a percentage of the measured roofline, or beating a named
baseline by a stated margin — set from `perf/findings.md` or
`perf/baseline_status.md`, never invented. The backend's aggregate score
lives in `perf/scoreboard.md`.

## Beam

### 1. x86 wide-ISA closure (AMX route, q4_0 W8A8 VNNI dot, per-format GGUF AVX-512 A/B)
- Parent result: Sapphire Rapids forced A/B measured W8A32 1.91x, INT8
  1.19x, and W4A8 1.34x from AVX-512/VNNI over AVX2, while AMX is detected
  with no implemented kernel route (2026-07-24, Sapphire Rapids
  AVX-512/VNNI dispatch closure). The q4_0 W8A8 SDOT entry left "an
  AVX2/VNNI x86 variant of this specific q4_0 W8A8 dot" open (2026-07-22,
  qgemv_w8a8 q4_0 SDOT); the q8_0 SDOT precedent was 14.4x over scalar
  (2026-07-07, quant_gemv q8_0 NEON DotProd variant).
- Hypothesis: the integer-dot kernels that won under VNNI extend to the
  q4_0 W8A8 lane, and AMX int8 tiles can beat VNNI on M>=16 prequantized
  GEMM; a per-format GGUF re-audit may find the 4/23 formats where AVX-512
  already led.
- Evidence so far: dispatch, feature gating (exact AVX-512F/BW/VL/DQ + VNNI
  predicates, 2026-07-25), and forced-route CTests exist; no AMX kernel and
  no q4_0 W8A8 x86 number exist.
- Next action: implement `q4_0_gemv_w8a8` AVX2/VNNI mirroring the NEON SDOT
  structure and run `quixicore_cpu_bench --preset quick --kernel
  qgemv_formats --threads 1` forced AVX2 vs VNNI on the Xeon Gold 6454S.
- Kill criteria: VNNI q4_0 W8A8 fails to beat the AVX2 route on the same
  host, or an AMX prototype fails to beat the retained VNNI INT8 route at
  M16 after one focused three-pass ladder — record and drop that lane.

### 2. FP8/MX decode-gap closure (predecoded-dense parity, native x86 packed FP8)
- Parent result: the MXFP8 lookup-decoder GEMM is still 1.67x/1.44x slower
  than predecoded dense at one/six threads (2026-07-22, MXFP8/NVFP4 and
  final sibling entry points); "native x86 packed FP8 remains open"
  (2026-07-23, M2 dual-FP8 GEMM output-panel reuse); AArch64 FP8 panels
  went from 0.61x to 1.09x-1.35x via eight-panel decode reuse (2026-07-23,
  M2 dual-FP8 GEMV output-panel reuse).
- Hypothesis: the panel-group decode-reuse pattern plus wider AVX2/AVX-512
  FP8 bit decode closes the remaining gap to predecoded dense on both
  architectures.
- Evidence so far: eight-panel reuse validated on AArch64; x86 FP8 runs
  the checked portable panel fallback; multi-row FP8 tiles flagged open
  (2026-07-23, M2 canonical dual-quant GEMV block dots).
- Next action: port the eight-output-panel FP8 kernel to AVX2 on the EPYC
  or Xeon host and ladder it against the retained portable route at
  M1/M16/M128 N512 K1024.
- Kill criteria: grouped x86 FP8 panels fail to beat the portable route, or
  AArch64 MXFP8 GEMM cannot reach parity with predecoded dense after one
  more focused pass — record the ceiling and drop.

### 3. Arm matrix/dot ISA depth (i8mm smmla, SVE/SME, BaseQN bit-unpack kernels)
- Parent result: open questions "i8mm (smmla) variant" (2026-07-07,
  quant_gemv q8_0 NEON DotProd variant) and "i8mm for qgemm" (2026-07-07,
  qgemv contract realignment) were never measured; BaseQN closed with
  "wider ISA-specific bit-unpack/dot kernels remain future optimization
  work" (2026-07-23, Metal BaseQN semantic port and direct projection).
- Hypothesis: smmla doubles int8 dot throughput over SDOT for M>=2
  quantized GEMM, and dedicated NEON bit-unpack dots lift BaseQN above its
  already-dequantized dense comparator (currently 0.95x/0.92x at t1,
  2026-07-23, Metal BaseQN grouped expert projection and SwiGLU).
- Evidence so far: runtime I8MM detection exists and DotProd/I8MM routes
  are dispatched for Colibri IDOT (2026-07-22, Colibri CPU algorithm
  excavation); no smmla kernel has been laddered.
- Next action: write an i8mm smmla variant of the q8_0 W8A8 multi-row GEMM
  and A/B it against the DotProd route on the M5 Max at M4/M16.
- Kill criteria: smmla fails to beat SDOT on M>=4 shapes on Apple silicon
  in a clean three-pass ladder — record and drop the smmla lane (SVE/SME
  stays parked until hardware exists).

### 4. Compressed-cache one-thread crossover and direct-vs-staged I/O selection
- Parent result: KV3 direct attention is 0.87x-0.97x (Apple) and
  0.46x-0.48x (EPYC) versus materialization at one thread while FP8/MXFP8/
  TurboQuant direct attention wins at t1 (2026-07-23, M4 BitNet a4.8 KV3
  codec and online attention); staged F16C/AVX2 conversion beats direct
  typed I/O for some x86 FP8 cache cases and Apple F16 staging wins at 16
  threads (2026-07-23, M4 typed FP8 cache A1/A2).
- Hypothesis: a measured per-format/thread-count crossover (direct vs
  gather-then-attend, direct vs staged conversion) recovers the recorded
  one-thread losses without giving up bounded memory where it wins.
- Evidence so far: all losing cells are recorded with numbers; no
  crossover selector has been built or measured.
- Next action: run a same-binary KV3 t1 A/B (direct attention vs full
  gather + FP32 paged attention) across S512 D64/D128 on both machines and
  decide whether a t1 crossover route is justified.
- Kill criteria: the A/B shows the direct route within variance of the
  composed route at t1, or the crossover's selection overhead erases the
  win — record and drop.

### 5. Threading placement (affinity pinning, hybrid 8-vs-12, NUMA when hardware exists)
- Parent result: 8 vs 12 threads is shape-dependent on the M4 Max hybrid
  part (qgemv N4096 4.46x at 8t vs 3.69x at 12t; N8192 4.47x at 12t) with
  no pinning, macOS QoS bias only; NUMA policy deferred until multi-socket
  hardware exists (2026-07-07, Threading layer — row-partitioned kernels).
  Threaded 16t CVs up to 0.36 on EPYC/Apple recur across 2026-07-23
  entries.
- Hypothesis: affinity pinning collapses the hybrid-part shape dependence
  and cuts the threaded CV that currently blocks many 16-thread claims.
- Evidence so far: none beyond the 2026-07-07 numbers; every later entry
  runs OS-default affinity.
- Next action: add an opt-in pinning knob to the harness and repeat the
  2026-07-07 qgemv/rms_norm 8/12-thread matrix on Apple silicon.
- Kill criteria: pinning fails to reduce threaded CV or change the 8/12
  ordering on the measured shapes — record and drop; NUMA stays parked
  until a multi-socket host exists.

## Parked (not on the beam)
- quant.py byte-parity fixtures as shared cross-backend test vectors
  (2026-07-07, qgemv contract realignment — open question).
- Vector GELU/SiLU approximations, only if umbrella tolerance and
  model-level checks permit (2026-07-23, M3 canonical fused projection
  epilogue (F1) — decision note).
- BitNet a4.8 Q9 sparse activation preparation, Phase-8 scope (2026-07-23,
  M3 F3 and F6/F7 decision notes).
- TurboQuant A5 query-transform application seam — canonical K is
  untransformed, so the transform stays a standalone contract seam
  (2026-07-23, M4 canonical TurboQuant codec and online attention).
- Parallel full-plane KV copy at larger-than-quick shapes; rejected only
  "for the quick shape" (2026-07-24, Metal Q8_0 KV-cache codec, block
  copy, and paged attention).
- Performance tiers for unmeasured Colibri primitives: E8/IQ3 conversion,
  MLA weight absorption, adjacent-to-split RoPE, paired projection, packed
  rows (2026-07-22, Colibri CPU algorithm excavation — correctness
  candidates only).
- NUMA scheduling policy — blocked on multi-socket hardware (2026-07-07,
  Threading layer).
- Multi-threaded harness probes beyond the current --threads flag
  (2026-07-07, Benchmark harness bring-up — open question; folded into
  family 5).
- Wider AVX2 weight block dots for x86 M1 layouts still on the portable
  row route (2026-07-23, M2 x86 canonical typed GEMV panels — folded into
  family 1).
- Further multi-row FP8 tiles on AArch64 (2026-07-23, M2 canonical
  dual-quant GEMV block dots — folded into family 2).
- x86 SwiGLU epilogue pass for BaseQN grouped experts (0.82x predecoded on
  EPYC, 2026-07-23, Metal BaseQN grouped expert projection and SwiGLU —
  candidate follow-on to family 3's BaseQN lane).

## Migrated sources
- "Open questions:" lines of the 2026-07-07 entries (harness bring-up,
  q8_0 scalar ref, NEON DotProd, rms_norm, threading layer, contract
  realignment) → families 1, 3, 5 and Parked (fixtures, NUMA, probes).
- 2026-07-22 qgemv_w8a8 q4_0 SDOT follow-up note (AVX2/VNNI x86 variant
  remains open) → family 1.
- 2026-07-22 MXFP8/NVFP4 decision ("native SIMD decode and packed
  microkernels remain future optimization work") → family 2.
- 2026-07-23 M2 dual-FP8 GEMV/GEMM decisions ("native x86 packed FP8
  remains open"; "further multi-row FP8 tiles remain open") → family 2 and
  Parked.
- 2026-07-23 Metal BaseQN decisions ("wider ISA-specific bit-unpack/dot
  kernels remain future optimization work"; EPYC SwiGLU 0.82x) → family 3
  and Parked.
- 2026-07-23 M4 A1/A2 and A9/A10 recorded one-thread losses and
  staged-vs-direct notes → family 4.
- 2026-07-24 Sapphire Rapids dispatch closure (AMX detected, no kernel;
  GGUF AVX-512 kept forceable "for future A/B work") → family 1.
- 2026-07-23 F1 and F3/F6-F7 decision notes (vector activation
  approximations; BitNet Q9) → Parked.
- 2026-07-23 A5-A8 decision note (standalone A5 transform seam) → Parked.
- 2026-07-24 Q8_0 KV entry (full-plane copy rejected at quick shape) →
  Parked.
- 2026-07-22 Colibri entry (unmeasured primitive tiers) → Parked.
- No CANDIDATE-verdict entries exist in the notebook index as of
  2026-07-25; nothing was seeded from that class.
