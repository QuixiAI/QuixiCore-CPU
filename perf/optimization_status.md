# Optimization Status

Running notebook of focused optimization runs on CPU kernel paths. Every
kernel implementation, routing change, benchmark change, or performance
claim must add an entry here.

Every future kernel implementation, routing change, benchmark change, or
performance claim must add a focused optimization entry here.

## 2026-07-24: Sapphire Rapids AVX-512/VNNI dispatch closure

Status: retain automatic AVX-512 for row-scaled W8A32 and AVX-512 VNNI for
prequantized W8A8 and dynamic W4A8. Retain AVX2 as the automatic GGUF GEMV
route while keeping the generic AVX-512 implementation forceable for
correctness and future A/B work. This is CPU-only ISA dispatch; the repository
does not contain or claim an AMX kernel.

Correctness: eight new forced-route CTest registrations exercise GGUF AVX2 and
AVX-512, W8A32 AVX2 and AVX-512, INT8 AVX2 and AVX-512 VNNI, and W4A8 AVX2
and AVX-512 VNNI. Each test verifies that a runtime-available requested route
was actually selected. All eight pass on Intel Sapphire Rapids. The complete
native x86-64 suite passes 57/57 locally executable tests (the sibling drift
test is unavailable in that isolated checkout); Apple AArch64 passes 55/55,
including the live parity manifest.

Three one-thread checked passes used the quick `qgemv_formats,colibri_ops`
set, three warmups, nine samples, and a 2 ms minimum sample:

| stage | GGUF decision | W8A32 M4x1024x1408 | INT8 M4x1024x1408 | W4A8 M4x1024x1408 | decision |
|---|---|---:|---:|---:|---|
| pass 1, force AVX2 | 23-format baseline | 0.7308 ms | 0.1542 ms | 0.2933 ms | baseline |
| pass 2, force AVX-512/VNNI | 19/23 medians trail AVX2 | 0.3817 ms | 0.1293 ms | 0.2183 ms | reject broad GGUF promotion; retain the other wide routes |
| pass 3, automatic dispatch | AVX2 selected | 0.3205 ms | 0.1184 ms | 0.2127 ms | confirm selected policy |

The pass-2 same-host ratios over pass 1 are 1.91x for W8A32, 1.19x for INT8,
and 1.34x for W4A8. GGUF's direct quant formats already delegate from the
AVX-512 wrapper to their AVX2 block-dot kernels; the wider generic decoded
fallback does not justify an automatic promotion on this CPU. Pass 3 is a
dispatch confirmation rather than an across-pass speedup claim because the
host frequency and scheduler were not pinned.

- Hardware/toolchain: Intel Xeon Gold 6454S (Sapphire Rapids), GCC 15.2.0,
  Linux x86-64 Release without LTO. Runtime detection reports AVX2, AVX-512F,
  AVX-512 VNNI, and AMX tile/int8/bf16.
- Command: `quixicore_cpu_bench --preset quick --kernel
  qgemv_formats,colibri_ops --threads 1 --warmup 3 --iters 9
  --min-sample-ms 2`, with the four route overrides set to AVX2 for pass 1,
  AVX-512/VNNI for pass 2, and unset for pass 3.
- Artifacts: `perf/results/2026-07-24/x64-wide-routes-pass1-avx2-t1/`,
  `x64-wide-routes-pass2-avx512-t1/`, and
  `x64-wide-routes-pass3-auto-t1/`.
- Decision: keep the implemented AVX-512/VNNI compute routes selected where
  they won. Runtime AMX feature detection is hardware evidence only; no AMX
  compute tier is asserted.

## 2026-07-24: CPU ports for Qwen temporal vision and scalar value clipping

Status: retain portable CPU implementations of the Metal-published
`extract_patches_3d`, `vision_patch_projection_3d`, `qwen_vision_rope_2d`,
and `value_clip` semantics. The patch routes consume NTHWC tensors, preserve
the `[B,OT*OH*OW,KT*KH*KW*C]` order for arbitrary stride/padding, and use
`[O,KT,KH,KW,C]` projection weights with optional bias. Qwen RoPE uses global
split-half pairing with x/y frequency sections. Value clipping accepts
infinite bounds, rejects NaN/reversed bounds, and preserves tensor storage.
These are CPU kernels; no Metal source or execution path is compiled here.

Correctness: `test_basert` checks all four operations at FP32, FP16, and BF16
boundaries against independent scalar oracles. Coverage includes padded and
overlapping temporal patches, exact patch order, projection and optional-bias
semantics, the Qwen layout distinct from Gemma axis blocks, infinite clip
bounds, and invalid-bound/empty-output rejection. Release `basert` passes on
Apple AArch64 and Intel Sapphire Rapids x86-64. Every benchmark oracle reports
zero error.

All three Apple passes used the checked quick BF16 cases at 16 threads:

| stage | value clip, 1120x768 / 750x1024 | 3-D patch, T2 / T8 | 3-D projection | Qwen 2-D RoPE | decision |
|---|---:|---:|---:|---:|---|
| pass 1, generic storage clamp and voxel-wise patch copy | 0.2069 / 0.1980 ms | 0.0937 / 0.2084 ms | 0.2776 ms | 0.3512 ms | semantic baseline |
| pass 2, direct BF16 clip, interior row copy, split projection geometry, and layout-branch hoist | 0.1155 / 0.1156 ms | 0.0493 / 0.0602 ms | 0.2795 ms | 0.2126 ms | retain clip, patch, and RoPE; projection neutral |
| pass 3, four-way interior projection reduction | 0.0963 / 0.0896 ms | 0.0476 / 0.0527 ms | 0.1513 ms | 0.2086 ms | retain all selected routes |

The retained Sapphire Rapids medians are 0.2436/0.3937 ms for clipping,
0.0588/0.0963 ms for T2/T8 extraction, 0.2792 ms for projection, and
0.7525 ms for Qwen RoPE at 16 threads. Cross-machine values are portability
evidence and are not used as optimization ratios.

- Hardware/toolchains: Apple M5 Max, Apple Clang 21.0.0; Intel Xeon Gold
  6454S Sapphire Rapids, GCC 15.2.0. Release, without LTO.
- Command: `quixicore_cpu_bench --preset quick --kernel
  basert_aux,basert_vision --threads 16`.
- Artifacts: `perf/results/2026-07-24/basert-wave4-pass{1,2,3}-apple-t16/`
  and `basert-wave4-final-quadb60-t16/`.
- Decision: retain BF16 bit-preserving clip, canonical interior row copies,
  global-split branch hoisting, and the four-accumulator projection dot. The
  general padded paths remain available; no accelerator or framework-wide
  speed claim is made.

## 2026-07-24: CPU ports for published position and relative-audio kernels

Status: retain portable CPU implementations of the Metal-published
`factorized_position_2d`, `pool_tokens_by_position`, `vision_rope_2d`,
`audio_causal_depthwise_conv1d`, and `audio_relative_attention` semantics.
They cover invalid-token zeroing, arbitrary coordinate scatter with FP32
output and mask, Gemma local-axis split-half rotation, left-only dilated
depthwise convolution, and chunk-relative attention with learned
per-dimension query scale, length masks, optional softcap, and FP32 online
softmax. Each compute route has FP32/FP16/BF16 storage boundaries.

Correctness: `test_basert` exercises every storage type with independent
oracles, including shuffled/invalid coordinates, clamped RoPE positions,
dilated causal geometry, relative shifts, context bounds, automatic scales,
softcap, and masked sequence tails. Apple AArch64 and Sapphire Rapids x86-64
Release tests pass; checked BF16 benchmarks report zero error except the
expected quantized relative-attention reporting metric (`4.07e-3`), which
passes the elementwise tolerance.

The three Apple passes used quick shapes at 16 threads:

| stage | factorized position | coordinate pool | 2-D RoPE | causal depthwise | relative attention | decision |
|---|---:|---:|---:|---:|---:|---|
| pass 1, bulk storage conversion and portable FP32 cores | 0.1191 ms | 0.1319 ms | 0.2148 ms | 0.2925 ms | 1.4445 ms | semantic baseline |
| pass 2, direct same-type routes and online relative softmax | 0.1981 ms | 0.2117 ms | 0.2333 ms | 0.8368 ms | 1.2534 ms | reject direct half conversion; retain online softmax |
| pass 3, restored bulk routes, channel-tile experiment, K5 interior specialization, and invariant-scale hoist | 0.1408 ms | 0.1606 ms | 0.3111 ms | 0.2125 ms | 1.0271 ms | reject pool tiling; retain causal and relative changes |
| selected confirmation | 0.1550 ms | 0.1422 ms | 0.2591 ms | 0.2028 ms | 0.9989 ms | final routes; no factor/rope speedup claim |

The Sapphire Rapids final medians are 0.4592, 0.7065, 0.7441, 1.0109, and
4.2807 ms in table order at 16 threads. Those results establish x86-64
correctness and performance evidence without comparing machines.

- Hardware/toolchains: Apple M5 Max, Apple Clang 21.0.0; Intel Xeon Gold
  6454S Sapphire Rapids, GCC 15.2.0. Release, without LTO.
- Command: `quixicore_cpu_bench --preset quick --kernel
  basert_vision,basert_audio --threads 16`.
- Artifacts: `perf/results/2026-07-24/basert-wave3-pass{1,2,3}-apple-t16/`,
  `basert-wave3-final-apple-t16/`, and `basert-wave3-final-quadb60-t16/`.
- Decision: retain online relative attention, hoisted learned scales, and the
  specialized K5 interior. Retain generic SIMD bulk conversion for the three
  bandwidth-oriented typed routes; measured direct conversion and pool
  tiling regressions were removed.

## 2026-07-24: published BaseRT patch projection and cross-attention

Status: retain the CPU ports of Metal-published `vision_patch_projection` and
`cross_attention`. Patch projection consumes NHWC input and `[O,KH,KW,C]`
weights, supports stride and padding, and stores `[B,OH*OW,O]`. Cross-attention
supports independent query/key lengths, MHA/GQA head mapping, per-batch valid
key lengths, optional FP32 score bias, automatic or explicit scale, score
softcap, and D64/D128/D256. Both expose FP32 cores and universal
FP32/FP16/BF16 storage boundaries with FP32 products/accumulation; attention
retains double-precision online-softmax normalization state.

Correctness: `test_basert` compares all three storage types with independent
patch/projection and two-pass attention oracles. It covers padding and overlap,
optional bias, explicit/automatic scale, score softcap, GQA, a zero-length
batch item, and exact typed output boundaries. Release `basert` passes on Apple
AArch64 and Intel Sapphire Rapids x86-64. Every quick benchmark check passes;
maximum reported relative error is `9.93e-7`.

All three optimization passes used the checked quick BF16 cases at 16 threads:

| stage | Apple patch projection | Apple cross-attention, long/short | decision |
|---|---:|---:|---|
| pass 1, scalar rows; attention query rows serial | 16.1148 ms | 1.2042 / 0.2461 ms | semantic/materialization baseline |
| pass 2, collision-free output/query-row scheduling | 2.2263 ms | 0.3305 / 0.0972 ms | retain parallel row scheduling |
| pass 3, fused direct patch projection and four-way FP32 attention dot reduction | 2.0294 ms | 0.2670 / 0.0675 ms | retain both changes |

The retained Sapphire Rapids result is 24.4507 / 6.7073 ms for patch
projection and 3.1828/0.3608 / 1.1820/0.1883 ms for long/short attention at
one/16 threads. The prior EPYC 7702 pass-1 portability result is 34.0841 ms and
4.2118/0.5515 ms at 16 threads. Cross-machine values are reported separately
and are not used as optimization ratios.

- Reference: published operations in
  `../QuixiCore-Metal/.quixicore/kernels.yaml` and the corresponding
  `kernels/{vision,audio}` source/test trees at sibling revision `6e8a7b6`.
- Hardware/toolchains: Apple M5 Max, Apple Clang 21.0.0; Intel Xeon Gold
  6454S Sapphire Rapids, GCC 15.2.0; AMD EPYC 7702, GCC 13.3.0. Release,
  without LTO.
- Command: `quixicore_cpu_bench --preset quick --kernel
  basert_vision,basert_audio --threads {1,16} --warmup 3 --iters 10
  --min-sample-ms 2`.
- Artifacts: `perf/results/2026-07-24/basert-wave2-pass{1,2,3}-apple-t16/`,
  `basert-wave2-pass1-epyc-t16/`, `basert-wave2-pass3-quadb60-t16/`, and
  `basert-wave2-selected-quadb60-t1/`.
- Decision: remove the temporary patch tensor, retain row-parallel projection,
  and retain the shorter-latency FP32 dot reduction. This is CPU-kernel
  evidence, not a claim about Metal execution or vendor-library GEMM.

## 2026-07-24: live Metal BaseRT auxiliary, embedding, vision, and audio batch

Status: retain nine CPU ports originally excavated from the live Metal source
tree and subsequently published in its operation manifest:
`calibration_absmax`, `logits_softcap`,
`embedding_lookup_types`, `masked_mean_pool_rms_l2`, `extract_patches_2d`,
`interpolate_position_2d`, `avg_pool2d_tokens`, `audio_conv1d_direct`, and
`audio_depthwise_conv1d`. All expose FP32 reference entry points plus
FP32/FP16/BF16 storage routes. The CPU layouts and edge rules match the live
Metal sources: calibration is channelwise with running-state merge and NaN
propagation; invalid embedding ids contribute zero; arbitrary nonzero masks
select tokens and empty rows emit zero; patch/pool/audio tensors are NHWC/NWC;
bilinear resize supports half-pixel and aligned corners; audio supports
stride/padding/dilation, optional bias, and depthwise SiLU.

Correctness: `test_basert` runs all nine operations for F32, F16, and BF16,
using independent scalar formulas. It covers running calibration, token/type
bounds, non-prefix masks and an empty masked row, patch ordering, bilinear
coordinates, partial ceil-mode pooling, strided general convolution,
depthwise SiLU, one/four-thread execution, and invalid cap/empty-output shapes.
Release tests pass on Apple AArch64 and AMD x86-64. All 17 quick benchmark
cases pass their embedded oracle; the largest relative reporting metric is
`1.08e-2` for BF16 masked pooling and the elementwise tolerance passes.

Every pass below ran every quick case at 16 threads. Values preserve the
case order within each family: auxiliary is two calibration plus two softcap;
embedding is two gathers plus two masked pools; vision is two patch, two
interpolation, and two average-pool cases; audio is two general plus one
depthwise convolution.

| stage | Apple medians (aux; embedding; vision; audio) | EPYC medians (aux; embedding; vision; audio) | decision |
|---|---|---|---|
| pass 1, generic bulk storage adapter and portable FP32 cores | 2.0392/20.3858/1.2897/4.9545; 0.1391/0.1625/0.1263/0.4033; 0.1469/0.4731/0.2199/0.3034/0.2586/0.3473; 3.3980/5.8215/0.4491 ms | 5.3166/36.5485/9.8400/33.9101; 0.2327/0.4215/0.4975/1.8266; 0.3556/0.8696/0.4915/0.4989/0.5825/1.3479; 7.1849/10.4370/1.9835 ms | semantic baseline |
| pass 2, direct same-type storage for every operation | 0.9584/19.3942/1.4342/5.7133; 0.0621/0.1267/0.1213/0.5473; 0.0752/0.4262/0.7132/0.7912/0.2844/0.4015; 10.9954/13.3795/0.8722 ms | 1.8485/39.0196/10.1988/40.5340; 0.1014/0.2931/0.6355/2.3841; 0.1306/1.5568/3.0635/3.0720/1.0953/1.8591; 29.3163/34.4750/4.1107 ms | retain direct sparse gather/copy; reject repeated scalar half conversion for compute-heavy routes |
| pass 3, hybrid storage selection, row-major calibration tiles, contiguous-feature masked/pool accumulation, raw patch copies, and channel-major depthwise traversal | 0.2999/2.5902/2.5095/4.9606; 0.0693/0.1394/0.0888/0.1568; 0.0773/0.6256/0.2410/0.3336/0.2588/0.4455; 3.6808/6.0359/0.4869 ms | 1.1230/9.4264/8.4030/34.5684; 0.1040/0.3089/0.2479/0.7533; 0.0742/0.6306/0.5004/0.4988/0.4387/0.3490; 7.1777/13.1504/0.9190 ms | retain selectively; calibration, masked pool, raw patch copy, x86 average-pool tail, and x86 depthwise improve; generic bulk conversion remains selected elsewhere |

The stable selected runs record Apple t1/t16 ranges of 2.0430-47.4224 /
0.3704-6.2913 ms for auxiliary, 0.0423-0.6158 / 0.0851-0.1920 ms for
embedding, 0.1596-2.4271 / 0.0809-0.3425 ms for vision, and
2.2620-75.4266 / 0.4386-8.3257 ms for audio. EPYC t1/t16 ranges are
7.5524-514.8111 / 1.1196-34.0617, 0.0827-1.8945 / 0.0999-0.6899,
0.2664-4.2967 / 0.0724-0.6345, and 4.3904-146.2330 /
0.9079-9.5038 ms respectively. These are implementation measurements, not a
claim against framework or vendor-library convolution.

- References: the Metal tree under
  `../QuixiCore-Metal/kernels/{quantization/quant_rt,sampling/sampling,
  serving/embedding,serving/mean_pool_rms_l2,vision/patch_ops,audio/conv1d}`,
  its correctness suites and `perf/bench_kernels.py`, now published at sibling
  revision `6e8a7b68dd891f88981ca1cf05f70c357efdc284`.
- Hardware/toolchains: Apple M5 Max, AArch64 NEON, Apple Clang 21.0.0; AMD
  EPYC 7702, x86-64 AVX2/FMA/F16C, GCC 13.3.0. CMake Release without LTO.
- Pass command: `quixicore_cpu_bench --preset quick --kernel
  basert_aux,basert_embedding,basert_vision,basert_audio --threads 16
  --warmup 2 --iters 5 --min-sample-ms 1`.
- Selected command: the same kernel set with `--threads {1,16} --warmup 3
  --iters 10 --min-sample-ms 2`.
- Artifacts: `perf/results/2026-07-24/basert-{pass1,pass2,pass3}-*-t16/`
  and `basert-selected-*-t{1,16}/`.
- Decision: keep direct typed token/type gather and raw typed patch copies;
  keep cache-friendly calibration and FP32 contiguous-feature reductions;
  retain the bulk storage adapter for transcendental/interpolation/general
  convolution paths. No blanket speedup or native-half-compute claim.

## 2026-07-24: direct F16-adapter LoRA

Status: retain `lora_apply_direct_f16` and its universal storage wrapper. The
operation consumes raw IEEE-FP16 A `[R,K]` and B `[N,R]` adapters, computes
`low=f16(x@A^T)`, `delta=f16(low@B^T)`, and emits
`base + scale*delta` in the requested FP32/FP16/BF16 storage type. Base is
optional, rank is 1-256, and the bounded low-rank intermediate stays on the
calling or worker stack.

Correctness: `test_lora` crosses F32/F16/BF16 activation, base, and output
storage; M1 K257 N193 R4 and M4 K1024 N513 R16; with/without base; one/four
threads; and invalid rank, pointer, and scale metadata. An independent
two-stage oracle rounds both projection results to FP16. Release tests pass
on Apple AArch64 and AMD x86-64. Quick benchmark checks pass with maximum
relative metric `7.81e-3`, within the BF16 contract tolerance used by the
live sibling test.

Three passes used M1/4/8/64, K=N4096, R16, BF16 tensor storage, raw FP16
adapters, three warmups, 15 samples, and a 4 ms minimum:

| stage | Apple medians, M1/M4/M8/M64 | EPYC medians, M1/M4/M8/M64 | decision |
|---|---|---|---|
| pass 1, row-scalar two-stage projection | 0.1181/0.4643/0.9073/7.5135 ms at t1 | 0.3508/1.3949/2.7916/22.3566 ms at t1 | semantic baseline |
| pass 2, decode output partition and prefill row partition | 0.0960/0.1723/0.2321/1.0316 ms at t16 | 0.3364/0.9232/0.9742/3.1833 ms at t16 | retain collision-free scheduling; keep a direct t1 route |
| pass 3, four FP32 accumulators and fixed-R16 up projection | 0.1204/0.4974/1.0214/7.4949 ms at t1 and 0.1196/0.2472/0.2767/1.1988 ms at t16 | 0.3444/1.3760/2.7540/22.0207 ms at t1 and 0.3343/0.8486/0.9054/2.9741 ms at t16 | retain only on x86; reject on Apple |

The selected inline sequential Apple path records
0.1088/0.4378/0.8855/7.0757 ms at one thread and
0.0964/0.1519/0.2044/0.9918 ms at 16. The selected x86 path records
0.3451/1.3747/2.7506/22.0252 ms and
0.2904/0.8464/0.8952/2.9669 ms. Against the independent scalar two-matmul
composition, selected ratios are 2.29x-2.38x and 2.60x-16.64x on Apple and
1.73x-1.74x and 2.06x-12.87x on EPYC at t1/t16. This comparator includes
the same BF16 and FP16 storage semantics; it is not a vendor BLAS baseline.

- References:
  `../QuixiCore-Metal/kernels/matmul/lora/lora.{h,cpp,metal}`, its correctness
  suite, Python/Torch binding routes, and `lora` benchmark cases, published at
  sibling revision `6e8a7b68dd891f88981ca1cf05f70c357efdc284`.
- Hardware/toolchains: Apple M5 Max, AArch64 NEON, Apple Clang 21.0.0; AMD
  EPYC 7702, x86-64 AVX2/FMA/F16C, GCC 13.3.0. CMake Release without LTO.
- Command: `quixicore_cpu_bench --preset quick --kernel lora --threads
  {1,16} --warmup 3 --iters 15 --min-sample-ms 4`.
- Artifacts: `lora-{apple,x86}-pass{1,2,3}-t*` and
  `lora-{apple,x86}-selected-t*` under `perf/results/2026-07-24/`.
- Decision: retain bounded stack storage, direct one-thread execution,
  decode-output and prefill-row scheduling, and the measured x86-only
  accumulator selection. No native FP16 arithmetic or vendor-BLAS claim is
  made.

## 2026-07-24: Metal GatedDeltaNet and split sigmoid gate

Status: retain `gdn_recur`, `gdn_short_conv`, `gdn_qkv_prepare`,
`gdn_gate_beta`, `gdn_gated_rmsnorm`, and split `sigmoid_mul` with analytic
backward. The GDN routes preserve packed variable-length request metadata,
MQA/GQA head mapping, D64/D128 Q/K and value heads, optional fresh state,
oldest-to-current depthwise-convolution weights, and functional FP32
state/history pools. FP32, FP16, and BF16 storage wrappers retain FP32
reductions and states.

Correctness: `test_gdn` uses independent recurrence, short-convolution,
QKV-normalization, decay/beta, gated-normalization, and sigmoid/gradient
oracles. It covers fresh and continued state, inactive convolution slots,
functional input pools, storage conversion for every operation, D64 heads,
F32/F16/BF16 tensors, and both one- and four-thread execution. The focused
Release suite passes on Apple AArch64 and AMD x86-64. Every quick benchmark
case has an embedded independent scalar check and reports zero error.

The six columns below are recurrence, short convolution, QKV preparation,
decay/beta, gated RMSNorm, and sigmoid multiplication. Quick shapes are R4
L16 Hk2 Hv8 Dk64 Dv64; R8 L64 C1024 K4; T256 Hk4 Hv8 D128; T2048 H32;
T256 H16 D128; and N1048576:

| stage | Apple medians | EPYC medians | decision |
|---|---|---|---|
| pass 1, direct portable scalar contracts | 1.6881/4.4832/0.1045/0.5914/1.3573/4.3560 ms at t1 | 4.7735/10.9763/0.3239/2.9735/5.0434/8.7933 ms at t1 | semantic and scheduling baseline |
| pass 2, collision-safe state-row/channel scheduling and row scheduling for stateless helpers | 0.2449/0.4850/0.0566/0.1325/0.2227/0.4385 ms at t16 | 0.8031/1.4342/0.0825/0.4378/0.6421/1.0785 ms at t16 | retain parallel decomposition; one-thread recurrence/convolution overhead requires a serial crossover |
| pass 3, direct one-thread crossover, fixed K4 history, and row-structured gate preparation | 1.6403/4.0029/0.1019/0.5644/1.3320/4.1730 ms at t1; 0.2443/0.4456/0.0568/0.1301/0.2149/0.4483 ms at t16 | 4.7915/8.4935/0.3225/2.7418/4.9900/8.7968 ms at t1; 0.8067/1.1710/0.0843/0.4117/0.6386/1.0835 ms at t16 | retain; serial K4 improves convolution 1.12x/1.29x and selected threaded cases improve 1.83x-10.27x/3.88x-9.95x over equal-work scalar baselines |

Parallel state mutation is enabled only when nonnegative request slots are
unique. Reused slots fall back to request order, and inactive negative
convolution slots write zero output without touching state. At one thread the
direct nested recurrence avoids task-index division; K4 convolution keeps
three history scalars live. No ISA-specific GDN tier or blanket one-thread
speedup is claimed.

- References: `../QuixiCore-Metal/kernels/linear_attention/gdn/gdn.{h,cpp,
  metal}`, `../QuixiCore-Metal/kernels/activations/glu/*`, their correctness
  suites, launch helpers, benchmark cases, and performance notebook at sibling
  revision `6e8a7b68dd891f88981ca1cf05f70c357efdc284`.
- Hardware/toolchains: Apple M5 Max, AArch64 NEON, Apple Clang 21.0.0; AMD
  EPYC 7702, x86-64 AVX2/FMA/F16C, GCC 13.3.0. CMake Release without LTO.
- Command: `quixicore_cpu_bench --preset quick --kernel gdn --threads
  {1,16} --warmup 3 --iters 15 --min-sample-ms 4`.
- Artifacts: `gdn-{apple,x86}-pass{1,2,3}-t1` and
  `gdn-{apple,x86}-pass{2,3}-t16` under `perf/results/2026-07-24/`.
- Decision: retain all six semantic ports, mixed-storage wrappers,
  collision-safe functional state scheduling, the one-thread crossover,
  fixed-K4 short convolution, and row-structured control kernels.

## 2026-07-24: Metal Q8_0 KV-cache codec, block copy, and paged attention

Status: retain `kv_cache_scatter_q8_0`, `kv_cache_gather_q8_0`,
`kv_cache_copy_blocks_q8_0`, and `paged_attention_q8_0`. The codec stores
independent signed-int8 K/V planes and raw IEEE-FP16 scales per 32 values.
Scatter accepts FP32/FP16/BF16 input, ignores negative slots, and preserves
Metal's unrounded FP32 `amax/127` encoding rule. Gather supports packed
sequences and zero-fills negative block entries. Block copy is functional: it
clones the input planes before applying source/destination mappings. Direct
attention consumes the compressed pages without materializing scores or an
FP32 cache and skips sparse negative blocks.

Correctness: `test_quant_cache_attention` checks exact int8 codes and FP16
scale bits against an independent encoder for all three input storage types,
packed gather reconstruction and zero-fill, functional block mappings, sparse
attention, and direct attention against complete Q8_0 decode plus FP32 paged
attention. The focused Release test passes on Apple AArch64 and native AMD
x86-64. The quick harness reports exact codec/copy results and maximum
attention relative error `2.65e-4`; elementwise checks pass their mixed
absolute/relative tolerance.

Three one-thread passes used codec T512 H8 D128, copy B32/M8/H8/D128, and
attention B2 Hq8 Hkv2 S512 at D64/D128:

| stage | Apple medians | EPYC medians | decision |
|---|---|---|---|
| pass 1, scalar per-element codec and materialized score row | scatter/gather/copy 1.0557/1.2822/0.0193 ms; attention 1.4107/2.6789 ms | 4.8374/2.4202/0.0354 ms; attention 3.2432/6.2267 ms | semantic baseline |
| pass 2, collision-safe scheduling, online tiled softmax, and block `std::copy` | 1.1146/1.4078/0.0158 ms; attention 0.3369/0.6597 ms | 5.4200/2.7892/0.0338 ms; attention 0.7067/1.3492 ms | retain online attention and block copy; codec scheduling alone regresses at one thread |
| pass 3, one-load scatter groups, once-per-group scale decode, F32 hot paths, and 32-row attention tiles | 0.9289/0.1103/0.0160 ms; attention 0.3259/0.6462 ms | 4.0439/0.3167/0.0300 ms; attention 0.7079/1.3410 ms | retain; gather is 11.63x/7.64x over pass 1 and attention is 4.14x-4.33x/4.59x-4.64x |

The selected 16-thread Apple medians are 0.1996/0.0590/0.0156 ms for
scatter/gather/copy and 0.0921/0.1458 ms for D64/D128 attention. EPYC records
0.5707/0.0844/0.0337 ms and 0.1475/0.3024 ms. A parallel full-plane-copy
candidate measured 0.0516 ms on Apple and 0.0906 ms on EPYC, so it was rejected
for the quick shape in favor of optimized library copies. Direct attention is
compared with the intentionally stronger materialize-FP32 baseline; codec and
copy baselines are equal-work contract oracles, so no blanket codec speedup is
claimed.

- References: `../QuixiCore-Metal/kernels/serving/kv_cache/kv_cache.{h,cpp,
  metal}`, its correctness suite, launch helpers, and sibling performance
  notebook.
- Hardware/toolchains: Apple M5 Max, AArch64 NEON, Apple Clang 21.0.0; AMD
  EPYC 7702, x86-64 AVX2/FMA/F16C, GCC 13.3.0. CMake Release without LTO.
- Command: `quixicore_cpu_bench --preset quick --kernel q8_kv --threads
  {1,16} --warmup 3 --iters 10 --min-sample-ms 5`.
- Artifacts: `q8-pass{1-scalar,2-online-parallel,3-selected}-*-t1` and
  `q8-final-selected2-*-t16` under `perf/results/2026-07-24/`.
- Decision: retain the canonical layout, collision-safe token/head
  scheduling, F32 codec specialization, scale hoisting, bounded online
  softmax, and serial optimized functional clone at the measured size.

## 2026-07-23: positioned RoPE, M-RoPE, and positioned QK norm/RoPE

Status: retain `rotary_positioned`, `mrope`, and
`qk_norm_rope_positioned`. Standalone rotation supports shared or per-batch
explicit positions, partial rotary prefixes, split-half or adjacent-pair
layout, and three-axis multimodal position selection. The fused route applies
per-head Q/K RMSNorm, positioned or multimodal partial rotation, and packed
Q/K/V output without a normalized intermediate tensor. Existing
`qk_norm_rope` delegates to the positioned implementation.

Correctness: `test_extended_ops` covers split and interleaved layouts,
per-batch positions, three-axis section maps, unrotated tails, positioned Q/K
normalization, multimodal fused rotation, and invalid metadata. Independent
scalar oracles pass on Apple AArch64 and AMD x86-64. Standalone rotation is
exact; fused checks use the established norm/RoPE tolerance and the final
harness reports maximum relative error `3.07e-3`.

Three one-thread passes used the five quick cases: positioned split D128,
positioned interleaved D256, interleaved M-RoPE D64, positioned fused QK D128,
and multimodal fused QK D64:

| stage | Apple medians, in case order | EPYC medians, in case order | decision |
|---|---|---|---|
| pass 1, scalar table gather and normalized-head scratch | 0.1980/0.1016/0.1512/1.0526/0.6337 ms | 0.6032/0.2049/0.4549/2.1938/1.5661 ms | semantic baseline |
| pass 2, allocation-free fused heads and independent head scheduling | 0.2003/0.0989/0.1516/1.0321/0.6662 ms | 0.6443/0.2413/0.4634/2.7573/1.7127 ms | retain the bounded fused structure; one-thread scheduling overhead is mixed |
| pass 3, FP32 RMS reduction and fixed layout/axis specializations | 0.2023/0.0968/0.1783/0.5945/0.4509 ms | 0.6011/0.2093/0.3766/2.0302/1.1642 ms | retain fused specialization and fastest standalone selections; reject slower standalone rewrites |

The final Apple medians are 0.2004/0.1026/0.1569/0.5834/0.4511 ms at one
thread and 0.0970/0.0631/0.0926/0.2008/0.1531 ms at 16. EPYC final medians
are 0.6011/0.2093/0.3766/2.0302/1.1642 ms at one thread and
0.1122/0.0723/0.0965/0.3809/0.2637 ms at 16. The fused positioned and
multimodal cases improve 1.80x/1.40x on Apple and 1.08x/1.35x on EPYC versus
pass 1; standalone changes are selected per measured route without a blanket
speedup claim.

- References: `../QuixiCore-Metal/kernels/attention/rotary/*` and
  `../QuixiCore-Metal/kernels/norms/qk_norm_rope/*`, including bindings and
  correctness tests.
- Hardware/toolchains: Apple M5 Max, AArch64 NEON, Apple Clang 21.0.0; AMD
  EPYC 7702, x86-64 AVX2/FMA/F16C, GCC 13.3.0. CMake Release without LTO.
- Command: `quixicore_cpu_bench --preset quick --kernel rotary_extended
  --threads {1,16} --warmup 3 --iters 10 --min-sample-ms 5`.
- Artifacts: `rotary-pass{1-scalar,2-fused-parallel,3-fp32-specialized}-apple-t1`,
  `rotary-final-selected2-apple-t{1,16}`, and
  `rotary-{pass1-scalar,pass2-fused-parallel,pass3-selected,final-selected}-x86-*`
  under `perf/results/2026-07-23/`.
- Decision: retain explicit-position and multimodal contract coverage,
  allocation-free fused Q/K processing, and per-route measured selections.

## 2026-07-23: Metal BaseQN grouped expert projection and SwiGLU

Status: retain `base_q_moe_gemm` and `base_q_moe_swiglu` for the canonical
BaseQ2/3/4/5/6/8 expert stack. Both consume the existing 32-row padded MoE
schedule, require one valid expert id per tile, preserve FP32/FP16/BF16 input
and output storage, and accumulate in FP32. Fused SwiGLU interprets each
expert's row axis as `[gate(intermediate), up(intermediate)]` and rounds only
the final product.

Correctness: `test_base_q` crosses all six bit widths, symmetric and affine
weights, both operations, alternating expert tiles, exact independent dense
oracles, typed FP16/BF16 output rounding, one/eight-thread execution, and
failure atomicity for an invalid expert id. It passes on Apple AArch64 and
native AMD x86-64.

Three Apple one-thread passes used Q4 E4 R128 K1024 with N/I512, two warmups,
eight samples, and a 3 ms minimum:

| stage | grouped GEMM | grouped SwiGLU | decision |
|---|---:|---:|---|
| pass 1, independent packed row projection | 41.5329 ms | 83.2327 ms | semantic baseline; replace repeated decode |
| pass 2, 32-row expert-tile decode reuse | 34.3138 ms | 69.0251 ms | retain concept; 1.21x over pass 1 |
| pass 3, output-tile scheduling and direct paired gate/up epilogue | 33.7562 ms | 57.0444 ms | retain; 1.23x/1.46x over pass 1 and no gate/up tensor |

The final Apple 16-thread medians are 3.0648 and 4.2514 ms. At one thread the
packed paths are 0.95x/0.92x a same-run already-dequantized dense reference,
so no single-thread speedup over dense weights is claimed; at 16 threads the
same reference is serial and is only scaling context. Native EPYC records
48.9405/86.7021 ms at one thread and 3.2685/5.4621 ms at 16. Grouped GEMM is
1.12x over its predecoded comparator at one thread, SwiGLU is 0.82x, and the
threaded CV reaches 0.31; those mixed results remain explicit.

- References: `../QuixiCore-Metal/kernels/quantization/base_q/base_q.{h,cpp,
  metal}`, its correctness suite, and `base_q_moe` benchmark cases.
- Hardware/toolchains: Apple M5 Max, AArch64 NEON, Apple Clang 21.0.0; AMD
  EPYC 7702, x86-64 AVX2/FMA/F16C, GCC 13.3.0. CMake Release without LTO.
- Artifacts: `base-q-moe-pass1-rowwise-t1`,
  `base-q-moe-pass2-tile-reuse-t1`,
  `base-q-moe-pass3-output-tiles-t{1,16}`, and
  `x86-base-q-moe-final-t{1,16}` under `perf/results/2026-07-23/`.
- Decision: retain direct 32-row weight reuse, output-tile parallelism, and
  paired gate/up accumulation. Routing, gather, finalize, and shared-expert
  policy remain the existing public MoE operations rather than being folded
  into the BaseQN projection contract.

## 2026-07-23: Metal BaseQN LM-head argmax drift closure

Status: retain `base_q_lm_head_argmax` for BaseQ2/3/4/5/6/8, symmetric and
affine reconstruction, and FP32/FP16/BF16 activation storage. It performs one
greedy selection per activation row, rounds each score to the activation
storage type before comparison as Metal's GEMV composition does, and resolves
ties to the lower token id. The final portable path reduces four vocabulary
rows together and stores only bounded partial winners. On x86, measured
multi-row shapes retain the existing decoded-weight-reuse GEMM composition.

Correctness: `test_base_q` crosses all six bit widths, both reconstruction
modes, all three activation storage types, batch three, exact ties, invalid
counts, null output, and one/eight-thread execution. The expected result uses
an independently decoded dense projection and applies the same F16/BF16 score
rounding only at the public storage boundary. The focused test passes on Apple
AArch64 and AMD x86-64.

Three Apple one-thread passes used Q4 B1/B4 and Q6 B1 at V4096 K1024, two
warmups, eight samples, and a 3 ms minimum:

| stage | Q4 B1 | Q4 B4 | Q6 B1 | decision |
|---|---:|---:|---:|---|
| pass 1, materialized BaseQ GEMM plus scan | 2.6956 ms | 10.9441 ms | 2.6173 ms | semantic baseline; replace allocation |
| pass 2, scalar streaming row projection | 3.4268 ms | 13.6105 ms | 2.7718 ms | bounded memory but 0.78x-1.00x versus composition; replace |
| pass 3, four-vocabulary-row shared activation traversal | 1.5642 ms | 6.3969 ms | 1.8486 ms | retain; 1.40x-2.12x versus same-run composition |

The retained Apple 16-thread route records 0.3922/1.2195/0.3610 ms and is
1.46x-1.69x over its materialized comparator. On native EPYC, the final
one-thread route records 5.2209/17.7320/5.5134 ms; Q4 B1 and Q6 B1 are
1.36x/1.35x over composition, while selected B4 composition is neutral at
1.00x. Sixteen-thread results are 1.0444/2.1788/0.6839 ms and span
0.85x-1.39x; the regressive Q4 B1 cell is recorded and no blanket x86 speedup
is claimed.

- References: `../QuixiCore-Metal/kernels/quantization/base_q/base_q.metal`,
  its Python/Torch wrapper and tests, and the sibling BaseQ performance case.
- Hardware/toolchains: Apple M5 Max, AArch64 NEON, Apple Clang 21.0.0; AMD
  EPYC 7702, x86-64 AVX2/FMA/F16C, GCC 13.3.0. CMake Release without LTO.
- Artifacts: `base-q-lm-head-pass1-materialized-t1`,
  `base-q-lm-head-pass2-streaming-t1`,
  `base-q-lm-head-pass3-tiled-final-t1`,
  `base-q-lm-head-pass3-tiled-t16`, and
  `x86-base-q-lm-head-{pass3-final,selected}-t{1,16}` under
  `perf/results/2026-07-23/`.
- Decision: retain the four-row fused reduction on AArch64 and selected x86
  single-row cells; retain the storage-exact GEMM composition for measured
  x86 multi-row cells. The complete logits tensor is absent from the fused
  route, but the selected x86 multi-row route deliberately trades memory for
  decoded-weight reuse.

## 2026-07-23: M5 portable fallback and GGUF Q4 routing

Status: retain an explicit portable-only build mode and make `blocked_ref` a
force-only GGUF experiment. Automatic generic Q4_0 reference dispatch now
uses the specialized portable `q4_0_gemv_ref`; AVX2 and NEON selection are
unchanged. Cached dispatch state is held by pointer rather than local
reference in six dispatchers, eliminating GCC dangling-reference false
positives without changing lifetime or selection semantics.

Three native EPYC one-thread Q4_0 N4096 K4096 passes isolate the portable and
ISA routes:

| stage | median | decision |
|---|---:|---|
| pass 0, forced blocked portable experiment | 86.2722 ms | reject as automatic route |
| pass 1, generic GGUF element reference | 93.3694 ms | semantic baseline |
| pass 2, specialized portable Q4_0 reference | 11.7684 ms | retain; 7.9x over generic reference |
| pass 3, runtime-selected AVX2 | 3.5348 ms | retain; 3.33x over specialized portable reference |

`QUIXICORE_CPU_ENABLE_ISA_VARIANTS=OFF` now omits all optional ISA sources and
forces cached support results false, making a reused build directory safe for
fallback validation. Native x86 Release, portable-only Release, x86
ASan/UBSan/float-cast-overflow, and Apple Release suites all pass: Apple is
52/52 including live sibling parity; x86 native and portable-only Release are
46/46 excluding the unavailable sibling checkout; x86 sanitizer is 45/45 with
benchmarks disabled and leak detection enabled. The local parity audit maps
105 llama operations, 27 llama stored quant types, 76 sibling operations, 30
sibling quant families, and 85 full quant-matrix cells.

- Hardware/toolchains: AMD EPYC 7702, AVX2/FMA/F16C, GCC 13.3.0; Apple M5
  Max, AArch64 NEON, Apple Clang 21.0.0.
- Artifacts: `m5-x86-portable-pass0-blocked-t1`,
  `m5-x86-portable-pass1-generic-ref-t1`,
  `m5-x86-portable-pass2-q4-ref-t1`, and `m5-x86-avx2-pass3-t1` under
  `perf/results/2026-07-23/`.
- Decision: retain the explicit build switch, specialized portable Q4 route,
  and AVX2 selection. AVX-512/VNNI/AMX are not performance-claimed on this
  host because its runtime feature set does not expose them.

## 2026-07-23: M4 BitNet a4.8 KV3 codec and online attention (A9/A10)

Status: retain typed KV3 scatter/gather and direct online paged attention.
Each cache head is a least-significant-bit-first 3-bit stream. Runtime config
keeps the declared group size, signed/unsigned code interpretation,
none/integer zero-point mode, and FP16/FP32 scale storage explicit. Duplicate
scatter slots preserve deterministic token order; unique increasing slots may
run in parallel.

Correctness: `test_quant_cache_attention` independently unpacks every emitted
code and reconstructs each group. It covers signed symmetric FP16 scales,
unsigned nonnegative FP32 scales, unsigned affine FP32 scales, signed affine
FP16 scales, groups 16/32/64, D64/D128/D256, every FP32/FP16/BF16 codec
boundary, and all 3x3 query/output attention pairs. Direct attention is checked
against complete KV3 gather plus FP32 paged attention. The focused test passes
on Apple AArch64 and native AMD x86-64.

The primary three passes and retained follow-up used B=2, Hq=8, Hkv=2, S=512
at D64 signed/FP16/G32 and D128 unsigned-affine/FP32/G32, plus T=512, H=8,
D=128 BF16 codec I/O:

| stage | Apple result | decision |
|---|---:|---|
| pass 1, generic scalar bit positions and per-element metadata | attention 1.8663-1.9918 ms and 0.33x-0.54x materialization; I/O 2.2328-2.3087 ms at one thread | semantic baseline |
| pass 2, 8-code/3-byte addressing and group metadata hoisting | attention 0.8407-1.2383 ms; gather 1.4528 ms at one thread | retain; attention improves 1.51x-2.37x |
| pass 3, collision-safe cache/query scheduling | attention 0.1892-0.2638 ms and I/O 0.2422-0.3244 ms at 16 threads | retain; 1.37x-1.85x attention and 1.15x-1.16x I/O |
| follow-up, reuse each loaded 24-bit packet for eight codes | attention 0.4025-0.7002 ms at one thread and 0.1207-0.1583 ms at 16 | retain; one-thread attention improves another 1.77x-2.03x |

The retained Apple direct path is 0.87x-0.97x materialization at one thread,
so no single-thread speedup is claimed; at 16 threads it is 2.14x-3.02x.
Native EPYC is likewise regressive at one thread (0.46x-0.48x), while its
16-thread route is 1.74x-2.08x. Threaded BF16 scatter/gather is 1.14x/1.24x
on Apple and 1.04x/1.33x on EPYC. All regressive results remain recorded.

- Hardware/toolchains: Apple M5 Max, aarch64 NEON, macOS 26.5.2, Apple Clang
  21.0.0; AMD EPYC 7702, x86-64 AVX2/FMA/F16C, Ubuntu Linux
  6.8.0-134-generic, GCC 13.3.0. CMake Release without LTO.
- Artifacts: `m4-bitnet-kv3-pass{1-scalar-t1,2-packets-groups-t1,
  3-threaded-t1,3-threaded-t16,4-packet-reuse-t1}`,
  `m4-bitnet-kv3-final-t16`, and `x86-m4-bitnet-kv3-final-t{1,16}` under
  `perf/results/2026-07-23/`.
- Decision: retain packet reuse, group metadata hoisting, collision-safe
  scheduling, and every explicit metadata mode. This closes A9/A10 and M4.

## 2026-07-23: M4 canonical TurboQuant codec and online attention (A5-A8)

Status: retain typed TurboQuant encode/decode, a standalone normalized signed
FWHT query transform, and direct paged attention over the canonical packed
cache. Keys remain untransformed and use 2-8-bit codes plus FP16-rounded scale
and zero metadata per 32 values. Values are sign-rotated by an unnormalized
FWHT, normalized by sqrt(D), RMS-scaled per group, and centroid-coded at 2-8
bits. Attention accumulates centroid values in the rotated domain and performs
one inverse FWHT/sign transform after online-softmax normalization.

Correctness: `test_quant_cache_attention` covers every diagonal K/V bit width
from 2 through 8, signed K8, D64/D128/D256, an independent query-transform
oracle, FP32/FP16/BF16 encode and decode boundaries, every 3x3 query/output
attention pair, and direct attention against complete TurboQuant decode plus
FP32 paged attention. The focused test passes on Apple AArch64 and native AMD
x86-64.

Three passes used B=2, Hq=8, Hkv=2, S=512 with D64 K2V2 and D128 K4V4/K8V8,
plus T=512, H=8, D=128 K4V4 BF16 codec I/O:

| stage | Apple result | decision |
|---|---:|---|
| pass 1, generic bit-position decode and per-element metadata | attention 0.6831-1.4198 ms and 0.56x-0.61x materialization at one thread; codec 1.0408-12.4134 ms | semantic baseline |
| pass 2, fixed 2/4/8-bit extraction plus group scale/zero hoisting | attention 0.2659-0.6177 ms and 1.30x-1.53x at one thread | retain; 2.30x-2.57x faster than pass 1 |
| pass 3, fixed stack scratch and collision-safe token/head scheduling | attention 0.1048-0.1541 ms and codec 0.2737-1.1391 ms at 16 threads | retain; duplicate insertion slots use ordered execution |

The stable Apple one-thread route is 0.2696-0.6184 ms and 1.29x-1.54x over
explicit cache materialization; at 16 threads it is 3.00x-3.28x. Native EPYC
is 1.1851-2.3671 ms and 1.36x-2.12x at one thread, or 0.2005-0.3705 ms and
6.28x-7.99x at 16. Typed codec staging is intentionally reported rather than
generalized: encode is approximately neutral on both machines, Apple decode
is neutral, and EPYC threaded decode is 1.29x.

- Hardware/toolchains: Apple M5 Max, aarch64 NEON, macOS 26.5.2, Apple Clang
  21.0.0; AMD EPYC 7702, x86-64 AVX2/FMA/F16C, Ubuntu Linux
  6.8.0-134-generic, GCC 13.3.0. CMake Release without LTO.
- Artifacts: `m4-turboquant-pass{1-scalar-t1,2-fixedbits-groups-t1,
  3-stack-threaded-t1,3-stack-threaded-t16}` and
  `x86-m4-turboquant-final-t{1,16}` under `perf/results/2026-07-23/`.
- Decision: retain specialized extractors, group metadata hoisting, bounded
  stack scratch, and collision-safe scheduling. The standalone A5 transform
  is not applied to canonical untransformed K. This closes A5-A8 only.

## 2026-07-23: M4 canonical MXFP8 cache and online attention (A3/A4)

Status: retain canonical MXFP8 cache scatter/gather and direct paged
attention. Each head row is D/32 consecutive 33-byte blocks with the E8M0
scale byte first and 32 E4M3FN codes after it. Dynamic scaling rounds upward
to the smallest E8M0 power of two that prevents finite overflow; zero blocks
emit an all-zero block. FP32, FP16, and BF16 storage share the same cache.

Correctness: `test_quant_cache_attention` crosses D64/D128, every typed source,
every typed gather, and every query/output type pair. It checks each embedded
scale/code block independently and compares direct online attention with a
fully decoded FP32 cache. Reserved E8M0 code 255 and cache geometry are
validated. The focused test passes on Apple AArch64 and native AMD x86-64.

Three passes used B=2, Hq=8, Hkv=2, S=512, D=64/128 for attention and T=512,
H=8, D=128 BF16 for cache I/O:

| stage | Apple result | decision |
|---|---:|---|
| pass 1, scalar E4M3/E8M0 decode | attention 1.8178-3.5415 ms; I/O 2.2599-3.5017 ms at one thread | semantic baseline |
| pass 2, decode tables and per-group scale hoisting | attention 0.3285-0.6202 ms; I/O 0.9615-3.4344 ms at one thread | retain; attention improves 5.53x-5.71x |
| pass 3, collision-safe token/head block scheduling | attention 0.0956-0.1419 ms; I/O 0.1705-0.4138 ms at 16 threads | retain; duplicate scatter slots fall back to ordered execution |

The retained Apple one-thread attention route is 1.95x-2.10x over explicit
cache materialization and its 16-thread route is 8.32x-15.57x. Native EPYC is
0.8052-1.5839 ms and 1.53x-1.57x at one thread, or 0.2508-0.3109 ms and
4.71x-7.13x at 16. EPYC BF16 scatter/gather is 12.9934/2.7385 ms at one
thread and 1.5859/0.2040 ms at 16; threaded gather is 2.14x over staging while
scatter remains 0.55x. No blanket cache-I/O speedup is claimed.

- Hardware/toolchains: Apple M5 Max, aarch64 NEON, macOS 26.5.2, Apple Clang
  21.0.0; AMD EPYC 7702, x86-64 AVX2/FMA/F16C, Ubuntu Linux
  6.8.0-134-generic, GCC 13.3.0. CMake Release without LTO.
- Artifacts: `m4-mxfp8-pass{1-scalar-t1,2-lut-hoist-t1,
  3-threaded-t16}` and `x86-m4-mxfp8-final-t{1,16}` under
  `perf/results/2026-07-23/`.
- Decision: retain canonical interleaved blocks, lookup decoding, scale
  hoisting, and collision-safe scheduling. This closes A3/A4 only.

## 2026-07-23: M4 typed FP8 cache and direct online paged attention (A1/A2)

Status: retain typed FP8 cache scale/scatter/gather APIs and direct E4M3FN or
E5M2 paged attention. Cache bytes remain canonical and scales are explicit
per KV head. Dynamic scaling uses the umbrella maxima 448/57344 and represents
an all-zero head with scale zero. FP32, FP16, and BF16 sources, queries,
gathers, and outputs share the same cache representation; attention never
materializes a complete FP32 cache.

Correctness: `test_quant_cache_attention` crosses both FP8 formats, D64/D128,
all three source types, all three gather types, and all 3x3 query/output type
pairs. Its independent oracle decodes canonical bytes before FP32 attention
and checks the final storage rounding boundary. It also covers per-head
dynamic scales, zero-scale heads, static-scale validation, and exact scatter /
gather codes. The focused test passes on Apple AArch64 and native AMD x86-64.

Direct attention used B=2, Hq=8, Hkv=2, S=512, D=64/128:

| stage | Apple one-thread result | decision |
|---|---:|---|
| pass 1, scalar FP8 decode in every K/V element | 2.0064-4.1616 ms, 0.29x-0.31x | semantic baseline; diagnose decoder and scale multiplication |
| pass 2, 256-entry decode tables plus head-scale hoisting | 0.3949-0.8290 ms | retain; removes repeated bit decode and V scaling |
| pass 3, online score tile 32 | 0.4036-0.8354 ms | reject; retain tile 16 |

The stable Apple one-thread confirmation is 0.3340-0.7449 ms and
1.22x-1.78x over explicit FP32 cache materialization. At 16 threads it is
0.0930-0.1528 ms and 6.47x-8.35x. Native EPYC is 0.9156-1.8530 ms and
3.83x-5.09x at one thread, or 0.1788-0.3575 ms and 13.12x-28.99x at 16.

Typed scatter/gather used T=512, H=8, D=128:

| stage | Apple one-thread result | decision |
|---|---:|---|
| pass 1, per-element generic storage branch and scalar decoder | 2.0064-3.6768 ms | semantic baseline |
| pass 2, storage-specialized loops and gather decode table | 0.6263-3.4701 ms | retain; gather improves 1.94x-3.20x |
| pass 3, collision-safe token/head threading | 0.1745-0.4668 ms at 16 threads | retain for unique ordered scatter slots and every gather |

Native EPYC typed I/O is 2.8620-11.9807 ms at one thread and
0.3684-1.6780 ms at 16. Staged F16C/AVX2 conversion remains faster for some
x86 typed I/O cases, and the Apple F16 staged comparator also wins at 16
threads. The direct routes are retained for bounded memory and exact cache
semantics; no blanket typed-I/O speedup is claimed.

- Hardware/toolchains: Apple M5 Max, aarch64 NEON, macOS 26.5.2, Apple Clang
  21.0.0; AMD EPYC 7702, x86-64 AVX2/FMA/F16C, Ubuntu Linux
  6.8.0-134-generic, GCC 13.3.0. CMake Release without LTO.
- Artifacts: `m4-fp8-attention-pass{1-scalar,2-lut-scale-hoist,
  3-score-tile32}-t1`, `m4-fp8-cache-pass{1-generic,
  2-specialized-lut}-t1`, `m4-fp8-cache-attention-pass3-threaded-t16`, and
  `x86-m4-fp8-cache-attention-final-t{1,16}` under
  `perf/results/2026-07-23/`.
- Decision: retain lookup decoding, scale hoisting, typed specialization, and
  collision-safe threading; reject score tile 32. This closes A1/A2 only.

## 2026-07-23: M3 canonical MoE dual activation and next-layer quantization

Status: retain `moe_grouped_prepacked_quantized` and
`moe_grouped_prepacked_swiglu_quantized`, completing the applicable S5/S6
matrix. The first accepts every canonical projection packet and every one of
the 11 prepared expert-weight layouts; measured direct block-pair kernels are
used on Apple, while multi-row x86 calls decode the activation once into
reusable workspace before entering the mature grouped panel path. The second
projects gate/up and writes canonical symmetric/affine INT4 or INT8, FP4, FP8
E4M3/E5M2, MXFP4, MXFP8, or NVFP4 output directly from bounded worker/domain
scratch. Tensor scale measurement and NVFP4 1-D/2-D domains preserve the
canonical packet contract. Sorted FP8 expert runs reuse multi-row decoded
panels in one global task schedule.

Correctness: `test_quant_serving_matrix` covers representative dual pairs for
all 11 weight layouts, including W4A8, affine W8A8, mixed E5M2/E4M3, MX,
NVFP4, and BitNet-A8. It independently dequantizes both operands for the dot
oracle. Direct SwiGLU packing crosses every weight layout with FP32/FP16/BF16
input and crosses all ten supported output packet layouts, tensor/group
scales, a partial 17-row NVFP4-2D domain, padded rows, and a sorted four-thread
panel route. The focused test passes on Apple AArch64 and native AMD x86-64.

Dual-activation optimization used E=8, M=128, N=512, K=1024 with W4A8,
E4M3/E4M3, MXFP8/MXFP8, MXFP4/MXFP4, and NVFP4/NVFP4:

| stage | Apple one-thread result | decision |
|---|---:|---|
| pass 1, generic packet decode inside grouped panels | 45.3848-84.2965 ms | semantic baseline; reject as the final fast route |
| pass 2, direct paired block dots and W4A8 integer dot | 3.6259-25.3727 ms | retain by supported pair; compact pairs become 1.54x-2.03x over per-token projection |
| pass 3, FP8 output-panel group sweep | group-16 FP8 26.6631 ms | reject; retain group 8 and the architecture/format selector |

Against dequantize-once plus grouped projection, the final Apple one-thread
range is 2.7345-24.8965 ms and 0.48x-2.12x; at 16 threads it is
0.3745-2.2389 ms and 0.63x-3.57x. FP8/MXFP8 direct dots remain slower on this
shape and are reported without a speedup claim. Native EPYC rejected direct
packet dots at 104.3126-382.5582 ms (0.09x-0.29x); the retained decode-once
selector is 29.8873-38.1184 ms and approximately 1.00x at one thread, and
4.6464-6.8273 ms at 16 threads with 0.85x-1.07x ratios.

Direct SwiGLU-output quantization used the same expert shape and INT4, FP8
E4M3, MXFP4, and NVFP4 weights, writing group-A8:

| stage | Apple one-thread result | decision |
|---|---:|---|
| pass 1, output-group tasks | 11.6000-46.9032 ms, 0.44x-1.05x | retain semantics; diagnose repeated FP8 activation traversal |
| pass 2, one full output row per task | 11.6150-46.3776 ms, 0.44x-1.06x | retain as the single-worker non-panel route; locality alone is insufficient for FP8 |
| pass 3, sorted expert multi-row panels | 11.4797-23.5455 ms, 0.97x-1.11x | retain FP8-family panel selection; FP8 falls from 46.9 to 19.5 ms |

The final Apple 16-thread range is 1.2036-2.3814 ms and 1.00x-1.83x versus
materialized SwiGLU plus standalone quantization. Native EPYC one-thread is
50.5101-63.4441 ms and 1.02x-1.06x for the final measured run. Its direct
16-thread route is 4.6062-6.9157 ms and 0.57x-1.08x; a pass-3 materializing
x86 experiment regressed INT4/MXFP4 further and was rejected. Thus the direct
route is retained for bounded memory and Apple throughput, with no blanket
x86 threaded speedup claim.

- Hardware/toolchains: Apple M5 Max, aarch64 NEON, macOS 26.5.2, Apple Clang
  21.0.0; AMD EPYC 7702, x86-64 AVX2/FMA/F16C, Ubuntu Linux
  6.8.0-134-generic, GCC 13.3.0. CMake Release without LTO.
- Runtime: one and 16 threads, OS-default affinity/frequency, hot prepared
  weights and reusable workspace/output capacity.
- Artifacts: `s5-moe-dual-pass{1-generic,2-direct-pairs-final,
  3-fp8-group16}-t1`, `s6-moe-swiglu-quant-pass{1-groups,2-full-rows,
  3-expert-panels}-t1`, `s5-s6-moe-final-t16`,
  `x86-s5-s6-moe-final-t1`, `x86-s5-moe-dual-pass3-adaptive-t1`,
  `x86-s5-s6-moe-adaptive-final-t16`, and
  `x86-s5-s6-moe-final2-t16` under `perf/results/2026-07-23/`.
- Decision: retain direct paired dots, sorted FP8 panel reuse, and the x86
  decode-once selector. The materializing threaded-output experiment and FP8
  panel-group-16 experiment remain as rejected evidence. Together with the
  stored-weight entry below, S5-S7 are complete for the canonical matrix and
  existing named sibling adapters.

## 2026-07-23: M3 canonical grouped MoE weight-only path (S5/S6/S7 partial)

Status: retain `moe_grouped_prepacked_storage` and
`moe_grouped_prepacked_swiglu_storage` for all 11 prepared canonical weight
layouts with FP32/FP16/BF16 activation and output storage. Sorted expert IDs
operate directly on contiguous expert runs. Unsorted IDs use a stable
workspace-backed grouping, typed gather, one projection per nonempty expert,
and typed scatter. `-1` padded rows store zero. Fused SwiGLU stores
`SiLU(gate)*up` directly and never allocates a rows-by-2I tensor. Named FP8,
WNA16, and NVFP4 entry points validate metadata and reuse the shared kernel.

Correctness: `test_quant_serving_matrix` compares grouped projection and fused
SwiGLU with independently dequantized expert weights for every layout and all
three storage types. It covers sorted and unsorted routes, per-expert bias and
activation, padded and all-padded inputs, typed rounding, named-wrapper format
checks, and failure-atomic invalid expert IDs. The focused test passes on Apple
AArch64 and native AMD x86-64.

The CPU translation uses expert-run grouping and row blocking from
`llama.cpp@2beefef68825:ggml/src/ggml-cpu/ggml-cpu.c`, padded/sorted expert
tiles and in-register quant decode from
`QuixiCore-Metal@bc968fc3215e:kernels/moe/moe/moe.metal`, and the
expert-block metadata approach in
`vllm@4b594b4aa1ed:csrc/libtorch_stable/moe/marlin_moe_wna16/`. CPU prepared
row panels, cache tiles, and the persistent thread pool replace GPU warps and
threadgroups.

Three optimization passes used E=8, M=128, N=512, K=1024 and INT4, FP8 E4M3,
MXFP4, and NVFP4 representatives against one prepared GEMV per token:

| stage | Apple result | decision |
|---|---:|---|
| pass 1, stable grouping plus direct sorted runs | 12.9025-26.4748 ms at one thread, 0.45x-2.47x | retain semantics and FP8 reuse; diagnose compact-format M16 expert groups |
| pass 2, architecture/format-adaptive token rows | 5.6688-22.7584 ms at one thread, 0.75x-2.26x | retain compact Apple row kernels; retain grouped FP8 |
| pass 3, one shared threaded dispatch across sorted experts | 1.1402-2.3186 ms at 16 threads, 3.18x-8.72x | retain; removes eight nested expert scheduling barriers |

The final Apple one-thread run is 5.7645-23.1802 ms and 0.91x-2.35x; compact
formats are intentionally treated as throughput-neutral while FP8 grouping is
2.01x-2.35x. Native EPYC testing rejected the Apple compact token selector:
its rejected route reached 410.5366 ms. The retained x86 grouped route is
29.5026-67.4725 ms and 3.68x-6.49x at one thread. At 16 threads it is
2.1855-4.5282 ms and 3.56x-8.30x. All benchmark checks pass.

- Hardware/toolchains: Apple M5 Max, aarch64 NEON, macOS 26.5.2, Apple Clang
  21.0.0; AMD EPYC 7702, x86-64 AVX2/FMA/F16C, Ubuntu Linux
  6.8.0-134-generic, GCC 13.3.0. CMake Release without LTO.
- Runtime: one and 16 threads, OS-default affinity/frequency, hot prepared
  weights and reusable workspace storage.
- Artifacts: `s5-s7-moe-pass1-grouped-t1`,
  `s5-s7-moe-pass2-adaptive-t{1,16}`,
  `s5-s7-moe-pass3-{final-t1,shared-dispatch-t16}`,
  `x86-s5-s7-moe-final-t{1,16}`, and
  `x86-s5-s7-moe-grouped-final-t1` under `perf/results/2026-07-23/`.
- Decision: retain all three passes and the architecture-specific selector.
  This closes the stored-weight/A16 portions of S5/S6 and the corresponding
  S7 adapters. Dual quantized activations, optional next-layer quantization,
  and their named fused routes remain open, so the aggregate MoE checklist is
  deliberately not marked complete.

## 2026-07-23: M3 canonical streaming LM head (S3/S4)

Status: retain the four prepared canonical LM-head APIs for sampling,
packed-mask top-k, CSR-like candidate selection, and beam advance. All 11
canonical stored-weight layouts and FP32/FP16/BF16 hidden storage use direct
prepared weights. Argmax/top-k/masked/candidate/beam keep bounded selection
state plus online log-sum-exp state. Exact categorical/top-p keeps one
vocabulary row because the nucleus is data-dependent; no route allocates a
rows-by-vocabulary logits tensor.

Correctness: `test_quant_serving_matrix` compares argmax, seeded top-k, seeded
top-p, masked log probabilities, candidate log probabilities, and beam
tokens/parents/scores against independently dequantized composition for every
layout. It covers all three hidden storage types, deterministic lowest-token
tie rules inherited from the selectors, invalid candidate rejection, and
failure-atomic outputs. The focused test passes on Apple AArch64 and native AMD
x86-64. The CPU tile/partial algorithm was adapted from
`QuixiCore-Metal@bc968fc:kernels/quantization/lm_head/lm_head.metal`.

Three primary passes used R=1/4/8, V=4096, H=1024, K=4, C=128 and INT4, FP8
E4M3, MXFP4, and BitNet representatives against prepared projection followed
by materialized selection:

| stage | Apple one-thread range ms | decision |
|---|---:|---|
| pass 1, direct row-dot streaming | 0.1289-11.9684 | retain for formats with mature direct row dots; exposed multi-row FP8 decode repetition |
| pass 2, multi-row prepared panels | 0.1220-7.6777 | retain for FP8; reject globally because INT4/MXFP4/BitNet row dots are faster on Apple |
| pass 3, format-selected traversal | 0.1222-7.5998 | retain; top-k is 1.14x-2.96x and structured routes 3.37x-130.40x |

Threaded follow-ups split vocabulary panels into independent partials, merge
with deterministic tie rules, and on x86 reuse each decoded panel across all
hidden rows in the tile. Final Apple 16-thread time is 0.1141-0.5263 ms with
ratios 0.84x-17.03x. Native EPYC one-thread time is 2.4480-23.1759 ms with
ratios 0.70x-12.49x; final 16-thread time is 0.8428-2.3653 ms with ratios
0.39x-3.85x. On EPYC, multi-row INT4/FP8/MXFP4/BitNet top-k ratios are
2.37x/2.04x/3.85x/2.46x. Single-row selectors and exact top-p are kept for
bounded memory and semantics even where materialized projection is faster, so
no blanket speedup is claimed.

- Hardware/toolchains: Apple M5 Max, aarch64 NEON, macOS 26.5.2, Apple Clang
  21.0.0; AMD EPYC 7702, x86-64 AVX2/FMA/F16C, Ubuntu Linux
  6.8.0-134-generic, GCC 13.3.0. CMake Release without LTO.
- Runtime: one and 16 threads, OS-default affinity/frequency, hot prepared
  weights and reusable output/workspace storage.
- Artifacts: `s3-s4-lm-head-pass{1-row-dot,2-mrow-panels,
  3-format-select}-t1`, `s3-s4-lm-head-final3-t16`,
  `x86-s3-s4-lm-head-panel-final-t1`, and
  `x86-s3-s4-lm-head-mrow-final-t16` under `perf/results/2026-07-23/`.
- Decision: retain architecture-selected row, panel, and partial routes. S3
  and S4 are complete; the remaining MoE cells are scoped in the entry above.

## 2026-07-23: M3 canonical embedding and embedding bag (S1/S2)

Status: retain `canonical_quantized_embedding_storage` and
`canonical_quantized_embedding_bag_storage`. Both decode selected rows
directly from all 11 canonical stored-weight layouts without materializing a
dequantized vocabulary table. Gather supports scale, optional independently
typed add input, and FP32/FP16/BF16 output. Bag supports sum, mean, and
weighted reduction with bounded per-worker scratch.

Correctness: `test_quant_serving_matrix` crosses every canonical layout,
FP32/FP16/BF16 gather output and add storage, and weighted mean bag reduction.
It compares with an independently dequantized full-table oracle, checks typed
rounding boundaries, and proves invalid IDs leave outputs unchanged. The
focused test passes on Apple AArch64 and native AMD x86-64.

Three optimization passes used V=4096, C=128, D=1024 and representative INT4,
FP8 E4M3, MXFP4, and BitNet tables. The same-binary comparator dequantizes the
complete table and then performs the requested gather or weighted bag:

| stage | Apple one-thread range ms | decision |
|---|---:|---|
| pass 1, scalar selected-row decode | recorded in artifact | correctness and direct-decode baseline |
| pass 2, direct FP32 store | recorded in artifact | retain for no-add FP32 gather |
| pass 3, item-major bag accumulation | 0.2765-0.7889 | retain; decodes each selected row once per bag |

Final Apple one-thread ratios span 2.57x-25.69x; at 16 threads the target is
0.1025-0.5095 ms with 3.86x-83.77x ratios. Native EPYC one-thread results are
1.1944-3.3880 ms and 1.84x-31.09x; 16-thread results are 0.2583-1.7255 ms and
3.65x-144.90x. All checks pass. The large ratios reflect removal of complete
table dequantization, so they are intentionally not described as equal-work
decode throughput improvements.

- Hardware/toolchains: Apple M5 Max, aarch64 NEON, macOS 26.5.2, Apple Clang
  21.0.0; AMD EPYC 7702, x86-64 AVX2/FMA/F16C, Ubuntu Linux
  6.8.0-134-generic, GCC 13.3.0. CMake Release without LTO.
- Runtime: one and 16 threads, OS-default affinity/frequency, hot inputs and
  reusable output/workspace storage.
- Artifacts: `s1-s2-embedding-pass{1-scalar,2-f32-store,3-row-accum}-t1`,
  `s1-s2-embedding-final-t16`, and `x86-s1-s2-embedding-final-t{1,16}` under
  `perf/results/2026-07-23/`.
- Decision: retain pass 3. S1 and S2 are complete; later serving cells are
  tracked in the entries above.

## 2026-07-23: M3 canonical RMSNorm/LayerNorm add-quant (F6/F7)

Status: retain `rms_norm_add_quantized_storage` and
`layer_norm_add_quantized_storage`. Each API accepts independently typed
FP32/FP16/BF16 input, residual, parameter, and residual-output storage and
directly emits a validated canonical symmetric/affine INT4 or INT8,
FP4, FP8 E4M3/E5M2, MXFP4, MXFP8, or NVFP4 packet. Row statistics and bounded
worker scratch replace the complete normalized tensor. Tensor-scaled FP and
NVFP4 use a required maximum-measurement traversal; NVFP4-2D shares each local
scale across at most 16 rows as specified by the umbrella contract.

Correctness: `test_norm_quant_matrix` crosses RMSNorm and LayerNorm, all ten
non-BitNet activation layouts plus both NVFP4 scale topologies, 17-row partial
NV domains, homogeneous FP32/FP16/BF16 storage, and a mixed
F16-input/BF16-residual/F32-parameter/BF16-output case. Residual stores are
checked at their exact typed rounding boundary. Canonical packets are
validated and compared after decode with an independently calculated
normalization followed by the standalone canonical packer. Invalid shapes,
unsupported BitNet output, and invalid 2-D scale requests are rejected. The
focused test passes on Apple AArch64 and native AMD x86-64.

Three optimization passes used R=128, H=2048 and representative group INT4,
group INT8, FP8, FP4, MXFP8, MXFP4, and NVFP4-2D outputs, with both norm types
and a same-binary preallocated norm-then-canonical-quant comparator:

| stage | Apple one-thread range ms | decision |
|---|---:|---|
| pass 1, direct bounded-group packing | 0.4351-3.1339 | correctness baseline; runtime storage access repeats in each group |
| pass 2, direct FP32 pointer traversal | 0.3392-3.5709 | retain; removes per-element storage dispatch on the measured path |
| pass 3, full-row normalization reuse | 0.3185-2.3169 | retain; removes repeated row/group setup and improves small MX blocks |

The retained Apple one-thread ratios span 0.93x-1.51x; RMS INT4/INT8 are
1.51x/1.36x and LayerNorm INT4/INT8 are 1.11x/1.07x. At 16 threads all 13
cases are faster than composition: 0.1360-0.4705 ms and 1.41x-5.71x, with CV
up to 0.2062. Native EPYC one-thread results are 1.2519-5.3854 ms and
0.85x-1.19x; the 16-thread results are 0.2895-0.9912 ms and 1.77x-6.52x, with
CV at most 0.0952. The retained value is therefore canonical coverage and
removal of the full normalized allocation, with strong threaded scaling but
no blanket one-thread speedup claim.

- Hardware/toolchains: Apple M5 Max, aarch64 NEON, macOS 26.5.2, Apple Clang
  21.0.0; AMD EPYC 7702, x86-64 AVX2/FMA/F16C, Ubuntu Linux
  6.8.0-134-generic, GCC 13.3.0. CMake Release without LTO.
- Runtime: one and 16 threads, OS-default affinity/frequency, hot inputs and
  reusable canonical output capacity.
- Working-tree label: `397f16a-dirty`; rsynced x86 metadata is `unknown`
  because `.git` is excluded.
- Artifacts: `f6-f7-norm-pass{1-direct-pack,2-f32-direct,
  3-full-row}-t1`, `f6-f7-norm-final-t16`, and
  `x86-f6-f7-norm-final-t{1,16}` under `perf/results/2026-07-23/`.
- Decision: retain pass 3. F6 and F7 are complete for canonical A4/A8/FP8/MX
  and NV activation output; BitNet Q9 sparse activation preparation remains a
  separate Phase-8 item.

## 2026-07-23: M3 canonical QKV plus RoPE plus typed KV write (F5)

Status: retain `qgemv_prepacked_qkv_rope_kv_storage` for all 11 non-cache
canonical prepared-weight layouts. The single-token API accepts direct
FP32/FP16/BF16 activation storage, applies split-half/GPT-NeoX RoPE to Q and K,
writes Q in FP32/FP16/BF16, and inserts K/V directly into independently typed
flattened `[slots,Hkv,D]` caches. `slot=-1` skips K/V projection and leaves both
caches untouched. No separate raw Q, K, or V staging allocation is made.
Compressed FP8/MX/TurboQuant caches remain M4 and are not claimed here.

Correctness: `test_quant_projection_matrix` crosses every prepared layout with
Hq=2, Hkv=1, D=4, K=32, all three input storage types, FP32 outputs, and mixed
Q-FP16/K-BF16/V-FP16 output storage. Projection values come from three
independently dequantized FP64 oracles before an independent RoPE/cache oracle;
typed stores are checked at the exact final rounding boundary. Non-target cache
slots and the `slot=-1` path are checked unchanged. The focused test passes on
Apple AArch64 and native AMD x86-64.

Optimization used Hq=8, Hkv=2, D=64, K=1024 with representative INT4, FP8
E4M3, MXFP4, and BitNet weights and a same-binary comparator consisting of F4
QKV projection followed by explicit RoPE and cache writes:

| stage | representative one-thread range ms | decision |
|---|---:|---|
| pass 1, direct row pairs | 0.0825-0.3328 | correctness baseline; typed conversion repeats |
| pass 2, one typed conversion per RoPE pair | 0.0799-0.3132 | retain as reordered-weight fallback |
| pass 3, generic paired panels | 0.3351-0.5857 | reject; generic decode loses mature row dots |
| follow-up group-4 generic panels | 0.0793-0.5894 | reject for typed Apple routes |
| follow-up NEON typed pair panels | 0.0765-0.2896 | retain for direct typed output |
| follow-up FP32 pair tiles | 0.0737-0.2898 | retain where locality is stable |

Pass 2 reduces Apple typed INT4 from 0.1459/0.1487 ms to
0.1052/0.1095 ms for FP16/BF16 input. The NEON row-panel follow-up reaches
0.0765/0.0776 ms by sharing one converted block across four RoPE pairs. The
generic decoded-panel experiments were rejected rather than hidden. The final
Apple one-thread range is 0.0778-0.2951 ms with 0.955x-1.028x ratios versus the
preallocated composition. At 16 threads it is 0.0572-0.1083 ms with
0.940x-1.005x ratios; CV reaches 0.279 on one short typed case, so this is a
materialization/semantic closure rather than a blanket throughput claim.

Native x86 follow-ups retain generic typed panel groups, use direct FP32 paired
panels for one-thread conventional FP8, and otherwise project raw values
directly into the final Q/cache destinations before in-place RoPE. This last
route preserves mature plane locality without allocating raw K/V tensors. The
stable one-thread final spans 1.1842-2.1662 ms: FP32 is 1.000x-1.105x and typed
input is 1.095x-1.148x versus composition, with CV 0.0009-0.0340. The
16-thread range is 0.1746-0.3345 ms with 0.931x-1.140x ratios and CV at most
0.0872; no blanket multithreaded speedup is claimed.

- Hardware/toolchains: Apple M5 Max, aarch64 NEON, macOS 26.5.2, Apple Clang
  21.0.0; AMD EPYC 7702, x86-64 AVX2/FMA/F16C, Ubuntu Linux
  6.8.0-134-generic, GCC 13.3.0. CMake Release without LTO.
- Runtime: one and 16 threads, OS-default affinity/frequency, prepared hot
  Q/K/V weights, hot activations, position 2 and slot 1 of four.
- Working-tree label: `397f16a-dirty`; rsynced x86 metadata is `unknown`
  because `.git` is excluded.
- Artifacts: `f5-qkv-rope-pass{1-direct-rows,2-pair-convert,
  3-paired-panels,4-panel-group4,5-neon-pair-panels,
  6-f32-pair-tiles}-t1`, `f5-qkv-rope-final2-t1`,
  `f5-qkv-rope-final3-t16`, `x86-f5-qkv-rope-pass{1-selected,
  2-f32-locality,3-inplace-final}-t1`, and
  `x86-f5-qkv-rope-final2-t16` under `perf/results/2026-07-23/`.
- Decision: retain the architecture/storage/thread-selected routes. F5 is
  complete for ordinary floating cache storage; compressed cache execution is
  still governed by the open M4 checklist.

## 2026-07-23: M3 canonical unequal-head Q/K/V projection (F4)

Status: retain `qgemm_prepacked_qkv_storage` and its M=1 wrapper for all 11
non-cache canonical prepared-weight layouts. Q, K, and V remain independently
prepared and may have different output-row counts and FP32/FP16/BF16 output
storage types, but execute under one CPU scheduling region. The selected
multi-row routes either share a decoded activation tile across simultaneously
active Q/K/V panels or reuse the mature architecture-specific typed panel
kernels. No combined weight or projection-output intermediate is allocated.

Correctness: `test_quant_projection_matrix` crosses every prepared layout with
unequal Nq=7, Nk=3, Nv=5, M=1/16/128, and FP32/FP16/BF16 inputs. FP32 outputs
are checked against three independently dequantized FP64 projection oracles;
mixed direct Q-FP16/K-BF16/V-FP16 stores are checked at their exact rounding
boundary, and the M=1 wrapper is checked separately. Incompatible Q/K/V packs
and unknown storage enums fail before modifying any output. The focused test
passes on Apple AArch64 and native AMD x86-64.

The primary Apple FP32 sweep used representative INT4, FP8 E4M3, MXFP4, and
BitNet weights at M=1/16/128, Nq/Nk/Nv=512/128/128, K=1024:

| stage | M1 range ms | M16 range ms | M128 range ms | decision |
|---|---:|---:|---:|---|
| pass 1, one schedule with mature per-plane traversal | 0.0732-0.2887 | 2.1360-2.8779 | 16.8872-18.6583 | correctness/scheduling baseline |
| pass 2, shared activation tiles | 0.0780-0.3056 | 2.2521-2.5564 | 16.5824-18.4100 | retain for large Apple prefill |
| pass 3, architecture/shape/thread selection and balanced stripes | 0.0726-0.2875 | 2.1784-2.4611 | 15.6088-16.8805 | retain selected routes |

Pass 2 made all four Apple M128 cases 1.01x-1.08x faster than three prepared
calls in its paired run, but FP8 M16 regressed; pass 3 therefore restores the
mature M16 route. A subsequent typed-storage sweep found that shared
FP16/INT4 and BF16 traversal could regress, so Apple typed QKV stays on the
measured M2 typed panel kernels. In the retained Apple run, FP32 M128 is
1.04x-1.08x over three prepared calls; the complete one-thread FP32/FP16/BF16
ratio range is 0.929x-1.080x, so no blanket one-thread speedup is claimed.
Sixteen-thread medians are 0.0496-1.8172 ms for FP32 and
0.0497-1.6843 ms for typed cases; ratios are 1.24x-2.91x, with CV up to 0.239
on the unstable BitNet/typed tails.

Native x86 pass 2 showed stable one-thread shared-traversal gains for M16 and
M128, but the typed follow-up exposed BF16 regressions and the longer
multithreaded run exposed large-M FP32 imbalance. The final selector shares
single-thread large-M FP32, small-M non-FP8 FP32, and non-FP8 FP16, while BF16
and multithreaded large-M FP32 use one schedule over mature panel groups. The
retained EPYC one-thread run spans 1.3112-38.9474 ms: M1 is neutral,
representative M16 ratios are 0.997x-1.142x, and M128 ratios are
0.999x-1.138x. The final 16-thread run is 0.1864-4.1566 ms and all paired
ratios are at least 1.011x, though typed CV reaches 0.268 and is scaling
context rather than a performance-tier claim.

- Hardware/toolchains: Apple M5 Max, aarch64 NEON, macOS 26.5.2, Apple Clang
  21.0.0; AMD EPYC 7702, x86-64 AVX2/FMA/F16C, Ubuntu Linux
  6.8.0-134-generic, GCC 13.3.0. CMake Release without LTO.
- Runtime: one and 16 threads, OS-default affinity/frequency, independently
  prepared hot Q/K/V weights and hot activations. Comparator is three public
  `qgemm_prepacked_storage` calls from the same binary.
- Working-tree label: `397f16a-dirty`; rsynced x86 metadata is `unknown`
  because `.git` is excluded.
- Artifacts: `f4-qkv-pass1-single-schedule-t1`,
  `f4-qkv-pass2-shared-activation-t1`,
  `f4-qkv-pass3-route-balance-t{1,16}`, `f4-qkv-final4-t{1,16}`,
  `x86-f4-qkv-pass2-shared-activation-t1`,
  `x86-f4-qkv-final2-t1`, and `x86-f4-qkv-final3-t16` under
  `perf/results/2026-07-23/`.
- Decision: retain the evidence-selected F4 routes. Canonical QKV projection
  is complete; RoPE and typed KV-cache insertion remain F5 and are not implied
  by this entry.

## 2026-07-23: M3 canonical fused SwiGLU and activation quantization (F3)

Status: retain `qgemm_prepacked_swiglu_storage` and
`qgemm_prepacked_swiglu_quantized`, including their M=1 wrappers. The first
stores FP32/FP16/BF16 without gate/up tensors. The second writes canonical
INT4/U4, symmetric/asymmetric INT8, FP4, FP8 E4M3/E5M2, MXFP4, MXFP8, or
NVFP4 bytes and scale metadata directly from worker-local output groups. It
does not materialize an M-by-N floating-point SwiGLU tensor. Output vector
capacity and the caller/thread workspace are reusable after warmup.

Correctness: `test_quant_projection_matrix` crosses all 11 non-cache prepared
weight layouts with M=1/16/128 and FP32/FP16/BF16 inputs. Direct typed SwiGLU
stores are checked before rounding. Quantized output covers symmetric and
affine A4/A8, row/group/tensor FP/INT scale modes, MX K32, and NVFP4 1-D/2-D
scale domains against the independently composed prepared-SwiGLU plus
canonical quantizer. The focused test passes on Apple AArch64 and AMD x86-64.
The checked benchmark uses group-A8 output and a same-binary fused-SwiGLU plus
standalone canonical quantizer as the stronger comparator.

The typed-store kernel received three Apple passes: (1) direct fused SiLU
store, (2) compile-time fused/unfused specialization, and (3) direct FP32
pointer storage after the activation. Its retained one-thread medians are
0.0995-0.3835 ms at M1, 2.4557-3.0861 ms at M16, and
18.2240-21.0936 ms at M128. Ratios versus paired projection plus a separate
SwiGLU pass are 0.969x-1.112x. The 16-thread range is
0.0533-2.2383 ms. Native x86 follow-ups phase gate/up planes and tile 16 output
rows; a forced INT4 panel experiment was rejected. Retained EPYC one-thread
medians are 1.5559-52.3149 ms with 0.890x-1.043x ratios and CV
0.0003-0.0168, so the x86 claim is allocation removal rather than a blanket
speedup.

The activation-quantizing kernel received three primary passes: (1) portable
row/group projection into per-worker scratch, (2) aligned output-panel and
32-row activation tiling, and (3) shape/format route selection. Pass 1 was
fast for NEON row-dot layouts but repeated FP8 decode at prefill. Pass 2 cut
FP8 M16/M128 from 6.9202/53.2569 ms to 2.7201/20.5146 ms, while its generic
M1 panel route regressed and was rejected. Pass 3 restores mature row dots for
Apple decode and selected A8 paths while retaining exact panel accumulation
where it wins or avoids a quantization-boundary mismatch. A native-x86
follow-up makes M>1 panel-only, uses selected FP8 M1 panels, and phases an
entire output row at one thread; INT4 M1 fell from 2.3884 to 1.9803 ms and FP8
M1 from 3.5081 to 2.5267 ms.

The longer retained Apple group-A8 run takes 0.0992-0.3854 ms at M1,
1.5766-2.7520 ms at M16, and 14.6521-23.5864 ms at M128. It is
0.975x-2.188x versus fused typed SwiGLU followed by standalone quantization;
the largest wins are MXFP4/BitNet M16 and BitNet M128. One-thread CV is
0.0058-0.1265. The retained EPYC one-thread run takes
1.5774-2.9895, 6.4416-8.1003, and 44.2381-52.4732 ms respectively, with
0.892x-1.052x comparator ratios and CV 0.0007-0.0032. The 16-thread runs are
scaling context only because CV reaches 0.1495 on Apple and 0.2537 on EPYC.

- Hardware/toolchains: Apple M5 Max, aarch64 NEON, macOS 26.5.2, Apple Clang
  21.0.0; AMD EPYC 7702, x86-64 AVX2/FMA/F16C, Ubuntu Linux
  6.8.0-134-generic, GCC 13.3.0. CMake Release without LTO.
- Runtime: one and 16 threads, OS-default affinity/frequency, prepared weights,
  warm reusable output/workspace storage. The EPYC uses one socket/NUMA node,
  64 physical cores with SMT off.
- Working-tree label: `397f16a-dirty`; rsynced x86 metadata is `unknown`
  because `.git` is excluded.
- Artifacts: `f3-swiglu-pass1-fused-store-t1`,
  `f3-swiglu-pass2-static-mode-t1`,
  `f3-swiglu-pass3-f32-direct-store-t1`, `f3-swiglu-final-t{1,16}`,
  `x86-f3-swiglu-pass{4-plane-local,5-row-tile,6-int4-panel}-t1`,
  `x86-f3-swiglu-retained-t{1,16}`, `f3-quant-pass{1-row-groups,
  2-panel-m32,3-shape-select}-t1`, `f3-quant-final-t{1,16}`,
  `x86-f3-quant-pass4-m1-locality-t1`, and
  `x86-f3-quant-final2-t{1,16}` under `perf/results/2026-07-23/`.
- Decision: retain the architecture- and shape-selected routes. This completes
  F3 for the requested A4/A8/FP8/MX/NV activation-output matrix; BitNet a4.8
  sparse activation preparation remains Q9 rather than being inferred here.

## 2026-07-23: M3 canonical paired gate/up projection (F2)

Status: retain `qgemm_prepacked_gate_up_storage` and its M=1 wrapper for all 11
non-cache canonical weight layouts and FP32/FP16/BF16 activation/output
storage. Both matrices remain independently prepared; one CPU schedule owns
the pair. Multi-row tiles decode both weight planes around one activation
traversal, while M=1 keeps the mature format-specific row/panel dots.

Implementation: pass 0 is the same-binary baseline of two prepared projection
calls. Pass 1 places both complete traversals under one pool schedule. Pass 2
interleaves gate/up chunks so each activation load or typed conversion feeds
both accumulator sets. Pass 3 restores a paired row-dot specialization at M=1.
A follow-up locality experiment makes plane-major M=1 traversal x86-only: the
blanket version regressed Apple MXFP4, while x86 needs it to avoid switching
between two packed matrices. FP8 and typed x86 shapes select paired panels.

Correctness: `test_quant_projection_matrix` checks every canonical layout at
M=1/16/128, FP32/FP16/BF16 input, FP32 output, and direct FP16/BF16 output
rounding against two independently dequantized oracles. The focused test passes
on Apple AArch64 and AMD x86-64. Benchmark cases use separate FP64 gate and up
oracles before timing.

The three primary Apple passes use four representative INT4/FP8/MXFP4/BitNet
layouts, M=1/16/128 N512 K1024, one thread, three warmups, ten samples, and a
5 ms minimum:

| stage | M1 range ms | M16 range ms | M128 range ms | decision |
|---|---:|---:|---:|---|
| pass 1 one schedule, plane traversal | 0.8002-0.9947 | 3.0662-5.0049 | 22.4736-39.2561 | correctness baseline |
| pass 2 shared activation traversal | 0.5310-0.8674 | 2.4866-3.1956 | 18.9675-24.1836 | retain for M>1 |
| pass 3 paired M1 row dots | 0.0980-0.3879 | 2.6379-2.9496 | 19.1425-23.1016 | retain by shape |

Pass 2 improves M16 by up to 1.69x and M128 by up to 1.78x over pass 1.
Pass 3 improves M1 by 2.06x-10.15x over pass 1. In the longer Apple final,
M1 is 1.00x-1.07x, M16 is 0.97x-1.23x, and M128 is 1.01x-1.27x versus two
prepared calls. The one BitNet M16 result is a 2.8% regression and is treated
as neutral rather than a family-wide speedup. One-thread CV is 0.0088-0.1137.
At 16 threads, the pair is 1.18x-1.79x over the two-call comparator, with
0.0531-2.4023 ms medians and CV up to 0.2297.

The first x86 M1 row interleave was 0.62x-1.01x versus composition. The
x86-selected plane-local row and paired-panel follow-up removes that regression:
the retained one-thread M1 ratio is 0.999x-1.041x, M16 is 1.09x-1.13x, and
M128 is 1.04x-1.17x. Medians span 1.5037-2.9741, 5.9321-7.4922, and
44.1112-52.2643 ms respectively, with CV 0.0008-0.0051. The 16-thread x86
run is 0.97x-1.16x but has CV up to 0.2464, so it is scaling context only.

- Hardware/toolchains: Apple M5 Max, aarch64 NEON, macOS 26.5.2, Apple Clang
  21.0.0; AMD EPYC 7702, x86-64 AVX2/FMA/F16C, Ubuntu Linux
  6.8.0-134-generic, GCC 13.3.0. CMake Release without LTO.
- Runtime: one and 16 threads, OS-default affinity/frequency, prepared weights
  and hot activation storage. The EPYC uses one socket/NUMA node, 64 physical
  cores with SMT off, 256 MiB aggregate L3, and `schedutil`/`acpi-cpufreq`.
- Working-tree label: `397f16a-dirty`; rsynced x86 metadata is `unknown`
  because `.git` is excluded.
- Artifacts: `f2-gate-up-pass1-single-schedule-t1`,
  `f2-gate-up-pass2-shared-activation-t1`,
  `f2-gate-up-pass3-m1-paired-rows-t1`,
  `f2-gate-up-pass4-m1-plane-locality-t1`, `f2-gate-up-final-t{1,16}`,
  `x86-f2-gate-up-pass{4-plane-locality,5-selected-m1}-t1`, and
  `x86-f2-gate-up-retained-t{1,16}` under `perf/results/2026-07-23/`.
- Decision: retain the shape- and architecture-selected routes. F2 is complete
  for the weight-only projection matrix. F3 activation quantization remains
  separate work.

## 2026-07-23: Metal BaseQN semantic port and direct projection

Status: retain the portable BaseQ2/3/4/5/6/8 implementation and all three
projection refinements. The public surface covers dequantization, GEMV, GEMM,
embedding, QKV projection, and gate/up SwiGLU for group sizes 32/64/128,
symmetric and affine reconstruction, BF16/F16/E8M0 scales, and Q8-only E4M3
scales. FP32, FP16, and BF16 activation/output storage uses FP32 accumulation.

Implementation: pass 0 decoded the group scale and optional bias for every
weight element and repeated packed-weight decode for every activation row.
Pass 1 hoists scale/bias conversion once per group. Pass 2 changes M>1 to a
16-row activation tile so one decoded packed group feeds all tile rows. Pass 3
specializes the FP32 activation/output storage path outside the element loop;
FP16/BF16 continue through the checked universal conversion path.

Correctness: `test_base_q` checks all six widths, all three group sizes, every
legal scale encoding, both reconstruction modes, little-endian cross-byte code
packing, FP32/FP16/BF16 storage, invalid metadata and embedding ids, and the
QKV/SwiGLU consumers. The focused test passes in Release on Apple AArch64 and
AMD x86-64. Every benchmark case uses a separately materialized FP64 oracle
before timing. `scripts/check_parity_manifest.py` passes with the seven new
Metal operations and `base_qn` family mapped.

The three exploratory Apple runs use M=1/16, N512 K1024, one thread, three
warmups, ten samples, and a 5 ms minimum:

| stage | M1 range ms | M16 range ms | result versus prior stage | decision |
|---|---:|---:|---:|---|
| pass 0 per-element metadata decode | 1.3647-1.6281 | 22.2232-25.6164 | baseline | replace |
| pass 1 group metadata hoist | 0.6898-0.7229 | 10.9164-11.3451 | 1.89x-2.36x / 1.97x-2.28x | retain |
| pass 2 16-row decode reuse | 0.6837-0.7073 | 6.9319-7.5767 | neutral M1 / 1.46x-1.62x M16 | retain |
| pass 3 FP32 storage specialization | 0.6067-0.6312 | 3.6544-4.3906 | 1.09x-1.17x / 1.67x-2.07x | retain |

Relative to pass 0, pass 3 is 2.22x-2.58x faster at M1 and 5.24x-6.13x at
M16. A longer checked Apple run (five warmups, 30 samples) records
0.6517-0.6817 ms at M1 and 3.1370-4.5519 ms at M16; 16-thread medians are
0.1489-0.1537 and 0.4717-0.5076 ms. CV reaches 0.1099 at one thread and
0.1862 at 16 threads, so these final Apple values are bounds rather than a
low-variance publication tier.

The native EPYC final records 1.2064-1.3513 ms at M1 and 6.4695-7.4704 ms at
M16 with one-thread CV 0.0006-0.0038. Sixteen-thread medians are
0.1805-0.1964 and 0.4347-0.8340 ms, but CV reaches 0.3625, so no scaling claim
is made. The already-dequantized scalar comparator excludes conversion and
uses four dense bytes per weight: the packed route is still slower at M1 and
mixed at M16, so the measured claim is improvement over the original direct
decoder, not superiority to dense GEMM.

- References: `../QuixiCore-Metal/kernels/quantization/base_q/` and
  `../QuixiCore-Metal/.reference/baseRT/base-convert/CANONICAL_QUANT_SPEC.md`.
- Hardware/toolchains: Apple M5 Max, aarch64 NEON, macOS 26.5.2, Apple Clang
  21.0.0; AMD EPYC 7702, x86-64 AVX2/FMA/F16C, Ubuntu Linux
  6.8.0-134-generic, GCC 13.3.0. Both use CMake Release without LTO.
- Runtime: one and 16 threads, OS-default affinity/frequency, steady-state
  packed inputs. The EPYC uses one socket/NUMA node, 64 physical cores with
  SMT off, 256 MiB aggregate L3, and `schedutil`/`acpi-cpufreq`.
- Working-tree label: `397f16a-dirty`; the rsynced x86 snapshot records
  `unknown` because `.git` is excluded.
- Artifacts: `base-q-pass0-scalar-t1`,
  `base-q-pass1-group-scale-t1`, `base-q-pass2-mrow-tile16-t1`,
  `base-q-pass3-f32-storage-t1`, `base-q-final-t{1,16}`, and
  `x86-base-q-final-t{1,16}` under `perf/results/2026-07-23/`.
- Decision: retain all three refinements and the portable semantic surface.
  Wider ISA-specific bit-unpack/dot kernels remain future optimization work.

## 2026-07-23: M3 canonical fused projection epilogue (F1)

Status: retain the allocation-free fused bias/activation store as the canonical
F1 implementation, with no blanket throughput-speedup claim. It covers all 11
non-cache canonical weight layouts, FP32/FP16/BF16 activation and output
storage, and none, GELU-erf, GELU-tanh, SiLU, and ReLU2. The activation is
applied to the FP32 accumulator before the selected output rounding.

Implementation: pass 0 was the correctness-first composition: project to the
requested FP32 output, or allocate a complete FP32 output tensor for typed
output, then traverse it again for bias, activation, and conversion. Pass 1
moves the post-op into every canonical projection kernel's final store and
removes that allocation/pass. Pass 2 changes the multi-row final reduction and
store traversal to row-major order. Pass 3 adds a NEON four-lane ReLU2 final
reduction, bias, clamp, square, and FP32 store; other activations retain the
portable exact scalar formulas.

Correctness: `test_quant_projection_matrix` checks every layout at M=1/16/128,
all five activations, channel bias, FP32 output, and direct FP16/BF16 output
rounding against the same-kernel FP32 result. The benchmark uses an independent
FP64 projection oracle and checks both SiLU and ReLU2 before timing. Focused
tests pass on Apple AArch64 and AMD x86-64.

Three one-thread AArch64 passes use M=1/16/128 N512 K1024, four representative
INT4/FP8/MXFP4/BitNet layouts, five warmups, 20 samples, and a 5 ms minimum:

| stage | change | observed result | decision |
|---|---|---|---|
| pass 0 | projection then materialized post-op | same-binary comparator | replaced API path |
| pass 1 | fused final scalar store | 0.899x-1.164x at M16; noisy M128 | retain allocation removal |
| pass 2 | row-major final traversal | 3/4 SiLU M16 medians improve over pass 1; M128 remains noisy | retain locality |
| pass 3 | NEON ReLU2 lane store | exact checked output; mostly neutral against composed baseline | retain ISA seam |

The longer retained Apple run uses eight warmups, 30 samples, and a 10 ms
minimum. FP32 cases take 0.0494-0.1915 ms at M1, 1.4142-1.6628 ms at M16,
and 11.3401-12.4317 ms at M128. Direct FP16/BF16 output representatives take
0.0514-0.0576, 1.4545-1.6839, and 11.6254-12.4837 ms. Relative to the stronger
preallocated composition comparator, M16/M128 spans 0.987x-1.071x; tiny M1
typed stores span 0.877x-0.973x and are not claimed faster. CV reaches 0.1626
on one tiny/noisy case.

On the native EPYC, the final checked quick run takes 0.7566-1.4132 ms at M1,
3.5727-4.3852 ms at M16, and 24.5616-28.3551 ms at M128. CV is
0.0004-0.0090. Against the preallocated composition comparator the complete
set spans 0.976x-1.014x; typed M16/128 spans 1.004x-1.014x. This establishes a
stable neutral-cost fused route, not a throughput improvement.

- Hardware/toolchains: Apple M5 Max, aarch64 NEON, macOS 26.5.2, Apple Clang
  21.0.0; AMD EPYC 7702, x86-64 AVX2/FMA/F16C, Ubuntu Linux
  6.8.0-134-generic, GCC 13.3.0. Both use CMake Release without LTO.
- Runtime: one thread, OS-default affinity/frequency; Apple high-QoS hint;
  EPYC `schedutil`; steady-state prepared panels. The comparator uses
  preallocated scratch and is therefore stronger than the removed public
  composition, which allocated typed-output scratch per invocation.
- Working-tree label: `397f16a-dirty`; the rsynced x86 snapshot records
  `unknown` because `.git` is deliberately excluded.
- Artifacts: `f1-fused-epilogue-pass1-t1`,
  `f1-fused-epilogue-pass2-{row-major,activation-matrix}-t1`,
  `f1-fused-epilogue-pass3-neon-relu2-t1`,
  `f1-fused-epilogue-final-t1`, and
  `x86-f1-fused-epilogue-final-t1` under `perf/results/2026-07-23/`.
- Decision: retain the fused API/kernel path for its materialization guarantee
  and complete semantics. Do not advertise an F1 speedup tier from these
  results; future work may add vector GELU/SiLU approximations only if the
  umbrella tolerance and model-level checks permit them.

## 2026-07-23: M2 x86 canonical typed GEMV panels (K1-K15)

Status: retain the runtime-gated AVX2/F16C block-conversion and four-output-
panel-group route for FP16/BF16 M1 projection across all 11 non-cache canonical
weight layouts. Retain the output-panel traversal for FP32 E4M3/E5M2 only;
the same experiment regressed the other FP32 layouts, which remain on the
portable row route.

Implementation: pass 1 adds an isolated AVX2 BF16 expander, reuses the existing
isolated F16C converter, and changes typed x86 projection from per-output-row
scalar conversion to conversion once per packed eight-row output panel. Pass 2
tests the panel traversal for FP32 and narrows dispatch to E4M3/E5M2 after the
integer, microscale, FP4, NVFP4, and BitNet results regress. Pass 3 groups four
typed output panels so one converted activation block feeds up to 32 output
rows. Runtime `CpuFeatures` checks preserve the portable fallback on x86 CPUs
without AVX2/F16C; ISA flags remain isolated to their source files.

Correctness: `test_float_storage` and `test_quant_projection_matrix` pass on
the native host after each route change. All 41 architecture-independent and
executable tests, including `bench_smoke`, pass in the native Release build;
the separately executed parity manifest is host-independent and passes in the
complete local checkout. Every benchmark performs its independent FP64 check
before timing.

The three focused passes use M1 N512 K1024, one thread, five warmups, 20
samples, and a 5 ms minimum sample duration:

| stage | FP16 range ms | BF16 range ms | FP32 E4M3/E5M2 ms | decision |
|---|---:|---:|---:|---|
| pass 0 scalar row conversion | 1.5866-2.6017 | 1.1196-2.3038 | 1.6434-1.6582 | baseline |
| pass 1 one typed output panel | 0.7951-1.8545 | 0.7624-1.8679 | unchanged | retain |
| pass 2 FP32 output panels | unchanged | unchanged | 1.2522-1.2534 | retain for E4M3/E5M2 only |
| pass 3 four typed panels | 0.7518-1.8092 | 0.7259-1.7515 | 1.2545-1.2548 | retain |

- Pass 1 removes repeated activation conversion and improves every typed case.
- Pass 2 improves FP32 E4M3/E5M2 by 1.31x-1.32x, but its blanket form is
  rejected; format-selective routing avoids the measured regressions.
- Pass 3 wins or is within 1% on every typed case, wins 19/22 medians, and is
  1.98x-1.99x faster than pass 1 for MXFP8. Across the complete typed matrix,
  the retained route is 1.19x-2.34x faster than pass 0.

A longer final one-thread run (eight warmups, 30 samples, 10 ms minimum) records
0.7261-1.8206 ms for FP16/BF16 and 0.6902-1.5825 ms for FP32, with CV
0.0005-0.0074 across the 33 weight-only cases. At 16 threads those cases take
0.1194-0.2451 ms. These are prepared-panel throughput results; preparation is
excluded and no predecoded-scalar speedup claim is made for layouts that remain
decode-bound.

- Hardware: AMD EPYC 7702, one socket, 64 physical/logical cores (SMT off),
  256 MiB aggregate L3, one NUMA node, 1.0 TiB memory; AVX2, FMA, and F16C
  available, AVX-512/VNNI/AMX unavailable.
- OS/toolchain: Ubuntu Linux 6.8.0-134-generic; GCC 13.3.0; CMake Release; no
  LTO.
- Runtime: one and 16 threads, OS-default affinity, `schedutil` governor with
  `acpi-cpufreq`, steady-state prepared panels; host was otherwise idle.
- Working-tree label: local source snapshot `397f16a-dirty`, rsynced without
  `.git`, so harness metadata records `unknown`.
- Artifacts: `x86-canonical-pass0-t1`,
  `x86-canonical-pass1-typed-panels-t1`,
  `x86-canonical-pass2-f32-panels-t1`,
  `x86-canonical-pass3-typed-group4-t1`, and
  `x86-canonical-final-t{1,16}` under `perf/results/2026-07-23/`.
- Decision: retain all three dispatch refinements. M1 now has native x86
  three-pass evidence; wider AVX2 weight block dots and native M16/128
  three-pass records remain open.

## 2026-07-23: M2 x86 canonical M16/M128 projection (K1-K15)

Status: retain the runtime-gated AVX2 M-tile microkernel and four-output-panel
activation-reuse schedule for FP32/FP16/BF16 weight-only GEMM. Retain direct
AVX2 packed-code block dots plus four-panel activation reuse for every checked
dual-quantized pair except W8A8; W8A8 stays on its faster portable panel route.
No AVX-512, VNNI, or AMX performance tier is inferred from this AVX2 host.

Implementation: pass 1 measures the existing portable 32-row reuse tile and
dual per-output-lane block dots on native x86. Pass 2 moves decoded weight-panel
accumulation into an isolated AVX2/FMA source and adds direct packed integer,
FP8, MXFP8, MXFP4, NVFP4, and BitNet block dots. Pass 3 groups four adjacent
output panels so one activation conversion/unpack feeds up to 32 output rows.
The pass-2 one-panel FP32 route improved M16 but regressed M128; the grouped
schedule replaces it. Direct W8A8 regressed 4.4258 to 5.8649 ms at M16, so the
final format selector restores the 4.4237 ms portable result. ISA flags remain
per-source and every call is guarded by runtime AVX2 detection.

The weight-only passes used all 11 canonical layouts at M16, representative
M128 cases, FP32/FP16/BF16 activation storage, N512 K1024, one thread, three
warmups, 12 samples, and a 3 ms minimum:

| representative | pass 1 portable tile ms | pass 2 AVX2 panel FMA ms | pass 3/final panel-group ms |
|---|---:|---:|---:|
| INT4 FP32 M16 / M128 | 3.5885 / 25.6470 | 2.8627 / 33.9845 | 1.3438 / 7.5883 |
| FP8 E4M3 FP32 M16 / M128 | 3.6743 / 26.2147 | 3.1008 / 34.8711 | 1.6041 / 8.6348 |
| MXFP4 FP32 M16 / M128 | 4.9966 / 29.0903 | 3.5376 / 36.6424 | 2.0849 / 10.4602 |
| INT4 FP16 M16 / M128 | 2.5608 / 19.2171 | 1.3869 / 7.8538 | 1.3875 / 7.8085 |

Across the complete paired matrix, the retained route is 2.05x-2.99x over
pass 1 for FP32 M16, 2.67x-3.69x for FP32 M128, 1.58x-2.67x for typed M16,
and 2.19x-2.73x for typed M128. Pass 2 is retained directly for typed tiles;
pass 3 is essential for FP32 because activation reuse removes the M128
one-panel regression.

The dual-quantized passes used the same shape and timing policy:

| representative | pass 1 portable ms | pass 2 AVX2 one-panel ms | pass 3 grouped ms | final selected ms |
|---|---:|---:|---:|---:|
| W4A4 M16 / M128 | 26.0655 / 208.5304 | 13.5980 / 108.9628 | 12.5460 / 99.9628 | 12.4246 / 99.2575 |
| FP8 E4M3 M16 / M128 | 22.5008 / 180.1281 | 11.7216 / 93.7579 | 10.0465 / 80.4037 | 10.0680 / 80.4545 |
| MXFP4 M16 / M128 | 51.4508 / 412.0276 | 12.3624 / 98.8412 | 11.4497 / 91.6209 | 11.4669 / 91.7445 |
| W8A8 M16 | 4.4258 | 6.2375 | 5.8649 | 4.4237 portable |

Final dual speedups over pass 1 span 1.00x-4.49x at M16 and 1.14x-4.49x
across the four M128 representatives. The checked 16-thread final takes
0.2014-0.3466 ms for weight-only M16, 1.0140-1.6037 ms for M128,
0.6706-1.7970 ms for dual M16, and 5.1828-6.3543 ms for dual M128. One-thread
CV is at most 0.0102 in the selected run. Unpinned 16-thread M128 dual CV
reaches 0.2966, so those threaded medians are scaling context rather than a
low-variance latency claim.

Correctness: `test_quant_projection_matrix` passes after every route change.
The final native EPYC Release suite passes all 46 locally executable tests;
the ISA-disabled and sanitizer results are recorded with the final validation
entry below. The benchmark's Q6_K comparator tolerance was corrected from an
overly strict local threshold to a still-stricter-than-contract 0.1% relative
threshold; its observed maximum relative error is 0.000603 and the final
checked benchmark has no error rows.

- Hardware/toolchain: AMD EPYC 7702, x86-64 AVX2/FMA/F16C, 64 cores with SMT
  off, Ubuntu Linux 6.8.0-134-generic, GCC 13.3.0, CMake Release without LTO.
- Runtime: one and 16 threads, OS-default affinity, `schedutil`, steady-state
  prepared panels; the host was otherwise idle.
- Working-tree label: local source snapshot `397f16a-dirty`, rsynced without
  `.git`, so harness metadata records `unknown`.
- Artifacts: `x86-m2-pass1-row-panels-t1`,
  `x86-m2-pass2-avx2-panel-fma-t1`,
  `x86-m2-pass3-panel-groups-dual-pass2-t1`,
  `x86-m2-final-dual-panel-groups-t1`, and
  `x86-m2-final-selected-t{1,16}` under `perf/results/2026-07-23/`.
- Decision: retain AVX2/FMA tile accumulation, four-panel activation reuse,
  direct dual-code dots, and the measured W8A8 fallback. This supplies native
  x86 M16/M128 three-pass evidence for the remaining K1-K15 projection cells.

## 2026-07-23: M2 dual-FP8 GEMM output-panel reuse (K10-K12)

Status: retain the eight-output-panel AArch64 route for E4M3/E4M3,
E5M2/E4M3, and dual MXFP8 at M=16/128. This closes the remaining one-thread
dual-FP8 prefill gap on the checked quick shapes: all three M16 pairs and the
M128 E4M3 representative now beat the same-run already-dequantized scalar
GEMM.

Implementation: the M1 grouped FP8 kernel is generalized over activation rows
and scheduled as activation-row/output-panel-group tasks. It decodes each
packed activation FP8 block once, retains independent accumulator panels across
K, and reuses that decode across two, four, then eight prepared weight panels.
Packed weight decode, E4M3/E5M2 scale application, MXFP8 E8M0 scaling, and FP32
reduction remain direct; no float operand matrix is materialized.

Correctness: `test_quant_projection_matrix` checks all three pairs at
M=1/16/128 with an independent exact-code FP64 oracle, including the generalized
activation-row indexing. Every benchmark checks before timing. The complete
Release suite passes 47/47 tests and the x86_64 library cross-build succeeds;
the grouped SIMD route is an AArch64 performance claim only.

The three follow-up passes use N512 K1024, one thread, five warmups, 20 samples,
and a 5 ms minimum sample duration:

| stage | M16 three-pair range ms | M128 E4M3 ms | decision |
|---|---:|---:|---|
| prior pass 3 one-panel activation reuse | 3.3941-3.6073 | 27.9594 | superseded baseline |
| pass 4 two-panel reuse | 3.0892-3.3093 | 25.5222 | retain concept |
| pass 5 four-panel reuse | 3.0380-3.1134 | 24.3162 | retain concept |
| pass 6 eight-panel reuse | 2.8416-3.0778 | 23.6880 | retain |

- Pass 4 improves the former route by 1.08x-1.10x at M16 and 1.10x at M128;
  all four checked cells cross above the scalar comparator.
- Pass 5 improves all medians again by up to 1.06x while retaining stable
  0.28%-2.67% CV.
- Pass 6 is the best median for every checked cell. Relative to the former
  accepted pass 3, it is 1.17x-1.19x faster.

At one thread, E4M3/E4M3, E5M2/E4M3, and MXFP8 M16 are 1.14x, 1.25x, and
1.10x over the same-run already-dequantized scalar GEMM; M128 E4M3 is 1.12x.
Pass-6 CV is 0.0033-0.0112. At six threads the M16 pairs take
0.5394-0.6335 ms and M128 E4M3 takes 4.4996 ms, with CV 0.0253-0.0638. The
comparator is serial, so threaded ratios are throughput context only.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON available.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release; no LTO.
- Runtime: one and six threads, OS-default affinity/frequency, high-QoS harness
  hint, steady-state prepared panels.
- Working-tree label: `397f16a-dirty`.
- Artifacts: `canonical-dual-fp8-gemm-pass4-panel-group2-t1`,
  `canonical-dual-fp8-gemm-pass5-panel-group4-t1`,
  `canonical-dual-fp8-gemm-pass6-panel-group8-t1`, and
  `canonical-dual-fp8-gemm-final-t6` under `perf/results/2026-07-23/`.
- Decision: retain pass 6. The checked AArch64 dual-FP8 M=1/16/128 tier now
  exceeds its predecoded scalar comparator; native x86 packed FP8 remains open.

## 2026-07-23: M2 dual-FP8 GEMV output-panel reuse (K10-K12)

Status: retain the eight-output-panel AArch64 route for E4M3/E4M3,
E5M2/E4M3, and dual MXFP8 at M=1. This follow-up closes the one-thread gap
left by the original per-row FP8 block dots: all three packed routes now beat
the same-run already-dequantized scalar GEMV at quick and comprehensive model
shapes.

Implementation: the previous kernel decoded both activation and weight FP8 for
every output row. Pass 4 switches to the existing 16-row prepared output panel,
decoding each activation block once per panel. Pass 5 introduces a dedicated
four-panel kernel that retains four sets of reduction chains across K and
reuses one decoded activation block across up to 64 output rows. Pass 6 widens
reuse to eight panels, or 128 output rows, while retaining enough tasks for
large-vocabulary parallelism. Weight E4M3/E5M2 decode, scale application, and
MXFP8 E8M0 scaling remain direct from packed bytes; no operand is materialized.

Correctness: `test_quant_projection_matrix` covers all three pairs at
M=1/16/128 with an independent exact-code FP64 oracle. The quick and
comprehensive benchmarks independently dequantize both operands and check the
public result before timing. The complete Release suite passes 47/47 tests and
the x86_64 library cross-build succeeds. The grouped SIMD implementation is
AArch64-specific; the x86 build retains the portable checked panel fallback.

The three follow-up passes use M1 N512 K1024, one thread, five warmups, 20
samples, and a 5 ms minimum sample duration:

| stage | three FP8 pairs ms | speedup vs same-run dequantized scalar | decision |
|---|---:|---:|---|
| prior pass 3 per-row FP8 SIMD | 0.3649-0.3708 | 0.61x-0.62x | superseded baseline |
| pass 4 one output panel | 0.2075-0.2194 | 0.94x-0.99x | retain as fallback shape |
| pass 5 four-panel reuse | 0.1865-0.1949 | 1.10x-1.15x | retain concept |
| pass 6 eight-panel reuse | 0.1784-0.1900 | 1.09x-1.16x | retain |

- Pass 4 removes repeated activation decode across the 16 rows in a prepared
  panel and improves the former accepted route by 1.66x-1.76x.
- Pass 5 reuses activation decode across four panels and improves pass 4 by
  1.11x-1.12x. One E5M2 sample has CV 0.1887, so pass 5 is not used as the
  publication result.
- Pass 6 doubles the group again, improves all three medians over pass 5, and
  records CV 0.0082-0.0125. Relative to the former pass 3, it is
  1.93x-2.05x faster.

At comprehensive M1 N4096 K4096, E4M3/E4M3, E5M2/E4M3, and MXFP8 take
6.0164, 5.8007, and 6.1269 ms and are 1.31x, 1.35x, and 1.27x over their
same-run already-dequantized scalar baselines; CV is 0.0120-0.0175. At six
threads on the quick shape they take 0.0461-0.0482 ms. The comparator is
serial, so its threaded speedup ratio is not used as an ISA claim.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON available.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release; no LTO.
- Runtime: one and six threads, OS-default affinity/frequency, high-QoS harness
  hint, steady-state prepared panels.
- Working-tree label: `397f16a-dirty`.
- Artifacts: `canonical-dual-fp8-pass4-output-panels-t1`,
  `canonical-dual-fp8-pass5-panel-group4-t1`,
  `canonical-dual-fp8-pass6-panel-group8-t{1,6}`, and
  `canonical-dual-fp8-final-comprehensive-t1` under
  `perf/results/2026-07-23/`.
- Decision: retain pass 6. The multi-row follow-up documented above closes the
  corresponding AArch64 prefill gap; native x86 packed microkernels remain
  open.

## 2026-07-23: M2 canonical FP16/BF16 GEMM panel groups (K1-K3)

Status: retain the AArch64 typed-input four-panel-group route for all 11
non-cache canonical weight layouts at M=16/128. FP16 and BF16 activations now
use native block conversion and reuse each converted MxK tile across up to four
prepared output panels. Act-order keeps the checked general gather kernel, and
non-AArch64 targets keep the portable scalar-conversion panel kernel.

Implementation: pass 1 replaces scalar conversion in the existing M32 output-
panel traversal with 32-element native FP16/BF16 conversion blocks. Pass 2
changes traversal to groups of two output panels, retaining two panel
accumulator sets across K so the converted activation tile is reused. Pass 3
widens that group to four panels. The retained kernel holds no full MxK FP32
temporary, keeps four independent FP32 reduction chains, and stores through
the public FP32/FP16/BF16 output adapter.

Correctness: `test_quant_projection_matrix` covers FP16/BF16 inputs and all
three output storage types for every layout at M=1/16/128, including ragged
panels and exact typed-output rounding. The benchmarks round their typed input
first, independently dequantize weights, use an FP64 oracle, and check before
timing. The complete Release suite passes 47/47 tests. The x86_64 library
cross-build succeeds; only the portable fallback is compiled there, so this is
not an x86 performance claim.

Three focused passes use N512 K1024, one thread, three warmups, 10 samples, and
a 5 ms minimum sample duration. M16 includes all 22 layout/storage pairs; M128
uses eight INT4/FP8/MXFP4/BitNet representatives:

| stage | M16 range ms | M128 range ms | decision |
|---|---:|---:|---|
| pass 0 scalar element conversion | 1.7876-3.9225 | 13.0439-27.5334 | baseline |
| pass 1 vector block conversion | 1.4499-1.7730 | 11.0241-11.9070 | retain as one-panel fallback |
| pass 2 two-panel activation reuse | 1.2530-2.2189 | 9.5976-10.8390 | retain concept; one noisy M16 tail |
| pass 3 four-panel activation reuse | 1.2485-1.6961 | 8.8737-10.8948 | retain |

- Pass 1 (conversion SIMD) makes FP16 and BF16 converge to the same tier and
  removes the FP16 scalar-conversion cliff. It improves pass 0 by as much as
  2.31x at M16 and 2.50x at M128.
- Pass 2 (two-panel reuse) reduces repeated activation conversion and improves
  all eight M128 representatives over pass 1; M16 has enough timing noise at
  this duration to produce one 2.2189 ms tail.
- Pass 3 (four-panel reuse) improves the arithmetic mean ratio over pass 2 by
  1.035x at M16 and 1.055x at M128. It wins 12/22 M16 cases and 6/8 M128
  representatives, removes the pass-2 tail, and is retained for its better
  aggregate and M128 behavior.

Relative to pass 0, every accepted one-thread case improves: 1.16x-2.90x at
M16 and 1.43x-2.91x at M128. The retained route is 2.08x-2.86x and
2.60x-3.23x over the same-run already-dequantized scalar GEMM at M16 and M128,
respectively. A longer six-thread confirmation takes 0.2808-0.3549 ms at M16
and 1.8708-2.1704 ms at M128; CV is 0.0192-0.1172 and 0.0258-0.0662. The
harness's scalar comparator is serial, so its six-thread 9.97x-15.04x ratios
are throughput context rather than an ISA-only speedup claim.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON and native FP16 conversion available.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release; no LTO.
- Runtime: one and six threads, OS-default affinity/frequency, high-QoS harness
  hint, steady-state prepared panels.
- Working-tree label: `397f16a-dirty`.
- Artifacts: `canonical-typed-gemm-pass0-t1`,
  `canonical-typed-gemm-pass1-block-convert-t1`,
  `canonical-typed-gemm-pass2-panel-group2-t1`,
  `canonical-typed-gemm-pass3-panel-group4-t1`, and
  `canonical-typed-gemm-final-stable-t6` under
  `perf/results/2026-07-23/`.
- Decision: retain pass 3. The AArch64 typed M=1/16/128 projection tier now has
  three-pass evidence; native x86 typed microkernels remain open M2 work.

## 2026-07-23: M2 canonical FP16/BF16 GEMV panels (K1-K15)

Status: retain the AArch64 typed-input output-panel route for all 11 non-cache
canonical weight layouts at M=1. Direct FP16 and BF16 activation storage now
matches the FP32 block-dot tier without a full K-element conversion buffer.
Act-order and irregular group maps deliberately retain the checked scalar-
gather fallback.

Implementation: pass 1 adds native eight-element FP16-to-FP32 conversion and
BF16 bit expansion in 32-element block scratch, feeding the existing prepared
weight block dots. Pass 2 changes typed M1 traversal from individual output
rows to 16-row output panels, so each activation conversion is reused across
up to 16 weight rows. Rowwise affine INT8, whose prepared block spans K, is
processed in checked 32-element conversion/dot chunks with block-level zero
correction. Output remains directly selectable as FP32, FP16, or BF16 through
the public storage API.

Correctness: `test_quant_projection_matrix` exercises FP16 and BF16 input and
output storage for all 11 layouts, including ragged panels and bit-exact typed
output rounding. The new benchmark cases construct rounded typed inputs,
independently dequantize weights, accumulate the oracle in FP64, and compare
before timing. The complete Release suite passes 47/47 tests. The x86_64
library cross-build succeeds; the new panel route is AArch64-only and the
portable scalar typed path remains the x86 fallback.

Three optimization passes use M1 N512 K1024 across 22 layout/storage pairs:

| stage | threads | typed range ms | speedup vs same-run dequantized scalar | decision |
|---|---:|---:|---:|---|
| pass 0 scalar element conversion | 1 | 0.3100-0.8555 | 0.25x-0.70x | baseline |
| pass 1 vector block conversion per row | 1 | 0.0677-0.6447 | 0.34x-3.30x | retain as direct-row fallback |
| pass 2 conversion reuse per output panel | 1 | 0.0432-0.2014 | 1.11x-5.22x | retain |
| pass 2 scheduling baseline | 6 | 0.0237-0.0701 | 3.23x-9.57x | retain grain 1 |
| pass 3 grain-2 scheduling candidate | 6 | 0.0238-0.0772 | 2.95x-9.48x | reject |

- Pass 1 (conversion SIMD): replace scalar per-element half/BF16 conversion
  with AArch64 vector conversion. It removes the conversion cliff for 20 of
  22 cases; K-wide affine INT8 intentionally remains on the scalar fallback at
  this stage.
- Pass 2 (panel reuse): convert each activation tile once for 16 prepared
  output lanes and add chunked K-wide INT8 handling. It improves pass 1 by up
  to 14.92x and makes every typed case faster than its dequantized scalar
  baseline. Relative to pass 0, the accepted route is 1.86x-17.13x faster.
- Pass 3 (scheduling): increase output-panel grain from one to two. Only 7 of
  22 cases improve, the average ratio regresses, and maximum CV rises from
  0.3575 to 0.5005, so grain one is restored.

At comprehensive M1 N4096 K4096, all typed routes pass at
1.5326-6.3958 ms and are 1.32x-5.56x over the same-run dequantized scalar
baseline. Quick one-thread CV is 0.0107-0.2118 and comprehensive CV is
0.0098-0.2200; the highest values belong to adaptively batched small cases.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON and native FP16 conversion available.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release; no LTO.
- Runtime: one and six threads, OS-default affinity/frequency, high-QoS harness
  hint, steady-state prepared panels; quick uses 3/10/5 ms and comprehensive
  uses 3/10/5 ms warmup/sample/minimum-duration settings.
- Working-tree label: `397f16a-dirty`.
- Artifacts: `canonical-typed-gemv-pass0-t1`,
  `canonical-typed-gemv-pass1-block-convert-t1`,
  `canonical-typed-gemv-pass2-panel-convert-t{1,6}`,
  `canonical-typed-gemv-pass3-grain2-t6`, and
  `canonical-typed-gemv-final-comprehensive-t1` under
  `perf/results/2026-07-23/`.
- Decision: retain pass 2 with grain one. Continue with native x86 typed and
  packed-format microkernels before closing M2.

## 2026-07-23: M2 canonical dual-quant GEMM panels (K5-K14)

Status: retain the portable cell kernel and AArch64 packed activation/output-
panel microkernel for M>=16. All ten checked dual-quant pairs now bypass the
generic per-element decoder at M=1/16/128. At this stage FP8-family M16/M128
was approximately at the one-thread dequantized scalar baseline; the grouped
FP8 follow-up documented above supersedes that result. The other pairs exceeded
the baseline at both measured thread counts.

Implementation: the portable pass extends the direct packed block-dot
framework across the MxN output grid. The retained AArch64 route schedules
activation-row/output-panel tasks, decodes each packed activation block once,
and reuses its integer, FP4-table, ternary, or FP8 vector representation across
up to 16 prepared weight rows. Four reduction chains preserve the M1 summation
strategy, output lanes remain contiguous, and scales/zero corrections stay at
block granularity. Blocks wider than the 32-element panel microtile and
irregular act-order/group maps retain the checked direct-cell or general panel
fallback.

Correctness: `test_quant_projection_matrix` covers every pair at M=1/16/128,
including ragged three-row output panels. The benchmark uses independent
dequantization, an FP64 oracle, and checked same-binary scalar baselines. The
full Release suite passes 47/47 tests, including benchmark smoke and the parity
manifest. The x86_64 library cross-build succeeds; only the portable fallback
is compiled there, so no x86 runtime or performance claim is made.

Three focused optimization passes used N512 K1024, one thread, three warmups,
10 samples, and a 5 ms minimum sample duration:

| stage | M16 all ten pairs ms | M128 representative pairs ms | decision |
|---|---:|---:|---|
| pass 0 general element decode | 5.0605-8.1888 | 39.9066-60.6835 | baseline |
| pass 1 direct output cells | 0.6044-5.3631 | 4.5918-42.5568 | retain as portable/fallback |
| pass 2 weight-major cells | 0.5512-5.3631 | 5.0732-42.6502 | reject; strided stores regress most pairs |
| pass 3 activation/panel NEON | 0.3911-3.6073 | 3.1676-27.9594 | retain on AArch64 |

- Pass 1 (packed dataflow): reuse the M1 block-dot implementations directly
  for each MxN output cell. Integer/FP4 M16 improves by 8.16x-10.65x and the
  representative integer/FP4 M128 cases improve by 8.69x-10.48x.
- Pass 2 (weight-major scheduling): keep one prepared weight row hot across M.
  MXFP4 and BitNet improve slightly, but contiguous-output pairs regress as
  much as 41%; this traversal was measured and removed.
- Pass 3 (panel microkernel): keep activation rows outermost and reuse one
  decoded activation block across up to 16 contiguous output lanes. It improves
  accepted pass 1 by 1.19x-1.66x at M16 and 1.15x-1.56x at M128. Relative to
  pass 0, the final route is 2.16x-14.98x faster at M16 and 2.17x-13.53x faster
  across the four M128 representatives.

At one thread, non-FP8 M16 pairs are 4.61x-8.64x over the same-run already-
dequantized scalar GEMM; E4M3/E4M3, E5M2/E4M3, and MXFP8 are 0.94x-1.00x.
Representative M128 W4A4/MXFP4/BitNet are 5.20x-8.55x, while E4M3/E4M3 is
0.96x. At six threads all checked M16 pairs take 0.0988-0.7014 ms and are
4.97x-38.12x over the serial scalar baseline used by the harness; the four
M128 representatives take 0.6414-5.4697 ms and are 5.15x-45.68x. The
six-thread CV is 0.0337-0.1485, so the smallest cases retain visible scheduler
noise.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON/DotProd/I8MM available.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release; no LTO.
- Runtime: one and six threads, OS-default affinity/frequency, high-QoS harness
  hint, steady-state prepared panels.
- Working-tree label: `397f16a-dirty`.
- Artifacts: `canonical-dual-gemm-pass0-t1`,
  `canonical-dual-gemm-pass1-cellwise-t1`,
  `canonical-dual-gemm-pass2-weight-major-t1`,
  `canonical-dual-gemm-pass3-panel-neon-t1`, and
  `canonical-dual-gemm-final-t6` under `perf/results/2026-07-23/`.
- Decision at this stage: retain pass 3 and the portable fallbacks. The later
  typed and grouped-FP8 follow-ups supersede its open AArch64 items; native x86
  panels remain open.

## 2026-07-23: M2 canonical dual-quant GEMV block dots (K4-K14)

Status: retain the portable M=1 packed-pair traversal and the AArch64 NEON
integer/FP4/ternary and FP8 block dots. The integer, FP4, NVFP4, and BitNet
routes beat the same-run already-dequantized scalar GEMV. At this stage FP8
E4M3/E5M2 and MXFP8 remained slower than that baseline; the output-panel
follow-up documented above supersedes that result and closes the gap.

Implementation: `qgemv_prepacked_quantized` now recognizes ten checked packed
weight/activation pairs: W4A4, W4A8, W8A8, E4M3/E4M3, E5M2/E4M3, MXFP8,
MXFP4, NVFP4, BitNet/A8, and BitNet/A4. The portable route traverses prepared
weight rows and packed activation blocks directly, hoists scales and zero
points, applies affine correction from integer sums, and uses four reduction
chains. AArch64 kernels use NEON nibble interleave/table decode, widened
integer products, vector ternary unpack, and vector FP8 exponent/mantissa
decode for both operands. Non-identity act-order, group-index maps, and scale
intervals that do not align with prepared blocks retain the general checked
panel kernel.

Correctness: the independent exact-code projection oracle covers all ten pairs
at M=1/16/128, including the optimized M=1 route and the portable multi-row
fallback. The benchmark independently dequantizes both operands, accumulates
its oracle in FP64, and checks before timing. The complete Release suite passes
47/47 tests and the x86_64 library cross-build succeeds; no x86 runtime or
performance claim is made.

Three focused passes used M1 N512 K1024, one thread, five warmups, 20 samples,
and a 5 ms minimum sample duration:

| pair | pass 0 general panel ms | pass 1 packed row ms | pass 2 integer/FP4 NEON ms | pass 3 FP8 NEON ms | pass 0 to final |
|---|---:|---:|---:|---:|---:|
| W4A4 | 0.7045 | 0.6113 | 0.0396 | 0.0414 | 17.00x |
| W4A8 | 0.6331 | 0.2268 | 0.0418 | 0.0437 | 14.48x |
| W8A8 | 0.5330 | 0.0577 | 0.0412 | 0.0414 | 12.86x |
| FP8 E4M3/E4M3 | 0.8797 | 0.6982 | 0.5426 | 0.3708 | 2.37x |
| FP8 E5M2/E4M3 | 0.8945 | 0.7162 | 0.5373 | 0.3649 | 2.45x |
| MXFP8 | 0.9483 | 0.6572 | 0.5345 | 0.3672 | 2.58x |
| MXFP4 | 0.8529 | 0.7379 | 0.0432 | 0.0390 | 21.88x |
| NVFP4 | 1.0264 | 0.6334 | 0.0579 | 0.0610 | 16.83x |
| BitNet/A8 | 0.5435 | 0.1223 | 0.0452 | 0.0484 | 11.22x |
| BitNet/A4 | 0.6108 | 0.6779 | 0.0481 | 0.0506 | 12.06x |

- Pass 1 (dataflow): replace per-element virtual layout decode inside the
  output-panel kernel with direct packed block products and block-level scale
  and zero-point application. W8A8 became 9.24x faster than pass 0; scalar
  nibble and FP decode remained costly.
- Pass 2 (integer SIMD): use NEON interleave/table unpack and widened integer
  products for W4A4, W4A8, W8A8, MXFP4, NVFP4, and BitNet. Those routes became
  1.40x-17.08x faster than pass 1.
- Pass 3 (FP8 SIMD): construct E4M3/E5M2 values from packed exponent/mantissa
  fields in vectors for both operands. The three FP8-family routes improved
  another 31.8%-32.6% from pass 2. Unchanged integer-path movement is run
  variance and does not represent a different route.

The final quick one-thread run is 0.0390-0.3708 ms and the integer/FP4/BitNet
routes are 3.78x-5.95x faster than the same-run dequantized scalar GEMV. The
three FP8-family routes were 0.61x-0.62x that baseline. At comprehensive M1
N4096 K4096, integer/FP4/BitNet routes take 1.1924-1.9143 ms and are
4.39x-7.01x faster; FP8-family routes take 11.6617-12.0155 ms and are
0.70x-0.73x. A six-thread quick artifact was retained as scheduling evidence,
but its 0.0272-0.4910 CV and sub-0.4 ms cases are too noisy for a scaling claim.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON/DotProd/I8MM available.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release; no LTO.
- Runtime: one-thread quick and comprehensive final evidence; OS-default
  affinity/frequency, high-QoS harness hint, steady-state prepared panels.
- Working-tree label: `397f16a-dirty`.
- Artifacts: `canonical-dual-gemv-pass0-t1`,
  `canonical-dual-gemv-pass1-portable-t1`,
  `canonical-dual-gemv-pass2-neon-integer-t1`,
  `canonical-dual-gemv-pass3-neon-fp8-t1`,
  `canonical-dual-gemv-final-t6`, and
  `canonical-dual-gemv-final-comprehensive-t1` under
  `perf/results/2026-07-23/`.
- Decision at this stage: retain the direct packed pair routes. The later
  output-panel follow-up supersedes the M1 FP8 scheduling; native x86 block
  dots and further multi-row FP8 tiles remain open.

## 2026-07-23: M2 canonical weight-only GEMV block dots (K1)

Status: keep as the AArch64 FP32 M=1 tier for all 11 non-cache canonical
layouts. M2 remains in progress for quantized-activation block dots,
format-specialized FP16/BF16 GEMV, and native x86 routes.

References inspected:

- llama.cpp `2beefef68825aed8de05f0d89981bf5d05266a3c`, especially
  `ggml/src/ggml-cpu/arch/arm/quants.c` Q4_0 and MXFP4 vector dots and
  `ggml/src/ggml-cpu/arch/arm/repack.cpp` output-row GEMV traversal;
- KleidiAI `13cd35993d8439143aff1e756a862d366acded0d`, especially the
  `qsi4c32p` NEON/DotProd 1x4 kernels under `kai/ukernels/matmul/`; and
- BitNet `16da220ae2b510caff437d403288882687f44ae5` and T-MAC
  `7042f8f73330bd083bc1e4bc5ccb3f88a4904aee`, for packed ternary/nibble
  lookup ordering. Their generated model-specific LUTs were not copied into
  the public CPU contract.

Implementation: M=1 no longer enters the M32 output-lane kernel. The portable
route traverses one prepared output row at a time, hoists block scale and zero
metadata, and uses four independent reductions. The AArch64 route unpacks
canonical adjacent INT4/FP4 nibbles with NEON zip/table operations, widens
INT8 and ternary codes, performs affine zero corrections per block, and
reduces contiguous activation vectors. FP8 E4M3/E5M2 are decoded directly
from exponent/mantissa bits in four-lane vectors, including subnormal and
non-finite encodings; MX and NV scale application stays outside the dot.
Non-identity act-order and group-index metadata deliberately retain the
checked portable gather path. No dequantized weight matrix or workspace is
allocated.

Correctness: `test_quant_projection_matrix` now executes the SIMD-sized
group-32 paths for signed/affine INT4 and FP4 and checks every finite E4M3 and
E5M2 code through prepared GEMV. Existing coverage retains all 11 layouts,
FP32/FP16/BF16 storage, M=1/16/128, dual-quant modes, ragged output panels,
act-order, and invalid inputs. The complete Release suite passes 47/47 tests,
including the parity manifest. All measured cases use the benchmark's
independent FP64 exact-code oracle and 1e-4 FP32 check.

Three focused passes used M1 N512 K1024, one thread:

| format | pass 0 shared panel ms | pass 1 row traversal ms | pass 2 NEON block dot ms | pass 3 vector code decode ms | speedup |
|---|---:|---:|---:|---:|---:|
| INT4 | 0.3871 | 0.3314 | 0.0453 | 0.0446 | 8.68x |
| FP8 E4M3 | 0.3881 | 0.3641 | 0.2843 | 0.1838 | 2.11x |
| MXFP4 | 0.4392 | 0.3669 | 0.0480 | 0.0479 | 9.17x |
| BitNet | 0.2760 | 0.3164 | 0.1805 | 0.0571 | 4.83x |

- Pass 1 (dataflow): change M=1 from activation-broadcast/output-lane order
  to an output-row/block traversal. INT4, FP8, and MXFP4 improved 6.2%-16.5%;
  the scalar ternary unpack regressed and was not accepted alone.
- Pass 2 (SIMD): add whole-block NEON nibble interleave, FP4 table lookup,
  widened INT8, affine correction, and ternary dot products. A failed initial
  signed-INT4 run exposed two's-complement rather than biased nibble semantics;
  `canonical-gemv-pass2-neon-fixed-t1` is the corrected checked artifact.
- Pass 3 (decode): replace scalar FP8 table gathers with vector exponent/
  mantissa construction and replace scalar ternary unpack with vector shift/
  zip. FP8 improved another 35.4% and BitNet 68.4% from corrected pass 2.

The final all-layout quick run is 0.0369-0.1876 ms at one thread and
0.0185-0.0435 ms at six threads. It is 1.10x-5.63x and 4.77x-11.61x faster,
respectively, than the same-run already-dequantized scalar matrix. At the
comprehensive M1 N4096 K4096 shape, all 11 layouts pass at 1.2205-6.0853 ms
and 1.26x-6.30x over that baseline. Quick-run CV is 0.0090-0.0466 at one
thread and 0.0300-0.1776 at six threads; the latter reflects sub-50-us cases
despite adaptive batching. Comprehensive CV is 0.0029-0.0183.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON/DotProd/I8MM available.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release; no LTO.
- Runtime: 1 and 6 threads, OS-default affinity/frequency, high-QoS harness
  hint, steady-state prepared panels; quick finals use 5 warmups, 30 samples,
  and 5 ms minimum sample duration. Comprehensive uses 3/20/10 ms.
- Working-tree label: `397f16a-dirty`.
- Artifacts: `canonical-gemv-pass0-t1`,
  `canonical-gemv-pass1-rowwise-t1`,
  `canonical-gemv-pass2-neon-fixed-t1`,
  `canonical-gemv-pass3-vector-decode-t1`,
  `canonical-gemv-final-all-t{1,6}`, and
  `canonical-gemv-comprehensive-all-t1` under
  `perf/results/2026-07-23/`.
- Decision: retain the portable gather fallback and AArch64 FP32 block dots.
  Do not extend this performance claim to typed or quantized activations, x86,
  SVE/SME, or unmeasured imported metadata topologies.

## 2026-07-23: M2 canonical universal projection (K1-K15 shared framework)

Status: keep as the portable/AArch64 M>=16 canonical projection tier. The M=1
measurements in this entry are superseded by the K1 block-dot entry above; M2
still requires dual-operand, typed-activation, and native x86 evidence.

Implementation: `qgemm_prepacked` now consumes version-2 canonical
`CpuPackedWeights` panels instead of rejecting them. The new typed route loads
FP32, FP16, or BF16 activations directly in the K loop and stores FP32, FP16,
or BF16 without an MxK staging allocation. Format selection is resolved before
the parallel region for INT4/U4, INT8, FP8 E4M3/E5M2, FP4, MXFP8, MXFP4,
NVFP4, and BitNet. M=16/128 use row-panel x 32-row tiles; M=1 uses the same
checked block decoders. `qgemm_prepacked_quantized` directly consumes canonical
A4/A8/FP8/MX/NV activation packets. GPTQ `act_order` is indexed in the hot K
loop and never materializes a permuted activation tensor.

Correctness: `test_quant_projection_matrix` covers all 11 non-cache canonical
weight layouts at M=1/16/128, each direct FP32/FP16/BF16 activation storage,
direct FP16/BF16 output conversion, W4A4, W4A8, W8A8, mixed FP8,
MXFP8, MXFP4, NVFP4, BitNet A8/A4, ragged output panels, invalid shapes, and a
non-identity GPTQ group map plus act-order. Exact-code oracles independently
dequantize both operands. The act-order test passes a `Workspace` and verifies
zero bytes used. The full Release suite passes 47/47 tests; the parity manifest
also passes. The benchmark uses an FP64 exact-code oracle and a 1e-4 FP32
parallel-reduction check, still 300x tighter than the umbrella quantized
tolerance.

Three focused passes used M16/M128 N512 K1024, one thread, with identical
checked INT4, FP8 E4M3, MXFP4, and BitNet cases:

| format | pass 0 M16 / M128 ms | pass 1 M16 / M128 ms | pass 2 M16 / M128 ms | final M16 / M128 ms |
|---|---:|---:|---:|---:|
| INT4 | 2.9119 / 23.7875 | 3.0640 / 20.2986 | 2.3568 / 13.6316 | 1.6166 / 11.2002 |
| FP8 E4M3 | 3.5715 / 29.9204 | 3.4952 / 22.7496 | 2.9084 / 16.3656 | 1.4161 / 11.5107 |
| MXFP4 | 3.0689 / 24.6277 | 3.1744 / 21.1991 | 2.4517 / 14.5228 | 1.6087 / 11.4949 |
| BitNet | 3.3171 / 27.2879 | 3.2308 / 21.7788 | 2.8099 / 14.9651 | 1.5592 / 11.5144 |

- Pass 1 (algorithm/dataflow): increase the reuse tile from 16 to 32 M rows.
  M128 improved 14.7%-24.0%; M16 was unchanged within run variance.
- Pass 2 (SIMD/microkernel): make output-row lanes contiguous and apply NEON
  FMA across the prepared row panel. This reduced the pass-1 M128 medians by
  27.8%-36.9% and the M16 medians by 13.0%-25.8%.
- Pass 3 (scheduling): retain one-panel scheduling after a checked six-thread
  run. Final six-thread medians are 0.0910-0.1343 ms at M1,
  0.3551-0.3711 ms at M16, and 2.6717-3.1051 ms at M128.
- Final remediation: hoist layout dispatch and block scales, use FP8 decode
  tables, and keep four independent FP32 reduction chains. These changes are
  included in the final rows above. At one thread the M16/M128 routes are
  2.21x-2.53x and 2.48x-2.56x faster than the same-run predecoded scalar GEMM.

In this earlier shared-framework run, M1 decode remained an honest gap: at the
small hot N512 K1024 quick shape the
canonical packed routes take 0.2704-0.4348 ms versus 0.2223-0.2265 ms for an
already-dequantized scalar matrix. At N4096 K4096 they take 8.4875-14.0134 ms.
The target avoids the 64 MiB dequantized matrix and its conversion cost. No M1
speedup is claimed from this artifact; the dedicated block-dot work and its
replacement evidence are recorded in the K1 entry above.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON/DotProd/I8MM available.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release; no LTO.
- Runtime: 1 and 6 threads, OS-default affinity/frequency, high-QoS harness
  hint, steady-state prepared panels; final 5 warmups, 20 samples, 5 ms minimum
  sample duration.
- Working-tree label: `397f16a-dirty`.
- Pass artifacts:
  `perf/results/2026-07-23/full-matrix-k1-k3-pass0-t1/`,
  `full-matrix-k1-k3-pass1-t1/`, `full-matrix-k1-k3-pass2-t1/`, and
  `full-matrix-k1-k3-pass3-t6/`. Stable finals:
  `full-matrix-k1-k3-final2-t1/`, `full-matrix-k1-k3-final2-t6/`, and
  `full-matrix-k1-final-t1/`.
- Decision: keep the allocation-free universal reference/tiled route and its
  M>=16 NEON performance tier. Use the K1 entry above for M1 claims and do not
  claim complete M2 until the remaining dual/typed/x86 passes are measured.

## 2026-07-22: M1 canonical lifecycle and checkpoint ingestion

Status: keep; M1 lifecycle exit gate complete. This does not claim M2 packed
projection support.

Implementation: `include/quixicore_cpu/quant_import.h` and
`kernels/quantization/quant_import_ref.cpp` add an owned canonical tensor,
FP32/FP16/BF16 packers for integer, FP8, FP4, MXFP8, MXFP4, NVFP4, and BitNet,
and import adapters for AWQ, GPTQ v1/v2 plus act-order, AutoRound target
formats, SmoothQuant/AZP, and checked BitNet I2_S blocks. Canonical bytes remain
unchanged when `CpuPackedWeights::prepare` builds version-2 row panels and
aligned side tables. At this milestone generic GGUF QGEMM rejected those
panels; the M2 entries above supersede that historical limitation.

References inspected:

- AutoRound `6ff414b15728f97848936551021d0b180dd40320`, especially
  `auto_round/export/export_to_awq/utils.py` and the AutoGPTQ/GGUF exporters;
- GPTQ `2d65066eeb06a5c9ff5184d8cebdf33662c67faf` and T-MAC
  `7042f8f73330bd083bc1e4bc5ccb3f88a4904aee`, for packed zeros, `g_idx`, and
  act-order semantics;
- SmoothQuant `c61476d728e42ae0d8a35e7e78494edcac3237b5`, for per-channel
  weights and activation metadata; and
- BitNet `16da220ae2b510caff437d403288882687f44ae5`, for canonical I2_S and
  rebuildable TL preparation boundaries.

Correctness: `test_quant_import` checks exact AWQ lane reversal and decoded
outputs, GPTQ v1 zero+1, GPTQ-v2 symmetric fields, non-identity `g_idx` and
act-order, every AutoRound target layout, SmoothQuant row sums/AZP, BitNet
reserved codes, all FP8 scale topologies, all canonical prepared layouts, and
FP16/BF16 input parity. `test_quantization` checks every finite FP8 code,
signed identity, every adjacent midpoint tie, and 20,000 deterministic random
values against an exhaustive nearest-value oracle. Full Release CTest is the
landing gate below. The Release build passes 46/46 CTest targets. A focused
ASan/UBSan build compiled the complete library but hit the repository's known
Apple libc++ ABI link mismatch while linking `test_quant_import`; no sanitizer
pass is claimed from that build.

Three optimization passes, all R/N128 K1024, one thread:

| format | pass 1 linear search ms | pass 2 binary table ms | pass 3 direct bits ms | final/pass 1 |
|---|---:|---:|---:|---:|
| FP8 E4M3FN | 34.8737 | 1.2458 | 0.3618 | 96.39x |
| FP8 E5M2 | 25.1908 | 1.1670 | 0.3656 | 68.89x |
| MXFP8 | 35.8365 | 1.2718 | 0.4114 | 87.10x |
| NVFP4 | 2.6581 | 0.6741 | 0.5000 | 5.32x |

Pass 1 used the original exact but O(127)-per-element FP8 search. Pass 2 kept
exact rounding with a monotonic representable-value table and binary search.
Pass 3 directly rounds IEEE FP32 exponent/significand bits and is retained;
the independent exhaustive test oracle prevents the optimization from defining
its own correctness. Non-FP8 pack/import cases stayed within ordinary run
variance, so no unsupported speedup is claimed for them.

- Hardware: Apple M5 Max, 18 logical/physical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON/DotProd/I8MM available.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release; no LTO.
- Runtime: one thread, OS-default affinity/frequency, high-QoS harness hint,
  steady-state buffers; final 5 warmups, 30 samples, 10 ms minimum samples.
- Working-tree label: `7f61388-dirty`.
- Pass artifacts: `perf/results/2026-07-22/m1-pass1-t1-fixed/`,
  `perf/results/2026-07-22/m1-pass2-t1/`,
  `perf/results/2026-07-22/m1-pass3-t1/`; stable final
  `perf/results/2026-07-22/m1-pass3-final2-t1/`.
- Decision: retain direct bit rounding and the complete canonical lifecycle;
  reject the original linear encoder and supersede the binary-search pass.

## Required Entry Fields

- Kernel, operation, dtype or quant format, and shape set.
- Correctness command and result.
- Baseline measurement.
- Candidate or current measurement.
- Hardware model, core count, memory configuration, and ISA target.
- Operating system, compiler, runtime, and thread settings.
- Command line, git commit or working-tree label, warmups, iterations, median,
  and variance or min/max.
- Keep or reject decision.

## Log

| Date | Kernel | Dtype / Format | Shape Set | Target | Baseline | Candidate | Decision | Evidence |
|---|---|---|---|---|---:|---:|---|---|
| 2026-07-22 | M0 full-matrix benchmark registration | f32 plus Q4_0/Q4_K/Q6_K, INT8, MXFP4/MXFP8, NVFP4, BitNet b1.58 | 23 pass-0 decode/prefill/fusion/serving/cache cases | Apple M5 Max, aarch64 NEON/DotProd/I8MM, 1 thread | same-run scalar, dequantized, materialized, or decomposed references: 0.0414-14.2072 ms where applicable | registered current routes: 0.0874-10.1409 ms; CV 0.0103-0.0775 | keep all eight executable M0 families; no new support/speedup claim; MXFP8 and BitNet results explicitly retain optimization gaps | `perf/results/2026-07-22/m0-pass0-t1/` |
| 2026-07-22 | universal floating storage, three passes | FP16/BF16 storage, FP32 accumulation | round trip N1048576; BF16 softmax R256 H4096 | Apple M5 Max, aarch64 FP16, 1/6 threads | scalar pass: FP16 1.6856/1.7549, BF16 0.5976/0.6010 ms; explicit softmax 1.4096/0.2985 ms | final FP16 0.0961/0.0466, BF16 0.1569/0.0548, typed softmax 1.4028/0.2995 ms | keep reusable typed adapter, compiler-vectorized BF16, threading, NEON FP16; manual unroll rejected; typed dispatch is performance-neutral | `perf/results/2026-07-22/float-storage-{pass123,native-final}-t{1,6}/` |
| 2026-07-22 | llama unary/GLU selector closure | f32 | all 22 unary and six GLU modes; softplus/OAI SwiGLU N1048576 | portable/aarch64, 1/6 threads | unary 2.6758/0.5109; OAI GLU 2.4515/0.6971 ms | unary final 2.5420/0.5358; OAI GLU final 1.1954/0.2811 ms | keep selector hoist and source-exact direct sigmoid; reject both manual unrolls; OAI 2.05x/2.48x over pass 1 | `perf/results/2026-07-22/{unary-pass*,glu-pass*,activation-final}-t{1,6}/` |
| 2026-07-22 | exhaustive llama operation parity | f32 | Conv C16 O32 S32; GLA B4 T64 H8 D32; DSV4 T4096 | portable/aarch64, 1/6 threads | serial conv 1.9942 ms; one-thread GLA/DSV4 0.2882/0.1385 ms | six-thread 0.7870/0.0772/0.0364 ms | keep portable semantics and outer-task parallelism; 2.53x/3.73x/3.80x | `perf/results/2026-07-22/parity-final-t{1,6}/` |
| 2026-07-22 | complete quant authoring/activation lifecycle | seven IQ formats; Q8_0/Q8_1/Q8_K | IQ R1 K4096; Q8 R64 K4096; Q4_KxQ8_K N1024 K4096 | portable/aarch64, 1/6 threads | one-thread 11.1633-75.6794 / 0.1509-0.3383 / 1.4484 ms | six-thread 2.8692-20.0169 / 0.0515-0.0991 / 0.2876 ms | keep block-parallel authoring/activation paths; weighted work matches uniform overhead | `perf/results/2026-07-22/parity-final-t{1,6}/` |
| 2026-07-22 | complete llama stored-format GEMV sweep | 23 non-Q4_0/Q8_0 stored formats / f32 activation | M1 N1024 K4096 | aarch64 NEON, 1 thread | same-run element decode 14.1884-17.3736 ms | direct packed block dot 0.3178-3.0547 ms | keep all direct routes; 4.68x-45.22x, CV 0.0079-0.0759 | `perf/results/2026-07-22/llama-quants-final-neon-t1-stable/` |
| 2026-07-22 | GGUF three-pass GEMV sweep | Q4_K/Q5_K/Q6_K and IQ formats / f32 | M1 N1024 K4096 | aarch64 NEON, 1/6 threads | pass-0 8.3503-14.7039 / 1.7567-2.9806 ms | final 0.3137-2.5965 / 0.0890-0.5164 ms | keep direct packed-block dots and IQ4 table lookup; 4.0x-31.1x / 4.3x-22.5x over pass 0 | `perf/results/2026-07-22/opt3-{pass0,final}-t{1,6}/` |
| 2026-07-22 | dense/quantized GEMM three-pass sweep | f32; Q4_0/K, Q6_K, IQ4_XS | M16/M128 N256/512/1024 K512/1024/1408 | aarch64 NEON, 1/6 threads | pass-0 dense 0.3282 / 0.1066 ms; Q4_K/Q6_K M16 1.3853-1.6778 / 0.3207-0.3802 ms | final dense 0.3319 / 0.0964 ms; Q4_K/Q6_K 0.9457-1.2375 / 0.2695-0.3599 ms | reject dense retile; keep generic panel SIMD and small-IQ4 canonical bypass | same |
| 2026-07-22 | paged attention + MLA three-pass sweep | f32 | paged B2 HQ8 HKV2 S512 D64; MLA B2 H8 S512 DL64 DR16 | aarch64 NEON, 1/6 threads | pass-0 paged 0.1935 / 0.0562 ms; materialized MLA 0.2437 / 0.2453 ms | final paged 0.1126 / 0.0457 ms; online MLA 0.2033 / 0.0595 ms | keep online score tiles, SIMD f32 score dot, and request/head-parallel MLA; reject value-update SIMD | `perf/results/2026-07-22/{opt3-final,opt3-mla-final}-t{1,6}/` |
| 2026-07-22 | fused decode projections three-pass sweep | Q4_0 / f32 | SwiGLU N1024 K1408; QKV HQ8 HKV2 D64 K1408 | aarch64 NEON, 1/6 threads | pass-0 0.2074/0.0774 and 0.0635/0.0335 ms | final 0.2094/0.0780 and 0.0677/0.0351 ms | keep paired gate/up semantic fusion; reject Q/K pairing and activation-load-sharing candidates; no new speedup claim | same |
| 2026-07-22 | streaming LM-head three-pass sweep | Q4_0 / f32 | R2 V8192 H1024 | aarch64 NEON, 1/6 threads | pass-0 1.2227 / 0.3927 ms | final 0.8795 / 0.2353 ms | keep vocabulary-parallel selection and multi-row packed decode reuse; reject grain retune | same |
| 2026-07-22 | grouped MoE three-pass sweep | Q4_0 / f32 | R64 E8 K256 I512 | aarch64 NEON, 1/6 threads | pass-0 1.2931 / 0.3810 ms | final 1.2636 / 0.3387 ms | keep paired gate/up block dot; reject shared float decode, activation reload, and output-tile candidates | same |
| 2026-07-22 | norm-add-quant three-pass sweep | f32 / group int8 | R512 H4096 G128 | aarch64 NEON, 1/6 threads | pass-0 1.0311 / 0.2664 ms | final 0.6351 / 0.1804 ms | keep four-way sumsq, 16-byte quant pack, and four-way maximum reduction | same |
| 2026-07-22 | FFT + Mamba three-pass sweep | f32 | B1 H2 L1024; B1 H2 L128 D32 | aarch64/portable, 1/6 threads | pass-0 FFT 0.0546/0.0569; Mamba 0.0973/0.0714 ms | final FFT 0.0550/0.0407; Mamba 0.0356/0.0342 ms | keep complex-f32/batch-one FFT and float recurrent state; reject normalization fusion and Mamba NEON | same |
| 2026-07-22 | GGUF SIMD GEMV | Q4_0/K, Q5_K, Q6_K, IQ formats / f32 | M1 N4096 K4096; K/IQ N1024 | aarch64 NEON, 1/6 threads; x86 variants compile/test | scalar/element decode 2.8578 / 0.5892 ms Q4_0; 13.8664-15.1367 / 2.4908-3.6649 ms K/IQ | 1.1953 / 0.2798 ms Q4_0; 8.4295-14.6360 / 1.7739-3.1180 ms K/IQ | keep dispatch; Q4_0 2.39x/2.11x, every K/IQ format improves 1.03x-1.64x | `perf/results/2026-07-22/qgemv-allformats-final-t{1,6}/` |
| 2026-07-22 | blocked dense/quantized GEMM | f32; Q4_0/K, Q6_K, IQ4_XS | M16/M128 N256/512/1024 K512/1024/1408 | aarch64 NEON, 1/6 threads | scalar dense 2.9337 / 3.0997 ms; canonical quantized 0.6082-14.1906 ms | dense 0.3336 / 0.1005 ms; packed quantized 0.3313-7.4150 ms | keep; dense 8.80x/30.85x, Q4_0 M128 1.84x/2.94x, K/IQ 6.75x-9.04x; bypass panel for small Q4_0 | `perf/results/2026-07-22/{optimization-plan-complete,prepacked-allformats-final2}-t{1,6}/` |
| 2026-07-22 | online attention, fused decode, streaming selection, tiled MoE | f32 / Q4_0 | focused decode/prefill shapes | aarch64 NEON, 1/6 threads | materialized/decomposed 0.0397-1.7029 ms at 6 threads | 0.0391-0.4526 ms at 6 threads | keep online attention (2.96x), fused SwiGLU (1.31x), fused QKV+RoPE+KV (1.02x), and grouped MoE (3.76x) at six threads; LM-head tiling is allocation-bounded and neutral/noisy | `perf/results/2026-07-22/optimization-plan-complete-t{1,6}/` |
| 2026-07-22 | fused norm-add-quant | f32 / group int8 | R512 H4096 G128 | aarch64 NEON, 1/6 threads | decomposed preallocated 3.4495 / 1.3100 ms | row-local fused 1.0248 / 0.2545 ms | keep; 3.37x/5.15x and no rows-by-hidden temporary | `perf/results/2026-07-22/norm-fused-neon-t{1,6}/` |
| 2026-07-22 | FFT convolution and recurrent Mamba2 | f32 | B1 H2 L1024; B1 H2 L128 D32 | portable radix-2/recurrent, 1/6 threads | direct convolution 1.8173 / 1.8128 ms; source expansion 3.7921 / 3.9839 ms | FFT 0.0553 / 0.0608 ms; Mamba 0.0986 / 0.0740 ms | keep; 29.79x-53.83x with direct fallback for non-power-of-two/small FFT lengths | `perf/results/2026-07-22/optimization-plan-complete-t{1,6}/` |
| 2026-07-22 | CPU packed-panel/workspace prerequisites | q4_0 weights / f32 activation | M16 N1024 K1408 | aarch64 NEON layout, 1/6 threads | canonical QGEMM 4.0272 / 1.7384 ms | prepacked QGEMM 2.0219 / 0.6028 ms | keep; exact output, 1.99x/2.88x, retained workspace | `perf/results/2026-07-22/prerequisites-t{1,6}/` |
| 2026-07-22 | Colibri CPU algorithm batch | row int8, packed int4, f32 selection, q4_0 MoE | M4 N1024 K1408; V65536; R16 W8192 K2048; R32 E8 K256 N512 | aarch64 NEON/DotProd/I8MM, 1/6 threads | scalar/full-sort/per-row 0.4502-4.9532 / 0.1656-4.1210 ms | 0.0663-1.8725 / 0.0357-1.5431 ms | keep optimized numeric/selection routes; keep MoE union only for shared multi-thread batches | `perf/results/2026-07-22/colibri-port-final-t{1,6}/` |
| 2026-07-22 | qgemv_w8a8 q4_0 SDOT (dotprod_i8) | q4_0 weight / int8 act | quant_matmul m=1 (perf validation pending) | aarch64 DotProd | q4_0 w8a8 scalar ref | NEON SDOT (perf pending) | land; correctness validated (NEON==ref 1.3e-6), perf pending maintainer link env | see "2026-07-22: qgemv_w8a8 q4_0 SDOT" section below |
| 2026-07-22 | MXFP8 logical-scale GEMM | E4M3 / E8M0 | M16 N128 K256 G32 | portable table-decoded reference, 1/6 threads | direct decoder 1.6688 / 0.3660 ms | lookup decoder 0.4608 / 0.1229 ms | keep lookup; 3.62x/2.98x faster, still below predecoded dense | `perf/results/2026-07-22/sibling-entrypoints-{final,lookup}-t{1,6}/` |
| 2026-07-22 | fused RMSNorm-add + dynamic quantization | f32 / group int8 | R512 H4096 G128 | portable fused reference, 1/6 threads | 3.2327 / 1.2658 ms | 3.2725 / 1.2906 ms | keep semantic fusion; 1.2-2.0% allocation cost, no speedup claim | `perf/results/2026-07-22/ported-ops-final-{t1,t6}/` |
| 2026-07-22 | q4_0 qgemv + q4_0/q8_0 qgemv_w8a8 | q4_0/q8_0, f32/int8 activation | quant_matmul m=1 N4096 K4096 | portable refs + aarch64 DotProd, 6 threads | q8 W8A8 ref 0.7394 ms | q8 W8A8 DotProd 0.1386 ms | keep public routes; q4 references are correctness anchors, q8 DotProd is 5.33x | `perf/results/2026-07-22/qgemv-formats-final-t6-fixed/` |
| 2026-07-22 | sibling semantic port batch | f32 / q8_0 | five representative quick shapes | aarch64 portable + existing NEON q8_0 route, 1/6 threads | scalar/decomposed 0.8276-31.7397 ms | 0.1520-3.8159 ms | keep portable candidates and parallel routes; no new ISA or family-wide support claim | `perf/results/2026-07-22/all-kernels-final-{t1,t6}/` |
| 2026-07-21 | qgemv + rms_norm correctness hardening | q8_0 / f32 | quant_matmul m=1 + decode_small + R512 stress | aarch64 NEON | qgemv 0.990 ms; RMS R512 0.259 ms | qgemv 0.969 ms; RMS R512 0.260 ms | keep; contract fixes with no material hot-path regression | `perf/results/2026-07-21/review-{baseline,candidate-final,rms-baseline-repeat,rms-candidate-repeat}/` |
| 2026-07-07 | qgemv (contract realignment) | q8_0 | quant_matmul m=1 (4096x4096, 8192x8192, 16384x4096) | aarch64 NEON f32-act | 4.127 ms | 1.034 ms | keep neon as contract default (4.0x over ref, family numerics); dotprod_i8 demoted to env-only | `perf/results/2026-07-07/033244-quick/` |
| 2026-07-07 | quant_gemv + rms_norm (threading) | q8_0 / f32 | quant_matmul m=1 + R512 stress | aarch64, 8-12 threads | 0.303 ms | 0.068 ms | keep row-partitioned threading (4.3-4.5x, saturates aggregate DRAM BW) | `perf/results/2026-07-07/0307{23,45,51}-quick/` |
| 2026-07-07 | rms_norm | f32 | decode_small (R1-R4, H2048/H4096) + R512 stress | aarch64 NEON | 2.26 us | 0.52 us | keep neon variant (4.3-4.6x over ref) | `perf/results/2026-07-07/024347-quick/` |
| 2026-07-07 | quant_gemv | q8_0 | quant_matmul m=1 (4096x4096, 8192x8192, 16384x4096) | aarch64 NEON DotProd | 4.314 ms | 0.301 ms | keep dotprod variant (14.4x, 51% of DRAM roofline) | `perf/results/2026-07-07/023619-quick/` |
| 2026-07-07 | quant_gemv | q8_0 | quant_matmul m=1 (4096x4096, 8192x8192, 16384x4096) | aarch64 baseline flags | 4.319 ms | 4.441 ms | reject multi-acc candidate; keep plain loop as ref | `perf/results/2026-07-07/022305-quick/` |

## 2026-07-22: exhaustive llama operation and quant lifecycle closure

Status: retained portable-reference coverage. The pinned llama.cpp CPU
inventory now has 105/105 operation symbols classified: 89 public numerical
mappings, seven validated view/layout adapters, and nine non-numerical enum
markers, callback hooks, or pool selectors. Nested drift tables additionally
cover all 22 numerical `GGML_UNARY_OP_*` modes and all six numerical
`GGML_GLU_OP_*` modes; `GGML_OP_SILU_BACK` maps to its exact derivative rather
than to a gated derivative. The quant inventory has canonical
pack, unpack, and f32 GEMV coverage for all 25 stored formats, plus public
Q8_1/Q8_K activation intermediates. These counts are enforced by
`scripts/check_parity_manifest.py`; ISA performance tiers remain separate.

New operation paths include scalar arithmetic/reductions/layout transforms;
im2col/col2im, 2-D/3-D/depthwise/transposed convolution and pooling/backward;
relative-position and window transforms; GLA; RWKV6/RWKV7; and all three DSV4
hyper-connection stages. Recurrent state is explicit and parallel work is
partitioned by sequence/head. Convolution partitions independent output
channels; DSV4 partitions tokens.

The quant lifecycle pass added exact Q2_K/Q3_K/Q4_K/Q5_K/Q6_K and IQ4_NL/XS
packers, weighted IQ2_XXS/IQ2_XS/IQ3_XXS/IQ3_S/IQ2_S/IQ1_S/IQ1_M packers,
canonical Q8_0/Q8_1/Q8_K activation layouts, and a quantized-activation GEMV
contract route. During sanitizer validation, the new IQ3_S encoder exposed an
eight-byte sign-scratch overwrite; its region scratch is now correctly sized
for all 64 sign codes and the complete sanitizer suite passes.

Three focused implementation/optimization passes were retained:

| pass | change | measured outcome |
|---|---|---|
| 1 | direct portable operation recurrences and multi-seed IQ lattice search | correctness baseline; IQ2_S R1 K256 was 3.8177 ms |
| 2 | remove repeated codebook-extrema scans and use llama-compatible f32 convolution accumulation | IQ2_S fell to 2.7280 ms; convolution became bit-exact to the independent f32 oracle |
| 3 | block/row/token/output-channel parallel regions with hoisted strides and state-local reuse | retained; all final quick cases pass their embedded oracle |

The nested unary selector received its own three passes: (1) a direct
per-element selector, (2) one selector dispatch per call with template-fixed
element loops, and (3) manual four-way unrolling. Pass 2 improved softplus from
2.6758/0.5109 ms to 2.4371/0.4640 ms at one/six threads. Pass 3 regressed to
2.5798/0.5158 ms and was rejected. The retained code's independent final repeat
was 2.5420/0.5358 ms (CV 0.0134/0.0463), exact to the scalar oracle.

OpenAI SwiGLU also received three passes: (1) the shared overflow-stable
sigmoid helper, (2) llama's source-exact direct exponential expression, and
(3) manual four-way unrolling. Pass 2 reduced N1048576 from
2.4515/0.6971 ms to 1.2074/0.2814 ms at one/six threads. Pass 3 regressed to
1.4535/0.3384 ms and was rejected. The retained final repeat measured
1.1954/0.2811 ms (CV 0.0126/0.0586), bit-exact to the independent expression.

Final Release medians:

| path | one thread ms | six threads ms | scaling / baseline |
|---|---:|---:|---:|
| seven weighted IQ packers, R1 K4096 | 11.1633-75.6794 | 2.8692-20.0169 | 3.78x-3.91x thread scaling |
| Q8_0/Q8_1/Q8_K pack, R64 K4096 | 0.1509/0.1791/0.3383 | 0.0515/0.0596/0.0991 | 2.93x/3.01x/3.41x |
| Q4_K x Q8_K GEMV, N1024 K4096 | 1.4484 | 0.2876 | 5.04x; exact versus explicit unpack |
| conv2d, C16 O32 S32 K3 | 3.7463 | 0.7870 | 2.53x versus 1.9942 ms serial baseline at six threads |
| GLA, B4 T64 H8 D32 | 0.2882 | 0.0772 | 3.73x thread scaling |
| DSV4 combine, T4096 | 0.1385 | 0.0364 | 3.80x thread scaling |
| unary softplus, N1048576 | 2.5420 | 0.5358 | 4.65x over same-run serial baseline at six threads |
| OpenAI SwiGLU, N1048576 | 1.1954 | 0.2811 | 4.59x over same-run serial baseline at six threads |

Correctness and portability: native aarch64 Release CTest passes 43/43;
ASan+UBSan+float-cast-overflow passes 42/42 with leak detection disabled because
Apple's ASan runtime does not support it; x86_64 Release compiles AVX2,
AVX-512, VNNI, and AMX source sets and passes 38/38 under Rosetta. The parity
checker also verifies that every mapped operation names a real public CPU
function, derives the complete live quant-type inventory from llama's
`enum ggml_type`, rejects unary or GLU selector drift, and maps all 66
operation-level plus 29 quant-family entries published by the pinned sibling
manifests. CUDA's pinned manifest remains family-level, which is recorded as a
source-metadata limit rather than inferred operation-level evidence.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; native aarch64 NEON with DotProd and I8MM.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release.
- Settings: one or six threads, OS-default affinity/frequency, 3 warmups,
  10 timed samples, 5 ms minimum sample; correctness enabled.
- Command: `quixicore_cpu_bench --preset quick --kernel
  quant_lifecycle,llama_parity --threads {1,6} --warmup 3 --iters 10
  --min-sample-ms 5`.
- Working-tree label: `7f61388-dirty`.
- Raw results: `perf/results/2026-07-22/parity-final-t{1,6}/`; exploratory
  passes are retained under `quant-lifecycle-pass*`, `llama-parity-pass*`, and
  `unary-pass{1,2,3}-t{1,6}`. The nested-selector final repeat is
  `perf/results/2026-07-22/activation-final-t{1,6}/`; OpenAI SwiGLU passes are
  retained under `glu-pass{1,2,3}-t{1,6}`.

## 2026-07-22: complete llama.cpp stored-quant CPU paths

Status: retained for canonical decode/GEMV and superseded for authoring by the
complete lifecycle entry above. Compute and authoring cover all 25 stored
llama.cpp weight formats. Q8_1 and Q8_K remain activation/dot-partner layouts,
not stored-weight enum members, but now have public activation pack/unpack APIs.

References inspected at implementation time: llama.cpp `2beefef68`, vLLM
Marlin `4b594b4aa`, and QuixiCore-Metal `bc968fc`. The CPU application of those
algorithms keeps canonical GGUF blocks at the API boundary, expands scales once
per block/sub-block, performs packed table/nibble dots directly, and leaves
ISA-specific panel packing behind `CpuPackedWeights`.

Coverage added in this pass:

- Canonical layouts and element decoders for Q1_0, Q2_0, IQ3_S, IQ2_S,
  IQ1_M, and TQ1_0; NVFP4 is now the canonical 64-value/36-byte four-scale
  layout. UE4M3 is decoded as unsigned, and IQ1_M uses the signed CPU grid
  ordering rather than the GPU-transposed lookup table.
- Canonical public packers for Q1_0, Q2_0, Q4_1, Q5_0, Q5_1, MXFP4, NVFP4,
  and TQ1_0, alongside Q4_0, Q8_0, and TQ2_0. The subsequent lifecycle pass
  added Q2_K through Q6_K, IQ4_NL/XS, and all seven importance-sensitive IQ
  encoders. `qgemv_pack_weighted` accepts an explicit importance matrix;
  `qgemv_pack` supplies uniform importance. Exact golden bytes pin every
  deterministic non-IQ layout, while IQ output is checked through canonical
  decode, error bounds, determinism, and llama.cpp interoperability.
- Direct aarch64 block-dot routes for every stored llama format. Classic Q4/Q5
  and FP4 use NEON nibble/table lookup; K, IQ, and ternary formats consume
  packed fields without a 256-float decode scratch block.
- `qgemv_formats` now has a focused case for every stored format that was not
  already covered by the dedicated Q4_0/Q8_0 cases.

Three measured passes:

| pass | candidate | result |
|---|---|---|
| 1 | element-at-a-time canonical decoder | correctness/performance baseline |
| 2 | decode one complete block to aligned f32 scratch, then blocked dot | 0.93x-1.20x versus element decode; useful portable fallback, rejected as the optimized default |
| 3 | direct packed NEON dot with block-local scale/table reuse | retained for all formats; TQ1_0 was further restructured into fixed 160/80/16-value base-3 sections after the branch/division version regressed |

Primary stable result at one thread, M1 N1024 K4096: 0.3178-3.0547 ms versus
14.1884-17.3736 ms for the same-binary independent element decoder,
4.68x-45.22x faster. All target CVs are 0.0079-0.0759 and all embedded
oracles pass; maximum reported relative error is 9.37e-4, within the benchmark
tolerance. Six-thread artifacts are retained as scaling context, but the
single-thread run is the performance claim because its complete format sweep
passes the handbook variance gate.

Correctness and portability:

- Native Release CTest: 38/38.
- ASan + UBSan + float-cast-overflow CTest: 37/37.
- x86_64 cross-build and Rosetta CTest, including AVX2/AVX-512 sources: 32/32.
- Focused canonical pack/unpack, arbitrary-payload block-dot, and forced-ref
  tests: 7/7.

Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
efficiency), 128 GB, aarch64 NEON. OS/toolchain: macOS 26.5.2, Apple Clang
21.0.0 (`clang-2100.1.1.101`). Primary command:
`QUIXICORE_CPU_GGUF_GEMV_VARIANT=neon quixicore_cpu_bench --preset quick
--kernel qgemv_formats --threads 1 --warmup 5 --iters 20 --min-sample-ms 30`.
Working-tree label: `1d3dc15-dirty`. Raw pass artifacts are
`perf/results/2026-07-22/llama-quants-pass{1-ref,2-blocked,3-neon}/`; the stable
retained artifact is
`perf/results/2026-07-22/llama-quants-final-neon-t1-stable/`.

## 2026-07-22: three-pass optimization sweep

Status: completed. Each of the eight planned kernel families received three
measured optimization passes. A pass is an experiment, not an obligation to
land code: candidates that missed correctness tolerances or regressed the
focused benchmark were removed. The final tree contains only the retained
pieces below and preserves the public contract and canonical GGUF bytes.

| family | pass 1 | pass 2 | pass 3 | final decision |
|---|---|---|---|---|
| GGUF SIMD GEMV | direct packed dots for Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS | direct packed dots for IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S | NEON table lookup/dot for IQ4_NL and IQ4_XS | keep all three |
| dense + quantized GEMM | float dense accumulation and direct generic panel accumulation failed tolerances and were rejected | 64-column dense tile rejected; four-row NEON accumulation retained for generic panels | 8-row dense tile and 16-row panel unroll rejected; small IQ4 uses the now-faster canonical GEMV route | keep the original 4x32 dense tile plus pass-2 panel work |
| paged attention + MLA | 16-score blockwise online softmax for advanced paged attention | NEON f32 score dot | vector primitives for plain attention/MLA retained; advanced value-update SIMD rejected | keep online tiles and score/MLA vectorization |
| fused decode projections | paired Q4 gate/up block traversal | paired Q/K RoPE-row traversal was neutral and rejected | shared activation-load pair dot was neutral and rejected | keep pass-1 gate/up fusion; no new QKV speedup claim |
| streaming LM-head | one vocabulary-parallel Q4 argmax region | reuse each decoded Q4 block across hidden rows | 64-token scheduling grain was neutral and reverted | keep passes 1-2 |
| grouped MoE | paired Q4 gate/up block dot | shared float dequantization regressed about 2x; activation-load reuse was also rejected | two-output task tiles were neutral/regressive | keep pass-1 pair dot |
| fused norm-add-quant | four-way f32 sum-of-squares accumulation | 16-code int8 narrowing/store | four-way normalized-maximum accumulation | keep all three |
| FFT + Mamba | complex-f32 FFT and projection fused into recurrent state update | one fork-join for batch-one FFT; f32 Mamba state | inverse-normalization fusion and NEON Mamba state update regressed and were rejected | keep passes 1-2 |

Pass-0 to final retained measurements:

| path | pass 0 ms (1 / 6 threads) | final ms (1 / 6 threads) | result |
|---|---:|---:|---|
| Q4_K/Q5_K/Q6_K GEMV N1024 K4096 | 9.7727-14.7039 / 2.0937-2.9806 | 1.5332-2.1639 / 0.3278-0.4313 | 6.3x-8.7x / 6.4x-8.6x |
| IQ GEMV N1024 K4096 | 8.3503-11.2778 / 1.7567-2.3323 | 0.3137-2.5965 / 0.0890-0.5164 | 4.0x-31.1x / 4.3x-22.5x |
| Q4_K/Q6_K panel GEMM M16 N256 K1024 | 1.3853-1.6778 / 0.3207-0.3802 | 0.9457-1.2375 / 0.2695-0.3599 | 1.36x-1.47x / 1.06x-1.19x |
| streaming LM-head R2 V8192 H1024 | 1.2227 / 0.3927 | 0.8795 / 0.2353 | 1.39x / 1.67x |
| online paged attention B2 HQ8 HKV2 S512 D64 | 0.1935 / 0.0562 | 0.1126 / 0.0457 | 1.72x / 1.23x |
| online MLA B2 H8 S512 DL64 DR16 | materialized 0.2437 / 0.2453 | 0.2033 / 0.0595 | 1.20x / 4.12x |
| grouped MoE R64 E8 K256 I512 | 1.2931 / 0.3810 | 1.2636 / 0.3387 | 1.02x / 1.12x |
| fused RMS-add-int8 R512 H4096 G128 | 1.0311 / 0.2664 | 0.6351 / 0.1804 | 1.62x / 1.48x |
| FFT B1 H2 L1024 | 0.0546 / 0.0569 | 0.0550 / 0.0407 | parity / 1.40x |
| recurrent Mamba B1 H2 L128 D32 | 0.0973 / 0.0714 | 0.0356 / 0.0342 | 2.73x / 2.09x |

The final same-binary independent baselines show 6.83x-45.24x at one thread
and 5.35x-30.41x at six threads for the listed K/IQ GEMVs; 1.42x/1.46x for
streaming LM-head; 1.90x/4.67x for paged attention; 5.02x/7.21x for fused
norm-add-int8; 32.89x/44.92x for FFT; 106.57x/115.19x for recurrent Mamba;
and 1.20x/4.12x for online MLA. Fused projections remain
semantic/materialization wins: the final run
is at parity at one thread and noisy from 0.90x to 1.21x at six threads, so no
new projection speedup is asserted.

- Correctness: final native Release CTest passes 38/38; ASan + UBSan +
  float-cast-overflow passes 37/37; and the x86_64 cross-build, including the
  AVX2/AVX-512 sources, passes 32/32 under Rosetta. Every timed case's embedded
  oracle passed. Final maximum relative errors are 1.19e-5 or lower for K/IQ
  GEMV, 3.39e-4 for paged attention, 3.46e-4 for Mamba, and 3.03e-2 for
  dequantized int8 norm output.
- Hardware: Apple M5 Max, 18 logical/physical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON with DotProd and I8MM.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release.
- Settings: one or six threads, OS-default affinity/frequency, 5 warmups, 30
  timed samples, 10 ms minimum sample. Final target CV range is
  0.0047-0.1260 at one thread and 0.0160-0.1600 at six threads.
- Command: `quixicore_cpu_bench --preset quick --kernel
  qgemv_formats,prerequisites,optimization_plan,ported_ops --threads {1,6}
  --warmup 5 --iters 30 --min-sample-ms 10`.
- Working-tree label: `1d3dc15-dirty`.
- Raw results: `perf/results/2026-07-22/opt3-pass0-t{1,6}/`,
  `opt3-pass1-fixed-t{1,6}/`, `opt3-pass2-fixed-t{1,6}/`,
  `opt3-pass3-fixed-t{1,6}/`, and `opt3-final-t{1,6}/` beneath the same date
  directory. MLA evidence is in `opt3-mla-final-t{1,6}/`. Rejected pass
  artifacts are retained beside them.

## 2026-07-22: planned P0-P2 CPU kernel batch

Status: implemented and retained with focused correctness, portability, and
performance evidence. Public API semantics and GGUF byte layouts are unchanged;
all packing, tiling, scratch storage, and ISA selection remain internal.

Implementation summary:

- GGUF weight-only GEMV now dispatches Q4_0, Q4_K, Q5_K, Q6_K, IQ4_NL,
  IQ4_XS, IQ2_XXS, IQ2_XS, IQ3_XXS, and IQ1_S through runtime-selected
  blocked/NEON/AVX2/AVX-512 functions. Q4_0 has a direct packed-nibble SIMD
  path; the retained NEON route now also evaluates every listed K/IQ format
  directly from its packed block, including table-lookup SIMD dots for IQ4.
- Dense GEMM uses 4-by-32 output tiling. Prepared quantized GEMM reuses each
  packed weight block across the activation-row tile for every listed GGUF
  format. Q4_0/Q8_0/IQ4 M<64 retains the faster canonical GEMV route; larger
  M and K formats use the panel kernel.
- Basic, advanced, FP8, absorbed-MLA, and paged-attention paths use online
  max/sum recurrence and do not materialize score vectors. Output rescaling is
  only performed when a new running maximum is observed.
- Gate/up+SwiGLU and Q/K/V projections share one scheduling region and consume
  quantized rows directly. The activation route no longer stores the full
  gate/up projection pair. A single-token route projects Q/K/V, rotates Q/K,
  and writes K/V directly to the requested cache slot without K/V tensors.
- Quantized LM-head argmax/top-k scans 4096-vocabulary tiles while retaining
  only selection state. Grouped MoE forms expert row groups and produces
  SwiGLU output directly without a rows-by-2I projection.
- Norm-add-quant is row-local and two-pass, with per-worker group scratch and
  NEON reductions/conversion. It no longer allocates a normalized tensor.
- Power-of-two FFT convolution uses an iterative radix-2 complex transform;
  short or non-power-of-two lengths keep the direct oracle. Mamba2 carries the
  recurrent D-by-D state instead of expanding every prior source position.

Correctness and portability:

- Native Release: 38/38 CTest cases pass, including benchmark smoke.
- ASan + UBSan + float-cast-overflow: 37/37 CTest cases pass.
- x86_64 cross-build: AVX2 and AVX-512 GGUF sources compile and 32/32 Rosetta
  CTest cases pass.
- Focused tests cover forced GGUF reference dispatch, all new packed formats,
  M16/M65 panel thresholds, FFT L64 versus direct convolution, multi-step
  Mamba2 versus source expansion, fused projection, norm/quantization,
  fused QKV+RoPE+KV insertion (including bounds), grouped MoE, and online
  attention/MLA entry points.

Measured decisions on Apple M5 Max:

| path | 1-thread candidate / baseline ms | speedup | 6-thread candidate / baseline ms | speedup | decision |
|---|---:|---:|---:|---:|---|
| Q4_0 GEMV N4096 K4096 | 1.1953 / 2.8578 | 2.39x | 0.2798 / 0.5892 | 2.11x | keep direct SIMD decode |
| K/IQ GEMV N1024 K4096 | 8.4295-14.6360 / 13.8664-15.1367 | 1.03x-1.64x | 1.7739-3.1180 / 2.4908-3.6649 | 1.05x-1.40x | keep block decode + SIMD dot |
| Q4_0 packed GEMM M128 N1024 K1408 | 7.4150 / 13.6738 | 1.84x | 1.6954 / 4.9766 | 2.94x | keep M>=64 panel route |
| Q4_K/Q6_K/IQ4_XS GEMM M16 | 1.4007-1.6641 / 9.8004-14.1906 | 6.97x-8.53x | 0.3313-0.3883 / 2.4374-3.5107 | 6.75x-9.04x | keep panel route |
| blocked dense M16 N512 K512 | 0.3336 / 2.9337 | 8.80x | 0.1005 / 3.0997 | 30.85x | keep tiled/parallel route |
| fused Q4_0 SwiGLU N1024 K1408 | 0.2107 / 0.2186 | 1.04x | 0.0687 / 0.0896 | 1.31x | keep fusion; removes intermediate |
| fused QKV+RoPE+KV HQ8 HKV2 D64 K1408 | 0.0801 / 0.0815 | 1.02x | 0.0391 / 0.0397 | 1.02x | keep fusion; removes Q/K/V staging and cache pass |
| streaming LM-head R2 V8192 H1024 | 1.2631 / 1.3157 | 1.04x | 0.4444 / 0.4067 | 0.92x | keep bounded-memory route; no speedup claim |
| grouped MoE R64 E8 K256 I512 | 1.3072 / 1.2815 | 0.98x | 0.4526 / 1.7029 | 3.76x | keep grouping for threaded batches |
| online paged attention B2 HQ8 HKV2 S512 D64 | 0.1942 / 0.2118 | 1.09x | 0.0753 / 0.2233 | 2.96x | keep one-pass online recurrence |
| fused RMS-add-int8 R512 H4096 G128 | 1.0248 / 3.4495 | 3.37x | 0.2545 / 1.3100 | 5.15x | keep row-local NEON route |
| FFT convolution B1 H2 L1024 | 0.0553 / 1.8173 | 32.85x | 0.0608 / 1.8128 | 29.79x | keep radix-2 route |
| recurrent Mamba2 B1 H2 L128 D32 | 0.0986 / 3.7921 | 38.45x | 0.0740 / 3.9839 | 53.83x | keep recurrent state |

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON with DotProd and I8MM.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release.
- Settings: one or six threads, OS-default affinity/frequency, 5 warmups, 30
  timed samples. The consolidated optimization batch uses a 20 ms minimum
  sample; GGUF, packed-GEMM, and norm sweeps use 5 ms.
- Commands: `quixicore_cpu_bench --preset quick --kernel
  {optimization_plan,qgemv_formats,prerequisites,ported_ops} --threads {1,6}
  --warmup 5 --iters 30 --min-sample-ms {20,5}`.
- Working-tree label: `1970aa1-dirty`.
- Raw results: `perf/results/2026-07-22/optimization-plan-complete-t{1,6}/`,
  `qgemv-allformats-final-t{1,6}/`, `prepacked-allformats-final2-t{1,6}/`, and
  `norm-fused-neon-t{1,6}/` beneath the same date directory.

## 2026-07-22: CPU packed-panel and workspace prerequisites

Status: retained CPU infrastructure and first measured consumer.

Current implementation: `CpuPackedWeights` preserves an owned copy of the
canonical QuixiCore/GGUF bytes and derives a private 64-byte-aligned row panel.
The automatic panel width is 4 rows on NEON, 8 on AVX2, 16 on AVX-512, and 1
for the portable fallback. `Workspace` is a segmented, pointer-stable arena;
caller-owned and persistent thread-local paths both retain capacity after a
frame rewinds. The first consumer, Q4_0/Q8_0 `qgemm_prepacked`, transposes the
M activations once and reuses each packed weight block across the M tile. The
fused norm/quantization and paired-projection paths now use retained internal
scratch instead of per-call vectors.

Correctness: the focused tests force every panel width and every registered
packed block geometry, verify byte placement, zero padding, 64-byte alignment,
canonical-byte preservation, Q4_0/Q8_0 output for M1/M3/M17 at one and four
threads, workspace pointer stability, and capacity reuse. The benchmark output
is exact versus canonical Q4_0 QGEMM. Native Release CTest passes 37/37,
including benchmark smoke; ASan + UBSan + float-cast-overflow passes 36/36;
the x86_64 AVX2/AVX-512 cross-build and Rosetta test run pass 31/31.

Baseline: canonical-layout `qgemm`, which dispatches one QGEMV per activation
row. Candidate: prepared Q4_0 QGEMM with panel construction outside timing and
a warmed caller-owned workspace.

| threads | candidate median ms (CV; p20-p80) | baseline median ms | speedup | decision |
|---:|---:|---:|---:|---|
| 1 | 2.021854 (0.0569; 1.970958-2.135056) | 4.027229 | 1.99x | keep panel reuse |
| 6 | 0.602766 (0.0356; 0.587861-0.626130) | 1.738412 | 2.88x | keep panel reuse and one fork-join region |

The existing R512 H4096 fused RMSNorm-add/int8 case was rerun after replacing
its vector with retained thread-local scratch: 3.350604 ms (CV 0.0687) at one
thread and 1.295057 ms (CV 0.0349) at six threads. Both outputs remain exact;
the result is consistent with the prior 3.272490/1.290578 ms run, so this
change claims allocation removal but no norm speedup.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON with DotProd and I8MM.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release.
- Settings: one or six threads, OS-default affinity/frequency, 5 warmups,
  30 timed samples, 5 ms minimum sample time.
- Command: `quixicore_cpu_bench --preset quick --kernel prerequisites
  --threads {1,6} --warmup 5 --iters 30 --min-sample-ms 5`; fused-path rerun
  substitutes `--kernel ported_ops`.
- Working-tree label: `33ad02b-dirty`.
- Raw results: `perf/results/2026-07-22/prerequisites-t{1,6}/` and
  `perf/results/2026-07-22/workspace-ported-t{1,6}/`.

## 2026-07-22: Colibri CPU algorithm excavation

Status: candidate kernels with focused aarch64 evidence.

Current implementation: Colibri's runner-local algorithms are exposed as CPU
kernels rather than copied as model orchestration. The batch adds row-scaled
W8A32 GEMM; prequantized W8A8 integer GEMM; int2, int3-g64, row/group int4,
dynamic W4A8, paired projection, packed-row, and E8/IQ3 conversion/compute
operations; adjacent-to-split RoPE; heap nucleus sampling; threshold top-k;
quantized dense/sparse MLA weight absorption; and stable MoE expert-batch
union. Portable references remain available. Runtime routes cover NEON,
DotProd, I8MM, AVX2, AVX-512, and VNNI where the operation and build target
support them. The exact source-to-kernel inventory and non-kernel exclusions
are in `docs/colibri-port-matrix.md`.

Correctness: focused tests cover layouts, pack/unpack, scalar-versus-ISA
routes, asymmetric int8 correction, deterministic sampling/ties, in-place
RoPE, dense/sparse MLA oracles, MoE grouping, and invalid inputs. E8/IQ3
packing additionally matches a fixed 98-byte oracle emitted by Colibri's
Python encoder. Native Release CTest passes 35/35. ASan + UBSan +
float-cast-overflow CTest passes 35/35. The x86_64 cross-build compiles every
new AVX2/AVX-512/VNNI file and its full Rosetta CTest run passes 29/29.

Baseline: every benchmark case carries a same-binary independent scalar,
full-sort, or dispatched-per-row baseline. Candidate: the public CPU route.
Both use identical inputs and correctness is checked before timing.

| case | 1-thread candidate / baseline ms (CV) | speedup | 6-thread candidate / baseline ms (CV) | speedup | decision |
|---|---:|---:|---:|---:|---|
| W8A32 M4 N1024 K1408 | 0.5601 / 3.0125 (0.0238) | 5.38x | 0.1645 / 0.5505 (0.0416) | 3.35x | keep NEON/runtime route |
| prequantized int8 IDOT M4 N1024 K1408 | 0.0663 / 1.1570 (0.0301) | 17.45x | 0.0357 / 0.2743 (0.0567) | 7.68x | keep DotProd/I8MM dispatch |
| int4 f32 M4 N1024 K1408 | 0.6231 / 4.9532 (0.0271) | 7.95x | 0.1879 / 0.8327 (0.0138) | 4.43x | keep NEON/runtime route |
| dynamic W4A8 M4 N1024 K1408 | 0.0966 / 0.4502 (0.0448) | 4.66x | 0.0502 / 0.1656 (0.0535) | 3.30x | keep quantize-once IDOT route |
| heap top-p V65536 | 1.8725 / 4.6414 (0.0630) | 2.48x | 1.5431 / 4.1210 (0.0478) | 2.67x | keep partial heap extraction |
| threshold top-k R16 W8192 K2048 | 1.0982 / 2.2536 (0.0293) | 2.05x | 0.3230 / 1.9589 (0.0342) | 6.06x | keep threshold partition + row parallelism |
| MoE union R32 E8 K256 N512 | 0.9235 / 0.8339 (0.1501) | 0.90x | 0.6481 / 0.8369 (0.0242) | 1.29x | use union only for shared experts with multiple threads |

The MoE one-thread candidate and baseline execute the same dispatched per-row
GEMV path after the retained guard; their 10% difference is measurement noise
with CV 0.1501, not a regression claim. An initial always-union experiment was
about 3x slower at one thread and was rejected. The final route preserves the
normal GEMV path for one-thread or all-unique routing and enables weight reuse
only for repeated experts with a live worker pool.

Decision: keep. The numeric and selection routes produce material, repeatable
speedups against their exact baselines. Keep the guarded MoE union because its
six-thread repeated-expert case is 1.29x faster and its one-thread route falls
back. The portable E8/IQ3, MLA, RoPE, paired-projection, and row primitives are
correctness candidates; this run does not claim a performance tier for those
unmeasured shapes.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON with DotProd and I8MM.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release.
- Settings: one or six threads, OS-default affinity/frequency, 5 warmups,
  30 timed samples, 5 ms minimum sample time.
- Command: `quixicore_cpu_bench --preset quick --kernel colibri_ops --threads
  {1,6} --warmup 5 --iters 30 --min-sample-ms 5`.
- Working-tree label: `33ad02b-dirty`.
- Raw results: `perf/results/2026-07-22/colibri-port-final-t{1,6}/`.

## 2026-07-22: MXFP8/NVFP4 and final sibling entry points

Status: candidate portable references.

Current implementation: the final sibling audit adds logical-layout MXFP8 and
NVFP4 producers/GEMMs, raw E2M1 packing, split-layout MXFP4/NVFP4 GEMV,
FP8-output attention-state merging, named FP8/WNA16/NVFP4 MoE paths, explicit
attention stages, serving metadata adapters, and the remaining named
projection/norm/SSD routes. Accelerator-only scale swizzles are intentionally
replaced by row-major CPU scale tables. No optimized dtype or ISA tier is
claimed.

Optimization experiment: the first MXFP8 GEMM decoded every E4M3 operand with
the generic arithmetic converter. The retained candidate uses one immutable
256-entry E4M3 decode table; accumulation and public semantics are unchanged.
The benchmark compares the quantized path with the same matrices decoded once
and passed to `dense_gemm`.

| threads | direct decode ms (CV; p20-p80) | lookup decode ms (CV; p20-p80) | predecoded dense ms | lookup speedup |
|---:|---:|---:|---:|---:|
| 1 | 1.668820 (0.0322; 1.616028-1.672236) | 0.460790 (0.0012; 0.460216-0.461152) | 0.275678 | 3.62x |
| 6 | 0.365973 (0.0081; 0.363771-0.368289) | 0.122948 (0.0126; 0.121838-0.123981) | 0.085398 | 2.98x |

Correctness: the MXFP8 benchmark agrees exactly with GEMM over independently
dequantized operands. Focused tests additionally reconstruct NVFP4 GEMM,
MXFP4/NVFP4 GEMV, FP4 packing, all three quantized MoE layouts, and FP8 state
merge. Release, ASan+UBSan+float-cast-overflow, and x86_64/Rosetta CTest each
pass 19/19.

Decision: keep the lookup decoder. It removes most portable conversion cost
but remains 1.67x/1.44x slower than predecoded dense at one/six threads, so no
performance improvement over dense GEMM is claimed. Native SIMD decode and
packed microkernels remain future optimization work.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release.
- Shape/format: M16 N128 K256, 32-value MX groups, E4M3 codes, logical E8M0
  power-of-two scales, f32 output.
- Settings: one or six threads, OS-default affinity/frequency, 5 warmups,
  30 timed samples, 5 ms minimum sample time.
- Command: `quixicore_cpu_bench --preset quick --kernel ported_ops --threads
  {1,6} --warmup 5 --iters 30 --min-sample-ms 5`.
- Working-tree label: `d1343ce-dirty`.
- Raw results: `perf/results/2026-07-22/sibling-entrypoints-{final,lookup}-t{1,6}/`.

## 2026-07-22: extended sibling semantic port completion

Status: candidate.

Current implementation: the final sibling inventory adds portable reference
semantics for the extended norm/attention/quantization/serving/MoE/SSM/vision
paths and host-reference collectives. The focused measured path is fused
RMSNorm-add followed by dynamic signed int8 group quantization. It uses the
public `rms_norm_add_quant_int8` entry point and the R512 H4096 shape with
128-value quantization groups.

Correctness: the benchmark's fused result is checked against public
`rms_norm_add` plus `quantize_int8` with preallocated intermediate storage.
Codes, scales, and residual output agree exactly (`max_abs_err=0`). The full
Release suite passes 18/18; ASan + UBSan + float-cast-overflow passes 18/18;
the x86_64 cross-build and full Rosetta test run pass 18/18. Sanitizer
initially caught two undersized test buffers; the corrected fixtures now
encode the documented `[rows,groups]` scale and `[rows,intermediate]` output
shapes and remain as regression coverage.

Baseline: the decomposed operations with a persistent intermediate buffer.
Candidate in this historical run: the fused public operation before its
temporary moved to the reusable workspace.

| threads | candidate median ms (CV; p20-p80) | baseline median ms | candidate / baseline | decision |
|---:|---:|---:|---:|---|
| 1 | 3.272490 (0.0791; 3.250896-3.354709) | 3.232729 | 1.012x | keep semantic entry; allocation cost is inside variance |
| 6 | 1.290578 (0.0239; 1.258834-1.316208) | 1.265828 | 1.020x | keep; 2.54x over the one-thread candidate, no fusion speedup claim |

Decision: keep. The public composition supplies the sibling semantics with a
small, measured allocation cost versus a caller-managed intermediate. No
performance improvement or optimized tier is claimed. The prerequisite run
above removes that allocation and records the post-change measurement.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release.
- Settings: one or six threads, OS-default affinity/frequency, 5 warmups,
  30 timed samples, 5 ms minimum sample time.
- Command: `quixicore_cpu_bench --preset quick --kernel ported_ops --threads
  {1,6} --warmup 5 --iters 30 --min-sample-ms 5`.
- Working-tree label: `c47957e-dirty`.
- Raw results: `perf/results/2026-07-22/ported-ops-final-{t1,t6}/`.

## 2026-07-22: sibling semantic port batch

Status: candidate.

Current implementation: portable f32 reference entry points now cover each of
the 16 active umbrella v0.1 families. The batch adds activations, softmax,
LayerNorm, dense and grouped GEMM, RoPE, dense/GQA/paged/MLA attention,
sampling, beam/speculative decode, serving utilities, linear attention, Mamba
S6 scan, MoE routing, AdamW, and a q8_0 QGEMM/LM-head composition. Existing
q8_0 GEMV and RMSNorm routes remain in place. GPU tiling variants were
collapsed into shared operation semantics; only f32 and q8_0 are in scope.

Current public route: `include/quixicore_cpu/ops.h`, `qgemv.h`, `qgemm.h`, and
`rms_norm.h`. Independent portable implementations live under `kernels/` and
parallelize disjoint rows/heads/channels through the existing thread pool.

References inspected: umbrella `registry/kernels.yaml`,
`registry/quant-formats.yaml`, `registry/benchmark-shapes.yaml`,
`registry/tolerances.yaml`, all `matrices/` and kernel/format specs; sibling
Metal, XPU, CUDA, and ROCm manifests and public operation surfaces.

Correctness: Release CTest 10/10; ASan + UBSan + float-cast-overflow CTest
10/10; x86_64 cross-build and full Rosetta CTest 10/10. The operation suite has
independent known-value or float64 checks for every new public entry point.
Thread-pool resize/startup generation ordering was fixed after sanitizer
stress exposed one intermittent early-completion failure; a 32-resize
regression test and 100 repeated sanitizer threading runs pass.

Baseline: each benchmark case carries an independent scalar or decomposed
baseline. Candidate: the public CPU route. Both were measured in the same
Release binary. The six-thread run is the stable performance-core-sized run on
this host; attempted eight-thread repeats had CV above 0.10 on several cases
and are retained as raw diagnostics but excluded from the decision table.

| case | 1-thread target ms (CV) | scalar/decomposed ms | 6-thread target ms (CV) | vs 1 thread | vs baseline at 6 threads | decision |
|---|---:|---:|---:|---:|---:|---|
| softmax R512 H4096 | 2.571875 (0.0450) | 3.487917 | 0.549761 (0.0463) | 4.68x | 6.33x | keep row parallelism |
| causal attention H8 S128 D64 | 2.570625 (0.0156) | 2.677979 | 0.555919 (0.0993) | 4.62x | 4.79x | keep head/query parallelism; CV is at threshold |
| MoE routing T1024 E64 K4 | 0.156679 (0.0936) | 0.827618 | 0.151962 (0.0506) | 1.03x | 5.46x | keep partial-sort implementation; intentionally serial |
| Mamba scan C256 S512 N16 | 3.560417 (0.0172) | 3.586625 | 0.773006 (0.0550) | 4.61x | 4.57x | keep channel parallelism |
| q8_0 QGEMM M16 N2048 K2048 | 3.815875 (0.0161) | 31.739709 | 1.377552 (0.0968) | 2.77x | 22.94x | keep composition over the existing dispatched q8_0 GEMV |

Experiments: portable scalar/reference implementations were the correctness
baseline. Disjoint outer-dimension partitioning was retained for softmax,
attention, scan, and q8_0 QGEMM. MoE routing remains serial because the tested
operation is already 0.15 ms and pool overhead did not produce a useful win.
The MoE comparison includes equivalent stable sorting and selected-expert
softmax work.

Decision: keep the semantic ports and threaded candidates. This establishes
correctness and representative performance evidence, not blanket support for
every dtype, quant format, sibling-only fusion, or performance tier. Families
remain conservatively unclaimed in backend metadata until operation-level
benchmark coverage is complete.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON with runtime DotProd/I8MM detection.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release.
- Settings: one or six threads, OS-default affinity/frequency; t1 uses 3
  warmups/20 samples/2 ms minimum; t6 uses 5 warmups/30 samples/5 ms minimum.
- Commands: `quixicore_cpu_bench --preset quick --kernel contract_ops
  --threads 1 --warmup 3 --iters 20 --min-sample-ms 2` and the six-thread
  equivalent with `--warmup 5 --iters 30 --min-sample-ms 5`.
- Working-tree label: `4a95ead-dirty`.
- Raw results: `perf/results/2026-07-22/all-kernels-final-t1/` and
  `perf/results/2026-07-22/all-kernels-final-t6/`.

## 2026-07-21: CPU kernel correctness hardening

Status: candidate.

Current implementation:

- Corrected f32-to-fp16 subnormal rounding so valid q8_0 subnormal scales do
  not collapse to zero.
- Made q8_0 packing reject non-finite inputs before integer conversion and
  added checked shape/packed-size arithmetic plus pointer validation.
- Removed activation-quantizing `dotprod_i8` from public qgemv dispatch,
  including the environment override; it remains an internal benchmark for a
  future separately contracted `qgemv_w8a8` operation.
- Made RMSNorm recompute its reduction in f64 only when the normal NEON f32
  sum is non-finite or subnormal, and normalized before applying the weight to
  avoid otherwise avoidable intermediate overflow.
- Changed benchmark correctness from passive error reporting to a finite,
  elementwise `atol + rtol * |reference|` gate that fails before timing.

Contract references inspected: umbrella `registry/kernels.yaml`,
`registry/quant-formats.yaml`, `registry/benchmark-shapes.yaml`,
`registry/tolerances.yaml`, and `matrices/`; CPU `perf/perf.md`.

Correctness: Release CTest 9/9; ASan + UBSan + float-cast-overflow CTest 9/9;
x86_64 cross-architecture compilation completed. Coverage includes exact fp16
subnormal encodings, q8_0 NaN/Inf rejection, low-amplitude nonzero packing,
overflowing dimensions, forced `dotprod_i8` fallback, extreme/tiny RMSNorm
inputs, and benchmark-gate NaN rejection.

Baseline: pre-change Release binary from the same `4a95ead-dirty` working-tree
line. Candidate: post-change Release build. Both use public entry points with
one thread. The repeat RMSNorm pair was run consecutively to reduce scheduler
drift. Medians and CV follow; deltas are candidate versus baseline.

| case | baseline ms (CV) | candidate ms (CV) | delta |
|---|---:|---:|---:|
| qgemv N4096 K4096 | 0.989646 (0.0796) | 0.969306 (0.0132) | -2.06% |
| qgemv N8192 K8192 | 3.783417 (0.0116) | 3.804854 (0.0117) | +0.57% |
| qgemv N16384 K4096 | 3.898792 (0.0180) | 3.913396 (0.0115) | +0.37% |
| RMSNorm R1 H2048, repeat | 0.000280 (0.0500) | 0.000267 (0.0489) | -4.48% |
| RMSNorm R1 H4096, repeat | 0.000532 (0.0349) | 0.000504 (0.0200) | -5.14% |
| RMSNorm R4 H4096, repeat | 0.002006 (0.0993) | 0.001991 (0.0179) | -0.78% |
| RMSNorm R512 H4096, repeat | 0.258800 (0.0195) | 0.260141 (0.0299) | +0.52% |

Experiments: the first RMSNorm candidate used f64 division and square root on
every row and regressed the R512 stress case by about 5.7%; rejected. The kept
version preserves the original f32 fast path and enters f64 only for exceptional
reductions. All candidate benchmark oracles passed; qgemv maximum absolute
error was at most `9.98e-6` and RMSNorm at most `4.02e-7`.

Decision: keep. The largest stable-shape candidate regressions are 0.57% for
qgemv and 0.52% for RMSNorm, both inside run variance; no performance
improvement is claimed. This entry supersedes the historical 2026-07-07
`dotprod_i8` environment-override routing note.

- Hardware: Apple M5 Max, 18 logical/physical cores, 128 GB; aarch64 NEON.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release.
- Settings: one thread, OS-default affinity/frequency, 3 warmups, 20 timed
  samples, 2 ms minimum sample time.
- Command: `scripts/bench --preset quick --kernel qgemv,rms_norm --threads 1
  --warmup 3 --iters 20 --min-sample-ms 2`, plus consecutive RMSNorm-only
  repeats with the same timing settings.
- Working-tree label: `4a95ead-dirty`.
- Raw results: `perf/results/2026-07-21/review-baseline/`,
  `review-candidate-final/`, `review-rms-baseline-repeat/`, and
  `review-rms-candidate-repeat/` under the same date directory.

## 2026-07-07: qgemv contract realignment with Metal/CUDA

Status: landed.
Current implementation: cross-backend review (QuixiCore-Metal
`dequant.metal` + `tk/quant.py`, QuixiCore-CUDA `quant_formats.cuh`) showed
the family `qgemv` contract is `out = dequantize(wq) @ x` with
full-precision activations and f32 accumulation; activation-quantized
integer math is a separate `qgemv_w8a8` op in both siblings. The CPU
backend had made its activation-quantizing dotprod path the default —
contract-divergent numerics at the public boundary. Realigned:
- Public API renamed `quant_gemv*` -> `qgemv*` (family op naming; the
  umbrella family key `quant_gemv` is the registry label, not the op).
- New `kernels/quantization/qgemv_neon.cpp` ("neon"): int8 weights widened
  to f32 in registers, FMA against f32 activations, per-block scale — the
  contract numerics, now the aarch64 default.
- The SDOT path renamed `dotprod_i8`, excluded from auto-selection
  (`QUIXICORE_CPU_QGEMV_VARIANT=dotprod_i8` only); it previews a future
  `qgemv_w8a8` twin op with family-matching per-token scales.
- `rms_norm` verified formula-identical to both siblings (eps inside sqrt
  on mean(x^2), multiplicative weight); default eps=1e-5 added. q8_0
  block bytes verified identical to Metal/CUDA (34 B, fp16 scale, RNE).
Current public route: `quixicore_cpu::qgemv(...)`; resolves `neon` on
aarch64, `ref` elsewhere.
References inspected: QuixiCore-Metal `qgemv.metal`/`qgemv.cpp`/`quant.py`;
QuixiCore-CUDA `qgemv.cu`/`quant_formats.cuh`/`quant_rt.cu`.
Correctness: `tests/correctness/test_qgemv.cpp` — the public entry now
meets the family oracle (float64 dequantize(wq) @ x) at < 1e-4 on every
platform (measured 1.4e-08 - 2.9e-08 in-harness); dotprod_i8 keeps its own
quantized-activation oracle; public==dispatched-variant bit-exact. 7/7
suites green.
Baseline: `ref` scalar (same build).
Experiments: single candidate (neon f32-activation structure). Result:
3.95-4.02x over ref; 17 W-GB/s single-thread. The dotprod_i8 baseline
remains ~3.5x faster (0.29 ms vs 1.03 ms at 4096^2) — that speed is
intentionally not reachable via default dispatch until it ships under the
family's `qgemv_w8a8` semantics.
Decision: keep neon as the contract default. Accepting the default-path
slowdown buys cross-backend numerical parity: one oracle, one contract,
six backends.
Open questions: `qgemv_w8a8` twin op (per-token activation scales, RNE,
matching Metal/CUDA); i8mm for qgemm; quant.py byte-parity fixtures as
shared test vectors.
Raw results: `perf/results/2026-07-07/033244-quick/`.

- Hardware: Apple M4 Max, 16 logical cores (12P+4E), 128 GB; macOS 26.5.1.
- Toolchain: Apple Clang 21.0.0 (clang-2100.1.1.101), CMake Release,
  baseline arch flags (NEON is aarch64 baseline).
- Command: `scripts/bench --preset quick --kernel qgemv --threads 1`.
- Working-tree label: `96b2c37-dirty`.

| shape (m=1) | neon ms | CV | vs ref | vs dotprod_i8 | W-GB/s | rel err (family oracle) | decision |
|---|---:|---:|---:|---:|---:|---|---|
| N4096 K4096 | 1.0344 | 0.027 | 3.99x | 0.28x | 17.2 | 2.92e-08 | keep as contract default |
| N8192 K8192 | 4.1164 | 0.022 | 4.02x | 0.29x | 17.3 | 1.43e-08 | keep as contract default |
| N16384 K4096 | 4.2517 | 0.016 | 3.95x | 0.28x | 16.8 | 2.92e-08 | keep as contract default |

## 2026-07-07: Threading layer — row-partitioned kernels on a fork-join pool

Status: landed.
Current implementation: `src/threading/thread_pool.{h,cpp}` — persistent
fork-join workers, deterministic contiguous-range partition, inline
execution for small counts / nesting / `num_threads()==1`, no per-call
allocation (fn-pointer + context trampoline). Public control:
`quixicore_cpu::set_num_threads()` (default 1). `quant_gemv` (both
variants) and `rms_norm` (both variants) partition rows; the harness
`--threads` flag now drives the pool, and `mem_triad` runs on it too so
one probe measures single-thread and aggregate rooflines.
Current public route: unchanged APIs; thread count via
`include/quixicore_cpu/threading.h`.
References inspected: llama.cpp/ggml threadpool chunking conventions.
Correctness: `tests/unit/test_threading.cpp` — exact range coverage,
small-count inlining, nested-call inline fallback, and bit-exact kernel
outputs at 1 vs 4 threads for quant_gemv and rms_norm (rows are never
split, so results are identical at any thread count). Full suite 7/7.
Baseline: 1-thread numbers from the same build (see table).
Experiments:
1. First wrapper used `std::function` + capture-heavy lambdas: measured
   1.6-1.9x SINGLE-THREAD regression on rms_norm (loop bounds/pointers
   reloaded through the capture frame every iteration — stores through the
   output pointer may alias the frame). Rejected; replaced with the
   fn-pointer trampoline and free-function loop bodies with by-value
   arguments. Single-thread numbers recovered (qgemv 0.303 ms, R512
   0.284 ms; residual ~80 ns/call on microsecond cases from the control
   mutex — accepted, decode-sized rms_norm calls stay inline anyway).
2. Row-partitioned execution at 8 and 12 threads (12P+4E hybrid; no
   pinning, macOS QoS bias only). Kept.
Decision: keep. Threaded q8_0 GEMV reaches 263-266 W-GB/s, matching the
aggregate DRAM roofline the threaded triad measures (251-304 GB/s) — the
kernel is bandwidth-saturated; more threads cannot help until memory does.
Decode-latency shapes (R1-R4) correctly stay on the inline path.
Open questions: 8 vs 12 threads is shape-dependent on the hybrid part
(E-core drag vs extra bandwidth); revisit with affinity pinning. NUMA
policy deferred until multi-socket hardware exists.
Raw results: `perf/results/2026-07-07/030723-quick/` (1t),
`030745-quick/` (8t), `030751-quick/` (12t).

- Hardware: Apple M4 Max, 16 logical cores (12P+4E), 128 GB; macOS 26.5.1.
- Toolchain: Apple Clang 21.0.0 (clang-2100.1.1.101), CMake Release.
- Command: `scripts/bench --preset quick --kernel qgemv,mem_triad,rms_norm
  --threads {1,8,12}`.
- Working-tree label: `a5c080a-dirty`.

| case | 1t ms | 8t ms | 8t speedup | 12t ms | 12t speedup | best BW | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| qgemv N4096 K4096 | 0.3029 | 0.0679 | 4.46x | 0.0820 | 3.69x | 263 W-GB/s | keep |
| qgemv N8192 K8192 | 1.1962 | 0.3166 | 3.78x | 0.2678 | 4.47x | 266 W-GB/s | keep |
| qgemv N16384 K4096 | 1.1749 | 0.3056 | 3.84x | 0.2696 | 4.36x | 264 W-GB/s | keep |
| rms_norm R512 H4096 | 0.2840 | 0.0663 | 4.28x | 0.0754 | 3.77x | 253 GB/s | keep |
| mem_triad ws_192MiB (roofline) | 1.72-2.24 | 0.8022 | — | 0.6632 | — | 251-304 GB/s | reference |
| mem_triad ws_24MiB (roofline) | 0.2480 | 0.0645 | — | 0.0685 | — | 368-390 GB/s | reference |

## 2026-07-07: rms_norm f32 reference + NEON variant

Status: landed.
Current implementation: `kernels/norms/rms_norm_ref.cpp` (scalar, float64
sum-of-squares accumulation — a near-oracle reference) and
`kernels/norms/rms_norm_neon.cpp` (f32 four-accumulator sum of squares +
vectorized scale pass; NEON is baseline on aarch64, no extra build flags).
Current public route: `quixicore_cpu::rms_norm` via
`src/dispatch/rms_norm.cpp`; resolves `neon` on aarch64, `ref` elsewhere;
`QUIXICORE_CPU_RMS_NORM_VARIANT` override.
References inspected: ggml rms_norm (float64-accumulating scalar
reference precedent).
Correctness: `tests/correctness/test_rms_norm.cpp` — float64 oracle at
umbrella fp32 tolerance (public and neon < 1e-5, measured ~2e-7; ref
< 1e-6), shapes with vector tails (H1, H7, H777), zero-row finiteness,
public-vs-variant bit-exact, determinism (norms family policy).
Baseline: `ref` scalar via direct call (the `ref_scalar` bench baseline).
Experiments: single candidate (NEON f32 structure above). Result:
4.34-4.58x over ref on all shapes. The ref pays for scalar float64
accumulation; the NEON f32 pairwise-style sums keep error at ~2e-7 while
vectorizing fully.
Decision: keep. Decode shapes run at 94-97 GB/s (cache-resident, vs
~246 GB/s in-cache triad); R512xH4096 at 63.8 GB/s.
Open questions: batch>1 threading once the thread pool exists; fused
residual-add variant if the contract adds it.
Raw results: `perf/results/2026-07-07/024347-quick/` (git-ignored; table
below).

- Hardware: Apple M4 Max, 16 logical cores (12P+4E), 128 GB; macOS 26.5.1.
- Toolchain: Apple Clang 21.0.0 (clang-2100.1.1.101), CMake Release,
  baseline arch flags throughout (NEON is aarch64 baseline).
- Command: `scripts/bench --preset quick --kernel rms_norm` (warmup 3,
  iters 20, min_sample_ms 2.0, threads 1).
- Working-tree label: `1eb4984-dirty`.

| shape | neon us | CV | vs ref | GB/s | rel err | decision |
|---|---:|---:|---:|---:|---|---|
| R1 H2048 | 0.25 | 0.046 | 4.43x | 97.1 | 7.5e-08 | keep |
| R1 H4096 | 0.52 | 0.056 | 4.34x | 94.4 | 1.3e-07 | keep |
| R4 H4096 | 2.03 | 0.031 | 4.40x | 72.7 | 1.2e-07 | keep |
| R512 H4096 | 263.33 | 0.052 | 4.58x | 63.8 | 2.2e-07 | keep |

## 2026-07-07: quant_gemv q8_0 NEON DotProd variant

Status: landed.
Current implementation: `kernels/quantization/qgemv_dotprod.cpp`
(`q8_0_gemv_dotprod`): activations quantized to int8 blocks once per call
(d = amax/127, round-to-nearest, grow-only thread-local scratch), then per
32-element block two `vdotq_s32` int8x16 dot products accumulated as
`float32x4` lanes with the combined weight*activation scale
(`vmlaq_n_f32`), two independent vector accumulators per row, scalar tail
for odd block counts. Compiled with `-march=armv8.2-a+dotprod` via
`quixicore_cpu_add_isa_sources()`; baseline build untouched.
Current public route: `quixicore_cpu::quant_gemv` dispatch resolves
`dotprod` when `cpu_features().dotprod` is true, else `ref`;
`QUIXICORE_CPU_QGEMV_VARIANT` still forces either.
References inspected: llama.cpp `ggml_vec_dot_q8_0_q8_0` NEON structure
(activation quantization + sdot + per-block scale fmla).
Correctness: `test_quant_gemv` extended — public entry bit-exact vs the
direct variant call; tight float64 oracle over dequantized weights AND
dequantized quantized activations < 1e-5 (int dot is exact; only f32
scale/accumulate rounds); contract oracle vs original f32 weights < 3e-2
on every shape (measured 2.8e-3 - 6.5e-3); bit-exact determinism. The
f32-activation oracle does not apply to this variant (activation
quantization error dominates on tiny outputs) and is asserted only for
`ref`.
Baseline: `ref` scalar via the same public route (previous entry).
Experiments: single candidate (the structure above). Result: 14.35-14.45x
over `ref` on all three shapes; 43.98-44.61x over the decomposed
dequant+sgemv path.
Decision: keep. 59.3-59.4 effective weight-GB/s = 51% of the same
machine's mem_triad single-thread DRAM roofline (~115-117 GB/s); the
remaining gap is per-call activation quantization plus non-streaming
access, to be revisited with i8mm/threading, not with more scalar work.
Open questions: i8mm (smmla) variant; multi-thread row partitioning; x86
VNNI equivalent blocked on hardware.
Raw results: `perf/results/2026-07-07/023619-quick/` (git-ignored; table
below).

- Hardware: Apple M4 Max, 16 logical cores (12P+4E), 128 GB; macOS 26.5.1.
- Toolchain: Apple Clang 21.0.0 (clang-2100.1.1.101), CMake Release; this
  TU `-march=armv8.2-a+dotprod`, everything else baseline flags.
- Command: `scripts/bench --preset quick --kernel qgemv` (warmup 3,
  iters 20, min_sample_ms 2.0, threads 1).
- Working-tree label: `1092837-dirty`.

| shape (m=1) | dotprod ms | CV | vs ref | vs dequant_sgemv | W-GB/s | GFLOP/s | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| N4096 K4096 | 0.3006 | 0.020 | 14.35x | 43.98x | 59.3 | 111.6 | keep |
| N8192 K8192 | 1.2010 | 0.018 | 14.45x | 44.61x | 59.4 | 111.8 | keep |
| N16384 K4096 | 1.2012 | 0.025 | 14.45x | 44.15x | 59.4 | 111.7 | keep |
| 2026-07-07 | harness (system probes) | f32 | mem_triad ladder + quant_matmul m=1 | aarch64 baseline flags | n/a | n/a | harness validated; no kernel, no speedup claim | `perf/results/2026-07-07/021238-quick/` |

## 2026-07-07: quant_gemv q8_0 scalar reference bring-up

Status: landed.
Current implementation: `kernels/quantization/qgemv_ref.cpp`
(`q8_0_gemv_ref`): plain per-block loop, f32 accumulation, single
accumulator, scale applied once per 32-element block. GGUF-compatible
`q8_0` packing (`34` bytes / 32 weights, fp16 scale + int8).
Current public route: `quixicore_cpu::quant_gemv(QuantFormat::kQ8_0, ...)`
via `src/dispatch/quant_gemv.cpp` (variant `ref`; env override
`QUIXICORE_CPU_QGEMV_VARIANT`).
References inspected: llama.cpp q8_0 block layout and quantization
semantics; QuixiCore-Metal qgemv bench conventions.
Correctness: `tests/correctness/test_quant_gemv.cpp` — fp16 roundtrip,
argument validation, pack/unpack bound (< 1% of amax), float64 oracles
(vs dequantized weights < 1e-4; vs original weights < 3e-2 = umbrella
quantized tolerance), bit-exact determinism. In-harness oracle max rel err
2.4e-07 - 2.6e-07 on the measured shapes.
Baseline: plain single-accumulator loop (shipped as `ref`).
Experiments: candidate = manual 4-way multi-accumulator split (hypothesis:
f32 accumulation chain is latency-bound like `sgemv_naive`). Result: the
candidate measured 1-3% SLOWER on every shape — Apple clang already
auto-vectorizes the plain int8→f32 loop, and the manual split obstructs it.
Decision: reject the multi-accumulator candidate; keep the plain loop as
the reference. The rejected variant is preserved as the `scalar_multiacc`
bench baseline for reproducibility.
Open questions: NEON/dotprod variant is the obvious next step — the ref
runs at ~4.1 weight-GB/s against a ~115 GB/s single-thread DRAM roofline
(mem_triad, same machine), ~28x headroom.
Raw results: `perf/results/2026-07-07/022305-quick/` (git-ignored; table
below). Earlier A/B with candidate as target:
`perf/results/2026-07-07/022143-quick/`.

- Hardware: Apple M4 Max, 16 logical cores (12P+4E), 128 GB; macOS 26.5.1.
- Toolchain: Apple Clang 21.0.0 (clang-2100.1.1.101), CMake Release,
  baseline arch flags.
- Command: `scripts/bench --preset quick --kernel qgemv` (warmup 3,
  iters 20, min_sample_ms 2.0, threads 1).
- Working-tree label: `cb47aff-dirty`.

| shape (m=1) | ref ms | CV | vs scalar_multiacc | vs dequant_sgemv | W-GB/s | GFLOP/s | rel err | decision |
|---|---:|---:|---:|---:|---:|---:|---|---|
| N4096 K4096 | 4.3189 | 0.014 | 1.03x | 3.05x | 4.1 | 7.8 | 2.39e-07 | keep plain loop |
| N8192 K8192 | 17.2801 | 0.022 | 1.01x | 3.09x | 4.1 | 7.8 | 2.55e-07 | keep plain loop |
| N16384 K4096 | 17.1908 | 0.011 | 1.03x | 3.08x | 4.1 | 7.8 | 2.39e-07 | keep plain loop |

## 2026-07-07: Benchmark harness bring-up (system probes only)

Status: landed.
Current implementation: `quixicore_cpu_bench` harness with two
system-characterization probes (`mem_triad`, `sgemv_naive`). No contract
kernel exists or is claimed; no speedup is claimed.
Current public route: n/a (probes call harness-local loops, not library
kernels).
References inspected: QuixiCore-Metal `perf/bench_kernels.py` (timing
discipline, row schema, presets), umbrella `docs/benchmarking.md` and
`registry/benchmark-shapes.yaml`.
Correctness: in-harness float64 oracle per case. mem_triad max rel err
0.00e+00 (exact in f32 for the chosen fills); sgemv_naive max rel err
1.7e-06 - 2.6e-06 across shapes, within fp32 contract tolerance
(rtol 1e-5).
Baseline: this run establishes the machine reference points below; there is
no prior measurement to compare against and no comparison is made.
Experiments: none (bring-up only).
Decision: harness validated end-to-end (correctness oracle, adaptive-batch
timing, JSONL/run.json/summary outputs, gitignored results, ctest smoke on
all CI platforms). No kernel implemented, no performance claim.
Open questions: affinity pinning and multi-threaded probes deferred until
threaded kernels exist.
Raw results: `perf/results/2026-07-07/021238-quick/` (git-ignored; summary
below).

- Hardware: Apple M4 Max, 16 logical cores (12P+4E), 128 GB; macOS 26.5.1.
- Toolchain: Apple Clang 21.0.0 (clang-2100.1.1.101), CMake Release,
  baseline arch flags (no `-march=native`).
- Command: `scripts/bench --preset quick` (warmup 3, iters 20,
  min_sample_ms 2.0, threads 1, QoS user-interactive).
- Working-tree label: `0220bc1-dirty`.

| case | variant | median ms | CV | GB/s | GFLOP/s | max rel err |
|---|---|---:|---:|---:|---:|---|
| mem_triad | ws_96KiB | 0.0004 | 0.071 | 246.3 | | 0.00e+00 |
| mem_triad | ws_1536KiB | 0.0132 | 0.176 | 118.9 | | 0.00e+00 |
| mem_triad | ws_24MiB | 0.2146 | 0.059 | 117.3 | | 0.00e+00 |
| mem_triad | ws_192MiB | 1.7480 | 0.060 | 115.2 | | 0.00e+00 |
| sgemv_naive | N2048_K2048 | 2.0521 | 0.040 | 8.2 | 4.1 | 1.72e-06 |
| sgemv_naive | N4096_K4096 | 8.7347 | 0.016 | 7.7 | 3.8 | 1.79e-06 |
| sgemv_naive | N8192_K8192 | 35.7706 | 0.028 | 7.5 | 3.8 | 2.44e-06 |
| sgemv_naive | N16384_K4096 | 34.9894 | 0.012 | 7.7 | 3.8 | 2.61e-06 |

Reading (context, not a claim): single-thread DRAM triad plateaus at
~115-117 GB/s; the naive sgemv runs at ~7.7 GB/s effective, an order of
magnitude under the roofline because strict-FP scalar accumulation is
latency-bound. That gap is the headroom future GEMV work is judged
against.

## 2026-07-22: q4_0 weight-only and q4_0/q8_0 W8A8 GEMV

Status: candidate.

Current implementation: q4_0 packing, unpacking, and full-f32-activation GEMV
now use the public `qgemv` API alongside q8_0. The separate `qgemv_w8a8` API
supports both q4_0 and q8_0 weights with per-32-element int8 activation
quantization. Portable scalar references anchor every combination; q8_0 W8A8
selects the existing aarch64 DotProd kernel when runtime support is present.
The weight-only qgemv dispatcher never selects activation-quantized numerics.

Current public route: `include/quixicore_cpu/qgemv.h` and
`qgemv_w8a8.h`; dispatch in `src/dispatch/qgemv.cpp` and
`qgemv_w8a8.cpp`; portable kernels and GGUF-compatible q4_0 layout in
`kernels/quantization/qgemv_w8a8_ref.cpp`.

References inspected: sibling q4_0/q8_0 layouts and W8A8 operation separation;
umbrella `registry/quant-formats.yaml`, `registry/benchmark-shapes.yaml`,
`registry/tolerances.yaml`, and `specs/kernels/quantization.md`.

Correctness: Release CTest 11/11; ASan + UBSan + float-cast-overflow CTest
11/11; x86_64 cross-build and full Rosetta CTest 11/11. Weight-only q4_0 is
checked against float64 accumulation over exactly dequantized weights. W8A8 is
checked against the exact semantic oracle
`dequant(W) @ dequant(blockwise_int8(x))`, plus format/shape/non-finite-input
validation, determinism, and bit-identical one-versus-four-thread output.

During the optimization run, moving W8A8 activation scratch to `thread_local`
storage initially made worker lambdas resolve each worker's empty TLS instance;
the six-thread benchmark crashed and ASan identified the null read. The kept
implementation snapshots the caller's scratch data pointers before entering
the pool. The sanitizer benchmark reproduction and multithread regression test
pass after the fix.

Baseline: direct portable reference in the same Release binary, plus a
dequantize-then-scalar-GEMV baseline. Candidate: the public route. The stable
six-thread performance-core-sized run is recorded below; all correctness gates
passed and all target CVs are below 0.10.

| case (N4096 K4096) | candidate ms (CV) | direct ref ms | dequant scalar ms | speedup vs ref | W-GB/s | decision |
|---|---:|---:|---:|---:|---:|---|
| q4_0 weight-only qgemv | 1.048475 (0.0358) | 1.072847 | 13.960396 | 1.02x | 9.0 | keep portable correctness anchor; no ISA speedup claim |
| q4_0 W8A8 qgemv | 0.858071 (0.0200) | 0.866313 | 13.762375 | 1.01x | 11.0 | keep portable correctness anchor; no ISA speedup claim |
| q8_0 W8A8 qgemv | 0.138599 (0.0852) | 0.739384 | 14.674480 | 5.33x | 128.6 | keep aarch64 DotProd dispatch |

Decision: keep all three public routes and the q8_0 DotProd selection. q4_0
remains a portable reference path until a separately measured NEON/AVX
implementation beats it. No unsupported GGUF format or cross-machine speedup
is claimed.

- Hardware: Apple M5 Max, 18 physical/logical cores (6 performance, 12
  efficiency), 128 GB; aarch64 NEON + DotProd.
- OS/toolchain: macOS 26.5.2; Apple Clang 21.0.0
  (`clang-2100.1.1.101`); CMake Release.
- Settings: six threads, OS-default affinity/frequency, 5 warmups, 30 samples,
  5 ms minimum sample time.
- Command: `quixicore_cpu_bench --preset quick --kernel qgemv_formats
  --threads 6 --warmup 5 --iters 30 --min-sample-ms 5`.
- Working-tree label: `15a78dc-dirty`.
- Raw results:
  `perf/results/2026-07-22/qgemv-formats-final-t6-fixed/`.

## 2026-07-22: qgemv_w8a8 q4_0 SDOT (dotprod_i8)

Status: landed; correctness validated; on-hardware perf pending in the
maintainer link environment.

New variant: `q4_0_gemv_w8a8_dotprod` — the aarch64 DotProd (SDOT) SIMD variant
of the q4_0-weight x int8-activation GEMV, completing the q4_0 lane of
`qgemv_w8a8` (which previously ran only the scalar `q4_0_gemv_w8a8_ref` while the
q8_0 lane already had its `dotprod` variant). Ported from embeddinggemma.c's
proven `ei_vec_dot_q4_0_q8_0_neon` (`src/quants.c`) and structured exactly like
the existing `q8_0_gemv_dotprod`: activations quantized once per call to
per-32-block int8 (shared, `thread_local` scratch), each 32-weight block
nibble-unpacked (`vandq_u8`/`vshrq_n_u8`, `-8` offset via `vsubq_s8`) and dotted
with `vdotq_s32`, lane-wise f32 accumulation with the combined fp16 scales,
rows partitioned across the thread pool. Selected last-wins by CPU feature via a
new `kQ4_0Variants` table mirroring `kQ8_0Variants`; forced with
`QUIXICORE_CPU_QGEMV_W8A8_VARIANT=dotprod`.

Correctness:
- Standalone runtime check (NEON variant vs the scalar reference on the same
  packed weights + per-block-quantized activations): **max rel diff 1.28e-6**
  (pure fp summation-order; the int8 dot is exact). The intrinsics are correct.
- fp64-oracle CTest wired: `qgemv_w8a8_forced_dotprod` re-runs
  `test_qgemv_w8a8` with the variant forced to `dotprod`, so the SDOT path is
  checked against `dequant(wq) @ x` at the umbrella quantized tolerance.
- Full library compiles clean (`cmake --build --target quixicore_cpu`, dotprod
  ISA source built, no warnings).

Measurement note (honesty gate): this environment cannot LINK the test
executables (the pre-existing libc++ ABI mismatch that also hits
`test_cpu_features`), so no CTest run or perf number was produced here. No
speedup is claimed. The q8_0 SDOT precedent (14.4x over scalar, ~51% of DRAM
roofline) suggests a comparable SDOT-bound win, but that is NOT a claim until
measured. To close the perf gate in the maintainer env (Apple Clang 21):

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
    ctest --test-dir build -R 'qgemv_w8a8'   # incl. qgemv_w8a8_forced_dotprod
    scripts/bench --preset quick             # q4_0 w8a8 dotprod vs ref/scalar

Follow-up: AVX2/VNNI x86 variant of the same q4_0 w8a8 dot (QuixiCore-CPU still
has no x86 SIMD in the quant family).
