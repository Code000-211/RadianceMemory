# libr3d

libr3d is radiance's kernel library for RDNA3 and RDNA3.5, the gfx11xx family: `gfx1100`,
`gfx1101`, `gfx1102`, `gfx1103`, `gfx1150`, `gfx1151`, `gfx1152`, `gfx1153`, and `gfx11-generic`,
the code object for a gfx11 card the build did not name. It is HIP kernels behind a plain C ABI
([`r3d.h`](r3d.h)) plus
the kernel-plugin rows the engine selects from (`r3d_rows.cpp`); `libr3d/CMakeLists.txt` names
every unit and its flags.

It claims `^gfx11` where [`libr4d`](../libr4d/README.md) claims `^gfx12`; a card gets one device
library, chosen by the rows, not the name. `rad_plugin_open` declines a machine whose KFD
topology lists no gfx11 card (a process without `/dev/kfd` sees none), so in an installation that
also carries libr4d the tools, which resolve kernels without the engine's code-object check, never
lay an RDNA4 container's weights out for libr3d. The kernels are not ports of libr4d's: gfx11 has no
bf16 pack and no VOPD, and a different integer instruction set, so every arithmetic path is its
own. The matrix work splits by dtype: the bf16/f16 GEMM rows (the m64 band and the prefill-tiled
form) and every MoE GEMM row run on the 16x16x16 wave-matrix (WMMA) builtins retiled onto a 16x128
block tile, as do the quantised m64 prefill rows; the integer rows (an int8 activation against
int8/int4/int2 weights) and the one-to-sixteen-row GEMMs run the dot2 path,
`v_dot2acc_f32_f16`, with software codecs for every format. The contract is shared with libr4d:
the same op schemas (copied slot for slot from libref, as the loader requires), the same numerics
rules, the same geometry in constraints. Family parts differ in CU count and clock, never in these
instructions, so one code object per family suffices.

## Naming

An entry point is `r3d_<op>_<dtype>[_<the geometry it is compiled for>]`: `gemm_nt_bf16_m16`
serves `gemm_nt` at bf16 for M <= 16 and refuses a larger M; `r3d_rope_bf16` serves bf16 QKV in
place and refuses anything else. A dimension in a name is a compile-time constant, and the entry
point rejects a mismatch instead of running. Dimensions shared by the whole library stay out of
the names: wave32, the attention rows' head size (128), and the target architecture, which
`rad_plugin_info`'s `build_target` reports, spelled as a device reports its arch, because the
loader warns when a plugin claims a card the running device is not.

Rows are `RadKernelInfo` entries in a table the engine reads from the dlopened `.so`; the plugin
links the ABI headers and nothing of the engine's.

**Schema equality at load.** The first library to declare an op fixes its schema, and a second
library declaring it with a different schema is refused by name, with every row it carries.
Arguments are positional, so a mistyped operand role costs the whole library, not one row.

**The tail order is the ABI's:** `init, fini, launch, scratch, layout, relayout, opd_shape,
unrelayout, replaces`. `replaces` is last on purpose: kernel tables are positional aggregate
initialisers, a field inserted mid-struct shifts every later one, and `nullptr` converts to any
function pointer, so part of such a shift would still compile.

There is no launch adapter: every entry point is the row's `launch`, `(const RadArgs*,
RadStream)`, named `r3d_rad_<op>` in [`r3d_plugin.h`](r3d_plugin.h). Behind it each unit has two
layers. The **launcher** (`r3d_<op>(...)` in `r3d.h`) takes pointers as longs, checks what a
`__global__` cannot, computes the grid and issues one kernel on the given stream. The **shim**,
named by the row, reads operands with [`r3d_args.h`](r3d_args.h), takes the dtype from the tensor
and the row count from the operand, checks strides and returns a RAD status. Neither allocates and
neither synchronises.

The table groups the rows by family; `r3d_rows.cpp` and `r3d.h` hold the full set.

| family | rows |
| --- | --- |
| `norm` | `rmsnorm_bf16` (optional residual add and multiplier), `rmsnorm_f32`, `layernorm_bf16` |
| `elem` | `add` (with `div`, and the broadcast band), `mul`, `silu`, `gelu`, `sigmoid`, `act_mul_{silu,gelu,sigmoid}`, `softmax`, `scale`, `fill` (bf16/f32/i32/i64), `gather_rows`, `scatter_rows`, bf16 and f32 where the op has both, plus seven `cast` pairs and `silu_mul_bf16` |
| `attn` | `rope_bf16` (neox and gptj, in place over packed QKV), `kv_store_{bf16,f16,fp8}`, `attn_paged_{bf16,fp8}_{decode,prefill}`: head_dim 128, split-KV combined in the same launch |
| `kv` | `kv_len_trim_i32` (per-sequence context lengths less `trim`, the QSA slice's own last row), `qsa_tail_store_bf16` (the QSA indexer's tail into a per-layer linear KV state, ring addressed by position) |
| `gemm` | `embed_lookup_bf16`, `gemm_nt_bf16_{m16,tiled}`, `gemm_nt_f16_m16`, `gemm_nt_q_fp8a16_m16`, `gemm_nt_q_fp8a8_{m16,tiled}`, `gemm_nt_q_mxfp4a8_{m16,tiled}` (OCP-MXFP4 weight: e2m1 codes, an e8m0 scale per 32 of K), `gemm_nt_q_w2a8_{m16,tiled}` (2-bit and 4-bit grids on the 128x128 tile), `gemm_nt_q_w4a8_m16` (signed int4 weight, decode band), `gemm_nt_q_w4a16asym_m64` and `gemm_nt_q_w5a16_m64` (4- and 5-bit codes against an f16 activation, 1..64 rows on the wave matrix), `gemm_nt_q_w8a8_{m16,m64}` (signed int8 weight, decode band and 17..127 prefill rows on the wave matrix), `gemm_nt_q_gated_fp8a8` |
| `vocab` | `logits_gemm_bf16`: the head, read once per token, as bf16 or as E4M3 codes with a bf16 scale per 128x128 block; the `impl` axis picks the decode (the LDS table, the per-byte ALU formula) |
| `quant` | `quant_act_fp8_bf16`, `quant_act_i8_bf16`, `quant_act_i8g_bf16`, `had_quant_act_{i8,fp8}_bf16` and `had_quant_act_i8g_bf16` (the same with a block Walsh-Hadamard rotation), `gated_had_quant_{i8,i8g,fp8}_bf16`, `gelu_mul_quant_i8_bf16`, `quant_cols_fp8_bf16`, `rmsnorm_quant_{fp8,i8}_bf16` (norm and quantiser in one pass), `gated_quant_fp8_bf16`, `dequant_fp8`, `dequant_e5m2`, `dequant_i8` |
| `sample` | pre-filters `sample_{penalties,dry,ngram_ban,mask,logprobs}_f32`; spine `sample_{argmax,temp,topk,pick,topp,minp,typical,xtc}_f32` — f32 logits in and out, the draw the reference's in its order, the top-R scan single-pass (the plane read once) |
| `moe` | `expert_scale_bf16` (the renormalised top-k weight times the expert's per-expert scale, in place), `moe_router_topk_bf16` (softmax top-k, renormalised if `norm`), `moe_router_topk_scatter` (the top-k and the counting sort as one op), `moe_router_sigmoid_topk` (the GLM-style router: sigmoid scores, selection on scores + bias, the kept weights scaled by `route_scale`), `moe_scatter_small` (the routed slots' stable counting sort: one workgroup up to 1024 slots, three passes over 256-slot tiles above), `moe_gather` (fold the expert outputs back to the token, with the optional shared arm), `moe_gemm_q_e4m3` (grouped E4M3 GEMM; the `impl` axis: 16x16 table decode, 16x16 ALU decode, 16x128 block with the F32 scales folded), `moe_gemm_bf16` (unquantised experts, wave matrix), `moe_gemm_mxfp4a8` (canonical MXFP4 experts), `moe_gemm_w4a8` (int8 activation, packed two's-complement int4 experts: the i32 wave matrix, an f32 a-scale and a bf16 w-scale per (row, 128 of K), token- and sorted-order `a`), `moe_gemm_w4a4` (int4 activation, packed two's-complement int4 experts: both widen to int8 on the i32 wave matrix, an f32 a-scale and a bf16 w-scale per (row, 128 of K), token- and sorted-order `a`), `row_topk_{bf16,f32}` (the reference files the op under `moe`, so the row does too) |
| `gdn` | `gdn_kkt_solve_c64` (`A = (I + strict_lower(diag(beta) K K^T e^{g_i-g_j}))^-1` per 64-chunk; libhalo's kernel), `gdn_chunk_scan_k128_v128_c64` (the chunked scan: WY recompute, state recurrence and output in one kernel; libhalo's), `gdn_gated_rmsnorm_h128`, `gdn_gated_norm_had_quant_i8` (the gated norm fused with the rotation and quantiser), `gdn_conv_prep_w4_h128` (causal conv, split, l2norm, gating and the per-chunk gate cumsum in one prefill kernel), `gdn_conv_update_w4_h128` (the same convolution at decode over a rolling conv-state buffer), `gdn_recurrent_update_k128_v128` (the decode-time recurrent state update, per_candidate or anchor; libhalo's), `gdn_conv_recurrent_update_k128_v128` (the convolution and the recurrent update as one decode kernel, q/k/v never materialised; libhalo's) |
| `draft` | `dflash_select_bf16`: the drafter's accept step, a walk over the draft block's candidate lattice |

The table holds 111 rows over 72 schemas; `r3d_selftest` checks every schema slot for slot against
the `libref.so` beside the plugin. Fourteen rows carry tune axes (see
[Tunable params](#tunable-params)); the rest are single-implementation.

There are two int8 activation rows because the reference has two ops and they are not
interchangeable. `quant_act_i8` writes one scale per row, with codes in the order an int8 matrix
operand is read (`i8_step_pos`, bits 2 and 3 of the column swapped); that is what
`w8a8`/`w4a8`/`w2a8` consumers expect. `quant_act_i8g` writes one scale per (row, group) with
row-major codes; that is what this library's `i8a8` GEMM reads and `dequant` inverts. The step
order belongs to the op: the oracle compares the plane element by element, and it is the one place
in libr3d where a store address carries a permutation.

## Tunable params

A row's tune axes are `RAD_P_INT` parameters the engine's `rad-tune` walks and the selftest
checks (each default among its ascending values). An axis value the kernel cannot serve is
refused by the launcher, not clamped. `rad-tune` merges its findings into the tune cache keyed by
machine, so a row's shipped number is the tuned default for the part; `rad-kbench --all-rows`
measures the arms a tuned default never reaches.

| axis | rows | what it selects |
| --- | --- | --- |
| `impl` | `gemm_nt_bf16_m16` / `gemm_nt_f16_m16`, `gemm_nt_q_fp8a16_m16` / `gemm_nt_q_fp8a8_m16`, `gemm_nt_q_mxfp4a8_{m16,tiled}`, `gemm_nt_q_w8a8_m16`, `gemm_nt_q_fp8a8_tiled`, `logits_gemm_bf16`, `gemm_nt_bf16_tiled`, `moe_gemm_q_e4m3` | the kernel arm. On the m16 rows: the dot2 pairs vs the 16x16x16 wave matrix. On `logits_gemm_bf16`: the LDS table decode vs the per-byte ALU decode. On `gemm_nt_bf16_tiled` the four arms are the fdot2 tile, the same tile with the ALU decode, the 128x128 wave matrix (the compute-dual, above) and the 16-column-tile m64 kernel. On `moe_gemm_q_e4m3`: the 16x16 table-decode tile, the 16x16 ALU-decode tile, the 16x128 block with the F32 scales folded |
| `mrows` | `moe_gemm_bf16` | the block's row count (16 / 32 / 64). The compiled points are the legal ones (the `moe_bf16_tune_valid` table); the rest refuse |
| `vgpr` | `moe_gemm_bf16`, `moe_gemm_q_e4m3` (arm 2 only) | the per-lane register budget the kernel is compiled at — the occupancy knob: on the 1024-VGPR/SIMD file, 128 is the 8-wave cap and 512 the natural allocation; each budget is a separate kernel emission, and the win is height-specific (extra waves only pay on latency-bound shapes, where spills can cost more than they buy) |
| `acc` | `moe_gemm_bf16`, `moe_gemm_q_e4m3` (arm 2 only), `gemm_nt_q_w4a16asym_m64` / `gemm_nt_q_w5a16_m64` | 0 = f32 accumulate (the default); 1 = the 16-bit-accumulate WMMA form. A precision knob, not a speed one: no throughput gain, a documented low-precision mode (the [wave-matrix builtins](#what-the-family-provides-and-what-it-lacks)) |
| `sk` | the m16 and m64 GEMM rows | the split-K count |
| `wv` | the m16 and m64 GEMM rows | waves per workgroup |
| `nt` | the m16 and m64 GEMM rows | the streaming-load form of the weight fetch |
| `mb` | the m64 GEMM rows | the m-block (row-tile) count per workgroup |
| `npw` | the m64 GEMM rows | n-per-wave (columns per wave) |

The `vgpr` and `acc` axes are arm 2's on `moe_gemm_q_e4m3` because only the 16x128 block is
compiled in more than one register budget and in both accumulator forms; the 16x16 tiles are
single-variant.

## Building

Out of tree, against an installed radiance, the way a third-party plugin is built:

```
cmake -S libr3d -B build -G Ninja \
      -DCMAKE_PREFIX_PATH="$RADIANCE_PREFIX" \
      -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
      -DRAD_GPU_TARGETS="gfx1100;gfx11-generic"
cmake --build build
```

`rad_plugin_gpu_targets(R3D_GPU_TARGETS "^gfx11")` filters the requested targets to the family,
and the project returns without a target when the filter is empty, so a build for gfx12 machines
never produces this library. `gfx11-generic` is the object for a gfx11 card the build did not
name, which is how one `.so` serves the family. In tree, `add_subdirectory(libr3d)` sits beside
libr4d's under the same `RAD_HIP_FOUND` guard.

`R3D_UNITS` lists the units as `"file|extra flags"`, each named after the entry point it provides;
the shared headers (`r3d_common.h`, `r3d_codecs.h`, `r3d_args.h`, `r3d_launch.h`, `r3d_tiling.h`,
`r3d_fp8_dequant.h`, `r3d_fwht.h`) provide no kernel. Every unit is compiled `-ffp-contract=off`,
which `rad_hipify` puts on all HIP sources; it is a correctness flag (see [Numerics](#numerics)).
`r3d_selftest.cpp` is a host-only CMake target that dlopens the built `.so`; it needs no HIP
toolchain to compile, so a build machine runs its static phases without a card.

## Adding a kernel

1. A unit named after its entry point, `r3d_<op>[_<geometry>].hip`, in `R3D_UNITS`.
2. The launcher in [`r3d.h`](r3d.h) and the shim in the unit, both under `extern "C"`.
3. The row in `r3d_rows.cpp`: the geometry, the dtypes, the constraints, and the tune axes only
   if the kernel has them. A geometry the kernel cannot serve is a constraint, not a check.
4. The selftest fixture for the row (the `Case` builder in `r3d_selftest.cpp`), with a geometry
   to the selftest's list.
5. The row runs with no new test code: the selftest walks the table, so a new row runs with no new
   test code; its geometry comes from the row's own `opd_shape`.
6. Measure it: `r3d_selftest <home>/kernels/libr3d.so --case <row>`, then
   `rad-kbench --home "$RADIANCE_HOME" --kernels libref:libr3d --op <op>`.

The constraints are the row's declaration of what it serves, read by the engine as a single source
of truth (the selftest checks every key against the op's declared parameters). Three rules: a
constraint on a key the band does not carry narrows the row to the graphs that state it (bind an
optional parameter and the row matches nothing for the graphs that omit it); a band's `dtype` is
the activation's, so a row whose dtype is a weight property (`logits_gemm`) constrains no dtype;
and a row's floor is its sibling's declared ceiling + 1 (the tiled rows open at `M >= 17` because
the m16 row declares `C_LE("M", 16)`). Hardware limits the kernel cannot violate (`n_vocab >= 4`,
`n_cand <= 1024`, `K % 32`/`K % 128`) go in the constraints too, checked at launch.

## What the family provides, and what it lacks

**The wave-matrix builtins.** All of them are wave32, 16x16x16: f16, bf16, int8 and int4 inputs,
accumulating in f32, f16, bf16, i32. The fragment layout is `idx = lane % 16`,
`k = 8*(e>>2) + 4*(lane>>4) + (e&3)`; both A and B want K-contiguous rows, so swapping the
operands of a WMMA transposes the result. The 16-bit-accumulate forms
(`v_wmma_f16_16x16x16_f16`, `v_wmma_bf16_16x16x16_bf16`) exist as selectable variants on the
quantised m64 rows (w4a16/w5a16) and both MoE wave-matrix rows, the tune's `acc` axis; they are a
documented low-precision mode (the f32 form is the default) — the precision floor and the two
codegen traps are in [Numerics](#numerics).

Every feature RDNA3 lacks is emulated in software:

| absent on gfx11 | software substitute |
| --- | --- |
| int8 dot (`s32sd4`, `s32ud4`, `__dp4a`, `madocx`) | widen codes to f16, dot on `dot4_f16x4` — ≈22-bit mantissa on integer operands, so the w2a8 rows carry a documented one-ulp tail |
| fp8 convert (`v_cvt_pk_f16_fp8`) | e4m3 bit maths in [`r3d_codecs.h`](r3d_codecs.h); the tiled rows' decode is a 256-entry f16 LDS table, the `impl` arms a per-byte ALU formula |
| bf16 pack (`v_cvt_pk_bf16_f32`, RDNA4) | two adds plus a `v_perm_b32` |
| vector f64 | soft-float at 1:64 (sets the sampler's draw path, the [Numerics](#numerics) divergence table) |
| transposed loads, VOPD (`global_load_tr_b128`, `ds_read_b64_tr_b16`) | transposes in registers |

**LLVM single-buffers LDS** and drains before a dependent read unless told otherwise: the dot2
rows have nothing to overlap a load with, so the plain barrier is the one they need, while the
wave-matrix rows double-buffer the k tile (the next tile stages while the current one multiplies).

## The GEMM rows

The GEMM rows follow a **dual-variant principle**: most families ship two implementations the
tune picks per geometry, along two independent axes.

| axis | variant 0 | variant 1 |
| --- | --- | --- |
| **decode** (E4M3 weight) | the 256-entry f16 LDS table | the per-byte ALU formula |
| **compute** (the multiply pipe) | the ALU `fdot2`/`dot2` lane-per-element path | the 16x16x16 wave matrix |

The compute axis is the **compute-dual**, and it is the one that matters at prefill. On
`gemm_nt_bf16_tiled` (the 128x128 tile) the two arms are:

| | arm 0 (ALU) | arm 2 (wave matrix) |
| --- | --- | --- |
| multiply pipe | VALU: f16 FMA / `v_dot2acc_f32_f16` | the 16x16x16 wave matrix |
| accumulators | 8x8 f32 a lane | 64 f32 a wave (eight 16x16 tiles) |
| k order | 32-code chunks | 32-code chunks |
| data movement | load, stage, barrier, store — **identical** | load, stage, barrier, store — **identical** |

The two are duals in the literal sense: they share the entire data-movement skeleton and differ
only in the pipe that performs the multiply, so a number measured on one is a bound on the other's
data movement. The arm that wins is the one on the faster pipe for the shape's mix: the f16 matrix
core runs about twice the ALU FMA rate (512 vs ~256 FLOPS/clk/CU), so at prefill heights, where the
tile is compute-bound, the WMMA twin is the faster, and at decode heights, where the row is
latency- and bytes-bound, the two are close and the ALU form's lighter register pressure can win.

The rule the rows follow: **every dot2 GEMM kernel ships a wave-matrix twin the tune can pick** —
the m16 rows carry it as the `impl` axis (dot2 vs 16x16x16 wave matrix), the tiled row as one of
its four arms, and the quantised m64 rows run on the wave matrix directly. The MoE wave-matrix rows
are the same idea at the grouped-GEMM scale: the 16x128 block makes every wave own unique columns,
so no wave re-runs a tile, and the run-walk re-stages the expert's weights per run.

### The 16x128 wave-matrix retile

The retile is 16 rows by 128 columns: eight waves, each wave one 16-column tile on the 16x16x16
wave matrix, the fragment layout lanes 0..15 replicated into 16..31, A one row per lane, B one
column per lane, accumulator row i at lane (i%2)*16+j in VGPR i/2. Against the naive 16x16 tile it
removes the 8x wave redundancy (every wave re-running the same 16 columns), stages the activation
once per 16x128 block, and reads the weight once per 16-column tile.

### The rows by row count (bf16/f16 and the quantised m64)

The GEMM band is covered by row count:

* `gemm_nt_bf16_m16` / `gemm_nt_f16_m16` (M ≤ 16) carry the `impl` axis: 0 = the dot2 pairs,
  1 = the 16x16x16 wave matrix. The two implementations share the row, so the tune can pick per
  shape.
* `gemm_nt_bf16_tiled` serves the prefill heights and the M = 17..127 band; its `impl` axis picks
  the kernel: 0 = 128x128 tiles through LDS with 8x8 f32 accumulators a lane, 1 = the same tile
  with the bf16 decode in registers (the per-byte ALU formula), 2 = the same tile on the wave
  matrix (eight 16x16 tiles a wave, 64 f32 accumulators, k in 32-code chunks -- every wave owns
  unique columns, so no wave re-runs a tile; the compute-dual of arm 0, with the multiply on the
  wave-matrix pipe), 3 = the 16-column-tile wave-matrix kernel for the 17..127 band (MT 16-row
  tiles a block, one 16-column tile a wave across the row tiles, k ascending in 16-wide
  wave-matrix steps; the B-fragment is loaded once a step and reused over the row tiles, so the
  weight stream is read once per 16-column tile; split-K over scratch, the partials reduced in
  split order). M ≤ 16 stays with the m16 rows, whose wave-per-run-of-columns shape is the right
  one there: the two ranges are complements, so neither competes for the other's band.

The quantised m64 prefill rows run on the wave matrix with the same tiling:
`gemm_nt_q_w4a16asym_m64` (1..64 rows; unsigned 4-bit codes, an f16 scale and an integer zero per
128 of K, against an f16 activation) and `gemm_nt_q_w5a16_m64` (1..64 rows; ParoQuant's int5: the
4-bit row with a fifth bit riding a plane of its own, one word a (tile, 64 of K, lane)) carry the
`tGemmM64` tune (`sk`, `wv`, `mb`, `npw`, `nt`, `acc`), and `gemm_nt_q_w8a8_m64` (17..127 rows;
signed int8 weight against int8 activations, the group's integer sub-sum scaled once) is the
symmetric grid's prefill arm. The weights arrive pre-permuted into fragment order at load (the
`layout` hooks), the split-K slabs reduce in the block's dynamic LDS in ascending split order. The
codes are exact in the tile's f16, and the store is the only narrowing.

### The fp8a8 rows

The bands this family targets are 42 layers plus a drafter at `group=128 dtype=fp8a8`, band
ceiling M=2048: N=12288 K=2048 x44 (the gated MLP, two halves of 6144), N=2560 K=2048 x44 (qkv),
N=2048 K=6144 x44 (o-proj), N=2048 K=2048 x44, and N=2048 K=10240 x3 (the drafter). Three rows
cover them by row count, with boundaries read from the constraint tables: `gemm_nt_q_fp8a8_m16`
declares `C_LE("M", 16)`, so `gemm_nt_q_fp8a8_tiled` opens at `C_GE("M", 17)`, and
`gemm_nt_q_gated_fp8a8` serves M=1..64 with N even and K % 128 == 0.

**The m16 row** gives each lane one output element and streams the weight past an activation kept
in registers, with no wave reduction, so a lane's chain is the reference's and the row is
bit-comparable at the store. Above 16 rows the same operands use a 128x128 tile through LDS with
8x8 f32 accumulators per lane. The two ranges are complements, and the m16 row is compared against
the tiled row on the same bytes, where a moved boundary would show.

**The tiled row folds the scales at staging.** The tile is f16, so both scale streams enter as f16
multiplies where the codes are staged: the activation's `a_scale[m][g]` per staged row, the
weight's `b_scale[nb][g]` once per chunk. The products are separable,
`(a*sa)*(b*sb) = (a*b)*sa*sb`, so the accumulator stays one f32 register. The reference's law is
the group sub-sum from raw codes scaled once per group (`ref_gemm.h`'s `quant_dot`); the fold
differs from it by `rel_l2` 2.77e-04 against a bf16 store that costs 3.9e-3, so it is invisible in
the output (see the divergence table).

#### The gated MLP row

`gemm_nt_q_gated_fp8a8` fuses the GEMM, the gate and the output quantiser, so the bf16 `[M, N]`
intermediate is never written, read or allocated. The gate couples column `i` with column `n + i`
(192 waves apart at n = 6144, out of reach of any lane exchange), so each lane computes both
products of its column, walking k ascending with two accumulators over weight rows `i` and `n + i`
— the same traffic the unfused GEMM spends on the two columns. A block is one group of columns by
`256/group` rows, closing the group's amax inside the block (a wave max folded over the group's
waves in LDS, one scale store, one code per lane). The op is one launch with no scratch, so the row
is concurrent. Both halves are narrowed to bf16 before the gate (the pair it replaces stores and
re-reads them); the activation is not — `gt * sg` is f32 from the narrowed gate, and the product
with `up` rounds once, at the store. The law is libref's `ref_gemm_nt_q_gated`, and its order must
be kept.

The launcher picks the arm by row count: fp8a8 at M <= 8 reads the activation and both weight rows
directly, 16 bytes at a time, decodes them from the block's table and multiplies, with no shared
memory between load and dot; M >= 9 (and every fp8a16 case) stage through LDS. The M <= 8 threshold
is a tunable, chosen for decode (M=1) and draft verify (M <= 8).

The row carries no relayout: libref reads the gated `[N, K]` weight row-major, libr4d stores fp8a8
weights in WMMA fragment order, and a row that inherited the fp8 GEMM's `unrelayout` would fail the
per-op oracle. The launcher refuses M > 64 (beyond the bands' ceiling the weight stream is a tiled
kernel's problem), N odd, a group not a power of two in [32, 256], `group > n` (the reference would
use n for the quantiser's group while keeping the weight's K block at the stated width),
`K % group != 0` (a ragged K block reads past the scale plane), `sk > 1` (the epilogue needs the
whole row), and an absent `act`.

### The MoE GEMMs

The grouped GEMM rows share the run-walk / gather structure: the block walks the runs in its
16 rows (a block spanning several experts at decode is correct), each run re-stages that expert's
weights, rows the run does not own are padding and their output is dropped, and a dropped slot
(sorted token < 0) writes zero. `y` is in sorted order.

The experts arrive as weight tables (`RAD_OPD_WTAB`, `r3d_moe_tab.h`): a device array of one
pointer per expert, or a stacked plane, read on the device. `a_order` "token" (gate_up) gathers row
`i` from `a[sorted_tok[i] / top_k]`, "sorted" (down) reads `a[i]`. `moe_gemm_q`'s odd class must
match the even one (`N_odd = N`, `K_odd = K`, as on one card); an uneven class is refused by name,
and `parts` is then the identity. `r3d_selftest` checks each row's values against the reference's
arithmetic on routed batches with dropped slots, an N tail block and both orders.

* `moe_gemm_bf16` — unquantised experts: the bf16 operands stage straight into the bf16 wave-matrix
  fragments, f32 accumulate over the whole of K, no scales, `y` bf16. Each expert is the
  checkpoint's own [N, K] row-major plane (no layout hook). Tunable over `mrows` (the block's row
  count), `vgpr` and `acc`; the compiled points are the legal ones (the `moe_bf16_tune_valid`
  table), the rest refuse.
* `moe_gemm_q_e4m3` — canonical E4M3 experts with F32 block scales, on the `moe_gemm_q`
  contract (`dtype` fp8a8f32): expert e is entry e/2 of the even (`w` / `w_scale`) or odd
  (`w_odd` / `w_odd_scale`) tables; the scales fold in f32 per 128x128 tile; `y` is bf16. The
  `impl` axis (the row's tune, dispatched by `r3d_rad_moe_gemm_q`): 0 = the
  16x16 tile with the 512-byte LDS decode table, 1 = the same tile with the per-byte ALU formula
  (no LDS table), 2 = the 16x128 block with the F32 scales folded in f32 (the `vgpr` and `acc`
  axes are arm 2's). The E4M3 operands widen exactly to f16, the f16 wave matrix (f32 accumulate)
  runs the multiply.
* `moe_gemm_mxfp4a8` — canonical MXFP4 experts: the E4M3 activation and the E2M1 weight (decoded
  exactly to E4M3) upcast to f16, the f16 wave matrix (f32 accumulate), the E8M0 weight scales fold
  in f32 per (row, 32 of K); the even/odd tables as above, `y` bf16.

On the prefill geometry (T = N = K = 4096, 8 experts, best / median of nine launches) the
`mrows`/`vgpr` points of `moe_gemm_bf16` run 41.67 ms at (16, 512), 26.19 ms at (32, 128) and
21.16 ms at (64, 512) — the wider the block's row tile the fewer expert re-stages, so (64, 512)
is the default's best point. The `moe_gemm_q_e4m3` 16x128 arm (vgpr 128) is 215 ms on the same
bytes: decode-bound on the E4M3 widen, which is why that row's tile is 16 rows and its arm 2 is
the one the prefill geometry reaches for.

`moe_gemm_w4a8` — the `moe_gemm_q` row at `dtype=w4a8`: int8 activation, packed two's-complement
int4 experts. The same run-walk structure as the siblings, a 16x128 block (the eight waves do
unique 16-column sub-tiles), the int8 activation and the packed int4 weight (decoded exactly to
int8, the native i32 wave matrix) with the scales folded per 128 of K — an f32 a-scale and a bf16
w-scale. The experts are two tables by parity (even `w`/`w_scale`, odd `w_odd`/`w_odd_scale`), as
every other `moe_gemm_q` dtype; `y` is bf16. Both token- and sorted-order `a`; the int8 arm folds
each 128-block's exact i32 product sum as `(sub * a_scale) * w_scale` in ascending block order, so
it is bit-exact against the libref reference, and the e4m3 arm is the half-precision wave matrix.
The remaining weight forms (the nl64/nl32 magnitude-table nibbles), the prefill run map and the
SK/UW/RS passes are outstanding, per its kernel header.

`moe_gemm_w4a4` — the `moe_gemm_q` row at `dtype=w4a4`: int4 activation (packed two per byte),
packed two's-complement int4 experts. Both operands widen exactly to int8 and run the native i32
wave matrix (no f16 rounding anywhere in the chain); the scales fold per 128 of K — an f32 a-scale
and a bf16 w-scale. The experts are two tables by parity (even `w`/`w_scale`, odd
`w_odd`/`w_odd_scale`); `y` is bf16. Both token- and sorted-order `a`; each 128-block's exact i32
product sum folds as `(sub * a_scale) * w_scale` in ascending block order, so it is bit-exact
against the libref reference (verified 4/4 on gfx1102).

`dequant_w4` — the packed 4-bit weight back to a plain row-major bf16 `[N, K]`, for the prefill
path where W4A8 gains nothing. One thread per packed dword scatters its eight nibbles; the scale
dword holds the f16 scale low and `-(1024+zero)` high, so one routine serves the symmetric
(`twos=1`) and asymmetric (`twos=0`) grids. Verified 2048/2048 on gfx1102.

`row_topk_merge` — the R largest of the all-gathered per-rank candidate sets, the other half of a
vocab-parallel top-R. One wave per row over `world_size * R` (id, value) pairs; a 32-slot top-R
heap per lane with a 5-step xor-butterfly extract. Ties break on the lower id. R <= 32. Verified
16/16 on gfx1102.

`ngram_ids` — the hashed row ids of the Per-Layer Embedding: `mixed(k) = XOR of t_s * mult[s]`
(64-bit wrapping), `ids[h] = mixed % vocab_sizes[h] + offsets[h]`. One thread a token, looping its
heads; n-gram shifts with an EOS walk; ngram <= 8. Verified 24/24 on gfx1102.

`gemm_nt_q_bias` — the narrow fp8 GEMM with a per-column bias epilogue: `y = a @ b^T + bias`.
M <= 16, fp8a8 codes and scales, bf16 bias and output. The bias is fused into the GEMM kernel's
store (no second pass). Verified 64/64 on gfx1102.

`gemm_nt_q_short` — the narrow fp8 GEMM with the short-prompt split rule: unsplit (sk=1) at
N <= 512, K <= 5120, 41..64 rows, where the split's per-slice floors dominate. M <= 64, fp8a8
codes and scales, bf16 output. Verified 256/256 on gfx1102.

`moe_gemm_q_mod4` — the grouped quantised GEMM with 4 expert classes: experts route to class
e%4 (entry e/4 within the class). Same E4M3 wave-matrix core as `moe_gemm_q_e4m3`, 8 individual
`R3dTab` parameters (4 weight + 4 scale tables). Verified 2048/2048 on gfx1102.

### Stored weights: the layout hooks

Six GEMM rows declare a `layout` — `gemm_nt_q_{w2a8_m16,w2a8_tiled,fp8a8_tiled,w4a16asym_m64,
w5a16_m64}` and `logits_gemm_bf16` — each with the `relayout` that produces the form it names;
`r3d_selftest` enforces the pair by name (a `layout` without a `relayout`, a `relayout` without a
`layout`, or any `unrelayout`, fails the table).

A GEMM reads a weight in the order it is stored, and what is stored is decided by the row's
`layout` hook, never by the kernel: **no plugin-private dtype, no repack into a format of this
plugin's invention, no container format changed from inside a kernel library.** The hooks exist
because the container's quantiser stores some weights as separate planes a kernel cannot read in
place (for a head, `fp8_e4m3 * bf16[128x128]`), and the wave-matrix m64 rows want the weight
pre-permuted into fragment order at load. The engine's canonical form for the split planes is the
planes joined behind one operand, `rad.planes`, sized by `rad_planes_plan` and written by
`rad_planes_relayout` at load. The rules:

* **A hook that answers is a promise** — `RAD_OK` means the stored form is described and
  `relayout` will produce it; with `relayout = nullptr` the model does not load.
* **A weight is stored once, so the first row to claim it owns its plan**; a second row gets the
  same answer by calling the first row's hook and reads the container by role, not by the
  arrangement it declared.
* **`RAD_E_UNSUPPORTED` means stored as is**, the only form the mapped tier can read in place; a
  joined weight is a resident weight.
* **A container arrives as `rank = 1`, `shape[0] = bytes`** (the joined form's byte count, not a
  plane); decode it by role on the host, where the header lives.
* **A `zero` plane is refused as `E_DTYPE`, not folded** — `w2a8` is the only scheme with an
  activation zero point, and a row that folded one would compute a number the oracle has no law for.

Gains are read in the dtype the checkpoint stores (bf16, f16, f32, i32 or i64), and the kernel's
arithmetic is the reference's in the reference's order.

## The GDN, QSA, and MoE plumbing rows

The chunk preamble, the chunked scan and the recurrent update are libhalo's kernels (`r3dh_*.hip`,
the `halo_` -> `r3dh_` rename): libr4d's ported to the gfx11 wave matrix, they compile for
`gfx11-generic` unchanged and pass `rad-kbench` against libref on gfx1151 (the kkt solve to
8e-5, the scan to 4e-3, the update to 6e-5, both state forms). They replace the first wave-matrix
ports of the same kernels, which disagreed with libref and whose scan wrote past its output.

`gdn_kkt_solve_c64` runs the gated-delta-net's chunk preamble: the gram
`A = (I + strict_lower(diag(beta) K K^T e^{g_i-g_j}))^-1` per 64-chunk, the gram and the blocked
triangular inverse in one kernel, the fp32 gram never reaching memory. `gdn_chunk_scan_k128_v128_c64`
is the chunked scan: WY recompute, the state recurrence and the output in one kernel, the state in
wave-matrix accumulators across the chunk loop so only q, k, v, A, g, beta, the state cache and o
touch HBM. `gdn_gated_rmsnorm_h128` is the layer's own output norm, and
`gdn_gated_norm_had_quant_i8` is the gated norm fused with a block Walsh-Hadamard rotation and the
per-row int8 quantiser. `gdn_conv_prep_w4_h128` is the causal conv width 4 + silu, the q/k/v split,
the qk l2norm, the gating and the per-chunk gate cumsum in one prefill kernel, and
`gdn_conv_update_w4_h128` / `gdn_recurrent_update_k128_v128` are the same convolution and the
recurrent state update at decode, over a rolling conv-state buffer and a paged f32 or f16 state
cache, per candidate token or as one committed state with a replay scratch slot (`anchor`).
`gdn_conv_recurrent_update_k128_v128` is the two as one kernel: the fold holds q, k and v in
registers, so they are never written and the step is one launch instead of two.

The QSA rows are the indexer's tail: `kv_len_trim_i32` trims each sequence's context lengths by
`trim` (the slice's own last row), and `qsa_tail_store_bf16` writes the QSA indexer's tail into a
per-layer linear KV state, ring addressed by position. The MoE plumbing rows are the router and
the sort: `moe_router_topk_bf16` (softmax top-k, renormalised if `norm`),
`moe_router_sigmoid_topk` (the GLM-style router: sigmoid scores, selection on scores + bias, the
kept weights scaled by `route_scale`), `moe_router_topk_scatter` (the top-k and the counting sort
as one op), `moe_scatter_small` (the routed slots' stable counting sort: up to 1024 slots one
workgroup, histogram in LDS; above, libhalo's three passes, a histogram per 256-slot tile into
scratch, the per-expert scan across tiles and the placement) and `moe_gather` (fold the expert
outputs back to the token,
`y[t,c] = sum_j expert_w[t,j] * y_expert[slot(t,j),c]`, with the optional shared arm; top_k ≤ 64).
`expert_scale_bf16` scales the renormalised top-k weights by the expert's per-expert scale in
place, and `dflash_select_bf16` is the drafter's accept step, a walk over the draft block's
candidate lattice.

## The paged attention, decode and prefill

Two rows, chosen by `q_len`: the decode entry (which takes `splits`) and the prefill entry. The
served geometry is n_head 16 over n_head_kv 2 at head_dim 128 (gqa 8), `block_size 16`, a 4096
context, and an `fp8_e4m3` cache with its descales. One kernel serves decode (q_len 1), speculative
verify (q_len 7) and a prefill chunk (q_len 2048).

The layout is the reference's: `kv_cache[num_blocks, kv_heads, block_size, 2 * head_dim]` (K then V
inside a slot), `block_table[seq, j / block_size]` names the block holding key j (a negative entry
hides its key), and `seqused[s]` is the sequence's total key count including this step's query rows.

The block table is the window the kernel may read: `jcap = min(max_ctx, cols * block_size)`, and the
last reachable index must lie inside the plane. `splits` is a maximum, resolved per sequence from its
own `seqused`; the contract is that `splits = 1` is bit-identical to the reference and `splits > 1`
is not (the last-bit difference is in the [Numerics](#numerics) divergence table).

## Numerics

The contract: accumulate in f32, narrow with IEEE round-to-nearest-even, keep subnormals, and do
not reassociate. `libref` is the oracle.

**The 16-bit-accumulate mode (the `acc` axis) is a documented low-precision mode, off by default.**
The f16 and bf16 WMMA accumulator forms quantise the running sum at the 16-bit accumulator's
precision on every wave-matrix step, and the loss grows with the number of steps. It buys no
throughput (the matrix core runs the same rate on either accumulator), so it is a register-pressure
option the tune can reach for, not a speed one. Integer-valued operands (e.g. +/-1 codes) stay
exact, because every partial sum is an integer below the format's exact range; that is also the
degenerate data AMD's own WMMA verification feeds, and the reason a probe that feeds such data
cannot see the loss. Reading the 16-bit accumulator is a direct value conversion, never a union
bit-cast: the `_tied` constraint ties srcC and the result to one physical register, and a bit-cast
breaks the tie (the compiler returns the stale srcC copy, and the accumulated tile reads back as
zero). A second trap: mixing the f32-acc and 16-bit-acc builtins in one function behind a runtime
if/else zeroes the 16-bit path, so the kernels select the accumulator with `if constexpr`
(separate instantiations), never a runtime branch.

**`-ffp-contract=off` on every HIP source** (`rad_hipify`, library-wide): a fused product-sum
rounds once where the reference rounds twice, so a contracted kernel cannot match it. The flag
also keeps the instruction counts quoted in kernel comments true.

**Accumulation order is part of the contract.** `v_dot2acc_f32_f16` folds two products into the
accumulator and rounds once, so the match cannot be bit-exact: under an ulp per pair, a random walk
of about `sqrt(K/2)` ulps over a K-long dot, about a thousandth of the bf16 a store narrows to and
inside what `rad-kbench` measures. The order must be the reference's: k ascending, one accumulator
per output, no reassociation across lanes. Lane maps are built so no lane has to break it (a group
spanning two lanes folds with one `shfl_xor`, in integers, before any float work).

**Reductions use the reference's tree:** a wave butterfly (`dpp_xor` at 16, 8, 4, 2, 1; the halves
of a 64-lane row read through `permlanex16` first where the reference sums a whole row), then wave
partials in wave order, which gives the same bracketing at any thread count. Activations, rope and
quantisation are the reference's formulas in its order (`1/(1+exp(-x))`,
`0.5*x*(1+erf(x*0.70710678118654752))`, amax/127 with ties to even), from `libref/ref_common.h`.

**Accepted divergences.** Where a row is checked against the law it implements rather than
libref's bytes, the unit that ships it names the divergence:

| divergence | reason and cost |
| --- | --- |
| an index at or past an operand's row count takes the padding path (zero row, no store); `kv_store` drops a slot past the cache's span | the reference validates index arrays on the host and refuses, and `kv_store` refuses such a slot with `E_SHAPE`; a kernel cannot, so the bound travels as an argument (`slot_max`) and a slot is lost rather than a step. A negative index is padding too and is not clamped to zero, which would duplicate row 0 across every padding row |
| rope divides the position by `scale` and evaluates the frequency ladder in double | `scale` is the position scale (the multiplier on a rotated pair is YaRN's magnitude; the shims refuse YaRN, so it is 1). `powf` leaves the reciprocal an ulp off, and at position 262144 an ulp of a frequency is a hundredth of a radian, the one place in that kernel where f32 is not enough |
| `block_sum`'s tree | wave butterfly, then partials in wave order: the reference's bracketing at any thread count, so kernel and reference agree at 256 threads and at 64 |
| `sample_logprobs` keeps a NaN out of the max and in the sum; `sample_mask` skips a row whose `mask_row` is negative | maxima are a strict `>` (`std::max`'s law), so a NaN is never adopted; the sum propagates it, because the reference's `exp(NaN - mx)` is NaN and is added. The k-list is per lane in the reference's order (descending by value, ties to the lower column), merged in the same butterfly; the pool is 64 deep and a larger k is refused, not truncated. The mask bound is the reference's test (`mask_row` is `-1` for no mask), read from the operand, not from M |
| the sampler's `exp` and `log` are `double` calls narrowed once | draws must reproduce: the device's `(float)exp((double)x)` is bit-identical to the host's over the candidate set, where `expf` differs by 1 ulp on a few percent of points. At 1:64 doubles this is the slow path, affordable on the candidate set (1024 exponentials per row) but not in attention |
| attention's `exp` is a degree-8 polynomial; the gated MLP's sigmoid uses `expf` | prefill needs tens of millions of exponentials per layer. Against `(float)exp((double)x)` over a dense grid of [-100, 0], every point is within one f32 ulp and most are bit-identical; the polynomial is the only place an explicit `__fmaf_rn` is allowed, because what must be reproduced is a function. The MLP's sigmoid feeds a bf16 narrowing (ulp 2^-8 against `expf`'s ~1e-7), so the bf16 differs only on a tie |
| `attn_paged`'s split-KV reducer rescales each partial by `exp(m_s - M)` before the sum | every term then arrives as `exp(s - m_s) * exp(m_s - M)` where the reference has `exp(s - M)`, a last-bit difference at f32 (about 2 ulp per term, 1 ulp on the sum) that vanishes under the bf16 store except on rounding boundaries. The row's text states the contract: `splits = 1` is bit-identical to the reference, `splits > 1` is not |

The fp8a8 tiled fold (scales applied at staging, the group sub-sum accumulated from raw codes and
scaled once per group in the reference) differs from the reference's law by a relative 2.77e-04
against a bf16 store that costs 3.9e-3, so it is invisible in the output.

## Testing

[`r3d_selftest.cpp`](r3d_selftest.cpp) dlopens the `.so`, so an undefined symbol is its failure
rather than a link failure, and runs five gates:

* **Codecs** against the ABI's host versions in `rad_plugin.h`: bf16 and f16 over all 65536
  patterns in both directions; e4m3 over all 256 codes plus the encoder's saturating, subnormal,
  signed-zero and NaN cases; int8 against `rad_round_even_sat` over every tie; the widenings over
  the range in which they are exact; the bf16 pair packer against `rad_f32_to_bf16`. kbench compares
  whole ops, where a codec disagreement shows up only as a diffuse percentage, so the codecs are
  tested here.
* **The table:** unique names, a launch on every row, an `init` with a `fini`, the device domain,
  prose on every row, a schema for every `op`, every constraint key a declared parameter, each
  tunable's default among its ascending values, and the layout pair rule.
* **Band reachability:** every constraint a row puts on an optional parameter is listed. It is not
  an error (it is how a row says it serves only the form that states the parameter), but it must be
  a deliberate choice.
* **The schemas against `libref.so`** beside the plugin, with the loader's own comparison
  (parameter count, keys, types, requiredness; operand count, roles, optionality). One mistyped
  role makes the loader refuse the whole library.
* **Every row, run** on a card: buffers from the row's own `opd_shape`, filled per its fill code,
  outputs checked whole, one child process per row so a GPU page fault names its row and the run
  continues. Without a card the static phases still run, so a build machine runs them.

The numerical authority is `rad-kbench`, replaying a recorded fixture against libref:

```
rad-kbench --home "$RADIANCE_HOME" --kernels libref:libr3d
```

`--all-rows` runs every row whose constraints hold, not only the one selection picks, which is how
tunable arms the tuned defaults never reach get measured. `--coverage DIR --coverage-write`
records the rows a fixture reaches, so a row that stops being reachable fails the run; the census
is [`kbench_rows_kernels_libr3d.txt`](../data/kbench_rows/kbench_rows_kernels_libr3d.txt), written
by the same run.

**A fixture covers a checkpoint, not the library.** A fixture is generated from the bands a
recorded checkpoint issues, so a row can be correct and absent from the census: the base fixture
has no `w2a8` band above M=16. Record one fixture per served checkpoint; `rad-kbench` records one
from a model (reading only the tables) and widens every case into the constraints of every row
under test:

```
rad-kbench -m /path/to/served.rad --record "$RADIANCE_HOME/kb-served.rkb"
rad-kbench --fixture "$RADIANCE_HOME/kb-served.rkb" --record-rows "$RADIANCE_HOME/kbrows.rkb" \
           --kernels libr3d,libref
rad-kbench --fixture "$RADIANCE_HOME/kbrows.rkb" --kernels libref:libr3d --all-rows \
           --no-cache --max-cases 2 --max-bytes 1024 --seed 42
```

The first command checks the served checkpoint's own bands against the reference on the
container's real weight bytes. The second and third take each case to the M and `q_len` values no
other harness reaches (the prefill and decode bands on the same bytes) and compare the whole
planes. The gates test a built binary, not the commit: build a clean export of the commit on the
target before trusting the gates for that commit.

## Notes on the kernels

Three properties are assumed throughout and not restated per kernel: **wave32 with a 1024-entry
VGPR file per SIMD**, so every lane map has a lane owning whole dwords and every budget is computed
against 1024; **64 KiB of LDS per workgroup**, which sets the candidate kernels' `n_cand <= 1024`
and the tile sizes; and **the dot2 tile**, shared by the tiled quantised rows in
[`r3d_tiling.h`](r3d_tiling.h) — a dot loop over LDS with an 8-element XOR swizzle. The wave-matrix
rows carry their own tiling (the 16x128 block tile, above).

**Clear the HIP error flag before a launch check.** `hipGetLastError()` reports the first error
since it was last cleared, so a check that does not clear first attributes to this launch whatever
the process raised before it. The symptom is a timing mode returning the device code on every
geometry of every row, disappearing under `HIP_LAUNCH_BLOCKING=1` (a synchronous launch clears a
pending asynchronous error first). Clear the flag after the arguments are staged and before the
launch, read it straight back, and on a refusal name the error and print the grid, block, waves,
splits and width the launcher chose.

**No collective rows and no chained forms.** The parts in this family are single-GPU, and an
all-reduce over the host link most of them sit on is slower than sending the bytes through the
host. Nothing here is built as a chained form, so both optional symbols are omitted, which is the
ABI's own answer.

## What is not here yet

**The seeded draw against a recording.** The fixture checks the plane whole, not the draw against a
recorded seed, over the parameter rows a real request sends. The sampler's draw is reproducible by
construction (the reference's order, the reference's `exp`/`log`), but a recording is the proof,
and there is no recording harness for the plugin's parameter rows.

**The other `w4a8` weight forms.** The `w4a8` form is now a row (`moe_gemm_w4a8`, bit-exact
against the libref reference); the remaining forms (w4nl64a8h / w4nl32a8h / w5nl64a8h, whose
nibbles decode through a magnitude table and a per-64/32 E4M3 scale), the prefill run map and the
SK/UW/RS optimisations are outstanding, per the kernel header.

**A chunked logits path.** The head is the largest read in a decode step, and the sampler reads
f32, so the store is typed, not narrowed, and a 2048-row prefill band writes 1.07 GB. Nothing
emits it in chunks for the sampler to consume in passes.

**A sub-byte zero plane.** `w2a8` reads a zero point one byte at a time and refuses a zero stored
as U2/U4/U5 by name; the bit-extract path would be the same `rad_load_bits` the codes use.

**A draft path that pays.** The draft head is served end to end (its bands by the 2-bit grid, its
accept step by `dflash_select`) and the accepted text is identical. The target's head read
amortises over the draft block, so the drafter's own steps have to pay for themselves, and on the
development part they do not; what is missing is a draft path that costs less than the head read it
saves.
