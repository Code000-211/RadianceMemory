# libr3d serves RDNA3 only, declared here and nowhere else. Its kernels rely on wave32 and on
# instructions RDNA3 has: v_dot2acc_f32_f16 and the 16x16x16 v_wmma wave-matrix family for the
# matrix work, v_cvt_pkrtz_f16_f32, DPP16 and v_permlanex16. The GEMM rows carry two arms each,
# the dot2 pairs and the wave matrix, points of the row's `impl` axis the tune cache picks per
# shape per machine. The kernels do not use gfx12-only instructions such as global_load_tr_b128,
# and they decode every quantised weight in software: the lane map, not a missing instruction, is
# the reason.
#
# A build targeting other cards skips libr3d. Without a device kernel library the engine refuses ops
# only a reference library implements, unless --debug-accept-reference-kernels is passed (spec §17).
# libr4d is the kernel library for gfx12, libr3d for gfx11.
#
# Built in radiance's tree, or standalone against an installed radiance
# (`cmake -S libr3d -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`, naming the HIP compiler as for
# the tree); see cmake/RadAddPlugin.cmake.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(libr3d LANGUAGES CXX HIP)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

# `^gfx11` matches the family: gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151 and
# `gfx11-generic`, as `^gfx12` does for libr4d. The RDNA3 chips differ in CU count and clock, not in
# the instructions these kernels issue, so one code object per family serves them all. A build
# listing both a chip and the family gets both, and the engine prefers the chip's object
# (docs/PLUGIN.md §10).
rad_plugin_gpu_targets(R3D_GPU_TARGETS "^gfx11")
if(NOT R3D_GPU_TARGETS)
  message(STATUS
    "libr3d: RDNA3 (gfx11xx) only, and this build targets ${RAD_GPU_TARGETS} -- not built")
  return()
endif()

# R3D_UNITS lists libr3d's device translation units, as "unit|extra flags".
#
# Every unit is compiled -ffp-contract=off (rad_hipify sets it on all HIP sources). libref
# accumulates in f32 with no FMA and no reassociation (ref_common.h), and a contracted kernel would
# disagree with that oracle. It also keeps the instruction counts quoted in kernel comments true.

# Kernels and the declaration (row table, operand descriptions) share one directory. Every entry
# point takes `(const RadArgs*, RadStream)` and sits beside its kernel. The plugin links rad_abi
# only, like an out-of-tree plugin.

set(R3D_UNITS
  # The model's small ops, one workgroup per row: rmsnorm, layernorm, the binary and unary element
  # ops, the activations, cast, softmax, embed_lookup, gather_rows and scatter_rows, rope and
  # kv_store. This is the set the engine refuses on a card with no device kernel library.
  "r3d_model_bf16|"
  # The sampler's pre-filters: penalties, DRY, the n-gram ban, the bitmask, and the logprobs
  # summary, which reads the logits plane once (the reference reads it three times).
  "r3d_sample_pre_f32|"
  # The speculative decoder's accept step: a walk over the draft block's candidate lattice, which is
  # a gather out of two codebooks rather than arithmetic.
  "r3d_dflash_f32|"
  # The gated MLP band: GEMM, gate and output quantiser in one launch, so the [M,N] intermediate
  # is never written to memory.
  "r3d_gemm_fp8a8_gated|"
  # The vocab projection at decode: U2 codes, int8 activations, a joined scale+zero operand.
  "r3d_gemm_nt_q_w2a8|"
  # The vocabulary projection at both heights: the head streamed for M <= 16, tiled above it.
  "r3d_vocab_bf16|"
  # The tiled GEMM on the wave-matrix: the compute-dual of the ALU fdot2 tile above.
  "r3d_vocab_bf16_wmma|"
  # The quantised GEMM at prefill heights.
  "r3d_gemm_fp8a8_tiled|"
  # The OCP-MXFP4 weight against the fp8 activation, both heights: e2m1 codes with one e8m0 scale
  # per 32 of K, the block's partials scaled at the block boundary.
  "r3d_gemm_mxfp4a8|"
  "r3d_gemm_inta8|"
  # The wave-matrix arm of the w8a8 grid: V_WMMA_I32_16X16X16_IU8 chains an exact i32 group
  # sub-sum, the group term the reference's. The `impl` point of the dot2 arm's row.
  "r3d_gemm_inta8_wmma|"
  # The w8a8 wave-matrix arm at prefill heights (17 <= M <= 127): two 16-row tiles a wave, the
  # weight column reused over the tiles, the group term the reference.
  "r3d_gemm_w8a8_nt_m64|"
  # Paged attention at head_dim 128: the decode leg (one head group a workgroup, 16-token tiles,
  # split-KV) and the prefill leg (a 16-token query tile staged in LDS), over a bf16 or an E4M3
  # cache. The walk is shared; the tile is what a workgroup stages once and every wave consumes.
  "r3d_attn_paged_h128|"
  # The sampler's spine: the row's top-R in one scan, and the stages that turn the candidate plane
  # the top-R produces into one token -- argmax, temp, top-k, the draw, and the four narrowings.
  "r3d_sample_f32|"
  # Activation quantisation: per-(row, group) symmetric E4M3 and int8, the gated form that
  # quantises a silu-gated pair, and the dequantise that reads a container's own quantised weights.
  "r3d_quant_fp8|"
  "r3d_had_quant_act_i8|"
  "r3d_gelu_mul_quant_i8|"
  "r3d_kv_len_trim|"
  "r3d_qsa_tail_store|"
  "r3d_gated_had_quant_i8|"
  "r3d_had_quant_act_fp8|"
  "r3d_gated_had_quant_fp8|"
  "r3d_quant_cols_fp8|"
  "r3d_had_quant_act_i8g|"
  "r3d_gated_had_quant_i8g|"
  "r3d_rmsnorm_quant_i8|"
  # The skinny bf16 GEMM for decode, M <= 16: one output element a lane, k ascending, so the
  # accumulation order is libref's.
  "r3d_gemm_bf16_nt_m16|"
  # The wave-matrix arm of the skinny bf16 GEMM: V_WMMA_F32_16X16X16_BF16 over a 16x16x16 tile, a
  # wave a tile, the `impl` point of the dot2 arm's row.
  "r3d_gemm_bf16_nt_m16_wmma|"
   # The wave-matrix bf16 GEMM for the M = 17..127 band: MT 16-row tiles a block, the B-fragment
   # reused over the tiles so the weight stream is read once per 16-column tile.
   "r3d_gemm_bf16_nt_m64_wmma|"
  # The same shape over an f16 checkpoint: the bytes are the dot's native pair, so the stage is a
  # straight copy and the weight streams into the fold with no widening.
  "r3d_gemm_f16_nt_m16|"
  # The wave-matrix arm of the skinny f16 GEMM: V_WMMA_F32_16X16X16_F16 over a 16x16x16 tile, the
  # `impl` point of its row.
  "r3d_gemm_f16_nt_m16_wmma|"
  # The 4-bit weight against an f16 activation, M up to 64: the codes through the tile's f16 on a
  # 1024 offset, the zero a packed f16 add, the group scale on the f32 accumulator. The weight's
  # fragment-order permute is the row's own relayout hook, at load.
  "r3d_gemm_w4a16_nt_m64|"
  "r3d_gemm_w5a16_nt_m64|"
  # The gated-delta-net chunk preamble, chunked scan and recurrent update: libhalo's kernels
  # (r3dh_*, the halo_ -> r3dh_ rename), checked against libref on gfx1151 and compiled here for
  # the whole family. The recurrent update serves both state forms (per_candidate and anchor).
  "r3dh_gdn_kkt_solve|"
  "r3dh_gdn_chunk_scan|"
  "r3dh_gdn_recurrent_update|"
  "r3d_gdn_gated_rmsnorm_h128|"
  "r3d_gdn_gated_norm_had_quant_i8|"
  "r3d_gdn_conv_w4_h128|"
  # Mixture of experts: each routing weight times its expert's own scale, in place.
  "r3d_expert_scale_bf16|"
  # The router: softmax over the expert logits, the top_k by value, renormalised if `norm`.
  "r3d_moe_router_topk_bf16|"
  # The routed slots' stable counting sort (the small form, one workgroup).
  "r3d_moe_scatter|"
  # Fold the expert outputs back to the token (the gather, and the inverse-table form).
  "r3d_moe_gather|"
  # The grouped w4a8 GEMM core: int8 activation x two's-complement 4-bit weight, widened to f16 and
  # run on the f16 wave-matrix, a per-(row, 128-of-K) scale fold. Phase 1: dense rows, no gather.
  "r3d_moe_gemm_w4a8|"
   # The grouped w4a4 GEMM: int4 activation x two-s-complement 4-bit weight, both widened to int8
   # and run on the exact i32 wave-matrix, a per-(row, 128-of-K) scale fold.
   "r3d_moe_gemm_w4a4|"
    "r3d_dequant_w4_bf16|"
    "r3d_rowtopk_merge|"
    "r3d_ngram_ids|"
    "r3d_gemm_fp8a8_bias|"
    "r3d_gemm_fp8a8_short|"
    "r3d_moe_gemm_q_mod4|"
  # The grouped E4M3 x E4M3 GEMM (moe_gemm_q): both operands fp8, staged compact and upcast to f16
  # in registers (the FeatherOps bit trick), the f16 wave-matrix, a per-(row, g-of-K) scale fold.
  "r3d_moe_gemm_q|"
  # The grouped GEMM with unquantised bf16 experts: the same run-walk / gather as moe_gemm_q, the
  # bf16 operands loaded straight into the bf16 wave-matrix fragments (f32 accumulate), no decode.
  "r3d_moe_gemm_bf16|"
  # The grouped GEMM with canonical MXFP4 experts: E2M1 codes (two a byte) decoded exactly to E4M3
  # and run on the f16 wave-matrix, the E8M0 scales folded in f32 per 32 of K.
  "r3d_moe_gemm_mxfp4a8|"
  "r3d_moe_gemm_fp8_f32scale|"
  # The same shape over a block-scaled fp8 weight, decoded to f16 exactly (E4M3's four exponent bits
  # and three mantissa bits are a subset of f16's), and over int8 codes.
  "r3d_gemm_fp8a16_nt_m1|"
  # A row in r3d_rows.cpp must name an entry point defined by one of these units: an undefined
  # symbol makes the whole plugin fail to load.
)

set(R3D_SRC "")
foreach(u ${R3D_UNITS})
  string(REPLACE "|" ";" parts "${u}")
  list(GET parts 0 unit)
  list(LENGTH parts n)
  set(extra "")
  if(n GREATER 1)
    list(GET parts 1 extra)
  endif()
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${unit}.hip")
  if(NOT EXISTS "${src}")
    message(FATAL_ERROR "libr3d unit ${unit}.hip is missing from ${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  if(extra)
    set_source_files_properties("${src}" PROPERTIES COMPILE_OPTIONS "${extra}")
  endif()
  list(APPEND R3D_SRC "${src}")
endforeach()

# The host sources are declaration only: the row table and the operand descriptions. They are pure
# host C++ that includes neither r3d.h nor hip_runtime.h, so the tables compile and can be inspected
# without ROCm. libr3d has no layout source: nothing it serves is repacked, because a dot2 GEMM
# reads a weight in the order it is stored (see r3d_rows.cpp). There is no fusion source because
# no fused row exists.
set(R3D_DECLARATION r3d_rows.cpp r3d_shapes.cpp)

rad_add_plugin(libr3d kernels
  SOURCES     ${R3D_DECLARATION}
  HIP_SOURCES ${R3D_SRC}
  GPU_TARGETS ${R3D_GPU_TARGETS}
  INCLUDES    ${CMAKE_CURRENT_SOURCE_DIR})

# rad_plugin_info's build_target is this build's own target list, spelled as the device reports its
# arch, because the loader warns when a plugin claims a card the running device is not
# (core/plugin/loader.cpp, target_covers). A library built for two archs says two.
string(REPLACE ";" " " R3D_BUILD_TARGET "${R3D_GPU_TARGETS}")
target_compile_definitions(libr3d PRIVATE R3D_BUILD_TARGET="${R3D_BUILD_TARGET}")

# Catalogue dump: prints every row and, on a card, calls each launch with a minimal valid input, to
# catch transcription errors in the constraint table. It dlopens the plugin so an undefined symbol
# in the .so is caught too.
add_executable(r3d_selftest ${CMAKE_CURRENT_SOURCE_DIR}/r3d_selftest.cpp)
# Links the HIP runtime to allocate the device buffers it feeds the kernels.
target_link_libraries(r3d_selftest PRIVATE radiance::abi ${CMAKE_DL_LIBS} ${RAD_HIP_RUNTIME})
target_include_directories(r3d_selftest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# The selftest's device phase (the per-row launch) needs the HIP runtime, which this build has.
target_compile_definitions(r3d_selftest PRIVATE RAD_HAVE_HIP)
set_target_properties(r3d_selftest PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# Installed beside the tools, so the image can run the rows' own card checks without a build tree:
#   r3d_selftest <home>/kernels/libr3d.so [--case <row>]
install(TARGETS r3d_selftest RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# C++17, the standard all radiance plugins build device code with.
set_target_properties(libr3d PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
