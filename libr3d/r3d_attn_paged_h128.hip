/* r3d_attn_paged_h128.hip -- paged attention at head_dim 128, decode and prefill bands.
 *
 *     out[tok, h, :] = softmax_j( scale * q[tok, h, :] . K[j, h/gqa, :] ) . V[j, h/gqa, :]
 *
 * over a paged cache, in libref's order. A tolerance on an attention output can hide a
 * summation-order difference, so the kernel keeps the reference's chain and states where it
 * departs from it.
 *
 * ============================== the layout ==============================
 *
 *     kv_cache[num_blocks, kv_heads, block_size, 2 * head_dim]      K then V inside a slot
 *
 * A slot's K is 128 elements and V the 128 after it, so a key's row is a 512-byte pair and both
 * halves are 16-byte aligned at every slot. `block_table[seq, j / block_size]` names the block
 * holding key j; ids may repeat, so one block can serve several keys. A negative entry is an
 * unmapped block and hides its keys (the engine's allocator frees a sequence by writing -1 into
 * its rows). `seqused[seq]` is the sequence's total key count, this step's query rows included, so
 * query row i of a len_s-row step sits at absolute position `seqused[s] - len_s + i`. One kernel
 * serves decode (q_len 1), a speculative verify (q_len 7) and a prefill chunk (q_len 2048).
 *
 * ============================== the ragged batch ==============================
 *
 * Sequences may contribute different numbers of query rows. `cu_seqlens`, when present, is
 * [n_seq + 1] i32 with cu[0] == 0 and cu[n_seq] == the query plane's row count; sequence s owns
 * rows [cu[s], cu[s+1]), `len_s = cu[s+1] - cu[s]` of them, at positions from `seqused[s] - len_s`.
 * This is libref's reading (its shape hook: `cu[n_seq] == rows`, `q_len = cus ? 0 : rows / n_seq`;
 * its walk takes each row's sequence, length and index from the descriptor). So `q_len` is a
 * per-sequence number; the total row count is the plane's, and a ragged batch has no q_len.
 *
 * One grid covers a ragged batch because the clamp described under "the width of a workgroup" is
 * per sequence: sequence s gets `ceil(len_s / rows_per_tile)` tiles, and its last tile's slots past
 * `len_s` compute a duplicate that stores nothing. `map_seq` lays out the rows by walking the
 * descriptor once per workgroup; a rectangular batch is the same walk with len_s = q_len and
 * cu[s] = s * q_len, so both forms share one mapping, mask, walk and store. The causal mask, the
 * window, the block-table walk and the descales all key off the row's sequence, as the reference's
 * `s = seq_of(cus, n_seq, tok)` does.
 *
 * ============================== the mask ==============================
 *
 *   causal = 1   key j visible when j <= pos
 *   causal = 0   key j visible when j <  ctx
 *   window > 0   causal: additionally pos - j < window; non-causal: |pos - j| < window, the
 *                two-sided reading libref's prose and libr4d's window kernel both state.
 *
 * A masked key is skipped, never scored to -inf, so the denominator has exactly the reference's
 * terms; a row with no visible key stores zeros (libref's `mx == -INFINITY` case).
 *
 * ============================== the tile ==============================
 *
 * An output row is (query row, query head): 128 dims, one f32 chain per dim. A lane owns one output
 * row, as in the skinny GEMM rows: the lane's chain over keys is the reference's, and no cross-lane
 * reduction touches the denominator. A wave's 32 output rows must read the same KV, which fixes the
 * mapping:
 *
 *     kh = blockIdx.y      h = kh * GQA + (lane & (GQA-1))      tok = base + (lane >> lg GQA)
 *
 * A wave is the GQA query heads of one KV head over 32/GQA query rows (8 heads x 4 rows at gqa 8).
 * Every lane of a wave reads the same cache row at the same moment, so every cache read is an LDS
 * broadcast (one `ds_read_b128` serves the wave) and the paged walk is per wave, not per lane. A
 * workgroup is WAVES waves = WAVES x (32/GQA) query rows of one sequence and one KV head: 16 rows
 * at gqa 8 with four waves. WAVES is a template parameter chosen by the launcher (see below).
 *
 * 32 rows per cache read is also the register-file ceiling: a second output row per lane needs
 * 2 x 64 VGPRs of widened query and 2 x 128 of f32 accumulator, 384, over the 256 that four waves a
 * SIMD allow.
 *
 * ============================== the width of a workgroup ==============================
 *
 * A row slot past its sequence's length computes a clamped duplicate: the walk, the tile and the
 * store run and the result is dropped. At q_len = 7 with gqa = 8 a four-wave workgroup carries 16
 * slots for 7 rows, twice the walk for half the outputs. So the launcher picks the narrowest
 * workgroup that carries the caller's q_len (pick_waves: one wave up to 32/GQA rows, two up to
 * 64/GQA, four above); prefill keeps four because its q_len is the whole chunk. A ragged batch is
 * picked on the average sequence (`ceil(rows / n_seq)`); the width only moves duplicate rows
 * between tiles, never a result.
 *
 * The tile is the same 8 KiB at every width, so the cache traffic is unchanged and the gain comes
 * from the duplicated rows. The kernel is issue-starved at the decode bands (about 15% of the
 * SIMD's VALU slots at 256 sequences, under 4% at eight), and its cost follows how many waves a
 * workgroup brings to each barrier: at q_len 7, gqa 8, two waves take about half the time of four.
 * The width is a run-time choice because it is the main lever on this op.
 *
 * ============================== levers that do not help ==============================
 *
 * Each of these changed nothing measurable on the decode and prefill bands:
 *
 *   s_setprio 1 around the consume leg of the walk (`s_setprio` takes an immediate, not an SGPR).
 *
 *   R3D_WAVES(2) to lift the scalar-register ceiling: the bf16 variants report sgpr_count 107 and
 *   about 98 spilled SGPRs at both 4 and 2 waves a SIMD. The compute-shader scalar budget does not
 *   follow waves/SIMD the way the VGPR budget does.
 *
 *   __noinline__ on the stage leg: spills drop to 72 and a 160-byte private frame appears. Note:
 *   while the bf16 variants spill SGPRs, a kernel in this TU that also needs a private segment
 *   fails to launch ("the operation cannot be performed in the present state").
 *
 *   A two-stage pipeline (two tiles, the next block's loads issued before the current one is
 *   consumed, one barrier a block instead of two): the per-block structure is the cost and the
 *   legs do not overlap; the workgroup width is the lever.
 *
 * ============================== the walk ==============================
 *
 * block_size = 16, so one step stages a block of the KV head: 16 slots x 128 elements, K and V,
 * widened to f16 pairs on the way in, 4 KiB each. The waves of a workgroup share the stage, so a
 * block is one global read per workgroup per pass, and the walk uses 8 KiB of the 64 KiB of LDS.
 * The second pass re-reads the same blocks; one KV head's cache is ctx x 256 B (1 MiB at 4096
 * keys), so the second pass hits L2 and DRAM sees each byte once.
 *
 * A wave's query rows are consecutive, so at gqa 8 its four positions span 3 and its diagonals lie
 * in one 16-key block, or two at a block boundary. Blocks entirely below the wave's minimum
 * position need no mask; the one or two diagonal blocks test `j <= pos` per lane; blocks past the
 * wave's maximum position are not walked. In a 2048-token causal prefill that is one masked block
 * in sixty-four.
 *
 * ============================== the two passes ==============================
 *
 * The reference is two-pass: pass one finds every visible key's score and the maximum; pass two,
 * ascending keys, computes `w = exp(score - max)`, `sum += w`, `acc[d] += w * v[d]`, then
 * `acc * (1/sum)` and one narrowing at the store. Online (flash) softmax, rescaling a running
 * accumulator by `exp(m_old - m_new)`, brackets the denominator and every acc[d] differently, so it
 * is not used. Keeping the scores would need jcap floats a row (32 rows x 4096 keys = 512 KiB a
 * wave against 64 KiB of LDS), so pass two recomputes them with the same order and operands,
 * bit-identical. QK is paid twice, about 1.5x the cost of keeping the scores.
 *
 * QK is 64 `v_dot2acc_f32_f16` a key a lane over widened f16 pairs, ascending d. A dot2 folds a
 * pair's two products into the accumulator with one rounding where the reference rounds per
 * product: the same one-ulp departure the bf16 GEMM row documents, on a fraction of a percent of
 * outputs, one bf16 ulp each.
 *
 * P . V is 128 f32 multiply-accumulates a key a lane, `acc[d] += w * v[d]` with -ffp-contract=off,
 * the reference's expression with vds folded into the staged V. dot2 cannot do this leg here: w is
 * a per-lane f32 scalar and the accumulation is per dim, so dot2 would need w folded into V at f16,
 * reading every weight to 11 bits (about 1e-4 relative, a few percent of outputs one ulp off after
 * the bf16 store). That variant is a third of the instructions and is not used.
 *
 * ============================== exp ==============================
 *
 * `ref_exp` is `(float)std::exp((double)x)`, correctly rounded to f32 through a double. RDNA3 has
 * no vector f64 (soft-float double costs hundreds of cycles a call), the device libm's `expf` is a
 * ~1-ulp polynomial and `__expf` is two orders of magnitude worse. `exp_ref` is a software expf:
 * two-term range reduction against a split ln2, a degree-8 polynomial, and 2^k assembled in the
 * integer domain. It is the only place in this library where an explicit `__fmaf_rn` is allowed.
 * The no-FMA rule exists because libref's expressions contain no FMA; here the target is a
 * function libref computes in double, and the polynomial's FMAs are local to this helper and
 * contract no expression the reference wrote.
 *
 * `exp(0)` is exactly 1.0: k is 0, r is exactly 0, and the polynomial's constant term passes
 * through. The argument is always <= 0 (the max is over the same set), so only x <= 0 matters.
 * Below -103.98 the result is under f32's smallest subnormal and the helper returns 0; the
 * reference's value there is of order 1e-44, which cannot change a denominator of order 1 or
 * survive a bf16 store. Between -103.98 and -87.3 the result is formed by one hardware multiply by
 * 2^-126, which the hardware flushes to zero (see exp_ref).
 *
 * ============================== split-KV ==============================
 *
 * The decode entry takes `splits`: a sequence's blocks are cut into slices of whole blocks, each
 * accumulating its own (acc[128], sum, max) into scratch, and a second kernel combines the slices
 * before the single narrowing. Each slice is rescaled by `exp(m_slice - M)` with M the slices'
 * maximum: the denominator becomes the sum over slices of `sum_s * exp(m_s - M)`, ascending in
 * slice, and every term arrives as `exp(s - m_s) * exp(m_s - M)` where the reference has
 * `exp(s - M)`. That is about 2 f32 ulp per term and 1 on the sum, visible after the bf16 store
 * only on outputs at a rounding boundary. `splits > 1` is not bit-identical to the reference;
 * `splits = 1` is.
 *
 * `splits` is a maximum, resolved per sequence from its own `seqused`: `nslice = ceil(nblk /
 * ceil(nblk / splits))` with nblk the sequence's block count, so a short sequence uses fewer slices
 * and there is no per-sequence legality check. The reducer recomputes `nslice` from the same two
 * numbers, so a partial needs no header beyond (acc, sum, max).
 *
 * ============================== what the shims refuse ==============================
 *
 * head_dim other than 128 (`RAD_E_SHAPE`); a cache dtype other than bf16 and fp8_e4m3 (an f32 cache
 * has no dot2 path, and u8/u4/int8 codes are not this cache form; `RAD_E_DTYPE`); a query dtype
 * other than bf16, and an out dtype that is not the query's (`RAD_E_DTYPE`); gqa outside {1,2,4,8}
 * or not dividing n_head (the wave's head split is a shift); block_size other than 16 (the stage is
 * sized for it); q_len < 1 in a rectangular batch, and a row count < 1 in a ragged one; a row count
 * past 2^31 (num_seqs * q_len, or cu[n_seq]), since row indexing is 32-bit; a q, kv or out stride
 * that puts a slot's halves off a 16-byte boundary, or a row pitch that is not a multiple of 8; a
 * `cu_seqlens` that is not i32 (`RAD_E_DTYPE`; the kernel reads int), not contiguous
 * (`RAD_E_STRIDE`), not [n_seq + 1] (`RAD_E_SHAPE`) or not 4-byte aligned (`RAD_E_ALIGN`); descales
 * attached to a bf16 cache, which the reference ignores and this kernel refuses rather than drops,
 * and one descale plane without the other (`RAD_E_SHAPE`); an output shorter than
 * rows * n_head * head_dim. A scratch arena too small for the requested `splits` is not refused:
 * the shim lowers `splits` until the partials fit, down to 1, which needs none.
 *
 * What the descriptor carries that the host cannot see is clamped, not refused, because a launcher
 * reads no device memory. cu[0] == 0, monotonicity and cu[n_seq] == the plane's row count are the
 * description's promises (the reference's shape hook states all three), and the kernel handles each
 * violation as the reference does: rows past the plane are dropped (`row_ok` tests the plane's row
 * count, so a long descriptor cannot write outside `out`); a segment with cu[s+1] <= cu[s] has no
 * tiles and contributes no rows (the reference's `if (qlen <= 0) return;`); and rows under cu[0]
 * belong to sequence 0 with a negative index into its query, as `seq_of` assigns them.
 */
#include "r3d.h"
#include "r3d_common.h"
#include "r3d_codecs.h"
#include "r3d_args.h"
#include "r3d_launch.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>   /* size_t, in the arena indexing and the shim's scratch sizing */
#include <cstdio>    /* std::fprintf, the RAD_R3D_TRACE lines */
#include <cstdlib>   /* std::getenv / std::atoi, RAD_R3D_TRACE and RAD_R3D_DEV */

/* ================================================================== the device record ring
 *
 * Diagnostic instrument, enabled by RAD_R3D_TRACE with its record budget from RAD_R3D_DEV (default
 * 64): thread 0 of workgroup (0,0,0) of each attn_kernel launch writes one DevRec describing what
 * the launch was handed, and the shim prints the records to stderr as "[r3d dev] L=... ctx=...
 * ids=..." lines.
 *
 * The kernel writes to memory rather than calling device `printf`: a printing kernel takes an
 * implicit `hidden_hostcall_buffer` argument that the engine's AQL dispatcher does not provide, so
 * the engine could not dispatch it at all.
 *
 * Four plugin-owned device globals hold the state, because neither the launch parameters nor
 * `RadArgs` (the row's ABI) can take a new field. The shim writes the budget with
 * `hipMemcpyToSymbol` only under RAD_R3D_TRACE:
 *
 *     r3d_dev_budget    records left to write; 0, the initial value, means record nothing
 *     r3d_dev_launches  launches seen, incremented once per launch by the recording thread
 *     r3d_dev_widx      the ring's write index, bumped once per record written
 *     r3d_dev_ring      kDevRing records of 216 bytes each, 55 KiB
 *
 * The record is written by the kernel because a step replayed from the engine's recorded pass only
 * writes queue packets and never calls the shim. A record holds the step's `ctx` as read from
 * `seqused`, the block ids the walk reads from the table, the operand addresses in the packet, and
 * three content folds (the query row, and the K rows of the first and last blocks the row walks),
 * so a block whose contents or address change underneath a step is visible.
 *
 * A replayed step's record is printed at the next shim call, one step late; the record's `L` and
 * `ctx`, and the counter the shim's plan line quotes, restore the order.
 *
 * Cost with the flag unset: a launch pays one atomic increment of the counter, one atomic decrement
 * of the budget and a not-taken branch, on one thread of one workgroup. The budget is decremented
 * atomically so several workgroups cannot record copies of one launch. A full ring drops records
 * but keeps counting, and the shim reports the loss on one line.
 *
 * The counter counts every `attn_kernel` launch, budget or not, but not the split-KV reduce. */
enum { kDevRing = 256, kDevIds = 8 };

/* One launch's record: a fixed plain struct with no pointers, floats or padding. `b` holds the
 * row's first kDevIds table columns (two columns cover a short context at block_size 16), and
 * `idsum` folds every column so a long context still yields one comparable number. */
struct DevRec
{
    unsigned long long L;                        /* the launch's own ordinal                */
    long long ctx, pos, slot0, slotL, idsum;     /* the step as the row reads it, and where */
    unsigned long long qh, kh, khL;              /* three content folds                     */
    unsigned long long kv, bt, su, cu, q, out;   /* the six addresses baked into the packet */
    int dec, n_seq, rows, q_len, s, len, span, start;
    int jcap, splits, waves, ncols, rowok, spare, spare2, spare3;
    int b[kDevIds];                              /* the row's ids; `ncols` of them are real */
};

extern "C" __device__ int r3d_dev_budget = 0;
extern "C" __device__ unsigned long long r3d_dev_launches = 0ULL;
extern "C" __device__ unsigned r3d_dev_widx = 0u;
/* `= {}` is required: an uninitialised `__device__` variable emits no host-side symbol, and the
 * loader's dlopen then fails with `undefined symbol: r3d_dev_ring` before any row is declared. */
extern "C" __device__ DevRec r3d_dev_ring[kDevRing] = {};


namespace {

typedef float r3d_f32x4 __attribute__((ext_vector_type(4)));

constexpr int kHeadDim  = 128;
constexpr int kPairs    = kHeadDim / 2;        // f16 pairs a head: one dword is two dims
constexpr int kBlock    = 16;                  // block_size: a step stages one of these
constexpr int kWaves    = 4;                   // 128 threads
constexpr int kThreads  = kWaves * 32;
constexpr int kPitch    = kPairs;              // dwords a staged row; rows are read as broadcasts
constexpr int kTileDw   = kBlock * kPitch;     // 1024 dwords = 4 KiB
constexpr int kPartStride = kHeadDim + 4;      // acc[128], sum, max, pad -- 528 B, whole 16 B rows

/* ------------------------------------------------------------------ exp */

/* exp(x) narrowed to f32, as close to `(float)std::exp((double)x)` as an ALU without f64 gets.
 *
 * Against `(float)std::exp((double)x)` over 1.31M points of [-100, 0] and the boundary cases,
 * every point is within one f32 ulp and 76.1% are bit-identical (the device libm's expf: 94.0%, no
 * better in ulps); exp(0) is exactly 1. Below 2^-126 (x under about -87.3) the result is zero, not
 * subnormal, because the hardware flushes the scaling multiply. That is invisible at the store: a
 * weight of 1e-38 beside a maximum weight of 1 cannot move an accumulator.
 *
 * Reduction by ln2 against a two-term constant: k = rn(x / ln2), r = x - k*LN_HI - k*LN_LO with
 * LN_HI the high 24 bits of ln2. A single-term reduction costs 1.5 ulp, since k*ln2 is not
 * representable and the subtraction's cancellation dominates the error. |r| <= 0.35, where a
 * degree-8 polynomial reaches ~0.55 ulp, then 2^k by exponent assembly. */
__device__ __forceinline__ float exp_ref(float x)
{
    /* Above the overflow point the reference's double path yields +inf, and so does this; NaN in,
     * NaN out. Neither occurs in the softmax (the argument is never positive); they keep the
     * function a true exp. */
    if (!(x < 88.7228390525719f))
        return x == x ? __int_as_float(0x7f800000) : __int_as_float(0x7fc00000);
    if (x < -103.9720770839916f) return 0.0f;                  /* under the subnormals: see header */
    const int k = (int)__float2int_rn(x * 1.4426950408889634f);
    const float LN_HI = 0.693145751953125f;                    /* ln2, high bits, exactly a float */
    const float LN_LO = 1.42860681039557e-06f;                 /* ln2 - LN_HI */
    float r = x - (float)k * LN_HI;
    r = r - (float)k * LN_LO;
    /* e^r, |r| <= 0.35: 1 + r + r^2/2 + sum(r^i / i!), i = 3..8, Estrin over r^2. */
    const float c8 = 1.98412698412698e-04f, c7 = 1.38888888888889e-03f;
    const float c6 = 8.33333333333333e-03f, c5 = 4.16666666666667e-02f;
    const float c4 = 1.38888888888889e-01f, c3 = 1.66666666666667e-01f;
    const float r2 = r * r;
    const float p4 = __fmaf_rn(__fmaf_rn(__fmaf_rn(c8, r, c7), r, c6), r, c5);
    const float p2 = __fmaf_rn(p4, r2, __fmaf_rn(c3, r, 5.0e-01f));
    const float e = __fmaf_rn(p2, r2, __fmaf_rn(r, 1.0f, 1.0f));   /* in [0.70, 1.42] */
    /* Scale by 2^k in the exponent field; x <= 0 here, so the normal case is the whole story and
     * the subnormal one is a single hardware multiply. */
    const int be = __float_as_int(e);
    const int field = ((be >> 23) & 0xff) + k;
    if (field >= 1) return __int_as_float((be & 0x007fffff) | (field << 23));
    if (field <= -23) return 0.0f;
    const float t = __int_as_float((be & 0x007fffff) | ((field + 126) << 23));
    return t * __int_as_float((127 - 126) << 23);              /* one rounding, the hardware's */
}

/* ------------------------------------------------------------------ the kernel's view */

struct Args {
    const uint16_t* q;                 /* bf16 or f16 (q16), rows x n_head x head_dim */
    const uint16_t* kv;                /* bf16 cache */
    const uint8_t*  kv8;               /* E4M3 cache, when FP8 */
    const int*      bt;                /* block table, i32: the shim refuses i64 indices */
    const int*      su;                /* seqused, i32 */
    const int*      cu;                /* cu_seqlens, n_seq + 1, i32; null = q_len rows each */
    const float*    kds;
    const float*    vds;
    uint16_t*       out;
    float*          part;              /* split-KV partials */

    /* Strides are 32-bit, in elements: a cache is blocks x kv_heads x 16 x 256 elements, a row
     * pitch is n_head x head_dim, a block table pitch is max_blocks, and the launcher checks that
     * each fits. Every address is a 64-bit base plus a 32-bit product because scalar registers run
     * out first: in compute-shader mode a scalar spill is not a slow path but a kernel the runtime
     * refuses to load. */
    int sb, sh, so, sd;                /* cache strides */
    int q_rs, o_rs;                    /* row pitches; the head stride is head_dim */
    int bt_rs;   /* the block table row pitch, in entries: the tensor's own stride */
    int bt_cs;   /* its column stride: 1 for every dense table the engine builds */
    int n_seq, q_len, rows, n_head, kv_heads, causal, window, jcap, splits, q16, waves;
    float scale;
    DevRec* trace_buf;               /* RAD_R3D_TRACE: one record slot in a hipMalloc buffer, which
                                       * the engine process reads back reliably, unlike a dlopened
                                       * plugin's device globals */
};

__device__ __forceinline__ long long ld_ix(const int* p, size_t i) { return p[i]; }

/* ------------------------------------------------------------------ the row map
 *
 * One map for both batch forms. With a descriptor, sequence s owns query rows [cu[s], cu[s+1]):
 * `len_s` rows, the row's index within the sequence `i = row - cu[s]`, its absolute position
 * `seqused[s] - len_s + i`. Without one, the same walk with len_s = q_len and cu[s] = s * q_len,
 * the reference's rectangular reading.
 *
 * A grid row is resolved by subtracting each sequence's tile count from blockIdx.x until the
 * remainder fits: n_seq loads and subtractions once per workgroup, against a walk of nblk blocks
 * with two barriers each. The launcher reads no device memory, so the grid is sized on a bound and
 * a grid row past the batch's last tile gets tile < 0 and returns without staging.
 *
 * `start` is the sequence's first dense row and `i0` the row its positions count from. They differ
 * only when cu[0] > 0: the reference's `seq_of` assigns rows under cu[0] to sequence 0 with a
 * negative index into its query, and that is reproduced rather than refused. Both are 32-bit: the
 * launcher has checked that the plane's element offsets fit in 31 bits. */
struct RowMap { int s, tile, start, i0, len, span; };

__device__ __forceinline__ RowMap map_seq(const Args& A, const int tpb, int g)
{
    const int n = A.n_seq;
    for (int s = 0; s < n; ++s) {
        const int c0 = A.cu ? (int)ld_ix(A.cu, (size_t)s) : s * A.q_len;
        const int c1 = A.cu ? (int)ld_ix(A.cu, (size_t)(s + 1)) : c0 + A.q_len;
        const int start = (s == 0 && c0 > 0) ? 0 : c0;
        const int nr = c1 - start;
        const int nt = nr > 0 ? (nr + tpb - 1) / tpb : 0;      /* tpb is a template constant */
        /* `len` is the sequence's query length, cu[s+1] - cu[s], which the positions count from
         * (the reference's qlen; pos = ctx - qlen + i). `span` is the rows this sequence's tiles
         * carry, cu[s+1] - start; the two differ by cu[0] on sequence 0 when the descriptor does
         * not start at row 0, because `start` then reaches below cu[0]. Stores are bounded by the
         * span and positions by the length; bounding the tiles by `len` would silently drop the
         * last cu[0] rows of the query plane. */
        if (g < nt) return RowMap{ s, g, start, c0, c1 - c0, nr };
        if (s + 1 >= n) return RowMap{ s, -1, start, c0, c1 - c0, nr };  /* past the batch's tiles */
        g -= nt;
    }
    return RowMap{ n - 1, -1, 0, 0, 0, 0 };
}

/* The same walk for a row rather than a tile: which sequence owns dense query row `r`. The reducer
 * needs it because a partial's slice count is the sequence's block count. */
__device__ __forceinline__ int seq_of_row(const Args& A, int r)
{
    const int n = A.n_seq;
    for (int s = 0; s < n; ++s) {
        const int c1 = A.cu ? (int)ld_ix(A.cu, (size_t)(s + 1)) : (s + 1) * A.q_len;
        if (r < c1 || s + 1 >= n) return s;
    }
    return n - 1;
}

/* ------------------------------------------------------------------ staging */

/* Sixteen E4M3 codes (one 16-byte load, the aligned unit a slot's 128-byte half divides into) as
 * eight f16 pairs, descaled on the way. A code is exact in f16, so a power-of-two descale (every
 * real cache scale) costs a multiply and no rounding; a general descale is rounded once, toward
 * zero by the packing instruction, where the reference keeps f32. The bf16 path has no descale. */
__device__ __forceinline__ void codes16_to_f16x8(const uint8_t* p, float s, uint32_t* w)
{
    const r3d_u32x4 c = *(const __attribute__((address_space(1))) r3d_u32x4*)(const void*)p;
    const uint32_t d[4] = { c.x, c.y, c.z, c.w };
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float a0 = e4m3_to_f32((uint8_t)(d[j] & 0xffu)) * s;
        const float a1 = e4m3_to_f32((uint8_t)((d[j] >> 8) & 0xffu)) * s;
        const float a2 = e4m3_to_f32((uint8_t)((d[j] >> 16) & 0xffu)) * s;
        const float a3 = e4m3_to_f32((uint8_t)(d[j] >> 24)) * s;
        w[j * 2] = f16x2_pack(a0, a1);
        w[j * 2 + 1] = f16x2_pack(a2, a3);
    }
}

/* Stage one block of one KV head: 16 slots x 128 elements, K and V, widened to f16 pairs.
 *
 * A slot's halves are 256 bytes each, so an 8-element bf16 chunk is 16 bytes on a 16-byte boundary,
 * and so is a 16-code E4M3 chunk. An 8-code chunk would be 8 bytes, and a misaligned 16-byte load
 * returns wrong data on this hardware; hence the two chunk sizes. The tile is 8 KiB; at 128 threads
 * each thread moves one K and one V chunk for fp8, two of each for bf16.
 *
 * Every offset is 32-bit and the launcher has checked that the whole cache fits in 2^31 elements:
 * 64-bit index arithmetic in this loop would exhaust the scalar registers (see Args). */
template <bool FP8, int NTH>
__device__ __forceinline__ void stage(Args const& A, uint32_t* sK, uint32_t* sV, int blk, int kh,
                                      float kds, float vds, int tid)
{
    constexpr int kC = FP8 ? 16 : 8;                            /* elements a 16-byte chunk */
    constexpr int kCPH = kHeadDim / kC;                         /* chunks a half, a slot */
    constexpr int kChunks = kBlock * kCPH;                      /* 128 code chunks, 256 bf16 */
    constexpr int kIter = kChunks / NTH;   /* bf16: 8, 4, 2 at one, two, four waves */
    static_assert(kIter * NTH == kChunks, "the stage must divide into whole per-thread trips: "
                                               "a bounds check inside the unrolled loop keeps the "
                                               "loop from unrolling, and the loop-invariant operands "
                                               "then get hoisted into scalar registers");
    const int base = blk * A.sb + kh * A.sh;                    /* this head's block, elements */

#pragma unroll
    for (int u = 0; u < kIter; ++u) {
        const int t = tid + u * NTH;
        const int slot = t / kCPH;
        const int c = (t % kCPH) * kC;                          /* element offset in the half */
        const int k0 = base + slot * A.so + c;
        const int v0 = k0 + kHeadDim;
        uint32_t* dk = sK + slot * kPitch + c / 2;
        uint32_t* dv = sV + slot * kPitch + c / 2;
        if constexpr (FP8) {                                    /* codes: element and byte agree */
            codes16_to_f16x8(A.kv8 + k0, kds, dk);
            codes16_to_f16x8(A.kv8 + v0, vds, dv);
        } else {
            const r3d_u32x4 ka =
                *(const __attribute__((address_space(1))) r3d_u32x4*)(const void*)(A.kv + k0);
            const r3d_u32x4 va =
                *(const __attribute__((address_space(1))) r3d_u32x4*)(const void*)(A.kv + v0);
            const uint32_t kd[4] = { ka.x, ka.y, ka.z, ka.w };
            const uint32_t vd[4] = { va.x, va.y, va.z, va.w };
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                dk[j] = bf16x2_to_f16x2(kd[j]);
                dv[j] = bf16x2_to_f16x2(vd[j]);
            }
        }
    }
}

/* ------------------------------------------------------------------ the walks */

/* The keys a lane may see: j is a candidate for the wave, the predicate is per lane. */
__device__ __forceinline__ bool visible(int j, long long pos, int causal, int window)
{
    if (causal && (long long)j > pos) return false;
    if (window > 0) {
        const long long d = pos - j;
        if (causal ? (d >= window) : (d >= window || d <= -window)) return false;
    }
    return true;
}

__device__ __forceinline__ float score_of(const uint32_t* qf, const uint32_t* kp, float scale)
{
    float dot = 0.0f;
#pragma unroll
    for (int p = 0; p < kPairs; ++p) dot = dot2_f16x2(qf[p], kp[p], dot);
    return dot * scale;
}

/* ------------------------------------------------------------------ the launch record
 *
 * `fold64` is FNV-1a over 64-bit words (multiply and xor per word, no float arithmetic, nothing a
 * compiler could contract), so a host-side checker can recompute it exactly. It is order-sensitive,
 * so the same elements read at a wrong (e.g. layer-major) offset fold to a different value.
 *
 * The words are the row's own bytes, read as the walk reads them: the query row at
 * `q + row * q_rs` for head 0 (32 words of bf16), and the K row of a block the walk names, at
 * `blk * sb + kh * sh + slot * so`, the same arithmetic as `stage()` (`slot_k_off`): 16 words of
 * E4M3 or 32 of bf16. Both are 16-byte aligned: the shim checks 16-aligned cache, query and output
 * bases and a row pitch that is a multiple of 8 elements.
 */
__device__ __forceinline__ unsigned long long fold64(const void* p, int words)
{
    const unsigned long long* w = (const unsigned long long*)p;
    unsigned long long h = 14695981039346656037ULL;                 /* FNV-1a 64 offset basis */
    for (int i = 0; i < words; ++i) h = (h * 1099511628211ULL) ^ w[i];
    return h;
}

__device__ __forceinline__ long long slot_k_off(const Args& A, int blk, int kh, int slot)
{
    return (long long)blk * A.sb + (long long)kh * A.sh + (long long)slot * A.so;
}

/* One record per launch, from block (0,0,0), thread 0, into the ring. `noinline` keeps the writer
 * out of the mapping code every launch runs, because the kernel's register budget is the scarcest
 * resource here (see Args).
 *
 * Its three reads are bounded:
 *
 *   the query row      by the plane's row count `A.rows`, the bound the store uses. `row` is the
 *                      caller's clamped row index and `row_ok` says whether it is real; a band key
 *                      (`q_len`) or the sequence's length bounds the shape, not the plane, so
 *                      reading through one could leave the plane. A tile with no rows records
 *                      `qh=0, rowok=0`, and `L` still counts the launch.
 *   the table row      by the columns the walk can name, ceil(jcap / block_size) with
 *                      jcap = min(ctx, bt_cols * block_size), which the shim proved the table
 *                      covers; the row is `M.s * bt_rs` with `M.s` a clamped sequence index.
 *   the cache          by block ids the table names; the shim proves the table's strides and that
 *                      its rows cover the walk, but the ids themselves are device data -- 0 <= id <
 *                      num_blocks is the caller's contract, shared with the reference.
 *                      The fold reads kv-head 0's slot-0 K row at the `stage()` offset, kHeadDim
 *                      elements (256 bytes bf16, 128 fp8), inside a slot's 2*kHeadDim. A negative
 *                      id (an unmapped block, printed -1) folds to 0 and is not dereferenced. */
template <bool FP8>
__device__ __noinline__ void dev_record(const Args& A, const RowMap& M, long long row, bool row_ok,
                                        long long ctx, int jend_all, long long pos)
{
    /* Every launch that runs is counted, budget or not. */
    const unsigned long long n = atomicAdd(&r3d_dev_launches, 1ULL) + 1ULL;
    /* The budget is decremented atomically, so several workgroups cannot record copies of one
     * launch; a caller that armed 3 gets 3 records. The buffer channel bypasses the budget: it is one
     * slot per launch, read back at the launch, and the engine process cannot arm the device
     * global. */
    if (!A.trace_buf && atomicSub(&r3d_dev_budget, 1) <= 0) return;

    DevRec r;
    r.L = n;
    r.dec = A.rows == 1 ? 1 : 0;
    r.n_seq = A.n_seq; r.rows = A.rows; r.q_len = A.q_len;
    r.s = M.s; r.len = M.len; r.span = M.span; r.start = M.start;
    r.ctx = ctx; r.pos = pos; r.jcap = jend_all; r.splits = A.splits; r.waves = A.waves;
    r.kv = (unsigned long long)(uintptr_t)(FP8 ? (const void*)A.kv8 : (const void*)A.kv);
    r.bt = (unsigned long long)(uintptr_t)A.bt;
    r.su = (unsigned long long)(uintptr_t)A.su;
    r.cu = (unsigned long long)(uintptr_t)A.cu;
    r.q = (unsigned long long)(uintptr_t)A.q;
    r.out = (unsigned long long)(uintptr_t)A.out;

    const int cols = (A.jcap + kBlock - 1) / kBlock;      /* <= bt_cols, by jcap's bound */
    const long long r0 = (long long)M.s * A.bt_rs;
    r.ncols = cols;
    unsigned long long is = 14695981039346656037ULL;      /* fold64's basis: idsum is an FNV-1a
                                                           * fold of the ids */
    for (int c = 0; c < cols; ++c) {
        const int b = (int)ld_ix(A.bt, (size_t)r0 + (size_t)c * (size_t)A.bt_cs);
        is = (is * 1099511628211ULL) ^ (unsigned long long)(unsigned int)b;
        if (c < kDevIds) r.b[c] = b;
    }
    for (int c = cols; c < kDevIds; ++c) r.b[c] = -2;     /* a column the row does not have; -1
                                                           * means an unmapped block */
    r.idsum = (long long)is;

    /* The query row, under the plane's bound, named above. */
    const bool q_ok = row_ok && row >= 0 && row < A.rows;
    r.rowok = q_ok ? 1 : 0;
    r.qh = q_ok ? fold64(A.q + row * (long long)A.q_rs, kHeadDim / 4) : 0ULL;

    /* The K rows of the first and last blocks the row walks. The walk covers
     * `jend_all = min(ctx, A.jcap)` keys, so it visits ceil(jend_all / kBlock) blocks, not the
     * table's `cols`: the table row spans the plane's full capacity and its far columns are -1 for
     * a short context. `ncols` and `idsum` cover the whole table row; `kh`, `khL`, `slot0`, `slotL`
     * and `bL` describe the walk's two ends, the head of an adopted prefix and the tail being
     * written, where a recycled block shows up. */
    const int walk_cols = (jend_all + kBlock - 1) / kBlock;
    const int b0 = walk_cols > 0 ? r.b[0] : -2;
    const int blast = walk_cols > 0
          ? (int)ld_ix(A.bt, (size_t)r0 + (size_t)(walk_cols - 1) * (size_t)A.bt_cs)
          : -2;
    const long long k0 = b0 >= 0 ? slot_k_off(A, b0, 0, 0) : 0;
    const long long kL = blast >= 0 ? slot_k_off(A, blast, 0, 0) : 0;
    r.slot0 = k0; r.slotL = kL;
    r.kh  = b0 < 0 ? 0ULL : (FP8 ? fold64(A.kv8 + k0, kHeadDim / 8) : fold64(A.kv + k0, kHeadDim / 4));
    r.khL = (kL == 0) ? 0ULL
          : (FP8 ? fold64(A.kv8 + kL, kHeadDim / 8) : fold64(A.kv + kL, kHeadDim / 4));
    r.spare = blast;                                          /* printed as bL */
    r.spare2 = cols; r.spare3 = 0;

    /* One slot, claimed by one atomic. Past the ring's end a record is counted and dropped, and the
     * shim reports the loss. The record is built in a local and copied in one statement; the drain
     * runs at a shim call after the engine has synced the step, so it reads whole records. */
    const unsigned w = atomicAdd(&r3d_dev_widx, 1u);
    if (w < (unsigned)kDevRing) r3d_dev_ring[w] = r;
    if (A.trace_buf) A.trace_buf[0] = r;   /* read back by launch() */
}

/* One wave's work: GQA heads x 32/GQA rows, over its slice of one sequence's blocks, for one KV
 * head. CAUSAL and SPLIT are template parameters because both change the loop, not just a flag. */
template <bool FP8, bool CAUSAL, bool SPLIT, int GQA, int WAVES>
__global__ __launch_bounds__(kThreads) R3D_WAVES(4)
void attn_kernel(const Args A)
{
    __shared__ uint32_t sK[kTileDw];
    __shared__ uint32_t sV[kTileDw];

    constexpr int TOKS = 32 / GQA;                       /* query rows a wave carries */
    constexpr int kNThr = WAVES * 32;                    /* 32, 64, 128 threads */
    constexpr int ROWSB = WAVES * TOKS;                  /* rows a workgroup carries */
    constexpr int LG = (GQA == 1) ? 0 : (GQA == 2) ? 1 : (GQA == 4) ? 2 : 3;

    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5, lane = tid & 31;
    const RowMap M = map_seq(A, ROWSB, (int)blockIdx.x);
    /* Nothing to do for a grid row past the batch's tiles (the grid is sized on a bound), a
     * sequence with no rows (cu[s+1] <= cu[s], the reference's `qlen <= 0`), or a sequence that
     * starts past the query plane (a descriptor running beyond the operand). */
    if (M.tile < 0 || M.len <= 0 || M.start >= A.rows) return;
    const int s = M.s;
    const int tile = M.tile;
    const int kh = (int)blockIdx.y;
    const int slice = SPLIT ? (int)blockIdx.z : 0;

    const int h = kh * GQA + (lane & (GQA - 1));
    const int tb = tile * ROWSB + wave * TOKS;        /* the wave's first row of the sequence */
    const int tok = tb + (lane >> LG);                /* this lane's, counted inside the sequence */
    /* A row the operands do not have: past the rows this sequence's tiles carry (the last tile of
     * any sequence, every tile of a short one), or past the query plane (a long descriptor). The
     * bound is the span, not the length: with cu[0] > 0 the span reaches below cu[0], and those
     * rows belong to sequence 0 in the reference too, so they are stored. */
    const bool row_ok = tok < M.span && M.start + tok < A.rows;
    /* The clamp target is a row that exists: the wave's first row if the sequence has one, else its
     * last, never past the plane. An idle lane loads a real query row so the walk stays
     * branch-free; it never stores. */
    int tclamp = tb < M.span ? tb : M.span - 1;
    if (M.start + tclamp >= A.rows) tclamp = A.rows - 1 - M.start;
    const long long row = (long long)M.start + (row_ok ? tok : tclamp);

    const long long ctx = ld_ix(A.su, (size_t)s);
    const int jend_all = (int)(ctx < A.jcap ? ctx : A.jcap);
    /* `ib` is 0 for every descriptor the description allows (cu[0] == 0); it is the negative index
     * into sequence 0's query that libref's seq_of gives rows under cu[0]. */
    const long long ib = (long long)M.start - M.i0;
    const long long pos = ctx - M.len + tok + ib;

    /* RAD_R3D_TRACE device record (see "the device record ring"): one thread of one workgroup
     * (`blockIdx.z` is the split-KV slice, so this is once per launch), gated on the budget global.
     * It sits before the query is widened and the walk starts, so the record covers launches whose
     * rows are all clamped and the writer overlaps the fewest live values. */
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
        dev_record<FP8>(A, M, row, row_ok, ctx, jend_all, pos);

    /* The workgroup's bounds, not the wave's. All waves share one staged tile, so they must walk
     * exactly the same blocks; a range narrowed to one wave's positions would make waves stage
     * different blocks into the same LDS and corrupt the tile. A workgroup's ROWSB consecutive rows
     * span ROWSB - 1 positions and keys past the last are unused; a window also trims the front.
     * The per-lane mask handles the rest. */
    const int gb = tile * ROWSB;
    const long long wmin = ctx - M.len + gb + ib;
    const long long wmax = wmin + ROWSB - 1;
    int jlo = 0, jhi = jend_all;
    if (CAUSAL) {
        if (wmax + 1 < jhi) jhi = (int)(wmax + 1);
    }
    if (A.window > 0) {
        const long long lo = wmin - A.window + 1;
        if (lo > jlo) jlo = (int)(lo > 0 ? lo : 0);
        if (!CAUSAL) {
            const long long hi = wmax + A.window;
            if (hi < jhi) jhi = (int)(hi > 0 ? hi : 0);
        }
    }
    const int nblk = (jend_all + kBlock - 1) / kBlock;
    int b0 = jlo / kBlock, b1 = (jhi + kBlock - 1) / kBlock;
    if (SPLIT) {
        const int tps = (nblk + A.splits - 1) / A.splits;
        const int s0 = slice * tps;
        const int s1 = s0 + tps < nblk ? s0 + tps : nblk;
        if (b0 < s0) b0 = s0;
        if (b1 > s1) b1 = s1;
    }

    const float kds = A.kds ? A.kds[(size_t)s * A.kv_heads + kh] : 1.0f;
    const float vds = A.vds ? A.vds[(size_t)s * A.kv_heads + kh] : 1.0f;

    /* The query row, widened once and kept for both walks: 64 dwords, the lane's 256 bytes. An idle
     * lane loads its clamped row (see above) and never stores. */
    uint32_t qf[kPairs];
    {
        const uint16_t* qp = A.q + row * (long long)A.q_rs + h * kHeadDim;
#pragma unroll
        for (int u = 0; u < kPairs / 4; ++u) {                 /* eight elements a 16-byte load */
            const r3d_u32x4 raw =
                *(const __attribute__((address_space(1))) r3d_u32x4*)(const void*)(qp + u * 8);
            const uint32_t d[4] = { raw.x, raw.y, raw.z, raw.w };
#pragma unroll
            for (int j = 0; j < 4; ++j) qf[u * 4 + j] = A.q16 ? d[j] : bf16x2_to_f16x2(d[j]);
        }
    }

    float acc[kHeadDim];
#pragma unroll
    for (int d = 0; d < kHeadDim; ++d) acc[d] = 0.0f;
    float sum = 0.0f, mx = -INFINITY;

    /* Walk one: the scores' maximum. Blocks entirely behind every row of the workgroup need no
     * mask, and the ones straddling a row's position mask per lane. */
    for (int b = b0; b < b1; ++b) {
        const int blk = (int)ld_ix(A.bt, (size_t)s * (size_t)A.bt_rs + (size_t)b * (size_t)A.bt_cs);
        group_barrier();
        if (blk >= 0) stage<FP8, kNThr>(A, sK, sV, blk, kh, kds, vds, tid);
        group_barrier();
        if (blk < 0) continue;
        const int j0 = b * kBlock;
        const int nu = j0 + kBlock <= jhi ? kBlock : (jhi > j0 ? jhi - j0 : 0);
        const bool nomask = !CAUSAL && A.window <= 0;
        if (nomask) {
            for (int u = 0; u < nu; ++u) {
                const float sc = score_of(qf, sK + u * kPitch, A.scale);
                if (sc > mx) mx = sc;
            }
        } else {
            for (int u = 0; u < nu; ++u) {
                if (!visible(j0 + u, pos, CAUSAL, A.window)) continue;
                const float sc = score_of(qf, sK + u * kPitch, A.scale);
                if (sc > mx) mx = sc;
            }
        }
    }

    /* Walk two: the same scores, exponentiated against the maximum, accumulated ascending. */
    if (mx > -INFINITY) {
        for (int b = b0; b < b1; ++b) {
            const int blk = (int)ld_ix(A.bt, (size_t)s * (size_t)A.bt_rs + (size_t)b * (size_t)A.bt_cs);
            group_barrier();
            if (blk >= 0) stage<FP8, kNThr>(A, sK, sV, blk, kh, kds, vds, tid);
            group_barrier();
            if (blk < 0) continue;
            const int j0 = b * kBlock;
            const int nu = j0 + kBlock <= jhi ? kBlock : (jhi > j0 ? jhi - j0 : 0);
            const bool nomask = !CAUSAL && A.window <= 0;
            for (int u = 0; u < nu; ++u) {
                if (!nomask && !visible(j0 + u, pos, CAUSAL, A.window)) continue;
                const float w = exp_ref(score_of(qf, sK + u * kPitch, A.scale) - mx);
                sum += w;
                const uint32_t* vp = sV + u * kPitch;
#pragma unroll
                for (int p = 0; p < kPairs; ++p) {
                    const uint32_t q2 = vp[p];
                    const int d = 2 * p;
                    acc[d] = acc[d] + w * f16_to_f32((uint16_t)(q2 & 0xffffu));
                    acc[d + 1] = acc[d + 1] + w * f16_to_f32((uint16_t)(q2 >> 16));
                }
            }
        }
    }

    /* A row slot past its sequence's length or the query plane is a clamped duplicate and stores
     * nothing, partial store included. Its `row` clamps to a real row, so an unguarded partial store
     * would add a second writer to (row, h, slice), inside the arena where no bounds check sees it;
     * the two 132-float records would interleave, the reduce would rescale one lane's accumulator by
     * another lane's maximum, and the output would be badly wrong. The reduce reads only real rows
     * (`nrows * n_head`), so slots a padding row leaves unwritten are never read. */
    if (!row_ok) return;

    if (SPLIT) {
        float* pp = A.part + ((size_t)row * A.n_head + h) * ((size_t)A.splits * kPartStride)
                    + (size_t)slice * kPartStride;
#pragma unroll
        for (int u = 0; u < kHeadDim / 8; ++u) {
            const r3d_f32x4 v = { acc[8 * u], acc[8 * u + 1], acc[8 * u + 2], acc[8 * u + 3] };
            const r3d_f32x4 w = { acc[8 * u + 4], acc[8 * u + 5], acc[8 * u + 6], acc[8 * u + 7] };
            *(r3d_f32x4*)(pp + 8 * u) = v;
            *(r3d_f32x4*)(pp + 8 * u + 4) = w;
        }
        pp[kHeadDim] = sum;
        pp[kHeadDim + 1] = mx;
        return;
    }

    uint16_t* op = A.out + row * (long long)A.o_rs + h * kHeadDim;
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;           /* no visible key: a zero row */
#pragma unroll
    for (int u = 0; u < kHeadDim / 8; ++u) {
        uint32_t q4[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = 8 * u + 2 * j;
            q4[j] = bf16x2_rne(acc[d] * inv, acc[d + 1] * inv);
        }
        *(r3d_u32x4*)(void*)(op + 8 * u) = r3d_u32x4{ q4[0], q4[1], q4[2], q4[3] };
    }
}

/* ------------------------------------------------------------------ split-KV reduce
 *
 * A wave per (query row, head): 32 lanes x 4 dims. The slices combine ascending, each rescaled by
 * exp(m_slice - M) with M the slices' maximum, which is where split-KV leaves the reference's
 * parentheses; see the header. */
__global__ __launch_bounds__(128) R3D_WAVES(4)
void attn_reduce(const Args A, long long nrows)
{
    const int w = (int)threadIdx.x >> 5, lane = (int)threadIdx.x & 31;
    const long long idx = (long long)blockIdx.x * 4 + w;
    const long long total = nrows * A.n_head;
    if (idx >= total) return;
    const long long row = idx / A.n_head, h = idx - row * A.n_head;
    const int s = seq_of_row(A, (int)row);            /* row / q_len with a null descriptor */
    const long long ctx = ld_ix(A.su, (size_t)s);
    const int jend_all = (int)(ctx < A.jcap ? ctx : A.jcap);
    const int nblk = (jend_all + kBlock - 1) / kBlock;
    const int tps = (nblk + A.splits - 1) / A.splits;
    const int nslice = tps > 0 ? (nblk + tps - 1) / tps : 0;

    float* pp = A.part + (size_t)idx * ((size_t)A.splits * kPartStride);
    float M = -INFINITY;
    for (int u = 0; u < nslice; ++u) {
        const float m = pp[(size_t)u * kPartStride + kHeadDim + 1];
        if (m > M) M = m;
    }
    uint16_t* o = A.out + row * (long long)A.o_rs + h * kHeadDim;
    const int d0 = lane * 4;
    float a[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float sum = 0.0f;
    if (M > -INFINITY) {
        for (int sl = 0; sl < nslice; ++sl) {
            const float* q = pp + (size_t)sl * kPartStride;
            const float f = exp_ref(q[kHeadDim + 1] - M);
            sum += q[kHeadDim] * f;
            a[0] += q[d0 + 0] * f;
            a[1] += q[d0 + 1] * f;
            a[2] += q[d0 + 2] * f;
            a[3] += q[d0 + 3] * f;
        }
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    *(r3d_u32x2*)(void*)(o + d0) = r3d_u32x2{ bf16x2_rne(a[0] * inv, a[1] * inv),
                                              bf16x2_rne(a[2] * inv, a[3] * inv) };
}

}  // namespace

/* ================================================================== launchers
 *
 * Both entries take raw addresses and extents and assume the dense layouts the shim has checked:
 * the cache at [blocks, kv_heads, 16, 256] and seqused one i32 a sequence. The shims refuse
 * anything else (a strided cache, an i64 index plane). The block table's columns and strides are
 * arguments.
 *
 * With `cu` (the ragged descriptor), `q_len` is the query plane's total row count, which
 * cu[num_seqs] states and libref reads off the tensor (`rows = numel(q) / (q_heads * head_dim)`);
 * without it, `q_len` is per sequence and the total is num_seqs * q_len. No entry reads the
 * descriptor: the grid is sized on a bound and the kernel does the walking.
 */
#define R3D_NEED(cond) do { if (!(cond)) return R3D_NO(RAD_E_SHAPE); } while (0)

namespace {

int fill(Args& A, long q, long kv, long bt, long seqused, long k_desc, long v_desc, long cu,
         long rows, int num_seqs, int q_len, int head_dim, int gqa, int n_head, int block_size,
         int bt_cols, int bt_rs, int bt_cs, int max_ctx, int window, int causal, float scale,
         int kv_dt, long q_ld, long out_ld, int splits, long scratch, int waves)
{
    A = Args{};
    if (head_dim != kHeadDim) return -2;                            /* RAD_E_SHAPE */
    if (num_seqs <= 0 || n_head <= 0) return -2;
    /* A rectangular batch is counted per sequence; a ragged one has no per-sequence length, and its
     * `rows` is the query plane's row count. Both have to be at least one row. */
    if (cu) { if (rows < 1 || rows > 2147483000LL) return -2; }
    else if (q_len < 1 || (long long)num_seqs * q_len > 2147483000LL) return -2;
    if (block_size != kBlock) return -2;
    if (gqa != 1 && gqa != 2 && gqa != 4 && gqa != 8) return -2;    /* the wave split is a shift */
    if (n_head % gqa) return -2;
    if (window < 0 || max_ctx < 1) return -2;
    if (kv_dt != RAD_BF16 && kv_dt != RAD_F8E4M3) return -1;        /* RAD_E_DTYPE */
    if (!q || !kv || !bt || !seqused) return -1;

    const int kv_heads = n_head / gqa;
    /* Two different counts. The cache holds ceil(max_ctx / block_size) blocks, which bounds a slot
     * address. The table has `bt_cols` columns, sized by the engine to the step (a five-token prompt
     * with nothing behind it is a [n_seq, 1] table). The walk is bounded by the table, as in the
     * reference (max_blocks from `bt->shape[last]`, row pitch `stride[rank-2]`, column stride
     * `stride[rank-1]`, jend = min(ctx, max_blocks * block_size)), so a context longer than the
     * table is truncated to the same keys. */
    const int cache_blocks = (max_ctx + block_size - 1) / block_size;
    if (bt_cols < 1 || bt_rs < 1 || bt_cs < 1) return -2;   /* the shim refuses this first, RAD_E_STRIDE */

    A.q = (const uint16_t*)(uintptr_t)q;
    A.kv = kv_dt == RAD_BF16 ? (const uint16_t*)(uintptr_t)kv : nullptr;
    A.kv8 = kv_dt == RAD_F8E4M3 ? (const uint8_t*)(uintptr_t)kv : nullptr;
    A.bt = (const int*)(uintptr_t)bt;
    A.su = (const int*)(uintptr_t)seqused;
    A.cu = (const int*)(uintptr_t)cu;
    A.kds = (const float*)(uintptr_t)k_desc;
    A.vds = (const float*)(uintptr_t)v_desc;
    /* The dense cache: a slot's K is 128 elements, V the 128 after it. Every stride has to fit the
     * 32-bit form the kernel's addressing uses. */
    A.so = 2 * kHeadDim;
    A.sh = block_size * A.so;
    A.sb = kv_heads * A.sh;
    A.sd = 1;
    if ((long long)A.sb * cache_blocks > 2147483000LL) return -2;
    if (q_ld > 2147483000LL || out_ld > 2147483000LL) return -2;
    A.q_rs = (int)q_ld;
    A.o_rs = (int)out_ld;
    A.bt_rs = bt_rs;
    A.bt_cs = bt_cs;
    A.n_seq = num_seqs; A.q_len = cu ? 0 : q_len;
    A.rows = cu ? (int)rows : num_seqs * q_len;
    A.n_head = n_head; A.kv_heads = kv_heads;
    A.causal = causal ? 1 : 0; A.window = window;
    A.jcap = max_ctx;
    if ((long long)bt_cols * block_size < max_ctx) A.jcap = bt_cols * block_size;
    /* RAD_R3D_TRACE: the table geometry and the walk's reach (jcap). A short jcap means the table
     * is narrower than the sequence, a plan fact rather than a kernel one. */
    if (r3d_trace_on())
        std::fprintf(stderr, "[r3d] attn: q_len %d rows %ld n_seq %d bs %d bt cols %d pitch %d "
                             "cstride %d max_ctx %d -> jcap %d\n", q_len, rows, num_seqs,
                     block_size, bt_cols, bt_rs, bt_cs, max_ctx, A.jcap);
    {   /* RAD_R3D_TRACE: a one-record hipMalloc buffer the kernel writes and launch() reads back,
         * allocated on first traced use. Null when the flag is unset. */
        static void* g_trace_buf = nullptr;
        A.trace_buf = nullptr;
        if (r3d_trace_on() && !g_trace_buf)
            (void)hipMalloc(&g_trace_buf, sizeof(DevRec));
        if (g_trace_buf) A.trace_buf = (DevRec*)g_trace_buf;
    }
    /* RAD_R3D_TRACE: synchronise the device and print, for up to 4 sequences, the first 16 block
     * table entries ("[r3d] devstate s=.. bt=..") and seqused with the KV base, then the first
     * block id. The device sync is needed because staging may run on the engine's stream, and a
     * copy on the default stream is unordered with it. */
    if (r3d_trace_on())
    {
        (void)hipDeviceSynchronize();
        const char* kvb = A.kv8 ? (const char*)A.kv8 : (const char*)A.kv;
        for (int s = 0; s < (int)num_seqs && s < 4; ++s) {
            int row[16];
            if (hipMemcpy(row, (const char*)A.bt + (size_t)s * (size_t)bt_rs * sizeof(int),
                          16 * sizeof(int), hipMemcpyDeviceToHost) == hipSuccess) {
                std::fprintf(stderr, "[r3d] devstate s=%d bt=", s);
                for (int c = 0; c < 16; ++c) std::fprintf(stderr, "%s%d", c ? " " : "", row[c]);
                std::fprintf(stderr, "\n");
            }
            int su = -1;
            if (A.su &&
                hipMemcpy(&su, (const char*)A.su + (size_t)s * sizeof(int), sizeof(int),
                          hipMemcpyDeviceToHost) == hipSuccess)
                std::fprintf(stderr, "[r3d] devstate s=%d seqused=%d kvbase=%p\n", s, su,
                             (const void*)kvb);
        }
        int first = -1;
        if (kvb &&
            hipMemcpy(&first, (const char*)A.bt, sizeof(int), hipMemcpyDeviceToHost) == hipSuccess &&
            first >= 0) {
            /* Only the block id is printed: a D2H copy from the cache base faults in the engine
             * process even where the kernel's own read is fine. */
            std::fprintf(stderr, "[r3d] devstate first_block=%d (block content not read)\n",
                         first);
        }
    }
    A.scale = scale > 0.0f ? scale : 1.0f / std::sqrt((float)kHeadDim);
    A.splits = splits > 0 ? splits : 1;
    A.waves = waves;
    A.part = A.splits > 1 ? (float*)(uintptr_t)scratch : nullptr;
    if (A.splits > 1 && !scratch) return -5;                        /* RAD_E_SCRATCH */
    return 0;
}

/* The narrowest workgroup that carries the caller's q_len (see "the width of a workgroup"). A wave
 * carries 32 / gqa query rows, and a slot past q_len computes a clamped duplicate. The tile is the
 * same 8 KiB at every width, so a narrower workgroup spreads the same cache reads over more
 * workgroups, the occupancy the op lacks at decode. Prefill keeps four waves: its q_len is the
 * whole chunk and every slot is used. The accumulation order does not depend on the width: the
 * softmax sum is over keys, ascending, inside one lane. */
int pick_waves(int q_len, int gqa)
{
    const int rows = 32 / (gqa > 0 ? gqa : 1);
    const int w = (q_len + rows - 1) / rows;
    return w <= 1 ? 1 : (w <= 2 ? 2 : 4);      /* the ladder the grid is cut on: 1, 2, 4 */
}

/* The rows per sequence as a host can know them: the plane's row count over the sequence count,
 * rounded up. With a descriptor the true lengths are device memory, so the width and split count of
 * a ragged step are picked on this average. Both choices only move duplicated row slots and grid
 * blocks, never a result. */
int avg_rows(int rows, int num_seqs) { return num_seqs > 0 ? (rows + num_seqs - 1) / num_seqs : rows; }

/* Every width pick_waves can return must be instantiated at every gqa: gqa = 4 asks for W = 4 at
 * q_len > 16, gqa = 2 at q_len > 32, gqa = 1 at q_len > 64, and the prefill entry always passes
 * kWaves = 4. The launcher dispatches through `if constexpr (waves_ok<...>::value)`, so a missing
 * instantiation silently compiles the launch out: the entry returns RAD_OK and the output keeps
 * whatever the caller put there. waves_ok therefore admits the whole ladder. */
template <int GQA, int W> struct waves_ok {
    static const bool value = (W == 1) || (W == 2) || (W == 4);
};

int tok_per_block(int gqa, int waves) { return waves * (32 / (gqa > 0 ? gqa : 1)); }

/* The decode entry's `splits` when the caller asked for the widest legal: enough workgroups to fill
 * a 16-CU card (two workgroups a CU at four waves a SIMD) and no more, on a doubling ladder. */
int pick_splits(int tiles, int num_seqs, int kv_heads)
{
    const long long blocks = (long long)tiles * num_seqs * kv_heads;
    int sp = 1;
    while (sp < 8 && blocks * sp < 32) sp *= 2;
    return sp;
}

static void dev_speak(const DevRec& r);   /* prints a DevRec; defined with the ring below */

int launch(const Args& A, long out, long stream)
{
    Args B = A;
    B.out = (uint16_t*)(uintptr_t)out;
    const int gqa = A.n_head / A.kv_heads;
    const int waves = A.waves > 0 ? A.waves : kWaves;
    const int tpb = tok_per_block(gqa, waves);
    /* The grid's row count. Rectangular: every sequence has the same tile count, so the product is
     * exact. Ragged: the tile count sum_s ceil(len_s / tpb) is in device memory, so the launch uses
     * the bound
     *     sum_s ceil(len_s / tpb)  <=  ceil(rows / tpb) + n_seq - 1
     * (ceil(x) + ceil(y) <= ceil(x + y) + 1, by induction over the sequences), and `map_seq`
     * returns early for a grid row past the last tile. The slack is at most n_seq - 1 empty
     * workgroups, each reading n_seq descriptor entries and returning. */
    const int tiles = A.cu ? (A.rows + tpb - 1) / tpb + A.n_seq - 1
                           : (A.q_len + tpb - 1) / tpb * A.n_seq;
    dim3 grid((unsigned)tiles, (unsigned)A.kv_heads,
              (unsigned)(A.splits > 1 ? A.splits : 1));
    const bool fp8 = A.kv8 != nullptr;
    const bool causal = A.causal != 0;
    hipStream_t st = (hipStream_t)(uintptr_t)stream;

#define R3D_ATTN(FP8V, CAUSALV, SPLITV, GQAV, WAVESV)                          \
    R3D_KLAUNCH((attn_kernel<FP8V, CAUSALV, SPLITV, GQAV, WAVESV>), grid,       \
                dim3(WAVESV * 32), st, B)

#define R3D_ATTN_SEL(GQAV, WAVESV)                                             \
    do {                                                                       \
        if constexpr (waves_ok<GQAV, WAVESV>::value) {                          \
            /* `sel` is three bits and all eight combinations are reachable (a single-sequence    \
             * causal decode picks splits 4 itself: pick_splits(1, 1, 8) = 4). Each needs its own   \
             * arm: a causal split launch routed to the non-causal instantiation would take its     \
             * maximum over keys the row must not see. */                                           \
            if (sel == 0) R3D_ATTN(false, false, false, GQAV, WAVESV);         \
            else if (sel == 1) R3D_ATTN(true, false, false, GQAV, WAVESV);     \
            else if (sel == 2) R3D_ATTN(false, true, false, GQAV, WAVESV);     \
            else if (sel == 3) R3D_ATTN(true, true, false, GQAV, WAVESV);      \
            else if (sel == 4) R3D_ATTN(false, false, true, GQAV, WAVESV);     \
            else if (sel == 5) R3D_ATTN(true, false, true, GQAV, WAVESV);      \
            else if (sel == 6) R3D_ATTN(false, true, true, GQAV, WAVESV);      \
            else R3D_ATTN(true, true, true, GQAV, WAVESV);                     \
        }                                                                      \
    } while (0)

#define R3D_ATTN_GQA(GQAV)                                                     \
    do {                                                                       \
        if (waves == 1) R3D_ATTN_SEL(GQAV, 1);                                 \
        else if (waves == 2) R3D_ATTN_SEL(GQAV, 2);                            \
        else R3D_ATTN_SEL(GQAV, 4);                                            \
    } while (0)

    const int sel = (fp8 ? 1 : 0) | (causal ? 2 : 0) | (A.splits > 1 ? 4 : 0);
    /* Clear the error flag here, after the arguments are staged and before the chevron, and read
     * it right after. `hipGetLastError()` reports the first error since it was last cleared, so
     * without the clear this launch would be blamed for an earlier op's error.
     * HIP_LAUNCH_BLOCKING=1 hides this kind of misattribution, because a synchronous launch clears a
     * pending asynchronous error before the check. */
    (void)hipGetLastError();
    switch (gqa) {
    case 1: R3D_ATTN_GQA(1); break;
    case 2: R3D_ATTN_GQA(2); break;
    case 4: R3D_ATTN_GQA(4); break;
    default: R3D_ATTN_GQA(8); break;
    }
#undef R3D_ATTN_GQA
#undef R3D_ATTN_SEL
#undef R3D_ATTN
    {
        const hipError_t ke = hipGetLastError();
        if (ke != hipSuccess) {
            /* Under RAD_R3D_TRACE, a launch failure prints the error, the grid, the workgroup
             * size and the chosen width. */
            if (r3d_trace_on())
                std::fprintf(stderr, "[r3d] attn: %s on the attn launch: grid (%u, %u, %u) block "
                                     "%d waves %d gqa %d splits %d rows %d n_seq %d jcap %d\n",
                             hipGetErrorName(ke), grid.x, grid.y, grid.z, waves * 32, waves, gqa,
                             A.splits, A.rows, A.n_seq, A.jcap);
            return -4;                                          /* RAD_E_DEVICE */
        }
    }

    if (B.trace_buf) {   /* RAD_R3D_TRACE: sync the stream and print this launch's record */
        (void)hipStreamSynchronize(st);
        DevRec rec;
        if (hipMemcpy(&rec, (const void*)B.trace_buf, sizeof rec, hipMemcpyDeviceToHost) == hipSuccess)
            dev_speak(rec);
    }

    if (A.splits > 1) {
        const long long rows = A.rows;              /* the plane's rows: cu[n_seq], or n_seq * q_len */
        const long long total = rows * A.n_head;
        dim3 rg((unsigned)((total + 3) / 4));
        (void)hipGetLastError();                                /* clear first, as above */
        R3D_KLAUNCH(attn_reduce, rg, dim3(128), st, B, rows);
        const hipError_t re = hipGetLastError();
        if (re != hipSuccess) {
            if (r3d_trace_on())
                std::fprintf(stderr, "[r3d] attn: %s on the reduce launch: grid (%u, 1, 1) block "
                                     "128 rows %lld heads %d splits %d\n", hipGetErrorName(re),
                             rg.x, rows, A.n_head, A.splits);
            return -4;
        }
    }
    return 0;
}

}  // namespace

extern "C" int r3d_attn_decode_h128(long q, long kv, long bt, long seqused, long k_desc,
                                    long v_desc, long cu, long out, int num_seqs, int q_len,
                                    int head_dim, int gqa, int n_head, int block_size, int bt_cols, int bt_rs, int bt_cs, int max_ctx,
                                    int window, int causal, float scale, int kv_dt, long q_ld,
                                    long out_ld, int splits, long scratch, long stream)
{
    /* With `cu` (the ragged descriptor) `q_len` is the query plane's row count (see the launchers'
     * header) and a sequence's rows are cu[s+1] - cu[s]. The width and split count are picked on
     * the average sequence because the descriptor is device memory; both only change how many row
     * slots are duplicated, never a result. */
    if (splits < 0) return -2;
    if (gqa <= 0 || n_head <= 0 || n_head % gqa) return -2;
    if (cu && num_seqs <= 0) return -2;
    Args A;
    const int kv_heads = n_head / gqa;
    const int per = cu ? avg_rows(q_len, num_seqs) : (q_len > 0 ? q_len : 1);
    const int waves = pick_waves(per, gqa);
    const int tpb = tok_per_block(gqa, waves);
    const int tiles = per > 0 ? (per + tpb - 1) / tpb : 1;
    int sp = splits > 0 ? splits : pick_splits(tiles, num_seqs, kv_heads);
    const int e = fill(A, q, kv, bt, seqused, k_desc, v_desc, cu, cu ? q_len : 0, num_seqs,
                       cu ? 0 : q_len, head_dim, gqa, n_head, block_size, bt_cols, bt_rs, bt_cs,
                       max_ctx, window, causal, scale, kv_dt, q_ld, out_ld, sp, scratch, waves);
    if (e) return e;
    return launch(A, out, stream);
}

extern "C" int r3d_attn_prefill_h128(long q, long kv, long bt, long seqused, long k_desc,
                                     long v_desc, long cu, long out, int num_seqs, int q_len,
                                     int head_dim, int gqa, int n_head, int block_size, int bt_cols, int bt_rs, int bt_cs,
                                     int max_ctx, int window, int causal, float scale, int kv_dt,
                                     long q_ld, long out_ld, long scratch, long stream)
{
    (void)scratch;                       /* the prefill kernel has no split pass, so no partials */
    /* Four waves with or without a descriptor. A rectangular prefill's q_len is the whole chunk, so
     * every slot is used. For a ragged step with few rows the number of waves on the GPU matters
     * more than unused slots: a 5-row step at gqa 8 is one workgroup per KV head, and four waves run
     * faster than two (16.7 vs 19.3 ms at 2048 keys), since a wave walks its tile once whatever rows
     * it carries. A step with one row per sequence is a decode step and goes to the decode entry,
     * where a narrow workgroup pays. */
    Args A;
    const int waves = kWaves;
    const int e = fill(A, q, kv, bt, seqused, k_desc, v_desc, cu, cu ? q_len : 0, num_seqs,
                       cu ? 0 : q_len, head_dim, gqa, n_head, block_size, bt_cols, bt_rs, bt_cs, max_ctx, window, causal,
                       scale, kv_dt, q_ld, out_ld, 1, 0, waves);
    if (e) return e;
    A.splits = 1;
    A.part = nullptr;
    return launch(A, out, stream);
}

/* ================================================================== shims
 *
 * The paged geometry as the tensors state it: n_seq from `seqused`, n_head and head_dim from the
 * query, kv_heads from the cache, block_size from the cache, max_ctx from the cache's block count,
 * q_len from the query's row count over n_seq. Params fill in what the tensors cannot say (gqa,
 * causal, window, scale). */
namespace {

enum { P_Q = 0, P_KV = 1, P_BT = 2, P_SU = 3, P_KDS = 4, P_VDS = 5, P_CU = 6, P_OUT = 7 };

/* ---- arming the device record ring ----------------------------------------------------------
 *
 * The ring's four globals are plugin-owned, at file scope, and `extern "C"` so the code object
 * carries these names. The shim's interface to them, all under RAD_R3D_TRACE:
 *
 *   hipGetSymbolAddress of the budget, once  -- checks the symbol resolves before arming
 *   hipMemcpyToSymbol of the budget          -- RAD_R3D_DEV records (default 64; 0 records nothing)
 *   hipMemcpyFromSymbol of the counter       -- quoted on the plan line as `dev_launches=`
 *   hipMemcpyFromSymbol of the ring          -- the drain: records printed as "[r3d dev]" lines
 *   hipMemcpyToSymbol of the write index     -- the drain's reset, so each record prints once
 *
 * The drain and the arming run on every traced shim call, drain first. The budget is rewritten on
 * each call, so a caller can arm an exact count between launches (arm 3, launch 10, get 3) and
 * the last write wins. A replayed step makes no shim call, so its records print at the next call,
 * one step late; `L` and `ctx` restore the order.
 *
 * With the flag unset, `dev_ring_drain` and `dev_trace_arm` return before any runtime call and the
 * budget stays 0.
 */
enum { kDevPrintDefault = 64 };

template <typename T>
void* dev_sym(void** slot, const char* name, const T& symbol)
{
    if (*slot) return *slot;
    void* p = nullptr;
    const hipError_t e = hipGetSymbolAddress(&p, HIP_SYMBOL(symbol));   /* the T& overload */
    *slot = (e == hipSuccess) ? p : nullptr;
    if (e != hipSuccess)
        std::fprintf(stderr, "[r3d dev] %s: hipGetSymbolAddress -> %s\n", name,
                     hipGetErrorName(e));
    return *slot;
}

void* g_dev_budget = nullptr;                      /* hipGetSymbolAddress cache, resolved once */
void* g_dev_launches = nullptr;

unsigned long long dev_launch_count()
{
    /* `hipMemcpyFromSymbol`, not `hipGetSymbolAddress` + `hipMemcpy`: in the engine process
     * `hipGetSymbolAddress` on a dlopened plugin's symbol can return an address the kernel's module
     * does not write, so the read would only ever see the initial value. The read must resolve the
     * symbol the same way the arming's `hipMemcpyToSymbol` does. */
    unsigned long long n = 0;
    (void)hipMemcpyFromSymbol(&n, HIP_SYMBOL(r3d_dev_launches), sizeof n);
    return n;
}

void* g_dev_widx = nullptr;
void* g_dev_ring = nullptr;

/* Prints one record as a "[r3d dev] L=... ids=... idsum=..." line; the field names and order are
 * a format log tooling parses. `kh` folds the first block walked, `khL` the last, and `idsum` every
 * table column. */
static void dev_speak(const DevRec& r)
{
    std::fprintf(stderr, "[r3d dev] L=%llu dec=%d n_seq=%d rows=%d q_len=%d s=%d len=%d span=%d "
                 "start=%d ctx=%lld pos=%lld jcap=%d splits=%d waves=%d kv=%p bt=%p su=%p cu=%p "
                 "q=%p out=%p bt0=%d bt1=%d bt2=%d slot0=%lld qh=%016llx kh=%016llx ids=",
                 (unsigned long long)r.L, r.dec, r.n_seq, r.rows, r.q_len, r.s, r.len, r.span,
                 r.start, (long long)r.ctx, (long long)r.pos, r.jcap, r.splits, r.waves,
                 (const void*)r.kv, (const void*)r.bt, (const void*)r.su, (const void*)r.cu,
                 (const void*)r.q, (const void*)r.out, r.b[0], r.b[1], r.b[2], (long long)r.slot0,
                 (unsigned long long)r.qh, (unsigned long long)r.kh);
    for (int c = 0; c < r.ncols && c < kDevIds; ++c)
        std::fprintf(stderr, "%s%d", c ? "," : "", r.b[c]);
    std::fprintf(stderr, " ncols=%d bL=%d slotL=%lld khL=%016llx idsum=%lld rowok=%d\n", r.ncols,
                 r.spare, (long long)r.slotL, (unsigned long long)r.khL, (long long)r.idsum,
                 r.rowok);
}

/* Copies at most kDevRing records (55 KiB) from the ring and prints them, then resets the write
 * index to 0 so each record prints once. The index is the kernel's own, so the drain never reads
 * past what the kernel wrote. The launch counter is not touched. */
void dev_ring_drain()
{
    if (!r3d_trace_on()) return;
    /* Read through `hipMemcpyFromSymbol` for the reason given at `dev_launch_count`. */
    unsigned w = 0;
    if (hipMemcpyFromSymbol(&w, HIP_SYMBOL(r3d_dev_widx), sizeof w) != hipSuccess) return;
    if (w == 0) return;
    static DevRec buf[kDevRing];
    const unsigned keep = w < (unsigned)kDevRing ? w : (unsigned)kDevRing;
    if (hipMemcpyFromSymbol(buf, HIP_SYMBOL(r3d_dev_ring), (size_t)keep * sizeof(DevRec)) != hipSuccess)
        return;
    for (unsigned i = 0; i < keep; ++i) dev_speak(buf[i]);
    if (w > keep)
        std::fprintf(stderr, "[r3d dev] ring: %u records written, %u kept, %u counted and dropped\n", w, keep, w - keep);
    const unsigned zero = 0u;
    (void)hipMemcpyToSymbol(HIP_SYMBOL(r3d_dev_widx), &zero, sizeof zero);
}

void dev_trace_arm()
{
    if (!r3d_trace_on()) return;                   /* RAD_R3D_TRACE unset: no-op */
    int want = kDevPrintDefault;
    if (const char* e = std::getenv("RAD_R3D_DEV")) {
        const int v = std::atoi(e);
        if (v > 0) want = v;                       /* an exact count; 0 disables */
        else if (v == 0) want = 0;
    }
    static bool said = false;
    const unsigned long long seen = dev_launch_count();
    if (!dev_sym(&g_dev_budget, "r3d_dev_budget", HIP_SYMBOL(r3d_dev_budget))) return;
    const hipError_t e = hipMemcpyToSymbol(HIP_SYMBOL(r3d_dev_budget), &want, sizeof want);
    if (!said || e != hipSuccess) {
        said = true;
        std::fprintf(stderr, "[r3d dev] armed: budget=%d launches_at_arm=%llu %s\n", want,
                     (unsigned long long)seen, e == hipSuccess ? "ok" : hipGetErrorName(e));
    }
}


/* ---- the plan line ----------------------------------------------------------------------------
 *
 * Under RAD_R3D_TRACE, prints one "[r3d] plan:" line per distinct shape: the batch geometry, the
 * walk's reach (jcap), the chosen width and splits, `seqused[0]`, the operand addresses, the
 * kernel's launch counter, the descriptor (`cu[0]`, `cu[n_seq]`, all of it when n_seq + 1 <= 8) and
 * the first row of the block table. The launcher never reads `cu_seqlens`, so this is the only
 * place its contents are visible.
 *
 * Every read is bounded. The descriptor holds exactly `n_seq + 1` ints (checked by the shim) and
 * the copy is capped at `kCuPrint`, with the last entry read separately when the plane is wider.
 * `seqused` is read for its first int only. The descriptor is read only after `r3d_has(CU)`, and
 * `SU->data` is required at the top of the shim. `hipMemcpyDefault` works for VRAM and host-mapped
 * planes alike.
 *
 * The latch is keyed on the shape (n_seq, the plane's rows, the kernel's q_len, the table's
 * columns) because the shim runs once a layer a step. It is a plain static array with no lock: a
 * race can at worst print a shape twice. */
enum { kCuPrint = 64, kPlanShapes = 512 };

/* RAD_R3D_TRACE=2: the same plan line with ctx added to the shape key (see plan_once). Read once. */
bool r3d_trace_wide()
{
    static int wide = -1;
    if (wide < 0) {
        const char* e = std::getenv("RAD_R3D_TRACE");
        wide = (e && e[0] == '2') ? 1 : 0;
    }
    return wide != 0;
}

/* The plan line's shape key: n_seq, the plane's rows, the kernel's q_len, the table's columns and,
 * under RAD_R3D_TRACE=2 only, `ctx`. Serving requests usually share one shape and differ only in
 * context depth, so the narrow key prints the block table once, for the first request; the wide
 * key prints again as the context grows. */
struct PlanKey { long long n_seq, rows, q_len, cols, ctx; };

bool plan_once(long long n_seq, long long rows, long long q_len, long long cols, long long ctx,
               bool by_ctx)
{
    static PlanKey seen[kPlanShapes];
    static int n = 0;
    for (int i = 0; i < n; ++i)
        if (seen[i].n_seq == n_seq && seen[i].rows == rows && seen[i].q_len == q_len &&
            seen[i].cols == cols && (!by_ctx || seen[i].ctx == ctx))
            return false;
    if (n < kPlanShapes) seen[n++] = PlanKey{n_seq, rows, q_len, cols, ctx};
    return true;
}

void plan_line(bool decode, const RadTensor* Q, const RadTensor* KV, const RadTensor* O,
               const RadTensor* CU, const RadTensor* SU, const RadTensor* BT, const void* scratch,
               long long n_seq, long long rows, long long q_len, long long bt_cols, long long bt_rs,
               long long bt_cs, int block_size, int max_ctx, int waves, int splits, bool by_ctx)
{
    /* The walk's bound, the same expression fill() uses: the TABLE's columns times the block,
     * capped by the cache's context (jend = min(ctx, max_blocks * block_size)). */
    const long long reach = (long long)bt_cols * block_size;
    const long long jcap = reach < max_ctx ? reach : (long long)max_ctx;

    long long ctx = -1;                              /* seqused[0], the first sequence's context */
    int ctx0 = 0;
    if (r3d_p(SU) && hipMemcpy(&ctx0, r3d_p(SU), sizeof(int), hipMemcpyDefault) == hipSuccess)
        ctx = ctx0;

    /* The latch comes after the ctx read because the wide key includes ctx. */
    if (!plan_once(n_seq, rows, q_len, bt_cols, ctx, by_ctx)) return;

    std::fprintf(stderr,
                 "[r3d] plan: %s n_seq=%lld rows=%lld q_len=%lld cu rank=%u numel=%lld ctx=%lld "
                 "bt cols=%lld pitch=%lld max_ctx=%d -> jcap=%lld waves=%d splits=%d",
                 decode ? "decode" : "prefill", n_seq, rows, q_len,
                 r3d_has(CU) ? (unsigned)CU->rank : 0u, r3d_has(CU) ? r3d_numel(CU) : 0LL, ctx,
                 bt_cols, bt_rs, max_ctx, jcap, waves, splits);

    /* The addresses the launch bakes into its arguments. A replayed graph re-runs the kernels with
     * the arguments frozen at capture, so if a later request's planes live at other addresses the
     * replay reads the earlier request's planes. Printed with `%p`, nothing dereferenced. */
    std::fprintf(stderr, " q=%p kv=%p bt=%p cu=%p su=%p out=%p part=%p",
                 r3d_p(Q), r3d_p(KV), r3d_p(BT), r3d_p(CU), r3d_p(SU), r3d_p(O), scratch);

    /* The kernel's own launch counter, so the shim's line and the kernel's records line up; the
     * shim sees only eagerly issued steps, the kernel every launch. One 8-byte device read. */
    std::fprintf(stderr, " dev_launches=%llu", (unsigned long long)dev_launch_count());

    if (!r3d_has(CU)) {           /* no descriptor: a rectangular step */
        std::fprintf(stderr, " cu=absent (rectangular)");
    } else {
        int cuh[kCuPrint];
        const int take = (int)(n_seq + 1 < kCuPrint ? n_seq + 1 : kCuPrint);
        if (hipMemcpy(cuh, r3d_p(CU), (size_t)take * sizeof(int), hipMemcpyDefault) !=
            hipSuccess) {
            std::fprintf(stderr, " cu=copy failed");
        } else {
            long long cu_last = -1;
            if (n_seq + 1 <= kCuPrint) cu_last = cuh[(int)n_seq];
            else {
                int last = 0;                        /* the plane's last int, nothing beyond it */
                if (hipMemcpy(&last, (const int*)r3d_p(CU) + n_seq, sizeof(int),
                              hipMemcpyDefault) == hipSuccess)
                    cu_last = last;
            }
            std::fprintf(stderr, " cu[0]=%d cu[n_seq]=%lld", cuh[0], cu_last);
            if (n_seq + 1 <= 8)                      /* the whole descriptor, when it is short */
                for (int i = 0; i <= (int)n_seq; ++i)
                    std::fprintf(stderr, " cu[%d]=%d", i, cuh[i]);
        }
    }

    /* The block table entries of the first row the step walks: a reused block id inside one row,
     * or an id the cache does not have, is otherwise invisible. Bounded to `kCuPrint` columns of row
     * 0, only columns the shim proved the plane covers, plus row 1's first entry when there is a
     * row 1. An unmapped block prints as -1, as the kernel sees it. */
    if (r3d_has(BT)) {
        const int take = (int)(bt_cols < kCuPrint ? bt_cols : kCuPrint);
        int bth[kCuPrint];
        const int* btp = (const int*)r3d_p(BT);
        bool got = true;
        for (int i = 0; i < take; ++i)
            if (hipMemcpy(&bth[i], btp + (long long)i * bt_cs, sizeof(int),
                          hipMemcpyDefault) != hipSuccess) { got = false; break; }
        std::fprintf(stderr, " bt0=");
        if (!got) std::fprintf(stderr, "(copy failed)");
        else {
            std::fprintf(stderr, "[");
            for (int i = 0; i < take; ++i) std::fprintf(stderr, "%s%d", i ? "," : "", bth[i]);
            std::fprintf(stderr, "]");
            if (take < bt_cols) std::fprintf(stderr, " (%lld cols)", bt_cols);
            if (n_seq >= 2) {
                int r1 = 0;
                if (hipMemcpy(&r1, btp + bt_rs, sizeof(int), hipMemcpyDefault) == hipSuccess)
                    std::fprintf(stderr, " bt1[0]=%d", r1);
            }
        }
    }
    std::fprintf(stderr, "\n");
}

int attn_shim(const RadArgs* a, RadStream s, bool decode)
{
    if (a->n_t < 8) return R3D_NO(RAD_E_INVAL);
    const RadTensor* Q = r3d_opt(a, P_Q);
    const RadTensor* KV = r3d_opt(a, P_KV);
    const RadTensor* BT = r3d_opt(a, P_BT);
    const RadTensor* SU = r3d_opt(a, P_SU);
    const RadTensor* O = r3d_opt(a, P_OUT);
    const RadTensor* KDS = r3d_opt(a, P_KDS);
    const RadTensor* VDS = r3d_opt(a, P_VDS);
    const RadTensor* CU = r3d_opt(a, P_CU);
    if (!Q || !KV || !BT || !SU || !O) return R3D_NO(RAD_E_INVAL);
    if (!Q->data || !KV->data || !BT->data || !SU->data || !O->data) return R3D_NO(RAD_E_INVAL);
    /* `cu_seqlens` present means a ragged batch (see "the ragged batch" in the header). The shim
     * can check the descriptor's shape and dtype; what it cannot see (cu[0] == 0, monotonicity,
     * cu[n_seq] == the plane's row count) the kernel clamps, as libref's row loop does. */
    const bool ragged = r3d_has(CU);

    const int head_dim = (int)r3d_geti_or(a, "head_dim", r3d_dim(Q, (int)Q->rank - 1));
    if (head_dim != kHeadDim) return R3D_NO(RAD_E_SHAPE);
    const int gqa = (int)r3d_geti_or(a, "gqa", 1);
    if (gqa != 1 && gqa != 2 && gqa != 4 && gqa != 8) return R3D_NO(RAD_E_SHAPE);
    const int causal = (int)r3d_geti_or(a, "causal", 1);
    const int window = (int)r3d_geti_or(a, "window", 0);
    const float scale = (float)r3d_getf(a, "scale", 0.0);

    /* The kernel has an f16 query path (`A.q16`), but the launcher's only dtype argument, `kv_dt`,
     * is the cache's, so the shim cannot tell the kernel that q is f16 and it would widen f16 as
     * bf16. f16 queries are refused until a `q_dt` argument exists; widening the row's `q_dtype` to
     * "bf16 f16" requires it. */
    if (Q->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (O->dtype != Q->dtype) return R3D_NO(RAD_E_DTYPE);
    if (KV->dtype != RAD_BF16 && KV->dtype != RAD_F8E4M3) return R3D_NO(RAD_E_DTYPE);
    if (BT->dtype != RAD_I32 || SU->dtype != RAD_I32) return R3D_NO(RAD_E_DTYPE);
    const bool quant = KV->dtype == RAD_F8E4M3;
    /* A quantised cache without descale planes has scale 1.0. The reference reads the planes only
     * when present:
     *   const float kds = kdes ? ldt(kdes, s * kv_heads + kh) : 1.0f;
     *   const float vds = vdes ? ldt(vdes, s * kv_heads + kh) : 1.0f;
     * and the op's operand description marks both optional, so the kernel carries the null through
     * (`A.kds ? ... : 1.0f`). */
    const bool kgiven = KDS && KDS->data, vgiven = VDS && VDS->data;
    if (!quant && (kgiven || vgiven))
        return R3D_NO(RAD_E_SHAPE);   /* a bf16 cache has no scales; refused, not ignored */
    if (quant && kgiven && !r3d_is_f32(KDS)) return R3D_NO(RAD_E_DTYPE);
    if (quant && vgiven && !r3d_is_f32(VDS)) return R3D_NO(RAD_E_DTYPE);

    const int n_head = (int)r3d_geti_or(a, "n_head", r3d_dim(Q, (int)Q->rank - 2));
    if (n_head <= 0 || n_head % gqa) return R3D_NO(RAD_E_SHAPE);
    const int kv_heads = n_head / gqa;
    const int block_size = (int)r3d_geti_or(a, "block_size", r3d_dim(KV, 2));
    if (block_size != kBlock) return R3D_NO(RAD_E_SHAPE);
    if (r3d_dim(KV, 1) != kv_heads) return R3D_NO(RAD_E_SHAPE);
    /* The slot stride is hardcoded to 2 * kHeadDim in fill() (K then V in one slot), so a
     * wider-last-dim cache would be read at the wrong stride: exact, not a lower bound. */
    if (r3d_dim(KV, 3) != 2 * kHeadDim) return R3D_NO(RAD_E_SHAPE);
    const long long nb = r3d_dim(KV, 0);
    const int max_ctx = (int)(nb * block_size);
    const int max_blocks = (int)((max_ctx + block_size - 1) / block_size);

    const long long n_seq = r3d_numel(SU);
    if (n_seq < 1 || n_seq > 65536) return R3D_NO(RAD_E_SHAPE);
    /* The descriptor is [n_seq + 1] i32, contiguous: the kernel reads it with int indices, so an
     * i64 plane is refused rather than misread. `seqused` is checked the same way below. */
    if (ragged) {
        if (CU->dtype != RAD_I32) { r3d_trace_desc("cu_seqlens", CU); return R3D_NO(RAD_E_DTYPE); }
        if (!r3d_contig(CU)) { r3d_trace_desc("cu_seqlens", CU); return R3D_NO(RAD_E_STRIDE); }
        if (r3d_numel(CU) != n_seq + 1) { r3d_trace_desc("cu_seqlens", CU); return R3D_NO(RAD_E_SHAPE); }
        if (!r3d_aligned(CU->data, 4)) return R3D_NO(RAD_E_ALIGN);
    }
    long long qh = 0, qt = 0, qld = 0;
    if (!r3d_head_rows(Q, kHeadDim, &qt, &qh, &qld) || qh != n_head) { r3d_trace_desc("q", Q);
        r3d_trace_desc("out", O); return R3D_NO(RAD_E_STRIDE); }
    long long oh = 0, ot = 0, old = 0;
    if (!r3d_head_rows(O, kHeadDim, &ot, &oh, &old) || oh != n_head) { r3d_trace_desc("out", O);
        return R3D_NO(RAD_E_STRIDE); }
    if (qld % 8 || old % 8) return R3D_NO(RAD_E_STRIDE);
    /* Rows per sequence: the plane's row count over the sequence count for a rectangular batch.
     * For a ragged one the reference's `q_len = cus ? 0 : rows / n_seq` applies, and the plane's
     * row count (which cu[n_seq] equals) is passed as the row count. A ragged batch need not divide
     * evenly (5 rows over 3 sequences is a normal shape). */
    if (qt < 1) return R3D_NO(RAD_E_SHAPE);
    if (qt > 2147483000LL) return R3D_NO(RAD_E_SHAPE);   /* the kernel's row index is 32-bit */
    if (!ragged && qt % n_seq) return R3D_NO(RAD_E_SHAPE);
    const long long q_len = ragged ? qt : qt / n_seq;      /* the per-sequence length, rectangular */
    const long long rows = qt;                             /* the plane's rows, both forms */
    if (q_len < 1) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(O) < rows * n_head * kHeadDim) return R3D_NO(RAD_E_SHAPE);
    /* The descales are per (sequence, KV head), the reference's indexing: `s * kv_heads + kh`. */
    if (quant && kgiven && (r3d_numel(KDS) < n_seq * kv_heads || !r3d_contig(KDS)))
        return R3D_NO(RAD_E_SHAPE);
    if (quant && vgiven && (r3d_numel(VDS) < n_seq * kv_heads || !r3d_contig(VDS)))
        return R3D_NO(RAD_E_SHAPE);
    if (quant && (kgiven != vgiven)) return R3D_NO(RAD_E_SHAPE);   /* half a pair is a caller bug */
    /* The cache is dense and the launcher assumes it: a slot's halves must land on a 16-byte
     * boundary at every slot, so the cache's strides have to be the dense ones. */
    if (!r3d_contig(KV)) { r3d_trace_desc("kv", KV); return R3D_NO(RAD_E_STRIDE); }
    /* The table is the engine's, sized to the step, so its geometry comes from the tensor: the
     * columns are `shape[last]`, the row pitch `stride[rank-2]` and the column stride
     * `stride[rank-1]` (the reference's table read). A five-token prompt carries [n_seq, 1]. The
     * shim requires that every index the walk can reach (the last row's pitch plus the last column)
     * is inside the plane; a contiguity check would demand a table as wide as the cache and refuse
     * steps the engine sends. */
    const long long bt_cols = BT->rank >= 2 ? r3d_dim(BT, (int)BT->rank - 1) : (n_seq ? r3d_numel(BT) / n_seq : 0);
    const long long bt_rs   = BT->rank >= 2 ? BT->stride[BT->rank - 2] : bt_cols;
    const long long bt_cs   = BT->rank >= 2 ? BT->stride[BT->rank - 1] : 1;
    if (bt_cols < 1 || bt_rs < 1 || bt_cs < 1) { r3d_trace_desc("block_tables", BT);
        return R3D_NO(RAD_E_STRIDE); }
    const long long bt_reach = (n_seq - 1) * bt_rs + (bt_cols - 1) * bt_cs + 1;
    if (r3d_numel(BT) < bt_reach) { r3d_trace_desc("block_tables", BT);
        std::fprintf(stderr, "[r3d] block table: %lld entries, the walk reaches %lld\n",
                     (long long)r3d_numel(BT), bt_reach);
        return R3D_NO(RAD_E_STRIDE); }
    if (!r3d_contig(SU)) { r3d_trace_desc("seqused", SU); return R3D_NO(RAD_E_STRIDE); }
    if (!r3d_aligned(Q->data, 16) || !r3d_aligned(KV->data, 16) || !r3d_aligned(O->data, 16))
        return R3D_NO(RAD_E_ALIGN);

    int splits = decode ? 0 : 1;   /* no `splits` param in the spec: 0 auto-selects via pick_splits */
    /* The width the launcher will use; a ragged batch is picked on the average sequence, as in the
     * entry itself. */
    const long long per = ragged ? (rows + n_seq - 1) / n_seq : q_len;
    const int waves = pick_waves((int)per, gqa);
    const int tiles = (int)((per + tok_per_block(gqa, waves) - 1) / tok_per_block(gqa, waves));
    if (splits == 0) splits = decode ? pick_splits(tiles, (int)n_seq, kv_heads) : 1;
    if (splits > 64) splits = 64;
    if (splits > 1) {
        /* Halve the split count until the partials fit the arena the engine offers, down to 1,
         * which needs none. One slice is the same keys in the same order, so a smaller count
         * changes speed, not the result, and the band is never refused for lack of scratch. */
        for (;;) {
            if (splits <= 1) { splits = 1; break; }
            const long long need = qt * n_head * (long long)splits * kPartStride * 4;
            if (a->scratch && a->scratch_bytes >= (size_t)need) break;
            splits /= 2;
        }
        if (r3d_trace_on() && splits > 1)
            std::fprintf(stderr, "[r3d] attn: %d slices into %zu of %zu bytes needed\n",
                         splits, a->scratch_bytes, (size_t)(qt * n_head *
                                                            (long long)splits * kPartStride * 4));
    }

    /* Trace output, all under RAD_R3D_TRACE and after every check (a refused step is reported by
     * the refusal lines). The ring is drained first, printing records written since the last call
     * (replayed steps included), then re-armed for this launch. The plan line's `q_len` is the
     * kernel's (0 with a descriptor) and `rows` is the plane's row count. */
    dev_ring_drain();
    dev_trace_arm();

    if (r3d_trace_on())
        plan_line(decode, Q, KV, O, CU, SU, BT, splits > 1 ? (const void*)a->scratch : nullptr,
                  n_seq, rows, ragged ? 0LL : q_len, bt_cols, bt_rs, bt_cs, block_size, max_ctx,
                  waves, splits, r3d_trace_wide());

    /* With a descriptor the entries take the plane's row count where a rectangular batch passes
     * its per-sequence length. */
    const int e = decode
        ? r3d_attn_decode_h128(r3d_addr(Q), r3d_addr(KV), r3d_addr(BT), r3d_addr(SU),
                               r3d_addr(KDS), r3d_addr(VDS), r3d_addr(CU), r3d_addr(O), (int)n_seq,
                               (int)(ragged ? rows : q_len),
                               head_dim, gqa, n_head, block_size, (int)bt_cols, (int)bt_rs,
                               (int)bt_cs, max_ctx, window, causal, scale,
                               (int)KV->dtype, qld, old, splits,
                               (long)(uintptr_t)(splits > 1 ? a->scratch : nullptr),
                               (long)(uintptr_t)s)
        : r3d_attn_prefill_h128(r3d_addr(Q), r3d_addr(KV), r3d_addr(BT), r3d_addr(SU),
                                r3d_addr(KDS), r3d_addr(VDS), r3d_addr(CU), r3d_addr(O), (int)n_seq,
                                (int)(ragged ? rows : q_len), head_dim, gqa, n_head, block_size,
                                (int)bt_cols, (int)bt_rs, (int)bt_cs, max_ctx, window,
                                causal, scale, (int)KV->dtype, qld, old,
                                (long)(uintptr_t)nullptr, (long)(uintptr_t)s);
    if (e == 0) return RAD_OK;
    return R3D_NO(e == -1 ? RAD_E_DTYPE : e == -2 ? RAD_E_SHAPE
                        : e == -3 ? RAD_E_ALIGN : e == -4 ? RAD_E_DEVICE
                        : e == -5 ? RAD_E_SCRATCH : RAD_E_INVAL);
}

}  // namespace

extern "C" int r3d_rad_attn_decode_h128(const RadArgs* a, RadStream s)
{
    return attn_shim(a, s, true);
}

extern "C" int r3d_rad_attn_prefill_h128(const RadArgs* a, RadStream s)
{
    return attn_shim(a, s, false);
}
