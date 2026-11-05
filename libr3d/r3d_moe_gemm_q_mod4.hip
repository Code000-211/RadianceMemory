/* r3d_moe_gemm_q_mod4.hip -- moe_gemm_q_mod4: the grouped quantised GEMM with 4 expert classes.
 *
 * Same as r3d_moe_gemm_q.hip but experts route to 4 classes (e%4) instead of 2 (e%2).
 * Each class has its own weight table and scale table.
 */
#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"
#include "r3d_codecs.h"
#include "r3d_fp8_dequant.h"
#include "r3d_moe_tab.h"

namespace {

constexpr int kThreads = 256;
constexpr int kWaves   = kThreads / kWave;
constexpr int kNBlock  = 128;
constexpr int kGroup   = 128;
constexpr int kPitch   = 33;

__device__ __forceinline__ _Float16 h16(uint32_t w) { return __builtin_bit_cast(_Float16, (uint16_t)w); }

/* Use two 2-class tabs: tw01 for classes 0-1, tw23 for classes 2-3 */

template <int ALU, int MR, int WV>
__global__ __launch_bounds__(WV * kWave) R3D_WAVES(WV) void moe_q_mod4_kernel(
    const uint8_t* __restrict__ a, const float* __restrict__ a_scale,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
    R3dTab w0, R3dTab w1, R3dTab w2, R3dTab w3,
    R3dTab s0, R3dTab s1, R3dTab s2, R3dTab s3,
    uint16_t* __restrict__ y, int T, int N, int K, int a_ld, int as_ld, int y_ld,
    int n_expert, int top_k, int a_sorted)
{
    __shared__ __align__(16) uint32_t s_a[MR * kPitch];
    __shared__ __align__(16) uint32_t s_w[(WV * 16) * kPitch];
    __shared__ __align__(16) uint16_t s_tbl[R3D_E4M3_TBL];
    __shared__ int32_t s_tok[MR];
    __shared__ int32_t s_nruns;
    __shared__ int32_t s_rs[MR], s_re[MR], s_e[MR];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & (kWave - 1);
    const uint32_t wv = tid / kWave;
    const uint32_t r = lane & 15;

    if (!ALU) r3d_e4m3_f16_init(s_tbl, (int)tid, WV * kWave);

    const uint32_t r0 = (uint32_t)blockIdx.y * MR;
    const uint32_t n0 = (uint32_t)blockIdx.x * (WV * 16);
    const uint32_t r1 = (r0 + MR < (uint32_t)T) ? r0 + MR : (uint32_t)T;

    if (tid < 16) {
        const uint32_t i = r0 + tid;
        const int32_t f = (i < (uint32_t)T) ? sorted[i] : -1;
        s_tok[tid] = (f < 0) ? -1 : (a_sorted ? (int32_t)i : f / top_k);
    }
    if (tid == 0) {
        int nruns = 0, i = (int)r0;
        while (i < (int)r1) {
            int32_t ee = -1;
            for (int32_t x = 0; x < n_expert; ++x) if (off[x] <= i && i < off[x + 1]) ee = x;
            const int re = (ee < 0) ? (int)r1 : (int)(off[ee + 1] < r1 ? off[ee + 1] : r1);
            s_rs[nruns] = i; s_re[nruns] = re; s_e[nruns] = ee;
            ++nruns; i = re;
        }
        s_nruns = nruns;
    }
    group_barrier();
    const int nruns = s_nruns;

    const uint32_t nc = n0 + wv * 16 + r;
    const int n_real = (int)(nc < (uint32_t)N ? nc : (uint32_t)N - 1);

    for (int run = 0; run < nruns; ++run) {
        const int rs = s_rs[run], re = s_re[run], e = s_e[run];
        v8float acc = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
        if (e >= 0) {
        /* 4-class routing: expert e is entry e/4 of class e%4 */
        const int cls = e & 3, e4 = e >> 2;
        const R3dTab* wtab = (cls == 0) ? &w0 : (cls == 1) ? &w1 : (cls == 2) ? &w2 : &w3;
        const R3dTab* stab = (cls == 0) ? &s0 : (cls == 1) ? &s1 : (cls == 2) ? &s2 : &s3;
        const uint8_t* wbase = r3d_tab_at(*wtab, e4);
        const int w_ld_b = wtab->ld;
        const float* wsb = (const float*)r3d_tab_at(*stab, e4);
        const int ws_ld = stab->ld;

        for (int b = 0; b < K / kGroup; ++b) {
            const int kt = b * kGroup;

            for (int i = (int)tid; i < MR * (kGroup / 4); i += (WV * kWave)) {
                const int row = i / (kGroup / 4), d = i % (kGroup / 4);
                const int src = (rs <= (int)(r0 + row) && (int)(r0 + row) < re) ? s_tok[row]
                                                                                 : s_tok[rs - (int)r0];
                const int rr = (src < 0) ? 0 : src;
                s_a[(size_t)row * kPitch + d] = *(const uint32_t*)(a + (size_t)rr * a_ld + kt + 4 * d);
            }
            for (int i = (int)tid; i < (WV * 16) * (kGroup / 4); i += (WV * kWave)) {
                const int col = i / (kGroup / 4), d = i % (kGroup / 4);
                const int nn = (int)(n0 + col < (uint32_t)N ? n0 + col : (uint32_t)N - 1);
                s_w[(size_t)col * kPitch + d] =
                    *(const uint32_t*)(wbase + (size_t)nn * w_ld_b + kt + 4 * d);
            }
            group_barrier();

            const uint32_t* ap = s_a + (size_t)r * kPitch;
            const uint32_t* wp = s_w + (size_t)(wv * 16 + r) * kPitch;
            v8float blk = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
#pragma unroll
            for (int k = 0; k < kGroup; k += 16) {
                uint32_t ain[4], af[8], bin[4], bf[8];
#pragma unroll
                for (int g = 0; g < 4; ++g) ain[g] = ap[k / 4 + g];
                r3d_e4m3x16_decode<ALU>(s_tbl, ain, af);
#pragma unroll
                for (int g = 0; g < 4; ++g) bin[g] = wp[k / 4 + g];
                r3d_e4m3x16_decode<ALU>(s_tbl, bin, bf);
                v16half fa, fb;
#pragma unroll
                for (int g = 0; g < 8; ++g) { fa[2 * g] = h16(af[g] & 0xffffu); fa[2 * g + 1] = h16(af[g] >> 16); }
#pragma unroll
                for (int g = 0; g < 8; ++g) { fb[2 * g] = h16(bf[g] & 0xffffu); fb[2 * g + 1] = h16(bf[g] >> 16); }
                blk = wmma_f16(fa, fb, blk);
            }
            group_barrier();

            const float sw = wsb[(size_t)(n_real / kGroup) * ws_ld + b];
#pragma unroll
            for (int g = 0; g < 8; ++g) {
                const int row = (int)(g * 2 + (lane >> 4));
                const int src = (rs <= (int)(r0 + row) && (int)(r0 + row) < re) ? s_tok[row]
                                                                                  : s_tok[rs - (int)r0];
                const float sa = a_scale[(size_t)(src < 0 ? 0 : src) * as_ld + b];
                acc[g] += blk[g] * sa * sw;
            }
        }

        }
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int row = (int)(g * 2 + (lane >> 4));
            const int m = (int)(r0 + row);
            if (rs <= m && m < re && nc < (uint32_t)N) y[(size_t)m * y_ld + nc] = f32_to_bf16(acc[g]);
        }
    }
}

}  // namespace

extern "C" {

extern "C" int r3d_rad_moe_gemm_q_mod4(const RadArgs* a, RadStream s)
{
    /* 13 operands: a, a_scale, w, w_scale, sorted_tok, expert_offset, w_1, w_1_scale,
     *              w_2, w_2_scale, w_3, w_3_scale, y */
    enum { O_A = 0, O_AS, O_W0, O_W0S, O_SORTED, O_OFF, O_W1, O_W1S,
           O_W2, O_W2S, O_W3, O_W3S, O_Y };
    const RadTensor* Aq     = r3d_opt(a, O_A);
    const RadTensor* As     = r3d_opt(a, O_AS);
    const RadTensor* W0     = r3d_opt(a, O_W0);
    const RadTensor* W0S    = r3d_opt(a, O_W0S);
    const RadTensor* sorted = r3d_opt(a, O_SORTED);
    const RadTensor* off    = r3d_opt(a, O_OFF);
    const RadTensor* W1     = r3d_opt(a, O_W1);
    const RadTensor* W1S    = r3d_opt(a, O_W1S);
    const RadTensor* W2     = r3d_opt(a, O_W2);
    const RadTensor* W2S    = r3d_opt(a, O_W2S);
    const RadTensor* W3     = r3d_opt(a, O_W3);
    const RadTensor* W3S    = r3d_opt(a, O_W3S);
    const RadTensor* Y      = r3d_opt(a, O_Y);
    if (!Aq || !As || !sorted || !off || !Y) return R3D_NO(RAD_E_INVAL);
    if (!W0 || !W0S || !W1 || !W1S || !W2 || !W2S || !W3 || !W3S) return R3D_NO(RAD_E_INVAL);
    if (!Aq->data || !As->data || !sorted->data || !off->data || !Y->data)
        return R3D_NO(RAD_E_INVAL);

    const long long N = r3d_geti_or(a, "N", 0);
    const long long K = r3d_geti_or(a, "K", 0);
    const long long n_expert = r3d_geti_or(a, "n_expert", 0);
    const long long top_k = r3d_geti_or(a, "top_k", 0);
    const long long group = r3d_geti_or(a, "group", 0);
    if (N <= 0 || K <= 0 || n_expert <= 0 || top_k <= 0) return R3D_NO(RAD_E_SHAPE);
    if (group != kGroup) return R3D_NO(RAD_E_SHAPE);
    if (K % kGroup) return R3D_NO(RAD_E_SHAPE);

    /* The class shapes default to class 0's; this kernel reads one N and one K for all four, so
     * an unequal class is refused, not approximated (as moe_gemm_q does with N_odd / K_odd). */
    static const char* const kNk[3][2] = { { "N_1", "K_1" }, { "N_2", "K_2" }, { "N_3", "K_3" } };
    for (int q = 0; q < 3; ++q)
        if (r3d_geti_or(a, kNk[q][0], N) != N || r3d_geti_or(a, kNk[q][1], K) != K)
            return R3D_NO(RAD_E_UNSUPPORTED);
    const long long parts = r3d_geti_or(a, "parts", 1);
    if (parts < 1 || N % parts) return R3D_NO(RAD_E_SHAPE);

    /* "token" (the default, gate_up): sorted row i reads a[sorted_tok[i] / top_k]. "sorted" (the
     * down projection): a already has one row per (token, slot) in the scatter's order. */
    const char* ord = r3d_gets(a, "a_order");
    const int a_sorted = ord && std::strcmp(ord, "sorted") == 0;
    if (ord && !a_sorted && std::strcmp(ord, "token") != 0) return R3D_NO(RAD_E_UNSUPPORTED);

    if (Aq->dtype != RAD_F8E4M3 || As->dtype != RAD_F32 || Y->dtype != RAD_BF16)
        return R3D_NO(RAD_E_DTYPE);
    if (sorted->dtype != RAD_I32 || off->dtype != RAD_I32) return R3D_NO(RAD_E_DTYPE);

    /* The activation's row count is checked, not inferred: a short operand means the caller
     * passed the gate_up projection's buffer to the down projection. */
    long long T = 0, yc = 0;
    if (r3d_rows_of(Y, &T, &yc) != RAD_OK || yc < N) return R3D_NO(RAD_E_SHAPE);
    long long ar = 0, ak = 0;
    if (r3d_rows_of(Aq, &ar, &ak) != RAD_OK || ak < K) return R3D_NO(RAD_E_SHAPE);
    if (ar < (a_sorted ? T : T / top_k)) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(sorted) < T || r3d_numel(off) < n_expert + 1) return R3D_NO(RAD_E_SHAPE);

    const int a_ld = (Aq->rank >= 2) ? (int)Aq->stride[0] : (int)K;
    const int as_ld = (As->rank >= 2) ? (int)As->stride[0] : (int)(K / kGroup);
    const int y_ld = (Y->rank >= 2) ? (int)Y->stride[0] : (int)N;

    /* Build 8 individual tabs */
    R3dTab w0, w1, w2, w3, s0, s1, s2, s3;
    const RadTensor* wts[4]  = { W0, W1, W2, W3 };
    const RadTensor* wss[4]  = { W0S, W1S, W2S, W3S };
    R3dTab* wtabs[4] = { &w0, &w1, &w2, &w3 };
    R3dTab* stabsg[4] = { &s0, &s1, &s2, &s3 };
    for (int q = 0; q < 4; ++q) {
        int rc_w = r3d_tab_of(wts[q], RAD_F8E4M3, 8, n_expert / 4, N, K, wtabs[q]);
        if (rc_w != RAD_OK) return rc_w;
        int rc_s = r3d_tab_of(wss[q], RAD_F32, 32, n_expert / 4, (N + kGroup - 1) / kGroup, K / kGroup,
                              stabsg[q]);
        if (rc_s != RAD_OK) return rc_s;
    }


    const uint8_t* pa = (const uint8_t*)r3d_p(Aq);
    const float* psa = (const float*)r3d_p(As);
    const int32_t* ps = (const int32_t*)r3d_p(sorted);
    const int32_t* po = (const int32_t*)r3d_p(off);
    uint16_t* py = (uint16_t*)r3d_wp(Y);

    const int mrows = r3d_geti_or(a, "mrows", 16);
    const int wv = r3d_geti_or(a, "wv", 8);
    const int ctile = wv * 16;
    dim3 grid((unsigned)((N + ctile - 1) / ctile), (unsigned)((T + mrows - 1) / mrows)), blk(wv * kWave);
    if (mrows == 8 && wv == 4)
        moe_q_mod4_kernel<0,8,4><<<grid, blk, 0, (hipStream_t)s>>>(pa, psa, ps, po, w0, w1, w2, w3, s0, s1, s2, s3, py, (int)T, (int)N, (int)K, a_ld, as_ld, y_ld, (int)n_expert, (int)top_k, a_sorted);
    else if (mrows == 8)
        moe_q_mod4_kernel<0,8,8><<<grid, blk, 0, (hipStream_t)s>>>(pa, psa, ps, po, w0, w1, w2, w3, s0, s1, s2, s3, py, (int)T, (int)N, (int)K, a_ld, as_ld, y_ld, (int)n_expert, (int)top_k, a_sorted);
    else if (wv == 4)
        moe_q_mod4_kernel<0,16,4><<<grid, blk, 0, (hipStream_t)s>>>(pa, psa, ps, po, w0, w1, w2, w3, s0, s1, s2, s3, py, (int)T, (int)N, (int)K, a_ld, as_ld, y_ld, (int)n_expert, (int)top_k, a_sorted);
    else
        moe_q_mod4_kernel<0,16,8><<<grid, blk, 0, (hipStream_t)s>>>(pa, psa, ps, po, w0, w1, w2, w3, s0, s1, s2, s3, py, (int)T, (int)N, (int)K, a_ld, as_ld, y_ld, (int)n_expert, (int)top_k, a_sorted);
    int hrc = (int)hipGetLastError();
    return hrc;
}

}  // extern "C"
