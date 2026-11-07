// r3d_ngram_ids.hip -- the hashed row ids of Qwen4-Exp's Per-Layer Embedding.
//
//   mixed(k) = t0*mult[0] ^ t1*mult[1] ^ ... ^ tk*mult[k]        int64, wrapping
//   ids[h]   = mixed(b+1) % vocab_sizes[h] + offsets[h]          h in block b
//
// One thread a token, looping its heads.
#include <hip/hip_runtime.h>
#include <cstdint>
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_launch.h"

namespace {

__device__ __forceinline__ long long ng_wrap_mul(long long a, long long b) {
    return (long long)((unsigned long long)a * (unsigned long long)b);
}
__device__ __forceinline__ long long ng_floor_mod(long long a, long long b) {
    const long long r = a % b;
    return r < 0 ? r + b : r;
}

enum { kNgMaxShift = 8, kNgThreads = 256 };

__global__ __launch_bounds__(kNgThreads) void r3d_ngram_ids_kernel(
    const int32_t* __restrict__ tok, int32_t* __restrict__ state,
    const int32_t* __restrict__ cu, const long long* __restrict__ mult,
    const long long* __restrict__ vsz, const long long* __restrict__ off,
    const int32_t* __restrict__ sidx, const int32_t* __restrict__ hini,
    const int32_t* __restrict__ nacc, int32_t* __restrict__ ids,
    uint32_t heads, uint32_t ngram, uint32_t per, uint32_t W,
    long long eos, uint32_t ids_ld, int32_t* __restrict__ ahead, uint32_t n_ahead,
    uint32_t ahead_ld)
{
    const uint32_t s   = blockIdx.y;
    const uint32_t bos = (uint32_t)cu[s];
    const uint32_t eot = (uint32_t)cu[s + 1];
    const uint32_t ctx = ngram - 1u;
    if (eot <= bos) return;
    const int32_t slot = sidx ? sidx[s] : (int32_t)s;
    if (slot < 0) return;

    const uint32_t slen = eot - bos;
    const bool warm = !hini || hini[s] != 0;
    const long long o = nacc ? (long long)nacc[s] - 1 : 0;
    const uint32_t lt = blockIdx.x * kNgThreads + threadIdx.x;

    const uint32_t reach = slen + ((ahead && s + 1u == gridDim.y) ? n_ahead : 0u);
    if (lt < reach) {
        const uint32_t t = bos + lt;
        long long prev[kNgMaxShift];
        for (uint32_t i = 0; i < ctx; ++i)
            prev[i] = warm ? (long long)state[(long long)slot * W + o + i] : eos;

        const uint32_t p = ctx + lt;
        long long sh[kNgMaxShift];
        sh[0] = (long long)tok[t];
        bool cut = false;
        for (uint32_t k = 1; k < ngram; ++k) {
            if (cut || sh[k - 1] == eos) { cut = true; sh[k] = eos; continue; }
            const uint32_t q = p - k;
            sh[k] = q < ctx ? prev[q] : (long long)tok[bos + (q - ctx)];
        }

        long long mixed = 0;
        int blk = -1;
        for (uint32_t k = 0; k < ngram; ++k) {
            const long long term = ng_wrap_mul(sh[k], mult[k]);
            mixed = k == 0 ? term : (mixed ^ term);
            if (k == 0) continue;
            ++blk;
            int32_t* const row = lt < slen ? ids + (long long)t * ids_ld
                                           : ahead + (long long)(lt - slen) * ahead_ld;
            for (uint32_t j = 0; j < per; ++j) {
                const uint32_t h = (uint32_t)blk * per + j;
                row[h] = (int32_t)(ng_floor_mod(mixed, vsz[h]) + off[h]);
            }
        }
    }

    __syncthreads();
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int32_t* const win = state + (long long)slot * W;
    if (nacc) {
        for (uint32_t k = 0; k + 1 < ctx; ++k)
            win[k] = warm ? win[o + k + 1] : (int32_t)eos;
        for (uint32_t t = 0; t < slen && ctx - 1u + t < W; ++t) win[ctx - 1u + t] = tok[bos + t];
        return;
    }
    long long keep[kNgMaxShift];
    for (uint32_t k = 0; k < ctx; ++k) {
        const long long src = (long long)slen - (long long)ctx + (long long)k;
        keep[k] = src >= 0 ? (long long)tok[bos + src]
                           : (warm ? (long long)win[o + src + ctx] : eos);
    }
    for (uint32_t k = 0; k < ctx; ++k) win[k] = (int32_t)keep[k];
}

}  // namespace

enum { NG_TOK = 0, NG_STATE, NG_CU, NG_MULT, NG_VSZ, NG_OFF, NG_CIDX, NG_HINIT, NG_NACC, NG_IDS,
       NG_AHEAD };

extern "C" int r3d_rad_ngram_ids(const RadArgs* a, RadStream s)
{
    const RadTensor* tok  = r3d_opt(a, NG_TOK);
    const RadTensor* st   = r3d_opt(a, NG_STATE);
    const RadTensor* cu   = r3d_opt(a, NG_CU);
    const RadTensor* mult = r3d_opt(a, NG_MULT);
    const RadTensor* vsz  = r3d_opt(a, NG_VSZ);
    const RadTensor* off  = r3d_opt(a, NG_OFF);
    const RadTensor* cidx = r3d_opt(a, NG_CIDX);
    const RadTensor* hini = r3d_opt(a, NG_HINIT);
    const RadTensor* nacc = r3d_opt(a, NG_NACC);
    const RadTensor* ids  = r3d_opt(a, NG_IDS);
    const RadTensor* ahd  = r3d_opt(a, NG_AHEAD);
    if (!r3d_p(tok) || !r3d_wp(st) || !r3d_p(cu) || !r3d_p(mult) ||
        !r3d_p(vsz) || !r3d_p(off) || !r3d_wp(ids)) return R3D_NO(RAD_E_INVAL);
    if (tok->dtype != RAD_I32 || st->dtype != RAD_I32 || ids->dtype != RAD_I32)
        return R3D_NO(RAD_E_DTYPE);
    if (cu->dtype != RAD_I32) return R3D_NO(RAD_E_DTYPE);
    if (mult->dtype != RAD_I64 || vsz->dtype != RAD_I64 || off->dtype != RAD_I64)
        return R3D_NO(RAD_E_DTYPE);

    const long long M     = r3d_geti_or(a, "M", 0);
    const long long heads = r3d_geti_or(a, "heads", 0);
    const long long ngram = r3d_geti_or(a, "ngram", 0);
    const long long eos   = r3d_geti_or(a, "eos", -1);
    if (M <= 0 || heads <= 0 || ngram < 2 || eos < 0) return R3D_NO(RAD_E_SHAPE);
    if (ngram > kNgMaxShift) return R3D_NO(RAD_E_UNSUPPORTED);
    if (heads % (ngram - 1)) return R3D_NO(RAD_E_SHAPE);
    const long long per = heads / (ngram - 1), ctx = ngram - 1;

    const long long T = ids->rank >= 1 ? ids->shape[0] : M;
    const long long ids_ld = ids->rank >= 2 ? ids->stride[0] : heads;

    long long A = 0, ahead_ld = heads;
    if (ahd && r3d_p(ahd)) {
        if (ahd->dtype != RAD_I32) return R3D_NO(RAD_E_DTYPE);
        A = ahd->rank >= 1 ? ahd->shape[0] : 0;
        ahead_ld = ahd->rank >= 2 ? ahd->stride[0] : heads;
    }
    if ((long long)tok->shape[0] < T + A) return R3D_NO(RAD_E_SHAPE);
    if ((long long)mult->shape[0] < ngram || (long long)vsz->shape[0] < heads ||
        (long long)off->shape[0] < heads) return R3D_NO(RAD_E_SHAPE);

    const long long N = (long long)cu->shape[0] - 1;
    if (N <= 0) return R3D_NO(RAD_E_SHAPE);
    const long long W = st->rank >= 2 ? st->shape[st->rank - 1] : ctx;
    if (W < ctx) return R3D_NO(RAD_E_SHAPE);

    const uint32_t tiles = (uint32_t)((T + A + kNgThreads - 1) / kNgThreads);
    dim3 grid(tiles, (uint32_t)N), blk(kNgThreads);
    R3D_KLAUNCH(r3d_ngram_ids_kernel, grid, blk, (hipStream_t)s,
                (const int32_t*)r3d_p(tok), (int32_t*)r3d_wp(st),
                (const int32_t*)r3d_p(cu), (const long long*)r3d_p(mult),
                (const long long*)r3d_p(vsz), (const long long*)r3d_p(off),
                cidx ? (const int32_t*)r3d_p(cidx) : nullptr,
                hini ? (const int32_t*)r3d_p(hini) : nullptr,
                nacc ? (const int32_t*)r3d_p(nacc) : nullptr,
                (int32_t*)r3d_wp(ids), (uint32_t)heads, (uint32_t)ngram,
                (uint32_t)per, (uint32_t)W, eos, (uint32_t)ids_ld,
                A > 0 ? (int32_t*)r3d_p(ahd) : nullptr, (uint32_t)A,
                (uint32_t)ahead_ld);
    return (int)hipGetLastError();
}
