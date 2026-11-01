// Full-vocabulary DSpark Markov greedy walk. Unlike dflash_select's candidate-lattice walk, this
// operation applies the learned bigram bias to every draft-head logit and greedily selects the
// largest biased score at each position, matching the pinned llama.cpp reference.
#include <hip/hip_runtime.h>
#include <climits>
#include "r4d_gdn_args.h"
#include "r4d_dflash_conv_body.h"  // r4d_bf2f, r4d_u16
#include "r4d_common.h"           // r4d_wave_sum

namespace {

constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kWarps = kThreads / kWave;
constexpr int kCandidatesPerTile = 64;
constexpr int kCandidatesPerWarp = kCandidatesPerTile / kWarps;
constexpr int kMaxRank = 512;

// One CTA scans 64 vocabulary rows, eight at a time (one warp per row). The rank reduction is
// parallel across the warp and codebook loads remain contiguous. Each CTA writes its exact local
// argmax; a second kernel reduces those tile winners before the next dependent position starts.
__global__ __launch_bounds__(kThreads) void markov_score_tiles(
    const r4d_u16* __restrict__ logits, long logits_pitch,
    const int* __restrict__ anchor, long anchor_stride,
    const r4d_u16* __restrict__ pred, const r4d_u16* __restrict__ succ,
    float* __restrict__ tile_scores, int* __restrict__ tile_tokens,
    const int* __restrict__ out, long out_pitch,
    int step, int steps, int rank, int vocab, int n_tiles) {
  const int seq = blockIdx.y;
  const int tile = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int warp = tid / kWave;
  const int prev = step == 0 ? anchor[(long)seq * anchor_stride]
                             : out[(long)seq * out_pitch + step - 1];
  const bool prev_ok = prev >= 0 && prev < vocab;

  __shared__ float s_pred[kMaxRank];
  __shared__ float s_best[kWarps];
  __shared__ int s_token[kWarps];
  for (int r = tid; r < rank; r += kThreads)
    s_pred[r] = prev_ok ? r4d_bf2f(pred[(long)prev * rank + r]) : 0.0f;
  __syncthreads();

  float warp_best = -INFINITY;
  int warp_token = tile * kCandidatesPerTile + warp;
#pragma unroll
  for (int j = 0; j < kCandidatesPerWarp; ++j) {
    const int token = tile * kCandidatesPerTile + warp + j * kWarps;
    if (token < vocab) {
      float dot = 0.0f;
#pragma unroll 8
      for (int r = lane; r < rank; r += kWave)
        dot += s_pred[r] * r4d_bf2f(succ[(long)token * rank + r]);
      dot = r4d_wave_sum(dot);
      if (lane == 0) {
        const float score = r4d_bf2f(logits[(long)seq * steps * logits_pitch +
                                             (long)step * logits_pitch + token]) + dot;
        if (score > warp_best || (score == warp_best && token < warp_token)) {
          warp_best = score;
          warp_token = token;
        }
      }
    }
  }
  if (lane == 0) {
    s_best[warp] = warp_best;
    s_token[warp] = warp_token;
  }
  __syncthreads();
  if (tid == 0) {
    float best = -INFINITY;
    int best_token = tile * kCandidatesPerTile;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      const float score = s_best[w];
      const int token = s_token[w];
      if (score > best || (score == best && token < best_token)) {
        best = score;
        best_token = token;
      }
    }
    tile_scores[(long)seq * n_tiles + tile] = best;
    tile_tokens[(long)seq * n_tiles + tile] = best_token;
  }
}

__global__ __launch_bounds__(kThreads) void markov_reduce_tiles(
    const float* __restrict__ tile_scores, const int* __restrict__ tile_tokens,
    int* __restrict__ out, long out_pitch, int step, int n_tiles) {
  const int seq = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float s_best[kThreads];
  __shared__ int s_token[kThreads];

  float best = -INFINITY;
  int best_token = INT_MAX;
  for (int tile = tid; tile < n_tiles; tile += kThreads) {
    const float score = tile_scores[(long)seq * n_tiles + tile];
    const int token = tile_tokens[(long)seq * n_tiles + tile];
    if (score > best || (score == best && token < best_token)) {
      best = score;
      best_token = token;
    }
  }
  s_best[tid] = best;
  s_token[tid] = best_token;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float score = s_best[tid + stride];
      const int token = s_token[tid + stride];
      if (score > s_best[tid] || (score == s_best[tid] && token < s_token[tid])) {
        s_best[tid] = score;
        s_token[tid] = token;
      }
    }
    __syncthreads();
  }
  if (tid == 0) out[(long)seq * out_pitch + step] = s_token[0];
}

int markov_emit(const r4d_u16* logits, long logits_pitch, const int* anchor, long anchor_stride,
                const r4d_u16* pred, const r4d_u16* succ, float* tile_scores, int* tile_tokens,
                int* out, long out_pitch, int sequences, int steps, int rank, int vocab,
                int n_tiles, RadStream stream) {
  for (int step = 0; step < steps; ++step) {
    hipLaunchKernelGGL(markov_score_tiles, dim3(n_tiles, sequences), dim3(kThreads), 0,
                       (hipStream_t)stream, logits, logits_pitch, anchor, anchor_stride, pred,
                       succ, tile_scores, tile_tokens, out, out_pitch, step, steps, rank, vocab,
                       n_tiles);
    hipLaunchKernelGGL(markov_reduce_tiles, dim3(sequences), dim3(kThreads), 0,
                       (hipStream_t)stream, tile_scores, tile_tokens, out, out_pitch, step,
                       n_tiles);
  }
  return (int)hipGetLastError();
}

}  // namespace

extern "C" int r4d_rad_dflash_markov_greedy(const RadArgs* a, RadStream stream) {
  constexpr int LOGITS = 0, ANCHOR = 1, PRED = 2, SUCC = 3;
  constexpr int TILE_SCORE = 4, TILE_TOKEN = 5, OUT = 6;
  const RadTensor* logits = r4d_opt(a, LOGITS);
  const RadTensor* anchor = r4d_opt(a, ANCHOR);
  const RadTensor* pred = r4d_opt(a, PRED);
  const RadTensor* succ = r4d_opt(a, SUCC);
  const RadTensor* scores = r4d_opt(a, TILE_SCORE);
  const RadTensor* tokens = r4d_opt(a, TILE_TOKEN);
  const RadTensor* out = r4d_opt(a, OUT);
  if (!logits || !anchor || !pred || !succ || !scores || !tokens || !out)
    return R4D_NO(RAD_E_INVAL);

  const long long steps = r4d_geti_or(a, "steps", 0);
  const long long rank = r4d_geti_or(a, "rank", 0);
  const long long vocab = r4d_geti_or(a, "n_vocab", 0);
  const long long anchor_stride = r4d_geti_or(a, "anchor_stride", 1);
  if (steps <= 0 || rank < kWave || rank > kMaxRank || rank % kWave || vocab <= 0 ||
      anchor_stride <= 0)
    return R4D_NO(RAD_E_INVAL);

  if (out->dtype != RAD_I32 || out->rank != 2 || out->shape[1] < steps)
    return R4D_NO(RAD_E_SHAPE);
  const long long sequences = out->shape[0];
  const long long out_pitch = r4d_row_pitch(out);
  const long long logits_pitch = r4d_row_pitch(logits);
  if (sequences <= 0 || out_pitch < steps || logits->dtype != RAD_BF16 || logits->rank != 2 ||
      logits->shape[0] < sequences * steps || logits->shape[1] < vocab || logits_pitch < vocab)
    return R4D_NO(RAD_E_SHAPE);

  if (anchor->dtype != RAD_I32 || !r4d_contig(anchor) ||
      r4d_numel(anchor) < (sequences - 1) * anchor_stride + 1)
    return R4D_NO(RAD_E_SHAPE);
  if (pred->dtype != RAD_BF16 || succ->dtype != RAD_BF16 || !r4d_contig(pred) ||
      !r4d_contig(succ) || r4d_numel(pred) < vocab * rank || r4d_numel(succ) < vocab * rank)
    return R4D_NO(RAD_E_SHAPE);

  const long long n_tiles = (vocab + kCandidatesPerTile - 1) / kCandidatesPerTile;
  if (scores->dtype != RAD_F32 || tokens->dtype != RAD_I32 || scores->rank != 2 ||
      tokens->rank != 2 || scores->shape[0] < sequences || tokens->shape[0] < sequences ||
      scores->shape[1] < n_tiles || tokens->shape[1] < n_tiles ||
      r4d_row_pitch(scores) < n_tiles || r4d_row_pitch(tokens) < n_tiles)
    return R4D_NO(RAD_E_SHAPE);

  return R4D_RAW(markov_emit((const r4d_u16*)logits->data, (long)logits_pitch,
                             (const int*)anchor->data, (long)anchor_stride,
                             (const r4d_u16*)pred->data, (const r4d_u16*)succ->data,
                             (float*)scores->data, (int*)tokens->data, (int*)out->data,
                             (long)out_pitch, (int)sequences, (int)steps, (int)rank,
                             (int)vocab, (int)n_tiles, stream));
}
