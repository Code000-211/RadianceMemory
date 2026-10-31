// R4D attention at sixteen queries per KV head, head_dim 128: Nemotron-H's attention layers (32
// query heads over 2 KV heads) at one and two ranks. Geometry only; r4d_attn_paged_gqa8.hip is the
// body. The ratio changes two bounds, both stated in the row table:
//
//   decode verify width   64 / gqa = 4 (a workgroup holds 64 (query, head) rows).
//   prefill block width   384 / gqa = 24 query positions.
//
// No head_dim 256 leg (R4D_ATTN_H256 0): nothing in scope is gqa 16 at 256. Page size 16.
#define R4D_ATTN_GQA  16
#define R4D_ATTN_H128 1
#define R4D_ATTN_H256 0

#include "r4d_attn_paged_gqa8.hip"
