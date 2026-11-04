# QSA: Qwen4-Exp's sparse attention

The full-attention layers of Qwen4-Exp (every fourth: 3, 7, … 47) do not attend to the whole
history. A small **indexer** scores 4-token blocks of the key history, and the attention reads the
best 512 of them plus the incomplete tail, so attention cost is constant past 2051 tokens.

The declaration side is `arch/common/rad_qsa.h`. The kernels are libr4d's `qsa_block_key`,
`qsa_score`, `qsa_select`, `qsa_work` and `qsa_tail_store`; the sparse attention is the ordinary
`attn_paged`. Schemas and operand contracts are in `docs/OPS.md`.

## Shape

From the checkpoint: `indexer_n_heads` 4, `indexer_kv_heads` 1, `indexer_head_dim` 128,
`indexer_budget` 2048, `indexer_compress_ratio` 4, so `block_topk = 2048 / 4 = 512` blocks.
`rotary_dim` is the main attention's (`head_dim 256 * partial_rotary_factor 0.25 = 64`), applied to
the first 64 of the indexer's 128 dims from the layer's cos/sin table; the plugin refuses a
checkpoint where it does not fit. Weights per QSA layer:

```
model.language_model.layers.<L>.self_attn.indexer.index_qk_proj.weight   [640, 2560]
model.language_model.layers.<L>.self_attn.indexer.q_layernorm.weight     [128]
model.language_model.layers.<L>.self_attn.indexer.k_layernorm.weight     [128]
```

640 = (4 query heads + 1 key head) x 128.

## Computation

```
qk        = index_qk_proj(hidden)             [T, 640]
q, raw_k  = split(qk, [512, 128])             q -> [T, 4, 128], raw_k -> [T, 128]
q         = rope(q_layernorm(q), pos)         per query, at the query's own position

per complete block b of 4 consecutive key tokens:
  pooled  = mean over the 4 raw_k, in f32, cast back to bf16
  blk_k_b = rope(k_layernorm(pooled), pos of the block's first token)

per query:
  score_b = sum over the 4 heads of relu(q_h . blk_k_b) / sqrt(128)
  keep    = indices of the top min(512, n_blocks) scores
  visible = the 4 tokens of every kept block, plus the tail
            (the (pos+1) % 4 tokens after the last complete block, attended unconditionally)
```

The query side is `qk_norm_rope_gate`'s k side at `n_head` 0, which reads `[tokens, heads,
head_dim]` at the operand's row pitch (the query half of the `heads + 1` projection). It needs the
model's cos/sin table and is bounded by the attention block's fusion row count, above which the
table is not filled; otherwise `rmsnorm` and `rope` run over column slices.

## Block keys

A block key is independent of the query and, once its four tokens exist, permanent. The reference
recomputes block keys inside the per-query loop; the engine computes each once, when the token with
`t % 4 == 3` arrives, and caches it. The indexer's cache is **block keys**, `[n_blocks, 128]` bf16
(64 bytes per token per QSA layer), plus **a tail** of at most 3 raw keys for the next block.

12 QSA layers x 64 B = 768 B per token, about 3% of the main KV cache's 24 KiB per token
(2 heads x 256 x 2 (K and V) x 2 bytes x 12 at bf16): 192 MiB per sequence at 262144 tokens,
24 MiB at 32768. In int8 (`--qsa-bkey-dtype i8`, below) a block key is 136 bytes, 34 a token a
layer.

## Exactness bound

With `pos + 1 <= budget + ratio - 1 = 2051` visible tokens there are at most 512 complete blocks,
the top-k keeps all of them and the tail is always attended, so QSA equals dense causal attention
bit for bit. Below the bound, a step with more than one query per sequence (prefill, mixed or
speculative) issues dense attention, which reads each key once per query tile where the sparse
path reads a table per query. Plain decode (one query per sequence) takes the sparse path at every
context: there is no tile to amortise, and the sparse path stays exercised. The bound is the test
for every change here (see "Checking").

## Attention over a selected set

With the full-attention KV group paged at 4, a compress block is a page and the selected set is a
paged block table, so sparse attention is the existing `attn_paged` with a 512-entry table.

The page size is a kernel constant. The QSA attention units (`r4d_attn_paged_h256_gqa12.hip`,
`r4d_attn_paged_h256_gqa6_bs4.hip`) build `r4d_attn_paged_gqa8.hip` with `G8_BLOCK_SIZE 4`, and
their rows declare `C_EQ("block_size", 4)`. `RadKVGroupDecl` carries no block size: the core takes
it from the resolved attention kernel, and the plugin asks for 4 through the attention op's
`block_size` parameter. **The rows and the `#define` must agree**: a row claiming 16 against a
kernel compiled at 4 resolves at declare and fails at issue.

`attn_paged` takes the block table and `seqused_k` as ordinary operands
(`arch/common/rad_block_attn_gated_fp8.h`), so the sparse attention is issued with `sel` in place
of the sequence's table and `qsa_select`'s `seqused` in place of its length. Nothing in `core/`
knows about QSA.

Declaration order is the reverse of issue order. `attn_paged` declares `sel` and `seqused` as
reads, and the buffer plan comes from declared reads, so the architecture sets them on the
attention block before `attn.declare()` and declares the indexer after it.

## Block-key storage

Block keys live **in their own `RAD_KV_FULL` group, one page per compress block, indexed by page
id**. The scorer reads `bk[page_id]` through a table the KV manager already maintains, and the
top-512 output is a list of attention page ids, which is the block table the attention reads.

A sequence's block-key pages differ from its attention pages, but both groups are paged at
`ratio`, so **block index `b` means the same in both**. Each op takes the table of the group it
serves:

| op | table it takes |
|---|---|
| `qsa_score` | the block-key group's, because it gathers block keys |
| `qsa_select` | the attention group's, because its output is what the attention reads |

Swapping them scores one sequence's keys through another's pages. A `RAD_BUF_PERSIST` array
indexed by `(slot, block)` would be simpler to declare but needs a translation in three places.

A `RAD_KV_FULL` group stores `[pages, kv_heads, block_size, 2*head_dim]`. With `kv_heads = 1`,
`block_size = ratio` and `head_dim = n / (2*ratio)` (here `128 / 8 = 16`), a page holds exactly
one block key of `n` values (`2 * head_dim * ratio == n` is asserted). A group's page size comes
from the first later-declared op with a `block_size` (`Builder::find_kv_consumer`), so the group is
declared immediately before `qsa_block_key`, whose `block_size` sizes it.

**In int8** (`--qsa-bkey-dtype i8`, RadBuildCtx::qsa_bkey_dtype RAD_I8) a page holds the key's `n`
int8 codes, one f16 scale and zero padding: `n + 2` bytes rounded up to the `2 * ratio` a page of
this geometry comes in, so the group is declared RAD_I8 with `head_dim = ceil((n + 2) / (2 * ratio))`,
136 bytes at `n` 128 and `ratio` 4 (8.5 bits a value, 6 bytes of padding). Everything that sizes a
page from the declaration -- the pool, the prefix cache, the residency store -- follows unchanged;
every QSA layer and the MTP head share the group, so all store int8 together. `qsa_block_key` and
`qsa_score` take `key_dtype` i8 (docs/OPS.md), only then, so the bf16 program is the one before the
flag existed, parameter for parameter.

- **The scale rule** is i8h's for a head: `scale = f16(fmin(amax / 127, 512))` over the key as the
  bf16 store would hold it, codes `clamp(rint(x / scale), -127, 127)`, 0 at a zero scale; a NaN is
  dropped from the max and stored as -127, an all-NaN key takes the cap. One scale a key, not a
  32-group: g32 bought 0.0008 of top-512 overlap for 0.37 more bits (prep/aq/m8/bkey/decision.md).
- **The score folds the scale out of the dot**: `relu(scale * (q . codes))` per head. A bf16 query
  times a code is exact in f32, so the products are the bf16 path's kind and the scale is one
  multiply a block and head; the matrix form widens codes to bf16 (exact) for the same WMMA.
- **Measured** on two R9700s (Flash-Next `--tp 2`, a 30 210-token prefill, layer 47's own
  selection, ovl.sh): top-512 overlap 0.932 mean against bf16 keys, the same as an fp8 attention
  cache's against bf16 (0.928); block keys rebuilt in isolation keep 0.996 (the rest is the drift
  of the earlier layers). KL against the bf16-cache reference, 4 long documents: fp8 0.0262 ->
  0.0256, i8h 0.0236 -> 0.0243, both differences inside their 95% paired bootstrap interval;
  window residency byte-identical off against on.

## The tail

A block key needs `ratio` consecutive raw keys; at decode three come from earlier steps. The tail
is a `RAD_KV_LINEAR` group with `state_dim` `{ratio + max_spec, head_dim}`, bound to every QSA
layer including the MTP head's. The alternatives all produce fluent wrong text:

| a tail stored as… | what goes wrong | when |
|---|---|---|
| one buffer for the whole model | layer L pools layer L-1's raw keys, because `index_qk_proj` is per layer | every decode step: three of every four keys in a completing block |
| a plane indexed by the batch row | a request that outlives an earlier one inherits its keys, because the scheduler orders a step by priority and stable-sorts decode first | any step after a concurrent request retires |
| `ratio` rows indexed `pos % ratio` | a rejected draft restarts the next step at a position this one already overwrote, so a completed block is rebuilt from a future token's key | any speculative step that straddles a block boundary and does not accept everything |

The tail is read on a later step than the one that wrote it and is keyed by layer, sequence and
position; `cu_seqlens` gives only a token's sequence within the step. A KV group slot is zeroed on
hand-out and checkpointed with the linear state, so a resume onto a half-built block reads the keys
it was built from. A `RAD_KV_WINDOW` of width `ratio` holds only the step's last four tokens, which
is the completing block at decode but not within a prefill chunk.

`qsa_work` gathers the blocks completing this step into the flat list `qsa_block_key` reads, and
`qsa_tail_store` writes this step's raw keys into the state. **`qsa_work` is issued before
`qsa_tail_store`**, because the gather reads tail keys the store overwrites. Position `p` lands at
row `p % ring`. The work list has `ceil(T / ratio) + n_seq` items; items past the real count get
`page = -1` and are skipped, so it is one launch per layer per step.

## Per-query selection (prefill)

At prefill every query in a chunk selects its own set from one shared history. Both scoring ops
take `cu_seqlens`, so `m` is a query row while the block table stays the sequence's. Each query is
bounded by its own position, `n = min(nc[seq], (pos + 1) / ratio)` (at decode, `nc[seq]`).

The attention is issued **one row per query**: `attn_paged` indexes its block table and length by
row, so `T` queries issued as `T` rows of `q_len` 1, each with its own `sel` and `seqused`, is
per-query sparse attention. Launches are chunked at `max(max_seqs, 256)` rows (the gated form takes
the whole step), because the decode kernel's split-KV scratch is sized at declare from `max_seqs`
and `q_len`.

**The partial page is decided by position.** At decode `bt[nc] == -1` on a block boundary, since
nothing is allocated past the sequence, so `bt[n] >= 0` would work there. Mid-chunk, the same
query finds a later token's live page in `bt[n]`, which `seqused` would then cover. The partial
page is appended only when `(pos + 1) % ratio != 0`.

## Hierarchical indexer (`--qsa-pool N`, experimental)

DeepSeek-V4.1-Flash's hierarchical indexer, as a switch to measure on a model that was not trained
for it. Off (`0`, the default) declares and issues nothing new: the program and every byte are
today's. With `N` blocks (at least the selection's `topk`, 512 on Flash-Next; at most 4095):

- the **first** trunk QSA layer (the one whose `declare` creates `kv_qsa_bkey`) runs its score and
  select as before, then an `iota` refills `qsa_ident` (`[max_seqs, max_blocks]`, `ident[s][b] =
  b`) and a second `qsa_select` at `topk` `N` reads its own (unchanged) plane with `qsa_ident` as
  the table. Its output `qsa_pool` `[max_tok, N + 1]` is therefore that layer's top-`N` *block
  indices* under the select's exact rule (ties, causal bound, partial block), one row a query;
- every **later** trunk QSA layer issues `qsa_pool_mask` between its score and its select: the
  columns of its plane its row of `qsa_pool` does not name become `-inf`, so its select takes its
  own top-`topk` from the pool. Everything stays on the device: no host sync, no new batch field;
- the **MTP head** keeps its own selection (no mask): its pass is not the trunk step whose rows
  wrote the pool.

The selection is exact whenever `N` covers a query's history (the pool is then every live block),
so below `N * ratio` tokens the text must be byte-identical to `--qsa-pool 0`. Above it, a later
layer can no longer pick a block the first layer ranked below `N`.

Left out on purpose (the smallest version that measures the quality): a later layer still scores
every block, so the switch saves no indexer time or memory yet. A pool-only score (`qsa_score`
over a gathered table of `N` pages) is the follow-up if the KL measurement justifies it. The block
keys, the tail, the prefix cache and window residency are untouched: the pool is per-step scratch,
and the select a later layer runs still maps indices to pages (cold entries included) through the
real table, so `qsa_page` sees the same kind of selection as before.

## Cost

The indexer is the only part of a decode step that grows with context: `qsa_score` and
`qsa_select` scale with the number of compressed blocks, once per layer. `r4d_selftest --perfqsa
CTX [M] [iters]` times them at a real context length with no model, passing the declared width and
the live context separately (launch geometry comes from the declaration, work from `nc`).

- `qsa_score` reads the key history once per 4-row query tile, matching a decode sequence's
  `1 + n_spec` rows at depth 3; a tile straddling two sequences is processed per part.
- `qsa_select` splits the block list into chunks that each find a local top-k, exact because a
  chunk's local top-k contains every global top-k member in it. The proof and the tie rule are in
  `libr4d/r4d_qsa_select_bf16.hip`.
- Wave reductions use DPP (`row_xmask` for XOR offsets 8..1, `v_permlanex16_b32` across rows), not
  `__shfl_xor`, which lowers to `ds_bpermute_b32` and queues on the LDS pipe. Same lanes and add
  order, so bit-identical (`libr4d/r4d_common.h`, `r4d_wave_sum`).

**Any two forms of the selection kernel must produce the same bytes**, not only the same set: the
attention sums its softmax over pages in table order, and float addition is not associative.

**The score plane is the one buffer whose size is a product of the deployment's bounds** — a row
of `max_ctx / ratio` floats a query, 512 MiB at a 2048-row step and 256K of context. Under
`--prefill-span`, where every step that mixes sequences fits in a MoE pass
(`RadBuildCtx::max_mixed_tok`), it is declared a pass tall instead of `--max-num-batched-tokens`
rows: a wider step is one sequence, scored and selected in windows of the plane's rows, each handed
as a sequence of its own rows. Every operand a query row reads is its own (query, position,
selection), so a row is scored and selected exactly as in one pass, and a row two windows share is
written twice with the same bytes.

**The caller bounds the history.** Both ops size their grid from the declared width (the live count
is device data), so a short prompt would otherwise launch for the whole `--max-model-len`. The
architecture narrows the score operand's column count to a host-side ceiling from the batch, with
the pitch unchanged. **Every term of the ceiling is generous**, because an underestimate silently
drops blocks.

## Checking

There is deliberately no switch that forces dense attention above the bound, where it is not the
model the checkpoint was trained as. libref implements all five QSA ops (`ref_qsa_block_key` in
`libref/ref_attn.cpp`; `ref_qsa_score`, `ref_qsa_select`, `ref_qsa_tail_store` and `ref_qsa_work`
in `libref/ref_qsa.cpp`), so each device kernel has a row to compare against. No oracle gives the
whole-subsystem comparison, a dense run against a sparse one above the bound. Changes are checked
with:

- **`r4d_selftest` oracles**: `qsa_block_key` and `qsa_score` against f64 evaluations,
  `qsa_work` and `qsa_tail_store` against a host walk of the batch, `qsa_select` against a host
  sort, a prefill case for the per-query join, the wide-launch form against the single-workgroup
  form, and split against unsplit at bit identity on mostly-tied scores, so the tie rule decides;
- **`RADIANCE_DEBUG_QSA`**: prints `nc`, `nsel`, `seqused` and the head of the table for the first
  few steps. `nc` above `topk` is intended dropping, not a divergence. It synchronises the stream.

**Count the reach, not the prompt.** The bound is on the context each query sees, so a prompt a
few tokens under it crosses the bound during decode and is expected to start differing there.
