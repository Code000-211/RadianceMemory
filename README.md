# [WIP]RadianceMemory — window residency for sparse-attention KV

RadianceMemory keeps a long conversation's KV cache on the card **only where the model actually reads
it**. Everything else sits in pinned host memory and comes back for the steps that need it. Those
steps produce the same bytes they would have produced with everything resident.

It is built for models whose attention reads a *selected* part of the history. For Qwen3.8-Flash-Next
(QSA) at 200 K tokens of context, the attention cache is 4.80 GiB a sequence, and a decode step reads
about 48 MiB of it. The rest is VRAM nothing touches.

This document covers only the KV memory system. The engine it runs in is radiance, documented in
[`docs/RADIANCE.md`](docs/RADIANCE.md) and the rest of `docs/`.

---

## Contents

1. [The idea in one picture](#1-the-idea-in-one-picture)
2. [What a block can be](#2-what-a-block-can-be)
3. [Which blocks stay on the card](#3-which-blocks-stay-on-the-card)
4. [A decode step: the pager](#4-a-decode-step-the-pager)
5. [The sweep: how blocks leave the card](#5-the-sweep-how-blocks-leave-the-card)
6. [Dense steps: promotion](#6-dense-steps-promotion)
7. [The prefix cache: cold blocks stay reusable](#7-the-prefix-cache-cold-blocks-stay-reusable)
8. [Memory accounting](#8-memory-accounting)
9. [Correctness: why the output does not change](#9-correctness-why-the-output-does-not-change)
10. [Using it](#10-using-it)
11. [What it measures](#11-what-it-measures)
12. [How it relates to `--kv-resident` (KV streaming)](#12-how-it-relates-to---kv-resident-kv-streaming)
13. [Limits and open points](#13-limits-and-open-points)
14. [Where the code is](#14-where-the-code-is)

---

## 1. The idea in one picture

A QSA layer does not attend to the whole history. A small **indexer** scores every 4-token block of
keys and keeps the best 512 blocks. The attention then reads those blocks plus the incomplete tail.
Which 512 blocks are kept changes from step to step, but two regions are read on nearly every step:
the very start of the conversation (the attention sinks) and its end (the recent tail).

```
 one sequence, 12 542 tokens = 3 136 blocks of 4 tokens
 ┌────────────┬──────────────────────────────────────────────────────┬──────────────────┐
 │ sink       │                     the middle                       │ tail             │
 │ 512 tokens │                                                      │ 2 048 tokens     │
 │ VRAM       │        host store, brought back only when selected   │ VRAM             │
 └────────────┴──────────────────────────────────────────────────────┴──────────────────┘
      ▲                     ▲          ▲        ▲                             ▲
      │                     │          │        │                             │
      └──── read every step ┘   a few blocks the indexer picks this step       └ read every step
```

Without RadianceMemory, the whole bar is in VRAM. With it, the middle leaves the card, and the
handful of middle blocks a step's selection names are paged back in for that step only.

---

## 2. What a block can be

The block table each sequence holds is the attention's map from a position in the history to the
bytes of that position's keys and values. An entry has one of three meanings:

| entry | meaning | where the bytes are |
|---|---|---|
| `>= 0` | a pool block id | in the VRAM KV pool |
| `-1` | empty | nowhere — no block at this position |
| `-2 - slot` | **cold** | in slot `slot` of the pinned host residency store |

The encoding is part of the plugin ABI (`rad_kv_cold_entry`, `rad_kv_entry_cold`,
`rad_kv_entry_slot` in `abi/rad_runtime.h`). A cold entry keeps its place in the table, so the
sequence's context is the same table of the same positions. Only where one entry's bytes live has
changed.

A block's life:

```mermaid
stateDiagram-v2
    [*] --> Resident : the sequence grows (ensure)
    Resident --> Demoting : the sweep picks it
    Demoting --> Cold : copy to the store has landed\n→ pool block freed
    Cold --> Resident : promotion (dense step)\nnew pool block, copy back
    Cold --> Paged : a decode step selects it
    Paged --> Cold : step ends\n(scratch page is per step)
    Resident --> Cold : re-point (unchanged block\nwhose origin slot is still held)\n→ no copy at all
    Resident --> [*] : sequence freed
    Cold --> [*] : last reference gone\n→ slot freed
```

* **Demoting.** The copy to the host store is in flight. The entry already reads cold, but the pool
  block keeps its reference until the copy has landed, so nothing can reuse it while it is being
  read.
* **Paged.** A decode step selected the cold block. The pager copied it into one of that step's
  scratch pages, and the step's selection points there. On the next step that copy is gone and the
  entry is cold again. Nothing in the table changed.

---

## 3. Which blocks stay on the card

The model plugin declares how much of the history its consumer reads. In `arch/common/rad_qsa.h` the
attention group `kv_attn` declares the indexer's budget, 2 048 tokens, through
`rad_kv_set_residency_window`. The block-key group `kv_qsa_bkey` declares **nothing**: the indexer
scores every block key on every step, so those pages must stay resident. At bf16 they are 3.25 of the
29.25 KiB a 4-token block costs, and no window can shrink them. `--qsa-bkey-dtype i8` stores them as
int8 with one f16 scale a key, 1.73 KiB a block (the page 136 bytes a layer), so the resident floor
under residency drops from 832 to 442 bytes a token a card ([`docs/QSA.md`](docs/QSA.md),
"Block-key storage").

Only a group that declares a window is ever demoted. For that group a sequence keeps two ends:

```
 table entry index   0 ............ H-1 | H ........................ N-T-1 | N-T ......... N-1
                     └─ sink: kept ─────┘└──── demotable ──────────────────┘└─ tail: kept ───┘
   H = ceil(--kv-residency-sink-tokens / block)      T = ceil(declared window / block)
   the last entry (the partial block an append writes into) is never demoted
```

**Why two ends and not one window.** The selection is not positional: the top 512 come from
anywhere. If only a tail window stayed resident, the sinks would be demoted and then promoted on
every step, which costs more than it saves.

**What a sweep never demotes:**

* the sink and the tail,
* the last, partial block, because the next token is written into it,
* a block another holder has a reference on and cannot follow into the store, such as a forked
  child. The prefix cache can follow, as described in [§7](#7-the-prefix-cache-cold-blocks-stay-reusable).

---

## 4. A decode step: the pager

The host never learns which blocks a decode step reads. `qsa_select` decides that on the card,
inside the step. The fetch therefore also happens on the card: a kernel, `qsa_page`, runs between
the selection and the attention of every QSA layer.

```mermaid
sequenceDiagram
    autonumber
    participant IDX as qsa_score / qsa_select<br/>(indexer)
    participant SEL as sel table<br/>[rows, 513]
    participant PG as qsa_page<br/>(the pager)
    participant ST as host store<br/>(pinned, host-mapped)
    participant SC as scratch pages<br/>(VRAM, per row)
    participant AT as attn_paged

    IDX->>SEL: top 512 blocks + tail<br/>(cold entries rank like resident ones)
    PG->>SEL: read every entry
    alt entry is cold (-2 - slot)
        PG->>ST: read this layer's fragment of slot
        PG->>SC: write it into scratch[row][j]
        PG->>SEL: entry := scratch[row][j]
    else resident or empty
        Note over PG,SEL: left as it is
    end
    AT->>SEL: read the table
    AT->>SC: cold blocks, now scratch pages
    Note over AT: the attention kernel is unchanged —<br/>it only ever sees pool block ids
```

Properties of the pager:

* **No host synchronisation.** The store is allocated host-mapped, and the card reads it over PCIe
  directly.
* **Deterministic.** Every position `(row, j)` has exactly one writer and there are no atomics. The
  result is byte-identical to the reference in `libref`, which `r4d_selftest` checks.
* **Per layer.** The plugin folds the layer's offset into the store pointer on the host, so the
  kernel never reads host-side tables.
* **Counted.** The kernel adds the number of entries it rewrote to one int64 on the card (`paged_in`).
  `/stats` reads that counter back on a poll, never per step.
* **Scratch.** Each decode row has `topk + 1` scratch pages of its own. They are reserved when the
  pool is carved and are valid for one step.

`qsa_select` changed in one place: a block is absent exactly when its entry is `-1`. It used to be
"any negative". This is what lets a cold block be selected at all. Its *score* is unaffected, because
`qsa_score` reads `kv_qsa_bkey`, which is never demoted.

---

## 5. The sweep: how blocks leave the card

After every committed decode step, the scheduler sweeps. It demotes at most a bounded number of
blocks per step, across all sequences, and only one sweep is in flight at a time.

```mermaid
sequenceDiagram
    autonumber
    participant SCH as scheduler<br/>(end of decode step)
    participant KV as KV manager
    participant TX as transfer stream
    participant CS as compute stream

    SCH->>SCH: reap: has the last sweep landed? (event poll, no wait)
    opt landed
        SCH->>KV: residency_demote_landed()<br/>→ pool blocks back to the free list
    end
    SCH->>KV: residency_demote(seq, sink, tail, ≤ 64 blocks)
    KV-->>SCH: moves (block → slot), entries now read cold
    SCH->>TX: issue_moves(to_host)
    TX->>CS: wait for everything compute was given
    TX->>TX: gather_rows → copy to store slots
    Note over SCH: the next build uploads the changed table entries.<br/>A step issued earlier still reads the old pool ids,<br/>which stay valid until the copy has landed
```

The rules that keep this safe without any host wait:

1. **A block is freed only after its copy has landed.** The copy waited on the compute stream, so
   every step that could read the block in place has already been queued ahead of it.
2. **A freed block's new owner writes it only in a later step** on the same in-order compute stream,
   after every earlier reader.
3. **A promotion waits on the compute stream before it writes**, because it writes into a block the
   pool just handed out, which a step issued ahead may still be reading under its previous owner.
4. **One sweep in flight.** Landing is a poll of the newest transfer, so a sweep issued every step
   would keep the newest transfer always pending and never free anything.

---

## 6. Dense steps: promotion

Prefill and mixed steps read the table densely, every entry. A cold entry there would be a store slot
read as a pool block id. Before such a step, the scheduler therefore promotes every sequence in it:

```mermaid
flowchart LR
    A[plan a prefill / mixed step] --> B{sequence has<br/>cold entries?}
    B -- no --> E[build the batch]
    B -- yes --> C[residency_promote_all<br/>fresh pool block per cold entry]
    C -->|pool full| P[ask the prefix cache for room]
    P -->|still full| Q[preempt THIS sequence<br/>by recompute]
    C --> D[issue_moves to_device<br/>compute waits on the copy]
    D --> E
    E --> F[the batch holds pool ids only]
```

Promotion is all or nothing per sequence. If the pool cannot hold a sequence's cold blocks even after
the prefix cache has given up room, **that** sequence is preempted and re-prefilled. Preempting the
back of the running set instead would usually hit the prompt that made the step mixed. That prompt
would be readmitted at once into the same failing step while the cold sequence never decodes, which
is a livelock that a test now covers.

---

## 7. The prefix cache: cold blocks stay reusable

A conversation that comes back for its next turn should hit the prefix cache, cold middle included.
To make that possible, store slots are **reference counted**. A cold entry can sit in several
sequence tables and in prefix-cache entries at the same time, and the slot is freed when its last
holder lets go.

```mermaid
flowchart TB
    subgraph S1["Turn 1 - decode"]
        A1["Sequence A"]
        C1["Entry 700 becomes cold, slot 42<br/>slot 42 refs = 1"]
    end
    subgraph S2["Turn 1 ends - publish"]
        PC["Prefix cache entry for block 700<br/>holds cold slot 42<br/>slot 42 refs = 2"]
        R1["Sequence A freed<br/>slot 42 refs = 1, the cache's"]
    end
    subgraph S3["Turn 2 - same conversation plus a new message"]
        L["Lookup hits the whole chain,<br/>cold entries included"]
        AD["Sequence B adopts cold slot 42<br/>slot 42 refs = 2"]
        PR["Dense step promotes:<br/>new pool block, copy back,<br/>B remembers origin slot 42"]
        RD["Later sweep, block unchanged:<br/>re-point to slot 42, no copy"]
    end
    A1 -->|sweep| C1
    C1 --> PC
    PC --> R1
    R1 --> L
    L --> AD
    AD --> PR
    PR --> RD
```

* **A hit is a copy, not a recompute.** On Flash-Next a warm 8 887-token prefill went from 1 601 ms
  back to 92.7 ms. With residency off it is 70.1 ms.
* **Re-point instead of re-copy.** A block promoted from slot S and not written since (prefix blocks
  are immutable, only the partial tail is written) is demoted by pointing its entry back at S. No
  bytes move, and the store does not gain a copy of the same prefix every turn.
* **The idle tiers** (the host tier for finished conversations, `--prefix-cache-mib`) are never handed
  an entry with a cold component. They gather pool blocks, and that part's bytes are already in host
  memory.

---

## 8. Memory accounting

```
                       VRAM (per rank)                              pinned host
  ┌──────────────────────────────────────────────────────┐   ┌───────────────────────────┐
  │ KV pool                                              │   │ residency store           │
  │ ┌──────────┬──────────────────────┬────────────────┐ │   │ --kv-residency-mib        │
  │ │ scratch  │ resident blocks      │ free blocks    │ │   │ one block a slot,         │
  │ │ pages    │ (sinks, tails, block │  ← demoted     │ │   │ every rank's shard        │
  │ │ held for │  keys, short seqs)   │    blocks      │ │   │ at its own offset         │
  │ │ the pager│                      │    land here   │ │   │                           │
  │ └──────────┴──────────────────────┴────────────────┘ │   └───────────────────────────┘
  └──────────────────────────────────────────────────────┘
```

| item | size | where it is charged |
|---|---|---|
| scratch pages | `(window/block + 1) × max_seqs × (1 + n_spec)` blocks per residency group, about 208 MiB on Flash-Next at 8 decode rows | fixed costs, before the VRAM split, so the budget resolver sees them |
| residency store | `--kv-residency-mib`, slot = one block of one group (every rank's shard); under a mixed `--kv-recipe`, the widest layout's block (a `bf16` layer's, twice fp8's, where the recipe has one) | pinned host memory, opened host-mapped |
| per-block bookkeeping | 4 bytes a block (the table entry already holds the slot) | none worth counting |

At 200 K tokens a sequence holds 50 K attention blocks. With a 512-token sink and a 2 048-token tail
resident, it needs about **51 MiB instead of 4.80 GiB** on the card, before the shared scratch. Its
block keys stay resident beside that: 159 MiB at bf16, 84 MiB with `--qsa-bkey-dtype i8`. The block-key
group's page is whatever its declaration says, so the pool, the prefix cache and the residency store
size it from there unchanged.

### Tiered precision: the demoted middle in a narrower layout

A recipe line may name a second layout for the blocks the sweep demotes (RAD_KVQ_DEMOTE,
`abi/rad_builder.h`):

```
vmean kv_vmean.txt
layer 3  i8h demote k4h_v2h vmean        # resident i8h, the store k4h_v2h
layer 15 fp8 rotated demote k4h_v4       # fp8 must be rotated: the demoted block keeps its basis
```

The sink and the tail, read on almost every step, stay in the resident layout; the middle sits in
the store in the demote layout and comes back expanded. Served: i8h or a rotated fp8 layer
resident, k4h_v4 or k4h_v2h demoted.

* **Where the requantise and the expand run.** On the card, as the op `kv_transcode`
  (docs/OPS.md): the demotion gathers the block's layers, transcodes every (token, head) row into
  the demote layout and copies those bytes to the slot; a promotion copies them back and expands
  them before the scatter; the pager (`qsa_page` with `store_layout` and `pool_layout`) expands a
  cold fragment straight into its scratch page. One codec serves all three, so attention reads one
  layout per layer and no attention kernel changed.
* **Basis and V mean.** The transcode decodes and encodes values in the stored units and does not
  rotate, centre or rescale. Both layouts therefore share the layer's rotation and V mean: a k4h
  demote layout needs a rotated layer, and `vmean` (accepted when either layout has integer V)
  centres the resident and the demoted bytes alike.
* **Store sizing.** A tiered layer's slot fragment is its rows at the demote width, so a slot is
  the widest *demote* block (startup prints `stored as k4h_v2h, N KiB a block`). On Flash-Next at
  `--tp 2` a rank's i8h block (12 layers + the head's, one KV head, 4 tokens) is 13 × 2 064 =
  26.2 KiB in VRAM and 13 × 896 = 11.4 KiB in the store as k4h_v2h, 13 × 1 152 = 14.6 KiB as k4h_v4.
* **Re-pointing.** A promoted tiered block keeps its origin slot even when no cache holds it, so an
  unchanged block re-demotes by pointing back (no bytes, no second rounding). A block is rounded
  through the demote layout once in its life; the cost is that a resident block can hold a slot.
* **Prefix cache.** Cold entries hold demote-precision bytes, and a returning conversation is
  promoted from them: its middle comes back at the demote precision, never better. The idle tiers
  still take only resident blocks.
* **The emulation** (`RADIANCE_DEBUG_KV_TIER=emulate`, with the same flags): the sweep picks exactly
  the blocks it would demote and the mover runs the demote → expand round trip on them in place;
  nothing moves. Residency-on output is compared with it byte for byte, both under
  `RADIANCE_DEBUG_KV_TIER=sync` semantics (each sweep is waited for, so which step first reads a
  demoted block does not depend on transfer timing). In the KL mode, which scores prefill chunks and
  runs no decode step, the emulation sweeps after every step, unbounded.

Measured on two R9700s (Flash-Next `--tp 2`, bf16-cache reference, residency on, KL through the
emulation; `prep/aq/m8/{kl,recall,mr}/fix-*.json`, all taken after the fix of the
sweep's shared index buffer, below).
Three corpora: `long.jsonl` (4 documents of ~34K tokens, every position past ~30K scored), `recall.jsonl`
(a ~2 000-token passage repeated ~26K tokens later, only the repeat scored) and `mr.jsonl` (four
conversations of four turns, each pasting ~6 000 tokens of a file and answered by the model itself at
bf16, then a fifth turn asking about the first file; only that last answer, 256 tokens, is scored):

| configuration | store a block (both ranks) | long KL / top-1 / ΔNLL | recall KL / top-1 | multi-round KL / top-1 / ΔNLL |
|---|--:|--:|--:|--:|
| uniform i8h | 52.5 KiB | 0.0236 / 94.77 % / −0.0024 | 0.0103 / 99.77 % | 0.0032 / 98.63 % / −0.0016 |
| i8h, demote k4h_v4 + vmean | 29.25 KiB | 0.0284 / 93.91 % / +0.0012 | 0.0151 / 99.62 % | 0.0047 / 98.43 % / −0.0003 |
| i8h, demote k4h_v2h + vmean | 22.75 KiB | 0.0408 / 93.00 % / +0.0028 | 0.0183 / 99.62 % | 0.0063 / 97.55 % / **+0.0043** |
| uniform k4h_v2h + vmean | 22.75 KiB | 0.0833 / 89.90 % / **+0.0132** | 0.0214 / 99.61 % | 0.0177 / 96.27 % / **+0.0190** |

For scale, the bf16 model itself computed in another prefill order measures 0.0243 (long) and 0.0033
(multi-round). A `k4h_v4` middle stays within that noise everywhere; a `k4h_v2h` middle costs less than
half of what `k4h_v2h` everywhere costs, but on the multi-round corpus its NLL change is significant
(+0.0043, 95 % from +0.0004 to +0.0111; uniform: +0.0190). ΔNLL intervals are block bootstraps.

A paged block moves its demote-layout fragment over PCIe: 896 B a layer and rank as k4h_v2h,
1 152 B as k4h_v4, 2 064 B as i8h. Residency on against the emulation, 12 452 prompt tokens and 256
greedy tokens, is byte-identical for both demote layouts (`prep/aq/m8/identity-tier.sh`).

**The shared index buffer.** Before this, every batch of a sweep gathered its blocks through one
pinned index buffer that the transfer stream read only when it got past compute, so a batch could
gather the last batch's blocks: residency output was then timing-dependent, tiered or not. Each batch
now owns an entry of a ring of 64, with a stream sync when it wraps; two KL runs of a tiered recipe
are byte-identical. **Open:** with a tiered recipe, int8 block keys (`--qsa-bkey-dtype i8`) and the MTP
drafter together, residency on and the emulation diverge (identical with `--num-speculative-tokens
0`); the drafter's acceptance falls from about 0.68 to 0.10 once blocks really move, also untiered.

### Two demote layouts, chosen by the QSA selection count

A tiered line may name two demote layouts, the wide one first (RAD_KVQ_DEMOTE2, ABI 43):

```
layer 3 i8h demote k4h_v4 k4h_v2h vmean    # each demoted block k4h_v4 or k4h_v2h
```

and `--kv-demote-split F` (default 0.5) is the fraction of demoted blocks that take the wide one.

* **The counter.** `qsa_count` (docs/OPS.md), issued by each trunk layer's indexer between
  `qsa_select` and the pager, adds one to a u32 per pool block (`RadKVGroupBatch::residency_count`)
  for every query whose selection was full -- it kept `topk` blocks out of more (or exactly as
  many) -- and kept that block. Prefill rows all count; of a decode sequence's rows only the first,
  the committed token's (the later ones verify drafts); the MTP head counts nothing. The engine
  zeroes a block's word when it hands the block out and copies the array to pinned host memory
  after every step, ahead of the sampled ids, so the commit's sweep reads that step's counts. So
  a count is the selections a block got while resident; the residency window is the window, and
  nothing decays.
* **The split rule.** At a block's first demotion its score is its count over what a uniform
  choice would have given it, `count / ln(now / max(end, window))` (a query at q keeps topk of q /
  ratio blocks); the block is wide when fewer than `F (n + 1)` of the last n <= 64 blocks the group
  decided scored at least as high, ties narrow. A fraction rather than a count threshold because a
  count's scale moves with the history length, the chunking and the layer count, while the
  fraction fixes the bytes; a short, recent look back because scores still drift with position (a
  sweep after a long prompt demotes thousands of blocks oldest first). Realized: 27 / 48.5 / 71 %
  wide at 0.25 / 0.5 / 0.75 in the KL runs (ties at a zero count go narrow), 50.4 % in the
  identity run at 0.5.
* **Slots.** A slot is the wide layout's block; a narrow block's rows sit at the front of each
  layer's fragment, so the layer offsets are the wide ones. Rank 0's manager records each slot's
  layout and names it in every move (`BlockMove::layout`), so every rank stores and expands a block
  the same way; each rank's mover writes the slot's byte into its device table
  (`RadKVGroupBatch::residency_slot_layout`) behind the slot's bytes, and the pager (`qsa_page`
  with `store_layout2`) expands each cold entry from its slot's layout. Promotion, re-pointing and
  the prefix cache's cold entries read the slot's layout; a block is still decided and rounded once.
* **TP.** Only rank 0 counts and decides; the other ranks' batches carry no counter.
* **The emulation** follows the same decision from the same counts.

Measured as above (`prep/aq/m8/card17.sh`, `{kl,recall,mr}/imp-*.json`); link bytes are a demoted
block's both ranks in the serving geometry (13 layers), the store slot is 29.25 KiB at every split:

| middle | link a block | long KL / top-1 / ΔNLL | recall KL / top-1 / ΔNLL | multi-round KL / top-1 / ΔNLL |
|---|--:|--:|--:|--:|
| k4h_v4 | 29.25 KiB | 0.0284 / 93.91 % / +0.0012 | 0.0151 / 99.62 % / −0.0084 | 0.0047 / 98.43 % / −0.0003 |
| split 0.75 (71 % wide) | 27.4 KiB | 0.0279 / 94.22 % / −0.0020 | 0.0171 / 99.64 % / −0.0089 | 0.0046 / 98.04 % / −0.0017 |
| split 0.5 (48.5 % wide) | 25.9 KiB | 0.0319 / 93.79 % / +0.0040 | 0.0175 / 99.59 % / −0.0082 | 0.0047 / 98.14 % / −0.0012 |
| split 0.25 (27 % wide) | 24.5 KiB | 0.0349 / 93.71 % / +0.0028 | 0.0164 / 99.62 % / −0.0090 | 0.0053 / 98.63 % / −0.0013 |
| k4h_v2h | 22.75 KiB | 0.0408 / 93.00 % / +0.0028 | 0.0183 / 99.62 % / −0.0073 | 0.0063 / 97.55 % / +0.0043 |
| control: split 0.5 by a hash (50.7 % wide) | 26.0 KiB | 0.0328 / 93.60 % / +0.0001 | 0.0152 / 99.64 % / −0.0046 | 0.0048 / 97.94 % / +0.0014 |

No ΔNLL of a split is significant (block bootstraps). The mixed middles sit between the two
uniform ones and, on the long corpus, below the byte-weighted line; but the control
(`RADIANCE_DEBUG_KV_SPLIT=hash`: the same rule and fraction scored by a hash of the block's place)
is as good: importance minus control at 0.5 is −0.0009 KL long (95 % −0.0024…+0.0003), +0.0024
recall (−0.0012…+0.0052), −0.0002 multi-round (−0.0008…+0.0006). So far the gain is the byte mix,
not the selection signal. Residency on against the emulation at split 0.5 is byte-identical with
`--num-speculative-tokens 0`; with the MTP drafter the two runs decide differently, since the
drafter's acceptance collapses under real residency (above: 206 decode steps instead of 79) and
decode counts follow the steps.

---

## 9. Correctness: why the output does not change

The only things that differ between a resident run and a paged one are where bytes live and which
page id the attention reads them through. The kernels never see a cold id:

```mermaid
flowchart LR
    subgraph decode
        S[qsa_select] --> P[qsa_page] --> A[attn_paged]
    end
    subgraph dense [prefill / mixed]
        PR[promote all cold entries] --> D[dense attention]
    end
    X[cold entry] -. never reaches .-> A
    X -. never reaches .-> D
```

The checks:

| check | what it shows | where |
|---|---|---|
| **identity on the card** | Flash-Next `--tp 2`, a 12 542-token prompt, 256 greedy tokens: paged output **byte-identical** to the fully resident run, with `paged_in > 0` **and** `freed_bytes > 0` (a run that pages nothing, or frees nothing, fails) | `prep/60-flashnext-identity.sh` |
| pager vs reference | `qsa_page` on the card equals libref byte for byte: aligned and odd fragments, a skewed store, the pool in its real KV shape | `r4d_selftest --case qsa_page_u8` |
| selection with cold entries | a cold entry is chosen exactly where a resident one with the same score would be, in both ranking modes | `r4d_selftest --case qsa_select_i32` |
| a block comes back intact | demoted, the pool overwritten, promoted: byte for byte, and only the named blocks moved | `mem_test` (host backend) |
| no leaks | slots and blocks across demote, promote, preempt, rollback, fork, eviction, trim, expiry, re-plan | `mem_test` |
| scheduler | sweep bound, sink and tail never demoted, dense steps see pool ids only, no livelock on a full pool | `sched_test` |

---

## 10. Using it

```sh
radiance --model flashnext.rad --tp 2 \
         --kv-residency-mib 4096 \           # the host store; 0 (default) = off
         --kv-residency-sink-tokens 512      # start of a conversation kept resident
```

* A model must declare a window for anything to happen. Today that is the QSA family
  (`arch/qwen4exp_fp8`). On other models the flag does nothing, and startup says so.
* The flag is exclusive with `--kv-resident` (KV streaming, [§12](#12-how-it-relates-to---kv-resident-kv-streaming)).
  The server refuses both together.
* Startup prints the store, its slot count, the scratch reservation and the window, for example
  `kv residency: window enforced -- after a decode step, blocks outside the sink and the 2048-token
  tail leave the pool for the store, 16 per batch and at most 64 per step`.

**Counters**, in `/stats` under `kv_residency` and per session in `/sessions`:

| field | meaning |
|---|---|
| `demoted`, `promoted` | blocks moved to the store / back to the pool |
| `freed_bytes` | VRAM the landed demotions gave back to the pool |
| `paged_in` | cold entries the pager rewrote on the card (summed over QSA layers and decode rows; rank 0) |
| `skipped` | blocks a sweep left (shared, partial, store full) |
| `link_bytes` | bytes moved over PCIe, both directions |
| `host_bytes`, `cap_bytes` | store in use / store size |
| `scratch_bytes` | VRAM the pager's scratch pages hold |

---

## 11. What it measures

On two Radeon AI PRO R9700 (gfx1201), Flash-Next `qwen3.8-next-flash-fp8-iq4r-moe` at `--tp 2`,
BetterBench with a warm cache (`unique_nonce: false`), 2 passes, everything else equal:

| | residency off | residency on |
|---|--:|--:|
| warm prefill TTFT, 3 013 tokens | 39.2 ms | 42.4 ms |
| warm prefill TTFT, 8 887 tokens | 70.1 ms | 92.7 ms |
| decode, chat / code / summarization (t/s) | 166.4 / 196.8 / 134.3 | 163.9 / 192.4 / 135.3 |
| VRAM given back (per rank) | — | 200 MiB |

Decode is unchanged within the noise, though two passes are few. The warm prefill pays for the copy
back of the conversation's cold middle. Before the prefix cache learned to index cold blocks
([§7](#7-the-prefix-cache-cold-blocks-stay-reusable)), the same two numbers were 596 ms and 1 601 ms.

These are short contexts. The saving grows with context length, because the sink and tail stay
fixed while the middle grows, and the measurement that matters for the feature's purpose
(sessions per card at 100 K+ tokens) is still to be run.

---

## 12. How it relates to `--kv-resident` (KV streaming)

radiance 1.15 ships its own answer to the same problem, KV streaming ([`docs/KV-STREAMING.md`](docs/KV-STREAMING.md)).
The two are alternatives, and a server runs one of them.

| | window residency (`--kv-residency-mib`) | KV streaming (`--kv-resident N`) |
|---|---|---|
| authoritative copy | **VRAM**; the unread middle is demoted | **host**; a VRAM ring caches what is read |
| what stays on the card | positional: sink + tail of the declared window | adaptive: a ring of N blocks, CLOCK eviction by what the selection reads |
| fetch on the card | `qsa_page` → per-step scratch pages | `kv_resolve` → ring slots with a residency map |
| smallest useful context | ~2.6 K tokens (sink + tail) | the ring floor of 32 768 tokens; below it nothing streams |
| prefix cache | cold blocks indexed and reusable (§7) | exclusive with the prefix cache's host tier |
| a long conversation is | mostly on the card, its middle in host memory | mostly in host memory |

Choose window residency for contexts below the streaming floor, for workloads that reuse long
prefixes turn after turn, and for a card that has the VRAM for the tails. Choose streaming for very
long contexts on a card that cannot hold them at all.

---

## 13. Limits and open points

* **Only the QSA family declares a window.** GLM-5.3's sparse block does not yet.
* **`paged_in`** is rank 0's count, per layer (one block paged across 12 layers counts 12), and not
  per session, because the card cannot attribute a row to a session.
* **A promoted block keeps its origin slot** while it is resident, so the store can be fuller than
  the number of cold entries. A tiered block always does (§8, tiered precision).
* **The emulation equals residency only while the store has room**: a full store leaves blocks
  resident at full precision, which the emulation does not model. No per-block importance signal
  yet chooses between two demote layouts.
* **A demotion is now a dependency of the next step** (the step that reads the block through the
  pager waits for its copy to land): a stream wait, not a host sync, which closes a window where the
  pager could read a slot still being written.
* **Not yet tested with a host test:** the reap at the end of a step under step-ahead issue. The host
  fixtures do not issue ahead, so only the identity run on the card covers it.
* **The device kernel cannot refuse** the way the reference does. An entry whose slot or scratch id is
  out of range is left as it was, so a bad table never writes out of bounds. The reference's refusals
  are what keep such a table from being built.
* **Measurements so far are 2-pass and short-context.** Decode numbers need `--passes 20`, and the
  feature's real case is 100 K+ tokens.

---

## 14. Where the code is

| part | files |
|---|---|
| ABI: cold entries, batch fields, window declaration | `abi/rad_runtime.h`, `abi/rad_builder.h`, `abi/rad_abi.h` (ABI 33) |
| block states, slots, demote / promote / re-point | `core/mem/kv.{h,cpp}` |
| host-mapped store | `core/mem/hostarena.{h,cpp}` (`open_mapped`) |
| prefix cache with cold entries | `core/mem/prefix.{h,cpp}` |
| transfers | `core/runtime/tierexec.{h,cpp}` (`issue_moves`) |
| sweep, reap, dense-step promotion | `core/sched/scheduler.cpp` (`res_sweep`, `res_reap`) |
| engine wiring, store, mover, counter | `core/engine.cpp`, `core/engine_bringup.cpp` |
| the window declaration and the pager issue | `arch/common/rad_qsa.h` |
| the pager | `libref/ref_qsa.cpp` (reference), `libr4d/r4d_qsa_page_u8.hip` (card), `libavx/avx_qsa.cpp` |
| the demote layout's codec (`kv_transcode`, the expanding pager) | `libref/ref_attn.cpp` (`kvx_rows`), `libr4d/r4d_kvx.h`, `libavx/avx_qsa.cpp` |
| the selection change | `libr4d/r4d_qsa_select_bf16.hip` |
| spec and operator docs | `spec.md` §7.3 "Window residency", `docs/GUIDE.md` §4.8, `docs/OPS.md` `qsa_page`, `docs/qsa-residency-bench.md` |
