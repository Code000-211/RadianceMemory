# Window residency: how to measure it

The feature is live (spec §7.3, "Window residency"): the sweep demotes, the pager pages. This file
is what a run has to do to turn that into a number, and what the numbers mean. Nothing here has been
run end to end on a card; the kernel's count was checked against the reference by `r4d_selftest`
and the engine's seam on the host backend.

## What is built

- `RadKVGroupDecl::residency_window`, set by `arch/common/rad_qsa.h` on `kv_attn` with the indexer's
  budget and never on `kv_qsa_bkey` (`arch_test` asserts both).
- `KVManager`'s cold table entries (`-2 - slot`), the pinned store (`--kv-residency-mib`), demote
  that returns the pool block once its copy has landed (`kv_residency.freed_bytes`), promote into a
  fresh pool block, and the skips a sweep must make.
- `TierExec::issue_moves`: the bytes, both directions, on the transfer stream, with a promotion made
  a dependency of the compute stream. `mem_test` checks it byte for byte after the pool is
  overwritten.
- `/stats` `kv_residency{demoted,promoted,skipped,misses,host_bytes,cap_bytes,freed_bytes,link_bytes,scratch_bytes,paged_in}`;
  `link_bytes` is demoted plus promoted bytes, `scratch_bytes` the pager's reserved VRAM, `paged_in`
  the cold entries `qsa_page` rewrote (summed over QSA layers and decode rows, rank 0's card
  counter, read back per poll; zero means no step ever read a paged block). `/sessions`
  carries `kv_residency{demoted,promoted,skipped,link_bytes,freed_bytes}` per session. The per-session
  numbers overlap where sessions share blocks (a move counts for the sequence that held the block,
  not every session whose prefix names it), so they do not sum to the server's.

## What is not built

Nothing on the engine side. The scheduler's sweep demotes after every step, a prefill row promotes
first, the engine's mover runs the batches on every rank's transfer stream, `landed` is one event
poll, and `qsa_page` pages on the card and counts what it paged. What does not exist yet is a
number from a card: the identity run and the BetterBench comparison below are the next step and have
not been run. The sweep bound (`TierExec::sweep_blocks`: 32 MiB a rank a step, 1 024 blocks on
Flash-Next; it was a fixed 64 blocks) is arithmetic, not a measurement; read `kv_residency.demoted`
against the steps taken if a long prefill's backlog drains too slowly, and the step time if the
sweep's host calls show.

## The smoke that works without a card

Any model, no card, proves the flags parse, the store opens and the report says what happened:

```sh
radiance --model tiny_chat --host 127.0.0.1 --port 8123 --kv-residency-mib 64 \
         --kv-residency-sink-tokens 512 --max-model-len 8192
```

A model that declares no window prints `kv residency: --kv-residency-mib 64 does nothing here: no KV
group declares a residency window` and installs nothing (`engine_e2e_test` holds this: same tokens
as without the flag, `/stats` `kv_residency` all zero). A model that does (Flash-Next) prints the
store line and then `kv residency: window enforced -- ...`.

## The run that means something

```sh
# server, one card is enough; --kv-cache-dtype fp8 halves what a block costs and doubles what the
# window saves, so run both
radiance --model flashnext.rad --max-model-len 200000 --max-num-seqs 2 \
         --kv-cache-dtype fp8 --kv-residency-mib 2048 --kv-residency-sink-tokens 512

# the benchmark, from the BetterBench checkout, against that server
python -m betterbench.cli --base-url http://127.0.0.1:8123/v1 --model <served-name> \
  --config /path/to/docs/qsa-residency-bench.json \
  --note residency=on --note kv_dtype=fp8 --out prep/state/betterbench/qsa-res-on.json
```

Run it twice, once with `--kv-residency-mib 0`, and diff. `betterbench` measures with a fresh nonce
every pass (`unique_nonce: true`), which is right for a stack comparison and wrong for a residency
measurement: a cache-cold pass never re-reads the history the window exists for. For the memory
question use `unique_nonce: false` in the config and a prompt past the QSA bound (2051 tokens), and
read `--prefill` depths of 4000 and 16000 rather than the decode categories.

## What to read

| number | where | what it says |
|---|---|---|
| `kv_residency.demoted` / `host_bytes` | `/stats` | VRAM the window gave back |
| `kv_residency.promoted` / `misses` | `/stats` | blocks a dense step promoted whole: the link traffic that pays for it |
| `kv_residency.paged_hits` | `/stats` | of `paged_in`, the entries the page cache served without a copy; `paged_hits / paged_in` is its hit rate (0 with `RADIANCE_QSA_PAGE_CACHE=0`) |
| `kv_residency.paged_in` | `/stats` | cold entries the pager rewrote for decode selections (divide by QSA layers for blocks); must be non-zero in the identity run |
| TTFT and decode tok/s, residency on vs off | BetterBench | whether the traffic is cheaper than the VRAM it bought |
| `kv_util` | `/stats` | with the window on, an admitted session should fit at a higher `--max-num-seqs` |

The premise from the spec: at 200K tokens a sequence holds 50K attention blocks of 26 KiB. Keeping
2048 of them (512-token sink plus a 1536-token tail) is 51 MiB against 4.80 GiB, so four times the
sessions fit. If `misses` per step is small the trade is a clear win; if the selection churns, the
promotion traffic will show up in TTFT long before it shows up in decode.

## The check that must pass before any of it ships

`scripts/tierident.sh` is the model for it: one prompt with the window fully resident, one with it
forced small so blocks were fetched mid-decode, and byte-identical output. A run whose blocks never
left VRAM (`demoted` zero) or were never paged back (`paged_in` zero) is a failure, not a pass. `mem_test`'s
`a_demoted_block_comes_back_byte_for_byte_after_the_pool_is_overwritten` is the block-level half of
that check and already runs on the host backend.
