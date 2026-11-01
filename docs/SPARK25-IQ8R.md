# Spark X2.6 4B IQ8R: recipes, provenance, and validation

`arch/spark25_w8a8` serves Spark X2.6's six dense projections per layer on the `fwht128` rotated
int8 grid. It requires `quantization_config.quant_method=w8a8` and
`quantization_config.fmt=fwht128`; activations use `had_quant_act_i8g` and weights use `gemm_nt_q`
with `dtype=i8a8h`. The embedding and small per-head attention gate stay BF16; norm gains stay
FP32. Calibration is TP=1 so the Gram matrices retain cross-rank terms.

## Recipe choice

The container format does not choose a recipe or change a launcher default. Select one explicitly
when converting a BF16 source checkpoint:

| Allocation | Recipe | Published container size | Measured distribution parity | Performance note |
|---|---|---:|---|---|
| Quality-preferred tested hybrid | `data/recipes/spark-x26-4b-iq8r-keep12.recipe` | 5.67 GiB (6,092,100,608 bytes) | 99.111% top-1 agreement; top-128-plus-tail KL mean 0.000677, p99 0.00923, p99.9 0.04013, max 0.24712 | About 12% less prefill and 17% less decode throughput than full IQ8R in the reported single-card comparison |
| Full IQ8R, explicit speed/size tradeoff | `data/recipes/spark-x26-4b-iq8r.recipe` | 4.52 GiB (4,853,470,208 bytes) | 98.595% top-1 agreement; top-128-plus-tail KL mean 0.004077, p99 0.04960, p99.9 0.34814, max 11.8832; 13 positions exceed 1 nat | Prefill 11.8k / 11.8k / 11.3k tok/s at 1.1k / 4.4k / 9.9k-token prompts; decode 93.7 tok/s at short context and 88.9 tok/s at 4.4k context |
| Source BF16 | no recipe | 8.29 GiB | reference | Baseline |

For the keep12 allocation, six explicit rules cover each projection in layers 12–35. Layers 0–11
have no matching rule and retain their BF16 checkpoint encoding. The output head remains RTN in both
IQ8R recipes; unmatched tensors (including the embedding and attention gate) retain their source
encoding. The full recipe is not quality-preferred and is not applied implicitly. Neither measured
allocation establishes task accuracy or universal quality for all prompts and uses.

The reported KLD values above are **not exact full-vocabulary KL**. They use Radiance's sparse
`--kld-top 128` form: the reference's top 128 token probabilities plus one bucket for all remaining
probability mass, compared with the candidate on the same coarsened support. The 31,391 positions
are from a 32-document held-out evaluation; the keep12 maximum is below one nat on this set, not a
claim about all inputs. Perplexity ratios were 1.00178 (keep12) and 1.00409 (full IQ8R).

## Calibration and evaluation provenance

The published calibration dataset is
[`gostar77/spark-x2.6-4b-iq8r-calibration`](https://huggingface.co/datasets/gostar77/spark-x2.6-4b-iq8r-calibration),
revision `4b622fa2ecca376a02a83ef23ec6dfb3b7cdc52b`. Its `CALIBRATION-MANIFEST.json` records
Spark X2.6 4B, `RADGRAM2`, TP=1, all 36 layers and six projection types. It contains 216 validated
Gram files (21,290,294,016 bytes; about 19.83 GiB) accumulated over 65,660 tokens. The prompts
were 100 selected document beginnings from `swift15-public-corpus`, with selection seed `20261031`
and every fourth candidate wrapped in the Spark chat format. The published dataset is CC BY-SA 4.0;
its `SHA256SUMS` covers the Gram files. The local manifest used for this record is
`/mnt/gostar_data/radiance-builds/hf-publish/spark-x2.6-4b-iq8r-calibration/CALIBRATION-MANIFEST.json`
(SHA-256 `f2225de42cfc885dca2720128143487296a5ba8c33cc797de0bac2b2bcebf007`); its local
`SHA256SUMS` has SHA-256 `f8d14fc575875e33045c2231013e87da52da8db4e21f692c363464d858dbbb18`.

The evaluation manifest is
`/mnt/gostar_data/radiance-builds/spark-x2-6-iq8r/independent-chat-eval.manifest.json`
(SHA-256 `a8528f193986b7267b7ec5b5a447c0d1f5271d5c2f7f004ceb8ba543e5168f4d`). It identifies a
separate held-out source file, `/mnt/gostar_data/radiance-calib/qwen3.8-flash-next-base-w5dn64/corpus/heldout.jsonl`
(source SHA-256 `2c3935512dbf0b09c8fea88be19f7d31447a540de57133a54f97e6e172f15532`), selection seed
`7132026`, and 8 examples each from tools, math, prose, and code. Two high-overlap candidates were
excluded; the manifest describes a sampled 20-gram overlap check against the 100 diverse calibration
documents and the original nine. That check is not proof of zero semantic overlap. Math/tool rows
score assistant continuations; prose/code rows score text after 32 tokens. The resulting local KLD
corpus is `/mnt/gostar_data/radiance-builds/spark-x2-6-iq8r/independent-chat-eval.jsonl`
(SHA-256 `468d9c6037a17475688ba432d804618c10d15288b4d5fe538b7ac4c0ab725ace`).

The recorded matched-reference artifacts are local and were not copied into this repository:

- BF16 sparse reference: `/mnt/gostar_data/radiance-builds/spark-x2-6-iq8r/rowfix-bf16-ref128/`
  (`kld.json` SHA-256 `b722b8a956b747dd932413204d9ae0775bd2bd50377f0466134cc922ad83b026`).
- Full IQ8R report and per-row file: `rowfix-diverse-kld.json` and
  `rowfix-diverse-kld.json.rows` (report SHA-256
  `0f7368c83a3c9a896aa42bcbdf8022db526517f7fc4a9e680747ccd2b66bdee9`).
- Keep12 report and per-row file: `rowfix-keep12-kld.json` and
  `rowfix-keep12-kld.json.rows` (report SHA-256
  `d6b577c1e6f7fc5c40814744bea22079998562229eaa2b2a892274544a149dbf`).
- `rowfix-keep12-kld.json` records `ref_form: "top-128 + tail"`, reference `rowfix-bf16-ref128`,
  `all.positions: 31391`, mean KL `0.0006770557376511436`, maximum KL `0.2471124678850174`, and
  99.1112% top-1 agreement. These are coarsened top-128-plus-tail KL values, not exact
  full-vocabulary KL. `rowfix-diverse-kld.json` uses the same `rowfix-bf16-ref128` reference and
  31,391 positions. The reference metadata records corpus FNV-1a `17d9069a2711c745`; the companion
  `.json.rows` files contain the per-position score rows (KL, candidate NLL, reference NLL, top-1
  match).

The measured rowfix comparison used Radiance source commit `54f2602367eb96ca08674095f7e32506fd7d752b`
(the build tree points to `/mnt/gostar_data/radiance-worktrees/spark-x2-6`), `libr4d@0.5.0`, TP=1,
BF16 KV, `--max-model-len 4096`, `--max-num-seqs 1`, `--max-num-batched-tokens 128`, no speculation,
and prefix cache disabled. The prerequisite shared prefill correction is #362,
`7480da3d46b0475d9d77966e31284cc19072759`. **Those parity figures are for the tree with #362**;
#363's original `5024aea` ancestry did not contain it. Do not attribute these numbers to standalone
#363 or to a build without #362. A fresh GPU TP2 validation is separate and remains outstanding.

Published container checksums from `/mnt/gostar_data/radiance-builds/hf-publish/SHA256SUMS`
(SHA-256 `3c14ea79cb8b7b3aba8807dd43101aa84110636e914eb2008eb2ba3a387e3d1b`):

| Container | SHA-256 |
|---|---|
| `spark-x2.6-4b-radiance-bf16.rad` | `39694f258a3bb1e8367690a5d26762061226f3166125e929c6e33a4981583b6c` |
| `spark-x2.6-4b-radiance-iq8r-diverse.rad` | `2f6982f12b43d1e7621326c5c05a3764ef62defbbd4f59c708f9db698a14b0e5` |
| `spark-x2.6-4b-radiance-iq8r-keep12.rad` | `5a12f1f4c6cb1b2a545344bcb800cd829dc53de6cefa7a6bd9d4e2ed9fee2740` |

## Sharding alignment

A shard must own whole 128-column scale groups for a column-sharded IQ8R input; the architecture
rejects any computed span whose count is not group-aligned. It splits the FFN in whole 128-channel
units when the model width permits, using a ragged remainder-first partition. For Spark's
10,240-channel FFN there are 80 such units: TP=3 gets 27, 27, and 26 groups (3,456, 3,456, and
3,328 channels), all aligned. TP=3 is **not** categorically unaligned or unsupported. The synthetic
wide fixture checks TP=1..8; actual device support still depends on the selected kernel library and
required collectives. Calibration itself remains TP=1.

## Reproduce matched KLD and tail replays

Use a private output directory; the reference is immutable once recorded. The harness hashes the
checkpoint, executable, corpus, evaluation manifest, and **kernel, architecture and optional quantizer
shared-object inventories** into `run.json`. This includes the architecture graph plugin, not only
`libr4d.so`; an optional source revision is context, not proof that a binary was built from it. The
inventory must remain unchanged during the measurement. These probes accept one explicit home, not
a colon-separated home search path. They replay the same corpus against BF16 once and score each
candidate with fixed chunking/cache/TP flags. A reference `provenance.json` binds the BF16 model bytes,
corpus, runtime/plugins, settings and recorded data files. `--reuse-reference` requires this binding
to match; legacy unbound references must be recorded afresh, not relabelled with the supplied model.
The harness snapshots inputs before recording, checks them immediately afterward before publishing
provenance, and checks them again before publishing the run receipt. These checks detect persistent
changes, not a concurrent change that is reverted between checks; keep inputs immutable throughout.
A retry removes any previous `run.json` before measurement, so retain old receipts separately.
Reports and per-position score files are hashed into the run receipt as well:

```sh
python3 tests/spark25_quality.py \\
  --radiance "$BUILD/bin/radiance" --home "$BUILD/radiance_home" \\
  --reference "$BF16" \\
  --corpus /mnt/gostar_data/radiance-builds/spark-x2-6-iq8r/independent-chat-eval.jsonl \\
  --provenance /mnt/gostar_data/radiance-builds/spark-x2-6-iq8r/independent-chat-eval.manifest.json \\
  --calibration-manifest /mnt/gostar_data/radiance-builds/hf-publish/spark-x2.6-4b-iq8r-calibration/CALIBRATION-MANIFEST.json \\
  --source-root "$PWD" --out "$RUN" \\
  --candidate "full=$FULL_IQ8R" --candidate "keep12=$KEEP12"
```

Run that process inside the host's exclusive GPU/device scope for `libr4d`. To replay a candidate's
highest-KL rows, run the following **inside the same kind of device scope**. The tool validates the
report's reference/corpus/model identities, report/row hashes, reference binding, and current runtime,
model and plugin hashes against `run.json`, then starts its own private candidate server with the
scoring run's TP, BF16 KV, model/chunk limits, speculation0 and prefix-cache-off settings. It verifies
the private served ID and the consumed prefix/completion/cache counts. Arbitrary external `--url`
servers are not accepted: a served alias alone cannot prove which checkpoint was loaded.

```sh
python3 tests/spark25_kld_tail.py \\
  --run "$RUN/run.json" \\
  --reference "$RUN/bf16-top128-ref" --report "$RUN/full-kld.json" \\
  --out "$RUN/full-tail-replay.json"
```

The replay compares candidate top-20 log-probabilities with the stored reference top-20 for the
selected rows. The full corpus KLD report remains the top-128-plus-tail metric; top-20 replay is a
focused diagnostic, not a replacement for the full evaluation.

## Reproducible serving QoS smoke

Run this separately for each model, one process/container at a time and under an exclusive device
scope. It reuses the same evaluation prompts, disables prefix caching and speculation, and stores
per-request latency plus p50/p95/max and aggregate generated-token throughput at each concurrency.
These are end-to-end request measurements (prefill plus decode and service overhead), not decode-only
throughput or task scores. Every response must report positive integer prompt usage and exactly
`--max-tokens` generated tokens with a `length` finish (EOS is ignored for this fixed-work smoke).
Missing, zero or short counts fail the run rather than producing a successful throughput receipt.
Runtime/model/plugin hashes are checked again before writing the receipt; use a fresh `--out`:

```sh
python3 tests/spark25_qos.py \\
  --radiance "$BUILD/bin/radiance" --home "$BUILD/radiance_home" \\
  --model "$KEEP12" \\
  --prompts /mnt/gostar_data/radiance-builds/spark-x2-6-iq8r/independent-chat-eval.jsonl \\
  --source-root "$PWD" --out "$RUN/keep12-qos" \\
  --concurrency 1,4,8 --requests 16 --repeats 3 --max-tokens 32
```

Use identical flags, prompt file and `RADIANCE_HOME` for BF16 and full IQ8R; run them sequentially
on the same device allocation. This smoke is a reproducible validation command, not a completed
concurrent/long-context QoS certification. The held-out distribution comparison is not an external
benchmark and does not establish universal model quality.
