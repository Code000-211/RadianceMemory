/* config.cpp -- the command line. Every pool is budgeted explicitly, with no profiling pass,
 * utilisation fraction or workload inference. The KV-versus-weights trade is workload-dependent
 * and the operator's to make; re-optimising it quietly would make benchmarks irreproducible
 * (spec §6).
 */
#include "rad_core.h"
#include "rad_plugin.h"
#include "kld.h"
#include "format/rope_override.h"
#include "sample/host_ref.h"
#include "server/oai.h"
#include "server/http.h"
#include "diag/diag.h"
#include "util/kernel_list.h"
#include "place/planner.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <sys/stat.h>

namespace rad {

struct Flag {
    const char* name;
    const char* arg;     /* null for a boolean */
    const char* help;
};

/* Built from kDraftWindowDefault so the default is stated once. Defined before the table that
 * points at it, which fixes their initialisation order. */
static const std::string g_draft_window_help =
    std::string("fixed | adaptive: how many drafts each sequence carries. fixed drafts\n"
                "                                    --num-speculative-tokens every step; adaptive makes that\n"
                "                                    the ceiling and follows each sequence's acceptance and the\n"
                "                                    measured step cost (a draft head, or prompt lookup alone). Default ") +
    draft_window_name(kDraftWindowDefault);

static const Flag g_flags[] = {
    { "--model",              "PATH",   "the .rad container, a checkpoint served as it stands, or a\n"
                                        "Hugging Face repo id (owner/name), fetched before startup" },
    { "--download-dir",       "DIR",    "where a repo id is fetched to; default\n"
                                        "$XDG_CACHE_HOME/radiance/hf, else ~/.cache/radiance/hf" },
    { "--model-revision",     "REV",    "the hub revision -- a branch, tag, full commit sha or refs/pr/N --\n"
                                        "a repo id is fetched at; default the repo's moving main branch" },
    { "--radiance-home",      "DIRS",   "where kernels/, architectures/ and quantizers/ live; several\n"
                                        "homes separated by ':' are searched in order, the first winning" },
    { "--kernels",            "LIST",   "the kernel libraries to use, separated by ',' or ':', first wins; unnamed ones are not loaded. Default: every library found" },

    { "--tp",                 "N",      "tensor-parallel ranks" },
    { "--tp-wire",            "MODE",   "cross-rank all-reduce payload: exact (default) or wht6, a\n"
                                        "lossy wire: rotated 6-bit at 2 ranks, tiered int8 at 3\n"
                                        "and 4, none at 8" },
    { "--tp-wire-min-kb",     "N",      "smallest all-reduce, in KiB, the lossy wire is used for" },
    { "--tp-attn-split",      "MODE",   "heads (default) | rows -- how a prefill-sized step's\n"
                                        "attention splits over the ranks: rows gives each rank all\n"
                                        "heads of its rows, a faster prefill for the projections at\n"
                                        "full width on every card; not bit-identical to heads" },
    { "--weight-prefetch",    "MODE",   "on (default) | off -- read a GEMM's weight into the cache\n"
                                        "beside the collective before it; off for an engine that\n"
                                        "shares its cards with another server" },
    { "--p2p",                "MODE",   "how the collectives reach the other cards: auto (default)\n"
                                        "asks the driver, on requires peer access and stops without\n"
                                        "it, off exchanges through pinned host memory" },
    { "--max-num-batched-tokens", "N",  "the largest token count one step may carry" },
    { "--busy-prefill-chunk", "MODE",   "a prefill chunk's share of a step that has other work: auto\n"
                                        "(default) leaves decode rows and new requests their room,\n"
                                        "off takes the whole step, N holds the chunk to N tokens" },
    { "--prefill-interleave", "N",      "decode-only steps after each prefill chunk while other\n"
                                        "sequences decode, at most 64; 0 (default) lets every step\n"
                                        "carry a chunk" },
    { "--preempt-wait-ms",    "N",      "how long a preempted prefill may wait for the whole-\n"
                                        "recompute gate before reclaiming cache blocks and\n"
                                        "preempting the lowest-priority running request; 0\n"
                                        "(default 5000) falls back on the next step" },
    { "--max-num-seqs",       "N",      "the largest number of sequences one step may carry" },
    { "--max-model-len",      "N",      "context bound; 0 takes the model's training context" },
    { "--rope-scaling",       "SPEC",   "yarn:FACTOR[:ORIG]: static YaRN over the container's own rotary\n"
                                        "keys, no reconvert (docs/YARN.md); none: off" },

    { "--gpu-headroom-mib",   "N",      "VRAM left unclaimed on every card (default 96)" },
    { "--kv-cache-dtype",     "T",      "bf16 | fp8 | fp8_v4 | k6_v4 -- fp8 halves what a token of\n"
                                        "context costs; fp8_v4 keeps V in 4 bits (0.78 of fp8), k6_v4\n"
                                        "also K in 6 (0.69); both Qwen3.8-27B and -Flash-Next" },
    { "--kv-recipe",          "FILE",   "each attention layer's layout from that set, k4h_v4\n"
                                        "(rotated int4 K and V, 0.56), k4h_v2h (int2 V, 0.44) or i8h\n"
                                        "(int8, a scale a token and head, 1.01; all rotated,\n"
                                        "Flash-Next) or bf16 (2; Flash-Next), a 'layer N\n"
                                        "LAYOUT [rotated] [demote LAYOUT] [vmean]' line a layer\n"
                                        "(docs/GUIDE.md); needs\n"
                                        "--kv-cache-dtype fp8" },
    { "--state-cache-dtype",  "T",      "f32 (default) | f16 -- the recurrent state cache (delta net):\n"
                                        "f16 halves the state pool, its checkpoints and its traffic,\n"
                                        "and is lossy; the qwen35 plugins serve it" },
    { "--qsa-bkey-dtype",     "T",      "bf16 (default) | i8 -- the QSA indexer's block keys: i8\n"
                                        "halves them (int8, an f16 scale a block; Flash-Next)\n"
                                        "and moves the indexer's selection slightly" },
    { "--qsa-bkey-split",     "MODE",   "off (default) | on -- the QSA indexer's block keys split\n"
                                        "across the --tp ranks (block b on rank b % tp), the\n"
                                        "ranks' top-k lists merged: the same selection byte for\n"
                                        "byte, block keys per card / tp (Flash-Next)" },
    { "--attn-prefill-fp8",   "MODE",   "off (default) | qk | pv | both -- the prefill attention's\n"
                                        "Q.K^T, P.V or both as e4m3 WMMA over an fp8, fp8_v4 or\n"
                                        "k6_v4 cache: faster at long context, and lossy" },
    { "--kv-resident",        "N",      "stream the sparse attention group's KV from host memory,\n"
    "                                    keeping N tokens of it in a VRAM ring (0, the default, is\n"
    "                                    off). Raised to 32768 and to what one decode step selects;\n"
    "                                    a ring that covers the context (sequences x\n"
    "                                    --max-model-len) streams nothing" },
    { "--host-pool-mib",      "N",      "pinned host tier" },
    { "--host-headroom-mib",  "N",      "host RAM to leave free for the system when radiance\n"
    "                                    pins its host memory at startup (default 4096)" },
    { "--embedding-placement", "WHERE", "ram (default) | vram: where the token embedding lives. ram\n"
    "                                    pins it in host memory beside --host-pool-mib, read over the\n"
    "                                    link in place; by default a model whose output head reads the\n"
    "                                    table, or a host without the memory, keeps it on the card" },
    { "--vision-tower-placement", "WHERE", "vram (default) | ram: where the encoder towers live (the\n"
    "                                    vision tower; the audio encoder too). ram pins them in host\n"
    "                                    memory beside --host-pool-mib, read over the link once an\n"
    "                                    image, and gives their VRAM to the KV cache" },
    { "--ngram-placement",    "WHERE",  "auto (default) | vram | ram | disk: where an n-gram\n"
    "                                    embedding table lives (Qwen3.8-Flash-Next: 47.7 GiB).\n"
    "                                    vram puts all of it on every card; ram copies it into\n"
    "                                    host memory at load, pinned for the cards to read\n"
    "                                    (locked, where only a host kernel reads it); disk reads\n"
    "                                    its rows from the container on the host. auto takes the\n"
    "                                    first of the three that holds it. ram=DIR makes the\n"
    "                                    pinned copy a file in DIR (a tmpfs) mapped shared, so\n"
    "                                    two engines on separate cards hold one table: the\n"
    "                                    first fills it, the last to exit deletes it" },
    { "--weights-disk-tier",  nullptr,  "read weights that fit neither VRAM nor the host pool from the\n"
    "                                    container on disk (O_DIRECT) -- the file tier, units on\n"
    "                                    Tier::SSD; a routed expert is read a layer at\n"
    "                                    a time, so a step is paced by the drive" },
    { "--prefix-cache-mib",   "N",      "copy finished conversations' KV to host memory, up to N MiB,\n"
    "                                    so their VRAM can go to experts and a return restores\n"
    "                                    instead of re-prefilling. A full tier drops its least\n"
    "                                    recently used entries. 0 is off. Off while a KV group\n"
    "                                    streams (--kv-resident)" },
    { "--prefix-cache-ttl",   "S",      "seconds a cached conversation no request uses is kept, in VRAM\n"
    "                                    or host memory (default 3600). 0 keeps it until the memory\n"
    "                                    is wanted" },
    { "--kv-residency-mib",   "N",      "host memory, up to N MiB, for the KV blocks a sparse\n"
    "                                    consumer's window does not cover (spec §7.3). A model\n"
    "                                    whose attention reads a selected part of its history\n"
    "                                    declares that window, and the rest of its cache then costs\n"
    "                                    link traffic instead of VRAM. 0 is off: every block stays"
    "                                    resident" },
    { "--kv-residency-sink-tokens", "N", "tokens at the start of a conversation kept resident\n"
    "                                    when --kv-residency-mib is on (default 512): a sparse\n"
    "                                    consumer scores them every step, so they are cheaper in\n"
    "                                    VRAM than in the store" },

    { "--lora",               "ADAPTER", "NAME=PATH: serve a LoRA adapter beside the base model\n"
    "                                    (repeatable). A text model takes a PEFT directory and\n"
    "                                    serves it as model NAME; an image or video generator takes\n"
    "                                    a .safetensors file (kohya, ComfyUI, PEFT, LyCORIS) that\n"
    "                                    a request names in `loras`. The base stays as it is"},
    { "--lora-dir",           "DIR",    "image and video generators: a request may name any\n"
    "                                    NAME.safetensors under DIR in `loras`, opened when first\n"
    "                                    asked for"},
    { "--max-loras",          "N",      "adapter slots resident at once (default: the number of\n"
    "                                    --lora, 2 with --lora-dir alone). A text model's slot holds\n"
    "                                    one adapter, a generator's one request's combination"},
    { "--max-lora-rank",      "R",      "the rank each slot is sized for (default 16 for a text\n"
    "                                    model, 64 for a generator); a text model's larger adapter is\n"
    "                                    refused, a generator's larger combination truncated"},
    { "--lora-target-modules", "LIST",  "comma-separated HF module leaf names adapters may\n"
    "                                    target (q_proj,k_proj,...). Default: what the --lora\n"
    "                                    adapters target. Needed with --max-loras and no --lora"},

    { "--placement",          "MODE",   "all_vram | layer_offload | expert_tiered | auto" },
    { "--control-vector",     "MODE",   "off | on | disabled: load the control-vector file found\n"
    "                                    for this model and start requests off (off) or on (on).\n"
    "                                    disabled -- the default, also with no flag at all -- loads\n"
    "                                    nothing: no projection op is declared, and a request that\n"
    "                                    asks for the vector is refused" },
    { "--deterministic",      nullptr,  "pin placement to the plan the loader made, so the resident\n"
    "                                    set cannot change under the heat engine. The sampler is\n"
    "                                    already order-independent and needs no pinning; note that\n"
    "                                    this does NOT make the engine bit-reproducible across\n"
    "                                    batch positions -- see spec §19" },
    { "--checkpoint-interval","N",      "linear-attention state checkpoint interval, in tokens" },
    { "--prefill-span",       "N",      "checkpoint intervals one step may carry of a prefill alone\n"
    "                                    in its step, from the prompt's second interval on (default\n"
    "                                    1); needs --max-num-batched-tokens of N intervals. Every\n"
    "                                    other step is planned at the budget less the extra\n"
    "                                    intervals, and the checkpoints inside a span are not\n"
    "                                    written" },
    /* A total, not per sequence: the snapshot store is one pool of N slots
     * (KVManager::set_checkpoint_store). */
    { "--checkpoint-slots",   "N",      "linear-state or sliding-window snapshots the server holds in\n"
    "                                    total, in pinned host memory. Default `auto`: what\n"
    "                                    --checkpoint-policy keeps for a --max-model-len session, plus\n"
    "                                    one, for each of --max-num-seqs, within an eighth of host\n"
    "                                    memory. N fixes the total; 0 keeps none" },
    { "--no-prefix-cache",    nullptr,  "disable prefix caching" },
    { "--prefix-resume",      "MODE",   "where a prefix hit resumes: exact (default) only where\n"
    "                                    an idle server's cold prefill of the prompt starts a\n"
    "                                    chunk, so the hit answers bit-identically to a cold run;\n"
    "                                    nearest at the deepest snapshot, up to a chunk less to\n"
    "                                    re-prefill" },
    { "--checkpoint-policy",  "NAME",   "which snapshots a session retains: geometric-backoff "
                                        "(default), tip-only, keep-all" },

    { "--num-speculative-tokens", "N",  "speculative window; 0 disables. Default `auto`: the drafter's\n"
    "                                    own operating point, which the plugin states. A container\n"
    "                                    with no draft head is unaffected -- auto resolves to 0." },
    { "--spec-ngram",         "K",      "speculate by prompt lookup alone: up to K drafts a step\n"
    "                                    copied from what followed the latest earlier occurrence of\n"
    "                                    the sequence's last n tokens (no weights, no draft pass).\n"
    "                                    Sets the window to K; a draft head the container carries is\n"
    "                                    not loaded. A proposal is verified only when its record\n"
    "                                    says it pays at the measured step cost. 0 (default) off" },
    { "--spec-ngram-min",     "N",      "shortest pattern prompt lookup matches (default 2)" },
    { "--spec-ngram-max",     "N",      "longest pattern prompt lookup matches, tried first\n"
                                        "(default 4, at most 15)" },
    { "--draft-window",       "MODE",   g_draft_window_help.c_str() },
    { "--draft-short-from",   "N",      "a block drafter that also serves a shorter block runs it\n"
                                        "from N decoding sequences on (0 never): the long block pays\n"
                                        "at one sequence, the short one at many. auto (the default):\n"
                                        "never for a chain, from 2 for a tree drafter" },
    { "--draft-coupling",     "MODE",   "off (default) | on -- a linear MTP drafter or DFlash2's walk\n"
    "                                    samples each proposal from its own shortlist and the sampler\n"
    "                                    couples the target's draw to it: more accepted drafts on sampled\n"
    "                                    requests, the same output law, other tokens than off for a fixed\n"
    "                                    seed" },
    { "--draft-model",        "PATH",   "NOT IMPLEMENTED -- merge the drafter at convert time" },

    { "--host",               "ADDR",   "listen address" },
    { "--port",               "N",      "listen port" },
    { "--api-key",            "KEY",    "require `Authorization: Bearer KEY` or `X-Api-Key: KEY` on every request but /health,\n"
    "                                    /v1/health, /ping and the dashboard page" },
    { "--api-key-env",        "NAME",   "read the API key from the named environment variable; keeps it off the\n"
    "                                    command line, where every process on the machine can read it" },
    { "--allow-host",         "NAME",   "a Host name the keyless loopback-only routes (/diagnostics,\n"
    "                                    /admin/control-vector, the LoRA and session admin) accept besides\n"
    "                                    localhost; repeatable" },
    { "--trusted-proxy",      "CIDR",   "a network whose sockets may name the client behind them via\n"
    "                                    X-Forwarded-For, for the log; vouching changes attribution,\n"
    "                                    never access; repeatable" },
    { "--served-model-name",  "NAME",   "the model id /v1/models lists and responses carry; default the container's name" },
    { "--mm-max-patches",     "N",      "patches one encoder pass carries, and so the largest image (a patch is 16x16\n"
                                        "pixels). Default `auto`: 16384 when the container carries a vision tower. 0\n"
                                        "serves text only and keeps the tower off the card" },
    { "--image-gen-max-pixels", "N",    "an image generator's largest image, in pixels (1048576: 1024 x 1024).\n"
      "                              Default: the model's. Its per-image buffers, and the activation\n"
      "                              arena with them, are sized for it; larger requests are refused" },
    { "--image-gen-max-refs", "N",      "the reference pictures an image generator takes a request. Default: the\n"
      "                              model's; its conditioning buffers are sized for them" },
    { "--generation-config",  "PATH",   "a generation_config.json whose sampler settings become the default\n"
    "                                    for requests that send none. Without it a <model>.generation.json\n"
    "                                    beside the container is used, then the container's own\n"
    "                                    `generation.*` metadata, then the built-in defaults" },
    { "--override-chat-template", "PATH", "a Jinja chat template file, used in place of the one the container carries" },
    { "--temp",               "F",      "default temperature for requests that do not set one" },
    { "--top-k",              "N",      "default top_k for requests that do not set one; 0 is the 1024 most\n"
    "                                    likely tokens, the sampler's candidate width" },
    { "--top-p",              "F",      "default top_p for requests that do not set one" },
    { "--min-p",              "F",      "default min_p (0 = off) for requests that do not set one.\n"
    "                                    These four override whatever the container or its generation\n"
    "                                    config asked for; unset, they are resolved from it" },
    { "--reasoning-effort",   "S",      "default `reasoning_effort` for chat requests that do not set one.\n"
    "                                    The value goes to the model's chat template verbatim, so the legal\n"
    "                                    set is the template's: Qwen3.8 takes xhigh|medium|low and defaults\n"
    "                                    to xhigh, which on a short task can spend the whole token budget\n"
    "                                    inside <think> and answer nothing. \"none\" turns thinking off" },
    { "--reasoning-effort-map", "MAP",  "FROM=TO[,FROM=TO...]: a `reasoning_effort` a request sends, replaced before\n"
    "                                    the chat template sees it -- high=xhigh lets a client that always sends\n"
    "                                    OpenAI's top level reach Qwen3.8's. TO is checked against the template at\n"
    "                                    startup; none and minimal turn thinking off. Repeatable" },
    { "--thinking-token-budget", "N",   "default `thinking_token_budget` for chat requests that do not set one: at\n"
    "                                    N reasoning tokens the reasoning block is closed and the answer follows,\n"
    "                                    so a client's max_tokens is not spent on reasoning alone. Unset is no\n"
    "                                    limit. Needs a reasoning block that ends in one token, as </think> is" },
    { "--typical-p",          "F",      "default typical_p, (0, 1]. This flag and the eleven below set the default\n"
    "                                    for requests that do not name the field; a request that names it wins" },
    { "--presence-penalty",   "F",      "default presence_penalty, [-2, 2]" },
    { "--frequency-penalty",  "F",      "default frequency_penalty, [-2, 2]" },
    { "--repetition-penalty", "F",      "default repetition_penalty, > 0" },
    { "--repeat-last-n",      "N",      "default repeat_last_n, the tokens those penalties look back over; -1 is all" },
    { "--dry-multiplier",     "F",      "default dry_multiplier; 0 is DRY off" },
    { "--dry-base",           "F",      "default dry_base, >= 1" },
    { "--dry-allowed-length", "N",      "default dry_allowed_length, >= 1" },
    { "--dry-penalty-last-n", "N",      "default dry_penalty_last_n; -1 is the whole context" },
    { "--dry-sequence-breakers", "JSON", "default dry_sequence_breakers, a JSON array of at most 16 strings" },
    { "--xtc-probability",    "F",      "default xtc_probability, [0, 1]" },
    { "--xtc-threshold",      "F",      "default xtc_threshold, [0, 1]" },
    { "--default-max-tokens", "N",      "max_tokens for requests that do not set one: N, or `auto` (default),\n"
    "                                    whatever the context leaves after the prompt" },
    { "--max-tokens-cap",     "N",      "the most any request may generate. A larger max_tokens is cut to it rather\n"
    "                                    than refused, and the reply ends with finish_reason \"length\". 0 is none\n"
    "                                    (default)" },
    { "--max-n",              "N",      "the largest `n` a request may ask for (default: the larger of 8 and\n"
    "                                    --max-num-seqs)" },
    { "--max-queued-requests", "N",     "requests admitted and unfinished at once, past which the answer is 429\n"
    "                                    (default 8 x --max-num-seqs)" },
    { "--max-stop-strings",   "N",      "stop strings one request may send (default 64)" },
    { "--max-stop-bytes",     "N",      "bytes in one stop string (default 4096)" },
    { "--chat-template-kwargs", "JSON", "variables handed to the chat template on every chat request: a JSON object,\n"
    "                                    or @FILE to read one. Repeatable, a later key replacing an earlier one.\n"
    "                                    A key the request's own chat_template_kwargs names is the request's, and a\n"
    "                                    request that says anything about thinking replaces enable_thinking and\n"
    "                                    reasoning_effort here" },
    { "--reasoning-format",   "MODE",   "auto (default): a reasoning block comes back as reasoning_content.\n"
    "                                    none: it stays inline in content" },

    { "--http-threads",       "N",      "HTTP workers kept while idle (default: one per core, at least 4)" },
    { "--read-timeout",       "S",      "seconds a client may go silent while sending a request (default 30)" },
    { "--write-timeout",      "S",      "seconds one write to a client may block (default 600)" },
    { "--keep-alive-timeout", "S",      "seconds an idle connection is held open between requests (default 5)" },
    { "--max-body-mib",       "N",      "the largest request body; images and video arrive base64 inside it\n"
    "                                    (default 512)" },
    { "--retry-after",        "S",      "the Retry-After a 429 carries, in seconds (default 1)" },
    { "--no-cors",            nullptr,  "send no CORS headers, so a page on another origin cannot read the API" },

    { "--image-min-pixels",   "N",      "an image is resized to at least N pixels (default: the container's)" },
    { "--image-max-pixels",   "N",      "and to at most N; one encoder pass (--mm-max-patches) still bounds it" },
    { "--video-min-pixels",   "N",      "a video's sampled frames together resize to at least N pixels" },
    { "--video-max-pixels",   "N",      "and to at most N" },
    { "--video-fps",          "F",      "frames sampled per second of video" },
    { "--video-min-frames",   "N",      "the fewest frames sampled from a video" },
    { "--video-max-frames",   "N",      "the most frames sampled from a video" },
    { "--video-max-frame-tokens", "N",  "the most prompt tokens one video frame may become; 0 is no cap (a token\n"
    "                                    budget processor -- Gemma 4's -- takes 70, 140, 280, 560 or 1120)" },
    { "--image-tokens",       "N",      "a token-budget processor's image budget (Gemma 4: 70, 140, 280, 560, 1120,\n"
    "                                    which a request's `image_tokens` overrides; GLM-5.3: the most tokens an\n"
    "                                    image becomes, at most --mm-max-patches / 4; default the container's)" },
    { "--max-source-pixels",  "N",      "the largest image or video frame the decoder accepts, before any resize\n"
    "                                    (default 67108864). The media flags above default to what the container\n"
    "                                    states, and do nothing for a model that takes no media" },

    { "--kld-record",         "DIR",    "instead of serving, run --kld-corpus through prefill and write this\n"
    "                                    model's log-probabilities at every scored position to DIR: the\n"
    "                                    reference a quantised model is measured against" },
    { "--kld-ref",            "DIR",    "instead of serving, score this model against the reference in DIR:\n"
    "                                    mean, median, 99/99.9/99.99th percentile KL divergence and top-1\n"
    "                                    agreement, over the positions and tokens DIR was recorded at" },
    { "--kld-corpus",         "FILE",   "JSONL, one {\"prompt\", \"score_from\"?, \"source\"?} a line; --kld-record only" },
    { "--calib-corpus",       "FILE",   "instead of serving, run a JSONL corpus ({\"prompt\"} a line) through\n"
    "                                    prefill once, every token, and record the calibration statistics\n"
    "                                    the model keeps (each linear's input Gram, GPTQ's Hessian)" },
    { "--calib-dir",          "DIR",    "where --calib-corpus writes the statistics; beside --kld-record, the\n"
    "                                    reference's run records them over the KL corpus too" },
    { "--calib-layers",       "RANGE",  "LO:HI, record layers LO..HI-1 only (default every layer), for a model\n"
    "                                    whose statistics do not fit in memory in one run" },
    { "--kld-top",            "N",      "--kld-record keeps each position's N most likely tokens exactly and the\n"
    "                                    mass of the rest, not the whole vocabulary (a corpus too large for\n"
    "                                    full rows); 0 (default) keeps the whole vocabulary" },
    { "--kld-out",            "FILE",   "the --kld-ref report as JSON, and every position's numbers in FILE.rows" },
    { "--kld-lora",           "NAME",   "run --kld-record / --kld-ref through the --lora adapter NAME" },
    { "--kld-lora-sources",   "PREFIX", "--kld-lora only for documents whose source starts with PREFIX; the\n"
    "                                    rest run as the base, in the same steps (a mixed batch)" },

    { "--debug-graph",        nullptr,  "print the entire declared graph and exit-quality detail" },
    { "--debug-selection",    nullptr,  "print the selection table" },
    { "--debug-placement",    nullptr,  "print the placement plan" },
    { "--debug-draft-oracle", "FILE",   "a verify-path test, not a drafter: with --spec-ngram, prompt\n"
    "                                    lookup proposes from the token ids in FILE (a plain greedy\n"
    "                                    run's continuation of the one prompt the test sends), 30% of\n"
    "                                    the drafts corrupted; the text must stay the plain run's" },
    { "--debug-accept-reference-kernels", nullptr,
      "serve even when ops resolve only to a reference library. NOT a production configuration: the reference\n"
      "                              implementation is an oracle and a last resort, and falling back\n"
      "                              to it is orders of magnitude of speed with nothing to say so." },
    { "--proceed-with-incompatible-configuration", nullptr,
      "start on a machine configuration radiance refuses (amdgpu CWSR off where cards serve).\n"
      "                              UNSUPPORTED: all guarantees are off, and the log and the startup\n"
      "                              report say so." },
    { "--profile-ops",        nullptr,  "per-op device timing (see the caveat it prints)" },
    { "--expert-stats",       "FILE",   "count the routed experts every layer selects, apart for prefill, decode,\n"
    "                                    mixed steps and draft-head passes, and write the counts to FILE as\n"
    "                                    JSON every 30 s and at shutdown. From the routing histograms the core\n"
    "                                    already copies back: no device work, no synchronisation" },
    { "--expert-profile",     "FILE",   "rank the expert plane by this radiance-expert-stats JSON's `decode`\n"
    "                                    counts instead of the container's activation profile, which is\n"
    "                                    prefill-heavy: decode-hot experts differ. The planner keeps the\n"
    "                                    hottest resident; --expert-pin-top makes a chosen slice unmovable" },
    { "--expert-pin-top",     "F",      "pin the hottest fraction F of the routed experts (0 < F <= 1) by\n"
    "                                    --expert-profile's share: heat never demotes them, so the decode-hot\n"
    "                                    set stays resident while the rest streams. 0 is off" },
    { "--draft-expert-placement", "PLACEMENT", "where a draft head's (MTP/DSpark) routed experts live.\n"
    "                                    `host` pins them in the host pool before any strategy runs, read in\n"
    "                                    place like a pooled expert: the drafter's experts fit beside a trunk\n"
    "                                    that already fills the card, instead of the strategy having to make\n"
    "                                    room on the card. `auto` (the default) leaves them to the strategy" },
    { "--live",               nullptr,  "the live view: cards, throughput, draft acceptance, the expert plane.\n"
    "                                    Redirected it prints one greppable key=value line instead" },
    { "-v",                   nullptr,  "verbose" },
    { "-vv",                  nullptr,  "very verbose" },
    { "--help",               nullptr,  "this" },
};

void config_usage(const char* argv0) {
    fprintf(stderr, "usage: %s --model FILE [options]\n\n", argv0);
    /* Align help continuation lines under the first, whatever indentation the string carries. */
    for (const Flag& f : g_flags) {
        char left[64];
        snprintf(left, sizeof left, "%s%s%s", f.name, f.arg ? " " : "", f.arg ? f.arg : "");
        std::string help;
        for (const char* p = f.help; *p; ++p) {
            help += *p;
            if (*p != '\n') continue;
            help.append(35, ' ');
            while (p[1] == ' ') ++p;
        }
        fprintf(stderr, "  %-32s %s\n", left, help.c_str());
    }
    fprintf(stderr,
        "\nEvery pool is budgeted explicitly: the split between KV blocks and resident weights\n"
        "is the operator's to set.\n");
}

/* The four names exactly, lower case; anything else is refused rather than mapped to a tier. */
static bool parse_table_placement(const char* v, TablePlacement* out) {
    if      (!std::strcmp(v, "auto")) *out = TablePlacement::Auto;
    else if (!std::strcmp(v, "vram")) *out = TablePlacement::Vram;
    else if (!std::strcmp(v, "ram"))  *out = TablePlacement::Ram;
    else if (!std::strcmp(v, "disk")) *out = TablePlacement::Disk;
    else return false;
    return true;
}

/* The whole argument must be a number: atof reads "high" as 0, which for --temp silently means
 * greedy decoding. */
static bool parse_num(const char* flag, const char* v, double* out) {
    char* end = nullptr;
    errno = 0;
    const double d = std::strtod(v, &end);
    if (!*v || !end || *end || errno == ERANGE || !std::isfinite(d)) {
        RAD_ERR("%s takes a number, not '%s'", flag, v);
        return false;
    }
    *out = d;
    return true;
}

static bool parse_int(const char* flag, const char* v, int64_t lo, int64_t hi, int64_t* out) {
    char* end = nullptr;
    errno = 0;
    const long long n = std::strtoll(v, &end, 10);
    if (!*v || !end || *end || errno == ERANGE) {
        RAD_ERR("%s takes a whole number, not '%s'", flag, v);
        return false;
    }
    if (n < lo || n > hi) {
        if (lo > INT32_MIN && (hi == INT32_MAX || hi == INT64_MAX))
            RAD_ERR("%s must be >= %lld, not %lld", flag, (long long)lo, n);
        else
            RAD_ERR("%s must be in [%lld, %lld], not %lld", flag, (long long)lo, (long long)hi, n);
        return false;
    }
    *out = n;
    return true;
}

/* Sampler defaults are checked against the request range (server::sampler_value_ok) at startup;
 * an out-of-range default would 400 every request that omits the field. */
static bool sampler_num(const char* flag, const char* field, const char* v, double* out) {
    if (!parse_num(flag, v, out)) return false;
    std::string why;
    if (!server::sampler_value_ok(field, *out, &why)) {
        RAD_ERR("%s %s: %s %s", flag, v, field, why.c_str());
        return false;
    }
    return true;
}
static bool sampler_int(const char* flag, const char* field, const char* v, int* out) {
    int64_t n = 0;
    double d = 0;
    if (!parse_int(flag, v, INT32_MIN, INT32_MAX, &n) || !sampler_num(flag, field, v, &d))
        return false;
    *out = (int)n;
    return true;
}

/* Names the renderer sets itself. Kwargs are written over the template context
 * (common_chat_template_direct_apply_impl), so a kwarg with one of these names would replace every
 * request's messages, tools or special tokens. */
static const char* const kTemplateOwnNames[] = {
    "messages", "tools", "bos_token", "eos_token", "add_generation_prompt",
};

/* --chat-template-kwargs: a JSON object, or @FILE holding one, merged key by key into `out`. */
static bool parse_template_kwargs(const char* v, std::map<std::string, std::string>* out) {
    std::string text = v;
    std::string src = "--chat-template-kwargs";
    if (!text.empty() && text[0] == '@') {
        std::ifstream f(text.substr(1), std::ios::binary);
        if (!f) {
            RAD_ERR("--chat-template-kwargs %s: cannot read %s", v, text.c_str() + 1);
            return false;
        }
        std::ostringstream ss;
        ss << f.rdbuf();
        src += " " + text.substr(1);
        text = ss.str();
    }
    const nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        RAD_ERR("%s takes a JSON object, such as '{\"enable_thinking\": false}'", src.c_str());
        return false;
    }
    for (auto it = j.begin(); it != j.end(); ++it) {
        for (const char* own : kTemplateOwnNames)
            if (it.key() == own) {
                RAD_ERR("%s: '%s' is set by the server for every request and cannot be a template "
                        "variable", src.c_str(), own);
                return false;
            }
        if (it.key() == "enable_thinking" && !it.value().is_boolean()) {
            RAD_ERR("%s: enable_thinking takes true or false", src.c_str());
            return false;
        }
        /* The template context takes JSON text, so strings stay quoted. */
        (*out)[it.key()] = it.value().dump();
    }
    return true;
}

/* ------------------------------------------------------------------ --kv-recipe */
/* The layouts a recipe can name, widest first: the name in the file, the RAD_KVQ_* the plugin
 * declares, and what one group of 32 K elements and 32 V elements costs in bits, scales included
 * (the unit rad-autoquant's bit budget is counted in). A row added here is a recipe word and a
 * calibration candidate at once: rad-autoquant's variants are each row as kv_recipe_parse reads it
 * plain, `rotated` and `vmean` (kv_autoquant_variants), so a new layout's rules (rotation implied,
 * no V mean) belong in the parser below and nowhere else. */
struct KvLayout { const char* name; int kvq; int group_bits; };
static const KvLayout kKvLayouts[] = {
    /* The cache --kv-cache-dtype bf16 keeps, for one layer: nothing rounded but bf16's own. Twice
     * fp8; what a layer too sensitive for any 8-bit layout keeps (rad-autoquant's base). */
    { "bf16",   RAD_KVQ_BF16, 32 * 16 + 32 * 16 },
    /* K and V int8, stored rotated, one f16 scale a token and head for each: at head_dim 256 a
     * group of 32 carries an eighth of a scale, 2 bits, so 258/256 of fp8. Wider than fp8; it buys
     * resolution (7 bits to E4M3's 3), not room. */
    { "i8h",    RAD_KVQ_I8H,  (32 * 8 + 2) + (32 * 8 + 2) },
    /* E4M3 K and V; the descales are per (sequence, head), nothing a group. */
    { "fp8",    RAD_KVQ_NONE, 32 * 8 + 32 * 8 },
    /* V int4 with an f16 scale a group: 25/32 of fp8. */
    { "fp8_v4", RAD_KVQ_V4,   32 * 8 + (32 * 4 + 16) },
    /* K int6 too, its own f16 scale: 22/32. */
    { "k6_v4",  RAD_KVQ_K6V4, (32 * 6 + 16) + (32 * 4 + 16) },
    /* K int4 as well, stored rotated: 18/32. The rotation is part of the layout (below). */
    { "k4h_v4", RAD_KVQ_K4H,  (32 * 4 + 16) + (32 * 4 + 16) },
    /* V int2 under KVarN's variance normalisation, rotated as k4h_v4: 14/32. */
    { "k4h_v2h", RAD_KVQ_K4HV2H, (32 * 4 + 16) + (32 * 2 + 16) },
};

int kv_layout_of(const char* name) {
    if (!name) return -1;
    for (const KvLayout& l : kKvLayouts)
        if (!std::strcmp(name, l.name)) return l.kvq;
    return -1;
}

const char* kv_layout_name(int kvq) {
    for (const KvLayout& l : kKvLayouts)
        if (l.kvq == kvq) return l.name;
    return nullptr;
}

int kv_layout_group_bits(int kvq) {
    for (const KvLayout& l : kKvLayouts)
        if (l.kvq == kvq) return l.group_bits;
    return -1;
}

/* A group of 32 K and 32 V elements' bits over 512 is a row's bytes over head_dim; i8h's scale is
 * a head's, so its row is that and not a group multiple. */
int64_t kv_layout_row_bytes(int kvq, int64_t hd) {
    const int bits = kv_layout_group_bits(kvq);
    if (bits < 0 || hd <= 0 || hd % 32) return -1;
    return kvq == RAD_KVQ_I8H ? 2 * hd + 4 : bits * hd / 256;
}

int64_t kv_layout_head_dim(int kvq, int64_t row) {
    const int bits = kv_layout_group_bits(kvq);
    if (bits < 0 || row <= 0) return -1;
    const int64_t hd = kvq == RAD_KVQ_I8H ? (row - 4) / 2 : row * 256 / bits;
    return kv_layout_row_bytes(kvq, hd) == row ? hd : -1;
}

std::vector<int> kv_layouts() {
    std::vector<int> v;
    for (const KvLayout& l : kKvLayouts) v.push_back(l.kvq);
    return v;
}

/* One `layer N LAYOUT` line per attention layer, `layer N LAYOUT rotated` for a layer that stores
 * its K and V rotated (RAD_KVQ_ROTATED: the normalised Hadamard over head_dim, its query rotated
 * before the attention and the output back after it); `#` starts a comment, blank lines are
 * skipped. k4h_v4, k4h_v2h and i8h are rotated layouts (RAD_KVQ_K4H, RAD_KVQ_K4HV2H,
 * RAD_KVQ_I8H): a line of any of them marks the layer rotated whether or not it ends in `rotated`. Implied rather than demanded, because the name already says it
 * (the h is the Hadamard), a k4h or i8h layer without the rotation is not a deployment anyone can ask
 * for, and rad-autoquant writes every layout as `layer N LAYOUT` from the table above.
 * `bf16` is the plain bf16 cache (RAD_KVQ_BF16) and takes neither word, nor `demote`: it rounds
 * nothing a rotation or a mean would help with, the bf16 rows serve neither, and it keeps its
 * blocks as they are.
 * A line may also end in `vmean` (after `rotated` when both): the layer subtracts a static V mean
 * before its store and adds it to the attention output (RAD_KVQ_VMEAN), taken from the file one
 * `vmean FILE` line names, which `vmean_file` receives as written. Only a layout with integer V
 * takes it: fp8's E4M3 V has no group scale for an offset to waste.
 * `demote LAYOUT`, after `rotated` and before `vmean` (RAD_KVQ_DEMOTE): under window residency the
 * layer's demoted blocks are stored in that second layout (k4h_v4 or k4h_v2h, of an i8h or a
 * rotated fp8 layer) and expanded back on the card when read, so the store holds more blocks a
 * MiB and every paged block moves fewer bytes; `vmean` then needs integer V in either layout.
 * Layer numbers are the checkpoint's (blk.N), so a recipe reads against the model's own config.
 * Strict: a word the format does not have, a layer named twice, or a layout outside the set is a
 * refusal with its line, never a skipped line, since a recipe that loads short is a different
 * deployment from the one its author measured. Which layers keep an attention cache only the
 * plugin knows, so that check is the plugin's (RadBuildCtx::kv_recipe). */
int kv_recipe_parse(const std::string& text, const char* name, std::vector<int32_t>* out,
                    std::string* vmean_file) {
    /* Past any model in this tree by far; it bounds the array the engine hands the plugin. */
    constexpr long kMaxLayer = 4096;
    out->clear();
    if (vmean_file) vmean_file->clear();
    std::vector<int32_t> r;
    std::string file;
    int vmean_layers = 0;
    std::istringstream in(text);
    std::string line;
    for (int ln = 1; std::getline(in, line); ++ln) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        std::istringstream words(line);
        std::string kw, num, layout, extra;
        if (!(words >> kw)) continue;
        if (kw == "vmean") {
            if (!(words >> file) || (words >> extra)) {
                RAD_ERR("%s line %d: 'vmean FILE' names the V mean file, two words", name, ln);
                return RAD_E_INVAL;
            }
            continue;
        }
        if (kw != "layer") {
            RAD_ERR("%s line %d: '%s' is not a recipe line; each is 'layer N LAYOUT'", name, ln,
                    kw.c_str());
            return RAD_E_INVAL;
        }
        /* After the layout: `rotated`, then `demote LAYOUT`, then `vmean`, each at most once and
         * in that order. */
        bool rot_word = false, vmean = false, bad = !(words >> num >> layout);
        std::string demote;
        for (std::string w; !bad && (words >> w);) {
            if (w == "rotated" && !rot_word && demote.empty() && !vmean) rot_word = true;
            else if (w == "demote" && demote.empty() && !vmean) bad = !(words >> demote);
            else if (w == "vmean" && !vmean) vmean = true;
            else bad = true;
        }
        if (bad) {
            RAD_ERR("%s line %d: a recipe line is 'layer N LAYOUT', three words, or 'layer N "
                    "LAYOUT rotated', then 'demote LAYOUT' where window residency stores the "
                    "layer's demoted blocks narrower, and 'vmean' last where the layer subtracts "
                    "its V mean", name, ln);
            return RAD_E_INVAL;
        }
        char* end = nullptr;
        errno = 0;
        const long l = std::strtol(num.c_str(), &end, 10);
        if (errno || !end || *end || num[0] == '-' || num[0] == '+' || l >= kMaxLayer) {
            RAD_ERR("%s line %d: layer '%s' is not a layer number (0 to %ld)", name, ln,
                    num.c_str(), kMaxLayer - 1);
            return RAD_E_INVAL;
        }
        const int q = kv_layout_of(layout.c_str());
        if (q < 0) {
            RAD_ERR("%s line %d: layer %ld: '%s' is not a KV layout; bf16, fp8, fp8_v4, k6_v4, "
                    "k4h_v4, k4h_v2h or i8h", name, ln, l, layout.c_str());
            return RAD_E_INVAL;
        }
        if (q == RAD_KVQ_BF16 && (rot_word || vmean)) {
            RAD_ERR("%s line %d: layer %ld: bf16 is the unquantised cache and takes no '%s'; a "
                    "rotation or a V mean only helps a layout that rounds to a narrower code",
                    name, ln, l, rot_word ? "rotated" : "vmean");
            return RAD_E_INVAL;
        }
        if (q == RAD_KVQ_BF16 && !demote.empty()) {
            RAD_ERR("%s line %d: layer %ld: bf16 is the unquantised cache and takes no 'demote'; "
                    "it keeps its blocks as they are, and a tiered layer is i8h or fp8", name, ln,
                    l);
            return RAD_E_INVAL;
        }
        /* The demote layout (RAD_KVQ_DEMOTE): served for an i8h or fp8 layer demoted to k4h_v4
         * or k4h_v2h. The store holds the same values in the layer's basis, so a demote layout
         * that only holds rotated keys needs the layer rotated: i8h is, fp8 must say `rotated`. */
        const int dq = demote.empty() ? 0 : kv_layout_of(demote.c_str());
        if (!demote.empty()) {
            if (dq != RAD_KVQ_K4H && dq != RAD_KVQ_K4HV2H) {
                RAD_ERR("%s line %d: layer %ld: 'demote %s': a demote layout is k4h_v4 or k4h_v2h",
                        name, ln, l, demote.c_str());
                return RAD_E_INVAL;
            }
            if (q != RAD_KVQ_I8H && q != RAD_KVQ_NONE) {
                RAD_ERR("%s line %d: layer %ld: 'demote %s' under %s: the resident layout of a "
                        "tiered layer is i8h or fp8", name, ln, l, demote.c_str(), layout.c_str());
                return RAD_E_INVAL;
            }
            if (q == RAD_KVQ_NONE && !rot_word) {
                RAD_ERR("%s line %d: layer %ld: 'demote %s' holds rotated keys only, and the "
                        "demoted block keeps the layer's basis; write 'layer %ld fp8 rotated "
                        "demote %s'", name, ln, l, demote.c_str(), l, demote.c_str());
                return RAD_E_INVAL;
            }
        }
        /* Whichever of the two layouts has integer V: the mean is the layer's, subtracted before
         * its store, so the resident and the demoted bytes are both centred. */
        if (vmean && q == RAD_KVQ_NONE && dq == 0) {
            RAD_ERR("%s line %d: layer %ld: 'vmean' is for a layout with integer V (fp8_v4, "
                    "k6_v4, k4h_v4, k4h_v2h, i8h); fp8 keeps V in E4M3", name, ln, l);
            return RAD_E_INVAL;
        }
        if ((size_t)l >= r.size()) r.resize((size_t)l + 1, -1);
        if (r[(size_t)l] >= 0) {
            RAD_ERR("%s line %d: layer %ld is named twice", name, ln, l);
            return RAD_E_INVAL;
        }
        const bool rotated_layout = q == RAD_KVQ_K4H || q == RAD_KVQ_K4HV2H || q == RAD_KVQ_I8H;
        r[(size_t)l] = q | (rot_word || rotated_layout ? RAD_KVQ_ROTATED : 0) |
                       (vmean ? RAD_KVQ_VMEAN : 0) | (dq << RAD_KVQ_DEMOTE_SHIFT);
        vmean_layers += vmean;
    }
    if (r.empty()) {
        RAD_ERR("%s names no layer; each attention layer needs a 'layer N LAYOUT' line", name);
        return RAD_E_INVAL;
    }
    /* Both halves or neither: a mean file nothing reads is a recipe that says one thing and does
     * another, and a `vmean` layer needs its means. */
    if (vmean_layers && file.empty()) {
        RAD_ERR("%s: %d layer(s) say 'vmean' and no 'vmean FILE' line names their means", name,
                vmean_layers);
        return RAD_E_INVAL;
    }
    if (!vmean_layers && !file.empty()) {
        RAD_ERR("%s: 'vmean %s' and no layer says 'vmean'; mark the layers that subtract it",
                name, file.c_str());
        return RAD_E_INVAL;
    }
    *out = std::move(r);
    if (vmean_file) *vmean_file = file;
    return RAD_OK;
}

std::string kv_recipe_summary(const std::vector<int32_t>& recipe) {
    std::string s;
    for (int rot : { 0, (int)RAD_KVQ_ROTATED })
        for (int q : kv_layouts())
            for (int dq : { (int)RAD_KVQ_NONE, (int)RAD_KVQ_K4H, (int)RAD_KVQ_K4HV2H }) {
                const int key = q | rot | (dq << RAD_KVQ_DEMOTE_SHIFT);
                std::string layers;
                for (size_t l = 0; l < recipe.size(); ++l)
                    if (recipe[l] >= 0 && (recipe[l] & ~RAD_KVQ_VMEAN) == key)
                        layers += (layers.empty() ? "" : ", ") + std::to_string(l);
                if (layers.empty()) continue;
                s += (s.empty() ? "" : "; ") + std::string(kv_layout_name(q)) +
                     (rot ? " rotated" : "") +
                     (dq ? std::string(" demote ") + kv_layout_name(dq) : std::string()) +
                     " at layer" + (layers.find(',') != std::string::npos ? "s " : " ") + layers;
            }
    std::string vm;
    for (size_t l = 0; l < recipe.size(); ++l)
        if (recipe[l] >= 0 && (recipe[l] & RAD_KVQ_VMEAN))
            vm += (vm.empty() ? "" : ", ") + std::to_string(l);
    if (!vm.empty())
        s += "; V mean subtracted at layer" + std::string(vm.find(',') != std::string::npos ? "s " : " ") + vm;
    return s;
}

/* The V mean file. Text, so a calibration's result can be read, diffed and checked by eye; the
 * values are printed at %.9g by the writer (arch/common/rad_calib_vmean.h), which an f32 reads
 * back exactly. A file holds every attention layer the calibration saw; a recipe takes the ones
 * it marks. */
int kv_vmean_parse(const std::string& text, const char* name, KvVMean* out) {
    *out = KvVMean{};
    KvVMean v;
    bool header = false;
    std::map<int, std::vector<char>> seen;   /* heads present, by layer */
    std::istringstream in(text);
    std::string line;
    int ln = 0;
    for (ln = 1; std::getline(in, line); ++ln) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        std::istringstream words(line);
        std::string kw;
        if (!(words >> kw)) continue;
        if (!header) {
            std::string ver, k1, k2, k3, extra;
            long long d = 0, h = 0, rows = 0;
            if (kw != "kv_vmean" || !(words >> ver >> k1 >> d >> k2 >> h >> k3 >> rows) ||
                ver != "1" || k1 != "head_dim" || k2 != "kv_heads" || k3 != "rows" ||
                (words >> extra) || d <= 0 || d > 4096 || h <= 0 || h > 1024 || rows < 0) {
                RAD_ERR("%s line %d: the first line is 'kv_vmean 1 head_dim D kv_heads H rows R'",
                        name, ln);
                return RAD_E_INVAL;
            }
            v.head_dim = d;
            v.n_head_kv = h;
            v.rows = rows;
            header = true;
            continue;
        }
        long long l = -1, h = -1;
        std::string hw;
        if (kw != "layer" || !(words >> l >> hw >> h) || hw != "head" || l < 0 || l >= 4096) {
            RAD_ERR("%s line %d: a line is 'layer L head H' and the head's %lld values", name, ln,
                    (long long)v.head_dim);
            return RAD_E_INVAL;
        }
        if (h < 0 || h >= v.n_head_kv) {
            RAD_ERR("%s line %d: layer %lld: head %lld is outside the file's %lld KV heads", name,
                    ln, l, h, (long long)v.n_head_kv);
            return RAD_E_INVAL;
        }
        std::vector<char>& hs = seen[(int)l];
        hs.resize((size_t)v.n_head_kv, 0);
        if (hs[(size_t)h]) {
            RAD_ERR("%s line %d: layer %lld head %lld is listed twice", name, ln, l, h);
            return RAD_E_INVAL;
        }
        hs[(size_t)h] = 1;
        std::vector<float>& m = v.layer[(int)l];
        m.resize((size_t)(v.n_head_kv * v.head_dim), 0.0f);
        int64_t n = 0;
        for (std::string w; words >> w; ++n) {
            char* end = nullptr;
            const float x = std::strtof(w.c_str(), &end);
            if (!end || *end || !std::isfinite(x)) {
                RAD_ERR("%s line %d: layer %lld head %lld: '%s' is not a finite number", name, ln,
                        l, h, w.c_str());
                return RAD_E_INVAL;
            }
            if (n < v.head_dim) m[(size_t)(h * v.head_dim + n)] = x;
        }
        if (n != v.head_dim) {
            RAD_ERR("%s line %d: layer %lld head %lld has %lld values; head_dim is %lld", name,
                    ln, l, h, (long long)n, (long long)v.head_dim);
            return RAD_E_INVAL;
        }
    }
    if (!header) {
        RAD_ERR("%s: no 'kv_vmean 1 head_dim D kv_heads H rows R' line", name);
        return RAD_E_INVAL;
    }
    for (const auto& [l, hs] : seen)
        for (size_t h = 0; h < hs.size(); ++h)
            if (!hs[h]) {
                RAD_ERR("%s: layer %d has no line for head %zu; a layer lists every KV head", name,
                        l, h);
                return RAD_E_INVAL;
            }
    v.path = name;
    *out = std::move(v);
    return RAD_OK;
}

int kv_vmean_table(const KvVMean& vm, const std::vector<int32_t>& recipe, int64_t head_dim,
                   int64_t n_head_kv, int64_t n_head, std::vector<uint16_t>* out) {
    out->clear();
    const char* name = vm.path.c_str();
    if (vm.head_dim != head_dim || vm.n_head_kv != n_head_kv) {
        RAD_ERR("%s holds %lld KV heads of %lld and this model has %lld of %lld; it was "
                "calibrated on another model", name, (long long)vm.n_head_kv,
                (long long)vm.head_dim, (long long)n_head_kv, (long long)head_dim);
        return RAD_E_INVAL;
    }
    /* The normalised Hadamard over head_dim in the order had_rotate keeps (libref's fwht, the
     * Sylvester order), in f64: a rotated layer stores H v / sqrt(D), so its mean is H mu /
     * sqrt(D). Any vector would keep the layer exact; this one is the one that helps. */
    if (head_dim <= 0 || (head_dim & (head_dim - 1)) || n_head_kv <= 0 || n_head % n_head_kv) {
        RAD_ERR("%s: head_dim %lld is not a power of two or %lld query heads do not share %lld KV "
                "heads evenly", name, (long long)head_dim, (long long)n_head, (long long)n_head_kv);
        return RAD_E_INVAL;
    }
    const int64_t D = head_dim, E = (n_head_kv + n_head) * D, gqa = n_head / n_head_kv;
    std::vector<uint16_t> t((size_t)(recipe.size() * 2 * E), 0);
    for (size_t l = 0; l < recipe.size(); ++l) {
        if (recipe[l] < 0 || !(recipe[l] & RAD_KVQ_VMEAN)) continue;
        const auto it = vm.layer.find((int)l);
        if (it == vm.layer.end()) {
            RAD_ERR("%s has no V mean for layer %zu, which the recipe marks 'vmean'", name, l);
            return RAD_E_INVAL;
        }
        for (int r = 0; r < 2; ++r) {
            uint16_t* e = t.data() + (l * 2 + (size_t)r) * (size_t)E;
            for (int64_t k = 0; k < n_head_kv; ++k) {
                std::vector<double> mu(it->second.begin() + k * D, it->second.begin() + (k + 1) * D);
                if (r) {
                    for (int64_t h = 1; h < D; h <<= 1)
                        for (int64_t i = 0; i < D; i += 2 * h)
                            for (int64_t j = i; j < i + h; ++j) {
                                const double a = mu[(size_t)j], b = mu[(size_t)(j + h)];
                                mu[(size_t)j] = a + b;
                                mu[(size_t)(j + h)] = a - b;
                            }
                    for (double& x : mu) x /= std::sqrt((double)D);
                }
                for (int64_t d = 0; d < D; ++d) {
                    /* bf16 rounds to nearest even, symmetric in sign: -bf16(mu) is bf16(-mu). */
                    const uint16_t pos = rad_f32_to_bf16((float)mu[(size_t)d]);
                    e[k * D + d] = (uint16_t)(pos ^ 0x8000u);
                    for (int64_t q = k * gqa; q < (k + 1) * gqa; ++q)
                        e[n_head_kv * D + q * D + d] = pos;
                }
            }
        }
    }
    *out = std::move(t);
    return RAD_OK;
}

/* Returns 0 on success, 1 for --help (caller exits 0), negative on a bad argument. An unknown flag
 * is an error, not a warning, since a mistyped budget flag changes the plan. */
int config_parse(int argc, char** argv, Config* c) {
    /* --help wins from anywhere on the line, even after an otherwise fatal argument. */
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) return 1;

    auto need = [&](int& i) -> const char* {
        if (i + 1 >= argc) { RAD_ERR("%s needs an argument", argv[i]); return nullptr; }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        const char* v = nullptr;
        auto S = [&](const char* n) { return a == n; };
        /* The flag's argument as a whole number in [lo, hi], stored into *out. */
        auto num = [&](int64_t lo, int64_t hi, auto* out) {
            int64_t n = 0;
            if (!(v = need(i)) || !parse_int(a.c_str(), v, lo, hi, &n)) return false;
            *out = static_cast<std::remove_reference_t<decltype(*out)>>(n);
            return true;
        };

        if (S("-v"))                           log_set_level(Log::Debug);
        else if (S("-vv"))                     log_set_level(Log::Trace);
        else if (S("--model"))                 { if (!(v = need(i))) return RAD_E_INVAL; c->model = v; }
        else if (S("--download-dir"))          { if (!(v = need(i))) return RAD_E_INVAL; c->download_dir = v; }
        else if (S("--model-revision"))        { if (!(v = need(i))) return RAD_E_INVAL; c->model_revision = v; }
        else if (S("--radiance-home"))         { if (!(v = need(i))) return RAD_E_INVAL; c->radiance_home = v; }
        else if (S("--kernels"))               { if (!(v = need(i))) return RAD_E_INVAL; c->kernel_hierarchy = rad_split_kernel_list(v); }
        else if (S("--tp"))                    { if (!num(1, INT32_MAX, &c->tp)) return RAD_E_INVAL; }
        else if (S("--tp-wire")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if      (!std::strcmp(v, "exact")) c->tp_wire_exact = 1;
            else if (!std::strcmp(v, "wht6"))  c->tp_wire_exact = 0;
            else { RAD_ERR("--tp-wire takes exact or wht6, not '%s'", v); return RAD_E_INVAL; }
        }
        else if (S("--tp-attn-split")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if      (!std::strcmp(v, "heads")) c->tp_attn_rows = 0;
            else if (!std::strcmp(v, "rows"))  c->tp_attn_rows = 1;
            else { RAD_ERR("--tp-attn-split takes heads or rows, not '%s'", v); return RAD_E_INVAL; }
        }
        else if (S("--weight-prefetch")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if      (!std::strcmp(v, "on"))  c->weight_prefetch = true;
            else if (!std::strcmp(v, "off")) c->weight_prefetch = false;
            else { RAD_ERR("--weight-prefetch takes on or off, not '%s'", v); return RAD_E_INVAL; }
        }
        else if (S("--p2p")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if      (!std::strcmp(v, "auto")) c->p2p = RAD_P2P_AUTO;
            else if (!std::strcmp(v, "on"))   c->p2p = RAD_P2P_ON;
            else if (!std::strcmp(v, "off"))  c->p2p = RAD_P2P_OFF;
            else { RAD_ERR("--p2p takes auto, on or off, not '%s'", v); return RAD_E_INVAL; }
        }
        else if (S("--tp-wire-min-kb"))        { if (!num(0, INT64_MAX, &c->tp_wire_min_kb)) return RAD_E_INVAL; }
        else if (S("--max-num-batched-tokens")){ if (!num(1, INT64_MAX, &c->max_tok)) return RAD_E_INVAL; }
        else if (S("--busy-prefill-chunk")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if      (!std::strcmp(v, "auto")) c->busy_prefill_chunk = -1;
            else if (!std::strcmp(v, "off"))  c->busy_prefill_chunk = 0;
            else if (!parse_int(a.c_str(), v, 1, INT64_MAX, &c->busy_prefill_chunk)) return RAD_E_INVAL;
        }
        else if (S("--prefill-interleave"))    { if (!num(0, 64, &c->prefill_interleave)) return RAD_E_INVAL; }
        else if (S("--preempt-wait-ms"))       { if (!num(0, INT64_MAX, &c->preempt_wait_ms)) return RAD_E_INVAL; }
        else if (S("--max-num-seqs"))          { if (!num(1, INT64_MAX, &c->max_seqs)) return RAD_E_INVAL; }
        else if (S("--max-model-len"))         { if (!num(0, INT64_MAX, &c->max_ctx)) return RAD_E_INVAL; }
        else if (S("--rope-scaling")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            /* Checked here so a typo is refused rather than served with the trained rotary. */
            RopeSpec rs;
            if (!rope_spec_parse(v, &rs) || rs.kind == RopeSpec::Keep) {
                RAD_ERR("--rope-scaling takes yarn:FACTOR (above 1) with an optional :ORIG trained "
                        "length, or none -- got '%s'", v);
                return RAD_E_INVAL;
            }
            c->rope_scaling = v;
        }
        /* The two pools are always sized by the budget resolver (core/mem/vram_budget.h); point
         * old launch lines at the flags that move the split. */
        else if (S("--vram-weights-mib") || S("--vram-kv-mib")) {
            RAD_ERR("%s was removed: the card is split between the weights and the cache from what "
                    "the model needs. --gpu-headroom-mib takes card space away, --max-model-len and "
                    "--max-num-seqs bound the cache, and --host-pool-mib decides which experts stay "
                    "on the card", a.c_str());
            return RAD_E_INVAL;
        }
        /* --gpu-headroom is accepted as an alias of --gpu-headroom-mib. */
        else if (S("--gpu-headroom-mib") || S("--gpu-headroom"))
                                               { if (!num(0, INT64_MAX, &c->gpu_headroom_mib)) return RAD_E_INVAL; }
        else if (S("--host-headroom-mib"))      { if (!num(0, INT64_MAX, &c->host_headroom_mib)) return RAD_E_INVAL; }
        else if (S("--kv-cache-dtype")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (std::strcmp(v, "bf16") && std::strcmp(v, "fp8") && std::strcmp(v, "fp8_v4") &&
                std::strcmp(v, "k6_v4")) {
                RAD_ERR("--kv-cache-dtype takes bf16, fp8, fp8_v4 or k6_v4, not '%s'", v);
                return RAD_E_INVAL;
            }
            c->kv_cache_dtype = v;
        }
        else if (S("--kv-recipe")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            std::ifstream f(v, std::ios::binary);
            if (!f) {
                RAD_ERR("--kv-recipe: cannot read '%s'", v);
                return RAD_E_INVAL;
            }
            const std::string text((std::istreambuf_iterator<char>(f)),
                                   std::istreambuf_iterator<char>());
            const std::string name = std::string("--kv-recipe ") + v;
            std::string vfile;
            const int rc = kv_recipe_parse(text, name.c_str(), &c->kv_recipe, &vfile);
            if (rc != RAD_OK) return rc;
            c->kv_recipe_path = v;
            c->kv_vmean = KvVMean{};
            if (!vfile.empty()) {
                /* Beside the recipe unless absolute, so a recipe and its means move together. */
                const std::string rp(v);
                const size_t slash = rp.find_last_of('/');
                const std::string vpath = vfile[0] == '/' || slash == std::string::npos
                                              ? vfile : rp.substr(0, slash + 1) + vfile;
                std::ifstream vf(vpath, std::ios::binary);
                if (!vf) {
                    RAD_ERR("%s: cannot read its V mean file '%s'", name.c_str(), vpath.c_str());
                    return RAD_E_INVAL;
                }
                const std::string vt((std::istreambuf_iterator<char>(vf)),
                                     std::istreambuf_iterator<char>());
                const int vr = kv_vmean_parse(vt, vpath.c_str(), &c->kv_vmean);
                if (vr != RAD_OK) return vr;
                for (size_t l = 0; l < c->kv_recipe.size(); ++l)
                    if (c->kv_recipe[l] >= 0 && (c->kv_recipe[l] & RAD_KVQ_VMEAN) &&
                        !c->kv_vmean.layer.count((int)l)) {
                        RAD_ERR("%s: layer %zu says 'vmean' and %s has no V mean for it",
                                name.c_str(), l, vpath.c_str());
                        return RAD_E_INVAL;
                    }
                for (const auto& [l, mu] : c->kv_vmean.layer)
                    if ((size_t)l >= c->kv_recipe.size() || c->kv_recipe[(size_t)l] < 0) {
                        RAD_ERR("%s: %s holds layer %d, which the recipe does not name; a mean "
                                "file from another model's recipe", name.c_str(), vpath.c_str(), l);
                        return RAD_E_INVAL;
                    }
            }
        }
        else if (S("--state-cache-dtype")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (std::strcmp(v, "f32") && std::strcmp(v, "f16")) {
                RAD_ERR("--state-cache-dtype takes f32 or f16, not '%s'", v);
                return RAD_E_INVAL;
            }
            c->state_cache_dtype = v;
        }
        else if (S("--qsa-bkey-dtype")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (std::strcmp(v, "bf16") && std::strcmp(v, "i8")) {
                RAD_ERR("--qsa-bkey-dtype takes bf16 or i8, not '%s'", v);
                return RAD_E_INVAL;
            }
            c->qsa_bkey_dtype = v;
        }
        else if (S("--qsa-bkey-split")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if      (!std::strcmp(v, "on"))  c->qsa_bkey_split = true;
            else if (!std::strcmp(v, "off")) c->qsa_bkey_split = false;
            else {
                RAD_ERR("--qsa-bkey-split takes on or off, not '%s'", v);
                return RAD_E_INVAL;
            }
        }
        else if (S("--attn-prefill-fp8")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if      (!std::strcmp(v, "off"))  c->attn_prefill_fp8 = 0;
            else if (!std::strcmp(v, "qk"))   c->attn_prefill_fp8 = RAD_ATTN_FP8_QK;
            else if (!std::strcmp(v, "pv"))   c->attn_prefill_fp8 = RAD_ATTN_FP8_PV;
            else if (!std::strcmp(v, "both")) c->attn_prefill_fp8 = RAD_ATTN_FP8_QK | RAD_ATTN_FP8_PV;
            else {
                RAD_ERR("--attn-prefill-fp8 takes off, qk, pv or both, not '%s'", v);
                return RAD_E_INVAL;
            }
        }
        else if (S("--kv-resident"))           { if (!num(0, INT64_MAX, &c->kv_resident)) return RAD_E_INVAL; }
        else if (S("--host-pool-mib"))         { if (!num(0, INT64_MAX, &c->host_pool_mib)) return RAD_E_INVAL; }
        else if (S("--embedding-placement") || S("--ngram-placement")) {
            const bool emb = S("--embedding-placement");
            if (!(v = need(i))) return RAD_E_INVAL;
            TablePlacement p;
            /* The embedding has two tiers: its lookup writes device activations, so there is no
             * disk tier, and with ram the default there is nothing for auto to choose. */
            if (emb && (!parse_table_placement(v, &p) ||
                        (p != TablePlacement::Ram && p != TablePlacement::Vram))) {
                if (parse_table_placement(v, &p) && p == TablePlacement::Auto)
                    RAD_ERR("--embedding-placement auto was removed: ram, the default, holds the "
                            "table in pinned host memory, and vram keeps it on the card");
                else
                    RAD_ERR("--embedding-placement takes ram or vram, not '%s'", v);
                return RAD_E_INVAL;
            }
            std::string share_dir;
            if (!emb && !std::strncmp(v, "ram=", 4)) {
                share_dir = v + 4;
                if (share_dir.empty()) {
                    RAD_ERR("--ngram-placement ram= needs a directory: ram=DIR, the tmpfs that "
                            "holds the shared table");
                    return RAD_E_INVAL;
                }
                p = TablePlacement::Ram;
            } else if (!emb && !parse_table_placement(v, &p)) {
                RAD_ERR("%s takes auto, vram, ram, ram=DIR or disk, not '%s'", a.c_str(), v);
                return RAD_E_INVAL;
            }
            if (!emb) c->ngram_share_dir = share_dir;
            (emb ? c->embedding_placement : c->ngram_placement) = p;
            if (emb) c->embedding_placement_given = true;
        }
        else if (S("--vision-tower-placement")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            TablePlacement p;
            /* Two tiers, as for the embedding: an encoder writes device activations. */
            if (!parse_table_placement(v, &p) || (p != TablePlacement::Ram && p != TablePlacement::Vram)) {
                RAD_ERR("--vision-tower-placement takes vram or ram, not '%s'", v);
                return RAD_E_INVAL;
            }
            c->tower_placement = p;
        }
        else if (S("--weights-disk-tier"))     { c->weights_disk_tier = true; }
        else if (S("--prefix-cache-mib"))      { if (!num(0, INT32_MAX, &c->prefix_cache_mib)) return RAD_E_INVAL; }
        else if (S("--prefix-cache-ttl"))      { if (!num(0, INT32_MAX, &c->prefix_cache_ttl)) return RAD_E_INVAL; }
        else if (S("--kv-residency-mib"))     { if (!num(0, INT32_MAX, &c->residency_mib)) return RAD_E_INVAL; }
        else if (S("--kv-residency-sink-tokens")) {
            if (!num(0, INT32_MAX, &c->residency_sink_tokens)) return RAD_E_INVAL;
        }
        else if (S("--lora")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            const char* eq = std::strchr(v, '=');
            if (!eq || eq == v || !eq[1]) {
                RAD_ERR("--lora wants NAME=PATH, got '%s'", v);
                return RAD_E_INVAL;
            }
            c->lora.adapters.emplace_back(std::string(v, eq), std::string(eq + 1));
        }
        else if (S("--lora-dir"))              { if (!(v = need(i))) return RAD_E_INVAL; c->lora.dir = v; }
        else if (S("--max-loras"))             { if (!num(0, 64, &c->lora.max_loras)) return RAD_E_INVAL; }
        else if (S("--max-lora-rank"))         { if (!num(1, 512, &c->lora.max_rank)) return RAD_E_INVAL; }
        else if (S("--lora-target-modules")) {
            /* Module leaf names, as PEFT's target_modules lists them: a dotted path or a pattern
             * would match no leaf and target nothing without a word. */
            if (!(v = need(i))) return RAD_E_INVAL;
            std::set<std::string> leaves;
            std::string cur;
            for (const char* q = v;; ++q) {
                if (*q == ',' || !*q) {
                    if (!cur.empty()) leaves.insert(cur);
                    cur.clear();
                    if (!*q) break;
                } else if (std::isalnum((unsigned char)*q) || *q == '_') {
                    cur += *q;
                } else if (*q != ' ') {
                    RAD_ERR("--lora-target-modules takes module leaf names (q_proj,k_proj,...); "
                            "'%s' holds '%c'", v, *q);
                    return RAD_E_INVAL;
                }
            }
            if (leaves.empty()) {
                RAD_ERR("--lora-target-modules names no module");
                return RAD_E_INVAL;
            }
            c->lora.targets.assign(leaves.begin(), leaves.end());
        }
        /* The prefix cache has no disk tier; the removed flags name their replacements. */
        else if (S("--prefix-cache-host-mib")) {
            RAD_ERR("--prefix-cache-host-mib was renamed: use --prefix-cache-mib");
            return RAD_E_INVAL;
        }
        else if (S("--prefix-cache-dir") || S("--prefix-cache-disk-mib")) {
            RAD_ERR("%s was removed: the prefix cache is kept in VRAM and host memory only "
                    "(--prefix-cache-mib), and --prefix-cache-ttl says how long", a.c_str());
            return RAD_E_INVAL;
        }
        else if (S("--placement")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            /* The planner's names, checked here so a typo stops the parse rather than startup
             * after the container is opened. */
            Strategy st;
            if (strategy_parse(v, &st) < 0) {
                RAD_ERR("--placement takes all_vram, layer_offload, expert_tiered or auto, not "
                        "'%s'", v);
                return RAD_E_INVAL;
            }
            c->placement = v;
        }
        else if (S("--control-vector")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (std::strcmp(v, "off") && std::strcmp(v, "on") && std::strcmp(v, "disabled")) {
                RAD_ERR("--control-vector takes off, on or disabled -- got '%s'", v);
                return RAD_E_INVAL;
            }
            c->control_vector = v;
        }
        else if (S("--deterministic"))         c->deterministic = true;
        else if (S("--checkpoint-interval"))   { if (!num(0, INT64_MAX, &c->checkpoint_interval)) return RAD_E_INVAL; }
        else if (S("--prefill-span"))          { if (!num(1, INT32_MAX, &c->prefill_span)) return RAD_E_INVAL; }
        else if (S("--checkpoint-slots")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (!std::strcmp(v, "auto")) {
                c->checkpoint_slots = -1;
            } else if (!*v || std::strspn(v, "0123456789") != std::strlen(v)) {
                RAD_ERR("--checkpoint-slots takes auto or a whole number, not '%s'", v);
                return RAD_E_INVAL;
            } else if (!parse_int(a.c_str(), v, 0, INT64_MAX, &c->checkpoint_slots)) {
                return RAD_E_INVAL;
            }
        }
        else if (S("--no-prefix-cache"))       c->prefix_cache = false;
        else if (S("--prefix-resume")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (!std::strcmp(v, "exact"))        c->prefix_exact = true;
            else if (!std::strcmp(v, "nearest")) c->prefix_exact = false;
            else {
                RAD_ERR("--prefix-resume '%s' is not one of exact, nearest", v);
                return RAD_E_INVAL;
            }
        }
        else if (S("--checkpoint-policy")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            /* Validated, so an unknown name does not silently fall back to the default. */
            if (std::strcmp(v, "geometric-backoff") && std::strcmp(v, "tip-only") &&
                std::strcmp(v, "keep-all")) {
                RAD_ERR("--checkpoint-policy '%s' is not one of geometric-backoff, tip-only, "
                        "keep-all", v);
                return RAD_E_INVAL;
            }
            c->checkpoint_policy = v;
        }
        else if (S("--num-speculative-tokens")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            /* "auto" is -1, resolved once the container is open from the architecture plugin
             * (rad_arch_probe): a serial head's measured operating point, or a block drafter's
             * trained block length. Any negative depth means auto downstream, so only the word
             * is accepted. */
            int64_t n = -1;
            if (std::strcmp(v, "auto") && !parse_int(a.c_str(), v, 0, INT32_MAX, &n))
                return RAD_E_INVAL;
            c->n_spec = (int)n;
        }
        else if (S("--spec-ngram"))            { if (!num(0, 64, &c->spec_ngram)) return RAD_E_INVAL; }
        else if (S("--spec-ngram-min"))        { if (!num(1, 15, &c->spec_ngram_min)) return RAD_E_INVAL; }
        else if (S("--spec-ngram-max"))        { if (!num(1, 15, &c->spec_ngram_max)) return RAD_E_INVAL; }
        else if (S("--draft-window")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if      (!std::strcmp(v, "fixed"))    c->draft_window = DraftWindow::Fixed;
            else if (!std::strcmp(v, "adaptive")) c->draft_window = DraftWindow::Adaptive;
            else {
                RAD_ERR("--draft-window takes fixed or adaptive, not '%s'", v);
                return RAD_E_INVAL;
            }
        }
        else if (S("--draft-short-from")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (!std::strcmp(v, "auto")) c->draft_short_from = -1;
            else if (!parse_int(a.c_str(), v, 0, INT32_MAX, &c->draft_short_from)) return RAD_E_INVAL;
        }
        else if (S("--draft-coupling")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if      (!std::strcmp(v, "off")) c->draft_coupling = false;
            else if (!std::strcmp(v, "on"))  c->draft_coupling = true;
            else {
                RAD_ERR("--draft-coupling takes off or on, not '%s'", v);
                return RAD_E_INVAL;
            }
        }
        else if (S("--draft-model")) {
            /* Refused, not ignored: separate draft models are not supported (spec §10), and the
             * drafter comes from the container (rad-convert --draft-model merges one in; an MTP
             * head inside the checkpoint needs no flag). Kept so vLLM users get a message saying
             * so. */
            if (!(v = need(i))) return RAD_E_INVAL;
            RAD_ERR("--draft-model is not implemented. This engine loads its drafter from the "
                    "model container, not from a second one: merge it at conversion time with "
                    "`rad-convert --draft-model DIR` and pass the container to --model. "
                    "An MTP head that ships inside the target checkpoint needs neither flag.");
            return RAD_E_INVAL;
        }
        else if (S("--host"))                  { if (!(v = need(i))) return RAD_E_INVAL; c->host = v; }
        else if (S("--port"))                  { if (!num(0, 65535, &c->port)) return RAD_E_INVAL; }
        else if (S("--api-key")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            /* An empty key is no key: --api-key "" or --api-key "$UNSET_VAR" must not start a
             * server that looks guarded and answers everything keyless. */
            if (!*v) { RAD_ERR("--api-key takes a non-empty key; an empty value guards nothing");
                       return RAD_E_INVAL; }
            c->api_key = v;
        }
        else if (S("--api-key-env")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (!*v) { RAD_ERR("--api-key-env takes the name of an environment variable");
                       return RAD_E_INVAL; }
            const char* k = std::getenv(v);
            /* The empty --api-key refusal's sibling: a unit or compose file whose variable was
             * never set must not start a server that looks guarded and answers everything
             * keyless. The name is recorded so the config report can say where the key came
             * from without showing it. */
            if (!k || !*k) {
                RAD_ERR("--api-key-env: %s is unset or empty; refusing to serve keyless under a "
                        "guarded-looking command line", v);
                return RAD_E_INVAL;
            }
            c->api_key = k;
            c->api_key_env = v;
        }
        else if (S("--allow-host")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (!*v) { RAD_ERR("--allow-host takes a non-empty name"); return RAD_E_INVAL; }
            c->allow_hosts.emplace_back(v);
        }
        else if (S("--trusted-proxy")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (!server::cidr_valid(v)) {
                RAD_ERR("--trusted-proxy takes a numeric network with a prefix length, like "
                        "127.0.0.0/8 or ::1/128 (no brackets, no bare address), got '%s'", v);
                return RAD_E_INVAL;
            }
            c->trusted_proxies.emplace_back(v);
        }
        else if (S("--served-model-name")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (!*v) { RAD_ERR("--served-model-name takes a non-empty name"); return RAD_E_INVAL; }
            c->served_model_name = v;
        }
        else if (S("--image-gen-max-pixels")) {
            if (!(v = need(i)) || !parse_int(a.c_str(), v, 256, INT64_MAX, &c->image_gen_max_pixels))
                return RAD_E_INVAL;
        }
        else if (S("--image-gen-max-refs")) {
            if (!(v = need(i)) || !parse_int(a.c_str(), v, 1, INT64_MAX, &c->image_gen_max_refs))
                return RAD_E_INVAL;
        }
        else if (S("--mm-max-patches")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (!std::strcmp(v, "auto")) c->mm_max_patches = -1;
            else if (!parse_int(a.c_str(), v, 0, INT64_MAX, &c->mm_max_patches)) return RAD_E_INVAL;
        }
        else if (S("--reasoning-effort"))      { if (!(v = need(i))) return RAD_E_INVAL;
                                                 c->reasoning_effort = v; }
        else if (S("--reasoning-effort-map"))  {
            if (!(v = need(i))) return RAD_E_INVAL;
            const std::string all = v;
            size_t at = 0;
            while (at <= all.size()) {
                const size_t end = std::min(all.find(',', at), all.size());
                const std::string pair = all.substr(at, end - at);
                const size_t eq = pair.find('=');
                if (eq == std::string::npos || eq == 0 || eq + 1 == pair.size() ||
                    pair.find('=', eq + 1) != std::string::npos) {
                    RAD_ERR("--reasoning-effort-map takes FROM=TO pairs separated by commas, "
                            "not '%s'", pair.c_str());
                    return RAD_E_INVAL;
                }
                c->reasoning_effort_map[pair.substr(0, eq)] = pair.substr(eq + 1);
                at = end + 1;
            }
        }
        else if (S("--thinking-token-budget")) { if (!num(0, INT32_MAX, &c->thinking_token_budget)) return RAD_E_INVAL; }
        else if (S("--generation-config"))     { if (!(v = need(i))) return RAD_E_INVAL;
                                                 c->generation_config = v; }
        else if (S("--override-chat-template")) { if (!(v = need(i))) return RAD_E_INVAL;
                                                 c->override_chat_template = v; }
        /* The range check refuses negatives, which these fields use for "not set". */
        else if (S("--temp") || S("--top-p") || S("--min-p")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            const char* field = S("--temp") ? "temperature" : S("--top-p") ? "top_p" : "min_p";
            double d = 0;
            if (!sampler_num(a.c_str(), field, v, &d)) return RAD_E_INVAL;
            (S("--temp") ? c->sample_temp : S("--top-p") ? c->sample_top_p : c->sample_min_p) = (float)d;
        }
        else if (S("--top-k"))                 { if (!(v = need(i))) return RAD_E_INVAL;
                                                 if (!sampler_int("--top-k", "top_k", v, &c->sample_top_k))
                                                     return RAD_E_INVAL; }
        else if (S("--typical-p") || S("--presence-penalty") || S("--frequency-penalty") ||
                 S("--repetition-penalty") || S("--dry-multiplier") || S("--dry-base") ||
                 S("--xtc-probability") || S("--xtc-threshold")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            struct F { const char* flag; const char* field; std::optional<float> Config::* m; };
            static const F kF[] = {
                { "--typical-p",          "typical_p",          &Config::sample_typical_p },
                { "--presence-penalty",   "presence_penalty",   &Config::sample_pres_penalty },
                { "--frequency-penalty",  "frequency_penalty",  &Config::sample_freq_penalty },
                { "--repetition-penalty", "repetition_penalty", &Config::sample_rep_penalty },
                { "--dry-multiplier",     "dry_multiplier",     &Config::sample_dry_multiplier },
                { "--dry-base",           "dry_base",           &Config::sample_dry_base },
                { "--xtc-probability",    "xtc_probability",    &Config::sample_xtc_probability },
                { "--xtc-threshold",      "xtc_threshold",      &Config::sample_xtc_threshold },
            };
            for (const F& f : kF) {
                if (!S(f.flag)) continue;
                double d = 0;
                if (!sampler_num(f.flag, f.field, v, &d)) return RAD_E_INVAL;
                c->*f.m = (float)d;
            }
        }
        else if (S("--repeat-last-n") || S("--dry-allowed-length") || S("--dry-penalty-last-n")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            const char* field = S("--repeat-last-n") ? "repeat_last_n"
                              : S("--dry-allowed-length") ? "dry_allowed_length"
                                                          : "dry_penalty_last_n";
            int n = 0;
            if (!sampler_int(a.c_str(), field, v, &n)) return RAD_E_INVAL;
            /* A DRY request with an allowed length below 1 is refused at admission, so such a
             * default would refuse every request that turns DRY on without stating one. */
            if (S("--dry-allowed-length") && n < 1) {
                RAD_ERR("--dry-allowed-length must be >= 1, not %d", n);
                return RAD_E_INVAL;
            }
            (S("--repeat-last-n") ? c->sample_penalty_last_n
             : S("--dry-allowed-length") ? c->sample_dry_allowed_length
                                         : c->sample_dry_penalty_last_n) = n;
        }
        else if (S("--dry-sequence-breakers")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            const nlohmann::json j = nlohmann::json::parse(v, nullptr, false);
            bool strings = j.is_array();
            if (strings) for (const auto& e : j) strings = strings && e.is_string();
            if (!strings) {
                RAD_ERR("--dry-sequence-breakers takes a JSON array of strings, such as "
                        "'[\"\\n\", \":\"]', not '%s'", v);
                return RAD_E_INVAL;
            }
            /* The sampler refuses more (sample/host_ref.h), so the default must not exceed it. */
            if (j.size() > kDryMaxBreakers) {
                RAD_ERR("--dry-sequence-breakers holds %zu strings; the sampler takes at most %zu",
                        j.size(), kDryMaxBreakers);
                return RAD_E_INVAL;
            }
            c->sample_dry_seq_breakers = j.get<std::vector<std::string>>();
        }
        else if (S("--default-max-tokens")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (!std::strcmp(v, "auto")) c->default_max_tokens = 0;
            else if (!parse_int(a.c_str(), v, 1, INT32_MAX, &c->default_max_tokens)) {
                RAD_ERR("--default-max-tokens takes a token count >= 1, or auto");
                return RAD_E_INVAL;
            }
        }
        else if (S("--max-tokens-cap"))        { if (!num(0, INT32_MAX, &c->max_tokens_cap)) return RAD_E_INVAL; }
        else if (S("--max-n"))                 { if (!num(1, INT32_MAX, &c->max_n)) return RAD_E_INVAL; }
        else if (S("--max-queued-requests"))   { if (!num(1, INT32_MAX, &c->max_queued_requests)) return RAD_E_INVAL; }
        else if (S("--max-stop-strings"))      { if (!num(1, INT32_MAX, &c->max_stop_strings)) return RAD_E_INVAL; }
        else if (S("--max-stop-bytes"))        { if (!num(1, INT32_MAX, &c->max_stop_bytes)) return RAD_E_INVAL; }
        else if (S("--chat-template-kwargs"))  { if (!(v = need(i)) ||
                                                     !parse_template_kwargs(v, &c->chat_template_kwargs))
                                                     return RAD_E_INVAL; }
        else if (S("--reasoning-format")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            /* The reply reader only distinguishes lifting reasoning out of content or not. */
            if (std::strcmp(v, "auto") && std::strcmp(v, "none")) {
                RAD_ERR("--reasoning-format takes auto or none, not '%s'", v);
                return RAD_E_INVAL;
            }
            c->reasoning_format = v;
        }
        else if (S("--http-threads") || S("--read-timeout") || S("--write-timeout") ||
                 S("--keep-alive-timeout") || S("--retry-after")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            /* The transport keeps at least four workers (server/http.cpp), so fewer is refused. */
            const int64_t lo = S("--http-threads") ? 4 : S("--retry-after") ? 0 : 1;
            int64_t n = 0;
            if (!parse_int(a.c_str(), v, lo, INT32_MAX, &n)) return RAD_E_INVAL;
            (S("--http-threads")    ? c->http_threads
             : S("--read-timeout")  ? c->http_read_timeout_s
             : S("--write-timeout") ? c->http_write_timeout_s
             : S("--retry-after")   ? c->retry_after_s
                                    : c->http_keep_alive_timeout_s) = (int)n;
        }
        else if (S("--max-body-mib"))          { if (!num(1, (int64_t)1 << 20, &c->http_max_body_mib)) return RAD_E_INVAL; }
        else if (S("--no-cors"))               c->cors = false;
        else if (S("--image-min-pixels") || S("--image-max-pixels") || S("--video-min-pixels") ||
                 S("--video-max-pixels") || S("--max-source-pixels")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            int64_t n = 0;
            if (!parse_int(a.c_str(), v, 1, INT64_MAX, &n)) return RAD_E_INVAL;
            (S("--image-min-pixels")   ? c->image_min_pixels
             : S("--image-max-pixels") ? c->image_max_pixels
             : S("--video-min-pixels") ? c->video_min_pixels
             : S("--video-max-pixels") ? c->video_max_pixels
                                       : c->max_source_pixels) = n;
        }
        else if (S("--video-fps")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            double d = 0;
            if (!parse_num(a.c_str(), v, &d)) return RAD_E_INVAL;
            if (!(d > 0.0)) { RAD_ERR("--video-fps must be > 0, not '%s'", v); return RAD_E_INVAL; }
            c->video_fps = d;
        }
        else if (S("--image-tokens")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            int64_t n = 0;
            if (!parse_int(a.c_str(), v, 1, INT32_MAX, &n)) return RAD_E_INVAL;
            c->image_tokens = (int32_t)n;
        }
        else if (S("--video-min-frames") || S("--video-max-frames") || S("--video-max-frame-tokens")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            int64_t n = 0;
            if (!parse_int(a.c_str(), v, S("--video-max-frame-tokens") ? 0 : 1, INT32_MAX, &n))
                return RAD_E_INVAL;
            (S("--video-min-frames")   ? c->video_min_frames
             : S("--video-max-frames") ? c->video_max_frames
                                       : c->video_max_frame_tokens) = (int32_t)n;
        }
        else if (S("--kld-record"))            { if (!(v = need(i))) return RAD_E_INVAL; c->kld_record = v; }
        else if (S("--kld-ref"))               { if (!(v = need(i))) return RAD_E_INVAL; c->kld_ref = v; }
        else if (S("--kld-corpus"))            { if (!(v = need(i))) return RAD_E_INVAL; c->kld_corpus = v; }
        else if (S("--kld-top"))               { if (!num(0, kKldTopMax, &c->kld_top)) return RAD_E_INVAL; }
        else if (S("--calib-corpus"))          { if (!(v = need(i))) return RAD_E_INVAL; c->calib_corpus = v; }
        else if (S("--calib-dir"))             { if (!(v = need(i))) return RAD_E_INVAL; c->calib_dir = v; }
        else if (S("--calib-layers")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            int lo = -1, hi = -1, used = 0;
            if (std::sscanf(v, "%d:%d%n", &lo, &hi, &used) != 2 || v[used] != '\0' || lo < 0 || hi <= lo) {
                RAD_ERR("--calib-layers takes LO:HI with 0 <= LO < HI, not '%s'", v);
                return RAD_E_INVAL;
            }
            c->calib_layer_lo = lo;
            c->calib_layer_hi = hi;
        }
        else if (S("--kld-out"))               { if (!(v = need(i))) return RAD_E_INVAL; c->kld_out = v; }
        else if (S("--kld-lora"))              { if (!(v = need(i))) return RAD_E_INVAL; c->kld_lora = v; }
        else if (S("--kld-lora-sources"))      { if (!(v = need(i))) return RAD_E_INVAL; c->kld_lora_sources = v; }
        else if (S("--debug-graph"))           { c->debug_graph = true; c->debug_selection = true; }
        else if (S("--debug-selection"))       c->debug_selection = true;
        else if (S("--debug-placement"))       c->debug_placement = true;
        else if (S("--debug-draft-oracle"))    { if (!(v = need(i))) return RAD_E_INVAL; c->debug_draft_oracle = v; }
        else if (S("--debug-accept-reference-kernels")) c->accept_reference_kernels = true;
        else if (S("--proceed-with-incompatible-configuration")) c->proceed_incompatible = true;
        else if (S("--profile-ops"))           c->profile_ops = true;
        else if (S("--expert-stats"))          { if (!(v = need(i))) return RAD_E_INVAL; c->expert_stats = v; }
        else if (S("--expert-profile"))        { if (!(v = need(i))) return RAD_E_INVAL; c->expert_profile = v; c->expert_profile_given = true; }
        else if (S("--expert-pin-top"))        {
            if (!(v = need(i))) return RAD_E_INVAL;
            char* e = nullptr;
            const double d = strtod(v, &e);
            if (e == v || *e != '\0' || d < 0.0 || d > 1.0) {
                RAD_ERR("--expert-pin-top: '%s' is not a fraction in [0, 1]", v);
                return RAD_E_INVAL;
            }
            c->expert_pin_top = (float)d;
            c->expert_pin_top_given = true;
        }
        else if (S("--draft-expert-placement")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (std::strcmp(v, "auto") != 0 && std::strcmp(v, "host") != 0) {
                RAD_ERR("--draft-expert-placement: '%s' is not auto or host", v);
                return RAD_E_INVAL;
            }
            c->draft_expert_placement = v;
            c->draft_expert_placement_given = true;
        }
        else if (S("--live"))                  c->live_view = true;
        else {
            RAD_ERR("unknown option '%s'", a.c_str());
            return RAD_E_INVAL;
        }
        /* Record the option's source for the startup report (core/diag/diag.h), under the
         * table's spelling so aliases report as their option. */
        c->given[S("--gpu-headroom") ? "--gpu-headroom-mib" : a] = "command line";
    }

    /* Both variables are read as the tools read them (rad_home() and rad_hierarchy_from_env() in
     * tools/iface.cpp), so a plugin validated with rad-kbench or rad-tune serves the same way. Set
     * but empty is unset, as for RADIANCE_KERNELS: an empty home would load no plugin at all. */
    if (c->radiance_home.empty()) {
        const char* h = getenv("RADIANCE_HOME");
        const bool set = h && *h;
        c->radiance_home = set ? h : RAD_DEFAULT_HOME;
        if (set) c->given["--radiance-home"] = "environment (RADIANCE_HOME)";
    }
    if (c->kernel_hierarchy.empty()) {
        const char* e = getenv("RADIANCE_KERNELS");
        if (e && *e) {
            c->kernel_hierarchy = rad_split_kernel_list(e);
            c->given["--kernels"] = "environment (RADIANCE_KERNELS)";
        }
    }
    /* No built-in default, deliberately: a named library would be wrong on other cards and would
     * rank the installed one behind the reference implementation. An empty hierarchy loads every
     * library found, ordered with the reference last (core/plugin/loader.cpp). A list loads only
     * the named libraries, in its order. */
    if (c->model.empty()) {
        RAD_ERR("--model is required");
        return RAD_E_INVAL;
    }

    /* Prompt lookup alone: the window is its K, stated once. */
    if (c->spec_ngram_min > c->spec_ngram_max) {
        RAD_ERR("--spec-ngram-min %d is longer than --spec-ngram-max %d", c->spec_ngram_min,
                c->spec_ngram_max);
        return RAD_E_INVAL;
    }
    if (c->spec_ngram > 0) {
        if (c->given.count("--num-speculative-tokens") && c->n_spec != c->spec_ngram) {
            RAD_ERR("--spec-ngram %d sets the speculative window to %d, and "
                    "--num-speculative-tokens asks for %s; give one", c->spec_ngram,
                    c->spec_ngram, c->n_spec < 0 ? "auto" : std::to_string(c->n_spec).c_str());
            return RAD_E_INVAL;
        }
        c->n_spec = c->spec_ngram;
    }

    /* KL mode: exactly one direction, each with the inputs it reads. */
    if (!c->kld_record.empty() && !c->kld_ref.empty()) {
        RAD_ERR("--kld-record writes a reference and --kld-ref scores against one; give one");
        return RAD_E_INVAL;
    }
    if (!c->kld_record.empty() && c->kld_corpus.empty()) {
        RAD_ERR("--kld-record needs --kld-corpus: the reference is recorded over a corpus");
        return RAD_E_INVAL;
    }
    if (!c->kld_ref.empty() && !c->kld_corpus.empty()) {
        RAD_ERR("--kld-ref reads its tokens from the reference, which recorded them; drop "
                "--kld-corpus");
        return RAD_E_INVAL;
    }
    if (c->calib() && c->calib_dir.empty()) {
        RAD_ERR("--calib-corpus records into --calib-dir; give one");
        return RAD_E_INVAL;
    }
    if ((!c->calib_dir.empty() || c->calib_layer_hi) && !c->calib() && c->kld_record.empty()) {
        RAD_ERR("--calib-dir and --calib-layers belong to a --calib-corpus or --kld-record run");
        return RAD_E_INVAL;
    }
    if (c->calib_layer_hi && c->calib_dir.empty()) {
        RAD_ERR("--calib-layers chooses what --calib-dir records; give one");
        return RAD_E_INVAL;
    }
    if (c->calib() && c->kld()) {
        RAD_ERR("--calib-corpus and the KL mode each replace the server; give one");
        return RAD_E_INVAL;
    }
    if (c->kld_top && c->kld_record.empty()) {
        RAD_ERR("--kld-top is how --kld-record stores a reference; --kld-ref reads the form from it");
        return RAD_E_INVAL;
    }
    if (!c->kld_out.empty() && c->kld_ref.empty()) {
        RAD_ERR("--kld-out is the --kld-ref report");
        return RAD_E_INVAL;
    }

    /* Runtime LoRA: the slot flags need slots, the named adapters must fit them, and KL mode's
     * adapter is one of them. */
    {
        const LoraConfig& L = c->lora;
        std::set<std::string> names;
        for (const auto& [name, path] : L.adapters)
            if (!names.insert(name).second) {
                RAD_ERR("--lora names the adapter '%s' twice", name.c_str());
                return RAD_E_INVAL;
            }
        const bool rank_given = c->given.count("--max-lora-rank") != 0;
        if (!L.enabled() && (rank_given || !L.targets.empty())) {
            RAD_ERR("%s sizes runtime LoRA slots, and there are none: give --lora or --max-loras",
                    rank_given ? "--max-lora-rank" : "--lora-target-modules");
            return RAD_E_INVAL;
        }
        if (!L.dir.empty()) {
            struct stat sb{};
            if (stat(L.dir.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)) {
                RAD_ERR("--lora-dir '%s' is not a directory", L.dir.c_str());
                return RAD_E_INVAL;
            }
        }
        if (!c->kld_lora.empty() && !c->kld()) {
            RAD_ERR("--kld-lora scores through an adapter in KL mode; give --kld-record or "
                    "--kld-ref");
            return RAD_E_INVAL;
        }
        if (!c->kld_lora.empty() && !names.count(c->kld_lora)) {
            RAD_ERR("--kld-lora '%s' is not one of the adapters --lora names", c->kld_lora.c_str());
            return RAD_E_INVAL;
        }
        if (!c->kld_lora_sources.empty() && c->kld_lora.empty()) {
            RAD_ERR("--kld-lora-sources picks the documents that run through --kld-lora's adapter; "
                    "give --kld-lora");
            return RAD_E_INVAL;
        }
    }

    /* --reasoning-effort is a template variable ("none" also sets enable_thinking false), so
     * either key in --chat-template-kwargs as well would give two defaults for one setting. */
    if (!c->reasoning_effort.empty() &&
        (c->chat_template_kwargs.count("reasoning_effort") ||
         c->chat_template_kwargs.count("enable_thinking"))) {
        RAD_ERR("--reasoning-effort and a reasoning_effort or enable_thinking key in "
                "--chat-template-kwargs both set the default for thinking; give one");
        return RAD_E_INVAL;
    }
    if (c->default_max_tokens > 0 && c->max_tokens_cap > 0 &&
        c->default_max_tokens > c->max_tokens_cap) {
        RAD_ERR("--default-max-tokens %lld is above --max-tokens-cap %lld, so no request could be "
                "served at the default", (long long)c->default_max_tokens,
                (long long)c->max_tokens_cap);
        return RAD_E_INVAL;
    }
    /* A request's choices must all fit the queue, else it is always 429. The defaults always fit,
     * so this is checked only when one was given. */
    if (c->max_n > 0 || c->max_queued_requests > 0) {
        const int64_t n = c->max_n > 0 ? c->max_n : std::max<int64_t>(8, c->max_seqs);
        const int64_t q = c->max_queued_requests > 0 ? c->max_queued_requests : c->max_seqs * 8;
        if (n > q) {
            RAD_ERR("--max-n %lld is above the %lld requests the queue admits "
                    "(--max-queued-requests), so a request asking for it could never be admitted",
                    (long long)n, (long long)q);
            return RAD_E_INVAL;
        }
    }
    if (c->image_min_pixels && c->image_max_pixels && *c->image_min_pixels > *c->image_max_pixels) {
        RAD_ERR("--image-min-pixels is above --image-max-pixels"); return RAD_E_INVAL; }
    if (c->video_min_pixels && c->video_max_pixels && *c->video_min_pixels > *c->video_max_pixels) {
        RAD_ERR("--video-min-pixels is above --video-max-pixels"); return RAD_E_INVAL; }
    if (c->video_min_frames && c->video_max_frames && *c->video_min_frames > *c->video_max_frames) {
        RAD_ERR("--video-min-frames is above --video-max-frames"); return RAD_E_INVAL; }
    /* Two answers to one question: KV streaming (--kv-resident, docs/KV-STREAMING.md) keeps the
     * sparse attention group in host memory with a ring on the card, window residency
     * (--kv-residency-mib, spec §7.3) keeps it on the card and demotes what the window does not
     * cover. Both rewrite the same group's selection and own its blocks' placement, so one server
     * runs one of them. */
    /* A recipe sets each layer's layout within the E4M3 cache (RadBuildCtx::kv_recipe), its `bf16`
     * layers (RAD_KVQ_BF16) the one exception it names per layer, and a drafter's cache stays
     * E4M3; under bf16 or one of the narrow names the two flags would say two things about the
     * same layers. */
    if (!c->kv_recipe.empty() && c->kv_cache_dtype != "fp8") {
        RAD_ERR("--kv-recipe %s sets each attention layer's layout within the fp8 cache, and "
                "--kv-cache-dtype is %s; pass --kv-cache-dtype fp8", c->kv_recipe_path.c_str(),
                c->kv_cache_dtype.c_str());
        return RAD_E_INVAL;
    }
    if (c->kv_resident > 0 && c->residency_mib > 0) {
        RAD_ERR("--kv-resident and --kv-residency-mib are two ways to keep a sparse attention "
                "group's KV off the card -- streaming from host memory with a ring on the card, or "
                "the cache on the card with its unread middle demoted -- and a server runs one of "
                "them. Pass only one");
        return RAD_E_INVAL;
    }

    return RAD_OK;
}

const char* attn_prefill_fp8_name(int bits) {
    switch (bits) {
        case 0:                                   return "off";
        case RAD_ATTN_FP8_QK:                     return "qk";
        case RAD_ATTN_FP8_PV:                     return "pv";
        case RAD_ATTN_FP8_QK | RAD_ATTN_FP8_PV:  return "both";
    }
    return "?";
}

/* ------------------------------------------------------------------ the startup report's rows */
namespace {

std::string onoff(bool b) { return b ? "on" : "off"; }
std::string num(int64_t v) { return std::to_string(v); }
std::string or_none(const std::string& s, const char* none) { return s.empty() ? std::string(none) : s; }
std::string fnum(double v) {
    char b[48];
    snprintf(b, sizeof b, "%g", v);
    return b;
}
template <class T>
std::string opt(const std::optional<T>& v, const char* unset) {
    if (!v) return unset;
    if constexpr (std::is_floating_point_v<T>) return fnum((double)*v);
    else return std::to_string(*v);
}
std::string joined(const std::map<std::string, std::string>& m, const char* sep) {
    std::string s;
    for (const auto& [k, v] : m) s += (s.empty() ? "" : ",") + k + sep + v;
    return s.empty() ? "none" : s;
}

using Show = std::string (*)(const Config&);
struct ShowRow { const char* flag; Show show; };

/* How each option's value is shown, one entry per option of the table above. config_rows walks
 * the table, so a missing entry shows "(not shown)" and fails diag_test. Sentinels for "resolved
 * at startup" (-1, 0, empty) are shown by meaning; the engine's decisions replace them. */
const ShowRow kShow[] = {
    { "--model",              [](const Config& c) { return c.model; } },
    { "--download-dir",       [](const Config& c) { return or_none(c.download_dir, "the XDG cache (~/.cache/radiance/hf)"); } },
    { "--model-revision",     [](const Config& c) { return or_none(c.model_revision, "the repo's main branch"); } },
    { "--radiance-home",      [](const Config& c) { return c.radiance_home; } },
    { "--kernels",            [](const Config& c) {
          std::string s;
          for (const std::string& k : c.kernel_hierarchy) s += (s.empty() ? "" : ",") + k;
          return or_none(s, "every library found"); } },
    { "--tp",                 [](const Config& c) { return num(c.tp); } },
    { "--tp-wire",            [](const Config& c) { return std::string(c.tp_wire_exact ? "exact" : "wht6"); } },
    { "--tp-attn-split",      [](const Config& c) { return std::string(c.tp_attn_rows ? "rows" : "heads"); } },
    { "--weight-prefetch",    [](const Config& c) { return onoff(c.weight_prefetch); } },
    { "--tp-wire-min-kb",     [](const Config& c) { return num(c.tp_wire_min_kb); } },
    { "--p2p",                [](const Config& c) {
          return std::string(c.p2p == RAD_P2P_ON ? "on" : c.p2p == RAD_P2P_OFF ? "off" : "auto"); } },
    { "--max-num-batched-tokens", [](const Config& c) { return num(c.max_tok); } },
    { "--busy-prefill-chunk", [](const Config& c) {
          return c.busy_prefill_chunk < 0 ? std::string("auto")
               : c.busy_prefill_chunk == 0 ? std::string("off") : num(c.busy_prefill_chunk); } },
    { "--prefill-interleave", [](const Config& c) {
          return c.prefill_interleave > 0 ? num(c.prefill_interleave) : std::string("off"); } },
    { "--preempt-wait-ms",    [](const Config& c) { return num(c.preempt_wait_ms); } },
    { "--max-num-seqs",       [](const Config& c) { return num(c.max_seqs); } },
    { "--max-model-len",      [](const Config& c) {
          return c.max_ctx ? num(c.max_ctx) : std::string("0 (the model's training context)"); } },
    { "--rope-scaling",       [](const Config& c) { return or_none(c.rope_scaling, "the checkpoint's own"); } },
    { "--gpu-headroom-mib",   [](const Config& c) { return num(c.gpu_headroom_mib); } },
    { "--kv-cache-dtype",     [](const Config& c) { return c.kv_cache_dtype; } },
    { "--kv-recipe",          [](const Config& c) { return or_none(c.kv_recipe_path, "none"); } },
    { "--state-cache-dtype",  [](const Config& c) { return c.state_cache_dtype; } },
    { "--qsa-bkey-dtype",     [](const Config& c) { return c.qsa_bkey_dtype; } },
    { "--qsa-bkey-split",     [](const Config& c) {
          return std::string(c.qsa_bkey_split ? "on" : "off"); } },
    { "--attn-prefill-fp8",   [](const Config& c) {
          return std::string(attn_prefill_fp8_name(c.attn_prefill_fp8)); } },
    { "--kv-resident",        [](const Config& c) {
          return c.kv_resident > 0 ? num(c.kv_resident) : std::string("0 (off)"); } },
    { "--host-pool-mib",      [](const Config& c) { return num(c.host_pool_mib); } },
    { "--host-headroom-mib",  [](const Config& c) { return num(c.host_headroom_mib); } },
    { "--embedding-placement", [](const Config& c) { return std::string(table_placement_name(c.embedding_placement)); } },
    { "--ngram-placement",    [](const Config& c) {
          std::string v = table_placement_name(c.ngram_placement);
          return c.ngram_share_dir.empty() ? v : v + "=" + c.ngram_share_dir; } },
    { "--vision-tower-placement", [](const Config& c) { return std::string(table_placement_name(c.tower_placement)); } },
    { "--weights-disk-tier",  [](const Config& c) { return onoff(c.weights_disk_tier); } },
    { "--prefix-cache-mib",   [](const Config& c) {
          return c.prefix_cache_mib ? num(c.prefix_cache_mib) : std::string("0 (off)"); } },
    { "--prefix-cache-ttl",   [](const Config& c) {
          return c.prefix_cache_ttl ? num(c.prefix_cache_ttl) + " s" : std::string("0 (never)"); } },
    { "--kv-residency-mib",   [](const Config& c) {
          return c.residency_mib ? num(c.residency_mib) : std::string("0 (off)"); } },
    { "--kv-residency-sink-tokens", [](const Config& c) { return num(c.residency_sink_tokens); } },
    { "--lora",               [](const Config& c) {
          std::string o;
          for (const auto& [n, p] : c.lora.adapters) o += (o.empty() ? "" : ", ") + n + "=" + p;
          return o.empty() ? std::string("none") : o; } },
    { "--lora-dir",           [](const Config& c) {
          return c.lora.dir.empty() ? std::string("none") : c.lora.dir; } },
    { "--max-loras",          [](const Config& c) {
          return c.lora.enabled() ? num(c.lora.slots()) : std::string("0 (off)"); } },
    { "--max-lora-rank",      [](const Config& c) {
          return c.lora.max_rank > 0 ? num(c.lora.max_rank) : std::string("auto (16 text, 64 generator)"); } },
    { "--lora-target-modules", [](const Config& c) {
          std::string o;
          for (const std::string& t : c.lora.targets) o += (o.empty() ? "" : ",") + t;
          return o.empty() ? std::string(c.lora.enabled() ? "the adapters' own" : "-") : o; } },
    { "--placement",          [](const Config& c) { return c.placement; } },
    { "--control-vector",     [](const Config& c) { return c.control_vector; } },
    { "--deterministic",      [](const Config& c) { return onoff(c.deterministic); } },
    { "--checkpoint-interval", [](const Config& c) { return num(c.checkpoint_interval); } },
    { "--prefill-span",       [](const Config& c) { return num(c.prefill_span); } },
    { "--checkpoint-slots",   [](const Config& c) {
          return c.checkpoint_slots < 0 ? std::string("auto") : num(c.checkpoint_slots); } },
    { "--no-prefix-cache",    [](const Config& c) {
          return std::string(c.prefix_cache ? "not given: prefix caching on" : "prefix caching off"); } },
    { "--prefix-resume",      [](const Config& c) {
          return std::string(c.prefix_exact ? "exact" : "nearest"); } },
    { "--checkpoint-policy",  [](const Config& c) { return c.checkpoint_policy; } },
    { "--num-speculative-tokens", [](const Config& c) { return c.n_spec < 0 ? std::string("auto") : num(c.n_spec); } },
    { "--spec-ngram",         [](const Config& c) {
          return c.spec_ngram > 0 ? num(c.spec_ngram) : std::string("0 (off)"); } },
    { "--spec-ngram-min",     [](const Config& c) { return num(c.spec_ngram_min); } },
    { "--spec-ngram-max",     [](const Config& c) { return num(c.spec_ngram_max); } },
    { "--draft-window",       [](const Config& c) { return std::string(draft_window_name(c.draft_window)); } },
    { "--draft-short-from",   [](const Config& c) {
          return c.draft_short_from < 0 ? std::string("auto")
               : c.draft_short_from > 0 ? num(c.draft_short_from) : std::string("0 (never)"); } },
    { "--draft-coupling",     [](const Config& c) { return onoff(c.draft_coupling); } },
    { "--draft-model",        [](const Config&) { return std::string("not implemented (the container's own drafter)"); } },
    { "--host",               [](const Config& c) { return c.host; } },
    { "--port",               [](const Config& c) { return num(c.port); } },
    /* Never print the secret, only whether it is set. */
    { "--api-key",            [](const Config& c) { return std::string(c.api_key.empty() ? "not set" : "set"); } },
    { "--api-key-env",        [](const Config& c) { return or_none(c.api_key_env, "not used"); } },
    { "--allow-host",         [](const Config& c) {
          std::string o;
          for (const std::string& h : c.allow_hosts) o += (o.empty() ? "" : ",") + h;
          return o.empty() ? std::string("the loopback names only") : o; } },
    { "--trusted-proxy",      [](const Config& c) {
          std::string o;
          for (const std::string& p : c.trusted_proxies) o += (o.empty() ? "" : ",") + p;
          return o.empty() ? std::string("no socket vouches for a caller") : o; } },
    { "--served-model-name",  [](const Config& c) { return or_none(c.served_model_name, "the container's name"); } },
    { "--mm-max-patches",     [](const Config& c) { return c.mm_max_patches < 0 ? std::string("auto") : num(c.mm_max_patches); } },
    { "--image-gen-max-pixels", [](const Config& c) { return c.image_gen_max_pixels > 0 ? num(c.image_gen_max_pixels) : std::string("the model's"); } },
    { "--image-gen-max-refs", [](const Config& c) { return c.image_gen_max_refs > 0 ? num(c.image_gen_max_refs) : std::string("the model's"); } },
    { "--generation-config",  [](const Config& c) { return or_none(c.generation_config, "none"); } },
    { "--override-chat-template", [](const Config& c) { return or_none(c.override_chat_template, "none (the container's)"); } },
    { "--temp",               [](const Config& c) { return c.sample_temp < 0 ? std::string("unset") : fnum(c.sample_temp); } },
    { "--top-k",              [](const Config& c) { return c.sample_top_k < 0 ? std::string("unset") : num(c.sample_top_k); } },
    { "--top-p",              [](const Config& c) { return c.sample_top_p < 0 ? std::string("unset") : fnum(c.sample_top_p); } },
    { "--min-p",              [](const Config& c) { return c.sample_min_p < 0 ? std::string("unset") : fnum(c.sample_min_p); } },
    { "--reasoning-effort",   [](const Config& c) { return or_none(c.reasoning_effort, "the template's default"); } },
    { "--reasoning-effort-map", [](const Config& c) { return joined(c.reasoning_effort_map, "="); } },
    { "--thinking-token-budget", [](const Config& c) {
          return c.thinking_token_budget < 0 ? std::string("none") : num(c.thinking_token_budget); } },
    { "--typical-p",          [](const Config& c) { return opt(c.sample_typical_p, "the request default"); } },
    { "--presence-penalty",   [](const Config& c) { return opt(c.sample_pres_penalty, "the request default"); } },
    { "--frequency-penalty",  [](const Config& c) { return opt(c.sample_freq_penalty, "the request default"); } },
    { "--repetition-penalty", [](const Config& c) { return opt(c.sample_rep_penalty, "the request default"); } },
    { "--repeat-last-n",      [](const Config& c) { return opt(c.sample_penalty_last_n, "the request default"); } },
    { "--dry-multiplier",     [](const Config& c) { return opt(c.sample_dry_multiplier, "the request default"); } },
    { "--dry-base",           [](const Config& c) { return opt(c.sample_dry_base, "the request default"); } },
    { "--dry-allowed-length", [](const Config& c) { return opt(c.sample_dry_allowed_length, "the request default"); } },
    { "--dry-penalty-last-n", [](const Config& c) { return opt(c.sample_dry_penalty_last_n, "the request default"); } },
    { "--dry-sequence-breakers", [](const Config& c) {
          /* Invalid UTF-8 is replaced, not thrown on: these strings come from outside. */
          return c.sample_dry_seq_breakers
                     ? nlohmann::json(*c.sample_dry_seq_breakers)
                           .dump(-1, ' ', false, nlohmann::json::error_handler_t::replace)
                     : std::string("the request default"); } },
    { "--xtc-probability",    [](const Config& c) { return opt(c.sample_xtc_probability, "the request default"); } },
    { "--xtc-threshold",      [](const Config& c) { return opt(c.sample_xtc_threshold, "the request default"); } },
    { "--default-max-tokens", [](const Config& c) {
          return c.default_max_tokens > 0 ? num(c.default_max_tokens)
                                          : std::string("auto (what the context leaves after the prompt)"); } },
    { "--max-tokens-cap",     [](const Config& c) { return c.max_tokens_cap > 0 ? num(c.max_tokens_cap) : std::string("none"); } },
    { "--max-n",              [](const Config& c) {
          return c.max_n > 0 ? num(c.max_n)
                             : std::string("auto (the larger of 8 and --max-num-seqs)"); } },
    { "--max-queued-requests", [](const Config& c) {
          return c.max_queued_requests > 0 ? num(c.max_queued_requests)
                                           : std::string("auto (8 x --max-num-seqs)"); } },
    { "--max-stop-strings",   [](const Config& c) { return num(c.max_stop_strings); } },
    { "--max-stop-bytes",     [](const Config& c) { return num(c.max_stop_bytes); } },
    { "--chat-template-kwargs", [](const Config& c) { return joined(c.chat_template_kwargs, "="); } },
    { "--reasoning-format",   [](const Config& c) { return c.reasoning_format; } },
    { "--http-threads",       [](const Config& c) {
          return c.http_threads > 0 ? num(c.http_threads)
                                    : std::string("auto (one a hardware thread, at least 4)"); } },
    { "--read-timeout",       [](const Config& c) { return num(c.http_read_timeout_s) + " s"; } },
    { "--write-timeout",      [](const Config& c) { return num(c.http_write_timeout_s) + " s"; } },
    { "--keep-alive-timeout", [](const Config& c) { return num(c.http_keep_alive_timeout_s) + " s"; } },
    { "--max-body-mib",       [](const Config& c) { return num(c.http_max_body_mib); } },
    { "--retry-after",        [](const Config& c) { return num(c.retry_after_s) + " s"; } },
    { "--no-cors",            [](const Config& c) {
          return std::string(c.cors ? "not given: CORS headers sent" : "no CORS headers"); } },
    { "--image-min-pixels",   [](const Config& c) { return opt(c.image_min_pixels, "the container's"); } },
    { "--image-max-pixels",   [](const Config& c) { return opt(c.image_max_pixels, "the container's"); } },
    { "--video-min-pixels",   [](const Config& c) { return opt(c.video_min_pixels, "the container's"); } },
    { "--video-max-pixels",   [](const Config& c) { return opt(c.video_max_pixels, "the container's"); } },
    { "--video-fps",          [](const Config& c) { return opt(c.video_fps, "the container's"); } },
    { "--video-min-frames",   [](const Config& c) { return opt(c.video_min_frames, "the container's"); } },
    { "--video-max-frames",   [](const Config& c) { return opt(c.video_max_frames, "the container's"); } },
    { "--video-max-frame-tokens", [](const Config& c) { return opt(c.video_max_frame_tokens, "the container's"); } },
    { "--image-tokens",       [](const Config& c) { return opt(c.image_tokens, "the container's"); } },
    { "--max-source-pixels",  [](const Config& c) { return opt(c.max_source_pixels, "67108864"); } },
    { "--kld-record",         [](const Config& c) { return or_none(c.kld_record, "none"); } },
    { "--kld-ref",            [](const Config& c) { return or_none(c.kld_ref, "none"); } },
    { "--kld-corpus",         [](const Config& c) { return or_none(c.kld_corpus, "none"); } },
    { "--kld-top",            [](const Config& c) { return c.kld_top ? num(c.kld_top) : std::string("0 (the whole vocabulary)"); } },
    { "--calib-corpus",       [](const Config& c) { return or_none(c.calib_corpus, "none"); } },
    { "--calib-dir",          [](const Config& c) { return or_none(c.calib_dir, "none"); } },
    { "--calib-layers",       [](const Config& c) { return c.calib_layer_hi ? num(c.calib_layer_lo) + ":" + num(c.calib_layer_hi) : std::string("every layer"); } },
    { "--kld-out",            [](const Config& c) { return or_none(c.kld_out, "none"); } },
    { "--kld-lora",           [](const Config& c) { return or_none(c.kld_lora, "none (the base)"); } },
    { "--kld-lora-sources",   [](const Config& c) { return or_none(c.kld_lora_sources, "every document"); } },
    { "--debug-graph",        [](const Config& c) { return onoff(c.debug_graph); } },
    { "--debug-selection",    [](const Config& c) { return onoff(c.debug_selection); } },
    { "--debug-placement",    [](const Config& c) { return onoff(c.debug_placement); } },
    { "--debug-draft-oracle", [](const Config& c) { return or_none(c.debug_draft_oracle, "none"); } },
    { "--debug-accept-reference-kernels", [](const Config& c) { return onoff(c.accept_reference_kernels); } },
    { "--proceed-with-incompatible-configuration", [](const Config& c) { return onoff(c.proceed_incompatible); } },
    { "--profile-ops",        [](const Config& c) { return onoff(c.profile_ops); } },
    { "--expert-stats",       [](const Config& c) { return or_none(c.expert_stats, "off"); } },
    { "--expert-profile",     [](const Config& c) { return or_none(c.expert_profile, "off"); } },
    { "--expert-pin-top",     [](const Config& c) { return fmt("%.4g", c.expert_pin_top); } },
    { "--draft-expert-placement", [](const Config& c) { return c.draft_expert_placement; } },
    { "--live",               [](const Config& c) { return onoff(c.live_view); } },
    { "-v",                   [](const Config& c) { return onoff(c.given.count("-v") > 0); } },
    { "-vv",                  [](const Config& c) { return onoff(c.given.count("-vv") > 0); } },
};

}  /* namespace */

namespace diag {

std::vector<ConfigRow> config_rows(const Config& c) {
    std::vector<ConfigRow> out;
    for (const Flag& f : g_flags) {
        /* --help is not a setting. */
        if (!std::strcmp(f.name, "--help")) continue;
        ConfigRow r;
        r.flag = f.name;
        r.value = "(not shown)";
        for (const ShowRow& s : kShow)
            if (!std::strcmp(s.flag, f.name)) { r.value = s.show(c); break; }
        auto g = c.given.find(f.name);
        r.source = g != c.given.end() ? g->second : "default";
        out.push_back(std::move(r));
    }
    return out;
}

std::vector<std::string> config_option_names() {
    std::vector<std::string> out;
    for (const Flag& f : g_flags)
        if (std::strcmp(f.name, "--help")) out.push_back(f.name);
    return out;
}

}  /* namespace diag */

}  /* namespace rad */
