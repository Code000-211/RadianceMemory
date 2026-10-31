#!/usr/bin/env python3
"""IQ8R differential for Spark X2.6 on the host kernels.

Builds the deterministic two-layer fixture (tests/spark25_reference.py's, at a geometry the
group-128 rotated int8 grid takes), converts it twice -- as it stands, and on IQ8R's grid -- and
serves both through the CPU kernels. The quantised model must stay close to its bf16 self and to
the Transformers reference: activations fed unrotated into rotated weights, or a scale plane read
at the wrong width, yields a model that runs, generates fluently and is wrong.

Requires torch, transformers and safetensors. GPUs are hidden from both processes; on a shared
machine run it in a GPU-denying device cgroup (ROCm may open device nodes while probing).

  tests/spark25_iq8r.py --reference <XHToken-Spark-X2.6-4B> --build <build dir>
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark25_reference import (PROMPTS, generate, load_reference, serve, write_tokenizer)  # noqa: E402

# Rows a token and widths the group-128 grid takes: hidden and every contraction in whole 128s,
# output rows in whole 16s. head_dim 32 x 8 heads keeps q and the o_proj contraction at 256.
GEOM = dict(vocab_size=3584, hidden_size=256, intermediate_size=256, num_hidden_layers=2,
            num_attention_heads=8, num_key_value_heads=4, head_dim=32)
TRUNK = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate_up", "ffn_down")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, type=Path, help="directory with modeling_spark.py")
    ap.add_argument("--build", required=True, type=Path, help="Radiance build directory")
    ap.add_argument("--keep", type=Path, help="write the fixture and both containers here")
    ap.add_argument("--tp", type=int, choices=range(1, 9), default=1,
                    help="number of fake host ranks (up to 8)")
    ap.add_argument("--wide-tp", action="store_true",
                    help="use 128-wide heads and a 1024-wide FFN to exercise TP up to 8")
    args = ap.parse_args()
    if args.tp > 2 and not args.wide_tp:
        ap.error("--tp above 2 needs --wide-tp so every rank owns a whole scale group")
    import torch
    import transformers
    from safetensors.torch import save_file

    torch.set_num_threads(2)
    cfgmod, modelmod = load_reference(args.reference.resolve())
    if int(transformers.__version__.split(".")[0]) >= 5:
        modelmod.Spark2_5ForCausalLM._tied_weights_keys = {"lm_head.weight": "model.embedding.weight"}
    geom = dict(GEOM)
    if args.wide_tp:
        geom.update(intermediate_size=1024, head_dim=128)
    config = cfgmod.Spark2_5Config(
        **geom, hidden_act="gelu", max_position_embeddings=128, rms_norm_eps=1e-6,
        tie_word_embeddings=True, headwise_attn_output_gate=True, gate_attn_act_mode="sigmoid",
        sliding_window=8, layer_types=["sliding_attention", "full_attention"],
        rope_parameters={"sliding_attention": {"rope_theta": 10000.0, "partial_rotary_factor": 1.0},
                         "full_attention": {"rope_theta": 5000000.0, "partial_rotary_factor": 0.5}},
        bos_token_id=256, eos_token_id=256, pad_token_id=256)
    torch.manual_seed(19)
    model = modelmod.Spark2_5ForCausalLM(config).eval().to(torch.bfloat16)
    with torch.no_grad():
        for name, w in model.named_parameters():
            if name.endswith("layernorm.weight") or name == "model.norm.weight":
                w.copy_((1 + 0.08 * torch.randn(w.shape)).to(w.dtype))
            # Scale the measured 4B weight standard deviations by sqrt(2560 / 256), so
            # this narrower random fixture has comparable projection and head logit magnitudes.
            elif name == "model.embedding.weight":
                w.copy_((0.09 * torch.randn(w.shape)).to(w.dtype))
            elif name.endswith("g_proj.weight"):
                w.copy_((0.055 * torch.randn(w.shape)).to(w.dtype))
            else:
                w.copy_((0.085 * torch.randn(w.shape)).to(w.dtype))

    def reference(ids):
        n = len(ids)
        q, k = torch.arange(n)[:, None], torch.arange(n)[None, :]
        full = torch.where(k <= q, 0.0, float("-inf"))[None, None, :, :]
        local = torch.where((k <= q) & (k > q - config.sliding_window), 0.0,
                            float("-inf"))[None, None, :, :]
        with torch.inference_mode():
            logits = model(torch.tensor([ids]), attention_mask={"full_attention": full,
                           "sliding_attention": local}, use_cache=False).logits[0, -1].float()
        return torch.log_softmax(logits, dim=-1)

    build = args.build.resolve()
    env = os.environ.copy()
    env.update(RADIANCE_HOME=str(build / "radiance_home"), RADIANCE_HOST_VRAM_MIB="2048",
               RADIANCE_HOST_DEVICES=str(args.tp))
    keep = args.keep.resolve() if args.keep else None
    root = keep or Path(tempfile.mkdtemp(prefix="spark25-iq8r-"))
    root.mkdir(parents=True, exist_ok=True)
    ckpt = root / "checkpoint"
    ckpt.mkdir(exist_ok=True)
    config_json = config.to_dict()
    config_json.update(model_type="spark2_5", architectures=["Spark2_5ForCausalLM"])
    (ckpt / "config.json").write_text(json.dumps(config_json, indent=2))
    state = {k: v.contiguous().clone() for k, v in model.state_dict().items()
             if k != "lm_head.weight"}
    save_file(state, ckpt / "model.safetensors")
    write_tokenizer(ckpt)

    bf16, trunk, full = root / "model-bf16.rad", root / "model-trunk.rad", root / "model-iq8r.rad"
    trunk_recipe = root / "iq8r-trunk.recipe"
    trunk_recipe.write_text("".join(f"blk.*.{w}.weight gptq codes=i8 group=128 scale=bf16 "
                                    f"transform=fwht128 clamp=sym rule=search\n" for w in TRUNK))
    full_recipe = root / "iq8r.recipe"
    full_recipe.write_text(trunk_recipe.read_text() +
                           "output.weight rtn codes=i8 group=128 scale=bf16 clamp=sym "
                           "rule=search\n")
    sets = ["--set", "quantization_config.quant_method=w8a8", "--set",
            "quantization_config.fmt=fwht128"]
    for out, extra in ((bf16, []), (trunk, ["--recipe", str(trunk_recipe)] + sets),
                       (full, ["--recipe", str(full_recipe)] + sets)):
        subprocess.run([str(build / "bin/rad-convert"), str(ckpt), "-o", str(out),
                        "--max-ctx", "64", "--max-tok", "32"] + extra, env=env, check=True,
                       stdout=subprocess.DEVNULL)

    got = {}
    with (root / "server.log").open("w+") as log:
        for tag, container in (("bf16", bf16), ("trunk", trunk), ("iq8r", full)):
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            proc, url = serve(build, container, port, env, log, tp=args.tp)
            try:
                for prompt in PROMPTS:
                    got.setdefault(tag, {})[prompt] = generate(url, prompt)
            except Exception:
                log.flush()
                log.seek(0)
                print(log.read(), file=sys.stderr)
                raise
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

    # What a step's token costs against the Transformers reference: the same rule
    # tests/spark25_reference.py holds the bf16 model to, with the margin the int8 grid buys. This
    # fixture is a random model, so its candidates sit close together and a quantised run can flip
    # a greedy token; what it may not do is drift far, which is what a connection fed unrotated
    # codes or a scale plane read at the wrong width produces.
    # Bounds cover BF16/TP reduction rounding and 8-bit grid error at this geometry, while
    # rejecting a wrong rotation or scale plane. Compare probabilities as well as the sampled id:
    # an apparently correct argmax alone can hide a badly corrupted distribution.
    bounds = {"bf16": (0.1, 0.15), "trunk": (0.35, 0.4), "iq8r": (0.35, 0.4)}
    for prompt in PROMPTS:
        for tag, (bound, logp_bound) in bounds.items():
            ids, exact, gap, max_error = list(prompt.encode()), 0, 0.0, 0.0
            for row in got[tag][prompt]:
                expected = reference(ids)
                gap = max(gap, float(expected.max() - expected[row["id"]]))
                for candidate in row["top_logprobs"]:
                    max_error = max(max_error, abs(candidate["logprob"] - float(expected[candidate["id"]])))
                if row["id"] == int(expected.argmax()):
                    exact += 1
                ids.append(row["id"])
            print(f"{tag:5s} {prompt!r}: {exact}/5 greedy tokens match the reference, worst step "
                  f"{gap:.5f} nats behind the reference's best candidate, max top-10 logp error "
                  f"{max_error:.5f}")
            if gap > bound:
                raise AssertionError(f"{tag} {prompt!r}: a step is {gap:.5f} nats behind the "
                                     f"reference's best candidate (bound {bound})")
            if max_error > logp_bound:
                raise AssertionError(f"{tag} {prompt!r}: top-10 logp error {max_error:.5f} "
                                     f"exceeds {logp_bound}")
    print(f"TP={args.tp}: iq8r stays within the test's configured reference bounds")


if __name__ == "__main__":
    main()
