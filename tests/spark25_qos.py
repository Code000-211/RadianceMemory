#!/usr/bin/env python3
"""Repeatable real-container Spark latency/throughput smoke (end-to-end, not decode-only).

Run this separately for BF16, full IQ8R, and keep12 under an exclusive device scope. Each run
starts one server, disables speculation and prefix caching, and replays the same held-out prompt
strings at the requested concurrency levels. It records artifact hashes, p50/p95/max request
latency, and aggregate generated-token throughput; it does not claim task quality.
"""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid


from spark25_validation import plugin_hashes, sha256


def quantile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    at = q * (len(values) - 1)
    lo = int(at)
    return values[lo] if lo + 1 == len(values) else values[lo] + (at - lo) * (values[lo + 1] - values[lo])


def prompts_from(path, raw):
    prompts = []
    for line_no, line in enumerate(raw.decode('utf-8').splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"{path}:{line_no}: expected a nonempty string prompt")
        prompts.append({"source": row.get("source", ""), "prompt": row["prompt"]})
    if not prompts:
        raise ValueError(f"{path}: no prompts")
    return prompts


def request(url, model, prompt, max_tokens, timeout):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0, "ignore_eos": True}
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.load(response)
    elapsed = time.perf_counter() - start
    usage = data.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("response is missing usage")
    prompt_tokens, completed = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if type(prompt_tokens) is not int or prompt_tokens <= 0:
        raise ValueError("response must report a positive integer prompt_tokens count")
    if type(completed) is not int or completed != max_tokens:
        raise ValueError(f"response must report exactly {max_tokens} completion tokens, got {completed!r}")
    choices = data.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") != "length":
        raise ValueError("forced-length QoS response did not finish at its requested length")
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("request duration is not a positive finite measurement")
    return {"seconds": elapsed, "prompt_tokens": prompt_tokens,
            "completion_tokens": completed, "finish_reason": "length"}


def verify_server(url, served, proc):
    if proc.poll() is not None:
        raise RuntimeError("candidate server exited during measurement")
    with urllib.request.urlopen(url + "/v1/models", timeout=5) as response:
        models = json.load(response)
    if served not in {item.get("id") for item in models.get("data", [])} or proc.poll() is not None:
        raise RuntimeError("private server identity does not match the launched candidate")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--radiance", required=True, type=Path)
    p.add_argument("--home", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--prompts", required=True, type=Path, help="JSONL with prompt and optional source")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--kernels", default="libr4d")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--concurrency", default="1,4,8", help="comma-separated sequence counts")
    p.add_argument("--requests", type=int, default=16, help="requests per concurrency/repetition")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--max-num-batched-tokens", type=int, default=4096)
    p.add_argument("--gpu-headroom-mib", type=int, default=1024)
    p.add_argument("--timeout", type=float, default=600)
    args = p.parse_args()

    radiance, home, model = args.radiance.resolve(), args.home.resolve(), args.model.resolve()
    prompt_path, out = args.prompts.resolve(), args.out.resolve()
    prompt_bytes = prompt_path.read_bytes()
    prompt_hash = hashlib.sha256(prompt_bytes).hexdigest()
    prompts = prompts_from(prompt_path, prompt_bytes)
    concurrencies = [int(x) for x in args.concurrency.split(",")]
    if not concurrencies or any(x < 1 for x in concurrencies):
        p.error("--concurrency must be comma-separated positive integers")
    if (args.requests < 1 or args.repeats < 1 or args.max_tokens < 1 or args.tp < 1 or
            args.max_model_len < 1 or args.max_num_batched_tokens < 1 or args.gpu_headroom_mib < 0 or
            not math.isfinite(args.timeout) or args.timeout <= 0):
        p.error("counts/limits/timeout must be positive and finite; headroom must be nonnegative")
    if args.requests < max(concurrencies):
        p.error("requests per repeat must be at least the largest requested concurrency")
    for path in (radiance, model):
        if not path.is_file():
            p.error(f"not a file: {path}")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "qos.json").exists():
        p.error("qos.json already exists; choose a fresh --out for this measurement")
    plugins = plugin_hashes(home)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    served = "spark-qos-" + uuid.uuid4().hex
    command = [str(radiance), "--model", str(model), "--radiance-home", str(home),
               "--kernels", args.kernels, "--host", "127.0.0.1", "--port", str(port),
               "--served-model-name", served, "--tp", str(args.tp), "--kv-cache-dtype", "bf16",
               "--max-num-seqs", str(max(concurrencies)), "--max-model-len", str(args.max_model_len),
               "--max-num-batched-tokens", str(args.max_num_batched_tokens),
               "--gpu-headroom-mib", str(args.gpu_headroom_mib),
               "--num-speculative-tokens", "0", "--no-prefix-cache"]
    receipt = {"model": str(model), "model_sha256": sha256(model),
               "radiance": str(radiance), "radiance_sha256": sha256(radiance),
               "radiance_home": str(home), "prompt_file": str(prompt_path),
               "prompt_file_sha256": prompt_hash, "kernels": args.kernels,
               "tp": args.tp, "kv_cache_dtype": "bf16", "prefix_cache": False,
               "speculation": 0, "max_num_seqs": max(concurrencies),
               "max_model_len": args.max_model_len,
               "max_num_batched_tokens": args.max_num_batched_tokens,
               "gpu_headroom_mib": args.gpu_headroom_mib,
               "requests_per_repeat": args.requests, "repeats": args.repeats,
               "max_tokens": args.max_tokens, "command": command, **plugins, "results": {}}
    if args.source_root:
        root = args.source_root.resolve()
        receipt["source"] = {
            "root": str(root),
            "revision": subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                        check=True, capture_output=True, text=True).stdout.strip(),
            "status_porcelain": subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                               check=True, capture_output=True, text=True).stdout.strip(),
        }
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    # Exclusive create refuses existing files and symlinks, including symlinks to inputs.
    server_log = (out / "server.log").open("x+")
    proc = subprocess.Popen(command, env=env, stdout=server_log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"Radiance exited before readiness; see {out / 'server.log'}")
            try:
                with urllib.request.urlopen(url + "/health", timeout=1):
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(0.2)
        else:
            raise TimeoutError(f"Radiance did not become ready; see {out / 'server.log'}")

        verify_server(url, served, proc)
        for concurrency in concurrencies:
            samples = []
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for repeat in range(args.repeats):
                    for base in range(0, args.requests, concurrency):
                        prompts_for_batch = [prompts[(repeat * args.requests + base + i) % len(prompts)]
                                             for i in range(min(concurrency, args.requests - base))]
                        futures = [pool.submit(request, url, served, item["prompt"],
                                               args.max_tokens, args.timeout)
                                   for item in prompts_for_batch]
                        samples.extend(f.result() for f in futures)
            wall = time.perf_counter() - started
            latencies = [x["seconds"] for x in samples]
            generated = sum(x["completion_tokens"] for x in samples)
            if (len(samples) != args.requests * args.repeats or
                    generated != args.requests * args.repeats * args.max_tokens or
                    not math.isfinite(wall) or wall <= 0):
                raise ValueError("QoS measurement did not complete the expected work")
            summary = {"requests": len(samples), "wall_seconds": wall,
                       "latency_seconds": {"p50": quantile(latencies, 0.50),
                                           "p95": quantile(latencies, 0.95),
                                           "max": max(latencies, default=0.0)},
                       "generated_tokens": generated,
                       "aggregate_completion_tokens_per_second": generated / wall if wall else 0.0,
                       "samples": samples}
            verify_server(url, served, proc)
            receipt["results"][str(concurrency)] = summary
            print(f"concurrency={concurrency} requests={len(samples)} p50={summary['latency_seconds']['p50']:.3f}s "
                  f"p95={summary['latency_seconds']['p95']:.3f}s max={summary['latency_seconds']['max']:.3f}s "
                  f"aggregate={summary['aggregate_completion_tokens_per_second']:.2f} tok/s")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        server_log.close()
    if (plugin_hashes(home) != plugins or sha256(radiance) != receipt["radiance_sha256"] or
            sha256(model) != receipt["model_sha256"] or sha256(prompt_path) != prompt_hash):
        raise ValueError("runtime, model, prompt file or plugin inventory changed during the measurement")
    report = out / "qos.json"
    with report.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    print(f"QoS receipt: {report}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, OSError, ValueError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        print(f"spark25_qos: {exc}", file=sys.stderr)
        sys.exit(1)
