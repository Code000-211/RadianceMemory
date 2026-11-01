#!/usr/bin/env python3
"""Repeatable DSpark serving/QoS smoke test against two already-running Radiance servers.

Start the same model/checkpoint and build on both endpoints with empty caches, changing only the
prefix-cache setting:

  python3 scripts/dspark_qos_regression.py \
    --cache-off-url http://127.0.0.1:8000 --cache-on-url http://127.0.0.1:8001 \
    --model spark --output dspark-qos.json

The off server must use --no-prefix-cache; the on server must have a usable cache. Restart both
with empty caches before each run (the prompt also carries a fresh run nonce). This script
reports first streamed text-chunk/end-to-end latency and token rates alongside Prometheus counter
deltas. A text chunk can contain multiple model tokens, so it is not necessarily true token TTFT. It
also checks a >512-token prefill, a generated crossing of the 512-token boundary, drafting with
unaccepted draft proposals, streaming disconnect accounting, EOS/stop/tool protocol behavior,
and repeated seeded sampling. Aggregate counters cannot identify a partially accepted verify or
prove rollback correctness. Pass --allow-unobserved-rejection only for a deliberately weaker
diagnostic run; the JSON retains every failed assertion.

This is a runtime release harness, not a numerical-equivalence test. Record model, server build,
GPU/kernel revision, launch arguments and the JSON artifact when comparing runs. No latency claim
is inferred from a single sample; the output contains repeated samples and p50/p95 summaries.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import platform
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METRICS = (
    "radiance:attn_cached_tokens_total",
    "radiance:linear_cached_tokens_total",
    "radiance:prefix_cache_evictions_total",
    "radiance:draft_tokens_total",
    "radiance:draft_accepted_total",
    "radiance:draft_refills_total",
    "radiance:draft_acceptance_ratio",
    "radiance:linear_prefix_cache_hit_rate",
    "radiance:prefill_tokens_per_second",
    "radiance:decode_tokens_per_second",
    "radiance:client_disconnect_total",
    "vllm:gpu_prefix_cache_hit_rate",
)
METRIC_LINE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|[+-]?Inf)(?:\s+\d+)?$"
)


class HarnessError(RuntimeError):
    pass


def endpoint(base: str, path: str) -> str:
    return base.rstrip("/") + path


def fetch(url: str, *, data: bytes | None = None, headers: dict[str, str] | None = None,
          timeout: float = 120.0):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="GET" if data is None else "POST")
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read(4096).decode("utf-8", "replace")
        raise HarnessError(f"HTTP {e.code} from {url}: {detail}") from e
    except urllib.error.URLError as e:
        raise HarnessError(f"request to {url} failed: {e}") from e


def scrape(base: str, api_key: str | None) -> dict[str, float]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with fetch(endpoint(base, "/metrics"), headers=headers) as r:
        raw = r.read().decode("utf-8", "replace")
    values: dict[str, float] = {}
    for line in raw.splitlines():
        m = METRIC_LINE.match(line.strip())
        if not m:
            continue
        name, value = m.groups()
        try:
            n = float(value)
        except ValueError:
            continue
        if math.isfinite(n):
            values[name] = values.get(name, 0.0) + n
    return values


def metric_delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    if name not in before or name not in after:
        raise HarnessError(f"required metric {name} missing from one or both scrapes")
    delta = after[name] - before[name]
    if delta < 0:
        raise HarnessError(f"counter {name} decreased; server restarted or metric labels changed")
    return delta


def selected_metrics(values: dict[str, float]) -> dict[str, float | None]:
    return {name: values.get(name) for name in METRICS}


def chat_request(base: str, model: str, messages: list[dict[str, Any]], *, api_key: str | None,
                 timeout: float, **options: Any) -> dict[str, Any]:
    body = {"model": model, "messages": messages, **options}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    start = time.monotonic()
    with fetch(endpoint(base, "/v1/chat/completions"), data=json.dumps(body).encode(),
               headers=headers, timeout=timeout) as r:
        payload = json.loads(r.read())
    elapsed = time.monotonic() - start
    choices = payload.get("choices") or []
    if not choices:
        raise HarnessError("chat completion returned no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    usage = payload.get("usage") or {}
    return {
        "elapsed_s": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "content": message.get("content") or "",
        "tool_calls": message.get("tool_calls") or [],
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "id": payload.get("id"),
    }


def stream_request(base: str, model: str, prompt: str, *, api_key: str | None,
                   timeout: float, cancel_after_first_event: bool = False,
                   **options: Any) -> dict[str, Any]:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "stream": True, **options}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    start = time.monotonic()
    response = fetch(endpoint(base, "/v1/chat/completions"), data=json.dumps(body).encode(),
                     headers=headers, timeout=timeout)
    content: list[str] = []
    finish = None
    first_event_s = None
    first_text_s = None
    events = 0
    try:
        while True:
            line = response.readline()
            if not line:
                break
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            events += 1
            if first_event_s is None:
                first_event_s = time.monotonic() - start
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                reasoning = delta.get("reasoning_content")
                if first_text_s is None and ((isinstance(piece, str) and piece) or
                                             (isinstance(reasoning, str) and reasoning)):
                    first_text_s = time.monotonic() - start
                if isinstance(piece, str):
                    content.append(piece)
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
            if cancel_after_first_event and first_event_s is not None:
                break
    finally:
        response.close()
    return {
        "elapsed_s": time.monotonic() - start,
        "first_event_s": first_event_s,
        "first_text_s": first_text_s,
        "finish_reason": finish,
        "content": "".join(content),
        "events": events,
        "cancelled": cancel_after_first_event,
    }


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    return ys[min(len(ys) - 1, max(0, math.ceil(p * len(ys)) - 1))]


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [float(s["elapsed_s"]) for s in samples if s.get("elapsed_s") is not None]
    first_text = [float(s["first_text_s"]) for s in samples if s.get("first_text_s") is not None]
    rates = []
    for s in samples:
        n, t = s.get("completion_tokens"), s.get("elapsed_s")
        if isinstance(n, (int, float)) and n > 0 and t and t > 0:
            rates.append(float(n) / float(t))
    return {
        "n": len(samples),
        "end_to_end_s": {"p50": percentile(elapsed, .50), "p95": percentile(elapsed, .95)},
        "first_text_s": {"p50": percentile(first_text, .50), "p95": percentile(first_text, .95)},
        "completion_tokens_per_second": {"p50": percentile(rates, .50), "p95": percentile(rates, .95)},
    }


def cache_probe(base: str, name: str, model: str, api_key: str | None,
                timeout: float, prompt: str, repeats: int) -> dict[str, Any]:
    before = scrape(base, api_key)
    first = chat_request(base, model, [{"role": "user", "content": prompt}], api_key=api_key,
                         timeout=timeout, temperature=0, seed=17, max_tokens=4)
    middle = scrape(base, api_key)
    warm = [chat_request(base, model, [{"role": "user", "content": prompt}], api_key=api_key,
                         timeout=timeout, temperature=0, seed=17, max_tokens=4)
            for _ in range(repeats)]
    after = scrape(base, api_key)
    return {
        "name": name, "first": first, "warm_samples": warm,
        "latency": summarize([first, *warm]),
        "first_attn_cache_tokens": metric_delta(before, middle, "radiance:attn_cached_tokens_total"),
        "warm_attn_cache_tokens": metric_delta(middle, after, "radiance:attn_cached_tokens_total"),
        "first_linear_cache_tokens": metric_delta(before, middle, "radiance:linear_cached_tokens_total"),
        "warm_linear_cache_tokens": metric_delta(middle, after, "radiance:linear_cached_tokens_total"),
        "evictions": metric_delta(before, after, "radiance:prefix_cache_evictions_total"),
    }


def require(result: dict[str, Any], name: str, passed: bool, detail: str) -> None:
    result.setdefault("assertions", []).append({"name": name, "passed": bool(passed), "detail": detail})
    if not passed:
        result.setdefault("failures", []).append(f"{name}: {detail}")


def stop_sequence_observed(unstopped: dict[str, Any], stopped: dict[str, Any], marker: str) -> bool:
    before_marker, found, _ = unstopped["content"].partition(marker)
    return bool(found) and stopped["finish_reason"] == "stop" and stopped["content"] == before_marker


def calibrated_rollover_prompt(base: str, model: str, api_key: str | None,
                               timeout: float, target: int = 480) -> tuple[str, int]:
    """Find a repeated prompt below the 512 boundary using response usage (no tokenizer guess)."""
    phrase = "rollover-boundary marker "
    lo, hi, best = 1, 1, None
    # Bracket from small inputs; starting at 1024 repetitions can exceed the server's
    # max-model-len before calibration has found a safe prompt.
    while True:
        r = chat_request(base, model, [{"role": "user", "content": phrase * hi}], api_key=api_key,
                         timeout=timeout, temperature=0, seed=23, max_tokens=1)
        n = r.get("prompt_tokens")
        if not isinstance(n, int):
            raise HarnessError("usage.prompt_tokens is required to calibrate the rollover boundary")
        if n > target:
            break
        best = (phrase * hi, n)
        lo = hi + 1
        hi *= 2
        if hi > 2048:
            raise HarnessError("could not bracket the rollover boundary in 2048 repetitions")
    while lo <= hi:
        count = (lo + hi) // 2
        prompt = phrase * count
        r = chat_request(base, model, [{"role": "user", "content": prompt}], api_key=api_key,
                         timeout=timeout, temperature=0, seed=23, max_tokens=1)
        n = r.get("prompt_tokens")
        if not isinstance(n, int):
            raise HarnessError("usage.prompt_tokens is required to calibrate the rollover boundary")
        if n <= target:
            best = (prompt, n)
            lo = count + 1
        else:
            hi = count - 1
    if best is None:
        raise HarnessError("could not construct a prompt below the 512-token rollover boundary")
    return best


def run(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "model": args.model, "cache_off_url": args.cache_off_url, "cache_on_url": args.cache_on_url,
        "configuration": {"short_repeats": args.short_repeats, "long_repeats": args.long_repeats,
                          "long_output_tokens": args.long_output_tokens,
                          "rollover_output_tokens": args.rollover_output_tokens,
                          "rejection_trials": args.rejection_trials},
        "failures": [], "assertions": [], "scenarios": {},
    }
    off_url, on_url = args.cache_off_url, args.cache_on_url
    off_initial = scrape(off_url, args.api_key)
    on_initial = scrape(on_url, args.api_key)
    nonce = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    long_cache_prompt = f"DSpark-{nonce} cache reuse marker. " + "cache prefix item. " * 180
    off = cache_probe(off_url, "prefix_cache_off", args.model, args.api_key, args.timeout,
                      long_cache_prompt, args.short_repeats)
    on = cache_probe(on_url, "prefix_cache_on", args.model, args.api_key, args.timeout,
                     long_cache_prompt, args.short_repeats)
    result["scenarios"]["prefix_cache"] = {"off": off, "on": on}
    require(result, "cache_off_has_no_reuse",
            off["warm_attn_cache_tokens"] == 0 and off["warm_linear_cache_tokens"] == 0,
            f"warm attention/linear hits={off['warm_attn_cache_tokens']}/{off['warm_linear_cache_tokens']}")
    require(result, "cache_on_reuses_prefix",
            on["warm_attn_cache_tokens"] > 0 or on["warm_linear_cache_tokens"] > 0,
            f"warm attention/linear hits={on['warm_attn_cache_tokens']}/{on['warm_linear_cache_tokens']}")

    # Long prompt and decode that begins well beyond the 512-token DSpark history.
    long_prompt = "DSpark long-context oracle marker 37. " * args.long_repeats
    long_samples = []
    long_before = scrape(on_url, args.api_key)
    for i in range(args.short_repeats):
        s = chat_request(on_url, args.model, [{"role": "user", "content": long_prompt}],
                         api_key=args.api_key, timeout=args.timeout, temperature=0, seed=31,
                         max_tokens=args.long_output_tokens)
        long_samples.append(s)
    long_after = scrape(on_url, args.api_key)
    long_tokens = long_samples[-1].get("prompt_tokens") if long_samples else None
    result["scenarios"]["long_context"] = {
        "prompt_tokens": long_tokens, "samples": long_samples, "latency": summarize(long_samples),
        "draft_tokens": metric_delta(long_before, long_after, "radiance:draft_tokens_total"),
        "draft_accepted": metric_delta(long_before, long_after, "radiance:draft_accepted_total"),
        "draft_refills": metric_delta(long_before, long_after, "radiance:draft_refills_total"),
    }
    require(result, "long_context_exceeds_512",
            isinstance(long_tokens, int) and long_tokens > 512,
            f"reported prompt_tokens={long_tokens}")

    # Precisely place the input below 512 from usage and generate across the boundary.
    rollover_prompt, calibrated_tokens = calibrated_rollover_prompt(
        on_url, args.model, args.api_key, args.timeout)
    roll_before = scrape(on_url, args.api_key)
    rollover = chat_request(on_url, args.model, [{"role": "user", "content": rollover_prompt}],
                            api_key=args.api_key, timeout=args.timeout, temperature=0, seed=43,
                            max_tokens=args.rollover_output_tokens)
    roll_after = scrape(on_url, args.api_key)
    completion = rollover.get("completion_tokens")
    actual_prompt_tokens = rollover.get("prompt_tokens")
    crossed = (isinstance(actual_prompt_tokens, int) and isinstance(completion, int) and
               actual_prompt_tokens < 512 and actual_prompt_tokens + completion >= 512)
    result["scenarios"]["window_rollover"] = {
        "calibrated_prompt_tokens": calibrated_tokens,
        "prompt_tokens": actual_prompt_tokens, "completion_tokens": completion,
        "crossed_absolute_512_boundary": crossed, "sample": rollover,
        "draft_tokens": metric_delta(roll_before, roll_after, "radiance:draft_tokens_total"),
        "draft_accepted": metric_delta(roll_before, roll_after, "radiance:draft_accepted_total"),
        "draft_refills": metric_delta(roll_before, roll_after, "radiance:draft_refills_total"),
    }
    require(result, "generation_crosses_512_boundary", crossed,
            f"calibrated={calibrated_tokens}, actual prompt={actual_prompt_tokens}, completion={completion}")

    # A true stream and a client-side cancellation; the latter is checked against the disconnect counter.
    stream = stream_request(on_url, args.model, "Give a concise description of a star.",
                            api_key=args.api_key, timeout=args.timeout, temperature=0, seed=53,
                            max_tokens=48)
    result["scenarios"]["streaming"] = stream
    require(result, "stream_emits_text", stream["events"] > 0 and stream["first_text_s"] is not None,
            f"events={stream['events']}, first_text={stream['first_text_s']}")
    disconnect_before = scrape(on_url, args.api_key)
    cancelled = stream_request(on_url, args.model,
                               "Continue an unbounded story about a long voyage through space.",
                               api_key=args.api_key, timeout=args.timeout,
                               cancel_after_first_event=True, temperature=0.8, seed=59,
                               max_tokens=2048)
    disconnect_after = scrape(on_url, args.api_key)
    deadline = time.monotonic() + 5.0
    while metric_delta(disconnect_before, disconnect_after, "radiance:client_disconnect_total") <= 0 and \
            time.monotonic() < deadline:
        time.sleep(0.1)
        disconnect_after = scrape(on_url, args.api_key)
    disconnect_delta = metric_delta(disconnect_before, disconnect_after,
                                    "radiance:client_disconnect_total")
    result["scenarios"]["cancellation"] = {**cancelled, "disconnect_counter_delta": disconnect_delta}
    require(result, "client_disconnect_is_counted", disconnect_delta > 0,
            f"radiance:client_disconnect_total delta={disconnect_delta}")

    # A stop finish alone does not prove the explicit marker fired. Compare a matching untruncated
    # no-stop control and require that it actually emits the marker.
    eos = chat_request(on_url, args.model,
                       [{"role": "user", "content": "Reply with exactly OK and then end the turn."}],
                       api_key=args.api_key, timeout=args.timeout, temperature=0, seed=61,
                       max_tokens=24)
    stop_prompt = [{"role": "user", "content": "Write exactly DS_STOP_6f41 and nothing after it."}]
    unstopped = chat_request(on_url, args.model, stop_prompt,
                             api_key=args.api_key, timeout=args.timeout, temperature=0, seed=67,
                             max_tokens=24)
    stopped = chat_request(on_url, args.model, stop_prompt,
                           api_key=args.api_key, timeout=args.timeout, temperature=0, seed=67,
                           max_tokens=24, stop=["DS_STOP_6f41"])
    result["scenarios"]["eos_and_stop"] = {"eos": eos, "without_stop": unstopped, "stop": stopped}
    require(result, "natural_end_of_turn", eos["finish_reason"] == "stop",
            f"finish_reason={eos['finish_reason']}")
    marker = "DS_STOP_6f41"
    stop_proven = stop_sequence_observed(unstopped, stopped, marker)
    require(result, "explicit_stop_sequence", stop_proven,
            f"control_contains_marker={marker in unstopped['content']}, finish_reason={stopped['finish_reason']}, "
            f"control={unstopped['content']!r}, stopped={stopped['content']!r}")

    tool = chat_request(
        on_url, args.model,
        [{"role": "user", "content": "Call emit_probe with value dspark_qos now; do not answer in text."}],
        api_key=args.api_key, timeout=args.timeout, temperature=0, seed=71, max_tokens=48,
        tools=[{"type": "function", "function": {
            "name": "emit_probe", "description": "Record the DSpark QoS tool-call probe.",
            "parameters": {"type": "object", "properties": {"value": {"type": "string"}},
                           "required": ["value"], "additionalProperties": False}}}],
        tool_choice={"type": "function", "function": {"name": "emit_probe"}},
    )
    result["scenarios"]["tool_call"] = tool
    require(result, "tool_call_is_emitted", bool(tool["tool_calls"]),
            f"finish_reason={tool['finish_reason']}, tool_calls={len(tool['tool_calls'])}")

    seeded_body = [{"role": "user", "content": "Write a short, varied sentence about a comet."}]
    seeded_a = chat_request(on_url, args.model, seeded_body, api_key=args.api_key, timeout=args.timeout,
                            temperature=0.75, seed=99173, max_tokens=32)
    seeded_b = chat_request(on_url, args.model, seeded_body, api_key=args.api_key, timeout=args.timeout,
                            temperature=0.75, seed=99173, max_tokens=32)
    same = seeded_a["content"] == seeded_b["content"]
    result["scenarios"]["seeded_sampling"] = {"first": seeded_a, "repeat": seeded_b, "identical": same}
    require(result, "seed_reproduces_sample", same,
            f"first={seeded_a['content']!r}, repeat={seeded_b['content']!r}")

    # Exercise concurrent batching, then look for unaccepted proposals. Aggregate per-request
    # counters cannot establish whether any single verify was partially accepted.
    def concurrent_call(idx: int) -> dict[str, Any]:
        return chat_request(on_url, args.model,
                            [{"role": "user", "content": f"In two sentences, describe a different planet. Case {idx}."}],
                            api_key=args.api_key, timeout=args.timeout, temperature=0.6,
                            seed=101 + idx, max_tokens=40)
    concurrent_before = scrape(on_url, args.api_key)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        concurrent_samples = list(pool.map(concurrent_call, (0, 1)))
    concurrent_after = scrape(on_url, args.api_key)
    result["scenarios"]["concurrent"] = {
        "samples": concurrent_samples, "latency": summarize(concurrent_samples),
        "draft_tokens": metric_delta(concurrent_before, concurrent_after, "radiance:draft_tokens_total"),
        "draft_accepted": metric_delta(concurrent_before, concurrent_after, "radiance:draft_accepted_total"),
    }

    rejection_trials = []
    any_rejection = False
    for i in range(args.rejection_trials):
        before = scrape(on_url, args.api_key)
        sample = chat_request(on_url, args.model,
                              [{"role": "user", "content": f"Continue this changing story with new details: trial {i}, a river crosses a city."}],
                              api_key=args.api_key, timeout=args.timeout, temperature=0.85,
                              seed=1201 + i, max_tokens=80)
        after = scrape(on_url, args.api_key)
        proposed = metric_delta(before, after, "radiance:draft_tokens_total")
        accepted = metric_delta(before, after, "radiance:draft_accepted_total")
        rejected = proposed > accepted
        any_rejection = any_rejection or rejected
        rejection_trials.append({"sample": sample, "draft_tokens": proposed,
                                 "draft_accepted": accepted, "unaccepted_proposals": rejected})
    result["scenarios"]["unaccepted_proposals"] = {"trials": rejection_trials,
                                                      "observed": any_rejection}
    require(result, "draft_proposals_are_observed",
            any(t["draft_tokens"] > 0 for t in rejection_trials),
            "no draft_tokens_total increase; confirm DSpark is enabled on this server")
    if not args.allow_unobserved_rejection:
        require(result, "unaccepted_proposals_are_observed", any_rejection,
                f"trials={len(rejection_trials)}; per-trial deltas are recorded")

    # QoS summary is deliberately descriptive. Compare these artifacts under matched launch args;
    # this harness does not invent universal latency or acceptance thresholds.
    result["summaries"] = {
        "prefix_cache_off": off["latency"], "prefix_cache_on": on["latency"],
        "long_context": result["scenarios"]["long_context"]["latency"],
        "concurrent": result["scenarios"]["concurrent"]["latency"],
        "unaccepted_proposal_trials": summarize([t["sample"] for t in rejection_trials]),
    }
    result["server_metrics"] = {
        "cache_off_initial": selected_metrics(off_initial),
        "cache_off_final": selected_metrics(scrape(off_url, args.api_key)),
        "cache_on_initial": selected_metrics(on_initial),
        "cache_on_final": selected_metrics(scrape(on_url, args.api_key)),
    }
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-off-url", required=True, help="root URL of --prefix-cache-mib 0 server")
    p.add_argument("--cache-on-url", required=True, help="root URL of server with prefix cache enabled")
    p.add_argument("--model", required=True, help="OpenAI-compatible model id")
    p.add_argument("--api-key", default=None, help="optional bearer token (never written to output)")
    p.add_argument("--output", type=Path, default=Path("dspark-qos.json"))
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--short-repeats", type=int, default=3)
    p.add_argument("--long-repeats", type=int, default=160,
                   help="repetitions of a marker phrase; response usage must confirm >512 tokens")
    p.add_argument("--long-output-tokens", type=int, default=48)
    p.add_argument("--rollover-output-tokens", type=int, default=96)
    p.add_argument("--rejection-trials", "--partial-rejection-trials", dest="rejection_trials",
                   type=int, default=8,
                   help="trials to look for unaccepted proposals; aggregate counts cannot prove partial rejection")
    p.add_argument("--allow-unobserved-rejection", action="store_true",
                   help="record but do not fail when sampled trials find no unaccepted proposals")
    args = p.parse_args()
    if args.short_repeats < 1 or args.long_repeats < 1 or args.rejection_trials < 1:
        p.error("repeat counts must be positive")
    try:
        result = run(args)
    except Exception as e:
        print(f"dspark_qos_regression: {e}", file=sys.stderr)
        return 2
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"failures={len(result['failures'])}; long_context={result['scenarios']['long_context']['prompt_tokens']} prompt tokens; "
          f"rollover={result['scenarios']['window_rollover']['crossed_absolute_512_boundary']}; "
          f"unaccepted_proposals={result['scenarios']['unaccepted_proposals']['observed']}")
    for failure in result["failures"]:
        print(f"FAIL {failure}")
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
