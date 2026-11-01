#!/usr/bin/env python3
"""Replay severe sparse-KL rows with a private, artifact-bound candidate server.

Requires run.json from spark25_quality.py. The server is started here, using that
run's candidate, executable, plugins and settings; an arbitrary external URL or
served-model alias cannot establish the model's identity. Run the whole command
inside the same exclusive GPU/device scope as the original measurement.
"""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

from spark25_validation import (plugin_hashes, reference_provenance, same_path, sha256,
                                validate_reference, validate_report)


def bound_run(run_path, ref_dir, report_path):
    run = json.loads(Path(run_path).read_text())
    ref = json.loads((ref_dir / "kld.json").read_text())
    report = json.loads(report_path.read_text())
    same_path(run["reference_dir"], ref_dir, "run reference directory")
    validate_reference(ref, run["reference_model"], run["corpus"])
    if run["reference_manifest"] != ref:
        raise ValueError("run reference manifest differs from the supplied reference")
    matches = [item for item in run["candidates"].values()
               if Path(item["report"]).resolve() == report_path]
    if len(matches) != 1:
        raise ValueError("report must identify exactly one candidate in run.json")
    candidate = matches[0]
    validate_report(report, ref_dir, run["reference_model"], run["corpus"],
                    candidate["model"], ref["rows"])
    same_path(candidate["rows"], str(report_path) + ".rows", "candidate row file")
    for path, expected in ((report_path, candidate["report_sha256"]),
                           (candidate["rows"], candidate["rows_sha256"]),
                           (candidate["model"], candidate["sha256"]),
                           (run["reference_model"], run["reference_sha256"]),
                           (run["corpus"], run["corpus_sha256"]),
                           (run["radiance"], run["radiance_sha256"])):
        if sha256(path) != expected:
            raise ValueError(f"artifact hash differs from run.json: {path}")
    plugins = plugin_hashes(run["radiance_home"])
    if any(run[key] != value for key, value in plugins.items()):
        raise ValueError("plugin inventory differs from the scored run")
    common = ["--radiance-home", run["radiance_home"], "--kernels", run["kernel_selector"],
              "--tp", str(run["tp"]), "--kv-cache-dtype", "bf16", "--max-num-seqs", "1",
              "--max-model-len", str(run["max_model_len"]), "--max-num-batched-tokens",
              str(run["max_num_batched_tokens"]), "--gpu-headroom-mib",
              str(run["gpu_headroom_mib"]), "--num-speculative-tokens", "0", "--no-prefix-cache"]
    inputs = {key: run[key] for key in ("reference_model", "reference_sha256", "corpus",
                                       "corpus_sha256", "corpus_fnv1a64", "radiance",
                                       "radiance_sha256")}
    inputs.update(plugins=plugins, arguments=common)
    if reference_provenance(ref_dir, inputs, reuse=True) != run["reference_provenance"]:
        raise ValueError("run reference provenance differs from the scored baseline")
    return run, candidate, ref, report, common


def get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post(url, model, prompt, timeout):
    request = urllib.request.Request(url + "/v1/completions", json.dumps({
        "model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0,
        "ignore_eos": True, "logprobs": 20}).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


@contextmanager
def candidate_server(run, candidate, common, log_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    model = "spark-tail-" + uuid.uuid4().hex
    url = f"http://127.0.0.1:{port}"
    command = [run["radiance"], "--model", candidate["model"], *common,
               "--host", "127.0.0.1", "--port", str(port), "--served-model-name", model]
    with log_path.open("w") as log:
        env = os.environ.copy()
        env.pop("HF_TOKEN", None)
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"candidate server exited; see {log_path}")
                try:
                    get(url + "/health", 1)
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.2)
            else:
                raise TimeoutError(f"candidate server readiness timeout; see {log_path}")
            models = get(url + "/v1/models", 5)
            if (proc.poll() is not None or
                    model not in {item.get("id") for item in models.get("data", [])}):
                raise ValueError("private server identity does not match the launched candidate")
            yield {"url": url, "model": model, "pid": proc.pid, "command": command}
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="bound quality run.json")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path, help="candidate KLD JSON with .rows")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if (args.limit < 1 or not math.isfinite(args.threshold) or args.threshold < 0 or
            not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("limit/timeout must be positive and threshold finite and nonnegative")
    run_path, ref_dir = args.run.resolve(), args.reference.resolve()
    report_path, out = args.report.resolve(), args.out.resolve()
    if out.exists():
        parser.error("replay receipt already exists; choose a fresh --out")
    run_digest = sha256(run_path)
    run, candidate, ref, report, common = bound_run(run_path, ref_dir, report_path)
    n = ref["rows"]
    raw_rows = Path(candidate["rows"]).read_bytes()
    if len(raw_rows) != n * 16:
        raise ValueError("candidate .rows file has an unexpected size")
    values = struct.unpack("<" + "f" * (n * 4), raw_rows)
    kls = [values[4 * i] for i in range(n)]
    # Preserve signed roundoff near zero from core/kld.cpp rather than clamp the data.
    if not all(math.isfinite(kl) for kl in kls):
        raise ValueError("candidate has non-finite coarsened KL rows")
    above = [i for i, kl in enumerate(kls) if kl > args.threshold]
    selected = set(sorted(range(n), key=kls.__getitem__, reverse=True)[:args.limit])
    selected.update(above)
    selected = sorted(selected, key=kls.__getitem__, reverse=True)
    docs = ref["docs"]
    tokens_raw = (ref_dir / "tokens.i32").read_bytes()
    if len(tokens_raw) % 4:
        raise ValueError("tokens.i32 has an unexpected size")
    tokens = struct.unpack("<" + "i" * (len(tokens_raw) // 4), tokens_raw)
    token_starts, total = [], 0
    for doc in docs:
        token_starts.append(total)
        total += int(doc["tokens"])
    if total != len(tokens):
        raise ValueError("tokens.i32 length does not match the reference manifest")
    spans = [(int(d["row0"]), int(d["row0"]) + int(d["tokens"]) - int(d["score_from"]) - 1)
             for d in docs]
    top_raw = (ref_dir / "top.bin").read_bytes()
    if len(top_raw) != n * 128 * 8:
        raise ValueError("top.bin has an unexpected size")
    results = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with candidate_server(run, candidate, common, out.with_suffix(".server.log")) as server:
        for row in selected:
            doc_index = next((i for i, (lo, hi) in enumerate(spans) if lo <= row < hi), None)
            if doc_index is None:
                raise ValueError(f"row {row} is outside every reference document")
            doc = docs[doc_index]
            token_pos = int(doc["score_from"]) + row - int(doc["row0"])
            begin = token_starts[doc_index]
            prompt = list(tokens[begin:begin + token_pos + 1])
            if not prompt:
                raise ValueError(f"row {row} produced an empty replay prompt")
            ref_entries = [struct.unpack_from("<if", top_raw, (row * 128 + i) * 8)
                           for i in range(128)]
            ref_top = sorted(ref_entries, key=lambda item: (-item[1], item[0]))[:20]
            response = post(server["url"], server["model"], prompt, args.timeout)
            usage = response.get("usage", {})
            if (usage.get("prompt_tokens") != len(prompt) or usage.get("completion_tokens") != 1 or
                    usage.get("prompt_tokens_details", {}).get("cached_tokens") != 0):
                raise ValueError(f"row {row}: prefix, completion or cold-cache accounting mismatch")
            content = response["choices"][0]["logprobs"]["content"]
            if len(content) != 1 or len(content[0]["top_logprobs"]) != 20:
                raise ValueError(f"row {row}: expected one scored output with top20")
            got_top = [{"token_id": int(item["id"]), "logprob": float(item["logprob"])}
                       for item in content[0]["top_logprobs"]]
            if not all(math.isfinite(item["logprob"]) for item in got_top):
                raise ValueError(f"row {row}: non-finite candidate logprob")
            result = {"row": row, "source": doc.get("source", ""), "document_index": doc_index,
                      "token_position": token_pos, "prompt_tokens": len(prompt), "coarsened_kl": kls[row],
                      "top1_agreement_from_kld": bool(values[4 * row + 3]),
                      "reference_top20": [{"token_id": int(i), "logprob": float(lp)} for i, lp in ref_top],
                      "candidate_top20": got_top}
            results.append(result)
            print(f"row={row} source={result['source']} token={token_pos} prompt={len(prompt)} "
                  f"top128+tail-KL={kls[row]:.6g} top1-agree={result['top1_agreement_from_kld']}")
    # A changed binary/model/report cannot silently acquire a successful replay receipt.
    bound_run(run_path, ref_dir, report_path)
    if sha256(run_path) != run_digest:
        raise ValueError("run.json changed during replay")
    out.write_text(json.dumps({"source_run": str(run_path), "source_run_sha256": run_digest,
        "reference": str(ref_dir), "candidate_report": str(report_path),
        "candidate_model": candidate["model"], "candidate_sha256": candidate["sha256"],
        "radiance_sha256": run["radiance_sha256"], "server": server,
        **{key: run[key] for key in plugin_hashes(run["radiance_home"])},
        "ref_form": report["ref_form"], "threshold": args.threshold,
        "rows_above_threshold": len(above), "replayed": results}, indent=2) + "\n")
    print(f"Replay receipt: {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, OSError, ValueError, TypeError, struct.error, urllib.error.URLError,
            TimeoutError, RuntimeError) as exc:
        print(f"spark25_kld_tail: {exc}", file=sys.stderr)
        sys.exit(1)
