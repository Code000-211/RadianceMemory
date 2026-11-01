#!/usr/bin/env python3
"""Replay the largest matched-reference IQ8R KLD rows as one-token requests.

Uses the exact token prefixes recorded with --kld-record and verifies the server consumed each
entire prefix. Run against one candidate server configured with the same TP, BF16 KV, chunk limit,
no speculation, and no prefix cache as the KLD scoring run. The report stores only row metadata and
top candidates, not the prompt token sequence.
"""
import argparse
import json
from pathlib import Path
import struct
import sys
import urllib.error
import urllib.request


def post(url, body, timeout):
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True, type=Path,
                   help="directory from --kld-record (top-128 + tail)")
    p.add_argument("--report", required=True, type=Path,
                   help="candidate --kld-out JSON; matching .rows file must be present")
    p.add_argument("--url", required=True, help="candidate Radiance base URL, e.g. http://127.0.0.1:5914")
    p.add_argument("--model", default="spark", help="served model name")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--limit", type=int, default=20, help="replay this many largest-KL rows")
    p.add_argument("--threshold", type=float, default=1.0,
                   help="also include every row above this coarsened KL threshold")
    p.add_argument("--timeout", type=float, default=120)
    args = p.parse_args()

    ref_dir, report_path, out = args.reference.resolve(), args.report.resolve(), args.out.resolve()
    ref = json.loads((ref_dir / "kld.json").read_text())
    report = json.loads(report_path.read_text())
    if ref.get("format") != "radiance-kld-top-1" or ref.get("top") != 128:
        p.error("reference must be radiance-kld-top-1 with top=128")
    if report.get("ref_form") != "top-128 + tail":
        p.error("candidate report was not scored against the top-128-plus-tail reference")
    n = int(ref["rows"])
    if report.get("all", {}).get("positions") != n:
        p.error("candidate report position count differs from the reference")
    raw_rows = Path(str(report_path) + ".rows").read_bytes()
    if len(raw_rows) != n * 4 * 4:
        p.error("candidate .rows file has an unexpected size")
    values = struct.unpack("<" + "f" * (n * 4), raw_rows)
    kls = [values[4 * i] for i in range(n)]
    above = [i for i, kl in enumerate(kls) if kl > args.threshold]
    selected = set(sorted(range(n), key=kls.__getitem__, reverse=True)[:args.limit])
    selected.update(above)
    selected = sorted(selected, key=kls.__getitem__, reverse=True)

    docs = ref["docs"]
    tokens_raw = (ref_dir / "tokens.i32").read_bytes()
    if len(tokens_raw) % 4:
        p.error("tokens.i32 has an unexpected size")
    tokens = struct.unpack("<" + "i" * (len(tokens_raw) // 4), tokens_raw)
    token_starts = []
    total = 0
    for doc in docs:
        token_starts.append(total)
        total += int(doc["tokens"])
    if total != len(tokens):
        p.error("tokens.i32 length does not match the reference manifest")
    spans = [(int(d["row0"]), int(d["row0"]) + int(d["tokens"]) - int(d["score_from"]) - 1)
             for d in docs]
    top_count = int(ref["top"])
    top_raw = (ref_dir / "top.bin").read_bytes()
    if len(top_raw) != n * top_count * 8:
        p.error("top.bin has an unexpected size")

    results = []
    for row in selected:
        doc_index = next((i for i, (lo, hi) in enumerate(spans) if lo <= row < hi), None)
        if doc_index is None:
            p.error(f"row {row} is outside every reference document")
        doc = docs[doc_index]
        token_pos = int(doc["score_from"]) + row - int(doc["row0"])
        begin = token_starts[doc_index]
        prompt = list(tokens[begin:begin + token_pos + 1])
        if not prompt:
            p.error(f"row {row} produced an empty replay prompt")
        ref_entries = [struct.unpack_from("<if", top_raw, (row * top_count + i) * 8)
                       for i in range(top_count)]
        ref_top = sorted(ref_entries, key=lambda item: (-item[1], item[0]))[:20]
        response = post(args.url, {"model": args.model, "prompt": prompt, "max_tokens": 1,
                                   "temperature": 0, "ignore_eos": True, "logprobs": 20},
                        args.timeout)
        if int(response.get("usage", {}).get("prompt_tokens", -1)) != len(prompt):
            p.error(f"row {row}: server did not consume the full {len(prompt)}-token replay prefix")
        choice = response["choices"][0]
        content = choice["logprobs"]["content"]
        if len(content) != 1:
            p.error(f"row {row}: expected one scored output token, got {len(content)}")
        got = content[0]["top_logprobs"]
        got_top = [{"token_id": int(item["id"]), "logprob": float(item["logprob"])}
                   for item in got]
        result = {
            "row": row, "source": doc.get("source", ""), "document_index": doc_index,
            "token_position": token_pos, "prompt_tokens": len(prompt), "coarsened_kl": kls[row],
            "top1_agreement_from_kld": bool(values[4 * row + 3]),
            "reference_top20": [{"token_id": int(i), "logprob": float(lp)} for i, lp in ref_top],
            "candidate_top20": got_top,
        }
        results.append(result)
        print(f"row={row} source={result['source']} token={token_pos} prompt={len(prompt)} "
              f"top128+tail-KL={kls[row]:.6g} top1-agree={result['top1_agreement_from_kld']}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"reference": str(ref_dir), "candidate_report": str(report_path),
                               "ref_form": report["ref_form"], "threshold": args.threshold,
                               "rows_above_threshold": len(above), "replayed": results}, indent=2) + "\n")
    print(f"Replay receipt: {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, OSError, ValueError, struct.error, urllib.error.URLError) as exc:
        print(f"spark25_kld_tail: {exc}", file=sys.stderr)
        sys.exit(1)
