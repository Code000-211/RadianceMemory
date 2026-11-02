#!/usr/bin/env python3
"""Exercise the compile-time-only direct KV rollback observer against a live Spark server.

Build the server with -DRAD_ENABLE_KV_ROLLBACK_TEST_HOOK=ON and set
RADIANCE_TEST_KV_ROLLBACK_REPORT in that server's environment to the same report path supplied
here. This script sends a real greedy completion; it does not synthesize KV data.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="server base URL, e.g. http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", required=True, type=Path,
                        help="same path configured in server's RADIANCE_TEST_KV_ROLLBACK_REPORT")
    parser.add_argument("--prompt", default=("Write a detailed, varied explanation of why "
                                               "forest ecosystems change over time. " * 20))
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    if args.report.exists():
        parser.error(f"report path already exists; choose a fresh path: {args.report}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    evidence = None
    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    for attempt in range(8):
        prompt = args.prompt + f"\nVariation {attempt + 1}: use a distinct example."
        body = json.dumps({"model": args.model, "messages": [{"role": "user", "content": prompt}],
                           "temperature": 0, "max_tokens": args.max_tokens,
                           "reasoning_effort": "none", "stream": False}).encode()
        request = urllib.request.Request(endpoint, body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                result = json.loads(response.read())
            if not result.get("choices"):
                raise RuntimeError("server returned no completion choices")
        except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"request {attempt + 1} failed: {exc}", file=sys.stderr)
            return 1
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if args.report.exists():
                try:
                    candidate = json.loads(args.report.read_text())
                except (OSError, json.JSONDecodeError):
                    time.sleep(0.05)
                    continue
                if candidate.get("next_greedy_decode_observed"):
                    evidence = candidate
                    break
            time.sleep(0.05)
        if evidence is not None:
            break
    if evidence is None:
        print("no completed direct KV observer report; a partial depth-7 rollback followed by "
              "the next greedy decode was not observed", file=sys.stderr)
        return 1
    required = {
        "partial_rejection_observed": evidence.get("partial_rejection_observed") is True,
        "depth_7": evidence.get("proposals") == 7 and 0 < evidence.get("accepted", 0) < 7,
        "rollback_length": evidence.get("prefix_tokens_after") == evidence.get("rollback_to_tokens"),
        "bytes_equal": evidence.get("bytes_equal") is True,
        "mapping_equal": evidence.get("mapping_equal") is True,
        "physical_slot_mapping": len(evidence.get("logical_blocks", [])) > 0 and
                                 len(evidence.get("logical_blocks", [])) ==
                                 len(evidence.get("physical_slots", [])) and
                                 evidence.get("logical_blocks") ==
                                 sorted(set(evidence.get("logical_blocks", []))) and
                                 len(evidence.get("physical_slots", [])) ==
                                 len(set(evidence.get("physical_slots", []))),
        "all_complete_prefix_blocks": evidence.get("block_size", 0) > 0 and
                                      evidence.get("prefix_tokens_before_verify", 0) >=
                                      evidence.get("first_block_pos", 0) and
                                      len(evidence.get("logical_blocks", [])) ==
                                      (evidence.get("prefix_tokens_before_verify", 0) -
                                       evidence.get("first_block_pos", 0)) //
                                      evidence.get("block_size", 1),
        "snapshot_geometry": evidence.get("layers", 0) > 0 and
                             evidence.get("bytes_per_layer", 0) * evidence.get("layers", 0) *
                             len(evidence.get("logical_blocks", [])) ==
                             evidence.get("committed_prefix_bytes", 0),
        "negative_control": evidence.get("negative_control_detected") is True,
        "next_greedy_decode": evidence.get("next_greedy_decode_observed") is True and
                              isinstance(evidence.get("next_greedy_token"), int) and
                              evidence.get("next_greedy_token") >= 0,
    }
    print(json.dumps({"checks": required, "evidence": evidence}, indent=2))
    return 0 if all(required.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
