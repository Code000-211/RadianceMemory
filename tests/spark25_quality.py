#!/usr/bin/env python3
"""Matched BF16-vs-IQ8R KLD run for real Spark containers (no conversion is done).

The KLD reference uses the exact same tokenized corpus for every candidate and records top-128
reference tokens plus the remaining probability mass as one tail bucket. Per-position candidate
rows are kept so severe cases can be replayed with spark25_kld_tail.py.

Example (run inside an externally managed GPU scope when using libr4d):
  python3 tests/spark25_quality.py --radiance build/bin/radiance \\
    --home build/radiance_home --reference /models/spark-bf16.rad \\
    --corpus /eval/independent-chat-eval.jsonl --provenance /eval/independent-chat-eval.manifest.json \\
    --out /tmp/spark-quality --candidate full=/models/spark-iq8r.rad \\
    --candidate keep12=/models/spark-iq8r-keep12.rad
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fnv1a64(path):
    # Match core/kld.cpp: getline hashes each line without its terminating LF.
    h = 1469598103934665603
    with Path(path).open("rb") as f:
        for line in f:
            if line.endswith(b"\n"):
                line = line[:-1]
            for b in line:
                h = ((h ^ b) * 1099511628211) & 0xffffffffffffffff
    return f"{h:016x}"


def parse_candidate(spec):
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError("candidate must be NAME=/path/to/model.rad")
    return name, Path(path).resolve()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--radiance", required=True, type=Path)
    p.add_argument("--home", required=True, type=Path, help="RADIANCE_HOME / plugin tree")
    p.add_argument("--reference", required=True, type=Path, help="BF16 .rad reference")
    p.add_argument("--corpus", required=True, type=Path, help="held-out KLD corpus JSONL")
    p.add_argument("--provenance", type=Path, help="evaluation manifest to hash into the run receipt")
    p.add_argument("--calibration-manifest", type=Path,
                   help="calibration dataset manifest to hash into the run receipt")
    p.add_argument("--source-root", type=Path, help="git checkout to record the exact source revision")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--candidate", required=True, action="append", type=parse_candidate,
                   metavar="NAME=RAD", help="repeat for each candidate container")
    p.add_argument("--kernels", default="libr4d")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-batched-tokens", type=int, default=128,
                   help="match the rowfix Spark quality replay (default: 128)")
    p.add_argument("--gpu-headroom-mib", type=int, default=1024)
    p.add_argument("--reuse-reference", action="store_true",
                   help="reuse an existing compatible top-128 reference instead of recording it")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    radiance, home, reference = args.radiance.resolve(), args.home.resolve(), args.reference.resolve()
    corpus, out = args.corpus.resolve(), args.out.resolve()
    candidates = args.candidate
    if len({name for name, _ in candidates}) != len(candidates):
        p.error("candidate names must be unique")
    for path in (radiance, reference, corpus, *(model for _, model in candidates)):
        if not path.is_file():
            p.error(f"not a file: {path}")
    ref_dir = out / "bf16-top128-ref"
    common = ["--radiance-home", str(home), "--kernels", args.kernels, "--tp", str(args.tp),
              "--kv-cache-dtype", "bf16", "--max-num-seqs", "1", "--max-model-len",
              str(args.max_model_len), "--max-num-batched-tokens", str(args.max_num_batched_tokens),
              "--gpu-headroom-mib", str(args.gpu_headroom_mib),
              "--num-speculative-tokens", "0", "--no-prefix-cache"]
    record = [str(radiance), "--model", str(reference), *common, "--kld-record", str(ref_dir),
              "--kld-corpus", str(corpus), "--kld-top", "128"]
    if args.dry_run:
        print(" ".join(record))
        for name, model in candidates:
            print(" ".join([str(radiance), "--model", str(model), *common, "--kld-ref",
                             str(ref_dir), "--kld-out", str(out / f"{name}-kld.json")]))
        return 0

    out.mkdir(parents=True, exist_ok=True)
    ref_meta_path = ref_dir / "kld.json"
    if args.reuse_reference:
        if not ref_meta_path.is_file():
            p.error(f"--reuse-reference requested but {ref_meta_path} is missing")
        ref_meta = json.loads(ref_meta_path.read_text())
        if (ref_meta.get("format") != "radiance-kld-top-1" or ref_meta.get("top") != 128 or
                ref_meta.get("corpus_fnv1a64") != fnv1a64(corpus)):
            p.error("existing reference is not the same top-128 reference for this corpus")
    else:
        if ref_dir.exists():
            p.error(f"reference directory already exists: {ref_dir}; choose another --out or use --reuse-reference")
        ref_dir.mkdir(parents=True)
        env = os.environ.copy()
        subprocess.run(record, env=env, check=True)
        ref_meta = json.loads(ref_meta_path.read_text())

    if ref_meta.get("format") != "radiance-kld-top-1" or ref_meta.get("top") != 128:
        p.error("reference is not radiance-kld-top-1 with top=128")
    if ref_meta.get("corpus_fnv1a64") != fnv1a64(corpus):
        p.error("KLD reference corpus hash does not match --corpus")

    run = {
        "reference_model": str(reference), "reference_sha256": sha256(reference),
        "reference_dir": str(ref_dir), "reference_form": "top-128 + tail",
        "corpus": str(corpus), "corpus_sha256": sha256(corpus),
        "corpus_fnv1a64": fnv1a64(corpus), "evaluation_manifest": None,
        "radiance": str(radiance), "radiance_sha256": sha256(radiance),
        "radiance_home": str(home), "kernel_selector": args.kernels,
        "tp": args.tp, "kv_cache_dtype": "bf16", "max_num_seqs": 1,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_headroom_mib": args.gpu_headroom_mib,
        "speculation": 0, "prefix_cache": False,
        "reference_manifest": ref_meta,
        "candidates": {},
    }
    if args.provenance:
        prov = args.provenance.resolve()
        run["evaluation_manifest"] = {"path": str(prov), "sha256": sha256(prov),
                                       "contents": json.loads(prov.read_text())}
    if args.calibration_manifest:
        calib = args.calibration_manifest.resolve()
        run["calibration_manifest"] = {"path": str(calib), "sha256": sha256(calib),
                                        "contents": json.loads(calib.read_text())}
    if args.source_root:
        root = args.source_root.resolve()
        def git(*cmd):
            return subprocess.run(["git", "-C", str(root), *cmd], check=True,
                                  stdout=subprocess.PIPE, text=True).stdout.strip()
        run["source"] = {"root": str(root), "revision": git("rev-parse", "HEAD"),
                         "status_porcelain": git("status", "--porcelain")}
    kernel_dir = home / "kernels"
    if kernel_dir.is_dir():
        run["kernel_shared_objects"] = {str(f.relative_to(home)): sha256(f)
                                        for f in sorted(kernel_dir.rglob("*.so"))}

    expected_rows = ref_meta.get("rows")
    for name, model in candidates:
        report = out / f"{name}-kld.json"
        command = [str(radiance), "--model", str(model), *common, "--kld-ref", str(ref_dir),
                   "--kld-out", str(report)]
        subprocess.run(command, env=os.environ.copy(), check=True)
        scored = json.loads(report.read_text())
        if scored.get("ref_form") != "top-128 + tail" or scored.get("all", {}).get("positions") != expected_rows:
            p.error(f"{name}: candidate report is not matched to the top-128 reference rows")
        values = scored["all"]["kld"]
        row_path = Path(str(report) + ".rows")
        raw = row_path.read_bytes()
        if len(raw) != expected_rows * 16:
            p.error(f"{name}: unexpected per-position rows size {len(raw)}")
        row_values = struct.unpack("<" + "f" * (expected_rows * 4), raw)
        kl = [row_values[i * 4] for i in range(expected_rows)]
        run["candidates"][name] = {"model": str(model), "sha256": sha256(model),
                                   "report": str(report), "rows": str(row_path),
                                   "metrics": scored["all"],
                                   "positions_over_1_nat": sum(x > 1.0 for x in kl)}
        print(f"{name}: {expected_rows} positions; ref_form={scored['ref_form']}; "
              f"mean={values['mean']:.6g}, p99={values['p99']:.6g}, "
              f"p99.9={values['p99_9']:.6g}, max={values['max']:.6g}, "
              f">1 nat={run['candidates'][name]['positions_over_1_nat']}")

    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"Run receipt: {out / 'run.json'}")
    print("Per-position rows are available as <name>-kld.json.rows; replay severe positions with "
          "tests/spark25_kld_tail.py. These are top-128-plus-tail KL values, not full-vocabulary KL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
