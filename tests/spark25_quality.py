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
import json
import math
import os
import shutil
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


from spark25_validation import (fnv1a64, plugin_hashes, reference_provenance, sha256,
                                validate_reference, validate_report)


class ScoreStaging:
    """Clean normally, but retain recoverable backups if rollback itself fails."""
    def __init__(self, out):
        self.path = Path(tempfile.mkdtemp(prefix=".spark25-score-", dir=out))
        self.keep = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if not self.keep:
            shutil.rmtree(self.path)


def parse_candidate(spec):
    name, sep, path = spec.partition("=")
    if not sep or not name or not path or Path(name).name != name or name in (".", ".."):
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
    if (args.tp < 1 or args.max_model_len < 1 or args.max_num_batched_tokens < 1 or
            args.gpu_headroom_mib < 0):
        p.error("TP/model/chunk limits must be positive and headroom nonnegative")

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

    inputs = [radiance, reference, corpus, *(model for _, model in candidates)]
    inputs += [path.resolve() for path in (args.provenance, args.calibration_manifest) if path]
    # Plugin symlinks can point outside the home; protect the target bytes too.
    inputs += [path.resolve() for sub in ("kernels", "architectures", "quantizers")
               for path in (home / sub).rglob("*.so")]
    destinations = [out / "run.json"]
    for name, _ in candidates:
        report = out / f"{name}-kld.json"
        destinations.extend((report, Path(str(report) + ".rows")))
    if out == home or home in out.parents:
        p.error("output directory must not be inside the plugin home")
    for dest in destinations:
        if dest.is_symlink() or dest.resolve() in inputs or (dest.exists() and not dest.is_file()):
            p.error(f"output path collides with an input, symlink or non-file: {dest}")
    if ref_dir.is_symlink():
        p.error("reference output directory must not be a symlink")
    if any(path == ref_dir or ref_dir in path.parents for path in inputs):
        p.error("an input is inside the reference output directory")
    out.mkdir(parents=True, exist_ok=True)
    # A failed retry must not leave a previous run's receipt looking like its result.
    (out / "run.json").unlink(missing_ok=True)

    def snapshot_inputs():
        return {
            "reference_model": str(reference), "reference_sha256": sha256(reference),
            "corpus": str(corpus), "corpus_sha256": sha256(corpus),
            "corpus_fnv1a64": fnv1a64(corpus), "radiance": str(radiance),
            "radiance_sha256": sha256(radiance), "plugins": plugin_hashes(home),
            "arguments": list(common),
        }

    reference_inputs = snapshot_inputs()
    plugins = reference_inputs["plugins"]

    def verify_inputs():
        if snapshot_inputs() != reference_inputs:
            raise ValueError("reference inputs changed during recording or scoring")

    ref_meta_path = ref_dir / "kld.json"
    if args.reuse_reference:
        if not ref_meta_path.is_file():
            p.error(f"--reuse-reference requested but {ref_meta_path} is missing")
        ref_meta = json.loads(ref_meta_path.read_text())
    else:
        if ref_dir.exists():
            p.error(f"reference directory already exists: {ref_dir}; choose another --out or use --reuse-reference")
        ref_dir.mkdir(parents=True)
        env = os.environ.copy()
        subprocess.run(record, env=env, check=True)
        # Check before trusting or binding any output from the recorder.
        verify_inputs()
        ref_meta = json.loads(ref_meta_path.read_text())

    validate_reference(ref_meta, reference, corpus)
    # Never publish a binding if the recorder may have observed different input bytes.
    verify_inputs()
    bound_reference = reference_provenance(ref_dir, reference_inputs, args.reuse_reference)

    run = {
        "reference_model": str(reference), "reference_sha256": reference_inputs["reference_sha256"],
        "reference_dir": str(ref_dir), "reference_form": "top-128 + tail",
        "corpus": str(corpus), "corpus_sha256": reference_inputs["corpus_sha256"],
        "corpus_fnv1a64": reference_inputs["corpus_fnv1a64"], "evaluation_manifest": None,
        "radiance": str(radiance), "radiance_sha256": reference_inputs["radiance_sha256"],
        "radiance_home": str(home), "kernel_selector": args.kernels,
        "tp": args.tp, "kv_cache_dtype": "bf16", "max_num_seqs": 1,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_headroom_mib": args.gpu_headroom_mib,
        "speculation": 0, "prefix_cache": False,
        "reference_manifest": ref_meta, "reference_provenance": bound_reference,
        **plugins, "candidates": {},
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

    expected_rows = ref_meta.get("rows")
    with ScoreStaging(out) as staging:
        staged_outputs = []
        for name, model in candidates:
            report = out / f"{name}-kld.json"
            row_path = Path(str(report) + ".rows")
            staged_report = staging.path / report.name
            staged_rows = Path(str(staged_report) + ".rows")
            command = [str(radiance), "--model", str(model), *common, "--kld-ref", str(ref_dir),
                       "--kld-out", str(staged_report)]
            candidate_sha256 = sha256(model)
            subprocess.run(command, env=os.environ.copy(), check=True)
            scored = json.loads(staged_report.read_text())
            validate_report(scored, ref_dir, reference, corpus, model, expected_rows)
            if sha256(model) != candidate_sha256:
                p.error(f"{name}: candidate changed during scoring")
            values = scored["all"]["kld"]
            raw = staged_rows.read_bytes()
            if len(raw) != expected_rows * 16:
                p.error(f"{name}: unexpected per-position rows size {len(raw)}")
            row_values = struct.unpack("<" + "f" * (expected_rows * 4), raw)
            kl = [row_values[i * 4] for i in range(expected_rows)]
            # core/kld.cpp's A/S - log(S) can retain tiny signed roundoff near zero.
            if not all(math.isfinite(value) for value in kl):
                p.error(f"{name}: non-finite coarsened KL row")
            run["candidates"][name] = {"model": str(model), "sha256": candidate_sha256,
                                       "report": str(report), "report_sha256": sha256(staged_report),
                                       "rows": str(row_path), "rows_sha256": sha256(staged_rows),
                                       "metrics": scored["all"],
                                       "positions_over_1_nat": sum(x > 1.0 for x in kl)}
            staged_outputs.extend(((staged_report, report), (staged_rows, row_path)))
            print(f"{name}: {expected_rows} positions; ref_form={scored['ref_form']}; "
                  f"mean={values['mean']:.6g}, p99={values['p99']:.6g}, "
                  f"p99.9={values['p99_9']:.6g}, max={values['max']:.6g}, "
                  f">1 nat={run['candidates'][name]['positions_over_1_nat']}")

        verify_inputs()
        reference_provenance(ref_dir, reference_inputs, reuse=True)
        receipt_temp = staging.path / "new-run.json"
        receipt_temp.write_text(json.dumps(run, indent=2) + "\n")
        backups = []
        promoted = []
        try:
            for index, (source, destination) in enumerate(staged_outputs):
                backup = staging.path / f"backup-{index}"
                if destination.exists():
                    os.replace(destination, backup)
                    backups.append((backup, destination))
                os.replace(source, destination)
                promoted.append(destination)
            os.replace(receipt_temp, out / "run.json")
        except BaseException as publication_error:
            # Continue after individual rollback errors; retain all remaining backups on failure.
            rollback_errors = []
            for destination in reversed(promoted):
                try:
                    destination.unlink(missing_ok=True)
                except OSError as exc:
                    rollback_errors.append(f"remove {destination}: {exc}")
            for backup, destination in reversed(backups):
                try:
                    os.replace(backup, destination)
                except OSError as exc:
                    rollback_errors.append(f"restore {backup} -> {destination}: {exc}")
            if rollback_errors:
                staging.keep = True
                raise RuntimeError(f"score publication failed ({publication_error}); rollback "
                                   f"incomplete; recover backups in {staging.path}: "
                                   + "; ".join(rollback_errors)) from publication_error
            raise
    print(f"Run receipt: {out / 'run.json'}")
    print("Per-position rows are available as <name>-kld.json.rows; replay severe positions with "
          "tests/spark25_kld_tail.py. These are top-128-plus-tail KL values, not full-vocabulary KL.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"spark25_quality: {exc}", file=sys.stderr)
        sys.exit(1)
