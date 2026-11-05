"""Artifact and reference bindings shared by the real-container Spark probes."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fnv1a64(path):
    # core/kld.cpp hashes each getline without its terminating LF.
    value = 1469598103934665603
    with Path(path).open("rb") as stream:
        for line in stream:
            if line.endswith(b"\n"):
                line = line[:-1]
            for byte in line:
                value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return f"{value:016x}"


def plugin_hashes(home):
    home = Path(home).resolve()
    if ":" in str(home):
        raise ValueError("these probes require one explicit RADIANCE_HOME, not a search path")
    result = {}
    for directory, key in (("kernels", "kernel_shared_objects"),
                           ("architectures", "architecture_shared_objects"),
                           ("quantizers", "quantizer_shared_objects")):
        result[key] = {str(path.relative_to(home)): sha256(path)
                       for path in sorted((home / directory).rglob("*.so"))}
        if directory != "quantizers" and not result[key]:
            raise ValueError(f"no shared objects found in {home / directory}")
    return result


def same_path(value, expected, label):
    if not isinstance(value, str) or not value or Path(value).resolve() != Path(expected).resolve():
        raise ValueError(f"{label} does not match {expected}")


def validate_reference(meta, reference, corpus):
    if meta.get("format") != "radiance-kld-top-1" or meta.get("top") != 128:
        raise ValueError("reference must be radiance-kld-top-1 with top=128")
    same_path(meta.get("model"), reference, "reference model")
    same_path(meta.get("corpus"), corpus, "reference corpus")
    if meta.get("corpus_fnv1a64") != fnv1a64(corpus):
        raise ValueError("reference corpus hash does not match the corpus")
    if type(meta.get("rows")) is not int or meta["rows"] <= 0:
        raise ValueError("reference must have a positive integer row count")


def validate_report(report, ref_dir, reference, corpus, candidate, rows):
    if report.get("ref_form") != "top-128 + tail":
        raise ValueError("report must use the top-128-plus-tail reference")
    for key, expected in (("ref_dir", ref_dir), ("reference", reference),
                          ("corpus", corpus), ("model", candidate)):
        same_path(report.get(key), expected, f"report {key}")
    if report.get("all", {}).get("positions") != rows:
        raise ValueError("report row count does not match the reference")


def reference_provenance(ref_dir, inputs, reuse):
    """Reject legacy/unbound references rather than relabel their baseline in a new receipt."""
    ref_dir = Path(ref_dir)
    path = ref_dir / "provenance.json"
    current = {"format": "spark25-reference-provenance-1", "inputs": inputs,
               "files": {name: sha256(ref_dir / name)
                         for name in ("kld.json", "tokens.i32", "top.bin", "rows.f32")}}
    if reuse:
        if not path.is_file():
            raise ValueError("reference lacks provenance.json; record a fresh bound reference")
        if json.loads(path.read_text()) != current:
            raise ValueError("reused reference baseline, runtime, settings or data identity differs")
    else:
        if path.exists():
            raise ValueError(f"refusing to replace existing reference provenance: {path}")
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(current, indent=2) + "\n")
        temporary.replace(path)
    return current
