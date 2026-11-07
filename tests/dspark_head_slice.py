#!/usr/bin/env python3
"""Extract a read-only BF16 head slice for dspark_device_test --head-weights.

The device probe expects the first 1025 rows of a [131072, 2560] BF16 embedding.
Output is created exclusively, never overwriting an existing artifact or model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("safetensors", type=Path)
p.add_argument("output", type=Path)
args = p.parse_args()
with args.safetensors.open("rb") as f:
    raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("short safetensors header")
    n = struct.unpack("<Q", raw)[0]
    if n > 100_000_000:
        raise ValueError("unreasonably large safetensors header")
    header = json.loads(f.read(n))
    t = header["model.embedding.weight"]
    if t["dtype"] != "BF16" or t["shape"] != [131072, 2560]:
        raise ValueError("expected Spark BF16 embedding [131072, 2560]")
    start, end = t["data_offsets"]
    size = 1025 * 2560 * 2
    if end - start < size:
        raise ValueError("short tensor")
    f.seek(8 + n + start)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("short read")
with args.output.open("xb") as f:
    f.write(data)
print(f"{args.safetensors}: model.embedding.weight rows 0..1024; "
      f"bytes={len(data)} sha256={hashlib.sha256(data).hexdigest()} -> {args.output}")
