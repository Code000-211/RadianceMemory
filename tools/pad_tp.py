#!/usr/bin/env python3
"""pad_tp.py — pad a Llama/Qwen-style HF safetensors checkpoint to a geometry that divides by a
tensor-parallel world size, with zero weights, so the padded model computes exactly what the
original does.

    pad_tp.py SRC DST --heads 36 --kv-heads 9 [--ff 8448] [--gdn-k 18 --gdn-v 54]

What is padded and why it is exact (fp32-accumulated, zero terms add nothing):
  attention   k_proj / v_proj gain zero rows (new kv heads); q_proj gains zero rows so every kv
              group has the new ratio gqa' = heads'/kv_heads' -- a group that existed keeps its real
              q heads first, then zero q heads; a new group is all zero. o_proj gains zero columns
              in the same head order. A zero q head attends uniformly, but its o_proj column is
              zero, so it contributes 0; a zero kv head stores zeros, read only by zero q heads.
  mlp         gate_proj / up_proj gain zero rows, down_proj zero columns: silu(0)*0 = 0.
  gdn         (Qwen3-Next, --gdn-k/--gdn-v) in_proj_qkvz / in_proj_ba gain zero rows per the
              fused layout, conv1d zero channels, A_log / dt_bias zero heads, out_proj zero columns.
Everything with n_embd as its only width (norms, embeddings, lm_head) is untouched. config.json is
rewritten with the new counts and an explicit head_dim. Dummies go LAST, as tcclaviger's vLLM tp_padding
does, so rank w-1 carries them.

bf16 is handled as raw u16 (no arithmetic happens here), so numpy alone suffices.
"""
import argparse, json, os, re, shutil, struct, subprocess, sys
import numpy as np

DT = {"BF16": (np.uint16, 2), "F16": (np.float16, 2), "F32": (np.float32, 4), "I64": (np.int64, 8),
      "I32": (np.int32, 4), "U8": (np.uint8, 1), "BOOL": (np.bool_, 1), "F8_E4M3": (np.uint8, 1)}

def st_read(path):
    """-> (header dict with tensor meta, raw bytes mmap). Tiny safetensors reader."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    return hdr, data

def st_tensor(hdr, data, name):
    m = hdr[name]; a, b = m["data_offsets"]; dt, _ = DT[m["dtype"]]
    return np.frombuffer(data[a:b], dtype=dt).reshape(m["shape"])

def st_write_stream(path, meta, items):
    """items: iterable of (name, dtype_str, fn) where fn() returns the ndarray. ONE tensor is live at
    a time (the 12:42 version held a whole 4.7 GB shard and was memcg-OOM-killed beside the daily):
    the header needs every shape and byte count first, so the pass is made twice -- shapes are
    cheap to compute from the source header, so items carry a `shape_of()` too."""
    hdr = {}; off = 0; order = []
    for name, dts, shape, nbytes, fn in items:
        hdr[name] = {"dtype": dts, "shape": list(shape), "data_offsets": [off, off + nbytes]}
        off += nbytes; order.append((name, fn))
    if meta: hdr["__metadata__"] = meta
    h = json.dumps(hdr, separators=(",", ":")).encode(); h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h)
        for name, fn in order:
            arr = np.ascontiguousarray(fn()); f.write(arr.tobytes()); del arr

def head_map(n_head, n_kv, n_head2, n_kv2):
    """For every NEW q-head slot, the OLD q head index or -1 (zero). Group g' (< n_kv) keeps its real
    heads first, then zeros; group g' >= n_kv is all zero."""
    gqa, gqa2 = n_head // n_kv, n_head2 // n_kv2
    assert n_head % n_kv == 0 and n_head2 % n_kv2 == 0 and gqa2 >= gqa and n_kv2 >= n_kv, "ratio/kv must not shrink"
    out = []
    for g in range(n_kv2):
        for j in range(gqa2):
            out.append(g * gqa + j if (g < n_kv and j < gqa) else -1)
    return out

def take_rows(w, idx, hd):
    """rows of w grouped by hd per head: new row block i = old block idx[i], or zeros."""
    out = np.zeros((len(idx) * hd,) + w.shape[1:], dtype=w.dtype)
    for i, o in enumerate(idx):
        if o >= 0: out[i * hd:(i + 1) * hd] = w[o * hd:(o + 1) * hd]
    return out

def take_cols(w, idx, hd):
    out = np.zeros(w.shape[:1] + (len(idx) * hd,) + w.shape[2:], dtype=w.dtype)
    for i, o in enumerate(idx):
        if o >= 0: out[:, i * hd:(i + 1) * hd] = w[:, o * hd:(o + 1) * hd]
    return out

def pad_rows(w, n2):
    out = np.zeros((n2,) + w.shape[1:], dtype=w.dtype); out[:w.shape[0]] = w; return out

def pad_cols(w, n2):
    out = np.zeros(w.shape[:1] + (n2,) + w.shape[2:], dtype=w.dtype); out[:, :w.shape[1]] = w; return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--heads", type=int, required=True); ap.add_argument("--kv-heads", type=int, required=True)
    ap.add_argument("--ff", type=int, default=0, help="new intermediate_size for DENSE mlp (0 = unchanged; MoE experts are never padded)")
    ap.add_argument("--gdn-k", type=int, default=0); ap.add_argument("--gdn-v", type=int, default=0)
    ap.add_argument("--sparse", action="store_true",
                    help="write ONLY the padded tensors, into new shards named to sort first; reflink every original "
                         "shard into dst unchanged. Radiance lists shards from the index, sorted by name, and keeps the "
                         "first tensor of a name (core/format/checkpoint.cpp emplace), so the padded copy shadows the "
                         "original. Costs the padded bytes only -- the 336 GB QFN source is not copied.")
    a = ap.parse_args()
    cfg_all = json.load(open(os.path.join(a.src, "config.json")))
    cfg = cfg_all["text_config"] if "text_config" in cfg_all else cfg_all      # Qwen3.8-Flash-Next nests it
    n_head, n_kv = cfg["num_attention_heads"], cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    n_embd = cfg["hidden_size"]; hd = cfg.get("head_dim") or n_embd // n_head
    n_ff = cfg.get("intermediate_size", 0); n_ff2 = a.ff or n_ff      # MoE-only models (QFN) have none
    qmap = head_map(n_head, n_kv, a.heads, a.kv_heads)
    kmap = list(range(n_kv)) + [-1] * (a.kv_heads - n_kv)
    gk, gv = cfg.get("linear_num_key_heads"), cfg.get("linear_num_value_heads")
    gkd, gvd = cfg.get("linear_key_head_dim"), cfg.get("linear_value_head_dim")
    gk2, gv2 = a.gdn_k or gk, a.gdn_v or gv
    print(f"attention {n_head}/{n_kv} (gqa {n_head//n_kv}) @ {hd} -> {a.heads}/{a.kv_heads} (gqa {a.heads//a.kv_heads}); "
          f"dense ff {n_ff} -> {n_ff2}" + (f"; gdn K {gk}->{gk2} V {gv}->{gv2}" if gk else ""), flush=True)
    os.makedirs(a.dst, exist_ok=True)
    idx_path = os.path.join(a.src, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx_path))["weight_map"].values())) if os.path.exists(idx_path) else ["model.safetensors"]
    # Shard names come from the checkpoint's own index: each must be a bare file name in it.
    for fn in files:
        if not isinstance(fn, str) or not fn or fn != os.path.basename(fn) or fn in (".", ".."):
            sys.exit(f"{idx_path}: shard name {fn!r} is not a file name in the checkpoint directory")

    def parts(w_, sizes, news, axis=0):
        """pad each consecutive segment of `sizes` along `axis` to `news` (zeros after the real rows)."""
        o = 0; outs = []
        for sz, s2 in zip(sizes, news):
            seg = np.take(w_, range(o, o + sz), axis=axis); o += sz
            pad = [(0, 0)] * w_.ndim; pad[axis] = (0, s2 - sz); outs.append(np.pad(seg, pad))
        assert o == w_.shape[axis], f"segments {sizes} do not cover axis {axis} of shape {w_.shape}"
        return np.concatenate(outs, axis=axis)

    def plan(name, shape):
        """-> (kind, transform) for a tensor, by name and SOURCE shape. None = untouched."""
        if re.search(r"self_attn\.q_proj\.(weight|bias)$", name):
            # Qwen3-Next gated attention fuses q and its gate: rows = n_head * 2 * hd, per head [q | gate].
            # A plain q_proj has n_head * hd rows. Either way one head is one block of rows.
            blk = shape[0] // n_head
            assert blk in (hd, 2 * hd), f"{name}: {shape[0]} rows is neither {n_head}*{hd} nor {n_head}*{2*hd}"
            return "q", lambda w: take_rows(w, qmap, blk)
        if re.search(r"self_attn\.[kv]_proj\.(weight|bias)$", name):   return "kv", lambda w: take_rows(w, kmap, hd)
        if re.search(r"self_attn\.o_proj\.weight$", name):             return "o",  lambda w: take_cols(w, qmap, hd)
        if n_ff2 != n_ff and re.search(r"mlp\.(gate|up)_proj\.weight$", name): return "ff", lambda w: pad_rows(w, n_ff2)
        if n_ff2 != n_ff and re.search(r"mlp\.down_proj\.weight$", name):      return "ff", lambda w: pad_cols(w, n_ff2)
        if gk and re.search(r"linear_attn\.", name) and (gk2 != gk or gv2 != gv):
            # Qwen3-Next GDN, both spellings. Fused: in_proj_qkvz rows [q gk*gkd][k gk*gkd][v gv*gvd][z gv*gvd],
            # in_proj_ba [b gv][a gv]. Separate (Qwen3.8-Flash-Next): in_proj_qkv [q][k][v], in_proj_z [z],
            # in_proj_a / in_proj_b [gv] each. conv1d channels = q,k,v. A_log / dt_bias per v head.
            # out_proj columns = gv*gvd. norm is per gvd and untouched.
            qkv_s, qkv_n = [gk*gkd, gk*gkd, gv*gvd], [gk2*gkd, gk2*gkd, gv2*gvd]
            if re.search(r"in_proj_qkvz\.weight$", name): return "gdn", lambda w: parts(w, qkv_s + [gv*gvd], qkv_n + [gv2*gvd])
            if re.search(r"in_proj_qkv\.weight$", name):  return "gdn", lambda w: parts(w, qkv_s, qkv_n)
            if re.search(r"in_proj_z\.weight$", name):    return "gdn", lambda w: pad_rows(w, gv2 * gvd)
            if re.search(r"in_proj_ba\.weight$", name):   return "gdn", lambda w: parts(w, [gv, gv], [gv2, gv2])
            if re.search(r"in_proj_[ab]\.weight$", name): return "gdn", lambda w: pad_rows(w, gv2)
            if re.search(r"conv1d\.(weight|bias)$", name): return "gdn", lambda w: parts(w, qkv_s, qkv_n)
            if re.search(r"(A_log|dt_bias)$", name):       return "gdn", lambda w: pad_rows(w, gv2)
            if re.search(r"out_proj\.weight$", name):      return "gdn", lambda w: pad_cols(w, gv2 * gvd)
        return None, None

    weight_map = {}; total = 0; touched = {}; sparse_items = []; sparse_bytes = 0
    for fn in files:
        hdr, data = st_read(os.path.join(a.src, fn)); meta = hdr.pop("__metadata__", None); items = []
        for name in hdr:
            dts = hdr[name]["dtype"]; esz = DT[dts][1]; src_shape = tuple(hdr[name]["shape"])
            kind, tf = plan(name, src_shape)
            if kind:
                shape = tf(np.zeros(src_shape, dtype=np.uint8)).shape      # from the source shape alone; cheap
                touched[kind] = touched.get(kind, 0) + 1
            else:
                shape = src_shape
            nbytes = int(np.prod(shape)) * esz; total += nbytes
            item = (name, dts, shape, nbytes, (lambda h=hdr, d=data, n=name, t=tf: t(st_tensor(h, d, n)) if t else st_tensor(h, d, n)))
            if a.sparse:
                if kind: sparse_items.append(item); sparse_bytes += nbytes
                weight_map[name] = fn                 # overridden below for padded names
            else:
                items.append(item); weight_map[name] = fn
        if a.sparse:
            dstf = os.path.join(a.dst, fn)
            if not os.path.exists(dstf):
                r = subprocess.run(["cp", "--reflink=always", os.path.join(a.src, fn), dstf],
                                   stderr=subprocess.DEVNULL).returncode
                if r: os.link(os.path.join(a.src, fn), dstf)      # not btrfs/xfs: hard link, still no copy
        else:
            st_write_stream(os.path.join(a.dst, fn), meta, items); print(f"  wrote {fn} ({len(items)} tensors)", flush=True)
    if a.sparse:
        # new shards, ~2 GB each, named to sort before every original ("model-00000-…" < "model-00001-…")
        CH = 2 << 30; chunks = [[]]; acc = 0
        for it in sparse_items:
            if acc + it[3] > CH and chunks[-1]: chunks.append([]); acc = 0
            chunks[-1].append(it); acc += it[3]
        for k, ch in enumerate(chunks):
            fn = f"model-00000-of-{len(files):05d}-tp3pad-{k:03d}.safetensors"
            st_write_stream(os.path.join(a.dst, fn), {"tp3pad": "zero-padded heads; shadows the originals by name order"}, ch)
            for it in ch: weight_map[it[0]] = fn
            print(f"  wrote {fn} ({len(ch)} tensors, {sum(i[3] for i in ch)/1e9:.2f} GB)", flush=True)
        print(f"sparse: {len(sparse_items)} padded tensors, {sparse_bytes/1e9:.2f} GB new; {len(files)} shards reflinked", flush=True)
    if len(files) > 1 or a.sparse:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
                  open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=2)
    cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"] = a.heads, a.kv_heads, hd
    if n_ff: cfg["intermediate_size"] = n_ff2
    if gk: cfg["linear_num_key_heads"], cfg["linear_num_value_heads"] = gk2, gv2
    cfg_all["_tp_padding"] = {"from": {"heads": n_head, "kv_heads": n_kv, "ff": n_ff, "gdn_k": gk, "gdn_v": gv},
                              "to": {"heads": a.heads, "kv_heads": a.kv_heads, "ff": n_ff2, "gdn_k": gk2, "gdn_v": gv2},
                              "zero_weights": True, "dummies": "last", "sparse": bool(a.sparse)}
    json.dump(cfg_all, open(os.path.join(a.dst, "config.json"), "w"), indent=2)
    for f in os.listdir(a.src):
        if (f.endswith(".json") and f not in ("config.json", "model.safetensors.index.json")) or f.endswith((".txt", ".model", ".jinja", ".md")):
            shutil.copy2(os.path.join(a.src, f), os.path.join(a.dst, f))
    print(f"padded tensors by kind: {touched}; total {total/1e9:.2f} GB -> {a.dst}", flush=True)

if __name__ == "__main__":
    main()
