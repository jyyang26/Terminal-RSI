#!/usr/bin/env python3
"""Repair a torch_dist->HF converted Qwen3.5-MoE checkpoint so it matches origin.

`convert_torch_dist_to_hf_parallel.py` produces a checkpoint that is NOT
loadable as-is for this architecture. Two independent defects:

1. MISSING keys. The torch_dist checkpoint only holds what Megatron trained --
   the language tower. `mtp.*` (785 keys) and `model.visual.*` (333 keys) are
   never written, so they must be copied back from the origin HF checkpoint.
   Without the visual tower a Qwen3_5MoeForConditionalGeneration is broken.

2. WRONG expert layout. The converter walks the "ungrouped" branch of
   `megatron_to_hf/qwen3_5.py` (line 93) and emits 36864 per-expert tensors
   (`...experts.{i}.gate_proj.weight` etc). But transformers' Qwen3_5MoeExperts
   declares FUSED nn.Parameters:
       gate_up_proj  [num_experts, 2*intermediate, hidden]
       down_proj     [num_experts, hidden, intermediate]
   (modeling_qwen3_5_moe.py:723-724). Per-expert keys simply do not match any
   parameter, so every expert would silently stay at its randomly-initialised
   value -- catastrophic, and invisible unless you diff the key sets.

Fusion layout is pinned by the two directions that already exist in-tree:
  * forward (mcore->hf), qwen3_5.py:99  `gate, up = param.chunk(2, dim=0)`
  * reverse (hf->mcore), mbridge/qwen3_5.py:302  `expert_w = w[global_expert_id]`
so fused[e] == cat([gate_e, up_e], dim=0), and down_proj stacks directly.

This script rewrites the whole checkpoint: fused experts + carried-over text
weights + mtp/visual restored from origin, then a fresh index.json. The result
is verified against origin key-by-key.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import safetensors.torch
import torch
from safetensors import safe_open

EXPERT_RE = "mlp.experts."


def load_index(model_dir: Path) -> dict:
    with (model_dir / "model.safetensors.index.json").open() as f:
        return json.load(f)


class ShardWriter:
    """Accumulate tensors and flush them into fixed-size safetensors shards."""

    def __init__(self, out_dir: Path, chunk_size: int) -> None:
        self.out_dir = out_dir
        self.chunk_size = chunk_size
        self.buf: dict[str, torch.Tensor] = {}
        self.buf_bytes = 0
        self.shard_idx = 0
        self.weight_map: dict[str, str] = {}
        self.total_size = 0
        self.tmp_names: list[str] = []

    def add(self, key: str, tensor: torch.Tensor) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        if self.buf and self.buf_bytes + nbytes > self.chunk_size:
            self.flush()
        self.buf[key] = tensor
        self.buf_bytes += nbytes
        self.total_size += nbytes

    def flush(self) -> None:
        if not self.buf:
            return
        self.shard_idx += 1
        name = f"model-{self.shard_idx:05d}-of-TOTAL.safetensors"
        safetensors.torch.save_file(self.buf, self.out_dir / name, metadata={"format": "pt"})
        for key in self.buf:
            self.weight_map[key] = name
        self.tmp_names.append(name)
        self.buf = {}
        self.buf_bytes = 0

    def finalize(self) -> None:
        """Rename TOTAL placeholders now that the shard count is known."""
        self.flush()
        total = self.shard_idx
        rename = {}
        for old in self.tmp_names:
            new = old.replace("-of-TOTAL", f"-of-{total:05d}")
            (self.out_dir / old).rename(self.out_dir / new)
            rename[old] = new
        self.weight_map = {k: rename[v] for k, v in self.weight_map.items()}


def fuse_experts(converted_dir: Path, conv_map: dict[str, str], writer: ShardWriter) -> int:
    """Stack per-expert tensors back into the fused parameters HF expects."""
    # layer -> kind -> {expert_idx: key}
    layers: dict[int, dict[str, dict[int, str]]] = defaultdict(lambda: defaultdict(dict))
    for key in conv_map:
        if EXPERT_RE not in key or not key.endswith(".weight"):
            continue
        parts = key.split(".")
        # model.language_model.layers.<L>.mlp.experts.<E>.<kind>.weight
        try:
            li = parts.index("layers")
            ei = parts.index("experts")
        except ValueError:
            continue
        if not parts[ei + 1].isdigit():
            continue  # already-fused or shared_experts
        layers[int(parts[li + 1])][parts[ei + 2]][int(parts[ei + 1])] = key

    fused = 0
    for layer in sorted(layers):
        kinds = layers[layer]
        prefix = f"model.language_model.layers.{layer}.mlp.experts"
        n_exp = len(kinds["gate_proj"])
        assert n_exp and len(kinds["up_proj"]) == len(kinds["down_proj"]) == n_exp, (
            f"layer {layer}: uneven expert counts " f"{ {k: len(v) for k, v in kinds.items()} }"
        )

        # Read every expert of this layer once, grouped by source shard so each
        # file is opened a single time.
        need = [k for kind in ("gate_proj", "up_proj", "down_proj") for k in kinds[kind].values()]
        by_file: dict[str, list[str]] = defaultdict(list)
        for k in need:
            by_file[conv_map[k]].append(k)
        cache: dict[str, torch.Tensor] = {}
        for fname, keys in by_file.items():
            with safe_open(converted_dir / fname, framework="pt", device="cpu") as f:
                for k in keys:
                    cache[k] = f.get_tensor(k)

        gate_up = torch.stack(
            [torch.cat([cache[kinds["gate_proj"][e]], cache[kinds["up_proj"][e]]], dim=0) for e in range(n_exp)]
        )
        down = torch.stack([cache[kinds["down_proj"][e]] for e in range(n_exp)])
        writer.add(f"{prefix}.gate_up_proj", gate_up)
        writer.add(f"{prefix}.down_proj", down)
        fused += 2
        del cache
    return fused


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin-hf-dir", required=True, type=Path)
    ap.add_argument("--converted-hf-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--chunk-size", type=int, default=5 * 1024**3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    origin_map = load_index(args.origin_hf_dir)["weight_map"]
    conv_index = load_index(args.converted_hf_dir)
    conv_map = conv_index["weight_map"]

    per_expert = {k for k in conv_map if EXPERT_RE in k and k.split(".")[-3].isdigit()}
    carry = sorted(set(conv_map) - per_expert)  # trained text weights, kept verbatim
    # What fuse_experts() will produce. Only the language tower is fused: origin
    # keeps mtp.* experts in PER-EXPERT form (mtp.layers.N.mlp.experts.E.*), so
    # those must be restored from origin like any other untrained key -- an
    # earlier version of this script excluded every `mlp.experts.` key here and
    # silently dropped all 768 of them.
    fused_targets = sorted(
        k
        for k in origin_map
        if EXPERT_RE in k and k.endswith((".gate_up_proj", ".down_proj")) and not k.split(".")[-2].isdigit()
    )
    missing = sorted(set(origin_map) - set(carry) - set(fused_targets))

    print(
        "before",
        f"origin={len(origin_map)}",
        f"converted={len(conv_map)}",
        f"per_expert={len(per_expert)}",
        f"carry={len(carry)}",
        f"restore_from_origin={len(missing)}",
        f"fuse_targets={len(fused_targets)}",
        flush=True,
    )

    if args.output_dir.exists():
        if not args.force:
            raise SystemExit(f"output exists, pass --force: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    writer = ShardWriter(args.output_dir, args.chunk_size)

    # 1) fused experts, rebuilt from the trained per-expert tensors
    n_fused = fuse_experts(args.converted_hf_dir, conv_map, writer)
    print(f"fused {n_fused} expert tensors", flush=True)

    # 2) trained non-expert weights, carried over untouched -- except that the
    #    dtype must match origin. Origin keeps 72 GDN tensors
    #    (linear_attn.A_log, linear_attn.norm.weight) in fp32 while Megatron
    #    trains and stores them as bf16, so a straight copy silently halves
    #    their width and makes total_size disagree with origin. Cast back to
    #    origin's dtype: the values are already bf16-rounded upstream, so this
    #    recovers the declared precision/layout, not lost information.
    origin_dtypes = {}
    by_file: dict[str, list[str]] = defaultdict(list)
    for k in carry:
        by_file[conv_map[k]].append(k)
    origin_by_file: dict[str, list[str]] = defaultdict(list)
    for k in carry:
        origin_by_file[origin_map[k]].append(k)
    for fname in sorted(origin_by_file):
        with safe_open(args.origin_hf_dir / fname, framework="pt", device="cpu") as f:
            for k in origin_by_file[fname]:
                origin_dtypes[k] = f.get_slice(k).get_dtype()
    dtype_map = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}
    n_cast = 0
    for fname in sorted(by_file):
        with safe_open(args.converted_hf_dir / fname, framework="pt", device="cpu") as f:
            for k in sorted(by_file[fname]):
                t = f.get_tensor(k)
                want = dtype_map.get(origin_dtypes.get(k, ""))
                if want is not None and t.dtype != want:
                    t = t.to(want)
                    n_cast += 1
                writer.add(k, t)
    print(f"carried {len(carry)} trained tensors (cast {n_cast} to origin dtype)", flush=True)

    # 3) mtp.* / model.visual.* restored from origin -- never trained, never
    #    written by the converter, and mandatory for the model to load.
    by_file = defaultdict(list)
    for k in missing:
        by_file[origin_map[k]].append(k)
    for fname in sorted(by_file):
        with safe_open(args.origin_hf_dir / fname, framework="pt", device="cpu") as f:
            for k in sorted(by_file[fname]):
                writer.add(k, f.get_tensor(k))
    print(f"restored {len(missing)} tensors from origin", flush=True)

    writer.finalize()

    got, want = set(writer.weight_map), set(origin_map)
    if got != want:
        raise SystemExit(f"key mismatch: missing={sorted(want - got)[:8]} extra={sorted(got - want)[:8]}")

    # total_size must match origin exactly -- but ONLY for an actor checkpoint.
    # It is the cheapest end-to-end proof that every tensor landed with origin's
    # shape AND dtype (the fp32 GDN tensors above were caught by exactly this).
    #
    # A CRITIC checkpoint legitimately fails it: its output_layer is a value head
    # (observed [1, 3072]) where origin carries the LM head [248320, 3072], so the
    # totals differ by exactly one lm_head -- 1525671936 bytes, which is what the
    # first critic sweep reported. Detect that case by comparing the shape of the
    # head we actually wrote against origin's, and drop the delta from the
    # expectation instead of failing. Every other tensor is still held to origin's
    # exact byte count, so a real shape/dtype regression is still caught.
    origin_index = load_index(args.origin_hf_dir)
    origin_total = int(origin_index.get("metadata", {}).get("total_size", 0))
    expected_total = origin_total
    head = "lm_head.weight"
    if origin_total and head in origin_map and head in writer.weight_map:
        with safe_open(args.origin_hf_dir / origin_map[head], framework="pt", device="cpu") as f:
            origin_head_bytes = f.get_slice(head).get_shape()
        with safe_open(args.output_dir / writer.weight_map[head], framework="pt", device="cpu") as f:
            ours_head = f.get_tensor(head)
        origin_numel = 1
        for d in origin_head_bytes:
            origin_numel *= d
        if ours_head.numel() != origin_numel:
            itemsize = ours_head.element_size()
            expected_total -= (origin_numel - ours_head.numel()) * itemsize
            print(
                f"critic head detected: {head} is {tuple(ours_head.shape)} vs origin "
                f"{tuple(origin_head_bytes)}; expected total_size adjusted to {expected_total}",
                flush=True,
            )
    if origin_total and writer.total_size != expected_total:
        raise SystemExit(
            f"total_size mismatch: expected={expected_total} repaired={writer.total_size} "
            f"delta={expected_total - writer.total_size}"
        )

    (args.output_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": writer.total_size}, "weight_map": dict(sorted(writer.weight_map.items()))},
            indent=2,
        )
        + "\n"
    )
    for src in args.origin_hf_dir.iterdir():
        if src.is_file() and not src.name.endswith(".safetensors") and src.name != "model.safetensors.index.json":
            shutil.copy2(src, args.output_dir / src.name)

    print(
        "after",
        f"keys={len(writer.weight_map)}",
        f"shards={writer.shard_idx}",
        f"total_size={writer.total_size}",
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
