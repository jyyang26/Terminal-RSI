#!/usr/bin/env python3
"""Repair a converted HF checkpoint by adding missing weights from origin HF.

The converted checkpoint keeps newly converted text weights. Any key absent
from it is copied from the original HF checkpoint into extra safetensors shards.
Use --include-prefix to run isolation experiments, e.g. only mtp.* or only
model.visual.*.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import safetensors.torch
from safetensors import safe_open


def load_weight_map(model_dir: Path) -> dict[str, str]:
    with (model_dir / "model.safetensors.index.json").open() as f:
        return json.load(f)["weight_map"]


def copy_converted_dir(converted_hf_dir: Path, output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise SystemExit(f"output exists, pass --force to overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for src in converted_hf_dir.iterdir():
        dst = output_dir / src.name
        if src.is_dir():
            shutil.copytree(src, dst)
            continue
        if src.name == "model.safetensors.index.json":
            shutil.copy2(src, dst)
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def add_missing_shards(
    origin_hf_dir: Path,
    missing_keys: list[str],
    origin_map: dict[str, str],
    output_dir: Path,
    chunk_size: int,
) -> tuple[dict[str, str], int]:
    added_map: dict[str, str] = {}
    shard: dict[str, object] = {}
    shard_size = 0
    shard_idx = 1
    total_size = 0

    def flush() -> None:
        nonlocal shard, shard_size, shard_idx
        if not shard:
            return
        filename = f"model-missing-from-origin-{shard_idx:05d}.safetensors"
        safetensors.torch.save_file(shard, output_dir / filename)
        for key in shard:
            added_map[key] = filename
        shard = {}
        shard_size = 0
        shard_idx += 1

    by_file: dict[str, list[str]] = {}
    for key in missing_keys:
        by_file.setdefault(origin_map[key], []).append(key)

    for filename in sorted(by_file):
        with safe_open(origin_hf_dir / filename, framework="pt", device="cpu") as f:
            for key in sorted(by_file[filename]):
                tensor = f.get_tensor(key)
                tensor_size = tensor.numel() * tensor.element_size()
                if shard and shard_size + tensor_size > chunk_size:
                    flush()
                shard[key] = tensor
                shard_size += tensor_size
                total_size += tensor_size
        flush()

    return added_map, total_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin-hf-dir", required=True, type=Path)
    parser.add_argument("--converted-hf-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=5 * 1024**3)
    parser.add_argument(
        "--include-prefix",
        action="append",
        default=[],
        help="Only add missing keys with this prefix. Can be repeated.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    origin_map = load_weight_map(args.origin_hf_dir)
    converted_map = load_weight_map(args.converted_hf_dir)
    missing_keys = sorted(set(origin_map) - set(converted_map))
    extra_keys = sorted(set(converted_map) - set(origin_map))
    selected_missing_keys = [
        key
        for key in missing_keys
        if not args.include_prefix or any(key.startswith(prefix) for prefix in args.include_prefix)
    ]

    print(
        "before_repair",
        f"origin_keys={len(origin_map)}",
        f"converted_keys={len(converted_map)}",
        f"missing={len(missing_keys)}",
        f"selected_missing={len(selected_missing_keys)}",
        f"include_prefix={args.include_prefix or ['<all>']}",
        f"extra={len(extra_keys)}",
    )
    if extra_keys:
        raise SystemExit(f"converted checkpoint has unexpected extra keys: {extra_keys[:10]}")

    copy_converted_dir(args.converted_hf_dir, args.output_dir, args.force)
    added_map, added_size = add_missing_shards(
        args.origin_hf_dir,
        selected_missing_keys,
        origin_map,
        args.output_dir,
        args.chunk_size,
    )

    repaired_map = dict(converted_map)
    repaired_map.update(added_map)
    if not args.include_prefix and set(repaired_map) != set(origin_map):
        remaining = sorted(set(origin_map) - set(repaired_map))
        raise SystemExit(f"repair failed; still missing {len(remaining)} keys: {remaining[:10]}")
    if args.include_prefix and set(added_map) != set(selected_missing_keys):
        remaining = sorted(set(selected_missing_keys) - set(added_map))
        raise SystemExit(f"partial repair failed; still missing selected keys: {remaining[:10]}")

    original_index = json.loads((args.converted_hf_dir / "model.safetensors.index.json").read_text())
    total_size = int(original_index.get("metadata", {}).get("total_size", 0)) + added_size
    repaired_index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(repaired_map.items())),
    }
    (args.output_dir / "model.safetensors.index.json").write_text(
        json.dumps(repaired_index, indent=2) + "\n"
    )

    print(
        "after_repair",
        f"repaired_keys={len(repaired_map)}",
        f"added={len(added_map)}",
        f"remaining_missing={len(set(origin_map) - set(repaired_map))}",
        f"added_size={added_size}",
        f"output={args.output_dir}",
    )


if __name__ == "__main__":
    main()
