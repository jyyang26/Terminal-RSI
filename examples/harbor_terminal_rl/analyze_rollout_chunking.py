#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any

import torch


def _quantile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[int(pct * (len(values) - 1))]


def _summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "min": 0,
            "mean": 0.0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
            "over_64k": 0,
            "over_128k": 0,
        }
    values = sorted(int(v) for v in values)
    return {
        "n": len(values),
        "min": values[0],
        "mean": round(float(statistics.mean(values)), 2),
        "p50": values[len(values) // 2],
        "p90": _quantile(values, 0.90),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "max": values[-1],
        "over_64k": sum(v > 65536 for v in values),
        "over_128k": sum(v > 131072 for v in values),
    }


def _group_id(sample: dict[str, Any]) -> str:
    return str(sample.get("group_id", sample.get("group_index", "unknown")))


def _loss_mask_stats(sample: dict[str, Any]) -> tuple[int, int]:
    mask = sample.get("loss_mask") or []
    ones = sum(int(x) for x in mask)
    return ones, len(mask) - ones


def analyze(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    samples = obj["samples"]

    by_gid: dict[str, list[tuple[int, dict[str, Any]]]] = collections.defaultdict(list)
    for index, sample in enumerate(samples):
        by_gid[_group_id(sample)].append((index, sample))

    chunk_counts = [len(group) for group in by_gid.values()]
    chunk_lengths: list[int] = []
    chunk_prompt_lengths: list[int] = []
    chunk_response_lengths: list[int] = []
    chunk_loss_tokens: list[int] = []
    chunk_zero_mask_tokens: list[int] = []
    common_suffix_tokens: list[int] = []
    mode_counts: collections.Counter[str] = collections.Counter()

    split_before_chunks = 0
    started_after_split_chunks = 0
    normalized_prefix_chunks = 0
    approximate_conditioning_chunks = 0

    for sample in samples:
        tokens = sample.get("tokens") or []
        response_length = int(sample.get("response_length") or 0)
        loss_tokens, zero_tokens = _loss_mask_stats(sample)
        metadata = sample.get("metadata") or {}

        chunk_lengths.append(len(tokens))
        chunk_prompt_lengths.append(len(tokens) - response_length)
        chunk_response_lengths.append(response_length)
        chunk_loss_tokens.append(loss_tokens)
        chunk_zero_mask_tokens.append(zero_tokens)
        common_suffix_tokens.append(
            int(metadata.get("harbor_trajectory_normalized_common_suffix_candidate_tokens") or 0)
        )
        mode_counts[str(metadata.get("harbor_sample_mode"))] += 1

        if metadata.get("harbor_trajectory_normalized_split_before_turn") is not None:
            split_before_chunks += 1
        if metadata.get("harbor_trajectory_chunk_started_after_split"):
            started_after_split_chunks += 1
        if int(metadata.get("harbor_trajectory_normalized_prefix_turns") or 0) > 0:
            normalized_prefix_chunks += 1
        if not metadata.get("harbor_train_inference_conditioning_match", True):
            approximate_conditioning_chunks += 1

    trajectory_rows: list[dict[str, Any]] = []
    for gid, group in by_gid.items():
        group = sorted(
            group,
            key=lambda item: (item[1].get("metadata") or {}).get("harbor_trajectory_chunk_index", item[0]),
        )
        first = group[0][1]
        first_response = int(first.get("response_length") or 0)
        first_prompt = len(first.get("tokens") or []) - first_response
        response_sum = 0
        loss_sum = 0
        zero_sum = 0
        chunk_total_sum = 0
        chunk_total_max = 0
        nturns = 0
        split_chunks = 0
        normalized_chunks = 0

        for _, sample in group:
            metadata = sample.get("metadata") or {}
            response = int(sample.get("response_length") or 0)
            loss, zero = _loss_mask_stats(sample)
            total = len(sample.get("tokens") or [])

            response_sum += response
            loss_sum += loss
            zero_sum += zero
            chunk_total_sum += total
            chunk_total_max = max(chunk_total_max, total)
            nturns = max(nturns, int(metadata.get("harbor_num_turns") or 0))
            split_chunks += int(bool(metadata.get("harbor_trajectory_chunk_started_after_split")))
            normalized_chunks += int(int(metadata.get("harbor_trajectory_normalized_prefix_turns") or 0) > 0)

        # Lower bound for a merged single-sample trajectory. A real merge may
        # need additional zero-mask context between chunks, so this is optimistic.
        one_sample_total_lower_bound = first_prompt + response_sum

        trajectory_rows.append(
            {
                "group_id": gid,
                "chunks": len(group),
                "num_turns": nturns,
                "split_chunks": split_chunks,
                "normalized_chunks": normalized_chunks,
                "first_prompt_tokens": first_prompt,
                "chunk_total_sum": chunk_total_sum,
                "chunk_total_max": chunk_total_max,
                "response_tokens_sum": response_sum,
                "loss_tokens_sum": loss_sum,
                "zero_mask_tokens_sum": zero_sum,
                "one_sample_total_lower_bound": one_sample_total_lower_bound,
            }
        )

    result = {
        "path": str(path),
        "rollout_id": obj.get("rollout_id"),
        "num_samples": len(samples),
        "num_trajectories": len(by_gid),
        "mode_counts": dict(mode_counts),
        "chunk_count_hist": dict(sorted(collections.Counter(chunk_counts).items())),
        "split_before_chunks": split_before_chunks,
        "started_after_split_chunks": started_after_split_chunks,
        "normalized_prefix_chunks": normalized_prefix_chunks,
        "approximate_conditioning_chunks": approximate_conditioning_chunks,
        "summaries": {
            "chunks_per_trajectory": _summary(chunk_counts),
            "chunk_total_len": _summary(chunk_lengths),
            "chunk_prompt_len": _summary(chunk_prompt_lengths),
            "chunk_response_len": _summary(chunk_response_lengths),
            "chunk_loss_tokens": _summary(chunk_loss_tokens),
            "chunk_zero_mask_tokens": _summary(chunk_zero_mask_tokens),
            "common_suffix_candidate_tokens": _summary(common_suffix_tokens),
            "one_sample_total_lower_bound": _summary(
                [row["one_sample_total_lower_bound"] for row in trajectory_rows]
            ),
            "one_sample_response_tokens": _summary([row["response_tokens_sum"] for row in trajectory_rows]),
            "one_sample_loss_tokens": _summary([row["loss_tokens_sum"] for row in trajectory_rows]),
            "one_sample_zero_mask_tokens": _summary([row["zero_mask_tokens_sum"] for row in trajectory_rows]),
        },
        "merge_same_trajectory_effect": {
            "samples_before": len(samples),
            "samples_after": len(by_gid),
            "reduction_factor": round(len(samples) / max(1, len(by_gid)), 3),
            "one_sample_over_64k": sum(row["one_sample_total_lower_bound"] > 65536 for row in trajectory_rows),
            "one_sample_over_128k": sum(row["one_sample_total_lower_bound"] > 131072 for row in trajectory_rows),
        },
        "top_by_chunks": sorted(
            trajectory_rows,
            key=lambda row: (row["chunks"], row["one_sample_total_lower_bound"]),
            reverse=True,
        )[:20],
        "top_by_merged_length": sorted(
            trajectory_rows,
            key=lambda row: row["one_sample_total_lower_bound"],
            reverse=True,
        )[:20],
    }
    return result


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Harbor Rollout Chunking Analysis",
        "",
        f"- dump: `{result['path']}`",
        f"- rollout_id: `{result['rollout_id']}`",
        f"- samples/chunks: `{result['num_samples']}`",
        f"- trajectories: `{result['num_trajectories']}`",
        f"- reduction if one sample per trajectory: `{result['merge_same_trajectory_effect']['reduction_factor']}x`",
        f"- optimistic merged samples over 64k: `{result['merge_same_trajectory_effect']['one_sample_over_64k']}`",
        f"- optimistic merged samples over 128k: `{result['merge_same_trajectory_effect']['one_sample_over_128k']}`",
        "",
        "## Chunk Count Histogram",
        "",
    ]
    for chunks, count in result["chunk_count_hist"].items():
        lines.append(f"- `{chunks}` chunks: `{count}` trajectories")

    lines.extend(["", "## Summaries", ""])
    for name, summary in result["summaries"].items():
        lines.append(f"### {name}")
        fields = ", ".join(f"{key}={value}" for key, value in summary.items())
        lines.append(fields)
        lines.append("")

    lines.extend(["## Top Trajectories By Chunk Count", ""])
    for row in result["top_by_chunks"]:
        lines.append(
            "- "
            f"group={row['group_id']} chunks={row['chunks']} turns={row['num_turns']} "
            f"split_chunks={row['split_chunks']} normalized_chunks={row['normalized_chunks']} "
            f"merged_lb={row['one_sample_total_lower_bound']} "
            f"loss={row['loss_tokens_sum']} zero={row['zero_mask_tokens_sum']}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()

    result = analyze(args.dump)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.md_out:
        write_markdown(result, args.md_out)


if __name__ == "__main__":
    main()
