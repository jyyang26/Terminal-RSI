#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


def _status_value(status: Any) -> str:
    return str(getattr(status, "value", status))


def _metadata(sample: dict[str, Any]) -> dict[str, Any]:
    return sample.get("metadata") or {}


def _response_tokens(sample: dict[str, Any]) -> list[int]:
    tokens = sample.get("tokens") or []
    response_length = int(sample.get("response_length", 0) or 0)
    return tokens[-response_length:] if response_length else []


def _prompt_tokens(sample: dict[str, Any]) -> list[int]:
    tokens = sample.get("tokens") or []
    response_length = int(sample.get("response_length", 0) or 0)
    return tokens[:-response_length] if response_length else tokens


def _trajectory_key(sample: dict[str, Any]) -> tuple[Any, Any, Any]:
    meta = _metadata(sample)
    return (
        meta.get("harbor_rollout_id"),
        meta.get("harbor_rollout_group_index", sample.get("group_index")),
        meta.get("harbor_sample_rank", meta.get("harbor_trial_sample_rank", sample.get("index"))),
    )


def _decode(tokenizer: Any, tokens: list[int]) -> str:
    return tokenizer.decode(tokens, skip_special_tokens=False)


def _has_prefix(value: list[int], prefix: list[int]) -> bool:
    return len(value) >= len(prefix) and value[: len(prefix)] == prefix


def _build_trajectory_record(
    tokenizer: Any,
    key: tuple[Any, Any, Any],
    samples: list[dict[str, Any]],
    *,
    include_turn_prompts: bool,
    include_token_ids: bool,
) -> dict[str, Any]:
    samples = sorted(samples, key=lambda sample: int(_metadata(sample).get("harbor_turn_index", 0) or 0))
    first = samples[0]
    first_meta = _metadata(first)

    status_counts: dict[str, int] = {}
    finish_counts: dict[str, int] = {}
    for sample in samples:
        status = _status_value(sample.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
        finish_reason = _metadata(sample).get("harbor_finish_reason_type")
        finish_key = str(finish_reason)
        finish_counts[finish_key] = finish_counts.get(finish_key, 0) + 1

    initial_prompt_tokens = _prompt_tokens(samples[0])
    reconstructed_tokens = list(initial_prompt_tokens)
    turns = []
    prefix_mismatch_turns = []

    for index, sample in enumerate(samples):
        meta = _metadata(sample)
        turn_index = int(meta.get("harbor_turn_index", index) or 0)
        prompt_ids = _prompt_tokens(sample)
        response_ids = _response_tokens(sample)
        finish_reason = meta.get("harbor_finish_reason_type")
        trial_status = _status_value(sample.get("status"))
        turn_generation_status = "truncated" if finish_reason == "length" else "completed"

        if not _has_prefix(prompt_ids, reconstructed_tokens):
            prefix_mismatch_turns.append(turn_index)
            if len(prompt_ids) > len(reconstructed_tokens):
                observation_delta_ids = prompt_ids[len(reconstructed_tokens) :]
            else:
                observation_delta_ids = []
        else:
            observation_delta_ids = prompt_ids[len(reconstructed_tokens) :]

        reconstructed_tokens.extend(observation_delta_ids)
        reconstructed_tokens.extend(response_ids)

        turn = {
            "turn_index": turn_index,
            "sample_index": sample.get("index"),
            "status": trial_status,
            "trial_status": trial_status,
            "turn_generation_status": turn_generation_status,
            "reward": sample.get("reward"),
            "prompt_length": len(prompt_ids),
            "response_length": int(sample.get("response_length", 0) or 0),
            "total_length": len(sample.get("tokens") or []),
            "finish_reason": finish_reason,
            "observation_delta_text": _decode(tokenizer, observation_delta_ids),
            "response_text": _decode(tokenizer, response_ids),
            "metadata": meta,
        }
        if include_turn_prompts:
            turn["prompt_text"] = _decode(tokenizer, prompt_ids)
        if include_token_ids:
            turn["prompt_tokens"] = prompt_ids
            turn["response_tokens"] = response_ids
            turn["observation_delta_tokens"] = observation_delta_ids
        turns.append(turn)

    record = {
        "source_dump_path": first.get("_source_dump_path"),
        "trajectory_key": {
            "rollout_id": key[0],
            "group_index": key[1],
            "sample_rank": key[2],
        },
        "trial_name": first_meta.get("harbor_trial_name"),
        "trial_uri": first_meta.get("harbor_trial_uri"),
        "task_name": first_meta.get("harbor_task_name"),
        "task_entry": first_meta.get("harbor_task_entry"),
        "sample_mode": first_meta.get("harbor_sample_mode"),
        "num_turn_samples": len(samples),
        "declared_num_turns": first_meta.get("harbor_num_turns"),
        "status_counts": status_counts,
        "finish_reason_counts": finish_counts,
        "trajectory_reward": first.get("reward"),
        "initial_prompt_length": len(initial_prompt_tokens),
        "reconstructed_total_tokens": len(reconstructed_tokens),
        "prefix_mismatch_turns": prefix_mismatch_turns,
        "initial_prompt_text": _decode(tokenizer, initial_prompt_tokens),
        "trajectory_text": _decode(tokenizer, reconstructed_tokens),
        "turns": turns,
    }
    if include_token_ids:
        record["initial_prompt_tokens"] = initial_prompt_tokens
        record["trajectory_tokens"] = reconstructed_tokens
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Export full trajectory records from a slime Harbor rollout dump.")
    parser.add_argument("dump_paths", nargs="+")
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--status")
    parser.add_argument("--task-name")
    parser.add_argument("--include-turn-prompts", action="store_true")
    parser.add_argument("--include-token-ids", action="store_true")
    parser.add_argument("--format", choices=["jsonl", "json"], default="jsonl")
    args = parser.parse_args()

    dump_paths: list[Path] = []
    for raw_path in args.dump_paths:
        path = Path(raw_path)
        if path.is_dir():
            dump_paths.extend(sorted(path.glob("*.pt"), key=lambda item: int(item.stem) if item.stem.isdigit() else item.stem))
        else:
            dump_paths.append(path)

    samples = []
    for dump_path in dump_paths:
        dump = torch.load(dump_path, map_location="cpu", weights_only=False)
        dump_samples = dump.get("samples", []) if isinstance(dump, dict) else dump
        for sample in dump_samples:
            copied = dict(sample)
            copied["_source_dump_path"] = str(dump_path)
            samples.append(copied)

    if args.status:
        samples = [sample for sample in samples if _status_value(sample.get("status")) == args.status]
    if args.task_name:
        samples = [sample for sample in samples if _metadata(sample).get("harbor_task_name") == args.task_name]

    grouped: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[_trajectory_key(sample)].append(sample)

    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    records = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_items = list(grouped.items())[: args.limit]
    if args.format == "jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for key, trajectory_samples in selected_items:
                record = _build_trajectory_record(
                    tokenizer,
                    key,
                    trajectory_samples,
                    include_turn_prompts=args.include_turn_prompts,
                    include_token_ids=args.include_token_ids,
                )
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
    else:
        for key, trajectory_samples in selected_items:
            records.append(
                _build_trajectory_record(
                    tokenizer,
                    key,
                    trajectory_samples,
                    include_turn_prompts=args.include_turn_prompts,
                    include_token_ids=args.include_token_ids,
                )
            )
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    summary = {
        "dump_paths": [str(path) for path in dump_paths],
        "output": str(output_path),
        "format": args.format,
        "input_samples": len(samples),
        "unique_trajectories": len(grouped),
        "exported_trajectories": len(records),
        "output_size_bytes": output_path.stat().st_size,
        "status_filter": args.status,
        "task_name_filter": args.task_name,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
