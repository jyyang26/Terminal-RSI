#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


def _status_value(status: Any) -> str:
    return str(getattr(status, "value", status))


def _decode_sample(tokenizer, sample: dict[str, Any], *, include_prompt: bool, max_chars: int) -> dict[str, Any]:
    tokens = sample.get("tokens") or []
    response_length = int(sample.get("response_length", 0) or 0)
    response_tokens = tokens[-response_length:] if response_length else []
    prompt_tokens = tokens[:-response_length] if response_length else tokens
    loss_mask = sample.get("loss_mask")
    metadata = sample.get("metadata") or {}

    response_text = tokenizer.decode(response_tokens, skip_special_tokens=False)
    record = {
        "sample_index": sample.get("index"),
        "group_index": sample.get("group_index"),
        "status": _status_value(sample.get("status")),
        "reward": sample.get("reward"),
        "response_length": response_length,
        "total_length": len(tokens),
        "prompt_length": len(prompt_tokens),
        "loss_mask_ones": sum(loss_mask) if isinstance(loss_mask, list) else None,
        "harbor_trial_name": metadata.get("harbor_trial_name"),
        "harbor_task_name": metadata.get("harbor_task_name"),
        "harbor_turn_index": metadata.get("harbor_turn_index"),
        "harbor_num_turns": metadata.get("harbor_num_turns"),
        "harbor_finish_reason_type": metadata.get("harbor_finish_reason_type"),
        "harbor_length_finish_turns": metadata.get("harbor_length_finish_turns"),
        "response_text": response_text[:max_chars] if max_chars > 0 else response_text,
        "metadata": metadata,
    }
    if include_prompt:
        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
        record["prompt_text"] = prompt_text[:max_chars] if max_chars > 0 else prompt_text
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a slime Harbor debug rollout dump.")
    parser.add_argument("dump_path")
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--filter-status")
    parser.add_argument("--filter-finish-reason")
    parser.add_argument("--jsonl-output")
    parser.add_argument("--include-prompt", action="store_true")
    parser.add_argument("--max-decode-chars", type=int, default=4000)
    args = parser.parse_args()

    dump = torch.load(args.dump_path, map_location="cpu", weights_only=False)
    samples = dump.get("samples", [])
    if args.filter_status is not None:
        samples = [sample for sample in samples if _status_value(sample.get("status")) == args.filter_status]
    if args.filter_finish_reason is not None:
        samples = [
            sample
            for sample in samples
            if (sample.get("metadata") or {}).get("harbor_finish_reason_type") == args.filter_finish_reason
        ]
    print("dump_keys", sorted(dump.keys()))
    print("num_samples", len(samples))

    lengths = [int(sample.get("response_length", 0)) for sample in samples]
    if lengths:
        print("response_len_min", min(lengths))
        print("response_len_max", max(lengths))
        print("response_len_mean", sum(lengths) / len(lengths))
        print("num_at_8096", sum(1 for length in lengths if length >= 8096))

    rewards = [sample.get("reward") for sample in samples]
    statuses = [str(sample.get("status")) for sample in samples]
    print("rewards", rewards[:10])
    print("statuses", statuses[:10])

    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    for index, sample in enumerate(samples[: args.max_samples]):
        tokens = sample.get("tokens") or []
        response_length = int(sample.get("response_length", 0))
        suffix = tokens[-response_length:] if response_length else []
        text = tokenizer.decode(suffix, skip_special_tokens=False)
        print(f"--- sample {index} ---")
        print("response_length", response_length)
        print("starts_with", repr(text[:300]))
        print("ends_with", repr(text[-300:]))
        metadata = sample.get("metadata")
        if metadata is not None:
            try:
                print("metadata_keys", sorted(metadata.keys()))
            except AttributeError:
                print("metadata", repr(metadata)[:500])

    if samples:
        metadata = samples[0].get("metadata") or {}
        for key in ("trial_dir", "trajectory_path", "task_name", "task_path"):
            value = metadata.get(key)
            if value:
                print(key, value)

    if args.jsonl_output:
        output_path = Path(args.jsonl_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for sample in samples:
                record = _decode_sample(
                    tokenizer,
                    sample,
                    include_prompt=args.include_prompt,
                    max_chars=args.max_decode_chars,
                )
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("jsonl_output", str(output_path))


if __name__ == "__main__":
    main()
