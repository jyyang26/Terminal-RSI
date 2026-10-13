#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import torch


def _get(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _status_value(status: Any) -> str:
    return str(getattr(status, "value", status))


def _q(values: list[int], pct: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * pct))))
    return int(values[idx])


def summarize(path: Path) -> None:
    dump = torch.load(path, map_location="cpu", weights_only=False)
    samples = dump.get("samples", []) if isinstance(dump, dict) else dump
    print(f"\n=== {path} ===")
    print(f"samples={len(samples)}")
    if not samples:
        return

    statuses = Counter(_status_value(_get(sample, "status")) for sample in samples)
    print("status_counts", dict(statuses))
    print("truncated_ratio", statuses.get("truncated", 0) / len(samples))

    response_lengths: list[int] = []
    total_lengths: list[int] = []
    effective_lengths: list[int] = []
    finish_reasons = Counter()
    length_finish_samples = 0
    any_length_finish_samples = 0
    length_finish_turns = 0
    num_turns = 0
    trimmed_samples = 0
    max_original_total = 0
    max_trimmed_total = 0

    truncated_by_finish_reason = Counter()
    truncated_examples: list[tuple[int, int, int, Any, Any, Any]] = []

    for idx, sample in enumerate(samples):
        response_length = int(_get(sample, "response_length", 0) or 0)
        tokens = _get(sample, "tokens", []) or []
        loss_mask = _get(sample, "loss_mask")
        metadata = _get(sample, "metadata", {}) or {}
        status = _status_value(_get(sample, "status"))

        response_lengths.append(response_length)
        total_lengths.append(len(tokens))
        effective_lengths.append(sum(loss_mask) if loss_mask is not None else response_length)

        finish_reason = metadata.get("harbor_finish_reason_type")
        finish_reasons[finish_reason] += 1
        sample_length_finish_turns = int(metadata.get("harbor_length_finish_turns", 0) or 0)
        sample_num_turns = int(metadata.get("harbor_num_turns", 0) or 0)
        length_finish_turns += sample_length_finish_turns
        num_turns += sample_num_turns
        if finish_reason == "length":
            length_finish_samples += 1
        if sample_length_finish_turns > 0:
            any_length_finish_samples += 1
        if "harbor_trimmed_total_length" in metadata:
            trimmed_samples += 1
            max_original_total = max(max_original_total, int(metadata.get("harbor_original_total_length", 0) or 0))
            max_trimmed_total = max(max_trimmed_total, int(metadata.get("harbor_trimmed_total_length", 0) or 0))

        if status == "truncated":
            truncated_by_finish_reason[finish_reason] += 1
            if len(truncated_examples) < 8:
                truncated_examples.append(
                    (
                        idx,
                        len(tokens),
                        response_length,
                        effective_lengths[-1],
                        finish_reason,
                        sample_length_finish_turns,
                        metadata.get("harbor_trial_name") or metadata.get("trial_name"),
                    )
                )

    print(
        "response_length",
        {
            "min": min(response_lengths),
            "mean": mean(response_lengths),
            "p50": _q(response_lengths, 0.50),
            "p95": _q(response_lengths, 0.95),
            "p99": _q(response_lengths, 0.99),
            "max": max(response_lengths),
        },
    )
    print(
        "total_length",
        {
            "min": min(total_lengths),
            "mean": mean(total_lengths),
            "p50": _q(total_lengths, 0.50),
            "p95": _q(total_lengths, 0.95),
            "p99": _q(total_lengths, 0.99),
            "max": max(total_lengths),
        },
    )
    print(
        "effective_loss_tokens",
        {
            "min": min(effective_lengths),
            "mean": mean(effective_lengths),
            "p95": _q(effective_lengths, 0.95),
            "max": max(effective_lengths),
        },
    )
    print("finish_reason_counts", dict(finish_reasons))
    print("step_length_finish_sample_ratio", length_finish_samples / len(samples))
    print("any_step_length_finish_sample_ratio", any_length_finish_samples / len(samples))
    print("step_length_finish_turn_ratio", (length_finish_turns / num_turns) if num_turns else 0.0)
    print("truncated_by_finish_reason", dict(truncated_by_finish_reason))
    print("trimmed_samples", trimmed_samples, "max_original_total", max_original_total, "max_trimmed_total", max_trimmed_total)
    print("truncated_examples idx,total,response,effective,finish_reason,length_finish_turns,trial")
    for item in truncated_examples:
        print(" ", item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    expanded: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.pt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem))
        else:
            expanded.append(path)
    for path in expanded:
        summarize(path)


if __name__ == "__main__":
    main()
