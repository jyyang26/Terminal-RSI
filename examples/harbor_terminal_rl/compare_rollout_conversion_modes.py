#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path
from typing import Any

from slime.utils.types import Sample
from slime_plugins.harbor.atif_adapter import samples_from_trial_result


_TRIAL_RE = re.compile(r"slime-r(?P<rollout>\d+)-g(?P<group>\d+)-s(?P<sample>\d+)-i(?P<index>\d+)")


def _quantile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[int(pct * (len(values) - 1))]


def _summary(values: list[int | float]) -> dict[str, Any]:
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


def _trial_ids(path: Path) -> dict[str, int]:
    match = _TRIAL_RE.search(path.parent.name)
    if not match:
        return {"rollout": 0, "group": 0, "sample": 0, "index": 0}
    return {key: int(value) for key, value in match.groupdict().items()}


def _base_sample(ids: dict[str, int]) -> Sample:
    sample = Sample()
    sample.group_index = ids["group"]
    sample.index = ids["index"]
    sample.metadata = {}
    return sample


def _metadata(ids: dict[str, int], result_path: Path) -> dict[str, Any]:
    return {
        "harbor_rollout_id": ids["rollout"],
        "harbor_rollout_group_index": ids["group"],
        "harbor_sample_rank": ids["sample"],
        "harbor_trial_dir": result_path.parent.name,
    }


def _loss_sum(sample: Sample) -> int:
    return int(sum(int(value) for value in (getattr(sample, "loss_mask", None) or [])))


def _sample_row(sample: Sample) -> dict[str, Any]:
    response = int(getattr(sample, "response_length", 0) or 0)
    tokens = getattr(sample, "tokens", None) or []
    metadata = getattr(sample, "metadata", None) or {}
    return {
        "total_len": len(tokens),
        "prompt_len": max(0, len(tokens) - response),
        "response_len": response,
        "loss_tokens": _loss_sum(sample),
        "zero_mask_tokens": response - _loss_sum(sample),
        "metadata": metadata,
    }


def _flatten_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "total_len": _summary([row["total_len"] for row in rows]),
        "prompt_len": _summary([row["prompt_len"] for row in rows]),
        "response_len": _summary([row["response_len"] for row in rows]),
        "loss_tokens": _summary([row["loss_tokens"] for row in rows]),
        "zero_mask_tokens": _summary([row["zero_mask_tokens"] for row in rows]),
    }


def analyze(trials_dir: Path, *, rollout_id: int | None, max_group: int | None, limit: int | None) -> dict[str, Any]:
    files = sorted(trials_dir.glob("*/result.json"))
    selected: list[Path] = []
    for path in files:
        ids = _trial_ids(path)
        if rollout_id is not None and ids["rollout"] != rollout_id:
            continue
        if max_group is not None and ids["group"] > max_group:
            continue
        selected.append(path)
        if limit is not None and len(selected) >= limit:
            break

    modes = ["trajectory", "trajectory_normalized"]
    mode_rows: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    per_trial: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for path in selected:
        ids = _trial_ids(path)
        try:
            result = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(path), "error": f"json_load: {exc!r}"})
            continue

        base = _base_sample(ids)
        metadata = _metadata(ids, path)
        trial_record: dict[str, Any] = {"trial": path.parent.name, "ids": ids, "modes": {}}

        for mode in modes:
            try:
                samples = samples_from_trial_result(
                    result,
                    base,
                    mode=mode,
                    metadata=metadata,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append({"path": str(path), "mode": mode, "error": repr(exc)})
                continue

            rows = [_sample_row(sample) for sample in samples]
            mode_rows[mode].extend(rows)
            trial_record["modes"][mode] = {
                "num_samples": len(samples),
                "total_lens": [row["total_len"] for row in rows],
                "response_lens": [row["response_len"] for row in rows],
                "loss_tokens": [row["loss_tokens"] for row in rows],
                "zero_mask_tokens": [row["zero_mask_tokens"] for row in rows],
            }

            if mode == "trajectory" and samples:
                md = getattr(samples[0], "metadata", None) or {}
                trial_record["modes"][mode].update(
                    {
                        "stitch_context_tokens": md.get("harbor_trajectory_stitch_context_tokens"),
                        "stitch_output_tokens": md.get("harbor_trajectory_stitch_output_tokens"),
                        "stitch_full_prompt_turns": md.get("harbor_trajectory_stitch_full_prompt_turns"),
                        "stitch_suffix_prompt_turns": md.get("harbor_trajectory_stitch_suffix_prompt_turns"),
                        "stitch_prefix_tail_turns": md.get("harbor_trajectory_stitch_prefix_tail_turns"),
                    }
                )

        norm_rows = trial_record["modes"].get("trajectory_normalized", {})
        norm_total_lens = norm_rows.get("total_lens", [])
        norm_response_lens = norm_rows.get("response_lens", [])
        if norm_total_lens and norm_response_lens:
            first_prompt = norm_total_lens[0] - norm_response_lens[0]
            merged_lower_bound = first_prompt + sum(norm_response_lens)
            trial_record["normalized_merge_lower_bound"] = merged_lower_bound
            trial_record["normalized_merge_response_tokens"] = sum(norm_response_lens)

        per_trial.append(trial_record)

    normalized_merge_lbs = [
        int(row["normalized_merge_lower_bound"]) for row in per_trial if "normalized_merge_lower_bound" in row
    ]
    normalized_merge_responses = [
        int(row["normalized_merge_response_tokens"])
        for row in per_trial
        if "normalized_merge_response_tokens" in row
    ]

    trajectory_md = [
        record["modes"].get("trajectory", {}) for record in per_trial if record["modes"].get("trajectory")
    ]
    result = {
        "trials_dir": str(trials_dir),
        "selected_results": len(selected),
        "converted_trials": len(per_trial),
        "errors": errors[:20],
        "error_count": len(errors),
        "mode_stats": {mode: _flatten_stats(rows) for mode, rows in mode_rows.items()},
        "trajectory_stitch_stats": {
            "context_tokens": _summary(
                [int(row.get("stitch_context_tokens") or 0) for row in trajectory_md]
            ),
            "output_tokens": _summary([int(row.get("stitch_output_tokens") or 0) for row in trajectory_md]),
            "full_prompt_turns": _summary(
                [int(row.get("stitch_full_prompt_turns") or 0) for row in trajectory_md]
            ),
            "suffix_prompt_turns": _summary(
                [int(row.get("stitch_suffix_prompt_turns") or 0) for row in trajectory_md]
            ),
            "prefix_tail_turns": _summary(
                [int(row.get("stitch_prefix_tail_turns") or 0) for row in trajectory_md]
            ),
        },
        "normalized_merge_lower_bound": {
            "total_len": _summary(normalized_merge_lbs),
            "response_len": _summary(normalized_merge_responses),
            "samples_before": len(mode_rows["trajectory_normalized"]),
            "samples_after": len(normalized_merge_lbs),
            "reduction_factor": round(len(mode_rows["trajectory_normalized"]) / max(1, len(normalized_merge_lbs)), 3),
        },
        "chunk_count_hist": dict(
            sorted(
                collections.Counter(
                    record["modes"].get("trajectory_normalized", {}).get("num_samples", 0)
                    for record in per_trial
                ).items()
            )
        ),
        "top_by_trajectory_total_len": sorted(
            per_trial,
            key=lambda record: max(record["modes"].get("trajectory", {}).get("total_lens", [0])),
            reverse=True,
        )[:20],
        "top_by_normalized_chunks": sorted(
            per_trial,
            key=lambda record: record["modes"].get("trajectory_normalized", {}).get("num_samples", 0),
            reverse=True,
        )[:20],
    }
    return result


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Harbor Conversion Mode Comparison",
        "",
        f"- trials_dir: `{result['trials_dir']}`",
        f"- selected result.json files: `{result['selected_results']}`",
        f"- converted trials: `{result['converted_trials']}`",
        f"- conversion errors: `{result['error_count']}`",
        "",
        "## Mode Stats",
        "",
    ]
    for mode, stats in result["mode_stats"].items():
        lines.append(f"### {mode}")
        lines.append(f"- samples: `{stats['samples']}`")
        for name in ["total_len", "response_len", "loss_tokens", "zero_mask_tokens"]:
            summary = stats[name]
            fields = ", ".join(f"{key}={value}" for key, value in summary.items())
            lines.append(f"- {name}: {fields}")
        lines.append("")

    lines.extend(["## Normalized Merge Lower Bound", ""])
    merge = result["normalized_merge_lower_bound"]
    lines.append(f"- samples before: `{merge['samples_before']}`")
    lines.append(f"- samples after: `{merge['samples_after']}`")
    lines.append(f"- reduction factor: `{merge['reduction_factor']}`")
    for name in ["total_len", "response_len"]:
        summary = merge[name]
        fields = ", ".join(f"{key}={value}" for key, value in summary.items())
        lines.append(f"- {name}: {fields}")

    lines.extend(["", "## Trajectory Stitch Stats", ""])
    for name, summary in result["trajectory_stitch_stats"].items():
        fields = ", ".join(f"{key}={value}" for key, value in summary.items())
        lines.append(f"- {name}: {fields}")

    lines.extend(["", "## Chunk Count Histogram", ""])
    for chunks, count in result["chunk_count_hist"].items():
        lines.append(f"- `{chunks}` chunks: `{count}` trajectories")

    lines.extend(["", "## Largest Trajectory-Mode Samples", ""])
    for record in result["top_by_trajectory_total_len"][:10]:
        mode = record["modes"].get("trajectory", {})
        lines.append(
            "- "
            f"{record['trial']} total={max(mode.get('total_lens', [0]))} "
            f"response={max(mode.get('response_lens', [0]))} "
            f"context={mode.get('stitch_context_tokens')} "
            f"full_prompt_turns={mode.get('stitch_full_prompt_turns')}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials-dir", required=True, type=Path)
    parser.add_argument("--rollout-id", type=int, default=None)
    parser.add_argument("--max-group", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()

    result = analyze(
        args.trials_dir,
        rollout_id=args.rollout_id,
        max_group=args.max_group,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.md_out:
        write_markdown(result, args.md_out)


if __name__ == "__main__":
    main()
