#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_torch(path: Path) -> Any:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _to_py(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _to_py(value.detach().cpu().tolist())
    if hasattr(value, "tolist") and not isinstance(value, list):
        try:
            return _to_py(value.tolist())
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(key): _to_py(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_py(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return _to_py(value.to_dict())
        except Exception:
            return str(value)
    return str(value)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _spans(values: list[int], target: int) -> list[list[int]]:
    spans: list[list[int]] = []
    start = None
    for i, value in enumerate(values):
        if value == target and start is None:
            start = i
        elif value != target and start is not None:
            spans.append([start, i])
            start = None
    if start is not None:
        spans.append([start, len(values)])
    return spans


def _numeric_summary(values: Any) -> dict[str, Any] | None:
    values = _to_py(values)
    if values is None:
        return None
    if isinstance(values, (int, float)):
        nums = [float(values)]
    elif isinstance(values, list):
        nums = []

        def visit(item: Any) -> None:
            if isinstance(item, (int, float)):
                nums.append(float(item))
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(values)
    else:
        return None
    if not nums:
        return {"len": 0, "sum": 0.0, "mean": None, "min": None, "max": None, "max_abs": 0.0}
    total = sum(nums)
    return {
        "len": len(nums),
        "sum": total,
        "mean": total / len(nums),
        "min": min(nums),
        "max": max(nums),
        "max_abs": max(abs(x) for x in nums),
    }


def _sample_index(sample: dict[str, Any]) -> Any:
    return sample.get("index")


def _load_rollout_samples(rollout_dir: Path | None) -> dict[Any, dict[str, Any]]:
    out: dict[Any, dict[str, Any]] = {}
    if rollout_dir is None or not rollout_dir.exists():
        return out
    for path in sorted(rollout_dir.glob("*.pt")):
        dump = _to_py(_load_torch(path))
        for sample in dump.get("samples", []):
            sample["_source_rollout_dump"] = str(path)
            out[_sample_index(sample)] = sample
    return out


def _per_sample_value(data: dict[str, Any], key: str, i: int) -> Any:
    values = data.get(key)
    if values is None:
        return None
    values = _to_py(values)
    if isinstance(values, list) and i < len(values):
        return values[i]
    return values


def _load_train_records(train_data_dir: Path | None) -> dict[Any, dict[str, Any]]:
    out: dict[Any, dict[str, Any]] = {}
    if train_data_dir is None or not train_data_dir.exists():
        return out
    if train_data_dir.is_file():
        paths = [train_data_dir]
    else:
        paths = sorted(train_data_dir.glob("*.pt"))
        if not paths:
            paths = sorted(path for path in train_data_dir.iterdir() if path.is_file())
    for path in paths:
        dump = _to_py(_load_torch(path))
        rollout_data = dump.get("rollout_data", {})
        sample_indices = _to_py(rollout_data.get("sample_indices", []))
        for i, sample_index in enumerate(sample_indices):
            record = {
                "_source_train_data": str(path),
                "rollout_id": dump.get("rollout_id"),
                "rank": dump.get("rank"),
                "sample_index": sample_index,
            }
            for key in (
                "tokens",
                "response_lengths",
                "loss_masks",
                "group_ids",
                "group_mask_sums",
                "rewards",
                "raw_reward",
                "truncated",
                "rollout_log_probs",
                "total_lengths",
                "advantages",
                "returns",
            ):
                record[key] = _per_sample_value(rollout_data, key, i)
            out[sample_index] = record
    return out


def _trajectory_file_summary(case_dir: Path) -> list[dict[str, Any]]:
    agent_dir = case_dir / "raw_trial" / "agent"
    if not agent_dir.exists():
        return []

    summaries = []
    for path in sorted(agent_dir.glob("trajectory*.json")):
        record: dict[str, Any] = {"path": str(path.relative_to(case_dir))}
        try:
            payload = _load_json(path)
            steps = payload.get("steps") or []
            subagent_refs = []
            copied_steps = 0
            llm_call_steps = 0
            for step in steps:
                if not isinstance(step, dict):
                    continue
                if step.get("is_copied_context"):
                    copied_steps += 1
                if step.get("llm_call_count"):
                    llm_call_steps += 1
                result = step.get("result") if isinstance(step.get("result"), dict) else {}
                refs = result.get("subagent_trajectory_ref") or []
                if isinstance(refs, list):
                    for ref in refs:
                        if isinstance(ref, dict):
                            subagent_refs.append(
                                {
                                    "trajectory_id": ref.get("trajectory_id"),
                                    "trajectory_path": ref.get("trajectory_path"),
                                    "agent_name": ref.get("agent_name"),
                                    "summary": (ref.get("extra") or {}).get("summary")
                                    if isinstance(ref.get("extra"), dict)
                                    else None,
                                }
                            )
            record.update(
                {
                    "steps": len(steps),
                    "session_id": payload.get("session_id"),
                    "continued_trajectory_ref": payload.get("continued_trajectory_ref"),
                    "continuation_index": ((payload.get("agent") or {}).get("extra") or {}).get("continuation_index"),
                    "source_counts": _counts(step.get("source") for step in steps if isinstance(step, dict)),
                    "copied_steps": copied_steps,
                    "llm_call_steps": llm_call_steps,
                    "subagent_ref_count": len(subagent_refs),
                    "subagent_refs": subagent_refs,
                }
            )
        except Exception as exc:
            record["error"] = str(exc)
        summaries.append(record)
    return summaries


def _counts(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return out


def _case_summary(
    case_dir: Path,
    *,
    rollout_samples_by_index: dict[Any, dict[str, Any]],
    train_records_by_index: dict[Any, dict[str, Any]],
) -> dict[str, Any]:
    conversion = _load_json(case_dir / "conversion.json")
    details = conversion.get("details", [])
    converted_samples = conversion.get("converted_samples", [])

    sample_checks = []
    for sample in converted_samples:
        sample_index = sample.get("index")
        metadata = sample.get("metadata") or {}
        detail_index = metadata.get("harbor_detail_index")
        step_index = metadata.get("harbor_step_index", metadata.get("harbor_turn_index"))
        loss_mask = sample.get("loss_mask") or []
        sample_prompt_ids = sample.get("prompt_token_ids") or []
        sample_response_ids = sample.get("response_token_ids") or []
        rollout_sample = rollout_samples_by_index.get(sample_index)
        train_record = train_records_by_index.get(sample_index)
        train_loss_mask = train_record.get("loss_masks") if train_record else None
        train_tokens = train_record.get("tokens") if train_record else None
        train_advantages = train_record.get("advantages") if train_record else None
        train_returns = train_record.get("returns") if train_record else None
        raw_prompt_ids = None
        raw_completion_ids = None
        if isinstance(detail_index, int) and 0 <= detail_index < len(details):
            raw = details[detail_index].get("raw_detail") or {}
            raw_prompts = raw.get("prompt_token_ids") or []
            raw_completions = raw.get("completion_token_ids") or []
            if isinstance(step_index, int):
                if 0 <= step_index < len(raw_prompts):
                    raw_prompt_ids = raw_prompts[step_index]
                if 0 <= step_index < len(raw_completions):
                    raw_completion_ids = raw_completions[step_index]
        case_tokens = (sample.get("tokens") or []) if isinstance(sample.get("tokens"), list) else None
        sample_checks.append(
            {
                "sample_index": sample_index,
                "sample_mode": metadata.get("harbor_sample_mode"),
                "detail_index": detail_index,
                "step_index": step_index,
                "normalized_chunk": metadata.get("harbor_trajectory_chunk_index"),
                "normalized_num_chunks": metadata.get("harbor_trajectory_num_chunks"),
                "normalized_chunk_turn_span": [
                    metadata.get("harbor_trajectory_chunk_start_turn"),
                    metadata.get("harbor_trajectory_chunk_end_turn"),
                ],
                "normalized_context_tokens": metadata.get("harbor_trajectory_normalized_context_tokens"),
                "normalized_output_tokens": metadata.get("harbor_trajectory_normalized_output_tokens"),
                "normalized_strict_prefix_turns": metadata.get("harbor_trajectory_normalized_strict_prefix_turns"),
                "normalized_prefix_turns": metadata.get("harbor_trajectory_normalized_prefix_turns"),
                "normalized_split_after_turns": metadata.get("harbor_trajectory_normalized_split_after_turns"),
                "normalized_split_before_turn": metadata.get("harbor_trajectory_normalized_split_before_turn"),
                "normalized_prompt_suffix_dropped_tokens": metadata.get(
                    "harbor_trajectory_normalized_prompt_suffix_dropped_tokens"
                ),
                "normalized_response_prefix_dropped_tokens": metadata.get(
                    "harbor_trajectory_normalized_response_prefix_dropped_tokens"
                ),
                "normalized_response_suffix_dropped_tokens": metadata.get(
                    "harbor_trajectory_normalized_response_suffix_dropped_tokens"
                ),
                "conditioning_match": metadata.get("harbor_train_inference_conditioning_match"),
                "case_response_length": sample.get("response_length"),
                "case_loss_mask_sum": sample.get("loss_mask_sum"),
                "case_zero_spans": _spans(loss_mask, 0) if isinstance(loss_mask, list) else None,
                "raw_prompt_length": len(raw_prompt_ids) if isinstance(raw_prompt_ids, list) else None,
                "raw_completion_length": len(raw_completion_ids) if isinstance(raw_completion_ids, list) else None,
                "sample_prompt_matches_raw": (
                    sample_prompt_ids == raw_prompt_ids if isinstance(raw_prompt_ids, list) else None
                ),
                "sample_response_matches_raw": (
                    sample_response_ids == raw_completion_ids if isinstance(raw_completion_ids, list) else None
                ),
                "rollout_found": rollout_sample is not None,
                "rollout_response_length": rollout_sample.get("response_length") if rollout_sample else None,
                "rollout_loss_mask_sum": (
                    sum(rollout_sample.get("loss_mask") or []) if rollout_sample else None
                ),
                "train_found": train_record is not None,
                "train_response_length": train_record.get("response_lengths") if train_record else None,
                "train_loss_mask_sum": sum(train_loss_mask) if isinstance(train_loss_mask, list) else None,
                "train_group_id": train_record.get("group_ids") if train_record else None,
                "train_group_mask_sum": train_record.get("group_mask_sums") if train_record else None,
                "train_reward": train_record.get("rewards") if train_record else None,
                "train_raw_reward": train_record.get("raw_reward") if train_record else None,
                "train_advantage_summary": _numeric_summary(train_advantages),
                "train_return_summary": _numeric_summary(train_returns),
                "train_zero_spans": _spans(train_loss_mask, 0) if isinstance(train_loss_mask, list) else None,
                "train_tokens_match_case": train_tokens == case_tokens if train_record and case_tokens is not None else None,
                "train_loss_mask_match_case": (
                    train_loss_mask == loss_mask if isinstance(train_loss_mask, list) and isinstance(loss_mask, list) else None
                ),
            }
        )

    detail_summaries = []
    for detail in details:
        raw = detail.get("raw_detail") or {}
        merged = detail.get("merged_segment") or {}
        v030_merged = detail.get("v030_merged_segment") or {}
        alignment = detail.get("alignment") or []
        stitched_alignment = detail.get("stitched_alignment") or []
        completion_lengths = raw.get("completion_lengths") or []
        raw_output_tokens = sum(completion_lengths) if isinstance(completion_lengths, list) else None
        detail_summaries.append(
            {
                "detail_index": detail.get("detail_index"),
                "num_turns": raw.get("num_completion_turns"),
                "prompt_lengths": raw.get("prompt_lengths"),
                "completion_lengths": completion_lengths,
                "raw_output_tokens": raw_output_tokens,
                "prefix_mismatch_turns": detail.get("prefix_mismatch_turns"),
                "base_changed_turns": [row.get("turn_index") for row in alignment if row.get("base_changed")],
                "prefix_drift_turns": [row.get("turn_index") for row in alignment if row.get("prefix_drift")],
                "context_tail_lengths": [row.get("context_tail_len") for row in alignment],
                "stitch_strategies": [row.get("strategy") for row in stitched_alignment],
                "stitched_context_lengths": [row.get("context_len") for row in stitched_alignment],
                "stitched_response_len": merged.get("response_len"),
                "stitched_loss_mask_sum": merged.get("loss_mask_sum"),
                "stitched_zero_spans": _spans(merged.get("loss_mask") or [], 0),
                "v030_response_len": v030_merged.get("response_len"),
                "v030_loss_mask_sum": v030_merged.get("loss_mask_sum"),
                "v030_zero_spans": _spans(v030_merged.get("loss_mask") or [], 0),
            }
        )

    return {
        "case_dir": str(case_dir),
        "mode": conversion.get("mode"),
        "reward": conversion.get("reward"),
        "status": conversion.get("status"),
        "num_rollout_details": conversion.get("num_rollout_details"),
        "details": detail_summaries,
        "sample_checks": sample_checks,
        "trajectory_files": _trajectory_file_summary(case_dir),
    }


def _global_findings(payload: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    cases = payload.get("cases") or []

    details_with_raw_outputs = [
        detail
        for case in cases
        for detail in case.get("details", [])
        if isinstance(detail.get("raw_output_tokens"), int)
    ]
    trajectory_details = [
        detail
        for detail in details_with_raw_outputs
        if isinstance(detail.get("stitched_loss_mask_sum"), int)
    ]
    v030_dropped = [
        detail
        for detail in details_with_raw_outputs
        if isinstance(detail.get("v030_loss_mask_sum"), int)
        and detail["v030_loss_mask_sum"] < detail["raw_output_tokens"]
    ]
    stitched_preserved = [
        detail
        for detail in trajectory_details
        if detail.get("stitched_loss_mask_sum") == detail.get("raw_output_tokens")
    ]

    if trajectory_details:
        findings.append(
            "Stitched trajectory conversion preserves assistant outputs in "
            f"{len(stitched_preserved)}/{len(trajectory_details)} rollout details "
            "(`stitched_loss_mask_sum == raw_output_tokens`)."
        )
    if v030_dropped:
        findings.append(
            "Native v0.3 `merge_turns` would train on fewer assistant tokens in "
            f"{len(v030_dropped)}/{len(details_with_raw_outputs)} rollout details; this indicates prompt-base drift "
            "where earlier assistant outputs are not represented by the merged response."
        )

    sample_checks = [check for case in cases for check in case.get("sample_checks", [])]
    normalized_checks = [
        check for check in sample_checks if check.get("sample_mode") == "trajectory_normalized"
    ]
    prompt_snapshot_checks = [
        check
        for check in sample_checks
        if check.get("case_response_length") == check.get("case_loss_mask_sum")
        and check.get("case_zero_spans") == []
    ]
    raw_snapshot_checks = [
        check
        for check in sample_checks
        if check.get("sample_prompt_matches_raw") is True and check.get("sample_response_matches_raw") is True
    ]
    if normalized_checks:
        split_free = [
            check
            for check in normalized_checks
            if check.get("normalized_chunk") == 0 and check.get("normalized_num_chunks") == 1
        ]
        conditioning_exact = [
            check for check in normalized_checks if check.get("conditioning_match") is True
        ]
        findings.append(
            f"Normalized trajectory conversion produced {len(normalized_checks)} stitched samples; "
            f"{len(split_free)}/{len(normalized_checks)} stayed as one chunk without fallback splitting."
        )
        findings.append(
            f"Strict train/inference conditioning match is marked true for "
            f"{len(conditioning_exact)}/{len(normalized_checks)} samples. A false value here means the stitcher "
            "used normalization, such as dropping renderer/template boundary tokens, rather than a raw-token prefix."
        )
    elif sample_checks and not trajectory_details:
        findings.append(
            f"Converted samples are pure prompt-snapshot response targets in "
            f"{len(prompt_snapshot_checks)}/{len(sample_checks)} samples "
            "(`loss_mask_sum == response_length`, no zero-mask context spans)."
        )
        findings.append(
            f"Converted sample prompt/response tokens match raw rollout details in "
            f"{len(raw_snapshot_checks)}/{len(sample_checks)} samples."
        )

    train_checks = [
        check
        for case in cases
        for check in case.get("sample_checks", [])
        if check.get("train_found")
    ]
    if train_checks:
        exact = [
            check
            for check in train_checks
            if check.get("train_response_length") == check.get("case_response_length")
            and check.get("train_loss_mask_sum") == check.get("case_loss_mask_sum")
            and check.get("train_zero_spans") == check.get("case_zero_spans")
            and check.get("train_tokens_match_case") is True
            and check.get("train_loss_mask_match_case") is True
        ]
        findings.append(
            f"Train data matches converted samples in {len(exact)}/{len(train_checks)} records "
            "(tokens, response length, loss-mask sum, and zero-mask spans)."
        )
        advantage_summaries = [
            check.get("train_advantage_summary")
            for check in train_checks
            if isinstance(check.get("train_advantage_summary"), dict)
        ]
        if advantage_summaries:
            max_abs = max(float(summary.get("max_abs") or 0.0) for summary in advantage_summaries)
            findings.append(
                "Train-data advantage tensors are present; maximum absolute advantage in indexed records is "
                f"{max_abs:.6g}."
            )
    else:
        findings.append("No train_data records were supplied for this report; checks stop at rollout/sample dumps.")

    copied_or_summarized = [
        item
        for case in cases
        for item in case.get("trajectory_files", [])
        if item.get("copied_steps")
        or item.get("continued_trajectory_ref")
        or item.get("subagent_ref_count")
        or item.get("continuation_index") is not None
    ]
    if copied_or_summarized:
        findings.append(
            "At least one raw trajectory contains copied context, continuation metadata, or subagent references. "
            "Those fields are raw ATIF evidence of history rewriting/summarization paths; the training sample is still "
            "built from `rollout_details`, so copied context can appear as prompt tokens but is not assigned loss."
        )

    return findings


def _case_findings(case: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    for detail in case.get("details", []):
        raw_outputs = detail.get("raw_output_tokens")
        v030_mask = detail.get("v030_loss_mask_sum")
        stitched_mask = detail.get("stitched_loss_mask_sum")
        if isinstance(raw_outputs, int) and isinstance(v030_mask, int) and v030_mask < raw_outputs:
            findings.append(
                "Native v0.3 merge would drop or hide earlier assistant tokens for this detail "
                f"({v030_mask}/{raw_outputs} assistant tokens remain trainable)."
            )
        if isinstance(raw_outputs, int) and stitched_mask == raw_outputs:
            findings.append(
                "Stitched conversion keeps every assistant output trainable "
                f"({stitched_mask}/{raw_outputs} assistant tokens are masked as loss=1)."
            )
        if detail.get("stitched_zero_spans"):
            findings.append(
                "Zero-mask spans in the stitched response are re-rendered context/terminal tokens, "
                "not model outputs."
            )

    train_checks = [check for check in case.get("sample_checks", []) if check.get("train_found")]
    if train_checks:
        exact = [
            check
            for check in train_checks
            if check.get("train_response_length") == check.get("case_response_length")
            and check.get("train_loss_mask_sum") == check.get("case_loss_mask_sum")
            and check.get("train_zero_spans") == check.get("case_zero_spans")
            and check.get("train_tokens_match_case") is True
            and check.get("train_loss_mask_match_case") is True
        ]
        if len(exact) == len(train_checks):
            findings.append("The final train_data record matches the converted sample tokens and masks for this case.")
        else:
            findings.append("At least one train_data record differs from the converted sample tokens or masks for this case.")
        advantage_summaries = [
            check.get("train_advantage_summary")
            for check in train_checks
            if isinstance(check.get("train_advantage_summary"), dict)
        ]
        if advantage_summaries:
            max_abs = max(float(summary.get("max_abs") or 0.0) for summary in advantage_summaries)
            if max_abs == 0:
                findings.append(
                    "Advantages are all zero for this case; this is expected when every sibling in the GRPO group "
                    "has identical normalized reward."
                )
            else:
                findings.append(f"Advantages are non-zero for this case (max_abs={max_abs:.6g}).")

    raw_checks = [
        check
        for check in case.get("sample_checks", [])
        if check.get("sample_prompt_matches_raw") is not None or check.get("sample_response_matches_raw") is not None
    ]
    if raw_checks and all(
        check.get("sample_prompt_matches_raw") is True and check.get("sample_response_matches_raw") is True
        for check in raw_checks
    ):
        findings.append("Each converted step sample exactly matches its raw rollout prompt/completion snapshot.")

    normalized_checks = [
        check for check in case.get("sample_checks", []) if check.get("sample_mode") == "trajectory_normalized"
    ]
    for check in normalized_checks:
        findings.append(
            "Normalized stitching emitted chunk {chunk}/{chunks} over turns {turns}: "
            "context_tokens={context}, output_tokens={output}, zero_spans={zeros}, "
            "conditioning_match={conditioning}, dropped_tokens(prompt_suffix/response_prefix/response_suffix)="
            "{prompt_drop}/{response_prefix_drop}/{response_suffix_drop}.".format(
                chunk=check.get("normalized_chunk"),
                chunks=check.get("normalized_num_chunks"),
                turns=check.get("normalized_chunk_turn_span"),
                context=check.get("normalized_context_tokens"),
                output=check.get("normalized_output_tokens"),
                zeros=check.get("case_zero_spans"),
                conditioning=check.get("conditioning_match"),
                prompt_drop=check.get("normalized_prompt_suffix_dropped_tokens"),
                response_prefix_drop=check.get("normalized_response_prefix_dropped_tokens"),
                response_suffix_drop=check.get("normalized_response_suffix_dropped_tokens"),
            )
        )

    trajectory_files = case.get("trajectory_files", [])
    copied_steps = sum(int(item.get("copied_steps") or 0) for item in trajectory_files)
    subagent_refs = sum(int(item.get("subagent_ref_count") or 0) for item in trajectory_files)
    has_continuation = any(
        item.get("continued_trajectory_ref") or item.get("continuation_index") is not None
        for item in trajectory_files
    )
    if copied_steps or subagent_refs or has_continuation:
        findings.append(
            "Raw trajectory contains copied/continued/subagent history markers "
            f"(copied_steps={copied_steps}, subagent_refs={subagent_refs}, continuation={has_continuation}). "
            "The current converter does not reconstruct hidden pre-summary assistant turns from those markers."
        )

    if not findings:
        findings.append("No conversion or mask issue was detected by the automatic checks for this case.")
    return findings

def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Harbor v0.3 Trajectory Case Study",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Case root: `{payload['case_root']}`",
        f"Rollout dump dir: `{payload.get('rollout_dump_dir') or ''}`",
        f"Train data dir: `{payload.get('train_data_dir') or ''}`",
        "",
        "## Summary",
        "",
        f"- Cases: {len(payload['cases'])}",
        f"- Rollout samples indexed: {payload['num_rollout_samples']}",
        f"- Train records indexed: {payload['num_train_records']}",
        "",
        "## Interpretation",
        "",
    ]
    lines.extend(f"- {finding}" for finding in _global_findings(payload))
    lines.append("")

    for i, case in enumerate(payload["cases"], start=1):
        lines.extend(
            [
                f"## Case {i}",
                "",
                f"- Directory: `{case['case_dir']}`",
                f"- Mode/status/reward: `{case['mode']}` / `{case['status']}` / `{case['reward']}`",
                f"- Rollout details: {case['num_rollout_details']}",
                "",
                "### Raw Trajectory Files",
                "",
            ]
        )
        if not case["trajectory_files"]:
            lines.append("- No copied trajectory files found.")
        for item in case["trajectory_files"]:
            lines.append(
                "- `{path}`: steps={steps}, continuation={cont}, copied_steps={copied}, "
                "llm_call_steps={llm_steps}, subagent_refs={refs}, sources={sources}".format(
                    path=item.get("path"),
                    steps=item.get("steps"),
                    cont=item.get("continued_trajectory_ref"),
                    copied=item.get("copied_steps"),
                    llm_steps=item.get("llm_call_steps"),
                    refs=item.get("subagent_ref_count"),
                    sources=item.get("source_counts"),
                )
            )
            for ref in item.get("subagent_refs") or []:
                lines.append(
                    "  - ref path={path}, agent={agent}, summary={summary}".format(
                        path=ref.get("trajectory_path"),
                        agent=ref.get("agent_name"),
                        summary=ref.get("summary"),
                    )
                )

        lines.extend(["", "### Conversion Details", ""])
        for detail in case["details"]:
            lines.append(
                "- detail {idx}: turns={turns}, prompt_lens={prompts}, completion_lens={outs}, "
                "raw_output_tokens={raw_outputs}, prefix_mismatch={mismatch}, base_changed={base}, "
                "prefix_drift={drift}, v030_len/mask={v030_resp}/{v030_mask}, "
                "stitched_len/mask={stitched_resp}/{stitched_mask}, stitched_strategies={strategies}, "
                "stitched_context_lens={contexts}, stitched_zero_spans={zeros}".format(
                    idx=detail.get("detail_index"),
                    turns=detail.get("num_turns"),
                    prompts=detail.get("prompt_lengths"),
                    outs=detail.get("completion_lengths"),
                    raw_outputs=detail.get("raw_output_tokens"),
                    mismatch=detail.get("prefix_mismatch_turns"),
                    base=detail.get("base_changed_turns"),
                    drift=detail.get("prefix_drift_turns"),
                    v030_resp=detail.get("v030_response_len"),
                    v030_mask=detail.get("v030_loss_mask_sum"),
                    stitched_resp=detail.get("stitched_response_len"),
                    stitched_mask=detail.get("stitched_loss_mask_sum"),
                    strategies=detail.get("stitch_strategies"),
                    contexts=detail.get("stitched_context_lengths"),
                    zeros=detail.get("stitched_zero_spans"),
                )
            )

        lines.extend(["", "### Sample/Train Data Checks", ""])
        for check in case["sample_checks"]:
            lines.append(
                "- sample {idx}: mode={mode}, detail/step={detail}/{step}, "
                "chunk={chunk}/{chunks} turns={turns}, conditioning_match={cond}, "
                "norm_context/output={nctx}/{nout}, norm_prefix={nprefix}, strict_prefix={sprefix}, "
                "dropped(prompt/resp_prefix/resp_suffix)={pdrop}/{rpdrop}/{rsdrop}, "
                "raw prompt/completion={rprompt}/{rcomp}, raw_match={pmatch}/{rmatch}, "
                "case len/mask={clen}/{cmask}, rollout len/mask={rlen}/{rmask}, "
                "train len/mask={tlen}/{tmask}, group={group}, group_mask_sum={gmask}, "
                "reward/raw={reward}/{raw_reward}, adv={adv}, returns={returns}, "
                "case_zero_spans={czero}, train_zero_spans={tzero}, "
                "train_match={ttokens}/{tmaskmatch}".format(
                    idx=check.get("sample_index"),
                    mode=check.get("sample_mode"),
                    detail=check.get("detail_index"),
                    step=check.get("step_index"),
                    chunk=check.get("normalized_chunk"),
                    chunks=check.get("normalized_num_chunks"),
                    turns=check.get("normalized_chunk_turn_span"),
                    cond=check.get("conditioning_match"),
                    nctx=check.get("normalized_context_tokens"),
                    nout=check.get("normalized_output_tokens"),
                    nprefix=check.get("normalized_prefix_turns"),
                    sprefix=check.get("normalized_strict_prefix_turns"),
                    pdrop=check.get("normalized_prompt_suffix_dropped_tokens"),
                    rpdrop=check.get("normalized_response_prefix_dropped_tokens"),
                    rsdrop=check.get("normalized_response_suffix_dropped_tokens"),
                    rprompt=check.get("raw_prompt_length"),
                    rcomp=check.get("raw_completion_length"),
                    pmatch=check.get("sample_prompt_matches_raw"),
                    rmatch=check.get("sample_response_matches_raw"),
                    clen=check.get("case_response_length"),
                    cmask=check.get("case_loss_mask_sum"),
                    rlen=check.get("rollout_response_length"),
                    rmask=check.get("rollout_loss_mask_sum"),
                    tlen=check.get("train_response_length"),
                    tmask=check.get("train_loss_mask_sum"),
                    group=check.get("train_group_id"),
                    gmask=check.get("train_group_mask_sum"),
                    reward=check.get("train_reward"),
                    raw_reward=check.get("train_raw_reward"),
                    adv=check.get("train_advantage_summary"),
                    returns=check.get("train_return_summary"),
                    czero=check.get("case_zero_spans"),
                    tzero=check.get("train_zero_spans"),
                    ttokens=check.get("train_tokens_match_case"),
                    tmaskmatch=check.get("train_loss_mask_match_case"),
                )
            )
        lines.append("")

        lines.extend(["### Interpretation", ""])
        lines.extend(f"- {finding}" for finding in _case_findings(case))
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Harbor v0.3 trajectory conversion case-study report.")
    parser.add_argument(
        "--root",
        default="/path/to/workspace/slime_harbor_v030_case_study",
    )
    parser.add_argument("--case-subdir", default="cases/no_summary")
    parser.add_argument("--rollout-dump-dir", default=None)
    parser.add_argument("--train-data-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-cases", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.root)
    case_root = root / args.case_subdir
    rollout_dump_dir = Path(args.rollout_dump_dir) if args.rollout_dump_dir else root / "run/no_summary/rollout_dumps"
    train_data_dir = Path(args.train_data_dir) if args.train_data_dir else root / "train_data/no_summary"
    output = Path(args.output) if args.output else root / f"CASE_STUDY_REPORT_{Path(args.case_subdir).name}.md"

    case_dirs = sorted(path.parent for path in case_root.glob("rollout_*/*/conversion.json"))
    if args.max_cases > 0:
        case_dirs = case_dirs[: args.max_cases]

    rollout_samples_by_index = _load_rollout_samples(rollout_dump_dir)
    train_records_by_index = _load_train_records(train_data_dir)

    payload = {
        "case_root": str(case_root),
        "rollout_dump_dir": str(rollout_dump_dir),
        "train_data_dir": str(train_data_dir),
        "num_rollout_samples": len(rollout_samples_by_index),
        "num_train_records": len(train_records_by_index),
        "cases": [
            _case_summary(
                case_dir,
                rollout_samples_by_index=rollout_samples_by_index,
                train_records_by_index=train_records_by_index,
            )
            for case_dir in case_dirs
        ],
    }

    _write_report(output, payload)
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(_to_py(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
