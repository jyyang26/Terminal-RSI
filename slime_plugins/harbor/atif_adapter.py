from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

import numpy as np

from slime.agent.trajectory import TokenSegment, TurnRecord, merge_turns
from slime.utils.types import Sample
from slime_plugins.harbor.routing_replay import RoutingRowsBuilder, decode_routed_experts_payload


_STEP_SAMPLE_MODES = {"step", "step_grouped"}
_GROUPED_STEP_SAMPLE_MODES = {"turn_grouped", "step", "step_grouped"}
_PER_CALL_SAMPLE_MODES = {"turn", "turn_grouped", *_STEP_SAMPLE_MODES}
_RETOKENIZED_TRAJECTORY_SAMPLE_MODES = {"trajectory_retokenized"}
_NORMALIZED_TRAJECTORY_SAMPLE_MODES = {"trajectory_normalized", *_RETOKENIZED_TRAJECTORY_SAMPLE_MODES}


class RolloutConversionError(ValueError):
    """Raised when Harbor rollout details cannot be converted to slime samples."""


def _get_attr_or_item(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_reward(result: Any, reward_key: str = "reward", default: float = 0.0) -> float:
    verifier_result = _get_attr_or_item(result, "verifier_result")
    rewards = _get_attr_or_item(verifier_result, "rewards")
    if not rewards:
        return default
    value = rewards.get(reward_key, default)
    return float(value)


def extract_status(result: Any) -> Sample.Status:
    exception_info = _get_attr_or_item(result, "exception_info")
    if exception_info is not None:
        exception_type = _get_attr_or_item(exception_info, "exception_type", "")
        if "Timeout" in str(exception_type):
            return Sample.Status.TRUNCATED
        return Sample.Status.FAILED
    return Sample.Status.COMPLETED


def extract_rollout_details(result: Any, *, include_subagents: bool = False) -> list[dict[str, Any]]:
    contexts = []
    agent_result = _get_attr_or_item(result, "agent_result")
    if agent_result is not None:
        contexts.append(agent_result)

    for step_result in _get_attr_or_item(result, "step_results", []) or []:
        step_agent_result = _get_attr_or_item(step_result, "agent_result")
        if step_agent_result is not None:
            contexts.append(step_agent_result)

    details: list[dict[str, Any]] = []
    for context in contexts:
        context_details = _get_attr_or_item(context, "rollout_details", []) or []
        for i, detail in enumerate(context_details):
            if i == 0 or include_subagents:
                details.append(dict(detail))
    return details


def samples_from_trial_result(
    result: Any,
    base_sample: Sample,
    *,
    reward_key: str = "reward",
    mode: str = "turn",
    include_subagents: bool = False,
    metadata: dict[str, Any] | None = None,
    require_routing_replay: bool = False,
    score_centering_top_k: int | None = None,
    enable_tb_gae: bool = False,
) -> list[Sample]:
    reward = extract_reward(result, reward_key=reward_key)
    status = extract_status(result)
    details = extract_rollout_details(result, include_subagents=include_subagents)
    sc_kwargs = {}
    if score_centering_top_k is not None:
        sc_kwargs["score_centering_top_k"] = score_centering_top_k
    if enable_tb_gae is True:
        from slime_plugins.harbor.tb_gae_data import trial_bootstrap_metadata

        if include_subagents:
            raise ValueError("TB-GAE first release does not support subagent rollouts")
        metadata = {**(metadata or {}), **trial_bootstrap_metadata(result)}
        if metadata["harbor_termination_kind"] == "truncated":
            status = Sample.Status.TRUNCATED
        sc_kwargs["enable_tb_gae"] = True
    return samples_from_rollout_details(
        details,
        base_sample,
        reward=reward,
        status=status,
        mode=mode,
        metadata={
            **_trial_metadata(result),
            **(metadata or {}),
        },
        require_routing_replay=require_routing_replay,
        **sc_kwargs,
    )


def explain_trial_result_conversion(
    result: Any,
    base_sample: Sample,
    *,
    reward_key: str = "reward",
    mode: str = "turn",
    include_subagents: bool = False,
    metadata: dict[str, Any] | None = None,
    converted_samples: list[Sample] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable case-study view of Harbor-to-slime conversion.

    The explanation is intentionally derived from the same rollout details and
    merge helpers used by the training conversion path. It is not a separate
    converter; it is a debug view for auditing prompt drift, masks and final
    sample fields.
    """
    reward = extract_reward(result, reward_key=reward_key)
    status = extract_status(result)
    details = extract_rollout_details(result, include_subagents=include_subagents)
    return explain_rollout_details_conversion(
        details,
        base_sample,
        reward=reward,
        status=status,
        mode=mode,
        metadata={
            **_trial_metadata(result),
            **(metadata or {}),
        },
        converted_samples=converted_samples,
    )


def explain_rollout_details_conversion(
    details: Iterable[dict[str, Any]],
    base_sample: Sample,
    *,
    reward: float,
    status: Sample.Status,
    mode: str = "turn",
    metadata: dict[str, Any] | None = None,
    converted_samples: list[Sample] | None = None,
) -> dict[str, Any]:
    details = list(details)
    explanation: dict[str, Any] = {
        "mode": mode,
        "reward": reward,
        "status": _status_value(status),
        "base_sample": _sample_debug_dict(base_sample, include_tokens=False),
        "metadata": _json_safe(metadata or {}),
        "num_rollout_details": len(details),
        "details": [],
    }
    if converted_samples is not None:
        explanation["converted_samples"] = [
            _sample_debug_dict(sample, include_tokens=True) for sample in converted_samples
        ]

    for detail_index, detail in enumerate(details):
        detail_debug: dict[str, Any] = {
            "detail_index": detail_index,
            "raw_detail": _rollout_detail_debug_dict(detail),
        }
        if mode == "trajectory":
            detail_debug.update(
                _trajectory_detail_debug(
                    detail,
                    base_sample,
                    reward=reward,
                    status=status,
                    detail_index=detail_index,
                    metadata=metadata,
                )
            )
        explanation["details"].append(detail_debug)
    return explanation


def _status_value(status: Any) -> str:
    return getattr(status, "value", str(status))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Sample.Status):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            return str(value)
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="json"))
        except Exception:
            return str(value)
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value))
        except Exception:
            return str(value)
    return str(value)


def _sample_debug_dict(sample: Sample, *, include_tokens: bool) -> dict[str, Any]:
    response_length = int(getattr(sample, "response_length", 0) or 0)
    tokens = list(getattr(sample, "tokens", []) or [])
    prompt_len = max(0, len(tokens) - response_length)
    loss_mask = getattr(sample, "loss_mask", None)
    rollout_log_probs = getattr(sample, "rollout_log_probs", None)

    payload: dict[str, Any] = {
        "index": getattr(sample, "index", None),
        "group_index": getattr(sample, "group_index", None),
        "group_id": getattr(sample, "group_id", None),
        "tokens_len": len(tokens),
        "prompt_len": prompt_len,
        "response_length": response_length,
        "reward": _json_safe(getattr(sample, "reward", None)),
        "status": _status_value(getattr(sample, "status", None)),
        "loss_mask_len": len(loss_mask) if loss_mask is not None else None,
        "loss_mask_sum": sum(loss_mask) if loss_mask is not None else None,
        "rollout_log_probs_len": len(rollout_log_probs) if rollout_log_probs is not None else None,
        "metadata": _json_safe(getattr(sample, "metadata", None) or {}),
        "train_metadata": _json_safe(getattr(sample, "train_metadata", None)),
    }
    if include_tokens:
        payload.update(
            {
                "tokens": tokens,
                "prompt_token_ids": tokens[:prompt_len],
                "response_token_ids": tokens[prompt_len:],
                "loss_mask": list(loss_mask) if loss_mask is not None else None,
                "rollout_log_probs": list(rollout_log_probs) if rollout_log_probs is not None else None,
                "response": getattr(sample, "response", ""),
            }
        )
    return payload


def _rollout_detail_debug_dict(detail: dict[str, Any]) -> dict[str, Any]:
    prompt_token_ids = _require_nested_int_lists(detail, "prompt_token_ids")
    completion_token_ids = _require_nested_int_lists(detail, "completion_token_ids")
    logprobs = detail.get("logprobs") or []
    return {
        "keys": sorted(str(key) for key in detail.keys()),
        "num_prompt_turns": len(prompt_token_ids),
        "num_completion_turns": len(completion_token_ids),
        "prompt_lengths": [len(ids) for ids in prompt_token_ids],
        "completion_lengths": [len(ids) for ids in completion_token_ids],
        "logprob_lengths": [len(values) if isinstance(values, list) else None for values in logprobs],
        "prompt_token_ids": prompt_token_ids,
        "completion_token_ids": completion_token_ids,
        "logprobs": _json_safe(logprobs),
        "extra": _json_safe(detail.get("extra", {})),
    }


def _turn_records_from_detail(
    detail: dict[str, Any],
    *,
    require_routing_replay: bool = False,
) -> list[TurnRecord]:
    prompt_token_ids = _require_nested_int_lists(detail, "prompt_token_ids")
    completion_token_ids = _require_nested_int_lists(detail, "completion_token_ids")
    logprobs = detail.get("logprobs") or []
    finish_reason_types = _finish_reason_types(detail)

    if len(prompt_token_ids) != len(completion_token_ids):
        raise RolloutConversionError(
            "Harbor rollout detail has mismatched prompt/completion turns: "
            f"{len(prompt_token_ids)} != {len(completion_token_ids)}"
        )
    routed_experts = _turn_routed_experts_from_detail(
        detail,
        prompt_token_ids,
        completion_token_ids,
        require_routing_replay=require_routing_replay,
    )

    turns = []
    for turn_index, (prompt_ids, completion_ids) in enumerate(zip(prompt_token_ids, completion_token_ids, strict=True)):
        finish_reason = finish_reason_types[turn_index] if turn_index < len(finish_reason_types) else ""
        turns.append(
            TurnRecord(
                prompt_ids=list(prompt_ids),
                output_ids=list(completion_ids),
                finish_reason=finish_reason or "",
                output_log_probs=_turn_log_probs(logprobs, turn_index, len(completion_ids)),
                routed_experts=routed_experts[turn_index],
            )
        )
    return turns


def _turn_routed_experts_from_detail(
    detail: dict[str, Any],
    prompt_token_ids: list[list[int]],
    completion_token_ids: list[list[int]],
    *,
    require_routing_replay: bool,
) -> list[np.ndarray | None]:
    encoded_values = _detail_extra_list(detail, "routed_experts")
    shape_values = _detail_extra_list(detail, "routed_experts_shape")
    num_layer_values = _detail_extra_list(detail, "routed_experts_num_layers")
    routed_by_turn: list[np.ndarray | None] = []

    for turn_index, (prompt_ids, completion_ids) in enumerate(
        zip(prompt_token_ids, completion_token_ids, strict=True)
    ):
        encoded = _list_get(encoded_values, turn_index)
        if not isinstance(encoded, str):
            if require_routing_replay:
                raise RolloutConversionError(f"Missing routed_experts for Harbor turn {turn_index}")
            routed_by_turn.append(None)
            continue

        shape = _int_triple(_list_get(shape_values, turn_index))
        if shape is None:
            raise RolloutConversionError(f"Missing routed_experts_shape for Harbor turn {turn_index}")

        num_layers = _as_int(_list_get(num_layer_values, turn_index))
        if num_layers is None:
            num_layers = shape[1]

        try:
            routed_experts = decode_routed_experts_payload(
                encoded,
                shape=shape,
                num_layers=num_layers,
            )
        except (TypeError, ValueError) as exc:
            raise RolloutConversionError(f"Invalid routed_experts for Harbor turn {turn_index}: {exc}") from exc

        expected_rows = len(prompt_ids) + len(completion_ids) - 1
        if routed_experts.shape[0] != expected_rows:
            raise RolloutConversionError(
                "Harbor routed_experts row count mismatch at "
                f"turn {turn_index}: {routed_experts.shape[0]} != {expected_rows}"
            )
        routed_by_turn.append(routed_experts)

    return routed_by_turn


def _turn_record_debug_dict(turn: TurnRecord, turn_index: int) -> dict[str, Any]:
    return {
        "turn_index": turn_index,
        "prompt_len": len(turn.prompt_ids),
        "output_len": len(turn.output_ids),
        "finish_reason": turn.finish_reason,
        "prompt_ids": list(turn.prompt_ids),
        "output_ids": list(turn.output_ids),
        "output_log_probs": list(turn.output_log_probs),
    }


def _segment_debug_dict(segment) -> dict[str, Any]:
    return {
        "prompt_len": len(segment.prompt_ids),
        "response_len": len(segment.response_ids),
        "loss_mask_len": len(segment.loss_mask),
        "loss_mask_sum": sum(segment.loss_mask),
        "rollout_log_probs_len": len(segment.rollout_log_probs),
        "prompt_ids": list(segment.prompt_ids),
        "response_ids": list(segment.response_ids),
        "loss_mask": list(segment.loss_mask),
        "rollout_log_probs": list(segment.rollout_log_probs),
        "metadata": _json_safe(segment.metadata),
    }


def _trajectory_alignment_debug(turns: list[TurnRecord]) -> list[dict[str, Any]]:
    if not turns:
        return []

    prompt_ids = list(turns[0].prompt_ids)
    response_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []
    output_spans: list[tuple[int, int]] = []
    alignment: list[dict[str, Any]] = []

    for turn_index, turn in enumerate(turns):
        record: dict[str, Any] = {
            "turn_index": turn_index,
            "prompt_len": len(turn.prompt_ids),
            "output_len": len(turn.output_ids),
            "finish_reason": turn.finish_reason,
            "base_changed": False,
            "prefix_drift": False,
            "matched_response_prefix_len": None,
            "context_tail_len": 0,
            "context_tail_token_ids": [],
            "dropped_response_tokens": 0,
            "masked_prior_output_tokens": 0,
            "response_len_before_turn": len(response_ids),
        }

        if turn_index > 0:
            if turn.prompt_ids[: len(prompt_ids)] != prompt_ids:
                record["base_changed"] = True
                record["dropped_response_tokens"] = len(response_ids)
                prompt_ids = list(turn.prompt_ids)
                response_ids = []
                loss_mask = []
                rollout_log_probs = []
                output_spans = []
            else:
                prompt_suffix = turn.prompt_ids[len(prompt_ids) :]
                matched_len = _common_prefix_len(response_ids, prompt_suffix)
                record["matched_response_prefix_len"] = matched_len

                if matched_len < len(response_ids):
                    record["prefix_drift"] = True
                    record["dropped_response_tokens"] = len(response_ids) - matched_len
                    for start, end in output_spans:
                        if start < matched_len < end:
                            masked = matched_len - start
                            record["masked_prior_output_tokens"] += masked
                            loss_mask[start:matched_len] = [0] * masked
                            rollout_log_probs[start:matched_len] = [0.0] * masked
                    response_ids = response_ids[:matched_len]
                    loss_mask = loss_mask[:matched_len]
                    rollout_log_probs = rollout_log_probs[:matched_len]
                    output_spans = [
                        (start, min(end, matched_len)) for start, end in output_spans if start < matched_len
                    ]

                context_tail = prompt_suffix[matched_len:]
                record["context_tail_len"] = len(context_tail)
                record["context_tail_token_ids"] = list(context_tail)
                response_ids.extend(context_tail)
                loss_mask.extend([0] * len(context_tail))
                rollout_log_probs.extend([0.0] * len(context_tail))

        output_start = len(response_ids)
        response_ids.extend(turn.output_ids)
        loss_mask.extend([1] * len(turn.output_ids))
        rollout_log_probs.extend(_turn_log_probs([turn.output_log_probs], 0, len(turn.output_ids)))
        output_end = len(response_ids)
        output_spans.append((output_start, output_end))

        record.update(
            {
                "output_response_span": [output_start, output_end],
                "response_len_after_turn": len(response_ids),
                "loss_mask_sum_after_turn": sum(loss_mask),
            }
        )
        alignment.append(record)

    return alignment


def _trajectory_detail_debug(
    detail: dict[str, Any],
    base_sample: Sample,
    *,
    reward: float,
    status: Sample.Status,
    detail_index: int,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    turns = _turn_records_from_detail(detail)
    prompt_token_ids = [turn.prompt_ids for turn in turns]
    completion_token_ids = [turn.output_ids for turn in turns]
    prefix_mismatch_turns = _strict_prefix_mismatch_turns(prompt_token_ids, completion_token_ids)
    v030_segment = merge_turns(
        turns,
        metadata={
            **(metadata or {}),
            "harbor_detail_index": detail_index,
            "harbor_sample_mode": "trajectory",
        },
    )
    segment, stitch_alignment = _stitch_turns_preserving_outputs(
        turns,
        metadata={
            **(metadata or {}),
            "harbor_detail_index": detail_index,
            "harbor_sample_mode": "trajectory",
        },
    )
    sample = _trajectory_sample_from_detail(
        detail,
        base_sample,
        reward=reward,
        status=status,
        detail_index=detail_index,
        metadata=metadata,
    )

    return {
        "turn_records": [_turn_record_debug_dict(turn, i) for i, turn in enumerate(turns)],
        "alignment": _trajectory_alignment_debug(turns),
        "stitched_alignment": stitch_alignment,
        "prefix_mismatch_turns": prefix_mismatch_turns,
        "v030_merged_segment": _segment_debug_dict(v030_segment) if v030_segment is not None else None,
        "merged_segment": _segment_debug_dict(segment) if segment is not None else None,
        "sample": _sample_debug_dict(sample, include_tokens=True),
    }


def samples_from_rollout_details(
    details: Iterable[dict[str, Any]],
    base_sample: Sample,
    *,
    reward: float,
    status: Sample.Status,
    mode: str = "turn",
    metadata: dict[str, Any] | None = None,
    require_routing_replay: bool = False,
    score_centering_top_k: int | None = None,
    enable_tb_gae: bool = False,
) -> list[Sample]:
    converted: list[Sample] = []
    sc_kwargs = {}
    if score_centering_top_k is not None:
        if mode not in _NORMALIZED_TRAJECTORY_SAMPLE_MODES:
            raise RolloutConversionError("SC only supports normalized trajectory modes")
        sc_kwargs["score_centering_top_k"] = score_centering_top_k
    details = list(details)
    if enable_tb_gae is True:
        if mode != "trajectory_retokenized" or len(details) != 1:
            raise ValueError("TB-GAE requires one trajectory_retokenized detail")
        if status not in {Sample.Status.COMPLETED, Sample.Status.TRUNCATED}:
            raise ValueError("TB-GAE requires completed or explicitly truncated trial")
        sc_kwargs["enable_tb_gae"] = True
    if not details:
        return converted

    for detail_index, detail in enumerate(details):
        if mode in _PER_CALL_SAMPLE_MODES:
            converted.extend(
                _turn_samples_from_detail(
                    detail,
                    base_sample,
                    reward=reward,
                    status=status,
                    detail_index=detail_index,
                    metadata=metadata,
                    sample_mode=mode,
                    require_routing_replay=require_routing_replay,
                )
            )
        elif mode == "last_turn":
            turns = _turn_samples_from_detail(
                detail,
                base_sample,
                reward=reward,
                status=status,
                detail_index=detail_index,
                metadata=metadata,
                sample_mode="turn",
                require_routing_replay=require_routing_replay,
            )
            if turns:
                converted.append(turns[-1])
        elif mode == "trajectory":
            converted.append(
                _trajectory_sample_from_detail(
                    detail,
                    base_sample,
                    reward=reward,
                    status=status,
                    detail_index=detail_index,
                    metadata=metadata,
                    require_routing_replay=require_routing_replay,
                )
            )
        elif mode in _NORMALIZED_TRAJECTORY_SAMPLE_MODES:
            converted.extend(
                _normalized_trajectory_samples_from_detail(
                    detail,
                    base_sample,
                    reward=reward,
                    status=status,
                    detail_index=detail_index,
                    metadata=metadata,
                    sample_mode=mode,
                    require_routing_replay=require_routing_replay,
                    **sc_kwargs,
                )
            )
        else:
            raise RolloutConversionError(f"Unsupported Harbor sample conversion mode: {mode!r}")

    return converted


def _turn_samples_from_detail(
    detail: dict[str, Any],
    base_sample: Sample,
    *,
    reward: float,
    status: Sample.Status,
    detail_index: int,
    metadata: dict[str, Any] | None,
    sample_mode: str,
    require_routing_replay: bool = False,
) -> list[Sample]:
    prompt_token_ids = _require_nested_int_lists(detail, "prompt_token_ids")
    completion_token_ids = _require_nested_int_lists(detail, "completion_token_ids")
    logprobs = detail.get("logprobs") or []
    finish_reason_types = _finish_reason_types(detail)
    length_finish_count = sum(1 for value in finish_reason_types if value == "length")
    total_output_tokens = sum(len(ids) for ids in completion_token_ids)
    shared_group_id = _trajectory_group_id(base_sample, metadata, detail_index)

    if len(prompt_token_ids) != len(completion_token_ids):
        raise RolloutConversionError(
            "Harbor rollout detail has mismatched prompt/completion turns: "
            f"{len(prompt_token_ids)} != {len(completion_token_ids)}"
        )
    routed_experts = _turn_routed_experts_from_detail(
        detail,
        prompt_token_ids,
        completion_token_ids,
        require_routing_replay=require_routing_replay,
    )

    samples = []
    for turn_index, (prompt_ids, completion_ids) in enumerate(zip(prompt_token_ids, completion_token_ids, strict=True)):
        if not completion_ids:
            continue

        sample = _clone_base_sample(base_sample, detail_index=detail_index, turn_index=turn_index)
        sample.tokens = prompt_ids + completion_ids
        sample.response_length = len(completion_ids)
        sample.reward = reward
        sample.status = status
        sample.loss_mask = [1] * sample.response_length
        if sample_mode in _GROUPED_STEP_SAMPLE_MODES:
            sample.group_id = shared_group_id
        sample.metadata = {
            **sample.metadata,
            **(metadata or {}),
            "harbor_detail_index": detail_index,
            "harbor_turn_index": turn_index,
            "harbor_step_index": turn_index,
            "harbor_sample_mode": sample_mode,
            "harbor_num_turns": len(completion_token_ids),
            "harbor_num_steps": len(completion_token_ids),
            "harbor_total_output_tokens": total_output_tokens,
            "harbor_length_finish_turns": length_finish_count,
            "harbor_terminal_reward": reward,
        }
        if sample_mode in _STEP_SAMPLE_MODES:
            sample.metadata.update(
                {
                    "harbor_conditioning_mode": "prompt_snapshot",
                    "harbor_step_grouping": "trajectory",
                    "harbor_train_inference_conditioning_match": True,
                }
            )
        if turn_index < len(finish_reason_types):
            sample.metadata["harbor_finish_reason_type"] = finish_reason_types[turn_index]

        sample.rollout_log_probs = _turn_log_probs(logprobs, turn_index, sample.response_length)
        if routed_experts[turn_index] is not None:
            sample.rollout_routed_experts = _validate_sample_routing(
                routed_experts[turn_index],
                token_count=len(sample.tokens),
            )

        samples.append(sample)
    return samples


def _trajectory_sample_from_detail(
    detail: dict[str, Any],
    base_sample: Sample,
    *,
    reward: float,
    status: Sample.Status,
    detail_index: int,
    metadata: dict[str, Any] | None,
    require_routing_replay: bool = False,
) -> Sample:
    turns = _turn_records_from_detail(detail, require_routing_replay=require_routing_replay)
    prompt_token_ids = [turn.prompt_ids for turn in turns]
    completion_token_ids = [turn.output_ids for turn in turns]
    finish_reason_types = _finish_reason_types(detail)
    length_finish_count = sum(1 for value in finish_reason_types if value == "length")
    total_output_tokens = sum(len(ids) for ids in completion_token_ids)

    if not turns:
        raise RolloutConversionError("Harbor rollout detail has no turns")

    prefix_mismatch_turns = _strict_prefix_mismatch_turns(prompt_token_ids, completion_token_ids)
    segment, _ = _stitch_turns_preserving_outputs(
        turns,
        metadata={
            **(metadata or {}),
            "harbor_detail_index": detail_index,
            "harbor_sample_mode": "trajectory",
        },
        require_routing_replay=require_routing_replay,
    )
    if segment is None:
        raise RolloutConversionError("Harbor rollout detail has no turns")

    sample = _clone_base_sample(base_sample, detail_index=detail_index, turn_index=0)
    sample.tokens = list(segment.prompt_ids) + list(segment.response_ids)
    sample.response_length = len(segment.response_ids)
    sample.reward = reward
    sample.status = status
    sample.loss_mask = list(segment.loss_mask)
    sample.group_id = _trajectory_group_id(base_sample, metadata, detail_index)
    if len(segment.rollout_log_probs) != sample.response_length:
        raise RolloutConversionError(
            "Internal rollout_log_probs mismatch: "
            f"{len(segment.rollout_log_probs)} != {sample.response_length}"
        )
    sample.rollout_log_probs = list(segment.rollout_log_probs)
    if segment.rollout_routed_experts is not None:
        sample.rollout_routed_experts = segment.rollout_routed_experts
    sample.metadata = {
        **sample.metadata,
        **(metadata or {}),
        **(segment.metadata or {}),
        "harbor_detail_index": detail_index,
        "harbor_sample_mode": "trajectory",
        "harbor_num_turns": len(completion_token_ids),
        "harbor_total_output_tokens": total_output_tokens,
        "harbor_length_finish_turns": length_finish_count,
    }
    if prefix_mismatch_turns:
        sample.metadata["harbor_trajectory_prefix_mismatch_turns"] = len(prefix_mismatch_turns)
        sample.metadata["harbor_trajectory_prefix_mismatch_first_turn"] = prefix_mismatch_turns[0]
    if finish_reason_types:
        sample.metadata["harbor_finish_reason_type"] = finish_reason_types[-1]
    if len(sample.loss_mask) != sample.response_length:
        raise RolloutConversionError(
            f"Internal loss mask mismatch: {len(sample.loss_mask)} != {sample.response_length}"
        )
    return sample


def _normalized_trajectory_samples_from_detail(
    detail: dict[str, Any],
    base_sample: Sample,
    *,
    reward: float,
    status: Sample.Status,
    detail_index: int,
    metadata: dict[str, Any] | None,
    sample_mode: str,
    require_routing_replay: bool = False,
    score_centering_top_k: int | None = None,
    enable_tb_gae: bool = False,
) -> list[Sample]:
    turns = _turn_records_from_detail(detail, require_routing_replay=require_routing_replay)
    if not turns:
        raise RolloutConversionError("Harbor rollout detail has no turns")

    sc_turn_heads = None
    if score_centering_top_k is not None:
        from slime.utils.score_centering import decode_turn_head

        envelopes = _detail_extra_list(detail, "score_centering_head")
        if len(envelopes) != len(turns):
            raise RolloutConversionError("SC envelope count differs from Harbor turn count")
        sc_turn_heads = [
            decode_turn_head(
                envelope,
                turn.prompt_ids,
                turn.output_ids,
                turn.output_log_probs,
                k=score_centering_top_k,
            )
            for envelope, turn in zip(envelopes, turns, strict=True)
        ]
    finish_reason_types = _finish_reason_types(detail)
    length_finish_count = sum(1 for value in finish_reason_types if value == "length")
    total_output_tokens = sum(len(turn.output_ids) for turn in turns)
    common_prompt_suffix = _common_prompt_suffix([turn.prompt_ids for turn in turns])
    allow_retokenized_alignment = sample_mode in _RETOKENIZED_TRAJECTORY_SAMPLE_MODES
    completion_texts = _detail_extra_list(detail, "completion_text")
    assistant_history_texts = _detail_extra_list(detail, "last_assistant_message_content")
    assistant_history_spans = _detail_extra_list(detail, "last_assistant_message_span")
    transitions = [
        _normalized_transition_alignment(
            turns[i],
            turns[i + 1],
            common_prompt_suffix,
            retokenized_context=(
                _retokenized_transition_context(
                    turns[i],
                    turns[i + 1],
                    common_prompt_suffix,
                    completion_text=_list_get(completion_texts, i),
                    next_assistant_history_text=_list_get(assistant_history_texts, i + 1),
                    next_assistant_history_span=_list_get(assistant_history_spans, i + 1),
                )
                if allow_retokenized_alignment
                else None
            ),
        )
        for i in range(len(turns) - 1)
    ]

    if enable_tb_gae is True:
        from slime_plugins.harbor.tb_gae_data import build_tb_masks

        if sample_mode != "trajectory_retokenized":
            raise ValueError("TB-GAE only supports trajectory_retokenized")
        if (metadata or {}).get("harbor_termination_kind") not in {"terminated", "truncated"}:
            raise ValueError("TB-GAE requires explicit termination metadata")
        if metadata["harbor_termination_kind"] == "truncated":
            if not metadata.get("harbor_bootstrap_token_ids"):
                raise ValueError("TB-GAE truncated trial needs bootstrap state")
            if metadata.get("harbor_bootstrap_action_count") != total_output_tokens:
                raise ValueError("TB-GAE bootstrap state does not follow all collected actions")
        tb_output_spans = []
    chunks: list[dict[str, Any]] = []
    prompt_ids = list(turns[0].prompt_ids)
    response_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []
    chunk_start_turn = 0
    chunk_started_after_split = False
    stats = _new_normalized_chunk_stats()
    routing_builder = _new_routing_builder(turns[0], require_routing_replay=require_routing_replay)

    def finalize_chunk(end_turn: int) -> None:
        if not response_ids:
            return
        routed_experts = _builder_routing_array(
            routing_builder,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
        )
        chunks.append(
            {
                "start_turn": chunk_start_turn,
                "end_turn": end_turn,
                "prompt_ids": list(prompt_ids),
                "response_ids": list(response_ids),
                "loss_mask": list(loss_mask),
                "rollout_log_probs": list(rollout_log_probs),
                "rollout_routed_experts": routed_experts,
                "routing_placeholder_rows": routing_builder.placeholder_rows if routing_builder is not None else 0,
                "routing_placeholder_loss_rows": (
                    routing_builder.placeholder_loss_rows if routing_builder is not None else 0
                ),
                "stats": dict(stats),
                "started_after_split": chunk_started_after_split,
            }
        )
        if enable_tb_gae is True:
            chunks[-1]["tb_output_spans"] = list(tb_output_spans)

    for turn_index, turn in enumerate(turns):
        output_placeholder_first = False
        if turn_index > 0:
            transition = transitions[turn_index - 1]
            if transition["strategy"] == "split":
                stats["split_after_turns"] += 1
                stats["split_before_turn"] = turn_index
                finalize_chunk(turn_index - 1)

                prompt_ids = list(turn.prompt_ids)
                response_ids = []
                if enable_tb_gae is True:
                    tb_output_spans = []
                loss_mask = []
                rollout_log_probs = []
                chunk_start_turn = turn_index
                chunk_started_after_split = True
                stats = _new_normalized_chunk_stats()
                routing_builder = _new_routing_builder(turn, require_routing_replay=require_routing_replay)
            else:
                context_ids = list(transition["context_ids"])
                response_ids.extend(context_ids)
                loss_mask.extend([0] * len(context_ids))
                rollout_log_probs.extend([0.0] * len(context_ids))
                stats["context_tokens"] += len(context_ids)
                placeholder_first_context = (
                    transition["strategy"] == "retokenized_prefix"
                    or int(transition.get("response_suffix_dropped_tokens") or 0) > 0
                )
                _append_context_routing(
                    routing_builder,
                    turn,
                    source_start=int(transition["routing_context_start"]),
                    token_count=len(context_ids),
                    placeholder_first=placeholder_first_context,
                )
                output_placeholder_first = placeholder_first_context and not context_ids
                if transition["strategy"] == "strict_prefix":
                    stats["strict_prefix_turns"] += 1
                elif transition["strategy"] == "retokenized_prefix":
                    stats["retokenized_prefix_turns"] += 1
                    stats["prompt_suffix_dropped_tokens"] += int(transition.get("prompt_suffix_dropped_tokens") or 0)
                    stats["retokenized_prompt_history_tokens"] += int(
                        transition.get("retokenized_prompt_history_tokens") or 0
                    )
                else:
                    stats["normalized_prefix_turns"] += 1
                    stats["prompt_suffix_dropped_tokens"] += int(transition.get("prompt_suffix_dropped_tokens") or 0)
                    stats["response_prefix_dropped_tokens"] += int(
                        transition.get("response_prefix_dropped_tokens") or 0
                    )
                    stats["response_suffix_dropped_tokens"] += int(
                        transition.get("response_suffix_dropped_tokens") or 0
                    )

        if enable_tb_gae is True:
            tb_output_spans.append((len(response_ids), len(response_ids) + len(turn.output_ids)))
        response_ids.extend(turn.output_ids)
        loss_mask.extend([1] * len(turn.output_ids))
        rollout_log_probs.extend(_turn_log_probs([turn.output_log_probs], 0, len(turn.output_ids)))
        _append_output_routing(
            routing_builder,
            turn,
            placeholder_first=output_placeholder_first,
        )
        stats["output_tokens"] += len(turn.output_ids)

    finalize_chunk(len(turns) - 1)

    samples: list[Sample] = []
    group_id = _trajectory_group_id(base_sample, metadata, detail_index)
    for chunk_index, chunk in enumerate(chunks):
        sample = _clone_base_sample(base_sample, detail_index=detail_index, turn_index=chunk["start_turn"])
        sample.tokens = list(chunk["prompt_ids"]) + list(chunk["response_ids"])
        sample.response_length = len(chunk["response_ids"])
        sample.reward = reward
        sample.status = status
        sample.loss_mask = list(chunk["loss_mask"])
        sample.group_id = group_id
        if len(chunk["rollout_log_probs"]) != sample.response_length:
            raise RolloutConversionError(
                "Internal rollout_log_probs mismatch: "
                f"{len(chunk['rollout_log_probs'])} != {sample.response_length}"
            )
        sample.rollout_log_probs = list(chunk["rollout_log_probs"])
        if chunk["rollout_routed_experts"] is not None:
            sample.rollout_routed_experts = chunk["rollout_routed_experts"]
        chunk_stats = chunk["stats"]
        approximate_conditioning = bool(
            chunk_stats.get("normalized_prefix_turns")
            or chunk_stats.get("retokenized_prefix_turns")
            or chunk_stats.get("prompt_suffix_dropped_tokens")
            or chunk_stats.get("response_prefix_dropped_tokens")
            or chunk_stats.get("response_suffix_dropped_tokens")
        )
        conditioning_mode = (
            "retokenized_trajectory"
            if sample_mode in _RETOKENIZED_TRAJECTORY_SAMPLE_MODES
            else "normalized_trajectory"
        )
        sample.metadata = {
            **sample.metadata,
            **(metadata or {}),
            "harbor_detail_index": detail_index,
            "harbor_sample_mode": sample_mode,
            "harbor_num_turns": len(turns),
            "harbor_total_output_tokens": total_output_tokens,
            "harbor_length_finish_turns": length_finish_count,
            "harbor_trajectory_chunk_index": chunk_index,
            "harbor_trajectory_num_chunks": len(chunks),
            "harbor_trajectory_chunk_start_turn": chunk["start_turn"],
            "harbor_trajectory_chunk_end_turn": chunk["end_turn"],
            "harbor_trajectory_chunk_started_after_split": bool(chunk["started_after_split"]),
            "harbor_trajectory_normalized_common_suffix_candidate_tokens": len(common_prompt_suffix),
            "harbor_trajectory_normalized_context_tokens": chunk_stats["context_tokens"],
            "harbor_trajectory_normalized_output_tokens": chunk_stats["output_tokens"],
            "harbor_trajectory_normalized_strict_prefix_turns": chunk_stats["strict_prefix_turns"],
            "harbor_trajectory_normalized_prefix_turns": chunk_stats["normalized_prefix_turns"],
            "harbor_trajectory_retokenized_prefix_turns": chunk_stats["retokenized_prefix_turns"],
            "harbor_trajectory_retokenized_prompt_history_tokens": chunk_stats[
                "retokenized_prompt_history_tokens"
            ],
            "harbor_trajectory_normalized_split_after_turns": chunk_stats["split_after_turns"],
            "harbor_trajectory_normalized_prompt_suffix_dropped_tokens": chunk_stats[
                "prompt_suffix_dropped_tokens"
            ],
            "harbor_trajectory_normalized_response_prefix_dropped_tokens": chunk_stats[
                "response_prefix_dropped_tokens"
            ],
            "harbor_trajectory_normalized_response_suffix_dropped_tokens": chunk_stats[
                "response_suffix_dropped_tokens"
            ],
            "harbor_conditioning_mode": conditioning_mode,
            "harbor_train_inference_conditioning_match": not approximate_conditioning,
        }
        if chunk["rollout_routed_experts"] is not None:
            sample.metadata.update(
                {
                    "harbor_routing_replay_enabled": True,
                    "harbor_routing_replay_placeholder_rows": int(chunk["routing_placeholder_rows"]),
                    "harbor_routing_replay_placeholder_loss_rows": int(
                        chunk["routing_placeholder_loss_rows"]
                    ),
                }
            )
        if chunk_stats.get("split_before_turn") is not None:
            sample.metadata["harbor_trajectory_normalized_split_before_turn"] = chunk_stats["split_before_turn"]
        if finish_reason_types and chunk["end_turn"] < len(finish_reason_types):
            sample.metadata["harbor_finish_reason_type"] = finish_reason_types[chunk["end_turn"]]
        if len(sample.loss_mask) != sample.response_length:
            raise RolloutConversionError(
                f"Internal loss mask mismatch: {len(sample.loss_mask)} != {sample.response_length}"
            )
        if score_centering_top_k is not None:
            _attach_normalized_sc_head(sample, chunk, turns, transitions, sc_turn_heads, k=score_centering_top_k)
        if enable_tb_gae is True:
            te, done = build_tb_masks(
                sample.response_length, chunk["tb_output_spans"], sample.loss_mask,
                termination_kind=metadata["harbor_termination_kind"], is_last_chunk=chunk_index == len(chunks) - 1,
            )
            sample.metadata["turn_end_mask"] = te
            sample.metadata["is_real_done"] = done
            last = chunk_index == len(chunks) - 1
            sample.metadata["tb_boundary"] = {
                "episode": str(group_id), "chunk_index": chunk_index, "chunk_count": len(chunks),
                "terminal": last and metadata["harbor_termination_kind"] == "terminated",
                "bootstrap_token_ids": metadata.get("harbor_bootstrap_token_ids") if last else None,
            }
            # Full successor context belongs only to the last chunk sidecar.
            sample.metadata.pop("harbor_bootstrap_token_ids", None)
            sample.metadata.pop("harbor_bootstrap_action_count", None)
        samples.append(sample)

    return samples


def _new_normalized_chunk_stats() -> dict[str, int | None]:
    return {
        "context_tokens": 0,
        "output_tokens": 0,
        "strict_prefix_turns": 0,
        "normalized_prefix_turns": 0,
        "retokenized_prefix_turns": 0,
        "retokenized_prompt_history_tokens": 0,
        "split_after_turns": 0,
        "split_before_turn": None,
        "prompt_suffix_dropped_tokens": 0,
        "response_prefix_dropped_tokens": 0,
        "response_suffix_dropped_tokens": 0,
    }


def _common_prompt_suffix(prompt_ids: list[list[int]], *, max_len: int = 96, min_len: int = 2) -> list[int]:
    if len(prompt_ids) < 2:
        return []
    suffix = list(prompt_ids[0][-max_len:])
    for ids in prompt_ids[1:]:
        n = min(len(suffix), len(ids), max_len)
        matched = 0
        while matched < n and suffix[-1 - matched] == ids[-1 - matched]:
            matched += 1
        suffix = suffix[len(suffix) - matched :]
        if not suffix:
            return []
    return suffix if len(suffix) >= min_len else []


def _normalized_transition_alignment(
    turn: TurnRecord,
    next_turn: TurnRecord,
    common_prompt_suffix: list[int],
    *,
    max_response_edge_drop: int = 16,
    retokenized_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    strict_prefix = list(turn.prompt_ids) + list(turn.output_ids)
    if _has_prefix(next_turn.prompt_ids, strict_prefix):
        return {
            "strategy": "strict_prefix",
            "context_ids": list(next_turn.prompt_ids[len(strict_prefix) :]),
            "routing_context_start": len(strict_prefix),
            "prompt_suffix_dropped_tokens": 0,
            "response_prefix_dropped_tokens": 0,
            "response_suffix_dropped_tokens": 0,
        }

    best: dict[str, Any] | None = None
    max_suffix_drop = len(common_prompt_suffix)
    max_leading_drop = min(max_response_edge_drop, len(turn.output_ids))

    for suffix_drop in range(max_suffix_drop + 1):
        prompt_base = _drop_suffix_len(turn.prompt_ids, suffix_drop)
        if not _has_prefix(next_turn.prompt_ids, prompt_base):
            continue

        next_after_prompt = next_turn.prompt_ids[len(prompt_base) :]
        for leading_drop in range(max_leading_drop + 1):
            remaining = len(turn.output_ids) - leading_drop
            max_trailing_drop = min(max_response_edge_drop, remaining)
            for trailing_drop in range(max_trailing_drop + 1):
                output_end = len(turn.output_ids) - trailing_drop if trailing_drop else len(turn.output_ids)
                output_projection = turn.output_ids[leading_drop:output_end]
                if not _has_prefix(next_after_prompt, output_projection):
                    continue

                context_ids = list(next_after_prompt[len(output_projection) :])
                routing_context_start = len(prompt_base) + len(output_projection)
                score = (
                    suffix_drop + leading_drop + trailing_drop,
                    suffix_drop,
                    leading_drop + trailing_drop,
                    -len(output_projection),
                )
                if best is None or score < best["score"]:
                    best = {
                        "score": score,
                        "suffix_drop": suffix_drop,
                        "leading_drop": leading_drop,
                        "trailing_drop": trailing_drop,
                        "context_ids": context_ids,
                        "routing_context_start": routing_context_start,
                    }

    if best is None:
        if retokenized_context is not None:
            return {
                "strategy": "retokenized_prefix",
                "context_ids": list(retokenized_context["context_ids"]),
                "routing_context_start": int(retokenized_context["routing_context_start"]),
                "prompt_suffix_dropped_tokens": int(retokenized_context["prompt_suffix_dropped_tokens"]),
                "response_prefix_dropped_tokens": 0,
                "response_suffix_dropped_tokens": 0,
                "retokenized_prompt_history_tokens": int(
                    retokenized_context["retokenized_prompt_history_tokens"]
                ),
            }

        normalized_projection = _drop_suffix_len(turn.prompt_ids, max_suffix_drop) + list(turn.output_ids)
        return {
            "strategy": "split",
            "context_ids": [],
            "prompt_suffix_dropped_tokens": 0,
            "response_prefix_dropped_tokens": 0,
            "response_suffix_dropped_tokens": 0,
            "matched_prefix_len": _common_prefix_len(normalized_projection, next_turn.prompt_ids),
        }

    return {
        "strategy": "normalized_prefix",
        "context_ids": list(best["context_ids"]),
        "routing_context_start": int(best["routing_context_start"]),
        "prompt_suffix_dropped_tokens": int(best["suffix_drop"]),
        "response_prefix_dropped_tokens": int(best["leading_drop"]),
        "response_suffix_dropped_tokens": int(best["trailing_drop"]),
    }


def _retokenized_transition_context(
    turn: TurnRecord,
    next_turn: TurnRecord,
    common_prompt_suffix: list[int],
    *,
    completion_text: Any,
    next_assistant_history_text: Any,
    next_assistant_history_span: Any,
) -> dict[str, Any] | None:
    if not isinstance(completion_text, str) or not isinstance(next_assistant_history_text, str):
        return None
    if not completion_text:
        return None
    if completion_text != next_assistant_history_text:
        return None

    span = _int_pair(next_assistant_history_span)
    if span is None:
        return None
    span_start, span_end = span
    if span_start < 0 or span_end < span_start or span_end > len(next_turn.prompt_ids):
        return None

    max_suffix_drop = len(common_prompt_suffix)
    best: tuple[int, int] | None = None
    for suffix_drop in range(max_suffix_drop + 1):
        prompt_base = _drop_suffix_len(turn.prompt_ids, suffix_drop)
        if len(prompt_base) > span_start:
            continue
        if not _has_prefix(next_turn.prompt_ids, prompt_base):
            continue
        score = (span_start - len(prompt_base), suffix_drop)
        if best is None or score < best:
            best = score
            best_suffix_drop = suffix_drop

    if best is None:
        return None

    return {
        "context_ids": list(next_turn.prompt_ids[span_end:]),
        "routing_context_start": span_end,
        "prompt_suffix_dropped_tokens": best_suffix_drop,
        "retokenized_prompt_history_tokens": span_end - span_start,
    }


def _detail_extra_list(detail: dict[str, Any], key: str) -> list[Any]:
    extra = detail.get("extra")
    value = extra.get(key) if isinstance(extra, dict) else None
    if value is None:
        value = detail.get(key)
    return list(value) if isinstance(value, list) else []


def _list_get(values: list[Any], index: int) -> Any:
    return values[index] if 0 <= index < len(values) else None


def _int_pair(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def _int_triple(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return int(value[0]), int(value[1]), int(value[2])
    except (TypeError, ValueError):
        return None


def _validate_sample_routing(routed_experts: np.ndarray, *, token_count: int) -> np.ndarray:
    expected_rows = max(0, token_count - 1)
    if routed_experts.shape[0] != expected_rows:
        raise RolloutConversionError(
            f"rollout_routed_experts row count mismatch: {routed_experts.shape[0]} != {expected_rows}"
        )
    return routed_experts


def _new_routing_builder(
    turn: TurnRecord,
    *,
    require_routing_replay: bool,
) -> RoutingRowsBuilder | None:
    if turn.routed_experts is None:
        if require_routing_replay:
            raise RolloutConversionError("Missing routed_experts for Harbor trajectory start turn")
        return None

    builder = RoutingRowsBuilder()
    try:
        builder.seed_prompt(turn.routed_experts, len(turn.prompt_ids))
    except ValueError as exc:
        raise RolloutConversionError(f"Invalid routed_experts prompt seed: {exc}") from exc
    return builder


def _append_context_routing(
    builder: RoutingRowsBuilder | None,
    turn: TurnRecord,
    *,
    source_start: int,
    token_count: int,
    placeholder_first: bool = False,
) -> None:
    if builder is None or token_count <= 0:
        return
    try:
        builder.append_from_source(
            turn.routed_experts,
            source_start=source_start,
            token_count=token_count,
            placeholder_first=placeholder_first,
            first_token_loss=False,
        )
    except ValueError as exc:
        raise RolloutConversionError(f"Invalid routed_experts context stitch: {exc}") from exc


def _append_output_routing(
    builder: RoutingRowsBuilder | None,
    turn: TurnRecord,
    *,
    placeholder_first: bool = False,
) -> None:
    if builder is None or not turn.output_ids:
        return
    try:
        builder.append_from_source(
            turn.routed_experts,
            source_start=len(turn.prompt_ids),
            token_count=len(turn.output_ids),
            placeholder_first=placeholder_first,
            first_token_loss=placeholder_first,
        )
    except ValueError as exc:
        raise RolloutConversionError(f"Invalid routed_experts output stitch: {exc}") from exc


def _builder_routing_array(
    builder: RoutingRowsBuilder | None,
    *,
    prompt_ids: list[int],
    response_ids: list[int],
) -> np.ndarray | None:
    if builder is None:
        return None
    routed_experts = builder.as_array()
    if routed_experts is None:
        return None
    return _validate_sample_routing(
        routed_experts,
        token_count=len(prompt_ids) + len(response_ids),
    )


def _drop_suffix_len(values: list[int], suffix_len: int) -> list[int]:
    if suffix_len <= 0:
        return list(values)
    return list(values[:-suffix_len])


def _stitch_turns_preserving_outputs(
    turns: list[TurnRecord],
    *,
    metadata: dict[str, Any] | None = None,
    require_routing_replay: bool = False,
) -> tuple[TokenSegment | None, list[dict[str, Any]]]:
    """Stitch Terminus-style prompts without dropping generated actions.

    slime.agent.trajectory.merge_turns assumes later prompts are strict
    continuations of the original chat token stream. Terminus2 re-renders each
    decision point from the current terminal state, so a later prompt often has
    a different base even without summarization. For Harbor training we keep
    every assistant output as a supervised target and insert any intervening
    prompt/context tokens as response-side tokens with loss mask 0.
    """
    if not turns:
        return None, []

    prompt_ids = list(turns[0].prompt_ids)
    response_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []
    alignment: list[dict[str, Any]] = []
    full_prompt_turns = 0
    suffix_prompt_turns = 0
    prefix_tail_turns = 0
    context_tokens = 0
    output_tokens = 0
    routing_builder = _new_routing_builder(turns[0], require_routing_replay=require_routing_replay)

    for turn_index, turn in enumerate(turns):
        record: dict[str, Any] = {
            "turn_index": turn_index,
            "prompt_len": len(turn.prompt_ids),
            "output_len": len(turn.output_ids),
            "strategy": "initial_prompt",
            "context_len": 0,
            "context_response_span": [len(response_ids), len(response_ids)],
            "matched_response_prefix_len": None,
            "response_len_before_turn": len(response_ids),
        }

        context_ids: list[int] = []
        context_source_start = 0
        context_placeholder_first = False
        output_placeholder_first = False
        if turn_index > 0:
            full_prefix = prompt_ids + response_ids
            if _has_prefix(turn.prompt_ids, full_prefix):
                context_ids = list(turn.prompt_ids[len(full_prefix) :])
                context_source_start = len(full_prefix)
                record["strategy"] = "prefix_tail"
                record["matched_response_prefix_len"] = len(response_ids)
                prefix_tail_turns += 1
            elif _has_prefix(turn.prompt_ids, prompt_ids):
                prompt_suffix = turn.prompt_ids[len(prompt_ids) :]
                matched_len = _common_prefix_len(response_ids, prompt_suffix)
                context_ids = list(prompt_suffix[matched_len:])
                context_source_start = len(prompt_ids) + matched_len
                context_placeholder_first = matched_len < len(response_ids)
                record["strategy"] = "prompt_suffix_tail"
                record["matched_response_prefix_len"] = matched_len
                suffix_prompt_turns += 1
            else:
                context_ids = list(turn.prompt_ids)
                context_source_start = 0
                context_placeholder_first = True
                record["strategy"] = "full_prompt"
                full_prompt_turns += 1

            context_start = len(response_ids)
            response_ids.extend(context_ids)
            loss_mask.extend([0] * len(context_ids))
            rollout_log_probs.extend([0.0] * len(context_ids))
            _append_context_routing(
                routing_builder,
                turn,
                source_start=context_source_start,
                token_count=len(context_ids),
                placeholder_first=context_placeholder_first,
            )
            output_placeholder_first = context_placeholder_first and not context_ids
            context_end = len(response_ids)
            context_tokens += len(context_ids)
            record["context_len"] = len(context_ids)
            record["context_response_span"] = [context_start, context_end]

        output_start = len(response_ids)
        response_ids.extend(turn.output_ids)
        loss_mask.extend([1] * len(turn.output_ids))
        rollout_log_probs.extend(_turn_log_probs([turn.output_log_probs], 0, len(turn.output_ids)))
        _append_output_routing(
            routing_builder,
            turn,
            placeholder_first=output_placeholder_first,
        )
        output_end = len(response_ids)
        output_tokens += len(turn.output_ids)
        record.update(
            {
                "output_response_span": [output_start, output_end],
                "response_len_after_turn": len(response_ids),
                "loss_mask_sum_after_turn": sum(loss_mask),
            }
        )
        alignment.append(record)

    segment_metadata = {
        **(metadata or {}),
        "harbor_trajectory_stitch_mode": "preserve_outputs",
        "harbor_trajectory_stitch_context_tokens": context_tokens,
        "harbor_trajectory_stitch_output_tokens": output_tokens,
        "harbor_trajectory_stitch_full_prompt_turns": full_prompt_turns,
        "harbor_trajectory_stitch_suffix_prompt_turns": suffix_prompt_turns,
        "harbor_trajectory_stitch_prefix_tail_turns": prefix_tail_turns,
    }
    routed_experts = _builder_routing_array(
        routing_builder,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
    )
    if routed_experts is not None:
        segment_metadata.update(
            {
                "harbor_routing_replay_enabled": True,
                "harbor_routing_replay_placeholder_rows": routing_builder.placeholder_rows
                if routing_builder is not None
                else 0,
                "harbor_routing_replay_placeholder_loss_rows": routing_builder.placeholder_loss_rows
                if routing_builder is not None
                else 0,
            }
        )

    return (
        TokenSegment(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            loss_mask=loss_mask,
            rollout_log_probs=rollout_log_probs,
            rollout_routed_experts=routed_experts,
            metadata=segment_metadata,
        ),
        alignment,
    )


def _clone_base_sample(base_sample: Sample, *, detail_index: int, turn_index: int) -> Sample:
    sample = copy.deepcopy(base_sample)
    sample.reward = None
    sample.loss_mask = None
    sample.rollout_log_probs = None
    sample.response_length = 0
    sample.response = ""
    if sample.index is not None:
        sample.index = sample.index * 1_000_000 + detail_index * 10_000 + turn_index
    return sample


def _require_nested_int_lists(detail: dict[str, Any], key: str) -> list[list[int]]:
    value = detail.get(key)
    if not isinstance(value, list):
        raise RolloutConversionError(f"Harbor rollout detail missing {key!r}")
    nested = []
    for i, item in enumerate(value):
        if not isinstance(item, list) or not all(isinstance(token, int) for token in item):
            raise RolloutConversionError(f"Harbor rollout detail {key}[{i}] must be list[int]")
        nested.append(list(item))
    return nested


def _turn_log_probs(logprobs: Any, turn_index: int, response_length: int) -> list[float]:
    turn_logprobs = logprobs[turn_index] if isinstance(logprobs, list) and turn_index < len(logprobs) else None
    if isinstance(turn_logprobs, list) and len(turn_logprobs) == response_length:
        return [float(v) for v in turn_logprobs]
    if isinstance(turn_logprobs, list):
        raise RolloutConversionError(
            "Harbor rollout detail has mismatched logprobs/completion length at "
            f"turn {turn_index}: {len(turn_logprobs)} != {response_length}"
        )
    return [0.0] * response_length


def _strict_prefix_mismatch_turns(
    prompt_token_ids: list[list[int]],
    completion_token_ids: list[list[int]],
) -> list[int]:
    if not prompt_token_ids:
        return []

    tokens = list(prompt_token_ids[0])
    mismatch_turns: list[int] = []
    for turn_index, completion_ids in enumerate(completion_token_ids):
        tokens.extend(completion_ids)
        if turn_index + 1 >= len(prompt_token_ids):
            continue

        next_prompt = prompt_token_ids[turn_index + 1]
        if _has_prefix(next_prompt, tokens):
            tokens.extend(next_prompt[len(tokens) :])
        else:
            mismatch_turns.append(turn_index)
            tokens = list(next_prompt)
    return mismatch_turns


def _has_prefix(value: list[int], prefix: list[int]) -> bool:
    return len(value) >= len(prefix) and value[: len(prefix)] == prefix


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _trajectory_group_id(base_sample: Sample, metadata: dict[str, Any] | None, detail_index: int) -> int:
    metadata = metadata or {}
    rollout_id = _as_int(metadata.get("harbor_rollout_id"))
    group_index = _as_int(metadata.get("harbor_rollout_group_index"))
    sample_rank = _as_int(
        metadata.get(
            "harbor_sample_rank",
            metadata.get("harbor_trial_sample_rank", getattr(base_sample, "index", None)),
        )
    )
    if rollout_id is not None and group_index is not None and sample_rank is not None:
        return rollout_id * 1_000_000_000_000 + group_index * 100_000_000 + sample_rank * 10_000 + detail_index

    base_group_id = _as_int(getattr(base_sample, "group_id", None))
    if base_group_id is None:
        base_group_id = _as_int(getattr(base_sample, "index", None))
    if base_group_id is None:
        base_group_id = 0
    return base_group_id * 10_000 + detail_index


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finish_reason_types(detail: dict[str, Any]) -> list[str | None]:
    extra = detail.get("extra")
    if not isinstance(extra, dict):
        return []
    values = extra.get("finish_reason")
    if not isinstance(values, list):
        return []

    reason_types: list[str | None] = []
    for value in values:
        if isinstance(value, dict):
            reason_type = value.get("type")
        else:
            reason_type = value
        reason_types.append(str(reason_type) if reason_type is not None else None)
    return reason_types


def _summarization_count(result: Any) -> int | None:
    """Total Terminus2 context-summarization subagents created for this trial.

    Terminus2 always records ``summarization_count`` on ``agent_result.metadata``
    (terminus_2.py run() finally-block), so a value of 0 means "summarization was
    enabled but never triggered" while ``None`` means the field was absent (e.g. a
    non-Terminus agent or a failed conversion). Multi-step trials carry one
    ``agent_result`` per step; we sum across whichever contexts are populated, the
    same way :func:`extract_rollout_details` collects them.
    """
    contexts = []
    agent_result = _get_attr_or_item(result, "agent_result")
    if agent_result is not None:
        contexts.append(agent_result)
    for step_result in _get_attr_or_item(result, "step_results", []) or []:
        step_agent_result = _get_attr_or_item(step_result, "agent_result")
        if step_agent_result is not None:
            contexts.append(step_agent_result)

    total: int | None = None
    for context in contexts:
        context_metadata = _get_attr_or_item(context, "metadata") or {}
        if not isinstance(context_metadata, dict):
            continue
        value = context_metadata.get("summarization_count")
        if value is None:
            continue
        try:
            total = (total or 0) + int(value)
        except (TypeError, ValueError):
            continue
    return total


def _trial_metadata(result: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("trial_name", "task_name", "trial_uri", "source"):
        value = _get_attr_or_item(result, key)
        if value is not None:
            metadata[f"harbor_{key}"] = str(value)
    summarization_count = _summarization_count(result)
    if summarization_count is not None:
        metadata["harbor_summarization_count"] = summarization_count
    return metadata


def _slice_sc_tracks(tokens, mask, sampled_logps, head_ids, head_logps, row_tokens, start, stop):
    """One slice for every response-aligned track; never infer a slice from drop counters."""
    from slime.utils.score_centering import validate_head

    tokens = np.asarray(tokens, dtype="<i4")
    mask = np.asarray(mask, dtype=np.int32)
    sampled_logps = np.asarray(sampled_logps, dtype=np.float32)
    if not 0 <= start <= stop <= len(tokens):
        raise RolloutConversionError("Invalid SC response slice")
    validate_head(head_ids, head_logps, row_tokens, k=head_ids.shape[1])
    if not np.array_equal(tokens, row_tokens):
        raise RolloutConversionError("SC token identity differs before slicing")
    if mask.shape != tokens.shape or sampled_logps.shape != tokens.shape:
        raise RolloutConversionError("SC parallel response tracks differ in length")
    selection = slice(start, stop)
    return tuple(
        np.array(value[selection], copy=True, order="C")
        for value in (tokens, mask, sampled_logps, head_ids, head_logps, row_tokens)
    )


def _attach_normalized_sc_head(sample, chunk, turns, transitions, turn_heads, *, k):
    """Mirror actual response construction; matching-projection drops are not response drops."""
    from slime.utils.score_centering import assert_sampled_logps, validate_head

    count = sample.response_length
    ids = np.empty((count, k), dtype="<i4")
    logps = np.empty((count, k), dtype="<f4")
    row_tokens = np.empty(count, dtype="<i4")
    cursor = 0
    for turn_index in range(chunk["start_turn"], chunk["end_turn"] + 1):
        if turn_index > chunk["start_turn"]:
            transition = transitions[turn_index - 1]
            if transition["strategy"] == "split":
                raise RolloutConversionError("SC chunk crosses a split boundary")
            context = transition["context_ids"]
            end = cursor + len(context)
            if list(chunk["response_ids"][cursor:end]) != list(context):
                raise RolloutConversionError("SC context slice differs from the constructed response")
            ids[cursor:end] = np.arange(k, dtype="<i4")
            logps[cursor:end] = -np.log(float(k))
            row_tokens[cursor:end] = context
            if any(chunk["loss_mask"][cursor:end]):
                raise RolloutConversionError("Tool/context rows must be masked")
            cursor = end
        turn = turns[turn_index]
        hi, hp, rt = turn_heads[turn_index]
        values = _slice_sc_tracks(
            turn.output_ids,
            [1] * len(turn.output_ids),
            turn.output_log_probs,
            hi,
            hp,
            rt,
            0,
            len(turn.output_ids),
        )
        tokens, mask, sampled, hi, hp, rt = values
        end = cursor + len(tokens)
        if end > count or list(chunk["response_ids"][cursor:end]) != tokens.tolist():
            raise RolloutConversionError("SC action token slice differs from the constructed response")
        if list(chunk["loss_mask"][cursor:end]) != mask.tolist():
            raise RolloutConversionError("SC action mask differs from the constructed response")
        if not np.allclose(chunk["rollout_log_probs"][cursor:end], sampled, rtol=0, atol=1e-6):
            raise RolloutConversionError("SC sampled logprobs differ from the constructed response")
        ids[cursor:end], logps[cursor:end], row_tokens[cursor:end] = hi, hp, rt
        cursor = end
    if cursor != count:
        raise RolloutConversionError("SC response was only partially filled")
    expected = np.asarray(sample.tokens[len(sample.tokens) - count :], dtype="<i4")
    if not np.array_equal(expected, row_tokens):
        raise RolloutConversionError("SC final response token identity mismatch")
    validate_head(ids, logps, row_tokens, k=k)
    assert_sampled_logps(ids, logps, row_tokens, sample.rollout_log_probs, sample.loss_mask)
    # Dynamic fields preserve the SC-off dataclass __dict__ and pickle payload.
    sample.rollout_topk_token_ids = ids
    sample.rollout_topk_log_probs = logps
    sample.sc_response_token_ids = row_tokens
