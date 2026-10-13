from __future__ import annotations

import logging
import os
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def cap_total_tokens_in_samples(args, all_samples, data_source=None) -> None:
    """Trim prompt prefixes while preserving complete response suffixes."""

    cap = _get_cap(args)
    if cap is None:
        return

    total = 0
    trimmed = 0
    max_before = 0
    max_after = 0

    for sample in _iter_samples(all_samples):
        tokens = getattr(sample, "tokens", None)
        response_length = int(getattr(sample, "response_length", 0) or 0)
        if not tokens or response_length <= 0:
            continue

        total += 1
        total_length = len(tokens)
        max_before = max(max_before, total_length)
        keep = _keep_length(total_length, response_length, cap)
        if keep < total_length:
            _record_trim(sample, total_length, keep, cap)
            sample.tokens = list(tokens[-keep:])
            routed_experts = getattr(sample, "rollout_routed_experts", None)
            if routed_experts is not None:
                sample.rollout_routed_experts = _routing_tail_slice(
                    routed_experts,
                    dropped=total_length - keep,
                    keep=keep,
                )
            trimmed += 1
            total_length = keep
        max_after = max(max_after, total_length)

    if trimmed:
        logger.info(
            "Harbor sample token cap trimmed %s/%s samples, cap=%s, max_total_before=%s, max_total_after=%s",
            trimmed,
            total,
            cap,
            max_before,
            max_after,
        )


def cap_total_tokens_in_rollout_data(args, rollout_id: int, rollout_data: dict[str, Any]) -> None:
    """Backstop for debug/train-data paths that bypass sample-level processing."""

    cap = _get_cap(args)
    if cap is None:
        return

    tokens = rollout_data.get("tokens") or []
    response_lengths = rollout_data.get("response_lengths") or []
    total_lengths = rollout_data.get("total_lengths") or []
    rollout_routed_experts = rollout_data.get("rollout_routed_experts")
    metadata = rollout_data.get("metadata")

    trimmed = 0
    max_before = max((int(v) for v in total_lengths), default=0)
    max_after = 0
    for idx, token_ids in enumerate(tokens):
        total_length = int(total_lengths[idx])
        response_length = int(response_lengths[idx])
        keep = _keep_length(total_length, response_length, cap)
        if keep < total_length:
            tokens[idx] = token_ids[-keep:].contiguous() if hasattr(token_ids, "contiguous") else token_ids[-keep:]
            if rollout_routed_experts is not None and idx < len(rollout_routed_experts):
                rollout_routed_experts[idx] = _routing_tail_slice(
                    rollout_routed_experts[idx],
                    dropped=total_length - keep,
                    keep=keep,
                )
            total_lengths[idx] = keep
            trimmed += 1
            if isinstance(metadata, list) and idx < len(metadata) and isinstance(metadata[idx], dict):
                metadata[idx]["harbor_original_total_length"] = total_length
                metadata[idx]["harbor_trimmed_total_length"] = keep
                metadata[idx]["harbor_train_total_token_cap"] = cap
            total_length = keep
        max_after = max(max_after, total_length)

    if trimmed:
        logger.info(
            "Harbor rollout_data token cap trimmed %s/%s samples at rollout %s, cap=%s, "
            "max_total_before=%s, max_total_after=%s",
            trimmed,
            len(tokens),
            rollout_id,
            cap,
            max_before,
            max_after,
        )


def cap_total_tokens_allow_response_trim_in_samples(args, all_samples, data_source=None) -> None:
    """Hard-cap samples for whole-trajectory training.

    Unlike ``cap_total_tokens_in_samples``, this function is allowed to trim the
    oldest response/observation suffix tokens when a trajectory response is
    longer than the cap. It updates ``response_length``, ``loss_mask`` and
    ``rollout_log_probs`` together so the slime training contract still holds.
    """

    cap = _get_cap(args)
    if cap is None:
        return
    if getattr(args, "enable_tb_gae", False) is True:
        from slime_plugins.harbor.tb_gae_chunks import split_tb_samples

        split_tb_samples(all_samples, cap)
        return

    total = 0
    trimmed = 0
    response_trimmed = 0
    max_before = 0
    max_after = 0

    for sample in _iter_samples(all_samples):
        tokens = getattr(sample, "tokens", None)
        response_length = int(getattr(sample, "response_length", 0) or 0)
        if not tokens or response_length <= 0:
            continue

        total += 1
        total_length = len(tokens)
        max_before = max(max_before, total_length)
        trim = _hard_trim_plan(total_length, response_length, cap)
        if trim.keep < total_length:
            loss_mask = getattr(sample, "loss_mask", None)
            if loss_mask is None:
                loss_mask = [1] * response_length
            logprobs = getattr(sample, "rollout_log_probs", None)

            _record_trim(
                sample,
                total_length,
                trim.keep,
                cap,
                response_trimmed_tokens=trim.response_drop,
            )
            sample.tokens = list(tokens[-trim.keep :])
            routed_experts = getattr(sample, "rollout_routed_experts", None)
            if routed_experts is not None:
                sample.rollout_routed_experts = _routing_tail_slice(
                    routed_experts,
                    dropped=total_length - trim.keep,
                    keep=trim.keep,
                )
            if getattr(args, "enable_score_centering", False):
                from slime_plugins.harbor.atif_adapter import _slice_sc_tracks

                tracks = _slice_sc_tracks(
                    tokens[len(tokens) - response_length :],
                    loss_mask,
                    logprobs,
                    sample.rollout_topk_token_ids,
                    sample.rollout_topk_log_probs,
                    sample.sc_response_token_ids,
                    trim.response_drop,
                    trim.response_drop + trim.response_length,
                )
                _, _, _, sample.rollout_topk_token_ids, sample.rollout_topk_log_probs, sample.sc_response_token_ids = (
                    tracks
                )
                if not (sample.sc_response_token_ids == sample.tokens[-trim.response_length :]).all():
                    raise ValueError("SC head/token identity failed after the hard response cap")
            sample.response_length = trim.response_length
            if getattr(args, "enable_tb_gae", False) is True:
                from slime_plugins.harbor.tb_gae_data import slice_tb_masks

                slice_tb_masks(
                    sample.metadata, start=trim.response_drop, stop=trim.response_drop + trim.response_length
                )
            sample.loss_mask = list(loss_mask[trim.response_drop : trim.response_drop + trim.response_length])
            if logprobs is not None and len(logprobs) == response_length:
                sample.rollout_log_probs = list(logprobs[trim.response_drop : trim.response_drop + trim.response_length])
            response_trimmed += trim.response_drop
            trimmed += 1
            total_length = trim.keep
        max_after = max(max_after, total_length)

    if trimmed:
        logger.info(
            "Harbor hard sample token cap trimmed %s/%s samples, cap=%s, "
            "response_trimmed_tokens=%s, max_total_before=%s, max_total_after=%s",
            trimmed,
            total,
            cap,
            response_trimmed,
            max_before,
            max_after,
        )


def cap_total_tokens_allow_response_trim_in_rollout_data(
    args, rollout_id: int, rollout_data: dict[str, Any]
) -> None:
    """Backstop hard cap for converted train_data dictionaries."""

    cap = _get_cap(args)
    if getattr(args, "enable_tb_gae", False) is True:
        if cap is not None and any(int(length) > cap for length in rollout_data["total_lengths"]):
            raise ValueError("TB-GAE token cap must run on Samples before DP scheduling and GAE")
        return
    if getattr(args, "enable_score_centering", False):
        # This callback runs AFTER old-policy/critic/GAE and CP slicing.
        # A late trim would invalidate logprobs, values, heads and group denominators.
        if cap is not None and any(int(length) > cap for length in rollout_data["total_lengths"]):
            raise ValueError("SC token cap must run on Samples before DP scheduling and old-policy forward")
        return
    if cap is None:
        return

    tokens = rollout_data.get("tokens") or []
    response_lengths = rollout_data.get("response_lengths") or []
    total_lengths = rollout_data.get("total_lengths") or []
    loss_masks = rollout_data.get("loss_masks") or []
    rollout_log_probs = rollout_data.get("rollout_log_probs")
    rollout_routed_experts = rollout_data.get("rollout_routed_experts")
    metadata = rollout_data.get("metadata")

    trimmed = 0
    response_trimmed = 0
    max_before = max((int(v) for v in total_lengths), default=0)
    max_after = 0
    for idx, token_ids in enumerate(tokens):
        total_length = int(total_lengths[idx])
        response_length = int(response_lengths[idx])
        trim = _hard_trim_plan(total_length, response_length, cap)
        if trim.keep < total_length:
            tokens[idx] = _tail_slice(token_ids, trim.keep)
            response_lengths[idx] = trim.response_length
            total_lengths[idx] = trim.keep
            if idx < len(loss_masks):
                loss_masks[idx] = _range_slice(loss_masks[idx], trim.response_drop, trim.response_length)
            if rollout_log_probs is not None and idx < len(rollout_log_probs):
                rollout_log_probs[idx] = _range_slice(rollout_log_probs[idx], trim.response_drop, trim.response_length)
            if rollout_routed_experts is not None and idx < len(rollout_routed_experts):
                rollout_routed_experts[idx] = _routing_tail_slice(
                    rollout_routed_experts[idx],
                    dropped=total_length - trim.keep,
                    keep=trim.keep,
                )
            trimmed += 1
            response_trimmed += trim.response_drop
            if isinstance(metadata, list) and idx < len(metadata) and isinstance(metadata[idx], dict):
                metadata[idx]["harbor_original_total_length"] = total_length
                metadata[idx]["harbor_trimmed_total_length"] = trim.keep
                metadata[idx]["harbor_train_total_token_cap"] = cap
                metadata[idx]["harbor_response_trimmed_tokens"] = trim.response_drop
            total_length = trim.keep
        max_after = max(max_after, total_length)

    if trimmed:
        logger.info(
            "Harbor hard rollout_data token cap trimmed %s/%s samples at rollout %s, cap=%s, "
            "response_trimmed_tokens=%s, max_total_before=%s, max_total_after=%s",
            trimmed,
            len(tokens),
            rollout_id,
            cap,
            response_trimmed,
            max_before,
            max_after,
        )


def _get_cap(args) -> int | None:
    value = getattr(args, "harbor_max_train_total_tokens", None)
    if value in (None, ""):
        value = os.environ.get("HARBOR_MAX_TRAIN_TOTAL_TOKENS")
    if value in (None, "", "0", 0):
        return None
    cap = int(value)
    if cap <= 0:
        return None
    return cap


def _keep_length(total_length: int, response_length: int, cap: int) -> int:
    if total_length <= cap:
        return total_length
    if response_length >= total_length:
        return total_length
    return min(total_length, max(cap, response_length + 1))


class _TrimPlan:
    def __init__(self, keep: int, response_drop: int, response_length: int):
        self.keep = keep
        self.response_drop = response_drop
        self.response_length = response_length


def _hard_trim_plan(total_length: int, response_length: int, cap: int) -> _TrimPlan:
    if total_length <= cap:
        return _TrimPlan(total_length, 0, response_length)

    keep = min(total_length, cap)
    dropped = total_length - keep
    prompt_length = max(0, total_length - response_length)
    response_drop = max(0, dropped - prompt_length)
    new_response_length = max(0, response_length - response_drop)

    # Keep at least one context token before the response when possible.
    if keep > 1 and new_response_length >= keep:
        extra_drop = new_response_length - (keep - 1)
        response_drop += extra_drop
        new_response_length = keep - 1

    return _TrimPlan(keep, response_drop, new_response_length)


def _tail_slice(value, keep: int):
    sliced = value[-keep:]
    return sliced.contiguous() if hasattr(sliced, "contiguous") else sliced


def _range_slice(value, start: int, length: int):
    sliced = value[start : start + length]
    return sliced.contiguous() if hasattr(sliced, "contiguous") else sliced


def _routing_tail_slice(value, *, dropped: int, keep: int):
    return _range_slice(value, dropped, max(0, keep - 1))


def _iter_samples(value) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_samples(item)
    else:
        yield value


def _record_trim(
    sample,
    total_length: int,
    keep: int,
    cap: int,
    *,
    response_trimmed_tokens: int = 0,
) -> None:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        sample.metadata = metadata
    metadata["harbor_original_total_length"] = total_length
    metadata["harbor_trimmed_total_length"] = keep
    metadata["harbor_train_total_token_cap"] = cap
    metadata["harbor_response_trimmed_tokens"] = response_trimmed_tokens
