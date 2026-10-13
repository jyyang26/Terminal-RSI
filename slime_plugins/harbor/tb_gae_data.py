"""CPU-only response masks for the explicitly enabled TB-GAE path.

Explicit termination and successor-state contracts for whole episodes, including
truncated and split trajectories. Generation finish_reason is not environment done.
"""

import numpy as np

TB_MASK_KEYS = ("turn_end_mask", "is_real_done")
TB_DATA_KEYS = (*TB_MASK_KEYS, "tb_boundary")


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


class TBGAETrialSkip(ValueError):
    """Raised when a trial should be skipped (infrastructure error, no valid state).

    Distinct from a hard TB-GAE algorithmic failure so callers can treat it as
    a droppable sample rather than a rollout-aborting error.
    """


def trial_termination_kind(result):
    """Read the explicit producer contract; never guess from EOS/status/reward."""
    exception = _get(result, "exception_info")
    if _get(result, "step_results"):
        raise ValueError("TB-GAE first release does not support multi-step trials")
    context = _get(result, "agent_result")
    metadata = _get(context, "metadata", {}) or {}
    kind = metadata.get("termination_kind")
    if exception is not None and _get(exception, "exception_type") not in {
        "AgentTimeoutError",
        "TimeoutError",
        "ContextLengthExceededError",
        "OutputLengthExceededError",
        "CancelledError",
    }:
        # Non-whitelisted exception (e.g. RuntimeError from a broken environment):
        # the agent loop may never have run, so there is no valid truncation state.
        # Raise TBGAETrialSkip so the rollout can drop this trial rather than abort.
        raise TBGAETrialSkip(
            f"TB-GAE skipping trial with unrecoverable exception "
            f"({_get(exception, 'exception_type')!r}); no valid truncation state available. "
            f"kind={kind!r}, reason={metadata.get('termination_reason')!r}"
        )
    if kind not in {"terminated", "truncated"}:
        raise ValueError(
            "TB-GAE requires explicit agent_result.metadata.termination_kind='terminated' or 'truncated'; "
            f"got kind={kind!r}, reason={metadata.get('termination_reason')!r}. "
            "unknown/continuing trials cannot be treated as complete episodes; "
            "missing metadata requires the Terminus-2 producer patch and record_termination_metadata=True"
        )
    if kind == "truncated":
        tokens = metadata.get("bootstrap_token_ids")
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError(
                f"TB-GAE truncated trial is missing a valid bootstrap state: {metadata.get('bootstrap_error')}"
            )
    return kind


def trial_bootstrap_metadata(result):
    kind = trial_termination_kind(result)
    metadata = _get(_get(result, "agent_result"), "metadata", {}) or {}
    return {
        "harbor_termination_kind": kind,
        "harbor_bootstrap_token_ids": metadata.get("bootstrap_token_ids") if kind == "truncated" else None,
        "harbor_bootstrap_action_count": metadata.get("bootstrap_action_count") if kind == "truncated" else None,
    }


def _mask(value, length, name):
    arr = np.asarray(value)
    if arr.ndim != 1 or len(arr) != length:
        raise ValueError(f"TB-GAE {name} must have response length {length}, got {arr.shape}")
    if arr.dtype.kind not in "biuf" or not np.isin(arr, [0, 1]).all():
        raise ValueError(f"TB-GAE {name} must contain only boolean/0/1 values")
    return np.ascontiguousarray(arr, dtype=np.bool_)


def validate_tb_masks(turn_end, done, action_mask, *, require_terminal=True):
    length = len(action_mask)
    action = _mask(action_mask, length, "action_mask")
    te = _mask(turn_end, length, "turn_end_mask")
    dn = _mask(done, length, "is_real_done")
    if (te & ~action).any() or (dn & ~te).any():
        raise ValueError("TB-GAE requires is_real_done <= turn_end_mask <= action_mask")
    indices = np.flatnonzero(action)
    if not len(indices):
        if te.any() or dn.any():
            raise ValueError("TB-GAE inactive samples must have zero masks")
        return te, dn
    if dn.sum() > 1 or (dn.any() and not dn[indices[-1]]):
        raise ValueError("TB-GAE done must occur only at the final active token")
    if require_terminal and not dn[indices[-1]]:
        raise ValueError("TB-GAE needs a real terminal action; missing bootstrap for nonterminal tail")
    return te, dn


def build_tb_masks(response_length, output_spans, action_mask, *, termination_kind, is_last_chunk):
    if termination_kind not in {"terminated", "truncated"}:
        raise ValueError("TB-GAE needs explicit terminated/truncated metadata")
    action = _mask(action_mask, response_length, "action_mask")
    te = np.zeros(response_length, dtype=np.bool_)
    dn = np.zeros(response_length, dtype=np.bool_)
    coverage = np.zeros(response_length, dtype=np.bool_)
    previous_end = 0
    for start, end in output_spans:
        if not 0 <= previous_end <= start <= end <= response_length:
            raise ValueError("TB-GAE output spans must be ordered response-space intervals")
        previous_end = end
        if end > start:
            if not action[start:end].all():
                raise ValueError("TB-GAE output span includes observation tokens")
            coverage[start:end] = True
            te[end - 1] = True
    if not np.array_equal(coverage, action) or not action.any():
        raise ValueError("TB-GAE output spans must cover every active token")
    if termination_kind == "terminated" and is_last_chunk:
        dn[np.flatnonzero(action)[-1]] = True
    return validate_tb_masks(te, dn, action, require_terminal=False)


def validate_and_collect_tb_masks(samples, final_loss_masks):
    if len(samples) != len(final_loss_masks):
        raise ValueError("TB-GAE samples/loss_masks count mismatch")
    result = {key: [] for key in TB_DATA_KEYS}
    for sample, action in zip(samples, final_loss_masks, strict=True):
        metadata = sample.metadata or {}
        if any(key not in metadata for key in TB_MASK_KEYS):
            raise ValueError(f"TB-GAE sample {sample.index} is missing turn_end_mask/is_real_done")
        if len(action) != sample.response_length:
            raise ValueError("TB-GAE final loss_mask/response_length mismatch")
        masks = [_mask(metadata[key], sample.response_length, key) for key in TB_MASK_KEYS]
        boundary = metadata.get("tb_boundary")
        if boundary is None:
            # Legacy complete terminal fixtures remain valid; incomplete tails never guessed.
            boundary = dict(
                episode=str(sample.group_id if sample.group_id is not None else sample.index),
                chunk_index=0,
                chunk_count=1,
                terminal=True,
                bootstrap_token_ids=None,
            )
        if type(boundary.get("terminal")) is not bool or not isinstance(boundary.get("episode"), str):
            raise ValueError("TB-GAE boundary needs a boolean terminal and string episode ID")
        if any(type(boundary.get(k)) is not int for k in ("chunk_index", "chunk_count")):
            raise ValueError("TB-GAE chunk index/count must be integers")
        if not 0 <= boundary["chunk_index"] < boundary["chunk_count"]:
            raise ValueError("TB-GAE chunk index out of range")
        if sample.remove_sample:
            masks = [np.zeros(sample.response_length, dtype=np.bool_) for _ in TB_MASK_KEYS]
        te, dn = validate_tb_masks(*masks, action, require_terminal=boundary["terminal"])
        if not boundary["terminal"] and dn.any():
            raise ValueError("TB-GAE nonterminal chunk cannot contain real done")
        if not boundary["terminal"] and boundary["chunk_index"] == boundary["chunk_count"] - 1:
            if not boundary.get("bootstrap_token_ids"):
                raise ValueError("TB-GAE nonterminal final chunk needs bootstrap_token_ids")
        result["turn_end_mask"].append(te)
        result["is_real_done"].append(dn)
        result["tb_boundary"].append(boundary)
    episodes = {}
    for b in result["tb_boundary"]:
        episodes.setdefault(b["episode"], []).append(b)
    for group in episodes.values():
        count = group[0]["chunk_count"]
        if sorted(b["chunk_index"] for b in group) != list(range(count)) or any(
            b["chunk_count"] != count for b in group
        ):
            raise ValueError("TB-GAE episode chunks incomplete/duplicated before DP distribution")
    return result


def slice_tb_masks(metadata, *, start, stop):
    if any(key not in metadata for key in TB_MASK_KEYS):
        raise ValueError("TB-GAE hard cap input is missing masks")
    lengths = [len(metadata[key]) for key in TB_MASK_KEYS]
    if lengths[0] != lengths[1] or not 0 <= start <= stop <= lengths[0]:
        raise ValueError("TB-GAE hard cap slice is outside response masks")
    for key in TB_MASK_KEYS:
        metadata[key] = _mask(metadata[key], lengths[0], key)[start:stop].copy()
