"""Preserve every action when TB training exceeds the model context cap."""

import copy

import numpy as np

from slime.utils.types import Sample


def split_tb_samples(all_samples, cap):
    if cap < 3:
        raise ValueError("TB-GAE context cap must allow a prompt and response")

    def split(sample):
        boundary = sample.metadata.get("tb_boundary")
        if boundary is None:
            raise ValueError("TB-GAE sample has no chunk boundary metadata")
        total, response = len(sample.tokens), sample.response_length
        if not 0 < response < total:
            raise ValueError("TB-GAE requires nonempty prompt/response")
        if total <= cap:
            return [sample]
        prompt = total - response
        stride = max(1, cap // 2)
        parts = []
        for start in range(0, response, stride):
            stop = min(response, start + stride)
            mask = sample.loss_mask[start:stop]
            if not any(mask):
                continue  # observations are context, never training actions
            first = prompt + start
            end = prompt + stop
            context_start = max(0, end - cap)
            part = copy.copy(sample)
            part.tokens = list(sample.tokens[context_start:end])
            part.response_length = stop - start
            part.loss_mask = list(mask)
            part.rollout_log_probs = list(sample.rollout_log_probs[start:stop])
            part.metadata = dict(sample.metadata)
            for key in ("turn_end_mask", "is_real_done"):
                part.metadata[key] = np.asarray(sample.metadata[key][start:stop], dtype=np.bool_).copy()
            part.metadata["tb_boundary"] = dict(boundary)
            if stop < response:
                part.metadata["tb_boundary"].update(terminal=False, bootstrap_token_ids=None)
            part.metadata["tb_context_dropped_tokens"] = context_start
            part.metadata["tb_original_response_range"] = [start, stop]
            if sample.rollout_routed_experts is not None:
                raise ValueError("TB-GAE hard chunking with routing replay is not supported")
            if part.response_length >= len(part.tokens) or first < context_start:
                raise ValueError("TB-GAE split left no prompt")
            parts.append(part)
        if not parts:
            raise ValueError("TB-GAE split has no action tokens")
        return parts

    def walk(node):
        if not isinstance(node, list):
            raise ValueError("TB-GAE sample container must be a list")
        replacement = []
        for value in node:
            if isinstance(value, Sample):
                replacement.extend(split(value))
            else:
                walk(value)
                replacement.append(value)
        node[:] = replacement

    walk(all_samples)
    episodes = {}

    def collect(node):
        for item in node:
            if isinstance(item, Sample):
                episodes.setdefault(item.metadata["tb_boundary"]["episode"], []).append(item)
            else:
                collect(item)

    collect(all_samples)
    for members in episodes.values():
        # Original list order is trajectory order, before DP packing reorders it.
        for index, member in enumerate(members):
            member.metadata["tb_boundary"].update(chunk_index=index, chunk_count=len(members))
