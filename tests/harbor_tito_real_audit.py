#!/usr/bin/env python3
"""Audit real Harbor rollout dumps for token-in/token-out continuity.

This is an offline diagnostic helper, not a CI test.  It loads one or more
``.pt`` dumps, finds rollout detail dictionaries, and checks whether each
turn's completion tokens appear verbatim in the next turn's prompt tokens.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


def _load_path(path: str) -> Any:
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _is_nested_int_lists(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, list) and all(isinstance(token, int) for token in item) for item in value
    )


def _iter_children(obj: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield str(key), value
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield str(index), value
    elif hasattr(obj, "__dict__"):
        for key, value in vars(obj).items():
            yield str(key), value


def _find_rollout_details(obj: Any, *, max_nodes: int = 200_000) -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    stack: list[tuple[str, Any]] = [("$", obj)]
    seen: set[int] = set()
    visited = 0

    while stack and visited < max_nodes:
        path, current = stack.pop()
        visited += 1
        if isinstance(current, (dict, list, tuple)) or hasattr(current, "__dict__"):
            ident = id(current)
            if ident in seen:
                continue
            seen.add(ident)

        if isinstance(current, dict) and _is_nested_int_lists(current.get("prompt_token_ids")):
            if _is_nested_int_lists(current.get("completion_token_ids")):
                found.append((path, current))
                continue

        for key, child in _iter_children(current):
            stack.append((f"{path}.{key}", child))

    return found


def _list_get(value: Any, index: int) -> Any:
    if isinstance(value, list) and 0 <= index < len(value):
        return value[index]
    return None


def _extra_list(detail: dict[str, Any], key: str) -> list[Any]:
    extra = detail.get("extra") or {}
    value = extra.get(key) if isinstance(extra, dict) else None
    return value if isinstance(value, list) else []


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle:
        return 0
    if len(needle) > len(haystack):
        return -1

    failure = [0] * len(needle)
    j = 0
    for i in range(1, len(needle)):
        while j and needle[i] != needle[j]:
            j = failure[j - 1]
        if needle[i] == needle[j]:
            j += 1
            failure[i] = j

    j = 0
    for i, token in enumerate(haystack):
        while j and token != needle[j]:
            j = failure[j - 1]
        if token == needle[j]:
            j += 1
            if j == len(needle):
                return i - len(needle) + 1
    return -1


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        count += 1
    return count


def _common_suffix_len(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(reversed(left), reversed(right), strict=False):
        if a != b:
            break
        count += 1
    return count


@dataclass
class TransitionAudit:
    source_path: str
    detail_path: str
    detail_index: int
    turn_index: int
    prev_output_len: int
    next_prompt_len: int
    exact_subseq_pos: int
    span: Any
    span_valid: bool
    span_len: int
    span_token_exact: bool
    span_common_prefix: int
    span_common_suffix: int
    text_match: bool
    completion_text_len: int | None
    history_text_len: int | None


def _audit_detail(source_path: str, detail_path: str, detail_index: int, detail: dict[str, Any]) -> list[TransitionAudit]:
    prompts = detail["prompt_token_ids"]
    outputs = detail["completion_token_ids"]
    completion_texts = _extra_list(detail, "completion_text")
    assistant_history_texts = _extra_list(detail, "last_assistant_message_content")
    assistant_history_spans = _extra_list(detail, "last_assistant_message_span")

    audits: list[TransitionAudit] = []
    for i in range(min(len(prompts), len(outputs)) - 1):
        prev_output = list(outputs[i])
        next_prompt = list(prompts[i + 1])
        span = _list_get(assistant_history_spans, i + 1)
        span_valid = (
            isinstance(span, (list, tuple))
            and len(span) == 2
            and isinstance(span[0], int)
            and isinstance(span[1], int)
            and 0 <= span[0] <= span[1] <= len(next_prompt)
        )
        span_tokens = next_prompt[span[0] : span[1]] if span_valid else []
        completion_text = _list_get(completion_texts, i)
        history_text = _list_get(assistant_history_texts, i + 1)
        text_match = isinstance(completion_text, str) and isinstance(history_text, str) and completion_text == history_text

        audits.append(
            TransitionAudit(
                source_path=source_path,
                detail_path=detail_path,
                detail_index=detail_index,
                turn_index=i,
                prev_output_len=len(prev_output),
                next_prompt_len=len(next_prompt),
                exact_subseq_pos=_find_subsequence(next_prompt, prev_output),
                span=span,
                span_valid=span_valid,
                span_len=len(span_tokens),
                span_token_exact=span_tokens == prev_output,
                span_common_prefix=_common_prefix_len(prev_output, span_tokens),
                span_common_suffix=_common_suffix_len(prev_output, span_tokens),
                text_match=text_match,
                completion_text_len=len(completion_text) if isinstance(completion_text, str) else None,
                history_text_len=len(history_text) if isinstance(history_text, str) else None,
            )
        )
    return audits


def _iter_input_paths(paths: list[str]) -> Iterable[str]:
    for path in paths:
        if os.path.isdir(path):
            for root, _, filenames in os.walk(path):
                for filename in sorted(filenames):
                    if filename.endswith((".pt", ".json")):
                        yield os.path.join(root, filename)
        else:
            yield path


def _print_summary(audits: list[TransitionAudit]) -> None:
    counts = Counter()
    for audit in audits:
        counts["transitions"] += 1
        counts["exact_subsequence"] += audit.exact_subseq_pos >= 0
        counts["missing_exact_subsequence"] += audit.exact_subseq_pos < 0
        counts["valid_span"] += audit.span_valid
        counts["span_token_exact"] += audit.span_token_exact
        counts["text_match"] += audit.text_match
        counts["retokenized_candidate"] += audit.text_match and audit.span_valid and not audit.span_token_exact

    print("transition_summary")
    for key in [
        "transitions",
        "exact_subsequence",
        "missing_exact_subsequence",
        "valid_span",
        "span_token_exact",
        "text_match",
        "retokenized_candidate",
    ]:
        print(f"  {key}: {counts[key]}")
    if counts["transitions"]:
        exact_rate = counts["exact_subsequence"] / counts["transitions"]
        span_exact_rate = counts["span_token_exact"] / counts["transitions"]
        print(f"  exact_subsequence_rate: {exact_rate:.6f}")
        print(f"  span_token_exact_rate: {span_exact_rate:.6f}")


def _print_examples(audits: list[TransitionAudit], *, limit: int) -> None:
    missing = [audit for audit in audits if audit.exact_subseq_pos < 0]
    missing.sort(
        key=lambda item: (
            not item.text_match,
            not item.span_valid,
            item.span_common_prefix + item.span_common_suffix,
            -item.prev_output_len,
        )
    )
    print(f"missing_exact_examples limit={limit}")
    for audit in missing[:limit]:
        print(
            "  "
            f"file={audit.source_path} detail={audit.detail_index} path={audit.detail_path} "
            f"turn={audit.turn_index}->{audit.turn_index + 1} "
            f"prev_output_len={audit.prev_output_len} next_prompt_len={audit.next_prompt_len} "
            f"span={audit.span} span_valid={audit.span_valid} span_len={audit.span_len} "
            f"span_token_exact={audit.span_token_exact} "
            f"span_common_prefix={audit.span_common_prefix} span_common_suffix={audit.span_common_suffix} "
            f"text_match={audit.text_match} completion_text_len={audit.completion_text_len} "
            f"history_text_len={audit.history_text_len}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Harbor .pt dump files or directories containing .pt files")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--quiet-details", action="store_true", help="Only print per-file counts and final summary")
    args = parser.parse_args()

    all_audits: list[TransitionAudit] = []
    for path in _iter_input_paths(args.paths):
        print(f"loading {path}")
        obj = _load_path(path)
        details = _find_rollout_details(obj)
        print(f"  top_type={type(obj).__module__}.{type(obj).__name__} rollout_details={len(details)}")
        for detail_index, (detail_path, detail) in enumerate(details):
            prompts = detail["prompt_token_ids"]
            outputs = detail["completion_token_ids"]
            if not args.quiet_details:
                print(
                    f"  detail[{detail_index}] path={detail_path} turns={len(prompts)} "
                    f"prompt_lens={[len(ids) for ids in prompts[:6]]} "
                    f"output_lens={[len(ids) for ids in outputs[:6]]} "
                    f"extra_keys={sorted((detail.get('extra') or {}).keys()) if isinstance(detail.get('extra'), dict) else []}"
                )
            all_audits.extend(_audit_detail(path, detail_path, detail_index, detail))

    _print_summary(all_audits)
    _print_examples(all_audits, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
