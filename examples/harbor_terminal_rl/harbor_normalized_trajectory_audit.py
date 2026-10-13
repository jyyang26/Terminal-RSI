#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ASSISTANT_PREFILL_TAILS = (
    "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "<|im_start|>assistant\n<think>\n</think>\n\n",
)

ASSISTANT_HISTORY_PREFIX = "<|im_start|>assistant\n"

RESPONSE_LEADING_ARTIFACTS = (
    "</think>\n\n",
    "</think>\n",
    "<think>\n\n</think>\n\n",
    "<think>\n</think>\n\n",
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _strip_assistant_generation_prefill(prompt_text: str) -> tuple[str, str | None]:
    for tail in ASSISTANT_PREFILL_TAILS:
        if prompt_text.endswith(tail):
            return prompt_text[: -len(tail)] + ASSISTANT_HISTORY_PREFIX, tail
    return prompt_text, None


def _strip_response_leading_artifact(response_text: str) -> tuple[str, str | None]:
    stripped = response_text.lstrip()
    for artifact in RESPONSE_LEADING_ARTIFACTS:
        if stripped.startswith(artifact):
            return stripped[len(artifact) :], artifact
    return response_text, None


def _history_projection(prompt_text: str, response_text: str) -> tuple[str, dict[str, Any]]:
    normalized_prompt, prompt_artifact = _strip_assistant_generation_prefill(prompt_text)
    normalized_response, response_artifact = _strip_response_leading_artifact(response_text)
    return (
        normalized_prompt + normalized_response,
        {
            "prompt_artifact": prompt_artifact,
            "response_artifact": response_artifact,
        },
    )


def _shorten(text: str, limit: int = 220) -> str:
    text = text.replace("\r", "\\r")
    if len(text) <= limit:
        return text
    half = max(1, (limit - 5) // 2)
    return text[:half] + " ... " + text[-half:]


def _mismatch_window(expected: str, actual: str, prefix_len: int, width: int = 180) -> dict[str, str]:
    start = max(0, prefix_len - width)
    end = prefix_len + width
    return {
        "expected": expected[start:end],
        "actual": actual[start:end],
    }


def _estimate_tail_tokens(next_prompt_text: str, next_prompt_tokens: int, tail_chars: int) -> int:
    if not next_prompt_text or next_prompt_tokens <= 0:
        return 0
    chars_per_token = len(next_prompt_text) / max(1, next_prompt_tokens)
    return int(round(tail_chars / max(chars_per_token, 1e-9)))


def _classify_transition(turn: dict[str, Any], next_turn: dict[str, Any]) -> dict[str, Any]:
    prompt_text = str(turn.get("prompt_text") or "")
    response_text = str(turn.get("response_text") or "")
    next_prompt_text = str(next_turn.get("prompt_text") or "")
    next_prompt_tokens = int(next_turn.get("prompt_length") or 0)

    strict_projection = prompt_text + response_text
    if next_prompt_text.startswith(strict_projection):
        tail_chars = len(next_prompt_text) - len(strict_projection)
        return {
            "classification": "strict_prefix",
            "prefix_chars": len(strict_projection),
            "projection_chars": len(strict_projection),
            "next_prompt_chars": len(next_prompt_text),
            "tail_chars": tail_chars,
            "tail_tokens_estimate": _estimate_tail_tokens(next_prompt_text, next_prompt_tokens, tail_chars),
            "next_prompt_tokens": next_prompt_tokens,
            "raw_full_prompt_tokens_if_reset": 0,
            "conditioning_note": "exact raw prompt+completion prefix",
        }

    normalized_projection, artifacts = _history_projection(prompt_text, response_text)
    if next_prompt_text.startswith(normalized_projection):
        tail_chars = len(next_prompt_text) - len(normalized_projection)
        return {
            "classification": "qwen_template_normalized_prefix",
            "prefix_chars": len(normalized_projection),
            "projection_chars": len(normalized_projection),
            "next_prompt_chars": len(next_prompt_text),
            "tail_chars": tail_chars,
            "tail_tokens_estimate": _estimate_tail_tokens(next_prompt_text, next_prompt_tokens, tail_chars),
            "next_prompt_tokens": next_prompt_tokens,
            "raw_full_prompt_tokens_if_reset": next_prompt_tokens,
            "conditioning_note": "text history aligns after assistant prefill / think-boundary normalization",
            **artifacts,
        }

    prefix_chars = _common_prefix_len(normalized_projection, next_prompt_text)
    ratio = prefix_chars / max(1, len(normalized_projection))
    if ratio < 0.50:
        classification = "reset_or_summary_candidate"
        conditioning_note = "low overlap after safe normalization; treat next prompt as a new context base"
    else:
        classification = "unresolved_normalized_mismatch"
        conditioning_note = "high-overlap mismatch remains after safe normalization; inspect parser/template differences"

    return {
        "classification": classification,
        "prefix_chars": prefix_chars,
        "projection_chars": len(normalized_projection),
        "next_prompt_chars": len(next_prompt_text),
        "prefix_ratio": ratio,
        "tail_chars": len(next_prompt_text),
        "tail_tokens_estimate": next_prompt_tokens,
        "next_prompt_tokens": next_prompt_tokens,
        "raw_full_prompt_tokens_if_reset": next_prompt_tokens,
        "conditioning_note": conditioning_note,
        **artifacts,
        "mismatch_window": _mismatch_window(normalized_projection, next_prompt_text, prefix_chars),
    }


def _analyze_record(record: dict[str, Any], source_path: Path) -> dict[str, Any]:
    turns = record.get("turns") or []
    transitions = []
    counts: Counter[str] = Counter()
    raw_reset_tokens = 0
    normalized_tail_tokens = 0
    chunks = 1 if turns else 0

    for i in range(max(0, len(turns) - 1)):
        item = _classify_transition(turns[i], turns[i + 1])
        item["turn_index"] = i
        item["next_turn_index"] = i + 1
        counts[item["classification"]] += 1
        raw_reset_tokens += int(item.get("raw_full_prompt_tokens_if_reset") or 0)
        normalized_tail_tokens += int(item.get("tail_tokens_estimate") or 0)
        if item["classification"] in {"reset_or_summary_candidate", "unresolved_normalized_mismatch"}:
            chunks += 1
        transitions.append(item)

    unresolved_examples = [
        item
        for item in transitions
        if item["classification"] in {"reset_or_summary_candidate", "unresolved_normalized_mismatch"}
    ][:3]

    return {
        "source_path": str(source_path),
        "trial_name": record.get("trial_name"),
        "task_name": record.get("task_name"),
        "sample_mode": record.get("sample_mode"),
        "turns": len(turns),
        "transitions": len(transitions),
        "counts": dict(counts),
        "estimated_raw_reset_context_tokens": raw_reset_tokens,
        "estimated_normalized_context_tokens": normalized_tail_tokens,
        "estimated_context_tokens_saved": max(0, raw_reset_tokens - normalized_tail_tokens),
        "normalized_chunk_count": chunks,
        "examples": unresolved_examples,
        "all_transitions": transitions,
    }


def _iter_records(raw_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    out = []
    for path in sorted(raw_dir.glob("*.json")):
        payload = _load_json(path)
        if isinstance(payload, list):
            for record in payload:
                if isinstance(record, dict):
                    out.append((path, record))
        elif isinstance(payload, dict):
            out.append((path, payload))
    return out


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _render_markdown(results: list[dict[str, Any]], raw_dir: Path) -> str:
    totals: Counter[str] = Counter()
    total_transitions = 0
    total_raw_tokens = 0
    total_norm_tokens = 0
    total_saved_tokens = 0
    total_chunks = 0
    total_turns = 0

    for item in results:
        totals.update(item["counts"])
        total_transitions += int(item["transitions"])
        total_raw_tokens += int(item["estimated_raw_reset_context_tokens"])
        total_norm_tokens += int(item["estimated_normalized_context_tokens"])
        total_saved_tokens += int(item["estimated_context_tokens_saved"])
        total_chunks += int(item["normalized_chunk_count"])
        total_turns += int(item["turns"])

    lines = [
        "# Normalized Trajectory Stitching Audit",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Raw export dir: `{raw_dir}`",
        f"- Records: {len(results)}",
        f"- Turns / transitions: {total_turns} / {total_transitions}",
        "",
        "## Summary",
        "",
        "| classification | count |",
        "| --- | ---: |",
    ]
    for key in (
        "strict_prefix",
        "qwen_template_normalized_prefix",
        "unresolved_normalized_mismatch",
        "reset_or_summary_candidate",
    ):
        lines.append(f"| `{key}` | {totals.get(key, 0)} |")

    lines.extend(
        [
            "",
            "## Token Estimate",
            "",
            "| metric | estimated tokens |",
            "| --- | ---: |",
            f"| raw full-prompt context inserted on non-strict turns | {total_raw_tokens} |",
            f"| normalized tail/reset context | {total_norm_tokens} |",
            f"| estimated context tokens saved | {total_saved_tokens} |",
            f"| normalized chunk count | {total_chunks} |",
            "",
            "Interpretation:",
            "",
            "- `qwen_template_normalized_prefix` means the next prompt is continuous after stripping only assistant generation prefill and leading think-boundary artifacts.",
            "- `unresolved_normalized_mismatch` means safe Qwen/template normalization is not enough, but overlap is still high. These are likely parser/template formatting differences and need manual inspection before automatic training conversion.",
            "- `reset_or_summary_candidate` means overlap is low after safe normalization. Treat the next prompt as a new context base rather than continuing one causal sequence.",
            "- The token estimate is approximate because this audit uses exported decoded text plus per-turn prompt token counts, not a live tokenizer.",
            "",
            "## Per Record",
            "",
            "| file | turns | strict | qwen-normalized | unresolved | reset-candidate | est raw ctx | est normalized ctx | est saved | chunks |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for item in results:
        counts = item["counts"]
        lines.append(
            "| {file} | {turns} | {strict} | {qwen} | {unresolved} | {reset} | {raw} | {norm} | {saved} | {chunks} |".format(
                file=Path(item["source_path"]).name,
                turns=item["turns"],
                strict=counts.get("strict_prefix", 0),
                qwen=counts.get("qwen_template_normalized_prefix", 0),
                unresolved=counts.get("unresolved_normalized_mismatch", 0),
                reset=counts.get("reset_or_summary_candidate", 0),
                raw=item["estimated_raw_reset_context_tokens"],
                norm=item["estimated_normalized_context_tokens"],
                saved=item["estimated_context_tokens_saved"],
                chunks=item["normalized_chunk_count"],
            )
        )

    lines.extend(["", "## Unresolved Examples", ""])
    for item in results:
        if not item["examples"]:
            continue
        lines.append(f"### {Path(item['source_path']).name}")
        lines.append("")
        for example in item["examples"]:
            lines.append(
                "- turn {turn}->{next_turn}: `{cls}`, prefix_ratio={ratio:.4f}, note={note}".format(
                    turn=example["turn_index"],
                    next_turn=example["next_turn_index"],
                    cls=example["classification"],
                    ratio=float(example.get("prefix_ratio") or 1.0),
                    note=example.get("conditioning_note", ""),
                )
            )
            window = example.get("mismatch_window") or {}
            expected = _shorten(str(window.get("expected") or ""))
            actual = _shorten(str(window.get("actual") or ""))
            lines.append("")
            lines.append("Expected normalized projection around mismatch:")
            lines.append("")
            lines.append("```text")
            lines.append(expected)
            lines.append("```")
            lines.append("")
            lines.append("Actual next prompt around mismatch:")
            lines.append("")
            lines.append("```text")
            lines.append(actual)
            lines.append("```")
            lines.append("")

    lines.extend(
        [
            "## Training Implication",
            "",
            "A normalized trajectory stitcher can recover most apparent raw-token prefix mismatches in these exports, especially the Qwen think-template boundary cases. It should still be treated as an approximate fast path unless the training pipeline can either:",
            "",
            "1. train on the same normalized assistant content that the next prompt renderer replays, or",
            "2. reset/split the causal sequence whenever generated raw tokens are removed from later rendered history.",
            "",
            "For summary or low-overlap resets, the correct behavior is to start a new segment from the summary-rendered prompt. Continuing the old raw trajectory would put tokens in attention that the rollout policy did not condition on.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit normalized text-level trajectory stitching on exported Harbor turn IO.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(
            "/path/to/workspace/code/slime/examples/harbor_terminal_rl/trajectory_turn_io_raw"
        ),
        help="Directory containing exported trajectory_turn_io_raw JSON files.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("examples/harbor_terminal_rl/docs/NORMALIZED_TRAJECTORY_STITCHING_AUDIT.md"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("examples/harbor_terminal_rl/docs/NORMALIZED_TRAJECTORY_STITCHING_AUDIT.json"),
    )
    args = parser.parse_args()

    records = _iter_records(args.raw_dir)
    results = [_analyze_record(record, path) for path, record in records if record.get("turns")]
    _write_json(args.output_json, {"raw_dir": str(args.raw_dir), "results": results})
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_render_markdown(results, args.raw_dir), encoding="utf-8")
    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
