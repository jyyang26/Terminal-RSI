#!/usr/bin/env python3
"""Extract representative Harbor rollout samples from .pt dumps."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


NUMERIC_DUMP_RE = re.compile(r"^(\d+)\.pt$")
EVAL_DUMP_RE = re.compile(r"^eval_(\d+)\.pt$")


def _import_torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyTorch is required to read rollout dump .pt files") from exc
    return torch


def _sample_get(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _reward_value(reward: Any) -> float | None:
    if reward is None:
        return None
    if isinstance(reward, dict):
        reward = reward.get("reward", reward.get("score"))
    if hasattr(reward, "item"):
        try:
            reward = reward.item()
        except Exception:
            pass
    try:
        return float(reward)
    except (TypeError, ValueError):
        return None


def _status_name(status: Any) -> str:
    if status is None:
        return "missing"
    name = getattr(status, "name", None)
    if name:
        return str(name).lower()
    text = str(status)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.lower()


def _task_key(sample: Any) -> str:
    metadata = _sample_get(sample, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("task_name", "task", "task_id", "prompt_id", "name", "slug"):
            value = metadata.get(key)
            if value:
                return str(value)
    prompt = _sample_get(sample, "prompt", None)
    if prompt is not None:
        return str(prompt)
    group_id = _sample_get(sample, "group_id", None)
    if group_id is not None:
        return str(group_id)
    return "<unknown>"


def _response_length(sample: Any) -> int | None:
    value = _sample_get(sample, "response_length", None)
    if value is None:
        response = _sample_get(sample, "response", None)
        if isinstance(response, str):
            return len(response)
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _loss_tokens(sample: Any) -> float | None:
    loss_mask = _sample_get(sample, "loss_mask", None)
    if loss_mask is None:
        return None
    if hasattr(loss_mask, "sum"):
        try:
            return float(loss_mask.sum().item())
        except Exception:
            pass
    try:
        return float(sum(float(x) for x in loss_mask))
    except Exception:
        return None


def _safe_torch_load(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _dump_sort_key(path: Path) -> tuple[int, int, str]:
    match = NUMERIC_DUMP_RE.match(path.name)
    if match:
        return (0, int(match.group(1)), path.name)
    match = EVAL_DUMP_RE.match(path.name)
    if match:
        return (1, int(match.group(1)), path.name)
    return (2, 0, path.name)


def _iter_dump_files(args: argparse.Namespace) -> list[Path]:
    if args.files:
        files: list[Path] = []
        for pattern in args.files:
            matches = [Path(item) for item in glob.glob(pattern)]
            if not matches and Path(pattern).exists():
                matches = [Path(pattern)]
            files.extend(matches)
    else:
        dump_dir = Path(args.dump_dir)
        pattern = "*.pt" if args.include_eval else "[0-9]*.pt"
        files = list(dump_dir.glob(pattern))
    deduped: dict[Path, None] = {}
    for path in files:
        if path.is_file():
            if not args.include_eval and EVAL_DUMP_RE.match(path.name):
                continue
            deduped[path] = None
    return sorted(deduped, key=_dump_sort_key)


def _parse_tasks(values: list[str]) -> set[str]:
    tasks: set[str] = set()
    for value in values:
        for part in re.split(r"[\s,]+", value.strip()):
            if part:
                tasks.add(part)
    return tasks


def _reward_matches(value: float | None, selector: str) -> bool:
    if selector == "any":
        return True
    if selector == "positive":
        return value is not None and value > 0
    if selector == "zero":
        return value == 0
    if selector == "missing":
        return value is None
    raise ValueError(selector)


def _clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n\n...<CLIPPED>...\n\n" + text[-tail:]


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _token_list(tokens: Any) -> list[int]:
    if tokens is None:
        return []
    if hasattr(tokens, "tolist"):
        try:
            return list(tokens.tolist())
        except Exception:
            pass
    return list(tokens)


def _decode_response(tokenizer: Any, sample: Any) -> str:
    tokens = _token_list(_sample_get(sample, "tokens", []))
    response_length = _response_length(sample) or 0
    response_tokens = tokens[-response_length:] if response_length else []
    return tokenizer.decode(response_tokens, skip_special_tokens=False)


def _record(
    sample: Any,
    dump_label: str,
    max_response_chars: int,
    tokenizer: Any | None,
) -> dict[str, Any]:
    reward = _reward_value(_sample_get(sample, "reward", None))
    response = _sample_get(sample, "response", "")
    if tokenizer is not None:
        response = _decode_response(tokenizer, sample)
    elif not isinstance(response, str):
        response = repr(response)
    metadata = _sample_get(sample, "metadata", None)
    train_metadata = _sample_get(sample, "train_metadata", None)
    return {
        "task": _task_key(sample),
        "dump": dump_label,
        "reward": reward,
        "status": _status_name(_sample_get(sample, "status", None)),
        "group_id": _sample_get(sample, "group_id", None),
        "index": _sample_get(sample, "index", None),
        "response_length": _response_length(sample),
        "loss_tokens": _loss_tokens(sample),
        "metadata": metadata if isinstance(metadata, dict) else None,
        "train_metadata": train_metadata if isinstance(train_metadata, dict) else None,
        "response": _clip_text(response, max_response_chars),
    }


def extract(args: argparse.Namespace) -> int:
    tasks = _parse_tasks(args.task)
    if not tasks:
        raise SystemExit("At least one --task is required")

    files = _iter_dump_files(args)
    if not files:
        raise SystemExit("No rollout dump files matched the requested inputs")

    torch = _import_torch()
    tokenizer = None
    if args.hf_checkpoint:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    selected: dict[str, list[dict[str, Any]]] = {task: [] for task in tasks}
    seen_by_task: dict[str, int] = {task: 0 for task in tasks}

    for index, path in enumerate(files, start=1):
        if args.progress:
            print(f"[{index}/{len(files)}] loading {path}", flush=True)
        data = _safe_torch_load(torch, path)
        samples = data.get("samples", data) if isinstance(data, dict) else data
        if not isinstance(samples, Iterable):
            continue
        dump_label = path.stem
        for sample in samples:
            task = _task_key(sample)
            if task not in tasks:
                continue
            seen_by_task[task] += 1
            if len(selected[task]) >= args.limit_per_task:
                continue
            reward = _reward_value(_sample_get(sample, "reward", None))
            if not _reward_matches(reward, args.reward):
                continue
            selected[task].append(_record(sample, dump_label, args.max_response_chars, tokenizer))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "samples.jsonl"
    with jsonl_path.open("w") as handle:
        for task in sorted(selected):
            for item in selected[task]:
                handle.write(json.dumps(item, sort_keys=True) + "\n")

    for task in sorted(selected):
        path = output_dir / f"{_slugify(task)}.txt"
        with path.open("w") as handle:
            handle.write(f"task: {task}\n")
            handle.write(f"seen_samples: {seen_by_task[task]}\n")
            handle.write(f"selected_samples: {len(selected[task])}\n")
            handle.write(f"reward_selector: {args.reward}\n")
            for idx, item in enumerate(selected[task], start=1):
                handle.write("\n" + "=" * 80 + "\n")
                handle.write(f"sample: {idx}\n")
                for key in (
                    "dump",
                    "reward",
                    "status",
                    "group_id",
                    "index",
                    "response_length",
                    "loss_tokens",
                ):
                    handle.write(f"{key}: {item.get(key)}\n")
                handle.write("\nresponse:\n")
                handle.write(item["response"])
                handle.write("\n")

    print("Wrote", jsonl_path)
    for task in sorted(selected):
        print(task, "seen", seen_by_task[task], "selected", len(selected[task]))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", default="rollout_dumps")
    parser.add_argument("--files", nargs="*")
    parser.add_argument("--include-eval", action="store_true")
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument(
        "--reward",
        choices=("positive", "zero", "missing", "any"),
        default="positive",
    )
    parser.add_argument("--limit-per-task", type=int, default=2)
    parser.add_argument("--max-response-chars", type=int, default=6000)
    parser.add_argument(
        "--hf-checkpoint",
        help="Optional HF checkpoint/tokenizer path used to decode sample tokens.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> int:
    return extract(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
