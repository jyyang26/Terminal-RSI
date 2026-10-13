#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TS_RE = re.compile(r"\[(?P<ts>20\d\d-\d\d-\d\d \d\d:\d\d:\d\d)\]")
ROLLOUT_RE = re.compile(
    r"(?:\brollout(?:_id)?[ =:]|\bperf\s+)(?P<rollout>\d+)\b",
    re.IGNORECASE,
)
WANDB_RE = re.compile(r"\bwandb:\s+(?P<key>[A-Za-z0-9_./-]+)\s+(?P<value>[-+0-9.eE]+)\s*$")

DEFAULT_KEYS = (
    "harbor/reward_mean",
    "harbor/num_samples",
    "harbor/num_trajectory_samples",
    "harbor/num_group_attempts",
    "harbor/num_partial_tail_trials_cancelled",
    "harbor/num_discarded_completed_group_tasks",
    "harbor/fully_async_queue_left",
    "harbor/avg_turns_per_trajectory",
    "harbor/avg_output_tokens_per_trajectory",
    "harbor/trial_timeout_ratio",
    "train/loss",
    "train/pg_loss",
    "train/entropy_loss",
    "train/pg_clipfrac",
    "train/ppo_kl",
    "train/grad_norm",
    "train/train_rollout_logprob_abs_diff",
    "train/global_batch_size",
    "eval/harbor",
)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def maybe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def extract_timestamp(line: str) -> str:
    match = TS_RE.search(line)
    return match.group("ts") if match else ""


def extract_rollout(line: str) -> str:
    match = ROLLOUT_RE.search(line)
    return match.group("rollout") if match else ""


def extract_literal_dict(line: str) -> dict[str, Any] | None:
    start = line.find("{")
    end = line.rfind("}")
    if start < 0 or end <= start:
        return None
    snippet = line[start : end + 1]
    try:
        value = ast.literal_eval(snippet)
    except Exception:
        return None
    if isinstance(value, dict):
        return value
    return None


def is_noise_line(line: str) -> bool:
    if "tmux_session.py" in line or "Sending keys:" in line:
        return True
    if "HarborTrialWorker" in line and "wandb:" not in line:
        return True
    return False


def iter_metric_rows(path: Path, keys: tuple[str, ...], print_matches: int) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    matches: list[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            if not any(key in raw_line for key in keys):
                continue
            line = strip_ansi(raw_line.rstrip("\n"))
            if is_noise_line(line):
                continue
            if print_matches and len(matches) < print_matches:
                matches.append(f"{path}:{line_no}: {line[:4000]}")

            ts = extract_timestamp(line)
            rollout = extract_rollout(line)

            wandb_match = WANDB_RE.search(line)
            if wandb_match:
                rows.append(
                    {
                        "source": str(path),
                        "line": line_no,
                        "timestamp": ts,
                        "rollout": rollout,
                        "kind": "wandb",
                        "key": wandb_match.group("key"),
                        "value": float(wandb_match.group("value")),
                    }
                )
                continue

            data = extract_literal_dict(line)
            if data is None:
                continue
            for key, value in data.items():
                numeric = maybe_float(value)
                if numeric is None:
                    continue
                if key not in keys and not any(key.startswith(prefix.rstrip("*")) for prefix in keys if prefix.endswith("*")):
                    continue
                rows.append(
                    {
                        "source": str(path),
                        "line": line_no,
                        "timestamp": ts,
                        "rollout": rollout,
                        "kind": "dict",
                        "key": key,
                        "value": numeric,
                    }
                )

    return rows, matches


def pivot_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["source"]),
            str(row["line"]),
            str(row["timestamp"]),
            str(row["rollout"]),
        )
        out = grouped.setdefault(
            key,
            {
                "source": row["source"],
                "line": row["line"],
                "timestamp": row["timestamp"],
                "rollout": row["rollout"],
            },
        )
        out[str(row["key"])] = row["value"]
    return list(grouped.values())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["source", "line", "timestamp", "rollout"]
    extra: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames and key not in extra:
                extra.append(key)
    fieldnames.extend(sorted(extra))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--print-matches", type=int, default=0)
    parser.add_argument("--key", action="append", default=[])
    args = parser.parse_args()

    keys = tuple(args.key or DEFAULT_KEYS)
    all_rows: list[dict[str, Any]] = []
    all_matches: list[str] = []
    for path in args.logs:
        rows, matches = iter_metric_rows(path, keys, args.print_matches)
        all_rows.extend(rows)
        all_matches.extend(matches)

    pivoted = pivot_rows(all_rows)
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_csv, pivoted)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(pivoted, indent=2, sort_keys=True), encoding="utf-8")

    print(f"metric_rows={len(all_rows)}")
    print(f"pivot_rows={len(pivoted)}")
    for line in all_matches:
        print(line)


if __name__ == "__main__":
    main()
