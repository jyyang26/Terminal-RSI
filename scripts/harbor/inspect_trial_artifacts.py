#!/usr/bin/env python3
"""Print compact Harbor trial artifacts for debugging train/eval mismatches."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_PATTERNS = [
    r"A1\s*=\s*np\.random\.randn",
    r"zA1\s*=\s*np\.random\.randn",
    r"Original A1 shape",
    r"True A1 shape",
    r"Stolen A1 shape",
    r"Failed to match rows",
    r"Rows with",
    r"All correlations",
    r"shape \(",
    r"passed",
    r"failed",
    r"reward",
]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


def _read_json(path: Path) -> dict[str, Any]:
    text = _read_text(path)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _print_matching_lines(path: Path, patterns: list[re.Pattern[str]], max_lines: int) -> None:
    text = _read_text(path)
    if not text:
        print(f"{path}: <missing>")
        return
    print(f"{path}:")
    count = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if any(pattern.search(line) for pattern in patterns):
            print(f"  {line_number}: {line[:500]}")
            count += 1
            if count >= max_lines:
                print(f"  ... truncated after {max_lines} matching lines")
                return
    if count == 0:
        print("  <no matching lines>")


def inspect_trial(trial_dir: Path, patterns: list[re.Pattern[str]], max_lines: int) -> None:
    print("=" * 100)
    print(trial_dir)

    result = _read_json(trial_dir / "result.json")
    if result:
        print("task_name:", result.get("task_name"))
        print("trial_name:", result.get("trial_name"))
        print("task_checksum:", result.get("task_checksum"))
        task_id = result.get("task_id")
        if isinstance(task_id, dict):
            print("task_path:", task_id.get("path"))
        verifier_result = result.get("verifier_result")
        print("verifier_result:", verifier_result)
        agent_result = result.get("agent_result")
        if isinstance(agent_result, dict):
            metadata = agent_result.get("metadata")
            if isinstance(metadata, dict):
                print("n_episodes:", metadata.get("n_episodes"))

    reward = _read_text(trial_dir / "verifier" / "reward.txt").strip()
    if reward:
        print("reward.txt:", reward)

    _print_matching_lines(trial_dir / "verifier" / "test-stdout.txt", patterns, max_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_dirs", nargs="+")
    parser.add_argument("--pattern", action="append", default=[])
    parser.add_argument("--max-lines", type=int, default=80)
    args = parser.parse_args()

    raw_patterns = args.pattern or DEFAULT_PATTERNS
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in raw_patterns]
    for trial_dir in args.trial_dirs:
        inspect_trial(Path(trial_dir), patterns, args.max_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
