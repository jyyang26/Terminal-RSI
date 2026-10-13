#!/usr/bin/env python3
"""Summarize decoded cancel-async-tasks rollout samples."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


KEYSTROKE_RE = re.compile(r'"keystrokes"\s*:\s*"((?:\\.|[^"\\])*)"')
RUN_PY_RE = re.compile(
    r"cat\s+>\s+/app/run\.py\s+<<\s+'?EOF'?\n(.*?)\nEOF",
    re.DOTALL,
)


def _load_json_string(raw: str) -> str:
    try:
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        return raw


def _commands(response: str) -> list[str]:
    return [_load_json_string(match.group(1)) for match in KEYSTROKE_RE.finditer(response)]


def _last_analysis(response: str) -> str:
    matches = list(re.finditer(r'"analysis"\s*:\s*"((?:\\.|[^"\\])*)"', response))
    if not matches:
        return ""
    return _load_json_string(matches[-1].group(1))


def _contains(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def _count_task_complete(response: str) -> int:
    return len(re.findall(r'"task_complete"\s*:\s*true', response, flags=re.IGNORECASE))


def _last_run_py(commands: list[str]) -> str:
    last = ""
    for command in commands:
        for match in RUN_PY_RE.finditer(command):
            last = match.group(1)
    return last


def _code_flag(code: str, pattern: str) -> bool:
    return bool(re.search(pattern, code, flags=re.MULTILINE))


def _summarize_record(label: str, index: int, record: dict[str, Any]) -> dict[str, Any]:
    response = str(record.get("response") or "")
    commands = _commands(response)
    command_text = "\n".join(commands)
    combined = f"{response}\n{command_text}"
    run_py = _last_run_py(commands)
    suspicious_coroutine_cancel = bool(
        re.search(r"(coroutines|coro_list)\s*=.*\n(?:.|\n)*\.(done|cancel)\(", run_py)
    )
    return {
        "label": label,
        "sample_index": index,
        "dump": record.get("dump"),
        "reward": record.get("reward"),
        "status": record.get("status"),
        "response_length": record.get("response_length"),
        "loss_tokens": record.get("loss_tokens"),
        "commands": len(commands),
        "task_complete_true": _count_task_complete(response),
        "has_run_py": bool(run_py),
        "async_def": _code_flag(run_py, r"^\s*async\s+def\s+run_tasks\b"),
        "plain_def": _code_flag(run_py, r"^\s*def\s+run_tasks\b"),
        "uses_taskgroup": "TaskGroup" in run_py,
        "uses_create_task": "create_task" in run_py,
        "uses_gather": "gather(" in run_py,
        "gather_return_exceptions": "return_exceptions=True" in run_py,
        "catches_cancelled_error": "CancelledError" in run_py,
        "catches_keyboard_interrupt": "KeyboardInterrupt" in run_py,
        "calls_asyncio_run": "asyncio.run" in run_py,
        "suspicious_coroutine_cancel": suspicious_coroutine_cancel,
        "tests_import_only": _contains(command_text, "from run import run_tasks")
        and not _contains(command_text, "send_signal")
        and not _contains(command_text, "SIGINT"),
        "tests_concurrency": _contains(command_text, "max_concurrent=2")
        or _contains(command_text, "--max-concurrent"),
        "tests_sigint": _contains(command_text, "SIGINT")
        or _contains(command_text, "send_signal")
        or _contains(command_text, "KeyboardInterrupt"),
        "final_analysis": _last_analysis(response)[:600].replace("\n", " "),
        "run_py_one_line": re.sub(r"\s+", " ", run_py.strip())[:1000],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="LABEL=path/to/samples.jsonl")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for item in args.inputs:
        if "=" not in item:
            raise SystemExit(f"input must be LABEL=PATH: {item}")
        label, raw_path = item.split("=", 1)
        for index, record in enumerate(_read_jsonl(Path(raw_path)), start=1):
            rows.append(_summarize_record(label, index, record))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    output_md = Path(args.output_md)
    with output_md.open("w") as handle:
        handle.write("# cancel-async-tasks Sample Summary\n\n")
        for row in rows:
            handle.write(
                f"## {row['label']} sample {row['sample_index']} "
                f"(reward={row['reward']}, dump={row['dump']})\n\n"
            )
            for key in (
                "async_def",
                "plain_def",
                "uses_taskgroup",
                "uses_create_task",
                "uses_gather",
                "gather_return_exceptions",
                "catches_cancelled_error",
                "catches_keyboard_interrupt",
                "calls_asyncio_run",
                "suspicious_coroutine_cancel",
                "tests_import_only",
                "tests_concurrency",
                "tests_sigint",
                "task_complete_true",
            ):
                handle.write(f"- {key}: {row[key]}\n")
            handle.write(f"- final_analysis: {row['final_analysis']}\n")
            if row["run_py_one_line"]:
                handle.write(f"- run_py: `{row['run_py_one_line']}`\n")
            handle.write("\n")

    print(f"Wrote {output_csv}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
