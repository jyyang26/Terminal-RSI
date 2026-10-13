#!/usr/bin/env python3
"""Summarize decoded Harbor rollout sample text.

The extractor writes one JSON object per selected sample.  This helper keeps the
analysis reproducible by scanning those decoded responses for concrete behavior:
commands issued, test evidence, and task-specific keywords.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


KEYSTROKE_RE = re.compile(r'"keystrokes"\s*:\s*"((?:\\.|[^"\\])*)"')


def _load_json_string(raw: str) -> str:
    try:
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        return raw


def extract_commands(response: str) -> list[str]:
    commands: list[str] = []
    for match in KEYSTROKE_RE.finditer(response):
        commands.append(_load_json_string(match.group(1)))
    return commands


def _contains_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _count_regex(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE))


def _last_analysis(response: str) -> str:
    matches = list(re.finditer(r'"analysis"\s*:\s*"((?:\\.|[^"\\])*)"', response))
    if not matches:
        return ""
    return _load_json_string(matches[-1].group(1))


def summarize_record(record: dict[str, Any], label: str, sample_index: int) -> dict[str, Any]:
    response = str(record.get("response") or "")
    commands = extract_commands(response)
    command_text = "\n".join(commands)
    combined = f"{response}\n{command_text}"

    direct_ccomplexity_test = _contains_any(
        combined,
        [
            "cython_higher_order_writhe",
            "test_ccomplexity",
            "higher_order_writhe",
        ],
    )
    repo_tests_only_signal = (
        _contains_any(combined, ["18 passed", "test_knot.py", "test_spacecurve.py"])
        and not direct_ccomplexity_test
    )

    task_complete_count = _count_regex(response, r'"task_complete"\s*:\s*true')
    final_analysis = _last_analysis(response)
    edit_commands = [
        command.strip().replace("\n", "\\n")
        for command in commands
        if _contains_any(command, ["sed ", "perl ", "python", "grep", "find ", "cat ", "vim "])
    ]
    test_commands = [
        command.strip().replace("\n", "\\n")
        for command in commands
        if _contains_any(command, ["pytest", "python3 -c", "python -c", "build_ext", "pip install"])
    ]

    return {
        "label": label,
        "sample_index": sample_index,
        "task": record.get("task"),
        "dump": record.get("dump"),
        "reward": record.get("reward"),
        "status": record.get("status"),
        "response_length": record.get("response_length"),
        "loss_tokens": record.get("loss_tokens"),
        "commands": len(commands),
        "task_complete_true": task_complete_count,
        "mentions_ccomplexity_pyx": _contains_any(combined, ["ccomplexity.pyx"]),
        "mentions_np_int": _contains_any(combined, ["np.int", "n.int", "numpy.int"]),
        "mentions_intp": _contains_any(combined, ["intp"]),
        "mentions_int64": _contains_any(combined, ["int64"]),
        "mentions_build_ext": _contains_any(combined, ["build_ext"]),
        "mentions_pip_install": _contains_any(combined, ["pip install"]),
        "mentions_18_passed": _contains_any(combined, ["18 passed"]),
        "direct_ccomplexity_test": direct_ccomplexity_test,
        "repo_tests_only_signal": repo_tests_only_signal,
        "final_analysis": final_analysis[:800].replace("\n", " "),
        "edit_commands": " || ".join(edit_commands[:10]),
        "test_commands": " || ".join(test_commands[:10]),
    }


def iter_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="LABEL=path/to/samples.jsonl")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for item in args.inputs:
        if "=" not in item:
            raise SystemExit(f"input must be LABEL=PATH: {item}")
        label, raw_path = item.split("=", 1)
        for index, record in enumerate(iter_records(Path(raw_path)), start=1):
            rows.append(summarize_record(record, label, index))

    fieldnames = list(rows[0].keys()) if rows else []
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        with output_md.open("w") as handle:
            handle.write("# Extracted Sample Summary\n\n")
            for row in rows:
                handle.write(
                    f"## {row['label']} sample {row['sample_index']} "
                    f"(reward={row['reward']}, dump={row['dump']})\n\n"
                )
                for key in (
                    "mentions_ccomplexity_pyx",
                    "mentions_np_int",
                    "mentions_intp",
                    "mentions_int64",
                    "mentions_18_passed",
                    "direct_ccomplexity_test",
                    "repo_tests_only_signal",
                    "task_complete_true",
                ):
                    handle.write(f"- {key}: {row[key]}\n")
                handle.write(f"- final_analysis: {row['final_analysis']}\n")
                if row["edit_commands"]:
                    handle.write(f"- edit_commands: `{row['edit_commands']}`\n")
                if row["test_commands"]:
                    handle.write(f"- test_commands: `{row['test_commands']}`\n")
                handle.write("\n")

    print(f"Wrote {output_csv}")
    if args.output_md:
        print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
