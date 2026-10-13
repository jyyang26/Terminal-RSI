#!/usr/bin/env python3
"""Summarize decoded largest-eigenval rollout samples."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


KEYSTROKE_RE = re.compile(r'"keystrokes"\s*:\s*"((?:\\.|[^"\\])*)"')
EIGEN_PY_RE = re.compile(
    r"cat\s+>\s+/app/eigen\.py\s+<<\s+'?EOF'?\n(.*?)\nEOF",
    re.DOTALL,
)
TIMING_RE = re.compile(r"Median time for ([0-9]+)x[0-9]+(?: \(ref\))?: ([0-9.]+) seconds")


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


def _last_eigen_py(commands: list[str]) -> str:
    last = ""
    for command in commands:
        for match in EIGEN_PY_RE.finditer(command):
            last = match.group(1)
    return last


def _contains(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def _count_task_complete(response: str) -> int:
    return len(re.findall(r'"task_complete"\s*:\s*true', response, flags=re.IGNORECASE))


def _summarize(label: str, index: int, record: dict[str, Any]) -> dict[str, Any]:
    response = str(record.get("response") or "")
    commands = _commands(response)
    command_text = "\n".join(commands)
    combined = f"{response}\n{command_text}"
    code = _last_eigen_py(commands)
    timing_mentions = len(TIMING_RE.findall(combined))
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
        "uses_np_linalg_eig": "np.linalg.eig" in code,
        "uses_np_linalg_eigvals": "np.linalg.eigvals" in code,
        "uses_scipy_linalg_eig": "scipy.linalg" in code or "scipy_eig" in code,
        "uses_sparse_eigs": "sparse.linalg" in code or " eigs(" in code,
        "uses_power_iteration": _contains(code, "power iteration") or "for _ in range" in code and "A @ v" in code,
        "check_finite_false": "check_finite=False" in code,
        "left_false": "left=False" in code,
        "right_true": "right=True" in code,
        "overwrite_a_true": "overwrite_a=True" in code,
        "installs_scipy": _contains(command_text, "pip install scipy"),
        "runs_public_eval": _contains(command_text, "eval.py"),
        "runs_pytest": _contains(command_text, "pytest"),
        "mentions_process_pool": _contains(combined, "ProcessPoolExecutor"),
        "mentions_odd_sizes": bool(re.search(r"\b[3579]x[3579]\b|\bsize\s*[=:]\s*[3579]\b", combined)),
        "timing_mentions": timing_mentions,
        "final_analysis": _last_analysis(response)[:700].replace("\n", " "),
        "eigen_py_one_line": re.sub(r"\s+", " ", code.strip())[:1200],
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
            rows.append(_summarize(label, index, record))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    output_md = Path(args.output_md)
    with output_md.open("w") as handle:
        handle.write("# largest-eigenval Sample Summary\n\n")
        for row in rows:
            handle.write(
                f"## {row['label']} sample {row['sample_index']} "
                f"(reward={row['reward']}, dump={row['dump']})\n\n"
            )
            for key in (
                "uses_np_linalg_eig",
                "uses_scipy_linalg_eig",
                "uses_sparse_eigs",
                "uses_power_iteration",
                "check_finite_false",
                "left_false",
                "overwrite_a_true",
                "installs_scipy",
                "runs_public_eval",
                "runs_pytest",
                "mentions_process_pool",
                "mentions_odd_sizes",
                "task_complete_true",
            ):
                handle.write(f"- {key}: {row[key]}\n")
            handle.write(f"- final_analysis: {row['final_analysis']}\n")
            if row["eigen_py_one_line"]:
                handle.write(f"- eigen_py: `{row['eigen_py_one_line']}`\n")
            handle.write("\n")

    print(f"Wrote {output_csv}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
