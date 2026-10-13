#!/usr/bin/env python3
"""Summarize headless-terminal rollout/eval implementation patterns."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


FLAG_PATTERNS: dict[str, tuple[str, ...]] = {
    "uses_tmux": ("tmux",),
    "uses_pexpect": ("pexpect",),
    "uses_pty_openpty": ("pty.openpty",),
    "uses_pty_fork": ("pty.fork",),
    "uses_subprocess_popen": ("subprocess.Popen", "subprocess.popen"),
    "uses_os_fork": ("os.fork",),
    "uses_os_setsid": ("os.setsid", "preexec_fn=os.setsid"),
    "uses_tty_setraw": ("tty.setraw",),
    "uses_tiocsctty": ("TIOCSCTTY", "tiocsctty"),
    "uses_bash_i": ("bash\", \"-i", "bash', '-i", "['-i']", "--interactive"),
    "uses_bash_l": ("bash\", \"-l", "bash', '-l", "['-l']", "login shell"),
    "uses_bash_rcfile": ("--rcfile",),
    "imports_base_terminal": ("from base_terminal import BaseTerminal",),
    "runs_pytest": ("pytest",),
    "runs_official_tests": ("test_outputs.py", "/tests/test_outputs.py"),
    "tests_vim": ("vim", "vim.txt"),
    "tests_startup_file": ("SPECIAL_FILE", ".bashrc", "special_file.txt"),
    "tests_shell_state": ("SECRET_FILE", "secret.txt"),
    "tests_background_server": ("http.server", "localhost:8000", "background"),
    "tests_ctrl_c": ("\\x03", "Ctrl+C", "Ctrl-C", "^C"),
    "claims_task_complete": ('"task_complete": true',),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def flags_for_text(text: str) -> dict[str, bool]:
    lower = text.lower()
    flags: dict[str, bool] = {}
    for key, patterns in FLAG_PATTERNS.items():
        flags[key] = any(pattern.lower() in lower for pattern in patterns)

    flags["pty_without_pexpect_or_tmux"] = (
        (flags["uses_pty_openpty"] or flags["uses_pty_fork"])
        and not flags["uses_pexpect"]
        and not flags["uses_tmux"]
    )
    flags["manual_fork_openpty_raw_stack"] = (
        flags["uses_pty_openpty"]
        and flags["uses_os_fork"]
        and flags["uses_os_setsid"]
        and flags["uses_tty_setraw"]
    )
    flags["official_test_gap"] = flags["claims_task_complete"] and not flags["runs_official_tests"]
    flags["covers_all_key_behaviors"] = all(
        flags[key]
        for key in (
            "tests_vim",
            "tests_startup_file",
            "tests_shell_state",
            "tests_background_server",
            "tests_ctrl_c",
        )
    )
    return flags


def records_from_jsonl(label: str, path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, sample in enumerate(read_jsonl(path), start=1):
        text = str(sample.get("response", ""))
        records.append(
            {
                "label": label,
                "source": str(path),
                "sample_index": index,
                "dump": sample.get("dump"),
                "reward": sample.get("reward"),
                "status": sample.get("status"),
                "group_id": sample.get("group_id"),
                "response_length": sample.get("response_length"),
                **flags_for_text(text),
            }
        )
    return records


def records_from_text(label: str, path: Path) -> list[dict[str, Any]]:
    return [
        {
            "label": label,
            "source": str(path),
            "sample_index": 1,
            "dump": None,
            "reward": None,
            "status": None,
            "group_id": None,
            "response_length": len(path.read_text(errors="replace")),
            **flags_for_text(path.read_text(errors="replace")),
        }
    ]


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_label.setdefault(str(record["label"]), []).append(record)

    summaries: list[dict[str, Any]] = []
    for label, items in sorted(by_label.items()):
        summary: dict[str, Any] = {"label": label, "samples": len(items)}
        if items:
            bool_keys = [key for key, value in items[0].items() if isinstance(value, bool)]
            for key in bool_keys:
                summary[key] = sum(1 for item in items if item[key])
            lengths = [
                float(item["response_length"])
                for item in items
                if item.get("response_length") is not None
            ]
            if lengths:
                summary["response_length_mean"] = sum(lengths) / len(lengths)
                summary["response_length_min"] = min(lengths)
                summary["response_length_max"] = max(lengths)
        summaries.append(summary)
    return summaries


def write_outputs(records: list[dict[str, Any]], summaries: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if records:
        csv_path = output_dir / "sample_flags.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")

    keys = [
        "uses_tmux",
        "uses_pexpect",
        "uses_pty_openpty",
        "uses_pty_fork",
        "uses_subprocess_popen",
        "uses_os_fork",
        "uses_os_setsid",
        "uses_tty_setraw",
        "uses_tiocsctty",
        "uses_bash_i",
        "uses_bash_l",
        "uses_bash_rcfile",
        "runs_official_tests",
        "runs_pytest",
        "tests_vim",
        "tests_startup_file",
        "tests_shell_state",
        "tests_background_server",
        "tests_ctrl_c",
        "covers_all_key_behaviors",
        "manual_fork_openpty_raw_stack",
        "official_test_gap",
        "claims_task_complete",
    ]

    lines = ["# headless-terminal sample summary", ""]
    for summary in summaries:
        lines.append(f"## {summary['label']}")
        lines.append("")
        lines.append(f"- samples: {summary['samples']}")
        for key in keys:
            if key in summary:
                lines.append(f"- {key}: {summary[key]}")
        if "response_length_mean" in summary:
            lines.append(
                "- response_length mean/min/max: "
                f"{summary['response_length_mean']:.1f}/"
                f"{summary['response_length_min']:.0f}/"
                f"{summary['response_length_max']:.0f}"
            )
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl-set", action="append", nargs=2, metavar=("LABEL", "PATH"), default=[])
    parser.add_argument("--text-set", action="append", nargs=2, metavar=("LABEL", "PATH"), default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for label, path_text in args.jsonl_set:
        records.extend(records_from_jsonl(label, Path(path_text)))
    for label, path_text in args.text_set:
        records.extend(records_from_text(label, Path(path_text)))

    write_outputs(records, summarize(records), Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
