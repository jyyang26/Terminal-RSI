#!/usr/bin/env python3
"""Summarize sqlite-db-truncate rollout sample failure modes."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


VALUE_PATTERNS = {
    "testword08_99_99": re.compile(r"testword08[^\n]{0,120}(?:99\.99|99\.990)", re.I),
    "testword09_0_5": re.compile(r"testword09[^\n]{0,120}(?:0\.5|0\.500)", re.I),
    "testword08_64": re.compile(r"testword08[^\n]{0,120}(?:value[=: ]+|0x40[^\n]{0,40}= ?)?64(?:\D|$)", re.I),
    "testword09_63": re.compile(r"testword09[^\n]{0,120}(?:value[=: ]+|0x3f[^\n]{0,40}= ?)?63(?:\D|$)", re.I),
    "testword08_8": re.compile(r"testword08[^\n]{0,120}value[=: ]+8(?:\D|$)", re.I),
    "testword09_9": re.compile(r"testword09[^\n]{0,120}value[=: ]+9(?:\D|$)", re.I),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def flag_text(text: str) -> dict[str, bool]:
    lower = text.lower()
    flags = {
        "uses_varint_or_serial_types": any(
            token in lower
            for token in (
                "serial_types",
                "serial type",
                "read_varint",
                "sqlite b-tree",
                "b-tree leaf page",
                "cell pointer",
            )
        ),
        "decodes_big_endian_double": any(
            token in lower
            for token in (
                "struct.unpack('>d'",
                'struct.unpack(">d"',
                "big-endian double",
                "64-bit float",
                "ieee",
            )
        ),
        "uses_single_byte_after_word": any(
            token in lower
            for token in (
                "byte immediately following",
                "byte after",
                "value_byte",
                "single-byte",
                "single byte",
            )
        ),
        "claims_task_complete": '"task_complete": true' in lower,
        "installs_sqlite3": "apt-get install" in lower and "sqlite3" in lower,
    }
    for key, pattern in VALUE_PATTERNS.items():
        flags[key] = bool(pattern.search(text))
    flags["has_correct_float_pair"] = flags["testword08_99_99"] and flags["testword09_0_5"]
    flags["has_wrong_64_63_pair"] = flags["testword08_64"] and flags["testword09_63"]
    flags["has_wrong_8_9_pair"] = flags["testword08_8"] and flags["testword09_9"]
    return flags


def summarize_set(label: str, path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    for idx, sample in enumerate(load_jsonl(path), start=1):
        text = str(sample.get("response", ""))
        flags = flag_text(text)
        record: dict[str, Any] = {
            "label": label,
            "sample_index": idx,
            "dump": sample.get("dump"),
            "reward": sample.get("reward"),
            "status": sample.get("status"),
            "group_id": sample.get("group_id"),
            "response_length": sample.get("response_length"),
            "loss_tokens": sample.get("loss_tokens"),
            **flags,
        }
        records.append(record)

    summary: dict[str, Any] = {
        "label": label,
        "path": str(path),
        "samples": len(records),
    }
    if records:
        bool_keys = [key for key, value in records[0].items() if isinstance(value, bool)]
        for key in bool_keys:
            summary[key] = sum(1 for record in records if record[key])
        numeric_lengths = [
            float(record["response_length"])
            for record in records
            if record.get("response_length") is not None
        ]
        if numeric_lengths:
            summary["response_length_mean"] = sum(numeric_lengths) / len(numeric_lengths)
            summary["response_length_min"] = min(numeric_lengths)
            summary["response_length_max"] = max(numeric_lengths)
    return records, summary


def write_outputs(records: list[dict[str, Any]], summaries: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "sample_flags.csv"
    if records:
        fieldnames = list(records[0].keys())
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")

    lines = ["# sqlite-db-truncate rollout sample summary", ""]
    for summary in summaries:
        lines.append(f"## {summary['label']}")
        lines.append("")
        lines.append(f"- samples: {summary['samples']}")
        for key in (
            "uses_varint_or_serial_types",
            "decodes_big_endian_double",
            "has_correct_float_pair",
            "uses_single_byte_after_word",
            "has_wrong_64_63_pair",
            "has_wrong_8_9_pair",
            "claims_task_complete",
            "installs_sqlite3",
        ):
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
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        nargs=2,
        metavar=("LABEL", "SAMPLES_JSONL"),
        required=True,
        help="Labeled samples.jsonl input. Can be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    all_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for label, path_text in args.sets:
        records, summary = summarize_set(label, Path(path_text))
        all_records.extend(records)
        summaries.append(summary)
    write_outputs(all_records, summaries, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
