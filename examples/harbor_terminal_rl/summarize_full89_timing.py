#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


TS_RE = re.compile(r"\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:[^\]]*)?\]")
PERF_RE = re.compile(r"\bperf (?P<rollout_id>\d+): (?P<payload>\{.*\})")
ROLLOUT_METRIC_RE = re.compile(r"\brollout (?P<rollout_id>\d+): (?P<payload>\{.*\})")
ASYNC_START_RE = re.compile(r"Harbor fully-async rollout (?P<rollout_id>\d+): target=(?P<target>.*)")
ASYNC_DONE_RE = re.compile(
    r"Harbor fully-async rollout (?P<rollout_id>\d+): done in (?P<seconds>[0-9.]+)s.*queue_left=(?P<queue_left>\d+)"
)
CLUSTER_READY_RE = re.compile(r"ray cluster has (?P<nodes>\d+)/(?P<expected>\d+) nodes after (?P<seconds>\d+)s")
JOB_ID_RE = re.compile(r"raysubmit_[A-Za-z0-9]+")


def parse_ts(line: str) -> datetime | None:
    match = TS_RE.search(line)
    if match is None:
        return None
    return datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S")


def parse_dict(payload: str) -> dict[str, Any] | None:
    try:
        value = ast.literal_eval(payload)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60:.2f}h"


def fmt_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def add_event(events: list[dict[str, Any]], kind: str, line_no: int, ts: datetime | None, text: str, **extra: Any) -> None:
    item: dict[str, Any] = {
        "kind": kind,
        "line": line_no,
        "timestamp": ts.isoformat(sep=" ") if ts else None,
        "text": text.strip(),
    }
    item.update(extra)
    events.append(item)


def row_for(rows: dict[int, dict[str, Any]], rollout_id: int) -> dict[str, Any]:
    return rows.setdefault(rollout_id, {"rollout_id": rollout_id})


def summarize(log_path: Path) -> dict[str, Any]:
    rows: dict[int, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    error_counts = {
        "traceback": 0,
        "oom": 0,
        "failed": 0,
        "timeout": 0,
        "not_enough_samples": 0,
    }
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    job_ids: list[str] = []

    for line_no, line in enumerate(log_path.read_text(errors="replace").splitlines(), start=1):
        ts = parse_ts(line)
        if ts is not None:
            first_ts = first_ts or ts
            last_ts = ts

        for job_id in JOB_ID_RE.findall(line):
            if job_id not in job_ids:
                job_ids.append(job_id)

        lowered = line.lower()
        if "traceback" in lowered:
            error_counts["traceback"] += 1
        if "outofmemory" in lowered or "cuda out of memory" in lowered or re.search(r"\boom\b", lowered):
            error_counts["oom"] += 1
        if re.search(r"\b(job .*failed|trial .* failed|failed:|failed to)\b", lowered):
            error_counts["failed"] += 1
        if "timed out" in lowered or "timeout after" in lowered:
            error_counts["timeout"] += 1
        if "not enough samples" in lowered:
            error_counts["not_enough_samples"] += 1

        cluster_ready = CLUSTER_READY_RE.search(line)
        if cluster_ready:
            add_event(
                events,
                "ray_cluster_ready",
                line_no,
                ts,
                line,
                nodes=int(cluster_ready.group("nodes")),
                expected=int(cluster_ready.group("expected")),
                seconds=int(cluster_ready.group("seconds")),
            )

        event_patterns = (
            ("run_config", "run config:"),
            ("ray_head_start", "starting Ray head:"),
            ("ray_job_agent_ready", "ray job agent ready after"),
            ("ray_job_submit", "submitting Ray job"),
            ("sglang_ready", "The server is fired up"),
            ("fully_async_worker_start", "starting Harbor fully-async rollout worker"),
            ("checkpoint_start", "saving checkpoint"),
            ("checkpoint_done", "successfully saved checkpoint"),
            ("worker_stop_marker", "wrote worker stop marker"),
            ("job_succeeded", "SUCCEEDED"),
            ("job_failed", "FAILED"),
        )
        for kind, needle in event_patterns:
            if needle in line:
                add_event(events, kind, line_no, ts, line)
                break

        async_start = ASYNC_START_RE.search(line)
        if async_start and "done in" not in line:
            rollout_id = int(async_start.group("rollout_id"))
            row_for(rows, rollout_id)["async_target"] = async_start.group("target").strip()
            add_event(events, "fully_async_rollout_start", line_no, ts, line, rollout_id=rollout_id)

        async_done = ASYNC_DONE_RE.search(line)
        if async_done:
            rollout_id = int(async_done.group("rollout_id"))
            row = row_for(rows, rollout_id)
            row["fully_async_done_seconds"] = float(async_done.group("seconds"))
            row["fully_async_queue_left"] = int(async_done.group("queue_left"))
            add_event(
                events,
                "fully_async_rollout_done",
                line_no,
                ts,
                line,
                rollout_id=rollout_id,
                seconds=row["fully_async_done_seconds"],
                queue_left=row["fully_async_queue_left"],
            )

        for regex, source in ((PERF_RE, "perf"), (ROLLOUT_METRIC_RE, "rollout_metrics")):
            match = regex.search(line)
            if match is None:
                continue
            payload = parse_dict(match.group("payload"))
            if payload is None:
                continue
            rollout_id = int(match.group("rollout_id"))
            row = row_for(rows, rollout_id)
            row[source] = payload
            row.setdefault("log_lines", []).append(line_no)
            for key, value in payload.items():
                if key.startswith("perf/") or key.startswith("harbor/") or key in {"reward_mean"}:
                    row[key] = value
            if source == "perf":
                if "perf/rollout_time" in payload:
                    add_event(events, "rollout_perf", line_no, ts, line, rollout_id=rollout_id)
                if "perf/train_time" in payload or "perf/actor_train_time" in payload:
                    add_event(events, "train_perf", line_no, ts, line, rollout_id=rollout_id)

    return {
        "log_path": str(log_path),
        "first_timestamp": first_ts.isoformat(sep=" ") if first_ts else None,
        "last_timestamp": last_ts.isoformat(sep=" ") if last_ts else None,
        "elapsed_seconds": (last_ts - first_ts).total_seconds() if first_ts and last_ts else None,
        "job_ids": job_ids,
        "error_counts": error_counts,
        "events": events,
        "rollouts": [rows[key] for key in sorted(rows)],
    }


def print_summary(data: dict[str, Any], max_events: int) -> None:
    print(f"log_path={data['log_path']}")
    print(f"first_timestamp={data['first_timestamp']}")
    print(f"last_timestamp={data['last_timestamp']}")
    print(f"elapsed={fmt_duration(data['elapsed_seconds'])}")
    print(f"job_ids={','.join(data['job_ids']) if data['job_ids'] else '-'}")
    print("errors=" + " ".join(f"{key}={value}" for key, value in data["error_counts"].items()))

    print("\nTimeline:")
    first_ts = datetime.fromisoformat(data["first_timestamp"]) if data["first_timestamp"] else None
    for event in data["events"][-max_events:]:
        ts_text = event["timestamp"] or "-"
        offset = "-"
        if first_ts and event["timestamp"]:
            offset = fmt_duration((datetime.fromisoformat(event["timestamp"]) - first_ts).total_seconds())
        rollout = f" r{event['rollout_id']}" if "rollout_id" in event else ""
        print(f"{ts_text} +{offset} {event['kind']}{rollout}: line={event['line']}")

    print("\nPer-rollout timing:")
    header = (
        "rid async_done rollout_perf train_wait log_probs actor_train train_time step_time "
        "groups samples failed missing reward queue ckpt"
    )
    print(header)
    for row in data["rollouts"]:
        ckpt = "yes" if any(
            event.get("kind") == "checkpoint_done" and event.get("timestamp") for event in data["events"]
        ) else "-"
        print(
            " ".join(
                [
                    str(row["rollout_id"]),
                    fmt_duration(row.get("fully_async_done_seconds")),
                    fmt_duration(row.get("perf/rollout_time")),
                    fmt_duration(row.get("perf/train_wait_time")),
                    fmt_duration(row.get("perf/log_probs_time")),
                    fmt_duration(row.get("perf/actor_train_time")),
                    fmt_duration(row.get("perf/train_time")),
                    fmt_duration(row.get("perf/step_time")),
                    fmt_value(row.get("harbor/num_groups")),
                    fmt_value(row.get("harbor/num_samples")),
                    fmt_value(row.get("harbor/num_failed_trials")),
                    fmt_value(row.get("harbor/num_missing_rollout_details")),
                    fmt_value(row.get("harbor/reward_mean")),
                    fmt_value(row.get("fully_async_queue_left")),
                    ckpt,
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Harbor full89 launcher timing from a slime log.")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--max-events", type=int, default=80)
    args = parser.parse_args()

    data = summarize(args.log_path)
    print_summary(data, max_events=args.max_events)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
