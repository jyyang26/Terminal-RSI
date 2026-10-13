#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def duration_seconds(timing: Any) -> float | None:
    if not isinstance(timing, dict):
        return None
    start = parse_ts(timing.get("started_at"))
    finish = parse_ts(timing.get("finished_at"))
    if start is None or finish is None:
        return None
    return (finish - start).total_seconds()


def summarize(name: str, values: list[float]) -> None:
    values = sorted(v for v in values if isinstance(v, (int, float)) and not math.isnan(v))
    if not values:
        return

    def q(frac: float) -> float:
        idx = min(len(values) - 1, int(round(frac * (len(values) - 1))))
        return values[idx]

    print(
        f"{name}: n={len(values)} min={values[0]:.1f} p50={q(0.5):.1f} "
        f"mean={statistics.mean(values):.1f} p90={q(0.9):.1f} "
        f"p95={q(0.95):.1f} p99={q(0.99):.1f} max={values[-1]:.1f}"
    )


def _trial_sort_key(trial_name: str | None) -> tuple[int | None, int, int, int]:
    if not trial_name:
        return (None, 0, 0, 0)
    match = re.search(r"-r(\d+)-g(\d+)-s(\d+)-i(\d+)", trial_name)
    if not match:
        return (None, 0, 0, 0)
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def simulate_wall(durations: list[float], concurrency: int) -> float:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    slots = [0.0] * concurrency
    for duration in durations:
        idx = min(range(concurrency), key=slots.__getitem__)
        slots[idx] += duration
    return max(slots) if slots else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trials_dir", type=Path)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--verifier-caps", type=float, nargs="*", default=[300, 600])
    parser.add_argument("--agent-caps", type=float, nargs="*", default=[600, 900])
    parser.add_argument("--simulate-concurrency", type=int, nargs="*", default=[16, 32, 64, 96])
    args = parser.parse_args()

    files = sorted(args.trials_dir.glob("slime-r*/result.json"))
    rows: list[dict[str, Any]] = []
    for path in files:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue

        started = parse_ts(data.get("started_at"))
        finished = parse_ts(data.get("finished_at"))
        total = (finished - started).total_seconds() if started and finished else None
        rewards = ((data.get("verifier_result") or {}).get("rewards") or {})
        rows.append(
            {
                "path": str(path),
                "trial": data.get("trial_name"),
                "task": data.get("task_name"),
                "total": total,
                "env": duration_seconds(data.get("environment_setup")),
                "agent_setup": duration_seconds(data.get("agent_setup")),
                "agent": duration_seconds(data.get("agent_execution")),
                "verifier": duration_seconds(data.get("verifier")),
                "exception": bool(data.get("exception_info")),
                "reward": rewards.get("reward"),
                "sort_key": _trial_sort_key(data.get("trial_name")),
            }
        )

    print(f"result_files={len(files)} parsed={len(rows)}")
    for key in ("total", "env", "agent_setup", "agent", "verifier"):
        summarize(key, [row[key] for row in rows if row.get(key) is not None])

    reward_values = [float(row["reward"]) for row in rows if row.get("reward") is not None]
    if reward_values:
        print(f"reward_mean={statistics.mean(reward_values):.4f}")
    print(f"exceptions={sum(1 for row in rows if row['exception'])}")

    for key, caps in (("verifier", args.verifier_caps), ("agent", args.agent_caps)):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if values:
            print(f"\nCap simulation for {key}:")
            original = sum(values)
            for cap in caps:
                capped = sum(min(v, cap) for v in values)
                saved = original - capped
                print(
                    f"cap={cap:.0f}s saved_total={saved:.1f}s "
                    f"saved_mean_per_trial={saved / len(values):.1f}s "
                    f"affected={sum(1 for v in values if v > cap)}/{len(values)}"
                )

    per_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("total") is not None:
            per_task[str(row.get("task"))].append(float(row["total"]))
    task_means = sorted(
        ((statistics.mean(vals), len(vals), task) for task, vals in per_task.items()),
        reverse=True,
    )
    print("\nSlowest tasks by mean total:")
    for mean, count, task in task_means[: args.top]:
        print(f"{mean:.1f}s n={count} task={task}")

    per_rollout: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rollout_id = row["sort_key"][0]
        if rollout_id is not None and row.get("total") is not None:
            per_rollout[rollout_id].append(row)

    if per_rollout:
        print("\nSimple list-scheduling wall-time simulation by rollout:")
        for rollout_id in sorted(per_rollout):
            rollout_rows = sorted(per_rollout[rollout_id], key=lambda item: item["sort_key"])
            durations = [float(row["total"]) for row in rollout_rows]
            pieces = [f"r{rollout_id} n={len(durations)}"]
            for concurrency in args.simulate_concurrency:
                pieces.append(f"c{concurrency}={simulate_wall(durations, concurrency):.1f}s")
            print(" ".join(pieces))

    for key in ("total", "agent", "verifier", "env"):
        print(f"\nTop {key}:")
        for row in sorted(rows, key=lambda item: item.get(key) or -1, reverse=True)[: args.top]:
            print(
                f"{key}={row.get(key)} total={row.get('total')} env={row.get('env')} "
                f"setup={row.get('agent_setup')} agent={row.get('agent')} "
                f"verifier={row.get('verifier')} reward={row.get('reward')} "
                f"exc={row.get('exception')} task={row.get('task')} trial={row.get('trial')}"
            )


if __name__ == "__main__":
    main()
