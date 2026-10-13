"""Small, CPU-serializable continuation summaries across DP partitions."""

import math


def solve_episode_boundaries(records):
    """Solve reverse chunk continuations without gathering full token values.

    A_first = base + value_coefficient * V_next + advantage_coefficient * A_next.
    Records are keyed by (episode, chunk_index), independent of DP packing.
    """
    episodes = {}
    for row in records:
        episodes.setdefault(row["episode"], []).append(row)
    result = {}
    for episode, rows in episodes.items():
        rows.sort(key=lambda row: row["chunk_index"])
        count = rows[0]["chunk_count"]
        if len(rows) != count or [r["chunk_index"] for r in rows] != list(range(count)):
            raise ValueError(f"TB-GAE incomplete/duplicate chunks for episode {episode}")
        if any(r["chunk_count"] != count for r in rows):
            raise ValueError("TB-GAE inconsistent chunk count")
        if any(r["terminal"] for r in rows[:-1]):
            raise ValueError("TB-GAE real done before final chunk")
        if any(r["active"] != rows[0]["active"] for r in rows):
            raise ValueError("TB-GAE cannot silently remove a subset of episode chunks")
        if not rows[0]["active"]:
            for row in rows:
                result[(episode, row["chunk_index"])] = (0.0, 0.0)
            continue
        if any(r["outcome"] != rows[-1]["outcome"] for r in rows):
            raise ValueError("TB-GAE chunks disagree on episode outcome reward")
        next_v = 0.0 if rows[-1]["terminal"] else rows[-1]["bootstrap"]
        if next_v is None or not math.isfinite(next_v):
            raise ValueError("TB-GAE truncated final chunk lacks a finite old-critic bootstrap value")
        next_a = 0.0  # Finite rollout GAE tail, distinct from a zero value bootstrap.
        for row in reversed(rows):
            if not all(
                math.isfinite(row[k]) for k in ("base", "first_value", "value_coefficient", "advantage_coefficient")
            ):
                raise ValueError("TB-GAE boundary summary must be finite")
            result[(episode, row["chunk_index"])] = (next_v, next_a)
            first_a = row["base"] + row["value_coefficient"] * next_v + row["advantage_coefficient"] * next_a
            next_v, next_a = row["first_value"], first_a
    return result


def gather_and_solve_boundaries(records, group=None):
    import torch.distributed as dist

    if group is None or dist.get_world_size(group) == 1:
        return solve_episode_boundaries(records)
    parts = [None] * dist.get_world_size(group)
    dist.all_gather_object(parts, records, group=group)
    return solve_episode_boundaries([row for part in parts for row in part])
