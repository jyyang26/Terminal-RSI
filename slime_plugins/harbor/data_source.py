from __future__ import annotations

import copy
import glob
import json
import os
import random
from collections import deque
from pathlib import Path
from typing import Any

import yaml

from slime.rollout.data_source import DataSource
from slime.utils.types import Sample


class HarborTaskDataSource(DataSource):
    """Data source that yields Harbor trial/task entries as slime Samples.

    Entries can be provided via ``args.harbor_tasks`` or ``args.harbor_tasks_path``.
    Each entry is kept in ``sample.metadata["harbor_task_entry"]`` and consumed by
    ``slime_plugins.harbor.rollout.generate_rollout``.
    """

    def __init__(self, args):
        self.args = args
        self.entries = _load_task_entries(args)
        self.requeued_groups = deque()
        self.sample_offset = 0
        self.sample_group_index = 0
        self.sample_index = 0

        # Task ORDER. _discover_task_entries returns `sorted(glob(...))`, and
        # get_samples walks that list with a monotonic cursor, so WITHOUT shuffling
        # every step draws an alphabetically CONTIGUOUS block of tasks. Since these
        # task names are family-prefixed (blender_*, cmake_*, openfoam_*, ...), a
        # 256-task batch is then drawn from only one or two families -- i.e. the
        # whole batch is uniformly hard or uniformly easy, and the check-count
        # scale (family p90 spans 1..795) swings step to step. That is bad for PPO:
        # the critic sees a non-stationary target distribution and the advantage
        # baseline is estimated from a non-representative sample.
        #
        # Shuffling per EPOCH (not once) keeps every step's batch family-mixed while
        # still visiting each task exactly once per epoch. The permutation is derived
        # deterministically from (rollout_seed, epoch), so it is reproducible and a
        # resume needs no extra state beyond sample_offset -- the epoch is recomputed
        # from the cursor. Gated on --rollout-shuffle, matching slime core's
        # RolloutDataSourceWithBuffer (data_source.py:85), so behaviour is unchanged
        # when the flag is off.
        self._shuffle = bool(getattr(args, "rollout_shuffle", False))
        self._shuffle_seed = int(getattr(args, "rollout_seed", 42) or 0)
        self._shuffled_epoch: int | None = None
        self._shuffled_order: list[int] = []

        # Held-out evaluation pool. Loaded from a separate task dir/glob so that
        # periodic eval (slime --eval-interval) measures generalization on a
        # fixed set (e.g. Terminal-Bench 2) instead of the training tasks, and
        # never advances the training cursor (`sample_offset`). Empty unless
        # `harbor_eval_tasks_dir` / `harbor_eval_tasks_glob` is configured.
        # NOTE: eval is deliberately NOT shuffled -- a fixed order keeps the
        # eval metric comparable across steps.
        self.eval_entries = _load_eval_task_entries(args)
        self.eval_sample_group_index = 0
        self.eval_sample_index = 0

    def _entry_at(self, offset: int) -> dict[str, Any]:
        """Task at absolute cursor ``offset``, honouring the per-epoch shuffle."""
        total = len(self.entries)
        position = offset % total
        if not self._shuffle:
            return self.entries[position]

        epoch = offset // total
        if self._shuffled_epoch != epoch:
            # random.Random over a fixed index list: no numpy dependency, and the
            # same (seed, epoch) always yields the same permutation on every actor.
            # Seed is derived arithmetically rather than via hash() so it is stable
            # across processes regardless of PYTHONHASHSEED.
            order = list(range(total))
            random.Random(self._shuffle_seed * 1_000_003 + epoch).shuffle(order)
            self._shuffled_epoch = epoch
            self._shuffled_order = order
        return self.entries[self._shuffled_order[position]]

    def get_samples(self, num_samples: int, evaluation: bool = False) -> list[list[Sample]]:
        if evaluation:
            return self._get_eval_samples()

        if not self.entries:
            raise ValueError("HarborTaskDataSource has no task entries")

        groups = []
        while self.requeued_groups and len(groups) < num_samples:
            groups.append(self.requeued_groups.popleft())

        for _ in range(num_samples):
            if len(groups) >= num_samples:
                break

            entry = copy.deepcopy(self._entry_at(self.sample_offset))
            self.sample_offset += 1

            group = []
            for sample_rank in range(getattr(self.args, "n_samples_per_prompt", 1)):
                sample = Sample()
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                sample.prompt = _entry_prompt(entry)
                sample.metadata = {
                    "harbor_task_entry": copy.deepcopy(entry),
                    "harbor_sample_rank": sample_rank,
                }
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            groups.append(group)
        return groups

    def _get_eval_samples(self) -> list[list[Sample]]:
        """Yield one group per held-out eval task (the FULL eval set).

        Unlike training, this iterates every entry in ``self.eval_entries``
        exactly once, draws ``n_samples_per_eval_prompt`` samples per task, and
        uses its own counters so the training cursor (``sample_offset`` /
        ``sample_group_index`` / ``sample_index``) is never touched. The
        requeue buffer is also bypassed — eval is a clean fixed-set pass.
        """
        if not self.eval_entries:
            raise ValueError(
                "HarborTaskDataSource has no eval task entries; set "
                "`harbor_eval_tasks_dir` or `harbor_eval_tasks_glob` to enable eval."
            )

        n_per_prompt = int(
            getattr(self.args, "n_samples_per_eval_prompt", None)
            or getattr(self.args, "n_samples_per_prompt", 1)
            or 1
        )

        groups = []
        for entry_template in self.eval_entries:
            entry = copy.deepcopy(entry_template)
            group = []
            for sample_rank in range(n_per_prompt):
                sample = Sample()
                sample.group_index = self.eval_sample_group_index
                sample.index = self.eval_sample_index
                sample.prompt = _entry_prompt(entry)
                sample.metadata = {
                    "harbor_task_entry": copy.deepcopy(entry),
                    "harbor_sample_rank": sample_rank,
                }
                self.eval_sample_index += 1
                group.append(sample)
            self.eval_sample_group_index += 1
            groups.append(group)
        return groups

    def add_samples(self, samples: list[list[Sample]]):
        self.requeued_groups.extend(samples)

    def save(self, rollout_id):
        save_dir = getattr(self.args, "save", None)
        if not save_dir:
            return
        import torch

        path = Path(save_dir) / "rollout" / f"harbor_data_source_state_{rollout_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sample_offset": self.sample_offset,
                "sample_group_index": self.sample_group_index,
                "sample_index": self.sample_index,
                "requeued_groups": list(self.requeued_groups),
            },
            path,
        )

    def load(self, rollout_id=None):
        load_dir = getattr(self.args, "load", None)
        if not load_dir:
            return
        if rollout_id is None:
            return
        path = Path(load_dir) / "rollout" / f"harbor_data_source_state_{rollout_id}.pt"
        if not path.exists():
            return
        import torch

        state = torch.load(path, weights_only=False)
        self.sample_offset = state.get("sample_offset", 0)
        self.sample_group_index = state.get("sample_group_index", 0)
        self.sample_index = state.get("sample_index", 0)
        self.requeued_groups = deque(state.get("requeued_groups", []))

    def __len__(self) -> int:
        return len(self.entries)


def _load_task_entries(args) -> list[dict[str, Any]]:
    entries = getattr(args, "harbor_tasks", None)
    if entries is None:
        stable_path = getattr(args, "harbor_stable_tasks_path", None)
        if stable_path:
            from slime_plugins.harbor.stable_tasks import load_stable_task_entries

            return load_stable_task_entries(
                stable_path,
                materialized_tasks_dir=getattr(args, "harbor_materialized_tasks_dir", "/tmp/slime-harbor-stable-tasks"),
                sif_cache_dir=getattr(args, "harbor_stable_sif_cache_dir", None),
                limit=getattr(args, "harbor_stable_tasks_limit", None),
                offset=int(getattr(args, "harbor_stable_tasks_offset", 0) or 0),
                apt_proxy=getattr(args, "harbor_stable_apt_proxy", None) or os.environ.get("APT_PROXY", ""),
            )
    if entries is None:
        task_dir = getattr(args, "harbor_tasks_dir", None)
        task_glob = getattr(args, "harbor_tasks_glob", None)
        if task_dir or task_glob:
            entries = _discover_task_entries(
                tasks_dir=Path(task_dir).expanduser() if task_dir else None,
                task_glob=str(task_glob) if task_glob else None,
                limit=getattr(args, "harbor_tasks_limit", None),
                offset=int(getattr(args, "harbor_tasks_offset", 0) or 0),
            )
    if entries is None:
        path = getattr(args, "harbor_tasks_path", None)
        if path:
            entries = _read_entries_file(Path(path))

    if entries is None:
        raise ValueError(
            "Set `harbor_tasks`, `harbor_tasks_path`, `harbor_tasks_dir`, "
            "`harbor_tasks_glob`, or `harbor_stable_tasks_path` "
            "in --custom-config-path for Harbor rollout."
        )
    if not isinstance(entries, list):
        raise ValueError("Harbor task entries must be a list")
    return [_normalize_entry(entry) for entry in entries]


def _load_eval_task_entries(args) -> list[dict[str, Any]]:
    """Load the held-out eval task pool. Returns ``[]`` when unconfigured.

    Mirrors the dir/glob discovery branch of ``_load_task_entries`` but is
    independent of the training pool and never raises when nothing is set, so
    runs without eval configured behave exactly as before.
    """
    eval_dir = getattr(args, "harbor_eval_tasks_dir", None)
    eval_glob = getattr(args, "harbor_eval_tasks_glob", None)
    if not eval_dir and not eval_glob:
        return []
    entries = _discover_task_entries(
        tasks_dir=Path(eval_dir).expanduser() if eval_dir else None,
        task_glob=str(eval_glob) if eval_glob else None,
        limit=getattr(args, "harbor_eval_tasks_limit", None),
        offset=int(getattr(args, "harbor_eval_tasks_offset", 0) or 0),
    )
    return [_normalize_entry(entry) for entry in entries]


def _discover_task_entries(
    *,
    tasks_dir: Path | None,
    task_glob: str | None,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    if tasks_dir is not None:
        pattern = task_glob or "*/task.toml"
        candidates = sorted(tasks_dir.glob(pattern))
    else:
        pattern = task_glob or "*/task.toml"
        if os.path.isabs(pattern):
            candidates = [Path(path) for path in sorted(glob.glob(pattern))]
        else:
            candidates = sorted(Path().glob(pattern))

    task_dirs = []
    for candidate in candidates:
        path = candidate
        if path.is_dir():
            task_toml = path / "task.toml"
            if not task_toml.exists():
                continue
            task_dirs.append(path)
        elif path.name == "task.toml":
            task_dirs.append(path.parent)

    selected = task_dirs[offset:]
    if limit is not None and int(limit) > 0:
        selected = selected[: int(limit)]

    return [{"task": {"path": str(path)}, "prompt": path.name} for path in selected]


def _read_entries_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open() as f:
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f) or []
        return json.load(f)


def _normalize_entry(entry: Any) -> dict[str, Any]:
    if isinstance(entry, str):
        return {"task": {"path": os.path.expanduser(entry)}}
    if not isinstance(entry, dict):
        raise ValueError(f"Harbor task entry must be dict or path string, got {type(entry)}")
    if "task" not in entry and "trial_config" not in entry:
        if "path" in entry or "name" in entry:
            return {"task": entry}
    return entry


def _entry_prompt(entry: dict[str, Any]) -> str:
    if prompt := entry.get("prompt"):
        return str(prompt)
    task = entry.get("task") or (entry.get("trial_config") or {}).get("task") or {}
    if isinstance(task, dict):
        return str(task.get("path") or task.get("name") or task)
    return str(task)
