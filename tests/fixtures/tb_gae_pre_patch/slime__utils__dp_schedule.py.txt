"""Per-group DP/microbatch scheduling.

Pure-Python logic that decides, for one rollout batch's worth of sample lengths,
how to group samples into micro-batches and which DP rank owns each mbs.
Lives outside the ray/sglang-importing modules so it can be unit-tested
under CPU-only CI.

The scheduling philosophy is **pack first, distribute second**:

  1. Group samples by training group id (``group_indices[i]`` =
     ``samples[i].group_id`` with a fallback to ``samples[i].index``) and
     split groups into steps of ``global_batch_size`` groups each. In the
     common case one rollout emits one training sample so this is the same as
     a contiguous chunk; under compact / subagent one rollout may emit
     multiple training samples, in which case all of those samples stay in the
     same step.
  2. For each step, pack its samples into ``K`` micro-batches with a
     single first-fit pass (dynamic batch) or fixed-size chunking
     (static batch).
  3. Adjust ``K`` to a multiple of ``dp_size * (mb_group if vpp>1 else 1)``
     by splitting the largest multi-sample bins (dynamic only).
  4. Distribute the ``K`` mbs across ``dp_size`` ranks, ``K / dp_size``
     each, with either a strided round-robin or a Karmarkar-Karp pass on
     mbs token sums.

Invariants guaranteed by :func:`build_dp_schedule` (asserted by the tests):
  - every DP rank runs the **same** ``num_microbatches`` per training step
    (required for PP sync);
  - every mbs (dynamic path) holds ``<= max_tokens_per_gpu * cp_size``
    tokens, with one exception — an individual sample larger than that cap
    lands alone in its own mbs (and that mbs is the only one allowed to
    exceed the cap);
  - the union of per-rank sample indices equals the set of samples kept
    after trimming trailing groups (every kept sample placed exactly
    once);
  - flattening ``micro_batch_indices`` for a rank yields
    ``range(num_samples_rank)`` (each rank's samples are tiled exactly
    once by its mbs schedule).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Hashable

from slime.utils.seqlen_balancing import expand_bins_by_splitting, first_fit_pack, get_seqlen_balanced_partitions

logger = logging.getLogger(__name__)


def _pack_step_into_mbs(
    step_lengths: list[int],
    *,
    use_dynamic_batch_size: bool,
    max_per_bin: int | None,
    micro_batch_size: int | None,
) -> list[list[int]]:
    """Group a step's samples into mbs. Returns ``mbs[k]`` = local indices into ``step_lengths``."""
    if use_dynamic_batch_size:
        assert max_per_bin is not None
        return first_fit_pack(step_lengths, max_per_bin)
    assert micro_batch_size is not None
    n = len(step_lengths)
    return [list(range(i, min(i + micro_batch_size, n))) for i in range(0, n, micro_batch_size)]


def _static_mbs_count(sample_count: int, micro_batch_size: int) -> int:
    return (sample_count + micro_batch_size - 1) // micro_batch_size


def select_zero_reward_groups_to_drop_for_static_alignment(
    group_indices: list[Hashable],
    raw_rewards: list[float],
    *,
    global_batch_size: int,
    micro_batch_size: int,
    align_to: int,
) -> list[Hashable]:
    """Choose zero-reward groups to drop so static DP micro-batches align.

    Static batching cannot split or pad micro-batches without changing its
    fixed-size invariant. When a rollout step has an odd extra sample from a
    split trajectory, prefer dropping a zero-reward training group that makes
    the step align after the following groups slide forward. If no such group
    exists, raise with enough context instead of silently changing the batch.
    """
    assert len(group_indices) == len(raw_rewards), (
        f"group_indices/raw_rewards length mismatch: {len(group_indices)} != {len(raw_rewards)}"
    )
    assert global_batch_size > 0, f"global_batch_size must be > 0, got {global_batch_size}"
    assert micro_batch_size > 0, f"micro_batch_size must be > 0, got {micro_batch_size}"
    if align_to <= 1:
        return []

    group_order: list[Hashable] = []
    group_sizes: dict[Hashable, int] = {}
    group_zero_reward: dict[Hashable, bool] = {}
    for group_id, reward in zip(group_indices, raw_rewards, strict=True):
        if group_id not in group_sizes:
            group_order.append(group_id)
            group_sizes[group_id] = 0
            group_zero_reward[group_id] = True
        group_sizes[group_id] += 1
        group_zero_reward[group_id] = group_zero_reward[group_id] and abs(float(reward)) <= 1e-12

    if len(group_order) < global_batch_size:
        return []

    def step_is_aligned(groups: list[Hashable], step_i: int) -> bool:
        start = step_i * global_batch_size
        end = start + global_batch_size
        step_sample_count = sum(group_sizes[group_id] for group_id in groups[start:end])
        return _static_mbs_count(step_sample_count, micro_batch_size) % align_to == 0

    remaining = list(group_order)
    dropped: list[Hashable] = []

    while True:
        num_steps = len(remaining) // global_batch_size
        bad_step = None
        bad_sample_count = 0
        bad_num_mbs = 0
        for step_i in range(num_steps):
            start = step_i * global_batch_size
            end = start + global_batch_size
            step_sample_count = sum(group_sizes[group_id] for group_id in remaining[start:end])
            num_mbs = _static_mbs_count(step_sample_count, micro_batch_size)
            if num_mbs % align_to != 0:
                bad_step = step_i
                bad_sample_count = step_sample_count
                bad_num_mbs = num_mbs
                break

        if bad_step is None:
            return dropped

        start = bad_step * global_batch_size
        end = start + global_batch_size
        candidate_entries: list[tuple[int, int, Hashable, list[Hashable]]] = []
        for pos, group_id in enumerate(remaining[start:end], start=start):
            if not group_zero_reward.get(group_id, False):
                continue
            trial_remaining = [candidate for candidate in remaining if candidate != group_id]
            if len(trial_remaining) // global_batch_size <= bad_step:
                continue
            if step_is_aligned(trial_remaining, bad_step):
                candidate_entries.append((group_sizes[group_id], pos, group_id, trial_remaining))

        if not candidate_entries:
            step_groups = remaining[start:end]
            zero_groups = [group_id for group_id in step_groups if group_zero_reward.get(group_id, False)]
            zero_group_sizes = {group_id: group_sizes[group_id] for group_id in zero_groups}
            raise AssertionError(
                "static path cannot auto-drop a zero-reward group to satisfy DP alignment: "
                f"step={bad_step}, step_groups={len(step_groups)}, step_samples={bad_sample_count}, "
                f"num_mbs={bad_num_mbs}, micro_batch_size={micro_batch_size}, align_to={align_to}, "
                f"zero_reward_group_sizes={zero_group_sizes}. "
                "Use dynamic batch size or ensure converted rollout groups produce an aligned sample count."
            )

        # Prefer dropping the largest zero-reward group; if tied, drop the
        # later group so earlier ordering shifts as little as possible.
        candidate_entries.sort(key=lambda item: (item[0], item[1]), reverse=True)
        _, _, drop_group_id, remaining = candidate_entries[0]
        dropped.append(drop_group_id)


def build_dp_schedule(
    args: Any,
    train_parallel_config: dict,
    total_lengths: list[int],
    *,
    global_batch_size: int,
    group_indices: list[int],
    allow_partial_global_batch: bool = False,
) -> tuple[list[list[int]], list[list[list[int]]], list[int], list[int]]:
    """Compute the per-rank DP partition and micro-batch schedule.

    See module docstring for the pack-first-distribute-second strategy.

    Args:
        args: Namespace with ``micro_batch_size``, ``use_dynamic_batch_size``,
            ``max_tokens_per_gpu``, ``balance_data``.
        train_parallel_config: ``{"dp_size", "cp_size", "vpp_size",
            "microbatch_group_size_per_vp_stage"}``.
        total_lengths: token count per sample, indexed globally.
        global_batch_size: number of groups (NOT training samples) per
            training step. Number of training steps =
            ``num_groups // global_batch_size``; trailing groups whose
            samples don't fit are dropped unless ``allow_partial_global_batch``
            is true.
        group_indices: group id for each sample. Samples sharing the same id
            are kept together in one step.
        allow_partial_global_batch: include a final short step instead of
            dropping trailing groups. This is intended for rollout modes that
            intentionally return fewer groups because straggler trials were
            cancelled.

    Returns:
        ``(partitions, micro_batch_indices, num_microbatches, global_batch_sizes)``.
        ``global_batch_sizes[s]`` = group count for step s (constant
        ``global_batch_size`` for every step).
    """
    dp_size = train_parallel_config["dp_size"]
    cp_size = train_parallel_config["cp_size"]
    vpp_size = train_parallel_config["vpp_size"]
    mb_group = train_parallel_config["microbatch_group_size_per_vp_stage"]

    max_per_bin = None
    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None
        max_per_bin = args.max_tokens_per_gpu * cp_size

    # mbs count per step must be divisible by (dp_size * mb_group_for_vpp) so
    # every rank ends up with the same num_mbs and (for VPP) the per-rank mbs
    # count is a multiple of mb_group.
    align_to = dp_size * (mb_group if vpp_size > 1 else 1)

    # Group samples by group id (preserve first-occurrence order). All samples
    # from one group stay in a single step so the per-group loss reducer is
    # well-defined.
    group_id_to_samples: dict[int, list[int]] = {}
    for sample_pos, group_id in enumerate(group_indices):
        group_id_to_samples.setdefault(group_id, []).append(sample_pos)
    group_ids = list(group_id_to_samples.keys())

    if allow_partial_global_batch:
        num_steps = math.ceil(len(group_ids) / global_batch_size)
    else:
        num_steps = len(group_ids) // global_batch_size
    assert num_steps >= 1, (
        f"num_groups ({len(group_ids)}) < global_batch_size ({global_batch_size}); "
        f"need at least one group per step."
    )

    partitions: list[list[int]] = [[] for _ in range(dp_size)]
    micro_batch_indices: list[list[list[int]]] = [[] for _ in range(dp_size)]
    num_microbatches: list[int] = []
    global_batch_sizes: list[int] = []

    for step_i in range(num_steps):
        step_groups = group_ids[step_i * global_batch_size : (step_i + 1) * global_batch_size]
        sample_indices = [pos for group_id in step_groups for pos in group_id_to_samples[group_id]]
        step_lengths = [total_lengths[i] for i in sample_indices]
        step_global_batch_size = len(step_groups)
        global_batch_sizes.append(step_global_batch_size)
        assert len(sample_indices) >= dp_size, (
            f"step {step_i}: {len(sample_indices)} samples < dp_size {dp_size}; "
            f"each step needs at least one sample per rank."
        )

        # 1. Pack samples in this step into mbs with one global pass.
        # ``step_mbs`` indices are LOCAL into ``sample_indices``.
        step_mbs = _pack_step_into_mbs(
            step_lengths,
            use_dynamic_batch_size=args.use_dynamic_batch_size,
            max_per_bin=max_per_bin,
            micro_batch_size=getattr(args, "micro_batch_size", None),
        )

        # 2. Align mbs count to a multiple of ``align_to``.
        target_K = max(((len(step_mbs) + align_to - 1) // align_to) * align_to, align_to)
        if target_K != len(step_mbs):
            if args.use_dynamic_batch_size:
                expand_bins_by_splitting(step_mbs, target_K, step_lengths)
                assert len(step_mbs) == target_K, (
                    f"dynamic path: could only produce {len(step_mbs)} mbs after maximal splitting; "
                    f"need {target_K}. step {step_i} has {len(sample_indices)} samples, below the "
                    f"alignment threshold ({align_to})."
                )
            else:
                raise AssertionError(
                    f"static path: num_mbs ({len(step_mbs)}) is not a multiple of "
                    f"dp_size * mb_group ({align_to}); got "
                    f"step_size={len(sample_indices)}, micro_batch_size={args.micro_batch_size}, "
                    f"dp_size={dp_size}, mb_group={mb_group if vpp_size > 1 else 1}. "
                    f"Splitting static mbs would break the fixed-size invariant; adjust the config "
                    f"so step_size % (dp_size * micro_batch_size * mb_group) == 0."
                )

        K = len(step_mbs)
        num_mbs_per_rank = K // dp_size
        num_microbatches.append(num_mbs_per_rank)

        # 3. Distribute mbs across ranks: KK on mbs token sums when balance_data is on,
        # otherwise a strided round-robin. Both produce ``num_mbs_per_rank`` mbs per
        # rank (equal_size=True is what KK needs for PP to stay synced).
        if args.balance_data:
            mbs_token_sums = [sum(step_lengths[i] for i in bin_) for bin_ in step_mbs]
            rank_mbs_idx = get_seqlen_balanced_partitions(mbs_token_sums, dp_size, equal_size=True)
        else:
            rank_mbs_idx = [list(range(r, K, dp_size)) for r in range(dp_size)]

        # 4. Build per-rank partitions (global sample indices) and micro_batch_indices
        # (local indices into partitions[r]).
        for r in range(dp_size):
            for mbs_idx in rank_mbs_idx[r]:
                mbs_locals = step_mbs[mbs_idx]  # local indices into sample_indices
                local_start = len(partitions[r])
                partitions[r].extend(sample_indices[i] for i in mbs_locals)
                micro_batch_indices[r].append(list(range(local_start, local_start + len(mbs_locals))))

    return partitions, micro_batch_indices, num_microbatches, global_batch_sizes
