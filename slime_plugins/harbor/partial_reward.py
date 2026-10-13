from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from slime.utils.types import Sample
from slime_plugins.harbor.turn_grouped_reward import (
    _group_and_trajectory_key,
    _normalize_rewards,
    _reward_value,
    _should_normalize,
)


DEFAULT_PARTIAL_REWARD_MAX = 0.4
# Global fixed divisor for the test-count reward variant. shaped = passed / SCALE,
# so a task with ~SCALE tests fully passed scores ~1.0. Chosen ~p90 of the synth-all
# test-count distribution (median 4, p90 ~19, max ~35); env-overridable. The absolute
# value is not critical because advantage whitening (--normalize-advantages) rescales
# downstream — SCALE only needs to keep critic value targets in a sane range.
DEFAULT_TESTCOUNT_SCALE = 20.0

# Max step-efficiency bonus (bonus at norm=1.0, i.e. the shortest solve in a
# same-test-count bucket). 0.05 == 1/DEFAULT_TESTCOUNT_SCALE == one passed test's
# worth of reward: efficiency is a visible signal but never overpowers the
# correctness (passed-test count) gap. Env-overridable (HARBOR_EFFICIENCY_BONUS_BETA).
DEFAULT_EFFICIENCY_BONUS_BETA = 0.05

# --- run#6 length-penalty reward (test_count_length_penalty_rewards) -----------------
# Maximum length penalty (at tokens >= BUDGET). 0.03 < 1/DEFAULT_TESTCOUNT_SCALE (0.05
# == one passed test) is DELIBERATE: it makes correctness strictly lexicographic over
# length — passing one more test (+0.05) always beats any length saving (<=0.03), so
# the model never trades a passed test for brevity, but among equally-correct
# trajectories the shorter one always wins. Env-overridable (HARBOR_LENGTH_PENALTY_LAMBDA).
DEFAULT_LENGTH_PENALTY_LAMBDA = 0.03
# Token budget the penalty normalizes against: penalty = lambda * clamp(tokens/BUDGET, 0, 1).
# 40000 ~ the observed max output length (synth-all eval: median ~10k, mean ~15k, max ~38k),
# so the longest trajectories pay ~full lambda and short ones pay ~0. Uses OUTPUT tokens
# (harbor_total_output_tokens: model-generated only, excludes prompt/context) — the true
# "output cost" the user asked to bound. Env-overridable (HARBOR_LENGTH_TOKEN_BUDGET).
DEFAULT_LENGTH_TOKEN_BUDGET = 40000.0

# --- run#7 MULTIPLICATIVE length-discount reward (test_count_length_discount_rewards) -----
# run#6's ADDITIVE penalty (passed/SCALE - lambda*L) failed catastrophically (avg_turns
# 22->50+, RUNBOOK §18): with lambda kept < one test's value to preserve "correctness
# strictly dominant", the fixed -lambda penalty was noise next to the correctness gradient
# (more turns -> more tests brushed -> higher passed/SCALE), so farming won. run#7 makes
# length erode correctness MULTIPLICATIVELY so it can never be ignored, with a free-zone so
# healthy-length solving is never punished, plus an additive penalty on the zero-pass
# trajectories that multiplication would otherwise leave at exactly 0 (the farming hot-spot).
#
#   turns_norm = clamp((turns - L_FREE) / (L_FULL - L_FREE), 0, 1)   # <=L_FREE: 0 ; >=L_FULL: 1
#   passed > 0:  reward = (passed/SCALE) * (1 - ALPHA * turns_norm)   # discount, anti-farming
#   passed == 0: reward = 0 - BETA * turns_norm                       # penalize zero-pass rambling
#
# L_FREE=25 (healthy avg_turns ~22 + margin): normal solving is never discounted.
# L_FULL=50 (run#6's farmed plateau): full discount kicks in at the farmed length.
# ALPHA=0.5: a fully-farmed solve keeps only half its correctness reward, so brushing one
#   extra test by rambling to 50 turns (0.25*0.5=0.125) scores BELOW a 4-test healthy solve
#   (0.20) — farming is strictly unprofitable, unlike run#5/#6.
# BETA=0.1: a zero-pass trajectory rambled to L_FULL scores -0.1 (vs 0 under pure
#   multiplication), removing the free reward-farming floor for unsolved long trajectories.
DEFAULT_LENGTH_DISCOUNT_FREE_TURNS = 25.0
DEFAULT_LENGTH_DISCOUNT_FULL_TURNS = 50.0
DEFAULT_LENGTH_DISCOUNT_ALPHA = 0.5
DEFAULT_LENGTH_DISCOUNT_ZERO_BETA = 0.1


def dispatch_rewards(args, samples: list[Sample], **kwargs) -> tuple[list[float], list[float]]:
    """Mode-selected reward entry point. Point ``--custom-reward-post-process-path`` here.

    Lets a recipe switch reward formulations WITHOUT editing any hook body, so an
    in-flight run keeps the exact function it resolved at startup:

      ``legacy_p90`` (default)  -> ``test_count_progress_rewards``
                                   R = passed / SCALE(family), unbounded above.
      ``interpolated_ctrf``     -> ``interpolated_ctrf_reward.interpolated_ctrf_rewards``
                                   Stage S2: bounded outcome-dominant score with a
                                   [p75, p90] divisor, structural weighting and an
                                   anti-hack gate.

    Mode comes from ``harbor_reward_mode`` (YAML -> args, the reliable path) or
    ``HARBOR_REWARD_MODE`` (env, only honoured when the var actually reaches the Ray
    workers -- it is NOT in the inner launcher's RUNTIME_ENV_JSON whitelist).
    Default is ``legacy_p90``, so omitting the key reproduces current behaviour.

    An unknown mode raises rather than silently falling back: a typo in the reward
    mode would otherwise train a whole run on the wrong objective.
    """
    mode = getattr(args, "harbor_reward_mode", None) or os.environ.get("HARBOR_REWARD_MODE") or "legacy_p90"
    mode = str(mode).strip().lower()

    if mode in ("legacy_p90", "legacy", "test_count_progress"):
        return test_count_progress_rewards(args, samples, **kwargs)

    if mode in ("interpolated_ctrf", "s2"):
        from slime_plugins.harbor.interpolated_ctrf_reward import interpolated_ctrf_rewards

        return interpolated_ctrf_rewards(args, samples, **kwargs)

    raise ValueError(
        f"unknown harbor_reward_mode {mode!r}; expected 'legacy_p90' or 'interpolated_ctrf'. "
        "NOTE: 'outcome_dense' is NOT implemented in this repo -- no such hook exists; "
        "use 'interpolated_ctrf' for the bounded outcome-dominant reward."
    )


def trajectory_grouped_progress_rewards(args, samples: list[Sample], **kwargs) -> tuple[list[float], list[float]]:
    """Add CTRF-based progress reward, then normalize by Harbor trajectory group.

    Terminal-bench tasks write verifier progress to ``verifier/ctrf.json`` when
    verifier execution reaches pytest. This hook keeps terminal success reward
    dominant, uses CTRF pass ratio as a capped partial reward for unsuccessful
    trials, and broadcasts the trajectory-level advantage to all per-step
    samples from the same trial.
    """

    partial_reward_max = _partial_reward_max(args)
    progress_cache: dict[str | None, ProgressScore] = {}

    group_to_trajectories: "OrderedDict[Any, OrderedDict[Any, list[int]]]" = OrderedDict()
    for index, sample in enumerate(samples):
        group_key, trajectory_key = _group_and_trajectory_key(sample)
        trajectories = group_to_trajectories.setdefault(group_key, OrderedDict())
        trajectories.setdefault(trajectory_key, []).append(index)

    raw_rewards = [0.0] * len(samples)
    for trajectories in group_to_trajectories.values():
        for indices in trajectories.values():
            trajectory_reward = _trajectory_shaped_reward(args, samples[indices[0]], partial_reward_max, progress_cache)
            for index in indices:
                _annotate_sample(samples[index], trajectory_reward)
                raw_rewards[index] = trajectory_reward.shaped_reward

    if not _should_normalize(args):
        return raw_rewards, raw_rewards

    processed = [0.0] * len(samples)
    for trajectories in group_to_trajectories.values():
        trajectory_items = list(trajectories.items())
        trajectory_rewards = [raw_rewards[indices[0]] for _, indices in trajectory_items]
        trajectory_advantages = _normalize_rewards(args, trajectory_rewards)
        for (_, indices), advantage in zip(trajectory_items, trajectory_advantages, strict=True):
            for index in indices:
                processed[index] = advantage

    return raw_rewards, processed


post_process_rewards = trajectory_grouped_progress_rewards


def test_count_progress_rewards(args, samples: list[Sample], **kwargs) -> tuple[list[float], list[float]]:
    """Test-count reward variant: shaped = passed_tests / SCALE (absolute test count).

    Unlike ``trajectory_grouped_progress_rewards`` (which uses the passed/total
    *ratio*, weighting every task equally at max 1.0), this version scores by the
    ABSOLUTE number of passed tests. Hard tasks with many tests can therefore earn
    a larger reward than easy few-test tasks, so their learning signal is not
    flattened by ratio normalization. A fully-passed task scores ``total / SCALE``
    (i.e. passed == total), which is your requested "全通过给满分=total".

    SCALE is a GLOBAL FIXED constant (``HARBOR_TESTCOUNT_SCALE``), NOT a per-batch
    max — a per-batch max would drift step to step and give the critic an
    inconsistent regression target. Cross-step scale consistency is what makes the
    value function learnable. Final scale handling is left to advantage whitening
    (``--normalize-advantages``, loss.py); under PPO ``_should_normalize`` is False
    so raw shaped rewards flow straight into critic + GAE.
    """

    scale = _testcount_scale(args)
    family_scales = _family_scale_table(args)
    progress_cache: dict[str | None, ProgressScore] = {}

    group_to_trajectories: "OrderedDict[Any, OrderedDict[Any, list[int]]]" = OrderedDict()
    for index, sample in enumerate(samples):
        group_key, trajectory_key = _group_and_trajectory_key(sample)
        trajectories = group_to_trajectories.setdefault(group_key, OrderedDict())
        trajectories.setdefault(trajectory_key, []).append(index)

    raw_rewards = [0.0] * len(samples)
    for trajectories in group_to_trajectories.values():
        for indices in trajectories.values():
            head = samples[indices[0]]
            # Per-family divisor: a full solve is worth ~1.0 in every family, so
            # advantages are comparable across a batch of mixed-difficulty tasks.
            # Falls back to the global scale when no table is configured.
            sample_scale, family_id = _scale_for_sample(head, scale, family_scales)
            trajectory_reward = _trajectory_testcount_reward(args, head, sample_scale, progress_cache)
            for index in indices:
                _annotate_sample(samples[index], trajectory_reward)
                _annotate_family_scale(samples[index], family_id=family_id, scale=sample_scale)
                raw_rewards[index] = trajectory_reward.shaped_reward

    if not _should_normalize(args):
        return raw_rewards, raw_rewards

    processed = [0.0] * len(samples)
    for trajectories in group_to_trajectories.values():
        trajectory_items = list(trajectories.items())
        trajectory_rewards = [raw_rewards[indices[0]] for _, indices in trajectory_items]
        trajectory_advantages = _normalize_rewards(args, trajectory_rewards)
        for (_, indices), advantage in zip(trajectory_items, trajectory_advantages, strict=True):
            for index in indices:
                processed[index] = advantage

    return raw_rewards, processed


def success_smoothed_progress_rewards(args, samples: list[Sample], **kwargs) -> tuple[list[float], list[float]]:
    """Reward redesign from rollout_distribution_analysis.html (kxm RESUME19 audit).

        reward = float(full_success) + passed / (passed + family_divisor)

    vs the legacy ``test_count_progress_rewards`` (``passed / SCALE(family)``), this
    changes TWO things together (the report shows the divisor change alone is not
    enough -- it would push failing trajectories with reward >= 1 from 158 to 382):

      1. SUCCESS BONUS. ``float(full_success)`` adds exactly 1.0 when the terminal
         gate passed (``terminal_reward >= 1.0``), else 0.0. This makes "actually
         solved" strictly dominate any amount of partial progress: every failure is
         < 1 (the smoothed term below is bounded in [0,1)) and every success is >= 1.
         It separates the 48 observed cases where all CTRF checks pass but the final
         gate fails -- which the legacy P/D cannot distinguish.

      2. SMOOTHED PROGRESS ``passed / (passed + D')``. Bounded in [0,1), saturating,
         so a family with many tests can no longer earn a huge raw partial score by
         brushing a lot of checks without solving (the report's point_cloud_lidar /
         blender high-reward-but-failing pathology). D' is the per-family divisor
         (``harbor_family_testcount_scales``); families absent from the table fall
         back to the global ``harbor_testcount_scale``. The companion divisor table
         (18 families re-tuned FOR this formula -- high-scoring-failure families x2,
         long-horizon-low-yield families /4) lives in the run's YAML, NOT here, so it
         stays data-not-code and survives the ray-job-submit env stripping.

    passed and terminal_reward come from the same CTRF parse as the legacy variant
    (``_trajectory_testcount_reward``); a genuinely solved task with no CTRF still
    scores >= 1 via the success bonus. Normalization passthrough is identical to
    ``test_count_progress_rewards`` (under PPO ``_should_normalize`` is False, so raw
    rewards flow straight to critic + GAE).
    """

    scale = _testcount_scale(args)
    family_scales = _family_scale_table(args)
    progress_cache: dict[str | None, ProgressScore] = {}

    group_to_trajectories: "OrderedDict[Any, OrderedDict[Any, list[int]]]" = OrderedDict()
    for index, sample in enumerate(samples):
        group_key, trajectory_key = _group_and_trajectory_key(sample)
        trajectories = group_to_trajectories.setdefault(group_key, OrderedDict())
        trajectories.setdefault(trajectory_key, []).append(index)

    raw_rewards = [0.0] * len(samples)
    for trajectories in group_to_trajectories.values():
        for indices in trajectories.values():
            head = samples[indices[0]]
            # D' = this family's divisor (fallback to the global scale when absent).
            divisor, family_id = _scale_for_sample(head, scale, family_scales)
            trajectory_reward = _trajectory_testcount_reward(args, head, divisor, progress_cache)
            passed = trajectory_reward.progress.passed or 0
            success_bonus = 1.0 if trajectory_reward.terminal_reward >= 1.0 else 0.0
            denom = passed + divisor
            smoothed = (passed / denom) if denom > 0 else 0.0
            reward = success_bonus + max(smoothed, 0.0)
            for index in indices:
                _annotate_sample(samples[index], trajectory_reward)
                _annotate_family_scale(samples[index], family_id=family_id, scale=divisor)
                raw_rewards[index] = reward

    if not _should_normalize(args):
        return raw_rewards, raw_rewards

    processed = [0.0] * len(samples)
    for trajectories in group_to_trajectories.values():
        trajectory_items = list(trajectories.items())
        trajectory_rewards = [raw_rewards[indices[0]] for _, indices in trajectory_items]
        trajectory_advantages = _normalize_rewards(args, trajectory_rewards)
        for (_, indices), advantage in zip(trajectory_items, trajectory_advantages, strict=True):
            for index in indices:
                processed[index] = advantage

    return raw_rewards, processed


def test_count_efficiency_rewards(args, samples: list[Sample], **kwargs) -> tuple[list[float], list[float]]:
    """Test-count reward + a step-efficiency bonus on FULLY-SOLVED trajectories.

    Base reward is identical to :func:`test_count_progress_rewards`
    (shaped = passed / SCALE). On top of that, this variant adds a bonus that
    rewards *solving with fewer steps* — but ONLY for trajectories that fully
    passed (terminal_reward >= 1.0), and only relative to same-difficulty peers.
    This teaches "solve it AND stop early" without penalising hard tasks that
    intrinsically need more turns, and without ever rewarding an unsolved task
    for being short.

    Bonus shape (human-decided 2026-08-09 — bucketed relative ranking):
      * Bucket trajectories in this rollout batch by their test COUNT
        (``progress.total``). Only fully-solved trajectories (terminal >= 1.0)
        participate; the count is the natural difficulty proxy (a task with N
        tests always has N tests regardless of how many were passed).
      * Within a bucket, the SHORTEST trajectory (fewest ``harbor_num_turns``)
        gets ``norm = 1.0`` and the LONGEST gets ``norm = 0.0``, linearly
        interpolated: ``norm = (max_steps - steps) / (max_steps - min_steps)``.
      * A singleton bucket (or all-equal steps) gets ``norm = 1.0`` — a task
        solved with no faster peer to compare against is treated as the best,
        so solving is never discouraged.
      * ``bonus = beta * norm`` with ``beta = HARBOR_EFFICIENCY_BONUS_BETA``
        (default 0.05 == the value of one passed test at SCALE=20). Efficiency
        is therefore a clear signal but never overpowers the multi-notch gap
        that correctness (passed-test count) creates.

    Bonus is added to the RAW shaped reward. Under PPO ``_should_normalize`` is
    False, so raw rewards (base + bonus) flow straight into critic + GAE, and
    advantage whitening (``--normalize-advantages``) rescales downstream — same
    contract as :func:`test_count_progress_rewards`.
    """

    scale = _testcount_scale(args)
    beta = _efficiency_beta(args)
    progress_cache: dict[str | None, ProgressScore] = {}

    group_to_trajectories: "OrderedDict[Any, OrderedDict[Any, list[int]]]" = OrderedDict()
    for index, sample in enumerate(samples):
        group_key, trajectory_key = _group_and_trajectory_key(sample)
        trajectories = group_to_trajectories.setdefault(group_key, OrderedDict())
        trajectories.setdefault(trajectory_key, []).append(index)

    # First pass: base test-count shaped reward per trajectory, plus the fields
    # needed for the efficiency bonus (solved?, test count, step count).
    raw_rewards = [0.0] * len(samples)
    trajectory_records: list[dict[str, Any]] = []
    for trajectories in group_to_trajectories.values():
        for indices in trajectories.values():
            representative = samples[indices[0]]
            trajectory_reward = _trajectory_testcount_reward(args, representative, scale, progress_cache)
            for index in indices:
                _annotate_sample(samples[index], trajectory_reward)
                raw_rewards[index] = trajectory_reward.shaped_reward
            trajectory_records.append(
                {
                    "indices": indices,
                    "base": trajectory_reward.shaped_reward,
                    "solved": trajectory_reward.terminal_reward >= 1.0,
                    "total": trajectory_reward.progress.total,
                    "steps": _sample_num_steps(representative),
                }
            )

    # Second pass: bucket fully-solved trajectories by test count and add the
    # within-bucket step-efficiency bonus.
    buckets: "OrderedDict[Any, list[dict[str, Any]]]" = OrderedDict()
    for record in trajectory_records:
        if not record["solved"] or record["steps"] is None:
            continue
        buckets.setdefault(record["total"], []).append(record)

    for bucket in buckets.values():
        step_values = [record["steps"] for record in bucket]
        max_steps = max(step_values)
        min_steps = min(step_values)
        span = max_steps - min_steps
        for record in bucket:
            norm = 1.0 if span <= 0 else (max_steps - record["steps"]) / span
            bonus = beta * norm
            for index in record["indices"]:
                raw_rewards[index] = record["base"] + bonus
                _annotate_efficiency(samples[index], norm=norm, bonus=bonus)

    if not _should_normalize(args):
        return raw_rewards, raw_rewards

    processed = [0.0] * len(samples)
    for trajectories in group_to_trajectories.values():
        trajectory_items = list(trajectories.items())
        trajectory_rewards = [raw_rewards[indices[0]] for _, indices in trajectory_items]
        trajectory_advantages = _normalize_rewards(args, trajectory_rewards)
        for (_, indices), advantage in zip(trajectory_items, trajectory_advantages, strict=True):
            for index in indices:
                processed[index] = advantage

    return raw_rewards, processed


def test_count_length_discount_rewards(args, samples: list[Sample], **kwargs) -> tuple[list[float], list[float]]:
    """run#7 reward: MULTIPLICATIVE length discount on correctness + zero-pass penalty.

        turns_norm = clamp((turns - L_FREE) / (L_FULL - L_FREE), 0, 1)
        passed > 0:  reward = (passed/SCALE) * (1 - ALPHA * turns_norm)
        passed == 0: reward = 0 - BETA * turns_norm

    Why this over run#6's additive penalty (RUNBOOK §18): run#6 did
    ``passed/SCALE - lambda*L`` with lambda kept below one test's value to preserve
    "correctness strictly dominant". That guaranteed failure — the fixed -lambda was
    noise next to the correctness gradient (more turns -> more tests brushed ->
    higher passed/SCALE), so the model farmed length to avg_turns 50+. Multiplication
    makes length erode correctness ITSELF (and proportionally erodes ALL already-earned
    correctness), so brushing one more test by rambling is double-taxed and becomes
    strictly unprofitable:
      4-test healthy solve (turns<=L_FREE): 0.20 * 1.0    = 0.20
      5-test farmed solve (turns>=L_FULL):  0.25 * (1-0.5)= 0.125  < 0.20  (farming loses)
      5-test healthy solve (turns<=L_FREE): 0.25 * 1.0    = 0.25   (efficiency wins)

    A FREE ZONE (turns <= L_FREE) means normal-length solving is never discounted, so
    genuinely-hard tasks that need ~healthy length are not penalized. The zero-pass
    additive term (BETA) plugs the hole multiplication leaves: ``0 * anything == 0``, so
    pure multiplication would give a rambled zero-pass trajectory the same 0 as a short
    one — yet run#6's stratification showed zero-pass long trajectories are the farming
    hot-spot. BETA makes a rambled zero-pass trajectory go slightly NEGATIVE.

    L is trajectory turn count (``harbor_num_turns``), the direct measure of "rambling"
    and the metric (avg_turns) we track. SCALE/L_FREE/L_FULL/ALPHA/BETA are global fixed
    constants (env-overridable). Under PPO ``_should_normalize`` is False so raw shaped
    rewards flow into critic + GAE; advantage whitening rescales downstream.
    """

    scale = _testcount_scale(args)
    l_free = _length_discount_free_turns(args)
    l_full = _length_discount_full_turns(args)
    alpha = _length_discount_alpha(args)
    beta = _length_discount_zero_beta(args)
    span = l_full - l_free
    progress_cache: dict[str | None, ProgressScore] = {}

    group_to_trajectories: "OrderedDict[Any, OrderedDict[Any, list[int]]]" = OrderedDict()
    for index, sample in enumerate(samples):
        group_key, trajectory_key = _group_and_trajectory_key(sample)
        trajectories = group_to_trajectories.setdefault(group_key, OrderedDict())
        trajectories.setdefault(trajectory_key, []).append(index)

    raw_rewards = [0.0] * len(samples)
    for trajectories in group_to_trajectories.values():
        for indices in trajectories.values():
            representative = samples[indices[0]]
            trajectory_reward = _trajectory_testcount_reward(args, representative, scale, progress_cache)
            base = trajectory_reward.shaped_reward  # passed/SCALE (>=0)
            passed = trajectory_reward.progress.passed or 0
            turns = _sample_num_steps(representative)
            if turns is None or span <= 0:
                turns_norm = 0.0
            else:
                turns_norm = _clamp((turns - l_free) / span, 0.0, 1.0)
            if passed > 0:
                shaped = base * (1.0 - alpha * turns_norm)
            else:
                shaped = -beta * turns_norm
            for index in indices:
                _annotate_sample(samples[index], trajectory_reward)
                _annotate_length_discount(samples[index], turns=turns, turns_norm=turns_norm, shaped=shaped)
                raw_rewards[index] = shaped

    if not _should_normalize(args):
        return raw_rewards, raw_rewards

    processed = [0.0] * len(samples)
    for trajectories in group_to_trajectories.values():
        trajectory_items = list(trajectories.items())
        trajectory_rewards = [raw_rewards[indices[0]] for _, indices in trajectory_items]
        trajectory_advantages = _normalize_rewards(args, trajectory_rewards)
        for (_, indices), advantage in zip(trajectory_items, trajectory_advantages, strict=True):
            for index in indices:
                processed[index] = advantage

    return raw_rewards, processed


def test_count_length_penalty_rewards(args, samples: list[Sample], **kwargs) -> tuple[list[float], list[float]]:
    """run#6 reward: test-count base MINUS a bounded output-length penalty.

    reward = passed/SCALE - lambda * clamp(output_tokens / TOKEN_BUDGET, 0, 1)

    Motivation (run#6, human-decided 2026-08-12): run#5's step-efficiency BONUS
    failed to shrink trajectories (see RUNBOOK §17) because it only rewarded the
    already-shortest fully-solved trajectories (~16%), while the 78% partial +
    zero-pass trajectories — the ones that stall and ramble to max_turns — got NO
    length signal and paid NO cost for growing. This variant fixes that by applying
    a length penalty to EVERY trajectory (the penalty term is unconditional), so a
    zero-pass trajectory that burns 38k tokens now scores NEGATIVE, directly
    penalizing the "stuck and rambling" behavior that was the actual disease.

    Why SUBTRACTION, not a ratio (passed/tokens):
      A ratio makes "1 test in 2k tokens" (5e-4) beat "5 tests in 20k tokens"
      (2.5e-4) — it rewards giving up early on hard tasks, and gives zero signal
      to distinguish a 500-token failure from a 38k-token one. Subtraction with a
      small bounded lambda keeps correctness strictly dominant (lexicographic:
      lambda 0.03 < one test's 0.05) while still ordering equally-correct
      trajectories by brevity.

    Token source: ``harbor_total_output_tokens`` (model-generated tokens only,
    excludes prompt/context) — the true output cost. Shared across a trajectory's
    samples (whole-trajectory field). SCALE, LAMBDA, BUDGET are global fixed
    constants (env-overridable) so the critic sees a consistent regression target;
    under PPO ``_should_normalize`` is False so raw shaped rewards flow straight
    into critic + GAE, and advantage whitening rescales downstream.
    """

    scale = _testcount_scale(args)
    lam = _length_penalty_lambda(args)
    budget = _length_token_budget(args)
    progress_cache: dict[str | None, ProgressScore] = {}

    group_to_trajectories: "OrderedDict[Any, OrderedDict[Any, list[int]]]" = OrderedDict()
    for index, sample in enumerate(samples):
        group_key, trajectory_key = _group_and_trajectory_key(sample)
        trajectories = group_to_trajectories.setdefault(group_key, OrderedDict())
        trajectories.setdefault(trajectory_key, []).append(index)

    raw_rewards = [0.0] * len(samples)
    for trajectories in group_to_trajectories.values():
        for indices in trajectories.values():
            representative = samples[indices[0]]
            trajectory_reward = _trajectory_testcount_reward(args, representative, scale, progress_cache)
            base = trajectory_reward.shaped_reward
            tokens = _sample_output_tokens(representative)
            if tokens is None:
                norm = 0.0
            else:
                norm = _clamp(tokens / budget, 0.0, 1.0) if budget > 0 else 0.0
            penalty = lam * norm
            shaped = base - penalty
            for index in indices:
                _annotate_sample(samples[index], trajectory_reward)
                _annotate_length_penalty(samples[index], tokens=tokens, norm=norm, penalty=penalty)
                raw_rewards[index] = shaped

    if not _should_normalize(args):
        return raw_rewards, raw_rewards

    processed = [0.0] * len(samples)
    for trajectories in group_to_trajectories.values():
        trajectory_items = list(trajectories.items())
        trajectory_rewards = [raw_rewards[indices[0]] for _, indices in trajectory_items]
        trajectory_advantages = _normalize_rewards(args, trajectory_rewards)
        for (_, indices), advantage in zip(trajectory_items, trajectory_advantages, strict=True):
            for index in indices:
                processed[index] = advantage

    return raw_rewards, processed


class ProgressScore:
    def __init__(
        self,
        score: float,
        *,
        source: str,
        passed: int | None = None,
        total: int | None = None,
        ctrf_path: str | None = None,
        error: str | None = None,
    ) -> None:
        self.score = score
        self.source = source
        self.passed = passed
        self.total = total
        self.ctrf_path = ctrf_path
        self.error = error


class TrajectoryReward:
    def __init__(self, terminal_reward: float, progress: ProgressScore, shaped_reward: float) -> None:
        self.terminal_reward = terminal_reward
        self.progress = progress
        self.shaped_reward = shaped_reward


def _trajectory_shaped_reward(
    args,
    sample: Sample,
    partial_reward_max: float,
    progress_cache: dict[str | None, ProgressScore],
) -> TrajectoryReward:
    terminal_reward = _reward_value(args, sample)
    trial_path = _trial_path(sample)
    if trial_path not in progress_cache:
        progress_cache[trial_path] = _load_progress_score(trial_path)
    progress = progress_cache[trial_path]

    if terminal_reward >= 1.0:
        shaped_reward = terminal_reward
    else:
        shaped_reward = max(terminal_reward, partial_reward_max * progress.score)
    shaped_reward = _clamp(shaped_reward, 0.0, max(1.0, terminal_reward))
    return TrajectoryReward(terminal_reward=terminal_reward, progress=progress, shaped_reward=shaped_reward)


def _trajectory_testcount_reward(
    args,
    sample: Sample,
    scale: float,
    progress_cache: dict[str | None, ProgressScore],
) -> TrajectoryReward:
    """Absolute test-count shaped reward: shaped = passed / scale.

    Full terminal success (terminal_reward >= 1.0) means every test passed, i.e.
    passed == total, so shaped = total / scale naturally (no special-casing needed
    beyond falling back to total when CTRF is present). If CTRF is missing but the
    task terminally succeeded, fall back to terminal_reward so a solved task is
    never scored 0.
    """
    terminal_reward = _reward_value(args, sample)
    trial_path = _trial_path(sample)
    if trial_path not in progress_cache:
        progress_cache[trial_path] = _load_progress_score(trial_path)
    progress = progress_cache[trial_path]

    passed = progress.passed if progress.passed is not None else 0
    if progress.source == "ctrf":
        # passed==total on full success -> shaped == total/scale (the "满分=total").
        shaped_reward = passed / scale if scale > 0 else 0.0
    else:
        # No CTRF parsed: don't zero out a genuinely solved task.
        shaped_reward = terminal_reward
    shaped_reward = max(shaped_reward, 0.0)
    return TrajectoryReward(terminal_reward=terminal_reward, progress=progress, shaped_reward=shaped_reward)


def _load_progress_score(trial_path: str | None) -> ProgressScore:
    if not trial_path:
        return ProgressScore(0.0, source="missing_trial_uri")

    ctrf_path = Path(trial_path) / "verifier" / "ctrf.json"
    if not ctrf_path.is_file():
        return ProgressScore(0.0, source="missing_ctrf", ctrf_path=str(ctrf_path))

    try:
        with ctrf_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return ProgressScore(0.0, source="invalid_ctrf", ctrf_path=str(ctrf_path), error=str(exc))

    summary = ((payload.get("results") or {}).get("summary") or {}) if isinstance(payload, dict) else {}
    passed = _int_field(summary, "passed")
    total = _effective_total(summary, passed=passed)
    if total <= 0:
        return ProgressScore(0.0, source="empty_ctrf", passed=passed, total=total, ctrf_path=str(ctrf_path))

    score = _clamp(passed / total, 0.0, 1.0)
    return ProgressScore(score, source="ctrf", passed=passed, total=total, ctrf_path=str(ctrf_path))


def _effective_total(summary: dict[str, Any], *, passed: int) -> int:
    failed = _int_field(summary, "failed")
    other = _int_field(summary, "other")
    pending = _int_field(summary, "pending")
    total = passed + failed + other + pending
    if total > 0:
        return total

    tests = _int_field(summary, "tests")
    skipped = _int_field(summary, "skipped")
    return max(0, tests - skipped)


def _int_field(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key, 0)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _trial_path(sample: Sample) -> str | None:
    metadata = getattr(sample, "metadata", None) or {}
    uri = metadata.get("harbor_trial_uri") or metadata.get("trial_uri")
    if not uri:
        return None
    if isinstance(uri, str) and uri.startswith("file://"):
        parsed = urlparse(uri)
        return unquote(parsed.path)
    return str(uri)


def _annotate_sample(sample: Sample, trajectory_reward: TrajectoryReward) -> None:
    metadata = getattr(sample, "metadata", None) or {}
    progress = trajectory_reward.progress
    progress_metadata = {
        "harbor_terminal_reward": trajectory_reward.terminal_reward,
        "harbor_progress_score": progress.score,
        "harbor_progress_reward": trajectory_reward.shaped_reward,
        "harbor_progress_source": progress.source,
        "harbor_progress_passed": progress.passed,
        "harbor_progress_total": progress.total,
    }
    metadata.update(progress_metadata)
    if progress.error:
        metadata["harbor_progress_error"] = progress.error
    sample.metadata = metadata

    train_metadata = dict(getattr(sample, "train_metadata", None) or {})
    train_metadata.update(progress_metadata)
    sample.train_metadata = train_metadata


def _annotate_family_scale(sample: Sample, *, family_id: str | None, scale: float) -> None:
    """Record which divisor this trajectory was scored with.

    Without this the per-family scaling is invisible: `rollout/rewards` alone
    cannot tell a small-family full solve from a large-family partial one. Written
    to both metadata and train_metadata, matching _annotate_sample.
    """
    scale_metadata = {
        "harbor_family_id": family_id,
        "harbor_testcount_scale_used": scale,
    }
    metadata = getattr(sample, "metadata", None) or {}
    metadata.update(scale_metadata)
    sample.metadata = metadata

    train_metadata = dict(getattr(sample, "train_metadata", None) or {})
    train_metadata.update(scale_metadata)
    sample.train_metadata = train_metadata


def _annotate_efficiency(sample: Sample, *, norm: float, bonus: float) -> None:
    """Record the step-efficiency bonus on the sample (metadata + train_metadata)."""
    efficiency_metadata = {
        "harbor_efficiency_norm": norm,
        "harbor_efficiency_bonus": bonus,
    }
    metadata = getattr(sample, "metadata", None) or {}
    metadata.update(efficiency_metadata)
    sample.metadata = metadata

    train_metadata = dict(getattr(sample, "train_metadata", None) or {})
    train_metadata.update(efficiency_metadata)
    sample.train_metadata = train_metadata


def _annotate_length_penalty(sample: Sample, *, tokens: int | None, norm: float, penalty: float) -> None:
    """Record the run#6 length penalty on the sample (metadata + train_metadata).

    Enables wandb observation of the penalty component (run#5's bonus was never
    logged as a metric, so it couldn't be observed — RUNBOOK §17.5).
    """
    length_metadata = {
        "harbor_length_output_tokens": tokens,
        "harbor_length_norm": norm,
        "harbor_length_penalty": penalty,
    }
    metadata = getattr(sample, "metadata", None) or {}
    metadata.update(length_metadata)
    sample.metadata = metadata

    train_metadata = dict(getattr(sample, "train_metadata", None) or {})
    train_metadata.update(length_metadata)
    sample.train_metadata = train_metadata


def _annotate_length_discount(sample: Sample, *, turns: int | None, turns_norm: float, shaped: float) -> None:
    """Record the run#7 multiplicative length-discount fields (metadata + train_metadata)."""
    discount_metadata = {
        "harbor_length_turns": turns,
        "harbor_length_turns_norm": turns_norm,
        "harbor_length_discounted_reward": shaped,
    }
    metadata = getattr(sample, "metadata", None) or {}
    metadata.update(discount_metadata)
    sample.metadata = metadata

    train_metadata = dict(getattr(sample, "train_metadata", None) or {})
    train_metadata.update(discount_metadata)
    sample.train_metadata = train_metadata


def _partial_reward_max(args) -> float:
    value = getattr(args, "harbor_partial_reward_max", None)
    if value is None:
        value = os.environ.get("HARBOR_PARTIAL_REWARD_MAX", DEFAULT_PARTIAL_REWARD_MAX)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = DEFAULT_PARTIAL_REWARD_MAX
    if not math.isfinite(value):
        value = DEFAULT_PARTIAL_REWARD_MAX
    return _clamp(value, 0.0, 1.0)


def _testcount_scale(args) -> float:
    value = getattr(args, "harbor_testcount_scale", None)
    if value is None:
        value = os.environ.get("HARBOR_TESTCOUNT_SCALE", DEFAULT_TESTCOUNT_SCALE)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = DEFAULT_TESTCOUNT_SCALE
    if not math.isfinite(value) or value <= 0:
        value = DEFAULT_TESTCOUNT_SCALE
    return value


def _family_scale_table(args) -> dict[str, float]:
    """Per-family CTRF divisors, keyed by the task's ``family_id``.

    WHY THIS EXISTS
    The global divisor is a single p90 over the WHOLE pool, but the check-count
    distribution is wildly family-dependent (measured on the 2087 valid-CTRF
    trials of the 20260919 run): openfoam_cfd p90=795 vs openscad_parametric_cad
    p90=1, a 795x spread. Under one global divisor a fully-solved openfoam task
    scores 795/82 = 9.7 while a fully-solved openscad task scores 1/82 = 0.012 --
    the same achievement differing 795x in reward, which both swamps the critic's
    regression target and makes `rollout/rewards` track batch family composition
    instead of policy quality.

    With a per-family divisor set to that family's own p90, a full solve scores
    ~1.0 in EVERY family, so advantages become comparable across tasks.

    Source of the table: ``harbor_family_testcount_scales`` (a YAML mapping, so it
    survives the ray-job-submit path that drops unlisted env vars) or
    ``HARBOR_FAMILY_TESTCOUNT_SCALES`` (a JSON object, for env-only callers).
    Unset -> empty dict -> every task falls back to the global scale, i.e. exactly
    the previous behaviour. Non-positive / non-finite entries are dropped rather
    than allowed to produce inf/NaN rewards.
    """
    table = getattr(args, "harbor_family_testcount_scales", None)
    if table is None:
        raw = os.environ.get("HARBOR_FAMILY_TESTCOUNT_SCALES")
        if raw:
            try:
                table = json.loads(raw)
            except (TypeError, ValueError):
                table = None
    if not isinstance(table, dict):
        return {}

    cleaned: dict[str, float] = {}
    for key, value in table.items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            cleaned[str(key)] = value
    return cleaned


def _sample_family_id(sample: Sample) -> str | None:
    """Resolve a sample's ``family_id`` for per-family reward scaling.

    Read straight from the task package's ``tests/private/manifest.json`` (the
    same file judge.py uses), located via the task dir that the data source
    recorded in ``harbor_task_entry``. Cached per task path because one rollout
    step touches only ~256 tasks but thousands of samples.
    """
    metadata = getattr(sample, "metadata", None) or {}
    entry = metadata.get("harbor_task_entry") or {}
    task = entry.get("task") or (entry.get("trial_config") or {}).get("task") or {}
    task_path = task.get("path") if isinstance(task, dict) else None
    if not task_path:
        return None
    return _family_id_for_task_path(str(task_path))


_FAMILY_ID_CACHE: dict[str, str | None] = {}


def _family_id_for_task_path(task_path: str) -> str | None:
    if task_path in _FAMILY_ID_CACHE:
        return _FAMILY_ID_CACHE[task_path]

    family_id: str | None = None
    try:
        manifest = Path(task_path) / "tests" / "private" / "manifest.json"
        if manifest.is_file():
            with manifest.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                value = payload.get("family_id")
                family_id = str(value) if value else None
    except (OSError, json.JSONDecodeError):
        family_id = None

    _FAMILY_ID_CACHE[task_path] = family_id
    return family_id


def _scale_for_sample(sample: Sample, global_scale: float, family_scales: dict[str, float]) -> tuple[float, str | None]:
    """Divisor for one trajectory: its family's p90 when known, else the global one."""
    if not family_scales:
        return global_scale, None
    family_id = _sample_family_id(sample)
    if family_id and family_id in family_scales:
        return family_scales[family_id], family_id
    return global_scale, family_id


def _efficiency_beta(args) -> float:
    """Max step-efficiency bonus (the value at ``norm=1.0``).

    Default 0.05 == 1/DEFAULT_TESTCOUNT_SCALE, i.e. one passed test's worth of
    reward, so efficiency is a clear signal that never dominates the correctness
    (passed-test-count) gap. Env-overridable via ``HARBOR_EFFICIENCY_BONUS_BETA``.
    """
    value = getattr(args, "harbor_efficiency_bonus_beta", None)
    if value is None:
        value = os.environ.get("HARBOR_EFFICIENCY_BONUS_BETA", DEFAULT_EFFICIENCY_BONUS_BETA)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = DEFAULT_EFFICIENCY_BONUS_BETA
    if not math.isfinite(value) or value < 0:
        value = DEFAULT_EFFICIENCY_BONUS_BETA
    return value


def _length_penalty_lambda(args) -> float:
    """Max length penalty (at tokens >= BUDGET). Default 0.03 < one test's 0.05, so
    correctness stays strictly lexicographic over length. Env: HARBOR_LENGTH_PENALTY_LAMBDA."""
    value = getattr(args, "harbor_length_penalty_lambda", None)
    if value is None:
        value = os.environ.get("HARBOR_LENGTH_PENALTY_LAMBDA", DEFAULT_LENGTH_PENALTY_LAMBDA)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = DEFAULT_LENGTH_PENALTY_LAMBDA
    if not math.isfinite(value) or value < 0:
        value = DEFAULT_LENGTH_PENALTY_LAMBDA
    return value


def _length_token_budget(args) -> float:
    """Output-token budget the length penalty normalizes against. Default 40000 ~ the
    observed max output length. Env: HARBOR_LENGTH_TOKEN_BUDGET."""
    value = getattr(args, "harbor_length_token_budget", None)
    if value is None:
        value = os.environ.get("HARBOR_LENGTH_TOKEN_BUDGET", DEFAULT_LENGTH_TOKEN_BUDGET)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = DEFAULT_LENGTH_TOKEN_BUDGET
    if not math.isfinite(value) or value <= 0:
        value = DEFAULT_LENGTH_TOKEN_BUDGET
    return value


def _length_discount_free_turns(args) -> float:
    """Turns at/below which no length discount applies (free zone). Env: HARBOR_LENGTH_DISCOUNT_FREE_TURNS."""
    return _env_float(
        args, "harbor_length_discount_free_turns", "HARBOR_LENGTH_DISCOUNT_FREE_TURNS",
        DEFAULT_LENGTH_DISCOUNT_FREE_TURNS, minimum=0.0,
    )


def _length_discount_full_turns(args) -> float:
    """Turns at/above which the discount is full. Env: HARBOR_LENGTH_DISCOUNT_FULL_TURNS."""
    return _env_float(
        args, "harbor_length_discount_full_turns", "HARBOR_LENGTH_DISCOUNT_FULL_TURNS",
        DEFAULT_LENGTH_DISCOUNT_FULL_TURNS, minimum=0.0,
    )


def _length_discount_alpha(args) -> float:
    """Max multiplicative discount on correctness (at full length). Env: HARBOR_LENGTH_DISCOUNT_ALPHA."""
    return _env_float(
        args, "harbor_length_discount_alpha", "HARBOR_LENGTH_DISCOUNT_ALPHA",
        DEFAULT_LENGTH_DISCOUNT_ALPHA, minimum=0.0, maximum=1.0,
    )


def _length_discount_zero_beta(args) -> float:
    """Additive penalty applied to zero-pass trajectories at full length. Env: HARBOR_LENGTH_DISCOUNT_ZERO_BETA."""
    return _env_float(
        args, "harbor_length_discount_zero_beta", "HARBOR_LENGTH_DISCOUNT_ZERO_BETA",
        DEFAULT_LENGTH_DISCOUNT_ZERO_BETA, minimum=0.0,
    )


def _env_float(args, attr: str, env: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    value = getattr(args, attr, None)
    if value is None:
        value = os.environ.get(env, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    if minimum is not None and value < minimum:
        value = default
    if maximum is not None and value > maximum:
        value = default
    return value


def _sample_num_steps(sample: Sample) -> int | None:
    """Trajectory step count (LLM-call turns). Shared across a trajectory's samples."""
    metadata = getattr(sample, "metadata", None) or {}
    value = metadata.get("harbor_num_steps", metadata.get("harbor_num_turns"))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sample_output_tokens(sample: Sample) -> int | None:
    """Trajectory total OUTPUT (model-generated) token count.

    Reads ``harbor_total_output_tokens`` — the whole-trajectory count of tokens the
    model generated (excludes prompt/context), shared across a trajectory's samples.
    This is the "output cost" the length penalty bounds. Returns None if absent so
    the caller can apply a zero penalty rather than crash.
    """
    metadata = getattr(sample, "metadata", None) or {}
    value = metadata.get("harbor_total_output_tokens")
    if value is None:
        return None
    try:
        tokens = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, tokens)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
