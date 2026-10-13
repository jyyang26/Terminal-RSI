"""Outcome-dominant dense reward (the "final" scheme) + TB-GAE.

WHY THIS FILE EXISTS AS A SEPARATE MODULE
-----------------------------------------
`partial_reward.py` is the live, proven reward path for the running job. Editing
it in place would change the reward of an in-flight run the moment a worker
re-imports. Everything new therefore lands here, and is opted into by pointing
`CUSTOM_REWARD_POST_PROCESS_PATH` at
`slime_plugins.harbor.outcome_dense_reward.outcome_dense_rewards`.
Reverting is a one-line config change, not a git revert.

WHAT IT IMPLEMENTS
------------------
1. Outcome dominance:      B = Y + alpha * (1 - Y) * q_H      (alpha = 0.30)
2. Structure/importance weighting per check:
                           W(c) = rho(type) * iota(c) * delta**depth
3. Anti-hack gate:         q_H := 0 unless >= 1 non-trivial `leaf` check passed
4. Bounded step cost:      R = B - min(C_max, sum_t c_t)      (c_t=.001, C_max=.10)

MEASURED BASIS (run 20260919_103217, 2087 valid-CTRF trials, 82,302 check items)
-------------------------------------------------------------------------------
  * 41.7% of parsable CTRF reports have total==0 (L1 hygiene gate never opened)
  * L2 break depth: 45.7% of trials expose only scenario 1; 20.2% reach all 3
  * L3 composition: leaf 32.2% / list_elem 41.2% / keys 16.4% / length 10.2%
    pure-structure checks are 27.7% of all PASSED items and pass 97.8% of the
    time -> they are near-free points, hence rho(keys)=rho(length)=0.25
  * depth histogram peaks at depth 6 (22.7% of all items), an artifact of
    nested-array expansion, not of difficulty -> delta**depth with delta=0.90

WHAT THIS MODULE DOES *NOT* DO
------------------------------
It reads the same `verifier/ctrf.json` the old hook reads. It therefore cannot
repair the L2 first-error `break` (the denominator still depends on how deep the
candidate survived) nor re-run any test. Fixing that needs a verifier-side
change; see TB4_REWARD_TBGAE_PATCH.md section "not fixed here".
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

from slime_plugins.harbor.partial_reward import (
    ProgressScore,
    TrajectoryReward,
    _annotate_family_scale,
    _annotate_sample,
    _int_field,
    _trial_path,
)

# These four live in turn_grouped_reward, NOT partial_reward -- partial_reward
# merely re-imports them. Importing from the true source keeps this module
# working even if partial_reward's import list is ever trimmed.
from slime_plugins.harbor.turn_grouped_reward import (
    _group_and_trajectory_key,
    _normalize_rewards,
    _reward_value,
    _should_normalize,
)

# ---------------------------------------------------------------------------
# Defaults. Every one is overridable from YAML (preferred: survives the
# ray-job-submit path, which drops env vars not on RUNTIME_ENV_JSON's allowlist)
# or from the environment (for DIRECT_RUN=1 callers).
# ---------------------------------------------------------------------------
DEFAULT_ALPHA = 0.30            # failure ceiling: a failed traj can never exceed this
DEFAULT_DELTA = 0.90            # per-depth decay, kills the depth-6 expansion peak
DEFAULT_STEP_COST = 0.001       # per-turn cost
DEFAULT_COST_CAP = 0.10         # total cost ceiling -- long exploration stays viable
DEFAULT_IOTA_FLOOR = 0.10       # weight floor for a 100%-always-pass check
DEFAULT_L1_PARTIAL = 0.05       # graded credit when only reconstruct.py is missing

RHO = {"leaf": 1.0, "list_elem": 0.6, "keys": 0.25, "length": 0.25}

_SCENARIO_RE = re.compile(r"^scenario\[([^\]]+)\]\.(.*)$")
_LIST_ELEM_RE = re.compile(r"\[\d+\]$")

# Filled from `harbor_check_passrate_table` when provided; empty means "no prior",
# and every unseen check gets iota = 1.0 (per spec: do not punish the unknown).
_PASSRATE_CACHE: dict[str, dict[str, float]] | None = None


# ---------------------------------------------------------------------------
# check classification + weighting
# ---------------------------------------------------------------------------
def classify_check(name: str) -> tuple[str, int, str]:
    """-> (type, depth, field_path). Mirrors ctrf_from_judge.py's naming scheme."""
    m = _SCENARIO_RE.match(name or "")
    path = m.group(2) if m else (name or "")
    if path.endswith(".keys"):
        kind = "keys"
    elif path.endswith(".length"):
        kind = "length"
    elif _LIST_ELEM_RE.search(path):
        kind = "list_elem"
    else:
        kind = "leaf"
    depth = path.count(".") + path.count("[")
    return kind, depth, path


def _iota(passrate: float | None, floor: float) -> float:
    """Binary entropy of the check's historical pass rate.

    An always-pass (p=1) or never-pass (p=0) check carries no information about
    policy quality, so it collapses to `floor`. A coin-flip check (p=0.5) gets
    the full 1.0. `None` (never observed) gets 1.0 -- an unseen check is assumed
    informative until measured, so a cold start never silently zeroes the signal.
    """
    if passrate is None:
        return 1.0
    p = min(max(float(passrate), 0.0), 1.0)
    if p <= 0.0 or p >= 1.0:
        return floor
    h = -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)
    return max(floor, h)


def check_weight(name: str, *, passrate: float | None, delta: float, floor: float) -> tuple[float, str]:
    kind, depth, _ = classify_check(name)
    w = RHO.get(kind, 1.0) * _iota(passrate, floor) * (delta ** depth)
    return w, kind


# ---------------------------------------------------------------------------
# q_H: weighted process quality in [0, 1], with the anti-hack gate
# ---------------------------------------------------------------------------
class WeightedProgress:
    __slots__ = ("q", "passed_w", "total_w", "n_passed", "n_total",
                 "leaf_hits", "gated", "source")

    def __init__(self, q, passed_w, total_w, n_passed, n_total, leaf_hits, gated, source):
        self.q = q
        self.passed_w = passed_w
        self.total_w = total_w
        self.n_passed = n_passed
        self.n_total = n_total
        self.leaf_hits = leaf_hits
        self.gated = gated
        self.source = source


def weighted_progress_from_ctrf(
    trial_path: str | None,
    *,
    delta: float = DEFAULT_DELTA,
    floor: float = DEFAULT_IOTA_FLOOR,
    passrates: dict[str, float] | None = None,
) -> WeightedProgress:
    """Read verifier/ctrf.json and compute the weighted q_H plus the gate.

    Fail-safe by construction: any missing file / bad JSON / empty test list
    returns q=0 with a `source` naming the reason, and NEVER raises. That matters
    because this runs inside the rollout post-process, where an exception would
    take down the whole step.
    """
    if not trial_path:
        return WeightedProgress(0.0, 0.0, 0.0, 0, 0, 0, True, "missing_trial_uri")
    p = Path(trial_path) / "verifier" / "ctrf.json"
    if not p.is_file():
        return WeightedProgress(0.0, 0.0, 0.0, 0, 0, 0, True, "missing_ctrf")
    try:
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return WeightedProgress(0.0, 0.0, 0.0, 0, 0, 0, True, "invalid_ctrf")

    results = (payload.get("results") or {}) if isinstance(payload, dict) else {}
    tests = results.get("tests") or []
    if not isinstance(tests, list) or not tests:
        # No per-item list: fall back to the summary so a report that only has
        # counts still yields SOMETHING rather than a silent zero.
        summary = results.get("summary") or {}
        passed = _int_field(summary, "passed")
        total = passed + _int_field(summary, "failed") + _int_field(summary, "other") + _int_field(summary, "pending")
        if total <= 0:
            return WeightedProgress(0.0, 0.0, 0.0, 0, 0, 0, True, "empty_ctrf")
        q = min(max(passed / total, 0.0), 1.0)
        # Unweighted + ungated: we cannot see types, so we cannot prove a leaf hit.
        # Treat as gated-open but flag the source so it is auditable.
        return WeightedProgress(q, float(passed), float(total), passed, total, 0, False, "ctrf_summary_only")

    pr = passrates or {}
    passed_w = 0.0
    total_w = 0.0
    n_passed = 0
    leaf_hits = 0
    for t in tests:
        if not isinstance(t, dict):
            continue
        name = t.get("name") or ""
        w, kind = check_weight(name, passrate=pr.get(name), delta=delta, floor=floor)
        total_w += w
        if t.get("status") == "passed":
            passed_w += w
            n_passed += 1
            if kind == "leaf":
                leaf_hits += 1
    n_total = len(tests)
    if total_w <= 0.0:
        return WeightedProgress(0.0, 0.0, 0.0, n_passed, n_total, leaf_hits, True, "zero_weight")

    q = min(max(passed_w / total_w, 0.0), 1.0)
    gated = leaf_hits == 0          # anti-hack: structure-only passes earn nothing
    if gated:
        q = 0.0
    return WeightedProgress(q, passed_w, total_w, n_passed, n_total, leaf_hits, gated, "ctrf")


# ---------------------------------------------------------------------------
# the hook
# ---------------------------------------------------------------------------
def outcome_dense_rewards(args, samples: list[Sample], **kwargs) -> tuple[list[float], list[float]]:
    """R = [Y + alpha*(1-Y)*q_H] - min(C_max, turns * c_t), clamped to [0, 1].

    Contract with the existing pipeline is unchanged: returns
    ``(raw_rewards, processed_rewards)``, one float per sample, with every chunk
    of a trajectory carrying that trajectory's scalar (PPO places it on the last
    response token; see loss.py's ppo branch).
    """
    alpha = _cfg_float(args, "harbor_outcome_alpha", "HARBOR_OUTCOME_ALPHA", DEFAULT_ALPHA, lo=0.0, hi=1.0)
    delta = _cfg_float(args, "harbor_depth_decay", "HARBOR_DEPTH_DECAY", DEFAULT_DELTA, lo=0.1, hi=1.0)
    step_cost = _cfg_float(args, "harbor_step_cost", "HARBOR_STEP_COST", DEFAULT_STEP_COST, lo=0.0, hi=1.0)
    cost_cap = _cfg_float(args, "harbor_cost_cap", "HARBOR_COST_CAP", DEFAULT_COST_CAP, lo=0.0, hi=1.0)
    floor = _cfg_float(args, "harbor_iota_floor", "HARBOR_IOTA_FLOOR", DEFAULT_IOTA_FLOOR, lo=0.0, hi=1.0)
    l1_partial = _cfg_float(args, "harbor_l1_partial", "HARBOR_L1_PARTIAL", DEFAULT_L1_PARTIAL, lo=0.0, hi=1.0)
    passrates = _passrate_table(args)

    groups: "OrderedDict[Any, OrderedDict[Any, list[int]]]" = OrderedDict()
    for i, s in enumerate(samples):
        gk, tk = _group_and_trajectory_key(s)
        groups.setdefault(gk, OrderedDict()).setdefault(tk, []).append(i)

    raw = [0.0] * len(samples)
    cache: dict[str | None, WeightedProgress] = {}
    for trajectories in groups.values():
        for indices in trajectories.values():
            head = samples[indices[0]]
            tp = _trial_path(head)
            if tp not in cache:
                cache[tp] = weighted_progress_from_ctrf(
                    tp, delta=delta, floor=floor, passrates=passrates
                )
            wp = cache[tp]

            terminal = _reward_value(args, head)
            y = 1.0 if terminal >= 1.0 else 0.0

            q = wp.q
            if wp.source == "empty_ctrf" and y < 1.0:
                # L1 gate never opened. Spec asks for graded, not scorched-earth:
                # a trajectory that produced *some* workspace still gets a sliver
                # so the early exploration is not a pure-zero gradient desert.
                q = l1_partial if _produced_workspace(head) else 0.0

            base = y + alpha * (1.0 - y) * q
            turns = _turns(head)
            cost = min(cost_cap, max(0, turns) * step_cost)
            shaped = min(max(base - cost, 0.0), 1.0)

            progress = ProgressScore(
                q, source=wp.source, passed=wp.n_passed, total=wp.n_total
            )
            tr = TrajectoryReward(terminal_reward=terminal, progress=progress, shaped_reward=shaped)
            for idx in indices:
                _annotate_sample(samples[idx], tr)
                _annotate_outcome_dense(
                    samples[idx], y=y, q=q, base=base, cost=cost,
                    shaped=shaped, wp=wp, turns=turns, alpha=alpha,
                )
                # keep the old field name populated so existing dashboards and
                # the per-family audit keep working
                _annotate_family_scale(samples[idx], family_id=_family_id(head), scale=1.0)
                raw[idx] = shaped

    if not _should_normalize(args):
        return raw, raw

    processed = [0.0] * len(samples)
    for trajectories in groups.values():
        items = list(trajectories.items())
        rewards = [raw[ix[0]] for _, ix in items]
        advs = _normalize_rewards(args, rewards)
        for (_, ix), a in zip(items, advs, strict=True):
            for idx in ix:
                processed[idx] = a
    return raw, processed


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _cfg_float(args, yaml_key: str, env_key: str, default: float, *, lo: float, hi: float) -> float:
    v = getattr(args, yaml_key, None)
    if v is None:
        v = os.environ.get(env_key, default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = default
    if not math.isfinite(v):
        v = default
    return min(max(v, lo), hi)


def _passrate_table(args) -> dict[str, float]:
    """{check_name: historical pass rate} for the iota weight. Empty is fine."""
    global _PASSRATE_CACHE
    if _PASSRATE_CACHE is not None:
        return _PASSRATE_CACHE.get("table", {})
    table: dict[str, float] = {}
    src = getattr(args, "harbor_check_passrate_table", None)
    if src is None:
        src = os.environ.get("HARBOR_CHECK_PASSRATE_TABLE")
    if isinstance(src, dict):
        table = {str(k): float(v) for k, v in src.items()}
    elif isinstance(src, str) and src:
        try:
            if src.lstrip().startswith("{"):
                table = {str(k): float(v) for k, v in json.loads(src).items()}
            else:
                with open(src, "r", encoding="utf-8") as f:
                    table = {str(k): float(v) for k, v in json.load(f).items()}
        except Exception:
            table = {}
    _PASSRATE_CACHE = {"table": table}
    return table


def _md(sample: Sample) -> dict:
    return getattr(sample, "metadata", None) or {}


def _turns(sample: Sample) -> int:
    v = _md(sample).get("harbor_num_turns")
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _family_id(sample: Sample) -> str | None:
    return _md(sample).get("harbor_family_id")


def _produced_workspace(sample: Sample) -> bool:
    """Did the agent create anything at all? Used only for the graded L1 fallback.

    Conservative: with no evidence either way we return False, so the fallback
    never hands out credit it cannot justify.
    """
    tp = _trial_path(sample)
    if not tp:
        return False
    ws = Path(tp) / "artifacts" / "workspace"
    try:
        return ws.is_dir() and any(ws.iterdir())
    except OSError:
        return False


def _annotate_outcome_dense(sample: Sample, *, y, q, base, cost, shaped, wp, turns, alpha) -> None:
    """Write the audit trail. THIS is what was missing before: run #1 and #2 have
    no way to answer "which divisor / which source did this sample use", because
    the annotations were never aggregated. Emitting them here at least puts them
    in sample.metadata and train_metadata for offline scans."""
    fields = {
        "harbor_odr_outcome": y,
        "harbor_odr_q_weighted": q,
        "harbor_odr_base": base,
        "harbor_odr_cost": cost,
        "harbor_odr_shaped": shaped,
        "harbor_odr_alpha": alpha,
        "harbor_odr_passed_weight": wp.passed_w,
        "harbor_odr_total_weight": wp.total_w,
        "harbor_odr_n_passed": wp.n_passed,
        "harbor_odr_n_total": wp.n_total,
        "harbor_odr_leaf_hits": wp.leaf_hits,
        "harbor_odr_anti_hack_gated": wp.gated,
        "harbor_odr_source": wp.source,
        "harbor_odr_turns": turns,
    }
    md = getattr(sample, "metadata", None)
    if md is None:
        md = {}
        sample.metadata = md
    md.update(fields)
    tm = getattr(sample, "train_metadata", None)
    if tm is None:
        tm = {}
        sample.train_metadata = tm
    tm.update(fields)
