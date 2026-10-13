"""Stage-S2 reward: bounded outcome-dominant CTRF score with a [p75, p90] divisor.

WHY THIS FILE IS SEPARATE FROM ``partial_reward.py``
----------------------------------------------------
``partial_reward.test_count_progress_rewards`` is the hook a LIVE run resolves by
dotted path at startup. Editing it in place would change the reward of any run
already in flight (and of every historical recipe that names it), so this stage
ships as a PARALLEL module selected by ``harbor_reward_mode`` /
``HARBOR_REWARD_MODE``. The legacy path is left byte-identical.

WHAT CHANGES vs THE LEGACY p90 PATH
-----------------------------------
Legacy: ``R = passed / SCALE(family)``, unbounded above, SCALE = family p90.
Two measured failure modes:

  * SMALL families become a shortcut. ``openscad_parametric_cad`` has p90 = 1, so
    passing a single check scored 1.0 -- the same as a 796-check openfoam solve.
  * LARGE families lose intermediate gradient. With one global divisor, openfoam
    partial progress was compressed into a narrow band near 0 while its full
    solve scored 9.68, so the batch mean tracked family composition rather than
    policy quality.

Stage S2 replaces this with a bounded, outcome-dominant score:

    S(f)   = max(S_min, beta * p75(f) + (1 - beta) * p90(f))
    q_H    = min(1, sum_{c in passed} W(c) / S(f))
    W(c)   = rho(type(c)) * iota(c) * delta ** depth(c)
    B(tau) = Y + alpha * (1 - Y) * AntiHack * q_H

``Y`` is the environment's own binary verdict (``reward.txt``), so a solved task
always scores exactly 1.0 and an unsolved one is capped at ``alpha`` = 0.30.
That 0.70 gap is what makes "partial work cannot outrank a real solve" hold by
construction rather than by tuning.

MEASURED INPUTS (5418 valid-CTRF trials, runs PROD3EP_120G + PROD3EP_RUN2)
--------------------------------------------------------------------------
Pass rate by check type -- this is what ``rho``/``iota`` exist to suppress::

    list_elem  n=7022  pass_rate=0.9769
    leaf       n=3358  pass_rate=0.9384
    keys       n=1606  pass_rate=0.9950   <- structural, ~always passes
    length     n=1381  pass_rate=0.9884   <- structural, ~always passes

Depth distribution: 44.0% at depth 4, 11.1% at depth >= 6 -- recursive expansion
of nested arrays is the combinatorial-inflation source ``delta`` damps.

HONEST SCOPE
------------
The weights ``rho``/``delta``/``alpha``/``beta``/``S_min`` are a DESIGN, informed
by the pass-rate and depth measurements above but not themselves validated
against a success-rate outcome. They are all overridable so a sweep can be run
without touching this file.
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
    _annotate_family_scale,
    _annotate_sample,
    _family_id_for_task_path,
    _int_field,
    _trial_path,
)
from slime_plugins.harbor.turn_grouped_reward import (
    _group_and_trajectory_key,
    _normalize_rewards,
    _reward_value,
    _should_normalize,
)

# --- S(f): divisor interpolation -----------------------------------------------------
# beta weights p75 against p90. beta=0.5 is the midpoint: it keeps the typical-sample
# scale (p75) from being swamped by the long tail (p90) while still granting large
# families a reachable denominator.
DEFAULT_INTERP_BETA = 0.50
# Anti-gaming floor. Families whose p90 is 1-5 checks (openscad_parametric_cad p90=1,
# astronomy_fits_photometry p90=5, point_cloud_lidar p90=15) would otherwise let a
# 1-2 check pass saturate q_H at 1.0. 6.0 is the smallest floor that forces a
# multi-check pass in every family we measured.
DEFAULT_MIN_SCALE = 6.0
# Fallback when a family is absent from the table: the GLOBAL p75/p90 midpoint over
# the same 5418 trials (p75=31, p90=102) -> 66.5.
DEFAULT_GLOBAL_SCALE = 66.5

# --- B(tau): outcome dominance -------------------------------------------------------
# Ceiling for an UNSOLVED trajectory. A solve is 1.0, so the success/failure gap is
# 1 - alpha = 0.70 regardless of how much partial credit accrues.
DEFAULT_ALPHA = 0.30

# --- W(c): structural weighting ------------------------------------------------------
# rho by node type. keys/length are pure schema assertions that passed 99.5%/98.8% of
# the time -- counting them at parity with a real value comparison is exactly the
# "build an empty dict, collect credit" surface.
DEFAULT_RHO = {"leaf": 1.0, "list_elem": 0.6, "keys": 0.25, "length": 0.25}
# delta ** depth damps nested-array expansion (11.1% of checks sit at depth >= 6).
DEFAULT_DELTA = 0.90
# iota floor for a check that has NEVER failed in the observed prior.
DEFAULT_IOTA_FLOOR = 0.10

_TYPE_KEYS = ("leaf", "list_elem", "keys", "length")
_LIST_INDEX_RE = re.compile(r"\[\d+\]")


# =====================================================================================
# public hook
# =====================================================================================
def interpolated_ctrf_rewards(args, samples: list[Sample], **kwargs) -> tuple[list[float], list[float]]:
    """Stage-S2 reward hook. Signature matches ``--custom-reward-post-process-path``.

    Returns ``(raw, processed)``. Under PPO ``_should_normalize`` is False, so both
    are the same list -- identical convention to the legacy hook.
    """
    cfg = _S2Config.from_args(args)
    score_cache: dict[str | None, _CtrfScore] = {}

    group_to_trajectories: OrderedDict[Any, OrderedDict[Any, list[int]]] = OrderedDict()
    for index, sample in enumerate(samples):
        group_key, trajectory_key = _group_and_trajectory_key(sample)
        trajectories = group_to_trajectories.setdefault(group_key, OrderedDict())
        trajectories.setdefault(trajectory_key, []).append(index)

    raw_rewards = [0.0] * len(samples)
    for trajectories in group_to_trajectories.values():
        for indices in trajectories.values():
            head = samples[indices[0]]
            reward, progress, scale, family_id = _trajectory_reward(args, head, cfg, score_cache)
            for index in indices:
                _annotate_sample(samples[index], _TrajectoryRewardView(progress, reward))
                _annotate_family_scale(samples[index], family_id=family_id, scale=scale)
                _annotate_s2(samples[index], progress)
                raw_rewards[index] = reward

    if not _should_normalize(args):
        return raw_rewards, raw_rewards

    processed = [0.0] * len(samples)
    for trajectories in group_to_trajectories.values():
        items = list(trajectories.items())
        rewards = [raw_rewards[idx[0]] for _, idx in items]
        for (_, indices), value in zip(items, _normalize_rewards(args, rewards), strict=False):
            for index in indices:
                processed[index] = value
    return processed, raw_rewards


# =====================================================================================
# config
# =====================================================================================
class _S2Config:
    """Resolved S2 knobs. YAML key first, then env var, then measured default.

    YAML is the load-bearing path: ``HARBOR_*`` env vars are NOT in the inner
    launcher's RUNTIME_ENV_JSON whitelist, so under ``ray job submit`` an exported
    value never reaches the Ray workers and would be silently ignored.
    """

    __slots__ = ("alpha", "beta", "min_scale", "global_scale", "delta", "rho", "iota_floor", "family_stats", "pass_prior")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])

    @classmethod
    def from_args(cls, args) -> _S2Config:
        rho = dict(DEFAULT_RHO)
        override = _dict_setting(args, "harbor_s2_rho", "HARBOR_S2_RHO")
        for key in _TYPE_KEYS:
            if key in override:
                value = _as_float(override[key])
                if value is not None and value >= 0:
                    rho[key] = value
        return cls(
            alpha=_float_setting(args, "harbor_s2_alpha", "HARBOR_S2_ALPHA", DEFAULT_ALPHA, 0.0, 1.0),
            beta=_float_setting(args, "harbor_s2_beta", "HARBOR_S2_BETA", DEFAULT_INTERP_BETA, 0.0, 1.0),
            min_scale=_float_setting(args, "harbor_s2_min_scale", "HARBOR_S2_MIN_SCALE", DEFAULT_MIN_SCALE, 1e-6, 1e9),
            global_scale=_float_setting(
                args, "harbor_s2_global_scale", "HARBOR_S2_GLOBAL_SCALE", DEFAULT_GLOBAL_SCALE, 1e-6, 1e9
            ),
            delta=_float_setting(args, "harbor_s2_delta", "HARBOR_S2_DELTA", DEFAULT_DELTA, 1e-6, 1.0),
            iota_floor=_float_setting(
                args, "harbor_s2_iota_floor", "HARBOR_S2_IOTA_FLOOR", DEFAULT_IOTA_FLOOR, 0.0, 1.0
            ),
            rho=rho,
            family_stats=_dict_setting(args, "harbor_family_ctrf_stats", "HARBOR_FAMILY_CTRF_STATS"),
            pass_prior=_dict_setting(args, "harbor_s2_pass_prior", "HARBOR_S2_PASS_PRIOR"),
        )

    def scale_for_family(self, family_id: str | None) -> float:
        """S(f) = max(S_min, beta*p75 + (1-beta)*p90); global midpoint when unknown."""
        stats = self.family_stats.get(family_id) if family_id else None
        if not isinstance(stats, dict):
            return max(self.min_scale, self.global_scale)
        p75 = _as_float(stats.get("p75"))
        p90 = _as_float(stats.get("p90"))
        if p75 is None and p90 is None:
            return max(self.min_scale, self.global_scale)
        if p75 is None:
            p75 = p90
        if p90 is None:
            p90 = p75
        interp = self.beta * float(p75) + (1.0 - self.beta) * float(p90)
        if not math.isfinite(interp) or interp <= 0:
            return max(self.min_scale, self.global_scale)
        return max(self.min_scale, interp)

    def weight_for_check(self, name: str) -> tuple[float, str]:
        """W(c) = rho(type) * iota(c) * delta**depth(c). Returns (weight, type)."""
        kind = _classify_check(name)
        depth = _check_depth(name)
        iota = self._iota(name)
        return self.rho.get(kind, 1.0) * iota * (self.delta**depth), kind

    def _iota(self, name: str) -> float:
        """Information-content weight from an observed pass-rate prior.

        iota = 1 - p_c, floored at ``iota_floor`` so an always-passing check still
        carries a token weight instead of vanishing. No prior -> 1.0 (neutral), so
        the default configuration is pure rho * delta**depth.
        """
        if not self.pass_prior:
            return 1.0
        p = _as_float(self.pass_prior.get(_check_path(name)))
        if p is None or not math.isfinite(p):
            return 1.0
        return max(self.iota_floor, 1.0 - min(max(p, 0.0), 1.0))


# =====================================================================================
# CTRF parsing + scoring
# =====================================================================================
class _CtrfScore:
    __slots__ = ("q_h", "anti_hack", "weighted_passed", "scale", "passed", "total", "source", "by_type", "error")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def _load_ctrf_score(trial_path: str | None, cfg: _S2Config, scale: float) -> _CtrfScore:
    """Parse ctrf.json into q_H plus the anti-hack gate. Never raises."""
    empty = dict(q_h=0.0, anti_hack=False, weighted_passed=0.0, scale=scale, passed=0, total=0, by_type={}, error=None)
    if not trial_path:
        return _CtrfScore(source="missing_trial_uri", **empty)

    path = Path(trial_path) / "verifier" / "ctrf.json"
    if not path.is_file():
        return _CtrfScore(source="missing_ctrf", **empty)
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return _CtrfScore(source="invalid_ctrf", **{**empty, "error": str(exc)})

    results = (payload.get("results") or {}) if isinstance(payload, dict) else {}
    summary = results.get("summary") or {}
    passed = _int_field(summary, "passed")
    total = passed + _int_field(summary, "failed") + _int_field(summary, "other") + _int_field(summary, "pending")
    if total <= 0:
        return _CtrfScore(source="empty_ctrf", **{**empty, "passed": passed, "total": total})

    tests = results.get("tests")
    if not isinstance(tests, list) or not tests:
        # Summary is valid but the per-check list is absent: fall back to the
        # unweighted count so a real partial solve is not silently zeroed.
        q_h = min(1.0, passed / scale) if scale > 0 else 0.0
        return _CtrfScore(
            source="ctrf_summary_only",
            q_h=q_h,
            anti_hack=passed > 0,
            weighted_passed=float(passed),
            scale=scale,
            passed=passed,
            total=total,
            by_type={},
            error=None,
        )

    weighted = 0.0
    by_type: dict[str, float] = {}
    has_leaf = False
    for test in tests:
        if not isinstance(test, dict) or test.get("status") != "passed":
            continue
        name = str(test.get("name", ""))
        weight, kind = cfg.weight_for_check(name)
        weighted += weight
        by_type[kind] = by_type.get(kind, 0.0) + weight
        if kind in ("leaf", "list_elem"):
            has_leaf = True

    q_h = min(1.0, weighted / scale) if scale > 0 else 0.0
    return _CtrfScore(
        source="ctrf",
        q_h=q_h,
        anti_hack=has_leaf,
        weighted_passed=weighted,
        scale=scale,
        passed=passed,
        total=total,
        by_type=by_type,
        error=None,
    )


def _trajectory_reward(args, sample: Sample, cfg: _S2Config, cache: dict):
    """B(tau) = Y + alpha*(1-Y)*AntiHack*q_H, clamped to [0, 1]."""
    terminal = _reward_value(args, sample)
    y = 1.0 if terminal >= 1.0 else 0.0

    family_id = _sample_family(sample)
    scale = cfg.scale_for_family(family_id)

    trial_path = _trial_path(sample)
    key = (trial_path, family_id)
    if key not in cache:
        cache[key] = _load_ctrf_score(trial_path, cfg, scale)
    score = cache[key]

    gate = 1.0 if score.anti_hack else 0.0
    reward = y + cfg.alpha * (1.0 - y) * gate * score.q_h
    reward = min(1.0, max(0.0, reward))

    progress = ProgressScore(
        score.q_h,
        source=score.source,
        passed=score.passed,
        total=score.total,
        ctrf_path=None,
        error=score.error,
    )
    return reward, progress, scale, family_id


class _TrajectoryRewardView:
    """Shape adapter so the legacy ``_annotate_sample`` can be reused verbatim."""

    __slots__ = ("terminal_reward", "progress", "shaped_reward")

    def __init__(self, progress: ProgressScore, shaped: float):
        self.progress = progress
        self.shaped_reward = shaped
        self.terminal_reward = 1.0 if shaped >= 1.0 else 0.0


def _annotate_s2(sample: Sample, progress: ProgressScore) -> None:
    """Expose q_H and the reward mode so the S2 path is visible in trial dumps."""
    meta = {"harbor_s2_q_h": progress.score, "harbor_reward_mode": "interpolated_ctrf"}
    m = getattr(sample, "metadata", None) or {}
    m.update(meta)
    sample.metadata = m
    tm = dict(getattr(sample, "train_metadata", None) or {})
    tm.update(meta)
    sample.train_metadata = tm


# =====================================================================================
# helpers
# =====================================================================================
def _sample_family(sample: Sample) -> str | None:
    metadata = getattr(sample, "metadata", None) or {}
    entry = metadata.get("harbor_task_entry") or {}
    task = entry.get("task") or (entry.get("trial_config") or {}).get("task") or {}
    path = task.get("path") if isinstance(task, dict) else None
    return _family_id_for_task_path(str(path)) if path else None


def _check_body(name: str) -> str:
    """Strip the ``scenario[<sid>].`` prefix and a leading ``$.`` root marker."""
    body = name.split("].", 1)[-1] if "]." in name else name
    return body[2:] if body.startswith("$.") else body


def _check_path(name: str) -> str:
    return _check_body(name)


def _classify_check(name: str) -> str:
    """leaf | list_elem | keys | length -- matches ctrf_from_judge's naming."""
    body = _check_body(name)
    if body.endswith(".keys") or body == "keys":
        return "keys"
    if body.endswith(".length") or body == "length":
        return "length"
    if _LIST_INDEX_RE.search(body):
        return "list_elem"
    return "leaf"


def _check_depth(name: str) -> int:
    body = _check_body(name)
    return body.count(".") + body.count("[")


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _float_setting(args, yaml_key: str, env_key: str, default: float, lo: float, hi: float) -> float:
    raw = getattr(args, yaml_key, None)
    if raw is None:
        raw = os.environ.get(env_key)
    value = _as_float(raw)
    if value is None:
        return default
    return min(hi, max(lo, value))


def _dict_setting(args, yaml_key: str, env_key: str) -> dict:
    raw = getattr(args, yaml_key, None)
    if raw is None:
        env = os.environ.get(env_key)
        if env:
            try:
                raw = json.loads(env)
            except (TypeError, ValueError):
                raw = None
    return raw if isinstance(raw, dict) else {}
