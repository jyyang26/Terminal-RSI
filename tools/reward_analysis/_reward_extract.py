#!/usr/bin/env python3
"""Stage 1: extract per-trajectory reward rows from both runs' rollout dumps.

For every trajectory we read the CTRF verifier summary (passed / total tests ==
validator count) and the task manifest (family_id), then recompute the EXACT
shaped reward each run's own post-process function produced:

  Run 1 (legacy):  R = passed / D1(family)                       -- test_count_progress_rewards
  Run 2 (ours):    R = 1[success] + passed / (passed + D2(family)) -- success_smoothed_progress_rewards

D1/D2 are the per-family divisor tables from each run's harbor_config_resolved.yaml
(fallback to the global harbor_testcount_scale = 82.0). Output: a single pickled
pandas DataFrame at tools/reward_analysis/_reward_rows.pkl.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd
import torch
import yaml

RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", "runs"))
OUT = Path(__file__).resolve().parent / "_reward_rows.pkl"

RUN1_DIR = RUNS_ROOT / "q38_27b_dense_ppo_256k_kxm_RESUME19_20261007_144111"
RUN2_DIR = RUNS_ROOT / "q38_27b_dense_ppo_256k_kxm_long_reward_RESUME19_20261009_160839"

N_DUMPS = 10          # latest N training dumps per run
GLOBAL_SCALE = 82.0

_ctrf_cache: dict[str, tuple] = {}
_fam_cache: dict[str, str] = {}


def load_divisors(cfg_path: Path) -> dict[str, float]:
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    table = cfg.get("harbor_family_testcount_scales") or {}
    out = {}
    for k, v in table.items():
        try:
            fv = float(v)
            if fv > 0:
                out[str(k)] = fv
        except (TypeError, ValueError):
            continue
    return out


def trial_path(md: dict) -> str | None:
    uri = md.get("harbor_trial_uri") or md.get("trial_uri")
    if not uri:
        return None
    if isinstance(uri, str) and uri.startswith("file://"):
        return unquote(urlparse(uri).path)
    return str(uri)


def read_ctrf(tp: str | None):
    """Return (passed, total, valid) replicating _load_progress_score semantics."""
    if not tp:
        return 0, 0, False
    if tp in _ctrf_cache:
        return _ctrf_cache[tp]
    ctrf = Path(tp) / "verifier" / "ctrf.json"
    res = (0, 0, False)
    if ctrf.is_file():
        try:
            payload = json.load(ctrf.open("r", encoding="utf-8"))
            summary = ((payload.get("results") or {}).get("summary") or {}) if isinstance(payload, dict) else {}

            def ifield(k):
                try:
                    return max(0, int(summary.get(k, 0)))
                except (TypeError, ValueError):
                    return 0

            passed = ifield("passed")
            total = passed + ifield("failed") + ifield("other") + ifield("pending")
            if total <= 0:
                total = max(0, ifield("tests") - ifield("skipped"))
            valid = total > 0
            res = (passed, total, valid)
        except (OSError, json.JSONDecodeError):
            res = (0, 0, False)
    _ctrf_cache[tp] = res
    return res


def read_family(md: dict) -> str:
    entry = md.get("harbor_task_entry") or {}
    task = entry.get("task") or (entry.get("trial_config") or {}).get("task") or {}
    tpath = task.get("path") if isinstance(task, dict) else None
    if not tpath:
        return "unknown"
    if tpath in _fam_cache:
        return _fam_cache[tpath]
    fam = "unknown"
    try:
        mani = Path(tpath) / "tests" / "private" / "manifest.json"
        if mani.is_file():
            payload = json.load(mani.open("r", encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("family_id"):
                fam = str(payload["family_id"])
    except (OSError, json.JSONDecodeError):
        fam = "unknown"
    _fam_cache[tpath] = fam
    return fam


def reward_run1(passed, total, valid, terminal, divisors, family):
    d = divisors.get(family, GLOBAL_SCALE)
    if valid:
        return max(passed / d, 0.0)
    return max(terminal, 0.0)


def reward_run2(passed, total, valid, terminal, divisors, family):
    d = divisors.get(family, GLOBAL_SCALE)
    p = passed if valid else 0
    success = 1.0 if terminal >= 1.0 else 0.0
    denom = p + d
    smoothed = (p / denom) if denom > 0 else 0.0
    return success + max(smoothed, 0.0)


def latest_dumps(run_dir: Path, n: int) -> list[Path]:
    files = [p for p in (run_dir / "rollout_dumps").glob("*.pt") if not p.name.startswith("eval_")]
    files.sort(key=lambda p: int(p.stem))
    return files[-n:]


def extract_run(run_label, run_dir, divisors, reward_fn):
    rows = []
    dumps = latest_dumps(run_dir, N_DUMPS)
    print(f"[{run_label}] dumps: {[p.name for p in dumps]}", flush=True)
    for dp in dumps:
        obj = torch.load(dp, map_location="cpu", weights_only=False)
        rollout_id = obj.get("rollout_id")
        samples = obj.get("samples") or []
        seen = {}
        for s in samples:
            md = s.get("metadata") or {}
            tp = trial_path(md)
            key = tp or md.get("harbor_trial_name") or id(s)
            # dedupe to one row per trajectory (reward is a trajectory-level scalar)
            resp_len = int(s.get("response_length") or 0)
            if key in seen:
                # keep max response_length across chunks
                if resp_len > seen[key]["response_length"]:
                    seen[key]["response_length"] = resp_len
                continue
            terminal = float(s.get("reward") or 0.0)
            passed, total, valid = read_ctrf(tp)
            family = read_family(md)
            r = reward_fn(passed, total, valid, terminal, divisors, family)
            seen[key] = {
                "run": run_label,
                "rollout_id": rollout_id,
                "trial": key,
                "family": family,
                "passed": passed,
                "total_validators": total,
                "valid_ctrf": valid,
                "terminal_success": 1 if terminal >= 1.0 else 0,
                "reward": r,
                "response_length": resp_len,
                "num_turns": int(md.get("harbor_num_turns") or 0),
            }
        rows.extend(seen.values())
        print(f"[{run_label}] {dp.name}: {len(samples)} samples -> {len(seen)} trajectories", flush=True)
        del obj, samples
    df = pd.DataFrame(rows)
    print(f"[{run_label}] total trajectories: {len(df)}", flush=True)
    return df


def main():
    d1 = load_divisors(RUN1_DIR / "harbor_config_resolved.yaml")
    d2 = load_divisors(RUN2_DIR / "harbor_config_resolved.yaml")
    print(f"Run1 divisor families: {len(d1)}  Run2 divisor families: {len(d2)}", flush=True)

    df1 = extract_run("Run 1 (Discrete Progress)", RUN1_DIR, d1, reward_run1)
    df2 = extract_run("Run 2 (Success-Smoothed)", RUN2_DIR, d2, reward_run2)
    df = pd.concat([df1, df2], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUT)
    print(f"\nSaved {len(df)} rows -> {OUT}", flush=True)
    print(df.groupby("run")[["reward", "total_validators", "passed"]].describe().T, flush=True)


if __name__ == "__main__":
    sys.exit(main())
