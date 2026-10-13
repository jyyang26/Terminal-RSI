#!/usr/bin/env python3
"""Cron watchdog for the synth3k standardOS288 relaunch.

Checks the Ray job status once per invocation (cron-driven, non-blocking).
On a terminal failure (FAILED / STOPPED) or a sustained UNKNOWN streak
(dashboard/GCS gone => cluster dead), it purges Daytona sandboxes whose name
starts with ``slime-`` (this run's trials only; other orgs' work untouched).

Idempotent: once a purge fires it writes a done-marker and every later
invocation is a no-op, so overlapping cron ticks never double-purge and a
SUCCEEDED/again-RUNNING flip never triggers a spurious purge.

The Daytona API key is read from the env file and NEVER printed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

JOB_ID = "raysubmit_HLxPiReEc9vpJYVV"  # 2026-07-13 FRESH from-scratch run (Qwen3.5+tmax15k strongest b384+pinnedspec); 15 nodes/120GPU excl .206 only (.219 restarted, .188 p2p-restaged from .202); standard os390 batch384 critic30x/actor2x, sandbox pinned 1CPU/2G/6G, num-rollout 100. New run (no resume) after user stopped GuEtPU resume to shrink sandbox footprint + raise batch.
RAY_ADDRESS = "http://127.0.0.1:8265"
ENV_FILE = "/path/to/workspace/.slime_run_env/daytona_relaunch_20260707.env"
PROXY = "http://star-proxy.oa.com:3128"
# >>2026-07-08 FIX: this run's Daytona sandboxes carry a UUID as their `name`; the
# "slime-" identity lives ONLY in the `harbor_session` label (e.g. slime-r12-g207-...).
# The cleanup script's --name-prefix matches the UUID `name`, so `--name-prefix slime-`
# matched ZERO sandboxes — a silent no-op on a real failure. Filter by the label that
# Harbor actually stamps on every managed sandbox instead. (Verified: --label
# harbor_managed=1 matched exactly this run's leaked sandboxes.)
CLEANUP_LABEL = "harbor_managed=1"
CLEANUP_SCRIPT = (
    "/path/to/workspace/code/slime-v0.3-harbor/"
    "examples/harbor_terminal_rl/daytona_cleanup_filtered_sandboxes.py"
)

STATE_DIR = Path("/path/to/workspace/.slime_run_env")
DONE_MARKER = STATE_DIR / f"watchdog_purged_{JOB_ID}.marker"
UNKNOWN_MARKER = STATE_DIR / f"watchdog_unknown_{JOB_ID}.count"
LOG_FILE = STATE_DIR / f"watchdog_{JOB_ID}.log"
UNKNOWN_THRESHOLD = 5  # consecutive UNKNOWN ticks => treat as dead


def log(msg: str) -> None:
    line = f"[watchdog] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def job_status() -> str:
    try:
        out = subprocess.run(
            ["ray", "job", "status", JOB_ID],
            env={**os.environ, "RAY_ADDRESS": RAY_ADDRESS},
            capture_output=True, text=True, timeout=60,
        )
        text = (out.stdout or "") + (out.stderr or "")
        for state in ("SUCCEEDED", "FAILED", "STOPPED", "RUNNING", "PENDING"):
            if state in text:
                return state
    except Exception as e:  # noqa: BLE001
        log(f"status poll error: {e}")
    return "UNKNOWN"


def load_key() -> tuple[str, str]:
    key = url = ""
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith("DAYTONA_API_KEY="):
                key = line.split("=", 1)[1]
            elif line.startswith("DAYTONA_API_URL="):
                url = line.split("=", 1)[1]
    return key, url or "https://app.daytona.io/api"


def purge_sandboxes(reason: str) -> None:
    if DONE_MARKER.exists():
        log(f"purge already done (marker exists); skipping. trigger={reason}")
        return
    key, api = load_key()
    env = {
        **os.environ,
        "HTTPS_PROXY": PROXY, "HTTP_PROXY": PROXY,
        "https_proxy": PROXY, "http_proxy": PROXY,
        "DAYTONA_API_KEY": key, "DAYTONA_API_URL": api,
    }
    log(f"TRIGGER={reason} -> purging Daytona sandboxes with label '{CLEANUP_LABEL}'")
    try:
        r = subprocess.run(
            [sys.executable, CLEANUP_SCRIPT, "--action", "delete", "--label", CLEANUP_LABEL],
            env=env, capture_output=True, text=True, timeout=1800,
        )
        log("cleanup stdout tail:\n" + "\n".join((r.stdout or "").splitlines()[-25:]))
        if r.returncode != 0:
            log("cleanup stderr tail:\n" + "\n".join((r.stderr or "").splitlines()[-25:]))
        DONE_MARKER.write_text(f"{reason} @ {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log("purge complete; done-marker written (future ticks are no-ops).")
    except Exception as e:  # noqa: BLE001
        log(f"cleanup invocation error: {e}")


def read_unknown() -> int:
    try:
        return int(UNKNOWN_MARKER.read_text().strip())
    except Exception:
        return 0


def write_unknown(n: int) -> None:
    try:
        UNKNOWN_MARKER.write_text(str(n))
    except Exception:
        pass


def main() -> int:
    if DONE_MARKER.exists():
        log("done-marker present; run already cleaned up. no-op.")
        return 0

    st = job_status()
    log(f"job {JOB_ID} status={st}")

    if st == "SUCCEEDED":
        log("SUCCEEDED — no purge (run's own cleanup_rollout owns graceful teardown).")
        return 0
    if st in ("FAILED", "STOPPED"):
        purge_sandboxes(st)
        return 1
    if st == "UNKNOWN":
        n = read_unknown() + 1
        write_unknown(n)
        log(f"UNKNOWN streak={n}/{UNKNOWN_THRESHOLD}")
        if n >= UNKNOWN_THRESHOLD:
            purge_sandboxes("UNKNOWN_STREAK")
            return 1
        return 0

    # RUNNING / PENDING -> healthy, reset UNKNOWN streak
    write_unknown(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
