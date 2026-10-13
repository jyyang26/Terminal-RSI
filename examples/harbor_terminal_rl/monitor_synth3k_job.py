#!/usr/bin/env python3
"""Monitor the synth3k Ray job; on FAILURE purge all Daytona sandboxes.

Standing constraint (user): if the training task FAILS, auto-clean Daytona.
This polls `ray job status <JOB_ID>`; when the status becomes a terminal
non-SUCCEEDED state (FAILED/STOPPED), it lists and deletes every sandbox in
the org (the run is dead, so no live trial needs them). SUCCEEDED -> no purge
(the run's own cleanup_rollout handles graceful teardown). All Daytona API
calls go through the star-proxy (required egress on these nodes). The API key
is read from the env file and NEVER printed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

JOB_ID = os.environ.get("SYNTH3K_JOB_ID", "raysubmit_7mp3q8NLLsNQvCix")
RAY_ADDRESS = os.environ.get("RAY_ADDRESS", "http://127.0.0.1:8265")
ENV_FILE = "/path/to/workspace/.slime_run_env/daytona_tier4.env"
PROXY = "http://star-proxy.oa.com:3128"
POLL_SEC = 120


def _job_status() -> str:
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
        print(f"[monitor] status poll error: {e}", flush=True)
    return "UNKNOWN"


def _load_key() -> tuple[str, str]:
    key = url = ""
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith("DAYTONA_API_KEY="):
                key = line.split("=", 1)[1]
            elif line.startswith("DAYTONA_API_URL="):
                url = line.split("=", 1)[1]
    return key, url or "https://app.daytona.io/api"


def _purge_all_sandboxes() -> None:
    key, api = _load_key()
    os.environ["HTTPS_PROXY"] = PROXY
    os.environ["HTTP_PROXY"] = PROXY
    os.environ["DAYTONA_API_KEY"] = key
    os.environ["DAYTONA_API_URL"] = api
    print("[monitor] job FAILED -> purging all Daytona sandboxes", flush=True)
    # Reuse the repo's paginated filtered-cleanup script with NO filters
    # (=> matches every sandbox) and --action delete. Uses urllib + the proxy
    # from env; prints FILTERED_CANDIDATES + per-sandbox delete results.
    script = "/path/to/workspace/code/slime-v0.3-harbor/examples/harbor_terminal_rl/daytona_cleanup_filtered_sandboxes.py"
    try:
        r = subprocess.run([sys.executable, script, "--action", "delete"],
                           capture_output=True, text=True, timeout=1800)
        print("[monitor] cleanup stdout tail:\n" + "\n".join((r.stdout or "").splitlines()[-20:]), flush=True)
        if r.returncode != 0:
            print("[monitor] cleanup stderr tail:\n" + "\n".join((r.stderr or "").splitlines()[-20:]), flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[monitor] cleanup invocation error: {e}", flush=True)


def main() -> int:
    print(f"[monitor] watching job {JOB_ID} every {POLL_SEC}s", flush=True)
    unknown_streak = 0
    while True:
        st = _job_status()
        print(f"[monitor] {time.strftime('%H:%M:%S')} status={st}", flush=True)
        if st == "SUCCEEDED":
            print("[monitor] job SUCCEEDED — no purge (graceful cleanup owns teardown).", flush=True)
            return 0
        if st in ("FAILED", "STOPPED"):
            _purge_all_sandboxes()
            return 1
        if st == "UNKNOWN":
            # Dashboard/GCS unreachable usually means the Ray cluster (and thus
            # the job) is already dead. After a sustained streak, treat it as a
            # failure and purge, so a dashboard-down death still triggers cleanup.
            unknown_streak += 1
            if unknown_streak >= 5:
                print("[monitor] status UNKNOWN for 5 consecutive polls — treating as failure.", flush=True)
                _purge_all_sandboxes()
                return 1
        else:
            unknown_streak = 0
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    raise SystemExit(main())
