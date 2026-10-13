#!/usr/bin/env bash
set -u

# Lightweight monitor for the Qwen3.5-9B Harbor terminal-bench v2 smoke run.
# It does not touch keeper, Daytona sandboxes, Ray processes, or checkpoints.

export PATH="${HOME}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

SERVER_NAME="${SERVER_NAME:-self_improving_terminal_agent_rl_Qwen3_5_slime_0_3_0_sglang_0_5_12_32GPU}"
INSTANCE_ID="${INSTANCE_ID:-8B1D89BD9F17C6AC019F2004E6040B53}"
JOB_ID="${JOB_ID:-raysubmit_cZgTeZ1CpU2hdjj6}"
REMOTE_HOST_INDEX="${REMOTE_HOST_INDEX:-0}"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://29.79.9.109:8265}"

EXEC_SCRIPT="${EXEC_SCRIPT:-/path/to/workspace/code/scripts/exec_multinode.exp}"
LAUNCHER_LOG="${LAUNCHER_LOG:-/tmp/slime_full89_32gpu_9b_a1r3_b3n32_trajnorm_fa_tb2v2_newkey_20260702_094700_launcher.log}"
RUN_BASE="${RUN_BASE:-/path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_32gpu_daytona_2train2rollout_60x_retokenized_smoke_tb2v2_newkey_20260702_094700}"
ROLLOUT_DUMPS_DIR="${ROLLOUT_DUMPS_DIR:-${RUN_BASE}/rollout_dumps}"
SAVE_DIR="${SAVE_DIR:-${RUN_BASE}/Qwen3.5-9B_full89_32gpu_daytona_2train2rollout_60x_retokenized_smoke_tb2v2_newkey_20260702_094700}"

INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
LOG_FILE="${LOG_FILE:-/tmp/slime_tb2v2_newkey_monitor.log}"
NOHUP_LOG="${NOHUP_LOG:-/tmp/slime_tb2v2_newkey_monitor.nohup.log}"
PID_FILE="${PID_FILE:-/tmp/slime_tb2v2_newkey_monitor.pid}"
CHECK_LOCK="${CHECK_LOCK:-/tmp/slime_tb2v2_newkey_monitor.check.lock}"
LOOP_LOCK="${LOOP_LOCK:-/tmp/slime_tb2v2_newkey_monitor.loop.lock}"

remote0() {
  local command="$1"
  /usr/bin/timeout 240 /usr/bin/expect "$EXEC_SCRIPT" "$SERVER_NAME" "$INSTANCE_ID" "$REMOTE_HOST_INDEX" "$command"
}

run_check() {
  {
    echo
    echo "===== $(date '+%F %T %Z') monitor start ====="
    echo "job_id=${JOB_ID}"
    echo "server=${SERVER_NAME}"
    echo "instance=${INSTANCE_ID}"

    echo
    echo "[ray job status]"
    remote0 "ray job status --address '${RAY_DASHBOARD_ADDRESS}' ${JOB_ID} 2>&1 | tail -80" || true

    echo
    echo "[key progress lines]"
    remote0 "tail -n 600 '${LAUNCHER_LOG}' 2>&1 | grep -E 'RolloutManager|Harbor fully-async rollout|reached partial cutoff|Accepting partial|update_weights|Timer|TrainLoop|global_step|save|checkpoint|Traceback|RuntimeError|ModuleNotFound|ImportError|OOM|OutOfMemory|FAILED|Trial .* failed' | grep -v 'tmux_session.py:639 - Sending keys' | tail -160 || true" || true

    echo
    echo "[rollout dumps]"
    remote0 "find '${ROLLOUT_DUMPS_DIR}' -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM %f %s\n' 2>/dev/null | sort || true" || true

    echo
    echo "[save dir latest]"
    remote0 "find '${SAVE_DIR}' -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM %p %s\n' 2>/dev/null | sort | tail -60 || true" || true

    echo "===== $(date '+%F %T %Z') monitor end ====="
  } >>"$LOG_FILE" 2>&1
}

run_once_with_lock() {
  (
    /usr/bin/flock -n 9 || {
      echo "$(date '+%F %T %Z') previous monitor check still running; skip" >>"$LOG_FILE"
      exit 0
    }
    run_check
  ) 9>"$CHECK_LOCK"
}

run_loop() {
  exec 8>"$LOOP_LOCK"
  /usr/bin/flock -n 8 || {
    echo "monitor loop already running; pid_file=${PID_FILE}" >>"$NOHUP_LOG"
    exit 0
  }

  echo "$$" >"$PID_FILE"
  echo "$(date '+%F %T %Z') monitor loop started pid=$$ interval=${INTERVAL_SECONDS}s" >>"$NOHUP_LOG"

  while true; do
    "$0" --once
    sleep "$INTERVAL_SECONDS"
  done
}

case "${1:---once}" in
  --loop)
    run_loop
    ;;
  --once)
    run_once_with_lock
    ;;
  --help|-h)
    cat <<EOF
Usage:
  $0 --once       Run one monitor check and append to ${LOG_FILE}
  $0 --loop       Run checks every INTERVAL_SECONDS=${INTERVAL_SECONDS}

Useful files:
  log:       ${LOG_FILE}
  nohup log: ${NOHUP_LOG}
  pid:       ${PID_FILE}
EOF
    ;;
  *)
    echo "unknown argument: $1" >&2
    exit 2
    ;;
esac
