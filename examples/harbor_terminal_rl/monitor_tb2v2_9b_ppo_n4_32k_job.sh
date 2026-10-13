#!/usr/bin/env bash
set -u

# Monitor for the Qwen3.5-9B TB2 train-v2 PPO n=4 32k run.
# It reads remote status/logs/artifacts. If the launcher has clearly failed,
# it cleans Daytona only for exact Harbor sessions created by this run.

export PATH="${HOME}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

SERVER_NAME="${SERVER_NAME:-self_improving_terminal_agent_rl_Qwen3_5_slime_0_3_0_sglang_0_5_12_32GPU}"
INSTANCE_ID="${INSTANCE_ID:-8B1D89BD9F17C6AC019F2004E6040B53}"
REMOTE_HOST_INDEX="${REMOTE_HOST_INDEX:-0}"
EXEC_SCRIPT="${EXEC_SCRIPT:-/path/to/workspace/code/scripts/exec_multinode.exp}"

RUN_TAG="${RUN_TAG:-p9bstrict_0704_1030}"
RUN_BASE="${RUN_BASE:-/path/to/workspace/slime_harbor_qwen35_9b_tb2_train_v2_32gpu_daytona_ppo_n4_bringup_${RUN_TAG}}"
SAVE_DIR="${SAVE_DIR:-${RUN_BASE}/Qwen3.5-9B_tb2_train_v2_32gpu_daytona_ppo_n4_bringup_${RUN_TAG}}"
TRIALS_DIR="${TRIALS_DIR:-/path/to/workspace/slime_harbor_trials/tb2_train_v2_32gpu_9b_daytona_ppo_n4_bringup_${RUN_TAG}}"
ROLLOUT_DUMPS_DIR="${ROLLOUT_DUMPS_DIR:-${RUN_BASE}/rollout_dumps}"
LAUNCHER_LOG="${LAUNCHER_LOG:-/tmp/slime_tb2v2_9b_ppo_n4_${RUN_TAG}_launcher.log}"

INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
LOG_FILE="${LOG_FILE:-/tmp/slime_tb2v2_9b_ppo_n4_32k_monitor.log}"
NOHUP_LOG="${NOHUP_LOG:-/tmp/slime_tb2v2_9b_ppo_n4_32k_monitor.nohup.log}"
PID_FILE="${PID_FILE:-/tmp/slime_tb2v2_9b_ppo_n4_32k_monitor.pid}"
CHECK_LOCK="${CHECK_LOCK:-/tmp/slime_tb2v2_9b_ppo_n4_32k_monitor.check.lock}"
LOOP_LOCK="${LOOP_LOCK:-/tmp/slime_tb2v2_9b_ppo_n4_32k_monitor.loop.lock}"

TARGET_FINAL_ITER="${TARGET_FINAL_ITER:-55}"
FAILGUARD_ENABLED="${FAILGUARD_ENABLED:-1}"
CLEAN_DAYTONA_ON_FAILURE="${CLEAN_DAYTONA_ON_FAILURE:-1}"
DAYTONA_ENV_FILE="${DAYTONA_ENV_FILE:-${HOME}/.daytona_harbor.env}"
DAYTONA_REGISTRY_PROXY_ENV="${DAYTONA_REGISTRY_PROXY_ENV:-/path/to/workspace/code/slime-v0.3-harbor/examples/harbor_terminal_rl/registry_proxy_env.sh}"
DAYTONA_SESSION_CLEANUP_SCRIPT="${DAYTONA_SESSION_CLEANUP_SCRIPT:-/path/to/workspace/code/slime-v0.3-harbor/examples/harbor_terminal_rl/daytona_cleanup_sessions_by_label.py}"
DAYTONA_CLEAN_CREATED_AFTER="${DAYTONA_CLEAN_CREATED_AFTER:-}"
DAYTONA_SESSION_FILE="${DAYTONA_SESSION_FILE:-/tmp/slime_tb2v2_9b_ppo_n4_${RUN_TAG}_sessions.txt}"
FAILGUARD_MARKER="${FAILGUARD_MARKER:-/tmp/slime_tb2v2_9b_ppo_n4_${RUN_TAG}_daytona_cleanup.done}"
FATAL_LOG_PATTERN="${FATAL_LOG_PATTERN:-Traceback \\(most recent call last\\)|DistStoreError|OutOfMemory|CUDA out of memory|torch\\.OutOfMemoryError|Segmentation fault|SIGKILL|SIGTERM|WandbUsageError|No API key configured|ray\\.exceptions|ActorDiedError|WorkerCrashedError}"

remote0() {
  local command="$1"
  /usr/bin/timeout 240 /usr/bin/expect "$EXEC_SCRIPT" "$SERVER_NAME" "$INSTANCE_ID" "$REMOTE_HOST_INDEX" "$command"
}

status_value() {
  local blob="$1"
  local key="$2"
  printf '%s\n' "$blob" | tr -d '\r' | sed -n "s/^${key}=//p" | tail -1
}

remote_failure_status() {
  remote0 "if ps -eo args | grep -E '[t]rain_async.py.*${RUN_TAG}' >/dev/null 2>&1; then echo TRAIN_ALIVE=1; else echo TRAIN_ALIVE=0; fi; latest=-1; if [ -f '${SAVE_DIR}/latest_checkpointed_iteration.txt' ]; then latest=\$(cat '${SAVE_DIR}/latest_checkpointed_iteration.txt' 2>/dev/null || echo -1); fi; echo LATEST_ITER=\$latest; if [ \"\$latest\" -ge '${TARGET_FINAL_ITER}' ] 2>/dev/null; then echo COMPLETE=1; else echo COMPLETE=0; fi; if tail -n 5000 '${LAUNCHER_LOG}' 2>/dev/null | grep -v 'HarborTrialWorker' | grep -v 'tmux_session.py:639 - Sending keys' | grep -E '${FATAL_LOG_PATTERN}' >/dev/null 2>&1; then echo FATAL_LOG=1; else echo FATAL_LOG=0; fi"
}

cleanup_daytona_after_failure() {
  [ "${FAILGUARD_ENABLED}" = "1" ] || {
    echo "[failguard] disabled"
    return 0
  }

  local status
  status="$(remote_failure_status 2>&1 || true)"
  printf '%s\n' "$status"

  local train_alive
  local latest_iter
  local complete
  local fatal_log
  train_alive="$(status_value "$status" TRAIN_ALIVE)"
  latest_iter="$(status_value "$status" LATEST_ITER)"
  complete="$(status_value "$status" COMPLETE)"
  fatal_log="$(status_value "$status" FATAL_LOG)"

  if [ "$complete" = "1" ]; then
    echo "[failguard] training appears complete at latest_iter=${latest_iter}; no Daytona cleanup"
    return 0
  fi
  if [ "$train_alive" = "1" ]; then
    echo "[failguard] train process alive; no Daytona cleanup"
    return 0
  fi
  if [ "$fatal_log" != "1" ]; then
    echo "[failguard] train process not found, but no fatal signature yet; no Daytona cleanup"
    return 0
  fi
  if [ -f "$FAILGUARD_MARKER" ]; then
    echo "[failguard] cleanup already attempted: ${FAILGUARD_MARKER}"
    return 0
  fi

  local created_before
  created_before="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "[failguard] failure detected; cleaning Daytona by exact harbor_session list"

  local cleanup_prefix
  cleanup_prefix="SESSION_FILE='${DAYTONA_SESSION_FILE}'; TRIALS_DIR='${TRIALS_DIR}'; find \"\$TRIALS_DIR\" -maxdepth 1 -type d -name 'slime-r*' -printf '%f\n' 2>/dev/null | sort -u > \"\$SESSION_FILE\"; echo session_file=\"\$SESSION_FILE\"; echo session_count=\$(wc -l < \"\$SESSION_FILE\" 2>/dev/null || echo 0); if [ ! -s \"\$SESSION_FILE\" ]; then echo '[failguard] no sessions found; skip Daytona cleanup'; exit 0; fi; set -a; if [ -f '${DAYTONA_ENV_FILE}' ]; then . '${DAYTONA_ENV_FILE}'; else echo 'missing_daytona_env=${DAYTONA_ENV_FILE}'; exit 2; fi; if [ -f '${DAYTONA_REGISTRY_PROXY_ENV}' ]; then . '${DAYTONA_REGISTRY_PROXY_ENV}'; fi; set +a; python3 '${DAYTONA_SESSION_CLEANUP_SCRIPT}' --session-file \"\$SESSION_FILE\" --created-before '${created_before}' ${DAYTONA_CLEAN_CREATED_AFTER:+--created-after '${DAYTONA_CLEAN_CREATED_AFTER}'} --progress-every 50"
  remote0 "${cleanup_prefix} --action audit" || true

  if [ "${CLEAN_DAYTONA_ON_FAILURE}" = "1" ]; then
    remote0 "${cleanup_prefix} --action delete" || true
    date '+%F %T %Z' >"$FAILGUARD_MARKER"
    echo "[failguard] cleanup marker written: ${FAILGUARD_MARKER}"
  else
    echo "[failguard] CLEAN_DAYTONA_ON_FAILURE=${CLEAN_DAYTONA_ON_FAILURE}; audit only"
  fi
}

run_check() {
  {
    echo
    echo "===== $(date '+%F %T %Z') monitor start ====="
    echo "run_tag=${RUN_TAG}"
    echo "server=${SERVER_NAME}"
    echo "instance=${INSTANCE_ID}"

    echo
    echo "[remote process status]"
    remote0 "tmux ls 2>/dev/null || true; ps -ef | grep -E 'train_async.py|RolloutManager|MegatronTrainRayActor|SGLangEngine|slime_tb2v2_9b_ppo_n4' | grep -v grep | tail -80 || true" || true

    echo
    echo "[ray status]"
    remote0 "ray status 2>&1 | tail -120 || true" || true

    echo
    echo "[key progress lines]"
    remote0 "tail -n 1200 '${LAUNCHER_LOG}' 2>&1 | grep -E 'rollout_id|rollout id|RolloutManager|Harbor fully-async|rollout auxiliary|Created|Trajectory dumped|update_weights|Timer|TrainLoop|global_step|critic|actor|save|checkpoint|eval_|Traceback|RuntimeError|AssertionError|OOM|OutOfMemory|FAILED|Trial .* failed' | grep -v 'tmux_session.py:639 - Sending keys' | tail -220 || true" || true

    echo
    echo "[rollout dumps]"
    remote0 "find '${ROLLOUT_DUMPS_DIR}' -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM %f %s\n' 2>/dev/null | sort || true" || true

    echo
    echo "[trial artifact counts]"
    remote0 "printf 'trial_dirs='; find '${TRIALS_DIR}' -maxdepth 1 -type d -name 'slime-r*' 2>/dev/null | wc -l; printf 'trajectory_json='; find '${TRIALS_DIR}' -maxdepth 3 -path '*/agent/trajectory*.json' -type f 2>/dev/null | wc -l" || true

    echo
    echo "[save dir latest]"
    remote0 "find '${SAVE_DIR}' -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM %p %s\n' 2>/dev/null | sort | tail -80 || true" || true

    echo
    echo "[failguard]"
    cleanup_daytona_after_failure || true

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
