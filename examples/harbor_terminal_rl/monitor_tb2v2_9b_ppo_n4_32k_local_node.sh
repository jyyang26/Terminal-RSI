#!/usr/bin/env bash
set -u

# Local-node monitor for a running 32GPU 9B TB2 train-v2 PPO job. This is meant
# to run inside a tmux session on the Jizhi launcher node, avoiding local crontab
# and repeated jizhi_client calls.

RUN_TAG="${RUN_TAG:-p9bstrict_standardos9_criticlr10x_0709}"
RUN_BASE="${RUN_BASE:-/path/to/workspace/slime_harbor_qwen35_9b_tb2_train_v2_32gpu_daytona_ppo_n4_bringup_${RUN_TAG}}"
SAVE_DIR="${SAVE_DIR:-${RUN_BASE}/Qwen3.5-9B_tb2_train_v2_32gpu_daytona_ppo_n4_bringup_${RUN_TAG}}"
TRIALS_DIR="${TRIALS_DIR:-/path/to/workspace/slime_harbor_trials/tb2_train_v2_32gpu_9b_daytona_ppo_n4_bringup_${RUN_TAG}}"
ROLLOUT_DUMPS_DIR="${ROLLOUT_DUMPS_DIR:-${RUN_BASE}/rollout_dumps}"
LAUNCHER_LOG="${LAUNCHER_LOG:-/tmp/slime_tb2v2_9b_ppo_n4_${RUN_TAG}_launcher.log}"

INTERVAL_SECONDS="${INTERVAL_SECONDS:-3600}"
LOG_FILE="${LOG_FILE:-/tmp/slime_tb2v2_9b_ppo_n4_${RUN_TAG}_local_monitor.log}"
TARGET_FINAL_ITER="${TARGET_FINAL_ITER:-55}"
FAILGUARD_ENABLED="${FAILGUARD_ENABLED:-1}"
CLEAN_DAYTONA_ON_FAILURE="${CLEAN_DAYTONA_ON_FAILURE:-1}"
DAYTONA_ENV_FILE="${DAYTONA_ENV_FILE:-${HOME}/.daytona_harbor.env}"
DAYTONA_REGISTRY_PROXY_ENV="${DAYTONA_REGISTRY_PROXY_ENV:-/path/to/workspace/code/slime-v0.3-harbor/examples/harbor_terminal_rl/registry_proxy_env.sh}"
DAYTONA_SESSION_CLEANUP_SCRIPT="${DAYTONA_SESSION_CLEANUP_SCRIPT:-/path/to/workspace/code/slime-v0.3-harbor/examples/harbor_terminal_rl/daytona_cleanup_sessions_by_label.py}"
DAYTONA_SESSION_FILE="${DAYTONA_SESSION_FILE:-/tmp/slime_tb2v2_9b_ppo_n4_${RUN_TAG}_sessions.txt}"
FAILGUARD_MARKER="${FAILGUARD_MARKER:-/tmp/slime_tb2v2_9b_ppo_n4_${RUN_TAG}_daytona_cleanup.done}"
FATAL_LOG_PATTERN="${FATAL_LOG_PATTERN:-Traceback \\(most recent call last\\)|DistStoreError|OutOfMemory|CUDA out of memory|torch\\.OutOfMemoryError|Segmentation fault|SIGKILL|SIGTERM|WandbUsageError|No API key configured|ray\\.exceptions|ActorDiedError|WorkerCrashedError}"

train_alive() {
  ps -eo args | grep -E "[t]rain_async.py.*${RUN_TAG}" >/dev/null 2>&1
}

latest_iter() {
  if [ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
    cat "${SAVE_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || echo -1
  else
    echo -1
  fi
}

fatal_log_seen() {
  tail -n 5000 "${LAUNCHER_LOG}" 2>/dev/null \
    | grep -v 'HarborTrialWorker' \
    | grep -v 'tmux_session.py:639 - Sending keys' \
    | grep -E "${FATAL_LOG_PATTERN}" >/dev/null 2>&1
}

cleanup_daytona_after_failure() {
  [ "${FAILGUARD_ENABLED}" = "1" ] || return 0
  local latest
  latest="$(latest_iter)"
  if [ "${latest}" -ge "${TARGET_FINAL_ITER}" ] 2>/dev/null; then
    echo "[failguard] complete latest_iter=${latest}; no cleanup"
    return 0
  fi
  if train_alive; then
    echo "[failguard] train process alive; no cleanup"
    return 0
  fi
  if ! fatal_log_seen; then
    echo "[failguard] train not found but no fatal signature; no cleanup"
    return 0
  fi
  if [ -f "${FAILGUARD_MARKER}" ]; then
    echo "[failguard] cleanup already attempted: ${FAILGUARD_MARKER}"
    return 0
  fi

  find "${TRIALS_DIR}" -maxdepth 1 -type d -name 'slime-r*' -printf '%f\n' 2>/dev/null | sort -u >"${DAYTONA_SESSION_FILE}"
  echo "[failguard] session_file=${DAYTONA_SESSION_FILE} session_count=$(wc -l <"${DAYTONA_SESSION_FILE}" 2>/dev/null || echo 0)"
  if [ ! -s "${DAYTONA_SESSION_FILE}" ]; then
    echo "[failguard] no sessions found; skip Daytona cleanup"
    return 0
  fi

  set -a
  [ -f "${DAYTONA_ENV_FILE}" ] && . "${DAYTONA_ENV_FILE}"
  [ -f "${DAYTONA_REGISTRY_PROXY_ENV}" ] && . "${DAYTONA_REGISTRY_PROXY_ENV}"
  set +a

  python3 "${DAYTONA_SESSION_CLEANUP_SCRIPT}" --session-file "${DAYTONA_SESSION_FILE}" --action audit --progress-every 50 || true
  if [ "${CLEAN_DAYTONA_ON_FAILURE}" = "1" ]; then
    python3 "${DAYTONA_SESSION_CLEANUP_SCRIPT}" --session-file "${DAYTONA_SESSION_FILE}" --action delete --progress-every 50 || true
    date '+%F %T %Z' >"${FAILGUARD_MARKER}"
    echo "[failguard] cleanup marker written: ${FAILGUARD_MARKER}"
  fi
}

run_once() {
  {
    echo
    echo "===== $(date '+%F %T %Z') local monitor start ====="
    echo "run_tag=${RUN_TAG}"
    echo "[processes]"
    tmux ls 2>/dev/null || true
    ps -ef | grep -E 'train_async.py|RolloutManager|MegatronTrainRayActor|SGLangEngine|slime_tb2v2_9b_ppo_n4' | grep -v grep | tail -100 || true
    echo "[ray]"
    ray status 2>&1 | tail -120 || true
    echo "[progress]"
    tail -n 1600 "${LAUNCHER_LOG}" 2>&1 \
      | grep -E 'rollout_id|rollout id|RolloutManager|Harbor rollout|rollout auxiliary|Created|Trajectory dumped|update_weights|Timer|TrainLoop|global_step|critic|actor|save|checkpoint|eval_|Traceback|RuntimeError|AssertionError|OOM|OutOfMemory|FAILED|Trial .* failed' \
      | grep -v 'tmux_session.py:639 - Sending keys' \
      | tail -240 || true
    echo "[rollout dumps]"
    find "${ROLLOUT_DUMPS_DIR}" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM %f %s\n' 2>/dev/null | sort || true
    echo "[trial counts]"
    printf 'trial_dirs='; find "${TRIALS_DIR}" -maxdepth 1 -type d -name 'slime-r*' 2>/dev/null | wc -l
    printf 'trajectory_json='; find "${TRIALS_DIR}" -maxdepth 3 -path '*/agent/trajectory*.json' -type f 2>/dev/null | wc -l
    echo "[save latest]"
    find "${SAVE_DIR}" -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM %p %s\n' 2>/dev/null | sort | tail -80 || true
    echo "[failguard]"
    cleanup_daytona_after_failure || true
    echo "===== $(date '+%F %T %Z') local monitor end ====="
  } >>"${LOG_FILE}" 2>&1
}

case "${1:---once}" in
  --once)
    run_once
    ;;
  --loop)
    echo "$(date '+%F %T %Z') local monitor loop started pid=$$ interval=${INTERVAL_SECONDS}s" >>"${LOG_FILE}"
    while true; do
      run_once
      sleep "${INTERVAL_SECONDS}"
    done
    ;;
  *)
    echo "usage: $0 [--once|--loop]" >&2
    exit 2
    ;;
esac
