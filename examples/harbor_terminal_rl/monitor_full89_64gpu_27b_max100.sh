#!/bin/bash

set -euo pipefail

JOB_ID=${JOB_ID:-}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-1800}
LOG_FILE=${LOG_FILE:-/tmp/slime_full89_64gpu_27b_max100_monitor.log}
LAUNCHER_LOG=${LAUNCHER_LOG:-/tmp/slime_full89_64gpu_27b_max100_launcher.log}
BASE_FOLDER=${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_64gpu_max100_n16_5epoch_resp2048_ctx128k_actorpp8_thd}
ROLLOUT_DUMPS_DIR=${ROLLOUT_DUMPS_DIR:-${BASE_FOLDER}/rollout_dumps}
SAVE_DIR=${SAVE_DIR:-${BASE_FOLDER}/Qwen3.5-27B_terminal_bench2_full89_64gpu_max100_n16_5epoch_resp2048_ctx128k_actorpp8_thd}
TRIALS_DIR=${TRIALS_DIR:-/path/to/workspace/slime_harbor_trials/tb2_full89_64gpu_27b_max100_resp2048_ctx128k_actorpp8_thd}
MONITOR_FIND_TIMEOUT_SECONDS=${MONITOR_FIND_TIMEOUT_SECONDS:-60}

count_pattern() {
  local pattern=$1
  if [ -f "${LAUNCHER_LOG}" ]; then
    grep -c "${pattern}" "${LAUNCHER_LOG}" || true
  else
    printf '0\n'
  fi
}

count_process_pattern() {
  local pattern=$1
  pgrep -af "${pattern}" 2>/dev/null | wc -l || true
}

pid_for_pattern() {
  local pattern=$1
  pgrep -f "${pattern}" 2>/dev/null | head -n 1 || true
}

print_pid_resource() {
  local label=$1
  local pattern=$2
  local pid
  pid="$(pid_for_pattern "${pattern}")"
  if [ -z "${pid}" ] || [ ! -d "/proc/${pid}" ]; then
    printf '%s_pid=none\n' "${label}"
    return
  fi
  printf '%s_pid=%s\n' "${label}" "${pid}"
  awk -v prefix="${label}" '
    /^VmRSS:/ {print prefix "_rss_kb=" $2}
    /^VmHWM:/ {print prefix "_hwm_kb=" $2}
    /^Threads:/ {print prefix "_threads=" $2}
  ' "/proc/${pid}/status" 2>/dev/null || true
  printf '%s_fd_count=%s\n' "${label}" "$(find "/proc/${pid}/fd" -maxdepth 1 -type l 2>/dev/null | wc -l || true)"
}

count_files_with_timeout() {
  local dir=$1
  shift
  if [ ! -d "${dir}" ]; then
    printf '0\n'
    return
  fi
  timeout "${MONITOR_FIND_TIMEOUT_SECONDS}" find "${dir}" "$@" 2>/dev/null | wc -l || printf 'timeout\n'
}

latest_job_id() {
  if [ -n "${JOB_ID}" ]; then
    printf '%s\n' "${JOB_ID}"
  elif [ -f "${LAUNCHER_LOG}" ]; then
    grep -o 'raysubmit_[A-Za-z0-9]*' "${LAUNCHER_LOG}" | tail -n 1 || true
  fi
}

write_snapshot() {
  local jid
  jid="$(latest_job_id)"
  {
    printf '\n===== %s =====\n' "$(date -Iseconds)"
    if [ -n "${jid}" ]; then
      printf 'job_id=%s\n' "${jid}"
      ray job status "${jid}" || true
    else
      printf 'job_id=unknown\n'
    fi
    ray status || true
    printf 'host_resource:\n'
    printf 'process_count_ray=%s\n' "$(count_process_pattern 'ray::|raylet|gcs_server|dashboard|runtime_env')"
    printf 'process_count_apptainer_singularity=%s\n' "$(count_process_pattern 'apptainer|singularity')"
    printf 'process_count_slime_trial_worker=%s\n' "$(count_process_pattern 'TrialWorker|slime_plugins.harbor')"
    print_pid_resource ray_gcs 'gcs_server'
    print_pid_resource raylet 'raylet'
    printf 'mount_count=%s\n' "$(wc -l < /proc/mounts 2>/dev/null || true)"
    printf 'tmp_df=%s\n' "$(df -h /tmp 2>/dev/null | tail -n 1 || true)"
    printf 'socket_summary=%s\n' "$(ss -s 2>/dev/null | tr '\n' ';' || true)"
    printf 'trajectory_json=%s\n' "$(count_files_with_timeout "${TRIALS_DIR}" -name trajectory.json)"
    printf 'rollout_dumps=%s\n' "$(count_files_with_timeout "${ROLLOUT_DUMPS_DIR}" -maxdepth 1 -type f -name '*.pt')"
    printf 'checkpoint_files=%s\n' "$(count_files_with_timeout "${SAVE_DIR}" -maxdepth 3 -type f)"
    printf 'cached_sif=%s\n' "$(count_pattern 'Using cached Singularity image')"
    printf 'fastapi_ready=%s\n' "$(count_pattern 'Slime Singularity FastAPI server is ready')"
    printf 'no_valid_json=%s\n' "$(count_pattern 'No valid JSON object found')"
    printf 'extra_text_json=%s\n' "$(count_pattern 'Extra text detected')"
    printf 'traceback=%s\n' "$(count_pattern 'Traceback')"
    printf 'failed_lines=%s\n' "$(count_pattern 'failed')"
    printf 'timeout_lines=%s\n' "$(count_pattern 'timeout')"
    printf 'recent_events:\n'
    if [ -f "${LAUNCHER_LOG}" ]; then
      grep -E 'raysubmit_|rollout|harbor/num|reward_mean|saving checkpoint|successfully saved checkpoint|actor train|Using cached Singularity image|Slime Singularity FastAPI server is ready|No valid JSON object|Extra text detected|Traceback|failed|timeout|OutOfMemory|OOM' "${LAUNCHER_LOG}" | tail -n 120 || true
    fi
  } >> "${LOG_FILE}" 2>&1
}

: > "${LOG_FILE}"
while true; do
  write_snapshot
  sleep "${INTERVAL_SECONDS}"
done
