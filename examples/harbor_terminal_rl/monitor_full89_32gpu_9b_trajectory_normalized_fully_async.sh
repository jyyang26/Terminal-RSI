#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

JOB_ID=${JOB_ID:-}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-300}
LOG_FILE=${LOG_FILE:-/tmp/slime_full89_32gpu_9b_trajnorm_fa_monitor.log}
LAUNCHER_LOG=${LAUNCHER_LOG:-/tmp/slime_full89_32gpu_9b_trajnorm_fa_launcher.log}
TIMING_SUMMARY=${TIMING_SUMMARY:-/tmp/slime_full89_32gpu_9b_trajnorm_fa_timing_summary.txt}
TIMING_JSON=${TIMING_JSON:-/tmp/slime_full89_32gpu_9b_trajnorm_fa_timing_summary.json}
BASE_FOLDER=${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_32gpu_actor2_rollout2_fully_async_trajectory_normalized_timeout3600}
SAVE_SUBDIR=${SAVE_SUBDIR:-Qwen3.5-9B_full89_32gpu_actor2_rollout2_fully_async_trajectory_normalized_timeout3600}
SAVE_DIR=${SAVE_DIR:-${BASE_FOLDER%/}/${SAVE_SUBDIR}}
ROLLOUT_DUMPS_DIR=${ROLLOUT_DUMPS_DIR:-${BASE_FOLDER%/}/rollout_dumps}
TRIALS_DIR=${TRIALS_DIR:-/path/to/workspace/slime_harbor_trials/tb2_full89_32gpu_9b_max100_actor2_rollout2_resp4096_ctx128k_timeout3600_trajectory_normalized}
MONITOR_FIND_TIMEOUT_SECONDS=${MONITOR_FIND_TIMEOUT_SECONDS:-60}

count_pattern() {
  local pattern=$1
  if [ -f "${LAUNCHER_LOG}" ]; then
    grep -c "${pattern}" "${LAUNCHER_LOG}" || true
  else
    printf '0\n'
  fi
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

write_timing_summary() {
  if [ ! -f "${LAUNCHER_LOG}" ]; then
    return
  fi
  python3 "${SCRIPT_DIR}/summarize_full89_timing.py" \
    "${LAUNCHER_LOG}" \
    --json-out "${TIMING_JSON}.tmp" \
    > "${TIMING_SUMMARY}.tmp" 2>&1 || true
  mv -f "${TIMING_SUMMARY}.tmp" "${TIMING_SUMMARY}" 2>/dev/null || true
  mv -f "${TIMING_JSON}.tmp" "${TIMING_JSON}" 2>/dev/null || true
}

write_snapshot() {
  local jid
  jid="$(latest_job_id)"
  write_timing_summary
  {
    printf '\n===== %s =====\n' "$(date -Iseconds)"
    if [ -n "${jid}" ]; then
      printf 'job_id=%s\n' "${jid}"
      ray job status "${jid}" || true
    else
      printf 'job_id=unknown\n'
    fi
    ray status || true
    printf 'gpu_summary:\n'
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 8 || true
    printf 'trajectory_json=%s\n' "$(count_files_with_timeout "${TRIALS_DIR}" -name trajectory.json)"
    printf 'result_json=%s\n' "$(count_files_with_timeout "${TRIALS_DIR}" -name result.json)"
    printf 'rollout_dumps=%s\n' "$(count_files_with_timeout "${ROLLOUT_DUMPS_DIR}" -maxdepth 1 -type f -name '*.pt')"
    printf 'checkpoint_files=%s\n' "$(count_files_with_timeout "${SAVE_DIR}" -maxdepth 3 -type f)"
    printf 'cached_sif=%s\n' "$(count_pattern 'Using cached Singularity image')"
    printf 'fastapi_ready=%s\n' "$(count_pattern 'Slime Singularity FastAPI server is ready')"
    printf 'fully_async_done=%s\n' "$(count_pattern 'Harbor fully-async rollout .* done in')"
    printf 'train_perf=%s\n' "$(count_pattern 'perf/actor_train_time')"
    printf 'traceback=%s\n' "$(count_pattern 'Traceback')"
    printf 'oom=%s\n' "$(count_pattern 'OutOfMemory\\|OOM\\|CUDA out of memory')"
    printf 'not_enough_samples=%s\n' "$(count_pattern 'Not enough samples')"
    printf 'timeout_lines=%s\n' "$(count_pattern 'timeout\\|timed out')"
    printf 'timing_summary=%s\n' "${TIMING_SUMMARY}"
    if [ -f "${TIMING_SUMMARY}" ]; then
      tail -n 60 "${TIMING_SUMMARY}" || true
    fi
    printf 'recent_events:\n'
    if [ -f "${LAUNCHER_LOG}" ]; then
      grep -E 'raysubmit_|fully-async|harbor/num|reward_mean|perf/rollout_time|perf/actor_train_time|saving checkpoint|successfully saved checkpoint|ray cluster has|The server is fired up|Traceback|failed|timeout|OutOfMemory|OOM|Not enough samples' "${LAUNCHER_LOG}" | tail -n 120 || true
    fi
  } >> "${LOG_FILE}" 2>&1
}

: > "${LOG_FILE}"
while true; do
  write_snapshot
  sleep "${INTERVAL_SECONDS}"
done
