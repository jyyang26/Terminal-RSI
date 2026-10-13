#!/bin/bash

set -euo pipefail

JOB_ID=${JOB_ID:-}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-1800}
LOG_FILE=${LOG_FILE:-/tmp/slime_full89_32gpu_max100_5epoch_monitor.log}
LAUNCHER_LOG=${LAUNCHER_LOG:-/tmp/slime_full89_32gpu_max100_5epoch_launcher.log}
BASE_FOLDER=${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_32gpu_max100_n16_5epoch_resp2048_ctx128k}
ROLLOUT_DUMPS_DIR=${ROLLOUT_DUMPS_DIR:-${BASE_FOLDER}/rollout_dumps}
SAVE_DIR=${SAVE_DIR:-${BASE_FOLDER}/Qwen3.5-9B_terminal_bench2_full89_32gpu_max100_n16_5epoch_resp2048_ctx128k}
TRIALS_DIR=${TRIALS_DIR:-/path/to/workspace/slime_harbor_trials/tb2_full89_32gpu_max100_5epoch_resp2048_ctx128k}

count_pattern() {
  local pattern=$1
  if [ -f "${LAUNCHER_LOG}" ]; then
    grep -c "${pattern}" "${LAUNCHER_LOG}" || true
  else
    printf '0\n'
  fi
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
    printf 'trajectory_json=%s\n' "$(find "${TRIALS_DIR}" -name trajectory.json 2>/dev/null | wc -l || true)"
    printf 'rollout_dumps=%s\n' "$(find "${ROLLOUT_DUMPS_DIR}" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l || true)"
    printf 'checkpoint_files=%s\n' "$(find "${SAVE_DIR}" -maxdepth 3 -type f 2>/dev/null | wc -l || true)"
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
