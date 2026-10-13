#!/bin/bash
# Health check for the 96-GPU (12-node) Qwen3.5-27B fully-async trajectory_normalized run.
#
# Runs on node0 (Ray head). Prints a machine-readable verdict block that the
# scheduled monitor reads to decide whether to downgrade trial concurrency
# (32 -> 24 per node) or restart the run.
#
# Usage:
#   bash monitor_full89_96gpu_27b_fully_async_health.sh [BASE_FOLDER]
#
# Verdicts:
#   HEALTHY               - ray up, job RUNNING, dumps progressing, no fatal errors
#   STALL                 - job RUNNING but no new rollout dump for > STALL_MINUTES
#   DEGRADED_CONCURRENCY  - heartbeat/ActorDied/TrialWorker-killed/node-died signatures
#                           (the "32 trials/node too aggressive" symptom set)
#   DEAD                  - ray not running, or job FAILED/STOPPED
#
set -uo pipefail

EXPECTED_NODES="${EXPECTED_NODES:-12}"
# A 27B actor step is ~75min and fully-async rollout collection adds a long tail,
# so a healthy run can legitimately go >2h between dumps. 150min avoids spamming
# STALL on normal long steps; a true hang will still trip it eventually.
STALL_MINUTES="${STALL_MINUTES:-150}"

# --- locate the active run --------------------------------------------------
BASE_FOLDER="${1:-${BASE_FOLDER:-}}"
if [ -z "${BASE_FOLDER}" ]; then
  # Newest matching run folder (tmux launcher appends _<RUN_TAG>).
  BASE_FOLDER="$(ls -dt /path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_96gpu_*fully_async_trajnorm_64k_timeout3600* 2>/dev/null | head -1 || true)"
fi

echo "===== HARBOR_96G_27B_HEALTH ====="
echo "ts=$(date '+%Y-%m-%d %H:%M:%S')"
echo "base_folder=${BASE_FOLDER:-<none-found>}"

VERDICT="HEALTHY"
REASONS=()

# --- 1. Ray cluster liveness ------------------------------------------------
RAY_STATUS="$(ray status 2>&1 || true)"
if echo "${RAY_STATUS}" | grep -qiE "Could not find any running Ray|ConnectionError"; then
  VERDICT="DEAD"; REASONS+=("ray-head-not-running")
  ACTIVE_NODES=0
else
  ACTIVE_NODES="$(echo "${RAY_STATUS}" | awk '/^Active:$/{f=1;next} /^Pending:$/{f=0} f && /node_/{c++} END{print c+0}')"
fi
echo "ray_active_nodes=${ACTIVE_NODES}/${EXPECTED_NODES}"
if [ "${VERDICT}" != "DEAD" ] && [ "${ACTIVE_NODES}" -lt "${EXPECTED_NODES}" ]; then
  # A dropped node is the classic heartbeat/over-concurrency symptom.
  VERDICT="DEGRADED_CONCURRENCY"; REASONS+=("ray-nodes-${ACTIVE_NODES}-below-${EXPECTED_NODES}")
fi

# --- 2. Ray job status ------------------------------------------------------
LAUNCHER_LOG="${LAUNCHER_LOG:-}"
if [ -z "${LAUNCHER_LOG}" ] && [ -n "${BASE_FOLDER}" ]; then
  LAUNCHER_LOG="$(ls -t "${BASE_FOLDER%/}"/logs/launcher_*.log 2>/dev/null | head -1 || true)"
fi
echo "launcher_log=${LAUNCHER_LOG:-<none>}"

JOB_ID=""
JOB_STATUS="UNKNOWN"
if [ -n "${LAUNCHER_LOG}" ] && [ -f "${LAUNCHER_LOG}" ]; then
  JOB_ID="$(grep -oE 'raysubmit_[A-Za-z0-9]+' "${LAUNCHER_LOG}" | tail -1 || true)"
fi
if [ -n "${JOB_ID}" ]; then
  JOB_STATUS="$(ray job status "${JOB_ID}" 2>&1 | grep -oE 'RUNNING|SUCCEEDED|FAILED|STOPPED|PENDING' | tail -1 || true)"
  [ -z "${JOB_STATUS}" ] && JOB_STATUS="UNKNOWN"
fi
echo "ray_job=${JOB_ID:-<none>} status=${JOB_STATUS}"
case "${JOB_STATUS}" in
  FAILED|STOPPED) VERDICT="DEAD"; REASONS+=("ray-job-${JOB_STATUS}") ;;
esac

# --- 3. Rollout dump progress ----------------------------------------------
DUMP_DIR="${BASE_FOLDER%/}/rollout_dumps"
DUMP_COUNT=0
DUMP_AGE_MIN=-1
if [ -d "${DUMP_DIR}" ]; then
  DUMP_COUNT="$(ls "${DUMP_DIR}"/*.pt 2>/dev/null | wc -l | tr -d ' ')"
  LATEST_DUMP="$(ls -t "${DUMP_DIR}"/*.pt 2>/dev/null | head -1 || true)"
  if [ -n "${LATEST_DUMP}" ]; then
    DUMP_AGE_MIN=$(( ( $(date +%s) - $(stat -c %Y "${LATEST_DUMP}") ) / 60 ))
  fi
fi
echo "rollout_dumps=${DUMP_COUNT} latest_age_min=${DUMP_AGE_MIN}"
if [ "${VERDICT}" = "HEALTHY" ] && [ "${JOB_STATUS}" = "RUNNING" ] \
   && [ "${DUMP_AGE_MIN}" -gt "${STALL_MINUTES}" ]; then
  VERDICT="STALL"; REASONS+=("no-new-dump-${DUMP_AGE_MIN}min")
fi

# --- 4. Local GPU utilization (node0 = actor node) --------------------------
GPU_UTIL_MAX="n/a"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_UTIL_MAX="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1 || echo n/a)"
fi
echo "node0_gpu_util_max=${GPU_UTIL_MAX} (actor node: 0%% during rollout-wait is normal)"

# --- 5. Fatal / instability signatures in the log --------------------------
# We deliberately separate BENIGN, self-healing signals from FATAL ones so the
# 27B run (long steps, occasional dropped trials) does not spam false alarms:
#   benign  : host-RAM "worker killed / Dropping failed Harbor rollout sample"
#             (over-sampling absorbs it) and container memory-watchdog kills
#             ("Container exceeded memory limit" -> only that one trial dies).
#   fatal   : real CUDA/GPU OOM on the actor; Ray node death / heartbeat loss;
#             and "could not produce enough complete trainable groups" (the
#             RuntimeError that actually killed run96retok).
SIG_COUNT=0
if [ -n "${LAUNCHER_LOG}" ] && [ -f "${LAUNCHER_LOG}" ]; then
  TAIL="$(tail -n 4000 "${LAUNCHER_LOG}")"

  # Real cluster-instability (node/actor death) — NOT counting graceful trial drops.
  CONC_SIG="$(echo "${TAIL}" | grep -ciE 'missing too many heartbeats|node has died|ActorDiedError|ActorUnavailableError|killed by ray\.kill|FD Shutdown' || true)"

  # GPU/CUDA OOM only. Strip the benign host-RAM lines and container-watchdog
  # kills first, then look for a genuine torch/CUDA OOM.
  GPU_OOM_SIG="$(echo "${TAIL}" \
    | grep -ivE 'running low on memory|Dropping failed Harbor rollout sample|Container exceeded memory limit|Memory limit exceeded|Memory explosion detected' \
    | grep -ciE 'CUDA out of memory|torch\.cuda\.OutOfMemoryError|CUDA error: out of memory' || true)"

  # The actual run96retok killer.
  GROUPS_SIG="$(echo "${TAIL}" | grep -ciE 'could not produce enough complete trainable groups' || true)"

  # Benign counters, reported for visibility only (do NOT change verdict).
  HOSTRAM_DROPS="$(echo "${TAIL}" | grep -ciE 'running low on memory' || true)"
  WD_KILLS="$(echo "${TAIL}" | grep -ciE 'Container exceeded memory limit|Memory limit exceeded' || true)"
  CONV_SIG="$(echo "${TAIL}" | grep -ciE 'RolloutConversionError' || true)"

  echo "log_sig conc=${CONC_SIG} gpu_oom=${GPU_OOM_SIG} insufficient_groups=${GROUPS_SIG} conversion=${CONV_SIG}"
  echo "log_benign host_ram_trial_drops=${HOSTRAM_DROPS} container_watchdog_kills=${WD_KILLS} (informational; self-healing)"

  if [ "${GROUPS_SIG}" -gt 0 ]; then
    VERDICT="DEAD"; REASONS+=("insufficient-trainable-groups-${GROUPS_SIG}")
  fi
  if [ "${CONC_SIG}" -gt 0 ]; then
    VERDICT="DEGRADED_CONCURRENCY"; REASONS+=("log-conc-signatures-${CONC_SIG}")
  fi
  if [ "${GPU_OOM_SIG}" -gt 0 ]; then
    [ "${VERDICT}" = "HEALTHY" ] && VERDICT="DEAD"
    REASONS+=("gpu-oom-${GPU_OOM_SIG}")
  fi
  SIG_COUNT=$((CONC_SIG + GPU_OOM_SIG + GROUPS_SIG))
fi

echo "verdict=${VERDICT}"
echo "reasons=${REASONS[*]:-none}"
echo "===== END ====="
