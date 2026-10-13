#!/bin/bash

set -euo pipefail

ROLE="${1:-launcher}"
MASTER_ADDR="${2:-}"

if [ "${ROLE}" != "launcher" ] && [ "${ROLE}" != "worker" ] && [ "${ROLE}" != "monitor" ]; then
  echo "usage: $0 [launcher|worker|monitor] [master_addr]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CODE_ROOT="${CODE_ROOT:-/path/to/workspace/code}"
SLIME_DIR="${SLIME_DIR:-${CODE_ROOT}/slime-v0.3-harbor}"
HARBOR_SRC="${HARBOR_SRC:-${CODE_ROOT}/harbor/src}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

if [ -z "${MASTER_ADDR}" ]; then
  MASTER_ADDR="$(hostname -I | awk '{print $1}')"
fi

CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/harbor_terminal_bench2_full89_32gpu_9b_max100_actor2_rollout2_128k_timeout3600_trajectory_normalized.yaml}"
BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_32gpu_actor2_rollout2_fully_async_trajectory_normalized_timeout3600_${RUN_TAG}}"
SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-9B_full89_32gpu_actor2_rollout2_fa_trajnorm_timeout3600_${RUN_TAG}}"
RUN_TRIALS_DIR="${RUN_TRIALS_DIR:-/path/to/workspace/slime_harbor_trials/tb2_full89_32gpu_9b_max100_actor2_rollout2_resp4096_ctx128k_timeout3600_trajectory_normalized_${RUN_TAG}}"
CONFIG="${CONFIG:-${BASE_FOLDER%/}/config/$(basename "${CONFIG_TEMPLATE}")}"
# Ray creates long session/socket suffixes under --temp-dir. Keep this prefix
# short enough for the 107-byte Unix socket path limit.
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rfa-${RUN_TAG}}"
RAY_NODE_IP_PREFIX="${RAY_NODE_IP_PREFIX:-29.79,29.81}"

LAUNCHER_LOG="${LAUNCHER_LOG:-/tmp/slime_full89_32gpu_9b_trajnorm_fa_${RUN_TAG}_launcher.log}"
WORKER_LOG="${WORKER_LOG:-/tmp/slime_full89_32gpu_9b_trajnorm_fa_${RUN_TAG}_worker.log}"
MONITOR_LOG="${MONITOR_LOG:-/tmp/slime_full89_32gpu_9b_trajnorm_fa_${RUN_TAG}_monitor.log}"
TIMING_SUMMARY="${TIMING_SUMMARY:-/tmp/slime_full89_32gpu_9b_trajnorm_fa_${RUN_TAG}_timing_summary.txt}"
TIMING_JSON="${TIMING_JSON:-/tmp/slime_full89_32gpu_9b_trajnorm_fa_${RUN_TAG}_timing_summary.json}"

if [ "${ROLE}" = "launcher" ]; then
  mkdir -p "$(dirname "${CONFIG}")"
  sed "s#^\([[:space:]]*trials_dir:\).*#\1 ${RUN_TRIALS_DIR}#" "${CONFIG_TEMPLATE}" > "${CONFIG}"
fi

case "${ROLE}" in
  launcher)
    SESSION="slime_full89_32gpu_9b_trajnorm_fa_launcher"
    LOG="${LAUNCHER_LOG}"
    WRAPPER="${SCRIPT_DIR}/run_full89_grpo_32gpu_9b_max100_actor2_rollout2_128k_timeout3600_trajectory_normalized_fully_async.sh"
    ;;
  worker)
    SESSION="slime_full89_32gpu_9b_trajnorm_fa_worker"
    LOG="${WORKER_LOG}"
    WRAPPER="${SCRIPT_DIR}/run_full89_grpo_32gpu_9b_max100_actor2_rollout2_128k_timeout3600_trajectory_normalized_fully_async.sh"
    ;;
  monitor)
    SESSION="slime_full89_32gpu_9b_trajnorm_fa_monitor"
    LOG="${MONITOR_LOG}"
    WRAPPER="${SCRIPT_DIR}/monitor_full89_32gpu_9b_trajectory_normalized_fully_async.sh"
    ;;
esac

if [ "${RESTART_TMUX:-0}" = "1" ]; then
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  echo "log: ${LOG}"
  exit 0
fi

ENV_ASSIGNMENTS=(
  "RUN_TAG=${RUN_TAG}"
  "RUN_ROLE=${ROLE}"
  "MASTER_ADDR=${MASTER_ADDR}"
  "RAY_NODE_IP_PREFIX=${RAY_NODE_IP_PREFIX}"
  "CODE_ROOT=${CODE_ROOT}"
  "SLIME_DIR=${SLIME_DIR}"
  "HARBOR_SRC=${HARBOR_SRC}"
  "CONFIG=${CONFIG}"
  "BASE_FOLDER=${BASE_FOLDER}"
  "SAVE_SUBDIR=${SAVE_SUBDIR}"
  "HARBOR_ROLLOUT_MODE=fully_async"
  "RAY_TEMP_DIR=${RAY_TEMP_DIR}"
  "EXPECTED_NODES=4"
  "ACTOR_NUM_NODES=2"
  "ACTOR_NUM_GPUS_PER_NODE=8"
  "ROLLOUT_NUM_GPUS=16"
  "RAY_CLUSTER_WAIT_SECONDS=900"
  "CLEAN_START=${CLEAN_START:-1}"
)

if [ "${ROLE}" = "worker" ]; then
  ENV_ASSIGNMENTS+=("WORKER_KEEP_ALIVE=1")
else
  ENV_ASSIGNMENTS+=("WORKER_KEEP_ALIVE=0")
fi

if [ "${ROLE}" = "monitor" ]; then
  ENV_ASSIGNMENTS+=(
    "LAUNCHER_LOG=${LAUNCHER_LOG}"
    "LOG_FILE=${MONITOR_LOG}"
    "TIMING_SUMMARY=${TIMING_SUMMARY}"
    "TIMING_JSON=${TIMING_JSON}"
    "TRIALS_DIR=${RUN_TRIALS_DIR}"
    "INTERVAL_SECONDS=${INTERVAL_SECONDS:-300}"
  )
fi

for name in \
  WORKER_ADDR \
  USE_WANDB \
  WANDB_MODE \
  WANDB_API_KEY \
  WANDB_GROUP \
  SAVE_DEBUG_TRAIN_DATA \
  DUMP_DETAILS \
  RAY_PORT
do
  if [ -n "${!name+x}" ]; then
    ENV_ASSIGNMENTS+=("${name}=${!name}")
  fi
done

ENV_CMD=()
for item in "${ENV_ASSIGNMENTS[@]}"; do
  ENV_CMD+=("$(printf "%q" "${item}")")
done

CMD="cd $(printf "%q" "${SLIME_DIR}") && exec env ${ENV_CMD[*]} bash $(printf "%q" "${WRAPPER}") > $(printf "%q" "${LOG}") 2>&1"

tmux new-session -d -s "${SESSION}" "${CMD}"

echo "started ${SESSION}"
echo "role: ${ROLE}"
echo "run_tag: ${RUN_TAG}"
echo "master: ${MASTER_ADDR}"
echo "base: ${BASE_FOLDER}"
echo "save_subdir: ${SAVE_SUBDIR}"
echo "trials: ${RUN_TRIALS_DIR}"
echo "config: ${CONFIG}"
echo "log: ${LOG}"
if [ "${ROLE}" = "monitor" ]; then
  echo "timing_summary: ${TIMING_SUMMARY}"
  echo "timing_json: ${TIMING_JSON}"
fi
