#!/bin/bash

set -euo pipefail

ROLE="${1:-launcher}"
MASTER_ADDR="${2:-${MASTER_ADDR:-}}"

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

CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/harbor_terminal_bench2_full89_32gpu_9b_max100_actor1_rollout3_batch3_n32_128k_timeout3600_trajectory_normalized.yaml}"
BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_32gpu_actor1_rollout3_batch3_n32_fully_async_trajectory_normalized_timeout3600_${RUN_TAG}}"
SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-9B_full89_32gpu_actor1_rollout3_batch3_n32_fa_trajnorm_timeout3600_${RUN_TAG}}"
RUN_TRIALS_DIR="${RUN_TRIALS_DIR:-/path/to/workspace/slime_harbor_trials/tb2_full89_32gpu_9b_max100_actor1_rollout3_batch3_n32_resp4096_ctx128k_timeout3600_trajectory_normalized_${RUN_TAG}}"
CONFIG="${CONFIG:-${BASE_FOLDER%/}/config/$(basename "${CONFIG_TEMPLATE}")}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rfa-a1r3-${RUN_TAG}}"
RAY_NODE_IP_PREFIX="${RAY_NODE_IP_PREFIX:-29.79,29.81}"
INSTALL_HARBOR_RUNTIME_DEPS="${INSTALL_HARBOR_RUNTIME_DEPS:-0}"
INSTALL_DAYTONA="${INSTALL_DAYTONA:-0}"

LAUNCHER_LOG="${LAUNCHER_LOG:-/tmp/slime_full89_32gpu_9b_a1r3_b3n32_trajnorm_fa_${RUN_TAG}_launcher.log}"
WORKER_LOG="${WORKER_LOG:-/tmp/slime_full89_32gpu_9b_a1r3_b3n32_trajnorm_fa_${RUN_TAG}_worker.log}"
MONITOR_LOG="${MONITOR_LOG:-/tmp/slime_full89_32gpu_9b_a1r3_b3n32_trajnorm_fa_${RUN_TAG}_monitor.log}"
TIMING_SUMMARY="${TIMING_SUMMARY:-/tmp/slime_full89_32gpu_9b_a1r3_b3n32_trajnorm_fa_${RUN_TAG}_timing_summary.txt}"
TIMING_JSON="${TIMING_JSON:-/tmp/slime_full89_32gpu_9b_a1r3_b3n32_trajnorm_fa_${RUN_TAG}_timing_summary.json}"

if [ "${ROLE}" = "launcher" ]; then
  mkdir -p "$(dirname "${CONFIG}")" "$(dirname "${LAUNCHER_LOG}")"
  sed "s#^\([[:space:]]*trials_dir:\).*#\1 ${RUN_TRIALS_DIR}#" "${CONFIG_TEMPLATE}" > "${CONFIG}"
fi

case "${ROLE}" in
  launcher)
    SESSION="slime_full89_32gpu_9b_a1r3_b3n32_trajnorm_fa_launcher"
    LOG="${LAUNCHER_LOG}"
    WRAPPER="${SCRIPT_DIR}/run_full89_grpo_32gpu_9b_max100_actor1_rollout3_batch3_n32_128k_timeout3600_trajectory_normalized_fully_async.sh"
    ;;
  worker)
    SESSION="slime_full89_32gpu_9b_a1r3_b3n32_trajnorm_fa_worker"
    LOG="${WORKER_LOG}"
    WRAPPER="${SCRIPT_DIR}/run_full89_grpo_32gpu_9b_max100_actor1_rollout3_batch3_n32_128k_timeout3600_trajectory_normalized_fully_async.sh"
    ;;
  monitor)
    SESSION="slime_full89_32gpu_9b_a1r3_b3n32_trajnorm_fa_monitor"
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
  "EXPECTED_NODES=${EXPECTED_NODES:-4}"
  "ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}"
  "ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}"
  "ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-24}"
  "ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-3}"
  "OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-3}"
  "N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-32}"
  "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-96}"
  "NUM_ROLLOUT=${NUM_ROLLOUT:-150}"
  "CP_SIZE=${CP_SIZE:-1}"
  "SEQUENCE_PARALLEL=${SEQUENCE_PARALLEL:-0}"
  "ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
  "ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-131072}"
  "SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-131072}"
  "SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-131072}"
  "SGLANG_CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
  "SEQ_LENGTH=${SEQ_LENGTH:-65536}"
  "HARBOR_MAX_TRAIN_TOTAL_TOKENS=${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-65536}"
  "SAVE_INTERVAL=${SAVE_INTERVAL:-30}"
  "RAY_CLUSTER_WAIT_SECONDS=${RAY_CLUSTER_WAIT_SECONDS:-900}"
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
  DAYTONA_API_KEY \
  DAYTONA_JWT_TOKEN \
  DAYTONA_ORGANIZATION_ID \
  HTTP_PROXY \
  HTTPS_PROXY \
  NO_PROXY \
  http_proxy \
  https_proxy \
  no_proxy \
  QKV_FORMAT \
  USE_DYNAMIC_BATCH_SIZE \
  USE_DYNAMIC_GLOBAL_BATCH_SIZE \
  MAX_TOKENS_PER_GPU \
  LOG_PROBS_MAX_TOKENS_PER_GPU \
  CUSTOM_REWARD_POST_PROCESS_PATH \
  HARBOR_PARTIAL_REWARD_MAX \
  SAVE_DEBUG_TRAIN_DATA \
  DUMP_DETAILS \
  RAY_raylet_start_wait_time_s \
  RAY_PORT \
  RAY_NODE_MANAGER_PORT \
  RAY_OBJECT_MANAGER_PORT \
  RAY_RUNTIME_ENV_AGENT_PORT \
  RAY_DASHBOARD_AGENT_LISTEN_PORT \
  RAY_DASHBOARD_AGENT_GRPC_PORT \
  RAY_METRICS_EXPORT_PORT \
  RAY_MIN_WORKER_PORT \
  RAY_MAX_WORKER_PORT
do
  if [ -n "${!name+x}" ]; then
    ENV_ASSIGNMENTS+=("${name}=${!name}")
  fi
done

ENV_CMD=()
for item in "${ENV_ASSIGNMENTS[@]}"; do
  ENV_CMD+=("$(printf "%q" "${item}")")
done

mkdir -p "$(dirname "${LOG}")"
INSTALL_CMD=""
if [ "${INSTALL_HARBOR_RUNTIME_DEPS}" = "1" ] && [ "${ROLE}" != "monitor" ]; then
  INSTALL_CMD="INSTALL_DAYTONA=$(printf "%q" "${INSTALL_DAYTONA}") bash $(printf "%q" "${SCRIPT_DIR}/install_harbor_runtime_deps.sh") && "
fi
CMD="cd $(printf "%q" "${SLIME_DIR}") && ${INSTALL_CMD}exec env ${ENV_CMD[*]} bash $(printf "%q" "${WRAPPER}") > $(printf "%q" "${LOG}") 2>&1"

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
