#!/bin/bash

set -euo pipefail

ROLE="${1:-launcher}"
MASTER_ADDR="${2:-${MASTER_ADDR:-}}"

if [ "${ROLE}" != "launcher" ] && [ "${ROLE}" != "worker" ]; then
  echo "usage: $0 [launcher|worker] [master_addr]" >&2
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

CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/harbor_terminal_bench2_train_v2_local_32gpu_9b_daytona_ppo_n4_bringup.yaml}"
BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_9b_tb2_train_v2_32gpu_daytona_ppo_n4_bringup_${RUN_TAG}}"
SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-9B_tb2_train_v2_32gpu_daytona_ppo_n4_bringup_${RUN_TAG}}"
RUN_TRIALS_DIR="${RUN_TRIALS_DIR:-/path/to/workspace/slime_harbor_trials/tb2_train_v2_32gpu_9b_daytona_ppo_n4_bringup_${RUN_TAG}}"
CONFIG="${CONFIG:-${BASE_FOLDER%/}/config/$(basename "${CONFIG_TEMPLATE}")}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/r9b-${RUN_TAG:0:16}}"
RAY_NODE_IP_PREFIX="${RAY_NODE_IP_PREFIX:-29.79,29.81}"
DAYTONA_ENV_FILE="${DAYTONA_ENV_FILE:-${HOME}/.daytona_harbor.env}"

LAUNCHER_LOG="${LAUNCHER_LOG:-/tmp/slime_tb2v2_9b_ppo_n4_${RUN_TAG}_launcher.log}"
WORKER_LOG="${WORKER_LOG:-/tmp/slime_tb2v2_9b_ppo_n4_${RUN_TAG}_worker.log}"

if [ "${ROLE}" = "launcher" ]; then
  mkdir -p "$(dirname "${CONFIG}")" "$(dirname "${LAUNCHER_LOG}")"
  sed "s#^\([[:space:]]*trials_dir:\).*#\1 ${RUN_TRIALS_DIR}#" "${CONFIG_TEMPLATE}" > "${CONFIG}"
fi

case "${ROLE}" in
  launcher)
    SESSION="slime_tb2v2_9b_ppo_n4_launcher"
    LOG="${LAUNCHER_LOG}"
    ;;
  worker)
    SESSION="slime_tb2v2_9b_ppo_n4_worker"
    LOG="${WORKER_LOG}"
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
  "DAYTONA_ENV_FILE=${DAYTONA_ENV_FILE}"
  "HARBOR_ROLLOUT_MODE=${HARBOR_ROLLOUT_MODE:-standard}"
  "RAY_TEMP_DIR=${RAY_TEMP_DIR}"
  "EXPECTED_NODES=${EXPECTED_NODES:-4}"
  "ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}"
  "ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}"
  "ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-24}"
  "TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}"
  "PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-8}"
  "CP_SIZE=${CP_SIZE:-1}"
  "SEQUENCE_PARALLEL=${SEQUENCE_PARALLEL:-0}"
  "ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}"
  "OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-9}"
  "N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-4}"
  "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}"
  "NUM_ROLLOUT=${NUM_ROLLOUT:-56}"
  "MEGATRON_CONFIG_PATH=${MEGATRON_CONFIG_PATH:-${SCRIPT_DIR}/megatron_roles_tb2v2_9b_criticlr10x.yaml}"
  "ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-ppo}"
  "EPS_CLIP=${EPS_CLIP:-0.2}"
  "NUM_CRITIC_ONLY_STEPS=${NUM_CRITIC_ONLY_STEPS:-1}"
  "USE_ROLLOUT_LOGPROBS=${USE_ROLLOUT_LOGPROBS:-0}"
  "KEEP_OLD_ACTOR=${KEEP_OLD_ACTOR:-1}"
  "CUSTOM_REWARD_POST_PROCESS_PATH=${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.partial_reward.trajectory_grouped_progress_rewards}"
  "HARBOR_PARTIAL_REWARD_MAX=${HARBOR_PARTIAL_REWARD_MAX:-1.0}"
  "HARBOR_LENGTH_REWARD_LAMBDA=${HARBOR_LENGTH_REWARD_LAMBDA:-0.0}"
  "USE_DYNAMIC_BATCH_SIZE=${USE_DYNAMIC_BATCH_SIZE:-1}"
  "USE_DYNAMIC_GLOBAL_BATCH_SIZE=${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"
  "MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-24576}"
  "ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
  "ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-32768}"
  "SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-32768}"
  "SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-32768}"
  "SGLANG_CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
  "SEQ_LENGTH=${SEQ_LENGTH:-32768}"
  "HARBOR_MAX_TRAIN_TOTAL_TOKENS=${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-32768}"
  "SAVE_INTERVAL=${SAVE_INTERVAL:-10}"
  "SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.50}"
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
  "USE_FAULT_TOLERANCE=${USE_FAULT_TOLERANCE:-1}"
  "ROLLOUT_HEALTH_CHECK_INTERVAL=${ROLLOUT_HEALTH_CHECK_INTERVAL:-30}"
  "ROLLOUT_HEALTH_CHECK_TIMEOUT=${ROLLOUT_HEALTH_CHECK_TIMEOUT:-120}"
  "ROLLOUT_HEALTH_CHECK_FIRST_WAIT=${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-600}"
  "USE_WANDB=${USE_WANDB:-1}"
  "WANDB_MODE=${WANDB_MODE:-online}"
  "EVAL_INTERVAL=${EVAL_INTERVAL:-10}"
  "N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT:-1}"
  "EVAL_DATASET_NAME=${EVAL_DATASET_NAME:-tb2}"
  "SKIP_EVAL_BEFORE_TRAIN=${SKIP_EVAL_BEFORE_TRAIN:-0}"
  "DIRECT_RUN=${DIRECT_RUN:-1}"
  "RAY_INCLUDE_DASHBOARD=${RAY_INCLUDE_DASHBOARD:-0}"
  "RAY_CLUSTER_WAIT_SECONDS=${RAY_CLUSTER_WAIT_SECONDS:-900}"
  "INSTALL_HARBOR_RUNTIME_DEPS=${INSTALL_HARBOR_RUNTIME_DEPS:-1}"
  "INSTALL_DAYTONA=${INSTALL_DAYTONA:-1}"
  "CLEAN_START=${CLEAN_START:-1}"
)

if [ "${ROLE}" = "worker" ]; then
  ENV_ASSIGNMENTS+=("WORKER_KEEP_ALIVE=1")
else
  ENV_ASSIGNMENTS+=("WORKER_KEEP_ALIVE=0")
fi

for name in \
  WORKER_ADDR \
  WANDB_MODE \
  WANDB_API_KEY \
  WANDB_GROUP \
  HTTP_PROXY \
  HTTPS_PROXY \
  NO_PROXY \
  http_proxy \
  https_proxy \
  no_proxy \
  QKV_FORMAT \
  LOG_PROBS_CHUNK_SIZE \
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
if [ "${INSTALL_HARBOR_RUNTIME_DEPS:-1}" = "1" ]; then
  INSTALL_CMD="INSTALL_DAYTONA=$(printf "%q" "${INSTALL_DAYTONA:-1}") bash $(printf "%q" "${SCRIPT_DIR}/install_harbor_runtime_deps.sh") && "
fi

WRAPPER="${SCRIPT_DIR}/run_tb2_train_v2_32gpu_9b_daytona_ppo_n4_bringup_fully_async.sh"
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
