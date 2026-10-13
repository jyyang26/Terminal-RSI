#!/bin/bash

set -euo pipefail

# Bring up the 96-GPU (12-node) Qwen3.5-27B fully-async trajectory_normalized run
# in tmux. Run on the launcher node, then on each of the 11 worker nodes:
#
#   # launcher node (node0):
#   bash start_full89_96gpu_27b_..._fully_async_tmux.sh launcher
#   # each worker node (node1..node11), passing the launcher IP:
#   bash start_full89_96gpu_27b_..._fully_async_tmux.sh worker <launcher_ip>
#
# The launcher submits the Ray job (train_async.py). Workers join the Ray head
# and stay alive on a shared STOP marker (WORKER_KEEP_ALIVE=1) so the cluster
# does not collapse when a worker's `ray start` returns. When the launcher exits
# it writes the STOP marker and workers tear down — this is the watchdog that
# prevents idle GPUs from being reclaimed (platform kills <30% util after 4h).

ROLE="${1:-launcher}"
MASTER_ADDR="${2:-}"

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

CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/harbor_terminal_bench2_full89_96gpu_27b_max120_actor4_rollout8_batch8_n32_64k_timeout3600_trajectory_normalized_fully_async.yaml}"
BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_96gpu_actor4_rollout8_batch8_n32_fully_async_trajnorm_64k_timeout3600_${RUN_TAG}}"
SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_full89_96gpu_actor4_rollout8_batch8_n32_fa_trajnorm_64k_timeout3600_${RUN_TAG}}"
RUN_TRIALS_DIR="${RUN_TRIALS_DIR:-/path/to/workspace/slime_harbor_trials/tb2_full89_96gpu_27b_max120_actor4_rollout8_batch8_n32_resp4096_ctx64k_traincap64k_timeout3600_trajectory_normalized_${RUN_TAG}}"
CONFIG="${CONFIG:-${BASE_FOLDER%/}/config/$(basename "${CONFIG_TEMPLATE}")}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/r27b96fa-${RUN_TAG}}"
# H20-018 cluster IP prefix; override if the allocated nodes differ.
RAY_NODE_IP_PREFIX="${RAY_NODE_IP_PREFIX:-29.68,29.79,29.81}"

LAUNCHER_LOG="${LAUNCHER_LOG:-${BASE_FOLDER%/}/logs/launcher_${RUN_TAG}.log}"
WORKER_LOG="${WORKER_LOG:-/tmp/slime_full89_96gpu_27b_fa_${RUN_TAG}_worker.log}"

# Materialize a per-run config with the run-specific trials_dir patched in.
if [ "${ROLE}" = "launcher" ]; then
  mkdir -p "$(dirname "${CONFIG}")" "$(dirname "${LAUNCHER_LOG}")"
  sed "s#^\([[:space:]]*trials_dir:\).*#\1 ${RUN_TRIALS_DIR}#" "${CONFIG_TEMPLATE}" > "${CONFIG}"
fi

case "${ROLE}" in
  launcher)
    SESSION="slime_full89_96gpu_27b_fa_launcher"
    LOG="${LAUNCHER_LOG}"
    ;;
  worker)
    SESSION="slime_full89_96gpu_27b_fa_worker"
    LOG="${WORKER_LOG}"
    ;;
esac
WRAPPER="${SCRIPT_DIR}/run_full89_grpo_96gpu_27b_max120_actor4_rollout8_batch8_n32_64k_timeout3600_trajectory_normalized_fully_async.sh"

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
  "EXPECTED_NODES=12"
  "ACTOR_NUM_NODES=4"
  "ACTOR_NUM_GPUS_PER_NODE=8"
  "ROLLOUT_NUM_GPUS=64"
  "RAY_CLUSTER_WAIT_SECONDS=1800"
  "CLEAN_START=${CLEAN_START:-1}"
)

if [ "${ROLE}" = "worker" ]; then
  ENV_ASSIGNMENTS+=("WORKER_KEEP_ALIVE=1")
else
  ENV_ASSIGNMENTS+=("WORKER_KEEP_ALIVE=0")
  ENV_ASSIGNMENTS+=("LAUNCHER_LOG=${LAUNCHER_LOG}")
fi

# Forward optional overrides if the caller exported them.
for name in \
  WORKER_ADDR \
  USE_WANDB \
  WANDB_MODE \
  WANDB_API_KEY \
  WANDB_GROUP \
  NUM_ROLLOUT \
  SAVE_INTERVAL \
  STAGE_HF_CKPT_TO_TMP \
  STAGE_TORCH_DIST_CKPT_TO_TMP \
  USE_ROLLOUT_LOGPROBS \
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

mkdir -p "$(dirname "${LOG}")"
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
