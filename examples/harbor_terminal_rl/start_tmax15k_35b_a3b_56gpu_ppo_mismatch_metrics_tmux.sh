#!/bin/bash
#
# Bring up the 56-GPU (7-node x 8 H20) Qwen3.5-35B-A3B (MoE) Harbor Terminal-RL run
# in tmux (DAYTONA backend), DP1 topology + train/inference log-prob MISMATCH METRICS.
#
# Sibling of start_tmax15k_35b_a3b_56gpu_ppo_strongest_tmux.sh, repointed at the
# mismatch-metrics wrapper and resized to the DP1 split:
#   actor = 2 nodes (16 GPU) TP2 x PP8 x CP1 x EP2 => DP1  (40/8 = 5 layers/stage, even)
#   rollout = 5 nodes (40 GPU) = 5 independent TP8 SGLang engines
#
#   # launcher node (node0):
#   DAYTONA_ENV_FILE=~/.slime_run_env/daytona_pass3.env \
#   HARBOR_SRC=/path/to/harbor/src HARBOR_TASKS_DIR=/path/to/tmax15k_terminal_bench \
#     bash start_tmax15k_35b_a3b_56gpu_ppo_mismatch_metrics_tmux.sh launcher <node0_ip>
#   # each of the other 6 worker nodes, same launcher IP:
#   HARBOR_SRC=/path/to/harbor/src \
#     bash start_tmax15k_35b_a3b_56gpu_ppo_mismatch_metrics_tmux.sh worker <node0_ip>

set -euo pipefail

ROLE="${1:-launcher}"
MASTER_ADDR="${2:-}"

if [ "${ROLE}" != "launcher" ] && [ "${ROLE}" != "worker" ]; then
  echo "usage: $0 [launcher|worker] [master_addr]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SLIME_DIR="${SLIME_DIR:-/path/to/workspace/slime-terminal-RL}"
CODE_ROOT="${CODE_ROOT:-${SLIME_DIR}}"
HARBOR_SRC="${HARBOR_SRC:?set HARBOR_SRC=/path/to/harbor/src before launching}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

# --- Daytona credentials --------------------------------------------------
if [ -n "${DAYTONA_ENV_FILE:-}" ] && [ -f "${DAYTONA_ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  set -a; source "${DAYTONA_ENV_FILE}"; set +a
fi
if [ "${ROLE}" = "launcher" ]; then
  if [ -z "${DAYTONA_API_KEY:-}" ] && { [ -z "${DAYTONA_JWT_TOKEN:-}" ] || [ -z "${DAYTONA_ORGANIZATION_ID:-}" ]; }; then
    echo "ERROR: set DAYTONA_API_KEY (or DAYTONA_JWT_TOKEN + DAYTONA_ORGANIZATION_ID)," >&2
    echo "       inline or via DAYTONA_ENV_FILE=/path/to/daytona.env, before launching." >&2
    exit 2
  fi
fi

if [ -z "${MASTER_ADDR}" ]; then
  MASTER_ADDR="$(hostname -I | awk '{print $1}')"
fi

HARBOR_TASKS_DIR="${HARBOR_TASKS_DIR:-}"
if [ "${ROLE}" = "launcher" ] && [ -z "${HARBOR_TASKS_DIR}" ]; then
  echo "ERROR: set HARBOR_TASKS_DIR=/path/to/terminal-bench task pool before launching." >&2
  exit 2
fi

CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/harbor_tmax15k_56gpu_35b_a3b_daytona_ppo_dp1_mismatch_metrics.yaml}"
BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_35b_a3b_tb2_56gpu_ppo_dp1_mismatch_${RUN_TAG}}"
SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-35B-A3B_56gpu_ppo_dp1_mismatch_${RUN_TAG}}"
RUN_TRIALS_DIR="${RUN_TRIALS_DIR:-/path/to/workspace/slime_harbor_trials/tmax15k_35b_a3b_dp1_mismatch_${RUN_TAG}}"
CONFIG="${CONFIG:-${BASE_FOLDER%/}/config/$(basename "${CONFIG_TEMPLATE}")}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtmax15k35bdp1}"
RAY_NODE_IP_PREFIX="${RAY_NODE_IP_PREFIX:-$(printf "%s\n" "${MASTER_ADDR}" | awk -F. '{print $1"."$2}')}"

INSTALL_HARBOR_RUNTIME_DEPS="${INSTALL_HARBOR_RUNTIME_DEPS:-1}"
INSTALL_DAYTONA="${INSTALL_DAYTONA:-1}"

LAUNCHER_LOG="${LAUNCHER_LOG:-${BASE_FOLDER%/}/logs/launcher_${RUN_TAG}.log}"
WORKER_LOG="${WORKER_LOG:-/tmp/slime_tmax35bdp1_${RUN_TAG}_worker.log}"

if [ "${ROLE}" = "launcher" ]; then
  mkdir -p "$(dirname "${CONFIG}")" "$(dirname "${LAUNCHER_LOG}")"
  sed -e "s#^\([[:space:]]*harbor_tasks_dir:\).*#\1 ${HARBOR_TASKS_DIR}#" \
      -e "s#^\([[:space:]]*trials_dir:\).*#\1 ${RUN_TRIALS_DIR}#" \
      "${CONFIG_TEMPLATE}" > "${CONFIG}"
fi

case "${ROLE}" in
  launcher)
    SESSION="slime_tmax35bdp1_launcher"
    LOG="${LAUNCHER_LOG}"
    ;;
  worker)
    SESSION="slime_tmax35bdp1_worker"
    LOG="${WORKER_LOG}"
    ;;
esac
WRAPPER="${WRAPPER:-${SCRIPT_DIR}/run_tmax15k_35b_a3b_56gpu_ppo_mismatch_metrics.sh}"

if [ "${RESTART_TMUX:-0}" = "1" ]; then
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  echo "log: ${LOG}"
  exit 0
fi

# DP1 topology: 2 actor nodes (16 GPU) + 5 rollout nodes (40 GPU). Overridable.
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
  "HARBOR_ROLLOUT_MODE=standard"
  "RAY_TEMP_DIR=${RAY_TEMP_DIR}"
  "EXPECTED_NODES=${EXPECTED_NODES:-7}"
  "ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2}"
  "ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}"
  "ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-40}"
  "RAY_CLUSTER_WAIT_SECONDS=1800"
  "CLEAN_START=${CLEAN_START:-1}"
)

if [ "${ROLE}" = "worker" ]; then
  ENV_ASSIGNMENTS+=("WORKER_KEEP_ALIVE=1")
else
  ENV_ASSIGNMENTS+=("WORKER_KEEP_ALIVE=0")
  ENV_ASSIGNMENTS+=("LAUNCHER_LOG=${LAUNCHER_LOG}")
fi

# Forward optional overrides + Daytona/W&B creds if the caller exported them.
for name in \
  WORKER_ADDR \
  USE_WANDB \
  WANDB_MODE \
  WANDB_API_KEY \
  WANDB_GROUP \
  DAYTONA_API_KEY \
  DAYTONA_JWT_TOKEN \
  DAYTONA_ORGANIZATION_ID \
  DAYTONA_API_URL \
  HARBOR_TASKS_DIR \
  HARBOR_CREATE_RATE_PER_SEC \
  HTTP_PROXY \
  HTTPS_PROXY \
  NO_PROXY \
  http_proxy \
  https_proxy \
  no_proxy \
  NUM_ROLLOUT \
  SAVE_INTERVAL \
  EVAL_INTERVAL \
  SKIP_EVAL_BEFORE_TRAIN \
  N_SAMPLES_PER_EVAL_PROMPT \
  EVAL_TEMPERATURE \
  EVAL_TOP_P \
  EVAL_TOP_K \
  MEGATRON_TO_HF_MODE \
  STAGE_HF_CKPT_TO_TMP \
  STAGE_TORCH_DIST_CKPT_TO_TMP \
  USE_ROLLOUT_LOGPROBS \
  GET_MISMATCH_METRICS \
  CUSTOM_TIS_FUNCTION_PATH \
  ADVANTAGE_ESTIMATOR \
  MAX_TOKENS_PER_GPU \
  TENSOR_MODEL_PARALLEL_SIZE \
  PIPELINE_MODEL_PARALLEL_SIZE \
  CP_SIZE \
  EXPERT_MODEL_PARALLEL_SIZE \
  EXPERT_TENSOR_PARALLEL_SIZE \
  USE_DYNAMIC_BATCH_SIZE \
  ROLLOUT_NUM_GPUS_PER_ENGINE \
  ROLLOUT_BATCH_SIZE \
  GLOBAL_BATCH_SIZE \
  LR \
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
INSTALL_CMD=""
if [ "${INSTALL_HARBOR_RUNTIME_DEPS}" = "1" ] && [ -f "${SCRIPT_DIR}/install_harbor_runtime_deps.sh" ]; then
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
