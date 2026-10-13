#!/bin/bash
#
# Bring up the 96-GPU (12-node) Qwen3.5-27B Daytona tmax-15k FULL PPO run in tmux,
# at the proven-safe 64k window. First REAL (non-overfit) run on the 14601-task
# tmax-15k corpus; eval is held-out TB2-89.
#
# Differences from start_tb2_full89_96gpu_27b_daytona_ppo_64k_tmux.sh:
#   - config template + wrapper -> the tmax15k_full ones.
#   - save/trials/RAY_TEMP/wandb identity -> tmax15k_full.
#   - DATA STAGING: each node extracts /tmp/tmax_harbor_15k_full from the shared-fs tar
#     before launch (the train pool is node-LOCAL /tmp, not shared fs; every rollout
#     node's HarborTrialWorker reads task dirs locally). Idempotent: skips if already
#     extracted (14601 task.toml present). The TB2-89 eval set stays on shared fs.
#   - forwards N_SAMPLES_PER_PROMPT / ROLLOUT_BATCH_SIZE / GLOBAL_BATCH_SIZE /
#     OVER_SAMPLING_BATCH_SIZE / USE_DYNAMIC_GLOBAL_BATCH_SIZE so the wrapper's n=1 +
#     oversample-280/train-256 defaults can be overridden by the caller.
#
#   # launcher node (node0), pass node0's 29.68.x IP as MASTER_ADDR:
#   DAYTONA_ENV_FILE=/path/to/daytona.env \
#     bash start_tmax15k_full_96gpu_27b_daytona_ppo_64k_tmux.sh launcher 10.0.0.11
#   # each worker node (node1..node11), passing the SAME launcher IP:
#   DAYTONA_ENV_FILE=/path/to/daytona.env \
#     bash start_tmax15k_full_96gpu_27b_daytona_ppo_64k_tmux.sh worker 10.0.0.11

set -euo pipefail

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

# --- tmax-15k data staging: extract the train pool to node-LOCAL /tmp -------
# The train pool MUST exist at this exact path on EVERY node (launcher + workers),
# because HarborTrialWorker reads task dirs node-locally. The tar lives on shared fs.
TMAX_DATA_TAR="${TMAX_DATA_TAR:-${CODE_ROOT}/harbor-code-synthesis/tmax_harbor_15k_full.tar}"
TMAX_DATA_DIR="${TMAX_DATA_DIR:-/tmp/tmax_harbor_15k_full}"
TMAX_EXPECT_TASKS="${TMAX_EXPECT_TASKS:-14601}"
stage_tmax_data() {
  local have
  have="$(find "${TMAX_DATA_DIR}" -maxdepth 2 -name task.toml 2>/dev/null | wc -l)"
  if [ "${have}" = "${TMAX_EXPECT_TASKS}" ]; then
    echo "[stage] tmax-15k already present at ${TMAX_DATA_DIR} (${have} tasks); skip."
    return 0
  fi
  if [ ! -f "${TMAX_DATA_TAR}" ]; then
    echo "[stage] ERROR: tar not found: ${TMAX_DATA_TAR}" >&2
    return 1
  fi
  echo "[stage] extracting tmax-15k (${have}/${TMAX_EXPECT_TASKS}) from ${TMAX_DATA_TAR} -> ${TMAX_DATA_DIR%/*} ..."
  rm -rf "${TMAX_DATA_DIR}" 2>/dev/null || true
  # tar top-level dir is tmax_harbor_15k_full/, so extract into /tmp.
  tar xf "${TMAX_DATA_TAR}" -C "$(dirname "${TMAX_DATA_DIR}")"
  have="$(find "${TMAX_DATA_DIR}" -maxdepth 2 -name task.toml 2>/dev/null | wc -l)"
  echo "[stage] done: ${have} tasks at ${TMAX_DATA_DIR}"
  [ "${have}" = "${TMAX_EXPECT_TASKS}" ] || { echo "[stage] WARNING: expected ${TMAX_EXPECT_TASKS}, got ${have}" >&2; }
}

# --- Daytona credentials --------------------------------------------------
DAYTONA_ENV_FILE="${DAYTONA_ENV_FILE:-/path/to/workspace/.slime_run_env/daytona_tier4.env}"
if [ -n "${DAYTONA_ENV_FILE:-}" ] && [ -f "${DAYTONA_ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  set -a
  source "${DAYTONA_ENV_FILE}"
  set +a
fi
if [ "${ROLE}" = "launcher" ]; then
  if [ -z "${DAYTONA_API_KEY:-}" ] && { [ -z "${DAYTONA_JWT_TOKEN:-}" ] || [ -z "${DAYTONA_ORGANIZATION_ID:-}" ]; }; then
    echo "ERROR: set DAYTONA_API_KEY (or DAYTONA_JWT_TOKEN + DAYTONA_ORGANIZATION_ID)" >&2
    echo "       e.g. via DAYTONA_ENV_FILE=/path/to/daytona.env before launching." >&2
    exit 2
  fi
fi

# --- W&B credentials --------------------------------------------------------
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/path/to/workspace/.slime_run_env/wandb_v2_retrain.env}"
if [ -n "${WANDB_ENV_FILE:-}" ] && [ -f "${WANDB_ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  set -a
  source "${WANDB_ENV_FILE}"
  set +a
fi

if [ -z "${MASTER_ADDR}" ]; then
  MASTER_ADDR="$(hostname -I | awk '{print $1}')"
fi

# Stage data on THIS node before anything else (both launcher and workers).
stage_tmax_data

CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/harbor_tmax15k_full_96gpu_27b_daytona_ppo_64k.yaml}"
BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_tmax15k_full_ppo_64k_${RUN_TAG}}"
SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_tmax15k_full_ppo_64k_${RUN_TAG}}"
RUN_TRIALS_DIR="${RUN_TRIALS_DIR:-/path/to/workspace/slime_harbor_trials/tmax15k_full_ppo_64k_${RUN_TAG}}"
CONFIG="${CONFIG:-${BASE_FOLDER%/}/config/$(basename "${CONFIG_TEMPLATE}")}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtmax15k}"
RAY_NODE_IP_PREFIX="${RAY_NODE_IP_PREFIX:-29.68,29.79,29.81}"

INSTALL_HARBOR_RUNTIME_DEPS="${INSTALL_HARBOR_RUNTIME_DEPS:-1}"
INSTALL_DAYTONA="${INSTALL_DAYTONA:-1}"

LAUNCHER_LOG="${LAUNCHER_LOG:-${BASE_FOLDER%/}/logs/launcher_${RUN_TAG}.log}"
WORKER_LOG="${WORKER_LOG:-/tmp/slime_27b96_tmax15k_daytona_${RUN_TAG}_worker.log}"

# Materialize a per-run config with the run-specific trials_dir patched in.
if [ "${ROLE}" = "launcher" ]; then
  mkdir -p "$(dirname "${CONFIG}")" "$(dirname "${LAUNCHER_LOG}")"
  sed "s#^\([[:space:]]*trials_dir:\).*#\1 ${RUN_TRIALS_DIR}#" "${CONFIG_TEMPLATE}" > "${CONFIG}"
fi

case "${ROLE}" in
  launcher)
    SESSION="slime_27b96_tmax15k_daytona_launcher"
    LOG="${LAUNCHER_LOG}"
    ;;
  worker)
    SESSION="slime_27b96_tmax15k_daytona_worker"
    LOG="${WORKER_LOG}"
    ;;
esac
WRAPPER="${WRAPPER:-${SCRIPT_DIR}/run_tmax15k_full_96gpu_27b_daytona_ppo_64k_fully_async.sh}"

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
  "EXPECTED_NODES=${EXPECTED_NODES:-12}"
  "ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-4}"
  "ACTOR_NUM_GPUS_PER_NODE=8"
  "ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-64}"
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
  HTTP_PROXY \
  HTTPS_PROXY \
  NO_PROXY \
  http_proxy \
  https_proxy \
  no_proxy \
  HARBOR_DAYTONA_S3_PROXY_URL \
  NUM_ROLLOUT \
  SAVE_INTERVAL \
  EVAL_INTERVAL \
  SKIP_EVAL_BEFORE_TRAIN \
  N_SAMPLES_PER_PROMPT \
  ROLLOUT_BATCH_SIZE \
  GLOBAL_BATCH_SIZE \
  OVER_SAMPLING_BATCH_SIZE \
  USE_DYNAMIC_GLOBAL_BATCH_SIZE \
  HARBOR_LENGTH_REWARD_LAMBDA \
  MEGATRON_TO_HF_MODE \
  CALCULATE_PER_TOKEN_LOSS \
  STAGE_HF_CKPT_TO_TMP \
  STAGE_TORCH_DIST_CKPT_TO_TMP \
  USE_ROLLOUT_LOGPROBS \
  ADVANTAGE_ESTIMATOR \
  EPS_CLIP \
  NUM_CRITIC_ONLY_STEPS \
  MAX_TOKENS_PER_GPU \
  TENSOR_MODEL_PARALLEL_SIZE \
  PIPELINE_MODEL_PARALLEL_SIZE \
  CP_SIZE \
  ROLLOUT_NUM_GPUS_PER_ENGINE \
  SLIME_GLOO_GROUP_TIMEOUT_MINUTES \
  SLIME_NEW_GROUP_TIMEOUT_MINUTES \
  USE_FAULT_TOLERANCE \
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
if [ "${INSTALL_HARBOR_RUNTIME_DEPS}" = "1" ]; then
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
echo "data: ${TMAX_DATA_DIR}"
echo "log: ${LOG}"
