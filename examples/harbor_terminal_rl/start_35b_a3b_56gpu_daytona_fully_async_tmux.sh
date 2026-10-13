#!/bin/bash
#
# Bring up the 56-GPU (7-node x 8 H20) Qwen3.5-35B-A3B (MoE) Harbor Terminal-RL run
# in tmux, using the DAYTONA cloud-sandbox backend. Sibling of
# start_tb2_full89_96gpu_27b_daytona_ppo_tmux.sh, resized from 12 -> 7 nodes and
# repointed at this checkout + the gy4 model paths.
#
#   # launcher node (node0), pass node0's IP as MASTER_ADDR:
#   DAYTONA_API_KEY=dtn_xxx \
#   HARBOR_SRC=/path/to/harbor/src \
#   HARBOR_TASKS_DIR=/path/to/terminal-bench-2 \
#     bash start_35b_a3b_56gpu_daytona_fully_async_tmux.sh launcher <node0_ip>
#   # each of the other 6 worker nodes, passing the SAME launcher IP:
#   HARBOR_SRC=/path/to/harbor/src \
#     bash start_35b_a3b_56gpu_daytona_fully_async_tmux.sh worker <node0_ip>
#
# The launcher submits the Ray job (train_async.py). Workers join the Ray head and
# stay alive on the shared STOP marker (WORKER_KEEP_ALIVE=1). Credentials can also be
# supplied via a DAYTONA_ENV_FILE (sourced) instead of inline DAYTONA_API_KEY.

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
# Harbor agent runtime (harbor/src) — set this to wherever you installed it.
HARBOR_SRC="${HARBOR_SRC:?set HARBOR_SRC=/path/to/harbor/src before launching}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

# --- Daytona credentials --------------------------------------------------
# Either export DAYTONA_API_KEY inline, or point DAYTONA_ENV_FILE at a file that
# sets it (and optionally DAYTONA_JWT_TOKEN / DAYTONA_ORGANIZATION_ID / DAYTONA_API_URL).
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

# Task pool (dir of "*/task.toml"). REQUIRED on the launcher; patched into the YAML.
HARBOR_TASKS_DIR="${HARBOR_TASKS_DIR:-}"
if [ "${ROLE}" = "launcher" ] && [ -z "${HARBOR_TASKS_DIR}" ]; then
  echo "ERROR: set HARBOR_TASKS_DIR=/path/to/terminal-bench task pool before launching." >&2
  exit 2
fi

CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/harbor_terminal_bench2_56gpu_35b_a3b_daytona_max120_actor4_rollout3_batch8_n32_64k_timeout3600_trajectory_normalized_fully_async.yaml}"
BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_35b_a3b_tb2_56gpu_daytona_${RUN_TAG}}"
SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-35B-A3B_56gpu_daytona_${RUN_TAG}}"
RUN_TRIALS_DIR="${RUN_TRIALS_DIR:-/path/to/workspace/slime_harbor_trials/tb2_56gpu_35b_a3b_daytona_${RUN_TAG}}"
CONFIG="${CONFIG:-${BASE_FOLDER%/}/config/$(basename "${CONFIG_TEMPLATE}")}"
# Ray AF_UNIX socket paths must stay < 107 bytes, so keep RAY_TEMP_DIR SHORT.
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/r35ba3b56}"
# Restrict Ray to the correct NIC subnet; override if the allocated nodes differ.
RAY_NODE_IP_PREFIX="${RAY_NODE_IP_PREFIX:-$(printf "%s\n" "${MASTER_ADDR}" | awk -F. '{print $1"."$2}')}"

INSTALL_HARBOR_RUNTIME_DEPS="${INSTALL_HARBOR_RUNTIME_DEPS:-1}"
INSTALL_DAYTONA="${INSTALL_DAYTONA:-1}"

LAUNCHER_LOG="${LAUNCHER_LOG:-${BASE_FOLDER%/}/logs/launcher_${RUN_TAG}.log}"
WORKER_LOG="${WORKER_LOG:-/tmp/slime_35ba3b56_daytona_${RUN_TAG}_worker.log}"

# Materialize a per-run config with the run-specific task/trials dirs patched in.
if [ "${ROLE}" = "launcher" ]; then
  mkdir -p "$(dirname "${CONFIG}")" "$(dirname "${LAUNCHER_LOG}")"
  sed -e "s#^\([[:space:]]*harbor_tasks_dir:\).*#\1 ${HARBOR_TASKS_DIR}#" \
      -e "s#^\([[:space:]]*trials_dir:\).*#\1 ${RUN_TRIALS_DIR}#" \
      "${CONFIG_TEMPLATE}" > "${CONFIG}"
fi

case "${ROLE}" in
  launcher)
    SESSION="slime_35ba3b56_daytona_launcher"
    LOG="${LAUNCHER_LOG}"
    ;;
  worker)
    SESSION="slime_35ba3b56_daytona_worker"
    LOG="${WORKER_LOG}"
    ;;
esac
WRAPPER="${WRAPPER:-${SCRIPT_DIR}/run_35b_a3b_56gpu_daytona_actor4_rollout3_batch8_n32_64k_timeout3600_trajectory_normalized_fully_async.sh}"

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
  "EXPECTED_NODES=7"
  "ACTOR_NUM_NODES=4"
  "ACTOR_NUM_GPUS_PER_NODE=8"
  "ROLLOUT_NUM_GPUS=24"
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
  MEGATRON_TO_HF_MODE \
  STAGE_HF_CKPT_TO_TMP \
  STAGE_TORCH_DIST_CKPT_TO_TMP \
  USE_ROLLOUT_LOGPROBS \
  ADVANTAGE_ESTIMATOR \
  MAX_TOKENS_PER_GPU \
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
