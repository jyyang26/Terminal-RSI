#!/bin/bash
#
# Launch the tmax-15k 27B DEBUG-TRAIN-ONLY harness on 8 idle nodes (64 GPU) in tmux.
#
# WHAT / WHY
#   Isolates the TRAIN half of the tmax-15k run to answer two questions the main run
#   (20260701_133022, died at rollout-2 weight sync w/ TorchMemorySaver OOM) couldn't
#   separate: (A) does the 27B torch_dist ckpt load cleanly on this hardware, and
#   (B) does 64k forward/backward/optimizer OOM. It replays the main run's REAL dumped
#   rollout data (rollout_dumps/{id}.pt) through the optimizer with SGLang + Daytona
#   OFF (--load-debug-rollout-data auto-forces --debug-train-only). Topology is
#   TP2 x PP8 x CP4 = 64 (CP doubled vs the main run's 32-GPU CP2, to halve per-GPU
#   activation memory and directly test whether more CP dodges the OOM).
#
#   Uses 8 nodes the main run does NOT touch. No tmax task data is staged (debug-train
#   reads no task dirs; data comes from the dumped .pt). Base 27B loads from shared fs.
#
# USAGE (from a node that can ssh the others; run launcher first, then workers):
#   # launcher node0 (pass node0's bond IP as MASTER_ADDR):
#   bash start_tmax15k_debug_train_only_64gpu_tmux.sh launcher <node0_ip>
#   # each of the 7 worker nodes, SAME launcher IP:
#   bash start_tmax15k_debug_train_only_64gpu_tmux.sh worker <node0_ip>

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

# --- W&B credentials --------------------------------------------------------
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/path/to/workspace/.slime_run_env/wandb_v2_retrain.env}"
if [ -n "${WANDB_ENV_FILE:-}" ] && [ -f "${WANDB_ENV_FILE}" ]; then
  set -a; source "${WANDB_ENV_FILE}"; set +a
fi

if [ -z "${MASTER_ADDR}" ]; then
  MASTER_ADDR="$(hostname -I | awk '{print $1}')"
fi

# The main run's dumped rollouts feed the optimizer. {rollout_id} is substituted per
# step by slime. Files 0.pt..5.pt exist; NUM_ROLLOUT=3 replays 0,1,2 (2 = the step
# the main run died on). MAIN_RUN_BASE + LOAD_DEBUG_ROLLOUT_DATA are forwarded to the
# wrapper (file 1), which sets the flag.
MAIN_RUN_BASE="${MAIN_RUN_BASE:-/path/to/workspace/slime_harbor_qwen35_27b_tmax15k_full_ppo_64k_20260701_133022}"
export MAIN_RUN_BASE
# NB: build the {rollout_id} template separately — a literal "{rollout_id}" inside a
# ${VAR:-default} expansion makes bash close the expansion at the first "}".
_default_replay="${MAIN_RUN_BASE}/rollout_dumps/{rollout_id}.pt"
export LOAD_DEBUG_ROLLOUT_DATA="${LOAD_DEBUG_ROLLOUT_DATA:-$_default_replay}"

# 64-GPU debug topology (CP4). These are exported so the wrapper + base inherit them.
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-8}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-8}"
export CP_SIZE="${CP_SIZE:-4}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-0}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-999}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-999}"
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-0}"
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-0}"

CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/harbor_tmax15k_full_96gpu_27b_daytona_ppo_64k.yaml}"
BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_tmax15k_dbgtrain_64k_${RUN_TAG}}"
SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_tmax15k_dbgtrain_64k_${RUN_TAG}}"
CONFIG="${CONFIG:-${BASE_FOLDER%/}/config/$(basename "${CONFIG_TEMPLATE}")}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtmax15k_dbg}"
RAY_NODE_IP_PREFIX="${RAY_NODE_IP_PREFIX:-29.68,29.79,29.81}"

INSTALL_HARBOR_RUNTIME_DEPS="${INSTALL_HARBOR_RUNTIME_DEPS:-1}"
INSTALL_DAYTONA="${INSTALL_DAYTONA:-1}"

LAUNCHER_LOG="${LAUNCHER_LOG:-${BASE_FOLDER%/}/logs/launcher_${RUN_TAG}.log}"
WORKER_LOG="${WORKER_LOG:-/tmp/slime_27b64_tmax15k_dbgtrain_${RUN_TAG}_worker.log}"

# Materialize a per-run config (trials_dir patched in; harmless in debug-train-only).
if [ "${ROLE}" = "launcher" ]; then
  mkdir -p "$(dirname "${CONFIG}")" "$(dirname "${LAUNCHER_LOG}")"
  cp "${CONFIG_TEMPLATE}" "${CONFIG}"
fi

case "${ROLE}" in
  launcher) SESSION="slime_27b64_tmax15k_dbgtrain_launcher"; LOG="${LAUNCHER_LOG}" ;;
  worker)   SESSION="slime_27b64_tmax15k_dbgtrain_worker";   LOG="${WORKER_LOG}" ;;
esac
WRAPPER="${WRAPPER:-${SCRIPT_DIR}/run_tmax15k_debug_train_only_64gpu.sh}"

if [ "${RESTART_TMUX:-0}" = "1" ]; then
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"; echo "log: ${LOG}"; exit 0
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
  "EXPECTED_NODES=8"
  "ACTOR_NUM_NODES=${ACTOR_NUM_NODES}"
  "ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE}"
  "ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS}"
  "RAY_CLUSTER_WAIT_SECONDS=1800"
  "CLEAN_START=${CLEAN_START:-1}"
)

if [ "${ROLE}" = "worker" ]; then
  ENV_ASSIGNMENTS+=("WORKER_KEEP_ALIVE=1")
else
  ENV_ASSIGNMENTS+=("WORKER_KEEP_ALIVE=0")
  ENV_ASSIGNMENTS+=("LAUNCHER_LOG=${LAUNCHER_LOG}")
fi

# Forward the debug/topology overrides + W&B creds if exported.
for name in \
  USE_WANDB WANDB_MODE WANDB_API_KEY WANDB_GROUP \
  MAIN_RUN_BASE LOAD_DEBUG_ROLLOUT_DATA LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE \
  TENSOR_MODEL_PARALLEL_SIZE PIPELINE_MODEL_PARALLEL_SIZE CP_SIZE \
  MAX_TOKENS_PER_GPU NUM_ROLLOUT SAVE_INTERVAL EVAL_INTERVAL \
  STAGE_HF_CKPT_TO_TMP STAGE_TORCH_DIST_CKPT_TO_TMP \
  ADVANTAGE_ESTIMATOR EPS_CLIP NUM_CRITIC_ONLY_STEPS \
  SLIME_GLOO_GROUP_TIMEOUT_MINUTES SLIME_NEW_GROUP_TIMEOUT_MINUTES \
  USE_FAULT_TOLERANCE LR RAY_PORT
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
echo "role: ${ROLE}   run_tag: ${RUN_TAG}   master: ${MASTER_ADDR}"
echo "topology: TP${TENSOR_MODEL_PARALLEL_SIZE} x PP${PIPELINE_MODEL_PARALLEL_SIZE} x CP${CP_SIZE} = $((TENSOR_MODEL_PARALLEL_SIZE*PIPELINE_MODEL_PARALLEL_SIZE*CP_SIZE)) GPU"
echo "replay: ${LOAD_DEBUG_ROLLOUT_DATA}"
echo "base: ${BASE_FOLDER}"
echo "log: ${LOG}"
