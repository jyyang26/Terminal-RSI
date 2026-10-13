#!/bin/bash

set -euo pipefail

ROLE="${1:-launcher}"
MASTER_ADDR="${2:-}"

if [ "${ROLE}" != "launcher" ] && [ "${ROLE}" != "worker" ]; then
  echo "usage: $0 [launcher|worker] [master_addr]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
WRAPPER="${SCRIPT_DIR}/run_v030_trajectory_normalized_smoke.sh"

if [ -z "${MASTER_ADDR}" ]; then
  MASTER_ADDR="$(hostname -I | awk '{print $1}')"
fi

RAY_NODE_IP_PREFIX="${RAY_NODE_IP_PREFIX:-$(printf "%s\n" "${MASTER_ADDR}" | awk -F. '{print $1"."$2}')}"
SESSION="slime_v030_norm_smoke_${ROLE}"
LOG="/tmp/${SESSION}.log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  echo "log: ${LOG}"
  exit 0
fi

WORKER_KEEP_ALIVE=0
if [ "${ROLE}" = "worker" ]; then
  WORKER_KEEP_ALIVE=1
fi

ENV_ASSIGNMENTS=(
  "RUN_ROLE=${ROLE}"
  "WORKER_KEEP_ALIVE=${WORKER_KEEP_ALIVE}"
  "MASTER_ADDR=${MASTER_ADDR}"
  "RAY_NODE_IP_PREFIX=${RAY_NODE_IP_PREFIX}"
)

for name in \
  HARBOR_ROLLOUT_MODE \
  CASE_ROOT \
  BASE_FOLDER \
  SAVE_SUBDIR \
  CONFIG \
  ROLLOUT_BATCH_SIZE \
  N_SAMPLES_PER_PROMPT \
  GLOBAL_BATCH_SIZE \
  OVER_SAMPLING_BATCH_SIZE \
  NUM_ROLLOUT \
  ROLLOUT_MAX_RESPONSE_LEN \
  USE_WANDB \
  CLEAN_START \
  SAVE_INTERVAL \
  SAVE_DEBUG_TRAIN_DATA \
  DUMP_DETAILS \
  WORKER_ADDR \
  RAY_PORT \
  RAY_TEMP_DIR \
  RAY_CLUSTER_WAIT_SECONDS
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
echo "master: ${MASTER_ADDR}"
echo "log: ${LOG}"
