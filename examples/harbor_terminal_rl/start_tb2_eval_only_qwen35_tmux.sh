#!/bin/bash
# Start one node of the 7-node/56-GPU TB2 SGLang-only eval cluster.
set -euo pipefail
ROLE="${1:-launcher}"; MASTER_ADDR="${2:-}"; MODEL_KIND="${MODEL_KIND:-${3:-27b}}"
[ "${ROLE}" = launcher ] || [ "${ROLE}" = worker ] || { echo "usage: $0 launcher|worker MASTER_ADDR [27b|35b-a3b]" >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="${SLIME_DIR:-/path/to/workspace/slime-terminal-RL}"
HARBOR_SRC="${HARBOR_SRC:-/path/to/workspace/harbor-code-synthesis/src}"
[ -n "${MASTER_ADDR}" ] || MASTER_ADDR="$(hostname -I | awk '{print $1}')"
if [ -n "${DAYTONA_ENV_FILE:-}" ] && [ -f "${DAYTONA_ENV_FILE}" ]; then set -a; source "${DAYTONA_ENV_FILE}"; set +a; fi
if [ "${ROLE}" = launcher ] && [ -z "${DAYTONA_API_KEY:-}" ] && { [ -z "${DAYTONA_JWT_TOKEN:-}" ] || [ -z "${DAYTONA_ORGANIZATION_ID:-}" ]; }; then
  echo "DAYTONA credentials are required" >&2; exit 2
fi
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_tb2_eval_${MODEL_KIND//-/_}}"
BASE_FOLDER="${BASE_FOLDER:-${SLIME_DIR}/checkpoints/eval_only/${MODEL_KIND}/${RUN_TAG}}"
RUN_TRIALS_DIR="${RUN_TRIALS_DIR:-/path/to/workspace/slime_harbor_trials/tb2_eval_only_${MODEL_KIND}_${RUN_TAG}}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/harbor_tb2_eval_only_tmax_timeouts.yaml}"
CONFIG="${CONFIG:-${BASE_FOLDER}/config/harbor_tb2_eval_only.yaml}"
LOG="${LOG:-${BASE_FOLDER}/logs/${ROLE}.log}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtb2eval35}"
if [ "${ROLE}" = launcher ]; then
  mkdir -p "$(dirname "${CONFIG}")" "$(dirname "${LOG}")"
  sed "s#^[[:space:]]*trials_dir:.*#  trials_dir: ${RUN_TRIALS_DIR}#" "${CONFIG_TEMPLATE}" > "${CONFIG}"
fi
SESSION="tb2_eval_${MODEL_KIND//-/_}_${ROLE}"
[ "${RESTART_TMUX:-1}" != 1 ] || tmux kill-session -t "${SESSION}" 2>/dev/null || true
ENVV=(
 "RUN_ROLE=${ROLE}" "MASTER_ADDR=${MASTER_ADDR}" "RAY_NODE_IP_PREFIX=$(echo "${MASTER_ADDR}" | awk -F. '{print $1"."$2}')"
 "SLIME_DIR=${SLIME_DIR}" "CODE_ROOT=${SLIME_DIR}" "HARBOR_SRC=${HARBOR_SRC}" "MODEL_KIND=${MODEL_KIND}"
 "CONFIG=${CONFIG}" "BASE_FOLDER=${BASE_FOLDER}" "RUN_TAG=${RUN_TAG}" "RUN_TRIALS_DIR=${RUN_TRIALS_DIR}"
 "EXPECTED_NODES=7" "ROLLOUT_NUM_GPUS=56" "RAY_TEMP_DIR=${RAY_TEMP_DIR}" "CLEAN_START=1"
 "USE_EXISTING_RAY_HEAD=0" "RAY_CLUSTER_WAIT_SECONDS=1800" "WANDB_MODE=${WANDB_MODE:-online}"
 "WANDB_PROJECT=${WANDB_PROJECT:-slime-terminal-rl}" "HARBOR_CREATE_RATE_PER_SEC=${HARBOR_CREATE_RATE_PER_SEC:-6.0}"
)
if [ "${ROLE}" = worker ]; then ENVV+=("WORKER_KEEP_ALIVE=1"); else ENVV+=("WORKER_KEEP_ALIVE=0"); fi
for n in N_SAMPLES_PER_EVAL_PROMPT EVAL_TEMPERATURE EVAL_TOP_P EVAL_TOP_K EXTRA_SGLANG_ARGS HF_CKPT STAGED_HF_CKPT STAGE_HF_CKPT_TO_TMP WANDB_GROUP DAYTONA_API_KEY DAYTONA_JWT_TOKEN DAYTONA_ORGANIZATION_ID DAYTONA_API_URL WANDB_API_KEY HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  [ -z "${!n+x}" ] || ENVV+=("${n}=${!n}")
done
Q=(); for x in "${ENVV[@]}"; do Q+=("$(printf %q "${x}")"); done
CMD="cd $(printf %q "${SLIME_DIR}") && exec env ${Q[*]} bash $(printf %q "${SCRIPT_DIR}/run_tb2_eval_only_qwen35.sh") $(printf %q "${MODEL_KIND}") > $(printf %q "${LOG}") 2>&1"
tmux new-session -d -s "${SESSION}" "${CMD}"
echo "started ${SESSION}; log=${LOG}; config=${CONFIG}; trials=${RUN_TRIALS_DIR}"
