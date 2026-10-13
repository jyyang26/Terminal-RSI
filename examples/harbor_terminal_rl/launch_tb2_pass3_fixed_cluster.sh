#!/bin/bash
# Launch and gate the corrected 56-GPU Qwen3.5-35B-A3B TB2 pass@3 eval.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HEAD="${HEAD:-29.127.33.153}"
WORKERS_STR="${WORKERS_STR:-29.119.97.70 29.127.68.155 29.127.68.53 29.127.69.167 29.127.81.202 29.232.240.217}"
read -r -a WORKERS <<<"${WORKERS_STR}"
: "${DAYTONA_API_KEY:?export DAYTONA_API_KEY before launch}"
TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)_tb2_pass3_fixed_tp8}"
BASE="${BASE_FOLDER:-/path/to/workspace/slime-terminal-RL/checkpoints/eval_only/35b-a3b-tp8/${TAG}}"
TRIALS="${RUN_TRIALS_DIR:-/path/to/workspace/slime_harbor_trials/tb2_pass3_fixed_${TAG}}"
LOG="${BASE}/logs/launcher.log"
METRICS="${TRIALS}/eval_0_metrics.json"
START="${SCRIPT_DIR}/start_tb2_eval_only_qwen35_tmux.sh"
COMMON="RUN_TAG=${TAG} MODEL_KIND=35b-a3b-tp8 BASE_FOLDER=${BASE} RUN_TRIALS_DIR=${TRIALS} N_SAMPLES_PER_EVAL_PROMPT=3 EVAL_TEMPERATURE=0.6 EVAL_TOP_P=0.95 EVAL_TOP_K=20 WANDB_GROUP=tb2-pass3-fixed-tp8-temp06-${TAG} HARBOR_SRC=/path/to/workspace/harbor-code-synthesis/src DAYTONA_API_KEY=${DAYTONA_API_KEY} RESTART_TMUX=1"
mkdir -p "${BASE}/logs"
printf '%s\n' "${TAG}" > /tmp/tb2_pass3_fixed_tag
for ip in "${WORKERS[@]}"; do
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${ip}" "${COMMON} bash '${START}' worker '${HEAD}' 35b-a3b-tp8" &
done
wait
ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${HEAD}" "${COMMON} LOG='${LOG}' bash '${START}' launcher '${HEAD}' 35b-a3b-tp8"
echo "started tag=${TAG} log=${LOG} metrics=${METRICS}"

# A genuine pass@3 may take up to three hours. Poll the machine-readable artifact.
deadline=$((SECONDS + ${WAIT_SECONDS:-12000}))
while [ ! -s "${METRICS}" ]; do
  if [ ${SECONDS} -ge ${deadline} ]; then
    echo "timed out waiting for ${METRICS}" >&2
    exit 2
  fi
  if grep -aqE "Job '.*' failed|Traceback.*HarborTrialWorkersUnavailable" "${LOG}" 2>/dev/null; then
    echo "eval job failed before metrics artifact was produced" >&2
    tail -120 "${LOG}" >&2
    exit 3
  fi
  sleep 30
done
python3 "${SCRIPT_DIR}/tb2_pass3_gate.py" "${METRICS}" --min 0.35 --max 0.45
