#!/bin/bash
#
# resume_tmax15k_qwen35_strongest_112gpu_no219.sh
# ---------------------------------------------------------------------------
# RESUME the STRONGEST run from iter_0000014 after the step15->16 update_weights
# hung on 4 dead SGLang engines on bad node .137.219 (FT failed to restart them;
# NCCL weight-broadcast rendezvous wedged ~1.5h, no self-heal). Job was stopped.
#
# CHANGES vs the original launch:
#   - EXCLUDE .137.219 (bad node, all 5+ engine deaths were here). Now 14 nodes:
#       actor   : 8 nodes = 64 GPU  (UNCHANGED — .219 was a ROLLOUT node, not actor)
#       rollout : 6 nodes = 48 GPU  (was 7n/56 GPU; dropped .219)
#       total   : 14 nodes = 112 GPU  (excl .137.206 AND .137.219)
#   - RESUME: reuse the SAME RUN_TAG so BASE_FOLDER/SAVE_SUBDIR resolve to the
#     existing dirs; the base wrapper auto-sets --load=SAVE_DIR (has iter14 actor
#     + _critic/iter14 critic) => warm-resume both actor AND critic from step14.
#     The wrapper also exports OVERRIDE_OPT_PARAM_SCHEDULER=1 (safe at constant LR).
#
# RUN_TAG MUST be passed in (the original run's tag) so it resumes, e.g.:
#   RUN_TAG=20260712_053239_qwen35_tmax15k_strongest bash resume_..._no219.sh
set -euo pipefail

LAUNCHER_IP="10.0.0.11"
WORKER_IPS=(
  10.0.0.3 10.0.0.12 10.0.0.8 10.0.0.13
  10.0.0.14 10.0.0.15 10.0.0.9
  10.0.0.17 10.0.0.7 10.0.0.19
  10.0.0.20 10.0.0.4
)
# excluded: .137.206 (bad NIC), .137.219 (dead engines this run),
#           .137.188 (/tmp wiped, ckpt re-stage too slow to beat launcher wait).
# NOTE: EXPECTED_NODES / ROLLOUT_NUM_GPUS below are overridable via env; the
# resume launch passes EXPECTED_NODES=13 ROLLOUT_NUM_GPUS=40 for this 13-node set.

SLIME_DIR="/path/to/workspace/code/slime-v0.3-harbor"
HARBOR_DIR="${SLIME_DIR}/examples/harbor_terminal_rl"
STARTER="${HARBOR_DIR}/start_tmax15k_full_96gpu_27b_daytona_ppo_64k_tmux.sh"
WRAPPER="${HARBOR_DIR}/run_tmax15k_qwen35_120gpu_ppo_criticlr30x_strongest.sh"
DAYTONA_ENV_FILE="${DAYTONA_ENV_FILE:-/path/to/workspace/.slime_run_env/daytona_relaunch_20260707.env}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/path/to/workspace/.slime_run_env/wandb_v2_retrain.env}"

# RUN_TAG MUST be the original run's tag (no date default) so it resumes from iter14.
if [ -z "${RUN_TAG:-}" ]; then
  echo "ERROR: RUN_TAG must be set to the original run's tag to resume from iter14." >&2
  echo "  e.g. RUN_TAG=\$(cat /tmp/strongest_run_tag.txt) bash $0" >&2
  exit 2
fi

BASE_FOLDER="/path/to/workspace/slime_harbor_qwen35_27b_tmax15k_strongest_criticlr30x_${RUN_TAG}"
SAVE_SUBDIR="Qwen3.5-27B_tmax15k_strongest_criticlr30x_${RUN_TAG}"
RUN_TRIALS_DIR="/path/to/workspace/slime_harbor_trials/tmax15k_strongest_${RUN_TAG}"
RAY_TEMP_DIR="/tmp/rtmax15k35"
WANDB_GROUP="q35-27b-tmax15k-strongest-criticlr30x-actorlr2x-standard-os260-64k-n1"

# --- Topology env: 14 nodes / 112 GPU (rollout 56->48, expected 15->14) ---------------
EXPECTED_NODES=${EXPECTED_NODES:-14}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-48}
read -r -d '' TOPO_ENV <<EOF || true
RUN_TAG=${RUN_TAG}
WRAPPER=${WRAPPER}
EXPECTED_NODES=${EXPECTED_NODES}
ACTOR_NUM_NODES=8
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS}
TENSOR_MODEL_PARALLEL_SIZE=2
PIPELINE_MODEL_PARALLEL_SIZE=16
CP_SIZE=2
MAX_TOKENS_PER_GPU=8192
USE_FAULT_TOLERANCE=1
OVERRIDE_OPT_PARAM_SCHEDULER=1
BASE_FOLDER=${BASE_FOLDER}
SAVE_SUBDIR=${SAVE_SUBDIR}
RUN_TRIALS_DIR=${RUN_TRIALS_DIR}
RAY_TEMP_DIR=${RAY_TEMP_DIR}
WANDB_GROUP=${WANDB_GROUP}
DAYTONA_ENV_FILE=${DAYTONA_ENV_FILE}
WANDB_ENV_FILE=${WANDB_ENV_FILE}
HARBOR_DAYTONA_S3_PROXY_URL=http://star-proxy.oa.com:3128
EOF

SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10"

echo "=============================================================="
echo " RESUME STRONGEST RUN from iter14  RUN_TAG=${RUN_TAG}"
echo " actor 8n(64GPU) + rollout 6n(48GPU) = 14n/112GPU"
echo " launcher=${LAUNCHER_IP}  workers=${#WORKER_IPS[@]}  (excl .206 + .219)"
echo " base=${BASE_FOLDER}"
echo "=============================================================="

# 1) launcher on THIS node (.137.106)
echo "[orch] starting LAUNCHER on ${LAUNCHER_IP} ..."
env $(echo "$TOPO_ENV" | tr '\n' ' ') bash "${STARTER}" launcher "${LAUNCHER_IP}"

# 2) workers over ssh (setsid so SIGHUP can't kill the tmux dispatch)
for ip in "${WORKER_IPS[@]}"; do
  echo "[orch] starting WORKER on ${ip} ..."
  $SSH "${ip}" "setsid env $(echo "$TOPO_ENV" | tr '\n' ' ') bash '${STARTER}' worker '${LAUNCHER_IP}'" \
    && echo "[orch]   ${ip} worker launched" \
    || echo "[orch]   !! ${ip} worker FAILED to launch"
done

echo "[orch] all nodes dispatched (RESUME). RUN_TAG=${RUN_TAG}"
echo "[orch] launcher log: ${BASE_FOLDER}/logs/launcher_${RUN_TAG}.log"
