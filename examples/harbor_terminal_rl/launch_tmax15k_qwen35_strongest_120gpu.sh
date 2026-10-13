#!/bin/bash
#
# launch_tmax15k_qwen35_strongest_120gpu.sh
# ---------------------------------------------------------------------------
# Launch the "STRONGEST RUN": Qwen3.5-27B + tmax-15k + best PPO recipe
# (cold-start critic, critic lr 30x / actor lr 2x, standard skim-oversample 260,
# 100-step ceiling) on 15 HEALTHY nodes = 120 GPU of server 019.
#
# Topology is the PROVEN 15-node tmax config (relaunch_tmax15k_120gpu_no206.sh,
# the §Z backward-OOM fix): bad-NIC node .137.206 is EXCLUDED.
#   actor   : 8 nodes = 64 GPU  TP2 x PP16 x CP2, max-tok 8192
#   rollout : 7 nodes = 56 GPU  28 SGLang engines
#   total   : 15 nodes = 120 GPU
#
# Fans out launcher (THIS node, .137.106) + 14 workers over plain ssh (we are ON
# the 019 head node, so no expect/jizhi_client — per CLAUDE.md). The experiment
# knobs (mode=standard, 3.5 ckpts, critic-lr YAML, os260, 100 steps, warmup2) live
# in the WRAPPER (self-exported defaults); this script owns TOPOLOGY + run IDENTITY
# + creds only. WRAPPER is passed via the starter's WRAPPER env (no starter edit).
set -euo pipefail

LAUNCHER_IP="10.0.0.11"
WORKER_IPS=(
  10.0.0.3 10.0.0.12 10.0.0.8 10.0.0.13
  10.0.0.14 10.0.0.15 10.0.0.9 10.0.0.16
  10.0.0.17 10.0.0.18 10.0.0.7 10.0.0.19
  10.0.0.20 10.0.0.4
)
# excluded (bad NIC): 10.0.0.21

SLIME_DIR="/path/to/workspace/code/slime-v0.3-harbor"
HARBOR_DIR="${SLIME_DIR}/examples/harbor_terminal_rl"
STARTER="${HARBOR_DIR}/start_tmax15k_full_96gpu_27b_daytona_ppo_64k_tmux.sh"
WRAPPER="${HARBOR_DIR}/run_tmax15k_qwen35_120gpu_ppo_criticlr30x_strongest.sh"
DAYTONA_ENV_FILE="${DAYTONA_ENV_FILE:-/path/to/workspace/.slime_run_env/daytona_relaunch_20260707.env}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/path/to/workspace/.slime_run_env/wandb_v2_retrain.env}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_qwen35_tmax15k_strongest}"

# --- run identity (strongest names; RUN_TAG-suffixed so it never collides) ------------
# The starter reads these via ${VAR:-default} at its line ~100 and then passes them to
# the wrapper, so setting them HERE overrides the starter's tmax15k_full defaults.
BASE_FOLDER="/path/to/workspace/slime_harbor_qwen35_27b_tmax15k_strongest_criticlr30x_${RUN_TAG}"
SAVE_SUBDIR="Qwen3.5-27B_tmax15k_strongest_criticlr30x_${RUN_TAG}"
RUN_TRIALS_DIR="/path/to/workspace/slime_harbor_trials/tmax15k_strongest_${RUN_TAG}"
RAY_TEMP_DIR="/tmp/rtmax15k35"
WANDB_GROUP="q35-27b-tmax15k-strongest-criticlr30x-actorlr2x-standard-os260-64k-n1"

# --- Topology env (shared by launcher + all workers) ----------------------------------
# PP16/CP2/TP2 + 8192 = the §Z 15-node backward-OOM fix. FT ON (evict dead engines).
read -r -d '' TOPO_ENV <<EOF || true
RUN_TAG=${RUN_TAG}
WRAPPER=${WRAPPER}
EXPECTED_NODES=15
ACTOR_NUM_NODES=8
ROLLOUT_NUM_GPUS=56
TENSOR_MODEL_PARALLEL_SIZE=2
PIPELINE_MODEL_PARALLEL_SIZE=16
CP_SIZE=2
MAX_TOKENS_PER_GPU=8192
USE_FAULT_TOLERANCE=1
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
echo " STRONGEST RUN  Qwen3.5-27B + tmax-15k  RUN_TAG=${RUN_TAG}"
echo " critic lr 30x(1.5e-5) / actor lr 2x(1e-6), standard os260, 100 steps"
echo " actor 8n(PP16xCP2xTP2,8192) + rollout 7n(56GPU) = 15n/120GPU"
echo " launcher=${LAUNCHER_IP}  workers=${#WORKER_IPS[@]}  (excl .137.206)"
echo " wrapper=${WRAPPER}"
echo " base=${BASE_FOLDER}"
echo "=============================================================="

# 1) launcher on THIS node (.137.106)
echo "[orch] starting LAUNCHER on ${LAUNCHER_IP} ..."
env $(echo "$TOPO_ENV" | tr '\n' ' ') bash "${STARTER}" launcher "${LAUNCHER_IP}"

# 2) workers over ssh (setsid so SIGHUP on the orchestrator can't kill the tmux dispatch)
for ip in "${WORKER_IPS[@]}"; do
  echo "[orch] starting WORKER on ${ip} ..."
  $SSH "${ip}" "setsid env $(echo "$TOPO_ENV" | tr '\n' ' ') bash '${STARTER}' worker '${LAUNCHER_IP}'" \
    && echo "[orch]   ${ip} worker launched" \
    || echo "[orch]   !! ${ip} worker FAILED to launch"
done

echo "[orch] all nodes dispatched. RUN_TAG=${RUN_TAG}"
echo "[orch] launcher log: ${BASE_FOLDER}/logs/launcher_${RUN_TAG}.log"
