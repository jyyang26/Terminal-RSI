#!/bin/bash
#
# launch_tmax15k_qwen35_strongest_b384_pinnedspec.sh
# ---------------------------------------------------------------------------
# FRESH launch (NEW RUN_TAG, from scratch -- NO resume, NO --load) of the STRONGEST
# run with two deltas vs the prior strongest run:
#   1) BATCH 256 -> 384 (wrapper: ROLLOUT/GLOBAL=384, oversample 390).
#   2) PINNED SANDBOX SPEC: every Daytona sandbox clamped to 1 CPU / 2 G / 6 G
#      (new CONFIG_TEMPLATE below), in-flight cap + trial semaphore 280 -> 400.
# Because it is a fresh RUN_TAG, BASE_FOLDER/SAVE_SUBDIR are brand-new dirs -- the prior
# run's iter4/9/14 checkpoints live under a DIFFERENT tag and cannot be overwritten.
#
# TOPOLOGY: 15 nodes / 120 GPU (excl ONLY .206 bad-NIC; .219 restarted-healthy, .188 re-staged):
#   actor   : 8 nodes = 64 GPU  TP2 x PP16 x CP2, max-tok 8192 (the §Z backward-OOM fix)
#   rollout : 7 nodes = 56 GPU
#
# The experiment knobs live in the WRAPPER (self-exported); this script owns TOPOLOGY +
# run IDENTITY + creds + the NEW CONFIG_TEMPLATE. WRAPPER + CONFIG_TEMPLATE are passed via
# the starter's env (no starter edit). We are ON the 019 head node -> plain ssh fan-out.
set -euo pipefail

LAUNCHER_IP="10.0.0.11"
WORKER_IPS=(
  10.0.0.3 10.0.0.12 10.0.0.8 10.0.0.13
  10.0.0.14 10.0.0.15 10.0.0.9 10.0.0.16
  10.0.0.17 10.0.0.18 10.0.0.7 10.0.0.19
  10.0.0.20 10.0.0.4
)
# excluded: 10.0.0.21 (bad NIC) ONLY.
#   .219 was restarted (verified healthy: 8 idle GPUs, weights staged).
#   .188 was re-staged node-to-node from .202 (~625 MB/s, ~18x the shared-FS speed).

SLIME_DIR="/path/to/workspace/code/slime-v0.3-harbor"
HARBOR_DIR="${SLIME_DIR}/examples/harbor_terminal_rl"
STARTER="${HARBOR_DIR}/start_tmax15k_full_96gpu_27b_daytona_ppo_64k_tmux.sh"
WRAPPER="${HARBOR_DIR}/run_tmax15k_qwen35_120gpu_ppo_criticlr30x_b384_pinnedspec.sh"
# NEW template: pinned sandbox spec (1/2/6) + concurrency 400. The starter seds trials_dir
# into a per-run copy and forwards it as CONFIG (overriding the wrapper's CONFIG default),
# so passing CONFIG_TEMPLATE here is REQUIRED -- otherwise the old 280-concurrency,
# unpinned-CPU/RAM yaml would be used.
CONFIG_TEMPLATE="${HARBOR_DIR}/harbor_tmax15k_strongest_b384_pinnedspec_64k.yaml"
DAYTONA_ENV_FILE="${DAYTONA_ENV_FILE:-/path/to/workspace/.slime_run_env/daytona_relaunch_20260707.env}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/path/to/workspace/.slime_run_env/wandb_v2_retrain.env}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_qwen35_tmax15k_b384_pinnedspec}"

# --- run identity (fresh names; RUN_TAG-suffixed so it never collides) -----------------
BASE_FOLDER="/path/to/workspace/slime_harbor_qwen35_27b_tmax15k_b384_pinnedspec_${RUN_TAG}"
SAVE_SUBDIR="Qwen3.5-27B_tmax15k_b384_pinnedspec_${RUN_TAG}"
RUN_TRIALS_DIR="/path/to/workspace/slime_harbor_trials/tmax15k_b384_pinnedspec_${RUN_TAG}"
RAY_TEMP_DIR="/tmp/rtmax15k35b384"
WANDB_GROUP="q35-27b-tmax15k-strongest-criticlr30x-actorlr2x-standard-b384-pinnedspec-64k-n1"

# --- Topology env: 15 nodes / 120 GPU (actor 8n/64 + rollout 7n/56) --------------------
EXPECTED_NODES=${EXPECTED_NODES:-15}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-56}
read -r -d '' TOPO_ENV <<EOF || true
RUN_TAG=${RUN_TAG}
WRAPPER=${WRAPPER}
CONFIG_TEMPLATE=${CONFIG_TEMPLATE}
EXPECTED_NODES=${EXPECTED_NODES}
ACTOR_NUM_NODES=8
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS}
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
echo " STRONGEST b384+pinnedspec  Qwen3.5-27B + tmax-15k  RUN_TAG=${RUN_TAG}"
echo " batch 384 / oversample 390, sandbox 1CPU/2G/6G, concurrency 400"
echo " critic lr 30x(1.5e-5) / actor lr 2x(1e-6), standard, 100 steps"
echo " actor 8n(PP16xCP2xTP2,8192) + rollout 7n(56GPU) = 15n/120GPU"
echo " launcher=${LAUNCHER_IP}  workers=${#WORKER_IPS[@]}  (excl .206 only)"
echo " wrapper=${WRAPPER}"
echo " config_template=${CONFIG_TEMPLATE}"
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

echo "[orch] all nodes dispatched (FRESH). RUN_TAG=${RUN_TAG}"
echo "[orch] launcher log: ${BASE_FOLDER}/logs/launcher_${RUN_TAG}.log"
