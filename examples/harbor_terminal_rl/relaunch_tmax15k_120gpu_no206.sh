#!/bin/bash
#
# relaunch_tmax15k_120gpu_no206.sh
# ---------------------------------------------------------------------------
# Relaunch the tmax-15k FULL PPO run on 15 HEALTHY nodes (120 GPU) after the
# 060921 run died from bad NIC on .137.206 (eth8 rx_err=8930; see
# PPO_INTEGRATION_NOTES.md §Z.2). Topology is IDENTICAL to 060921 on the actor
# side (the §Z backward-OOM fix), only rollout is trimmed 8->7 nodes and the
# bad node is excluded:
#
#   actor   : 8 nodes = 64 GPU  TP2 x PP16 x CP2, max-tok 8192   (UNCHANGED)
#   rollout : 7 nodes = 56 GPU  28 SGLang engines
#   total   : 15 nodes = 120 GPU  (16 minus .137.206)
#
# Fans out launcher + 14 workers over ssh from the launcher node (.137.106).
# model-cache + tmax data are ALREADY on every node's /tmp (verified), so no
# re-staging happens (the starter's stage step is idempotent and will skip).
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
STARTER="${SLIME_DIR}/examples/harbor_terminal_rl/start_tmax15k_full_96gpu_27b_daytona_ppo_64k_tmux.sh"
DAYTONA_ENV_FILE="/path/to/workspace/.slime_run_env/daytona_tier4.env"
WANDB_ENV_FILE="/path/to/workspace/.slime_run_env/wandb_v2_retrain.env"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

# --- Topology env (shared by launcher + all workers) -----------------------
# PP16/CP2/TP2 + 8192 = the §Z backward-OOM fix, untouched from 060921.
# Only ROLLOUT_NUM_GPUS 64->56 and EXPECTED_NODES 16->15 changed.
# USE_FAULT_TOLERANCE=1 (NEW, see PPO_INTEGRATION_NOTES §8): all 3 prior runs
# died because a dead member (OOM rank / bad-NIC node / crashed SGLang engine)
# was never evicted from the fixed weight-sync client set, so the rendezvous
# hung 30min then DistStoreError'd the job. FT starts RolloutHealthMonitor
# (health-check every 30s -> kill dead engine -> mark None) and calls
# recover_updatable_engines() before each update_weights (restarts dead engines
# + rebuilds the NCCL group). This directly fixes the 115515 engine-crash death.
read -r -d '' TOPO_ENV <<EOF || true
RUN_TAG=${RUN_TAG}
EXPECTED_NODES=15
ACTOR_NUM_NODES=8
ROLLOUT_NUM_GPUS=56
TENSOR_MODEL_PARALLEL_SIZE=2
PIPELINE_MODEL_PARALLEL_SIZE=16
CP_SIZE=2
MAX_TOKENS_PER_GPU=8192
USE_FAULT_TOLERANCE=1
DAYTONA_ENV_FILE=${DAYTONA_ENV_FILE}
WANDB_ENV_FILE=${WANDB_ENV_FILE}
HARBOR_DAYTONA_S3_PROXY_URL=http://star-proxy.oa.com:3128
EOF

SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10"

echo "=============================================================="
echo " Relaunch tmax-15k FULL PPO  RUN_TAG=${RUN_TAG}"
echo " actor 8n(PP16xCP2xTP2,8192) + rollout 7n(56GPU) = 15n/120GPU"
echo " launcher=${LAUNCHER_IP}  workers=${#WORKER_IPS[@]}  (excl .137.206)"
echo "=============================================================="

# 1) launcher on THIS node (.137.106)
echo "[orch] starting LAUNCHER on ${LAUNCHER_IP} ..."
env $(echo "$TOPO_ENV" | tr '\n' ' ') bash "${STARTER}" launcher "${LAUNCHER_IP}"

# 2) workers over ssh
for ip in "${WORKER_IPS[@]}"; do
  echo "[orch] starting WORKER on ${ip} ..."
  $SSH "${ip}" "env $(echo "$TOPO_ENV" | tr '\n' ' ') bash '${STARTER}' worker '${LAUNCHER_IP}'" \
    && echo "[orch]   ${ip} worker launched" \
    || echo "[orch]   !! ${ip} worker FAILED to launch"
done

echo "[orch] all nodes dispatched. RUN_TAG=${RUN_TAG}"
echo "[orch] launcher log: /path/to/workspace/slime_harbor_qwen35_27b_tmax15k_full_ppo_64k_${RUN_TAG}/logs/launcher_${RUN_TAG}.log"
