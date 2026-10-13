#!/bin/bash
#
# launch_synth3k_120gpu_no206.sh
# ---------------------------------------------------------------------------
# Launch the synth-3k (2820-task) PPO run on 15 HEALTHY nodes (120 GPU), excluding
# the bad-NIC node .137.206. Topology is IDENTICAL to the proven tmax-15k 120-GPU
# actor side (the §Z backward-OOM fix):
#
#   actor   : 8 nodes = 64 GPU  TP2 x PP16 x CP2, max-tok 8192
#   rollout : 7 nodes = 56 GPU  28 SGLang engines
#   total   : 15 nodes = 120 GPU  (16 minus .137.206)
#
# WHAT'S NEW vs tmax-15k (all via env, no core changes):
#   - DATA: 2820 synth tasks (shared-fs symlink dir; NO /tmp staging).
#   - REWARD: CTRF process reward (pass-ratio), HARBOR_PARTIAL_REWARD_MAX=1.0.
#   - EPOCHS: NUM_ROLLOUT=22 (2 epochs over 2820 @ batch 256).
#   - CRITIC: warm-load 001109 iter_56 value head via load-root symlinks; actor
#             cold-starts from 27B base; NUM_CRITIC_ONLY_STEPS=0.
#   - SANDBOX: 1 CPU / 2 GB / 6 GB (override_* in the yaml).
#
# Fans out launcher + 14 workers over ssh from the launcher node (.137.106).
# The 27B base + 001109 critic load from SHARED FS (no /tmp staging), so there is
# no per-node ckpt replication step.
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
STARTER="${SLIME_DIR}/examples/harbor_terminal_rl/start_synth3k_2820_96gpu_27b_daytona_ppo_partial_64k_tmux.sh"
DAYTONA_ENV_FILE="/path/to/workspace/.slime_run_env/daytona_tier4.env"
WANDB_ENV_FILE="/path/to/workspace/.slime_run_env/wandb_v2_retrain.env"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

# --- Topology + experiment env (shared by launcher + all workers) ----------
# PP16/CP2/TP2 + 8192 = the §Z backward-OOM fix. rollout 7n=56 GPU. FT on.
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
NUM_ROLLOUT=22
CUSTOM_REWARD_POST_PROCESS_PATH=slime_plugins.harbor.partial_reward.trajectory_grouped_progress_rewards
HARBOR_PARTIAL_REWARD_MAX=1.0
HARBOR_LENGTH_REWARD_LAMBDA=0.0
NUM_CRITIC_ONLY_STEPS=0
STAGE_TORCH_DIST_CKPT_TO_TMP=0
SYNTH3K_LOAD_ROOT=/path/to/workspace/.slime_load_roots/synth3k_ppo
PRETRAINED_CKPT=/path/to/workspace/.slime_load_roots/synth3k_ppo/Qwen3.5-27B_base
HF_CKPT=/path/to/models/Qwen3.5-27B
DAYTONA_ENV_FILE=${DAYTONA_ENV_FILE}
WANDB_ENV_FILE=${WANDB_ENV_FILE}
HARBOR_DAYTONA_S3_PROXY_URL=http://star-proxy.oa.com:3128
HARBOR_CREATE_RATE_PER_SEC=6.0
EOF

SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10"

echo "=============================================================="
echo " Launch synth-3k (2820) PPO  RUN_TAG=${RUN_TAG}"
echo " actor 8n(PP16xCP2xTP2,8192) + rollout 7n(56GPU) = 15n/120GPU"
echo " reward=CTRF pass-ratio (max=1.0)  critic=warm 001109 iter56  epochs=2 (22 steps)"
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
echo "[orch] launcher log: /path/to/workspace/slime_harbor_qwen35_27b_synth3k_2820_ppo_partial_64k_${RUN_TAG}/logs/launcher_${RUN_TAG}.log"
