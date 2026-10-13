#!/usr/bin/env bash
# launch_27b_sc_iter263_crit34.sh
#
# PPO + Score-Centering (SC) ONLY, on SWR3000 (mixed3000), BatchSize 256.
#   actor  start = checkpoints/q38_27b_iter263_torch_dist  (converted SFT base, marker "release")
#   critic start = RESUME19_CURRICULUM .../mixed3000_critic/iter_0000034  (WARM value head)
#
# This is the "SC only" branch of the SC-vs-TB-GAE decision: TB-GAE is OFF (mutually
# exclusive with SC, and its GPU training path is still unproven). See
# SC_TBGAE_IMPLEMENTATION_REPORT_20260929.md for why both-on is unsafe.
#
# =============================== LOAD-ROOT / start_rollout_id ===============================
# CRITICAL: the actor load root MUST carry the literal marker "release" in its
# latest_checkpointed_iteration.txt. arguments.py only forces start_rollout_id=0 when the
# marker == "release" (slime/utils/arguments.py ~1826); a NUMERIC marker (e.g. the older
# run2_loadroot's "263") instead yields start_rollout_id = loaded_iteration+1, which makes
# train_async.py's range(start_rollout_id, num_rollout) EMPTY -> silent no-train. The 27B
# chain does not pass --start-rollout-id, so the marker is the only control.
#
# The dedicated pair below (built 2026-09-29) uses "release" markers so start_rollout_id=0:
#   q38_27b_sc_iter263_crit34_loadroot/release        -> q38_27b_iter263_torch_dist/release
#   q38_27b_sc_iter263_crit34_loadroot_critic/release -> RESUME19 .../iter_0000034
# WARM_CRITIC=1 is honoured because the "_critic" sibling exists (placement_group.py:195
# redirects the critic --load to "<load>_critic").
#
# =============================== SMOKE vs FORMAL ===============================
# Defaults here are the FORMAL run (batch 256, critic-only 2, saves every 5).
# For the gating SMOKE, override on the CLI (see the two invocations at the bottom).
#
# Usage:
#   # SMOKE (reduced batch, no-save, actor trains from step 0 so SC is visible immediately):
#   ROLLOUT_BATCH_SIZE=32 GLOBAL_BATCH_SIZE=32 OVER_SAMPLING_BATCH_SIZE=40 \
#     NUM_CRITIC_ONLY_STEPS=0 NUM_ROLLOUT=3 SAVE_INTERVAL=1000 \
#     RUN_TAG=SC_ITER263_CRIT34_SMOKE_$(date +%Y%m%d_%H%M%S) \
#     bash examples/harbor_terminal_rl/launch_27b_sc_iter263_crit34.sh
#
#   # FORMAL (batch 256, warm-critic recalibration for 2 steps, saves):
#   bash examples/harbor_terminal_rl/launch_27b_sc_iter263_crit34.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_ROOT=/path/to/workspace

# ---- LOAD ROOT: iter263 base actor + iter34 warm critic (release markers) ------------
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-${USER_ROOT}/checkpoints/q38_27b_sc_iter263_crit34_loadroot}"
export WARM_CRITIC="${WARM_CRITIC:-1}"
export START_ROLLOUT_ID="${START_ROLLOUT_ID:-0}"

# ---- CRITIC WARMUP: 2 steps to recalibrate the warm iter34 value head to the fresh -----
# iter263 actor policy before the actor joins (RUN2 default). Set 0 for the smoke so SC
# metrics appear at rollout_id 0.
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-2}"

# ---- REWARD: RUN2 original CTRF progress reward (unchanged by SC) --------------------
export CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/harbor_mixed3000_128gpu_27b_dense_daytona_ppo_cp4_256k_run2_ctrf82.yaml}"
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.partial_reward.test_count_progress_rewards}"
unset HARBOR_REWARD_MODE
export HARBOR_TESTCOUNT_SCALE="${HARBOR_TESTCOUNT_SCALE:-82}"

# ---- SC: the ONLY behavioural change vs plain PPO. Per-role yaml carries the SC block. -
export MEGATRON_CONFIG_PATH="${MEGATRON_CONFIG_PATH:-${SCRIPT_DIR}/megatron_roles_27b_criticlr30x_warmcritic_sc_active.yaml}"

# ---- LOAD-SIDE FLAGS: fresh optimizer (actor base has none compatible) + TB-GAE off ---
# Optimizer/RNG are NOT loaded (built fresh, still SAVED -> NO_SAVE_OPTIM stays 0).
export EXTRA_MISC_ARGS="${EXTRA_MISC_ARGS:---no-load-optim --no-load-rng --no-enable-tb-gae --save-keep-last 4}"

# ---- PPO clipping: unchanged from RUN2 ----------------------------------------------
export EPS_CLIP="${EPS_CLIP:-0.3}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-5.0}"

# ---- TOPOLOGY: 64 train + 56 infer on the 15 surviving nodes ------------------------
export NODE_IP_LIST="${NODE_IP_LIST:-$(cat /tmp/node_ip_list_15.txt 2>/dev/null || echo "")}"
export EXPECTED_NODES="${EXPECTED_NODES:-15}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-8}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-56}"

# ---- BATCH: 256 (the 27B chain default; explicit here for the record) ----------------
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-288}"

# ---- RUN LENGTH + IDENTITY ----------------------------------------------------------
export NUM_ROLLOUT="${NUM_ROLLOUT:-40}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export RUN_TAG="${RUN_TAG:-SC_ITER263_CRIT34_$(date +%Y%m%d_%H%M%S)}"
export WANDB_PROJECT="${WANDB_PROJECT:-terminal3_rl}"
export WANDB_GROUP="${WANDB_GROUP:-q38-27b-dense-ppo-sc-iter263-crit34}"

echo "[sc-i263c34] ============ SC-ONLY PPO: iter263 actor + iter34 WARM critic ============"
echo "[sc-i263c34] actor load  = ${PRETRAINED_CKPT}/release (marker $(cat "${PRETRAINED_CKPT}/latest_checkpointed_iteration.txt" 2>/dev/null)) -> start_rollout_id 0"
echo "[sc-i263c34] critic load = ${PRETRAINED_CKPT}_critic/release (RESUME19 iter34, WARM)"
echo "[sc-i263c34] warm critic = ${WARM_CRITIC}; critic-only steps = ${NUM_CRITIC_ONLY_STEPS} (SC metrics appear at rollout_id >= this)"
echo "[sc-i263c34] reward      = RUN2 original CTRF, scale ${HARBOR_TESTCOUNT_SCALE} (unchanged by SC)"
echo "[sc-i263c34] SC role cfg = ${MEGATRON_CONFIG_PATH}"
echo "[sc-i263c34] SC params   = score / json / k32 / scale 1.0  (from the role yaml)"
echo "[sc-i263c34] TB-GAE      = OFF (--no-enable-tb-gae; mutually exclusive with SC)"
echo "[sc-i263c34] batch       = ${GLOBAL_BATCH_SIZE} (rollout ${ROLLOUT_BATCH_SIZE}, oversample ${OVER_SAMPLING_BATCH_SIZE})"
echo "[sc-i263c34] num_rollout = ${NUM_ROLLOUT} (END INDEX => steps 0..$(( NUM_ROLLOUT - 1 )))"
echo "[sc-i263c34] save_interval = ${SAVE_INTERVAL} (~1059 GB per save, NO rotation)"
echo "[sc-i263c34] confirm SC  : grep 'sc_applied' in the run log; expect 1.0 once actor trains"
echo "[sc-i263c34] ========================================================================"

exec bash "${SCRIPT_DIR}/run_27b_dense_128gpu_ppo_256k.sh"
