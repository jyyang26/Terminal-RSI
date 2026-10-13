#!/usr/bin/env bash
# launch_27b_tbgae_s3_smoke.sh
#
# CONTROLLED GPU SMOKE for TB-GAE (S3: gamma=lambda=1.0). NOT a production run.
#
# ================================ WHY THIS IS A SMOKE, NOT PRODUCTION ================================
# As of 2026-09-29 the TB-GAE truncation Value-Bootstrap + cross-chunk continuation are IMPLEMENTED
# and pass CPU/Gloo tests (see TB_GAE_IMPLEMENTATION_REPORT_20260927.md section 7), and the two
# earlier code-level blockers are gone:
#   - truncated (max_turns / timeout / cancelled / context_length) trajectories are no longer rejected;
#     they enter the batch with is_real_done=False and a real successor-state old-critic value.
#   - multi-chunk (retokenized split / context-cap) trajectories recurse GAE across DP.
# BUT the report itself states (7.1): NO production cluster and NO 27B/256k/PP2 GPU run has happened.
# The entire new DISTRIBUTED path has zero GPU execution:
#   - an extra 256k old-critic bootstrap forward (tb_gae_bootstrap.py::compute_bootstrap_values),
#   - a cross-DP MAX count align + boundary gather (tb_gae_boundary.py),
#   - a CP=4 scalar all_gather of the successor value,
#   - the Terminus-2 v2 producer emitting valid bootstrap_token_ids under REAL Daytona timeout/cancel.
# These are exactly the paths that fail first time (shape/rank, OOM from the extra forward, collective
# hang, or the producer not emitting a valid successor state). Hence: SMOKE FIRST, then scale.
#
# ================================ WHAT THIS SMOKE DOES ================================
# Reuses the proven RUN2 warm-critic recipe (converted SFT iter263 + warm critic sibling + original
# CTRF reward @ SCALE 82) so the bootstrap forward runs against a CALIBRATED value head (a cold critic
# would make bootstrap values meaningless and hide whether the numbers are sane). Overlays only:
#   - the TB-GAE S3 role yaml (enable_tb_gae:true, gamma=lambda=1.0),
#   - --enable-tb-gae on the CLI (belt-and-braces with the yaml),
#   - NUM_CRITIC_ONLY_STEPS=0 so the actor + TB path runs from the FIRST step (warm critic allows it),
#   - a SHORT run (NUM_ROLLOUT=3) that does NOT save (SAVE_INTERVAL huge), so no disk / no checkpoints.
# Everything else (topology, data, memory) is the unchanged RUN2 path.
#
# ================================ WHAT TO WATCH (pass/fail of the smoke) ================================
#   START     : job reaches training; no import/validate error; preflight shows enable_tb_gae true.
#   PRODUCER  : trials that hit max_turns record termination_kind=truncated WITH bootstrap_token_ids
#               (grep the trial dumps / logs). If you see "missing a valid bootstrap state" the
#               Terminus-2 v2 producer or encode_bootstrap_state is not firing.
#   NO DROP   : failed=0 / missing=0 for truncated trials (they must ENTER the batch, not be dropped).
#   BOOTSTRAP : the extra old-critic forward completes without OOM/hang; bootstrap values finite.
#   GRADIENT  : actor and critic produce finite grad_norm and a joint optimizer.step at step 0.
#   MEMORY    : the extra 256k bootstrap forward does NOT push the last-PP-stage over the H20 budget.
# If any of these fail, that is the smoke doing its job — fix before scaling to mixed3000.
#
# ================================ AFTER THE SMOKE PASSES ================================
# Only then scale up: raise NUM_ROLLOUT, restore SAVE_INTERVAL=5, and decide warmup. Do NOT jump
# straight to a 100-step mixed3000 run off an unproven distributed path.
#
# Usage (multi-node cluster, RUN2 prerequisites must exist):
#   bash examples/harbor_terminal_rl/launch_27b_tbgae_s3_smoke.sh
#   NUM_ROLLOUT=2 bash examples/harbor_terminal_rl/launch_27b_tbgae_s3_smoke.sh   # even shorter

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_ROOT=/path/to/workspace

# ---- RUN2 warm-critic load root (calibrated value head for meaningful bootstrap values) ----
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-${USER_ROOT}/checkpoints/q38_27b_ppo256k_run2_loadroot}"
export WARM_CRITIC="${WARM_CRITIC:-1}"
export START_ROLLOUT_ID="${START_ROLLOUT_ID:-0}"

# ---- No warmup: exercise the actor + TB path from step 0 (critic is already warm) ----
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-0}"

# ---- Reward: RUN2 original (TB-GAE changes the ADVANTAGE, not the reward) ----
export CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/harbor_mixed3000_128gpu_27b_dense_daytona_ppo_cp4_256k_run2_ctrf82.yaml}"
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.partial_reward.test_count_progress_rewards}"
unset HARBOR_REWARD_MODE
export HARBOR_TESTCOUNT_SCALE="${HARBOR_TESTCOUNT_SCALE:-82}"

# ---- TB-GAE S3 role yaml (enable_tb_gae:true, gamma=lambda=1.0, warm-critic per-role block) ----
export MEGATRON_CONFIG_PATH="${MEGATRON_CONFIG_PATH:-${SCRIPT_DIR}/megatron_roles_27b_tbgae_s3.yaml}"

# ---- Load-side flags + explicit TB-GAE enable (yaml already sets it; CLI is belt-and-braces) ----
# RUN2 converted checkpoint has no compatible optimizer state -> optimizer/RNG not loaded (still built
# fresh). --enable-tb-gae is redundant with the yaml but makes the switch visible in the args dump.
export EXTRA_MISC_ARGS="${EXTRA_MISC_ARGS:---no-load-optim --no-load-rng --enable-tb-gae --tb-gae-gamma-turn 1.0 --tb-gae-lambda-turn 1.0}"

# ---- PPO clipping: unchanged from RUN2 ----
export EPS_CLIP="${EPS_CLIP:-0.3}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-5.0}"

# ---- Topology: unchanged 64 train + 56 infer ----
export NODE_IP_LIST="${NODE_IP_LIST:-$(cat /tmp/node_ip_list_15.txt 2>/dev/null || echo "")}"
export EXPECTED_NODES="${EXPECTED_NODES:-15}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-8}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-56}"

# ---- SHORT run, NO saves (smoke) ----
export NUM_ROLLOUT="${NUM_ROLLOUT:-3}"          # END INDEX -> steps 0,1,2
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"   # effectively never saves during the smoke
export RUN_TAG="${RUN_TAG:-TBGAE_S3_SMOKE_$(date +%Y%m%d_%H%M%S)}"
export WANDB_PROJECT="${WANDB_PROJECT:-terminal3_rl}"
export WANDB_GROUP="${WANDB_GROUP:-q38-27b-dense-ppo-tbgae-s3-smoke}"

echo "[tbgae-smoke] ============ TB-GAE S3 GPU SMOKE (NOT production) ============"
echo "[tbgae-smoke] load root   = ${PRETRAINED_CKPT} (warm critic) + _critic sibling"
echo "[tbgae-smoke] warmup      = ${NUM_CRITIC_ONLY_STEPS} critic-only steps (0 => actor+TB from step 0)"
echo "[tbgae-smoke] reward      = RUN2 original CTRF scale ${HARBOR_TESTCOUNT_SCALE} (TB changes advantage, not reward)"
echo "[tbgae-smoke] TB role cfg = ${MEGATRON_CONFIG_PATH} (S3: gamma=lambda=1.0)"
echo "[tbgae-smoke] misc flags  = ${EXTRA_MISC_ARGS}"
echo "[tbgae-smoke] num_rollout = ${NUM_ROLLOUT} (SHORT smoke; save_interval=${SAVE_INTERVAL} => no ckpt)"
echo "[tbgae-smoke] WATCH: truncated trials carry bootstrap_token_ids; failed=0/missing=0;"
echo "[tbgae-smoke]        bootstrap forward no OOM/hang; step0 finite actor+critic grad_norm."
echo "[tbgae-smoke] This is a first-execution smoke of an all-CPU-tested distributed path."
echo "[tbgae-smoke] ==============================================================="

exec bash "${SCRIPT_DIR}/run_27b_dense_128gpu_ppo_256k.sh"
