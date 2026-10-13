#!/usr/bin/env bash
# launch_27b_s2_resume_iter4.sh
#
# Stage S1 + S2 resume: 27B dense Terminal PPO 256k, restarted from run #2's
# iter_0000004 with turn censoring lifted (S1) and the bounded interpolated-CTRF
# reward (S2). Thin wrapper -- only the env that differs from run #2 is set here,
# so run_27b_dense_128gpu_ppo_256k.sh stays the single source of truth for
# topology / memory / proxy / keepalive.
#
# ============================== WHAT THIS CHANGES ==============================
#
# S1 · max_turns 60 -> 500  (carried by the YAML)
#   Run #2 measured 76.6% of trajectories truncated at 60 turns with an 84% failure
#   rate among them -- the cap was censoring the long-horizon tasks rather than
#   bounding them. Physical safety is still enforced by the layers that actually
#   protect the machine: agent override_timeout_sec 3600s and the 229376-token
#   compaction threshold (32k reserved inside the 256k wall), both unchanged.
#
# S2 · bounded outcome-dominant reward  (carried by the YAML + a parallel module)
#   B(tau) = Y + alpha*(1-Y)*AntiHack*q_H, q_H = min(1, sum W(c) / S(f)),
#   S(f) = max(6.0, 0.5*p75(f) + 0.5*p90(f)).
#   Selected via harbor_reward_mode: interpolated_ctrf. The legacy hook body is
#   untouched, so nothing about an in-flight run changes.
#
# value_clip 4.0 -> 0.08  (per-role YAML)
#   The reward is now hard-bounded to [0,1]; a 4.0 clip is ~50x wider than the whole
#   reward range. See megatron_roles_27b_s2_resume_iter4.yaml for the full rationale.
#
# ===================== RESUME: WEIGHTS *AND* OPTIMIZER STATE ====================
# The load roots are a PAIR, both pinned to iter_0000004 (verified: 130 files and
# 128 distrib_optim .distcp shards per side, common.pt carries iteration=4 and an
# 'optimizer' key on BOTH actor and critic):
#   checkpoints/q38_27b_s2_resume_iter4         -> actor
#   checkpoints/q38_27b_s2_resume_iter4_critic  -> critic  (placement_group.py:195
#                                                  redirects the critic's --load here)
#
# Three flags MUST NOT reach the training command:
#   --no-load-optim  would discard the Adam moment tensors
#   --no-load-rng    would break sampling-state continuity
#   --finetune       would reset the iteration counter / LR curve
#
# Two places would otherwise inject them, so both are overridden here:
#
#   (a) EXTRA_MISC_ARGS. run_27b_dense_128gpu_ppo_256k.sh:300 defaults it to
#       "--no-load-optim --no-load-rng" (correct for a FRESH PPO run off an SFT
#       checkpoint, wrong for resuming a PPO run into itself). Overridden to a
#       non-empty INERT value: it must not be empty, because the grandparent
#       resolves ${EXTRA_MISC_ARGS:-<moe default>} and `:-` substitutes on empty
#       too, which would re-inject --moe-token-dispatcher-type alltoall.
#       --moe-token-dispatcher-type alltoall is verified inert on this dense model
#       (MoELayer is never instantiated when num_experts is None).
#
#   (b) The WARM-CRITIC path. placement_group.py:215-222 sets
#       no_load_optim / no_load_rng / finetune = True on the critic whenever a
#       "<load>_critic" sibling exists, UNLESS --critic-load-optim is passed. That
#       default exists to avoid an OptimizerParamScheduler assert when the source
#       run's schedule differs -- but here we are resuming a run INTO ITSELF
#       (same num_rollout, same batch, same lr), which is exactly the case the flag
#       documents. CRITIC_LOAD_OPTIM=1 therefore restores the critic's moments too.
#
# =================== CRITIC WARMUP: WHY THE NUMBER IS NOT 2 ====================
# train_async.py:75 gates the actor on an ABSOLUTE rollout id:
#     actor_trains_this_step = rollout_id >= args.num_critic_only_steps
# Resuming from iter_0000004 makes start_rollout_id = 5, so
# --num-critic-only-steps 2 would yield ZERO warmup steps (5 >= 2 is already true
# on the first step). To get the requested TWO critic-only steps the value must be
# start_rollout_id + 2 = 7, which holds the actor frozen for rollout_id 5 and 6 and
# releases it at rollout_id 7. This is computed below rather than hard-coded so it
# stays correct if the resume point moves.
#
# Usage:
#   bash examples/harbor_terminal_rl/launch_27b_s2_resume_iter4.sh          # smoke (3 steps)
#   NUM_ROLLOUT=36 bash .../launch_27b_s2_resume_iter4.sh                   # full run

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_ROOT=/path/to/workspace

# ---- RESUME POINT -------------------------------------------------------------------
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-${USER_ROOT}/checkpoints/q38_27b_s2_resume_iter4}"
export WARM_CRITIC="${WARM_CRITIC:-1}"
RESUME_ITER="${RESUME_ITER:-4}"
# Megatron sets start_rollout_id = loaded_iteration + 1 for a numeric tracker.
export START_ROLLOUT_ID="${START_ROLLOUT_ID:-$(( RESUME_ITER + 1 ))}"

# ---- FULL OPTIMIZER / RNG / ITERATION RESTORE ---------------------------------------
# See (a) and (b) in the header. Inert non-empty value, NOT the empty string.
export EXTRA_MISC_ARGS="${EXTRA_MISC_ARGS:---moe-token-dispatcher-type alltoall}"
export CRITIC_LOAD_OPTIM="${CRITIC_LOAD_OPTIM:-1}"

# ---- 2-STEP CRITIC WARMUP (absolute-id gate, see header) ----------------------------
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-$(( START_ROLLOUT_ID + 2 ))}"

# ---- S1 + S2 (both carried by this config) ------------------------------------------
export CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/harbor_mixed3000_128gpu_27b_dense_daytona_ppo_cp4_256k_s2_interp.yaml}"

# The hook path MUST be redirected to the dispatcher. The grandparent hard-defaults it
# to ...partial_reward.test_count_progress_rewards (the LEGACY function), and slime
# resolves that dotted path directly -- so the yaml's `harbor_reward_mode` would never
# be consulted and the run would silently train on the legacy /82 reward.
# Caught in the first smoke attempt: preflight printed
#   [preflight] reward = ...partial_reward.test_count_progress_rewards (SCALE=20)
# i.e. neither the S2 mode NOR the per-family table was in play.
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.partial_reward.dispatch_rewards}"
# Belt-and-braces for a future DIRECT_RUN=1 invocation (which DOES inherit this shell's
# env). Under `ray job submit` this is dropped by the RUNTIME_ENV_JSON whitelist, which
# is exactly why the authoritative switch lives in the yaml.
export HARBOR_REWARD_MODE="${HARBOR_REWARD_MODE:-interpolated_ctrf}"

# ---- per-role lr 30x + recalibrated critic value_clip --------------------------------
export MEGATRON_CONFIG_PATH="${MEGATRON_CONFIG_PATH:-${SCRIPT_DIR}/megatron_roles_27b_s2_resume_iter4.yaml}"

# ---- PPO clipping: unchanged from run #2 --------------------------------------------
export EPS_CLIP="${EPS_CLIP:-0.3}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-5.0}"

# ---- TOPOLOGY: 64 train + 56 infer on the surviving 15 nodes ------------------------
export NODE_IP_LIST="${NODE_IP_LIST:-$(cat /tmp/node_ip_list_15.txt 2>/dev/null || echo "")}"
export EXPECTED_NODES="${EXPECTED_NODES:-15}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-8}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-56}"

# ---- RUN LENGTH: SMOKE BY DEFAULT ---------------------------------------------------
# Deliberately 3, not 36. This wrapper is meant to prove the resume + warmup + reward
# chain end to end; the operator raises NUM_ROLLOUT explicitly for the production run.
export NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export RUN_TAG="${RUN_TAG:-S2RESUME_$(date +%Y%m%d_%H%M%S)}"
export WANDB_PROJECT="${WANDB_PROJECT:-terminal3_rl}"
export WANDB_GROUP="${WANDB_GROUP:-q38-27b-dense-ppo-s2-interp-resume4}"

echo "[s2] ================== STAGE S1+S2 RESUME =================="
echo "[s2] resume from  = ${PRETRAINED_CKPT} (iter ${RESUME_ITER})"
echo "[s2] critic ckpt  = ${PRETRAINED_CKPT}_critic (WARM_CRITIC=${WARM_CRITIC}, CRITIC_LOAD_OPTIM=${CRITIC_LOAD_OPTIM})"
echo "[s2] start_rollout_id   = ${START_ROLLOUT_ID}"
echo "[s2] num_critic_only    = ${NUM_CRITIC_ONLY_STEPS}  => actor frozen at rollout_id ${START_ROLLOUT_ID},$(( START_ROLLOUT_ID + 1 )); unfreezes at $(( START_ROLLOUT_ID + 2 ))"
echo "[s2] optimizer restore  = EXTRA_MISC_ARGS='${EXTRA_MISC_ARGS}' (no --no-load-optim / --no-load-rng / --finetune)"
echo "[s2] reward mode        = interpolated_ctrf (S2, bounded [0,1])"
echo "[s2] config             = ${CONFIG_TEMPLATE}"
echo "[s2] per-role cfg       = ${MEGATRON_CONFIG_PATH}"
echo "[s2] steps              = ${NUM_ROLLOUT} (save every ${SAVE_INTERVAL})"
echo "[s2] ========================================================"

exec bash "${SCRIPT_DIR}/run_27b_dense_128gpu_ppo_256k.sh"
