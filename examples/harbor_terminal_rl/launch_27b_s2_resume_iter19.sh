#!/usr/bin/env bash
# launch_27b_s2_resume_iter19.sh
#
# Stage S1 + S2 PRODUCTION run: 27B dense Terminal PPO 256k, resumed from run #2's
# iter_0000019 with turn censoring lifted (S1) and the bounded interpolated-CTRF
# reward at alpha=0.50 (S2). Thin wrapper -- only the env that differs from run #2 is
# set here, so run_27b_dense_128gpu_ppo_256k.sh stays the single source of truth for
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
# S2 · bounded outcome-dominant reward at alpha=0.50  (YAML + parallel module)
#   B(tau) = Y + 0.50*(1-Y)*AntiHack*q_H, q_H = min(1, sum W(c) / S(f)),
#   S(f) = max(6.0, 0.5*p75(f) + 0.5*p90(f)).
#   Success/failure step is a rigid 1 - alpha = 0.50, so a fully-saturated unsolved
#   trajectory can never tie the weakest genuine solve. Selected via
#   harbor_reward_mode: interpolated_ctrf. The legacy hook body is untouched.
#
# value_clip 4.0 -> 0.08  (per-role YAML)
#   The reward is now hard-bounded to [0,1]; a 4.0 clip is ~50x wider than the whole
#   reward range. See megatron_roles_27b_s2_resume_iter19.yaml for the full rationale.
#
# ===================== RESUME: WEIGHTS *AND* OPTIMIZER STATE ====================
# The load roots are a PAIR, both pinned to iter_0000019. VERIFIED: 130 files and 128
# distrib_optim .distcp shards per side; common.pt carries iteration=19 plus
# 'optimizer' and 'opt_param_scheduler' keys on BOTH actor and critic; the DCP
# metadata exposes 48 optimizer entries incl. 16 exp_avg tensors on each side.
#   checkpoints/q38_27b_s2_resume_iter19         -> actor
#   checkpoints/q38_27b_s2_resume_iter19_critic  -> critic  (placement_group.py:195
#                                                   redirects the critic's --load here)
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
#       (same batch, same lr), which is exactly the case the flag documents.
#       CRITIC_LOAD_OPTIM=1 therefore restores the critic's moments too.
#
# NOTE: one "successfully loaded checkpoint ... at iteration 0" line is EXPECTED and
# benign -- it is the reference-model pass. --ref-load points at the same directory,
# and actor.py:688-693 load_other_checkpoint locally forces no_load_optim/no_load_rng/
# finetune for that pass only, then restores them. The actor's own resume is the
# "at iteration 19" line.
#
# =================== CRITIC WARMUP: WHY THE NUMBER IS NOT 2 ====================
# train_async.py:75 gates the actor on an ABSOLUTE rollout id:
#     actor_trains_this_step = rollout_id >= args.num_critic_only_steps
# Resuming from iter_0000019 makes start_rollout_id = 20, so
# --num-critic-only-steps 2 would yield ZERO warmup steps (20 >= 2 is already true on
# the first step). To get the requested TWO critic-only steps the value must be
# start_rollout_id + 2 = 22, which holds the actor frozen for rollout_id 20 and 21 and
# releases it at rollout_id 22. This is computed below rather than hard-coded so it
# stays correct if the resume point moves.
#
# ========================= NUM_ROLLOUT IS AN END INDEX =========================
# train_async.py:64 loops `range(args.start_rollout_id, args.num_rollout)`, so
# num_rollout is an ABSOLUTE END INDEX, not a step count. NUM_ROLLOUT=100 from
# start_rollout_id=20 therefore executes 80 steps (rollout_id 20..99), not 100.
# NUM_ROLLOUT=120 would be the value for 100 *additional* steps. 100 is used here as
# instructed -- see the disk note below for why more is not currently affordable.
#
# =============================== DISK BUDGET ===================================
# There is NO checkpoint rotation anywhere in this codebase (no keep-last-N); every
# save persists until deleted by hand. Each save is ~1059 GB (actor 538 + critic 521).
# With SAVE_INTERVAL=5 the save trigger is (rollout_id+1) % 5 == 0 plus a forced final
# save at rollout_id == num_rollout-1, i.e. rollout_id 24,29,...,94,99 = 16 saves
# = ~16.6 TB against ~6.9 TB free. The run WILL exhaust the filesystem around save #6
# (near rollout_id 49) unless checkpoints are pruned as it goes. Either raise
# SAVE_INTERVAL or prune older iters during the run.
#
# Usage:
#   bash examples/harbor_terminal_rl/launch_27b_s2_resume_iter19.sh          # 80 steps (20..99)
#   NUM_ROLLOUT=30 bash .../launch_27b_s2_resume_iter19.sh                   # short: 20..29

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_ROOT=/path/to/workspace

# ---- RESUME POINT -------------------------------------------------------------------
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-${USER_ROOT}/checkpoints/q38_27b_s2_resume_iter19}"
export WARM_CRITIC="${WARM_CRITIC:-1}"
RESUME_ITER="${RESUME_ITER:-19}"
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
# Caught in the iter4 smoke attempt: preflight printed
#   [preflight] reward = ...partial_reward.test_count_progress_rewards (SCALE=20)
# i.e. neither the S2 mode NOR the per-family table was in play.
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.partial_reward.dispatch_rewards}"
# Belt-and-braces for a future DIRECT_RUN=1 invocation (which DOES inherit this shell's
# env). Under `ray job submit` this is dropped by the RUNTIME_ENV_JSON whitelist, which
# is exactly why the authoritative switch lives in the yaml.
export HARBOR_REWARD_MODE="${HARBOR_REWARD_MODE:-interpolated_ctrf}"

# ---- per-role lr 30x + recalibrated critic value_clip --------------------------------
export MEGATRON_CONFIG_PATH="${MEGATRON_CONFIG_PATH:-${SCRIPT_DIR}/megatron_roles_27b_s2_resume_iter19.yaml}"

# ---- PPO clipping: unchanged from run #2 --------------------------------------------
export EPS_CLIP="${EPS_CLIP:-0.3}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-5.0}"

# ---- TOPOLOGY: 64 train + 56 infer on the surviving 15 nodes ------------------------
export NODE_IP_LIST="${NODE_IP_LIST:-$(cat /tmp/node_ip_list_15.txt 2>/dev/null || echo "")}"
export EXPECTED_NODES="${EXPECTED_NODES:-15}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-8}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-56}"

# ---- RUN LENGTH ---------------------------------------------------------------------
# End index, not a count: 20..99 = 80 executed steps. See header.
export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export RUN_TAG="${RUN_TAG:-S2PROD19_$(date +%Y%m%d_%H%M%S)}"
export WANDB_PROJECT="${WANDB_PROJECT:-terminal3_rl}"
export WANDB_GROUP="${WANDB_GROUP:-q38-27b-dense-ppo-s2-interp-resume19}"

echo "[s2] ============== STAGE S1+S2 PRODUCTION (resume iter19) =============="
echo "[s2] resume from  = ${PRETRAINED_CKPT} (iter ${RESUME_ITER})"
echo "[s2] critic ckpt  = ${PRETRAINED_CKPT}_critic (WARM_CRITIC=${WARM_CRITIC}, CRITIC_LOAD_OPTIM=${CRITIC_LOAD_OPTIM})"
echo "[s2] start_rollout_id   = ${START_ROLLOUT_ID}"
echo "[s2] num_critic_only    = ${NUM_CRITIC_ONLY_STEPS}  => actor frozen at rollout_id ${START_ROLLOUT_ID},$(( START_ROLLOUT_ID + 1 )); unfreezes at $(( START_ROLLOUT_ID + 2 ))"
echo "[s2] optimizer restore  = EXTRA_MISC_ARGS='${EXTRA_MISC_ARGS}' (no --no-load-optim / --no-load-rng / --finetune)"
echo "[s2] reward mode        = interpolated_ctrf (S2, bounded [0,1], alpha=0.50)"
echo "[s2] config             = ${CONFIG_TEMPLATE}"
echo "[s2] per-role cfg       = ${MEGATRON_CONFIG_PATH}"
echo "[s2] num_rollout        = ${NUM_ROLLOUT} (END INDEX => $(( NUM_ROLLOUT - START_ROLLOUT_ID )) steps: ${START_ROLLOUT_ID}..$(( NUM_ROLLOUT - 1 )))"
echo "[s2] save_interval      = ${SAVE_INTERVAL}  (~1059 GB per save, NO rotation in this codebase)"
echo "[s2] =================================================================="

exec bash "${SCRIPT_DIR}/run_27b_dense_128gpu_ppo_256k.sh"
