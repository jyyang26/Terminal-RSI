#!/usr/bin/env bash
# launch_27b_resume_iter19_curriculum.sh
#
# 27B dense Terminal PPO 256k, resumed from run #2's iter_0000019.
#
# FULL REVERT to run #2's reward + hyperparameters. The ONLY behavioural change is a
# turn-budget curriculum: max_turns ramps 100 -> +5/step instead of sitting at a fixed
# 60 (run #2, censoring) or 500 (the S2 run, too much horizon at once).
#
# ============================== WHAT IS REVERTED ==============================
# Reward   : back to slime_plugins.harbor.partial_reward.test_count_progress_rewards,
#            i.e. R = passed / SCALE(family) with harbor_testcount_scale 82.0 and the
#            45-entry per-family divisor table. The S2 module still exists on disk but
#            is NOT referenced: harbor_reward_mode / harbor_s2_* are absent from the
#            yaml, and CUSTOM_REWARD_POST_PROCESS_PATH points straight at the legacy
#            function rather than at the dispatcher. No alpha gate, no leaf weighting,
#            no [0,1] bound.
# value_clip: back to 4.0 (run #2's per-role override -- see the per-role yaml for why
#            this is 4.0 and not the 0.2 that also appears in the checkpoint).
# lr / wd / clip_grad / adam betas / eps_clip: unchanged from run #2. Every one of these
#            was read back out of iter_0000019/common.pt rather than copied from a
#            template, so the revert is verified against the checkpoint itself.
#
# ==================== CRITIC WARMUP: DELIBERATELY ZERO STEPS ===================
# train_async.py:75 gates the actor on an ABSOLUTE rollout id:
#     actor_trains_this_step = rollout_id >= args.num_critic_only_steps
# With start_rollout_id = 20, setting --num-critic-only-steps 20 makes the condition
# true on the very first step (20 >= 20), so the actor and critic update jointly from
# rollout_id 20 onward -- zero critic-only steps, as intended. The critic is already
# calibrated to this exact reward scale because the reward is reverted, so there is
# nothing for a warmup to re-fit.
# (Contrast: the S2 run needed start_rollout_id + 2 = 22 precisely BECAUSE the gate is
# absolute. Here the requested behaviour is "no warmup", which is what 20 encodes.)
#
# ===================== RESUME: WEIGHTS *AND* OPTIMIZER STATE ====================
# Load roots are a PAIR, both pinned to iter_0000019. VERIFIED: 130 files / 128
# distrib_optim .distcp shards per side; common.pt carries iteration=19 plus 'optimizer'
# and 'opt_param_scheduler'; DCP metadata exposes 48 optimizer entries incl. 16 exp_avg
# tensors on each side.
#   checkpoints/q38_27b_s2_resume_iter19         -> actor
#   checkpoints/q38_27b_s2_resume_iter19_critic  -> critic  (placement_group.py:195
#                                                   redirects the critic's --load here)
# The directory names still say "s2" because they are just symlink roots pointing at
# run #2's own iter_0000019 -- the S2 reward is NOT in play. Kept as-is rather than
# renamed so the already-verified pair is reused untouched.
#
# Three flags MUST NOT reach the training command:
#   --no-load-optim  would discard the Adam moment tensors
#   --no-load-rng    would break sampling-state continuity
#   --finetune       would reset the iteration counter / LR curve
# Two places would otherwise inject them, so both are overridden:
#   (a) EXTRA_MISC_ARGS -- run_27b_dense_128gpu_ppo_256k.sh:300 defaults it to
#       "--no-load-optim --no-load-rng". Overridden to a non-empty INERT value; it must
#       not be empty because the grandparent resolves ${EXTRA_MISC_ARGS:-<moe default>}
#       and `:-` substitutes on empty too. --moe-token-dispatcher-type alltoall is
#       verified inert on this dense model (MoELayer is never instantiated).
#   (b) The WARM-CRITIC path -- placement_group.py:215-222 forces all three on the
#       critic unless --critic-load-optim is passed. CRITIC_LOAD_OPTIM=1 opts out.
#
# One "successfully loaded checkpoint ... at iteration 0" line is EXPECTED and benign:
# it is the reference-model pass (--ref-load points at the same dir, and
# actor.py:688-693 load_other_checkpoint locally forces weights-only for it, then
# restores the flags). The actor's own resume is the "at iteration 19" line.
#
# ========================= SAVE NUMBERING CONTINUES 19+ ========================
# Megatron writes iter_{iteration:07d} from the restored counter, so with iteration=19
# restored the next save is iter_0000020 and onward -- numbering continues rather than
# restarting. Saves land in a NEW run-tagged directory (run #2's originals are left
# untouched); each save includes the full optimizer state because --no-load-optim and
# friends are absent and save-optim is on by default.
#
# ========================= NUM_ROLLOUT IS AN END INDEX =========================
# train_async.py:64 loops `range(args.start_rollout_id, args.num_rollout)`, so
# num_rollout is an ABSOLUTE END INDEX, not a step count. NUM_ROLLOUT=100 from
# start_rollout_id=20 executes 80 steps (rollout_id 20..99). Use 120 for 100 more steps.
#
# =============================== DISK BUDGET ===================================
# There is NO checkpoint rotation in this codebase; every save persists until deleted
# by hand. Each save is ~1059 GB (actor 538 + critic 521). With SAVE_INTERVAL=5 the
# trigger is (rollout_id+1) % 5 == 0 plus a forced final save, i.e. rollout_id
# 24,29,...,94,99 = 16 saves = ~16.6 TB against ~5.7 TB free. The run will exhaust the
# filesystem around save #5 unless checkpoints are pruned as it goes.
#
# Usage:
#   bash examples/harbor_terminal_rl/launch_27b_resume_iter19_curriculum.sh   # 80 steps
#   NUM_ROLLOUT=25 bash .../launch_27b_resume_iter19_curriculum.sh            # short: 20..24

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
export EXTRA_MISC_ARGS="${EXTRA_MISC_ARGS:---moe-token-dispatcher-type alltoall}"
export CRITIC_LOAD_OPTIM="${CRITIC_LOAD_OPTIM:-1}"

# ---- NO CRITIC WARMUP: actor joins on the very first step (see header) ---------------
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-${START_ROLLOUT_ID}}"

# ---- CONFIG: run #2 reward + per-family divisors + turn curriculum -------------------
export CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/harbor_mixed3000_128gpu_27b_dense_daytona_ppo_cp4_256k_resume19_curriculum.yaml}"

# Reward hook reverted to run #2's LEGACY function (NOT the S2 dispatcher). This is the
# grandparent's own hard default, but it is set explicitly so the revert is visible in
# the preflight line and cannot be changed by an inherited env var.
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.partial_reward.test_count_progress_rewards}"
# Guard: if HARBOR_REWARD_MODE is left over in the environment from the S2 run it is
# harmless here (the legacy function never reads it), but unset it so nothing downstream
# can pick it up.
unset HARBOR_REWARD_MODE
export HARBOR_TESTCOUNT_SCALE="${HARBOR_TESTCOUNT_SCALE:-82}"

# ---- per-role lr 30x + run #2's value_clip 4.0 --------------------------------------
export MEGATRON_CONFIG_PATH="${MEGATRON_CONFIG_PATH:-${SCRIPT_DIR}/megatron_roles_27b_resume19_curriculum.yaml}"

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
export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export RUN_TAG="${RUN_TAG:-RESUME19_CURRICULUM_$(date +%Y%m%d_%H%M%S)}"
export WANDB_PROJECT="${WANDB_PROJECT:-terminal3_rl}"
export WANDB_GROUP="${WANDB_GROUP:-q38-27b-dense-ppo-resume19-curriculum}"

CUR_BASE=100
CUR_INC=5
echo "[cur] ========= RUN2 REVERT + TURN CURRICULUM (resume iter19) ========="
echo "[cur] resume from  = ${PRETRAINED_CKPT} (iter ${RESUME_ITER})"
echo "[cur] critic ckpt  = ${PRETRAINED_CKPT}_critic (WARM_CRITIC=${WARM_CRITIC}, CRITIC_LOAD_OPTIM=${CRITIC_LOAD_OPTIM})"
echo "[cur] start_rollout_id = ${START_ROLLOUT_ID}"
echo "[cur] num_critic_only  = ${NUM_CRITIC_ONLY_STEPS}  => NO warmup; actor trains from rollout_id ${START_ROLLOUT_ID}"
echo "[cur] optimizer restore = EXTRA_MISC_ARGS='${EXTRA_MISC_ARGS}' (no --no-load-optim / --no-load-rng / --finetune)"
echo "[cur] reward         = REVERTED to run #2 legacy: passed / SCALE(family), scale ${HARBOR_TESTCOUNT_SCALE}"
echo "[cur] reward hook    = ${CUSTOM_REWARD_POST_PROCESS_PATH}"
echo "[cur] value_clip     = 4.0 (run #2 per-role value; NOT 0.2 default, NOT S2's 0.08)"
echo "[cur] max_turns      = curriculum ${CUR_BASE} + ${CUR_INC}*(r-${START_ROLLOUT_ID})  =>  r20:${CUR_BASE}  r21:$(( CUR_BASE + CUR_INC ))  r22:$(( CUR_BASE + 2*CUR_INC ))  r99:$(( CUR_BASE + 79*CUR_INC ))"
echo "[cur] config         = ${CONFIG_TEMPLATE}"
echo "[cur] per-role cfg   = ${MEGATRON_CONFIG_PATH}"
echo "[cur] num_rollout    = ${NUM_ROLLOUT} (END INDEX => $(( NUM_ROLLOUT - START_ROLLOUT_ID )) steps: ${START_ROLLOUT_ID}..$(( NUM_ROLLOUT - 1 )))"
echo "[cur] save_interval  = ${SAVE_INTERVAL}  (~1059 GB per save, NO rotation in this codebase)"
echo "[cur] ================================================================"

exec bash "${SCRIPT_DIR}/run_27b_dense_128gpu_ppo_256k.sh"
