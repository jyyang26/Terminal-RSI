#!/bin/bash
# ====================================================================================
# sh9 PORT of run_tmax15k_122b_a10b_scalable_ppo_criticlr30x_strongest_partial_reward.sh
# ====================================================================================
# The original recipe was written for the gy4 cluster
# (/path/to/shared_storage/...). This cluster is sh9
# (/path/to/shared_storage/...) and the trees correspond one-to-one. Every path in
# the original is a `${VAR:-default}`, so this wrapper re-points them by EXPORT ONLY --
# the shared recipe scripts are left untouched, which keeps this port reviewable and
# means an upstream change to the recipe is inherited automatically.
#
# PATH MAP (verified to exist before launch by the pre-flight below):
#   gy4 .../user/<user>/slime-terminal-RL        -> sh9 .../user/<user>/slime-terminal-RL
#   gy4 .../user/<user>/harbor-code-synthesis/src-> sh9 .../user/<user>/harbor/src
#   gy4 .../hf_models/Qwen/Qwen3.5-122B-A10B         -> sh9 .../models/Qwen3.5-122B-A10B
#   gy4 .../hf_models/Qwen/Qwen3.5-122B-A10B_torch_dist -> sh9 .../models/Qwen3.5-122B-A10B_torch_dist
#   gy4 .../user/<user>/terminal-bench-2-main    -> sh9 .../user/<user>/terminal-bench-2
#   gy4 .../slime-terminal-RL_data/...              -> sh9 .../slime-terminal-RL_data/...
#
# USAGE
#   # PPO smoke test (validate rollout + actor update, COLD critic, no warm-up):
#   ALGO=ppo NUM_CRITIC_ONLY_STEPS=0 ./run_sh9_122b_a10b_128gpu.sh
#
#   # GRPO production run:
#   ALGO=grpo ./run_sh9_122b_a10b_128gpu.sh
#
#   # inspect the resolved plan without spending GPUs:
#   DRY_RUN=1 ALGO=grpo ./run_sh9_122b_a10b_128gpu.sh
set -euo pipefail

SH9_USER=/path/to/workspace
SH9_MODELS=/path/to/models
SCRIPT_DIR="${SH9_USER}/slime-terminal-RL/examples/harbor_terminal_rl"

# ---- code + data roots (sh9) --------------------------------------------------------
export SLIME_DIR="${SLIME_DIR:-${SH9_USER}/slime-terminal-RL}"
export CODE_ROOT="${CODE_ROOT:-${SLIME_DIR}}"
# NOTE the rename: gy4 called this "harbor-code-synthesis/src", sh9 checks it out as
# "harbor/src". Same package (`import harbor` resolves from it) -- verified by importing
# harbor.models.task.task on all 16 nodes.
export HARBOR_SRC="${HARBOR_SRC:-${SH9_USER}/harbor/src}"

# ---- MODEL: cold actor from the sh9 torch_dist, COLD critic -------------------------
# The gy4 recipe warm-loaded the critic from a previous run's iter_69 via a LOAD_ROOT of
# symlinks (<load>_critic sibling). That checkpoint does not exist on sh9, and the task
# explicitly asks for NO critic warm-up, so we point --load straight at the plain
# torch_dist. With no "<load>_critic" sibling on disk, slime's
# _isolate_critic_checkpoint_dirs leaves the critic to cold-start and its value head is
# re-initialised on shape mismatch (model.py:_reinitialize_critic_output_layer).
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-${SH9_MODELS}/Qwen3.5-122B-A10B_torch_dist}"
export HF_CKPT="${HF_CKPT:-${SH9_MODELS}/Qwen3.5-122B-A10B}"
# Must stay 0: the torch_dist is read from the shared fs by all 128 ranks. Staging it to
# node-local /tmp breaks the loader ("Repo id must be...") and wastes 228 GB x 16.
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-0}"
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-0}"
# Neutralise the parent's warm-critic LOAD_ROOT so it cannot re-point --load at gy4.
export STRATEGY15K_LOAD_ROOT=""

# ---- ALGORITHM ----------------------------------------------------------------------
# ALGO=ppo  -> --advantage-estimator ppo, critic active, NUM_CRITIC_ONLY_STEPS honoured.
# ALGO=grpo -> --advantage-estimator grpo, NO critic at all (group-relative baseline), so
#              NUM_CRITIC_ONLY_STEPS must NOT be emitted -- the launcher only appends
#              --num-critic-only-steps when the var is non-empty, so we blank it.
ALGO="${ALGO:-grpo}"
export ADVANTAGE_ESTIMATOR="${ALGO}"
case "${ALGO}" in
  ppo)
    # Cold critic + "no warm-up" as instructed: the actor updates from rollout_id 0.
    export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-0}"
    ;;
  grpo|gspo)
    # GRPO replaces the critic with a GROUP-RELATIVE baseline: the advantage of each
    # sample is its reward minus the mean reward of its group. That needs n>1 samples per
    # prompt -- at n=1 every group has one member, the baseline equals the reward, and
    # every advantage is identically 0, so the actor would never move.
    export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"

    # Keep the TRAJECTORY count per step equal to the validated PPO run (512), because
    # rollout wall-clock scales with total trajectories, not with prompts. The PPO step
    # measured 512 trajectories in ~16 min of rollout; leaving ROLLOUT_BATCH_SIZE at 512
    # while n=8 would mean 4096 trajectories (~2 h of rollout per step) and make a 5-step
    # stability check take most of a day.
    #   prompts 64 x n 8 = 512 trajectories = GLOBAL_BATCH_SIZE
    # Oversampling keeps the parent's ~1.09x skim ratio (64 -> 70) so slow-tail groups can
    # be cancelled without starving the batch.
    export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
    export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-70}"
    export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"

    # ★ MUST be 0, and must be SET rather than unset. The parent recipe does
    # `export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-2}"`, so merely unsetting it
    # here lets the parent re-default it to 2 -- and train_async.py gates the actor on
    # `rollout_id >= num_critic_only_steps`, which under GRPO (where no critic exists at
    # all) would silently skip the actor update for the first TWO steps. 0 is non-empty,
    # so it survives the parent's `:-` and the emitted flag is a harmless no-op.
    export NUM_CRITIC_ONLY_STEPS=0
    ;;
  *) echo "ERROR: ALGO must be ppo|grpo|gspo, got '${ALGO}'" >&2; exit 1 ;;
esac

# ---- CHECKPOINTS: weights only (no optimizer), as requested -------------------------
# --no-save-optim is assembled into EXTRA_MISC_ARGS by the partial_reward wrapper from
# this knob. Saves are much smaller/faster; the cost is that these checkpoints cannot
# resume a run (weights-only), which is accepted here.
export NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-1}"

# ---- DATA + eval pool (sh9) ---------------------------------------------------------
export TASKS_DIR="${TASKS_DIR:-${SH9_USER}/slime-terminal-RL_data/strategy_deepseek_repaired_v4_final_15k/materialized_top15k/tasks}"
# Two sh9-ported yamls, both with harbor_tasks_dir / harbor_eval_tasks_dir / trials_dir
# re-pointed. They differ ONLY in group-completion policy:
#   *_sh9_*            PPO/n=1. harbor_partial_group_min_units: 1 -- accept a group as
#                      soon as one trajectory lands, drop the slow tail. Correct when the
#                      group has one member anyway.
#   *_sh9_grpo_n8_*    GRPO/n=8. Requires FULL 8-member groups, because GRPO's baseline is
#                      the group mean: a truncated group yields advantage 0 and the actor
#                      gets a zero gradient (observed exactly that on the first attempt).
if [ "${ALGO}" = "ppo" ]; then
  _DEFAULT_CONFIG="${SCRIPT_DIR}/harbor_strategy15k_122b_a10b_sh9_cp4_bs512_84k_partial_reward.yaml"
else
  _DEFAULT_CONFIG="${SCRIPT_DIR}/harbor_strategy15k_122b_a10b_sh9_grpo_cp4_n8_84k_partial_reward.yaml"
fi
export CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${_DEFAULT_CONFIG}}"

# ---- RUN IDENTITY (sh9) -------------------------------------------------------------
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export TOPOLOGY="${TOPOLOGY:-128cp4}"
export BASE_FOLDER="${BASE_FOLDER:-${SH9_USER}/slime_harbor_runs/q35_122b_a10b_sh9_${ALGO}_${TOPOLOGY}_${RUN_TAG}}"
export RUN_TRIALS_DIR="${RUN_TRIALS_DIR:-${SH9_USER}/slime_harbor_trials/sh9_${ALGO}_${TOPOLOGY}_${RUN_TAG}}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-122B-A10B_sh9_${ALGO}_${TOPOLOGY}}"
export WANDB_GROUP="${WANDB_GROUP:-q35-122b-a10b-sh9-${ALGO}-${TOPOLOGY}-84k}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rsh9122b}"

# ---- CLUSTER: reuse the running 16-node head ---------------------------------------
# CLEAN_START MUST be 0. The base launcher defaults it to 1, which runs `ray stop --force`
# + `pkill raylet` BEFORE the head-reuse branch -- and the head lives on this box, so that
# would kill the head and drop all 15 workers.
export CLEAN_START="${CLEAN_START:-0}"
export USE_EXISTING_RAY_HEAD="${USE_EXISTING_RAY_HEAD:-1}"
# Must be explicit: the launcher's default `hostname -I | awk '{print $1}'` picks
# 10.0.0.6 on this box (11 interfaces), but the Ray head bound 10.0.0.7.
export MASTER_ADDR="${MASTER_ADDR:-10.0.0.7}"

# ---- CREDENTIALS --------------------------------------------------------------------
export KEYS_FILE="${KEYS_FILE:-/tmp/bs512_keys.env}"
if [ -z "${DAYTONA_API_KEY:-}" ] && [ -f "${KEYS_FILE}" ]; then
  set +u; . "${KEYS_FILE}"; set -u
fi

# ---- WANDB: online, logging to the user's account -----------------------------------
# Account: tonyy10264611 (tonyy10264611-national-university-of-singapore), project
# slime-terminal-rl. The key lives in ${KEYS_FILE} (0600) and is sourced above.
#
# Reaching api.wandb.ai from these nodes REQUIRES the corporate proxy. The base launcher
# already forwards http_proxy/https_proxy into RUNTIME_ENV_JSON, so the Ray workers
# inherit it; verified end-to-end with a real online wandb.init from this box.
#
# HISTORY: an earlier 36-char key was rejected client-side by wandb 0.27.2 ("must have
# 40+ characters") and killed the Ray job during init, before any GPU work. The check
# below fails fast and loudly rather than repeating that 4-minute round trip.
export WANDB_MODE="${WANDB_MODE:-online}"
if [ "${WANDB_MODE}" = "online" ]; then
  [ -n "${WANDB_API_KEY:-}" ] || { echo "ERROR: WANDB_MODE=online but WANDB_API_KEY is empty (expected it from ${KEYS_FILE})." >&2; exit 1; }
  if [ "${#WANDB_API_KEY}" -lt 40 ]; then
    echo "ERROR: WANDB_API_KEY is ${#WANDB_API_KEY} chars; wandb $(python3 -c 'import wandb;print(wandb.__version__)' 2>/dev/null) rejects <40 client-side." >&2
    echo "       Online mode would abort the Ray job during init. Use the full-length key," >&2
    echo "       or pass WANDB_MODE=offline to record locally and 'wandb sync' later." >&2
    exit 1
  fi
fi

# ====================================================================================
# PRE-FLIGHT: every sh9 path must resolve BEFORE we spend 15-20 min on init
# ====================================================================================
_die() { echo "ERROR: $*" >&2; exit 1; }
for p in "${SLIME_DIR}" "${HARBOR_SRC}" "${PRETRAINED_CKPT}" "${HF_CKPT}" "${TASKS_DIR}"; do
  [ -e "${p}" ] || _die "sh9 path does not exist: ${p}"
done
[ -f "${CONFIG_TEMPLATE}" ] || _die "config template missing: ${CONFIG_TEMPLATE}"

# GRPO ZERO-GRADIENT GUARD. GRPO's advantage is reward - mean(reward of its group), so any
# setting that lets a group be accepted with fewer than n_samples_per_prompt members
# produces advantage 0 and a silent no-op update (train/grad_norm 0.0). This was a real
# failure, not a hypothetical -- catch it in seconds instead of after a 25-minute rollout.
if [ "${ALGO}" != "ppo" ]; then
  _MINU=$(grep -oP '^harbor_partial_group_min_units:\s*\K[0-9]+' "${CONFIG_TEMPLATE}" || echo 0)
  _FULL=$(grep -oP '^harbor_require_full_rollout_groups:\s*\K\S+' "${CONFIG_TEMPLATE}" || echo false)
  if [ "${_MINU}" != "0" ]; then
    _die "${ALGO} with a partial-group config: harbor_partial_group_min_units=${_MINU} in
       ${CONFIG_TEMPLATE}
       A group accepted with <n members makes every advantage 0 -> train/grad_norm 0.0.
       Set it to 0 (require full groups)."
  fi
  if [ "${_FULL}" != "true" ]; then
    _die "${ALGO} needs harbor_require_full_rollout_groups: true in ${CONFIG_TEMPLATE}
       (found '${_FULL}'), else short groups reach the trainer as zero-gradient batches."
  fi
  [ "${N_SAMPLES_PER_PROMPT}" -ge 2 ] || _die "${ALGO} needs N_SAMPLES_PER_PROMPT >= 2 (got ${N_SAMPLES_PER_PROMPT}); at n=1 every group baseline equals its own reward."
fi
# The ported yaml must carry NO gy4 path outside comments, or rollout silently reads the
# unmounted old cluster and every trial fails.
if grep -vE '^\s*#' "${CONFIG_TEMPLATE}" | grep -q 'shared-storage\|shared-storage'; then
  _die "${CONFIG_TEMPLATE} still references a gy4/gy2 path outside comments."
fi
# The actor checkpoint must carry its iteration marker, else Megatron loads nothing and
# silently trains from random init.
[ -f "${PRETRAINED_CKPT}/latest_checkpointed_iteration.txt" ] \
  || _die "no latest_checkpointed_iteration.txt under ${PRETRAINED_CKPT}"

echo "[sh9] algo         = ${ALGO} (advantage-estimator ${ADVANTAGE_ESTIMATOR})"
if [ "${ALGO}" = "ppo" ]; then
  echo "[sh9] critic       = COLD (no _critic sibling), critic-only steps = ${NUM_CRITIC_ONLY_STEPS}"
else
  echo "[sh9] critic       = NONE (grpo group baseline), critic-only steps = ${NUM_CRITIC_ONLY_STEPS} (actor trains from step 0)"
  echo "[sh9] grpo batch   = ${ROLLOUT_BATCH_SIZE} prompts x n${N_SAMPLES_PER_PROMPT} = $((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT)) trajectories/step (global batch ${GLOBAL_BATCH_SIZE})"
fi
echo "[sh9] actor load   = ${PRETRAINED_CKPT} (iter $(cat "${PRETRAINED_CKPT}/latest_checkpointed_iteration.txt" 2>/dev/null))"
echo "[sh9] slime        = ${SLIME_DIR}"
echo "[sh9] harbor src   = ${HARBOR_SRC}"
echo "[sh9] config       = ${CONFIG_TEMPLATE}"
echo "[sh9] base folder  = ${BASE_FOLDER}"
echo "[sh9] no-save-optim= ${NO_SAVE_OPTIM}"
echo "[sh9] wandb        = mode=${WANDB_MODE} (offline runs land in \${BASE_FOLDER}/wandb)"
echo "[sh9] master addr  = ${MASTER_ADDR} (clean_start=${CLEAN_START})"
echo

# Hand off to the ORIGINAL recipe, unmodified. It layers the reward/topology/PPO settings
# on top of the exports above -- all of which it reads as `${VAR:-default}`, so ours win.
exec "${SCRIPT_DIR}/run_tmax15k_122b_a10b_scalable_ppo_criticlr30x_strongest_partial_reward.sh"
