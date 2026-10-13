#!/usr/bin/env bash
# launch_27b_fresh_base_critic_warmup2.sh
#
# FRESH 27B dense Terminal PPO 256k run, started from the BASE model
#   /path/to/models/Qwen3.8-27B
# with a COLD critic and a 2-step critic-only warmup.
#
# Everything else -- topology, reward, lr / 30x ratio, critic knobs, Harbor task pool
# and environment, turn curriculum -- is inherited verbatim from the run that was just
# stopped (launch_27b_resume_iter19_curriculum.sh + its two yamls), per the instruction
# to reuse the mature configuration unchanged.
#
# =============================== WHAT IS DIFFERENT ===============================
#  (1) --load is the CONVERTED base, and the rollout counter starts at 0.
#  (2) COLD critic (random value head) instead of a warm one.
#  (3) 2 critic-only warmup steps (rollout_id 0 and 1); actor joins at rollout_id 2.
#  (4) Optimizer state is NOT loaded (there is none) but IS created and IS saved.
#  (5) Curriculum anchor re-keyed 20 -> 0 so the ramp starts on this run's first step.
#
# =========================== (1) LOAD PATH: CONVERSION IS REQUIRED ===========================
# The requested start path is a HuggingFace directory (18 *.safetensors, config.json, no
# tracker, no *.distcp). Megatron's --load goes through the torch_dist reader
# (checkpointing.py:_is_megatron_checkpoint) and CANNOT read it; the runtime HF bridge
# path is documented to fail for this model family at PP>1, and this run is PP=2. So the
# base is converted offline first, exactly as run #2's actor was:
#     bash examples/harbor_terminal_rl/convert_q38_27b_base_torch_dist.sh
#   -> ${USER_ROOT}/checkpoints/q38_27b_base_torch_dist   (tracker "release")
# This launcher hard-fails if that output is missing rather than silently falling back.
#
# NOTE the base is a VLM checkpoint (Qwen3_5ForConditionalGeneration: 850
# model.language_model.* keys + 333 model.visual.* + 15 mtp.* + lm_head). No stripping is
# needed: slime_plugins/mbridge/qwen3_5.py maps the model.language_model.* prefix and the
# mapping is pulled mcore->hf, so the vision tower and MTP head are simply never
# requested. See the converter's header for the verified key counts.
#
# NO symlink load-root is needed here (unlike the iter263 / iter19 cases). The converter
# writes the literal tracker "release", and arguments.py:1814-1815 special-cases that to
# set start_rollout_id = 0. A NUMERIC tracker would instead leave start_rollout_id at
# loaded_iteration+1 -- the failure documented at run_27b_dense_128gpu_ppo_256k.sh:105-128
# where range(176, 2) was empty and training never ran. START_ROLLOUT_ID=0 is also set
# explicitly below so the value does not depend on that inference at all.
#
# ================================= (2) COLD CRITIC =================================
# WARM_CRITIC=0 and NO "<load>_critic" sibling may exist. placement_group.py:195
# redirects the critic's --load to "<load>_critic" when that directory is present, which
# would silently warm-load a trained value head. With it absent, the critic takes the
# actor's backbone and model.py:_critic_output_layer_needs_reinit sees the shape mismatch
# (actor LM head [248320, 5120] vs value head [1, 5120]) and
# _reinitialize_critic_output_layer runs AFTER load_checkpoint:
#     weight.normal_(mean=0.0, std=0.02); bias.zero_()  + optimizer.reload_model_params()
# The wrapper hard-fails if the sibling exists (run_27b_dense_128gpu_ppo_256k.sh:473-475),
# and this launcher asserts it too.
#
# ========================== (3) CRITIC WARMUP: EXACTLY 2 STEPS ==========================
# train_async.py:75 gates the actor on an ABSOLUTE rollout id:
#     actor_trains_this_step = rollout_id >= args.num_critic_only_steps
# and train_async.py:64 loops range(start_rollout_id, num_rollout).
# With start_rollout_id = 0, NUM_CRITIC_ONLY_STEPS=2 gives exactly what is requested:
#     rollout_id 0 -> 0 >= 2 False -> critic only
#     rollout_id 1 -> 1 >= 2 False -> critic only
#     rollout_id 2 -> 2 >= 2 True  -> actor joins, joint actor+critic update
# This is NOT the same arithmetic as the iter19 resume, where start_rollout_id = 20 meant
# 2 warmup steps had to be written as 22 and a literal 2 would have given ZERO warmup.
# Here the run starts at 0, so the requested literal 2 is correct as-is.
# Confirmed in preflight by:
#   [preflight] critic-only steps = 2 (actor joins at rollout_id 2); save every 5
#
# ==================== (4) OPTIMIZER: NOT LOADED, BUT CREATED AND SAVED ====================
# These are two independent switches and the requirement pulls them in opposite directions.
#
#  LOAD side  -- EXTRA_MISC_ARGS="--no-load-optim --no-load-rng"
#    The base checkpoint has NO optimizer state at all (verified: 0 optimizer entries in
#    its DCP metadata, 956 weight/rng entries only). Without --no-load-optim Megatron
#    looks for distrib_optim shards that do not exist and aborts. --no-load-rng is paired
#    with it for the same reason: there is no saved RNG state to continue from on a fresh
#    run. This is exactly the wrapper's own default for a cold start
#    (run_27b_dense_128gpu_ppo_256k.sh:299) and is set explicitly here so it is visible.
#
#  RUNTIME side -- a full AdamW is still built from scratch.
#    --no-load-optim only skips READING optimizer state; Megatron still constructs the
#    distributed optimizer and the opt_param_scheduler for every trainable parameter.
#    There is no "weights-only training" mode being entered: the moments simply start at
#    zero, which is the correct initial condition for a fresh run. Critically, this is
#    NOT the same as --finetune (which would also reset the iteration counter / LR curve)
#    -- --finetune is NOT passed.
#
#  SAVE side  -- NO_SAVE_OPTIM=0
#    Every checkpoint packs the complete optimizer state (the distrib_optim .distcp
#    shards + opt_param_scheduler in common.pt), so the run stays resumable. Measured on
#    the previous run: 130 files / 128 .distcp per side, 538 GB actor + 521 GB critic.
#    The 122B wrapper defaults this to 1 (weights-only, unresumable); here it must be 0.
#
# ============================== (5) TURN CURRICULUM ==============================
# max_turns(r) = 100 + 5*r, capped at 500 (the yaml re-keys the anchor 20 -> 0 so the
# ramp starts on this run's first step instead of clamping flat). Logged per step as
#   [Rollout {r}] Curriculum max_turns updated to: {n}
# Enforced in slime_plugins/harbor/rollout.py::_build_trial_config; eval is excluded.
#
# ========================= NUM_ROLLOUT IS AN ABSOLUTE END INDEX =========================
# train_async.py:64 loops range(start_rollout_id, num_rollout). From start_rollout_id=0,
# NUM_ROLLOUT=100 executes 100 steps (rollout_id 0..99). (On the iter19 resume the same
# 100 meant only 80 steps, because that run started at 20.)
#
# =============================== DISK BUDGET: TIGHT ===============================
# There is NO checkpoint rotation anywhere in this codebase; every save persists until
# deleted by hand. At ~1059 GB per save and SAVE_INTERVAL=5 the triggers are rollout_id
# 4,9,...,94,99 = 20 saves = ~21 TB. Free space at launch time is far below that, so
# checkpoints MUST be pruned as the run proceeds.
#
# Usage:
#   bash examples/harbor_terminal_rl/convert_q38_27b_base_torch_dist.sh   # once, needs 8 idle GPU
#   bash examples/harbor_terminal_rl/launch_27b_fresh_base_critic_warmup2.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_ROOT=/path/to/workspace

# ---- (1) LOAD: converted base, rollout counter from 0 -------------------------------
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-${USER_ROOT}/checkpoints/q38_27b_base_torch_dist}"
export HF_CKPT="${HF_CKPT:-/path/to/models/Qwen3.8-27B}"
export START_ROLLOUT_ID="${START_ROLLOUT_ID:-0}"

if [ ! -f "${PRETRAINED_CKPT}/latest_checkpointed_iteration.txt" ]; then
  echo "[fresh] FATAL: ${PRETRAINED_CKPT} is not a Megatron torch_dist checkpoint." >&2
  echo "[fresh]        The base model at ${HF_CKPT} is HuggingFace safetensors and" >&2
  echo "[fresh]        Megatron --load cannot read it. Convert first:" >&2
  echo "[fresh]          bash ${SCRIPT_DIR}/convert_q38_27b_base_torch_dist.sh" >&2
  exit 1
fi

# ---- (2) COLD CRITIC ----------------------------------------------------------------
export WARM_CRITIC="${WARM_CRITIC:-0}"
if [ -e "${PRETRAINED_CKPT}_critic" ]; then
  echo "[fresh] FATAL: ${PRETRAINED_CKPT}_critic exists -> placement_group.py:195 would" >&2
  echo "[fresh]        WARM-load a value head. A cold start requires it absent." >&2
  exit 1
fi

# ---- (3) CRITIC WARMUP: 2 critic-only steps (rollout_id 0,1) ------------------------
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-2}"

# ---- (4) OPTIMIZER: skip LOADING (none exists), but CREATE and SAVE it --------------
export EXTRA_MISC_ARGS="${EXTRA_MISC_ARGS:---no-load-optim --no-load-rng}"
export NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-0}"
# --finetune is deliberately NOT passed (it would reset the iteration counter / LR curve).
# CRITIC_LOAD_OPTIM is deliberately NOT set: there is no critic optimizer state to load.

# ---- (5) CONFIG: run #2 reward + per-family divisors + curriculum anchored at 0 ------
export CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/harbor_mixed3000_128gpu_27b_dense_daytona_ppo_cp4_256k_fresh_base.yaml}"
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.partial_reward.test_count_progress_rewards}"
unset HARBOR_REWARD_MODE
export HARBOR_TESTCOUNT_SCALE="${HARBOR_TESTCOUNT_SCALE:-82}"

# ---- per-role lr 30x + critic value_clip 4.0 / clip_grad 100 ------------------------
export MEGATRON_CONFIG_PATH="${MEGATRON_CONFIG_PATH:-${SCRIPT_DIR}/megatron_roles_27b_fresh_base.yaml}"

# ---- PPO clipping: unchanged ---------------------------------------------------------
export EPS_CLIP="${EPS_CLIP:-0.3}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-5.0}"

# ---- TOPOLOGY: 64 train + 56 infer on the 15 surviving nodes ------------------------
export NODE_IP_LIST="${NODE_IP_LIST:-$(cat /tmp/node_ip_list_15.txt 2>/dev/null || echo "")}"
export EXPECTED_NODES="${EXPECTED_NODES:-15}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-8}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-56}"

# ---- RUN LENGTH + IDENTITY (new run dir, no collision with history) -----------------
export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export RUN_TAG="${RUN_TAG:-FRESH_BASE_WARMUP2_$(date +%Y%m%d_%H%M%S)}"
export WANDB_PROJECT="${WANDB_PROJECT:-terminal3_rl}"
export WANDB_GROUP="${WANDB_GROUP:-q38-27b-dense-ppo-fresh-base-warmup2}"

CUR_BASE=100
CUR_INC=5
echo "[fresh] ========== FRESH PPO FROM BASE + 2-STEP CRITIC WARMUP =========="
echo "[fresh] base model     = ${HF_CKPT} (HF safetensors)"
echo "[fresh] --load         = ${PRETRAINED_CKPT} (torch_dist, tracker '$(cat "${PRETRAINED_CKPT}/latest_checkpointed_iteration.txt")')"
echo "[fresh] start_rollout_id = ${START_ROLLOUT_ID}"
echo "[fresh] critic         = COLD (random value head; no _critic sibling present)"
echo "[fresh] critic warmup  = ${NUM_CRITIC_ONLY_STEPS} steps => rollout_id 0,1 critic-only; actor joins at rollout_id ${NUM_CRITIC_ONLY_STEPS}"
echo "[fresh] optimizer LOAD = SKIPPED via '${EXTRA_MISC_ARGS}' (base has no optimizer state)"
echo "[fresh] optimizer SAVE = $([ "${NO_SAVE_OPTIM}" = "0" ] && echo "INCLUDED (distrib_optim shards packed every save)" || echo "OMITTED")"
echo "[fresh] reward         = run #2 legacy: passed / SCALE(family), global scale ${HARBOR_TESTCOUNT_SCALE}"
echo "[fresh] reward hook    = ${CUSTOM_REWARD_POST_PROCESS_PATH}"
echo "[fresh] value_clip     = 4.0 (critic only); actor clip_grad 1.0 / critic 100.0"
echo "[fresh] max_turns      = curriculum ${CUR_BASE} + ${CUR_INC}*r  =>  r0:${CUR_BASE}  r1:$(( CUR_BASE + CUR_INC ))  r2:$(( CUR_BASE + 2*CUR_INC ))  r99:500 (cap)"
echo "[fresh] config         = ${CONFIG_TEMPLATE}"
echo "[fresh] per-role cfg   = ${MEGATRON_CONFIG_PATH}"
echo "[fresh] num_rollout    = ${NUM_ROLLOUT} (END INDEX => ${NUM_ROLLOUT} steps: 0..$(( NUM_ROLLOUT - 1 )))"
echo "[fresh] save_interval  = ${SAVE_INTERVAL}  (~1059 GB per save, NO rotation in this codebase)"
echo "[fresh] run tag        = ${RUN_TAG}"
echo "[fresh] ================================================================"

exec bash "${SCRIPT_DIR}/run_27b_dense_128gpu_ppo_256k.sh"
