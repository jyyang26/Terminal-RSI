#!/usr/bin/env bash
# launch_27b_run2_warmcritic_ctrf82.sh
#
# Run #2 of the 27B dense Terminal PPO 256k recipe. Thin launcher: it only sets the
# env that differs from run #1 and then calls the unmodified wrapper
# run_27b_dense_128gpu_ppo_256k.sh, so the proven recipe stays the single source of
# truth for topology / memory / proxy / keepalive behaviour.
#
# ============================== WHAT CHANGES vs RUN #1 ==============================
#
# 1. NEW ACTOR. iter_0000263 of the SFT run, converted HF -> Megatron torch_dist.
#    The conversion is NOT optional: --load goes through Megatron's torch_dist reader
#    (checkpoint.py:_is_megatron_checkpoint), and the runtime HF "bridge" path is
#    documented in convert_qwen35_122b_a10b_torch_dist.sh to fail for this model
#    family at PP>1 (AttributeError on output_layer) -- this run is PP=2. Converted
#    offline with TP1xPP8 (tools/convert_hf_to_torch_dist.py, 51 GB, verified
#    output_layer.weight (248320,5120) + embedding (248320,5120)); a torch_dist ckpt
#    reloads at any TP/PP/CP layout, so TP4xCP4xPP2 training reads it fine.
#
# 2. WARM CRITIC. The value head trained in run #1 (step 10 == iter_0000009) is
#    reused instead of a random init. placement_group.py:195 redirects the critic's
#    --load to "<load>_critic" when that sibling exists, so the load root is a PAIR:
#       q38_27b_ppo256k_run2_loadroot          -> actor  (converted iter263)
#       q38_27b_ppo256k_run2_loadroot_critic   -> critic (run #1 iter_0000009)
#    Both trackers say 263 because Megatron resolves <load>/iter_<tracker>; the
#    critic's iter_0000263 is a symlink to run #1's iter_0000009. Verified the critic
#    ckpt holds output_layer.weight (1,5120) + bias (1,), which MATCHES the runtime
#    value-head shape, so _critic_output_layer_needs_reinit returns False and the
#    trained head survives (a shape mismatch is what triggers re-randomisation).
#    Warm-load is weights-only by design (placement_group.py:215-222 sets
#    no_load_optim/finetune) -- the critic's Adam moments from run #1 are discarded
#    because that run's lr schedule (36 rollouts) does not match this one.
#
# 3. CRITIC WARMUP. NUM_CRITIC_ONLY_STEPS=2: steps 0-1 train the critic ONLY
#    (train_async.py:75 `actor_trains_this_step = rollout_id >= num_critic_only_steps`),
#    so the value head re-calibrates to the NEW reward scale before the actor takes
#    its first step. This matters specifically because of change (4): the loaded head
#    predicts on the old /20 scale while returns now arrive ~4x smaller, so the first
#    actor update would otherwise consume advantages computed from a mis-scaled value
#    function.
#
# 4. CTRF DIVISOR 20 -> 82 (+ per-family). Set in the YAML, NOT via
#    HARBOR_TESTCOUNT_SCALE: that env var is absent from the inner launcher's
#    RUNTIME_ENV_JSON, and this run submits through `ray job submit` (DIRECT_RUN=0),
#    so an exported value never reaches the Ray workers and the reward would silently
#    keep dividing by 20. See the yaml's own comment block for the measurement.
#
# 5. TOKEN CLIPPING eps_low=0.3 / eps_high=5.0 (was 0.2/0.2). Deliberately
#    asymmetric: the low side bounds how far the policy may DOWN-weight a token
#    (0.3 -> ratio floor 0.7) while the high side is effectively released
#    (ratio ceiling 6.0), letting the policy move quickly onto newly-rewarded
#    behaviour. NOTE this is a large upper clip; pg_clipfrac is the metric to watch --
#    if it collapses to ~0 the clip has stopped constraining anything and the run is
#    relying entirely on clip_grad 1.0 for actor stability.
#
# 6. CRITIC VALUE-CLIP + GRAD-CLIP relaxed (per-role yaml, actor untouched).
#    Run #1 logged critic-value_clipfrac == 0.000 at every step alongside
#    critic-grad_norm up to 2193 against clip_grad 1.0 -- i.e. the critic's gradient
#    was being scaled down ~2000x, which is why explained_variance took ~9 steps to
#    reach 0. value_clip 0.2 -> 4.0 and clip_grad 1.0 -> 100.0 for the CRITIC ONLY.
#
# 7. SHUFFLE. The Harbor data source walked `sorted(glob(...))` with a monotonic
#    cursor, so each 256-task batch was an alphabetically contiguous -- hence
#    family-contiguous -- block: measured 5-6 families per batch. Now shuffled per
#    epoch (deterministic from rollout_seed+epoch): measured 46 families per batch,
#    full per-epoch coverage, reproducible across actors.
#
# Usage:
#   bash examples/harbor_terminal_rl/launch_27b_run2_warmcritic_ctrf82.sh
#   NUM_ROLLOUT=3 ... (default here is 3 steps, per the "train three steps" request)

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_ROOT=/path/to/workspace

# ---- (1) NEW ACTOR + (2) WARM CRITIC -------------------------------------------------
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-${USER_ROOT}/checkpoints/q38_27b_ppo256k_run2_loadroot}"
export WARM_CRITIC="${WARM_CRITIC:-1}"

# ---- (3) CRITIC WARMUP: 2 critic-only steps -----------------------------------------
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-2}"

# ---- (4) CTRF 82 + per-family scales (carried by the yaml) ---------------------------
export CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/harbor_mixed3000_128gpu_27b_dense_daytona_ppo_cp4_256k_run2_ctrf82.yaml}"
# Belt-and-braces: also export it, so a future DIRECT_RUN=1 invocation (which DOES
# inherit this shell's env) agrees with the yaml instead of silently using 20.
export HARBOR_TESTCOUNT_SCALE="${HARBOR_TESTCOUNT_SCALE:-82}"

# ---- (5) TOKEN CLIPPING: eps_low 0.3 / eps_high 5.0 ---------------------------------
export EPS_CLIP="${EPS_CLIP:-0.3}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-5.0}"

# ---- (6) per-role lr 30x + relaxed critic clips -------------------------------------
export MEGATRON_CONFIG_PATH="${MEGATRON_CONFIG_PATH:-${SCRIPT_DIR}/megatron_roles_27b_criticlr30x_warmcritic.yaml}"

# ---- (7) SHUFFLE ---------------------------------------------------------------------
# Set in the YAML (rollout_shuffle: true + rollout_seed: 42), not here: ROLLOUT_SEED
# has no plumbing anywhere in the launcher chain, so exporting it would be a no-op
# that merely looks like it works.

# ---- TOPOLOGY: 64 train + 56 infer on the surviving 15 nodes ------------------------
# Node 10.0.0.5 was evicted in run #1 (GPU 6 fell off the PCI bus), so the
# cluster is 15 nodes / 120 GPU. 56 % SGLANG_TP(4) == 0 -> 14 engines.
export NODE_IP_LIST="${NODE_IP_LIST:-$(cat /tmp/node_ip_list_15.txt 2>/dev/null || echo "")}"
export EXPECTED_NODES="${EXPECTED_NODES:-15}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-8}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-56}"

# ---- RUN LENGTH + IDENTITY ----------------------------------------------------------
# 3 steps as requested: 2 critic-only warmup (steps 0-1) + 1 full actor+critic step
# (step 2). SAVE_INTERVAL=3 so the run saves once, at the end of step 2.
export NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-3}"
export RUN_TAG="${RUN_TAG:-RUN2_WARMCRITIC_CTRF82_$(date +%Y%m%d_%H%M%S)}"
export WANDB_PROJECT="${WANDB_PROJECT:-terminal3_rl}"
export WANDB_GROUP="${WANDB_GROUP:-q38-27b-dense-ppo-run2-warmcritic-ctrf82}"

echo "[run2] ================= RUN #2 CONFIG ================="
echo "[run2] actor ckpt   = ${PRETRAINED_CKPT} (iter $(cat "${PRETRAINED_CKPT}/latest_checkpointed_iteration.txt" 2>/dev/null))"
echo "[run2] critic ckpt  = ${PRETRAINED_CKPT}_critic (WARM_CRITIC=${WARM_CRITIC})"
echo "[run2] critic warmup= ${NUM_CRITIC_ONLY_STEPS} critic-only steps"
echo "[run2] ctrf scale   = ${HARBOR_TESTCOUNT_SCALE} global + per-family p90 table"
echo "[run2] eps clip     = low ${EPS_CLIP} / high ${EPS_CLIP_HIGH}"
echo "[run2] per-role cfg = ${MEGATRON_CONFIG_PATH}"
echo "[run2] steps        = ${NUM_ROLLOUT} (save every ${SAVE_INTERVAL})"
echo "[run2] =================================================="

exec bash "${SCRIPT_DIR}/run_27b_dense_128gpu_ppo_256k.sh"
