#!/bin/bash
#
# DEBUG-TRAIN-ONLY harness to validate the SHAPE-CONVERGENCE fix for the CP2@96k
# step0 2.5h cold-compile storm (and the engine-OOM it triggers), WITHOUT paying
# the ~66min TB2 rollout or standing up the 64-GPU rollout half.
#
# WHAT WE ARE TESTING (see PPO_INTEGRATION_NOTES §X)
#   The step0 critic forward/train at 96k took ~2.5h with a wildly oscillating
#   per-microbatch time (3s <-> 250s). Root cause = variable-length terminal
#   trajectories (token len 1855..77586) + dynamic batching produce ~60 DISTINCT
#   padded thd microbatch shapes per step at the stock pad granularity
#   (data_pad_size_multiplier=128 -> pad_size = TP*128 = 256). Each new shape
#   triggers a fresh Triton/GDN JIT compile (tens..200s); seen shapes run in ~3s.
#
#   Two knobs, multiplicative:
#     1. --data-pad-size-multiplier 2048  -> pad_size = TP*2048 = 4096 collapses
#        the distinct-shape count from ~60 to ~22 (CPU-measured on the real dump).
#        2048 is the sweet spot: max padded T just reaches the 98304 bin cap;
#        4096 would overflow it.
#     2. persistent TRITON_CACHE_DIR -> compiled shapes survive across steps/
#        epochs, so step1 only compiles the few NEW shapes (CPU-measured: 5 new
#        vs 22 at step0), and a 2nd epoch is ~all-hit / zero-compile.
#
# REPLAY DATA
#   Replays the SAME dump that produced the 2.5h storm: the memfrac0.6 run's
#   rollout_dumps/{0,1}.pt (264 samples, token len max 77586). 0.pt = step0
#   critic warmup input, 1.pt = step1 actor train input. --load-debug-rollout-data
#   makes arguments.py skip SGLang + force debug_train_only, so placement_group.py
#   allocates ONLY the 4 actor nodes (32 GPU); rollout.py torch.loads the .pt.
#   => straight to step0 critic forward in ~minutes.
#
#   NOTE: this debug path exercises the ACTOR-side Megatron GDN compile (the
#   dominant cost of the 2.5h step0). The engine-side fla.l2norm OOM is a
#   downstream consequence of the same long compile window; shrinking the actor
#   shape space + persistent cache addresses the window itself. Engine-side
#   behaviour still needs the full run to confirm, but if the actor compile
#   storm collapses here, the window that starves the engines collapses too.
#
# A/B USAGE (run twice, compare step0 wall-time + per-mb curve):
#   # baseline (stock pad, fresh cache):
#   DATA_PAD_SIZE_MULTIPLIER=128  RUN_TAG=pad128  bash run_..._padmult_tritoncache.sh
#   # fix (coarse pad, fresh cache):
#   DATA_PAD_SIZE_MULTIPLIER=2048 RUN_TAG=pad2048 bash run_..._padmult_tritoncache.sh
# Each RUN_TAG gets its own EMPTY TRITON_CACHE_DIR so step0 measures a true cold
# compile; NUM_ROLLOUT=2 then lets step1 show the cross-step cache hit.
#
# VERDICT
#   PASS = step0 critic forward+train wall-time drops sharply vs the 2.5h pad128
#          baseline AND the per-mb time stops oscillating (few slow compiles up
#          front, then steady fast mbs); step1 is faster than step0 (cache hit);
#          loss finite (numerics safe under coarse pad). step1 actor fwd/bwd runs.
#   FAIL = still oscillating/slow (shape space didn't collapse) OR loss NaN/Inf
#          (coarse pad broke masking) OR OOM.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# --- the fix under test ----------------------------------------------------
# Coarsen thd packing pad granularity. base launcher passes
# --data-pad-size-multiplier when this is non-empty. 2048 = sweet spot.
export DATA_PAD_SIZE_MULTIPLIER="${DATA_PAD_SIZE_MULTIPLIER:-2048}"

# Per-run-tag identity so A/B runs (pad128 vs pad2048) don't collide.
RUN_TAG="${RUN_TAG:-pad${DATA_PAD_SIZE_MULTIPLIER}}"

# Fresh, EMPTY Triton cache per RUN_TAG so step0 measures a true COLD compile
# (and step1 measures the warm cross-step hit). New path => guaranteed empty,
# no need to rm across nodes. base launcher exports this into RUNTIME_ENV_JSON.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_dbg_${RUN_TAG}}"

# --- replay the memfrac0.6 run's dump (the one that produced the 2.5h storm) -
# {rollout_id} -> 0.pt / 1.pt. Keep the literal placeholder OUT of a ${VAR:-..}
# expansion (bash mangles the inner "}").
CP2_96K_BASE=/path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_overfit_ppo_lenreward_cp2_96k
if [ -z "${LOAD_DEBUG_ROLLOUT_DATA:-}" ]; then
  LOAD_DEBUG_ROLLOUT_DATA="${CP2_96K_BASE}/rollout_dumps/{rollout_id}.pt"
fi
export LOAD_DEBUG_ROLLOUT_DATA

# --- debug-train-only knobs (no wandb, 4 nodes, replay 2 dumps, no ckpt save) -
export USE_WANDB="${USE_WANDB:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export EXPECTED_NODES="${EXPECTED_NODES:-4}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-0}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-2}"            # step0 (0.pt) + step1 (1.pt)
export SAVE_INTERVAL="${SAVE_INTERVAL:-999}"      # no ckpts in a debug probe
export SLIME_NEW_GROUP_TIMEOUT_MINUTES="${SLIME_NEW_GROUP_TIMEOUT_MINUTES:-30}"

# Separate run identity per RUN_TAG (artifacts / ray-temp don't collide).
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_overfit_ppo_lenreward_cp2_96k_DBGTRAIN_${RUN_TAG}}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_tb2_cp2_96k_dbgtrain_${RUN_TAG}}"
export WANDB_GROUP="${WANDB_GROUP:-q35-27b-tb2-cp2-96k-DBGTRAIN-${RUN_TAG}}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtb2dbg_${RUN_TAG}}"

echo "[dbgtrain padmult] RUN_TAG=${RUN_TAG} DATA_PAD_SIZE_MULTIPLIER=${DATA_PAD_SIZE_MULTIPLIER} TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo "[dbgtrain padmult] replay=${LOAD_DEBUG_ROLLOUT_DATA} nodes=${EXPECTED_NODES} num_rollout=${NUM_ROLLOUT}"

# Everything else (TP2×CP2×PP8, 5 walls @98304, MAX_TOKENS 49152, PPO/critic
# flags, raw ckpt load, alloc conf) inherited unchanged from the CP2@96k wrapper.
exec "${SCRIPT_DIR}/run_tb2_full89_96gpu_27b_daytona_ppo_lenreward_overfit_fully_async.sh"
