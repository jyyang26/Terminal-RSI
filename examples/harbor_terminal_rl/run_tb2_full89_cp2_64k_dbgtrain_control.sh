#!/bin/bash
#
# DEBUG-TRAIN-ONLY 64k CONTROL run.
#
# WHY
#   We stopped chasing 96k (CP2@96k's step0 mb0 cold-compile window for the big,
#   variable-length dump runs ~70min and trips the CP-subgroup COALESCED watchdog
#   -- see memory cp2-96k-padmult-tritoncache-debug-experiment). Before doing
#   anything else we want a CLEAN baseline: does the SAME topology at 64k still run
#   fine under debug-train-only? It should -- the 64k overfit run ppo_033051
#   (Jun 22, BEFORE the new_group timeout patch) trained cleanly and dumped 3 steps,
#   which means the 64k cold-compile window is short enough to fit inside NCCL's
#   default 10min and never trips the CP-subgroup timeout.
#
# WHAT THIS DOES
#   Replays ppo_033051's REAL dumped rollout data (rollout_dumps/{0,1,2}.pt --
#   287 samples, max token len 49083 < 65536, all <= 64k) straight into the train
#   forward. --load-debug-rollout-data makes arguments.py skip SGLang + force
#   debug_train_only, so placement_group allocates ONLY the 4 actor nodes (32 GPU,
#   no rollout) and rollout.py torch.loads the .pt. => straight to step0 critic
#   forward in ~minutes, no 66min TB2 rollout, no Daytona sandboxes.
#
# TOPOLOGY (identical to the 96k run, only length halved -- the whole point of the
# control): TP2 x CP2 x PP8, seq-length 65536, MAX_TOKENS_PER_GPU 32768
# (microbatch cap = 32768 x cp2 = 65536 = 64k). raw (torch_dist) ckpt load.
#
# VERDICT
#   PASS = step0 critic forward advances past 0/N (mb0 cold-compile survived),
#          step1 actor fwd+bwd runs, no OOM, no watchdog. The steady-state s/mb
#          (after Triton cache is warm) tells us how fast 64k GDN actually is.
#   FAIL = if even 64k trips a COALESCED watchdog or OOM, the problem is NOT
#          96k-specific and we have a deeper topology/compile bug to chase.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# >>DEBUG: replay the clean 64k overfit run's dumped rollout data.
# {rollout_id} -> 0.pt (step0 critic warmup) / 1.pt (step1 actor train) / 2.pt.
# The literal "{rollout_id}" placeholder must NOT sit inside a ${VAR:-default}
# expansion (bash eats the inner "}"); build it in two steps.
PPO_64K_BASE=/path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_overfit_ppo_lenreward_ppo_033051
if [ -z "${LOAD_DEBUG_ROLLOUT_DATA:-}" ]; then
  LOAD_DEBUG_ROLLOUT_DATA="${PPO_64K_BASE}/rollout_dumps/{rollout_id}.pt"
fi
export LOAD_DEBUG_ROLLOUT_DATA

# >>DEBUG: no wandb in a throwaway probe run (avoids "No API key" crash at init).
export USE_WANDB="${USE_WANDB:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# >>DEBUG: only the 4 actor nodes (32 GPU). debug_train_only ignores rollout GPUs
# and skip_sglang means no engines -- so we wait for 4 nodes, not 12.
export EXPECTED_NODES="${EXPECTED_NODES:-4}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-0}"

# >>DEBUG: replay both critic-warmup (0.pt) and actor-train (1.pt). 2 is enough to
# confirm step0 critic fwd + step1 actor fwd/bwd both run.
export NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-999}"     # don't save ckpts in a debug run.

# new_group timeout floor. 64k should never need it (the whole hypothesis is that
# 64k's compile window fits in the default 10min), but keep the patch's 30min floor
# so a PASS here cannot be attributed to the floor masking a slow compile.
export SLIME_NEW_GROUP_TIMEOUT_MINUTES="${SLIME_NEW_GROUP_TIMEOUT_MINUTES:-30}"

# >>64k CONTROL: override every 96k context knob back down to 64k. These are all
# env-driven ${VAR:-default} in the overfit wrapper, so we just set them here.
export SEQ_LENGTH="${SEQ_LENGTH:-65536}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-65536}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"   # x cp2 = 65536 microbatch cap
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-65536}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-65536}"
# Topology unchanged: TP2 x CP2 x PP8 (same as 96k run, only length differs).
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-8}"
export CP_SIZE="${CP_SIZE:-2}"

# Separate run identity so this debug run's artifacts/ray-temp don't collide.
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_overfit_ppo_lenreward_cp2_64k_DBGTRAIN}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_tb2_full89_overfit_ppo_lenreward_cp2_64k_dbgtrain}"
export WANDB_GROUP="${WANDB_GROUP:-q35-27b-tb2-cp2-64k-DBGTRAIN-control}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtb2dbg_64k}"

# Everything else (PPO/critic flags, raw ckpt load, alloc conf) is inherited
# unchanged from the overfit PPO wrapper.
exec "${SCRIPT_DIR}/run_tb2_full89_96gpu_27b_daytona_ppo_lenreward_overfit_fully_async.sh"
