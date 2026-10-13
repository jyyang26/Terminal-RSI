#!/bin/bash
#
# DEBUG-TRAIN-ONLY harness to validate the new_group timeout fix for the CP2@96k
# deadlock, WITHOUT paying the ~66min TB2 rollout each iteration.
#
# WHAT THIS DOES
#   Replays the rollout data that the crashed CP2@96k run dumped to disk
#   (rollout_dumps/{0,1}.pt -- the real 96k-long trajectories that triggered the
#   deadlock) straight into the train forward. Setting --load-debug-rollout-data
#   makes arguments.py:1507 skip SGLang + arguments.py:1816 force debug_train_only,
#   so placement_group.py:100 allocates ONLY the 4 actor nodes (32 GPU, no rollout
#   64 GPU) and rollout.py:806 torch.loads the .pt instead of running Harbor.
#   => goes straight to step0 critic forward in ~minutes, not ~66min.
#
# THE FIX UNDER TEST
#   slime/utils/reloadable_process_group.py monkey_patch_torch_dist() now injects a
#   default timeout (30min, SLIME_NEW_GROUP_TIMEOUT_MINUTES) into every new_group()
#   created without an explicit one. The CP2@96k run died because the GDN
#   context-parallel coalescing subgroup (PG 21) was created without a timeout and
#   inherited NCCL's 10min default; the per-PP-stage GDN Triton autotune skew at 96k
#   blew that 10min window (observed: COALESCED Numel=0 ran 691s > 600000ms).
#   Raising the floor to 30min should let the late-compiling rank still join.
#   NOTE: --distributed-timeout-minutes (already 60 in the base) does NOT cover this
#   subgroup -- new_group() does not inherit the init_process_group timeout. That is
#   the whole reason this patch exists.
#
# VERDICT
#   PASS  = critic forward advances past 0/7 (first microbatch's compile skew was
#           survived), step1 actor fwd+bwd runs, no OOM. Then the per-microbatch
#           STEADY-STATE time (after Triton cache is warm) tells us whether 96k GDN
#           is fast enough to be usable at all (the still-open CP1-B algebra墙 question).
#   FAIL  = still dies on a COALESCED watchdog -> the dying group is created by a path
#           the new_group patch doesn't cover (report which PG/timeout).

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# >>DEBUG: replay the crashed CP2@96k run's dumped rollout data. {rollout_id} ->
# 0.pt (rollout0 = step0 critic warmup input) and 1.pt (rollout1 = step1 actor train).
# NOTE: the literal "{rollout_id}" placeholder must NOT sit inside a ${VAR:-default}
# expansion -- bash treats the inner "}" as the end of the parameter expansion and
# mangles it to "{rollout_id.pt}". Build it in two steps instead.
CP2_96K_BASE=/path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_overfit_ppo_lenreward_cp2_96k_20260625_124601
if [ -z "${LOAD_DEBUG_ROLLOUT_DATA:-}" ]; then
  LOAD_DEBUG_ROLLOUT_DATA="${CP2_96K_BASE}/rollout_dumps/{rollout_id}.pt"
fi
export LOAD_DEBUG_ROLLOUT_DATA

# >>DEBUG: no wandb in a throwaway deadlock-probe run (avoids the "No API key
# configured" crash at init). We read the verdict from the driver log, not wandb.
export USE_WANDB="${USE_WANDB:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# >>DEBUG: only the 4 actor nodes (32 GPU). debug_train_only (auto-set by
# load_debug_rollout_data) ignores rollout_num_gpus, and skip_sglang means no rollout
# engines come up -- so we wait for 4 nodes, not 12.
export EXPECTED_NODES="${EXPECTED_NODES:-4}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-0}"

# >>DEBUG: replay both dumps (rollout 0 = critic warmup, rollout 1 = actor train).
export NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-999}"     # don't save ckpts in a debug run.

# new_group timeout floor (the fix). 30min >> worst-case GDN compile skew at 96k.
# The patch defaults to 30 even unset, but we make it explicit + visible here.
export SLIME_NEW_GROUP_TIMEOUT_MINUTES="${SLIME_NEW_GROUP_TIMEOUT_MINUTES:-30}"

# Separate run identity so this debug run's artifacts/wandb/ray-temp don't collide.
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_overfit_ppo_lenreward_cp2_96k_DBGTRAIN}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_tb2_full89_overfit_ppo_lenreward_cp2_96k_dbgtrain}"
export WANDB_GROUP="${WANDB_GROUP:-q35-27b-tb2-cp2-96k-DBGTRAIN-newgroup-timeout30}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtb2ppo_dbgtrain}"

# Everything else (TP2×CP2×PP8, 5 walls @98304, MAX_TOKENS 49152, PPO/critic flags,
# raw ckpt load, alloc conf) is inherited unchanged from the CP2@96k PPO wrapper.
exec "${SCRIPT_DIR}/run_tb2_full89_96gpu_27b_daytona_ppo_lenreward_overfit_fully_async.sh"
