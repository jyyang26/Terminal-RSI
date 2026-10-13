#!/bin/bash
#
# debug-train-only harness for the tmax-15k 27B PPO run, on 64 idle GPUs.
#
# WHY THIS EXISTS
#   Main run #3 (20260701_133022) trained ~3.5h then died at the rollout-2 weight
#   sync: init_process_group DistStoreError "63/65 clients joined" alongside
#   TorchMemorySaver OOM on the train actors -> a 64k train-side GPU memory spike
#   likely killed 2 ranks, the rest hung at the weight-sync barrier. Separately,
#   PPO_INTEGRATION_NOTES.md's older conclusions were written when the hardware was
#   faulty (model load failing), so they're untrustworthy.
#
#   This harness isolates the TRAIN half: --debug-train-only (auto-forced by
#   --load-debug-rollout-data) skips SGLang + Daytona entirely and replays the REAL
#   rollout data the main run dumped (rollout_dumps/{id}.pt) through the optimizer.
#   It answers two questions the main run couldn't cleanly separate:
#     A) does the 27B torch_dist ckpt load cleanly on this hardware? (the old suspect)
#     B) does 64k forward/backward/optimizer step OOM? (the rollout-2 death trigger)
#
# TOPOLOGY (64 GPU, CP doubled vs the 32-GPU main run to halve per-GPU activation mem)
#   Main run actor = TP2 x PP8 x CP2 = 32.  Here = TP2 x PP8 x CP4 = 64, DP=1.
#   Doubling CP splits each 64k sequence across 4 ranks instead of 2, so per-GPU
#   activation memory ~halves -- directly targeting the rollout-2 OOM. The 27B
#   torch_dist ckpt is .distcp (resharding-free), so CP2->CP4 needs no re-convert.
#   8 nodes x 8 = 64.
#
# USAGE (via the tmux launcher; do not run directly on a single node):
#   see start_tmax15k_debug_train_only_64gpu_tmux.sh

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# --- REPLAY REAL ROLLOUT DATA (this flag auto-forces debug_train_only=True) -------
# Points at the main run's dumped rollouts. {rollout_id} is substituted per step by
# slime/ray/rollout.py::_get_rollout_data. Files 0.pt..5.pt exist; we replay 0,1,2
# (2.pt = the step the main run died on -> the exact 64k train load to stress-test).
MAIN_RUN_BASE="${MAIN_RUN_BASE:-/path/to/workspace/slime_harbor_qwen35_27b_tmax15k_full_ppo_64k_20260701_133022}"
# NB: assign the {rollout_id} template in two steps. A literal "{rollout_id}" inside
# a ${VAR:-default} expansion makes bash treat the first "}" as closing the expansion
# (yielding a corrupted ".../{rollout_id.pt}"), so build the default separately.
_default_replay="${MAIN_RUN_BASE}/rollout_dumps/{rollout_id}.pt"
export LOAD_DEBUG_ROLLOUT_DATA="${LOAD_DEBUG_ROLLOUT_DATA:-$_default_replay}"

# --- 64-GPU topology: CP doubled to 4 (halves per-GPU 64k activation memory) ------
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-8}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-8}"
export CP_SIZE="${CP_SIZE:-4}"
# No SGLang engines in debug-train-only.
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-0}"

# Keep the proven 64k window (5 walls at 65536). With CP4 a 64k sequence is split
# across 4 ranks, so one 64k seq per microbatch needs MAX_TOKENS_PER_GPU x CP4 =
# 16384 x 4 = 65536. Halving MAX_TOKENS_PER_GPU from 32768 -> 16384 matches CP4.
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"

# --- Debug schedule: replay a few steps, no save, no eval -------------------------
export NUM_ROLLOUT="${NUM_ROLLOUT:-3}"        # replay rollout_dumps 0,1,2
export SAVE_INTERVAL="${SAVE_INTERVAL:-999}"  # never save in a debug run
export EVAL_INTERVAL="${EVAL_INTERVAL:-999}"  # never eval (no SGLang anyway)

# Load base 27B directly from shared fs first (fast: verify load works before paying
# the 51G /tmp staging cost). Flip STAGE_TORCH_DIST_CKPT_TO_TMP=1 to replicate the
# main run's /tmp load path once load is confirmed healthy.
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-0}"
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-0}"

# --- Separate run identity so nothing collides with the main run's artifacts ------
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_tmax15k_dbgtrain_64k_${RUN_TAG}}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_tmax15k_dbgtrain_64k_${RUN_TAG}}"
export WANDB_GROUP="${WANDB_GROUP:-q35-27b-tmax15k-dbgtrain-cp4-64gpu-64k}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtmax15k_dbg}"

# Fault tolerance off; gloo/nccl timeout patches inherited from the main wrapper.
export USE_FAULT_TOLERANCE="${USE_FAULT_TOLERANCE:-0}"

# Everything else (PPO/critic flags, raw ckpt load, alloc conf, 64k walls,
# timeout patches) is inherited unchanged from the tmax-15k full wrapper.
exec "${SCRIPT_DIR}/run_tmax15k_full_96gpu_27b_daytona_ppo_64k_fully_async.sh"
