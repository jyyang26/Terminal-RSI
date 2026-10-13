#!/bin/bash
#
# "STRONGEST RUN" wrapper, batch-384 + PINNED-SANDBOX-SPEC variant. 2026-07-12.
#
# WHAT CHANGED vs run_tmax15k_qwen35_120gpu_ppo_criticlr30x_strongest.sh
#   This is a FRESH run FROM SCRATCH (new RUN_TAG/BASE_FOLDER via the launch script), NOT
#   a resume -- so there is no --load and no risk of clobbering the prior run's iter4/9/14.
#   Two deltas from the prior strongest wrapper, everything else identical (cold-start
#   critic, critic lr 30x / actor lr 2x, standard skim-oversample, 64k walls, 100 steps,
#   Qwen3.5-27B pre-staged weights, FT on):
#     1) BATCH 256 -> 384. ROLLOUT_BATCH_SIZE=GLOBAL_BATCH_SIZE=384. OVER_SAMPLING_BATCH_
#        SIZE 260 -> 390 (~1.016x faithful skim, same ratio as the old 260/256). Bigger
#        batch = lower PPO gradient noise + more robust critic fit + more tolerant of the
#        standard-mode slow-tail cancel.
#     2) CONFIG -> harbor_tmax15k_strongest_b384_pinnedspec_64k.yaml, which PINS every
#        Daytona sandbox to 1 CPU / 2 G RAM / 6 G disk (override_cpus/memory_mb/storage_mb)
#        and raises the in-flight cap + trial semaphore 280 -> 400 so they don't throttle
#        the 390 oversample window. The pinned spec ~4-5x cuts per-sandbox footprint,
#        which frees the Daytona capacity the larger batch needs.
#
# TIMEOUTS UNCHANGED (user requirement): the new YAML inherits the tmax agent/verifier/
# trial/eval timeouts verbatim. This wrapper does NOT touch any *_timeout_sec.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# --- ROLLOUT MODE: force standard (starter hardcodes fully_async; must override) ------
export HARBOR_ROLLOUT_MODE="standard"

# --- MODEL: Qwen3.5-27B (raw torch_dist load; cold-start critic from base, no _critic) -
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR:-/path/to/workspace/code/slime-v0.3-harbor}/scripts/models/qwen3.5-27B.sh}"
export MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-raw}"
export HF_CKPT="${HF_CKPT:-/path/to/models/Qwen3.5-27B}"
# PRETRAINED_CKPT = RAW shared-FS 3.5 torch_dist (NO _critic sibling) -> critic COLD START
# (value head reinit). Deliberately NOT the synth3k_ppo load-root (warm stuck critic).
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/models/Qwen3.5-27B_torch_dist_pp8_timeout60}"
# 3.5 HF + torch_dist already staged on all nodes with .slime_stage_complete -> instant skip.
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-1}"
export STAGED_HF_CKPT="${STAGED_HF_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.5-27B}"
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-1}"
export STAGED_TORCH_DIST_CKPT="${STAGED_TORCH_DIST_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.5-27B_torch_dist_pp8_timeout60}"

# --- per-role lr: critic 30x (1.5e-5) / actor 2x (1.0e-6) ------------------------------
export MEGATRON_CONFIG_PATH="${MEGATRON_CONFIG_PATH:-${SCRIPT_DIR}/megatron_roles_tmax15k_criticlr30x_actorlr2x.yaml}"
# Top-level --lr stays canonical 5e-7; the YAML actor override raises it to 1e-6.
export LR="${LR:-5e-7}"
# Critic cold-start warmup: 2 critic-only steps before the actor starts training.
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-2}"

# --- 5 context walls + train-side length, all 64k (the proven-safe window) ------------
export SEQ_LENGTH="${SEQ_LENGTH:-65536}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-65536}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-65536}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-65536}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"

# --- batch / n / oversample / schedule ------------------------------------------------
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
# >>BATCH 384 (2026-07-12): 1.5x the prior 256. GLOBAL == ROLLOUT (n=1).
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-384}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-384}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"
# standard mode: OVER_SAMPLING_BATCH_SIZE IS the skim window (fly 390, train first 384,
# cancel slow tail). 390 = ~1.016x faithful (same ratio as old 260/256).
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-390}"
# length reward OFF (n=1 -> group-rank shaping is a no-op; make it explicit).
export HARBOR_LENGTH_REWARD_LAMBDA="${HARBOR_LENGTH_REWARD_LAMBDA:-0.0}"

# 100-step hard ceiling (~2.6 epoch over 14601 @ batch 384). step0/1 = critic warmup.
export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"

# Rollout-engine fault tolerance ON (multi-node run; evict dead engines from weight-sync).
export USE_FAULT_TOLERANCE="${USE_FAULT_TOLERANCE:-1}"
export SLIME_GLOO_GROUP_TIMEOUT_MINUTES="${SLIME_GLOO_GROUP_TIMEOUT_MINUTES:-60}"

# tmax-15k batch-384 + PINNED-SPEC YAML (1 CPU/2G/6G sandboxes, concurrency 400).
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_tmax15k_strongest_b384_pinnedspec_64k.yaml}"

# --- run identity: independent dirs/wandb so nothing collides with prior runs ---------
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_tmax15k_ppo_criticlr30x_b384_pinnedspec}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_tmax15k_ppo_criticlr30x_b384_pinnedspec}"
export WANDB_GROUP="${WANDB_GROUP:-q35-27b-tmax15k-ppo-criticlr30x-actorlr2x-standard-b384-pinnedspec-64k-n1}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtmax15k35b384}"

# Daytona build-context S3 proxy (obstore ignores HTTPS_PROXY; patched in rollout.py).
export HARBOR_DAYTONA_S3_PROXY_URL="${HARBOR_DAYTONA_S3_PROXY_URL:-http://star-proxy.oa.com:3128}"
# Daytona create rate limit (600/min per-org): pace the oversample burst to 6/s.
export HARBOR_CREATE_RATE_PER_SEC="${HARBOR_CREATE_RATE_PER_SEC:-6.0}"

# Everything else inherited unchanged from the 96gpu overfit PPO wrapper.
exec "${SCRIPT_DIR}/run_tb2_full89_96gpu_27b_daytona_ppo_lenreward_overfit_fully_async.sh"
