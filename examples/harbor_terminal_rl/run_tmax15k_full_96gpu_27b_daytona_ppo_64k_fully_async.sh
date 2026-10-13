#!/bin/bash
#
# tmax-15k FULL real-run wrapper: 14601 synthesized terminal tasks on Daytona, 96-GPU 27B.
#
# WHAT THIS IS
#   The first REAL (non-overfit) training run on the tmax-15k corpus. It is a thin shell
#   over run_tb2_full89_96gpu_27b_daytona_ppo_lenreward_overfit_fully_async.sh that keeps
#   the PROVEN PPO recipe + 64k window and changes ONLY what this experiment requires:
#     - DATA: train pool -> tmax-15k (14601 tasks, node-local /tmp); eval -> TB2 89 (held-out).
#     - n=4 -> n=1 (each task rolled out once).
#     - dynamic oversample: fly 280 groups, train on first 256, drop the slow tail.
#     - length reward OFF (lambda=0): n=1 makes the group-rank length shaping a no-op anyway.
#     - NUM_ROLLOUT=57 (= 1 epoch over 14601 tasks at batch 256: 14601/256 = 57.0).
#     - SAVE every 5 steps, EVAL every 10 steps.
#   Everything else (CP2xPP8xTP2 topology, PPO eps-clip 0.2, critic-warmup 1, KL off,
#   no rollout-logprobs, entropy off, raw ckpt load, NCCL 30min + Gloo 60min timeout
#   patches, Daytona infra) is inherited unchanged from the 64k overfit wrapper.
#
# WHY 64k
#   The proven-safe full-length window: the overfit runs trained end-to-end at 64k with
#   zero deadlock/OOM. We reuse it verbatim (5 context walls at 65536, mem-fraction 0.75).
#
# DATA LOCATION (IMPORTANT)
#   tmax-15k is on each node's LOCAL /tmp at /tmp/tmax_harbor_15k_full, NOT shared fs.
#   The start_*_tmux.sh launcher extracts it on every node from the shared-fs tar before
#   the run. HarborTrialWorker reads task dirs node-locally, so every rollout node MUST
#   have the data at this exact path. The TB2 eval set stays on shared fs.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# --- 5 context walls + train-side length, all 64k (the proven-safe window) ---
export SEQ_LENGTH="${SEQ_LENGTH:-65536}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-65536}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-65536}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-65536}"
# microbatch cap = MAX_TOKENS_PER_GPU x cp2 = 32768 x 2 = 65536 = one 64k seq per mb.
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"

# --- THIS RUN'S CHANGES: batch / n / oversample / length-reward / schedule ---
# n=1: each task rolled out once. rollout_batch 256 x n1 = global 256 (same train batch
# as the overfit run's 64x4). Dynamic global batch ON so the trainer takes whatever the
# 256 accepted groups produce.
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"
# Oversample: collect 280 (=max_concurrent_groups in the yaml) but train on 256.
# OVER_SAMPLING_BATCH_SIZE is forwarded to --over-sampling-batch-size for the data source.
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-280}"

# length reward OFF (n=1 -> group-rank shaping is a no-op; make it explicit).
export HARBOR_LENGTH_REWARD_LAMBDA="${HARBOR_LENGTH_REWARD_LAMBDA:-0.0}"

# 1 epoch over 14601 tasks @ batch 256 = 57 steps. step0 = critic-only warmup.
export NUM_ROLLOUT="${NUM_ROLLOUT:-57}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"

# Rollout-engine fault tolerance OFF (same as the proven overfit run).
export USE_FAULT_TOLERANCE="${USE_FAULT_TOLERANCE:-0}"

# Gloo barrier timeout patch (CRASH#2 fix): 60min. NCCL group timeout (CRASH#1 fix) is
# already defaulted to 30min by the base wrapper's SLIME_NEW_GROUP_TIMEOUT_MINUTES.
export SLIME_GLOO_GROUP_TIMEOUT_MINUTES="${SLIME_GLOO_GROUP_TIMEOUT_MINUTES:-60}"

# Point at the tmax-15k 64k YAML (local-/tmp train pool, TB2-89 eval pool, oversample 280).
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_tmax15k_full_96gpu_27b_daytona_ppo_64k.yaml}"

# Separate run identity so tmax-15k artifacts/ray-temp/wandb don't collide with overfit.
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_tmax15k_full_ppo_64k}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_tmax15k_full_ppo_64k}"
export WANDB_GROUP="${WANDB_GROUP:-q35-27b-tmax15k-full-ppo-noLenReward-noKL-nologprob-cp2-64k-n1-oversample280-criticwarmup1}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtmax15k}"

# Daytona build-context S3 proxy (2026-07-01 fix): the SDK's obstore S3 client does
# NOT read HTTPS_PROXY and builds S3Store inside isolated_env() (clears os.environ),
# so build-context uploads to prod-daytona-build-context time out on these nodes.
# ensure_daytona_s3_proxy() (rollout.py::_run_one_trial) patches S3Store's
# client_options.proxy_url from this var. See PPO_INTEGRATION_NOTES.md.
export HARBOR_DAYTONA_S3_PROXY_URL="${HARBOR_DAYTONA_S3_PROXY_URL:-http://star-proxy.oa.com:3128}"

# Daytona create rate limit (600/min per-org): pace sandbox .create() to 6/s so the
# 280-group burst at rollout start doesn't exceed it. (Not the root blocker — proxy was
# — but the limit is real.) Read via getattr in rollout.py; also set in the yaml.
export HARBOR_CREATE_RATE_PER_SEC="${HARBOR_CREATE_RATE_PER_SEC:-6.0}"

# Everything else (PPO/critic flags, raw ckpt load, alloc conf, topology, Daytona infra)
# is inherited unchanged from the 96k overfit PPO wrapper.
exec "${SCRIPT_DIR}/run_tb2_full89_96gpu_27b_daytona_ppo_lenreward_overfit_fully_async.sh"
