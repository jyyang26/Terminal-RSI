#!/bin/bash
#
# synth-3k REAL-run wrapper: 2820 synthesized terminal tasks on Daytona, 120-GPU 27B,
# PPO with CTRF PROCESS REWARD (pass-ratio) and a WARM-LOADED critic.
#
# WHAT THIS IS
#   A thin shell over run_tb2_full89_96gpu_27b_daytona_ppo_lenreward_overfit_fully_async.sh
#   (the proven PPO + 64k recipe). It keeps that recipe and changes ONLY what this
#   experiment requires:
#     - DATA: train pool -> synth-2820 (shared-fs SYMLINK dir, NO /tmp staging);
#             eval -> TB2-89 (held-out).
#     - REWARD: process reward = CTRF pass-ratio via partial_reward, with
#             HARBOR_PARTIAL_REWARD_MAX=1.0 (pure ratio: shaped = max(terminal, passed/total)).
#             length reward OFF (lambda=0; n=1 makes it a no-op anyway).
#     - EPOCHS: 2 epochs over 2820 @ batch 256 = 2 * ceil(2820/256=11.0) = 22 steps.
#     - SANDBOX: 1 CPU / 2 GB / 6 GB per sandbox (override_* in the yaml).
#     - CRITIC WARM-LOAD: actor cold-starts from the 27B base (release ckpt ->
#             start_rollout_id=0), critic warm-loads its trained value head from run
#             20260703_001109 iter_56. Wired WITHOUT touching slime core, via a
#             LOAD-ROOT of symlinks: PRETRAINED_CKPT points at
#             <LOAD_ROOT>/Qwen3.5-27B_base (-> 27B base torch_dist), and its sibling
#             <LOAD_ROOT>/Qwen3.5-27B_base_critic (-> 001109 _critic) is what
#             slime's _isolate_critic_checkpoint_dirs auto-loads for the critic.
#             NUM_CRITIC_ONLY_STEPS=0 (value head already trained; skip warmup).
#     - STAGE_TORCH_DIST_CKPT_TO_TMP=0: actor loads the base from SHARED FS (not /tmp)
#             so the critic's <load>_critic symlink resolves on the same shared path.
#             (The critic ckpt is 397G and must be read from shared fs regardless.)
#     - SAVE every 5 steps, EVAL every 10 steps.
#   Everything else (CP2xPP16xTP2 topology on 120 GPU, PPO eps-clip 0.2, KL off,
#   no rollout-logprobs, entropy off, NCCL 30min + Gloo 60min timeout patches,
#   Daytona infra + S3 proxy) is inherited unchanged from the 64k overfit wrapper.
#
# WHY start_rollout_id is safe with a warm critic
#   placement_group.py sets start_rollout_id from the CRITIC when use_critic (would
#   be 57 from iter_56, which would make range(57,22) an empty loop). But arguments.py
#   sets args.start_rollout_id=0 first because the actor's base ckpt marker is
#   "release", and placement_group only overrides when it is None. So the run correctly
#   starts at rollout_id=0. (Verified against the code paths before launch.)

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# --- 5 context walls + train-side length, all 64k (the proven-safe window) ---
export SEQ_LENGTH="${SEQ_LENGTH:-65536}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-65536}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-65536}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-65536}"
# MAX_TOKENS_PER_GPU=8192 is the PROVEN §Z backward-OOM fix on PP16xCP2 (a 64k seq is
# split into 64k/8192=8 microbatches; backward peaked at 100% but did NOT OOM). The
# relaunch orchestrator's TOPO_ENV also sets this; keep the two in sync.
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"

# --- batch / n / oversample / length-reward / schedule ---
# n=1: each task rolled out once. rollout_batch 256 x n1 = global 256.
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"
# >>2026-07-07 撇油偏差修复（见 PPO_INTEGRATION_NOTES.md §11）：退出 fully-async。
# fully_async 的持久 worker 会 backfill，后期步实际 churn 580-735 组只留最快 256，
# 有效超采样比冲到 3.9×，系统性偏向短/易任务 → step0 reward 虚高、随后随批次难度上升而跌。
# 改用 standard 模式：每步新建任务集、发 OVER_SAMPLING 个、收最先完成的 256、cancel 慢尾、
# 不 backfill。OVER_SAMPLING=288 (1.125×) 保留极小余量防单个怪物任务卡死单步，撇油比从
# 3.9× 降到 1.125×。standard 模式下 yaml 的 harbor_fully_async_max_concurrent_groups 不再被读取。
export HARBOR_ROLLOUT_MODE="${HARBOR_ROLLOUT_MODE:-standard}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-288}"

# 2 epochs over 2820 tasks @ batch 256: ceil(2820/256)=11 steps/epoch -> 22 steps.
export NUM_ROLLOUT="${NUM_ROLLOUT:-22}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
# NOTE: use ${VAR-default} (not ${VAR:-default}) so an explicit EVAL_INTERVAL="" from
# the caller survives and disables periodic eval (downstream wrapper emits
# --skip-eval-before-train when EVAL_INTERVAL is empty). ${VAR:-default} would treat
# "" as unset and force it back to 10.
export EVAL_INTERVAL="${EVAL_INTERVAL-10}"

# --- PROCESS REWARD: CTRF pass-ratio, PURE (max=1.0). length reward OFF. ---
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.partial_reward.trajectory_grouped_progress_rewards}"
export HARBOR_PARTIAL_REWARD_MAX="${HARBOR_PARTIAL_REWARD_MAX:-1.0}"
export HARBOR_LENGTH_REWARD_LAMBDA="${HARBOR_LENGTH_REWARD_LAMBDA:-0.0}"

# --- CRITIC WARM-LOAD via load-root symlinks (no core changes) ---
# LOAD_ROOT/Qwen3.5-27B_base        -> 27B base torch_dist (actor cold start, marker "release")
# LOAD_ROOT/Qwen3.5-27B_base_critic -> 001109 iter_56 critic (auto-loaded by _isolate_*)
export SYNTH3K_LOAD_ROOT="${SYNTH3K_LOAD_ROOT:-/path/to/workspace/.slime_load_roots/synth3k_ppo}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-${SYNTH3K_LOAD_ROOT}/Qwen3.5-27B_base}"
export HF_CKPT="${HF_CKPT:-/path/to/models/Qwen3.5-27B}"
# Load actor base from SHARED FS (NOT /tmp) so <load>_critic symlink resolves.
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-0}"
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-1}"
# Critic value head is already trained -> skip critic-only warmup.
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-0}"

# Rollout-engine fault tolerance (recover dead SGLang engines before weight-sync).
export USE_FAULT_TOLERANCE="${USE_FAULT_TOLERANCE:-1}"

# Gloo barrier timeout patch (CRASH#2 fix): 60min. NCCL group timeout (CRASH#1 fix) is
# already defaulted to 30min by the base wrapper's SLIME_NEW_GROUP_TIMEOUT_MINUTES.
export SLIME_GLOO_GROUP_TIMEOUT_MINUTES="${SLIME_GLOO_GROUP_TIMEOUT_MINUTES:-60}"

# Point at the synth-3k 64k YAML (shared-fs symlink train pool, TB2-89 eval, oversample 280).
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_synth3k_2820_96gpu_27b_daytona_ppo_partial_64k.yaml}"

# Separate run identity so synth-3k artifacts/ray-temp/wandb don't collide with others.
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_synth3k_2820_ppo_partial_64k}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_synth3k_2820_ppo_partial_64k}"
export WANDB_GROUP="${WANDB_GROUP:-q35-27b-synth3k-2820-ppo-partialReward1.0-noLenReward-noKL-nologprob-cp2-64k-n1-standardOS288-criticWarm001109iter56-2epoch}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rsynth3k}"

# Daytona build-context S3 proxy (2026-07-01 fix): the SDK's obstore S3 client does
# NOT read HTTPS_PROXY and builds S3Store inside isolated_env() (clears os.environ),
# so build-context uploads to prod-daytona-build-context time out on these nodes.
export HARBOR_DAYTONA_S3_PROXY_URL="${HARBOR_DAYTONA_S3_PROXY_URL:-http://star-proxy.oa.com:3128}"

# Daytona create rate limit (600/min per-org): pace sandbox .create() to 6/s.
export HARBOR_CREATE_RATE_PER_SEC="${HARBOR_CREATE_RATE_PER_SEC:-6.0}"

# Everything else (PPO/critic flags, raw ckpt load, alloc conf, topology, Daytona infra)
# is inherited unchanged from the 96k overfit PPO wrapper.
exec "${SCRIPT_DIR}/run_tb2_full89_96gpu_27b_daytona_ppo_lenreward_overfit_fully_async.sh"
