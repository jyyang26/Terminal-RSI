#!/bin/bash
#
# PPO 64k REAL-RUN wrapper: Terminal-Bench 2 FULL 89 tasks on Daytona, 96-GPU 27B.
#
# WHAT THIS IS
#   The "40-step" overfit PPO run, taken back down from 96k to the PROVEN-SAFE 64k
#   full-length window. It is a thin shell over
#   run_tb2_full89_96gpu_27b_daytona_ppo_lenreward_overfit_fully_async.sh: it overrides
#   ONLY the context walls (5 of them) + the SGLang mem-fraction back to the values from
#   the 64k baseline ppo_033051 (Jun 22), which trained the full overfit budget cleanly
#   (eval 27.5%->38.6%, zero deadlock/OOM). Everything else is inherited unchanged:
#   CP2xPP8xTP2 topology, NUM_ROLLOUT=40, SAVE_INTERVAL=10, batch64/n4/256 concurrency,
#   PPO (advantage-estimator ppo, eps-clip 0.2, critic-warmup 1), length-reward lambda
#   0.4, KL OFF, no rollout-logprobs, entropy off, periodic TB2 eval every 10 steps.
#
# WHY 64k AND NOT 96k
#   debug-train-only (--load-debug-rollout-data) deadlocks at step0 critic forward mb0
#   in EVERY window we tried (64k included), while the IDENTICAL dump trains in 5:21 in a
#   REAL run -- so the deadlock is a debug-train-only MODE artifact, NOT a 96k / topology
#   / uncommitted-code problem (reverting the 3 core files did not fix it). See memory
#   debug-train-only-deadlocks-at-step0-not-real-run. We therefore stop probing 96k via
#   debug-train and go back to a REAL run at the length we KNOW trains end-to-end: 64k.
#
# THE 5 CONTEXT WALLS (all lowered to 65536 / 64000; lockstep or 64k gets silently
# truncated or rejected somewhere):
#   1) ROLLOUT_MAX_CONTEXT_LEN   65536  -- slime per-call prompt+gen cap (llm.py:104)
#   2) SGLANG_CONTEXT_LENGTH     65536  -- SGLang engine KV window (else 400-rejects)
#   3) SGLANG_MAX_PREFILL_TOKENS 65536  -- single-prefill cap
#   4) max_context_tokens        64000  -- Terminus2 agent's OWN budget; lives in the
#                                          64k YAML (env can't reach it), NOT here.
#   5) HARBOR_MAX_TRAIN_TOTAL_TOKENS 65536 -- train-side hard trim (sample_postprocess)
#   + SEQ_LENGTH 65536 = --seq-length / --max-position-embeddings (train-side legal max).
#
# MEM-FRACTION back to 0.75: the 96k run dropped it to 0.6 because 96k long-context
# prefill/decode OOM'd the engines mid-step0. At 64k that pressure is far lower and the
# baseline ran clean at 0.75, so restore the baseline value (more KV headroom).

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# --- 5 context walls + train-side length, all down to 64k (matches ppo_033051) ---
export SEQ_LENGTH="${SEQ_LENGTH:-65536}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-65536}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-65536}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-65536}"
# microbatch cap = MAX_TOKENS_PER_GPU x cp2 = 32768 x 2 = 65536 = 64k -> one 64k seq
# fills exactly one microbatch and forwards whole (never split/truncated).
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"

# 64k OOM headroom is comfortable; restore the baseline's 0.75 (96k needed 0.6).
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"

# Rollout-engine fault tolerance: OFF for this rerun (revert-to-known-good experiment).
#
# WHY OFF NOW: the FT run cp2_64k_ft_20260627_151656 (Plan A) wedged EARLIER than the
# non-FT run -- not at step0->step1 broadcast, but at the INITIAL weight sync
# (train_async.py:38) inside disconnect_rollout_engines_from_distributed (ray.get(refs)
# on engine.destroy_weights_update_group, NO timeout), with NO engine death. So FT did
# not help and its altered update_weights path is itself suspect. This rerun instead
# reproduces the PROVEN-GOOD ppon4_055519 (commit 6306aaa3, clean core, NO FT, ran
# critic-step 0->6 with zero engine crashes): 3 core files reverted to 6306aaa3 and FT
# turned OFF here. If a sporadic engine OOM recurs WITHOUT the 93-line core changes, that
# isolates the cause; FT can be re-enabled later as a separate variable.
export USE_FAULT_TOLERANCE="${USE_FAULT_TOLERANCE:-0}"
export ROLLOUT_HEALTH_CHECK_INTERVAL="${ROLLOUT_HEALTH_CHECK_INTERVAL:-30}"
export ROLLOUT_HEALTH_CHECK_TIMEOUT="${ROLLOUT_HEALTH_CHECK_TIMEOUT:-120}"
export ROLLOUT_HEALTH_CHECK_FIRST_WAIT="${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-600}"

# Topology unchanged from the 96k wrapper (CP2 x PP8 x TP2) -- left to its defaults.

# Point at the 64k YAML (max_context_tokens=64000 + its own trials_dir).
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_tb2_full89_96gpu_27b_daytona_ppo_lenreward_overfit_64k.yaml}"

# Separate run identity so 64k artifacts/ray-temp/wandb don't collide with the 96k run.
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_overfit_ppo_lenreward_cp2_64k}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_tb2_full89_overfit_ppo_lenreward_cp2_64k}"
export WANDB_GROUP="${WANDB_GROUP:-q35-27b-tb2-full89-overfit-ppo-lenreward-l0.4-noKL-nologprob-cp2-64k-to3600-criticwarmup1-memfrac0.75}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtb2ppo64k}"

# Everything else (NUM_ROLLOUT=40, SAVE_INTERVAL=10, batch/n, PPO/critic flags, raw ckpt
# load, alloc conf, new_group timeout floor, Daytona infra) is inherited unchanged from
# the 96k overfit PPO wrapper.
exec "${SCRIPT_DIR}/run_tb2_full89_96gpu_27b_daytona_ppo_lenreward_overfit_fully_async.sh"
