#!/bin/bash
#
# "STRONGEST RUN" wrapper: Qwen3.6-27B + tmax-15k + best PPO recipe, on 120 GPU.
# 2026-07-11.
#
# WHAT THIS IS
#   The strongest single run to date: the BEST DATA (tmax-15k, 14601 synthesized
#   terminal tasks) x the BEST RECIPE (cold-start critic + high critic-lr, standard
#   skim-oversample mode, checkpoint-isolation fix) x the NEWEST MODEL (Qwen3.6-27B).
#   It is a thin shell over run_tb2_full89_96gpu_27b_daytona_ppo_lenreward_overfit_
#   fully_async.sh -> run_qwen35_9b_terminal_bench2_64h20.sh and changes ONLY what this
#   experiment requires. Everything else (PPO eps-clip 0.2, KL off, entropy off, no
#   rollout-logprobs, raw ckpt load, offload alloc-conf, NCCL/Gloo timeout patches,
#   Daytona infra) is inherited unchanged.
#
# WHY THIS COMBINATION (see PPO_INTEGRATION_NOTES.md + memory critic-coldstart-lr10x)
#   The 10x and 20x synth-2820 runs PROVED cold-start critic + high critic-lr gets
#   explained_variance from -39 to +0.11 (20x: first positive step30). But reward
#   stayed FLAT across BOTH lr's and BOTH oversampling values -> the bottleneck is the
#   DATA, not the critic. So this run swaps synth-2820 (2820 tasks, reward-saturated at
#   2-3 epochs) for tmax-15k (14601 tasks, far larger/more diverse) to give reward room
#   to actually climb, and pushes critic-lr to 30x + actor-lr to 2x on top.
#
# THIS RUN'S CHANGES vs the historical tmax-15k fully_async wrapper
#   - MODEL: Qwen3.5-27B -> Qwen3.6-27B (architecture identical; reuses the 3.5-27B spec).
#   - critic COLD START: --load points at the 3.6 base torch_dist which has NO _critic
#     sibling, so the value head auto-reinitializes (model.py:_reinitialize_critic_
#     output_layer). NUM_CRITIC_ONLY_STEPS=2 warms it before the actor starts.
#   - critic lr 30x (1.5e-5) / actor lr 2x (1.0e-6) via --megatron-config-path YAML.
#   - ROLLOUT MODE: standard (fully_async OFF). The starter hardcodes fully_async in its
#     env, so this wrapper FORCE-OVERRIDES it (unconditional export below, no :- default).
#     standard skim-oversample was proven stable across synth-2820's 55/55 steps.
#   - oversample 260 (near-1.0x faithful sampling; standard mode reads OVER_SAMPLING_
#     BATCH_SIZE as the skim window, NOT the yaml max_concurrent_groups).
#   - NUM_ROLLOUT=100 (hard step ceiling; ~1.75 epoch over 14601 @ batch 256).
#
# TIMEOUTS UNCHANGED (user requirement "rollout timeout 不变"): all harbor per-trial /
#   eval / remote timeouts stay at the tmax yaml's values. This wrapper does NOT touch
#   override_timeout_sec / harbor_trial_remote_timeout_sec / harbor_eval_total_timeout_sec.
#
# TOPOLOGY comes from the LAUNCH script (launch_tmax15k_qwen36_strongest_120gpu.sh):
#   actor 8 nodes PP16xCP2xTP2 + MAX_TOKENS_PER_GPU=8192 (the §Z 15-node backward-OOM
#   fix), rollout 7 nodes = 56 GPU, EXPECTED_NODES=15, total 120 GPU (excl bad-NIC .206).
#   This wrapper deliberately does NOT set topology/MAX_TOKENS so the launch script owns it.
#
# DATA: node-LOCAL /tmp/tmax_harbor_15k_full (staged from shared-fs tar by the starter).

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# --- ROLLOUT MODE: force standard (starter hardcodes fully_async; must override) ------
export HARBOR_ROLLOUT_MODE="standard"

# --- MODEL: Qwen3.6-27B (raw torch_dist load; cold-start critic from base, no _critic) -
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR:-/path/to/workspace/code/slime-v0.3-harbor}/scripts/models/qwen3.5-27B.sh}"
export MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-raw}"
export HF_CKPT="${HF_CKPT:-/path/to/models/Qwen3.6-27B}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/workspace/.slime_load_roots/synth3k_ppo_qwen36/Qwen3.6-27B_base}"
# Stage the 3.6 HF + torch_dist to node-local /tmp (cold-start is safe to stage; the
# 51G base read from shared FS is the cold-start bottleneck — see memory harbor-torch-dist-staging-speedup).
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-1}"
export STAGED_HF_CKPT="${STAGED_HF_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.6-27B}"
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-1}"
export STAGED_TORCH_DIST_CKPT="${STAGED_TORCH_DIST_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.6-27B_torch_dist}"

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
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"
# standard mode: OVER_SAMPLING_BATCH_SIZE IS the skim window (fly 260, train first 256,
# cancel slow tail). 260 = near-1.0x faithful (vs 280's 1.09x); more hard tasks survive.
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-260}"
# length reward OFF (n=1 -> group-rank shaping is a no-op; make it explicit).
export HARBOR_LENGTH_REWARD_LAMBDA="${HARBOR_LENGTH_REWARD_LAMBDA:-0.0}"

# 100-step hard ceiling (~1.75 epoch over 14601 @ batch 256). step0/1 = critic warmup.
export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"

# Rollout-engine fault tolerance ON (15-node run; a dead engine must be evicted from the
# weight-sync set or the rendezvous hangs — see relaunch_tmax15k_120gpu_no206.sh notes).
export USE_FAULT_TOLERANCE="${USE_FAULT_TOLERANCE:-1}"
export SLIME_GLOO_GROUP_TIMEOUT_MINUTES="${SLIME_GLOO_GROUP_TIMEOUT_MINUTES:-60}"

# tmax-15k data/timeout YAML reused UNCHANGED (timeouts stay put per user requirement).
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_tmax15k_full_96gpu_27b_daytona_ppo_64k.yaml}"

# --- run identity: independent dirs/wandb so nothing collides with prior runs ---------
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen36_27b_tmax15k_ppo_criticlr30x_actorlr2x_strongest}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.6-27B_tmax15k_ppo_criticlr30x_actorlr2x_strongest}"
export WANDB_GROUP="${WANDB_GROUP:-q36-27b-tmax15k-ppo-criticlr30x-actorlr2x-standard-os260-64k-n1-strongest}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtmax15k36}"

# Daytona build-context S3 proxy (obstore ignores HTTPS_PROXY; patched in rollout.py).
export HARBOR_DAYTONA_S3_PROXY_URL="${HARBOR_DAYTONA_S3_PROXY_URL:-http://star-proxy.oa.com:3128}"
# Daytona create rate limit (600/min per-org): pace the oversample burst to 6/s.
export HARBOR_CREATE_RATE_PER_SEC="${HARBOR_CREATE_RATE_PER_SEC:-6.0}"

# Everything else inherited unchanged from the 96gpu overfit PPO wrapper.
exec "${SCRIPT_DIR}/run_tb2_full89_96gpu_27b_daytona_ppo_lenreward_overfit_fully_async.sh"
