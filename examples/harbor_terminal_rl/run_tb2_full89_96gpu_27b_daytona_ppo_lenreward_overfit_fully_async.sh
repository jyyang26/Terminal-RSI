#!/bin/bash
#
# PPO run-wrapper: Terminal-Bench 2 FULL 89 tasks on Daytona, 96-GPU 27B.
# A/B sibling of run_tb2_full89_96gpu_27b_daytona_gspo_lenreward_overfit_fully_async.sh.
# IDENTICAL recipe (raw mode, length-reward lambda 0.4, KL OFF, no rollout-logprobs,
# entropy OFF + display, periodic TB2 eval, same parallelism/batch/Daytona infra) with
# ONE algorithmic change: GSPO -> PPO (actor-critic, GAE).
#
#   >>ESTIMATOR: ADVANTAGE_ESTIMATOR=ppo. arguments.py derives use_critic=True
#       (slime/utils/arguments.py:1818). This allocates an INDEPENDENT 27B critic
#       train-group (value head output_size=1) that SHARES the actor's placement
#       group (slime/ray/placement_group.py: result["critic"] = result["actor"]),
#       so PPO uses the SAME 96 GPUs as the GSPO run -- NO extra cards. actor and
#       critic time-share GPU memory via forced offload (use_critic ->
#       offload_train=True, arguments.py:1862) and train SERIALLY each step
#       (train_async.py:72-75: critic.async_train then actor.async_train(external_data=values)).
#
#   >>EPS_CLIP: 0.2 (PPO standard per-token-ratio clip), NOT the 4e-4 used by GSPO.
#       GSPO's ratio is a length-normalized geometric mean (variance ~1) so it needs
#       a tiny clip; PPO clips the raw per-token ratio and 4e-4 would essentially
#       freeze updates. 0.2 matches the repo's geo3k PPO example.
#
#   >>CRITIC WARMUP: NUM_CRITIC_ONLY_STEPS=1. The first rollout step trains the
#       critic ONLY (actor frozen) to warm up value estimates before policy updates
#       begin (train_async.py: actor_trains_this_step = rollout_id >= N). Matches the
#       geo3k PPO example (--num-critic-only-steps 1).
#
#   >>SMOKE LENGTH: NUM_ROLLOUT=8 (NOT 28). This is a first PPO bring-up on 27B in
#       this pipeline; the main unknowns are (a) memory peak with a second 27B model
#       (critic weights + optimizer state) time-shared via offload, and (b) per-step
#       wall time (critic+actor serial, each a full 27B fwd+bwd). Run 8 steps to
#       confirm the critic trains end-to-end without OOM, then decide whether to
#       extend to the full 28-step overfit budget. Bump NUM_ROLLOUT=28 to go full.
#
#   >>CRITIC CHECKPOINT: raw mode. The critic LM backbone loads from the SAME
#       torch_dist --load/--ref-load as the actor (placement_group.py uses
#       copy.deepcopy(args) for the critic when no --megatron-config-path); the
#       value head (output_size=1) is auto-reinitialized on shape-mismatch
#       (model.py:_reinitialize_critic_output_layer, normal_(0,0.02)). No separate
#       critic checkpoint is needed. bridge mode is NOT usable for 27B (stock GDN
#       NotImplementedError on packed-seq), and the critic shares the same Megatron
#       forward, so stay on raw.
#
# KEPT from the GSPO overfit run: raw (torch_dist) checkpoint load, length-reward
# shaping (lambda 0.4), token-cap postprocess, entropy OFF + display, KL OFF, no
# rollout-logprobs, all parallelism / batch / Daytona infra + leak fixes,
# override_timeout 3600s, periodic TB2 eval every 10 steps, NO step-0 baseline eval.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CODE_ROOT="${CODE_ROOT:-/path/to/workspace/code}"

export CODE_ROOT
export SLIME_DIR="${SLIME_DIR:-${CODE_ROOT}/slime-v0.3-harbor}"
export HARBOR_SRC="${HARBOR_SRC:-${CODE_ROOT}/harbor/src}"
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-27B.sh}"

export HARBOR_ROLLOUT_MODE="${HARBOR_ROLLOUT_MODE:-fully_async}"
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_tb2_full89_96gpu_27b_daytona_ppo_lenreward_overfit.yaml}"

# Separate save dir so this PPO run's checkpoints don't collide with the GSPO run.
# >>正式 CP2@96k overfit (2026-06-26): 独立 save dir, 不和之前的 smoke / 64k-128k 探索
# run 的 ckpt 混淆。cp2_96k 后缀标明这是验证过 new_group timeout 修复后的正式 96k run。
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_overfit_ppo_lenreward_cp2_96k}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_tb2_full89_overfit_ppo_lenreward_cp2_96k}"

# raw (torch_dist) checkpoint load -- the PROVEN training path; bridge crashes on
# 27B packed-seq GDN. actor AND critic load the LM backbone from this torch_dist
# ckpt via --ref-load/--load; the critic value head is auto-reinitialized.
export MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-raw}"
export HF_CKPT="${HF_CKPT:-/path/to/models/Qwen3.5-27B}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/models/Qwen3.5-27B_torch_dist_pp8_timeout60}"
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-1}"
export STAGED_HF_CKPT="${STAGED_HF_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.5-27B}"
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-1}"
export STAGED_TORCH_DIST_CKPT="${STAGED_TORCH_DIST_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.5-27B_torch_dist_pp8_timeout60}"

# >>CP2-96k (2026-06-25): 12 nodes = 4 actor (32 GPU = TP2×CP2×PP8) + 8 rollout (64 GPU
# = 32 SGLang engines). This is EXACTLY the 64k baseline topology (tb2of_raw_20260620:
# TP2×CP2×PP8, actor 4 nodes, rollout 64) — the ONLY config that ran all 28 steps with
# zero deadlock — except we raise the sequence wall 64k->96k. The critic shares the
# actor's 32 GPUs (no extra). See PPO_INTEGRATION_NOTES.md §5.9 节7 for why CP2@96k is
# the most promising point: CP2 avoids the CP4 deadlock (validated) AND shards the
# non-GDN activations 2x (relieves the near-miss 97/95GB OOM that Plan-B hit at 128k);
# 96k is 25% shorter than 128k so both the OOM margin and GDN-kernel time ease back.
export EXPECTED_NODES="${EXPECTED_NODES:-12}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-64}"

# 27B parallelism CP2-96k (2026-06-25): TP2*CP2*PP8=32 (actor 32 GPU, no DP). This is
# the VALIDATED-no-deadlock 64k-baseline topology (tb2of_raw_20260620 ran all 28 steps),
# now pushed to a 96k sequence wall. History of why NOT the other CP values:
#  - CP4 (C-1/C-2/Plan-A): GDN all_gather × PP pipeline schedule -> lazy NCCL comm-init
#    cross-locks -> watchdog hang. The deadlock IS CP>=4 here; eager-init only got further
#    (a subgroup PG575 still hung). See §5.9.
#  - CP1 (CP1-B): no deadlock, but per-rank computes the FULL sequence with NO activation
#    sharding -> at 128k it sat at 92/97GB (allocator thrash) and the GDN fla kernels were
#    pathologically slow (21min/microbatch). CP1 is the worst memory point.
#  - CP2 is the sweet spot: avoids the CP4 deadlock (proven) AND shards non-GDN activations
#    2x (48k/rank at 96k vs CP1's full 96k), relieving the memory pressure. KEY FACT: CP
#    does NOT shard GDN compute (hf_attention.py:147 runs GDN on the full all_gather'd
#    sequence regardless of CP), so GDN-kernel speed is identical across CP — CP2 is chosen
#    purely for the no-deadlock + memory-relief balance, not speed.
# PP8 matches the existing torch_dist_pp8 checkpoint (no reshard needed) and is the
# baseline's PP. recompute full is kept.
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-8}"
export CP_SIZE="${CP_SIZE:-2}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-1}"
export QKV_FORMAT="${QKV_FORMAT:-thd}"

# NVTE_BATCH_MHA_P2P_COMM=1: at CP2 the TE full-attn-layer CP P2P auto-uses the safe
# batched path anyway (cp_size==2 + cc<10.0, §5.7), so this is effectively a no-op here;
# left set as documented history. The CP1/C-1/C-2/Plan-A deadlock saga is in §5.7–5.9.
export NVTE_BATCH_MHA_P2P_COMM="${NVTE_BATCH_MHA_P2P_COMM:-1}"

# >>NCCL-FIX (2026-06-26): the new_group default-timeout floor that fixed the CP2@96k
# COALESCED deadlock. new_group() does NOT inherit the init_process_group timeout, so a
# GDN CP coalescing subgroup created without timeout= fell back to NCCL's 10min default
# and the per-PP-stage GDN Triton autotune skew at 96k blew that window (PG 21, ran 691s
# > 600000ms). monkey_patch_torch_dist (reloadable_process_group.py) now injects this
# floor into every new_group/reload without an explicit timeout. 30min >> worst-case
# compile skew. VALIDATED via debug-train-only (raysubmit_VEswz7RPaU4jB4JW): step0/step1
# critic forward+train all passed, zero watchdog/OOM. See PPO_INTEGRATION_NOTES.md §5.10.
export SLIME_NEW_GROUP_TIMEOUT_MINUTES="${SLIME_NEW_GROUP_TIMEOUT_MINUTES:-30}"

# batch/n = mid830 那套 (2026-06-24): n4/batch64, 训练批 256。overfit 原 16-group 配置
# 喂不饱 32 个 SGLang 引擎 (实测 KV 0.18, #running-req 0-1); batch64/n4 + 并发256 (yaml)
# 让 64 个不同 prompt 同时在飞 (1:1 喂满 32 引擎 x 2 in-flight), rollout 吞吐显著↑。
# n=4 仍够 group advantage。
# >>正式 overfit (2026-06-26): NCCL new_group timeout 修复已验证 (debug-train-only 跑通
# step0/step1 critic forward+train, 零 watchdog/OOM, 稳态 20-38s/mbs)。改回正式 batch:
# batch8→64 / global32→256 / NUM_ROLLOUT 2→40 / SAVE 999→10。yaml 并发同步回 256。
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-64}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"

# >>正式 overfit (2026-06-26): 40 rollout steps (64k 基线 28 步跑完 eval 27.5%→38.6%;
# 40 步给 96k 长上下文更多收敛时间)。step0 = critic-only warmup (num_critic_only_steps=1)。
export NUM_ROLLOUT="${NUM_ROLLOUT:-40}"
# 每 10 步存 actor+critic ckpt (critic 隔离到 <dir>_critic, 见 _isolate_critic_checkpoint_dirs)。
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

# Unified 96k context — LOWERED from 128k (2026-06-25, CP2-96k). Goal: rollout生成最长
# 96k 轨迹, 这些长样本不截断直接进训练. 链路上有 5 道独立的墙, 缺一个都会让 96k 在某处
# 被悄悄截断或被拒, 必须同步设成 98304 (=96*1024):
#   1) ROLLOUT_MAX_CONTEXT_LEN  — slime 单次 LLM 调用 prompt+gen 上限 (llm.py:104 主动拦截)
#   2) SGLANG_CONTEXT_LENGTH    — SGLang 引擎 KV 窗口; 不抬, 引擎直接 400 拒掉 >窗口 请求
#   3) SGLANG_MAX_PREFILL_TOKENS — 单次 prefill 上限; 不抬, 长 prompt prefill 被拒
#   4) max_context_tokens (在 YAML agent.llm_kwargs) — Terminus2 agent 自己的上下文预算,
#      到这个值就摘要/unwind. 不在这里, 改 YAML (设 96000), 否则 agent 永远不让 prompt 超它.
#   5) HARBOR_MAX_TRAIN_TOTAL_TOKENS — 训练侧硬截断 (sample_postprocess.py:_hard_trim_plan):
#      把 total_length > cap 的样本从尾部砍到 cap 再进训练. 不设, 长样本被砍.
# 另: SEQ_LENGTH (= --seq-length / --max-position-embeddings) 是训练侧合法序列上限, >它会越界报错.
# HF config max_position_embeddings=262144, 所以 98304 合法, 且不增加激活显存 (RoPE 即时算).
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-98304}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-98304}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-98304}"
# chunked prefill 削峰: 长 prefill 一次性算占大显存峰值, 分块算更稳. 长上下文建议保持开.
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"

export SEQ_LENGTH="${SEQ_LENGTH:-98304}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-98304}"
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-1}"
# >>96k CP2-96k (2026-06-25): MAX_TOKENS_PER_GPU = 49152. The microbatch cap =
# MAX_TOKENS_PER_GPU × cp_size (dp_schedule.py:213) = 49152 × 2 = 98304 = 96k, so one
# 96k sequence fills exactly one microbatch and forwards whole (never split/truncated).
# Per-rank NON-GDN activation = 49152 (48k) — vs the 64k baseline's 32k/rank (passed) and
# vs Plan-B's 64k/rank at 128k that hit 97/95GB (near-miss OOM). 48k/rank sits between
# those, so OOM is the open risk this smoke answers; 96k is 25% under 128k so the margin
# should land back under 95GB. GDN layers still compute the full 96k/rank (all_gather'd,
# CP doesn't shard GDN), but 96k GDN is 25% lighter than the 128k that was slow at CP1.
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-49152}"

export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
# >>96k OOM-FIX (2026-06-26): 0.75 -> 0.6. The first cp2_96k_overfit run
# (raysubmit_YtuDGrCAp7JPqDVt) FAILED not from the (now-fixed) CP2@96k NCCL deadlock --
# step0 critic forward ran 73/73 clean at steady 24.6s/mb -- but because the rollout
# SGLang ENGINES OOM'd during 96k long-context prefill/decode (137.141: "Triton Error
# [CUDA]: out of memory" @06:37; 137.150: CUBLAS_STATUS_EXECUTION_FAILED @04:50). A dead
# engine then refused the step0->step1 /init_weights_update_group, so update_weights()
# wedged in dist.barrier(gloo) for the full 30min SLIME_NEW_GROUP_TIMEOUT and the job
# died. The engines live on the 8 ROLLOUT nodes (NOT shared with the 4 actor nodes), so
# this is pure engine-side memory pressure: mem_fraction_static=0.75 reserves too much
# for the static KV pool and starves the runtime allocator of headroom for 96k prefill
# activation spikes (chunked_prefill 8192 over a 98304 context). Dropping to 0.6 frees
# ~14GB/card (96GB H20) back to the activation allocator. KV pool is still ample for
# fully-async terminal rollouts (low concurrent decode depth -- perf showed token usage
# ~0.05-0.15). debug-train-only could not surface this: it runs rollout-num-gpus=0 and
# reads dumped rollout data, so no live engine ever exists to OOM. See memory
# cp2-96k-sglang-oom-during-step0-compile.
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"

# >>PPO/OFFLOAD-CRITICAL: use_critic forces offload_train=True (arguments.py:1862),
# which drives actor/critic GPU-memory time-sharing through torch_memory_saver
# (pause/resume at actor.py:197/206, margin at :79). torch_memory_saver's
# _sanity_checks HARD-RAISES if PYTORCH_CUDA_ALLOC_CONF contains
# `expandable_segments:True` ("TorchMemorySaver is disabled ... expandable_segments
# is not supported yet"). The base script defaults this to expandable_segments:True
# (for fragmentation control), which is fine for the non-offload GSPO run but CRASHES
# the PPO run at critic init. Override to a NON-expandable alloc conf so the offload
# path works. Cost: more fragmentation risk without expandable segments; max_split is
# a mild mitigation. (The geo3k PPO example sidesteps this entirely by running on NPU,
# which doesn't use the CUDA torch_memory_saver.)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"

# 27B memory peak control. NO ROLLOUT LOGPROBS.
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-32}"
export USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-0}"
export KEEP_OLD_ACTOR="${KEEP_OLD_ACTOR:-1}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-0}"

# >>PPO actor-critic with GAE. eps-clip 0.2 (standard per-token clip, NOT GSPO's 4e-4).
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-ppo}"
export EPS_CLIP="${EPS_CLIP:-0.2}"
# >>Critic warmup: first step trains critic only (actor frozen) to stabilize values.
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-1}"

# ENTROPY LOSS OFF + entropy display — KEPT.
export USE_CUSTOM_NO_ENTROPY_LOSS="${USE_CUSTOM_NO_ENTROPY_LOSS:-1}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.0}"

# KL OFF — kept from the GSPO run. PPO does not depend on a KL anchor here.
export USE_KL_LOSS="${USE_KL_LOSS:-0}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
export KL_COEF="${KL_COEF:-0.00}"
export LR="${LR:-5e-7}"

# length-aware reward shaping — lambda 0.4, IDENTICAL to the GSPO run (clean A/B).
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.length_shaped_reward.trajectory_grouped_length_shaped_rewards}"
export HARBOR_LENGTH_REWARD_LAMBDA="${HARBOR_LENGTH_REWARD_LAMBDA:-0.4}"
export ROLLOUT_ALL_SAMPLES_PROCESS_PATH="${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples}"
export ROLLOUT_DATA_POSTPROCESS_PATH="${ROLLOUT_DATA_POSTPROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data}"

export USE_WANDB="${USE_WANDB:-1}"
export WANDB_GROUP="${WANDB_GROUP:-q35-27b-tb2-full89-overfit-ppo-lenreward-l0.4-noKL-nologprob-cp2-96k-to3600-criticwarmup1-memfrac0.6}"

# >>EVAL: periodic held-out eval on the same Terminal-Bench 2 (89 tasks) — KEPT.
# Every EVAL_INTERVAL=10 rollout steps. With NUM_ROLLOUT=8 (smoke) eval will NOT
# fire during the run; bump NUM_ROLLOUT>=10 to exercise eval. NO step-0 baseline.
# Use ${VAR-default} (not :-) so an explicit EVAL_INTERVAL="" from the caller survives
# and disables periodic eval (downstream emits --skip-eval-before-train).
export EVAL_INTERVAL="${EVAL_INTERVAL-10}"
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
export EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-tb2}"
export SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-1}"

export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtb2ppo}"
export RAY_CLUSTER_WAIT_SECONDS="${RAY_CLUSTER_WAIT_SECONDS:-1800}"

exec "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
