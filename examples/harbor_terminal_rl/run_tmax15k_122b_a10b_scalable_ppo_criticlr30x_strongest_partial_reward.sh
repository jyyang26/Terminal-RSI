#!/bin/bash
#
# ====================================================================================
# HOW TO RUN
# ====================================================================================
#   cd /path/to/workspace/slime-terminal-RL
#
#   # 0) one-time prerequisites
#   #    umask 077
#   #    printf 'export DAYTONA_API_KEY=...\nexport WANDB_API_KEY=...\n' > /tmp/bs512_keys.env
#   #    EVERY node needs fla 0.4.2 (CP>1 hard dep) plus the full harbor package set --
#   #    see envs/README_cluster_env_setup_zh.md, then verify with
#   #      bash envs/verify_node_env.sh          # must be IDENTICAL on all nodes
#
#   # 1) dry run first (prints the resolved plan, spends no GPU):
#   DRY_RUN=1 examples/harbor_terminal_rl/run_tmax15k_122b_a10b_scalable_ppo_criticlr30x_strongest_partial_reward.sh
#
#   # 2) launch (176 GPU: 128 train + 48 infer, CP4, bs512, 84k, dense/partial reward):
#   examples/harbor_terminal_rl/run_tmax15k_122b_a10b_scalable_ppo_criticlr30x_strongest_partial_reward.sh
#
#   # 2') on a 17-node / 136-GPU cluster (128 train + 8 infer, 1 engine -> slower rollout):
#   TOPOLOGY=136cp4 examples/harbor_terminal_rl/run_...strongest_partial_reward.sh
#
#   # 3) exactly 3 epochs over 15000 tasks @ bs512 (ceil(15000/512)=30 x 3 = 90):
#   NUM_ROLLOUT=90 examples/harbor_terminal_rl/run_...strongest_partial_reward.sh
#
#   # 4) reward-variant A/B (see "REWARD VARIANTS" below):
#   CUSTOM_REWARD_POST_PROCESS_PATH=slime_plugins.harbor.partial_reward.test_count_length_discount_rewards \
#     examples/harbor_terminal_rl/run_...strongest_partial_reward.sh
#
# ====================================================================================
# WHAT THIS IS
# ====================================================================================
# A wrapper over run_tmax15k_122b_a10b_scalable_ppo_criticlr30x_strongest.sh. It keeps
# that script's PPO recipe and memory/CP engineering verbatim, and overrides ONLY the
# axes this experiment changes. Current configuration (2026-08-17):
#
#   TOPOLOGY=176cp4   128 train GPU (TP2 x PP4 x CP4 => DP4, EP16), 12 layers/stage
#                    + 48 infer GPU (6 x TP8)   = 176 GPU / 22 nodes
#   batch 512 / rollout 512 / oversample 560 / n=1 train, pass@3 eval
#   84k walls on BOTH sides (SEQ_LENGTH = SGLANG_CONTEXT_LENGTH = 86016)
#   PPO eps-clip 0.2, KL OFF, entropy OFF, WARM critic, NUM_CRITIC_ONLY_STEPS=2
#   critic lr 30x (1.5e-5) / actor lr 2x (1.0e-6)  [megatron_roles_tmax15k_criticlr30x_actorlr2x.yaml]
#   MAX_TOKENS_PER_GPU 8192, max_split_size_mb:256, CP>1 anti-straggler trio
#   R3 routing replay ON, --no-save-optim ON
#
# WHAT DIFFERS FROM THE PARENT (everything else is inherited untouched):
#   (0)  TOPOLOGY 176cp4 (128+48, CP4, bs512) -- defined here, not in the parent's case block
#   (0b) CONTEXT both sides 84k, with a 64k/20k compaction split (see that section)
#   (1)  REWARD  binary terminal -> CTRF DENSE (partial/process) reward
#   (2)  DATA    tmax-15k -> strategy_deepseek_repaired_v4_final_15k (15000 tasks), SHUFFLED
#   (2b) MODELS  actor = RST_models (cold), critic = WARM from tmax15k 88cp4 iter_69
#   (3)  NUM_CRITIC_ONLY_STEPS=2 (warm critic needs only re-calibration)
#   (4)  USE_ROUTING_REPLAY=1 -> --use-rollout-routing-replay
#        NO_SAVE_OPTIM=1      -> --no-save-optim  (weights-only ckpts, NOT resumable)
#   plus run-identity isolation (BASE_FOLDER / SAVE_SUBDIR / WANDB / RAY_TEMP_DIR /
#   trials_dir) so nothing collides with previous runs.
#
#   Every one of these is a `${VAR:-default}` and can be overridden from the environment,
#   e.g. TOPOLOGY=136cp4 / USE_ROUTING_REPLAY=0 / NUM_CRITIC_ONLY_STEPS=5.
#
# ====================================================================================
# (1) THE REWARD -- CTRF dense / partial credit
# ====================================================================================
# test_count_progress_rewards reads each trial's verifier/ctrf.json and returns
#       shaped = passed_tests / HARBOR_TESTCOUNT_SCALE           (SCALE = 20)
# the ABSOLUTE passed-test count, NOT the passed/total ratio. Rationale: the ratio
# weights every task equally at max 1.0, flattening the signal for hard many-test
# tasks; the absolute count lets them earn proportionally more. A fully-passed task
# scores total/SCALE, which CAN EXCEED 1.0 for tasks with >20 tests -- intended, PPO's
# critic+GAE absorbs the scale (measured max on real trials: 2.0 for a 40-test task).
#
# Under --advantage-estimator ppo, partial_reward's _should_normalize() is False, so
# raw shaped rewards flow straight into critic + GAE with NO group normalization.
#
# WHY THIS DATA POOL MAKES IT WORK (verified, not assumed): the strategy-15k tasks'
# tests/test.sh runs `uvx -w pytest-json-ctrf pytest --ctrf /logs/verifier/ctrf.json`,
# so per-test pass/fail is emitted -- exactly what _load_progress_score parses. The
# pre-flight below re-checks up to 20 sampled tasks and refuses to launch unless a
# MAJORITY of them emit it.
# On the previous (RTS) pool the same reward produced 93-96% ctrf hit rate with ~80% of
# trials strictly-partial (0<passed<total) and rewards up to 2.75 -- a real dense signal.
#
# >>CONTRAST: the tmax-15k pool's test.sh writes only verifier/reward.json (binary +
# test__<name> fields) and NO ctrf.json. Point this wrapper at tmax-15k and every
# trajectory falls back to terminal_reward -- the dense reward silently becomes a
# no-op. That is why the pre-flight verifies CTRF on whatever pool is configured.
#
# REWARD VARIANTS (all four ported from v3; override CUSTOM_REWARD_POST_PROCESS_PATH):
#   test_count_progress_rewards        <- DEFAULT. passed/SCALE. pure dense credit.
#   test_count_efficiency_rewards      run#5. + step-efficiency bonus on solved trajs.
#                                      env: HARBOR_EFFICIENCY_BONUS_BETA (0.05)
#   test_count_length_penalty_rewards  run#6. - lambda*clamp(out_tokens/BUDGET).
#                                      v3 RUNBOOK records this FAILED (avg_turns 22->50+).
#                                      env: HARBOR_LENGTH_PENALTY_LAMBDA / _TOKEN_BUDGET
#   test_count_length_discount_rewards run#7. MULTIPLICATIVE length discount + zero-pass
#                                      penalty; v3's fix for run#6. env:
#                                      HARBOR_LENGTH_DISCOUNT_{FREE_TURNS,FULL_TURNS,ALPHA,ZERO_BETA}
#   trajectory_grouped_progress_rewards  the OLD ratio reward: max(terminal, 0.4*passed/total).
#                                      env: HARBOR_PARTIAL_REWARD_MAX (0.4)
#
# NOT ENABLED: --normalize-advantages (advantage whitening). v3's RUNBOOK §14 claims the
# 38k run used it, but NORMALIZE_ADVANTAGES appears NOWHERE in v3's starter script
# (grep count 0) -- it was never wired, so that run did NOT actually whiten. Keeping it
# off preserves "algorithm identical to the parent". To A/B it on, you MUST repeat the
# parent's own value or you will clobber the DeepEP-off dispatcher setting:
#   EXTRA_MISC_ARGS="--moe-token-dispatcher-type alltoall --normalize-advantages"
#
# ====================================================================================
# (2) THE DATA -- strategy_deepseek_repaired_v4_final_15k, 15000 tasks, SHUFFLED
# ====================================================================================
# Pool:
#   .../slime-terminal-RL_data/strategy_deepseek_repaired_v4_final_15k/materialized_top15k/tasks
# 15000 REAL task directories (not symlinks), each with task.toml / instruction.md /
# tests/ / solution/ / environment/. Selected from the R1-R15 pool by DeepSeek-V4-Pro
# semantic auditing (see the pool's README.md): weighted quality_score_5 over 8
# dimensions, with hard_reject for hidden requirements, test leakage, solution
# shortcuts, or verifiers too weak to validate the task goal. So this is a
# quality-filtered 15k, not a raw dump.
#
# >>CTRF CONFIRMED: the tasks' tests/test.sh runs
#     uvx -w pytest-json-ctrf pytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py
#   so per-test pass/fail is emitted -- which is exactly what the dense reward parses.
#   (Verified directly on a sampled task before launch; the pre-flight re-checks 20.)
#
# >>SHUFFLE (rollout_shuffle: true in the yaml): the pool is materialised in rank order
#   (quality15k_v4_00001..), so sorted(glob) would sweep best-to-worst monotonically and
#   each batch would come from one narrow quality band. Shuffle reorders per epoch
#   (data_source.py::_ensure_epoch). REQUIRED here.
#
# SCHEDULE: 3 epochs over 15000 @ bs512 = ceil(15000/512)=30 steps/epoch x 3 = 90 steps.
# NUM_ROLLOUT is deliberately NOT set here -- it inherits the parent's 100. Pass
# NUM_ROLLOUT=90 for exactly 3 epochs (or leave 100 for ~3.4 epochs).
#
# ====================================================================================
# TRIAL DIR: fresh per run, never reuse the 27B dirs
# ====================================================================================
# Trial dirs are an OUTPUT (training writes verifier/ctrf.json, the reward reads it
# back) -- nothing needs pre-populating. But _trial_name() (harbor/rollout.py) is
# "slime-r{rollout}-g{group}-s{rank}-i{index}" with NO run tag, so pointing trials_dir
# at the old 27B run would collide name-for-name. In that old dir only 47/400 sampled
# trials have a ctrf.json, so any new trial that fails to emit its own would silently
# read the 27B MODEL's leftover ctrf and be scored on another model's results.
# => this script materializes the config with a ${RUN_TAG}-scoped trials_dir and
#    refuses known-stale paths.
# ====================================================================================

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# ====================================================================================
# (0) TOPOLOGY: 176 GPU / 22 nodes -- 128 train + 48 infer, CP4, batch 512, 84k walls
# ====================================================================================
# The parent's case-block has no 176-GPU preset, but it honours EXPLICIT topology vars
# (`export EXPECTED_NODES / ACTOR_NUM_NODES / ROLLOUT_NUM_GPUS / PIPELINE_...`) and only
# falls into the case block for its own named presets. We therefore define this topology
# HERE and let TOPOLOGY act purely as a label -- no edit to the shared parent script.
#
#   128 train GPU = 16 nodes: TP2 x PP4 x CP4 => DP4, EP16, 12 layers/stage
#    48 infer GPU =  6 nodes: 6 x TP8
#   ---------------------------------------------------------------
#   176 GPU / 22 nodes  (the cluster has 136 GPU / 17 nodes -> see the NOTE below)
#
# WHY THESE NUMBERS (all three constraints are hard):
#   * CP MUST be 4 at an 84k wall. The binding term is the fp32 logits on the LAST
#     pipeline stage (vocab 248320): local T = 86016/CP.
#       CP2 -> local 43008 -> logits 29.8 GB -> peak 89.5 GB (+6.1 GB, too tight)
#       CP4 -> local 21504 -> logits 14.9 GB -> peak 64.4 GB (+31.2 GB, safe)
#   * GLOBAL_BATCH_SIZE % DP == 0. DP = 128/(2*4*4) = 4, and 512 % 4 == 0. OK.
#     (This is why 96 or 160 train GPU are ILLEGAL here: they give DP 6 / 10, which
#      break the EP constraint below.)
#   * EP*ETP must divide TP*CP*DP: 16 | 2*4*4 = 32. OK.
#   * 48 layers % PP == 0 and (48/PP) % full_attention_interval(4) == 0 -> PP=4 gives
#     12 layers/stage, so every stage gets the same [lin,lin,lin,full] mix.
#
# COST/BENEFIT vs the previous 88-GPU bs256 run (measured: 33 min/step):
#   bs512 doubles the rollout work, but 48 infer GPU (6 engines vs 3) halves it back;
#   DP4 keeps 512/4 = 128 micro-batches per rank. Projected ~42 min/step => 441 steps
#   in ~13 days.
#
# NOTE ON 136 vs 176: the extended cluster is 17 nodes / 136 GPU. This preset asks for
# 176. Set TOPOLOGY=136cp4 (below) to fit 128 train + 8 infer on the real cluster, or
# raise the allocation to 22 nodes for the full 48-GPU inference side. The pre-flight
# refuses to launch if the live cluster is smaller than the requested topology, rather
# than letting Ray hang waiting for GPUs that will never arrive.
export TOPOLOGY="${TOPOLOGY:-128cp4}"
case "${TOPOLOGY}" in
  # ★ DEFAULT. 128 GPU TOTAL (train + infer), 16 nodes: 64 train + 64 infer (8 x TP8).
  #   64/(2*4*4) = DP2. The ONLY legal split of a 128-GPU budget at bs512 + 84k:
  #     * 512 % DP == 0            -> DP in {1,2,4,8,...}
  #     * EP*ETP (16) | TP*CP*DP   -> kills DP6 (96 train) and DP10 (160 train)
  #     * train_gpus % (TP*PP*CP) == 0
  #     * (48/PP) % full_attention_interval(4) == 0 -> PP=4
  #   96/112/120-train variants are all ILLEGAL under those constraints, so the split is
  #   forced to 64+64. CP4 (not CP2) because at an 84k wall CP2 peaks at 89.5 GB (+6.1 GB
  #   only), while CP4 peaks at 78.6 GB (+17.0 GB).
  #   COST: DP2 at bs512 = 256 grad-accum micro-batches per rank, so the train side is
  #   ~41 min/step; the 8 inference engines cut rollout to ~16 min. ~57 min/step total,
  #   i.e. ~3.5 days for 90 steps.
  128cp4)   _EXP_NODES=16; _ACT_NODES=8;  _ROLL_GPUS=64; _PP=4; _CP=4 ;;  # 64 train, DP2, 8 engines
  # The presets below use 128 TRAIN GPU (=DP4, faster training) but need a bigger cluster.
  176cp4)   _EXP_NODES=22; _ACT_NODES=16; _ROLL_GPUS=48; _PP=4; _CP=4 ;;  # 128 train, DP4, 6 engines
  152cp4)   _EXP_NODES=19; _ACT_NODES=16; _ROLL_GPUS=24; _PP=4; _CP=4 ;;  # 128 train, DP4, 3 engines
  136cp4)   _EXP_NODES=17; _ACT_NODES=16; _ROLL_GPUS=8;  _PP=4; _CP=4 ;;  # 128 train, DP4, 1 engine (rollout-bound)
  *)        _EXP_NODES=""; _ACT_NODES=""; _ROLL_GPUS=""; _PP=""; _CP="" ;;  # defer to parent preset
esac
if [ -n "${_ACT_NODES}" ]; then
  export EXPECTED_NODES="${EXPECTED_NODES:-${_EXP_NODES}}"
  export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${_ACT_NODES}}"
  export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-${_ROLL_GPUS}}"
  export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-${_PP}}"
  export CP_SIZE="${CP_SIZE:-${_CP}}"
  # bs512 as requested. OVER_SAMPLING 560 keeps the same ~1.09x skim ratio as bs256/280.
  export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
  export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-512}"
  export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-560}"

  # ★ The parent ALWAYS evaluates its own `case "${TOPOLOGY}"` and HARD-EXITS on a name it
  # does not know -- exporting the topology vars does NOT skip that block. So keep our
  # label for logging / run identity, and hand the parent a preset name it accepts.
  # This is safe because every value its preset would set is already exported above and
  # the parent assigns with `${VAR:-default}` / `: "${VAR:=default}"`, so its preset cannot
  # override us. 88cp4-84k is the closest stand-in (PP4 / CP4 / 84k walls); note it also
  # does `: "${GLOBAL_BATCH_SIZE:=256}"`, which is exactly why 512 is exported first.
  export TOPOLOGY_LABEL="${TOPOLOGY}"
  export TOPOLOGY="88cp4-84k"
fi

# ====================================================================================
# (0b) CONTEXT: train side and rollout side BOTH 84k, with a 20k compaction buffer
# ====================================================================================
# Previously the two sides DISAGREED: training was walled at 64k while SGLang served 84k,
# and the agent believed it had 128k. Now every wall is 86016 (84k):
#
#   SEQ_LENGTH / HARBOR_MAX_TRAIN_TOTAL_TOKENS  86016   training-side token budget
#   ROLLOUT_MAX_CONTEXT_LEN / SGLANG_CONTEXT_LENGTH / SGLANG_MAX_PREFILL_TOKENS  86016
#   agent max_context_tokens (in the yaml)      86016   <- was 128000, ABOVE the server wall
#
# The 64k / 20k split you asked for is implemented by the agent's compaction trigger
# (`proactive_summarization_threshold: 65536` in the yaml): the agent keeps rolling until
# its context reaches 64k -- that is the part that is rolled out AND trained on -- and the
# remaining 86016-65536 = 20480 tokens are headroom for the summarisation/compaction pass
# to run without ever crossing the 84k server wall.
export SEQ_LENGTH="${SEQ_LENGTH:-86016}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-86016}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-86016}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-86016}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-86016}"


# --- (1) REWARD: CTRF dense / partial credit ----------------------------------------
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.partial_reward.test_count_progress_rewards}"
export HARBOR_TESTCOUNT_SCALE="${HARBOR_TESTCOUNT_SCALE:-20}"
# Kept explicit so an A/B revert to the ratio reward needs no second edit.
export HARBOR_PARTIAL_REWARD_MAX="${HARBOR_PARTIAL_REWARD_MAX:-0.4}"
# The separate length-SHAPED reward plugin stays OFF (this run's length handling, if
# any, comes from the reward variant above, not from length_shaped_reward.py).
export HARBOR_LENGTH_REWARD_LAMBDA="${HARBOR_LENGTH_REWARD_LAMBDA:-0.0}"

# --- (3) CRITIC WARM-UP: 2 critic-only steps before the actor joins -------------------
# rollout_id < NUM_CRITIC_ONLY_STEPS trains the critic ONLY (train_async.py:75,85).
# 2 is enough here because the critic is WARM-LOADED (see the load-root section below):
# it already carries a trained value head from the tmax15k 88cp4-84k run's iter_69, so it
# only needs to re-calibrate to the new reward scale / data distribution, not learn from
# scratch. (The previous run used 5 because its critic was cold-started.)
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-2}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

# --- (4) ROUTING REPLAY (R3) + no-save-optim, assembled into EXTRA_MISC_ARGS ---------
# NEITHER flag has an env knob in the base launcher (verified: grep is empty in
# run_qwen35_9b_terminal_bench2_64h20.sh), so both can ONLY arrive via EXTRA_MISC_ARGS.
# That variable is OVERWRITTEN, not appended, by the parent's
# `export EXTRA_MISC_ARGS="${EXTRA_MISC_ARGS:---moe-token-dispatcher-type alltoall}"` --
# so hand-passing it has historically DROPPED the DeepEP-off dispatcher setting, or
# silently lost routing replay entirely. We therefore ASSEMBLE it here from knobs
# instead of asking the caller to remember the full string.
#
# --use-rollout-routing-replay (R3, arxiv 2510.11370): replays the exact MoE routing
#   captured during rollout when the actor recomputes log-probs, removing the
#   rollout/train expert-selection mismatch. arguments.py:1924 also sets
#   use_routing_replay=True as a side effect. NOTE this makes the initial actor/ref
#   KL NON-ZERO by design (model.py:805-813 disables the KL==0 assert for R3), because
#   the actor replays rollout routing while the reference forward routes normally.
#   Do NOT read a non-zero kl_loss as a bug.
# --no-save-optim: checkpoints carry weights only. Smaller/faster saves, and harmless
#   for the PPO critic (placement_group.py:196-224 already warm-loads the critic
#   weights-only by default). COST, per the arg's own help text: it "disables training
#   resumption from the saved checkpoint" -- these ckpts are for eval / fresh cold-start,
#   NOT for mid-run resume. A crash means restarting, not resuming.
export USE_ROUTING_REPLAY="${USE_ROUTING_REPLAY:-1}"
export NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-1}"
_MISC="${EXTRA_MISC_ARGS:---moe-token-dispatcher-type alltoall}"
case " ${_MISC} " in *" --moe-token-dispatcher-type "*) ;; *) _MISC="--moe-token-dispatcher-type alltoall ${_MISC}" ;; esac
if [ "${USE_ROUTING_REPLAY}" = "1" ]; then
  case " ${_MISC} " in *" --use-rollout-routing-replay "*) ;; *) _MISC="${_MISC} --use-rollout-routing-replay" ;; esac
fi
if [ "${NO_SAVE_OPTIM}" = "1" ]; then
  case " ${_MISC} " in *" --no-save-optim "*) ;; *) _MISC="${_MISC} --no-save-optim" ;; esac
fi
export EXTRA_MISC_ARGS="${_MISC}"

# --- (2b) MODELS: RST actor (cold) + WARM critic from tmax15k 88cp4 iter_69 -----------
# Wired WITHOUT touching slime core, via a LOAD-ROOT of symlinks. slime's
# _isolate_critic_checkpoint_dirs (placement_group.py:151-224) redirects the critic's
# --load to "<load>_critic" IF that sibling exists on disk, and loads it WEIGHTS-ONLY
# (no_load_optim / no_load_rng / finetune). So:
#
#   <LOAD_ROOT>/Qwen3.5-122B-A10B_base        -> RST torch_dist  (actor, marker "release")
#   <LOAD_ROOT>/Qwen3.5-122B-A10B_base_critic -> tmax15k 88cp4 iter_69 critic (marker 69)
#
# WHY start_rollout_id stays 0 even though the critic says iter 69: arguments.py sets
# start_rollout_id=0 first because the ACTOR's marker is "release", and placement_group
# only fills it in when it is still None. The critic's weights-only load means the
# 69-vs-441 schedule mismatch cannot assert-crash
# (OptimizerParamScheduler is skipped entirely).
export STRATEGY15K_LOAD_ROOT="${STRATEGY15K_LOAD_ROOT:-/path/to/workspace/.slime_load_roots/strategy15k_ppo_partialreward_warmcritic69}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-${STRATEGY15K_LOAD_ROOT}/Qwen3.5-122B-A10B_base}"
export HF_CKPT="${HF_CKPT:-/path/to/shared_storage/hf_models/Qwen/RST_models/Qwen3.5-122B-A10B}"
# Load from the SHARED fs (not /tmp) so the "<load>_critic" sibling resolves on every node.
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-0}"

# --- (5) NCCL diagnostics + longer group timeouts -----------------------------------
# The 2026-08-19 crash died with `Cuda failure 999 'unknown error'` inside a
# sequence-parallel reduce_scatter, ~9 minutes after an offload -> "Reloading 166 process
# groups" -> wake_up cycle. GPU and host memory were both fine at the time (67 GB free
# GPU, 1267 GB free host) and every GPU still reports 0 ECC errors afterwards, so it was
# a transient fault in that destroy/rebuild path rather than a config error or bad card.
#
# PPO forces offload_train=True (actor+critic share GPUs), so EVERY step tears down and
# rebuilds ~166 NCCL groups -- 90 of those cycles per run is a lot of exposure. Raising
# both timeouts gives a slow rebuild room to finish instead of being declared dead.
export SLIME_NCCL_GROUP_TIMEOUT_MINUTES="${SLIME_NCCL_GROUP_TIMEOUT_MINUTES:-240}"
export SLIME_NEW_GROUP_TIMEOUT_MINUTES="${SLIME_NEW_GROUP_TIMEOUT_MINUTES:-240}"
# NCCL_DEBUG=WARN prints the driver/transport reason behind a failure (nearly free at
# WARN; INFO would flood the log). NOTE this only reaches the workers because
# "NCCL_DEBUG" was added to the launcher's RUNTIME_ENV_JSON env_vars block -- exporting
# it here alone would be silently dropped, like NORMALIZE_ADVANTAGES was.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# --- (2) DATA + run identity ---------------------------------------------------------
export TASKS_DIR="${TASKS_DIR:-/path/to/workspace/slime-terminal-RL_data/strategy_deepseek_repaired_v4_final_15k/materialized_top15k/tasks}"
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_q35_122b_a10b_strategy15k_partialreward_${TOPOLOGY_LABEL:-$TOPOLOGY}_${RUN_TAG}}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-122B-A10B_strategy15k_${TOPOLOGY_LABEL:-$TOPOLOGY}gpu_ppo_criticlr30x_actorlr2x_partialreward}"
export WANDB_GROUP="${WANDB_GROUP:-q35-122b-a10b-strategy15k-${TOPOLOGY_LABEL:-$TOPOLOGY}g-ppo-partialReward-84k-bs512-warmcritic69-shuffle}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rstrategy15k122b}"

# trials_dir is run-scoped by materializing the config template (the established
# pattern in this repo, cf. start_tb2_full89_96gpu_27b_daytona_ppo_tmux.sh).
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/harbor_strategy15k_122b_a10b_daytona_ppo_cp4_bs512_84k_partial_reward.yaml}"
RUN_TRIALS_DIR="${RUN_TRIALS_DIR:-/path/to/workspace/slime_harbor_trials/strategy15k_122b_a10b_ppo_partial_reward_${RUN_TAG}}"

# ====================================================================================
# PRE-FLIGHT (fail in seconds, not 15-20 min into init)
# ====================================================================================
_die() { echo "ERROR: $*" >&2; exit 1; }

# -- reward hook must import, and must be one of the known variants -----------------
[ -f "${CONFIG_TEMPLATE}" ] || _die "config template not found: ${CONFIG_TEMPLATE}"
_RM_MOD="${CUSTOM_REWARD_POST_PROCESS_PATH%.*}"
_RM_FN="${CUSTOM_REWARD_POST_PROCESS_PATH##*.}"
PYTHONPATH="${PYTHONPATH:-}:${SLIME_DIR:-/path/to/workspace/slime-terminal-RL}" \
python3 - "${_RM_MOD}" "${_RM_FN}" <<'PY' || _die "cannot import ${CUSTOM_REWARD_POST_PROCESS_PATH}"
import importlib, sys, types
# partial_reward only needs the Sample dataclass; stub torch so this check needs no GPU stack
if "torch" not in sys.modules:
    st = types.ModuleType("torch"); st.dtype = type("dtype", (), {})
    st.Size = tuple; st.Tensor = type("Tensor", (), {}); sys.modules["torch"] = st
mod, fn = sys.argv[1], sys.argv[2]
m = importlib.import_module(mod)
assert callable(getattr(m, fn)), f"{mod}.{fn} is not callable"
PY

# -- data pool must be UNPACKED and complete ----------------------------------------
[ -d "${TASKS_DIR}" ] || _die "task pool missing: ${TASKS_DIR}
       Unpack it first (see the DATASET section at the top of this file)."
_N_TASKS=$(python3 -c "from pathlib import Path;print(len(list(Path('${TASKS_DIR}').glob('*/task.toml'))))")
if [ "${_N_TASKS}" -eq 0 ]; then
  _die "0 resolvable */task.toml under ${TASKS_DIR}.
       Either the tars were never extracted, or you pointed at the SYMLINK farm
       (harbor_synth_all_20260725/dataset), whose links target the unmounted
       /path/to/shared_storage and /path/to/shared_storage. Extract the tars instead."
fi
_MIN_TASKS="${MIN_TASKS:-14900}"
if [ "${_N_TASKS}" -lt "${_MIN_TASKS}" ]; then
  _die "only ${_N_TASKS} tasks resolve under ${TASKS_DIR} (expected ~15000).
       Extraction looks incomplete. Re-run it, or lower MIN_TASKS to proceed anyway."
fi

# -- the config must point at that same pool (catch a stale/edited yaml) -------------
_YAML_DIR=$(grep -oP '^harbor_tasks_dir:\s*\K\S+' "${CONFIG_TEMPLATE}" || true)
[ "${_YAML_DIR}" = "${TASKS_DIR}" ] || _die "config harbor_tasks_dir (${_YAML_DIR}) != TASKS_DIR (${TASKS_DIR})."

# -- CTRF sanity: dense reward is a NO-OP on a pool that emits no ctrf --------------
# Sample 20 tasks; require most of them to run pytest --ctrf.
_CTRF_OK=$(python3 - "${TASKS_DIR}" <<'PY'
import itertools, pathlib, sys
root = pathlib.Path(sys.argv[1]); hit = n = 0
for d in itertools.islice((p for p in root.iterdir() if p.is_dir()), 20):
    sh = d / "tests" / "test.sh"
    if not sh.is_file():
        continue
    n += 1
    hit += "ctrf" in sh.read_text(errors="ignore")
print(f"{hit}/{n}")
PY
)
# Judge the RATE, not an absolute count: the sample loop skips dirs without tests/test.sh
# (still-copying pools, metadata dirs), so a healthy pool can legitimately yield fewer
# than 20 inspectable tasks. Requiring "hit >= 10" once rejected a pool at 6/6 = 100%.
_CTRF_HIT="${_CTRF_OK%%/*}"
_CTRF_N="${_CTRF_OK##*/}"
if [ "${_CTRF_N:-0}" -lt 3 ]; then
  _die "could not inspect enough tasks under ${TASKS_DIR} (only ${_CTRF_OK} had tests/test.sh).
       If the pool is still being copied, wait for it to finish."
fi
if [ $(( _CTRF_HIT * 2 )) -lt "${_CTRF_N}" ]; then
  _die "only ${_CTRF_OK} sampled tasks emit CTRF (pytest --ctrf).
       ${CUSTOM_REWARD_POST_PROCESS_PATH} reads verifier/ctrf.json; without it EVERY
       trajectory falls back to the binary terminal reward and the dense reward is a
       silent NO-OP. Check the task pool before spending GPUs."
fi

# -- the live Ray cluster must actually be big enough for the requested topology ------
# Without this, Ray simply BLOCKS forever waiting for GPUs that will never arrive (the
# placement group is unschedulable), which looks like a hang rather than a config error.
_NEED_GPUS=$(( ACTOR_NUM_NODES * ${ACTOR_NUM_GPUS_PER_NODE:-8} + ROLLOUT_NUM_GPUS ))
_LIVE_GPUS=$(timeout 120 ray status 2>/dev/null | grep -oE '[0-9.]+/[0-9.]+ GPU' | head -1 | sed 's#.*/##; s# GPU##; s#\..*##')
if [ -n "${_LIVE_GPUS}" ]; then
  if [ "${_LIVE_GPUS}" -lt "${_NEED_GPUS}" ]; then
    _die "TOPOLOGY=${TOPOLOGY_LABEL:-$TOPOLOGY} needs ${_NEED_GPUS} GPU (${ACTOR_NUM_NODES}x${ACTOR_NUM_GPUS_PER_NODE:-8} train + ${ROLLOUT_NUM_GPUS} infer)
       but the live Ray cluster only has ${_LIVE_GPUS}.
       Either extend the cluster (bash stable_rl/scripts/start_ray_cluster.sh after the
       new nodes are allocated, then install deps per envs/README_cluster_env_setup_zh.md
       and check with envs/verify_node_env.sh), or pick a topology that fits:
         TOPOLOGY=136cp4  -> 128 train +  8 infer = 136 GPU (17 nodes)
         TOPOLOGY=152cp4  -> 128 train + 24 infer = 152 GPU (19 nodes)
         TOPOLOGY=176cp4  -> 128 train + 48 infer = 176 GPU (22 nodes)
         TOPOLOGY=88cp2   -> the previous 64+24 bs256 recipe (needs GLOBAL_BATCH_SIZE=256)"
  fi
  echo "[preflight] cluster  = ${_LIVE_GPUS} GPU live, need ${_NEED_GPUS} -> OK"
else
  echo "[preflight] WARN: could not read 'ray status'; skipping the cluster-size check."
fi

# -- refuse a stale/shared trials_dir (silent cross-model reward contamination) ------
case "${RUN_TRIALS_DIR}" in
  *synth_all_38k_ppo_dense_shuffle*|*Recursive-Task-Synthesis-Trajectories*)
    _die "RUN_TRIALS_DIR points into the previous 27B run's trial tree:
       ${RUN_TRIALS_DIR}
       Trial names are deterministic (slime-r*-g*-s*-i*), so this run would collide
       with the 27B dirs and could read ANOTHER MODEL's leftover ctrf.json as its own
       reward. Use a fresh run-scoped directory." ;;
esac
if [ -d "${RUN_TRIALS_DIR}" ] && [ -n "$(ls -A "${RUN_TRIALS_DIR}" 2>/dev/null)" ]; then
  if [ "${ALLOW_DIRTY_TRIALS_DIR:-0}" != "1" ]; then
    _die "RUN_TRIALS_DIR already exists and is NOT empty: ${RUN_TRIALS_DIR}
       Pre-existing slime-r*/verifier/ctrf.json could be read as this run's reward.
       Use a fresh dir, or set ALLOW_DIRTY_TRIALS_DIR=1 if you are resuming on purpose."
  fi
  echo "[preflight] WARN: reusing non-empty trials_dir (ALLOW_DIRTY_TRIALS_DIR=1)."
fi

# ====================================================================================
# MATERIALIZE THE CONFIG (only the trials_dir line is rewritten)
# ====================================================================================
mkdir -p "${BASE_FOLDER}" "${RUN_TRIALS_DIR}"
CONFIG="${BASE_FOLDER}/harbor_config_resolved.yaml"
sed "s#^\([[:space:]]*trials_dir:\).*#\1 ${RUN_TRIALS_DIR}#" "${CONFIG_TEMPLATE}" > "${CONFIG}"
grep -q "trials_dir: ${RUN_TRIALS_DIR}" "${CONFIG}" || _die "failed to rewrite trials_dir into ${CONFIG}"
export CONFIG   # the parent uses `if [ -z "${CONFIG:-}" ]`, so this export wins

echo "[preflight] reward   = ${CUSTOM_REWARD_POST_PROCESS_PATH} (SCALE=${HARBOR_TESTCOUNT_SCALE})"
echo "[preflight] data     = ${TASKS_DIR}  (${_N_TASKS} tasks, ctrf ${_CTRF_OK} sampled)"
echo "[preflight] shuffle  = $(grep -oP '^rollout_shuffle:[[:space:]]*\K(true|false)' "${CONFIG}" || echo '(unset)')"
echo "[preflight] context  = train ${SEQ_LENGTH} / rollout ${ROLLOUT_MAX_CONTEXT_LEN} / sglang ${SGLANG_CONTEXT_LENGTH}"
echo "[preflight]            agent ceiling $(grep -oP 'max_context_tokens:\s*\K\S+' "${CONFIG}" || echo '?')," \
     "compaction at $(grep -oP 'proactive_summarization_threshold:\s*\K\S+' "${CONFIG}" || echo '?')" \
     "(=> $(( $(grep -oP 'max_context_tokens:\s*\K[0-9]+' "${CONFIG}" || echo 0) - $(grep -oP 'proactive_summarization_threshold:\s*\K[0-9]+' "${CONFIG}" || echo 0) )) token compact buffer)"
echo "[preflight] actor    = ${PRETRAINED_CKPT}"
echo "[preflight] critic   = $(if [ -d "${PRETRAINED_CKPT}_critic" ]; then echo "WARM <- ${PRETRAINED_CKPT}_critic (iter $(cat "${PRETRAINED_CKPT}_critic/latest_checkpointed_iteration.txt" 2>/dev/null), weights-only)"; else echo "COLD (no _critic sibling; value head reinitialised)"; fi)"
echo "[preflight] trials   = ${RUN_TRIALS_DIR}"
echo "[preflight] config   = ${CONFIG}"
echo "[preflight] batch    = ${GLOBAL_BATCH_SIZE} (rollout ${ROLLOUT_BATCH_SIZE}, oversample ${OVER_SAMPLING_BATCH_SIZE})"
echo "[preflight] critic-only steps = ${NUM_CRITIC_ONLY_STEPS} (actor joins at rollout_id ${NUM_CRITIC_ONLY_STEPS}); save every ${SAVE_INTERVAL}"
echo "[preflight] misc     = ${EXTRA_MISC_ARGS}"
echo "[preflight] nccl     = NCCL_DEBUG=${NCCL_DEBUG:-<unset>}  group_timeout=${SLIME_NCCL_GROUP_TIMEOUT_MINUTES:-?}min  new_group_timeout=${SLIME_NEW_GROUP_TIMEOUT_MINUTES:-?}min"
echo "[preflight] data cut = harbor_tasks_offset=$(grep -oP '^harbor_tasks_offset:\s*\K[0-9]+' "${CONFIG}" || echo 0) (skips already-trained tasks)"
if [ "${USE_ROUTING_REPLAY}" = "1" ]; then
  # R3 turns a missing/unsafe routing row into a HARD failure, and this config sets
  # harbor_fail_on_conversion_error: true -- so the exception is re-raised
  # (harbor/rollout.py:1204-1206) and takes the whole rollout down rather than dropping
  # one sample. Surface that up front so a crash here is recognised instantly.
  _FOCE=$(grep -oP '^harbor_fail_on_conversion_error:\s*\K\S+' "${CONFIG}" || echo 'true')
  echo "[preflight] R3 ON: routing replay requires routed_experts on EVERY turn."
  echo "[preflight]   harbor_fail_on_conversion_error=${_FOCE} -> a missing routing row will"
  if [ "${_FOCE}" = "true" ]; then
    echo "[preflight]   ABORT THE ROLLOUT (not skip the sample). Watch for 'Missing routed_experts'."
  else
    echo "[preflight]   DROP the sample and continue (introduces selection bias -- watch counts)."
  fi
  echo "[preflight]   Also expected: train/kl_loss is NON-ZERO under R3 (by design, not a bug)."
fi
echo "[preflight] 3 epochs @ bs${GLOBAL_BATCH_SIZE} would be NUM_ROLLOUT=$(( ( (_N_TASKS + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE ) * 3 )); currently NUM_ROLLOUT=${NUM_ROLLOUT:-100 (inherited)}"
echo "[preflight] all checks passed -> handing off to the parent strongest recipe (TOPOLOGY=${TOPOLOGY_LABEL:-$TOPOLOGY} -> parent preset ${TOPOLOGY})"
echo

# Everything else -- topology, PPO recipe, per-role lr, memory/CP knobs, credentials,
# cluster reuse, DRY_RUN/BACKGROUND handling -- is inherited UNCHANGED.
exec "${SCRIPT_DIR}/run_tmax15k_122b_a10b_scalable_ppo_criticlr30x_strongest.sh"
