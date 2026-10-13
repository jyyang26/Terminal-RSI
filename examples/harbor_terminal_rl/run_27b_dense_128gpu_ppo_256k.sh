#!/usr/bin/env bash
# run_27b_dense_128gpu_ppo_256k.sh
#
# 27B DENSE Qwen3.5/3.8-27B, Terminal PPO, 256k context, 128 GPU on the sh9 cluster.
#
# Structure follows run_sh9_122b_a10b_128gpu.sh: this wrapper only exports env and
# then execs the shared parent recipe. No shared script is edited.
#
#   ./run_27b_dense_128gpu_ppo_256k.sh              # full run
#   SMOKE=1 ./run_27b_dense_128gpu_ppo_256k.sh      # 1-step 256k smoke test
#
# ============================== WHY THIS EXISTS ==============================
# The base template targets the 122B MoE. Porting to 27B DENSE at 256k changes
# four things that are HARD constraints, not preferences:
#
#  1. TP=4 CEILING. num_query_groups=4, and Megatron needs num_query_groups % TP
#     == 0. TP=8 is ILLEGAL (4/8 = 0.5) for BOTH Megatron and the SGLang engines.
#  2. NO EP/MoE. common.pt reports num_experts=None. Every --moe-* / EP arg is
#     omitted rather than set to 1, and the template's EP divisibility rule is void.
#  3. CP=4 REQUIRED + fla-core. CP>1 on this GDN-hybrid model needs fla.ops.cp,
#     which ships in `fla-core` (NOT `flash-linear-attention`, despite the error
#     text). Verified fla-core==0.4.2 on all 16 nodes.
#  4. 256k VOCAB-CE PEAK. ~45.4 GB on the last PP stage at CP=4 (calibrated from
#     the template's own 84k measurements: 2.996 bytes/token/vocab). This dwarfs
#     the 27B weights (~6.3 GB/GPU) and is the real OOM risk.
#
# ============================ KEEPALIVE LIFECYCLE ============================
# main_dynamics.py must be RUNNING ON EVERY NODE BEFORE AND DURING the launch.
# The platform reclaims the cluster when GPU utilisation stays low, and startup
# (SGLang boot + 351 GB checkpoint load) is a 5-10 minute window at ~0% util --
# precisely when an unprotected cluster gets killed.
#
# Leaving it up during training is SAFE because the keepalive is self-regulating:
# main_dynamics.py:48-58 samples utilisation and only burns cycles on GPUs whose
# average util is BELOW --min_util (default 80). Once real training saturates the
# GPUs it does nothing, so it costs throughput only while we are idle anyway.
#
# Therefore this wrapper ENSURES the keepalive is up before handing off (it does
# NOT stop it), and the EXIT trap re-asserts it on every exit path -- crash,
# Ctrl-C, or clean completion -- so the cluster is never left unprotected.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_ROOT=/path/to/workspace
KEEPALIVE_CTL="${USER_ROOT}/keepalive_ctl.sh"

# ---- EGRESS PROXY: required for Daytona ---------------------------------------------
# These nodes have NO direct outbound route. Without this, Daytona sandbox file
# staging (harbor daytona.py:_sdk_upload_file -> fs.upload_file) times out:
# measured 732s DaytonaTimeoutError without the proxy env vs 1.1s with it.
# The SDK's REST layer additionally needs Configuration.proxy set explicitly
# (urllib3 ignores proxy env vars) -- that is handled by daytona_api_proxy.py,
# injected into the Ray workers' PYTHONPATH/env below.
# star_proxy_env.sh also sets a no_proxy list covering .oa.com, tencent mirrors
# AND the cluster's own 29.68./29.63. ranges, so Ray/NCCL/SGLang traffic never
# traverses the proxy.
if [ -f "${USER_ROOT}/star_proxy_env.sh" ]; then
  # shellcheck disable=SC1091
  source "${USER_ROOT}/star_proxy_env.sh"
  echo "[27b256k] egress proxy: ${https_proxy}"
else
  echo "[27b256k] WARN: star_proxy_env.sh missing -- Daytona uploads will time out" >&2
fi

_restore_keepalive() {
  rc=$?
  echo
  echo "[27b256k] training exited (rc=${rc}); re-asserting GPU keepalive on all nodes"
  if [ -x "${KEEPALIVE_CTL}" ]; then
    "${KEEPALIVE_CTL}" start || echo "[27b256k] WARN: keepalive restore failed -- run '${KEEPALIVE_CTL} start' manually"
  else
    echo "[27b256k] WARN: ${KEEPALIVE_CTL} not executable; restore manually"
  fi
  exit "${rc}"
}
trap _restore_keepalive EXIT INT TERM

# ---- SLIME / MODEL ------------------------------------------------------------------
export SLIME_DIR="${SLIME_DIR:-${USER_ROOT}/slime-terminal-RL}"
# HARBOR_SRC must be the root that makes `import harbor` work, i.e. the dir
# CONTAINING the harbor package -- verified: <root>/harbor/src/harbor/__init__.py.
# NOT slime's own plugin dir (slime_plugins/harbor), which is a different thing:
# that holds slime's Harbor *integration* hooks, while this is the Harbor
# framework itself (environments/daytona.py etc.). The inherited default points
# at an unmounted /path/to/shared_storage/.../harbor-code-synthesis/src.
export HARBOR_SRC="${HARBOR_SRC:-${USER_ROOT}/harbor/src}"
# 27B DENSE model shape args (num_query_groups 4, 64 layers, vocab 248320, GDN hybrid).
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-27B.sh}"
export HF_CKPT="${HF_CKPT:-/path/to/models/Qwen3.8-27B}"

# ---- CHECKPOINT: load root with the tracker file -----------------------------------
# Megatron resolves --load via <load>/latest_checkpointed_iteration.txt
# (checkpointing.py:281). The raw checkpoint dir has NO tracker and holds two
# iterations, so prepare_load_root.sh builds a dedicated symlink root pinned to
# iter_0000175 instead of mutating the shared dir.
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-${USER_ROOT}/checkpoints/q38_27b_ppo256k_loadroot}"

# ---- ALGO: PPO with a COLD critic ---------------------------------------------------
export ALGO="${ALGO:-ppo}"
# Cold critic: backbone copied from the actor ckpt, value head randomly
# re-initialised (normal_(0, 0.02), bias 0) by model.py after load_checkpoint.
# Requires that NO "<load>_critic" sibling exists -- verified by prepare_load_root.sh.
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-0}"

# ---- START_ROLLOUT_ID: MUST be 0 (this is why nothing ever trained) -----------------
# ROOT CAUSE of the "rollout ran, training never did" failure:
#
# train_async.py:64  ->  for rollout_id in range(args.start_rollout_id, args.num_rollout)
#
# When --start-rollout-id is unset, arguments.py leaves it None for a checkpoint whose
# tracker records a NUMERIC iteration, and placement_group.py:309-310 then sets it to
# loaded_iteration + 1. Our load root's latest_checkpointed_iteration.txt contains
# "175" (the SFT iteration), so start_rollout_id became 176 and the loop was
#     range(176, 2) == []   -> body NEVER executed
# i.e. no async_train, no backward pass, no optimizer step, no save. The 256 groups
# that DID get generated came from line 63, which kicks off
# `rollout_manager.generate.remote(args.start_rollout_id)` BEFORE the loop -- hence
# the log line "perf 176: ... num_accepted_groups: 256" followed immediately by
# "Calling rollout cleanup function for reason=dispose". 48 minutes of rollout,
# discarded, exit code 0.
#
# arguments.py:1779-1780 only auto-resets to 0 when the tracker says the literal
# "release". The reference 27B script never hit this because its PRETRAINED_CKPT is
# a *base/release* checkpoint; ours is a trained SFT iteration.
#
# This is a FRESH PPO run off an SFT checkpoint: the RL rollout counter must start at
# 0, independent of the SFT iteration number.
export START_ROLLOUT_ID="${START_ROLLOUT_ID:-0}"

# ---- CHECKPOINT SAVING: optimizer state REQUIRED -----------------------------------
# The 122B sh9 wrapper sets NO_SAVE_OPTIM=1 (weights-only, cannot resume). The
# requirement here is the OPPOSITE: save full optimizer state every 5 steps so the
# run is resumable. Saves are correspondingly much larger.
export NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-0}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"

# ---- TOPOLOGY: 64 train + 64 infer = 128 GPU ---------------------------------------
# Trainer 64 = TP4 x CP4 x PP2 => DP2      (8 nodes)
# Rollout 64 = 16 SGLang engines x TP4     (8 nodes)
# Exported EXPLICITLY so the parent's own case-block cannot override them; the
# parent assigns with ${VAR:-default}. TOPOLOGY is handed a name the parent
# accepts while TOPOLOGY_LABEL carries our identity (same trick the 122B wrapper uses).
export EXPECTED_NODES="${EXPECTED_NODES:-16}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-8}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-64}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-2}"
export CP_SIZE="${CP_SIZE:-4}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-4}"
# SGLang rollout TP -- 4, NOT 8 (same GQA constraint applies to the engines).
export SGLANG_TP="${SGLANG_TP:-4}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"

# ---- BATCH --------------------------------------------------------------------------
# GLOBAL_BATCH_SIZE 256 as specified; 256 % DP(2) == 0. OK.
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
# Dynamic global batch: tolerate the variable sample count that trajectory-mode
# rollouts produce (256 groups -> ~718 samples here), instead of asserting a fixed
# count. Without it a run can die with "Not enough samples N for global_batch_size".
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"

# ---- ROLLOUT MODE: standard, NOT fully_async ----------------------------------------
# Adopted from the proven 27B PPO recipe
# (run_synth3k_2820_96gpu_27b_daytona_ppo_partial_64k_fully_async.sh), which
# DELIBERATELY left fully-async mode after measuring a skimming bias
# (PPO_INTEGRATION_NOTES.md §11): fully_async's persistent workers backfill, so late
# steps churned 580-735 groups and kept only the fastest 256 -- an effective
# oversampling ratio of 3.9x that systematically favours short/easy tasks. The result
# was an inflated step-0 reward that then decayed as batch difficulty rose, i.e. a
# misleading learning curve.
#
# standard mode builds a fresh task set each step, dispatches OVER_SAMPLING, accepts
# the first 256 to finish, cancels the slow tail, and does NOT backfill -- skim ratio
# 288/256 = 1.125x. NOTE: in standard mode the yaml's
# harbor_fully_async_max_concurrent_groups is no longer read.
export HARBOR_ROLLOUT_MODE="${HARBOR_ROLLOUT_MODE:-standard}"
# 1.125x skim margin (matches the reference), enough to absorb one pathological task.
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-288}"

# ---- ROBUSTNESS (from the same proven recipe) ---------------------------------------
export USE_FAULT_TOLERANCE="${USE_FAULT_TOLERANCE:-1}"
export SLIME_GLOO_GROUP_TIMEOUT_MINUTES="${SLIME_GLOO_GROUP_TIMEOUT_MINUTES:-60}"
# STAGE_HF_CKPT_TO_TMP MUST stay 0. The inner launcher hard-codes
#   STAGED_HF_CKPT=${STAGED_HF_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.5-9B}
# (run_qwen35_9b_terminal_bench2_64h20.sh:49) -- the path is fixed to the 9B model
# regardless of HF_CKPT. Enabling staging silently rewrote --hf-checkpoint to
# .../Qwen3.5-9B (observed in a resolved TRAIN_CMD), which would train our 27B against
# a 9B tokenizer/config. Reading the 27B HF config from shared storage is slower but
# correct. (Set STAGED_HF_CKPT explicitly if staging is ever wanted.)
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-0}"
# --load points at shared storage so the "<load>_critic" sibling resolves on all nodes.
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-0}"

# ---- CONTEXT: 256k on BOTH sides ----------------------------------------------------
# The reference's cardinal rule is that train and rollout walls must AGREE (a prior
# run had training at 64k while SGLang served 84k and the agent believed 128k).
# All five walls go to 262144. max_position_embeddings is 262144, so 256k is NATIVE.
export SEQ_LENGTH="${SEQ_LENGTH:-262144}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-262144}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-262144}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-262144}"

# HARBOR_MAX_TRAIN_TOTAL_TOKENS is the TRAIN-SIDE wall and the single most
# important OOM knob. Per TRAINING_INTERRUPTION_FAILURES.md (~line 1582): a token
# cap like MAX_TOKENS_PER_GPU only bounds batch PACKING and "cannot split a single
# sample already longer than the cap" -- a 64k-capped run OOM'd trying to allocate
# 56.96 GiB for one ~55k-token sample. At a 256k wall a single sample can be 4x
# that, so this wall is what actually protects the trainer.
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-262144}"

# ---- MEMORY: the 256k vocab-CE peak -------------------------------------------------
# MEASURED, not estimated. probe_256k_memory.py isolates the vocab-projection +
# cross-entropy term at the real target shape (local_T=65536, local_V=248320/4)
# on an H20 (95.0 GB usable, not 97.9):
#
#   chunk       chunks   peak alloc   peak resv   headroom
#   unchunked        1     46.72 GB    54.30 GB    40.71 GB   <- matches the 45.4 GB projection (2.9% err)
#   32            2048     18.82 GB    24.33 GB    70.67 GB
#   512            128     18.70 GB    19.83 GB    75.17 GB
#   1024            64     18.37 GB    20.13 GB    74.88 GB
#   2048            32     18.16 GB    20.66 GB    74.35 GB
#   4096            16     18.60 GB    19.72 GB    75.28 GB
#
# Two conclusions:
#  (a) Chunking is the decisive mitigation: 46.72 -> ~18.5 GB (2.5x).
#  (b) The floor is FLAT from 32 to 4096 -- the residual is the retained hidden
#      states + backward graph, NOT the chunk buffer. So the template's
#      LOG_PROBS_CHUNK_SIZE=32 (tuned for a different, smaller-context failure)
#      buys ZERO extra memory here while running 2048 chunks instead of 16 --
#      128x the kernel launches for nothing. We use 1024 as a middle ground:
#      near-minimum memory with 64 chunks.
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-1}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-1024}"
export RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
export RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
export RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
# Fragmentation guard for the long-sequence allocator pattern.
# MUST NOT include expandable_segments:True: PPO forces offload_train=True
# (arguments.py:1862), so actor.py:76-78 sets torch_memory_saver.memory_margin_bytes,
# and its _sanity_checks() (torch_memory_saver/entrypoint.py:199) hard-raises
#   "TorchMemorySaver is disabled ... expandable_segments is not supported yet"
# on EVERY train actor, killing the job before the checkpoint even loads.
# Confirmed empirically in the 256k train-only smoke run before this fix.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
# 256k KV cache is large; leave headroom vs the template's default.
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"

# ---- DENSE MODEL: neutralise the 122B MoE inheritance -------------------------------
# The parent chain hard-codes 122B MoE assumptions that are WRONG for a dense 27B.
# Observed in the dry run: "train=64 (TP4 x PP2 x CP4 => DP2, EP16)" and
# "misc = --moe-token-dispatcher-type alltoall --use-rollout-routing-replay".
# Each of these must be cleared:
#
#  * EXPERT_MODEL_PARALLEL_SIZE=16 / EXPERT_TENSOR_PARALLEL_SIZE=1 -- there are no
#    experts (num_experts=None). EP must be 1, not 16. The parent also asserts
#    "EP*ETP must divide TP*CP*DP"; with EP=1 that is trivially satisfied.
#  * --moe-token-dispatcher-type alltoall -- INERT on this model, left as-is.
#    Both parents resolve it via `${EXTRA_MISC_ARGS:---moe-token-dispatcher-type
#    alltoall}`, and `:-` substitutes the default for an EMPTY value too, so
#    exporting "" cannot remove it. Verified safe rather than forced out:
#    the dispatcher is constructed only inside MoELayer (megatron moe_layer.py),
#    which is never instantiated when num_experts is None, and Megatron only
#    asserts on the "flex" dispatcher. So it stays unused config.
#    (Deliberately NOT replaced with a filler like --normalize-advantages:
#    arguments.py:1791 shows that flag is semantically load-bearing for some
#    advantage estimators and would alter PPO's advantage math.)
#  * USE_ROUTING_REPLAY / --use-rollout-routing-replay -- replays per-token expert
#    routing from rollout into training. There is no router on a dense model, and
#    the parent warns this ABORTS the whole rollout on a missing routing row
#    (harbor_fail_on_conversion_error=true). Must be off.
export EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-1}"
export EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
export USE_ROUTING_REPLAY="${USE_ROUTING_REPLAY:-0}"

# ---- CHECKPOINT LOAD: weights only, NOT the SFT optimizer state ---------------------
# The source checkpoint carries optimizer state from the SFT run (common.pt has an
# 'optimizer' key). Loading it here fails:
#   megatron/core/optimizer/distrib_optimizer.py:1778
#     sharded_param_state_fs_model_space -> mapping.py:134 validate_metadata_integrity
#   CheckpointingException: ShardedTensor.flattened_range is not supported.
# because the distributed-optimizer state was flattened for the SFT layout
# (TP4/PP2/CP2, actor only) and we are loading into TP4/PP2/CP4 with a critic.
#
# That state is not wanted anyway: this is a FRESH PPO run off an SFT checkpoint,
# so the optimizer should start clean (Adam moments from SFT are meaningless for
# RL). --no-load-optim takes the weights and skips the optimizer.
# NOTE this is LOAD-side only; NO_SAVE_OPTIM=0 above still SAVES full optimizer
# state every 5 steps, as required for resumability.
#
# Injected via EXTRA_MISC_ARGS, which the launcher word-splits into MISC_ARGS ->
# TRAIN_CMD (run_qwen35_9b_terminal_bench2_64h20.sh:670-672). This also gives
# EXTRA_MISC_ARGS a non-empty value, so the parents' `${EXTRA_MISC_ARGS:-<moe default>}`
# no longer injects --moe-token-dispatcher-type alltoall (harmless on a dense
# model, but cleaner absent).
export EXTRA_MISC_ARGS="${EXTRA_MISC_ARGS:---no-load-optim --no-load-rng}"

# ---- CONFIG -------------------------------------------------------------------------
export CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/harbor_mixed3000_128gpu_27b_dense_daytona_ppo_cp4_256k.yaml}"

# ---- TASK POOL ----------------------------------------------------------------------
# The parent validates TASKS_DIR and asserts it EQUALS the yaml's harbor_tasks_dir.
# Without this export it inherits the 122B default, which points at an unmounted
# /path/to/shared_storage path and dies in preflight ("task pool missing").
export TASKS_DIR="${TASKS_DIR:-/path/to/workspace/slime-terminal-RL_data/software_reconstruction/mixed3000_27b_ppo_256k_symlink}"
# The parent hard-asserts ~15000 tasks (MIN_TASKS default 14900) because the 122B
# recipe trains on the 15k strategy pool. Ours is the 3000-task mixed pool
# (2400 + 600), so lower the floor to the TRUE expected count -- keeping it a real
# assert that still catches an incomplete/mis-globbed pool rather than disabling it.
export MIN_TASKS="${MIN_TASKS:-3000}"

# ---- DENSE REWARD -------------------------------------------------------------------
# These tasks are semantic-verifier tasks: judge.py per-field-compares the candidate
# against a private oracle across 2-3 HIDDEN scenarios, but collapsed everything to a
# 0/1 exit code, so slime's partial-reward hook (which reads ONLY
# <trial>/verifier/ctrf.json) was a silent no-op -- every trajectory fell back to the
# binary terminal reward.
#
# patch_tasks_emit_ctrf.py added one fail-safe line to all 3000 tests/test.sh that
# converts judge.py's existing per-field report into CTRF, with
# --granularity semantic: ONLY the per-scenario oracle field comparisons count, NOT
# the ~7 hygiene checks (workspace_nonempty / has_reconstruct_py / ...). Measured on a
# synthetic gaming trajectory (files created, zero real progress):
#     granularity=all      -> 9/14 = 0.643   (rewards pure gaming)
#     granularity=semantic -> 0/3  = 0.000   (correctly denies credit)
# while a genuinely partial trajectory still scores 13/18 = 0.722 instead of 0.0.
# judge.py itself is untouched, so reward.txt and pass/fail semantics are unchanged.

# ---- CLUSTER: REUSE the existing head, never tear it down ---------------------------
# CLEAN_START MUST be 0: the base launcher defaults it to 1, which runs
# `ray stop --force` + `pkill raylet` and would kill the 16-node cluster.
export CLEAN_START="${CLEAN_START:-0}"
export USE_EXISTING_RAY_HEAD="${USE_EXISTING_RAY_HEAD:-1}"
# Explicit: the head bound 10.0.0.4:23456 (NON-default port). `hostname -I`
# would pick a different interface on this multi-homed box (this box is .7 /
# 10.0.0.10, NOT the head).
export MASTER_ADDR="${MASTER_ADDR:-10.0.0.4}"
export RAY_ADDRESS="${RAY_ADDRESS:-10.0.0.4:23456}"
# RAY_PORT is REQUIRED, not redundant with RAY_ADDRESS: the inner launcher does
#     export RAY_ADDRESS="${MASTER_ADDR}:${RAY_PORT}"     (line 749)
# with RAY_PORT defaulting to 6379, so it OVERWRITES whatever RAY_ADDRESS we set.
# This cluster's head listens on the NON-default 23456; without this the driver
# retries "Failed to connect to GCS at address 10.0.0.4:6379" forever
# (observed: stuck in that loop for 3+ minutes before being killed).
export RAY_PORT="${RAY_PORT:-23456}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/r27b256k}"

# DIRECT_RUN=1 runs train_async.py in THIS process (connecting through
# RAY_ADDRESS) instead of `ray job submit --address=http://127.0.0.1:8265`.
# Required here for two reasons:
#   1. The inner launcher hard-codes dashboard port 8265, but this cluster's head
#      serves its job API on 23457 -- verified: 8265 is NOT listening, 23457 is.
#      The submit path would poll 8265 for 60s and then fail.
#   2. Foreground execution keeps this shell alive for the whole run, so the
#      EXIT trap actually fires and the keepalive is restored. A detached
#      (`--no-wait`) submit bypassed the trap in the earlier smoke run and left
#      all 16 nodes with procs=0.
# DIRECT_RUN=0: use the launcher's `ray job submit --runtime-env-json` path.
# That is the PROVEN mechanism (the previously SUCCEEDED jobs on this cluster all
# used it) and, critically, it is the ONLY path that applies RUNTIME_ENV_JSON --
# PYTHONPATH, proxies, NCCL/IB settings -- to the Ray workers.
#
# With DIRECT_RUN=1 the submit is skipped and RUNTIME_ENV_JSON is never applied;
# exporting PYTHONPATH in this shell does NOT help because Ray workers inherit
# from the RAYLET, and these raylets were started without one (verified by
# reading /proc/<raylet>/environ). Result: every train actor died with
#   ModuleNotFoundError: No module named 'megatron.training'
#
# The submit path hard-codes `--address=http://127.0.0.1:8265`, but this cluster's
# head serves its job API on 23457 (8265 is not listening). Rather than edit that
# shared script, raybin/ray is a PATH shim that rewrites exactly that address.
export DIRECT_RUN="${DIRECT_RUN:-0}"
export PATH="${USER_ROOT}/raybin:${PATH}"
export SLIME_RAY_DASHBOARD="${SLIME_RAY_DASHBOARD:-http://${MASTER_ADDR}:23457}"
# Belt-and-braces: BACKGROUND=0 so the parent chain does not nohup-detach either.
export BACKGROUND="${BACKGROUND:-0}"

# ---- WORKER ENVIRONMENT (required because DIRECT_RUN bypasses runtime_env) ----------
# The inner launcher builds RUNTIME_ENV_JSON (PYTHONPATH, proxies, NCCL vars) but
# only applies it via `ray job submit`. Under DIRECT_RUN=1 that submit is skipped,
# so RUNTIME_ENV_JSON is NEVER APPLIED and Ray workers inherit only what this
# process exports. Without the exports below the actors die with
#   ModuleNotFoundError: No module named 'megatron.training'
# (observed: MegatronTrainRayActor failed to import on every worker).
#
# Ray propagates the driver's environment to workers it starts, so exporting here
# is sufficient. /root/Megatron-LM is present on every node (verified).
export PYTHONPATH="/root/Megatron-LM:${HARBOR_SRC}:${SLIME_DIR}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-bond1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"

# These MUST be non-empty. The launcher forwards them into RUNTIME_ENV_JSON with
# `"${VAR:-}"`, so when unset they arrive at the workers as EMPTY STRINGS -- and
# the consuming code uses `os.environ.get(NAME, "<default>")`, whose default only
# applies when the key is ABSENT. An empty value therefore reaches float()/int()
# and raises. Observed: slime/ray/utils.py:64
#   float(os.environ.get("SLIME_ENGINE_LOCK_TIMEOUT_S", "1800"))
#   -> ValueError: could not convert string to float: ''
# which killed the Lock actor (and with it the weight-broadcast path).
export SLIME_ENGINE_LOCK_TIMEOUT_S="${SLIME_ENGINE_LOCK_TIMEOUT_S:-1800}"
export SLIME_NCCL_GROUP_TIMEOUT_MINUTES="${SLIME_NCCL_GROUP_TIMEOUT_MINUTES:-240}"
export SLIME_NEW_GROUP_TIMEOUT_MINUTES="${SLIME_NEW_GROUP_TIMEOUT_MINUTES:-240}"
# Cluster-internal traffic must bypass the proxy (Ray head + all node IPs).
export no_proxy="${no_proxy:+${no_proxy},}localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"
export NO_PROXY="${no_proxy}"

# ---- RUN IDENTITY -------------------------------------------------------------------
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export TOPOLOGY_LABEL="${TOPOLOGY_LABEL:-128tp4cp4pp2-256k}"
export TOPOLOGY="${TOPOLOGY:-88cp4-84k}"   # a preset name the parent accepts
export BASE_FOLDER="${BASE_FOLDER:-${USER_ROOT}/slime_harbor_runs/q38_27b_dense_ppo_256k_${RUN_TAG}}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.8-27B_dense_ppo_256k_mixed3000}"
# RUN_TRIALS_DIR must be set too: the parent defaults it to an /path/to/shared_storage path
# and then `mkdir -p`s it, which fails with "Transport endpoint is not connected"
# because gy4 is not mounted on this cluster. The parent also rewrites this value
# into the yaml's trials_dir, and refuses to reuse a non-empty dir.
export RUN_TRIALS_DIR="${RUN_TRIALS_DIR:-${USER_ROOT}/slime_harbor_trials/q38_27b_ppo_256k_${RUN_TAG}}"

# ---- CREDENTIALS + WANDB ------------------------------------------------------------
export WANDB_KEY="${WANDB_KEY:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_KEY}}"
export WANDB_PROJECT="${WANDB_PROJECT:-terminal3_rl}"
export WANDB_GROUP="${WANDB_GROUP:-q38-27b-dense-ppo-${TOPOLOGY_LABEL}}"
export WANDB_MODE="${WANDB_MODE:-online}"
export DAYTONA_API_KEY="${DAYTONA_API_KEY:-}"

# ---- SMOKE MODE ---------------------------------------------------------------------
# One 256k step end-to-end on the real 128-GPU topology, to prove: critic cold-start
# loads + emits a scalar value, Daytona rollout generates, and the trainer completes
# one update under CP=4 + full recompute without OOM.
if [ "${SMOKE:-0}" = "1" ]; then
  export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
  export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
  export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE_SMOKE:-8}"
  export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE_SMOKE:-8}"
  export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE_SMOKE:-10}"
  export HARBOR_NUM_CONCURRENT_TRIALS="${HARBOR_NUM_CONCURRENT_TRIALS_SMOKE:-10}"
  export WANDB_MODE=offline
  export RUN_TAG="smoke_${RUN_TAG}"
  export BASE_FOLDER="${USER_ROOT}/slime_harbor_runs/q38_27b_dense_ppo_256k_SMOKE_${RUN_TAG}"
  # RUN_TRIALS_DIR was computed from the pre-smoke RUN_TAG above, so recompute it
  # here. The parent refuses to start if RUN_TRIALS_DIR exists and is non-empty,
  # so each smoke run needs its own fresh path.
  export RUN_TRIALS_DIR="${USER_ROOT}/slime_harbor_trials/q38_27b_ppo_256k_SMOKE_${RUN_TAG}"
  echo "[27b256k] *** SMOKE MODE: NUM_ROLLOUT=1, batch=8, wandb offline ***"
fi

# ---- PREFLIGHT ----------------------------------------------------------------------
_die() { echo "ERROR: $*" >&2; exit 1; }

for p in "${SLIME_DIR}" "${HARBOR_SRC}" "${PRETRAINED_CKPT}" "${HF_CKPT}" "${CONFIG_TEMPLATE}" "${MODEL_SCRIPT}"; do
  [ -e "${p}" ] || _die "missing path: ${p}"
done

[ -f "${PRETRAINED_CKPT}/latest_checkpointed_iteration.txt" ] \
  || _die "no latest_checkpointed_iteration.txt under ${PRETRAINED_CKPT} -- run prepare_load_root.sh"

# The critic must COLD start: a _critic sibling would silently switch it to warm-load.
# WARM_CRITIC=1 opts out, for a follow-on run that DELIBERATELY resumes a trained
# value head (placement_group.py:195 redirects --load to "<load>_critic" when that
# sibling exists). Left as a hard gate by default because a _critic dir appearing by
# accident -- e.g. a stale symlink from a previous run -- would silently change the
# algorithm from cold- to warm-start with no other visible signal.
if [ "${WARM_CRITIC:-0}" != "1" ]; then
  [ -d "${PRETRAINED_CKPT}_critic" ] \
    && _die "${PRETRAINED_CKPT}_critic exists -> critic would WARM-load; cold start requires it absent (set WARM_CRITIC=1 if intended)"
else
  [ -d "${PRETRAINED_CKPT}_critic" ] \
    || _die "WARM_CRITIC=1 but ${PRETRAINED_CKPT}_critic is absent -> critic would COLD start with a random value head"
  [ -f "${PRETRAINED_CKPT}_critic/latest_checkpointed_iteration.txt" ] \
    || _die "WARM_CRITIC=1 but ${PRETRAINED_CKPT}_critic has no latest_checkpointed_iteration.txt"
fi

# CP>1 needs fla.ops.cp (from fla-core). Fail here rather than at step 1.
python3 -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('fla.ops.cp') else 1)" \
  || _die "fla.ops.cp missing -- pip install --no-deps 'fla-core>=0.4.2' (NOT just flash-linear-attention)"

# Topology arithmetic.
_TRAIN_GPUS=$(( ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE ))
_NEED=$(( _TRAIN_GPUS + ROLLOUT_NUM_GPUS ))
_DIV=$(( TENSOR_MODEL_PARALLEL_SIZE * CP_SIZE * PIPELINE_MODEL_PARALLEL_SIZE ))
[ $(( _TRAIN_GPUS % _DIV )) -eq 0 ] \
  || _die "train GPUs ${_TRAIN_GPUS} not divisible by TP*CP*PP=${_DIV}"
_DP=$(( _TRAIN_GPUS / _DIV ))
[ $(( GLOBAL_BATCH_SIZE % _DP )) -eq 0 ] \
  || _die "GLOBAL_BATCH_SIZE ${GLOBAL_BATCH_SIZE} not divisible by DP ${_DP}"
[ $(( 4 % TENSOR_MODEL_PARALLEL_SIZE )) -eq 0 ] \
  || _die "num_query_groups(4) % TP(${TENSOR_MODEL_PARALLEL_SIZE}) != 0 -- TP>4 is illegal for this model"
[ $(( ROLLOUT_NUM_GPUS % SGLANG_TP )) -eq 0 ] \
  || _die "ROLLOUT_NUM_GPUS ${ROLLOUT_NUM_GPUS} not divisible by SGLANG_TP ${SGLANG_TP}"

# Live cluster must actually have the GPUs, or Ray hangs forever waiting.
_AVAIL=$(RAY_ADDRESS="${RAY_ADDRESS}" ray status 2>/dev/null | grep -oE '0\.0/[0-9]+\.0 GPU' | grep -oE '/[0-9]+' | tr -d '/' | head -1)
[ -n "${_AVAIL}" ] || _die "cannot read GPU count from ray status at ${RAY_ADDRESS}"
[ "${_AVAIL}" -ge "${_NEED}" ] \
  || _die "cluster has ${_AVAIL} GPU but topology needs ${_NEED}"

echo "[27b256k] ---------------- preflight ----------------"
echo "[27b256k] model      = 27B DENSE (no EP/MoE), GDN hybrid, vocab 248320"
echo "[27b256k] topology   = ${_TRAIN_GPUS} train (TP${TENSOR_MODEL_PARALLEL_SIZE} x CP${CP_SIZE} x PP${PIPELINE_MODEL_PARALLEL_SIZE} => DP${_DP}) + ${ROLLOUT_NUM_GPUS} infer (${SGLANG_TP}-way TP x $((ROLLOUT_NUM_GPUS/SGLANG_TP)) engines) = ${_NEED} GPU"
echo "[27b256k] cluster    = ${_AVAIL} GPU available at ${RAY_ADDRESS}"
echo "[27b256k] context    = train ${SEQ_LENGTH} / rollout ${ROLLOUT_MAX_CONTEXT_LEN} / sglang ${SGLANG_CONTEXT_LENGTH} / train-wall ${HARBOR_MAX_TRAIN_TOTAL_TOKENS}"
echo "[27b256k] compaction = threshold 229376 (=262144-32768, last 32k reserved)"
echo "[27b256k] batch      = global ${GLOBAL_BATCH_SIZE}, rollout ${ROLLOUT_BATCH_SIZE}, oversample ${OVER_SAMPLING_BATCH_SIZE}"
echo "[27b256k] memory     = dynbatch ${MAX_TOKENS_PER_GPU} tok/gpu, lpc ${LOG_PROBS_CHUNK_SIZE}, recompute ${RECOMPUTE_GRANULARITY}/${RECOMPUTE_METHOD}/${RECOMPUTE_NUM_LAYERS}"
echo "[27b256k] critic     = $([ "${WARM_CRITIC:-0}" = "1" ] && echo "WARM (value head from ${PRETRAINED_CKPT}_critic)" || echo "COLD (random value head, no _critic sibling)"), critic-only steps ${NUM_CRITIC_ONLY_STEPS}"
echo "[27b256k] ckpt       = load ${PRETRAINED_CKPT} (iter $(cat "${PRETRAINED_CKPT}/latest_checkpointed_iteration.txt"))"
echo "[27b256k] save       = every ${SAVE_INTERVAL} steps, optimizer state $([ "${NO_SAVE_OPTIM}" = "0" ] && echo INCLUDED || echo OMITTED)"
echo "[27b256k] config     = ${CONFIG_TEMPLATE}"
echo "[27b256k] wandb      = ${WANDB_PROJECT} / ${WANDB_GROUP} (${WANDB_MODE})"
echo "[27b256k] -------------------------------------------"

# ---- ENSURE THE KEEPALIVE IS UP, THEN HAND OFF --------------------------------------
# Deliberately `start`, NOT `stop`: the cluster is reclaimed if utilisation stays
# low, and the startup window (SGLang boot + checkpoint load) runs at ~0% util.
# The keepalive backs off automatically once the GPUs are busy (min_util 80).
echo "[27b256k] ensuring GPU keepalive is running on all nodes before launch"
if [ -x "${KEEPALIVE_CTL}" ]; then
  "${KEEPALIVE_CTL}" start
  _ka_missing=$("${KEEPALIVE_CTL}" status 2>/dev/null | grep -c "procs=0" || true)
  if [ "${_ka_missing}" -gt 0 ]; then
    _die "keepalive still missing on ${_ka_missing} node(s) -- the cluster would be reclaimed; fix before launching"
  fi
  echo "[27b256k] keepalive confirmed on all 16 nodes"
else
  echo "[27b256k] WARN: ${KEEPALIVE_CTL} not executable -- cluster is UNPROTECTED" >&2
fi

PARENT="${SCRIPT_DIR}/run_tmax15k_122b_a10b_scalable_ppo_criticlr30x_strongest_partial_reward.sh"
[ -x "${PARENT}" ] || [ -f "${PARENT}" ] || _die "parent recipe not found: ${PARENT}"
echo "[27b256k] handing off to ${PARENT}"
bash "${PARENT}"
