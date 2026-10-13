#!/usr/bin/env bash
# examples/terminal_rsi/common.sh
#
# Shared launcher for the Terminal-RSI Recursive Policy-Refine (RPR) phases on
# Qwen3.8-27B (dense, GatedDeltaNet hybrid, 256k context). Do not run directly --
# it is sourced by run_rpr1_fdr.sh / run_rpr2_spr.sh, which set the phase-specific
# variables below and then call `rpr_launch`.
#
# Phase-specific variables (set by the phase script):
#   PHASE                 rpr1 | rpr2
#   HARBOR_CONFIG         Harbor YAML template (configs/harbor_*.yaml)
#   MEGATRON_ROLE_CONFIG  per-role actor/critic overrides (configs/megatron_roles_*.yaml)
#   REWARD_FN             dotted path of the reward post-process hook
#   ACTOR_LR              top-level --lr (must agree with the per-role actor block)
#
# Required user variables:
#   HF_CKPT          HuggingFace checkpoint dir of Qwen3.8-27B (tokenizer + config)
#   LOAD_ROOT        Megatron torch_dist load root of the ACTOR; the warm CRITIC must
#                    live in the sibling "${LOAD_ROOT}_critic" (see prepare_load_root.sh)
#   TASKS_DIR        training task pool (Harbor task dirs, one task.toml each)
#   EVAL_TASKS_DIR   held-out eval pool (e.g. Terminal-Bench 2.x tasks)
#   HARBOR_SRC       directory that makes `import harbor` work (Harbor framework src/)
#   DAYTONA_API_KEY  Daytona sandbox credentials (environment.type: daytona)
#
# Optional user variables (defaults in parentheses):
#   OUTPUT_ROOT (./outputs)  MEGATRON_LM_DIR (/root/Megatron-LM)  RUN_TAG (timestamp)
#   ACTOR_NUM_NODES (8)  ACTOR_NUM_GPUS_PER_NODE (8)  ROLLOUT_NUM_GPUS (56)
#   NUM_ROLLOUT (36)  SAVE_INTERVAL (2)  NUM_CRITIC_ONLY_STEPS (4)
#   WARM_CRITIC (1; 0 = cold value head, used for the Warm-up Critic stage)
#   USE_WANDB (1)  WANDB_API_KEY  WANDB_PROJECT (terminal-rsi)  WANDB_GROUP
#   RAY_DASHBOARD (http://127.0.0.1:8265)  DIRECT_RUN (0)  DRY_RUN (0)
#   SAVE_ROLLOUT_DUMPS (0)  EXTRA_ARGS ("")
set -euo pipefail

RSI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RSI_DIR}/../.." && pwd)"

_die() { echo "[terminal-rsi] ERROR: $*" >&2; exit 1; }

rpr_launch() {
  : "${PHASE:?}" "${HARBOR_CONFIG:?}" "${MEGATRON_ROLE_CONFIG:?}" "${REWARD_FN:?}" "${ACTOR_LR:?}"
  : "${HF_CKPT:?set HF_CKPT to the Qwen3.8-27B HuggingFace checkpoint dir}"
  local eval_only="${EVAL_ONLY:-0}"
  if [ "${eval_only}" = "1" ]; then
    # Rollout-only evaluation: SGLang serves HF_CKPT directly, no trainer / critic.
    LOAD_ROOT="${LOAD_ROOT:-${HF_CKPT}}"
    TASKS_DIR="${TASKS_DIR:-${EVAL_TASKS_DIR:-}}"
  fi
  : "${LOAD_ROOT:?set LOAD_ROOT to the actor torch_dist load root (critic in \${LOAD_ROOT}_critic)}"
  : "${TASKS_DIR:?set TASKS_DIR to the training task pool}"
  : "${EVAL_TASKS_DIR:?set EVAL_TASKS_DIR to the evaluation task pool}"
  : "${HARBOR_SRC:?set HARBOR_SRC to the Harbor framework src/ dir}"

  local run_tag="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
  local output_root="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
  local run_dir="${output_root}/terminal_rsi_27b_${PHASE}_${run_tag}"
  local save_dir="${run_dir}/checkpoints"
  local trials_dir="${run_dir}/harbor_trials"
  local megatron_lm_dir="${MEGATRON_LM_DIR:-/root/Megatron-LM}"

  # ---- preflight -------------------------------------------------------------------
  for p in "${HF_CKPT}" "${LOAD_ROOT}" "${TASKS_DIR}" "${EVAL_TASKS_DIR}" "${HARBOR_SRC}" \
           "${HARBOR_CONFIG}" "${MEGATRON_ROLE_CONFIG}"; do
    [ -e "${p}" ] || _die "missing path: ${p}"
  done
  if [ "${eval_only}" != "1" ]; then
    [ -f "${LOAD_ROOT}/latest_checkpointed_iteration.txt" ] \
      || _die "${LOAD_ROOT} has no latest_checkpointed_iteration.txt (run prepare_load_root.sh)"
    # slime redirects the critic's --load to the "<load>_critic" sibling when it exists;
    # without it the critic copies the actor backbone and re-initialises the value head.
    if [ "${WARM_CRITIC:-1}" = "1" ]; then
      [ -f "${LOAD_ROOT}_critic/latest_checkpointed_iteration.txt" ] \
        || _die "warm critic missing: ${LOAD_ROOT}_critic (RPR requires the retained critic; WARM_CRITIC=0 for a cold start)"
    else
      [ ! -e "${LOAD_ROOT}_critic" ] \
        || _die "WARM_CRITIC=0 but ${LOAD_ROOT}_critic exists -- the critic would silently warm-load"
    fi
  fi
  [ "${DRY_RUN:-0}" = "1" ] || [ "${eval_only}" = "1" ] \
    || python3 -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('fla.ops.cp') else 1)" \
    || _die "fla.ops.cp missing -- pip install 'fla-core>=0.4.2' (needed for CP>1 on GatedDeltaNet)"
  [ "${DRY_RUN:-0}" = "1" ] || [ -n "${DAYTONA_API_KEY:-}" ] || _die "DAYTONA_API_KEY is not set"

  # ---- materialise the Harbor config (only the three path placeholders change) ------
  mkdir -p "${run_dir}" "${trials_dir}"
  local config="${run_dir}/harbor_config_resolved.yaml"
  sed -e "s#__TASKS_DIR__#${TASKS_DIR}#" \
      -e "s#__EVAL_TASKS_DIR__#${EVAL_TASKS_DIR}#" \
      -e "s#__TRIALS_DIR__#${trials_dir}#" \
      "${HARBOR_CONFIG}" > "${config}"
  ! grep -q '__[A-Z_]*_DIR__' "${config}" || _die "unresolved placeholder in ${config}"

  # ---- model: Qwen3.8-27B dense (64 layers, GQA 24/4, gated attention, GDN hybrid) ---
  local MODEL_ARGS=(
    --spec slime_plugins.models.qwen3_5 get_qwen3_5_spec
    --disable-bias-linear --qk-layernorm --group-query-attention
    --num-attention-heads 24 --num-query-groups 4 --kv-channels 256
    --num-layers 64 --hidden-size 5120 --ffn-hidden-size 17408
    --use-gated-attention --normalization RMSNorm --apply-layernorm-1p
    --position-embedding-type rope --norm-epsilon 1e-6 --rotary-percent 0.25
    --swiglu --untie-embeddings-and-output-weights --vocab-size 248320
    --rotary-base 10000000 --attention-output-gate
  )

  # ---- checkpoints: weights-only warm start of actor (LOAD_ROOT) + critic (_critic) --
  local CKPT_ARGS=(
    --train-backend megatron
    --hf-checkpoint "${HF_CKPT}"
    --load "${LOAD_ROOT}" --ref-load "${LOAD_ROOT}"
    --save "${save_dir}" --save-interval "${SAVE_INTERVAL:-2}"
    --start-rollout-id 0
    --no-load-optim --no-load-rng
  )

  # ---- Harbor agentic rollout (Terminus-2 on Daytona sandboxes) ----------------------
  local ROLLOUT_ARGS=(
    --rollout-function-path slime_plugins.harbor.rollout.generate_rollout
    --data-source-path slime_plugins.harbor.data_source.HarborTaskDataSource
    --custom-config-path "${config}"
    --rollout-batch-size 256 --n-samples-per-prompt 1
    --num-rollout "${NUM_ROLLOUT:-36}"
    --rollout-max-response-len 8192
    --rollout-temperature 1.0 --rollout-top-p 1.0 --rollout-top-k -1
    --global-batch-size 256 --balance-data
    --rollout-max-context-len 262144
    --over-sampling-batch-size 288 --use-dynamic-global-batch-size
    --rollout-all-samples-process-path slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples
    --rollout-data-postprocess-path slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data
    --custom-reward-post-process-path "${REWARD_FN}"
    --use-fault-tolerance
    --rollout-health-check-interval 30 --rollout-health-check-timeout 120
    --rollout-health-check-first-wait 600
  )

  # ---- evaluation on the held-out pool (harbor_eval_tasks_dir in the YAML) -----------
  local EVAL_ARGS=(
    --eval-interval 10
    --eval-function-path slime_plugins.harbor.rollout.generate_rollout
    --eval-prompt-data tb2 /dev/null
    --n-samples-per-eval-prompt 3
    --eval-temperature 1.0 --eval-top-p 0.95 --eval-top-k 20
    --skip-eval-before-train
  )

  # ---- optimizer (per-role overrides in MEGATRON_ROLE_CONFIG are authoritative) ------
  local OPTIMIZER_ARGS=(
    --optimizer adam --lr "${ACTOR_LR}" --lr-decay-style constant
    --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
    --use-precision-aware-optimizer --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d
    --megatron-config-path "${MEGATRON_ROLE_CONFIG}"
    --override-opt_param-scheduler
  )

  # ---- RPR objective: PPO actor-critic, asymmetric clip, no KL / entropy -------------
  local RPR_ARGS=(
    --advantage-estimator ppo
    --kl-loss-coef 0.0 --kl-loss-type low_var_kl --kl-coef 0.00 --entropy-coef 0.0
    --eps-clip 0.3 --eps-clip-high 5.0
    --keep-old-actor
    --loss-type custom_loss
    --custom-loss-function-path slime_plugins.harbor.train_loss.policy_loss_no_entropy_when_disabled
    # phase-local critic calibration: the first N rollouts update only the critic
    --num-critic-only-steps "${NUM_CRITIC_ONLY_STEPS:-4}"
  )

  # ---- parallelism: train TP4 x CP4 x PP2 (DP2 on 64 GPUs), rollout 14 x TP4 engines -
  local PERF_ARGS=(
    --tensor-model-parallel-size 4 --pipeline-model-parallel-size 2
    --context-parallel-size 4
    --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --qkv-format thd --micro-batch-size 1 --sequence-parallel
    --distributed-timeout-minutes 180
    --seq-length 262144 --max-position-embeddings 262144
    --use-dynamic-batch-size --max-tokens-per-gpu 8192
    --data-pad-size-multiplier 2048 --log-probs-chunk-size 1024
    --attention-dropout 0.0 --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
    --attention-backend flash --moe-token-dispatcher-type alltoall
  )

  local SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 4
    --sglang-mem-fraction-static 0.75
    --sglang-context-length 262144 --sglang-max-prefill-tokens 262144
    --sglang-chunked-prefill-size 8192
    --sglang-mamba-scheduler-strategy extra_buffer
    --sglang-disable-custom-all-reduce
  )

  local LOG_ARGS=()
  if [ "${SAVE_ROLLOUT_DUMPS:-0}" = "1" ]; then
    LOG_ARGS+=(--save-debug-rollout-data "${run_dir}/rollout_dumps/{rollout_id}.pt")
  fi
  if [ "${USE_WANDB:-1}" = "1" ]; then
    LOG_ARGS+=(
      --use-wandb --wandb-mode "${WANDB_MODE:-online}"
      --wandb-project "${WANDB_PROJECT:-terminal-rsi}"
      --wandb-group "${WANDB_GROUP:-terminal-rsi-27b-${PHASE}}"
      --wandb-dir "${run_dir}/wandb" --disable-wandb-random-suffix
    )
  fi

  local entrypoint=train_async.py
  if [ "${eval_only}" = "1" ]; then
    # Same model / rollout / SGLang stack, but: no trainer, no critic, evaluate once at
    # step 0 (train.py's eval-only path for num_rollout == 0) and exit.
    entrypoint=train.py
    CKPT_ARGS=(--train-backend megatron --hf-checkpoint "${HF_CKPT}" --debug-rollout-only)
    EVAL_ARGS=(
      --eval-interval 1
      --eval-function-path slime_plugins.harbor.rollout.generate_rollout
      --eval-prompt-data tb2 /dev/null
      --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-3}"
      --eval-temperature 1.0 --eval-top-p 0.95 --eval-top-k 20
    )
    local filtered=() skip_next=0 a
    for a in "${ROLLOUT_ARGS[@]}"; do
      if [ "${skip_next}" = "1" ]; then skip_next=0; continue; fi
      if [ "${a}" = "--num-rollout" ]; then skip_next=1; continue; fi
      filtered+=("${a}")
    done
    ROLLOUT_ARGS=("${filtered[@]}" --num-rollout 0)
    RPR_ARGS=(--advantage-estimator grpo)
    OPTIMIZER_ARGS=()
  fi

  local EXTRA=()
  # shellcheck disable=SC2206
  [ -n "${EXTRA_ARGS:-}" ] && EXTRA=(${EXTRA_ARGS})

  local TRAIN_CMD=(
    python3 "${entrypoint}"
    --actor-num-nodes "${ACTOR_NUM_NODES:-8}"
    --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE:-8}"
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS:-56}"
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${EVAL_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}" "${RPR_ARGS[@]}" "${PERF_ARGS[@]}" "${SGLANG_ARGS[@]}"
    "${LOG_ARGS[@]}" "${EXTRA[@]}"
  )

  # ---- worker environment ------------------------------------------------------------
  # The train-side token wall bounds a single 256k sample (MAX_TOKENS_PER_GPU only bounds
  # packing); expandable_segments must stay off because PPO enables torch_memory_saver.
  export PYTHONPATH="${megatron_lm_dir}:${HARBOR_SRC}:${REPO_ROOT}:${PYTHONPATH:-}"
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-262144}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
  export SLIME_ENGINE_LOCK_TIMEOUT_S="${SLIME_ENGINE_LOCK_TIMEOUT_S:-1800}"
  export SLIME_NCCL_GROUP_TIMEOUT_MINUTES="${SLIME_NCCL_GROUP_TIMEOUT_MINUTES:-240}"
  export SLIME_NEW_GROUP_TIMEOUT_MINUTES="${SLIME_NEW_GROUP_TIMEOUT_MINUTES:-240}"
  export SLIME_GLOO_GROUP_TIMEOUT_MINUTES="${SLIME_GLOO_GROUP_TIMEOUT_MINUTES:-60}"
  export HARBOR_REPORT_ENTROPY="${HARBOR_REPORT_ENTROPY:-1}"

  local runtime_env
  runtime_env=$(python3 - <<'PY'
import json, os
keys = ["PYTHONPATH", "CUDA_DEVICE_MAX_CONNECTIONS", "HARBOR_MAX_TRAIN_TOTAL_TOKENS",
        "PYTORCH_CUDA_ALLOC_CONF", "SLIME_ENGINE_LOCK_TIMEOUT_S",
        "SLIME_NCCL_GROUP_TIMEOUT_MINUTES", "SLIME_NEW_GROUP_TIMEOUT_MINUTES",
        "SLIME_GLOO_GROUP_TIMEOUT_MINUTES", "HARBOR_REPORT_ENTROPY",
        "DAYTONA_API_KEY", "DAYTONA_API_URL", "DAYTONA_ORGANIZATION_ID", "DAYTONA_JWT_TOKEN",
        "WANDB_API_KEY", "WANDB_MODE", "MASTER_ADDR",
        "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME",
        "NCCL_IB_DISABLE", "NCCL_IB_GID_INDEX",
        "http_proxy", "https_proxy", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"]
print(json.dumps({"env_vars": {k: os.environ[k] for k in keys if os.environ.get(k)}}))
PY
)

  echo "[terminal-rsi] phase        = ${PHASE}"
  if [ "${eval_only}" = "1" ]; then
    echo "[terminal-rsi] model (HF)   = ${HF_CKPT}"
    echo "[terminal-rsi] eval pool    = ${EVAL_TASKS_DIR}"
  else
    echo "[terminal-rsi] reward       = ${REWARD_FN}"
    echo "[terminal-rsi] actor lr     = ${ACTOR_LR}  (roles: ${MEGATRON_ROLE_CONFIG})"
    echo "[terminal-rsi] load (actor) = ${LOAD_ROOT}  (iter $(cat "${LOAD_ROOT}/latest_checkpointed_iteration.txt"))"
    echo "[terminal-rsi] load (critic)= ${LOAD_ROOT}_critic"
  fi
  echo "[terminal-rsi] run dir      = ${run_dir}"

  cd "${REPO_ROOT}"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '%q ' "${TRAIN_CMD[@]}"; echo
  elif [ "${DIRECT_RUN:-0}" = "1" ]; then
    "${TRAIN_CMD[@]}"
  else
    ray job submit --address="${RAY_DASHBOARD:-http://127.0.0.1:8265}" \
      --runtime-env-json="${runtime_env}" \
      -- "${TRAIN_CMD[@]}"
  fi
}
