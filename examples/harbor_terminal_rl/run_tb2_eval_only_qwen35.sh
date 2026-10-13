#!/bin/bash
# SGLang-only Harbor TB2 evaluation. MODEL_KIND=27b or 35b-a3b.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_KIND="${MODEL_KIND:-${1:-27b}}"
export SLIME_DIR="${SLIME_DIR:-/path/to/workspace/slime-terminal-RL}"
export CODE_ROOT="${CODE_ROOT:-${SLIME_DIR}}"
export HARBOR_SRC="${HARBOR_SRC:-/path/to/workspace/harbor-code-synthesis/src}"
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_tb2_eval_only_tmax_timeouts.yaml}"
export HARBOR_ROLLOUT_MODE=standard
export TRAIN_ENTRYPOINT=train.py
export DEBUG_ROLLOUT_ONLY=1
export MEGATRON_TO_HF_MODE=bridge
export NUM_ROLLOUT=0
export EVAL_INTERVAL=1
export SKIP_EVAL_BEFORE_TRAIN=0
export ROLLOUT_BATCH_SIZE=512
export N_SAMPLES_PER_PROMPT=1
export GLOBAL_BATCH_SIZE=512
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
export EVAL_DATASET_NAME=tb2
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
export EVAL_TOP_P="${EVAL_TOP_P:-0.95}"
export EVAL_TOP_K="${EVAL_TOP_K:-20}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-56}"
export EXPECTED_NODES="${EXPECTED_NODES:-7}"
export ACTOR_NUM_NODES=1
export ACTOR_NUM_GPUS_PER_NODE=8
export ROLLOUT_MAX_CONTEXT_LEN=65536
export SGLANG_CONTEXT_LENGTH=65536
export SGLANG_MAX_PREFILL_TOKENS=65536
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"
export ROLLOUT_MAX_RESPONSE_LEN=4096
export ADVANTAGE_ESTIMATOR=grpo
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-terminal-rl}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE=0
export OVER_SAMPLING_BATCH_SIZE=""
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-1}"
export STAGE_TORCH_DIST_CKPT_TO_TMP=0
export CLEAN_START="${CLEAN_START:-0}"
export USE_EXISTING_RAY_HEAD="${USE_EXISTING_RAY_HEAD:-1}"
export DIRECT_RUN=0
export HARBOR_CREATE_RATE_PER_SEC="${HARBOR_CREATE_RATE_PER_SEC:-6.0}"
case "${MODEL_KIND}" in
  27b)
    export HF_CKPT=/path/to/shared_storage/hf_models/Qwen/Qwen3.5-27B
    export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/shared_storage/hf_models/Qwen/Qwen3.5-27B_torch_dist}"
    export STAGED_HF_CKPT=/tmp/slime-harbor-model-cache/Qwen3.5-27B
    export MODEL_SCRIPT="${SLIME_DIR}/scripts/models/qwen3.5-27B.sh"
    export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
    export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
    export EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:-}"
    export SAVE_SUBDIR="tb2_eval_only_qwen35_27b"
    export WANDB_GROUP="${WANDB_GROUP:-tb2-eval-only-qwen35-27b-bs512-n1-tmax-timeouts}"
    ;;
  35b-a3b|35b)
    export HF_CKPT=/path/to/shared_storage/hf_models/Qwen/Qwen3.5-35B-A3B
    export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/shared_storage/hf_models/Qwen/Qwen3.5-35B-A3B_torch_dist}"
    export STAGED_HF_CKPT=/tmp/slime-harbor-model-cache/Qwen3.5-35B-A3B
    export MODEL_SCRIPT="${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"
    export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
    export ROLLOUT_DP_SIZE=8
    export ROLLOUT_EP_SIZE=8
    export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
    export EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:---sglang-enable-dp-attention --sglang-dp-size 8 --sglang-ep-size 8 --sglang-enable-dp-lm-head --sglang-moe-dense-tp-size 1 --sglang-mamba-scheduler-strategy extra_buffer}"
    export SAVE_SUBDIR="tb2_eval_only_qwen35_35b_a3b"
    export WANDB_GROUP="${WANDB_GROUP:-tb2-eval-only-qwen35-35b-a3b-bs512-n1-tmax-timeouts}"
    ;;
  35b-a3b-tp8|35b-tp8|35b-a3b-hf-tp8|35b-a3b-official-v2|35b-a3b-thinking-v3|35b-a3b-coding-v4)
    if [ "${MODEL_KIND}" = "35b-a3b-hf-tp8" ]; then
      export HF_CKPT=/path/to/shared_storage/hf_models/Qwen/Qwen3.5-35B-A3B_hf
      export STAGED_HF_CKPT=/tmp/slime-harbor-model-cache/Qwen3.5-35B-A3B_hf
      export SAVE_SUBDIR="tb2_eval_only_qwen35_35b_a3b_hf_tp8"
      export WANDB_GROUP="${WANDB_GROUP:-tb2-eval-only-qwen35-35b-a3b-hf-tp8-bs512-n1}"
    else
      export HF_CKPT=/path/to/shared_storage/hf_models/Qwen/Qwen3.5-35B-A3B
      export STAGED_HF_CKPT=/tmp/slime-harbor-model-cache/Qwen3.5-35B-A3B
      export SAVE_SUBDIR="tb2_eval_only_qwen35_35b_a3b_tp8"
      export WANDB_GROUP="${WANDB_GROUP:-tb2-eval-only-qwen35-35b-a3b-tp8-bs512-n1}"
    fi
    if [ "${MODEL_KIND}" = "35b-a3b-official-v2" ]; then
      export SAVE_SUBDIR="tb2_eval_only_qwen35_35b_a3b_official_v2"
      export WANDB_GROUP="${WANDB_GROUP:-tb2-eval-only-qwen35-35b-a3b-official-v2}"
      export ROLLOUT_MAX_CONTEXT_LEN=131072
      export SGLANG_CONTEXT_LENGTH=131072
      export SGLANG_MAX_PREFILL_TOKENS=131072
      export ROLLOUT_MAX_RESPONSE_LEN=8192
    fi
    if [ "${MODEL_KIND}" = "35b-a3b-thinking-v3" ]; then
      export SAVE_SUBDIR="tb2_eval_only_qwen35_35b_a3b_thinking_v3"
      export WANDB_GROUP="${WANDB_GROUP:-tb2-eval-only-qwen35-35b-a3b-thinking-v3}"
      export ROLLOUT_MAX_CONTEXT_LEN=131072
      export SGLANG_CONTEXT_LENGTH=131072
      export SGLANG_MAX_PREFILL_TOKENS=131072
      export ROLLOUT_MAX_RESPONSE_LEN=8192
    fi
    if [ "${MODEL_KIND}" = "35b-a3b-coding-v4" ]; then
      export SAVE_SUBDIR="tb2_eval_only_qwen35_35b_a3b_coding_v4"
      export WANDB_GROUP="${WANDB_GROUP:-tb2-eval-only-qwen35-35b-a3b-coding-v4}"
    fi
    export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/shared_storage/hf_models/Qwen/Qwen3.5-35B-A3B_torch_dist}"
    export MODEL_SCRIPT="${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"
    export ROLLOUT_NUM_GPUS_PER_ENGINE=8
    export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
    export EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:---sglang-mamba-scheduler-strategy extra_buffer}"
    ;;
  122b-a10b|122b)
    # Qwen3.5-122B-A10B (MoE, ~122B total / ~10B active). Same qwen3_5_moe family
    # as 35B-A3B but larger (hidden 3072, 48 layers, moe/shared inter 1024). Served
    # with the proven-stable TP8 topology: 7 SGLang engines x TP8 = 56 GPU, no
    # dp-attention / EP split (the TP8 config that fixed the 35B single-request
    # batching problem). Bridge mode => SGLang loads the HF checkpoint directly,
    # so no pre-converted torch_dist is needed for rollout-only eval.
    export HF_CKPT=/path/to/shared_storage/hf_models/Qwen/Qwen3.5-122B-A10B
    export STAGED_HF_CKPT=/tmp/slime-harbor-model-cache/Qwen3.5-122B-A10B
    # 245GB of weights: staging to node-local /tmp on every node would need ~245GB
    # free per node and a long rsync. Read from the shared fs instead. Forced OFF
    # (not :-0) because line ~40 already exported STAGE_HF_CKPT_TO_TMP=1 as the
    # global default, so a :-0 fallback here would never win.
    export STAGE_HF_CKPT_TO_TMP=0
    export MODEL_SCRIPT="${SLIME_DIR}/scripts/models/qwen3.5-122B-A10B.sh"
    export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
    export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
    # A10B active params > 35B-A3B; give the KV/graph pool a bit less static frac room.
    export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
    export EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:---sglang-mamba-scheduler-strategy extra_buffer --sglang-disable-custom-all-reduce}"
    export SAVE_SUBDIR="tb2_eval_only_qwen35_122b_a10b_tp8"
    export WANDB_GROUP="${WANDB_GROUP:-tb2-eval-only-qwen35-122b-a10b-tp8-bs512}"
    ;;
  *) echo "unsupported MODEL_KIND" >&2; exit 2;;
esac
export BASE_FOLDER="${BASE_FOLDER:-${SLIME_DIR}/checkpoints/eval_only/${MODEL_KIND}}"
mkdir -p "${BASE_FOLDER}"
exec "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
