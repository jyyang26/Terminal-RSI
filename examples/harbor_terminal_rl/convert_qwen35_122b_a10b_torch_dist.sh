#!/bin/bash

set -euo pipefail

# Offline conversion Qwen3.5-122B-A10B HF -> Megatron torch_dist, on ONE 8-GPU node.
# Cloned from convert_qwen35_27b_torch_dist.sh. Output goes to the shared gy4 hf_models
# dir so every training node can read it. This is needed because the RUNTIME bridge
# path (megatron.bridge in slime/backends/megatron_utils/checkpoint.py) fails for this
# 122B model at PP>1 (AttributeError on output_layer). The OFFLINE convert here uses
# slime_plugins.mbridge (Qwen3_5Bridge), which DOES map output_layer.weight->lm_head,
# producing a torch_dist that the training run can then load with MEGATRON_TO_HF_MODE=raw.
#
# TP1 x PP8 single-node convert; the resulting dist checkpoint is re-loadable at any
# training TP/PP/EP layout (e.g. the training run's TP2 x PP2 x EP16).

SLIME_DIR=${SLIME_DIR:-/path/to/workspace/slime-terminal-RL}
HF_CKPT=${HF_CKPT:-/path/to/shared_storage/hf_models/Qwen/Qwen3.5-122B-A10B}
SAVE_DIR=${SAVE_DIR:-/path/to/shared_storage/hf_models/Qwen/Qwen3.5-122B-A10B_torch_dist}
MODEL_SCRIPT=${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-122B-A10B.sh}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-12377}
LOG_FILE=${LOG_FILE:-${SAVE_DIR%/}.convert.log}

# PP-only conversion on one 8-GPU node (48 layers / 8 = 6 per stage). TP conversion can
# stall in mbridge tensor-parallel scatter for Qwen3.5 GatedDeltaNet, so keep TP1.
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-8}
DECODER_LAST_PIPELINE_NUM_LAYERS=${DECODER_LAST_PIPELINE_NUM_LAYERS:-}
DISTRIBUTED_TIMEOUT_MINUTES=${DISTRIBUTED_TIMEOUT_MINUTES:-60}

test -d "${SLIME_DIR}"
test -d "${HF_CKPT}"
test -f "${MODEL_SCRIPT}"
mkdir -p "$(dirname "${SAVE_DIR}")"
mkdir -p "$(dirname "${LOG_FILE}")"

cd "${SLIME_DIR}"
source "${MODEL_SCRIPT}"

export PYTHONPATH="/root/Megatron-LM:${SLIME_DIR}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

CONVERT_ARGS=(
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
  --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}"
)
if [ -n "${DECODER_LAST_PIPELINE_NUM_LAYERS}" ]; then
  CONVERT_ARGS+=(--decoder-last-pipeline-num-layers "${DECODER_LAST_PIPELINE_NUM_LAYERS}")
fi

torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  "${CONVERT_ARGS[@]}" \
  --hf-checkpoint "${HF_CKPT}" \
  --save "${SAVE_DIR}" \
  2>&1 | tee "${LOG_FILE}"

test -f "${SAVE_DIR}/latest_checkpointed_iteration.txt"
echo "=== CONVERT DONE ==="
ls -la "${SAVE_DIR}/"
du -sh "${SAVE_DIR}/"
