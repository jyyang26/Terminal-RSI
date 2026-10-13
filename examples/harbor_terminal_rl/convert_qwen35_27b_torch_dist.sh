#!/bin/bash

set -euo pipefail

# Offline conversion for Qwen3.5-27B HF -> Megatron torch_dist. The output is
# stored on gy2 so every node can read it; do not write model weights to /cq_1.

CODE_ROOT=${CODE_ROOT:-/path/to/workspace/code}
SLIME_DIR=${SLIME_DIR:-${CODE_ROOT}/slime}
HF_CKPT=${HF_CKPT:-/path/to/models/Qwen3.5-27B}
SAVE_DIR=${SAVE_DIR:-/path/to/models/Qwen3.5-27B_torch_dist_pp8_timeout60}
MODEL_SCRIPT=${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-27B.sh}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-12356}
LOG_FILE=${LOG_FILE:-${SAVE_DIR%/}.convert.log}

# Use PP-only conversion on one 8-GPU node. The upstream 27B examples use this
# layout because TP conversion can stall in mbridge tensor-parallel scatter for
# Qwen3.5 GatedDeltaNet. Megatron dist checkpoints can be loaded later with the
# training TP/PP layout.
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
