#!/bin/bash

set -euo pipefail

# Offline conversion Qwen3.5-122B-A10B HF -> Megatron torch_dist.
# The runtime bridge can fail for pipeline-parallel loading of this model.
# The offline Qwen3_5Bridge maps output_layer.weight to lm_head.
# Conversion resources and parallelism are provided by the caller.

SLIME_DIR=${SLIME_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
HF_CKPT=${HF_CKPT:-/path/to/Qwen3.5-122B-A10B}
SAVE_DIR=${SAVE_DIR:-${HF_CKPT%/}_torch_dist}
MASTER_PORT=${MASTER_PORT:-12377}
LOG_FILE=${LOG_FILE:-${SAVE_DIR%/}.convert.log}

for name in MODEL_SCRIPT NPROC_PER_NODE TENSOR_MODEL_PARALLEL_SIZE PIPELINE_MODEL_PARALLEL_SIZE; do
  [ -n "${!name:-}" ] || { echo "[convert] ERROR: set ${name} for your environment" >&2; exit 1; }
done
for name in NPROC_PER_NODE TENSOR_MODEL_PARALLEL_SIZE PIPELINE_MODEL_PARALLEL_SIZE; do
  [[ "${!name}" =~ ^[1-9][0-9]{0,5}$ ]] || { echo "[convert] ERROR: ${name} must be a positive integer" >&2; exit 1; }
done
(( NPROC_PER_NODE == TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE )) \
  || { echo "[convert] ERROR: conversion process count must match the parallel layout" >&2; exit 1; }
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
