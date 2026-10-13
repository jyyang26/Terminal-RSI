#!/bin/bash

set -ex

# Offline conversion for Qwen3.5-9B HF -> Megatron torch_dist.
# Run this on a GPU node before training. The output path should be visible to
# the actor node(s) used by the training job.

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

CODE_ROOT=${CODE_ROOT:-/path/to/workspace/code}
SLIME_DIR=${SLIME_DIR:-${CODE_ROOT}/slime}
HF_CKPT=${HF_CKPT:-/path/to/models/Qwen3.5-9B}
SAVE_DIR=${SAVE_DIR:-/path/to/models/Qwen3.5-9B_torch_dist}
MODEL_SCRIPT=${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-9B.sh}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-12355}
LOG_FILE=${LOG_FILE:-${SAVE_DIR%/}.convert.log}

test -d "${SLIME_DIR}"
test -d "${HF_CKPT}"
test -f "${MODEL_SCRIPT}"
mkdir -p "$(dirname "${SAVE_DIR}")"
mkdir -p "$(dirname "${LOG_FILE}")"

cd "${SLIME_DIR}"
source "${MODEL_SCRIPT}"

export PYTHONPATH="/root/Megatron-LM:${SLIME_DIR}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CKPT}" \
  --save "${SAVE_DIR}" \
  2>&1 | tee "${LOG_FILE}"

test -f "${SAVE_DIR}/latest_checkpointed_iteration.txt"
echo "=== CONVERT DONE ==="
ls -la "${SAVE_DIR}/"
du -sh "${SAVE_DIR}/"
