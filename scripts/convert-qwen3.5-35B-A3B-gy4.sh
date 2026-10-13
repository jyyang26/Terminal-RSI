#!/bin/bash
# Convert Qwen3.5-35B-A3B (MoE + hybrid linear attention) HF checkpoint -> Megatron torch_dist,
# using the model + save paths on the shared-storage share (this environment).
#
# Runs locally on 8x H20. World size = 8 -> 8-way PP (40 layers, satisfies world_size <= num_layers).
# Output is the --ref-load / initial --load dir consumed by the 56-GPU training wrapper
# (examples/harbor_terminal_rl/run_35b_a3b_56gpu_daytona_..._fully_async.sh).
#
# Usage (single 8-GPU node):
#   bash scripts/convert-qwen3.5-35B-A3B-gy4.sh
set -ex

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SLIME_DIR}"

source "${SCRIPT_DIR}/models/qwen3.5-35B-A3B.sh"

# gy4-share paths (override via env if your layout differs).
HF_CKPT="${HF_CKPT:-/path/to/shared_storage/hf_models/Qwen/Qwen3.5-35B-A3B}"
SAVE_DIR="${SAVE_DIR:-/path/to/shared_storage/hf_models/Qwen/Qwen3.5-35B-A3B_torch_dist}"

export PYTHONPATH="/root/Megatron-LM:${SLIME_DIR}:${PYTHONPATH:-}"

PYTHONPATH="${PYTHONPATH}" torchrun --nproc-per-node "${CONVERT_NPROC:-8}" \
   tools/convert_hf_to_torch_dist.py \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint "${HF_CKPT}" \
   --save "${SAVE_DIR}"

echo "=== conversion done -> ${SAVE_DIR} ==="
