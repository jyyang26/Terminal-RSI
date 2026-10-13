#!/bin/bash
# Convert Qwen3.5-35B-A3B (MoE + hybrid linear attention) HF checkpoint -> Megatron torch_dist.
# Runs locally on 8x H20. World size = 8 -> 8-way PP (40 layers, satisfies world_size <= num_layers).
# Output is the --ref-load dir consumed by training (e.g. examples/harbor_terminal_rl / SFT scripts).
set -ex

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SLIME_DIR}"

source "${SCRIPT_DIR}/models/qwen3.5-35B-A3B.sh"

BASE_FOLDER="${BASE_FOLDER:-/path/to/models}"
HF_CKPT="${BASE_FOLDER}/Qwen3.5-35B-A3B"
SAVE_DIR="${BASE_FOLDER}/Qwen3.5-35B-A3B_torch_dist"

export PYTHONPATH="/root/Megatron-LM:${PYTHONPATH}"

PYTHONPATH="/root/Megatron-LM:${PYTHONPATH}" torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint "${HF_CKPT}" \
   --save "${SAVE_DIR}"

echo "=== conversion done -> ${SAVE_DIR} ==="
