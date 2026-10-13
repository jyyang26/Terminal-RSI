#!/bin/bash
# Convert Qwen3.6-27B HF safetensors -> Megatron torch_dist checkpoint, for use as the
# actor cold-start --ref-load in the synth-2820 PPO run. Qwen3.6-27B has the SAME
# architecture as 3.5-27B (model_type qwen3_5_text: 64L/5120/24h/4kv/17408/vocab248320/
# head_dim256), so it reuses scripts/models/qwen3.5-27B.sh unchanged. Mirrors
# scripts/convert-qwen3.5-35B-A3B.sh. Run on an idle 8-GPU node (~30-60min).
set -ex
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SLIME_DIR}"

source "${SCRIPT_DIR}/models/qwen3.5-27B.sh"

HF_CKPT="${HF_CKPT:-/path/to/models/Qwen3.6-27B}"
SAVE_DIR="${SAVE_DIR:-/path/to/models/Qwen3.6-27B_torch_dist}"

export PYTHONPATH="/root/Megatron-LM:${PYTHONPATH}"
PYTHONPATH="/root/Megatron-LM:${PYTHONPATH}" torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint "${HF_CKPT}" \
   --save "${SAVE_DIR}"

echo "=== conversion done -> ${SAVE_DIR} ==="
