#!/usr/bin/env bash
# convert_q38_27b_base_torch_dist.sh
#
# Offline HF -> Megatron torch_dist conversion of the BASE model
#   /path/to/models/Qwen3.8-27B
# so it can be used as --load for a fresh PPO run.
#
# ============================ WHY THIS STEP EXISTS ============================
# The requested start path is a HuggingFace directory: 18 *.safetensors shards +
# config.json, with NO latest_checkpointed_iteration.txt and no *.distcp. Megatron's
# --load goes through the torch_dist reader (checkpointing.py:_is_megatron_checkpoint),
# so pointing --load straight at the HF dir cannot work -- and the runtime HF "bridge"
# load path can fail for pipeline-parallel loading of this model family.
# Conversion resources and parallelism are provided by the caller.
# A torch_dist checkpoint can be resharded into a compatible training layout.
# Tensor-parallel conversion can stall in mbridge's scatter for GatedDeltaNet;
# validate the chosen conversion layout against the model and backend.
#
# ======================== THE BASE IS A VLM CHECKPOINT ========================
# config.json says architectures=[Qwen3_5ForConditionalGeneration], model_type=qwen3_5,
# with the text tower under text_config and weights named model.language_model.*:
#   model.language_model.*  850 keys   <- the only ones we want
#   model.visual.*          333 keys   <- vision tower, no Megatron counterpart
#   mtp.*                    15 keys   <- multi-token-prediction head, not in MODEL_ARGS
#   lm_head.weight            1 key
# This is NOT a problem and needs no pre-stripping:
#   * slime_plugins/mbridge/qwen3_5.py registers model_type "qwen3_5" and its mapping
#     table is written against the model.language_model.* prefix (lines 19-89).
#     Verified: AutoBridge.from_pretrained resolves to Qwen3_5Bridge for this exact
#     config (it then only fails on "tensor model parallel group is not initialized",
#     i.e. the registry lookup succeeded and just needs torchrun).
#   * The mapping runs mcore -> hf, i.e. it PULLS the keys Megatron asks for. Extra HF
#     keys (visual, mtp) are simply never requested. scripts/models/qwen3.5-27B.sh
#     declares 64 dense layers and no MTP, so nothing asks for them.
# Output includes decoder, embedding and output-layer weights, with no visual,
# MTP or optimizer entries. RNG sharding depends on the configured resources.
#
# ===================== THE OUTPUT TRACKER SAYS "release" =====================
# convert_hf_to_torch_dist.py writes iteration "release". A fresh training run
# uses this tracker to start at rollout 0; a numeric tracker instead resumes at
# loaded_iteration + 1. The output directory can be used directly as --load.
#
# ======================== NO _critic SIBLING MAY EXIST ========================
# A cold critic requires the absence of "<load>_critic"; otherwise the critic
# would silently load the existing value head. This script checks the sibling.
#
# Allocate conversion resources separately from active training jobs.
#
# Usage:
#   bash examples/harbor_terminal_rl/convert_q38_27b_base_torch_dist.sh

set -euo pipefail

USER_ROOT=${USER_ROOT:-/path/to/workspace}
SLIME_DIR=${SLIME_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
HF_CKPT=${HF_CKPT:-/path/to/models/Qwen3.8-27B}
SAVE_DIR=${SAVE_DIR:-${USER_ROOT}/checkpoints/q38_27b_base_torch_dist}
MASTER_PORT=${MASTER_PORT:-12377}
LOG_FILE=${LOG_FILE:-${SAVE_DIR%/}.convert.log}
DISTRIBUTED_TIMEOUT_MINUTES=${DISTRIBUTED_TIMEOUT_MINUTES:-60}

for name in MODEL_SCRIPT NPROC_PER_NODE TENSOR_MODEL_PARALLEL_SIZE PIPELINE_MODEL_PARALLEL_SIZE; do
  [ -n "${!name:-}" ] || { echo "[convert] ERROR: set ${name} for your environment" >&2; exit 1; }
done
for name in NPROC_PER_NODE TENSOR_MODEL_PARALLEL_SIZE PIPELINE_MODEL_PARALLEL_SIZE; do
  [[ "${!name}" =~ ^[1-9][0-9]{0,5}$ ]] || { echo "[convert] ERROR: ${name} must be a positive integer" >&2; exit 1; }
done
(( NPROC_PER_NODE == TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE )) \
  || { echo "[convert] ERROR: conversion process count must match the parallel layout" >&2; exit 1; }

test -d "${SLIME_DIR}"
test -d "${HF_CKPT}"
test -f "${MODEL_SCRIPT}"
test -f "${HF_CKPT}/model.safetensors.index.json"

# A stale partial output would be silently reused by --load; refuse instead.
if [ -e "${SAVE_DIR}" ]; then
  if [ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "[convert] ${SAVE_DIR} already converted (tracker=$(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt")) -- nothing to do."
    exit 0
  fi
  echo "[convert] FATAL: ${SAVE_DIR} exists but has no tracker file (partial conversion?). Remove it by hand." >&2
  exit 1
fi
# Cold critic requires the absence of this sibling.
[ -e "${SAVE_DIR}_critic" ] && {
  echo "[convert] FATAL: ${SAVE_DIR}_critic exists -> the critic would WARM-load. Cold start needs it absent." >&2
  exit 1
}

mkdir -p "$(dirname "${SAVE_DIR}")"

cd "${SLIME_DIR}"
# shellcheck disable=SC1090
source "${MODEL_SCRIPT}"

export PYTHONPATH="/root/Megatron-LM:${SLIME_DIR}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

echo "[convert] hf     = ${HF_CKPT}"
echo "[convert] save   = ${SAVE_DIR}"
echo "[convert] log    = ${LOG_FILE}"

torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}" \
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}" \
  --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}" \
  --hf-checkpoint "${HF_CKPT}" \
  --save "${SAVE_DIR}" \
  2>&1 | tee "${LOG_FILE}"

# ---- verification -------------------------------------------------------------------
test -f "${SAVE_DIR}/latest_checkpointed_iteration.txt"
TRACKER=$(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt")
echo "[convert] tracker = '${TRACKER}'"
[ "${TRACKER}" = "release" ] || echo "[convert] WARNING: tracker is not 'release' -> start_rollout_id will NOT auto-reset to 0; set START_ROLLOUT_ID=0 explicitly."

python3 - "${SAVE_DIR}/${TRACKER}" <<'PYEOF'
import sys, collections
import torch.distributed.checkpoint as dcp
p = sys.argv[1]
md = dcp.FileSystemReader(p).read_metadata()
ks = sorted(md.state_dict_metadata.keys())
top = collections.Counter(k.split('.')[0] for k in ks)
print("[verify] entries:", len(ks))
print("[verify] top-level:", dict(top))
for pat in ("visual", "mtp", "optimizer"):
    n = sum(1 for k in ks if pat in k)
    print(f"[verify] {pat:10s} -> {n} (expected 0)")
for want in ("embedding.word_embeddings.weight", "output_layer.weight"):
    t = md.state_dict_metadata.get(want)
    print(f"[verify] {want} -> {getattr(t, 'size', None)}")
    assert t is not None, f"missing {want}"
print("[verify] OK")
PYEOF

echo "=== CONVERT DONE ==="
ls -la "${SAVE_DIR}/"
du -sh "${SAVE_DIR}/"
