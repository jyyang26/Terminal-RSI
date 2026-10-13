#!/bin/bash

set -ex

# Single-node Qwen3.5-4B + Harbor Terminus rollout.
# Uses one 8-GPU node with colocated Megatron actor and SGLang rollout engines.

export PYTHONBUFFERED=16
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

CODE_ROOT=${CODE_ROOT:-/path/to/workspace/code}
SLIME_DIR=${SLIME_DIR:-${CODE_ROOT}/slime}
HARBOR_SRC=${HARBOR_SRC:-${CODE_ROOT}/harbor/src}
BASE_FOLDER=${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_4b}
HF_CKPT=${HF_CKPT:-/path/to/models/Qwen3.5-4B}
CONFIG=${CONFIG:-${SLIME_DIR}/examples/harbor_terminal_rl/harbor_slime_config.yaml}
MODEL_SCRIPT=${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-4B.sh}
MEGATRON_TO_HF_MODE=${MEGATRON_TO_HF_MODE-bridge}
SAVE_SUBDIR=${SAVE_SUBDIR:-Qwen3.5-4B_harbor_slime}

MASTER_ADDR=${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
CP_SIZE=${CP_SIZE:-1}
SOCKET_IFNAME=${SOCKET_IFNAME:-bond1}

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-1}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2}
NUM_ROLLOUT=${NUM_ROLLOUT:-2}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-2048}
CLEAN_START=${CLEAN_START:-1}
DEBUG_ROLLOUT_ONLY=${DEBUG_ROLLOUT_ONLY:-0}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.5}
QKV_FORMAT=${QKV_FORMAT-bshd}

if [ "${CLEAN_START}" = "1" ]; then
  pkill -9 sglang || true
  ray stop --force || true
  pkill -9 -f "raylet|gcs_server|dashboard_agent|runtime_env_agent|log_monitor" || true
  pkill -9 -f "python3 train.py|python train.py" || true
  sleep 3
fi

test -d "${SLIME_DIR}"
test -d "${HARBOR_SRC}"
test -d "${HF_CKPT}"
test -f "${CONFIG}"
test -f "${MODEL_SCRIPT}"
mkdir -p "${BASE_FOLDER}/rollout_dumps"
SAVE_DIR="${BASE_FOLDER%/}/${SAVE_SUBDIR}"

if ! command -v singularity >/dev/null 2>&1; then
  if command -v apptainer >/dev/null 2>&1; then
    mkdir -p /tmp/slime-harbor-bin
    ln -sf "$(command -v apptainer)" /tmp/slime-harbor-bin/singularity
    export PATH=/tmp/slime-harbor-bin:${PATH}
  else
    echo "WARNING: neither singularity nor apptainer is in PATH; Harbor singularity tasks will fail."
  fi
fi

cd "${SLIME_DIR}"
source "${MODEL_SCRIPT}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --save "${SAVE_DIR}/"
   --save-interval 20
)
if [ -n "${MEGATRON_TO_HF_MODE}" ]; then
   CKPT_ARGS+=(--megatron-to-hf-mode "${MEGATRON_TO_HF_MODE}")
fi
if [ -d "${SAVE_DIR}/" ] && [ -n "$(ls -A "${SAVE_DIR}/" 2>/dev/null)" ]; then
   CKPT_ARGS+=(--load "${SAVE_DIR}/")
fi

HARBOR_ROLLOUT_ARGS=(
   --rollout-function-path slime_plugins.harbor.rollout.generate_rollout
   --data-source-path slime_plugins.harbor.data_source.HarborTaskDataSource
   --custom-config-path "${CONFIG}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature 1.0
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
   --skip-eval-before-train
)

DEBUG_DUMP_ARGS=(
   --save-debug-rollout-data "${BASE_FOLDER}/rollout_dumps/{rollout_id}.pt"
)
if [ "${DEBUG_ROLLOUT_ONLY}" = "1" ]; then
   DEBUG_DUMP_ARGS+=(--debug-rollout-only)
fi

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CP_SIZE}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --calculate-per-token-loss
   --micro-batch-size 1
)
if [ -n "${QKV_FORMAT}" ]; then
   PERF_ARGS+=(--qkv-format "${QKV_FORMAT}")
fi

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export PYTHONPATH="${HARBOR_SRC}:${SLIME_DIR}:${PYTHONPATH:-}"
export no_proxy="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"

ray start --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
    "GLOO_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "NCCL_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": "/root/Megatron-LM/:${HARBOR_SRC}:${SLIME_DIR}:${PYTHONPATH:-}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "SLIME_SKIP_NUMA_AFFINITY": "1"
  }
}
EOF_JSON
)

for i in $(seq 1 60); do
  if curl -sf --max-time 3 "http://127.0.0.1:8265/api/jobs/" >/dev/null 2>&1; then
    echo "ray job agent ready after ${i}s"
    break
  fi
  sleep 1
done
sleep 10

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${HARBOR_ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${DEBUG_DUMP_ARGS[@]}"
