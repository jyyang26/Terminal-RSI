#!/bin/bash

set -e

# Tiny non-debug training smoke for a 4x8 H20 job.
# The base launcher still starts Ray on all 4 nodes, but the slime placement
# group is 8 actor GPUs + 24 rollout GPUs = 32 GPUs total.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_terminal_bench2_full89_multinode_train_smoke.yaml}"
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_9b_tb2_train_smoke_step}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-train_smoke_32gpu_actor8_rollout24}"

export EXPECTED_NODES="${EXPECTED_NODES:-4}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-24}"

export DEBUG_ROLLOUT_ONLY="${DEBUG_ROLLOUT_ONLY:-0}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
# train_async.py loops over range(start_rollout_id, num_rollout). A fresh
# torch_dist checkpoint starts at rollout_id=1, so NUM_ROLLOUT=2 gives exactly
# one train step while still keeping this smoke short.
export NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"

export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-512}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-4096}"
export SEQ_LENGTH="${SEQ_LENGTH:-4096}"

export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.50}"

export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-8}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-0}"
export CP_SIZE="${CP_SIZE:-1}"
export QKV_FORMAT="${QKV_FORMAT:-thd}"

export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-0}"
export USE_WANDB="${USE_WANDB:-0}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/slime-harbor-ray-train-smoke}"

exec bash "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
