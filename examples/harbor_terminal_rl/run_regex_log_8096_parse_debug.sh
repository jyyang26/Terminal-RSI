#!/bin/bash

set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/path/to/workspace/code}
SLIME_DIR=${SLIME_DIR:-${CODE_ROOT}/slime}

export MASTER_ADDR=${MASTER_ADDR:-10.0.0.4}
export RAY_NODE_IP_PREFIX=${RAY_NODE_IP_PREFIX:-29.68}
export EXPECTED_NODES=${EXPECTED_NODES:-8}

export CONFIG=${CONFIG:-${SLIME_DIR}/examples/harbor_terminal_rl/harbor_terminal_bench2_config_gy2_sif.yaml}
export BASE_FOLDER=${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_9b_tb2_regex_log_8096_parse_debug}
export HF_CKPT=${HF_CKPT:-/path/to/models/Qwen3.5-9B}
export PRETRAINED_CKPT=${PRETRAINED_CKPT:-/path/to/models/Qwen3.5-9B_torch_dist}

export DEBUG_ROLLOUT_ONLY=${DEBUG_ROLLOUT_ONLY:-1}
export NUM_ROLLOUT=${NUM_ROLLOUT:-1}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-1}
export N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1}
export ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-8096}

export ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
export ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
export ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-56}
export ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
export SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.50}
export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}

export STAGE_HF_CKPT_TO_TMP=${STAGE_HF_CKPT_TO_TMP:-1}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-/tmp/slime-harbor-apptainer-cache}

exec "${SLIME_DIR}/examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh"
