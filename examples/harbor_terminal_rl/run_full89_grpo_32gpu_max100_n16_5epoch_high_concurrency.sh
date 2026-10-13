#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# 89 tasks * 5 epochs = 445 prompt groups. With rollout_batch_size=16 and
# start_rollout_id=1, NUM_ROLLOUT=29 trains rollout ids 1..28, covering
# 448 prompt groups. n=16 gives 256 trajectories per rollout.
#
# Higher-concurrency retry for the replacement 32-H20 job: one actor node and
# three rollout nodes. Each rollout node can host up to 128 Apptainer trials,
# so the global env concurrency is 384 instead of the previous 192.
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_terminal_bench2_full89_32gpu_max100_5epoch_high_concurrency_last_turn.yaml}"
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_32gpu_max100_n16_5epoch_resp4096_ctx128k_rb16_os32_conc384_pn128}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-9B_terminal_bench2_full89_32gpu_max100_n16_5epoch_resp4096_ctx128k_rb16_os32_conc384_pn128}"

export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-32}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-29}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"

export WANDB_GROUP="${WANDB_GROUP:-qwen35-9b-tb2-full89-32gpu-max100-n16-5epoch-resp4096-ctx128k-rb16-os32-conc384-pn128}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray9bf89m100c384}"

exec "${SCRIPT_DIR}/run_full89_grpo_32gpu_max100_n16_5epoch.sh"
