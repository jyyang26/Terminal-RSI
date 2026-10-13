#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# 89 tasks * 5 epochs = 445 prompt groups. With rollout_batch_size=16 and
# start_rollout_id=1, NUM_ROLLOUT=29 trains rollout ids 1..28, covering
# 448 prompt groups. n=16 gives 256 trajectories per rollout.
#
# 32-H20 Qwen3.5-9B run with turn_grouped samples: every Terminus LLM turn is
# trained, while GRPO grouping and reward normalization remain trajectory-level.
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_terminal_bench2_full89_32gpu_max100_5epoch_high_concurrency_turn_grouped.yaml}"
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_32gpu_max100_n16_5epoch_resp4096_ctx128k_rb16_os32_conc384_pn128_turngrouped}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-9B_terminal_bench2_full89_32gpu_max100_n16_5epoch_resp4096_ctx128k_rb16_os32_conc384_pn128_turngrouped}"

export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-32}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-29}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"

export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.partial_reward.trajectory_grouped_progress_rewards}"
export HARBOR_PARTIAL_REWARD_MAX="${HARBOR_PARTIAL_REWARD_MAX:-0.4}"
export USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-1}"
export KEEP_OLD_ACTOR="${KEEP_OLD_ACTOR:-1}"

export WANDB_GROUP="${WANDB_GROUP:-qwen35-9b-tb2-full89-32gpu-turngrouped-logp-max100-n16-5epoch-r4096-ctx128k-rb16-os32-conc384-pn128}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray9bf89m100turn}"

exec "${SCRIPT_DIR}/run_full89_grpo_32gpu_max100_n16_5epoch.sh"
