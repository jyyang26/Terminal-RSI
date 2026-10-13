#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export LOG_FILE="${LOG_FILE:-/tmp/slime_full89_32gpu_max100_5epoch_high_concurrency_monitor.log}"
export LAUNCHER_LOG="${LAUNCHER_LOG:-/tmp/slime_full89_32gpu_max100_5epoch_high_concurrency_launcher.log}"
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_32gpu_max100_n16_5epoch_resp4096_ctx128k_rb16_os32_conc384_pn128}"
export SAVE_DIR="${SAVE_DIR:-${BASE_FOLDER}/Qwen3.5-9B_terminal_bench2_full89_32gpu_max100_n16_5epoch_resp4096_ctx128k_rb16_os32_conc384_pn128}"
export TRIALS_DIR="${TRIALS_DIR:-/path/to/workspace/slime_harbor_trials/tb2_full89_32gpu_max100_5epoch_resp4096_ctx128k_rb16_os32_conc384_pn128}"

exec "${SCRIPT_DIR}/monitor_full89_32gpu_max100_5epoch.sh"
