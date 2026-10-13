#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export LOG_FILE="${LOG_FILE:-/tmp/slime_full89_64gpu_27b_max100_high_concurrency_lastturn_monitor.log}"
export LAUNCHER_LOG="${LAUNCHER_LOG:-/tmp/slime_full89_64gpu_27b_max100_high_concurrency_lastturn_launcher.log}"
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_64gpu_max100_n16_5epoch_resp4096_ctx128k_actor4_tp2pp8cp2_thd_rb16_os32_conc96_pn24_traincap128k_lpc32_logp_lastturn}"
export ROLLOUT_DUMPS_DIR="${ROLLOUT_DUMPS_DIR:-${BASE_FOLDER}/rollout_dumps}"
export SAVE_DIR="${SAVE_DIR:-${BASE_FOLDER}/Qwen3.5-27B_terminal_bench2_full89_64gpu_max100_n16_5epoch_resp4096_ctx128k_actor4_tp2pp8cp2_thd_rb16_os32_conc96_pn24_traincap128k_lpc32_logp_lastturn}"
export TRIALS_DIR="${TRIALS_DIR:-/path/to/workspace/slime_harbor_trials/tb2_full89_64gpu_27b_max100_resp4096_ctx128k_actor4_tp2pp8cp2_thd_rb16_os32_conc96_pn24_traincap128k_lpc32_logp_lastturn}"

exec "${SCRIPT_DIR}/monitor_full89_64gpu_27b_max100.sh"
