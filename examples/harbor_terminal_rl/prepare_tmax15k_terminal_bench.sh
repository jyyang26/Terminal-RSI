#!/bin/bash
#
# prepare_tmax15k_terminal_bench.sh
# ---------------------------------------------------------------------------
# One-shot data prep: convert the allenai/tmax TMax-15k dataset (tasks.zip, 14601
# task_*/ dirs in allenai/tmax native format) into harbor terminal-bench task dirs
# (one <task>/task.toml each), consumable by the harbor rollout via
#   HARBOR_TASKS_DIR=<OUT_DIR>   (glob "*/task.toml")
#
# Wraps convert_tmax15k_to_terminal_bench.py. After conversion it VERIFIES the
# produced task.toml count and prints the HARBOR_TASKS_DIR to export.
#
# Usage:
#   bash prepare_tmax15k_terminal_bench.sh                 # full 14601 conversion
#   LIMIT=20 bash prepare_tmax15k_terminal_bench.sh        # smoke: first 20 tasks
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

SRC_ZIP="${SRC_ZIP:-/path/to/workspace/TMax-15k/tasks.zip}"
OUT_DIR="${OUT_DIR:-/path/to/workspace/slime-terminal-RL_data/tmax15k_terminal_bench}"
UNZIP_DIR="${UNZIP_DIR:-/tmp/tmax15k_src_unzip}"
WORKERS="${WORKERS:-32}"
LIMIT="${LIMIT:-0}"
EXPECTED_COUNT="${EXPECTED_COUNT:-14601}"

echo "=============================================================="
echo " TMax-15k -> harbor terminal-bench conversion"
echo "   src zip : ${SRC_ZIP}"
echo "   out dir : ${OUT_DIR}"
echo "   workers : ${WORKERS}   limit: ${LIMIT}"
echo "=============================================================="

test -f "${SRC_ZIP}"
mkdir -p "${OUT_DIR}"

LIMIT_ARG=()
if [ "${LIMIT}" -gt 0 ] 2>/dev/null; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

python3 "${SCRIPT_DIR}/convert_tmax15k_to_terminal_bench.py" \
  --src-zip "${SRC_ZIP}" \
  --out-dir "${OUT_DIR}" \
  --unzip-dir "${UNZIP_DIR}" \
  --workers "${WORKERS}" \
  "${LIMIT_ARG[@]}"

PRODUCED="$(find "${OUT_DIR}" -maxdepth 2 -name task.toml 2>/dev/null | wc -l)"
echo "--------------------------------------------------------------"
echo "[verify] produced task.toml: ${PRODUCED}"
if [ "${LIMIT}" -eq 0 ] 2>/dev/null && [ "${PRODUCED}" -ne "${EXPECTED_COUNT}" ]; then
  echo "[WARN] expected ${EXPECTED_COUNT} tasks but produced ${PRODUCED}; check the [failures] list above." >&2
fi
echo ""
echo "DONE. Point the run at this dir:"
echo "    export HARBOR_TASKS_DIR=${OUT_DIR}"
echo "(eval set stays terminal-bench-2-main; only training tasks change.)"
