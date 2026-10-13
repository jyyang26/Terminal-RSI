#!/bin/bash

set -euo pipefail

TAR_PATH="${TB2_TRAIN_V2_TAR_PATH:-/path/to/workspace/terminal-bench-2-train-v2.tar.gz}"
DEST_ROOT="${TB2_TRAIN_V2_DEST_ROOT:-/tmp/slime-harbor-tasks}"
DEST_DIR="${TB2_TRAIN_V2_DEST_DIR:-${DEST_ROOT}/terminal-bench-2-train-v2}"
LOCAL_TAR="${TB2_TRAIN_V2_LOCAL_TAR:-${DEST_ROOT}/terminal-bench-2-train-v2.tar.gz}"
EXPECTED_TASKS="${TB2_TRAIN_V2_EXPECTED_TASKS:-89}"
MARKER="${DEST_DIR}/.slime_extract_complete"

count_tasks() {
  if [ ! -d "${DEST_DIR}" ]; then
    echo 0
    return
  fi
  find "${DEST_DIR}" -mindepth 2 -maxdepth 2 -name task.toml | wc -l
}

mkdir -p "${DEST_ROOT}"

if [ -f "${MARKER}" ] && [ "$(count_tasks)" = "${EXPECTED_TASKS}" ]; then
  echo "tb2_train_v2_local_tasks_ready ${DEST_DIR} tasks=${EXPECTED_TASKS}"
  exit 0
fi

test -f "${TAR_PATH}"
cp "${TAR_PATH}" "${LOCAL_TAR}"
tar -xzf "${LOCAL_TAR}" -C "${DEST_ROOT}"

TASK_COUNT="$(count_tasks)"
if [ "${TASK_COUNT}" != "${EXPECTED_TASKS}" ]; then
  echo "ERROR: expected ${EXPECTED_TASKS} task.toml files under ${DEST_DIR}, got ${TASK_COUNT}" >&2
  exit 1
fi

date -Iseconds > "${MARKER}"
echo "tb2_train_v2_local_tasks_ready ${DEST_DIR} tasks=${TASK_COUNT}"
