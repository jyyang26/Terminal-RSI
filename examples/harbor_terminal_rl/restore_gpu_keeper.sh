#!/usr/bin/env bash
set -euo pipefail

SERVER_NAME="${1:-self_improving_terminal_agent_RL_32GPU_slime_v4}"
NODE_INDEX="${2:-0}"
SESSION="${GPU_KEEPER_SESSION:-gpu_keeper}"
PID_DIR="/tmp/gpu_keeper_pids_${SERVER_NAME}_node${NODE_INDEX}"
LOG_FILE="${GPU_KEEPER_LOG:-/tmp/gpu_keeper.log}"
WORKER_FILE="/tmp/gpu_keeper_worker.py"
LAUNCHER_FILE="/tmp/gpu_keeper_launcher_${SERVER_NAME}_node${NODE_INDEX}.sh"
MATRIX="${GPU_KEEPER_MATRIX:-8192}"
LOOPS="${GPU_KEEPER_LOOPS:-64}"
SLEEP_SECS="${GPU_KEEPER_SLEEP:-0.0}"

if [[ -n "${GPU_KEEPER_NUM_GPUS:-}" ]]; then
  NUM_GPUS="${GPU_KEEPER_NUM_GPUS}"
else
  NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
fi

if [[ "${NUM_GPUS}" -le 0 ]]; then
  echo "No GPUs found; keeper not started." >&2
  exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
fi

if [[ -d "${PID_DIR}" ]]; then
  for pid_file in "${PID_DIR}"/*.pid; do
    [[ -e "${pid_file}" ]] || continue
    old_pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]]; then
      kill "${old_pid}" 2>/dev/null || true
    fi
  done
fi

rm -rf "${PID_DIR}"
mkdir -p "${PID_DIR}"

cat > "${WORKER_FILE}" <<'PY'
import os
import time

import torch


gpu = os.environ.get("GPU_KEEPER_GPU", "0")
matrix = int(os.environ.get("GPU_KEEPER_MATRIX", "2048"))
loops = int(os.environ.get("GPU_KEEPER_LOOPS", "32"))
sleep_secs = float(os.environ.get("GPU_KEEPER_SLEEP", "0.2"))

torch.cuda.set_device(0)
a = torch.randn((matrix, matrix), device="cuda", dtype=torch.float16)
b = torch.randn((matrix, matrix), device="cuda", dtype=torch.float16)
print(
    f"[gpu_dummy] GPU {gpu} occupied, pid={os.getpid()}, matrix={matrix}, loops={loops}",
    flush=True,
)

while True:
    for _ in range(loops):
        c = a @ b
    torch.cuda.synchronize()
    time.sleep(sleep_secs)
PY

{
  echo "#!/usr/bin/env bash"
  echo "set -u"
  printf "PID_DIR=%q\n" "${PID_DIR}"
  printf "LOG_FILE=%q\n" "${LOG_FILE}"
  printf "WORKER_FILE=%q\n" "${WORKER_FILE}"
  printf "SERVER_NAME=%q\n" "${SERVER_NAME}"
  printf "NODE_INDEX=%q\n" "${NODE_INDEX}"
  printf "NUM_GPUS=%q\n" "${NUM_GPUS}"
  printf "MATRIX=%q\n" "${MATRIX}"
  printf "LOOPS=%q\n" "${LOOPS}"
  printf "SLEEP_SECS=%q\n" "${SLEEP_SECS}"
  cat <<'SH'

cleanup() {
  for pid_file in "${PID_DIR}"/*.pid; do
    [[ -e "${pid_file}" ]] || continue
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    [[ -n "${pid}" ]] || continue
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM

timestamp() {
  date "+%F %T"
}

launch_one() {
  local gpu="$1"
  local pid_file="${PID_DIR}/dummy_gpu${gpu}.pid"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    GPU_KEEPER_GPU="${gpu}" \
    GPU_KEEPER_MATRIX="${MATRIX}" \
    GPU_KEEPER_LOOPS="${LOOPS}" \
    GPU_KEEPER_SLEEP="${SLEEP_SECS}" \
    nohup python3 "${WORKER_FILE}" >> "${LOG_FILE}" 2>&1 &
  echo "$!" > "${pid_file}"
  echo "[gpu_keeper $(timestamp)] launched gpu=${gpu} pid=$!" >> "${LOG_FILE}"
}

echo "[gpu_keeper $(timestamp)] GPU Keeper restored: server=${SERVER_NAME} node=${NODE_INDEX} pid=$$ num_gpus=${NUM_GPUS}" >> "${LOG_FILE}"

while true; do
  for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    pid_file="${PID_DIR}/dummy_gpu${gpu}.pid"
    pid=""
    [[ -f "${pid_file}" ]] && pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
      launch_one "${gpu}"
    fi
  done
  sleep 30
done
SH
} > "${LAUNCHER_FILE}"

chmod +x "${LAUNCHER_FILE}"
tmux new-session -d -s "${SESSION}" "bash ${LAUNCHER_FILE}"

echo "Started ${SESSION}: server=${SERVER_NAME} node=${NODE_INDEX} num_gpus=${NUM_GPUS} pid_dir=${PID_DIR} log=${LOG_FILE}"
