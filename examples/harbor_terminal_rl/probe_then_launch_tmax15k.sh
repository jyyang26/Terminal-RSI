#!/bin/bash
#
# probe_then_launch_tmax15k.sh — gate the tmax-15k launch on a CONCURRENT Daytona
# image-build probe, then fan out launcher + 11 workers only if the probe PASSES.
#
# WHY: the tmax-15k run was stopped twice (2026-07-01 02:24 & 06:27) by the same
# Daytona prod-daytona-build-context S3 outage. A single-shot image-build probe
# passed both times (2s), but 280-concurrency at rollout start triggered persistent
# S3 HEAD timeouts. So we gate on a *concurrent* probe, not a single create.
#
# USAGE (run on node0 / launcher):
#   bash probe_then_launch_tmax15k.sh
#   PROBE_CONCURRENCY=280 PROBE_MIN_RATE=0.85 bash probe_then_launch_tmax15k.sh
#   DRY_RUN=1 bash probe_then_launch_tmax15k.sh      # probe only, never launch
#
# Env knobs:
#   PROBE_CONCURRENCY (default 96)   how many image-builds to fire at once
#   PROBE_MIN_RATE    (default 0.90) success-rate threshold to allow launch
#   PROBE_TIMEOUT     (default 420)  per-build timeout (sec)
#   DRY_RUN           (default 0)    1 = run probe, report, but DO NOT launch
#   MASTER_ADDR       (default 10.0.0.11)
set -uo pipefail

CODE_ROOT="${CODE_ROOT:-/path/to/workspace/code}"
SLIME_DIR="${SLIME_DIR:-${CODE_ROOT}/slime-v0.3-harbor}"
PY="${PY:-${CODE_ROOT}/harbor/.venv/bin/python}"
PROBE="${SLIME_DIR}/examples/harbor_terminal_rl/daytona_concurrent_build_probe.py"
START="${SLIME_DIR}/examples/harbor_terminal_rl/start_tmax15k_full_96gpu_27b_daytona_ppo_64k_tmux.sh"

PROBE_CONCURRENCY="${PROBE_CONCURRENCY:-96}"
PROBE_MIN_RATE="${PROBE_MIN_RATE:-0.90}"
PROBE_TIMEOUT="${PROBE_TIMEOUT:-420}"
TASKS_DIR="${TASKS_DIR:-/tmp/tmax_harbor_15k_full}"
DRY_RUN="${DRY_RUN:-0}"
MASTER_ADDR="${MASTER_ADDR:-10.0.0.11}"

# Worker bond IPs (node0 + these 11 = 12 nodes / 96 GPU). From NODE_IP_LIST.
WORKERS="${WORKERS:-10.0.0.12 10.0.0.18 10.0.0.15 10.0.0.7 10.0.0.14 10.0.0.9 10.0.0.8 10.0.0.16 10.0.0.3 10.0.0.4 10.0.0.21}"

echo "================================================================"
echo "  STEP 1/2: concurrent Daytona image-build probe"
echo "  concurrency=${PROBE_CONCURRENCY} min_rate=${PROBE_MIN_RATE} timeout=${PROBE_TIMEOUT}s"
echo "================================================================"

"${PY}" "${PROBE}" \
  --tasks-dir "${TASKS_DIR}" \
  --concurrency "${PROBE_CONCURRENCY}" \
  --min-success-rate "${PROBE_MIN_RATE}" \
  --build-timeout "${PROBE_TIMEOUT}"
PROBE_RC=$?

if [ "${PROBE_RC}" -ne 0 ]; then
  echo ""
  echo "[gate] PROBE FAILED (rc=${PROBE_RC}) — Daytona S3 not healthy under load. NOT launching."
  echo "[gate] Re-run later; data is staged, config unchanged. Nothing was started, nothing to clean."
  exit "${PROBE_RC}"
fi

echo ""
echo "[gate] PROBE PASSED — Daytona S3 healthy under concurrency=${PROBE_CONCURRENCY}."

if [ "${DRY_RUN}" = "1" ]; then
  echo "[gate] DRY_RUN=1 — stopping here, not launching."
  exit 0
fi

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
echo "================================================================"
echo "  STEP 2/2: launching 12-node tmax-15k run  RUN_TAG=${RUN_TAG}"
echo "================================================================"

echo "[gate] launcher on node0 (${MASTER_ADDR}) ..."
RUN_TAG="${RUN_TAG}" bash "${START}" launcher "${MASTER_ADDR}" 2>&1 | sed 's/^/[launcher] /'

echo "[gate] fanning out to 11 workers ..."
for ip in ${WORKERS}; do
  timeout 60 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${ip}" \
    "RUN_TAG=${RUN_TAG} bash ${START} worker ${MASTER_ADDR}" 2>&1 \
    | grep -E "started|run_tag|ERROR|error" | sed "s/^/  [${ip}] /"
done

echo ""
echo "[gate] all 12 nodes dispatched. RUN_TAG=${RUN_TAG}"
echo "[gate] launcher log: /path/to/workspace/slime_harbor_qwen35_27b_tmax15k_full_ppo_64k_${RUN_TAG}/logs/launcher_${RUN_TAG}.log"
echo "[gate] REMEMBER: arm a fresh failguard cron for RUN_TAG=${RUN_TAG} and the new job id."
