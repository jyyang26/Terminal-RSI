#!/bin/bash

set -e

# Qwen3.5-9B + Harbor Terminus2 RL on terminal-bench-2 with 8 x 8 H20.
# Usage:
#   1. Convert HF -> torch_dist with convert_qwen35_9b_torch_dist.sh.
#   2. Start this script on the launcher.
#   3. Start this same script with RUN_ROLE=worker on each worker node.

export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

RUN_ROLE=${RUN_ROLE:-launcher}

log_event() {
  printf '[%s %s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${RUN_ROLE}" "$*"
}

# Raise the open-file (nofile) soft limit BEFORE any `ray start`, so the raylet and
# every child (SGLang engines, trial workers, train actors) inherit a high cap.
# The default soft limit is 1024 while the hard limit is ~1e6. With 320 concurrent
# Daytona trials, the COORDINATOR node's raylet accumulates one socket+pipe set per
# trial future + per SGLang connection; at ~794 sockets + ~199 pipes it hits 1024 and
# starts spewing "File descriptor limit reached. Retrying", which DEADLOCKS rollout
# collection (trial result futures can't be delivered back to the coordinator -> the
# fully-async loop sits at "collected 0/16, completed=0" forever). The fast-finishing
# GSPO run happened to stay under 1024 by churning fds quickly; a run that collects
# slower (e.g. PPO's critic+actor serial step, or just an unlucky slow first rollout)
# crosses the line and never recovers. Lift soft -> hard so this can't happen.
RAISE_NOFILE_TO=${RAISE_NOFILE_TO:-$(ulimit -Hn 2>/dev/null || echo 1048576)}
if [ -n "${RAISE_NOFILE_TO}" ] && [ "${RAISE_NOFILE_TO}" != "unlimited" ]; then
  ulimit -n "${RAISE_NOFILE_TO}" 2>/dev/null \
    && log_event "raised nofile soft limit to $(ulimit -n)" \
    || log_event "WARN: failed to raise nofile soft limit (current=$(ulimit -n), target=${RAISE_NOFILE_TO})"
fi

CODE_ROOT=${CODE_ROOT:-/path/to/workspace/code}
SLIME_DIR=${SLIME_DIR:-${CODE_ROOT}/slime}
HARBOR_SRC=${HARBOR_SRC:-${CODE_ROOT}/harbor/src}
BASE_FOLDER=${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_9b_terminal_bench2}
HF_CKPT=${HF_CKPT:-/path/to/models/Qwen3.5-9B}
PRETRAINED_CKPT=${PRETRAINED_CKPT:-/path/to/models/Qwen3.5-9B_torch_dist}
CONFIG=${CONFIG:-${SLIME_DIR}/examples/harbor_terminal_rl/harbor_terminal_bench2_config.yaml}
MODEL_SCRIPT=${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-9B.sh}
SAVE_SUBDIR=${SAVE_SUBDIR:-Qwen3.5-9B_terminal_bench2_slime}
STAGE_HF_CKPT_TO_TMP=${STAGE_HF_CKPT_TO_TMP:-0}
STAGED_HF_CKPT=${STAGED_HF_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.5-9B}
STAGE_HF_CKPT_FORCE=${STAGE_HF_CKPT_FORCE:-0}
STAGE_TORCH_DIST_CKPT_TO_TMP=${STAGE_TORCH_DIST_CKPT_TO_TMP:-0}
STAGED_TORCH_DIST_CKPT=${STAGED_TORCH_DIST_CKPT:-/tmp/slime-harbor-model-cache/$(basename "${PRETRAINED_CKPT}")}
STAGE_TORCH_DIST_CKPT_FORCE=${STAGE_TORCH_DIST_CKPT_FORCE:-0}
APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-/tmp/slime-harbor-apptainer-cache}
WORKER_KEEP_ALIVE=${WORKER_KEEP_ALIVE:-0}
RAY_CLUSTER_ONLY=${RAY_CLUSTER_ONLY:-0}
ENABLE_PROXY=${ENABLE_PROXY:-1}
PROXY_ENV_FILE=${PROXY_ENV_FILE:-${SCRIPT_DIR}/registry_proxy_env.sh}

MASTER_ADDR=${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}
RAY_NODE_IP_PREFIX=${RAY_NODE_IP_PREFIX:-$(printf "%s\n" "${MASTER_ADDR}" | awk -F. '{print $1"."$2}')}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-56}
EXPECTED_NODES=${EXPECTED_NODES:-8}
CP_SIZE=${CP_SIZE:-1}
SOCKET_IFNAME=${SOCKET_IFNAME:-bond1}

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-1}
OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2}
NUM_ROLLOUT=${NUM_ROLLOUT:-2}
HARBOR_ROLLOUT_MODE=${HARBOR_ROLLOUT_MODE:-standard}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-2048}
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-}
DEBUG_ROLLOUT_ONLY=${DEBUG_ROLLOUT_ONLY:-0}
LOAD_DEBUG_ROLLOUT_DATA=${LOAD_DEBUG_ROLLOUT_DATA:-}
LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE=${LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE:-}
SAVE_DEBUG_TRAIN_DATA=${SAVE_DEBUG_TRAIN_DATA:-}
DUMP_DETAILS=${DUMP_DETAILS:-}
CLEAN_START=${CLEAN_START:-1}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.80}
SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-}
SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-}
SGLANG_CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-1}
# Expert parallelism for MoE models (Qwen3.5-35B-A3B etc.). Defaults to 1 so dense
# models (9B/27B) keep their exact previous behaviour; MoE wrappers override these.
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
DECODER_LAST_PIPELINE_NUM_LAYERS=${DECODER_LAST_PIPELINE_NUM_LAYERS:-}
QKV_FORMAT=${QKV_FORMAT:-thd}
SEQUENCE_PARALLEL=${SEQUENCE_PARALLEL:-1}
SEQ_LENGTH=${SEQ_LENGTH:-}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-}
# Padding granularity for thd sequence packing (data.py: pad_size = TP * this).
# Larger = fewer distinct padded microbatch shapes = fewer Triton/GDN JIT
# recompiles. Default 128 = slime stock; raise to 2048 at 96k to collapse the
# step0 cold-compile storm (see PPO_INTEGRATION_NOTES §X). Empty = don't pass.
DATA_PAD_SIZE_MULTIPLIER=${DATA_PAD_SIZE_MULTIPLIER:-}
USE_DYNAMIC_BATCH_SIZE=${USE_DYNAMIC_BATCH_SIZE:-0}
USE_DYNAMIC_GLOBAL_BATCH_SIZE=${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-0}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-}
ROLLOUT_ALL_SAMPLES_PROCESS_PATH=${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-}
ROLLOUT_DATA_POSTPROCESS_PATH=${ROLLOUT_DATA_POSTPROCESS_PATH:-}
CUSTOM_REWARD_POST_PROCESS_PATH=${CUSTOM_REWARD_POST_PROCESS_PATH:-}
HARBOR_MAX_TRAIN_TOTAL_TOKENS=${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-}
USE_CUSTOM_NO_ENTROPY_LOSS=${USE_CUSTOM_NO_ENTROPY_LOSS:-0}
CUSTOM_LOSS_FUNCTION_PATH=${CUSTOM_LOSS_FUNCTION_PATH:-slime_plugins.harbor.train_loss.policy_loss_no_entropy_when_disabled}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}
KEEP_OLD_ACTOR=${KEEP_OLD_ACTOR:-0}
USE_ROLLOUT_LOGPROBS=${USE_ROLLOUT_LOGPROBS:-0}
CALCULATE_PER_TOKEN_LOSS=${CALCULATE_PER_TOKEN_LOSS:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-20}
# Space-separated extra flags appended verbatim to the SGLang / MISC arg groups.
# Used by MoE wrappers to inject expert-parallel + mamba serving flags without
# disturbing dense (9B/27B) runs, which leave these empty.
EXTRA_SGLANG_ARGS=${EXTRA_SGLANG_ARGS:-}
EXTRA_MISC_ARGS=${EXTRA_MISC_ARGS:-}

USE_WANDB=${USE_WANDB:-0}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_PROJECT=${WANDB_PROJECT:-slime-harbor-terminal-rl}
WANDB_GROUP=${WANDB_GROUP:-qwen35-9b-terminal-bench2}
WANDB_DIR=${WANDB_DIR:-${BASE_FOLDER%/}/wandb}
WANDB_HOST=${WANDB_HOST:-}
WANDB_TEAM=${WANDB_TEAM:-}
WANDB_RUN_ID=${WANDB_RUN_ID:-}
WANDB_DISABLE_RANDOM_SUFFIX=${WANDB_DISABLE_RANDOM_SUFFIX:-1}

RAY_PORT=${RAY_PORT:-6379}
RAY_NODE_MANAGER_PORT=${RAY_NODE_MANAGER_PORT:-18001}
RAY_OBJECT_MANAGER_PORT=${RAY_OBJECT_MANAGER_PORT:-18002}
RAY_RUNTIME_ENV_AGENT_PORT=${RAY_RUNTIME_ENV_AGENT_PORT:-18003}
RAY_DASHBOARD_AGENT_LISTEN_PORT=${RAY_DASHBOARD_AGENT_LISTEN_PORT:-18004}
RAY_DASHBOARD_AGENT_GRPC_PORT=${RAY_DASHBOARD_AGENT_GRPC_PORT:-18005}
RAY_METRICS_EXPORT_PORT=${RAY_METRICS_EXPORT_PORT:-18006}
RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT:-20000}
RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-20999}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/slime-harbor-ray}
RAY_CLUSTER_WAIT_SECONDS=${RAY_CLUSTER_WAIT_SECONDS:-240}
USE_EXISTING_RAY_HEAD=${USE_EXISTING_RAY_HEAD:-0}
DIRECT_RUN=${DIRECT_RUN:-0}
RAY_INCLUDE_DASHBOARD=${RAY_INCLUDE_DASHBOARD:-1}
WORKER_STOP_MARKER=${WORKER_STOP_MARKER:-${BASE_FOLDER%/}/.stop_workers_${MASTER_ADDR//./_}_${RAY_PORT}}

if [ "${ENABLE_PROXY}" = "1" ]; then
  if [ -f "${PROXY_ENV_FILE}" ]; then
    # Includes the star-proxy settings needed by W&B and registry diagnostics.
    source "${PROXY_ENV_FILE}"
  fi
  export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"
  export https_proxy="${https_proxy:-${http_proxy}}"
  export ftp_proxy="${ftp_proxy:-${http_proxy}}"
  export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
  export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
  export FTP_PROXY="${FTP_PROXY:-${ftp_proxy}}"
fi

ACTOR_TOTAL_GPUS=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
ACTOR_PARALLEL_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * CP_SIZE))
if [ "${ACTOR_TOTAL_GPUS}" -lt "${ACTOR_PARALLEL_SIZE}" ]; then
  echo "ERROR: actor GPUs (${ACTOR_TOTAL_GPUS}) < TP*PP*CP (${ACTOR_PARALLEL_SIZE})." >&2
  exit 1
fi
if [ $((ACTOR_TOTAL_GPUS % ACTOR_PARALLEL_SIZE)) -ne 0 ]; then
  echo "ERROR: actor GPUs (${ACTOR_TOTAL_GPUS}) must be divisible by TP*PP*CP (${ACTOR_PARALLEL_SIZE})." >&2
  exit 1
fi

RAY_PORT_ARGS=(
  --object-manager-port "${RAY_OBJECT_MANAGER_PORT}"
  --node-manager-port "${RAY_NODE_MANAGER_PORT}"
  --runtime-env-agent-port "${RAY_RUNTIME_ENV_AGENT_PORT}"
  --dashboard-agent-listen-port "${RAY_DASHBOARD_AGENT_LISTEN_PORT}"
  --dashboard-agent-grpc-port "${RAY_DASHBOARD_AGENT_GRPC_PORT}"
  --metrics-export-port "${RAY_METRICS_EXPORT_PORT}"
  --min-worker-port "${RAY_MIN_WORKER_PORT}"
  --max-worker-port "${RAY_MAX_WORKER_PORT}"
)

pick_ray_node_ip() {
  local ip prefix other count
  local ips
  read -r -a ips <<< "$(hostname -I)"
  if [ -n "${RAY_NODE_IP_PREFIX}" ]; then
    IFS=',' read -r -a prefixes <<< "${RAY_NODE_IP_PREFIX}"
    for prefix in "${prefixes[@]}"; do
      prefix="$(printf "%s" "${prefix}" | xargs)"
      [ -n "${prefix}" ] || continue
      for ip in "${ips[@]}"; do
        case "${ip}" in
          "${prefix}".*) echo "${ip}"; return 0 ;;
        esac
      done
    done
  fi
  for ip in "${ips[@]}"; do
    count=0
    for other in "${ips[@]}"; do
      if [ "${ip}" = "${other}" ]; then
        count=$((count + 1))
      fi
    done
    if [ "${count}" -gt 1 ]; then
      echo "${ip}"
      return 0
    fi
  done
  printf "%s\n" "${ips[0]}"
}

if [ "${CLEAN_START}" = "1" ]; then
  log_event "CLEAN_START=1; stopping stale local Ray/SGLang processes"
  pkill -9 sglang || true
  ray stop --force || true
  pkill -9 -f "[r]aylet|[g]cs_server|[d]ashboard_agent|[r]untime_env_agent|[l]og_monitor" || true
  pkill -9 -f "[p]ython3 train_async.py|[p]ython train_async.py|[p]ython3 train.py|[p]ython train.py" || true
  sleep 3
fi

if ! command -v singularity >/dev/null 2>&1; then
  if command -v apptainer >/dev/null 2>&1; then
    mkdir -p /tmp/slime-harbor-bin
    ln -sf "$(command -v apptainer)" /tmp/slime-harbor-bin/singularity
    export PATH=/tmp/slime-harbor-bin:${PATH}
  else
    echo "WARNING: neither singularity nor apptainer is in PATH; Harbor singularity tasks will fail."
  fi
fi

if [ "${STAGE_HF_CKPT_TO_TMP}" = "1" ]; then
  mkdir -p "$(dirname "${STAGED_HF_CKPT}")"
  STAGE_MARKER="${STAGED_HF_CKPT}/.slime_stage_complete"
  if [ "${STAGE_HF_CKPT_FORCE}" = "1" ] || [ ! -f "${STAGE_MARKER}" ]; then
    mkdir -p "${STAGED_HF_CKPT}"
    rsync -a --delete "${HF_CKPT%/}/" "${STAGED_HF_CKPT%/}/"
    date -Iseconds > "${STAGE_MARKER}"
  fi
  HF_CKPT="${STAGED_HF_CKPT}"
fi
if [ "${STAGE_TORCH_DIST_CKPT_TO_TMP}" = "1" ]; then
  mkdir -p "$(dirname "${STAGED_TORCH_DIST_CKPT}")"
  STAGE_MARKER="${STAGED_TORCH_DIST_CKPT}/.slime_stage_complete"
  if [ "${STAGE_TORCH_DIST_CKPT_FORCE}" = "1" ] || [ ! -f "${STAGE_MARKER}" ]; then
    mkdir -p "${STAGED_TORCH_DIST_CKPT}"
    rsync -a --delete "${PRETRAINED_CKPT%/}/" "${STAGED_TORCH_DIST_CKPT%/}/"
    date -Iseconds > "${STAGE_MARKER}"
  fi
  PRETRAINED_CKPT="${STAGED_TORCH_DIST_CKPT}"
fi
mkdir -p "${APPTAINER_CACHEDIR}"
export APPTAINER_CACHEDIR
export PYTHONPATH="${HARBOR_SRC}:${SLIME_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF

if [ "${RUN_ROLE}" = "worker" ]; then
  test -n "${MASTER_ADDR}"
  WORKER_MARKER_START_EPOCH="$(date +%s)"
  WORKER_ADDR=${WORKER_ADDR:-$(pick_ray_node_ip)}
  mkdir -p "${RAY_TEMP_DIR}"
  log_event "starting Ray worker: master=${MASTER_ADDR}:${RAY_PORT} worker_addr=${WORKER_ADDR}"
  for i in $(seq 1 120); do
    if ray start \
      --address "${MASTER_ADDR}:${RAY_PORT}" \
      --node-ip-address "${WORKER_ADDR}" \
      --num-gpus 8 \
      --temp-dir "${RAY_TEMP_DIR}" \
      "${RAY_PORT_ARGS[@]}" \
      --disable-usage-stats; then
      break
    fi
    log_event "waiting for Ray head ${MASTER_ADDR}:${RAY_PORT} before starting worker (${i}/120)"
    sleep 5
  done
  ray status || true
  if [ "${WORKER_KEEP_ALIVE}" = "1" ]; then
    log_event "worker keep-alive enabled; waiting for stop marker: ${WORKER_STOP_MARKER}"
    while true; do
      if [ -f "${WORKER_STOP_MARKER}" ]; then
        MARKER_MTIME="$(stat -c %Y "${WORKER_STOP_MARKER}" 2>/dev/null || echo 0)"
        if [ "${MARKER_MTIME}" -ge "${WORKER_MARKER_START_EPOCH}" ]; then
          break
        fi
        log_event "ignoring stale worker stop marker from a previous run: ${WORKER_STOP_MARKER}"
        rm -f "${WORKER_STOP_MARKER}" || true
      fi
      sleep 60
    done
    log_event "worker stop marker detected; stopping Ray worker"
    ray stop --force || true
  fi
  exit 0
fi

test -d "${SLIME_DIR}"
test -d "${HARBOR_SRC}"
test -d "${HF_CKPT}"
# In bridge mode the actor weights are loaded directly from the HF checkpoint via
# mbridge, so no pre-converted torch_dist PRETRAINED_CKPT is required.
if [ "${MEGATRON_TO_HF_MODE:-raw}" != "bridge" ]; then
   test -d "${PRETRAINED_CKPT}"
fi
test -f "${CONFIG}"
test -f "${MODEL_SCRIPT}"
mkdir -p "${BASE_FOLDER}/rollout_dumps"
SAVE_DIR="${BASE_FOLDER%/}/${SAVE_SUBDIR}"
log_event "run config: mode=${HARBOR_ROLLOUT_MODE} base=${BASE_FOLDER} save=${SAVE_DIR} config=${CONFIG}"
log_event "run config: actor_nodes=${ACTOR_NUM_NODES} actor_gpus_per_node=${ACTOR_NUM_GPUS_PER_NODE} rollout_gpus=${ROLLOUT_NUM_GPUS} expected_nodes=${EXPECTED_NODES}"
log_event "run config: rollout_batch_size=${ROLLOUT_BATCH_SIZE} n_samples_per_prompt=${N_SAMPLES_PER_PROMPT} global_batch_size=${GLOBAL_BATCH_SIZE} num_rollout=${NUM_ROLLOUT}"

cd "${SLIME_DIR}"
source "${MODEL_SCRIPT}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
log_event "HAS_NVLINK=${HAS_NVLINK} detected_nvlink_refs=${NVLINK_COUNT}"

LOAD_DIR="${PRETRAINED_CKPT}"
if [ -d "${SAVE_DIR}/" ] && [ -n "$(ls -A "${SAVE_DIR}/" 2>/dev/null)" ]; then
   LOAD_DIR="${SAVE_DIR}/"
fi
log_event "checkpoint load_dir=${LOAD_DIR} megatron_to_hf_mode=${MEGATRON_TO_HF_MODE:-raw}"

case "${HARBOR_ROLLOUT_MODE}" in
   standard|async)
      HARBOR_ROLLOUT_FUNCTION="slime_plugins.harbor.rollout.generate_rollout"
      ;;
   fully_async|full_async)
      HARBOR_ROLLOUT_FUNCTION="slime_plugins.harbor.fully_async_rollout.generate_rollout_fully_async"
      ;;
   *)
      echo "Unsupported HARBOR_ROLLOUT_MODE=${HARBOR_ROLLOUT_MODE}; expected standard or fully_async" >&2
      exit 1
      ;;
esac

CKPT_ARGS=(
   --train-backend megatron
   --hf-checkpoint "${HF_CKPT}"
   --save "${SAVE_DIR}/"
   --save-interval "${SAVE_INTERVAL}"
)
if [ "${MEGATRON_TO_HF_MODE:-raw}" = "bridge" ]; then
   # bridge mode: actor weights are loaded directly from the HF checkpoint via
   # mbridge at startup (no pre-converted torch_dist needed, no --ref-load).
   # Only attach --load when resuming from a previously saved Megatron ckpt.
   # NOTE: bridge mode is incompatible with --use-kl-loss unless --ref-load points
   # at a Megatron ckpt (arguments.py asserts ref_load exists for KL); callers using
   # bridge must therefore run with KL off (USE_KL_LOSS=0).
   CKPT_ARGS+=(--megatron-to-hf-mode bridge)
   if [ -d "${SAVE_DIR}/" ] && [ -n "$(ls -A "${SAVE_DIR}/" 2>/dev/null)" ]; then
      CKPT_ARGS+=(--load "${SAVE_DIR}/")
   fi
else
   CKPT_ARGS+=(
      --ref-load "${PRETRAINED_CKPT}"
      --load "${LOAD_DIR}"
   )
fi

HARBOR_ROLLOUT_ARGS=(
   --rollout-function-path "${HARBOR_ROLLOUT_FUNCTION}"
   --data-source-path slime_plugins.harbor.data_source.HarborTaskDataSource
   --custom-config-path "${CONFIG}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
   --rollout-top-p "${ROLLOUT_TOP_P:-1.0}"
   --rollout-top-k "${ROLLOUT_TOP_K:--1}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
)
# Periodic held-out eval (e.g. Terminal-Bench 2). Opt-in via EVAL_INTERVAL.
#   * --eval-interval N fires eval every N rollout steps (train_async.py).
#   * Eval MUST use the STANDARD rollout fn: the fully_async rollout fn raises on
#     evaluation=True by design, and --eval-function-path defaults to
#     --rollout-function-path. Pin it to slime_plugins.harbor.rollout.generate_rollout,
#     which routes evaluation=True to a dedicated full-set, no-oversample,
#     no-tail-cancel pass (_generate_eval_async). This is independent of the
#     fully_async TRAINING path.
#   * The eval task pool is configured in the YAML via harbor_eval_tasks_dir/glob,
#     consumed by HarborTaskDataSource without touching the training cursor.
#   * --eval-prompt-data is a DUMMY placeholder: slime asserts args.eval_datasets
#     is non-empty when --eval-interval is set (arguments.py), but the Harbor eval
#     path never reads it (it uses the data_source eval pool). The path is not
#     checked for existence.
if [ -n "${EVAL_INTERVAL:-}" ]; then
   HARBOR_ROLLOUT_ARGS+=(
      --eval-interval "${EVAL_INTERVAL}"
      --eval-function-path slime_plugins.harbor.rollout.generate_rollout
      --eval-prompt-data "${EVAL_DATASET_NAME:-tb2}" /dev/null
      --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
      --eval-temperature "${EVAL_TEMPERATURE:-${ROLLOUT_TEMPERATURE:-1.0}}"
      --eval-top-p "${EVAL_TOP_P:-${ROLLOUT_TOP_P:-1.0}}"
      --eval-top-k "${EVAL_TOP_K:-${ROLLOUT_TOP_K:--1}}"
   )
   if [ "${SKIP_EVAL_BEFORE_TRAIN:-0}" = "1" ]; then
      HARBOR_ROLLOUT_ARGS+=(--skip-eval-before-train)
   fi
else
   HARBOR_ROLLOUT_ARGS+=(--skip-eval-before-train)
fi
if [ -n "${ROLLOUT_MAX_CONTEXT_LEN}" ]; then
   HARBOR_ROLLOUT_ARGS+=(--rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}")
fi
if [ -n "${OVER_SAMPLING_BATCH_SIZE}" ]; then
   HARBOR_ROLLOUT_ARGS+=(--over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}")
fi
if [ "${USE_DYNAMIC_GLOBAL_BATCH_SIZE}" = "1" ]; then
   HARBOR_ROLLOUT_ARGS+=(--use-dynamic-global-batch-size)
fi
if [ -n "${ROLLOUT_ALL_SAMPLES_PROCESS_PATH}" ]; then
   HARBOR_ROLLOUT_ARGS+=(--rollout-all-samples-process-path "${ROLLOUT_ALL_SAMPLES_PROCESS_PATH}")
fi
if [ -n "${CUSTOM_REWARD_POST_PROCESS_PATH}" ]; then
   HARBOR_ROLLOUT_ARGS+=(--custom-reward-post-process-path "${CUSTOM_REWARD_POST_PROCESS_PATH}")
fi
# Rollout-engine fault tolerance (opt-in via USE_FAULT_TOLERANCE=1; default OFF =
# backward-compatible). When ON, a RolloutHealthMonitor pings each SGLang engine's
# /health_generate every HEALTH_CHECK_INTERVAL s; a dead/hung engine is killed and
# then RESTARTED at the next update_weights() (actor.py recover_updatable_engines ->
# srv.recover()), so a single engine that dies mid-run (e.g. a GDN-kernel cold-compile
# CUDA-OOM SIGQUIT) no longer wedges the 65-rank weight-broadcast gloo barrier until
# timeout -> job exit 1. This is exactly the failure that killed cp2_64k_20260627_122131
# (see PPO_INTEGRATION_NOTES.md §Y). FIRST_WAIT gives a grace period so engines still
# cold-compiling GDN kernels at step0 are not mistaken for dead and killed; TIMEOUT is
# how long one /health_generate may take before the engine is declared dead.
if [ "${USE_FAULT_TOLERANCE:-0}" = "1" ]; then
   HARBOR_ROLLOUT_ARGS+=(--use-fault-tolerance)
   HARBOR_ROLLOUT_ARGS+=(--rollout-health-check-interval "${ROLLOUT_HEALTH_CHECK_INTERVAL:-30}")
   HARBOR_ROLLOUT_ARGS+=(--rollout-health-check-timeout "${ROLLOUT_HEALTH_CHECK_TIMEOUT:-120}")
   HARBOR_ROLLOUT_ARGS+=(--rollout-health-check-first-wait "${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-600}")
fi

DEBUG_DUMP_ARGS=(
   --save-debug-rollout-data "${BASE_FOLDER}/rollout_dumps/{rollout_id}.pt"
)
if [ "${DEBUG_ROLLOUT_ONLY}" = "1" ]; then
   DEBUG_DUMP_ARGS+=(--debug-rollout-only)
fi
if [ -n "${LOAD_DEBUG_ROLLOUT_DATA}" ]; then
   DEBUG_DUMP_ARGS+=(--load-debug-rollout-data "${LOAD_DEBUG_ROLLOUT_DATA}")
fi
if [ -n "${LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE}" ]; then
   DEBUG_DUMP_ARGS+=(--load-debug-rollout-data-subsample "${LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE}")
fi
if [ -n "${SAVE_DEBUG_TRAIN_DATA}" ]; then
   DEBUG_DUMP_ARGS+=(--save-debug-train-data "${SAVE_DEBUG_TRAIN_DATA}")
fi
if [ -n "${DUMP_DETAILS}" ]; then
   DEBUG_DUMP_ARGS+=(--dump-details "${DUMP_DETAILS}")
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
   --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
   --context-parallel-size "${CP_SIZE}"
   --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
   --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE}"

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --qkv-format "${QKV_FORMAT}"
   --micro-batch-size 1

   # NCCL/distributed collective watchdog timeout. slime defaults this to 10min
   # (arguments.py reset_arg --distributed-timeout-minutes default=10), which a
   # slow/imbalanced rank can blow during a 27B PP forward -> the watchdog aborts
   # the whole process group and kills the run (observed: PPO by_length run
   # twppon4_160454 died at step3's critic PP8 forward, NCCL COALESCED op ran
   # 600012ms > 600000ms). Raise it well above any legitimate collective so a
   # transient rank stall self-recovers instead of tearing down the job.
   --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES:-60}"
)
if [ "${CALCULATE_PER_TOKEN_LOSS}" = "1" ]; then
   PERF_ARGS+=(--calculate-per-token-loss)
fi
if [ -n "${DECODER_LAST_PIPELINE_NUM_LAYERS}" ]; then
   PERF_ARGS+=(--decoder-last-pipeline-num-layers "${DECODER_LAST_PIPELINE_NUM_LAYERS}")
fi
if [ "${SEQUENCE_PARALLEL}" = "1" ]; then
   PERF_ARGS+=(--sequence-parallel)
fi
if [ -n "${SEQ_LENGTH}" ]; then
   PERF_ARGS+=(--seq-length "${SEQ_LENGTH}" --max-position-embeddings "${SEQ_LENGTH}")
fi
if [ "${USE_DYNAMIC_BATCH_SIZE}" = "1" ]; then
   PERF_ARGS+=(--use-dynamic-batch-size)
fi
if [ -n "${MAX_TOKENS_PER_GPU}" ]; then
   PERF_ARGS+=(--max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}")
fi
if [ -n "${DATA_PAD_SIZE_MULTIPLIER}" ]; then
   PERF_ARGS+=(--data-pad-size-multiplier "${DATA_PAD_SIZE_MULTIPLIER}")
fi
if [ -n "${LOG_PROBS_CHUNK_SIZE}" ]; then
   PERF_ARGS+=(--log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}")
fi
if [ -n "${ROLLOUT_DATA_POSTPROCESS_PATH}" ]; then
   PERF_ARGS+=(--rollout-data-postprocess-path "${ROLLOUT_DATA_POSTPROCESS_PATH}")
fi

WANDB_ARGS=()
if [ "${USE_WANDB}" = "1" ]; then
   if [ "${WANDB_MODE}" = "online" ] && [ -z "${WANDB_API_KEY:-}" ]; then
      echo "WARNING: USE_WANDB=1 but WANDB_API_KEY is empty; export it before launch or set WANDB_MODE=offline."
   fi
   WANDB_ARGS+=(
      --use-wandb
      --wandb-mode "${WANDB_MODE}"
      --wandb-project "${WANDB_PROJECT}"
      --wandb-group "${WANDB_GROUP}"
      --wandb-dir "${WANDB_DIR}"
   )
   if [ "${WANDB_DISABLE_RANDOM_SUFFIX}" = "1" ]; then
      WANDB_ARGS+=(--disable-wandb-random-suffix)
   fi
   if [ -n "${WANDB_HOST}" ]; then
      WANDB_ARGS+=(--wandb-host "${WANDB_HOST}")
   fi
   if [ -n "${WANDB_TEAM}" ]; then
      WANDB_ARGS+=(--wandb-team "${WANDB_TEAM}")
   fi
   if [ -n "${WANDB_RUN_ID}" ]; then
      WANDB_ARGS+=(--wandb-run-id "${WANDB_RUN_ID}")
   fi
fi

# ADVANTAGE_ESTIMATOR: "grpo" (token-level IS, default) or "gspo" (sequence-level,
# length-normalized IS). EPS_CLIP must be retuned per estimator: GRPO uses ~0.2 on the
# per-token ratio; GSPO's ratio is a length-normalized geometric mean (variance much
# smaller, ~1), so it needs a far tighter clip (repo GSPO configs use 4e-4). Leaving
# EPS_CLIP at 0.2 under GSPO would never clip and forfeit GSPO's stability gains.
# EPS_CLIP_HIGH is optional; when unset, arguments.py defaults eps_clip_high = eps_clip.
GRPO_ARGS=(
   --advantage-estimator "${ADVANTAGE_ESTIMATOR:-grpo}"
   --kl-loss-coef "${KL_LOSS_COEF:-0.00}"
   --kl-loss-type low_var_kl
   --kl-coef "${KL_COEF:-0.00}"
   --entropy-coef "${ENTROPY_COEF:-0.00}"
   --eps-clip "${EPS_CLIP:-0.2}"
)
if [ -n "${EPS_CLIP_HIGH:-}" ]; then
   GRPO_ARGS+=(--eps-clip-high "${EPS_CLIP_HIGH}")
fi
# --kl-loss-coef alone does NOT engage the KL term; it is gated by --use-kl-loss
# (slime_plugins/harbor/train_loss.py + slime/backends/megatron_utils/loss.py).
# --use-kl-loss also triggers reference-model load (slime/utils/arguments.py:1682).
if [ "${USE_KL_LOSS:-0}" = "1" ]; then
   GRPO_ARGS+=(--use-kl-loss)
fi
if [ "${USE_ROLLOUT_LOGPROBS}" = "1" ]; then
   GRPO_ARGS+=(--use-rollout-logprobs)
fi
# Diagnostic train/inference log-prob mismatch metrics (train/tis, train/tis_abs,
# train/ois, train/train_rollout_logprob_abs_diff). --get-mismatch-metrics REQUIRES
# --custom-tis-function-path (arguments.py assert); point it at a record-only TIS fn
# (slime_plugins.harbor.mismatch_metrics.record_only_tis_function) to log the metrics
# WITHOUT truncating pg_loss. rollout_log_probs reach the batch even with
# USE_ROLLOUT_LOGPROBS=0 (Harbor sets sample.rollout_log_probs unconditionally), so the
# PPO ratio stays on the training-side old_log_probs — recipe unchanged.
if [ "${GET_MISMATCH_METRICS:-0}" = "1" ]; then
   GRPO_ARGS+=(--get-mismatch-metrics)
fi
if [ -n "${CUSTOM_TIS_FUNCTION_PATH:-}" ]; then
   GRPO_ARGS+=(--custom-tis-function-path "${CUSTOM_TIS_FUNCTION_PATH}")
fi
if [ "${KEEP_OLD_ACTOR}" = "1" ]; then
   GRPO_ARGS+=(--keep-old-actor)
fi
if [ "${USE_CUSTOM_NO_ENTROPY_LOSS}" = "1" ]; then
   GRPO_ARGS+=(--loss-type custom_loss --custom-loss-function-path "${CUSTOM_LOSS_FUNCTION_PATH}")
fi
# PPO only: when ADVANTAGE_ESTIMATOR=ppo, arguments.py derives use_critic=True and
# allocates an independent critic train-group that SHARES the actor's placement group
# (slime/ray/placement_group.py: result["critic"] = result["actor"], so no extra GPUs).
# The critic LM backbone loads from the same --load/--ref-load as the actor; its value
# head (output_size=1, model_provider.py) is auto-reinitialized on shape-mismatch
# (model.py:_reinitialize_critic_output_layer). NUM_CRITIC_ONLY_STEPS trains critic-only
# for the first N rollout steps to warm up value estimates before the actor starts
# (train_async.py: actor_trains_this_step = rollout_id >= num_critic_only_steps). Only
# emit the flag when explicitly set so non-PPO (grpo/gspo) runs are unaffected.
if [ -n "${NUM_CRITIC_ONLY_STEPS:-}" ]; then
   GRPO_ARGS+=(--num-critic-only-steps "${NUM_CRITIC_ONLY_STEPS}")
fi

# Warm-load the critic's OPTIMIZER / LR-scheduler / RNG, not just its value-head
# weights. By default placement_group.py:215-222 forces no_load_optim/no_load_rng/
# finetune on any warm critic (a "<load>_critic" sibling on disk), because the source
# run's LR schedule usually mismatches and OptimizerParamScheduler then asserts.
# That default is wrong when resuming a run INTO ITSELF (same num-rollout / batch /
# lr), where discarding the critic's Adam moments needlessly resets the value head's
# optimizer. Opt in with CRITIC_LOAD_OPTIM=1. Unset => unchanged behaviour.
if [ "${CRITIC_LOAD_OPTIM:-0}" = "1" ]; then
   GRPO_ARGS+=(--critic-load-optim)
fi

# Explicit starting rollout id. Needed when resuming from a checkpoint that was
# re-published with a "release" marker (weights-only cold start): arguments.py then sets
# start_rollout_id=0, so the wandb x-axis restarts at 0 and OVERWRITES the previous run's
# curve. placement_group.py only fills start_rollout_id when it is still None, so passing
# it explicitly wins. Set it to the step the checkpoint represents (e.g. 20 for iter_0000019)
# so the new curve continues where the old one stopped. Unset => unchanged behaviour.
if [ -n "${START_ROLLOUT_ID:-}" ]; then
   CKPT_ARGS+=(--start-rollout-id "${START_ROLLOUT_ID}")
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --use-precision-aware-optimizer
)
# Optimizer CPU offload: default ON (matches prior behavior). Set
# OPTIMIZER_CPU_OFFLOAD=0 to keep optimizer state on GPU instead — needed when
# host RAM is the bottleneck (e.g. 122B actor+critic offload + many daytona
# workers drove host_free to ~259GB and the OS OOM-killer SYSTEM_ERROR-killed
# the train actor right at the rollout->train handoff, while GPU stayed at
# ~44/95GB used). Only drop offload when GPU has headroom for the fp32 Adam state.
if [ "${OPTIMIZER_CPU_OFFLOAD:-1}" = "1" ]; then
   OPTIMIZER_ARGS+=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d)
fi
# Optional per-role Megatron config (e.g. a higher critic lr than the actor's --lr).
# When MEGATRON_CONFIG_PATH is set, slime's parse_megatron_role_args applies its
# role=critic overrides to the critic optimizer only; the actor keeps --lr above.
if [ -n "${MEGATRON_CONFIG_PATH:-}" ]; then
   OPTIMIZER_ARGS+=(--megatron-config-path "${MEGATRON_CONFIG_PATH}")
fi
# When RESUMING a checkpoint with a DIFFERENT --num-rollout than it was saved with,
# the loaded opt_param_scheduler's total-iteration count won't match, and Megatron's
# OptimizerParamScheduler.load_state_dict asserts (cls_value != sd_value). Setting
# --override-opt_param-scheduler makes it adopt the CURRENT args' schedule instead of
# the checkpoint's. Safe here because lr-decay-style=constant → lr stays LR regardless
# of total iters; only the (unused-at-constant) decay horizon changes.
if [ "${OVERRIDE_OPT_PARAM_SCHEDULER:-0}" = "1" ]; then
   OPTIMIZER_ARGS+=(--override-opt_param-scheduler)
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
)
if [ -n "${SGLANG_CONTEXT_LENGTH}" ]; then
   SGLANG_ARGS+=(--sglang-context-length "${SGLANG_CONTEXT_LENGTH}")
fi
if [ -n "${SGLANG_MAX_PREFILL_TOKENS}" ]; then
   SGLANG_ARGS+=(--sglang-max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS}")
fi
if [ -n "${SGLANG_CHUNKED_PREFILL_SIZE}" ]; then
   SGLANG_ARGS+=(--sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}")
fi
if [ -n "${EXTRA_SGLANG_ARGS}" ]; then
   # shellcheck disable=SC2206
   SGLANG_ARGS+=(${EXTRA_SGLANG_ARGS})
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)
if [ -n "${EXTRA_MISC_ARGS}" ]; then
   # shellcheck disable=SC2206
   MISC_ARGS+=(${EXTRA_MISC_ARGS})
fi

mkdir -p "${RAY_TEMP_DIR}"
mkdir -p "$(dirname "${WORKER_STOP_MARKER}")"
rm -f "${WORKER_STOP_MARKER}"
cleanup_workers_on_exit() {
  mkdir -p "$(dirname "${WORKER_STOP_MARKER}")" || true
  date -Iseconds > "${WORKER_STOP_MARKER}" || true
  log_event "wrote worker stop marker: ${WORKER_STOP_MARKER}"
}
trap cleanup_workers_on_exit EXIT
if [ "${USE_EXISTING_RAY_HEAD}" = "1" ]; then
  log_event "USE_EXISTING_RAY_HEAD=1; reusing existing Ray head at ${MASTER_ADDR}:${RAY_PORT}"
else
  DASHBOARD_ARGS=()
  if [ "${DIRECT_RUN}" = "1" ] || [ "${RAY_INCLUDE_DASHBOARD}" = "0" ]; then
    DASHBOARD_ARGS+=(--include-dashboard=false)
  else
    DASHBOARD_ARGS+=(--dashboard-host=0.0.0.0 --dashboard-port=8265)
  fi
  log_event "starting Ray head: master=${MASTER_ADDR}:${RAY_PORT}"
  ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --port "${RAY_PORT}" \
    --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
    --temp-dir "${RAY_TEMP_DIR}" \
    "${RAY_PORT_ARGS[@]}" \
    --disable-usage-stats \
    "${DASHBOARD_ARGS[@]}"
fi

RAY_CLUSTER_WAIT_STEPS=$(((RAY_CLUSTER_WAIT_SECONDS + 1) / 2))
for i in $(seq 1 "${RAY_CLUSTER_WAIT_STEPS}"); do
  NODE_COUNT=$(ray status 2>/dev/null | awk '/^Active:$/{flag=1; next} /^Pending:$/{flag=0} flag && /node_/{count++} END{print count+0}')
  if [ "${NODE_COUNT}" -ge "${EXPECTED_NODES}" ]; then
    log_event "ray cluster has ${NODE_COUNT}/${EXPECTED_NODES} nodes after $((i * 2))s"
    break
  fi
  if [ $((i % 10)) -eq 0 ]; then
    log_event "waiting for ray cluster... ${NODE_COUNT}/${EXPECTED_NODES} nodes ($((i * 2))s)"
  fi
  sleep 2
done

ray status || true
if [ "${RAY_CLUSTER_ONLY}" = "1" ]; then
  log_event "RAY_CLUSTER_ONLY=1; leaving Ray head alive for external stability checks"
  tail -f /dev/null
fi

RAY_NODE_IPS=$(
  python3 -c 'import ray; ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR"); print(",".join(sorted({n["NodeManagerAddress"] for n in ray.nodes() if n.get("Alive", True)})))' 2>/dev/null || true
)
JOB_NO_PROXY="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"
if [ -n "${RAY_NODE_IPS}" ]; then
  JOB_NO_PROXY="${JOB_NO_PROXY},${RAY_NODE_IPS}"
fi
if [ -n "${no_proxy:-}" ]; then
  JOB_NO_PROXY="${JOB_NO_PROXY},${no_proxy}"
fi
export no_proxy="${JOB_NO_PROXY}"
export NO_PROXY="${JOB_NO_PROXY}"
export GLOO_SOCKET_IFNAME="${SOCKET_IFNAME}"
export TP_SOCKET_IFNAME="${SOCKET_IFNAME}"
export NCCL_SOCKET_IFNAME="${SOCKET_IFNAME}"
export MASTER_ADDR
export PYTHONPATH="/root/Megatron-LM:${HARBOR_SRC}:${SLIME_DIR}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
export SLIME_SKIP_NUMA_AFFINITY=1
export HARBOR_MAX_TRAIN_TOTAL_TOKENS
export WANDB_MODE
export DAYTONA_API_KEY="${DAYTONA_API_KEY:-}"
export DAYTONA_JWT_TOKEN="${DAYTONA_JWT_TOKEN:-}"
export DAYTONA_ORGANIZATION_ID="${DAYTONA_ORGANIZATION_ID:-}"
if [ "${DIRECT_RUN}" = "1" ]; then
  export RAY_ADDRESS="${MASTER_ADDR}:${RAY_PORT}"
fi

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "http_proxy": "${http_proxy:-}",
    "https_proxy": "${https_proxy:-}",
    "ftp_proxy": "${ftp_proxy:-}",
    "HTTP_PROXY": "${HTTP_PROXY:-}",
    "HTTPS_PROXY": "${HTTPS_PROXY:-}",
    "FTP_PROXY": "${FTP_PROXY:-}",
    "no_proxy": "${JOB_NO_PROXY}",
    "NO_PROXY": "${JOB_NO_PROXY}",
    "GLOO_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "NCCL_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": "/root/Megatron-LM:${HARBOR_SRC}:${SLIME_DIR}:${PYTHONPATH:-}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "${PYTORCH_CUDA_ALLOC_CONF}",
    "PYTORCH_ALLOC_CONF": "${PYTORCH_ALLOC_CONF}",
    "NVTE_BATCH_MHA_P2P_COMM": "${NVTE_BATCH_MHA_P2P_COMM:-0}",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "SLIME_SKIP_NUMA_AFFINITY": "1",
    "SLIME_NEW_GROUP_TIMEOUT_MINUTES": "${SLIME_NEW_GROUP_TIMEOUT_MINUTES:-}",
    "SLIME_NCCL_GROUP_TIMEOUT_MINUTES": "${SLIME_NCCL_GROUP_TIMEOUT_MINUTES:-}",
    "SLIME_ENGINE_LOCK_TIMEOUT_S": "${SLIME_ENGINE_LOCK_TIMEOUT_S:-}",
    "TRITON_CACHE_DIR": "${TRITON_CACHE_DIR:-}",
    "FLA_INTRACARD_CP": "${FLA_INTRACARD_CP:-}",
    "SLIME_CP_SKIP_SHAPE_CHECK": "${SLIME_CP_SKIP_SHAPE_CHECK:-}",
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "${TORCH_NCCL_TRACE_BUFFER_SIZE:-}",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "${TORCH_NCCL_DUMP_ON_TIMEOUT:-}",
    "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": "${TORCH_NCCL_DEBUG_INFO_TEMP_FILE:-}",
    "NCCL_DEBUG": "${NCCL_DEBUG:-}",
    "NCCL_DEBUG_SUBSYS": "${NCCL_DEBUG_SUBSYS:-}",
    "APPTAINER_CACHEDIR": "${APPTAINER_CACHEDIR}",
    "DAYTONA_API_KEY": "${DAYTONA_API_KEY:-}",
    "DAYTONA_JWT_TOKEN": "${DAYTONA_JWT_TOKEN:-}",
    "DAYTONA_ORGANIZATION_ID": "${DAYTONA_ORGANIZATION_ID:-}",
    "HARBOR_MAX_TRAIN_TOTAL_TOKENS": "${HARBOR_MAX_TRAIN_TOTAL_TOKENS}",
    "HARBOR_LENGTH_REWARD_LAMBDA": "${HARBOR_LENGTH_REWARD_LAMBDA:-}",
    "HARBOR_PARTIAL_REWARD_MAX": "${HARBOR_PARTIAL_REWARD_MAX:-}",
    "HARBOR_REPORT_ENTROPY": "${HARBOR_REPORT_ENTROPY:-1}",
    "WANDB_API_KEY": "${WANDB_API_KEY:-}",
    "WANDB_MODE": "${WANDB_MODE}"
  }
}
EOF_JSON
)

TRAIN_ENTRYPOINT=${TRAIN_ENTRYPOINT:-train_async.py}
TRAIN_CMD=(
   python3 "${TRAIN_ENTRYPOINT}"
   --actor-num-nodes "${ACTOR_NUM_NODES}"
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${HARBOR_ROLLOUT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${GRPO_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${MISC_ARGS[@]}"
   "${DEBUG_DUMP_ARGS[@]}"
   "${WANDB_ARGS[@]}"
)

_XTRACE_WAS_ON=0
case "$-" in
   *x*) _XTRACE_WAS_ON=1; set +x ;;
esac
if [ "${DIRECT_RUN}" = "1" ]; then
   log_event "running ${TRAIN_ENTRYPOINT} directly with USE_WANDB=${USE_WANDB}, WANDB_PROJECT=${WANDB_PROJECT}, WANDB_GROUP=${WANDB_GROUP}"
   "${TRAIN_CMD[@]}"
else
   for i in $(seq 1 60); do
     if curl -sf --max-time 3 "http://127.0.0.1:8265/api/jobs/" >/dev/null 2>&1; then
       log_event "ray job agent ready after ${i}s"
       break
     fi
     sleep 1
   done
   sleep 10

   log_event "submitting Ray job with USE_WANDB=${USE_WANDB}, WANDB_PROJECT=${WANDB_PROJECT}, WANDB_GROUP=${WANDB_GROUP}"
   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- "${TRAIN_CMD[@]}"
fi
if [ "${_XTRACE_WAS_ON}" = "1" ]; then
   set -x
fi
