#!/bin/bash

# =============================================================================
# setup_fresh_container_env.sh
#
# Bring a FRESH slimerl docker container up to the exact state needed to run the
# 96-GPU Qwen3.5-27B Harbor fully-async run (and the other Harbor terminal RL
# runs). Reusable after every docker image bump.
#
# WHAT THE IMAGE ALREADY HAS (do NOT reinstall): the heavy stack is baked into
# the slimerl image -> torch+cu129, sglang, megatron, slime, flash-attn,
# transformer-engine, apex, mbridge, flash-linear-attention, sglang_router+slime,
# torch_memory_saver, sgl-deep-gemm. build_conda.sh is NOT needed on this image.
#
# WHAT THE IMAGE DROPS (this script installs): the two per-node runtime layers
# that were never baked into the image and must be reinstalled every bump:
#   1. apptainer/singularity  (Harbor runs each terminal task in a SIF sandbox)
#   2. Harbor python runtime deps: tenacity, litellm, shortuuid, toml, dirhash,
#      claude-agent-sdk, supabase  (+ openai pinned to 2.6.1 for SGLang)
#
# Usage:
#   bash setup_fresh_container_env.sh            # set up THIS node + verify
#   bash setup_fresh_container_env.sh fanout     # ssh to every node in
#                                                # $NODE_IP_LIST and set up each
#   VERIFY_ONLY=1 bash setup_fresh_container_env.sh   # skip installs, only check
#
# Notes:
#  - Idempotent and safe to re-run (apt + pip skip already-satisfied work).
#  - Run from a path on the SHARED FS so `fanout` can invoke the same file on
#    every node (this script lives under the repo, which is shared).
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

CODE_ROOT="${CODE_ROOT:-/path/to/workspace/code}"
SLIME_DIR="${SLIME_DIR:-${CODE_ROOT}/slime-v0.3-harbor}"
HARBOR_SRC="${HARBOR_SRC:-${CODE_ROOT}/harbor/src}"
APT_PROXY="${APT_PROXY:-http://star-proxy.oa.com:3128}"

INSTALL_APPTAINER="${SCRIPT_DIR}/install_apptainer_ubuntu.sh"
INSTALL_HARBOR_DEPS="${SCRIPT_DIR}/install_harbor_runtime_deps.sh"

# A SIF used only for the apptainer smoke test (any sif in the cache works).
SIF_CACHE_DIR="${SIF_CACHE_DIR:-/path/to/workspace/harbor_sif_cache_tb2_full_harbor_runtime}"

log()  { printf '[setup %s] %s\n' "$(hostname -I | awk '{print $1}')" "$*"; }
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------------------
# fanout mode: run this same script on every node in $NODE_IP_LIST
# ---------------------------------------------------------------------------
if [ "${1:-}" = "fanout" ]; then
  if [ -z "${NODE_IP_LIST:-}" ]; then
    echo "ERROR: fanout needs \$NODE_IP_LIST (comma-sep 'ip:gpus', first = launcher)." >&2
    exit 1
  fi
  IPS=$(printf '%s\n' "${NODE_IP_LIST}" | tr ',' '\n' | cut -d: -f1)
  SELF="$(printf '%s\n' "${NODE_IP_LIST}" | cut -d',' -f1 | cut -d: -f1)"
  log "fanout to $(echo "$IPS" | wc -l) nodes; self=${SELF}"

  # this node first (canary), so a broken image fails fast before fan-out
  log "=== local node (${SELF}) ==="
  bash "${BASH_SOURCE[0]}" || { echo "local setup FAILED; aborting fanout" >&2; exit 1; }

  declare -A RC
  for ip in $IPS; do
    [ "$ip" = "$SELF" ] && continue
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$ip" \
      "bash --noprofile --norc -c 'CODE_ROOT=${CODE_ROOT} bash ${BASH_SOURCE[0]} > /tmp/env_setup.log 2>&1; echo EXIT=\$? > /tmp/env_setup.done'" &
  done
  wait
  echo
  log "=== fanout results ==="
  FAIL=0
  for ip in $IPS; do
    [ "$ip" = "$SELF" ] && { printf '%-16s local (see above)\n' "$ip"; continue; }
    r=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$ip" \
          "bash --noprofile --norc -c 'cat /tmp/env_setup.done 2>/dev/null'" 2>/dev/null | grep -aoE 'EXIT=[0-9]+')
    printf '%-16s %s\n' "$ip" "${r:-NO_RESULT}"
    case "$r" in EXIT=0) ;; *) FAIL=1 ;; esac
  done
  [ "$FAIL" = 0 ] && log "ALL NODES OK" || { log "SOME NODES FAILED — check /tmp/env_setup.log on them"; exit 1; }
  exit 0
fi

# ---------------------------------------------------------------------------
# 0. Sanity: confirm the heavy stack is present (i.e. this IS the slimerl image)
# ---------------------------------------------------------------------------
log "checking baked-in python stack (torch/sglang/megatron/slime/native deps)"
STACK_OK=$(python3 - <<'PY'
mods = ["torch","sglang","megatron","slime","flash_attn","transformer_engine",
        "mbridge","apex","fla","sglang_router","torch_memory_saver","deep_gemm"]
missing=[]
import importlib.util as u
for m in mods:
    if u.find_spec(m) is None: missing.append(m)
print("MISSING:"+",".join(missing) if missing else "ALLPRESENT")
PY
)
if [ "${STACK_OK}" != "ALLPRESENT" ]; then
  bad "baked-in stack incomplete (${STACK_OK})."
  echo "  -> This is probably NOT the slimerl image. You likely need build_conda.sh." >&2
  echo "  -> Set FORCE=1 to continue anyway." >&2
  [ "${FORCE:-0}" = "1" ] || exit 1
else
  ok "baked-in stack present (no build_conda.sh needed)"
fi

# ---------------------------------------------------------------------------
# 1 + 2. Install apptainer + Harbor runtime deps (skip if VERIFY_ONLY)
# ---------------------------------------------------------------------------
if [ "${VERIFY_ONLY:-0}" != "1" ]; then
  log "installing apptainer/singularity"
  APT_PROXY="${APT_PROXY}" bash "${INSTALL_APPTAINER}" >/tmp/setup_apptainer.log 2>&1 \
    && ok "apptainer install done" || { bad "apptainer install (see /tmp/setup_apptainer.log)"; tail -5 /tmp/setup_apptainer.log; }

  log "installing Harbor runtime deps (tenacity/litellm/shortuuid/toml/dirhash/claude-agent-sdk/supabase + openai==2.6.1)"
  bash "${INSTALL_HARBOR_DEPS}" >/tmp/setup_harbor_deps.log 2>&1 \
    && ok "harbor deps install done" || { bad "harbor deps install (see /tmp/setup_harbor_deps.log)"; tail -5 /tmp/setup_harbor_deps.log; }
fi

# ---------------------------------------------------------------------------
# 3. Verify
# ---------------------------------------------------------------------------
log "verifying"
RC=0

# 3a. apptainer present
if command -v singularity >/dev/null 2>&1 || command -v apptainer >/dev/null 2>&1; then
  ok "singularity/apptainer: $( (apptainer --version 2>/dev/null || singularity --version 2>/dev/null) )"
else
  bad "no singularity/apptainer in PATH"; RC=1
fi

# 3b. apptainer SIF smoke test (with Harbor's exact no-mount flags)
SIF=$(ls "${SIF_CACHE_DIR}"/*.sif 2>/dev/null | head -1)
if [ -n "${SIF}" ]; then
  if (cd /tmp && timeout 120 singularity exec --no-mount home --no-mount tmp \
        --no-mount bind-paths --no-mount proc --writable-tmpfs --containall --pid \
        "${SIF}" echo smoke_ok 2>/dev/null | grep -q smoke_ok); then
    ok "SIF smoke test passed"
  else
    bad "SIF smoke test failed"; RC=1
  fi
else
  log "  (no SIF in ${SIF_CACHE_DIR}; skipping smoke test)"
fi

# 3c. THE decisive check: the trial-worker import chain that broke on a fresh image
CHK=/tmp/_check_trial_import.py
cat > "${CHK}" <<'PY'
import sys
try:
    import slime_plugins.harbor.rollout  # noqa
    from slime_plugins.harbor.import_utils import ensure_harbor_distribution_metadata
    ensure_harbor_distribution_metadata()  # patches importlib.metadata so source-tree harbor resolves
    import importlib.metadata as m
    import harbor.trial.trial  # noqa  <- ImportError'd here (tenacity) on a fresh image
    from harbor.environments.singularity.singularity import SingularityEnvironment  # noqa
    print("TRIAL_IMPORT_OK harbor=" + m.version("harbor"))
except Exception as e:
    print("TRIAL_IMPORT_FAIL " + type(e).__name__ + ": " + str(e)[:200]); sys.exit(1)
PY
OUT=$(cd /tmp && PYTHONPATH="${HARBOR_SRC}:${SLIME_DIR}" python3 "${CHK}" 2>/dev/null | grep -aE 'TRIAL_IMPORT')
case "${OUT}" in
  TRIAL_IMPORT_OK*) ok "trial-worker import chain: ${OUT}" ;;
  *) bad "trial-worker import chain: ${OUT:-<no output>}"; RC=1 ;;
esac

echo
[ "${RC}" = 0 ] && log "NODE READY ✅" || log "NODE NOT READY ❌"
exit "${RC}"
