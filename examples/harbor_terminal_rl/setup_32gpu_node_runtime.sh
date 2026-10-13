#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CODE_ROOT="${CODE_ROOT:-/path/to/workspace/code}"
SLIME_DIR="${SLIME_DIR:-${CODE_ROOT}/slime}"
HARBOR_SRC="${HARBOR_SRC:-${CODE_ROOT}/harbor/src}"
APT_PROXY="${APT_PROXY:-http://star-proxy.oa.com:3128}"
SIF_SMOKE_TEST="${SIF_SMOKE_TEST:-/path/to/workspace/harbor_sif_cache_tb2_full_harbor_runtime/alexgshaw_regex-log_20251031.sif}"
INSTALL_APPTAINER="${INSTALL_APPTAINER:-1}"

backup_stale_slime_editable() {
  local site="/usr/local/lib/python3.12/dist-packages"
  local backup="/tmp/slime_editable_backup_32gpu_$(date +%Y%m%d%H%M%S)"
  local moved=0
  local path
  local pth
  local tmp_pth

  for path in \
    "${site}/__editable__.slime-0.2.4.pth" \
    "${site}/__editable___slime_0_2_4_finder.py" \
    "${site}/slime-0.2.4.dist-info" \
    "${site}/slime.egg-link"; do
    if [ -e "${path}" ]; then
      mkdir -p "${backup}"
      mv "${path}" "${backup}/"
      moved=1
    fi
  done

  shopt -s nullglob
  for path in "${site}"/*slime*.egg-link; do
    mkdir -p "${backup}"
    mv "${path}" "${backup}/"
    moved=1
  done
  shopt -u nullglob

  # Some old nodes carry an easy-install.pth entry pointing at /cq_1/.../slime.
  # pip scans legacy editables before doing anything else and crashes if that
  # path is absent, so strip only stale slime workspace lines.
  for pth in "${site}/easy-install.pth" "${site}/easy_install.pth"; do
    if [ -f "${pth}" ] && grep -qE '/path/to/workspace/code/slime' "${pth}"; then
      mkdir -p "${backup}"
      cp "${pth}" "${backup}/$(basename "${pth}").bak"
      tmp_pth="${pth}.tmp.$$"
      grep -vE '/path/to/workspace/code/slime' "${pth}" > "${tmp_pth}"
      mv "${tmp_pth}" "${pth}"
      moved=1
    fi
  done

  if [ "${moved}" = "1" ]; then
    echo "Backed up stale slime editable metadata to ${backup}"
  fi
}

cd "${SLIME_DIR}"
backup_stale_slime_editable

if [ "${INSTALL_APPTAINER}" = "1" ]; then
  APT_PROXY="${APT_PROXY}" \
  SIF_SMOKE_TEST="${SIF_SMOKE_TEST}" \
  bash "${SCRIPT_DIR}/install_apptainer_ubuntu.sh"
fi

bash "${SCRIPT_DIR}/install_harbor_runtime_deps.sh"

export PYTHONPATH="${SLIME_DIR}:${HARBOR_SRC}:${PYTHONPATH:-}"
python3 - <<'PY'
import openai
from slime_plugins.harbor.import_utils import ensure_harbor_distribution_metadata

ensure_harbor_distribution_metadata()
import slime.backends.sglang_utils
import slime_plugins.harbor.rollout
import slime_plugins.harbor.singularity_env
import harbor.trial.trial
import harbor.agents.terminus_2.terminus_2
import tenacity, toml, shortuuid, litellm

try:
    import daytona
except ImportError:
    daytona = None

print("runtime_ok")
print("openai", openai.__version__)
print("daytona", getattr(daytona, "__version__", None) if daytona else "not_installed")
PY

echo "setup_32gpu_node_runtime_done"
