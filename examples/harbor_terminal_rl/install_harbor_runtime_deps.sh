#!/bin/bash

set -ex

export PIP_BREAK_SYSTEM_PACKAGES="${PIP_BREAK_SYSTEM_PACKAGES:-1}"
export PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK:-1}"
INSTALL_DAYTONA="${INSTALL_DAYTONA:-0}"

pip_install_clean() {
  (
    cd /tmp
    unset PYTHONPATH
    python3 -m pip install --break-system-packages --no-cache-dir "$@"
  )
}

# Runtime deps needed when Harbor is used from source via PYTHONPATH.
# Keep openai pinned to the SGLang image requirement; the Harbor slime_sglang
# path does not use LiteLLM to send model requests.
pip_install_clean \
  "shortuuid>=1.0.13" \
  "toml>=0.10.2" \
  "tenacity>=9.1.2" \
  "litellm>=1.83.14" \
  "dirhash>=0.5.0" \
  "claude-agent-sdk>=0.1.17" \
  "supabase>=2.28.2"

pip_install_clean "openai==2.6.1"

if [ "${INSTALL_DAYTONA}" = "1" ]; then
  pip_install_clean "daytona==0.168.0"
fi
