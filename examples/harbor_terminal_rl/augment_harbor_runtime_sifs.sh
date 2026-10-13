#!/bin/bash

set -euo pipefail

# Add Harbor runtime dependencies to prebuilt terminal-bench SIF images without
# modifying the original cache. This creates an augmented cache with the same
# SIF filenames so Harbor's docker-image-to-SIF lookup keeps working.

SRC_CACHE=${SRC_CACHE:-/path/to/workspace/harbor_sif_cache_tb2_full}
DST_CACHE=${DST_CACHE:-/path/to/workspace/harbor_sif_cache_tb2_full_harbor_runtime}
WORK_DIR=${WORK_DIR:-/tmp/slime-harbor-sif-augment}
FORCE=${FORCE:-0}
MAX_IMAGES=${MAX_IMAGES:-0}
TRUST_EXISTING=${TRUST_EXISTING:-0}

if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN=${APPTAINER_BIN:-$(command -v apptainer)}
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN=${APPTAINER_BIN:-$(command -v singularity)}
else
  echo "ERROR: neither apptainer nor singularity found in PATH" >&2
  exit 1
fi

if [ "$(id -u)" = "0" ]; then
  DEFAULT_BUILD_FLAGS=()
else
  DEFAULT_BUILD_FLAGS=(--fakeroot)
fi
read -r -a EXTRA_BUILD_FLAGS <<< "${APPTAINER_BUILD_FLAGS:-}"
if [ "${#EXTRA_BUILD_FLAGS[@]}" -gt 0 ]; then
  BUILD_FLAGS=("${EXTRA_BUILD_FLAGS[@]}")
else
  BUILD_FLAGS=("${DEFAULT_BUILD_FLAGS[@]}")
fi

mkdir -p "${DST_CACHE}" "${WORK_DIR}"

validate_sif() {
  local sif_path=$1
  "${APPTAINER_BIN}" exec "${sif_path}" python3 - <<'PY'
import fastapi
import uvicorn
print("python runtime ok", uvicorn.__version__, fastapi.__version__)
PY
  "${APPTAINER_BIN}" exec "${sif_path}" sh -lc 'command -v tmux && tmux -V'
}

build_one() {
  local src=$1
  local name dst def tmp_out
  name=$(basename "${src}")
  dst="${DST_CACHE%/}/${name}"
  def="${WORK_DIR%/}/${name%.sif}.def"
  tmp_out="${WORK_DIR%/}/${name%.sif}.runtime.sif"

  if [ -f "${dst}" ] && [ "${FORCE}" != "1" ]; then
    if [ "${TRUST_EXISTING}" = "1" ]; then
      echo "[skip] ${name} already exists (TRUST_EXISTING=1)"
      return 0
    fi
    if validate_sif "${dst}" >/dev/null 2>&1; then
      echo "[skip] ${name} already has Harbor runtime deps"
      return 0
    fi
    echo "[rebuild] ${name} exists but failed validation"
  fi

  cat > "${def}" <<EOF_DEF
Bootstrap: localimage
From: ${src}

%post
    set -eux
    export DEBIAN_FRONTEND=noninteractive
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export http_proxy="\${http_proxy:-http://star-proxy.oa.com:3128}"
    export https_proxy="\${https_proxy:-http://star-proxy.oa.com:3128}"
    export HTTP_PROXY="\${HTTP_PROXY:-\${http_proxy}}"
    export HTTPS_PROXY="\${HTTPS_PROXY:-\${https_proxy}}"
    export no_proxy="\${no_proxy:-localhost,127.0.0.1,mirror.ccs.tencentyun.com}"

    if command -v apt-get >/dev/null 2>&1; then
        mkdir -p /etc/apt/apt.conf.d
        printf '%s\n' \\
            "Acquire::http::Proxy \"\${http_proxy}\";" \\
            "Acquire::https::Proxy \"\${https_proxy}\";" \\
            "Acquire::Retries \"3\";" \\
            "Acquire::http::Timeout \"30\";" \\
            "Acquire::https::Timeout \"30\";" \\
            > /etc/apt/apt.conf.d/99slime-harbor-proxy || true
        apt-get update
        apt-get install -y --no-install-recommends ca-certificates tmux
        if ! command -v python3 >/dev/null 2>&1; then
            apt-get install -y --no-install-recommends python3
        fi
        if command -v python3 >/dev/null 2>&1 && ! python3 - <<'PY' >/dev/null 2>&1; then
import sys
if sys.version_info < (3, 12):
    import distutils.cmd
PY
            apt-get install -y --no-install-recommends python3-distutils || true
        fi
        rm -rf /var/lib/apt/lists/*
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache ca-certificates tmux
        if ! command -v python3 >/dev/null 2>&1; then
            apk add --no-cache python3
        fi
    elif command -v dnf >/dev/null 2>&1; then
        dnf install -y ca-certificates tmux
        if ! command -v python3 >/dev/null 2>&1; then
            dnf install -y python3
        fi
        dnf clean all
    elif command -v yum >/dev/null 2>&1; then
        yum install -y ca-certificates tmux
        if ! command -v python3 >/dev/null 2>&1; then
            yum install -y python3
        fi
        yum clean all
    fi

    PY="\$(command -v python3 || true)"
    if [ -z "\${PY}" ]; then
        echo "python3 is required for Harbor runtime deps" >&2
        exit 1
    fi
    if ! "\${PY}" -m pip --version >/dev/null 2>&1; then
        "\${PY}" - <<'PY'
import socket
import sys
import urllib.request
socket.setdefaulttimeout(120)
if sys.version_info < (3, 10):
    url = f"https://bootstrap.pypa.io/pip/{sys.version_info.major}.{sys.version_info.minor}/get-pip.py"
else:
    url = "https://bootstrap.pypa.io/get-pip.py"
urllib.request.urlretrieve(url, "/tmp/get-pip.py")
PY
        "\${PY}" /tmp/get-pip.py --quiet --break-system-packages || "\${PY}" /tmp/get-pip.py --quiet
    fi
    "\${PY}" -m pip install --no-cache-dir --break-system-packages --ignore-installed uvicorn fastapi \\
        || "\${PY}" -m pip install --no-cache-dir --break-system-packages uvicorn fastapi \\
        || "\${PY}" -m pip install --no-cache-dir uvicorn fastapi
    "\${PY}" - <<'PY'
import fastapi
import uvicorn
print("installed Harbor runtime deps", uvicorn.__version__, fastapi.__version__)
PY
    command -v tmux

%labels
    org.harbor.runtime-deps true
    org.harbor.runtime-deps.packages "uvicorn fastapi tmux"
EOF_DEF

  rm -f "${tmp_out}"
  echo "[build] ${name}"
  "${APPTAINER_BIN}" build "${BUILD_FLAGS[@]}" "${tmp_out}" "${def}"

  echo "[validate] ${name}"
  validate_sif "${tmp_out}"

  mv -f "${tmp_out}" "${dst}"
  echo "[done] ${dst}"
}

declare -a images=()
if [ "$#" -gt 0 ]; then
  for item in "$@"; do
    if [[ "${item}" == */* ]]; then
      images+=("${item}")
    else
      images+=("${SRC_CACHE%/}/${item}")
    fi
  done
else
  while IFS= read -r path; do
    images+=("${path}")
  done < <(find "${SRC_CACHE}" -maxdepth 1 -type f -name '*.sif' | sort)
fi

count=0
for src in "${images[@]}"; do
  if [ ! -f "${src}" ]; then
    echo "ERROR: SIF not found: ${src}" >&2
    exit 1
  fi
  build_one "${src}"
  count=$((count + 1))
  if [ "${MAX_IMAGES}" -gt 0 ] && [ "${count}" -ge "${MAX_IMAGES}" ]; then
    break
  fi
done

echo "Augmented ${count} image(s) into ${DST_CACHE}"
