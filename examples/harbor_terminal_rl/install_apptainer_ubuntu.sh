#!/usr/bin/env bash

# Install Apptainer/Singularity on Ubuntu pods and prepare Harbor's fakeroot mode.
# Optional:
#   APT_PROXY=http://star-proxy.oa.com:3128
#   SIF_SMOKE_TEST=/path/to/ubuntu-24-04_latest.sif

set -euo pipefail

APT_PROXY="${APT_PROXY:-http://star-proxy.oa.com:3128}"
SIF_SMOKE_TEST="${SIF_SMOKE_TEST:-}"

if [ "$(id -u)" -eq 0 ]; then
  SUDO=()
else
  SUDO=(sudo)
fi

APT_OPTS=()
if [ -n "${APT_PROXY}" ]; then
  APT_OPTS+=(
    -o "Acquire::http::Proxy=${APT_PROXY}"
    -o "Acquire::https::Proxy=${APT_PROXY}"
  )
  export http_proxy="${APT_PROXY}"
  export https_proxy="${APT_PROXY}"
  export HTTP_PROXY="${APT_PROXY}"
  export HTTPS_PROXY="${APT_PROXY}"
fi

export DEBIAN_FRONTEND=noninteractive

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This helper expects an Ubuntu/Debian node with apt-get." >&2
  exit 1
fi

sanitize_ld_so_preload() {
  local preload_file="/etc/ld.so.preload"
  if [ ! -f "${preload_file}" ]; then
    return
  fi

  # Some Jizhi worker pods carry this Tencent preload entry, but Singularity
  # treats the literal source path as a bind mount and fails if it is absent:
  #   mount /$LIB/libtdsp_block.so->/$LIB/libtdsp_block.so error
  if grep -q '^/\$LIB/libtdsp_block\.so$' "${preload_file}"; then
    local backup="${preload_file}.slime-backup.$(date +%Y%m%d%H%M%S)"
    "${SUDO[@]}" cp "${preload_file}" "${backup}"
    "${SUDO[@]}" sed -i '/^\/\$LIB\/libtdsp_block\.so$/d' "${preload_file}"
    echo "Removed missing libtdsp preload entry from ${preload_file}; backup: ${backup}"
  fi
}

"${SUDO[@]}" apt-get "${APT_OPTS[@]}" update

if ! command -v add-apt-repository >/dev/null 2>&1; then
  "${SUDO[@]}" apt-get "${APT_OPTS[@]}" install -y software-properties-common
fi

if ! grep -R "ppa.launchpadcontent.net/apptainer/ppa" \
  /etc/apt/sources.list /etc/apt/sources.list.d >/dev/null 2>&1; then
  "${SUDO[@]}" env \
    http_proxy="${http_proxy:-}" \
    https_proxy="${https_proxy:-}" \
    HTTP_PROXY="${HTTP_PROXY:-}" \
    HTTPS_PROXY="${HTTPS_PROXY:-}" \
    add-apt-repository -y ppa:apptainer/ppa
fi

"${SUDO[@]}" apt-get "${APT_OPTS[@]}" update

if apt-cache policy apptainer | grep -q 'Candidate: (none)'; then
  # Ubuntu 24.04's universe package exposes /usr/bin/singularity. It is enough
  # for Harbor if the FUSE helpers below are present.
  "${SUDO[@]}" apt-get "${APT_OPTS[@]}" install -y uidmap singularity-container
else
  "${SUDO[@]}" apt-get "${APT_OPTS[@]}" install -y uidmap apptainer
fi

"${SUDO[@]}" apt-get "${APT_OPTS[@]}" install -y fuse squashfuse squashfs-tools

if ! command -v fusermount >/dev/null 2>&1 && command -v fusermount3 >/dev/null 2>&1; then
  "${SUDO[@]}" ln -sf "$(command -v fusermount3)" /usr/local/bin/fusermount
fi

sanitize_ld_so_preload

if ! grep -q "^root:" /etc/subuid 2>/dev/null; then
  "${SUDO[@]}" usermod --add-subuids 100000-165535 root
fi

if ! grep -q "^root:" /etc/subgid 2>/dev/null; then
  "${SUDO[@]}" usermod --add-subgids 100000-165535 root
fi

if command -v apptainer >/dev/null 2>&1; then
  apptainer --version
fi
singularity --version
command -v fusermount
command -v squashfuse
grep "^root:" /etc/subuid
grep "^root:" /etc/subgid
cat /etc/ld.so.preload 2>/dev/null || true

if [ -n "${SIF_SMOKE_TEST}" ]; then
  (
    cd /tmp
    singularity exec \
    --no-mount home \
    --no-mount tmp \
    --no-mount bind-paths \
    --no-mount proc \
    --writable-tmpfs \
    --fakeroot \
    --containall \
    --pid \
    "${SIF_SMOKE_TEST}" \
    echo harbor_exact_no_mount_ok
  )
fi
