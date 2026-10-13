#!/bin/bash
# Harbor Singularity bootstrap tuned for Jizhi/Apptainer pods.
# First arg is WORKDIR (container cwd), rest are server args.

export WORKDIR="${1:-/app}"; shift
export HARBOR_STAGING="/staging/env_files"
mkdir -p "$WORKDIR"

export DEBIAN_FRONTEND=noninteractive
export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"
export https_proxy="${https_proxy:-http://star-proxy.oa.com:3128}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,mirror.ccs.tencentyun.com}"

_harbor_log() {
  printf '[harbor][%s] %s\n' "$(date -Iseconds)" "$*" >&2
}

_HARBOR_APT_READY=0
_prepare_apt_once() {
  if [ "$_HARBOR_APT_READY" = "1" ] || ! command -v apt-get >/dev/null 2>&1; then
    return 0
  fi
  mkdir -p /etc/apt/apt.conf.d
  printf '%s\n' \
    "Acquire::http::Proxy \"$http_proxy\";" \
    "Acquire::https::Proxy \"$https_proxy\";" \
    > /etc/apt/apt.conf.d/99slime-harbor-proxy || true
  _harbor_log "apt-get update starting"
  _apt_start=$SECONDS
  apt-get update -qq || _harbor_log "apt-get update failed; continuing"
  _harbor_log "apt-get update finished in $((SECONDS - _apt_start))s"
  _HARBOR_APT_READY=1
}

if ! command -v python3 >/dev/null 2>&1; then
  _harbor_log "python3 not found, installing"
  if command -v apt-get >/dev/null 2>&1; then
    _prepare_apt_once
    apt-get install -y -qq python3 python3-pip || _harbor_log "apt-get python3 install failed; continuing"
  elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache python3 py3-pip || _harbor_log "apk python3 install failed; continuing"
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y python3 python3-pip || _harbor_log "dnf python3 install failed; continuing"
  elif command -v yum >/dev/null 2>&1; then
    yum install -y python3 python3-pip || _harbor_log "yum python3 install failed; continuing"
  fi
fi

_HARBOR_PY="$(command -v python3 || true)"
if [ -z "$_HARBOR_PY" ]; then
  _harbor_log "FATAL: cannot find python3"
  exit 1
fi

if ! "$_HARBOR_PY" -m pip --version >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1 && ! "$_HARBOR_PY" - <<'PY' >/dev/null 2>&1; then
import sys
if sys.version_info < (3, 12):
    import distutils.cmd
PY
    _harbor_log "Installing python3-distutils via apt-get for get-pip.py"
    _prepare_apt_once
    apt-get install -y -qq python3-distutils || _harbor_log "apt-get python3-distutils install failed; continuing"
  fi
  _harbor_log "Bootstrapping pip via get-pip.py"
  "$_HARBOR_PY" - <<'PY'
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
  "$_HARBOR_PY" /tmp/get-pip.py --quiet --break-system-packages || true
fi

if ! "$_HARBOR_PY" -m pip --version >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    _harbor_log "Installing python3-pip via apt-get"
    _prepare_apt_once
    apt-get install -y -qq python3-pip || _harbor_log "apt-get python3-pip install failed; continuing"
  fi
fi

if ! "$_HARBOR_PY" -m pip --version >/dev/null 2>&1; then
  _harbor_log "FATAL: cannot bootstrap pip"
  exit 1
fi

if ! "$_HARBOR_PY" -c "import uvicorn; import fastapi" 2>/dev/null; then
  _harbor_log "Installing server dependencies (uvicorn/fastapi)"
  _pip_start=$SECONDS
  "$_HARBOR_PY" -m pip install --break-system-packages uvicorn fastapi \
    || "$_HARBOR_PY" -m pip install uvicorn fastapi \
    || {
      _harbor_log "FATAL: failed to install uvicorn/fastapi"
      exit 1
    }
  _harbor_log "server dependencies install finished in $((SECONDS - _pip_start))s"
fi

export HARBOR_PYTHON="$_HARBOR_PY"
export TMUX_TMPDIR="${TMUX_TMPDIR:-/tmp/.harbor-tmux}"
mkdir -p "$TMUX_TMPDIR"

for f in "$HOME/.bashrc" "$HOME/.bash_profile"; do
  [ -f "$f" ] || touch "$f"
  grep -q 'TMUX_TMPDIR' "$f" 2>/dev/null || echo "alias tmux='tmux -S $TMUX_TMPDIR/default'" >> "$f"
done

if [ -f "$HARBOR_STAGING/setup.sh" ]; then
  _harbor_log "Running task setup.sh"
  _setup_start=$SECONDS
  source "$HARBOR_STAGING/setup.sh"
  _harbor_log "task setup.sh finished in $((SECONDS - _setup_start))s"
fi

if ! command -v tmux >/dev/null 2>&1; then
  _harbor_log "Installing tmux"
  _tmux_start=$SECONDS
  if command -v apt-get >/dev/null 2>&1; then
    _prepare_apt_once
    apt-get install -y -qq tmux || _harbor_log "apt-get tmux install failed; continuing"
  elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache tmux || _harbor_log "apk tmux install failed; continuing"
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y tmux || _harbor_log "dnf tmux install failed; continuing"
  elif command -v yum >/dev/null 2>&1; then
    yum install -y tmux || _harbor_log "yum tmux install failed; continuing"
  fi
  _harbor_log "tmux install step finished in $((SECONDS - _tmux_start))s"
fi

_harbor_log "starting Harbor environment server: $*"
exec "$HARBOR_PYTHON" "$@"
