#!/usr/bin/env bash
# examples/terminal_rsi/prepare_load_root.sh
#
# Build the actor/critic load-root PAIR that an RPR phase starts from:
#
#   <LOAD_ROOT>/release          -> <actor torch_dist iteration dir>
#   <LOAD_ROOT>/latest_checkpointed_iteration.txt   ("release")
#   <LOAD_ROOT>_critic/release   -> <critic torch_dist iteration dir>
#   <LOAD_ROOT>_critic/latest_checkpointed_iteration.txt
#
# slime loads the actor from --load and redirects the critic to "<load>_critic" when that
# sibling exists, so the two models can come from different runs (e.g. the RSR-2 actor
# and the retained RPR-1 critic). The "release" tracker makes Megatron load weights only
# and lets the RL rollout counter start at 0. Nothing is copied: both entries are symlinks.
#
# Usage: prepare_load_root.sh <actor_iter_dir> <critic_iter_dir|none> <load_root>
#   e.g. prepare_load_root.sh ckpts/rsr2/iter_0000527 ckpts/rpr1_critic/iter_0000019 ckpts/loadroot_rpr2
# Pass "none" as the critic to build an actor-only root (cold value head, WARM_CRITIC=0).
set -euo pipefail

[ $# -eq 3 ] || { sed -n "2,21p" "$0"; exit 1; }
ACTOR_ITER="$(realpath "$1")"
CRITIC_ITER="none"
[ "$2" = "none" ] || CRITIC_ITER="$(realpath "$2")"
LOAD_ROOT="$3"

for d in "${ACTOR_ITER}" "${CRITIC_ITER}"; do
  [ "${d}" = "none" ] && continue
  [ -d "${d}" ] || { echo "not a directory: ${d}" >&2; exit 1; }
  [ -f "${d}/common.pt" ] || [ -f "${d}/.metadata" ] \
    || echo "WARN: ${d} does not look like a Megatron torch_dist iteration dir" >&2
done

for root in "${LOAD_ROOT}" "${LOAD_ROOT}_critic"; do
  if [ -e "${root}" ]; then
    echo "refusing to overwrite existing ${root}" >&2
    exit 1
  fi
done

mkdir -p "${LOAD_ROOT}"
ln -s "${ACTOR_ITER}" "${LOAD_ROOT}/release"
echo release > "${LOAD_ROOT}/latest_checkpointed_iteration.txt"
echo "actor  : ${LOAD_ROOT}/release -> ${ACTOR_ITER}"

if [ "${CRITIC_ITER}" = "none" ]; then
  echo "critic : none (cold value head; launch with WARM_CRITIC=0)"
else
  mkdir -p "${LOAD_ROOT}_critic"
  ln -s "${CRITIC_ITER}" "${LOAD_ROOT}_critic/release"
  echo release > "${LOAD_ROOT}_critic/latest_checkpointed_iteration.txt"
  echo "critic : ${LOAD_ROOT}_critic/release -> ${CRITIC_ITER}"
fi
