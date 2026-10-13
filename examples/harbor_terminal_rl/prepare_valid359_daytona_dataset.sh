#!/bin/bash
#
# Prepare the WGDS-v3 Round9 valid359 dataset for the DAYTONA backend.
#
# WHY: every task.toml in `harbor_dataset_step4_step28_valid359_v1` sets
#   [environment] docker_image = "<local .sif path>"
# The Singularity backend can boot from a local .sif file, but the Daytona
# backend boots from a REGISTRY image (Image.base("<repo>:<tag>")). It cannot
# read a local filesystem .sif path. There is no global YAML override for the
# per-task docker_image (task.toml deep-merges OVER harbor_trial_defaults and
# wins on key conflict — see slime_plugins/harbor/rollout.py:_build_trial_config),
# so the only clean fix is to rewrite the per-task docker_image to the registry
# tag the .sif was built from.
#
# All 359 tasks share ONE base image:
#   .sif:      .../alexgshaw_mteb-leaderboard_20251031.sif
#   registry:  alexgshaw/mteb-leaderboard:20251031   (== TB2 mteb-leaderboard task image)
#
# The dataset is tiny (~17MB, 359 dirs), so we copy the whole thing and rewrite
# in place on the COPY. The original (<user>'s) tree is never modified.
#
# Usage:
#   bash prepare_valid359_daytona_dataset.sh            # use defaults below
#   SRC=... DST=... REGISTRY_IMAGE=... bash prepare_valid359_daytona_dataset.sh

set -euo pipefail

SRC="${SRC:-/path/to/workspace/Agent_training/wgds_v3/outputs/iter_round9_evolutionary_pipeline_89tasks/harbor_dataset_step4_step28_valid359_v1}"
DST="${DST:-/path/to/workspace/code/harbor_dataset_step4_valid359_daytona}"
REGISTRY_IMAGE="${REGISTRY_IMAGE:-alexgshaw/mteb-leaderboard:20251031}"

echo "=== prepare_valid359_daytona_dataset ==="
echo "SRC             = ${SRC}"
echo "DST             = ${DST}"
echo "REGISTRY_IMAGE  = ${REGISTRY_IMAGE}"
echo

test -d "${SRC}" || { echo "ERROR: SRC not found: ${SRC}" >&2; exit 2; }

SRC_COUNT="$(find "${SRC}" -maxdepth 1 -type d -name 'wgds9-*' | wc -l | tr -d ' ')"
echo "source wgds9-* task dirs: ${SRC_COUNT}"
[ "${SRC_COUNT}" -gt 0 ] || { echo "ERROR: no wgds9-* tasks under SRC" >&2; exit 2; }

# Copy (resolve symlinks with -L so the copy is self-contained; the upstream
# valid359 tree may itself contain symlinked task dirs).
if [ -e "${DST}" ]; then
  echo "ERROR: DST already exists: ${DST}" >&2
  echo "       remove it or set DST=... to a fresh path before re-running." >&2
  exit 2
fi
mkdir -p "$(dirname "${DST}")"
echo "copying ${SRC} -> ${DST} ..."
cp -aL "${SRC}" "${DST}"

# Rewrite docker_image in every task.toml on the COPY only.
python3 - "${DST}" "${REGISTRY_IMAGE}" <<'PY'
import pathlib, re, sys

dst, registry = sys.argv[1], sys.argv[2]
root = pathlib.Path(dst)
tomls = sorted(root.glob("wgds9-*/task.toml"))
pat = re.compile(r'^(\s*docker_image\s*=\s*)"[^"]*"\s*$', re.M)

rewritten = unchanged = 0
for t in tomls:
    txt = t.read_text()
    new, n = pat.subn(rf'\g<1>"{registry}"', txt)
    if n:
        t.write_text(new)
        rewritten += 1
    else:
        unchanged += 1

print(f"task.toml files            : {len(tomls)}")
print(f"docker_image rewritten     : {rewritten}")
print(f"docker_image NOT found/kept: {unchanged}")
if unchanged:
    print("WARNING: some task.toml had no docker_image line — inspect before training.")
PY

echo
echo "=== verification ==="
DST_COUNT="$(find "${DST}" -maxdepth 1 -type d -name 'wgds9-*' | wc -l | tr -d ' ')"
echo "copied wgds9-* task dirs   : ${DST_COUNT}"
echo "docker_image still .sif    : $(grep -rl '\.sif"' "${DST}"/wgds9-*/task.toml 2>/dev/null | wc -l | tr -d ' ')  (must be 0)"
echo "docker_image = registry    : $(grep -rl "docker_image = \"${REGISTRY_IMAGE}\"" "${DST}"/wgds9-*/task.toml 2>/dev/null | wc -l | tr -d ' ')  (should be ${DST_COUNT})"
echo
echo "sample task.toml [environment]:"
grep -A1 '^\[environment\]' "${DST}"/wgds9-absorbance-batch-modernize/task.toml 2>/dev/null || \
  grep 'docker_image' "$(find "${DST}" -name task.toml | head -1)"
echo
echo "DONE. Point the harbor config at:"
echo "  harbor_tasks_dir: ${DST}"
echo "  harbor_tasks_glob: \"wgds9-*/task.toml\""
