#!/bin/bash
set -euo pipefail
BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARBALL="${SGLANG_SOURCE_TARBALL:-/tmp/sglang515.tgz}"
test -f "${TARBALL}"
tmp="$(mktemp -d /tmp/qwen35-bundle-validate-XXXXXX)"
trap 'chmod -R u+w "${tmp}" 2>/dev/null || true; rm -rf "${tmp}" 2>/dev/null || true' EXIT
tar -xzf "${TARBALL}" -C "${tmp}" --strip-components=1
git -C "${tmp}" init -q
git -C "${tmp}" add .
git -C "${tmp}" -c user.name=validate -c user.email=validate@local commit -qm base
for patch in \
  sglang.patch \
  sglang-top_p.patch \
  sglang-release_hicache.patch \
  sglang-pull_weights.patch \
  sglang-rl-cuda-graph-recapture.patch; do
  test -s "${BUNDLE_DIR}/${patch}"
  git -C "${tmp}" apply --check "${BUNDLE_DIR}/${patch}"
  git -C "${tmp}" apply "${BUNDLE_DIR}/${patch}"
  echo "applied ${patch}"
done
python -m py_compile \
  "${tmp}/python/sglang/srt/managers/scheduler_components/weight_updater.py" \
  "${tmp}/python/sglang/srt/model_executor/model_runner.py"
grep -q 'cuda_graphs_need_recapture' "${tmp}/python/sglang/srt/managers/scheduler_components/weight_updater.py"
grep -q 'out=self.gemma_weight' "${tmp}/python/sglang/srt/layers/layernorm.py"
echo 'Patch bundle validation: PASS'
