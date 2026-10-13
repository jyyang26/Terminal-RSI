#!/bin/bash
set -euo pipefail
BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${BUNDLE_DIR}/../../.." && pwd)"
PATCH_VERSION="$(basename "${BUNDLE_DIR}")"
IMAGE_TAG="${IMAGE_TAG:-slimerl/slime:${PATCH_VERSION}}"
docker build \
  -f "${REPO_ROOT}/docker/Dockerfile" \
  --build-arg SGLANG_IMAGE_TAG=v0.5.15.post1-cu129 \
  --build-arg PATCH_VERSION="${PATCH_VERSION}" \
  --build-arg MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087 \
  --build-arg ENABLE_SGLANG_PATCH=1 \
  -t "${IMAGE_TAG}" \
  "${REPO_ROOT}"
echo "Built ${IMAGE_TAG}"
