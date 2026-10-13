# Qwen3.5/Qwen3.6 35B-A3B MoE RL Docker patch bundle

This is an immutable Docker `PATCH_VERSION` bundle for:

- Base: `slimerl/sglang:v0.5.15.post1-cu129`
- SGLang commit: `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`
- Megatron commit: `1dcf0dafa884ad52ffb243625717a3471643e087`
- Additional training fix: semantic backport of SGLang PR #27140

## Why it exists

Qwen3.5/3.6 35B-A3B combines MoE and Gated DeltaNet recurrent state. During
Slime RL weight sync, pause/resume can restore recurrent buffers at new GPU
addresses. CUDA graphs captured before the update then replay stale pointers,
causing coherent rollout 0 but gibberish after the first training step.

This bundle includes:

1. All THUDM/slime `docker/patch/latest` patches from 2026-07-29.
2. GemmaRMSNorm captured-buffer correctness fix via the v0.5.15 base.
3. `sglang-rl-cuda-graph-recapture.patch`, which marks graphs stale after all
   online weight update paths and recaptures them after memory resume.

## Build

From repository root on a machine with Docker/BuildKit:

```bash
docker build \
  -f docker/Dockerfile \
  --build-arg SGLANG_IMAGE_TAG=v0.5.15.post1-cu129 \
  --build-arg PATCH_VERSION=qwen35-moe-v0.5.15.post1-cu129-pr27140-20260729 \
  --build-arg MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087 \
  --build-arg ENABLE_SGLANG_PATCH=1 \
  -t slimerl/slime:qwen35-moe-v0.5.15.post1-cu129-pr27140-20260729 \
  .
```

Or run:

```bash
bash docker/patch/qwen35-moe-v0.5.15.post1-cu129-pr27140-20260729/build.sh
```

## Validate without building the image

```bash
bash docker/patch/qwen35-moe-v0.5.15.post1-cu129-pr27140-20260729/validate.sh
```

`SGLANG_SOURCE_TARBALL` may point to a local v0.5.15.post1 source tarball;
default: `/tmp/sglang515.tgz`.
