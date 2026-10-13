from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path
from urllib.parse import quote

import requests


ACCEPT = (
    "application/vnd.docker.distribution.manifest.v2+json,"
    "application/vnd.oci.image.manifest.v1+json"
)


def _docker_image_from_task(task_toml: Path) -> str | None:
    with task_toml.open("rb") as f:
        data = tomllib.load(f)
    image = (data.get("environment") or {}).get("docker_image")
    if not image:
        return None
    if ":" not in image.rsplit("/", 1)[-1]:
        image = f"{image}:latest"
    return image


def _dockerhub_manifest(image: str, timeout: float) -> dict:
    repository, tag = image.split(":", 1)
    if repository.startswith("docker.io/"):
        repository = repository[len("docker.io/") :]
    if "/" not in repository:
        repository = f"library/{repository}"

    token_url = "https://auth.docker.io/token"
    token_response = requests.get(
        token_url,
        params={"service": "registry.docker.io", "scope": f"repository:{repository}:pull"},
        timeout=timeout,
    )
    token_response.raise_for_status()
    token = token_response.json()["token"]

    manifest_url = f"https://registry-1.docker.io/v2/{repository}/manifests/{quote(tag, safe='')}"
    response = requests.get(
        manifest_url,
        headers={"Authorization": f"Bearer {token}", "Accept": ACCEPT},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _manifest_size_bytes(manifest: dict) -> int:
    if "manifests" in manifest:
        return sum(item.get("size", 0) for item in manifest["manifests"])
    return sum(item.get("size", 0) for item in manifest.get("layers", []))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks_dir", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    tasks = sorted(args.tasks_dir.glob("*/task.toml"))
    if args.limit:
        tasks = tasks[: args.limit]

    rows = []
    for task_toml in tasks:
        image = _docker_image_from_task(task_toml)
        if not image:
            continue
        try:
            manifest = _dockerhub_manifest(image, args.timeout)
            size_bytes = _manifest_size_bytes(manifest)
            rows.append((size_bytes, task_toml.parent.name, image))
        except Exception as exc:
            print(f"ERROR\t{task_toml.parent.name}\t{image}\t{exc}", file=sys.stderr)

    for size_bytes, task_name, image in sorted(rows):
        print(f"{size_bytes / (1024**2):9.1f} MiB\t{task_name}\t{image}")

    if not rows:
        print("No image sizes resolved. Set HTTP_PROXY/HTTPS_PROXY if needed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    raise SystemExit(main())
