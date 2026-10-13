from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
from pathlib import Path

_metadata_version = importlib.metadata.version


def _version_with_harbor_src_fallback(name: str) -> str:
    if name == "harbor":
        try:
            return _metadata_version(name)
        except importlib.metadata.PackageNotFoundError:
            return "0.0.0+src"
    return _metadata_version(name)


importlib.metadata.version = _version_with_harbor_src_fallback

from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one terminal-bench-2 task through Harbor with the nop agent."
    )
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--sif-cache-dir", required=True)
    parser.add_argument("--trials-dir", required=True)
    parser.add_argument("--trial-name", default=None)
    parser.add_argument("--timeout-multiplier", type=float, default=1.0)
    parser.add_argument("--singularity-no-mount", default="home,tmp,bind-paths,proc")
    parser.add_argument("--registry-proxy", default="http://star-proxy.oa.com:3128")
    parser.add_argument("--registry-no-proxy", default=None)
    parser.add_argument("--registry-mirror", default="docker.1panel.live")
    parser.add_argument("--bind-host-proc", action="store_true", default=True)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    task_dir = Path(args.task_dir).resolve()
    task_name = task_dir.name
    trial_name = args.trial_name or f"tb2-nop-smoke-{task_name}"

    environment_kwargs = {
        "singularity_image_cache_dir": str(Path(args.sif_cache_dir).resolve()),
        "singularity_no_mount": args.singularity_no_mount,
        "singularity_bind_host_proc": args.bind_host_proc,
    }
    if args.registry_proxy:
        environment_kwargs["singularity_registry_proxy"] = args.registry_proxy
    if args.registry_no_proxy:
        environment_kwargs["singularity_registry_no_proxy"] = args.registry_no_proxy
    if args.registry_mirror:
        environment_kwargs["singularity_registry_mirror"] = args.registry_mirror

    config = TrialConfig.model_validate(
        {
            "task": {"path": str(task_dir)},
            "trial_name": trial_name,
            "trials_dir": args.trials_dir,
            "timeout_multiplier": args.timeout_multiplier,
            "agent": {"name": "nop"},
            "environment": {
                "import_path": "slime_plugins.harbor.singularity_env:SlimeSingularityEnvironment",
                "kwargs": environment_kwargs,
            },
        }
    )

    trial = await Trial.create(config)
    result = await trial.run()
    payload = result.model_dump(mode="json")
    print(
        json.dumps(
            {
                "trial_name": trial_name,
                "task_path": str(task_dir),
                "trials_dir": args.trials_dir,
                "status": payload.get("status"),
                "exception_info": payload.get("exception_info"),
                "verifier_result": payload.get("verifier_result"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
