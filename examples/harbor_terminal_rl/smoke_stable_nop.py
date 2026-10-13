from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial

from slime_plugins.harbor.stable_tasks import load_stable_task_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize one stable_rl_tb_rl task and run Harbor with nop agent."
    )
    parser.add_argument("--stable-tasks-path", required=True)
    parser.add_argument("--materialized-tasks-dir", required=True)
    parser.add_argument("--sif-cache-dir", required=True)
    parser.add_argument("--trials-dir", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--apt-proxy", default="http://star-proxy.oa.com:3128")
    parser.add_argument("--singularity-no-mount", default="home,tmp,bind-paths,proc")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    entries = load_stable_task_entries(
        args.stable_tasks_path,
        materialized_tasks_dir=args.materialized_tasks_dir,
        sif_cache_dir=args.sif_cache_dir,
        limit=args.limit,
        offset=args.offset,
        apt_proxy=args.apt_proxy,
    )
    if not entries:
        raise RuntimeError("No stable tasks were materialized.")

    entry = entries[0]
    task_path = entry["task"]["path"]
    task_name = Path(task_path).name
    config = TrialConfig.model_validate(
        {
            "task": entry["task"],
            "trial_name": f"stable-nop-smoke-{task_name}",
            "trials_dir": args.trials_dir,
            "timeout_multiplier": 1.0,
            "agent": {"name": "nop"},
            "environment": {
                "import_path": "slime_plugins.harbor.singularity_env:SlimeSingularityEnvironment",
                "kwargs": {"singularity_no_mount": args.singularity_no_mount},
            },
        }
    )

    trial = await Trial.create(config)
    result = await trial.run()
    payload = result.model_dump(mode="json")
    print(
        json.dumps(
            {
                "trial_name": config.trial_name,
                "task_path": task_path,
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
