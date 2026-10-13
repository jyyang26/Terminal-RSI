"""Run isolated SC offline suites and CPU distributed checks; never start Ray/SGLang."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

SUITES = {
    "math_args": ["tests/test_ppo_sc_math.py", "tests/test_ppo_sc_args.py"],
    "transport_roles": [
        "tests/test_ppo_sc_transport.py",
        "tests/test_harbor_atif_adapter.py",
        "tests/test_harbor_llm.py",
        "tests/test_harbor_sample_postprocess.py",
        "tests/test_harbor_routing_replay.py",
        "tests/test_critic_checkpoint_isolation.py",
        "tests/utils/test_megatron_role_config.py",
    ],
    "loss_integration": ["tests/test_ppo_sc_harbor_integration.py"],
    "pipeline": ["tests/test_ppo_sc_pipeline.py"],
    "baseline_source": ["tests/ppo_sc_baseline_check.py"],
    "existing_regressions": [
        "tests/test_harbor_rollout_async.py",
        "tests/test_harbor_fully_async_rollout.py",
        "tests/test_harbor_entropy_reporting.py",
        "tests/test_harbor_partial_reward.py",
        "tests/test_dp_schedule.py",
        "tests/test_rollout_aux_workers.py",
        "tests/test_sample.py",
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Pre-implementation source tree captured before edits")
    parser.add_argument("--output-dir", required=True)
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(options.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        PPO_SC_BASELINE_DIR=str(Path(options.baseline).resolve()),
        PYTHONPATH=str(root) + os.pathsep + env.get("PYTHONPATH", ""),
    )
    results = {}
    commands = {
        name: [sys.executable, "-B", "-m", "pytest", "-o", "addopts=", "-p", "no:cacheprovider", "-q", *files]
        for name, files in SUITES.items()
    }
    commands.update(
        {
            f"tp{tp}_cp{cp}": [
                sys.executable,
                "-B",
                "tests/ppo_sc_distributed_worker.py",
                "--tp",
                str(tp),
                "--cp",
                str(cp),
            ]
            for tp, cp in ((2, 2), (1, 4))
        }
    )
    for name, command in commands.items():
        start = time.monotonic()
        print(f"Running {name}", flush=True)
        completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=240)
        text = completed.stdout + completed.stderr
        (out / f"{name}.log").write_text(text)
        passed = re.search(r"(\d+) passed", text)
        skipped = re.search(r"(\d+) skipped", text)
        results[name] = {
            "returncode": completed.returncode,
            "passed": int(passed[1]) if passed else None,
            "skipped": int(skipped[1]) if skipped else 0,
            "seconds": round(time.monotonic() - start, 2),
        }
        (out / "results.json").write_text(json.dumps(results, indent=2))
        if completed.returncode or skipped:
            raise SystemExit(f"{name} failed or skipped checks; inspect {out/name}.log")
        print(f"PASS {name}: {results[name]}", flush=True)
    print(
        json.dumps({"passed": sum(v["passed"] or 0 for v in results.values()), "distributed_jobs": 2, "skipped": 0}),
        flush=True,
    )


if __name__ == "__main__":
    main()
