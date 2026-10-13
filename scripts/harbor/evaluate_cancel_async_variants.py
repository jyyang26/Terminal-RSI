#!/usr/bin/env python3
"""Replay extracted cancel-async-tasks variants against local equivalent tests."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


KEYSTROKE_RE = re.compile(r'"keystrokes"\s*:\s*"((?:\\.|[^"\\])*)"')
RUN_PY_RE = re.compile(
    r"cat\s+>\s+/app/run\.py\s+<<\s+'?EOF'?\n(.*?)\nEOF",
    re.DOTALL,
)

TEST_PY = """\
import argparse
import asyncio
import os

from run import run_tasks


TASK_SLEEP = float(os.environ.get("CANCEL_ASYNC_TASK_SLEEP", "2"))
CLEANUP_SLEEP = float(os.environ.get("CANCEL_ASYNC_CLEANUP_SLEEP", "1"))


async def task() -> None:
    try:
        print("Task started.", flush=True)
        await asyncio.sleep(TASK_SLEEP)
        print("Task finished.", flush=True)
    finally:
        await asyncio.sleep(CLEANUP_SLEEP)
        print("Cleaned up.", flush=True)


async def main(n_tasks: int, max_concurrent: int) -> None:
    tasks = [task] * n_tasks
    await run_tasks(tasks, max_concurrent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type=int, required=True)
    parser.add_argument("--max-concurrent", type=int, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.n_tasks, args.max_concurrent))
"""


def _load_json_string(raw: str) -> str:
    try:
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        return raw


def _commands(response: str) -> list[str]:
    return [_load_json_string(match.group(1)) for match in KEYSTROKE_RE.finditer(response)]


def _last_run_py(record: dict[str, Any]) -> str:
    last = ""
    for command in _commands(str(record.get("response") or "")):
        for match in RUN_PY_RE.finditer(command):
            last = match.group(1)
    return last


def _records(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return out


def _run(cmd: list[str], cwd: Path, timeout: float, env: dict[str, str]) -> tuple[bool, str, float]:
    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - start
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return False, f"timeout stdout={stdout!r} stderr={stderr!r}", elapsed
    elapsed = time.monotonic() - start
    if result.returncode != 0:
        return False, f"rc={result.returncode} stdout={result.stdout!r} stderr={result.stderr!r}", elapsed
    return True, result.stdout, elapsed


def _run_sigint(
    cwd: Path,
    n_tasks: int,
    max_concurrent: int,
    signal_after: float,
    timeout: float,
    env: dict[str, str],
) -> tuple[bool, str, float]:
    start = time.monotonic()
    proc = subprocess.Popen(
        [
            sys.executable,
            "test.py",
            "--n-tasks",
            str(n_tasks),
            "--max-concurrent",
            str(max_concurrent),
        ],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    time.sleep(signal_after)
    proc.send_signal(signal.SIGINT)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=2)
        return False, f"timeout stdout={stdout!r} stderr={stderr!r}", time.monotonic() - start
    finally:
        proc.kill()
    ok = stdout.count("Task started.") == min(n_tasks, max_concurrent)
    ok = ok and stdout.count("Cleaned up.") == min(n_tasks, max_concurrent)
    return ok, f"stdout={stdout!r} stderr={stderr!r}", time.monotonic() - start


def _evaluate(code: str, quick: bool) -> dict[str, Any]:
    if not code.strip():
        return {"all_passed": False, "first_failure": "no run.py code"}
    with tempfile.TemporaryDirectory(prefix="cancel_async_eval_") as raw_tmp:
        tmp = Path(raw_tmp)
        (tmp / "run.py").write_text(code)
        (tmp / "test.py").write_text(TEST_PY)
        task_sleep = 0.8 if quick else 2.0
        cleanup_sleep = 0.2 if quick else 1.0
        signal_after = 0.3 if quick else 0.5
        cancel_timeout = 2.5 if quick else 5.0
        env = os.environ.copy()
        env["CANCEL_ASYNC_TASK_SLEEP"] = str(task_sleep)
        env["CANCEL_ASYNC_CLEANUP_SLEEP"] = str(cleanup_sleep)

        tests: list[tuple[str, bool, str, float]] = []
        ok, detail, elapsed = _run([sys.executable, "-c", "from run import run_tasks; print(run_tasks)"], tmp, 5, env)
        tests.append(("import", ok, detail, elapsed))
        ok, detail, elapsed = _run([sys.executable, "test.py", "--n-tasks", "2", "--max-concurrent", "2"], tmp, 2 if quick else 5, env)
        ok = ok and detail.count("Task started.") == 2 and detail.count("Cleaned up.") == 2
        tests.append(("concurrent", ok, detail, elapsed))
        ok, detail, elapsed = _run([sys.executable, "test.py", "--n-tasks", "2", "--max-concurrent", "1"], tmp, 3 if quick else 10, env)
        min_elapsed = 2 * (task_sleep + cleanup_sleep)
        ok = ok and detail.count("Task started.") == 2 and detail.count("Cleaned up.") == 2 and elapsed >= min_elapsed * 0.8
        tests.append(("max_concurrent", ok, detail, elapsed))
        tests.append(("cancel_below", *_run_sigint(tmp, 2, 3, signal_after, cancel_timeout, env)))
        tests.append(("cancel_at", *_run_sigint(tmp, 2, 2, signal_after, cancel_timeout, env)))
        tests.append(("cancel_above", *_run_sigint(tmp, 3, 2, signal_after, cancel_timeout, env)))

    first_failure = ""
    for name, ok, detail, _elapsed in tests:
        if not ok:
            first_failure = f"{name}: {detail[:500]}"
            break
    return {
        "all_passed": not first_failure,
        "first_failure": first_failure,
        "test_results": "; ".join(f"{name}={ok}" for name, ok, _detail, _elapsed in tests),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="LABEL=path/to/samples.jsonl")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--quick", action="store_true", help="Use shorter sleeps for faster behavioral classification.")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for item in args.inputs:
        if "=" not in item:
            raise SystemExit(f"input must be LABEL=PATH: {item}")
        label, raw_path = item.split("=", 1)
        for index, record in enumerate(_records(Path(raw_path)), start=1):
            result = _evaluate(_last_run_py(record), quick=args.quick)
            rows.append(
                {
                    "label": label,
                    "sample_index": index,
                    "dump": record.get("dump"),
                    "reward": record.get("reward"),
                    **result,
                }
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
