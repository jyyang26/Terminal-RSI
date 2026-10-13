#!/usr/bin/env python3
"""Probe Daytona concurrent sandbox capacity with cleanup.

This creates lightweight Daytona sandboxes in increasing target counts and
keeps each successful batch alive long enough to verify the concurrent quota.
All probe sandboxes are named/labeled with a unique prefix and deleted at exit
unless --keep is set.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


PROBE_LABEL_KEY = "slime_probe"
PROBE_LABEL_VALUE = "daytona_capacity"


def _rest_request(method: str, path: str, *, timeout: float) -> Any:
    api_key = os.environ["DAYTONA_API_KEY"]
    api_url = os.environ.get("DAYTONA_API_URL") or "https://app.daytona.io/api"
    url = api_url.rstrip("/") + path
    req = urllib.request.Request(
        url,
        method=method,
        headers={
            "Accept": "application/json",
            "Authorization": "Bearer " + api_key,
            "X-Daytona-Source": "slime-capacity-probe",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8")) if data else {}


def _sandbox_disk_gib(item: dict[str, Any]) -> float:
    for key in ("disk", "diskGiB", "disk_gib"):
        value = item.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
    resources = item.get("resources") or {}
    for key in ("disk", "diskGiB", "disk_gib"):
        value = resources.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def iter_remote_sandboxes(*, timeout: float, limit: int = 100):
    cursor = None
    while True:
        query = {"limit": limit}
        if cursor:
            query["cursor"] = cursor
        page = _rest_request("GET", "/sandbox?" + urllib.parse.urlencode(query), timeout=timeout)
        for item in page.get("items") or []:
            yield item
        cursor = page.get("nextCursor")
        if not cursor:
            break


def _is_probe_item(item: dict[str, Any], prefix: str) -> bool:
    name = str(item.get("name") or "")
    labels = item.get("labels") or item.get("metadata") or {}
    return name.startswith(prefix) or labels.get(PROBE_LABEL_KEY) == PROBE_LABEL_VALUE


async def delete_remote_probe_sandboxes(prefix: str, *, timeout: float) -> tuple[int, int]:
    items = [item for item in iter_remote_sandboxes(timeout=timeout) if _is_probe_item(item, prefix)]
    errors = 0
    for item in items:
        sandbox_id = item.get("id") or item.get("name")
        name = item.get("name")
        try:
            target = urllib.parse.quote(str(sandbox_id), safe="")
            _rest_request("DELETE", f"/sandbox/{target}", timeout=timeout)
            print(f"deleted_probe id={sandbox_id} name={name}")
        except Exception as exc:
            errors += 1
            print(f"delete_error id={sandbox_id} name={name} err={exc!r}")
    return len(items), errors


@dataclass
class CreateResult:
    name: str
    ok: bool
    elapsed_sec: float
    sandbox: Any | None = None
    error: str | None = None


async def create_one(
    daytona: Any,
    params_cls: Any,
    image_cls: Any,
    resources_cls: Any,
    *,
    name: str,
    image: str,
    disk_gib: int,
    cpu: int,
    memory_gib: int,
    timeout: float,
    auto_delete_interval: int,
    auto_stop_interval: int,
) -> CreateResult:
    start = time.monotonic()
    params = params_cls(
        name=name,
        image=image_cls.base(image),
        resources=resources_cls(cpu=cpu, memory=memory_gib, disk=disk_gib),
        labels={
            PROBE_LABEL_KEY: PROBE_LABEL_VALUE,
            "slime_probe_prefix": name.rsplit("-", 1)[0],
        },
        auto_delete_interval=auto_delete_interval,
        auto_stop_interval=auto_stop_interval,
    )
    try:
        sandbox = await daytona.create(params=params, timeout=timeout)
        return CreateResult(
            name=name,
            ok=True,
            elapsed_sec=time.monotonic() - start,
            sandbox=sandbox,
        )
    except Exception as exc:
        return CreateResult(
            name=name,
            ok=False,
            elapsed_sec=time.monotonic() - start,
            error=f"{exc.__class__.__name__}: {exc}",
        )


async def create_until_target(
    *,
    daytona: Any,
    params_cls: Any,
    image_cls: Any,
    resources_cls: Any,
    prefix: str,
    current: list[Any],
    target: int,
    image: str,
    disk_gib: int,
    cpu: int,
    memory_gib: int,
    timeout: float,
    create_parallelism: int,
    auto_delete_interval: int,
    auto_stop_interval: int,
) -> bool:
    missing = target - len(current)
    if missing <= 0:
        return True

    sem = asyncio.Semaphore(create_parallelism)

    async def worker(index: int) -> CreateResult:
        async with sem:
            return await create_one(
                daytona,
                params_cls,
                image_cls,
                resources_cls,
                name=f"{prefix}-{index:04d}",
                image=image,
                disk_gib=disk_gib,
                cpu=cpu,
                memory_gib=memory_gib,
                timeout=timeout,
                auto_delete_interval=auto_delete_interval,
                auto_stop_interval=auto_stop_interval,
            )

    start_index = len(current)
    results = await asyncio.gather(*(worker(i) for i in range(start_index, target)))
    ok_results = [result for result in results if result.ok]
    current.extend(result.sandbox for result in ok_results if result.sandbox is not None)
    failed = [result for result in results if not result.ok]
    elapsed = max((result.elapsed_sec for result in results), default=0.0)
    print(
        "target_result "
        f"target={target} requested={missing} created={len(ok_results)} "
        f"live_tracked={len(current)} failed={len(failed)} max_create_sec={elapsed:.1f}"
    )
    for result in failed[:8]:
        print(f"create_error name={result.name} elapsed_sec={result.elapsed_sec:.1f} err={result.error}")
    return not failed and len(current) >= target


async def main_async(args: argparse.Namespace) -> int:
    from daytona import AsyncDaytona, CreateSandboxFromImageParams, Image, Resources

    prefix = args.prefix
    if not prefix:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        prefix = f"slime-capacity-{stamp}"

    print(
        "probe_start "
        f"prefix={prefix} targets={args.targets} disk_gib={args.disk_gib} "
        f"image={args.image} parallelism={args.create_parallelism}"
    )

    if args.cleanup_first:
        count, errors = await delete_remote_probe_sandboxes(prefix, timeout=args.rest_timeout)
        print(f"cleanup_first count={count} errors={errors}")
        if errors:
            return 2

    daytona = AsyncDaytona()
    tracked: list[Any] = []
    success_targets: list[int] = []
    failed_target: int | None = None
    try:
        for target in args.targets:
            ok = await create_until_target(
                daytona=daytona,
                params_cls=CreateSandboxFromImageParams,
                image_cls=Image,
                resources_cls=Resources,
                prefix=prefix,
                current=tracked,
                target=target,
                image=args.image,
                disk_gib=args.disk_gib,
                cpu=args.cpu,
                memory_gib=args.memory_gib,
                timeout=args.create_timeout,
                create_parallelism=args.create_parallelism,
                auto_delete_interval=args.auto_delete_interval,
                auto_stop_interval=args.auto_stop_interval,
            )
            remote_items = [
                item
                for item in iter_remote_sandboxes(timeout=args.rest_timeout)
                if _is_probe_item(item, prefix)
            ]
            remote_disk = sum(_sandbox_disk_gib(item) for item in remote_items)
            print(
                "remote_probe_state "
                f"target={target} remote_count={len(remote_items)} remote_disk_gib={remote_disk:g}"
            )
            if not ok:
                failed_target = target
                break
            success_targets.append(target)
            if args.hold_sec:
                await asyncio.sleep(args.hold_sec)
    finally:
        if args.keep:
            print(f"keeping_probe prefix={prefix} tracked={len(tracked)}")
        else:
            count, errors = await delete_remote_probe_sandboxes(prefix, timeout=args.rest_timeout)
            print(f"cleanup_final count={count} errors={errors}")
            if errors:
                return 3
        try:
            await daytona.close()
        except Exception:
            pass

    print(
        "probe_summary "
        f"success_targets={success_targets} failed_target={failed_target} "
        f"max_success={max(success_targets) if success_targets else 0}"
    )
    return 1 if failed_target is not None else 0


def parse_targets(value: str) -> list[int]:
    targets = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not targets:
        raise argparse.ArgumentTypeError("targets cannot be empty")
    return sorted(set(targets))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=parse_targets, default=[64])
    parser.add_argument("--disk-gib", type=int, default=5)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--memory-gib", type=int, default=1)
    parser.add_argument("--image", default="ubuntu:22.04")
    parser.add_argument("--create-timeout", type=float, default=180.0)
    parser.add_argument("--rest-timeout", type=float, default=30.0)
    parser.add_argument("--create-parallelism", type=int, default=16)
    parser.add_argument("--auto-delete-interval", type=int, default=3600)
    parser.add_argument("--auto-stop-interval", type=int, default=3600)
    parser.add_argument("--hold-sec", type=float, default=0.0)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--cleanup-first", action="store_true")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0)
