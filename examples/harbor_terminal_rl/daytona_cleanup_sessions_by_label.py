#!/usr/bin/env python3
"""Delete Daytona sandboxes for an explicit Harbor session list via labels."""

from __future__ import annotations

import argparse
import asyncio
import datetime
import sys


def _iso_epoch(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return None


def _sandbox_created_epoch(sandbox) -> float | None:
    for name in ("created_at", "createdAt", "created"):
        value = getattr(sandbox, name, None)
        if value is not None:
            if isinstance(value, datetime.datetime):
                return value.timestamp()
            return _iso_epoch(str(value))
    return None


def _sandbox_labels(sandbox) -> dict:
    labels = getattr(sandbox, "labels", None)
    return labels if isinstance(labels, dict) else {}


async def _list_by_label(daytona, labels: dict[str, str]):
    import inspect

    def has_labels(sb) -> bool:
        sb_labels = _sandbox_labels(sb)
        return all(sb_labels.get(k) == v for k, v in labels.items())

    if inspect.isasyncgenfunction(daytona.list):
        try:
            from daytona import ListSandboxesQuery

            gen = daytona.list(ListSandboxesQuery(labels=labels))
        except Exception:
            gen = daytona.list()
        return [sandbox async for sandbox in gen if has_labels(sandbox)]

    result = daytona.list(labels=labels)
    if inspect.isawaitable(result):
        result = await result
    items = getattr(result, "items", None)
    if items is None:
        items = result if isinstance(result, list) else []
    return [sandbox for sandbox in items if has_labels(sandbox)]


def _in_time_window(sandbox, *, created_after: float | None, created_before: float | None) -> bool:
    if created_after is None and created_before is None:
        return True
    created = _sandbox_created_epoch(sandbox)
    if created is None:
        return False
    if created_after is not None and created < created_after:
        return False
    if created_before is not None and created >= created_before:
        return False
    return True


async def _run(args) -> int:
    try:
        from daytona import AsyncDaytona
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR import daytona SDK: {exc!r}")
        return 2

    with open(args.session_file, encoding="utf-8") as fh:
        sessions = [line.strip() for line in fh if line.strip()]
    sessions = sorted(set(sessions))

    created_after = _iso_epoch(args.created_after)
    created_before = _iso_epoch(args.created_before)
    if args.created_after and created_after is None:
        print(f"ERROR invalid --created-after: {args.created_after!r}")
        return 2
    if args.created_before and created_before is None:
        print(f"ERROR invalid --created-before: {args.created_before!r}")
        return 2

    print(
        "SESSION_LABEL_CLEANUP "
        f"action={args.action} sessions={len(sessions)} "
        f"created_after={args.created_after!r} created_before={args.created_before!r}"
    )

    daytona = AsyncDaytona()
    candidates = 0
    cleaned = 0
    errors = 0
    sessions_with_candidates = 0

    for idx, session in enumerate(sessions, start=1):
        labels = {"harbor_managed": "1", "harbor_session": session}
        try:
            items = await asyncio.wait_for(_list_by_label(daytona, labels), timeout=args.list_timeout)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"ERROR list session={session}: {exc!r}")
            continue

        items = [
            sandbox
            for sandbox in items
            if _in_time_window(
                sandbox,
                created_after=created_after,
                created_before=created_before,
            )
        ]
        if items:
            sessions_with_candidates += 1
        for sandbox in items:
            candidates += 1
            sandbox_id = getattr(sandbox, "id", None)
            state = getattr(sandbox, "state", None)
            print(f"CANDIDATE session={session} id={sandbox_id} state={state}")
            if args.action == "audit":
                continue
            try:
                await asyncio.wait_for(daytona.delete(sandbox), timeout=args.delete_timeout)
                cleaned += 1
                print(f"CLEANED session={session} id={sandbox_id}")
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"ERROR delete session={session} id={sandbox_id}: {exc!r}")
        if idx % args.progress_every == 0:
            print(
                "PROGRESS "
                f"sessions_checked={idx}/{len(sessions)} candidates={candidates} "
                f"cleaned={cleaned} errors={errors}"
            )

    print(
        "summary "
        f"sessions={len(sessions)} sessions_with_candidates={sessions_with_candidates} "
        f"candidates={candidates} cleaned={cleaned} errors={errors}"
    )
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-file", required=True)
    parser.add_argument("--action", choices=("audit", "delete"), default="audit")
    parser.add_argument("--created-after", default=None)
    parser.add_argument("--created-before", default=None)
    parser.add_argument("--list-timeout", type=float, default=20.0)
    parser.add_argument("--delete-timeout", type=float, default=30.0)
    parser.add_argument("--progress-every", type=int, default=25)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0)
