#!/usr/bin/env python3
"""Archive/delete Harbor Daytona sandboxes by deterministic trial name.

Daytona list() can be slow or time out when many sandboxes exist. Harbor trial
names are deterministic, so this utility cleans known rollout/group/sample
names directly. Prefer --action archive-direct when list/get time out.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import urllib.parse
import urllib.request


def iter_trial_names(rollout_id: int, max_group: int, samples_per_group: int):
    for group_index in range(max_group):
        for sample_rank in range(samples_per_group):
            sample_index = group_index * samples_per_group + sample_rank
            yield f"slime-r{rollout_id}-g{group_index}-s{sample_rank}-i{sample_index}"


def _rest_request(method: str, path: str, *, timeout: float):
    api_key = os.environ["DAYTONA_API_KEY"]
    api_url = os.environ.get("DAYTONA_API_URL") or "https://app.daytona.io/api"
    url = api_url.rstrip("/") + path
    req = urllib.request.Request(
        url,
        method=method,
        headers={
            "Accept": "application/json",
            "Authorization": "Bearer " + api_key,
            "X-Daytona-Source": "slime-cleanup",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8")) if data else {}


def iter_remote_sandboxes(*, name_prefix: str | None, timeout: float, limit: int = 100):
    cursor = None
    while True:
        query = {"limit": limit}
        if cursor:
            query["cursor"] = cursor
        if name_prefix:
            query["name"] = name_prefix
        page = _rest_request("GET", "/sandbox?" + urllib.parse.urlencode(query), timeout=timeout)
        for item in page.get("items") or []:
            yield item
        cursor = page.get("nextCursor")
        if not cursor:
            break


def _sandbox_disk_gib(item: dict) -> float:
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


def _iso_age_min(ts: str | None, *, now_epoch: float) -> float | None:
    """Minutes since ISO-8601 timestamp ``ts`` (UTC), or None if unparseable."""
    if not ts:
        return None
    try:
        epoch = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None
    return (now_epoch - epoch) / 60.0


def audit_sandboxes(*, name_prefix: str | None, timeout: float, zombie_idle_min: float) -> int:
    """Read-only census of ALL remote sandboxes (full cursor pagination).

    Replaces page-1-only ``curl`` snapshots, which silently truncate at the
    100-item page cap and report a misleading "TOTAL=100, ZOMBIES=0" no matter
    how many sandboxes actually exist. A sandbox is flagged a zombie when it is
    still ``started`` AND has been idle longer than ``zombie_idle_min`` -- a
    real leak rather than a live long-tail trial. Exit code is the zombie count
    (capped at 250) so a patrol can branch on "leak vs clean".
    """
    now_epoch = datetime.datetime.now(datetime.timezone.utc).timestamp()
    total = 0
    by_state: dict[str, int] = {}
    by_auto_delete: dict[str, int] = {}
    zombies = 0
    oldest_min = 0.0
    never_stops = 0  # started + autoStopInterval==0 (the config that leaks forever)
    for item in iter_remote_sandboxes(name_prefix=name_prefix, timeout=timeout):
        total += 1
        state = str(item.get("state") or item.get("status") or "?")
        by_state[state] = by_state.get(state, 0) + 1
        adi = str(item.get("autoDeleteInterval"))
        by_auto_delete[adi] = by_auto_delete.get(adi, 0) + 1
        age = _iso_age_min(item.get("createdAt") or item.get("created_at"), now_epoch=now_epoch)
        idle = _iso_age_min(
            item.get("lastActivityAt") or item.get("updatedAt") or item.get("createdAt"),
            now_epoch=now_epoch,
        )
        if age is not None:
            oldest_min = max(oldest_min, age)
        started = state in {"started", "running"}
        if started and (item.get("autoStopInterval") in (0, "0", None)):
            never_stops += 1
        if started and idle is not None and idle >= zombie_idle_min:
            zombies += 1
    print(
        "AUDIT "
        f"total={total} zombies={zombies} never_stops={never_stops} "
        f"oldest_min={oldest_min:.0f} states={json.dumps(by_state, sort_keys=True)} "
        f"autoDeleteInterval={json.dumps(by_auto_delete, sort_keys=True)} "
        f"(zombie_idle_min={zombie_idle_min:g}, prefix={name_prefix!r})"
    )
    return min(zombies, 250)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-id", type=int, default=1)
    parser.add_argument("--max-group", type=int, default=8)
    parser.add_argument("--samples-per-group", type=int, default=32)
    parser.add_argument(
        "--action",
        choices=(
            "audit",
            "archive-direct",
            "delete-direct",
            "get-delete",
            "archive-rest",
            "delete-rest",
            "archive-all-rest",
            "delete-all-rest",
        ),
        default="archive-direct",
        help="audit = read-only full-pagination census (no deletes); "
        "*-rest uses direct REST calls; *-all-rest pages over remote sandboxes.",
    )
    parser.add_argument("--name-prefix", default=None)
    parser.add_argument(
        "--zombie-idle-min",
        type=float,
        default=15.0,
        help="audit: a started sandbox idle >= this many minutes is counted a zombie/leak.",
    )
    parser.add_argument(
        "--created-before",
        default=None,
        help="Only clean/list all-rest candidates with created timestamp before this ISO string.",
    )
    parser.add_argument("--request-timeout", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.action == "audit":
        return audit_sandboxes(
            name_prefix=args.name_prefix,
            timeout=args.request_timeout,
            zombie_idle_min=args.zombie_idle_min,
        )

    if args.action in {"archive-all-rest", "delete-all-rest"}:
        method = "POST" if args.action == "archive-all-rest" else "DELETE"
        suffix = "/archive" if args.action == "archive-all-rest" else ""
        items = list(
            iter_remote_sandboxes(
                name_prefix=args.name_prefix,
                timeout=args.request_timeout,
            )
        )
        if args.created_before:
            items = [
                item
                for item in items
                if (item.get("createdAt") or item.get("created_at") or item.get("created") or "")
                < args.created_before
            ]
        total_disk = sum(_sandbox_disk_gib(item) for item in items)
        print(f"remote candidates={len(items)} disk_gib={total_disk:g} prefix={args.name_prefix!r}")
        cleaned = 0
        errors = 0
        for item in items:
            sandbox_id = item.get("id")
            name = item.get("name")
            state = item.get("state")
            disk = _sandbox_disk_gib(item)
            created = item.get("createdAt") or item.get("created_at") or item.get("created")
            updated = item.get("updatedAt") or item.get("updated_at") or item.get("updated")
            labels = item.get("labels") or item.get("metadata") or {}
            print(
                "CANDIDATE "
                f"id={sandbox_id} name={name} state={state} disk_gib={disk:g} "
                f"created={created} updated={updated} labels={json.dumps(labels, sort_keys=True)[:400]}"
            )
            if args.dry_run:
                continue
            target = urllib.parse.quote(str(sandbox_id or name), safe="")
            try:
                _rest_request(method, f"/sandbox/{target}{suffix}", timeout=args.request_timeout)
                cleaned += 1
                print(f"CLEANED id={sandbox_id} name={name}")
            except Exception as exc:
                errors += 1
                print(f"ERROR clean id={sandbox_id} name={name}: {exc!r}")
        print(f"summary candidates={len(items)} cleaned={cleaned} errors={errors}")
        return 1 if errors else 0

    daytona = None
    if not args.dry_run and args.action in {"archive-direct", "delete-direct", "get-delete"}:
        try:
            from daytona import Daytona
        except ModuleNotFoundError as exc:
            parser.error(f"Daytona SDK is required for {args.action}: {exc}")
        daytona = Daytona()
    found = 0
    cleaned = 0
    missed = 0
    errors = 0

    for name in iter_trial_names(args.rollout_id, args.max_group, args.samples_per_group):
        if args.dry_run:
            print(f"CANDIDATE {name}")
            continue

        if args.action == "archive-direct":
            try:
                daytona._sandbox_api.archive_sandbox(name, _request_timeout=args.request_timeout)
                cleaned += 1
                print(f"ARCHIVED {name}")
            except Exception as exc:
                if "not found" in str(exc).lower() or exc.__class__.__name__.endswith("NotFoundError"):
                    missed += 1
                    continue
                errors += 1
                print(f"ERROR archive {name}: {exc!r}")
            continue

        if args.action == "archive-rest":
            try:
                target = urllib.parse.quote(name, safe="")
                _rest_request("POST", f"/sandbox/{target}/archive", timeout=args.request_timeout)
                cleaned += 1
                print(f"ARCHIVED {name}")
            except Exception as exc:
                if "not found" in str(exc).lower() or getattr(exc, "code", None) == 404:
                    missed += 1
                    continue
                errors += 1
                print(f"ERROR archive {name}: {exc!r}")
            continue

        if args.action == "delete-rest":
            try:
                target = urllib.parse.quote(name, safe="")
                _rest_request("DELETE", f"/sandbox/{target}", timeout=args.request_timeout)
                cleaned += 1
                print(f"DELETED {name}")
            except Exception as exc:
                if "not found" in str(exc).lower() or getattr(exc, "code", None) == 404:
                    missed += 1
                    continue
                errors += 1
                print(f"ERROR delete {name}: {exc!r}")
            continue

        if args.action == "delete-direct":
            try:
                daytona._sandbox_api.delete_sandbox(name, _request_timeout=args.request_timeout)
                cleaned += 1
                print(f"DELETED {name}")
            except Exception as exc:
                if "not found" in str(exc).lower() or exc.__class__.__name__.endswith("NotFoundError"):
                    missed += 1
                    continue
                errors += 1
                print(f"ERROR delete {name}: {exc!r}")
            continue

        try:
            sandbox = daytona.get(name)
        except Exception as exc:  # SDK raises typed not-found errors, but keep this robust.
            if "not found" in str(exc).lower() or exc.__class__.__name__.endswith("NotFoundError"):
                missed += 1
                continue
            errors += 1
            print(f"ERROR get {name}: {exc!r}")
            continue

        found += 1
        state = getattr(sandbox, "state", None)
        sandbox_id = getattr(sandbox, "id", None)
        try:
            daytona.delete(sandbox, timeout=args.request_timeout)
            cleaned += 1
            print(f"DELETED {name} id={sandbox_id} state={state}")
        except Exception as exc:
            errors += 1
            print(f"ERROR delete {name} id={sandbox_id}: {exc!r}")

    print(f"summary found={found} cleaned={cleaned} missed={missed} errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0)
