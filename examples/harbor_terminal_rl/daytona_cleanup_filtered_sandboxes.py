#!/usr/bin/env python3
"""Audit/delete Daytona sandboxes with conservative label and time filters."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import urllib.parse
import urllib.request


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
            "X-Daytona-Source": "slime-filtered-cleanup",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8")) if data else {}


def _iter_remote_sandboxes(*, name_prefix: str | None, timeout: float, limit: int = 100):
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


def _iso_epoch(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return None


def _item_labels(item: dict) -> dict:
    labels = item.get("labels")
    if isinstance(labels, dict):
        return labels
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return {}


def _parse_label_filters(values: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --label {value!r}; expected key=value")
        key, label_value = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid --label {value!r}; empty key")
        filters[key] = label_value
    return filters


def _sandbox_disk_gib(item: dict) -> float:
    resources = item.get("resources") or {}
    for source in (item, resources):
        for key in ("disk", "diskGiB", "disk_gib"):
            value = source.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0
    return 0.0


def _matches_filters(
    item: dict,
    *,
    label_filters: dict[str, str],
    harbor_sessions: set[str] | None,
    created_after_epoch: float | None,
    created_before_epoch: float | None,
) -> bool:
    labels = _item_labels(item)
    for key, value in label_filters.items():
        if str(labels.get(key)) != value:
            return False
    if harbor_sessions is not None and str(labels.get("harbor_session")) not in harbor_sessions:
        return False

    if created_after_epoch is None and created_before_epoch is None:
        return True

    created = _iso_epoch(item.get("createdAt") or item.get("created_at") or item.get("created"))
    if created is None:
        return False
    if created_after_epoch is not None and created < created_after_epoch:
        return False
    if created_before_epoch is not None and created >= created_before_epoch:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("audit", "delete"), default="audit")
    parser.add_argument("--name-prefix", default=None)
    parser.add_argument("--label", action="append", default=[], help="Require key=value label; repeatable.")
    parser.add_argument(
        "--harbor-session-file",
        default=None,
        help="Only touch sandboxes whose harbor_session label is listed in this file.",
    )
    parser.add_argument("--created-after", default=None)
    parser.add_argument("--created-before", default=None)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    args = parser.parse_args()

    try:
        label_filters = _parse_label_filters(args.label)
    except ValueError as exc:
        parser.error(str(exc))

    created_after_epoch = _iso_epoch(args.created_after)
    created_before_epoch = _iso_epoch(args.created_before)
    if args.created_after and created_after_epoch is None:
        parser.error(f"invalid --created-after timestamp: {args.created_after!r}")
    if args.created_before and created_before_epoch is None:
        parser.error(f"invalid --created-before timestamp: {args.created_before!r}")
    harbor_sessions = None
    if args.harbor_session_file:
        with open(args.harbor_session_file, encoding="utf-8") as fh:
            harbor_sessions = {line.strip() for line in fh if line.strip()}

    items = [
        item
        for item in _iter_remote_sandboxes(name_prefix=args.name_prefix, timeout=args.request_timeout)
        if _matches_filters(
            item,
            label_filters=label_filters,
            harbor_sessions=harbor_sessions,
            created_after_epoch=created_after_epoch,
            created_before_epoch=created_before_epoch,
        )
    ]
    total_disk = sum(_sandbox_disk_gib(item) for item in items)
    print(
        "FILTERED_CANDIDATES "
        f"count={len(items)} disk_gib={total_disk:g} action={args.action} "
        f"prefix={args.name_prefix!r} labels={json.dumps(label_filters, sort_keys=True)} "
        f"created_after={args.created_after!r} created_before={args.created_before!r}"
    )

    cleaned = 0
    errors = 0
    for item in items:
        sandbox_id = item.get("id")
        name = item.get("name")
        state = item.get("state") or item.get("status")
        disk = _sandbox_disk_gib(item)
        created = item.get("createdAt") or item.get("created_at") or item.get("created")
        updated = item.get("updatedAt") or item.get("updated_at") or item.get("updated")
        labels = _item_labels(item)
        print(
            "CANDIDATE "
            f"id={sandbox_id} name={name} state={state} disk_gib={disk:g} "
            f"created={created} updated={updated} labels={json.dumps(labels, sort_keys=True)[:400]}"
        )
        if args.action == "audit":
            continue
        target = urllib.parse.quote(str(sandbox_id or name), safe="")
        try:
            _rest_request("DELETE", f"/sandbox/{target}", timeout=args.request_timeout)
            cleaned += 1
            print(f"CLEANED id={sandbox_id} name={name}")
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"ERROR clean id={sandbox_id} name={name}: {exc!r}")

    print(f"summary candidates={len(items)} cleaned={cleaned} errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0)
