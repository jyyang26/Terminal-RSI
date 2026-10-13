#!/usr/bin/env python3
"""Validate the Daytona leak-fix primitives end-to-end against the live API.

This tests the EXACT mechanism the source fix in
harbor/src/harbor/environments/daytona.py now relies on:

  1. create a sandbox WITH a `harbor_session` label  (改动1)
  2. find it via daytona.list(labels={"harbor_session": ...})  (改动2 lookup)
  3. delete it via daytona.delete(sandbox)  (改动2 delete)
  4. confirm it's gone (list returns empty)

If all four steps pass, the label-based orphan-delete fallback that replaces
the leaky `stop()` skip-path is sound. Pure SDK, no cluster needed.

Run:
  set -a; source <daytona_tier4.env>; set +a
  HTTPS_PROXY=... HTTP_PROXY=... python validate_daytona_label_fix.py
"""
import asyncio
import sys
import time


SESSION = f"slime-leakfix-test-{int(time.time())}"
LABELS = {"harbor_managed": "1", "harbor_session": SESSION}


async def _list_by_label(daytona, labels):
    """Version-agnostic list (mirrors the fix in daytona.py)."""
    import inspect

    def has_labels(sb):
        sbl = getattr(sb, "labels", None) or {}
        return all(sbl.get(k) == v for k, v in labels.items())

    if inspect.isasyncgenfunction(daytona.list):
        try:
            from daytona import ListSandboxesQuery
            gen = daytona.list(ListSandboxesQuery(labels=labels))
        except Exception:
            gen = daytona.list()
        return [s async for s in gen if has_labels(s)]
    result = daytona.list(labels=labels)
    if inspect.isawaitable(result):
        result = await result
    items = getattr(result, "items", None)
    if items is None:
        items = result if isinstance(result, list) else []
    return [s for s in items if has_labels(s)]


async def main() -> int:
    try:
        from daytona import (
            AsyncDaytona,
            CreateSandboxFromImageParams,
            Image,
            Resources,
        )
    except Exception as e:  # noqa: BLE001
        print(f"FAIL import daytona SDK: {e!r}")
        return 2

    import daytona as _d
    print(f"[0] daytona SDK at {_d.__file__}")

    daytona = AsyncDaytona()
    created = None
    ok = True
    try:
        # --- 1. create WITH label -------------------------------------------
        print(f"[1] creating sandbox with label harbor_session={SESSION} ...")
        params = CreateSandboxFromImageParams(
            image=Image.base("ubuntu:22.04"),
            resources=Resources(cpu=1, memory=2, disk=5),
            labels=dict(LABELS),
            auto_delete_interval=60,
        )
        created = await asyncio.wait_for(daytona.create(params=params, timeout=120), timeout=180)
        print(f"    created id={getattr(created, 'id', '?')} labels={getattr(created, 'labels', {})}")

        # --- 2. find BY label (this is what stop() fallback does) -----------
        print(f"[2] listing by label harbor_session={SESSION} (version-agnostic) ...")
        items = await _list_by_label(daytona, {"harbor_session": SESSION})
        found_ids = [getattr(s, "id", None) for s in items]
        print(f"    found {len(items)} sandbox(es): {found_ids}")
        if getattr(created, "id", None) not in found_ids:
            print("    FAIL: created sandbox NOT found by label -> fallback would miss it")
            ok = False

        # --- 3. delete the listed sandbox -----------------------------------
        if items:
            print(f"[3] deleting {len(items)} sandbox(es) found by label ...")
            for s in items:
                await daytona.delete(s)
                print(f"    deleted id={getattr(s, 'id', '?')}")
            created = None  # already deleted via label path

        # --- 4. confirm gone -------------------------------------------------
        await asyncio.sleep(3)
        remaining = await _list_by_label(daytona, {"harbor_session": SESSION})
        print(f"[4] post-delete list by label: {len(remaining)} remaining")
        if remaining:
            print("    FAIL: sandbox still present after label-based delete")
            ok = False
    except Exception as e:  # noqa: BLE001
        print(f"FAIL during test: {e!r}")
        ok = False
    finally:
        # safety: if we created one and didn't delete it, clean up
        if created is not None:
            try:
                await daytona.delete(created)
                print(f"[cleanup] deleted leftover id={getattr(created, 'id', '?')}")
            except Exception as e:  # noqa: BLE001
                print(f"[cleanup] WARN could not delete leftover: {e!r}")

    print("RESULT:", "PASS ✅ label-based create/find/delete all work" if ok else "FAIL 🔴")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
