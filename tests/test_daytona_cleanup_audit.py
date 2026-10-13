"""CPU unit tests for the Daytona cleanup script's read-only ``audit`` action.

The audit action replaced the page-1-only ``curl`` snapshots that silently
truncated at the API's 100-item page cap (reporting "TOTAL=100, ZOMBIES=0"
regardless of the true count -- the blind spot that hid 308 leaked sandboxes
after tb2of_raw_20260620_062054 finished). These tests pin the classification:
full pagination is exercised by the real ``iter_remote_sandboxes`` paginator
elsewhere; here we feed synthetic sandboxes and assert the leak-vs-live logic.
"""

import datetime
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_cleanup_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "harbor_terminal_rl"
        / "daytona_cleanup_slime_sandboxes.py"
    )
    module_name = "test_daytona_cleanup_audit_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _iso_minutes_ago(minutes: float) -> str:
    ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=minutes)
    return ts.isoformat().replace("+00:00", "Z")


def _sandbox(*, created_min, idle_min, state="started", auto_stop=0, auto_delete=90):
    return {
        "id": f"sb-{created_min}-{idle_min}",
        "state": state,
        "autoStopInterval": auto_stop,
        "autoDeleteInterval": auto_delete,
        "createdAt": _iso_minutes_ago(created_min),
        "lastActivityAt": _iso_minutes_ago(idle_min),
    }


def test_iso_age_min_parses_and_handles_garbage():
    mod = _load_cleanup_module()
    now = datetime.datetime.now(datetime.timezone.utc).timestamp()
    age = mod._iso_age_min(_iso_minutes_ago(30), now_epoch=now)
    assert age is not None and 29.0 <= age <= 31.0
    assert mod._iso_age_min(None, now_epoch=now) is None
    assert mod._iso_age_min("not-a-timestamp", now_epoch=now) is None


def test_audit_flags_idle_started_sandboxes_as_zombies(monkeypatch):
    mod = _load_cleanup_module()
    # 3 live (idle<15min) + 4 leaked (started, idle long) + 1 stopped (not a zombie).
    fleet = (
        [_sandbox(created_min=10, idle_min=2) for _ in range(3)]
        + [_sandbox(created_min=200, idle_min=120) for _ in range(4)]
        + [_sandbox(created_min=300, idle_min=200, state="stopped")]
    )
    monkeypatch.setattr(mod, "iter_remote_sandboxes", lambda **_kw: iter(fleet))
    rc = mod.audit_sandboxes(name_prefix=None, timeout=5.0, zombie_idle_min=15.0)
    # exit code == zombie count: only the 4 started+idle ones (stopped excluded).
    assert rc == 4


def test_audit_returns_zero_when_clean(monkeypatch):
    mod = _load_cleanup_module()
    monkeypatch.setattr(mod, "iter_remote_sandboxes", lambda **_kw: iter([]))
    assert mod.audit_sandboxes(name_prefix=None, timeout=5.0, zombie_idle_min=15.0) == 0


def test_audit_young_fleet_not_flagged(monkeypatch):
    mod = _load_cleanup_module()
    # 40 young, busy sandboxes (the healthy steady-state of a live run).
    fleet = [_sandbox(created_min=20, idle_min=1) for _ in range(40)]
    monkeypatch.setattr(mod, "iter_remote_sandboxes", lambda **_kw: iter(fleet))
    assert mod.audit_sandboxes(name_prefix=None, timeout=5.0, zombie_idle_min=15.0) == 0


def test_audit_caps_exit_code_at_250(monkeypatch):
    mod = _load_cleanup_module()
    fleet = [_sandbox(created_min=200, idle_min=120) for _ in range(400)]
    monkeypatch.setattr(mod, "iter_remote_sandboxes", lambda **_kw: iter(fleet))
    # Real leak count is 400, but exit code is capped so it stays a valid status.
    assert mod.audit_sandboxes(name_prefix=None, timeout=5.0, zombie_idle_min=15.0) == 250


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
