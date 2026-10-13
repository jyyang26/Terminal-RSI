"""CPU unit test for the Gloo control-group timeout.

``init_gloo_group`` (slime/utils/distributed_utils.py) must create the slime-wide
Gloo barrier group with an EXPLICIT timeout, rather than inheriting Gloo's
built-in 30-min default. That default is what tore down the cp2_64k PPO run on
2026-06-30: an actor rank stalled on the ``update_weights`` gloo barrier and the
30-min send-timeout fired (``gloo/transport/tcp/unbound_buffer.cc:129: Timed out
waiting 1800000ms``). The NCCL timeout fix in ``reloadable_process_group``
deliberately skips gloo groups, so gloo needs its own knob.

These tests pin: (1) the default is 60 min, (2) ``SLIME_GLOO_GROUP_TIMEOUT_MINUTES``
overrides it, and (3) the group actually created by ``init_gloo_group`` carries
that timeout (not Gloo's 1800s default). The real-group test runs a single-rank
gloo process group on CPU -- no GPU/NCCL needed.
"""

from __future__ import annotations

import importlib
import os
from datetime import timedelta

import pytest


def _fresh_module(monkeypatch, minutes: str | None):
    if minutes is None:
        monkeypatch.delenv("SLIME_GLOO_GROUP_TIMEOUT_MINUTES", raising=False)
    else:
        monkeypatch.setenv("SLIME_GLOO_GROUP_TIMEOUT_MINUTES", minutes)
    import slime.utils.distributed_utils as du

    return importlib.reload(du)


def test_default_timeout_is_60_min(monkeypatch):
    du = _fresh_module(monkeypatch, None)
    assert du._gloo_group_timeout() == timedelta(minutes=60)


def test_env_override(monkeypatch):
    du = _fresh_module(monkeypatch, "120")
    assert du._gloo_group_timeout() == timedelta(minutes=120)


def test_default_is_not_gloo_builtin_30min(monkeypatch):
    # The whole point: must differ from Gloo's 1800s (30-min) built-in default.
    du = _fresh_module(monkeypatch, None)
    assert du._gloo_group_timeout() != timedelta(minutes=30)


def test_created_group_carries_timeout(monkeypatch):
    """End-to-end: the PG init_gloo_group builds reflects our timeout."""
    import torch
    import torch.distributed as dist

    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29557")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("SLIME_GLOO_GROUP_TIMEOUT_MINUTES", "60")

    du = _fresh_module(monkeypatch, "60")
    dist.init_process_group(backend="gloo", timeout=timedelta(minutes=5))
    try:
        du.GLOO_GROUP = None  # ensure a fresh build on this rank
        group = du.init_gloo_group()
        backend = group._get_backend(torch.device("cpu"))
        # Gloo backend exposes its op timeout; must be our 60 min, not 30.
        assert backend.options._timeout == timedelta(minutes=60)
        # And the barrier actually runs on the timed group (single-rank no-op).
        dist.barrier(group=group)
    finally:
        du.GLOO_GROUP = None
        dist.destroy_process_group()
