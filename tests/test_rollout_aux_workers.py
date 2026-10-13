from __future__ import annotations

import importlib
import sys
import types

import pytest

NUM_GPUS = 0


def _install_placement_group_import_stubs(monkeypatch):
    ray_module = types.ModuleType("ray")

    def remote(*args, **kwargs):
        if args and isinstance(args[0], type):
            return args[0]

        def decorator(obj):
            return obj

        return decorator

    ray_module.remote = remote
    ray_module.get = lambda value: value
    ray_module.kill = lambda *args, **kwargs: None
    ray_module.util = types.SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1")

    ray_util_module = types.ModuleType("ray.util")
    ray_placement_group_module = types.ModuleType("ray.util.placement_group")
    ray_placement_group_module.placement_group = lambda *args, **kwargs: object()
    ray_scheduling_module = types.ModuleType("ray.util.scheduling_strategies")
    ray_scheduling_module.PlacementGroupSchedulingStrategy = object

    actor_group_module = types.ModuleType("slime.ray.actor_group")
    actor_group_module.RayTrainGroup = object
    rollout_module = types.ModuleType("slime.ray.rollout")
    rollout_module.RolloutManager = object

    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util_module)
    monkeypatch.setitem(sys.modules, "ray.util.placement_group", ray_placement_group_module)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", ray_scheduling_module)
    monkeypatch.setitem(sys.modules, "slime.ray.actor_group", actor_group_module)
    monkeypatch.setitem(sys.modules, "slime.ray.rollout", rollout_module)
    sys.modules.pop("slime.ray.placement_group", None)


def test_unique_node_bundles_returns_first_bundle_per_node(monkeypatch):
    _install_placement_group_import_stubs(monkeypatch)
    placement_group = importlib.import_module("slime.ray.placement_group")

    try:
        assert placement_group.unique_node_bundles(
            [8, 9, 0, 1, 16],
            ["10.0.0.2", "10.0.0.2", "10.0.0.1", "10.0.0.1", "10.0.0.3"],
        ) == [(8, "10.0.0.2"), (0, "10.0.0.1"), (16, "10.0.0.3")]
    finally:
        sys.modules.pop("slime.ray.placement_group", None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
