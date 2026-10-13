import sys
import types
from types import SimpleNamespace

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_distributed_stub = types.ModuleType("torch.distributed")
    torch_distributed_stub.is_initialized = lambda: False
    torch_distributed_stub.get_rank = lambda: 0
    torch_stub.dtype = type("dtype", (), {})
    torch_stub.Size = tuple
    torch_stub.Tensor = type("Tensor", (), {})
    torch_stub.distributed = torch_distributed_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.distributed"] = torch_distributed_stub

if "ray" not in sys.modules:
    ray_stub = types.ModuleType("ray")
    ray_stub._private = SimpleNamespace(services=SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1"))
    sys.modules["ray"] = ray_stub

if "slime.rollout.data_source" not in sys.modules:
    data_source_stub = types.ModuleType("slime.rollout.data_source")
    data_source_stub._harbor_test_stub = True

    class DataSource:
        pass

    data_source_stub.DataSource = DataSource
    sys.modules["slime.rollout.data_source"] = data_source_stub

from slime_plugins.harbor.data_source import HarborTaskDataSource

if getattr(sys.modules.get("slime.rollout.data_source"), "_harbor_test_stub", False):
    sys.modules.pop("slime.rollout.data_source", None)
    rollout_pkg = sys.modules.get("slime.rollout")
    if rollout_pkg is not None and getattr(getattr(rollout_pkg, "data_source", None), "_harbor_test_stub", False):
        delattr(rollout_pkg, "data_source")


def _make_task_dir(parent, name):
    task_dir = parent / name
    task_dir.mkdir()
    (task_dir / "task.toml").write_text("[task]\nname = \"x\"\n")
    return task_dir


def test_eval_pool_returns_full_set_without_touching_train_cursor(tmp_path):
    # Separate train and eval task trees.
    train_root = tmp_path / "train"
    eval_root = tmp_path / "eval"
    train_root.mkdir()
    eval_root.mkdir()
    for name in ["t-a", "t-b", "t-c"]:
        _make_task_dir(train_root, name)
    for name in ["e-a", "e-b"]:
        _make_task_dir(eval_root, name)

    args = SimpleNamespace(
        harbor_tasks_dir=str(train_root),
        harbor_eval_tasks_dir=str(eval_root),
        n_samples_per_prompt=4,
        n_samples_per_eval_prompt=1,
    )
    ds = HarborTaskDataSource(args)

    assert len(ds.eval_entries) == 2

    # Advance the training cursor first.
    ds.get_samples(1)
    cursor_before = (ds.sample_offset, ds.sample_group_index, ds.sample_index)
    assert cursor_before[0] == 1  # one training group consumed

    # Eval returns ALL eval tasks (the full fixed set), one group per task.
    eval_groups = ds.get_samples(0, evaluation=True)
    assert len(eval_groups) == 2
    # n_samples_per_eval_prompt=1 -> one sample per group, regardless of n_samples_per_prompt.
    assert all(len(group) == 1 for group in eval_groups)
    eval_paths = sorted(g[0].metadata["harbor_task_entry"]["task"]["path"] for g in eval_groups)
    assert eval_paths[0].endswith("e-a")
    assert eval_paths[1].endswith("e-b")

    # The training cursor must be untouched by the eval pass.
    assert (ds.sample_offset, ds.sample_group_index, ds.sample_index) == cursor_before


def test_eval_pool_respects_n_samples_per_eval_prompt(tmp_path):
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    _make_task_dir(eval_root, "e-a")
    _make_task_dir(eval_root, "e-b")

    args = SimpleNamespace(
        harbor_tasks_dir=str(eval_root),  # train pool nonempty so __init__ is happy
        harbor_eval_tasks_dir=str(eval_root),
        n_samples_per_prompt=1,
        n_samples_per_eval_prompt=3,
    )
    ds = HarborTaskDataSource(args)

    eval_groups = ds.get_samples(0, evaluation=True)
    assert len(eval_groups) == 2
    assert all(len(group) == 3 for group in eval_groups)


def test_eval_pool_falls_back_to_n_samples_per_prompt(tmp_path):
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    _make_task_dir(eval_root, "e-a")

    # No n_samples_per_eval_prompt set -> fall back to n_samples_per_prompt.
    args = SimpleNamespace(
        harbor_tasks_dir=str(eval_root),
        harbor_eval_tasks_dir=str(eval_root),
        n_samples_per_prompt=2,
    )
    ds = HarborTaskDataSource(args)

    eval_groups = ds.get_samples(0, evaluation=True)
    assert len(eval_groups) == 1
    assert len(eval_groups[0]) == 2


def test_no_eval_dir_means_empty_eval_pool_and_unchanged_training(tmp_path):
    train_root = tmp_path / "train"
    train_root.mkdir()
    _make_task_dir(train_root, "t-a")
    _make_task_dir(train_root, "t-b")

    # No eval dir configured at all.
    args = SimpleNamespace(harbor_tasks_dir=str(train_root), n_samples_per_prompt=2)
    ds = HarborTaskDataSource(args)

    assert ds.eval_entries == []

    # Training path behaves exactly as before.
    groups = ds.get_samples(2)
    assert len(groups) == 2
    assert all(len(g) == 2 for g in groups)

    # Asking for eval with no eval pool is a clear error, not a silent train fallback.
    try:
        ds.get_samples(0, evaluation=True)
    except ValueError as exc:
        assert "eval" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError when eval pool is empty")


def test_eval_group_indices_are_independent_across_passes(tmp_path):
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    _make_task_dir(eval_root, "e-a")
    _make_task_dir(eval_root, "e-b")

    args = SimpleNamespace(
        harbor_tasks_dir=str(eval_root),
        harbor_eval_tasks_dir=str(eval_root),
        n_samples_per_prompt=1,
        n_samples_per_eval_prompt=1,
    )
    ds = HarborTaskDataSource(args)

    first = ds.get_samples(0, evaluation=True)
    second = ds.get_samples(0, evaluation=True)

    # Both passes cover the full set...
    assert len(first) == len(second) == 2
    # ...and the eval counter advances monotonically (does not reset/collide with train).
    first_gi = [g[0].group_index for g in first]
    second_gi = [g[0].group_index for g in second]
    assert first_gi == [0, 1]
    assert second_gi == [2, 3]
    # Training cursor stayed at zero throughout.
    assert ds.sample_offset == 0
    assert ds.sample_group_index == 0
