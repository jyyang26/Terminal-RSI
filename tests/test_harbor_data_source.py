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


def test_harbor_task_data_source_discovers_task_dirs(tmp_path):
    for name in ["b-task", "a-task"]:
        task_dir = tmp_path / name
        task_dir.mkdir()
        (task_dir / "task.toml").write_text("[task]\nname = \"x\"\n")

    args = SimpleNamespace(harbor_tasks_dir=str(tmp_path), harbor_tasks_limit=1, n_samples_per_prompt=2)
    data_source = HarborTaskDataSource(args)

    groups = data_source.get_samples(1)

    assert len(data_source) == 1
    assert len(groups) == 1
    assert len(groups[0]) == 2
    entry = groups[0][0].metadata["harbor_task_entry"]
    assert entry["task"]["path"].endswith("a-task")


def test_harbor_task_data_source_requeues_samples_before_new_entries(tmp_path):
    task_dir = tmp_path / "task-a"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text("[task]\nname = \"x\"\n")

    args = SimpleNamespace(harbor_tasks_dir=str(tmp_path), n_samples_per_prompt=1)
    data_source = HarborTaskDataSource(args)

    first_group = data_source.get_samples(1)[0]
    first_group[0].metadata["requeued"] = True
    data_source.add_samples([first_group])

    groups = data_source.get_samples(2)

    assert groups[0][0].metadata["requeued"] is True
    assert groups[1][0].metadata["harbor_task_entry"]["task"]["path"].endswith("task-a")
