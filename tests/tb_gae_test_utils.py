"""Frozen-source and CPU harness shared by the TB-GAE release tests."""

import ast
import copy
import hashlib
import json
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = Path(__file__).parent / "fixtures/tb_gae_pre_patch"


def old_source(relative):
    manifest = json.loads((SNAPSHOTS / "manifest.json").read_text())
    entry = manifest[relative]
    source = (SNAPSHOTS / entry["snapshot"]).read_bytes()
    assert hashlib.sha256(source).hexdigest() == entry["sha256"], relative
    return source.decode()


def before_module(relative, package):
    module = types.ModuleType(package + ".pre_tb_baseline")
    module.__package__ = package
    exec(compile(old_source(relative), relative + "[pre-TB]", "exec"), module.__dict__)
    return module


def frozen_method(relative, class_name, name, globals_dict):
    tree = ast.parse(old_source(relative))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    env = dict(globals_dict)
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative + "[pre-TB]", "exec"), env)
    return env[name]


def assert_bits_equal(first, second):
    assert type(first) is type(second), (type(first), type(second))
    if isinstance(first, torch.Tensor):
        assert first.dtype == second.dtype and first.shape == second.shape
        assert torch.equal(
            first.detach().contiguous().reshape(-1).view(torch.uint8),
            second.detach().contiguous().reshape(-1).view(torch.uint8),
        )
    elif isinstance(first, np.ndarray):
        assert first.dtype == second.dtype and first.shape == second.shape and first.tobytes() == second.tobytes()
    elif isinstance(first, dict):
        assert list(first) == list(second)
        for key in first:
            assert_bits_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            assert_bits_equal(a, b)
    else:
        assert first == second


def detail(kind="normal"):
    data = {
        "prompt_token_ids": [[10, 11], [10, 11, 20, 30, 31]],
        "completion_token_ids": [[20], [40, 41]],
        "logprobs": [[-0.5], [-0.7, -0.8]],
    }
    if kind == "adjacent":
        data["prompt_token_ids"][1] = [10, 11, 20]
    if kind == "split":
        data["prompt_token_ids"][1] = [70, 71]
    if kind == "retokenized":
        data = {
            "prompt_token_ids": [[10, 11, 900, 901], [10, 11, 200, 40, 900, 901]],
            "completion_token_ids": [list(range(100, 140)), [50]],
            "logprobs": [[-i / 100 for i in range(100, 140)], [-0.5]],
            "extra": {
                "completion_text": ["same", "next"],
                "last_assistant_message_content": [None, "same"],
                "last_assistant_message_span": [None, [2, 3]],
            },
        }
    return copy.deepcopy(data)


def tb_args(**changes):
    args = Namespace(
        enable_tb_gae=True,
        tb_gae_gamma_turn=1.0,
        tb_gae_lambda_turn=1.0,
        advantage_estimator="ppo",
        use_critic=True,
        compute_advantages_and_returns=True,
        qkv_format="thd",
        allgather_cp=False,
        context_parallel_size=1,
        kl_coef=0.0,
        use_opd=False,
        enable_score_centering=False,
        harbor_sample_mode="trajectory_retokenized",
        custom_advantage_function_path=None,
        gamma=1.0,
        lambd=1.0,
        normalize_advantages=False,
        use_rollout_logprobs=False,
        rollout_temperature=1.0,
        log_probs_chunk_size=-1,
        eps_clip=0.3,
        eps_clip_high=5.0,
        entropy_coef=0.0,
        use_opsm=False,
        use_tis=False,
        use_kl_loss=False,
        use_unbiased_kl=False,
        get_mismatch_metrics=False,
        custom_tis_function_path=None,
        calculate_per_token_loss=False,
        kl_loss_coef=0.0,
        kl_loss_type="low_var_kl",
        keep_old_actor=True,
        harbor_report_entropy="0",
        value_clip=4.0,
        log_multi_turn=False,
        log_passrate=False,
        log_correct_samples=False,
    )
    vars(args).update(changes)
    return args


@pytest.fixture
def cpu_mpu(monkeypatch, tmp_path):
    from megatron.core import mpu

    assert not dist.is_initialized(), "TB tests require an isolated CPU process group"
    dist.init_process_group("gloo", init_method=(tmp_path / "gloo").as_uri(), rank=0, world_size=1)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_context_parallel_group", lambda: dist.group.WORLD)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: dist.group.WORLD)
    monkeypatch.setattr(mpu, "get_data_parallel_group", lambda **kw: dist.group.WORLD)
    monkeypatch.setattr(mpu, "get_data_parallel_group_gloo", lambda **kw: dist.group.WORLD)
    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda **kw: True)
    try:
        yield mpu
    finally:
        dist.destroy_process_group()
