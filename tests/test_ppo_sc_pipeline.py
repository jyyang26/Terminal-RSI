"""CPU integration of the real forward-only scheduler callback and data transport."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from slime.utils.types import Sample

NUM_GPUS = 0
pytestmark = pytest.mark.unit


def test_forward_only_emits_old_head_and_restores_dynamic_sample_order(monkeypatch):
    from megatron.core import mpu
    from slime.backends.megatron_utils import model as model_utils
    from slime.backends.megatron_utils.data import DataIterator

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda **kw: True)
    args = SimpleNamespace(
        enable_score_centering=True,
        sc_formula="ppo_weighted",
        sc_weight_denominator="old_actor",
        sc_top_k=2,
        sc_chunk_size=2,
        qkv_format="thd",
        allgather_cp=False,
        data_pad_size_multiplier=1,
        use_rollout_entropy=False,
        custom_megatron_before_log_prob_hook_path=None,
        seq_length=8,
        micro_batch_size=1,
        use_dynamic_batch_size=True,
    )
    data = {
        "tokens": [torch.tensor([1, 2, 3, 4]), torch.tensor([2, 3, 4, 5, 6])],
        "loss_masks": [torch.tensor([1, 1]), torch.tensor([1, 0])],
        "total_lengths": [4, 5],
        "response_lengths": [2, 2],
        "rollout_topk_token_ids": [torch.tensor([[3, 0], [4, 0]]), torch.tensor([[5, 0], [6, 0]])],
        "sc_response_token_ids": [torch.tensor([3, 4]), torch.tensor([5, 6])],
    }
    iterator = DataIterator(data, [[1], [0]])

    def get_batch(it, keys, *unused):
        batch = it.get_next(keys)
        batch["unconcat_tokens"] = batch["tokens"]
        batch["packed_seq_params"] = None
        batch["full_loss_masks"] = torch.ones((1, batch["total_lengths"][0]))
        batch["tokens"] = batch["tokens"][0][None]
        return batch

    monkeypatch.setattr(model_utils, "get_batch", get_batch)
    monkeypatch.setattr(model_utils, "get_model_config", lambda model: SimpleNamespace(timers=None))
    monkeypatch.setattr(model_utils, "_disable_tqdm_for_non_main_rank", lambda: True)
    monkeypatch.setattr(model_utils, "_wrap_forward_step_with_microbatch_pbar", lambda fn, pbar: fn)

    class Model:
        def eval(self):
            pass

        def train(self):
            pass

        def __call__(self, **kw):
            values = kw["input_ids"].float()
            return (
                torch.arange(8, dtype=torch.float32)[None, None, :]
                + values[:, :, None] * torch.arange(8)[None, None, :] / 10
            )

    model = Model()

    def scheduler(**kw):
        result = []
        for unused in range(kw["num_microbatches"]):
            logits, callback = kw["forward_step_func"](kw["data_iterator"][0], kw["model"][0])
            empty, output = callback(logits)
            result.append(output)
        return result

    monkeypatch.setattr(model_utils, "get_forward_backward_func", lambda: scheduler)

    def base_callback(logits, **kw):
        return torch.empty(0), {"log_probs": [torch.zeros(r) for r in kw["response_lengths"]]}

    outputs = model_utils.forward_only(base_callback, args, [model], [iterator], [2])
    assert len(outputs["sc_old_head_log_probs"]) == 2
    for i, actual in enumerate(outputs["sc_old_head_log_probs"]):
        full = model(input_ids=data["tokens"][i][None])[0]
        expected = full[-3:-1].log_softmax(-1).gather(-1, data["rollout_topk_token_ids"][i])
        valid = data["loss_masks"][i].bool()
        torch.testing.assert_close(actual[valid], expected[valid])
    args.enable_score_centering = False
    outputs = model_utils.forward_only(base_callback, args, [model], [iterator], [2])
    assert set(outputs) == {"log_probs"}


def manager_class():
    from slime.ray.rollout import RolloutManager

    return RolloutManager.__ray_metadata__.modified_class


def test_sample_pack_dp_partition_and_off_payload(monkeypatch):
    from slime.ray import rollout as module

    cls = manager_class()
    manager = object.__new__(cls)
    manager.args = SimpleNamespace(
        enable_score_centering=True, sc_top_k=2, sc_max_batch_bytes=10000, global_batch_size=2
    )
    manager.train_parallel_config = {"dp_size": 2}
    manager.custom_convert_samples_to_train_data_func = None
    manager._drop_zero_reward_groups_for_static_dp_alignment = lambda values: values
    manager._post_process_rewards = lambda samples: ([s.reward for s in samples], [s.reward for s in samples])
    manager._allow_partial_global_batch = lambda: True
    samples = []
    for index in range(3):
        sample = Sample(
            index=index,
            group_id=index,
            tokens=[1, 2, 3, 4],
            response_length=2,
            reward=float(index),
            loss_mask=[1, 1],
            rollout_log_probs=[-0.5, -0.5],
        )
        sample.rollout_topk_token_ids = np.array([[3, 0], [4, 0]], dtype="<i4")
        sample.rollout_topk_log_probs = np.array([[-0.5, -10], [-0.5, -10]], dtype="<f4")
        sample.sc_response_token_ids = np.array([3, 4], dtype="<i4")
        samples.append(sample)
    packed = manager._convert_samples_to_train_data(samples)
    captured = []
    monkeypatch.setattr(
        module, "build_dp_schedule", lambda *args, **kw: ([[2, 0], [1]], [[[0], [1]], [[0]]], [2], [3])
    )
    monkeypatch.setattr(module.ray, "put", lambda value: captured.append(value) or len(captured))
    manager._split_train_data_by_dp(packed)
    for payload, partition in zip(captured, [[2, 0], [1]], strict=True):
        assert payload["sample_indices"] == partition
        for offset, index in enumerate(partition):
            assert payload["rollout_topk_token_ids"][offset] is samples[index].rollout_topk_token_ids
    manager.args.enable_score_centering = False
    baseline = manager._convert_samples_to_train_data(samples)
    assert not any(
        key in baseline for key in ("rollout_topk_token_ids", "rollout_topk_log_probs", "sc_response_token_ids")
    )
    manager.args.enable_score_centering = True
    samples[1].sc_response_token_ids[0] = 8
    with pytest.raises(ValueError, match="token IDs changed"):
        manager._convert_samples_to_train_data(samples)


def test_actor_and_critic_head_ownership(monkeypatch):
    from slime.backends.megatron_utils import actor as module

    monkeypatch.setattr(module, "process_rollout_data", lambda *args: copy.deepcopy(source))
    monkeypatch.setattr(module.mpu, "get_data_parallel_rank", lambda **kw: 0)
    monkeypatch.setattr(module.mpu, "get_data_parallel_world_size", lambda **kw: 1)
    monkeypatch.setattr(module.torch.cuda, "current_device", lambda: "cpu")
    source = {
        "tokens": [[1, 2, 3]],
        "response_lengths": [1],
        "total_lengths": [3],
        "loss_masks": [[1]],
        "rollout_topk_token_ids": [np.array([[3, 0]], dtype="<i4")],
        "rollout_topk_log_probs": [np.array([[-0.5, -10]], dtype="<f4")],
        "sc_response_token_ids": [np.array([3], dtype="<i4")],
    }
    instance = object.__new__(module.MegatronTrainRayActor)
    instance.args = SimpleNamespace(enable_score_centering=True, qkv_format="thd")
    instance.role = "actor"
    monkeypatch.setattr(module.mpu, "is_pipeline_last_stage", lambda **kw: True)
    result = instance._get_rollout_data(None)
    assert result["rollout_topk_token_ids"][0].device.type == "cpu"
    result["rollout_topk_token_ids"][0][0, 0] = 7
    assert source["rollout_topk_token_ids"][0][0, 0] == 3
    monkeypatch.setattr(module.mpu, "is_pipeline_last_stage", lambda **kw: False)
    assert "rollout_topk_token_ids" not in instance._get_rollout_data(None)
    instance.role = "critic"
    instance.args.enable_score_centering = False
    assert "rollout_topk_token_ids" not in instance._get_rollout_data(None)


def test_real_slime_validation_loads_common_sc_before_role_copies(monkeypatch, tmp_path):
    import sys
    import yaml
    from slime.utils.arguments import (
        get_slime_extra_args_provider,
        slime_validate_args,
        _pre_parse_mode,
        parse_megatron_role_args,
    )
    from slime.backends.megatron_utils.arguments import megatron_parse_args
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    original = root / "examples/terminal_rsi/configs/megatron_roles_rpr1.yaml"
    derived = root / "tests/fixtures/megatron_roles_rpr1_sc_shadow.yaml"
    for role_file, enabled in ((original, False), (derived, True)):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "test",
                "--rollout-batch-size",
                "8",
                "--num-rollout",
                "1",
                "--advantage-estimator",
                "ppo",
                "--loss-type",
                "custom_loss",
                "--custom-loss-function-path",
                "slime_plugins.harbor.train_loss.policy_loss_no_entropy_when_disabled",
                "--rollout-function-path",
                "slime_plugins.harbor.rollout.generate_rollout",
                "--megatron-config-path",
                str(role_file),
                "--keep-old-actor",
            ],
        )
        args = megatron_parse_args(get_slime_extra_args_provider(), skip_hf_validate=True)
        for key, value in vars(_pre_parse_mode()).items():
            setattr(args, key, value)
        # Actual custom YAML merge, not manually setting SC after validation.
        config = tmp_path / "harbor.yaml"
        config.write_text(yaml.safe_dump({"harbor_sample_mode": "trajectory_retokenized"}))
        args.custom_config_path = str(config)
        slime_validate_args(args)
        actor = parse_megatron_role_args(args, str(role_file), "actor")
        critic = parse_megatron_role_args(args, str(role_file), "critic")
        assert args.enable_score_centering is enabled and actor.enable_score_centering is enabled
        assert critic.enable_score_centering is False
        assert critic.lr / actor.lr == pytest.approx(30.0)
