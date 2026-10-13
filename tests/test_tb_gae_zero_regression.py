"""Compare the real disabled path to captured pre-patch source, byte for byte."""

import builtins
import copy
import pickle

import numpy as np
import pytest
import torch
from tb_gae_test_utils import (
    assert_bits_equal,
    before_module,
    cpu_mpu,
    detail,
    frozen_method,
    old_source,
    tb_args,
)

from slime.utils.types import Sample

NUM_GPUS = 0
pytestmark = pytest.mark.unit


@pytest.fixture
def forbid_tb_runtime_import(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {"slime_plugins.harbor.tb_gae", "slime_plugins.harbor.tb_gae_data"}:
            raise AssertionError(f"Disabled path imported {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.mark.parametrize("switch", ["absent", False])
@pytest.mark.parametrize("kind", ["normal", "adjacent", "split", "retokenized"])
def test_sample_and_hard_cap_bitwise_against_old_source(switch, kind, forbid_tb_runtime_import):
    from slime_plugins.harbor import atif_adapter, sample_postprocess

    old = before_module("slime_plugins/harbor/atif_adapter.py", "slime_plugins.harbor")
    old_cap = before_module("slime_plugins/harbor/sample_postprocess.py", "slime_plugins.harbor")
    base = Sample(index=7, group_index=3, prompt="task", metadata={"source": "frozen"})
    kw = dict(reward=0.7, status=Sample.Status.COMPLETED, mode="trajectory_retokenized")
    new_kw = {} if switch == "absent" else {"enable_tb_gae": False}
    a = old.samples_from_rollout_details([detail(kind)], copy.deepcopy(base), **kw)
    b = atif_adapter.samples_from_rollout_details([detail(kind)], copy.deepcopy(base), **kw, **new_kw)
    args = tb_args(enable_tb_gae=False, harbor_max_train_total_tokens=4)
    if switch == "absent":
        del args.enable_tb_gae
    for x, y in zip(a, b, strict=True):
        assert_bits_equal(x.to_dict(), y.to_dict())
        assert "turn_end_mask" not in y.metadata and "is_real_done" not in y.__dict__
    old_cap.cap_total_tokens_allow_response_trim_in_samples(args, a)
    sample_postprocess.cap_total_tokens_allow_response_trim_in_samples(args, b)
    for x, y in zip(a, b, strict=True):
        assert_bits_equal(x.to_dict(), y.to_dict())
        assert_bits_equal(y.to_dict(), Sample.from_dict(y.to_dict()).to_dict())


def make_manager(cls, enabled):
    obj = object.__new__(cls)
    obj.args = tb_args(
        enable_tb_gae=enabled, global_batch_size=4, use_dynamic_batch_size=False, micro_batch_size=1, balance_data=True
    )
    obj.train_parallel_config = {"dp_size": 2, "cp_size": 4, "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1}
    obj.custom_convert_samples_to_train_data_func = None
    obj._drop_zero_reward_groups_for_static_dp_alignment = lambda samples: samples
    obj._post_process_rewards = lambda samples: ([s.reward for s in samples], [s.reward for s in samples])
    obj._allow_partial_global_batch = lambda: False
    return obj


def transport_samples():
    return [
        Sample(
            index=i,
            group_id=i,
            tokens=list(range(4 + i)),
            response_length=2,
            reward=0.2 * i,
            loss_mask=[1, 1],
            rollout_log_probs=[-0.4, -0.7],
            remove_sample=i == 1,
        )
        for i in range(4)
    ]


@pytest.mark.parametrize("switch", ["absent", False])
def test_convert_and_real_dp_schedule_payload_bitwise(monkeypatch, switch, forbid_tb_runtime_import):
    from slime.ray import rollout as module

    cls = module.RolloutManager.__ray_metadata__.modified_class
    old = make_manager(cls, False)
    new = make_manager(cls, False)
    if switch == "absent":
        del new.args.enable_tb_gae
    convert = frozen_method("slime/ray/rollout.py", "RolloutManager", "_convert_samples_to_train_data", vars(module))
    split = frozen_method("slime/ray/rollout.py", "RolloutManager", "_split_train_data_by_dp", vars(module))
    a = convert(old, transport_samples())
    b = new._convert_samples_to_train_data(transport_samples())
    assert_bits_equal(a, b)
    captured = []
    monkeypatch.setattr(module.ray, "put", lambda payload: captured.append(copy.deepcopy(payload)) or len(captured))
    split(old, a)
    expected = captured[:]
    captured.clear()
    new._split_train_data_by_dp(b)
    assert_bits_equal(expected, captured)
    assert pickle.dumps(expected) == pickle.dumps(captured)
    assert all("turn_end_mask" not in p and "is_real_done" not in p for p in captured)
    assert sorted(i for p in captured for i in p["sample_indices"]) == list(range(4))


def test_enabled_dp_transport_uses_identical_partition(monkeypatch):
    from slime.ray import rollout as module

    cls = module.RolloutManager.__ray_metadata__.modified_class
    manager = make_manager(cls, True)
    samples = transport_samples()
    for sample in samples:
        sample.metadata.update(turn_end_mask=np.array([0, 1], dtype=bool), is_real_done=np.array([0, 1], dtype=bool))
    data = manager._convert_samples_to_train_data(samples)
    captured = []
    monkeypatch.setattr(module.ray, "put", lambda payload: captured.append(payload) or len(captured))
    manager._split_train_data_by_dp(data)
    for payload in captured:
        for i, index in enumerate(payload["sample_indices"]):
            assert payload["turn_end_mask"][i] is data["turn_end_mask"][index]
            assert payload["is_real_done"][i] is data["is_real_done"][index]
    del data["is_real_done"]
    with pytest.raises(ValueError, match="is_real_done"):
        manager._split_train_data_by_dp(data)


def test_enabled_adapter_to_dp_to_both_roles(cpu_mpu, monkeypatch):
    from slime.backends.megatron_utils.loss import compute_advantages_and_returns
    from slime.ray import rollout as module
    from slime.utils.arguments import _apply_megatron_role_overrides
    from slime_plugins.harbor.atif_adapter import samples_from_trial_result
    from slime_plugins.harbor.sample_postprocess import cap_total_tokens_allow_response_trim_in_samples

    cls = module.RolloutManager.__ray_metadata__.modified_class
    manager = make_manager(cls, True)
    samples = []
    for index in range(4):
        result = {
            "agent_result": {"rollout_details": [detail()], "metadata": {"termination_kind": "terminated"}},
            "verifier_result": {"rewards": {"reward": 0.7}},
        }
        converted = samples_from_trial_result(
            result, Sample(index=index, group_id=index), mode="trajectory_retokenized", enable_tb_gae=True
        )
        samples.extend(converted)
    # Keep one chunk per episode here; cross-DP multi-chunk continuation is
    # exercised by the real Gloo bootstrap test rather than a fake local DP.
    cap_total_tokens_allow_response_trim_in_samples(tb_args(harbor_max_train_total_tokens=16), samples)
    data = manager._convert_samples_to_train_data(samples)
    captured = []
    monkeypatch.setattr(module.ray, "put", lambda payload: captured.append(copy.deepcopy(payload)) or len(captured))
    manager._split_train_data_by_dp(data)
    for payload in captured:
        payload["total_lengths"] = [payload["total_lengths"][i] for i in payload.pop("partition")]
        payload["loss_masks"] = [torch.tensor(m) for m in payload["loss_masks"]]
        payload["rollout_log_probs"] = [torch.tensor(p) for p in payload["rollout_log_probs"]]
        payload["values"] = [torch.linspace(0.1, 0.3, r) for r in payload["response_lengths"]]
        outputs = []
        for role in ("actor", "critic"):
            current = copy.deepcopy(payload)
            compute_advantages_and_returns(_apply_megatron_role_overrides(tb_args(), {}, role), current)
            for ret, mask in zip(current["returns"], current["loss_masks"], strict=True):
                torch.testing.assert_close(ret[mask.bool()], torch.full_like(ret[mask.bool()], 0.7))
            outputs.append(current)
        assert_bits_equal(outputs[0]["returns"], outputs[1]["returns"])


def test_off_public_trial_entry_uses_no_tb_module(forbid_tb_runtime_import):
    from slime_plugins.harbor.atif_adapter import samples_from_trial_result

    old = before_module("slime_plugins/harbor/atif_adapter.py", "slime_plugins.harbor")
    result = {"agent_result": {"rollout_details": [detail()]}, "verifier_result": {"rewards": {"reward": 0.7}}}
    a = old.samples_from_trial_result(result, Sample(index=0), mode="trajectory_retokenized")
    b = samples_from_trial_result(result, Sample(index=0), mode="trajectory_retokenized", enable_tb_gae=False)
    assert_bits_equal([s.to_dict() for s in a], [s.to_dict() for s in b])


def loss_inputs():
    gen = torch.Generator().manual_seed(519)
    totals, responses = [9, 7], [4, 3]
    data = dict(
        total_lengths=totals,
        response_lengths=responses,
        unconcat_tokens=[torch.randint(0, 32, (t,), generator=gen) for t in totals],
        loss_masks=[torch.tensor([1, 0, 1, 1]), torch.tensor([1, 1, 1])],
        values=[torch.randn(r, generator=gen) for r in responses],
        rewards=[0.7, 1.2],
        log_probs=[torch.full((r,), -3.7) for r in responses],
        rollout_log_probs=[torch.full((r,), -3.9) for r in responses],
        ref_log_probs=[torch.full((r,), -3.8) for r in responses],
        group_mask_sums=None,
    )
    return data, torch.randn(1, sum(totals), 32, generator=gen)


@pytest.mark.parametrize("switch", ["absent", False])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("discount", [1.0, 0.97])
def test_targets_loss_all_logits_gradients_and_update_bitwise(
    cpu_mpu,
    monkeypatch,
    switch,
    normalize,
    discount,
    forbid_tb_runtime_import,
):
    from slime.backends.megatron_utils import loss as current
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
    from slime_plugins.harbor.train_loss import policy_loss_no_entropy_when_disabled

    prior = before_module("slime/backends/megatron_utils/loss.py", "slime.backends.megatron_utils")
    args = tb_args(enable_tb_gae=False, normalize_advantages=normalize, gamma=discount, lambd=discount)
    if switch == "absent":
        del args.enable_tb_gae
    first, logits = loss_inputs()
    second = copy.deepcopy(first)
    rng = torch.random.get_rng_state().clone()
    calls = []
    native = current.get_advantages_and_returns_batch
    monkeypatch.setattr(
        current, "get_advantages_and_returns_batch", lambda *a, **kw: calls.append(1) or native(*a, **kw)
    )
    prior.compute_advantages_and_returns(args, first)
    current.compute_advantages_and_returns(args, second)
    assert calls == [1]
    assert_bits_equal(rng, torch.random.get_rng_state())
    assert_bits_equal(first, second)
    reducer = get_sum_of_sample_mean(first["total_lengths"], first["response_lengths"], first["loss_masks"])
    # Exercise both real policy implementations and the real critic loss.
    for old_fn, new_fn, width in (
        (prior.policy_loss_function, current.policy_loss_function, 32),
        (policy_loss_no_entropy_when_disabled, policy_loss_no_entropy_when_disabled, 32),
        (prior.value_loss_function, current.value_loss_function, 1),
    ):
        x = logits[..., :width].clone().requires_grad_()
        y = logits[..., :width].clone().requires_grad_()
        loss_old, metrics_old = old_fn(args, first, x, reducer)
        loss_new, metrics_new = new_fn(args, second, y, reducer)
        grad_old = torch.autograd.grad(loss_old, x)[0]
        grad_new = torch.autograd.grad(loss_new, y)[0]
        assert_bits_equal(loss_old, loss_new)
        assert_bits_equal(metrics_old, metrics_new)
        assert_bits_equal(grad_old, grad_new)
        p = torch.nn.Parameter(x.detach().clone())
        q = torch.nn.Parameter(y.detach().clone())
        opt_p, opt_q = torch.optim.AdamW([p], lr=5e-7), torch.optim.AdamW([q], lr=5e-7)
        p.grad, q.grad = grad_old, grad_new
        opt_p.step()
        opt_q.step()
        assert_bits_equal(p, q)
        assert_bits_equal(opt_p.state[p], opt_q.state[q])


def test_native_gae_and_sample_type_source_unchanged():
    from tb_gae_test_utils import ROOT

    for relative in ("slime/utils/ppo_utils.py", "slime/utils/types.py", "slime/backends/megatron_utils/cp_utils.py"):
        assert (ROOT / relative).read_text() == old_source(relative)


def test_mask_logging_does_not_reduce_non_scalars(cpu_mpu, monkeypatch):
    from slime.backends.megatron_utils import data as module

    captured = []
    monkeypatch.setattr(module.mpu, "get_data_parallel_world_size", lambda **kw: 1)
    monkeypatch.setattr(module, "gather_log_data", lambda *a: captured.append(a[-1]))
    data = dict(
        total_lengths=[3],
        response_lengths=[1],
        loss_masks=[torch.ones(1)],
        global_batch_sizes=[1],
        turn_end_mask=[np.ones(1, dtype=bool)],
        is_real_done=[np.ones(1, dtype=bool)],
    )
    module.log_rollout_data(0, tb_args(ci_test=False), data)
    assert "turn_end_mask" not in captured[0] and "is_real_done" not in captured[0]
