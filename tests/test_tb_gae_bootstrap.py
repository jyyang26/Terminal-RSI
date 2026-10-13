"""CPU tests for successor values, full-episode GAE, and action-preserving chunks."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tb_gae_test_utils import cpu_mpu, detail, tb_args  # noqa: F401

from slime.backends.megatron_utils.tb_gae_bootstrap import make_bootstrap_batch
from slime.utils.types import Sample
from slime_plugins.harbor.atif_adapter import samples_from_rollout_details
from slime_plugins.harbor.tb_gae import compute_tb_gae, tb_gae_advantage
from slime_plugins.harbor.tb_gae_boundary import solve_episode_boundaries
from slime_plugins.harbor.tb_gae_chunks import split_tb_samples
from slime_plugins.harbor.tb_gae_data import validate_and_collect_tb_masks

NUM_GPUS = 0
pytestmark = pytest.mark.unit
# Imported pytest fixtures are intentionally shadowed by test parameters.
# ruff: noqa: F811


def episode_samples(*, terminal, split=False, episode=1):
    kw = dict(harbor_termination_kind="terminated" if terminal else "truncated")
    if not terminal:
        kw.update(harbor_bootstrap_token_ids=[71, 72, 73], harbor_bootstrap_action_count=3)
    return samples_from_rollout_details(
        [detail("split" if split else "normal")],
        Sample(index=episode, group_id=episode),
        reward=0.7,
        status=Sample.Status.COMPLETED if terminal else Sample.Status.TRUNCATED,
        mode="trajectory_retokenized",
        enable_tb_gae=True,
        metadata=kw,
    )


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("gamma", [1.0, 0.999])
def test_full_episode_matches_unsplit_reference(cpu_mpu, monkeypatch, terminal, split, gamma):
    monkeypatch.setattr(cpu_mpu, "get_data_parallel_group_gloo", lambda **kw: dist.group.WORLD)
    samples = episode_samples(terminal=terminal, split=split)
    fields = validate_and_collect_tb_masks(samples, [s.loss_mask for s in samples])
    values = [torch.linspace(0.1 + i / 10, 0.4 + i / 10, s.response_length) for i, s in enumerate(samples)]
    data = dict(
        fields,
        values=values,
        rewards=[0.7] * len(samples),
        loss_masks=[torch.tensor(s.loss_mask) for s in samples],
        response_lengths=[s.response_length for s in samples],
        total_lengths=[len(s.tokens) for s in samples],
        tb_bootstrap_values=[0.8] * len(samples),
    )
    full_v = torch.cat(values)[None]
    full_a = torch.cat([torch.tensor(s.loss_mask).bool() for s in samples])[None]
    full_te = torch.cat([torch.tensor(s.metadata["turn_end_mask"]) for s in samples])[None]
    full_dn = torch.cat([torch.tensor(s.metadata["is_real_done"]) for s in samples])[None]
    reward = torch.zeros_like(full_v)
    reward[0, -1] = 0.7
    expected = compute_tb_gae(
        full_v,
        reward,
        full_a,
        full_te,
        full_dn,
        gamma,
        gamma,
        bootstrap_values=torch.tensor([0.8]),
        continuation_advantages=torch.zeros(1),
    )
    tb_gae_advantage(tb_args(tb_gae_gamma_turn=gamma, tb_gae_lambda_turn=gamma), data)
    torch.testing.assert_close(torch.cat(data["advantages"]), expected[0][0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.cat(data["returns"]), expected[1][0], atol=1e-6, rtol=1e-6)
    if not terminal and gamma == 1.0:
        torch.testing.assert_close(torch.cat(data["returns"])[full_a[0]], torch.full((3,), 1.5))


@pytest.mark.parametrize("cap", [3, 4, 5, 6, 8])
@pytest.mark.parametrize("terminal", [False, True])
def test_hard_cap_keeps_all_action_tokens_and_endpoints(cap, terminal):
    samples = episode_samples(terminal=terminal)
    expected = [
        t for s in samples for t, active in zip(s.tokens[-s.response_length :], s.loss_mask, strict=True) if active
    ]
    original_turns = sum(s.metadata["turn_end_mask"].sum() for s in samples)
    split_tb_samples(samples, cap)
    actual = [
        t for s in samples for t, active in zip(s.tokens[-s.response_length :], s.loss_mask, strict=True) if active
    ]
    assert actual == expected
    assert all(len(s.tokens) <= cap for s in samples)
    assert sum(s.metadata["turn_end_mask"].sum() for s in samples) == original_turns
    assert sum(s.metadata["is_real_done"].sum() for s in samples) == int(terminal)
    if not terminal:
        assert samples[-1].metadata["tb_boundary"]["bootstrap_token_ids"] == [71, 72, 73]
    assert [s.metadata["tb_boundary"]["chunk_index"] for s in samples] == list(range(len(samples)))
    validate_and_collect_tb_masks(samples, [s.loss_mask for s in samples])


def test_bootstrap_probe_selects_successor_not_last_action(cpu_mpu):
    from slime.backends.megatron_utils.loss import get_values

    data = {
        "tb_boundary": [dict(terminal=False, chunk_index=0, chunk_count=1, bootstrap_token_ids=[3, 7, 11])],
        "micro_batch_indices": [[0]],
        "num_microbatches": [1],
        "global_batch_sizes": [1],
    }
    batch, required = make_bootstrap_batch(data, device="cpu", max_tokens=64, padded_count=2)
    assert batch["tokens"][0].tolist() == [3, 7, 11, 0]
    assert required == [True] and batch["tokens"][1].tolist() == [0, 0]
    # Causal toy critic: state value = sum(prefix). Dummy token must not shift target.
    logits = batch["tokens"][0].float().cumsum(0)[None, :, None]
    _, result = get_values(
        logits, args=tb_args(), unconcat_tokens=[batch["tokens"][0]], total_lengths=[4], response_lengths=[1]
    )
    assert result["values"][0].item() == 21.0


def test_bootstrap_real_forward_callback_and_schedule(cpu_mpu, monkeypatch):
    import megatron.core.utils

    from slime.backends.megatron_utils import model as model_utils
    from slime.backends.megatron_utils.tb_gae_bootstrap import compute_bootstrap_values
    from slime.utils import distributed_utils

    monkeypatch.setattr(distributed_utils, "get_gloo_group", lambda: dist.group.WORLD)
    # Production get_batch uses .cuda() for packed-sequence lengths. Keep its
    # actual packing/slicing math but leave that metadata on CPU in this test.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(cpu_mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)
    config = SimpleNamespace(timers=None, microbatch_group_size_per_vp_stage=1)
    monkeypatch.setattr(megatron.core.utils, "get_model_config", lambda model: config)
    monkeypatch.setattr(model_utils, "get_model_config", lambda model: config)
    monkeypatch.setattr(model_utils, "_disable_tqdm_for_non_main_rank", lambda: True)
    monkeypatch.setattr(model_utils, "_wrap_forward_step_with_microbatch_pbar", lambda fn, bar: fn)

    class Critic:
        def eval(self):
            pass

        def train(self):
            pass

        def __call__(self, **kw):
            return kw["input_ids"].float().cumsum(1)[..., None]

    def scheduler(**kw):
        output = []
        assert kw["micro_batch_size"] == 1
        for _ in range(kw["num_microbatches"]):
            logits, callback = kw["forward_step_func"](kw["data_iterator"][0], kw["model"][0])
            _, result = callback(logits)
            output.append(result)
        return output

    monkeypatch.setattr(model_utils, "get_forward_backward_func", lambda: scheduler)
    data = {
        "tokens": [torch.tensor([1, 2]), torch.tensor([1, 2])],
        "tb_boundary": [
            dict(terminal=False, chunk_index=0, chunk_count=1, bootstrap_token_ids=[3, 7, 11]),
            dict(terminal=True, chunk_index=0, chunk_count=1, bootstrap_token_ids=None),
        ],
    }
    args = tb_args(
        seq_length=64,
        data_pad_size_multiplier=1,
        use_rollout_entropy=False,
        custom_megatron_before_log_prob_hook_path=None,
        micro_batch_size=1,
    )
    assert compute_bootstrap_values(args, [Critic()], data) == [21.0, 0.0]


def test_missing_or_conflicting_boundaries_fail():
    row = dict(
        episode="a",
        chunk_index=0,
        chunk_count=2,
        terminal=False,
        active=True,
        outcome=0.7,
        base=0.0,
        first_value=0.2,
        value_coefficient=1.0,
        advantage_coefficient=1.0,
        bootstrap=None,
    )
    with pytest.raises(ValueError, match="incomplete"):
        solve_episode_boundaries([row])
    second = dict(row, chunk_index=1, bootstrap=0.8)
    assert solve_episode_boundaries([second, row])[("a", 1)] == (0.8, 0.0)
    with pytest.raises(ValueError, match="outcome"):
        solve_episode_boundaries([row, dict(second, outcome=0.3)])
    with pytest.raises(ValueError, match="finite old-critic"):
        solve_episode_boundaries([row, dict(second, bootstrap=None)])


def test_incomplete_episode_rejected_before_dp():
    samples = episode_samples(terminal=False, split=True)
    with pytest.raises(ValueError, match="incomplete"):
        validate_and_collect_tb_masks(samples[1:], [samples[1].loss_mask])


def test_real_critic_lifecycle_bootstraps_before_update(cpu_mpu, monkeypatch):
    from slime.backends.megatron_utils import actor as module
    from slime.backends.megatron_utils import tb_gae_bootstrap

    events = []
    current_weight = [1.0]
    model = object.__new__(module.MegatronTrainRayActor)
    model.args = tb_args()
    model.model = [object()]
    model.optimizer = model.opt_param_scheduler = None
    data = {"num_microbatches": [1], "global_batch_sizes": [1], "micro_batch_indices": [[0]]}
    monkeypatch.setattr(module, "get_data_iterator", lambda data: [])

    def forward(*unused):
        events.append("old_values")
        return {"values": [torch.tensor([current_weight[0]])]}

    def bootstrap(*unused):
        events.append("bootstrap")
        return [current_weight[0] * 2]

    def advantages(args, payload):
        events.append("targets")
        assert payload["tb_bootstrap_values"] == [2.0]
        assert payload["values"][0].item() == 1.0

    def update(*unused):
        events.append("optimizer")
        current_weight[0] = 99.0

    monkeypatch.setattr(module, "forward_only", forward)
    monkeypatch.setattr(tb_gae_bootstrap, "compute_bootstrap_values", bootstrap)
    monkeypatch.setattr(module, "compute_advantages_and_returns", advantages)
    monkeypatch.setattr(module, "train", update)
    result = model.train_critic(0, data)
    assert events == ["old_values", "bootstrap", "targets", "optimizer"]
    assert result["tb_bootstrap_values"] == [2.0] and result["values"][0].item() == 1.0


def test_actor_receives_old_bootstrap_without_new_critic_forward(cpu_mpu, monkeypatch):
    from contextlib import nullcontext

    from slime.backends.megatron_utils import actor as module

    instance = object.__new__(module.MegatronTrainRayActor)
    instance.args = tb_args(
        use_rollout_routing_replay=False,
        use_routing_replay=False,
        use_rollout_logprobs=True,
        offload_train=False,
        loss_type="policy_loss",
        ref_update_interval=None,
    )
    instance.model = [object()]
    instance.optimizer = instance.opt_param_scheduler = None
    instance.weights_backuper = SimpleNamespace(backup_tags=[], backup=lambda tag: None)
    instance.prof = SimpleNamespace(step=lambda **kw: None)
    instance.weight_updater = SimpleNamespace(pop_metrics=lambda: {})
    instance.rollout_data_postprocess = None
    instance._active_model_tag = "actor"
    instance._switch_model = lambda tag: None
    monkeypatch.setattr(module, "timer", lambda *args: nullcontext())
    monkeypatch.setattr(module, "inverse_timer", lambda *args: nullcontext())
    monkeypatch.setattr(module, "get_data_iterator", lambda data: [])
    monkeypatch.setattr(module, "log_rollout_data", lambda *args: None)
    monkeypatch.setattr(module, "train", lambda *args: None)
    monkeypatch.setattr(module.train_dump_utils, "save_debug_train_data", lambda *args, **kw: None)
    monkeypatch.setattr(module, "log_perf_data", lambda *args, **kw: None)
    from slime.backends.megatron_utils import data as data_module

    monkeypatch.setattr(data_module, "tensors_to_gpu", lambda values: values)
    seen = []
    monkeypatch.setattr(
        module, "compute_advantages_and_returns", lambda args, data: seen.append(data["tb_bootstrap_values"])
    )
    source = {"num_microbatches": [1], "global_batch_sizes": [1], "micro_batch_indices": [[0]]}
    instance.train_actor(0, source, external_data={"values": [torch.tensor([0.2])], "tb_bootstrap_values": [0.8]})
    assert seen == [[0.8]]
