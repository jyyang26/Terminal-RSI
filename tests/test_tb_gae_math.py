import copy

import numpy as np
import pytest
import torch
from tb_gae_test_utils import cpu_mpu, detail, tb_args  # noqa: F401

from slime.utils.types import Sample
from slime_plugins.harbor.atif_adapter import samples_from_rollout_details, samples_from_trial_result
from slime_plugins.harbor.tb_gae import compute_tb_gae, tb_gae_advantage
from slime_plugins.harbor.tb_gae_data import build_tb_masks, slice_tb_masks, validate_and_collect_tb_masks

NUM_GPUS = 0
pytestmark = pytest.mark.unit


def reference(values, rewards, action, turns, done, gamma, lam):
    advantages, returns = torch.zeros_like(values), torch.zeros_like(values)
    for b in range(len(values)):
        idx = torch.nonzero(action[b], as_tuple=True)[0]
        last = torch.zeros((), dtype=values.dtype)
        for j in reversed(range(len(idx))):
            pos = idx[j]
            next_v = values[b, idx[j + 1]] if j + 1 < len(idx) else 0.0
            g, l = (gamma, lam) if turns[b, pos] and j + 1 < len(idx) else (1.0, 1.0)
            nonterm = 1.0 - done[b, pos].to(values.dtype)
            delta = rewards[b, pos] + g * nonterm * next_v - values[b, pos]
            last = delta + g * l * nonterm * last
            advantages[b, pos], returns[b, pos] = last, last + values[b, pos]
    return advantages, returns


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("length", [1, 7, 129, 513])
def test_s3_matches_native_gae(cpu_mpu, interleaved, length):
    from slime.utils.ppo_utils import get_advantages_and_returns_batch

    generator = torch.Generator().manual_seed(length)
    values = torch.randn(2, length, generator=generator)
    rewards = torch.zeros_like(values)
    rewards[:, -1] = torch.tensor([0.7, 1.2])
    action = torch.ones_like(values, dtype=torch.bool)
    if interleaved:
        action[:, 1:-1:3] = False
    turns = action.clone()
    done = torch.zeros_like(action)
    done[:, -1] = True
    a, r = compute_tb_gae(values, rewards, action, turns, done)
    old_a, old_r = get_advantages_and_returns_batch(
        [length + 3] * 2,
        [length] * 2,
        list(values),
        list(rewards),
        1.0,
        1.0,
    )
    assert (a[action] - torch.stack(old_a)[action]).abs().max() < 1e-4
    assert (r[action] - torch.stack(old_r)[action]).abs().max() < 1e-4
    assert not a[~action].any() and not r[~action].any()


@pytest.mark.parametrize("gamma,lam", [(1.0, 1.0), (0.999, 0.999), (0.8, 0.7)])
def test_scan_matches_sequential_reference(gamma, lam):
    g = torch.Generator().manual_seed(6)
    v = torch.randn(3, 277, generator=g, dtype=torch.float64)
    r = torch.zeros_like(v)
    r[:, -1] = 0.7
    action = torch.ones_like(v, dtype=torch.bool)
    action[:, 2:-1:4] = False
    turns = torch.zeros_like(action)
    turns[:, 0::4] = True
    turns[:, -1] = True
    done = torch.zeros_like(action)
    done[:, -1] = True
    actual = compute_tb_gae(v, r, action, turns, done, gamma, lam)
    expected = reference(v, r, action, turns, done, gamma, lam)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_s4_discounts_only_at_turn_transitions():
    v = torch.zeros(1, 150, dtype=torch.float64)
    action = torch.ones_like(v, dtype=torch.bool)
    turns = torch.zeros_like(action)
    turns[:, 2::3] = True
    done = torch.zeros_like(action)
    done[:, -1] = True
    reward = v.clone()
    reward[:, -1] = 1.0
    a, r = compute_tb_gae(v, reward, action, turns, done, 0.999, 0.999)
    expected = torch.tensor([(0.999 * 0.999) ** (49 - j // 3) for j in range(150)], dtype=torch.float64)[None]
    torch.testing.assert_close(a, expected, atol=1e-13, rtol=1e-13)
    assert torch.equal(a, r)


def test_no_grad_and_explicit_nonterminal_bootstrap():
    v = torch.tensor([[0.2, 0.4]], requires_grad=True)
    r = torch.zeros_like(v, requires_grad=True)
    action = torch.ones_like(v, dtype=torch.bool)
    done = torch.zeros_like(action)
    with pytest.raises(ValueError, match="bootstrap"):
        compute_tb_gae(v, r, action, action, done)
    a, ret = compute_tb_gae(
        v, r, action, action, done, bootstrap_values=torch.tensor([0.7]), continuation_advantages=torch.tensor([0.1])
    )
    torch.testing.assert_close(ret, torch.full_like(v, 0.8))
    assert not a.requires_grad and a.grad_fn is None and not ret.requires_grad and ret.grad_fn is None
    assert v.grad is None and r.grad is None


@pytest.mark.parametrize("kind", ["normal", "adjacent", "retokenized"])
def test_adapter_masks_and_response_cap(kind):
    from slime_plugins.harbor.sample_postprocess import cap_total_tokens_allow_response_trim_in_samples

    samples = samples_from_rollout_details(
        [detail(kind)],
        Sample(index=4),
        reward=0.7,
        status=Sample.Status.COMPLETED,
        mode="trajectory_retokenized",
        metadata={"harbor_termination_kind": "terminated"},
        enable_tb_gae=True,
    )
    sample = samples[0]
    te, done = sample.metadata["turn_end_mask"], sample.metadata["is_real_done"]
    assert te.dtype == np.bool_ and done.dtype == np.bool_ and te.nbytes == sample.response_length
    assert te.sum() == 2 and done.sum() == 1 and done[-1]
    before = copy.deepcopy(sample)
    cap_total_tokens_allow_response_trim_in_samples(tb_args(harbor_max_train_total_tokens=4), samples)
    assert all(len(s.tokens) <= 4 for s in samples)
    expected = [t for t, m in zip(before.tokens[-before.response_length:], before.loss_mask, strict=True) if m]
    actual = [t for s in samples for t, m in zip(s.tokens[-s.response_length:], s.loss_mask, strict=True) if m]
    assert actual == expected
    assert samples[-1].metadata["is_real_done"][-1]
    assert sum(s.metadata["turn_end_mask"].sum() for s in samples) == te.sum()
    out = validate_and_collect_tb_masks(samples, [s.loss_mask for s in samples])
    assert len(out["tb_boundary"]) == len(samples)


@pytest.mark.parametrize("termination", [None, "truncated", "continuing", "unknown"])
def test_trial_requires_explicit_environment_termination(termination):
    result = {"agent_result": {"rollout_details": [detail()], "metadata": {"termination_kind": termination}}}
    with pytest.raises(ValueError, match="termination_kind|bootstrap state"):
        samples_from_trial_result(result, Sample(), mode="trajectory_retokenized", enable_tb_gae=True)


def test_trial_entry_and_split_continuation_metadata():
    result = {
        "agent_result": {"rollout_details": [detail()], "metadata": {"termination_kind": "terminated"}},
        "verifier_result": {"rewards": {"reward": 0.7}},
    }
    samples = samples_from_trial_result(result, Sample(), mode="trajectory_retokenized", enable_tb_gae=True)
    assert samples[0].metadata["is_real_done"][-1]
    chunks = samples_from_rollout_details(
            [detail("split")],
            Sample(),
            reward=0.7,
            status=Sample.Status.COMPLETED,
            mode="trajectory_retokenized",
            enable_tb_gae=True,
            metadata={"harbor_termination_kind": "terminated"},
    )
    assert len(chunks) == 2
    assert not chunks[0].metadata["is_real_done"].any()
    assert chunks[1].metadata["is_real_done"][-1]
    assert [c.metadata["tb_boundary"]["chunk_index"] for c in chunks] == [0, 1]


def test_both_roles_dispatch_and_outcome_reward(cpu_mpu):
    from slime.backends.megatron_utils.loss import compute_advantages_and_returns
    from slime.utils.arguments import _apply_megatron_role_overrides

    base = tb_args()
    outputs = []
    for role in ("actor", "critic"):
        args = _apply_megatron_role_overrides(base, {}, role)
        data = dict(
            values=[torch.tensor([0.2, 0.4], requires_grad=True)],
            rewards=[0.7],
            loss_masks=[torch.tensor([1, 1])],
            total_lengths=[4],
            response_lengths=[2],
            turn_end_mask=[np.array([1, 1], dtype=bool)],
            is_real_done=[np.array([0, 1], dtype=bool)],
        )
        compute_advantages_and_returns(args, data)
        torch.testing.assert_close(data["returns"][0], torch.tensor([0.7, 0.7]))
        assert data["returns"][0].grad_fn is None and data["advantages"][0].grad_fn is None
        outputs.append(data)
    assert torch.equal(outputs[0]["returns"][0], outputs[1]["returns"][0])
    assert torch.equal(outputs[0]["advantages"][0], outputs[1]["advantages"][0])


@pytest.mark.parametrize("missing", ["turn_end_mask", "is_real_done"])
def test_missing_fields_fail_before_math(cpu_mpu, missing):
    data = dict(values=[torch.zeros(1)], turn_end_mask=[np.ones(1, dtype=bool)], is_real_done=[np.ones(1, dtype=bool)])
    data.pop(missing)
    with pytest.raises(RuntimeError, match=missing):
        tb_gae_advantage(tb_args(), data)


def test_removed_sample_masks_zero_and_bad_masks_fail():
    te, dn = build_tb_masks(3, [(0, 1), (2, 3)], [1, 0, 1], termination_kind="terminated", is_last_chunk=True)
    sample = Sample(index=1, response_length=3, remove_sample=True, metadata={"turn_end_mask": te, "is_real_done": dn})
    result = validate_and_collect_tb_masks([sample], [[0, 0, 0]])
    assert not result["turn_end_mask"][0].any() and not result["is_real_done"][0].any()
    with pytest.raises(ValueError):
        build_tb_masks(3, [(0, 3)], [1, 0, 1], termination_kind="terminated", is_last_chunk=True)
    with pytest.raises(ValueError):
        slice_tb_masks(sample.metadata, start=0, stop=4)


def test_late_cap_is_validation_only():
    from slime_plugins.harbor.sample_postprocess import cap_total_tokens_allow_response_trim_in_rollout_data

    data = {"total_lengths": [10]}
    with pytest.raises(ValueError, match="before DP"):
        cap_total_tokens_allow_response_trim_in_rollout_data(tb_args(harbor_max_train_total_tokens=4), 0, data)
    assert data == {"total_lengths": [10]}


def test_direct_hook_cannot_enable_itself():
    with pytest.raises(ValueError, match="explicit"):
        tb_gae_advantage(tb_args(enable_tb_gae=False), {})
