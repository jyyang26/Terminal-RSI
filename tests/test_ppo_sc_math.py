"""Offline mathematical contracts for score and PPO-weighted centering."""

import math

import pytest
import torch

from slime.utils.score_centering import calculate_head_log_probs, ppo_effective_weight, score_centering_correction

NUM_GPUS = 0
pytestmark = pytest.mark.unit


def reference_coefficients(lp, lq, advantage, formula, old=None, eps=1e-8):
    # Independent scalar-loop oracle: the coefficients are frozen before autograd.
    output = []
    for p_row, q_row, adv, index in zip(
        lp.detach().exp(), lq.detach().exp(), advantage.detach(), range(len(lp)), strict=True
    ):
        p, q = p_row.tolist(), q_row.tolist()
        denominator = q if old is None else old[index].detach().exp().tolist()
        p_tail, q_tail = max(1 - sum(p), eps), max(1 - sum(q), 0)

        def weight(ratio):
            if (adv > 0 and ratio > 1.2) or (adv < 0 and ratio < 0.8):
                return 0.0
            return ratio

        if formula == "score":
            output.append([qv - q_tail / p_tail * pv for pv, qv in zip(p, q, strict=True)])
        else:
            tail_weight = weight(p_tail / max(1 - sum(denominator), eps))
            output.append(
                [
                    qv * weight(pv / ov) - q_tail / p_tail * tail_weight * pv
                    for pv, qv, ov in zip(p, q, denominator, strict=True)
                ]
            )
    return lp.new_tensor(output)


@pytest.mark.parametrize("formula", ["score", "ppo_weighted"])
@pytest.mark.parametrize("old_policy", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradient_equals_independent_frozen_coefficient(formula, old_policy, dtype):
    p = torch.tensor([[0.2, 0.3], [0.15, 0.4], [0.3, 0.2]], dtype=dtype).log().requires_grad_()
    q = torch.tensor([[0.1, 0.25], [0.3, 0.2], [0.1, 0.4]], dtype=dtype).log().requires_grad_()
    old = (
        torch.tensor([[0.25, 0.1], [0.2, 0.2], [0.4, 0.3]], dtype=dtype).log().requires_grad_() if old_policy else None
    )
    adv = torch.tensor([2.0, -3.0, 0.0], dtype=dtype, requires_grad=True)
    correction, metrics = score_centering_correction(
        train_head_logp=p, sampler_head_logp=q, advantages=adv, formula=formula, old_head_logp=old
    )
    inputs = (p, q, adv) if old is None else (p, q, adv, old)
    grads = torch.autograd.grad((correction * adv.detach()).sum(), inputs, allow_unused=True)
    expected = reference_coefficients(p, q, adv, formula, old) * adv.detach()[:, None]
    torch.testing.assert_close(grads[0], expected, atol=2e-7 if dtype == torch.float32 else 1e-12, rtol=1e-5)
    assert all(grad is None for grad in grads[1:])
    assert correction.dtype == dtype
    assert all(not value.requires_grad for value in metrics.values())


@pytest.mark.parametrize("sign", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("ratio", [0.3, 0.79, 0.8, 0.81, 1.0, 1.19, 1.2, 1.21, 3.0])
def test_ppo_weight_matches_real_clipped_autograd_including_boundaries(sign, ratio):
    r = torch.tensor(ratio, dtype=torch.float64, requires_grad=True)
    loss = torch.maximum(-sign * r, -sign * r.clamp(0.8, 1.2))
    derivative = torch.autograd.grad(loss, r)[0] * r.detach()
    w = ppo_effective_weight(r.detach(), torch.tensor(sign), 0.2, 0.2)
    torch.testing.assert_close(derivative, -sign * w)


@pytest.mark.parametrize("formula", ["score", "ppo_weighted"])
def test_full_vocab_gradient_matches_unfolded_distribution(formula):
    g = torch.Generator().manual_seed(22)
    logits = torch.randn(3, 13, dtype=torch.float64, generator=g, requires_grad=True)
    q = torch.randn(3, 13, dtype=torch.float64, generator=g).softmax(-1)
    old = torch.randn(3, 13, dtype=torch.float64, generator=g).softmax(-1)
    adv = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    lp = logits.log_softmax(-1)
    correction, _ = score_centering_correction(
        train_head_logp=lp, sampler_head_logp=q.log(), old_head_logp=old.log(), advantages=adv, formula=formula
    )
    actual = torch.autograd.grad((adv * correction).sum(), logits, retain_graph=True)[0]
    coef = q if formula == "score" else q * ppo_effective_weight(lp.detach().exp() / old, adv[:, None], 0.2, 0.2)
    expected = torch.autograd.grad((adv[:, None] * coef.detach() * lp).sum(), logits)[0]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)


@pytest.mark.parametrize("formula", ["score", "ppo_weighted"])
def test_proportional_tail_folding_matches_full_vocab(formula):
    p = torch.tensor([[0.2, 0.15, 0.1, 0.25, 0.3], [0.3, 0.1, 0.2, 0.15, 0.25]], dtype=torch.float64)
    q = torch.tensor([[0.1, 0.2, 0.1, 0.25, 0.3], [0.25, 0.2, 0.2, 0.15, 0.25]], dtype=torch.float64)
    q[:, 2:] *= ((1 - q[:, :2].sum(-1)) / (q[:, 2:].sum(-1)))[:, None]
    old = torch.tensor([[0.3, 0.1, 0.1, 0.25, 0.3], [0.2, 0.3, 0.2, 0.15, 0.25]], dtype=torch.float64)
    old[:, 2:] *= ((1 - old[:, :2].sum(-1)) / old[:, 2:].sum(-1))[:, None]
    logits = p.log().requires_grad_()
    lp = logits.log_softmax(-1)
    adv = torch.tensor([1.0, -2.0])
    correction, _ = score_centering_correction(
        train_head_logp=lp[:, :2],
        sampler_head_logp=q[:, :2].log(),
        old_head_logp=old[:, :2].log(),
        advantages=adv,
        formula=formula,
    )
    actual = torch.autograd.grad(correction.sum(), logits, retain_graph=True)[0]
    coef = q if formula == "score" else q * ppo_effective_weight(p / old, adv[:, None], 0.2, 0.2)
    expected = torch.autograd.grad((coef.detach() * lp).sum(), logits)[0]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)


@pytest.mark.parametrize("formula", ["score", "ppo_weighted"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_masked_nan_inf_and_zero_tail_are_finite(formula, dtype):
    lp = torch.tensor(
        [[float("nan"), float("inf")], [math.log(0.5), math.log(0.5)], [-1.0, -2.0]], dtype=dtype, requires_grad=True
    )
    q = lp.detach().clone()
    mask = torch.tensor([False, True, True])
    adv = torch.tensor([0.0, 1.0, -1.0])
    c, diag = score_centering_correction(
        train_head_logp=lp, sampler_head_logp=q, advantages=adv, formula=formula, loss_mask=mask
    )
    c.sum().backward()
    assert torch.isfinite(c).all() and torch.isfinite(lp.grad).all()
    assert lp.grad[0].eq(0).all()
    assert all(torch.isfinite(v).all() for v in diag.values())


@pytest.mark.parametrize("formula", ["score", "ppo_weighted"])
def test_k1_empty_rows_and_fp64_tail_epsilon(formula):
    lp = torch.empty((0, 1), requires_grad=True)
    c, _ = score_centering_correction(
        train_head_logp=lp, sampler_head_logp=lp.detach(), advantages=torch.empty(0), formula=formula
    )
    c.sum().backward()
    assert lp.grad.shape == lp.shape
    p = torch.tensor([[1.0]], dtype=torch.float64, requires_grad=True).log()
    q = torch.tensor([[0.5]], dtype=torch.float64).log()
    c, _ = score_centering_correction(
        train_head_logp=p, sampler_head_logp=q, advantages=torch.ones(1), formula=formula
    )
    assert torch.isfinite(c).all()


@pytest.mark.parametrize("chunk", [1, 3, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_packed_gather_forward_backward_and_input_immutability(chunk, dtype):
    torch.manual_seed(37)
    x = torch.randn(9, 17, dtype=dtype, requires_grad=True)
    saved = x.detach().clone()
    rows = torch.tensor([0, 3, 6, 8])
    ids = torch.tensor([[0, 1, 16], [3, 5, 7], [2, 4, 8], [1, 9, 15]])
    actual = calculate_head_log_probs(x, rows, ids, chunk_size=chunk)
    expected = x.log_softmax(-1)[rows].gather(-1, ids)
    coef = torch.randn_like(actual)
    ga = torch.autograd.grad((actual * coef).sum(), x, retain_graph=True)[0]
    ge = torch.autograd.grad((expected * coef).sum(), x)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(ga, ge)
    assert torch.equal(x, saved)
    assert ga[torch.tensor([1, 2, 4, 5, 7])].eq(0).all()


def test_empty_gather_has_zero_gradient():
    x = torch.randn(5, 13, requires_grad=True)
    result = calculate_head_log_probs(x, torch.empty(0, dtype=torch.long), torch.empty((0, 2), dtype=torch.long))
    result.sum().backward()
    assert x.grad.eq(0).all()


@pytest.mark.parametrize(
    "rows,ids",
    [
        ([0, 0], [[1], [2]]),
        ([-1], [[1]]),
        ([7], [[1]]),
        ([1], [[-1]]),
        ([1], [[13]]),
    ],
)
def test_gather_rejects_invalid_indices(rows, ids):
    with pytest.raises(ValueError):
        calculate_head_log_probs(torch.randn(7, 13), torch.tensor(rows), torch.tensor(ids))


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_epsilon_rejected(value):
    with pytest.raises(ValueError):
        score_centering_correction(
            train_head_logp=torch.tensor([[-1.0]]),
            sampler_head_logp=torch.tensor([[-1.0]]),
            advantages=torch.ones(1),
            tail_eps=value,
        )


def test_nonfinite_unmasked_head_rejected():
    with pytest.raises(ValueError, match="Nonfinite"):
        score_centering_correction(
            train_head_logp=torch.tensor([[float("nan")]]),
            sampler_head_logp=torch.tensor([[-1.0]]),
            advantages=torch.ones(1),
        )
