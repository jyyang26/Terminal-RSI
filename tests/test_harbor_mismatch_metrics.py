"""Unit tests for the record-only mismatch-metrics TIS function.

``slime_plugins.harbor.mismatch_metrics.record_only_tis_function`` is wired in via
``--custom-tis-function-path`` when ``--get-mismatch-metrics`` is set. Its whole point
is to log the train/inference log-prob mismatch (``tis`` / ``tis_abs``) WITHOUT
perturbing ``pg_loss`` — unlike the built-in ``vanilla_tis_function`` which multiplies
``pg_loss`` by clamped importance-sampling weights.

These tests pin the two invariants that matter:
    1. ``pg_loss`` (and ``loss_masks``) are returned bit-identical to the input — proof
       the policy loss is unaffected.
    2. ``tis`` / ``tis_abs`` equal the framework's definitions:
           tis     = exp(train_logp - rollout_logp)
           tis_abs = |exp(train_logp - rollout_logp) - 1|
"""

from __future__ import annotations

import torch

from slime_plugins.harbor.mismatch_metrics import record_only_tis_function


class _Args:
    """Minimal stand-in for the argparse Namespace the fn ignores."""


def test_pg_loss_returned_unchanged():
    pg_loss = torch.tensor([0.3, -1.2, 4.5, 0.0])
    loss_masks = [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0])]
    train = [torch.tensor([-0.5, -0.7]), torch.tensor([-1.0, -0.2])]
    rollout = [torch.tensor([-0.4, -0.9]), torch.tensor([-1.3, -0.2])]

    out_pg, out_masks, metrics = record_only_tis_function(
        _Args(),
        pg_loss=pg_loss,
        train_log_probs=train,
        rollout_log_probs=rollout,
        loss_masks=loss_masks,
    )
    # pg_loss must be the SAME object/values — no IS correction applied.
    assert out_pg is pg_loss
    assert torch.equal(out_pg, pg_loss)
    # loss_masks returned untouched (so the caller's reducer/denominator is unchanged).
    assert out_masks is loss_masks
    # metrics are the diagnostic-only payload; presence checked in detail below.
    assert set(metrics) == {"tis", "tis_abs", "train_rollout_logprob_abs_diff_aligned"}


def test_tis_and_tis_abs_values():
    # train - rollout = [0.1, -0.2, 0.3, -0.1]; tis = exp(diff); tis_abs = |exp(diff)-1|.
    train = [torch.tensor([-0.5, -0.7]), torch.tensor([-1.0, -0.2])]
    rollout = [torch.tensor([-0.6, -0.5]), torch.tensor([-1.3, -0.1])]
    diff = torch.tensor([0.1, -0.2, 0.3, -0.1])

    _, _, metrics = record_only_tis_function(
        _Args(),
        pg_loss=torch.zeros(4),
        train_log_probs=train,
        rollout_log_probs=rollout,
        loss_masks=[torch.ones(2), torch.ones(2)],
    )
    assert set(metrics) == {"tis", "tis_abs", "train_rollout_logprob_abs_diff_aligned"}
    assert torch.allclose(metrics["tis"], torch.exp(diff), atol=1e-6)
    assert torch.allclose(metrics["tis_abs"], (torch.exp(diff) - 1).abs(), atol=1e-6)
    # aligned logprob abs-diff = |train - rollout| exactly (the trustworthy, correctly
    # normalized replacement for the inflated built-in train_rollout_logprob_abs_diff).
    assert torch.allclose(metrics["train_rollout_logprob_abs_diff_aligned"], diff.abs(), atol=1e-6)
    # for small drift, aligned |d| ≈ tis_abs (|exp(d)-1| ≈ |d|)
    assert torch.allclose(metrics["train_rollout_logprob_abs_diff_aligned"], metrics["tis_abs"], atol=0.05)


def test_identical_logprobs_give_unit_tis_zero_abs():
    # When train == rollout (no drift), tis == 1 everywhere and tis_abs == 0.
    lp = [torch.tensor([-0.3, -0.9, -1.1])]
    _, _, metrics = record_only_tis_function(
        _Args(),
        pg_loss=torch.zeros(3),
        train_log_probs=lp,
        rollout_log_probs=[t.clone() for t in lp],
        loss_masks=[torch.ones(3)],
    )
    assert torch.allclose(metrics["tis"], torch.ones(3), atol=1e-7)
    assert torch.allclose(metrics["tis_abs"], torch.zeros(3), atol=1e-7)


def test_metrics_are_detached():
    # Returned metric tensors must not carry grad (they are logged, not backprop'd).
    train = [torch.tensor([-0.5, -0.7], requires_grad=True)]
    rollout = [torch.tensor([-0.6, -0.5])]
    _, _, metrics = record_only_tis_function(
        _Args(),
        pg_loss=torch.zeros(2),
        train_log_probs=train,
        rollout_log_probs=rollout,
        loss_masks=[torch.ones(2)],
    )
    assert not metrics["tis"].requires_grad
    assert not metrics["tis_abs"].requires_grad
