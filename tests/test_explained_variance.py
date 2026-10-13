"""CPU unit test for the critic explained-variance diagnostic.

``compute_value_explained_variance`` (slime/backends/megatron_utils/loss.py) is
a read-only metric: EV = 1 - Var(returns - values) / Var(returns), computed over
response tokens with the value-loss mask and reduced across the DP-with-CP group.

These tests pin the math on hand-checkable cases at cp_size == 1 (so the
``all_reduce`` is a single-rank no-op and the global stats equal the local
stats). They guard against sign/factor regressions and confirm the documented
"return None, log nothing" behaviour on degenerate inputs.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch


def _install_mpu_stub(cp_size: int = 1) -> None:
    """Minimal ``megatron.core.mpu`` stub so loss.py imports on CPU.

    Only the handful of accessors ``compute_value_explained_variance`` and the
    module-level imports of loss.py actually touch are provided.
    """
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    mpu = types.ModuleType("megatron.core.mpu")

    mpu.get_context_parallel_world_size = lambda: cp_size
    mpu.get_data_parallel_group = lambda with_context_parallel=False: None
    # The fused-CE import in ppo_utils is only hit on the real forward path;
    # the EV helper never calls it, but loss.py imports ppo_utils at module
    # load, which in turn lazily imports megatron fusions inside functions only.
    core.mpu = mpu
    megatron.core = core
    sys.modules.setdefault("megatron", megatron)
    sys.modules["megatron.core"] = core
    sys.modules["megatron.core.mpu"] = mpu


@pytest.fixture
def ev_fn(monkeypatch):
    _install_mpu_stub(cp_size=1)
    from slime.backends.megatron_utils import loss as loss_mod

    # all_reduce is a single-rank no-op for these tests (sum in place == itself).
    monkeypatch.setattr(loss_mod.dist, "all_reduce", lambda t, group=None: t)
    return loss_mod.compute_value_explained_variance


class _Args:
    qkv_format = "thd"


def _call(ev_fn, values, returns, masks=None):
    """Helper: one sample, cp_size == 1 so masks align 1:1 with tokens."""
    n = values.numel()
    if masks is None:
        masks = torch.ones(n)
    return ev_fn(
        _Args(),
        values=[values],
        returns=[returns],
        loss_masks=[masks],
        total_lengths=[n],
        response_lengths=[n],
        max_seq_lens=None,
    )


def test_perfect_fit_is_one(ev_fn):
    r = torch.tensor([0.1, 0.4, -0.2, 0.9, 0.3])
    ev = _call(ev_fn, values=r.clone(), returns=r)
    assert ev == pytest.approx(1.0, abs=1e-6)


def test_predicting_mean_is_zero(ev_fn):
    r = torch.tensor([0.1, 0.4, -0.2, 0.9, 0.3])
    v = torch.full_like(r, r.mean())  # critic = constant mean -> EV == 0
    ev = _call(ev_fn, values=v, returns=r)
    assert ev == pytest.approx(0.0, abs=1e-6)


def test_worse_than_mean_is_negative(ev_fn):
    # Systematic overestimate (the live-run symptom): values >> returns.
    r = torch.tensor([0.39, 0.41, 0.40, 0.42, 0.40])
    v = torch.tensor([0.86, 0.71, 0.88, 0.80, 0.84])
    ev = _call(ev_fn, values=v, returns=r)
    assert ev < 0.0  # critic worse than predicting the mean


def test_matches_numpy_reference(ev_fn):
    torch.manual_seed(0)
    r = torch.randn(64)
    v = r + 0.3 * torch.randn(64)
    ev = _call(ev_fn, values=v, returns=r)
    # Reference uses the same population (biased) variance the helper does.
    resid = r - v
    ref = 1.0 - (resid.var(unbiased=False) / r.var(unbiased=False)).item()
    assert ev == pytest.approx(ref, abs=1e-5)


def test_mask_excludes_tokens(ev_fn):
    # Masked-out tokens must not affect the result: corrupt them wildly and
    # confirm EV equals the clean-on-unmasked computation.
    r = torch.tensor([0.1, 0.4, -0.2, 0.9, 0.3, 0.0])
    v = r.clone()
    v[4:] = 999.0  # garbage in masked positions
    mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    ev = _call(ev_fn, values=v, returns=r, masks=mask)
    assert ev == pytest.approx(1.0, abs=1e-6)  # perfect fit on the unmasked part


def test_zero_variance_returns_none(ev_fn):
    r = torch.full((8,), 0.5)  # constant returns -> undefined EV
    v = torch.randn(8)
    assert _call(ev_fn, values=v, returns=r) is None


def test_empty_returns_none(ev_fn):
    assert (
        ev_fn(_Args(), values=[], returns=[], loss_masks=[], total_lengths=[], response_lengths=[], max_seq_lens=None)
        is None
    )


def test_single_token_returns_none(ev_fn):
    # w < 2 guard: not enough samples for a variance.
    r = torch.tensor([0.5])
    v = torch.tensor([0.2])
    assert _call(ev_fn, values=v, returns=r) is None
