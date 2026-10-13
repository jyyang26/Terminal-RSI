"""Contract tests for the length-adaptive GAE hook.

The bug these exist to prevent (cost: a full 11-node run, ~2.5h of 88 GPUs):
the hook wrote its lambda diagnostics as bare Python floats. Every value in
``rollout_data`` is walked by ``log_rollout_data``, which accepts ONLY
list/tuple or ``torch.Tensor`` and raises ``ValueError: Unsupported type``
otherwise. Because the raise happens AFTER advantages/returns are computed
and AFTER both critic-only warmup steps, it detonated at the end of the first
JOINT actor+critic step -- the most expensive possible place to learn that a
metric had the wrong Python type.

So the load-bearing assertion here is not "the math is right", it is
"every key this hook writes survives the type dispatch in log_rollout_data".
That is `test_every_written_key_survives_log_rollout_data_dispatch`, which
replicates the real dispatch chain rather than trusting a hand-written list
of allowed types.
"""

from __future__ import annotations

# Import the megatron stub BEFORE anything that reaches slime internals.
import _cp_dist_helpers  # noqa: F401
import pytest
import torch

from slime_plugins.harbor.length_adaptive_gae import (
    DEFAULT_ALPHA,
    _policy_lambda,
    length_adaptive_gae,
)

NUM_GPUS = 0

# Keys the hook is contractually required to populate.
REQUIRED_KEYS = ("advantages", "returns")
# Diagnostic keys the hook adds on top; these are the ones that crashed.
METRIC_KEYS = (
    "gae_lambda_policy_mean",
    "gae_lambda_policy_min",
    "gae_lambda_policy_max",
)


class _Args:
    """Minimal stand-in for the argparse Namespace the hook reads."""

    def __init__(self, gamma: float = 1.0, kl_coef: float = 0.0) -> None:
        self.gamma = gamma
        self.kl_coef = kl_coef


def _make_rollout_data(response_lengths: list[int]) -> dict:
    """A rollout_data dict shaped like the real one on the last PP stage.

    ``total_length == response_length`` keeps the prompt empty so the
    offsets inside ``get_advantages_and_returns`` stay trivial; this test is
    about types and bookkeeping, not about prompt masking.
    """
    return {
        "values": [torch.zeros(n) for n in response_lengths],
        "rewards": [1.0 for _ in response_lengths],
        "kl": [torch.zeros(n) for n in response_lengths],
        "response_lengths": list(response_lengths),
        "total_lengths": list(response_lengths),
    }


# ---------------------------------------------------------------------------
# The regression test for the crash.
# ---------------------------------------------------------------------------


def _dispatch_kind(val) -> str:
    """Classify a value the way ``log_rollout_data``'s if/elif chain does.

    Mirrors slime/backends/megatron_utils/data.py: list/tuple -> the
    (sum, count) reduction path, scalar Tensor -> count=1 path, anything
    else -> ``raise ValueError(f"Unsupported type: ...")``.
    """
    if isinstance(val, (list, tuple)):
        return "list"
    if isinstance(val, torch.Tensor):
        return "tensor"
    return "UNSUPPORTED"


def test_every_written_key_survives_log_rollout_data_dispatch():
    """No key the hook writes may fall through to the ValueError branch.

    This is the test whose absence cost a full multi-hour cluster run.
    """
    args = _Args()
    data = _make_rollout_data([1, 7, 250, 4096])
    before = set(data)

    length_adaptive_gae(args, data)

    written = set(data) - before | {"advantages", "returns"}
    assert written, "hook wrote nothing -- test is not exercising the hook"

    unsupported = {k: type(data[k]).__name__ for k in written if _dispatch_kind(data[k]) == "UNSUPPORTED"}
    assert not unsupported, (
        f"these keys would crash log_rollout_data with 'Unsupported type': {unsupported}. "
        "Wrap scalars in a per-sample list."
    )


def test_metric_keys_are_per_sample_length_aligned():
    """Metric lists must be one entry per sample.

    ``log_rollout_data`` uses ``len(val)`` as the reduction ``count``, so a
    length mismatch silently mis-weights the DP average rather than raising.
    """
    args = _Args()
    lengths = [3, 64, 900]
    data = _make_rollout_data(lengths)

    length_adaptive_gae(args, data)

    for key in METRIC_KEYS:
        assert key in data, f"missing diagnostic key {key}"
        assert len(data[key]) == len(lengths), (
            f"{key} has len {len(data[key])}, expected {len(lengths)} (one per sample) "
            "or the (sum, count) reduction will mis-weight across DP ranks"
        )
        assert all(isinstance(x, float) for x in data[key]), f"{key} must hold plain floats"


def test_reported_mean_matches_true_sample_weighted_mean():
    """The mean list must reduce to the arithmetic mean of per-sample lambdas."""
    args = _Args()
    lengths = [1, 10, 100, 5000]
    data = _make_rollout_data(lengths)

    length_adaptive_gae(args, data)

    expected = sum(_policy_lambda(n, DEFAULT_ALPHA) for n in lengths) / len(lengths)
    reported = sum(data["gae_lambda_policy_mean"]) / len(data["gae_lambda_policy_mean"])
    assert reported == pytest.approx(expected)

    assert data["gae_lambda_policy_min"][0] == pytest.approx(min(_policy_lambda(n, DEFAULT_ALPHA) for n in lengths))
    assert data["gae_lambda_policy_max"][0] == pytest.approx(max(_policy_lambda(n, DEFAULT_ALPHA) for n in lengths))


# ---------------------------------------------------------------------------
# Shape / math contract.
# ---------------------------------------------------------------------------


def test_advantages_and_returns_have_per_token_shapes():
    args = _Args()
    lengths = [5, 33]
    data = _make_rollout_data(lengths)

    length_adaptive_gae(args, data)

    for key in REQUIRED_KEYS:
        assert len(data[key]) == len(lengths)
        for tensor, n in zip(data[key], lengths, strict=True):
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == (n,), f"{key} entry has shape {tensor.shape}, expected ({n},)"
            assert torch.isfinite(tensor).all(), f"{key} contains non-finite values"


@pytest.mark.parametrize(
    "length, expected",
    [
        (0, 0.0),  # fully-masked sample: guarded, no div-by-zero
        (1, 1.0 - 1.0 / 1.5),
        (10, 1.0 - 1.0 / 15.0),
        (100, 1.0 - 1.0 / 150.0),
    ],
)
def test_policy_lambda_values(length, expected):
    assert _policy_lambda(length, DEFAULT_ALPHA) == pytest.approx(expected)


def test_policy_lambda_is_clamped_below_one():
    """Never exactly 1.0 -- a lambda of 1 with gamma=1 makes GAE degenerate."""
    assert _policy_lambda(10**9, DEFAULT_ALPHA) < 1.0


def test_policy_lambda_is_monotonic_in_length():
    lams = [_policy_lambda(n, DEFAULT_ALPHA) for n in (1, 2, 5, 10, 100, 1000, 10000)]
    assert lams == sorted(lams), f"lambda must increase with length, got {lams}"


def test_empty_rollout_writes_no_metric_keys():
    """A DP rank with zero samples must not emit half-formed metrics."""
    args = _Args()
    data = _make_rollout_data([])

    length_adaptive_gae(args, data)

    assert data["advantages"] == []
    assert data["returns"] == []
    for key in METRIC_KEYS:
        assert key not in data, f"{key} written for an empty rollout"


def test_missing_critic_values_raises():
    args = _Args()
    data = _make_rollout_data([4])
    data["values"] = None

    with pytest.raises(AssertionError, match="requires a critic"):
        length_adaptive_gae(args, data)


def test_alpha_override_via_env(monkeypatch):
    monkeypatch.setenv("GAE_ALPHA", "3.0")
    args = _Args()
    data = _make_rollout_data([10])

    length_adaptive_gae(args, data)

    assert data["gae_lambda_policy_mean"][0] == pytest.approx(1.0 - 1.0 / 30.0)


def test_non_positive_alpha_rejected(monkeypatch):
    monkeypatch.setenv("GAE_ALPHA", "0")
    args = _Args()
    data = _make_rollout_data([10])

    with pytest.raises(AssertionError, match="must be positive"):
        length_adaptive_gae(args, data)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
