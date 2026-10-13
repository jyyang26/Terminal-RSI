"""Stage 3 integration tests: PPO + Score Centering on the Harbor PPO path.

These drive the *real* loss functions -- ``slime_plugins.harbor.train_loss.
policy_loss_no_entropy_when_disabled`` (what the production recipe actually
runs via ``--loss-type custom_loss``) and core ``policy_loss_function`` -- under
a real single-rank Megatron model-parallel group on CPU.

Two things are asserted:

  * **Equivalence.** With ``enable_score_centering=False``, the loss and every
    gradient must be bit-identical to the same call on an args object that has
    no SC attributes at all (i.e. the pre-patch world).
  * **Function.** With it on, the correction must be applied, be reported, be
    finite, and shift the gradient by exactly the folded drift vector.

Run with: ``python -m pytest tests/test_ppo_sc_harbor_integration.py -q``

See PPO_SC_INTEGRATION_PLAN.md section 5 (Stage 3).
"""

from __future__ import annotations

import importlib
import os
from argparse import Namespace

import pytest
import torch

NUM_GPUS = 0

VOCAB = 64
TOP_K = 8
EPS_CLIP = 0.2


# --------------------------------------------------------------------------
# single-rank Megatron harness
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def megatron_single_rank():
    """A real 1-rank TP=CP=PP=1 group.

    The loss path calls ``mpu.get_context_parallel_world_size()`` and
    ``mpu.get_tensor_model_parallel_group()``, and an uninitialized mpu reports
    a CP world size of 0, which matches none of the branches. Standing up a
    genuine single-rank gloo group is cheaper and far more faithful than
    monkeypatching mpu.
    """
    import torch.distributed as dist
    from megatron.core import mpu

    # Several existing tests install a fake ``megatron.core.mpu`` into
    # ``sys.modules`` and never restore it (e.g. tests/test_explained_variance.py
    # :41). If one of those ran first, the real model-parallel API is gone for
    # the rest of the session and there is nothing we can initialize. Skip
    # rather than error: these tests pass on their own and in CI, where the
    # suite is sharded, and the alternative would be to reach into another
    # test's globals.
    if not hasattr(mpu, "initialize_model_parallel") or not hasattr(mpu, "get_context_parallel_group"):
        pytest.skip(
            "megatron.core.mpu has been replaced by a stub from an earlier test module; "
            "run this file on its own (pytest tests/test_ppo_sc_harbor_integration.py)"
        )

    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29577")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        created = True

    # Probe for a live CP group rather than trusting the world size: before
    # initialize_model_parallel, get_context_parallel_world_size() returns 0,
    # but get_context_parallel_group() asserts. Only the latter distinguishes
    # "not initialized" from "initialized with size 1".
    try:
        mpu.get_context_parallel_group()
        initialized = True
    except AssertionError:
        initialized = False
    if not initialized:
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
        )

    yield mpu

    if created:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def baseline_args(**overrides):
    """An args object matching the two production launch scripts.

    Values resolved in PPO_SC_INTEGRATION_PLAN.md section 1.3: PPO estimator,
    eps_clip 0.2, keep_old_actor on, rollout logprobs off, no TIS.
    """
    args = Namespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        allgather_cp=False,
        log_probs_chunk_size=-1,
        advantage_estimator="ppo",
        eps_clip=EPS_CLIP,
        eps_clip_high=EPS_CLIP,
        eps_clip_c=None,
        entropy_coef=0.0,
        use_rollout_logprobs=False,
        use_opsm=False,
        use_tis=False,
        use_kl_loss=False,
        use_unbiased_kl=False,
        get_mismatch_metrics=False,
        custom_tis_function_path=None,
        custom_pg_loss_reducer_function_path=None,
        calculate_per_token_loss=False,
        kl_coef=0.0,
        kl_loss_coef=0.0,
        kl_loss_type="low_var_kl",
        keep_old_actor=True,
        harbor_report_entropy="0",  # keep the CPU test cheap and deterministic
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def sc_args(**overrides):
    defaults = {
        "enable_score_centering": True,
        "sc_top_k": TOP_K,
        "sc_tail_eps": 1e-8,
        "sc_weight_denominator": "sampler",
        "sc_formula": "score",
        "sc_chunk_size": 16,
        "sc_correction_scale": 1.0,
        "sc_assert_alignment": "full",
    }
    defaults.update(overrides)
    return baseline_args(**defaults)


def make_batch(*, seed=0, total_lengths=(9, 7), response_lengths=(4, 3), with_heads=False, sharpness=1.0):
    """Build a micro-batch plus the ``[1, T, V]`` logits the loss consumes.

    The sampler heads are derived from a *perturbed* copy of the trainer
    distribution, which is what training-inference mismatch looks like: the
    sampler is a biased copy of the trainer, not an unrelated distribution.
    """
    torch.manual_seed(seed)
    total = sum(total_lengths)
    logits = (torch.randn(1, total, VOCAB, dtype=torch.float32) / sharpness).requires_grad_(True)

    unconcat_tokens, loss_masks, advantages, rollout_log_probs = [], [], [], []
    head_ids, head_logps = [], []

    with torch.no_grad():
        flat = logits.detach().squeeze(0)
        logp_full = torch.log_softmax(flat, dim=-1)
        # A sampler that differs from the trainer by a modest logit perturbation.
        sampler_logp_full = torch.log_softmax(flat + torch.randn_like(flat) * 0.6, dim=-1)

    cursor = 0
    for total_length, response_length in zip(total_lengths, response_lengths, strict=True):
        unconcat_tokens.append(torch.randint(0, VOCAB, (total_length,)))
        loss_masks.append(torch.ones(response_length, dtype=torch.int))
        advantages.append(torch.randn(response_length))

        # Rows for this sample's response, matching get_responses' cp1 slicing:
        # logits[start - 1 : end - 1] where end = cursor + total_length.
        end = cursor + total_length
        start = end - response_length
        rows = slice(start - 1, end - 1)
        response_tokens = unconcat_tokens[-1][-response_length:]

        with torch.no_grad():
            sampler_rows = sampler_logp_full[rows]
            rollout_log_probs.append(sampler_rows.gather(-1, response_tokens.unsqueeze(-1)).squeeze(-1).clone())
            if with_heads:
                # A real sampler emits its own top-k, with distinct ids. Rather
                # than overwriting column 0 with the sampled token (which can
                # duplicate an id already present), bias the sampled token's
                # logit so it lands in the top-k naturally -- the realistic
                # case, since the sampler drew it from this distribution.
                biased = sampler_rows.clone()
                biased.scatter_(
                    -1,
                    response_tokens.unsqueeze(-1),
                    torch.full((response_length, 1), float(sampler_rows.max()) + 1.0),
                )
                head_index = biased.topk(TOP_K, dim=-1).indices
                head_ids.append(head_index.to(torch.int32))
                head_logps.append(sampler_rows.gather(-1, head_index).clone())
        cursor += total_length

    batch = {
        "unconcat_tokens": unconcat_tokens,
        "total_lengths": list(total_lengths),
        "response_lengths": list(response_lengths),
        "loss_masks": loss_masks,
        "advantages": advantages,
        "rollout_log_probs": rollout_log_probs,
        "group_mask_sums": None,
        "log_probs": None,
    }
    if with_heads:
        batch["rollout_topk_token_ids"] = head_ids
        batch["rollout_topk_log_probs"] = head_logps
        batch["sc_response_token_ids"] = [
            tokens[-r:].int() for tokens, r in zip(unconcat_tokens, response_lengths, strict=True)
        ]
    return batch, logits


def num_samples(batch):
    """The reducer's implicit scale factor.

    ``get_sum_of_sample_mean`` *sums* per-sample means, and the real consumer
    (``train_one_step``) divides the result by ``step_global_batch_size``. These
    tests call the reducer directly, so a metric that is a per-token constant
    ``c`` arrives here as ``c * num_samples``. Dividing by this recovers ``c``.
    """
    return len(batch["response_lengths"])


def run_loss(loss_fn, args, batch, logits):
    """Invoke a slime loss function and return ``(loss, metrics, grad)``."""
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean

    reducer = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        batch["group_mask_sums"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens"),
    )
    loss, metrics = loss_fn(args, batch, logits, reducer)
    (grad,) = torch.autograd.grad(loss, logits, retain_graph=True)
    return loss, metrics, grad


HARBOR_LOSS = "slime_plugins.harbor.train_loss", "policy_loss_no_entropy_when_disabled"
CORE_LOSS = "slime.backends.megatron_utils.loss", "policy_loss_function"


def load_loss(spec):
    module, name = spec
    return getattr(importlib.import_module(module), name)


# --------------------------------------------------------------------------
# equivalence: switch off == pre-patch
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("spec", [HARBOR_LOSS, CORE_LOSS], ids=["harbor_fork", "core"])
def test_switch_off_is_bit_identical_to_a_prepatch_args_object(megatron_single_rank, spec):
    """The strongest equivalence statement available without a git checkout.

    ``no_sc`` has *no* SC attributes whatsoever, so it is byte-for-byte the
    namespace the loss saw before this feature existed; ``off`` has them all,
    set to their defaults. Identical loss and identical gradients means the
    disabled path cannot have changed behaviour.
    """
    loss_fn = load_loss(spec)

    batch_a, logits_a = make_batch(seed=3)
    loss_off, metrics_off, grad_off = run_loss(loss_fn, sc_args(enable_score_centering=False), batch_a, logits_a)

    batch_b, logits_b = make_batch(seed=3)
    loss_pre, metrics_pre, grad_pre = run_loss(loss_fn, baseline_args(), batch_b, logits_b)

    assert torch.equal(loss_off, loss_pre), (loss_off.item(), loss_pre.item())
    assert torch.equal(grad_off, grad_pre)
    assert set(metrics_off) == set(metrics_pre)
    for key in metrics_pre:
        assert torch.equal(metrics_off[key], metrics_pre[key]), key


@pytest.mark.unit
@pytest.mark.parametrize("spec", [HARBOR_LOSS, CORE_LOSS], ids=["harbor_fork", "core"])
def test_switch_off_emits_no_sc_metrics(megatron_single_rank, spec):
    loss_fn = load_loss(spec)
    batch, logits = make_batch(seed=4)
    _, metrics, _ = run_loss(loss_fn, sc_args(enable_score_centering=False), batch, logits)
    assert not [key for key in metrics if key.startswith("sc_")]


@pytest.mark.unit
@pytest.mark.parametrize("spec", [HARBOR_LOSS, CORE_LOSS], ids=["harbor_fork", "core"])
def test_switch_off_ignores_heads_even_when_present(megatron_single_rank, spec):
    """Heads on the batch must not change anything while the flag is off.

    This is the "no silent behaviour shift" guarantee: a rollout that started
    attaching heads (say after a config change elsewhere) must not quietly turn
    the correction on.
    """
    loss_fn = load_loss(spec)

    batch_heads, logits_heads = make_batch(seed=5, with_heads=True)
    loss_with, _, grad_with = run_loss(loss_fn, sc_args(enable_score_centering=False), batch_heads, logits_heads)

    batch_plain, logits_plain = make_batch(seed=5, with_heads=False)
    loss_without, _, grad_without = run_loss(loss_fn, baseline_args(), batch_plain, logits_plain)

    assert torch.equal(loss_with, loss_without)
    assert torch.equal(grad_with, grad_without)


# --------------------------------------------------------------------------
# function: switch on
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("spec", [HARBOR_LOSS, CORE_LOSS], ids=["harbor_fork", "core"])
def test_switch_on_reports_metrics_and_stays_finite(megatron_single_rank, spec):
    loss_fn = load_loss(spec)
    batch, logits = make_batch(seed=6, with_heads=True)
    loss, metrics, grad = run_loss(loss_fn, sc_args(), batch, logits)

    for key in (
        "sc_correction",
        "sc_alpha",
        "sc_sampler_head_mass",
        "sc_train_head_mass",
        "sc_coefficient_l1",
        "sc_applied",
    ):
        assert key in metrics, sorted(metrics)
        assert torch.isfinite(metrics[key]).all(), f"{key} = {metrics[key]}"
    assert torch.isfinite(loss).all()
    assert torch.isfinite(grad).all()
    # The head cross-check ran and found the sampled token.
    assert metrics["sc_head_hit"] > 0.0


@pytest.mark.unit
@pytest.mark.parametrize("spec", [HARBOR_LOSS, CORE_LOSS], ids=["harbor_fork", "core"])
def test_switch_on_changes_the_gradient(megatron_single_rank, spec):
    """Sanity: the feature must actually do something when enabled."""
    loss_fn = load_loss(spec)

    batch_on, logits_on = make_batch(seed=7, with_heads=True)
    _, _, grad_on = run_loss(loss_fn, sc_args(), batch_on, logits_on)

    batch_off, logits_off = make_batch(seed=7, with_heads=True)
    _, _, grad_off = run_loss(loss_fn, sc_args(enable_score_centering=False), batch_off, logits_off)

    assert not torch.allclose(grad_on, grad_off)


@pytest.mark.unit
@pytest.mark.parametrize("spec", [HARBOR_LOSS, CORE_LOSS], ids=["harbor_fork", "core"])
def test_correction_scale_zero_is_shadow_mode(megatron_single_rank, spec):
    """scale=0 must report the correction while leaving the gradient alone.

    This is the mechanism Stage 4.3 relies on to observe SC on a live run
    without changing what the optimizer does.
    """
    loss_fn = load_loss(spec)

    batch_shadow, logits_shadow = make_batch(seed=8, with_heads=True)
    _, metrics_shadow, grad_shadow = run_loss(loss_fn, sc_args(sc_correction_scale=0.0), batch_shadow, logits_shadow)

    batch_off, logits_off = make_batch(seed=8, with_heads=True)
    _, _, grad_off = run_loss(loss_fn, sc_args(enable_score_centering=False), batch_off, logits_off)

    assert torch.equal(grad_shadow, grad_off), "shadow mode must not perturb the gradient"
    assert "sc_correction" in metrics_shadow
    assert metrics_shadow["sc_applied"] == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("spec", [HARBOR_LOSS, CORE_LOSS], ids=["harbor_fork", "core"])
def test_gradient_delta_equals_the_folded_drift(megatron_single_rank, spec):
    """d(loss_sc - loss_ppo)/d(logits) must equal the analytic correction term.

    Recomputes ``sum_of_sample_mean(A * correction)`` independently and
    differentiates it, then checks the difference of the two full losses. This
    ties the mounted term to the module-level maths rather than merely checking
    that "something changed".
    """
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
    from slime.backends.megatron_utils.sc_terms import get_score_centering_terms

    loss_fn = load_loss(spec)
    args = sc_args()

    batch, logits = make_batch(seed=9, with_heads=True)
    loss_on, _, grad_on = run_loss(loss_fn, args, batch, logits)

    batch_off, logits_off = make_batch(seed=9, with_heads=True)
    _, _, grad_off = run_loss(loss_fn, sc_args(enable_score_centering=False), batch_off, logits_off)

    # Independent recomputation of just the added term, on a third graph.
    batch_ref, logits_ref = make_batch(seed=9, with_heads=True)
    advantages = torch.cat(batch_ref["advantages"], dim=0)
    correction, _ = get_score_centering_terms(args, batch_ref, logits_ref, advantages=advantages)
    reducer = get_sum_of_sample_mean(
        batch_ref["total_lengths"],
        batch_ref["response_lengths"],
        batch_ref["loss_masks"],
        batch_ref["group_mask_sums"],
        args.calculate_per_token_loss,
        args.qkv_format,
        None,
    )
    (grad_term,) = torch.autograd.grad(reducer(correction), logits_ref)

    torch.testing.assert_close(grad_on - grad_off, grad_term, atol=1e-6, rtol=1e-5)


@pytest.mark.unit
def test_sharp_distribution_drives_the_correction_toward_zero(megatron_single_rank):
    """A near-deterministic sampler leaves nothing to clip, hence nothing to center.

    The zero anchor (T-4) expressed end-to-end: with a very peaked distribution
    the head covers nearly all the mass and the ratios sit at 1, so the folded
    coefficients collapse and the correction becomes negligible.
    """
    loss_fn = load_loss(HARBOR_LOSS)
    args = sc_args(sc_top_k=VOCAB)  # whole vocabulary in the head: no tail model

    batch, logits = make_batch(seed=10, with_heads=True, sharpness=1.0)
    # Overwrite the heads with the trainer's own distribution: q == p exactly.
    with torch.no_grad():
        flat = torch.log_softmax(logits.detach().squeeze(0), dim=-1)
        cursor = 0
        for index, (total_length, response_length) in enumerate(
            zip(batch["total_lengths"], batch["response_lengths"], strict=True)
        ):
            end = cursor + total_length
            rows = flat[end - response_length - 1 : end - 1]
            top = rows.topk(VOCAB, dim=-1)
            batch["rollout_topk_token_ids"][index] = top.indices.to(torch.int32)
            batch["rollout_topk_log_probs"][index] = top.values.clone()
            tokens = batch["unconcat_tokens"][index][-response_length:]
            batch["rollout_log_probs"][index] = rows.gather(-1, tokens.unsqueeze(-1)).squeeze(-1).clone()
            cursor += total_length

    _, metrics, _ = run_loss(loss_fn, args, batch, logits)
    # Full-vocabulary SC has a nonzero entropy-like scalar even when q=p;
    # its gradient is zero because sum_v p_v grad(log p_v)=0.
    batch_off, logits_off = make_batch(seed=10, with_heads=False, sharpness=1.0)
    _, _, grad_off = run_loss(loss_fn, sc_args(enable_score_centering=False), batch_off, logits_off)
    _, _, grad_on = run_loss(loss_fn, args, batch, logits)
    torch.testing.assert_close(grad_on, grad_off, atol=1e-6, rtol=1e-5)


# --------------------------------------------------------------------------
# defensive assertion
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_assertion_fires_when_sc_requested_but_never_applied():
    """The guard that turns silent no-op into a hard failure."""
    from slime.backends.megatron_utils import sc_terms

    sc_terms.reset_score_centering_applied()
    with pytest.raises(RuntimeError, match="never"):
        sc_terms.assert_score_centering_was_applied(
            sc_args(loss_type="custom_loss", custom_loss_function_path="some.other.loss")
        )


@pytest.mark.unit
def test_assertion_is_silent_when_sc_is_off():
    from slime.backends.megatron_utils import sc_terms

    sc_terms.reset_score_centering_applied()
    sc_terms.assert_score_centering_was_applied(baseline_args())


@pytest.mark.unit
def test_assertion_passes_after_the_loss_ran(megatron_single_rank):
    from slime.backends.megatron_utils import sc_terms

    sc_terms.reset_score_centering_applied()
    batch, logits = make_batch(seed=11, with_heads=True)
    run_loss(load_loss(HARBOR_LOSS), sc_args(), batch, logits)
    assert sc_terms.score_centering_was_applied()
    sc_terms.assert_score_centering_was_applied(sc_args())


# --------------------------------------------------------------------------
# contract failures surface loudly
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_missing_heads_raise_a_clear_error(megatron_single_rank):
    batch, logits = make_batch(seed=12, with_heads=False)
    with pytest.raises(ValueError, match="missing training batch"):
        run_loss(load_loss(HARBOR_LOSS), sc_args(), batch, logits)


@pytest.mark.unit
def test_wrong_head_width_raises(megatron_single_rank):
    batch, logits = make_batch(seed=13, with_heads=True)
    with pytest.raises(ValueError, match="shape"):
        run_loss(load_loss(HARBOR_LOSS), sc_args(sc_top_k=TOP_K + 1), batch, logits)


@pytest.mark.unit
def test_misaligned_head_logprobs_are_detected(megatron_single_rank):
    """Shift the head *values* by one row and require the cross-check to catch it.

    This is the failure mode that is otherwise invisible: the ids still line up
    with the tokens, so shapes match, the sampled token is still found in the
    head, rho stays plausible and the loss still descends -- but every head
    log-prob now belongs to the neighbouring token, so the correction is
    computed against the wrong distribution. Only reconciling the head against
    the independently transported ``rollout_log_probs`` catches it.
    """
    batch, logits = make_batch(seed=14, with_heads=True)
    batch["rollout_topk_log_probs"] = [value.roll(shifts=1, dims=0) for value in batch["rollout_topk_log_probs"]]

    with pytest.raises(ValueError, match="differs"):
        run_loss(load_loss(HARBOR_LOSS), sc_args(), batch, logits)


@pytest.mark.unit
def test_misaligned_head_ids_collapse_the_hit_rate(megatron_single_rank):
    """Rolling the ids too is caught by the reported hit rate rather than a raise.

    When the ids move as well, the sampled token usually stops appearing in the
    head at all, so there is nothing to reconcile and no deviation to measure.
    The observable signal is ``sc_head_hit`` collapsing -- which is why it
    is logged, and why Stage 4 gates on it being close to 1.
    """
    batch, logits = make_batch(seed=14, with_heads=True)
    for key in ("rollout_topk_token_ids", "rollout_topk_log_probs"):
        batch[key] = [value.roll(shifts=1, dims=0) for value in batch[key]]

    _, metrics, _ = run_loss(load_loss(HARBOR_LOSS), sc_args(), batch, logits)
    hit_rate = metrics["sc_head_hit"] / num_samples(batch)
    assert hit_rate < 0.5, hit_rate

    # Contrast: correctly aligned heads find the sampled token every time.
    good_batch, good_logits = make_batch(seed=14, with_heads=True)
    _, good_metrics, _ = run_loss(load_loss(HARBOR_LOSS), sc_args(), good_batch, good_logits)
    good_rate = good_metrics["sc_head_hit"] / num_samples(good_batch)
    assert good_rate == pytest.approx(1.0)


@pytest.mark.unit
def test_alignment_check_cannot_be_bypassed(megatron_single_rank):
    """Production data alignment is mandatory, including callers bypassing CLI validation."""
    batch, logits = make_batch(seed=14, with_heads=True)
    batch["rollout_topk_log_probs"] = [value.roll(shifts=1, dims=0) for value in batch["rollout_topk_log_probs"]]
    with pytest.raises(ValueError, match="differs"):
        run_loss(load_loss(HARBOR_LOSS), sc_args(sc_assert_alignment="off"), batch, logits)
