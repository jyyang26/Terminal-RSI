from __future__ import annotations

import os


def _should_report_entropy(args) -> bool:
    """Whether to compute policy entropy for logging when entropy_coef == 0.

    Default ON: entropy is a cheap (no_grad) but critical health metric. Disable
    by setting ``args.harbor_report_entropy`` or env ``HARBOR_REPORT_ENTROPY`` to
    a falsey value ("0", "false", "no", "off").
    """

    value = getattr(args, "harbor_report_entropy", None)
    if value is None:
        value = os.environ.get("HARBOR_REPORT_ENTROPY", "1")
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def policy_loss_no_entropy_when_disabled(args, batch, logits, sum_of_sample_mean):
    """Policy loss variant that skips entropy graph construction when entropy_coef is zero."""

    import torch

    from slime.backends.megatron_utils.cp_utils import all_gather_with_cp, get_sum_of_sample_mean
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy, vanilla_tis_function
    from slime.utils.misc import load_function
    from slime.utils.pg_objectives import apply_score_centering
    from slime.utils.ppo_utils import (
        compute_approx_kl,
        compute_gspo_kl,
        compute_opsm_mask,
        compute_policy_loss,
    )

    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch.get("log_probs")

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)
    with_entropy = float(getattr(args, "entropy_coef", 0.0) or 0.0) != 0.0

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=with_entropy,
        max_seq_lens=max_seq_lens,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    if not args.use_rollout_logprobs and not old_log_probs:
        old_log_probs = [log_prob.detach() for log_prob in log_probs]
    train_log_probs_for_tis = batch.get("log_probs")
    if not train_log_probs_for_tis:
        train_log_probs_for_tis = [log_prob.detach() for log_prob in log_probs]

    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"
    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)

    # The optional SC addend leaves the original clipped surrogate intact.
    sc_metrics = {}
    if getattr(args, "enable_score_centering", False):
        pg_loss, sc_metrics = apply_score_centering(args, batch, logits, advantages=advantages, pg_loss=pg_loss)

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    if args.get_mismatch_metrics or args.use_tis:
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"
        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": train_log_probs_for_tis,
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # NOTE: positional args must match get_sum_of_sample_mean's signature
        # (total_lengths, response_lengths, loss_masks, sample_denoms,
        #  calculate_per_token_loss, qkv_format, max_seq_lens). The previous call
        # omitted batch["group_mask_sums"] for the 4th (sample_denoms) slot, so
        # calculate_per_token_loss (False=0) landed in sample_denoms -> denom
        # clamped to 1 -> pg_loss / entropy / train_rollout_logprob_abs_diff were
        # left un-normalized (inflated ~mask-token-count, ~1000x) whenever
        # get_mismatch_metrics or use_tis rebuilt the reducer. Pass group_mask_sums
        # to match the core loss.py:1055 call and restore correct per-rollout means.
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            batch["group_mask_sums"],
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
        )

    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    if with_entropy:
        entropy = torch.cat(log_probs_and_entropy["entropy"], dim=0)
        entropy_loss = sum_of_sample_mean(entropy)
    else:
        # entropy_coef == 0: entropy is NOT part of the optimization objective, but
        # we still want it as a monitoring signal (entropy collapse is the canonical
        # RL-degeneration tell). Compute it under torch.no_grad() so NO backward
        # graph / activations are retained -- this is what makes it memory-safe on
        # the 27B (the whole reason this custom loss skips the grad-enabled entropy).
        # Gated by HARBOR_REPORT_ENTROPY (default on); set to 0 to fully skip.
        entropy_loss = logits.new_zeros(())
        if _should_report_entropy(args):
            with torch.no_grad():
                _, entropy_only = get_log_probs_and_entropy(
                    logits,
                    args=args,
                    unconcat_tokens=batch["unconcat_tokens"],
                    total_lengths=total_lengths,
                    response_lengths=response_lengths,
                    with_entropy=True,
                    max_seq_lens=max_seq_lens,
                )
                entropy_report = torch.cat(entropy_only["entropy"], dim=0)
                # Reuse the same per-sample-mean reducer so the reported number is
                # directly comparable to the entropy_coef != 0 path.
                entropy_loss = sum_of_sample_mean(entropy_report).detach()

    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)
        loss = loss + args.kl_loss_coef * kl_loss

    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        train_rollout_logprob_abs_diff = sum_of_sample_mean((old_log_probs - rollout_log_probs).abs())

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        for metric_key, metric_value in tis_metrics.items():
            reported_loss[f"{metric_key}"] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    # Empty unless score centering is on, so this loop is a no-op on the baseline.
    for metric_key, metric_value in sc_metrics.items():
        reported_loss[metric_key] = sum_of_sample_mean(metric_value.detach()).clone().detach()

    if "opd_reverse_kl" in batch:
        opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
        reported_loss["opd_reverse_kl"] = sum_of_sample_mean(opd_reverse_kl).clone().detach()

    return loss, reported_loss
