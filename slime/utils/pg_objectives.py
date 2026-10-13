"""Shared optional SC mount; preserve the original PPO tensor when disabled."""

import torch


def apply_score_centering(args, batch, logits, *, advantages, pg_loss):
    if not getattr(args, "enable_score_centering", False):
        return pg_loss, {}
    from slime.backends.megatron_utils.sc_terms import get_score_centering_terms

    if args.sc_correction_scale == 0:
        with torch.no_grad():
            addend, metrics = get_score_centering_terms(args, batch, logits, advantages=advantages)
    else:
        addend, metrics = get_score_centering_terms(args, batch, logits, advantages=advantages)
        pg_loss = pg_loss + args.sc_correction_scale * addend
    metrics["sc_applied"] = torch.full_like(addend.detach(), float(args.sc_correction_scale != 0))
    return pg_loss, metrics
