"""Record-only training/inference log-prob mismatch metrics for the Harbor RL runs.

`get_mismatch_metrics` in slime requires a `--custom-tis-function-path`
(``slime/utils/arguments.py``: assert at ~1786), and the built-in
``vanilla_tis_function`` (``slime/backends/megatron_utils/loss.py``) applies a
truncated-importance-sampling *correction* to ``pg_loss`` (``pg_loss *=
clamp(tis)``). We want the diagnostic metrics **without** perturbing the policy
loss — the run's PPO recipe must stay bit-identical to a run with the metrics
off.

``record_only_tis_function`` mirrors ``vanilla_tis_function``'s signature and
metric math (so the values line up with the framework's own definitions) but
returns ``pg_loss`` and ``loss_masks`` **unchanged**. slime then folds the
returned metrics into ``reported_loss`` and they surface on wandb as
``train/tis`` / ``train/tis_abs`` (plus ``train/ois`` which slime adds from
``exp(-ppo_kl)``). ``train/train_rollout_logprob_abs_diff`` is computed
independently by loss.py whenever the batch carries ``rollout_log_probs``.

Semantics (over response tokens; aggregated by the caller's
``sum_of_sample_mean``):
  tis     = exp(train_logp - rollout_logp)            token-level IS ratio
  tis_abs = |exp(train_logp - rollout_logp) - 1|      train/inference mismatch magnitude

Here ``train_logp`` is the training engine's (re)computed old log-prob and
``rollout_logp`` is the SGLang inference-time log-prob; their gap is the
train-vs-inference distribution drift we want to watch.
"""

from __future__ import annotations

import os
from typing import Any

import torch

# DEBUG-ONLY: install deep loss instrumentation when triggered by env OR a sentinel file.
# This module is load_function-imported by the exact ray actor process that runs the loss
# (via --custom-tis-function-path), so importing here guarantees the monkeypatch lands in
# the compute process. A sentinel FILE is used in addition to the env var because ray actors
# under DIRECT_RUN do not always inherit arbitrary driver env vars. No effect in production
# (env unset AND sentinel file absent).
_DBG_SENTINEL = "/path/to/workspace/slime_dbg_out/.ENABLE_DEBUG"
if os.environ.get("SLIME_DEBUG_LOGPROB_SHAPES", "0") == "1" or os.path.exists(_DBG_SENTINEL):
    try:
        from . import _debug_logprob_shapes as _dbg

        _dbg.install()
    except Exception as _e:  # noqa: BLE001
        import sys

        print(f"[DBG] mismatch_metrics install hook failed: {_e}", file=sys.stderr, flush=True)


def record_only_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    log_ratio = old_log_probs - rollout_log_probs
    tis = torch.exp(log_ratio)
    tis_abs = (torch.exp(log_ratio) - 1).abs()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
        # CORRECTLY-normalized train/inference logprob abs-diff. The built-in
        # loss.py `train_rollout_logprob_abs_diff` (loss.py:1110) is aggregated by the
        # REBUILT `sum_of_sample_mean` (denom=group_mask_sums) which — with
        # micro_batch_size=1 + per-step /step_global_batch_size — sums per-token
        # contributions without dividing by the mask-token count, inflating it ~1000x
        # (e.g. 163/269 vs the true ~0.02). Metrics returned here are instead aggregated
        # by `sum_of_sample_mean_for_mismatch_metrics` (loss.py:1129-1133), the SAME
        # correct per-sample-mask reducer that keeps tis_abs at ~0.02. So this key is
        # the trustworthy value; expect it ≈ tis_abs (since |exp(d)-1|≈|d| for small d).
        "train_rollout_logprob_abs_diff_aligned": log_ratio.abs().clone().detach(),
    }
    # NOTE: pg_loss and loss_masks are returned UNCHANGED — no importance-sampling
    # correction is applied, so the policy loss is identical to a metrics-off run.
    return pg_loss, loss_masks, metrics
