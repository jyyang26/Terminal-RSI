import logging
import os
import time

import wandb

from . import wandb_utils
from .tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        _init_secondary_tolerant(args, **kwargs)


def _init_secondary_tolerant(args, **kwargs):
    """Attach a SECONDARY (non-primary) wandb logger, tolerating failure.

    WHY THIS EXISTS: a 2026-08-21 128-GPU run was killed during init by

        File "slime/ray/rollout.py", line 397, in __init__
          init_tracking(args, primary=False)
        ...
        wandb.errors.CommError: Run initialization has timed out after 90.0 sec

    ``wandb.init`` was a bare call, so one slow round-trip to api.wandb.ai through the
    corporate proxy propagated out of ``RolloutManager.__init__`` into train_async.py and
    took the whole job down (exit 1) before a single training step ran. The PRIMARY run had
    already been created successfully -- only the RolloutManager's attach to it timed out.

    A secondary logger is an OBSERVER: it re-attaches to the run the driver already made
    (mode="shared", x_primary=False) so rollout-side metrics land on the same curves. Every
    metric that gates this recipe (grad_norm, value_loss, reward, batch size) is logged by
    the DRIVER, not here. So this must never be able to kill training. Two changes:

      1. retry a few times -- the failure mode is a transient proxy stall, and the second
         attempt usually succeeds within seconds;
      2. on final failure, log loudly and CONTINUE. ``logging_utils.log`` calls
         ``wandb.log``, which is a no-op when no run is attached in this process, so the
         degraded path is safe: we lose some rollout-side lines in wandb, not the run.

    Deliberately NOT applied to the primary logger: if the driver cannot reach wandb at
    all, failing fast is correct -- that run would have no curves at all, which for a
    multi-hour 128-GPU job is worth stopping for.
    """
    log = logging.getLogger(__name__)
    attempts = int(os.environ.get("SLIME_WANDB_SECONDARY_ATTEMPTS", "3"))
    for attempt in range(1, attempts + 1):
        try:
            wandb_utils.init_wandb_secondary(args, **kwargs)
            if attempt > 1:
                log.warning("secondary wandb attached on attempt %d/%d", attempt, attempts)
            return
        except Exception as e:
            if attempt < attempts:
                backoff = 5 * attempt
                log.warning(
                    "secondary wandb init failed (attempt %d/%d): %s -- retrying in %ds",
                    attempt,
                    attempts,
                    e,
                    backoff,
                )
                time.sleep(backoff)
            else:
                log.error(
                    "secondary wandb init failed after %d attempts: %s -- CONTINUING WITHOUT IT. "
                    "Training is unaffected; driver-side metrics (grad_norm, value_loss, reward) "
                    "still reach wandb. Only some rollout-side log lines are lost.",
                    attempts,
                    e,
                )


def update_tracking_open_metrics(args, router_addr):
    wandb_utils.reinit_wandb_primary_with_open_metrics(args, router_addr)


def finish_tracking(args):
    if not args.use_wandb:
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if args.use_wandb:
        # wandb.log RAISES ("You must call wandb.init() before wandb.log()") when no run is
        # attached in this process -- it is not a silent no-op. That matters because
        # _init_secondary_tolerant now lets a process continue with no wandb run: without
        # this guard the job would survive init only to die at the first rollout log call,
        # which is a worse failure than the one we set out to fix.
        if wandb.run is not None:
            wandb.log(metrics)
        else:
            logging.getLogger(__name__).debug("no wandb run in this process; skipping %d metrics", len(metrics))

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
