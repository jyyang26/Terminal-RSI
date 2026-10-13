import logging
import os
import shutil

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from slime.utils.misc import should_run_periodic_action
from slime_plugins.harbor.eval_gate import validate_baseline_eval_gate

logger = logging.getLogger(__name__)


def _rotate_checkpoints(save_dir: str, keep: int) -> None:
    """Delete the oldest iter_* checkpoints under save_dir, keeping only the most recent `keep`."""
    if keep <= 0 or not os.path.isdir(save_dir):
        return
    iters = sorted(
        [d for d in os.listdir(save_dir) if d.startswith("iter_") and os.path.isdir(os.path.join(save_dir, d))]
    )
    to_delete = iters[: max(0, len(iters) - keep)]
    for name in to_delete:
        path = os.path.join(save_dir, name)
        logger.info(f"checkpoint rotation: removing {path}")
        shutil.rmtree(path, ignore_errors=True)




# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    rollout_manager = None
    rollout_data_next_future = None
    tracking_started = False
    try:
        # allocate the GPUs
        pgs = create_placement_groups(args)
        init_tracking(args)
        tracking_started = True

        # create the rollout manager, with sglang engines inside.
        # need to initialize rollout manager first to calculate num_rollout
        rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

        # Update primary W&B with SGLang metrics endpoint now that servers are up.
        router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
        update_tracking_open_metrics(args, router_addr)

        # create the actor and critic models
        actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

        # Always push actor weights to rollout once weights are loaded.
        actor_model.update_weights()

        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

        # Step-0 baseline eval, BEFORE any training. train.py (sync) runs this
        # inside its loop via `rollout_id == 0 and not skip_eval_before_train`,
        # but the async loop never had that branch, so --skip-eval-before-train
        # was silently ignored and the pre-training baseline never ran. Mirror it
        # here, gated on start_rollout_id == 0 so resumed runs (start_rollout_id
        # > 0) don't re-run the baseline. Runs on the freshly-loaded weights
        # pushed by update_weights() above, before the first training rollout is
        # kicked off (no engine contention).
        if (
            args.eval_interval is not None
            and not args.skip_eval_before_train
            and args.start_rollout_id == 0
        ):
            baseline_eval_metrics = ray.get(rollout_manager.eval.remote(args.start_rollout_id))
            validate_baseline_eval_gate(args, baseline_eval_metrics)

        # async train loop.
        rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            # Sync the last generation
            if rollout_data_next_future is not None:
                rollout_data_curr_ref = ray.get(rollout_data_next_future)
                rollout_data_next_future = None

            # Start the next rollout early.
            if rollout_id + 1 < args.num_rollout:
                rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

            if args.use_critic:
                actor_trains_this_step = rollout_id >= args.num_critic_only_steps
                value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)
                if actor_trains_this_step:
                    ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref, external_data=value_refs))
                else:
                    ray.get(value_refs)
            else:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

            if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
                # Skip saves that fall inside the critic-only warmup window (rollout_id < num_critic_only_steps).
                # This prevents a half-trained critic checkpoint from being written and rotated into the keep set.
                # For example, with num_critic_only_steps=2 and save_interval=2, rollout_id=1 triggers the
                # periodic check but the actor has not yet trained, so we skip it entirely.
                _in_critic_warmup = args.use_critic and rollout_id < args.num_critic_only_steps
                if not _in_critic_warmup:
                    is_last = rollout_id == args.num_rollout - 1
                    if (not args.use_critic) or rollout_id >= args.num_critic_only_steps:
                        actor_model.save_model(rollout_id, force_sync=is_last)
                        _rotate_checkpoints(
                            args.save,
                            keep=getattr(args, "save_keep_last", 4),
                        )
                    if args.use_critic:
                        critic_model.save_model(rollout_id, force_sync=is_last)
                        _rotate_checkpoints(
                            args.save + "_critic",
                            keep=getattr(args, "save_keep_last", 4),
                        )
                    if args.rollout_global_dataset:
                        ray.get(rollout_manager.save.remote(rollout_id))
                else:
                    logger.info(
                        f"checkpoint save skipped at rollout_id={rollout_id}: "
                        f"still in critic-only warmup (num_critic_only_steps={args.num_critic_only_steps})"
                    )

            if (rollout_id + 1) % args.update_weights_interval == 0:
                # sync generate before update weights to prevent update weight in the middle of generation
                rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
                rollout_data_next_future = None
                actor_model.update_weights()

            if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
                ray.get(rollout_manager.eval.remote(rollout_id))
    finally:
        if rollout_data_next_future is not None:
            try:
                ray.cancel(rollout_data_next_future, force=False)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to cancel outstanding rollout generation during async train shutdown.")
        if rollout_manager is not None:
            try:
                ray.get(rollout_manager.dispose.remote())
            except Exception:  # noqa: BLE001
                logger.exception("Failed to dispose RolloutManager during async train shutdown.")
        if tracking_started:
            finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
