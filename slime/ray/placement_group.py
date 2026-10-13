import copy
import logging
import os
import socket

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .actor_group import RayTrainGroup
from .rollout import RolloutManager

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)


def _create_placement_group(num_gpus):
    """Create a placement group with the specified number of GPUs."""
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    ray.get(pg.ready())
    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]
    # Map from logical index -> node IP. This is parallel to the bundle/GPU
    # lists and lets rollout-side auxiliary workers be pinned per node.
    pg_reordered_node_ips = [gpu_ids[info[0]][0] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids, pg_reordered_node_ips


def unique_node_bundles(reordered_bundle_indices, reordered_node_ips):
    """Return one representative placement-group bundle per distinct node."""
    assert len(reordered_bundle_indices) == len(reordered_node_ips)
    seen: set[str] = set()
    node_bundles: list[tuple[int, str]] = []
    for bundle_idx, node_ip in zip(reordered_bundle_indices, reordered_node_ips, strict=True):
        if node_ip in seen:
            continue
        seen.add(node_ip)
        node_bundles.append((bundle_idx, node_ip))
    return node_bundles


def create_placement_groups(args):
    """Create placement groups for actor, critic, and rollout engines."""

    num_gpus = 0
    if args.debug_train_only:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
    elif args.debug_rollout_only:
        num_gpus = args.rollout_num_gpus
        rollout_offset = 0
    elif args.colocate:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
    else:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node + args.rollout_num_gpus
        rollout_offset = args.actor_num_nodes * args.actor_num_gpus_per_node

    logger.info(f"Creating placement group with {num_gpus} GPUs...")
    (
        pg,
        actor_pg_reordered_bundle_indices,
        actor_pg_reordered_gpu_ids,
        actor_pg_reordered_node_ips,
    ) = _create_placement_group(num_gpus)
    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]
    rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:]
    rollout_pg_reordered_node_ips = actor_pg_reordered_node_ips[rollout_offset:]

    result = {
        "actor": (pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids, actor_pg_reordered_node_ips),
        "rollout": (
            pg,
            rollout_pg_reordered_bundle_indices,
            rollout_pg_reordered_gpu_ids,
            rollout_pg_reordered_node_ips,
        ),
    }

    result["critic"] = result["actor"] if args.use_critic else None

    return result


def allocate_train_group(args, num_nodes, num_gpus_per_node, pg, role="actor"):
    return RayTrainGroup(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=0.4,
        role=role,
    )


def _isolate_critic_checkpoint_dirs(critic_args):
    """Redirect the critic's --save to a ``<dir>_critic`` sibling.

    The critic args are a deepcopy of the actor args, so they share --save.
    Megatron's save_checkpoint writes role-agnostic shards into
    {save}/iter_NNNNNNN/, and train_async saves actor THEN critic, so without
    this the critic overwrites the actor's checkpoint in place (observed: the
    saved iter dir held only the critic's value head [1, 5120], the actor's LM
    head clobbered). Suffixing --save keeps each role's checkpoints separate.

    ONLY --save is suffixed, NOT --load. On a cold start the actor's --load
    points at the shared pretrained backbone (PRETRAINED_CKPT / the staged
    torch_dist ckpt), which has no ``_critic`` sibling — suffixing --load there
    makes the critic try to load a nonexistent dir and assert-crash at init
    (checkpoint.py:102). The critic loads the SAME backbone as the actor and
    reinitializes its value head (model.py:_critic_output_layer_needs_reinit),
    so sharing --load with the actor is correct on cold start. No-op for an
    unset --save.

    Warm-load (implemented below): when ``<load>_critic`` exists on disk, the
    critic loads its trained value head from there but with
    no_load_optim/no_load_rng/finetune set, so ONLY the weights transfer — not
    the source run's optimizer/LR-schedule/RNG (which would mismatch this run's
    schedule and assert-crash). Cold start (no ``_critic`` sibling) is unaffected
    and still loads the shared backbone + reinitializes the value head.

    This MUST run for every critic, including when a per-role --megatron-config-path
    is used: parse_megatron_role_args / _apply_megatron_role_overrides only override
    lr/kl/etc. and do NOT touch --save, so a config-driven critic still inherits the
    actor's --save and would clobber the actor's checkpoint. Suffixing is idempotent
    (no-op if --save already ends in ``_critic``), so it is safe to call regardless
    of how critic_args were built.
    """
    save = getattr(critic_args, "save", None)
    if save and not save.rstrip("/").endswith("_critic"):
        critic_args.save = save.rstrip("/") + "_critic"
    # Resume the critic's own value head when its checkpoint sibling exists.
    # On a warm resume the actor's --load points at SAVE_DIR (the actor's
    # LM-head ckpt); the critic must instead load SAVE_DIR_critic, which holds
    # its trained value head. Only redirect when that sibling is actually on
    # disk, so cold start (where only the shared backbone exists, no _critic
    # sibling) still falls through to loading the backbone + reinitializing the
    # value head — guarding the assert-crash described above.
    load = getattr(critic_args, "load", None)
    if load and os.path.isdir(load.rstrip("/") + "_critic"):
        critic_args.load = load.rstrip("/") + "_critic"
        # By DEFAULT a warm critic is a weights-only transfer: skip its
        # optimizer / LR-scheduler / RNG. Those encode the SOURCE run's training
        # progress (e.g. lr_decay_steps = source num_rollout * global_batch),
        # which almost never matches THIS run's schedule. Loading them makes
        # Megatron's OptimizerParamScheduler.load_state_dict assert-crash on the
        # mismatch (observed: "class input value 5632 and checkpoint value 14592
        # for total number of iterations do not match" when warm-loading a
        # 57-rollout critic into a 22-rollout run).
        #
        # ``--critic-load-optim`` opts out, for the case this guard was never
        # meant to cover: resuming a run into ITSELF (same num_rollout, same
        # batch, same lr schedule), where the critic's Adam moments are exactly
        # the state you want to keep -- discarding them resets the value head's
        # optimizer and re-runs the explained-variance warmup from scratch.
        # Safe because the schedule is identical by construction; note that
        # ``--override-opt_param-scheduler`` (which this recipe already passes)
        # makes _check_and_set return early, so the assert above is unreachable
        # anyway when it is set.
        if getattr(critic_args, "critic_load_optim", False):
            critic_args.no_load_optim = False
            critic_args.no_load_rng = False
            critic_args.finetune = False
        else:
            critic_args.no_load_optim = True
            critic_args.no_load_rng = True
            critic_args.finetune = True
    return critic_args


def _assert_actor_critic_checkpoint_dirs_isolated(actor_args, critic_args):
    """Fail before model initialization if actor and critic would clobber each other."""
    actor_save = getattr(actor_args, "save", None)
    critic_save = getattr(critic_args, "save", None)
    if not actor_save or not critic_save:
        return

    actor_dir = os.path.abspath(os.path.expanduser(os.fspath(actor_save)))
    critic_dir = os.path.abspath(os.path.expanduser(os.fspath(critic_save)))
    assert actor_dir != critic_dir, (
        "actor and critic checkpoint directories must differ: " f"actor={actor_save!r}, critic={critic_save!r}"
    )


def create_training_models(args, pgs, rollout_manager):
    actor_args = args
    if args.megatron_config_path is not None:
        from slime.utils.arguments import parse_megatron_role_args

        actor_args = parse_megatron_role_args(args, args.megatron_config_path, role="actor")

    actor_model = allocate_train_group(
        args=actor_args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pgs["actor"],
    )

    critic_model = None
    if args.use_critic:
        from slime.utils.arguments import parse_megatron_role_args

        critic_args = (
            parse_megatron_role_args(args, args.megatron_config_path, role="critic")
            if args.megatron_config_path is not None
            else copy.deepcopy(args)
        )
        if args.megatron_config_path is None:
            critic_args.disable_param_buffers_cpu_backup = False

        # Isolate the critic's --save from the actor's. The critic is a deepcopy
        # of the actor args (or a per-role parse that still inherits --save), so
        # without this it writes to the SAME --save dir. save_checkpoint writes
        # role-agnostic shards into {save}/iter_NNNNNNN/, and train_async saves
        # actor THEN critic, so the critic would OVERWRITE the actor's checkpoint
        # in place (observed: the saved iter dir held only the critic's value head
        # [1, 5120], the actor's LM head was clobbered — a whole multi-day run's
        # actor weights lost this way when --megatron-config-path was used).
        # Suffixing --save with "_critic" keeps each role's checkpoint separate.
        # --load is deliberately NOT suffixed (see _isolate_critic_checkpoint_dirs:
        # on cold start critic loads the shared backbone from the actor's --load).
        # MUST run for BOTH dir-based args AND per-role megatron configs:
        # parse_megatron_role_args does NOT set a role-specific --save, so a
        # config-driven critic inherits the actor's --save just like a deepcopy
        # would. The suffixing is idempotent, so calling it unconditionally is safe.
        _isolate_critic_checkpoint_dirs(critic_args)
        _assert_actor_critic_checkpoint_dirs_isolated(actor_args, critic_args)

        critic_model = allocate_train_group(
            args=critic_args,
            num_nodes=args.critic_num_nodes,
            num_gpus_per_node=args.critic_num_gpus_per_node,
            pg=pgs["critic"],
            role="critic",
        )
        critic_start_rollout_ids = ray.get(critic_model.async_init(critic_model.args, role="critic", with_ref=False))

    actor_start_rollout_ids = ray.get(
        actor_model.async_init(
            actor_args,
            role="actor",
            with_ref=actor_args.kl_coef != 0 or actor_args.use_kl_loss,
            with_opd_teacher=actor_args.use_opd and actor_args.opd_type == "megatron",
        )
    )
    # TODO how to decide rollout start id when critic is involved? For now we just require user to specify it via args.
    if args.use_critic:
        start_rollout_ids = critic_start_rollout_ids
    else:
        start_rollout_ids = actor_start_rollout_ids

    assert len(set(start_rollout_ids)) == 1

    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    actor_model.set_rollout_manager(rollout_manager)
    if args.use_critic:
        critic_model.set_rollout_manager(rollout_manager)

    if args.rollout_global_dataset:
        ray.get(rollout_manager.load.remote(args.start_rollout_id - 1))

    return actor_model, critic_model


def create_rollout_manager(args, pg):
    rollout_manager = RolloutManager.options(
        num_cpus=1,
        num_gpus=0,
    ).remote(args, pg)

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))
        ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))

    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())

    return rollout_manager, num_rollout_per_epoch
