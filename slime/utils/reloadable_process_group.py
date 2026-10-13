import logging
import os
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist

from slime.utils.memory_utils import available_memory, clear_memory, print_memory

logger = logging.getLogger(__name__)

old_new_group_dict = {}


def _default_nccl_group_timeout():
    """Default timeout for NCCL process groups created/recreated by slime.

    torch's ``new_group`` otherwise falls back to ``default_pg_timeout`` (10 min).
    On every offload reload (see ``reload_process_groups``) the model-parallel
    groups are recreated, and a transient rank stall during weight sync then
    trips that 10-min watchdog and tears down the whole job. Megatron passes an
    explicit timeout on INITIAL creation (``--distributed-timeout-minutes``),
    which we preserve by only filling this default in when the caller omits one.
    Tunable via ``SLIME_NCCL_GROUP_TIMEOUT_MINUTES`` (default 30).
    """
    return timedelta(minutes=int(os.environ.get("SLIME_NCCL_GROUP_TIMEOUT_MINUTES", "30")))

_COMM_MEMORY_CHECK_SKIP_OPS = {
    "all_gather_into_tensor",
    "allgather_into_tensor_coalesced",
    "barrier",
    "broadcast_object_list",
    "reduce_scatter_tensor",
    "all_to_all_single",
    "isend",
    "irecv",
}


def _should_check_memory_for_comm(op_name):
    return op_name not in _COMM_MEMORY_CHECK_SKIP_OPS


def monkey_patch_torch_dist():
    pid = os.getpid()
    if pid in old_new_group_dict:
        assert dist.old_new_group == old_new_group_dict[pid]
        return

    logger.info("Applying monkey patch to torch.distributed")

    old_new_group = dist.new_group
    old_new_group_dict[pid] = old_new_group
    dist.old_new_group = old_new_group

    def new_group(*args, **kwargs):
        # Fill in a generous NCCL group timeout when the caller omits one, so
        # recreated groups don't silently inherit torch's 10-min default. Megatron
        # passes its own timeout (kwarg or positionally as the 2nd arg), which we
        # leave untouched. Signature: new_group(ranks, timeout, backend, ...).
        is_gloo = kwargs.get("backend") == "gloo" or (len(args) >= 3 and args[2] == "gloo")
        timeout_given = "timeout" in kwargs or len(args) >= 2
        if not is_gloo and not timeout_given:
            kwargs["timeout"] = _default_nccl_group_timeout()
        group = old_new_group(*args, **kwargs)
        # skip none nccl group.
        if len(args) >= 3 and args[2] == "gloo" or "backend" in kwargs and kwargs["backend"] == "gloo":
            return group

        # Get ranks from arguments
        if len(args) >= 1 and args[0] is not None:
            ranks = args[0]
        elif "ranks" in kwargs and kwargs["ranks"] is not None:
            ranks = kwargs["ranks"]
        else:
            # If no ranks specified, use all ranks in world
            ranks = list(range(dist.get_world_size()))

        if len(ranks) == 1:
            return group

        group = ReloadableProcessGroup(group, ranks)
        return group

    dist.new_group = new_group

    def get_new_query_function(func):
        """Wrap query functions (get_rank, get_world_size, etc.) without memory check."""

        def new_function(*args, **kwargs):
            args = tuple([arg.group if isinstance(arg, ReloadableProcessGroup) else arg for arg in args])
            kwargs = {k: (v.group if isinstance(v, ReloadableProcessGroup) else v) for k, v in kwargs.items()}
            return func(*args, **kwargs)

        return new_function

    def get_new_comm_function(func, op_name=None):
        """Wrap communication functions with memory check."""

        def new_function(*args, **kwargs):
            args = tuple([arg.group if isinstance(arg, ReloadableProcessGroup) else arg for arg in args])
            kwargs = {k: (v.group if isinstance(v, ReloadableProcessGroup) else v) for k, v in kwargs.items()}
            check_memory = True if op_name is None else _should_check_memory_for_comm(op_name)
            with _wrap_low_level_call(check_memory=check_memory):
                return func(*args, **kwargs)

        return new_function

    dist.get_rank = get_new_query_function(dist.get_rank)
    dist.get_world_size = get_new_query_function(dist.get_world_size)
    dist.get_backend = get_new_query_function(dist.get_backend)
    dist.get_global_rank = get_new_query_function(dist.get_global_rank)
    dist.get_group_rank = get_new_query_function(dist.get_group_rank)
    dist.get_process_group_ranks = get_new_query_function(dist.get_process_group_ranks)

    dist.all_reduce = get_new_comm_function(dist.all_reduce)
    dist.all_gather = get_new_comm_function(dist.all_gather)
    dist.all_gather_into_tensor = get_new_comm_function(dist.all_gather_into_tensor, "all_gather_into_tensor")
    dist.all_gather_object = get_new_comm_function(dist.all_gather_object)
    dist.all_to_all = get_new_comm_function(dist.all_to_all)
    dist.all_to_all_single = get_new_comm_function(dist.all_to_all_single, "all_to_all_single")
    dist.broadcast = get_new_comm_function(dist.broadcast)
    dist.broadcast_object_list = get_new_comm_function(dist.broadcast_object_list, "broadcast_object_list")
    dist.reduce = get_new_comm_function(dist.reduce)
    dist.reduce_scatter = get_new_comm_function(dist.reduce_scatter)
    dist.reduce_scatter_tensor = get_new_comm_function(dist.reduce_scatter_tensor, "reduce_scatter_tensor")
    dist.scatter = get_new_comm_function(dist.scatter)
    dist.gather = get_new_comm_function(dist.gather)
    dist.barrier = get_new_comm_function(dist.barrier, "barrier")
    dist.send = get_new_comm_function(dist.send)
    dist.recv = get_new_comm_function(dist.recv)
    dist._coalescing_manager = get_new_comm_function(dist._coalescing_manager)

    # p2p
    old_isend = dist.isend
    old_irecv = dist.irecv

    dist.isend = get_new_comm_function(dist.isend, "isend")
    dist.irecv = get_new_comm_function(dist.irecv, "irecv")

    def get_new_p2pop_function(func):
        def new_function(*args, **kwargs):
            def convert(arg):
                if isinstance(arg, ReloadableProcessGroup):
                    return arg.group
                elif arg == dist.isend:
                    arg = old_isend
                elif arg == dist.irecv:
                    arg = old_irecv
                return arg

            args = (convert(arg) for arg in args)
            kwargs = {k: convert(v) for k, v in kwargs.items()}
            return func(*args, **kwargs)

        return new_function

    dist.P2POp.__new__ = get_new_p2pop_function(dist.P2POp.__new__)
    dist.P2POp.__init__ = get_new_p2pop_function(dist.P2POp.__init__)


class ReloadableProcessGroup(torch.distributed.ProcessGroup):
    GROUPS = {}

    def __init__(self, group, ranks):
        super().__init__(
            rank=dist.get_rank(group),
            size=dist.get_world_size(group),
        )
        self.group = group
        self.group_info = {
            "ranks": ranks,
        }
        pid = os.getpid()
        if pid not in ReloadableProcessGroup.GROUPS:
            ReloadableProcessGroup.GROUPS[pid] = []
        ReloadableProcessGroup.GROUPS[pid].append(self)

    def __getattr__(self, name):
        return getattr(self.group, name)

    @staticmethod
    def destroy_process_groups():
        pid = os.getpid()
        for reloadable_group in ReloadableProcessGroup.GROUPS.get(pid, []):
            if reloadable_group.group is None:
                continue
            try:
                dist.destroy_process_group(reloadable_group.group)
            except ValueError as e:
                logger.warning(
                    f"Process group already invalid/destroyed; skipping cleanup. Exception: {e}",
                    exc_info=True,
                )

            del reloadable_group.group
            reloadable_group.group = None

    @staticmethod
    def reload_process_groups():
        pid = os.getpid()
        reloadable_groups = ReloadableProcessGroup.GROUPS.get(pid, [])
        logger.info(f"Reloading {len(reloadable_groups)} process groups in pid {pid}")
        old_new_group = old_new_group_dict.get(pid)
        for reloadable_group in reloadable_groups:
            if reloadable_group.group is not None:
                continue
            group = old_new_group(
                ranks=reloadable_group.group_info["ranks"],
                backend="nccl",
                timeout=_default_nccl_group_timeout(),
            )
            reloadable_group.group = group

        # NOTE (2026-08-26): a post-reload `dist.barrier()` on each rebuilt group was tried
        # here and REMOVED because it DEADLOCKS the run. Do not reinstate it.
        #
        # The reasoning that made it look safe was wrong. I argued "every member of a group
        # rebuilds it in the same reload cycle, so these barriers cannot deadlock". But under
        # PPO the ACTOR and the CRITIC are separate Ray actor processes that SHARE the same
        # placement group and GPUs (placement_group.py:135 `result["critic"] = result["actor"]`),
        # and each calls reload_process_groups() from its OWN wake_up() at its OWN time
        # (actor.py:217) -- the critic wakes to train first, the actor later. So a barrier here
        # blocks whichever side reloads first while its peer has not reached reload yet, and
        # nothing ever releases it.
        #
        # Observed 2026-08-26 at rollout 50, after ~17 h wedged: every training rank sat in
        #     barrier (distributed_c10d.py:5030)
        #     new_function (reloadable_process_group.py)
        #     reload_process_groups (reloadable_process_group.py)
        # with the driver blocked in ray.get at train_async.py:78. Idle nodes showed 0% GPU
        # while others showed 100% (a NCCL spin-wait), and there were no worker deaths and no
        # errors -- so nothing surfaced it. It is a RACE on actor-vs-critic wake ordering,
        # which is why several earlier 5-step runs passed before it bit.
        #
        # It also never earned its keep: across every run it reported ZERO real failures, so
        # removing it loses no diagnostic coverage. new_group() below is itself a collective,
        # so a genuinely broken rebuild still fails there.

    def rank(self) -> int:
        return self.group.rank()

    def size(self) -> int:
        return self.group.size()

    def name(self) -> str:
        return self.group.name()

    def shutdown(self) -> None:
        if self.group is not None:
            self.group.shutdown()

    def abort(self) -> None:
        if self.group is not None:
            self.group.abort()

    def _fwd(self, method, *args, **kwargs):
        inner = self.group
        if inner is None:
            raise RuntimeError("ReloadableProcessGroup: inner PG is None, call reload() first.")
        with _wrap_low_level_call(check_memory=_should_check_memory_for_comm(method)):
            return getattr(inner, method)(*args, **kwargs)

    def _fwd_query(self, method, *args, **kwargs):
        """Forward non-communication calls without memory check."""
        inner = self.group
        if inner is None:
            raise RuntimeError("ReloadableProcessGroup: inner PG is None, call reload() first.")
        return getattr(inner, method)(*args, **kwargs)

    def barrier(self, *a, **kw):
        return self._fwd("barrier", *a, **kw)

    def broadcast(self, *a, **kw):
        return self._fwd("broadcast", *a, **kw)

    def allreduce(self, *a, **kw):
        return self._fwd("allreduce", *a, **kw)

    def allreduce_coalesced(self, *a, **kw):
        return self._fwd("allreduce_coalesced", *a, **kw)

    def reduce(self, *a, **kw):
        return self._fwd("reduce", *a, **kw)

    def allgather(self, *a, **kw):
        return self._fwd("allgather", *a, **kw)

    def _allgather_base(self, *a, **kw):
        return self._fwd("_allgather_base", *a, **kw)

    def allgather_coalesced(self, *a, **kw):
        return self._fwd("allgather_coalesced", *a, **kw)

    def allgather_into_tensor_coalesced(self, *a, **kw):
        return self._fwd("allgather_into_tensor_coalesced", *a, **kw)

    def gather(self, *a, **kw):
        return self._fwd("gather", *a, **kw)

    def scatter(self, *a, **kw):
        return self._fwd("scatter", *a, **kw)

    def reduce_scatter(self, *a, **kw):
        return self._fwd("reduce_scatter", *a, **kw)

    def _reduce_scatter_base(self, *a, **kw):
        return self._fwd("_reduce_scatter_base", *a, **kw)

    def reduce_scatter_tensor_coalesced(self, *a, **kw):
        return self._fwd("reduce_scatter_tensor_coalesced", *a, **kw)

    def alltoall_base(self, *a, **kw):
        return self._fwd("alltoall_base", *a, **kw)

    def alltoall(self, *a, **kw):
        return self._fwd("alltoall", *a, **kw)

    def send(self, *a, **kw):
        return self._fwd("send", *a, **kw)

    def recv(self, *a, **kw):
        return self._fwd("recv", *a, **kw)

    def recv_anysource(self, *a, **kw):
        return self._fwd("recv_anysource", *a, **kw)

    def _start_coalescing(self, *a, **kw):
        return self._fwd_query("_start_coalescing", *a, **kw)

    def _end_coalescing(self, *a, **kw):
        return self._fwd("_end_coalescing", *a, **kw)

    def _get_backend_name(self):
        return self._fwd_query("_get_backend_name")

    def _get_backend(self, *a, **kw):
        return self._fwd_query("_get_backend", *a, **kw)

    def _set_default_backend(self, *a, **kw):
        return self._fwd_query("_set_default_backend", *a, **kw)

    @property
    def bound_device_id(self):
        return self.group.bound_device_id

    @bound_device_id.setter
    def bound_device_id(self, dev):
        self.group.bound_device_id = dev


def destroy_process_groups():
    """Destroy all reloadable process groups."""
    ReloadableProcessGroup.destroy_process_groups()


def reload_process_groups():
    """Reload all reloadable process groups."""
    ReloadableProcessGroup.reload_process_groups()


@contextmanager
def _wrap_low_level_call(check_memory=True):
    try:
        if check_memory:
            mem_info = available_memory()
            if mem_info["free_GB"] < 3:
                clear_memory()
        yield
    except Exception as e:
        mem_info = print_memory("after torch distributed error")
        e.add_note(f"{mem_info=}")
        raise
