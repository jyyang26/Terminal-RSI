"""Opt-in successor-state critic forward before any critic optimizer update."""

import copy

import torch
import torch.distributed as dist


def make_bootstrap_batch(rollout_data, *, device, max_tokens, padded_count=None):
    """One probe per local sample, same schedule on every TP/CP/PP rank.

    A dummy token after the successor prefix lets get_values select the last
    prefix logit (the state before the next action). The dummy is never trained.
    Non-bootstrap samples have a tiny zero-loss probe for equal PP schedules.
    """
    tokens, masks, lengths = [], [], []
    required = []
    if max_tokens < 2:
        raise ValueError("TB-GAE bootstrap context must allow prefix + probe")
    for boundary in rollout_data["tb_boundary"]:
        needs_value = not boundary["terminal"] and boundary["chunk_index"] == boundary["chunk_count"] - 1
        prefix = boundary.get("bootstrap_token_ids") if needs_value else [0]
        if not isinstance(prefix, (list, tuple)) or not prefix or any(type(t) is not int or t < 0 for t in prefix):
            raise ValueError("TB-GAE needs a valid successor token prefix")
        # Same explicit left-context window policy as capped training chunks;
        # preserve the complete latest observation / generation boundary tail.
        prefix = prefix[-(max_tokens - 1) :]
        tokens.append(torch.tensor([*prefix, 0], dtype=torch.long, device=device))
        masks.append(torch.ones(1, dtype=torch.int, device=device))
        lengths.append(len(prefix) + 1)
        required.append(needs_value)
    padded_count = len(tokens) if padded_count is None else padded_count
    if padded_count < len(tokens):
        raise ValueError("TB-GAE bootstrap schedule cannot drop probes")
    while len(tokens) < padded_count:
        tokens.append(torch.tensor([0, 0], dtype=torch.long, device=device))
        masks.append(torch.ones(1, dtype=torch.int, device=device))
        lengths.append(2)
    batch = dict(
        tokens=tokens,
        loss_masks=masks,
        total_lengths=lengths,
        response_lengths=[1] * len(tokens),
        micro_batch_indices=[[i] for i in range(len(tokens))],
        num_microbatches=[len(tokens)],
        global_batch_sizes=[len(tokens)],
    )
    return batch, required


@torch.no_grad()
def compute_bootstrap_values(args, model, rollout_data, *, forward_fn=None):
    from megatron.core import mpu
    from megatron.core.utils import get_model_config

    from slime.backends.megatron_utils.cp_utils import all_gather_with_cp
    from slime.backends.megatron_utils.data import get_data_iterator
    from slime.backends.megatron_utils.loss import get_values
    from slime.backends.megatron_utils.model import forward_only
    from slime.utils.distributed_utils import get_gloo_group

    if getattr(args, "enable_tb_gae", False) is not True:
        raise ValueError("Bootstrap forward requires explicit TB-GAE opt-in")
    # All model-parallel ranks run the same probes, even if their CP response is empty.
    # One probe per microbatch avoids packing many long successor states into
    # a bin sized using the (possibly short) original response. Equal counts
    # across DP and PP keep forward collectives/schedules aligned.
    count = torch.tensor(len(rollout_data["tb_boundary"]), dtype=torch.long)
    dist.all_reduce(count, op=dist.ReduceOp.MAX, group=get_gloo_group())
    quantum = getattr(get_model_config(model[0]), "microbatch_group_size_per_vp_stage", 1) or 1
    padded_count = ((int(count) + quantum - 1) // quantum) * quantum
    batch, required = make_bootstrap_batch(
        rollout_data,
        device=rollout_data["tokens"][0].device,
        max_tokens=args.seq_length,
        padded_count=padded_count,
    )
    # Never skip independently per DP rank: DDP/PP parameter synchronization
    # may still run during forward. Every replica uses the original schedule.
    local_args = copy.copy(args)
    local_args.use_rollout_entropy = False
    # Preserve rollout sample order for fixed and dynamic microbatch schedules.
    local_args.use_dynamic_batch_size = True
    local_args.micro_batch_size = 1
    result = (forward_fn or forward_only)(
        get_values,
        local_args,
        model,
        get_data_iterator(batch),
        batch["num_microbatches"],
    )
    if not mpu.is_pipeline_last_stage():
        return None
    values = []
    for i, (total, local) in enumerate(zip(batch["total_lengths"], result["values"], strict=True)):
        full = all_gather_with_cp(local.detach(), total, 1).detach()
        if full.numel() != 1 or not torch.isfinite(full).all():
            raise ValueError("TB-GAE successor critic forward returned an invalid value")
        if i < len(required):
            values.append(float(full.item()) if required[i] else 0.0)
    return values
