"""SC batching with one gather over packed logits, full CPU heads and explicit CP maps."""

from __future__ import annotations

import torch
import torch.distributed as dist
from megatron.core import mpu

from slime.utils.score_centering import calculate_head_log_probs, score_centering_correction

_APPLIED = False


def score_centering_was_applied():
    return _APPLIED


def reset_score_centering_applied():
    global _APPLIED
    _APPLIED = False


def assert_score_centering_was_applied(args):
    if getattr(args, "enable_score_centering", False) and not _APPLIED:
        raise RuntimeError("SC enabled but the selected policy loss never computed its correction")


def response_row_layout(args, batch, logits):
    from .cp_utils import get_logits_and_tokens_offset_with_cp

    if logits.dtype != torch.float32 or logits.ndim != 3:
        raise ValueError("SC expects float32 [B,T,V/TP] policy logits")
    flat = logits.reshape(-1, logits.size(-1))
    cp = mpu.get_context_parallel_world_size()
    rank = mpu.get_context_parallel_rank()
    if args.qkv_format == "thd" and logits.size(0) != 1:
        raise ValueError("SC THD logits require batch dimension 1")
    if args.allgather_cp and args.qkv_format != "thd":
        raise ValueError("SC allgather-CP requires THD")
    layouts = []
    packed_offset = 0
    global_offset = 0
    for i, (total, response) in enumerate(zip(batch["total_lengths"], batch["response_lengths"], strict=True)):
        if not 0 < response < total:
            raise ValueError("SC requires nonempty prompt and response")
        prompt = total - response
        maximum = batch.get("max_seq_lens")
        maximum = maximum[i] if maximum is not None else None
        if cp == 1:
            base = i * maximum if args.qkv_format == "bshd" else global_offset
            rows = torch.arange(base + prompt - 1, base + total - 1, device=flat.device)
            response_rows = torch.arange(response, device=flat.device)
        elif args.allgather_cp:
            lo = max(global_offset + prompt - 1, rank * flat.size(0))
            hi = min(global_offset + total - 1, (rank + 1) * flat.size(0))
            hi = max(lo, hi)
            rows = torch.arange(lo - rank * flat.size(0), hi - rank * flat.size(0), device=flat.device)
            response_rows = torch.arange(
                lo - (global_offset + prompt - 1), hi - (global_offset + prompt - 1), device=flat.device
            )
        else:
            chunk, chunks, offsets, _ = get_logits_and_tokens_offset_with_cp(total, response, args.qkv_format, maximum)
            row_parts, response_parts = [], []
            for half in (0, 1):
                lo, hi = offsets[half]
                if hi > lo:
                    row_parts.append(
                        torch.arange(
                            packed_offset + half * chunk + lo - chunks[half][0],
                            packed_offset + half * chunk + hi - chunks[half][0],
                            device=flat.device,
                        )
                    )
                    response_parts.append(torch.arange(lo - prompt + 1, hi - prompt + 1, device=flat.device))
            rows = torch.cat(row_parts) if row_parts else torch.empty(0, device=flat.device, dtype=torch.long)
            response_rows = (
                torch.cat(response_parts) if response_parts else torch.empty(0, device=flat.device, dtype=torch.long)
            )
            packed_offset += 2 * chunk
        layouts.append((rows.long(), response_rows.long()))
        global_offset += total
    return flat, layouts


def _gather_current_heads(args, batch, logits):
    flat, layouts = response_row_layout(args, batch, logits)
    candidates, packed_rows, masks, response_ids = [], [], [], []
    for i, ((rows, response_rows), total, response) in enumerate(
        zip(layouts, batch["total_lengths"], batch["response_lengths"], strict=True)
    ):
        cpu_index = response_rows.cpu()
        full_ids = batch["rollout_topk_token_ids"][i]
        if tuple(full_ids.shape) != (response, args.sc_top_k):
            raise ValueError("SC expects full response heads on CPU; no actor-side CP slicing")
        ids = full_ids.index_select(0, cpu_index).to(device=flat.device, dtype=torch.long)
        token_ids = batch["unconcat_tokens"][i][total - response : total].index_select(0, response_rows)
        audit_ids = batch["sc_response_token_ids"][i].index_select(0, cpu_index).to(flat.device)
        if not torch.equal(token_ids, audit_ids.long()):
            raise ValueError("SC token identity mismatch after scheduling or CP mapping")
        masks.append(batch["loss_masks"][i].index_select(0, response_rows).bool())
        response_ids.append(token_ids)
        candidates.append(ids)
        packed_rows.append(rows)
    local_mask = torch.cat(masks)
    ids = torch.cat(candidates)
    packed = torch.cat(packed_rows)
    # Masked tool rows never enter softmax/gather and cannot cause 0 * NaN.
    valid_rows = torch.nonzero(local_mask, as_tuple=False).flatten()
    selected = calculate_head_log_probs(
        flat,
        packed.index_select(0, valid_rows),
        ids.index_select(0, valid_rows),
        mpu.get_tensor_model_parallel_group(),
        args.sc_chunk_size,
    )
    head = selected.new_zeros((len(packed), args.sc_top_k)).index_copy(0, valid_rows, selected)
    return head, layouts, local_mask, ids, torch.cat(response_ids)


def old_policy_head_callback(callback, args, batch):
    def collect(logits):
        result, outputs = callback(logits)
        with torch.no_grad():
            head, layouts, _, _, _ = _gather_current_heads(args, batch, logits)
            lengths = [len(rows) for rows, _ in layouts]
            values = []
            for part, (_, response_rows), response in zip(
                head.split(lengths), layouts, batch["response_lengths"], strict=True
            ):
                full = part.new_zeros((response, args.sc_top_k))
                full.index_copy_(0, response_rows, part)
                if mpu.get_context_parallel_world_size() > 1:
                    dist.all_reduce(full, group=mpu.get_context_parallel_group())
                values.append(full.cpu())
            outputs["sc_old_head_log_probs"] = values
        return result, outputs

    return collect


def _full_advantages(args, batch):
    from .cp_utils import get_logits_and_tokens_offset_with_cp

    values = []
    for i, (adv, total, response) in enumerate(
        zip(batch["advantages"], batch["total_lengths"], batch["response_lengths"], strict=True)
    ):
        if mpu.get_context_parallel_world_size() == 1:
            values.append(adv.detach())
            continue
        maximum = batch.get("max_seq_lens")
        maximum = maximum[i] if maximum is not None else None
        _, _, offsets, _ = get_logits_and_tokens_offset_with_cp(total, response, args.qkv_format, maximum)
        prompt = total - response
        pieces = [torch.arange(lo - prompt + 1, hi - prompt + 1, device=adv.device) for lo, hi in offsets if hi > lo]
        indices = torch.cat(pieces) if pieces else torch.empty(0, device=adv.device, dtype=torch.long)
        full = adv.new_zeros(response)
        full.index_copy_(0, indices.long(), adv.detach())
        dist.all_reduce(full, group=mpu.get_context_parallel_group())
        values.append(full)
    return values


def get_score_centering_terms(args, batch, logits, *, advantages):
    global _APPLIED
    for key in ("rollout_topk_token_ids", "rollout_topk_log_probs", "sc_response_token_ids", "rollout_log_probs"):
        if batch.get(key) is None:
            raise ValueError(f"SC missing training batch field: {key}")
        if len(batch[key]) != len(batch["response_lengths"]):
            raise ValueError(f"SC sample count mismatch: {key}")
    for i, response in enumerate(batch["response_lengths"]):
        if tuple(batch["rollout_topk_log_probs"][i].shape) != (response, args.sc_top_k):
            raise ValueError("SC sampler head shape mismatch")
        if tuple(batch["sc_response_token_ids"][i].shape) != (response,):
            raise ValueError("SC response token identity shape mismatch")
    head, layouts, mask, ids, sampled = _gather_current_heads(args, batch, logits)
    local_q, local_old, local_adv, local_sampled_lp = [], [], [], []
    weighted = args.sc_formula == "ppo_weighted"
    need_old = weighted and args.sc_weight_denominator == "old_actor"
    if need_old and batch.get("sc_old_head_log_probs") is None:
        raise ValueError("Old-policy SC head producer did not run")
    if need_old:
        if len(batch["sc_old_head_log_probs"]) != len(layouts):
            raise ValueError("Old-policy SC sample count mismatch")
        for old_head, response in zip(batch["sc_old_head_log_probs"], batch["response_lengths"], strict=True):
            if tuple(old_head.shape) != (response, args.sc_top_k):
                raise ValueError("Old-policy SC head shape mismatch")
    full_adv = _full_advantages(args, batch) if args.allgather_cp else None
    for i, (_, response_rows) in enumerate(layouts):
        cpu_index = response_rows.cpu()
        local_q.append(batch["rollout_topk_log_probs"][i].index_select(0, cpu_index).to(logits.device))
        if need_old:
            local_old.append(batch["sc_old_head_log_probs"][i].index_select(0, cpu_index).to(logits.device))
        if full_adv is not None:
            local_adv.append(full_adv[i].index_select(0, response_rows))
        else:
            local_adv.append(batch["advantages"][i])
        # rollout_log_probs is zigzag on the actor; reconstruct only for allgather-CP.
        if args.allgather_cp:
            continue
        local_sampled_lp.append(batch["rollout_log_probs"][i])
    if args.allgather_cp:
        temporary = dict(batch, advantages=batch["rollout_log_probs"])
        all_lp = _full_advantages(args, temporary)
        local_sampled_lp = [value.index_select(0, index) for value, (_, index) in zip(all_lp, layouts, strict=True)]
    q = torch.cat(local_q)
    adv = torch.cat(local_adv).detach()
    adv = torch.where(mask, adv, torch.zeros_like(adv))
    if not torch.isfinite(adv).all():
        raise ValueError("SC requires finite advantages on unmasked tokens")
    logged = torch.cat(local_sampled_lp)
    if len(adv) != len(head) or len(logged) != len(head):
        raise ValueError("SC advantages/sampled-logprobs do not match local response rows")
    hit_matrix = ids == sampled[:, None]
    hit = hit_matrix.any(-1) & mask
    first = hit_matrix.long().argmax(-1, keepdim=True)
    if not torch.isfinite(logged[mask]).all():
        raise ValueError("SC requires finite sampled-token logprobs")
    deviation = (q.gather(-1, first).squeeze(-1) - logged).abs()
    if hit.any() and (deviation[hit] > 1e-4).any():
        raise ValueError("SC sampled-token logprob differs from transported head")
    correction, metrics = score_centering_correction(
        train_head_logp=head,
        sampler_head_logp=q,
        advantages=adv,
        formula=args.sc_formula,
        old_head_logp=torch.cat(local_old) if need_old else None,
        eps_clip_low=args.eps_clip,
        eps_clip_high=args.eps_clip_high,
        tail_eps=args.sc_tail_eps,
        loss_mask=mask,
    )
    metrics["sc_head_hit"] = hit.float().detach()
    metrics["sc_head_logp_deviation"] = torch.where(hit, deviation, torch.zeros_like(deviation)).detach()
    metrics["sc_correction"] = correction.detach()
    # Multiply advantages before redistribution because weighted SC uses their signs.
    scaled = adv * correction
    lengths = [len(rows) for rows, _ in layouts]
    values = {"sc_addend": list(scaled.split(lengths))}
    values.update({key: list(value.split(lengths)) for key, value in metrics.items()})
    if args.allgather_cp and mpu.get_context_parallel_world_size() > 1:
        from .loss import _allgather_cp_redistribute

        _allgather_cp_redistribute(
            values,
            logits_local_len=logits.size(1),
            args=args,
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            max_seq_lens=batch.get("max_seq_lens"),
        )
    addend = torch.cat(values.pop("sc_addend"))
    if addend.shape != advantages.shape:
        raise ValueError("SC final correction is not in the PPO advantage layout")
    _APPLIED = True
    return addend, {key: torch.cat(value).detach() for key, value in values.items()}
