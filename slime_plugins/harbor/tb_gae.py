"""Opt-in turn-based GAE with ordinary THD context parallelism.

Masks use full response coordinates; values and token rewards use the existing
shifted-logit CP slices. Targets never participate in autograd.
"""

import math

import torch


def _discount(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= 1
    ):
        raise ValueError(f"{name} must be finite and in (0, 1]")


@torch.no_grad()
def _reverse_affine_scan(delta, weight):
    """A[j] = delta[j] + weight[j]*A[j+1], using O(log N) tensor passes.

    Unlike dividing cumulative products, this handles zero terminal weights
    without division/underflow and has no per-token host synchronization.
    """
    result = delta.clone()
    product = weight.clone()
    offset = 1
    while offset < delta.numel():
        result = torch.cat((result[:-offset] + product[:-offset] * result[offset:], result[-offset:]))
        product = torch.cat((product[:-offset] * product[offset:], product[-offset:]))
        offset *= 2
    return result


@torch.no_grad()
def compute_tb_gae(
    values,
    rewards,
    action_mask,
    turn_end_mask,
    is_real_done,
    gamma_turn=1.0,
    lambda_turn=1.0,
    *,
    bootstrap_values=None,
    continuation_advantages=None,
):
    """Dense [B,T] reference contract, with a parallel scan over active tokens.

    The last active token must terminate, or explicit bootstrap and continuation
    boundary values must be supplied. Non-action rewards are rejected.
    """
    _discount(gamma_turn, "gamma_turn")
    _discount(lambda_turn, "lambda_turn")
    if values.ndim != 2 or not values.is_floating_point():
        raise ValueError("values must be a floating [B,T] tensor")
    for name, tensor in (
        ("rewards", rewards),
        ("action_mask", action_mask),
        ("turn_end_mask", turn_end_mask),
        ("is_real_done", is_real_done),
    ):
        if tensor.shape != values.shape or tensor.device != values.device:
            raise ValueError(f"{name} shape/device must match values")
    for name, mask in (("action_mask", action_mask), ("turn_end_mask", turn_end_mask), ("is_real_done", is_real_done)):
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{name} must contain boolean/0/1 values")
    action = action_mask.bool()
    te_full, dn_full = turn_end_mask.bool(), is_real_done.bool()
    if torch.any(te_full & ~action) or torch.any(dn_full & ~te_full):
        raise ValueError("is_real_done <= turn_end_mask <= action_mask is required")
    if torch.any(rewards[~action] != 0):
        raise ValueError("TB-GAE rewards must be placed on action tokens")
    if not torch.isfinite(values).all() or not torch.isfinite(rewards).all():
        raise ValueError("TB-GAE values/rewards must be finite")
    dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    batch = values.shape[0]
    for name, boundary in (
        ("bootstrap_values", bootstrap_values),
        ("continuation_advantages", continuation_advantages),
    ):
        if boundary is not None:
            if boundary.shape != (batch,) or boundary.device != values.device or not torch.isfinite(boundary).all():
                raise ValueError(f"{name} must be finite [B] on the values device")
    advantages, returns = torch.zeros_like(values), torch.zeros_like(values)
    for b in range(batch):
        indices = torch.nonzero(action[b], as_tuple=True)[0]
        if indices.numel() == 0:
            continue
        v = values[b, indices].to(dtype)
        r = rewards[b, indices].to(dtype)
        te = te_full[b, indices]
        done = dn_full[b, indices]
        terminal = bool(done[-1])
        if not terminal and (bootstrap_values is None or continuation_advantages is None):
            raise ValueError("Nonterminal TB-GAE tail requires explicit bootstrap and continuation")
        next_v = torch.cat((v[1:], torch.zeros_like(v[:1])))
        if not terminal:
            next_v[-1] = bootstrap_values[b]
        crossing = te.clone()
        crossing[-1] = not terminal and bool(te[-1])
        g = torch.where(crossing, torch.full_like(v, gamma_turn), torch.ones_like(v))
        lam = torch.where(crossing, torch.full_like(v, lambda_turn), torch.ones_like(v))
        nonterm = (~done).to(dtype)
        delta = r + g * nonterm * next_v - v
        weight = g * lam * nonterm
        if not terminal:
            delta[-1] += weight[-1] * continuation_advantages[b]
        adv = _reverse_affine_scan(delta, weight)
        advantages[b, indices] = adv.to(values.dtype)
        returns[b, indices] = (adv + v).to(values.dtype)
    return advantages.detach(), returns.detach()


def _validate_layout(qkv_format, allgather_cp, cp_size):
    if qkv_format != "thd" or allgather_cp is not False or cp_size not in (1, 4):
        raise ValueError("TB-GAE supports THD ordinary zigzag CP=1/4 only (allgather_cp=False)")


@torch.no_grad()
def compute_tb_gae_cp(
    values_shards,
    token_reward_shards,
    action_masks_full,
    turn_end_masks_full,
    is_real_done_full,
    *,
    total_lengths,
    response_lengths,
    qkv_format="thd",
    max_seq_lens=None,
    allgather_cp=False,
    gamma_turn=1.0,
    lambda_turn=1.0,
):
    """Ragged local tensors + full masks -> ragged local detached targets."""
    from megatron.core import mpu
    from slime.backends.megatron_utils.cp_utils import all_gather_with_cp, slice_log_prob_with_cp

    _validate_layout(qkv_format, allgather_cp, mpu.get_context_parallel_world_size())
    batch = len(values_shards)
    fields = (
        token_reward_shards,
        action_masks_full,
        turn_end_masks_full,
        is_real_done_full,
        total_lengths,
        response_lengths,
    )
    if any(len(field) != batch for field in fields):
        raise ValueError("TB-GAE batch field counts differ")
    if max_seq_lens is not None and len(max_seq_lens) != batch:
        raise ValueError("TB-GAE max_seq_lens count differs")
    advantages, returns = [], []
    for i in range(batch):
        total, response = total_lengths[i], response_lengths[i]
        if not 0 < response < total:
            raise ValueError("TB-GAE requires a nonempty prompt and response")
        local_v, local_r = values_shards[i], token_reward_shards[i]
        if local_v.ndim != 1 or local_r.shape != local_v.shape or local_r.device != local_v.device:
            raise ValueError("TB-GAE local values/rewards shape/device mismatch")
        masks = [
            torch.as_tensor(field[i], device=local_v.device)
            for field in (action_masks_full, turn_end_masks_full, is_real_done_full)
        ]
        if any(mask.shape != (response,) for mask in masks):
            raise ValueError("TB-GAE masks must use full response coordinates")
        maximum = max_seq_lens[i] if max_seq_lens is not None else None
        expected = slice_log_prob_with_cp(masks[0], total, response, qkv_format, maximum)
        if local_v.numel() != expected.numel():
            raise ValueError("TB-GAE values do not match the native CP response slice")
        # All CP ranks must participate, including ranks with no local response.
        full_v = all_gather_with_cp(local_v.detach(), total, response).detach()
        full_r = all_gather_with_cp(local_r.detach(), total, response).detach()
        if full_v.shape != (response,) or full_r.shape != (response,):
            raise ValueError("TB-GAE CP gather returned an incorrect response length")
        adv, ret = compute_tb_gae(
            full_v[None],
            full_r[None],
            *(mask[None] for mask in masks),
            gamma_turn=gamma_turn,
            lambda_turn=lambda_turn,
        )
        a = slice_log_prob_with_cp(adv[0], total, response, qkv_format, maximum)
        r = slice_log_prob_with_cp(ret[0], total, response, qkv_format, maximum)
        if a.shape != local_v.shape or r.shape != local_v.shape:
            raise ValueError("TB-GAE CP target shape differs from local values")
        advantages.append(a.detach())
        returns.append(r.detach())
    return advantages, returns


@torch.no_grad()
def tb_gae_advantage(args, rollout_data):
    """Shared actor/critic entry. A generic custom hook cannot bypass opt-in."""
    if getattr(args, "enable_tb_gae", False) is not True:
        raise ValueError("TB-GAE requires explicit enable_tb_gae=True")
    from megatron.core import mpu
    from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp
    from slime.utils.tb_gae_config import validate_tb_gae_config
    from slime_plugins.harbor.tb_gae_data import validate_tb_masks

    validate_tb_gae_config(args)
    _validate_layout(args.qkv_format, args.allgather_cp, mpu.get_context_parallel_world_size())
    if "tb_boundary" in rollout_data:
        _episode_tb_gae(args, rollout_data)
        return
    values = rollout_data.get("values")
    if values is None:
        raise ValueError("TB-GAE requires critic values on the last pipeline stage")
    for key in ("turn_end_mask", "is_real_done"):
        if rollout_data.get(key) is None:
            raise RuntimeError(f"TB-GAE needs per-token turn_end_mask and is_real_done; missing {key}")
    batch = len(values)
    for key in ("rewards", "loss_masks", "turn_end_mask", "is_real_done", "total_lengths", "response_lengths"):
        if len(rollout_data[key]) != batch:
            raise ValueError(f"TB-GAE batch count mismatch: {key}")
    local_rewards, action_masks, turn_masks, done_masks = [], [], [], []
    for i, local_v in enumerate(values):
        action = rollout_data["loss_masks"][i]
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        te, dn = validate_tb_masks(
            rollout_data["turn_end_mask"][i],
            rollout_data["is_real_done"][i],
            action,
        )
        action = torch.as_tensor(action, dtype=torch.bool)
        response, total = rollout_data["response_lengths"][i], rollout_data["total_lengths"][i]
        if action.shape != (response,):
            raise ValueError("TB-GAE loss_mask must use full response coordinates")
        outcome = rollout_data["rewards"][i]
        if torch.is_tensor(outcome):
            if outcome.numel() != 1:
                raise ValueError("TB-GAE requires a scalar outcome reward per sample")
            outcome = outcome.item()
        if not isinstance(outcome, (int, float)) or not math.isfinite(outcome):
            raise ValueError("TB-GAE requires a finite scalar outcome reward")
        full_reward = torch.zeros(response, dtype=local_v.dtype, device=local_v.device)
        active = torch.nonzero(action, as_tuple=True)[0]
        if active.numel():
            full_reward[int(active[-1])] = outcome
        maximum = rollout_data.get("max_seq_lens")
        local_rewards.append(
            slice_log_prob_with_cp(
                full_reward,
                total,
                response,
                args.qkv_format,
                maximum[i] if maximum else None,
            )
        )
        action_masks.append(action)
        turn_masks.append(te)
        done_masks.append(dn)
    advantages, returns = compute_tb_gae_cp(
        values,
        local_rewards,
        action_masks,
        turn_masks,
        done_masks,
        total_lengths=rollout_data["total_lengths"],
        response_lengths=rollout_data["response_lengths"],
        qkv_format=args.qkv_format,
        max_seq_lens=rollout_data.get("max_seq_lens"),
        allgather_cp=args.allgather_cp,
        gamma_turn=args.tb_gae_gamma_turn,
        lambda_turn=args.tb_gae_lambda_turn,
    )
    rollout_data["advantages"], rollout_data["returns"] = advantages, returns


@torch.no_grad()
def _episode_tb_gae(args, data):
    from megatron.core import mpu
    from slime.backends.megatron_utils.cp_utils import all_gather_with_cp, slice_log_prob_with_cp
    from slime_plugins.harbor.tb_gae_boundary import gather_and_solve_boundaries
    from slime_plugins.harbor.tb_gae_data import validate_tb_masks

    count = len(data["values"])
    for key in (
        "tb_boundary",
        "loss_masks",
        "turn_end_mask",
        "is_real_done",
        "total_lengths",
        "response_lengths",
        "rewards",
    ):
        if len(data[key]) != count:
            raise ValueError(f"TB-GAE sample count mismatch: {key}")
    bootstrap = data.get("tb_bootstrap_values")
    if bootstrap is not None and len(bootstrap) != count:
        raise ValueError("TB-GAE bootstrap count mismatch")
    gamma, lam = args.tb_gae_gamma_turn, args.tb_gae_lambda_turn
    full_rows, summaries = [], []
    for i, (local, boundary) in enumerate(zip(data["values"], data["tb_boundary"], strict=True)):
        total, response = data["total_lengths"][i], data["response_lengths"][i]
        action = data["loss_masks"][i]
        action_cpu = action.detach().cpu().numpy() if torch.is_tensor(action) else action
        te, done = validate_tb_masks(
            data["turn_end_mask"][i], data["is_real_done"][i], action_cpu, require_terminal=boundary["terminal"]
        )
        masks = [torch.as_tensor(m, device=local.device, dtype=torch.bool) for m in (action_cpu, te, done)]
        if any(m.shape != (response,) for m in masks):
            raise ValueError("TB-GAE full masks are not response aligned")
        if local.shape != slice_log_prob_with_cp(masks[0], total, response).shape:
            raise ValueError("TB-GAE values are not native CP slices")
        full = all_gather_with_cp(local.detach(), total, response).detach()
        reward = torch.zeros_like(full)
        idx = torch.nonzero(masks[0], as_tuple=True)[0]
        outcome = float(data["rewards"][i])
        last = boundary["chunk_index"] == boundary["chunk_count"] - 1
        if len(idx) and last:
            reward[idx[-1]] = outcome
        zero = full.new_zeros(1)
        adv, _ = compute_tb_gae(
            full[None],
            reward[None],
            *(m[None] for m in masks),
            gamma,
            lam,
            bootstrap_values=zero,
            continuation_advantages=zero,
        )
        if len(idx):
            # Match the recurrence's boundary discounts, including mid-turn splits.
            turns = masks[1][idx]
            g = torch.where(turns, torch.full_like(full[idx], gamma), torch.ones_like(full[idx]))
            lambdas = torch.where(turns, torch.full_like(full[idx], lam), torch.ones_like(full[idx]))
            if boundary["terminal"]:
                g[-1] = 0.0
            w = g * lambdas
            cv = float((torch.prod(w[:-1]) * g[-1]).item())
            ca = float(torch.prod(w).item())
            base, first = float(adv[0, idx[0]].item()), float(full[idx[0]].item())
        else:
            cv = ca = base = first = 0.0
        summaries.append(
            {
                "episode": boundary["episode"],
                "chunk_index": boundary["chunk_index"],
                "chunk_count": boundary["chunk_count"],
                "terminal": boundary["terminal"],
                "active": bool(len(idx)),
                "outcome": outcome,
                "base": base,
                "first_value": first,
                "value_coefficient": cv,
                "advantage_coefficient": ca,
                "bootstrap": bootstrap[i] if bootstrap is not None else None,
            }
        )
        full_rows.append((full, reward, masks))
    # Each CP/TP lane exchanges the same small summaries across the DP replicas.
    group = mpu.get_data_parallel_group_gloo(with_context_parallel=False)
    continuations = gather_and_solve_boundaries(summaries, group)
    advantages, returns = [], []
    for i, ((full, reward, masks), boundary) in enumerate(zip(full_rows, data["tb_boundary"], strict=True)):
        next_v, next_a = continuations[(boundary["episode"], boundary["chunk_index"])]
        a, r = compute_tb_gae(
            full[None],
            reward[None],
            *(m[None] for m in masks),
            gamma,
            lam,
            bootstrap_values=full.new_tensor([next_v]),
            continuation_advantages=full.new_tensor([next_a]),
        )
        total, response = data["total_lengths"][i], data["response_lengths"][i]
        advantages.append(slice_log_prob_with_cp(a[0], total, response).detach())
        returns.append(slice_log_prob_with_cp(r[0], total, response).detach())
    data["advantages"], data["returns"] = advantages, returns
