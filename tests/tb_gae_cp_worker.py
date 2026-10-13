"""Real localhost CPU Gloo collectives with Megatron's native CP mapping."""

import copy
from datetime import timedelta

import torch
import torch.distributed as dist


def worker(rank, world_size, rendezvous, gamma, interleaved):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    try:
        from megatron.core import mpu
        from tb_gae_test_utils import assert_bits_equal, tb_args

        from slime.backends.megatron_utils.cp_utils import all_gather_with_cp, slice_log_prob_with_cp
        from slime_plugins.harbor.tb_gae import compute_tb_gae, compute_tb_gae_cp, tb_gae_advantage

        mpu.get_context_parallel_world_size = lambda: world_size
        mpu.get_context_parallel_rank = lambda: rank
        mpu.get_context_parallel_group = lambda: dist.group.WORLD
        tp_group = None
        for member in range(world_size):
            group = dist.new_group([member], backend="gloo")
            if member == rank:
                tp_group = group
        mpu.get_tensor_model_parallel_world_size = lambda: 1
        mpu.get_tensor_model_parallel_rank = lambda: 0
        mpu.get_tensor_model_parallel_group = lambda: tp_group
        mpu.get_data_parallel_group = lambda **kw: tp_group
        mpu.is_pipeline_last_stage = lambda **kw: True
        totals, responses = [2, 7, 8, 9, 17, 64, 4096], [1, 3, 7, 1, 12, 1, 2048]
        generator = torch.Generator().manual_seed(803)
        local_values, local_rewards, masks, turns, dones, references = [], [], [], [], [], []
        for total, response in zip(totals, responses, strict=True):
            values = torch.randn(response, generator=generator)
            reward = torch.zeros(response)
            reward[-1] = 0.7
            action = torch.ones(response, dtype=torch.bool)
            if interleaved:
                action[1:-1:3] = False
            turn = action.clone()
            done = torch.zeros(response, dtype=torch.bool)
            done[-1] = True
            local = slice_log_prob_with_cp(values, total, response)
            with torch.no_grad():
                gathered = all_gather_with_cp(local, total, response)
            assert_bits_equal(gathered, values)
            ids = torch.arange(response, dtype=torch.float32)
            local_ids = slice_log_prob_with_cp(ids, total, response)
            with torch.no_grad():
                restored_ids = all_gather_with_cp(local_ids, total, response)
            assert_bits_equal(restored_ids, ids)
            local_values.append(local.clone().requires_grad_())
            local_rewards.append(slice_log_prob_with_cp(reward, total, response))
            masks.append(action)
            turns.append(turn)
            dones.append(done)
            references.append(
                compute_tb_gae(values[None], reward[None], action[None], turn[None], done[None], gamma, gamma)
            )
        actual = compute_tb_gae_cp(
            local_values,
            local_rewards,
            masks,
            turns,
            dones,
            total_lengths=totals,
            response_lengths=responses,
            gamma_turn=gamma,
            lambda_turn=gamma,
        )
        for i, (total, response) in enumerate(zip(totals, responses, strict=True)):
            for output, expected in zip(actual, references[i], strict=True):
                target = slice_log_prob_with_cp(expected[0], total, response)
                assert_bits_equal(output[i], target)
                assert output[i].shape == local_values[i].shape
                assert output[i].grad_fn is None and not output[i].requires_grad
        # The actual public hook must dispatch through the CP wrapper too.
        data = dict(
            values=local_values,
            rewards=[0.7] * len(totals),
            loss_masks=masks,
            turn_end_mask=turns,
            is_real_done=dones,
            total_lengths=totals,
            response_lengths=responses,
        )
        tb_gae_advantage(
            tb_args(context_parallel_size=world_size, tb_gae_gamma_turn=gamma, tb_gae_lambda_turn=gamma), data
        )
        assert_bits_equal(data["advantages"], actual[0])
        assert_bits_equal(data["returns"], actual[1])
        # Freeze-vs-current baseline under the same *real CP group*, including
        # the shifted logits and every policy/critic logit gradient.
        from tb_gae_test_utils import before_module

        from slime.backends.megatron_utils import loss as loss_module
        from slime.backends.megatron_utils.cp_utils import get_logits_and_tokens_offset_with_cp, get_sum_of_sample_mean

        prior = before_module("slime/backends/megatron_utils/loss.py", "slime.backends.megatron_utils")
        old = copy.deepcopy(data)
        old.pop("advantages")
        old.pop("returns")
        old.pop("turn_end_mask")
        old.pop("is_real_done")
        # The frozen native PPO adds outcome to rank0[-1] and cannot handle
        # rank0 owning an empty response for an extremely short/padded sample.
        # TB mapping above still tests ALL extremes. Bitwise regression below
        # uses the two inputs on which the unmodified native baseline runs.
        keep = [5, 6]
        old = {key: [value[i] for i in keep] for key, value in old.items()}
        totals = [totals[i] for i in keep]
        responses = [responses[i] for i in keep]
        masks = [masks[i] for i in keep]
        old["values"] = [v.detach() for v in old["values"]]
        new = copy.deepcopy(old)
        off = tb_args(enable_tb_gae=False, context_parallel_size=world_size, gamma=gamma, lambd=gamma)
        prior.compute_advantages_and_returns(off, old)
        loss_module.compute_advantages_and_returns(off, new)
        assert_bits_equal(old["advantages"], new["advantages"])
        assert_bits_equal(old["returns"], new["returns"])
        tokens = [torch.arange(t) % 8 for t in totals]
        chunks = []
        for total, response in zip(totals, responses, strict=True):
            full = torch.randn(total, 8, generator=generator)
            if world_size == 1:
                chunks.append(full)
            else:
                chunk, offsets, _, _ = get_logits_and_tokens_offset_with_cp(total, response)
                padded = torch.cat((full, full.new_zeros((chunk * 2 * world_size - total, 8))))
                chunks.append(torch.cat([padded[s:e] for s, e in offsets]))
        packed = torch.cat(chunks)[None]
        for batch in (old, new):
            batch["unconcat_tokens"] = tokens
            batch["group_mask_sums"] = None
            batch["log_probs"] = [torch.full_like(v, -2.1) for v in batch["values"]]
        reducer = get_sum_of_sample_mean(totals, responses, masks)
        for name, width in (("policy_loss_function", 8), ("value_loss_function", 1)):
            x, y = packed[..., :width].clone().requires_grad_(), packed[..., :width].clone().requires_grad_()
            a, metrics_a = getattr(prior, name)(off, old, x, reducer)
            b, metrics_b = getattr(loss_module, name)(off, new, y, reducer)
            assert_bits_equal(a, b)
            assert_bits_equal(metrics_a, metrics_b)
            assert_bits_equal(torch.autograd.grad(a, x)[0], torch.autograd.grad(b, y)[0])
    finally:
        dist.destroy_process_group()
