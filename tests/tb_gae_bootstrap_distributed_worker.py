"""CPU/Gloo DP=2 x CP=1/4; chunks of one episode live on different DP ranks."""

from datetime import timedelta

import torch
import torch.distributed as dist


def worker(rank, cp_size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2 * cp_size, timeout=timedelta(seconds=90)
    )
    try:
        from megatron.core import mpu
        from tb_gae_test_utils import tb_args

        from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp
        from slime_plugins.harbor.tb_gae import compute_tb_gae, tb_gae_advantage

        dp_rank, cp_rank = rank // cp_size, rank % cp_size
        cp_group = dp_group = None
        for d in range(2):
            group = dist.new_group(list(range(d * cp_size, (d + 1) * cp_size)), backend="gloo")
            if d == dp_rank:
                cp_group = group
        for c in range(cp_size):
            group = dist.new_group([c, cp_size + c], backend="gloo")
            if c == cp_rank:
                dp_group = group
        mpu.get_context_parallel_world_size = lambda: cp_size
        mpu.get_context_parallel_rank = lambda: cp_rank
        mpu.get_context_parallel_group = lambda: cp_group
        mpu.get_data_parallel_group_gloo = lambda **kw: dp_group

        # Real bootstrap callback, native packing/get_values and CP gather.
        # Pipeline scheduler/model are CPU test doubles; DP counts are unequal.
        from types import SimpleNamespace

        import megatron.core.utils

        from slime.backends.megatron_utils import model as model_utils
        from slime.backends.megatron_utils.tb_gae_bootstrap import compute_bootstrap_values
        from slime.utils import distributed_utils

        mpu.get_tensor_model_parallel_world_size = lambda: 1
        mpu.get_virtual_pipeline_model_parallel_world_size = lambda: None
        mpu.is_pipeline_last_stage = lambda **kw: True
        distributed_utils.get_gloo_group = lambda: dist.group.WORLD
        config = SimpleNamespace(timers=None, microbatch_group_size_per_vp_stage=2)
        megatron.core.utils.get_model_config = lambda model: config
        model_utils.get_model_config = lambda model: config
        model_utils._disable_tqdm_for_non_main_rank = lambda: True
        model_utils._wrap_forward_step_with_microbatch_pbar = lambda fn, bar: fn
        torch.Tensor.cuda = lambda self, *args, **kw: self
        schedules = []

        class ToyCritic:
            def eval(self):
                pass

            def train(self):
                pass

            def __call__(self, **kw):
                # A local causal state function that requires no fake CP allgather.
                return kw["input_ids"].float()[..., None] * 2

        def scheduler(**kw):
            schedules.append(kw["num_microbatches"])
            results = []
            for _ in range(kw["num_microbatches"]):
                logits, callback = kw["forward_step_func"](kw["data_iterator"][0], kw["model"][0])
                _, result = callback(logits)
                results.append(result)
            return results

        model_utils.get_forward_backward_func = lambda: scheduler
        nlocal = 2 if dp_rank == 0 else 3
        probe_data = {
            "tokens": [torch.tensor([1, 2])] * nlocal,
            "tb_boundary": [
                dict(terminal=False, chunk_index=0, chunk_count=1, bootstrap_token_ids=[7, 8, 11 + i])
                for i in range(nlocal)
            ],
        }
        probe_args = tb_args(
            seq_length=64,
            data_pad_size_multiplier=1,
            use_rollout_entropy=False,
            custom_megatron_before_log_prob_hook_path=None,
            micro_batch_size=1,
        )
        actual_bootstrap = compute_bootstrap_values(probe_args, [ToyCritic()], probe_data)
        assert actual_bootstrap == [2.0 * (11 + i) for i in range(nlocal)]
        assert schedules == [4]  # Max count 3 rounded to VPP group multiple 2.

        # Uneven local sample counts and chunks out of order; include tiny CP shards.
        slices = [(0, 3), (3, 10), (10, 11), (11, 20), (20, 25)]
        owner = [0, 1, 0, 1, 1]
        values = torch.sin(torch.arange(25, dtype=torch.float64))
        action = torch.ones(25, dtype=torch.bool)
        action[[1, 5, 8, 13, 18, 22]] = False
        turns = torch.zeros_like(action)
        turns[[2, 7, 10, 17, 24]] = True
        for terminal in (False, True):
            for gamma in (1.0, 0.999):
                done = torch.zeros_like(action)
                done[-1] = terminal
                reward = torch.zeros_like(values)
                reward[-1] = 0.7
                expected = compute_tb_gae(
                    values[None],
                    reward[None],
                    action[None],
                    turns[None],
                    done[None],
                    gamma,
                    gamma,
                    bootstrap_values=torch.tensor([0.8], dtype=torch.float64),
                    continuation_advantages=torch.zeros(1, dtype=torch.float64),
                )
                data = {
                    k: []
                    for k in (
                        "values",
                        "rewards",
                        "loss_masks",
                        "turn_end_mask",
                        "is_real_done",
                        "tb_boundary",
                        "tb_bootstrap_values",
                        "total_lengths",
                        "response_lengths",
                    )
                }
                indices = [i for i in reversed(range(len(slices))) if owner[i] == dp_rank]
                for i in indices:
                    start, stop = slices[i]
                    response = stop - start
                    total = response + (1 if i == 2 else 7)
                    data["values"].append(
                        slice_log_prob_with_cp(values[start:stop], total, response).clone().requires_grad_()
                    )
                    data["rewards"].append(0.7)
                    data["loss_masks"].append(action[start:stop])
                    data["turn_end_mask"].append(turns[start:stop])
                    data["is_real_done"].append(done[start:stop])
                    data["tb_boundary"].append(
                        dict(
                            episode="shared",
                            chunk_index=i,
                            chunk_count=len(slices),
                            terminal=terminal and i == len(slices) - 1,
                        )
                    )
                    data["tb_bootstrap_values"].append(0.8 if i == len(slices) - 1 else 0.0)
                    data["total_lengths"].append(total)
                    data["response_lengths"].append(response)
                args = tb_args(context_parallel_size=cp_size, tb_gae_gamma_turn=gamma, tb_gae_lambda_turn=gamma)
                tb_gae_advantage(args, data)
                for local_i, i in enumerate(indices):
                    start, stop = slices[i]
                    total, response = data["total_lengths"][local_i], stop - start
                    for key, ref in zip(("advantages", "returns"), expected, strict=True):
                        local_ref = slice_log_prob_with_cp(ref[0, start:stop], total, response)
                        got = data[key][local_i]
                        torch.testing.assert_close(got, local_ref, atol=1e-12, rtol=1e-12)
                        assert got.shape == data["values"][local_i].shape
                        assert not got.requires_grad and got.grad_fn is None
    finally:
        dist.destroy_process_group()
