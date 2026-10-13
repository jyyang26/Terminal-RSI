"""Isolated CPU/gloo worker; exercises production SC kernels with TP x CP groups."""

import argparse
from datetime import timedelta
import socket
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, port, tp_size, cp_size):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=tp_size * cp_size,
        timeout=timedelta(seconds=90),
    )
    from megatron.core import mpu
    from slime.backends.megatron_utils import cp_utils, sc_terms
    from slime.backends.megatron_utils.loss import get_responses
    from slime.utils.score_centering import calculate_head_log_probs, score_centering_correction

    tp_group = cp_group = None
    for cp in range(cp_size):
        members = list(range(cp * tp_size, (cp + 1) * tp_size))
        group = dist.new_group(members)
        if rank in members:
            tp_group = group
    for tp in range(tp_size):
        members = list(range(tp, tp_size * cp_size, tp_size))
        group = dist.new_group(members)
        if rank in members:
            cp_group = group
    cp_rank, tp_rank = rank // tp_size, rank % tp_size
    mpu.get_context_parallel_world_size = lambda: cp_size
    mpu.get_context_parallel_rank = lambda: cp_rank
    mpu.get_context_parallel_group = lambda: cp_group
    mpu.get_tensor_model_parallel_group = lambda: tp_group
    mpu.get_tensor_model_parallel_world_size = lambda: tp_size
    # TP normalization and backward factor, with masked/padding rows outside the gather.
    gen = torch.Generator().manual_seed(42)
    full = torch.randn(7, 16 * tp_size, dtype=torch.float64, generator=gen)
    rows = torch.tensor([0, 2, 4, 6])
    ids = torch.tensor([[0, 16 * tp_size - 1], [2, 7], [4, 9], [6, 10]])
    coefficients = torch.randn(4, 2, dtype=torch.float64, generator=gen)
    for chunk in (1, 3, 256):
        local = full[:, tp_rank * 16 : (tp_rank + 1) * 16].clone().requires_grad_()
        actual = calculate_head_log_probs(local, rows, ids, tp_group, chunk)
        reference = full.clone().requires_grad_()
        expected = reference.log_softmax(-1)[rows].gather(-1, ids)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        (actual * coefficients).sum().backward()
        (expected * coefficients).sum().backward()
        torch.testing.assert_close(
            local.grad, reference.grad[:, tp_rank * 16 : (tp_rank + 1) * 16], atol=1e-12, rtol=1e-12
        )
    # Include a tiny sample and a fully masked sample so some CP ranks own no useful rows.
    totals = [7, 11, 2]
    responses = [3, 6, 1]
    vocab = 16 * tp_size
    k = 3
    tokens = [torch.arange(t) % vocab for t in totals]
    full = [torch.randn(t, vocab, generator=gen) for t in totals]
    old = [torch.randn(t, vocab, generator=gen) for t in totals]
    qfull = [torch.randn(r, vocab, generator=gen).log_softmax(-1) for r in responses]
    candidates = [
        torch.stack([tok[-r:], (tok[-r:] + 1) % vocab, (tok[-r:] + 2) % vocab], -1)
        for tok, r in zip(tokens, responses, strict=True)
    ]
    q = [lp.gather(-1, ix) for lp, ix in zip(qfull, candidates, strict=True)]
    masks = [torch.tensor([1, 0, 1]), torch.tensor([1, 1, 0, 1, 1, 1]), torch.tensor([0])]
    adv = [torch.tensor([1.0, -2.0, 3.0]), torch.tensor([-1.0, 2.0, -3.0, 1.0, -2.0, 3.0]), torch.tensor([1.0])]

    def layout(values, gather, fmt="thd"):
        if gather:
            flat = torch.cat(values)
            pad = (-len(flat)) % cp_size
            packed = torch.cat([flat, flat.new_zeros((pad, vocab))]).chunk(cp_size)[cp_rank][None]
        else:
            parts = []
            for value, total in zip(values, totals, strict=True):
                maximum = 16 if fmt == "bshd" else total
                if cp_size == 1:
                    part = torch.cat([value, value.new_zeros((maximum - total, vocab))])
                else:
                    chunk = (maximum + 2 * cp_size - 1) // (2 * cp_size)
                    padded = torch.cat([value, value.new_zeros((2 * cp_size * chunk - total, vocab))])
                    part = torch.cat(
                        [
                            padded[cp_rank * chunk : (cp_rank + 1) * chunk],
                            padded[(2 * cp_size - cp_rank - 1) * chunk : (2 * cp_size - cp_rank) * chunk],
                        ]
                    )
                parts.append(part)
            packed = torch.stack(parts) if fmt == "bshd" else torch.cat(parts)[None]
        return packed[..., tp_rank * 16 : (tp_rank + 1) * 16].clone()

    for fmt, gather in [("thd", False), ("thd", True), ("bshd", False)]:
        args = SimpleNamespace(
            qkv_format=fmt,
            allgather_cp=gather,
            sc_top_k=k,
            sc_chunk_size=2,
            sc_formula="ppo_weighted",
            sc_weight_denominator="old_actor",
            eps_clip=0.2,
            eps_clip_high=0.2,
            sc_tail_eps=1e-8,
        )

        def local_response(x, t, r):
            return cp_utils.slice_log_prob_with_cp(x, t, r, fmt, 16 if fmt == "bshd" else None)

        batch = dict(
            total_lengths=totals,
            response_lengths=responses,
            unconcat_tokens=tokens,
            loss_masks=masks,
            max_seq_lens=[16] * 3 if fmt == "bshd" else None,
            rollout_topk_token_ids=candidates,
            rollout_topk_log_probs=q,
            sc_response_token_ids=[t[-r:].int() for t, r in zip(tokens, responses, strict=True)],
            advantages=[local_response(x, t, r) for x, t, r in zip(adv, totals, responses, strict=True)],
            rollout_log_probs=[
                local_response(lp.gather(-1, tok[-r:, None]).squeeze(-1), t, r)
                for lp, tok, t, r in zip(qfull, tokens, totals, responses, strict=True)
            ],
        )
        logits = layout(full, gather, fmt)
        flat, layouts = sc_terms.response_row_layout(args, batch, logits)
        real = get_responses(
            logits,
            args=args,
            unconcat_tokens=tokens,
            total_lengths=totals,
            response_lengths=responses,
            max_seq_lens=batch["max_seq_lens"],
            apply_temperature=False,
        )
        for i, ((packed_rows, response_rows), (values, tok)) in enumerate(zip(layouts, real, strict=True)):
            assert torch.equal(flat[packed_rows], values)
            assert torch.equal(tokens[i][-responses[i] :][response_rows], tok)
        collect = sc_terms.old_policy_head_callback(lambda logits: (torch.empty(0), {}), args, batch)
        _, old_result = collect(layout(old, gather, fmt))
        batch.update(old_result)
        for o, actual, ix, t, r, mask in zip(
            old, old_result["sc_old_head_log_probs"], candidates, totals, responses, masks, strict=True
        ):
            expected = o[t - r - 1 : t - 1].log_softmax(-1).gather(-1, ix)
            torch.testing.assert_close(actual[mask.bool()], expected[mask.bool()], atol=1e-6, rtol=1e-6)
            assert actual.device.type == "cpu" and not actual.requires_grad
        for formula in ("score", "ppo_weighted"):
            args.sc_formula = formula
            logits = layout(full, gather, fmt).requires_grad_()
            addend, metrics = sc_terms.get_score_centering_terms(
                args, batch, logits, advantages=torch.cat(batch["advantages"])
            )
            addend.sum().backward()
            refs = []
            reference_logits = [f.clone().requires_grad_() for f in full]
            for f, o, ql, ix, av, mask, t, r in zip(
                reference_logits, old, q, candidates, adv, masks, totals, responses, strict=True
            ):
                lp = f[t - r - 1 : t - 1].log_softmax(-1).gather(-1, ix)
                ol = o[t - r - 1 : t - 1].log_softmax(-1).gather(-1, ix)
                c, _ = score_centering_correction(
                    train_head_logp=lp,
                    sampler_head_logp=ql,
                    advantages=av,
                    formula=formula,
                    old_head_logp=ol,
                    loss_mask=mask,
                )
                refs.append(av * c)
            expected = torch.cat([local_response(x, t, r) for x, t, r in zip(refs, totals, responses, strict=True)])
            torch.testing.assert_close(addend, expected, atol=1e-5, rtol=1e-5)
            sum(x.sum() for x in refs).backward()
            expected_gradient = layout([x.grad for x in reference_logits], gather, fmt)
            torch.testing.assert_close(logits.grad, expected_gradient, atol=1e-6, rtol=1e-5)
            assert all(torch.isfinite(value).all() for value in metrics.values())
    if rank == 0:
        print(f"TP{tp_size}_CP{cp_size}: forward, backward, layout, mask, old-policy PASS", flush=True)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--cp", type=int, default=2)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(port, args.tp, args.cp), nprocs=args.tp * args.cp, join=True)


if __name__ == "__main__":
    main()
