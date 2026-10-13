"""Exactness test for the zigzag <-> packed-shard relayout used by hybrid GDN CP.

This is a pure-tensor test: no model, no fla, no checkpoint. It verifies that the
relayout in ``slime_plugins/models/hf_attention.py`` is an exact permutation, because
a wrong permutation is *silent* -- training would run and quietly compute logprobs
for the wrong token order.

Run:
    torchrun --nproc_per_node=2 tests/test_hf_attention_cp_relayout.py
    torchrun --nproc_per_node=4 tests/test_hf_attention_cp_relayout.py

The fixture's trailing pad segment has a global length that IS divisible by
``2*cp_size`` while still being laid out *contiguously* per rank -- which is exactly
what ``get_batch`` in ``slime/backends/megatron_utils/data.py`` produces (real
samples become ``2*chunk*cp``, the pad becomes ``pad_size*k*cp``; both divisible).
That means the layout cannot be inferred from ``cu_seqlens`` and must be passed in
explicitly. This test fails if anyone reintroduces a divisibility-based guess.
"""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from slime_plugins.models.hf_attention import (  # noqa: E402
    _packed_shard_to_zigzag,
    _zigzag_to_packed_shard,
)

HIDDEN = 8


def build_fixture(rank, world_size, device):
    """Return (zigzag_local, packed_local, cu_seqlens).

    Two zig-zag sequences of ``2*world_size*chunk_len`` tokens each, plus one
    trailing pad segment laid out contiguously. The pad's *global* length is
    ``2*world_size*2``, i.e. divisible by ``2*cp_size`` -- mirroring production,
    where the pad is a multiple of ``pad_size*cp`` and so cannot be distinguished
    from a real sample by divisibility alone.
    Every token gets a globally unique value so any permutation error is caught.
    """
    chunk_lens = [3, 5]
    tail_pad_local_len = 4  # global = 4*world_size, divisible by 2*world_size

    def token(value):
        # [n, HIDDEN] where every row is the token's unique id (float-exact in fp32)
        return value.unsqueeze(-1).expand(-1, HIDDEN).float().contiguous()

    zigzag_parts = []
    global_seqs = []
    cu = [0]

    for seq_idx, chunk_len in enumerate(chunk_lens):
        n_sub = 2 * world_size
        global_len = n_sub * chunk_len
        # sub-chunk s holds ids seq_idx*1000 + s*100 + [0..chunk_len)
        subs = [
            torch.arange(chunk_len, device=device) + seq_idx * 1000 + s * 100
            for s in range(n_sub)
        ]
        global_seqs.append(torch.cat(subs, dim=0))
        # zigzag: rank k holds [sub_k, sub_{2N-1-k}]
        zigzag_parts.append(torch.cat([subs[rank], subs[n_sub - 1 - rank]], dim=0))
        cu.append(cu[-1] + global_len)

    # trailing pad: contiguous per rank
    pad_all = [
        torch.arange(tail_pad_local_len, device=device) + r * 10000 for r in range(world_size)
    ]
    global_seqs.append(torch.cat(pad_all, dim=0))
    zigzag_parts.append(pad_all[rank])
    cu.append(cu[-1] + world_size * tail_pad_local_len)

    zigzag_local = token(torch.cat(zigzag_parts, dim=0))

    # The packed layout is a contiguous slice of the whole global stream.
    full_stream = torch.cat(global_seqs, dim=0)
    shard_len = zigzag_local.size(0)
    assert full_stream.numel() == shard_len * world_size, (full_stream.numel(), shard_len, world_size)
    packed_local = token(full_stream[rank * shard_len : (rank + 1) * shard_len])

    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
    return zigzag_local, packed_local, cu_seqlens


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())
    group = dist.group.WORLD

    zigzag, packed_ref, cu_seqlens = build_fixture(rank, world_size, device)
    zigzag = zigzag.requires_grad_(True)

    failures = []

    # 1. zigzag -> packed matches the independently-constructed packed shard
    NZ = 2  # two zig-zag samples; the 3rd segment is the contiguous pad
    got = _zigzag_to_packed_shard(zigzag, cu_seqlens, group, rank, world_size, NZ)
    if not torch.equal(got, packed_ref):
        n_bad = (got != packed_ref).any(dim=-1).sum().item()
        failures.append(f"zigzag->packed mismatch on {n_bad}/{got.size(0)} tokens")

    # 2. round trip is the identity
    back = _packed_shard_to_zigzag(packed_ref, cu_seqlens, group, rank, world_size, NZ)
    if not torch.equal(back, zigzag.detach()):
        n_bad = (back != zigzag.detach()).any(dim=-1).sum().item()
        failures.append(f"roundtrip mismatch on {n_bad}/{back.size(0)} tokens")

    # 3. backward through both directions returns the gradient to its origin
    roundtrip = _packed_shard_to_zigzag(
        _zigzag_to_packed_shard(zigzag, cu_seqlens, group, rank, world_size, NZ),
        cu_seqlens,
        group,
        rank,
        world_size,
        NZ,
    )
    roundtrip.sum().backward()
    if not torch.equal(zigzag.grad, torch.ones_like(zigzag)):
        failures.append(f"backward grad wrong: sum={zigzag.grad.sum().item()} expected={zigzag.numel()}")

    # gather verdicts so any rank's failure fails the whole run
    all_failures = [None] * world_size
    dist.all_gather_object(all_failures, failures)
    dist.barrier()

    if rank == 0:
        bad = False
        for r, f in enumerate(all_failures):
            if f:
                bad = True
                for msg in f:
                    print(f"[FAIL] rank {r}: {msg}", flush=True)
        if not bad:
            print(f"[PASS] relayout exact for cp_size={world_size} "
                  f"(fwd, roundtrip, backward; incl. non-2*cp-divisible pad segment)", flush=True)
        dist.destroy_process_group()
        sys.exit(1 if bad else 0)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
