"""End-to-end hybrid-CP test through the real HuggingfaceAttention wrapper path.

The two other CP tests check the pieces in isolation (relayout permutation exactness;
GDN CP on a single sequence). This one exercises what production actually does:

    zigzag hidden_states  ->  _zigzag_to_packed_shard  ->  GDN with fla cp_context
                          ->  _packed_shard_to_zigzag  ->  zigzag output

with a *multi-sequence varlen* batch shaped like ``get_batch`` in
``slime/backends/megatron_utils/data.py`` produces it: several real samples in
zig-zag layout followed by a trailing pad segment laid out contiguously per rank.

The reference is the current production path: the same GDN run on the full
gathered sequence (what ``_AllGatherForDuplicatedComputation`` gives today),
then sliced back to zigzag. So a PASS here means "switching to hybrid CP does
not change what training computes".

Run:
    torchrun --nproc_per_node=2 tests/test_qwen3_5_gdn_cp_e2e_varlen.py
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
from slime_plugins.models.qwen3_5 import Qwen3_5GatedDeltaNet  # noqa: E402


class Qwen35_122B_GDNConfig:
    hidden_size = 3072
    linear_num_value_heads = 64
    linear_num_key_heads = 16
    linear_key_head_dim = 128
    linear_value_head_dim = 128
    linear_conv_kernel_dim = 4
    hidden_act = "silu"
    rms_norm_eps = 1e-6
    dtype = torch.bfloat16


def build_varlen_batch(rank, cp_size, hidden, device, dtype):
    """Mimic get_batch(): zigzag samples + contiguous trailing pad.

    Returns (zigzag_local[T_local, 1, H], full_global[T_global, 1, H], cu_seqlens).
    Sample lengths are chosen to look like real tmax15k trajectories.
    """
    # per-rank chunk_size from slice_with_cp: ceil(len / (2*cp)) then padded up
    sample_lens = [8192, 4096, 2048]
    pad_local = 256  # trailing pad segment, contiguous per rank

    torch.manual_seed(4321)  # identical global data on every rank
    global_seqs = []
    cu = [0]
    for L in sample_lens:
        assert L % (2 * cp_size) == 0, f"{L} not divisible by 2*cp={2 * cp_size}"
        global_seqs.append(torch.randn(L, 1, hidden, device=device, dtype=dtype) * 0.02)
        cu.append(cu[-1] + L)
    global_seqs.append(torch.randn(pad_local * cp_size, 1, hidden, device=device, dtype=dtype) * 0.02)
    cu.append(cu[-1] + pad_local * cp_size)

    full_global = torch.cat(global_seqs, dim=0)

    # build this rank's zigzag shard
    parts = []
    for seq_idx, L in enumerate(sample_lens):
        seq = global_seqs[seq_idx]
        n_sub = 2 * cp_size
        sub = L // n_sub
        parts.append(torch.cat([seq[rank * sub : (rank + 1) * sub],
                                seq[(n_sub - 1 - rank) * sub : (n_sub - rank) * sub]], dim=0))
    pad_seq = global_seqs[-1]
    parts.append(pad_seq[rank * pad_local : (rank + 1) * pad_local])

    zigzag_local = torch.cat(parts, dim=0)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
    return zigzag_local, full_global, cu_seqlens


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    cp_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())
    group = dist.group.WORLD

    cfg = Qwen35_122B_GDNConfig()
    torch.manual_seed(1234)
    gdn = Qwen3_5GatedDeltaNet(cfg, layer_idx=0).to(device=device, dtype=cfg.dtype)

    zigzag_local, full_global, cu_seqlens = build_varlen_batch(
        rank, cp_size, cfg.hidden_size, device, cfg.dtype
    )
    sample_lens_ref = [8192, 4096, 2048]  # must match build_varlen_batch
    T_global = full_global.size(0)
    T_local = zigzag_local.size(0)

    # ---------- reference: today's duplicated-compute path ----------
    # full sequence, no cp_group configured -> plain non-CP GDN
    x_ref = full_global.clone().requires_grad_(True)
    out_ref_full = gdn(hidden_states=x_ref.permute(1, 0, 2), cu_seqlens=cu_seqlens).permute(1, 0, 2)
    # slice back to zigzag exactly as hf_attention does today
    ref_parts = []
    for i in range(len(cu_seqlens) - 1):
        seq = out_ref_full[cu_seqlens[i] : cu_seqlens[i + 1]]
        if i < len(sample_lens_ref):
            chunks = torch.chunk(seq, 2 * cp_size, dim=0)
            ref_parts += [chunks[rank], chunks[2 * cp_size - 1 - rank]]
        else:
            ref_parts.append(seq.chunk(cp_size, dim=0)[rank])
    out_ref = torch.cat(ref_parts, dim=0)

    # Backward through the FULL output, because that is the loss hybrid CP realises:
    # each CP rank backprops its own shard, and the sum over ranks is the full loss.
    # (Backpropping only this rank's slice would give a partial gradient, since GDN
    # is recurrent -- later tokens' loss also flows into earlier tokens' inputs.)
    out_ref_full.float().pow(2).sum().backward()
    ref_wgrads = {k: (p.grad.clone() if p.grad is not None else None) for k, p in gdn.named_parameters()}
    gdn.zero_grad(set_to_none=True)

    # ---------- hybrid CP path ----------
    gdn.cp_group = group
    gdn.cp_rank = rank
    gdn.cp_world_size = cp_size

    x_cp = zigzag_local.clone().requires_grad_(True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)

    NZ = len(sample_lens_ref)  # real samples are zig-zag; trailing pad is not
    packed = _zigzag_to_packed_shard(x_cp, cu_seqlens, group, rank, cp_size, NZ)
    out_packed = gdn(hidden_states=packed.permute(1, 0, 2), cu_seqlens=cu_seqlens).permute(1, 0, 2)
    out_cp = _packed_shard_to_zigzag(out_packed, cu_seqlens, group, rank, cp_size, NZ)
    out_cp.float().pow(2).sum().backward()

    torch.cuda.synchronize()
    cp_peak = (torch.cuda.max_memory_allocated(device) - base) / 2**30
    cp_wgrads = {k: (p.grad.clone() if p.grad is not None else None) for k, p in gdn.named_parameters()}

    # ---------- compare ----------
    out_diff = (out_cp.float() - out_ref.float()).abs().max().item()
    out_scale = out_ref.float().abs().max().item()

    # input grad: the CP grad lives on this rank's zigzag shard. Relayout it back to
    # the global packed stream and all-reduce, then compare against the reference's
    # full-stream grad. (all_reduce because each rank only holds its own shard's
    # contribution; zeros elsewhere.)
    with torch.no_grad():
        cp_grad_packed = _zigzag_to_packed_shard(x_cp.grad, cu_seqlens, group, rank, cp_size, NZ)
        gathered_grad = [torch.empty_like(cp_grad_packed) for _ in range(cp_size)]
        dist.all_gather(gathered_grad, cp_grad_packed.contiguous(), group=group)
        cp_grad_full = torch.cat(gathered_grad, dim=0)
    grad_diff = (cp_grad_full.float() - x_ref.grad.float()).abs().max().item()
    grad_scale = x_ref.grad.float().abs().max().item()

    # weight grads: reference is computed on the FULL sequence by one rank, while
    # hybrid CP splits the work, so the CP weight grads must be summed over ranks.
    wg_report = []
    for k in ref_wgrads:
        if ref_wgrads[k] is None:
            continue
        summed = cp_wgrads[k].clone().float()
        dist.all_reduce(summed, group=group)
        d = (summed - ref_wgrads[k].float()).abs().max().item()
        s = ref_wgrads[k].float().abs().max().item()
        wg_report.append((k, d, s, d / s if s else 0.0))

    stats = {
        "rank": rank, "out_diff": out_diff, "out_scale": out_scale,
        "grad_diff": grad_diff, "grad_scale": grad_scale,
        "cp_peak": cp_peak, "wg": wg_report,
    }
    gathered = [None] * cp_size
    dist.all_gather_object(gathered, stats)

    if rank == 0:
        print(f"\n=== E2E hybrid CP through wrapper path, CP={cp_size} ===", flush=True)
        print(f"  varlen batch: samples 8192+4096+2048 + pad {256 * cp_size}  "
              f"=> T_global={T_global}, T_local={T_local}", flush=True)
        for s in gathered:
            print(f"  rank {s['rank']}: out_maxdiff={s['out_diff']:.3e} (scale {s['out_scale']:.3e})  "
                  f"in_grad_maxdiff={s['grad_diff']:.3e} (scale {s['grad_scale']:.3e})  "
                  f"peak={s['cp_peak']:.3f} GB", flush=True)
        print("\n  weight grads (CP summed over ranks) vs full-sequence reference:", flush=True)
        worst_w = 0.0
        for k, d, s, r in gathered[0]["wg"]:
            print(f"    {k:24s} absdiff={d:.3e} scale={s:.3e} rel={r:.2e}", flush=True)
            worst_w = max(worst_w, r)

        worst_out = max(s["out_diff"] / max(s["out_scale"], 1e-30) for s in gathered)
        worst_grad = max(s["grad_diff"] / max(s["grad_scale"], 1e-30) for s in gathered)
        # bf16 has ~7.8e-3 relative resolution (1 ULP); allow a few ULP of drift
        tol = 5e-2
        ok = worst_out < tol and worst_grad < tol and worst_w < tol
        print(f"\n  {'[PASS]' if ok else '[FAIL]'} worst rel: out={worst_out:.2e} "
              f"in_grad={worst_grad:.2e} w_grad={worst_w:.2e} (tol {tol}, 1 bf16 ULP = 7.8e-3)", flush=True)
        dist.destroy_process_group()
        sys.exit(0 if ok else 1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
