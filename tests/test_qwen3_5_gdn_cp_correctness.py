"""GDN native-CP correctness + memory test with the real Qwen3.5-122B-A10B GDN dims.

Compares a CP-sharded GatedDeltaNet against a single-rank full-sequence reference
running on the same GPU with the same weights, and reports peak activation memory
per rank. This is the go/no-go check for enabling CP=2 on the 122B run: we need
(a) outputs and gradients to match, and (b) memory to actually drop ~1/cp, since
the whole point is surviving the GDN backward on long trajectories.

Run:
    torchrun --nproc_per_node=2 tests/test_qwen3_5_gdn_cp_correctness.py
    CP_SEQ_LEN=32768 torchrun --nproc_per_node=2 tests/test_qwen3_5_gdn_cp_correctness.py

Requires flash-linear-attention >= 0.4.2 (for fla.ops.cp).
"""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from slime_plugins.models.qwen3_5 import Qwen3_5GatedDeltaNet  # noqa: E402


class Qwen35_122B_GDNConfig:
    """The linear-attention block dims from
    /path/to/shared_storage/hf_models/Qwen/Qwen3.5-122B-A10B/config.json
    """

    hidden_size = 3072
    linear_num_value_heads = 64
    linear_num_key_heads = 16
    linear_key_head_dim = 128
    linear_value_head_dim = 128
    linear_conv_kernel_dim = 4
    hidden_act = "silu"
    rms_norm_eps = 1e-6
    dtype = torch.bfloat16


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    cp_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())
    group = dist.group.WORLD

    total_seq_len = int(os.environ.get("CP_SEQ_LEN", 4096))
    assert total_seq_len % (2 * cp_size) == 0, f"{total_seq_len} must be divisible by 2*cp_size"
    local_seq_len = total_seq_len // cp_size

    cfg = Qwen35_122B_GDNConfig()
    torch.manual_seed(1234)  # same seed on every rank -> identical weights

    # Reference: full sequence, no CP.
    model_ref = Qwen3_5GatedDeltaNet(cfg, layer_idx=0).to(device=device, dtype=cfg.dtype)
    # CP model: identical weights, but configured for native fla CP.
    model_cp = Qwen3_5GatedDeltaNet(cfg, layer_idx=0).to(device=device, dtype=cfg.dtype)
    model_cp.load_state_dict(model_ref.state_dict())
    # mimic detect_and_setup_hybrid_cp()
    model_cp.cp_group = group
    model_cp.cp_rank = rank
    model_cp.cp_world_size = cp_size

    # One packed sequence covering the whole stream. GLOBAL cu_seqlens: that is the
    # contract of build_gdn_cp_context (fla derives per-rank boundaries itself).
    global_cu_seqlens = torch.tensor([0, total_seq_len], dtype=torch.int32, device=device)
    hidden_full = torch.randn(1, total_seq_len, cfg.hidden_size, device=device, dtype=cfg.dtype) * 0.02

    # --- reference forward/backward on the full sequence ---
    x_ref = hidden_full.clone().requires_grad_(True)
    out_ref = model_ref(hidden_states=x_ref, cu_seqlens=global_cu_seqlens)
    out_ref.float().pow(2).sum().backward()

    # --- CP forward/backward on this rank's contiguous shard ---
    # Contiguous, NOT zigzag: the relayout lives in the HuggingfaceAttention wrapper,
    # so by the time the GDN sees data it is already a packed shard.
    x_cp = hidden_full[:, rank * local_seq_len : (rank + 1) * local_seq_len, :].clone().requires_grad_(True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    base_mem = torch.cuda.memory_allocated(device)

    out_cp = model_cp(hidden_states=x_cp, cu_seqlens=global_cu_seqlens)
    # Scale so the upstream grad matches the reference's on this shard.
    out_cp.float().pow(2).sum().backward()

    torch.cuda.synchronize()
    cp_peak = torch.cuda.max_memory_allocated(device) - base_mem

    # --- compare against the matching slice of the reference ---
    ref_slice = out_ref[:, rank * local_seq_len : (rank + 1) * local_seq_len, :]
    ref_grad_slice = x_ref.grad[:, rank * local_seq_len : (rank + 1) * local_seq_len, :]

    out_diff = (out_cp.float() - ref_slice.float()).abs().max().item()
    out_scale = ref_slice.float().abs().max().item()
    grad_diff = (x_cp.grad.float() - ref_grad_slice.float()).abs().max().item()
    grad_scale = ref_grad_slice.float().abs().max().item()

    # Also measure the reference's peak on rank 0 for the memory comparison.
    if rank == 0:
        del out_ref, x_ref
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        x_ref2 = hidden_full.clone().requires_grad_(True)
        torch.cuda.reset_peak_memory_stats(device)
        base2 = torch.cuda.memory_allocated(device)
        o2 = model_ref(hidden_states=x_ref2, cu_seqlens=global_cu_seqlens)
        o2.float().pow(2).sum().backward()
        torch.cuda.synchronize()
        ref_peak = torch.cuda.max_memory_allocated(device) - base2
    else:
        ref_peak = 0

    stats = {
        "rank": rank,
        "out_diff": out_diff,
        "out_scale": out_scale,
        "grad_diff": grad_diff,
        "grad_scale": grad_scale,
        "cp_peak_gb": cp_peak / 2**30,
        "ref_peak_gb": ref_peak / 2**30,
    }
    gathered = [None] * cp_size
    dist.all_gather_object(gathered, stats)

    if rank == 0:
        print(f"\n=== Qwen3.5-122B-A10B GDN, CP={cp_size}, T={total_seq_len} "
              f"(local {local_seq_len}), bf16 ===", flush=True)
        print(f"  dims: hidden={cfg.hidden_size} v_heads={cfg.linear_num_value_heads} "
              f"k_heads={cfg.linear_num_key_heads} head_dim={cfg.linear_key_head_dim} "
              f"conv_k={cfg.linear_conv_kernel_dim}", flush=True)
        worst_out = max(s["out_diff"] for s in gathered)
        worst_grad = max(s["grad_diff"] for s in gathered)
        for s in gathered:
            print(f"  rank {s['rank']}: out_maxdiff={s['out_diff']:.3e} (scale {s['out_scale']:.3e})  "
                  f"grad_maxdiff={s['grad_diff']:.3e} (scale {s['grad_scale']:.3e})  "
                  f"peak_activation={s['cp_peak_gb']:.3f} GB", flush=True)
        print(f"\n  no-CP reference peak activation (full T on 1 GPU): "
              f"{gathered[0]['ref_peak_gb']:.3f} GB", flush=True)
        if gathered[0]["ref_peak_gb"] > 0:
            ratio = gathered[0]["ref_peak_gb"] / max(gathered[0]["cp_peak_gb"], 1e-9)
            print(f"  memory reduction: {ratio:.2f}x  (ideal {cp_size:.2f}x)", flush=True)

        # bf16 tolerance, same threshold the upstream PR used
        tol = 1e-2
        ok = worst_out < tol and worst_grad < tol
        print(f"\n  {'[PASS]' if ok else '[FAIL]'} worst out_diff={worst_out:.3e} "
              f"worst grad_diff={worst_grad:.3e} (tol {tol})", flush=True)
        dist.destroy_process_group()
        sys.exit(0 if ok else 1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
