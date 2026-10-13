"""Native context-parallel (CP) support for GatedDeltaNet layers.

slime's generic CP path all-gathers the sequence back to full length before the
linear-attention module and runs it *duplicated* on every CP rank
(``_AllGatherForDuplicatedComputation`` in :mod:`hf_attention`). That is correct
but saves no activation memory inside the GDN — the dominant cost for long
trajectories.

``fla >= 0.4.2`` ships ``fla.ops.cp``, which makes the GDN recurrence itself
context-parallel: each rank computes over only its own ``T/cp`` tokens, summarises
its shard as an affine operator on the recurrent state, and a single all-gather +
Triton merge replays the recurrence prefix. Forward and backward are both
supported. This module wires that in.

Ported from radixark/miles#885 ("real cp support with relayout fix for qwen3.5
train/rollout mismatch").
"""

import logging

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from fla.ops.cp import build_cp_context as _fla_build_cp_context
except ImportError:
    _fla_build_cp_context = None

from .hf_attention import HuggingfaceAttention

logger = logging.getLogger(__name__)


def build_gdn_cp_context(module: nn.Module, cu_seqlens: torch.Tensor, device: torch.device):
    """Build an fla CP context for a GatedDeltaNet module from packed sequence boundaries.

    Args:
        module: GDN module carrying ``cp_group`` / ``cp_world_size`` / ``conv_kernel_size``
            (set by :func:`detect_and_setup_hybrid_cp`).
        cu_seqlens: **Global** packed sequence boundaries, i.e.
            ``packed_seq_params.cu_seqlens_q`` before CP partitioning. fla derives the
            per-rank local boundaries itself and exposes them as ``ctx.cu_seqlens``.
        device: Target device.

    Returns ``None`` when CP is not configured on this module, which makes the
    caller fall back to the plain single-rank GDN path.
    """
    cp_group = getattr(module, "cp_group", None)
    if cp_group is None:
        return None
    if _fla_build_cp_context is None:
        raise RuntimeError(
            "Hybrid CP requires fla.ops.cp (flash-linear-attention >= 0.4.2) but it could not be imported."
        )
    if cu_seqlens is None or cu_seqlens.numel() < 2:
        raise ValueError(f"Hybrid CP requires valid cu_seqlens (at least 2 elements) but got {cu_seqlens}")
    return _fla_build_cp_context(
        cu_seqlens=cu_seqlens.to(device=device, dtype=torch.int32),
        group=cp_group,
        conv1d_kernel_size=module.conv_kernel_size,
    )


def detect_and_setup_hybrid_cp(
    model: nn.Module,
    cp_group: dist.ProcessGroup,
    cp_rank: int,
    cp_world_size: int,
) -> int:
    """Scan for GatedDeltaNet modules and configure them for native fla CP.

    Sets ``cp_group`` / ``cp_rank`` / ``cp_world_size`` on the inner ``linear_attn``
    (the GDN), and flips ``hybrid_cp`` on the outer :class:`HuggingfaceAttention`
    wrapper so it swaps the duplicate-compute all-gather for the zigzag->packed
    relayout that fla's CP kernels expect.

    Returns the number of GDN modules configured.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, HuggingfaceAttention):
            linear_attn = getattr(module, "linear_attn", None)
            if linear_attn is not None:
                linear_attn.cp_group = cp_group
                linear_attn.cp_rank = cp_rank
                linear_attn.cp_world_size = cp_world_size
                module.hybrid_cp = True
                count += 1

    if count > 0:
        logger.info(f"Configured hybrid CP on {count} GDN modules (fla native state passing)")
    return count
