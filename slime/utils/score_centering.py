"""PPO plus score centering; no disk-backed tensor dependency.

All sampler/old-policy coefficients are stop-gradient. The top-k log-probability
operator normalizes over the complete TP-sharded vocabulary.
"""

from __future__ import annotations

import base64
import hashlib
import math

import numpy as np
import torch
import torch.distributed as dist

HEAD_KEYS = (
    "rollout_topk_token_ids",
    "rollout_topk_log_probs",
    "sc_response_token_ids",
)
EPS = 1e-8


def _fp(tensor):
    return tensor if tensor.dtype == torch.float64 else tensor.float()


def validate_head(ids, logps, tokens, *, k, vocab_size=None):
    ids, logps, tokens = np.asarray(ids), np.asarray(logps), np.asarray(tokens)
    if ids.dtype != np.dtype("<i4") or logps.dtype != np.dtype("<f4"):
        raise ValueError("SC heads require little-endian int32/float32")
    if tokens.dtype != np.dtype("<i4") or tokens.ndim != 1:
        raise ValueError("SC row token IDs require a one-dimensional int32 array")
    if ids.shape != (len(tokens), k) or logps.shape != ids.shape:
        raise ValueError("SC head rows/width do not match the response token IDs")
    if k < 1:
        raise ValueError("SC k must be positive")
    for start in range(0, len(tokens), 1024):
        a = ids[start : start + 1024]
        b = logps[start : start + 1024].astype(np.float64)
        t = tokens[start : start + 1024]
        if (a < 0).any() or (t < 0).any():
            raise ValueError("SC token IDs must be nonnegative")
        if vocab_size is not None and ((a >= vocab_size).any() or (t >= vocab_size).any()):
            raise ValueError("SC token ID exceeds the actual vocabulary")
        if (np.diff(np.sort(a, axis=-1), axis=-1) == 0).any():
            raise ValueError("SC head contains duplicate token IDs")
        if not np.isfinite(b).all() or (b > 0).any():
            raise ValueError("SC head requires finite, nonpositive log probabilities")
        if (np.exp(b).sum(-1) > 1.0001).any():
            raise ValueError("SC head is not a probability subdistribution")
    return ids, logps, tokens


def _decode(encoded, dtype, shape):
    if not isinstance(encoded, str):
        raise ValueError("SC binary field must be a base64 string")
    size = math.prod(shape) * np.dtype(dtype).itemsize
    if len(encoded) != 4 * ((size + 2) // 3):
        raise ValueError("SC base64 length does not match declared shape")
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != size:
        raise ValueError("SC decoded byte length does not match declared shape")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def decode_sampler_head(meta, tokens, *, k, wire_format, max_bytes, vocab_size):
    tokens = np.asarray(tokens, dtype="<i4")
    count = len(tokens)
    if count * (8 * k + 4) > max_bytes:
        raise ValueError("SC response exceeds the configured head byte budget")
    binary = ("output_topk_token_ids", "output_topk_log_probs", "output_topk_shape")
    if wire_format == "binary":
        if any(key not in meta for key in binary):
            raise ValueError("SC binary response missing; use the verified SC SGLang runtime")
        if meta[binary[2]] != [count, k]:
            raise ValueError("SGLang SC shape differs from output_token_logprobs")
        ids = _decode(meta[binary[0]], "<i4", (count, k))
        logps = _decode(meta[binary[1]], "<f4", (count, k))
    elif wire_format == "json":
        rows = meta.get("output_top_logprobs")
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError("SGLang JSON head row count mismatch")
        ids = np.empty((count, k), dtype="<i4")
        logps = np.empty((count, k), dtype="<f4")
        for i, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != k:
                raise ValueError(f"SGLang JSON head width mismatch at row {i}")
            for j, item in enumerate(row):
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    raise ValueError("Invalid SGLang JSON top-logprob entry")
                lp, token = item[:2]
                if isinstance(token, bool) or not isinstance(token, int):
                    raise ValueError("SGLang head token IDs must be integers")
                if not 0 <= token < vocab_size:
                    raise ValueError("SGLang head token ID is outside the vocabulary")
                ids[i, j], logps[i, j] = token, lp
    else:
        raise ValueError("SC wire format must be binary or json")
    return validate_head(ids, logps, tokens, k=k, vocab_size=vocab_size)


def assert_sampled_logps(ids, logps, tokens, sampled_logps, mask=None):
    expected = np.asarray(sampled_logps, dtype=np.float64)
    if expected.shape != (len(tokens),) or not np.isfinite(expected).all():
        raise ValueError("SC requires finite sampled-token logprobs for every response row")
    trainable = np.ones(len(tokens), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if trainable.shape != expected.shape:
        raise ValueError("SC loss mask shape mismatch")
    for start in range(0, len(tokens), 1024):
        stop = start + 1024
        matches = ids[start:stop] == tokens[start:stop, None]
        hit = matches.any(-1) & trainable[start:stop]
        if hit.any():
            column = matches.argmax(-1)
            actual = np.take_along_axis(logps[start:stop], column[:, None], axis=1)[:, 0]
            if not np.allclose(actual[hit], expected[start:stop][hit], rtol=0, atol=1e-4):
                raise ValueError("SC head/sampled-logprob cross-check failed")


def encode_turn_head(ids, logps, tokens, prompt_ids):
    def b64(value):
        return base64.b64encode(np.ascontiguousarray(value).tobytes()).decode("ascii")

    prompt = np.asarray(prompt_ids, dtype="<i4")
    digest = hashlib.sha256()
    for value in (prompt, tokens, ids, logps):
        digest.update(np.ascontiguousarray(value).tobytes())
    return {
        "version": 1,
        "shape": list(ids.shape),
        "ids": b64(ids),
        "logps": b64(logps),
        "tokens": b64(tokens),
        "digest": digest.hexdigest(),
    }


def decode_turn_head(envelope, prompt_ids, output_ids, sampled_logps, *, k):
    if not isinstance(envelope, dict) or envelope.get("version") != 1:
        raise ValueError("Harbor turn is missing its SC envelope")
    count = len(output_ids)
    if envelope.get("shape") != [count, k]:
        raise ValueError("Harbor SC envelope shape mismatch")
    ids = _decode(envelope.get("ids"), "<i4", (count, k))
    logps = _decode(envelope.get("logps"), "<f4", (count, k))
    tokens = _decode(envelope.get("tokens"), "<i4", (count,))
    validate_head(ids, logps, tokens, k=k)
    if not np.array_equal(tokens, np.asarray(output_ids, dtype="<i4")):
        raise ValueError("Harbor SC row token IDs differ from completion_token_ids")
    expected = encode_turn_head(ids, logps, tokens, prompt_ids)["digest"]
    if envelope.get("digest") != expected:
        raise ValueError("Harbor SC prompt/token/head digest mismatch")
    assert_sampled_logps(ids, logps, tokens, sampled_logps)
    return ids, logps, tokens


def score_centering_request(*, k, sampling_params, max_bytes):
    if k < 1:
        raise ValueError("SC k must be positive")
    required = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    for key, value in required.items():
        if sampling_params.get(key, value) != value:
            raise ValueError(f"SC requires actual request {key}={value}")
    for key in ("json_schema", "regex", "ebnf", "structural_tag", "logit_bias", "custom_params"):
        if sampling_params.get(key):
            raise ValueError(f"SC does not support constrained sampling: {key}")
    count = sampling_params.get("max_new_tokens")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("SC requires a positive integer max_new_tokens")
    if count * (8 * k + 4) > max_bytes:
        raise ValueError("SC request exceeds the configured head byte budget")
    return {"top_logprobs_num": k, "return_text_in_logprobs": False}


def ppo_effective_weight(ratio, adv_sign, eps_clip_low, eps_clip_high):
    killed = ((adv_sign > 0) & (ratio > 1 + eps_clip_high)) | ((adv_sign < 0) & (ratio < 1 - eps_clip_low))
    return torch.where(killed, torch.zeros_like(ratio), ratio)


def score_centering_correction(
    *,
    train_head_logp,
    sampler_head_logp,
    advantages,
    formula="score",
    old_head_logp=None,
    eps_clip_low=0.2,
    eps_clip_high=0.2,
    tail_eps=EPS,
    loss_mask=None,
):
    if formula not in {"score", "ppo_weighted"}:
        raise ValueError("Unknown SC formula")
    if not math.isfinite(tail_eps) or not 0 < tail_eps < 1:
        raise ValueError("SC tail epsilon must lie in (0, 1)")
    lp = _fp(train_head_logp)
    if lp.ndim != 2 or sampler_head_logp.shape != lp.shape:
        raise ValueError("SC requires matching [tokens, k] head tensors")
    if advantages.shape != lp.shape[:1]:
        raise ValueError("SC advantage shape mismatch")
    valid = torch.ones(lp.size(0), dtype=torch.bool, device=lp.device)
    if loss_mask is not None:
        if loss_mask.shape != advantages.shape:
            raise ValueError("SC mask shape mismatch")
        valid = loss_mask.to(device=lp.device, dtype=torch.bool)
    # Sanitize masked rows BEFORE exp, ratio, and multiplication.
    safe_lp = torch.where(valid[:, None], lp, torch.zeros_like(lp))
    with torch.no_grad():
        qlp = torch.where(valid[:, None], _fp(sampler_head_logp.detach()), torch.zeros_like(lp))
        if not torch.isfinite(safe_lp).all() or not torch.isfinite(qlp).all():
            raise ValueError("Nonfinite unmasked SC head")
        p, q = safe_lp.detach().exp(), qlp.exp()
        p = torch.where(valid[:, None], p, torch.zeros_like(p))
        q = torch.where(valid[:, None], q, torch.zeros_like(q))
        pm, qm = p.sum(-1), q.sum(-1)
        if ((pm > 1.0001) | (qm > 1.0001) | (p > 1.0001).any(-1) | (q > 1.0001).any(-1)).any():
            raise ValueError("SC inputs must be full-vocabulary-normalized logprobs")
        pt = (1 - pm).clamp_min(tail_eps)
        qt = (1 - qm).clamp_min(0)
        rho = qt / pt
        alpha = rho
        weight = torch.ones_like(p)
        if formula == "ppo_weighted":
            denominator = qlp if old_head_logp is None else _fp(old_head_logp.detach())
            if denominator.shape != lp.shape:
                raise ValueError("SC old-policy head shape mismatch")
            denominator = torch.where(valid[:, None], denominator, torch.zeros_like(denominator))
            if not torch.isfinite(denominator).all():
                raise ValueError("Nonfinite old-policy SC head")
            old = denominator.exp()
            old = torch.where(valid[:, None], old, torch.zeros_like(old))
            if ((old.sum(-1) > 1.0001) | (old > 1.0001).any(-1)).any():
                raise ValueError("Invalid old-policy probability subdistribution")
            ratio = (safe_lp.detach() - denominator).exp()
            sign = advantages.detach().sign()
            weight = ppo_effective_weight(ratio, sign[:, None], eps_clip_low, eps_clip_high)
            old_tail = (1 - old.sum(-1)).clamp_min(tail_eps)
            tail_weight = ppo_effective_weight(pt / old_tail, sign, eps_clip_low, eps_clip_high)
            alpha = rho * tail_weight
        coefficient = (q * weight - alpha[:, None] * p).detach()
        coefficient = torch.where(valid[:, None], coefficient, torch.zeros_like(coefficient)).detach()
        if not torch.isfinite(coefficient).all():
            raise ValueError("SC ratio/coefficient overflow; abort rather than silently clip it")
        metrics = {
            "sc_sampler_head_mass": qm.detach(),
            "sc_train_head_mass": pm.detach(),
            "sc_alpha": torch.where(valid, alpha, torch.zeros_like(alpha)).detach(),
            "sc_coefficient_l1": coefficient.abs().sum(-1).detach(),
        }
    correction = (coefficient.detach() * safe_lp).sum(-1)
    return correction, metrics


def _tp_info(group):
    if group is None:
        return 0, 1
    if not dist.is_initialized():
        raise RuntimeError("A TP group was supplied before distributed initialization")
    return dist.get_rank(group), dist.get_world_size(group)


def _reduce(tensor, operation, group):
    if group is not None and dist.get_world_size(group) > 1:
        dist.all_reduce(tensor, op=operation, group=group)


class _HeadLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, row_ids, token_ids, group, chunk_size):
        rank, _ = _tp_info(group)
        n, width = token_ids.shape
        vp = logits.size(1)
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        result = torch.empty((n, width), dtype=dtype, device=logits.device)
        maximums = torch.empty((n, 1), dtype=dtype, device=logits.device)
        denominators = torch.empty_like(maximums)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            work = logits.index_select(0, row_ids[start:end]).to(dtype=dtype)
            maximum = work.max(-1, keepdim=True).values
            _reduce(maximum, dist.ReduceOp.MAX, group)
            ids = token_ids[start:end]
            local = (ids >= rank * vp) & (ids < (rank + 1) * vp)
            relative = (ids - rank * vp).clamp(0, vp - 1)
            selected = work.gather(-1, relative).masked_fill(~local, 0)
            _reduce(selected, dist.ReduceOp.SUM, group)
            work.sub_(maximum).exp_()
            denominator = work.sum(-1, keepdim=True)
            _reduce(denominator, dist.ReduceOp.SUM, group)
            result[start:end] = selected - maximum - denominator.log()
            maximums[start:end], denominators[start:end] = maximum, denominator
        ctx.save_for_backward(logits, row_ids, token_ids, maximums, denominators)
        ctx.chunk_size, ctx.vocab_start = chunk_size, rank * vp
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        logits, row_ids, token_ids, maximums, denominators = ctx.saved_tensors
        grad_input = torch.zeros_like(logits)
        vp = logits.size(1)
        dtype = maximums.dtype
        for start in range(0, len(row_ids), ctx.chunk_size):
            end = min(start + ctx.chunk_size, len(row_ids))
            index = row_ids[start:end]
            work = logits.index_select(0, index).to(dtype=dtype)
            work.sub_(maximums[start:end]).exp_().div_(denominators[start:end])
            grad = grad_output[start:end].to(dtype=dtype)
            work.mul_(-grad.sum(-1, keepdim=True))
            ids = token_ids[start:end]
            local = (ids >= ctx.vocab_start) & (ids < ctx.vocab_start + vp)
            relative = (ids - ctx.vocab_start).clamp(0, vp - 1)
            work.scatter_add_(-1, relative, grad.masked_fill(~local, 0))
            grad_input.index_copy_(0, index, work.to(dtype=logits.dtype))
        # The loss is replicated over TP; another all-reduce would multiply its gradient.
        return grad_input, None, None, None, None


def calculate_head_log_probs(logits, row_ids, token_ids, group=None, chunk_size=256):
    if logits.ndim != 2 or not logits.is_floating_point():
        raise ValueError("SC expects flattened floating logits [T, V/TP]")
    if row_ids.dtype != torch.long or row_ids.ndim != 1:
        raise ValueError("SC packed row indices must be int64 [N]")
    if token_ids.dtype != torch.long or token_ids.ndim != 2 or token_ids.size(0) != len(row_ids):
        raise ValueError("SC candidate IDs must be int64 [N,k]")
    if row_ids.device != logits.device or token_ids.device != logits.device:
        raise ValueError("SC gather indices and logits must share a device")
    if chunk_size < 1:
        raise ValueError("SC chunk size must be positive")
    _, world = _tp_info(group)
    if ((row_ids < 0) | (row_ids >= logits.size(0))).any():
        raise ValueError("SC packed row index out of bounds")
    if torch.unique(row_ids).numel() != row_ids.numel():
        raise ValueError("SC packed rows must be unique")
    if ((token_ids < 0) | (token_ids >= world * logits.size(1))).any():
        raise ValueError("SC global candidate token ID out of bounds")
    return _HeadLogProbs.apply(logits, row_ids, token_ids, group, chunk_size)
