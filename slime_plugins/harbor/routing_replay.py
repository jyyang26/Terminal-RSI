from __future__ import annotations

import base64
from typing import Any

import numpy as np

try:
    import pybase64 as _base64
except ModuleNotFoundError:
    _base64 = base64


def validate_routed_experts_payload(
    encoded: str,
    *,
    token_rows: int,
    num_layers: int,
    moe_router_topk: int,
) -> list[int]:
    """Validate an SGLang routed_experts payload and return its decoded shape."""
    if token_rows < 0:
        raise ValueError(f"Invalid routed experts token row count: {token_rows}")
    if num_layers <= 0 or moe_router_topk <= 0:
        raise ValueError(
            f"num_layers and moe_router_topk must be positive, got {num_layers=} {moe_router_topk=}"
        )

    raw = _decode_routed_experts_bytes(encoded)
    if len(raw) % np.dtype(np.int32).itemsize != 0:
        raise ValueError(f"Decoded routed_experts byte length is not int32-aligned: {len(raw)}")

    num_ints = len(raw) // np.dtype(np.int32).itemsize
    if token_rows == 0:
        if num_ints != 0:
            raise ValueError(
                "Decoded routed_experts has data for an empty token axis: "
                f"num_ints={num_ints}, token_rows={token_rows}"
            )
        return [0, num_layers, moe_router_topk]

    denominator = token_rows * moe_router_topk
    if num_ints % denominator != 0:
        raise ValueError(
            "Decoded routed_experts size does not match token/topk axes: "
            f"num_ints={num_ints}, token_rows={token_rows}, topk={moe_router_topk}"
        )

    actual_layers = num_ints // denominator
    if actual_layers < num_layers:
        raise ValueError(
            "Decoded routed_experts has fewer layers than the training model expects: "
            f"actual={actual_layers}, expected={num_layers}"
        )
    return [token_rows, actual_layers, moe_router_topk]


def make_routed_experts_extra(
    encoded: str,
    *,
    token_rows: int,
    num_layers: int,
    moe_router_topk: int,
) -> dict[str, Any]:
    shape = validate_routed_experts_payload(
        encoded,
        token_rows=token_rows,
        num_layers=num_layers,
        moe_router_topk=moe_router_topk,
    )
    return {
        "routed_experts": encoded,
        "routed_experts_shape": shape,
        "routed_experts_num_layers": int(num_layers),
        "routed_experts_moe_router_topk": int(moe_router_topk),
    }


def decode_routed_experts_payload(
    encoded: str,
    *,
    shape: list[int] | tuple[int, int, int],
    num_layers: int,
) -> np.ndarray:
    token_rows, actual_layers, topk = [int(v) for v in shape]
    raw = _decode_routed_experts_bytes(encoded)
    experts = np.frombuffer(raw, dtype=np.int32).reshape(token_rows, actual_layers, topk)
    if actual_layers > num_layers:
        experts = experts[:, :num_layers, :]
    return np.ascontiguousarray(experts)


class RoutingRowsBuilder:
    """Incrementally build rows aligned to a final token sequence.

    Row ``j`` stores the experts used to predict token ``j + 1``. Appending
    ``token_count`` tokens therefore appends ``token_count`` routing rows, except
    for the initial prompt seed which appends ``prompt_len - 1`` rows.
    """

    def __init__(self) -> None:
        self.rows: list[np.ndarray] = []
        self.placeholder_rows = 0
        self.placeholder_loss_rows = 0
        self._row_shape: tuple[int, ...] | None = None

    def seed_prompt(self, routed_experts: np.ndarray | None, prompt_len: int) -> None:
        if prompt_len <= 1:
            return
        self.append_from_source(routed_experts, source_start=1, token_count=prompt_len - 1)

    def append_from_source(
        self,
        routed_experts: np.ndarray | None,
        *,
        source_start: int,
        token_count: int,
        placeholder_first: bool = False,
        first_token_loss: bool = False,
    ) -> None:
        if token_count <= 0:
            return
        if routed_experts is not None:
            self._row_shape = tuple(routed_experts.shape[1:])

        remaining = token_count
        source_row_start = source_start - 1
        if placeholder_first:
            self._append_placeholder(first_token_loss=first_token_loss)
            remaining -= 1
            source_row_start = source_start
        if remaining <= 0:
            return

        if routed_experts is None:
            raise ValueError("routed_experts is required to append non-placeholder routing rows")
        if source_row_start < 0 or source_row_start + remaining > routed_experts.shape[0]:
            raise ValueError(
                "Routing row slice is out of bounds: "
                f"start={source_row_start}, count={remaining}, shape={routed_experts.shape}"
            )
        self.rows.extend(
            np.asarray(row, dtype=np.int32).copy()
            for row in routed_experts[source_row_start : source_row_start + remaining]
        )

    def as_array(self) -> np.ndarray | None:
        if not self.rows:
            return None
        return np.ascontiguousarray(np.stack(self.rows, axis=0).astype(np.int32, copy=False))

    def _append_placeholder(self, *, first_token_loss: bool) -> None:
        self.placeholder_rows += 1
        if first_token_loss:
            self.placeholder_loss_rows += 1
            raise ValueError("Unsafe routing replay placeholder would predict a loss-masked token")
        if self.rows:
            self.rows.append(np.asarray(self.rows[-1], dtype=np.int32).copy())
            return
        if self._row_shape is None:
            raise ValueError("Cannot create routing replay placeholder before row shape is known")
        self.rows.append(np.zeros(self._row_shape, dtype=np.int32))


def _decode_routed_experts_bytes(encoded: str) -> bytes:
    if not isinstance(encoded, str):
        raise TypeError(f"routed_experts must be a base64 string, got {type(encoded)!r}")
    try:
        return _base64.b64decode(encoded.encode("ascii"))
    except Exception as exc:
        raise ValueError("SGLang returned non-base64 routed_experts") from exc
