from __future__ import annotations

import base64
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.dtype = type("dtype", (), {})
    torch_stub.Size = tuple
    torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch_stub

from slime.utils.types import Sample
from slime_plugins.harbor.atif_adapter import RolloutConversionError, samples_from_rollout_details


NUM_GPUS = 0


def _base_sample() -> Sample:
    return Sample(index=3, group_index=1, prompt="task", metadata={})


def _encoded_routing(rows: int, *, base: int) -> tuple[np.ndarray, str, list[int]]:
    arr = np.empty((rows, 2, 2), dtype=np.int32)
    for row in range(rows):
        arr[row, :, :] = base + row
    return arr, base64.b64encode(arr.tobytes()).decode("ascii"), list(arr.shape)


def _with_routing(detail: dict, routings: list[tuple[np.ndarray, str, list[int]]]) -> dict:
    detail = dict(detail)
    detail["extra"] = {
        **detail.get("extra", {}),
        "routed_experts": [encoded for _, encoded, _ in routings],
        "routed_experts_shape": [shape for _, _, shape in routings],
        "routed_experts_num_layers": [2 for _ in routings],
    }
    return detail


def _row_values(routed_experts: np.ndarray) -> list[int]:
    return [int(row[0, 0]) for row in routed_experts]


def test_r3_strict_trajectory_retokenized_keeps_exact_rows():
    r0 = _encoded_routing(3, base=0)
    r1 = _encoded_routing(5, base=100)
    detail = _with_routing(
        {
            "prompt_token_ids": [[10, 11], [10, 11, 20, 21, 30]],
            "completion_token_ids": [[20, 21], [40]],
            "logprobs": [[-0.2, -0.3], [-0.4]],
        },
        [r0, r1],
    )

    samples = samples_from_rollout_details(
        [detail],
        _base_sample(),
        reward=1.0,
        status=Sample.Status.COMPLETED,
        mode="trajectory_retokenized",
        require_routing_replay=True,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.tokens == [10, 11, 20, 21, 30, 40]
    assert sample.loss_mask == [1, 1, 0, 1]
    assert _row_values(sample.rollout_routed_experts) == [0, 1, 2, 103, 104]
    assert sample.rollout_routed_experts.shape[0] == len(sample.tokens) - 1


def test_r3_retokenized_context_uses_safe_placeholder_for_dropped_boundary():
    output_ids = list(range(100, 140))
    r0 = _encoded_routing(41, base=0)
    r1 = _encoded_routing(6, base=100)
    detail = _with_routing(
        {
            "prompt_token_ids": [[1, 2], [1, 2, 200, 201, 30, 31]],
            "completion_token_ids": [output_ids, [40]],
            "logprobs": [[-float(i) / 100 for i in output_ids], [-0.3]],
            "extra": {
                "completion_text": ["hello", "done"],
                "last_assistant_message_content": [None, "hello"],
                "last_assistant_message_span": [None, [2, 4]],
            },
        },
        [r0, r1],
    )

    sample = samples_from_rollout_details(
        [detail],
        _base_sample(),
        reward=1.0,
        status=Sample.Status.COMPLETED,
        mode="trajectory_retokenized",
        require_routing_replay=True,
    )[0]

    assert sample.tokens == [1, 2, *output_ids, 30, 31, 40]
    assert sample.loss_mask == [1] * len(output_ids) + [0, 0, 1]
    assert _row_values(sample.rollout_routed_experts) == [*range(41), 40, 104, 105]
    assert sample.metadata["harbor_routing_replay_placeholder_rows"] == 1
    assert sample.metadata["harbor_routing_replay_placeholder_loss_rows"] == 0


def test_r3_retokenized_without_context_rejects_loss_placeholder():
    output_ids = list(range(100, 140))
    r0 = _encoded_routing(41, base=0)
    r1 = _encoded_routing(4, base=100)
    detail = _with_routing(
        {
            "prompt_token_ids": [[1, 2], [1, 2, 200, 201]],
            "completion_token_ids": [output_ids, [40]],
            "logprobs": [[-float(i) / 100 for i in output_ids], [-0.3]],
            "extra": {
                "completion_text": ["hello", "done"],
                "last_assistant_message_content": [None, "hello"],
                "last_assistant_message_span": [None, [2, 4]],
            },
        },
        [r0, r1],
    )

    with pytest.raises(RolloutConversionError, match="placeholder"):
        samples_from_rollout_details(
            [detail],
            _base_sample(),
            reward=1.0,
            status=Sample.Status.COMPLETED,
            mode="trajectory_retokenized",
            require_routing_replay=True,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
