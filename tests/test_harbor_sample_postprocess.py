from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

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
from slime_plugins.harbor.sample_postprocess import (
    cap_total_tokens_allow_response_trim_in_samples,
    cap_total_tokens_allow_response_trim_in_rollout_data,
)


NUM_GPUS = 0


def test_hard_cap_samples_updates_response_mask_and_rollout_logprobs():
    sample = Sample(
        index=1,
        tokens=list(range(10)),
        response_length=8,
        loss_mask=[1, 0, 0, 1, 1, 0, 1, 1],
        rollout_log_probs=[float(-i) for i in range(8)],
        rollout_routed_experts=list(range(9)),
        metadata={},
    )

    cap_total_tokens_allow_response_trim_in_samples(
        SimpleNamespace(harbor_max_train_total_tokens=6),
        [[sample]],
    )

    assert sample.tokens == [4, 5, 6, 7, 8, 9]
    assert sample.response_length == 5
    assert sample.loss_mask == [1, 1, 0, 1, 1]
    assert sample.rollout_log_probs == [-3.0, -4.0, -5.0, -6.0, -7.0]
    assert sample.rollout_routed_experts == [4, 5, 6, 7, 8]
    assert len(sample.rollout_routed_experts) == len(sample.tokens) - 1
    assert sample.metadata["harbor_response_trimmed_tokens"] == 3
    assert len(sample.loss_mask) == sample.response_length
    assert len(sample.rollout_log_probs) == sample.response_length


def test_hard_cap_rollout_data_updates_lengths_and_masks():
    rollout_data = {
        "tokens": [list(range(10))],
        "response_lengths": [8],
        "total_lengths": [10],
        "loss_masks": [[1, 0, 0, 1, 1, 0, 1, 1]],
        "rollout_log_probs": [[float(-i) for i in range(8)]],
        "rollout_routed_experts": [list(range(9))],
        "metadata": [{}],
    }

    cap_total_tokens_allow_response_trim_in_rollout_data(
        SimpleNamespace(harbor_max_train_total_tokens=6),
        rollout_id=3,
        rollout_data=rollout_data,
    )

    assert rollout_data["tokens"][0] == [4, 5, 6, 7, 8, 9]
    assert rollout_data["response_lengths"][0] == 5
    assert rollout_data["total_lengths"][0] == 6
    assert rollout_data["loss_masks"][0] == [1, 1, 0, 1, 1]
    assert rollout_data["rollout_log_probs"][0] == [-3.0, -4.0, -5.0, -6.0, -7.0]
    assert rollout_data["rollout_routed_experts"][0] == [4, 5, 6, 7, 8]
    assert len(rollout_data["rollout_routed_experts"][0]) == len(rollout_data["tokens"][0]) - 1
    assert rollout_data["metadata"][0]["harbor_response_trimmed_tokens"] == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
