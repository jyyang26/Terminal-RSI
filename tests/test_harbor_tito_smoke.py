from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")
    torch_stub.dtype = type("dtype", (), {})
    torch_stub.Size = tuple
    torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch_stub

from slime.utils.types import Sample
from slime_plugins.harbor.atif_adapter import samples_from_rollout_details


NUM_GPUS = 0


@dataclass(frozen=True)
class TurnTitoAudit:
    turn_index: int
    is_exact: bool
    training_prefix: list[int]
    rollout_prompt: list[int]
    common_prefix_len: int


def _base_sample() -> Sample:
    return Sample(index=11, group_index=2, prompt="task")


def _retokenized_detail() -> dict:
    output_ids = list(range(100, 140))
    return {
        "prompt_token_ids": [[10, 11, 900, 901], [10, 11, 200, 40, 900, 901]],
        "completion_token_ids": [output_ids, [50]],
        "logprobs": [[-float(i) / 100 for i in output_ids], [-0.5]],
        "extra": {
            "completion_text": ["same assistant text", "next output"],
            "last_assistant_message_content": [None, "same assistant text"],
            "last_assistant_message_span": [None, [2, 3]],
        },
    }


def _strict_prefix_detail() -> dict:
    return {
        "prompt_token_ids": [[10, 11], [10, 11, 20, 21, 30]],
        "completion_token_ids": [[20, 21], [40]],
        "logprobs": [[-0.2, -0.3], [-0.4]],
    }


def _prompt_len(sample: Sample) -> int:
    return len(sample.tokens) - sample.response_length


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _prefix_before_response_offset(sample: Sample, response_offset: int) -> list[int]:
    return list(sample.tokens[: _prompt_len(sample) + response_offset])


def _two_turn_tito_audit(samples: list[Sample], detail: dict) -> list[TurnTitoAudit]:
    """Audit whether each turn's training prefix equals its rollout prompt.

    This smoke helper is intentionally scoped to the two-turn synthetic fixtures
    below. It keeps the TITO check explicit without needing a model or tokenizer.
    """
    prompts = detail["prompt_token_ids"]
    outputs = detail["completion_token_ids"]
    first_sample = samples[0]
    second_sample = samples[-1]

    first_training_prefix = _prefix_before_response_offset(first_sample, 0)
    second_response_offset = second_sample.response_length - len(outputs[1])
    second_training_prefix = _prefix_before_response_offset(second_sample, second_response_offset)

    audits = [
        TurnTitoAudit(
            turn_index=0,
            is_exact=first_training_prefix == prompts[0],
            training_prefix=first_training_prefix,
            rollout_prompt=list(prompts[0]),
            common_prefix_len=_common_prefix_len(first_training_prefix, prompts[0]),
        ),
        TurnTitoAudit(
            turn_index=1,
            is_exact=second_training_prefix == prompts[1],
            training_prefix=second_training_prefix,
            rollout_prompt=list(prompts[1]),
            common_prefix_len=_common_prefix_len(second_training_prefix, prompts[1]),
        ),
    ]
    return audits


def _shape(samples: list[Sample], detail: dict) -> dict[str, int]:
    audits = _two_turn_tito_audit(samples, detail)
    return {
        "samples": len(samples),
        "total_tokens": sum(len(sample.tokens) for sample in samples),
        "response_tokens": sum(sample.response_length for sample in samples),
        "loss_tokens": sum(sum(sample.loss_mask or []) for sample in samples),
        "tito_exact_turns": sum(audit.is_exact for audit in audits),
        "turns": len(audits),
    }


def test_tito_audit_flags_retokenized_transition_as_non_exact_token_in():
    detail = _retokenized_detail()
    samples = samples_from_rollout_details(
        [detail],
        _base_sample(),
        reward=1.0,
        status=Sample.Status.COMPLETED,
        mode="trajectory_retokenized",
    )

    audits = _two_turn_tito_audit(samples, detail)

    assert len(samples) == 1
    assert [audit.is_exact for audit in audits] == [True, False]
    assert audits[1].common_prefix_len == 2
    assert audits[1].rollout_prompt == [10, 11, 200, 40, 900, 901]
    assert audits[1].training_prefix[:4] == [10, 11, 900, 901]
    assert samples[0].metadata["harbor_trajectory_retokenized_prefix_turns"] == 1
    assert samples[0].metadata["harbor_train_inference_conditioning_match"] is False


def test_mode_ablation_trades_exact_tito_for_fewer_retokenized_chunks():
    detail = _retokenized_detail()

    normalized = samples_from_rollout_details(
        [detail],
        _base_sample(),
        reward=1.0,
        status=Sample.Status.COMPLETED,
        mode="trajectory_normalized",
    )
    retokenized = samples_from_rollout_details(
        [detail],
        _base_sample(),
        reward=1.0,
        status=Sample.Status.COMPLETED,
        mode="trajectory_retokenized",
    )

    normalized_shape = _shape(normalized, detail)
    retokenized_shape = _shape(retokenized, detail)

    assert normalized_shape == {
        "samples": 2,
        "total_tokens": 51,
        "response_tokens": 41,
        "loss_tokens": 41,
        "tito_exact_turns": 2,
        "turns": 2,
    }
    assert retokenized_shape == {
        "samples": 1,
        "total_tokens": 48,
        "response_tokens": 44,
        "loss_tokens": 41,
        "tito_exact_turns": 1,
        "turns": 2,
    }
    assert normalized[0].metadata["harbor_trajectory_num_chunks"] == 2
    assert retokenized[0].metadata["harbor_trajectory_num_chunks"] == 1


def test_conditioning_drift_smoke_quantifies_retokenized_prefix_delta():
    detail = _retokenized_detail()
    sample = samples_from_rollout_details(
        [detail],
        _base_sample(),
        reward=1.0,
        status=Sample.Status.COMPLETED,
        mode="trajectory_retokenized",
    )[0]
    drift = _two_turn_tito_audit([sample], detail)[1]

    train_extra = drift.training_prefix[drift.common_prefix_len :]
    rollout_extra = drift.rollout_prompt[drift.common_prefix_len :]

    assert drift.common_prefix_len == 2
    assert rollout_extra == [200, 40, 900, 901]
    assert train_extra[:2] == [900, 901]
    assert train_extra[2:42] == list(range(100, 140))
    assert 200 not in drift.training_prefix
    assert sample.loss_mask == [1] * 40 + [0, 0, 0, 1]
    assert sample.metadata["harbor_trajectory_retokenized_prompt_history_tokens"] == 1


def test_tito_audit_passes_for_strict_prefix_trajectory():
    detail = _strict_prefix_detail()
    samples = samples_from_rollout_details(
        [detail],
        _base_sample(),
        reward=1.0,
        status=Sample.Status.COMPLETED,
        mode="trajectory_retokenized",
    )

    audits = _two_turn_tito_audit(samples, detail)

    assert len(samples) == 1
    assert [audit.is_exact for audit in audits] == [True, True]
    assert samples[0].metadata["harbor_trajectory_normalized_strict_prefix_turns"] == 1
    assert samples[0].metadata["harbor_train_inference_conditioning_match"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
