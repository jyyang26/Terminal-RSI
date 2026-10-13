from __future__ import annotations

import sys
import types
from types import SimpleNamespace

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.dtype = type("dtype", (), {})
    torch_stub.Size = tuple
    torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch_stub

from slime.utils.types import Sample
from slime_plugins.harbor.atif_adapter import (
    RolloutConversionError,
    explain_rollout_details_conversion,
    samples_from_rollout_details,
    samples_from_trial_result,
)


NUM_GPUS = 0


def _base_sample() -> Sample:
    return Sample(index=7, group_index=3, prompt="task", metadata={"source": "unit"})


def _detail():
    return {
        "prompt_token_ids": [[10, 11], [10, 11, 20, 30, 31]],
        "completion_token_ids": [[20], [40, 41]],
        "logprobs": [[-0.5], [-0.7, -0.8]],
        "extra": {"finish_reason": [{"type": "stop"}, {"type": "length"}]},
    }


def test_turn_mode_converts_each_llm_call_to_sample():
    samples = samples_from_rollout_details(
        [_detail()],
        _base_sample(),
        reward=1.0,
        status=Sample.Status.COMPLETED,
        mode="turn",
    )

    assert len(samples) == 2
    assert samples[0].tokens == [10, 11, 20]
    assert samples[0].response_length == 1
    assert samples[0].loss_mask == [1]
    assert samples[0].reward == 1.0
    assert samples[0].status == Sample.Status.COMPLETED
    assert samples[0].rollout_log_probs == [-0.5]
    assert samples[0].metadata["harbor_turn_index"] == 0
    assert samples[0].metadata["harbor_num_turns"] == 2
    assert samples[0].metadata["harbor_length_finish_turns"] == 1
    assert samples[0].metadata["harbor_finish_reason_type"] == "stop"
    assert samples[0].group_id is None

    assert samples[1].tokens == [10, 11, 20, 30, 31, 40, 41]
    assert samples[1].response_length == 2
    assert samples[1].loss_mask == [1, 1]
    assert samples[1].rollout_log_probs == [-0.7, -0.8]
    assert samples[1].metadata["harbor_finish_reason_type"] == "length"


def test_turn_grouped_mode_marks_turn_samples_for_trajectory_grouping():
    samples = samples_from_rollout_details(
        [_detail()],
        _base_sample(),
        reward=1.0,
        status=Sample.Status.COMPLETED,
        mode="turn_grouped",
        metadata={"harbor_rollout_id": 4, "harbor_rollout_group_index": 5},
    )

    assert len(samples) == 2
    assert samples[0].metadata["harbor_sample_mode"] == "turn_grouped"
    assert samples[0].metadata["harbor_rollout_id"] == 4
    assert samples[0].metadata["harbor_rollout_group_index"] == 5
    assert samples[0].metadata["harbor_terminal_reward"] == 1.0
    assert samples[1].metadata["harbor_turn_index"] == 1
    assert samples[0].group_id is not None
    assert samples[0].group_id == samples[1].group_id


def test_step_mode_uses_each_rollout_prompt_snapshot_and_groups_trajectory():
    detail = {
        "prompt_token_ids": [[10, 11], [50, 51, 52]],
        "completion_token_ids": [[20], [60, 61]],
        "logprobs": [[-0.5], [-0.7, -0.8]],
    }

    samples = samples_from_rollout_details(
        [detail],
        _base_sample(),
        reward=0.75,
        status=Sample.Status.COMPLETED,
        mode="step",
        metadata={"harbor_rollout_id": 4, "harbor_rollout_group_index": 5, "harbor_sample_rank": 6},
    )

    assert len(samples) == 2
    assert samples[0].tokens == [10, 11, 20]
    assert samples[0].response_length == 1
    assert samples[0].loss_mask == [1]
    assert samples[0].rollout_log_probs == [-0.5]

    assert samples[1].tokens == [50, 51, 52, 60, 61]
    assert samples[1].response_length == 2
    assert samples[1].loss_mask == [1, 1]
    assert samples[1].rollout_log_probs == [-0.7, -0.8]

    assert samples[0].group_id == samples[1].group_id
    assert samples[0].group_id == 4_000_500_060_000
    assert samples[0].metadata["harbor_sample_mode"] == "step"
    assert samples[0].metadata["harbor_step_index"] == 0
    assert samples[1].metadata["harbor_step_index"] == 1
    assert samples[1].metadata["harbor_num_steps"] == 2
    assert samples[1].metadata["harbor_conditioning_mode"] == "prompt_snapshot"
    assert samples[1].metadata["harbor_step_grouping"] == "trajectory"
    assert samples[1].metadata["harbor_train_inference_conditioning_match"] is True


def test_step_grouped_mode_is_explicit_grouped_step_alias():
    samples = samples_from_rollout_details(
        [_detail()],
        _base_sample(),
        reward=1.0,
        status=Sample.Status.COMPLETED,
        mode="step_grouped",
        metadata={"harbor_rollout_id": 1, "harbor_rollout_group_index": 2},
    )

    assert len(samples) == 2
    assert samples[0].metadata["harbor_sample_mode"] == "step_grouped"
    assert samples[0].metadata["harbor_conditioning_mode"] == "prompt_snapshot"
    assert samples[0].group_id is not None
    assert samples[0].group_id == samples[1].group_id


def test_last_turn_mode_keeps_only_final_llm_call():
    samples = samples_from_rollout_details(
        [_detail()],
        _base_sample(),
        reward=0.25,
        status=Sample.Status.COMPLETED,
        mode="last_turn",
    )

    assert len(samples) == 1
    assert samples[0].tokens == [10, 11, 20, 30, 31, 40, 41]
    assert samples[0].response_length == 2
    assert samples[0].loss_mask == [1, 1]
    assert samples[0].reward == 0.25
    assert samples[0].metadata["harbor_length_finish_turns"] == 1
    assert samples[0].metadata["harbor_finish_reason_type"] == "length"


def test_trajectory_mode_masks_environment_observation_deltas():
    samples = samples_from_rollout_details(
        [_detail()],
        _base_sample(),
        reward=1.0,
        status=Sample.Status.COMPLETED,
        mode="trajectory",
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.tokens == [10, 11, 20, 30, 31, 40, 41]
    assert sample.response_length == 5
    assert sample.loss_mask == [1, 0, 0, 1, 1]
    assert sample.rollout_log_probs == [-0.5, 0.0, 0.0, -0.7, -0.8]
    assert len(sample.loss_mask) == sample.response_length
    assert sample.metadata["harbor_sample_mode"] == "trajectory"
    assert sample.metadata["harbor_length_finish_turns"] == 1
    assert sample.metadata["harbor_finish_reason_type"] == "length"
    assert sample.group_id is not None


def test_trajectory_mode_preserves_all_outputs_across_prompt_drift():
    drift_detail = {
        "prompt_token_ids": [[10], [10, 11, 21], [10, 11, 22]],
        "completion_token_ids": [[11], [12], [13]],
        "logprobs": [[-0.11], [-0.12], [-0.13]],
    }

    samples = samples_from_rollout_details(
        [drift_detail],
        _base_sample(),
        reward=0.0,
        status=Sample.Status.FAILED,
        mode="trajectory",
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.tokens == [10, 11, 21, 12, 22, 13]
    assert sample.response_length == 5
    assert sample.loss_mask == [1, 0, 1, 0, 1]
    assert sample.rollout_log_probs == [-0.11, 0.0, -0.12, 0.0, -0.13]
    assert sample.status == Sample.Status.FAILED
    assert sample.metadata["harbor_trajectory_prefix_mismatch_turns"] == 1
    assert sample.metadata["harbor_trajectory_prefix_mismatch_first_turn"] == 1
    assert sample.metadata["harbor_trajectory_stitch_prefix_tail_turns"] == 1
    assert sample.metadata["harbor_trajectory_stitch_suffix_prompt_turns"] == 1
    assert sample.metadata["harbor_trajectory_stitch_output_tokens"] == 3


def test_trajectory_mode_preserves_outputs_when_prompt_base_changes():
    base_changed_detail = {
        "prompt_token_ids": [[10, 11], [50, 51, 52]],
        "completion_token_ids": [[20], [60, 61]],
        "logprobs": [[-0.5], [-0.7, -0.8]],
    }

    samples = samples_from_rollout_details(
        [base_changed_detail],
        _base_sample(),
        reward=0.0,
        status=Sample.Status.COMPLETED,
        mode="trajectory",
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.tokens == [10, 11, 20, 50, 51, 52, 60, 61]
    assert sample.response_length == 6
    assert sample.loss_mask == [1, 0, 0, 0, 1, 1]
    assert sample.rollout_log_probs == [-0.5, 0.0, 0.0, 0.0, -0.7, -0.8]
    assert sample.metadata["harbor_trajectory_stitch_full_prompt_turns"] == 1
    assert sample.metadata["harbor_trajectory_stitch_context_tokens"] == 3
    assert sample.metadata["harbor_trajectory_stitch_output_tokens"] == 3


def test_trajectory_normalized_mode_reuses_template_normalized_history():
    detail = {
        "prompt_token_ids": [[10, 11, 900, 901], [10, 11, 20, 30, 40, 900, 901]],
        "completion_token_ids": [[20, 30], [50]],
        "logprobs": [[-0.2, -0.3], [-0.5]],
    }

    samples = samples_from_rollout_details(
        [detail],
        _base_sample(),
        reward=1.0,
        status=Sample.Status.COMPLETED,
        mode="trajectory_normalized",
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.tokens == [10, 11, 900, 901, 20, 30, 40, 900, 901, 50]
    assert sample.response_length == 6
    assert sample.loss_mask == [1, 1, 0, 0, 0, 1]
    assert sample.rollout_log_probs == [-0.2, -0.3, 0.0, 0.0, 0.0, -0.5]
    assert sample.metadata["harbor_sample_mode"] == "trajectory_normalized"
    assert sample.metadata["harbor_trajectory_num_chunks"] == 1
    assert sample.metadata["harbor_trajectory_normalized_prefix_turns"] == 1
    assert sample.metadata["harbor_trajectory_normalized_prompt_suffix_dropped_tokens"] == 2
    assert sample.metadata["harbor_trajectory_normalized_context_tokens"] == 3
    assert sample.metadata["harbor_train_inference_conditioning_match"] is False


def test_trajectory_normalized_mode_handles_leading_response_artifact():
    detail = {
        "prompt_token_ids": [[10, 900, 901], [10, 20, 30, 40, 900, 901]],
        "completion_token_ids": [[777, 20, 30], [50]],
        "logprobs": [[-0.7, -0.2, -0.3], [-0.5]],
    }

    samples = samples_from_rollout_details(
        [detail],
        _base_sample(),
        reward=1.0,
        status=Sample.Status.COMPLETED,
        mode="trajectory_normalized",
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.tokens == [10, 900, 901, 777, 20, 30, 40, 900, 901, 50]
    assert sample.loss_mask == [1, 1, 1, 0, 0, 0, 1]
    assert sample.rollout_log_probs == [-0.7, -0.2, -0.3, 0.0, 0.0, 0.0, -0.5]
    assert sample.metadata["harbor_trajectory_normalized_response_prefix_dropped_tokens"] == 1


def test_trajectory_normalized_mode_splits_unresolved_context_resets():
    detail = {
        "prompt_token_ids": [[10, 11, 900, 901], [70, 71, 900, 901]],
        "completion_token_ids": [[20], [80]],
        "logprobs": [[-0.2], [-0.8]],
    }

    samples = samples_from_rollout_details(
        [detail],
        _base_sample(),
        reward=0.5,
        status=Sample.Status.COMPLETED,
        mode="trajectory_normalized",
        metadata={"harbor_rollout_id": 4, "harbor_rollout_group_index": 5, "harbor_sample_rank": 6},
    )

    assert len(samples) == 2
    assert samples[0].tokens == [10, 11, 900, 901, 20]
    assert samples[0].loss_mask == [1]
    assert samples[0].metadata["harbor_trajectory_chunk_index"] == 0
    assert samples[0].metadata["harbor_trajectory_num_chunks"] == 2
    assert samples[0].metadata["harbor_trajectory_chunk_started_after_split"] is False
    assert samples[0].metadata["harbor_trajectory_normalized_split_after_turns"] == 1
    assert samples[0].metadata["harbor_trajectory_normalized_split_before_turn"] == 1

    assert samples[1].tokens == [70, 71, 900, 901, 80]
    assert samples[1].loss_mask == [1]
    assert samples[1].metadata["harbor_trajectory_chunk_index"] == 1
    assert samples[1].metadata["harbor_trajectory_chunk_started_after_split"] is True
    assert samples[0].group_id == samples[1].group_id
    assert samples[0].group_id == 4_000_500_060_000


def test_trajectory_retokenized_mode_reuses_rendered_history_span():
    output_ids = list(range(100, 140))
    output_logprobs = [-float(i) / 100 for i in range(100, 140)]
    detail = {
        "prompt_token_ids": [[10, 11, 900, 901], [10, 11, 200, 40, 900, 901]],
        "completion_token_ids": [output_ids, [50]],
        "logprobs": [output_logprobs, [-0.5]],
        "extra": {
            "completion_text": ["same assistant text", "next output"],
            "last_assistant_message_content": [None, "same assistant text"],
            "last_assistant_message_span": [None, [2, 3]],
        },
    }

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

    assert len(normalized) == 2
    assert len(retokenized) == 1
    sample = retokenized[0]
    assert sample.tokens == [10, 11, 900, 901, *output_ids, 40, 900, 901, 50]
    assert sample.response_length == 44
    assert sample.loss_mask == [1] * 40 + [0, 0, 0, 1]
    assert sample.rollout_log_probs == output_logprobs + [0.0, 0.0, 0.0, -0.5]
    assert sample.metadata["harbor_sample_mode"] == "trajectory_retokenized"
    assert sample.metadata["harbor_conditioning_mode"] == "retokenized_trajectory"
    assert sample.metadata["harbor_trajectory_num_chunks"] == 1
    assert sample.metadata["harbor_trajectory_retokenized_prefix_turns"] == 1
    assert sample.metadata["harbor_trajectory_retokenized_prompt_history_tokens"] == 1
    assert sample.metadata["harbor_trajectory_normalized_split_after_turns"] == 0
    assert sample.metadata["harbor_train_inference_conditioning_match"] is False


def test_trajectory_retokenized_mode_splits_when_history_text_was_rewritten():
    output_ids = list(range(100, 140))
    detail = {
        "prompt_token_ids": [[10, 11, 900, 901], [10, 11, 200, 40, 900, 901]],
        "completion_token_ids": [output_ids, [50]],
        "logprobs": [[-float(i) / 100 for i in range(100, 140)], [-0.5]],
        "extra": {
            "completion_text": ["raw assistant text", "next output"],
            "last_assistant_message_content": [None, "rewritten assistant text"],
            "last_assistant_message_span": [None, [2, 3]],
        },
    }

    samples = samples_from_rollout_details(
        [detail],
        _base_sample(),
        reward=1.0,
        status=Sample.Status.COMPLETED,
        mode="trajectory_retokenized",
    )

    assert len(samples) == 2
    assert samples[0].tokens == [10, 11, 900, 901, *output_ids]
    assert samples[0].metadata["harbor_trajectory_normalized_split_before_turn"] == 1
    assert samples[1].tokens == [10, 11, 200, 40, 900, 901, 50]


def test_conversion_rejects_mismatched_logprob_lengths():
    detail = {
        "prompt_token_ids": [[10, 11]],
        "completion_token_ids": [[20, 30]],
        "logprobs": [[-0.2]],
    }

    try:
        samples_from_rollout_details(
            [detail],
            _base_sample(),
            reward=1.0,
            status=Sample.Status.COMPLETED,
            mode="trajectory_retokenized",
        )
    except RolloutConversionError as exc:
        assert "mismatched logprobs/completion length" in str(exc)
    else:
        raise AssertionError("expected RolloutConversionError")


def test_trajectory_explanation_records_turn_records_and_mask_alignment():
    drift_detail = {
        "prompt_token_ids": [[10], [10, 11, 21], [10, 11, 22]],
        "completion_token_ids": [[11], [12], [13]],
        "logprobs": [[-0.11], [-0.12], [-0.13]],
    }
    converted_samples = samples_from_rollout_details(
        [drift_detail],
        _base_sample(),
        reward=0.0,
        status=Sample.Status.FAILED,
        mode="trajectory",
    )

    explanation = explain_rollout_details_conversion(
        [drift_detail],
        _base_sample(),
        reward=0.0,
        status=Sample.Status.FAILED,
        mode="trajectory",
        converted_samples=converted_samples,
    )

    detail = explanation["details"][0]
    assert detail["raw_detail"]["prompt_lengths"] == [1, 3, 3]
    assert detail["turn_records"][1]["prompt_ids"] == [10, 11, 21]
    assert detail["alignment"][2]["prefix_drift"] is True
    assert detail["stitched_alignment"][1]["strategy"] == "prefix_tail"
    assert detail["stitched_alignment"][2]["strategy"] == "prompt_suffix_tail"
    assert detail["prefix_mismatch_turns"] == [1]
    assert detail["v030_merged_segment"]["response_ids"] == [11, 22, 13]
    assert detail["v030_merged_segment"]["loss_mask"] == [1, 0, 1]
    assert detail["merged_segment"]["response_ids"] == [11, 21, 12, 22, 13]
    assert detail["merged_segment"]["loss_mask"] == [1, 0, 1, 0, 1]
    assert detail["sample"]["loss_mask_sum"] == 3
    assert explanation["converted_samples"][0]["loss_mask"] == [1, 0, 1, 0, 1]


def test_trial_result_conversion_extracts_reward_status_and_metadata():
    result = SimpleNamespace(
        trial_name="trial-a",
        task_name="task-a",
        trial_uri="file:///tmp/trial-a",
        verifier_result=SimpleNamespace(rewards={"reward": 0.5}),
        exception_info=None,
        agent_result=SimpleNamespace(rollout_details=[_detail()]),
        step_results=None,
    )

    samples = samples_from_trial_result(result, _base_sample(), mode="last_turn")

    assert len(samples) == 1
    assert samples[0].reward == 0.5
    assert samples[0].status == Sample.Status.COMPLETED
    assert samples[0].metadata["harbor_trial_name"] == "trial-a"
    assert samples[0].metadata["harbor_task_name"] == "task-a"


def test_trial_result_propagates_summarization_count_and_total_output_tokens():
    # _detail() has two turns with completion lengths 1 and 2 -> 3 output tokens.
    result = SimpleNamespace(
        trial_name="trial-a",
        task_name="task-a",
        trial_uri="file:///tmp/trial-a",
        verifier_result=SimpleNamespace(rewards={"reward": 1.0}),
        exception_info=None,
        agent_result=SimpleNamespace(
            rollout_details=[_detail()],
            metadata={"summarization_count": 2, "n_episodes": 9},
        ),
        step_results=None,
    )

    samples = samples_from_trial_result(result, _base_sample(), mode="trajectory")

    assert len(samples) == 1
    assert samples[0].metadata["harbor_summarization_count"] == 2
    assert samples[0].metadata["harbor_num_turns"] == 2
    assert samples[0].metadata["harbor_total_output_tokens"] == 3


def test_trial_result_summarization_count_absent_when_agent_did_not_report():
    result = SimpleNamespace(
        trial_name="trial-a",
        task_name="task-a",
        trial_uri="file:///tmp/trial-a",
        verifier_result=SimpleNamespace(rewards={"reward": 0.0}),
        exception_info=None,
        agent_result=SimpleNamespace(rollout_details=[_detail()]),  # no metadata attr
        step_results=None,
    )

    samples = samples_from_trial_result(result, _base_sample(), mode="last_turn")

    assert len(samples) == 1
    assert "harbor_summarization_count" not in samples[0].metadata
    # total_output_tokens is still emitted from the rollout detail.
    assert samples[0].metadata["harbor_total_output_tokens"] == 3

