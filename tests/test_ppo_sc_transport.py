"""Harbor wire, response identity, true trimming and unchanged-off transport contracts."""

import asyncio
import base64
import copy
import json
import sys
import types
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from slime.utils.score_centering import (
    HEAD_KEYS,
    assert_sampled_logps,
    decode_sampler_head,
    decode_turn_head,
    encode_turn_head,
    score_centering_request,
    validate_head,
)
from slime.utils.types import Sample
from slime_plugins.harbor.atif_adapter import samples_from_rollout_details
from slime_plugins.harbor.sample_postprocess import (
    cap_total_tokens_allow_response_trim_in_samples,
    cap_total_tokens_allow_response_trim_in_rollout_data,
)

NUM_GPUS = 0
pytestmark = pytest.mark.unit


def head_fixture():
    ids = np.array([[3, 0], [4, 0]], dtype="<i4")
    logps = np.array([[-0.5, -10], [-0.5, -10]], dtype="<f4")
    tokens = np.array([3, 4], dtype="<i4")
    return ids, logps, tokens


def meta_fixture(wire):
    ids, logps, tokens = head_fixture()
    if wire == "binary":
        return dict(
            output_topk_shape=[2, 2],
            output_topk_token_ids=base64.b64encode(ids.tobytes()).decode(),
            output_topk_log_probs=base64.b64encode(logps.tobytes()).decode(),
        )
    return {
        "output_top_logprobs": [
            [[float(lp), int(token), None] for lp, token in zip(lps, row, strict=True)]
            for row, lps in zip(ids, logps, strict=True)
        ]
    }


@pytest.mark.parametrize("wire", ["binary", "json"])
def test_wire_roundtrip_and_envelope(wire):
    expected = head_fixture()
    actual = decode_sampler_head(meta_fixture(wire), [3, 4], k=2, wire_format=wire, max_bytes=1024, vocab_size=10)
    assert all(np.array_equal(a, b) for a, b in zip(expected, actual, strict=True))
    envelope = encode_turn_head(*actual, [1, 2])
    json.dumps(envelope)
    decoded = decode_turn_head(envelope, [1, 2], [3, 4], [-0.5, -0.5], k=2)
    assert all(np.array_equal(a, b) for a, b in zip(expected, decoded, strict=True))


@pytest.mark.parametrize(
    "change",
    [
        {"output_topk_shape": [3, 2]},
        {"output_topk_token_ids": "bad"},
        {"output_topk_log_probs": "!" * 24},
        {"output_topk_shape": [2, 3]},
    ],
)
def test_corrupt_binary_is_rejected(change):
    meta = meta_fixture("binary")
    meta.update(change)
    with pytest.raises(ValueError):
        decode_sampler_head(meta, [3, 4], k=2, wire_format="binary", max_bytes=1024, vocab_size=10)


@pytest.mark.parametrize(
    "bad_ids,bad_lp",
    [
        ([[3, 3], [4, 0]], None),
        ([[-1, 0], [4, 0]], None),
        ([[10, 0], [4, 0]], None),
        (None, [[0.0, 0.0], [-0.5, -10]]),
        (None, [[float("nan"), -10], [-0.5, -10]]),
        (None, [[float("-inf"), -10], [-0.5, -10]]),
    ],
)
def test_invalid_subdistribution_rejected(bad_ids, bad_lp):
    ids, lp, tokens = head_fixture()
    if bad_ids is not None:
        ids = np.array(bad_ids, dtype="<i4")
    if bad_lp is not None:
        lp = np.array(bad_lp, dtype="<f4")
    with pytest.raises(ValueError):
        validate_head(ids, lp, tokens, k=2, vocab_size=10)


def test_tail_sample_not_in_topk_is_valid():
    ids, lp, _ = head_fixture()
    assert_sampled_logps(ids, lp, np.array([8, 9]), [-5.0, -6.0])


@pytest.mark.parametrize(
    "key,value",
    [
        ("temperature", 0.9),
        ("top_p", 0.9),
        ("top_k", 10),
        ("min_p", 0.1),
        ("presence_penalty", 1.0),
        ("logit_bias", {"1": 2}),
    ],
)
def test_actual_request_distribution_is_checked(key, value):
    with pytest.raises(ValueError):
        score_centering_request(k=2, sampling_params={"max_new_tokens": 2, key: value}, max_bytes=1000)


def test_limits_are_checked_before_decode_or_generation():
    with pytest.raises(ValueError):
        score_centering_request(k=32, sampling_params={"max_new_tokens": 10000}, max_bytes=1000)
    with pytest.raises(ValueError):
        decode_sampler_head({}, [3, 4], k=2, wire_format="binary", max_bytes=1, vocab_size=10)


FIXTURES = [
    {
        "prompt_token_ids": [[10, 11, 900, 901], [10, 11, 20, 30, 40, 900, 901]],
        "completion_token_ids": [[20, 30], [50]],
        "logprobs": [[-0.2, -0.3], [-0.5]],
    },
    {
        "prompt_token_ids": [[10, 900, 901], [10, 20, 30, 40, 900, 901]],
        "completion_token_ids": [[777, 20, 30], [50]],
        "logprobs": [[-0.7, -0.2, -0.3], [-0.5]],
    },
    {
        "prompt_token_ids": [[10, 900, 901], [10, 20, 30, 40, 900, 901]],
        "completion_token_ids": [[20, 30, 777], [50]],
        "logprobs": [[-0.2, -0.3, -0.7], [-0.5]],
    },
    {
        "prompt_token_ids": [[10, 11, 900, 901], [70, 71, 900, 901]],
        "completion_token_ids": [[20], [80]],
        "logprobs": [[-0.2], [-0.8]],
    },
    {
        "prompt_token_ids": [[10, 11, 900, 901], [10, 11, 200, 40, 900, 901]],
        "completion_token_ids": [list(range(100, 140)), [50]],
        "logprobs": [[-i / 100 for i in range(100, 140)], [-0.5]],
        "extra": {
            "completion_text": ["same", "next"],
            "last_assistant_message_content": [None, "same"],
            "last_assistant_message_span": [None, [2, 3]],
        },
    },
]


def add_envelopes(detail):
    detail = copy.deepcopy(detail)
    envelopes = []
    for prompt, tokens, lps in zip(
        detail["prompt_token_ids"], detail["completion_token_ids"], detail["logprobs"], strict=True
    ):
        t = np.array(tokens, dtype="<i4")
        ids = np.stack([t, np.zeros_like(t)], -1)
        lp = np.stack([np.array(lps, dtype="<f4"), np.full(len(t), -10, dtype="<f4")], -1)
        envelopes.append(encode_turn_head(ids, lp, t, prompt))
    detail.setdefault("extra", {})["score_centering_head"] = envelopes
    return detail


@pytest.mark.parametrize("detail", FIXTURES)
def test_normalized_and_retokenized_true_identity(detail):
    kw = dict(reward=1.0, status=Sample.Status.COMPLETED, mode="trajectory_retokenized")
    off = samples_from_rollout_details([detail], Sample(index=1), **kw)
    on = samples_from_rollout_details([add_envelopes(detail)], Sample(index=1), score_centering_top_k=2, **kw)
    for baseline, sample in zip(off, on, strict=True):
        output = sample.to_dict()
        for key in HEAD_KEYS:
            output.pop(key)
        assert baseline.to_dict() == output
        assert np.array_equal(sample.sc_response_token_ids, sample.tokens[-sample.response_length :])
        recovered = Sample.from_dict(sample.to_dict())
        assert np.array_equal(sample.rollout_topk_token_ids, recovered.rollout_topk_token_ids)
        assert not any(hasattr(baseline, key) for key in HEAD_KEYS)


def test_response_cap_slices_every_track_and_retains_routing():
    detail = add_envelopes(FIXTURES[1])
    sample = samples_from_rollout_details(
        [detail],
        Sample(index=1),
        reward=1.0,
        status=Sample.Status.COMPLETED,
        mode="trajectory_retokenized",
        score_centering_top_k=2,
    )[0]
    sample.rollout_routed_experts = list(range(len(sample.tokens) - 1))
    args = SimpleNamespace(enable_score_centering=True, harbor_max_train_total_tokens=4)
    original = sample.sc_response_token_ids.copy()
    cap_total_tokens_allow_response_trim_in_samples(args, [[sample]])
    assert sample.response_length == 3 and np.array_equal(sample.sc_response_token_ids, original[-3:])
    assert sample.sc_response_token_ids.tolist() == sample.tokens[-3:]
    assert len(sample.rollout_routed_experts) == len(sample.tokens) - 1
    assert len(sample.loss_mask) == len(sample.rollout_topk_log_probs) == 3
    cap_total_tokens_allow_response_trim_in_rollout_data(args, 0, {"total_lengths": [4]})
    with pytest.raises(ValueError):
        cap_total_tokens_allow_response_trim_in_rollout_data(args, 0, {"total_lengths": [5]})


@pytest.mark.parametrize("corruption", ["digest", "tokens", "prompt", "missing"])
def test_misaligned_envelope_is_rejected(corruption):
    detail = add_envelopes(FIXTURES[0])
    envelope = detail["extra"]["score_centering_head"][0]
    if corruption == "digest":
        envelope["digest"] = "0" * 64
    elif corruption == "tokens":
        detail["completion_token_ids"][0][0] += 1
    elif corruption == "prompt":
        detail["prompt_token_ids"][0][0] += 1
    else:
        detail["extra"]["score_centering_head"].pop()
    with pytest.raises(ValueError):
        samples_from_rollout_details(
            [detail],
            Sample(),
            reward=1.0,
            status=Sample.Status.COMPLETED,
            mode="trajectory_retokenized",
            score_centering_top_k=2,
        )


def test_llm_wire_gate_and_trial_budget(monkeypatch):
    from slime_plugins.harbor import llm as module

    monkeypatch.setattr(module, "ensure_harbor_distribution_metadata", lambda: None)
    fake_base = types.ModuleType("harbor.llms.base")
    fake_base.ContextLengthExceededError = type("ContextLengthExceededError", (Exception,), {})
    fake_base.LLMResponse = lambda **kw: kw
    fake_metric = types.ModuleType("harbor.models.metric")
    fake_metric.UsageInfo = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "harbor.llms.base", fake_base)
    monkeypatch.setitem(sys.modules, "harbor.models.metric", fake_metric)
    calls = []

    async def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "text": "x",
                "meta_info": dict(
                    meta_fixture("json"), finish_reason={"type": "stop"}, output_token_logprobs=[[-0.5, 3], [-0.5, 4]]
                ),
            },
        )

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)
    )

    def make(enabled):
        obj = object.__new__(module.SlimeSGLangLLM)
        obj.__dict__.update(
            _temperature=1.0,
            _top_p=1.0,
            _top_k=-1,
            _min_p=0.0,
            _presence_penalty=0.0,
            _repetition_penalty=1.0,
            _max_new_tokens=2,
            _stop=None,
            _stop_token_ids=None,
            _max_context_tokens=None,
            _use_rollout_routing_replay=False,
            _record_rollout_text=False,
            _timeout_sec=2,
            _router_url="http://mock",
            _model_name="mock",
            _tokenizer=list(range(10)),
        )
        obj._encode_messages = lambda messages: [1, 2]
        if enabled:
            obj.__dict__.update(
                _score_centering_top_k=2,
                _score_centering_wire_format="json",
                _score_centering_max_response_bytes=1000,
                _score_centering_max_trial_bytes=40,
                _score_centering_reserved_bytes=0,
            )
        return obj

    async def check():
        off = await make(False).call("hello")
        assert "score_centering_head" not in off["extra"]
        enabled = make(True)
        on = await enabled.call("hello")
        decode_turn_head(on["extra"]["score_centering_head"], [1, 2], [3, 4], [-0.5, -0.5], k=2)
        with pytest.raises(RuntimeError, match="budget"):
            await enabled.call("hello")

    asyncio.run(check())
    assert "top_logprobs_num" not in calls[0] and calls[1]["top_logprobs_num"] == 2
    assert len(calls) == 2


def test_sc_remote_eval_overrides_never_mutate_worker_training_args(monkeypatch):
    from slime_plugins.harbor import rollout
    from slime_plugins.harbor.trial_worker import HarborTrialWorker

    worker = object.__new__(HarborTrialWorker)
    worker.args = SimpleNamespace(enable_score_centering=True, rollout_temperature=1.0)
    monkeypatch.setattr("ray.util.get_node_ip_address", lambda: "127.0.0.1")
    observed = []

    async def run(args, *unused):
        observed.append(args)
        return []

    monkeypatch.setattr(rollout, "_run_one_trial", run)
    overrides = dict(
        enable_score_centering=False,
        harbor_trial_name_template="slime-eval-r{rollout_id}",
        rollout_temperature=0.7,
        rollout_top_p=0.95,
        rollout_top_k=-1,
    )
    asyncio.run(worker.run_trial(0, 0, 0, Sample(), evaluation_overrides=overrides))
    assert observed[0].enable_score_centering is False and observed[0].rollout_temperature == 0.7
    assert worker.args.enable_score_centering is True and worker.args.rollout_temperature == 1.0
