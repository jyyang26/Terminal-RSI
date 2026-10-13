"""Actual agent truncation events produce successor states without generation."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_tb_gae_termination_smoke import convert, execute, make_agent

from harbor.models.agent.context import AgentContext
from slime_plugins.harbor.llm import SlimeSGLangLLM
from slime_plugins.harbor.tb_gae_data import validate_and_collect_tb_masks

NUM_GPUS = 0
pytestmark = pytest.mark.unit


def test_encoder_uses_exact_rollout_chat_template_without_network():
    llm = object.__new__(SlimeSGLangLLM)
    seen = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen.append((messages, kwargs))
            return [11, 22, 33]

    llm._tokenizer = Tokenizer()
    llm._add_generation_prompt = True
    llm._apply_chat_template_kwargs = {"enable_thinking": False}
    history = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "action"}]
    actual = llm.encode_bootstrap_state("post-action observation", message_history=history)
    assert actual == llm._encode_messages([*history, {"role": "user", "content": "post-action observation"}])
    assert seen[0] == seen[1] and seen[0][1]["add_generation_prompt"] is True


@pytest.mark.parametrize("replies", [[False, False], [True, False], [False, True]])
def test_max_turns_keeps_trajectory_with_nonterminal_masks(monkeypatch, tmp_path, replies):
    agent, llm = make_agent(monkeypatch, tmp_path, replies, max_turns=2)
    context = execute(agent)
    assert llm.calls == 2
    assert context.metadata["bootstrap_action_count"] == 4
    assert llm.bootstrap_messages[-1]["role"] == "user"
    assert "terminal ready" in llm.bootstrap_messages[-1]["content"]
    samples = convert(context)
    data = validate_and_collect_tb_masks(samples, [s.loss_mask for s in samples])
    assert sum(sum(s.loss_mask) for s in samples) == 4
    assert not data["is_real_done"][0].any()
    assert data["tb_boundary"][0]["bootstrap_token_ids"] == context.metadata["bootstrap_token_ids"]


def test_timeout_waiting_for_next_model_response_preserves_last_observation(monkeypatch, tmp_path):
    agent, llm = make_agent(monkeypatch, tmp_path, [False, "wait"])
    context = AgentContext()

    async def run():
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(agent.run("task", SimpleNamespace(), context), timeout=0.03)

    asyncio.run(run())
    assert context.metadata["bootstrap_action_count"] == 2
    assert llm.calls == 1
    assert llm.bootstrap_messages[-1]["content"] == "terminal ready"
    assert len(convert(context)) == 1


def test_timeout_during_tool_collects_actual_terminal_observation(monkeypatch, tmp_path):
    agent, llm = make_agent(monkeypatch, tmp_path, [False])
    original = llm.call

    async def with_command(*args, **kw):
        response = await original(*args, **kw)
        response.content = json.dumps(
            {
                "analysis": "run",
                "plan": "wait",
                "commands": [{"keystrokes": "long-running", "duration": 1}],
                "task_complete": False,
            }
        )
        return response

    llm.call = with_command

    async def slow_tool(*args, **kw):
        await asyncio.Event().wait()

    agent._session.send_keys = slow_tool
    agent._session.get_incremental_output = AsyncMock(side_effect=["before", "partial tool result at timeout"])
    context = AgentContext()

    async def run():
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(agent.run("task", SimpleNamespace(), context), timeout=0.03)

    asyncio.run(run())
    assert context.metadata["termination_kind"] == "truncated"
    assert llm.bootstrap_messages[-1]["content"] == "partial tool result at timeout"
    assert context.metadata["bootstrap_action_count"] == 2
    assert not convert(context)[0].metadata["is_real_done"].any()


def test_real_trial_timeout_exception_is_accepted_with_successor(monkeypatch, tmp_path):
    from harbor.models.trial.result import ExceptionInfo
    from harbor.trial.trial import AgentTimeoutError, Trial
    from slime.utils.types import Sample
    from slime_plugins.harbor.atif_adapter import samples_from_trial_result

    agent, _ = make_agent(monkeypatch, tmp_path, [False, "wait"])
    trial = object.__new__(Trial)
    trial._agent, trial._environment = agent, SimpleNamespace()
    trial._task = SimpleNamespace(instruction="task")
    trial._result = SimpleNamespace(agent_result=None, agent_execution=None)
    trial._agent_timeout_sec = 0.03
    trial._invoke_hooks = AsyncMock()
    try:
        asyncio.run(trial._execute_agent())
    except AgentTimeoutError as exc:
        exception = ExceptionInfo.from_exception(exc)
    result = {
        "agent_result": trial.result.agent_result,
        "exception_info": exception,
        "verifier_result": {"rewards": {"reward": 0.3}},
    }
    samples = samples_from_trial_result(result, Sample(index=9), mode="trajectory_retokenized", enable_tb_gae=True)
    assert samples[0].status == Sample.Status.TRUNCATED
    assert samples[0].metadata["tb_boundary"]["bootstrap_token_ids"]


def test_mixed_rollout_keeps_natural_maxturns_and_timeout_without_backfill(monkeypatch, tmp_path):
    from slime.utils.types import Sample
    from slime_plugins.harbor import rollout

    completed, _ = make_agent(monkeypatch, tmp_path, [True, True], max_turns=2)
    capped, _ = make_agent(monkeypatch, tmp_path, [False, False], max_turns=2)
    waiting, _ = make_agent(monkeypatch, tmp_path, [False, "wait"])
    contexts = [execute(completed), execute(capped)]
    timeout_context = AgentContext()

    async def wait():
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(waiting.run("task", SimpleNamespace(), timeout_context), timeout=0.03)

    asyncio.run(wait())
    contexts.append(timeout_context)
    calls = []

    async def trial(args, rid, gid, rank, base, semaphore):
        calls.append(gid)
        samples = convert(contexts[gid])
        for s in samples:
            s.group_id = gid
            s.metadata["tb_boundary"]["episode"] = str(gid)
        return samples

    class Source:
        def __init__(self):
            self.requests = []

        def get_samples(self, n):
            self.requests.append(n)
            return [[Sample(index=i)] for i in range(n)]

    monkeypatch.setattr(rollout, "_run_one_with_semaphore", trial)
    args = SimpleNamespace(
        enable_tb_gae=True,
        enable_score_centering=False,
        rollout_batch_size=3,
        over_sampling_batch_size=3,
        n_samples_per_prompt=1,
        harbor_num_concurrent_trials=3,
        harbor_retry_missing_samples=0,
        harbor_fail_on_trial_error=False,
        harbor_require_full_rollout_groups=True,
        harbor_sample_mode="trajectory_retokenized",
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
    )
    source = Source()
    output = asyncio.run(rollout._generate_rollout_async(args, 0, source))
    samples = [s for group in output["samples"] for s in group]
    assert source.requests == [3] and sorted(calls) == [0, 1, 2]
    assert len(samples) == 3
    assert sum(s.metadata["tb_boundary"]["terminal"] for s in samples) == 1
    assert output["metrics"]["harbor/num_failed_trials"] == 0
    validate_and_collect_tb_masks(samples, [s.loss_mask for s in samples])
