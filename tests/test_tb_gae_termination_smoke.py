"""Real Terminus2.run -> Chat/parser -> metadata -> adapter -> TB-GAE on CPU.

Only model replies and terminal I/O are replaced. No hand-written termination
metadata, remote models, Ray cluster, Docker, or production run directories.
Run with the Harbor environment and both source roots on PYTHONPATH.
"""

import ast
import asyncio
import copy
import hashlib
import inspect
import json
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from harbor.agents.terminus_2 import terminus_2 as harbor_module
from harbor.llms.base import BaseLLM, ContextLengthExceededError, LLMResponse
from harbor.models.agent.context import AgentContext
from harbor.models.metric import UsageInfo

from slime.utils.types import Sample
from slime_plugins.harbor.agent import SlimeTerminus2
from slime_plugins.harbor.atif_adapter import samples_from_trial_result
from slime_plugins.harbor.sample_postprocess import cap_total_tokens_allow_response_trim_in_samples
from slime_plugins.harbor.tb_gae import compute_tb_gae
from slime_plugins.harbor.tb_gae_termination import AGENT_PATH, configure_termination_producer

NUM_GPUS = 0
pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


class ScriptedLLM(BaseLLM):
    def __init__(self, replies):
        self.replies = iter(replies)
        self.prefix = [10, 11]
        self.calls = 0

    async def call(self, prompt, **kwargs):
        reply = next(self.replies)
        if isinstance(reply, BaseException):
            raise reply
        if reply == "wait":
            await asyncio.Event().wait()
        self.calls += 1
        content = json.dumps({"analysis": "inspect", "plan": "finish", "commands": [], "task_complete": reply})
        output = [20 + self.calls, 30 + self.calls]
        response = LLMResponse(
            content=content,
            model_name="offline/model",
            prompt_token_ids=list(self.prefix),
            completion_token_ids=output,
            logprobs=[-0.4, -0.6],
            usage=UsageInfo(prompt_tokens=len(self.prefix), completion_tokens=2, cache_tokens=0, cost_usd=0),
            extra={"finish_reason": {"type": "stop"}},
        )
        self.prefix += output + [90, 91]
        return response

    def get_model_context_limit(self):
        return 262144

    def get_model_output_limit(self):
        return 1024

    def encode_bootstrap_state(self, prompt, *, message_history):
        self.bootstrap_messages = [*message_history, {"role": "user", "content": prompt}]
        return list(self.prefix)


def make_agent(monkeypatch, tmp_path, replies, *, max_turns=4, track=True, cls=SlimeTerminus2):
    llm = ScriptedLLM(replies)
    monkeypatch.setattr(cls, "_init_llm", lambda self, **kwargs: llm)
    kwargs = {} if track is None else {"record_termination_metadata": track}
    agent = cls(
        logs_dir=tmp_path,
        model_name="offline/model",
        max_turns=max_turns,
        session_id="offline-termination",
        enable_summarize=False,
        record_terminal_session=False,
        collect_rollout_details=True,
        suppress_max_turns_warning=True,
        **kwargs,
    )
    agent._session = SimpleNamespace(
        is_session_alive=AsyncMock(return_value=True),
        get_incremental_output=AsyncMock(return_value="terminal ready"),
        send_keys=AsyncMock(),
    )
    agent._build_skills_section = AsyncMock(return_value="")
    # Avoid even temp trajectory/episode files; exercise real run, parser,
    # Chat accumulation, command execution and completion confirmation.
    agent._setup_episode_logging = lambda *args: (None, None, None)
    agent._dump_trajectory = lambda: None
    agent._record_asciinema_marker = lambda *args: None
    return agent, llm


def execute(agent):
    context = AgentContext()
    asyncio.run(agent.run("Finish the task", SimpleNamespace(), context))
    # Real JSON serialization must not drop the producer fields.
    return AgentContext.model_validate_json(context.model_dump_json())


def convert(context, reward=0.7, enabled=True):
    return samples_from_trial_result(
        {"agent_result": context, "verifier_result": {"rewards": {"reward": reward}}},
        Sample(index=3),
        mode="trajectory_retokenized",
        enable_tb_gae=enabled,
    )


@pytest.mark.parametrize("gamma", [1.0, 0.999])
@pytest.mark.parametrize("reward", [0.0, 0.7])
def test_actual_agent_completion_through_adapter_and_gae(monkeypatch, tmp_path, gamma, reward):
    agent, llm = make_agent(monkeypatch, tmp_path, [True, True], max_turns=2)
    context = execute(agent)
    assert llm.calls == 2  # At the exact limit, actual confirmation wins over truncation.
    assert context.metadata["termination_kind"] == "terminated"
    assert context.metadata["termination_reason"] == "task_complete_confirmed"
    samples = convert(context, reward=reward)
    sample = samples[0]
    assert sample.metadata["turn_end_mask"].sum() == 2
    assert sample.metadata["is_real_done"].sum() == 1
    assert sample.metadata["is_real_done"][-1]
    mask = torch.tensor(sample.loss_mask, dtype=torch.bool)[None]
    values = torch.linspace(-0.1, 0.3, sample.response_length)[None]
    rewards = torch.zeros_like(values)
    rewards[0, -1] = reward
    te = torch.tensor(sample.metadata["turn_end_mask"])[None]
    done = torch.tensor(sample.metadata["is_real_done"])[None]
    advantage, returns = compute_tb_gae(values, rewards, mask, te, done, gamma, gamma)
    # Independent sequential action reference with real producer-derived masks.
    idx = torch.nonzero(mask[0], as_tuple=True)[0]
    last = 0.0
    expected = torch.zeros_like(values)
    for j in reversed(range(len(idx))):
        t = idx[j]
        g = gamma if te[0, t] and j < len(idx) - 1 else 1.0
        nv = values[0, idx[j + 1]] if j + 1 < len(idx) else 0.0
        last = rewards[0, t] + g * nv - values[0, t] + g * g * last
        expected[0, t] = last
    torch.testing.assert_close(advantage, expected, atol=1e-6, rtol=1e-6)
    if gamma == 1:
        torch.testing.assert_close(returns[mask], torch.full_like(returns[mask], reward))
    assert advantage.grad_fn is None and returns.grad_fn is None
    cap_total_tokens_allow_response_trim_in_samples(
        SimpleNamespace(enable_tb_gae=True, harbor_max_train_total_tokens=4), samples
    )
    assert samples[-1].metadata["is_real_done"][-1]
    assert all(len(s.tokens) <= 4 for s in samples)
    assert sum(sum(s.loss_mask) for s in samples) == int(mask.sum())


@pytest.mark.parametrize("replies", [[False, False], [False, True], [True, False]])
def test_max_turns_and_unconfirmed_completion_are_not_done(monkeypatch, tmp_path, replies):
    agent, _ = make_agent(monkeypatch, tmp_path, replies, max_turns=2)
    context = execute(agent)
    assert context.metadata["termination_kind"] == "truncated"
    assert context.metadata["termination_reason"] == "max_turns"
    samples = convert(context)
    assert not samples[-1].metadata["is_real_done"].any()
    assert samples[-1].metadata["tb_boundary"]["bootstrap_token_ids"]


def test_reused_agent_does_not_leak_previous_termination(monkeypatch, tmp_path):
    agent, _ = make_agent(monkeypatch, tmp_path, [True, True], max_turns=2)
    assert execute(agent).metadata["termination_kind"] == "terminated"
    agent._llm = ScriptedLLM([False, False])
    assert execute(agent).metadata["termination_kind"] == "truncated"


def test_lost_session_is_truncated(monkeypatch, tmp_path):
    agent, _ = make_agent(monkeypatch, tmp_path, [False])
    agent._session.is_session_alive.side_effect = [True, False]
    context = execute(agent)
    assert context.metadata["termination_reason"] == "session_ended"
    assert not convert(context)[0].metadata["is_real_done"].any()


@pytest.mark.parametrize(
    "error,kind,reason",
    [
        (TimeoutError("clock"), "truncated", "timeout"),
        (ContextLengthExceededError("context"), "truncated", "context_length"),
        (RuntimeError("broken"), "error", "agent_exception"),
        (asyncio.CancelledError(), "truncated", "cancelled"),
    ],
)
@pytest.mark.parametrize("phase", ["initial_io", "loop"])
def test_error_paths_publish_metadata_and_reraise(monkeypatch, tmp_path, error, kind, reason, phase):
    agent, _ = make_agent(monkeypatch, tmp_path, [False])
    if phase == "initial_io":
        agent._session.get_incremental_output.side_effect = error
    else:
        agent._session.is_session_alive.side_effect = error
    context = AgentContext()
    with pytest.raises(type(error)):
        asyncio.run(agent.run("task", SimpleNamespace(), context))
    assert context.metadata["termination_kind"] == kind
    assert context.metadata["termination_reason"] == reason
    with pytest.raises(ValueError, match="bootstrap state|" + reason):
        convert(context)


def test_real_wait_for_timeout_is_never_terminated(monkeypatch, tmp_path):
    agent, _ = make_agent(monkeypatch, tmp_path, [False, "wait"])
    context = AgentContext()

    async def run():
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(agent.run("task", SimpleNamespace(), context), timeout=0.03)

    asyncio.run(run())
    # wait_for cancels the coroutine. Trial adds AgentTimeoutError later.
    assert context.metadata["termination_kind"] == "truncated"
    assert context.metadata["termination_reason"] == "cancelled"
    assert len(context.rollout_details[0]["completion_token_ids"]) == 1


@pytest.mark.parametrize("track", [None, False])
@pytest.mark.parametrize("replies", [[True, True], [False, False]])
def test_disabled_agent_matches_pre_fix_context_and_samples(monkeypatch, tmp_path, track, replies):
    snapshot = ROOT / "tests/fixtures/tb_gae_termination_pre_patch"
    manifest = json.loads((snapshot / "manifest.json").read_text())
    entry = next(v for k, v in manifest.items() if k.endswith("terminus_2.py"))
    source = (snapshot / entry["snapshot"]).read_bytes()
    assert hashlib.sha256(source).hexdigest() == entry["sha256"]
    module = types.ModuleType("harbor.agents.terminus_2.pre_termination")
    module.__file__ = harbor_module.__file__
    exec(compile(source, "pre_termination.py", "exec"), module.__dict__)
    monkeypatch.setattr(harbor_module.time, "time", lambda: 100.0)
    old, _ = make_agent(monkeypatch, tmp_path, replies, max_turns=2, track=None, cls=module.Terminus2)
    new, _ = make_agent(monkeypatch, tmp_path, replies, max_turns=2, track=track)
    a, b = execute(old), execute(new)
    assert a.model_dump() == b.model_dump()
    with pytest.raises(ValueError, match="kind=None"):
        convert(a)  # Reproduce the original blocker using actual old-agent output.
    assert not any("termination" in key for key in b.metadata)
    assert not any(key.startswith("_termination") for key in vars(new))
    assert convert(a, enabled=False)[0].to_dict() == convert(b, enabled=False)[0].to_dict()


def trial_agent_config(enabled, *, evaluation=False):
    """Execute the actual config function without importing Ray (not installed here)."""
    source = (ROOT / "slime_plugins/harbor/rollout.py").read_text()
    function = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_set_default_agent_config"
    )
    namespace = {"Any": object, "_router_url": lambda args: "http://offline", "_is_eval_args": lambda args: evaluation}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "rollout-config", "exec"), namespace)
    config = {}
    args = SimpleNamespace(enable_tb_gae=enabled, hf_checkpoint="offline", harbor_sample_mode="trajectory_retokenized")
    namespace["_set_default_agent_config"](args, config)
    return config["agent"]


def test_real_config_enables_real_producer(monkeypatch, tmp_path):
    config = trial_agent_config(True)
    assert config["kwargs"]["record_termination_metadata"] is True
    agent, _ = make_agent(monkeypatch, tmp_path, [True, True], track=config["kwargs"]["record_termination_metadata"])
    assert execute(agent).metadata["termination_kind"] == "terminated"
    assert str(Path(harbor_module.__file__).resolve()).startswith(str(ROOT.parent / "harbor/src"))


@pytest.mark.parametrize("enabled,evaluation", [(False, False), (True, True)])
def test_off_and_eval_config_unchanged(enabled, evaluation):
    config = trial_agent_config(enabled, evaluation=evaluation)
    assert "record_termination_metadata" not in config["kwargs"]


def test_old_harbor_fails_before_creating_trial(monkeypatch):
    monkeypatch.delattr(harbor_module.Terminus2, "TERMINATION_METADATA_VERSION")
    with pytest.raises(ValueError, match="loaded Harbor.*terminus_2.py"):
        trial_agent_config(True)


@pytest.mark.parametrize(
    "updates",
    [
        {"import_path": "another.agent:Agent"},
        {"name": "terminus-2"},
        {"kwargs": {"record_termination_metadata": False, "collect_rollout_details": True}},
        {"kwargs": {"collect_rollout_details": False}},
    ],
)
def test_incompatible_agent_config_rejected(updates):
    config = {"import_path": AGENT_PATH, "kwargs": {"collect_rollout_details": True}}
    config.update(copy.deepcopy(updates))
    with pytest.raises(ValueError):
        configure_termination_producer(config)


def test_producer_is_real_loop_not_test_metadata():
    assert SlimeTerminus2._run_agent_loop is harbor_module.Terminus2._run_agent_loop
    assert "task_complete_confirmed" in inspect.getsource(SlimeTerminus2._run_agent_loop)


@pytest.mark.parametrize("timeout", [False, True])
def test_real_factory_and_trial_agent_execution(monkeypatch, tmp_path, timeout):
    from harbor.agents.factory import AgentFactory
    from harbor.models.trial.config import AgentConfig
    from harbor.trial.trial import AgentTimeoutError, Trial

    config = trial_agent_config(True)
    config["kwargs"].update(max_turns=2, suppress_max_turns_warning=True, enable_summarize=False)
    scripted = ScriptedLLM([False, "wait"] if timeout else [True, True])
    monkeypatch.setattr(SlimeTerminus2, "_init_llm", lambda self, **kwargs: scripted)
    agent = AgentFactory.create_agent_from_config(AgentConfig.model_validate(config), logs_dir=tmp_path)
    assert type(agent) is SlimeTerminus2
    agent._session = SimpleNamespace(
        is_session_alive=AsyncMock(return_value=True),
        get_incremental_output=AsyncMock(return_value="ready"),
        send_keys=AsyncMock(),
    )
    agent._build_skills_section = AsyncMock(return_value="")
    agent._setup_episode_logging = lambda *args: (None, None, None)
    agent._dump_trajectory = lambda: None
    agent._record_asciinema_marker = lambda *args: None
    trial = object.__new__(Trial)
    trial._agent = agent
    trial._task = SimpleNamespace(instruction="task")
    trial._environment = SimpleNamespace()
    trial._result = SimpleNamespace(agent_result=None, agent_execution=None)
    trial._invoke_hooks = AsyncMock()
    trial._agent_timeout_sec = 0.03 if timeout else 5
    if timeout:
        with pytest.raises(AgentTimeoutError):
            asyncio.run(trial._execute_agent())
        assert trial.result.agent_result.metadata["termination_kind"] == "truncated"
    else:
        asyncio.run(trial._execute_agent())
        assert convert(trial.result.agent_result)[0].metadata["is_real_done"][-1]
    assert trial.result.agent_execution.finished_at is not None


def test_remote_worker_evaluation_disables_tb_without_mutating_training_args(monkeypatch):
    import sys

    captured = []

    async def run_one(args, *unused):
        captured.append(args)
        return []

    # Run the actual worker method; transport/logging are irrelevant offline.
    fake_rollout = types.ModuleType("slime_plugins.harbor.rollout")
    fake_rollout._run_one_trial = run_one
    monkeypatch.setitem(sys.modules, "slime_plugins.harbor.rollout", fake_rollout)
    source = (ROOT / "slime_plugins/harbor/trial_worker.py").read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "HarborTrialWorker")
    node = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_trial")
    ns = {
        "ray": SimpleNamespace(util=SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1")),
        "logger": SimpleNamespace(info=lambda *args: None),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "trial-worker", "exec"), ns)
    original = SimpleNamespace(enable_tb_gae=True, enable_score_centering=False)
    worker = SimpleNamespace(args=original)
    overrides = dict(
        enable_score_centering=False,
        enable_tb_gae=False,
        harbor_trial_name_template="slime-eval-{rollout_id}",
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
    )
    asyncio.run(ns["run_trial"](worker, 0, 0, 0, Sample(), evaluation_overrides=overrides))
    assert captured[0].enable_tb_gae is False and original.enable_tb_gae is True
    overrides["enable_tb_gae"] = True
    with pytest.raises(ValueError, match="must disable"):
        asyncio.run(ns["run_trial"](worker, 0, 0, 0, Sample(), evaluation_overrides=overrides))


def test_real_eval_coordinator_copies_and_marks_tb_args():
    source = (ROOT / "slime_plugins/harbor/rollout.py").read_text()
    node = next(
        n for n in ast.parse(source).body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_generate_eval_async"
    )
    captured = []

    async def run_group(args, *unused, **kwargs):
        captured.append(args)
        return [Sample(reward=0.0)], {"converted": 1}

    ns = {
        "asyncio": asyncio,
        "copy": copy,
        "Any": object,
        "Sample": Sample,
        "_run_one_group": run_group,
        "_float_arg": lambda *args: 0.0,
        "_deduplicate_eval_trajectory_outcomes": lambda groups: ([0.0], [False]),
        "_group_eval_trajectory_rewards_by_task": lambda groups, **kw: [[0.0]],
        "_compute_eval_count_metrics": lambda **kw: {},
        "_write_eval_metrics_artifact": lambda *args: "offline-no-write",
        "logger": SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "eval-coordinator", "exec"), ns)
    args = SimpleNamespace(enable_tb_gae=True, enable_score_centering=False)
    data_source = SimpleNamespace(get_samples=lambda *args, **kw: [[Sample()]])
    asyncio.run(ns["_generate_eval_async"](args, 0, data_source))
    assert args.enable_tb_gae is True
    assert captured[0].enable_tb_gae is False and captured[0]._tb_gae_eval_dispatch is True
    assert "slime-eval-" in captured[0].harbor_trial_name_template


def test_real_remote_dispatch_sends_tb_eval_override(monkeypatch):
    import concurrent.futures
    import itertools
    import sys

    source = (ROOT / "slime_plugins/harbor/rollout.py").read_text()
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_run_one_with_semaphore"
    )
    captured = []

    def remote(*args, **kwargs):
        captured.append(kwargs)
        done = concurrent.futures.Future()
        done.set_result([])
        return SimpleNamespace(future=lambda: done)

    fake_ray = types.ModuleType("ray")
    exceptions = types.ModuleType("ray.exceptions")
    exceptions.RayActorError = type("RayActorError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.exceptions", exceptions)
    ns = {
        "asyncio": asyncio,
        "itertools": itertools,
        "Sample": Sample,
        "_worker_cycle": None,
        "_worker_cycle_len": 0,
        "_worker_cycle_fingerprint": None,
        "_get_create_rate_limiter": lambda args: None,
        "_float_arg": lambda *args: 0.0,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "eval-dispatch", "exec"), ns)
    args = SimpleNamespace(
        _harbor_trial_workers=[SimpleNamespace(run_trial=SimpleNamespace(remote=remote))],
        _tb_gae_eval_dispatch=True,
        enable_tb_gae=False,
        harbor_trial_name_template="slime-eval-{rollout_id}",
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
    )
    asyncio.run(ns["_run_one_with_semaphore"](args, 0, 0, 0, Sample(), asyncio.Semaphore(1)))
    assert captured[0]["evaluation_overrides"]["enable_tb_gae"] is False
