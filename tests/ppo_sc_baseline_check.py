"""Compare SC-off behavior against a frozen pre-implementation working tree.

The exact old source can be supplied via PPO_SC_BASELINE_DIR; absence fails this
release check instead of silently substituting an args-only comparison.
"""

import copy
import os
from pathlib import Path
import types

import pytest
import torch

from test_ppo_sc_harbor_integration import (
    CORE_LOSS,
    HARBOR_LOSS,
    baseline_args,
    load_loss,
    make_batch,
    megatron_single_rank,
)

NUM_GPUS = 0
pytestmark = pytest.mark.unit


def run_loss(loss_fn, args, batch, logits):
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean

    reducer = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        batch["group_mask_sums"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens"),
    )
    loss, metrics = loss_fn(args, batch, logits, reducer)
    # This regression uses one backward per graph, matching production and the
    # compiled KL operator's donated-buffer contract.
    grad = torch.autograd.grad(loss, logits)[0]
    return loss, metrics, grad


def before_module(relative, package):
    directory = os.environ.get("PPO_SC_BASELINE_DIR")
    if not directory:
        pytest.fail("Set PPO_SC_BASELINE_DIR to the captured pre-implementation source tree")
    path = Path(directory) / relative
    module = types.ModuleType(package + ".pre_sc_baseline")
    module.__package__ = package
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("spec", [HARBOR_LOSS, CORE_LOSS], ids=["harbor", "core"])
@pytest.mark.parametrize("entropy", [0.0, 0.01])
@pytest.mark.parametrize("kl", [False, True])
def test_off_loss_all_gradients_optimizer_step_and_metrics_match_old_source(megatron_single_rank, spec, entropy, kl):
    current = load_loss(spec)
    relative = spec[0].replace(".", "/") + ".py"
    prior = getattr(before_module(relative, spec[0].rpartition(".")[0]), spec[1])
    args = baseline_args(enable_score_centering=False, entropy_coef=entropy, use_kl_loss=kl, kl_loss_coef=0.03)
    first, x = make_batch(seed=42, with_heads=True)
    second, y = make_batch(seed=42, with_heads=True)
    if kl:
        refs = [torch.full((r,), -4.0) for r in first["response_lengths"]]
        first["ref_log_probs"] = refs
        second["ref_log_probs"] = [t.clone() for t in refs]
    # Use real frozen old logprobs, rather than the loss's on-policy fallback.
    first["log_probs"] = [t.clone() + 0.2 for t in first["rollout_log_probs"]]
    second["log_probs"] = [t.clone() + 0.2 for t in second["rollout_log_probs"]]
    loss_new, metrics_new, grad_new = run_loss(current, args, first, x)
    loss_old, metrics_old, grad_old = run_loss(prior, args, second, y)
    assert torch.equal(loss_new, loss_old) and torch.equal(grad_new, grad_old)
    assert metrics_new.keys() == metrics_old.keys()
    for key in metrics_new:
        assert torch.equal(metrics_new[key], metrics_old[key]), key
    pn = torch.nn.Parameter(x.detach().clone())
    po = torch.nn.Parameter(y.detach().clone())
    on = torch.optim.AdamW([pn], lr=5e-7)
    oo = torch.optim.AdamW([po], lr=5e-7)
    pn.grad = grad_new
    po.grad = grad_old
    on.step()
    oo.step()
    assert torch.equal(pn, po)
    for key in on.state[pn]:
        assert torch.equal(on.state[pn][key], oo.state[po][key])


def test_original_warmcritic_role_values_match_previous_parser():
    from slime.utils.arguments import parse_megatron_role_args
    from test_ppo_sc_args import make_args, ROLES

    prior = before_module("slime/utils/arguments.py", "slime.utils")
    for role in ("actor", "critic"):
        old = prior.parse_megatron_role_args(make_args(), str(ROLES), role)
        new = parse_megatron_role_args(make_args(), str(ROLES), role)
        assert vars(old) == vars(new)


def test_off_adapter_and_hard_cap_equal_previous_source():
    from slime_plugins.harbor.atif_adapter import samples_from_rollout_details
    from slime_plugins.harbor.sample_postprocess import cap_total_tokens_allow_response_trim_in_samples
    from slime.utils.types import Sample
    from test_ppo_sc_transport import FIXTURES

    prior = before_module("slime_plugins/harbor/atif_adapter.py", "slime_plugins.harbor")
    prior_cap = before_module("slime_plugins/harbor/sample_postprocess.py", "slime_plugins.harbor")
    args = types.SimpleNamespace(enable_score_centering=False, harbor_max_train_total_tokens=4)
    for detail in FIXTURES:
        kwargs = dict(reward=1.0, status=Sample.Status.COMPLETED, mode="trajectory_retokenized")
        old = prior.samples_from_rollout_details([copy.deepcopy(detail)], Sample(index=1), **kwargs)
        new = samples_from_rollout_details([copy.deepcopy(detail)], Sample(index=1), **kwargs)
        prior_cap.cap_total_tokens_allow_response_trim_in_samples(args, old)
        cap_total_tokens_allow_response_trim_in_samples(args, new)
        assert [s.to_dict() for s in old] == [s.to_dict() for s in new]


def test_non_sc_parser_defaults_equal_frozen_source():
    from argparse import ArgumentParser
    from slime.utils.arguments import get_slime_extra_args_provider
    from slime.utils.score_centering_config import SC_DEFAULTS

    prior = before_module("slime/utils/arguments.py", "slime.utils")
    old = prior.get_slime_extra_args_provider()(ArgumentParser()).parse_args(["--rollout-batch-size", "8"])
    new = get_slime_extra_args_provider()(ArgumentParser()).parse_args(["--rollout-batch-size", "8"])

    def core(args):
        return {key: value for key, value in vars(args).items() if key not in SC_DEFAULTS}

    assert core(old) == core(new)


def test_off_http_payload_bytes_and_response_equal_frozen_client(monkeypatch):
    import asyncio
    import httpx
    import sys
    from slime_plugins.harbor import llm as current

    prior = before_module("slime_plugins/harbor/llm.py", "slime_plugins.harbor")
    for module in (prior, current):
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
        calls.append(request.content)
        return httpx.Response(
            200,
            json={
                "text": "x",
                "meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.5, 3], [-0.5, 4]]},
            },
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(handler), **kw))

    def make(module):
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
        return obj

    async def check():
        a = await make(prior).call("hello")
        b = await make(current).call("hello")
        assert a == b

    asyncio.run(check())
    assert calls[0] == calls[1]
