from __future__ import annotations

from typing import Any

from slime_plugins.harbor.import_utils import ensure_harbor_distribution_metadata


class SlimeTerminus2Mixin:
    """Mixin that lets Terminus2 use slime's active SGLang rollout server."""

    def _init_llm(
        self,
        llm_backend,
        model_name: str,
        temperature: float | None,
        collect_rollout_details: bool,
        llm_kwargs: dict | None,
        api_base: str | None,
        session_id: str | None,
        max_thinking_tokens: int | None,
        reasoning_effort,
        model_info: dict | None,
        use_responses_api: bool,
    ):
        backend_value = getattr(llm_backend, "value", llm_backend)
        if backend_value == "slime_sglang":
            ensure_harbor_distribution_metadata()
            from harbor.llms.base import BaseLLM

            from slime_plugins.harbor.llm import SlimeSGLangLLM

            class _SlimeSGLangHarborLLM(SlimeSGLangLLM, BaseLLM):
                pass

            kwargs: dict[str, Any] = dict(llm_kwargs or {})
            kwargs.setdefault("model_name", model_name)
            if temperature is not None:
                kwargs.setdefault("temperature", temperature)
            return _SlimeSGLangHarborLLM(**kwargs)

        return super()._init_llm(
            llm_backend=llm_backend,
            model_name=model_name,
            temperature=temperature,
            collect_rollout_details=collect_rollout_details,
            llm_kwargs=llm_kwargs,
            api_base=api_base,
            session_id=session_id,
            max_thinking_tokens=max_thinking_tokens,
            reasoning_effort=reasoning_effort,
            model_info=model_info,
            use_responses_api=use_responses_api,
        )


def _load_terminus_base():
    ensure_harbor_distribution_metadata()
    from harbor.agents.terminus_2 import Terminus2

    return Terminus2


class SlimeTerminus2(SlimeTerminus2Mixin, _load_terminus_base()):
    """Terminus2 subclass that supports ``llm_backend='slime_sglang'``."""
