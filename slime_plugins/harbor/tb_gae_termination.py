"""Opt-in wiring for the real Terminus-2 termination event producer."""

import inspect

AGENT_PATH = "slime_plugins.harbor.agent:SlimeTerminus2"


def configure_termination_producer(agent: dict) -> None:
    """Called only for TB training, before Trial.create allocates an environment."""
    if agent.get("import_path") != AGENT_PATH or agent.get("name") is not None:
        raise ValueError(f"TB-GAE termination tracking requires import_path={AGENT_PATH!r} without agent.name")
    kwargs = agent.setdefault("kwargs", {})
    if "record_termination_metadata" in kwargs and kwargs["record_termination_metadata"] is not True:
        raise ValueError("TB-GAE requires record_termination_metadata=True")
    if kwargs.get("collect_rollout_details") is not True:
        raise ValueError("TB-GAE requires collect_rollout_details=True")

    from slime_plugins.harbor.agent import SlimeTerminus2

    source = inspect.getfile(SlimeTerminus2._run_agent_loop)
    parameters = inspect.signature(SlimeTerminus2.__init__).parameters
    if (
        getattr(SlimeTerminus2, "TERMINATION_METADATA_VERSION", None) != 2
        or "record_termination_metadata" not in parameters
    ):
        raise ValueError(
            f"TB-GAE bootstrap producer unavailable in loaded Harbor: {source}. Deploy the Terminus-2 termination metadata v2 patch to HARBOR_SRC and restart workers."
        )
    kwargs["record_termination_metadata"] = True
