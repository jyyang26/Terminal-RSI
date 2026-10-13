"""Resolve SC once for rollout and actor; keep role-specific overrides isolated.

This module has no torch/Megatron/Ray imports. Precedence follows the existing
custom-config convention: parser/CLI < common role-file block < custom YAML.
An explicit CLI enable conflicts with a YAML disable rather than disappearing.
"""

from collections.abc import Mapping

SC_DEFAULTS = {
    "enable_score_centering": False,
    "sc_top_k": 32,
    "sc_tail_eps": 1e-8,
    "sc_formula": "score",
    "sc_weight_denominator": "old_actor",
    "sc_correction_scale": 1.0,
    "sc_assert_alignment": "full",
    "sc_chunk_size": 256,
    "sc_wire_format": "binary",
    "sc_max_response_bytes": 268435456,
    "sc_max_trial_bytes": 536870912,
    "sc_max_batch_bytes": 8589934592,
}
SC_ALIASES = {"use_score_centering": "enable_score_centering", "score_centering_top_k": "sc_top_k"}


def canonical_sc_values(values, *, strict=False):
    if not isinstance(values, Mapping):
        raise ValueError("score_centering must be a YAML mapping")
    resolved = {}
    for key, value in values.items():
        canonical = SC_ALIASES.get(key, key)
        if canonical not in SC_DEFAULTS:
            if strict or str(key).startswith("sc_"):
                raise ValueError(f"Unknown score-centering setting: {key}")
            continue
        if canonical in resolved and resolved[canonical] != value:
            raise ValueError(f"Conflicting SC aliases for {canonical}")
        if canonical == "enable_score_centering" and type(value) is not bool:
            raise ValueError("enable_score_centering/use_score_centering must be a YAML boolean")
        resolved[canonical] = value
    return resolved


def configure_score_centering(args, *, role_config=None, custom_config=None, cli_enabled=False):
    """Apply only SC keys; never change learning rates, checkpoint paths or PPO args."""
    common = (role_config or {}).get("score_centering", {})
    values = canonical_sc_values(common, strict=True)
    values.update(canonical_sc_values(custom_config or {}))
    if cli_enabled and values.get("enable_score_centering") is False:
        raise ValueError("Conflicting CLI enable and YAML disable for score centering")
    for key, value in values.items():
        setattr(args, key, value)
    # A single internal state: aliases must never re-enable a copied critic.
    for alias in SC_ALIASES:
        if hasattr(args, alias):
            delattr(args, alias)


def isolate_score_centering_role(base_args, role_args, overrides, role):
    """Permit an explicit critic disable and idempotent actor overrides only."""
    values = canonical_sc_values(overrides)
    enabled = getattr(base_args, "enable_score_centering", False)
    if role == "actor":
        for key, value in values.items():
            if value != getattr(base_args, key, SC_DEFAULTS[key]):
                raise ValueError(
                    f"Actor-only SC override {key} would desynchronize rollout; "
                    "put SC settings in the top-level score_centering block"
                )
    elif values.get("enable_score_centering") is True:
        raise ValueError("Score centering is actor-only; critic cannot enable it")
    if enabled or values:
        role_args.enable_score_centering = enabled if role == "actor" else False
        for alias in SC_ALIASES:
            if hasattr(role_args, alias):
                delattr(role_args, alias)
