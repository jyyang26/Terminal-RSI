"""Driver-side TB-GAE configuration; no tensor/rollout imports.

The shared switch is resolved before constructing rollout or role workers.
Only explicitly supplied CLI values override YAML, including explicit False.
"""

import argparse
import copy
import math
from collections.abc import Mapping

TB_KEYS = ("enable_tb_gae", "tb_gae_gamma_turn", "tb_gae_lambda_turn")
TB_HOOK = "slime_plugins.harbor.tb_gae.tb_gae_advantage"


class _ExplicitTBAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        value = option_string == "--enable-tb-gae" if self.nargs == 0 else values
        explicit = dict(getattr(namespace, "_tb_gae_cli_explicit", {}))
        if self.dest in explicit and explicit[self.dest] != value:
            parser.error(f"conflicting TB-GAE options for {self.dest}")
        explicit[self.dest] = value
        setattr(namespace, "_tb_gae_cli_explicit", explicit)
        setattr(namespace, self.dest, value)


def add_tb_gae_arguments(parser):
    parser.add_argument(
        "--enable-tb-gae",
        "--no-enable-tb-gae",
        dest="enable_tb_gae",
        action=_ExplicitTBAction,
        nargs=0,
        default=False,
        help="Explicitly enable/disable turn-based GAE (default: disabled).",
    )
    for key in TB_KEYS[1:]:
        parser.add_argument("--" + key.replace("_", "-"), type=float, default=1.0, action=_ExplicitTBAction)
    return parser


def validate_tb_cli_argv(argv):
    """Reject TB typos and boolean values that Megatron's parse_known_args hides."""
    allowed = {"--enable-tb-gae", "--no-enable-tb-gae", "--tb-gae-gamma-turn", "--tb-gae-lambda-turn"}
    for i, token in enumerate(argv):
        name = token.split("=", 1)[0]
        if name.startswith(("--tb-", "--enable-tb", "--no-enable-tb")) and name not in allowed:
            raise ValueError(f"Unknown TB-GAE option: {token}")
        if name in {"--enable-tb-gae", "--no-enable-tb-gae"}:
            if "=" in token or (i + 1 < len(argv) and not argv[i + 1].startswith("--")):
                raise ValueError(f"{name} takes no value; use --no-enable-tb-gae to disable")


def reject_tb_role_overrides(role_config):
    for entry in role_config.get("megatron", []):
        for field in ("overrides", "args"):
            local = entry.get(field) or {}
            invalid = set(TB_KEYS).intersection(local)
            if invalid:
                raise ValueError(f"TB-GAE is global; role {field} cannot contain {sorted(invalid)}")


def validate_tb_role_config(args, role_config):
    """Validate effective role args before even the first training worker exists."""
    reject_tb_role_overrides(role_config)
    if getattr(args, "enable_tb_gae", False) is not True:
        return
    for entry in role_config.get("megatron", []):
        effective = copy.copy(args)
        overrides = entry.get("overrides") or entry.get("args") or {}
        for key, value in overrides.items():
            setattr(effective, key, value)
        validate_tb_gae_config(effective)


def resolve_tb_gae_config(args, *, cli_explicit=None, task_config=None, role_config=None):
    task_config = {} if task_config is None else task_config
    role_config = {} if role_config is None else role_config
    if cli_explicit is None:
        cli_explicit = getattr(args, "_tb_gae_cli_explicit", {})
    layers = (role_config, task_config, cli_explicit)
    for layer in layers:
        if not isinstance(layer, Mapping):
            raise ValueError("TB-GAE configuration must be a mapping")
        if "enable_tb_gae" in layer and type(layer["enable_tb_gae"]) is not bool:
            raise ValueError("enable_tb_gae must be a YAML boolean true/false, not a string/number/null")
    reject_tb_role_overrides(role_config)
    values = dict(zip(TB_KEYS, (False, 1.0, 1.0), strict=True))
    for layer in layers:
        values.update({k: layer[k] for k in TB_KEYS if k in layer})
    args.enable_tb_gae = values["enable_tb_gae"]
    # Dormant discounts have no effect on the baseline and cannot enable TB.
    for key in TB_KEYS[1:]:
        value = values[key] if args.enable_tb_gae is True else 1.0
        if args.enable_tb_gae is True:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a finite number in (0, 1]")
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{key} must be a finite number in (0, 1]")
        setattr(args, key, float(value))
    vars(args).pop("_tb_gae_cli_explicit", None)
    hook = getattr(args, "custom_advantage_function_path", None)
    if hook == TB_HOOK:
        if args.enable_tb_gae is not True:
            raise ValueError("TB-GAE hook requires explicit enable_tb_gae=True")
        args.custom_advantage_function_path = None


def validate_tb_gae_config(args):
    enabled = getattr(args, "enable_tb_gae", False)
    if type(enabled) is not bool:
        raise ValueError("enable_tb_gae must be bool")
    if enabled is not True:
        return
    requirements = {
        "advantage_estimator": "ppo",
        "use_critic": True,
        "compute_advantages_and_returns": True,
        "qkv_format": "thd",
        "allgather_cp": False,
        "kl_coef": 0.0,
        "use_opd": False,
        "enable_score_centering": False,
        "harbor_sample_mode": "trajectory_retokenized",
        "custom_advantage_function_path": None,
        "custom_convert_samples_to_train_data_path": None,
        "harbor_include_subagent_rollouts": False,
        "use_rollout_routing_replay": False,
        "rollout_function_path": "slime_plugins.harbor.rollout.generate_rollout",
    }
    for key, expected in requirements.items():
        actual = getattr(args, key, expected)
        if actual != expected:
            raise ValueError(f"TB-GAE requires {key}={expected!r}, got {actual!r}")
    if getattr(args, "context_parallel_size", 1) not in (1, 4):
        raise ValueError("TB-GAE first release supports CP=1 or CP=4")
    for key in TB_KEYS[1:]:
        value = getattr(args, key, 1.0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= 1
        ):
            raise ValueError(f"{key} must be a finite number in (0, 1]")
