import argparse
import copy

import pytest
import yaml
from tb_gae_test_utils import tb_args

from slime.utils.tb_gae_config import (
    TB_HOOK,
    TB_KEYS,
    add_tb_gae_arguments,
    resolve_tb_gae_config,
    validate_tb_cli_argv,
    validate_tb_gae_config,
)

NUM_GPUS = 0
pytestmark = pytest.mark.unit


def parse(flags=()):
    validate_tb_cli_argv(flags)
    return add_tb_gae_arguments(argparse.ArgumentParser(allow_abbrev=False)).parse_args(flags)


@pytest.mark.parametrize(
    "flags,task,role,expected",
    [
        ([], {}, {}, False),
        ([], {"enable_tb_gae": False}, {}, False),
        ([], {}, {"enable_tb_gae": True}, True),
        ([], {"enable_tb_gae": False}, {"enable_tb_gae": True}, False),
        (["--enable-tb-gae"], {"enable_tb_gae": False}, {}, True),
        (["--no-enable-tb-gae"], {"enable_tb_gae": True}, {}, False),
        (["--tb-gae-gamma-turn", "0.999"], {}, {}, False),
    ],
)
def test_switch_precedence(flags, task, role, expected):
    args = parse(flags)
    resolve_tb_gae_config(args, task_config=task, role_config=role)
    assert args.enable_tb_gae is expected
    assert type(args.enable_tb_gae) is bool


def test_float_precedence_and_default():
    args = parse(["--enable-tb-gae", "--tb-gae-lambda-turn", "1.0"])
    resolve_tb_gae_config(
        args,
        task_config={"tb_gae_gamma_turn": 0.9},
        role_config={"tb_gae_gamma_turn": 0.8, "tb_gae_lambda_turn": 0.999},
    )
    assert (args.tb_gae_gamma_turn, args.tb_gae_lambda_turn) == (0.9, 1.0)
    args = parse()
    assert (args.enable_tb_gae, args.tb_gae_gamma_turn, args.tb_gae_lambda_turn) == (False, 1.0, 1.0)


@pytest.mark.parametrize("value", ['"false"', '"true"', "0", "1", "null"])
@pytest.mark.parametrize("layer", ["task_config", "role_config"])
def test_boolean_schema_not_truthiness(value, layer):
    with pytest.raises(ValueError, match="boolean"):
        resolve_tb_gae_config(parse(["--no-enable-tb-gae"]), **{layer: yaml.safe_load("enable_tb_gae: " + value)})


@pytest.mark.parametrize("key", TB_KEYS)
@pytest.mark.parametrize("field", ["overrides", "args"])
def test_role_local_injection_rejected(key, field):
    with pytest.raises(ValueError, match="global"):
        resolve_tb_gae_config(parse(), role_config={"megatron": [{"role": "critic", field: {key: False}}]})


@pytest.mark.parametrize(
    "flags",
    [
        ["--enable-tb-gae", "false"],
        ["--enable-tb-gae=true"],
        ["--tb-gae-lamda-turn", "1"],
        ["--enable-tb-gae", "--no-enable-tb-gae"],
        ["--tb-gae-gamma-turn", "0.9", "--tb-gae-gamma-turn", "0.8"],
    ],
)
def test_bad_cli_rejected(flags):
    with pytest.raises((ValueError, SystemExit)):
        parse(flags)


@pytest.mark.parametrize("value", [True, 0.0, -1.0, 1.1, float("nan"), float("inf"), "0.999"])
def test_invalid_active_discount(value):
    with pytest.raises(ValueError, match="finite"):
        resolve_tb_gae_config(parse(), task_config={"enable_tb_gae": True, "tb_gae_gamma_turn": value})


@pytest.mark.parametrize(
    "changes",
    [
        {"qkv_format": "bshd"},
        {"allgather_cp": True},
        {"context_parallel_size": 2},
        {"advantage_estimator": "grpo"},
        {"kl_coef": 0.1},
        {"use_opd": True},
        {"enable_score_centering": True},
        {"harbor_sample_mode": "turn"},
        {"custom_advantage_function_path": "other.hook"},
        {"use_critic": False},
        {"custom_convert_samples_to_train_data_path": "other.convert"},
    ],
)
def test_active_support_guards(changes):
    with pytest.raises(ValueError):
        validate_tb_gae_config(tb_args(**changes))


def test_off_validator_reads_only_switch():
    class Off:
        enable_tb_gae = False

        def __getattr__(self, key):
            raise AssertionError(key)

    validate_tb_gae_config(Off())


def test_hook_cannot_bypass_switch():
    args = parse()
    args.custom_advantage_function_path = TB_HOOK
    with pytest.raises(ValueError, match="explicit"):
        resolve_tb_gae_config(args)
    args = parse(["--enable-tb-gae"])
    args.custom_advantage_function_path = TB_HOOK
    resolve_tb_gae_config(args)
    assert args.custom_advantage_function_path is None


def test_real_parser_registration_and_role_copies(tmp_path):
    from slime.utils.arguments import get_slime_extra_args_provider, parse_megatron_role_args

    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(["--rollout-batch-size", "8", "--enable-tb-gae"])
    args.advantage_estimator = "ppo"
    args.harbor_sample_mode = "trajectory_retokenized"
    args.rollout_function_path = "slime_plugins.harbor.rollout.generate_rollout"
    roles = {
        "enable_tb_gae": True,
        "tb_gae_gamma_turn": 0.999,
        "tb_gae_lambda_turn": 0.999,
        "megatron": [{"role": "actor", "overrides": {"lr": 5e-7}}, {"role": "critic", "overrides": {"lr": 1.5e-5}}],
    }
    resolve_tb_gae_config(args, role_config=roles)
    path = tmp_path / "roles.yaml"
    path.write_text(yaml.safe_dump(roles))
    for role in ("actor", "critic"):
        out = parse_megatron_role_args(args, str(path), role)
        assert (out.enable_tb_gae, out.tb_gae_gamma_turn, out.tb_gae_lambda_turn) == (True, 0.999, 0.999)
    invalid = copy.deepcopy(roles)
    invalid["megatron"][1]["overrides"]["enable_tb_gae"] = False
    path.write_text(yaml.safe_dump(invalid))
    with pytest.raises(ValueError, match="global"):
        parse_megatron_role_args(args, str(path), "actor")


@pytest.mark.parametrize(
    "flags,task_enabled,role_enabled,expected",
    [
        ([], None, None, False),
        ([], True, False, True),
        ([], False, True, False),
        (["--enable-tb-gae"], False, False, True),
        (["--no-enable-tb-gae"], True, True, False),
    ],
)
def test_actual_slime_yaml_merge_before_worker_creation(
    monkeypatch, tmp_path, flags, task_enabled, role_enabled, expected
):
    import sys

    from slime.backends.megatron_utils.arguments import megatron_parse_args
    from slime.utils.arguments import _pre_parse_mode, get_slime_extra_args_provider, slime_validate_args

    task = {"harbor_sample_mode": "trajectory_retokenized", "tb_gae_gamma_turn": 0.999, "tb_gae_lambda_turn": 0.999}
    roles = {"megatron": [{"role": "actor", "overrides": {}}, {"role": "critic", "overrides": {}}]}
    if task_enabled is not None:
        task["enable_tb_gae"] = task_enabled
    if role_enabled is not None:
        roles["enable_tb_gae"] = role_enabled
    task_file, role_file = tmp_path / "task.yaml", tmp_path / "roles.yaml"
    task_file.write_text(yaml.safe_dump(task))
    role_file.write_text(yaml.safe_dump(roles))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test",
            "--rollout-batch-size",
            "8",
            "--num-rollout",
            "1",
            "--advantage-estimator",
            "ppo",
            "--rollout-function-path",
            "slime_plugins.harbor.rollout.generate_rollout",
            "--custom-config-path",
            str(task_file),
            "--megatron-config-path",
            str(role_file),
            *flags,
        ],
    )
    args = megatron_parse_args(get_slime_extra_args_provider(), skip_hf_validate=True)
    vars(args).update(vars(_pre_parse_mode()))
    slime_validate_args(args)
    assert args.enable_tb_gae is expected
    assert args.tb_gae_gamma_turn == (0.999 if expected else 1.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"qkv_format": "bshd"},
        {"kl_coef": 0.1},
        {"allgather_cp": True},
        {"custom_advantage_function_path": "other.hook"},
    ],
)
def test_incompatible_role_override_fails_globally(overrides):
    from slime.utils.tb_gae_config import validate_tb_role_config

    with pytest.raises(ValueError):
        validate_tb_role_config(tb_args(), {"megatron": [{"role": "critic", "overrides": overrides}]})
