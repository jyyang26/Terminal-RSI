"""Offline SC parser, common YAML, actor/critic and warm checkpoint contracts."""

import copy
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pytest
import yaml

from slime.utils.arguments import get_slime_extra_args_provider, parse_megatron_role_args
from slime.utils.score_centering_config import SC_DEFAULTS, configure_score_centering
from slime.utils.score_centering_validate import validate_score_centering_args

NUM_GPUS = 0
pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]
ROLES = ROOT / "examples/terminal_rsi/configs/megatron_roles_rpr1.yaml"
SC_SHADOW_ROLES = ROOT / "tests/fixtures/megatron_roles_rpr1_sc_shadow.yaml"


def make_args(**updates):
    values = dict(
        SC_DEFAULTS,
        advantage_estimator="ppo",
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=None,
        keep_old_actor=True,
        use_rollout_logprobs=False,
        use_tis=False,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        qkv_format="thd",
        allgather_cp=False,
        loss_type="custom_loss",
        custom_loss_function_path="slime_plugins.harbor.train_loss.policy_loss_no_entropy_when_disabled",
        rollout_function_path="slime_plugins.harbor.rollout.generate_rollout",
        harbor_sample_mode="trajectory_retokenized",
        vocab_size=248320,
        lr=5e-7,
        min_lr=5e-7,
        clip_grad=1.0,
        value_clip=0.2,
        kl_coef=0.0,
        use_opd=False,
        custom_advantage_function_path="slime_plugins.harbor.tb_gae.compute",
        untie_embeddings_and_output_weights=False,
        load="/tmp/actor-checkpoint",
        save="/tmp/actor-save",
        no_load_optim=True,
        no_load_rng=True,
        finetune=True,
    )
    values.update(updates)
    return Namespace(**values)


@pytest.mark.parametrize("flag", ["--use-score-centering", "--enable-score-centering"])
def test_real_parser_aliases_and_defaults(flag):
    parser = get_slime_extra_args_provider()(ArgumentParser())
    defaults = parser.parse_args(["--rollout-batch-size", "8"])
    assert {key: getattr(defaults, key) for key in SC_DEFAULTS} == SC_DEFAULTS
    enabled = parser.parse_args(["--rollout-batch-size", "8", flag, "--score-centering-top-k", "64"])
    assert enabled.enable_score_centering and enabled.sc_top_k == 64
    assert not hasattr(enabled, "use_score_centering")


def test_off_validator_does_not_read_other_attributes():
    class Off:
        enable_score_centering = False

        def __getattr__(self, key):
            raise AssertionError(key)

    validate_score_centering_args(Off())


@pytest.mark.parametrize(
    "change",
    [
        {"sc_top_k": 0},
        {"sc_top_k": True},
        {"sc_top_k": 1025},
        {"sc_tail_eps": 1e-6},
        {"sc_tail_eps": float("nan")},
        {"sc_chunk_size": False},
        {"sc_chunk_size": 0},
        {"sc_correction_scale": float("nan")},
        {"sc_correction_scale": -1.0},
        {"sc_formula": "reinforce"},
        {"sc_assert_alignment": "off"},
        {"sc_max_trial_bytes": 0},
        {"sc_max_batch_bytes": True},
        {"sc_wire_format": "auto"},
        {"advantage_estimator": "gspo"},
        {"use_tis": True},
        {"use_opsm": True},
        {"use_opd": True},
        {"eps_clip_c": 3.0},
        {"rollout_top_p": 0.95},
        {"rollout_temperature": 0.9},
        {"rollout_top_k": 32},
        {"qkv_format": "bshd", "allgather_cp": True},
        {"loss_type": "value_loss"},
        {"custom_loss_function_path": "invalid"},
        {"harbor_sample_mode": "turn"},
        {"sc_formula": "ppo_weighted", "sc_weight_denominator": "sampler"},
        {"sc_formula": "ppo_weighted", "keep_old_actor": False},
        {"sc_formula": "ppo_weighted", "use_rollout_logprobs": True, "sc_weight_denominator": "old_actor"},
    ],
)
def test_invalid_enabled_config_fails(change):
    with pytest.raises(ValueError):
        validate_score_centering_args(make_args(enable_score_centering=True, **change))


@pytest.mark.parametrize("formula", ["score", "ppo_weighted"])
def test_enabled_baseline_recipe(formula):
    validate_score_centering_args(make_args(enable_score_centering=True, sc_formula=formula))


def test_sampler_denominator_must_match_ppo():
    validate_score_centering_args(
        make_args(
            enable_score_centering=True,
            sc_formula="ppo_weighted",
            use_rollout_logprobs=True,
            sc_weight_denominator="sampler",
        )
    )


@pytest.mark.parametrize("switch", [False, True])
def test_common_warmcritic_yaml_preserves_roles_and_checkpoint_loading(tmp_path, switch):
    base = make_args()
    original = copy.deepcopy(vars(base))
    raw = yaml.safe_load(ROLES.read_text())
    raw["score_centering"] = {"enable_score_centering": switch, "sc_top_k": 32}
    raw["megatron"][1]["overrides"]["enable_score_centering"] = False
    role_file = tmp_path / "roles.yaml"
    role_file.write_text(yaml.safe_dump(raw))
    configure_score_centering(base, role_config=raw)
    actor = parse_megatron_role_args(base, str(role_file), "actor")
    critic = parse_megatron_role_args(base, str(role_file), "critic")
    assert actor.enable_score_centering is switch and critic.enable_score_centering is False
    assert actor.lr == 5e-7 and critic.lr == 1.5e-5 and critic.lr / actor.lr == pytest.approx(30)
    assert actor.clip_grad == 1.0 and critic.clip_grad == 100.0 and critic.value_clip == 4.0
    for key in ("load", "save", "no_load_optim", "no_load_rng", "finetune"):
        assert getattr(actor, key) == original[key] and getattr(critic, key) == original[key]
    from slime.ray.placement_group import _isolate_critic_checkpoint_dirs

    backbone = tmp_path / "model"
    warm = tmp_path / "model_critic"
    backbone.mkdir()
    warm.mkdir()
    critic.load = str(backbone)
    critic.save = str(backbone)
    _isolate_critic_checkpoint_dirs(critic)
    assert critic.load == str(warm) and critic.save == str(warm)
    assert critic.no_load_optim and critic.no_load_rng and critic.finetune
    assert base.enable_score_centering is switch


def test_original_role_yaml_missing_sc_unchanged():
    base = make_args()
    actor = parse_megatron_role_args(base, str(ROLES), "actor")
    critic = parse_megatron_role_args(base, str(ROLES), "critic")
    assert not actor.enable_score_centering and not critic.enable_score_centering
    assert actor.lr == 5e-7 and critic.lr == 1.5e-5
    assert not hasattr(base, "score_centering")


@pytest.mark.parametrize("role,value", [("actor", True), ("critic", True)])
def test_role_only_enable_fails_before_rollout_desynchronizes(tmp_path, role, value):
    file = tmp_path / "bad.yaml"
    file.write_text(yaml.safe_dump({"megatron": [{"role": role, "overrides": {"enable_score_centering": value}}]}))
    with pytest.raises(ValueError):
        parse_megatron_role_args(make_args(), str(file), role)


def test_common_config_and_custom_yaml_precedence_and_aliases():
    args = make_args()
    configure_score_centering(
        args,
        role_config={"score_centering": {"enable_score_centering": True, "sc_top_k": 32}},
        custom_config={"use_score_centering": False, "score_centering_top_k": 64},
    )
    assert args.enable_score_centering is False and args.sc_top_k == 64
    with pytest.raises(ValueError):
        configure_score_centering(args, custom_config={"use_score_centering": False}, cli_enabled=True)
    with pytest.raises(ValueError):
        configure_score_centering(args, custom_config={"use_score_centering": True, "enable_score_centering": False})
    with pytest.raises(ValueError):
        configure_score_centering(args, role_config={"score_centering": {"enable_score_centering": "false"}})


def test_derived_shadow_file_only_adds_common_sc():
    original = yaml.safe_load(ROLES.read_text())
    shadow = yaml.safe_load(SC_SHADOW_ROLES.read_text())
    assert original["megatron"] == shadow["megatron"]
    base = make_args()
    configure_score_centering(base, role_config=shadow)
    validate_score_centering_args(base)
    assert base.sc_wire_format == "json" and base.sc_correction_scale == 0.0
