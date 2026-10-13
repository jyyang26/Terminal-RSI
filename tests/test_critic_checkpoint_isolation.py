"""Unit tests for critic checkpoint-dir isolation in placement_group.

Regression guard for the bug where the critic (a deepcopy of the actor args)
shared the actor's --save dir, so train_async's save-actor-then-save-critic
sequence had the critic overwrite the actor's checkpoint in place (the saved
iter dir ended up holding only the critic's value head, the actor LM head
clobbered).

Also guards the follow-up regression: an earlier fix suffixed --load too, which
crashed the critic at init on a cold start because the actor's --load points at
the shared pretrained backbone (no ``_critic`` sibling exists). --load must be
left untouched so the critic loads the same backbone as the actor.

Also guards the multi-day-actor-loss regression: the isolation call was wrapped
in ``if args.megatron_config_path is None:``, so a critic built from a per-role
--megatron-config-path (which does NOT set a role-specific --save) skipped
isolation entirely and clobbered the actor's checkpoint. The suffixing is now
idempotent and called unconditionally; these tests pin that behavior.
"""

import copy
from argparse import Namespace

import pytest

import slime.ray.placement_group as placement_group
from slime.ray.placement_group import (
    _assert_actor_critic_checkpoint_dirs_isolated,
    _isolate_critic_checkpoint_dirs,
)


class TestCriticCheckpointIsolation:
    def test_save_suffixed_load_untouched(self):
        # cold start: actor --load = the shared pretrained backbone. Only --save
        # may be redirected; --load must stay pointing at the backbone (which has
        # no _critic sibling), or the critic assert-crashes at init.
        actor = Namespace(save="/data/run/Qwen-27B", load="/ckpt/backbone", use_critic=True)
        critic = copy.deepcopy(actor)

        _isolate_critic_checkpoint_dirs(critic)

        # critic --save redirected to a sibling dir
        assert critic.save == "/data/run/Qwen-27B_critic"
        # critic --load left exactly as-is (shared backbone, no _critic suffix)
        assert critic.load == "/ckpt/backbone"
        # actor left untouched (no collision)
        assert actor.save == "/data/run/Qwen-27B"
        assert actor.load == "/ckpt/backbone"

    def test_trailing_slash_stripped_before_suffix(self):
        critic = Namespace(save="/data/run/x/", load="/ckpt/backbone/")
        _isolate_critic_checkpoint_dirs(critic)
        assert critic.save == "/data/run/x_critic"
        # --load never suffixed, so its trailing slash is irrelevant / untouched
        assert critic.load == "/ckpt/backbone/"

    def test_none_save_left_none(self):
        # the suffix must not turn a None --save into the string "None_critic".
        critic = Namespace(save=None, load="/ckpt/backbone")
        _isolate_critic_checkpoint_dirs(critic)
        assert critic.save is None
        assert critic.load == "/ckpt/backbone"

    def test_none_load_left_none(self):
        critic = Namespace(save="/ckpt/foo", load=None)
        _isolate_critic_checkpoint_dirs(critic)
        assert critic.save == "/ckpt/foo_critic"
        assert critic.load is None

    def test_cold_load_keeps_shared_backbone(self, tmp_path):
        backbone = tmp_path / "backbone"
        backbone.mkdir()
        critic = Namespace(save="/data/run/model", load=str(backbone))

        _isolate_critic_checkpoint_dirs(critic)

        assert critic.load == str(backbone)
        assert not hasattr(critic, "no_load_optim")
        assert not hasattr(critic, "no_load_rng")
        assert not hasattr(critic, "finetune")

    def test_warm_load_uses_critic_weights_without_training_state(self, tmp_path):
        actor_checkpoint = tmp_path / "model"
        critic_checkpoint = tmp_path / "model_critic"
        actor_checkpoint.mkdir()
        critic_checkpoint.mkdir()
        critic = Namespace(save=str(actor_checkpoint), load=str(actor_checkpoint))

        _isolate_critic_checkpoint_dirs(critic)

        assert critic.save == str(critic_checkpoint)
        assert critic.load == str(critic_checkpoint)
        assert critic.no_load_optim is True
        assert critic.no_load_rng is True
        assert critic.finetune is True

    def test_critic_load_optim_keeps_training_state(self, tmp_path):
        """``--critic-load-optim`` must preserve the critic's optimizer/RNG.

        Resuming a run into itself is the one case the weights-only default
        gets wrong: the LR schedule matches by construction, so discarding the
        value head's Adam moments only resets its explained-variance warmup.
        """
        actor_checkpoint = tmp_path / "model"
        critic_checkpoint = tmp_path / "model_critic"
        actor_checkpoint.mkdir()
        critic_checkpoint.mkdir()
        critic = Namespace(save=str(actor_checkpoint), load=str(actor_checkpoint), critic_load_optim=True)

        _isolate_critic_checkpoint_dirs(critic)

        # isolation must still hold -- the flag only affects training state
        assert critic.save == str(critic_checkpoint)
        assert critic.load == str(critic_checkpoint)
        assert critic.no_load_optim is False
        assert critic.no_load_rng is False
        assert critic.finetune is False

    def test_critic_load_optim_is_noop_on_cold_start(self, tmp_path):
        """No ``_critic`` sibling: the flag must not force an optimizer load.

        On a cold start the critic loads the shared backbone and reinitializes
        its value head; there is no critic optimizer to restore, so the flag
        must leave the load path and training-state args untouched.
        """
        actor_checkpoint = tmp_path / "model"
        actor_checkpoint.mkdir()
        critic = Namespace(save=str(actor_checkpoint), load=str(actor_checkpoint), critic_load_optim=True)

        _isolate_critic_checkpoint_dirs(critic)

        assert critic.load == str(actor_checkpoint)
        assert not hasattr(critic, "no_load_optim")

    def test_actor_and_critic_never_share_save_dir(self):
        # the core invariant: after isolation, the SAVE dirs are not shared, so
        # the critic cannot overwrite the actor's checkpoint in place.
        actor = Namespace(save="/s", load="/ckpt/backbone")
        critic = copy.deepcopy(actor)
        _isolate_critic_checkpoint_dirs(critic)
        assert critic.save != actor.save
        # --load IS shared on cold start — that's correct (same backbone).
        assert critic.load == actor.load

    def test_suffix_is_idempotent(self):
        # the call site now invokes this unconditionally (no megatron_config_path
        # guard), so applying it to already-isolated args must be a no-op — never
        # "/s_critic_critic".
        critic = Namespace(save="/data/run/x", load="/ckpt/backbone")
        _isolate_critic_checkpoint_dirs(critic)
        assert critic.save == "/data/run/x_critic"
        _isolate_critic_checkpoint_dirs(critic)
        assert critic.save == "/data/run/x_critic"

    def test_idempotent_with_trailing_slash(self):
        critic = Namespace(save="/data/run/x_critic/", load=None)
        _isolate_critic_checkpoint_dirs(critic)
        # already a critic dir (modulo trailing slash) -> not double-suffixed
        assert critic.save == "/data/run/x_critic/"

    def test_config_driven_critic_still_isolated(self):
        # Regression for the multi-day actor-weights loss: a per-role megatron
        # config only overrides lr/kl/etc. and leaves --save inherited from the
        # actor. Isolation must still redirect --save so the critic does not
        # overwrite the actor checkpoint. (Simulates critic_args as returned by
        # parse_megatron_role_args: same --save as the actor, tweaked lr.)
        actor_save = "/data/run/Qwen-27B"
        critic = Namespace(save=actor_save, load="/ckpt/backbone", lr=5e-6)
        _isolate_critic_checkpoint_dirs(critic)
        assert critic.save == "/data/run/Qwen-27B_critic"
        assert critic.save != actor_save

    def test_assert_rejects_equivalent_save_dirs(self):
        actor = Namespace(save="/data/run/model")
        critic = Namespace(save="/data/run/./model/")

        with pytest.raises(AssertionError, match="checkpoint directories must differ"):
            _assert_actor_critic_checkpoint_dirs_isolated(actor, critic)

    def test_config_call_path_asserts_before_critic_allocation(self, monkeypatch):
        args = Namespace(
            megatron_config_path="roles.yaml",
            use_critic=True,
            save="/data/run/model",
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            critic_num_nodes=1,
            critic_num_gpus_per_node=8,
        )

        def parse_role_args(source, _path, role):
            parsed = copy.deepcopy(source)
            parsed.role = role
            return parsed

        allocations = []

        def fake_allocate_train_group(**kwargs):
            allocations.append(kwargs.get("role", "actor"))
            return object()

        monkeypatch.setattr("slime.utils.arguments.parse_megatron_role_args", parse_role_args)
        monkeypatch.setattr(placement_group, "allocate_train_group", fake_allocate_train_group)
        monkeypatch.setattr(placement_group, "_isolate_critic_checkpoint_dirs", lambda _args: None)

        with pytest.raises(AssertionError, match="checkpoint directories must differ"):
            placement_group.create_training_models(args, {"actor": object(), "critic": object()}, None)

        assert allocations == ["actor"]
