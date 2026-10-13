"""Validation runs only after SC is explicitly enabled."""

import math

from slime.utils.score_centering_config import SC_DEFAULTS


def validate_score_centering_args(args):
    if not getattr(args, "enable_score_centering", False):
        return
    for key, default in SC_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, default)
    if type(args.enable_score_centering) is not bool:
        raise ValueError("enable_score_centering must be a boolean")
    if getattr(args, "use_score_centering", True) is not True:
        raise ValueError("Conflicting SC switches")
    if args.sc_weight_denominator not in {"sampler", "old_actor"}:
        raise ValueError("Unknown SC weight denominator")
    if args.sc_formula not in {"score", "ppo_weighted"}:
        raise ValueError("Unknown SC formula")
    if isinstance(args.sc_top_k, bool) or not isinstance(args.sc_top_k, int) or not 1 <= args.sc_top_k <= 1024:
        raise ValueError("SC top-k must lie in [1,1024]")
    vocab = getattr(args, "vocab_size", None)
    if vocab is not None and args.sc_top_k > vocab:
        raise ValueError("SC top-k exceeds vocabulary size")
    if args.sc_tail_eps != 1e-8:
        raise ValueError("This integration fixes SC tail epsilon at 1e-8")
    if (
        isinstance(args.sc_correction_scale, bool)
        or not isinstance(args.sc_correction_scale, (int, float))
        or not math.isfinite(args.sc_correction_scale)
        or args.sc_correction_scale < 0
    ):
        raise ValueError("SC scale must be finite and nonnegative")
    if isinstance(args.sc_chunk_size, bool) or not isinstance(args.sc_chunk_size, int) or args.sc_chunk_size < 1:
        raise ValueError("SC chunk size must be positive")
    for name in ("sc_max_response_bytes", "sc_max_batch_bytes", "sc_max_trial_bytes"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if args.sc_wire_format not in {"binary", "json"}:
        raise ValueError("SC wire format must be binary or json")
    if args.sc_assert_alignment != "full":
        raise ValueError("Production SC requires full alignment checks")
    if getattr(args, "train_backend", "megatron") != "megatron":
        raise ValueError("SC requires Megatron")
    if args.advantage_estimator != "ppo":
        raise ValueError("This integration requires PPO advantages")
    if args.qkv_format not in {"thd", "bshd"}:
        raise ValueError("Unknown QKV layout")
    if args.allgather_cp and args.qkv_format != "thd":
        raise ValueError("SC allgather-CP requires THD, matching get_responses")
    if args.loss_type == "custom_loss":
        expected = "slime_plugins.harbor.train_loss.policy_loss_no_entropy_when_disabled"
        if args.custom_loss_function_path != expected:
            raise ValueError("SC custom loss must use the audited Harbor mount")
    elif args.loss_type != "policy_loss":
        raise ValueError("SC requires a policy loss")
    if getattr(args, "rollout_function_path", None) != "slime_plugins.harbor.rollout.generate_rollout":
        raise ValueError("SC data transport in this port requires Harbor rollout")
    if getattr(args, "harbor_sample_mode", "turn") not in {"trajectory_normalized", "trajectory_retokenized"}:
        raise ValueError("SC requires an audited normalized Harbor trajectory builder")
    if not getattr(args, "harbor_fail_on_conversion_error", True):
        raise ValueError("SC conversion errors must terminate the rollout")
    for name in ("use_tis", "get_mismatch_metrics", "use_opsm", "use_opd", "use_unbiased_kl", "enable_mtp_training"):
        if getattr(args, name, False):
            raise ValueError(f"SC does not support {name}")
    for name in (
        "custom_tis_function_path",
        "custom_pg_loss_reducer_function_path",
        "custom_convert_samples_to_train_data_path",
        "rollout_logprob_server_url",
        "rollout_sample_filter_path",
        "rollout_sample_postprocess_path",
    ):
        if getattr(args, name, None):
            raise ValueError(f"SC requires an audited adapter for {name}")
    permitted_hooks = {
        "rollout_data_postprocess_path": "slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data",
        "rollout_all_samples_process_path": "slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples",
    }
    for name, supported in permitted_hooks.items():
        if getattr(args, name, None) not in (None, supported):
            raise ValueError(f"SC only supports the audited token-cap hook for {name}")
    if not getattr(args, "compute_advantages_and_returns", True):
        raise ValueError("SC requires the standard actor advantage/old-policy lifecycle")
    if getattr(args, "eps_clip_c", None) is not None:
        raise ValueError("SC does not support dual-clip")
    high = args.eps_clip if args.eps_clip_high is None else args.eps_clip_high
    if not (math.isfinite(args.eps_clip) and 0 <= args.eps_clip < 1):
        raise ValueError("SC requires finite 0 <= eps_clip < 1")
    if not math.isfinite(high) or high < 0:
        raise ValueError("SC requires finite nonnegative eps_clip_high")
    if args.rollout_temperature != 1 or args.rollout_top_p != 1 or args.rollout_top_k != -1:
        raise ValueError("This SC transport requires temperature=1, top_p=1, top_k=-1")
    if args.sc_formula == "ppo_weighted":
        if args.use_rollout_logprobs:
            if args.sc_weight_denominator != "sampler":
                raise ValueError("PPO denominator is sampler; select --sc-weight-denominator sampler")
        elif args.sc_weight_denominator != "old_actor" or not args.keep_old_actor:
            raise ValueError("PPO old-actor denominator requires old_actor SC and --keep-old-actor")
