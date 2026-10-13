"""Length-Adaptive GAE for PPO.

Two decoupled lambdas instead of one:

    lambda_policy = 1 - 1/(alpha * L)     (advantages; alpha defaults to 1.5)
    lambda_critic = 1.0                   (returns)

Motivation: with a single shared lambda you must trade off policy-gradient bias
against critic-target variance. Long terminal-agent trajectories (L up to tens of
thousands of tokens here) want a lambda very close to 1 for the policy, while short
ones want more value bootstrapping. Making lambda a function of the sequence length
gives each sample the amount of bootstrapping its own horizon justifies, and pinning
the critic at lambda=1 gives it an unbiased Monte-Carlo regression target.

    L=1     -> lambda_policy = 0.333   (heavy bootstrapping, short horizon)
    L=10    -> 0.933
    L=100   -> 0.993
    L=5000  -> 0.99987               (essentially pure Monte-Carlo)

Wired in via ``--custom-advantage-function-path`` so slime core stays untouched:

    EXTRA_MISC_ARGS="... --custom-advantage-function-path \
        slime_plugins.harbor.length_adaptive_gae.length_adaptive_gae"

NOTE on the path format: ``slime.utils.misc.load_function`` does
``path.rpartition(".")``, so the separator is a DOT, not the colon shown in the
``--custom-advantage-function-path`` help string.

CONTEXT PARALLEL: we delegate to ``get_advantages_and_returns``, which already
all-gathers rewards/values over the CP group and re-slices the result. Its
``dist.nn.all_reduce`` is unconditional (no early-exit on empty chunks), and this
module's loop length is ``len(response_lengths)`` -- identical on every rank -- so
the per-rank collective count stays in lockstep and CP cannot desync here.

IMPORTANT consequences to be aware of when reading metrics:
  * ``returns != advantages + values`` any more. That identity only holds when the
    two lambdas are equal. This is intentional, but it means
    ``compute_value_explained_variance`` (loss.py) is now measuring the critic
    against a lambda=1 target rather than against its own bootstrap -- a shifted
    baseline, NOT a regression.
  * ``--normalize-advantages`` still runs AFTER this hook (loss.py) and will
    whiten whatever we return. Disable it if you want the raw scale.
"""

import os

import torch
from megatron.core import mpu

from slime.utils.ppo_utils import get_advantages_and_returns

DEFAULT_ALPHA = 1.5


def _policy_lambda(response_len: int, alpha: float) -> float:
    """lambda_policy = 1 - 1/(alpha * L), clamped to [0, 1).

    ``response_len`` can be 0 for a fully-masked sample; guard against div-by-zero
    and against the negative lambda that L < 1/alpha would produce (a negative
    lambda would make the GAE recursion alternate sign, which is meaningless).
    """
    if response_len <= 0:
        return 0.0
    lam = 1.0 - 1.0 / (alpha * response_len)
    return min(max(lam, 0.0), 1.0 - 1e-8)


def length_adaptive_gae(args, rollout_data) -> None:
    """Populate ``rollout_data["advantages"]`` / ``["returns"]`` in place.

    Contract per ``--custom-advantage-function-path``: called from
    ``compute_advantages_and_returns`` after KL is computed, only on the last
    pipeline stage, and must set both keys.
    """
    values = rollout_data["values"]
    rewards = rollout_data["rewards"]
    kl = rollout_data["kl"]
    response_lengths = rollout_data["response_lengths"]
    total_lengths = rollout_data["total_lengths"]

    alpha = float(os.environ.get("GAE_ALPHA", DEFAULT_ALPHA))
    assert alpha > 0, f"GAE_ALPHA must be positive, got {alpha}"
    assert values is not None, "length_adaptive_gae requires a critic (advantage_estimator=ppo)"

    # Reward assembly -- must stay byte-for-byte equivalent to the built-in "ppo"
    # branch in slime/backends/megatron_utils/loss.py: per-token reward is the
    # KL penalty, and the scalar terminal reward is added to the LAST token of
    # only cp_rank 0 (adding it on every CP rank would multiply it by cp_size).
    kl_coef = -args.kl_coef
    cp_rank = mpu.get_context_parallel_rank()
    token_rewards = []
    for reward, k in zip(rewards, kl, strict=False):
        k = k * kl_coef
        if cp_rank == 0:
            k[-1] += reward
        token_rewards.append(k)

    advantages: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    lambdas: list[float] = []

    for i, response_len in enumerate(response_lengths):
        lam_policy = _policy_lambda(response_len, alpha)
        lambdas.append(lam_policy)

        # advantages <- lambda_policy (length-adaptive)
        adv, _ = get_advantages_and_returns(
            total_lengths[i], response_len, values[i], token_rewards[i], args.gamma, lam_policy
        )
        # returns <- lambda_critic = 1.0 (unbiased Monte-Carlo target)
        _, ret = get_advantages_and_returns(
            total_lengths[i], response_len, values[i], token_rewards[i], args.gamma, 1.0
        )
        advantages.append(adv)
        returns.append(ret)

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns

    # Surface the realised lambda range so it is visible in the logs / wandb and a
    # degenerate distribution (e.g. everything pinned at 0) is obvious immediately.
    #
    # MUST be a per-sample list, NOT a scalar: log_rollout_data iterates over every
    # rollout_data key and only accepts list/tuple or torch.Tensor -- a bare float
    # hits its `raise ValueError(f"Unsupported type: ...")` and kills the run at the
    # END of the first joint actor step (advantages are already computed by then, so
    # the failure looks like a training crash rather than a metrics bug).
    # A list of len(response_lengths) also rides the standard (sum, count) DP/CP
    # reduction, so the reported mean is sample-weighted across ranks instead of a
    # mean-of-per-rank-means. min/max are emitted as flat per-sample lists too --
    # they reduce as means, so they read as "mean over ranks of the per-rank
    # extreme"; the mean is the metric to trust, the other two only flag a
    # degenerate spread.
    if lambdas:
        rollout_data["gae_lambda_policy_mean"] = list(lambdas)
        rollout_data["gae_lambda_policy_min"] = [min(lambdas)] * len(lambdas)
        rollout_data["gae_lambda_policy_max"] = [max(lambdas)] * len(lambdas)
