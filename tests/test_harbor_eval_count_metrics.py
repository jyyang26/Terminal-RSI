"""Unit tests for the absolute-count eval metrics in the Harbor rollout.

Regression guard for the gap the user hit: eval only logged ``eval/harbor`` (a
mean reward whose denominator is the number of *scored* samples, not the number
of eval tasks), so failed / timed-out tasks vanished from the picture and you
could not tell how many tasks were correct vs incorrect vs dropped. These tests
pin the self-consistency invariants of ``_compute_eval_count_metrics``:

    resolved + unresolved          == num_scored_tasks (task-level pass@k buckets)
    num_tasks - num_scored_tasks   == num_unscored_tasks (tasks that never scored)
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

# rollout.py imports torch/ray at module load; stub them so this stays CPU-only.
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.dtype = type("dtype", (), {})
    torch_stub.Size = tuple
    torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch_stub

if "ray" not in sys.modules:
    ray_stub = types.ModuleType("ray")
    ray_stub._private = SimpleNamespace(services=SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1"))
    sys.modules["ray"] = ray_stub

from slime_plugins.harbor.rollout import _compute_eval_count_metrics


class TestEvalCountMetrics:
    def test_all_tasks_scored_binary_rewards(self):
        # 10 tasks, n=1, 4 resolved (reward 1.0), 6 unresolved (reward 0.0).
        rewards = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        m = _compute_eval_count_metrics(
            rewards=rewards, num_eval_tasks=10, num_scored_groups=10, n_per_prompt=1
        )
        assert m["harbor/eval_num_tasks"] == 10
        assert m["harbor/eval_num_samples"] == 10
        assert m["harbor/eval_num_resolved"] == 4
        assert m["harbor/eval_num_unresolved"] == 6
        assert m["harbor/eval_num_unscored_tasks"] == 0
        # scored-only mean == resolved rate when nothing dropped
        assert m["harbor/eval_reward_mean"] == 0.4
        assert m["harbor/eval_resolved_rate_full"] == 0.4

    def test_resolved_plus_unresolved_equals_num_tasks(self):
        rewards = [1.0, 0.0, 1.0, 0.0, 0.0]
        m = _compute_eval_count_metrics(
            rewards=rewards, num_eval_tasks=8, num_scored_groups=5, n_per_prompt=1
        )
        # eval_num_resolved/unresolved are now pass@1 TASK counts over the full eval
        # set, so they sum to num_eval_tasks (missing tasks count as unresolved).
        assert m["harbor/eval_num_resolved"] + m["harbor/eval_num_unresolved"] == m["harbor/eval_num_tasks"]

    def test_dropped_tasks_split_the_two_rates(self):
        # 89 eval tasks, but only 79 produced a scored sample (10 failed/timed out).
        # 32 of the 79 resolved. The scored-only mean (32/79) is OPTIMISTIC; the
        # full-set rate (32/89) is the honest one. Both must be reported, and the
        # 10 dropped tasks must show up as unscored.
        rewards = [1.0] * 32 + [0.0] * 47  # 79 scored samples
        m = _compute_eval_count_metrics(
            rewards=rewards, num_eval_tasks=89, num_scored_groups=79, n_per_prompt=1
        )
        assert m["harbor/eval_num_samples"] == 79
        assert m["harbor/eval_num_resolved"] == 32
        # unresolved is now pass@1 over the FULL eval set: 89 - 32 = 57 (the 10
        # unscored tasks are unresolved failures too).
        assert m["harbor/eval_num_unresolved"] == 57
        assert m["harbor/eval_num_unscored_tasks"] == 10
        assert abs(m["harbor/eval_reward_mean"] - 32 / 79) < 1e-9
        assert abs(m["harbor/eval_resolved_rate_full"] - 32 / 89) < 1e-9
        # the honest full-set rate is strictly below the scored-only mean here
        assert m["harbor/eval_resolved_rate_full"] < m["harbor/eval_reward_mean"]

    def test_threshold_buckets_near_binary_values(self):
        # 0.5 threshold: robust to float noise on a nominally binary reward.
        rewards = [0.999, 1.0, 0.5, 0.4999, 0.0]
        m = _compute_eval_count_metrics(
            rewards=rewards, num_eval_tasks=5, num_scored_groups=5, n_per_prompt=1
        )
        # >= 0.5 resolved: 0.999, 1.0, 0.5  -> 3 ; below: 0.4999, 0.0 -> 2
        assert m["harbor/eval_num_resolved"] == 3
        assert m["harbor/eval_num_unresolved"] == 2

    def test_pass_at_3_resolves_task_when_any_attempt_succeeds(self):
        # Four tasks, three independent attempts each. Task outcomes are:
        #   resolved, unresolved, resolved, unresolved -> pass@3 = 2/4.
        by_task = [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        rewards = [reward for task in by_task for reward in task]
        m = _compute_eval_count_metrics(
            rewards=rewards,
            rewards_by_task=by_task,
            num_eval_tasks=4,
            num_scored_groups=4,
            n_per_prompt=3,
        )
        assert m["harbor/eval_pass_at_k"] == 3
        assert m["harbor/eval_num_samples"] == 12
        # eval_num_resolved is now the pass@1 (rank-0) task count: first attempts
        # [0,0,1,0] -> 1 resolved, 3 unresolved (over 4 tasks).
        assert m["harbor/eval_num_resolved"] == 1
        assert m["harbor/eval_num_unresolved"] == 3
        # pass@k (any-of-3) resolved-task count kept under its own key: 2/4.
        assert m["harbor/eval_num_resolved_pass_at_k"] == 2
        assert m["harbor/eval_num_unresolved_pass_at_k"] == 2
        assert m["harbor/eval_num_resolved_attempts"] == 3
        # eval_resolved_rate_full is the avg@3 alias (resolved attempts / (4*3) = 3/12).
        assert m["harbor/eval_resolved_rate_full"] == 3 / 12
        assert m["harbor/eval_reward_mean"] == 0.5
        assert m["harbor/eval_attempt_reward_mean"] == 3 / 12
        assert m["harbor/eval_resolved_rate_full_pass@3"] == 0.5
        assert m["harbor/eval_resolved_rate_full_avg@3"] == 3 / 12
        assert m["harbor/eval_resolved_rate_full_pass@1"] == 0.25
        assert m["harbor/eval_attempt_coverage"] == 1.0
        assert m["harbor/eval_pass_at_1_first_attempt_full"] == 0.25
        assert m["harbor/eval_pass_at_k_ge_pass_at_1"] == 1.0

    def test_pass_at_3_missing_task_counts_as_failure(self):
        # Four expected tasks, only three produced attempts. The missing task is
        # a full-set failure even though two of the three scored tasks resolved.
        by_task = [[1.0, 0.0, 0.0], [0.0, 0.0], [0.0, 1.0, 0.0]]
        rewards = [reward for task in by_task for reward in task]
        m = _compute_eval_count_metrics(
            rewards=rewards,
            rewards_by_task=by_task,
            num_eval_tasks=4,
            num_scored_groups=3,
            n_per_prompt=3,
        )
        # eval_num_resolved is pass@1 (rank-0): first attempts [1.0, 0.0, 0.0] -> 1
        # resolved, 3 unresolved over 4 tasks. pass@k count (2) under its own key.
        assert m["harbor/eval_num_resolved"] == 1
        assert m["harbor/eval_num_unresolved"] == 3
        assert m["harbor/eval_num_resolved_pass_at_k"] == 2
        assert m["harbor/eval_num_unscored_tasks"] == 1
        assert m["harbor/eval_num_missing_attempts"] == 4
        assert m["harbor/eval_reward_mean"] == 2 / 3
        # avg@3 alias: resolved attempts (2) / full denom (4 tasks * 3) = 2/12.
        assert m["harbor/eval_resolved_rate_full"] == 2 / 12
        assert m["harbor/eval_resolved_rate_full_pass@3"] == 2 / 4
        assert m["harbor/eval_resolved_rate_full_pass@1"] == 1 / 4
        assert m["harbor/eval_attempt_coverage"] == 8 / 12
        assert m["harbor/eval_pass_at_k_ge_pass_at_1"] == 1.0

    def test_pass_at_k_requires_task_boundaries(self):
        try:
            _compute_eval_count_metrics(
                rewards=[1.0, 0.0, 0.0], num_eval_tasks=1, num_scored_groups=1, n_per_prompt=3
            )
        except ValueError as exc:
            assert "rewards_by_task" in str(exc)
        else:
            raise AssertionError("expected task boundary validation for pass@k")

    def test_no_samples_omits_rate_keys(self):
        # Entire eval failed: no scored samples. Counts are zero, rate keys absent
        # (no division by zero), and every task shows as unscored.
        m = _compute_eval_count_metrics(
            rewards=[], num_eval_tasks=12, num_scored_groups=0, n_per_prompt=1
        )
        assert m["harbor/eval_num_samples"] == 0
        assert m["harbor/eval_num_resolved"] == 0
        # pass@1 unresolved is over the full eval set: all 12 tasks unresolved.
        assert m["harbor/eval_num_unresolved"] == 12
        assert m["harbor/eval_num_unscored_tasks"] == 12
        assert "harbor/eval_reward_mean" not in m
        assert "harbor/eval_resolved_rate_full" not in m

class TestEvalTrajectoryDedup:
    def test_retokenized_chunks_count_once(self):
        from slime_plugins.harbor.rollout import _deduplicate_eval_trajectory_outcomes

        class S:
            def __init__(self, reward, group, rank, truncated=False):
                self.reward = reward
                self.group_index = group
                self.index = rank
                self.status = "truncated" if truncated else "completed"
                self.metadata = {
                    "harbor_rollout_id": 0,
                    "harbor_rollout_group_index": group,
                    "harbor_sample_rank": rank,
                }

        # Task/group 0 produced 3 retokenized chunks from one trajectory; group 1 one chunk.
        rewards, _ = _deduplicate_eval_trajectory_outcomes(
            [[S(1.0, 0, 0), S(1.0, 0, 0), S(1.0, 0, 0)], [S(0.0, 1, 0)]]
        )
        assert rewards == [1.0, 0.0]
        m = _compute_eval_count_metrics(
            rewards=rewards, num_eval_tasks=2, num_scored_groups=2, n_per_prompt=1
        )
        assert m["harbor/eval_num_samples"] == 2
        assert m["harbor/eval_num_resolved"] == 1
        assert m["harbor/eval_resolved_rate_full"] == 0.5

def test_grouped_eval_trajectory_rewards_preserve_task_boundaries_for_pass_at_3():
    from slime_plugins.harbor.rollout import _group_eval_trajectory_rewards_by_task

    class S:
        def __init__(self, reward, group, rank):
            self.reward = reward
            self.group_index = group
            self.index = rank
            self.status = "completed"
            self.metadata = {
                "harbor_rollout_id": 0,
                "harbor_rollout_group_index": group,
                "harbor_sample_rank": rank,
            }

    # Task 0: attempt 0 emitted two retokenized chunks and failed; attempt 1
    # succeeded; attempt 2 failed. Task 1 failed all three attempts.
    output_groups = [
        [S(0.0, 0, 0), S(0.0, 0, 0), S(1.0, 0, 1), S(0.0, 0, 2)],
        [S(0.0, 1, 0), S(0.0, 1, 1), S(0.0, 1, 2)],
    ]
    by_task = _group_eval_trajectory_rewards_by_task(output_groups, n_per_prompt=3)
    assert by_task == [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    rewards = [reward for task in by_task for reward in task]
    m = _compute_eval_count_metrics(
        rewards=rewards,
        rewards_by_task=by_task,
        num_eval_tasks=2,
        num_scored_groups=2,
        n_per_prompt=3,
    )
    # eval_resolved_rate_full is the avg@3 alias: resolved attempts (1) / (2*3) = 1/6.
    assert m["harbor/eval_resolved_rate_full"] == 1 / 6
    assert m["harbor/eval_resolved_rate_full_pass@3"] == 0.5

def test_rank_aligned_attempts_do_not_promote_rank1_to_pass_at_1():
    from slime_plugins.harbor.rollout import _group_eval_trajectory_rewards_by_task

    class S:
        def __init__(self, rank, reward):
            self.reward = reward
            self.group_index = 0
            self.index = rank
            self.status = "completed"
            self.metadata = {
                "harbor_rollout_id": 0,
                "harbor_rollout_group_index": 0,
                "harbor_sample_rank": rank,
            }

    by_task = _group_eval_trajectory_rewards_by_task([[S(1, 1.0)]], n_per_prompt=3)
    assert by_task == [[None, 1.0, None]]
    m = _compute_eval_count_metrics(
        rewards=[1.0], rewards_by_task=by_task,
        num_eval_tasks=1, num_scored_groups=1, n_per_prompt=3,
    )
    # eval_resolved_rate_full is the avg@3 alias: 1 resolved attempt / (1*3) = 1/3.
    # pass@1 is still 0.0 (rank-0 missing); pass@3 resolves via the rank-1 success.
    assert m["harbor/eval_resolved_rate_full"] == 1 / 3
    assert m["harbor/eval_resolved_rate_full_pass@3"] == 1.0
    assert m["harbor/eval_resolved_rate_full_pass@1"] == 0.0
    assert m["harbor/eval_pass_at_1_first_attempt_full"] == 0.0
    assert m["harbor/eval_pass_at_k_ge_pass_at_1"] == 1.0
