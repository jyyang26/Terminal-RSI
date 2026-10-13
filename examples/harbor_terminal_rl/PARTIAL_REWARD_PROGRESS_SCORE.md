# Harbor CTRF Progress Reward

本文记录 2026-05-30 新增的 partial reward 逻辑。目标是给 terminal-bench-2
里“没有最终通过、但 verifier 已经显示部分测试通过”的 trajectory 一个小的连续信号，缓解
GRPO 中大量 all-zero group 的问题。

## CTRF 覆盖确认

源码层面：

- `/path/to/workspace/code/terminal-bench-2-main`
  下共有 89 个 task 目录。
- 89/89 个 `tests/test.sh` 都包含 `--ctrf /logs/verifier/ctrf.json`。

运行产物层面：

- 旧 9B turn-grouped run：
  `tb2_full89_32gpu_max100_5epoch_resp4096_ctx128k_rb16_os32_conc384_pn128_turngrouped`
- trial 目录数：2560。
- 有 `verifier/ctrf.json` 的 trial：2003。

因此结论是：CTRF 是每个 task 的统一 verifier 输出格式，但不是每个 trial 都一定能落盘。
缺失通常发生在 agent/environment 侧异常、cancel、timeout 后没有进入 verifier，或者 verifier 没完整结束。

## progress_score 定义

新代码只使用 verifier 产物，不给 timeout 本身 reward。

优先读取：

```text
<trial_dir>/verifier/ctrf.json
```

从 CTRF summary 中取：

```python
passed = summary["passed"]
effective_total = passed + failed + other + pending
```

如果 `effective_total == 0`，fallback 到：

```python
effective_total = tests - skipped
```

最终：

```python
progress_score = clamp(passed / effective_total, 0.0, 1.0)
```

如果 CTRF 缺失、空、损坏，`progress_score = 0.0`。

## shaped reward

新增 hook：

```text
slime_plugins.harbor.partial_reward.trajectory_grouped_progress_rewards
```

默认 partial cap：

```text
HARBOR_PARTIAL_REWARD_MAX=0.4
```

reward 计算：

```python
if terminal_reward >= 1.0:
    shaped_reward = terminal_reward
else:
    shaped_reward = max(terminal_reward, HARBOR_PARTIAL_REWARD_MAX * progress_score)
```

例子：

```text
passed=9, failed=2
progress_score=9/11=0.818
shaped_reward=0.4*0.818=0.327
```

成功 trajectory 仍然是 1.0，partial reward 不会超过 0.4。这样可以给接近通过的失败样本一点
排序信号，但不会让部分通过样本接近真正成功样本。

## turn_grouped 处理

训练仍然使用 turn-level sample，但 reward/advantage 在 trajectory 级别处理：

1. 按 `(harbor_rollout_id, harbor_rollout_group_index)` 找 prompt group。
2. 按 `harbor_sample_rank` 找同一 prompt group 里的 trajectory。
3. 每条 trajectory 只读取一次 CTRF 并计算 shaped reward。
4. 对同一 prompt group 的 trajectory reward 做 GRPO normalization。
5. 把同一 trajectory 的 normalized advantage 广播给它的所有 turn sample。

每个 sample metadata 会补充：

```text
harbor_terminal_reward
harbor_progress_score
harbor_progress_reward
harbor_progress_source
harbor_progress_passed
harbor_progress_total
```

同一批字段也会写入 `sample.train_metadata`，所以 `_convert_samples_to_train_data`
生成的 train data 会包含 `metadata` 字段，方便后续从训练侧 dump 里检查 progress reward。

`harbor_progress_source` 常见值：

```text
ctrf
missing_ctrf
missing_trial_uri
invalid_ctrf
empty_ctrf
```

## 修改文件

```text
slime_plugins/harbor/partial_reward.py
tests/test_harbor_partial_reward.py
examples/harbor_terminal_rl/run_full89_grpo_32gpu_max100_n16_5epoch_high_concurrency_turn_grouped.sh
examples/harbor_terminal_rl/run_qwen35_27b_full89_64gpu_max100_n16_5epoch_high_concurrency_turn_grouped.sh
```

两个 turn-grouped 启动脚本默认改成：

```bash
CUSTOM_REWARD_POST_PROCESS_PATH=slime_plugins.harbor.partial_reward.trajectory_grouped_progress_rewards
HARBOR_PARTIAL_REWARD_MAX=0.4
```

如需回滚，可以在启动前覆盖：

```bash
export CUSTOM_REWARD_POST_PROCESS_PATH=slime_plugins.harbor.turn_grouped_reward.trajectory_grouped_turn_rewards
```

## 验证

已运行：

```bash
PYTHONPYCACHEPREFIX=/tmp/slime-pycache python -m py_compile \
  slime_plugins/harbor/partial_reward.py \
  tests/test_harbor_partial_reward.py

python -m pytest \
  tests/test_harbor_atif_adapter.py \
  tests/test_harbor_rollout_async.py \
  tests/test_harbor_partial_reward.py \
  tests/test_harbor_turn_grouped_reward.py
```

结果：

```text
20 passed
```
