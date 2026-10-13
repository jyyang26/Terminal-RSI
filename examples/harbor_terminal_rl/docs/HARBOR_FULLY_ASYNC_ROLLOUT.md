# Harbor Fully-Async Rollout

本文记录这次给 Harbor/Terminus2 RL 增加 fully-async rollout 可选模式的设计思路、代码改动和使用方法。

## 背景

这里需要区分两个概念：

- `train_async.py`：v0.3 训练 driver。它会把 rollout N+1 和 actor train N 做 pipeline overlap。我们之前 Harbor async full-style 训练已经在这条路径上跑过。
- fully-async rollout：rollout 侧有一个常驻后台 worker，持续拉 task group、持续跑 trial，并把完成的 group 放入队列。训练侧每轮只 drain 已完成的 group，不需要每轮等待当前 batch 里最慢的 trial。

v0.3 自带的 fully-async 示例使用：

```bash
--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
--custom-generate-function-path ...
```

但 Harbor rollout 不是普通 prompt generation。它还有 TrialWorker 派发、Apptainer/Singularity 环境、TerminalBench trial、ATIF/Terminus2 trajectory 转换、完整 GRPO group backfill、case-study dump、trajectory-level group unit 等逻辑。因此这次没有直接把 Harbor 塞进通用 `slime.rollout.fully_async_rollout`，而是新增 Harbor 专用 fully-async rollout function。

## 修改思路

核心原则：复用现有 Harbor rollout 的单组 trial 执行和样本转换逻辑，只替换 batch 收集方式。

普通 Harbor rollout 当前逻辑是：

1. 每轮 rollout 从 `HarborTaskDataSource` 取 `rollout_batch_size` 个 group，或者取 over-sampling window。
2. 对每个 group 调 `_run_one_group`。
3. `_run_one_group` 内部调用 `_run_one_with_semaphore` / `_run_one_trial`。
4. `_run_one_trial` 负责真实 Harbor `Trial.run()`、ATIF 转换、case-study dump。
5. rollout 主循环等待本轮 batch 收齐后返回给 actor train。

新增 fully-async 后变成：

1. 第一次进入 rollout 时启动 `HarborFullyAsyncRolloutWorker` 后台线程。
2. 后台 worker 持续从 `HarborTaskDataSource` 拉 group。
3. 每个 group 仍然调用原来的 `_run_one_group`，所以 TrialWorker、reward/mask metadata、trajectory conversion 都不复制实现。
4. 完成的 group 放入 worker queue。
5. 训练侧的 `generate_rollout_fully_async()` 每轮只 drain queue，直到拿到 `rollout_batch_size` 个完整 group。
6. 不完整 group 会 drop，worker 会继续拉新 group 来 backfill。

这种做法的目标是减少 terminal-bench 长尾 trial 对每轮训练的阻塞，同时不改变 Harbor 样本语义。

## 主要代码改动

- `slime_plugins/harbor/fully_async_rollout.py`
  - 新增 `generate_rollout_fully_async(args, rollout_id, data_source, evaluation=False)`。
  - 新增 `HarborFullyAsyncRolloutWorker` 常驻后台 worker。
  - 复用 `slime_plugins.harbor.rollout._run_one_group`。
  - 复用现有 group completeness、health metrics、logprob validation、sample hooks。

- `slime_plugins/harbor/data_source.py`
  - `HarborTaskDataSource.add_samples()` 从只读改成 FIFO requeue。
  - data source checkpoint 会保存/恢复 `requeued_groups`。
  - 这是为了兼容 fully-async/abort/requeue 语义。Harbor 专用 fully-async 目前对 incomplete group 是 drop/backfill，不把 incomplete group requeue。

- `examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh`
  - 新增 `HARBOR_ROLLOUT_MODE`。
  - 默认 `standard`，保持旧行为。
  - 设置为 `fully_async` 或 `full_async` 时切换 rollout function。

- Tests
  - `tests/test_harbor_fully_async_rollout.py`
  - `tests/test_harbor_data_source.py`
  - 已跑过 focused pytest，覆盖 data source requeue 和 fully-async incomplete group backfill。

## 如何启用

默认不启用，仍然走原来的 Harbor rollout：

```bash
bash examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh
```

启用 Harbor fully-async rollout：

```bash
HARBOR_ROLLOUT_MODE=fully_async \
bash examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh
```

等价写法：

```bash
HARBOR_ROLLOUT_MODE=full_async \
bash examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh
```

脚本会把 rollout function 从：

```bash
slime_plugins.harbor.rollout.generate_rollout
```

切到：

```bash
slime_plugins.harbor.fully_async_rollout.generate_rollout_fully_async
```

训练 driver 仍然是 `train_async.py`。也就是说，启用后是：

- train driver async overlap
- Harbor rollout producer fully-async

## 推荐 smoke

先用非常小的配置验证路径，不要直接上 full89：

```bash
HARBOR_ROLLOUT_MODE=fully_async \
NUM_ROLLOUT=2 \
ROLLOUT_BATCH_SIZE=1 \
N_SAMPLES_PER_PROMPT=2 \
GLOBAL_BATCH_SIZE=2 \
OVER_SAMPLING_BATCH_SIZE=2 \
SAVE_DEBUG_TRAIN_DATA="/path/to/workspace/slime_harbor_v030_case_study/train_data/fully_async/{rollout_id}_{rank}.pt" \
bash examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh
```

如果用现有 multi-node wrapper，保持同样的 env 传入 launcher 和 worker。`NUM_ROLLOUT=2` 是因为 `train_async.py` 的 loop 是 `range(start_rollout_id, num_rollout)`；fresh/checkpoint start 为 1 时，`NUM_ROLLOUT=2` 才会跑一个 train step。

## 可调参数

`harbor_fully_async_max_concurrent_groups`

- 放在 Harbor YAML 里即可，因为 `--custom-config-path` 会把 YAML key 挂到 `args`。
- 控制后台 worker 同时跑多少个 Harbor group。
- 不设置时默认使用 `_over_sampling_group_window(args, rollout_batch_size)`，也就是优先跟 `OVER_SAMPLING_BATCH_SIZE` 对齐。

示例：

```yaml
harbor_fully_async_max_concurrent_groups: 4
```

仍然保留的原有参数：

```yaml
harbor_num_concurrent_trials: 32
harbor_retry_missing_samples: 0
harbor_require_full_rollout_groups: true
harbor_rollout_group_attempt_multiplier: 4
```

建议先保守设置：

- `rollout_batch_size=1`
- `n_samples_per_prompt=2`
- `over_sampling_batch_size=2`
- `harbor_fully_async_max_concurrent_groups=2`

确认没有语义问题后再扩大。

## 需要看哪些日志和指标

日志里应该出现：

```text
starting Harbor fully-async rollout worker
Harbor fully-async rollout <id>: target=...
Harbor fully-async rollout <id>: done in ... queue_left=...
```

metrics 里重点看：

- `harbor/fully_async_enabled = 1`
- `harbor/fully_async_queue_left`
- `harbor/fully_async_max_concurrent_groups`
- `harbor/num_groups`
- `harbor/num_samples`
- `harbor/num_accepted_groups`
- `harbor/num_incomplete_groups_dropped`
- `harbor/num_missing_rollout_details`
- `harbor/reward_mean`
- `harbor/num_trajectory_samples`

训练侧仍然需要检查：

- rollout dump 是否落盘。
- train_data 是否落盘。
- train_data 的 tokens/loss_mask/zero spans 是否和 case-study conversion 对齐。
- advantage/returns 是否符合 GRPO group 语义。

## 当前限制

1. 不支持 eval mode。

   `generate_rollout_fully_async(..., evaluation=True)` 会直接报错。这个和 v0.3 通用 fully-async 示例一致，因为常驻 producer 和 eval 的一次性、确定性语义冲突。

2. 不支持 terminal partial resume。

   如果某个 terminal trial 被中断，当前不是恢复容器/文件系统/ATIF state，而是靠新 group backfill。Terminal agent 的 partial state 不只是 token prefix，不能按普通 LLM partial rollout 处理。

3. fully-async 的 `rollout_id` 是训练侧当前 drain 轮次的 hint。

   后台 worker 持续生产，跨 rollout 边界时可能已经有部分 group 在队列里。这个不影响训练样本本身，但 case-study/trial name 里的 rollout id 应按“异步生产”理解，不应当作为严格同步 batch 边界。

4. 还需要集群 smoke。

   本地单测覆盖了 worker 收集/backfill/data source requeue；还没有在 GPU 集群上跑 Harbor fully-async end-to-end smoke。正式 full89 前必须先跑小 smoke，并做 case study 对比 rollout sample 和 train_data。

## 已做验证

本地 focused tests：

```bash
python -m pytest tests/test_harbor_data_source.py tests/test_harbor_fully_async_rollout.py tests/test_harbor_rollout_async.py
```

结果：

```text
12 passed
```

编译和脚本语法：

```bash
python -m py_compile \
  slime_plugins/harbor/data_source.py \
  slime_plugins/harbor/fully_async_rollout.py \
  slime_plugins/harbor/rollout.py

bash -n examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh
```

均通过。

## 建议下一步

1. 跑 `HARBOR_ROLLOUT_MODE=fully_async` 的小 smoke。
2. 保存 rollout dump、case-study dump、train_data。
3. 用现有 case study report 比较：
   - raw trajectory
   - converted `Sample`
   - train_data
   - loss mask / zero spans
   - advantage / returns
4. 如果 smoke 通过，再把 `harbor_fully_async_max_concurrent_groups` 和 `OVER_SAMPLING_BATCH_SIZE` 逐步扩大到 full89 配方。
