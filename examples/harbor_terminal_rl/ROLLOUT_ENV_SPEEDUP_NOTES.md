# Rollout / Env Speedup Notes

Date: 2026-05-21

Run inspected:

- Job: `self_improving_terminal_agent_rl_Qwen3_5_slime_fixed_dop-fuse`
- Ray job: `raysubmit_ym3HN7NixzLiZjEK`
- Config: `harbor_terminal_bench2_full89_gy2_runtime_sif_last_turn.yaml`
- Trial dir on launcher: `/tmp/slime-harbor-terminal-bench2-full89-runtime-trials`

## Current Rollout Shape

- `rollout_batch_size=16`
- `n_samples_per_prompt=16`
- One rollout therefore requires 256 terminal trajectories.
- `harbor_num_concurrent_trials=16`, so only 16 Harbor trials run at once.
- `RolloutManager` is a single Ray actor with `num_cpus=1, num_gpus=0`; all Harbor trial containers are launched from that actor's node.

This means the 64-GPU job is not using all cluster CPU/env capacity for terminal environments. The SGLang engines are distributed, but the terminal containers are effectively driven from one RolloutManager process/node.

## Measured Timing From Completed Trials

Helper added:

```bash
python3 examples/harbor_terminal_rl/analyze_harbor_trial_timings.py \
  /tmp/slime-harbor-terminal-bench2-full89-runtime-trials
```

On the launcher node, 966 parsed `result.json` files showed:

| component | p50 | mean | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| total trial | 377.2s | 462.2s | 874.5s | 1076.6s | 1946.6s |
| env setup | 3.4s | 7.1s | 18.4s | 24.5s | 59.8s |
| agent setup | 0.2s | 0.7s | 1.4s | 2.4s | 14.8s |
| agent execution | 230.5s | 298.7s | 588.9s | 778.5s | 1511.9s |
| verifier | 32.0s | 153.8s | 507.2s | 610.1s | 1208.0s |

Conclusion: container startup is not the main bottleneck. The dominant costs are:

1. agent execution time inside terminal loop
2. verifier long tail and 1200s verifier timeouts
3. low trial concurrency and single-node env execution

## Concurrency Simulation

Using observed trial durations and simple list scheduling:

| rollout | n | c16 | c32 | c64 | c96 |
| --- | ---: | ---: | ---: | ---: | ---: |
| r1 | 256 | 8681s | 4851s | 2897s | 2258s |
| r2 | 256 | 8246s | 4670s | 2975s | 2385s |
| r3 | 256 | 6500s | 3543s | 2208s | 1807s |

This matches observed rollout wall time at c16, so increasing effective trial concurrency should help materially, assuming CPU, memory, Apptainer, and filesystem contention remain acceptable.

## Timeout Cap Simulation

Verifier cap savings over 950 verifier timings:

- cap verifier at 180s: saves ~75,677s total, affects 243/950 trials, about 79.7s/trial average.
- cap verifier at 300s: saves ~49,867s total, affects 194/950 trials, about 52.5s/trial average.

Agent cap savings over 966 agent timings:

- cap agent at 600s: saves ~19,494s total, affects 93/966 trials, about 20.2s/trial average.

Verifier caps are the lower-risk first timeout optimization because most 1200s cases are failed or no-reward verifier paths.

## Recommended Experiments

1. Raise `harbor_num_concurrent_trials` on the current single RolloutManager node.
   - Try 32 first, then 48/64 if node CPU/memory allows.
   - This is the fastest experiment because it is config-only.
   - Watch Apptainer process count, launcher CPU load, memory, and filesystem pressure.

2. Cap verifier timeout in `harbor_trial_defaults`.
   - Add `verifier.max_timeout_sec: 300` for an exploratory run.
   - If reward quality drops on tasks with legitimate slow tests, move to 600 or per-task allowlist.

3. Add per-task slow/timeout policy.
   - Slowest mean tasks observed: `mteb-retrieve`, `mcmc-sampling-stan`, `distribution-search`, `build-pmars`, `fix-ocaml-gc`.
   - These can be capped, downsampled, or moved to a separate curriculum phase.

4. Implement distributed Harbor trial workers.
   - New rollout module, not modifying original slime/Harbor core.
   - Create Ray actors on multiple nodes; each actor runs Harbor trials locally and returns converted `Sample`s.
   - Keep SGLang router shared.
   - This addresses the real structural issue: all terminal containers are currently launched by one RolloutManager process.

5. Add fine-grained rollout telemetry.
   - Add metrics for env setup, agent setup, agent execution, verifier, and conversion time directly in rollout metrics.
   - Current helper parses `result.json` after the fact; W&B should receive these online.

## Lower Priority Ideas

- Container/server reuse is not the first target. Env setup mean is only 7.1s, while agent+verifier mean is ~453s.
- Optimizing tmux command round trips may help, but only after concurrency and timeout caps.
- Reducing `max_turns` can help agent execution, but this changes task-solving behavior and should be evaluated against pass rate.

## 2026-05-22: Speed Plan For 100+ Terminus Turns

The current config keeps `max_turns: 20`. Raising it to 100+ is feasible from
the Harbor/Terminus side, but it changes the rollout bottleneck sharply:

- Each extra turn adds one SGLang `/generate` call plus terminal observation
  capture.
- The prompt history grows over the trajectory, so later turns have larger
  prefill cost unless prefix/cache behavior is effective.
- With `enable_summarize: false`, very long conversations can hit context
  limits; with `enable_summarize: true`, Harbor notes that collected rollout
  details may be incomplete after summarization.
- We currently train `harbor_sample_mode: last_turn`, so training only consumes
  one sample per trial. More turns mostly increase rollout cost, not train batch
  size.

Latest launcher-node timing over 2805 parsed trials:

| component | p50 | mean | p90 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| total trial | 488.4s | 546.7s | 989.3s | 1151.3s | 1468.4s | 2020.7s |
| env setup | 3.5s | 7.3s | 18.6s | 25.2s | 42.4s | 75.0s |
| agent execution | 231.9s | 303.4s | 617.8s | 790.4s | 1000.2s | 1511.9s |
| verifier | 65.4s | 235.3s | 604.9s | 621.0s | 1005.0s | 1232.6s |

The 100-turn plan therefore should focus on four layers, in this order.

### 1. Keep More Terminal Trials In Flight

Current structural limit: one slime `RolloutManager` Ray actor drives all Harbor
containers from one node, and `harbor_num_concurrent_trials=16`.

Measured list-scheduling simulation from current trials:

| rollout | n | c16 | c32 | c64 | c96 | c128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r9 | 256 | 12286s | 6406s | 3557s | 2511s | 2021s |
| r10 | 256 | 9762s | 5250s | 2984s | 2213s | 1855s |
| r11 | 245 | 8117s | 4244s | 2780s | 2331s | 2067s |

Actions:

- Config-only experiment: raise `harbor_num_concurrent_trials` to 32 on the
  current single RolloutManager node.
- If CPU/memory/Apptainer pressure is acceptable, try 48 or 64.
- Real fix: implement distributed Harbor trial workers as Ray actors across the
  8 nodes. This avoids pinning all terminal containers to the launcher node.

For 100 turns, distributed workers matter more than for 20 turns because slow
trials occupy slots much longer.

### 2. Add Timeout Caps Before Raising Turns

Current task `task.toml` timeouts are often 900/1800/3600s, and our config does
not cap them. Cap simulation over the latest 2805 trials:

- `verifier.max_timeout_sec=180`: saves 139.0s/trial on average, affects
  1138/2767 verifier runs.
- `verifier.max_timeout_sec=300`: saves 92.8s/trial on average, affects
  999/2767 verifier runs.
- `agent.max_timeout_sec=300`: saves 84.1s/trial on average, affects
  1044/2805 agent runs.
- `agent.max_timeout_sec=600`: saves 20.6s/trial on average, affects
  305/2805 agent runs.

Recommended first caps:

```yaml
harbor_trial_defaults:
  agent:
    max_timeout_sec: 600
  verifier:
    max_timeout_sec: 300
```

If we increase `max_turns` to 100, using only `max_turns` without an agent cap
is unsafe for throughput: failed or wandering trajectories can consume the full
task timeout and block rollout slots.

### 3. Reduce Per-Turn Wasted Work

Terminus2 currently calls:

```python
session.send_keys(..., block=False, min_timeout_sec=command.duration_sec)
```

and clamps model-proposed command duration to `min(duration, 60)`. Because this
path is non-blocking, the logged `max_timeout_sec=180` is not a hard command
timeout. For 100 turns, repeated 30-60s sleeps can dominate.

Low-risk inherited-agent knobs to add in `SlimeTerminus2` rather than editing
Harbor core:

- `max_command_duration_sec`: clamp JSON command duration to 10/20/30s instead
  of hard-coded 60s.
- `idle_observation_early_return`: if terminal pane has stopped changing, do
  not wait the full requested duration.
- `long_command_policy`: optionally hard-timeout foreground commands, but keep
  this off by default because some tasks intentionally start long-running
  servers/builds.
- `max_consecutive_parse_errors`: stop or shorten loops that repeatedly produce
  invalid JSON.
- `task_complete_double_check`: keep the current completion confirmation logic,
  but avoid extra turns after verifier-style success signals if we can detect
  them from terminal output.

The safest first implementation is only a command-duration clamp, exposed via
YAML under `agent.kwargs`, e.g. `max_command_duration_sec: 20`.

### 4. Move From Batch Rollout To Harbor-Specific Fully Async

Current `train_async.py` overlaps rollout N+1 with training N, but each Harbor
rollout call still waits for a full 16x16 batch and backfill before returning.
The fully async example in `examples/fully_async` uses a persistent background
worker, but it assumes simple SGLang prompt rollouts and cannot directly manage
stateful Harbor containers.

Needed Harbor-specific design:

- Add `slime_plugins.harbor.fully_async_rollout`, not changes to slime core.
- Start persistent Ray worker actors that continuously launch Harbor trials.
- Maintain prompt groups keyed by dataset index and sample rank.
- Return only complete GRPO groups to slime, or generate zero-loss placeholder
  failed samples to preserve group size.
- Keep producing rollout data while training/logprob/checkpoint work is running.
- Add online metrics: in-flight trials, completed queue size, group age, retry
  count, timeout count, verifier cap hits.

This will not make an individual 100-turn trajectory faster, but it reduces
training idle time and smooths long-tail trials.

### 5. Model-Serving Optimizations For Long Multi-Turn Prompts

At 100 turns, SGLang generation cost can become prompt-prefill dominated.
Promising experiments:

- Ensure the same trial/session routes consistently to the same SGLang engine so
  any prefix/cache mechanism can help. `SlimeSGLangLLM` can pass a stable
  `session_id` header if the router policy supports it.
- Do not disable radix/prefix cache unless memory pressure forces it.
- Try `--sglang-chunked-prefill-size` for long prompts; slime passes SGLang
  `ServerArgs` via `--sglang-*` arguments.
- Consider SGLang prefill/decode disaggregation through `--sglang-config` if
  long prompt prefill dominates after 100-turn experiments.
- Keep `rollout_max_context_len` and train `seq_length` separate. We can allow
  longer rollout context while still capping train samples through
  `HARBOR_MAX_TRAIN_TOTAL_TOKENS`.

### Recommended Experiment Ladder

1. Keep `max_turns=20`; add verifier/agent caps and `harbor_num_concurrent_trials=32`.
2. Raise `max_turns=50`; keep `agent.max_timeout_sec=600` and check pass rate,
   parse errors, agent duration, and verifier duration.
3. Add inherited `max_command_duration_sec=20`; compare agent duration and pass
   rate against step 2.
4. Raise `max_turns=100`; only proceed if p95 agent execution stays below the
   agent cap often enough to produce full groups.
5. Implement distributed Harbor worker actors if c32/c64 on one node becomes
   CPU/IO bound.
6. Implement Harbor fully async once distributed workers and timeout policy are
   stable.

Avoid switching to `trajectory` training mode for the 100-turn speed experiment.
`trajectory` is semantically cleaner, but it can create very long training
samples and also requires adjacent Harbor prompt histories to be strict prefix
extensions. Continue with `last_turn` until rollout throughput is under control.

## 2026-05-22：`agent_execution` 到底包含什么，以及为什么一个 slime step 很久

### `agent_execution` 的边界

Harbor 的 `agent_execution` 计时来自 `Trial._execute_agent()`：

```python
self.result.agent_execution = TimingInfo(started_at=...)
await asyncio.wait_for(
    self._agent.run(
        instruction=self._task.instruction,
        environment=self._environment,
        context=self.result.agent_result,
    ),
    timeout=self._agent_timeout_sec,
)
self.result.agent_execution.finished_at = ...
```

所以它不是单个 LLM 请求，也不是单个 shell 命令，而是一条 Terminus2
agent 轨迹的完整解题循环。它不包含：

- `environment_setup`：Apptainer/SIF/FastAPI server 启动。
- `agent_setup`：创建 tmux session。
- `verifier`：任务测试/评分。
- slime 的 logprob、训练、checkpoint。

它包含：

- 初始 terminal pane 抓取。
- 构造 initial prompt。
- 最多 `max_turns` 个 Terminus episode，目前配置是 20。
- 每个 episode 的 SGLang `/generate`。
- Terminus JSON 解析和 parse error 反馈。
- 把 parsed command 发进 tmux。
- 按模型 JSON 里的 `duration` 等待；当前 Harbor 原始代码 clamp 到最多
  60 秒。
- 抓 terminal 输出，作为下一轮 observation。
- 记录 ATIF trajectory step、token ids、logprobs。
- 每个 episode 后 dump trajectory。
- 如果模型声明 task complete，还会进入一次确认逻辑。

当前配置里 `enable_summarize: false`，所以没有 proactive summarization 成本；
如果以后为了 100+ turns 打开 summarization，还要额外计入 summarization
subagent 的 LLM 调用，并且 Harbor 注释里说明 summarization 后 rollout details
可能不完整。

### 当前实测 episode 和 LLM 请求时间

在 launcher 节点对 2816 个 `result.json` 统计：

```text
n_episodes: n=2816 min=4.00 p50=20.00 mean=18.76 p90=20.00 p95=20.00 max=20.00
llm_request_sec: n=52821 min=0.41 p50=4.01 mean=6.28 p90=13.35 p95=18.68 max=163.57
```

结论：

- 至少一半 trial 跑满了当前 `max_turns=20`。
- 平均每条 trial 跑 18.76 个 episode。
- 单次 LLM 请求平均 6.28 秒，光生成部分就约
  `18.76 * 6.28 = 117.8s/trial`。
- 剩余时间来自 tmux command 等待、真实 shell 命令执行、terminal capture、
  parse error 重试、trajectory dump，以及任务本身长命令。

### 为什么一个 slime step 可以接近 4 小时

这里的 “step” 是 slime 的 rollout/train step，不是 Terminus 的一个
episode。当前每个 slime rollout step 要收集：

```text
rollout_batch_size=16
n_samples_per_prompt=16
=> 16 * 16 = 256 条完整 Harbor trials
```

而当前 Harbor 并发是：

```text
harbor_num_concurrent_trials=16
```

也就是说一个 slime step 大约要跑 256 条 trial，分成约 16 波完成。
用当前实测平均 trial 总时长估算：

```text
mean_total_trial = 546.7s
256 trials / 16 concurrency * 546.7s ~= 8747s ~= 2.43h
```

这还没有算最慢 trial 的长尾、backfill、logprob、训练、checkpoint。
实际最后一个完整 step 的日志：

```text
rollout 11 perf/rollout_time = 8487.87s ~= 2.36h
train_wait_time = 7577.49s
log_probs_time = 190.14s
actor_train_time = 646.24s
train_time = 837.08s
step_time = 8414.57s
wait_time_ratio = 0.9005
save_model = 242.6s
```

这说明 step 慢的主要原因不是训练本身，而是训练进程在等 rollout。`wait_time_ratio`
约 90%，是最直接的证据。

更具体地说，瓶颈在：

1. `RolloutManager` 等一个完整的 Harbor batch 返回。
2. 每个 batch 要 256 条 terminal trials。
3. 每条 trial 多数跑满 20 个 Terminus episodes。
4. 每条 trial 还要跑 verifier；verifier mean 235.3 秒，p95 621 秒。
5. 当前 terminal containers 主要由单个 RolloutManager actor 驱动，
   `harbor_num_concurrent_trials=16`，没有把 8 个节点的 CPU/env 容量充分摊开。
6. 最后还要做 logprob、actor train、checkpoint；这些是分钟级，不是小时级。

因此，之前看起来 “一次 step 四小时” 的根因是 Harbor rollout batch 的长尾：
不是单个 agent episode 卡住，也不是 GPU 训练卡住，而是 256 条 stateful terminal
trials 在 16 并发下串行分波完成。

这也解释了为什么简单增加 H20 数量没有线性提速：SGLang engines 和训练 GPU 在，
但 terminal env 并发和 RolloutManager 结构没有等比例扩展。

直接提速方向：

- 把 `harbor_num_concurrent_trials` 从 16 提到 32/48/64 做压测。
- 给 `agent.max_timeout_sec` 和 `verifier.max_timeout_sec` 加 cap。
- 在 `SlimeTerminus2` 继承类里加 `max_command_duration_sec`，先把 60 秒 clamp
  降到 20/30 秒做对照。
- 做 distributed Harbor trial workers，把 terminal trials 分散到多节点。
- 后续再做 Harbor-specific fully async rollout，避免训练侧每次等完整 batch。
