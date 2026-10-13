# Harbor Rollout 长尾缓解机制调研与本次改造

日期：2026-05-23

## 背景

slime 原生 `slime.rollout.sglang_rollout` 已经有一套处理 rollout 长尾的机制，包括
`over-sampling-batch-size`、`FIRST_COMPLETED`、`abort`、`partial-rollout`、
`dynamic-sampling-filter` 和训练侧 dynamic batch size。

但 Terminus2 训练当前走的是自定义入口：

```text
--rollout-function-path slime_plugins.harbor.rollout.generate_rollout
```

所以 slime 原生 SGLang rollout 里的机制不会自动覆盖到 Harbor/Terminus2 terminal
trial，需要逐项判断是否已经接入。

## 当前支持矩阵

| 机制 | slime 原生 SGLang rollout | Harbor rollout 当前状态 |
| --- | --- | --- |
| `--over-sampling-batch-size` | 已支持 | 本次已接入 Harbor group 层：用它作为 accepted + pending 的窗口上限，先完成且有效的 group 先进入 batch |
| `FIRST_COMPLETED` 先完成先消费 | 已支持 | 本次已开始接入：group 层和 group 内 sample 层都改为先完成先处理 |
| 达标后 `abort()` 慢请求 | 已支持 | 本次已做 Harbor task 层 cancellation：收够 batch 后取消 pending group/sample task，并在等待 Ray remote 时调用 `ray.cancel(..., force=True)`；但这不保证底层 Apptainer 容器一定立刻退出 |
| `--partial-rollout` / off-policy mask | 已支持 token 级续跑 | 暂不支持。terminal agent 的 partial state 包含容器、shell session、文件系统、ATIF trajectory 和 verifier 状态，不能只靠 token prefix 续跑 |
| `--dynamic-sampling-filter-path` | 已支持 | 未接入 Harbor group 收集阶段。Harbor 目前支持 `rollout_sample_filter_path` 和 `rollout_all_samples_process_path`，但它们发生在 batch 收齐之后 |
| rollout health check | 保护 SGLang engine | Harbor 有 TrialWorker 启动/健康检查和 remote timeout，但还不是 slime 原生 engine health monitor 语义 |
| `--use-dynamic-batch-size` / `--max-tokens-per-gpu` | 已支持 | 已支持。该机制在训练侧，对 Harbor 产出的 Sample/train_data 同样生效 |
| `--use-dynamic-global-batch-size` | 已支持 | 理论可用，脚本也已暴露；它可以间接容忍样本数浮动和更激进的丢组策略，但不直接加速 env 执行 |

## 为什么当前高并发仍会慢

改造前的 Harbor rollout 有两个按序等待点：

1. `_generate_rollout_async()` 会一次创建多个 `_run_one_group()` task，但随后按 `group_tasks`
   列表顺序 `await task`。
2. `_run_one_group()` 会一次创建多个 sample trial task，但随后按 `tasks` 列表顺序
   `await task`。

这意味着 trial 已经并发提交，但结果收集不是先完成先消费。如果排在前面的 group/sample
是长尾，后面已经完成的结果也不能及时进入统计、补采或返回路径。

这个问题会放大 terminal-bench 这类任务的 P99 长尾：任务耗时主要来自 Apptainer
容器、terminal command、agent 多步执行、verifier，而不是单纯模型生成。

## 本次已实现的改造

本次先实现两个低风险改动，不改变 Sample 数据契约、不引入 partial rollout 语义：

1. Harbor group 层改为 pending task set + `asyncio.wait(..., FIRST_COMPLETED)`。
   - 先提交最多 `target_groups` 个 group。
   - 任意 group 完成后立即处理结果。
   - 如果 group 不完整并被丢弃，会立即按 open slot 补采新 group。
   - 父流程结束或异常时，会取消还在 pending 的 group task。

2. Harbor group 内 sample 层改为 pending task dict + `asyncio.wait(..., FIRST_COMPLETED)`。
   - 同一个 GRPO group 内的多个 trial 不再按提交顺序等待。
   - 某个 sample trial 失败/无结果后，如果允许 retry，会立即提交该 sample 的 retry，
     不再等同一轮其他 sample 结束。
   - group task 被取消时，会取消仍在运行的 sample task。

3. Harbor group 层接入 `--over-sampling-batch-size`。
   - `rollout_batch_size` 仍然是最终需要返回给训练的 group 数。
   - `over_sampling_batch_size` 作为 `已接受 group + pending group` 的窗口上限。
   - 这里的单位是 group，不是单条 sample；例如 `rollout_batch_size=16, n=16`
     表示目标是 16 个 prompt group / 256 条 trajectory。
   - 初始提交最多 `over_sampling_batch_size` 个 group。
   - 有 group 被丢弃或消耗后，会按窗口空位继续补采，直到达到 `rollout_batch_size`
     或耗尽 `harbor_rollout_group_attempt_multiplier` 允许的尝试次数。
   - `harbor_rollout_group_attempt_multiplier` 现在按超采窗口放大 attempt budget：
     `max(target_groups * multiplier, over_sampling_groups * multiplier)`。这样首轮
     over-sampling 不会一次性吃光全部补采预算。

4. 达标后取消 over-sampled tail。
   - 收够 `rollout_batch_size` 后，剩余 pending group task 会被取消。
   - group task 被取消后，会继续取消内部 sample trial task。
   - 如果 sample trial 正在等待 Ray `TrialWorker.run_trial.remote(...)`，取消会向 Ray
     发送 `ray.cancel(ref, force=True)`。
   - 这不是 Terminus2 容器级 checkpoint/resume；如果底层环境无法响应 Ray cancel，仍可能
     需要更强的 TrialWorker/container kill 逻辑。

5. 增加 over-sampling 与长尾观测指标。
   - `harbor/num_submitted_groups`：提交过的 group 数。
   - `harbor/num_completed_group_tasks`：已完成并被取出结果的 group task 数。
   - `harbor/num_consumed_group_tasks`：达标前进入正常处理路径的 group task 数。
   - `harbor/num_accepted_groups`：最终进入训练 batch 的 group 数。
   - `harbor/num_cancelled_tail_group_tasks`：达标后被取消的 pending group task 数。
   - `harbor/num_discarded_completed_group_tasks`：同一轮 wait 中已经完成但因 batch 达标被丢弃的 group task 数。
   - `harbor/group_latency_{mean,p95,p99,max}_seconds`：已完成 group task 的耗时。
   - `harbor/cancelled_tail_age_{mean,p95,p99,max}_seconds`：达标后被取消的 pending group 已运行多久。

代码位置：

```text
slime_plugins/harbor/rollout.py
examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh
```

涉及函数：

```text
_generate_rollout_async
_run_one_group
_run_one_with_semaphore
```

启动脚本现在支持通过环境变量传入：

```text
OVER_SAMPLING_BATCH_SIZE=64
```

不设置该变量时，行为保持 slime 默认语义：`over_sampling_batch_size` 等于
`rollout_batch_size`，不会额外超采。

## 预期效果

这次改动主要解决“并发提交但按序收结果”的问题。直接收益包括：

- 快完成的 group/sample 可以先进入统计和 batch 收集路径。
- 不完整 group 可以更早触发补采，而不是等当前批次所有 group 都结束。
- sample retry 可以更早启动，减少 group 内部长尾。
- `over_sampling_batch_size > rollout_batch_size` 时，慢 group 不再阻塞已经完成的快 group。
- 收够训练 batch 后会取消额外 pending group，避免 over-sampling 无限消耗 rollout 资源。
- over-sampling 越大，越容易绕开 P99 长尾，但也越可能产生已完成但被丢弃的有效 group。
  需要用 `num_discarded_completed_group_tasks` 和 cancelled tail age 指标平衡速度与算力浪费。

需要注意：这次的 cancel 是 asyncio/Ray task 层面的取消，不是完整的 Terminus2
partial rollout。它能让 Harbor rollout manager 不继续等待多余 trial，并尽量通知 Ray
取消 remote task；是否能立即释放 Apptainer 容器，还取决于 TrialWorker 和环境实现。

## 仍待实现

下一阶段建议按优先级继续做：

1. 增加更强的 TrialWorker/container kill 机制，用来处理 Ray cancel 后底层 Apptainer
   进程仍不退出的情况。这是 over-sampling 长尾治理的 P0 风险：Ray task 取消不等价于
   容器和子进程一定释放。
2. 接入 `--dynamic-sampling-filter-path`，在 group append 前调用 dynamic filter，零方差
   或无效 group 直接丢弃并补采。
3. 评估 Harbor-specific fully async rollout producer，把 rollout 和训练 buffer 更彻底解耦。
4. 如果要做真正 partial rollout，需要 Harbor/Terminus2 增加 container/session/ATIF
   checkpoint 与 resume 能力。

## 验证记录

已完成：

```text
PYTHONPYCACHEPREFIX=/tmp/slime_pycache python -m py_compile slime_plugins/harbor/rollout.py
PYTHONPYCACHEPREFIX=/tmp/slime_pycache python -m py_compile slime_plugins/harbor/rollout.py tests/test_harbor_rollout_async.py
python -m pytest tests/test_harbor_rollout_async.py -q
```

说明：直接 `python -m py_compile` 会尝试写入 `slime_plugins/harbor/__pycache__`，当前目录
权限不允许写新的 Python 3.13 pyc 临时文件，因此改用 `/tmp` 作为 pycache 前缀完成语法检查。

新增单测覆盖：

- group 内 sample trial 先完成先处理，失败/空结果 sample 可以立即 retry。
- group 层不完整 group 先完成后可以立即触发补采，不再等待更早提交但更慢的 group。
- `over_sampling_batch_size` 会扩大首轮 group 提交窗口，且 attempt budget 会按超采窗口乘
  `harbor_rollout_group_attempt_multiplier` 放大。
- 收够 batch 后取消慢尾 group，并记录取消/丢弃/timing 指标。
- 取消正在等待的 Ray remote trial 时，会调用 `ray.cancel(ref, force=True)`。

## 2026-05-23 64GPU 27B OOM 与重启参数

64GPU 27B 高并发 run `raysubmit_Yx6tHC8Me24x78cm` 在 actor 训练阶段失败，不是
Terminus env 或 rollout 阶段失败。

失败点：

```text
ray::MegatronTrainRayActor.train()
slime_plugins/harbor/train_loss.py -> get_log_probs_and_entropy
slime/utils/ppo_utils.py -> compute_log_probs(logits_chunk.clone(), ...)
Megatron fused_vocab_parallel_cross_entropy
torch.OutOfMemoryError: Tried to allocate 244.00 MiB
GPU 6 free: 231.06 MiB
```

当时启动参数：

```text
rollout_max_response_len: 2048
rollout_max_context_len: 131072
seq_length: 32768
harbor_max_train_total_tokens: 32768
log_probs_chunk_size: 256
actor_num_nodes: 2
pipeline_model_parallel_size: 8
rollout_batch_size: 24
n_samples_per_prompt: 16
global_batch_size: 384
```

从日志看，rollout 已经能产出训练 batch，async 侧还在继续执行下一轮 `r2` 的 trial。
真正打挂 Ray job 的是第一轮 actor train 中单个 microbatch 的 logprob/cross-entropy
显存峰值。`log_probs_chunk_size=256` 对 27B + 248320 vocab 会产生约 256 * vocab
级别的 fp32 临时 buffer，这次只差十几 MiB 就能通过，但整体显存已经贴满，继续使用
32768 train cap 风险很高。

本次重启修改：

```text
examples/harbor_terminal_rl/run_qwen35_27b_full89_64gpu_max100_n16_5epoch_high_concurrency.sh
  ROLLOUT_MAX_RESPONSE_LEN=4096
  OVER_SAMPLING_BATCH_SIZE=48
  SEQ_LENGTH=32768
  HARBOR_MAX_TRAIN_TOTAL_TOKENS=32768
  LOG_PROBS_CHUNK_SIZE=32

examples/harbor_terminal_rl/harbor_terminal_bench2_full89_64gpu_27b_max100_high_concurrency_last_turn.yaml
  trials_dir 后缀改为 traincap32k_lpc32

examples/harbor_terminal_rl/monitor_full89_64gpu_27b_max100_high_concurrency.sh
  BASE/SAVE/TRIALS 跟随新后缀
```

目录后缀：

```text
resp4096_ctx128k_actorpp8_thd_rb24_os48_conc384_pn64_traincap32k_lpc32
```

这次把 64GPU 的 `rollout_max_response_len` 从 2048 提到 4096，用来减少单步回复被截断。
这会增加 rollout 单步生成耗时，也可能提高训练 response suffix 长度；显存侧用
`LOG_PROBS_CHUNK_SIZE=32` 控制 cross-entropy 峰值。

保留 rollout 侧的 `rollout_max_context_len=131072`，因为它影响 SGLang/agent 看到的长上下文；
训练侧通过 `HARBOR_MAX_TRAIN_TOTAL_TOKENS=32768` 只裁掉 prompt prefix，保留 response suffix。

## 2026-05-28 64GPU 失败复盘后的稳定性改动

本轮 64GPU 27B 失败的直接错误是 `ActorUnavailableError`，随后 driver 60 秒内无法连接
Ray GCS。最新完整 rollout dump 停在 `23.pt`，checkpoint 停在 `iter_0000019`，说明
`r24` 中途 Ray control plane 或 RolloutManager/TrialWorker 通路不可用。

这次复盘后新增/调整：

- 开启 `USE_ROLLOUT_LOGPROBS=1`，即 CLI 增加 `--use-rollout-logprobs`。Harbor
  `SlimeSGLangLLM` 已经用 `return_logprob=True` 请求 SGLang，ATIF adapter 会把
  `logprobs` 写入 `Sample.rollout_log_probs`，Slime train_data 会继续携带
  `rollout_log_probs`。现在 rollout 侧还会在开启该选项时校验每条 sample 都有完整
  rollout logprob，否则直接报出 bad sample 数。
- 降低 64GPU high-concurrency 参数：`harbor_num_concurrent_trials 256 -> 192`，
  `harbor_trials_per_node 64 -> 48`。新目录后缀使用 `conc192_pn48_..._logp`，
  避免混入旧 run 产物。
- 增加 W&B rollout 指标：
  - `harbor/trial_timeout_ratio`：Harbor status 为 `TRUNCATED` 的样本比例。当前
    ATIF 路径中这主要来自 Terminus trial timeout。
  - `harbor/step_length_finish_turn_ratio`：ATIF `extra.finish_reason.type == length`
    的 turn 比例，用来区分单个模型 step hit `max_new_tokens`。
  - `harbor/train_context_trimmed_sample_ratio`：训练侧 token cap 裁剪过 prompt prefix
    的样本比例。
  - `harbor/rollout_logprob_sample_ratio` / `harbor/rollout_logprob_token_ratio`：
    验证 `--use-rollout-logprobs` 所需字段是否完整。
- 增加 Ray/TrialWorker 观测：
  - `harbor/trial_worker_count`
  - `harbor/trial_worker_healthy_count`
  - `harbor/trial_worker_healthcheck_failed`
  - `harbor/trial_worker_healthcheck_killed`
  - monitor 脚本额外记录 Ray/GCS/raylet pid、RSS、fd count、mount count、socket summary、
    Apptainer/Singularity 进程数。

额外发现：当前 TrialWorker healthcheck 和 trial 执行用的是同一个 Ray actor 并发池。
当 `harbor_trials_per_node=64` 且 actor slots 被 trial 或 cancel cleanup 占满时，
`get_node_ip()` healthcheck 也需要排队，5 秒 timeout 后旧逻辑会直接 `ray.kill(worker)`。
64GPU 失败日志里存在大量 `TrialWorker ... killed by ray.kill` 记录，这和“忙碌 worker
被误判为 unhealthy”路径吻合。新逻辑把 `harbor_trial_worker_healthcheck_timeout_sec`
提高到 30，并设置 `harbor_kill_unhealthy_trial_workers: false`，healthcheck 失败时先保留
worker，由真正的 `run_trial.remote(...)` 错误路径处理死 actor。

另一个稳定性开关是 `harbor_fallback_to_inprocess_trials: false`。当多节点 TrialWorker 全部
不可用时，不再把高并发 Apptainer trials fallback 到 RolloutManager/head 节点，避免把本来要
分散到 rollout 节点的容器压力集中打到 Ray head。

## 2026-05-28 9B full89：trajectory 训练、128k context 对齐与 CP4

9B full89 GRPO 的第二轮质量下降复盘后，主要修正三类问题：

1. 不再使用 `harbor_sample_mode: last_turn`。
2. rollout / Harbor agent / SGLang / actor train 的 context window 先统一到 128k。
3. 64GPU 场景下默认使用 32 张卡做 actor train，并用 context parallel 降低 128k 训练压力。

### sample mode

新增配置：

```text
examples/harbor_terminal_rl/harbor_terminal_bench2_full89_gy2_runtime_sif_trajectory.yaml
  harbor_sample_mode: trajectory
```

这里没有直接改成 `turn`。原因是 `turn` 会把一个 terminal trajectory 拆成多个 Slime
sample，当前 GRPO reward normalization 会看到不等长/额外展开后的样本序列，容易破坏
`rollout_batch_size * n_samples_per_prompt` 的固定 group 语义。

`trajectory` 的语义更适合这次目标：

- 一个 Harbor/Terminus trial 仍然对应一个 Slime sample。
- 这个 sample 内包含整条多轮 token trajectory。
- 所有 assistant completion token 的 `loss_mask=1`。
- 环境 observation / terminal output delta 的 `loss_mask=0`。
- 一个 task 的 `n_samples_per_prompt=16` 仍然是 16 条 trajectory 做 GRPO group norm。

所以它能训练每个 assistant turn，同时保持 GRPO 分组不变。

### context window 对齐

`run_full89_grpo_2epoch_n16.sh` 默认改为：

```text
ROLLOUT_MAX_CONTEXT_LEN=131072
SGLANG_CONTEXT_LENGTH=131072
SGLANG_MAX_PREFILL_TOKENS=131072
SGLANG_CHUNKED_PREFILL_SIZE=8192
SEQ_LENGTH=131072
HARBOR_MAX_TRAIN_TOTAL_TOKENS=131072
```

另外 `SlimeSGLangLLM` 新增 `max_context_tokens` 参数，`rollout.py` 会把
`--rollout-max-context-len` 透传到 Harbor LLM：

```text
slime_plugins/harbor/llm.py
slime_plugins/harbor/rollout.py
```

这样 Harbor agent 侧 `get_model_context_limit()` 不再回退到 tokenizer
`model_max_length` 或默认 131072，而是显式跟本次 rollout 参数一致。SGLang server 侧也通过
`--sglang-context-length 131072` 和 `--sglang-max-prefill-tokens 131072` 对齐。

### 64GPU / 32 actor / CP4 默认布局

`run_full89_grpo_2epoch_n16.sh` 默认改成 64GPU preset：

```text
ACTOR_NUM_NODES=4
ACTOR_NUM_GPUS_PER_NODE=8
ROLLOUT_NUM_GPUS=32
TENSOR_MODEL_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=8
CP_SIZE=4
```

对应 actor train 并行度：

```text
TP * PP * CP = 1 * 8 * 4 = 32
```

这正好用满 32 张 actor GPU。通用启动脚本增加了启动前校验：

```text
ACTOR_TOTAL_GPUS = ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE
ACTOR_PARALLEL_SIZE = TP * PP * CP
```

要求 `ACTOR_TOTAL_GPUS >= ACTOR_PARALLEL_SIZE`，且 `ACTOR_TOTAL_GPUS` 必须能被
`ACTOR_PARALLEL_SIZE` 整除，避免 `CP_SIZE` 和 actor GPU 数不匹配时提交后才在
Megatron/Ray 阶段失败。

### async old-policy 与 loss weighting

本次 full89 入口还默认开启：

```text
KEEP_OLD_ACTOR=1
CALCULATE_PER_TOKEN_LOSS=0
```

`KEEP_OLD_ACTOR=1` 会给通用脚本追加 `--keep-old-actor`，用于修正 `train_async.py`
一边训练当前 rollout、一边提前生成下一轮 rollout 时的 old-policy 对齐问题。

`CALCULATE_PER_TOKEN_LOSS=0` 会让这条 Harbor GRPO 实验不再默认追加
`--calculate-per-token-loss`。之前 scalar GRPO advantage 被复制到每个 response token
后，会变成按 response length 加权；terminal 任务里长失败回复会被放大，这和样本级 GRPO
目标不一致。

### 修改文件

```text
examples/harbor_terminal_rl/harbor_terminal_bench2_full89_gy2_runtime_sif_trajectory.yaml
examples/harbor_terminal_rl/run_full89_grpo_2epoch_n16.sh
examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh
slime_plugins/harbor/llm.py
slime_plugins/harbor/rollout.py
tests/test_harbor_llm.py
```

### 验证记录

已完成：

```text
bash -n examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh \
  examples/harbor_terminal_rl/run_full89_grpo_2epoch_n16.sh

PYTHONPYCACHEPREFIX=/tmp/slime_pycache python -m py_compile \
  slime_plugins/harbor/llm.py \
  slime_plugins/harbor/rollout.py \
  tests/test_harbor_llm.py

PYTHONPYCACHEPREFIX=/tmp/slime_pycache PYTEST_ADDOPTS='-p no:cacheprovider' \
  python -m pytest \
  tests/test_harbor_llm.py \
  tests/test_harbor_rollout_async.py \
  tests/test_harbor_atif_adapter.py
```

结果：`18 passed`。

## 2026-05-28 27B 64GPU 重启配置：trajectory + rollout logprobs + 128k

这次 27B 64GPU run 按同一条 trajectory-level 路线重启，但和上面的 9B 配方有一个关键差异：
27B 之前在 TP=1/PP=8 下已经在 actor logprob/cross-entropy 路径 OOM，所以这里保留
`TP=2` 来分摊 vocab-parallel logprob tensor 和模型状态。32 张 actor GPU 下如果继续用
`PP=8`，可行的 context parallel 是 `CP=2`：

```text
ACTOR_NUM_NODES=4
ACTOR_NUM_GPUS_PER_NODE=8
ROLLOUT_NUM_GPUS=32
TENSOR_MODEL_PARALLEL_SIZE=2
PIPELINE_MODEL_PARALLEL_SIZE=8
CP_SIZE=2
SEQUENCE_PARALLEL=1
```

对应 actor 并行度：

```text
TP * PP * CP = 2 * 8 * 2 = 32
```

本次启动入口：

```text
examples/harbor_terminal_rl/run_qwen35_27b_full89_64gpu_max100_n16_5epoch_high_concurrency.sh
```

主要配置：

```text
CONFIG=examples/harbor_terminal_rl/harbor_terminal_bench2_full89_64gpu_27b_max100_high_concurrency_trajectory.yaml
harbor_sample_mode: trajectory
USE_ROLLOUT_LOGPROBS=1
KEEP_OLD_ACTOR=1
CALCULATE_PER_TOKEN_LOSS=0
ROLLOUT_MAX_RESPONSE_LEN=4096
ROLLOUT_MAX_CONTEXT_LEN=131072
SGLANG_CONTEXT_LENGTH=131072
SGLANG_MAX_PREFILL_TOKENS=131072
SGLANG_CHUNKED_PREFILL_SIZE=8192
SEQ_LENGTH=131072
HARBOR_MAX_TRAIN_TOTAL_TOKENS=131072
LOG_PROBS_CHUNK_SIZE=32
RAY_CLUSTER_WAIT_SECONDS=1800
WANDB_GROUP=q35-27b-tb2-64g-traj-logp-ctx128k-r4096-a4-tp2pp8cp2-rb16os32-c192p48
```

`RAY_CLUSTER_WAIT_SECONDS` 是这次新增的启动保护。worker 分支会先把 HF 和 torch_dist
checkpoint stage 到本机 `/tmp/slime-harbor-model-cache`，再执行 `ray start`。如果部分节点
torch_dist cache marker 缺失，4 分钟默认等待可能只看到少数节点就继续提交 Ray job；
27B trajectory wrapper 现在默认等待 30 分钟，避免 3/8 节点时提前提交。

通用启动脚本还新增了 `USE_EXISTING_RAY_HEAD=1`。用途是处理“Ray head 已起来，但 worker
还在 stage checkpoint”的情况：可以先停掉当前 launcher shell 防止提前提交，保留 Ray head
和正在 stage 的 worker；等 worker 都 join 后，用
`USE_EXISTING_RAY_HEAD=1 CLEAN_START=0` 重新跑 launcher 分支，只执行等待/提交，不重复
`ray start --head`。

27B high wrapper 也显式继承 trajectory 配方里的 old-policy / loss-weighting 设置：
`KEEP_OLD_ACTOR=1` 会追加 `--keep-old-actor`，避免 async 下一轮 rollout 和当前 actor
更新互相穿插时 old-policy 对齐进一步漂移；`CALCULATE_PER_TOKEN_LOSS=0` 会移除
`--calculate-per-token-loss`，避免整条 trajectory 里更长的 response 自动获得更大 GRPO
权重。

为了让 trajectory sample 在训练侧严格满足 slime 的 sample contract，这次不再使用只裁
prompt prefix 的 cap 函数，而是改成允许裁掉最旧 response/observation suffix 的 hard cap：

```text
ROLLOUT_ALL_SAMPLES_PROCESS_PATH=slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples
ROLLOUT_DATA_POSTPROCESS_PATH=slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data
```

这两个函数会同步更新 `tokens`、`response_length`、`loss_mask` 和 `rollout_log_probs`。
如果 128k 仍然不够，会在 metadata/metric 里体现：

```text
harbor/train_response_trimmed_sample_ratio
harbor/train_response_trimmed_tokens
harbor/train_context_trimmed_sample_ratio
harbor/train_context_max_original_total_length
harbor/train_context_max_trimmed_total_length
```

解读方式：

- `train_response_trimmed_sample_ratio > 0`：说明真实 trajectory 超过 128k，训练已经开始丢最早的 assistant/observation suffix。
- `train_context_trimmed_sample_ratio > 0` 且 response trim 为 0：说明只裁掉了 prompt prefix，response suffix 仍完整。
- `rollout/truncated` 或 `harbor/step_length_finish_*` 继续升高：说明单 turn 4096 response 仍被截断，和训练 cap 不是同一个问题。

并发也从之前更激进的 256/64 降到：

```text
harbor_num_concurrent_trials: 192
harbor_trials_per_node: 48
OVER_SAMPLING_BATCH_SIZE=32
```

这次继续保留 TrialWorker 稳定性开关：

```text
harbor_trial_worker_healthcheck_timeout_sec: 30
harbor_kill_unhealthy_trial_workers: false
harbor_fallback_to_inprocess_trials: false
```

对应输出目录带 `logp_traj` 和 `traincap128k` 后缀，和之前 last-turn / 32k cap / 64k cap
实验隔离。

注意：W&B `group` 有 128 字符限制。第一次重启时原 group 名把所有参数都展开进名字里，
W&B 报 `128 limit exceeded for GroupName` 并在训练前退出；脚本已改成短 group 名，完整参数
仍保留在输出目录和脚本文档里。

### 2026-05-28 13:32 64GPU 27B trajectory 重启状态

实际重启中踩到两个启动前错误：

1. 第一轮提交 `raysubmit_2J9MFaS85iVJ5Ya6` 死在 trajectory YAML 缩进错误：
   `harbor_trial_defaults.trials_dir` 必须缩进在 `harbor_trial_defaults` 下。修复后本地和远端
   PyYAML 均能解析。
2. 第二轮提交 `raysubmit_UxALeRAi6iqKaWuQ` 死在 W&B group 超过 128 字符。脚本已改为短
   group：
   `q35-27b-tb2-64g-traj-logp-ctx128k-r4096-a4-tp2pp8cp2-rb16os32-c192p48`。

当前有效 Ray job：

```text
raysubmit_jthxyh2CHshhWchr
```

已确认的健康信号：

- Ray cluster active 8 nodes / 64 GPUs，placement group 成功创建 64 个 GPU bundle。
- W&B run 创建成功，run id: `v6gf2j64`。
- CLI 参数确认：
  - `--use-rollout-logprobs`
  - `--keep-old-actor`
  - 无 `--calculate-per-token-loss`
  - `calculate_per_token_loss=False`
  - `keep_old_actor=True`
  - `use_rollout_logprobs=True`
- 16 个 SGLang TP=2 engine 已完成 weight load / cuda graph capture，并出现
  `The server is fired up and ready to roll!`。
- Rollout side 已 spawn 4 个 TrialWorker，每个 `max_concurrency=48`，总并发上限 192。
- Actor -> rollout engine 初始 `update_weights` 已完成，耗时约 14.3s。
- 13:32 左右已经进入第一批 env rollout，日志出现：
  `running r1 g0 s0..s3`。

13:32 时还没有 `trajectory.json` 或 `rollout_dumps/*.pt`，这是正常的早期状态；TrialWorker
刚开始执行 terminal-bench task，下一步观察点是 trajectory 数量增长、首个 rollout dump
落盘，以及第一步 train 是否越过 128k trajectory cap / rollout logprob 校验。

13:37 的人工状态查询曾被全目录 `find ${TRIALS_DIR} -name trajectory.json` 卡住。为避免监控
tmux 被共享盘慢扫描拖死，`monitor_full89_64gpu_27b_max100.sh` 增加
`MONITOR_FIND_TIMEOUT_SECONDS`，默认 60s；当前 27B monitor tmux 用
`MONITOR_FIND_TIMEOUT_SECONDS=20` 重启。

### 2026-05-28 13:42 trajectory conversion 新卡点

当前 27B job 仍是 RUNNING，但日志显示大量 sample 在 ATIF -> slime `trajectory` sample 转换阶段被丢弃：

```text
RolloutConversionError: Cannot build whole-trajectory slime sample because the next Harbor prompt is not a prefix extension at detail=0 turn=0. Use mode='turn' or 'last_turn'.
Dropping failed Harbor rollout sample ...
```

代表性失败调用栈：

```text
slime_plugins/harbor/rollout.py:_run_one_group
slime_plugins/harbor/rollout.py:_run_one_with_semaphore
slime/ray/rollout.py:TrialWorker.run_trial
slime_plugins/harbor/rollout.py:_run_one_trial
slime_plugins/harbor/atif_adapter.py:samples_from_trial_result
slime_plugins/harbor/atif_adapter.py:samples_from_rollout_details
slime_plugins/harbor/atif_adapter.py:_trajectory_sample_from_detail
```

这不是 Ray / SGLang / W&B / Apptainer 启动问题。到 13:42 为止这些都已经过关：

- job status: RUNNING
- 8 nodes / 64 GPUs placement OK
- W&B run OK
- SGLang 128k engine ready
- TrialWorker 正在跑 task，已经从 `g0` 推进到 `g12`

真正问题是 adapter 里的 trajectory 拼接假设太强。当前实现要求：

```text
next_prompt_token_ids == current_tokens + observation_delta
current_tokens = initial_prompt + completion_0 + observation_delta_0 + ...
```

但 Harbor/Terminus2 的 `Chat.rollout_details` 只记录每一次 LLM call 的：

```text
prompt_token_ids
completion_token_ids
logprobs
```

下一轮 `prompt_token_ids` 是 chat history 通过 chat template 重新序列化后的完整 prompt。
它可能包含上一轮 assistant message、tool/observation/user message wrapper、special tokens、role
tokens 等；这些 token 不保证和 provider 返回的上一轮 `completion_token_ids` 逐 token 拼接后
形成严格 prefix。因此 `last_turn` 能工作，`turn` 能工作，但目前的 strict `trajectory` 会在
多轮样本上失败。

这说明原来文档里“trajectory 模式要求相邻 prompt history 是严格 prefix extension”的假设在真实
Terminus2/SlimeSGLang 路径上不成立。可行修法有三类：

1. **保守可跑**：临时退回 `harbor_sample_mode: last_turn`，保留 `--use-rollout-logprobs`、
   128k、`--keep-old-actor` 和 per-token-loss 关闭，先恢复训练闭环。
2. **近似 trajectory**：在 `_trajectory_sample_from_detail` 中去掉 strict prefix，改用
   “turn prompt + completion” 拼接，每个 turn 的 prompt segment mask=0、completion
   mask=1。这能训练所有 assistant turns，但会重复 prompt history，序列明显变长，而且后续 turn
   的训练上下文不完全等价于真实生成上下文。
3. **正确 trajectory**：基于 Harbor ATIF `trajectory.json` 或 chat message history 重新按同一个
   tokenizer/chat template 构造 token 序列，同时记录每个 assistant message span，给 assistant
   span 置 `loss_mask=1`、observation/tool/user span 置 0。这是语义最干净的方案，但需要比当前
   rollout detail 更多的结构化信息，或在 Harbor agent 侧新增轻量字段。

当前有效 job `raysubmit_jthxyh2CHshhWchr` 因为 conversion errors 会持续丢样本，即使保持
RUNNING，得到的 batch 也不可信。下一步建议先停掉这轮 27B trajectory run，改 adapter 或退回
last-turn 后再重启。

### 2026-05-28 13:49 处理决定：停 trajectory，先用 last-turn 恢复训练闭环

已停止有问题的 trajectory job：

```text
raysubmit_jthxyh2CHshhWchr
```

停止原因：该 job 虽然 Ray/SGLang/W&B/TrialWorker 都已经跑起来，但 ATIF strict trajectory
conversion 会持续抛 `RolloutConversionError` 并丢样本，继续跑得到的 batch 不可信。

当前采用保守 fallback，而不是立即在 64 卡上验证新的近似 trajectory adapter：

- `harbor_sample_mode: last_turn`
- 继续保留 `--use-rollout-logprobs`
- 继续保留 `--keep-old-actor`
- 继续关闭 `--calculate-per-token-loss`
- 继续使用 128k rollout/SGLang/train context
- 继续使用 TP2/PP8/CP2，actor 32 GPUs，rollout 32 GPUs
- 继续使用 192 TrialWorker 并发、over-sampling 32

改动文件：

```text
examples/harbor_terminal_rl/harbor_terminal_bench2_full89_64gpu_27b_max100_high_concurrency_last_turn.yaml
examples/harbor_terminal_rl/run_qwen35_27b_full89_64gpu_max100_n16_5epoch_high_concurrency.sh
examples/harbor_terminal_rl/monitor_full89_64gpu_27b_max100_high_concurrency.sh
```

新的隔离目录后缀是：

```text
resp4096_ctx128k_actor4_tp2pp8cp2_thd_rb16_os32_conc192_pn48_traincap128k_lpc32_logp_lastturn
```

新的 W&B group：

```text
q35-27b-tb2-64g-lastturn-logp-ctx128k-r4096-a4-tp2pp8cp2-rb16os32-c192p48
```

本地验证：

```text
bash -n \
  examples/harbor_terminal_rl/run_qwen35_27b_full89_64gpu_max100_n16_5epoch_high_concurrency.sh \
  examples/harbor_terminal_rl/monitor_full89_64gpu_27b_max100_high_concurrency.sh \
  examples/harbor_terminal_rl/monitor_full89_64gpu_27b_max100.sh

python -c "import yaml, pathlib; p=pathlib.Path('examples/harbor_terminal_rl/harbor_terminal_bench2_full89_64gpu_27b_max100_high_concurrency_last_turn.yaml'); yaml.safe_load(p.read_text()); print('yaml ok')"
```

下一步：清理旧 Ray/tmux/Apptainer 后，启动 last-turn 版本；如果该版本能完成首个 rollout dump
和 train step，再单独实现真正 trajectory-level adapter。

### 2026-05-28 13:57 last-turn 版本已重启

已在 64 H20 job 上完成旧进程清理，并重新启动保守可跑版本：

```text
Taiji instance: 8B1D81989E25E915019E3C1BB0361459
Ray job: raysubmit_szHFbi9r9SCwP23q
W&B run: wqn4l6t5
W&B group: q35-27b-tb2-64g-lastturn-logp-ctx128k-r4096-a4-tp2pp8cp2-rb16os32-c192p48
launcher tmux: slime_full89_64gpu_27b_lastturn
worker tmux: slime_full89_64gpu_27b_lastturn_worker
monitor tmux: slime_full89_64gpu_27b_lastturn_monitor
```

启动状态：

- Ray cluster 已经达到 8/8 nodes、64/64 GPUs。
- worker 节点进入 `WORKER_KEEP_ALIVE=1`。
- SGLang rollout engine 正在加载 16 个 TP2 engine。
- 新 job 已确认是 last-turn 配置，不再使用 strict trajectory conversion。

关键参数已经在 Ray entrypoint 中确认：

```text
--custom-config-path .../harbor_terminal_bench2_full89_64gpu_27b_max100_high_concurrency_last_turn.yaml
--rollout-max-response-len 4096
--rollout-max-context-len 131072
--sglang-context-length 131072
--sglang-max-prefill-tokens 131072
--tensor-model-parallel-size 2
--pipeline-model-parallel-size 8
--context-parallel-size 2
--qkv-format thd
--use-rollout-logprobs
--keep-old-actor
```

同时确认没有启用 `--calculate-per-token-loss`。这点很重要：当前 `last_turn` sample 只训练最后一轮
assistant response，先避免上一次 trajectory conversion 造成的全量丢样本问题；真正的
trajectory-level 训练需要后续重写 adapter。

运维注意：

- 不要在共享日志里直接贴完整 `ray job list` 输出，因为 Ray 会展开 `runtime_env.env_vars`。
  后续状态检查优先用 `ray job status <submission_id>`、launcher log、monitor log 和 W&B。
- monitor 脚本已经给 shared disk 上的 `find` 加了 timeout，避免之前 `TRIALS_DIR` 统计卡住。

### 2026-05-28 14:06 last-turn run 健康度检查

当前 job 仍在运行：

```text
Ray job: raysubmit_szHFbi9r9SCwP23q
Status: RUNNING
```

启动关键阶段已经通过：

- actor checkpoint 已经加载完成；
- 32 张 actor GPU 的 `update_weights` 已经成功推到 16 个 TP2 SGLang rollout engine；
- `Timer update_weights end (elapsed: 13.4s)`；
- TrialWorker 已经开始执行 `r1` 的 task group；
- 日志里没有再出现 trajectory 版本的 `RolloutConversionError`；
- 目前也没有看到 Traceback、OOM、`Dropping failed Harbor rollout sample` 这类错误。

rollout 节点抽样资源情况：

```text
node4 GPU util: 91%-100%, memory ~77.8GB/GPU, apptainer/singularity process count ~97
node5 GPU util: 87%-90%, memory ~77.8GB/GPU, apptainer/singularity process count ~97
```

这说明高并发 rollout 已经压上去，当前不是启动阶段空转。到 14:06 暂时还没有首个
`rollout_dumps/{rollout_id}.pt`，需要继续观察第一轮 rollout 是否能顺利凑满 batch 并进入首个
train step。

### 2026-05-28 14:12 first rollout 仍在收集中

最新检查：

```text
Ray job: raysubmit_szHFbi9r9SCwP23q
Status: RUNNING
rollout_dumps: 0
RolloutConversionError: 0
Traceback: 0
OOM/out of memory: 0
```

从 TrialWorker 日志看，`r1` 正在持续推进：

```text
14:00:39 开始 r1 的 group/sample 执行
14:03 左右已经推进到 g11
14:07-14:11 继续推进 g12/g13 的多个 sample
```

当前还没有首个 dump 的原因不是启动失败，而是第一轮 rollout 还没凑满可交给训练侧的 batch。
这和 `max_turns=100`、`rollout_max_response_len=4096`、terminal-bench 任务执行较长有关。
目前日志没有显示 conversion error 或 OOM，GPU 利用率也处于高位，因此状态判断是“健康运行，
等待首批 rollout 完成”，不是“卡在初始化”。

### 2026-05-29 02:45 64 卡 last-turn run 停止原因

用户反馈 W&B/任务看起来又停了。远程检查结论：

```text
Taiji instance: 8B1D81989E25E915019E3C1BB0361459
Ray job: raysubmit_szHFbi9r9SCwP23q
Final status: FAILED
Exit code: 1
rollout_dumps: 1.pt, 2.pt
checkpoint_files: 0
```

这次不是 trajectory conversion 失败，也不是启动失败。它已经完成了第一轮 rollout 和第一个
train step：

```text
Timer actor_train end (elapsed: 5487.8s)
Timer train end (elapsed: 5488.0s)
step 1: {
  'train/loss': 0.010738333687186241,
  'train/pg_loss': 0.010738333687186241,
  'train/pg_clipfrac': 0.05203016474843025,
  'train/ppo_kl': 0.06377356499433517,
  'train/train_rollout_logprob_abs_diff': 0.0,
  'train/grad_norm': 0.7311346924990939,
  'train/step': 1
}
perf 1: {
  'perf/update_weights_time': 13.31,
  'perf/train_wait_time': 4520.61,
  'perf/actor_train_time': 5487.86,
  'perf/train_time': 5488.15,
  'perf/actor_train_tok_per_s': 1221.53,
  'perf/step_time': 10008.76,
  'perf/wait_time_ratio': 0.4517
}
```

停止发生在 step 1 完成后、进入下一次 `update_weights` 时。训练侧调用
`actor_model.update_weights()`，内部要求所有 rollout engines 先 `pause_generation`，但其中一个
SGLangEngine actor 已经因为 Ray 节点死亡而消失：

```text
ray.exceptions.RayTaskError(ActorDiedError): ray::MegatronTrainRayActor.update_weights()
...
ray.exceptions.ActorDiedError: The actor died unexpectedly before finishing this task.
class_name: SGLangEngine
ip: 10.0.0.9
The actor died because its node has died.
the actor's node was terminated unexpectedly: health check failed due to missing too many heartbeats
```

因此直接原因是 **Ray 把 10.0.0.9 这台 rollout 节点判死了**，随后某个
SGLangEngine actor 消失，第二次 `update_weights` 无法完成，Ray job 以 exit code 1 失败。

当前推断的根因更偏资源/系统稳定性，而不是模型或 ATIF adapter：

- 这次高并发配置是 `TrialWorker=192`、rollout 节点上约 97 个 Apptainer/Singularity 相关进程；
- monitor 里看到 `timeout_lines=57002`、`traceback=407`、`failed_lines=296`，说明 env/agent 层有大量
  长尾和失败输出；
- 失败后 Ray 进程数为 0，GPU 显存归零，但 worker 节点上仍残留大量 Apptainer fuse/overlay 进程；
- Ray 的报错是 heartbeat miss，不是 CUDA OOM、不是 `RolloutConversionError`，也不是
  `train_rollout_logprob_abs_diff` 相关断言。

这轮有两个有价值结论：

1. `--use-rollout-logprobs` 在我们自定义 Harbor rollout 路径下是可用的，step 1 的
   `train/train_rollout_logprob_abs_diff=0.0`，说明 rollout logprobs 与训练侧 old logprobs 对齐。
2. 192 TrialWorker / 高 Apptainer 并发对 Ray 节点心跳稳定性有风险。下一次重启应降低并发，或者给
   rollout 和 env 执行拆更稳的资源隔离；否则高 GPU 利用率会换来 Ray 节点被判死。

下一步建议：

- 先清理 64 卡 job 上残留的 Apptainer fuse/overlay 和 Ray/tmux；
- 重启时把 TrialWorker 并发从 192 降到更保守值，例如 96 或 128；
- 保留 `rollout_max_response_len=4096`、128k context、`--use-rollout-logprobs`；
- 如果继续用高并发，需要增加 Ray node heartbeat 的稳定性配置，并降低每个节点同时启动/销毁
  Apptainer 容器的峰值。

### 2026-05-29 并发降到 96

用户确认下一轮先把并发降到 96。已修改当前实际使用的 64 卡 27B high-concurrency last-turn
入口配置：

```text
examples/harbor_terminal_rl/run_qwen35_27b_full89_64gpu_max100_n16_5epoch_high_concurrency.sh
examples/harbor_terminal_rl/harbor_terminal_bench2_full89_64gpu_27b_max100_high_concurrency_last_turn.yaml
examples/harbor_terminal_rl/monitor_full89_64gpu_27b_max100_high_concurrency.sh
```

新配置：

```text
harbor_num_concurrent_trials: 96
harbor_trials_per_node: 24
```

输出目录、trials 目录和 W&B group 同步从 `conc192_pn48` 改成 `conc96_pn24`，避免和上一轮失败数据混在一起：

```text
...resp4096_ctx128k_actor4_tp2pp8cp2_thd_rb16_os32_conc96_pn24_traincap128k_lpc32_logp_lastturn
q35-27b-tb2-64g-lastturn-logp-ctx128k-r4096-a4-tp2pp8cp2-rb16os32-c96p24
```

验证：

```text
bash -n examples/harbor_terminal_rl/run_qwen35_27b_full89_64gpu_max100_n16_5epoch_high_concurrency.sh \
        examples/harbor_terminal_rl/monitor_full89_64gpu_27b_max100_high_concurrency.sh
python -c "... yaml.safe_load ...; assert harbor_num_concurrent_trials == 96; assert harbor_trials_per_node == 24"
```

当前训练样本模式仍是：

```text
harbor_sample_mode: last_turn
```

trajectory-level 版本没有作为当前主路径启用。原因仍是 2026-05-28 的 strict trajectory adapter
失败：真实 Terminus2/SlimeSGLang 的下一轮 prompt 不是上一轮 prompt+completion 的严格 token
prefix extension，导致大量 `RolloutConversionError`。要恢复真正 trajectory-level update，需要重写
adapter：基于 ATIF/chat message history 重新用同一 tokenizer/chat template 构造整条轨迹，并显式
标注 assistant spans 的 loss mask，而不是依赖相邻 `prompt_token_ids` 的 strict prefix 关系。

### 2026-05-29 turn_grouped 更新模式

用户指出 `last_turn` 只训练最后一步是不对的，同时希望每个 turn 的 advantage 直接等于对应
trajectory 的 advantage，不做 `1 / num_turns` 之类的额外缩放。本轮实现了一个新的
`turn_grouped` 模式，作为 strict whole-trajectory adapter 失败后的折中方案：

- 训练样本粒度是 turn：一个 Terminus trial 里每次 LLM call 都会变成一个 slime `Sample`。
- GRPO group 粒度仍是 trajectory：`n_samples_per_prompt=16` 仍然表示同一个 task 的 16 条完整
  trajectory，而不是 16 个 turn sample。
- reward normalization 在 trajectory 级别做：先对同一个 task group 的 16 条 trajectory reward
  做 GRPO mean/std normalization，然后把每条 trajectory 的 normalized advantage 原样广播给它的所有
  turn sample。
- 不再依赖相邻 `prompt_token_ids` 的 strict prefix 关系，因此避开了当前 `trajectory` 模式在真实
  Terminus2 prompt history 上的 `RolloutConversionError`。

这不是简单地把配置改成 `harbor_sample_mode: turn`。直接用旧 `turn` 会有两个严重问题：

1. `rollout.py` 旧逻辑用 `len(group)` 判断 group 是否完整，并会把超过
   `n_samples_per_prompt` 的样本裁掉。turn 展开后 `len(group)` 是 turn 数，不是 trajectory 数，
   会导致一个 group 里大部分 turn 被错误截掉。
2. slime 默认 reward postprocess 会按 flat sample 序列 reshape 做 GRPO normalization。turn 数是
   变长的，不能再假设 flat samples 每 16 个就是同一个 task 的 16 条 trajectory。

已修改和新增文件：

```text
slime_plugins/harbor/atif_adapter.py
slime_plugins/harbor/rollout.py
slime_plugins/harbor/turn_grouped_reward.py
tests/test_harbor_atif_adapter.py
tests/test_harbor_rollout_async.py
tests/test_harbor_turn_grouped_reward.py
examples/harbor_terminal_rl/harbor_terminal_bench2_full89_64gpu_27b_max100_high_concurrency_turn_grouped.yaml
examples/harbor_terminal_rl/run_qwen35_27b_full89_64gpu_max100_n16_5epoch_high_concurrency_turn_grouped.sh
examples/harbor_terminal_rl/monitor_full89_64gpu_27b_max100_high_concurrency_turn_grouped.sh
examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh
```

核心代码语义：

```text
harbor_sample_mode: turn_grouped
harbor_group_unit: trajectory
```

`atif_adapter.py`：

- `turn_grouped` 复用 turn-level conversion。
- 每个 turn sample 会带上 `harbor_sample_mode=turn_grouped`、
  `harbor_turn_index`、`harbor_num_turns`、`harbor_terminal_reward` 等 metadata。

`rollout.py`：

- 新增 `harbor_group_unit` 概念。
- `turn_grouped` 默认使用 `trajectory` group unit。
- group 完整性判断改为：

```text
完整 group = 成功 trajectory 数达到 n_samples_per_prompt
```

- 对 `turn_grouped` 不再把 group 裁剪到 16 个 turn sample，而是保留 16 条 trajectory 的所有 turn。
- 增加相关 metrics：

```text
harbor/num_trajectory_samples
harbor/num_accepted_group_units
harbor/num_overflow_group_units
harbor/num_discarded_trajectory_samples
```

`turn_grouped_reward.py`：

- 新增 `trajectory_grouped_turn_rewards(args, samples)`。
- 按 `(harbor_rollout_id, harbor_rollout_group_index)` 找到 task group。
- 按 `harbor_sample_rank` 找到同一 task group 内的不同 trajectory。
- 先在 trajectory rewards 上做 GRPO normalization。
- 同一条 trajectory 的所有 turn 直接拿同一个 normalized advantage，不做额外缩放。

64 卡新入口：

```text
examples/harbor_terminal_rl/run_qwen35_27b_full89_64gpu_max100_n16_5epoch_high_concurrency_turn_grouped.sh
```

默认配置保持上一轮较稳的 96 并发：

```text
harbor_num_concurrent_trials: 96
harbor_trials_per_node: 24
ROLLOUT_BATCH_SIZE=16
OVER_SAMPLING_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=16
ROLLOUT_MAX_RESPONSE_LEN=4096
ROLLOUT_MAX_CONTEXT_LEN=131072
USE_ROLLOUT_LOGPROBS=1
KEEP_OLD_ACTOR=1
CALCULATE_PER_TOKEN_LOSS=0
USE_DYNAMIC_GLOBAL_BATCH_SIZE=1
CUSTOM_REWARD_POST_PROCESS_PATH=slime_plugins.harbor.turn_grouped_reward.trajectory_grouped_turn_rewards
```

这里开启 `USE_DYNAMIC_GLOBAL_BATCH_SIZE=1` 是为了避免 slime 在 flat turn samples 上按固定
`GLOBAL_BATCH_SIZE=256` 做尾部裁剪。turn 展开后样本数会随 trajectory 步数变化，固定裁剪可能切到
某些 trajectory 的部分 turn。

验证结果：

```text
bash -n examples/harbor_terminal_rl/run_qwen35_27b_full89_64gpu_max100_n16_5epoch_high_concurrency_turn_grouped.sh \
        examples/harbor_terminal_rl/monitor_full89_64gpu_27b_max100_high_concurrency_turn_grouped.sh \
        examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh

PYTHONPYCACHEPREFIX=/tmp/slime-pycache python -m py_compile \
        slime_plugins/harbor/atif_adapter.py \
        slime_plugins/harbor/rollout.py \
        slime_plugins/harbor/turn_grouped_reward.py

python -m pytest \
        tests/test_harbor_atif_adapter.py \
        tests/test_harbor_rollout_async.py \
        tests/test_harbor_turn_grouped_reward.py
```

`py_compile` 需要设置 `PYTHONPYCACHEPREFIX=/tmp/slime-pycache`，因为
`slime_plugins/harbor/__pycache__` 里有之前运行留下的非当前用户可写 pyc 文件。重定向 pycache 后编译通过。

单测结果：

```text
16 passed
```

当前剩余风险：

- turn sample 数量会比 last_turn 多很多，训练 token 吞吐和显存压力会上升。虽然我们现在开了
  `CALCULATE_PER_TOKEN_LOSS=0`、`LOG_PROBS_CHUNK_SIZE=32`、128k context 和 dynamic global batch，
  仍需要先用 64 卡跑一轮 smoke 看 actor train time 和 GPU memory。
- dynamic global batch 只保证按 DP size 对齐，仍然可能丢少量 flat turn samples。相比固定
  `GLOBAL_BATCH_SIZE=256`，它的尾部浪费会小很多，但如果后续要完全保证 trajectory 原子性，需要再加
  Harbor 专用 trim hook。
- 这仍不是真正的 single-sample whole trajectory update。它是 turn-level update，优势来自 trajectory
  reward。真正 whole trajectory 还需要重写 adapter，从 chat message history 重建 assistant spans。

### 2026-05-29 64GPU turn_grouped 重启记录

按用户要求，用新实现的 `turn_grouped` 模式在 64 张 H20 任务上重启 full89 / 27B 训练。

Taiji instance:

```text
self_improving_terminal_agent_rl_Qwen3_5_slime_fixed_dop-fuse
8B1D81989E25E915019E3C1BB0361459
```

使用入口：

```text
examples/harbor_terminal_rl/run_qwen35_27b_full89_64gpu_max100_n16_5epoch_high_concurrency_turn_grouped.sh
examples/harbor_terminal_rl/harbor_terminal_bench2_full89_64gpu_27b_max100_high_concurrency_turn_grouped.yaml
```

启动前清理：

- kill 旧的 `slime_full89_64gpu_27b_lastturn_*` tmux session。
- `ray stop --force`。
- 清理旧的 `sglang`、`train_async.py`、Ray 进程和 Apptainer/Singularity 进程。
- 额外用 `pkill -TERM -x fuse-overlayfs` 清理残留 fuse overlay 进程。

本次 tmux session：

```text
node0:
  slime_full89_64gpu_27b_turngrouped_launcher
  slime_full89_64gpu_27b_turngrouped_monitor

node1-7:
  slime_full89_64gpu_27b_turngrouped_worker
```

启动参数要点：

```text
MASTER_ADDR=10.0.0.4
RAY_NODE_IP_PREFIX=29.68
harbor_sample_mode=turn_grouped
harbor_group_unit=trajectory
harbor_num_concurrent_trials=96
harbor_trials_per_node=24
ROLLOUT_MAX_RESPONSE_LEN=4096
ROLLOUT_MAX_CONTEXT_LEN=131072
SGLANG_CONTEXT_LENGTH=131072
SEQ_LENGTH=131072
USE_DYNAMIC_GLOBAL_BATCH_SIZE=1
USE_ROLLOUT_LOGPROBS=1
CUSTOM_REWARD_POST_PROCESS_PATH=slime_plugins.harbor.turn_grouped_reward.trajectory_grouped_turn_rewards
```

启动后初始健康检查：

- `ray status` 显示 8 个 active nodes，无 pending nodes，无 recent failures。
- W&B run：

```text
https://wandb.ai/yucheng/slime-harbor-terminal-rl/runs/1fk9tgq2
```

- 日志确认 SGLang server 已通过 `/health_generate`、`/health`、`/server_info`、`/model_info`。
- actor 初始化后完成首次 `update_weights`：

```text
Timer update_weights end (elapsed: 13.3s)
```

- 随后进入 rollout：

```text
TrialWorker ... running r1 g0 s*
Using cached Singularity image: .../harbor_sif_cache_tb2_full_harbor_runtime/*.sif
Starting Slime Singularity container ...
```

05:40 左右的资源快照：

```text
Ray: 8 active nodes, 64 total GPUs
node0 GPU memory: about 14-22GB per GPU
rollout node example GPU memory: about 77-78GB per GPU
```

这说明 turn_grouped 版本已经跑过 Ray/SGLang/actor weight update，进入 terminal-bench rollout 阶段。
后续还需要继续观察第一轮 rollout 是否能完成并生成 `rollout_dumps/1.pt`，以及是否出现
`turn_grouped_reward` 或 dynamic global batch 相关异常。

### 2026-05-29 32GPU 9B turn_grouped 重启记录

用户重新拿到 32 张 H20 任务后，同步用新实现的 `turn_grouped` 模式重启 9B full89 训练。

Taiji instance:

```text
self_improving_terminal_agent_RL_32GPU_slime_v4
8B1D81049E49C12E019E5B4BDF170FCA
```

新增/使用入口：

```text
examples/harbor_terminal_rl/harbor_terminal_bench2_full89_32gpu_max100_5epoch_high_concurrency_turn_grouped.yaml
examples/harbor_terminal_rl/run_full89_grpo_32gpu_max100_n16_5epoch_high_concurrency_turn_grouped.sh
examples/harbor_terminal_rl/monitor_full89_32gpu_max100_5epoch_high_concurrency_turn_grouped.sh
```

启动前检查和环境修复：

- 4 个节点都能看到 `/path/to/workspace/code/slime`。
- 初始 `apptainer`/`singularity` 不在 PATH，按之前的服务器配置脚本在 4 个节点补装。
- 直接 import smoke 通过：

```text
python3 -c "import slime, slime_plugins.harbor.turn_grouped_reward"
```

- SIF smoke 在 node0-node3 均通过：

```text
singularity exec --no-mount home --no-mount tmp --no-mount bind-paths \
  --no-mount proc --writable-tmpfs --fakeroot --containall --pid \
  .../alexgshaw_regex-log_20251031.sif echo sif_ok
```

本次节点和 session：

```text
MASTER_ADDR=29.81.226.123
RAY_NODE_IP_PREFIX=29.81

node0:
  slime_full89_32gpu_9b_turngrouped_launcher
  slime_full89_32gpu_9b_turngrouped_monitor

node1-node3:
  slime_full89_32gpu_9b_turngrouped_worker
```

启动参数要点：

```text
harbor_sample_mode=turn_grouped
harbor_group_unit=trajectory
harbor_num_concurrent_trials=384
harbor_trials_per_node=128
ROLLOUT_BATCH_SIZE=16
OVER_SAMPLING_BATCH_SIZE=32
GLOBAL_BATCH_SIZE=256
USE_DYNAMIC_GLOBAL_BATCH_SIZE=1
ROLLOUT_MAX_RESPONSE_LEN=4096
ROLLOUT_MAX_CONTEXT_LEN=131072
USE_ROLLOUT_LOGPROBS=1
KEEP_OLD_ACTOR=1
CUSTOM_REWARD_POST_PROCESS_PATH=slime_plugins.harbor.turn_grouped_reward.trajectory_grouped_turn_rewards
```

这里的并发比 64 卡 27B 更激进：每个 rollout 节点 128 个 TrialWorker 并发，总并发 384。这样做是为了测试
32 卡场景下是否能把 terminal env 长尾隐藏掉，但它也更容易触发 Ray heartbeat、container startup 或文件系统压力。
如果后续出现 worker timeout / actor restart / container 启动抖动，优先降到 `harbor_trials_per_node=64`
和 `harbor_num_concurrent_trials=192`。

启动后初始健康检查：

- `ray status` 显示 4 个 active nodes，32 张 GPU，无 pending nodes，无 recent failures。
- W&B run：

```text
https://wandb.ai/yucheng/slime-harbor-terminal-rl/runs/qvr7kcx7
```

- SGLang 24 个 engine 均已启动并通过 `/health_generate`、`/health`、`/server_info`、`/model_info`。
- actor 从 `/path/to/models/Qwen3.5-9B_torch_dist` 加载成功。
- 首次 `update_weights` 成功：

```text
Timer update_weights end (elapsed: 15.6s)
```

- TrialWorker 已进入 terminal-bench rollout：

```text
TrialWorker ... running r1 g0 s2
TrialWorker ... running r1 g4 s12
terminus_2.py:282 - max_turns ... limited to 100
```

06:16 左右资源快照：

```text
Ray: 4 active nodes, 32 total GPUs
launcher node: 8 GPUs reserved by actor
rollout nodes: SGLang model loaded, about 52GB GPU memory per GPU before env traffic ramps up
```

当前仍需继续观察：

- 第一轮 rollout 是否能稳定产出 `trajectory_json` 和 `rollout_dumps/1.pt`。
- 384 并发下 Apptainer/Singularity container 数是否能稳定拉满，是否有 container startup timeout。
- `turn_grouped_reward` 是否在第一轮 batch 后正常写出 reward/advantage，没有触发 sample metadata 缺失。

06:22 二次健康检查：

```text
Ray: 4 active nodes, no pending nodes, no recent failures
node1 GPU util: about 90%-93%, about 52GB memory per GPU
node2 GPU util: about 33%-100%, about 52GB memory per GPU
node1 apptainer/singularity process count: 256
node2 apptainer/singularity process count: 256
```

launcher 日志确认 rollout 已经进入实际生成，而不是只在冷启动：

```text
SGLangEngine ... "POST /generate HTTP/1.1" 200 OK [repeated 115x across cluster]
Trajectory dumped to .../slime_harbor_trials/.../agent/trajectory.json [repeated 105x across cluster]
```

这次 32 卡 9B turn_grouped 高并发版本已经跑过：

- Ray 多节点组网。
- SGLang 24 engine 启动和健康检查。
- actor 加载 9B torch-dist checkpoint。
- 首次 actor -> rollout `update_weights`。
- 384 并发 TrialWorker/container 启动。
- 实际 `/generate` 和 trajectory dump。

下一步需要等第一轮 rollout group 收齐，看是否进入 reward post process / training step。如果这里失败，
重点检查 `turn_grouped_reward` metadata、dynamic global batch 的 sample trim、以及 container 长尾取消路径。

### 2026-05-30 32GPU 9B 停机原因与 32k/128k context 说明

01:19 检查 32GPU 9B turn_grouped 任务：

```text
self_improving_terminal_agent_RL_32GPU_slime_v4
8B1D81049E49C12E019E5B4BDF170FCA
```

结论：训练已经停了。Ray head 不存在，launcher tmux 也已经退出，只剩 monitor 和 gpu_keeper：

```text
ConnectionError: Could not find any running Ray instance.
tmux:
  gpu_keeper
  slime_full89_32gpu_9b_turngrouped_monitor
```

实际进度：

```text
trajectory_json=2560
rollout_dumps=5
checkpoint_files=0
```

没有 checkpoint 的原因是本次 `SAVE_INTERVAL=20`，而任务只跑到 rollout/train step 4 后失败，还没到第 20
次保存。

失败点不是 rollout 直接超时，也不是 08:58 出现的单个超长 context trial。主任务是在第 4 个 actor train
完成后，准备把新权重同步到 SGLang rollout engines 时失败：

```text
Timer actor_train end (elapsed: 5650.8s)
step 4: ...
Timer update_weights start
Timer update_weights end (elapsed: 21.1s)

ray.exceptions.ActorUnavailableError:
  ray::MegatronTrainRayActor.update_weights()
  self.weight_updater.update_weights()
  ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])

The actor ... is unavailable:
RpcError: failed to connect to all addresses;
last error: UNKNOWN: ipv4:29.81.228.139:20012:
Failed to connect to remote host: FD Shutdown
```

解释：actor train 第 4 步完成了，但下一次 `update_weights` 需要先对所有 rollout engines
发 `pause_generation`。其中一个 SGLang/Ray actor 在 worker 节点 `29.81.228.139:20012`
已经不可连接，Ray 把整个 job 判失败。更像是 rollout worker / SGLang engine 在长时间高并发和大量
container 残留压力下被重启、退出或网络连接断开，而不是 loss 计算本身报错。

本次 32GPU 9B turn_grouped 实际提交命令里有：

```text
--rollout-max-context-len 131072
--sglang-context-length 131072
--sglang-max-prefill-tokens 131072
--seq-length 32768
HARBOR_MAX_TRAIN_TOTAL_TOKENS=32768
```

含义必须区分：

- `rollout-max-context-len` / `sglang-context-length` 是 rollout 生成阶段的上下文窗口。这里确实是 128k。
- `HARBOR_MAX_TRAIN_TOTAL_TOKENS` 是把已经生成好的样本送进 actor train 前的训练侧 token cap。
  32GPU 9B 脚本为了避免 Megatron actor 训练 OOM，默认是 32768。
- `SEQ_LENGTH=32768` 是 actor train 的 Megatron sequence length，对应训练侧最大序列。

所以 32GPU 的 128k rollout 并没有变成 32k；真正变成 32k 的是 actor 训练输入。训练侧 cap 只影响
policy update 时保留多少 token，不限制 agent 在环境里能看多长上下文。

64GPU 27B turn_grouped 脚本不同，当前入口显式设成 128k train：

```text
examples/harbor_terminal_rl/run_qwen35_27b_full89_64gpu_max100_n16_5epoch_high_concurrency_turn_grouped.sh

ROLLOUT_MAX_CONTEXT_LEN=131072
SGLANG_CONTEXT_LENGTH=131072
SEQ_LENGTH=131072
HARBOR_MAX_TRAIN_TOTAL_TOKENS=131072
CALCULATE_PER_TOKEN_LOSS=0
```

因此：

- 32GPU 9B 这次：rollout 128k，train 32k。
- 64GPU 27B turn_grouped 脚本：rollout 128k，train 128k。

`rollout/truncated` 或 `truncated_ratio` 不是由 `HARBOR_MAX_TRAIN_TOTAL_TOKENS=32768`
直接造成的。更准确地说：

- slime 原生统计的是 `sample.status == Sample.Status.TRUNCATED` 的 sample 比例。
- 当前 Harbor adapter 里，trial 结果如果带有 `exception_info` 且 `exception_type` 包含
  `Timeout`，会被映射成 `Sample.Status.TRUNCATED`。
- turn/turn_grouped 模式会把这个 trial-level status 复制到同一个 trajectory 展开的每个
  turn sample。因此一个 100-step trajectory 如果最后 timeout，会把多个 turn sample 都记成
  truncated，`rollout/truncated` 会被 turn 展开放大。

这和单次 LLM 调用的 `finish_reason == "length"` 不是同一个指标。在 turn_grouped adapter
里，每个 turn sample 另外会记录：

```text
harbor_finish_reason_type
harbor_length_finish_turns
```

如果模型某个 step 的 completion 打满 `ROLLOUT_MAX_RESPONSE_LEN=4096`，finish_reason 会是
`length`，应主要看 `harbor/step_length_finish_*` 指标，而不是只看 `rollout/truncated`。
另外如果 prompt 已接近 128k，再请求 4096 token，也会触发 SGLang 400：

```text
prompt_tokens=127678
max_new_tokens=4096
requested total=131774 > context_len=131072
```

这类超长 context 是 rollout 阶段的问题，需要通过“根据剩余 context 动态降低 max_new_tokens”、
“到接近 context 上限时提前终止/压缩历史”、或“降低每步 response 上限/增加 history trimming”解决。
训练侧 32k cap 不会降低 trial timeout 或单步 length finish，它只会让 actor train 更省显存。

2026-05-30 用 32GPU 9B run 的 rollout dumps 做了离线核验：

```text
dump   truncated_ratio   finish_reason=length sample ratio   max response   max train total
1.pt   0.8358            0.0033                            4096           32768
2.pt   0.7152            0.0006                            4096           32768
3.pt   0.7593            0.0017                            4096           32768
4.pt   0.8375            0.0010                            4096           32768
5.pt   0.7543            0.0034                            4096           32768
```

同时 `truncated_by_finish_reason` 里绝大多数都是 `stop`，例如 `1.pt` 是
`{'stop': 5363, 'length': 17}`。这说明当前这批高 `rollout/truncated` 的主因不是单步
4096 token 打满，而是 trial-level Timeout 被 `turn_grouped` 展开到每个 turn sample。

后续调参时应分开看三类指标：

- `rollout/truncated_ratio` / `harbor/status_truncated_sample_ratio`：当前主要代表 trial timeout
  status 被展开后的 sample 比例。
- `harbor/step_length_finish_turn_ratio` / `harbor/step_length_finish_sample_ratio`：单个 LLM
  turn 是否因为 `ROLLOUT_MAX_RESPONSE_LEN` 打满而停止。
- `harbor/trimmed_*` 和 rollout dump 中的 `harbor_original_total_length`：训练侧是否因为
  `HARBOR_MAX_TRAIN_TOTAL_TOKENS` 截掉了过长 sample。

### 2026-05-30 新 32GPU 9B run 重启记录：2 train + 2 rollout + 128k train/rollout

新申请的 32GPU Taiji 实例：

```text
task: self_improving_terminal_agent_RL_32GPU_slime_v4
instance: 8B1D820C9E6E175B019E76664E8F259C
launcher: 29.81.224.250
worker-0: 29.81.226.123
worker-1: 29.81.228.79
worker-2: 29.81.242.55
```

本轮目标配置：

```text
model: /path/to/models/Qwen3.5-9B
torch_dist: /path/to/models/Qwen3.5-9B_torch_dist
actor nodes: 2
rollout nodes: 2
ROLLOUT_MAX_CONTEXT_LEN=131072
SGLANG_CONTEXT_LENGTH=131072
SEQ_LENGTH=131072
HARBOR_MAX_TRAIN_TOTAL_TOKENS=131072
ROLLOUT_MAX_RESPONSE_LEN=4096
HARBOR_MAX_TURNS=100
timeout: 3600s
sample mode: turn_grouped
progress reward: enabled
USE_ROLLOUT_LOGPROBS=1
CALCULATE_PER_TOKEN_LOSS=0
```

启动入口：

```text
examples/harbor_terminal_rl/run_full89_grpo_32gpu_9b_max100_actor2_rollout2_128k_timeout3600_turn_grouped.sh
examples/harbor_terminal_rl/harbor_terminal_bench2_full89_32gpu_9b_max100_actor2_rollout2_128k_timeout3600_turn_grouped.yaml
```

本轮先遇到的问题是 worker 分支在 `ray start --address ...` 后直接退出，导致 Ray head
短暂看到 4 个节点后 worker 变成不可用，placement group 最后只剩 launcher 的 8 张 GPU。
修复方式是在 worker tmux 里显式设置：

```text
WORKER_KEEP_ALIVE=1
```

重新启动后检查结果：

```text
2026-05-30 09:36
Ray Active: 4 nodes
Ray GPU total: 32
Pending nodes: none
Recent failures: none
```

launcher 日志确认 placement group 覆盖四台机器，每台 8 张 GPU；W&B run 创建成功：

```text
wandb run: 6iu6rpnf
wandb group: q35-9b-tb2-32g-turngrouped-progress-ctx128k-train128k-a2r2-timeout3600
```

SGLang rollout engine 在两个 rollout 节点启动成功，关键参数已经是 128k：

```text
context_len=131072
max_prefill_tokens=131072
max_total_num_tokens=510725
```

Megatron actor 在两个 train 节点启动并加载 9B torch_dist checkpoint 成功，随后进入初始
`update_weights`。日志里已经看到：

```text
loading release distributed checkpoint from /path/to/models/Qwen3.5-9B_torch_dist
Timer train_wait start
Timer update_weights start
```

09:42 左右已进入第一轮 Harbor rollout。日志中出现大量 `TrialWorker` terminal command：

```text
tmux_session.py:639 - Sending keys: [...]
terminus_2.py:1981 - Trajectory dumped to .../slime-r1-g4-s7-i71/agent/trajectory.json
POST /generate HTTP/1.1 200 OK
Decode batch ... gen throughput ...
```

当前 GPU 利用率快照符合 2 train + 2 rollout 的阶段性特征：

```text
node0 train: memory loaded, util 0% while waiting rollout batch
node1 train: memory loaded, util 0% while waiting rollout batch
node2 rollout: mostly 59-100% util, ~52GB used per GPU
node3 rollout: mostly 83-100% util, ~52GB used per GPU
```

这说明当前不是卡在 Ray/SGLang/Apptainer 冷启动，也不是只有 CPU/env 在跑；rollout 已经开始实际
请求模型并执行 terminal agent。下一步重点观察：

- 首个 rollout dump 是否落盘。
- `harbor/progress_reward_*` 是否出现。
- 首个 actor train step 是否能在 128k train cap 下通过。
- `rollout/truncated_ratio` 是否随 3600s timeout 明显下降。
- `train/train_rollout_logprob_abs_diff` 是否保持低位，验证 `--use-rollout-logprobs` 和 128k context 对齐。

### 2026-05-30 32GPU 9B 被查杀：最后 3 小时 GPU util 为 0

Taiji instance detail 显示本轮新 32GPU run 最终不是正常训练完成，而是被低利用率规则自动释放：

```text
instance: 8B1D820C9E6E175B019E76664E8F259C
state: END
create_time: 2026-05-30 09:01:17
TRAINING_RUNNING: 2026-05-30 09:08:20 -> 2026-05-30 21:45:24
msg: 【查杀执行】
GPU利用率(最近3小时): 平均0.00% / 最大0.00% / 最小0.00%
触发条件: GPU平均利用率 < 30% 且 运行时长 > 4小时
执行结果: 系统已自动释放该实例
```

能确定的事实：

- 09:36 左右 Ray 已经是 4 active nodes / 32 GPU，placement group 成功。
- 09:42 左右已经进入第一轮 Harbor rollout，当时 rollout 两个节点 GPU util 在 59%-100%。
- 21:45 查杀前最近 3 小时 GPU util 全为 0，说明至少 18:45-21:45 没有有效 GPU 进程在跑。

当前无法从节点 `/tmp` 取回 launcher tmux log，因为实例已释放；`jizhi_client logs` 只能看到 Taiji
启动/挂载层日志，拿不到 `/tmp/slime_full89_32gpu_9b_a2r2_128k_launcher.log`。当前 Codex 容器也没有
gy2 共享目录挂载权限，无法直接读取本轮的 W&B 本地日志/rollout dump，只能根据 Taiji 状态和启动脚本逻辑判断。

最可能的原因不是“rollout 正常跑但 GPU 偶尔空闲”，而是主 Ray job 已经失败、退出或进入不可恢复空转后，
worker 侧 keepalive 继续让 Taiji 实例存活：

```bash
if [ "${WORKER_KEEP_ALIVE}" = "1" ]; then
  echo "worker keep-alive enabled; keeping Ray worker session attached"
  tail -f /dev/null
fi
```

这次我们为了防止 worker 分支 `ray start --address ...` 后退出，显式用 `WORKER_KEEP_ALIVE=1`
启动了 3 个 worker tmux。这个修复解决了 Ray worker 生命周期问题，但副作用是：

- 如果 launcher 上的 `ray job submit ... train_async.py` 后来失败/退出；
- 或 Ray job 内部停止了 SGLang/Megatron/TrialWorker actors；
- worker pod 仍可能只剩 `tail -f /dev/null` / Ray daemon / tmux shell 这类非 GPU 进程；
- Taiji 仍认为实例 Running，但 GPU util 长时间为 0，最后触发查杀。

后续必须补一个外层 watchdog，不能只依赖 Taiji 低利用率告警：

1. launcher 侧 `ray job submit` 返回后，应主动 `ray stop --force`，并通过共享目录写 `STOP` marker。
2. worker keepalive 循环不要无条件 `tail -f /dev/null`，应每 60 秒检查 Ray head / job 状态 / STOP marker；
   如果主 job 不存在或 STOP marker 出现，就退出 worker 脚本。
3. launcher 侧另开轻量 watchdog：连续 N 分钟没有 `rollout_dumps/*.pt` 更新、没有 active
   SGLang/Megatron GPU 进程、或 Ray job 已非 RUNNING，则自动清理并退出。
4. 训练日志不要只写 `/tmp`，关键 launcher log 应同时 tee 到 `${BASE_FOLDER}/logs/launcher.log`，
   否则实例释放后无法复盘真正的 Ray job 退出点。

### 2026-05-31 32GPU 9B 64k + token dynamic batch restart

新 32GPU 实例：

```text
task: self_improving_terminal_agent_RL_32GPU_slime_v4
instance: 8B1D801D9E6E1752019E7980B85A3387
launcher: 29.81.242.16
workers: 29.81.240.69, 29.81.240.171, 29.81.244.168
```

本轮不再使用 128k train cap，改成 rollout / SGLang / actor train 统一 64k，并启用 token
dynamic batch：

```text
--rollout-max-context-len 65536
--sglang-context-length 65536
--sglang-max-prefill-tokens 65536
--seq-length 65536
HARBOR_MAX_TRAIN_TOTAL_TOKENS=65536
--use-dynamic-batch-size
--max-tokens-per-gpu 32768
CP_SIZE=2
```

这里 `max_tokens_per_gpu=32768` 是按 slime 的 CP 逻辑设置的：dynamic batch packing 时会乘
`context_parallel_size`，因此 CP=2 时等价于每个 sample 最多约 64k tokens；短 turn 可以被
更好地打包，长 turn 不再把训练侧直接推到 128k OOM 配置。

新增文件：

```text
examples/harbor_terminal_rl/harbor_terminal_bench2_full89_32gpu_9b_max100_actor2_rollout2_64k_timeout3600_turn_grouped.yaml
examples/harbor_terminal_rl/run_full89_grpo_32gpu_9b_max100_actor2_rollout2_64k_timeout3600_turn_grouped_dynbatch.sh
```

同时 `run_qwen35_9b_terminal_bench2_64h20.sh` 的 worker keepalive 已从无条件
`tail -f /dev/null` 改成共享 STOP marker：

```text
${BASE_FOLDER}/.stop_workers_${MASTER_ADDR//./_}_${RAY_PORT}
```

launcher 退出或失败时会写 marker，worker 每 60 秒检查一次并 `ray stop --force` 后退出。
这能避免主 job OOM 后 worker 空挂数小时、GPU util 为 0 直到 Taiji 查杀的问题。

本轮 Ray job 已提交成功：

```text
ray job id: raysubmit_6Wfwi7mtjykXscVk
wandb run id: 9c507c9s
wandb url: https://wandb.ai/yucheng/slime-harbor-terminal-rl/runs/9c507c9s
wandb group: q35-9b-tb2-32g-turngrouped-progress-ctx64k-train64k-dynbatch-a2r2-timeout3600
```

启动阶段确认：

- SGLang 16 个 engine 分布在 node0 `29.81.242.16` 和 node3 `29.81.244.168`。
- actor/train placement 在 node1 `29.81.240.69` 和 node2 `29.81.240.171`。
- TrialWorker 数量为 2，每个 rollout 节点 1 个，`max_concurrency=64`。
- SGLang server args 显示 `context_length=65536`、`max_prefill_tokens=65536`、`mem_fraction_static=0.5`。
- 00:07 左右 actor checkpoint 加载完成并执行 `update_weights`，随后开始 `r1` rollout。
- 00:10 检查已有 4 个 `agent/trajectory.json` 落盘，`rollout_dumps/*.pt` 仍为 0，说明第一批
  trajectory 正在回收中但还没凑满一个训练 batch。

### 2026-05-31 00:20 64GPU 27B turn_grouped 健康检查

64GPU 任务仍在运行：

```text
task: self_improving_terminal_agent_rl_Qwen3_5_slime_fixed_dop-fuse
instance: 8B1D81989E25E915019E3C1BB0361459
ray job: raysubmit_Gksd53naheHtk8cv
status: RUNNING
active ray nodes: 8/8
```

当前不是 rollout 阶段，而是在 27B actor train 阶段。日志显示：

```text
actor train: 66%|...| 5237/7945 [5:06:55<...]
```

GPU 利用形态符合 actor/train 阶段：

- actor 节点 4 个：node0 / node1 / node2 / node7，GPU 基本 95%-100% util。
- rollout/SGLang 节点 4 个：node3 / node4 / node5 / node6，GPU util 0%，但每卡约 77-79GB
  显存常驻，这是 generation 被 `pause_generation` 后等待下一轮 rollout 的状态。

截至本次检查：

```text
rollout_dumps: 6
latest dump: 6.pt, 2026-05-30 20:35
trajectory_json in monitor: 244 at 00:00
launcher log: no current Traceback/OOM/Killed in tail
```

结论：这轮没有挂死，也没有进入 GPU 全 0 空挂状态；它正在进行第 6 个 rollout dump 对应的
actor update。主要问题仍然是 27B + 128k + turn_grouped 的单次 actor update 很长。
