# slime-terminal-RL：Terminal Agent 的强化学习训练全解

> 本文描述 `slime-terminal-RL`（即 **slime v0.3.0 的 Harbor 变体**）如何对一个"终端智能体"（terminal agent，Harbor/Terminus2）做 RL 后训练：**流程、基础设施（infra）、算法、数据、流水线** 各自是怎么做的。
>
> 结论先行：这套系统把 **Harbor** 负责的"终端 agent trial / 环境 / verifier / 轨迹采集"，与 **slime** 负责的"SGLang rollout server / Megatron actor / 权重同步 / RL loss"缝合在一起。**缝合完全通过 slime v0.3.0 的公开插件钩子（`--*-path` dotted import）完成，不修改 slime 核心**。所有 Harbor 代码集中在 `slime_plugins/harbor/` 与 `examples/harbor_terminal_rl/`。

---

## 目录

1. [整体架构与角色分工](#1-整体架构与角色分工)
2. [什么是 Terminal Agent / Terminus2 / Trial](#2-什么是-terminal-agent--terminus2--trial)
3. [端到端流水线（一次 rollout 的完整数据流）](#3-端到端流水线一次-rollout-的完整数据流)
4. [Sample 模式：把一条轨迹变成训练样本](#4-sample-模式把一条轨迹变成训练样本)
5. [基础设施 / 执行层：终端环境怎么跑起来](#5-基础设施--执行层终端环境怎么跑起来)
6. [长尾问题与全异步 rollout](#6-长尾问题与全异步-rollout)
7. [RL 算法与奖励设计](#7-rl-算法与奖励设计)
8. [数据流水线：任务集、物化、课程学习、held-out eval](#8-数据流水线任务集物化课程学习held-out-eval)
9. [训练主循环与关键配置](#9-训练主循环与关键配置)
10. [长上下文 / 显存治理](#10-长上下文--显存治理)
11. [可观测性、指标与踩坑清单](#11-可观测性指标与踩坑清单)
12. [关键文件索引](#12-关键文件索引)

---

## 1. 整体架构与角色分工

slime 本身是一个把 **Megatron（训练）** 与 **SGLang（推理/rollout）** 通过 **Ray** 连接起来的 LLM RL 后训练框架，三个模块通过一个共享的 **Data Buffer** 交换数据：

- **训练侧（Megatron）** —— `slime/backends/megatron_utils/`：从 buffer 读 rollout 数据，跑优化器 step，再把新权重同步给 rollout。
- **Rollout 侧（SGLang + router）** —— `slime/rollout/` + `slime/backends/sglang_utils/`：生成轨迹（带 reward / verifier 输出），写回 buffer。`RolloutManager`（`slime/ray/rollout.py`）持有 SGLang server group + router。
- **Data Buffer** —— `slime/rollout/data_source.py`：桥接 prompt 初始化、自定义数据与 rollout 生成。

Harbor 变体在这个骨架之上，**只替换 rollout 生成逻辑和数据源**，把"生成一条轨迹"从"单轮/多轮文本生成"换成"让一个终端 agent 在真实容器里操作 shell 直到完成任务、再由 verifier 打分"。

```
                       ┌─────────────────────────────────────────────┐
                       │                 Ray 集群                     │
                       │                                              │
   train.py /          │   ┌────────────┐        ┌────────────────┐  │
   train_async.py ─────┼──▶│ RayTrainGroup │◀──weight sync──│ RolloutManager │  │
   (主循环)            │   │  (Megatron)  │       │  SGLang engines │  │
                       │   └──────┬───────┘       │   + router      │  │
                       │          │ 读            └───────┬────────┘  │
                       │      ┌───▼────────────┐          │ /generate │
                       │      │  Data Buffer   │          │           │
                       │      │ HarborTaskData │   ┌──────▼─────────┐ │
                       │      │    Source      │   │  Harbor 插件    │ │
                       │      └────────────────┘   │ generate_rollout│ │
                       └────────────────────────────┼───────────────┼─┘
                                                     │  Trial.run()  │
                                          ┌──────────▼────────────┐
                                          │  终端环境（沙箱）      │
                                          │ Singularity/Apptainer  │
                                          │   或 Daytona 云沙箱    │
                                          │  tmux + shell + verifier│
                                          └────────────────────────┘
```

**两个训练入口（顶层脚本）：**

- `train.py` —— 同步 / 可 colocate 的循环（rollout → train → 更新权重，按 `rollout_id` 顺序进行）。
- `train_async.py` —— 异步循环：在训练当前 rollout 的同时启动下一次 rollout；要求 `--colocate` 关闭。Harbor 的真实大规模训练用的就是它。

两者都调用 `parse_args()` 然后 `train(args)`，共用 `slime/ray/placement_group.py` 里的 `create_placement_groups` / `create_rollout_manager` / `create_training_models`。

**Harbor 如何"挂"进来（不改核心）：**

```bash
python train.py \
  --rollout-function-path slime_plugins.harbor.rollout.generate_rollout \
  --data-source-path     slime_plugins.harbor.data_source.HarborTaskDataSource \
  --custom-config-path   examples/harbor_terminal_rl/harbor_slime_config.yaml
```

`--*-path` 传的是"点号导入路径"（`module.submodule.function_or_Class`），由 `slime.utils.misc.load_function` 动态导入。因此**被这些 flag 指向的模块文件名不能带点号**。

---

## 2. 什么是 Terminal Agent / Terminus2 / Trial

- **Terminus2** 是 Harbor 的终端 agent（`harbor/src/harbor/agents/terminus_2`）。它跑一个**多轮 agent 循环**：每一轮 LLM 输出一个 **JSON 动作**（由 Terminus2 parser 解析），动作通过 **tmux** 在容器终端里执行，终端观察（命令输出、文件内容、报错）再拼进下一轮 prompt。典型命令如 `pip install ...`、`cat > /app/x.py`、`python3 solve.py`、`ls`、`pwd`。
- **一次 Trial** 就是一个 agent 对一个任务的完整运行：`Trial.create(config).run()` 产出一个 `TrialResult` 和一份 `trajectory.json`。Trial 命名模板为 `slime-r{rollout}-g{group}-s{sample}-i{index}`。
- **打分 / 验证**：Harbor 的 **verifier** 跑任务自带的测试，产出 `VerifierResult`。奖励从 `result.verifier_result.rewards["reward"]` 读取（key 可用 `harbor_reward_key` 配置）。终端任务集通常是 **Terminal-Bench 2**（binary checker：1.0 解决 / 0.0 未解决）。
- **ATIF / rollout_details**：Trial 暴露 `rollout_details` —— 每次 LLM 调用一条记录，含 `prompt_token_ids`、`completion_token_ids`、逐 token `logprobs`、`finish_reason` 等。这是把 Harbor 轨迹转成 slime 训练样本的原料。

slime 侧对 Terminus2 只做了一个极薄的子类化（`slime_plugins/harbor/agent.py`）：

```python
class SlimeTerminus2(SlimeTerminus2Mixin, Terminus2):
    """只在 llm_backend == 'slime_sglang' 时，把 LLM 换成 SlimeSGLangLLM，其余走 Harbor 原路。"""
```

`SlimeSGLangLLM`（`slime_plugins/harbor/llm.py`）实现 Harbor 的 `BaseLLM` 接口，但后端指向 slime 正在跑的 SGLang server：

- 用 `args.hf_checkpoint` 的**真实 tokenizer** + `apply_chat_template` 把 message 列表编码成 `prompt_token_ids`；
- POST 到 `http://{sglang_router_ip}:{sglang_router_port}/generate`，`return_logprob=True`；
- 返回 Harbor `LLMResponse`（含 `prompt_token_ids` / `completion_token_ids` / `logprobs`）。

两个关键正确性细节：

1. **token 计数用同一个真实 tokenizer**（`count_message_tokens`）。Terminus2 默认用 `litellm.token_counter`，对自定义 `model_name`（如 `slime-policy`）会退回 GPT `cl100k` tokenizer，**系统性低估** Qwen 的真实 token 数，导致 agent 的"上下文预算/主动摘要阈值"算错、prompt 溢出。这里改用真实 tokenizer 消除偏差。
2. **主动 context-limit 守卫**：发送前若 `len(prompt) + max_new_tokens > max_context_tokens`，直接抛 `ContextLengthExceededError`（而不是让 SGLang 返回 400 后被当成通用 RuntimeError），这样 Terminus2 的 fallback 摘要逻辑才能接住、摘要后重试，而不是整轮丢失。

---

## 3. 端到端流水线（一次 rollout 的完整数据流）

```
slime train_async.py
  └─ generate_rollout(args, rollout_id, data_source)         # slime_plugins/harbor/rollout.py
       ├─ data_source.get_samples(rollout_batch_size)        # 取任务 → base Sample（含 harbor_task_entry）
       ├─ 对每个任务组并发跑 Trial：
       │    _run_one_group → _run_one_with_semaphore → _run_one_trial
       │      ├─ _build_trial_config(...)                    # 合并 args.harbor_trial_defaults + task entry
       │      ├─ Trial.create(config).run()                  # SlimeTerminus2 在容器里多轮操作终端
       │      │     └─ SlimeSGLangLLM.call → SGLang /generate # 每轮 LLM 调用走 slime 的 SGLang
       │      └─ samples_from_trial_result(...)              # atif_adapter：TrialResult → list[Sample]
       ├─ 组级过采样 / 回填 / 尾部取消（保证凑够完整 group）
       ├─ _apply_rollout_sample_hooks(...)                   # reward 后处理、token 截断等
       └─ return RolloutFnTrainOutput(samples=list[list[Sample]], metrics=dict)
  └─ (Megatron) 用这些 Sample 跑 GRPO/GSPO 优化器 step
  └─ update_weights → 同步到 SGLang engines → 下一次 rollout
```

`generate_rollout` 的 v0.3.0 契约：

- 训练：返回 `RolloutFnTrainOutput(samples=list[list[Sample]], metrics=dict)`，其中 `samples` 是"分组"的（一个 group = 一个 prompt 的 n 条采样）。
- 评估：返回 `RolloutFnEvalOutput(data=dict, metrics=dict)`。

**并发与组装（`_generate_rollout_async`）** 的核心逻辑：

- `target_groups = rollout_batch_size`；每组要 `n_samples_per_prompt` 个"单位"。
- 用 `asyncio.Semaphore(harbor_num_concurrent_trials)` 限制**全局在飞 trial 数**。
- **过采样窗口** `over_sampling_batch_size`：允许"已接受 + 在飞"的组数超过 `target_groups`。
- 用 `asyncio.wait(..., FIRST_COMPLETED)` 收割：**先完成的组先进 batch**，不按提交顺序阻塞（早期版本这里有"按序 await"的 bug，慢组会卡住快组，已修）。
- **不完整组丢弃 + 回填**：若某组凑不齐 `n_samples_per_prompt`（有 trial 超时/失败），根据 `harbor_require_full_rollout_groups` 决定丢弃并从数据源再取一个新任务组回填，直到凑够 `target_groups`（上限 `rollout_batch_size × harbor_rollout_group_attempt_multiplier` 次尝试）。
- **尾部取消**：一旦够了 `target_groups`，剩余在飞组/样本 `task.cancel()` + `ray.cancel(force=...)`，避免长尾拖慢整步。

`_build_trial_config` 用深度合并顺序构造 Harbor `TrialConfig`：`args.harbor_trial_defaults` → entry 的 `trial_config` → entry 顶层的 `task`/`agent`/`environment`/`verifier`。默认 agent = `slime_plugins.harbor.agent:SlimeTerminus2`、`model_name=slime-policy`、`llm_backend=slime_sglang`；默认 environment 由 `harbor_environment_type`（`singularity` 或 `daytona`）决定。

---

## 4. Sample 模式：把一条轨迹变成训练样本

一条 Terminus2 轨迹是多轮 LLM 调用。把它转成 slime 训练样本（`atif_adapter.py`）时有多种"切法"，因为**tokenizer 对不断增长的对话历史并不总是把新 prompt 编成旧 prompt 的严格 byte 前缀**（Qwen 模板会插 `<think></think>`、摘要会重写历史等），朴素的"整轨拼接"会失败。因此演化出一组模式：

| 模式 | 一条轨迹产出 | loss_mask | 关键约束 / 用途 |
|---|---|---|---|
| `turn` | 每次 LLM 调用一个 Sample（最多 ~turn 数个） | 整个 completion `=1` | **最稳**。每个 sample 恰好是 `prompt[i]+completion[i]`。缺点：n=16 时一个任务组可能膨胀成上百样本，GRPO 组结构被 turn 数放大 |
| `last_turn` | 只取每条轨迹的**最后一次** LLM 调用 | 同上 | 保证每个 prompt 组恰好 n 条样本，GRPO 组内比较最稳。缺点：最后一轮的 **prompt** 仍含全部终端历史，可能极长（见 §10） |
| `trajectory` | 整条轨迹拼成 **1 个** Sample | 模型输出 token `=1`，环境观察 `=0` | 语义最完整，但要求相邻轮 prompt 严格前缀扩展，否则抛 `RolloutConversionError` |
| `trajectory_normalized` | 对齐则复用上下文，不对齐则**新开一个 chunk** | 输出 `=1`，上下文 `=0` | 容忍小的 prompt 后缀/响应边缘漂移；同一条 trial 的多个 chunk 共享一个 `group_id`；元数据标记是否近似匹配 |
| `trajectory_retokenized` | 针对"重新 tokenize 历史后重放同一段 assistant 文本"的场景合并 | 重放历史段 mask，后续上下文 `=0` | 依赖 `last_assistant_message_span`；文本完全一致才合并，内容被改写就 split。适合摘要开启的长轨迹 |
| `turn_grouped` | 每次 LLM 调用一个 Sample | 输出 `=1` | 同 `turn`，但**所有 turn 共享同一 `group_id`**，使 GRPO 分组仍在**轨迹**层级、而非被 turn 数放大 |
| `step` / `step_grouped` | 每步一个 Sample | 输出 `=1` | 类似 `turn_grouped`，带 `harbor_conditioning_mode=prompt_snapshot` 等步级元数据 |

统一契约：每个 Sample 满足 `len(loss_mask) == response_length`（slime 校验要求），并携带 `harbor_rollout_id / harbor_rollout_group_index / harbor_sample_rank` 元数据，用于后续按"轨迹"重新分组。

`group_id` 是 v0.3.0 的关键机制：**同一条 Terminus 轨迹派生出的兄弟样本共享 `group_id`**，slime 用它做 DP 切分和组感知的 loss 聚合，于是"轨迹级"的核算不再需要大量自定义核心逻辑。

---

## 5. 基础设施 / 执行层：终端环境怎么跑起来

Terminus2 需要一个**真实的、可交互的 Linux 环境**来执行 shell 命令。系统支持两条环境后端，通过 `environment.type` 选择：

### 5.1 Singularity / Apptainer（离线 HPC 集群）

不能依赖 Docker daemon，用 Apptainer/Singularity 直接跑 `.sif` 镜像。`SlimeSingularityEnvironment`（`singularity_env.py`）子类化 Harbor 的 `SingularityEnvironment`，只替换容器内的 bootstrap 脚本。实际启动大致是：

```
singularity exec --no-mount home --no-mount tmp --no-mount bind-paths --no-mount proc \
  --writable-tmpfs --fakeroot --containall --pid <task>.sif bash -c 'exec /staging/bootstrap.sh ...'
```

踩过的坑与对策（都在插件层解决，不改 Harbor 源）：

- `--containall --pid` 下无法挂 `/proc`（`operation not permitted`）→ `singularity_no_mount: home,tmp,bind-paths,proc`；但 tmux 需要 `/proc` → 再 `singularity_bind_host_proc: true`（`-B /proc:/proc`）。
- `--containall` 导致空 `/etc/resolv.conf` → 绑定宿主 `/etc/resolv.conf` 供 DNS/代理。
- 容器内服务硬编码 `127.0.0.1`，宿主健康检查够不到 → 把 staging 副本改成 Uvicorn host `0.0.0.0`。
- **`--writable-tmpfs` 反复重装依赖**：预建 tb2 SIF 只有任务镜像，缺 `uvicorn`/`fastapi`/`tmux`；`--writable-tmpfs` 让 pip/apt 装到临时层、每个 trial 重装一遍，在高并发下引发 `Environment start timed out`。对策：`augment_harbor_runtime_sifs.sh` 预先把这些运行时依赖烤进每个 `.sif`（输出到独立缓存、文件名不变），89/89 全部增强后启动不再重装。

### 5.2 Daytona 云沙箱（当前大规模训练主用）

`environment.type: daytona`，每个 trial 创建一个云沙箱。关键运维要素：

- **创建限流**：Daytona 每组织默认 **600 次 create/分钟 = 10/s**。全异步 rollout 一开始会在 ~1s 内爆发上百次 create（~28× 超限），全部被拒（`Generic S3 error: HEAD .../context.tar`）。对策：`rollout.py` 里的进程级 `_AsyncTokenBucket`，由 `harbor_create_rate_per_sec`（tokens/s）+ `harbor_create_rate_burst` 控制，在并发信号量内再叠一层限速。
- **S3 代理**：Daytona 上传 build context 到 S3 用的是 Rust `obstore.S3Store`（忽略 Python 代理环境变量）。`ensure_daytona_s3_proxy()`（rollout 模块加载时调用）给它打补丁，从 `HARBOR_DAYTONA_S3_PROXY_URL`/`HTTPS_PROXY` 等读代理。
- **沙箱生命周期 / 防泄漏**：`harbor_num_concurrent_trials` 是**协程**上限、不是**存活沙箱**上限。曾因 `stop()` 在 `self._sandbox is None` 时跳过删除 → 泄漏 538 个沙箱撑爆 RAM 配额（1000 GiB）。正确配方（见 curriculum YAML）：
  - 用 label（`harbor_managed`, `harbor_session=<trial_name>`）标记，`stop()` 在丢引用时按 label 删孤儿；
  - **同时设** `auto_stop_interval_mins`（> agent 墙钟，如 45min）**和** `auto_delete_interval_mins`（如 10min）—— 因为 Daytona 的 auto_delete 时钟只在 sandbox 被 stop 后才开始；只设 auto_delete 是无效的；
  - 外加每 20min 一次的 cron 兜底清理 `>45min` 的孤儿。
- **容量探测**：`daytona_capacity_probe.py` 逐步创建轻量沙箱探配额（disk/CPU/RAM）。

### 5.3 多节点 trial 分发（node-spread）

`RolloutManager` 是**单个** Ray actor（`num_cpus=1`）。若所有 trial 容器都是它的子进程，就会全挤在一个节点上，而 SGLang engine 和训练 GPU 却是分布式的。对策 `harbor_trial_node_spread: true`：

- 启动时按 placement group 的 `(node_ip, gpu_id)`，**每个节点起一个长驻 `HarborTrialWorker` Ray actor**（`trial_worker.py`），`max_concurrency = harbor_trials_per_node`；
- `_run_one_with_semaphore` 用 `itertools.cycle` **轮询**分发到各 worker，`worker.run_trial.remote(...)`；worker 死了就换下一个，全挂了才回落到进程内（可用 `harbor_fallback_to_inprocess_trials: false` 禁止回落，避免把高并发容器压力砸到 head 节点）；
- worker 用 `num_cpus=0.05`（沙箱算力在远端），健康检查超时 `harbor_trial_worker_healthcheck_timeout_sec`，`harbor_kill_unhealthy_trial_workers: false`（避免误杀繁忙 worker）。

**全局并发 = `harbor_num_concurrent_trials`（信号量）与 `harbor_trials_per_node × 节点数` 的下界**，二者叠加。

---

## 6. 长尾问题与全异步 rollout

Agentic rollout 的本质痛点：**不同 trial 时长差异极大**（agent 执行循环 + verifier 长尾，p50 ~377s、p95 ~1077s、max ~1947s；~90% 的 step 时间是 trainer 在等 rollout）。终端任务**无法从 token 前缀恢复**（状态 = 容器 + shell 会话 + 文件系统 + 轨迹 + verifier 状态），所以 slime 原生的 `--partial-rollout`（off-policy token mask）用不上。缓解手段分层：

1. **组级过采样 + 尾部取消**（`generate_rollout`）：`over_sampling_batch_size` 多发几组，先凑够先训练，剩下尾巴取消。
2. **组内部分截断**（`harbor_partial_group_min_units`，别名 `harbor_partial_group_min_trajectories`）：一个组发了 `n_samples_per_prompt` 个 trial，但只要收够 `min_units` 个可训练单位就早停、取消尾部 trial（`ray.cancel(force=False)` 让 `Trial.run()` 有机会清理沙箱）。这对 Daytona 并发配额有限的场景很关键。
3. **全异步 rollout**（`fully_async_rollout.py`）：与 `train_async.py` 的"pipeline N+1 与 N"不同，它在 RolloutManager 内起一个**常驻后台线程** `HarborFullyAsyncRolloutWorker`，持续不断地取组、跑 `_run_one_group`、把完成的组塞进队列；训练侧每次 rollout 只 drain 已完成的组。
   - 开关：`--rollout-function-path slime_plugins.harbor.fully_async_rollout.generate_rollout_fully_async` + 环境变量 `HARBOR_ROLLOUT_MODE=fully_async`；
   - 关键旋钮：`harbor_fully_async_max_concurrent_groups`（后台在飞组数，默认跟随过采样窗口）；
   - 关停清理：`rollout_cleanup_function_path` 指向 `cleanup_rollout`，`harbor_fully_async_shutdown_drain_timeout_sec` / `..._join_timeout_sec` 控制优雅排空；
   - 不支持 eval 模式；有 **stale-weight 风险**（暖队列里的组可能跨越权重更新边界）。

三层组装的典型配置（curriculum YAML，96-GPU 27B）：`n=20` 采样/组、`harbor_partial_group_min_units: 16`（收够 16 条取消 4 条尾巴）、`harbor_fully_async_max_concurrent_groups: 20`、`harbor_num_concurrent_trials: 320` → 320 并发沙箱。

---

## 7. RL 算法与奖励设计

### 7.1 优化器目标

支持 **GRPO / GSPO / REINFORCE++**（`advantage_estimator`）。分组优势归一化：组内 `reward - mean`，GRPO/GSPO 再可选除以 `std + 1e-6`（`grpo_std_normalization`）。

- **GRPO**（早期默认）：`rollout_batch_size × n_samples_per_prompt = global_batch_size`（如 16×16=256）。
- **GSPO**（当前大规模主用）：序列级、长度归一化的重要性采样 + 序列级 clipping，更适配长终端轨迹 + 全异步 off-policy gap + 训练/推理精度漂移。**要求极小的 clip**：`--eps-clip 4e-4`（GRPO 的 0.2 永远不会触发 clip）。

### 7.2 自定义 loss（`train_loss.py`）

`policy_loss_no_entropy_when_disabled`（通过 `--loss-type custom_loss --custom-loss-function-path ...` 挂入）与 slime 原生 policy loss 主体一致，但：

- 当 `entropy_coef == 0` 时**不构建带梯度的 entropy 图**（省下 27B 上 `[T, vocab]` 的显存尖峰）；但仍在 `torch.no_grad()` 下算 entropy 作为监控信号（`HARBOR_REPORT_ENTROPY`，默认开）—— entropy collapse 是 RL 退化的经典信号。
- 完整实现了 GSPO 分支（跨 CP `all_gather` 全序列 log-probs）、OPSM、TIS、KL loss（`use_kl_loss` + `kl_loss_coef`，作为对参考策略的锚）等。

### 7.3 奖励后处理钩子（`--custom-reward-post-process-path`）

设置该钩子后，**它替代 slime 默认的组归一化**，因此每个钩子都自己做归一化。都返回 `(raw_rewards, processed_advantages)`，且**优势按轨迹算一次、广播到该轨迹的所有 turn/chunk 样本**：

- **`turn_grouped_reward.trajectory_grouped_turn_rewards`**：把 `turn_grouped`/`step` 展开的多样本按轨迹重新分组，一条轨迹一个归一化优势，广播给它的每个 step。
- **`partial_reward.trajectory_grouped_progress_rewards`（CTRF 进度奖励）**：终端任务的 verifier 会把 pytest 进度写到 `verifier/ctrf.json`。对**未成功**的 trial，用 `passed/total` 通过率作为**有上限的部分奖励**：
  ```
  shaped = terminal >= 1.0 ? terminal : max(terminal, partial_reward_max * progress_ratio)
  ```
  `partial_reward_max` 默认 0.4（`harbor_partial_reward_max` / `HARBOR_PARTIAL_REWARD_MAX`）。终端成功奖励始终主导。
- **`length_shaped_reward.trajectory_grouped_length_shaped_rewards`（长度整形）**：只在一个 prompt 组的**成功轨迹之间**做 tie-break：输出越短奖励越高，幅度限制在 `± λ/2 × terminal`（`λ` 默认 0.2，`harbor_length_reward_lambda`），**永远不会让成功掉到失败之下**（失败恒为 0）。长度用 loss-mask 后的 `effective_response_length` 跨 chunk 求和。

---

## 8. 数据流水线：任务集、物化、课程学习、held-out eval

### 8.1 任务来源（`HarborTaskDataSource`）

数据源是**只读**的，按 `sample_offset` 顺序消费（**默认无 shuffle**）。任务可来自：

- `harbor_tasks_dir` + `harbor_tasks_glob`（如 `*/task.toml`）—— 目录发现，最常用（Terminal-Bench 2 的 89 个任务、TerminalWorld 的 1530 个任务）；
- `harbor_tasks` / `harbor_tasks_path`（内联 list / jsonl / yaml）；
- `harbor_stable_tasks_path`（parquet + `tasks_info.json`）—— 经 `stable_tasks.py` **物化**成 Harbor 任务目录（`instruction.md` / `task.toml` / `environment/Dockerfile` / `tests/` / `solution/`），并把已知 base image 映射到本地 SIF。

每个任务落到 `sample.metadata["harbor_task_entry"]`，供 `_build_trial_config` 消费。

### 8.2 课程学习（curriculum）

不是靠代码逻辑，而是**靠命名 + 无 shuffle 顺序消费**实现：把 1530 个任务做成 `NNNNNN_lenNNNNN_tw_XXXXXX` 的符号链接，按解答脚本行数从短到长零填充命名；`sorted(tasks_dir.glob("*/task.toml"))` 的字典序 == 由易到难的课程序。**关键：必须关掉 shuffle 课程才成立。**

### 8.3 Held-out eval

数据源有一个**独立的** eval 池（`harbor_eval_tasks_dir` / `harbor_eval_tasks_glob`，如 Terminal-Bench 2 的 89 任务），有自己的计数器，**永不推进训练游标**。`generate_rollout(evaluation=True)` 走 `_generate_eval_async`：

- 把**整个 eval 集各跑一次**，等所有任务完成（**不做尾部取消** —— 慢任务也是真实数据点），只加一个"总墙钟" `harbor_eval_total_timeout_sec` 兜底防病态挂死；
- **不**施加训练专用的样本钩子（长度整形、动态过滤），保证 `eval/harbor` 是干净的平均任务奖励（== TB2 pass rate）；
- 复用 `_run_one_group`，agent/沙箱/转换路径与训练完全一致；
- 产出绝对计数指标：`eval_num_resolved` / `eval_num_unresolved` / `eval_num_unscored_tasks` / `eval_resolved_rate_full`（分母 = 全 eval 集，把丢失样本记为未解决，是"悲观/诚实"的通过率）。

由 `--eval-interval` 触发，`SKIP_EVAL_BEFORE_TRAIN=0` 时还跑 step-0 baseline。

---

## 9. 训练主循环与关键配置

真实大规模训练都用 `train_async.py`（异步 driver）。本节给出 **96 卡** 与 **32 卡** 两套完整启动方式与关键配置。所有脚本都在 `examples/harbor_terminal_rl/` 下。

### 9.0 启动机制（两种规模通用）

启动分三层，从外到内：

1. **`start_*_tmux.sh {launcher|worker} [master_ip]`** —— **每个节点各跑一次**的编排入口。它在一个 tmux session 里设好该节点的角色/环境变量，然后 `exec` 一个 `run_*.sh` 包装脚本。launcher 节点还会把 config 模板里的 `trials_dir` 用 `sed` 替换成本次 run 专属路径。
2. **`run_*.sh`（包装脚本）** —— 设置模型 shape（`MODEL_SCRIPT`）、并行度、batch、上下文、算法等一大堆 `export`，最后 `exec` 到公共底座脚本 `run_qwen35_9b_terminal_bench2_64h20.sh`。
3. **`run_qwen35_9b_terminal_bench2_64h20.sh`（公共底座）** —— 真正干活：拉起 Ray（`RUN_ROLE=launcher` 跑 `ray start --head`，`worker` 跑 `ray start --address <master>:<port>`），等够 `EXPECTED_NODES` 个节点后，launcher 用 `ray job submit ... -- python3 train_async.py <一大串 args>` 提交训练。

关键机制：
- **多节点靠 `RAY_NODE_IP_PREFIX` 选对网卡 IP**（节点有多张网卡，必须挑与 `MASTER_ADDR` 同网段的，否则 GCS 判节点死、集群缩水）。
- **worker 靠 STOP-marker 保活**（`WORKER_KEEP_ALIVE=1`）：`ray start` 返回后 worker 不退出，直到 launcher 结束写下停止标记；这防止平台把"空闲"GPU 回收（利用率 <30% 超 4h 会被杀）。
- **权重同步**：actor 载入 torch_dist checkpoint 后 `update_weights` 推给 SGLang engine（期间 `pause_generation`，完成恢复）。
- **checkpoint 提速**：`STAGE_HF_CKPT_TO_TMP=1` / `STAGE_TORCH_DIST_CKPT_TO_TMP=1` 先把权重 rsync 到节点本地 `/tmp`，否则直接读共享盘会让 SGLang 卡在 `Multi-thread loading shards`、actor 进 `D` 态。

> 命名约定：`fully_async` 表示走 §6 的全异步 rollout（`HARBOR_ROLLOUT_MODE=fully_async`），driver 仍是 `train_async.py`。

---

### 9.1 96 卡启动（12 节点 · Qwen3.5-27B · Terminal-Bench 2 full89）

**代表脚本**：`start_full89_96gpu_27b_fully_async_tmux.sh`
→ 包装 `run_full89_grpo_96gpu_27b_max120_actor4_rollout8_batch8_n32_64k_timeout3600_trajectory_normalized_fully_async.sh`
→ config `harbor_terminal_bench2_full89_96gpu_27b_max120_actor4_rollout8_batch8_n32_64k_timeout3600_trajectory_normalized_fully_async.yaml`

**如何拉起**（launcher 节点先跑，其余 11 个 worker 节点各跑一次并传 launcher IP）：

```bash
cd examples/harbor_terminal_rl

# node0（launcher，先起）：
bash start_full89_96gpu_27b_fully_async_tmux.sh launcher

# node1 .. node11（每个 worker 节点各一次，<launcher_ip> 换成 node0 的 IP）：
bash start_full89_96gpu_27b_fully_async_tmux.sh worker <launcher_ip>
```

启动前按集群改这些：`RAY_NODE_IP_PREFIX`（默认 `29.68,29.79,29.81`，务必匹配你的节点网段）、`CODE_ROOT`/`SLIME_DIR`/`HARBOR_SRC` 路径、`BASE_FOLDER`（存 ckpt/日志）、`USE_WANDB`/`WANDB_API_KEY`。

**96 卡关键配置**：

| 维度 | 值 |
|---|---|
| 节点 / GPU | `EXPECTED_NODES=12`；**4 actor 节点（32 GPU）+ 8 rollout 节点（64 GPU）** |
| 模型 | `MODEL_SCRIPT=scripts/models/qwen3.5-27B.sh`（dense 27B） |
| 训练并行 | `TP=2 · PP=8 · CP=2 = 32`，`SEQUENCE_PARALLEL=1`，`QKV_FORMAT=thd` |
| rollout 引擎 | `ROLLOUT_NUM_GPUS_PER_ENGINE=2` → 32 个 SGLang engine，`SGLANG_MEM_FRACTION_STATIC=0.75` |
| batch / 采样 | `ROLLOUT_BATCH_SIZE=8` × `N_SAMPLES_PER_PROMPT=32` = **256 条轨迹/次更新**；`GLOBAL_BATCH_SIZE=256`；`USE_DYNAMIC_BATCH_SIZE=1`、`MAX_TOKENS_PER_GPU=32768` |
| rollout 模式 | `HARBOR_ROLLOUT_MODE=fully_async`；`harbor_sample_mode=trajectory_retokenized`；`OVER_SAMPLING_BATCH_SIZE=8` |
| 环境 / 并发 | Singularity（`SlimeSingularityEnvironment`，见 §5.1）；`harbor_trial_node_spread=true`、`harbor_trials_per_node=32`、`harbor_num_concurrent_trials=256`、`harbor_fully_async_max_concurrent_groups=8`；单容器 RAM 上限 `singularity_memory_limit_mb=51200`（50GB，防单个 `make -j` 撑爆节点） |
| 上下文 / 显存 | `ROLLOUT_MAX_RESPONSE_LEN=4096`、`ROLLOUT_MAX_CONTEXT_LEN=SGLANG_CONTEXT_LENGTH=65536`、`SEQ_LENGTH=65536`、`HARBOR_MAX_TRAIN_TOTAL_TOKENS=65536`、`LOG_PROBS_CHUNK_SIZE=32`、`USE_ROLLOUT_LOGPROBS=1` |
| 算法 | `advantage_estimator=grpo`、`ENTROPY_COEF=0.01`、`USE_KL_LOSS=1`、`KL_LOSS_COEF=0.001`、`LR=5e-7` |
| Agent | `max_turns=120`、`enable_summarize=true`、Terminus2 execution timeout 3600s |
| run 时长 / 存档 | `NUM_ROLLOUT=24`（≈2 epoch）、`SAVE_INTERVAL=2` |

> **96 卡的其它跑法**（同样是 12 节点 96 GPU，改包装脚本/config 即可）：
> - `start_terminalworld_full1530_96gpu_27b_daytona_tmux.sh` —— TerminalWorld 1530 任务、**Daytona 云沙箱**、GSPO + 长度整形 + 课程（`n=20`、`partial_group_min_units=16`、320 并发沙箱）。
> - `start_tb2_full89_96gpu_27b_daytona_overfit_tmux.sh` —— TB2 full89 过拟合验证。
> - **MoE 版**：`run_full89_grpo_96gpu_35b_a3b_...sh`（Qwen3.5-35B-A3B，`TP2×PP1×EP8`，DeepEP + SGLang dp-attention/EP，见 `MoE_adaptation.md`）。

---

### 9.2 32 卡启动（4 节点 · Qwen3.5-9B · Terminal-Bench 2 full89）

**代表脚本**：`start_full89_32gpu_9b_actor1_rollout3_batch3_n32_trajectory_normalized_fully_async_tmux.sh`
→ 包装 `run_full89_grpo_32gpu_9b_max100_actor1_rollout3_batch3_n32_128k_timeout3600_trajectory_normalized_fully_async.sh`
→ config `harbor_terminal_bench2_full89_32gpu_9b_max100_actor1_rollout3_batch3_n32_128k_timeout3600_trajectory_normalized.yaml`

**如何拉起**（launcher + 3 个 worker；该脚本还多一个 `monitor` 角色可选，用于打点计时）：

```bash
cd examples/harbor_terminal_rl

# node0（launcher）：
bash start_full89_32gpu_9b_actor1_rollout3_batch3_n32_trajectory_normalized_fully_async_tmux.sh launcher

# node1 .. node3（每个 worker 节点各一次）：
bash start_full89_32gpu_9b_actor1_rollout3_batch3_n32_trajectory_normalized_fully_async_tmux.sh worker <launcher_ip>

# 可选：任一节点起监控（打点 rollout/训练计时）：
bash start_full89_32gpu_9b_actor1_rollout3_batch3_n32_trajectory_normalized_fully_async_tmux.sh monitor
```

启动前注意：`RAY_NODE_IP_PREFIX`（默认 `29.79,29.81`）；若容器缺 Harbor/Daytona 运行时依赖，可设 `INSTALL_HARBOR_RUNTIME_DEPS=1`（启动时自动装）。

**32 卡关键配置**：

| 维度 | 值 |
|---|---|
| 节点 / GPU | `EXPECTED_NODES=4`；**1 actor 节点（8 GPU）+ 3 rollout 节点（24 GPU）** |
| 模型 | `MODEL_SCRIPT=scripts/models/qwen3.5-9B.sh`（dense 9B） |
| 训练并行 | `TP=1 · PP=8 · CP=1 = 8`，`SEQUENCE_PARALLEL=0`（单 actor 节点 8 GPU 全给 PP） |
| rollout 引擎 | `ROLLOUT_NUM_GPUS_PER_ENGINE=1` → 24 个 SGLang engine |
| batch / 采样 | `ROLLOUT_BATCH_SIZE=3` × `N_SAMPLES_PER_PROMPT=32` = **96 条轨迹/次更新**；`GLOBAL_BATCH_SIZE=96` |
| rollout 模式 | `HARBOR_ROLLOUT_MODE=fully_async`；`harbor_sample_mode=trajectory_normalized`；`OVER_SAMPLING_BATCH_SIZE=3` |
| 环境 / 并发 | Singularity；`harbor_trial_node_spread=true`、`harbor_trials_per_node=32`、`harbor_num_concurrent_trials=96`、`harbor_fully_async_max_concurrent_groups=3` |
| 上下文 / 显存 | `ROLLOUT_MAX_RESPONSE_LEN=4096`、`ROLLOUT_MAX_CONTEXT_LEN=SGLANG_CONTEXT_LENGTH=131072`（128k）、`SEQ_LENGTH=65536`、`HARBOR_MAX_TRAIN_TOTAL_TOKENS=32768`（32k，比 96 卡更紧，9B 单 actor 节点显存有限） |
| 算法 | `advantage_estimator=grpo`、`LR=5e-7`（其余同底座默认） |
| Agent | `max_turns=100`、`enable_summarize=false` |
| run 时长 / 存档 | `NUM_ROLLOUT=150`、`SAVE_INTERVAL=30` |

> **32 卡的其它跑法**：
> - `start_full89_32gpu_9b_trajectory_normalized_fully_async_tmux.sh` —— 2 actor + 2 rollout 的切法（`actor2_rollout2`，TP 布局不同）。
> - `start_tb2_train_v2_32gpu_9b_daytona_ppo_n4_bringup_tmux.sh` —— **PPO**（带 critic）+ Daytona bring-up。
> - `start_full89_32gpu_9b_daytona_2train2rollout_60x_retokenized_smoke_tmux.sh` —— Daytona + 部分组截断（`60x`）冒烟。

---

### 9.3 96 卡 vs 32 卡 —— 差异速查

| 项 | 96 卡（27B） | 32 卡（9B） |
|---|---|---|
| 节点划分 | 4 actor + 8 rollout | 1 actor + 3 rollout |
| 训练并行 | TP2·PP8·CP2，SP 开 | TP1·PP8·CP1，SP 关 |
| 每引擎 GPU | 2（32 engine） | 1（24 engine） |
| 轨迹/更新 | 8×32 = 256 | 3×32 = 96 |
| 训练截断上限 | 64k | 32k（显存更紧） |
| SGLang 上下文 | 64k | 128k |
| sample 模式 | trajectory_retokenized | trajectory_normalized |
| 全异步在飞组 | 8 | 3 |
| Agent max_turns / 摘要 | 120 / 开 | 100 / 关 |

核心取舍：**GPU 越少 → actor 节点越少 → 训练侧显存越紧**，于是 32 卡把训练截断压到 32k、单引擎只占 1 GPU、每步轨迹数减到 96；而 96 卡有 4 个 actor 节点撑起 TP2·PP8·CP2，可以吃 256 条轨迹/步、开 CP 与 SP。两者的 rollout/agent 语义完全一致，只是并行/batch/显存的量级不同。

---

### 9.4 56 卡启动（7 节点 · Qwen3.5-35B-A3B MoE · Daytona 沙箱）

本节描述在**当前 Ray 集群（7 节点 × 8 H20 = 56 卡）**上训练 MoE 模型
`Qwen3.5-35B-A3B`（256 专家 top-8，混合 GatedDeltaNet 线性注意力）的完整配置。它复用
上面的三层启动机制（tmux → wrapper → 基座脚本 `run_qwen35_9b_terminal_bench2_64h20.sh`
→ `ray job submit train_async.py`），只在**集群规模、路径、环境后端**上与 96 卡 35B-A3B
配方不同。涉及文件：

| 文件 | 作用 |
|---|---|
| `scripts/convert-qwen3.5-35B-A3B-gy4.sh` | 前置：HF → Megatron torch_dist 转换 |
| `examples/harbor_terminal_rl/run_35b_a3b_56gpu_daytona_actor4_rollout3_batch8_n32_64k_timeout3600_trajectory_normalized_fully_async.sh` | 训练 wrapper（导出全部超参后 exec 基座脚本） |
| `examples/harbor_terminal_rl/harbor_terminal_bench2_56gpu_35b_a3b_daytona_...fully_async.yaml` | rollout / agent / Daytona 环境配置 |
| `examples/harbor_terminal_rl/start_35b_a3b_56gpu_daytona_fully_async_tmux.sh` | 7 节点 tmux 启动器（转发 Daytona key 等） |

#### 前置步骤：checkpoint 转换（务必先做）

MoE actor 权重从 Megatron `torch_dist` 加载。在**单台 8 卡节点**上先跑一次转换：

```bash
bash scripts/convert-qwen3.5-35B-A3B-gy4.sh
# HF_CKPT = /path/to/shared_storage/hf_models/Qwen/Qwen3.5-35B-A3B
# SAVE_DIR= /path/to/shared_storage/hf_models/Qwen/Qwen3.5-35B-A3B_torch_dist
```

内部即 `torchrun --nproc-per-node 8 tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" --hf-checkpoint ... --save ...`
（world_size 8 ≤ 40 层；torch_dist 可重切分，能直接加载进下面的 PP1/EP8 训练布局）。产物同时用作
`--ref-load`（KL 参考模型）与首次 `--load`（后续断点续训自动切到 `SAVE_DIR`）。

#### 集群与节点划分

| 项 | 值 |
|---|---|
| 总资源 | 7 节点 × 8 H20 = 56 GPU |
| actor（训练） | 4 节点 / 32 GPU |
| rollout（推理） | 3 节点 / 24 GPU（每 8 卡一个 SGLang 引擎 ⇒ 3 引擎） |

slime 训练/推理资源是**分离**的：`ACTOR_NUM_NODES=4`、`ACTOR_NUM_GPUS_PER_NODE=8`、
`ROLLOUT_NUM_GPUS=24`、`EXPECTED_NODES=7`。基座脚本会断言 `actor_gpus % (TP·PP·CP)==0`。

#### 并行度（MoE 关键，与 96 卡一致）

```
TENSOR_MODEL_PARALLEL_SIZE=2   PIPELINE_MODEL_PARALLEL_SIZE=1   CP_SIZE=1
EXPERT_MODEL_PARALLEL_SIZE=8   EXPERT_TENSOR_PARALLEL_SIZE=1    SEQUENCE_PARALLEL=1
```

- actor：`TP2·PP1·CP1 = 2` ⇒ `DP = 32/2 = 16`；`EP8` 把 256 专家摊到每卡 32 个。
- **CP 必须为 1**：GatedDeltaNet 的循环 scan（`slime_plugins/models/qwen3_5.py:Qwen3_5GatedDeltaNet`）
  是逐 rank 的递归核，**不支持 context-parallel**。专家并行用 `EP8` 代替 CP 来省显存。

#### 批量与上下文

```
ROLLOUT_BATCH_SIZE=8   N_SAMPLES_PER_PROMPT=32   GLOBAL_BATCH_SIZE=256   OVER_SAMPLING_BATCH_SIZE=8
USE_DYNAMIC_BATCH_SIZE=1   USE_DYNAMIC_GLOBAL_BATCH_SIZE=1   NUM_ROLLOUT=24   SAVE_INTERVAL=2
SEQ_LENGTH / ROLLOUT_MAX_CONTEXT_LEN / SGLANG_CONTEXT_LENGTH / SGLANG_MAX_PREFILL_TOKENS = 65536
SGLANG_CHUNKED_PREFILL_SIZE=8192   HARBOR_MAX_TRAIN_TOTAL_TOKENS=65536   MAX_TOKENS_PER_GPU=16384
```

- 每步 8 组 × 32 条 = 256 条轨迹；上下文全链路统一 64k。
- `MAX_TOKENS_PER_GPU=16384` 是 MoE 保守起点（CP=1 无序列折半，vocab=248320 的交叉熵峰值更高）；
  若训练侧算力吃不饱，这是**第一个上调**的旋钮；若 actor step OOM 则下调。

#### 算法与正则（GRPO + KL 锚定，与 96 卡配方一致）

```
ADVANTAGE_ESTIMATOR=grpo   ENTROPY_COEF=0.005   USE_KL_LOSS=1   KL_LOSS_COEF=0.001   KL_COEF=0.00
LR=5e-7   KEEP_OLD_ACTOR=1   USE_ROLLOUT_LOGPROBS=0   CALCULATE_PER_TOKEN_LOSS=0   USE_CUSTOM_NO_ENTROPY_LOSS=1
```

- `USE_ROLLOUT_LOGPROBS=0`：old_log_probs 在训练侧重算（`train_loss.py` 两条路径都支持）。
- `KL_COEF` 必须为 0（`arguments.py` 断言 kl_coef 与 kl_loss_coef 不能同时非零）。
- 样本后处理 + CTRF 进度奖励沿用：`ROLLOUT_ALL_SAMPLES_PROCESS_PATH` /
  `ROLLOUT_DATA_POSTPROCESS_PATH`（token 截断）+
  `CUSTOM_REWARD_POST_PROCESS_PATH=slime_plugins.harbor.partial_reward.trajectory_grouped_progress_rewards`，
  `HARBOR_PARTIAL_REWARD_MAX=0.4`。

#### rollout 引擎（MoE 服务）

```
ROLLOUT_NUM_GPUS_PER_ENGINE=8   SGLANG_MEM_FRACTION_STATIC=0.75
EXTRA_SGLANG_ARGS="--sglang-enable-dp-attention --sglang-dp-size 8 --sglang-ep-size 8 \
                   --sglang-enable-dp-lm-head --sglang-moe-dense-tp-size 1 \
                   --sglang-mamba-scheduler-strategy extra_buffer"
EXTRA_MISC_ARGS="--moe-token-dispatcher-type flex --moe-enable-deepep"   # 训练侧 DeepEP all-to-all
```

24 rollout GPU ⇒ 3 个引擎，每引擎跨 8 卡，开 dp-attention + EP8 + mamba 调度。

#### Daytona 环境后端

与 96 卡 35B-A3B 模板（用本地 Singularity）不同，本配方在 YAML 里用 **Daytona 云沙箱**：

```yaml
environment:
  type: daytona
  delete: true
  override_storage_mb: 10240
  kwargs:
    connection_pool_maxsize: 128
    auto_stop_interval_mins: 90     # > 60min agent wall，正常 trial 不会被误回收
    auto_delete_interval_mins: 10
```

沙箱是**远端**的，不占那 24 张 rollout GPU；并发上限由 Daytona 账号配额而非 GPU 决定
（`harbor_num_concurrent_trials=96`、`harbor_fully_async_max_concurrent_groups=8`，按配额调）。
`HARBOR_CREATE_RATE_PER_SEC=6.0` 限制建沙箱速率。基座脚本会把
`DAYTONA_API_KEY / DAYTONA_JWT_TOKEN / DAYTONA_ORGANIZATION_ID` 注入 Ray `runtime_env`。

#### 需要你在启动前配置的占位项

本环境里 **`harbor` / `terminus2` / `daytona_sdk` 三个 python 包未安装、terminal-bench 任务集也不在本盘**，
因此以下项用环境变量占位，需在启动前设置（tmux 启动器会做校验）：

| 变量 | 含义 |
|---|---|
| `DAYTONA_API_KEY` | 你的 Daytona key（rollout 沙箱） |
| `HARBOR_SRC` | harbor 运行时源码目录（`harbor/src`，加入 PYTHONPATH） |
| `HARBOR_TASKS_DIR` | terminal-bench 任务池目录（含 `*/task.toml`），启动时 `sed` 进 YAML |
| `HARBOR_TRIALS_DIR` | trial 输出的临时目录（启动器默认给到 gy4 盘） |

（`harbor`/`daytona_sdk` 会由 `install_harbor_runtime_deps.sh` 在启动器里尝试安装，
若该脚本不存在则跳过，需自行 `pip install`。）

#### 启动方式

```bash
# 0) 先转换（单台 8 卡节点跑一次）
bash scripts/convert-qwen3.5-35B-A3B-gy4.sh

# 1) launcher 节点（node0），把 node0 的 IP 作为 MASTER_ADDR：
DAYTONA_API_KEY=dtn_xxx HARBOR_SRC=/path/harbor/src HARBOR_TASKS_DIR=/path/tb2 \
  bash examples/harbor_terminal_rl/start_35b_a3b_56gpu_daytona_fully_async_tmux.sh launcher <node0_ip>

# 2) 其余 6 个 worker 节点，传同一个 launcher IP：
HARBOR_SRC=/path/harbor/src \
  bash examples/harbor_terminal_rl/start_35b_a3b_56gpu_daytona_fully_async_tmux.sh worker <node0_ip>
```

launcher 提交 Ray job，worker 以 `WORKER_KEEP_ALIVE=1` 常驻直到收到 STOP marker。
日志见 `${BASE_FOLDER}/logs/launcher_<tag>.log`。首次可先用极小覆盖（如
`NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=2 GLOBAL_BATCH_SIZE=2`）冒烟，
确认 7 节点成环、3 个 SGLang 引擎起来、单个 Daytona trial 能跑通后再上全量。

#### 与 96 卡 / 32 卡 的差异

| 项 | 96 卡（27B dense） | **56 卡（35B-A3B MoE）** | 32 卡（9B dense） |
|---|---|---|---|
| 节点划分 | 4 actor + 8 rollout | **4 actor + 3 rollout** | 1 actor + 3 rollout |
| 训练并行 | TP2·PP8·CP2 | **TP2·PP1·CP1·EP8（DP16）** | TP1·PP8·CP1 |
| 每引擎 GPU | 2 | **8（3 engine，dp-attn+EP8）** | 1 |
| 轨迹/更新 | 256 | **256（8×32）** | 96 |
| 环境后端 | Daytona/Singularity | **Daytona** | Singularity |
| 关键约束 | — | **CP 必须=1（GatedDeltaNet）** | — |

---

## 10. 长上下文 / 显存治理

终端 agent 的 prompt 会随轮数快速膨胀（`linear_history: true` 每轮堆叠全历史）。这带来两个**独立**的失败模式：

1. **训练侧 OOM（不是 response 太长，而是 prompt 太长）**：`last_turn`/长轨迹的 prompt 可达 5–6 万 token，全序列前向 + `[T, vocab]` logits/entropy 尖峰直接 OOM。三层治理（全在插件层）：
   - **prompt 前缀截断**：`sample_postprocess.cap_total_tokens_*` 把 `tokens = tokens[-cap:]`，**只裁 prompt 前缀、完整保留 response 后缀**（`response_length`/`loss_mask`/`reward` 不变），`HARBOR_MAX_TRAIN_TOTAL_TOKENS` 控制；
   - **logprob 分块**：`LOG_PROBS_CHUNK_SIZE`；
   - **no-entropy 自定义 loss**（§7.2）。
   - 27B 上还需靠 **TP** 切分 vocab-parallel 的 logprob 张量（加 actor 节点/DP 不降单卡峰值）。
2. **推理侧 context-exceeded（prompt 超过 SGLang 窗口，返回 400）**：靠 `enable_summarize: true`（Terminus2 摘要重写历史）+ `SlimeSGLangLLM` 的主动 `ContextLengthExceededError` 守卫（§2）+ 把 SGLang 400 翻译成 typed error 让摘要接住。与 `trajectory_retokenized` 兼容：摘要重写导致前缀不对齐 → 该模式自动 split 出新 chunk，不会把未 conditioning 的 token 混进 loss。

为什么**不直接上 128k**：slime 动态批的 `first_fit_pack` 不切分超大样本，一个 131072-token 轨迹会独占一个 bin，逼 `MAX_TOKENS_PER_GPU` 翻倍 → 27B OOM；因此选择保持 64k + 摘要。

---

## 11. 可观测性、指标与踩坑清单

### 常用指标（W&B）

- Rollout 组装：`harbor/num_groups`、`num_samples`、`num_converted_samples`、`num_failed_trials`、`num_missing_rollout_details`、`num_group_attempts`、`num_incomplete_groups_dropped`、`num_partial_groups_accepted`、`num_cancelled_tail_group_tasks`、`group_latency_{mean,p95,p99,max}_seconds`。
- 奖励 / 健康：`harbor/reward_mean`、`status_*_sample_ratio`、`trial_timeout_ratio`、`any_step_length_finish_sample_ratio`（单轮撞 4096 的比例，长度崩塌的最响信号）、`train_context_trimmed_sample_ratio`、`avg_turns_per_trajectory`、`avg_output_tokens_per_turn`、`summarization_triggered_trajectory_ratio`。
- Eval：`harbor/eval_reward_mean`、`eval_num_resolved`、`eval_resolved_rate_full`、`eval_num_unscored_tasks`。
- 全异步：`harbor/fully_async_enabled`、`fully_async_queue_left`、`fully_async_max_concurrent_groups`。

### 高频踩坑（都有对应缓解）

- **`rollout/rewards ≈ 0` 不是 bug**：GRPO/GSPO 组归一化后优势本就均值≈0，看 `harbor/reward_mean` 和 `rollout/raw_reward`。
- **entropy=0 + KL=0 → 长度/熵崩塌**：策略退化成"重复刷到 4096"。加 `--entropy-coef 0.001~0.01`、小 `--kl-loss-coef`、或对 `finish_reason=length` 加惩罚；**别**调大 `ROLLOUT_MAX_RESPONSE_LEN`（多出来的是垃圾重复）。
- **tokenizer 返回 `BatchEncoding`**：朴素 `list(ids)` 得到的是 key 名 → SGLang 400。`_normalize_token_ids` 统一处理。
- **`Not enough samples 255 for global_batch_size 256`**：一个 verifier 超时 trial 就破坏固定 batch/组结构 → 用回填/过采样/部分组解决，别降 `GLOBAL_BATCH_SIZE`。
- **Ray 多节点心跳丢失**：`update_weights`（暂停所有 engine）时高 Apptainer 并发下节点失联 → 降并发、`healthcheck_timeout=30`、`kill_unhealthy=false`、更频繁 checkpoint。
- **Daytona 沙箱泄漏**：`stop()` 丢引用不删 + `auto_stop` 默认 0 → 撑爆配额；用 label 删孤儿 + `auto_stop`&`auto_delete` 双设 + cron 兜底。
- **别把 `trajectory.json` 计数当组完成**：它在 `result.json`/reward/转换之前就写了。
- **`--num-epoch` 会丢尾巴任务**（`len(data) % rollout_batch_size != 0`）→ 显式设 `NUM_ROLLOUT`。
- **SGLang context 参数要在 SGLang 侧也设**（`--sglang-context-length` 等），只改 slime 的 `rollout_max_context_len` 不够。

---

## 12. 关键文件索引

**Harbor 插件（`slime_plugins/harbor/`）**

| 文件 | 作用 |
|---|---|
| `rollout.py` | 训练/评估 rollout 入口 `generate_rollout`；组并发、过采样、回填、尾部取消、TrialWorker 分发、Daytona 限速；`_build_trial_config` |
| `fully_async_rollout.py` | 常驻后台线程全异步 rollout（`HarborFullyAsyncRolloutWorker` + `generate_rollout_fully_async` + `cleanup_rollout`） |
| `data_source.py` | `HarborTaskDataSource`：训练/eval 任务加载、顺序消费、checkpoint 状态 |
| `stable_tasks.py` | parquet/`tasks_info.json` → Harbor 任务目录物化 |
| `agent.py` | `SlimeTerminus2`：Terminus2 子类，切换到 slime SGLang 后端 |
| `llm.py` | `SlimeSGLangLLM`：Harbor `BaseLLM` → slime `/generate`，真实 tokenizer、context 守卫、routing replay、retokenized span |
| `atif_adapter.py` | `TrialResult`/rollout_details → `list[Sample]`，实现全部 sample 模式与 loss_mask |
| `partial_reward.py` | CTRF 进度部分奖励 + 轨迹分组归一化 |
| `length_shaped_reward.py` | 成功轨迹间的长度 tie-break 奖励整形 |
| `turn_grouped_reward.py` | 轨迹分组的 turn 奖励广播（GRPO/GSPO 归一化基元） |
| `train_loss.py` | `policy_loss_no_entropy_when_disabled`（含 GSPO/OPSM/TIS/KL 分支） |
| `sample_postprocess.py` | `cap_total_tokens_*`：prompt 前缀截断，保 response 后缀 |
| `singularity_env.py` | `SlimeSingularityEnvironment`：Apptainer/Singularity 任务层 |
| `trial_worker.py` | `HarborTrialWorker`：每节点长驻 Ray actor |
| `import_utils.py` | Harbor 发行元数据补丁、Daytona S3 代理补丁 |
| `routing_replay.py` | MoE routed-experts 回放（R3） |

**示例与文档（`examples/harbor_terminal_rl/`）**

- `README.md`、`INTEGRATION_SUMMARY.md`（集成/调试日志）
- `harbor_terminalworld_*` / `harbor_terminal_bench2_*` YAML + 对应 `run_*.sh` / `start_*_tmux.sh`
- 文档：`HARBOR_ROLLOUT_LONG_TAIL_ASYNC_NOTES.md`、`ROLLOUT_ENV_SPEEDUP_NOTES.md`、`MULTI_NODE_TRIAL_DISPATCH.md`、`PERF_DEBUG.md`、`TRAINING_INTERRUPTION_FAILURES.md`、`PARTIAL_REWARD_PROGRESS_SCORE.md`、`EVAL_TB2_INTEGRATION.md`、`docs/` 下的 fully-async / partial-cutoff 说明
- 运维工具：`augment_harbor_runtime_sifs.sh`、`daytona_capacity_probe.py`、`daytona_cleanup_slime_sandboxes.py`、`inspect_rollout_dump.py`

**顶层**：`train.py`（同步）、`train_async.py`（异步）、`HARBOR_V030_MIGRATION_PLAN.md`（迁移设计）、`CLAUDE.md`（仓库指南）。

---

*本文档基于当前 `harbor/v030-port` 分支源码与 `examples/harbor_terminal_rl/` 下的工程文档整理。具体旋钮默认值以代码为准。*
