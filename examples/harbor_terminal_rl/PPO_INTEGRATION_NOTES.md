# Harbor Terminal RL — PPO (Actor-Critic) 接入复盘 + 实验配置

更新时间：2026-06-22

本文记录把 **PPO（actor-critic + GAE）** 接入 Harbor 27B pipeline 过程中遇到的问题、根因、解决方案，以及最终的 **n=4 正式实验 setting**。PPO 是相对 GRPO/GSPO 的一条新路径，主要差异是引入了独立的 **critic（value 网络）**。

slime 把 PPO 作为一等 advantage estimator 支持（`--advantage-estimator ppo`），代码路径完整（无 `NotImplementedError`）。但 27B + Daytona + fully-async 这套组合下，PPO 的两个特性（**强制 offload** + **第二个 27B 模型**）触发了三个此前 GRPO/GSPO 从未碰到的坑。

---

## 0. PPO vs GRPO/GSPO 的本质区别（决定配置怎么设）

| | baseline 来源 | 需要组内多样本 (n>1) 吗 |
| --- | --- | --- |
| GRPO/GSPO | `get_grpo_returns`：advantage = 组内 reward 减组均值（再除 std）。baseline **就是兄弟样本的均值** | **必须**。n=1 时组内方差 0，advantage 全 0，无梯度 |
| **PPO** | `get_advantages_and_returns_batch`（`slime/utils/ppo_utils.py`）：标准 **GAE**，baseline 是 critic 的 `V(s)`，**逐条 trajectory 独立算** | **不需要**。critic 替代了"兄弟样本均值"这个 baseline 角色 |

代码确认（`slime/backends/megatron_utils/loss.py:635-653`）：PPO 分支调 `get_advantages_and_returns_batch(total_lengths, response_lengths, values, rewards, gamma, lambd)`，完全不看同组其它样本。

**推论**：PPO 跑 n=20 在 advantage 估计上是**冗余的** —— 这正是 critic 的意义。所以正式 PPO run 应把 n 调小（省 rollout），用更多不同 prompt 把 batch 补回来。详见 §4。

### critic 怎么接（无需单独 ckpt）
- `args.use_critic = (advantage_estimator == "ppo")`（`arguments.py:1818`），自动派生。
- critic 与 actor **共用同一个 placement group**（`placement_group.py:134` `result["critic"] = result["actor"]`）—— **不额外占 GPU**，96 卡照跑。
- critic 是独立的 `RayTrainGroup`（独立 optimizer / save），`copy.deepcopy(args)` 继承 actor 配置，从**同一个** `--load`/`--ref-load` 加载 27B LM backbone。
- value head（`output_size=1`）在加载后检测到 shape 不匹配（LM head `(248320,5120)` vs value `(1,5120)`）**自动重初始化**（`model.py:_reinitialize_critic_output_layer`，`normal_(0,0.02)`）。无需准备 critic checkpoint。
- ⚠️ **CKPT 覆盖 BUG（已修，2026-06-23）**：critic 是 `copy.deepcopy(args)`，**继承了 actor 完全相同的 `--save`/`--load`**（无后缀）。`save_checkpoint` 写的是 role-agnostic 分片到 `{save}/iter_NNNNNNN/`，而 `train_async.py:82-90` **先存 actor 再存 critic**，于是 **critic 把 actor 的 ckpt 原地覆盖**了。实测后果：`ppon4_055519` overfit run 跑完存下的 `iter_0000006/` 里 `output_layer.weight=[1,5120]`（critic value head），**actor 的 LM head `[248320,5120]` 被冲掉**——那个 ckpt 不能直接做 policy 推理，只有共享的 backbone 还在。**修复**（`placement_group.py` `_isolate_critic_checkpoint_dirs`）：critic 的 `save`/`load` 重定向到 `<dir>_critic` 兄弟目录，两个 role 各存各的；resume-load 对称（load 用同一 arg）。CPU 单测 `tests/test_critic_checkpoint_isolation.py`（5 passed）。**任何 `--use-critic`（PPO）run 此前都中招**——查存下来的 ckpt 用 `output_layer.weight` 的 shape 鉴别（`[1,5120]`=critic，`[248320,5120]`=actor）。

- 训练时序（`train_async.py:70-90`）：每步 `actor_trains_this_step = rollout_id >= num_critic_only_steps`。`critic.async_train()` 先算 values → 再 `actor.async_train(external_data=values)`，**串行**。
- actor 训练的两个阶段在日志里是 `actor forward`（算 logprob）→ `actor train`（更新）；critic 是 `critic train`。**盯日志时别只 grep `actor train`，会漏掉 `actor forward`。**

---

## 1. 总览：PPO 特有的三个坑

| 时间 | 实验 | 失败位置 | 直接错误 | 核心判断 |
| --- | --- | --- | --- | --- |
| 2026-06-22 | 96GPU 27B PPO smoke | critic `async_init` | `RuntimeError: TorchMemorySaver is disabled ... expandable_segments is not supported yet` | PPO `use_critic` 强制 `offload_train=True` → 激活 torch_memory_saver → 它拒绝 `expandable_segments:True`。GSPO 不 offload 所以从没碰到 |
| 2026-06-22 | 96GPU 27B PPO smoke（重启后） | rollout 0 收集 | 无报错，`collected 0/16, completed=0` 卡死 39min | 320 并发把 coordinator 节点 raylet 的 nofile 软上限 1024 打满（794 socket+199 pipe），trial 结果 future 回传被阻塞。**这是潜伏在每个 320 并发 run 里的坑，GSPO 只是 trial 完成快、fd 周转快没触顶** |
| 2026-06-22 | 96GPU 27B PPO smoke | actor train 帧 = 0（虚惊） | （非 bug） | 我 grep `actor train` 全 0，误判 actor 没训。实际 actor 先 `actor forward` 再 `actor train`，且按 `num_critic_only_steps=1` 要等 rollout 1 的 critic step 跑完才轮到 actor。grep 关键词错了 |

---

## 2. 坑 1 — TorchMemorySaver × expandable_segments（critic init 崩溃）

**症状**：job 提交后，`create_training_models` 里 `critic_model.async_init` 崩：
```
RuntimeError: TorchMemorySaver is disabled for the current process because expandable_segments is not supported yet.
  at slime/backends/megatron_utils/actor.py:79 (Set torch_memory_saver.memory_margin_bytes)
  -> torch_memory_saver/entrypoint.py:199 _sanity_checks
```

**根因链**：
1. PPO → `args.use_critic=True` → **强制 `offload_train=True`**（`arguments.py:1862`）。
2. offload 走 `torch_memory_saver`（`actor.py:79` 设 margin / `:197/206` pause-resume，actor 和 critic 在同批 GPU 上时分复用显存）。
3. `torch_memory_saver._sanity_checks` **硬性拒绝** `PYTORCH_CUDA_ALLOC_CONF` 含 `expandable_segments:True`。
4. base 脚本默认 `expandable_segments:True`（line 90，防碎片化）→ 崩。

**为什么 GSPO 没事**：GSPO 从不 `offload_train`，根本不碰这条路径。repo 里那个"能跑"的 PPO 范例（`examples/geo3k_vlm_multi_turn/run_ppo_npu.py`）是 **NPU**，不用 CUDA 的 torch_memory_saver，所以绕过了。

**修复**：PPO wrapper 里覆盖 alloc conf 为**不含 expandable_segments** 的值：
```bash
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"
```
代价：失去 expandable_segments 的碎片化抑制，`max_split_size_mb:256` 是温和缓解。实测显存其实宽裕（`free_GB 71-91`），margin-OOM 是 transient warning（见下）。

---

## 3. 坑 2 — 320 并发打满 coordinator nofile（rollout 卡死 collected 0/16）

**症状**：集群正常、critic init 通过、SGLang 健康、trial 在跑（agent 不停发命令、dump trajectory），但 `RolloutManager` 持续报 `rollout 0: collected 0/16, queue=0, completed=0, elapsed=...`，卡了 39 分钟一个 trajectory 都没收回。

**根因**（`ssh <coordinator> 'rp=$(pgrep -f raylet|head -1); grep "Max open files" /proc/$rp/limits; ls /proc/$rp/fd|wc -l'`）：
- coordinator 节点（launcher + RolloutManager 所在 = node0）raylet 的 **nofile 软上限默认 1024**（硬上限 ~1e6）。
- 320 个并发 trial 的结果 future + SGLang 连接，每个吃 socket+pipe。在 coordinator 聚集到 **794 socket + 199 pipe = 1024**，触顶 → `File descriptor limit reached. Retrying`（刷了 395 次）。
- trial 结果回传不到 coordinator → 正反馈死锁（fd 只进不出），**不会自愈**。worker 节点没事（raylet ~76-130 fd），只在 coordinator 聚集。

**为什么 GSPO 没事**：GSPO 完成的 run（同样 320 并发、同 base 脚本）fd 错误 = 0，纯粹因为 trial 完成快（首个 completion 在 elapsed=300s），fd 快速周转始终低于 1024。**慢一点（PPO critic+actor 串行，或运气差的慢首轮）就越过 1024 死锁。潜伏在每个 320 并发 run 里。**

**修复**：base 脚本 `run_qwen35_9b_terminal_bench2_64h20.sh` 在 `log_event` 定义后、**任何 `ray start` 之前**把 nofile 软上限拉到硬上限：
```bash
RAISE_NOFILE_TO=${RAISE_NOFILE_TO:-$(ulimit -Hn 2>/dev/null || echo 1048576)}
if [ -n "${RAISE_NOFILE_TO}" ] && [ "${RAISE_NOFILE_TO}" != "unlimited" ]; then
  ulimit -n "${RAISE_NOFILE_TO}" 2>/dev/null && log_event "raised nofile soft limit to $(ulimit -n)" || ...
fi
```
raylet + 所有子进程（SGLang、trial worker、train actor）继承这个高上限。验证：launcher log 有 `raised nofile soft limit to 1000000`，且 `/proc/<raylet>/limits` 的 `Max open files` = 1000000。这是 base 脚本改动，**所有 run 受益**（不只 PPO）。

---

## 4. margin-OOM（良性，非崩溃）

PPO 双 27B 时分复用显存时，日志会刷大量：
```
[torch_memory_saver.cpp] TorchMemorySaver::malloc return OOM since memory_margin_bytes=1073741824 (alloc)size=... free_bytes=1.1-1.7GB
```
**这是良性的**：torch_memory_saver 保留 1GB margin，当某次分配会吃掉 margin 时**拒绝并让上层 fallback**（重试/换路径），而不是让 CUDA 真 OOM 崩。判据：(1) 训练步照常推进（`N.pt` 落盘、rollout 前进）；(2) 无 python 级 `RuntimeError`/`CUDA out of memory`/Traceback。smoke run 触发 738 次全是 warning，训练正常。若**真**显存紧（伴随 python OOM），退路：`MAX_TOKENS_PER_GPU 32768→16384`。

---

## 5. 当前正式实验 setting（n=4 PPO）

文件（`examples/harbor_terminal_rl/`）：
- wrapper: `run_tb2_full89_96gpu_27b_daytona_ppo_n4_fully_async.sh`
- yaml: `harbor_tb2_full89_96gpu_27b_daytona_ppo_n4.yaml`
- starter: `start_tb2_full89_96gpu_27b_daytona_ppo_n4_tmux.sh`
- 首个 run：RUN_TAG `ppon4_055519`，2026-06-22 起于 server 018。

### 相对 n=20 smoke 的三处刻意改动（用户指定）

| 维度 | n=20 smoke | **n=4 正式** | 理由 |
| --- | --- | --- | --- |
| `N_SAMPLES_PER_PROMPT` | 20 | **4** | PPO 靠 critic baseline，组内多样本对 advantage 冗余（§0）。n=4 不取 1 是为保 group plumbing + length-off 后仍有多条样本 |
| length reward | λ=0.4 ON | **OFF** | 清空 `CUSTOM_REWARD_POST_PROCESS_PATH` → reward 回归纯 terminal binary（0/1）。base 脚本对空值跳过该 flag。**token-cap postprocess（`ROLLOUT_ALL_SAMPLES_PROCESS_PATH` / `ROLLOUT_DATA_POSTPROCESS_PATH`）不是 reward，保留** |
| oversampling | n=20 发、收前 16 砍尾（组内）+ 组级 | **仅组级超采** | `partial_group_min_units=4`（=n，**不砍尾**，一组 4 个全收）+ `max_concurrent_groups=24`（>batch，快组补慢组，抗 Daytona 长尾）。组内砍尾在 n=4 下省不了多少且削弱 PPO 有效 batch |

### batch / 并发 / epoch

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `ROLLOUT_BATCH_SIZE`（prompt 数） | **64** | 把 batch 从 16 拉到 64 补回规模；PPO 不在乎样本来自多少不同 prompt，多 prompt 反而对 value 函数更好 |
| `N_SAMPLES_PER_PROMPT` | 4 | — |
| `GLOBAL_BATCH_SIZE` | **256** | = 64 × 4 traj/步（smoke 是 320） |
| `OVER_SAMPLING_BATCH_SIZE` | 64 | 组级超采窗口 |
| `harbor_fully_async_max_concurrent_groups` | **24** | >64-prompt 稳态，快组补慢组 |
| `harbor_partial_group_min_units` | **4** | =n，不砍尾 |
| `harbor_num_concurrent_trials` | **96** | 24 组 × 4 = 96 并发 sandbox（smoke 是 320，省 70%，~960GiB << 5000GiB tier4 配额） |
| `harbor_trials_per_node` | 12 | 96 / 8 rollout 节点 |
| `NUM_ROLLOUT` | **7** | 89/64≈1.4 步/epoch → 7 步 ≈ **5 epoch**（对齐 GSPO overfit 的 5 epoch；注意 batch 大 4 倍后 7 步就是 5 epoch，跑 28 步会是 ~20 epoch 过拟合崩） |

### 不变的部分（与 smoke / GSPO 对齐）
- **raw 模式**（bridge 训不了 27B GDN packed-seq，见 `bridge-mode` memory）；actor+critic 从 torch_dist `--load`/`--ref-load` 加载，value head 自动重初始化。
- `--advantage-estimator ppo`、`--eps-clip 0.2`（PPO 标准 per-token clip，**不是** GSPO 的 4e-4）、`--num-critic-only-steps 1`（step0 critic-only 暖启）。
- KL OFF、no rollout-logprobs、entropy OFF + display。
- 数据：TB2 full 89（train == eval，**overfit** run）；override_timeout 3600s；verifier wall 900s；eval total wall 3600s。
- TP2×PP8×CP2=32；12 节点（4 actor + 8 rollout）。
- 两个修复都带上：base 脚本 nofile ulimit + wrapper 的 `max_split_size_mb:256` alloc conf。
- Daytona 防泄漏：`auto_stop_interval_mins: 90` + `auto_delete_interval_mins: 10`（见 `daytona-auto-delete-needs-auto-stop` memory）。

### 观察看点
1. **reward_mean 趋势**：纯 binary（无 length tie-break），看起点（smoke 是 0.27）和 5 epoch 内能否往上走。
2. **value_loss**：critic 是否在学、下降。
3. **单步 wall time**：critic+actor 串行各跑一遍 27B，仍是主要耗时；n=4/96 并发让 rollout 收集压力小很多。
4. **margin-OOM**：预期出现但应是良性 warning（§4）。
5. **96 并发是否够**：组级超采下收集应顺畅，不应再现 collected 卡死（nofile 已修）。

---

## 5.5 并发吞吐调研（2026-06-24，rollout 才是瓶颈）

`twppon4_20260624_003801`（by_length mid830 PPO n=4）实测发现：**瓶颈在 rollout 产率，不在训练**——这推翻了之前"训练慢所以不必拉满并发"的假设。本节记录调研结论，供下次重启提并发用。

### 实测的瓶颈证据

| 观测 | 数值 | 来源 |
| --- | --- | --- |
| step0 critic 训练计算 | ~28min（02:37 算梯度 → 03:05 权重同步完） | driver `timer.py` 时间戳 |
| rollout 1 填满 64 组耗时 | **>1h40min**（03:05 等到 04:45 仍未出 step1） | driver log |
| 单步 wall（冷启动+坏镜像期） | **~1.5–2h**，绝大部分耗在 rollout | 上两行相减 |
| **SGLang 32 引擎 token 占用** | 峰值 **0.23**（`full token usage`，上限 0.75） | SGLangEngine `Decode batch` 行 |
| **SGLang `#running-req` 分布** | 大部分 **0–1**（2663 次=1，1441 次=0，零星到 8–9） | 同上 |

结论：64 GPU、32 个 SGLang 引擎（`ROLLOUT_NUM_GPUS=64` / `ROLLOUT_NUM_GPUS_PER_ENGINE=2`）严重空转，推理侧有大量余量，rollout 慢是因为**同时在飞的 trial 太少**（96 个 sandbox 喂 32 个引擎，平均每引擎 ≤1 个请求）。

### 三个并发旋钮怎么联动（代码已查实）

| 旋钮 | 现值 | 消费点（file:line） | 作用 |
| --- | --- | --- | --- |
| `harbor_fully_async_max_concurrent_groups` | 24 | `slime_plugins/harbor/fully_async_rollout.py:106-110`（property）/ `:128`（producer `while len(active_tasks) < max_concurrent_groups`） | producer 同时在飞的 **group** 数 |
| `harbor_num_concurrent_trials` | 96 | `fully_async_rollout.py:119-120` 建 **单个全局** `asyncio.Semaphore`；`rollout.py:800-827` 每个 sample `async with semaphore` | 卡同时活着的 **trial=sandbox** 数 |
| `harbor_trials_per_node` | 12 | `slime/ray/rollout.py:508` 每个 `HarborTrialWorker` actor 的 `max_concurrency=per_node` | 每节点 TrialWorker actor 并发槽 |

**硬约束**（三者必须一起改，否则零增益）：
```
num_concurrent_trials  ==  max_concurrent_groups × n_samples_per_prompt  ==  trials_per_node × num_rollout_nodes
        96             ==            24 × 4                              ==          12 × 8
```
- 现在三者精确咬合到 96。
- **只加 `max_concurrent_groups` 没用**：semaphore 仍卡 96，更多 group 只会更激烈地抢同样 96 个槽，延迟升、吞吐零增益（subagent 确认：`24→40` 但 trials 不动 = actor pool 仍 96 槽，group 序列化更严重）。
- retry budget `_max_rollout_group_attempts = max(batch×mult, groups×mult)`（`rollout.py:614-617`），`mult=12`：只要 `max_concurrent_groups < rollout_batch_size(64)`，budget 不变（`max(64×12, 48×12)=768`）。

### 三道资源墙（实测都安全）

1. **SGLang 推理**：当前峰值 KV 占用 0.23，翻倍并发预计 ~0.46，**远低于 0.75 上限**。注意：之前担心的"4× SGLang KV 压力"是 n=4 **组内** n 的事，不是这里的并发——并发翻倍只是让更多**不同** prompt 同时推理，引擎本来就闲。
2. **Daytona sandbox**：当前稳态 ≤96。tier4 历史验证过 320 trials（`fully1530` 配置 320×10GiB=3200GiB < 5000GiB 配额）。但 **>256 会进 `nofile=320` 死锁区**（见 `daytona-320-trials-needs-high-nofile-ulimit` memory / §3），所以提并发**别超 ~192–256**。
3. **训练计算**：step0 critic ~28min，满 actor+critic step 量级 30–50min。提并发后 rollout 产率上来，稳态每步会向训练计算下限收敛——但以 27B 中难度 agent 的 rollout 速度看，rollout 大概率**仍是**瓶颈，实际稳态每步可能落在 1–1.5h（而非现在的 1.5–2h）。

### 提并发方案（三旋钮一起翻倍，留足余量）

```yaml
harbor_fully_async_max_concurrent_groups: 24 → 48
harbor_num_concurrent_trials:             96 → 192   # = 48 × 4
harbor_trials_per_node:                   12 → 24    # = 192 / 8
```
- 三者仍精确咬合：`48×4 = 192 = 24×8`。
- SGLang KV ~0.46、Daytona 192（« 320 死锁线、« 配额）、retry budget 不变（48<64）——全部安全。
- 预期：rollout 产率翻倍 → 凑满 64 组时间砍半 → 每步 wall 接近训练计算下限。

### 两个重要提醒

1. **改并发必须重启 run**：config 在 starter 里 materialize（`start_*.sh` 用 `sed` 把 yaml 拷进 `BASE_FOLDER/config/`），**在线改 yaml 不生效**。
2. **坏镜像独立浪费产能**：rollout 1 里 ~15% 的 trial 报 `BUILD_FAILED: pull access denied`，缺失镜像至少 4 个（`terminalworld-env-{124494,237304,441243,474733}`）—— Daytona 拉不到这些任务指定的 Docker 镜像（不是账号/KEY 问题，其他任务正常建箱）。失败 trial 不污染训练（无 reward 被丢），但白占用 producer 容量。重启时最好把这些坏镜像任务从 830 切片里滤掉，与提并发一起做。
3. **硬 `ray job stop` 不跑 cleanup 钩子**：停 run 后在飞的 sandbox 会变孤儿（不自愈，靠 auto_stop 90min+auto_delete 兜底要等 ~100min）。手动清：`python3 daytona_cleanup_slime_sandboxes.py --action delete-all-rest`（先 `--dry-run` 确认全是本 run 的 `harbor_managed` 孤儿）。

---

## 5.6 把 context window 拉到 128k（rollout 真生成 128k + 长样本不截断进训练，2026-06-24）

目标（用户指定）：**rollout 时模型有真·128k 上下文（而非 64k），这些 128k 长样本不做 truncate，直接进训练。** Qwen3.5-27B 的 HF config `max_position_embeddings=262144`、`rope_theta=1e7`，128k 原生支持，**无需任何 rope scaling**——瓶颈纯粹是我们配置里的若干 64k 墙 + 训练激活显存。

### 链路上有 5 道独立的 64k 墙（缺一个，128k 就在某处被悄悄截断/被拒）

从 agent 生成 → SGLang serving → 训练消费，逐段排查到的全部上限（按位置）：

| # | 设置 | 旧值 | 新值 | 性质 / 不改的后果 | 在哪改 |
|---|---|---|---|---|---|
| 1 | `ROLLOUT_MAX_CONTEXT_LEN` | 65536 | 131072 | slime 侧单次 LLM 调用 prompt+gen 上限（`harbor/llm.py:104` 主动拦截，抛 `ContextLengthExceededError`） | wrapper env 默认值 |
| 2 | `SGLANG_CONTEXT_LENGTH` | 65536 | 131072 | **SGLang 引擎 KV 窗口**。不抬，引擎直接 HTTP 400 拒掉 >64k 请求 | wrapper env 默认值 |
| 3 | `SGLANG_MAX_PREFILL_TOKENS` | 65536 | 131072 | 单次 prefill token 上限。不抬，长 prompt prefill 被拒 | wrapper env 默认值 |
| 4 | `max_context_tokens`（**YAML**） | 64000 | 128000 | **最隐蔽**。Terminus2 agent 自己的上下文预算（`agent.llm_kwargs`），到这个值就摘要/unwind，**agent 永远不让 prompt 超过它**。不改这个，上面 3 个抬了也白搭——agent 先把历史压缩了，rollout 永远到不了 128k | **YAML**（不是 env！starter 用 sed materialize，改模板即生效） |
| 5 | `HARBOR_MAX_TRAIN_TOTAL_TOKENS` | 65536 | 131072 | **训练侧硬截断**。`sample_postprocess.py:_hard_trim_plan` 把 `total_length > cap` 的样本从尾部砍到 `cap` 再进训练。不抬，**128k 样本被砍回 64k 进训练**——正是要消除的 truncate | wrapper env 默认值 |

外加一个训练侧上限（不是"墙"，但 >它会报错）：

| 设置 | 旧值 | 新值 | 性质 |
|---|---|---|---|
| `SEQ_LENGTH`（= `--seq-length` / `--max-position-embeddings`） | 65536 | 131072 | 训练侧合法序列上限。128k 样本 >它会越界报错。HF config 262144 → 131072 合法。**不增加激活显存**（RoPE 即时算，不是查表；激活由实际 token 数定，不由 seq_length 定） |

> 关键确认：harbor 侧**没有**别的隐藏截断/丢长样本逻辑。`atif_adapter` 只如实记录长度；`filter_long_prompt`（`slime/utils/data.py`）是 base `DataSource` 的逻辑，harbor 用自己的 `HarborTaskDataSource`，不走那条。所以抬完上面 6 项，长样本能**完整**进训练。

### 训练激活显存：方案 B（不动并行布局）

`trajectory_retokenized` 模式下 **一个 Sample = 一整条多 turn 轨迹重新 tokenize**，所以单条训练序列可以真到 128k——这正是"长数据不截断进训练"的含义，也意味着 train 侧会真遇到 128k 单序列，**显存方案是必须的，不是可选**。

- per-rank 激活 token ≈ `MAX_TOKENS_PER_GPU`：dynamic-batch 把每个 mbs 填到 `max_tokens_per_gpu × cp_size`，CP 再切成 cp_size 份（`dp_schedule.py:213`）。单条序列不能拆 microbatch，必须整条 forward。
- 64k 验证过的安全点：`MAX_TOKENS_PER_GPU=32768`（= 64k / CP2 = 32k/rank）。
- 真·128k 单序列 / CP2 = **64k/rank = 2× 激活**。
- **方案 B（先试）**：`MAX_TOKENS_PER_GPU 32768→65536`，**不动 CP/PP/ckpt**，per-rank 吃 64k，靠 `--recompute-granularity full`（全重算）省激活，**实测能不能过**。
- **方案 A（B 爆了再上）**：`CP2→CP4 + PP8→PP4`（保持 32 actor GPU），128k/CP4 = 32k/rank 回到验证点。代价：① PP4 每 stage 16 层、每卡权重翻倍，需确认 actor+critic 两个 27B 装得下 96GB H20；② torch_dist ckpt 名写死 `_pp8_`，需先验能否 reshard 到 PP4，否则要重 stage 一份 PP4 ckpt。

### SGLang 长上下文 serving

- `SGLANG_CHUNKED_PREFILL_SIZE=8192` 保持开：128k prefill 一次性算占大显存峰值，分块削峰。
- `SGLANG_MEM_FRACTION_STATIC=0.75` 维持：Qwen3.5 是 hybrid（GatedDeltaNet 线性注意力为主，KV 不随长度涨），128k KV 涨幅远小于 dense；当前 KV 占用峰值才 0.23，翻倍 ≈0.46 < 0.75，够。

### 改动落地（overfit 载具，2026-06-24）

改在 overfit 这套文件上当 smoke 载具（不新建文件）：`run_tb2_full89_..._ppo_lenreward_overfit_fully_async.sh`（6 项 env 默认值）+ `harbor_tb2_full89_..._ppo_lenreward_overfit.yaml`（`max_context_tokens`）。starter `start_tb2_full89_..._ppo_tmux.sh` 只透传一个子集的 env（含 `MAX_TOKENS_PER_GPU`，但**不**含那几个 context 变量）——所以改 wrapper 默认值是对的：启动 shell 不设这些变量 → 走 wrapper 新默认 131072。

### smoke 验证看点

冷启（无可续训的 actor ckpt——三个 lenreward overfit run 都没到 SAVE_INTERVAL=20；`ppon4_055519/iter_6` 是被 critic 覆盖的坏 ckpt，见 §0/[[ppo-critic-overwrites-actor-ckpt]]）、`NUM_ROLLOUT=5`、不存 ckpt。第一步盯两件事：
1. **agent 真产出 >64k 轨迹了吗**：看 rollout 日志的 `tokens_len` / `response_length`，确认有样本突破旧的 64k 上限（否则任务本身就短，没压到 128k）。
2. **train step 显存峰值**：`nvidia-smi` 看 actor train（step1+，step0 是 critic warmup）是否 OOM。爆 → 退方案 A。

---

## 5.7 128k 并行布局探索史：方案B→A(C-1)→C-2，撞上 CP4×PP 的 NCCL 死锁（2026-06-24）

§5.6 只写到"方案 B（先试）/方案 A（爆了再上）"的计划。这一节记录**实际跑出来的结果**——4 个配置全部翻车，最终定位到一个与显存无关的分布式死锁。这是这条 128k 线目前的 BLOCKER。

### 0) 128k 逼着动并行的数学（所有配置的出发点）

- 单 microbatch token 上限 = `MAX_TOKENS_PER_GPU × CP_SIZE`（`dp_schedule.py:213`）；CP 把序列维切开，每个 CP rank 只算 `L/CP` 个 token 的激活。
- 实测验证过的**安全激活点 = 32k token/rank**（超过贴顶/OOM）。
- 要让 128k 序列 per-rank 落在 32k：`128k / CP = 32k ⟹ CP 必须 = 4`。**这是 128k 的硬约束**，CP2 撑不住（128k/2=64k/rank = 2× 安全点）。
- 硬拓扑约束：`TP × PP × CP = actor GPU`（TP 全程=2）。

### 1) 试过的全部配置（总表）

| # | 配置 | TP×PP×CP | actor GPU | MAX_TOKENS | per-rank激活(128k) | 上下文 | 结果 | 翻车点 |
|---|---|---|---|---|---|---|---|---|
| **基线** | 28-step overfit | 2×8×**2** | 32 | 32768 | — (任务只到44k) | 64k | ✅ 干净训完28步 | 无；但只验证到64k，没碰128k |
| **B** | 方案B（只抬token） | 2×8×**2** | 32 | **65536** | **64k/rank** | 128k | ❌ OOM | step0 critic 97/95GB贴顶，单mbs卡17min，OOM-killer 杀 worker |
| **A=C-1** | 方案A | 2×**4**×**4** | 32 | **16384** | 32k/rank | 128k | ❌ NCCL死锁 | 显存过了；PP4×CP4 死锁 |
| **C-2** | C-2 | 2×**8**×**4** | 64 | 32768 | 32k/rank | 128k | ❌ NCCL死锁 | 显存+rollout都过了；PP8×CP4 同样死锁 |

三维度各管什么：**TP** 切单层 matmul（摊薄单层权重+激活）；**PP** 切 64 层成 stage（PP8=8层/stage，PP4=16层/stage，摊薄每卡权重）；**CP** 用 ring attention 切序列（摊薄长序列激活，但 ring 内部要 P2P send/recv 传 KV）。

### 2) 逐配置翻车点

- **基线（CP2/PP8/64k）✅**：唯一干净训完的（tb2 89-task overfit 28步，eval 27.5%→38.6%）。但 overfit 任务最长样本才 44k，**从没真压到 64k 单序列**，更没碰 128k。这是后面所有判断的"已知好点"。
- **方案B（CP2不动，MAX_TOKENS→65536）❌ 纯显存**：CP2 下 128k per-rank = 64k = 2× 安全点。叠加 PPO 的 actor+critic 两个 27B 时间共享同批 GPU，step0 critic forward **单 microbatch 卡 17min**（显存边缘 GC 死亡螺旋），峰值 97133/97871 MiB 贴顶，worker 被 OOM-killer 杀。连最吃显存的 actor backward 都没到。→ 结论：CP2 装不下 128k，必须 CP4。
- **C-1（CP4+PP4，32GPU，MAX_TOKENS 16384）❌ NCCL死锁**：CP2→CP4 解决显存，但 TP2×PP4×CP4=32 强迫 **PP4**（16层/stage，权重翻倍），MAX_TOKENS 降 16384 补偿（16384×16层 ≈ 验证点 32768×8层）。**显存这关过了**（critic forward microbatch 真推进 1/104→7/104，278s→13s，actor 稳 79-93GB 无 OOM——真进展）。**但 critic forward 推进到 7/104 后死锁**：node2 全8 rank 在子进程组 **PG21** 上一个**空 COALESCED collective(Numel=0)** 10min watchdog 抛 fatal；node1 ranks 100%util/平直113W 自旋等已死 rank。当时判断"是 PP4 specifically 的问题，扩 64GPU 用 PP8 能避开"——**这个判断被 C-2 证伪**。
- **C-2（CP4+PP8，64GPU，MAX_TOKENS 32768）❌ 同样NCCL死锁**：actor 扩 64GPU 让 TP2×PP8×CP4=64，回 PP8（8层/stage），MAX_TOKENS 恢复验证值 32768。rollout 缩 32GPU=16引擎。**显存过了 + rollout 也过了**（rollout 0 健康填满 64/64 ~88min，16引擎够用零泄漏，update_weights 成功）。**但 step0 critic forward 一进去就死锁，和 C-1 同病**：
  - 末 PP stage（137.141, ranks56-63）：`critic forward 0/59` 冻死 13min+，**util=0%**（真死等）。
  - 前 PP stage（136.111, ranks0-7）：**100%util + 功耗6+次采样平直钉113-118W**（真 compute 震荡 300W+，平直=NCCL自旋）。
  - **冒烟枪**：`An unbatched P2P op (send/recv) was called on this ProcessGroup with size 4 ... a new 2-rank NCCL communicator to be created`。**size 4 = CP4 组**。

### 3) 死锁机制分析（结合冒烟枪）

CP 用 **ring attention**——每个 CP rank 把自己那段 KV **P2P send/recv 给环上邻居**，CP4=4 rank 成环，邻居间建多个 **2-rank NCCL communicator**。这些 comm 是**惰性创建**（第一次用到才建），而惰性创建是个 **init-time 集合操作，要求该组所有 rank 同时进入**。在 **PP 流水 schedule** 下，不同 PP stage 的 rank 到达"第一次 CP P2P"的时机**不对称**（有的 stage 还在等上游），于是 comm 创建在部分 rank enqueue、部分没 enqueue → **永久挂死**。

**为何 C-2 watchdog 没像 C-1 抛 fatal**：惰性 comm-init 不是被 watchdog 计时的 `WorkNCCL` 对象，不触发 600000ms 超时，会**无限静默挂**。（C-1 抛 fatal 是因为它卡在已建好 comm 的空 COALESCED collective 上，那个被计时。两者同病的不同触发点。）

### 4) 判死锁 vs 慢算的金标准（这条线踩过坑，方法很重要）

- **功耗**是金标准：真 27B matmul forward/backward 功耗**震荡到 300W+**；NCCL 自旋是 100%util 但功耗**平直钉在 ~110W（TDP 23%）+ 满 SM 时钟**。
- **绝不能只看 `memory.used`**（caching allocator 下恒定，不能判活），**也别只看 util**（自旋也是 100%）。
- 三者结合：**多轮采样功耗 + 日志 mtime 是否推进 + watchdog 是否出现**。
- 注意末 PP stage util=0% 是**正常 pipeline-fill**（要等 microbatch 0 流过前面 stage），不是它自己挂——要看**前 PP stage 是否在 100%util 平直功耗自旋**才判死锁。

### 5) 已坐实 vs 待确认

**坐实**：① CP2 已知好、CP4 已知坏（C-1 PP4 死、C-2 PP8 死），变量从好→坏就是 CP2→CP4；② 加深/改 PP 解决不了（PP4、PP8 在 CP4 下都死，当初"PP4 specifically"错了）；③ 显存和 rollout 两个老问题 C-2 都解决了，**唯一 BLOCKER 就是 CP4 的分布式死锁**。

**待确认**（下一步最小验证要回答的）：① size=4 子组**确实是 CP 组**吗？（强推断，但还没读 Megatron 建组代码逐行确认，不排除别的 size-4 子组）；② 为什么 CP2 不死、CP4 死？（假说：CP2 ring 只单个 P2P 对、创建顺序无从发散；CP4 多个 2-rank comm 在 PP schedule 下创建顺序才发散——**假说未验证**）；③ 是不是"CP4 本身"还是"CP4×PP 交互"？

### 6) 下一步候选

1. **最对症：预先 eager-init 所有 CP 通信组**，别让它在 PP schedule 中途惰性创建。若死锁真是"惰性创建顺序发散"，训练前 warmup 一遍 CP ring 的 P2P 强制建好所有 2-rank comm，死锁路径就消失，**CP4 可保留=128k 显存方案不动**。改动小最精准。
2. **放弃 CP，纯 TP+PP 扛 128k**（如 TP8×PP8=64GPU 无CP）。128k 激活靠 TP8 切 attention。彻底绕开 CP×PP 路径。代价：TP8 通信开销大，需重调 MAX_TOKENS，且 TP8 不一定切得动 GQA 的 KV head 数。
3. **【最小验证，先做】CP4 去掉 PP**（如纯 CP4 无PP，或 CP4+PP2）的 tiny smoke，确认"去掉/降低 PP 后 CP4 是否还死"——直接证伪/坐实"是 CP×PP 交互"，成本低信息量大。**这决定走候选1（eager-init保CP4）还是候选2（弃CP）。**
4. **退而求其次：CP2+64k**（已知稳），若 128k 非死要求。

> 详细记录见 memory：[[ctx128k-planB-oom-need-planA]]、[[ctx128k-c1-passed-mem-but-pp4cp4-nccl-deadlock]]、[[ctx128k-c2-pp8cp4-same-deadlock-cp4-is-root]]。

## 5.8 死锁真根因（py-spy 坐实）+ NVTE 修复失败 + 关 CP 的显存账（2026-06-25）

§5.7 的"CP4 ring-attention P2P 惰性 comm-init"假说**被 py-spy 调用栈推翻**。这一节记录：① NVTE 修复为什么没用；② py-spy 抓到的真实死锁位置；③ 这个混合模型为什么 CP 不能简单关掉；④ 实际可行的 128k 路径。

### 1) NVTE_BATCH_MHA_P2P_COMM=1 验证失败（run 20260625_020445，C-2 同配置 + 唯一改 NVTE=1）

- 之前读 TE 源码定位到 `context_parallel.py:flash_attn_p2p_communicate`：cp_size==4 默认走 unbatched isend/irecv（裸 P2P → 惰性 2-rank NCCL comm），cp_size==2 才自动 batched。判断设 `NVTE_BATCH_MHA_P2P_COMM=1` 强制 batched 就解。
- **smoke 跑了，结果仍死在 step0 critic forward 0/8，功耗 signature 与 C-2 一模一样**（stage0=100%util平直114W自旋、末stage=0%util冻，冻 10min+ tqdm 零推进）。
- **关键：修复确实生效了，但打错靶子**：① `NVTE_BATCH_MHA_P2P_COMM=1` 在 `ray::MegatronTrainRayActor` 进程 env 确认 live；② C-2 那条 `unbatched P2P ... lazy ... 2-rank NCCL communicator` 警告**消失了**（证明 batched 路径被走了，TE 那层修复起作用了）；③ **但还是死** → 死锁根本不在 TE flash-attn P2P 那条路径。

### 2) py-spy 调用栈坐实真根因（铁证，推翻 §5.7 的 ring-attention 假说）

dump 8 个 actor 节点全部训练 rank（2026-06-25 03:04-03:09），死锁几何：
- **PP stage 0（136.111 ranks0-7）**：全卡 `send_forward → _communicate`（`megatron/core/pipeline_parallel/p2p_communication.py:398` req.wait），100%util 平直 114W 自旋。
- **中间 stage（137.6, 136.164）**：**卡在模型 forward 内部** —— `slime_plugins/models/hf_attention.py:147`（hf_forward）+ `fla/ops/gated_delta_rule/chunk.py:169`，**0%util**。forward 没算完就阻塞。
- **中间 stage（137.78, 136.210）**：卡 `_communicate_shapes`（PP 形状握手 :236/:242，batch_isend_irecv）。
- **末 stage（137.141 ranks56-63）**：卡 `recv_forward → _communicate_shapes → batch_isend_irecv → _coalescing_manager`（`torch/distributed/distributed_c10d.py:2785`）—— 就是 C-1 watchdog 抛 fatal 的那个空 COALESCED collective。

**同一个 PP stage 内 8 rank（=TP2×CP4）劈成两半**：一部分已过 all_gather 在算 GDN kernel，一部分还卡在 all_gather 前 —— 这是 collective desync 的定义。

### 3) 真根因：GDN linear-attn 层的自定义 `all_gather` 在 PP 流水线下 rank 不同步（不是 TE，不是 ring attention）

**模型结构**：Qwen3.5-27B 是混合注意力，64 层里每第 4 层（idx 3,7…63，共16层）是 full-attention，其余 48 层是 GDN linear-attention。两类层走**完全不同的 CP 通信**：
- **full-attn 层** → stock TE self-attn → TE flash-attn P2P（NVTE flag 管的那条，**与死锁无关**）。
- **GDN 层（48/64）** → `qwen3_5.Attention(HuggingfaceAttention)` → `hf_attention.py:117-143` 自己写的 CP 路径：在 `cp_group` 上做一次**对称阻塞 `all_gather`**（`_AllGatherForDuplicatedComputation`，:121），把 CP 切开的序列拼回完整、复制计算、再切片。GDN kernel 本身**不做任何 collective**（grep 证实，纯本地 recurrent）。

**为什么 CP2 不死、CP4 死**：`all_gather` 是阻塞 collective，要求 cp_group 全员同时进入。每个 GDN 层都要 cp_group 到齐才能过。PP 流水线里不同 microbatch 在不同 stage 流动，CP2 时同步窗口小易对齐；CP4 时 cp_group 翻倍 + PP8 的 16 个 stage-edge P2P 交错，all_gather 进入时机在 critic forward warmup 第一个 microbatch 就错开 → 死锁。**与 TE batched/unbatched 完全无关** —— C-2 那条 `size 4 unbatched P2P` 警告只是 full-attn 层的噪声，真正挂住的是 GDN 层的 all_gather。

### 4) 关键显存账：为什么不能简单"关 CP"（CP1 跑 128k 必 OOM）

死锁来自 CP>1，自然想关 CP。但**这个模型里 CP 是唯一能沿序列维切单层激活的手段**，关掉会 OOM，理由是代码事实：
- **单 microbatch 单卡 token 上限 = `MAX_TOKENS_PER_GPU × CP`**（`dp_schedule.py:213`）。CP1 后要塞 128k microbatch，单 rank 实算**完整 128k 序列激活**，CP 切序列的省显存全没了。
- **GDN 层不吃 TP**：GDN 投影全是裸 `nn.Linear`（`qwen3_5.py:65-84`），每个 TP rank 完整复制。TP4 只切 full-attn 的 16 层 + SP 的层间 hidden，**切不了 GDN 的单层激活**。GDN `conv_dim=10240` 比 full-attn QKV 还大，是激活大头。
- **单 rank 实算序列 = 激活硬墙**（历史实测对照）：CP4=32k/rank ✅不OOM｜CP2(PlanB)=64k/rank ❌OOM(97/95GB贴顶)｜**CP1=128k/rank ❌❌必OOM**（单层激活是CP4的4倍，PP加深救不了单层峰值）。
- TP8 救不了：kv_heads=4 不能被 8 整除（实测 num_key_value_heads=4 %8≠0），full-attn 层会挂。TP4 全维度可整除但仍切不动 GDN。

### 5) 实际可行的 128k 路径（按推荐度）

| 路 | 拓扑(=actor GPU) | 解死锁 | 放下128k | 风险 |
|---|---|---|---|---|
| **B（先试）** | **TP2×PP16×CP1=32** | CP1 无 GDN all_gather | PP16 每stage 4层，GDN激活/权重峰值减半 + recompute | PP16 气泡大需≥16 microbatch；128k 单层激活仍是 PlanB(64k) 的2倍，**可能仍 OOM/擦边**——只能实测 |
| A | TP4×PP8×CP1=32 | CP1 无死锁 | 配置最简最稳 | 单rank实算128k，比B更易OOM（TP切不动GDN）|
| C（最干净但难） | 保 CP4 + 修 GDN all_gather × PP desync | 真根因修复 | CP4=32k/rank 显存确定够 | 要改 slime 自定义 CP forward 或 eager-init/序列化通信，成本高 |

**当前决定（用户指定）：先起 B = TP2×PP16×CP1，MAX_TOKENS_PER_GPU=131072（单 microbatch 塞整条 128k），smoke 验证 step0 critic forward 是否 OOM。**
- 不 OOM → CP1 既无死锁又放得下，跑正式 overfit。
- OOM → CP1 在 128k 死路，转攻路 C（保 CP4 显存优势 + 修 GDN all_gather × PP 冲突）。

### 6) CP1-B 验证结果（2026-06-25）：不是 OOM、不是死锁，是 GDN forward 在 128k 单 rank 慢到不可用 → 转路 C

run cp1b_20260625_065505。actor 4节点32GPU=TP2×PP16×CP1（DP=1），rollout 64GPU=8节点=32引擎。CP=1 确实绕开了 GDN all_gather × PP 死锁（`hf_attention.py:117 if cp_size>1` 分支不执行）。但 **step0 critic forward 卡在 0/9 共 ~21min，microbatch 0 一次 forward 都没走完** → 判非可用，停 run。

**py-spy 全 actor 节点诊断（区分 OOM / 死锁 / 慢算，金标准）：**
- 中段 PP stage（136.210）：active rank **真的在 GDN forward 内部逐 kernel 前进**——多轮采样依次 `causal_conv1d_fwd` → `l2norm_fwd` → `layer_norm_gated_fwd`，**栈在变 = 没死锁没挂**，只是极慢。该节点 util=0%、显存仅 11-27GB（GDN kernel 内存带宽瓶颈，不饱和 SM）。
- PP stage0（136.111）：8 rank 全卡 `send_forward → _communicate → cuda.synchronize()`，util=100% 自旋等下游，显存 86-92GB（驻留激活，波动非单调贴顶）。
- 末段：全卡 `recv_forward → _communicate_shapes → _coalescing_manager`（PP P2P 等永远不来的 microbatch）。
- margin-OOM 警告（§4 良性那种）只累计 7→9 条，**不是 PlanB 那种疯狂刷的死亡螺旋**。显存 92GB < 97GB 上限，**没真 OOM**。

**根因**：CP1 把完整 128k 序列压到单 rank 的 GDN linear-attn kernel（conv1d/l2norm/chunked-delta/gated-norm，全 `fla` Triton kernel）。这些 kernel 在 128k 单序列下**慢到病态**（CP4 时每 rank 只 32k 就正常）。不是显存爆，是**算力/带宽**：一个 microbatch forward 21min 没走完 PP16 一遍 → 正式训练（每 step N microbatch + backward）完全不可行。GDN 投影是裸 nn.Linear、TP 切不动，PP16 摊层也救不了单 rank 单序列的 GDN kernel 耗时。

**结论：CP1 这条路（不管 PP 多深）对 128k 死路** —— 不是放不下（显存没爆），是 GDN forward 在 128k 单 rank 太慢。CP 既是死锁源、又是 128k 性能的必需品（切序列让每 rank 只算 L/CP）。**下一步唯一出路 = 回 CP4（性能+显存都验证过 OK，每 rank 32k）+ 修 GDN all_gather × PP desync 死锁本身（路 C）。** 修法候选：把 `_AllGatherForDuplicatedComputation`（hf_attention.py:121 阻塞 all_gather）换成不依赖全 CP 同步进入的实现，或在 PP schedule 里对齐 CP all_gather 的进入点。

> 详细记录见 memory：[[ctx128k-c2-pp8cp4-same-deadlock-cp4-is-root]]（已更新 py-spy 推翻 + 显存账 + CP1-B 失败）。

---

## 5.9 方案 A：eager-init 所有并行组 NCCL communicator，从根上修 CP4×PP 死锁（2026-06-25）

§5.8 的结论是"回 CP4 + 修 GDN all_gather × PP desync 死锁"。这一节记录方案 A 的实现与依据。另外 GitHub issue [THUDM/slime#1871](https://github.com/THUDM/slime/issues/1871) 提供了两个关键旁证（见下）。

### 0) issue #1871 澄清的两件事
1. **我们走的不是 `--allgather-cp`**。slime 有两条 CP 路径：默认 **zigzag CP**（`allgather_cp=False`，每 rank 拿交错 chunk，普通模型含 Qwen3.5 用这条）和 **`--allgather-cp`**（每 rank 拿连续切片，只给 DSA 模型 DeepSeek-V3.2/GLM-5）。`grep allgather` 我们的 wrapper 0 命中 = 走 zigzag。issue #1871 的 bug（Qwen3.5 误开 allgather-cp → token 顺序乱）和我们**无关**；修复 #1891 只是加守卫 `arguments.py:28 _validate_allgather_cp_supported` 把这个误配置拦死。"Qwen3.5 + allgather-cp 不兼容" ≠ "Qwen3.5 不能用 CP"——zigzag CP4 在我们这儿显存+数值一直 OK，只是死锁。
2. **真·180k Qwen3.5 是能跑的**。issue 作者明确说他们把 CP 推进 FLA kernel（`fla.ops.cp.build_cp_context`）消掉 outer all_gather，在 304×A800 上跑通 Qwen3.5-122B 真·180k、28 rollouts、零 NCCL/Triton 错。这逐字印证我们 py-spy 的诊断：*the outer all-gather is the dominant source of activation blow-up at long context* = 就是 `hf_attention.py:121` 的 `_AllGatherForDuplicatedComputation`。**但我们装的 `fla 0.4.1` 没有 `fla.ops.cp`**（已探，`ModuleNotFoundError`），那是作者自己 fork 的未上游 prototype → kernel-CP 是方案 B（后手），不是现在能直接用的。

### 1) `hf_attention.py:117-162` 在干什么（CP4 下 GDN 的 CP 路径）
GDN 是 recurrent 扫描，序列维不能像 flash-attn 那样切并行。所以策略是 **all_gather 全序列 → 每 rank 冗余跑完整 L 的 GDN → 切回自己那份**：
- `:121` `all_gather(hidden_states)` 收齐所有 CP rank 分片；
- `:127-143` 拼回完整全局序列（un-zigzag）；
- `:147` `output = hf_forward(完整 L)` ← GDN 在**完整 L** 上跑；
- `:152-162` output 切回本 rank 的 zigzag 分片。

**致命含义**：GDN 的算力 per-rank 在 CP1 和 CP4 **相同**（都算完整 L），CP 只切 all_gather/残差/full-attn/MLP，**不切 GDN 算力**。所以 CP1-B 看到的"GDN 在 128k 慢"理论上 CP4 也存在。**CP4 仍可能比 CP1 快的唯一理由**：CP1 显存贴顶 92/97GB → allocator 抖动 + margin-OOM，throttle 了 GDN kernel；CP4 把非 GDN 激活切 4×（32k/rank）→ 显存压力大降 → GDN kernel 可能跑得开。28-step overfit（CP2@64k，GDN 算完整 64k/rank）没事 → CP4@128k 是 2× 那个 GDN 量但显存更松。这是方案 A smoke 要同时回答的第 2 个问题。

### 2) 死锁机制（为什么 `:121` 的 all_gather 在 PP 下挂）
Qwen3.5 layer 0/1/2 都是 GDN（full-attn 是每第 4 层 3,7,…），所以**前向第一个 attention 层 = GDN = `:121` 的 all_gather = 整个训练第一个 CP 组集合通信**。py-spy（C-2，§5.8 节 2）坐实：中段 stage 卡 forward 内部 0%util、上游 PP send 自旋、末段 PP recv 等。本质是 **CP 子组(GDN all_gather)和 PP 子组(P2P)的 NCCL communicator 惰性初始化跨组互锁**——惰性 init 本身是阻塞集合操作，需全组同序进入；1F1B 调度下各 rank 首次触碰不同子组的顺序不一致 → 创建阶段死等。惰性 init 不是被计时的 WorkNCCL → watchdog 不抛 fatal → 无限挂（解释为什么不像 C-1 那样 10min 报错）。

### 3) 方案 A 实现：`actor.py:_eager_init_parallel_comms`
在 `init()` 里、`initialize_model_and_optimizer` 之后、`role=="critic"` 提前 return **之前**调用（actor+critic 都预热；死锁正是 step0 critic forward）。此时所有 rank 在 init 阶段、还没进 PP 调度 = 天然 lockstep 点。按**固定 rank-uniform 顺序** CP→TP→DP→PP，对每个 world>1 的组做：① 一次 dummy `all_reduce`（建 collective communicator）② 一个 isend(next)/irecv(prev) 的 P2P 环（建 Megatron `batch_isend_irecv` 用的 P2P comm）。建好后训练循环里的真 all_gather/batch_isend_irecv 复用已建好的 comm，不再惰性 init → 死锁消除。
- **env 门控** `SLIME_EAGER_INIT_PARALLEL_COMMS`（默认 0，非 CP run 零影响）。已加进 base `RUNTIME_ENV_JSON`（:698，否则到不了 train-actor 进程，这是 NVTE 修复踩过的坑）。wrapper 默认 1。
- **为什么不会自己死锁**：各 phase 每个 rank 都对自己组做 collective，组划分世界（partition），末尾 gloo barrier 把出口对齐。

### 4) 方案 A smoke 配置（回 C-2 拓扑 + 开 eager-init）
wrapper 从 CP1-B 改回：`TP2×PP8×CP4=64`（actor 8 节点）、`ROLLOUT_NUM_GPUS=32`（4 节点=16 引擎）、`MAX_TOKENS_PER_GPU=32768`（×CP4=131072=真 128k microbatch cap，非 GDN 激活 32k/rank=验证安全点）、`SLIME_EAGER_INIT_PARALLEL_COMMS=1`。yaml 并发 32 不变（`trials_per_node 8 × 4 nodes`）。smoke 参数同前（batch8/n4/NUM_ROLLOUT2/不存ckpt）。

**判据**：① launcher 日志出现 `[eager-init] all parallel-group communicators pre-created` = 预热跑到；② step0 critic forward **推进**（tqdm 0/N→N/N，不再冻 0/N）+ step1 actor forward+backward 过 = **死锁解了**；③ 同时量 critic forward 耗时——若可接受 = CP4@128k 全链路通（白捡，改回 batch64/256 跑正式 overfit）；若 forward 仍慢到病态（类 CP1-B）= 死锁解了但撞 GDN 算力墙 → 坐实必须上方案 B（kernel-CP，啃 `fla.ops.cp`）。

### 5) 改了哪些文件（方案 A）
- `slime/backends/megatron_utils/actor.py`：新增 `_eager_init_parallel_comms()` 方法 + 在 `init()` 里 `train_parallel_config` 之后、`start_rollout_id` 之前调用。**唯一的 core 改动**，env 门控默认关。
- `examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh`：`RUNTIME_ENV_JSON` 加 `SLIME_EAGER_INIT_PARALLEL_COMMS`（默认 0）。
- `examples/harbor_terminal_rl/run_tb2_full89_96gpu_27b_daytona_ppo_lenreward_overfit_fully_async.sh`：CP1→CP4、PP16→PP8、MAX_TOKENS 131072→32768、节点 4/64→8/32、加 `SLIME_EAGER_INIT_PARALLEL_COMMS=1`。
- `examples/harbor_terminal_rl/harbor_tb2_full89_96gpu_27b_daytona_ppo_lenreward_overfit.yaml`：`trials_per_node 4→8`（rollout 回 4 节点，硬约束 32==8×4）。

### 6) smoke 实测结果（planA_20260625_094957）：❌ 部分解锁，最终仍崩在一个未覆盖的 CP 子组
- ✅ `[eager-init] warmed context_parallel(world=4)/tensor(2)/pipeline(8) → all pre-created`，actor+critic 都跑到，4 秒完成不卡。
- ✅ **比 C-1/C-2 走得远**：step0 critic forward 进入并执行了——首个 microbatch 经历一次性 Triton JIT autotune+make_cubin（~5min，ptxas 子进程在churn、triton cache 持续增长），然后某 PP stage(137.188) 整机 6/8 GPU @ 420-455W+92%util = 真·27B GDN compute（power 爬升=compute 不是 NCCL 平直自旋），HOT stage 还会迁移=流水线在推进。C-1/C-2/NVTE-smoke 全都没活到「kernel 真跑」这步。
- ❌ **但 11:08:43 job exited code 1**：actor(136.210) NCCL watchdog 抛 fatal。铁证：`WorkNCCL(SeqNum=1, OpType=COALESCED, NumelIn=0, NumelOut=0, Timeout=600000) ran 710034ms`，`PG ID 20 PG GUID 575`，`last completed work: -1`（该 PG 第一个 collective 从未完成）。**这是 C-1 的空-COALESCED 签名**（`_coalescing_manager` 上零工作量集合操作，某子组首次触碰惰性 init 挂死，10min watchdog 抛 fatal）。
- **为什么 eager-init 没拦住**：我只 warmup 了 `mpu.get_context_parallel_group()`/TP/DP/PP 这 4 个**主组**。挂死的是 **PG ID 20 / GUID 575**——一个我没 warmup 的**额外子组**（疑为 `_coalescing_manager` 的 per-stage CP 子组 / TP×CP 组合 / hierarchical-CP 子组）。它的第一个 collective 仍惰性 init 在 schedule 中途触发 → 挂。
- eager-init 改动让 run 走得比以往都远，但**不充分**——覆盖不全。

### 7) 最终决定（2026-06-25）：放弃「继续修死锁」这条路，因为修好了也撞算力墙

复盘后判定**不再投入修死锁**（无论是补全 warmup 还是全局 eager-comm-init），理由是两层：

**第一层——继续修死锁本身胜率低且方向不明。** 崩的是 `Numel=0 空 COALESCED`，无法区分它是「惰性 init 跨组互锁」（eager-init 类手段能修）还是「真 desync——部分 rank 根本没到这个 collective」（修不了）。而且 PyTorch 的 `device_id` eager-init 只覆盖默认组 + `new_group()` 建的组；若 PG 575 是运行时按需建的子组则覆盖不到。连它是哪个组都还没定位。低胜率赌。

**第二层（决定性）——就算死锁完美修好，128k 立刻撞 GDN 算力墙，还是跑不动。** 这是代码级确定性，不是猜测：
- `hf_attention.py:117-147` 的 CP 路径是 `:121 all_gather 全序列 → :127 拼回完整 L → :147 hf_forward(完整 L)`。**CP 根本不切 GDN 算力**，每个 rank 的 GDN 都算完整序列，CP1 和 CP4 逐字节相同。
- Qwen3.5 **48/64 层是 GDN**（75%）。CP1-B 已实测：128k 单 rank GDN forward 一个 microbatch 跑 21min。CP4 的 GDN 算力与 CP1 相同 → CP4@128k 必然同样慢。
- **死结**：CP 既是死锁来源，又对 128k 提速无效（只省显存，不省 GDN 算力）。死锁和算力墙是**两个独立拦路虎**，修死锁顶多拆第一个，第二个原地不动。

**唯一真能上 128k 的是方案 B（kernel-CP），但我们没有工具。** issue#1871 作者跑通真·180k 靠 `fla.ops.cp.build_cp_context` 把 CP 切进 GDN kernel（每 rank 真只算 L/CP，同时消 all_gather 死锁 + 消冗余计算 = 一石二鸟）。我们 `fla 0.4.1` 没这个模块（`ModuleNotFoundError`），那是作者 fork 的未上游 prototype。

**结论：用现有 fla，128k via CP 这条路本质走不通**（不管死锁修不修）。剩三个现实选择，已交用户定夺：
1. **降到 64k（CP2）** —— 已验证能跑（28-step overfit 就是 CP2，无死锁、无算力墙，每 rank GDN 只算 64k）。零风险、立即可用。
2. **投入方案 B** —— 升级/移植 fla 拿 `fla.ops.cp`。唯一真能上 128k，但工程量大、风险高（fla kernel 是 rollout+训练共用，升级可能动 SGLang 推理）、不保证能正确接上。
3. **128k 线先搁置。**

### 8) 改动落地 + 回滚（2026-06-25）
方案 A 的所有改动在本决定后**回滚**：
- `slime/backends/megatron_utils/actor.py`：删除 `_eager_init_parallel_comms()` 方法 + init 里的调用（恢复到只剩本 session 之前的 bridge 改动）。
- base `run_qwen35_9b_terminal_bench2_64h20.sh`：`RUNTIME_ENV_JSON` 删 `SLIME_EAGER_INIT_PARALLEL_COMMS` 行（保留 NVTE 行）。
- wrapper / yaml（untracked 新文件）：从方案 A 的 CP4 配置改回 CP1-B 状态（git 不追踪，手动恢复）。

eager-init 这段代码的思路本身没错（确实让 run 走得更远、证明覆盖到的组不再挂），但因为「修好也撞算力墙」而不值得保留在主线上。若将来上方案 B（kernel-CP 仍需 CP>1，仍可能要预建通信组），可从本节 §3 的实现重新捡起。

## 5.10 CP2@96k smoke：同款 COALESCED 死锁，推翻"CP2 deadlock-safe"，定位到 GDN 逐 stage 编译时滞（2026-06-25）

§5.8 §4 算出"CP1@128k 必 OOM、CP 不切 GDN 算力"后，§5.9 §7 把 128k 降级为 **CP2@96k**（折中点：96k 比 128k 短 25%，CP2 的非 GDN 激活 48k/rank 介于基线 32k 和方案B 64k 之间；而且 CP2 是"基线验证过不死锁"的拓扑，理论上只是把序列墙从 64k 抬到 96k）。这一节记录 CP2@96k 实跑结果——**又死锁了**，而且推翻了一个一直当作前提的假设。

### 1) 配置 + 结果

| 项 | 值 |
|---|---|
| run_tag | `cp2_96k_20260625_124601`（job raysubmit_afapuXBKdERSAQKC） |
| 拓扑 | TP2×CP2×PP8=32 actor(4节点) + rollout 64GPU(8节点32引擎)；**与 64k 基线 tb2of_raw_20260620 完全同拓扑** |
| 序列墙 | 5 道墙 + SEQ_LENGTH = 98304；max_context_tokens=96000；MAX_TOKENS_PER_GPU=49152（×CP2=98304=96k microbatch 上限） |
| smoke | NUM_ROLLOUT=2（step0=critic warmup, step1=actor train）、batch8/n4/groups8、不存ckpt |
| **结果** | ❌ **NCCL Watchdog COALESCED 死锁**——step0 critic forward 卡死，job exited |

### 2) 死锁铁证（和 C-1/Plan-A 同款空-COALESCED 签名）

- `WorkNCCL(SeqNum=1, OpType=COALESCED, NumelIn=0, NumelOut=0, Timeout(ms)=600000) ran for 691373 milliseconds`，`PG ID 21 GUID 563/565 Rank 2`，14:42:25 抛 fatal。
- critic forward **全程卡 0/7**，从未推进一个 microbatch。
- **不是 OOM**：GPU 峰值 used_GB 仅 **23.89**（远低于 95GB 上限）。48k/rank 非 GDN 激活的显存担心是多余的——run 根本没跑到分配训练激活那一步。

### 3) py-spy 坐实机制：GDN Triton 逐 PP-stage 串行编译，时滞超 600s watchdog

收够 8/8 触发 critic forward 后，逐 rank dump 抓到清晰的"编译级联"：
- **PP stage0（.111 rank 3959766）**：先在 `triton autotuner → do_bench → make_cubin`（GDN kernel 在 96k 新 shape 重新 autotune，0%util+110W 是编译期 CPU/ptxas-bound，**不是 NCCL 自旋**），编完后栈走到 `layer_norm_gated_fwd` → 进 `send_forward`/`synchronize`（100%util 等下游）。
- **此时 node .164 还有 8 个 rank 在 COMPILE + 13 个 ptxas/cicc 进程在跑**——下游 PP stage 还在编它们自己的 GDN kernel。
- 那个空 COALESCED collective（GDN 的 `_AllGatherForDuplicatedComputation` CP all_gather，Numel=0 init-only）**起始于 14:30:53——正好是 stage0 编完进 PP P2P 的瞬间**。下游 stage 还在编译、没到对应的 CP all_gather 入口 → 691s 后超 600s watchdog → fatal。
- 编译是 **per-PP-stage 串行**的，**96k 的 autotune 比 64k 重得多**（shape 更大→benched 的 config 更多→编译更久），所以 **stage 间编译时滞在 96k 超过了 600s watchdog 窗口，在 64k 没超**。

### 4) 关键认知更新：死锁触发是**序列长度相关**，不只是 CP 度数

§5.7–5.8 一直把结论钉在"CP4 是根、CP2 安全"。**CP2@96k 推翻了这个前提**：
- CP2 在 **64k** 安全（基线跑完 28 步），在 **96k** 不安全（同拓扑、只抬序列墙→死锁）。
- 所以触发条件不是"CP≥4"这个离散阈值，而是**编译时滞 vs watchdog 窗口的连续竞争**：序列越长 → GDN autotune 越重 → stage 间时滞越大 → 越容易撞 600s 墙。CP4 只是在更短序列就把时滞推过了线，CP2 要到 96k 才过线。
- 这也解释了为什么 §5.9 的 eager-init"部分解锁、走得更远"——它确实建好了部分 comm，但没解决"编译时滞让晚编完的 rank 迟到"这个真问题。

### 5) 修复 + 验证（2026-06-26）：monkeypatch new_group 注入默认 timeout，✅ 死锁消除、96k 可用

判据指向"一次性编译时滞"（空 Numel=0 collective 起在编译边界），所以抬 watchdog timeout 是对的方向。但**正确的杠杆不是想当然的那个**——查实后踩掉两个坑：

- **`TORCH_NCCL_TIMEOUT` 环境变量在本 torch 2.11 build 根本不被识别**（grep libtorch .so strings：有 `TORCH_NCCL_BLOCKING_WAIT`/`TORCH_NCCL_ASYNC_ERROR_HANDLING` 等，但没有 `TORCH_NCCL_TIMEOUT`）。设了也是静默忽略。
- **`--distributed-timeout-minutes` 也不行**：base launcher 早就传了 60min（`run_qwen35_9b_terminal_bench2_64h20.sh:461`，还是上次同样死法后加的），但崩的子组仍是 `Timeout(ms)=600000`=10min。gloo 单测坐实根因：**`new_group()` 不继承 `init_process_group` 的 timeout**——没显式传 `timeout=` 的子组落到**后端默认**（NCCL=10min）。Megatron 的 60min 只经 `create_group` 传给它自己建的组；死的 GDN CP coalescing 子组（PG 21/GUID 563-569）由别的路径建、没传 timeout → 拿了 10min 默认。

**修法**：在已有的 `slime/utils/reloadable_process_group.py:monkey_patch_torch_dist()` 里给 `new_group` 注入默认 timeout（它本就 wrap 了 `new_group`、在每个 actor 进程于 Megatron 建组前运行，是天然注入点）。要点：
- 注入条件是 **none-or-None**，不是 absent——因为 Megatron `create_group` 在 torch≥2.4 永远把 `timeout` 放进 kwargs（哪怕值是 `None`，`parallel_state.py:228`）；位置参数 `new_group(ranks, timeout, ...)` 也判掉不误伤；显式 60min 保留。
- `reload_process_groups` 里那个裸 `old_new_group(...)` 也一起修（step 间 reload 子组同样会落默认 10min）。
- 默认 30min，可用 `SLIME_NEW_GROUP_TIMEOUT_MINUTES` 调（0 关闭）。

**验证**（debug-train-only，见 §6）：job raysubmit_VEswz7RPaU4jB4JW，复用崩掉 run dump 的 0.pt/1.pt 跳过 66min rollout，4 actor 节点/32 GPU。结果：
- 32 ranks 全部 `Applying monkey patch to torch.distributed` ✓；step 间 `Reloading 103 process groups`（reload 修复路径生效）。
- **step0 critic forward 13/13 @19.79s/mbs + critic train(backward) 13/13 @37.88s/mbs；step1 critic forward 10/10 + critic train 全程推进。**
- **零 watchdog、零 COALESCED、零 CUDA-OOM**（GPU 峰值 23.78GB）。功耗 130↔367W 震荡 + hot stage 迁移 = 真 27B compute 流水推进，不是 110W 平直自旋。
- 稳态 ~20-38s/microbatch —— **不是 CP1-B 那种 10-21min/microbatch 的算力墙**。所以 §5.8 §4 担心的"96k GDN 算力墙"在 96k **不成立，96k 可用**。

**结论**：杀掉 C-1/C-2/Plan-A/CP2@96k 全部 CP>1 长上下文尝试的那个死锁，本质是**一次性启动编译时滞撞 10min 默认 watchdog**，不是真·分布式死锁，也不是算力墙。一个 ~15 行的 new_group timeout 注入就解了。

### 6) ★ 用 `--debug-train-only` + dump 的 rollout 数据快迭代死锁修复（关键工具，已验证）

死锁出在**训练 forward**，和 rollout 无关，但每次复现都要先等 **66min 的 TB2 rollout**（Daytona 长任务）才到 critic forward——迭代 watchdog/编译修复极其昂贵。slime 自带的 debug 通道正好把这段砍掉：

**机制（代码已查实）**：
- base launcher **永远**传 `--save-debug-rollout-data ${BASE_FOLDER}/rollout_dumps/{rollout_id}.pt`（`run_qwen35_9b_terminal_bench2_64h20.sh:422`），所以每个 run 在 train 之前就把 rollout 数据落盘了。**崩掉的 CP2@96k run 已留下 `0.pt`(rollout0, 14:17 落盘, 就是触发死锁的那批 96k 长轨迹) 和 `1.pt`(rollout1, 14:40)**，在 `…_cp2_96k_20260625_124601/rollout_dumps/`。
- 设 `LOAD_DEBUG_ROLLOUT_DATA=<path>/{rollout_id}.pt` → wrapper 透传 `--load-debug-rollout-data`（base:427-428）。
- `arguments.py:1507` `skip_sglang = load_debug_rollout_data is not None`，`:1811` 起 `load_debug_rollout_data` 时**不起 SGLang、不连 Daytona、只跑训练进程**；`:1816` 自动置 `debug_train_only=True`。
- `placement_group.py:100` `debug_train_only` 下只分配 `actor_num_nodes×gpus`（32 GPU，无 rollout 那 64 GPU），`rollout.py:806` `_get_rollout_data` 直接 `torch.load` 那个 .pt 喂训练，**完全跳过 rollout**。

**怎么用（验 #1 抬 watchdog timeout）**：
```bash
# 只要 4 节点 actor（32 GPU），不需要 rollout 的 8 节点
export LOAD_DEBUG_ROLLOUT_DATA=/path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_overfit_ppo_lenreward_cp2_96k_20260625_124601/rollout_dumps/{rollout_id}.pt
export TORCH_NCCL_TIMEOUT=1800          # 待透传：base 的 RUNTIME_ENV_JSON 需加这行（见下）
export NUM_ROLLOUT=2                     # 复用 0.pt(critic warmup) + 1.pt(actor train)
# 其余 CP2@96k 配置不变（TP2×CP2×PP8、5道墙98304、MAX_TOKENS 49152）
```
启动后**直接进 step0 critic forward**（无 66min rollout），~10min 内就能看 GDN 编译时滞下抬高的 watchdog 能不能让它 join 过去。

**注意**：
- `TORCH_NCCL_TIMEOUT` 要进 actor 进程环境——base 的 `RUNTIME_ENV_JSON` env_vars 是显式枚举的（§5.9 §8 提到回滚时删过 `SLIME_EAGER_INIT_PARALLEL_COMMS` 行），**需要在 base launcher 的 env_vars 里显式加 `TORCH_NCCL_TIMEOUT`**（光 export 不一定透传到 ray actor）。这是改 #1 前要先落地的一步。
- dump 的 `0.pt`/`1.pt` 是 96k 长轨迹（response_len max 48798，avg 13398），**完整保留了触发死锁的负载特征**，不是缩水样本，所以复现是真复现。
- debug-train-only 只占 32 GPU（4 节点），剩 8 节点空着——迭代期间可以并行干别的，或临时缩集群。

> 详细记录见 memory：[[ctx96k-cp2-deadlocks-at-96k-not-64k]]。

---

- **critic init 崩 expandable_segments** → 确认 wrapper 的 `PYTORCH_CUDA_ALLOC_CONF` 不含 `expandable_segments`（§2）。
- **rollout 卡 collected 0/N** → `ssh <coordinator> 'rp=$(pgrep -f raylet|head -1); grep "Max open files" /proc/$rp/limits; ls /proc/$rp/fd|wc -l'`；若 current==soft limit 且日志有 `File descriptor limit reached` → nofile 没拉起来（§3）。
- **actor 好像没训** → grep `actor forward` 和 `actor train` 两个标签；确认已过 `num_critic_only_steps` 步（step0 是 critic-only）（§0 末）。
- **大量 margin-OOM** → 先看有没有伴随 python `RuntimeError`/`CUDA out of memory`；没有就是良性 warning，训练在推进就别动（§4）。
- **bridge 想省 torch_dist** → 不行，27B GDN packed-seq 在 bridge 下 `NotImplementedError`，必须 raw。

---

## §X. 96k step0 慢 2.5h + 引擎 OOM 的真根因：变长序列 → 反复 JIT 重编译（2026-06-26 查实）

### 反常现象
mem-fraction 0.6 修复版 run（raysubmit_8NsrrvimvG9eHQ6W）：死锁修复仍稳，但 **step0 critic forward+train 耗 ~2.5h**，且 critic train 71/71 一完成、首次权重广播刚起，引擎 137.150 的 TP-rank 就在 `l2norm_fwd_kernel` Triton OOM→SIGQUIT→单 rank 死→广播 NCCL 组全员挂→gloo 30min timeout→FAILED。**同款死亡点精确复现，0.6 没挡住。**

疑点：debug-train-only 验证 96k 时是"稳态 20-38s/mb"，64k 正式 run 也快——**为什么同样 96k，正式 run 却 50-200s/mb 震荡耗 2.5h？**

### 实测根因（不是"96k 比 64k 慢"，是变长→重编译）
1. **critic train 是震荡不是冷编译曲线**：实测 s/mb = `3→252→62→141→55→14→104→177→250`，快慢交替**永不收敛**。冷编译签名是单调递减后收敛；这是**每个 microbatch 形状变了反复触发新 JIT 编译**的签名。
2. **rollout 序列长度跨度巨大**：rollout0 response_len min=658/median=5413/**max=35640**；rollout1 min=536/median=10656/**max=50750**。terminal-agent 轨迹长度天然极不均匀（差 ~100×）。
3. **链条**：命令行带 `--use-dynamic-batch-size --max-tokens-per-gpu 49152 --qkv-format thd`，slime 自己在 `data.py` thd 分支做 packing（多条变长样本 `torch.cat` 成一条流 + cu_seqlens，pad 到 `pad_size=TP(2)×128=256` 的倍数）。microbatch 总 token 在 ~98304 量级浮动、pad 粒度只有 256 → 总长有 ~384 种取值；**且 thd kernel 形状 key 还含 cu_seqlens 结构（bin 内几条序列、各多长）→ 形状空间巨大**。Triton/GDN kernel 按形状缓存，每遇新形状现场编译几十~两百秒（=250s mb），见过的秒过（=3s mb）。
4. **为何 debug/64k 不慢**：debug 读固定 dump、样本形状重复度高、无 fully-async 引擎抢资源→几遍编完；64k 上下文上限低→长尾被截短→**形状空间小得多**→编译少。96k 放开上限→长尾到 35k-50k token→**独特形状空间爆炸**→2.5h 都没编完。
5. **OOM 与慢同源**：引擎侧 OOM 之所以发生在权重广播期，正是某个长尾大形状的 `fla.l2norm_fwd_kernel`（SGLang 引擎侧 GDN，`sglang/srt/layers/attention/fla/l2norm.py`，**非** actor 侧 Megatron GDN）恰在那个高显存时刻才第一次被 JIT 编译。形状空间小、早编完就不会撞高水位。**→ 缩形状空间一招同治"慢"和"OOM"。**

### 关键坑：`rl_use_sequence_packing` flag 对 slime 无效
launcher 打印的 `rl_use_sequence_packing=False` 是 **Megatron 自带 RL 框架（`megatron.rl`）的 arg**，slime **不走那条路**（grep `megatron.rl|perform_rl_step` 在 slime 全空）。slime 用自己 `data.py` 的 thd packing。**直接传 `--rl-use-sequence-packing` 无效甚至冲突，别传。** slime 侧真正控形状的旋钮是 `--data-pad-size-multiplier`（默认 128，`arguments.py:1363`，base 脚本从未设过）。

### 修法（按治本排序）
- **A. 持久 `TRITON_CACHE_DIR`（治本，零风险，必加）**：当前没设（用默认 `~/.triton/cache`，容器内每 run 重置）。设到本地/共享盘 → 编过的形状跨 step/跨 epoch/跨 run **永久命中**，第 2 个 epoch 起不再重编，且消除"迟到编译撞高水位 OOM"。要在 base launcher 的 `RUNTIME_ENV_JSON.env_vars`（`run_qwen35_9b_terminal_bench2_64h20.sh:680-711`，**显式枚举**，光 export 不透传到 ray actor）里加 `"TRITON_CACHE_DIR"`。同理可给 actor + 引擎进程都设。
- **B. 调大 `--data-pad-size-multiplier`（收敛 T 维）**：把 pad 粒度从 256 调粗（如 1024/2048），让 microbatch 总长收敛到少数 bucket，减少 T 维形状数。代价=多 pad 一点算力。**注意**：单独此项不彻底，因 thd 形状还含 cu_seqlens 结构；需与 A 合用。
- **C. 缩 actor 侧 2.5h 冷编译窗** = 另一条线，工程量大，A+B 够用再说。

> 详细 memory：[[cp2-96k-sglang-oom-during-step0-compile]]。l2norm OOM 在引擎侧 fla，非 actor Megatron GDN——这是两套独立 GDN 实现，别混。

### CPU 小实验实测（真实崩溃 dump `0.pt`/`1.pt`，2026-06-26）
用真实 rollout0（264 样本/71 mb，token len min1855→median18742→**max77586**）跑 slime 真实 `first_fit_pack` + `data.py` thd pad 逻辑：

| pad_mult | pad_size | distinct (T,n_seq) = **单 step JIT 编译数** | pad 浪费 | max padded T |
|---|---|---|---|---|
| **128（当前默认）** | 256 | **60**（71 mb 里 60 个形状各异→几乎每 mb 都重编）| 0.1% | — |
| 1024 | 2048 | 29 | 1.0% | — |
| **2048（甜点）** | 4096 | **22** | 1.6% | **98304（顶 cap，headroom=0）** |
| 4096 | 8192 | 17 | 3.3% | **溢出风险**（>98304 cap）|

**跨 step 缓存命中（持久 TRITON_CACHE 的价值，rollout0→rollout1）**：
- pad128 无缓存：step1 的 66 形状 **全部重编**（命中 0）。
- pad128 + 持久缓存：step1 命中 19、新编 47。
- **pad2048 + 持久缓存：step0 编 22，step1 只新编 5（命中 9）**。第 2 epoch 起基本全命中→零编译。

**结论**：两旋钮是乘法关系。`pad2048 + 持久 TRITON_CACHE` → step0 一次性编 22（几分钟），后续步个位数，2.5h 冷编译窗塌缩；长尾大形状在 step0（显存空）就编完进缓存→广播期无"迟到 JIT 撞高水位"→**慢和 OOM 同治**。

**约束**：
- pad_mult **不能盲调大**：pad2048 时 max padded T 已顶满 98304 cap（headroom=0），pad4096 会溢出 bin cap。**pad2048 是上限甜点**。
- 正确性安全：pad 段作为独立 cu_seqlens 段加在末尾（`data.py:107-110`），attention 不跨段、不参与有效计算；`loss_masks` 不含 pad（`loss.py:695`）。
- `--data-pad-size-multiplier` 是已存在 arg（默认 128，`arguments.py:1363`），base 脚本从未设过 → 直接透传即可，无需改 core。
- `TRITON_CACHE_DIR` 要加进 base launcher 的 `RUNTIME_ENV_JSON.env_vars`（`run_qwen35_9b_terminal_bench2_64h20.sh:680-711` 显式枚举，光 export 不透传到 ray actor）；引擎侧 fla kernel 同样吃这个缓存。用**本地盘**（每节点独立，避免共享盘并发写竞争），跨 run 复用同一节点的产物。

### 已落地改动（2026-06-26，向后兼容，默认行为不变）
- **base launcher** `run_qwen35_9b_terminal_bench2_64h20.sh`：
  - `DATA_PAD_SIZE_MULTIPLIER` env（默认空）→ 非空时 `PERF_ARGS+=(--data-pad-size-multiplier ...)`（行 ~495）。空=不传=stock 128。
  - `TRITON_CACHE_DIR` env（默认 `/tmp/triton_cache_${USER}`，行 ~121）→ `mkdir -p` + `export`（行 ~268）+ 进 `RUNTIME_ENV_JSON.env_vars`（行 ~714，这步是关键，否则不透传到 ray actor）。
- **新 debug 脚本** `run_tb2_full89_cp2_96k_dbgtrain_padmult_tritoncache.sh`：基于现成 dbgtrain 脚本，设 `DATA_PAD_SIZE_MULTIPLIER=2048` + 每 RUN_TAG 独立空 `TRITON_CACHE_DIR`（公平测冷编译）+ 独立 run identity，复用 memfrac06 run 的 `_cp2_96k/rollout_dumps/{0,1}.pt`（264 样本/max77586，和 2.5h 慢编译同源）。

### debug-train-only A/B 实测（进行中，2026-06-26→27）
job **raysubmit_8Ph4sCujubs2DxbU**，4 节点 32GPU（head 10.0.0.8 + 111/210/137.6），DL=`/tmp/rtb2dbg_pad2048/session_latest/logs/job-driver-raysubmit_8Ph4sCujubs2DxbU.log`。已确认 train cmd 里 `--data-pad-size-multiplier 2048`、`debug_train_only=True`、不起 sglang、replay 同源 dump。**验证点**：step0 critic forward s/mb 是否从震荡(3↔252)变"前几个慢编译后稳态快"、总耗时是否 <<2.5h；step1(1.pt) 是否比 step0 快(跨 step 缓存命中)；loss 不 NaN(粗 pad 不破 masking)。**baseline 对照**=已观测的 pad128 正式 run=step0 2.5h/震荡。

**ssh 起 head/worker 坑**：`env VAR=val` 不能带 `export` 关键字（`env: 'export': No such file or directory`）；nohup-via-ssh 配短 timeout 易超时退 143 但 nohup 实际已落地，需单独 verify worker/launcher 是否真起来。

### ⚠️ 实测结论（2026-06-27）：pad2048 本身导致 step0 首 mb HANG → 作废
- **大 dump pad2048**（job raysubmit_8Ph4sCujubs2DxbU）：critic forward 卡 0/71 12+min，全 32GPU 平直 0%util/110W，无 ptxas、无 watchdog（未到 30min 边界）= 真 wedge。
- **小 dump pad2048**（job raysubmit_b2xqiFv74SEKp4PH，`_20260625` 15MB/35样本，但长度分布≈大dump max76941/median32758，只是样本少）：**同样卡 0/13 13min**。
- **决定性对照**：之前 PASS 的 dbgtrain run（`/tmp/rtb2ppo_dbgtrain/.../raysubmit_VEswz7RPaU4jB4JW.log`）**同一小 dump、pad 默认 128、第一个 mb 2:07 就出来（127s 冷编译正常）、跑完 13/13**。**唯一变量 = `--data-pad-size-multiplier 2048`** → 是 pad2048 改动本身造成 hang。
- **CPU 模拟为何没抓到**：复现了 data.py thd+CP 真实流程（per-sample `slice_with_cp` 各 pad 到 `2*cp*chunk_size` → concat → 整流 pad 到 pad_size 倍数 → 追加 pad 段 → cu_seqlens×cp）。pad2048 下 max padded T=49152（同 pad128）、distinct shapes 13→7、pad 段从不是 max seqlen、无超 cap——CPU 看全干净。hang 只在 runtime 真 GDN/NCCL 路径显现。
- **嫌疑（未证实）**：pad2048 让追加的 pad cu_seqlens 段变大（1732 vs pad128 的 58 token），改变 thd packing layout，可能让 GDN Triton kernel 在该 padded 形状上 hang，或触发 CP 子组 NCCL 路径异常。
- **结论**：`--data-pad-size-multiplier 2048` 收敛形状的思路 CPU 上成立但 **runtime 会 hang，作废，别再设**。两个 env 旋钮（`DATA_PAD_SIZE_MULTIPLIER`/`TRITON_CACHE_DIR`）保留在 base launcher（默认空/默认路径=向后兼容）。`TRITON_CACHE_DIR` 持久缓存改动独立、无害、可保留。
- **下一步候选**：① 只上 `TRITON_CACHE_DIR`（不动 pad）看跨 step 缓存能省多少冷编译；② py-spy 到真 compute rank/cuda 层定位 pad2048 具体 hang 在 GDN/thd 哪步；③ 回原始 2.5h-慢-但能跑配置，靠持久缓存让第 2 个 epoch 起命中、step0 一次性付。

> 详细 memory：[[cp2-96k-padmult-tritoncache-debug-experiment]]。

### ⚠️⚠️ 重大修正（2026-06-27）：大 dump + pad默认128 在 debug-train-only 下也死在 mb0 → 推翻"pad2048 是唯一元凶 / 大dump pad128 没问题只是慢"

为坐实"是否 pad2048 单独造成 hang"，把 pad **改回默认 128** 跑**同一份大 dump**（job **raysubmit_BXFPV6Y9q5DPdYAY**，4 节点 32GPU debug-train-only，264 样本/71mb，head 10.0.0.8 + 111/210/137.6，DL=`/tmp/rtb2dbg_pad128big/session_latest/logs/job-driver-raysubmit_BXFPV6Y9q5DPdYAY.log`）。结果**不是 wedge，是真编译了 ~70min 然后死于 CP 子组超时**：

- 02:30:41 critic forward 起 `0/71`。
- ~70min 内节点 111 反复抓到**新 ptxas（etimes 1-4s）→ 随后 100%util/351W**，driver mtime 阶梯式推进（02:49→03:06→03:27）= 逐 PP-stage 冷编译，**确实在干活**（不是自旋）。
- **03:28 死**：CP 子组 `COALESCED Numel=0` watchdog timeout（`PG ID 20 PG GUID 345 Rank 5`，ran 38min=2287048ms，`last completed work: -1`，traceback `p2p_communication.py:475 send_forward`）。
- **始终没过 `0/71`，一个 microbatch 都没完成**。

**厘清了两种独立的失败模式**（之前混为一谈）：

| | 模式 A：pad2048 wedge | 模式 B：大dump pad128（本次）|
|---|---|---|
| 0/71 时 GPU | **无 ptxas、无 watchdog**，纯 0%util/110W 自旋 | **真编译 70min**（反复 ptxas + 间歇 100%util/351W）|
| 死法 | 静默 wedge（未到 30min 边界） | CP 子组 COALESCED 超时 abort（30min 触发）|
| 元凶 | **pad2048 改动本身**（小dump 改回 pad128 就 PASS）| **大dump 的长冷编译窗本身**（和 pad 无关）|
| 依赖 | pad-specific | dump-size / shape-count-specific |
| 同族签名 | — | = §5.10 / [[ctx96k-cp2-deadlocks-at-96k-not-64k]] 同款空-COALESCED |

**为什么和 §5.10「30min new_group 修复已验证通过」矛盾**：§5.10 的验证用的是**小 dump**（shape 少 → mb0 冷编译窗短 → 不超时）。**大 dump（264 样本、max 77586、~60 distinct shapes）的 mb0 冷编译窗本身就 ≥30-70min，超过 30min CP 子组超时** → 30min monkeypatch 对大 dump 不够。小 dump pad128 PASS 的对照（§604）仍成立，但**不能外推到大 dump**。

**修正后的结论**：
1. `--data-pad-size-multiplier 2048` 仍作废（模式 A runtime wedge，§604 不变）。
2. **大 dump 在 debug-train-only 下根本过不了 mb0**（模式 B：70min 冷编译窗 > 30min CP 子组超时）。之前"大dump pad128 没问题只是慢"是错的。
3. 原始正式 run 能跑 71/71（2.5h）说明正式 run 某处侥幸过了 mb0（mb0 没到 70min / 拓扑差异 / 缓存差异），但**根因坐实 = 大 dump 变长序列的 mb0 冷编译窗本身长到逼近甚至超过 CP 子组超时**。慢和死锁同源。

**下一步候选（更新，按对症模式 B 的性价比排序）**：
1. **抬 `SLIME_NEW_GROUP_TIMEOUT_MINUTES` 到 90+min**（最便宜，一行 env）：让 CP 子组扛过 70min 冷编译窗，立刻测大 dump 能否过 mb0。本身也是无害的稳健性改进。
2. **TRITON_CACHE 预热**：先用小 dump / 单形状把 cache 焐热，正式 run mb0 命中缓存 → 冷编译窗塌缩到不超时。把"持久缓存"思路真正用起来（治本且对症）。
3. 真·GDN 预编译（最治本，工作量大）。
4. pad 换更温和的值（256/512，避开 2048 的模式 A，又比 128 少些 shape）—— 需重新 runtime 验证，优先级低。

> 详细 memory（已更正）：[[cp2-96k-padmult-tritoncache-debug-experiment]]。

---

## §Y. 64k 正式 run（cp2_64k_20260627_122131）FAILED 根因：偶发引擎冷编译 OOM + PPO 权重重连对死引擎零容错（2026-06-27 实测，靠双对照 run 的命令行 diff 坐实）

**TL;DR**：TRITON 修复（删 base launcher 未提交的 `TRITON_CACHE_DIR` 重定向）**真有效**——step0 用热 `~/.triton` 缓存完美跑完（critic forward 5:59 + critic train 17:30 + 优化器步 critic-step 0，零冻结）。**但 run 仍在 step0→step1 权重同步处 FAILED**，且**根因不是 64k、不是 batch、不是 PPO 同步点本身**——是**一次偶发的 SGLang 引擎冷编译 OOM 死亡 + PPO 每步强制重建 65-rank NCCL 组对死引擎零容错**的叠加。

### 失败时间线（job raysubmit_QjqayTCsj2Dm6AEP）
1. **step0 全过**：rollout0 ~44min 填满 → critic forward 139/139 (5:59) → critic train 139/139 (17:30) → `critic-step 0` 优化器步落地（value_loss 5.58, grad_norm 196.9）。
2. **13:20 一个引擎 rank 死**：node 137.72 的 SGLang TP rank `pid=3877771` 在为新 padded shape 冷编译 `l2norm_fwd_kernel`、`_init_handles()/load_binary()` 把 kernel 二进制 load 进 GPU 时撞 `Triton Error [CUDA]: out of memory` → SIGQUIT → **进程死、不自愈**（13:20:06 后该 pid 再无任何输出；同节点其他 pid 3877770/72/73 活到 14:20——我一度误判"137.72 恢复了"，其实只是别的 rank 还活着）。完整 traceback 栈：`chunk_gated_delta_rule → l2norm_fwd → l2norm_fwd_kernel[grid] → _init_handles → load_binary → CUDA OOM`。**OOM 发生在 kernel load_binary（编译期把二进制塞进 GPU），不是算激活——降 mem-fraction 不一定对症，是瞬时编译 buffer + 已贴顶显存的叠加。**
3. **13:50 actor 进 `update_weights()` 卡死**：PPO（`use_critic=True` + `offload_train` + 非 colocate）下 `reconnect_rollout_engines=True`，**每步**都强制 `wake_up()` + `connect_rollout_engines()` 重建 world_size=65 的 NCCL 组（`actor.py:615-628`）。调到死掉的 `pid=3877771` 的 `init_weights_update_group` 报 `Unhandled error` → 该 rank 永远加不进组。
4. **卡在重建之后那道 `dist.barrier(group=get_gloo_group())`（actor.py:627）**：要 65 rank 全到齐才放行，缺死 rank → 永久缺人。`before update_weights` 从未打印（证明卡在 barrier、根本没走到真广播）。
5. 等满 gloo timeout（30min）→ `RuntimeError: Timed out waiting 1800000ms for send operation` → exit 1 FAILED（14:21，head/raylet/dashboard 全拆）。137.78 的 CUBLAS 崩在 14:06，是卡死 16min 后的次生事件，不是主因。

### 为什么基线没事、这次挂——双对照 run 命令行 diff（终结所有"配置理论"猜测）
对照三个 64k run（其余 flag 几乎逐字相同：batch、mem-fraction 0.75、seq-length/context 65536、拓扑 TP2×CP2×PP8 全同）：

| run | batch | num-rollout | 实际跑到 | 引擎崩(Scheduler exc/OOM/SIGQUIT) | 结果 |
|---|---|---|---|---|---|
| `ppo_033051`（基线，6/22） | **16×20** | 8 | ~step1 **被手动 stop** | **0** | stopped（没跑完）|
| `ppon4_055519`（n4，6/22） | **64×4** | 7 | **critic-step 0→6, 16 次权重同步全过** | **真 0**（CUBLAS/gloo 命中全是 trial-worker sandbox 噪声）| **succeeded** |
| `cp2_64k_122131`（本次，6/27） | **64×4** | 40 | step0 全过, step1 重连卡死 | ≥2（137.72 TritonOOM 死 + 137.78 CUBLAS）| **FAILED** |

**关键证伪**：`ppon4_055519` 用**完全相同的 `64×4` batch**跑完 7 步、16 次权重重连、9.8h，**引擎一次没崩**，succeeded。所以：
- ❌ 不是 batch/并发（64×4 本身 n4 证明能稳跑）。
- ❌ 不是 64k 本身（三个 run 同 64k）。
- ❌ 不是 PPO 重连同步点脆弱（n4 稳过 16 次）。
- ❌ "基线跑完 6 步穿过同步点很多次"是**错前提**——基线只到 ~step1 就被手动停、且 0 引擎崩；它是**没踩雷**通过，不是更安全。

**真正差异 = 命令行 diff 只有 4 处**：`save-interval`(20→10)、`num-rollout`(7→40)、本次多了 `--custom-reward-post-process-path`(长度奖励)、本次多了 `--distributed-timeout-minutes 60`（n4 用默认）。其中决定"一次偶发引擎崩→整 job 死"的是：

**根因 = 物理偶发 + 配置放大的叠加**：
1. **（偶发触发，物理概率）** SGLang 引擎在长上下文运行期为新 shape 冷编译 GDN `l2norm_fwd_kernel` 时 `load_binary` 撞显存 OOM 死亡。这是概率事件——n4 跑 9.8h 一次没中，本次 step0 期间中了一次。**跑得越久（40 步 vs 7 步）、撞上的窗口越大。**
2. **（放大器，系统性）** PPO `offload+非colocate` 每步强制重建 65-rank NCCL 组 + 全员 gloo barrier，**对"已死的引擎 rank"零容错**——一个死 rank 就让整组 barrier 干等到超时崩，而不是跳过/重启它。`--distributed-timeout-minutes 60` 把这个干等拉到 30min 才抛。

= 与 [[cp2-96k-sglang-oom-during-step0-compile]] 同一族杀手（`l2norm_fwd` Triton OOM → SIGQUIT → 权重同步 barrier 卡满超时 FAILED），**且证明 64k 也会中招，不是 96k 专属**。

### 修法（按对症性排序）
1. **治偶发触发——别让引擎在长上下文下冷编译 OOM**：(a) **限 `max_running_requests`**（两 run 都默认 None=不限，动态批可能太激进）削同时在飞的长 decode 数=削显存峰，给 load_binary 留余量；(b) **预热 SGLang 引擎 GDN kernel**（启动后跑覆盖典型 shape 的 warmup 请求），把"运行期边服务边冷编译 load_binary"危险窗消掉——最治本。
2. **治放大器——权重重连 barrier 对死引擎容错**（更治本但要碰 `actor.py` core 同步逻辑，CLAUDE.md 反对改 core，风险高，先不动）：理想是 `connect_rollout_engines` 探测到死引擎时跳过/触发重启而非让 barrier 干等。
3. 短期：缩 `--distributed-timeout-minutes` 让卡死更快暴露（治标，不治崩）。

> 详细 memory：[[tb2-64k-real-run-restart]]（含 TRITON 修复坐实 + 本失败诊断）、[[cp2-96k-sglang-oom-during-step0-compile]]（同族杀手）。

---

## Daytona 沙箱创建限流：tmax-15k run 两次被"假 S3 故障"打停的真因 + 客户端限速补丁（2026-07-01）

> ⚠️ **更正（见下一节 "真根因：obstore 不走 proxy"）**：本节把根因判成"限流 + S3 负载敏感"是**错的**。真正原因是 Daytona SDK 上传 build-context 用的 **obstore（Rust S3 客户端）不读 `HTTPS_PROXY`**，而我们的机器必须走 star-proxy 才能连 AWS S3 → 每个 signed HEAD/PUT 超时。限流补丁仍应保留（600/min 限流真实存在，280 一次性冲会超限），但它**不是**训练起不来的原因。本节保留作为排查过程记录。

### 现象
tmax-15k FULL PPO run（14601 任务、n=1、oversample 280/train 256）**两次拉起、两次在 rollout 0 全军覆没**：
- attempt#1 `20260701_020009`（job FbS42Uz，W&B d7nbxyfy）02:24 停
- attempt#2 `20260701_060445`（job YZHgE7hN，W&B m7pbdniz）06:27 停

两次签名完全一致：`collected 0/256` 始终为 0、`Sending keys`=0（**没有一个 agent 真正进过 sandbox**）、Daytona census=0（**没有一个 sandbox 起来过**）、`completed` 狂涨（全是失败 trial 在 attempt_multiplier 下重试堆积）。100% 失败都是：
```
Failed to create sandbox: Generic S3 error: Error performing HEAD
https://s3.us-east-1.amazonaws.com/prod-daytona-build-context/<org>/<id>/context.tar
in ~56-69s, after 10 retries, max_retries: 10, retry_timeout: 180s
```

### 误判与纠偏：单发探测 ≠ 并发健康（关键教训）
重启前我用**单发** image-build 探测（`AsyncDaytona` + `Image.from_dockerfile` + `CreateSandboxFromImageParams`，复刻 Harbor `_DaytonaDirect.start()` 真实路径）验证："S3 恢复了"——单个 sandbox 2.0~2.4s 就建好（newest key 和训练 KEY6 都过）。**据此重启，结果第二次又挂在同一个地方。**

于是写了**并发** image-build 探测器，一次性坐实问题：
- 单发：2s 成功 ✅（误导性）
- **32 并发：0/32 成功**，32 个全 `Generic S3 error` ❌
- 280 并发（真实训练）：100% 失败 ❌

→ **教训：S3/创建路径是负载敏感的，单发探测不能作为"可拉起"的证据。拉起前必须用并发探测 gate。**

### 真因：Daytona 按 org 限流 600/min（=10/s），卡的是 `.create()`
找 Daytona 团队确认：
1. **每个 org 的 sandbox 创建限流 = 600/min（≈10/s）**，文档 https://www.daytona.io/docs/en/limits/#limits
2. 限流是 **per-org**（多 key 分摊无效）
3. 卡的是 **`.create()` 本身**（snapshot 创建也走 `.create()`、同样计入 → "预建 snapshot 绕过"也绕不开）

我们的 `fully_async_rollout` 是**单进程单例** `HarborFullyAsyncRolloutWorker`，在 rollout 开始时 `while len(active_tasks) < max_concurrent_groups(280)` 一个 tick 内**一次性创建 280 个 group task**，且 `harbor_num_concurrent_trials` 故意设成 280（注释原话"raise it to 280 so it never throttles"）→ 从单个 API 客户端**~1s 内打出 ~280 个 `.create()`**，超限约 28×。这就是 S3 build-context HEAD 被限流拒绝的根因。**不是 Daytona 故障，是我们打爆了限流。**（32 并发探测=32/s 也已超 10/s，所以也 100% 失败。）

### 修法：客户端令牌桶限速（不动训练逻辑，只给 create 节流）
所有 trial 派发都过单进程漏斗 `rollout.py::_run_one_with_semaphore`（实际 `.create()` 在远程 TrialWorker actor 上，但每个 trial 恰好一次 create），所以**一个进程级令牌桶**就能精确控制全 org 创建速率。

- `rollout.py`：新增 `_AsyncTokenBucket`（async 令牌桶：`rate` tokens/s + `capacity` burst）+ 进程级单例 `_get_create_rate_limiter(args)`。在 `_run_one_with_semaphore` 派发 trial **前** `await limiter.acquire()`（放在并发 semaphore 之内，使"在飞上限 280"与"创建速率上限"正交叠加）。
- 配置（YAML，经 `getattr` 读取，无需改 arguments.py）：
  - `harbor_create_rate_per_sec: 8.0` —— 8/s，留余量给失败重试+稳态 churn（限 10/s 之下）
  - `harbor_create_rate_burst: 8.0`
  - `0` = 不限速（向后兼容，旧行为）
  - 若 Daytona 把 org 限提到 ~3000/min，可调到 ~45/s
- `max_concurrent_groups(280)` 仍控制**在飞上限**；令牌桶只控制**创建速率**——280 个不再一瞬间冲出，而是按 8/s 在 ~35s 内陆续起来。

### 工具脚本（拉起前的 gate）
- `examples/harbor_terminal_rl/daytona_concurrent_build_probe.py` —— 并发 image-build 探测器：从训练池采真实 Dockerfile 并发起 N 个 `.create()`，报告成功率/耗时分位/失败分类（专门标 `S3_BUILD_CONTEXT`）；建完即删；**退出码即 gate**（成功率≥阈值→0）。带 `--rate-per-sec` 复刻训练同款限速，可端到端验证节流是否解决问题。
- `examples/harbor_terminal_rl/probe_then_launch_tmax15k.sh` —— 先并发探测、**通过才**拉起 launcher+11 worker；失败则不拉起、不留垃圾（`DRY_RUN=1` 只探测）。

> 详细 memory：[[tmax15k-full-ppo-run-20260701]]（两次停训练全过程）、[[daytona-cleanup-needs-proxy]]（egress 走 star-proxy）。

---

## 真根因：obstore（Daytona SDK 的 S3 客户端）不走 proxy —— build-context 上传全超时（2026-07-01，已修复+验证）

上一节的"限流/S3 负载敏感"结论**是错的**。经过给 Daytona 提工单、逐层最小复现，真根因简单得多，且**完全在我们这侧的网络环境**，不是 Daytona 后端、不是限流、不是 key、不是 org。

### 关键最小复现（一步步逼出真相）
`create(Image.from_dockerfile(...))` 会让 Daytona SDK 把 build context 上传到它的 S3 桶 `prod-daytona-build-context`（us-east-1）；上传前先对 `context.tar` 发一个 **signed HEAD**（`AsyncObjectStorage._file_exists_in_s3` → `obstore.S3Store.head_async`）。逐个测：

| 测试 | 结果 | 说明 |
|---|---|---|
| `get_push_access()`（拿 S3 临时凭证） | ✅ 1.0s | Daytona API 正常 |
| **匿名** HEAD（`skip_signature=True`）+ proxy | ✅ 0.8s (403) | 网络/proxy 通 |
| **signed** HEAD（用下发凭证）+ 走 proxy 环境变量 | ❌ **58s 超时** 10 retries | 就是训练里的错误 |
| **signed** HEAD + `client_options={"proxy_url": PX}` | ✅ **1.2s** (404) | **proxy_url 参数生效** |
| signed HEAD + **不走 proxy** | ❌ 58s 超时 | 证明 obstore 直连 AWS |
| 32 并发 @ **1/s**（实测 ~8s/个） | ❌ **0/32** | 证明与速率无关 |
| 两个不同 key（同 org `56846f3c` + 另一 org `03e2a149`） | ❌ 都 58s | 证明与 key/org 无关 |
| 6/30 之前同一 image-build 路径 64 并发 | ✅ 99% | 证明这条路本来能用 |

### 根因
1. **obstore（SDK 底层的 Rust `object_store` 客户端）不读 `HTTPS_PROXY`/`HTTP_PROXY` 环境变量。** 它只认构造时传入的 `client_options.proxy_url`（obstore `ClientConfig.proxy_url` 字段）。
2. **更糟**：Daytona SDK 在 `daytona/_utils/environment.py::isolated_env()` 里构造 `S3Store` —— 这个上下文管理器会 **`os.environ.clear()`**，把环境变量整个清空再建 store，且构造时**没传 `client_options`**。双重保证 obstore 拿不到 proxy。
3. 我们这些节点连 `s3.us-east-1.amazonaws.com` **必须走 star-proxy**（curl 实测：走 proxy 0.8s 通、不走 proxy 15s 墙）。于是 obstore 直连 AWS → 每个 signed HEAD/PUT 超时 → 10 次重试 ~58s → `Failed to create sandbox: Generic S3 error: HEAD .../context.tar after 10 retries`。

**为什么绕了一大圈**：单发/匿名请求避开或碰巧不触发这条超时路径，看起来"偶尔能成"；错误信息里醒目的 `s3.us-east-1.amazonaws.com` + `after 10 retries` 把我们引向了"S3 故障 / 限流"的猜测。其实从头到尾是本地 proxy 没接上。Daytona 工程师反复问"为什么你会发这个 S3 请求""是不是该用别的方式"——他一直在暗示问题在我们这侧。

### 修复（`slime_plugins/harbor/import_utils.py::ensure_daytona_s3_proxy`）
monkey-patch SDK 在 `daytona._async.object_storage` / `daytona._sync.object_storage` 里用的 `S3Store` 符号，让每次构造都注入 `client_options={"proxy_url": <proxy>}`（若调用方没设）。proxy 取自 `HARBOR_DAYTONA_S3_PROXY_URL`（显式优先），否则回退到 `HTTPS_PROXY`/`HTTP_PROXY` 等。幂等；无 proxy 配置时 no-op；SDK 布局变了也安全跳过。在 `rollout.py::_run_one_trial` 开头调用（紧挨 `ensure_harbor_distribution_metadata()`），每个 trial worker 进程都会生效。

> **不改 Daytona SDK 源码**（升级会丢），也不改 slime core —— 用插件侧 monkey-patch，符合 CLAUDE.md "用钩子不 fork" 原则。

**启动时需设** `HARBOR_DAYTONA_S3_PROXY_URL=http://star-proxy.oa.com:3128`（或确保 `HTTPS_PROXY` 已在 rollout worker 环境里）。

### 验证链（全部实测通过）
- signed HEAD + `proxy_url` → **1.2s**（vs 无 proxy 58s）
- 走 Harbor SDK 真实路径（`AsyncObjectStorage._file_exists_in_s3`，打补丁后）→ **1.2s**
- **真实 `create(Image.from_dockerfile(真实 tmax Dockerfile))`（打补丁后）→ 47.4s 建出 sandbox、`exec`=OK、干净删除** ✅

### 与限速补丁的关系
限速补丁（`harbor_create_rate_per_sec`，上一节）**保留** —— 600/min（10/s）的 per-org 限流是真实的，280 一次性冲仍会超限。但它不是训练起不来的原因；proxy 修复才是。两者正交叠加：proxy 让 create 能成，限速让并发不超 600/min。

> 详细 memory：[[tmax15k-full-ppo-run-20260701]]、[[daytona-cleanup-needs-proxy]]（egress 走 star-proxy 的老坑，这次又是它的新变种——连 SDK 内部的 obstore 都要单独喂 proxy）。

---

## §Z. tmax-15k 正式 run #3 死于 64k **backward** 显存不足（debug-train-only 坐实，2026-07-02）

Daytona/proxy 修复后，正式 run #3（`20260701_133022`，job `raysubmit_K5Mgt5SMkCpKGaF3`）**成功训练 ~3.5h**：跑完 rollout 0/1/2、存了 `iter_0000004`。但在 **rollout-2 → train 的权重同步**阶段死掉：`update_weights()` → `init_process_group` 卡 `DistStoreError: Timed out after 1801s, 63/65 clients joined`（2 个 rank 没到），伴随大量 `TorchMemorySaver::malloc return OOM`（free 仅 1-2GB）。

当时的假说：这是 CRASH#1/#2 家族（权重同步 barrier 对 dead-rank 零容错），触发器是**训练侧 64k 显存尖峰**挤死 2 个 rank。为了把「load 是否健康」「64k 训练侧是否 OOM」两个问题彻底分开，用 `--debug-train-only`（SGLang+Daytona 全关，只用 actor 卡、replay 主 run dump 的真实 rollout 数据）在空闲卡上做了一串隔离实验。**结论比原假说更硬也更糟：不是权重同步的偶发问题，是 64k backward 系统性显存不足。**

### 1) 三次 debug 实验（全部 replay 真实 rollout_dumps/*.pt）

| 配置 | GPU | load | 64k forward | 64k backward | 结论 |
|---|---|---|---|---|---|
| **TP2×PP8×CP4** | 64 | ✅ | ❌ 第一个 mb 就死锁 | — | CP4 collective 死锁（先疑节点 .136.114 单点，100%util/118W 自旋、其余 0%） |
| **TP2×PP8×CP4**（排除 .114） | 64 | ✅ | ❌ 仍死锁（全体 0%util，无自旋） | — | 排除节点仍死锁 → **CP4 配置本身死锁，与节点无关**；.114 是无辜的 |
| **TP2×PP16×CP2** | 64 | ✅ | ✅ **0%→100%, oom=0** | ❌ **一进 backward 全集群 OOM** | forward 能跑，**backward 立刻 OOM** |

（注：CP4 在 64k 下的死锁与 §5.7–5.10 记录的 GDN linear-attn 层 `all_gather` 在 PP 下 rank 不同步是同一族问题——CP 度数 × 序列长度相关的 NCCL 死锁。这次没再深挖 CP4 死锁，因为发现了更根本的 backward 显存墙。）

### 2) 铁证：`--debug-train-only` + recompute 拉满，backward 仍 OOM

PP16 run（`20260702_040022`）的关键观测：
- `critic forward` **100% 完成，oom=0**（中途 95% 那个"停顿"只是 forward→backward 过渡，不是 stall）。
- 一进 `critic train`（backward），**全集群密集 `TorchMemorySaver::malloc return OOM`**（`free_bytes` ~1.1–1.6GB，`repeated 46x/10x/7x across cluster`），microbatch 速率断崖 **1.5s → 25s+/mb**（渐进恶化，到 ~35/110 仍在劣化）。TorchMemorySaver 是 retry 而非硬 crash，所以没立刻崩，但这**正是主 run rollout-2 死亡的前兆形态**：显存尖峰 → malloc retry 风暴 → 拖慢 → 某 rank 最终拿不到显存 → 挂 → 其余在 barrier 干等超时。

两个条件让这个结论无可辩驳：
1. **recompute 已拉到最满**：命令行 `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`（逐层重算，最省激活显存的配置），backward 仍 OOM。
2. **这是 debug-train-only**：actor 独占全部 GPU 显存，**没有 SGLang 抢**。纯训练侧 64k backward 本身就吃不下。

→ **PP16 只解决了 forward，没解决 backward。** backward 的激活+梯度显存需求远高于 forward，64k 下即使 PP16（每 stage 4 层）+ full recompute 仍超单卡容量。**主 run rollout-2 的死是系统性的 64k backward 显存不足，不是权重同步 fluke。**

顺带彻底澄清的事实：
- **「硬件坏导致 27B load 失败」这个旧前提彻底作废** —— 三种配置下 load 全部正常（`.distcp` reshard-free，PP8→PP16 / CP2→CP4 自动重切）。§5 之前那些因 load 失败写下的旧结论，其 load 部分现在可信了。
- **`.136.114` 不是坏节点** —— PP16 带着它照样把 forward 跑完。

### 3) ★ 待解决的关键判据：PP16 的 backward OOM 是**激活主导**还是**权重/优化器主导**？

这决定了下一步该往哪个方向降显存，是**必须先回答**的问题（目前**尚未测定**，下次 debug 优先查这个）：

- **若激活（activation）主导** → **CP（context parallel）有救**。CP 把一条 64k 序列切到多个 rank，直接砍激活（forward+backward 都砍）。CP8 把序列切到 8 rank，单卡激活 ≈ CP2 的 1/4，可能一刀解决。
- **若权重/梯度/优化器状态主导** → **CP8×PP4 反而更糟**（考虑到 recompute 已拉满、激活理论上本该很小，这个可能性不低）。

为什么"权重主导"下 CP8 更糟——**CP 和 PP 砍的是不同的显存**：
- **PP 砍权重/梯度/优化器**（切参数到不同 stage）。
- **CP 砍激活**（切序列，不动参数）。
- 64 GPU 固定、`TP2×CP8=16` → **PP 被迫 =4**，唯一拓扑是 **TP2×PP4×CP8=64**（27B 64 层 → PP4=16 层/stage）。相比 PP16 的每 rank **4 层**，PP4 每 rank **16 层** → 权重+梯度+Adam fp32 优化器状态**翻 4 倍**。而 PPO 是 **actor + critic 两个 27B 同卡**，优化器显存本就吃紧。
- 所以若 PP16 的 backward OOM 主要来自权重/优化器而非激活，换 CP8 会把权重显存抬 4 倍、激活的收益抵不过 → 净负。

**怎么测定（下次 debug 的第一步）**：在 PP16×CP2 backward OOM 时，抓 `torch.cuda.memory_summary()` / Megatron 的 mem-report，拆出 **参数+梯度+优化器 state** vs **激活** 各占多少。或做个便宜的 A/B：同拓扑下把 `MAX_TOKENS_PER_GPU` 减半（16384→8192），若 backward OOM 明显缓解 → **激活主导**（token 数直接决定激活量）→ CP8 方向对；若几乎无改善 → **权重/优化器主导** → 别碰 CP8×PP4，该从优化器显存下手（如 distributed optimizer / offload）。

### 4) 下一步候选（按把握度排序，均未执行）

1. **PP16×CP2 + `MAX_TOKENS_PER_GPU` 16384→8192**（把握最大，且顺带回答 §Z-3）。逻辑最干净：PP16 已把权重/优化器压到最低（4 层/rank），再用 8192 把每 mb token 减半 → 直接砍激活。**不碰 CP 死锁风险，也不抬权重显存。** microbatch 数翻倍（慢，但能跑）。
2. **CP8（TP2×PP4×CP8=64）**：只有在 §Z-3 测出「激活主导」时才合理。有两个真实风险：(a) PP4 抬权重显存 4×（见上）；(b) CP4 已死锁两次、CP8 用同一套 CP collective，可能同样/更严重死锁。
3. **降优化器显存**（若测出权重/优化器主导）：distributed optimizer、optimizer state offload、或减小同卡的 actor+critic 二选一常驻。

> **不要**照现在的 64k / `MAX_TOKENS_PER_GPU`=16384 直接上全量 —— 会撞同一个 backward OOM（run #3 就是这么死的）。先在 debug-train-only 上确认某个配置能**干净跑完一个完整 critic-only step（forward+backward+optimizer）无 OOM**，再上全量。debug-train-only 无 Daytona sandbox，跑完只需 `ray stop`，无需清理。
>
> 详细 memory：[[tmax15k-full-ppo-run-20260701]]（含三次 debug 的逐次时间线与 GPU 遥测签名）。相关死锁历史见本文 §5.7–5.10（CP×PP 的 GDN all_gather NCCL 死锁族）。

---

## §Z.2 正式 run（PP16×CP2, max-tok **8192**）：8192 坐实解决 backward OOM，但死于**节点掉线（网络分区）**（2026-07-02）

§Z 的"候选 1"（PP16×CP2 + `MAX_TOKENS_PER_GPU` 8192）落地成正式 16 节点 full PPO run `20260702_060921`（job `raysubmit_fXkyDKA78GivZ5eq`）。**8192 的修复被彻底验证成功**，但 run 又死了——死因和 §Z / CRASH#1/#2 全都不同，是一个**基础设施故障**（一个计算节点在权重同步中途网络失联），跟配置和代码无关。

### 1) ✅ 8192 解决了 64k backward OOM（本次决定性验证）

- 拓扑 TP2×PP16×CP2、16 节点、`--actor-num-nodes 8`、`rollout-num-gpus 64`、`MAX_TOKENS_PER_GPU 8192`、64k 上下文。
- step0 critic-only warmup 的 **critic backward 干净跑到 100%（188/188）**，全程只有 8→数十个**软** `TorchMemorySaver::malloc return OOM`（1GB margin-guard 拒绝后自动重试成功，`free_bytes` ~1.1–1.6GB，散布全程、不停顿）——与 run #3 那种"一进 backward 就密集 OOM + microbatch 1.5s→25s 断崖 → 死"的死亡螺旋形态**完全相反**。
- rollout 0/1/2 都跑完，还进到了 actor train（曾观测到 `actor train 76%`）。→ **§Z 的 backward 显存墙已被 8192 拆除，PP16×CP2×8192 是 64k 下可跑的训练配置。**

### 2) ❌ 真死因 = 节点 `.137.206` 掉线导致权重同步 all_gather 永久卡住（不是 OOM，不是显存）

时间线（launcher log 坐实）：

| 时刻 | 事件 |
|---|---|
| 10:35:08 | rollout-2 训完，进入 `update_weights`（actor→SGLang 权重同步，`_send_weights → _iter_non_expert_chunks → all_gather_param`，`update_weight_from_distributed.py:144/165` + `common.py:35`） |
| 10:38:00 | `.136.111` 的 actor 打出 **`NCCL WARN socketStartConnect: connect to 10.0.0.21<33773> returned No route to host, exceeded error retry count after 35 attempts`** → 节点 **.137.206 网络失联** |
| 10:45+ | 一堆 `Unknown Error in LLM interaction`（同一分区把路由到 .137.206 的 SGLang 请求也打断——旁证） |
| 11:05:17 | NCCL watchdog 超时：`WorkNCCL(SeqNum=10, OpType=ALLGATHER, NumelIn=44564480, Timeout(ms)=1800000) ran for 1800044 ms`，子组 **PG 52**，掉队 rank `enqueued 10, completed 1`。1800s = 我们 `SLIME_NCCL_GROUP_TIMEOUT_MINUTES=30` 补丁值 → watchdog abort → PG terminate → actor 死 → job FAILED |

**排除 OOM**：全程 host_available 最低 ~940GB、GPU free 77–90GB。之前一度误判"主机 OOM-killer"（Ray 的 `SYSTEM_ERROR: connection error code 2 / End of file` 通用报错会列 OOM 为可能原因之一），**实际是网络分区**——`No route to host` 是铁证。

### 3) 与历史死因的区分（重要，别再混淆）

| 死因族 | 触发 | 卡在哪 | 本质 |
|---|---|---|---|
| §Z run#3 / backward OOM | 64k backward 激活显存不足 | update_weights barrier（显存尖峰挤死 rank 的**后果**） | **已被 8192 解决** |
| §5.7–5.10 CP×PP 死锁 | CP≥4 的 GDN all_gather × PP schedule desync | step0 critic forward 第一个 mb（空 COALESCED / all_gather） | 配置性，CP2 不触发 |
| **§Z.2 本次** | **一个节点网络掉线（No route to host）** | update_weights 的 all_gather_param（PG 52） | **基础设施故障，非配置/代码** |

三者形态都是"NCCL 集合通信卡死 → watchdog 超时"，但**根因完全不同**。判别关键：本次 launcher log 有明确的 `No route to host ... 10.0.0.21`，且 backward 已 100% 通过（排除显存），且 CP=2（排除 CP×PP 死锁）。

### 4) 下一步（重启前，用户定夺）

- **没有 checkpoint 可热恢复**：`--save-interval 5` 且 step0 是 critic-only warmup，死在"第一次 rollout 后的权重同步"，**一个 iter 都没存下**（checkpoint 目录空）。只能冷启重跑。
- **候选**：(a) 重启前 ping 全部 16 节点，**排除/替换 `.137.206`**（若它是偶发坏节点）后冷启重跑——最直接；(b) 给权重同步 barrier 加**快速失败+重试**的容错（检测 dead rank 立即 abort 而非干等 30min），但要动 slime core，较重，且救不回已丢的这一步。
- 延长 NCCL timeout（30min 补丁）对"节点真掉线"**只能推迟死亡，救不回**——补丁本身按预期工作了。
- 失败已由 failguard 自动清 Daytona（candidates=216 cleaned=215，1 个卡 `building_snapshot`→409 的已排一次性补删）。

### 5) ★ 坐实：坏网卡 = 真凶（2026-07-02 SSH 体检坐实）

死后对全部 16 节点做 SSH 健康体检（**ICMP ping 全被防火墙拦，ping 不通≠节点挂，必须用 ssh/TCP 判**）：

- 全部 16 节点 ssh 可达、8 GPU 齐全、**ECC 错误全 0**、IB/RDMA 全 `Active/LinkUp/200` —— 包括当事节点 `.137.206`（说明当时是**瞬时中断**，非永久宕机）。
- **冒烟枪**：`.137.206` 的 **`eth8`（绑进 `bond5`）有 `rx_err=8930, tx_err=1169`——全 16 节点里独此一家**（peer `.137.106` 同一块网卡 0/0）。这块物理网卡有硬件/链路级收发错误，正好解释当时的 `No route to host 10.0.0.21`：NCCL 权重同步流量走 eth8/bond5 撞上坏链路 → 断连 → all_gather 卡死。
- dmesg 里的 PCIe `Hardware Error`（device `0x2329` = H20 的 PCIe 桥 AER）**健康节点也有**（如 `.137.106`），是这类机器的常见噪声；ECC=0 说明 **GPU 本身没坏**。

**结论：死因坐实 = `.137.206` 的 `eth8/bond5` 网卡硬件错误，不是 GPU、不是配置、不是代码。重启只需把 `.137.206` 从节点列表剔除、用其它节点顶替即可。**

### 6) 重启方案（2026-07-02 定，用户确认）：15 节点 / 120 GPU，actor 8 + rollout 7，PP16 原封保留

坏节点 `.137.206` 是 **rollout/SGLang 节点**（不是 actor；权重同步时 actor `.136.111` 要连它 → 撞坏网卡）。server 019 就是 16 节点整，**没有第 17 个节点可顶替**（gpu_keeper 里的 001/004/007 等是别的独立 Ray 集群，跨不进来）。所以去掉 1 个坏节点后**上限 = 15 节点 = 120 GPU**。

**最终拓扑（改动最小、命中根因）**：
```
actor   : 8 节点 = 64 GPU  → TP2×PP16×CP2, max-tok 8192   ← 原封不动 = §Z 已验证的 backward-OOM 修复
rollout : 7 节点 = 56 GPU  → 28 个 SGLang 引擎 (per-engine 2 GPU)
合计    : 15 节点 = 120 GPU
```
相比死掉的 060921（actor 8 + rollout 8 = 128 GPU），**唯一改动 = rollout 从 8 节点砍到 7**（踢掉坏的 `.137.206`），**actor 侧一个字不动**。§5.9 C-2 早已实测 rollout 侧余量极大（KV 占用峰值才 0.23），少 1 个 rollout 节点几乎无感。

**为什么不降到 PP8（用户提过的 TP2×PP8×CP2 + 8192，32 训 + 64 rollout = 96 GPU）**：PP8 vs PP16 在**同一 max-tok 8192** 下，每 rank 扛的层数翻倍（8 层 vs 4 层）→ 权重 + 梯度 + Adam fp32 优化器状态 per-rank ≈ **2×**，checkpoint 激活也 ≈ 2×。而 060921 那个"验证成功"的 PP16 run 软 OOM 时 **free 只剩 1.1–1.6GB**（贴边过的）→ **PP8×8192 极可能直接撞回硬 OOM**。max-tok 32768→8192 砍的是激活（有效），但 PP8→PP16 砍的是权重/优化器，两者砍的是不同的显存，不能相互替代。**能不降 PP 就别降。**

**为什么 48 卡训练没有干净拓扑**：`TP2×PP×CP=48` 且 27B 64 层要求 PP 整除 64（1/2/4/8/16/32）。PP12→64/12 不整除；TP2×PP8×**CP3**=48 → CP3 是奇数 ring，而 CP≥3 有 NCCL 死锁前科（§5.7–5.10），风险高。没有既保 PP16 又只用 48 卡的方案。

### 7) 120-GPU run（115515）也死了：SGLang 引擎崩溃 → 权重同步 barrier 55/57 卡死（2026-07-02）

15 节点/120 GPU relaunch（`20260702_115515`，job `raysubmit_PiCLZjg9BFNVSUrw`）**又死了**，但**再次是新触发器、同一个致命族**。这次彻底暴露了真正反复出现的系统性杀手。

**时间线**：
- 12:00 起，健康训练 ~2.5h，`BACKWARD_CLEAN 62% oom=0`（8192 修复再次复现✅），rollout 0/1/2 正常产出。
- **14:48:35** `.137.202` 上一个 SGLang 引擎（pid 597177）在 Qwen3.5 `linear_attn.out_proj` 的 `F.linear` 撞 `CUDA error: CUBLAS_STATUS_EXECUTION_FAILED` → **SIGQUIT**（"one child failed"）。该引擎进程死了、**没重建**。当时 GPU ECC=0、其余引擎正常，判为"暂不致命"——但预警了它会在下次权重同步缺席。
- **~15:12** 进到 `update_weights`：`init_weights_update_group` 的 TCPStore 需要 **57 个 client 会合**，但死掉的引擎（TP2=2 rank）缺席 → **只有 55/57 到齐**。
- **15:42** barrier 死等满 **1801s（30min timeout）** → `DistStoreError: Timed out after 1801 seconds waiting for clients. 55/57 clients joined`（`MegatronTrainRayActor.update_weights()` @ ip .137.78）→ job **Terminated**。

**★ 真正的系统性杀手（三次 run 的共同致命族，现在彻底看清）**：**权重同步的分布式 barrier（`init_weights_update_group` 的 TCPStore + 后续 all_gather）对缺席 client 零容错。** 任何一个 client——训练 rank 或 SGLang 引擎——中途死掉，无论因为：
- run#3：64k backward 显存挤死 rank（已被 8192 修）
- 060921：坏网卡 `.137.206` 节点掉线（已换节点）
- **115515：SGLang 引擎 CUBLAS 崩溃**（本次）

——都会让其余全体在 barrier 干等 30min 然后集体 DistStoreError 死。**换节点、修显存都只是消除单个触发器；barrier 不容错这个根本缺陷一直在，下一个偶发故障（引擎崩、rank OOM、网络抖动）会再次触发。**

**下一步方向（必须根治 barrier，不能再靠消除触发器）**：
1. **权重同步容错**：`init_weights_update_group` 前健康检查所有 SGLang 引擎，剔除/重建死引擎再同步；或让 barrier 检测到 client 缺席时快速失败+重试而非干等 1801s。要动 slime core（`update_weight_from_distributed.py` + rollout 引擎管理）。
2. **SGLang 引擎崩溃自愈**：引擎 SIGQUIT 后自动重启并重新注册进权重同步组（router 已能绕开 unhealthy 引擎做推理，但权重同步的 client 集合是启动时固定的，死引擎不会被移除）。
3. **降低单点崩溃率**：CUBLAS execution failed 可能是显存压力/瞬时错误——SGLang 侧 mem-fraction 或 batch 调低减少引擎崩溃概率（治标）。

CUBLAS 崩本身（`.137.202` GPU ECC=0、非坏节点）像是偶发瞬时错误，不是硬件坏。但**关键不是这一次为什么崩，而是"任何一次崩都会通过不容错的 barrier 杀死整个 run"**。

**失败已由 failguard 自动清 Daytona**：candidates 211 cleaned 191（+后续 pass），最终 audit total=1（那 1 个是 `state=destroying` 正在销毁中的，会自行消失）。两个监控 cron 已删。

> 详细 memory：[[tmax15k-run-060921-death-network-partition]]、[[slime-weightsync-nccl-10min-timeout-gotcha]]、[[tmax15k-full-ppo-run-20260701]]。

### 8) ★★ 深度调研：三次 run 同一根因 = 权重同步 barrier 不容错 + FT 开关被关（2026-07-02，两个 Explore agent + 日志取证坐实）

对三次连续失败（#3 133022 / 060921 / 115515）做了代码路径 + 日志双重取证，结论**确定无疑**：8192 修复成功（backward OOM 已解），三次死于**同一个机制的三个不同触发器**，而 slime 自带的解药 `--use-fault-tolerance` 被 wrapper 显式关掉了。

#### 三次死亡 = 同一机制、三个触发器

| Run | 触发器（谁掉队） | 卡在哪 | 结局 |
|---|---|---|---|
| #3 (133022) | 64k backward OOM 挤死一个训练 rank | update_weights barrier | DistStoreError 超时 |
| 060921 | `.137.206` 坏网卡 → 节点掉线 | all_gather_param barrier | NCCL watchdog 1800s 超时 |
| 115515 | SGLang 引擎 CUBLAS 崩溃 → 进程死 | `init_weights_update_group` rendezvous | **55/57 clients joined，1801s 超时** |

每次都是：**某成员死 → 权重同步 rendezvous/barrier 死等全体 → 30min 超时 → 全局 DistStoreError → job 死**。

#### 代码级根因（file:line 坐实）

1. **权重同步 client 集合启动时固定、永不刷新**：`connect_rollout_engines_from_distributed`（`slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py:299`）算 `world_size = sum(engine_gpu_counts) + 1`，基于启动时的引擎列表。引擎中途死了 `world_size` 不变，rendezvous 仍等那个死引擎。
2. **rendezvous 无条件死等**：`init_process_group` 的 TCPStore 等满 `world_size` 个 client，缺一个就等到 `SLIME_NCCL_GROUP_TIMEOUT_MINUTES`（30min）超时。**我们之前把超时 10min→30min 只是让它死得更慢，没治病。**
3. **SGLang 引擎无自动重启**：Ray actor 建时没设 `max_restarts`（`slime/ray/rollout.py:137-186`）；router `disable_health_check=True`（`rollout.py:1278`，推理会绕开死引擎，但**权重同步的成员集合固定，死引擎不会被移除**）。
4. **FT 关时无任何健康检查/恢复**：`update_weights`（`actor.py:599`）里 `recover_updatable_engines` 和健康监控都被 `if self.args.use_fault_tolerance:` 门控。

#### ★ 关键发现：slime 本来就有解药，但被关了

`--use-fault-tolerance` 门控了一整套机制：
- **`RolloutHealthMonitor`**（`slime/utils/health_monitor.py`）：后台线程每 `--rollout-health-check-interval`（默认 30s）health-check 每个引擎，发现死的就 `ray.kill` 并标记 `all_engines[i]=None`（`:160-177`）。
- **`recover_updatable_engines`**（`actor.py:599`，每次 update_weights 前）：调 `ServerGroup.recover()`（`rollout.py:280`）**重启死引擎**，设 `num_new_engines>0`。
- **NCCL 组重建**（`actor.py:615`）：`num_new_engines>0` 时**重新 `connect_rollout_engines`**，把重启的引擎重新纳入权重同步组。

**这正好能救 115515 的死法**：引擎崩 → 健康监控 30s 内发现并重启 → 下次 update_weights 前 recover → 重建 NCCL 组 → 权重同步用活引擎完成。

#### 日志实锤

115515 全程 **RolloutHealthMonitor 从未启动、recover_updatable_engines 从未调用**（grep 命中 0；那 31 个 "health" 全是 router 启动的普通 HTTP `GET /health`）——因为 FT=0。那个引擎 **14:48:35 崩**，到 **15:11:27 update_weights**，中间空了 **23 分钟**——FT 开着的话健康监控早在 30s 内就重启它了。

#### 为什么当初关了 FT

wrapper 注释只写 `# Rollout-engine fault tolerance OFF (same as the proven overfit run)`——是从 overfit smoke run 沿袭的默认，**没有任何实测证据说 FT 有害**，只是"proven run 没开过"就一直没开。这是个**未经检验的保守默认**，不是有意规避某个 FT bug。task #12 "Keep FT OFF; rely on timeout patches" 现在被推翻：timeout 补丁治不了缺席成员，FT 才是对症的。

#### 修复方案（按推荐度）

- **方案 A（首选，零改 core）：开 `USE_FAULT_TOLERANCE=1`** + `--rollout-health-check-interval 30`。直接命中 115515（引擎崩→重启→重建组），也覆盖 060921（节点掉线若引擎在其上会被健康检查发现）。风险：FT 路径在 27B/PPO/fully-async 组合下没实测过，可能有自己的坑（重启引擎要重新 load 权重、重建 NCCL 组本身也可能卡）→ **需先 smoke 验证**。
- **方案 B（补强）**：FT 只管 rollout 引擎，**管不了训练 rank 死**（#3 那种）。但 #3 的触发器已被 8192 消除，当前主要威胁是引擎崩（115515）和节点掉线（060921），FT 都覆盖。
- **方案 C（根治但重，改 core）**：让 rendezvous 容错——`connect_rollout_engines_from_distributed` 前健康检查、把死引擎从 `world_size` 剔除、barrier 快速失败+重试而非死等 1801s。工程量大，FT 已能覆盖绝大多数场景，优先级低于 A。

**当前决定（用户指定）：开 FT（方案 A）重新拉起，保留 8192 + 排除坏节点 `.137.206`。**

> 详细 memory：[[tmax15k-run-115515-death-engine-crash-barrier]]、[[tmax15k-run-060921-death-network-partition]]、[[slime-weightsync-nccl-10min-timeout-gotcha]]。

### 9) FT run（001109）活着，但暴露第 4 类触发器：`destroy_process_group` NCCL 清理卡死（2026-07-03，已加固）

方案 A 落地成 FT run `20260703_001109`（job `raysubmit_FTFLzVqarGYT2U3X`，15 节点/120 GPU，8192，排除 `.137.206`）。**FT 兜住了，run 活着推进**（已过 rollout 0→3、多次 weight-sync），但过程中观察到一个**新的、之前 3 次没暴露的**卡点，值得单独记录并加固。

#### 现象（本 run 实测）

weight-sync 耗时序列：**28.9s → 432.6s → 131.8s**。第二次（00:55–01:02）卡了 ~7min，py-spy dump 坐实卡在 `.137.141` 引擎的：
```
destroy_process_group (torch/distributed/distributed_c10d.py:2347)
  ncclCommDestroy (init.cc:2388) -> commReclaim (init.cc:2345) -> commCleanup (init.cc:2259) -> commFree
destroy_weights_update_group (model_runner.py:1988 / tp_worker.py:118 / scheduler_update_weights_mixin.py:80)
```
即 SGLang 引擎在**销毁权重同步 NCCL 通信组**时，`ncclCommDestroy` 的 comm 回收卡了 ~7min 才返回。期间该引擎被健康监控标死（engine 4-7 → None），但 **NCCL destroy 最终自己返回了**，job 没超时（7min < 30min NCCL / 60min Gloo），继续跑。

#### 为什么这次没死（和前 3 次的关键区别）

- destroy 慢（7min）**落在超时窗口内**，barrier 没触发 DistStoreError；
- FT 把死引擎标 None、健康监控正常 skip、后续 recover 重建组。
- **补丁（30/60min 超时）+ FT 一起把它兜住了**——这正是方案 A 想要的效果，符合预期。

#### 但暴露的真缺陷：SGLang 控制面 HTTP 无 timeout（潜在致命）

调研（Explore agent 追代码 + 日志取证）定位到根因不在"destroy 慢"本身，而在：
- **`slime/backends/sglang_utils/sglang_engine.py:_make_request` 的 `requests.post` 没有 timeout**（裸调用，`trust_env=True` 还会吃 proxy）。所有控制面端点共用它，含 `destroy_weights_update_group`。
- 后果：这次运气好——NCCL destroy 只是**慢**、最终返回了；但若某引擎**真死/真卡**，这个 POST 会**永久阻塞训练 rank**，连 30min NCCL 超时都等不到（那个超时只管 group 创建时的 rendezvous，管不到 HTTP 调用本身）。这是一条"无限挂"的隐藏路径，与另一团队的 sglang-dead 分析建议 #2（控制面加 timeout）指向同一处。
- 调用频率：`destroy_weights_update_group` **不是每步**都发生，只在 offload-train 的 reconnect / 引擎重启后重建组时触发（`actor.py:608` `reconnect_rollout_engines = offload_train and use_critic and not colocate`）。PPO+critic+fully-async 会命中，所以每个 rollout 周期都有暴露面。

#### 修复（已实现，2026-07-03，下次启动生效，不影响当前 run）

改 `slime/backends/sglang_utils/sglang_engine.py`：
1. `_make_request` 加**可选** `timeout` 参数（默认 `None` = 保持现状）。**关键**：`update_weights_from_tensor/disk` 是合法的分钟级权重拷贝，绝不能一刀切短超时——所以默认不变，只给控制面调用传有界值。
2. 同一处显式 `proxies={"http": None, "https": None}`：这些是**节点本地**到自己 SGLang HTTP server 的调用，永不该走 egress proxy（覆盖另一团队建议 #2）。
3. `destroy_weights_update_group` 传 `timeout=SLIME_DESTROY_GROUP_TIMEOUT_SECONDS`（默认 **120s**）。超时后与"组不存在"一样吞掉异常，交给 FT（`RolloutHealthMonitor` → `recover_updatable_engines`）重启引擎并在下次 update_weights 重建组。**把"引擎真死→永久挂"变成"快速失败→FT 恢复"。**

设计取舍：**没有**为"destroy 慢"做 update 前 drain-rollout 之类的大改——它在超时窗口内自愈、FT 兜得住，ROI 低、风险高。只做隔离的小加固（一个文件，加 timeout + 绕 proxy），把偶发触发器的**最坏路径**从"无限挂"降级为"有界失败 + 自动恢复"。

> 相关：[[slime-weightsync-nccl-10min-timeout-gotcha]]。此为第 4 类触发器（前 3 类：64k backward OOM / 坏网卡掉线 / 引擎 CUBLAS 崩），机制同族——都作用在权重同步这条脆弱路径上；根治方向仍是让这条路径整体容错，本次是其中一刀（控制面有界超时）。

## §10. eval/harbor 分母塌方：限流桶的 `asyncio.Lock` 跨事件循环 → 92% eval 任务被丢（2026-07-03，已修复+回归测试）

FT run `20260703_001109` 训练完全健康，但 rollout-9 那次 eval 的 `eval/harbor` 只有 **0.2727**，看着像模型很差。直接 load `rollout_dumps/eval_9.pt` 复原真值坐实了问题**不在模型，而在 eval 管线**：

- eval 集本应是 **89 个 TB2 任务**，但 dump 里只有 **11 个 sample**（rewards `[1,0,0,0,0,0,0,0,0,1,1]` → 3/11 = 0.2727，和 wandb 对上）。
- 日志 `06:59:28` 同一秒爆出 **82 条** `Harbor trial failed for sample index 7..88` + `Dropping failed ... <asyncio.locks.Lock object ...> is bound to a different event loop`。全 log 该错误串共 **246** 次，**全部集中在这次 eval**；训练 rollout 8/9/10/11 各有 277/286/269/272 个健康 sample，**训练路径零命中**。
- 即 **89 个任务里 82 个在打分前就因这个异常被丢**，分母从 89 塌到 11。`eval/harbor=0.27` 是幸存者偏差；诚实全集通过率 `eval_resolved_rate_full = 3/89 ≈ 3.4%`，且同样被 bug 污染，**这一列 eval 数完全不能用、也不能跟历史 0.548 比**。

#### 真根因：模块级限流桶的单把锁绑死在训练 loop 上

`slime_plugins/harbor/rollout.py::_AsyncTokenBucket`（Daytona `.create()` 限流，6/s）是**进程级全局单例**，但被**两个不同的事件循环**复用：

1. **训练**走 `FullyAsyncTrialRunner._thread_main → asyncio.run(self._loop())`（`fully_async_rollout.py:113`）——工作线程上一个**独立、持久存活**的 loop。
2. **eval** 走 `generate_rollout(evaluation=True) → slime.utils.async_utils.run(...)`（`rollout.py:113`）——另一个持久的 `AsyncLoopThread`（`async_utils.py`）。

`asyncio.Lock` 的 `_get_loop()` 语义是**首次 await 时绑定到当时运行的 loop**，之后任何**别的 loop** 再用就 `raise RuntimeError('... is bound to a different event loop')`（Py3.12 `asyncio/mixins.py` 坐实）。训练 loop 常驻、先 await → 锁绑死在训练 loop；eval loop 一旦 `async with self._lock` 就必炸。280 并发 eval trial 全走 `acquire()` 的争用慢路径（`_get_loop`），因此**整批 eval 任务全灭**。

> 注：本地极简 repro 命中率不稳定（uncontended fast-path 会跳过 `_get_loop`），但生产日志 246 次同款异常 + `_get_loop` 源码即铁证——争用场景下必然触发。

#### 修复（已实现，改插件不碰 core，下次启动生效，不影响当前 run）

`_AsyncTokenBucket` 改为**每个事件循环一把锁**（`self._locks: dict[loop_id, asyncio.Lock]`，`_lock_for_running_loop()` 按 `id(get_running_loop())` 惰性建锁）：

- 每个 loop 拿到**自己 loop 原生的锁**，跨 loop 异常根除。
- token 预算（`_tokens/_updated`）仍全局共享——限流的语义（进程级一份 create 配额）不变。两线程并发时 token 计数只是**松同步**，最坏偶发几 token 误差，对 6/s 软限流（硬限 600/min，真正的在飞并发另由 trial semaphore 卡）完全无所谓。
- `rate<=0` 仍在碰锁前短路（保持 no-op 语义）。

**回归测试**（`tests/test_harbor_rollout_async.py`，全绿 20 passed）：
- `test_async_token_bucket_shared_across_two_event_loops`：一个 loop 常驻绑锁、另一个 loop 并发 `acquire()`，断言零异常且 `len(_locks)==2`。修复前这就是 001109 的死法。
- `test_async_token_bucket_rate_zero_is_noop`：`rate<=0` 不碰锁。

#### 影响与后续

- **只影响 eval 指标可信度，不影响训练/权重**——这次训练本身一直健康。001109 已产出的 eval 列作废；下次 relaunch 起 eval 才有意义。
- 与 §9 同为"eval/rollout 并发路径的隐性缺陷"，但**不同族**：§9 是权重同步 NCCL 控制面，本条是 eval 协程调度的 asyncio 事件循环绑定。两者独立。

> 详细取证时间线见 memory [[harbor-eval-asyncio-lock-cross-loop-drop]]。

## §11. "step 0 reward 奇高、随即下跌"的真根因：全异步超采样只收最快 K 个 → 难度幸存者偏差（**Qwen3.5-27B**，run `synth3k_2820_ppo_partial_64k_20260707_085549`，2026-07-07 查实）

**模型 / 实验**：actor 从 **Qwen3.5-27B base** 冷启动（`start_rollout_id=0`），critic 从 run `20260703_001109` iter_56 warm-load 训练好的 value head。PPO（actor-critic + GAE），**KL OFF**，**n=1**，partial reward `max(terminal, 1.0*passed/total)`，`rollout_batch_size=256` / `max_concurrent_groups=280`，`sample_mode=trajectory_retokenized`。训练池 = synth-2820（2820 合成 terminal 任务），eval 池 = TB2-89 留出集。wandb run `5l5r74n7`。

### 反常现象（用户观察，跨 run 反复出现）
`rollout/rewards` 在 **step 0 = 0.79** 冲高，step 1 立刻掉到 0.71、step 2 到 0.68，之后长期在 0.67–0.72 徘徊。**同样的形状在 TB2-overfit、Tmax 训练里也反复出现**——正因为跨数据集复现，怀疑是系统性伪影而非"第一轮数据碰巧简单"。

### 先毙掉两个想当然的解释
- **❌ KL 惩罚 step0 后启动**：配置 `KL OFF`，dump/日志实测 `rollout/kl ≈ 0.0002`（每一步都是）、`train/ppo_kl ≈ 1e-5`。根本没有 KL 惩罚项，此假说与数据矛盾。
- **❌ "只是第一轮任务简单"**：**只对了一半**。任务构成确实在变，但真相精确得多，且真凶是**采样机制**（见 §11.3），不是数据本身"碰巧"。

### §11.1 伪影一：wandb 的 reward ≠ dump 里的真实 reward（口径差）
直接 load `rollout_dumps/{0,1,5,11}.pt` 复原真值：

| rollout | wandb `rollout/rewards` | dump 内均值 | dump 内 reward 取值 |
|---|---|---|---|
| 0 | **0.790** | **0.441** | 只有 0.0 / 1.0（0% partial） |
| 1 | 0.707 | 0.320 | 只有 0.0 / 1.0 |
| 5 | 0.715 | 0.297 | 只有 0.0 / 1.0 |
| 11 | — | 0.302 | 只有 0.0 / 1.0 |

**机制**：dump 在 `slime/ray/rollout.py:719 _save_debug_rollout_data` 保存，发生在 `_convert_samples_to_train_data`（`:724`）**之前**；wandb 记录（`data.py:206`）来自转换**之后**的训练数据。转换里两步抬高均值：
1. `_drop_zero_reward_groups_for_static_dp_alignment`（`rollout.py:937`）丢掉部分 0-reward 组做静态 DP 对齐；
2. partial-reward shaping `trajectory_grouped_progress_rewards`（`slime_plugins/harbor/partial_reward.py`）：`max(terminal, passed/total)`。

⇒ **dump 存的是原始 terminal 0/1，wandb 画的是 shaping 后的 partial reward。** 两者都在 step0 后下降，所以"下降"是真的；但要归因"为何降"，必须看 dump 的原始 pass 率。

### §11.2 伪影二（决定性对照）：同一批任务上策略并没有退化
**固定队列对照**：锁定 step 0 出现的 **249 个任务**，在后续每一步只统计这批任务的原始 terminal reward：

| rollout | 固定队列 reward（同一批任务） | 全体 reward（wandb 那条曲线） |
|---|---|---|
| 0 | 0.454 | 0.454 |
| 2 | 0.398 | 0.322 |
| 4 | **0.446** | 0.368 |
| 8 | **0.440** | 0.364 |
| 11 | 0.414 | 0.307 |

**同一批任务上 reward 稳在 ~0.40，而全体一路跌到 ~0.31。差值 100% 来自批次构成变化，不是策略变差。** step0 vs step11 的 99 个共同任务里 20% 变好 / 15% 变坏（净 +0.05）。

### §11.3 ★ 真根因：全异步"超采样 + 只收最先完成的 256 个" → 难度幸存者偏差
每一步**首次进入批次**（第一次被接收）的任务难度：

| rollout | 新任务 reward | 新任务 turns | 新任务 tokens |
|---|---|---|---|
| 0 | **0.454** | 10.7 | 7,342 |
| 1 | 0.270 | 16.5 | 12,243 |
| 2 | 0.227 | 27.2 | 20,868 |
| 3 | **0.143** | 27.6 | 19,780 |

对照：**step0 接收的任务**均 10.7 轮 / 7.3k tok / reward 0.454；**首次出现在 step≥3 的任务**均 **27.9 轮 / 19.9k tok / reward 0.232**（2.6× 长、2× 难）。

机制链（`slime_plugins/harbor/fully_async_rollout.py`）：
1. `HarborFullyAsyncRolloutWorker._loop`（`:128`）一次并发放飞 `max_concurrent_groups=280` 个 group；
2. `_generate_rollout_async`（`:307` `while len(output_groups) < target_groups`）在收满 **256 个最先完成**的 group 时冻结这一批，仍在跑的丢弃/延后（`discarded_completed_groups`）；
3. **最先完成 = 轨迹最短 = 最简单**。step 0 系统性"撇油"，只带走跑得最快的容易任务；
4. 慢/难/长任务这一步没跑完、游标又前进到新任务，于是**难任务在后续步骤持续累积**，批次难度单调上升 → 全体 reward 单调下滑。

**佐证 A（长度爆炸与 reward 同步）**：`rollout/response_lengths` 7215→10329→12561（+74%），turns 10.6→18.4，`truncated` 0→9%，与 reward 下降完全同步。

**佐证 B（组内 pass/fail 长度比）**：step 0 失败轨迹只比成功长 1.24×；step 2+ 失败膨胀到 **1.5–2.1×**（14–16k tok、20–24 轮），而成功轨迹始终短（~7–8k tok、~11 轮）——即模型在难任务上"瞎逛/刷长度"，不是在已会的任务上变差。

**佐证 C（工作集回收）**：12 步只消费 **584 个不同任务**（而非 sorted-sequential 应有的 ~2820），且任务集在 **lag-7 相关性最高（45%）**——慢任务在一个 ~7 步窗口里被反复重抽，形成滚动的"慢任务工作集"。（`requeued_groups` 在全异步路径未启用，游标 `sample_offset` 线性推进，见 `data_source.py:57`；584 的塌缩纯由超采样撇油造成。）

### §11.4 放大器：n=1 让每步 reward 噪声极大
`n_samples_per_prompt=1` ⇒ 每任务 reward 是**单次伯努利抽样**。被抽到 ≥3 次的 485 个任务里，**65% 的 reward 在 pass↔fail 间翻转**。固定队列上那 ~0.05 抖动完全在噪声内。**此设置下，train-batch reward 的逐点波动本质上不能解读为策略质量。**

### §11.5 为什么 TB2 / Tmax 也复现（回答"耐人寻味"）
病根在**基础设施层、与数据无关**，凡满足以下条件的 run 都会中招：
- 全异步 + 超采样**只收最快 K 个**（撇油偏差）；
- 冷启动策略下**任务"完成速度"与"难度"强相关**（简单任务先返回）；
- 把**动态构成的 train-batch** reward 当成固定评测集在看；
- **n=1** 放大噪声。

TB2-overfit、Tmax 共享同一套 Harbor 全异步 rollout 管线，故现象复现。**step 0 的高点是"撇油撇出来的假高点"，不是策略的真实基线水平。**

### §11.5.5 ★ 实测撇油比远不止名义 1.09×：持久 worker 的 backfill 把它放大到 3.9×（2026-07-07 launcher 日志坐实）
配置上看撇油似乎很轻：`max_concurrent_groups=280` / `rollout_batch_size=256` → 名义超采样比仅 **280/256 = 1.09×**。但从 launcher 日志把每步的 harbor async 计数抠出来后，真实比例吓人：

| 指标（每步） | 早期步 | 中期步 | **后期步** |
|---|---|---|---|
| `num_completed_group_tasks`（本步实际完成的组数） | 260, 264, 266 | 292, 337, 368 | **843, 1000** |
| `num_discarded_completed_group_tasks`（完成后被丢弃） | 0, 0, 0 | 72, 105, 133 | **580, 735** |
| `num_accepted_groups`（真正进训练） | 256 | 256 | 256 |

后期步为了收满 256 个，**实际完成并丢弃了 580–735 个组** → 有效超采样比冲到 **3.3×–3.9×**，是名义值的 3~4 倍。

**根因是 fully-async worker 的 backfill**（`slime_plugins/harbor/fully_async_rollout.py:128` `while len(active_tasks) < max_concurrent_groups(): 补新组`）：它是**进程级持久** worker（`_get_global_worker`，跨 rollout step 不销毁），一个组完成就立刻 backfill 一个新的进来。后期步 collection 窗口很长（慢任务堆积），持久 worker 在这个长窗口里 churn 了近千个组，只留下**最先完成的 256 个**。⇒ 撇油偏差比 §11.3 从名义比推测的严重得多；这也是为什么"仅把 `max_concurrent_groups` 降到 256"只能削弱、不能根除——accept-first-256 + backfill 这个组合本身就偏向快任务。

> 复现命令：`grep -hoE "harbor/(num_completed_group_tasks|num_discarded_completed_group_tasks|num_accepted_groups)['\"]?: ?[0-9]+" <run>/logs/*.log`。

### §11.6 判定真实趋势 & 修法
- **只信留出集 eval**（唯一不受 train-batch 构成污染的信号）：本 run `eval_9.pt`（TB2-89）terminal pass = **0.342**（26/76）。建议把 eval 曲线作主看指标，train reward 仅作过程诊断。
- **消除假高点（★ 主药，直接治病根）**：去掉 accept-first-K 撇油——`harbor_fully_async_max_concurrent_groups` 设为 = `rollout_batch_size`（不超采样），或改成"接收全部完成组"。这样每步不再优先收走"最快完成=最简单"的任务，批次难度分布无偏，step0 假高点消失。**下一个 run 就采用这个改动**（`max_concurrent_groups = rollout_batch_size`）。
- **正交改进：提高 n（n>1）**——注意它 **不修** step0 假高点（假高点是批次构成问题，不是噪声问题），但独立地改善训练信号质量。这里要区分两种估计器，避免误导：
  - **当前是 PPO（actor-critic）**：advantage 的基线来自 **critic（value function）**，**不做**组内归一化（`turn_grouped_reward.py:54 _should_normalize` 仅对 `{grpo,gspo,reinforce_plus_plus_baseline}` 为真，PPO 恒 False，partial-reward hook 在 PPO 下直接 `return raw, raw`）。所以 PPO 用 n=1 也能训。提高 n 的收益是：①每任务 reward 从"单次伯努利抽样(std≈0.49)"变成 n 次平均(std≈0.49/√n)，喂给 critic 的回归目标和 advantage 更干净；②小样本对照/eval 更可信。
  - **若改用 GRPO/GSPO**：n>1 **不是锦上添花而是必需**——组内 advantage = `(r_i − 组内均值)/组内std`，n=1 时组里只有一个样本 → advantage 恒为 0 → 没有梯度。这才是"让分组归一化有意义"的确切含义。（原 §11.6 把这条笼统写在 PPO run 下，易误导，特此更正。）
- **诊断手法**：并排画 `rollout/response_lengths` 与"新任务占比"，与 reward 下降对齐即可坐实撇油偏差。
- **附带健康提示**：`rollout/explained_variance ≈ -4.7`，warm-load 的 critic 在 synth-2820 上标定很差（value 解释力为负），PPO advantage 质量存疑，值得单独排查。

> **2026-07-07 实际改动（relaunch 用，已落地）**：因 §11.5.5 坐实"仅降 `max_concurrent_groups` 到 256"治不了本（backfill 仍偏向快任务），改为**退出 fully-async、切到 `standard` rollout 模式 + 轻 oversample**。standard 模式（`slime_plugins/harbor/rollout.py:231-291`）每步新建任务集、一次发 `over_sampling_batch_size` 个、收最先完成的 256、`finally` 里 cancel 慢尾、**不 backfill**（`HARBOR_ROLLOUT_MODE` 分派见 `run_qwen35_9b_terminal_bench2_64h20.sh:334`）。撇油比从实测 3.9× 压到 **288/256 = 1.125×**（留极小余量防单个 60 轮/44k-token 怪物任务卡死单步）。**零 `.py` 改动，纯配置**：
>
> | 文件 | 改动 |
> |---|---|
> | `run_synth3k_2820_96gpu_27b_daytona_ppo_partial_64k_fully_async.sh` | 新增 `HARBOR_ROLLOUT_MODE=${…:-standard}`；`OVER_SAMPLING_BATCH_SIZE` 280→**288**；`WANDB_GROUP` 后缀 `oversample280`→`standardOS288` |
> | `start_synth3k_2820_96gpu_27b_daytona_ppo_partial_64k_tmux.sh` | `HARBOR_ROLLOUT_MODE` 从写死 `fully_async` 改为可覆盖、默认 `standard`（`:124`） |
> | `harbor_synth3k_2820_96gpu_27b_daytona_ppo_partial_64k.yaml` | 加注释：standard 模式下 `harbor_fully_async_max_concurrent_groups` 不再被读取（保留供 A/B 回退 fully_async） |
>
> 其余全部不变（n=1、PPO、KL off、64k、NUM_ROLLOUT=22、critic warm-load 001109 iter56），与上个 fully-async run 干净 A/B。**发射后核心看点**：①step0 与后续步 reward 落差是否消失/收窄；②固定队列 reward 与全体 reward 是否收敛一致；③`rollout/response_lengths` 是否不再随 step 单调爆炸；④`perf/step_time` rollout 段是否明显变长（standard 被最慢 group 卡住是已知代价，若不可接受则调高 OVER_SAMPLING 或回退 fully_async）。
>
> 备注：standard + OS288 **并非零撇油**（仍是"收最先完成的 256"），只是把比例降到 1.125×；真正零撇油需 `over_sampling_batch_size=256`（发 256 等全部完成、一个不砍），代价是单步被最慢任务完全支配。此处按"轻 oversample 折中"取舍。

> 复核脚本模式：固定队列对照 + 新任务/重复任务难度对比（load `rollout_dumps/*.pt`，按 `metadata.harbor_task_name` 聚合）。详见 memory [[step0-reward-survivorship-bias]]。同类"dump 复原真值推翻 wandb 表象"的手法亦见 §10、[[harbor-eval-asyncio-lock-cross-loop-drop]]。

### §11.7 本 run（standardOS288）的 timeout 全景（2026-07-08 梳理，Qwen3.5-27B）

这个 run 的超时是**分层嵌套**的，从内到外一层套一层。一句话记住嵌套：**一个任务从发出到判分，最外层被 Ray 远程调用的 90min 兜住；里面 agent 跑 60min/60 turns、判分 15min；序列长度锁死 64k；即便 slime 侧没主动删，Daytona sandbox 也会在 90+10min 后自杀；训练侧用 30/60min 的 NCCL/Gloo 墙防 hang。** 所有值均可用同名 env 覆盖，下面列的是本 run 实际生效值 + 出处。

**一、单个 trial（任务）内部——决定一个 sandbox 能跑多久**

| 层 | 值 | 含义 / 出处 |
|---|---|---|
| `agent.override_timeout_sec` | **3600s（60min）** | Terminus2 agent 跑单任务的墙钟上限（yaml `:63`） |
| `agent.max_turns` | **60** | 单任务最多 60 次 LLM 调用（agent loop，yaml `:66`） |
| `verifier.override_timeout_sec` | **900s（15min）** | 跑 `tests/test.sh`（pytest→CTRF）的上限（yaml `:82`） |
| `harbor_trial_remote_timeout_sec` | **5400s（90min）** | Ray 远程 trial 调用总墙钟，套在 agent 60 + verifier 15 之外，留 15min 余量（yaml `:132`） |

单个 LLM call **无独立 per-call timeout**——靠 SGLang 默认 + agent 的 60min 总预算兜住。agent 上下文超 `proactive_summarization_threshold=20000` tokens 触发主动摘要（yaml `:69`）。

**二、Context 墙（5 道）——决定序列长度，全 64k**

| 变量 | 值 | 出处 |
|---|---|---|
| `SEQ_LENGTH` / `max-position-embeddings`（Megatron） | 65536 | wrapper `:44` |
| `HARBOR_MAX_TRAIN_TOTAL_TOKENS` | 65536 | wrapper `:45` |
| `ROLLOUT_MAX_CONTEXT_LEN` | 65536 | wrapper `:46` |
| `SGLANG_CONTEXT_LENGTH` / `SGLANG_MAX_PREFILL_TOKENS` | 65536 | wrapper `:47-48` |
| agent `max_context_tokens`（Terminus2 内部，第 4 道墙） | 64000 | yaml `:80` |

外 4 道 65536、agent 自留 64000 头。`MAX_TOKENS_PER_GPU=8192`（wrapper `:52`）是 §Z 坐实的 PP16×CP2 backward-OOM 解药，非 context 墙但同族。

**三、Daytona sandbox 生命周期——防泄漏/防烧钱的后备（watchdog 之外的第二道保险）**

| 参数 | 值 | 作用 / 出处 |
|---|---|---|
| `environment.delete` | `true` | 正常结束的 trial 立即删，不等 auto（yaml `:85`） |
| `auto_stop_interval_mins` | **90min** | 空闲 90min 自动 stop（> agent 60min 预算，不误杀正常 trial，yaml `:96`） |
| `auto_delete_interval_mins` | **10min** | stop 后 10min 自动删（yaml `:97`） |

兜底逻辑：即便某 trial 的 `stop()` 没跑到，sandbox 最多活 90+10min 就被 Daytona 自清。与 cron watchdog（FAILED/STOPPED/UNKNOWN×5 → 按 `slime-` 前缀 purge）互为冗余。

**四、分布式 / NCCL / Gloo——防训练侧 hang 死（都是踩过坑的修复，见 §Z.2/§8/§9）**

| 变量 | 值 | 修的坑 / 出处 |
|---|---|---|
| `--distributed-timeout-minutes` | **60min** | NCCL collective watchdog；slime 默认仅 10min，critic PP forward 慢会误杀（base wrapper `:482`） |
| `SLIME_GLOO_GROUP_TIMEOUT_MINUTES` | **60min** | CRASH#2：65-rank 权重广播 gloo barrier（本 wrapper `:97`；core 默认见 `slime/utils/distributed_utils.py:34`） |
| `SLIME_NEW_GROUP_TIMEOUT_MINUTES` | **30min** | CRASH#1：NCCL new_group（torch 原生 fallback 才 10min，`reloadable_process_group.py:19`；30 由 base wrapper `run_tb2_full89_…_fully_async.sh:127` 兜底） |
| `ROLLOUT_HEALTH_CHECK_TIMEOUT` | **120s** | step0 冷编译 GDN kernel 时别把 rollout 误判为死（base wrapper `:438`） |

**五、Coordinator / 集群级**

| 参数 | 值 | 出处 |
|---|---|---|
| `harbor_eval_total_timeout_sec` | **3600s** | 整个 TB2-89 eval 总墙钟，到点 cancel 未完成的、按已完成算 pass-rate（yaml `:136`） |
| `harbor_trial_worker_startup_timeout_sec` | 120s | yaml `:128` |
| `harbor_trial_worker_healthcheck_timeout_sec` | 30s | yaml `:129`（`harbor_kill_unhealthy_trial_workers: false`，超时不杀 worker） |
| `RAY_CLUSTER_WAIT_SECONDS` | **1800s** | 等齐 15 节点（tmux `:130` 覆盖 base 默认 240） |

**★ standard 模式下 inert 的 timeout（本 run 不生效，保留供 A/B 回退 fully_async）**：`harbor_fully_async_shutdown_drain_timeout_sec=30` / `harbor_fully_async_shutdown_join_timeout_sec=45`（yaml `:138-139`）——仅 fully_async 的持久 worker 优雅退出用。本 run 是 standard 模式（见 §11.6），它俩不被读取，符合改动意图。

## §12. standardOS288 run 完成复盘 + critic 诊断 + 续训 4 epoch（critic lr 10×）（**Qwen3.5-27B**，2026-07-08）

### §12.1 run 正常 SUCCEEDED，但 22 步无学习信号（撇油修复成功 ≠ 学到东西）
`20260708_relaunch_standardOS288`（job `raysubmit_yC2TJLi2DRvRu6Yb`，wandb run `xqk4vuik`）跑满 22 步（2 epoch）正常结束，ckpt 存到 iter_21（actor+critic）。
- **撇油修复 ✅（本次主目标达成）**：每步实际发起 293–299 trial（`slime-rN-*` 命名，见 [[daytona-sandbox-label-not-name-prefix]]），收 256，撇油比 **1.16×**（对比旧 fully-async 的 3.9×）。dump 真值 reward 全程 **0.31–0.43** 平稳随机游走，step0 不再虚高。
- **但零学习信号**：22 步 train reward 线性斜率 t=−0.28（无显著趋势）；步间 std(0.032)≈单步 SEM(0.030)，波动全是 n=1 采样噪声。
- **两次 TB2 eval**：`eval_9`=0.304(n=135, 88 uniq tasks) → `eval_19`=0.244(n=131)。差 −0.059，**z=1.09 不显著**（TB2-89 样本太小，SEM≈0.04），方向不利但统计上无法断定变差。

### §12.2 ★ critic 诊断：explained_variance 全程为负 = PPO 学不动的机制性根因（离线解析 .wandb 坐实）
dump 是训练前 Sample（无 value/returns），critic 指标只在 wandb。在线 run 随集群关停，但**本地 `.wandb` 事件文件存活**——用 `wandb.sdk.internal.datastore` 离线解析 `run-20260708_063937-xqk4vuik/run-xqk4vuik.wandb`（history record 的字段在 `nested_key`，不是 `key`），提取全 22 步：

| step | EV | values | returns | advantage |
|---|---|---|---|---|
| 0 | −4.67 | 0.86 | 0.77 | −0.09 |
| 7 | −4.71 | 0.81 | 0.72 | −0.09 |
| 14 | −3.63 | 0.73 | 0.76 | +0.03 |
| 21 | **−1.84** | 0.93 | 0.70 | −0.22 |

**判读**：
- `rollout/explained_variance` **0/22 步为正**（−4.67→−1.84，均 −3.69）。EV<0 意味着 critic 的 value 预测**比常数基线还差** → advantage = returns−V 基本是噪声（数据里 adv 符号频繁翻转、±0.22 无规律）→ 策略梯度方向随机 → **这就是 22 步 reward 零斜率、学不动的机制性解释**。
- **但 EV 单调回升、无停滞**（22 步改善 61%，后期稳定在 −1.8~−2.6）：warm-load 的 value head（来自 001109 iter56，异分布）正在 synth-2820 上重新标定，只是没标定完。**这直接支持"续训有价值"**——EV 大概率会继续爬向 0，一旦转正 advantage 才有意义。
- KL≈0.0002（确认 KL off，非 KL 惩罚导致）。

### §12.3 续训方案：从 iter_21 续 4 epoch（+44 步→NUM_ROLLOUT=66），critic lr 提到 10×
**根因是 critic 标定慢**，故不动 actor、**只给 critic 加速**。原 run actor/critic **共用** `--lr 5e-7`（没用 `--megatron-config-path`），对"从异分布重标定 value head"的 critic 偏慢。

**改动（3 处，零 core 修改，已 bash -n + 真实 parser 双验证）**：
| 文件 | 改动 |
|---|---|
| `megatron_roles_synth3k_criticlr10x.yaml`（新建） | per-role config：actor lr=5e-7、**critic lr=5e-6（10×）**（`min_lr` 同值，`lr-decay-style constant`） |
| `run_qwen35_9b_terminal_bench2_64h20.sh` OPTIMIZER_ARGS | opt-in 追加 `--megatron-config-path`（仅当 `MEGATRON_CONFIG_PATH` 设置，默认行为不变） |
| `start_synth3k_..._tmux.sh` env 白名单 | 加 `MEGATRON_CONFIG_PATH` 透传 |

机制:`create_training_models` 调 `parse_megatron_role_args(role="critic")` 把 `overrides.lr` 只作用到 critic 优化器(`slime/utils/arguments.py:1546-1626`、`placement_group.py:220-251`),actor 保持顶层 `--lr`。10× 比例对齐 slime 官方 `tests/test_qwen3_4B_ppo.py`。

**续训启动关键参数**(原地 resume,复用 `RUN_TAG=20260708_relaunch_standardOS288` 让 `SAVE_DIR` 非空 → wrapper `:329` 自动 `--load` 指向它 → 从 iter_21/rollout 22 续;`harbor_data_source_state_21.pt` 已确认存在,任务游标无缝接续):
- `NUM_ROLLOUT=66`(绝对目标步数,非增量;22+44)
- `MEGATRON_CONFIG_PATH=.../megatron_roles_synth3k_criticlr10x.yaml`
- `WANDB_GROUP` 后缀改 `-6epoch-resume-criticlr10x`(新开 wandb run,不续 xqk4vuik)
- 其余不变(actor lr 5e-7、n=1、PPO、KL off、64k、standard OS288)
- **续训成功判据**:首个新 dump 是 `rollout_dumps/22.pt`(非 0.pt);wandb 新 run step 从 22 起;重点看 **EV 是否随续训转正** + reward 在 epoch 3-6 是否出现真实斜率。

> 备注:原 run 结束后 Ray 集群已关停,续训需重走 [[harbor-multinode-relaunch-tmux-gotcha]] 的 15 节点启动流程(launcher+14 worker,`RESTART_TMUX=1`,`STAGE_HF_CKPT_TO_TMP=1` 重 stage 52G)。监控复用已修好的 `watchdog_synth3k_standardOS288.py`(已改 `--label harbor_managed=1`,见 [[daytona-sandbox-label-not-name-prefix]]),需更新其 `JOB_ID` 常量为续训新 job id。

## §13. 续训四连败复盘 + 转向 Qwen3.6-27B 全新 run + 冷启动加速（2026-07-08~09）

### §13.1 "从 iter_21 续训 +4 epoch"连续 4 次失败——resume 改总步数的连锁坑
§12.3 的续训方案（从 iter_21 加载 optimizer state + critic lr 10× + NUM_ROLLOUT 22→66）实际**连续崩了 4 次**，每次一个新坑，最终放弃续训改走全新 run：

| # | 崩因 | 根因 | 修法 |
|---|---|---|---|
| 1 | job 启动即崩 | `WANDB_GROUP` 142 字符 > wandb 128 上限 | 缩短命名 |
| 2 | ckpt 加载崩 `CheckpointingException` | critic-lr 的 `--megatron-config-path` 给 critic 生成独立 args，改变分布式优化器分组 → 加载 iter_21 的 `optimizer.distributed.dp_group_idx`（2.5B vs 1.87B）shape mismatch | 放弃 critic-lr（去掉 config） |
| 3 | ckpt 加载崩 `AssertionError` | 干净 resume 加载 optimizer state 时校验 lr scheduler 的 total-iters：iter_21 存的 22×256=5632 ≠ 现在 66×256=16896 | 加 `--override-opt_param-scheduler`（constant lr 下安全，只跳过 total-iters 校验） |
| 4 | **熬过加载但从头重训** | override_opt_param_scheduler=True 虽让 scheduler 校验通过，但 `start_rollout_id` 从 0 起（非 22）→ 覆盖写 iter_4、把 `latest_checkpointed_iteration.txt` 从 21 改成 4 | **杀 job 止损**：恢复指针=21、删 iter_4、iter_21 完好（413G/130文件未损） |

**死结教训**：`--override-opt_param-scheduler` 与"续训改总步数"存在根本冲突——不加它 scheduler 断言崩（#3），加了它连带 rollout 计数归零、从头重训（#4）。**要在 slime v0.3 里"resume 一个 ckpt 且改 num-rollout"没有干净解**（至少 raw+critic-lr 组合下）。教训：与其在续训上死磕，不如全新 run。每次失败都清理了 ray/tmux/stop-marker/sandbox（见 [[daytona-sandbox-label-not-name-prefix]]），iter_21 原始成果始终未损。

### §13.2 转向：Qwen3.6-27B 全新 5-epoch run（critic 冷启 + lr10x，2026-07-09）
放弃续训，改用**同一份 synth-2820 数据 + 换 Qwen3.6-27B 从头训**。**全新冷启动没有 optimizer state 要加载 → §13.1 的 #2/#3/#4 坑全部消失**，critic lr 10× 这次能干净生效（这是当初放弃它的唯一障碍）。
- **Qwen3.6-27B 架构与 3.5-27B 完全一致**（`model_type=qwen3_5_text`：64L/5120/24h/4kv/17408/vocab248320/head_dim256），复用 `scripts/models/qwen3.5-27B.sh` spec 不变。
- **HF→torch_dist 转换**（raw 模式必需，`--ref-load` 要 Megatron dist ckpt）：新建 `scripts/convert-qwen3.6-27B.sh`（仿 `convert-qwen3.5-35B-A3B.sh`，8卡 torchrun `tools/convert_hf_to_torch_dist.py`），~7min 转出 51G，产物 `Qwen3.6-27B_torch_dist/release/` 含 `.distcp`+`.metadata`、`latest_checkpointed_iteration.txt=release`。
- **critic 冷启**（`model.py:_critic_output_layer_needs_reinit`）：新 load-root `synth3k_ppo_qwen36/Qwen3.6-27B_base` symlink，**不建 `_critic` symlink** → critic backbone 从 3.6 base 加载、value head 因 base 无 value 层自动重初始化（日志 `model.py:120 Will reinit`）。`NUM_CRITIC_ONLY_STEPS=2` 前 2 步只训 critic 预热。`_isolate_critic_checkpoint_dirs`（placement_group.py:150）在无 `_critic` 目录时保持 critic 与 actor 同 `--load`，不 assert-crash。
- **配置**（job args 核实）：`NUM_ROLLOUT=55`（5ep）、`SAVE_INTERVAL=10`、`eval_interval=None`（eval 关闭）、`num_critic_only_steps=2`、critic lr 5e-6（10× actor 5e-7，via `megatron_roles_synth3k_criticlr10x.yaml`）、`override_opt_param_scheduler=False`、ppo、KL off。RUN_TAG `20260709_qwen36_synth3k_5ep_criticlr10x`，wandb group `q35-synth3k-qwen36-5ep-criticlr10x`。job `raysubmit_DWc7Am1i13fBjCgh`。
- **eval 关闭的坑**：`EVAL_INTERVAL=""` 被**两层** wrapper 的 `${EVAL_INTERVAL:-10}` 重置回 10（`:-` 对空串也取默认）。改成 `${EVAL_INTERVAL-10}`（仅 unset 才取默认，保留显式空值）——**两处都要改**：`run_synth3k_..._fully_async.sh:77` + `run_tb2_full89_..._fully_async.sh:249`。改完 job args 确认 `eval_interval=None`。启动了 3 次才定位到第二层。

### §13.3 ★ 冷启动慢的拆解 + torch_dist staging 加速（2026-07-09）
27B 冷启到首个 rollout dump ~40-50min，拆解：①train_async.py 冷启（import megatron/sglang/torch 全栈）~17min，固定；②SGLang 引擎+tokenizer+TrialWorker ~13min；③**actor 从共享 FS `shared-storage` 直接读 51G torch_dist base**——可优化瓶颈（共享 FS 读带宽，拥塞时 7-15 MB/s）。
- **解药**：`STAGE_TORCH_DIST_CKPT_TO_TMP=1`（`run_qwen35_9b:246` rsync base→本地/tmp 再重指向，`.slime_stage_complete` 幂等）→ actor 从本地 SSD 读，远快于共享 FS。
- **synth3k 默认关着**（`run_synth3k_..._fully_async.sh:91` 默认 0，注释 `:25-27`）：因原 3.5 **warm-load critic** 要 `<load>_critic` symlink 在共享路径解析，stage 到 /tmp 会破坏。
- **critic 冷启时此顾虑消失**（无 `_critic` symlink）→ **下次 3.6 冷启 run 应带 `STAGE_TORCH_DIST_CKPT_TO_TMP=1`**。warm-load critic 的 run 则**不能**开（互斥）。详见 memory [[harbor-torch-dist-staging-speedup]]。

> **★ 本次启动踩的坑：staged 目录名误导 = `Qwen3.5-…`，但内容其实是 3.6（2026-07-09 坐实）**。开 staging 重启后，`--ref-load` 指向本地 `/tmp/slime-harbor-model-cache/Qwen3.5-27B_torch_dist_pp8_timeout60`——**名字带 `Qwen3.5`，第一眼像是 stage 错了模型/要训成 3.5**。核查经过：
> - 起因：`STAGED_TORCH_DIST_CKPT=/tmp/slime-harbor-model-cache/$(basename "${PRETRAINED_CKPT}")`（`run_qwen35_9b:52`）。PRETRAINED_CKPT 是 symlink `synth3k_ppo_qwen36/Qwen3.6-27B_base` → 实体 `Qwen3.6-27B_torch_dist`；但本地那个 `Qwen3.5-…pp8_timeout60` 目录是**上个 3.5 run 遗留的旧 staging 目录**，rsync `-a --delete` 把 3.6 内容**覆盖**进了这个旧名目录（basename 解析/复用到了历史名）。→ **目录名 stale，内容是新的 3.6**。
> - **不能靠 mtime/文件名判断**（mtime 一致可能是 rsync 保留源 mtime 的巧合）。**权威核验 = 比对权重字节 sha**：本地 staged 的 `release/__0_0.distcp` sha256(前200MB)=`ae82138e…` **等于** 3.6 源、**不等于** 3.5 源（3.5 源同分片 sha=`3672cd51…`，大小虽都 4335922174 但内容不同）。→ 坐实 stage 的是 3.6 ✅，`--ref-load` 加载正确。
> - **教训**：staging 目录名来自 basename 且会复用/覆盖旧同名目录，**名字不可信**；换模型后要么先 `rm -rf` 旧 staged 目录再启动、要么用内容 sha 核验，别被 `Qwen3.5-…` 的名字吓到或误判。

**本 run 核心看点**：冷启动 critic + 10× lr + 2 步预热，能否让 `explained_variance` 比 3.5 run（卡在 −4.7→−1.8 从未转正，见 §12.2）更快转正 + reward 出现真实上升斜率。首个 dump 是 `0.pt`（全新 run，非续训）。job `raysubmit_YiMMjjP18iuewhzA`（staging 版，13:07 重启）。

### §13.4 run 完成复盘：EV 转正成功 + reward 平（数据瓶颈）+ ★ actor ckpt 被 critic 覆盖丢失（2026-07-10）

**训练 SUCCEEDED**（2026-07-10 07:48，55/55 步，job `raysubmit_YiMMjjP18iuewhzA`）。两条科学结论坐实：

**① 冷启 critic + 10× lr → EV 成功转正**（回答 §12.2 悬案）。`rollout/explained_variance` 逐步（离线解析 .wandb，`rollout/step` 配对）：
```
step 2:  -39.19   step 7: -11.72   step13: -3.43   step18: -0.96   step44: -0.002
step 4:  -23.26   step10:  -7.29   step16: -2.05   step27: -0.35   step50: +0.010 ← 首个正值
```
峰值 +0.032。**对比 3.5 warm-load critic 卡在 −4.7→−1.8 磨 21 步、从未沾 0**：这次冷启起点虽深得多（−39），但 14 个 actor-training step 就追平 3.5 终点、~48 步干净转正。**结论：critic 冷启（value head 重初始化，无 optimizer state 冲突）+ critic lr ≈ 10× actor lr，在学习速度（数量级）和最终校准质量（EV 转正 = value 优于常数基线）上全面胜过 warm-load。**

**② 但 critic 修好 ≠ reward 提升**。`rollout/rewards` 全程 53 步线性回归 **slope≈−0.00000, t≈+0.07（教科书级水平线）**，mean=0.720，first8=0.726 vs last8=0.728，**4.6 epoch 零上行、零过拟合、零下降**。用户中途问过"reward 是不是一直下降"——答案是统计上平（近因错觉：step18=0.558/29=0.624/47=0.698 都是孤立低谷、随后都反弹）。**判 reward 趋势必须用全序列回归+t 检验，不能看最近几步。** critic-lr 实验成功**排除了"critic 没学好"假设**，把 reward 瓶颈定位到**数据本身**（27B 在固定难度分布的 synth-2820 上 2-3 epoch 摸到上限）。**下一步该换/扩数据（更难更多样），而非继续调 critic/lr。** 详见 memory [[critic-coldstart-lr10x-ev-positive]]。

> **★★ 严重事故：actor checkpoint 被 critic 原地覆盖，整个 run 的 actor 权重全丢（2026-07-10 坐实 + 已修复）**
>
> 训完发现 **6 个存点（iter_9/19/29/39/49/54）没有一个是 actor**——用 DCP 读 `.metadata` 看 `output_layer.weight` shape，**全部是 `[1,5120]`（critic value head）**，而非 actor 的 `[248320,5120]`（词表投影）。iter_9 就已如此 → 从第一个存点起 actor 就被覆盖。**训了几天的 actor 权重不可恢复，只剩 critic + rollout_dumps。**
>
> **判定命令**（存 ckpt 后必做一次）：
> ```python
> from torch.distributed.checkpoint import FileSystemReader
> md = FileSystemReader("<iter_dir>").read_metadata()
> print(md.state_dict_metadata["output_layer.weight"].size)  # actor=[vocab,hidden]; critic=[1,hidden]
> ```
>
> **根因**：`slime/ray/placement_group.py` 的 `_isolate_critic_checkpoint_dirs`（给 critic `--save` 加 `_critic` 后缀，防覆盖）被包在 `if args.megatron_config_path is None:` 里。critic 与 actor **共享 --save**；Megatron `save_checkpoint` 写 role-agnostic 分片到 `{save}/iter_N/`，train_async **先存 actor 后存 critic** → critic 覆盖 actor。而 `parse_megatron_role_args`/`_apply_megatron_role_overrides` 只改 lr/kl、**不碰 --save** → 用 config 时守卫跳过隔离，actor 必被覆盖。**讽刺**：正是为了 critic lr 10× 才用 `MEGATRON_CONFIG_PATH`，这个 config 恰好触发 actor 丢失。
>
> **修复（已提交 + 回归测试）**：`_isolate_critic_checkpoint_dirs` 改为**幂等**（已带 `_critic` 后缀不重复加），调用点**去掉 `if megatron_config_path is None` 守卫 → 无条件隔离**。修复后 actor 存 `{save}/iter_N/`、critic 存 `{save}_critic/iter_N/`，各自独立。`tests/test_critic_checkpoint_isolation.py` 新增 `test_config_driven_critic_still_isolated` + 幂等性测试（8 passed）。详见 memory [[critic-config-clobbers-actor-ckpt]]。
>
> **影响面**：**所有"开 critic + 用 --megatron-config-path"且在此修复前跑的 run，actor ckpt 都已被覆盖丢失**（本 run、以及任何同配置的老 run）。修复前的 critic-lr 实验只能拿到科学结论（EV/reward 曲线），拿不到可部署的 actor。重跑前务必 `git pull` 拿修复。
>
> **收尾**：Daytona 审计 `count=0`（run 自身 cleanup_rollout 已拆完 sandbox，无泄漏）；watchdog 写 SUCCEEDED done-marker（未 purge，SUCCEEDED 由 run 自身优雅收尾）；监控 cron 已删除。

### §13.5 20× 复跑完成复盘：EV 转正快 40%、peak 3.5×、reward 仍平 + actor-save 修复实战验证（2026-07-11）

**背景**：§13.4 的 10× run 科学结论成立但 **actor 产物全丢**（被 critic 覆盖）。本次带上 `placement_group.py` 的 actor-save 修复重跑，同时把两个旋钮再推一步试探：**critic lr 10×→20×（1.0e-5，actor 仍 5e-7）**、**oversampling 288→260**（近 1.0× faithful 采样，含更多难样本、少幸存者偏差）。全新冷启（无 optimizer state）→ config 干净生效。配置：`NUM_ROLLOUT=55`（5 epoch）、`SAVE_INTERVAL=10`、`NUM_CRITIC_ONLY_STEPS=2`、eval 关闭、critic 冷启（无 `_critic` symlink）。

**训练 SUCCEEDED**（2026-07-11 23:41:55，55/55 步，job `raysubmit_qipUbYfm7EGf5JxN`）。

**① EV 转正更快、峰值更高（20× vs 10×）**。`rollout/explained_variance` 逐步（离线解析 .wandb）：
```
step30: +0.049 ← 首个正值(10× 是 step50 → 提前 20 步 = 40% 更快)
step42: +0.084   step45: +0.110   step46: +0.113 ← peak(10× peak 仅 +0.032，本次高 3.5×)
step50-53: +0.102/+0.095/+0.098/+0.099 ← 尾段稳在 +0.10 平台
```
全程 `#EV>0=23`，收尾 step30 起近 25 步几乎连续为正。**结论：critic lr 越大，value head 学得越快、校准质量越高——20× 明显优于 10×。** critic 冷启 + 高 lr 的方向被两个独立 run 反复坐实。

**② reward 仍然平（决定性数据瓶颈证据）**。`rollout/raw_reward` 全程 55 步回归 **slope≈−0.00006, t=−0.29（水平线）**，mean=0.692，5 epoch 零上行零过拟合。**oversampling 缩到 260（更 faithful、更难）没有改变 reward 水平**（~0.69，与 os288/10× 的 ~0.72 同量级）。→ **连续两个 run（10×/20×）× 两个 oversampling（288/260）四种组合，critic 从"学不动"修到"EV 稳定 +0.10"，reward 全部纹丝不动 → reward 瓶颈 100% 是数据，不是 critic/lr/采样。调 critic 已到极限、无剩余收益，下一步必须换/扩数据（更难更多样）。**

**③ ★ actor-save 修复实战验证通过**。逐个 SAVE_INTERVAL 存点在线核验：**iter_9/19/29/39/49/54 全部为真 actor `[248320,5120]`**，critic 独立存 `{save}_critic/iter_N/` `[1,5120]`。**6 个存点全干净**，与 §13.4 的 10× run（actor 全丢）形成对比——修复（幂等 + 无条件隔离）全程稳定生效，本 run 产物完整可部署。最终 ckpt：`.../Qwen3.5-27B_synth3k_2820_ppo_partial_64k_20260710_qwen36_synth3k_5ep_criticlr20x_os260/iter_0000054/`（`latest_checkpointed_iteration.txt`=54）。

**收尾**：job 优雅 SUCCEEDED（launcher log `Job ... succeeded` + worker stop marker；watchdog 末 tick 见 `UNKNOWN`=dashboard 拆除，非失败，**未 purge**——符合"仅失败时清 Daytona"规则）；写 SUCCEEDED done-marker；监控 cron 已删。Daytona 审计因 head 节点当时无外网出口（http=000/timeout）未能核到 count，但 SUCCEEDED 由 run 自身优雅收尾 → 无需 purge。详见 memory [[critic-coldstart-lr10x-ev-positive]]（20× 段）。
